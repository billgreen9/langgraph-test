"""意图路由 LangGraph（RouterGraph）。

职责单一：把一条 user 消息（chat_record）转换为一个可执行的 chat_task。
本图不执行任何业务技能，只做「消息 → 任务」的路由与落库。

流程（5 节点 1 分支）：

    START → load_message → route_intents → (create | attach) → persist → END

- load_message : 读取消息内容与会话信息载入 state（消息不存在则失败）
- route_intents: 用 skills/intents/*.md 意图技能识别意图（关键词 → LLM → 兜底），
  并判定分支：消息已关联非终态任务 → attach（幂等重跑护栏；P2 扩展为
  collecting 任务的跨消息归并）；否则 → create（P1 主路径：新建任务）
- create / attach : 仅产出任务标识，不写库（create 生成新 id；attach 复用已有 id）
- persist      : 唯一写库点——insert chat_task + 建立消息-任务多对多关联

checkpoint thread_id = chat_id（消息 id）。
P1：一条消息固定产出一个任务；多意图拆分与跨消息归并在 P2 引入。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from ..db import ChatTask, db
from .graph import Runtime, _runtime  # 复用同一运行时（LLM/连接池/加载器）
from .intent_loader import IntentLoader, IntentSpec
from .tooling import parse_tool_calls, tool_name_for

logger = logging.getLogger(__name__)

# 非终态任务状态：可被 attach 复用（终态 completed/failed 不可复用）
OPEN_TASK_STATUSES = {"collecting", "pending", "running", "paused"}


class RouterState(TypedDict, total=False):
    chat_id: str
    # load_message 载入
    session_id: str
    user_id: str | None
    message_content: str
    # route_intents 产出
    intent_id: str
    entry_skill_id: str
    title: str
    mode: str  # create / attach
    existing_task_ids: list[str]
    # create / attach 产出
    task_id: str
    task_ids: list[str]


def _intent_tool(spec: IntentSpec) -> dict[str, Any]:
    """意图 → OpenAI 工具定义（工具名即意图，参数只带 reason）。"""
    return {
        "type": "function",
        "function": {
            "name": tool_name_for(spec.intent_id),
            "description": f"[{spec.intent_id}] {spec.name}。{spec.body}",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": [],
            },
        },
    }


# ---------- 意图匹配 ----------
def _intent_score(spec: IntentSpec, text: str) -> int:
    low = text.lower()
    return sum(low.count(k.lower()) for k in spec.keywords if k)


def _fallback_intent(intents: list[IntentSpec], text: str) -> IntentSpec:
    scored = [(s, _intent_score(s, text)) for s in intents]
    best = max((score for _, score in scored), default=0)
    if best == 0:
        chat = next((s for s, _ in scored if s.intent_id == "chat"), None)
        return chat or intents[0]
    positives = sorted(
        ((s, score) for s, score in scored if score > 0),
        key=lambda item: (-item[1], item[0].order),
    )
    return positives[0][0]


def match_intent(
    intents: list[IntentSpec], text: str, rt: Runtime | None = None
) -> IntentSpec:
    """关键词命中优先；无命中时让 LLM 以 tool_calls 协议裁决；失败则兜底。"""
    if not intents:
        raise ValueError("skills/intents 下没有任何意图技能")
    if len(intents) == 1:
        return intents[0]
    if any(_intent_score(s, text) > 0 for s in intents):
        return _fallback_intent(intents, text)
    if rt is not None and not rt.force_keyword:
        listing = "\n".join(
            f"- {s.intent_id}：{s.name}。{s.body}" for s in intents
        )
        prompt = (
            "你是意图路由器。根据用户消息，通过调用候选意图对应的工具选择唯一最合适的"
            "一个（工具名即意图），不要执行任务。\n候选意图：\n"
            f"{listing}\n用户消息：{text}"
        )
        try:
            resp = rt.llm.bind_tools(
                [_intent_tool(s) for s in intents]
            ).invoke([SystemMessage(content=prompt), HumanMessage(content=text)])
            rev = {tool_name_for(s.intent_id): s for s in intents}
            for name, _args in parse_tool_calls(resp):
                selected = rev.get(name)
                if selected is not None:
                    logger.info("LLM tool_calls 意图选择 %s", selected.intent_id)
                    return selected
        except Exception:
            logger.exception("LLM 意图路由失败，使用关键词/默认兜底")
    return _fallback_intent(intents, text)


# ---------- 图节点 ----------
async def load_message_node(
    state: RouterState, config: RunnableConfig
) -> dict[str, Any]:
    """读取 user 消息，把内容与会话信息载入 state。"""
    chat_id = state["chat_id"]
    message = await asyncio.to_thread(db.messages.get, chat_id)
    if message is None:
        raise ValueError(f"聊天消息不存在：{chat_id}")
    return {
        "session_id": message.session_id,
        "user_id": message.user_id,
        "message_content": message.content,
    }


async def route_intents_node(
    state: RouterState, config: RunnableConfig
) -> dict[str, Any]:
    """意图识别 + create/attach 分支判定（只决策，不写库）。"""
    rt = _runtime(config)
    chat_id = state["chat_id"]
    text = state["message_content"]

    intents = IntentLoader().load_all()
    # 入口技能闭环校验：每个 intent.entry_skill 必须是存在的一级业务技能
    level1 = await rt.level1_candidates()
    valid_ids = {m.skill_id for m in level1}
    intents = [s for s in intents if s.entry_skill in valid_ids]
    if not intents:
        raise ValueError(
            f"没有 entry_skill 命中一级业务技能 {sorted(valid_ids)} 的有效意图"
        )

    spec = match_intent(intents, text, rt)

    # 分支判定：消息已关联非终态任务 → attach（幂等重跑护栏；P2 扩展为跨消息归并）
    linked = await asyncio.to_thread(
        db.task_messages.list_tasks_for_message, chat_id
    )
    open_ids = [t.task_id for t in linked if t.status in OPEN_TASK_STATUSES]
    mode = "attach" if open_ids else "create"
    logger.info("[router] chat_id=%s 意图=%s 分支=%s",
                chat_id, spec.intent_id, mode)
    return {
        "intent_id": spec.intent_id,
        "entry_skill_id": spec.entry_skill,
        "title": spec.name,
        "mode": mode,
        "existing_task_ids": open_ids,
    }


async def create_node(
    state: RouterState, config: RunnableConfig
) -> dict[str, Any]:
    """create 分支：生成新任务标识（不写库，写库统一在 persist）。"""
    task_id = f"task-{uuid.uuid4().hex[:8]}"
    logger.info("[router] chat_id=%s 新建任务 %s -> %s",
                state["chat_id"], task_id, state["entry_skill_id"])
    return {"task_id": task_id, "task_ids": [task_id]}


async def attach_node(
    state: RouterState, config: RunnableConfig
) -> dict[str, Any]:
    """attach 分支：复用消息已关联的非终态任务（不写库，写库统一在 persist）。"""
    logger.info("[router] chat_id=%s 挂接已有任务 %s",
                state["chat_id"], state["existing_task_ids"])
    return {"task_ids": list(state["existing_task_ids"])}


async def persist_node(
    state: RouterState, config: RunnableConfig
) -> dict[str, Any]:
    """唯一写库点：create 落新任务，attach 补齐关联（link 幂等）。"""
    chat_id = state["chat_id"]
    if state["mode"] == "create":
        task = ChatTask(
            task_id=state["task_id"],
            session_id=state["session_id"],
            title=state["title"],
            content=state["message_content"],
            entry_skill_id=state["entry_skill_id"],
            status="pending",
        )
        await asyncio.to_thread(db.tasks.insert, task)
        await asyncio.to_thread(db.task_messages.link, task.task_id, chat_id)
        logger.info("[router] chat_id=%s persist 任务=%s entry=%s",
                    chat_id, task.task_id, task.entry_skill_id)
    else:
        for tid in state["task_ids"]:
            await asyncio.to_thread(db.task_messages.link, tid, chat_id)
        logger.info("[router] chat_id=%s persist 挂接任务=%s",
                    chat_id, state["task_ids"])
    return {}


def _after_route_intents(state: RouterState) -> str:
    """route_intents 出口：返回分支节点名（create | attach）。"""
    return state["mode"]


def build_router_graph(rt: Runtime):
    g = StateGraph(RouterState)
    g.add_node("load_message", load_message_node)
    g.add_node("route_intents", route_intents_node)
    g.add_node("create", create_node)
    g.add_node("attach", attach_node)
    g.add_node("persist", persist_node)

    g.add_edge(START, "load_message")
    g.add_edge("load_message", "route_intents")
    g.add_conditional_edges(
        "route_intents", _after_route_intents,
        {"create": "create", "attach": "attach"},
    )
    g.add_edge("create", "persist")
    g.add_edge("attach", "persist")
    g.add_edge("persist", END)
    return g.compile(checkpointer=rt.saver)
