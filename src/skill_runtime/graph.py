"""技能执行 LangGraph。

图结构（支持任意深度的动态技能递归）：

    START
      → route       一级路由（候选来自数据库 skill_registry，即后台线程预加载的一级技能）
      → descend     category：渐进式加载下一层子技能并匹配（可多层）
      → plan        dynamic：LLM 动态规划步骤（失败走关键词兜底规划）
      → start_step  取出 dynamic 的下一步，实例化子技能执行节点（可再嵌套 dynamic）
      → execute     atomic：执行一次 function_call（暂停闸门 interrupt 在此）
      → complete    收尾当前节点，驱动父 dynamic 推进步骤或继续上弹
      → finish      汇总原子技能输出，写回聊天记录，END

记忆：PostgresSaver 以 thread_id=chat_id 持久化 AgentState
（递归技能树 tree / status / level / skill_id / cursor），凭 chat_id 即可暂停/恢复。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field

from ..config import settings
from ..db import ChatRecord
from .functions import FUNCTIONS
from .loader import SkillLoader
from .schema import SkillManifest
from .state import (
    AgentState,
    PlanStep,
    S,
    SkillExecutionNode,
    append_child,
    collect_leaf_outputs,
    get_node,
    tree_to_view,
    update_node,
    utcnow,
)

logger = logging.getLogger(__name__)

# category/atomic/dynamic 三类节点到图节点名的映射
NEXT_BY_TYPE = {"atomic": "execute", "category": "descend", "dynamic": "plan"}


# ---------- LLM 结构化输出模型 ----------
class LlmChoice(BaseModel):
    skill_id: str
    reason: str = ""


class LlmPlanStep(BaseModel):
    skill_id: str
    objective: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    input_from: str = "user"


class LlmPlan(BaseModel):
    steps: list[LlmPlanStep]


# ---------- 运行时（连接池 / checkpointer / 加载器 / LLM） ----------
class Runtime:
    def __init__(self, dsn: str | None = None,
                 loader: SkillLoader | None = None) -> None:
        self.dsn = dsn or settings.pg_dsn
        self.loader = loader or SkillLoader()
        # 业务查询用 dict_row；checkpointer 独立连接池，避免 row_factory 相互干扰
        self.biz_pool = AsyncConnectionPool(
            conninfo=self.dsn, min_size=1, max_size=5,
            kwargs={"row_factory": dict_row}, open=False,
        )
        # checkpointer 的 setup 迁移含 CREATE INDEX CONCURRENTLY，
        # 必须在 autocommit 连接上执行（官方推荐配置）
        self.cp_pool = AsyncConnectionPool(
            conninfo=self.dsn, min_size=1, max_size=10, open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        self.saver: AsyncPostgresSaver | None = None
        self._llm = None
        # 离线/测试开关：跳过 LLM，路由与规划全部走关键词兜底
        self.force_keyword = os.getenv("SKILL_FORCE_KEYWORD", "") == "1"
        # 每完成一个原子 function_call 后触发的钩子 (state, node) -> None；
        # 主程序用它在确定性的步骤边界上写入暂停请求
        self.on_leaf = None

    async def __aenter__(self) -> Runtime:  # noqa: PYI034
        await self.biz_pool.open()
        await self.cp_pool.open()
        serde = JsonPlusSerializer(allowed_msgpack_modules=[
            ("src.skill_runtime.state", "SkillExecutionNode"),
            ("src.skill_runtime.state", "PlanStep"),
        ])
        self.saver = AsyncPostgresSaver(self.cp_pool, serde=serde)
        await self.saver.setup()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.biz_pool.close()
        await self.cp_pool.close()

    @property
    def llm(self):
        if self._llm is None:
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(**settings.get_llm_kwargs())
        return self._llm

    # ---- DB 辅助 ----
    async def level1_candidates(self) -> list[SkillManifest]:
        """读取后台线程预加载的一级技能；表为空时退化为直接扫描。"""
        async with self.biz_pool.connection() as conn:
            cur = await conn.execute(
                "SELECT skill_id FROM skill_registry WHERE level = 1 ORDER BY skill_id"
            )
            rows = await cur.fetchall()
        ids = [r["skill_id"] for r in rows]
        if not ids:
            logger.warning("skill_registry 中暂无一级技能，退化为直接扫描 skills 目录")
            return self.loader.scan_level1()
        return [self.loader.require(sid) for sid in ids]

    async def persist_children(self, manifests: list[SkillManifest]) -> None:
        """渐进式加载到的更深层级技能，顺手缓存进注册表（prefetched=FALSE）。"""
        async with self.biz_pool.connection() as conn:
            for m in manifests:
                await conn.execute(
                    """
                    INSERT INTO skill_registry
                        (skill_id, name, description, skill_type, level, parent_id,
                         fs_path, keywords, has_children, prefetched, manifest, updated_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
                    ON CONFLICT (skill_id) DO UPDATE SET
                        name=EXCLUDED.name, description=EXCLUDED.description,
                        skill_type=EXCLUDED.skill_type, level=EXCLUDED.level,
                        parent_id=EXCLUDED.parent_id, fs_path=EXCLUDED.fs_path,
                        keywords=EXCLUDED.keywords, has_children=EXCLUDED.has_children,
                        manifest=EXCLUDED.manifest, updated_at=now()
                    """,
                    (m.skill_id, m.name, m.description, m.type, m.level, m.parent_id,
                     m.fs_path, json.dumps(m.keywords, ensure_ascii=False),
                     m.has_children, False,
                     json.dumps(m.model_dump(), ensure_ascii=False)),
                )
            await conn.commit()

    async def is_pause_requested(self, chat_id: str) -> bool:
        async with self.biz_pool.connection() as conn:
            cur = await conn.execute(
                "SELECT pause_requested FROM chat_records WHERE chat_id = %s", (chat_id,)
            )
            row = await cur.fetchone()
        return bool(row and row["pause_requested"])

    async def mark_running(self, chat_id: str) -> None:
        async with self.biz_pool.connection() as conn:
            await conn.execute(
                "UPDATE chat_records SET status='running', pause_requested=FALSE, "
                "updated_at=now() WHERE chat_id=%s",
                (chat_id,),
            )
            await conn.commit()

    async def mark_terminal(self, chat_id: str, status: str, response: str) -> None:
        async with self.biz_pool.connection() as conn:
            await conn.execute(
                "UPDATE chat_records SET status=%s, response=%s, updated_at=now() "
                "WHERE chat_id=%s",
                (status, response, chat_id),
            )
            await conn.commit()


def _runtime(config: RunnableConfig) -> Runtime:
    return config["configurable"]["runtime"]  # type: ignore[index]


# ---------- 匹配与规划 ----------
def _keyword_score(manifest: SkillManifest, text: str) -> int:
    low = text.lower()
    return sum(low.count(k.lower()) for k in manifest.keywords if k)


def _fallback_choice(manifests: list[SkillManifest], text: str) -> SkillManifest:
    scored = [(m, _keyword_score(m, text)) for m in manifests]
    best_score = max((s for _, s in scored), default=0)
    if best_score == 0:
        # 全部 0 命中：优先闲聊兜底技能，否则取文件系统顺序第一个
        chat = next((m for m, _ in scored if m.skill_id == "chat"), None)
        return chat or manifests[0]
    # 正分：分数高者胜，同分时按清单 order（chat 的 order=100，自然排最后）
    positives = sorted(
        ((m, s) for m, s in scored if s > 0),
        key=lambda item: (-item[1], item[0].order),
    )
    return positives[0][0]


def match_skill(manifests: list[SkillManifest], text: str,
                rt: Runtime | None = None) -> SkillManifest:
    """关键词打分；完全无命中时让 LLM 裁决，LLM 不可用则兜底。"""
    if not manifests:
        raise ValueError("没有可匹配的子技能")
    if len(manifests) == 1:
        return manifests[0]

    if rt is not None and rt.force_keyword:
        return _fallback_choice(manifests, text)

    if any(_keyword_score(m, text) > 0 for m in manifests):
        return _fallback_choice(manifests, text)

    if rt is not None:
        listing = "\n".join(
            f"- {m.skill_id}：{m.name}。{m.description}" for m in manifests
        )
        prompt = (
            "你是技能路由器。根据用户请求，从候选技能中选择最合适的一个。"
            "只输出 skill_id，不要执行任务。\n候选技能：\n"
            f"{listing}\n用户请求：{text}"
        )
        try:
            choice = rt.llm.with_structured_output(LlmChoice).invoke(
                [SystemMessage(content=prompt), HumanMessage(content=text)]
            )
            selected = next((m for m in manifests if m.skill_id == choice.skill_id), None)
            if selected is not None:
                logger.info("LLM 路由选择 %s（%s）", selected.skill_id, choice.reason)
                return selected
        except Exception:
            logger.exception("LLM 路由失败，使用关键词/默认兜底")
    return _fallback_choice(manifests, text)


SUMMARY_KEYWORDS = {"简报", "总结", "建议", "翻译", "译文", "brief"}


def _is_summary_like(m: SkillManifest) -> bool:
    return bool(set(m.keywords) & SUMMARY_KEYWORDS)


def _fallback_plan(manifests: list[SkillManifest], text: str,
                   max_steps: int) -> list[PlanStep]:
    """关键词兜底规划。

    - 总结/翻译类步骤固定置后（消费上一步产物）；
    - 仅当某个关键词被重复提及（>=2 次，强信号）时才让该步骤越过清单 order，
      否则尊重 order，避免“出差/出行”这类泛词把实时天气步骤挤到后面。
    """
    def max_single_hit(m: SkillManifest) -> int:
        low = text.lower()
        return max((low.count(k.lower()) for k in m.keywords if k), default=0)

    ordered = sorted(
        manifests,
        key=lambda m: (
            _is_summary_like(m),
            0 if max_single_hit(m) >= 2 else 1,
            m.order,
        ),
    )
    steps: list[PlanStep] = []
    for m in ordered[:max_steps]:
        steps.append(PlanStep(
            skill_id=m.skill_id,
            objective=m.description,
            # 总结/翻译类步骤消费上一步产物，其余消费用户原始输入
            input_from="prev_output" if _is_summary_like(m) else "user",
        ))
    return steps


def build_plan(rt: Runtime, dynamic: SkillManifest, children: list[SkillManifest],
               text: str) -> list[PlanStep]:
    """动态规划：LLM 拆步骤，限定在子技能集合内；失败走兜底规划。"""
    max_steps = dynamic.planner.max_steps if dynamic.planner else 5
    hint = dynamic.planner.objective_hint if dynamic.planner else ""

    if rt.force_keyword:
        return _fallback_plan(children, text, max_steps)

    listing = "\n".join(
        f"- {m.skill_id}：{m.name}。{m.description}（关键词：{','.join(m.keywords)}）"
        for m in children
    )
    prompt = (
        "你是动态技能规划器。把用户请求拆成有序步骤，每一步必须且只能引用候选子技能中的一个。"
        f"最多 {max_steps} 步。为每步给出 skill_id、objective、arguments（参数，可为空对象）、"
        "input_from（user 表示输入为用户原始请求；prev_output 表示输入为上一步输出，"
        "翻译/总结类步骤用 prev_output）。\n"
        f"动态技能：{dynamic.name}。{dynamic.description}\n"
        f"规划提示：{hint}\n候选步骤技能：\n{listing}\n用户请求：{text}"
    )
    try:
        plan = rt.llm.with_structured_output(LlmPlan).invoke(
            [SystemMessage(content=prompt), HumanMessage(content=text)]
        )
        valid: list[PlanStep] = []
        seen: set[str] = set()
        by_id = {m.skill_id: m for m in children}
        for s in plan.steps:
            if s.skill_id not in by_id or s.skill_id in seen:
                continue
            seen.add(s.skill_id)
            valid.append(PlanStep(
                skill_id=s.skill_id,
                objective=s.objective or by_id[s.skill_id].description,
                arguments=s.arguments or {},
                input_from=s.input_from if s.input_from in ("user", "prev_output") else "user",
            ))
            if len(valid) >= max_steps:
                break
        if valid:
            logger.info("技能 %s 动态规划出 %d 步：%s",
                        dynamic.skill_id, len(valid), [s.skill_id for s in valid])
            return valid
    except Exception:
        logger.exception("LLM 动态规划失败，使用关键词兜底规划")
    return _fallback_plan(children, text, max_steps)


# ---------- 节点辅助 ----------
def _node_from_manifest(m: SkillManifest) -> SkillExecutionNode:
    return SkillExecutionNode(
        skill_id=m.skill_id,
        name=m.name,
        node_type=m.type,
        level=m.level,
        fs_path=m.fs_path,
        function=m.function,
        status=S.RUNNING,
        started_at=utcnow(),
    )


def _last_output(node: SkillExecutionNode) -> str | None:
    """节点自身的输出；若节点是容器则取其子树中最后一个原子输出。"""
    if node.output:
        return node.output
    for child in reversed(node.steps):
        out = _last_output(child)
        if out:
            return out
    return None


def _resolve_input(tree: SkillExecutionNode, path: list[int], user_input: str) -> str:
    """决定某步骤的实际文本输入：用户输入，或“上一步产物”。

    prev_output 的查找顺序：
    1. 同一动态技能内、本步骤之前的兄弟步骤输出；
    2. 本层没有（本步骤是嵌套动态技能的第一步）时，向上找父动态节点
       在其父技能中的前序步骤输出——即“喂给该嵌套动态技能的输入”。
    """
    node = get_node(tree, path)
    if node.input_from != "prev_output":
        return user_input

    cur = list(path)
    while cur:
        parent = get_node(tree, cur[:-1]) if cur[:-1] else tree
        idx = cur[-1]
        for sib in reversed(parent.steps[:idx]):
            out = _last_output(sib)
            if out:
                return out
        cur = cur[:-1]  # 本层无前置产物，继续向上回溯
    return user_input


async def _pause_gate(state: AgentState, node: SkillExecutionNode,
                      config: RunnableConfig) -> None:
    """步骤边界的暂停闸门：pause_requested=TRUE 时通过 interrupt 挂起 graph。"""
    rt = _runtime(config)
    if await rt.is_pause_requested(state["chat_id"]):
        logger.info("chat_id=%s 在 skill=%s(level=%s) 处挂起",
                    state["chat_id"], node.skill_id, node.level)
        interrupt({
            "reason": "pause_requested",
            "chat_id": state["chat_id"],
            "skill_id": node.skill_id,
            "level": node.level,
        })


def _common(node: SkillExecutionNode, status: str = S.RUNNING) -> dict[str, Any]:
    return {"level": node.level, "skill_id": node.skill_id, "status": status}


# ---------- 图节点 ----------
async def route_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """一级路由：从数据库预加载的一级 skills 中匹配（渐进式加载的第一层）。"""
    rt = _runtime(config)
    candidates = await rt.level1_candidates()
    chosen = match_skill(candidates, state["user_input"], rt)
    logger.info("[route] 一级匹配 -> %s（%s）", chosen.skill_id, chosen.type)

    root = _node_from_manifest(chosen)
    return {"tree": root, "cursor": [], **_common(root)}


async def descend_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """category 节点：渐进式加载直接子技能并向下匹配（可跨多层）。"""
    rt = _runtime(config)
    tree = state["tree"]
    parent = get_node(tree, state["cursor"])
    manifest = rt.loader.require(parent.skill_id)
    children = rt.loader.load_children(manifest)
    try:
        await rt.persist_children(children)
    except Exception:
        logger.exception("缓存子技能到 DB 失败（不影响执行）")

    match_text = f"{state['user_input']}\n{parent.objective}".strip()
    chosen = match_skill(children, match_text, rt)
    logger.info("[descend] %s 下探 -> %s（%s）",
                parent.skill_id, chosen.skill_id, chosen.type)

    child = _node_from_manifest(chosen)
    child.objective = parent.objective
    new_tree, new_cursor = append_child(tree, state["cursor"], child)
    return {"tree": new_tree, "cursor": new_cursor, **_common(child)}


async def plan_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """dynamic 节点：加载候选子技能并动态规划步骤。"""
    rt = _runtime(config)
    tree = state["tree"]
    path = state["cursor"]

    manifest = rt.loader.require(get_node(tree, path).skill_id)
    children = rt.loader.load_children(manifest)
    if not children:
        raise ValueError(f"动态技能 {manifest.skill_id} 没有可用于规划的子技能")
    try:
        await rt.persist_children(children)
    except Exception:
        logger.exception("缓存子技能到 DB 失败（不影响执行）")

    context_text = _resolve_input(tree, path, state["user_input"])
    plan = build_plan(rt, manifest, children, context_text)

    def mutate(n: SkillExecutionNode) -> None:
        n.status = S.RUNNING
        n.plan = plan
        n.step_index = 0
        if not n.started_at:
            n.started_at = utcnow()

    new_tree, node = update_node(tree, path, mutate)
    logger.info("[plan] %s 规划 %d 步：%s",
                node.skill_id, len(plan), [s.skill_id for s in plan])
    return {"tree": new_tree, "status": S.RUNNING,
            "level": node.level, "skill_id": node.skill_id}


async def start_step_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """取 dynamic 的当前步骤，实例化对应子技能执行节点（可能再嵌套 dynamic）。"""
    rt = _runtime(config)
    tree = state["tree"]
    path = state["cursor"]
    dynamic = get_node(tree, path)
    step = dynamic.plan[dynamic.step_index]

    await _pause_gate(state, dynamic, config)

    child_manifest = rt.loader.require(step.skill_id)
    child = _node_from_manifest(child_manifest)
    child.objective = step.objective
    child.arguments = step.arguments
    child.input_from = step.input_from

    def mark(s: PlanStep) -> None:
        s.status = S.RUNNING

    def mutate(n: SkillExecutionNode) -> None:
        mark(n.plan[n.step_index])

    new_tree, _ = update_node(tree, path, mutate)
    new_tree, new_cursor = append_child(new_tree, path, child)
    logger.info("[step] %s 开始第 %d/%d 步 -> %s（%s, level=%s）",
                dynamic.skill_id, dynamic.step_index + 1, len(dynamic.plan),
                child.skill_id, child.node_type, child.level)
    return {"tree": new_tree, "cursor": new_cursor, **_common(child)}


async def execute_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """atomic 节点：执行一次 function_call。"""
    rt = _runtime(config)
    tree = state["tree"]
    path = state["cursor"]
    node = get_node(tree, path)

    await _pause_gate(state, node, config)

    fn = FUNCTIONS.get(node.function or "")
    text_input = _resolve_input(tree, path, state["user_input"])

    if fn is None:
        def fail(n: SkillExecutionNode) -> None:
            n.status = S.FAILED
            n.error = f"未注册的函数：{node.function}"
            n.finished_at = utcnow()

        new_tree, failed = update_node(tree, path, fail)
        return {"tree": new_tree, "status": S.FAILED,
                "level": failed.level, "skill_id": failed.skill_id}

    try:
        result = fn(text_input, **node.arguments)
    except Exception as e:  # 函数自身异常不应使整个进程崩溃
        logger.exception("技能 %s 执行失败", node.skill_id)
        error_message = str(e)

        def fail(n: SkillExecutionNode) -> None:
            n.status = S.FAILED
            n.error = error_message
            n.finished_at = utcnow()

        new_tree, failed = update_node(tree, path, fail)
        return {"tree": new_tree, "status": S.FAILED,
                "level": failed.level, "skill_id": failed.skill_id}

    def done(n: SkillExecutionNode) -> None:
        n.status = S.COMPLETED
        n.output = str(result)
        n.finished_at = utcnow()

    new_tree, completed = update_node(tree, path, done)
    logger.info("[execute] %s 完成：%s", completed.skill_id,
                (completed.output or "")[:120])
    if rt.on_leaf is not None:
        await rt.on_leaf({**state, "tree": new_tree}, completed)
    return {"tree": new_tree, **_common(completed)}


async def complete_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """完成当前节点；若是某 dynamic 的步骤则推进步骤游标，并向上收束。"""
    tree = state["tree"]
    path = state["cursor"]
    current = get_node(tree, path)

    def finish_self(n: SkillExecutionNode) -> None:
        n.status = S.COMPLETED
        n.finished_at = utcnow()

    tree, _ = update_node(tree, path, finish_self)

    # 根节点完成 → 整个执行结束
    if not path:
        return {"tree": tree, "cursor": [],
                "level": current.level, "skill_id": current.skill_id,
                "status": S.COMPLETED}

    parent_path = path[:-1]

    def advance(n: SkillExecutionNode) -> None:
        if n.node_type == "dynamic":
            n.plan[n.step_index].status = S.COMPLETED
            n.step_index += 1

    tree, parent = update_node(tree, parent_path, advance)

    # 父节点就是根 category：其唯一子节点已完成，顺手收束根节点（根 dynamic
    # 若还有剩余步骤则由条件边继续 start_step；步骤耗尽则走 complete 自环收束）
    if not parent_path and parent.node_type == "category":
        tree, parent = update_node(tree, [], lambda n: setattr_finish(n))
        return {"tree": tree, "cursor": [],
                "level": parent.level, "skill_id": parent.skill_id,
                "status": S.COMPLETED}

    return {"tree": tree, "cursor": parent_path,
            "level": parent.level, "skill_id": parent.skill_id,
            "status": S.RUNNING}


def setattr_finish(n: SkillExecutionNode) -> None:
    n.status = S.COMPLETED
    n.finished_at = utcnow()


async def finish_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """汇总所有原子技能输出，写回聊天记录。"""
    rt = _runtime(config)
    tree = state["tree"]
    failed = state.get("status") == S.FAILED
    outputs = collect_leaf_outputs(tree)
    answer = "\n".join(outputs) if outputs else "（技能执行未产生输出）"
    final_status = S.FAILED if failed else S.COMPLETED
    try:
        await rt.mark_terminal(state["chat_id"], final_status, answer)
    except Exception:
        logger.exception("写回聊天终态失败")
    logger.info("chat_id=%s 执行树：\n%s", state["chat_id"], tree_to_view(tree))
    return {
        "status": final_status,
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
    }


# ---------- 条件边 ----------
def _classify(state: AgentState) -> str:
    node = get_node(state["tree"], state["cursor"])
    return NEXT_BY_TYPE[node.node_type]


def _after_complete(state: AgentState) -> str:
    if state.get("status") == S.FAILED:
        return "finish"
    cursor = state["cursor"]
    if not cursor:
        return "finish"
    parent = get_node(state["tree"], cursor)
    if parent.node_type == "category":
        return "complete"  # category 的唯一子节点已完成，继续收束自身
    if parent.step_index < len(parent.plan):
        return "start_step"
    return "complete"  # dynamic 的所有步骤已完成，收束自身


def build_graph(rt: Runtime):
    g = StateGraph(AgentState)
    g.add_node("route", route_node)
    g.add_node("descend", descend_node)
    g.add_node("plan", plan_node)
    g.add_node("start_step", start_step_node)
    g.add_node("execute", execute_node)
    g.add_node("complete", complete_node)
    g.add_node("finish", finish_node)

    g.add_edge(START, "route")
    g.add_conditional_edges(
        "route", _classify,
        {"execute": "execute", "descend": "descend", "plan": "plan"},
    )
    g.add_conditional_edges(
        "descend", _classify,
        {"execute": "execute", "descend": "descend", "plan": "plan"},
    )
    g.add_edge("plan", "start_step")
    g.add_conditional_edges(
        "start_step", _classify,
        {"execute": "execute", "descend": "descend", "plan": "plan"},
    )
    g.add_edge("execute", "complete")
    g.add_conditional_edges(
        "complete", _after_complete,
        {"start_step": "start_step", "complete": "complete", "finish": "finish"},
    )
    g.add_edge("finish", END)

    return g.compile(checkpointer=rt.saver)


def _config(rt: Runtime, chat_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": chat_id, "runtime": rt}}


# ---------- 对外：启动 / 恢复 / 查看记忆 ----------
async def start_chat(rt: Runtime, app, chat: ChatRecord) -> dict[str, Any]:
    """为一条聊天记录启动 graph（thread_id=chat_id）。"""
    await rt.mark_running(chat.chat_id)
    initial: AgentState = {
        "messages": [HumanMessage(content=chat.content)],
        "chat_id": chat.chat_id,
        "user_input": chat.content,
        "status": S.RUNNING,
        "level": 0,
        "skill_id": "",
        "cursor": [],
    }
    try:
        return await app.ainvoke(initial, _config(rt, chat.chat_id))
    except GraphInterrupt:
        logger.info("chat_id=%s 已在步骤边界暂停", chat.chat_id)
        snap = await app.aget_state(_config(rt, chat.chat_id))
        return snap.values


async def resume_chat(rt: Runtime, app, chat_id: str) -> dict[str, Any]:
    """凭 chat_id 从 checkpoint 恢复被暂停的 graph。"""
    await rt.mark_running(chat_id)
    try:
        return await app.ainvoke(Command(resume={"resume": True}), _config(rt, chat_id))
    except GraphInterrupt:
        logger.info("chat_id=%s 仍处于暂停（又收到新的暂停请求）", chat_id)
        snap = await app.aget_state(_config(rt, chat_id))
        return snap.values


async def get_memory(rt: Runtime, app, chat_id: str) -> dict[str, Any] | None:
    """读取某聊天 id 在 LangGraph（PostgreSQL checkpoint）中持久化的记忆快照。"""
    snap = await app.aget_state(_config(rt, chat_id))
    return snap.values if snap else None
