"""任务执行 LangGraph（TaskGraph）。

一级意图路由由独立的 RouterGraph（router_graph.py）完成，本图只负责执行单个
chat_task，checkpoint 的 thread_id = task_id。

图结构（支持任意深度的动态技能递归）：

    START
      → enter_task  按 task_id 读 chat_task，以 entry_skill_id 建执行树根节点
      → descend     category：渐进式加载下一层子技能并匹配（可多层）
      → plan        dynamic：LLM 动态规划步骤（失败走关键词兜底规划）
      → start_step  取出 dynamic 的下一步，实例化子技能执行节点（可再嵌套 dynamic/react）
      → execute     atomic：执行一次 function_call（暂停闸门 interrupt 在此）
      → complete    收尾当前节点，驱动父 dynamic 推进步骤或继续上弹
      → finish      汇总原子技能输出，写回 chat_task，END

    ReAct 回路（react 类型技能；暂停闸门在每轮 think 入口）：
      → react_think        LLM 基于往轮观测决定：done（→complete）或给出下一批动作
      → parallel_execute   同批动作经 Send 扇出并发执行 atomic function_call
      → react_join         屏障扇入，按轮次把分支结果合入执行树
      → react_think …      观测结果喂入下一轮，直到 done/无有效动作/达到最大轮次

记忆：PostgresSaver 以 thread_id=task_id 持久化 AgentState
（递归技能树 tree / status / level / skill_id / cursor），凭 task_id 即可暂停/恢复。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from ..config import settings
from .functions import resolve_function
from .loader import SkillLoader
from .match_config import UNKNOWN_ANSWER
from .matching import (
    KIND_AUTO,
    KIND_LLM,
    llm_confirm_skill,
    match_scope,
)
from .rerank import get_rerank_llm
from .schema import SkillManifest
from .tooling import (
    REACT_FINISH_TOOL,
    manifest_tool,
    parse_tool_calls,
    tool_name_for,
)
from .state import (
    AgentState,
    PlanStep,
    S,
    SkillExecutionNode,
    append_child,
    collect_leaf_outputs,
    get_node,
    iter_nodes,
    tree_to_view,
    update_node,
    utcnow,
)

logger = logging.getLogger(__name__)

# category/atomic/dynamic/react 四类节点到图节点名的映射
NEXT_BY_TYPE = {
    "atomic": "execute",
    "category": "descend",
    "dynamic": "plan",
    "react": "react_think",
}


# ---------- ReAct 决策内部模型（经 tool_calls 协议解析后填充） ----------
@dataclass
class ReactAction:
    """ReAct 单轮规划出的一个原子动作。"""

    skill_id: str
    objective: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReactDecision:
    """ReAct 单轮决策：要么给出最终回答，要么给出下一批可并发动作。"""

    done: bool = False
    final_answer: str = ""
    actions: list[ReactAction] = field(default_factory=list)


# 同一原子技能在一个 ReAct 节点内最多允许失败重试的次数
REACT_MAX_RETRY = 2


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

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        """读取任务行（TaskGraph 的入口数据）。"""
        async with self.biz_pool.connection() as conn:
            cur = await conn.execute(
                "SELECT * FROM chat_task WHERE task_id = %s", (task_id,)
            )
            return await cur.fetchone()

    async def is_pause_requested(self, task_id: str) -> bool:
        async with self.biz_pool.connection() as conn:
            cur = await conn.execute(
                "SELECT pause_requested FROM chat_task WHERE task_id = %s", (task_id,)
            )
            row = await cur.fetchone()
        return bool(row and row["pause_requested"])

    async def mark_running(self, task_id: str) -> None:
        async with self.biz_pool.connection() as conn:
            await conn.execute(
                "UPDATE chat_task SET status='running', pause_requested=FALSE, "
                "updated_at=now() WHERE task_id=%s",
                (task_id,),
            )
            await conn.commit()

    async def mark_terminal(self, task_id: str, status: str,
                            output: str, error: str | None = None) -> None:
        async with self.biz_pool.connection() as conn:
            await conn.execute(
                "UPDATE chat_task SET status=%s, output=%s, error=%s, "
                "updated_at=now() WHERE task_id=%s",
                (status, output, error, task_id),
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
                rt: Runtime | None = None) -> SkillManifest | None:
    """本层 intent_math：高分直接命中，灰区 LLM 可拒绝，未过线返回 None。"""
    if not manifests:
        raise ValueError("没有可匹配的子技能")
    level = manifests[0].level
    ids = {m.skill_id for m in manifests}
    by_id = {m.skill_id: m for m in manifests}
    force_lexical = rt is None or rt.force_keyword
    llm = None if force_lexical else get_rerank_llm()
    try:
        band = match_scope(
            text,
            skill_level=level,
            skill_ids=ids,
            forbid=False,
            use_vector=not force_lexical,
            llm=llm,
            force_lexical=force_lexical,
        )
    except Exception:
        logger.exception("intent_math 下探失败")
        return None
    if band.kind == KIND_AUTO and band.skill_id in by_id:
        logger.info("下探直接命中 %s score=%.3f", band.skill_id, band.score)
        return by_id[band.skill_id]
    if band.kind == KIND_LLM:
        cands = [by_id[s] for s in band.skill_ids if s in by_id]
        if force_lexical or rt is None:
            return None
        return llm_confirm_skill(get_rerank_llm() or rt.llm, text, cands)
    return None


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
        "你是动态技能规划器。把用户请求拆成有序步骤：每一步通过调用候选子技能对应的"
        "工具表达（工具名即技能，多次调用按顺序作为步骤顺序）。"
        f"最多 {max_steps} 步，每步只能引用候选子技能中的一个。"
        "调用参数除技能自身参数外，可附 objective（步骤目标）与 input_from"
        "（user 表示输入为用户原始请求；prev_output 表示输入为上一步输出，"
        "翻译/总结类步骤用 prev_output）。\n"
        f"动态技能：{dynamic.name}。{dynamic.description}\n"
        f"规划提示：{hint}\n候选步骤技能：\n{listing}\n用户请求：{text}"
    )
    try:
        resp = rt.llm.bind_tools(
            [manifest_tool(m, "plan") for m in children]
        ).invoke([SystemMessage(content=prompt), HumanMessage(content=text)])
        valid: list[PlanStep] = []
        seen: set[str] = set()
        by_id = {m.skill_id: m for m in children}
        rev = {tool_name_for(m.skill_id): m.skill_id for m in children}
        for name, args in parse_tool_calls(resp):
            skill_id = rev.get(name)
            if skill_id is None or skill_id in seen:
                continue
            args = args or {}
            arguments = {k: v for k, v in args.items()
                         if k not in ("objective", "input_from")}
            seen.add(skill_id)
            valid.append(PlanStep(
                skill_id=skill_id,
                objective=args.get("objective") or by_id[skill_id].description,
                arguments=arguments,
                input_from=args.get("input_from")
                if args.get("input_from") in ("user", "prev_output") else "user",
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
    node = SkillExecutionNode(
        skill_id=m.skill_id,
        name=m.name,
        node_type=m.type,
        level=m.level,
        fs_path=m.fs_path,
        function=m.function,
        module=m.module,
        status=S.RUNNING,
        started_at=utcnow(),
    )
    if m.type == "react" and m.react is not None:
        node.react_max_rounds = m.react.max_rounds
    return node


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
    if await rt.is_pause_requested(state["task_id"]):
        logger.info("task_id=%s 在 skill=%s(level=%s) 处挂起",
                    state["task_id"], node.skill_id, node.level)
        interrupt({
            "reason": "pause_requested",
            "task_id": state["task_id"],
            "skill_id": node.skill_id,
            "level": node.level,
        })


def _common(node: SkillExecutionNode, status: str = S.RUNNING) -> dict[str, Any]:
    return {"level": node.level, "skill_id": node.skill_id, "status": status}


# ---------- 图节点 ----------
async def enter_task_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """任务入口：按 task_id 读 chat_task，以 entry_skill_id 构建执行树根节点。

    一级意图路由已由独立的 RouterGraph 完成，这里不再做任何匹配。
    """
    rt = _runtime(config)
    task_id = state["task_id"]
    row = await rt.get_task(task_id)
    if row is None:
        raise ValueError(f"任务不存在：{task_id}")
    await rt.mark_running(task_id)

    manifest = rt.loader.require(row["entry_skill_id"])
    root = _node_from_manifest(manifest)
    root.objective = row.get("title") or ""
    logger.info("[enter_task] task_id=%s 入口技能 -> %s（%s）",
                task_id, manifest.skill_id, manifest.type)
    return {
        "tree": root, "cursor": [], **_common(root),
        "user_input": state.get("user_input") or row.get("content") or "",
        "session_id": row.get("session_id") or state.get("session_id", ""),
    }


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

    match_text = state["user_input"]
    chosen = match_skill(children, match_text, rt)
    if chosen is None:
        logger.info("[descend] %s 下探未命中", parent.skill_id)

        def mark_unknown(n: SkillExecutionNode) -> None:
            n.output = UNKNOWN_ANSWER
            n.status = S.COMPLETED
            n.finished_at = utcnow()

        new_tree, node = update_node(tree, state["cursor"], mark_unknown)
        return {"tree": new_tree, "cursor": state["cursor"], **_common(node, S.COMPLETED)}

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

    fn_error: str | None = None
    try:
        fn = resolve_function(node.function, node.module, node.fs_path,
                              node.skill_id)
    except Exception as e:
        fn, fn_error = None, f"本地函数加载失败：{e}"
    text_input = _resolve_input(tree, path, state["user_input"])

    if fn is None:
        def fail(n: SkillExecutionNode) -> None:
            n.status = S.FAILED
            n.error = fn_error or f"未注册的函数：{node.function}"
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


# ---------- ReAct 节点：增量规划 + 并发扇出/扇入 ----------
def _react_observations(node: SkillExecutionNode) -> str:
    """把 react 节点已合入的各原子子节点结果整理为给 LLM 的往轮观测文本。"""
    if not node.steps:
        return "（暂无已执行动作的结果）"
    lines: list[str] = []
    for i, child in enumerate(node.steps, start=1):
        if child.status == S.COMPLETED:
            lines.append(f"[{i}] 动作 {child.skill_id} 成功，返回：{child.output or ''}")
        else:
            lines.append(f"[{i}] 动作 {child.skill_id} 失败，错误：{child.error or ''}")
    return "\n".join(lines)


def _action_fingerprint(function: str, arguments: dict[str, Any]) -> str:
    return f"{function}|{json.dumps(arguments, ensure_ascii=False, sort_keys=True)}"


def _normalize_actions(
    raw: list[ReactAction],
    by_id: dict[str, SkillManifest],
    node: SkillExecutionNode,
) -> list[dict[str, Any]]:
    """校验 LLM 返回的动作批次：候选存在性、失败重试上限、同批复刻去重。"""
    actions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for a in raw:
        manifest = by_id.get(a.skill_id)
        if manifest is None or not manifest.function:
            continue
        if node.react_retries.get(manifest.skill_id, 0) >= REACT_MAX_RETRY:
            continue
        arguments = dict(a.arguments or {})
        fingerprint = _action_fingerprint(manifest.function, arguments)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        actions.append({
            "skill_id": manifest.skill_id,
            "function": manifest.function,
            "module": manifest.module,
            "fs_path": manifest.fs_path,
            "objective": a.objective or manifest.description,
            "arguments": arguments,
            "parallelizable": manifest.parallelizable and not manifest.self_exclusive,
        })
    return actions


def _react_keyword_actions(
    children: list[SkillManifest], node: SkillExecutionNode, text: str
) -> list[dict[str, Any]]:
    """离线/LLM 不可用兜底：每轮只发一个关键词最高分、未执行且未超重试上限的动作。"""
    executed = {c.skill_id for c in node.steps}
    candidates = [
        c for c in children
        if c.skill_id not in executed
        and node.react_retries.get(c.skill_id, 0) < REACT_MAX_RETRY
    ]
    if not candidates:
        return []
    chosen = _fallback_choice(candidates, text)
    return [{
        "skill_id": chosen.skill_id,
        "function": chosen.function or "",
        "module": chosen.module,
        "fs_path": chosen.fs_path,
        "objective": chosen.description,
        "arguments": {},
        "parallelizable": chosen.parallelizable and not chosen.self_exclusive,
    }]


async def _react_llm_decide(
    rt: Runtime,
    react_manifest: SkillManifest,
    children: list[SkillManifest],
    user_input: str,
    observations: str,
    round_no: int,
    extra_hint: str = "",
) -> ReactDecision | None:
    """调用 LLM 做单轮 ReAct 决策（tool_calls 协议）；失败/无效返回 None。

    模型要么调用 ``finish`` 工具收束（done=true），要么并发调用候选原子技能
    对应的工具给出下一批动作（一次响应中的多个 tool_calls 即同批并发动作，
    顺序即模型给定的顺序）。
    """
    listing = "\n".join(
        f"- {c.skill_id}：{c.name}。{c.description}（关键词：{','.join(c.keywords)}）"
        for c in children
    )
    hint = react_manifest.react.objective_hint if react_manifest.react else ""
    prompt = (
        "你是 ReAct 执行器。每轮根据用户请求与往轮观测结果二选一：\n"
        "1) 信息已足够回答用户：调用 finish 工具，在 final_answer 给出中文综合结论；\n"
        "2) 否则通过调用候选原子技能对应的工具，给出【下一批要并发执行】的动作。\n"
        "同批动作必须互不依赖：它们只能引用用户原始输入；若某动作需要往轮结果，"
        "必须把所需内容显式写进该动作的 arguments（例如 source_text），"
        "绝不能假设同批动作之间存在先后顺序。\n"
        "每个动作只能引用候选原子技能，不要重复已经成功的动作；没有有用的动作可做时调用 finish。\n"
        f"ReAct 技能：{react_manifest.name}。{react_manifest.description}\n"
        f"规划提示：{hint}\n候选原子技能：\n{listing}\n"
        f"用户请求：{user_input}\n往轮观测：\n{observations}\n{extra_hint}"
    )
    try:
        resp = await rt.llm.bind_tools(
            [manifest_tool(c, "react") for c in children] + [REACT_FINISH_TOOL]
        ).ainvoke([SystemMessage(content=prompt), HumanMessage(content=user_input)])
        calls = parse_tool_calls(resp)
        rev = {tool_name_for(c.skill_id): c.skill_id for c in children}

        # finish 优先：无论出现在批次哪个位置都视为收束
        for name, args in calls:
            if name == REACT_FINISH_TOOL["function"]["name"]:
                logger.info("[react] %s 第 %d 轮 LLM 调用 finish",
                            react_manifest.skill_id, round_no)
                return ReactDecision(done=True,
                                     final_answer=str(args.get("final_answer") or ""))
        actions = [ReactAction(skill_id=rev[name], arguments=args)
                   for name, args in calls if name in rev]
        logger.info("[react] %s 第 %d 轮 LLM 决策：actions=%s",
                    react_manifest.skill_id, round_no,
                    [a.skill_id for a in actions])
        return ReactDecision(done=False, actions=actions)
    except Exception:
        logger.exception("[react] %s 第 %d 轮 LLM 决策失败",
                         react_manifest.skill_id, round_no)
        return None


async def _react_synthesize(
    rt: Runtime, react_manifest: SkillManifest, user_input: str,
    node: SkillExecutionNode,
) -> str:
    """让 LLM 综合往轮观测给最终回答；失败时退化为拼接原子输出。"""
    outputs = [c.output for c in node.steps
               if c.status == S.COMPLETED and c.output]
    fallback = "\n".join(outputs) if outputs else "（技能执行未产生输出）"
    if rt.force_keyword:
        return fallback
    prompt = (
        "根据以下动作观测结果，用中文为用户请求给出最终综合回答，直接给结论，不要罗列过程。\n"
        f"用户请求：{user_input}\n观测结果：\n{_react_observations(node)}"
    )
    try:
        resp = await rt.llm.ainvoke(
            [SystemMessage(content=prompt), HumanMessage(content=user_input)]
        )
        text = resp.content if isinstance(resp, AIMessage) else str(resp)
        if text and text.strip():
            return text.strip()
    except Exception:
        logger.exception("[react] %s 综合回答失败，使用拼接输出兜底",
                         react_manifest.skill_id)
    return fallback


async def react_think_node(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    """ReAct 思考节点：基于往轮观测，让 LLM 决定收尾或给出下一批并发动作。"""
    rt = _runtime(config)
    tree = state["tree"]
    path = state["cursor"]

    # 暂停闸门必须在任何 tree 变更之前：interrupt 会丢弃本次未完成的节点写入，
    # 恢复时整节重放，轮次计数不会重复累加
    await _pause_gate(state, get_node(tree, path), config)

    node = get_node(tree, path)
    manifest = rt.loader.require(node.skill_id)
    children = [
        c for c in rt.loader.load_children(manifest)
        if c.type == "atomic" and c.function
    ]
    if not children:
        raise ValueError(f"ReAct 技能 {manifest.skill_id} 没有可调用的 atomic 子技能")
    try:
        await rt.persist_children(children)
    except Exception:
        logger.exception("缓存子技能到 DB 失败（不影响执行）")
    by_id = {c.skill_id: c for c in children}

    def enter_round(n: SkillExecutionNode) -> None:
        n.react_round += 1
        n.status = S.RUNNING
        if not n.started_at:
            n.started_at = utcnow()

    tree, node = update_node(tree, path, enter_round)
    round_no = node.react_round

    async def finish_react(answer: str, reason: str) -> dict[str, Any]:
        def done(n: SkillExecutionNode) -> None:
            n.status = S.COMPLETED
            n.output = answer
            n.react_pending = []
            n.finished_at = utcnow()

        new_tree, completed = update_node(tree, path, done)
        logger.info("[react] %s 收尾（%s），共 %d 轮",
                    completed.skill_id, reason, round_no)
        return {"tree": new_tree, "cursor": list(path), "final_answer": answer,
                "status": S.COMPLETED, "level": completed.level,
                "skill_id": completed.skill_id}

    # 护栏：达到最大轮次强制收尾
    if round_no > node.react_max_rounds:
        answer = await _react_synthesize(rt, manifest, state["user_input"], node)
        return await finish_react(answer, f"达到最大轮次 {node.react_max_rounds}")

    observations = _react_observations(node)
    actions: list[dict[str, Any]] = []

    if rt.force_keyword:
        actions = _react_keyword_actions(children, node, state["user_input"])
    else:
        decision = await _react_llm_decide(
            rt, manifest, children, state["user_input"], observations, round_no
        )
        if decision is None:
            # LLM 整体失败：退化为每轮一个关键词动作（顺序 ReAct）
            actions = _react_keyword_actions(children, node, state["user_input"])
        elif decision.done:
            answer = decision.final_answer.strip() or await _react_synthesize(
                rt, manifest, state["user_input"], node
            )
            return await finish_react(answer, "LLM 判定完成")
        else:
            actions = _normalize_actions(decision.actions, by_id, node)
            # 护栏：一批全是非法/超限动作，带纠正提示重询一次
            if decision.actions and not actions:
                retry = await _react_llm_decide(
                    rt, manifest, children, state["user_input"], observations,
                    round_no,
                    extra_hint=(
                        "\n注意：你上一批动作全部无效（技能不存在、超过失败重试上限或完全重复），"
                        "请重新给出有效动作，或直接 done。"
                    ),
                )
                if retry is not None and not retry.done:
                    actions = _normalize_actions(retry.actions, by_id, node)
                elif retry is not None and retry.done:
                    answer = retry.final_answer.strip() or await _react_synthesize(
                        rt, manifest, state["user_input"], node
                    )
                    return await finish_react(answer, "重询后 LLM 判定完成")

    # 护栏：没有可执行动作（兜底模式耗尽候选 / 模型空转）→ 强制收尾
    if not actions:
        answer = await _react_synthesize(rt, manifest, state["user_input"], node)
        return await finish_react(answer, "无有效动作")

    # 护栏：批次含不可并发（parallelizable=False/self_exclusive）动作时机械拆波，
    # 本轮只放第一个，其余动作下一轮重新规划
    if len(actions) > 1 and any(not a["parallelizable"] for a in actions):
        blocked = [a["skill_id"] for a in actions if not a["parallelizable"]]
        logger.info("[react] %s 批次含不可并发动作 %s，机械拆波：本轮只执行 %s",
                    node.skill_id, blocked, actions[0]["skill_id"])
        actions = actions[:1]

    def store(n: SkillExecutionNode) -> None:
        n.status = S.RUNNING
        n.react_pending = actions

    tree, node = update_node(tree, path, store)
    logger.info("[react] %s 第 %d 轮派发 %d 个并发动作：%s",
                node.skill_id, round_no, len(actions),
                [a["skill_id"] for a in actions])
    return {"tree": tree, "cursor": list(path), "status": S.RUNNING,
            "level": node.level, "skill_id": node.skill_id}


async def parallel_execute_node(
    payload: dict[str, Any], config: RunnableConfig
) -> dict[str, Any]:
    """ReAct 扇出分支：执行单个原子 function_call。

    只写 branch_results 扇入通道，绝不写 tree/cursor/status
    （多个分支并发写同一标量通道会触发 InvalidUpdateError）。
    同步函数统一丢到线程池，避免阻塞事件循环导致“假并行”。
    """
    rt = _runtime(config)

    def make_result(status: str, output: str | None, error: str | None) -> dict[str, Any]:
        return {
            "round": payload["round"],
            "seq": payload["seq"],
            "skill_id": payload["skill_id"],
            "function": payload["function"],
            "status": status,
            "output": output,
            "error": error,
        }

    fn_error: str | None = None
    try:
        fn = resolve_function(payload["function"], payload.get("module"),
                              payload.get("fs_path"), payload["skill_id"])
    except Exception as e:
        fn, fn_error = None, f"本地函数加载失败：{e}"
    if fn is None:
        result = make_result(S.FAILED, None,
                             fn_error or f"未注册的函数：{payload['function']}")
    else:
        try:
            value = await asyncio.to_thread(
                fn, payload["text_input"], **payload.get("arguments", {})
            )
            result = make_result(S.COMPLETED, str(value), None)
        except Exception as e:  # 分支异常隔离：失败转成观测结果喂回下一轮
            logger.exception("[parallel] round=%s seq=%s 动作 %s 执行失败",
                             payload["round"], payload["seq"], payload["skill_id"])
            result = make_result(S.FAILED, None, str(e))
    logger.info("[parallel] round=%s seq=%s %s -> %s%s",
                payload["round"], payload["seq"], payload["skill_id"],
                result["status"],
                f"：{(result['output'] or result['error'] or '')[:80]}")
    return {"branch_results": [result]}


async def react_join_node(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    """ReAct 扇入屏障：所有并行分支结束后只执行一次，按 seq 合入执行树。

    合入后 react 节点的 steps 仍是唯一事实源，observation/记忆展示/finish
    全部复用既有逻辑；branch_results 中往轮的残留结果按 round 忽略。
    """
    rt = _runtime(config)
    tree = state["tree"]
    path = state["cursor"]
    react_node = get_node(tree, path)
    round_no = react_node.react_round
    by_seq = {
        r["seq"]: r
        for r in state.get("branch_results", [])
        if r.get("round") == round_no
    }

    appended: list[SkillExecutionNode] = []

    def merge(n: SkillExecutionNode) -> None:
        for seq, action in enumerate(n.react_pending):
            result = by_seq.get(seq)
            child = _node_from_manifest(rt.loader.require(action["skill_id"]))
            child.objective = action.get("objective", "")
            child.arguments = dict(action.get("arguments", {}))
            if result is None:  # 屏障保证不应发生，保守记为失败喂回下轮
                child.status = S.FAILED
                child.error = "分支结果缺失"
            else:
                child.status = result["status"]
                child.output = result.get("output")
                child.error = result.get("error")
            child.finished_at = utcnow()
            n.steps.append(child)
            appended.append(child)
            if child.status == S.FAILED:
                n.react_retries[child.skill_id] = (
                    n.react_retries.get(child.skill_id, 0) + 1
                )
        n.react_pending = []

    new_tree, merged = update_node(tree, path, merge)
    for child in appended:
        if rt.on_leaf is not None:
            await rt.on_leaf({**state, "tree": new_tree}, child)
    logger.info("[react-join] %s 第 %d 波合入 %d 个分支（成功 %d/失败 %d）",
                merged.skill_id, round_no, len(appended),
                sum(c.status == S.COMPLETED for c in appended),
                sum(c.status == S.FAILED for c in appended))
    return {"tree": new_tree, "cursor": list(path), "status": S.RUNNING,
            "level": merged.level, "skill_id": merged.skill_id}


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
    answer = state.get("final_answer") or (
        "\n".join(outputs) if outputs else (tree.output or "（技能执行未产生输出）")
    )
    final_status = S.FAILED if failed else S.COMPLETED
    failed_errors = [n.error for n in iter_nodes(tree)
                     if n.status == S.FAILED and n.error]
    error = "; ".join(failed_errors) if failed_errors else None
    try:
        await rt.mark_terminal(state["task_id"], final_status, answer, error)
    except Exception:
        logger.exception("写回任务终态失败")
    logger.info("task_id=%s 执行树：\n%s", state["task_id"], tree_to_view(tree))
    return {
        "status": final_status,
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
    }


# ---------- 条件边 ----------
def _classify(state: AgentState) -> str:
    node = get_node(state["tree"], state["cursor"])
    return NEXT_BY_TYPE[node.node_type]


def _after_descend(state: AgentState) -> str:
    node = get_node(state["tree"], state["cursor"])
    if node.status == S.COMPLETED:
        return "complete"
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
    if parent.node_type == "react":
        return "complete"  # react 节点由自身回路收束，complete 只负责向上收束
    if parent.step_index < len(parent.plan):
        return "start_step"
    return "complete"  # dynamic 的所有步骤已完成，收束自身


def _after_react_think(state: AgentState):
    """react_think 出口：收尾走 complete；否则把当前批次全部扇出为并发分支。"""
    node = get_node(state["tree"], state["cursor"])
    if node.status == S.COMPLETED or not node.react_pending:
        return "complete"
    path = state["cursor"]
    return [
        Send("parallel_execute", {
            "parent_path": list(path),
            "round": node.react_round,
            "seq": seq,
            "task_id": state["task_id"],
            "skill_id": action["skill_id"],
            "function": action["function"],
            "module": action.get("module"),
            "fs_path": action.get("fs_path"),
            "objective": action["objective"],
            "arguments": action["arguments"],
            "text_input": state["user_input"],
        })
        for seq, action in enumerate(node.react_pending)
    ]


def _after_react_join(state: AgentState) -> str:
    """扇入后：react 节点已收尾则向上收束，否则进入下一轮思考。"""
    node = get_node(state["tree"], state["cursor"])
    if node.status == S.COMPLETED:
        return "complete"
    return "react_think"


def build_graph(rt: Runtime):
    """TaskGraph：执行单个 chat_task（thread_id=task_id）。

    一级意图路由已在独立的 RouterGraph 完成，本图从 enter_task 起步。
    """
    g = StateGraph(AgentState)
    g.add_node("enter_task", enter_task_node)
    g.add_node("descend", descend_node)
    g.add_node("plan", plan_node)
    g.add_node("start_step", start_step_node)
    g.add_node("execute", execute_node)
    g.add_node("complete", complete_node)
    g.add_node("finish", finish_node)
    g.add_node("react_think", react_think_node)
    g.add_node("parallel_execute", parallel_execute_node)
    g.add_node("react_join", react_join_node)

    g.add_edge(START, "enter_task")
    g.add_conditional_edges(
        "enter_task", _classify,
        {"execute": "execute", "descend": "descend", "plan": "plan",
         "react_think": "react_think"},
    )
    g.add_conditional_edges(
        "descend", _after_descend,
        {"execute": "execute", "descend": "descend", "plan": "plan",
         "react_think": "react_think", "complete": "complete"},
    )
    g.add_edge("plan", "start_step")
    g.add_conditional_edges(
        "start_step", _classify,
        {"execute": "execute", "descend": "descend", "plan": "plan",
         "react_think": "react_think"},
    )
    g.add_edge("execute", "complete")
    g.add_conditional_edges(
        "complete", _after_complete,
        {"start_step": "start_step", "complete": "complete", "finish": "finish"},
    )
    g.add_edge("finish", END)

    # ReAct 回路：think 增量规划 → 同批动作扇出并发执行 → join 屏障合入 → 下一轮
    g.add_conditional_edges(
        "react_think", _after_react_think,
        {"complete": "complete", "execute": "parallel_execute"},
    )
    g.add_edge("parallel_execute", "react_join")
    g.add_conditional_edges(
        "react_join", _after_react_join,
        {"react_think": "react_think", "complete": "complete"},
    )

    return g.compile(checkpointer=rt.saver)


def run_config(
    rt: Runtime,
    thread_id: str,
    *,
    run_name: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> RunnableConfig:
    """构造图调用配置。

    run_name/tags/metadata 会透传到 LangSmith 等追踪后端，便于排查问题：
    - run_name : 本次运行的显示名，如 task_task-6beb3138_天气查询
    - tags     : 按图类型 / 技能 / 会话筛选（task / skill:xxx / session:xxx）
    - metadata : 业务键值（task_id、session_id、entry_skill_id 等）
    """
    cfg: RunnableConfig = {"configurable": {"thread_id": thread_id, "runtime": rt}}
    if run_name:
        cfg["run_name"] = run_name
    if tags:
        cfg["tags"] = tags
    if metadata:
        cfg["metadata"] = metadata
    return cfg


# ---------- 对外：启动 / 恢复 / 查看记忆 ----------
async def start_task(rt: Runtime, app, task_id: str) -> dict[str, Any]:
    """启动一个任务的 TaskGraph（thread_id=task_id）。"""
    task = await rt.get_task(task_id) or {}
    entry = task.get("entry_skill_id") or "unknown"
    title = task.get("title") or entry
    session = task.get("session_id") or "unknown"
    initial: AgentState = {
        "task_id": task_id,
        "status": S.RUNNING,
        "level": 0,
        "skill_id": "",
        "cursor": [],
    }
    try:
        return await app.ainvoke(
            initial,
            run_config(
                rt,
                task_id,
                run_name=f"task_{task_id}_{title}",
                tags=["task", f"skill:{entry}", f"session:{session}"],
                metadata={
                    "task_id": task_id,
                    "session_id": session,
                    "entry_skill_id": entry,
                    "title": title,
                    "phase": "start",
                },
            ),
        )
    except GraphInterrupt:
        logger.info("task_id=%s 已在步骤边界暂停", task_id)
        snap = await app.aget_state(run_config(rt, task_id))
        return snap.values


async def resume_task(rt: Runtime, app, task_id: str) -> dict[str, Any]:
    """凭 task_id 从 checkpoint 恢复被暂停的 TaskGraph。"""
    await rt.mark_running(task_id)
    task = await rt.get_task(task_id) or {}
    entry = task.get("entry_skill_id") or "unknown"
    session = task.get("session_id") or "unknown"
    try:
        return await app.ainvoke(
            Command(resume={"resume": True}),
            run_config(
                rt,
                task_id,
                run_name=f"resume_{task_id}",
                tags=["task", "resume", f"skill:{entry}", f"session:{session}"],
                metadata={
                    "task_id": task_id,
                    "session_id": session,
                    "entry_skill_id": entry,
                    "phase": "resume",
                },
            ),
        )
    except GraphInterrupt:
        logger.info("task_id=%s 仍处于暂停（又收到新的暂停请求）", task_id)
        snap = await app.aget_state(run_config(rt, task_id))
        return snap.values


async def get_memory(rt: Runtime, app, task_id: str) -> dict[str, Any] | None:
    """读取某任务在 LangGraph（PostgreSQL checkpoint）中持久化的记忆快照。"""
    snap = await app.aget_state(run_config(rt, task_id))
    return snap.values if snap else None
