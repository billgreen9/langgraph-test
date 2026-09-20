"""RouterGraph：forbid 先行，再只匹配 skill_level=1。"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from ..db import ChatTask, db
from .graph import Runtime, _runtime
from .matching import (
    KIND_AUTO,
    KIND_FORBID,
    KIND_FORBID_LLM,
    KIND_LLM,
    llm_classify_forbid,
    llm_confirm_skill,
    match_scope,
    unknown_answer,
)
from .rerank import get_rerank_llm

logger = logging.getLogger(__name__)

OPEN_TASK_STATUSES = {"collecting", "pending", "running", "paused"}


class RouterState(TypedDict, total=False):
    chat_id: str
    session_id: str
    user_id: str | None
    message_content: str
    intent_id: str
    entry_skill_id: str
    title: str
    mode: str
    reply_text: str
    existing_task_ids: list[str]
    task_id: str
    task_ids: list[str]


def _reply(text: str) -> dict[str, Any]:
    return {
        "mode": "reply",
        "reply_text": text,
        "intent_id": "",
        "entry_skill_id": "",
        "title": "",
        "existing_task_ids": [],
    }


async def load_message_node(
    state: RouterState, config: RunnableConfig
) -> dict[str, Any]:
    chat_id = state["chat_id"]
    message = await asyncio.to_thread(db.messages.get, chat_id)
    if message is None:
        raise ValueError(f"聊天消息不存在：{chat_id}")
    return {
        "session_id": message.session_id,
        "user_id": message.user_id,
        "message_content": message.content,
    }


def _match_forbid(text: str, force_lexical: bool) -> Any:
    llm = None if force_lexical else get_rerank_llm()
    return match_scope(
        text,
        skill_level=0,
        forbid=True,
        use_vector=not force_lexical,
        llm=llm,
        force_lexical=force_lexical,
    )


def _pick_level1(rt: Runtime, text: str, level1: list, force_lexical: bool):
    ids = {m.skill_id for m in level1}
    by_id = {m.skill_id: m for m in level1}
    llm = None if force_lexical else get_rerank_llm()
    band = match_scope(
        text,
        skill_level=1,
        skill_ids=ids,
        forbid=False,
        use_vector=not force_lexical,
        llm=llm,
        force_lexical=force_lexical,
    )
    if band.kind == KIND_AUTO and band.skill_id in by_id:
        return by_id[band.skill_id]
    if band.kind == KIND_LLM:
        if force_lexical:
            return None
        cands = [by_id[s] for s in band.skill_ids if s in by_id]
        return llm_confirm_skill(get_rerank_llm() or rt.llm, text, cands)
    return None


async def route_intents_node(
    state: RouterState, config: RunnableConfig
) -> dict[str, Any]:
    rt = _runtime(config)
    chat_id = state["chat_id"]
    text = state["message_content"]

    linked = await asyncio.to_thread(
        db.task_messages.list_tasks_for_message, chat_id
    )
    open_ids = [t.task_id for t in linked if t.status in OPEN_TASK_STATUSES]
    if open_ids:
        return {
            "intent_id": "",
            "entry_skill_id": "",
            "title": "",
            "mode": "attach",
            "existing_task_ids": open_ids,
        }

    force_lexical = rt.force_keyword
    try:
        forbid_band = await asyncio.to_thread(_match_forbid, text, force_lexical)
    except Exception:
        logger.exception("无关检测失败")
        forbid_band = None
    if forbid_band is not None and forbid_band.kind == KIND_FORBID:
        logger.info("[router] 命中无关 score=%.3f", forbid_band.score)
        return _reply(forbid_band.answer or unknown_answer())
    if forbid_band is not None and forbid_band.kind == KIND_FORBID_LLM and not force_lexical:
        if llm_classify_forbid(get_rerank_llm() or rt.llm, text, forbid_band.answer):
            return _reply(forbid_band.answer or unknown_answer())

    level1 = await rt.level1_candidates()
    if not level1:
        return _reply(unknown_answer())
    try:
        chosen = await asyncio.to_thread(_pick_level1, rt, text, level1, force_lexical)
    except Exception:
        logger.exception("一级匹配失败")
        chosen = None
    if chosen is None or chosen.level != 1:
        return _reply(unknown_answer())

    logger.info("[router] 一级命中 %s", chosen.skill_id)
    return {
        "intent_id": chosen.skill_id,
        "entry_skill_id": chosen.skill_id,
        "title": chosen.name,
        "mode": "create",
        "existing_task_ids": [],
    }


async def create_node(state: RouterState, config: RunnableConfig) -> dict[str, Any]:
    task_id = f"task-{uuid.uuid4().hex[:8]}"
    logger.info("[router] 新建任务 %s -> %s", task_id, state["entry_skill_id"])
    return {"task_id": task_id, "task_ids": [task_id]}


async def attach_node(state: RouterState, config: RunnableConfig) -> dict[str, Any]:
    return {"task_ids": list(state["existing_task_ids"])}


async def persist_node(state: RouterState, config: RunnableConfig) -> dict[str, Any]:
    if state.get("mode") == "reply":
        return {"task_ids": []}
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
    else:
        for tid in state["task_ids"]:
            await asyncio.to_thread(db.task_messages.link, tid, chat_id)
    return {}


def _after_route_intents(state: RouterState) -> str:
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
        {"create": "create", "attach": "attach", "reply": "persist"},
    )
    g.add_edge("create", "persist")
    g.add_edge("attach", "persist")
    g.add_edge("persist", END)
    return g.compile(checkpointer=rt.saver)
