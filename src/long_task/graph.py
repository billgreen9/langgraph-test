"""只含一个长任务节点的 LangGraph。

节点会请求 API 的 ``GET /slow``。该接口本身会挂起超过 5 分钟，
图在这里等待 HTTP 响应返回。不依赖模型、数据库或检查点。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.request
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

DEFAULT_SLOW_URL = "http://127.0.0.1:8000/slow"
# /slow 固定睡 5 分 5 秒，客户端超时留出余量。
REQUEST_TIMEOUT_SECONDS = 10 * 60


class LongTaskState(TypedDict, total=False):
    url: str
    status: str
    elapsed_seconds: float
    response: dict[str, Any]


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


async def long_task_node(state: LongTaskState) -> LongTaskState:
    """请求刚才的长耗时 HTTP 接口，直到它返回。"""
    url = state.get("url") or DEFAULT_SLOW_URL
    logger.info("long_task 开始请求 %s", url)
    started = time.monotonic()
    body = await asyncio.to_thread(_get_json, url, REQUEST_TIMEOUT_SECONDS)
    elapsed = round(time.monotonic() - started, 1)
    logger.info("long_task 收到响应，耗时 %.1f 秒：%s", elapsed, body)
    return {
        "status": str(body.get("status", "done")),
        "elapsed_seconds": elapsed,
        "response": body,
    }


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(LongTaskState)
    graph.add_node("long_task", long_task_node)
    graph.add_edge(START, "long_task")
    graph.add_edge("long_task", END)
    return graph.compile()
