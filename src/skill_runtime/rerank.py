"""交叉编码器精排。

优先用 Chat LLM 对 (query, msg) 打 0~1 同义分（现有 OpenAI 兼容接口即可）。
失败或 force_lexical 时退回 token 覆盖率。后续可把 RERANK_MODEL 换成专用 reranker。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..config import settings
from .tokenize import token_overlap

logger = logging.getLogger(__name__)

_ARRAY = re.compile(r"\[[\s\d.,eE+-]+\]")

_rerank_llm = None


def get_rerank_llm():
    """temperature=0 的打分模型；RERANK_MODEL 可指向专用 rerank/chat 模型。"""
    global _rerank_llm
    if _rerank_llm is not None:
        return _rerank_llm
    if not settings.openai_api_key:
        return None
    try:
        from langchain_openai import ChatOpenAI

        kwargs = settings.get_llm_kwargs()
        kwargs["temperature"] = 0
        if settings.rerank_model:
            kwargs["model"] = settings.rerank_model
        _rerank_llm = ChatOpenAI(**kwargs)
        return _rerank_llm
    except Exception:
        logger.exception("初始化精排 LLM 失败")
        return None


def lexical_scores(query: str, docs: list[str]) -> list[float]:
    return [token_overlap(query, d) for d in docs]


def _parse_score_array(text: str, n: int) -> list[float] | None:
    text = (text or "").strip()
    match = _ARRAY.search(text.replace("\n", " "))
    raw = match.group(0) if match else text
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or len(data) != n:
        return None
    out: list[float] = []
    for item in data:
        try:
            out.append(min(1.0, max(0.0, float(item))))
        except (TypeError, ValueError):
            return None
    return out


def llm_cross_encode(llm: Any, query: str, docs: list[str]) -> list[float] | None:
    if llm is None or not docs:
        return None
    listing = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(docs))
    prompt = (
        "判断用户问句与各候选说法是否同义。只输出一个 JSON 数组，"
        f"长度必须为 {len(docs)}，每项是 0 到 1 的小数，1 表示完全同一意图。\n"
        f"用户问句：{query}\n候选：\n{listing}"
    )
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        resp = llm.invoke(
            [
                SystemMessage(content="只输出 JSON 数组，不要解释。"),
                HumanMessage(content=prompt),
            ]
        )
        content = getattr(resp, "content", None) or str(resp)
        if not isinstance(content, str):
            content = str(content)
        return _parse_score_array(content, len(docs))
    except Exception:
        logger.exception("交叉编码器 LLM 打分失败")
        return None


def rerank_scores(
    query: str,
    docs: list[str],
    *,
    llm: Any = None,
    force_lexical: bool = False,
) -> list[float]:
    if not docs:
        return []
    if not force_lexical:
        scored = llm_cross_encode(llm, query, docs)
        if scored is not None:
            return scored
    return lexical_scores(query, docs)
