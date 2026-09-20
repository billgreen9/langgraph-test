"""forbid 先行；业务按 skill_level 双路召回 → RRF → 交叉编码器定分。"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from rank_bm25 import BM25Okapi

from ..db import db
from ..db.models import IntentMathRow
from .embeddings import embed_query
from .match_config import (
    UNKNOWN_ANSWER,
    MatchYaml,
    llm_skip_for,
    load_match_yaml,
)
from .rerank import rerank_scores
from .schema import SkillManifest
from .tokenize import tokenize, tokens_for_tsquery
from .tooling import manifest_tool, parse_tool_calls, tool_name_for

logger = logging.getLogger(__name__)

KIND_FORBID = "forbid"
KIND_AUTO = "auto"
KIND_LLM = "llm_confirm"
KIND_UNKNOWN = "unknown"
KIND_FORBID_LLM = "forbid_llm"

REJECT_TOOL = {
    "type": "function",
    "function": {
        "name": "none_of_these",
        "description": "候选技能都不合适，应拒绝匹配。",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


@dataclass
class ScoredHit:
    row: IntentMathRow
    score: float
    rrf: float = 0.0


@dataclass
class MatchBand:
    kind: str
    skill_id: str | None = None
    answer: str = ""
    score: float = 0.0
    skill_ids: list[str] = field(default_factory=list)
    hits: list[ScoredHit] = field(default_factory=list)


def _as_int(raw: str | None, default: int) -> int:
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


def _as_float(raw: str | None, default: float) -> float:
    try:
        return float(raw) if raw is not None else default
    except ValueError:
        return default


def runtime_params(yaml_cfg: MatchYaml | None = None) -> tuple[MatchYaml, dict[str, str]]:
    yaml_cfg = yaml_cfg or load_match_yaml()
    stored: dict[str, str] = {}
    try:
        stored = db.intent_config.get_all()
    except Exception:
        logger.debug("intent_config 不可用，使用 yaml")
    return yaml_cfg, stored


def unknown_answer() -> str:
    yaml_cfg, stored = runtime_params()
    return stored.get("unknown_answer") or yaml_cfg.unknown_answer or UNKNOWN_ANSWER


def rrf_merge(*ranked: list[IntentMathRow], k: int = 60) -> list[IntentMathRow]:
    scores: dict[int, float] = defaultdict(float)
    by_id: dict[int, IntentMathRow] = {}
    for rows in ranked:
        for rank, row in enumerate(rows, start=1):
            if row.id is None:
                continue
            by_id[row.id] = row
            scores[row.id] += 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [by_id[i] for i, _ in ordered]


def _bm25_order(query: str, rows: list[IntentMathRow]) -> list[IntentMathRow]:
    if not rows:
        return []
    corpus = [tokenize(r.msg) for r in rows]
    if not any(corpus):
        return list(rows)
    raw = BM25Okapi(corpus).get_scores(tokenize(query))
    paired = list(zip(rows, raw, strict=True))
    paired.sort(key=lambda item: float(item[1]), reverse=True)
    return [r for r, _ in paired]


def decide_band(
    hits: list[ScoredHit],
    *,
    forbid: bool,
    score_floor: float,
    llm_skip: float,
    unknown: str,
) -> MatchBand:
    """过线：精排分 > 行上 score_limit（及可选的层 floor）。"""
    passed = [
        h for h in hits
        if h.score > h.row.score_limit and h.score > score_floor
    ]
    if not passed:
        return MatchBand(kind=KIND_UNKNOWN, answer=unknown, hits=hits)
    best = passed[0]
    if forbid:
        if best.score > llm_skip:
            return MatchBand(
                kind=KIND_FORBID,
                answer=best.row.answer or unknown,
                score=best.score,
                hits=passed,
            )
        return MatchBand(
            kind=KIND_FORBID_LLM,
            answer=best.row.answer or unknown,
            score=best.score,
            hits=passed,
        )
    ids: list[str] = []
    for h in passed:
        if h.row.skill_id and h.row.skill_id not in ids:
            ids.append(h.row.skill_id)
    if best.score > llm_skip:
        return MatchBand(
            kind=KIND_AUTO,
            skill_id=best.row.skill_id,
            score=best.score,
            skill_ids=ids,
            hits=passed,
        )
    return MatchBand(
        kind=KIND_LLM,
        skill_id=best.row.skill_id,
        score=best.score,
        skill_ids=ids,
        hits=passed,
        answer=unknown,
    )


def _recall(
    query: str,
    *,
    forbid: int,
    skill_level: int,
    skill_ids: set[str] | None,
    use_vector: bool,
    yaml_cfg: MatchYaml,
    stored: dict[str, str],
    candidates: list[IntentMathRow] | None,
) -> list[IntentMathRow]:
    if candidates is not None:
        return [
            r for r in candidates
            if int(r.forbid) == forbid
            and r.skill_level == skill_level
            and r.enabled
            and (skill_ids is None or r.skill_id in skill_ids)
        ]
    kw_n = _as_int(stored.get("keyword_top_k"), yaml_cfg.keyword_top_k)
    se_n = _as_int(stored.get("semantic_top_k"), yaml_cfg.semantic_top_k)
    fts = db.intents.keyword_search(
        tokens_for_tsquery(query),
        forbid=forbid,
        skill_level=skill_level,
        skill_ids=skill_ids,
        limit=kw_n,
    )
    vec: list[IntentMathRow] = []
    if use_vector:
        emb = embed_query(query)
        if emb:
            vec = db.intents.vector_search(
                emb,
                forbid=forbid,
                skill_level=skill_level,
                skill_ids=skill_ids,
                limit=se_n,
            )
    bm25 = _bm25_order(query, _dedupe(fts + vec))
    k = _as_int(stored.get("rrf_k"), yaml_cfg.rrf_k)
    return rrf_merge(fts, vec, bm25, k=k)


def _dedupe(rows: list[IntentMathRow]) -> list[IntentMathRow]:
    seen: dict[int, IntentMathRow] = {}
    extra: list[IntentMathRow] = []
    for row in rows:
        if row.id is None:
            extra.append(row)
        elif row.id not in seen:
            seen[row.id] = row
    return list(seen.values()) + extra


def match_scope(
    query: str,
    *,
    skill_level: int,
    skill_ids: set[str] | None = None,
    forbid: bool = False,
    use_vector: bool = True,
    llm: Any = None,
    force_lexical: bool = False,
    candidates: list[IntentMathRow] | None = None,
    yaml_cfg: MatchYaml | None = None,
) -> MatchBand:
    yaml_cfg, stored = runtime_params(yaml_cfg)
    unknown = stored.get("unknown_answer") or yaml_cfg.unknown_answer
    rows = _recall(
        query,
        forbid=1 if forbid else 0,
        skill_level=0 if forbid else skill_level,
        skill_ids=None if forbid else skill_ids,
        use_vector=use_vector,
        yaml_cfg=yaml_cfg,
        stored=stored,
        candidates=candidates,
    )
    top_n = _as_int(stored.get("rerank_top_n"), yaml_cfg.rerank_top_n)
    shortlist = rows[: max(1, top_n)] if rows else []
    scores = rerank_scores(
        query,
        [r.msg for r in shortlist],
        llm=llm,
        force_lexical=force_lexical,
    )
    hits = [
        ScoredHit(row=row, score=score)
        for row, score in zip(shortlist, scores, strict=True)
    ]
    hits.sort(key=lambda h: h.score, reverse=True)
    if forbid:
        skip = _as_float(stored.get("forbid_llm_skip"), yaml_cfg.forbid_llm_skip)
        floor = 0.0
    else:
        skip = llm_skip_for(yaml_cfg, stored, skill_level)
        floor = 0.0
    return decide_band(
        hits, forbid=forbid, score_floor=floor, llm_skip=skip, unknown=unknown
    )


def llm_confirm_skill(llm: Any, text: str, manifests: list[SkillManifest]) -> SkillManifest | None:
    """灰区确认：可调用 none_of_these 拒绝。"""
    if not manifests or llm is None:
        return None
    listing = "\n".join(f"- {m.skill_id}：{m.name}。{m.description}" for m in manifests)
    prompt = (
        "根据用户请求选择最合适的一个候选技能（调用对应工具）。"
        "若都不合适，必须调用 none_of_these。\n候选：\n"
        f"{listing}\n用户请求：{text}"
    )
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        tools = [manifest_tool(m, "choose") for m in manifests] + [REJECT_TOOL]
        resp = llm.bind_tools(tools).invoke(
            [SystemMessage(content=prompt), HumanMessage(content=text)]
        )
        rev = {tool_name_for(m.skill_id): m for m in manifests}
        for name, _args in parse_tool_calls(resp):
            if name == "none_of_these":
                return None
            selected = rev.get(name)
            if selected is not None:
                logger.info("LLM 确认技能 %s", selected.skill_id)
                return selected
    except Exception:
        logger.exception("LLM 确认技能失败")
    return None


def llm_classify_forbid(llm: Any, text: str, answer: str) -> bool:
    """灰区无关：True 表示确认为本系统无关。"""
    if llm is None:
        return False
    prompt = (
        "判断用户消息是否属于天气、翻译、计算之外的无关请求。"
        "只回答 yes 或 no。\n"
        f"若无关将回复：{answer}\n用户：{text}"
    )
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        resp = llm.invoke(
            [SystemMessage(content=prompt), HumanMessage(content=text)]
        )
        content = (getattr(resp, "content", None) or str(resp)).strip().lower()
        return content.startswith("y") or "yes" in content or content.startswith("无关")
    except Exception:
        logger.exception("LLM 无关判定失败")
        return False
