"""按技能 name/description 与无关种子写入 intent_math。"""

from __future__ import annotations

import logging

from ..db import db
from ..db.models import IntentMathRow
from .embeddings import embed_texts
from .loader import SkillLoader
from .match_config import load_match_yaml, score_limit_for
from .schema import SkillManifest
from .tokenize import tokens_for_tsvector

logger = logging.getLogger(__name__)

SOURCE_SEED = "seed"
SOURCE_LLM = "llm_gen"
SOURCE_FORBID_SEED = "forbid_seed"
SOURCE_FORBID_LLM = "forbid_gen"


def _clean(text: str) -> str:
    return " ".join((text or "").split()).strip()[:1024]


def _upsert(row: IntentMathRow) -> None:
    msg = _clean(row.msg)
    if not msg:
        return
    row.msg = msg
    db.intents.upsert(row, tokens_for_tsvector(msg))


def _seed_msgs(manifest: SkillManifest) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in [manifest.name, manifest.description, *manifest.keywords]:
        msg = _clean(str(raw))
        if not msg or msg in seen:
            continue
        seen.add(msg)
        out.append(msg)
    return out


def sync_intent_seeds(loader: SkillLoader | None = None) -> dict[str, int]:
    yaml_cfg = load_match_yaml()
    db.intent_config.insert_missing(yaml_cfg.config_defaults())
    loader = loader or SkillLoader()
    manifests = loader.iter_all()
    n = 0
    ids: set[str] = set()
    for m in manifests:
        ids.add(m.skill_id)
        limit = score_limit_for(yaml_cfg, skill_type=m.type, level=m.level)
        for msg in _seed_msgs(m):
            _upsert(
                IntentMathRow(
                    msg=msg,
                    score_limit=limit,
                    skill_id=m.skill_id,
                    skill_level=m.level,
                    answer="",
                    forbid=0,
                    source=SOURCE_SEED,
                    seed=m.skill_id,
                )
            )
            n += 1
    db.intents.prune_missing_skills(ids)

    forbid_n = 0
    valid: set[str] = set()
    for item in yaml_cfg.forbid:
        valid.add(item.seed)
        for msg in [_clean(item.seed), *(_clean(q) for q in item.queries)]:
            if not msg:
                continue
            _upsert(
                IntentMathRow(
                    msg=msg,
                    score_limit=yaml_cfg.forbid_score_limit,
                    skill_id=None,
                    skill_level=0,
                    answer=item.answer,
                    forbid=1,
                    source=SOURCE_FORBID_SEED,
                    seed=item.seed,
                )
            )
            forbid_n += 1
    db.intents.prune_forbid_seeds(valid)
    logger.info("intent_math 种子：业务 %d，无关 %d", n, forbid_n)
    return {"skill_queries": n, "forbid_queries": forbid_n, "skills": len(ids)}


def fill_missing_embeddings(batch_size: int = 32) -> int:
    filled = 0
    while True:
        rows = db.intents.list_missing_embeddings(limit=batch_size)
        if not rows:
            break
        vectors = embed_texts([r.msg for r in rows])
        any_ok = False
        for row, vec in zip(rows, vectors, strict=True):
            if vec is None or row.id is None:
                continue
            db.intents.fill_embedding(row.id, vec)
            filled += 1
            any_ok = True
        if not any_ok:
            break
    return filled


def _parse_lines(text: str) -> list[str]:
    out: list[str] = []
    for line in (text or "").splitlines():
        line = line.strip().lstrip("-•* ").lstrip("0123456789.、) ")
        line = _clean(line)
        if line:
            out.append(line)
    return out


def _llm_lines(prompt: str) -> list[str]:
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI

        from ..config import settings

        kwargs = settings.get_llm_kwargs()
        kwargs["temperature"] = 0.4
        llm = ChatOpenAI(**kwargs)
        resp = llm.invoke(
            [
                SystemMessage(content="只输出问法列表，每行一条，不要解释。"),
                HumanMessage(content=prompt),
            ]
        )
        content = getattr(resp, "content", None) or str(resp)
        return _parse_lines(content if isinstance(content, str) else str(content))
    except Exception:
        logger.exception("LLM 造句失败")
        return []


def generate_skill_queries(loader: SkillLoader | None = None, per_skill: int = 12) -> int:
    yaml_cfg = load_match_yaml()
    loader = loader or SkillLoader()
    total = 0
    for m in loader.iter_all():
        limit = score_limit_for(yaml_cfg, skill_type=m.type, level=m.level)
        prompt = (
            f"技能名称：{m.name}\n技能描述：{m.description}\n"
            f"请生成 {per_skill} 条用户可能说的中文口语问法，覆盖短句、完整句、同义改写。"
            "不要生成明显属于其他业务（翻译/计算/闲聊互串）的句子。"
        )
        for msg in _llm_lines(prompt)[:per_skill]:
            _upsert(
                IntentMathRow(
                    msg=msg,
                    score_limit=limit,
                    skill_id=m.skill_id,
                    skill_level=m.level,
                    answer="",
                    forbid=0,
                    source=SOURCE_LLM,
                    seed=m.skill_id,
                )
            )
            total += 1
    return total


def generate_forbid_queries(per_seed: int = 8) -> int:
    yaml_cfg = load_match_yaml()
    total = 0
    for item in yaml_cfg.forbid:
        prompt = (
            f"领域词：{item.seed}\n"
            f"请生成 {per_seed} 条与该领域相关、且明显不属于天气/翻译/计算的中文问题。"
        )
        for msg in _llm_lines(prompt)[:per_seed]:
            _upsert(
                IntentMathRow(
                    msg=msg,
                    score_limit=yaml_cfg.forbid_score_limit,
                    skill_id=None,
                    skill_level=0,
                    answer=item.answer,
                    forbid=1,
                    source=SOURCE_FORBID_LLM,
                    seed=item.seed,
                )
            )
            total += 1
    return total


def sync_all(loader: SkillLoader | None = None, *, generate: bool = False) -> dict[str, int]:
    stats = sync_intent_seeds(loader)
    if generate:
        stats["llm_skill"] = generate_skill_queries(loader)
        stats["llm_forbid"] = generate_forbid_queries()
    stats["embeddings"] = fill_missing_embeddings()
    return stats
