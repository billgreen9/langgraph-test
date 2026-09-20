"""skills/skills.yaml：分层过线、LLM 开关、无关种子。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import settings

CONFIG_FILENAME = "skills.yaml"
UNKNOWN_ANSWER = "对不起，我没有听懂，请详细说明"


@dataclass
class ForbidSeed:
    seed: str
    answer: str
    queries: list[str] = field(default_factory=list)


@dataclass
class MatchYaml:
    rank_top_k: int = 5
    keyword_top_k: int = 20
    semantic_top_k: int = 20
    rerank_top_n: int = 8
    rrf_k: int = 60
    atomic_score_limit: float = 0.82
    forbid_score_limit: float = 0.55
    forbid_llm_skip: float = 0.82
    score_limit_by_level: dict[int, float] = field(
        default_factory=lambda: {1: 0.45, 2: 0.40, 3: 0.35, 4: 0.30}
    )
    llm_skip_by_level: dict[int, float] = field(
        default_factory=lambda: {1: 0.78, 2: 0.74, 3: 0.70, 4: 0.66}
    )
    unknown_answer: str = UNKNOWN_ANSWER
    forbid: list[ForbidSeed] = field(default_factory=list)

    def config_defaults(self) -> dict[str, str]:
        data = {
            "rank_top_k": str(self.rank_top_k),
            "keyword_top_k": str(self.keyword_top_k),
            "semantic_top_k": str(self.semantic_top_k),
            "rerank_top_n": str(self.rerank_top_n),
            "rrf_k": str(self.rrf_k),
            "unknown_answer": self.unknown_answer,
            "atomic_score_limit": str(self.atomic_score_limit),
            "forbid_score_limit": str(self.forbid_score_limit),
            "forbid_llm_skip": str(self.forbid_llm_skip),
        }
        for level, value in self.score_limit_by_level.items():
            data[f"score_limit_level_{level}"] = str(value)
        for level, value in self.llm_skip_by_level.items():
            data[f"llm_skip_level_{level}"] = str(value)
        return data


def _as_int_map(raw: Any) -> dict[int, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[int, float] = {}
    for key, value in raw.items():
        try:
            out[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def load_match_yaml(path: Path | None = None) -> MatchYaml:
    cfg_path = path or (Path(settings.skills_dir) / CONFIG_FILENAME)
    data: dict[str, Any] = {}
    if cfg_path.is_file():
        loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            data = loaded
    forbid: list[ForbidSeed] = []
    for item in data.get("forbid") or []:
        if not isinstance(item, dict):
            continue
        seed = str(item.get("seed") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if not seed or not answer:
            continue
        extra = item.get("queries") or []
        if isinstance(extra, str):
            extra = [extra]
        forbid.append(
            ForbidSeed(
                seed=seed,
                answer=answer,
                queries=[str(q).strip() for q in extra if str(q).strip()],
            )
        )
    yaml_cfg = MatchYaml(forbid=forbid)
    limits = _as_int_map(data.get("score_limit_by_level"))
    skips = _as_int_map(data.get("llm_skip_by_level"))
    if limits:
        yaml_cfg.score_limit_by_level = limits
    if skips:
        yaml_cfg.llm_skip_by_level = skips
    for attr in (
        "rank_top_k", "keyword_top_k", "semantic_top_k", "rerank_top_n", "rrf_k",
        "atomic_score_limit", "forbid_score_limit", "forbid_llm_skip", "unknown_answer",
    ):
        if attr in data and data[attr] is not None:
            current = getattr(yaml_cfg, attr)
            setattr(yaml_cfg, attr, type(current)(data[attr]))
    return yaml_cfg


def value_by_level(mapping: dict[int, float], level: int, floor: float) -> float:
    level = max(0, int(level or 0))
    if level in mapping:
        return mapping[level]
    if not mapping:
        return floor
    if level < min(mapping):
        return mapping[min(mapping)]
    known = max(mapping)
    step = 0.04
    if known >= 2 and (known - 1) in mapping:
        step = mapping[known - 1] - mapping[known]
    return max(floor, mapping[known] - step * (level - known))


def score_limit_for(yaml_cfg: MatchYaml, *, skill_type: str, level: int) -> float:
    if skill_type == "atomic":
        return yaml_cfg.atomic_score_limit
    return value_by_level(yaml_cfg.score_limit_by_level, level, floor=0.20)


def llm_skip_for(yaml_cfg: MatchYaml, stored: dict[str, str], level: int) -> float:
    key = f"llm_skip_level_{level}"
    if key in stored:
        try:
            return float(stored[key])
        except ValueError:
            pass
    return value_by_level(yaml_cfg.llm_skip_by_level, level, floor=0.55)
