"""高精度匹配：forbid 先行、过线、LLM 开关、按层过滤。"""

from __future__ import annotations

from src.db.models import IntentMathRow
from src.skill_runtime.match_config import load_match_yaml, score_limit_for
from src.skill_runtime.matching import (
    KIND_AUTO,
    KIND_FORBID,
    KIND_FORBID_LLM,
    KIND_LLM,
    KIND_UNKNOWN,
    ScoredHit,
    decide_band,
    match_scope,
)
from src.skill_runtime.rerank import _parse_score_array, lexical_scores


def _hit(msg: str, score: float, *, limit: float, forbid: int = 0,
         skill_id: str | None = None, level: int = 1, answer: str = "") -> ScoredHit:
    return ScoredHit(
        row=IntentMathRow(
            msg=msg,
            score_limit=limit,
            skill_id=skill_id,
            skill_level=level,
            answer=answer,
            forbid=forbid,
        ),
        score=score,
    )


def test_forbid_high_confidence_direct_answer():
    band = decide_band(
        [_hit("买股票", 0.9, limit=0.55, forbid=1, level=0, answer="不炒股")],
        forbid=True,
        score_floor=0.0,
        llm_skip=0.82,
        unknown="没听懂",
    )
    assert band.kind == KIND_FORBID
    assert band.answer == "不炒股"


def test_forbid_gray_zone_needs_llm():
    band = decide_band(
        [_hit("基金怎么看", 0.6, limit=0.55, forbid=1, level=0, answer="不理财")],
        forbid=True,
        score_floor=0.0,
        llm_skip=0.82,
        unknown="没听懂",
    )
    assert band.kind == KIND_FORBID_LLM


def test_below_limit_unknown():
    band = decide_band(
        [_hit("天气", 0.3, limit=0.45, skill_id="weather")],
        forbid=False,
        score_floor=0.0,
        llm_skip=0.78,
        unknown="没听懂",
    )
    assert band.kind == KIND_UNKNOWN
    assert band.answer == "没听懂"


def test_above_skip_auto():
    band = decide_band(
        [_hit("上海天气", 0.9, limit=0.45, skill_id="weather")],
        forbid=False,
        score_floor=0.0,
        llm_skip=0.78,
        unknown="没听懂",
    )
    assert band.kind == KIND_AUTO
    assert band.skill_id == "weather"


def test_gray_zone_llm_confirm():
    band = decide_band(
        [_hit("天气相关", 0.55, limit=0.45, skill_id="weather")],
        forbid=False,
        score_floor=0.0,
        llm_skip=0.78,
        unknown="没听懂",
    )
    assert band.kind == KIND_LLM
    assert band.skill_ids == ["weather"]


def test_atomic_limit_stricter_than_level1():
    yaml_cfg = load_match_yaml()
    assert score_limit_for(yaml_cfg, skill_type="atomic", level=1) > score_limit_for(
        yaml_cfg, skill_type="category", level=1
    )


def test_level_filter_does_not_pick_child():
    rows = [
        IntentMathRow(msg="查天气", score_limit=0.2, skill_id="weather", skill_level=1, forbid=0),
        IntentMathRow(
            msg="查天气", score_limit=0.2, skill_id="weather.current", skill_level=2, forbid=0
        ),
        IntentMathRow(
            msg="股票", score_limit=0.2, skill_id=None, skill_level=0, forbid=1, answer="不炒股"
        ),
    ]
    l1 = match_scope(
        "查天气",
        skill_level=1,
        skill_ids={"weather"},
        candidates=rows,
        use_vector=False,
        force_lexical=True,
    )
    assert l1.kind in {KIND_AUTO, KIND_LLM}
    assert l1.skill_id == "weather"

    forbid = match_scope(
        "股票", skill_level=0, forbid=True, candidates=rows, use_vector=False, force_lexical=True
    )
    assert forbid.kind in {KIND_FORBID, KIND_FORBID_LLM}
    assert "炒股" in forbid.answer or forbid.answer == "不炒股"


def test_rerank_json_parse_and_lexical():
    assert _parse_score_array("[0.1, 0.9]", 2) == [0.1, 0.9]
    assert lexical_scores("查天气", ["查天气", "买股票"])[0] == 1.0
