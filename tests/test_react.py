"""ReAct 增量规划/并发调度的纯逻辑测试（不依赖 DB 与 LLM）。"""

from __future__ import annotations

from src.skill_runtime.graph import (
    ReactAction,
    _normalize_actions,
    _react_keyword_actions,
    _react_observations,
)
from src.skill_runtime.schema import SkillManifest
from src.skill_runtime.state import S, SkillExecutionNode


def _atomic(skill_id: str, function: str, keywords: list[str] | None = None,
            parallelizable: bool = True, self_exclusive: bool = False) -> SkillManifest:
    return SkillManifest(
        skill_id=skill_id, name=skill_id, type="atomic",
        function=function, keywords=keywords or [],
        parallelizable=parallelizable, self_exclusive=self_exclusive,
    )


def _react_node() -> SkillExecutionNode:
    return SkillExecutionNode(
        skill_id="weather.trip_react", name="react", node_type="react", level=2,
    )


def test_normalize_drops_unknown_and_duplicates() -> None:
    children = [_atomic("a", "fn_a"), _atomic("b", "fn_b")]
    by_id = {c.skill_id: c for c in children}
    node = _react_node()

    actions = _normalize_actions(
        [
            ReactAction(skill_id="a", arguments={"city": "上海"}),
            ReactAction(skill_id="a", arguments={"city": "上海"}),  # 完全重复
            ReactAction(skill_id="ghost", arguments={}),           # 不存在
            ReactAction(skill_id="b"),
        ],
        by_id, node,
    )

    assert [a["skill_id"] for a in actions] == ["a", "b"]
    assert all(a["function"] for a in actions)
    assert actions[0]["parallelizable"] is True


def test_normalize_respects_retry_cap() -> None:
    children = [_atomic("a", "fn_a")]
    by_id = {c.skill_id: c for c in children}
    node = _react_node()
    node.react_retries = {"a": 2}

    assert _normalize_actions([ReactAction(skill_id="a")], by_id, node) == []


def test_normalize_flags_non_parallelizable() -> None:
    children = [_atomic("a", "fn_a"), _atomic("b", "fn_b", self_exclusive=True)]
    by_id = {c.skill_id: c for c in children}

    actions = _normalize_actions(
        [ReactAction(skill_id="a"), ReactAction(skill_id="b")],
        by_id, _react_node(),
    )
    flags = {a["skill_id"]: a["parallelizable"] for a in actions}
    assert flags == {"a": True, "b": False}


def test_keyword_actions_skip_executed_and_pick_best() -> None:
    children = [
        _atomic("r_current", "get_weather", keywords=["天气"]),
        _atomic("r_forecast", "get_weather_forecast", keywords=["未来", "预报"]),
    ]
    node = _react_node()

    first = _react_keyword_actions(children, node, "查一下未来的预报")
    assert [a["skill_id"] for a in first] == ["r_forecast"]

    # 模拟该动作已合入执行树
    done = SkillExecutionNode(
        skill_id="r_forecast", name="f", node_type="atomic", level=3,
        status=S.COMPLETED, output="预报...",
    )
    node.steps.append(done)

    second = _react_keyword_actions(children, node, "查一下未来的预报")
    assert [a["skill_id"] for a in second] == ["r_current"]

    node.steps.append(SkillExecutionNode(
        skill_id="r_current", name="c", node_type="atomic", level=3,
        status=S.COMPLETED, output="天气...",
    ))
    assert _react_keyword_actions(children, node, "查一下未来的预报") == []


def test_observations_include_success_and_failure() -> None:
    node = _react_node()
    node.steps = [
        SkillExecutionNode(skill_id="a", name="a", node_type="atomic", level=3,
                           status=S.COMPLETED, output="晴天"),
        SkillExecutionNode(skill_id="b", name="b", node_type="atomic", level=3,
                           status=S.FAILED, error="boom"),
    ]
    text = _react_observations(node)
    assert "a 成功" in text and "晴天" in text
    assert "b 失败" in text and "boom" in text
