"""OpenAI function calling（tools / tool_calls）协议适配层。

所有 LLM 决策点（意图选择、技能选择、dynamic 规划、react 动作决策）统一走
原生 tool_calls 协议：

- 工具名由 skill_id 映射而来（OpenAI 工具名只允许 ``[A-Za-z0-9_-]``，
  因此把 ``.`` 替换为 ``__``），解析时按名字映射回 skill_id；
- atomic 技能的工具参数 schema 直接从注册函数签名推导
  （排除注入参数 ``text`` 与 ``**kwargs``），provider 端即可校验参数；
- ``choose`` 模式用于二选一的"选择类"决策（意图/技能），工具只带轻量
  ``reason`` 参数；``plan`` 模式附加 objective/input_from 规划元参数；
- LLM 返回的 ``AIMessage.tool_calls`` 由 :func:`parse_tool_calls` 统一解析。
"""

from __future__ import annotations

import inspect
import logging
import typing
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage

from .functions import resolve_function
from .schema import SkillManifest

logger = logging.getLogger(__name__)

# ReAct 收束工具：信息足够时由模型调用它给出最终回答
REACT_FINISH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "信息已足够回答用户：结束 ReAct 并给出最终中文结论",
        "parameters": {
            "type": "object",
            "properties": {"final_answer": {"type": "string"}},
            "required": ["final_answer"],
        },
    },
}


def tool_name_for(skill_id: str) -> str:
    """skill_id → 合法工具名（``.`` 非法，替换为 ``__``，保持唯一可逆）。"""
    return skill_id.replace(".", "__")


def parse_tool_calls(message: AIMessage) -> list[tuple[str, dict[str, Any]]]:
    """提取一次响应中的全部工具调用（保持返回顺序），[(工具名, 参数)]。"""
    calls: list[tuple[str, dict[str, Any]]] = []
    for tc in getattr(message, "tool_calls", None) or []:
        calls.append((tc["name"], dict(tc.get("args") or {})))
    return calls


# ---------- JSON Schema 推导 ----------
_JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_schema_type(annotation: Any) -> dict[str, Any]:
    """类型注解 → JSON Schema 片段；未知/联合类型退化为 string。"""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}
    args = typing.get_args(annotation)
    if args:  # Optional[X] / X | None：取第一个非 None 类型
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _json_schema_type(non_none[0])
    return {"type": _JSON_TYPES.get(annotation, "string")}


def function_parameters(fn: Callable[..., Any]) -> dict[str, Any]:
    """从函数签名推导参数 JSON Schema。

    排除 ``text``（由执行层注入的输入通道）与 ``*args``/``**kwargs``。
    """
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, p in inspect.signature(fn).parameters.items():
        if name == "text" or p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
            continue
        properties[name] = _json_schema_type(p.annotation)
        if p.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


def manifest_tool(manifest: SkillManifest, mode: str) -> dict[str, Any]:
    """把技能清单转为 OpenAI 工具定义。

    - ``choose``：选择类决策，工具名即候选，参数只带 reason（轻量）
    - ``plan``：dynamic 规划，atomic 参数取自函数签名，并附加
      objective / input_from 两个规划元参数
    - ``react``：atomic 动作调用，参数即函数真实参数
    """
    name = tool_name_for(manifest.skill_id)
    description = f"[{manifest.skill_id}] {manifest.name}。{manifest.description}"

    if mode == "choose":
        parameters: dict[str, Any] = {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": [],
        }
    else:
        # atomic：参数 schema 从解析后的函数签名推导（本地 module 或全局注册表）
        fn: Callable[..., Any] | None = None
        if manifest.type == "atomic":
            try:
                fn = resolve_function(manifest.function, manifest.module,
                                      manifest.fs_path, manifest.skill_id)
            except Exception:
                logger.warning("技能 %s 函数解析失败，工具参数 schema 为空",
                               manifest.skill_id)
        if fn is not None:
            parameters = function_parameters(fn)
        else:
            parameters = {"type": "object", "properties": {}, "required": []}
        if mode == "plan":
            parameters = {
                **parameters,
                "properties": {
                    **parameters["properties"],
                    "objective": {"type": "string",
                                  "description": "本步骤目标（可选）"},
                    "input_from": {"type": "string",
                                   "enum": ["user", "prev_output"],
                                   "description": "输入来源：user=用户原话；"
                                                  "prev_output=上一步产物"},
                },
            }
    return {
        "type": "function",
        "function": {"name": name, "description": description,
                     "parameters": parameters},
    }
