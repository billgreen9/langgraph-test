"""技能目录本地函数（skill.json module 字段）的加载与解析测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.skill_runtime.functions import (
    FUNCTIONS,
    load_skill_function,
    resolve_function,
)
from src.skill_runtime.schema import SkillManifest
from src.skill_runtime.tooling import function_parameters, manifest_tool

LOCAL_FN = '''
def shout(text: str = "", suffix: str = "!", **_: object) -> str:
    """原样返回文本并追加后缀。"""
    return f"{(text or '').upper()}{suffix}"
'''

MANIFEST_TMPL = {
    "name": "本地函数测试",
    "description": "目录内 functions.py 的本地函数",
    "type": "atomic",
    "function": "shout",
    "module": "functions",
    "keywords": ["本地"],
}


def _make_skill(root: Path, dirname: str, fn_body: str = LOCAL_FN) -> SkillManifest:
    """在临时目录下构造一个带本地 functions.py 的技能目录与 manifest。"""
    d = root / dirname
    d.mkdir(parents=True)
    (d / "functions.py").write_text(fn_body, encoding="utf-8")
    (d / "skill.json").write_text(
        json.dumps(MANIFEST_TMPL, ensure_ascii=False), encoding="utf-8"
    )
    return SkillManifest(
        **MANIFEST_TMPL,
        skill_id=f"demo.{dirname}",
        fs_path=str(d),
    )


def test_load_local_function_and_namespace_isolation(tmp_path: Path) -> None:
    """同名函数存在于两个技能目录：按 skill_id 命名空间隔离，互不冲突。"""
    m1 = _make_skill(tmp_path, "alpha")
    m2 = _make_skill(
        tmp_path, "beta",
        'def shout(text: str = "", suffix: str = "?", **_: object) -> str:\n'
        '    return f"{text}{suffix}"\n',
    )

    fn1 = load_skill_function(m1.fs_path, m1.module or "", "shout", m1.skill_id)
    fn2 = load_skill_function(m2.fs_path, m2.module or "", "shout", m2.skill_id)

    assert fn1 is not fn2  # 不同命名空间，非同一对象
    assert fn1("hi") == "HI!"
    assert fn2("hi") == "hi?"
    # 内置注册表不受本地加载影响
    assert "shout" not in FUNCTIONS


def test_resolve_function_falls_back_to_registry(tmp_path: Path) -> None:
    """未声明 module 时走全局注册表；声明的技能解析到本地函数。"""
    m_local = _make_skill(tmp_path, "gamma")
    m_global = SkillManifest(
        skill_id="demo.global_calc", name="calc", type="atomic", function="calc",
        fs_path=str(tmp_path),
    )

    fn_local = resolve_function("shout", "functions", m_local.fs_path, m_local.skill_id)
    fn_global = resolve_function("calc", None, m_global.fs_path, m_global.skill_id)

    assert fn_local is not None and fn_local("ok") == "OK!"
    assert fn_global is FUNCTIONS["calc"]


def test_missing_module_and_symbol(tmp_path: Path) -> None:
    """本地模块缺失 / 符号缺失分别抛出可定位的异常。"""
    m = _make_skill(tmp_path, "delta")

    with pytest.raises(FileNotFoundError):
        resolve_function("shout", "no_such_module", m.fs_path, m.skill_id)
    with pytest.raises(AttributeError):
        resolve_function("no_such_symbol", "functions", m.fs_path, m.skill_id)


def test_tool_schema_derived_from_local_signature(tmp_path: Path) -> None:
    """tool_calls 工具参数 schema 能从本地函数签名推导。"""
    m = _make_skill(tmp_path, "epsilon")
    tool = manifest_tool(m, "react")
    props = tool["function"]["parameters"]["properties"]

    assert tool["function"]["name"] == "demo__epsilon"
    assert props == {"suffix": {"type": "string"}}
    assert function_parameters(
        load_skill_function(m.fs_path, "functions", "shout", m.skill_id)
    )["properties"]["suffix"] == {"type": "string"}
