"""原子 function_call 注册表。

每个原子技能（skill.json 中 type=atomic）通过 ``function`` 字段绑定到这里
注册的一个 Python 函数。统一签名：

    def fn(text: str, **arguments) -> str

- ``text``：本次调用的输入文本（用户输入，或动态规划中上一步产物）
- ``arguments``：规划器为该步骤生成的参数（可缺省，函数自行从 text 解析）
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import operator
import re
import sys
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------- 注册表 ----------
FunctionRegistry = dict[str, Callable[..., str]]
FUNCTIONS: FunctionRegistry = {}


def register_function(name: str) -> Callable[[Callable[..., str]], Callable[..., str]]:
    def deco(fn: Callable[..., str]) -> Callable[..., str]:
        FUNCTIONS[name] = fn
        fn.__function_name__ = name  # type: ignore[attr-defined]
        return fn

    return deco


# ---------- 技能目录本地函数（渐进式加载） ----------
# 缓存键：skill_id 命名空间 + 模块 + 符号，进程内只导入一次
_LOCAL_CACHE: dict[tuple[str, str, str], Callable[..., str]] = {}


def load_skill_function(base_dir: str | Path, module: str, symbol: str,
                        namespace: str) -> Callable[..., str]:
    """从技能目录内的 Python 文件惰性加载函数。

    - ``base_dir``：技能目录（manifest.fs_path）；``module`` 是相对它的
      Python 文件（``"functions"`` 或 ``"functions.py"``）；
    - 以 ``namespace``（skill_id）派生唯一模块别名注册进 sys.modules，
      不同技能目录中的同名函数互不冲突（命名空间隔离）；
    - 进程内缓存，同一技能函数只导入一次。
    """
    key = (namespace, module, symbol)
    if key in _LOCAL_CACHE:
        return _LOCAL_CACHE[key]

    p = Path(base_dir) / module
    if p.suffix != ".py":
        p = p.with_suffix(".py")
    if not p.is_file():
        raise FileNotFoundError(f"技能本地模块不存在：{p}")

    alias = f"skills.{namespace.replace('.', '_')}.{p.stem}"
    spec = importlib.util.spec_from_file_location(alias, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法构造模块 spec：{p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)

    fn = getattr(mod, symbol, None)
    if not callable(fn):
        raise AttributeError(f"模块 {p} 中未找到可调用函数：{symbol}")
    _LOCAL_CACHE[key] = fn
    logger.info("已加载技能本地函数 %s:%s（模块别名 %s）", namespace, symbol, alias)
    return fn


def resolve_function(name: str | None, module: str | None = None,
                     base_dir: str | None = None,
                     namespace: str = "") -> Callable[..., str] | None:
    """统一解析技能绑定的可调用对象。

    - 未声明 ``module``：查全局 FUNCTIONS 注册表（内置/共享函数）；
    - 声明了 ``module``：从技能目录本地文件按 skill_id 命名空间惰性导入，
      加载失败抛异常（文件不存在 / 符号缺失）。
    """
    if not module:
        return FUNCTIONS.get(name or "")
    return load_skill_function(base_dir or ".", module, name or "", namespace)


# ---------- 天气 ----------
_WEATHER_DB = {
    "北京": "晴，26°C，西北风2级",
    "上海": "多云转晴，29°C，东南风3级",
    "广州": "雷阵雨，31°C，湿度85%",
    "深圳": "阵雨，30°C，南风2级",
    "杭州": "阴，27°C，东风2级",
    "成都": "小雨，23°C，北风1级",
}

_CITY_PATTERN = re.compile(r"(北京|上海|广州|深圳|杭州|成都|西安|南京|武汉|重庆)")
_DAYS_PATTERN = re.compile(r"(\d+|[一二两三四五六七八九十])\s*天")
_CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _extract_city(text: str, city: str | None) -> str:
    if city:
        return city
    m = _CITY_PATTERN.search(text or "")
    return m.group(1) if m else "北京"


def _extract_days(text: str, days: int | None) -> int:
    if days:
        return int(days)
    m = _DAYS_PATTERN.search(text or "")
    if not m:
        return 3
    raw = m.group(1)
    return _CN_NUM.get(raw, int(raw) if raw.isdigit() else 3)


@register_function("get_weather")
def get_weather(text: str = "", city: str | None = None, **_: object) -> str:
    """查询指定城市当前天气。"""
    city = _extract_city(text, city)
    info = _WEATHER_DB.get(city, "晴，25°C 左右（模拟数据）")
    return f"{city}当前天气：{info}"


@register_function("get_weather_forecast")
def get_weather_forecast(text: str = "", city: str | None = None,
                         days: int | None = None, **_: object) -> str:
    """查询指定城市未来几天的天气预报。"""
    city = _extract_city(text, city)
    n = _extract_days(text, days)
    return f"{city}未来{n}天预报：白天多云为主，局部有短时小雨，气温 18~29°C，适合出行备伞。"


# ---------- 翻译 ----------
_TRANS_TABLE = {
    "hello": "你好",
    "你好": "hello",
    "天气": "weather",
    "weather": "天气",
    "出差": "business trip",
}


@register_function("translate_text")
def translate_text(text: str = "", target_lang: str | None = None,
                   source_text: str | None = None, **_: object) -> str:
    """把文本翻译成目标语言（简易词典 + 标记翻译的模拟实现）。"""
    raw = (source_text or text or "").strip()

    # 显式给出待译文本时直接使用
    content = raw
    if not source_text:
        # 在“翻译成/译成/翻译为/翻译”处切分：前半是正文，后半是目标语言
        m = re.search(r"(翻译成|译成|翻译为|翻译)", raw)
        if m:
            head, tail = raw[: m.start()], raw[m.end():]
            content = re.sub(r"^(请|帮我|把|将)\s*", "", head).strip(" :，,")
            if not target_lang:
                target_lang = tail.strip()
        content = content.strip("\"'“”‘’")

    if content in _TRANS_TABLE:
        return _TRANS_TABLE[content]

    has_chinese = bool(re.search(r"[\u4e00-\u9fff]", content))
    if target_lang:
        tgt = target_lang.lower()
        if ("en" in tgt or "英" in tgt) and has_chinese:
            return f"[zh→en] {content}"
        if ("zh" in tgt or "中" in tgt or "汉" in tgt) and not has_chinese:
            return f"[en→zh] {content}"
    # 默认：中文翻英文，外文翻中文
    if has_chinese:
        return f"[zh→en] {content}"
    return f"[en→zh] {content}"


# ---------- 数学计算 ----------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _safe_eval(node.operand)
        return v if isinstance(node.op, ast.UAdd) else -v
    raise ValueError("不支持的表达式")


_EXPR_PATTERN = re.compile(r"[\d\s+\-*/%().]+[\d)]")


@register_function("calc")
def calc(text: str = "", expr: str | None = None, **_: object) -> str:
    """安全求值数学表达式。"""
    expression = expr
    if not expression:
        m = _EXPR_PATTERN.search(text or "")
        expression = m.group(0).strip() if m else ""
    expression = expression.strip()
    if not expression:
        return "未找到可计算的数学表达式"
    try:
        value = _safe_eval(ast.parse(expression, mode="eval"))
    except (ValueError, SyntaxError, ZeroDivisionError) as exc:
        return f"表达式无法计算：{exc}"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{expression} = {value}"


# ---------- 闲聊 / 总结（兜底原子技能） ----------
@register_function("chat_reply")
def chat_reply(text: str = "", **_: object) -> str:
    """规则化的兜底回复与简报总结，保证无 LLM 时也可闭环。"""
    t = (text or "").strip()
    if not t:
        return "我在，有什么可以帮你？"
    if "笑话" in t:
        return "程序员最擅长的运动：push（代码），最不擅长的：commit 承诺。"
    if any(k in t for k in ("你好", "您好", "在吗")):
        return "你好！我可以帮你查天气、做翻译、算数学，或者进行出行规划。"
    if any(k in t for k in ("简报", "建议", "总结", "天气", "预报", "°C", "气温")):
        return "出行简报：根据以上天气信息，建议合理安排行程、随身带伞、注意温差。"
    return f"收到：{t}"
