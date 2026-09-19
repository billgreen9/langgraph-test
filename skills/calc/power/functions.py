def power(text: str = "", base: float | None = None,
          exponent: float | None = None, **_: object) -> str:
    """本地函数示例：计算 base 的 exponent 次幂（仅声明在本技能目录内）。"""
    if base is None or exponent is None:
        import re

        m = re.search(r"(-?\d+(?:\.\d+)?)\s*的\s*(-?\d+(?:\.\d+)?)\s*(?:次方|次幂)", text or "")
        if not m:
            return "未找到幂运算参数（需要形如：2 的 10 次方）"
        base, exponent = float(m.group(1)), float(m.group(2))
    value = base ** exponent
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{base:g} 的 {exponent:g} 次方 = {value}"
