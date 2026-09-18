# 安装：pip install langchain-openai langgraph pydantic

import ast
import operator
from typing import Annotated, Literal, TypedDict

from langchain_openai import ChatOpenAI

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from .config import settings


# ---------- 0. 一个基础模型，各域按需绑定工具（互不影响） ----------
base = ChatOpenAI(**settings.get_llm_kwargs())

# ---------- 1. 叶子技能：真正的可执行能力 ----------
@tool
def get_weather(city: str) -> str:
    """查询指定城市当前天气"""
    data = {"北京": "晴 26°C", "上海": "多云 29°C"}
    return data.get(city, f"暂无 {city} 的天气数据")

@tool
def get_weather_forecast(city: str, days: int) -> str:
    """查询指定城市未来几天的天气预报"""
    return f"{city} 未来 {days} 天：晴转多云，18~29°C"

@tool
def translate_text(text: str, target_lang: str) -> str:
    """把文本翻译成目标语言"""
    table = {"hello": "你好", "你好": "hello"}
    return table.get(text, f"[{target_lang}] {text}")

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow}

def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval(node.operand)
        return v if isinstance(node.op, ast.UAdd) else -v
    raise ValueError("不支持的表达式")

@tool
def calc(expr: str) -> str:
    """计算数学表达式，例如 '(1+2)*4'"""
    return str(_eval(ast.parse(expr, mode="eval")))

# ---------- 2. 二级：按技能域分组（隔离边界就在这） ----------
TOOL_GROUPS = {
    "weather":   [get_weather, get_weather_forecast],
    "translate": [translate_text],
    "calc":      [calc],
    "chat":      [],   # 无工具域
}

# ---------- 3. 一级：路由决策（LLM 结构化输出） ----------
class RouteDecision(BaseModel):
    domain: Literal["weather", "translate", "calc", "chat"] = Field(description="选中的技能域")
    reason: str = Field(description="路由理由")

router_llm = base.with_structured_output(RouteDecision)

ROUTER_PROMPT = (
    "你是多级路由器的第一级。根据用户请求选择技能域：\n"
    "- weather：查询天气/气温/预报\n"
    "- translate：翻译\n"
    "- calc：数学计算\n"
    "- chat：其他闲聊\n"
    "只输出判断，不要执行任务。"
)

# ---------- 4. 状态与节点 ----------
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    domain: str

def router_node(state: State):
    decision = router_llm.invoke([SystemMessage(content=ROUTER_PROMPT), *state["messages"]])
    return {"domain": decision.domain}

def make_domain_node(tools):
    """把一个技能域封装成子 Agent：域内由模型自主决定调用哪个工具、循环到回答"""
    agent = create_react_agent(base, tools)
    def node(state: State):
        result = agent.invoke({"messages": state["messages"]})
        return {"messages": result["messages"]}
    return node

# ---------- 5. 构图：一级路由 → 二级域执行 ----------
graph = StateGraph(State)
graph.add_node("router", router_node)
for name in TOOL_GROUPS:
    graph.add_node(name, make_domain_node(TOOL_GROUPS[name]))

graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    lambda s: s["domain"],
    {name: name for name in TOOL_GROUPS},   # domain 值 → 对应域节点
)
for name in TOOL_GROUPS:
    graph.add_edge(name, END)

app = graph.compile()

# ---------- 6. 运行 ----------
def run(q: str):
    r = app.invoke({"messages": [HumanMessage(content=q)]})
    print(f"[路由到 {r['domain']}] {r['messages'][-1].content}")

if __name__ == '__main__':
    #run("北京今天天气怎么样？")   # → weather：域内模型自己决定用哪个天气工具
    run("把 hello 翻译成中文")   # → translate
    # run("(1+2)*4 等于多少")      # → calc
    # run("讲个笑话")              # → chat，无工具直接回答

    for name in TOOL_GROUPS:
        print(name)
