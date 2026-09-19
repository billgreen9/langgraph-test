"""LangGraph 记忆（checkpoint state）。

记忆中包含：
- ``task_id``：任务 id，作为 LangGraph thread_id，后续凭它启动/暂停/恢复
- ``session_id``：会话 id（消息归并/结果聚合作用域）
- ``status``：整体运行状态（running/planning/paused/completed/failed）
- ``level``：当前执行深度
- ``skill_id``：当前执行的技能 id
- ``tree``：**递归数组结构**的技能执行树，记录执行中所有动态技能及其子步骤
  （category 节点有 1 个选中的子节点；dynamic 节点的 steps 是其规划出的多个
   子技能步骤，子技能本身还可以是 dynamic，从而形成任意深度的递归）
"""

from __future__ import annotations

import operator
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# ---------- 状态常量 ----------
class S:
    PENDING = "pending"
    PLANNING = "planning"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------- 递归执行树 ----------
class PlanStep(BaseModel):
    """动态技能规划出的单个步骤（指向某个子技能）。"""

    skill_id: str
    objective: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    input_from: str = "user"  # user / prev_output：本步输入取用户输入还是上一步产物
    status: str = S.PENDING


class SkillExecutionNode(BaseModel):
    """技能执行树的递归节点。"""

    skill_id: str
    name: str = ""
    node_type: str  # category / atomic / dynamic
    level: int
    fs_path: str = ""
    function: str | None = None
    module: str | None = None  # 声明时 function 从技能目录本地文件惰性导入
    objective: str = ""  # 若本节点由动态规划步骤实例化，记录该步骤目标
    arguments: dict[str, Any] = Field(default_factory=dict)
    input_from: str = "user"

    status: str = S.PENDING
    # dynamic 技能的规划结果
    plan: list[PlanStep] = Field(default_factory=list)
    step_index: int = 0
    # react 技能的增量规划状态
    react_round: int = 0  # 已进入的思考轮次（从 1 开始）
    react_max_rounds: int = 5
    # 当前轮待并发执行的动作批次（think 写入、fan-out 边读取、join 清空）
    react_pending: list[dict[str, Any]] = Field(default_factory=list)
    # skill_id -> 失败次数，用于限制同一动作反复重试
    react_retries: dict[str, int] = Field(default_factory=dict)
    # 递归数组：子技能执行节点（dynamic/react 可有多个；category 只有被选中的 1 个）
    steps: list[SkillExecutionNode] = Field(default_factory=list)

    output: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


SkillExecutionNode.model_rebuild()


# ---------- LangGraph state ----------
class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    # TaskGraph 的执行身份：thread_id = task_id；session_id 为消息归并/聚合作用域
    task_id: str
    session_id: str
    user_input: str
    status: str
    level: int
    skill_id: str
    # 游标路径：tree.steps 上的索引序列，[] 表示根节点
    cursor: list[int]
    tree: SkillExecutionNode
    # ReAct 并行分支结果扇入通道：各 parallel_execute 分支只写这里，
    # 由 react_join 按轮次（round）消费并合入 tree。元素为原生类型 dict。
    branch_results: Annotated[list[dict[str, Any]], operator.add]
    final_answer: str


# ---------- 树的纯函数式操作（深拷贝后修改，保证 checkpoint 可追踪） ----------
def get_node(tree: SkillExecutionNode, path: list[int]) -> SkillExecutionNode:
    node = tree
    for idx in path:
        node = node.steps[idx]
    return node


def update_node(
    tree: SkillExecutionNode, path: list[int], mutate
) -> tuple[SkillExecutionNode, SkillExecutionNode]:
    """返回 (新树, 被更新节点的副本)。``mutate`` 在副本节点上原地修改。"""
    new_tree = tree.model_copy(deep=True)
    target = get_node(new_tree, path)
    mutate(target)
    return new_tree, target


def append_child(
    tree: SkillExecutionNode, path: list[int], child: SkillExecutionNode
) -> tuple[SkillExecutionNode, list[int]]:
    """在 path 所指节点下追加一个子执行节点，返回新树与子节点游标路径。"""
    new_tree = tree.model_copy(deep=True)
    parent = get_node(new_tree, path)
    parent.steps.append(child)
    return new_tree, path + [len(parent.steps) - 1]


def iter_nodes(tree: SkillExecutionNode) -> Iterator[SkillExecutionNode]:
    yield tree
    for child in tree.steps:
        yield from iter_nodes(child)


def collect_dynamic_skills(tree: SkillExecutionNode) -> list[SkillExecutionNode]:
    """收集执行树中所有 dynamic 技能节点（即"递归数组中所有动态 skills"）。"""
    return [n for n in iter_nodes(tree) if n.node_type == "dynamic"]


def collect_leaf_outputs(tree: SkillExecutionNode) -> list[str]:
    """按执行顺序收集所有已完成原子技能的输出。"""
    out: list[str] = []
    for n in iter_nodes(tree):
        if n.node_type == "atomic" and n.status == S.COMPLETED and n.output:
            out.append(n.output)
    return out


def tree_to_view(tree: SkillExecutionNode, indent: int = 0) -> str:
    """生成便于日志展示的树形视图。"""
    lines = ["  " * indent + f"- [{tree.status}] {tree.skill_id} "
                             f"(type={tree.node_type}, level={tree.level})"]
    for child in tree.steps:
        lines.append(tree_to_view(child, indent + 1))
    return "\n".join(lines)
