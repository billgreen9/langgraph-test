"""技能清单（skill.json）的数据模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SkillType = Literal["category", "atomic", "dynamic", "react"]


class PlannerConfig(BaseModel):
    """动态技能的规划器配置。"""

    max_steps: int = Field(default=5, ge=1, le=20)
    objective_hint: str = ""


class ReactConfig(BaseModel):
    """ReAct 技能的增量规划配置（每轮只规划一批可并发的原子动作）。"""

    max_rounds: int = Field(default=5, ge=1, le=20)
    objective_hint: str = ""


class SkillManifest(BaseModel):
    """一个技能目录的完整描述。

    - category：容器型技能，路由按渐进式原则继续向下匹配子技能
    - atomic：原子技能，直接映射到 functions 注册表中的一次 function_call
    - dynamic：动态规划技能，运行时由规划器拆成多个步骤逐步执行
    - react：ReAct 技能，每轮由 LLM 基于已观测结果规划下一批可并发原子动作
    """

    skill_id: str
    name: str
    description: str = ""
    type: SkillType
    keywords: list[str] = Field(default_factory=list)
    function: str | None = None  # atomic 技能绑定的函数名
    planner: PlannerConfig | None = None
    react: ReactConfig | None = None
    # atomic 并发声明：parallelizable=False 或 self_exclusive=True 的技能
    # 即使被 ReAct 规划器与其他动作同批返回，也会被调度器机械拆为单独一波
    parallelizable: bool = True
    self_exclusive: bool = False
    # 兜底规划时的步骤顺序提示（越小越靠前），默认 100
    order: int = 100

    # 由加载器根据目录位置推导
    level: int = 1
    parent_id: str | None = None
    fs_path: str = ""
    has_children: bool = False

    def to_row(self, prefetched: bool) -> dict:
        """转换为 db.SkillRow 所需的 dict（延迟导入避免循环依赖）。"""
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "skill_type": self.type,
            "level": self.level,
            "parent_id": self.parent_id,
            "fs_path": self.fs_path,
            "keywords": self.keywords,
            "has_children": self.has_children,
            "manifest": self.model_dump(),
            "prefetched": prefetched,
        }
