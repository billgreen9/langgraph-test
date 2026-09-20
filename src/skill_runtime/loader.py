"""技能目录加载器。

渐进式加载原则：
1. ``scan_level1()`` 只读取 skills 根目录下的一级技能（后台线程定时调用）；
2. 运行时路由命中某个技能后，才调用 ``load_children()`` 加载它的直接子技能，
   多级目录按层依次展开，避免一次性全量载入。

目录约定：每个技能是一个包含 ``skill.json`` 的目录；
skill_id 由相对路径推导（如 ``weather/trip_plan`` -> ``weather.trip_plan``）。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from ..config import settings
from .schema import SkillManifest

MANIFEST_FILE = "skill.json"


class SkillLoader:
    def __init__(self, skills_dir: Path | str | None = None) -> None:
        self.skills_dir = Path(skills_dir or settings.skills_dir)
        # skill_id -> SkillManifest；只有被访问过的层才会进入缓存
        self._cache: dict[str, SkillManifest] = {}
        self._lock = threading.RLock()

    # ---------- 内部工具 ----------
    def _rel_parts(self, path: Path) -> list[str]:
        return path.relative_to(self.skills_dir).parts

    @staticmethod
    def _child_dirs(skill_dir: Path) -> list[Path]:
        """包含 skill.json 的直接子目录（保持文件系统顺序稳定，按名字排序）。"""
        if not skill_dir.is_dir():
            return []
        return sorted(
            (p for p in skill_dir.iterdir() if p.is_dir() and (p / MANIFEST_FILE).is_file()),
            key=lambda p: p.name,
        )

    def _load_dir(self, skill_dir: Path) -> SkillManifest:
        manifest_path = skill_dir / MANIFEST_FILE
        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        parts = self._rel_parts(skill_dir)
        skill_id = ".".join(parts)
        children = self._child_dirs(skill_dir)
        manifest = SkillManifest(
            **data,
            skill_id=skill_id,
            level=len(parts),
            parent_id=".".join(parts[:-1]) or None,
            fs_path=str(skill_dir),
            has_children=bool(children),
        )
        # atomic 技能必须能解析到函数：声明 module 的走技能目录本地文件
        # （只校验文件存在，导入推迟到执行时），否则查全局 FUNCTIONS 注册表
        if manifest.type == "atomic" and manifest.function:
            if manifest.module:
                local = Path(manifest.fs_path) / manifest.module
                if local.suffix != ".py":
                    local = local.with_suffix(".py")
                if not local.is_file():
                    raise ValueError(
                        f"技能 {skill_id} 声明的本地函数模块不存在：{local}"
                    )
            else:
                from .functions import FUNCTIONS

                if manifest.function not in FUNCTIONS:
                    raise ValueError(
                        f"技能 {skill_id} 绑定的函数 '{manifest.function}' 未在 FUNCTIONS 中注册"
                    )
        with self._lock:
            self._cache[skill_id] = manifest
        return manifest

    # ---------- 对外 API ----------
    def scan_level1(self) -> list[SkillManifest]:
        """只加载一级技能（skills 根目录下含 skill.json 的直接子目录）。"""
        result: list[SkillManifest] = []
        if not self.skills_dir.is_dir():
            return result
        for child in sorted(
            (p for p in self.skills_dir.iterdir()
             if p.is_dir() and (p / MANIFEST_FILE).is_file()),
            key=lambda p: p.name,
        ):
            result.append(self._load_dir(child))
        return result

    def load_children(self, parent: SkillManifest) -> list[SkillManifest]:
        """渐进式加载：读取某技能的直接子技能（仅下一层）。"""
        parent_dir = Path(parent.fs_path)
        return [self._load_dir(child) for child in self._child_dirs(parent_dir)]

    def iter_all(self) -> list[SkillManifest]:
        """遍历整棵技能树（同步 intent_math）。"""
        result: list[SkillManifest] = []

        def walk(parent: SkillManifest | None) -> None:
            children = self.scan_level1() if parent is None else self.load_children(parent)
            for child in children:
                result.append(child)
                if child.has_children:
                    walk(child)

        walk(None)
        return result

    def get(self, skill_id: str) -> SkillManifest | None:
        """按 id 获取技能；缓存未命中时按路径直接加载对应层。"""
        with self._lock:
            if skill_id in self._cache:
                return self._cache[skill_id]
        target = self.skills_dir.joinpath(*skill_id.split("."))
        if (target / MANIFEST_FILE).is_file():
            return self._load_dir(target)
        return None

    def require(self, skill_id: str) -> SkillManifest:
        m = self.get(skill_id)
        if m is None:
            raise KeyError(f"技能不存在：{skill_id}")
        return m

    def cached_ids(self) -> set[str]:
        with self._lock:
            return set(self._cache.keys())
