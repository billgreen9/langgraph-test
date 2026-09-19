"""skill_registry 表仓储：技能注册表（后台扫描线程维护 + 渐进式缓存）。"""

from __future__ import annotations

import json
from typing import Any

from .base import TableRepo
from .models import SkillRow, utcnow


def row_to_skill(row: dict[str, Any]) -> SkillRow:
    data = dict(row)
    if isinstance(data.get("keywords"), str):
        data["keywords"] = json.loads(data["keywords"])
    if isinstance(data.get("manifest"), str):
        data["manifest"] = json.loads(data["manifest"])
    return SkillRow(**data)


class SkillRegistryRepo(TableRepo):
    def upsert(self, skill: SkillRow) -> None:
        sql = """
            INSERT INTO skill_registry
                (skill_id, name, description, skill_type, level, parent_id, fs_path,
                 keywords, has_children, prefetched, manifest, updated_at)
            VALUES
                (%(skill_id)s, %(name)s, %(description)s, %(skill_type)s, %(level)s,
                 %(parent_id)s, %(fs_path)s, %(keywords)s, %(has_children)s,
                 %(prefetched)s, %(manifest)s, %(ts)s)
            ON CONFLICT (skill_id) DO UPDATE SET
                name         = EXCLUDED.name,
                description  = EXCLUDED.description,
                skill_type   = EXCLUDED.skill_type,
                level        = EXCLUDED.level,
                parent_id    = EXCLUDED.parent_id,
                fs_path      = EXCLUDED.fs_path,
                keywords     = EXCLUDED.keywords,
                has_children = EXCLUDED.has_children,
                prefetched   = EXCLUDED.prefetched,
                manifest     = EXCLUDED.manifest,
                updated_at   = EXCLUDED.updated_at
        """
        with self.pool.connection() as conn:
            conn.execute(
                sql,
                {
                    "skill_id": skill.skill_id,
                    "name": skill.name,
                    "description": skill.description,
                    "skill_type": skill.skill_type,
                    "level": skill.level,
                    "parent_id": skill.parent_id,
                    "fs_path": skill.fs_path,
                    "keywords": json.dumps(skill.keywords, ensure_ascii=False),
                    "has_children": skill.has_children,
                    "prefetched": skill.prefetched,
                    "manifest": json.dumps(skill.manifest, ensure_ascii=False),
                    "ts": utcnow(),
                },
            )
            conn.commit()

    def get_level1(self) -> list[SkillRow]:
        """只读取一级 skills（后台线程预加载的那一层）。"""
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM skill_registry WHERE level = 1 ORDER BY skill_id"
            ).fetchall()
        return [row_to_skill(r) for r in rows]

    def get(self, skill_id: str) -> SkillRow | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM skill_registry WHERE skill_id = %s", (skill_id,)
            ).fetchone()
        return row_to_skill(row) if row else None

    def delete(self, skill_id: str) -> None:
        with self.pool.connection() as conn:
            conn.execute("DELETE FROM skill_registry WHERE skill_id = %s", (skill_id,))
            conn.commit()

    def prune_level1(self, valid_ids: set[str]) -> None:
        """删除磁盘上已不存在的一级 skill（保留更深层级的渐进式缓存）。"""
        with self.pool.connection() as conn:
            conn.execute(
                "DELETE FROM skill_registry WHERE level = 1 AND skill_id <> ALL(%s)",
                (list(valid_ids),),
            )
            conn.commit()
