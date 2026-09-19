"""意图技能加载器（Markdown 版）。

意图技能与业务技能完全隔离：
- 位置：``skills/intents/*.md``（文件型，不是含 skill.json 的目录，
  因此不会被 SkillLoader 扫描、不会进入 skill_registry、不会被执行图调用）
- 格式：YAML front matter 提供结构化元数据，Markdown 正文是给路由 LLM 的
  判别说明（意图边界与正反例）

一个意图固定映射一个一级业务技能（``entry_skill``），只决定任务的入口，
入口技能内部如何 category/dynamic/react 由 TaskGraph 自行决定。
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from ..config import settings

logger = logging.getLogger(__name__)

INTENTS_DIRNAME = "intents"


class IntentSpec(BaseModel):
    intent_id: str
    name: str
    keywords: list[str] = Field(default_factory=list)
    entry_skill: str
    order: int = 100
    body: str = ""  # Markdown 正文（意图判别说明）
    fs_path: str = ""


class IntentLoader:
    def __init__(self, intents_dir: Path | str | None = None) -> None:
        base = Path(intents_dir) if intents_dir else Path(settings.skills_dir)
        self.intents_dir = base / INTENTS_DIRNAME

    def load_all(self) -> list[IntentSpec]:
        """全量加载意图目录下的 .md 文件（意图数量小，无需渐进式/缓存）。"""
        result: list[IntentSpec] = []
        if not self.intents_dir.is_dir():
            logger.warning("意图目录不存在：%s", self.intents_dir)
            return result
        for path in sorted(self.intents_dir.glob("*.md"), key=lambda p: p.name):
            spec = self._load_one(path)
            if spec is not None:
                result.append(spec)
        logger.info("加载 %d 个意图技能：%s",
                    len(result), [s.intent_id for s in result])
        return result

    def _load_one(self, path: Path) -> IntentSpec | None:
        text = path.read_text(encoding="utf-8")
        meta, body = self._split_front_matter(text)
        if meta is None:
            logger.error("意图文件缺少 YAML front matter，已跳过：%s", path)
            return None
        intent_id = str(meta.get("intent_id") or path.stem).strip()
        name = str(meta.get("name") or intent_id).strip()
        entry_skill = str(meta.get("entry_skill") or "").strip()
        if not entry_skill:
            logger.error("意图 %s 缺少必填的 entry_skill，已跳过", intent_id)
            return None
        keywords = meta.get("keywords") or []
        if isinstance(keywords, str):
            keywords = [keywords]
        return IntentSpec(
            intent_id=intent_id,
            name=name,
            keywords=[str(k) for k in keywords],
            entry_skill=entry_skill,
            order=int(meta.get("order", 100)),
            body=body.strip(),
            fs_path=str(path),
        )

    @staticmethod
    def _split_front_matter(text: str) -> tuple[dict | None, str]:
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return None, text
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                meta = yaml.safe_load("\n".join(lines[1:i])) or {}
                return meta, "\n".join(lines[i + 1:])
        return None, text

    def validate_entry_skills(self, valid_skill_ids: set[str]) -> None:
        """校验每个意图的 entry_skill 都指向存在的一级业务技能。"""
        for spec in self.load_all():
            if spec.entry_skill not in valid_skill_ids:
                raise ValueError(
                    f"意图 {spec.intent_id} 的 entry_skill='{spec.entry_skill}' "
                    f"不在一级业务技能集合 {sorted(valid_skill_ids)} 中"
                )
