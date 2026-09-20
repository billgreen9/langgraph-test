"""后台技能扫描线程。

主程序启动后，本线程定时从 skills 目录加载【一级技能】并 upsert 到
PostgreSQL 的 skill_registry 表（prefetched=TRUE）；更深层级由运行时
按渐进式加载原则按需读取并缓存。
"""

from __future__ import annotations

import logging
import threading

from ..db import Database, SkillRow
from .loader import SkillLoader

logger = logging.getLogger(__name__)


class SkillScanner(threading.Thread):
    def __init__(
        self,
        db: Database,
        loader: SkillLoader,
        interval_seconds: float,
        run_immediately: bool = True,
    ) -> None:
        super().__init__(name="skill-scanner", daemon=True)
        self.db = db
        self.loader = loader
        self.interval = interval_seconds
        self.run_immediately = run_immediately
        self._stop_event = threading.Event()
        self._scan_lock = threading.Lock()

    def stop(self, timeout: float | None = None) -> None:
        self._stop_event.set()
        self.join(timeout=timeout)

    def scan_once(self) -> list[str]:
        """执行一次一级技能扫描并同步到数据库，返回本次加载的 skill_id 列表。"""
        with self._scan_lock:
            manifests = self.loader.scan_level1()
            ids: list[str] = []
            for m in manifests:
                row = SkillRow(**m.to_row(prefetched=True))
                self.db.skills.upsert(row)
                ids.append(m.skill_id)
            # 清理磁盘上已经删除的一级技能（深层渐进缓存不受影响）
            if ids:
                self.db.skills.prune_level1(set(ids))
            logger.info("一级技能扫描完成：%s", ids or "<空>")
            try:
                from .intent_sync import sync_intent_seeds

                sync_intent_seeds(self.loader)
            except Exception:
                logger.exception("intent_math 种子同步失败")
            return ids

    def run(self) -> None:
        if self.run_immediately:
            self._safe_scan()
        while not self._stop_event.wait(self.interval):
            self._safe_scan()

    def _safe_scan(self) -> None:
        try:
            self.scan_once()
        except Exception:  # 后台线程不能因为单次扫描失败而退出
            logger.exception("技能扫描失败，将在下个周期重试")
