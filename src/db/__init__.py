"""PostgreSQL 数据访问层（按表拆分的仓储包）。

结构：
- models.py            表模型（ChatRecord / ChatTask / SkillRow）
- schema.py            建表 DDL
- base.py              仓储基类（共享连接池）
- chat_record.py       chat_record        消息时间线
- chat_task.py         chat_task          执行任务与运行控制
- chat_task_message.py chat_task_message  任务-消息多对多关联
- skill_registry.py    skill_registry     技能注册表
- intent_math.py       intent_math        意图 query
- intent_config.py     intent_config      分层匹配阈值
- database.py          Database 组合根 + 模块级单例 db

对外只暴露模型与 Database/db；业务代码按表调用，例如
``db.messages.insert(...)``、``db.tasks.get(task_id)``。
"""

from .chat_record import ChatRecordRepo
from .chat_task import ChatTaskRepo
from .chat_task_message import ChatTaskMessageRepo
from .database import Database, TableRepo, db
from .models import ChatRecord, ChatTask, IntentMathRow, SkillRow

__all__ = [
    "ChatRecord",
    "ChatTask",
    "SkillRow",
    "IntentMathRow",
    "Database",
    "TableRepo",
    "ChatRecordRepo",
    "ChatTaskRepo",
    "ChatTaskMessageRepo",
    "SkillRegistryRepo",
    "db",
]
