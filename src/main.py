"""主程序入口。

新架构（RouterGraph / TaskGraph 解耦，经 chat_task 表交互）：
1. 后台守护线程定时把 skills 一级业务技能同步到 skill_registry；
2. seed 写入一条 user 消息（chat_record，纯聊天记录）；
3. run 一条消息：RouterGraph 识别意图并落 pending 任务（chat_task），
   每个任务在独立 TaskGraph 中执行（thread_id=task_id），聚合后写 assistant 消息；
4. pause/resume/state 均以 task_id 为粒度。

用法：
    python -m src.main main demo                      # seed + run 端到端演示
    python -m src.main seed --session s1 "你好"       # 写入一条 user 消息
    python -m src.main run <chat_id>                  # 路由+执行+聚合一条消息
    python -m src.main pause <task_id>                # 请求暂停某任务
    python -m src.main resume <task_id>               # 恢复某任务
    python -m src.main state <task_id>                # 查看某任务的 LangGraph 记忆
    python -m src.main tasks [--session s1]           # 列出任务
    python -m src.main list                           # 列出消息
    python -m src.main gen-queries                    # 按技能描述造句写入 intent_math
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import uuid

from langgraph.errors import GraphInterrupt

from .config import settings
from .db import ChatRecord, db
from .runner import process_message, resume_one_task, show_task_memory
from .skill_runtime.loader import SkillLoader
from .skill_runtime.scanner import SkillScanner
from .skill_runtime.state import tree_to_view

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("main")

DEMO_TEXT = "用ReAct智能规划查一下上海天气并给中文简报"
DEFAULT_SESSION = "default"


# ---------- 工具 ----------
def start_scanner(loader: SkillLoader) -> SkillScanner:
    scanner = SkillScanner(db, loader, settings.skill_scan_interval)
    scanner.start()
    return scanner


def wait_level1_loaded(loader: SkillLoader, timeout: float = 5.0) -> None:
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if db.skills.get_level1():
            return
        time.sleep(0.1)
    logger.warning("等待一级技能入库超时，继续执行（Router 有目录扫描兜底）")


def print_memory(mem: dict | None) -> None:
    if not mem:
        print("（无 LangGraph 记忆）")
        return
    tree = mem.get("tree")
    print("─" * 60)
    print(f"task_id : {mem.get('task_id')}")
    print(f"session : {mem.get('session_id')}")
    print(f"status  : {mem.get('status')}")
    print(f"level   : {mem.get('level')}")
    print(f"skill_id: {mem.get('skill_id')}")
    if tree:
        print("递归技能执行树：")
        print(tree_to_view(tree))
    print("─" * 60)


# ---------- CLI 处理 ----------
def cmd_seed(text: str, session_id: str) -> str:
    chat = ChatRecord(
        chat_id=f"chat-{uuid.uuid4().hex[:8]}",
        session_id=session_id,
        role="user",
        content=text,
    )
    db.messages.insert(chat)
    print(f"已写入消息：{chat.chat_id}（session={session_id}）")
    return chat.chat_id


async def cmd_run(chat_id: str) -> None:
    loader = SkillLoader()
    scanner = start_scanner(loader)
    wait_level1_loaded(loader)
    try:
        assistant = await process_message(chat_id, loader=loader)
    finally:
        scanner.stop(timeout=2)

    tasks = db.task_messages.list_tasks_for_message(chat_id)
    print("\n>>> 关联任务：")
    for t in tasks:
        print(f"- {t.task_id}  [{t.status}]  {t.title} -> {t.entry_skill_id}")
    print("\n>>> 最终回复：")
    print(assistant.content)


async def cmd_resume(task_id: str) -> None:
    try:
        await resume_one_task(task_id)
    except GraphInterrupt:
        pass
    await cmd_state(task_id)
    task = db.tasks.get(task_id)
    if task:
        print(f"任务状态：{task.status}")
        if task.output:
            print(f"任务产物：\n{task.output}")


async def cmd_state(task_id: str) -> None:
    print_memory(await show_task_memory(task_id))


def cmd_tasks(session_id: str | None) -> None:
    for t in db.tasks.list(session_id=session_id):
        print(f"{t.task_id:14} {t.status:10} {t.session_id:12} "
              f"{t.entry_skill_id:12} {t.title}")


def cmd_list() -> None:
    for m in db.messages.list_recent():
        print(f"{m.chat_id:14} {m.session_id:12} {m.role:9} {m.content[:60]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="意图路由 + 任务化技能执行")
    parser.add_argument(
        "command", nargs="?", default="demo",
        choices=["demo", "seed", "run", "pause", "resume", "state", "tasks", "list", "gen-queries"],
    )
    parser.add_argument("arg", nargs="?", help="chat_id / task_id / 聊天文本")
    parser.add_argument("--text", default=DEMO_TEXT, help="demo/seed 使用的文本")
    parser.add_argument("--session", default=DEFAULT_SESSION, help="会话 id")
    args = parser.parse_args()

    db.connect()
    db.init_schema()

    if args.command == "demo":
        chat_id = cmd_seed(args.text, args.session)
        asyncio.run(cmd_run(chat_id))
    elif args.command == "seed":
        cmd_seed(args.arg or args.text, args.session)
    elif args.command == "run":
        if not args.arg:
            raise SystemExit("run 需要提供 chat_id（可先 seed 或用 demo）")
        asyncio.run(cmd_run(args.arg))
    elif args.command == "pause":
        db.tasks.request_pause(args.arg)
        print(f"已请求暂停任务 {args.arg}")
    elif args.command == "resume":
        asyncio.run(cmd_resume(args.arg))
    elif args.command == "state":
        asyncio.run(cmd_state(args.arg))
    elif args.command == "tasks":
        cmd_tasks(args.session if args.session != DEFAULT_SESSION else None)
    elif args.command == "list":
        cmd_list()
    elif args.command == "gen-queries":
        from .skill_runtime.intent_sync import sync_all

        loader = SkillLoader()
        stats = sync_all(loader, generate=True)
        print("intent_math 造句完成：", stats)
    db.close()


if __name__ == "__main__":
    main()
