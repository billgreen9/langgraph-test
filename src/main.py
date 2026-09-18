"""主程序入口。

功能：
1. 启动后台守护线程，定时从 skills 目录加载【一级技能】到 PostgreSQL；
2. 从数据库读取一条用户聊天记录（pending）；
3. 按渐进式加载原则在多级 skills 目录中逐层匹配技能并执行：
   - category 逐层下探；atomic 直接 function_call；dynamic 动态规划、逐步执行；
4. 每一步都写入 LangGraph 在 PostgreSQL 中的 checkpoint 记忆
   （递归技能树、chat_id、status、level、skill_id），
   并可凭 chat_id 暂停 / 恢复该 langgraph。

用法：
    python -m src.main demo                      # 端到端演示（执行中自动暂停再恢复）
    python -m src.main seed "你好"               # 写入一条 pending 聊天记录
    python -m src.main run [chat_id]             # 从 DB 取聊天并运行
    python -m src.main pause <chat_id>           # 请求暂停运行中的 graph
    python -m src.main resume <chat_id>          # 凭 chat_id 恢复
    python -m src.main state <chat_id>           # 查看该聊天的 LangGraph 记忆
    python -m src.main list                      # 列出聊天记录
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import uuid

from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from .config import settings
from .db import ChatRecord, db
from .skill_runtime.graph import (
    Runtime,
    build_graph,
    get_memory,
)
from .skill_runtime.loader import SkillLoader
from .skill_runtime.scanner import SkillScanner
from .skill_runtime.state import S, collect_dynamic_skills, tree_to_view

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("main")

DEMO_TEXT = "我要去上海出差3天，帮我做出行天气规划，并给出中文简报"


# ---------- 工具 ----------
def start_scanner(db_, loader: SkillLoader) -> SkillScanner:
    scanner = SkillScanner(db_, loader, settings.skill_scan_interval)
    scanner.start()
    return scanner


def wait_level1_loaded(loader: SkillLoader, timeout: float = 5.0) -> None:
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if db.get_level1_skills():
            return
        time.sleep(0.1)
    # 超时也继续：graph 内有直接扫描兜底
    logger.warning("等待一级技能入库超时，继续执行")


def print_memory(mem: dict | None) -> None:
    if not mem:
        print("（无 LangGraph 记忆）")
        return
    tree = mem.get("tree")
    print("─" * 60)
    print(f"chat_id : {mem.get('chat_id')}")
    print(f"status  : {mem.get('status')}")
    print(f"level   : {mem.get('level')}")
    print(f"skill_id: {mem.get('skill_id')}")
    dynamic = collect_dynamic_skills(tree) if tree else []
    print(f"执行中的动态 skills（递归数组）: {[d.skill_id for d in dynamic]}")
    if tree:
        print("递归技能执行树：")
        print(tree_to_view(tree))
    print("─" * 60)


# ---------- 核心：运行 / 恢复 ----------
async def _consume(stream) -> None:
    try:
        async for _chunk in stream:
            pass
    except GraphInterrupt:
        pass


async def run_chat(chat_id: str | None = None, *, auto_pause_after: int = 0) -> None:
    """读取聊天记录并运行 graph；auto_pause_after>0 时在完成指定数量原子步骤后自动暂停。"""
    chat = db.get_chat(chat_id) if chat_id else db.get_pending_chat()
    if chat is None:
        print("数据库中没有待处理的聊天记录，可先用 `seed` 写入")
        return

    loader = SkillLoader()
    scanner = start_scanner(db, loader)
    wait_level1_loaded(loader)

    async with Runtime(loader=loader) as rt:
        app = build_graph(rt)
        config = {"configurable": {"thread_id": chat.chat_id, "runtime": rt}}
        await rt.mark_running(chat.chat_id)

        from langchain_core.messages import HumanMessage

        initial = {
            "messages": [HumanMessage(content=chat.content)],
            "chat_id": chat.chat_id,
            "user_input": chat.content,
            "status": S.RUNNING,
            "level": 0,
            "skill_id": "",
            "cursor": [],
        }

        # 在确定性的步骤边界触发暂停：第 N 个原子技能完成后写入暂停请求，
        # graph 会在下一个步骤边界（start_step/execute 的闸门）挂起
        pause_requested = False
        if auto_pause_after:
            leaves_done = 0

            async def after_leaf(st, node) -> None:
                nonlocal pause_requested, leaves_done
                leaves_done += 1
                if leaves_done >= auto_pause_after and not pause_requested:
                    pause_requested = True
                    await asyncio.to_thread(db.request_pause, chat.chat_id)
                    logger.info("已请求暂停 chat_id=%s（%s 完成后的步骤边界）",
                                chat.chat_id, node.skill_id)

            rt.on_leaf = after_leaf

        try:
            await _consume(app.astream(initial, config, stream_mode="updates"))
        except GraphInterrupt:
            pass

        mem = await get_memory(rt, app, chat.chat_id)
        row = db.get_chat(chat.chat_id)
        if pause_requested and row and row.status == S.PAUSED:
            print("\n>>> Graph 已在步骤边界暂停，当前 LangGraph 记忆：")
            print_memory(mem)

            print(">>> 凭 chat_id 恢复执行……")
            db.request_resume(chat.chat_id)
            try:
                stream2 = app.astream(
                    Command(resume={"resume": True}), config, stream_mode="updates"
                )
                await _consume(stream2)
            except GraphInterrupt:
                pass

        mem = await get_memory(rt, app, chat.chat_id)
        print("\n>>> 最终 LangGraph 记忆：")
        print_memory(mem)
        print("最终回答：\n" + (mem.get("final_answer") or "") if mem else "")
        print(f"\n数据库状态：{db.get_chat(chat.chat_id).status}")

    scanner.stop(timeout=2)


async def resume_only(chat_id: str) -> None:
    loader = SkillLoader()
    async with Runtime(loader=loader) as rt:
        app = build_graph(rt)
        db.request_resume(chat_id)
        config = {"configurable": {"thread_id": chat_id, "runtime": rt}}
        try:
            await _consume(app.astream(Command(resume={"resume": True}), config,
                                       stream_mode="updates"))
        except GraphInterrupt:
            pass
        print_memory(await get_memory(rt, app, chat_id))


async def show_state(chat_id: str) -> None:
    loader = SkillLoader()
    async with Runtime(loader=loader) as rt:
        app = build_graph(rt)
        print_memory(await get_memory(rt, app, chat_id))


# ---------- CLI ----------
def cmd_seed(text: str) -> str:
    chat_id = f"chat-{uuid.uuid4().hex[:8]}"
    db.upsert_chat(ChatRecord(chat_id=chat_id, content=text, status="pending"))
    print(f"已写入聊天记录：{chat_id}")
    return chat_id


def main() -> None:
    parser = argparse.ArgumentParser(description="多级技能渐进式加载 + LangGraph 执行")
    parser.add_argument("command", nargs="?", default="demo",
                        choices=["demo", "seed", "run", "pause", "resume", "state", "list"])
    parser.add_argument("arg", nargs="?", help="chat_id 或聊天文本")
    parser.add_argument("--text", default=DEMO_TEXT, help="demo/seed 使用的文本")
    args = parser.parse_args()

    db.connect()
    db.init_schema()

    if args.command == "demo":
        chat_id = cmd_seed(args.text)
        asyncio.run(run_chat(chat_id, auto_pause_after=1))
    elif args.command == "seed":
        cmd_seed(args.arg or args.text)
    elif args.command == "run":
        asyncio.run(run_chat(args.arg, auto_pause_after=0))
    elif args.command == "pause":
        db.request_pause(args.arg)
        print(f"已请求暂停 {args.arg}")
    elif args.command == "resume":
        asyncio.run(resume_only(args.arg))
    elif args.command == "state":
        asyncio.run(show_state(args.arg))
    elif args.command == "list":
        for c in db.list_chats():
            print(f"{c.chat_id:16} {c.status:10} {c.content[:50]}")
    db.close()


if __name__ == "__main__":
    main()
