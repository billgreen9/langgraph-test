"""技能系统测试。

- 纯逻辑单测：加载器、原子函数、递归执行树、关键词匹配与兜底规划（无需 DB/LLM）
- 集成测试：PostgreSQL 注册表 + LangGraph checkpoint 全链路 + 暂停/恢复
  （本机 PostgreSQL 不可用时自动 skip）
"""

from __future__ import annotations

import uuid

import pytest

from src.skill_runtime import graph as graph_mod
from src.skill_runtime.functions import FUNCTIONS
from src.skill_runtime.loader import SkillLoader
from src.skill_runtime.state import (
    S,
    SkillExecutionNode,
    append_child,
    collect_dynamic_skills,
    get_node,
)


# ---------- 纯逻辑单测 ----------
def test_loader_scan_level1_and_progressive():
    loader = SkillLoader()
    l1 = loader.scan_level1()
    ids = {m.skill_id for m in l1}
    assert ids == {"weather", "translate", "calc", "chat"}
    assert all(m.level == 1 for m in l1)

    weather = next(m for m in l1 if m.skill_id == "weather")
    assert weather.type == "category" and weather.has_children

    # 渐进式：此时尚未加载二级
    assert "weather.trip_plan" not in loader.cached_ids()

    children = loader.load_children(weather)
    child_ids = {m.skill_id for m in children}
    assert child_ids == {"weather.current", "weather.forecast", "weather.trip_plan"}

    trip_plan = loader.require("weather.trip_plan")
    assert trip_plan.type == "dynamic"
    # 更深一级：嵌套动态技能
    deeper = loader.load_children(trip_plan)
    assert "weather.trip_plan.briefing" in {m.skill_id for m in deeper}
    briefing = loader.require("weather.trip_plan.briefing")
    leaves = loader.load_children(briefing)
    assert {m.skill_id for m in leaves} == {
        "weather.trip_plan.briefing.translate_brief",
        "weather.trip_plan.briefing.chat_brief",
    }


def test_atomic_functions():
    assert "上海" in FUNCTIONS["get_weather"]("去上海出差")
    assert "3天" in FUNCTIONS["get_weather_forecast"]("出差3天")
    assert "= 12" in FUNCTIONS["calc"]("(1+2)*4")
    assert FUNCTIONS["translate_text"]("把 hello 翻译成中文") == "你好"
    assert FUNCTIONS["chat_reply"]("讲个笑话")  # 兜底原子函数有输出即可


def test_recursive_execution_tree():
    root = SkillExecutionNode(skill_id="root", node_type="dynamic", level=1)
    tree, p1 = append_child(
        root, [], SkillExecutionNode(skill_id="step-atomic", node_type="atomic", level=2)
    )
    tree, p2 = append_child(
        tree, [], SkillExecutionNode(skill_id="step-dynamic", node_type="dynamic", level=2)
    )
    tree, p3 = append_child(
        tree, p2, SkillExecutionNode(skill_id="nested-atomic", node_type="atomic", level=3)
    )

    assert p1 == [0] and p2 == [1] and p3 == [1, 0]
    assert get_node(tree, p3).skill_id == "nested-atomic"
    # 递归数组中能收集到所有动态技能（含嵌套）
    assert [d.skill_id for d in collect_dynamic_skills(tree)] == ["root", "step-dynamic"]


def test_resolve_input_walks_up_for_nested_dynamic():
    """嵌套动态技能第一步的 prev_output 应回溯到父动态节点的前序步骤产物。"""
    # root(category weather) -> [0]=trip_plan(dynamic)
    #   -> [0,0] weather_check(atomic, 已完成, 有输出)
    #   -> [0,1] forecast_check(atomic, 已完成, 有输出)
    #   -> [0,2] briefing(dynamic)
    #       -> [0,2,0] translate_brief(atomic, prev_output)
    root = SkillExecutionNode(skill_id="weather", node_type="category", level=1)
    tree, p_dyn = append_child(
        root, [], SkillExecutionNode(skill_id="trip_plan", node_type="dynamic", level=2)
    )

    def completed_atomic(skill_id: str, output: str, level: int):
        n = SkillExecutionNode(skill_id=skill_id, node_type="atomic", level=level)
        n.status = S.COMPLETED
        n.output = output
        return n

    tree, _ = append_child(tree, p_dyn, completed_atomic("weather_check", "实时天气OK", 3))
    tree, _ = append_child(tree, p_dyn, completed_atomic("forecast_check", "预报OK", 3))
    tree, p_brief = append_child(
        tree, p_dyn, SkillExecutionNode(skill_id="briefing", node_type="dynamic", level=3)
    )
    translate = SkillExecutionNode(
        skill_id="translate_brief", node_type="atomic", level=4, input_from="prev_output"
    )
    tree, p_tr = append_child(tree, p_brief, translate)

    text = graph_mod._resolve_input(tree, p_tr, "用户原始输入")
    assert text == "预报OK"  # 取到的是喂给嵌套动态技能的上一步产物
    # 普通 user 输入模式不受影响
    user_node = SkillExecutionNode(
        skill_id="weather_check2", node_type="atomic", level=3, input_from="user"
    )
    tree, p_user = append_child(tree, p_dyn, user_node)
    assert graph_mod._resolve_input(tree, p_user, "用户原始输入") == "用户原始输入"


def test_keyword_match_and_fallback_plan():
    loader = SkillLoader()
    weather = loader.require("weather")
    children = loader.load_children(weather)

    # 关键词直接命中动态技能
    chosen = graph_mod.match_skill(children, "帮我做出行天气规划")
    assert chosen.skill_id == "weather.trip_plan"

    # 兜底规划：核心步骤按 order 在前，总结类（嵌套动态 briefing）置后
    trip_plan = loader.require("weather.trip_plan")
    grandchildren = loader.load_children(trip_plan)
    plan = graph_mod._fallback_plan(
        grandchildren, "我要去上海出差3天，做出行规划", max_steps=4
    )
    assert [s.skill_id for s in plan] == [
        "weather.trip_plan.weather_check",
        "weather.trip_plan.forecast_check",
        "weather.trip_plan.briefing",
    ]
    assert plan[0].input_from == "user"
    assert plan[-1].input_from == "prev_output"


# ---------- PostgreSQL 集成测试 ----------
def _pg_available() -> bool:
    import psycopg

    from src.config import settings

    try:
        with psycopg.connect(settings.pg_dsn, connect_timeout=2) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:  # noqa: BLE001 - 探测性连接，任何失败都视为 PG 不可用
        return False


pg_required = pytest.mark.skipif(not _pg_available(), reason="本机 PostgreSQL 不可用")


def _force_keyword_paths(monkeypatch) -> None:
    """让路由/规划走纯关键词兜底路径，测试不依赖外部 LLM。"""
    monkeypatch.setattr(
        graph_mod, "match_skill",
        lambda manifests, text, rt=None: graph_mod._fallback_choice(manifests, text),
    )
    monkeypatch.setattr(
        graph_mod, "build_plan",
        lambda rt, dynamic, children, text: graph_mod._fallback_plan(
            children, text, dynamic.planner.max_steps if dynamic.planner else 5
        ),
    )


async def _run(rt, app, chat_id, content, command=None):
    from langchain_core.messages import HumanMessage

    config = {"configurable": {"thread_id": chat_id, "runtime": rt}}
    await rt.mark_running(chat_id)
    initial = {
        "messages": [HumanMessage(content=content)],
        "chat_id": chat_id,
        "user_input": content,
        "status": S.RUNNING,
        "level": 0,
        "skill_id": "",
        "cursor": [],
    }
    cmd = command if command is not None else initial
    chunks = []
    async for chunk in app.astream(cmd, config, stream_mode="updates"):
        chunks.append(chunk)
    return chunks, config


@pg_required
async def test_graph_dynamic_nested_and_checkpoint(monkeypatch):
    from src.db import ChatRecord, db
    from src.skill_runtime.graph import Runtime, build_graph, get_memory
    from src.skill_runtime.scanner import SkillScanner

    _force_keyword_paths(monkeypatch)
    db.connect()
    db.init_schema()

    loader = SkillLoader()
    scanner = SkillScanner(db, loader, interval_seconds=60, run_immediately=False)
    scanner.scan_once()  # 一级技能入库

    # 后台线程只预取一级技能：prefetched=TRUE 的记录必须全部是 level=1
    with db.pool.connection() as conn:
        row = conn.execute(
            "SELECT count(*) AS n FROM skill_registry "
            "WHERE prefetched = TRUE AND level > 1"
        ).fetchone()
        deep_prefetched = row["n"]
    assert deep_prefetched == 0

    chat_id = f"it-{uuid.uuid4().hex[:8]}"
    db.upsert_chat(ChatRecord(
        chat_id=chat_id,
        content="我要去上海出差3天，帮我做出行天气规划，并给出中文简报",
        status="pending",
    ))

    async with Runtime(loader=loader) as rt:
        app = build_graph(rt)
        await _run(rt, app, chat_id, db.get_chat(chat_id).content)
        mem = await get_memory(rt, app, chat_id)

        assert mem["status"] == S.COMPLETED
        assert mem["tree"].skill_id == "weather"
        assert mem["tree"].status == S.COMPLETED  # 根 category 节点被收束
        dynamic_ids = [d.skill_id for d in collect_dynamic_skills(mem["tree"])]
        assert "weather.trip_plan" in dynamic_ids
        assert "weather.trip_plan.briefing" in dynamic_ids  # 嵌套动态技能

        outputs = mem["final_answer"]
        assert "上海" in outputs and "预报" in outputs
        # 嵌套动态技能的翻译步骤消费的是“预报产物”，而不是用户原始输入
        briefing = next(
            d for d in collect_dynamic_skills(mem["tree"])
            if d.skill_id == "weather.trip_plan.briefing"
        )
        translate_leaf = next(
            n for n in briefing.steps
            if n.skill_id.endswith("translate_brief")
        )
        assert "预报" in (translate_leaf.output or "")

        # 深层技能在执行过程中被渐进式缓存进 DB
        assert db.get_skill("weather.trip_plan") is not None
        assert db.get_skill("weather.trip_plan.briefing") is not None

    row = db.get_chat(chat_id)
    assert row.status == "completed" and row.response


@pg_required
async def test_graph_pause_and_resume_by_chat_id(monkeypatch):
    from langgraph.types import Command

    from src.db import ChatRecord, db
    from src.skill_runtime.graph import Runtime, build_graph, get_memory

    _force_keyword_paths(monkeypatch)
    db.connect()
    db.init_schema()
    loader = SkillLoader()

    chat_id = f"it-{uuid.uuid4().hex[:8]}"
    content = "我要去上海出差3天，帮我做出行天气规划，并给出中文简报"
    db.upsert_chat(ChatRecord(chat_id=chat_id, content=content, status="pending"))

    async with Runtime(loader=loader) as rt:
        app = build_graph(rt)
        config = {"configurable": {"thread_id": chat_id, "runtime": rt}}

        # 启动前即请求暂停：应在第一个步骤边界挂起
        db.request_pause(chat_id)
        from langchain_core.messages import HumanMessage

        await rt.mark_running(chat_id)
        # mark_running 会清暂停标记，这里重新置位模拟“运行中被请求暂停”
        db.request_pause(chat_id)
        initial = {
            "messages": [HumanMessage(content=content)],
            "chat_id": chat_id, "user_input": content,
            "status": S.RUNNING, "level": 0, "skill_id": "", "cursor": [],
        }
        async for _ in app.astream(initial, config, stream_mode="updates"):
            pass

        paused = await get_memory(rt, app, chat_id)
        # 已路由到 weather -> trip_plan 动态技能，但步骤未执行
        assert paused["tree"].skill_id == "weather"
        assert db.get_chat(chat_id).status == "paused"

        # 凭 chat_id 恢复，跑到完成
        db.request_resume(chat_id)
        async for _ in app.astream(
            Command(resume={"resume": True}), config, stream_mode="updates"
        ):
            pass
        final = await get_memory(rt, app, chat_id)
        assert final["status"] == S.COMPLETED
        assert "上海" in final["final_answer"]

    assert db.get_chat(chat_id).status == "completed"


@pg_required
async def test_graph_atomic_level1_and_level2(monkeypatch):
    from src.db import ChatRecord, db
    from src.skill_runtime.graph import Runtime, build_graph, get_memory

    _force_keyword_paths(monkeypatch)
    db.connect()
    db.init_schema()
    loader = SkillLoader()

    async with Runtime(loader=loader) as rt:
        app = build_graph(rt)

        # 一级原子技能（function_call）
        cid1 = f"it-{uuid.uuid4().hex[:8]}"
        db.upsert_chat(ChatRecord(chat_id=cid1, content="讲个笑话", status="pending"))
        await _run(rt, app, cid1, "讲个笑话")
        mem1 = await get_memory(rt, app, cid1)
        assert mem1["status"] == S.COMPLETED
        assert mem1["tree"].node_type == "atomic"
        assert mem1["tree"].function == "chat_reply"

        # category 下探到二级原子技能
        cid2 = f"it-{uuid.uuid4().hex[:8]}"
        db.upsert_chat(ChatRecord(chat_id=cid2, content="把 hello 翻译成中文", status="pending"))
        await _run(rt, app, cid2, "把 hello 翻译成中文")
        mem2 = await get_memory(rt, app, cid2)
        assert mem2["tree"].skill_id == "translate"
        leaf = mem2["tree"].steps[0]
        assert leaf.skill_id == "translate.text" and leaf.status == S.COMPLETED
        assert leaf.output == "你好"
