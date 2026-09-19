# langgraph-test

基于 **LangGraph** 的「意图路由 → 任务化技能执行」系统：用户发送一条消息，RouterGraph 识别意图并拆成任务落库，TaskGraph 为每个任务独立执行一棵可递归、可并发、可暂停恢复的技能树，最后聚合成一条回复。

两张图**独立编译、通过 PostgreSQL 表解耦**；技能以「目录 + `skill.json`」方式声明，采用**渐进式加载**。

## 核心特性

- **双图解耦**：RouterGraph（消息→任务，`thread_id=chat_id`）与 TaskGraph（任务→执行，`thread_id=task_id`）只通过 `chat_task` 表交互，任务间状态隔离，可独立伸缩。
- **消息-任务多对多**：一条消息可拆多个任务、一个任务可归并多条消息（关联表 `chat_task_message`）。
- **四类技能编排**：`category`（容器下探）、`atomic`（原子函数调用）、`dynamic`（LLM 顺序规划多步骤）、`react`（LLM 增量规划 + 同批动作并发扇出）。
- **递归执行树**：技能步骤可任意深度嵌套，整棵树随 LangGraph checkpoint 持久化到 PostgreSQL，支持任务级暂停/恢复（含跨进程）。
- **渐进式技能加载**：后台线程只预取一级技能，深层技能执行到时按需加载并缓存。
- **可观测**：每次图调用带 `run_name` / `tags` / `metadata`，接入 LangSmith 即可追踪。

## 架构总览

```
┌──────────────────────────── runner.process_message（编排层）────────────────────────────┐
│                                                                                          │
│  chat_record ──▶ RouterGraph ──▶ chat_task(pending) ──▶ TaskGraph × N (gather 并发)       │
│   (user 消息)      消息→任务            ▲ 表解耦              任务→技能执行树               │
│                                                                                          │
│                         回读终态 → _synthesize 聚合 → chat_record(assistant)              │
└──────────────────────────────────────────┬───────────────────────────────────────────────┘
                                           │
              PostgreSQL：chat_record / chat_task / chat_task_message / skill_registry
              LangGraph checkpoint：AsyncPostgresSaver（thread_id = task_id）
```

### 一条消息的生命周期

1. `seed`：user 消息写入 `chat_record`。
2. `run` 调用 RouterGraph：意图识别 → 写 `chat_task(pending)` + 消息关联。
3. 编排层查出该消息的全部 pending 任务，`asyncio.gather` 各起一个 TaskGraph 执行。
4. TaskGraph 沿技能树下探/规划/ReAct 并发执行，`finish` 节点回写任务终态（completed/failed + output/error）。
5. 回读终态 → `_synthesize` 聚合（单任务直出；多任务 LLM 综合，失败降级拼接）→ 写 assistant 消息并关联回任务。

## RouterGraph：消息 → 任务

```
START → load_message → route_intents → (create | attach) → persist → END
```

| 节点 | 职责 |
|---|---|
| `load_message` | 读取消息内容与会话信息载入 state |
| `route_intents` | 意图识别（关键词打分 → 0 命中才 LLM 裁决 → `chat` 兜底）+ 判定 create/attach |
| `create` / `attach` | 只产出任务 id，不写库 |
| `persist` | **唯一写库点**：create 落新任务并关联消息；attach 幂等补关联 |

分支判定：消息已关联非终态任务（collecting/pending/running/paused）→ `attach`（P1 作为幂等重跑护栏，P2 扩展为跨消息归并）；否则 `create`。

意图定义为 `skills/intents/*.md`（YAML front matter：`intent_id / name / keywords / entry_skill / order` + Markdown 说明正文），入口技能必须命中一级业务技能。

## TaskGraph：任务 → 技能执行树

四类技能由 `NEXT_BY_TYPE` 映射到不同入口节点：

| 类型 | 行为 | 入口节点 |
|---|---|---|
| `category` | 容器，按关键词在子技能中选 1 个继续下探 | `descend` |
| `atomic` | 叶子，调用 `FUNCTIONS` 注册表中的一个 Python 函数 | `execute` |
| `dynamic` | LLM 一次性规划为有序步骤逐步执行，步骤可再嵌套 | `plan → start_step` |
| `react` | 每轮 LLM 增量规划一批**可并发**原子动作，观察后再决策 | `react_think → Send 扇出 → parallel_execute → react_join` |

```
START → enter_task ─┬─ atomic ───▶ execute ───────────────┐
                    ├─ category ─▶ descend（再分流）        │
                    ├─ dynamic ──▶ plan → start_step        ├─▶ complete ─┐
                    └─ react ────▶ react_think              │             │
                                      ├─ done ─────────────▶ complete     │
                                      └─ Send×N → parallel_execute        │
                                                   → react_join → 下一轮  │
                  execute / 步骤完成 → complete ─还有步骤? start_step┘     │
                                                       └─收束─▶ finish → END
```

### 递归执行树

- 状态中的 `tree` 是 `SkillExecutionNode` 递归结构，`cursor`（索引路径）定位当前节点；category 有 1 个选中子节点，dynamic/react 可有多个，形成任意深度递归。
- dynamic 节点用 `plan`（规划蓝图）+ `step_index`（进度指针）+ `steps`（已实例化的执行轨迹，与 plan 下标对齐）三件套支持步骤推进、暂停恢复与嵌套。
- 所有树改动走 `model_copy(deep=True)` 纯函数式更新，保证 checkpoint 可追踪。

### ReAct 并发

- `react_think` 输出结构化 `ReactDecision`（`done / final_answer / actions[]`），`_after_react_think` 用 LangGraph `Send` 把同批动作扇出。
- `parallel_execute` 用 `asyncio.to_thread` 跑同步函数；各分支只写 `branch_results`（`operator.add` reducer），`react_join` 屏障合入树。
- 护栏：`max_rounds` 轮次上限、动作校验与指纹去重、失败重试上限；`parallelizable=false` 或 `self_exclusive=true` 的原子技能会被机械拆为单独一波。

### 暂停 / 恢复

任务粒度。`pause_requested` 闸门位于 `start_step` / `execute` / `react_think`（react 在轮次边界），interrupt 在节点变更 tree 之前调用，恢复时整节重放不重复计数；`mark_running` 会清除暂停位，因此暂停只在下一个步骤边界生效。

## 技能体系

### 声明方式

每个技能是一个含 `skill.json` 的目录，`skill_id` 由目录路径推导：

```
skills/weather/trip_react/r_current/skill.json  →  skill_id = weather.trip_react.r_current
```

`skill.json` 示例（react 类型）：

```json
{
  "name": "出行 ReAct 智能规划",
  "type": "react",
  "keywords": ["react", "智能", "并发", "智能规划"],
  "order": 4,
  "react": { "max_rounds": 4, "objective_hint": "天气与预报互不依赖，应同批并发..." }
}
```

### 函数注册（不使用 OpenAI tools 协议）

LLM 只输出 Pydantic 结构化 JSON（skill_id + arguments），执行层按名查表调用 Python 函数：

```python
FUNCTIONS: dict[str, dict] = {}

@register_function("calc")
def calc(text: str = "", expr: str | None = None, **_) -> str:
    ...
```

- atomic 技能在 `skill.json` 的 `"function"` 字段绑定函数名。
- 所有函数统一签名：`fn(text: str, **arguments) -> str`，`text` 为用户原话或上一步产物。
- 已注册函数：`get_weather`、`get_weather_forecast`、`translate_text`、`calc`、`chat_reply`。

### 渐进式加载

- `SkillScanner` 守护线程默认每 10 秒扫描 `skills/` 根目录，只把一级技能 upsert 进 `skill_registry`（`prefetched=true`），并清理已删除目录（支持热增删）。
- `SkillLoader.require(skill_id)` 在执行命中时才加载深层技能，并以 `prefetched=false` 顺手缓存入库。
- `skills/intents/` 下是意图 `.md`，不含 `skill.json`，不会被技能扫描器收录。

## 数据模型（PostgreSQL）

数据访问层位于 `src/db/`，按表一个仓储模块；`Database` 只管理连接池与建表，通过属性暴露各仓储：

| 表 | 仓储入口 | 说明 |
|---|---|---|
| `chat_record` | `db.messages` | 纯聊天时间线（chat_id / session_id / role / content），不含执行态 |
| `chat_task` | `db.tasks` | 任务：status（collecting/pending/running/paused/completed/failed）、entry_skill_id、arguments、output/error、暂停控制 |
| `chat_task_message` | `db.task_messages` | 任务↔消息多对多关联（UNIQUE task_id+chat_id） |
| `skill_registry` | `db.skills` | 技能注册表（一级预取 + 深层渐进缓存） |

包内文件：`models.py`（表模型）、`schema.py`（DDL）、`base.py`（仓储基类）、`chat_record.py` / `chat_task.py` / `chat_task_message.py` / `skill_registry.py`（四表仓储）、`database.py`（组合根 + 单例 `db`）。

首次启动自动建表；检测到旧版单体 `chat_records` 表（含 status 列）会自动 DROP 重建（旧数据不迁移）。

## 目录结构

```
langgraph-test/
├── src/
│   ├── main.py                  # CLI 入口：demo/seed/run/pause/resume/state/tasks/list
│   ├── runner.py                # 编排层：router → 任务并发执行 → 聚合写 assistant
│   ├── config.py                # 环境变量配置（LLM / PG DSN / 连接池 / 扫描间隔）
│   ├── db/                      # 数据访问层（按表拆分的仓储包，见上）
│   └── skill_runtime/
│       ├── router_graph.py      # RouterGraph：load_message→route_intents→create/attach→persist
│       ├── graph.py             # TaskGraph：技能执行引擎 + Runtime + start/resume/get_memory
│       ├── state.py             # AgentState、递归执行树 SkillExecutionNode/PlanStep
│       ├── schema.py            # SkillManifest（skill.json 模型，四类技能）
│       ├── loader.py            # 技能加载器（一级扫描 + 深层按需 require）
│       ├── scanner.py           # 后台守护线程 SkillScanner
│       ├── intent_loader.py     # skills/intents/*.md 解析
│       └── functions.py         # @register_function 函数注册表
├── skills/
│   ├── intents/                 # 意图定义（Markdown + YAML front matter）
│   ├── weather/  calc/  chat/  translate/   # 业务技能树（每目录一个 skill.json）
└── tests/                       # pytest（真实 PG 集成测试 + ReAct 单测，14 例）
```

> `src/graph.py`、`src/IntentRouter.py` 为早期单体 demo 遗留，当前架构未使用。

## 快速开始

### 1. 准备 PostgreSQL

```bash
open -a Docker
docker start pgvector-demo        # pgvector/pgvector:pg16，映射 5432
```

默认连接串：`postgresql://postgres:123456@localhost:5432/graph-test`
（可用环境变量 `POSTGRES_DSN` 覆盖）。

### 2. 安装依赖

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

在项目根目录准备 `.env`，至少包含 LLM 配置：

```bash
OPENAI_API_KEY=<your-api-key>
OPENAI_BASE_URL=<llm-base-url>      # 如豆包 ark endpoint
MODEL_NAME=<model-name>
```

### 3. 运行

必须以模块方式运行（包内为相对导入）：

```bash
.venv/bin/python -m src.main demo
```

## CLI 命令

统一入口：`python -m src.main <command> [arg] [--text ...] [--session ...]`

| 命令 | 作用 |
|---|---|
| `demo`（默认） | 一键演示：写入默认消息 → 路由 → 执行 → 打印回复 |
| `seed "<文本>"` | 只写入一条 user 消息并打印 chat_id（不执行） |
| `run <chat_id>` | 对已有消息执行完整链路 |
| `pause <task_id>` | 请求暂停任务（下一个步骤边界挂起） |
| `resume <task_id>` | 从 checkpoint 恢复任务继续执行 |
| `state <task_id>` | 查看任务的 checkpoint 记忆快照与递归执行树 |
| `tasks` | 列出任务（可加 `--session <id>` 过滤） |
| `list` | 列出最近的聊天消息 |

全局选项：`--text "..."`（自定义 demo/seed 文本，默认为上海天气 ReAct 简报）、`--session <id>`（会话 id，默认 `default`）。

示例：

```bash
# 自定义文本的完整演示
python -m src.main demo --text "把 hello 翻译成中文" --session demo1

# 分步：先 seed 取 chat_id，再 run
CID=$(python -m src.main seed "计算 (2+3)*4" | grep -o 'chat-[a-f0-9]*')
python -m src.main run "$CID"

# 纯关键词路由（跳过 LLM 意图裁决，便于离线调试）
SKILL_FORCE_KEYWORD=1 python -m src.main demo --text "计算 1+1"

# 暂停 / 恢复 / 查看状态（支持跨进程）
python -m src.main pause  task-xxxxxxxx
python -m src.main resume task-xxxxxxxx
python -m src.main state  task-xxxxxxxx
```

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `POSTGRES_DSN` | `postgresql://postgres:123456@localhost:5432/graph-test` | PostgreSQL 连接串 |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `MODEL_NAME` | — | LLM 配置 |
| `TEMPERATURE` | `0.7` | LLM 温度 |
| `SKILLS_DIR` | `<项目根>/skills` | 技能目录根路径 |
| `SKILL_SCAN_INTERVAL` | `10` | 一级技能后台扫描间隔（秒） |
| `DB_POOL_MIN_SIZE` / `DB_POOL_MAX_SIZE` | `1` / `10` | 连接池大小 |
| `SKILL_FORCE_KEYWORD` | 未设置 | 设为 `1` 时全程走关键词兜底，不调用 LLM |
| `LANGSMITH_TRACING` 等 | — | LangSmith 标准追踪环境变量，设置后自动上报 |

## 测试

```bash
.venv/bin/python -m pytest tests/ -q
```

集成测试直连本地 PostgreSQL（库/表会自动初始化）；`asyncio_mode=auto`。

## 可观测性

所有图调用统一通过 `run_config(rt, thread_id, run_name=..., tags=..., metadata=...)` 构造配置：

- RouterGraph：`run_name=router_{chat_id}`，tag `router`
- TaskGraph 启动：`run_name=task_{task_id}_{标题}`，tags `task / skill:{入口技能} / session:{会话}`
- TaskGraph 恢复：`run_name=resume_{task_id}`，metadata `phase=resume`
- 结果综合 LLM：`run_name=synthesize_final_answer`，tag `aggregate`

配置 `LANGSMITH_TRACING=true` 与 API key 后即可在 LangSmith 按上述名称/标签筛选每次运行。

## 路线图

- **P1（已完成）**：双图解耦、单消息→单任务、任务级暂停恢复、多任务并发与聚合框架。
- **P2**：一条消息多意图拆分（1→N 任务）；`collecting` 状态支持跨消息任务归并。
- **P3**：chat 级广播暂停/恢复；跨任务的聚合记忆视图。
