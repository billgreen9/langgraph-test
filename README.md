# 项目

## 实现概览

### 1. 后台线程定时加载一级技能
- [scanner.py](file:///Users/bill/PycharmProjects/langgraph-test/src/skill_runtime/scanner.py)：守护线程 `SkillScanner`，默认每 10 秒扫描一次 `skills/` 根目录，**只把一级技能** upsert 进 PostgreSQL 的 `skill_registry` 表（`prefetched=TRUE`），并自动清理已删除目录（已实测热增删生效）。
- [loader.py](file:///Users/bill/PycharmProjects/langgraph-test/src/skill_runtime/loader.py)：`scan_level1()` 只加载一层；`load_children()` 在运行时命中某技能后才展开**下一层**，实现渐进式加载，深层技能顺手以 `prefetched=FALSE` 缓存入库。

### 2. 三类技能（skills 多级目录）
每个技能是一个含 `skill.json` 的目录，`skill_id` 由路径推导（如 `weather.trip_plan.briefing`）：
- **category**：容器型，逐层下探匹配（`weather` → `trip_plan`）
- **dynamic**：LLM 动态规划出有序步骤逐步执行，步骤可以再嵌套 dynamic（已实现 `trip_plan` → 嵌套 `briefing` 的 4 层深度示例）
- **atomic**：原子 function_call，绑定 [functions.py](file:///Users/bill/PycharmProjects/langgraph-test/src/skill_runtime/functions.py) 注册表中的 Python 函数（`get_weather`/`calc`/`translate_text`/`chat_reply`）；`chat` 本身就是一级原子技能

### 3. LangGraph 记忆（PostgreSQL checkpoint）
[state.py](file:///Users/bill/PycharmProjects/langgraph-test/src/skill_runtime/state.py) 中的记忆包含你要求的全部字段：
- **递归数组** `tree.steps`：记录执行中的所有动态技能及其子步骤，任意深度嵌套
- `chat_id`：即 LangGraph `thread_id`
- `status`：running/planning/paused/completed/failed
- `level`：当前执行深度（已实测挂起时 level=2，最深执行到 level=4）
- `skill_id`：当前技能
- 每个节点（步骤）切换都经 `AsyncPostgresSaver` 写入 Postgres；步骤产物支持 `prev_output` 跨层传递（嵌套动态技能自动消费父级上一步产物）

### 4. 图与暂停/恢复
[graph.py](file:///Users/bill/PycharmProjects/langgraph-test/src/skill_runtime/graph.py)：`route → descend → plan → start_step → execute → complete → finish`，条件边驱动递归；暂停通过在步骤边界检查 DB 标志位并调用 `interrupt()` 挂起，凭 chat_id 用 `Command(resume=...)` 从 checkpoint 恢复。

### 5. 数据库与主程序
- [db.py](file:///Users/bill/PycharmProjects/langgraph-test/src/db.py)：psycopg3 连接池、`chat_records`（聊天记录+暂停标志+终态）、`skill_registry` 两张表
- [main.py](file:///Users/bill/PycharmProjects/langgraph-test/src/main.py)：启动后台线程 → 从 DB 读取 pending 聊天 → 执行。CLI：
  - `python -m src.main demo` 端到端演示（执行 1 步后自动暂停、打印记忆、再恢复跑完）
  - `seed/run/pause/resume/state/list`，已验证**跨进程** `pause` → checkpoint 挂起 → 另一进程 `resume` 跑完

### 验证结果
- 9 个 pytest 用例全部通过（4 个纯逻辑 + 3 个 PG 集成，含暂停恢复、嵌套动态、一级/二级原子）
- 真实 LLM（豆包）链路：规划出 3 步、嵌套简报规划、最终回答正确写回 DB
- `ruff check` 与 IDE 诊断均无问题
- 依赖已加入 pyproject：`psycopg[binary]`、`langgraph-checkpoint-postgres`；DSN 默认 `postgresql://postgres:123456@localhost:5432/graph-test`（复用了你已有的 pgvector-demo 容器，库名/密码正好匹配）
- 离线快速验证可用 `SKILL_FORCE_KEYWORD=1`（跳过 LLM 走关键词兜底）

# 七节点流程说明

整个流程在 [graph.py](file:///Users/bill/PycharmProjects/langgraph-test/src/skill_runtime/graph.py) 中实现，是一棵递归执行树的遍历过程，由条件边根据节点 `node_type`（atomic/category/dynamic）分流。

## 节点职责

### 1. [route_node](file:///Users/bill/PycharmProjects/langgraph-test/src/skill_runtime/graph.py#L407-L415) — 一级路由
- 从 DB 的 `skill_registry` 读取**后台线程预加载的一级技能**（渐进式加载第一层）
- 用 `match_skill()`（关键词打分 → LLM 兜底）匹配用户输入
- 创建根 `SkillExecutionNode`，cursor 置为根
- **入口节点**，由 `START` 直连

### 2. [descend_node](file:///Users/bill/PycharmProjects/langgraph-test/src/skill_runtime/graph.py#L418-L438) — category 下探
- 仅 **category 容器型节点**进入
- 调用 `loader.load_children()` **渐进式加载**下一层子技能（运行时按需展开，非预加载）
- 顺手 `persist_children()` 把深层技能缓存到 DB（`prefetched=FALSE`）
- 在子技能中再匹配，append 为子节点，cursor 下移
- 可多层嵌套（条件边自循环回 descend）

### 3. [plan_node](file:///Users/bill/PycharmProjects/langgraph-test/src/skill_runtime/graph.py#L441-L470) — dynamic 动态规划
- 仅 **dynamic 节点**进入
- 加载候选子技能，调用 [build_plan()](file:///Users/bill/PycharmProjects/langgraph-test/src/skill_runtime/graph.py#L289-L335)：LLM 拆步骤（失败走 `_fallback_plan` 关键词兜底）
- 限定 skill_id 必须在候选集合内，去重，最多 `planner.max_steps` 步
- 把 `plan` 和 `step_index=0` 写入节点记忆
- 直连 `start_step`

### 4. [start_step_node](file:///Users/bill/PycharmProjects/langgraph-test/src/skill_runtime/graph.py#L473-L500) — 步骤实例化
- 从 dynamic 的 `plan[step_index]` 取下一步
- **调用 `_pause_gate()`**：检查 DB 中 `pause_requested`，若为 TRUE 则 `interrupt()` 挂起 graph（这是暂停的确定性触发点）
- 实例化对应子技能节点（可能是 atomic / category / dynamic，**支持再嵌套 dynamic**）
- 标记该 PlanStep 为 RUNNING，cursor 下移到子节点

### 5. [execute_node](file:///Users/bill/PycharmProjects/langgraph-test/src/skill_runtime/graph.py#L503-L550) — atomic 执行
- 仅 **atomic 节点**进入
- 再次过 `_pause_gate()` 暂停闸门
- 从 `FUNCTIONS` 注册表查函数，用 `_resolve_input()` 决定输入（user 或 prev_output，支持跨层向上回溯）
- 执行 function_call，捕获异常 → 节点置 FAILED；成功 → 置 COMPLETED 并写 output
- 触发 `rt.on_leaf` 钩子（主程序借此在原子步骤边界写暂停请求）
- 直连 `complete`

### 6. [complete_node](file:///Users/bill/PycharmProjects/langgraph-test/src/skill_runtime/graph.py#L553-L590) — 收束与上推
- 把当前节点 status 置 COMPLETED
- 若是根节点 → 直接返回终态
- 若父节点是 dynamic → 推进 `step_index`，上推 cursor 到父
- 若父节点是根 category → 顺手把根也置 COMPLETED（根节点特殊收束，避免 status 卡在 running）
- 由条件边 [_after_complete](file:///Users/bill/PycharmProjects/langgraph-test/src/skill_runtime/graph.py#L624-L635) 决定下一步：
  - FAILED → finish
  - 父是 category → 继续自环 complete
  - 父 dynamic 还有剩余步骤 → start_step
  - 父 dynamic 步骤耗尽 → 自环 complete 收束自身

### 7. [finish_node](file:///Users/bill/PycharmProjects/langgraph-test/src/skill_runtime/graph.py#L598-L615) — 汇总与落库
- `collect_leaf_outputs()` 收集所有原子节点输出
- 调用 `rt.mark_terminal()` 把最终答案 + status（COMPLETED/FAILED）写回 `chat_records` 表
- 打印整棵执行树视图 `tree_to_view()`
- 返回 `final_answer` + `AIMessage`，连 END

## 一图概括

```
START→route ──┬─→ execute  (atomic 命中)
              ├─→ descend ─→ ... (category 下探)
              └─→ plan ──→ start_step ─┬─→ execute (atomic)
                                       ├─→ descend (嵌套 category)
                                       └─→ plan    (嵌套 dynamic，递归)
execute → complete ─┬─→ start_step (dynamic 还有步骤)
                    ├─→ complete   (继续上推收束)
                    └─→ finish → END
```

核心机制：**条件边按 `node_type` 分流**，**complete 自环实现递归上推**，**`_pause_gate` 在 start_step 与 execute 两处检查暂停标志**实现跨进程暂停/恢复。


# skills中三种节点类型的作用

类型在 [skill.json](file:///Users/bill/PycharmProjects/langgraph-test/skills/weather/skill.json) 的 `type` 字段声明，由 [NEXT_BY_TYPE](file:///Users/bill/PycharmProjects/langgraph-test/src/skill_runtime/graph.py#L57) 映射决定进入哪个图节点：

```python
NEXT_BY_TYPE = {"atomic": "execute", "category": "descend", "dynamic": "plan"}
```

## 1. atomic — 原子 function_call
**叶子节点，真正干活的**。
- 进入 [execute_node](file:///Users/bill/PycharmProjects/langgraph-test/src/skill_runtime/graph.py#L503)，调用 `FUNCTIONS[function]` 执行一次函数调用
- 产生 `output`，是执行树唯一产出实际结果的地方
- 失败置 FAILED，成功置 COMPLETED
- 例：`weather.current`（get_weather）、`calc.math`（calc）、`chat`（chat_reply 兜底闲聊）

## 2. category — 容器/目录型
**组织用，自身不执行任何逻辑，只做下探**。
- 进入 [descend_node](file:///Users/bill/PycharmProjects/langgraph-test/src/skill_runtime/graph.py#L418)，加载子技能并匹配一个继续向下
- 不产生 output，最终输出靠 `collect_leaf_outputs` 收集子树所有原子结果
- 作用：**构建多级目录树**，让技能按领域分组（weather → current/forecast/trip_plan）
- 可多层嵌套（category 套 category 也行）
- 例：`weather`（一级 category，order=1）、`translate`、`calc`

## 3. dynamic — 动态规划型
**运行时按用户意图拆步骤、按顺序编排多个子技能**。
- 进入 [plan_node](file:///Users/bill/PycharmProjects/langgraph-test/src/skill_runtime/graph.py#L441)，用 LLM（或关键词兜底）把请求拆成有序 `PlanStep` 列表
- 维护 `plan` + `step_index`，由 [start_step_node](file:///Users/bill/PycharmProjects/langgraph-test/src/skill_runtime/graph.py#L473) 逐个实例化子节点执行
- 步骤间可传递产物（`input_from="prev_output"`）
- **支持嵌套**：某个 step 可以是 dynamic，从而在 plan→start_step→plan 之间递归
- 例：`weather.trip_plan`（出差规划，动态拆成 查天气→查预报→出简报）、`trip_plan.briefing`（嵌套 dynamic，再拆 翻译→总结）

## 三者的协作关系

以"出差去北京三天，给我中文简报"为例：

```
weather (category)              ← 下探
└─ trip_plan (dynamic)          ← 规划 3 步
   ├─ weather_check (atomic)    ← 执行：北京天气
   ├─ forecast_check (atomic)   ← 执行：3 天预报
   └─ briefing (dynamic)        ← 嵌套规划 2 步
      ├─ translate_brief (atomic) ← 执行：翻译
      └─ chat_brief (atomic)      ← 执行：生成简报
```

- **category** 负责"组织分层"（静态目录结构）
- **dynamic** 负责"运行时编排"（按请求动态拆步骤）
- **atomic** 负责"实际产出"（唯一产生 output 的节点）

三者通过 `_classify` 条件边分流，`complete_node` 自环实现递归上推收束，最终所有 atomic 的 output 在 [finish_node](file:///Users/bill/PycharmProjects/langgraph-test/src/skill_runtime/graph.py#L598) 汇总成最终答案。



# SkillExecutionNode 中 `plan` / `steps` / `step_index` 三者关系与作用

三者只对 **dynamic 节点**有意义（atomic/category 节点的 `plan` 为空、`step_index` 恒为 0）。它们构成 dynamic 节点的"规划—执行—推进"三件套：

## 各自的职责

| 字段 | 类型 | 角色 | 比喻 |
|------|------|------|------|
| `plan` | `list[PlanStep]` | **规划蓝图**：LLM 规划出的有序步骤清单（skill_id / objective / arguments / input_from / status） | 菜谱 |
| `step_index` | `int` | **进度指针**：指向 `plan` 中**下一个要执行**的步骤下标 | 翻到第几页 |
| `steps` | `list[SkillExecutionNode]` | **执行轨迹**：已实例化的子执行节点（递归树），带真实 output/status | 做出来的菜 |

## 生命周期（一个 dynamic 节点的完整流程）

以 `trip_plan` 规划出 3 步为例：

```
plan_node 阶段：
  plan = [weather_check, forecast_check, briefing]
  step_index = 0
  steps = []
  ──────────────────────────────────────────────

第 1 轮：start_step(step_index=0)
  读 plan[0] → 实例化 weather_check 节点 → append 到 steps
  steps = [weather_check]   ← len(steps) == step_index + 1
  complete 后：step_index = 1, plan[0].status = COMPLETED

第 2 轮：start_step(step_index=1)
  读 plan[1] → 实例化 forecast_check → append
  steps = [weather_check, forecast_check]
  complete 后：step_index = 2, plan[1].status = COMPLETED

第 3 轮：start_step(step_index=2)
  读 plan[2] → 实例化 briefing（本身是 dynamic，递归进入 plan→start_step...）
  steps = [weather_check, forecast_check, briefing]
  complete 后：step_index = 3, plan[2].status = COMPLETED

_after_complete 判断：step_index(3) >= len(plan)(3) → 走 complete 收束自身
```

## 对应关系

核心不变量：**`plan[i]` ↔ `steps[i]`**（下标对齐）

```
plan:  [step0,   step1,   step2]
          ↓       ↓       ↓
steps: [node0,   node1,   node2]
          ↑
      已完成(i < step_index) / 进行中(i == step_index) / 未开始(i > step_index)
```

- `i < step_index`：`plan[i]` 已完成（`status=COMPLETED`），`steps[i]` 有 output
- `i == step_index`：`plan[i]` 正在执行（`status=RUNNING`），`steps[i]` 正在跑
- `i > step_index`：`plan[i]` 待执行（`status=PENDING`），`steps[i]` 尚未实例化

## 为什么要同时存 `plan` 和 `steps`

| 只存 `plan` 不行 | 只存 `steps` 不行 |
|---|---|
| 暂停/恢复后不知道下一步该跑哪个子技能、用什么参数 | 无法知道总共有几步、步骤是否已全部规划完成 |

`plan` 是**持久化的执行计划**，`steps` 是**运行时的执行产物**。两者配合才能支持：
1. **暂停/恢复**：恢复时读 `step_index`，从 `plan[step_index]` 继续，已完成步骤的 output 留在 `steps` 里供 `input_from="prev_output"` 回溯
2. **可观测性**：`collect_dynamic_skills` 收集所有 dynamic 节点，既能看规划（`plan`）又能看执行进度（`step_index`）和产物（`steps`）
3. **嵌套递归**：`steps[i]` 本身可以是 dynamic 节点，拥有自己的 `plan`/`step_index`/`steps`，形成任意深度的递归执行树

对应代码入口：[plan_node](file:///Users/bill/PycharmProjects/langgraph-test/src/skill_runtime/graph.py#L441) 写 `plan`，[start_step_node](file:///Users/bill/PycharmProjects/langgraph-test/src/skill_runtime/graph.py#L473) 读 `plan[step_index]` 并 append 到 `steps`，[complete_node](file:///Users/bill/PycharmProjects/langgraph-test/src/skill_runtime/graph.py#L573-L577) 推进 `step_index`。


