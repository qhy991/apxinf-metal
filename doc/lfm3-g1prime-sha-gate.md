# QUEUE #8 — G1′ suite builder 聚合 sha 门预登记（round-162）

**性质**：预登记（仅写 Gate 与 Decision；**未动任何脚本、未开任何设备**）。commit 先于脚本改动。
**触发**：`/Users/haiyan/lfm3-dig/STATUS.md` §649 冻结的二选一之 ②：「builder 加固预登记档
`doc/lfm3-g1prime-sha-gate.md`：Gate = 重建 task-set/id-seq 与冻结值逐字一致才放行、否则 abort；
Decision = 单 gate 单 PR，不与任何设备测量混；commit 后才允许动脚本」。
**日期**：2026-11-22 16:03 CST，`date` 实取；base `origin/feat/lfm3-metal` @ `11a8fa7d`
（`git fetch` 本轮核实），receipt 分支本档为 `agent/lfm3-g1prime-sha-gate`（从 `11a8fa7d` 新建）。
**用户授权**：用户本轮直接指令「执行队列当前项的下一个最小可完成步骤…不要等批准」⇒ 免 Lead 预批执行本档。
**不改**：QUEUE #8 `doc/lfm3-native-4bit.md` §2 / §5 的任何 Gate、阈值与五档判据；
不碰生成路径、不碰任何 model / weights / apxinf-mlx 代码。

---

## 0. 为什么这一档（QUEUE #8 自选义务：先量化 + 先证明不改单轨迹数值）

round-83（§633 F1）发现：`scripts/build-g1prime-suite.py` 逐 task 冻结
prompt/test/code 的 12-char 前缀（`FROZEN` 表，line 74–102，漂移 ⇒
`FROZEN_PAYLOAD_MISMATCH` abort），但**从不计算、也不校验** suite 的两个聚合冻结值：
- task-set sha = `13a96e891bf484b86fd92dd04db8e0611c190709362e1c3e9a76366f6fc72910`
- id-seq sha = `31157e868ae01230386f484daef0703c63ec637d0f229dddbabd77a3b49242d6`

`scripts/run-g1prime.sh:48` 只算整文件 `shasum` 写 log，**不与任何冻结值比较**；
且整文件 shasum 因 `frozen_at` 时间戳每次必变（round-84 已证），
**不能**作为冻结门 —— 冻结的只能是两个 payload 聚合 sha。

round-84（§645，M4 锁外只读）复算：当前 builder（md5 未变）
从冻结输入重建的两个聚合 sha 与冻结值**逐字一致**（F1 = CONFIRMED）。
机制定义（round-84 锁定，可复算）：
- task-set sha = `sha256("".join(t["sha256"] for t in tasks))`（`t["sha256"]` = builder line 129
  写入的 per-task payload sha，顺序 = suite.json 顺序）；
- id-seq sha = `sha256("".join(t["id"] for t in tasks))`。

本档据此提出**加固**：让脚本自身算出并校验这两个聚合 sha，不一致即 abort
（fail-closed），使「payload 漂移但 per-task FROZEN 表被逐行补丁绕过」这类静默失效无法发生。

**量化残差**：这一档**不是性能优化，而是 provenance 门** —— 收益是「冻结 suite 的可证伪性」，
不产生 GB/s、不改任何速度数字。故本档不涉及「0.5% cell decode」类收益阈，
也不进 QUEUE #8 五档加速结论（见 §5）。

**不触碰单轨迹数值**（QUEUE #8 / §0 义务 2）：本档唯一目标文件是
`scripts/build-g1prime-suite.py`（host 侧离线构建脚本），它**不在**生成路径
（不进 `build_graph` / prefill / decode / argmax），不改权重、不改 kernel。
按构造不改任何贪心 token；该命题仍由 §1 G1 门要求 token 一致性证明背书，
不豁免于「代数等价 ≠ token 等价」的通用教训。

---

## 1. Gate（冻结；任一硬门触发即 INVALID）

> **G1 token 一致性（硬门）**：本档改动后的脚本构建的 suite，与改动前脚本
> （commit `78e8a2c0` 同 md5 的 `8f167512…` 版本）在 **同一冻结输入** 上产出的
> **task 集合逐 task 相同**（id、prompt、test、gold code、per-task sha256 全部逐字段相等；
> `frozen_at` 允许不同）。任一 task 任一字段不同 ⇒ **INVALID**。
> （本门是脚本产物的一致性门，不涉及模型生成；它证明加固没有改变 suite 本身。）
> **G2 不触碰生成路径（按构造 + diff 门）**：`git diff --name-only` 必须仅命中
> `scripts/build-g1prime-suite.py`（可选地含本 doc）；`git diff` 不得出现
> `build_graph|prefill|decode|argmax|matmul|quantized_matmul|Compiled|model` 字样的
> **新增**行（除已存在的字段名与注释）。**违反 ⇒ INVALID**。
> **G3 聚合 sha 门生效（effect proof / 空臂门）**：加固后的脚本在冻结输入上
> **必须真实打印** `task_set_sha=<64hex>` 与 `id_seq_sha=<64hex>` 两行（或等价单行
> 含两字段），且两值均等于冻结值；打印缺失或值不符 ⇒ **G3 FAIL**（不能以「per-task
> FROZEN 表已过」代替 —— 那正是 round-83 发现的缺口）。
> **G4 mismatch 负控（门必须是 load-bearing 的）**：人为扰动一个 task payload
> （构造一份与冻结输入仅一处字节不同的 cache 副本，落 `--mbpp-cache` 指向的临时文件；
> **实测收紧解释（round-163 §7）**：MBPP 侧 prompt/test/code 已被 per-task `FROZEN`
> 前缀表钉死，MBPP 扰动必然先被 `FROZEN_PAYLOAD_MISMATCH` 拦下 —— 「仅新门干活」的
> 构造在 MBPP 上不存在；故 G4 的有效构造 = 扰动 per-task `FROZEN` **未覆盖**的 payload
> 字段（如 HumanEval 的 `problem`，经 `--ag2-suite` 副本），该路径须由**聚合门**拦截），
> 脚本**必须 abort**（非零退出）且错误信息指向 task-set 或 id-seq 的 mismatch
> （不能静默写出新 suite）。**不 abort ⇒ INVALID**（门是死代码）。
> **G5 A/A**：本档不跑任何设备窗，**A/A N/A**。若后续基于此 suite 开任何
> latency / quality 窗，该窗须 A/A ≤5% 且**不可引用 #121 S1 5.02%**（§648 既有纪律）。

---

## 2. Decision（冻结；不可静默改）

> **方向**：QUEUE #8 自选 —— G1′ suite builder **聚合 sha 门加固**（离线、host 侧、
> provenance 类；**非速度优化**）。
> **实现范围**：仅 `scripts/build-g1prime-suite.py`（line ~128 后新增聚合 sha 计算、
> 与冻结值比对、不一致 abort，并在成功路径打印两行）；**禁改** `doc/lfm3-native-4bit.md`
> §2/§5 的任何 Gate、阈值、判据，禁改 `model.rs` / `weights.rs` / `apxinf-mlx` /
> `gateway.rs` / `supervisor.rs`，禁改 `FROZEN` 表与 `run-g1prime.sh` 的行为。
> **流程**：预登记（本文）→ commit（**先于任何脚本改动**）→ 实现脚本改动 →
> 本地 `python3 -I` 自检（`--help`、语法）→ M4 `python3 -I` 冻结输入复现 G1/G3
> （无需双 Metal 锁，builder 不占锁）→ G4 负控 → 五档判定 → 写入本文 §5 Results →
> 一个 commit，一个 PR（base `feat/lfm3-metal`）。
> **完成档判定的档位命名勘误（round-163 §7 实际填写时）**：§2/§4 旧文称「五档判定 →
> 写入本文 §5 Results」，但 RUNNER §5 五档（WIN/PARITY/PARTIAL/FALSIFIED/INCONCLUSIVE）
> 仅适用于设备加速结论；本档无测量 ⇒ **不产生五档**，实际写入 = **完成档 PASS**（§4 已
> 明定 PASS/INVALID/INCONCLUSIVE 三档），§7 按此执行。该勘误不改任何门、不改判据。
> **单 gate 单 PR**：本档不与任何设备测量窗、不与 P2 W1、不与 GSM8K bug-2 混在一个
> PR（RUNNER §6「一个方向一个 PR」）。
> **不静默开优化**：本档**不授权**任何对 suite 内容、任务数、评测协议的改动；
> 若要改 suite，须另起预登记并经 Lead。

**Frozen inputs（round-84 实取，M4）**：
`~/ag2-datasets-cache/google-research-datasets__mbpp__sanitized__test.json`（154,279 B）、
`~/ag-run/suite/AG2-suite-v1.json`（11,912 B）；`--k 19` ⇒ 20 题（MBPP rows 0..19 +
HumanEval/80）。

**冻结值（逐字；出处 `doc/lfm3-native-4bit.md:178`、`doc/lfm3-campaign-summary.md:33`）**：
- task-set sha = `13a96e891bf484b86fd92dd04db8e0611c190709362e1c3e9a76366f6fc72910`
- id-seq sha = `31157e868ae01230386f484daef0703c63ec637d0f229dddbabd77a3b49242d6`

---

## 3. Cells / Order（无设备窗；仅脚本自检顺序）

| 步 | 内容 | 门 |
| --- | --- | --- |
| S1 | `python3 -I scripts/build-g1prime-suite.py --help` + `py_compile`（本地，改动后） | 语法 |
| S2 | M4 冻结输入 `--force --k 19` 正向构建（改动后） | G1/G3 |
| S3 | 同一输入跑**改动前**脚本（`git show 78e8a2c0:scripts/build-g1prime-suite.py` 落 `/tmp`） | G1 |
| S4 | 逐 task 字段比对 S2 vs S3（忽略 `frozen_at`） | G1 |
| S5 | G4 负控：扰动 1 字节的 cache 副本 ⇒ 必须 abort | G4 |
| S6 | `git diff --check` + `git diff --name-only` 复核 | G2 |

**Order**：S1 → (S3 备份) → S2 → S4 → S5 → S6。产物（两份 suite、比对结果、负控
stderr）落 M4 `~/lfm3-dig-run/g1prime-sha-gate/`，**不覆盖**任何历史 receipt。

---

## 4. 机制算术 / 判据（本档无速度机制；记录 provenance 算术）

- **per-task payload sha**（已有，line 128–129）：
  `t["sha256"] = sha256(json.dumps(t, sort_keys=True, ensure_ascii=False))`
  —— 20 题 × 64 hex char = 1280 chars 进入 task-set 串联；
- **task-set 串联**：`"".join(t["sha256"] for t in tasks)`（20 × 64 = 1280 chars）⇒
  `sha256(...)` = 冻结 `13a96e89…`；
- **id-seq 串联**：`"".join(t["id"] for t in tasks)`（id 形如 `mbpp_0011` 9 chars × 19
  + `humaneval_0080` 13 chars = 171 + 13 = 184 chars）⇒ `sha256(...)` = 冻结 `31157e86…`；
- **整文件 shasum 不可用**（round-84）：`frozen_at` 时间戳使整文件每次必变，
  故 G3 只认两个 payload 聚合 sha；
- **launch 数 / GB/s**：本档不涉及（host 离线脚本），**机制未知不适用**；
- **G4 负控算术**：扰动 1 byte ⇒ 至少一个 per-task sha 变化 ⇒ task-set 串联变化 ⇒
  task-set sha 变化（SHA-256 抗单字节碰撞）⇒ 门必须 abort。这条是 G4 可执行的
  机制依据；若实测不 abort，说明实现只算了打印、没做比较（死代码）⇒ INVALID。

**五档（本档适用性说明）**：RUNNER §5 的 WIN/PARITY/PARTIAL/FALSIFIED/INCONCLUSIVE
是**加速/假设检验**结论，须有设备测量与机制算术。本档是**预登记 + 门加固**，
**不产生加速五档**；本档自身的判定档位（按 §629/§647 先例，记为「完成档」而非五档）：
- **PASS**：G1–G5 全过 ⇒ 聚合 sha 门落地，QUEUE #8 F1 的「无脚本门」缺口关闭；
- **INVALID**：任一硬门失败（G1 产物变化 / G2 diff 越界 / G3 无打印或值错 /
  G4 负控不 abort）；
- **INCONCLUSIVE**：M4 冻结输入缺失或不可达（无法执行 G1/G3）—— 记录挂起，
  不得以「per-task FROZEN 表过了」冒充 PASS。

---

## 5. Tests（对应 §3 步骤）

| 编号 | 内容 | 门 |
| --- | --- | --- |
| **T1** | 改动后脚本 `--help` + `py_compile` 通过 | 语法（S1） |
| **T2** | M4 冻结输入正向构建，打印两聚合 sha 等于冻结值 | G1 + G3 |
| **T3** | 改动前脚本同输入构建，逐 task 字段与改动后完全相等（忽略 `frozen_at`） | G1 |
| **T4** | G4 负控：扰动 cache 1 byte ⇒ 非零退出且错误指向聚合 sha | G4 |
| **T5** | `git diff --check` 0 行 + `git diff --name-only` 仅本 doc + 脚本 | G2 |

**本档不跑**：任何 `apxinf-*`、`cargo test`（无 Rust 改动）、任何双 Metal 锁操作、
`pmset` / `caffeinate` / server / probe / bench —— 全部不触发。

---

## 6. Receipts（执行后填写；round-84 已提供的前置 receipt）

- **前置 receipt（round-84，§645，已在 `agent/lfm3-dig` commit `78e8a2c0` / PR #164）**：
  M4 锁外只读复算，两聚合 sha 逐字一致；builder md5
  `8f16751206fce2e4e8fe17a4fc7c1109` == `origin/feat/lfm3-metal` 同文件。
- 预登记 commit：`ae71939f`（round-162，先于任何脚本改动 ✅）。
- 实现 commit：`9fcefb17` `feat(scripts): abort g1prime-v2 build on Lead-frozen
  aggregate-sha mismatch`（round-163，2026-11-22 16:15 CST push `ae71939f..9fcefb17`
  RC=0 至 `origin/agent/lfm3-g1prime-sha-gate`；脚本 md5
  `63e28d80ef7f16050d14f7b95d396431`，+20/−0，v1 `k<19` 路径与 `FROZEN` 表逐字未动）。
- G1/G3 M4 receipt：M4 `~/lfm3-dig-run/g1prime-sha-gate/`，2026-11-22 16:16–16:18 CST
  （builder 不占双 Metal 锁；M4 网络侧 `Proxy CONNECT aborted` ⇒ 脚本经 `scp` 传输
  落盘 md5 对拍 `8f167512…`/`63e28d80…` 与本地一致，未在 M4 侧 git fetch）：
  - `before.py`（`origin/feat/lfm3-metal` 版，md5 `8f16751206fce2e4e8fe17a4fc7c1109`）、
    `after.py`（`origin/agent/lfm3-g1prime-sha-gate` 版，md5
    `63e28d80ef7f16050d14f7b95d396431`）；
  - **G3 PASS**：`after.py --force --k 19 …` RC=0，`after.stdout` 实印
    `task_set_sha=13a96e891bf484b86fd92dd04db8e0611c190709362e1c3e9a76366f6fc72910` +
    `id_seq_sha=31157e868ae01230386f484daef0703c63ec637d0f229dddbabd77a3b49242d6`
    （均 = 冻结值，两值等值打印非「per-task FROZEN 表替代」）；
  - **G1 PASS**：`before.py` 同输入 RC=0 产出 `suite-before.json`；
    `suite-after.json` vs `suite-before.json` 逐 task 全字段相等
    （`tasks_identical_ignoring_frozen_at = True`、`id_order_identical = True`、
    `diffing_fields = []`、`other_meta_diffs = {}`，n=20/20），且**两文件整文件
    sha256 同为 `989bd0bb390163b86cbf1d9390034d7b5d260185f293fe3d071b22f6171075b5`**
    （本轮两次构建恰好同秒写入 `frozen_at`，整文件亦逐字节相同 —— 整文件 shasum
    因时间戳「可能」变，本轮未变，不影响门；G3 只认 payload 聚合）；
  - 从 `suite-after.json` 反推复核两聚合 sha 与冻结值一致（RC=0）。
- G4 负控 receipt：同一目录，两次扰动（HARNESS 双盲点，如实记录 —— 见 §7）：
  - **N-A（MBPP cache 扰动）**：`mbpp-perturbed.json`（md5
    `5c1b4102e9fe3a6e699bd7b2a65c724719aa2e26a0390096f3e8a4ed6e3b0201`，
    rows[0].prompt 尾追 1 字节）⇒ RC=**1**、`neg.stderr` =
    `FROZEN_PAYLOAD_MISMATCH mbpp_0011.prompt …`、`suite-negative.json` 未写出 ——
    由**既有 per-task 前缀门**拦截，未触到聚合门；因 MBPP 的
    prompt/test/code 三字段已被 `FROZEN` 表钉死，MBPP 侧不存在 1-byte 扰动
    可绕过前缀门直达聚合门的构造路径（构造路径不存在，非负控失败）。
  - **N-B（AG2 HumanEval 扰动，命中新门）**：`ag2-perturbed.json`（md5
    `fbe7a4fa0d57ac491ccb5592a2923c4be29ef6125c82c35dc1516439737634b6`，
    `humaneval_0080.problem` 尾追 10 字节 —— `problem` 不在 per-task
    `FROZEN` 三字段内）⇒ RC=**1**、`neg2.stderr` =
    `AGGREGATE_SHA_MISMATCH: task_set_sha=ae7e69fa… (want 13a96e89…) id_seq_sha=31157e86… (want 31157e86…)
    — emitted task set differs from the Lead-frozen g1prime-v2 suite; aborting before any device run`、
    `suite-negative2.json` 未写出 ⇒ **聚合门 load-bearing、G4 PASS**。
    算术：id-seq 不变（31157e86…）而 task-set 变（13a96e89…→ae7e69fa…），
    恰与「单 task payload 变 ⇒ 其 per-task sha 变 ⇒ 串联变 ⇒ task-set 变、
    id 序列不变」的机制预测一致。
- PR：_（待填；base `feat/lfm3-metal`，head `agent/lfm3-g1prime-sha-gate` —— 按本 doc
  §2 冻结流程，实现+测试+完成档判定齐后由本分支开出，不为实现中途抢先开 PR）_

---

## 7. Results（round-163 实现 + 测试；2026-11-22 16:15–16:18 CST，`date` 实取）

**完成档判定 = PASS（G1–G5 全过；非 RUNNER §5 加速五档，理由见 §4）。**

| 门 | 结果 | 依据 |
| --- | --- | --- |
| **G1** 产物一致性（硬） | **PASS** | `suite-after.json` vs `suite-before.json` 逐 task 全字段相等（忽略 `frozen_at` 恒等、
  `diffing_fields=[]`、id 序同、n=20/20）；加固未改 suite 本身 |
| **G2** 不触碰生成路径（硬） | **PASS** | `git diff --name-only` = 仅 `scripts/build-g1prime-suite.py`（本轮 results commit
  前 T5 复核）；新增行 `grep -E 'build_graph\|prefill\|decode\|argmax\|matmul\|quantized_matmul\|Compiled\|model'`
  = **0 命中**；无任何 Rust / 模型 / kernel 改动 |
| **G3** 聚合 sha 门生效（硬，空臂门） | **PASS** | `after.stdout` 真实打印两行且逐字等于冻结值（`13a96e89…` / `31157e86…`）；
  从输出 suite 反推复核同值 |
| **G4** mismatch 负控（硬，load-bearing） | **PASS** | N-B（`ag2-perturbed.json` md5 `fbe7a4fa…`，`humaneval_0080.problem` 不在 per-task
  `FROZEN` 三字段内）⇒ RC=1、stderr 含 `AGGREGATE_SHA_MISMATCH`（task-set 变、id-seq 不变，符合机制预测）、
  无 suite 落盘；N-A（MBPP cache 尾追 1 字节）被既有 per-task 前缀门拦在 `FROZEN_PAYLOAD_MISMATCH`，
  该构造路径不存在（MBPP prompt/test/code 已被 `FROZEN` 表钉死），如实记录不冒充 |
| **G5** A/A | **N/A** | 本档无设备窗（builder 不占双 Metal 锁，全程 `python3 -I` 离线） |

**T1–T5**：T1 `--help` + `py_compile` 本地通过（16:14）✅；T2 M4 正向构建打印两聚合 sha ✅；
T3 改动前同输入逐 task 相等 ✅；T4 负控非零退出且错误指向聚合 sha ✅；
T5 `git diff --check` RC=0 + `--name-only` 仅脚本（+ 本 doc）✅。

**机制算术**：本档不产生 GB/s、launch 数或速度数字（host 离线脚本，如实标注）；唯一机制
证据 = G4 N-B 的「task-set 13a96e89…→ae7e69fa…、id-seq 保持 31157e86…」对分叉，
证明门比对的是 payload 聚合而非仅 id 列表；「1-byte 扰动 ⇒ 1 per-task sha 变 ⇒ 串联变 ⇒
task-set 变」由 SHA-256 抗单字节碰撞保证，本轮以 10-byte 扰动实证（N-B）。

**QUEUE #8 F1 缺口关闭**：「builder 无聚合 sha 门」（round-83 §633 F1）自本轮起 =
**已关闭**（PASS）；round-84 §645 F1 = CONFIRMED 不变；**QUEUE #8 总体判定不变** =
PARTIAL（G1′ n=20）+ G4-speed PASS ⇒ **4-bit opt-in、BF16 默认**。

**非静默开优化声明**：本轮未改任何 suite 内容、任务数、评测协议；`doc/lfm3-native-4bit.md`
§2/§5 全部 Gate/阈值/判据、`FROZEN` 表、`run-g1prime.sh` 行为逐字未动。

**HARNESS 双盲点（如实记录，不美化）**：
1. **G4 构造的下界**：G4 的机制算术（§4）以「扰动 1 byte ⇒ task-set sha 变」为前提，
   但 N-A 显示 MBPP 缓存扰动会被**已有 per-task 前缀门**先拦下 —— 在 MBPP 侧，
   「只让新门干活」的构造在数学上不存在（三字段已被 12-char 前缀钉死）。
   因此 G4 的可执行路径 = 扰动 **per-task `FROZEN` 未覆盖的 payload**（本轮用
   HumanEval `problem`）；这是对原 Gate 文本「cache 副本」字面义的收紧解释，
   已在本节显式记录，不留待后人误判。
2. **本轮两文件整文件 sha256 恰同**（均 `989bd0bb…`）：因两次构建写入同一
   `frozen_at` 秒。整文件 shasum 因时间戳**可能**每次变（round-84 已证），故 G3
   **只认 payload 聚合 sha** —— 本轮整文件相同是巧合，不可引用为冻结门。

---

## Relation / 不重开

- **QUEUE #8 G1′**：本档是其 F1「无脚本门」的加固，**不改** G1′ 判据与
  PARTIAL 结论（4-bit opt-in、BF16 默认）；
- **GSM8K bug-2**：仍 BLOCKED-on-Lead（§649 a 项），本档不触碰；
- **P2 W1 / `lfm3-ane-load` 锁**：本档全程离线、不连 M4 双锁，不触碰 pid 9884/24582；
- **QUEUE #6 batch harness（§649 二选一之 ①）**：本档不涉及；F2 OPEN 项按 §628 挂起，
  待 Lead 裁决 NOT-WORTH 是否连 harness 也禁；
- **98,552,320 doc 修订**：仍待 Lead 裁定，本档不碰。
