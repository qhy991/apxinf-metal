# 4-bit 能否设主力 — 真实验收负载门（Gate + Decision）

预登记日期：2026-11-20 10:15 CST（`date` 实取）。预登记先行 commit，设备运行（M4 窗）必须在本 commit 之后。

- 相关：Lead 追加任务 2026-11-20 10:14（STATUS 原文）：用户问"原生 4-bit 能否作为主力"，现有证据仅为代码题 proxy（n=20，有效分辨题 6 道），不足以支撑设默认。
- 权威代码基座：`agent/lfm3-dig-primary`，base `feat/lfm3-metal`（`e353eda5`，PR #132 已合并）。

## 1. Gate（冻结；Lead 2026-11-20 10:14 原文逐条）

| # | 门 | 判据 |
|---|----|------|
| **P1（主判据）solved rate** | 真实 agent 端到端，同二进制、同 prompt、同 seed，`native-4bit`（`APXINF_LFM3_4BIT=1`）vs `native-bf16`（unset）：MBPP/HumanEval 多轮任务，**≥12 题**，每臂 **≥2 段 ABBA** |
| **P2 端到端墙钟** | 含所有轮次的进程墙钟（`code_loop` 进程 wall clock，不受 #120 SSE 全缓冲影响），或引擎侧 decode 时间 × 轮次数（口径二选一并写明）。目标：**≥2×** 下降（3.3× decode 加速是否转化到 agent 层） |
| **P3 失败分类** | 失败样本分为「早期 token 翻转导致跑偏」vs「根本没解出来」，记数与样例 |
| **G3 effect proof** | 每段 banner `native_4bit=<true\|false>` 与 env 同源；armed 臂必须 `=true`；`find crates -newer` binary 为空（无空臂）。缺失 ⇒ INVALID |
| **G5 A/A** | 每臂 ≥2 段：段间 solved 集合一致；不一致 ⇒ 报 DISAGREE，以逐任务表为 tiebreaker（沿用 G1′ n=20 冻结口径，不自动 INVALID）；但窗口若同时测墙钟，墙钟段 spread >5% ⇒ 该窗口墙钟数字作废 |

## 2. Decision（判定，冻结）

- **差 ≤3pp 且端到端墙钟显著下降 ≥2×** ⇒ 4-bit 可设为**主力默认**（保留 `APXINF_LFM3_BF16=1` 回退开关；注意：回退开关形态需实现轮确认 env 名，不静默改既有 `APXINF_LFM3_4BIT` 语义）。
- **差 3–10pp** ⇒ **opt-in 为主**，特定延迟敏感场景启用。
- **差 >10pp** ⇒ 不设主力。
- INVALID 仅限：G3 空臂 / 窗口作废后 A/A 墙钟仍 >5% / 协议不一致（非同二进制同 prompt 同 seed）。跨精度 4-bit ≠ BF16 的 token 差异**不是**门（Lead 00:16 冻结：跨精度不设逐位门）。

## 3. 实现路线（不新发明）

- 复用 `scripts/code-loop.py`（PV-11 已审计动作模板：模型只产代码、harness 写文件跑测试，≤3 retry，`returncode==0 and "OK" in stdout` 判分）+ `scripts/run-g1prime.sh`（G3 banner/census/freshness 三件套、双 Metal 锁 fcntl、`pmset displaysleepnow` + `caffeinate -si`）+ `scripts/build-g1prime-suite.py`（冻结 payload sha 校验、gold 不进 prompt）。
- 现有 suite 仅 20 题（`g1prime-v2`，19 MBPP rows 0..19 + humaneval_0080），Lead 本项要求 **≥12 题且多轮** ⇒ 本项需扩/改 suite（任务池现成：M4 MBPP sanitized cache 257 行 + AG2-suite-v1 HumanEval 池），并把「多轮」语义写清（code_loop 的 retry 轮次即视为多轮，还是另加 multi-turn 任务模板——属实现轮决策，需在实现 commit 前说明）。
- 4-bit 臂与 BF16 臂同二进制：`APXINF_LFM3_4BIT` 经 `--worker-env` 传入，不进 supervisor 进程环境（round-71 同口径）。
- 墙钟口径（Lead 10:14 注记 #120）：主用 **code_loop 进程墙钟**（含所有轮次，不受 SSE 缓冲影响）；并行记引擎侧 `decode_ms_per_token × completion_tokens` 汇总作 cross-check，两者都写进 receipt。

## 4. Cells / Order（实现轮冻结 2026-11-20 10:2x CST）

- ABBA 顺序 A B B A，A = `native-4bit`（`APXINF_LFM3_4BIT=1`）、B = `native-bf16`（unset）；每臂 2 段，共 4 段（≥12 题要求满足：20 题 ≥ 12）。
- **题集冻结**：复用 `g1prime-v2` 20 题 suite（`build-g1prime-suite.py --k 19`），**不扩不改**：
  - ids：`mbpp_0011,0012,0014,0016,0017,0018,0019,0020,0056,0057,0058,0059,0061,0062,0063,0064,0065,0066,0067`（19 MBPP sanitized cache rows 0..19）+ `humaneval_0080`；
  - **task-set sha256 = `13a96e891bf484b86fd92dd04db8e0611c190709362e1c3e9a76366f6fc72910`**（M4 `python3 -I` builder 实取，= round-63 §86 / round-65 §97 冻结值，payload 冻结未破）；id 序列 sha = `31157e868ae01230386f484daef0703c63ec637d0f229dddbabd77a3b49242d6`；
  - M4 构建运行 **2026-11-20 10:2x**（`--force --k 19`，builder 0 error，`/tmp/g1p-primary-suite.json`）。开窗前 `run` 必须重建并校验该 sha，不符 ⇒ abort（G3 类 provenance 门）。
- **「多轮」语义（实现轮决策，写死）**：采用 **code_loop retry 轮次即为多轮**（P1 的「多轮任务」指 `code-loop.py` 每题 ≤3 retry 的多轮交互：attempt 1 首解 + 失败后带 test feedback 的 attempt 2/3），**不另加 multi-turn 任务模板**。理由：① 不新发明（RUNNER §3 / doc §3）；② 现有 suite payload（含 sha）已被 G1′ n=20 窗验证，改模板会破冻结；③ 每题 attempts ≥1 即覆盖「多轮」语义下 4-bit 臂是否在 retry 轮次退化（这是 3.3× decode 加速是否转化为 agent 层 P2 的核心）。receipt：`native-*-seg*.json` 每题含 `attempt_count` + 各 attempt `wall_s`/`timing`。
- 双 Metal 锁 fcntl 顺序等待（`/tmp/enginetailor-metal.lock` → `~/.cache/enginetailor/metal-measure.lock`），等不抢。

## 5. Results（待填；设备运行后回填原始数字 + 判定）

- suite 题集 sha 已冻结（见 §4）：`13a96e891bf484b86fd92dd04db8e0611c190709362e1c3e9a76366f6fc72910`。开窗前 build 校验必须命中，否则 abort 不动设备。

