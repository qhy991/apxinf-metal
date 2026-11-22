# QUEUE #8 质量档离线复核：GSM8K / suite 冻结口径（round-83）

**性质**：离线复核，不占 M4 锁、不动设备、不改任何测量口径。仅为记录与下一轮交接。
**触发**：STATUS §631（round-82 冻结的本步）。
**日期**：2026-11-22 15:39 CST，base `origin/feat/lfm3-metal` @ `11a8fa7d`。
**不改**：任何 `Gate:` / `Decision:` / 冻结值。所有需要 Lead 裁决的项原样挂起。

## 1. 复核范围（三问，逐字对应 §631）

1. `doc/lfm3-native-4bit.md` §2 G1′ suite v2 记录的 task-set sha
   `13a96e891bf484b86fd92dd04db8e0611c190709362e1c3e9a76366f6fc72910`
   与 id-seq sha `31157e868ae01230386f484daef0703c63ec637d0f229dddbabd77a3b49242d6`
   是否与当前 `scripts/build-g1prime-suite.py` 输出一致？
2. GSM8K 是否有未修复 bug-2 记录？
3. n=20 的 A/A DISAGREE（10.00pp / 5.00pp）是否需要 Lead 裁决扩到 ≥40 题？

## 2. 证据（本轮实取，receipt）

### 2.1 suite sha：冻结有 receipt，但"当前脚本输出"本地不可复算、且无脚本门

- `scripts/build-g1prime-suite.py`：本地文件 md5 `8f16751206fce2e4e8fe17a4fc7c1109`，
  与 `origin/feat/lfm3-metal` 该文件完全一致；全历史仅 `e353eda5` 一个 commit 触碰。
- 脚本行为（line 128–129）：为每个 task 计算
  `sha256(json.dumps(t, sort_keys=True, ensure_ascii=False))`（**per-task** payload sha），
  并写进 `suite["tasks"][i]["sha256"]`；`FROZEN` 表（line 74–102）逐 task 冻结
  prompt/test/code 的 12-char 前缀，漂移即 `FROZEN_PAYLOAD_MISMATCH` abort。
  **脚本从不计算、也不校验 task-set 聚合 sha `13a96e89…` 或 id-seq `31157e86…`。**
- `scripts/run-g1prime.sh`：只对整个 `suite.json` 文件算
  `sha=$(shasum -a 256 "$SUITE" ...)` 并写 log（line 48），**不与任何冻结值比较**。
- 两个冻结 sha 的出处：`doc/lfm3-native-4bit.md:178`、`doc/lfm3-campaign-summary.md:33`、
  `doc/lfm3-4bit-primary.md` §4，receipt 指向 round-63 §86 / round-65 §97 的 M4 `python3 -I` 实取。
- 复算前置：builder 输入 `--mbpp-cache` / `--ag2-suite` 为 **M4-only** 路径
  （`$HOME/ag2-datasets-cache/…json`、`$HOME/ag-run/suite/AG2-suite-v1.json`），
  M1 无副本 ⇒ **本轮无法离线复算，无法本地证伪或证实该 sha。**

**结论 F1（round-83 = INCONCLUSIVE；round-84 M4 复算后 = CONFIRMED，见 §2.1.1）**：冻结 sha 有历史 receipt、
脚本的 per-task FROZEN 表把关 payload 一致性，round-83 因 M1 无复算输入、且无脚本门，
"当前 builder 输出 == 13a96e89…"一度无法证实或证伪。挂起项：
- 复算须 M4（锁内或离线均可，builder 本身不需锁），round-63/65 同法 `python3 -I`；
- 可选加固（须先预登记再改脚本）：builder 输出 task-set 聚合 sha 与 id-seq sha 并与冻结值
  比对，不一致 abort。**本轮不动脚本。**

### 2.1.1 round-84 复算结果（2026-11-22 15:52 CST，M4 锁外、只读，零双锁）

- 输入在位（M4 实取）：`~/ag2-datasets-cache/google-research-datasets__mbpp__sanitized__test.json`
  154,279 B（11-16 21:32）、`~/ag-run/suite/AG2-suite-v1.json` 11,912 B（11-17 19:21）；
  历史产物 `~/lfm3-dig-run/window-4bit/g1prime/suite.json` 17,537 B，
  `frozen_at = 2026-11-20T02:47:02+08:00`。
- builder 未改：本地 md5 == `origin/feat/lfm3-metal` md5 == `8f16751206fce2e4e8fe17a4fc7c1109`。
- 重建（`python3 -I scripts/build-g1prime-suite.py --force --k 19 --mbpp-cache … --ag2-suite …
  --out /tmp/suite-recheck-20204/suite.json`，FROZEN 表门通过、无 abort）：
  tasks=20，版本 `g1prime-v2`。
- 冻结公式确认（round-63/65 同法，本轮锁定定义）：
  - task-set sha = `sha256("".join(t["sha256"] for t in tasks))`（t 为 builder 写入的
    per-task payload sha，顺序即 suite.json 顺序）；
  - id-seq sha = `sha256("".join(t["id"] for t in tasks))`。
- 重建值 vs 冻结值：
  - task-set sha = `13a96e891bf484b86fd92dd04db8e0611c190709362e1c3e9a76366f6fc72910` ✅ 逐字一致；
  - id-seq sha = `31157e868ae01230386f484daef0703c63ec637d0f229dddbabd77a3b49242d6` ✅ 逐字一致；
  - 与 11-20 历史产物逐字段比对：`task_payloads_identical_ignoring_frozen_at = True`
    （仅 `frozen_at` 因重建时刻不同而变：02:47:02 → 15:52:12）。
- 机制注记：`run-g1prime.sh:48` 的整文件 `shasum` 每次运行必然不同（`frozen_at` 是时间戳），
  故该值**不能**作为冻结门 —— 冻结的只能是 task-set / id-seq 两个 payload 聚合 sha。
  builder 加固（输出并校验这两个聚合 sha，不一致 abort）仍是**可选项、须先预登记再改脚本**，
  round-84 未动脚本。

**F1 档位由 INCONCLUSIVE 改为 CONFIRMED（PASS）**：冻结 task-set / id-seq 与当前 builder
从冻结输入重建的输出逐字一致，payload 未漂移。

### 2.2 GSM8K bug-2：未修复，且无 Lead 批准（F2，BLOCKED）

- 证据链（`doc/lfm3-4bit-knowledge-eval.md` §6，仅存于未合并的
  `agent/lfm3-dig-knowledge` 分支，base 上**不存在**该 doc，见 §5）：
  - round-97 冒烟窗：20/20 GSM8K 请求 `cause=length`、`completion_tokens=64`，
    截断处尚在推理、未出答案 ⇒ 64-token 下 GSM8K accuracy **无效**（P1 caveat）。
  - round-99（`8b224bd3`）原文："**待 Lead 拍板（bug-2，同上，仍未动）**：GSM8K
    `max_tokens` 64 → 256 … 在批准前，本 doc 不给最终五档/主力判定"。
- 批准搜索：全部本地分支 + STATUS.md `grep -n "bug-2"`（命中仅 round-97/98/99 记录、
  campaign summary "GSM8K 待 bug-2"、queue6 review 提及）；**无任何 Lead 批准记录**。
- 权限：`max_tokens=64` 是 Lead 10:20 Gate 逐字冻结值，**worker 无权单方面改**。

**结论 F2（BLOCKED-on-Lead）**：bug-2 未修复、无批准。GSM8K 维持"待 bug-2"，
knowledge 门（MMLU 500 + GSM8K 200）最终五档不可出。批准后另起一轮 + 重开冒烟窗。
**本轮未动。**

### 2.3 n=20 A/A DISAGREE → ≥40 扩样本：Lead 已裁决，无需再裁（F3，CLOSED）

- 事实：G1′ n=20 窗 A/A spread = 10.00pp / 5.00pp，均 >5% ⇒ 该窗标注 DISAGREE；
  solved-rate delta = both −5.00pp / either 0.00pp ⇒ PARTIAL 候选。
- Lead 09:31 裁决（`doc/lfm3-native-4bit.md` §7 尾 + `doc/lfm3-campaign-summary.md` §2/§4）：
  接受该 opt-in 形态为 campaign 收口状态，**明确"未要求扩 ≥40 题"、"无待裁决项"**。
- 按冻结五档（doc §5）：G1′ = **PARTIAL**，QUEUE #8 总体 = PARTIAL
  ⇒ **4-bit opt-in、BF16 默认**（G4-speed PASS 3.3× 不变）。

**结论 F3（CLOSED，无新裁决）**：DISAGREE 是窗口质量标注，已被 Lead 接受；
≥40 扩样本不需要、也未被要求。QUEUE #8 判定维持 PARTIAL。

## 3. QUEUE #8 三问汇总

| # | 问题 | 结论 | 档 | 是否需 Lead |
|--- | --- | --- | --- | --- |
| F1 | 冻结 task-set sha 是否等于当前 builder 输出 | round-83 离线不可复算（INCONCLUSIVE）→ **round-84 M4 复算逐字一致** | **CONFIRMED** | 加固须预登记；裁决无需 |
| F2 | GSM8K bug-2 是否未修复 | 未修复，20/20 `cause=length`，无批准 | BLOCKED | **是（批准 64→256 或替代方案）** |
| F3 | A/A DISAGREE 是否需扩 ≥40 | Lead 09:31 已接受，明确不扩 | CLOSED | 否 |

**QUEUE #8 当前状态不变**：PARTIAL（G1′）+ G4-speed PASS ⇒ 4-bit opt-in、BF16 默认；
knowledge 门 final verdict 挂起于 F2。F1 由 INCONCLUSIVE 改判 CONFIRMED
（§2.1.1），不改变上述总体判定，仅清掉一条 Lead 挂起项
（suite aggregate-sha M4 复算）。

## 4. 本轮动作与未动作

- 做了：读 RUNNER/QUEUE/STATUS（§631）、读 builder/run-g1prime/doc、查全分支与 STATUS
  bug-2 记录、核对 base 与本地脚本 md5、写本文档。
- 恢复：本轮开始时 `agent/lfm3-dig` 被 reset 到 `11a8fa7d`（round-82 后的 sync），
  本地未推送 commit `bcb30a48`（queue6 review，round-82 因网络未推）从指针丢失，
  **已从 reflog cherry-pick 恢复为 `e86fda01`**（本文档在其上再提交）。
  `167fd236`（queue7 census）已在 PR #164，无需恢复。
- 未做（round-83）：未连 M4、未取双 Metal 锁、未 pmset/caffeinate、未开测量窗、未改任何脚本/冻结值/生成路径。
  bug-2、IOSurface 常数 98,552,320、`lfm3-ane-load` 锁持有者（pid 9884/24582）三项 Lead 挂起项**均未动**。
- round-84 追加（2026-11-22 15:52 CST）：M4 锁外复算 F1（§2.1.1），
  `CONFIRMED`。做了：读三件套、确认 builder md5 未变、M4 重建 suite 并比对两个聚合 sha。
  未做：仍未取双 Metal 锁、未 pmset/caffeinate、未开测量窗、未改任何脚本/冻结值/生成路径、
  未动 F2（bug-2）与另两项 Lead 挂起项；`lfm3-ane-load` 持有者本轮未触碰。
  纪律：锁外只读 builder 复算 ≠ 测量窗（RUNNER §3）；`python3 -I` 隔离解释器；
  产物落 `/tmp/suite-recheck-20204/suite.json`，未覆盖任何历史 receipt。

## 5. 附：知识评测 artifacts 位置说明（交接）

`doc/lfm3-4bit-knowledge-eval.md`、`scripts/run-knowledge-eval.sh`、
`samples/knowledge-samples.json` 等**不在 `origin/feat/lfm3-metal`**，
仅存在于本地分支 `agent/lfm3-dig-knowledge`（最新 `8b224bd3`）。Lead 裁决 bug-2 后，
该分支需重新 rebase 到最新 base 再继续；本文档只引用其内容作证据，不迁移这些文件。
