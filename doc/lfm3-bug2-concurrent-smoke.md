# 预登记：native 4-bit 知识评测冒烟窗 —— 并发分段跑法（bug-2 后的 256-token GSM8K）

**状态**：预登记已冻结，commit 在先，测量窗必须在本 commit 之后（RUNNER §1）。
**起草**：round-167，2026-11-22 ~16:40 CST。
**性质**：非速度优化、非质量判据变更 —— 是对 QUEUE #8 附属门（`doc/lfm3-4bit-knowledge-eval.md`，Gate/Decision 判据原文不动）冒烟窗"跑法"的执行形态变更。
**与已冻结事实的关系**：
- bug-2（GSM8K `max_tokens` 64→256，MMLU 维持 64）= Lead 2026-11-22 10:52 已批准（round-165 §670–673 承接，本 doc 不重开该批准）。
- 双上限接线 = commit `408d2858`（round-166 §676–679，已 push 上远端 `agent/lfm3-dig-knowledge`）。
- `doc/lfm3-4bit-knowledge-eval.md` §7 冻结的下一最小步骤 ② = "重开冒烟窗（M4 锁，两臂 × 10 题/段 × 1 段）" —— 本 doc 是**该步骤的执行形态**（怎么跑），不是新的评测方向。

## 1. Gate（冻结原文，供 PR 引用）

```
G1 效应证明（空臂门，RUNNER §3 同因）：
  A 臂 banner 必须 native_4bit=true 且 quantized_triples=154/154；
  B 臂 banner 必须 native_4bit=false。任一不符 ⇒ 该臂 INVALID（既有
  check_banner 已实现，本档不改其实现、不放宽判定）。
G2 A/A ≤ 5%（分段中位的最坏 spread，RUNNER §4）：
  本档 A/B 两臂各 1 段、每段 n_mmlu=10 / n_gsm8k=10；单段两臂的
  accuracy spread 无统计意义（round-97 先例），故 A/A 在本档以
  "两侧引擎 basis ms/token 相对 round-97 实取值的偏差" 量化：
  偏差 ≤ 5% ⇒ 窗口有效；> 5% ⇒ 窗口作废重跑，不许取中位凑。
  （round-97 实取：A 10.74 ms/tok、B 36.06 ms/tok，engine basis。）
G3 token 一致性（RUNNER §2，硬门）：
  本档只改跑法（并发执行请求），不改模型执行图、不改 prompt 模板、
  不改 max_tokens 上限（MMLU 64 / GSM8K 256，round-166 接线值）。
  但贪心输出对执行序敏感 ⇒ G3 要求：A/B 两臂的 per-item 逐 token
  输出与 round-97 冒烟窗同 prompt 同子集的输出逐 token 比对（若有
  不同 ⇒ 记第一处差异位置 + 解释（并发调度改变了 server 侧 KV
  复用/队列顺序 → 累积顺序漂移）⇒ 该 arm INVALID，不看速度）。
  （注：GSM8K 从 64 改到 256 后回复必然更长，故只对 MMLU 段做逐
  token 门；GSM8K 段以 "cause=length 计数归零 + 能解析出收尾答案"
  作为有效性前提，accuracy 数字本身照常判。）
G4 样本 sha 门（doc §5 冻结值）：
  mmlu-500.jsonl dbb9d807…、gsm8k-200.jsonl 1ce79a9f…、
  mmlu-500.ids.json 01110464…、gsm8k-200.ids.json 88243d86…
  全部命中，否则 abort 不动设备（既有 SAMPLE_SPECS 已实现，不改）。
G5 双锁 fcntl 顺序等待（RUNNER §3）：
  /tmp/enginetailor-metal.lock → ~/.cache/enginetailor/metal-measure.lock
  等，不抢；取到后 pmset displaysleepnow + caffeinate -si（不要 -u）。
  本档 hold_locks.py 调用方式不变。
G6 结论五档（RUNNER §5）：
  本档只产生"冒烟窗是否有效 + MMLU/GSM8K direction"，不产生加速
  五档（无 GB/s/launch 机制算术可言）；最终 knowledge verdict 仍
  须 doc §7 冻结的回填步骤（全样本窗）才能进五档。
```

## 2. Decision（冻结原文，供 PR 引用）

```
方向：把 doc/lfm3-4bit-knowledge-eval.md §7 冻结步骤 ② "重开冒烟窗"
  的执行形态确定为"并发分段"，理由如下 ——

现状（round-97 实取，doc §6）：
  B 臂（bf16）GSM8K 20 题 cause=length 100%、completion=64。在 64
  token 上限时每题 decode ≈ 64 × 36.06 ms ≈ 2.31 s，20 题串行
  ≈ 46 s；bug-2 后单题上限 256，但实际短答案 ~70–100 token，decode
  ≈ 100 × 36.06 ≈ 3.6 s，20 题串行 ≈ 72 s，仍远在 Lead 10:52
  "1–2 分钟" 设备时间内 —— **串行其实可行，但 A 臂（4bit）会更快
  （10.74 ms/tok）**，两臂时长差 ~3.3×，A/B 交错窗的绝对时长差异
  让 A/A 的引擎 basis 偏差更难归因（round-97 已见 A 10.74 / B 36.06）。

本档的并发做法（只改 host 侧请求编排，不碰生成路径）：
  - `analyze-knowledge-eval.py run_segment` 新增 `--concurrency N`
    （默认 1，即串行，向后兼容；本窗用 N=4）。
  - MMLU 段：N=4 线程并发（每个样本一次 curl，互不共享状态）；
    结果按原 index 顺序写入 `results`（`ThreadPoolExecutor.map`
    保序），打印行保持 `[i/total]` 格式，`per_item` 解析逻辑不变。
  - GSM8K 段：**串行**（N=1）。原因：round-97/99 已见 GSM8K 输出
    长（256 上限下尤其），并发会让 server 侧 KV cache 队列交错，
    而 GSM8K 段恰恰是"逐 token 输出必须与单发一致"最敏感的地方
    （doc §6 P1 caveat：64-token 截断的中间量；并发下更易扰动
    prefill/decode 边界）。MMLU 段答案只有 1 字母，输出短、对队列
    交错不敏感，并发的收益（B 臂 10 题 × ~1.5 s ≈ 15 s → ~4 s）
    直接压缩冒烟窗总时长。
  - `run-knowledge-eval.sh` 的 `seg_arm` 调用 analyzer 时对
    MMLU 传 `--concurrency 4`，GSM8K 不传（默认 1）。
  - engine basis 的 request ledger join（`join_timing`）在并发下
    仍按"下一个未认领的 request_id"匹配 —— 并发会打乱 file 顺序，
    但 request_id 唯一，且 A/B 两臂各自独占一个 gateway、单段内
    只有本臂的请求 ⇒ 认领语义不变；本档不要求 ledger 的 wall-clock
    顺序与 results 顺序一致（`timing` 仅用于打印/后续 aggregate，
    per-item correctness 不依赖 timing）。

范围（surgical，单方向）：
  - 只改 scripts/analyze-knowledge-eval.py（+`--concurrency` flag、
    MMLU 段并发调度）+ scripts/run-knowledge-eval.sh（seg_arm 传参）。
  - 不改：doc/lfm3-4bit-knowledge-eval.md 任何一行（Gate/Decision/
    判据原文）、extract_mmlu/extract_gsm8k、per_item、aggregate、
    decide、prereg_reading、samples/ 冻结样本、crates/ 任何 .rs、
    apxinf-serve/apxinf-native-worker 二进制、hold_locks.py。
  - 不改 max_tokens 上限（MMLU 64 / GSM8K 256，round-166 值）；
    不改 n_mmlu=10 / n_gsm8k=10 / segs_per_arm=1（Lead 10:52 冒烟形）。

流程（commit 先于测试与测量，RUNNER §1）：
  ① 本 doc commit（预登记，先于任何代码改动）；
  ② 实现轮（改两脚本）→ commit → 本地 py_compile / zsh -n /
     git diff --check / git diff --name-only 自检；
  ③ M4 测量轮（取双锁 → pmset + caffeinate -si →
     scripts/run-knowledge-eval.sh w1 ~/lfm3-dig-run（默认冒烟形））；
  ④ analyzer analyze 出 knowledge-verdict.json；
  ⑤ 回填 §6（doc/lfm3-4bit-knowledge-eval.md，单独 commit）；
  ⑥ 一个方向一个 PR（base feat/lfm3-metal，head
     agent/lfm3-dig-knowledge，PR 正文含 Gate 原文、before/after
     原始数字、A/A、token 一致性、receipts 路径、
     🤖 Generated with [Claude Code](https://claude.com/claude-code)）。

机制算术（离线估算，非测量；实测以 receipt 为准）：
  - MMLU 段 B 臂串行 ≈ 10 题 × 64 tok × 36.06 ms ≈ 23.1 s；
    并发 N=4 ⇒ 理想 ≈ 23.1 / 4 ≈ 5.8 s（权重读共享上限：
    3.4576 GB/token ÷ 100.6 GB/s = 34.36 ms 理论下限，B 臂
    36.06 已接近，故并发无法突破单 token 带宽墙，只能减少
    题间空隙）。
  - A 臂 MMLU 串行 ≈ 10 × 64 × 10.74 ms ≈ 6.9 s，并发 ≈ 1.7 s。
  - GSM8K 段（串行，256 上限）：B 臂 10 题 × ~100 tok × 36.06
    ≈ 36 s；A 臂 ≈ 10.7 s。两臂合计 ≈ 47 s + MMLU 并发后
    ≈ 7.5 s ⇒ 窗内 ≈ 55 s，满足 Lead 10:52 "1–2 分钟"。
  - 显存/带宽：并发只增加 host 侧 curl 进程，不增加单次请求的
    权重读字节；4bit 臂每 token 0.973 GB（QUEUE #8 F1），并发
    不改变 per-request 字节数，只改变请求重叠度 ⇒ **不构成
    "batch 吞吐"优化**（QUEUE #6 才是那个方向，仍挂起）。

完成判据（冒烟窗五档，仅本窗，非最终 knowledge verdict）：
  - WIN：G1–G5 全过、A/A ≤ 5%、MMLU delta ≤ ±1 题（Lead 10:52
    "差 ≤1 题 ⇒ 大方向没问题"）、GSM8K cause=length 计数 = 0 且
    至少 8/10 能解析出收尾答案、逐 token 门（MMLU）通过。
  - PARTIAL：门全过但 MMLU delta > 1 题 或 GSM8K 解析率 < 80%。
  - INVALID：任一硬门失败（空臂 / A/A > 5% / 逐 token 分叉 /
    样本 sha 不符 / 双锁拿不到）。
  - 最终 knowledge 门的五档判定不在本窗产生（须全样本窗 + doc
    §6 回填，按 doc §2 冻结判据：delta ≥ −3pp / −3~−10pp /
    < −10pp）。
```

## 3. Frozen inputs / 不重开

- Lead 10:52 bug-2 批准原文与 round-165 承接记录：`doc/lfm3-4bit-knowledge-eval.md` §7。
- round-166 双上限接线：commit `408d2858`（已 push 上远端）。
- round-97/99 冒烟窗 receipts（MMLU parsed 0/7 flip、GSM8K 64-token 无效态）：doc §6 逐字。
- 样本 sha 与抽样 seed：doc §5 冻结（MMLU seed 20261120 / GSM8K seed 20261121；子集 seed 20261122 / 20261123）。
- `lfm3-ane-load` 锁（P2 W1，§616 停止自动复探）：本 doc 不触碰，harness 的 hold_locks.py 若发现双锁被占会等待（6 h deadline），与 P2 W1 不冲突（同锁不同持有者，等不抢）。
- QUEUE #6 batch harness（§628 挂起）、98,552,320（待 Lead 裁定）：不触碰。
- `doc/lfm3-native-4bit.md` Gate/判据原文：不改。
