# QUEUE #3 — 短 prompt prefill 加速（128/512 token）

## Hypothesis

短 prompt（128/512 token）prefill 比 llama.cpp 慢 8–12%；2048 token 基本持平。
2048 追平 ⇒ 瓶颈不在稳态吞吐，而在**每 prompt 的固定开销**（24 个 ConvBlock
chunked-prefill 分块调用 × layer 的图构建/launch 序列 + conv cache 状态清理 + 多次
`eval_async`/stream wait）在 N≤512 时摊不平。

## Gate（Gate:）

1. **Token 硬门（§5.1，不可谈判）**：改动后 4 prompts × 128 token，改动臂 greedy
   输出 vs baseline 逐 token 相等；**一个 token 不同 ⇒ INVALID**，窗口不可解读
   （与 T1/C1/K3 同款 harness：同一进程跑 baseline 取 expected、再跑改动臂，
   `--test-threads=1`）。
2. **形状门**：Conv1D/ConvBlock 的输出 shape、kernel width、padding、chunk 划分
   对外契约完全不变；padded / widened / bucketed 形状 ⇒ INVALID（#128 教训）。
3. **A/A ≤ 5%**（硬纪律 4）：每臂每 cell 分段中位最坏 spread > 5% ⇒ 窗口作废
   重跑。**S1 prefill 128/512 不可引用 #121 的 S1 短生成 cell**（其 A/A 曾达
   5.02%，lead note 明确禁用）；A/A 必须用**本 ticket 自己**的 128/512/2048 三档
   各自分段中位测得。
4. **空臂门**：`APXINF_LFM3_PREFILL_FAST=1` 开关关时（baseline 臂）改动零成本，
   worker banner `prefill_fast=<false|true>` 与同一 env 同源；gate-0 census 确认
   baseline 臂改动路径 0 dispatch。
5. 任一硬门触发 ⇒ INVALID，按 lead note 直接记录五档结论进 QUEUE #4（自选方向，
   须先量化），不反复迭代窗口。

## Decision:

**开关 + 实现方向**：`APXINF_LFM3_PREFILL_FAST=1`（默认关），worker banner 增
`prefill_fast` 字段。**唯一第一轮实现**：把 prefill 的 24 层图在形状确定后**只
构建一次、复用（cache last graph per static shape）**，而不是每个 prompt 从零
`build_graph`。这是纯 host 侧改动：
- **不动 GPU kernel / 累积顺序 / weight 读序** —— 按构造 token 等价（与 #115
  关掉的编译/分桶无关；我们不 `Compiled`、不分桶、不重排 eval）；
- 只减少图构建与 launch 编排的 host 开销，直接对应"2048 追平、短 prompt 固定
  开销摊不平"的假设；
- **不做**：合并 ConvBlock chunk（会改 chunk 边界 ⇒ risk 改变 conv cache 状态
  更新时序，留给 census 后证据说话）、不做 batch/prefix cache。

（若第一轮 cache-last-graph 实现被证明不可行（state 数组每步动态），则回退到
第二方案：prefill 分块时**合并相邻层的若干 elementwise/小 kernel 为一个图**，
但必须经 token 硬门。第二方案属下一个 commit，不静默替换第一方案。）

**关系（Relation to #115）**：#115 关的是"整步编译 decode + 位置分桶"。本 ticket
**不分桶、不 Compiled、不改 decode**，只复用 prefill 图的 host 构建；decode 路径
完全不动，不重开 #115 的门。

## Cells / Order / Gates

- **Cells**：S1 prefill **128 / 512 / 2048** token 三档，各跑 128 token 生成后
  取 prefill 段 p50（与基线 8–12% 差距的度量同口径：`llama-server` vs
  `apxinf-serve` 已有 receipts 的比值复现）；每档**分段中位**，不用 S1 短生成 cell。
- **Order**：`A B B A A B`（A = baseline `PREFILL_FAST` unset，B = =1），
  同一 prompt 集，双臂交替，单进程内 baseline 先取 expected 用于 token 门。
- **硬门**（任一 ⇒ INVALID）：① token 不一致；② shape 漂移；③ A/A > 5%
  （任一 cell）；④ census 显示 baseline 臂有改动路径 dispatch（空臂门破）。
- **WIN**：S1 prefill p50 比值 < 1.05（即差距从 8–12% 收到 <5%），双 cell
  全 segment 同向，token/A/A 干净。
- **PARITY**：token 干净但比值 ≥ 1.05（假设无效，实测证伪 host 构建开销主因）。
- **FALSIFIED**：① launch/census 显示 host 构建开销在短 prompt 下本就不是主项
  （机制算术：host 四项和测得占 < 预期 Δ）；② 或"kernel 合并"类改动省了 launch
  但 prefill p50 不降（机制未知则如实写）。

## Census §8

- host 侧：`StreamCounters`（`apxinf-mlx/array.rs:153` 的 `trace_callbacks /
  eval_calls / uploads / downloads`）+ 自插桩 prefill host 时间（图构建 vs eval）
  —— 不需要 Metal 锁的编译阶段即可写插桩代码。
- per-kernel GB/s：M4 上 `xctrace` 不可用（CLT only）、MLX SDK 头无 per-kernel
  timer ⇒ 写"机制未知"，不用替代指标冒充（round-18 #2 census 注记同款纪律）。

## Tests

- **P1** token 门（与 T1/C1/K3 同 harness）：4 prompts × 128 token，baseline vs
  `PREFILL_FAST=1` 逐 prompt 相等。entry/exit 都复位开关防污染。
- **P2** 空臂门：开关关时 `prefill_fast` helper 返回 false，插桩计数 0。
- **P3** cache 正确性：同进程连跑两个不同 prompt（不同 prefill 长度 128 与 512），
  第二个 prompt 结果必须与全新进程单跑结果 token 相等（防 stale cache）。
- `cpu_pipeline_*` 四测必须原样过（不碰 schedule driver）。

## Relation to QUEUE #1 / #2

- #1（compiled decode）判定 **INCONCLUSIVE**（lead correction 17:31）；本 ticket
  引用其"未测量"事实，不继承"compiled 已死"框架；我们不用 Compiled。
- #2（custom kernels）判定 **INVALID**（K3 token 门失败）；kkernel switch/K1–K4
  留分支作 receipt，**默认关、空臂、不影响 baseline**；本 ticket 不依赖其任何代码。

## Results

*(window pending — appended after the locked M4 run; only dated amendments to
the pre-registration above may appear before the verdict.)*

### 2026-11-19 19:45 Verdict — INVALID (§5.1 token gate)

**Verdict: INVALID.** The P1 hard gate fired on prompt 1. No §8 census, no
A/A window, and no prefill p50 (128/512/2048) measurement were run, so there
are no ms/token numbers to report; none are fabricated.

**What ran.** P1, P2 and P3 all executed on the M4 (`Apple M4`), double
Metal lock held (acquired 19:45:04, released 19:45:59, `locks-p.log`),
`APXINF_LFM3_TEST_MODEL=~/lfm3-spec/weights-local/LFM3-1.7B`, `--test-threads=1`,
`run-p-tests.sh` (PID 90533; waiting for the lock since round 27).
Exits: `p1=101 p2=0 p3=101`.

- **P1 HARD GATE FAIL ⇒ INVALID.** `p1_fast_prefill_greedy_matches_baseline_token_for_token`
  panicked at `model.rs:1944` on **prompt 1**: fast arm (left) = 128× `99` (a
  single token repeated to degeneration), baseline arm (right) =
  `82906, 8, 229, 25, 22, 229, 25, 22, …` repeating four-token loop. Test
  runtime 35.44 s. This is exactly the §Gate.1 trigger ("one token differs ⇒
  INVALID"). The armed arm was confirmed armed (`prefill_fast=true`, P2
  proved the unset arm stays empty), so the gate fired on a live path, not an
  empty arm.
- **P2 PASS (empty-arm gate).** With `APXINF_LFM3_PREFILL_FAST` unset:
  `prefill_fast_enabled()` false, model has no `prefill_fast` flag, and after
  32 generated tokens `prefill_cache.is_none()` — the changed path dispatched
  zero times. Test ok, 2.08 s.
- **P3 FAIL (stale cache).** `p3_cached_second_prompt_matches_fresh_run`
  panicked at `model.rs:2002` with **"P3: second prompt in-process diverged
  from a fresh run — stale cache"**: warm model cache-hit arm (left) =
  `33257, 989, 21, 26121, 73914, 63372, …` loop; fresh single run (right) =
  `354, 426, 1211, 349, 11386, 59487, …`. This independently confirms the
  implementation defect already flagged in round-25 audit note 1: the cached
  graph's input ids are **not rebound** to the new prompt, so a same-shape
  cache hit returns the previous prompt's logits.

**Mechanism arithmetic.** No ms/token delta, because the armed arm never
reached the census or timing window. Host-side graph-build vs eval split,
per-kernel GB/s, and launch count: **机制未知 (never measured — INVALID at the
token gate before any census)**. The byte budget is unchanged (3.4576 GB/token
and the 34.36 ms/token roofline floor are untouched; this arm does not alter
GPU kernels).

**Why INVALID and not the other tiers.** §5 defines WIN/PARITY/PARTIAL/
FALSIFIED against a *running, token-consistent* arm inside a clean A/A window.
An arm that fails the token gate is §5.1 INVALID regardless of speed, and no
window may be interpreted. P1 is the hard gate; P3's stale-cache failure is
the root cause, not a separate verdict. Per the lead's standing instruction,
do not iterate the first scheme further — the second scheme (merging adjacent
elementwise kernels) is also gated on tokens and is not pursued here.

**Disposition.** QUEUE #3 is closed as **INVALID**. The cache-last-prefill
arm stays on the branch as the receipt (`APXINF_LFM3_PREFILL_FAST` switch,
P1–P3 tests); it is off by default (empty-arm, P2 PASS) and does not affect
the baseline path. No re-windowing of this arm is authorized. Proceed
directly to **QUEUE #4 (self-selected, requires prior quantification)** in
the next round: pre-register a candidate only after quantifying it (open
leads include the still-unverified launch-gap drain from #127, INCONCLUSIVE
QUEUE #1; kernel efficiency, INVALID QUEUE #2; prefill, this INVALID).
