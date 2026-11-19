# QUEUE #5 — 结构重组类负结果收口报告 (structural-reorg negative-results closeout)

Source: `QUEUE.md` (v2, 2026-11-19 21:30 Lead update) item 5.
Campaign: `lfm3-dig`, base `feat/lfm3-metal`.
Written offline on M1; **no device, no Metal lock, no new measurement.**

Lead decision (2026-11-19 21:30) that this report answers: *不要再开第五个
"改执行图结构"的方向。五个尝试（含 #128）已经证明这条栈上任何图结构改动
都会扰动 token，而剩余 roofline 只有 ~2 ms。* This report is the
institutional record of those five attempts so the rule does not have to be
re-learned per campaign.

**The five attempts**: PR #128 (merged GEMV), QUEUE #1 (whole-step
compiled decode), QUEUE #2 (custom lm_head GEMV + fused silu_mul), QUEUE #3
(cache-last prefill graph), QUEUE #4 (eval/sync split probe). QUEUE #1–#4
are this branch's records (`agent/lfm3-dig`, PR #132); #128 predates the
branch and is cited from its own PR plus the amendment at
`doc/lfm3-lm-head-custom-kernel.md:249`.

---

## 1. Per-attempt ledger

### Attempt A — PR #128: 合并 GEMV (merged GEMV)

| field | value |
| --- | --- |
| **Hypothesis** | Concatenating weight tiles / issuing a wider matmul merges several GEMV dispatches into one, cutting launch count. |
| **What changed** | Execution-graph shape: concatenated weights ⇒ wider matmul shape. |
| **Gate that fired** | **Token hard gate → INVALID.** Greedy output changed by at least one token. |
| **First divergence** | First greedy token (exact token ids not recorded in this branch; #128's own PR is the receipt). |
| **Why** | Concatenating weights / widening the matmul ⇒ MLX picks a **different tiling** ⇒ different **accumulation order** ⇒ BF16/FP32 rounding path drifts ⇒ argmax flips a token. Mechanism stated verbatim at `doc/lfm3-lm-head-custom-kernel.md:254-256` and `RUNNER.md:8` ("拼接权重/改变 matmul 形状 ⇒ MLX 换 tiling ⇒ 累积顺序变 ⇒ 数值漂移"). No speed number is meaningful: the window is void the moment the token gate fires. |
| **Receipts** | PR #128 (qhy991/apxinf-metal); `RUNNER.md:8`; `QUEUE.md:8`; `doc/lfm3-lm-head-custom-kernel.md:249-262` (Relation to #128). |

### Attempt B — QUEUE #1: 整步 `mx.compile` (whole-step compiled decode)

| field | value |
| --- | --- |
| **Hypothesis** | The ~700-op single-token eager graph re-traces per step, leaving an intra-step launch gap of ≈0.46–0.51 ms/token (the `P = sync − pipe` remainder #127 could not attribute to its four host stages); one `Compiled` callable should erase it. |
| **What changed** | Execution graph lifecycle: the whole decode step replayed through one MLX `Compiled` callable with 70 explicit state inputs, exact-shape (`shapeless=false`), `APXINF_LFM3_COMPILED_DECODE=1` switch + banner `decode_compiled`. |
| **Gate that fired** | **None. → INCONCLUSIVE (not FALSIFIED, per lead correction 17:31).** The arm never ran. |
| **First divergence** | Not observed — C1 token gate was written (`843d6d0a`) but never executed. |
| **Why** | Two causes, both true: (1) the compiled closure over the step's 70 state inputs **never reached a state that both compiled and passed a C1-shaped harness** — an integration defect beyond the one-line `?` fix (`2512f50c`), reverted round-14 → `ddce3ffc`; (2) M4's two Metal locks were held continuously by an unrelated campaign (`ef3_round.py`, PIDs 28706 → 34945 → 41663 → 48188), "等，不抢", so C1, the §8 census and the timing window never ran. A never-run arm has no token result and no A/A ⇒ cannot be FALSIFIED (§5.4/§5.5 require a *running, token-consistent, A/A-clean* arm). |
| **Receipts** | `doc/lfm3-compiled-decode-step.md` §Results (17:37 verdict INCONCLUSIVE); commits `6179da19`, `843d6d0a`, `2512f50c`, `ddce3ffc`; M4 logs `~/lfm3-dig-run/check-revert.log`, `c1-outer.log`; lead correction 17:31 in `STATUS.md`. |

### Attempt C — QUEUE #2: 自写 lm_head GEMV + silu 融合 (shape-preserving custom kernels)

| field | value |
| --- | --- |
| **Hypothesis** | lm_head's tied table is 524,288,000 B = 15.1% of the 3.4576 GB/token bytes; a self-written shape-preserving GEMV (32-lane → 32 output columns, one FP32 accumulator per output, ascending k) plus an F32 elementwise `silu(g)*u` fusion could cut that cost without touching shapes. |
| **What changed** | *Which kernels run inside the existing eager graph*: `lm_head_gemv.metal`, `silu_mul_fused.metal`, `KKernel` switch `APXINF_LFM3_KKERNEL=1` + banner `kkernel`. Graph shape unchanged (K4 negative controls prove it). |
| **Gate that fired** | **K3 token gate → INVALID.** Prompt 0: first generated token diverged — armed `128× 37179` (degenerate repeat) vs baseline `20, 229, 1217, 24, …`. |
| **First divergence** | **The first generated token itself** — i.e. inside the armed lm_head projection, not a downstream accumulation. |
| **Why** | Same shape, **different reduction form**: the custom kernel accumulates `float acc += float(x[k])*float(w[k])` in ascending k, while MLX's matmul uses its own reduce order ⇒ logit-level deltas (K1 `max_abs_diff=2.260e-4`; K2 fused silu `1.562e-2`, `1113/2048` exact) that are large enough at this model's decision margins to flip argmax. K1/K2 deltas were §9 K1 amendment candidates (logit-level only), but the token gate is the hard gate and it failed — the amendment question is moot. The shape gate (K4) was load-bearing and *passed*; it was necessary but not sufficient. |
| **Receipts** | `doc/lfm3-lm-head-custom-kernel.md` §Results (17:58 verdict INVALID); commits `f2898d24`, `5cf3d9ac`, `cbecb9ae`, `ad52ef25`; M4 logs `~/lfm3-dig-run/k-runner-outer.log`, `locks-k.log`. |

### Attempt D — QUEUE #3: prefill 图 cache-last 复用 (cache-last prefill graph)

| field | value |
| --- | --- |
| **Hypothesis** | 128/512-token prompts are 8–12% slower per token than 2048 ⇒ a per-prompt fixed **host** cost (24-layer chunked-prefill graph construction + conv state bookkeeping) is not amortized; reusing the last prefill graph per static shape should remove the build, not the eval. |
| **What changed** | Execution graph **lifetime/input binding**: `PrefillCache` stores the last built graph + state snapshot; `APXINF_LFM3_PREFILL_FAST=1` + banner `prefill_fast` re-`eval`s the cached graph on a same-shape hit. Pure host-side; no kernel/weight/decode change. |
| **Gate that fired** | **P1 token gate → INVALID.** Prompt 1: fast arm `128× 99` vs baseline `82906, 8, 229, 25, 22, 229, 25, 22, …`. P2 (empty-arm) PASS; P3 (stale-cache canary) FAIL. |
| **First divergence** | Prompt 1, first generated token. |
| **Why** | A cached graph's **inputs are the ones it was built with**. On a same-shape hit the new prompt's ids were never rebound into the captured graph, so the arm re-evaluated the **previous prompt's logits** (`P3`: warm cache-hit `33257, 989, …` vs fresh `354, 426, …`). Reusing a graph is "algebraically" the same forward pass but **semantically** a different function of the new input ⇒ token divergence at the first opportunity. Round-25 audit note 1 had flagged this exact risk pre-run; P1/P3 confirmed it. |
| **Receipts** | `doc/lfm3-short-prompt-prefill.md` §Results (19:45 verdict INVALID); commits `049d4d20`, `66d55517`, `dc071431`, `c476f7af`; M4 logs `~/lfm3-dig-run/p-runner-outer.log`, `locks-p.log`. |

### Attempt E — QUEUE #4: eval/sync 拆分探针 (async_eval/sync split probe)

| field | value |
| --- | --- |
| **Hypothesis** | The 0.46–0.51 ms residual is launch-gap drain *during* the ~700-op enqueue: moving the host wait boundary (`async_eval` + explicit `synchronize` instead of blocking `eval`, same `required`, same `Stream`, same graph) should let the host submit without being parked behind the device. |
| **What changed** | Only *where the host waits* inside `execute_inner`'s single-token eval — the minimal-invasion probe (QUEUE #4 §机制). Switch `APXINF_LFM3_DECODE_ASYNC_EVAL=1` + banner `decode_async_eval`; schedule split amendment (Window S = sync, decisive; Window P = pipelined, census-only). |
| **Gate that fired** | **Window S → INVALID**, two independent hard-gate fires: (1) §1 effect proof — **all 8 banners are the 3-field line** `decode_schedule=sync overlaps_callbacks=false decode_stages=true`, the `decode_async_eval` field is **absent** (stale 16:04 binaries; the runner only builds `if [ ! -x "$SERVE" ]`, so no build ran before the window ⇒ Arm B was an empty arm, A vs A); (2) §5.3 A/A — A(1024) segment medians `40.0336 / 37.4735 / 37.4300` ⇒ spread **6.95% > 5%**. |
| **First divergence** | None. (Unit-level E1 *passed* the token gate on the freshly-compiled armed code: 4 prompts × 128 token, tokens equal. The window's INVALID is effect-proof + A/A, not tokens — which is why it is recorded as **机制未知 with a negative signal**, not FALSIFIED: the census's `eval_wait_ms` *rose* +0.138 under a no-op arm, so it cannot be read as mechanism.) |
| **Why** | The effect proof exists precisely for this: a banner field sourced from the same env as the switch makes "the arm actually did the thing" checkable before any timing is read. Without it, a stale binary silently converts a probe window into an A/A window. A/A then independently voids the window per RUNNER §4 (no median-massaging, no rerun-to-fix). §7 default rule: probe ⇒ revert switch + banner + seam (`1922c77f`), keep record only. |
| **Receipts** | `doc/lfm3-decode-eval-sync-split.md` §Results (21:20 Window S verdict); commits `5880b74b`, `813c674d`, `85a9ddf7`, `c3897858`, `2fa2c483`, `1922c77f`; M4 logs `~/lfm3-dig-run/window-s/` (`window-s.log`, `raw/`, `logs/decomp-census*.json`); PR #132. |

---

## 2. Cross-attempt pattern

| attempt | token gate | empty-arm gate | A/A | verdict |
| --- | --- | --- | --- | --- |
| A #128 merged GEMV | **FAIL** (1st token) | n/a | void | **INVALID** |
| B QUEUE #1 compiled | not run | not run | not run | **INCONCLUSIVE** |
| C QUEUE #2 custom kernels | **FAIL** (1st token) | PASS (`kkernel=true`) | void | **INVALID** |
| D QUEUE #3 prefill cache | **FAIL** (prompt 1) | PASS (P2) | void | **INVALID** |
| E QUEUE #4 eval split | PASS (unit E1) | **FAIL** (banner missing) | **6.95%** | **INVALID** |

Three distinct failure *classes*, one shared lesson:

1. **Wrong reduction / wrong tiling** (#128, QUEUE #2) — same declared shape,
   different accumulation order ⇒ logit drift ⇒ token flip. The most common
   and most seductive class, because the change *looks* arithmetic-only.
2. **Wrong input binding / wrong graph semantics** (QUEUE #3) — a graph is a
   function of the inputs it was *built* with; caching or reusing it without
   rebinding inputs silently evaluates the wrong function.
3. **Wrong proof of effect + noisy window** (QUEUE #4) — even a
   numerics-neutral host-side change is unmeasurable unless the banner proves
   the armed path ran and the A/A floor holds. (QUEUE #1 is the *absence* of
   all three: never ran ⇒ INCONCLUSIVE, not FALSIFIED.)

Notably, **shape preservation was necessary but not sufficient**: QUEUE #2's
K4 shape gate passed while K3's token gate failed. And **host-only changes
are not automatically safe**: QUEUE #3's cache reuse was host-side and still
failed the token gate.

---

## 3. 可复用规则 (reusable rule)

> **代数等价 ≠ token 等价。任何结构改动先跑 token 门，再谈速度。**
> *Algebraic equivalence is not token equivalence. Run the token gate before
> the speed conversation — on any structural change.*

Expanded into operational form for this stack:

1. **Token gate is the hard gate, and it comes first.** Any structural change
   (graph shape, matmul shape/tiling, weight layout, kernel identity,
   reduction order, graph caching/reuse, compile/bucketing) is
   **presumptively token-unsafe** until 4 prompts × 128 greedy tokens match
   the baseline arm-for-arm. A speed number from a token-divergent arm is
   void, not "fast".
2. **Prove the effect before measuring it.** Same-source banner + switch +
   **rebuilt binaries** (assert the new banner field in every segment's
   gateway log; a missing field ⇒ empty arm ⇒ void). A stale binary turns a
   probe into an A/A run (QUEUE #4).
3. **A/A is per-ticket.** Reuse of another ticket's cell is forbidden where
   that cell's A/A was uncertifiable (#121's S1 short-generation cell, 5.02%,
   may not be cited; QUEUE #3/4 had to carry their own). Worst segmented-
   median spread > 5% ⇒ window void; no median-massaging, no
   rerun-to-fix-the-number (RUNNER §4).
4. **An arm that never ran cannot be FALSIFIED** — only INCONCLUSIVE. Gates
   (token/A/A/mechanism) require a *running* arm; #115 and #547 were closed
   by real measurements, and this campaign paid for blurring that line once
   (QUEUE #1's withdrawn 17:20 FALSIFIED label, corrected 17:31).
5. **When a token gate fires, stop.** Record the five-tier verdict with the
   raw numbers and move to the next QUEUE item; do not iterate the window
   (RUNNER §5, lead notes #2 and 21:30).

**Corollary for what remains**: with structural reorganization closed and
~2 ms of roofline left (34.36 ms/token floor at 100.6 GB/s on 3.4576
GB/token; measured 36.07/36.40 ms/token), the only remaining directions are
ones that *cannot* touch single-trajectory numerics: batch throughput
(QUEUE v2 #6, whose hard gate is per-request token equality vs batch=1) and
non-generation-path overheads (QUEUE v2 #7: load time, tokenizer,
gateway round-trip). A future structural idea must first answer: *"what in
this change could alter a single token's value, and how do I prove it
doesn't, before I time it?"* — if the answer is anything other than a
constructive argument **plus** the token gate receipt, the idea is closed
by this report.

---

## 4. Receipts index

| attempt | doc (this branch) | key commits | M4 logs / external |
| --- | --- | --- | --- |
| A #128 | `doc/lfm3-lm-head-custom-kernel.md:249-262` | PR #128 (pre-branch) | PR #128; `RUNNER.md:8`, `QUEUE.md:8` |
| B QUEUE #1 | `doc/lfm3-compiled-decode-step.md` §Results | `6179da19`, `843d6d0a`, `2512f50c`, `ddce3ffc` | `~/lfm3-dig-run/check-revert.log`, `c1-outer.log` |
| C QUEUE #2 | `doc/lfm3-lm-head-custom-kernel.md` §Results | `f2898d24`, `5cf3d9ac`, `cbecb9ae`, `ad52ef25` | `~/lfm3-dig-run/k-runner-outer.log`, `locks-k.log` |
| D QUEUE #3 | `doc/lfm3-short-prompt-prefill.md` §Results | `049d4d20`, `66d55517`, `dc071431`, `c476f7af` | `~/lfm3-dig-run/p-runner-outer.log`, `locks-p.log` |
| E QUEUE #4 | `doc/lfm3-decode-eval-sync-split.md` §Results | `5880b74b`, `813c674d`, `85a9ddf7`, `c3897858`, `2fa2c483`, `1922c77f` | `~/lfm3-dig-run/window-s/` (see §1.E) |
| campaign record | — | branch `agent/lfm3-dig`, PR #132 | `STATUS.md` rounds 1–39 |

*This document is offline analysis of existing receipts; it introduces no
new measurement and requires no Metal lock.*
