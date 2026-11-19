# LFM3 compiled decode step (whole-step `mx.compile`, no bucketing): launch-gap harvest

Source: `QUEUE.md` item 1 (campaign `lfm3-dig`, base `feat/lfm3-metal`).
Baseline: PR #127 (`apxinf-pipe` 36.07 / 36.40 ms/token, `feat/lfm3-metal` tip
`1ed3f2f3`). PR target: `feat/lfm3-metal`.

The question is specific and falsifiable. The #127 sync-arm decomposition
(`doc/lfm3-pipelined-decode.md`, `doc/receipts/lfm3-pipelined-decode/decomposition.json`)
measured, over 1,030 instrumented decode steps with reconciliation 100.0%:

| stage | ms/token |
|--- | --- |
| `build_ms` (host graph, ~700 MLX ops) | 0.263 |
| `eval_wait_ms` (blocking `stream.eval`) | 37.052 |
| `sample_ms` (cast + argmax + sync read) | 0.353 |
| `callback_ms` (detokenize, IPC, controls) | 0.009 |

The four host stages sum to 0.624 ms/token, but the pipelined schedule recovers
**P = sync − pipe = +1.08 / +1.13 ms/token**. The unexplained remainder
(≈0.46–0.51 ms/token) is attributed in #127 to per-step launch-gap drain: the
eager single-token graph is ~700 `mx` op submissions, and a step's final
kernels can start before the host has finished enqueuing the graph, so the
device periodically drains the command buffer. This ticket tests the fix
pre-registered in `QUEUE.md` item 1: **the whole one-token step as one
`Compiled` (`mx.compile`) callable**, eliminating host-side re-tracing and
op-dispatch gaps *inside* the step, with **no KV bucketing and no weight,
kernel, or shape change**.

Status: **pre-registered (this commit, before any device run).** The decision
rule and thresholds below are frozen from here; amendments are dated
append-only at the end.

---

## 预登记 (pre-registration)

### 1. Arms

| arm | change | switch |
| --- | --- | --- |
| `apxinf-pipe` (baseline) | none; current pipelined decode (`build_graph` eager + `async_eval` lookahead) | unset |
| `apxinf-pipe-compiled` | same binary as baseline; the decode step body is replayed through one MLX `Compiled` callable | `APXINF_LFM3_COMPILED_DECODE=1` |

Mechanism (fixed here): `Lfm3Model::decode` builds the step through
`apxinf_mlx::Compiled::call` instead of calling `build_graph` directly. The
compiled body receives the token id plus every mutable state array (KV keys and
values, conv windows, rotary tables) as explicit `Array` inputs, returns the
logits plus the 24×2+24 = 70 updated state arrays, and MLX replays the captured
graph as one dispatch. First use in a request calls `Compiled::prepare_timed`
once (trace + compile + Metal JIT + one warm eval); the traced profile is
**exact-shape** (`shapeless = false`): any retracing on a later call fails
loudly (`prepare_timed`'s "retraced on replay" error → request error), which
makes a bucketed or shape-shifting arm impossible to run unnoticed.

Effect proof (the 空臂 trap): the worker banner
(`crates/apxinf-serving/src/native_worker.rs:529`) gains a
`decode_compiled=<false|true>` field derived from the same env switch; a
compiled segment whose gateway log lacks `decode_compiled=true` is **void**.
Stream counters `trace_callbacks` are recorded: the baseline accumulates
`trace_callbacks` per step (eager op trace on every call); the compiled arm
must show a flat count across the steady-state steps of a request.

### 2. Cells

S2 decode only, exactly #127's 口径: contexts 1024 and 4096, `max_tokens=128`,
`--engine-timings` on both apxinf arms, `scripts/bench_engines.sh` +
`scripts/bench_engines_report.py`, per-segment corpus seeds (each arm in a
segment shares the seed; distinct text per segment so no prefix-cache hits).
S1 decode-at-8 cells stay excluded (A/A not certifiable). No tier A/B rerun,
no S1/S3/S4/S5, no prefill cells: prefill uses `execute_inner` chunk-wise and
is untouched by this change.

### 3. Order

`A B B A A B` where A = `apxinf-pipe` (baseline), B =
`apxinf-pipe-compiled`: three segments per arm, paired medians per cell (each
segment contributes one median per arm; a cell's paired value is the median of
the three segment medians). The `llama.cpp` arm is **not re-run** — this
ticket's decision is within-apxinf (`P_c = A − B`); llama's numbers in #127's
certified window remain the external reference.

### 4. Notation (per cell, from paired medians)

- **C** = `apxinf-pipe` − `apxinf-pipe-compiled` (ms/token) — the launch-gap
  harvest.
- **C/A** = compiled / baseline.
- **A/A** — worst segmented-median spread per arm per cell (§6).

### 5. Falsifiers (mechanism must actually account for the win)

1. **Token inconsistency (hard gate)** — emitted token ids differ between the
   two arms for any of 4 prompts × 128 tokens ⇒ **INVALID**. This is the
   QUEUE item-1 falsifier verbatim, reusing #127's T1 parity probe.
2. **Kernel-shape drift (hard gate)** — compiled arm's step retraces with a
   different op/shape profile, or the compiler buckets (e.g. KV padding to a
   wider shape) ⇒ **INVALID**. Guarded by `shapeless=false` and the retracing
   failure, plus a gate-0 kernel/launch census (§8).
3. **A/A > 5%** for either arm in either cell ⇒ window **INVALID**.
4. **FALSIFIED (mechanism)** — token-consistent and A/A-clean but
   `C < 0.2·(eval_wait_ms)` i.e. C < 0.2 · 37.052 = 7.41 ms/token is *not*
   the bar; the expected gain is ~0.3–0.5 ms/token, so the falsifier is
   **C ≤ 0.10 ms/token** (the residual is below the A/A-certifiable noise of
   this harness) *or* compiled is slower: the "launch-gap drain inside the
   step" mechanism fails, and §11's diagnosis (which gap actually remains)
   explains the numbers before the next round.
5. **launch census contradiction** — if compiled reduces the per-step host
   submission count but the request total does not fall, the gap is
   GPU-scheduler-side between steps, not launch-gap drain inside the step;
   recorded as FALSIFIED with the census in §8.

### 6. Decision rule (frozen)

- **INVALID** — any of the three hard gates in §5 fires.
- **WIN** — both cells C ≥ 0.25 ms/token, and in all three segments
  `apxinf-pipe-compiled` < `apxinf-pipe` for both cells.
- **PARITY** — both cells 0 < C < 0.25 ms/token (token-consistent, A/A-clean)
  but WIN does not hold.
- **PARTIAL** — C ≥ 0.25 ms/token on one cell only, or 0.10 < C < 0.25 with
  the launch census showing the in-step submission gap was indeed removed:
  the in-step gap is real but not the full mechanism; §11 names what remains
  (expected: inter-step gaps ⇒ next round = overlap between *steps*, or the
  pipeline's extra sync).
- **FALSIFIED** — §5.4 or §5.5 fires.
- Anything else: **INCONCLUSIVE**, all numbers reported.

The queue's expected 0.3–0.5 ms/token is the WIN band's center; the WIN
threshold is set at 0.25 ms/token (half of the lower expectation) so a
half-harvest is still a WIN and the A/A-certifiable noise floor does not
define the verdict.

### 7. Default rule

If the window is not INVALID and WIN or PARITY holds, compiled decode becomes
the **default** greedy decode step, with `APXINF_LFM3_COMPILED_DECODE=0` kept
as the operator escape hatch (same policy as #127's schedule switch; no dead
opt-in path). If PARTIAL, the compiled step stays behind the switch until the
remaining gap is addressed; FALSIFIED/INVALID: the switch and its code are
reverted, only this record remains.

### 8. Gate-0 launch / trace census (before any timing window)

Same locked M4 session, one 64-token request per arm, warm model:

- per-step host MLX submission count (op-level `mx` calls while tracing the
  body) — expected ≈700 baseline vs ≈1 traced step compiled;
- `stream.counters().trace_callbacks` delta per step — expected +~700
  baseline, +0 compiled steady-state;
- total Metal command-buffer / kernel launches per request if the MLX
  counters expose them (`StreamCounters`), else `MX_METAL_CAPTURE` /
  Metal System Trace is **not** required this round: if the counters do not
  exist, write "launch count: 机制未知" per discipline rule 5 rather than
  substituting a proxy;
- kernel shapes: dump the compiled step's traced op names and shapes and
  diff against the baseline's last decode step. Any shape change ⇒ §5.2.

Reconciliation sanity: `APXINF_LFM3_DECODE_STAGES=1` decomposition on the
compiled arm must still reconcile ≤10% of its engine `decode_ms/token`
(build/eval/sample/callback means), as in #127 §4. If the compiled arm's
`build_ms` collapses to ~0 (expected: graph build is trace-once + replay) and
`eval_wait_ms` does not fall by a comparable amount, that is the §11 data.

### 9. Tests (before device runs)

Real BF16 checkpoint, Metal-locked, `#[ignore]`-marked, run with
`APXINF_LFM3_TEST_MODEL=<snapshot dir>` on the M4 (same harness as #127's
T1/T2/T3, same `--test-threads=1`):

- **C1 token equality** — 4 prompts × 128 tokens, `APXINF_LFM3_COMPILED_DECODE`
  unset vs `=1`, identical token ids (the §5.1 gate).
- **C2 budget/EOS/cancel edges** — budget 1, budget 8, synthetic first-token
  EOS, cancel right after first publish: compiled == sync (the
  `APXINF_MLX_SYNC_DECODE=1` reference), `position == P+N−1` on rollback; the
  conv-window restore must still apply to the discarded lookahead step.
- **C3 continuation state** — after a compiled run with the extra lookahead
  step rolled back, all KV rows `[0, position)` and all 24 conv windows are
  byte-identical to a sync run's, and a 16-token continuation matches sync
  token-for-token.
- **C4 retracing guard (negative control)** — force a second profile by
  running compiled decode at two different prepared capacities in one process
  (or a unit-level `Compiled` body with a differing input shape): the call
  must error "retraced on replay" / the request must fail loudly, and a
  `#[cfg(test)]` seam that flips `shapeless=true` must make the same run
  *succeed with different tokens or a different traced op count* — the proof
  that the no-bucketing gate is load-bearing. Output pasted in the PR.
- CPU-side pipeline tests in `crates/apxinf-model/src/mlx_decode.rs` (the
  four `cpu_pipeline_*` tests) must still pass unchanged: the schedule driver
  is not modified by this change.

### 10. Window discipline

M4 (`haiyan-mac-mini`) only, per RUNNER.md: `SDKROOT=
/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk`, `MLX_ROOT=~/mlx-sdk`,
both Metal locks taken in order (`/tmp/enginetailor-metal.lock` then
`~/.cache/enginetailor/metal-measure.lock`), **waited, never stolen**;
`pmset displaysleepnow` + `caffeinate -si` (no `-u`) before the window;
engine-reported timings (`--engine-timings`); swap / thermal / load recorded
before and after; device budget ≤45 min (gate-0 census + ABBA + parity probe).
M2 Air is not measured; nothing is extrapolated across hosts. Cross-window
absolute numbers are not comparable (§11 caveat 5 of #127): this ticket's
verdict rests only on within-window paired medians.

### 11. Implementation limits

- **No bucketing**: KV shapes are exactly `[ctx, heads, head_dim]` per layer
  for every ctx; the compiled profile is exact (`shapeless=false`). A padded
  or bucketed KV makes C3's byte-equality fail and §5.2 INVALID.
- **No numerics change**: identical `build_graph` body (same `run_layer`,
  same `fused_conv_window`, same RoPE `cast/mul/neg/concat/mul/add` chain,
  same `weights.project` matmul) — MLX compile wraps the *existing* graph;
  the tiling/accumulation hazard from PR #128 applies only if shapes change,
  and §5.2 blocks that.
- **No kernel rewrite, no wider matmul** (QUEUE item 2's scope; distinct
  ticket).
- Harness changes only: env switch, worker banner field, the decomposed
  stage instrumentation is reused as-is. `bench_engines.sh` /
  `bench_engines_report.py` verdict logic and gates untouched.
- Prefill (`execute_inner`, chunked, `len>1`) is not compiled this round
  (QUEUE item 3's scope); the compiled callable exists only for the
  single-token device decode path.

---

## 2026-11-19 amendment — Relation to #115

This section is appended (dated, as §"Status" allows) to answer the lead
note's requirement that the ticket's relation to #115 be explicit, rather than
appearing to re-open a door #115 already closed.

What #115 measured (per the lead note): "整步编译 decode **+0.7%**（无收益）",
with KV-position **bucketing** within ±2% of no-bucketing. That ticket closed
the combination of whole-step `mx.compile` *together with* position bucketing
as an optimization direction.

What is being tested here: the **same whole-step `mx.compile` on top of the
pipelined greedy schedule (#127), without bucketing** (`shapeless = false`,
§11). This is a different hypothesis because the substrate changed: #127's
pipeline already hides most of the step's host work behind the device's
previous kernels, and the residual question (§"The question is specific") is
whether a *per-step re-trace / intra-step launch gap* survives the pipeline
(`P = sync − pipe` leaves ≈0.46–0.51 ms/token unexplained by the four host
stages). Bucketing is explicitly not part of this arm: the retracing hard gate
(§5.2) is what keeps that #115 ingredient out.

This ticket therefore **cites #115's finding as the prior expectation** (per
the lead note: expect PARITY or NEGATIVE, and if so record the five-level
verdict with numbers and move to QUEUE item 2 without iterating the window).
The one thing it adds over #115's record is the pipeline interaction and the
gate-0 census (§8) that lets the verdict name which gap, if any, remains.

---

## Results

*(window pending — appended after the locked M4 run; only dated amendments to
the pre-registration above may appear before the verdict.)*

---

## 2026-11-19 17:37 CST — Verdict: **INCONCLUSIVE** (amended per lead correction 17:31; supersedes the 17:20 FALSIFIED label)

The 17:20 verdict of FALSIFIED was wrong and is withdrawn. An arm that never
ran to a token result and never ran a timing window cannot be FALSIFIED
(§5.4/§5.5 require "token-consistent and A/A-clean but mechanism disproved").
Per §5.6 this ticket is **INCONCLUSIVE**. The lead's requested substance:

- **This ticket produced no new mechanism evidence.** Nothing was measured on
  this branch for this arm.
- **#115's prior stands — but it is a prior, not this ticket's measurement**:
  whole-step `mx.compile` on decode measured +0.7% (no benefit), buckets
  within ±2%. No result of this ticket refutes or confirms it.
- **The "launch-gap drain" hypothesis (≈0.46–0.51 ms/token unexplained by the
  host four stages) is unverified** — neither refuted nor confirmed.
- **Why no arm ran** (both causes true): (1) the compiled closure over the
  decode step's 70 state inputs never reached a state that both compiled and
  passed a C1-shaped harness — an integration defect beyond the one-line `?`
  fix (`2512f50c`), reverted in round-14 → commit `ddce3ffc`; (2) M4's two
  Metal locks were continuously held by an unrelated campaign
  (`ef3_round.py`, PIDs 28706 → 34945 → 41663 → 48188), "等，不抢", so C1
  itself, the §8 census, and the timing window never executed.
- **The revert is retained** (`ddce3ffc`): code that never ran should not
  merge; the reverted tree was compile-checked clean (0 errors).

Mechanism arithmetic: no ms/token numbers exist for this arm, so none are
invented. Gate census: launch count / trace_callbacks delta — **机制未知
(never measured)**. Byte accounting unchanged (same `build_graph` body, no
shape/weight change): 3.4576 GB/token ÷ 100.6 GB/s ⇒ 34.36 ms/token floor.

### What ran and what did not

- **Ran**: implementation (`6179da19`, compiled decode scaffold), the C1
  token-equality gate written (`843d6d0a`), a `Result`-propagation fix found by
  the first `--features mlx,coreml` compile (`2512f50c`, `model.rs` lm_head
  project), `cargo check` / `cargo test --no-run` on M4 — all clean
  (`CARGO_EXIT=0`, C1 linked in the test binary).
- **Did not run**: C1 itself, the §8 gate-0 launch census, and the A/B
  `A B B A A B` timing window. The C1 runner waited on the two Metal locks for
  the full session; the locks were held continuously by an unrelated campaign
  (`ef3_round.py`, PIDs 28706 → 34945 → 41663 → 48188), per "等，不抢".
- **Why the arm was dropped rather than awaited**: round-14 found the compiled
  closure path had an unresolved integration defect beyond the one-line
  `Result` fix — the `Compiled` body over the decode step's 70 state inputs
  could not be brought to a compiling, behavior-verified state within the round
  budget, and the session reverted the integration. The reverted tree was then
  compile-checked (`cargo check -p apxinf-model -p apxinf-serving
  --features mlx,coreml`, 0 errors, `~/lfm3-dig-run/check-revert.log` on M4).

### Basis of the verdict (mechanism arithmetic)

No ms/token numbers exist for this arm (see the amended verdict above), so
none are invented. Two facts explain why no arm ran, without being
disguised as mechanism findings:

1. **Prior measurement (#115)**: whole-step `mx.compile` on decode measured
   **+0.7% (no benefit)**, with and without position bucketing inside ±2%.
   This ticket's only delta over #115 was the pipeline substrate (#127) and
   the §8 census; neither was executed. That #115 prior therefore **still
   stands, but as a prior** — this ticket neither refuted nor confirmed it,
   and it is not evidence produced by this arm.
2. **Unresolved integration defect**: the compiled arm's closure never reached
   a state that both compiled and passed the C1-shaped harness. The §5.2
   retracing hard gate was never exercised either — the no-bucketing property
   could not be verified on a path that does not run.

Gate census: **launch count / trace_callbacks delta — 机制未知 (never measured).
Byte accounting unchanged (same `build_graph` body, no shape/weight change) —
3.4576 GB/token against the 100.6 GB/s ceiling ⇒ 34.36 ms/token floor.**
The §11 "no numerics change" property is preserved by the revert: `model.rs`
returns to the eager pipelined body, and the eager-vs-eager parity of the
remaining branch is the only thing the deleted C1 would have shown.

### Residual branch state

- Reverted on branch: `model.rs` compiled path, C1 test, `native_worker` banner
  field, `Weights`/`causal_conv` direct handles — commit `ddce3ffc`.
- **Scaffolding left in place** (not part of this revert, documented here so
  the PR reviewer can see it is deliberate): `apxinf_mlx::Compiled` and its
  `prepare_timed` exact-shape/retrace guard, `causal_conv` array plumbing,
  `weights` head-array access, and this pre-registration document. The QUEUE #2
  PR may keep or strip these; they are inert (no call sites reference them
  after `ddce3ffc`, verified by grep `COMPILED_DECODE` = 0 matches).

### Disposition

QUEUE #1 is closed as **INCONCLUSIVE (not measured)**. Per discipline rule 7
("被否就记录并进下一项") the campaign proceeds directly to **QUEUE #2 —
GPU-side kernel efficiency (shape-preserving lm_head + elementwise fusion)**,
pre-registered in `doc/lfm3-lm-head-custom-kernel.md`. QUEUE #1's question
(launch-gap drain ≈0.46–0.51 ms/token) remains open for a future item if a
window ever becomes affordable; it is *not* settled by this record.
