# QUEUE #4 — decode-step `eval` 拆分：launch-gap 假设的直接测量（minimally-invasive census）

Source: `QUEUE.md` item 4 (campaign `lfm3-dig`, base `feat/lfm3-metal`).
Baseline: PR #127 (`apxinf-pipe` 36.07 / 36.40 ms/token, `feat/lfm3-metal`
tip `1ed3f2f3`, our branch carries it at `origin/feat/lfm3-metal`). PR target:
`feat/lfm3-metal`.

The question is specific and falsifiable. #127's sync-arm decomposition
(`doc/lfm3-pipelined-decode.md`, 1,030 instrumented steps, reconciliation
100.0%) measured:

| stage | ms/token |
| --- | --- |
| `build_ms` (host graph, ~700 MLX ops) | 0.263 |
| `eval_wait_ms` (blocking `stream.eval`) | 37.052 |
| `sample_ms` (cast + argmax + sync read) | 0.353 |
| `callback_ms` (detokenize, IPC, controls) | 0.009 |
| **host four-stage sum** | **0.624** |

The pipelined schedule recovers **P = sync − pipe = +1.08 / +1.13 ms/token**
(ctx 1024 / 4096). The unexplained remainder, **≈0.46–0.51 ms/token**, is
attributed in #127 to per-step launch-gap drain: a step's final kernels can
start before the host has finished enqueuing the graph's ~700 op submissions,
so the device periodically drains the command buffer inside the step's eval.
This assumption has never been directly tested:

- **QUEUE #1 (compiled decode) was INCONCLUSIVE** — the arm never reached a
  runnable state and no window ran (`doc/lfm3-compiled-decode-step.md`,
  lead correction 17:31). The launch-gap hypothesis is **open**.
- **#115's prior** (whole-step `mx.compile` decode = +0.7%, buckets ±2%)
  suggests the harvest is small, but #115 ran **without** the pipelined
  schedule and without an exact-shape in-step census; it is a prior, not a
  measurement of this hypothesis.
- **QUEUE #2 (custom kernels) and #3 (prefill) were INVALID** at their token
  gates; neither produced a timing window (`doc/lfm3-lm-head-custom-kernel.md`,
  `doc/lfm3-short-prompt-prefill.md`).

This ticket measures the hypothesis directly with the **minimal intervention**
that can move the host/enqueue boundary relative to the GPU: a `build`/`async`
/`sync` split of the single eval in the pipelined decode step. It is a
census-class probe (expected win: ~0), **not** an optimization PR — it is the
quantification QUEUE #4 requires before any follow-up direction is chosen.

**Candidate enumeration and why this one is first** (QUEUE #4 discipline:

"须先量化"):

| candidate | prior | position |
| --- | --- | --- |
| launch-gap drain inside the step | 0.46–0.51 ms/token unexplained (P − host sum), mechanism never tested | **this ticket — first, because it is the only candidate with a quantitative residual and it requires the least invasive measurement** |
| GPU kernel efficiency (QUEUE #2's mechanism) | INVALID at K3 token gate; roofline space 1.7–2.5 ms/token still exists | not re-opened here — needs a different shape-preserving kernel idea; would require its own pre-registration |
| short-prompt prefill (QUEUE #3's mechanism) | INVALID at P1 token gate; 8–12% gap still exists | not re-opened here; queue closed INVALID with no window |

If this ticket is FALSIFIED or INCONCLUSIVE, the kernel-efficiency and
prefill candidates keep their priors and each must be pre-registered
individually; nothing about this ticket's result closes them.

Status: **pre-registered (this commit, before any device run).** The decision
rule and thresholds below are frozen from here; amendments are dated
append-only at the end.

---

## 预登记 (pre-registration)

### 1. Arms

| arm | change | switch |
| --- | --- | --- |
| `apxinf-pipe` (baseline) | none; current pipelined decode (`build_graph` eager + one blocking `stream.eval(logits, states…)` in `execute_inner`) | unset |
| `apxinf-pipe-async` | same binary; the step's eval is split host-side into `stream.async_eval(required)` immediately after `build_graph`, then `stream.synchronize()` after the post-build work | `APXINF_LFM3_DECODE_ASYNC_EVAL=1` |

Mechanism (fixed here): `crates/apxinf-model/src/metal_lfm3/model.rs`'s
`execute_inner`, single-token step. Baseline today:

```rust
let logits = self.build_graph(&ids, start, len, last_only)?;
let mut required = vec![logits.clone()];
if len == 1 { required.extend(self.step_state_arrays()); }
stream.eval(&required)?;          // enqueue + wait, host blocked for ~37 ms
```

Armed:

```rust
let logits = self.build_graph(&ids, start, len, last_only)?;
let mut required = vec![logits.clone()];
if len == 1 { required.extend(self.step_state_arrays()); }
stream.async_eval(&required)?;    // enqueue only; returns immediately
// any post-build host work runs here, overlapping the device's early kernels
stream.synchronize()?;            // wait; same logical boundary as eval
```

The split is host-side only: the **same** `required` arrays are evaluated, on
the same `Stream`, with the same graph. `async_eval` is the bridge's
`mx::async_eval` (`crates/apxinf-mlx/native/bridge.cpp:74`) and `synchronize`
is `mx::synchronize` (`bridge.cpp:48`); the baseline `eval` is exactly
`mx::eval(arrays)` + `mx::synchronize(stream)` (`bridge.cpp:72`), so the armed
path is the same two MLX calls with the host wait moved after the enqueue
returns. No op, kernel, shape, weight, or dtype is touched ⇒ by construction
the evaluated arrays are identical and the greedy token stream must be
identical.

What the host can do in the window between `async_eval` and `synchronize`
(the mechanism under test): the post-build work that exists in this
function — `step_state_arrays()` is already collected before the eval in both
arms, and the stage-report / cache-putback bookkeeping runs between build and
eval today. The armed arm reorders so the enqueue returns first; the residual
host work (if any is non-zero) overlaps the device's early kernels. The
**decisive** measurement is not which host line fills the window, but whether
moving the host wait off the enqueue boundary changes `eval_wait_ms` and the
request's engine-reported total at all: if it does not, the "host blocked
*inside* the eval's enqueue path" reading of the 0.46–0.51 ms is wrong.

Effect proof (the 空臂 trap): `decode_stages_enabled()`-style helper
`decode_async_eval_enabled()` reads `APXINF_LFM3_DECODE_ASYNC_EVAL=="1"` at
`Lfm3Model::new` and stores it in a `Lfm3Model` field; the worker banner
(`crates/apxinf-serving/src/native_worker.rs`) gains
`decode_async_eval=<false|true>` from the **same env**, and the gateway log's
banner line is the arm's identity. A segment whose banner lacks
`decode_async_eval=true` is void.

Instrumentation for the census: with `APXINF_LFM3_DECODE_STAGES=1` (existing,
default off), the sync arm already emits `decode_stage` JSON lines
(`report_stage`, `model.rs:261`) with `build_ms` / `eval_ms` per step. The
armed arm emits the **same** lines; `eval_ms` for the armed arm covers
`async_eval + synchronize` (the timer brackets both), and the build timer is
unchanged. `StreamCounters` deltas (`trace_callbacks`, `eval_calls`,
`uploads_bytes`, `downloads_bytes`, `crates/apxinf-mlx/src/array.rs:153`) are
read before/after one warm 64-token request per arm in the census to confirm
the armed path performs the **same** `eval_calls` count (+1 per step) and
byte counters as baseline (the split must not add an eval boundary or a
download).

### 2. Cells

S2 decode only, exactly #127's口径: contexts 1024 and 4096, `max_tokens=128`,
`--engine-timings` on both apxinf arms, `scripts/bench_engines.sh` +
`scripts/bench_engines_report.py`, per-segment corpus seeds (each arm in a
segment shares the segment's seed; distinct text per segment so no
prefix-cache hits). S1 decode-at-8 cells stay excluded (A/A not certifiable).
No tier A/B rerun, no S1/S3/S4/S5, no prefill cells: prefill uses
`execute_inner` chunk-wise (`len>1`) and the armed path is gated to the
single-token step only.

### 3. Order

`A B B A A B` where A = `apxinf-pipe` (baseline, unset), B =
`apxinf-pipe-async` (`=1`): three segments per arm, paired medians per cell
(each segment contributes one median per arm; a cell's paired value is the
median of the three segment medians). The `llama.cpp` arm is **not re-run** —
this ticket's decision is within-apxinf (`E = A − B`); #127's certified
llama numbers remain the external reference.

### 4. Notation (per cell, from paired medians)

- **E** = `apxinf-pipe` − `apxinf-pipe-async` (ms/token) — the launch-gap
  harvest.
- **E/A** = async / baseline.
- **A/A** — worst segmented-median spread per arm per cell (§6).

### 5. Falsifiers (mechanism must actually account for the win)

1. **Token inconsistency (hard gate)** — emitted token ids differ between the
   two arms for any of 4 prompts × 128 tokens ⇒ **INVALID**. Same harness as
   #127's T1 and QUEUE #1's C1 (`#[ignore]`, real BF16 checkpoint,
   `--test-threads=1`), same entry/exit switch reset discipline.
2. **Host/GPU split drift (hard gate)** — the armed arm must evaluate exactly
   the same `required` arrays with the same `eval_calls` delta per step and
   the same byte counters; any extra eval boundary, download, or shape change
   ⇒ **INVALID**. Guarded by the §8 `StreamCounters` census and the existing
   shape contract (`step_state_arrays` is unchanged).
3. **A/A > 5%** for either arm in either cell ⇒ window **INVALID**.
4. **FALSIFIED (mechanism)** — token-consistent and A/A-clean but
   **E ≤ 0.10 ms/token** on both cells (below the A/A-certifiable noise floor
   of this harness), or `eval_ms` (the decomposition's eval_wait) does not
   fall by a comparable amount while the request total falls: the
   "host blocked inside the eval's enqueue path" mechanism fails; §11 names
   the reading that remains.
5. **census contradiction** — the armed arm's `eval_calls`/bytes counters
   differ from baseline, or the `async_eval`→`synchronize` window contains no
   host work (measured by the stage timers as a wall-clock gap of ~0): the
   split did not actually move the host/enqueue boundary; recorded as
   **INCONCLUSIVE** with the counters, not as a win.

### 6. Decision rule (frozen)

- **INVALID** — any of the three hard gates in §5 fires.
- **WIN** — both cells E ≥ 0.25 ms/token, and in all three segments
  `apxinf-pipe-async` < `apxinf-pipe` for both cells.
- **PARITY** — both cells 0 < E < 0.25 ms/token (token-consistent, A/A-clean)
  but WIN does not hold.
- **PARTIAL** — E ≥ 0.25 ms/token on one cell only, or 0.10 < E < 0.25 with
  the census showing the host wait was genuinely moved off the enqueue
  boundary: the in-step gap is real but not the full mechanism; §11 names
  what remains.
- **FALSIFIED** — §5.4 fires.
- **INCONCLUSIVE** — §5.5 fires, or the numbers land outside every band with
  all facts reported.
- Anything else: **INCONCLUSIVE**, all numbers reported.

The queue's expected 0.3–0.5 ms/token is the WIN band's center; the WIN
threshold is set at 0.25 ms/token (half of the lower expectation) so a
half-harvest is still a WIN and the A/A-certifiable noise floor does not
define the verdict.

### 7. Default rule

This ticket is a **census/probe**, not an optimization: if the window is not
INVALID, the `APXINF_LFM3_DECODE_ASYNC_EVAL` switch and its banner field are
**reverted**, and only this record is kept — regardless of WIN/PARITY. If
this ticket produces a WIN, the follow-up (a real fix that harvests the
measured gap, e.g. QUEUE #1's compiled step revisited with this census, or
a different in-step overlap) must be a **new pre-registration** citing this
document. If INVALID/FALSIFIED/INCONCLUSIVE, the switch is reverted the same
way; the record stays. No opt-in dead path is left in the tree by this
ticket.

### 8. Gate-0 census (before any timing window)

Same locked M4 session, one warm 64-token request per arm:

- **`StreamCounters` deltas** (before/after the request, `crates/apxinf-mlx/src/array.rs:153`):
  `eval_calls` per step (expected +1 both arms — the split must not add an
  eval boundary), `trace_callbacks` (expected equal — same eager graph),
  `uploads_bytes` / `downloads_bytes` (expected equal — the probe performs no
  I/O). Any divergence ⇒ §5.2 INVALID.
- **decomposition** (`APXINF_LFM3_DECODE_STAGES=1`, separate non-timed
  segments): `build_ms` must be unchanged (build happens before the split in
  both arms); `eval_ms` (armed = `async_eval` + `synchronize`) is the
  quantity that should fall if the host wait was inside the enqueue path.
  Reconciliation must stay ≤10% against each arm's engine `decode_ms/token`,
  as in #127 §4.
- **host work inside the armed window**: measured as `eval_ms` minus the
  device-observed work — if the MLX side exposes no device timer
  (`xctrace` unavailable, MLX SDK headers have no per-kernel timer; same
  honesty note as QUEUE #2's §8), the window's host content is reported as
  **机制未知** and only the wall-clock `eval_ms` delta is used. No substitute
  metric is presented as a GPU measurement.
- **kernel-launch count**: `StreamCounters` does not expose it; `xctrace` is
  unavailable ⇒ **kernel launches: 机制未知**. The window still runs; §5.4
  can only be assessed from the request total and the decomposition.

### 9. Tests (before device runs)

Real BF16 checkpoint, Metal-locked, `#[ignore]`-marked, run with
`APXINF_LFM3_TEST_MODEL=<snapshot dir>` on the M4 (same harness as #127's
T1/T2/T3, same `--test-threads=1`):

- **E1 token equality** — 4 prompts × 128 tokens, `APXINF_LFM3_DECODE_ASYNC_EVAL`
  unset vs `=1`, identical token ids (the §5.1 gate); entry and exit both
  reset the switch so a prior panic cannot pollute the baseline arm.
- **E2 budget/EOS/cancel edges** — budget 1, budget 8, synthetic first-token
  EOS, cancel right after first publish: async == baseline (the default
  schedule is the reference), `position == P+N−1` on rollback; the conv-window
  restore must still apply to the discarded lookahead step. (These exercise
  the schedule driver; the split itself is confined to `execute_inner`'s
  single-token eval.)
- **E3 continuation state** — after an armed run with the extra lookahead
  step rolled back, all KV rows `[0, position)` and all 24 conv windows are
  byte-identical to a baseline run's, and a 16-token continuation matches
  token-for-token.
- **E4 empty-arm / split negative control (unit level)** — with the switch
  unset, a model constructed in-process must report `decode_async_eval=false`
  and the `async_eval`/`synchronize` branch must be unreachable (grep-able /
  field-asserted); with `=1`, `async_eval` must be called before
  `synchronize` (a `#[cfg(test)]` counter seam or a `StreamCounters.eval_calls`
  ordering assertion). Output pasted in the PR.
- The four `cpu_pipeline_*` tests in `crates/apxinf-model/src/mlx_decode.rs`
  must still pass unchanged: the schedule driver is not modified by this
  change.

### 10. Window discipline

M4 (`haiyan-mac-mini`) only, per RUNNER.md: `SDKROOT=
/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk`, `MLX_ROOT=~/mlx-sdk`,
both Metal locks taken in order (`/tmp/enginetailor-metal.lock` then
`~/.cache/enginetailor/metal-measure.lock`), **waited, never stolen**;
`pmset displaysleepnow` + `caffeinate -si` (no `-u`) before the window;
engine-reported timings (`--engine-timings`); swap / thermal / load recorded
before and after; device budget ≤45 min (gate-0 census + ABBA + E1 parity
probe). M2 Air is not measured; nothing is extrapolated across hosts.
Cross-window absolute numbers are not comparable (#127 §11 caveat 5): this
ticket's verdict rests only on within-window paired medians.

### 11. Implementation limits

- **No numerics change, no shape/weight/kernel change**: the armed path
  enqueues the identical `required` arrays on the same `Stream`; only the
  host-side wait boundary moves. The tiling/accumulation hazard from PR #128
  applies only if shapes change, and §5.2 blocks that.
- **No compiled graph, no bucketing** — distinct from QUEUE #1; this ticket
  wraps nothing and changes no op count (`trace_callbacks` must stay equal,
  §8).
- **No prefill change**: armed path gated to `len == 1` (single-token device
  decode); `execute_inner`'s chunked path is untouched.
- **Single-token eval only**: `DeviceDecode::decode` (the pipelined
  lookahead step) uses `build_graph` and does not call `execute_inner`'s
  eval; the armed switch therefore affects the sync reference path's eval
  boundary as well (see §Relation — this is intentional: the decomposition
  that produced the 0.46–0.51 ms residual was measured on the **sync** arm,
  so the probe must be able to move that arm's boundary).
- Harness changes only: env switch, worker banner field, existing stage
  instrumentation reused as-is. `bench_engines.sh` /
  `bench_engines_report.py` verdict logic and gates untouched.

---

## Relation to QUEUE #1 / #2 / #3, #115, #127

- **#127 (pipelined decode)** — the baseline substrate and the source of the
  0.46–0.51 ms residual. This ticket measures that residual; it does not
  modify #127's schedule (`GreedySequence`, `mlx_decode.rs`).
- **QUEUE #1 (compiled decode)** — **INCONCLUSIVE** (lead correction 17:31):
  no arm ran, no window ran; the launch-gap hypothesis is **open**. This
  ticket is the minimal census that QUEUE #1's §8 census wanted but never
  executed; it deliberately avoids `Compiled` (the unresolved integration
  defect) and measures the host/enqueue boundary directly instead.
- **#115** — its +0.7% prior on whole-step compile stands as a **prior**, not
  as a measurement of this probe: #115 ran without the pipelined schedule and
  without an exact-shape in-step census. This ticket cites it as the
  expectation that the harvest is small; it does not treat it as settled.
- **QUEUE #2 (custom kernels)** — **INVALID** (K3 token gate); not re-opened.
  Its roofline-prior (1.7–2.5 ms/token) remains a separate candidate; if this
  ticket FALSIFIEDs the launch-gap mechanism, kernel efficiency is the next
  item to pre-register individually.
- **QUEUE #3 (prefill)** — **INVALID** (P1 token gate); not re-opened. Its
  8–12% short-prompt gap remains a separate candidate.
- **#128 (merged GEMV INVALID)** — the shape-preserving constraint this
  ticket inherits; §5.2 re-states it as the host/GPU split drift gate.

## Results

### 2026-11-19 20:32 Amendment (dated, pre-window) — schedule split of the arms

The §9 tests discovered that the probe's mechanism location and the
measurement schedule do not coincide. Recorded here, pre-window, so the arms
below are the ones §3's order actually names.

**Facts (from commit `813c674d`'s tests, `e4_empty_arm_and_split_ordering`):**

- The split lives in `execute_inner`'s single-token eval, which only the
  **sync** schedule reaches (`SyncTokenSequence` → `forward` → `execute`).
- The **pipelined** `GreedySequence` (the substrate the 0.46–0.51 ms residual
  was measured on) builds each step via `DeviceDecode::decode`, which calls
  `build_graph` and never `execute_inner`'s eval. With `=1` on the pipelined
  schedule the seam stays empty: `async_eval` → `synchronize` is never
  executed. This is by design, not a bug — `execute_inner` is the sync
  reference's step function.

**Consequence for §3's arms (this is the adjudication lead reserved):**

§3's `A B B A A B` as written names two pipelined arms, but on the pipelined
schedule the B arm is byte-identical to A (empty switch, zero dispatch) — an
A/A ≈ 0 "measurement" that would report WIN/PARITY without the probe ever
running (the empty-arm trap §1's effect proof exists to prevent). Therefore
the window will **not** run the §3 order as written. It will run two separate
sync-schedule windows instead:

- **Window S** (mechanism window; decides the ticket): A = sync, split off;
  B = sync, `APXINF_LFM3_DECODE_ASYNC_EVAL=1`. Order `A B B A A B`, S2
  ctx 1024/4096 × 128 tok, `--engine-timings`. Decision rule §5/§6 unchanged;
  **E = sync−async** on the schedule the decomposition that produced the
  residual was measured on, which is what §11 already justified.
- **Window P** (census only, no verdict): pipelined `apxinf-pipe` with the
  switch unset, then `=1`, `eval_calls`/`trace_callbacks`/bytes deltas
  recorded; expected outcome is the empty arm (`=1` ≡ unset, `test_split_trace`
  empty, `StreamCounters` delta 0). Confirms §5.2's "the pipelined harvest is
  not reachable by this switch" reading. The pre-registered 1.08/1.13 ms
  `P = sync − pipe` residual itself is not re-measured here (#127 certified it).

The switch affects only the sync path, so pipelined apxinf-pipe baseline
numbers used as the external reference are unaffected (`=1` unset ⇒ empty
arm ⇒ baseline).

**Fallback:** if lead rejects splitting the window this way, the alternative is
INCONCLUSIVE by §5.5 (the switch cannot move the boundary of the schedule the
residual was measured on) — not a WIN/PARITY read from a no-op B arm. The
§7 default rule (revert switch + banner, keep record) applies either way.

*(window pending — appended after the locked M4 run; only dated amendments to
the pre-registration above may appear before the verdict.)*
