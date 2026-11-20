# LFM3 launch-gap harvest: compiled-subgraph decode step (issue #130 item 1)

Source: issue #130 resident queue, item 1 ("eval_wait 内的 launch-gap 收割",
expected ~0.3–0.5 ms/token), campaign `lfm3-launch-gap`, base
`feat/lfm3-metal` (tip `bbaceed3`). PR target: `feat/lfm3-metal`.

Baseline: PR #127 (`apxinf-pipe` 36.07 / 36.40 ms/token in its certified
window). PR target: `feat/lfm3-metal`.

The question is specific and falsifiable. #127's sync-arm decomposition
(`doc/lfm3-pipelined-decode.md`, 1,030 instrumented steps, reconciliation
100.0%) measured `eval_wait_ms = 37.052` against a host four-stage sum of
0.624 ms/token, and the pipelined schedule recovered only P = +1.08 / +1.13
ms/token of it; the residual ≈0.46–0.51 ms/token is attributed there to
per-step launch-gap drain inside the step's eval (~700 eager `mx`
submissions per one-token step). This ticket tests the fix allowed by issue
#130 item 1 — merging the step's kernel submissions — in the only form that
can actually replay on this graph.

## Why not the whole-step single `Compiled` arm (a-priori, mechanism)

The prior whole-step attempt (`doc/lfm3-compiled-decode-step.md`, dig QUEUE
#1, INCONCLUSIVE) scaffolded one exact-shape `Compiled` over the whole step
and could not be brought to a compiling, behavior-verified state. Three
mechanical reasons, each sufficient on its own:

1. **The sdpa KV window changes shape every step.** Full attention reads
   `keys[0..start+1]`; sliding reads `keys[start+1-256..start+1]`. `mx::compile`
   specializes traced shapes (there is no dynamic-bound `slice` in the traced
   tape), so an exact-shape whole-step profile retraces at every position —
   the no-retrace hard gate fails by construction in steady state.
2. **No host scalar reads inside a trace.** The whole-step body needs
   `position` (slice bounds, RoPE rows, write index); reading it via
   `to_u32_scalar()` forces a device sync mid-trace, which compile tracing
   does not admit. The scaffold's closure had exactly this shape.
3. **Static-window alternatives change numerics.** Padding/bucketed KV or a
   full-capacity masked sdpa changes matmul/softmax reduction extents ⇒
   tiling/accumulation order changes ⇒ presumptive token drift (PR #128
   precedent; issue #130 rule 3). Pre-registration forbids them.

So this ticket compiles **every shape-static region** and keeps the
position-shaped remainder eager. This stays inside issue #130 item 1's text
("把每步的 kernel 提交合并") and its falsifier.

## 预登记 (pre-registration)

Status: **pre-registered (this commit, before any device run).** The
decision rule and thresholds below are frozen from here; amendments are
dated append-only at the end.

### 1. Arms

| arm | change | switch |
| --- | --- | --- |
| `apxinf-pipe` (baseline) | none; current pipelined eager decode step | unset |
| `apxinf-pipe-compiled` | same binary; every shape-static region of the single-token step runs through exact-shape `Compiled` callables; the position-shaped attention window stays eager | `APXINF_LFM3_COMPILED_DECODE=1` |

Mechanism (fixed here): `crates/apxinf-model/src/metal_lfm3/model.rs`,
`DeviceDecode::decode`, `len == 1` only (prefill untouched). Per step:

- embed: eager (`weights.embed` + reshape), unchanged.
- **conv layer** (`kind.is_conv()`): one `Compiled` per conv layer,
  inputs `(x, conv_state)` → outputs `(x_out, next_state)`; body is exactly
  `norm(operator_norm)` + `conv_step` + `mix_and_residual`. Position-free,
  capacity-free shapes.
- **attention layer**:
  - `Compiled` `pre`, inputs `(x, cos_row, sin_row)` → `(n, q, k, v)`;
    body is exactly `norm(operator_norm)` + the q/k/v projections + q/k
    norms + `apply_rope`. `cos_row`/`sin_row` are the existing
    `cos_sin(kind, start, 1)` outputs, sliced eagerly on the host per step
    (same values the eager arm slices).
  - eager: `index = Array::from_i32([start])`, two `slice_update` (KV
    write), two `slice_axis` (window read), one `sdpa` — the identical ops,
    order, shapes, and scale as `attention_step_parts` today, including the
    `low = end−256` sliding bound.
  - `Compiled` `post`, inputs `(x, n, attention)` → `x_out`; body is
    exactly the sigmoid gate, gated multiply, transpose/reshape, `o_proj`,
    and `mix_and_residual`.
- final: one `Compiled` `head`, input `(x)` → `logits`; body is exactly
    `norm(final_norm)` + last-row slice + `weights.project` + reshape.

Nothing else changes: same weights, same dtypes, same `Linear::call`
matmuls, same `fused_conv_window` kernel, same state arrays
(`slice_update` outputs for KV, conv callable outputs for windows), same
snapshot/restore rollback in `finish_decode`, same `GreedySequence`
schedule driver. No compiled region sees a capacity-shaped or
position-shaped array, so **one profile per callable serves every position
and every context length**; callables are built lazily on the first
compiled decode step of a process, warmed with `prepare_timed` (two evals +
the retrace guard), and replayed afterwards. A retrace in steady state is
the §5.2 hard gate and fails loudly via the existing guard.

Effect proof (the 空臂 trap): the worker banner
(`crates/apxinf-serving/src/native_worker.rs`) gains a
`decode_compiled=<false|true>` field derived from the same env switch; an
armed segment whose gateway log lacks `decode_compiled=true` is **void**.
Per Window S's lesson (doc/lfm3-decode-eval-sync-split.md §"Hard gate 1"),
the runner must rebuild the binaries before the window and assert the
banner field before the first timed request.

### 2. Cells

S2 decode only, exactly #127's 口径: contexts 1024 and 4096,
`max_tokens=128`, `--engine-timings` on both apxinf arms,
`scripts/bench_engines.sh` + `scripts/bench_engines_report.py`, per-segment
corpus seeds (each arm in a segment shares the seed; distinct text per
segment so no prefix-cache hits). No S1 decode-at-8 cells, no prefill
cells, no tier A/B rerun. The `llama.cpp` arm is **not re-run** — this
ticket's decision is within-apxinf (`C = A − B`).

### 3. Order

Warm-up first (per Window S's A/A lesson): after the rebuild and banner
assert, run one untimed 128-token decode request per arm to warm the
compiled profiles and the machine, then `A B B A A B` (A = `apxinf-pipe`,
B = `apxinf-pipe-compiled`), three segments per arm, paired medians per
cell (a cell's paired value is the median of the three segment medians).
The warm-up request is recorded in the receipts but contributes no numbers
to the cells.

### 4. Notation (per cell, from paired medians)

- **C** = `apxinf-pipe` − `apxinf-pipe-compiled` (ms/token) — the
  launch-gap harvest.
- **C/A** = compiled / baseline.
- **A/A** — worst segmented-median spread per arm per cell, same formula as
  #127 and the dig windows.

### 5. Falsifiers (mechanism must actually account for the win)

1. **Token inconsistency (hard gate)** — emitted token ids differ between
   the two arms for any of 4 prompts × 128 tokens ⇒ **INVALID**. Reuses
   #127's T1 parity probe and the dig arm's C1 shape.
2. **Retrace / shape drift (hard gate)** — any compiled callable retraces
   in steady state (trace_callbacks delta > 0 across steps 2..N of the
   census request), or any kernel-shape drift vs the eager step profile ⇒
   **INVALID**.
3. **Empty arm (hard gate)** — an armed segment whose banner lacks
   `decode_compiled=true`, or a runner that timed against a stale binary ⇒
   **void ⇒ INVALID**.
4. **A/A > 5%** for either arm in either cell ⇒ window **INVALID** (no
   median-massaging; rerun the window once after a cool-down if the first
   window fires this, and report both).
5. **FALSIFIED (mechanism)** — token-consistent, retrace-clean, A/A-clean
   but **C ≤ 0.10 ms/token** in either cell (the residual is below the
   A/A-certifiable noise of this harness), or compiled is slower. The
   launch-gap-drain mechanism then fails and §8's census names what
   remains.

### 6. Decision rule (frozen)

- **INVALID** — any hard gate in §5 fires.
- **WIN** — both cells C ≥ 0.25 ms/token, and in all three segments
  `apxinf-pipe-compiled` < `apxinf-pipe` for both cells.
- **PARITY** — both cells 0.10 < C < 0.25 ms/token, token-consistent,
  A/A-clean.
- **PARTIAL** — C ≥ 0.25 ms/token on one cell only; or 0 < C ≤ 0.10 with
  the §8 census showing the in-step submission count fell by ≥50% (the gap
  is submissions but the drain is elsewhere — §8 names it).
- **FALSIFIED** — §5.5 fires.
- Anything else: **INCONCLUSIVE**, all numbers reported.

### 7. Default rule

If the window is not INVALID and WIN or PARITY holds, the compiled-subgraph
step becomes the **default** greedy decode step, with
`APXINF_LFM3_COMPILED_DECODE=0` kept as the operator escape hatch. If
PARTIAL, it stays behind the switch until the remaining gap is addressed.
FALSIFIED/INVALID: the switch and its code are reverted; only this record
and the receipts remain.

### 8. Gate-0 launch / trace census (before any timing window)

Same locked M4 session, one 64-token request per arm after warm-up:

- `stream.counters().trace_callbacks` delta per step for both arms —
  expected ≈700/step baseline vs (number of eager ops) + (number of
  compiled calls) for the armed arm, flat across steps 2..N (§5.2);
- per-step eager op count in the armed arm, from the same counter;
- banner field check (`decode_compiled=true` present in the armed census
  gateway log);
- `eval_calls` / byte counters equal-shape sanity (no extra sync boundary
  per step vs baseline);
- kernel launch count: if the MLX counters do not expose it, write
  "launch count: 机制未知" per discipline rule 5 rather than substituting a
  proxy.

Mechanism arithmetic to reconcile against: baseline step = 3.4576 GB/token
⇒ 34.36 ms floor at 100.6 GB/s; the harvestable residual is P − host-sum =
0.46–0.51 ms/token. If C lands near that band and the submission count
fell by the census-measured amount, the mechanism is accounted; if C is
small while submissions fell, the drain is GPU-scheduler-side between
steps, recorded as FALSIFIED with the census.

### 9. Tests (before device runs)

Real BF16 checkpoint, Metal-locked, `#[ignore]`-marked, run with
`APXINF_LFM3_TEST_MODEL=<snapshot dir>` on the M4 (same harness as #127's
T1/T2/T3, same `--test-threads=1`):

- **C1 token equality** — 4 prompts × 128 tokens,
  `APXINF_LFM3_COMPILED_DECODE` unset vs `=1`, identical token ids (§5.1).
- **C2 budget/EOS/cancel edges** — budget 1, budget 8, synthetic
  first-token EOS, cancel right after first publish: compiled == sync
  (`APXINF_MLX_SYNC_DECODE=1` reference), `position == P+N−1` on rollback;
  the conv-window restore still applies to the discarded lookahead step.
- **C3 continuation state** — after a compiled run with the extra lookahead
  step rolled back, all KV rows `[0, position)` and all conv windows are
  byte-identical to a sync run's, and a 16-token continuation matches sync
  token-for-token.
- **C4 retrace guard (negative control)** — a `#[cfg(test)]` seam that
  feeds a deliberately shape-mismatched input to one compiled region must
  fail loudly (retrace guard / shape error), proving the no-retrace
  property is load-bearing and observable.
- The CPU-side pipeline tests in `crates/apxinf-model/src/mlx_decode.rs`
  (the four `cpu_pipeline_*` tests) must still pass unchanged.

### 10. Window discipline

M4 (`haiyan-mac-mini`) only: `SDKROOT=
/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk`, `MLX_ROOT=~/mlx-sdk`,
both Metal locks taken in fcntl order (`/tmp/enginetailor-metal.lock` then
`~/.cache/enginetailor/metal-measure.lock`), **waited, never stolen**;
`pmset displaysleepnow` + `caffeinate -si` (no `-u`) before the window;
engine-reported timings (`--engine-timings`); swap / thermal / load
recorded before and after; device budget ≤60 min (gate-0 census + ABBA +
parity probe + C1–C4). M1 Pro is not measured; nothing is extrapolated
across hosts. Cross-window absolute numbers are not comparable: the verdict
rests only on within-window paired medians.

### 11. Implementation limits

- **No shape, weight, dtype, or op-sequence change**: the compiled bodies
  are the existing functions refactored to take their context explicitly;
  the eager remainder is the existing `attention_step_parts` tail verbatim.
- **No bucketing, no padding, no masked full-capacity sdpa** (§"Why not",
  item 3).
- **No kernel rewrite, no wider matmul** (issue #130 item 2's scope).
- Harness changes only: env switch, banner field, compiled-callable
  plumbing. `bench_engines.sh` / `bench_engines_report.py` verdict logic
  and gates untouched.
- If C1 fails **and** the census attributes the drift to the compiler's
  fusion/simplification passes (not to a body bug), one dated amendment may
  retry once with a no-fuse compile mode exposed through the FFI; a C1
  failure with any other attribution is INVALID and stands.

---

### 2026-11-20 12:15 Amendment (dated, pre-window) — census instrument correction

Reading `crates/apxinf-mlx/src/array.rs` before building the arm shows
`StreamCounters.trace_callbacks` increments **only** in the `Compiled`
trace callback (one per trace/retrace), not per eager op — the dig QUEUE
#1 doc's "baseline accumulates trace_callbacks per step" was wrong about
this counter. Corrected §8 census:

- **Retrace gate (the load-bearing one)**: baseline arm `trace_callbacks`
  delta = 0 across the whole census request (no `Compiled` exists);
  armed arm = exactly the number of regions (13 conv + 2×11 attention +
  1 head = **36**) after the warm-up step, then **delta 0** across every
  later step and across a second request (per-position, per-request
  profile reuse — §5.2's direct observable).
- **Eager host-submission count**: no counter exists in `StreamCounters`
  (`uploads_bytes`, `downloads_bytes`, `eval_calls`, `trace_callbacks`
  only). Reported as a **static estimate from the code** (baseline ≈700
  ops/step from the build-graph recipe; armed ≈ 11×12 + 13 + 2 ≈ 148
  eager ops + 36 compiled calls/step) and labelled "静态估计" rather than
  measured; kernel launches stay "机制未知" unless an MLX/Metal counter
  appears. §8's decision logic does not depend on this estimate.
- **`eval_calls` / byte counters**: armed arm must show no extra
  evaluation boundary per step vs baseline beyond the one-time warm-up
  evals (prepare_timed's two evals per region on the first step only).
