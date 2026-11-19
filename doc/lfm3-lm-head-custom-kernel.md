# LFM3 shape-preserving custom kernels: lm_head GEMV + decode-step elementwise fusion

Source: `QUEUE.md` item 2 (campaign `lfm3-dig`, base `feat/lfm3-metal`).
Baseline: PR #127 (`apxinf-pipe` 36.07 / 36.40 ms/token, `feat/lfm3-metal` tip
`1ed3f2f3`). PR target: `feat/lfm3-metal`.

The question is specific and falsifiable. #127 certified apxinf BF16 at
**0.99× llama.cpp F16** (36.07/36.40 vs 36.42/36.67 ms/token), and #127's
bandwidth probe measured the M4's practical streaming ceiling at **100.6
GB/s**, which gives a hard floor of `3.4576 GB / 100.6 GB/s = 34.36
ms/token`. Both engines are at 92–97% of that ceiling. The remaining
**~1.7–2.5 ms/token** is the GPU-side roofline headroom: kernels that are not
yet at peak effective GB/s. QUEUE item 2 targets exactly this remainder, by
name, with **shape-preserving** custom Metal kernels — the shapes, the weight
bytes, and the per-output accumulation order stay identical to the MLX graph
that #127 certified, so the token output must be byte-identical.

Status: **pre-registered (this commit, before any device run).** The decision
rule and thresholds below are frozen from here; amendments are dated
append-only at the end.

---

## 预登记 (pre-registration)

### 1. Arms

| arm | change | switch |
| --- | --- | --- |
| `apxinf-pipe` (baseline) | none; current pipelined decode (MLX graph, fused conv/rope kernels) | unset |
| `apxinf-pipe-kkernel` | same binary; the decode step's lm_head GEMV and the RoPE / sigmoid-gate / silu×mul elementwise chains are dispatched through custom `MetalKernel`s that reproduce the same per-output accumulation order | `APXINF_LFM3_KKERNEL=1` |

Mechanism (fixed here):

- **lm_head** (`Weights::project`, `crates/apxinf-model/src/metal_lfm3/weights.rs:353`):
  `[1, 1, 2048] × [2048, 128000]` BF16 → `[1, 128000]` FP32 logits. The tied
  head table is 524,288,000 B (2 × 2048 × 128000) and is re-read every token,
  ≈15.1% of the 3.4576 GB/token. MLX's generic matmul tiles both operands; a
  one-row GEMV only needs the weight stream, and the custom kernel reads each
  output column `j`'s 2048 weights **in ascending k order into one FP32
  accumulator** — the same reduction form the reference uses for a single
  row. No splitting of the K dimension, no transposed read, no wider output
  tile that changes which accumulator each output row uses.
- **RoPE** (`apply_rope`, `model.rs:72`): `cast→mul→(rotate_half cast/mul/neg/concat)→mul→add→cast`
  replaced by one kernel that performs the identical elementwise sequence in
  the identical per-element order (the chain is element-local, so fusion
  cannot change any value; the existing `bf16_qk_table_rope` kernel is the
  shape/fusion precedent to extend, not a new op).
- **sigmoid gate** (`silu_mul`, `model.rs:82`: `g * sigmoid(g) * u`) and the
  residual/norm elementwise chains: fused where the graph currently issues
  two or three elementwise launches over the same shape. Elementwise fusion
  is value-identical by construction; the only thing that can change tokens
  here is a different rounding path (e.g. computing sigmoid in F16 instead
  of F32), which the gate keeps out.

Proof of effect (空臂 trap): the worker banner
(`crates/apxinf-serving/src/native_worker.rs:529`) gains a
`kkernel=<false|true>` field derived from the same env switch; a segment whose
gateway log lacks `kkernel=true` is **void**. A gate-0 census (§8) confirms
the custom kernels actually dispatched (kernel-launch count per step, and
lm_head path name) before any timing window.

### 2. Cells

S2 decode only, exactly #127's 口径: contexts 1024 and 4096, `max_tokens=128`,
`--engine-timings` on both apxinf arms, `scripts/bench_engines.sh` +
`scripts/bench_engines_report.py`, per-segment corpus seeds (each arm in a
segment shares the seed; distinct text per segment so no prefix-cache hits).
S1 decode-at-8 cells stay excluded (A/A not certifiable). No tier A/B rerun,
no S1/S3/S4/S5, no prefill cells: prefill uses `execute_inner` chunk-wise; the
single-row lm_head shape only occurs in decode.

### 3. Order

`A B B A A B` where A = `apxinf-pipe` (baseline), B =
`apxinf-pipe-kkernel`: three segments per arm, paired medians per cell (each
segment contributes one median per arm; a cell's paired value is the median of
the three segment medians). The `llama.cpp` arm is **not re-run** — this
ticket's decision is within-apxinf (`P_c = A − B`); #127's certified llama
numbers remain the external reference.

### 4. Notation (per cell, from paired medians)

- **K** = `apxinf-pipe` − `apxinf-pipe-kkernel` (ms/token) — the kernel
  efficiency harvest.
- **B/A** = custom-kernel / baseline.
- **A/A** — worst segmented-median spread per arm per cell (§6).

### 5. Falsifiers (mechanism must actually account for the win)

1. **Token inconsistency (hard gate)** — emitted token ids differ between the
   two arms for any of 4 prompts × 128 tokens ⇒ **INVALID**. This is the
   QUEUE item-2 falsifier verbatim, reusing #127's T1 parity probe harness
   (T1 already exists in `metal_lfm3/model.rs` and stays as the sync-arm
   reference).
2. **Shape drift (hard gate)** — any custom kernel dispatches at a padded,
   bucketed, widened, or otherwise different shape than the MLX op it
   replaces (including any output-tile padding that writes extra rows, or a
   lm_head K-split) ⇒ **INVALID**. Guarded by §8's shape diff and by C3's
   byte-equality of the post-step state.
3. **A/A > 5%** for either arm in either cell ⇒ window **INVALID**.
4. **FALSIFIED (kernel count vs request total)** — custom kernels dispatch,
   kernel-launch count per step falls, but the request total does not:
   either the replaced kernels were not the bottleneck, or the saved time is
   re-spent on the sync/read path. Recorded as FALSIFIED with the §8 census.
5. **FALSIFIED (lm_head arithmetic)** — lm_head kernel at its measured
   effective GB/s cannot close the gap: if the custom lm_head reaches ≥90%
   of the 100.6 GB/s ceiling on its own 524 MB stream but the cell's K is
   still ≤0.10 ms/token, the roofline attribution was wrong and the residual
   is in the attention/KV or host paths — mechanism unknown, QUEUE #2 closes
   with that statement.
6. **FALSIFIED (fusion overflow)** — elementwise fusion reduces launch count
   but the cell's K ≤ 0.10 ms/token; the launches were not the bound.

### 6. Decision rule (frozen)

- **INVALID** — any of the three hard gates in §5 fires.
- **WIN** — both cells K ≥ 0.25 ms/token, and in all three segments
  `apxinf-pipe-kkernel` < `apxinf-pipe` for both cells.
- **PARITY** — both cells 0 < K < 0.25 ms/token (token-consistent,
  A/A-clean) but WIN does not hold.
- **PARTIAL** — K ≥ 0.25 ms/token on one cell only, or 0.10 < K < 0.25 with
  the §8 census showing the targeted kernels were replaced and their
  effective GB/s rose: the kernel inefficiency is real but not the full
  mechanism; §11 names what remains (expected: attention SDPA / KV streaming,
  or the host-side sampling sync).
- **FALSIFIED** — §5.4, §5.5, or §5.6 fires.
- Anything else: **INCONCLUSIVE**, all numbers reported.

QUEUE item 2 states the roofline space as ~1.7–2.5 ms/token; the WIN threshold
is set at 0.25 ms/token (the A/A-certifiable noise floor of this harness is
~5% of 36 ms ≈ 1.8 ms *per window*, but paired segments recover sub-ms
medians) so a half-harvest on one direction still counts, and the
#127/#128 lesson (token equality beats speed) is the gate, not the target.

### 7. Default rule

If the window is not INVALID and WIN or PARITY holds, the custom kernels
become the **default** decode path, with `APXINF_LFM3_KKERNEL=0` kept as the
operator escape hatch (same policy as #127's schedule switch). If PARTIAL, the
targeted kernels stay behind the switch until the remaining gap is addressed;
FALSIFIED/INVALID: the kernels and the switch are reverted, only this record
remains.

### 8. Gate-0 census (before any timing window)

Same locked M4 session, one 64-token request per arm, warm model:

- per-step Metal kernel-launch count (Metal System Trace via `xctrace`, if
  available; otherwise the MLX-side `trace_callbacks` / op-count proxy plus
  the custom-kernel dispatch counter). **Honesty note (2026-11-19)**: on the
  current M4 build, `xctrace` is not available (active developer directory is
  Command Line Tools only; `xcrun --find xctrace` fails) and the MLX SDK
  headers expose no per-kernel GPU-time API (`~/mlx-sdk/include/mlx/backend/metal/*`
  has no timer/counter entry points), and this repo's `StreamCounters`
  (`crates/apxinf-mlx/src/array.rs:153`) only tracks `trace_callbacks`,
  `eval_calls`, `uploads_bytes`, `downloads_bytes`. If no counting mechanism
  is reachable at execution time, §8 reports **"kernel launches: 机制未知"**
  and the timing window still runs — but §5.4 can then only be assessed from
  the request total, and §5.5 must use the custom kernel's own self-reported
  GB/s (see below).
- **per-kernel effective GB/s, computed in-kernel and exported**: the custom
  kernels record their own `bytes / (end − start)` using Metal GPU timestamps
  (the kernel writes a small `[2]` F64 buffer of `start/end` ticks; the
  client reads it after the request). This gives, for the census only: lm_head
  effective GB/s (expect ≥90% of 100.6 on its 524 MB), RoPE/gate/silu elementwise
  GB/s. The baseline's equivalent is **机制未知** unless the same
  instrumentation exists for MLX's kernels — if it does not, state that the
  census compares the custom arm against the #127 bandwidth probe, not against
  a measured baseline kernel. No substitute metric is presented as GB/s.
- **shape diff**: dump the custom kernels' dispatch shapes and the replaced
  MLX op's traced shapes (from `APXINF_MLX_TRACE` or the existing op trace);
  any mismatch ⇒ §5.2 INVALID.

### 9. Tests (before device runs)

Real BF16 checkpoint, Metal-locked, `#[ignore]`-marked, run with
`APXINF_LFM3_TEST_MODEL=<snapshot dir>` on the M4 (same harness as #127's
T1/T2/T3, same `--test-threads=1`):

- **K1 lm_head kernel vs MLX matmul** — random `[1, 1, 2048]` BF16 input,
  custom kernel output `allclose` vs `x.matmul(&head)` at **F32 rtol=0,
  atol=0** for the single-output-column case, and identical token ids on a
  greedy decode of 4 prompts × 128 tokens with the kernel armed. (The
  absolute-value equality is the strictest form; if MLX's reduction order is
  not bit-identical to ascending-k FP32 accumulation, relax to the T1-style
  token-equality assertion and record the numeric delta as an amendment — the
  hard gate is tokens, not logits.)
- **K2 elementwise kernels** — RoPE kernel vs `apply_rope` reference and
  `silu_mul` fused vs `silu_mul` reference: `allclose` F32 rtol=1e-6,
  atol=1e-6 on the exact graph shapes, plus the C1-shaped token sweep.
- **K3 token gate** — the K1/K2 harness combined into 4 prompts × 128 tokens,
  default arm first, `APXINF_LFM3_KKERNEL=1` second, token-for-token equality
  (the §5.1 gate). Switch reset at test end, `--test-threads=1`.
- **K4 shape guard (negative control)** — the kernel dispatcher is given a
  padded input (KV padded to a wider ctx, or lm_head with an output tile that
  includes extra rows) and must **fail loudly** (dispatch returns
  `Err` / the request errors), proving §5.2 is load-bearing, mirroring
  #127's retracing-failure pattern and #128's lesson that a shape change is
  exactly how tokens drift.
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
before and after; device budget ≤45 min (gate-0 census + ABBA + parity
probe). M2 Air is not measured; nothing is extrapolated across hosts.
Cross-window absolute numbers are not comparable (#127 §11 caveat 5): this
ticket's verdict rests only on within-window paired medians.

### 11. Implementation limits

- **No bucketing, no padding, no wider matmul** — the hard constraint that
  makes this a *new* hypothesis rather than a re-run of #128 (which showed
  that a merged/wider matmul changes tiling ⇒ accumulation order ⇒ tokens).
  Every custom kernel takes the exact MLX op's input shapes and writes the
  exact output shape. KV/conv state shapes are unchanged.
- **lm_head: one output column per accumulator, ascending-K FP32
  accumulation.** The tied head is read once per token (524 MB, 15.1% of the
  token bytes). No K-splitting, no two-pass partial-sum reduction, no
  transposed weight read. If MLX's own lm_head uses a different reduction
  form, the token gate (§9 K1) decides whether the ascending-K form is
  acceptable; if not, the lm_head arm is dropped and the ticket records
  PARITY/PARTIAL without it rather than changing accumulation order.
- **Elementwise fusion only** — RoPE (`cast/mul/neg/concat/mul/add`),
  sigmoid gate, silu×mul, and residual/norm chains. Each fused kernel must
  preserve the operation order and FP32 intermediate precision of the
  unfused graph; F16 intermediates are prohibited (would change rounding ⇒
  tokens).
- **No kernel rewrite of the attention SDPA or the KV cache writes** this
  round: those are the largest remaining byte streams but their shapes
  (`[16, ctx, 128]`, GQA 4) and fused-attention semantics are out of scope;
  if §11's census names them as the residual, that is QUEUE #4 material.
- Harness changes only: env switch, worker banner field, in-kernel timestamp
  export (census only). `bench_engines.sh` / `bench_engines_report.py`
  verdict logic and gates untouched.
- Prefill (`execute_inner`, chunked, `len>1`) is untouched: the single-row
  lm_head shape does not occur, and prefill kernels are out of scope.

---

## 2026-11-19 amendment — Relation to #128 and to QUEUE #1

Two prior items constrain this ticket and are cited here so it does not look
like reopening a closed door.

**#128 (merged GEMV) was INVALID on tokens.** #128's finding — concatenating
weights / making a wider matmul changes MLX's tiling ⇒ changes the
accumulation order ⇒ greedy tokens drift — is the reason this ticket is
**shape-preserving by construction**: the custom lm_head does not concatenate
or widen anything; it reads the same `[2048, 128000]` table column-by-column
in the same k order, one FP32 accumulator per output, which is the
single-output reduction order the reference uses. §5.2 + §9 K4 are the
load-bearing guards, and the token gate (§5.1 + §9 K3) is the actual hard
gate, not the arithmetic.

**QUEUE #1 (compiled decode) is FALSIFIED and reverted** (`ddce3ffc`) — see
`doc/lfm3-compiled-decode-step.md` §Results. This ticket is independent of
that revert: it changes *which kernels run inside the existing eager
`build_graph`*, not how the step is traced or scheduled. The `Compiled`
scaffolding left in the branch (see QUEUE #1 §"Residual branch state") is
inert and not used by this arm; if a PR reviewer prefers, it may be stripped
in the QUEUE #2 PR as a separate clean-up commit — it is not part of this
ticket's decision.

---

## Results

*(window pending — appended after the locked M4 run; only dated amendments to
the pre-registration above may appear before the verdict.)*