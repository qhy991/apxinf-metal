# Qwen3-0.6B decode: W8 weight path attribution and default switch

Date: 2026-11-10. Hosts: Apple M1 Pro (applegpu_g13s, 16 GiB) and Apple M4
(Mac16,10, 16 GiB). Model: Qwen/Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca`, batch 1, greedy.
Branch `agent/weight-w8-path-388f52f6d7db`; probe `metal_text_probe` with
`APXINF_METAL_TEXT_MODEL=metal_qwen3` and explicit
`APXINF_METAL_TEXT_VARIANT` arms (Issue #68, W8 leg).

## TL;DR

- `mixed-w8` already had the right weight format (W8/group-64 projections at
  one input row, BF16 prefill) but its generation loop fell back to the
  synchronous host-sampled path: per token it downloaded the full
  `[1, 151936]` logits, ran a host argmax, and never overlapped steps. That
  pipeline choice, not dequantization, is what capped the historical gain at
  −13.2% instead of the device-side −28%.
- This leg admits `mixed-w8` to the device greedy path (one-step async
  lookahead) and makes `mixed-w8` the default variant; `bf16-compiled` remains
  the explicit opt-out. Measured paired TPOT (medians, 3 rounds, alternating
  arm order, Swapouts=0 on both hosts):

| Host | `bf16-compiled` | `mixed-w8` (this leg) | Δ TPOT |
|---|---:|---:|---:|
| M1 Pro | 9.809 ms | **6.702 ms** | **−31.7%** |
| M4 | 13.561 ms | **7.582 ms** | **−44.1%** |

- 1024-token consistency: free-running outputs match for the first 82 tokens
  (both hosts); teacher-forced replay (same prefix) gives **96.9% / 97.5%**
  (W8 on BF16 teacher, M1 Pro / M4) and **98.5% / 98.4%** (BF16 on W8 teacher)
  argmax agreement, with disagreements concentrated at near-tie low-confidence
  steps (median top probability there 0.31–0.42; mean |Δp| ≤ 0.0012). On M4
  the same-variant replay noise floor is already 1.4–1.5%, so cross-variant
  disagreement is at that floor; on M1 Pro both variants reproduce their own
  greedy trajectories 100% under replay.

## Where the historical −13.2% went

Decode per token (M1 Pro, serialized stage timing behind
`APXINF_QWEN3_DECODE_STAGES=1`; medians over 96 steady tokens):

| Stage | `bf16-compiled` | `mixed-w8` | Δ |
|---|---:|---:|---:|
| host graph build | 0.198 ms | 0.325 ms | +0.13 ms |
| device eval | 10.262 ms | 7.425 ms | **−2.84 ms (−27.7%)** |

Roofline bookkeeping per token (weights 1.19 GB BF16 → ≈0.62 GB W8+scales):

| Component | `bf16-compiled` | `mixed-w8` |
|---|---:|---:|
| weight-byte floor @200 GB/s | 5.95 ms | ≈3.1 ms |
| kernel-tap GPU busy (corrected) | ≈6.3 ms | ≈4.3 ms |
| kernels / token (tap) | 570 | 628 |
| command buffers / token | 21.6 | 21.1 |
| plain TPOT before this leg | 9.79 ms | 8.20 ms |
| plain TPOT after this leg | 9.78 ms | **6.67 ms** |

Attribution of the old gap:

1. **Device side worked as designed**: halving weight bytes cut device eval by
   27.7% (10.26 → 7.43 ms). Quantized GEMV runs at lower effective bandwidth
   than BF16 GEMV (≈83 vs ≈116 GB/s per weight byte), so the eval saving is
   less than the byte halving — that is the dequantization cost, and it is
   real but not dominant.
2. **The sync generation loop then gave ~1.5 ms/token back**: host argmax over
   151936 logits, ≈300 KB logits download per token, and no lookahead overlap
   (`mixed-w8` TPOT 8.20 ms vs its own device eval 7.43 ms). The same loop on
   `bf16-compiled` (`APXINF_MLX_SYNC_DECODE=1`) costs 11.2–12.5 ms vs 9.78 ms.
3. Fixing the loop (device greedy, this leg) recovers the overlap: W8 TPOT
   6.67 ms, below its serialized device eval because step N+1's eval overlaps
   step N's token readback.
4. What remains above the ≈3.1 ms W8 floor: 628 kernels / 21.1 command
   buffers per token of GPU-side scheduling (≈2.4 ms) plus qmm dequant
   throughput (≈1.2 ms). A compiled W8 decoder block is the next lever.

## Two-machine paired TPOT (acceptance)

3 rounds per host, alternating arm order, fresh process per arm, 3 repeats
each (steady repeats 1–2 reported), `vm_stat` Swapouts delta = 0 on every arm,
both Metal measurement locks held. 32-token outputs identical across repeats
and arms on M1 Pro; on M4 the two variants agree for 31/32 tokens (final-token
near tie, see the M4 table note).

### M1 Pro (applegpu_g13s)

| Round | bf16-compiled TPOT (steady) | mixed-w8 TPOT (steady) |
|---|---|---|
| g5 | 9.866 / 9.879 | 6.722 / 6.661 |
| g6 | 9.736 / 9.785 | 6.715 / 6.688 |
| g7 | 9.832 / 9.732 | 6.718 / 6.671 |
| **median** | **9.809 ms** | **6.702 ms** |

TTFT medians: 46.9 ms (bf16) vs 41.2 ms (w8). Long-run check (1024-token
parity case): 11.0 → 7.59 ms/token (−31%). Rows g5–g7 are the final PR
binary; the pre-rebase probe (identical except inter-repeat cleanup) measured
9.778 → 6.666 ms over rounds g1/g3/g4, within noise.

### M4 (Mac16,10)

Idle-machine re-take (review follow-up on #73): 4 paired rounds, alternating
arm order, `vm_stat` Swapouts delta = 0 on every arm, no probe/build/lane
processes on the mini (standing background is the Logitech Options+ agents,
load ≈2.3–2.5 during the window, identical for both arms):

| Round | bf16-compiled TPOT (steady) | mixed-w8 TPOT (steady) |
|---|---|---|
| t1 | 13.584 / 13.452 | 7.815 / 7.481 |
| t2 | 13.430 / 13.647 | 7.449 / 7.728 |
| t3 | 13.791 / 13.603 | 7.896 / 7.682 |
| t4 | 13.538 / 13.429 | 7.453 / 7.414 |
| **median** | **13.561 ms** | **7.582 ms** |

TTFT medians: 42.2 ms (bf16) vs 33.2 ms (w8). Two earlier suites on the same
branch measured 13.659 → 7.641 ms and 15.007 → 8.392 ms (the latter under
heavier background load); the paired delta is **−44.1% in all three suites**.

The M4's absolute BF16 baseline is slower than the M1 Pro's (13.56 vs 9.81
ms) even when idle — that is the machine, not contention: decode is
weight-bandwidth-bound and the base M4 (Mac16,10) has ≈120 GB/s against the
M1 Pro's ≈200 GB/s, putting the BF16 floor at ≈9.9 ms vs 5.95 ms. Measured
against those floors the M4 is actually closer to its roofline (1.37×) than
the M1 Pro (1.65×), and the weight-byte halving is worth more there. Staged
(serialized) decode medians on M4: build 0.145 → 0.234 ms, device eval
13.992 → 7.914 ms (−43.5%), consistent with the idle re-take TPOTs. Long-run
parity case: 14.55 → 8.27 ms/token (−43%), first divergence at token 82, both
run to 1024. 32-token outputs on M4 agree for 31/32 tokens, diverging only at
the final-token near tie (bf16 31806 vs w8 8357); the same 1024-token
near-tie behavior quantified below. M4 bf16 ids match M1 Pro bf16 ids
exactly.

## Text consistency gate (1024 tokens)

Corpus case: 44-token tutorial prompt, greedy, `max_tokens=1024`
(`devlocal/weight-w8-path/results/parity-corpus.json`). Baseline
`bf16-compiled` output stops at 960 tokens on M1 Pro (EOS) and runs to 1024 on
M4; `mixed-w8` runs to 1024 on both. The free-running pair diverges first at
token 82 (near-tie step) on both hosts. Token-identical over the full output
is therefore not claimed.

Teacher-forced replay (`APXINF_METAL_TEXT_FORCE_TOKENS`, this leg's probe
mode; each arm replays the other arm's ids through prefill + one-row forwards
and reports per-step argmax probabilities):

| Replay | M1 Pro | M4 |
|---|---:|---:|
| bf16 on its own ids (sanity) | 960/960 (100%) | 946/960 (98.5%) |
| w8 on its own ids (sanity) | 1024/1024 (100%) | 1010/1024 (98.6%) |
| **w8 on bf16 teacher** | **930/960 (96.9%)**, max \|Δp\| 0.077 | **936/960 (97.5%)**, max \|Δp\| 0.285 |
| **bf16 on w8 teacher** | **1009/1024 (98.5%)**, max \|Δp\| 0.299 | **1008/1024 (98.4%)**, max \|Δp\| 0.183 |

Mean \|Δp\| is ≤ 0.0012 over every replay. All disagreement steps across both
directions and both hosts sit at low-confidence near-ties (median top
probability 0.31–0.42), i.e. steps where either choice is defensible. On M4
the same-variant sanity replays already disagree at 1.4–1.5% of steps (device
vs host argmax tie-breaking), so the cross-variant 1.6–2.5% disagreement is at
the replay-method noise floor there; the M4 32-token final-token flip is the
same near-tie class, not a W8-specific defect. This matches the historical
acceptance shape for this family (the `bf16-compiled` rounding change was
accepted with 222/1024 free-running token differences and 221/222
external-teacher agreement): W8 quantization is intentionally lossy, and the
default switch ships with the above evidence.

## Changes (branch `agent/weight-w8-path-388f52f6d7db`)

- `perf(metal-qwen3): run mixed-w8 decode on the device greedy path` —
  `token_sequence` admits `MixedW8` to `GreedySequence`; the device decode
  path was already variant-agnostic. `APXINF_MLX_SYNC_DECODE=1` still forces
  the synchronous loop.
- `perf(metal-qwen3): make mixed-w8 the default selection` — decode defaults
  to W8 projections with BF16 prefill; `bf16-compiled` stays as the explicit
  opt-out variant.
- `feat(metal-qwen3): per-token decode stage timing behind
  APXINF_QWEN3_DECODE_STAGES` — host-build/device-eval split per one-row
  step; steps are evaluated synchronously under the flag.
- `feat(diagnostics): teacher-forced replay mode in metal_text_probe` —
  `APXINF_METAL_TEXT_FORCE_TOKENS` for the parity gate above.
- Probe env overrides (`APXINF_METAL_TEXT_MODEL` / `_VARIANT`, `none` cleanup)
  carried over from the previous leg (`e5d9a8eb`).

## Verification

- `cargo test -p apxinf-model --features mlx --release metal_qwen3 -- --test-threads=1 --include-ignored`:
  all 11 metal_qwen3 tests pass on M1 Pro, including the Metal lifecycle tests
  (capacity growth, prepared-state, prefill schedule, reference trajectory).
- 32-token greedy outputs are identical across the two variants on M1 Pro
  (all rounds); on M4 they agree for 31/32 tokens (final-token near tie).
- Kerneltap (DYLD_INSERT_LIBRARIES dispatch/CB tap from the previous leg):
  mixed-w8+greedy steady window shows 628.1 kernels/token, 21.12 CB/token,
  host inter-dispatch gaps p50 0.06 µs; TPOT under the tap 6.694 ms matches
  the untapped 6.666 ms (tap overhead < 0.5%).

## Remaining risks / next candidates

1. **Compiled W8 decoder block** (biggest lever): fold the eager per-layer
   chain (628 kernels, 21 CBs) into a `Compiled` block like `bf16-compiled`'s,
   carrying packed W8 weights as explicit inputs. Expected −1.5~2.5 ms/token
   (GPU-side scheduling ≈2.4 ms). Falsifier: if qmm inside a compiled
   callable forces materialization boundaries and CB count stays ~20, the win
   shrinks to elementwise fusion only.
2. **qmm dequant throughput**: W8 GEMV effective bandwidth ≈83 GB/s vs ≈116
   GB/s for BF16 GEMV; a better M=1 quantized kernel shape would push toward
   the ≈3.1 ms floor.
3. Memory: `mixed-w8` keeps BF16 prefill + W8 decode weights resident
   (≈1.8 GB vs 1.2 GB); documented as a decode-performance selection, not a
   memory saver.

Evidence paths (M1 Pro): `devlocal/weight-w8-path/{logs,results,scripts}` —
paired logs `probe_g*`, staged `probe_staged_*`, parity `parity_*`,
teacher-forced `force_*`, kerneltap `tap_w8_*`. M4 evidence on the M4 host's
worktree `~/Agent4Kernel/apxinf-wt/weight-w8-path-388f52f6d7db/devlocal/`.

## Compiled mixed-W8 decoder block — measured NOT-WORTH (2026-11-10)

The previous section left one candidate: the mixed-W8 decode step issues 628 kernels and 21
command buffers per token, and the M1 Pro sits ~2.4 ms above the ≈3.1 ms W8 weight-byte floor.
The natural next step was to compile the whole decode block (the same tool the `bf16-compiled`
selection already uses) so one callable replaces the per-kernel sequence.

**Implementation**: `decoder_block_w8` mirrors `decoder_block` but takes six
`[packed, scales, biases]` triples (q, k, v, o, fused gate/up, down) plus the four norm weights,
and runs the layer's `quantized_matmul` projections inside the compiled graph. It is prepared once
per attention width by `warm_decoder_w8` (same mask-array scheme as the BF16 block) and selected
only for `MixedW8` decode after prefill, behind `APXINF_QWEN3_W8_BLOCK` (default on in the
experiment branch).

**Measurement** (M1 Pro, `metal_text_probe`, `APXINF_METAL_TEXT_VARIANT=mixed-w8`, one binary,
three alternating rounds, 32-token generation):

| round | eager device decode (block off) | compiled W8 block (on) |
|---|---|---|
| 1 | 6.72 ms | 6.89 ms |
| 2 | 6.68 ms | 6.91 ms |
| 3 | 6.72 ms | 6.82 ms |

The compiled block is **1.5–3 % slower**, consistently, with identical output text. It does what it
was built to do — the per-token host graph build drops from ~0.47 ms to ~0.19 ms — but that saving
does not reach the wall clock: the host build was already only 4.2 % of the token, and the
remaining 2.4 ms lives on the **GPU-side dispatch timeline** (628 kernels / 21 command buffers),
which compiling the graph does not remove. MLX's `Compiled` collapses host-side graph construction;
it does not fuse `quantized_matmul` with its neighbours.

**Conclusion**: the compiled-W8-block direction is closed. The remaining gap to the W8 floor has to
be attacked with fewer GPU kernels (fused quantized GEMV + norm + activation, or a lower dispatch
count per token), not with graph compilation. The code was not kept: it is a rejected direction and
the repository does not carry dead paths.

## Where the remaining gap lives, and the W4 experiment (2026-11-10)

Two measurements before touching kernels.

**A bandwidth probe** (`examples/qw3_gemv_probe.rs`) times MLX's `quantized_matmul` at the six
Qwen3-0.6B decode shapes. Two modes:

| mode | one layer (16.7 MB) | full token (28 layers, 468 MB) |
|---|---:|---:|
| each call synchronized | 0.20–0.23 ms (~9 GB/s) | — |
| 168 calls pipelined, one eval | — | **3.13 / 3.78 / 4.94 ms → 149 / 124 / 95 GB/s** |

The pipelined sweep reaches **149 GB/s, i.e. the ≈3.1 ms W8 floor**, while the real decode runs at
6.7 ms. The difference is not kernel efficiency: the decode chain is **serially dependent** (each
layer consumes the previous one), so every one of the ~628 kernels per token exposes its launch
latency instead of overlapping. That is also why compiling the block did not help (previous
section): `Compiled` collapses host graph construction, not device kernels.

**W4 decode weights** halve the bytes at the same kernel count. `APXINF_QWEN3_W8_BITS=4` selects
the width (8 default, 6 also accepted), one env for the whole decode lane:

| arm | TPOT (32 tokens) | TPOT (192 tokens) |
|---|---:|---:|
| W8 (default) | 6.72 / 6.69 ms | 6.91 ms |
| **W4** | **5.56 / 5.57 ms** | **5.74 ms** |
| delta | **−17 %** | **−17 %** |

Quality: the 32-token answer is textually identical; the 192-token answer shares a 24-token prefix
and then differs in wording while staying coherent and on-topic — the expected shape of 4-bit
noise. W4 stays **opt-in**: a lossy width does not become the default without a teacher-forced
consistency campaign on the same bar the W8 flip used.

**Remaining plan**: with bytes halved the dependency-length part is now the majority of the token
(~4 ms of 5.6 ms), so the next lever is fewer kernels (fused norm+QKV, o+residual, gate/up+SiLU),
not another width change.

## The dispatch bill: where the last 3.8 ms goes (2026-11-10)

`qw3_gemv_probe` grew two modes that separate bandwidth from dispatch:

| probe | work | result |
|---|---|---:|
| chained | 168 `quantized_matmul` back to back, each consuming the previous output | **2.9–4.6 ms** |
| skeleton | 168 × (`rms_norm` + add + SiLU + `slice_update`) on small tensors (~84 MB) | **4.54 ms** |
| sweep | 168 independent `quantized_matmul` (468 MB) | 3.1–4.2 ms (110–149 GB/s) |
| real decode (W8) | — | **6.7 ms** |

The chained probe rules out a dependency tax: a strictly serial chain of the same projections runs
at the bandwidth floor. The skeleton is the finding — those elementwise ops move ~84 MB, which is
~0.4 ms of bandwidth, yet cost **4.54 ms**: about **6.7 µs of fixed dispatch per kernel**, times the
~672 tiny kernels a token issues.

That consistency check also explains the earlier negative results: fusing q/k/v removed 56
*bandwidth* kernels and changed nothing, and compiling the block cannot reduce the device kernel
count at all. The 6.7 ms is roughly 2.9 ms of quantized-matmul bandwidth plus **~3.1 ms of small-op
dispatch**.

So the lever is fusing the *elementwise* work into the projections, by kernel count:

| fusion | kernels | expected saving |
|---|---|---:|
| `add + rms_norm` → one kernel (twice per layer) | 56 → 28 | ~0.19 ms |
| SiLU·mul into the gate/up epilogue | 28 → 0 | ~0.19 ms |
| residual add into the o epilogue | 28 → 0 | ~0.19 ms |
| residual add into the down epilogue | 28 → 0 | ~0.19 ms |
| KV `slice_update` → in-place write | 56 → 28 | ~0.19 ms |
| total | ~460 → ~200 | **~1.7 ms** |

On top of the W4 width (−1.1 ms) that projects to **5.56 → ~2.6–3.0 ms** per token. The order that
keeps each step independently verifiable is: `rms_norm_add`, then the residual epilogues, then
in-place KV, then the SiLU epilogue.

## Fused residual-add + RMS-norm kernel — KEEP (2026-11-10)

First item of the fusion plan: the post-attention site computes `hidden = x + y` and then
`norm(hidden)`. `apxinf_mlx::RmsNormAdd` does both in one launch (one threadgroup per row,
BF16 add rounded before the statistics, F32 weight), and returns the sum as well so the
residual path is unchanged.

| arm (mixed-w8, 192-token generation) | TPOT |
|---|---:|
| add + `fast_rms_norm` (unfused) | 6.88 ms |
| **fused `rms_norm_add`** | **6.52–6.55 ms** |
| delta | **−4.8…−5.0 %** |

Parity: the fused **sum is bit-identical** (`hidden: 0/3072` differ on the unit check), and the
normalized row differs by **one BF16 ULP** at most (`max |delta| 7.6e-3`, 3069/3072 elements) —
the reduction order of the mean differs from MLX's. Over a 192-token generation the outputs are
**192/192 identical**, so the rounding difference does not move a token here. `RmsNormAdd` is
on by default for the Qwen3 decode path; `APXINF_QWEN3_FUSED_NORM=0` restores the two-op path.

Next items of the plan (`o`/`down` residual epilogues, in-place KV, SiLU epilogue) each remove
another ~28 kernels, i.e. ~0.19 ms at the measured 6.7 µs dispatch cost.

## Fusion ledger: one KEEP, three measured negatives (2026-11-10)

The dispatch measurements in the previous section suggested each removed launch was worth ~6.7 µs,
so four fusions were attempted. Only one moved the wall:

| fusion | kernels removed / token | measured TPOT | verdict |
|---|---|---|---|
| **residual add + RMS norm** (post-attention) | 28 | 6.88 → **6.52 ms** | **KEEP** |
| q / k / v projection → one quantized GEMM | 56 | 6.82 → 6.80 ms | **NOT-WORTH** (no change) |
| SiLU(gate) · up from the fused gate/up output | 56–84 | 6.51 → 6.47/6.55 ms | **NOT-WORTH** (no change) |
| KV `slice_update` → in-place write | 28 | not attempted | — |

The pattern: the fusion that helped sat **on the dependency chain between attention and the MLP**,
where its two kernels each had to finish before the next projection could start. The two that did
nothing sit next to kernels that already dominate (the quantized projections), so the removed
launches simply hid behind work that was going to happen anyway. The skeleton probe's 6.7 µs per
kernel is therefore an **upper bound on isolated launches**, not a per-kernel budget in the model.

Consequence for the plan: fusing more small kernels is not where the remaining ~3 ms is. The two
untried structural options keep the same shape as the one that worked — (a) defer the `down`
residual so the *next* layer's input norm consumes it through `RmsNormAdd` (moves one more kernel
off the chain), and (b) in-place KV writes. Both are worth trying; neither is a sure thing.
