# d3-lite Metal — final-layer row prune (P4, direction D3-B)

This is the experiment record for issue #11's D3-B direction
([`notworth-audit.md`](notworth-audit.md) §3 rank 4 / §4.2 D3-B): the readout pools **only the
last position**, so the final layer's first `L−1` output rows are dead work.
The final layer is a full-attention layer, so its query/gate projection, its
attention (one query row against the whole prompt), its output projection and
its MLP can all run on one row. `k`/`v` still cover the whole prompt because
the final attention is non-causal.

- Date: 2026-11-06; host: Apple M1 Pro 16 GB, macOS 27.0.1, MLX 0.32.3
  (`applegpu_g13s`)
- Workload: EngineTailor `testdata/workload.json` — 48 gate cases and the 8
  flagged timing cases; image suite: the 8 P1b cases against the checkpoint's
  own PyTorch/MPS reference
- Discipline: 3 rounds reverse arm order, one Metal lock window per round
  (`~/.cache/enginetailor/metal-measure.lock` then
  `/tmp/enginetailor-metal.lock`), per-arm-process **and** per-decision
  `vm_stat` Swapouts, 48-case text + 8-case image parity, KEEP =
  median ≥ 2 % **and** CI95 lower > 0 **and** parity passes

## 1. The gate

| Gate | Values (first = default **after** this PR) | What it selects |
| --- | --- | --- |
| `APXINF_D3_LAST_ROW` | `full`, `off`, `attn` | How much of the final layer shrinks to the readout's single row. `full`: one q/gate row, one attention row, one `o_proj` row, one MLP row. `off`: the reference shape end to end. `attn`: full attention output, sliced before `o_proj`, so only `o_proj` and the MLP shrink. |

`APXINF_D3_LAST_ROW` is read once at model load like every other d3-lite gate,
so a campaign arm is one process with one fixed configuration.

## 2. Contract (written before measurement)

**Why one row is exact.** The schedule is 24 layers with full attention at
every 4th layer (`is_attention = (layer + 1) % 4 == 0` → layers 3, 7, 11, 15,
19, **23**), so the final layer is attention. The only consumer of the final
layer's output is `stack`'s closing `fast_rms_norm(&weights.norm)` and then
`readout_tail`, which slices `hidden[:, L-1, :]`. RMSNorm is row-local and no
later layer exists, so rows `0..L-2` of layer 23 are dead. The GatedDeltaNet
layers are **not** prunable: their output is recurrent over the prompt, so
every row feeds the next row's state; they keep the full sequence.

**What must stay full.** `k`/`v` (the final attention is explicitly
non-causal, `ATTENTION_MODE = noncausal_full_attention`, so the readout row
attends to every token) and every layer before 23. The image path runs the
same `stack`, its vision features are scattered into the embeddings *before*
the stack, and `run_start` never reaches layer 23 — so it inherits the prune
unchanged.

**Two shape traps, both covered.** The packed q projection also produces the
output gate, so `q` and `gate` are sliced together. And the rotary position
must be the prompt's last position (`L−1`), never a fresh one-row range (which
would silently rotate the readout row as position 0): text slices the `[1, L]`
positions to `[1, 1]` holding `L−1`, image slices the mRoPE cos/sin `[L, 64]`
tables on axis 0. `last_row_slice_keeps_the_final_position` pins both, and the
invariant is also enforced by the kernels' own shape contracts
(`HalfSplitRope::call` requires `positions.shape() == [B, T]`, `mrope_apply`
requires `cos.shape() == [length, rotary]`).

**What changes numerically.** `full` turns the final layer's q projection into
a GEMV, dispatches a one-query-row SDPA, and turns `o_proj` + the three MLP
GEMMs into GEMVs; `attn` leaves the q projection and attention activation
order untouched and only shrinks `o_proj`/MLP. Both are layer-23-only, so any
ulp-level difference passes through one row-local norm, the FP32 readout and
the softmax — no later layer amplifies it.

**Expected gain.** The final layer is ~1/24 of the backbone GEMM work, and
`full` deletes ~75 % of that layer's GEMMs (q, o_proj, MLP) plus ~`L`× the
attention FLOPs; the ceiling is ~4 % end-to-end for long prompts and close to
zero for very short ones (weight streaming dominates a GEMV).

## 3. Parity

The `off` arm is the reference shape and must reproduce the shipped
pre-change receipts bit for bit; the pruned arm must clear both declared
gates.

### 3.1 48-case text gate (EngineTailor `gate.py`, FP32 oracle + stock anchor)

| arm | verdict | worst max\|Δp\| vs oracle | budget (2× stock) | mean KL | KL budget | decisions | byte-identical |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `off` | **passed** | 0.024349 | 0.037659 | 2.066e-4 | 3.978e-4 | 48/48 | yes |
| `full` | **passed** | 0.023195 | 0.037659 | 2.021e-4 | 3.978e-4 | 48/48 | yes |
| `attn` | **passed** | 0.024349 | 0.037659 | 2.070e-4 | 3.978e-4 | 48/48 | yes |

Worst per-case drift `full` vs `off`: **0.004889** (`b2400_vision_topic`). The
pruned arm is marginally *closer* to the FP32 oracle than the reference-shape
arm on the worst case; no decision moved. `attn` (the drift-isolating arm)
drifts **0.000262** from `off` on text and **1.2e-5** on image, i.e. the
`full` arm's drift is dominated by the one-row q projection + one-row SDPA,
not by the GEMV rounding of `o_proj`/MLP.

### 3.2 8-case image gate (checkpoint PyTorch/MPS reference)

| arm | decisions | worst max\|Δp\| | declared budget 0.06 | strict gate 0.0575 | headroom to 0.06 | byte-identical |
| --- | ---: | ---: | --- | --- | ---: | --- |
| `off` | 8/8 | 0.055385 | pass | pass | 0.004615 | yes |
| `full` | 8/8 | 0.055251 | pass | pass | 0.004749 | yes |
| `attn` | 8/8 | 0.055385 | pass | pass | 0.004615 | yes |

Worst per-case drift `full` vs `off`: **0.001547**
(`mario_obstacle_pos_f09`). The image budget's ~0.0046 of headroom is
**not** spent: the pruned arm's worst case is 0.000134 *closer* to the
reference than the unpruned arm's, and the strict 0.0575 gate this direction
pre-declared passes on both arms.

### 3.3 Gate-off inertness

`off` vs the pre-change shipped receipts (the D3-A KEEP arm recorded before
this change): text **0.0**, image **0.0** (bit for bit). The new code path is
inert when the gate is absent.

## 4. Latency — 3-round paired campaign

Arms differ only in `APXINF_D3_LAST_ROW` (`off` vs `full`); the other gates
are pinned to the shipped defaults (`APXINF_D3_CONV1D=apxinf`,
`APXINF_D3_GDN_CORE=eight_lanes`, `APXINF_D3_GDN_CORE_IMAGE=four_values`).
One arm = one `d3_lite_bench` process over the 8 flagged timing cases
(warmup 1, 5 samples/case); even rounds run the baseline first, odd rounds the
literal reverse order; every arm of a round runs serially inside one lock
window.

- Clean rounds: **2 of 3** (a round is clean only when both arm processes
  and the bench's per-decision Swapouts are 0). Round 0 is excluded: its
  `full` arm swapped **+4071** pages (the bench's per-decision gate was
  unclean too); both arms were clean in rounds 1 and 2.
- Paired per-case gains (16 paired samples over the clean rounds):
  **median +4.374 %**, CI95 **[+4.124, +4.511] %**, **16/16** cases faster.
- Per-round medians: **+4.17 %** (round 1), **+4.50 %** (round 2).
- Per-case medians: **+2.31 %** (`b0150_quickstart_route`, 150 tokens) …
  **+4.97 %** (`b2400_suite_kind`, 2400 tokens); every case faster.
- A/A: `off` reproduced its probabilities bit for bit across clean rounds
  (max|Δp| **0.0**); `full` vs `off` on the timing cases max|Δp| =
  **0.001491** (ulp-level drift, no decision change).

| arm | median per-case latency | gate |
| --- | ---: | --- |
| `off` | 314.79 ms | reference shape |
| `full` | 301.13 ms | **−4.34 %** (pooled paired median −4.374 %) |

**Verdict: KEEP** — median ≥ 2 %, CI95 lower > 0, both parity suites pass, and
the gate-off path is bit-identical to the reference receipts.

## 5. Operator timeline (before / after)

[`d3-lite-last-row-timeline.png`](d3-lite-last-row-timeline.png): the same
2400-token text request recorded with xctrace (Metal System Trace + Points of
Interest) on both arms, one Metal lock window each, both with `vm_stat`
Swapouts **+0**. One request in the 0–1098 ms shared scale, host stages lane,
GPU command-buffer lane, and the 10 ms both/GPU-only/host-only/none strip.

| metric | before (`off`) | after (`full`) | Δ |
| --- | ---: | ---: | ---: |
| window ms | 1066.1 | 1019.2 | **−46.9 (−4.40 %)** |
| stage sum (input+vision+forward+sync) | 1066.1 | 1019.2 | −46.9 |
| forward ms | 1066.0 | 1019.1 | −46.9 |
| GPU busy % | 99.56 | 98.86 | −0.70 |
| host busy % | 100.00 | 100.00 | 0.00 |
| host-only stall % | 0.44 | 1.14 | +0.70 |
| GPU-only uncovered % | 0.00 | 0.00 | 0.00 |
| true bubble % | 0.00 | 0.00 | 0.00 |
| GPU command buffers | 62 | 59 | −3 |

Reading: the window shrinks by 4.4 % while the bubble stays at 0 % and GPU
busy stays ~99 %, so this is **deleted GPU work**, not a scheduling or
overlap change. The +0.7 pp host-only stall is the fixed per-request launch
gap becoming a larger fraction of a shorter window.

## 6. Limits

- The gain scales with prompt length: 2.3 % on the 150-token case, 4.97 % on
  the 2400-token case (the layer's GEMM work grows with `L` while its weight
  streaming does not). Short prompts gain little.
- The image path is measured for parity only; the same prune applies to it,
  but its latency was not the target of this campaign (the demo path is image
  plus a short prompt, where the expected gain is small).
- Steady state only: the campaign warms the compiled profile. The extra GEMV
  shapes may add JIT/prepare work on the very first request of a profile; the
  first-decision budget is not measured here.
- Latency is this host's clean-Swapouts window, not a cross-machine claim.
- Opus adjudication: one `-e medium -t 300` call, run by the relay partner
  before this leg started (98 s, 8610 B, `devlocal/notworth-audit/d3b/opus-design.md`).
  Its ranking (implement `attn`-style first, treat the one-row SDPA kernel as
  the largest drift source, pre-declare a stricter image gate, run the
  gate-off drift probe) shaped this design; the contract statements above were
  then re-derived from the code in this worktree.

## 7. Evidence

- Implementation: `crates/apxinf-model/src/metal_d3_lite/model.rs`
  (`APXINF_D3_LAST_ROW`, `last_row_slice`, `stack`, `attention_forward`).
- Parity receipts: `devlocal/d3-last-row/results/parity/`
  (`parity.json`, per-arm `text/` and `image/` receipts).
- Campaign: `devlocal/d3-last-row/results/last-row/summary.json`.
- Timeline: `devlocal/d3-last-row/results/timeline/last-row/{before,after}/`
  (`op.trace`, `export/metrics.json`, `record_meta.json`).
- Runners: `devlocal/d3-last-row/scripts/run_last_row_parity.py`,
  `run_last_row_campaign.py`, `record_d3_timeline.py`, `render_d3_timeline.py`.
- Tests: `cargo test -p apxinf-model --features mlx --lib metal_d3_lite::model::tests`
  (5 passed, incl. `last_row_slice_keeps_the_final_position`) and, with the
  shipped default (`full`) under the Metal lock,
  `compiled_forward_matches_eager_replays_and_binds_changed_inputs` and
  `image_decision_matches_pytorch_reference_through_the_public_path`
  (`--ignored`, both pass).
