# d3-lite Metal port (P1a): architecture mapping, kernel reuse, parity and baseline

This is the port report for issue #11 experiment 1 (P1a): the d3-lite text
path implemented as an ApxInf-owned MLX/Metal backend inside `apxinf-metal`,
its parity against the reference receipts, and the **unoptimized** M1 Pro
baseline. No optimization is claimed or performed in this phase.

## 1. Port definition

| Item | Value |
| --- | --- |
| Checkpoint | `~/Documents/Infinity/Agent4Kernel/models/d3-lite-9c10703` (`vllm-sr/d3-lite`, manifest `model_sha256=fda5cfc7…`, 1.6 GiB) |
| Backbone | Qwen3.5-0.8B text config: 24 layers (18 GatedDeltaNet + 6 non-causal full attention), hidden 1024, BF16 as stored |
| Readout | FP32 `readout.safetensors['weight']` `[255, 1024]`, applied at the last position, temperature 1, masked softmax |
| Target | Apple M1 Pro, `Device::Metal(0)`, native MLX 0.32.3 (`MLX_ROOT=~/apxinf-runtime/mlx-sdk`), release bench binary |
| Metal device string | `Apple M1 Pro (applegpu_g13s)` as reported by the MLX stream |
| Public API | `AutoModel` → `LoadedModel::Decision(DecisionModel::Code(_))`, `CodeDecisionRuntime`, CLI `apxinf decide --device metal[:N]` |
| Reference receipts | EngineTailor `enginetailor/models/d3lite` (`graft@2e54f7f`): `collect.py` **oracle** (all weights/readout in F32) and **stock** (BF16, eager, stock kernels) over `testdata/workload.json`. Its README records the executor, oracle, workload, gate and promoted point as coming from Graft's M1 Pro runs and byte-identical to Graft's checkpoint-reference receipts. |
| Workload | 48 gate cases (12 per bucket 150/400/1000/2400; `option_count` ∈ {2,3,4,22}) + the 8 flagged timing cases; `codes=255`, tolerance 0.15 |
| Gates | R1: all decisions agree with the oracle wherever stock does (no near-ties in this set). R2: worst max\|Δp\| ≤ max(2×stock, 1e-4) and mean KL ≤ max(2×stock, 1e-7). R3: two runs byte-identical and rows finite/summing to 1. |
| Latency | baseline only: warm requests, per-case medians, `vm_stat` Swapouts/Swapins deltas per round; non-zero Swapouts ⇒ contaminated |

Provenance note: the checkpoint's own `d3_engine.py`/`d3_runtime.py` is a
PyTorch package (CUDA/CPU) and cannot run on this Apple host without a large
Torch install; the maintained FP32 oracle used here is the EngineTailor MLX
collector, whose receipts are documented as byte-identical to the
checkpoint-reference receipts. The checkpoint's `modeling_d3.py` semantics are
what both implement.

Copy-not-share note: the request allowed reusing the CUDA `d3` family's
config/weights parsing if convenient. The repository's backend-native port rule
(`model-layer-architecture.md`, "Where backend-native model code lives") and
`scripts/check_model_family_boundaries.sh` require a port to be self-contained
and never import another model-family directory, including a port of the same
family. The checkpoint contract is therefore copied into this port (with the
d3-lite geometry that the CUDA copy does not accept) rather than imported.

## 2. Architecture mapping

| Responsibility | Reference (EngineTailor MLX) | This port (`metal_d3_lite`) |
| --- | --- | --- |
| Config / package identity | `engine.load`: `config.json` `text_config` + `decision_config.json` | `config.rs`: `Config::from_dir` parses `config.json`, `decision_config.json`, `MODEL_MANIFEST.json`; exact d3-lite geometry, `layer_types` schedule, manifest parameters and embedded decision identity validated; `vision_config` pinned (depth 12 / 768 / out 1024) but not executed |
| Weight mapping | strip `language_model.`; `conv1d` `[C,1,S]→[C,S,1]`; `1+w` for input/post/final and q/k norms | `weights.rs`: `safetensors_files` + `apxinf_mlx::load_safetensors` (keep `language_model.*` and readout `weight`; `visual.*` never read); typed `Take` shape/dtype checks and `finish()` leftover rejection; same fixed transforms once at load |
| Embedding | `embed_tokens[ids]` | `Weights::embed.take(ids, 0)` |
| Layer composition | pre-norm → mixer → residual → post-norm → MLP → residual | `forward()` in `model.rs`, single owner of order |
| GDN layer | packed `in_proj_qkv/z/b/a`, depthwise causal conv over 3 zero rows, SiLU, official q/k l2-norm, recurrence, gated RMSNorm, `out_proj` | `gated_delta_forward()`: `GatedDeltaRecurrence` (apxinf four-values Metal kernel) + MLX `conv1d` with explicit zero history; `decay_beta()` = `exp(-exp(A_log_f32)·softplus(a+dt_bias))`, `sigmoid(b)` |
| Full-attention layer | packed q+gate split, q/k norm, partial RoPE, non-causal SDPA, output gate, `o_proj` | `attention_forward()`: `HalfSplitRope` over 64 of 256 dims; `sdpa(..., causal=false, mask=None)` |
| MLP | `down(silu(gate)·up)` | `mlp_forward()` |
| Readout | last position → F32 → `@ readout.T`; mask `index ≥ count` to −∞; FP32 softmax | `forward()` tail with `Array::where_select` and `count <= index` |
| Request lifecycle | `Arm.probabilities(ids, count)`; `mx.compile` whole forward when selected | `Model::probabilities`: validate ids/count, `Compiled::prepare` one `(length, count)` profile on first use, replay without retracing, one output download; `Model::eager_probabilities` keeps the eager body for acceptance tests |
| Decision output | `argmax` (first tie), first `count` probabilities | `D3LiteRunner::decide`: first-maximum, `DecisionScore` window, `LoadedModel::Decision(DecisionModel::Code(_))` |
| Gate / bench | `collect.py` → `probs.json`, `gate.py` R1/R2/R3, `bench.py` ABBA/BAAB | [`examples/d3_lite_bench.rs`](../crates/apxinf-model/examples/d3_lite_bench.rs): reusable CLI (`--model-dir`, `--set`, `--bucket`, `--warmup/--repeat/--rounds`, `--reference`); `--records` writes a gate-compatible receipt, `--summary` writes parity, latency and swap aggregates |

## 3. Kernel reuse

| Semantics | Asset | Classification | Use here |
| --- | --- | --- | --- |
| Gated delta recurrence | `apxinf-mlx/src/gated_delta.rs` (`apxinf_gated_delta_four_values`) | fused custom Metal kernel; BF16 q/k/v/beta, FP32 g/state, `[1,Hv,Dv,Dk]` state | **used** in all 18 GDN layers; shape-compatible (Dk=Dv=128, Hk=Hv=16, Hv multiple of Hk) |
| Half-split partial RoPE | `apxinf-mlx/src/half_split_rope.rs` (`apxinf_rope_half_split`) | fused custom Metal kernel; FP32 angles, one rounding to input dtype | **used** on attention q and k (rotary dim 64) |
| Depthwise causal conv | no ApxInf kernel; `Array::conv1d` primitive | primitive: explicit `S-1` zero history + `groups=C`, `stride=1,padding=0,dilation=1` | **used** as the unoptimized baseline. The FP32 tap-order `causal_depthwise_conv` kernel exists only in the private EngineTailor reference; porting it is P3 |
| RMSNorm | `Array::fast_rms_norm` / `fast_rms_norm_unweighted` | fused stock MLX | **used** for all norms. GDN q/k l2-norm uses mean epsilon `1e-6/k_dim` (official `x·rsqrt(sum x²+1e-6)`), not the mlx-lm mean-epsilon variant |
| SDPA | `Array::sdpa` | fused stock MLX fast attention | **used**, explicitly non-causal |
| Matmul | `Array::matmul` | primitive | **used** for every projection and the FP32 readout |
| Elementwise | `sigmoid`, `mul`, `exp`, `logaddexp`, `where_select`, … | primitives | **used** (SiLU, softplus, masking) |
| Safetensors → Array | `apxinf_mlx::load_safetensors` | primitive bridge, no host staging, evaluated per file | **used** at load; vision tensors excluded by the keep-filter |
| Compiled execution | `Compiled::with_shapeless` / `prepare` / `call_and_eval` | prepared compiled callable (MLX compile; **not** a CUDA Graph) | **used** as the whole-request execution: `(ids [1,L], count [1]) → [1,255]` probabilities |

Not ported (out of scope for an unoptimized baseline; these are the
EngineTailor `PROMOTED` optimization points for P3): `gated_delta_eight_lanes`,
`causal_depthwise_conv`, `gdn_fused_prep`, and F16 weight copies
(`gemm_dtype=f16`). Consequently this baseline contains no F16 drift.

## 4. Registration and public path

- Module `crates/apxinf-model/src/metal_d3_lite/`, gated `mlx + macOS/aarch64` in `lib.rs`.
- Registry names `metal_d3` (checkpoint detection `d3` → `metal_registry_name`) and `metal_d3_lite` (explicit `--model-name`), registered in `builtin.rs`.
- `AutoModel`: the CUDA-only d3 intercept now applies to **non-Metal** devices, so `Device::Metal` reaches the registered Metal factory; CUDA/CPU requests keep the original behaviour.
- Admission: `model_variant ∈ {auto,bf16,bf16-compiled}`, precision `{auto,bf16}`; CUDA tuning/calibration/synthetic/assets and the d3 CUDA-graph profile knobs are rejected explicitly.
- CLI needs no change: `apxinf decide --model MODEL --device metal:0 --prompt … [--candidates N]`.

## 5. Parity — 48 gate cases

Run: release `d3_lite_bench --set gate --bucket all --warmup 1 --repeat 2
--rounds 1`, receipt compared to `oracle-gate` (FP32) and then through the
reference `gate.py` with `--oracle oracle-gate --stock stock-gate --candidate
native/gate/probs.json`.

| Measure | Native (BF16 + compiled) | Stock anchor (BF16, eager) | Gate limit | Result |
| --- | ---: | ---: | ---: | --- |
| Decisions agreeing with oracle | **48 / 48** | 48 / 48 | all non-near-tie cases (48) | **R1 pass**, 0 failures, 0 flips |
| Worst max\|Δp\| vs oracle | **0.0291797** | 0.0188293 | ≤ 0.0376587 (2× stock) | **R2 pass** |
| Mean case max\|Δp\| | 0.0056734 | — | — | — |
| Mean KL(oracle ‖ candidate) | **2.4487e-4** | 1.9891e-4 | ≤ 3.9783e-4 | **R2 pass** |
| Two runs byte-identical | **yes** | yes | required | **R3 pass** |
| Rows finite / sum ≈ 1 | yes / ≤1.1e-8 | yes | 1e-6 | pass |
| `gate.py` verdict | **`passed (numerics differ; within budget)`** | — | — | exact = false |

Worst case by max\|Δp\|: `b1000_suite_topic` (0.02918, KL 2.73e-3) — decision
still agrees (index 17); the next cases are 0.01758 and 0.01726, so one case
dominates the maximum while the mean stays at 0.00567.

**Drift attribution (bf16 vs the FP32 oracle).** All weights and the readout
are BF16 in the native and stock arms; the oracle promotes everything to F32.
The 48 decisions are identical across all three arms, so the difference is
arithmetic, not logic: FP32 accumulations and norms in the oracle versus BF16
matmul/norm rounding in both BF16 arms, plus two BF16-arm differences from
each other — (a) the whole forward runs as one `mx.compile`d graph rather than
an eager lazy graph, and (b) the apxinf `GatedDeltaRecurrence` and
`HalfSplitRope` kernels use their own FP32 SIMD reduction order and single
output rounding where the stock arm uses the vendored kernel and
`mx.fast.rope`. No F16 weights are involved (`gemm_dtype=f16` is not ported).
The result stays inside the declared 2×-stock budget on both the maximum and
the mean.

## 6. Baseline — 8 timing cases (M1 Pro)

Run: release `d3_lite_bench --set timing --bucket all --warmup 2 --repeat 1
--rounds 7`; per-round `vm_stat` snapshots around each timed decision. **All
56 timed samples had Swapouts = 0** (max Swapins = 541 pages), so the window
is clean and these numbers are KEEP-eligible as a baseline. This phase is a
**single-arm baseline**: there is no comparison arm, so the paired
same-lock-window ABBA/BAAB and reverse-order rules apply from P3 onward; the
per-round swap gate here is the part that carries over unchanged. Latency is
the full public decision call: host input upload → compiled forward → output
readback, excluding model load, warmup and profile tracing.

| Case | Bucket | Tokens | Options | Median (ms) | Min (ms) | Max (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `b0150_engines_kind` | 150 | 129 | 4 | 84.65 | 84.53 | 85.89 |
| `b0150_quickstart_route` | 150 | 128 | 3 | 65.63 | 65.06 | 66.18 |
| `b0400_license_kind` | 400 | 397 | 4 | 199.65 | 198.92 | 201.02 |
| `b0400_kitreadme_topic` | 400 | 438 | 22 | 207.29 | 205.44 | 207.81 |
| `b1000_suite_kind` | 1000 | 974 | 4 | 449.04 | 447.59 | 451.25 |
| `b1000_readme_topic` | 1000 | 1005 | 22 | 454.64 | 453.15 | 455.05 |
| `b2400_suite_kind` | 2400 | 2411 | 4 | 1116.36 | 1114.18 | 1123.22 |
| `b2400_readme_topic` | 2400 | 2404 | 22 | 1116.09 | 1111.07 | 1120.33 |

Per-bucket medians (2 cases × 7 rounds): **150 → 84.5 ms**, **400 → 205.4
ms**, **1000 → 453.2 ms**, **2400 → 1116.4 ms**. Overall across the 56
samples: median **447.6 ms**, mean 461.9 ms, p95 **1119.5 ms**.

Context: the 48-case gate window recorded +4528 Swapouts in one round and is
therefore **not** used for latency claims (gate cases are statistics-only,
never a timing arm). No operator timeline is claimed this phase — the signpost
timeline methodology (Clef `TIMELINE-OPERATOR.md`) applies once a before/after
timeline is required in P3; this report leaves the baseline numbers as the
"before".

## 7. Verification

| Check | Result |
| --- | --- |
| `cargo check -p apxinf-model --features mlx --lib --examples` | pass |
| `cargo test -p apxinf-model --features mlx --lib metal_d3_lite` | 9 passed, 1 ignored |
| `APXINF_D3_CHECKPOINT=… cargo test … compiled_forward_matches_eager -- --ignored` (Metal) | **pass**: compiled vs eager max\|Δp\| < 1e-4; replay adds 0 Rust trace callbacks and reproduces identical bits; changed token IDs change the output; second length traces exactly once; prepared profiles `[(37,255),(64,255)]` |
| `bash scripts/check_model_family_boundaries.sh` | pass |
| `d3_lite_bench --set gate` (48) + reference `gate.py` | **passed (numerics differ; within budget)** |
| `d3_lite_bench --set timing` (8 × 7 rounds) | pass, Swapouts = 0 on every sample |
| CLI `apxinf decide --device metal:0` | exercised separately (see PR) |

## 8. Known limitations

- Vision tower not executed in this phase (`capabilities() == TEXT_ONLY`);
  `visual.*` was parsed only for package identity. **Since delivered as P1b:**
  see the [vision report](d3-lite-metal-vision.md). The text path (and this
  report's parity/baseline numbers) is unchanged — the P1a gate receipt is
  reproduced bit for bit.
- Whole-request MLX compilation is a **new scope** for this family (no prior
  MLX family compiles the entire backbone); it is qualified by the
  compiled/eager acceptance test and by the 48-case gate, not inherited from
  Clef or Qwen3. Each new `(length, candidate_count)` profile traces once, so
  a workload with many distinct lengths pays one trace per length and retains
  one graph per profile. (P1b adds a second compiled callable for image
  requests with its own `(length, candidate_count, image_tokens)` profiles.)
- `inv_freq` is evaluated on the host at load (`apxinf-mlx` exposes no `log`
  unary); the formula is the reference's `exp(-(i/d)·log(base))` in F32. Any
  ulp-level difference from the device-side evaluation is inside the reported
  parity budget. The P1b vision path uses the reference's `powf` form instead
  (bit-exact), deliberately leaving this text formula untouched.
- Per-case timing reflects this host's current background load; the clean
  Swapouts window is recorded, but the suite is not a cross-machine claim.

## 9. Artifacts

Private reproducibility evidence (ignored `devlocal/d3-lite-align/`): reference
oracle/stock receipts, native gate/timing receipts and summaries,
`gate.py` verdict, and the run script. Paths and revisions are recorded there;
nothing private or generated is committed.
