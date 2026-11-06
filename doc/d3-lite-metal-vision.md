# d3-lite vision port (P1b): architecture mapping, kernel reuse, parity and baseline

This is the port report for issue #11 experiment 1, stage **P1b**: the
checkpoint's vision tower wired into the ApxInf-owned MLX/Metal backend
(`metal_d3_lite`), its parity against the checkpoint's **own PyTorch reference**
(`pipeline_d3.py` / `modeling_d3.py` through transformers 5.17.0), and the
**unoptimized** M1 Pro baseline for an image request. The text path is unchanged
from [P1a](d3-lite-metal.md); this phase performs no optimization.

## 1. Port definition

| Item | Value |
| --- | --- |
| Checkpoint | `~/Documents/Infinity/Agent4Kernel/models/d3-lite-9c10703` (`vllm-sr/d3-lite`, manifest `model_sha256=fda5cfc7…`, 1.6 GiB) |
| Vision tower | `qwen3_5_vision`: depth 12, hidden 768, 12 heads × 64, intermediate 3072, patch 16, temporal patch 2, spatial merge 2, out_hidden 1024, learned table 2304 (48 × 48), BF16 as stored, **153 tensors** |
| Preprocessor | `preprocessor_config.json` + the checkpoint runtime's budget: factor 32, min 65 536 px, max 1 638 400 px, mean/std `[0.5, 0.5, 0.5]`, patch row 1536 = 3·2·16·16, spatial-merge-block order |
| Target | Apple M1 Pro, `Device::Metal(0)`, native MLX 0.32.3 (`MLX_ROOT=~/apxinf-runtime/mlx-sdk`), release bench binary |
| Public API | `AutoModel` → `LoadedModel::Decision(DecisionModel::Code(_))`, `capabilities() == VISION`, `D3LiteRunner::decide(DecisionInput::with_vision(…))`, CLI `apxinf decide --device metal:0` (text only; the image request enters through `DecisionInput`) |
| Reference | The checkpoint's own runtime: `d3_runtime.D3.from_pretrained(…, device=mps\|cpu)` → `modeling_d3.py` → stock `Qwen3_5Model.visual` (transformers **5.17.0**, torch **2.12.1**, BF16 backbone, FP32 readout). MPS is the parity baseline; CPU is a secondary arm used only to measure the reference's own device-to-device noise |
| Cases | **8 image requests**, one still image each: 6 Mario crops from the Clef suite (`frames/00xx/{enemy,obstacle,gap}.png`, ground-truth labels available) + 2 generic repository test images. `option_count` ∈ {2, 3}; lengths 238–723; image tokens 108–625 |
| Gates | R1: all 8 decisions agree with the reference. R2: max\|Δp\| ≤ **0.06** (declared: 3× the reference's own MPS↔CPU noise of 0.0201, see §5). R3: repeated runs byte-identical, rows finite and summing to 1. Preprocessor: element-wise exact for PNG (the declared hard check) |
| Latency | Single-arm baseline: one window, one `vm_stat` snapshot per series per round, `Swapouts = 0` required. `text` and `text_plus_image` are reported **separately** (requester's baseline rule); the headline arm is `text_plus_image` |

Reference-environment note: the checkpoint's runtime was exercised with
transformers 5.17.0 exactly as `requirements.txt` pins it. No CUDA dependency
was installed; the PyTorch kernels that would be GPU-only (`flash-linear-attention`,
`causal_conv1d`) are absent, so the reference uses its own slower PyTorch
GatedDeltaNet/conv implementations on MPS or CPU, as intended for a small-case
reference.

## 2. Architecture mapping

| Responsibility | Reference (checkpoint PyTorch) | This port (`metal_d3_lite`) |
| --- | --- | --- |
| Package identity + geometry | `config.json` `vision_config`, `decision_config.json`, `MODEL_MANIFEST.json` | `config.rs`: parses and validates the full vision geometry, the 4 multimodal token IDs, the preprocessor contract and the manifest; 153-tensor identity pinned |
| Image canonicalization | `Qwen3VLProcessor` + `Qwen2VLImageProcessor` (torchvision backend): `smart_resize` → `tvF.resize` bicubic antialiased → **fused** rescale+normalize → `patchify` | `preprocess.rs`: host-CPU replica — interleaved HWC in; CHW separable bicubic with PyTorch's int16 fixed-point weights (`a = −0.5`, horizontal then vertical, `u8` intermediates); fused normalization `(u8 − 127.5) / 127.5`; merge-block patchify with temporal duplication |
| Patch embedding | `Conv3d(kernel=stride)` replaced by its matrix product (`d3_runtime.linearize_patch_embed` → `F.linear`) | `Vision::forward`: `pixels [N,1536] @ patch_weightᵀ [1536,768] + bias`, weight reshaped once at load |
| Learned position table | `get_vision_interpolation_indices_and_weights` (bilinear, `align_corners=True`, border padding) + `nn.Embedding` gather, weighted F32 sum, cast to BF16 | `Vision::interpolation`: host-built `[N,4]` taps → `pos_embed.take` → F32 cast, multiply, `sum(1)`, cast to BF16 |
| Vision blocks ×12 | `Qwen3_5VisionBlock`: LayerNorm → qkv → axial RoPE (FP32, one cast to BF16) → non-causal SDPA per `cu_seqlens` segment → out proj → residual → LayerNorm → `gelu_pytorch_tanh` MLP → residual | `vision.rs::vision_block`: `fast_layer_norm`, `biased_project`, primitive RoPE recomposed as `[H | W | H | W]`, `sdpa` per image segment, `proj`, residual; MLP activation composed in F32 and rounded once to BF16 |
| Spatial merger | `merger`: LayerNorm(768) → `view(−1,3072)` (four consecutive patches = one merge block) → fc1 → `nn.GELU()` → fc2 → `[M,1024]` | `merger_forward`: `fast_layer_norm` → `reshape([M,3072])` → `biased_project` → exact erf-GELU in F32 → `biased_project` |
| Placeholder scatter | `embeds.masked_scatter(image_token_id mask, pooled)` | `Model::image_probabilities` computes slot indices on the host; `image_forward` writes one contiguous run with `embeds.slice_update(features, run_start, axis=1)` |
| Multimodal positions | `compute_3d_position_ids` → `get_rope_index` from `mm_token_type_ids` + `image_grid_thw`: text advances 1/token; an image run takes H/W grid coordinates and advances `max(h,w)/merge` | `multimodal_layout`: same walk from `token_ids` (`image_token_id` marks a run) and `grid_thw`; returns `[3,L]` positions + `[M]` slots. Verified element-identical against the reference receipt for all 8 cases |
| Text layers under mRoPE | `PromptEncoder(embeds, attention_mask, mrope)`: `Qwen3_5TextRotaryEmbedding.recomposition_frequencies` (interleaved sections, `i % 3`), cos/sin cast to BF16; GDN layers positional-free | `stack` + `RopeMode::Mrope`: host positions `[3,L]` × the 32 inverse frequencies, `[T | H | W]` recomposition, duplicate over 64, cast to BF16; `mrope_apply` rotates the first 64 of 256 head dims. The stack body is the same code the text path uses |
| Readout | `hidden.float() @ readout.T`, mask `index ≥ count` to −∞, FP32 softmax | `readout_tail` (shared with the text callable), unchanged from P1a |
| Execution | Eager per request | New compiled callable `(ids [1,L], positions [3,L], count [1], features [1,M,D], run_start [1]) → [1,255]`, one `(length, count, image_tokens)` profile traced on first use via `Compiled::prepare` and replayed without retracing. The **text** callable and its `(length, count)` profiles are untouched |
| Gate / bench | `D3.system_one(state, questions, images=[…])` | `examples/d3_lite_bench.rs`: `--set image` / `--images SUITE`, parity vs `--reference`, preprocessor element-wise rows, and the `text` vs `text_plus_image` timing split |

## 3. Kernel reuse

| Semantics | Asset | Classification | Use here |
| --- | --- | --- | --- |
| Patch embed (Conv3d = matrix product) | `Array::matmul` | primitive | **used**; ≤ 1 BF16 ulp vs the reference's `F.linear` |
| Position-table resample | `Array::take` + `Array::sum` | primitive | **used**; host taps, device gather and F32 weighted sum |
| Vision axial RoPE | `Array::cos` / `sin` / `concat` | primitives | **used**; `[H | W | H | W]` recomposition in F32, one cast to BF16 |
| Vision attention | `Array::sdpa` | fused stock MLX | **used**, explicitly non-causal, one segment per image (the reference's `cu_seqlens` split) |
| Vision LayerNorm | `Array::fast_layer_norm` | fused stock MLX | **used** for all 24 block norms and the merger norm |
| Vision MLP activation (`gelu_pytorch_tanh`) | composed from `Array::sigmoid` | primitive composition (`tanh(z) = 1 − 2·sigmoid(−2z)`) | **used** in F32 with one cast to BF16; ≤ 1 ulp vs torch's `tanh` |
| Merger activation (`nn.GELU()`) | `Array::erf` | primitive | **used** exact erf GELU in F32 |
| Scatter into placeholders | `Array::slice_update` with a dynamic start array | primitive | **used**, one start index (one contiguous image run) |
| Gated-delta recurrence | `apxinf-mlx/src/gated_delta.rs` | fused custom Metal kernel | **used** unchanged in the image request's 18 GDN layers |
| Text partial RoPE | `apxinf-mlx/src/half_split_rope.rs` | fused custom Metal kernel | **used** by the text callable; the image callable uses the mRoPE composition above for the same rotary block |
| Depthwise causal conv, RMSNorm, matmul, masking | MLX primitives | primitives | **used** unchanged from P1a |
| Whole-request compile | `Compiled::prepare` / `call_and_eval` | prepared compiled callable | **used**: the image callable above, prepared per `(length, count, image_tokens)`; features are evaluated before the call so a prepared profile replays without a Rust trace callback |
| Safetensors → Array | `apxinf_mlx::load_safetensors` | primitive bridge, no host staging | **used**; a second filtered read now also keeps `visual.*` |

Host-side work added this phase: `smart_resize`, the bicubic tap/weight
calculation, position-table interpolation indices, RoPE coordinates and the
multimodal layout. These are checkpoint-fixed canonicalizations computed per
request on the host, matching the reference's own host/device division; the
device never receives raw image containers.

## 4. Preprocessing — element-wise comparison

Run: `d3_lite_bench --images …/image_suite.json --reference …/cases.json`, one
sample per case; native PNG/JPEG decode → `preprocess` → compared against the
reference receipt's `pixel_values` (produced by `Qwen2VLImageProcessor`).

| Case | Source | Resized | Grid `[1,h,w]` | Elements (`N × 1536`) | max\|Δ\| vs reference |
| --- | --- | --- | --- | ---: | ---: |
| `mario_enemy_pos_f12` | 320 × 400 PNG | **320 × 384** (400 → 384) | `[1,24,20]` | 737 280 | **0.0** |
| `mario_enemy_neg_f00` | 320 × 400 PNG | 320 × 384 | `[1,24,20]` | 737 280 | **0.0** |
| `mario_obstacle_pos_f09` | 384 × 288 PNG | 384 × 288 (identity) | `[1,18,24]` | 663 552 | **0.0** |
| `mario_obstacle_neg_f00` | 384 × 288 PNG | identity | `[1,18,24]` | 663 552 | **0.0** |
| `mario_gap_pos_f45` | 480 × 320 PNG | identity | `[1,20,30]` | 921 600 | **0.0** |
| `mario_gap_neg_f00` | 480 × 320 PNG | identity | `[1,20,30]` | 921 600 | **0.0** |
| `generic_512_choice` | 512 × 512 JPEG | identity | `[1,32,32]` | 1 572 864 | 0.031 373 |
| `generic_800_score` | 800 × 800 JPEG | identity | `[1,50,50]` | 3 840 000 | 0.031 373 |

- The **6 PNG cases are bit-exact**, including `mario_enemy_*`, which is the
  only case that actually resamples (`smart_resize` rounds 400/32 = 12.5 down to
  12, Python banker's rounding, giving 384). This exercises the native bicubic
  replica on a real image against `tvF.resize`; a synthetic ramp oracle additionally
  pins the separable order and int16 rounding for both up- and down-sampling
  (`bicubic_resize_matches_the_torch_uint8_oracle`).
- The **JPEG residual** is decode-only: the port uses the Rust `image` crate
  (zune-jpeg) while PIL uses libjpeg; the largest decoded difference is
  **4 u8 levels** (`0.031 373 × 127.5 = 4`), and the normalized residual is
  `(Δu8) / 127.5`, the exact fused-normalization contract. Patchify ordering,
  temporal duplication and normalization are exact for those cases as well.
- The declared hard check is **element-wise exact for PNG**; it passes on all 6.

## 5. Parity — 8 image cases

Run: release `d3_lite_bench --set image --warmup 2 --repeat 2 --rounds 7`;
probabilities are from the repeated (byte-identical) run; compared with the
reference receipt `cases.json` (transformers 5.17.0 on **MPS**).

| Case | `L` | Image tokens `M` | Count | Decision (native / ref) | max\|Δp\| | KL(ref ‖ native) |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `mario_enemy_pos_f12` | 249 | 120 | 2 | **1 / 1** | 0.003 029 | 2.14e−5 |
| `mario_enemy_neg_f00` | 249 | 120 | 2 | **0 / 0** | 0.010 744 | 2.98e−4 |
| `mario_obstacle_pos_f09` | 238 | 108 | 2 | **1 / 1** | 0.011 346 | 2.66e−4 |
| `mario_obstacle_neg_f00` | 238 | 108 | 2 | **0 / 0** | **0.055 385** | 8.31e−3 |
| `mario_gap_pos_f45` | 273 | 150 | 2 | **0 / 0** | 0.010 457 | 4.58e−4 |
| `mario_gap_neg_f00` | 273 | 150 | 2 | **0 / 0** | 0.000 553 | 9.66e−7 |
| `generic_512_choice` | 369 | 256 | 3 | **2 / 2** | 0.000 872 | 1.51e−5 |
| `generic_800_score` | 723 | 625 | 3 | **0 / 0** | 0.004 610 | 6.41e−5 |

| Measure | Result | Gate |
| --- | ---: | --- |
| Decisions agreeing with the reference | **8 / 8** (0 flips) | R1: all |
| Worst max\|Δp\| | **0.055 385** | R2: ≤ 0.06 |
| Mean case max\|Δp\| | 0.012 124 | — |
| Two runs byte-identical / rows finite / ∑ = 1 | yes / yes / ≤ 1e−6 | R3 |
| End-to-end (native PNG) vs model-only (reference pixels in) | identical on PNG cases; worst 0.055 385 in both | — |

**Tolerance declaration (BF16 vs FP32).** The backbone is BF16 in both arms
and the readout is FP32. Two facts bound the expected drift:

1. The reference **moves by itself**: the same weights, same BF16, run on MPS
   vs CPU, give 8/8 decisions but a worst max\|Δp\| of **0.020 093** and
   merged-feature relative movement of **4.0–46.0 %** across the 8 cases.
2. The vision tower's late residual stream is ill-conditioned: the block-11
   hidden reaches O(10³), and the merger's LayerNorm turns direction error into
   output error. Native Metal's merged features sit **8.1–28.1 %** from the
   reference — inside the reference's own 4–46 % device-to-device envelope
   (`vision_tower_matches_pytorch_reference` asserts `< 50 %`).

The declared probability budget is **max\|Δp\| ≤ 0.06 = 3 × 0.020 093**, with
decision agreement on every case as the hard gate. The worst case
(`mario_obstacle_neg_f00`) is the same case that is worst in the reference's
own MPS↔CPU comparison, and it stays inside the budget. The dominant residual
is BF16 GEMM accumulation order between MLX and torch MPS, amplified by the
tower; the FP32 readout and softmax are identical ops in both arms.

## 6. Baseline — M1 Pro (single-arm, 7 rounds)

Run: release `d3_lite_bench --set image --warmup 2 --repeat 2 --rounds 7`,
one window, `vm_stat` snapshot before/after each series in each round. This is
a **baseline only**: there is no optimization candidate, no ABBA, no reverse
order. `text_plus_image` is the headline arm; `text` is the same prompt's text
stack with the image placeholders at their embedding vectors, reported
separately so the vision-tower increment is visible (the requester's
"文本+图分开报" rule).

| Series | Samples | Median (ms) | Mean | p95 | Max | Worst Swapouts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **text_plus_image** (public image `decide`) | 56 | **216.1** | 267.9 | 687.6 | 690.8 | **0** |
| text (text stack, no vision) | 56 | **142.6** | 162.4 | 339.7 | 340.3 | **0** |
| host input prep (decode + preprocess, before `decide`) | 8 | **2.6** | 2.9 | 6.9 | 6.9 | — |

All 112 timed samples had **Swapouts = 0** (max Swapins 239 pages), so the
window is clean and KEEP-eligible as a baseline. Latency is the full public
decision call (host pixel upload → vision tower → compiled text forward →
output readback), excluding model load, warmup and profile tracing; host input
preparation is reported alongside, not inside, that call.

Per-case medians (the split is where the vision cost lives):

| Case | `L` | `M` | text (ms) | text + image (ms) | vision ≈ Δ (ms) | host prep (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mario_enemy_pos_f12` | 249 | 120 | 123.6 | 183.1 | 59.5 | 2.4 |
| `mario_enemy_neg_f00` | 249 | 120 | 124.0 | 183.2 | 59.1 | 2.9 |
| `mario_obstacle_pos_f09` | 238 | 108 | 122.6 | 176.0 | 53.3 | 1.6 |
| `mario_obstacle_neg_f00` | 238 | 108 | 122.3 | 175.7 | 53.4 | 1.6 |
| `mario_gap_pos_f45` | 273 | 150 | 143.7 | 217.8 | 74.1 | 2.6 |
| `mario_gap_neg_f00` | 273 | 150 | 143.4 | 218.4 | 75.0 | 1.3 |
| `generic_512_choice` | 369 | 256 | 180.9 | 301.9 | 121.0 | 3.7 |
| `generic_800_score` | 723 | 625 | 339.7 | 687.6 | 348.0 | 6.9 |

Text-only baselines (P1a, **separate** window and suite, not blended with the
image numbers): 150 → 84.5 ms, 400 → 205.4 ms, 1000 → 453.2 ms,
2400 → 1116.4 ms. The text row above (142.6 ms pooled over lengths 238–723)
is the image suite's own text-stack series and should be read against those
figures, not added to them.

No operator timeline is claimed this phase; the signpost methodology applies
once a before/after is required in P3.

## 7. Verification

| Check | Result |
| --- | --- |
| `cargo check -p apxinf-model --features mlx --lib --examples` | pass |
| `cargo test -p apxinf-model --features mlx --lib metal_d3_lite` | **17 passed**, 4 ignored (device-gated) |
| `compiled_forward_matches_eager…` (Metal, text) | pass; replay adds 0 trace callbacks, bit-identical |
| **Text-path regression vs P1a**: `d3_lite_bench --set gate` (48) vs the EngineTailor oracle | 48/48, flips 0, max\|Δp\| **0.029 179 69**, mean KL **2.4487e−4**, byte-identical — **identical to the P1a receipt; refactor drift = 0.0** |
| `image_decision_matches_pytorch_reference…` (Metal, **public** registry path) | **8/8 decisions, worst max\|Δp\| 0.055 4 ≤ 0.06** |
| `vision_tower_matches_pytorch_reference…` (Metal, staged vs reference receipt) | pass; native 8–28 % vs reference self 4–46 %; `b0_cos`/`b0_sin` exact |
| `preprocessing_reproduces_the_reference_pixel_values` (Metal, real image) | pass; element-wise exact (PNG) |
| `d3_lite_bench --set image --warmup 2 --repeat 2 --rounds 7` | 8/8 parity, byte-identical, Swapouts = 0 on all samples |
| `bash scripts/check_model_family_boundaries.sh` | pass |
| `rustfmt --check` on changed files | pass |

## 8. Known limitations

- **One still image per request.** The placeholder scatter is one dynamic
  `slice_update` start, so a second image run is rejected with a named error
  (`multiple images are not wired`). The tower's packed-grid signature and the
  position layout are general; multi-image scatter is a follow-up.
- **Video is not wired.** `grid_t ≠ 1` and `video_token_id` in a prompt are
  rejected by name; the P1b suite is still images only.
- **JPEG decode is not bit-identical to PIL** (zune-jpeg vs libjpeg, ≤ 4 u8
  levels; see §4). PNG is exact.
- **The vision tower is numerically ill-conditioned.** Native merged features
  sit 8–28 % from the reference while the reference itself moves 4–46 % between
  MPS and CPU; the declared 0.06 probability budget (3× the reference's own
  0.020 093) absorbs this, and all 8 decisions agree. Closing this further
  requires matching torch's GEMM kernels, which is not a local change.
- **Two `inv_freq` evaluations, deliberately.** The text path keeps P1a's host
  `exp(−x·ln base)` (1 ulp from torch) to preserve the 48-case anchor bit for
  bit; the vision path uses f32 `powf`, which is bit-exact against the
  reference's `inv_freq`. Unifying them would move the text path off its
  anchor.
- **GDN `softplus` stays in BF16** (the P1a/EngineTailor contract) whereas the
  checkpoint's PyTorch GatedDeltaNet computes `softplus(a.float() + dt_bias)` in
  F32. This is an inherited text-path difference, not changed here; it is not
  separately attributed in the image budget.
- The two timing series run in a **fixed order** (text+image first) without
  ABBA. They are an attribution decomposition of one baseline, not an
  optimization comparison; the headline arm is `text_plus_image`.
- No CLI change: `apxinf decide` remains text-only; image requests enter
  through `DecisionInput::with_vision` and the bench. No UI change accompanies
  this PR.
- No operator timeline or cross-machine claim; numbers are from this host's
  clean Swapouts window.

## 9. Artifacts

Private reproducibility evidence (ignored `devlocal/d3-vision-metal/`): the
PyTorch reference receipts on MPS and CPU (`ref/primary/`, `ref/primary_cpu/`),
per-stage vision dumps (`ref/*/vision_stages.json` + `stages/*.f32`), the native
baseline and gate-regression summaries, and the harness scripts
(`scripts/dump_reference_cases.py`, `dump_reference_vision_stages.py`,
`probe_resize_replica.py`, `probe_preprocess_steps.py`). Paths and revisions
are recorded there; nothing private or generated is committed.
