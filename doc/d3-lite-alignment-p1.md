# D3-lite P1 alignment report (native apxinf-metal ↔ references)

Status: **BLOCKED before execution** — the native `d3` module cannot build or run on
this host, and it cannot load the d3-lite checkpoint. No parity or latency numbers
are claimed.

## 1. Scope and inputs

Source request: issue #11 comment `d089347800f2a1e3785d65465f621f13` (P1: align
native d3 with the references and establish an M1 Pro baseline; no new
optimization this phase) plus methodology supplement `23510733a5a62c9143312dcef7d51ccd`
(systematically reuse the Clef line's methodology and conclusions).

| Input | Path / revision | Role |
| --- | --- | --- |
| Native implementation | `crates/apxinf-model/src/d3/` (2648 lines; checkout `7bab554e`) | Target under alignment |
| Checkpoint reference | `~/Documents/Infinity/Agent4Kernel/models/d3-lite-9c10703` (`vllm-sr/d3-lite`, `model_sha256=fda5cfc7…`) | FP32/original parity oracle (`d3_engine.py`, `modeling_d3.py`, `pipeline_d3.py`) |
| EngineTailor reference | `~/src/EngineTailor/enginetailor/models/d3lite` (`graft@2e54f7f` per its README) | MLX text path, `gate.py`/`bench.py`, `testdata/workload.json`, `PROMOTED` |
| MLX SDK | `~/apxinf-runtime/mlx-sdk` (0.32.3) | Metal execution for the MLX side |
| Clef methodology (extracted) | `devlocal/d3-lite-align/reports/clef-*.md`, PR #5 body | Timeline method, measurement discipline, closed directions |

Workload facts (`testdata/workload.json`, schema `graft-d3-workload/1`): 48 gate
cases (12 per length bucket: 150 / 400 / 1000 / 2400 tokens), 8 flagged timing
cases (`b0150_engines_kind`, `b0150_quickstart_route`, `b0400_license_kind`,
`b0400_kitreadme_topic`, `b1000_suite_kind`, `b1000_readme_topic`,
`b2400_suite_kind`, `b2400_readme_topic`), decision tolerance 0.15, 255 codes.

## 2. Architecture mapping

| Concern | Native `d3` (apxinf-metal) | Checkpoint Python | EngineTailor `d3lite` |
| --- | --- | --- | --- |
| Device backend | `apxinf-cuda-new` L3/fixed ops; CUDA Graph `PreparedPhase` profiles | PyTorch / transformers | MLX (`mlx.core`) |
| Config contract | `config.rs`: **d3-mini** geometry and manifest hardcoded | `config.json` of the checkpoint | `config.json` of the checkpoint |
| Text forward | `model.rs` (907 lines): embedding → 3 GDN + 1 full-attn schedule → readout | `modeling_d3.py` | `engine.py` `_forward` / `_layer` |
| GDN recurrence | CUDA `ops` GDN prefill + BF16 state | reference implementation | `kernels.py` `gated_delta` (`stock` / `apxinf_four_values` / `apxinf_eight_lanes`) |
| Causal conv | inside CUDA GDN operand chain | reference | `kernels.py` `causal_conv` (`stock` / `apxinf`) |
| RoPE | CUDA `ops` rope | reference | `kernels.py` `half_split_rope` (`stock` / `apxinf`) |
| Full attention | CUDA `ops` attention policy | reference | MLX SDPA-style ops |
| Vision tower | `vision.rs` (641 lines), CUDA only | `modeling_d3.py` visual + `pipeline_d3.py` image preprocessing | **not loaded** (text path only) |
| Runner / lifecycle | `runner.rs` (467 lines): fixed-length text profiles, CUDA Graph replay, vision profiles | `d3_engine.py` / `d3_server.py` | `engine.Arm(model, options)` |
| Load | `load.rs` (72 lines) + `weights/` | `AutoModel` custom map | `engine.load` |
| Gate / bench harness | **none in this repo** | none | `collect.py` (receipts), `gate.py` (R1/R2/R3), `bench.py` (ABBA/BAAB), `rank_stats` |

Reusable native pieces if a Metal execution path is ever authorized:
`apxinf-mlx/src/gated_delta.rs` (`apxinf_gated_delta_four_values`),
`apxinf-mlx/src/half_split_rope.rs` (`apxinf_rope_half_split`), MLX built-in
`Array::conv1d`, and the Clef signpost shim pattern. Missing from `apxinf-mlx`
today: `gated_delta_eight_lanes.metal`, `causal_depthwise_conv.metal`, and the
`gdn_fused_prep` kernel (all present in EngineTailor's `native/`).

## 3. Finding A — the native module is CUDA-only; it cannot run on M1 Pro

Evidence (all on this Apple M1 Pro host, no NVIDIA device):

1. `crates/apxinf-model/src/d3/mod.rs:11-20` gates `backend`, `load`, `model`,
   `runner`, `vision` behind `#[cfg(feature = "cuda")]`; only `config` and
   `weights` are unconditional.
2. `crates/apxinf-model/src/auto.rs:425-430`: for `model_name = "d3"`, without
   `cuda` the loader returns `d3-mini requires CUDA support`.
3. `crates/apxinf-model/src/d3/load.rs:20-22`: a non-CUDA `Device` returns
   `d3-mini requires CUDA`.
4. `crates/apxinf-model/src/builtin.rs` registers `metal_clef` but no
   `metal_d3` factory; no Metal-owned factory can admit the d3 family.
5. `cargo check -p apxinf-model --features cuda --example d3_smoke` passes
   (CUDA crate compiles), but the produced example **fails to link on arm64**:
   `ld: symbol(s) not found for architecture arm64` for
   `_cudaStreamCreateWithFlags`, `_cudaStreamBeginCapture`,
   `_cudaStreamSynchronize`, etc. — no CUDA runtime library exists on macOS.
6. `system_profiler SPDisplaysDataType`: only the built-in Apple M1 Pro GPU
   (Metal 4); no NVIDIA device, no `nvidia-smi`.
7. Every remote branch was searched for `cfg(feature = "mlx")` inside
   `crates/apxinf-model/src/d3/` — **no match on any ref**. The d3 directory is
   byte-identical in backend terms across `main` and `showcase/clef-vision`.

Conclusion: requirement 1 ("构建原生 d3，确认在 M1 Pro 能跑通") is not achievable
with the current code. Making it achievable means implementing a new
MLX/Metal device backend for d3 (a model port), not aligning an existing one.

## 4. Finding B — the native module targets d3-mini, not d3-lite

Evidence:

1. `config.rs:189-243` hardcodes the d3-mini text geometry
   (`hidden_size == 2560`, `intermediate_size == 9216`,
   `num_hidden_layers == 32`, `num_attention_heads == 16`,
   `num_key_value_heads == 4`, `linear_num_value_heads == 32`, …).
2. `config.rs:246-261` hardcodes the d3-mini vision geometry
   (`depth == 24`, `hidden_size == 1024`, `out_hidden_size == 2560`).
3. `config.rs:305-323` hardcodes the d3-mini manifest identity and parameter
   counts (`model_name == "d3-mini"`, `repo_id == "vllm-sr/d3-mini"`,
   `loaded == 4_539_918_336`).
4. Running the package contract test against the actual checkpoint:
   `APXINF_D3_CHECKPOINT=…/d3-lite-9c10703 cargo test -p apxinf-model
   official_package_files_pass_contract -- --ignored` →
   `Err("d3-mini config: unsupported d3-mini text geometry")`.

Geometry diff (checkpoint `config.json` vs native validation):

| Field | Native expects (d3-mini) | d3-lite checkpoint |
| --- | --- | --- |
| text `hidden_size` | 2560 | 1024 |
| text `intermediate_size` | 9216 | 3584 |
| text `num_hidden_layers` | 32 (24 GDN + 8 full) | 24 (18 GDN + 6 full) |
| text `num_attention_heads` | 16 | 8 |
| text `num_key_value_heads` | 4 | 2 |
| text `linear_num_value_heads` | 32 | 16 |
| vision `depth` / `hidden_size` / `out_hidden_size` | 24 / 1024 / 2560 | 12 / 768 / 1024 |
| manifest `model_name` / `repo_id` | d3-mini / vllm-sr/d3-mini | d3-lite / vllm-sr/d3-lite |
| manifest `loaded` params | 4,539,918,336 | 853,247,040 |

Compatible aspects: `format_id=d3-code-readout-v1`, `prompt=d3`,
`attention_mode=noncausal_full_attention`, `pooling=last`, `readout_dtype=float32`,
255 codes / 255 token IDs, `full_attention_interval=4`,
`linear_conv_kernel_dim=4`, `head_dim=256`, `partial_rotary_factor=0.25`,
`mrope_section=[11,11,10]`, `rope_theta=1e7`, vocabulary 248320.

Conclusion: even on a CUDA host, the existing native module would reject this
checkpoint. Requirement 2 ("文本路径对齐 … vs checkpoint 参考") therefore needs a
d3-lite configuration/geometry contract change in addition to a runnable backend.

## 5. PROMOTED options — evaluation against the native module

| EngineTailor `PROMOTED` key | Reference implementation | Native `d3` today | Reuse / gap |
| --- | --- | --- | --- |
| `gdn_core=apxinf_eight_lanes` | `kernels.py` + `native/gated_delta_eight_lanes.metal` | CUDA GDN prefill op; **no option knob** | `apxinf-mlx` has only `gated_delta_four_values`; eight-lanes kernel must be added for Metal |
| `conv1d=apxinf` | `native/causal_depthwise_conv.metal` | CUDA GDN operand chain; **no knob** | `apxinf-mlx` has MLX built-in `conv1d`, not the ApxInf causal kernel |
| `rope=apxinf` | `native/rope_half_split.metal` | CUDA rope op; **no knob** | **Directly reusable**: `apxinf-mlx/src/half_split_rope.rs` |
| `gemm_dtype=f16` | F16 weight copies + F16 GEMMs | `load.rs:23-33` enforces the official BF16 variant only | No F16 variant path; would be a new variant (optimization phase) |
| `compile=model` | `mx.compile` over the whole forward | CUDA Graph `PreparedPhase::prepare_and_capture` + replay (`runner.rs:116-132`) | **Mechanism analogue exists** on the CUDA side (different backend; not comparable to `mx.compile`) |
| `gdn_block=fused_prep` | `kernels.py` `fused_prep` (Graft) | CUDA GDN op; **no knob** | Missing kernel; EngineTailor source is the pinned origin |

Verdict for this phase: porting the `PROMOTED` points presupposes a running
native implementation and, on Metal, three new kernels plus a model forward.
That is optimization/backend-port work (the issue's "from-scratch port" that P1
explicitly is not) and is therefore recorded as an **evaluation only**, not
implemented here.

## 6. Parity and baseline status

| Deliverable | Status | Reason |
| --- | --- | --- |
| 48-case decision parity vs checkpoint reference | **not measurable** | Native d3 cannot link or load d3-lite on this host (Findings A + B) |
| max\|Δp\| report (incl. bf16/f16 drift explanation) | **not measurable** | Same |
| Receipts vs EngineTailor `gate.py` | **not measurable for native**; EngineTailor receipts (`collect.py`) could be generated on MLX but would compare reference↔reference, not native↔reference | Same |
| `d3::vision` on M1 Pro + image-case parity vs `pipeline_d3.py` | **not measurable** | `vision.rs` is `#[cfg(feature = "cuda")]`; no Metal vision path |
| Text-path and multimodal latency baseline (incl. 8 timing cases) | **not measurable for native** | No runnable native path |
| Reusable bench entry | **not delivered** | Needs a runnable family; existing `d3_bench` example is CUDA-only |

Planned measurement protocol once a runnable path exists (from Clef standing
rules): paired arms inside one Metal lock window, ABBA/BAAB with reversed arm
order across rounds, per-arm `vm_stat` Swapouts (non-zero ⇒ contaminated and
excluded), gate cases statistics-only (never traced), Core AI instrument not
used, `xctrace` limited to `Metal System Trace + Points of Interest`,
`os_signpost` env-gated operator timeline per `TIMELINE-OPERATOR.md`.

## 7. Clef line carry-over (methodology supplement)

Extracted to `devlocal/d3-lite-align/reports/`: `TIMELINE-OPERATOR.md`,
`HANDOVER.md`, `FUSION-MATMUL.md`, `EAGER-EVAL.md`, `ANE-COLUMN-BALANCE.md`,
`PREP-IMAGE-CACHE.md`, plus PR #5's record. Transferable conclusions to apply
when D3 becomes runnable (not re-verified here): engines already overlap ~89%
→ do not start with deeper splits; deleting work beats speeding it up; cold
start dominates small token counts; quantization/fusion can break FP32
accumulation order and must pass the parity gate; environment limits unchanged.
D3-specific hypotheses (vision-prefix reuse for same-image multi-question,
ANE column split, timeline-first) stay in the P3 candidate pool.

## 8. Blocker and smallest action

Blocker: the source request assumes a native d3 that builds and runs d3-lite on
M1 Pro. The repository contains a **CUDA-only, d3-mini-only** implementation.
Closing the gap requires either a new MLX/Metal d3-lite port (a full model
backend port — outside this alignment-only phase) or a CUDA host carrying a
d3-lite-compatible native build (not available on this machine).

Smallest action needed from the requester: choose the execution basis for P1 —
(new MLX/Metal d3-lite port ticket), (CUDA host + d3-lite native build), or
(narrow P1 to reference↔checkpoint alignment on MLX and defer native parity).
