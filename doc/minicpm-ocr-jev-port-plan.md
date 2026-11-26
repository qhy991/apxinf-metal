# Port plan: MiniCPM-OCR-Jev (EngineTailor's accelerated OCR lane → ApxInf)

Status: **proposal, no code yet.** This plan ports the two accelerations that
EngineTailor admitted in its Jev OCR lane. It is the only implementation in the
EngineTailor → ApxInf support gap that currently has an admitted speed effect.

Baselines compared:

- EngineTailor `origin/main` `64092d2f` (2026-11-17): lane
  `enginetailor/models/jev/`, target `targets/minicpm-ocr-jev-m1pro-bf16-v1/`.
- ApxInf fork `origin/main` `a4836413`: no OCR/VLM-Jev model module. The closest
  neighbouring code is `crates/apxinf-model/src/metal_minicpm5/` (MiniCPM5-2B,
  MLX on Apple silicon) and the `Decision` result family
  (`crates/apxinf-model/src/decision.rs`).

## 1. Why this model, and not the other seven gaps

The support screen (EngineTailor lanes vs `crates/apxinf-model/src`) finds eight
implementations that exist in EngineTailor but not in ApxInf. Seven have **no
admitted speed effect** in EngineTailor's own receipts:

| gap lane | acceleration evidence at `64092d2f` | in this plan? |
|---|---|---|
| **MiniCPM-OCR-Jev** | **two WIN receipts, §2** | **yes** |
| Qwen3-ASR-1.7B | no admitted gain: `runs/asr-m2-10h-final-qualification` ("no optimization delta or qualified gain"), `runs/asr-m2-24h-encoder-source-pair` (B/A 1.0119, STOP), `campaigns/qwen3-asr-m2-astra-msl-10h-20261008` ("zero WINs, no admitted speedup") | no |
| Qwen3.5 native MoE | no measurement; `models/qwen35/bench_moe.py` is baseline tooling only | no |
| Qwen3.5-4B + DLite | no admitted e2e gain: `runs/dlite-prefill-p0-20261112` measures 62.06% prefill share (headroom, not a gain), the ANE-share C arm never ran, `runs/dlite-decode-accept-20261113` is PARK at 0.00% e2e | no |
| FLUX.1 VAE decode | `models/flux/bench_vae.py` measures a candidate; no comparison/admission receipt | no |
| Oasis rollout | gate/bench boundary only, no gain receipt | no |
| OMDepth / BridgeDepth | external adapter gate only ("no local model runtime") | no |
| Kimi-K3 SiTU | operator ABI gate only ("not a Kimi model engine") | no |

Rule used: write a port plan only for a lane that already has an **admitted**
acceleration receipt. A capability-only gap, or a lane with a headroom study but
no admitted gain, should first register its acceptance criterion and then be
planned after its data exists — the "no direction change without a registered
rule" practice.

## 2. The two admitted accelerations (EngineTailor receipts)

Model: `openbmb/MiniCPM-OCR-Jev` revision
`56960237f0569eed11983fccfdc411c97d5db906`; scoring corpus
`openbmb/JevOCRBench` `a26c4b32dcd3a95490842ca637b80edd2b71d975`. Request =
original crop + rendered candidate text + OCR text; output = two-class
consistency logits/probabilities (`probabilities[1]` is consistency). Serial B1,
≤3072 tokens, no truncation, FP32 norms/A_log/logits/head. Host: Apple M1 Pro
16 GiB. 8 timing rows span text/formula/table inputs.

| stage | suite total (8 rows) | vs BF16 stock | gate |
|---|---:|---:|---|
| stock BF16 | 23728.8 ms | 1.000 | baseline |
| + whole-forward FP16 (`runs/jev-fp16-compute`) | 20252.0 ms | **0.8535 (−14.65%)** | PASS: mean KL 0.084× stock, worst per-case error 0.42× stock |
| + GPU/ANE diagonal wavefront (`runs/jev-ane-wavefront`) | 18072.1 ms | **0.7616 (−23.84% vs stock, −10.83% vs FP16)** | PASS: A/A twin −0.05% (p 0.79), worst error 0.42× stock |

- **FP16 receipt** (`verdict: WIN`): 12/12 valid blocks, all 8 rows +15.3…17.4%,
  no row slower; twin usable at +0.022% spread, p 2.4e-4. Default-on with
  `JEV_FP16_COMPUTE=0` rollback. The decisive detail: the tower's memoized
  pos/rpe are also produced in FP16 — an earlier FP16-weights-only attempt lost
  exactly to per-op promotion.
- **Wavefront receipt** (`verdict: WIN`): 13/16 valid blocks, suite +12.43%,
  p 1.2e-4, long rows +10.0…19.4%, short rows within ±1.2%, no cell regression.
  Default-on under `minicpm-ocr-env-coreml`, rollback `JEV_ANE_WAVEFRONT=0`.
  The owner applied swap tolerance #136 before admission; the gain is
  conservative.

## 3. Mechanisms to port

### 3.1 Whole-forward FP16 compute policy (−14.65%)

- Weights, embeddings and activations FP16; **norms, `A_log`, the final logits
  and the two-class head stay FP32**.
- The vision tower's memoized positional/relative-position tables are produced
  in FP16 as well; without that, every downstream op promotes back to FP32 and
  the gain disappears.
- Output arithmetic unchanged: logits → FP32 softmax → `probabilities`.

### 3.2 GPU/ANE diagonal wavefront for the text stack (−10.83% on top of FP16)

- Applies only to rows with `1600 < T ≤ 2816`; shorter rows take the stock path
  **bit-exactly**, longer rows are out of the supported envelope (≤3072 tokens).
- The 24-layer text stack is processed in K=2 chunks padded to `PAD=1408` in the
  `[1, 2048, 1, 1408]` conv layout (1×1 convs are position-independent, so the
  first `tlen` output columns are the exact result).
- Per layer, the fused whole-MLP Core ML program runs on the ANE; the GPU builds
  `mixer(L, k)` and pre-builds + `mx.async_eval`s the **next** unit's mixer, so
  the GPU no longer serializes behind a host-blocking `predict`.
- Traps already paid for in EngineTailor (do not re-learn them): (1) carry real
  per-layer `ArraysCache`(GDN) / `KVCache`(attention) objects — `cache=None`
  diverges chunk 1's GDN (max |dp| 0.38); (2) `position_ids` stays 3-D
  `(3, B, Lk)` when sliced; (3) the model returns the full-T concatenated output.
- Rows `T ≤ 1600` are returned by the stock path bit-exactly; guard row 673 is
  bit-exact in the receipt.

## 4. Where it lands in ApxInf

New module, following `doc/adding-a-new-model.md` and the `metal_minicpm5`
layout:

```
crates/apxinf-model/src/metal_minicpm_ocr_jev/
  mod.rs          registration, capability surface, exports
  config.rs       checkpoint config (Qwen3.5-VL-style multimodal stack)
  weights.rs      safetensors mapping; trained `score.weight` shape [2, 2048]
  preprocess.rs   RGB/aspect padding; caller supplies the rendered image
  model.rs        vision tower + 24-layer text stack (GDN + causal attention) + last-nonpadding pooling + two-class head
  ane.rs          per-layer MLP Core ML programs + the two-lane diagonal schedule
  README.md       supported workload, precision policy, rollback flags
```

- **Registration**: `registry::register("metal_jev", …)` plus a
  `metal_minicpm_ocr_jev` alias; detection through the existing `AutoModel`
  path (`builtin.rs`), no new registry mechanism.
- **Result contract**: one `Decision`-family output per request — question
  `consistency`, options `true`/`false`, `probabilities[1]` = consistency,
  full logits retained where the CPU oracle needs them. P0 settles whether
  `Decision` carries logits or a small model-local payload is added; the goal is
  model-neutral, not a new trait.
- **Reuse**: `apxinf-mlx` (MLX), `apxinf-coreml` (Core ML programs),
  `apxinf-loader` (safetensors), `apxinf-tokenizer`, and the ANE admission
  whitelist pattern from `metal_clef/ane_admission.rs`
  (FLOAT16/INT8/INT32/BOOL/STRING). Async two-lane submission depends on what
  `apxinf-mlx` exposes — see risk R3.

## 5. Phases (each a reviewable PR)

| phase | deliverable | exit criterion | size |
|---|---|---|---|
| **P0** contract & baseline | request schema (`orig_image`, `render_image`, `ocr_text`), snapshot identity, token cap 3072 / no truncation, tokenizer+prompt parity, result mapping; BF16 baseline numbers (suite 23728.8 ms, 8 rows) imported from the EngineTailor receipt | tokenizer output and prompt bytes match EngineTailor for all 72 corpus rows; result mapping reproduced on CPU oracle | 1–2 d |
| **P1** BF16 parity module | `config`/`weights`/`preprocess`/`model` + `Decision` output; independent FP32 CPU oracle from EngineTailor (`models/jev/reference.py`) as ground truth | parity gate: mean KL ≤1.5× stock, worst per-case probability error ≤2× stock; 8 timing rows within ±3% of the EngineTailor BF16 baseline; no row slower | 3–5 d |
| **P2** FP16 whole-forward | precision policy from §3.1 + rollback flag | suite ≤0.86× BF16 (target 0.854×), all 8 rows ≥+15%; gate re-run; A/A twin within ±3% | 2–3 d |
| **P3** ANE wavefront | Core ML MLP programs for 24 layers + admission check + K=2/PAD=1408 chunking + two-lane schedule; fallback to stock path outside `1600<T≤2816` | suite ≤0.90× FP16 (target 0.892×); long rows +10…19%; short rows bit-exact; admission passes the FLOAT16-or-INT8 whitelist; rollback flag | 3–5 d |
| **P4** qualification | 3 A/A windows, parity, 8-row bench, receipts, README, default-on/rollback table | all acceptance criteria below hold on a clean boot (swap <0.5 GB, uptime <3 d) | 2 d |

## 6. Acceptance criteria (all must hold)

1. Request/response contract matches EngineTailor: exactly
   `orig_image`/`render_image`/`ocr_text`; two-class output with
   `probabilities[1]` = consistency; ≤3072 tokens, no truncation.
2. Parity gate passes on the 72-row corpus: mean KL ≤1.5× stock, worst per-case
   probability error ≤2× stock (the BF16 receipt's own gate).
3. Timing: suite ≥+12% over BF16 with no timing row slower; A/A twin within
   ±3%; both measured on a clean-boot window.
4. Short-row guard (`T ≤ 1600`) is bit-exact against the stock path; rows above
   the supported envelope are rejected by name, not silently truncated.
5. Rollback: each acceleration is default-on with an env kill switch
   (`APXINF_JEV_FP16_COMPUTE=0`, `APXINF_JEV_ANE_WAVEFRONT=0`) and produces
   bit-identical output when disabled.
6. No new shared abstraction is introduced for a single model (repo YAGNI rule).

## 7. Risks and mitigations

| # | risk | mitigation |
|---|---|---|
| R1 | Re-implementing MLX-VLM 0.7.11 scoring semantics in Rust drifts (positions, GDN state, last-nonpadding pooling, two-class head) | keep EngineTailor's independent FP32 CPU oracle as ground truth; port prompt/tokenizer by parity test, not by reading; P1 gate blocks on it |
| R2 | FP16 promotion trap: tower pos/rpe in FP32 re-promotes every op and erases the gain | encode the policy in config with a unit test asserting the dtype of each stage; reproduce the earlier FP16-weights-only failure as a negative test |
| R3 | `apxinf-mlx` may not expose async eval / two-lane submission needed by the wavefront | P3 is additive: if async submission is unavailable, land P2 (FP16) and record the wavefront as blocked with the exact missing primitive |
| R4 | ANE admission: the fused whole-MLP Core ML program may not pass the tensor-type whitelist or may exceed the per-program compile budget (24 programs) | run `ane_admission` on the converted program in P3 before scheduling work; fall back to the split MLP path from `metal_clef/ane.rs` if admission fails |
| R5 | Core ML compile time/size for 24 per-layer programs (cache growth) | measure per-arm compile + cache size; reuse `APXINF_COREML_CACHE` per-arm convention; document the disk cost |
| R6 | Absolute levels are not comparable across windows (AGENTS §5.8) | compare only same-window paired ratios; record machine state (`uptime`, `swap`, `load1`, accelerator idle) per window |

## 8. Non-goals

- No Qwen3.5-MoE, ASR, DLite, FLUX-VAE, Oasis or OMDepth port here; they have no
  admitted acceleration receipt and are listed in §1 for the record.
- No paper claim: this plan only moves an already-measured optimization between
  engines. Numbers above are EngineTailor's, on M1 Pro; ApxInf must re-measure.
- No change to the Jev caller contract (caller still supplies the rendered
  image; the plan does not approximate the official rendering service).

## 9. References

EngineTailor (branch `main` at `64092d2f`):

- lane: `enginetailor/models/jev/{README.md,engine.py,ane.py,bench.py,gate.py,reference.py}`
- target: `targets/minicpm-ocr-jev-m1pro-bf16-v1/`
- FP16 WIN: `runs/jev-fp16-compute/RESULT.md`
- wavefront WIN: `runs/jev-ane-wavefront/RESULT.md`
- overlap probe and its correction: `runs/jev-ane-overlap-probe/{RESULT.md,CORRECTION.md}`
- campaign record: `campaigns/jev-m1pro-20261101f/RESULT.md`

ApxInf (fork `main` at `a4836413`):

- registration: `doc/adding-a-new-model.md`
- MLX family layout: `crates/apxinf-model/src/metal_minicpm5/`
- ANE admission pattern: `crates/apxinf-model/src/metal_clef/{ane.rs,ane_admission.rs}`
- Core ML crate: `crates/apxinf-coreml/`; prior ANE trade study: `doc/qwen-image-ane-offload/`
- result family: `crates/apxinf-model/src/decision.rs`
