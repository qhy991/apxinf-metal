# Mario demo optimization — handover (branch `showcase/clef-vision`)

Context for a follow-up agent working on the Clef Flash 9B × Mario decision demo
(ApxInf Metal). Measurement host: Apple M1 Pro 16 GB, macOS 27.0.1 (applegpu_g13s).

## What this branch adds

- `crates/apxinf-model/src/metal_clef/trace_json.rs` — env-gated Chrome Trace Event recorder
  (`APXINF_CLEF_CHROME_TRACE`), append-only durable flush, per-request/per-layer/per-chunk spans.
- `metal_clef/model.rs` — prefix KV cache (`APXINF_CLEF_PREFIX_CACHE`, default ON):
  attention layers cache KV, GatedDelta layers cache recurrence state + causal-conv history;
  chunk0 (system+chat prefix + `<|vision_start|>`, ~37–50 tokens) skips all 32 layers.
- `metal_clef/vision.rs`, `vision_ane.rs`, `preprocess.rs` — vision tower path + host preprocessing.
- `metal_clef/ane.rs` — GPU/ANE column split with `ane_wait_pack` / `ane_predict` instrumentation
  and placement retry (`APXINF_CLEF_PLACEMENT_RETRIES`).
- `crates/apxinf-mlx/src/fusions.rs` + `native/{rms_norm_scale,gdn_conv_split,mrope_half_split}.metal`,
  `src/mrope_rope.rs`, `src/causal_conv.rs` — fusion candidates (mostly measured NO-GAIN, flags default off).
- `src/bin/apxinf-decide-server.rs` — long-running decision server used by the Mario demo.

## Measured results (42 single-image cases, 2-block interleaved 3-arm)

| arm | median | vs MLX | decisions |
|---|---|---|---|
| mlx-clef baseline | 2226.43 ms | 1.000 | — |
| apxinf reference (GPU) | 2134.44 ms | 0.9592 | 42/42 |
| apxinf auto (GPU+ANE) | **1428.88 ms** | **0.6424 (1.56×)** | 42/42 |

- Prefix cache, reference tier: 2435.95 → 2223.60 ms = **+8.72 %** (parity 42/42, max|Δp| 0.0153,
  candidate Swapouts +0) → **KEEP, default ON**.
- Prefix cache, auto tier: +1.68–2.15 % (CI lower bound > 0 but below the 2 % KEEP gate) → **NOT-WORTH**.
  An earlier +3.273 % KEEP was withdrawn after a runner statistics bug (`pooled.update(per)` overwrote
  per case instead of accumulating; the CI was the last round only).
- Demo end-to-end: reference 6599 ms/decision; auto 4467.5 ms/decision (0.677), identical confusion matrix.

## Closed directions (do not redo)

fusion combos (−0.372 %, CI<0), deeper ANE column split (−2.48 %, parity gate fail),
wavefront K=2 (ANE gap median 1.3 ms — overlap already achieved), full-GPU megakernel (<2 % bound),
`mx.compile` vision (0.7 ms host build), vis014 row split (single image below the 1024-row floor),
vision Metal batch (<2 %), hardware `xctrace` timeline (unusable on this host: injected-dyld hang).

## Measurement discipline

- Shared Metal locks: `~/.cache/enginetailor/metal-measure.lock` then `/tmp/enginetailor-metal.lock`
  (outer `/tmp` first when taking both).
- Pair arms within one lock window, reverse arm order across rounds, record `vm_stat` swap deltas;
  a sample with non-zero Swapouts is contaminated. 42-case runs are for statistics — never enable
  the Chrome trace for them (trace only 1–2 representative requests; volume ≈ events × requests).
- ANE caches are bound to the binary that compiled them; a mismatched cache yields
  `Clef runtime placement is unknown`.

## Where the long-form record lives

The operating worktree keeps `devlocal/` (gitignored): `DECISIONS.md` (D0–D39), `QUEUE.md`,
`STATUS.md`, `HANDOVER.md`, `TIMELINE-METHOD.md`. Ask the operator for access when needed.
