# Mario demo optimization — handover (branch `showcase/clef-vision`)

Context for a follow-up agent working on the Clef Flash 9B × Mario decision demo
(ApxInf Metal). Measurement host: Apple M1 Pro 16 GB, macOS 27.0.1 (applegpu_g13s).

## What this branch adds

- `crates/apxinf-model/src/metal_clef/trace_json.rs` — env-gated Chrome Trace Event recorder
  (`APXINF_CLEF_CHROME_TRACE`), append-only durable flush, per-request/per-layer/per-chunk spans.
- `metal_clef/model.rs` — prefix KV cache (`APXINF_CLEF_PREFIX_CACHE`, default ON):
  attention layers cache KV, GatedDelta layers cache recurrence state + causal-conv history;
  chunk0 (system+chat prefix + `<|vision_start|>`, ~37–50 tokens) skips all 32 layers.
- `metal_clef/model.rs` + `encode.rs` + `preprocess.rs` — Tier-2 image-row prefix cache
  (`APXINF_CLEF_PREFIX_CACHE_TIER2`, default ON after the reference-tier KEEP gate):
  a same-image hit skips the vision tower, splice, and the image-row backbone prefix; the cache
  key is the prompt hash plus a SHA-256 of the prepared patch matrix and grids. Tier-2 is only
  attempted after the same prepared image has been seen at least once in the resident process, so
  the first occurrence of a unique image stays on the exact Tier-1 path.
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
- Tier-2 image-row prefix cache, reference tier, same-image multi-question suite (6 images × 4 question
  variants, 3 paired rounds, n=72 case-aligned gains): Tier-1-only 2174.6–2180.9 ms → Tier-1+2
  1127.9–1130.9 ms = **+45.327 %**, paired bootstrap **CI95 [42.799 %, 47.640 %]**, 100 % cases faster;
  parity vs Tier-1-only **48/48 decisions per round**, max|Δp| **0.020407**; A/A **24/24 per arm per
  round**; all 6 arms **Swapouts +0** → **KEEP**.
- Tier-2, reference tier, 42-case parity double gate (post-reboot): vs frozen MLX **42/42 decisions**,
  max|Δp| **0.032783** (worst `f0045-enemy/enemy/true`); vs Tier-1-only no-hit control **42/42**,
  max|Δp| **0.000671**. Clean-round no-hit latency 2211.96 → 2212.47 ms, paired median **−0.155 %**,
  CI [−0.567 %, +0.593 %] → no-hit path is neutral.
- Tier-2, auto tier: **not-measured** — ANE programs were warmed in `/tmp/clef-ane-cache-tier2-postboot`
  and a single-arm probe was clean (Swapouts +0), but every full paired round had candidate-arm
  Swapouts > 0 (r0 +31,900; r1 +5,304 / baseline +11,308). Per policy the reference result is not
  substituted for the demo conclusion.
- Demo end-to-end: reference 6599 ms/decision; auto 4467.5 ms/decision (0.677), identical confusion matrix.

## Closed directions (do not redo)

fusion combos (−0.372 %, CI<0), deeper ANE column split (−2.48 %, parity gate fail),
wavefront K=2 (ANE gap median 1.3 ms — overlap already achieved), full-GPU megakernel (<2 % bound),
`mx.compile` vision (0.7 ms host build), vis014 row split (single image below the 1024-row floor),
vision Metal batch (<2 %), matmul-level projection fusion (reference tier +0.109 %, CI95
[−0.100 %, +0.303 %], parity gate failed 0.049 > 0.035 — evidence and methodology in
[FUSION-MATMUL.md](FUSION-MATMUL.md); the implementation was dropped per "不留死代码").
Hardware `xctrace` timeline is **reopened and delivered**:
the earlier "injected-dyld hang" is a TCC-boundary `open()` hang when the traced
process reads `~/Documents`; staging the checkpoint and case image outside that
boundary records cleanly. See [TIMELINE-OPERATOR.md](TIMELINE-OPERATOR.md)
(host os_signpost operator labels aligned to the Metal command-buffer lane;
Core AI remains forbidden and the ANE hardware axis is still unmeasured).

### Autonomous round 2026-11-05 (three more closed directions)

Full records: [ANE-COLUMN-BALANCE.md](ANE-COLUMN-BALANCE.md) (shallower GPU/ANE
column split, 6656→7680: NOT-WORTH — auto tier Swapouts-contaminated so latency
not measurable, parity 0.0483 > 0.035, smoke +8 %), [PREP-IMAGE-CACHE.md](PREP-IMAGE-CACHE.md)
(repeat-image prepared cache: NOT-WORTH by hard bound, 12.7 ms preprocess ÷
1140 ms hit window ≈ 1.1 % < 2 %; bit-identical when it hits), and
[EAGER-EVAL.md](EAGER-EVAL.md) (per-layer `async_eval` of the lazy backbone:
NOT-WORTH — A/A bit-identical and parity unchanged at 0.0328, but median
+0.115 %, CI95 [−0.038 %, +0.264 %]; chrome shows each layer span absorbed
the GPU execution instead of overlapping the host build). After these, no
remaining timeline-backed candidate clears the <2 % / parity-risk /
KEEP-measurability filters — the candidate pool is dry on this host and
protocol; see EAGER-EVAL.md §3.3 for the screening table and the three
follow-ups worth doing instead.

## Measurement discipline

- Shared Metal locks: take `~/.cache/enginetailor/metal-measure.lock` first, then
  `/tmp/enginetailor-metal.lock` (the existing campaign runner order), and hold both for an entire
  paired round.
- Pair arms within one lock window, reverse arm order across rounds, record `vm_stat` swap deltas;
  a sample with non-zero Swapouts is contaminated. 42-case runs are for statistics — never enable
  the Chrome trace for them (trace only 1–2 representative requests; volume ≈ events × requests).
- ANE caches are bound to the binary that compiled them; a mismatched cache yields
  `Clef runtime placement is unknown`.

## Where the long-form record lives

The operating worktree keeps `devlocal/` (gitignored): `DECISIONS.md` (D0–D39), `QUEUE.md`,
`STATUS.md`, `HANDOVER.md`, `TIMELINE-METHOD.md`. Ask the operator for access when needed.
