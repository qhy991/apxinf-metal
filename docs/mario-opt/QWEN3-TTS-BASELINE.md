# Qwen3-TTS end-to-end baseline (M1 Pro, 2026-11-10)

First end-to-end measurement of the native Qwen3-TTS path. Driver:
`crates/apxinf-model/examples/qwen3_tts_probe.rs` (loads the checkpoint through `AutoModel`, calls
`SpeechRuntime::synthesize`, reports audio length, wall time, RTF and completeness).

## Baseline

| text | audio | wall | **RTF** |
|---|---:|---:|---:|
| `你好，这是一个语音合成测试。` | 3.120 s | 2.67–2.73 s | **0.855–0.877** |
| `The quick brown fox jumps over the lazy dog.` | 2.800 s | 2.43–2.44 s | **0.868–0.872** |

Sampling is the checkpoint's own (temperature 0.9, top-k 50) with a fixed seed, so a run is
reproducible: the same text and seed produce the same duration every time.

## Predictor fusion on this chip: measured, no gain

The module gates its fused code-predictor on `applegpu_g16g` (M4) because that is where it was
measured; M1 Pro (`applegpu_g13s`) falls back to the reference SwiGLU. A new
`APXINF_TTS_PREDICTOR_FUSION=1|0` override allows measuring either side on any chip:

| arm | audio | wall | RTF |
|---|---:|---:|---:|
| reference SwiGLU | 3.120 s | 2.67–2.73 s | 0.855–0.877 |
| fused predictor | 3.120 s | 2.67–2.68 s | 0.855–0.859 |

The fused path is within noise on the M1 Pro and produces the same duration, so the existing
`g16g`-only gate is correct; the override stays as a measurement tool. (An early single run read
2.400 s for the fused arm; repeating it four times gave 3.120 s consistently, so that was a cold-start
artifact, not a semantic difference.)

## Where the remaining time is

RTF ≈ 0.86 means 2.7 s of compute for 3.12 s of speech. At the checkpoint's 12 Hz frame rate that is
~39 frames, i.e. **~69 ms per frame**. A 1.7B talker forward of 3.4 GB at the ~100 GB/s this machine
sustains is ~34 ms, so roughly half the per-frame cost is elsewhere — most likely the per-codebook
code-predictor steps, which are small sequential forwards and therefore the launch-bound shape the
fusion rules in `doc/optimization-transfer-rules.md` describe. That is the next thing to measure, not
a kernel to write yet.

## Per-frame breakdown (M1 Pro, `APXINF_TTS_STAGES=1`)

39 frames, 2.79 s wall, RTF 0.894:

| stage | median per frame | total | share |
|---|---:|---:|---:|
| talker graph build | 0.89 ms | 0.03 s | 1 % |
| talker forward + sampling sync | 23.35 ms | 1.01 s | 36 % |
| **code predictor (15 groups)** | **36.99 ms** | **1.47 s** | **53 %** |
| frame total | 61.24 ms | 2.51 s | — |

The talker's own number is misleading in isolation — MLX is lazy, so its 3.4 GB pass is materialized by
the `token.eval()` inside the sampling step, which is why that row carries 23 ms. **The code predictor
is the largest cost**, and it is 15 strictly sequential small forwards per frame.

Deeper instrumentation of the 15 groups shows the host is not the problem: each `predictor` call costs
**0.16 ms of graph construction** and each sample **0.01 ms**, yet the frame's single `stream.eval`
takes ~37 ms. So the predictor's cost is **device execution of 15 tiny dependent forwards**, i.e. the
launch-bound shape this repository's rules describe — not a host or Python overhead problem.

Next: measure one group's isolated GPU time (evaluate after each step instead of once per frame) to
separate per-step dispatch from per-step work, then decide whether fusing the predictor's
norm/add chain is worth a kernel. No kernel is proposed before that measurement.

## Inside the predictor (`APXINF_TTS_PREDICTOR_STAGES=1`)

Evaluating after each predictor layer, over 39 frames × 15 groups:

| step | median |
|---|---:|
| predictor layer (each of 5) | **0.648–0.675 ms** |
| predictor call (5 layers) | ~3.2 ms |
| × 15 groups per frame | ~49 ms |

That reproduces the frame-level 37–38 ms within the noise of the eager-evaluation mode (the first
group of each frame also carries the talker's pending pass, so the first group reads much higher).

**0.65 ms for one layer of a five-layer model on a 16-row cache is dispatch, not arithmetic.** A layer
here is attention over ≤16 positions plus a `SwigluMlpGemv`-fused MLP, so the kernels are tiny and
numerous. This is the launch-bound shape again — but the budget is what matters: fusing the two
add+norm edges per layer would remove ~2 launches × 5 layers × 15 groups = **150 launches per frame**,
i.e. ~1 ms of a ~61 ms frame (≈1.6 %). The 15-group chain is inherently sequential (group g consumes
group g-1's code), so the launches cannot be collapsed into one another.

**Conclusion: the TTS predictor is dispatch-bound but the individual fusions are not worth their
budget.** The remaining lever with real upside is the frame's structure — pipelining the talker's next
step against the current frame's predictor — not more kernels. That is where the next attempt should
go if this line is picked up again.
## Per-step dispatch versus per-step work (`APXINF_TTS_GROUP_EVAL=1`)

Synchronizing after every predictor group instead of once per frame:

| mode | RTF (run 1) | RTF (run 2) | per-step |
|---|---:|---:|---:|
| one eval per frame (baseline) | 0.953 | 1.466 | — |
| eval after every group | 1.606 | 1.073 | 2.295–4.497 ms |

**Run-to-run variance on this host is large**: the same configuration read RTF 0.83–1.47 across the
day, and the root causes (thermal state, background agents) are not under the benchmark's control. The
*ordering* held in both runs of every paired comparison, but any single absolute RTF should be treated
as ±30 % until the host is quiet. The per-frame attribution below came from a single quiet window and
should be re-taken with paired arms before it is used to size an optimization.

Two conclusions. First, batching the 15 groups into one evaluation is already worth ~1.8× over
synchronizing each — so the frame-level structure is right and must not be replaced by per-step
evaluation. Second, and more important: **2.5 ms of device work per predictor step is far too much for
a small model step**. The predictor is 15 sequential forwards of a few layers; a step of that size
should be tens of microseconds of arithmetic. Whatever costs 2.5 ms per step is structural (per-step
cache writes or an oversized per-step pass), not dispatch.

That is the next thing to open up: instrument the predictor's own layers/cache for one frame, and if
the per-step cost is dominated by cache materialization, the fix is the same class as the Qwen3 KV
finding — but here it would actually pay, because it happens 15× per frame instead of once per token.

## Correction: the predictor is bandwidth-bound, not dispatch-bound

The dispatch reading above does not survive its own arithmetic. 150 launches at 6.7 µs is ~1 ms, and
the predictor costs 37 ms, so launches cannot be the cost. Counting bytes settles it:

| quantity | value |
|---|---:|
| predictor parameters (5 layers, hidden 1024, MLP 3072) | 68.2 M |
| BF16 weights per group pass | 136.3 MB |
| passes per frame (15 groups, strictly sequential) | 15 |
| **bytes read per frame** | **2.04 GB** |
| implied bandwidth at the measured 37 ms | **≈55 GB/s** |
| at this machine's ~120 GB/s | 17.0 ms |

The predictor re-reads its whole weight set once per codebook group — 15 times per frame — at ~55 GB/s,
i.e. **bandwidth-bound at ≈46 % of what the machine can do**. The lever is bytes, not launches: a W8
predictor copy halves 2.04 GB to ~1.02 GB, a potential **~17 ms of the ~61 ms frame (−28 %)** if the
quantized kernels keep their bandwidth (the caveat `doc/optimization-transfer-rules.md` records — on
MiniCPM5-2B they did not).

The 15 passes cannot be collapsed: group *g* consumes group *g−1*'s sampled code, so there is no
batching escape, only fewer bytes per pass. This also supersedes the "fusions below budget" conclusion
of the previous section — the fusions were correctly costed at ~1.6 %, but they were the wrong lever.

## The W8 predictor gate passes (probe, `examples/qmm_probe.rs`)

Before wiring W8 into the predictor, the precondition from `doc/optimization-transfer-rules.md` was
tested: *does the quantized lane keep its bandwidth on these shapes?* MiniCPM5-2B failed it. The TTS
predictor's five projection shapes pass it comfortably (M1 Pro, `[out, in]`, M=1, 200 iterations after
50 warm-ups):

| shape | BF16 | W8 | speed-up |
|---|---:|---:|---:|
| 1024 × 1024 | 0.033 ms (64 GB/s) | 0.023 ms (46 GB/s) | **1.46×** |
| 1024 × 1024 | 0.025 ms (83 GB/s) | 0.016 ms (65 GB/s) | **1.56×** |
| 3072 × 1024 | 0.075 ms (84 GB/s) | 0.022 ms (141 GB/s) | **3.36×** |
| 3072 × 1024 | 0.033 ms (189 GB/s) | 0.027 ms (119 GB/s) | **1.26×** |
| 1024 × 3072 | 0.046 ms (137 GB/s) | 0.026 ms (120 GB/s) | **1.76×** |

W8 wins on every shape, 1.26–3.36× (≈1.9× mean). Applied to the measured 37 ms predictor pass, that
projects **≈17–19 ms saved of the ~61 ms frame (−28…−31 %)**, cutting the 2.04 GB per frame to ~1.02 GB.

Two caveats worth keeping: the repeated 1024 × 1024 and 3072 × 1024 rows disagree with themselves by up
to 2× (host noise, see the discipline section — pair the arms when it matters), and the win is a
per-shape property, so the predictor's actual end-to-end gain must still be measured rather than
assumed from this table.

The remaining work is mechanical: re-add the `apxinf_mlx::Quantized` decode wrapper (it existed on the
MiniCPM5 branch and was reverted with that negative result), build the five predictor layers' W8 copies
at load, and use them for the M=1 lane.
