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
