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
