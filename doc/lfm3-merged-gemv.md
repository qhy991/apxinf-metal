# LFM3 merged GEMV projections — pre-registration (written before any device measurement)

Status: registered 2026-11-19, before any build or measurement of this change.

## Why

The matched-precision comparison (#121) puts apxinf-native at **37.20 / 37.50 ms/token** vs
llama.cpp F16 at **36.42 / 36.67** (ctx 1024 / 4096, gen 128) — an apxinf deficit of
**0.78 / 0.83 ms per token** that is constant across context length, i.e. fixed per-token cost,
not bandwidth or attention.

Two independent sources of that fixed cost were identified in issue #122:

1. **Host-side gaps**: the decode loop synchronises per token (graph build, `stream.eval` wait,
   sample + sync, boundary callbacks) — being addressed by issue #122's pipelined decode;
2. **Kernel count / fusion**: ~700 launches per step, with every launch carrying a barrier. This
   change attacks one part of it: the four attention projections (q, k, v, per-head gate) read the
   **same** `n` and the MLP's `w1`/`w3` read the **same** `ffn`, yet each is a separate matmul.
   Merging them removes **3 launches per attention layer** and **1 per layer** for the MLP:
   11 attention layers × 3 + 24 × 1 = **57 fewer launches per token**.

An MLX-lane precedent exists: the EngineTailor lane runs `_merge_mode=both` (qkv + w13 merged) by
default; this brings the same algebraic idea to the native Metal path.

## Change

Behind the cargo feature `merged-gemv` (default off; a feature, not an env switch, because the
gateway spawns the worker with `env_clear()` — see the campaign's "empty arm" lesson):

- `weights.rs`: at load time build `qkvg = concat(q, k, v, gate)` along the output axis into one
  `[hidden, q + 2*kv + heads]` "linear", and `w13 = concat(w1, w3)` into `[hidden, 2*inter]`.
  Original projections are kept, so the feature-on binary carries the extra merged copies
  (≈ +0.3–0.5 GB) — a cost to report, not to hide.
- `model.rs`: both the chunked path (`attention_layer`) and the single-token path
  (`attention_step`, the one that matters for ms/token) take one matmul and slice the four
  outputs; the gated-MLP takes one matmul and slices `gate` / `up`.

## Industrial gate (fixed now, before measuring)

- **Correctness (hard)**: greedy output must be **token-identical** to the feature-off binary on
  ≥ 4 prompts × 64 tokens. A single differing token ⇒ **INVALID**, no matter the speed. (Note:
  concatenating weights can change matmul tiling and therefore accumulation order, so this is not
  assumed — it is tested.)
- **Performance**: M4, both metal locks, display asleep, same window, ABBA `A B B A A B`,
  ≥ 3 segments/arm, S2 decode ctx 1024 and 4096, gen 128, engine-reported timing
  (`--engine-timings`; client timing is unusable until issue #120 is fixed), paired medians,
  A/A ≤ 5 % per arm per cell.
- **Decision rule**:
  - both cells ≥ **+3 %** faster and tokens identical ⇒ **KEEP**, enable the feature by default;
  - 0–3 % faster ⇒ record, stay opt-in, do not flip the default;
  - slower in either cell ⇒ **NOT-WORTH**, revert, record verbatim.
- **Report**: per-cell ms/token for both arms, paired delta, A/A, effective GB/s (bytes identical
  in both arms — only launch count changes), and the launch-count reduction as arithmetic.

## Out of scope

Pipelined decode (#122), RoPE/fusion of the element-wise chains, compile/bucketing (#115 closed
those), any change to prefill or numerics.
