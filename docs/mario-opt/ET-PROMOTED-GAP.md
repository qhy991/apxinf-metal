# EngineTailor promoted config vs apxinf native — gap matrix

Source of truth for the EngineTailor (ET) side: `enginetailor/models/d3lite/engine.py`
`PROMOTED` (the only model in ET that declares a promoted point). apxinf side:
`crates/apxinf-model/src/metal_d3_lite/` on `main`.

## d3-lite: 4 of 6 promoted items are already ported

| ET option | ET promoted value | apxinf native | Status |
|---|---|---|---|
| `gdn_core` | `apxinf_eight_lanes` | `GatedDeltaRecurrence` with `APXINF_D3_GDN_CORE` **defaulting to `eight_lanes`** (image path stays `four_values` for parity headroom) | ✅ ported (PR #21) |
| `conv1d` | `apxinf` | `CausalDepthwiseConv`; `APXINF_D3_CONV1D=stock` restores the composed MLX `conv1d` | ✅ ported (PR #21) |
| `rope` | `apxinf` | `HalfSplitRope`, FP32 angles, one rounding to the input dtype | ✅ ported (PR #12) |
| `compile` | `model` | `Compiled::prepare` per admitted length, replay on every request | ✅ ported (PR #12) |
| `gemm_dtype` | `f16` | BF16 only — the variant contract admits `auto`/`bf16`/`bf16-compiled` and **rejects f16** (`metal_d3_lite/mod.rs`) | ❌ missing (blocked by the variant contract, not by a kernel) |
| `gdn_block` | `fused_prep` | the GDN operand chain runs as separate ops (conv, silu, q/k l2-norm, recurrence, gated norm, gate) | ❌ missing |

So the remaining migration surface for d3-lite is **two items**, not six.

### What that means for the "≈24 % gap"

The review's `EngineTailor promoted 2808 ms vs apxinf native 3693 ms` predates PR #21
(the promoted-config port). Any current gap claim has to be re-measured on the same
suite with the same warm-up; the two numbers in the review were not taken at the same
point in the port's history and are not a like-for-like pair.

### The two open items, ordered by risk

1. **`gdn_block=fused_prep`** — in ET this runs the GDN operand chain as one kernel
   (`gdn_fused_prep`). In apxinf it is 6–7 separate dispatches per GDN layer. The
   mechanism is proven on the ET side; the port needs a new Metal kernel and a parity
   pass on all 24 GDN layers (the 42-case gate plus the image path).
2. **`gemm_dtype=f16`** — ET measured a roofline band of 0–7 % for F16 weight copies.
   apxinf rejects the variant today, so the port also means extending the variant
   contract and its parity evidence. F16 changes accumulation rounding, so this one
   carries a real parity risk on the 0.0022 headroom of the Clef-style gates; for
   d3-lite the budget is wider (48-case receipt, 0.15 tolerance) but still needs proof.

## Other families

ET declares no `PROMOTED` for `clef_flash`, `jev`, `minicpm`, `qwen3`, `flux`, `kimi` or
`llama`; their wins live in campaign recipes rather than a promoted point. For those the
migration is per-win, not per-config, and each needs its own evidence trail.

## Recommended first move

Re-measure apxinf native d3-lite against ET promoted **on the ET gate suite** before
porting either open item: if the gap is now small, porting `fused_prep` is the only
remaining item worth doing; if it is still large, the difference is not in the option
list and needs a stage-level comparison instead.
