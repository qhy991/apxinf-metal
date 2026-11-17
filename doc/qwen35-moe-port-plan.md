# Port plan (capability, criteria-first): Qwen3.5 native MoE → ApxInf CUDA

Status: **capability gap only — EngineTailor has a native MoE lane but no
admitted acceleration. Do not start code until the criteria in §4 are met.**
This is the third entry of the EngineTailor → ApxInf port-plan set; it differs
from the other two in that there is not even a headroom-mode receipt yet: the
lane has a frozen baseline harness and an operator census, but no comparative
effect size.

Baselines: EngineTailor `origin/main` `64092d2f` (lane
`enginetailor/models/qwen35/`, MoE files `model_moe.py`, `loader_moe.py`,
`graph_decode_moe.py`, `gate_moe.py`, `bench_moe.py`); ApxInf fork
`origin/main` `a4836413` (dense Qwen3.5 only, through the `qwen38` CUDA
registration; no expert routing anywhere in `crates/apxinf-model/src`).

## 1. What the lane is

- A native Qwen3.5-MoE implementation in EngineTailor: routed experts plus
  dense/shared path, with its own loader, decode graph and gate. It is **not**
  the external SGLang TP2 provider lane
  (`targets/qwen35-122b-a10b-fp8-b200-tp2/`), which stays out of scope.
- `bench_moe.py` defines a frozen **B16/B32, 32K-native decode baseline** and
  reports median CUDA ms per batch step, aggregate tokens/s and wall tokens/s.
  It is a measurement harness, not a result: no WIN/ENABLE receipt exists for
  this lane at `64092d2f`.

## 2. What is known about where the time goes (no effect size)

| observation | value | source |
|---|---|---|
| routed-MoE share of a profiled window | 30.86% (B16) / 39.65% (B32) | `docs/ENGINEATLOR-ENGINE-PLATFORM.md` routed-MoE census |
| one 2.618389 s window routed-MoE time | 1,110.742 ms (42.421%), 41,568 events | same |
| prefill 8K / 32767 full-attention share | 20.19% / 46.87% | same |
| slow-native agreement gate (dense FP8 / routed-MoE / GDN / full attention) | 397 / 405 / 406 / 405 of 416 | same |

Reading: MoE routing is a large share of the window, which makes the family
interesting, but EngineTailor never turned that into an admitted gain. A port
therefore starts with a baseline, not with an optimization claim.

## 3. Why it would come to ApxInf at all

ApxInf's `qwen38` is dense-only; an agent serving Qwen3.5-MoE today has no native
ApxInf path. The capability increase is real, and the routing share suggests
headroom, but the plan does not promise a speedup.

## 4. Pre-registered acceptance criteria (registered before any measurement)

- **C1 (baseline first)**: reproduce EngineTailor's frozen B16/B32 32K decode
  baseline on the same hardware class (CUDA) with the same workload definition;
  record tokens/s, ms/batch-step and the routing census. No optimization is
  planned before this is reproducible.
- **C2 (parity)**: match or exceed the slow-native agreement gate per section
  (dense FP8 / routed-MoE / GDN / full attention) on the 416-token check set,
  with the same tokenizer and prompt contract.
- **C3 (performance gate, registered after C1)**: a target effect size for
  decode tokens/s vs the reproduced baseline must be written down **before**
  optimization starts (and before any kernel work); A/A twin within ±3% on a
  clean-boot window.
- **C4 (scope)**: decode at B16/B32, 32K context first. Prefill at 8K/32767 and
  multi-GPU/TP are follow-up tickets, not part of the first port.

## 5. If C1–C2 pass: mapping to ApxInf

- New module `crates/apxinf-model/src/cuda_qwen35_moe/` (or an explicit MoE
  branch inside `qwen38` if the dense path can share the loader), registered as
  `qwen3_5_moe`, reusing `apxinf-loader`, `apxinf-tokenizer` and the
  `crates/apxinf-cuda-new` kernel surface.
- First work is an inventory: which grouped-GEMM / routing primitives already
  exist in `apxinf-cuda-new`, what the static-FP8 calibration path looks like,
  and whether the shared expert can reuse the dense `qwen38` blocks.
- The result contract is text generation; no new trait.

## 6. Non-goals

- No Metal/MLX MoE (Apple silicon) in this plan.
- No SGLang TP2 provider port, no multi-GPU work.
- No speedup claim until C3 exists with data.
