# What the Neural Engine is worth in Qwen-Image, and where it stops

The MLP column split ships and is worth −30.9% at 512x512 (97.8 s against
141.6 s GPU-only). Three further
offload shapes were implemented and measured on the same host, and all three
are slower than the split alone. This document is the mechanism that explains
them; the rule derived from it is in
[`recipe-ane-offload-marginal-trade.md`](recipe-ane-offload-marginal-trade.md).

None of the three is in this tree. Each is a routine reject, so its exact diff
and its per-arm receipts and logs live in the run archive at
[`experiments/qwen-image21-round25-ane-attn-20261109/`](https://github.com/qhy991/EngineTailor/tree/main/experiments/qwen-image21-round25-ane-attn-20261109),
and what stays here is the part that is reusable: the mechanism, the rule, and
the two probes that measure them.

Host: Apple M4, 16 GB, macOS 26.5. Workload: 512x512, 30 steps, 32 DiT blocks,
so 960 block calls per render. Every leg below is a recorded total divided by
those 960 calls.

## The three refuted shapes

| shape | what moved to the Neural Engine | best point | verdict |
|---|---|---:|---|
| Phase A | the `to_out` columns of 8 of 32 heads | +8.6% | STOP |
| token split | whole tokens, all 12288 MLP rows | +4.0% | STOP |
| Phase C | 4 whole attention heads, end to end | +11.1% | STOP |

All three are regressions. The acceptance criterion was a 5% improvement, so
each fails by its sign before its size matters.

## Phase C in full, because its mechanism is the general one

Phase C moved whole attention heads: a head's q/k/v projections, its RMSNorm,
its RoPE, the text prefix concatenation, the masked softmax, the AV product
and that head's rows of `to_out`. Its feed is the modulated norm, which exists
before any attention work, so unlike Phase A it had real overlap available.

### Measured

| per 960 calls, ms | h=0 | h=4 | h=8 | h=16 |
|---|---:|---:|---:|---:|
| `wait`, blocking on the GPU's MLP feed | 61.44 | 45.48 | 42.51 | 149.07 |
| `attn_wait`, blocking on the head feed | 0 | 12.37 | 10.13 | 12.01 |
| `predict`, the MLP program | 28.19 | 28.45 | 31.37 | 125.10 |
| `attn_predict`, the head program | 0 | 13.60 | 24.48 | 149.58 |
| `convert` | 0.67 | 0.66 | 0.62 | 0.43 |
| **accounted per call** | **90.29** | **100.56** | **109.11** | **436.18** |
| per-step median, ms | 3029.9 | 3367.6 | 3644.1 | 14163.4 |
| **vs control, per-step median** | — | **+11.1%** | **+20.3%** | **+367%** |
| step spread | 1.07% | 12.22% | 6.57% | 22.71% |
| resident programs | 2.94 GiB | 3.44 | 3.94 | 4.94 |
| swapouts during the run | 50 740 | 80 224 | 295 454 | 3 179 813 |

The per-step median is the headline rather than end-to-end wall time. On the
h=8 arm `text_cache` took 8.60 s against the control's 1.38 s and `vae_decode`
5.68 s against 1.88 s, both outside the split and both memory-pressure
artifacts of a run that moved 859 MB to swap. Block accounting and step median
agree to within 0.6 percentage points at every head count, which is what says
the regression lives in the DiT block rather than around it.

### Mechanism 1: the per-head trade is about 3 to 1 against

Read the legs in two groups. GPU-blocking time is `wait + attn_wait`; Engine
time is `predict + attn_predict`.

| | h=0 | h=4 | h=8 | marginal per head |
|---|---:|---:|---:|---:|
| GPU-blocking | 61.44 | 57.85 | 52.64 | **−0.90 then −1.30 ms** |
| Engine | 28.19 | 42.05 | 55.85 | **+3.47 then +3.45 ms** |

Moving one head costs the Engine about 3.46 ms and gives the GPU back about
1.1 ms. The ratio is flat across the sweep, so no head count escapes it.

Splitting `wait` from `attn_wait` is what makes this look better than it is.
`wait` alone falls 61.44 to 42.51 at h=8, which reads as 18.9 ms of GPU time
recovered, but 10.1 ms of that merely moved into `attn_wait`. The real
recovery is 8.8 ms. An earlier report of this work quoted the 18.9 ms figure
and inferred from it that the GPU's payback collapses after the first four
heads; both the number and the inference were wrong, and the corrected reading
is the flat 3-to-1 above.

### Mechanism 2: the two Core ML programs contend on one Engine

The head program's marginal cost is `(24.48 − 13.60) / 4 = 2.72 ms` per head.
A head is 2.831 GMAC per block, so the Engine delivers **1.04 TMAC/s** on it.
The same program measured standalone, on an otherwise idle Engine, runs at
**1.88 TMAC/s**. The offload gets 55% of the rate the probe promised.

The missing half is not inference. It is visible directly in a leg that should
not have moved at all: `predict`, the MLP program, which Phase C never
touches, rises 28.19 to 28.45 to 31.37 ms, **+11.3% at h=8**. Two Core ML
programs queued against one Engine inside the same block crowd each other for
weight bandwidth. The standalone probe measured an Engine with nothing else on
it; the render is not that.

This is why a rate measured by a probe cannot be carried into a split without
a contention term, and the term is large.

### Mechanism 3: the critical path moves off the GPU

At h=4 the host already spends 25.97 ms per block on the Engine
(`attn_wait + attn_predict`), rising to 34.61 ms at h=8. That time is serial
host time: each `predict` is a blocking call. Past the point where it exceeds
the GPU's remaining queue, **the block's critical path is the host's serial
Engine time, not the GPU**, and taking further work off the GPU takes it off
something that was already idle.

The recovery side is lossy even before that. A head is 2.831 GMAC, which at
the GPU's measured 1.60 to 1.86 TMAC/s is 1.5 to 1.8 ms of arithmetic. The GPU
gives back 1.1 ms, so roughly a third of what it sheds never becomes wall
time.

### Mechanism 4: a hard residency wall

Four FP16 projections of 4096x128 per head per block is 4.0 MiB, so every head
offloaded costs **128 MiB** across the 32 blocks. At h=16 resident programs
reach 4.94 GiB, the run takes 3.18 M swapouts and every leg inflates four to
five fold. On a 16 GB host h=16 is unusable independently of everything above.

`attn_wait` also carries about 11 ms per call that does not depend on the head
count at all, the download of the 9.4 MB activation. Driving it to zero would
not change the verdict, because mechanism 1 is negative per head.

### The implementation is correct, and off it changes nothing

Renders at h=4 and h=8 against the control differ by **mean 0.232 and 0.205
out of 255**, worst pixel 22 and 29. That is FP16 rounding. The share computed
on the Engine is the same quantity the GPU computes, so the timings above
convict a working program rather than a broken one.

With its switch at the default 0, the Phase C build rendered
**byte-identical** output to the control that predates it
(`phase-c-rollback-int8` in the run archive), at 92.79 s against 93.67 s. The
shipped path never moved, which is also why nothing of that implementation
needed to land.

The four-case quality comparison was not run. The criterion was conjunctive
and the speed arm is decided.

## Why the earlier probe pointed the other way

A standalone probe put the real 57-op program at 1.88 TMAC/s and predicted
Phase C would win. It was measured on an idle Engine with one program resident
and no second program in the block. Mechanisms 2 and 3 are both properties of
the composition, not of the program, and neither is observable in that setup.
The probe was not wrong about the program; it was answering a different
question than the one the split asks.

## Two directions closed without implementation

**Megakernel**, a persistent Metal kernel with a device-side work queue:
ceiling **1.76%**, against a 5% bar. Launch overhead is 0.12% of a step
(2.1 us per dispatch, 1664 dispatches per step). The seven kernels in `Fused`
already perform the register-level fusion a megakernel would add, measured as
isolated ops totalling 102.76 ms against the real block's 90.29 ms. The GEMM
segments already run at 88 to 94% of the attainable 1.77 TMAC/s peak, so there
is no ALU idle to hide memory traffic behind.

**Request-level scheduling** of four images across GPU and Engine: **0%**.
Makespan is `max(GPU, ANE) = max(361.2, 112.8) ms = 361.2 ms`, which is
90.3 ms per call, exactly today's. The Engine's 68.8% idle is wall-clock idle
the host cannot reach, for the reason mechanism 3 gives.

A separate, real workflow-level gain survives both: `ane-install` is 23.47 s
of program loading paid **once per process**, so rendering four images in one
process instead of four is **−14.7%** per image. It is not an engine
optimization and does not interact with any of the above.

## Reproducing

The two probes are in this tree and need no switch. Building them needs
`SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk` and
`MLX_ROOT=<mlx sdk>`; running needs `DYLD_LIBRARY_PATH=<mlx sdk>/lib`.

Standalone Core ML program rates, including the 57-op Phase C program at each
head count:

```
cargo run -p apxinf-coreml --release --example sdpa_probe --features native
```

GPU op roles, the dispatch floor and the attainable matmul peak:

```
cargo run -p apxinf-model --release --example qimage_gpu_roles --features mlx
```

Re-measuring an end-to-end arm needs the implementation back. Its diff and the
harness that drove it are in the run archive,
[`experiments/qwen-image21-round25-ane-attn-20261109/`](https://github.com/qhy991/EngineTailor/tree/main/experiments/qwen-image21-round25-ane-attn-20261109),
against the commit the archive names. Both offloads were correct, so a rerun
is worth doing if the host's memory, the Neural Engine's contention behaviour
or the block's dependency chain changes; nothing else about them would.
