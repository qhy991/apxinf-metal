# When an optimization transfers: two levers and their preconditions

Written after porting the Qwen3-0.6B decode work (W8 weights, device greedy decoding, fused
residual-add + RMS norm, deferred residual) to three other model paths and measuring each port
instead of assuming it. The result was two KEEPs and four negatives, and the negatives are as
informative as the KEEPs: **an optimization transfers by graph shape, not by model family.**

Every number below was taken on an M1 Pro with paired arms (reverse order across rounds,
`Swapouts == 0` per arm, three rounds minimum). Evidence lives in
`docs/mario-opt/QWEN3-DECODE.md`; the PRs are #73, #79, #81, #85, #87, #88.

## Lever 1 — remove launches (fusion): only for launch-bound graphs

A per-launch fusion pays only when the removed dispatch is a visible fraction of the graph's wall
time. Measure that fraction before writing a kernel: `kernels_per_graph × 6.7 µs ÷ wall_time`.

| model path | graph shape | dispatch share | fused add+norm result |
|---|---|---:|---|
| Qwen3-0.6B decode | 628 kernels / 6.7 ms | ~63 % | **KEEP**: −5 %, −7.2 % with the deferred residual |
| MiniCPM5-2B decode | ~320 kernels / 33.5 ms | ~7 % | no change (33.56 vs 33.64 ms) |
| d3-lite single request | ~224 kernels / 409 ms | ~0.05 % | no change (3 270 vs 3 276 ms) |

Two further results from the same line, both negative and both worth remembering:

- **Fusing projections does not help even when fusion does.** Merging q/k/v into one quantized GEMM
  removes 56 launches per token and changed nothing: those launches are *bandwidth* kernels, and
  removing two of three hides the saving behind the bytes that still must move.
- **Compiling the whole block does not help either**: MLX's `Compiled` collapses host graph
  construction (already 4.2 % of the token) but leaves the device kernel count untouched, and the
  compiled W8 block measured 1.5–3 % *slower*.

What did work is the fusion that sat on the dependency edge between attention and the MLP, where both
kernels had to retire before the next projection could start. **Fusion pays on the critical chain,
not next to bandwidth kernels.**

## Lever 2 — remove bytes (quantization): only when the quantized kernel keeps bandwidth

W8 decode weights halve the bytes a token reads, but MLX's `quantized_matmul` at one row reaches
~72 GB/s where the BF16 GEMM reaches 100–300 GB/s on the same shapes. The lever therefore pays only
when the quantized kernel keeps **at least half** the BF16 bandwidth.

| model path | BF16 decode | W8 variant | result |
|---|---:|---|---|
| Qwen3-0.6B | 9.81 ms/token | −31.7 % (W8 + device greedy), −17 % more with W4 | **KEEP** |
| MiniCPM5-2B | 33.8 ms/token | implemented, 34.4 ms/token | **+1.7 %, reverted** |

Corollary: the *same* quantization has opposite signs on two models because their shapes differ, so
"W8 helped on Qwen3" is not evidence for MiniCPM5. Port the shape probe first.

## The pre-check, before writing any kernel

Run these three measurements on the target (they are cheap and they are what made the negative
results cheap too):

1. **Bandwidth sweep** — the target's projection shapes, BF16 and quantized, pipelined: tells you
   whether the bytes lever can pay and how far from the floor the path already is.
2. **Dependency chain** — the same calls chained, each consuming the previous output: separates
   "serialization tax" from "bandwidth".
3. **Skeleton** — the same *count* of small elementwise ops on the same tensors: gives the dispatch
   cost per launch in isolation. It is an upper bound, not a budget.

If the sweep already sits at the bandwidth floor, no launch-level change will move the wall. If the
skeleton's implied dispatch share is below ~10 % of the token, fusion will hide below noise.

## Transfer checklist

1. Pick the lever by graph shape (dispatch share for fusion, bandwidth headroom for quantization).
2. Measure the target's baseline with paired arms and the swap gate before touching code.
3. Implement behind an env gate with an opt-out, keep the parity evidence (token-level identity for
   decode; the model's own quality gate for image/audio targets).
4. Revert when it does not pay. A rejected direction ships as a measurement in the line's document,
   never as dead code on `main`.

## Host noise comes from the desktop session, not from throttling

The TTS line spent a day treating its own variance as thermal: the same configuration read RTF 0.83 to
1.47 on the M1 Pro. It was not thermal. `ps` on the same machine showed a desktop session holding a core
continuously:

```
PID    %CPU  ELAPSED      COMMAND
441    52.3  05-23:32:54  .../SkyLight.framework/.../WindowServer
25144  17.5  02-11:03:48  .../Ventura.appex/Contents/MacOS/Ventura
```

Two system UI processes at ~70 % of one core for days. That is enough to move a 3 s synthesis both ways
and to make a 6.7 ms decode token read 6.1–6.9 ms.

**Rule.** Record the host state with every measurement — `ps -Ao pid,pcpu,etime,comm | sort -k2 -rn |
head -5` — and say which session it was taken in. If a system UI process is above ~20 % CPU, absolute
numbers are good to about ±30 %; paired deltas taken in the same window are still usable, which is why
the paired-arm discipline matters more than the absolute reading. Laptop measurements taken with a
logged-in desktop are not comparable to ones taken on a headless or locked machine, and no amount of
repeating the same arm fixes that.
