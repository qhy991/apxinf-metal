# Recipe: `ane-offload-marginal-trade-and-engine-contention`

**When to apply.** You are considering moving a share of a GPU stage onto the
Apple Neural Engine, or growing a share that already runs there, and you have
a standalone rate for the Core ML program you intend to add.

**The claim this recipe refutes.** "The Engine is idle most of the time and
runs this program at R TMAC/s, so moving W work there saves W/R of GPU time."
That reasoning has been wrong three times on this model. It is wrong for two
reasons that a standalone probe cannot see.

## The rule

An offload is worth doing only when, per unit of share,

```
GPU wall time recovered  >  work moved / (Engine rate x contention factor)
```

and each term is measured, not assumed:

- **Engine rate** is from a probe of the real program shape, including every
  cheap op. A simplified probe overstates it. On this model a 16-op stand-in
  read 36% faster than the 57-op program the split actually needs.
- **Contention factor** is the penalty for running this program on an Engine
  that already has another program queued in the same dependency chain.
  Measured here at **0.55**: the head program fell from 1.88 to 1.04 TMAC/s.
  There is no way to read this off a single-program probe, and it is not small.
- **GPU wall time recovered** is measured on the GPU's *blocking* legs, and
  must be summed over all of them. Attributing the fall of one blocking leg to
  the offload, when a second blocking leg absorbed most of it, overstates the
  recovery. Here that error was 2x.
- **Recovery is lossy.** Removing A arithmetic from the GPU returns less than
  `A / GPU rate` of wall time, because some of what you remove was overlapping
  something else. Measured here at about two thirds.

## Two gates that kill an offload independently of the rate

**The host's serial Engine time must stay under the GPU's queue.** Each Core
ML `predict` is a blocking host call. Once `sum(Engine host time per stage)`
exceeds the GPU work left in the queue for that stage, the stage's critical
path is the host, and further offload removes GPU work that was already idle.
Check this before sizing a share, because past that point the rate terms stop
mattering.

**Residency must fit with the share's ceiling, not its first point.** Cost the
share at the largest value you intend to sweep. 128 MiB per attention head
across 32 blocks looked affordable at the first point and put a 16 GB host
into thrashing at the fourth (3.18 M swapouts, every leg four to five fold
worse).

## Corollary: "the Engine is N% idle" is not capacity

Idle measured on wall time is not idle a blocking host loop can reach. If the
dependency chain is `norm -> attention -> norm -> MLP` and each Engine call
blocks, the Engine's gaps fall where the host is not able to issue work. This
is the same fact as the request-level scheduling result: makespan is
`max(GPU, Engine)`, so an Engine at 68.8% idle changed nothing about four
images.

## What has passed this test, and what has not

| offload | Engine rate | result |
|---|---:|---|
| MLP columns, 6656 of 12288 rows | 5.36 TMAC/s | ships, −30.9% |
| MLP by whole tokens | 3.55 TMAC/s | +4.0%, STOP |
| `to_out` columns of 8 heads (Phase A) | n/a, no overlap available | +8.6%, STOP |
| whole attention heads (Phase C) | 1.04 TMAC/s under contention | +11.1%, STOP |

The one that ships is the one where the Engine's rate is high **because the
program is a constant-weight matmul over many rows**, and where it is the only
Core ML program in the block. Every refuted shape either introduced a second
program into the same block or moved work whose arithmetic the Engine is bad
at: activation-by-activation matmuls plus a long chain of cheap ops ran at
1.88 TMAC/s standalone against 5.36 for the MLP at comparable weight volume,
which is the evidence that weight volume was never the limiter.

The mechanism in full: [`README.md`](README.md). The per-arm receipts, logs
and the implementations' exact diffs: [`experiments/qwen-image21-round25-ane-attn-20261109/`](https://github.com/qhy991/EngineTailor/tree/main/experiments/qwen-image21-round25-ane-attn-20261109).
