# Port plan (capability, criteria-first): Qwen3.5-4B + DLite → ApxInf Metal

Status: **capability candidate — no admitted acceleration yet. Do not start code
until the criteria in §4 are met.** This is the second entry of the
EngineTailor → ApxInf port-plan set; unlike MiniCPM-OCR-Jev
(`minicpm-ocr-jev-port-plan.md`), this lane has measured **headroom** but no
admitted end-to-end gain, so the plan registers its acceptance rule *before* the
missing measurement.

Baselines: EngineTailor `origin/main` `64092d2f` (lane
`enginetailor/models/qwen35_dlite/`, target `targets/qwen35-4b-dlite-m4-bf16-v1/`);
ApxInf fork `origin/main` `a4836413` (no Qwen3.5 Metal module; dense Qwen3.5
exists only on CUDA through the `qwen38` registration).

## 1. What the lane is

- Target: `Qwen/Qwen3.5-4B` revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (32 layers: 24 Gated DeltaNet + 8
  full attention, hidden 2560, tied 248320-token embedding), Apple M4 16 GB,
  BF16, serial B1, greedy.
- Draft: `SJTU-DAI-Sys/Qwen3.5-4B-DLite` revision
  `bf98d74ae84ac9cd74219a73b4a603f16ff42332` (MIT, 0.6B). The draft reads the
  target's hidden states at 12 layers for the current anchor, keeps no draft KV,
  proposes 7 tokens through a 5-layer bidirectional block + low-rank RNN head;
  the target verifies anchor+7 in one 8-row forward, keeps the longest matching
  prefix, and restores GDN/conv state on partial acceptance.
- Bench: paired complete-request timing against a detached baseline worktree,
  ABBA/BAAB palindrome blocks, twin A/A arm.

## 2. Measured headroom (not a gain)

| measurement | value | source |
|---|---|---|
| prefill share of weighted suite | **62.06%** (10/10 cases; 42.42% after 256-budget normalisation) | `runs/dlite-prefill-p0-20261112/RESULT.md` |
| decode share of weighted request | 37.94% (agent-4096 alone: 29.15%) | `runs/dlite-decode-accept-20261113/RESULT.md` |
| GPU/ANE overlap available (`S − P`) | 2.472 s (20.8%) @4K, 6.041 s (22.9%) @8K | `runs/dlite-prefill-p0-20261112` |
| MLP-share vs ANE leg margin | 10.27 s vs 0.76 s → **13.5×** @8K/25% | same |
| decode accept ceiling with the current draft | 3.4638 token/round; 5% e2e needs 3.9896 (**+15.2%**) | `runs/dlite-decode-accept-20261113` |
| decode-with-current-draft verdict | **PARK, 0.00% e2e gain** | same |
| ANE-share C arm | **never ran** (two owner preconditions) | `runs/dlite-ane-share-p1-20261113` |
| sync-point cost | 13.8–18.5 µs/call, channel ceiling −0.276% | `runs/dlite-prefill-step-20261113` |
| verify split | proj-GEMM 47–50% of round, GDN 18%, attention 7–11%, lm_head 9–10% | `runs/dlite-verify-split/RESULT.md` |

Reading: prefill is the largest term and the ANE leg has a 13.5× overlap margin,
but the only e2e lever measured so far (acceptance with the current draft) is
exhausted at 0%. The lane is therefore a **capability gap with a concrete
headroom hypothesis**, not an admitted acceleration.

## 3. Why it would come to ApxInf at all

ApxInf has no Metal Qwen3.5: `qwen38` (dense Qwen3.5/Qwen3.8) is
`#[cfg(feature = "cuda")]`-only. So a port is two stacked pieces:

1. **Qwen3.5-4B dense on Metal** — 24 GDN + 8 attention layers, tied embedding,
   greedy decode; the prerequisite and a useful standalone family.
2. **DLite speculative draft + 8-row verify** — draft, state restore on partial
   acceptance, and the paired bench harness.

If the criteria below fail, only piece 1 is worth discussing (it is a capability
increase), and this plan stops at piece 1.

## 4. Pre-registered acceptance criteria (registered before the missing data)

These are fixed now; if one is later judged wrong, add a dated amendment rather
than editing it.

- **C1 (gate for porting the acceleration)**: a registered ANE-share A/B on the
  M4 target (the never-run C arm) shows **e2e ≥ +5%** over the detached
  baseline with the A/A twin within ±3%. If C1 fails, do not port the ANE share.
- **C2 (parity)**: on the frozen workload, greedy argmax agreement with the
  current EngineTailor implementation ≥ 99.5% of tokens with identical
  first-miss restore behaviour; no silent truncation.
- **C3 (regression)**: no case slower by >2% and the twin A/A within ±3% on a
  clean-boot window (swap <0.5 GB, uptime <3 d).
- **C4 (dependency)**: piece 1 (Metal Qwen3.5-4B dense) must first pass its own
  parity/bench gate; piece 2 is not planned before that.

## 5. If C1 passes: mapping to ApxInf

- New modules `crates/apxinf-model/src/metal_qwen35/` (dense) and
  `crates/apxinf-model/src/metal_qwen35_dlite/` (draft + verify), reusing
  `apxinf-mlx`, `apxinf-loader`, `apxinf-tokenizer`, and the `metal_qwen3`
  family layout; registration `metal_qwen35` / `metal_qwen35_dlite`.
- GDN/conv state restore must follow the upstream contract exactly (the
  EngineTailor lane documents the restore-on-partial-accept behaviour).
- ANE share reuses the `apxinf-coreml` + `metal_clef/ane_admission` pattern with
  the overlap geometry from §2; fallback to GPU-only when admission fails.
- Bench: port the ABBA/BAAB paired harness so the criteria are measurable on the
  same terms.

## 6. Non-goals

- No ApxInf port of the SGLang TP2 external provider
  (`qwen35-122b-a10b-fp8-b200-tp2`); that is a separate external lane.
- No paper claim from this plan; the headroom numbers are EngineTailor's on M4.
