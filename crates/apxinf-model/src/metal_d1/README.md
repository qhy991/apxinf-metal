# metal_d1 — LiquidAI d1-2B decision model on the native MLX path

`metal_d1` is the Metal (MLX) port of **`LiquidAI/d1-2B` @ `a6bce774e6dde338dcc3036ceb5d91f78bf4c19e`** (1.92B, BF16): a *decision* model, not a chat model. It answers named typed questions over a state — noul / choice / score / text / number / date / box / boxes — in one prefill pass per packed sequence. There is **no token generation and no `LlmTrait` implementation**: the readout is slot logits through the tied embedding, pointer spans into the state, and calibration tables.

The port record is [`doc/d1-2b-port.md`](../../../../doc/d1-2b-port.md) (issue #123), with the P0 reference binding, frozen fixtures, role map, measured reference self-noise and the pre-registered R1/R2/R3 gates.

## Provenance and family isolation

The text backbone is field-for-field LFM3-1.7B: `config.rs` (text geometry/validation), `weights.rs` (checkpoint mapping) and the layer arithmetic in `model.rs` are **copied** from `metal_lfm3` (`origin/feat/lfm3-metal` @ 1ed3f2f3) and adapted — never imported, per the backend-native port rule. Everything d1-specific is new family code: the set-attention program in `model.rs`, and the question engine (`questions.rs`, `prompt.rs`, `readout.rs`, `runner.rs`, `calibration.rs`, `parsing.rs`).

## Checkpoint and supported contract (P1a)

| Item | Value |
| --- | --- |
| Checkpoint | single `model.safetensors` 3,836,417,920 B BF16, sha256 `b5e57368…203719f`; tied embeddings; `calibration.json` temperature + tables |
| Text backbone | 24 layers (13 conv + 6 full + 5 sliding), hidden 2048, 16Q/4KV heads of 128, conv width 7, sliding 256, sandwich norms, tied head |
| Attention | `d1_attention = "set"`: full layers bidirectional per the reference `Sees` mask (state sees the whole packed row; a question sees the state and itself); sliding and conv layers stay causal |
| Readout | `d1_codes 2`, `d1_numbers true` (number grids as tokenizer numbers), `d1_set_size 16`, calibration temperature + monotone tables |
| P1a scope | text-only requests; one sequence per forward (`set_size` honored for packing); `system_one` per request |
| Rejected at load | non-BF16 tensors, untied embeddings, conv bias, no-sandwich, `d1_attention != "set"`, `d1_codes != 2`, `!d1_numbers`, video/audio media |

The shared `decision.rs` protocol for d1's typed answers is a proposal under
confirmation on issue #123 §6; until it lands, the family loads through
`metal_d1::load` and the `d1_fixtures` example, not `AutoModel`.

## Execution

Per layer: `operator_norm → attention(n) | short_conv(n) → post_operator_norm → residual; ffn_norm → MLP → post_ffn_norm → residual` (copied LFM3 recipe: BF16 weighted RMSNorm, FP32 zero-tail RoPE tables built with the platform libm, FP32 depthwise conv, per-head sigmoid gate). One sequence per call; the D1 runner (`runner.rs`) builds the packed sequences (`prompt.pack_set` semantics), runs `model.forward_hidden`, gathers slot logits (FP32 log-softmax through the tied head), runs pointer reads (query·state/√D over valid state positions) and assembles calibrated answers (`readout.rs`, `calibration.rs`) — the same math as the reference's `runner.py`/`prompt.py`/`pointer.py`/`box.py`/`api.py`.

## Verification

- **Prompt gate** (host-only, no weights): all 22 text fixtures' packed token ids, state lengths, question tags and answer-slot positions are byte-identical to the frozen reference prompts (`devlocal/d1-2b-port/fixtures/prompts/`).
- **Parity** on the 22 text fixtures vs the frozen MPS BF16 reference answers: labels/spans exact; max |Δp| 0.0134 ≤ 0.0644 (pre-registered R2, 3× reference self-noise); native run1 vs run2 byte-identical (R3).
- **Baseline**: no optimization is claimed or performed in P1a; eager composition only.
