# d1-2B Metal port (P0→P3): reference binding, fixtures, role map, contract proposal, parity and baseline

This is the port record for issue #123: `LiquidAI/d1-2B` (1.92B decision
model, LFM3-VL backbone + System One readout) into the apxinf-metal native
Metal/MLX path as the `metal_d1` family. Structure follows
[d3-lite-metal.md](d3-lite-metal.md). P0 (this phase) ships **no product
code** — reference binding, oracle, frozen fixtures, role map, and the shared
`decision.rs` contract proposal for confirmation.

## 1. Port definition

| Item | Value |
| --- | --- |
| Checkpoint | `LiquidAI/d1-2B` @ `a6bce774e6dde338dcc3036ceb5d91f78bf4c19e` |
| Weights | single `model.safetensors`, 3,836,417,920 B BF16, sha256 `b5e573687315e82ea80f4f7dd8d76b5cec3b6ee97e3509fd0493cfebb203719f` (LFS record), no shards; tied embeddings |
| License | LFM Open License v1.0; reference code read for parity only, any copied line carries provenance |
| Text backbone | LFM3: 24 layers (13 conv + 6 full + 5 sliding), hidden 2048, 16Q/4KV, head_dim 128, inter 7680, conv k=7, sliding 256, sandwich norm, dual RoPE (full θ1e6 partial 0.5; sliding θ1e4) |
| Vision tower | `lfm3_vl_vision`: 14 layers (7 conv + 7 full attention), hidden 1024, 16Q/8KV, inter 2560, patch 16, temporal 2, merge 2, axial RoPE θ1e6, downsample 2 |
| Projector | LayerNorm(1e-6) → 4×1024→4096 GELU → 4096→2048, bias |
| D1 readout | `d1_codes 2`, `d1_numbers true`, `d1_attention "set"`, `d1_set_size 16`, `d1_vision_max_pixels 1,638,400`, `calibration.json` temperature + tables |
| Special tokens | bos 124894 / eos 124900 / pad 124893 / vision_start 125009 / vision_end 125010 / image 125030 / video 125029 |
| Reference runtime | transformers 5.19.0 + `trust_remote_code=True`; vendored LFM3 classes (the checkpoint's own `modeling_lfm3*.py` register over transformers'); `AutoModel` → `D1Model`; `model.system_one(state, questions, media)` |
| Target | Apple M1 Pro 16GB, `Device::Metal(0)`, MLX SDK `~/apxinf-runtime/mlx-sdk` |
| Public API | decision family (non-generative); **no token generation — not `LlmTrait`** |
| Gates (pre-registered in `devlocal/d1-2b-port/LEDGER.md`) | R1 labels/spans exact; R2 max\|Δp\| ≤ 3× reference self-noise (MPS↔CPU, measured at P0); R3 two runs bit-identical, probabilities finite and normalized |

## 2. Reference binding (P0)

- Snapshot `~/models/LiquidAI--d1-2B` (39 files incl. `assets/`,
  `.eval_results/`); weights sha256 verified against the HF LFS record.
- A pre-existing local snapshot (`~/models/d1-2B-bf16`, 2026-11-16) had the
  same byte size but sha256 `70a1c060…` ≠ pinned — treated as
  untrusted/stale, superseded by the verified download, and removed after
  verification (3.6 GiB reclaimed). Cause: an interrupted chunk-parallel
  download leaves a non-prefix partial; re-downloads must verify sha256
  (done here: `SHA256 OK` from a clean sequential fetch).
- Oracle venv `devlocal/d1-2b-port/oracle-venv`: python 3.12, torch 2.15.0
  (MPS available), transformers 5.19.0. `torch.compile` is not used on MPS
  (inductor MPS unsupported); the reference runs **eager** on both arms.
- CUDA-only reference deps absent on this host (`lfm3_causal_conv1d*` Triton,
  flash-attn varlen, `causal_conv1d`): the reference falls back to its pure
  PyTorch paths (`Lfm3ShortConv._conv`, chunked SDPA, `F.conv1d`) — recorded
  per the d3-lite precedent; no semantics replaced.
- Disk pre-check (hard constraint): 29 GiB free at start; peak usage
  weights 3.84 GiB + venv ~3 GiB + fixture artifacts < 0.1 GiB.

## 3. Frozen fixture corpus (P0)

27 fixtures (`devlocal/d1-2b-port/fixtures/corpus.json`), each with inputs,
full `answers` JSON, `usage.input_tokens`, wall time, and a `_run.json`
environment record per arm. Independently of the weights, the exact prompt
text, token ids, `state_len`, per-token question ids and answer-slot
positions of every fixture are frozen in `fixtures/prompts/` (image fixtures
include the expanded `<|image|>` token counts and `pixel_values` shapes), so
the native port can check its prompt construction byte-for-byte before any
forward. `usage.input_tokens` from the oracle matches the freeze exactly on
every fixture.

| Coverage | Fixtures |
| --- | --- |
| text-only noul | t01 (EN), t02 (ZH) |
| choice single | t03 (EN, 3), t04 (ZH, 4) |
| choice multiple | t05 (EN, 4), t06 (ZH, 4) |
| score | t07 (EN, 5), t08 (ZH, 3) |
| text read | t09 (EN, nested JSON + `path`), t10 (ZH) |
| number read | t11 (EN, unit EUR), t12 (ZH) |
| number estimate (range+step) | t13 (EN, step 1), t14 (ZH, step 10) |
| date read | t15 (EN), t16 (ZH) |
| multi-question same state | t17 (EN, 4), t18 (ZH, 3) |
| `set_size=1` | t19 (3 questions, 3 sequences) |
| `set_size=16` | t20 (16 questions packed, 2 read double-slots) |
| `system_one_batch` | t21 (2 requests), t22 (3 requests, mixed) |
| image + number | i02 (512px) |
| image + box | i03 (800px + text state) |
| image + boxes | i04 (800px, 80 slots) |
| image + noul / choice | i01 (256px), i05 (ZH + state + media) |

Images: `fixtures/images/{image-256,image-512,image-800}.jpg` (pinned
sha256, copies of `tests/images/`), the same cat photo at three sizes.
Chinese states: t02/t04/t06/t08/t10/t12/t14/t16/t18/i05. English states:
t01/t03/t05/t07/t09/t11/t13/t15/t17.

## 4. Oracle reproducibility and self-noise (P0)

Arms: MPS BF16 (official GPU usage) run twice (separate processes); CPU BF16
and CPU FP32 (official CPU usage) once each. All eager (`torch.compile` has
no MPS inductor backend).

- **Reproducibility (MPS, two runs): 27/27 fixtures byte-identical** answers
  JSON (`reports/compare-mps-run1-vs-run2.json`). P0 oracle-repeatability
  gate: **pass**.
- Self-noise MPS↔CPU (**BF16 both arms**, isolates device/kernel-order
  noise): worst max|Δp| over all probability leaves =
  **0.021458** (t20 `q06.confidence`); text bucket worst 0.021458, image
  bucket worst 0.001768; 7/27 fixtures byte-identical across devices.
  **Label-level diffs: 0/27** — every choice, span, text, number, date, box
  and count agrees across devices (`reports/compare-mps-vs-cpu-bf16.json`).
- MPS-BF16 ↔ CPU-FP32 (dtype+device, context anchor):
  worst max|Δp| = 0.014783 (same t20 fixture), text worst 0.014783, image
  worst 0.001768; label-level diffs 0/27
  (`reports/compare-mps-vs-cpu-fp32.json`). The BF16 quantization share of
  the cross-device noise is smaller than the device share; the R2 anchor
  stays the BF16 cross-device number.
- `usage.input_tokens` agrees with the independent prompt freeze on all
  27 fixtures × 4 arms (108/108).
- **R2 budget** (pre-registered formula `3 × max_ref_self_noise`, BF16
  cross-device): text-only fixtures max|Δp| ≤ **0.0644**, image fixtures
  max|Δp| ≤ **0.0053**. Filled from measurement, not changed after.

## 5. Role map and operator gaps

Full table in `devlocal/d1-2b-port/reports/role-map.md`; summary:

- **Reuse with provenance (text backbone)**: the 24-layer LFM3 stack is
  field-for-field identical to `metal_lfm3` (`origin/feat/lfm3-metal`);
  `metal_d1/model.rs` copies it (family isolation rule — no cross-family
  imports).
- **Adapt existing patterns**: slot-logit readout through the tied LM head
  (d3-lite's masked-softmax readout pattern), vision placeholder scatter
  (d3-lite P1b), non-causal SDPA for the vision tower.
- **New family compositions (no new backend kernels expected)**: set-attention
  block masks on full-attention layers (state bidirectional / question sees
  state + itself; conv+sliding stay causal), pointer readout
  (query·state/√D over valid state positions + span argmax), number-grid
  readout tables, calibration tables, box decode (median/interp edges,
  Halton-512 IoU), bicubic `smart_resize` element-exact preprocessor, axial
  RoPE, gated depthwise Conv2d over the patch grid, patchifier+projector.
- **Explicitly unsupported at first**: video and audio media (reference
  accepts them; native side rejects explicitly rather than silently
  degrading).

## 6. Contract proposal — shared `decision.rs` (needs confirmation before P2)

### 6.1 Can the existing protocols express d1?

**No.** Both existing protocols fail on the output side, and the code
protocol also fails on the input side:

| Protocol | Why d1 does not fit |
| --- | --- |
| `CodeDecisionRuntime` (d3) | Caller pre-encodes the whole ChatML prompt (state render, question blocks, slot offsets, set packing, `<think></think>` lead) — i.e. the caller would re-implement the family. Output is a fixed package code table: cannot carry spans over state text, found flags, parsed number/date values, intervals, or boxes. |
| `SchemaDecisionRuntime` (clef) | Input is close (`state: JSON`, typed questions) but d1's question schema is richer (`multiple`, `range`, `step`, `unit`, `max`) and its answers are typed values: `text` spans with `found`/`path`, `number` with unit and optionally `interval`, ISO dates, `box`/`boxes` with per-block confidence and presence vector. `SchemaDecisionAnswer` (question id + option scores) carries none of these. |

### 6.2 Proposed model-neutral extension

Add a **third decision protocol**, without changing the two existing ones:

```rust
/// Request: named typed questions over a state (JSON value, string or null
/// with media alone). Mirrors the reference API's question schema.
pub struct TypedDecisionRequest {
    pub state: serde_json::Value,
    pub questions: Vec<TypedDecisionQuestion>, // document order kept
    pub media: Vec<DecisionMedia>,             // images as pixel files/decoded tensors
    pub set_size: Option<usize>,               // None = checkpoint's trained size
}

pub struct TypedDecisionQuestion {
    pub id: String,
    pub kind: TypedQuestionKind,               // noul/choice/score/text/number/date/box/boxes
    pub instructions: String,
    pub criteria: Option<serde_json::Value>,   // choice map or score levels
    pub multiple: bool,
    pub range: Option<(f64, f64)>,             // number estimate bounds
    pub step: Option<f64>,
    pub unit: Option<String>,
    pub max: Option<usize>,                    // boxes blocks
}

/// Output: one answer per question, in request order.
pub enum TypedDecisionAnswer {
    Noul { yes: f64 },
    Choice { selected: usize, confidence: f64, probabilities: Vec<f64> },
    MultiChoice { selected: Vec<usize>, probabilities: Vec<f64> },
    Score { level: f64, confidence: f64, probabilities: Vec<f64> },
    Read { found: f64, text: Option<String>, span: Option<(usize, usize)>,
           path: Option<String>, value: Option<TypedValue> },
    Estimate { value: f64, interval: (f64, f64), unit: Option<String> },
    Box { found: f64, rect: Option<DecisionBox>, confidence: f64 },
    Boxes { boxes: Vec<FoundDecisionBox>, count: usize, at_least: bool,
            presence: Vec<f64> },
}

pub enum TypedValue { Text(String), Number(f64, Option<String>), Date(String) }
pub struct DecisionBox { pub top: f64, pub bottom: f64, pub left: f64, pub right: f64 }
pub struct FoundDecisionBox { pub rect: DecisionBox, pub found: f64, pub confidence: f64 }

pub struct TypedDecisionOutput {
    pub answers: Vec<TypedDecisionAnswer>,     // request order
    pub usage_input_tokens: u64,               // output_tokens is always 0
}

pub trait TypedDecisionRuntime {
    fn capabilities(&self) -> LlmCapabilities;
    fn decide_typed(&mut self, request: &TypedDecisionRequest) -> Result<TypedDecisionOutput>;
}

// DecisionModel gains a third variant:
pub enum DecisionModel { Code(..), Schema(..), Typed(Box<dyn TypedDecisionRuntime>) }
```

Contracts the proposal pins down:

1. **Result order** = the request's question order (reference `system_one`
   returns answers keyed by name; names are preserved in
   `TypedDecisionQuestion.id`, answers are positional in request order —
   same rule as `SchemaDecisionOutput`). `boxes` lists blocks in reading
   order (left edge, then top) exactly as the reference.
2. **Probabilities are calibrated** (temperature + monotone tables), matching
   the reference answers field-for-field — parity gates compare against
   `system_one`'s JSON, not raw logits.
3. **Validation failures are errors, not degradation**: unknown question
   type, malformed range/step, `max` above the prompt-version cap, media
   kinds the family does not serve (video/audio at first), more questions
   than the set size, a box question with ≠1 image → a typed error naming
   the question and the violated rule.
4. **Spans index the state as rendered by the prompt** (`state_text`),
   matching the reference's `pointer.py` offsets; `found < 0.5` ⇒
   `text/span/value = None` and `found` kept as the calibrated probability.
5. **No token generation**: the trait has no sampling surface;
   `usage.output_tokens` is 0 by construction.

Alternatives considered and rejected: (a) force d1 through
`CodeDecisionRuntime` (caller re-implements the family, typed answers lost);
(b) extend `SchemaDecisionOutput` with optional typed fields (breaks its
question↔options invariant and the clef callers' assumptions).

### 6.3 CLI (P2)

`apxinf decide --model DIR --request request.json` already dispatches on the
loaded model's protocol; the typed variant adds a `Typed` arm printing the
answers JSON (one line per question, full JSON with `--json`-style flag TBD
in P2). No new subcommand needed unless review prefers one.

## 7. Phase plan

- **P1a text**: `metal_d1/{config,weights,model}.rs` + question/slot encoding
  + D1 readout + calibration; text-only fixtures; R1/R2/R3 gates.
- **P1b vision**: vision tower + element-exact preprocess + projector +
  image-token layout; image fixtures.
- **P2 typed answers + surface**: full typed answer set, `set_size`,
  `system_one_batch` semantics through the public call, CLI arm, explicit
  rejections.
- **P3 closeout**: unoptimized baseline numbers only (no optimization
  claims), final doc, family README, native gates + repo guard scripts, PR.

## 8. Artifacts (private evidence, not committed)

`devlocal/d1-2b-port/`: `LEDGER.md` (pre-registered gates),
`fixtures/corpus.json` + `fixtures/answers/<device>/<run>/`,
`fixtures/images/` (pinned sha256), `scripts/run_oracle.py`,
`scripts/compare_answers.py`, `reports/` (role map, noise, downloads),
`oracle-venv/`.
