# P0 contract: MiniCPM-OCR-Jev (ApxInf port)

Status: **P0 in progress.** Contract and snapshot identity are pinned; tokenizer /
prompt parity against the EngineTailor oracle is the remaining P0 exit check.
Parent plan: `doc/minicpm-ocr-jev-port-plan.md`.

## 1. Request and response contract

A request is exactly three fields, all required, paths absolute:

```json
{
  "orig_image": "/abs/path/original_crop.png",
  "render_image": "/abs/path/rendered_candidate_text.png",
  "ocr_text": "the candidate OCR string"
}
```

- The caller supplies both images (the official rendering service is not
  reimplemented). RGB conversion and upstream aspect-ratio padding are preserved.
- One response per request: two-class consistency output.
  `probabilities[1]` is the consistency probability; full logits are retained
  where the CPU oracle needs them.
- Serial B1; `encoded_tokens <= 3072`, **no truncation** — longer requests are
  rejected by name. No autoregressive decode, no KV reuse, no generation settings.
- ApxInf mapping: one `Decision`-family result (question `consistency`, options
  `true`/`false`), settled in P1 when the module lands.

## 2. Snapshot identity

- Model: `openbmb/MiniCPM-OCR-Jev`, revision
  `56960237f0569eed11983fccfdc411c97d5db906`.
- Corpus: `openbmb/JevOCRBench`, revision `a26c4b32dcd3a95490842ca637b80edd2b71d975`
  (72 quality rows across text/formula/table × en/zh × pos/neg; 8 timing rows).
- Snapshot files present in the P0 workspace (sha256, computed locally from a
  tar whose sha256 matches the source: `2803563a…`):

| file | sha256 |
|---|---|
| `config.json` | `fd3d48eacdf595b290afc504ac05aa4a4e44343a227406e60eb61516dc772b34` |
| `generation_config.json` | `daa38670fac2e808a88a68c925f3af81e62b5447bcd81c048756753c0b1cf8d6` |
| `preprocessor_config.json` | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| `processor_config.json` | `d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1` |
| `tokenizer.json` | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| `tokenizer_config.json` | `792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b` |
| `chat_template.jinja` | `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80` |
| `README.md` | `52d4f7c04efd10de9ba8b90b4479233e536d5c410dd2919c875daf60b9b6efa2` |
| `.gitattributes` | `2092b5f59e45c3f6a9109b1cb5bb2ef94bfe0102282bf4f7819334ac09d80c4f` |

- Cross-check: two of these hashes equal their HF blob names
  (`blobs/06b95093…` = tokenizer.json, `blobs/273d8e0e…` = chat_template), which
  confirms the files are the published bytes.
- `model.safetensors` (blob `6e8e4304…`, ~3.9 GiB bf16) is **not yet local** on
  the P0 machine: the only copies are on `haiyan-mac` (M1) and
  `haiyan-mac-mini` (M4), and the tailscale path is currently ~40 KB/s
  (DERP relay). P1 (BF16 parity forward) is therefore blocked on the transfer;
  P0 tokenizer/prompt parity does not need it.

## 3. Precision policy (from the parent plan)

- BF16 baseline first (P1). Then whole-forward FP16 (P2): weights/embeddings/
  activations FP16, **FP32 norms/A_log/logits/head**, and the tower pos/rpe also
  produced in FP16 (the promotion trap).
- ANE wavefront (P3): per-layer fused whole-MLP Core ML program, K=2 chunks,
  `PAD=1408`, rows `1600 < T <= 2816` only; other rows take the stock path
  bit-exactly.
- Each acceleration is default-on with an env kill switch
  (`APXINF_JEV_FP16_COMPUTE=0`, `APXINF_JEV_ANE_WAVEFRONT=0`).

## 4. P0 exit check (remaining)

For all 72 corpus rows: the prompt string from `contract.build_prompt` and the
token ids from the snapshot tokenizer must match the EngineTailor oracle
(`enginetailor/models/jev/reference.py`, Transformers FP32 CPU) byte-for-byte.
Only after that does P1 start, so a later parity failure cannot be blamed on the
tokenizer.

## 5. Notes

- No code in this document's PR; the module lands with P1.
- Machine choice: P0 on the M2 (ApxInf build tree + 23 GiB free). The weight
  transfer is the open logistics item; options are an overnight rsync, copying
  from the mini, or running P1 on `haiyan-mac` where the snapshot is local.
