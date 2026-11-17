#!/usr/bin/env python3
"""P0 parity: reproduce the 72 gate prompts + text token ids from the snapshot tokenizer.

Inputs:
  --anchors   EngineTailor oracle-gate.json (72 cases: id, input_ids, n_tokens, input_sha256)
  --corpus    openbmb/JevOCRBench jevocrbench.jsonl (item_id -> ocr_text)
  --snapshot  local snapshot dir (tokenizer.json; contract hashes must match the anchor binding)
  --contract  EngineTailor enginetailor/models/jev/contract.py (stdlib only), for build_prompt

Method: the anchor `input_ids` are the full processor output (vision blocks expanded). Collapse
every run of the image-pad id to one marker and compare, byte-for-byte, with the text-token
sequence of `build_prompt(ocr_text)` tokenized with `add_special_tokens=False`. The per-image
padding counts are P1's business; P0 owns prompt bytes and text ids.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
from tokenizers import Tokenizer

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--contract", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    sys.path.insert(0, str(Path(a.contract).parent))
    import contract  # noqa: E402

    tk_path = Path(a.snapshot) / "tokenizer.json"
    tk = Tokenizer.from_file(str(tk_path))
    pad_id = tk.token_to_id("<|image_pad|>")
    ph_ids = tk.encode(contract.VISION_PLACEHOLDER, add_special_tokens=False).ids
    assert ph_ids == [tk.token_to_id("<|vision_start|>"), pad_id, tk.token_to_id("<|vision_end|>")], ph_ids

    corpus = {}
    for line in Path(a.corpus).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            corpus[str(r["item_id"])] = r["ocr_text"]

    anchor = json.loads(Path(a.anchors).read_text())
    binding = anchor["binding"]
    snap_hashes = {name: hashlib.sha256((Path(a.snapshot) / name).read_bytes()).hexdigest()
                   for name in ("tokenizer.json", "tokenizer_config.json", "config.json")}
    for name, want in (("tokenizer.json", "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"),
                       ("config.json", "fd3d48eacdf595b290afc504ac05aa4a4e44343a227406e60eb61516dc772b34")):
        if snap_hashes[name] != want:
            print(f"ABORT: snapshot {name} hash {snap_hashes[name][:16]} != anchor binding {want[:16]}", file=sys.stderr)
            return 3

    rows, failures = [], []
    for case in anchor["cases"]:
        cid = str(case["id"])
        ocr = corpus[cid]
        prompt = contract.build_prompt(ocr)
        text_ids = tk.encode(prompt, add_special_tokens=False).ids
        # collapse image-pad runs in the anchor to a single pad marker
        compressed, run = [], 0
        for i in case["input_ids"]:
            if i == pad_id:
                run += 1
            else:
                if run:
                    compressed.append(pad_id); run = 0
                compressed.append(i)
        if run:
            compressed.append(pad_id)
        raw_ids = tk.encode(prompt, add_special_tokens=False).ids
        prompt_ids, run = [], 0
        for i in raw_ids:
            if i == pad_id:
                run += 1
            else:
                if run:
                    prompt_ids.append(pad_id); run = 0
                prompt_ids.append(i)
        if run:
            prompt_ids.append(pad_id)
        ok = compressed == prompt_ids
        rows.append({"id": cid, "label": corpus and None, "n_tokens": case["n_tokens"],
                     "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                     "text_tokens": len(text_ids), "input_sha256": case["input_sha256"],
                     "match": ok})
        if not ok:
            failures.append(cid)
            if len(failures) <= 3:
                for k, (x, y) in enumerate(zip(compressed, prompt_ids)):
                    if x != y:
                        print(f"case {cid}: first divergence at {k}: anchor={x} ours={y}", file=sys.stderr)
                        break
                else:
                    print(f"case {cid}: length mismatch anchor={len(compressed)} ours={len(prompt_ids)}", file=sys.stderr)

    out = {
        "schema": "apxinf.jev.p0-parity.v1",
        "anchor_schema": anchor.get("schema"),
        "binding": {"workload_sha256": binding.get("workload_sha256"), "split": binding.get("split"),
                    "model_revision": binding["model"]["revision"],
                    "tokenizer_sha256": snap_hashes["tokenizer.json"]},
        "cases": len(rows), "matched": len(rows) - len(failures), "failures": failures,
        "rows": rows,
    }
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: out[k] for k in ("cases", "matched", "failures")}, ensure_ascii=False))
    print("max text tokens:", max(r["text_tokens"] for r in rows), "| max n_tokens:", max(r["n_tokens"] for r in rows))
    return 0 if not failures else 1

if __name__ == "__main__":
    raise SystemExit(main())
