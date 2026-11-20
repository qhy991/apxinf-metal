#!/usr/bin/env python3
"""Build QUEUE #8 knowledge-eval sample sets: MMLU 500 (subject-stratified)
and GSM8K 200 (uniform), freezing seeds/shas per
doc/lfm3-4bit-knowledge-eval.md §5.

Offline, M1, pyarrow from the mini-swe-agent venv, run as:
  python3 -I scripts/build-knowledge-samples.py \
      --datasets ~/lfm3-dig/datasets --out-dir samples

`python3 -I` disables user site + cwd so a planted json.py/pyarrow in the
current directory cannot shadow stdlib/third-party imports; the script lives
in the repo and is passed in by path (CLAUDE.md §Security).

Emits:
  samples/mmlu-500.jsonl   fields: idx, subject, question, choices, gold
  samples/gsm8k-200.jsonl  fields: idx, question, gold (numeric only)
  samples/mmlu-500.ids.json, samples/gsm8k-200.ids.json (idx sequences, for
      the id-sequence sha256 receipt)
  samples/knowledge-samples.json — receipt: file sha256, id-sequence sha256,
      seeds, quotas, counts, dataset file sha256, frozen_at.
`gold` never appears in the prompt; run the grep self-check in the doc.

Sample files are written once: a second run with different content aborts
unless --force (frozen-sample protection, G3-class provenance).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

MMLU_SEED = 20261120
GSM8K_SEED = 20261121
MMLU_N = 500
GSM8K_N = 200


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def jsonl_bytes(rows: list[dict]) -> bytes:
    return ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode()


def write_once(path: Path, data: bytes, force: bool) -> None:
    if path.exists() and not force and path.read_bytes() != data:
        raise SystemExit(
            f"FROZEN_SAMPLE_MISMATCH {path} already exists with different "
            f"content; refusing to overwrite (pass --force to rebuild the "
            f"suite, which invalidates any prior measurement window)")
    path.write_bytes(data)


def file_sha256(path: Path) -> str:
    with path.open("rb") as f:
        return sha256_bytes(f.read())


def mmlu_stratified(quota: dict[str, int], rng: random.Random, rows: list[dict]) -> list[dict]:
    """quota[subject] = how many of that subject to take; consumed in
    subject-dict order so RNG stream is deterministic given sorted subjects."""
    by_subject: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_subject[str(r["subject"])].append(i)
    for s in by_subject:
        by_subject[s].sort()  # stable within-subject order before sampling
    picked: list[int] = []
    for s in sorted(by_subject):
        k = quota[s]
        if k == 0:
            continue
        picked.extend(rng.sample(by_subject[s], k))
    # exact-count check: quotas built to floor + remainder sum to MMLU_N
    if len(picked) != MMLU_N:
        raise SystemExit(f"MMLU_STRATIFIED_COUNT {len(picked)} != {MMLU_N}")
    return picked


def mmlu_quota(counts: dict[str, int]) -> dict[str, int]:
    total = sum(counts.values())
    if total != 14042:
        raise SystemExit(f"MMLU_TOTAL_ROWS {total} != 14042")
    quota = {s: math.floor(MMLU_N * n / total) for s, n in counts.items()}
    leftover = MMLU_N - sum(quota.values())
    # remainder filled in subject dict order (doc §2 freeze)
    for s in sorted(counts):
        if leftover == 0:
            break
        quota[s] += 1
        leftover -= 1
    if sum(quota.values()) != MMLU_N or leftover != 0:
        raise SystemExit(f"MMLU_QUOTA_BROKEN sum={sum(quota.values())} leftover={leftover}")
    return quota


def gsm8k_gold(answer_text: str) -> int:
    # doc §2: gold = last number; parquet answer contains "#### <gold>" tail
    tail = answer_text.rsplit("####", 1)[-1]
    digits = "".join(c for c in tail if c.isdigit() or (c == "-" and tail.startswith("-")))
    if not digits:
        raise SystemExit(f"GSM8K_NO_GOLD {answer_text!r}")
    return int(digits)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True, help="dir holding mmlu-test.parquet and gsm8k-test.parquet")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ds = Path(args.datasets).expanduser().resolve()
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    mmlu_path, gsm8k_path = ds / "mmlu-test.parquet", ds / "gsm8k-test.parquet"
    if not mmlu_path.exists() or not gsm8k_path.exists():
        raise SystemExit(f"DATASET_MISSING {mmlu_path} {gsm8k_path}")

    mmlu_sha, gsm8k_sha = file_sha256(mmlu_path), file_sha256(gsm8k_path)

    # ---- MMLU 500 ----
    mmlu = pq.read_table(mmlu_path)
    if mmlu.num_rows != 14042 or mmlu.column_names != ["question", "subject", "choices", "answer"]:
        raise SystemExit(f"MMLU_SCHEMA rows={mmlu.num_rows} cols={mmlu.column_names}")
    mrows = [{"question": q.as_py(), "subject": s.as_py(), "choices": c.as_py(), "answer": int(a.as_py())}
             for q, s, c, a in zip(mmlu["question"], mmlu["subject"], mmlu["choices"], mmlu["answer"])]
    counts: dict[str, int] = defaultdict(int)
    for r in mrows:
        counts[r["subject"]] += 1
        if not (0 <= r["answer"] <= 3):
            raise SystemExit(f"MMLU_ANSWER_RANGE idx_answer={r['answer']}")
    quota = mmlu_quota(counts)
    rng = random.Random(MMLU_SEED)
    mmlu_ids = mmlu_stratified(quota, rng, mrows)
    mmlu_out = [
        {"idx": i, "subject": mrows[i]["subject"], "question": mrows[i]["question"],
         "choices": mrows[i]["choices"], "gold": mrows[i]["answer"]}
        for i in mmlu_ids
    ]

    # ---- GSM8K 200 ----
    gsm = pq.read_table(gsm8k_path)
    if gsm.num_rows != 1319 or gsm.column_names != ["question", "answer"]:
        raise SystemExit(f"GSM8K_SCHEMA rows={gsm.num_rows} cols={gsm.column_names}")
    grows = [{"question": q.as_py(), "answer": a.as_py()} for q, a in zip(gsm["question"], gsm["answer"])]
    rng2 = random.Random(GSM8K_SEED)
    gsm8k_ids = sorted(rng2.sample(range(len(grows)), GSM8K_N))
    gsm8k_out = [
        {"idx": i, "question": grows[i]["question"], "gold": gsm8k_gold(grows[i]["answer"])}
        for i in gsm8k_ids
    ]

    # ---- write ----
    mmlu_jsonl = jsonl_bytes(mmlu_out)
    gsm_jsonl = jsonl_bytes(gsm8k_out)
    mmlu_ids_bytes = json.dumps(mmlu_ids).encode()
    gsm_ids_bytes = json.dumps(gsm8k_ids).encode()
    mmlu_file = out / "mmlu-500.jsonl"
    gsm_file = out / "gsm8k-200.jsonl"
    write_once(mmlu_file, mmlu_jsonl, args.force)
    write_once(gsm_file, gsm_jsonl, args.force)
    write_once(out / "mmlu-500.ids.json", mmlu_ids_bytes, args.force)
    write_once(out / "gsm8k-200.ids.json", gsm_ids_bytes, args.force)

    # ---- receipt ----
    import datetime
    receipt = {
        "frozen_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "builder": "scripts/build-knowledge-samples.py",
        "datasets": {
            "mmlu-test.parquet": {"sha256": mmlu_sha, "rows": mmlu.num_rows},
            "gsm8k-test.parquet": {"sha256": gsm8k_sha, "rows": gsm.num_rows},
        },
        "samples": {
            "mmlu": {
                "n": len(mmlu_out), "seed": MMLU_SEED,
                "file": mmlu_file.name, "file_sha256": sha256_bytes(mmlu_jsonl),
                "ids_sha256": sha256_bytes(mmlu_ids_bytes),
                "n_subjects": len(counts), "quota_by_subject": quota,
            },
            "gsm8k": {
                "n": len(gsm8k_out), "seed": GSM8K_SEED,
                "file": gsm_file.name, "file_sha256": sha256_bytes(gsm_jsonl),
                "ids_sha256": sha256_bytes(gsm_ids_bytes),
                "id_order": "ascending (sorted after sample)",
            },
        },
    }
    receipt_path = out / "knowledge-samples.json"
    write_once(receipt_path, (json.dumps(receipt, indent=2, ensure_ascii=False) + "\n").encode(), args.force)

    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
