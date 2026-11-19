#!/usr/bin/env python3
"""Build QUEUE #8 G1' quality suite: 6 deterministic code tasks for
4-bit vs BF16 solved-rate comparison.

Frozen after AG2-suite-v1 (which must not be edited); selection rule:
MBPP sanitized rows 0..5 (task_id order as cached) + HumanEval/80
(from AG2-suite-v1, row_idx 80) = 6 tasks >= doc/lfm3-native-4bit.md G1' min.

Inputs (M4, offline):
  --mbpp-cache  ~/ag2-datasets-cache/google-research-datasets__mbpp__sanitized__test.json
  --ag2-suite   ~/ag-run/suite/AG2-suite-v1.json   (HumanEval/80 source)
  --out         ~/lfm3-dig-run/window-4bit/g1prime/suite.json
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def mbpp_tasks(rows: list[dict], k: int) -> list[dict]:
    out = []
    for r in rows[:k]:
        problem = r["prompt"].strip()
        stub = r["prompt"].strip()  # MBPP gives prose only; emit full def
        test_code = "\n".join(r["test_list"]) + "\n"
        out.append({
            "id": f"mbpp_{r['task_id']:04d}",
            "benchmark": "mbpp",
            "kind": "code",
            "problem": problem,
            "prompt_stub": stub,
            "test_code": test_code,
            "test_imports": r.get("test_imports") or "",
            "entry_point": None,
            "source": {"dataset": "google-research-datasets/mbpp",
                       "config": "sanitized", "split": "test",
                       "row_idx": None, "task_id": r["task_id"]},
        })
    return out


def humaneval_task(ag2: dict, wanted_id: str) -> dict:
    for t in ag2["tasks"]:
        if t["id"] == wanted_id:
            t = dict(t)
            t.pop("sha256", None)
            return t
    raise SystemExit(f"{wanted_id} not in AG2-suite-v1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mbpp-cache", required=True)
    ap.add_argument("--ag2-suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    mbpp = json.loads(Path(args.mbpp_cache).read_text())
    ag2 = json.loads(Path(args.ag2_suite).read_text())
    # frozen payload hashes (M4 receipt, 2026-11-20 01:3x): the cache must
    # produce byte-identical problem/test payloads or the window aborts
    # (G3-class provenance check; gold `code` is never passed to the model)
    FROZEN = {
        "mbpp_0235": {"prompt": "75c892922e8d", "test": "ba45e64c575d", "code": "40d775056ff0"},
        "mbpp_0238": {"prompt": "8c6e19543e5d", "test": "5eda047bdfc5", "code": "4ed2db6f13d5"},
        "mbpp_0426": {"prompt": "b8467fd4af28", "test": "217d890ea70b", "code": "7938814e3181"},
    }
    tasks = mbpp_tasks(mbpp, args.k) + [humaneval_task(ag2, "humaneval_0080")]
    if args.k == 5:
        import hashlib
        rows = {f"mbpp_{r['task_id']:04d}": r for r in mbpp}
        # selection above is cache rows 0..4 (task_id order); map the frozen
        # ids to their cache rows and verify their gold/test hashes
        cache_ids = [f"mbpp_{r['task_id']:04d}" for r in mbpp[:args.k]]
        for tid, want in FROZEN.items():
            if tid not in cache_ids:
                continue
            r = rows[tid]
            got = {
                "prompt": hashlib.sha256(r["prompt"].encode()).hexdigest()[:12],
                "test": hashlib.sha256(("\n".join(r["test_list"]) + "\n").encode()).hexdigest()[:12],
                "code": hashlib.sha256(r["code"].encode()).hexdigest()[:12],
            }
            for field in ("prompt", "test", "code"):
                if got[field] != want[field]:
                    raise SystemExit(
                        f"FROZEN_PAYLOAD_MISMATCH {tid}.{field}: "
                        f"got {got[field]} want {want[field]} (cache is not the "
                        "frozen AG2 cache row set; aborting before any device run)")
    for t in tasks:
        t["sha256"] = sha256(json.dumps(t, sort_keys=True, ensure_ascii=False))
    suite = {
        "suite_version": "g1prime-v1",
        "frozen_at": None,  # set by builder run time below
        "selection": {
            "rule": ("MBPP sanitized rows 0..%d (task_id order as cached) + "
                     "HumanEval/80 from AG2-suite-v1 row_idx 80" % (args.k - 1)),
            "note": "frozen after AG2-suite-v1; AG2-suite-v1 itself is not edited",
        },
        "datasets": {
            "mbpp": {"dataset": "google-research-datasets/mbpp",
                     "config": "sanitized", "split": "test"},
            "humaneval": {"dataset": "openai/openai_humaneval",
                          "config": "openai_humaneval", "split": "test"},
        },
        "tasks": tasks,
    }
    import datetime
    suite["frozen_at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(suite, indent=2, ensure_ascii=False))
    print(f"wrote {args.out} tasks={len(tasks)} ids={[t['id'] for t in tasks]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
