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
    ap.add_argument("--force", action="store_true",
                    help="always rebuild (scripts/run-g1prime.sh passes this for the "
                         "20-task suite so a stale suite.json is never reused)")
    args = ap.parse_args()

    mbpp = json.loads(Path(args.mbpp_cache).read_text())
    ag2 = json.loads(Path(args.ag2_suite).read_text())
    # frozen payload hashes: 12-char prefixes of sha256(payload) per field,
    # receipt M4 round-63 (v2, cache rows 0..19) and 2026-11-20 01:3x (v1
    # ids, which are rows 107/109/207 of the same cache). The cache must
    # produce byte-identical prompt/test/gold payloads or the builder
    # aborts before any device run (G3-class provenance check). Gold `code`
    # is never passed to the model — it is hashed here only to prove the
    # test data itself did not change.
    FROZEN = {
        # v1 selection receipt (cache rows 107, 109, 207) — kept so the
        # ids named in the original FROZEN table stay verifiable.
        "mbpp_0235": {"prompt": "75c892922e8d", "test": "ba45e64c575d", "code": "40d775056ff0"},
        "mbpp_0238": {"prompt": "8c6e19543e5d", "test": "5eda047bdfc5", "code": "4ed2db6f13d5"},
        "mbpp_0426": {"prompt": "b8467fd4af28", "test": "217d890ea70b", "code": "7938814e3181"},
        # v2 selection receipt: cache rows 0..19 (MBPP sanitized, task_id
        # order as cached), payload sha256 prefixes 2026-11-20 round-63.
        "mbpp_0011": {"prompt": "66e2a75428bc", "test": "a97090a5fc16", "code": "40106cf3ba07"},
        "mbpp_0012": {"prompt": "e6e4d613a155", "test": "3368efbc1d59", "code": "fc8543a9fcf0"},
        "mbpp_0014": {"prompt": "44f60de72740", "test": "b182f5aea4ec", "code": "6029b3f8b145"},
        "mbpp_0016": {"prompt": "1cad3cce5762", "test": "bc7ab37f3d12", "code": "4f6d10d88e7c"},
        "mbpp_0017": {"prompt": "68145d5728d7", "test": "58a6c0f0ca02", "code": "6bb82afd7280"},
        "mbpp_0018": {"prompt": "46f190653aab", "test": "3143e73bb8ba", "code": "b89ddaa87a06"},
        "mbpp_0019": {"prompt": "f3eb8d101b43", "test": "f7d05a06dac7", "code": "2c145ba10af6"},
        "mbpp_0020": {"prompt": "f999caa6ab6f", "test": "f2de4d050e5a", "code": "05a06ba137a4"},
        "mbpp_0056": {"prompt": "0d84a7799bfd", "test": "9a6d8ed66c18", "code": "4600a0dc8a60"},
        "mbpp_0057": {"prompt": "88aa571abd41", "test": "d1ac9fba1249", "code": "65b7402d6f0f"},
        "mbpp_0058": {"prompt": "b9f053523db9", "test": "e21fed340a1f", "code": "6edaef5c0c5f"},
        "mbpp_0059": {"prompt": "36fd3c08213e", "test": "a90253d1c48b", "code": "03b30988161a"},
        "mbpp_0061": {"prompt": "2b53edac3877", "test": "624bfb0b05ae", "code": "e22d3a5ae3f2"},
        "mbpp_0062": {"prompt": "c9faaffce698", "test": "65b9aac63d19", "code": "b8517ec74d6c"},
        "mbpp_0063": {"prompt": "5b2b5697e799", "test": "992780d1ff70", "code": "c53cbd17f68d"},
        "mbpp_0064": {"prompt": "e284da0c44bd", "test": "0d1d7e91bfcc", "code": "f21994adb024"},
        "mbpp_0065": {"prompt": "fd23b3499bc6", "test": "e31c0c446dcc", "code": "430f43c607da"},
        "mbpp_0066": {"prompt": "ee0be487aea3", "test": "65c97e2d3359", "code": "719d4c184b2f"},
        "mbpp_0067": {"prompt": "88f29f1082d7", "test": "6d7588b6ae13", "code": "37eed046df7a"},
        "mbpp_0068": {"prompt": "f46e736ef355", "test": "69e9df285385", "code": "859b878caf56"},
    }
    tasks = mbpp_tasks(mbpp, args.k) + [humaneval_task(ag2, "humaneval_0080")]
    if args.k >= 5:
        rows = {f"mbpp_{r['task_id']:04d}": r for r in mbpp}
        # selection is cache rows 0..k-1 (task_id order); map every emitted
        # id to its cache row and verify prompt/test/gold hashes. For k=5
        # this is the v1 receipt above; for k=19 it pins all 19 v2 rows.
        cache_ids = [f"mbpp_{r['task_id']:04d}" for r in mbpp[:args.k]]
        for tid in cache_ids:
            want = FROZEN.get(tid)
            if want is None:
                raise SystemExit(f"NO_FROZEN_PAYLOAD for {tid} (add its M4 "
                                 f"hash receipt before emitting it)")
            r = rows[tid]
            import hashlib
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
        "suite_version": "g1prime-v2" if args.k >= 19 else "g1prime-v1",
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
