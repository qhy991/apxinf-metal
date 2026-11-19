#!/usr/bin/env python3
"""QUEUE #8 G1' analysis (doc/lfm3-native-4bit.md @ f10cae5f).

Reads run-g1prime.sh's 4 segment result files and prints:
  SEG   per-segment solved / attempt-count / engine timing (receipt)
  A/A   solved-count consistency across the 2 segments per arm
  CELL  per-task solved-in-both / solved-in-either (ABBA robustness)
  G1'   solved-rate delta (4-bit vs BF16) and PASS / PARTIAL / STOP

Exit 0 if the analysis produced a verdict (PASS, PARTIAL or STOP are all
valid G1' outcomes); exit 2 if any arm is missing a segment (window
incomplete => no verdict).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

SEGS = {"native-4bit-A": [0, 3], "native-bf16-B": [1, 2]}
# gate: doc/lfm3-native-4bit.md §2 G1' / §5 verdicts
PASS_MAX_DELTA_PP = 3.0
STOP_DELTA_PP = 10.0


def load(res: Path, arm: str, seg: int) -> dict | None:
    p = res / f"{arm}-seg{seg}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def seg_receipt(doc: dict) -> dict:
    solved = doc["solved"]
    n = doc["n"]
    attempts = sum(r["attempt_count"] for r in doc["results"])
    # engine timing receipt: all attempts, joined from the request ledger
    pref, dec = [], []
    joined = 0
    for r in doc["results"]:
        for a in r["attempts"]:
            t = a.get("timing") or {}
            if t.get("engine_prefill_ms") is not None:
                pref.append(float(t["engine_prefill_ms"]))
            if t.get("engine_decode_ms_per_token") is not None:
                dec.append(float(t["engine_decode_ms_per_token"]))
            if t.get("engine_request_id"):
                joined += 1
    return {
        "solved": solved, "n": n, "rate_pct": 100.0 * solved / n if n else 0.0,
        "attempts": attempts,
        "ledger_joined": joined,
        "prefill_ms_median": (sorted(pref)[len(pref) // 2] if pref else None),
        "decode_ms_per_token_median": (sorted(dec)[len(dec) // 2] if dec else None),
        "solved_ids": sorted(r["task_id"] for r in doc["results"] if r["solved"]),
        "failed_ids": sorted(r["task_id"] for r in doc["results"] if not r["solved"]),
        "ids": sorted(r["task_id"] for r in doc["results"]),
    }


def rate(solved: int, n: int) -> float:
    return 100.0 * solved / n if n else 0.0


def main() -> int:
    res = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    docs: dict[str, list[dict]] = {}
    for arm, segs in SEGS.items():
        docs[arm] = []
        for seg in segs:
            doc = load(res, arm, seg)
            if doc is None:
                print(f"SEG arm={arm} seg={seg} MISSING => window incomplete")
                return 2
            docs[arm].append(doc)

    receipts = {arm: [seg_receipt(d) for d in docs[arm]] for arm in SEGS}
    for arm, recs in receipts.items():
        for seg, rec in zip(SEGS[arm], recs):
            print("SEG arm=%s seg=%d solved=%d/%d rate=%.1f%% attempts=%d "
                  "ledger_joined=%d prefill_ms_med=%s decode_ms_per_token_med=%s "
                  "solved_ids=%s"
                  % (arm, seg, rec["solved"], rec["n"], rec["rate_pct"],
                     rec["attempts"], rec["ledger_joined"],
                     ("%.2f" % rec["prefill_ms_median"]) if rec["prefill_ms_median"] else "-",
                     ("%.3f" % rec["decode_ms_per_token_median"]) if rec["decode_ms_per_token_median"] else "-",
                     ",".join(rec["solved_ids"]) or "-"))
        rates = [r["rate_pct"] for r in recs]
        spread = max(rates) - min(rates)
        print("A/A arm=%s rates=%s spread_pp=%.2f %s"
              % (arm, "/".join("%.1f" % r for r in rates), spread,
                 "CONSISTENT" if spread < 1e-9 else "DISAGREE (see per-task table)"))

    ids = sorted(set(receipts["native-4bit-A"][0]["ids"]))
    for arm in SEGS:
        for rec in receipts[arm][1:]:
            if sorted(rec["ids"]) != ids:
                print("CELL_SET_MISMATCH arm=%s ids=%s expected=%s"
                      % (arm, sorted(rec["ids"]), ids))
                return 2

    per_task = []
    for tid in ids:
        row = {"task_id": tid}
        for arm in SEGS:
            solved_segs = sum(1 for rec in receipts[arm] if tid in rec["solved_ids"])
            row[arm] = solved_segs
        per_task.append(row)
        print("CELL task=%s 4bit_solved_segs=%d/%d bf16_solved_segs=%d/%d => %s"
              % (tid, row["native-4bit-A"], len(SEGS["native-4bit-A"]),
                 row["native-bf16-B"], len(SEGS["native-bf16-B"]),
                 "BOTH_AGREE" if {row["native-4bit-A"], row["native-bf16-B"]} <= {0, 2}
                 else ("DISAGREE" if row["native-4bit-A"] != row["native-bf16-B"]
                       else "BOTH_AGREE")))

    n = len(ids)
    # primary = per-request solved-in-both-segments (conservative); the
    # report also carries either-segment for the Arm-disagreement case.
    a_both = sum(1 for r in per_task if r["native-4bit-A"] == 2)
    b_both = sum(1 for r in per_task if r["native-bf16-B"] == 2)
    a_either = sum(1 for r in per_task if r["native-4bit-A"] >= 1)
    b_either = sum(1 for r in per_task if r["native-bf16-B"] >= 1)
    delta_both = rate(a_both, n) - rate(b_both, n)
    delta_either = rate(a_either, n) - rate(b_either, n)
    disagree = sum(1 for r in per_task
                   if (r["native-4bit-A"] == 2) != (r["native-bf16-B"] == 2))
    print("CELLSET n=%d" % n)

    worst = min(delta_both, delta_either)
    if worst > -PASS_MAX_DELTA_PP:
        verdict = "PASS"
        verdict_line = ("G1' PASS: 4-bit solved rate does not fall more than %.1f pp "
                        "(worst delta=%.2f pp vs BF16)" % (PASS_MAX_DELTA_PP, worst))
        rc = 0
    elif worst > -STOP_DELTA_PP:
        verdict = "PARTIAL"
        verdict_line = ("G1' PARTIAL candidate: drop %.2f pp (between %.1f and %.1f) "
                        "-> Lead adjudication; 4-bit stays opt-in"
                        % (-worst, PASS_MAX_DELTA_PP, STOP_DELTA_PP))
        rc = 0
    else:
        verdict = "STOP"
        verdict_line = ("G1' STOP: drop %.2f pp > %.1f pp -> 4-bit must not be default"
                        % (-worst, STOP_DELTA_PP))
        rc = 0
    print("G1PRIME solved_rate 4bit_both=%d/%d (%.1f%%) bf16_both=%d/%d (%.1f%%) "
          "delta_both=%.2fpp 4bit_either=%d/%d (%.1f%%) bf16_either=%d/%d (%.1f%%) "
          "delta_either=%.2fpp disagreements=%d/%d"
          % (a_both, n, rate(a_both, n), b_both, n, rate(b_both, n), delta_both,
             a_either, n, rate(a_either, n), b_either, n, rate(b_either, n),
             delta_either, disagree, n))
    print("G1PRIME_VERDICT %s" % verdict)
    print(verdict_line)
    out = res / "g1prime-verdict.json"
    out.write_text(json.dumps({
        "gate": "G1'", "verdict": verdict, "n_tasks": n,
        "4bit_solved_both": a_both, "bf16_solved_both": b_both,
        "delta_both_pp": round(delta_both, 2),
        "4bit_solved_either": a_either, "bf16_solved_either": b_either,
        "delta_either_pp": round(delta_either, 2),
        "disagreements": disagree,
        "per_task": per_task,
        "segments": {arm: receipts[arm] for arm in SEGS},
    }, indent=2))
    print("wrote %s" % out)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
