#!/usr/bin/env python3
"""QUEUE #8 knowledge/reasoning eval: segment driver + analyzer.

Two subcommands (run-knowledge-eval.sh keeps no inline heredoc — round-55
lesson):

  segment  replay one segment: send greedy requests for the subset
           (MMLU + GSM8K) to the freshly-started gateway, join engine
           timings from the gateway --request-log by response id
           (ordered next-unclaimed fallback, code-loop.py join), and
           write {arm}-seg{seg}.json. Gateway errors are recorded with
           their full body (round-50-57 lesson: always read the body).
  analyze  read all segment files, print SEG / A/A / CELL / delta /
           flip-rate lines, decide the verdict, write knowledge-verdict.json.

Answer extraction (prereg doc §1, frozen):
  MMLU   first [ABCD] in the reply; gold = chr(65 + sample["gold"]);
         no letter found => unparsed.
  GSM8K  last number in the reply (sign kept, commas stripped);
         gold = sample["gold"]; no number => unparsed.

Verdict (Lead 10:52 scale revision, doc §1/STATUS 10:52):
  parsed answers only, accuracy is over N answers of that dataset
  (unparsed counted separately and do NOT add to the denominator);
  difference = |acc_4bit - acc_bf16| in questions (pp with n=10):
    <= 1 question  => SMOKE_OK      (difference small, may scale up)
    >= 3 questions => SMOKE_STOP    (4-bit stays opt-in, not primary)
    2 questions    => SMOKE_GREY    (report digits, Lead decides)
  exit 0 for any verdict; exit 2 if the window is incomplete.

Smoke mode (default n=10/10, 1 segment per arm) is explicitly NOT a
statistical conclusion — the analyzer prints that caveat with the
verdict. The full prereg verdict (delta >= -3pp PASS / -3..-10 PARTIAL /
< -10 FAIL) is printed alongside whenever n is large enough to be
meaningful (documented as a secondary reading, not the smoke verdict).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LETTER = re.compile(r"[ABCD]")
LAST_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

PASS_MAX_DELTA_PP = 3.0
STOP_DELTA_PP = 10.0

# smoke-revision gates (Lead 10:52)
SMOKE_OK_QUESTIONS = 1
SMOKE_STOP_QUESTIONS = 3

MMLU_TEMPLATE = ("Answer the following multiple choice question. Reply with only the letter "
                 "of the correct answer.\n\n{question}\n"
                 "A) {a}\nB) {b}\nC) {c}\nD) {d}")
GSM8K_TEMPLATE = 'Solve the problem. End with "The answer is <number>".\n\n{question}'


# ---- request / prompt ----

def mmlu_prompt(sample: dict) -> str:
    ch = sample["choices"]
    if len(ch) != 4:
        raise ValueError("MMLU sample %s has %d choices" % (sample["idx"], len(ch)))
    return MMLU_TEMPLATE.format(question=sample["question"],
                                a=ch[0], b=ch[1], c=ch[2], d=ch[3])


def gsm8k_prompt(sample: dict) -> str:
    return GSM8K_TEMPLATE.format(question=sample["question"])


def ask(gateway: str, model: str, prompt: str, max_tokens: int, timeout: int) -> dict:
    """One greedy completion. Returns {"ok": bool, "reply": str, "error": str,
    "usage": {...}, "finish_reason": str, "wall_s": float}. Gateway error
    bodies are always surfaced (round-50-57 lesson)."""
    payload = {"model": model, "max_tokens": max_tokens, "temperature": 0,
               "messages": [{"role": "user", "content": prompt}]}
    url = gateway.rstrip("/") + "/v1/chat/completions"
    t0 = time.time()
    proc = subprocess.run(
        ["curl", "-s", "-m", str(timeout), "-X", "POST", url,
         "-H", "content-type: application/json", "-H", "x-api-key: sk-local",
         "-d", json.dumps(payload)],
        capture_output=True, text=True)
    wall = time.time() - t0
    if proc.returncode != 0:
        return {"ok": False, "reply": "", "error": "curl rc=%d %s"
                % (proc.returncode, proc.stderr[-300:]), "usage": {},
                "finish_reason": None, "wall_s": round(wall, 3)}
    body = proc.stdout.strip()
    try:
        doc = json.loads(body)
    except json.JSONDecodeError:
        return {"ok": False, "reply": "", "error": "non-json body: %s" % body[:400],
                "usage": {}, "finish_reason": None, "wall_s": round(wall, 3)}
    if not isinstance(doc, dict) or "choices" not in doc:
        return {"ok": False, "reply": "", "error": "gateway error body: %s" % body[:400],
                "usage": {}, "finish_reason": None, "wall_s": round(wall, 3)}
    try:
        choice = doc["choices"][0]
        reply = (choice.get("message") or {}).get("content") or ""
        usage = doc.get("usage") or {}
        finish = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError) as exc:
        return {"ok": False, "reply": "", "error": "malformed success body: %s (%s) %s"
                % (exc, type(exc).__name__, body[:300]),
                "usage": {}, "finish_reason": None, "wall_s": round(wall, 3)}
    return {"ok": True, "reply": reply, "error": "", "usage": usage,
            "finish_reason": finish, "wall_s": round(wall, 3)}


# ---- ledger join (code-loop.py pattern) ----

def ledger_rows(path: str) -> list[dict]:
    rows = []
    if not path:
        return rows
    try:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("request_id"):
                rows.append(row)
    except OSError:
        return rows
    return rows


def join_timing(rows: list[dict], claimed: set[str]) -> dict:
    """Next unclaimed ledger row in file order (one request -> one row;
    ordering guaranteed because the gateway is exclusively owned and
    requests are serial)."""
    for row in rows:
        rid = row["request_id"]
        if rid in claimed:
            continue
        claimed.add(rid)
        return {"engine_prefill_ms": row.get("prefill_ms"),
                "engine_decode_ms_per_token": row.get("decode_ms_per_token"),
                "engine_decode_ms": row.get("decode_ms"),
                "engine_completion_tokens": row.get("completion_tokens"),
                "engine_prompt_tokens": row.get("prompt_tokens"),
                "engine_source": "apxinf.ledger",
                "engine_request_id": rid}
    return {}


# ---- extraction ----

def extract_mmlu(reply: str) -> str | None:
    m = LETTER.search(reply)
    return m.group(0) if m else None


def extract_gsm8k(reply: str) -> int | None:
    nums = LAST_NUMBER.findall(reply)
    if not nums:
        return None
    raw = nums[-1].replace(",", "")
    try:
        val = float(raw)
    except ValueError:
        return None
    if val.is_integer():
        return int(val)
    return None


# ---- segment driver ----

def load_subset(samples: Path, n_mmlu: int, n_gsm8k: int,
                mmlu_seed: int, gsm8k_seed: int) -> tuple[list[dict], list[dict]]:
    mmlu_all = [json.loads(l) for l in (samples / "mmlu-500.jsonl").read_text().splitlines() if l.strip()]
    gsm_all = [json.loads(l) for l in (samples / "gsm8k-200.jsonl").read_text().splitlines() if l.strip()]
    if len(mmlu_all) != 500 or len(gsm_all) != 200:
        raise SystemExit("SAMPLE_COUNT mmlu=%d gsm8k=%d" % (len(mmlu_all), len(gsm_all)))
    rm = random.Random(mmlu_seed)
    rg = random.Random(gsm8k_seed)
    mmlu_pick = sorted(rm.sample(range(500), n_mmlu))
    gsm_pick = sorted(rg.sample(range(200), n_gsm8k))
    mmlu_sub = [mmlu_all[i] for i in mmlu_pick]
    gsm_sub = [gsm_all[i] for i in gsm_pick]
    return mmlu_sub, gsm_sub


def run_segment(args) -> int:
    res = Path(args.res)
    samples = Path(args.samples)
    mmlu_sub, gsm_sub = load_subset(samples, args.n_mmlu, args.n_gsm8k,
                                     args.mmlu_seed, args.gsm8k_seed)
    ledger = ledger_rows(args.request_log)
    claimed: set[str] = set()
    results: list[dict] = []

    def one(ds: str, sample: dict, idx: int, total: int) -> dict:
        prompt = mmlu_prompt(sample) if ds == "mmlu" else gsm8k_prompt(sample)
        cap = args.max_tokens_mmlu if ds == "mmlu" else args.max_tokens_gsm8k
        a = ask(args.gateway, args.model, prompt, cap, args.timeout)
        timing = join_timing(ledger, claimed) if args.request_log else {}
        if ds == "mmlu":
            parsed = extract_mmlu(a["reply"]) if a["ok"] else None
            gold_letter = chr(65 + int(sample["gold"]))
            row = {"dataset": ds, "idx": sample["idx"], "gold": gold_letter,
                   "parsed": parsed, "parsed_ok": parsed is not None,
                   "correct": (parsed == gold_letter) if parsed is not None else None}
        else:
            parsed_n = extract_gsm8k(a["reply"]) if a["ok"] else None
            row = {"dataset": ds, "idx": sample["idx"], "gold": sample["gold"],
                   "parsed": parsed_n, "parsed_ok": parsed_n is not None,
                   "correct": (parsed_n == sample["gold"]) if parsed_n is not None else None}
        row.update({"ok": a["ok"], "error": a["error"], "finish_reason": a["finish_reason"],
                    "wall_s": a["wall_s"], "usage": a["usage"], "timing": timing,
                    "reply_tail": a["reply"][-300:] if a["ok"] else ""})
        print("[%d/%d] %s idx=%s ok=%s parsed=%s correct=%s wall=%.2f err=%s"
              % (idx, total, ds, sample["idx"], a["ok"], row["parsed"],
                 row["correct"], a["wall_s"], a["error"][:120]), flush=True)
        return row

    total = len(mmlu_sub) + len(gsm_sub)
    # MMLU answers are one letter (short), so host-side concurrency (doc
    # lfm3-bug2-concurrent-smoke Decision) only shortens the smoke window;
    # GSM8K stays serial.  Thread-safety: each sample's "one" is a single
    # curl, join_timing's claimed set is the only shared mutable state.
    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            results.extend(ex.map(lambda it: one("mmlu", it[1], it[0], total),
                                   enumerate(mmlu_sub, 1)))
    else:
        for i, s in enumerate(mmlu_sub, 1):
            results.append(one("mmlu", s, i, total))
    for i, s in enumerate(gsm_sub, 1):
        results.append(one("gsm8k", s, len(mmlu_sub) + i, total))

    repo = args.repo
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                 capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    mmlu_ids = [r["idx"] for r in results if r["dataset"] == "mmlu"]
    gsm_ids = [r["idx"] for r in results if r["dataset"] == "gsm8k"]
    import hashlib
    doc = {
        "arm": args.arm, "seg": args.seg,
        "gateway": args.gateway, "model": args.model,
        "request_log": args.request_log, "harness_commit": commit,
        "subset": {"n_mmlu": args.n_mmlu, "n_gsm8k": args.n_gsm8k,
                   "mmlu_subset_seed": args.mmlu_seed,
                   "gsm8k_subset_seed": args.gsm8k_seed,
                   "mmlu_idx": mmlu_ids, "gsm8k_idx": gsm_ids,
                   "mmlu_idx_sha256": hashlib.sha256(json.dumps(mmlu_ids).encode()).hexdigest(),
                   "gsm8k_idx_sha256": hashlib.sha256(json.dumps(gsm_ids).encode()).hexdigest()},
        "max_tokens_mmlu": args.max_tokens_mmlu,
        "max_tokens_gsm8k": args.max_tokens_gsm8k,
        "max_tokens": args.max_tokens_mmlu, "timeout": args.timeout,
        "n": len(results),
        "ok_count": sum(1 for r in results if r["ok"]),
        "error_count": sum(1 for r in results if not r["ok"]),
        "results": results,
    }
    Path(args.out).write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print("wrote %s n=%d errors=%d" % (args.out, doc["n"], doc["error_count"]))
    return 0 if doc["error_count"] == 0 else 1


# ---- analyzer ----

def discover_segs(res: Path) -> dict[str, list[int]] | None:
    found: dict[str, list[int]] = {}
    for arm in ("native-4bit-A", "native-bf16-B"):
        segs = []
        for p in res.glob(f"{arm}-seg*.json"):
            try:
                segs.append(int(p.stem.split("seg")[-1]))
            except ValueError:
                pass
        found[arm] = sorted(segs)
    if not all(found.values()) or len({len(s) for s in found.values()}) != 1:
        return None
    return found


def load(res: Path, arm: str, seg: int) -> dict | None:
    p = res / f"{arm}-seg{seg}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def seg_accuracy(doc: dict, dataset: str) -> dict:
    rows = [r for r in doc["results"] if r["dataset"] == dataset]
    n = len(rows)
    parsed = [r for r in rows if r.get("parsed_ok")]
    correct = [r for r in parsed if r["correct"]]
    errors = [r for r in rows if not r["ok"]]
    return {"n": n, "parsed": len(parsed), "unparsed": n - len(parsed),
            "correct": len(correct),
            "acc_pct": (100.0 * len(correct) / n if n else 0.0),
            "acc_parsed_pct": (100.0 * len(correct) / len(parsed) if parsed else 0.0),
            "errors": len(errors),
            "idx": [r["idx"] for r in rows],
            "answers": {r["idx"]: r["parsed"] for r in rows},
            "correct_idx": sorted(r["idx"] for r in correct),
            "parsed_idx": sorted(r["idx"] for r in parsed),
            "unparsed_idx": sorted(r["idx"] for r in rows if not r.get("parsed_ok")),
            "error_bodies": [r["error"] for r in errors]}


def aggregate(arm: str, segs: list[int], docs: list[dict], dataset: str) -> dict:
    per_seg = [seg_accuracy(d, dataset) for d in docs]
    # per-arm accuracy = mean of per-segment accuracies (each segment is
    # an independent gateway instance; equal n across segments)
    rates = [s["acc_pct"] for s in per_seg]
    spread = max(rates) - min(rates) if rates else 0.0
    return {"n_per_seg": per_seg[0]["n"] if per_seg else 0,
            "segments": segs, "per_seg": per_seg,
            "acc_pct": (sum(rates) / len(rates) if rates else 0.0),
            "rates_pct": rates, "spread_pp": spread,
            "parsed": sum(s["parsed"] for s in per_seg),
            "unparsed": sum(s["unparsed"] for s in per_seg),
            "correct": sum(s["correct"] for s in per_seg),
            "errors": sum(s["errors"] for s in per_seg),
            "total_n": sum(s["n"] for s in per_seg)}


def flip_rate(a_rows: list[dict], b_rows: list[dict]) -> dict:
    """On items parsed on BOTH arms: fraction whose answer differs.
    Unparsed on either arm excluded from the denominator, counted alone."""
    a = {r["idx"]: r["parsed"] for r in a_rows if r.get("parsed_ok")}
    b = {r["idx"]: r["parsed"] for r in b_rows if r.get("parsed_ok")}
    both = sorted(set(a) & set(b))
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    neither = sorted(set(r["idx"] for r in a_rows) - set(a) - set(b))
    if not both:
        return {"denom": 0, "flips": 0, "flip_pct": 0.0,
                "unparsed_4bit": len(only_a), "unparsed_bf16": len(only_b),
                "unparsed_both": len(neither),
                "flip_idx": [], "unparsed_idx_4bit": only_a,
                "unparsed_idx_bf16": only_b, "unparsed_idx_both": neither}
    flips = [i for i in both if a[i] != b[i]]
    return {"denom": len(both), "flips": len(flips),
            "flip_pct": 100.0 * len(flips) / len(both),
            "unparsed_4bit": len(only_a), "unparsed_bf16": len(only_b),
            "unparsed_both": len(neither),
            "flip_idx": flips,
            "unparsed_idx_4bit": only_a, "unparsed_idx_bf16": only_b,
            "unparsed_idx_both": neither}


def decide(agg4: dict, aggb: dict, n_answers: int) -> tuple[str, str]:
    """Lead 10:52 smoke verdict: difference in correct-question counts over
    the parsed answers of the dataset. Returns (verdict, note)."""
    d = agg4["acc_pct"] - aggb["acc_pct"]
    q = abs(d) * n_answers / 100.0
    q_round = round(q)
    if q_round <= SMOKE_OK_QUESTIONS:
        return "SMOKE_OK", "difference <= %d question(s) (raw %.2f pp / %.2f questions) => direction ok, may scale up" % (
            SMOKE_OK_QUESTIONS, d, q)
    if q_round >= SMOKE_STOP_QUESTIONS:
        return "SMOKE_STOP", "difference >= %d question(s) (raw %.2f pp / %.2f questions) => 4-bit stays opt-in, not primary" % (
            SMOKE_STOP_QUESTIONS, d, q)
    return "SMOKE_GREY", "difference = 2 questions (raw %.2f pp / %.2f) => grey zone, report digits to Lead" % (d, q)


def prereg_reading(delta_mmlu: float, delta_gsm: float) -> str:
    if delta_mmlu >= -PASS_MAX_DELTA_PP and delta_gsm >= -PASS_MAX_DELTA_PP:
        return "PASS"
    if min(delta_mmlu, delta_gsm) > -STOP_DELTA_PP:
        return "PARTIAL"
    return "FAIL"


def run_analyze(args) -> int:
    res = Path(args.res)
    segs = discover_segs(res)
    if segs is None:
        print("SEG discover failed: need >=1 %s-seg*.json per arm with equal counts (got %s)"
              % ("native-4bit-A/native-bf16-B", {p.name for p in res.glob("*-seg*.json")}))
        return 2
    print("SEGS %s" % " ".join("%s=%s" % (a, ",".join(map(str, s))) for a, s in segs.items()))
    docs: dict[str, list[dict]] = {}
    for arm, sl in segs.items():
        docs[arm] = []
        for seg in sl:
            d = load(res, arm, seg)
            if d is None:
                print("SEG arm=%s seg=%d MISSING => window incomplete" % (arm, seg))
                return 2
            docs[arm].append(d)

    # per-segment receipts
    for arm, sl in segs.items():
        for seg, d in zip(sl, docs[arm]):
            m, g = seg_accuracy(d, "mmlu"), seg_accuracy(d, "gsm8k")
            ledger_joined = sum(1 for r in d["results"] if (r.get("timing") or {}).get("engine_request_id"))
            print("SEG arm=%s seg=%d n=%d ok=%d errors=%d mmlu=%d/%d parsed=%d unparsed=%d gsm8k=%d/%d parsed=%d unparsed=%d ledger_joined=%d"
                  % (arm, seg, d["n"], d.get("ok_count", 0), d.get("error_count", 0),
                     m["correct"], m["n"], m["parsed"], m["unparsed"],
                     g["correct"], g["n"], g["parsed"], g["unparsed"], ledger_joined))
            for r in d["results"]:
                if not r["ok"]:
                    print("  ERR arm=%s seg=%d dataset=%s idx=%s body=%s"
                          % (arm, seg, r["dataset"], r["idx"], r["error"][:200]))

    agg = {"native-4bit-A": {}, "native-bf16-B": {}}
    for arm, sl in segs.items():
        for ds in ("mmlu", "gsm8k"):
            agg[arm][ds] = aggregate(arm, sl, docs[arm], ds)
        m, g = agg[arm]["mmlu"], agg[arm]["gsm8k"]
        print("A/A arm=%s dataset=mmlu rates=%s spread_pp=%.2f"
              % (arm, "/".join("%.1f" % r for r in m["rates_pct"]), m["spread_pp"]))
        print("A/A arm=%s dataset=gsm8k rates=%s spread_pp=%.2f"
              % (arm, "/".join("%.1f" % r for r in g["rates_pct"]), g["spread_pp"]))

    # cellset ids consistency across segments
    ids = sorted(agg["native-4bit-A"]["mmlu"]["per_seg"][0]["idx"])
    for arm in agg:
        for s in agg[arm]["mmlu"]["per_seg"]:
            if sorted(s["idx"]) != ids:
                print("CELL_SET_MISMATCH arm=%s dataset=mmlu" % arm)
                return 2
    ids_g = sorted(agg["native-4bit-A"]["gsm8k"]["per_seg"][0]["idx"])
    for arm in agg:
        for s in agg[arm]["gsm8k"]["per_seg"]:
            if sorted(s["idx"]) != ids_g:
                print("CELL_SET_MISMATCH arm=%s dataset=gsm8k" % arm)
                return 2

    flips = {}
    for ds in ("mmlu", "gsm8k"):
        # per-item answer: majority across segments (unparsed if no segment parsed)
        def per_item(arm: str) -> list[dict]:
            merged = {}
            for r in docs[arm][0]["results"]:
                if r["dataset"] == ds:
                    merged[r["idx"]] = {"idx": r["idx"], "parsed": r.get("parsed"),
                                         "parsed_ok": bool(r.get("parsed_ok"))}
            for d in docs[arm][1:]:
                for r in d["results"]:
                    if r["dataset"] == ds and r.get("parsed_ok"):
                        merged[r["idx"]]["parsed"] = r["parsed"]
                        merged[r["idx"]]["parsed_ok"] = True
            return list(merged.values())
        flips[ds] = flip_rate(per_item("native-4bit-A"), per_item("native-bf16-B"))
        f = flips[ds]
        print("FLIP dataset=%s denom=%d flips=%d flip_pct=%.1f%% unparsed_4bit=%d unparsed_bf16=%d unparsed_both=%d"
              % (ds, f["denom"], f["flips"], f["flip_pct"],
                 f["unparsed_4bit"], f["unparsed_bf16"], f["unparsed_both"]))
        if f["flip_idx"]:
            print("FLIP ids dataset=%s %s" % (ds, ",".join(map(str, sorted(f["flip_idx"])))))

    # cellset: per dataset, item-level correctness summed across segments
    cellset = {}
    for ds in ("mmlu", "gsm8k"):
        rows = []
        idset = sorted(agg["native-4bit-A"][ds]["per_seg"][0]["idx"])
        for i in idset:
            row = {"idx": i}
            for arm in agg:
                c = sum(1 for s in agg[arm][ds]["per_seg"] if i in s["correct_idx"])
                row[arm] = c
            rows.append(row)
        cellset[ds] = rows
        print("CELLSET dataset=%s n=%d" % (ds, len(rows)))
        for row in rows:
            print("  CELL idx=%s 4bit_correct_segs=%d/%d bf16_correct_segs=%d/%d"
                  % (row["idx"], row["native-4bit-A"], len(agg["native-4bit-A"][ds]["per_seg"]),
                     row["native-bf16-B"], len(agg["native-bf16-B"][ds]["per_seg"])))

    a_m, a_g = agg["native-4bit-A"], agg["native-bf16-B"]
    n_mmlu = a_m["mmlu"]["n_per_seg"]
    n_gsm = a_g["gsm8k"]["n_per_seg"]
    print("ACC dataset=mmlu 4bit=%.1f%% (%d/%d parsed %d unparsed) bf16=%.1f%% (%d/%d parsed %d unparsed) delta=%.2fpp"
          % (a_m["mmlu"]["acc_pct"], a_m["mmlu"]["correct"], a_m["mmlu"]["n_per_seg"], a_m["mmlu"]["unparsed"],
             a_g["mmlu"]["acc_pct"], a_g["mmlu"]["correct"], a_g["mmlu"]["n_per_seg"], a_g["mmlu"]["unparsed"],
             a_m["mmlu"]["acc_pct"] - a_g["mmlu"]["acc_pct"]))
    print("ACC dataset=gsm8k 4bit=%.1f%% (%d/%d parsed %d unparsed) bf16=%.1f%% (%d/%d parsed %d unparsed) delta=%.2fpp"
          % (a_m["gsm8k"]["acc_pct"], a_m["gsm8k"]["correct"], a_m["gsm8k"]["n_per_seg"], a_m["gsm8k"]["unparsed"],
             a_g["gsm8k"]["acc_pct"], a_g["gsm8k"]["correct"], a_g["gsm8k"]["n_per_seg"], a_g["gsm8k"]["unparsed"],
             a_m["gsm8k"]["acc_pct"] - a_g["gsm8k"]["acc_pct"]))

    # smoke verdict (Lead 10:52)
    v_m, note_m = decide(a_m["mmlu"], a_g["mmlu"], n_mmlu)
    v_g, note_g = decide(a_m["gsm8k"], a_g["gsm8k"], n_gsm)
    # overall = worst dataset verdict
    rank = {"SMOKE_OK": 0, "SMOKE_GREY": 1, "SMOKE_STOP": 2}
    overall = max((v_m, v_g), key=lambda v: rank[v])
    smoke_n = n_mmlu + n_gsm
    print("SMOKE verdict_dataset=mmlu %s (%s)" % (v_m, note_m))
    print("SMOKE verdict_dataset=gsm8k %s (%s)" % (v_g, note_g))
    print("KNOWLEDGE_SMOKE_VERDICT %s (n_mmlu=%d n_gsm8k=%d, smoke, NOT a statistical conclusion)"
          % (overall, n_mmlu, n_gsm))

    # prereg reading (secondary; only meaningful at full scale)
    delta_m = a_m["mmlu"]["acc_pct"] - a_g["mmlu"]["acc_pct"]
    delta_g = a_m["gsm8k"]["acc_pct"] - a_g["gsm8k"]["acc_pct"]
    pre = prereg_reading(delta_m, delta_g)
    print("KNOWLEDGE_PREREG_READING %s (secondary at this scale) delta_mmlu=%.2fpp delta_gsm8k=%.2fpp"
          % (pre, delta_m, delta_g))
    if smoke_n < 50:
        print("CAVEAT scale revision Lead 10:52: n=%d total answers is a smoke test, 1 question = %dpp, "
              "no statistical conclusion; only answers 'is the difference large'."
              % (smoke_n, 100.0 / min(n_mmlu, n_gsm) if min(n_mmlu, n_gsm) else 0.0))

    out = res / "knowledge-verdict.json"
    out.write_text(json.dumps({
        "gate": "knowledge", "verdict": overall,
        "verdict_datasets": {"mmlu": v_m, "gsm8k": v_g},
        "smoke": smoke_n < 50,
        "n_answers": {"mmlu": n_mmlu, "gsm8k": n_gsm},
        "prereg_secondary": {"verdict": pre, "delta_mmlu_pp": round(delta_m, 2),
                             "delta_gsm8k_pp": round(delta_g, 2)},
        "accuracy": {"mmlu": {"4bit": a_m["mmlu"]["acc_pct"], "bf16": a_g["mmlu"]["acc_pct"],
                              "delta_pp": round(delta_m, 2),
                              "4bit_n": a_m["mmlu"]["n_per_seg"], "bf16_n": a_g["mmlu"]["n_per_seg"]},
                     "gsm8k": {"4bit": a_m["gsm8k"]["acc_pct"], "bf16": a_g["gsm8k"]["acc_pct"],
                               "delta_pp": round(delta_g, 2),
                               "4bit_n": a_m["gsm8k"]["n_per_seg"], "bf16_n": a_g["gsm8k"]["n_per_seg"]}},
        "flip_rate": flips,
        "a_a": {arm: {"mmlu": agg[arm]["mmlu"]["spread_pp"], "gsm8k": agg[arm]["gsm8k"]["spread_pp"]}
                for arm in agg},
        "unparsed": {arm: {ds: agg[arm][ds]["unparsed"] for ds in ("mmlu", "gsm8k")} for arm in agg},
        "errors": {arm: {ds: agg[arm][ds]["errors"] for ds in ("mmlu", "gsm8k")} for arm in agg},
        "segments_per_arm": {"4bit": len(segs["native-4bit-A"]), "bf16": len(segs["native-bf16-B"])},
        "subset": docs["native-4bit-A"][0].get("subset"),
        "cellset": cellset,
        "segments": {arm: [{"seg": seg, "mmlu": agg[arm]["mmlu"]["per_seg"][k],
                            "gsm8k": agg[arm]["gsm8k"]["per_seg"][k]}
                           for k, seg in enumerate(segs[arm])] for arm in agg},
    }, indent=2, ensure_ascii=False) + "\n")
    print("wrote %s" % out)
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "segment":
        ap = argparse.ArgumentParser(prog="analyze-knowledge-eval.py segment")
        ap.add_argument("--res", required=True)
        ap.add_argument("--arm", required=True)
        ap.add_argument("--seg", type=int, required=True)
        ap.add_argument("--gateway", required=True)
        ap.add_argument("--model", required=True)
        ap.add_argument("--samples", required=True)
        ap.add_argument("--n-mmlu", type=int, required=True)
        ap.add_argument("--n-gsm8k", type=int, required=True)
        ap.add_argument("--mmlu-seed", type=int, required=True)
        ap.add_argument("--gsm8k-seed", type=int, required=True)
        ap.add_argument("--request-log", default="")
        ap.add_argument("--repo", required=True)
        ap.add_argument("--out", required=True)
        # bug-2 (Lead 2026-11-22 10:52): GSM8K cap 64->256, MMLU stays 64.
        # Kept as a single flag whose value now drives MMLU, and a new
        # GSM8K flag; old receipts with "max_tokens" stay readable.
        ap.add_argument("--max-tokens", type=int, default=64,
                        help="MMLU max tokens (default 64)")
        ap.add_argument("--max-tokens-mmlu", type=int, default=None,
                        help="MMLU max tokens (overrides --max-tokens)")
        ap.add_argument("--max-tokens-gsm8k", type=int, default=256,
                        help="GSM8K max tokens (default 256, bug-2)")
        ap.add_argument("--concurrency", type=int, default=1,
                        help="MMLU segment request concurrency (default 1, "
                             "serial; doc lfm3-bug2-concurrent-smoke)")
        ap.add_argument("--timeout", type=int, default=180)
        args = ap.parse_args(sys.argv[2:])
        args.max_tokens_mmlu = (args.max_tokens_mmlu
                                if args.max_tokens_mmlu is not None
                                else args.max_tokens)
        return run_segment(args)
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        res = sys.argv[2] if len(sys.argv) > 2 else "."
        class _A:
            pass
        a = _A()
        a.res = res
        return run_analyze(a)
    print("usage: analyze-knowledge-eval.py segment <args...> | analyze <res_dir>")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
