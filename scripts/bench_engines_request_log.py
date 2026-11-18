"""Correlate engine-owned request ledgers with benchmark records.

`bench_engines.sh --request-log <path>` exists so that an engine which publishes
its own per-request durations (`prefill_ms`, `decode_ms`, `decode_ms_per_token`,
`prompt_tokens`, `completion_tokens`) can be measured on its own timing basis. The
gateway writes one JSON object per line; every row carries the same `request_id`
that also appears in the response (`chatcmpl-<request_id>` on the OpenAI surface).

Two things used to make that join fragile, and this module fixes both:

1. **The id was matched by string surgery only.** A surface that prefixes the id
   differently, or logs it with different case/whitespace, silently produced zero
   matches; the raw file then showed `engine_source: null` for every request and
   the paired report had an empty column for that arm.
2. **A sparse line could shadow the rich one.** The gateway appends both a
   `{"event": "request_settled", ...}` line and a full ledger row for the same
   request; whichever came last in the file won the old dict build. A sparse
   line carries the id but no durations, so the request looked "correlated" while
   its timings were `None`.

The join therefore works in two passes, both declared in the raw file:

- `engine_ledger_join: "request_id"` — the response id (normalized) matches a
  ledger row directly. Rows that carry durations win over sparse rows.
- `engine_ledger_join: "ordered"` — fallback for a segment whose response ids do
  not match: take the ledger rows inside that invocation's wall-clock window
  (one invocation = one segment), drop rows already claimed by id, skip the
  warm-up requests the harness sent before the recorded ones, then zip in order.

`engine_source` is always `apxinf.ledger` on a successful join; a record whose id
never matches and whose segment has no spare row stays uncorrelated rather than
being filled with a guessed row.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

ENGINE_SOURCE = "apxinf.ledger"

# Prefixes an OpenAI-compatible surface may put in front of the ledger id.
_RESPONSE_PREFIXES = ("chatcmpl-", "cmpl-")


def normalize_request_id(value):
    """Ledger/response id -> comparable key (None-safe)."""
    if not isinstance(value, str):
        return ""
    text = value.strip()
    for prefix in _RESPONSE_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text.strip().lower()


def parse_timestamp(value):
    """Parse the ledger's ISO timestamp (`...Z` or `+00:00`); None when unusable."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp


def load_ledger(path):
    """Ledger rows in file order. Lines without a `request_id` are ignored."""
    rows = []
    if not path:
        return rows
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and normalize_request_id(row.get("request_id")):
                    rows.append(row)
    except OSError:
        return []
    return rows


def _has_durations(row):
    return row.get("prefill_ms") is not None or row.get("decode_ms") is not None


def index_ledger(rows):
    """request_id -> best row: a row carrying durations beats a sparse row."""
    index = {}
    for row in rows:
        key = normalize_request_id(row.get("request_id"))
        if not key:
            continue
        prior = index.get(key)
        if prior is None or (_has_durations(row) and not _has_durations(prior)):
            index[key] = row
    return index


def _apply(record, row, join):
    # Engine-reported durations stay flat next to the client ones: the report
    # reads one schema for every arm and never has to know which surface the
    # number came from.
    record["engine_prefill_ms"] = row.get("prefill_ms")
    record["engine_decode_ms"] = row.get("decode_ms")
    record["engine_decode_ms_per_token"] = row.get("decode_ms_per_token")
    record["engine_decode_tokens"] = row.get("completion_tokens")
    record["engine_ttft_ms"] = row.get("ttft_ms")
    record["engine_ledger_request_id"] = row.get("request_id")
    record["engine_ledger_join"] = join
    record["engine_source"] = ENGINE_SOURCE


def attach_request_log(records, path, window=None, n_warmup=0):
    """Attach engine-reported durations to `records`; returns the match count.

    `window` is the invocation's (started_at, finished_at) ISO pair — one
    invocation drives one segment, so it bounds the ordered fallback. `n_warmup`
    is the number of warm-up requests the harness sent before `records`; their
    ledger rows are the first ones in the window and are skipped by the fallback.
    """
    rows = load_ledger(path)
    if not rows:
        return 0
    index = index_ledger(rows)
    matched = 0
    claimed = set()
    unmatched = []
    for record in records:
        key = normalize_request_id(record.get("response_id"))
        row = index.get(key) if key else None
        if row is None:
            unmatched.append(record)
            continue
        _apply(record, row, "request_id")
        claimed.add(key)
        matched += 1

    if unmatched:
        candidates = rows
        if window:
            start = parse_timestamp(window[0])
            end = parse_timestamp(window[1])
            if start is not None and end is not None:
                in_window = []
                for row in rows:
                    stamp = parse_timestamp(row.get("ts"))
                    if stamp is not None and start <= stamp <= end:
                        in_window.append(row)
                if in_window:
                    candidates = in_window
        spare = [row for row in candidates
                 if normalize_request_id(row.get("request_id")) not in claimed]
        if n_warmup:
            spare = spare[int(n_warmup):]
        for record, row in zip(unmatched, spare):
            _apply(record, row, "ordered")
            matched += 1
    return matched
