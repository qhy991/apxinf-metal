#!/usr/bin/env python3
"""Unit tests for the --request-log correlation used by scripts/bench_engines.sh.

Run with:  python3 scripts/test_bench_engines_request_log.py
The fixture ledger is written to a temporary file; nothing touches a real host.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bench_engines_request_log as ledger  # noqa: E402


class NormalizeTest(unittest.TestCase):
    def test_response_prefix_and_whitespace_are_normalized(self):
        self.assertEqual(ledger.normalize_request_id("chatcmpl-ABC-123 "), "abc-123")
        self.assertEqual(ledger.normalize_request_id("cmpl-abc"), "abc")
        self.assertEqual(ledger.normalize_request_id("abc"), "abc")
        self.assertEqual(ledger.normalize_request_id(None), "")

    def test_timestamp_accepts_zulu_and_offset(self):
        self.assertIsNotNone(ledger.parse_timestamp("2026-11-18T19:50:41.430Z"))
        self.assertIsNotNone(ledger.parse_timestamp("2026-11-18T19:50:41+08:00"))
        self.assertIsNone(ledger.parse_timestamp("not-a-time"))
        self.assertIsNone(ledger.parse_timestamp(None))


class AttachTest(unittest.TestCase):
    def _write(self, rows):
        handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        with handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        self.addCleanup(lambda: os.unlink(handle.name))
        return handle.name

    def test_request_id_join_and_rich_row_beats_sparse(self):
        path = self._write([
            {"request_id": "r1", "kind": "chat", "prefill_ms": 143.0, "decode_ms": 259.5,
             "decode_ms_per_token": 37.07, "prompt_tokens": 106, "completion_tokens": 8,
             "ts": "2026-11-18T19:50:41.000Z"},
            # the sparse settle line arrives later: it must NOT shadow the row above
            {"event": "request_settled", "request_id": "r1", "status": "completed"},
        ])
        records = [{"response_id": "chatcmpl-r1", "prompt_tokens": 106, "completion_tokens": 8}]
        self.assertEqual(ledger.attach_request_log(records, path), 1)
        record = records[0]
        self.assertEqual(record["engine_source"], "apxinf.ledger")
        self.assertEqual(record["engine_ledger_join"], "request_id")
        self.assertEqual(record["engine_prefill_ms"], 143.0)
        self.assertEqual(record["engine_decode_ms"], 259.5)
        self.assertEqual(record["engine_decode_tokens"], 8)
        self.assertEqual(record["engine_ledger_request_id"], "r1")

    def test_ordered_fallback_skips_warmup_rows(self):
        path = self._write([
            {"request_id": "warm", "prefill_ms": 450.0, "decode_ms": 261.0, "completion_tokens": 8,
             "ts": "2026-11-18T19:50:39.000Z"},
            {"request_id": "a", "prefill_ms": 143.0, "decode_ms": 259.0, "completion_tokens": 8,
             "ts": "2026-11-18T19:50:41.000Z"},
            {"request_id": "b", "prefill_ms": 432.0, "decode_ms": 259.0, "completion_tokens": 8,
             "ts": "2026-11-18T19:50:42.000Z"},
        ])
        # response ids that the ledger does not contain -> ordered fallback
        records = [
            {"response_id": "chatcmpl-unknown-one", "prompt_tokens": 106, "completion_tokens": 8},
            {"response_id": "chatcmpl-unknown-two", "prompt_tokens": 401, "completion_tokens": 8},
        ]
        matched = ledger.attach_request_log(
            records, path,
            window=("2026-11-18T19:50:38.000Z", "2026-11-18T19:50:59.000Z"),
            n_warmup=1)
        self.assertEqual(matched, 2)
        self.assertEqual([r["engine_ledger_join"] for r in records], ["ordered", "ordered"])
        self.assertEqual([r["engine_prefill_ms"] for r in records], [143.0, 432.0])
        self.assertEqual([r["engine_ledger_request_id"] for r in records], ["a", "b"])

    def test_mixed_join_does_not_double_use_a_row(self):
        path = self._write([
            {"request_id": "x1", "prefill_ms": 100.0, "decode_ms": 10.0, "ts": "2026-11-18T19:50:40.000Z"},
            {"request_id": "x2", "prefill_ms": 200.0, "decode_ms": 20.0, "ts": "2026-11-18T19:50:41.000Z"},
        ])
        records = [
            {"response_id": "chatcmpl-x1"},          # id match
            {"response_id": "chatcmpl-mystery"},     # fallback must take x2, not x1 again
        ]
        matched = ledger.attach_request_log(
            records, path,
            window=("2026-11-18T19:50:39.000Z", "2026-11-18T19:50:59.000Z"))
        self.assertEqual(matched, 2)
        self.assertEqual(records[0]["engine_ledger_request_id"], "x1")
        self.assertEqual(records[1]["engine_ledger_request_id"], "x2")

    def test_unmatched_record_stays_uncorrelated(self):
        path = self._write([
            {"request_id": "only", "prefill_ms": 1.0, "ts": "2026-11-18T19:50:40.000Z"},
        ])
        records = [
            {"response_id": "chatcmpl-only"},
            {"response_id": "chatcmpl-nope"},
        ]
        self.assertEqual(ledger.attach_request_log(records, path), 1)
        self.assertEqual(records[0]["engine_source"], "apxinf.ledger")
        self.assertNotIn("engine_source", records[1])

    def test_missing_file_is_a_noop(self):
        records = [{"response_id": "chatcmpl-x"}]
        self.assertEqual(ledger.attach_request_log(records, "/nonexistent/ledger.jsonl"), 0)
        self.assertNotIn("engine_source", records[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
