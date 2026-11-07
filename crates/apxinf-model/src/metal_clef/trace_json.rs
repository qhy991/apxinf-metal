//! Env-gated Chrome Tracing JSON recorder (the ApxInf analogue of a local
//! nsys timeline, see DECISIONS D16 / QUEUE Q22).
//!
//! `APXINF_CLEF_CHROME_TRACE=<path>` turns on the recorder; unset →
//! [`enabled`] returns false and every [`record`] call is a `OnceLock` check
//! with no allocation (the demo and all default paths pay nothing). The output
//! is the [Chrome Trace Event Format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU)
//! v1 JSON, viewable in Perfetto (`ui.perfetto.dev`), `chrome://tracing` or
//! `about:tracing`.
//!
//! Timestamps are microseconds since a process-global epoch captured once at
//! first use (any non-negative base is valid for the format). Events
//! accumulate across requests (D17: one run-wide timeline, so Q21/V0 analysis
//! can compute per-request medians rather than keep only the last request).
//! [`flush_request`] writes `pending + "]}"` — `pending` itself never holds
//! the closing braces, so it is never chopped and the file stays parseable.

use serde_json::{json, Value};
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

struct Recorder {
    path: PathBuf,
    epoch: Instant,
    /// `{"traceEvents":[` + every event recorded so far (comma-separated),
    /// without the closing `]}`. The `usize` is the byte length of the event
    /// list already written to the file: durable flushes append only
    /// `pending[written..]` (overwrite the previous `]}` with `,<tail>]}`),
    /// so each flush is O(new events) rather than rewriting the whole
    /// timeline. Both fields are only touched while the mutex is held.
    pending: Mutex<(String, usize)>,
    /// When `APXINF_CLEF_TRACE_DURABLE_EVENTS=1`, `pending` is drained into
    /// the file on every flush (instead of being rewritten whole) so events
    /// recorded after a previous flush — the lazy graph build queued between
    /// sync points — stay on the timeline without O(N²) write amplification.
    durable: bool,
}

static RECORDER: OnceLock<Option<Recorder>> = OnceLock::new();

fn recorder() -> Option<&'static Recorder> {
    RECORDER.get_or_init(|| {
        let path = std::env::var_os("APXINF_CLEF_CHROME_TRACE")?;
        if path.is_empty() {
            return None;
        }
        // `APXINF_CLEF_TRACE_DURABLE_EVENTS=1` drains `pending` into the file
        // on every flush (append) instead of rewriting the whole timeline, so
        // events recorded after a previous flush — the per-layer lazy-graph
        // spans, which are queued between sync points — still survive.
        let durable = std::env::var_os("APXINF_CLEF_TRACE_DURABLE_EVENTS")
            .map(|v| v == "1")
            .unwrap_or(false);
        Some(Recorder {
            path: PathBuf::from(path),
            epoch: Instant::now(),
            pending: Mutex::new((String::from("{\"traceEvents\":["), HEADER.len())),
            durable,
        })
    })
    .as_ref()
}

/// True when `APXINF_CLEF_CHROME_TRACE` is set to a non-empty path.
pub fn enabled() -> bool {
    recorder().is_some()
}

fn event(
    r: &Recorder,
    name: &str,
    lane: &str,
    tid: &str,
    ts_us: f64,
    dur_us: Option<f64>,
    args: Value,
) {
    let obj = json!({
        "name": name,
        "cat": lane,
        "ph": "X",
        "pid": lane,
        "tid": tid,
        "ts": ts_us,
        "dur": dur_us.unwrap_or(0.0),
        "args": args,
    });
    let text = obj.to_string();
    let mut pending = r.pending.lock().unwrap_or_else(|e| e.into_inner());
    if pending.0.len() > HEADER.len() {
        pending.0.push(',');
    }
    pending.0.push_str(&text);
}

const HEADER: &str = "{\"traceEvents\":[";

/// Record a complete slice (microseconds since the process epoch) on the given
/// lane. `args` is merged into the event's `args`. No-op when disabled.
pub fn record(name: &str, lane: &str, tid: &str, ts_us: f64, dur_us: f64, args: Value) {
    let Some(r) = recorder() else { return };
    event(r, name, lane, tid, ts_us, Some(dur_us), args);
}

/// Record an instant marker (e.g. the ANE predict return stamp, or an ANE
/// service gap observation) at `ts_us` microseconds since the epoch.
pub fn instant(name: &str, lane: &str, tid: &str, ts_us: f64, args: Value) {
    let Some(r) = recorder() else { return };
    event(r, name, lane, tid, ts_us, None, args);
}

/// Convert an [`Instant`] to microseconds since the recorder's epoch. Returns
/// `None` when the recorder is off (callers should then skip recording).
pub fn to_us(t: Instant) -> Option<f64> {
    let r = recorder()?;
    Some(t.saturating_duration_since(r.epoch).as_secs_f64() * 1_000_000.0)
}

/// Write the accumulated events to `APXINF_CLEF_CHROME_TRACE` as a parseable
/// Chrome Tracing JSON document containing **every request so far** (one
/// run-wide timeline — needed for per-request medians in Q21/V0 analysis;
/// Perfetto selects the window). Idempotent; safe to call on every request
/// boundary. Errors are swallowed (the recorder must never fail a request).
///
/// Durable mode is append-only: the file is opened without `truncate`, the
/// cursor seeks to just past the header, and only `pending[written..]` (the
/// events since the previous flush, preceded by the comma that replaces the
/// previous `]`) is written, then `]}`. That replaces the old buggy approach
/// — writing `pending + "]}"` to a `body` copy while `pending` never held
/// `"]}"`, then `truncate(len - 2)` chopping the last event's closing braces
/// — which corrupted the JSON after the first flush (D31; reproduced by the
/// regression test below and by a 2-request device run).
pub fn flush_request() {
    let Some(r) = recorder() else { return };
    let mut pending = r.pending.lock().unwrap_or_else(|e| e.into_inner());
    if pending.0.len() <= HEADER.len() {
        return;
    }
    let written = pending.1;
    if r.durable && written > HEADER.len() {
        // Events already in the file: `file[written]` is the `]` of the
        // previous flush. `pending.0[written..]` is exactly the suffix added
        // since (each new event was prefixed with its own `,` by `event()`),
        // so overwrite `]` with `<suffix>]}` — no extra comma. `pending`
        // keeps everything (so the next flush's slice is correct), and only
        // the tail is rewritten — O(new events), not O(all events).
        let tail = format!("{}]}}", &pending.0[written..]);
        if let Ok(mut f) = OpenOptions::new().create(true).write(true).open(&r.path) {
            let _ = f.seek(SeekFrom::Start(written as u64));
            let _ = f.write_all(tail.as_bytes());
            let _ = f.set_len(written as u64 + tail.len() as u64);
            let _ = f.flush();
        }
        pending.1 = pending.0.len();
    } else {
        // First flush (or non-durable: rewrite the whole document so the file
        // is always the complete run-wide timeline).
        let mut body = String::with_capacity(pending.0.len() + 2);
        body.push_str(&pending.0);
        body.push_str("]}");
        if let Ok(mut f) = OpenOptions::new().create(true).truncate(true).write(true).open(&r.path) {
            let _ = f.write_all(body.as_bytes());
            let _ = f.flush();
        }
        pending.1 = pending.0.len();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `RECORDER` is a process-global `OnceLock`, so each env case must run in
    /// its own process. Parent tests spawn a child with `APXINF_CLEF_TEST_CASE`
    /// set to select which assertions to run; the child only runs that case.
    const CASE: &str = "APXINF_CLEF_TEST_CASE";

    fn spawn_case(case: &str, with_trace: bool, durable: bool) -> std::process::Output {
        let exe = std::env::current_exe().unwrap();
        let mut cmd = std::process::Command::new(exe);
        // Filter to this test only: otherwise the child runs the whole suite
        // (including parent branches that spawn further children). Match by
        // the unique bare function name, NOT `--exact`: libtest's exact match
        // needs the full module path, so `--exact chrome_trace_...` matched
        // zero tests and every child exited 0 vacuously — the regression
        // assertions below never ran (QUEUE's "0 filtered ≠ PASS" trap, hit
        // inside the test itself). The guard in `assert_child_ok` below
        // locks this in: a child that ran 0 tests is a failure.
        cmd.arg("chrome_trace_emits_valid_json_and_flushes_per_request")
            .env(CASE, case);
        if with_trace {
            cmd.env("APXINF_CLEF_CHROME_TRACE", "/tmp/apxinf-chrome-trace-test.json");
        } else {
            cmd.env_remove("APXINF_CLEF_CHROME_TRACE");
        }
        if durable {
            cmd.env("APXINF_CLEF_TRACE_DURABLE_EVENTS", "1");
        } else {
            cmd.env_remove("APXINF_CLEF_TRACE_DURABLE_EVENTS");
        }
        cmd.output().expect("spawn test case")
    }

    fn assert_child_ok(out: &std::process::Output) {
        let stderr = String::from_utf8_lossy(&out.stderr);
        // Child assertion output (libtest) goes to stdout; include both.
        let combined = format!("{stderr}{}", String::from_utf8_lossy(&out.stdout));
        assert!(
            out.status.success(),
            "child failed (exit {:?}):\n{}",
            out.status.code(),
            combined
        );
        // Each child runs exactly this one test. A vacuous child (filter
        // matched nothing) also exits 0 with "0 passed" — that is how the
        // assertion set below was silently skipped. Treat 0 tests as failure.
        assert!(
            combined.contains("1 passed") && combined.contains("0 failed"),
            "child ran no assertions (vacuous pass): {combined}"
        );
    }

    #[test]
    fn chrome_trace_emits_valid_json_and_flushes_per_request() {
        match std::env::var(CASE).as_deref() {
            Ok("enabled") => {
                // Child: RECORDER initializes from this process's env.
                assert!(enabled());
                let t0 = Instant::now();
                let ts = to_us(t0).unwrap();
                record("preprocess", "host", "cpu", ts, 1234.0, json!({"rows": 480}));
                instant("ane_gap", "ane", "ane", ts + 2000.0, json!({"gap_ms": 64.0}));
                flush_request();
                let text = std::fs::read_to_string("/tmp/apxinf-chrome-trace-test.json").unwrap();
                let v: Value = serde_json::from_str(&text).expect("valid JSON");
                let events = v["traceEvents"].as_array().unwrap();
                assert_eq!(events.len(), 2);
                assert_eq!(events[0]["name"], "preprocess");
                assert_eq!(events[0]["ph"], "X");
                assert_eq!(events[0]["pid"], "host");
                assert_eq!(events[0]["tid"], "cpu");
                assert_eq!(events[0]["dur"], 1234.0);
                assert_eq!(events[1]["name"], "ane_gap");
                // `to_us` is relative to the process-global epoch, so assert
                // the offset, not an absolute value (the recorder epoch is
                // captured once at first use — in the recorder() call made
                // above — so ts = epoch_offset + 2000.0).
                let offset = events[0]["ts"].as_f64().unwrap() - 0.0;
                assert!(
                    (events[1]["ts"].as_f64().unwrap() - offset - 2000.0).abs() < 1e-6,
                    "ts must be offset+2000 relative to the recorder epoch"
                );
                // A second flush (next request) still yields valid JSON and
                // keeps accumulating: the file holds every request so far
                // (run-wide timeline for per-request medians).
                record("head", "host", "cpu", ts + 5000.0, 10.0, json!({}));
                flush_request();
                let v2: Value = serde_json::from_str(
                    &std::fs::read_to_string("/tmp/apxinf-chrome-trace-test.json").unwrap(),
                )
                .expect("valid JSON after second flush");
                let ev2 = v2["traceEvents"].as_array().unwrap();
                assert_eq!(ev2.len(), 3, "events accumulate across requests");
                assert_eq!(ev2[2]["name"], "head");
            }
            Ok("durable") => {
                // Child: durable mode drains pending on every flush (append),
                // so events recorded after a previous flush — the lazy-graph
                // layer spans queued between sync points — still land on the
                // timeline.
                assert!(enabled());
                let ts = to_us(Instant::now()).unwrap();
                record("chunk1", "chunk1", "rest", ts, 500.0, json!({"rest_tokens": 8}));
                flush_request();
                // Simulate an event recorded after the previous flush (a lazy
                // layer span queued between sync points).
                record("layer", "layer_0", "gated_delta", ts + 100.0, 40.0, json!({"index": 0}));
                flush_request();
                // A third flush with no new events must be a no-op: the file
                // must not shrink or gain a duplicate of earlier events
                // (regression guard for the append-only durable design).
                flush_request();
                let text = std::fs::read_to_string("/tmp/apxinf-chrome-trace-test.json").unwrap();
                let v: Value = serde_json::from_str(&text).expect("valid JSON");
                let events = v["traceEvents"].as_array().unwrap();
                assert_eq!(events.len(), 2, "events recorded between flushes survive");
                assert_eq!(events[0]["name"], "chunk1");
                assert_eq!(events[1]["name"], "layer");
            }
            Ok("disabled") => {
                // Child: no env → recorder is off and calls are null checks.
                assert!(!enabled());
                record("x", "host", "t", 0.0, 1.0, json!({}));
                instant("i", "ane", "t", 1.0, json!({}));
                assert!(to_us(Instant::now()).is_none());
                flush_request();
            }
            _ => {
                // Parent: run each case in a fresh process.
                assert_child_ok(&spawn_case("enabled", true, false));
                assert_child_ok(&spawn_case("durable", true, true));
                assert_child_ok(&spawn_case("disabled", false, false));
            }
        }
    }
}
