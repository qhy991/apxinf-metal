//! Host-interference telemetry attached to measured requests.
//!
//! A normal (unprivileged) process on macOS is not told its thermal level:
//! `pmset -g therm` reports only "no level recorded" on Apple Silicon and
//! `powermetrics` needs root. Rather than invent a level, this reports the load
//! average — the signal that actually identified a disqualified timing window on
//! the shared development hosts — together with an explicit availability flag.

use serde_json::{json, Value};
use std::{
    sync::Mutex,
    time::{Duration, Instant},
};

const CACHE: Duration = Duration::from_secs(1);

static LAST: Mutex<Option<(Instant, Value)>> = Mutex::new(None);

/// Load average and its availability, cached for one second.
pub fn state() -> Value {
    if let Ok(guard) = LAST.lock() {
        if let Some((at, value)) = guard.as_ref() {
            if at.elapsed() < CACHE {
                return value.clone();
            }
        }
    }
    let value = read();
    if let Ok(mut guard) = LAST.lock() {
        *guard = Some((Instant::now(), value.clone()));
    }
    value
}

fn read() -> Value {
    // Absolute path first: a worker process runs with a cleared environment whose
    // PATH does not include /usr/sbin.
    let output = ["/usr/sbin/sysctl", "/sbin/sysctl", "sysctl"]
        .iter()
        .find_map(|program| {
            std::process::Command::new(program)
                .args(["-n", "vm.loadavg"])
                .output()
                .ok()
        });
    let parsed = output.and_then(|output| {
        let text = String::from_utf8_lossy(&output.stdout);
        let numbers: Vec<f64> = text
            .trim()
            .trim_matches(|c: char| !c.is_ascii_digit() && c != '.' && c != ' ')
            .split_whitespace()
            .filter_map(|part| part.parse::<f64>().ok())
            .collect();
        (numbers.len() >= 3).then(|| (numbers[0], numbers[1], numbers[2]))
    });
    match parsed {
        Some((one, five, fifteen)) => json!({
            "source": "sysctl vm.loadavg",
            "thermal_state": "unavailable",
            "load_avg_1m": one,
            "load_avg_5m": five,
            "load_avg_15m": fifteen,
        }),
        None => json!({
            "source": "none",
            "thermal_state": "unavailable",
            "load_avg_1m": Value::Null,
            "load_avg_5m": Value::Null,
            "load_avg_15m": Value::Null,
        }),
    }
}
