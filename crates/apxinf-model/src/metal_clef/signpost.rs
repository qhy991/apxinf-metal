//! Env-gated `os_signpost` operator timeline for the Clef request path.
//!
//! `APXINF_CLEF_SIGNPOST=1` turns the timeline on; unset → [`enabled`] is
//! false and every [`signpost_interval`] call costs one relaxed atomic load
//! plus a branch: no allocation, no string formatting and no FFI (the `$arg`
//! expressions are only evaluated inside the enabled branch). The signposts
//! use the `PointsOfInterest` os_log category, so Instruments' Points of
//! Interest instrument records them next to Metal System Trace's
//! command-buffer lane.
//!
//! The begin/end pairs are emitted by the C shim in `native/signpost_shim.c`:
//! os_signpost's public API is a set of macros over clang's compile-time
//! `__builtin_os_log_format`, so the literal format strings cannot be
//! expanded from Rust. The shim switches on [`REQUEST`]…[`SYNC`] and calls
//! each macro with a literal format, which is exactly what Instruments needs
//! to attribute the recorded arguments.
//!
//! **Attribution boundary.** These intervals are *host* signposts: they
//! bracket the CPU-side graph build, the blocking tile wait and the Core ML
//! call on the request thread. They are not GPU kernel spans. GPU execution
//! appears on Metal System Trace's hardware command-buffer lane, and the
//! `ane_*` intervals are software wall-clock around the Neural Engine calls.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

/// Operator labels on the timeline. The values must match the `enum` in
/// `native/signpost_shim.c`.
pub(crate) const REQUEST: u32 = 0;
pub(crate) const PREPROCESS: u32 = 1;
pub(crate) const ENCODE: u32 = 2;
pub(crate) const VISION: u32 = 3;
pub(crate) const LAYER: u32 = 4;
pub(crate) const HEAD: u32 = 5;
// Only `ane.rs` (feature `coreml`) emits these two; an `mlx`-only build still
// compiles the rest of the operator timeline.
#[cfg_attr(not(feature = "coreml"), allow(dead_code))]
pub(crate) const ANE_WAIT: u32 = 6;
#[cfg_attr(not(feature = "coreml"), allow(dead_code))]
pub(crate) const ANE_PREDICT: u32 = 7;
pub(crate) const SYNC: u32 = 8;

const STATE_UNKNOWN: u8 = 0;
const STATE_OFF: u8 = 1;
const STATE_ON: u8 = 2;
/// 0 = env not read yet, 1 = `APXINF_CLEF_SIGNPOST` off, 2 = on. Once the
/// first call has read the env, every later call is a single relaxed load.
static STATE: AtomicU8 = AtomicU8::new(STATE_UNKNOWN);

extern "C" {
    pub(crate) fn apx_sp_log() -> *mut c_void;
    pub(crate) fn apx_sp_begin(kind: u32, log: *mut c_void, spid: u64, ...);
    pub(crate) fn apx_sp_end(kind: u32, log: *mut c_void, spid: u64);
    fn os_signpost_id_generate(log: *mut c_void) -> u64;
}

/// The shim's log handle, created on first use.
struct Log(*mut c_void);
// SAFETY: `os_log_t` is an immutable, process-wide handle.
unsafe impl Send for Log {}
unsafe impl Sync for Log {}
static LOG: OnceLock<Log> = OnceLock::new();

fn log() -> *mut c_void {
    LOG.get_or_init(|| Log(unsafe { apx_sp_log() })).0
}

/// True when `APXINF_CLEF_SIGNPOST=1`.
pub(crate) fn enabled() -> bool {
    match STATE.load(Ordering::Relaxed) {
        STATE_ON => true,
        STATE_OFF => false,
        _ => {
            let on = std::env::var("APXINF_CLEF_SIGNPOST").map_or(false, |v| v == "1");
            STATE.store(if on { STATE_ON } else { STATE_OFF }, Ordering::Relaxed);
            on
        }
    }
}

/// One open signpost interval. Dropping it emits the matching interval end
/// (including on `?` early returns), so callers bind it with `let _sp = ...`.
pub(crate) struct Span {
    pub(crate) log: *mut c_void,
    pub(crate) spid: u64,
    kind: u32,
}

/// Open an interval for `kind`, or `None` when signposts are off.
pub(crate) fn start(kind: u32) -> Option<Span> {
    if !enabled() {
        return None;
    }
    let log = log();
    let spid = unsafe { os_signpost_id_generate(log) };
    Some(Span { log, spid, kind })
}

impl Drop for Span {
    fn drop(&mut self) {
        unsafe { apx_sp_end(self.kind, self.log, self.spid) }
    }
}

/// Begin an `os_signpost` interval when the env gate is on, and return the
/// guard that closes it on drop.
///
/// ```ignore
/// let _sp = signpost_interval!(signpost::LAYER, index, mixer.as_ptr(), tokens, cached);
/// ```
///
/// `$kind` is one of the constants above (it selects the shim's literal
/// format); the `$arg` expressions are evaluated only when signposts are on
/// and are forwarded to `apx_sp_begin`.
macro_rules! signpost_interval {
    ($kind:expr $(, $arg:expr)* $(,)?) => {{
        let __apx_span = $crate::metal_clef::signpost::start($kind);
        if let Some(__g) = __apx_span.as_ref() {
            unsafe {
                $crate::metal_clef::signpost::apx_sp_begin(
                    $kind,
                    __g.log,
                    __g.spid
                    $(, $arg)*
                )
            }
        }
        __apx_span
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `STATE` and `LOG` are process-global, so each env case runs in its own
    /// process (the same pattern as `trace_json`'s recorder tests).
    const CASE: &str = "APXINF_CLEF_TEST_CASE";

    fn spawn(case: &str, on: bool) -> std::process::Output {
        let exe = std::env::current_exe().expect("test executable");
        let mut cmd = std::process::Command::new(exe);
        cmd.arg("signpost_env_gate").env(CASE, case);
        if on {
            cmd.env("APXINF_CLEF_SIGNPOST", "1");
        } else {
            cmd.env_remove("APXINF_CLEF_SIGNPOST");
        }
        cmd.output().expect("spawn signpost env case")
    }

    fn child_text(out: &std::process::Output) -> String {
        format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    }

    #[test]
    fn signpost_env_gate() {
        match std::env::var(CASE).as_deref() {
            Ok("on") => {
                assert!(enabled(), "APXINF_CLEF_SIGNPOST=1 must enable signposts");
                // Opening a span exercises the shim's log handle + id path;
                // dropping it emits the matching interval end.
                drop(start(REQUEST).expect("enabled span"));
            }
            Ok("off") => {
                assert!(!enabled(), "unset APXINF_CLEF_SIGNPOST must stay off");
                // The disabled path must not touch the shim.
                assert!(start(REQUEST).is_none());
            }
            _ => {
                // Parent: fresh process per case, with a vacuous-child guard.
                for (case, on) in [("on", true), ("off", false)] {
                    let out = spawn(case, on);
                    let text = child_text(&out);
                    assert!(
                        out.status.success(),
                        "signpost child {case} failed:\n{text}"
                    );
                    assert!(
                        text.contains("1 passed") && text.contains("0 failed"),
                        "signpost child {case} ran no assertions:\n{text}"
                    );
                }
            }
        }
    }
}
