//! Isolated CPU subprocess checks for unavailable diagnostic output.
use super::lifecycle_regressions::{cpu_service, cpu_worker, request, CpuModel};
use super::*;
use std::os::unix::process::CommandExt;
use std::process::{Child as ProbeChild, Command as ProbeCommand};
use tokio::io::AsyncReadExt;

const PROBE_MODE: &str = "APXINF_SERVING_DIAGNOSTIC_PROBE";
const PROBE_MODEL: &str = "APXINF_SERVING_DIAGNOSTIC_MODEL";
const COMPLETED_MARKER: &str = "APXINF_DIAGNOSTIC_PROBE_COMPLETED";
const PROGRESS_MARKER: &str = "APXINF_DIAGNOSTIC_PROBE_PROGRESS";
/// A transcript is only used to explain a failure; keep it bounded so a probe
/// that floods stdout cannot grow the parent's memory without limit.
const MAX_TRANSCRIPT_BYTES: usize = 64 * 1024;
/// A probe step waits for a Python fixture, a re-executed test binary, or both.
/// The parent therefore watches for stalled progress instead of enforcing a
/// fixed wall-clock deadline that a loaded host can exceed without a bug.
/// The bound assumes a host at up to roughly four times oversubscription; a CI
/// host that cannot schedule the probe at all for this long should raise it and
/// record the observation instead of treating the stall as a probe bug.
const PROBE_STALL_GRACE: Duration = Duration::from_secs(30);
/// The absolute upper bound only catches a probe that keeps making progress
/// without ever finishing; it must stay well above a loaded full-suite run.
const PROBE_HARD_LIMIT: Duration = Duration::from_secs(120);
/// Probe wait limits, parameterized so both branches are unit-tested.
#[derive(Debug, Clone, Copy)]
struct ProbeLimits {
    stall: Duration,
    hard: Duration,
}

const PROBE_LIMITS: ProbeLimits = ProbeLimits {
    stall: PROBE_STALL_GRACE,
    hard: PROBE_HARD_LIMIT,
};

/// The terminal observation of the probe wait loop.
#[derive(Debug)]
enum ProbeWait {
    Exited(std::process::ExitStatus),
    Stalled(Duration),
    ExceededHardLimit(Duration),
}

fn wait_for_probe(
    child: &mut ProbeChild,
    last_progress: &std::sync::Mutex<Instant>,
    started: Instant,
    limits: ProbeLimits,
) -> ProbeWait {
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return ProbeWait::Exited(status);
        }
        let idle = last_progress.lock().unwrap().elapsed();
        if idle >= limits.stall {
            return ProbeWait::Stalled(idle);
        }
        if started.elapsed() >= limits.hard {
            return ProbeWait::ExceededHardLimit(limits.hard);
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// Keep the diagnostic transcript bounded without splitting a UTF-8 character.
fn append_transcript(transcript: &mut String, chunk: &str) {
    let remaining = MAX_TRANSCRIPT_BYTES.saturating_sub(transcript.len());
    if remaining == 0 {
        return;
    }
    let end = (0..=chunk.len().min(remaining))
        .rev()
        .find(|&index| chunk.is_char_boundary(index))
        .unwrap_or(0);
    transcript.push_str(&chunk[..end]);
}
/// The flood scenario proves that stderr backpressure cannot block stdout, so
/// its waits only need to catch a real deadlock and can tolerate a loaded host.
const FLOOD_LIVENESS_GRACE: Duration = Duration::from_secs(30);
/// Killing an owned process group and observing its exit is load-sensitive.
const GROUP_REAP_GRACE: Duration = Duration::from_secs(15);

fn progress(step: &str) {
    println!("{PROGRESS_MARKER}:{step}");
}

extern "C" {
    fn kill(pid: i32, signal: i32) -> i32;
}

struct OwnedProbe {
    child: ProbeChild,
    group: i32,
}

impl OwnedProbe {
    fn reap_group(&mut self) {
        // process_group(0) gives this child an exclusive, owned process group.
        unsafe { kill(-self.group, 9) };
        let _ = self.child.wait();
    }
}

impl Drop for OwnedProbe {
    fn drop(&mut self) {
        self.reap_group();
    }
}

fn group_has_live_members(group: i32) -> bool {
    let output = ProbeCommand::new("/bin/ps")
        .args(["-axo", "pgid=,stat="])
        .output()
        .expect("The owned process group must be inspectable.");
    assert!(output.status.success());
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .any(|line| {
            let fields: Vec<_> = line.split_whitespace().collect();
            fields.len() == 2
                && fields[0].parse::<i32>() == Ok(group)
                && !fields[1].starts_with('Z')
        })
}

fn run_isolated(output_mode: &str, scenario: &str) {
    let model = CpuModel::new(json!({}));
    let mut command = ProbeCommand::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            "supervisor::diagnostic_regressions::diagnostic_probe_entry",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(PROBE_MODE, format!("{output_mode}:{scenario}"))
        .env(PROBE_MODEL, model.model())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    let child = command.spawn().unwrap();
    let group = child.id() as i32;
    let mut probe = OwnedProbe { child, group };
    let mut output = probe.child.stdout.take().unwrap();
    let error_pipe = probe.child.stderr.take().unwrap();
    // Keep the full-pipe read endpoint open without reading any bytes.
    let _undrained = if output_mode == "closed" {
        drop(error_pipe);
        None
    } else {
        Some(error_pipe)
    };
    // Read the probe's progress while it runs. Every step prints a marker, so
    // the parent can distinguish a loaded host from a blocked probe.
    let transcript = Arc::new(std::sync::Mutex::new(String::new()));
    let last_progress = Arc::new(std::sync::Mutex::new(Instant::now()));
    let reader = {
        let transcript = transcript.clone();
        let last_progress = last_progress.clone();
        std::thread::spawn(move || {
            let mut buffer = [0_u8; 4096];
            loop {
                match output.read(&mut buffer) {
                    Ok(0) | Err(_) => break,
                    Ok(count) => {
                        append_transcript(
                            &mut transcript.lock().unwrap(),
                            &String::from_utf8_lossy(&buffer[..count]),
                        );
                        *last_progress.lock().unwrap() = Instant::now();
                    }
                }
            }
        })
    };
    let started = Instant::now();
    let mut blocked = String::new();
    let status = match wait_for_probe(&mut probe.child, &last_progress, started, PROBE_LIMITS) {
        ProbeWait::Exited(status) => Some(status),
        ProbeWait::Stalled(idle) => {
            blocked = format!("made no progress for {idle:?}");
            None
        }
        ProbeWait::ExceededHardLimit(limit) => {
            blocked = format!("exceeded the {limit:?} hard limit");
            None
        }
    };
    // A failing or blocked probe cannot leave its worker alive.
    probe.reap_group();
    let cleanup_deadline = Instant::now() + GROUP_REAP_GRACE;
    while group_has_live_members(group) && Instant::now() < cleanup_deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        !group_has_live_members(group),
        "An owned probe process remains alive."
    );
    let _ = reader.join();
    let transcript = transcript.lock().unwrap().clone();
    let status = status
        .unwrap_or_else(|| panic!("The {output_mode}/{scenario} probe {blocked}. {transcript}"));
    assert!(
        status.success(),
        "The {output_mode}/{scenario} probe failed: {status}. {transcript}"
    );
    assert!(
        transcript.contains(COMPLETED_MARKER),
        "The probe did not confirm lifecycle completion. {transcript}"
    );
}

fn fill_own_stderr_pipe() {
    // Only this probe and its children share fd 2. Fill it without blocking here.
    let status = ProbeCommand::new("python3")
        .args([
            "-c",
            "import os\nos.set_blocking(2, False)\ntry:\n for size in (4096, 1):\n  try:\n   while True: os.write(2, b'x' * size)\n  except BlockingIOError: pass\nfinally:\n os.set_blocking(2, True)\n",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .status()
        .unwrap();
    assert!(
        status.success(),
        "The isolated stderr pipe must become full."
    );
}

async fn require_completed_request(service: &Service, ticket: &mut Ticket) {
    assert!((&mut ticket.begin).await.unwrap().is_ok());
    loop {
        match ticket.output.recv().await {
            Some(Event::Part(_)) => {}
            Some(Event::Done { .. }) => break,
            _ => panic!("A CPU request must complete while diagnostic output is unavailable."),
        }
    }
    assert!(service.available.load(Ordering::Acquire));
}

async fn probe_lifecycle(scenario: &str, model: PathBuf) {
    if scenario == "flood" {
        probe_worker_stderr_flood().await;
        return;
    }
    let (mut worker, ready) = cpu_worker(model.clone()).await.unwrap();
    progress("worker-started");
    let worker_pid = worker.child.id().unwrap() as i32;
    let mut service = cpu_service(ready, Duration::from_secs(2), 2);
    let (sender, stopped) = watch::channel(None);
    Arc::get_mut(&mut service).unwrap().stopped = stopped;
    if scenario == "rotation" {
        worker.sent_commands = contracts::MAX_COMMANDS - REQUEST_COMMAND_ALLOWANCE + 1;
    }
    let mut first = (scenario != "fault").then(|| service.enqueue(request(false)).unwrap());
    if scenario == "fault" {
        worker.child.start_kill().unwrap();
        progress("fault-injected");
    }
    let replacement_model = model;
    let coordinator = tokio::spawn(coordinate_and_report(
        worker,
        service.clone(),
        1024,
        move || cpu_worker(replacement_model.clone()),
        sender,
    ));
    if scenario == "fault" {
        // Unavailable diagnostic output must not prevent the supervisor from
        // fencing and restarting the faulted worker.
        tokio::time::timeout(Duration::from_secs(3), async {
            while service.metrics.worker_restarts.load(Ordering::Relaxed) == 0 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("A checked worker fault must restart the worker.");
        progress("worker-restarted");
        assert!(service.available.load(Ordering::Acquire));
        assert_eq!(service.metrics.worker_faults.load(Ordering::Relaxed), 1);
        assert_eq!(service.metrics.worker_restarts.load(Ordering::Relaxed), 1);
        assert_eq!(
            service.metrics.worker_restart_failures.load(Ordering::Relaxed),
            0
        );
        service.shutdown();
        service.wait_stopped().await.unwrap();
    } else {
        require_completed_request(&service, first.as_mut().unwrap()).await;
        progress("request-settled");
        let mut second = service.enqueue(request(false)).unwrap();
        require_completed_request(&service, &mut second).await;
        progress("second-request-settled");
        assert_eq!(service.metrics.completed.load(Ordering::Relaxed), 2);
        assert_eq!(service.metrics.worker_faults.load(Ordering::Relaxed), 0);
        if scenario == "rotation" {
            assert_eq!(service.metrics.worker_rotations.load(Ordering::Relaxed), 1);
        }
        progress("shutdown");
        service.shutdown();
        service.wait_stopped().await.unwrap();
        progress("stopped");
    }
    coordinator.await.unwrap();
    assert_eq!(service.metrics.active.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 0);
    // wait_stopped must confirm process exit, not just abandon a worker handle.
    assert_eq!(unsafe { kill(worker_pid, 0) }, -1);
    println!("{COMPLETED_MARKER}");
}

async fn probe_worker_stderr_flood() {
    progress("flood-start");
    let script = "import os\nremaining = memoryview(b'\\xff\\x00A' * 350000)\nwhile remaining:\n written = os.write(2, remaining)\n remaining = remaining[written:]\nos.write(1, b'worker stdout completed\\n')\n";
    let mut child = Command::new("python3")
        .args(["-c", script])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let pid = child.id().unwrap() as i32;
    let diagnostics = crate::diagnostics::WorkerDiagnostics::start(
        child.stderr.take().unwrap(),
        uuid::Uuid::new_v4().to_string(),
    );
    let mut stdout = child.stdout.take().unwrap();
    let mut output = Vec::new();
    tokio::time::timeout(FLOOD_LIVENESS_GRACE, stdout.read_to_end(&mut output))
        .await
        .expect("Worker stderr must not block stdout completion.")
        .unwrap();
    progress("flood-stdout");
    assert_eq!(output, b"worker stdout completed\n");
    assert!(tokio::time::timeout(FLOOD_LIVENESS_GRACE, child.wait())
        .await
        .expect("The flood worker must exit.")
        .unwrap()
        .success());
    assert_eq!(unsafe { kill(pid, 0) }, -1);
    progress("flood-finished");
    drop(diagnostics);
    tokio::time::timeout(FLOOD_LIVENESS_GRACE, crate::diagnostics::finish())
        .await
        .expect("Diagnostic delivery must have a bounded completion interval.");
    progress("flood-drained");
    let mut metrics = String::new();
    crate::diagnostics::append_metrics(&mut metrics);
    let dropped: u64 = metrics
        .lines()
        .filter(|line| line.starts_with("apxinf_diagnostic_dropped_total{"))
        .map(|line| line.rsplit_once(' ').unwrap().1.parse::<u64>().unwrap())
        .sum();
    assert!(
        dropped > 0,
        "Unavailable output must discard records instead of blocking."
    );
    println!("{COMPLETED_MARKER}");
}

#[test]
fn diagnostic_probe_entry() {
    let Ok(mode) = std::env::var(PROBE_MODE) else {
        return;
    };
    std::panic::set_hook(Box::new(|information| {
        use std::io::Write;
        let mut output = std::io::stdout().lock();
        let _ = writeln!(output, "APXINF_DIAGNOSTIC_PROBE_PANIC: {information}");
        let _ = output.flush();
    }));
    let (output_mode, scenario) = mode.split_once(':').unwrap();
    progress("entry");
    if output_mode == "full" {
        fill_own_stderr_pipe();
        progress("stderr-filled");
    }
    let model = PathBuf::from(std::env::var_os(PROBE_MODEL).unwrap());
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(probe_lifecycle(scenario, model));
}

#[test]
fn diagnostic_closed_stderr_does_not_prevent_worker_fencing() {
    run_isolated("closed", "fault");
}

#[test]
fn diagnostic_full_stderr_does_not_prevent_worker_fencing() {
    run_isolated("full", "fault");
}

#[test]
fn diagnostic_closed_stderr_preserves_settlement_and_following_requests() {
    run_isolated("closed", "settlement");
}

#[test]
fn diagnostic_full_stderr_preserves_settlement_and_following_requests() {
    run_isolated("full", "settlement");
}

#[test]
fn diagnostic_closed_stderr_preserves_rotation_and_replacement_requests() {
    run_isolated("closed", "rotation");
}

#[test]
fn diagnostic_full_stderr_preserves_rotation_and_replacement_requests() {
    run_isolated("full", "rotation");
}

#[test]
fn diagnostic_closed_stderr_drains_unterminated_invalid_utf8_worker_output() {
    run_isolated("closed", "flood");
}

#[test]
fn diagnostic_full_stderr_drains_unterminated_invalid_utf8_worker_output() {
    run_isolated("full", "flood");
}

#[test]
fn a_probe_that_stops_making_progress_is_detected() {
    let mut child = ProbeCommand::new("/bin/sh")
        .args(["-c", "sleep 30"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let last_progress = std::sync::Mutex::new(Instant::now());
    let outcome = wait_for_probe(
        &mut child,
        &last_progress,
        Instant::now(),
        ProbeLimits {
            stall: Duration::from_millis(50),
            hard: Duration::from_secs(5),
        },
    );
    assert!(
        matches!(outcome, ProbeWait::Stalled(idle) if idle >= Duration::from_millis(50)),
        "{outcome:?}"
    );
    let _ = child.kill();
    let _ = child.wait();
}

#[test]
fn a_probe_that_keeps_progressing_hits_the_hard_limit() {
    let mut child = ProbeCommand::new("/bin/sh")
        .args(["-c", "sleep 30"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let last_progress = Arc::new(std::sync::Mutex::new(Instant::now()));
    let running = Arc::new(AtomicBool::new(true));
    let keeper = {
        let last_progress = last_progress.clone();
        let running = running.clone();
        std::thread::spawn(move || {
            while running.load(Ordering::Relaxed) {
                *last_progress.lock().unwrap() = Instant::now();
                std::thread::sleep(Duration::from_millis(5));
            }
        })
    };
    let outcome = wait_for_probe(
        &mut child,
        &last_progress,
        Instant::now(),
        ProbeLimits {
            stall: Duration::from_secs(5),
            hard: Duration::from_millis(100),
        },
    );
    running.store(false, Ordering::Relaxed);
    keeper.join().unwrap();
    assert!(
        matches!(outcome, ProbeWait::ExceededHardLimit(limit) if limit == Duration::from_millis(100)),
        "{outcome:?}"
    );
    let _ = child.kill();
    let _ = child.wait();
}

#[test]
fn an_exited_probe_reports_its_exit_status() {
    let mut child = ProbeCommand::new("/bin/sh")
        .args(["-c", "exit 3"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let last_progress = std::sync::Mutex::new(Instant::now());
    let outcome = wait_for_probe(
        &mut child,
        &last_progress,
        Instant::now(),
        ProbeLimits {
            stall: Duration::from_secs(5),
            hard: Duration::from_secs(5),
        },
    );
    match outcome {
        ProbeWait::Exited(status) => assert_eq!(status.code(), Some(3), "{status:?}"),
        other => panic!("An exited probe must report its status: {other:?}"),
    }
}

#[test]
fn a_probe_transcript_stays_bounded_without_splitting_utf8() {
    let mut transcript = String::new();
    append_transcript(&mut transcript, &"a".repeat(MAX_TRANSCRIPT_BYTES + 4096));
    assert_eq!(transcript.len(), MAX_TRANSCRIPT_BYTES);
    append_transcript(&mut transcript, "the failure marker must not append");
    assert_eq!(transcript.len(), MAX_TRANSCRIPT_BYTES);
    // One byte short of the cap, a two-byte character cannot be appended.
    let mut near_cap = "b".repeat(MAX_TRANSCRIPT_BYTES - 1);
    append_transcript(&mut near_cap, "é");
    assert_eq!(near_cap.len(), MAX_TRANSCRIPT_BYTES - 1);
    assert!(near_cap.is_char_boundary(near_cap.len()));
    let mut small = String::new();
    append_transcript(&mut small, "APXINF_MARKER");
    assert_eq!(small, "APXINF_MARKER");
}
