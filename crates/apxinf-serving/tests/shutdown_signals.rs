#![cfg(unix)]
use apxinf_serving::{host_pressure::HostPressurePolicy, supervisor::Config};
use std::{
    fs,
    io::{Read, Write},
    net::TcpStream,
    process::{Child, Command, ExitStatus, Stdio},
    thread,
    time::{Duration, Instant},
};
#[path = "support/signal_fixture.rs"]
mod signal_fixture;
use signal_fixture::Fixture;

struct Cli {
    child: Child,
    fixture: Fixture,
    diagnostics_file: Option<std::path::PathBuf>,
}

fn cli_command(fixture: &Fixture, extra: &[&str]) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_apxinf-serve"));
    command
        .arg("--python")
        .arg(&fixture.config.python)
        .arg("--worker")
        .arg(&fixture.config.worker)
        .arg("--model")
        .arg(&fixture.config.model)
        .args([
            "--host-pressure-policy",
            "disabled",
            "--port",
            "0",
            "--max-context",
            "64",
            "--max-output-tokens",
            "16",
            "--memory-budget-bytes",
            "4096",
            "--sequence-reservation-bytes",
            "1024",
            "--timeout-seconds",
            "5",
        ])
        .args(extra);
    command
}

impl Cli {
    fn start(mode: &str) -> Self {
        Self::start_extra(mode, &[])
    }

    fn start_extra(mode: &str, extra: &[&str]) -> Self {
        let fixture = Fixture::new(mode);
        let log = fs::File::create(fixture.root.join("cli.log")).unwrap();
        let child = cli_command(&fixture, extra)
            .stdout(Stdio::null())
            .stderr(log)
            .spawn()
            .unwrap();
        Self {
            child,
            fixture,
            diagnostics_file: None,
        }
    }

    fn start_with_diagnostics(mode: &str) -> Self {
        let fixture = Fixture::new(mode);
        let path = fixture.root.join("diagnostics.jsonl");
        let file = path.to_str().unwrap().to_owned();
        let log = fs::File::create(fixture.root.join("cli.log")).unwrap();
        let child = cli_command(&fixture, &["--diagnostics-file", &file])
            .stdout(Stdio::null())
            .stderr(log)
            .spawn()
            .unwrap();
        Self {
            child,
            fixture,
            diagnostics_file: Some(path),
        }
    }

    fn until(&mut self, mut ready: impl FnMut(&Fixture) -> bool) {
        let until = Instant::now() + Duration::from_secs(5);
        while !ready(&self.fixture) {
            assert!(
                self.child.try_wait().unwrap().is_none(),
                "CLI exited early: {}",
                self.log()
            );
            assert!(
                Instant::now() < until,
                "CLI did not reach the test boundary: {}",
                self.log()
            );
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn log(&self) -> String {
        fs::read_to_string(self.fixture.root.join("cli.log")).unwrap()
    }

    fn signal(&self, signal: &str) {
        assert!(Command::new("/bin/kill")
            .args([signal, &self.child.id().to_string()])
            .status()
            .unwrap()
            .success());
    }

    fn wait(&mut self) -> ExitStatus {
        self.wait_for(Duration::from_secs(5))
    }

    fn wait_for(&mut self, timeout: Duration) -> ExitStatus {
        let until = Instant::now() + timeout;
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return status;
            }
            assert!(
                Instant::now() < until,
                "CLI did not reap and exit: {}",
                self.log()
            );
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn bound_address(&mut self) -> String {
        self.until(|fixture| {
            fs::read_to_string(fixture.root.join("cli.log"))
                .unwrap()
                .contains("ApxInf listener bound at http://")
        });
        self.log()
            .lines()
            .find_map(|line| line.strip_prefix("ApxInf listener bound at http://"))
            .unwrap()
            .to_owned()
    }

    /// The listener now binds before loading, so wait for readiness before a
    /// test sends request traffic.
    fn address(&mut self) -> String {
        let address = self.bound_address();
        let until = Instant::now() + Duration::from_secs(5);
        loop {
            if http_status(&address, "/readyz") == Some(200) {
                return address;
            }
            assert!(
                self.child.try_wait().unwrap().is_none(),
                "CLI exited early: {}",
                self.log()
            );
            assert!(
                Instant::now() < until,
                "CLI did not become ready: {}",
                self.log()
            );
            thread::sleep(Duration::from_millis(10));
        }
    }
}

fn http_request(address: &str, method: &str, path: &str, body: Option<&str>) -> String {
    let payload = body.unwrap_or("");
    let mut connection = TcpStream::connect(address).unwrap();
    connection
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    write!(connection,
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{payload}",
        payload.len()).unwrap();
    let mut response = String::new();
    connection.read_to_string(&mut response).unwrap();
    response
}

fn http_status(address: &str, path: &str) -> Option<u16> {
    http_request(address, "GET", path, None)
        .lines()
        .next()?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

impl Drop for Cli {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

#[test]
fn sigint_during_loading_reaps_before_cli_exit() {
    loading_signal("-INT");
}

#[test]
fn sigterm_during_loading_reaps_before_cli_exit() {
    loading_signal("-TERM");
}

fn loading_signal(signal: &str) {
    let mut cli = Cli::start("loading");
    let address = cli.bound_address();
    cli.until(|fixture| fixture.worker_pid().is_some());
    // The listener now serves the loading surface before readiness.
    let response = http_request(&address, "GET", "/healthz", None);
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    let response = http_request(&address, "GET", "/readyz", None);
    assert!(response.starts_with("HTTP/1.1 503"), "{response}");
    assert!(response.contains("\"phase\":\"loading\""), "{response}");
    cli.signal(signal);
    assert!(
        cli.wait().success(),
        "Startup cancellation failed: {}",
        cli.log()
    );
    cli.fixture.assert_reaped();
}

#[test]
fn loading_listener_reports_the_loading_phase() {
    let mut cli = Cli::start("loading");
    let address = cli.bound_address();
    cli.until(|fixture| fixture.worker_pid().is_some());
    let body = r#"{"model":"cpu-signals","messages":[{"role":"user","content":"count this"}]}"#;
    for (method, path, payload) in [
        ("GET", "/v1/models", None),
        ("POST", "/v1/messages", Some(body)),
        ("POST", "/v1/messages/count_tokens", Some(body)),
        ("POST", "/v1/chat/completions", Some(body)),
        ("POST", "/v1/systemone", Some(body)),
        ("GET", "/metrics", None),
    ] {
        let response = http_request(&address, method, path, payload);
        assert!(
            response.starts_with("HTTP/1.1 503"),
            "{method} {path}: {response}"
        );
        assert!(
            response.contains("\"code\":\"model_unavailable\""),
            "{method} {path}: {response}"
        );
    }
    // The declared-route set does not change the framework responses.
    let response = http_request(&address, "GET", "/not-a-route", None);
    assert!(response.starts_with("HTTP/1.1 404"), "{response}");
    let response = http_request(&address, "GET", "/v1/messages", None);
    assert!(response.starts_with("HTTP/1.1 405"), "{response}");
    cli.signal("-TERM");
    assert!(
        cli.wait().success(),
        "Loading shutdown failed: {}",
        cli.log()
    );
    cli.fixture.assert_reaped();
}

#[test]
fn loading_surface_switches_to_ready_on_the_same_listener() {
    let mut cli = Cli::start("delayed");
    let address = cli.bound_address();
    // The worker is still loading, so this listener reports the loading phase.
    let response = http_request(&address, "GET", "/readyz", None);
    assert!(response.starts_with("HTTP/1.1 503"), "{response}");
    assert!(response.contains("\"phase\":\"loading\""), "{response}");
    // The same address must switch to ready without refusing a connection.
    let until = Instant::now() + Duration::from_secs(10);
    loop {
        let response = http_request(&address, "GET", "/readyz", None);
        assert!(
            response.starts_with("HTTP/1.1 "),
            "The listener dropped during the switch: {response}"
        );
        if response.starts_with("HTTP/1.1 200") {
            assert!(response.contains("\"ready\":true"), "{response}");
            break;
        }
        assert!(response.starts_with("HTTP/1.1 503"), "{response}");
        assert!(
            Instant::now() < until,
            "The listener never became ready: {}",
            cli.log()
        );
        thread::sleep(Duration::from_millis(10));
    }
    let response = http_request(&address, "GET", "/v1/models", None);
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    let response = http_request(&address, "GET", "/metrics", None);
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    assert_eq!(
        cli.log()
            .matches("ApxInf listener bound at http://")
            .count(),
        1,
        "The switch must not rebind the listener: {}",
        cli.log()
    );
    cli.signal("-TERM");
    assert!(
        cli.wait().success(),
        "Shutdown failed after the switch: {}",
        cli.log()
    );
    cli.fixture.assert_reaped();
}

#[test]
fn sigterm_after_ready_stops_and_reaps_the_worker() {
    let mut cli = Cli::start("ready");
    let _ = cli.address();
    cli.signal("-TERM");
    assert!(
        cli.wait().success(),
        "Runtime shutdown failed: {}",
        cli.log()
    );
    cli.fixture.assert_reaped();
}

#[test]
fn diagnostics_file_records_the_startup_evidence() {
    let mut cli = Cli::start_with_diagnostics("ready");
    let address = cli.address();
    let response = http_request(&address, "GET", "/healthz", None);
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    cli.signal("-TERM");
    assert!(cli.wait().success(), "{}", cli.log());
    let path = cli
        .diagnostics_file
        .clone()
        .expect("The file destination is configured.");
    let contents = fs::read_to_string(&path).unwrap();
    assert!(
        contents.contains("ApxInf diagnostics file at"),
        "{contents}"
    );
    assert_eq!(
        contents.matches("ApxInf listener bound at http://").count(),
        1,
        "{contents}"
    );
    assert_eq!(
        cli.log()
            .matches("ApxInf listener bound at http://")
            .count(),
        1,
        "stderr keeps the same records: {}",
        cli.log()
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "The file must be created owner-only.");
    }
}

#[test]
fn unopenable_diagnostics_file_fails_before_startup() {
    let fixture = Fixture::new("ready");
    let path = fixture.root.join("missing-directory/diagnostics.jsonl");
    let output = cli_command(&fixture, &["--diagnostics-file", path.to_str().unwrap()])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
        .wait_with_output()
        .unwrap();
    assert!(!output.status.success());
    let log = String::from_utf8_lossy(&output.stderr);
    assert!(log.contains("Cannot open the diagnostics file:"), "{log}");
    assert!(
        !log.contains("listener bound"),
        "Startup must fail before binding: {log}"
    );
}

#[test]
fn a_special_diagnostics_path_fails_before_startup() {
    let fixture = Fixture::new("ready");
    let path = fixture.root.join("diagnostics-directory");
    fs::create_dir(&path).unwrap();
    let output = cli_command(&fixture, &["--diagnostics-file", path.to_str().unwrap()])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
        .wait_with_output()
        .unwrap();
    assert!(!output.status.success());
    let log = String::from_utf8_lossy(&output.stderr);
    assert!(log.contains("must be a regular file"), "{log}");
    assert!(
        !log.contains("listener bound"),
        "Startup must fail before binding: {log}"
    );
}

#[test]
fn sigint_preserves_the_active_operation_until_settlement() {
    let mut cli = Cli::start("ready");
    let mut connection = count_request(&mut cli);
    cli.until(|fixture| fixture.root.join("preparing").exists());
    cli.signal("-INT");
    thread::sleep(Duration::from_millis(150));
    assert!(
        cli.child.try_wait().unwrap().is_none(),
        "Active work was abandoned."
    );
    fs::write(cli.fixture.root.join("release"), "finish").unwrap();
    let mut response = String::new();
    connection.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    assert!(response.contains("\"input_tokens\":2"), "{response}");
    assert!(
        cli.wait().success(),
        "Active shutdown failed: {}",
        cli.log()
    );
    cli.fixture.assert_reaped();
}

fn count_request(cli: &mut Cli) -> TcpStream {
    let mut connection = TcpStream::connect(cli.address()).unwrap();
    connection
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let body = r#"{"model":"apxinf-local","messages":[{"role":"user","content":"count this"}]}"#;
    write!(connection,
        "POST /v1/messages/count_tokens HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()).unwrap();
    connection
}

#[test]
fn unfinished_http_body_has_a_drain_limit_after_worker_reaping() {
    let mut cli = Cli::start("ready");
    let mut active = count_request(&mut cli);
    cli.until(|fixture| fixture.root.join("preparing").exists());
    let mut incomplete = TcpStream::connect(cli.address()).unwrap();
    incomplete
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    incomplete.write_all(b"POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nExpect: 100-continue\r\nContent-Length: 100000\r\n\r\n").unwrap();
    let mut interim = [0; 25];
    incomplete.read_exact(&mut interim).unwrap();
    assert_eq!(&interim, b"HTTP/1.1 100 Continue\r\n\r\n");
    incomplete.write_all(b"{").unwrap();
    cli.signal("-TERM");

    // An overall 20-second shutdown timer would incorrectly include this active operation.
    thread::sleep(Duration::from_secs(2));
    assert!(cli.child.try_wait().unwrap().is_none());
    fs::write(cli.fixture.root.join("release"), "finish").unwrap();
    let mut response = String::new();
    active.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    cli.until(|fixture| {
        !Command::new("/bin/kill")
            .args(["-0", &fixture.worker_pid().unwrap().to_string()])
            .output()
            .unwrap()
            .status
            .success()
    });
    let reaped = Instant::now();
    assert!(
        cli.wait_for(Duration::from_secs(25)).success(),
        "{}",
        cli.log()
    );
    assert!(
        reaped.elapsed() >= Duration::from_secs(19),
        "HTTP grace included worker settlement."
    );
    assert!(cli
        .log()
        .contains("HTTP connections did not drain within 20 seconds after worker reaping."));
    cli.fixture.assert_reaped();
    drop(incomplete);
}

#[test]
fn startup_error_returns_failure_after_reaping() {
    let mut cli = Cli::start("invalid_ready");
    assert!(!cli.wait().success());
    assert!(cli.log().contains("ApxInf failed:"));
    cli.fixture.assert_reaped();
}
