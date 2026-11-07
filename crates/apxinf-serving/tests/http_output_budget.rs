//! Real HTTP checks for the public output-budget contract.
//!
//! The gateway keeps an omitted `max_tokens` inside the deployment output
//! budget and rejects an explicit zero before admission. These checks pin that
//! behavior on the HTTP boundary instead of only inside `normalize()`: the
//! clamped budget must reach the worker, and a rejected limit must not.
//!
//! The fixture is the existing original CPU protocol worker; it loads no MLX
//! package and makes no inference claim.
#![cfg(unix)]

use serde_json::{json, Value};
use std::{
    fs,
    io::{Read, Write},
    net::TcpStream,
    path::PathBuf,
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

const MODEL: &str = "original-cpu-fixture";

/// A real gateway process over the original CPU protocol fixture.
struct Gateway {
    child: Option<Child>,
    root: PathBuf,
    port: u16,
}

impl Gateway {
    fn start(name: &str, budget: usize) -> Self {
        let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let root = std::env::temp_dir();
        let root = root.join(format!("apxinf-http-budget-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let worker = root.join("python/apxinf/apxinf/serving/text_worker.py");
        fs::create_dir_all(worker.parent().unwrap()).unwrap();
        fs::create_dir_all(root.join("scripts")).unwrap();
        fs::create_dir_all(root.join("model")).unwrap();
        fs::write(root.join("model/fixture.txt"), "Original CPU fixture.\n").unwrap();
        let copies = [
            ("tests/fixtures/serving/http_worker.py", worker.clone()),
            (
                "python/apxinf/apxinf/serving/contracts.py",
                worker.with_file_name("contracts.py"),
            ),
            (
                "scripts/apxinf_mlx_generate.py",
                root.join("scripts/apxinf_mlx_generate.py"),
            ),
        ];
        for (from, to) in copies {
            fs::copy(source.join(from), to).unwrap();
        }
        let output = Command::new("python3")
            .args(["-c", "import sys; print(sys.executable)"])
            .output()
            .expect("Python 3 is required for the HTTP fixture.");
        assert!(
            output.status.success(),
            "Python 3 must run the CPU fixture."
        );
        let python = PathBuf::from(String::from_utf8(output.stdout).unwrap().trim());
        let child = Command::new(env!("CARGO_BIN_EXE_apxinf-serve"))
            .arg("--model")
            .arg(root.join("model"))
            .args(["--model-id", MODEL, "--python"])
            .arg(python)
            .args(["--worker"])
            .arg(&worker)
            .args(["--port", "0", "--host-pressure-policy", "disabled"])
            .args(["--max-context", "128", "--queue-capacity", "1"])
            .args(["--timeout-seconds", "5", "--memory-budget-bytes", "8192"])
            .args(["--sequence-reservation-bytes", "1024"])
            .arg("--max-output-tokens")
            .arg(budget.to_string())
            .stdout(Stdio::null())
            .stderr(fs::File::create(root.join("gateway.log")).unwrap())
            .spawn()
            .unwrap();
        let mut gateway = Self {
            child: Some(child),
            root,
            port: 0,
        };
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let log = gateway.log();
            let port = log.lines().find_map(|line| {
                line.strip_prefix("ApxInf listener bound at http://127.0.0.1:")
                    .and_then(|port| port.trim().parse().ok())
            });
            if let Some(port) = port {
                gateway.port = port;
                break;
            }
            gateway.assert_running("startup", deadline);
        }
        while gateway.request("/readyz", None).0 != 200 {
            gateway.assert_running("readiness", deadline);
        }
        gateway
    }

    fn log(&self) -> String {
        fs::read_to_string(self.root.join("gateway.log")).unwrap_or_default()
    }

    fn assert_running(&mut self, phase: &str, deadline: Instant) {
        let child = self.child.as_mut().unwrap();
        assert!(
            child.try_wait().unwrap().is_none(),
            "Gateway exited during {phase}: {}",
            self.log()
        );
        assert!(
            Instant::now() < deadline,
            "Gateway did not reach {phase}: {}",
            self.log()
        );
        thread::sleep(Duration::from_millis(10));
    }

    /// Send one request and return the status with the raw response payload.
    fn exchange(&self, method: &str, path: &str, body: &str) -> (u16, String) {
        let mut connection = TcpStream::connect(("127.0.0.1", self.port)).unwrap();
        connection
            .set_read_timeout(Some(Duration::from_secs(60)))
            .unwrap();
        let headers = format!(
            "Content-Type: application/json\r\nContent-Length: {}",
            body.len()
        );
        write!(
            connection,
            "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n{headers}\r\n\r\n{body}"
        )
        .unwrap();
        let mut raw = String::new();
        connection.read_to_string(&mut raw).unwrap();
        let (head, payload) = raw
            .split_once("\r\n\r\n")
            .unwrap_or_else(|| panic!("{path} returned no HTTP headers: {raw}"));
        assert!(
            !head
                .to_ascii_lowercase()
                .contains("transfer-encoding: chunked"),
            "{path} returned an unexpected chunked body."
        );
        let status = head
            .lines()
            .next()
            .and_then(|line| line.split(' ').nth(1).and_then(|code| code.parse().ok()))
            .unwrap_or_else(|| panic!("{path} returned no status: {head}"));
        (status, payload.to_owned())
    }

    fn request(&self, path: &str, document: Option<&Value>) -> (u16, Value) {
        let body = document.map(Value::to_string).unwrap_or_default();
        let method = if document.is_some() { "POST" } else { "GET" };
        let (status, payload) = self.exchange(method, path, &body);
        (
            status,
            serde_json::from_str(&payload).unwrap_or(Value::Null),
        )
    }

    /// Read one Prometheus sample from `/metrics`.
    fn metric(&self, sample: &str) -> Option<f64> {
        let (status, payload) = self.exchange("GET", "/metrics", "{}");
        assert_eq!(status, 200);
        payload
            .lines()
            .find_map(|line| line.strip_prefix(sample)?.strip_prefix(' ')?.parse().ok())
    }

    fn events(&self) -> Vec<Value> {
        let text = fs::read_to_string(self.root.join("http-worker-events.jsonl"));
        text.map(|text| {
            text.lines()
                .filter_map(|line| serde_json::from_str(line).ok())
                .collect()
        })
        .unwrap_or_default()
    }

    fn accepted(&self) -> usize {
        let events = self.events();
        events
            .iter()
            .filter(|event| event["event"] == "accepted")
            .count()
    }

    fn settled(&self) {
        let deadline = Instant::now() + Duration::from_secs(15);
        while self.metric("apxinf_active_requests") != Some(0.0)
            || self.metric("apxinf_queued_requests") != Some(0.0)
        {
            assert!(
                Instant::now() < deadline,
                "Resources did not settle: {}",
                self.log()
            );
            thread::sleep(Duration::from_millis(10));
        }
    }

    /// Interrupt the CLI, require a clean exit, and require a reaped worker.
    fn stop(mut self) {
        let mut child = self.child.take().unwrap();
        let signal = Command::new("/bin/kill")
            .args(["-INT", &child.id().to_string()])
            .output()
            .unwrap();
        assert!(signal.status.success());
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(status.success(), "Graceful shutdown failed: {}", self.log());
                break;
            }
            assert!(
                Instant::now() < deadline,
                "Gateway did not exit: {}",
                self.log()
            );
            thread::sleep(Duration::from_millis(10));
        }
        let worker = self.events();
        let pid = worker.iter().find_map(|event| event["pid"].as_u64());
        let worker = pid.expect("The CPU worker must start.").to_string();
        let alive = Command::new("/bin/kill").args(["-0", &worker]).output();
        assert!(
            !alive.unwrap().status.success(),
            "The CLI must reap worker {worker}."
        );
    }
}

impl Drop for Gateway {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn body(scenario: &str) -> Value {
    let content = format!("fixture:{scenario}");
    json!({"model": MODEL, "messages": [{"role": "user", "content": content}]})
}

/// A small deployment budget still admits the most common request shape.
#[test]
fn omitted_output_limit_is_admitted_with_the_deployment_budget() {
    let gateway = Gateway::start("clamped", 8);

    // A zero limit is a request error and never reaches queue admission.
    for path in ["/v1/messages", "/v1/chat/completions"] {
        let mut document = body("text");
        document["max_tokens"] = json!(0);
        let (status, reply) = gateway.request(path, Some(&document));
        assert_eq!(status, 400, "{path}: {reply}");
        assert_eq!(reply["error"]["code"], "invalid_request", "{path}: {reply}");
        let message = reply["error"]["message"].as_str().unwrap_or_default();
        assert!(message.contains("at least 1"), "{path}: {reply}");
    }
    assert_eq!(
        gateway.accepted(),
        0,
        "Rejected limits must not reach the worker."
    );
    assert_eq!(
        gateway.metric("apxinf_requests_total{status=\"failed\"}"),
        Some(0.0)
    );

    // Counting tokens ignores the output budget.
    let (status, reply) = gateway.request("/v1/messages/count_tokens", Some(&body("text")));
    assert_eq!((status, reply), (200, json!({"input_tokens": 2})));

    // An omitted limit reserves the deployment budget, so the worker emits
    // exactly that many tokens instead of receiving 512 and failing the budget.
    // The fixture needs about 0.16s per token, so this deployment stays small.
    let (status, reply) = gateway.request("/v1/chat/completions", Some(&body("slow")));
    assert_eq!(status, 200, "{reply}");
    assert_eq!(reply["choices"][0]["finish_reason"], "length", "{reply}");
    assert_eq!(reply["usage"]["completion_tokens"], 8, "{reply}");
    let (status, reply) = gateway.request("/v1/messages", Some(&body("slow")));
    assert_eq!(status, 200, "{reply}");
    assert_eq!(reply["stop_reason"], "max_tokens", "{reply}");
    assert_eq!(reply["usage"]["output_tokens"], 8, "{reply}");

    gateway.settled();
    assert_eq!(gateway.accepted(), 2);
    assert_eq!(
        gateway.metric("apxinf_requests_total{status=\"completed\"}"),
        Some(3.0)
    );
    gateway.stop();
}

/// An exhausted deployment output budget stays ready and keeps token counting.
#[test]
fn exhausted_output_budget_rejects_generation_but_counts_tokens() {
    let gateway = Gateway::start("exhausted", 0);
    assert_eq!(gateway.request("/readyz", None).0, 200);

    for path in ["/v1/messages", "/v1/chat/completions"] {
        let (status, reply) = gateway.request(path, Some(&body("text")));
        assert_eq!(status, 400, "{path}: {reply}");
        assert_eq!(reply["error"]["code"], "invalid_request", "{path}: {reply}");
        assert_eq!(
            reply["error"]["message"], "This deployment allows no output tokens.",
            "{path}: {reply}"
        );
    }
    // An explicit limit above an exhausted deployment keeps the older message.
    let mut over = body("text");
    over["max_tokens"] = json!(1);
    let (status, reply) = gateway.request("/v1/chat/completions", Some(&over));
    assert_eq!(status, 400, "{reply}");
    assert_eq!(
        reply["error"]["message"], "max_tokens exceeds the deployment output limit.",
        "{reply}"
    );

    let (status, reply) = gateway.request("/v1/messages/count_tokens", Some(&body("text")));
    assert_eq!((status, reply), (200, json!({"input_tokens": 2})));

    gateway.settled();
    assert_eq!(
        gateway.accepted(),
        0,
        "Rejected limits must not reach the worker."
    );
    assert_eq!(
        gateway.metric("apxinf_requests_total{status=\"failed\"}"),
        Some(0.0)
    );
    gateway.stop();
}
