//! CPU regressions for automatic worker recovery after a fault.
use super::lifecycle_regressions::{cpu_service, cpu_worker, request, CpuModel};
use super::*;
use std::net::SocketAddr;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

extern "C" {
    fn kill(pid: i32, signal: i32) -> i32;
}

/// A worker that accepts one command and exits, producing a checked worker fault.
fn faulting_worker() -> Worker {
    let script = "import sys\nline = sys.stdin.readline()\nif not line:\n    raise SystemExit(0)\nraise SystemExit(3)\n";
    let mut child = Command::new("python3")
        .args(["-u", "-c", script])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    Worker {
        protocol: PROTOCOL,
        input: child.stdin.take(),
        frames: read_worker_frames(child.stdout.take().unwrap()),
        child,
        _diagnostics: None,
        epoch: uuid::Uuid::new_v4().to_string(),
        controls: Default::default(),
        sent_commands: 0,
        retired: false,
    }
}

/// A worker that exits before any command, producing an idle-pipe fault.
fn dying_worker() -> Worker {
    let script = "raise SystemExit(3)\n";
    let mut child = Command::new("python3")
        .args(["-u", "-c", script])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    Worker {
        protocol: PROTOCOL,
        input: child.stdin.take(),
        frames: read_worker_frames(child.stdout.take().unwrap()),
        child,
        _diagnostics: None,
        epoch: uuid::Uuid::new_v4().to_string(),
        controls: Default::default(),
        sent_commands: 0,
        retired: false,
    }
}

async fn fixture_ready(model: &CpuModel) -> Value {
    let (mut worker, ready) = cpu_worker(model.model()).await.unwrap();
    let _ = worker.child.start_kill();
    let _ = worker.child.wait().await;
    ready
}

fn service_with_restart(
    ready: Value,
    timeout: Duration,
    capacity: usize,
    restart: RestartPolicy,
) -> (Arc<Service>, watch::Sender<Option<StopResult>>) {
    let mut service = cpu_service(ready, timeout, capacity);
    let (completion, stopped) = watch::channel(None);
    let shared = Arc::get_mut(&mut service).unwrap();
    shared.stopped = stopped;
    shared.restart = restart;
    (service, completion)
}

async fn wait_until(condition: impl Fn() -> bool, timeout: Duration) {
    tokio::time::timeout(timeout, async {
        while !condition() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("The restart probe did not reach its expected state.");
}

fn observed(service: &Service, name: &str, labels: &str) -> f64 {
    let key = format!("{name}{{{labels}}} ");
    service
        .metrics_text()
        .lines()
        .find_map(|line| line.strip_prefix(&key))
        .unwrap_or_else(|| panic!("Missing metric: {key}"))
        .parse()
        .unwrap()
}

async fn stop_reported(service: &Arc<Service>, task: tokio::task::JoinHandle<()>) {
    service.shutdown();
    let stopped = tokio::time::timeout(Duration::from_secs(5), service.wait_stopped())
        .await
        .expect("Shutdown must confirm worker reaping.");
    task.await.unwrap();
    assert!(!service.available.load(Ordering::Acquire));
    stopped.unwrap();
}

async fn stop(service: &Arc<Service>, task: tokio::task::JoinHandle<StopResult>) {
    service.shutdown();
    let result = tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("Shutdown must end the coordinator.")
        .unwrap();
    assert!(!service.available.load(Ordering::Acquire));
    result.unwrap();
}

#[test]
fn restart_backoff_doubles_up_to_the_ceiling() {
    let mut backoff = RestartBackoff::new(RestartPolicy::default());
    let delays: Vec<u64> = (0..10)
        .map(|_| backoff.next_delay().as_millis() as u64)
        .collect();
    assert_eq!(
        delays,
        [250, 500, 1000, 2000, 4000, 8000, 16000, 30000, 30000, 30000]
    );
    backoff.reset();
    assert_eq!(backoff.next_delay().as_millis(), 250);

    let mut immediate = RestartBackoff::new(RestartPolicy::immediate(2));
    for _ in 0..5 {
        assert_eq!(immediate.next_delay(), Duration::ZERO);
    }
}

#[tokio::test]
async fn faulted_worker_restarts_after_backoff_and_serves_new_requests() {
    let model = CpuModel::new(json!({}));
    let ready = fixture_ready(&model).await;
    let (service, _completion) = service_with_restart(
        ready,
        Duration::from_secs(5),
        4,
        RestartPolicy {
            max_attempts: Some(3),
            initial_backoff: Duration::from_secs(1),
            max_backoff: Duration::from_secs(1),
        },
    );
    let worker = dying_worker();
    let faulted_epoch = worker.epoch.clone();
    let path = model.model();
    let started = Instant::now();
    // The initial worker is already dead; idle EOF is the checked fault.
    let task = tokio::spawn(coordinate(worker, service.clone(), 1024, move || {
        cpu_worker(path.clone())
    }));
    wait_until(
        || service.metrics.worker_faults.load(Ordering::Relaxed) == 1,
        Duration::from_secs(5),
    )
    .await;
    assert!(!service.available.load(Ordering::Acquire));
    let error = service
        .enqueue(request(true))
        .err()
        .expect("Backoff must reject admission.");
    assert_eq!((error.status, error.code.as_str()), (503, "worker_lost"));
    assert!(service
        .metrics_text()
        .contains("apxinf_worker_restarts_total 0"));
    wait_until(
        || service.metrics.worker_restarts.load(Ordering::Relaxed) == 1,
        Duration::from_secs(5),
    )
    .await;
    assert!(
        started.elapsed() >= Duration::from_secs(1),
        "The first restart must wait for its backoff."
    );
    assert!(service.available.load(Ordering::Acquire));
    assert_eq!(service.metrics.worker_faults.load(Ordering::Relaxed), 1);
    assert_eq!(
        service
            .metrics
            .worker_restart_failures
            .load(Ordering::Relaxed),
        0
    );
    assert_ne!(service.ready_snapshot()["worker_epoch"], faulted_epoch);

    let mut ticket = service.enqueue(request(false)).unwrap();
    assert!(ticket.begin.await.unwrap().is_ok());
    assert!(matches!(
        ticket.output.recv().await,
        Some(Event::Done { .. })
    ));
    assert_eq!(service.metrics.completed.load(Ordering::Relaxed), 1);
    stop(&service, task).await;
    assert_eq!(service.metrics.worker_restarts.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn in_flight_fault_reports_worker_lost_once_and_the_service_recovers() {
    let model = CpuModel::new(json!({}));
    let ready = fixture_ready(&model).await;
    let (service, completion) = service_with_restart(
        ready,
        Duration::from_secs(10),
        4,
        RestartPolicy {
            max_attempts: Some(3),
            initial_backoff: Duration::from_millis(50),
            max_backoff: Duration::from_millis(100),
        },
    );
    let path = model.model();
    let task = tokio::spawn(coordinate_and_report(
        faulting_worker(),
        service.clone(),
        1024,
        move || cpu_worker(path.clone()),
        completion,
    ));
    let ticket = service.enqueue(request(false)).unwrap();
    let error = match ticket.begin.await.unwrap() {
        Err(error) => error,
        Ok(_) => panic!("An in-flight worker fault must not admit the request."),
    };
    assert_eq!((error.status, error.code.as_str()), (503, "worker_lost"));
    wait_until(
        || service.metrics.worker_restarts.load(Ordering::Relaxed) == 1,
        Duration::from_secs(5),
    )
    .await;
    assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 1);
    assert_eq!(
        observed(
            &service,
            "apxinf_request_outcomes_total",
            "operation=\"generate\",status=\"failed\""
        ),
        1.0
    );
    assert_eq!(
        observed(
            &service,
            "apxinf_public_terminal_to_settlement_seconds_count",
            "operation=\"generate\""
        ),
        1.0
    );
    assert_eq!(
        observed(&service, "apxinf_cleanup_pending", "operation=\"generate\""),
        0.0
    );

    let mut next = service.enqueue(request(false)).unwrap();
    assert!(next.begin.await.unwrap().is_ok());
    assert!(matches!(next.output.recv().await, Some(Event::Done { .. })));
    assert_eq!(service.metrics.completed.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.failed.load(Ordering::Relaxed), 1);
    stop_reported(&service, task).await;
}

#[tokio::test]
async fn unconfirmed_reaping_blocks_settlement_and_restart() {
    let model = CpuModel::new(json!({}));
    let ready = fixture_ready(&model).await;
    let (mut service, completion) = service_with_restart(
        ready,
        Duration::from_secs(10),
        4,
        RestartPolicy {
            max_attempts: Some(3),
            initial_backoff: Duration::from_millis(50),
            max_backoff: Duration::from_millis(50),
        },
    );
    Arc::get_mut(&mut service)
        .unwrap()
        .unconfirmed_reaping
        .store(true, Ordering::Release);
    let path = model.model();
    let launches = Arc::new(AtomicU64::new(0));
    let counted = launches.clone();
    let task = tokio::spawn(coordinate_and_report(
        faulting_worker(),
        service.clone(),
        1024,
        move || {
            let path = path.clone();
            counted.fetch_add(1, Ordering::Relaxed);
            cpu_worker(path)
        },
        completion,
    ));
    let ticket = service.enqueue(request(false)).unwrap();
    let error = match ticket.begin.await.unwrap() {
        Err(error) => error,
        Ok(_) => panic!("An in-flight worker fault must not admit the request."),
    };
    assert_eq!((error.status, error.code.as_str()), (503, "worker_lost"));
    let error = tokio::time::timeout(Duration::from_secs(5), service.wait_stopped())
        .await
        .expect("An unconfirmed worker exit must end the coordinator.")
        .expect_err("An unconfirmed worker exit must stay a failure.");
    assert!(error.contains("could not be reaped"), "{error}");
    task.await.unwrap();
    // No replacement may load while the old process may still own memory.
    assert_eq!(launches.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.worker_faults.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.worker_restarts.load(Ordering::Relaxed), 0);
    assert_eq!(
        service
            .metrics
            .worker_restart_failures
            .load(Ordering::Relaxed),
        0
    );
    assert!(service.queue.is_closed());
    assert!(!service.available.load(Ordering::Acquire));
    // The unresolved request keeps its reservation and stays in the gauge.
    assert_eq!(service.metrics.active.load(Ordering::Relaxed), 1);
    assert_eq!(service.metrics.reserved_bytes.load(Ordering::Relaxed), 1024);
    assert_eq!(
        observed(
            &service,
            "apxinf_request_outcomes_total",
            "operation=\"generate\",status=\"failed\""
        ),
        1.0
    );
    assert_eq!(
        observed(&service, "apxinf_cleanup_pending", "operation=\"generate\""),
        1.0
    );
    assert_eq!(
        observed(
            &service,
            "apxinf_public_terminal_to_settlement_seconds_count",
            "operation=\"generate\""
        ),
        0.0
    );
    // The fixture trace keeps its single identity start: no model was loaded.
    assert_eq!(
        model
            .events()
            .iter()
            .filter(|event| event["event"] == "started")
            .count(),
        1
    );
}

#[tokio::test]
async fn consecutive_launch_failures_back_off_then_recover() {
    let model = CpuModel::new(json!({}));
    let ready = fixture_ready(&model).await;
    let (service, completion) =
        service_with_restart(ready, Duration::from_secs(5), 4, RestartPolicy::default());
    let path = model.model();
    let attempts = Arc::new(AtomicU64::new(0));
    let counted = attempts.clone();
    let started = Instant::now();
    let task = tokio::spawn(coordinate_and_report(
        dying_worker(),
        service.clone(),
        1024,
        move || {
            let path = path.clone();
            let attempt = counted.fetch_add(1, Ordering::Relaxed) + 1;
            async move {
                if attempt == 1 {
                    // The first replacement cannot start; the next attempt must
                    // observe the doubled backoff.
                    return Err(LaunchError::Worker(
                        "The replacement process could not start.".into(),
                    ));
                }
                cpu_worker(path).await.map_err(LaunchError::Worker)
            }
        },
        completion,
    ));
    wait_until(
        || service.metrics.worker_restarts.load(Ordering::Relaxed) == 1,
        Duration::from_secs(8),
    )
    .await;
    assert_eq!(attempts.load(Ordering::Relaxed), 2);
    assert_eq!(
        service
            .metrics
            .worker_restart_failures
            .load(Ordering::Relaxed),
        1
    );
    assert_eq!(service.metrics.worker_faults.load(Ordering::Relaxed), 1);
    assert!(
        started.elapsed() >= WORKER_RESTART_INITIAL_BACKOFF * 3,
        "A failed attempt must not skip the doubled backoff: {:?}",
        started.elapsed()
    );
    assert!(service
        .metrics_text()
        .contains("apxinf_worker_restarts_total 1"));
    assert!(service
        .metrics_text()
        .contains("apxinf_worker_restart_failures_total 1"));
    assert!(service.available.load(Ordering::Acquire));
    stop_reported(&service, task).await;
}

#[tokio::test]
async fn consecutive_worker_faults_restart_after_every_fault() {
    let model = CpuModel::new(json!({}));
    let (initial, ready) = cpu_worker(model.model()).await.unwrap();
    let first_epoch = initial.epoch.clone();
    let (service, completion) = service_with_restart(
        ready,
        Duration::from_secs(5),
        4,
        RestartPolicy {
            max_attempts: Some(4),
            initial_backoff: Duration::from_millis(50),
            max_backoff: Duration::from_millis(100),
        },
    );
    let path = model.model();
    let task = tokio::spawn(coordinate_and_report(
        initial,
        service.clone(),
        1024,
        move || cpu_worker(path.clone()),
        completion,
    ));
    let started_pid = |exclude: &str| {
        model
            .events()
            .into_iter()
            .filter(|event| event["event"] == "started")
            .filter(|event| event["worker_epoch"] != exclude)
            .last()
            .and_then(|event| event["pid"].as_u64())
    };
    // Fault 1: kill the worker that the service started with.
    unsafe { kill(started_pid("").unwrap() as i32, 9) };
    wait_until(
        || service.metrics.worker_restarts.load(Ordering::Relaxed) == 1,
        Duration::from_secs(5),
    )
    .await;
    assert_ne!(
        service.ready_snapshot()["worker_epoch"].as_str().unwrap(),
        first_epoch.as_str()
    );
    let second_pid = started_pid(&first_epoch).expect("The replacement must trace its pid.");
    // Fault 2: kill the replacement; the supervisor must recover again.
    unsafe { kill(second_pid as i32, 9) };
    wait_until(
        || service.metrics.worker_restarts.load(Ordering::Relaxed) == 2,
        Duration::from_secs(5),
    )
    .await;
    assert_eq!(service.metrics.worker_faults.load(Ordering::Relaxed), 2);
    assert_eq!(
        service
            .metrics
            .worker_restart_failures
            .load(Ordering::Relaxed),
        0
    );
    assert!(service.available.load(Ordering::Acquire));
    let mut ticket = service.enqueue(request(false)).unwrap();
    assert!(ticket.begin.await.unwrap().is_ok());
    assert!(matches!(
        ticket.output.recv().await,
        Some(Event::Done { .. })
    ));
    stop_reported(&service, task).await;
}

#[tokio::test]
async fn shutdown_during_backoff_skips_the_restart() {
    let model = CpuModel::new(json!({}));
    let ready = fixture_ready(&model).await;
    let (service, completion) = service_with_restart(
        ready,
        Duration::from_secs(5),
        4,
        RestartPolicy {
            max_attempts: Some(2),
            initial_backoff: Duration::from_secs(5),
            max_backoff: Duration::from_secs(5),
        },
    );
    let path = model.model();
    let launches = Arc::new(AtomicU64::new(0));
    let counted = launches.clone();
    let task = tokio::spawn(coordinate_and_report(
        dying_worker(),
        service.clone(),
        1024,
        move || {
            let path = path.clone();
            counted.fetch_add(1, Ordering::Relaxed);
            cpu_worker(path)
        },
        completion,
    ));
    wait_until(
        || service.metrics.worker_faults.load(Ordering::Relaxed) == 1,
        Duration::from_secs(5),
    )
    .await;
    assert!(!service.available.load(Ordering::Acquire));
    let stopped_at = Instant::now();
    service.shutdown();
    tokio::time::timeout(Duration::from_secs(3), service.wait_stopped())
        .await
        .expect("Shutdown during backoff must reap the fenced worker promptly.")
        .unwrap();
    task.await.unwrap();
    assert!(
        stopped_at.elapsed() < Duration::from_secs(2),
        "Shutdown must not wait for the backoff to elapse."
    );
    assert_eq!(launches.load(Ordering::Relaxed), 0);
    assert_eq!(service.metrics.worker_restarts.load(Ordering::Relaxed), 0);
    assert!(service.queue.is_closed());
    assert!(service.enqueue(request(true)).is_err());
    // The fixture traced the identity probe only; no replacement was started.
    assert_eq!(
        model
            .events()
            .iter()
            .filter(|event| event["event"] == "started")
            .count(),
        1
    );
}

#[tokio::test]
async fn exhausted_restart_budget_fences_the_service() {
    let model = CpuModel::new(json!({}));
    let ready = fixture_ready(&model).await;
    let (service, completion) = service_with_restart(
        ready,
        Duration::from_secs(5),
        4,
        RestartPolicy::immediate(2),
    );
    let task = tokio::spawn(coordinate_and_report(
        dying_worker(),
        service.clone(),
        1024,
        || async {
            Err::<(Worker, Value), _>(LaunchError::Worker(
                "The replacement can never start.".into(),
            ))
        },
        completion,
    ));
    let error = tokio::time::timeout(Duration::from_secs(5), service.wait_stopped())
        .await
        .expect("An exhausted restart budget must end the coordinator.")
        .expect_err("An exhausted restart budget must report a failed coordinator outcome.");
    assert_eq!(
        error,
        "The worker restart budget of 2 attempts is exhausted."
    );
    task.await.unwrap();
    assert_eq!(
        service
            .metrics
            .worker_restart_failures
            .load(Ordering::Relaxed),
        2
    );
    assert_eq!(service.metrics.worker_restarts.load(Ordering::Relaxed), 0);
    assert!(!service.available.load(Ordering::Acquire));
    assert!(service.queue.is_closed());
    let error = service.enqueue(request(true)).err().unwrap();
    assert_eq!((error.status, error.code.as_str()), (503, "worker_lost"));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let router = crate::gateway::router(service.clone());
    let server = tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    let (status, body) = http_request(address, http_get("/readyz")).await;
    assert_eq!(status, 503, "A fenced service must report unready: {body}");
    server.abort();
}

async fn http_request(addr: SocketAddr, request: String) -> (u16, String) {
    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    stream.write_all(request.as_bytes()).await.unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.unwrap();
    let text = String::from_utf8_lossy(&response).into_owned();
    let status = text
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse::<u16>()
        .unwrap();
    (status, text)
}

fn http_get(path: &str) -> String {
    format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
}

fn http_post(path: &str, body: &str) -> String {
    format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
}

#[tokio::test]
async fn readyz_reports_backoff_then_serves_a_recovered_request() {
    let model = CpuModel::new(json!({}));
    let ready = fixture_ready(&model).await;
    let (service, completion) = service_with_restart(
        ready,
        Duration::from_secs(10),
        4,
        RestartPolicy {
            max_attempts: Some(2),
            initial_backoff: Duration::from_secs(2),
            max_backoff: Duration::from_secs(2),
        },
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let router = crate::gateway::router(service.clone());
    let server = tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    let path = model.model();
    let task = tokio::spawn(coordinate_and_report(
        dying_worker(),
        service.clone(),
        1024,
        move || cpu_worker(path.clone()),
        completion,
    ));
    wait_until(
        || service.metrics.worker_faults.load(Ordering::Relaxed) == 1,
        Duration::from_secs(5),
    )
    .await;
    let (status, body) = http_request(address, http_get("/readyz")).await;
    assert_eq!(
        status, 503,
        "Backoff must report an unready service: {body}"
    );
    assert!(body.contains("\"ready\":false"));

    let recovered = tokio::time::timeout(Duration::from_secs(8), async {
        loop {
            let (status, body) = http_request(address, http_get("/readyz")).await;
            if status == 200 {
                return body;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("A restarted worker must restore readiness.");
    assert!(recovered.contains("\"ready\":true"));

    let body = r#"{"model":"cpu-fixture","messages":[{"role":"user","content":"count this"}]}"#;
    let (status, response) =
        http_request(address, http_post("/v1/messages/count_tokens", body)).await;
    assert_eq!(
        status, 200,
        "A recovered service must serve requests: {response}"
    );
    assert!(response.contains("\"input_tokens\":2"), "{response}");

    stop_reported(&service, task).await;
    server.abort();
}
