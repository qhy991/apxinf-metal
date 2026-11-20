//! Waiting requests release their queue slots before model execution finishes.
use super::{ApiError, Job, Metrics};
use crate::metrics::Outcome;
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
    time::{Duration, Instant},
};
use tokio::sync::Notify;

struct State {
    jobs: VecDeque<Job>,
    closed: bool,
    paused: bool,
}

pub(super) struct PendingQueue {
    state: Mutex<State>,
    changed: Notify,
    capacity: usize,
}

impl PendingQueue {
    pub(super) fn is_closed(&self) -> bool {
        self.state
            .lock()
            .expect("The waiting queue lock is poisoned.")
            .closed
    }

    pub(super) async fn wait_closed(&self) {
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if self.is_closed() {
                return;
            }
            changed.await;
        }
    }

    pub(super) fn new(capacity: usize) -> Self {
        Self {
            state: Mutex::new(State {
                jobs: VecDeque::new(),
                closed: false,
                paused: false,
            }),
            changed: Notify::new(),
            capacity,
        }
    }

    pub(super) fn push(
        &self,
        job: Job,
        metrics: &Metrics,
        timeout: Duration,
    ) -> Result<(), ApiError> {
        let mut state = self
            .state
            .lock()
            .expect("The waiting queue lock is poisoned.");
        Self::sweep_locked(&mut state, metrics, timeout);
        if state.closed {
            return Err(ApiError::worker("The request coordinator is unavailable."));
        }
        if state.paused {
            return Err(ApiError::worker(
                "The model worker is restarting after a fault.",
            ));
        }
        if state.jobs.len() >= self.capacity {
            return Err(ApiError::new(
                429,
                "queue_full",
                "The request queue is full.",
            ));
        }
        state.jobs.push_back(job);
        metrics.queued.fetch_add(1, Ordering::Relaxed);
        drop(state);
        self.changed.notify_one();
        Ok(())
    }

    pub(super) fn sweep(&self, metrics: &Metrics, timeout: Duration) {
        let mut state = self
            .state
            .lock()
            .expect("The waiting queue lock is poisoned.");
        Self::sweep_locked(&mut state, metrics, timeout);
    }

    #[cfg(test)]
    pub(super) async fn recv(&self, metrics: &Metrics) -> Option<Job> {
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            {
                let mut state = self
                    .state
                    .lock()
                    .expect("The waiting queue lock is poisoned.");
                if let Some(job) = state.jobs.pop_front() {
                    metrics.queued.fetch_sub(1, Ordering::Relaxed);
                    job.observation
                        .queue_end(&metrics.observations, Instant::now());
                    return Some(job);
                }
                if state.closed {
                    return None;
                }
                // A paused queue holds no work until the supervisor resumes it.
            }
            changed.await;
        }
    }

    /// Receives up to `max` ready jobs in one scheduler turn. QUEUE #6's
    /// pre-registration freezes the dispatch mechanism here: pop up to B jobs
    /// "whose prompts are ready", preserving admission order so the
    /// coordinator's later demux keeps per-request output channels aligned.
    ///
    /// Returns as soon as at least one job is ready (including a partial
    /// batch) so a single warm request is never held waiting for siblings.
    /// Parks while the open queue is empty, mirrors [`Self::recv`] on close,
    /// and floors `max` at 1 so a misconfigured bound can never starve work.
    pub(super) async fn recv_batch(&self, metrics: &Metrics, max: usize) -> Option<Vec<Job>> {
        let bound = max.max(1);
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            {
                let mut state = self
                    .state
                    .lock()
                    .expect("The waiting queue lock is poisoned.");
                let count = state.jobs.len().min(bound);
                if count > 0 {
                    let now = Instant::now();
                    let mut batch = Vec::with_capacity(count);
                    for _ in 0..count {
                        let job = state
                            .jobs
                            .pop_front()
                            .expect("A ready job must exist before the batch pop.");
                        metrics.queued.fetch_sub(1, Ordering::Relaxed);
                        job.observation.queue_end(&metrics.observations, now);
                        batch.push(job);
                    }
                    return Some(batch);
                }
                if state.closed {
                    return None;
                }
                // A paused queue holds no work until the supervisor resumes it.
            }
            changed.await;
        }
    }

    pub(super) fn close(&self, metrics: &Metrics, timeout: Duration) {
        let mut state = self
            .state
            .lock()
            .expect("The waiting queue lock is poisoned.");
        state.closed = true;
        state.paused = true;
        Self::sweep_locked(&mut state, metrics, timeout);
        Self::drain_locked(&mut state, metrics, "The service is shutting down.");
        drop(state);
        self.changed.notify_waiters();
    }

    /// Stops admission and rejects waiting work without ending the service.
    pub(super) fn pause(&self, metrics: &Metrics, timeout: Duration) {
        let mut state = self
            .state
            .lock()
            .expect("The waiting queue lock is poisoned.");
        if state.closed {
            return;
        }
        state.paused = true;
        Self::sweep_locked(&mut state, metrics, timeout);
        Self::drain_locked(
            &mut state,
            metrics,
            "The model worker is restarting after a fault.",
        );
        drop(state);
        self.changed.notify_waiters();
    }

    /// Reopens admission after a worker restart. Returns false during shutdown.
    pub(super) fn resume(&self, available: &AtomicBool) -> bool {
        let mut state = self
            .state
            .lock()
            .expect("The waiting queue lock is poisoned.");
        if state.closed {
            return false;
        }
        state.paused = false;
        available.store(true, Ordering::Release);
        drop(state);
        self.changed.notify_waiters();
        true
    }

    fn drain_locked(state: &mut State, metrics: &Metrics, message: &str) {
        while let Some(job) = state.jobs.pop_front() {
            metrics.queued.fetch_sub(1, Ordering::Relaxed);
            metrics.failed.fetch_add(1, Ordering::Relaxed);
            let now = Instant::now();
            job.observation.queue_end(&metrics.observations, now);
            job.observation.settled(&metrics.observations, now);
            job.observation
                .public_terminal(&metrics.observations, Outcome::Failed, now);
            let _ = job.begin.send(Err(ApiError::worker(message)));
        }
    }

    fn sweep_locked(state: &mut State, metrics: &Metrics, timeout: Duration) {
        let now = Instant::now();
        let waiting = state.jobs.len();
        for _ in 0..waiting {
            let job = state
                .jobs
                .pop_front()
                .expect("A waiting request must exist.");
            let error = if now.saturating_duration_since(job.ingress) >= timeout {
                metrics.expired.fetch_add(1, Ordering::Relaxed);
                Some(ApiError::new(
                    504,
                    "deadline_exceeded",
                    "The request expired in the queue.",
                ))
            } else if job.cancel.load(Ordering::Acquire)
                || job.begin.is_closed()
                || job.output.is_closed()
            {
                metrics.cancelled.fetch_add(1, Ordering::Relaxed);
                Some(ApiError::new(
                    499,
                    "cancelled",
                    "The request was cancelled in the queue.",
                ))
            } else {
                None
            };
            if let Some(error) = error {
                metrics.queued.fetch_sub(1, Ordering::Relaxed);
                job.observation.queue_end(&metrics.observations, now);
                job.observation.settled(&metrics.observations, now);
                job.observation.public_terminal(
                    &metrics.observations,
                    if error.code == "deadline_exceeded" {
                        Outcome::Expired
                    } else {
                        Outcome::Cancelled
                    },
                    now,
                );
                let _ = job.begin.send(Err(error));
            } else {
                state.jobs.push_back(job);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::supervisor::{Request, Ticket, OUTPUT_PART_CAPACITY};
    use serde_json::json;
    use std::{
        sync::{atomic::AtomicBool, Arc},
        time::Instant,
    };
    use tokio::sync::{mpsc, oneshot};

    fn waiting_job(index: u64, ingress: Instant) -> (Job, Ticket) {
        let (begin, begin_rx) = oneshot::channel();
        let (output, output_rx) = mpsc::channel(OUTPUT_PART_CAPACITY + 1);
        let terminal_output = output.clone().try_reserve_owned().unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        (
            Job {
                observation: Arc::new(crate::metrics::RequestObservation::new(ingress, false)),
                workload: super::super::Workload::Text(Request {
                    selection: crate::sampling::Selection::Greedy,
                    messages: vec![json!({"role": "user", "content": "hello"})],
                    tools: Vec::new(),
                    max_tokens: 1,
                    stops: Vec::new(),
                    count_only: false,
                }),
                id: format!("00000000-0000-4000-8000-{index:012}"),
                ingress,
                begin,
                output,
                terminal_output: Some(terminal_output),
                cancel: cancel.clone(),
            },
            Ticket {
                begin: begin_rx,
                output: output_rx,
                cancel,
            },
        )
    }

    #[tokio::test]
    async fn cancelled_waiter_frees_capacity_without_an_execution_step() {
        let queue = PendingQueue::new(1);
        let metrics = Metrics::default();
        let timeout = Duration::from_secs(60);
        let (cancelled, cancelled_ticket) = waiting_job(1, Instant::now());
        queue.push(cancelled, &metrics, timeout).unwrap();
        cancelled_ticket.cancel.store(true, Ordering::Release);
        let (replacement, _replacement_ticket) = waiting_job(2, Instant::now());

        queue
            .push(replacement, &metrics, timeout)
            .expect("A cancelled waiter must not reject the next request.");

        let error = cancelled_ticket.begin.await.unwrap().err().unwrap();
        assert_eq!(error.code, "cancelled");
        assert_eq!(metrics.queued.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.cancelled.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.failed.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn deadline_wins_over_cancellation_and_settles_once() {
        let queue = PendingQueue::new(1);
        let metrics = Metrics::default();
        let timeout = Duration::from_millis(10);
        let (expired, ticket) = waiting_job(1, Instant::now() - Duration::from_secs(1));
        queue.push(expired, &metrics, timeout).unwrap();
        ticket.cancel.store(true, Ordering::Release);

        queue.sweep(&metrics, timeout);
        queue.sweep(&metrics, timeout);
        queue.close(&metrics, timeout);

        let error = ticket.begin.await.unwrap().err().unwrap();
        assert_eq!(error.status, 504);
        assert_eq!(error.code, "deadline_exceeded");
        assert_eq!(metrics.queued.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.expired.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.cancelled.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.failed.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn disconnected_waiters_settle_without_an_execution_step() {
        let queue = PendingQueue::new(2);
        let metrics = Metrics::default();
        let timeout = Duration::from_secs(60);
        let (first, mut first_ticket) = waiting_job(1, Instant::now());
        let (second, mut second_ticket) = waiting_job(2, Instant::now());
        queue.push(first, &metrics, timeout).unwrap();
        queue.push(second, &metrics, timeout).unwrap();
        first_ticket.begin.close();
        second_ticket.output.close();

        queue.sweep(&metrics, timeout);

        let error = second_ticket.begin.await.unwrap().err().unwrap();
        assert_eq!(error.code, "cancelled");
        assert_eq!(metrics.queued.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.cancelled.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.expired.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.failed.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn sweep_preserves_live_request_order_and_capacity() {
        let queue = PendingQueue::new(5);
        let metrics = Metrics::default();
        let timeout = Duration::from_secs(60);
        let mut tickets = Vec::new();
        for index in 1..=5 {
            let (job, ticket) = waiting_job(index, Instant::now());
            queue.push(job, &metrics, timeout).unwrap();
            tickets.push(ticket);
        }
        let (overflow, _overflow_ticket) = waiting_job(6, Instant::now());
        assert_eq!(
            queue.push(overflow, &metrics, timeout).unwrap_err().status,
            429
        );
        tickets[1].cancel.store(true, Ordering::Release);
        tickets[3].cancel.store(true, Ordering::Release);

        queue.sweep(&metrics, timeout);

        for index in [1, 3, 5] {
            let job = queue.recv(&metrics).await.unwrap();
            assert_eq!(job.id, format!("00000000-0000-4000-8000-{index:012}"));
        }
        assert_eq!(metrics.queued.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.cancelled.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.failed.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn recv_batch_returns_up_to_max_in_fifo_order() {
        let queue = PendingQueue::new(5);
        let metrics = Metrics::default();
        let timeout = Duration::from_secs(60);
        // Keep every ticket alive: dropping the receiver closes the job's
        // channels and sweep_locked would settle it as cancelled on the next
        // push, draining the queue before recv_batch ever runs.
        let mut tickets = Vec::new();
        for index in 1..=5 {
            let (job, ticket) = waiting_job(index, Instant::now());
            queue.push(job, &metrics, timeout).unwrap();
            tickets.push(ticket);
        }

        let first = queue.recv_batch(&metrics, 3).await.unwrap();
        assert_eq!(
            first
                .iter()
                .map(|job| job.id.as_str())
                .collect::<Vec<_>>(),
            [
                "00000000-0000-4000-8000-000000000001",
                "00000000-0000-4000-8000-000000000002",
                "00000000-0000-4000-8000-000000000003",
            ],
            "A batch must preserve admission order so per-request output demux stays aligned."
        );
        assert_eq!(metrics.queued.load(Ordering::Relaxed), 2);

        let second = queue.recv_batch(&metrics, 8).await.unwrap();
        assert_eq!(second.len(), 2, "recv_batch must not over-pop the queue.");
        assert_eq!(
            second
                .iter()
                .map(|job| job.id.as_str())
                .collect::<Vec<_>>(),
            [
                "00000000-0000-4000-8000-000000000004",
                "00000000-0000-4000-8000-000000000005",
            ]
        );
        assert_eq!(metrics.queued.load(Ordering::Relaxed), 0);
        drop(tickets);
    }

    #[tokio::test]
    async fn recv_batch_floors_zero_max_to_one() {
        let queue = PendingQueue::new(2);
        let metrics = Metrics::default();
        let timeout = Duration::from_secs(60);
        // Tickets must stay alive for the whole test or sweep_locked will
        // discard the queued job as a disconnected waiter.
        let mut tickets = Vec::new();
        for index in 1..=2 {
            let (job, ticket) = waiting_job(index, Instant::now());
            queue.push(job, &metrics, timeout).unwrap();
            tickets.push(ticket);
        }

        let batch = queue.recv_batch(&metrics, 0).await.unwrap();
        assert_eq!(
            batch.len(),
            1,
            "A zero batch bound must degrade to one job, never starve the queue."
        );
        assert_eq!(metrics.queued.load(Ordering::Relaxed), 1);
        drop(tickets);
    }

    #[tokio::test]
    async fn recv_batch_drains_ready_jobs_then_returns_none_after_close() {
        let queue = PendingQueue::new(2);
        let metrics = Metrics::default();
        let timeout = Duration::from_secs(60);
        // Hold the tickets so the queue is not swept as disconnected waiters.
        let mut tickets = Vec::new();
        for index in 1..=2 {
            let (job, ticket) = waiting_job(index, Instant::now());
            queue.push(job, &metrics, timeout).unwrap();
            tickets.push(ticket);
        }

        let batch = queue.recv_batch(&metrics, 8).await.unwrap();
        assert_eq!(batch.len(), 2);
        drop(tickets);
        queue.close(&metrics, timeout);

        assert!(
            queue.recv_batch(&metrics, 8).await.is_none(),
            "A closed, drained queue must end the dispatcher loop."
        );
        assert_eq!(metrics.queued.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.failed.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn recv_batch_waits_while_open_then_wakes_for_admission() {
        let queue = PendingQueue::new(1);
        let metrics = Metrics::default();
        let timeout = Duration::from_secs(60);
        let receive = queue.recv_batch(&metrics, 4);
        tokio::pin!(receive);
        assert!(
            futures_util::poll!(&mut receive).is_pending(),
            "An open, empty queue must keep the dispatcher parked instead of spinning."
        );
        let (job, _ticket) = waiting_job(1, Instant::now());
        queue.push(job, &metrics, timeout).unwrap();

        let batch = tokio::time::timeout(Duration::from_secs(1), &mut receive)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(batch.len(), 1);
        assert_eq!(metrics.queued.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn close_settles_waiters_with_worker_lost_instead_of_dispatching_them() {
        let queue = PendingQueue::new(2);
        let metrics = Metrics::default();
        let timeout = Duration::from_secs(60);
        let (job, ticket) = waiting_job(1, Instant::now());
        queue.push(job, &metrics, timeout).unwrap();

        // close() drains ready work with a synthetic failure before notifying,
        // so a dispatcher blocked in recv_batch must observe None, not a stale
        // job that the shutdown path already settled.
        queue.close(&metrics, timeout);

        assert!(queue.recv_batch(&metrics, 4).await.is_none());
        let error = ticket.begin.await.unwrap().err().unwrap();
        assert_eq!(error.code, "worker_lost");
        assert_eq!(metrics.queued.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.failed.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn close_preserves_existing_outcomes_and_rejects_new_requests() {
        let queue = PendingQueue::new(3);
        let metrics = Metrics::default();
        let timeout = Duration::from_secs(60);
        let (live, live_ticket) = waiting_job(1, Instant::now());
        let (cancelled, cancelled_ticket) = waiting_job(2, Instant::now());
        let (expired, expired_ticket) = waiting_job(3, Instant::now() - Duration::from_secs(120));
        queue.push(live, &metrics, timeout).unwrap();
        queue.push(cancelled, &metrics, timeout).unwrap();
        queue.push(expired, &metrics, timeout).unwrap();
        cancelled_ticket.cancel.store(true, Ordering::Release);

        queue.close(&metrics, timeout);
        queue.close(&metrics, timeout);

        for (ticket, code) in [
            (live_ticket, "worker_lost"),
            (cancelled_ticket, "cancelled"),
            (expired_ticket, "deadline_exceeded"),
        ] {
            assert_eq!(ticket.begin.await.unwrap().err().unwrap().code, code);
        }
        let (later, _later_ticket) = waiting_job(4, Instant::now());
        assert_eq!(
            queue.push(later, &metrics, timeout).unwrap_err().status,
            503
        );
        assert!(queue.recv(&metrics).await.is_none());
        assert_eq!(metrics.queued.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.cancelled.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.expired.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.failed.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn waiting_receiver_wakes_for_admission_and_close() {
        let queue = PendingQueue::new(1);
        let metrics = Metrics::default();
        let timeout = Duration::from_secs(60);
        let receive = queue.recv(&metrics);
        tokio::pin!(receive);
        assert!(futures_util::poll!(&mut receive).is_pending());
        let (job, _ticket) = waiting_job(1, Instant::now());

        queue.push(job, &metrics, timeout).unwrap();

        assert!(tokio::time::timeout(Duration::from_secs(1), &mut receive)
            .await
            .unwrap()
            .is_some());
        let receive = queue.recv(&metrics);
        tokio::pin!(receive);
        assert!(futures_util::poll!(&mut receive).is_pending());

        queue.close(&metrics, timeout);

        assert!(tokio::time::timeout(Duration::from_secs(1), receive)
            .await
            .unwrap()
            .is_none());
        assert_eq!(metrics.queued.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn closing_during_admission_never_leaves_an_unsettled_waiter() {
        for index in 1..=16 {
            let queue = PendingQueue::new(1);
            let metrics = Metrics::default();
            let timeout = Duration::from_secs(60);
            let (job, ticket) = waiting_job(index, Instant::now());
            let start = std::sync::Barrier::new(2);
            let accepted = std::thread::scope(|threads| {
                let admission = threads.spawn(|| {
                    start.wait();
                    queue.push(job, &metrics, timeout).is_ok()
                });
                start.wait();
                queue.close(&metrics, timeout);
                admission.join().unwrap()
            });

            let outcome = ticket.begin.await;
            if accepted {
                assert_eq!(outcome.unwrap().err().unwrap().code, "worker_lost");
            } else {
                assert!(outcome.is_err());
            }
            assert!(queue.recv(&metrics).await.is_none());
            assert_eq!(metrics.queued.load(Ordering::Relaxed), 0);
            assert_eq!(metrics.failed.load(Ordering::Relaxed), u64::from(accepted));
        }
    }
}
