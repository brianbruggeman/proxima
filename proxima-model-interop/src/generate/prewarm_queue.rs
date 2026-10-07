//! Where an answer leaves its end-of-answer prewarm until the device is idle
//! (`proxima-tensor/specs/prefix-cache-reuse/SPEC.md`).
//!
//! A request that has stored its answer submits the prefix the next turn will
//! share ([`PrewarmQueue::submit`]) and returns; it never runs the prefill.
//! Whoever drives the queue does: the worker thread
//! [`LoadedModel::with_prewarm_worker`](super::LoadedModel::with_prewarm_worker)
//! scopes, or a caller polling
//! [`LoadedModel::run_pending_prewarm`](super::LoadedModel::run_pending_prewarm)
//! when it is idle. The slot holds one job and the newest replaces an older
//! one: a prefix whose answer has since been followed by another answer is the
//! one nobody will send.
//!
//! The mutex and condition variables here (`proxima_primitives::sync::blocking`) are the
//! "dedicated blocking worker" case of principle 21: they park the worker
//! thread the caller handed over, never a request and never a task. The request side takes the mutex for the
//! length of a pointer swap. Who may use the device while a job runs is
//! [`super::prewarm_gate`]'s decision, not this module's.

use proxima_primitives::sync::blocking::{Condvar, Mutex, MutexGuard};

use super::CacheKey;

/// The prefix a finished answer leaves to be prefilled: the prompt, the answer
/// and the registered turn-boundary suffix.
pub(super) struct PrewarmJob {
    pub(super) prefix: Vec<u32>,
    /// What the request that submitted it built its entry under; a worker
    /// whose own config derives another key would only fill rows the next
    /// request cannot reuse.
    pub(super) key: CacheKey,
    pub(super) forced_draft_width: Option<u16>,
}

struct QueueState {
    job: Option<PrewarmJob>,
    running: bool,
    worker_attached: bool,
    stopping: bool,
}

pub(super) struct PrewarmQueue {
    state: Mutex<QueueState>,
    work: Condvar,
    idle: Condvar,
}

/// A worker thread attached to the queue until this drops, which tells it to
/// stop whether the scope that attached it returned or unwound.
pub(super) struct WorkerAttachment<'queue> {
    queue: &'queue PrewarmQueue,
}

impl Drop for WorkerAttachment<'_> {
    fn drop(&mut self) {
        self.queue.lock().stopping = true;
        self.queue.work.notify_all();
    }
}

/// Held by the worker thread for as long as it runs: when the thread ends,
/// even by unwinding, nothing is left to run a queued job, so a caller
/// waiting for idle must stop waiting for one.
pub(super) struct WorkerLife<'queue> {
    queue: &'queue PrewarmQueue,
}

impl Drop for WorkerLife<'_> {
    fn drop(&mut self) {
        self.queue.lock().worker_attached = false;
        self.queue.idle.notify_all();
    }
}

impl PrewarmQueue {
    pub(super) const fn new() -> Self {
        Self {
            state: Mutex::new(QueueState {
                job: None,
                running: false,
                worker_attached: false,
                stopping: false,
            }),
            work: Condvar::new(),
            idle: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, QueueState> {
        self.state.lock()
    }

    /// Queues `job`, returning `true` when it replaced one nobody ran.
    pub(super) fn submit(&self, job: PrewarmJob) -> bool {
        let replaced = self.lock().job.replace(job).is_some();
        self.work.notify_one();
        replaced
    }

    /// Marks a worker as coming so [`Self::wait_idle`] knows a queued job
    /// will be served. The first guard stops the worker when it drops; the
    /// second moves into the worker thread.
    pub(super) fn attach_worker(&self) -> (WorkerAttachment<'_>, WorkerLife<'_>) {
        let mut state = self.lock();
        state.worker_attached = true;
        state.stopping = false;
        drop(state);
        (WorkerAttachment { queue: self }, WorkerLife { queue: self })
    }

    /// Blocks until a job is queued, `true`, or the worker is told to stop,
    /// `false`. The job is left in the slot for [`Self::run_next`].
    pub(super) fn wait_for_work(&self) -> bool {
        let mut state = self.lock();
        self.work
            .wait_while(&mut state, |held| held.job.is_none() && !held.stopping);
        state.job.is_some() && !state.stopping
    }

    /// Takes the queued job, if any, and runs `run` on it. The queue counts as
    /// busy for the whole call, including when `run` unwinds.
    pub(super) fn run_next<T>(&self, run: impl FnOnce(PrewarmJob) -> T) -> Option<T> {
        let mut state = self.lock();
        let job = state.job.take()?;
        state.running = true;
        drop(state);
        let _finished = FinishedJob { queue: self };
        Some(run(job))
    }

    /// Blocks until no job is running and none is queued for an attached
    /// worker. With no worker attached a queued job is left alone: nothing
    /// would ever run it, so waiting for it would never return.
    pub(super) fn wait_idle(&self) {
        let mut state = self.lock();
        self.idle.wait_while(&mut state, |held| {
            held.running || (held.job.is_some() && held.worker_attached)
        });
    }
}

struct FinishedJob<'queue> {
    queue: &'queue PrewarmQueue,
}

impl Drop for FinishedJob<'_> {
    fn drop(&mut self) {
        self.queue.lock().running = false;
        self.queue.idle.notify_all();
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::serving::ServingConfig;

    fn job(prefix: &[u32]) -> PrewarmJob {
        PrewarmJob {
            prefix: prefix.to_vec(),
            key: CacheKey::of(
                &ServingConfig::default(),
                false,
                crate::RopeScaling::None,
                0,
                0,
            ),
            forced_draft_width: None,
        }
    }

    #[test]
    fn a_newer_job_replaces_one_nobody_ran() {
        let queue = PrewarmQueue::new();

        let first_replaced = queue.submit(job(&[2, 105, 2364]));
        let second_replaced = queue.submit(job(&[2, 105, 9259]));
        let ran = queue.run_next(|job| job.prefix);

        assert!(!first_replaced);
        assert!(second_replaced);
        assert_eq!(ran, Some(vec![2, 105, 9259]));
        assert!(queue.run_next(|job| job.prefix).is_none());
    }

    #[test]
    fn a_running_job_keeps_the_queue_busy_until_it_returns() {
        let queue = PrewarmQueue::new();
        queue.submit(job(&[2, 105]));

        queue.run_next(|_| {
            assert!(queue.lock().running);
        });

        assert!(!queue.lock().running);
    }

    #[test]
    fn a_job_that_unwinds_still_frees_the_queue() {
        let queue = PrewarmQueue::new();
        queue.submit(job(&[2, 105]));

        let outcome = std::thread::scope(|scope| {
            scope
                .spawn(|| queue.run_next(|_| panic!("prewarm died mid-chunk")))
                .join()
        });

        assert!(outcome.is_err());
        assert!(!queue.lock().running);
        queue.wait_idle();
    }

    #[test]
    fn a_waiting_worker_wakes_for_a_submitted_job_and_for_a_stop() {
        let queue = PrewarmQueue::new();
        let served = AtomicUsize::new(0);

        std::thread::scope(|scope| {
            let (attachment, life) = queue.attach_worker();
            let worker = scope.spawn(|| {
                let _life = life;
                while queue.wait_for_work() {
                    queue.run_next(|_| served.fetch_add(1, Ordering::SeqCst));
                }
            });
            queue.submit(job(&[2, 105, 2364]));
            queue.wait_idle();
            assert_eq!(served.load(Ordering::SeqCst), 1);
            drop(attachment);
            worker.join().expect("the worker stops when detached");
        });
    }

    #[test]
    fn idle_does_not_wait_on_a_queued_job_when_no_worker_would_run_it() {
        let queue = PrewarmQueue::new();
        queue.submit(job(&[2, 105]));

        queue.wait_idle();

        assert!(queue.run_next(|job| job.prefix).is_some());
    }

    #[test]
    fn a_worker_that_died_stops_idle_waiting_on_the_job_it_left_queued() {
        let queue = PrewarmQueue::new();
        let (attachment, life) = queue.attach_worker();
        queue.submit(job(&[2, 105, 2364]));

        let outcome = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let _life = life;
                    panic!("worker died before taking the job");
                })
                .join()
        });
        queue.wait_idle();

        assert!(outcome.is_err());
        assert!(queue.run_next(|job| job.prefix).is_some());
        drop(attachment);
    }

    #[test]
    fn a_stop_before_the_worker_waits_ends_it_without_a_job() {
        let queue = PrewarmQueue::new();
        drop(queue.attach_worker());

        assert!(!queue.wait_for_work());
    }
}
