//! Host fallback driver -- one timer-owner thread per driver instance.
//!
//! Polling threads never spawn: [`Driver::schedule_wake`], `refresh_wake` and
//! `cancel_wake` push a command onto a lock-free [`Ring`] (the Vyukov MPMC ring
//! in `proxima_core::ring`) and return. The single owner thread drains the ring
//! into a fixed-capacity table of deadlines sorted so the earliest sits last
//! (O(1) pop, O(log n) keyed lookup for dedup and cancel), wakes what is due,
//! and parks until the earliest deadline or a submission that beats it.
//!
//! Capacities come from `proxima-core.toml` `[timer]` (`submission_cap`,
//! `table_cap`). A full ring or table is a counted outcome
//! ([`StdThreadDriver::submission_overflows`], [`StdThreadDriver::table_overflows`]):
//! the waker is woken at once, so the task re-polls instead of hanging.
//!
//! The right driver under heavy timer workloads on a per-core runtime is
//! `prime-wheel`; this one keeps `proxima_core::time` usable on a plain host.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering, fence};
use std::sync::{Arc, OnceLock};
use std::task::Waker;
use std::thread::{self, Thread};
use std::time::Duration;

use crate::ring::Ring;
use crate::sized::{TIMER_SUBMISSION_CAP, TIMER_TABLE_CAP};
use crate::time::{Driver, Instant};

/// Singleton driver instance -- referenced by the static binding emitted
/// by `proxima-core`'s `build.rs` when `timer = "std-thread"`.
pub static DRIVER: StdThreadDriver = StdThreadDriver::new();

/// State for the host-thread driver. The epoch and the owner thread are created
/// lazily on first use so the driver remains construct-in-static-context.
pub struct StdThreadDriver {
    epoch: OnceLock<std::time::Instant>,
    shared: OnceLock<Arc<Shared>>,
}

enum Command {
    Schedule { deadline: Instant, waker: Waker },
    Refresh { deadline: Instant, waker: Waker },
    Cancel { deadline: Instant, task: usize },
}

struct Entry {
    deadline: Instant,
    task: usize,
    waker: Waker,
    refs: u32,
}

struct Shared {
    ring: Ring<Command>,
    owner: OnceLock<Thread>,
    earliest_nanos: AtomicU64,
    live: AtomicUsize,
    closed: AtomicBool,
    submission_overflows: AtomicU64,
    table_overflows: AtomicU64,
}

fn nanos(deadline: Instant) -> u64 {
    u64::try_from(deadline.into_monotonic().as_nanos()).unwrap_or(u64::MAX)
}

impl Command {
    fn deadline_nanos(&self) -> Option<u64> {
        match self {
            Self::Schedule { deadline, .. } | Self::Refresh { deadline, .. } => {
                Some(nanos(*deadline))
            }
            Self::Cancel { .. } => None,
        }
    }
}

impl Shared {
    fn start(epoch: std::time::Instant) -> Arc<Self> {
        let shared = Arc::new(Self {
            ring: Ring::with_capacity(TIMER_SUBMISSION_CAP),
            owner: OnceLock::new(),
            earliest_nanos: AtomicU64::new(u64::MAX),
            live: AtomicUsize::new(0),
            closed: AtomicBool::new(false),
            submission_overflows: AtomicU64::new(0),
            table_overflows: AtomicU64::new(0),
        });
        let owned = Arc::clone(&shared);
        let spawned = thread::Builder::new()
            .name("proxima-timer".into())
            .spawn(move || owned.run(epoch));
        if let Ok(handle) = spawned {
            let _ = shared.owner.set(handle.thread().clone());
        }
        shared
    }

    fn submit(&self, command: Command) {
        let Some(owner) = self.owner.get() else {
            return self.reject(command);
        };
        let deadline_nanos = command.deadline_nanos();
        if let Err(command) = self.ring.push(command) {
            return self.reject(command);
        }
        fence(Ordering::SeqCst);
        if deadline_nanos.is_some_and(|due| due < self.earliest_nanos.load(Ordering::SeqCst)) {
            owner.unpark();
        }
    }

    fn reject(&self, command: Command) {
        self.submission_overflows.fetch_add(1, Ordering::Relaxed);
        match command {
            Command::Schedule { waker, .. } | Command::Refresh { waker, .. } => waker.wake(),
            Command::Cancel { .. } => {}
        }
    }

    fn run(&self, epoch: std::time::Instant) {
        let mut table: Vec<Entry> = Vec::with_capacity(TIMER_TABLE_CAP);
        while !self.closed.load(Ordering::Acquire) {
            while let Some(command) = self.ring.dequeue() {
                self.apply(&mut table, command);
            }
            let now = u64::try_from(epoch.elapsed().as_nanos()).unwrap_or(u64::MAX);
            fire_due(&mut table, now);
            self.live.store(table.len(), Ordering::Release);
            let earliest = table.last().map_or(u64::MAX, |entry| nanos(entry.deadline));
            self.earliest_nanos.store(earliest, Ordering::SeqCst);
            fence(Ordering::SeqCst);
            if !self.ring.is_empty() {
                continue;
            }
            match earliest {
                u64::MAX => thread::park(),
                due => thread::park_timeout(Duration::from_nanos(due.saturating_sub(now))),
            }
        }
    }

    fn apply(&self, table: &mut Vec<Entry>, command: Command) {
        match command {
            Command::Schedule { deadline, waker } => {
                let task = waker.data().addr();
                match locate(table, deadline, task) {
                    Ok(index) => {
                        let entry = &mut table[index];
                        entry.refs = entry.refs.saturating_add(1);
                        entry.rearm(waker);
                    }
                    Err(index) => self.insert(table, index, deadline, task, waker),
                }
            }
            Command::Refresh { deadline, waker } => {
                let task = waker.data().addr();
                match locate(table, deadline, task) {
                    Ok(index) => table[index].rearm(waker),
                    Err(index) => self.insert(table, index, deadline, task, waker),
                }
            }
            Command::Cancel { deadline, task } => {
                if let Ok(index) = locate(table, deadline, task) {
                    release(table, index);
                }
            }
        }
    }

    fn insert(
        &self,
        table: &mut Vec<Entry>,
        index: usize,
        deadline: Instant,
        task: usize,
        waker: Waker,
    ) {
        if table.len() >= TIMER_TABLE_CAP {
            self.table_overflows.fetch_add(1, Ordering::Relaxed);
            return waker.wake();
        }
        table.insert(
            index,
            Entry {
                deadline,
                task,
                waker,
                refs: 1,
            },
        );
    }
}

impl Entry {
    fn rearm(&mut self, waker: Waker) {
        if !self.waker.will_wake(&waker) {
            self.waker = waker;
        }
    }
}

fn locate(table: &[Entry], deadline: Instant, task: usize) -> Result<usize, usize> {
    table.binary_search_by(|entry| (deadline, task).cmp(&(entry.deadline, entry.task)))
}

fn release(table: &mut Vec<Entry>, index: usize) {
    match table[index].refs {
        0 | 1 => {
            table.remove(index);
        }
        _ => table[index].refs -= 1,
    }
}

fn fire_due(table: &mut Vec<Entry>, now: u64) {
    while table.last().is_some_and(|entry| nanos(entry.deadline) <= now) {
        if let Some(entry) = table.pop() {
            entry.waker.wake();
        }
    }
}

impl StdThreadDriver {
    /// A driver with its own owner thread, started on first use.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            epoch: OnceLock::new(),
            shared: OnceLock::new(),
        }
    }

    fn epoch(&self) -> std::time::Instant {
        *self.epoch.get_or_init(std::time::Instant::now)
    }

    fn shared(&self) -> &Shared {
        self.shared
            .get_or_init(|| Shared::start(self.epoch()))
            .as_ref()
    }

    /// Commands refused because the submission ring was full (or the owner
    /// thread could not start); each one woke its task immediately instead.
    #[must_use]
    pub fn submission_overflows(&self) -> u64 {
        self.shared
            .get()
            .map_or(0, |shared| shared.submission_overflows.load(Ordering::Relaxed))
    }

    /// Registrations refused because the deadline table was full; each one woke
    /// its task immediately instead.
    #[must_use]
    pub fn table_overflows(&self) -> u64 {
        self.shared
            .get()
            .map_or(0, |shared| shared.table_overflows.load(Ordering::Relaxed))
    }

    /// Live (task, deadline) registrations as of the owner's last drain.
    #[must_use]
    pub fn pending_timers(&self) -> usize {
        self.shared
            .get()
            .map_or(0, |shared| shared.live.load(Ordering::Acquire))
    }
}

impl Default for StdThreadDriver {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for StdThreadDriver {
    fn drop(&mut self) {
        let Some(shared) = self.shared.get() else {
            return;
        };
        shared.closed.store(true, Ordering::Release);
        if let Some(owner) = shared.owner.get() {
            owner.unpark();
        }
    }
}

impl Driver for StdThreadDriver {
    fn now(&self) -> Instant {
        let elapsed = std::time::Instant::now().saturating_duration_since(self.epoch());
        Instant::from_monotonic(elapsed)
    }

    fn schedule_wake(&self, deadline: Instant, waker: Waker) {
        self.shared().submit(Command::Schedule { deadline, waker });
    }

    fn refresh_wake(&self, deadline: Instant, waker: &Waker) {
        self.shared().submit(Command::Refresh {
            deadline,
            waker: waker.clone(),
        });
    }

    fn cancel_wake(&self, deadline: Instant, task: usize) {
        self.shared().submit(Command::Cancel { deadline, task });
    }
}
