//! Host fallback driver -- one timer-owner thread per driver instance.
//!
//! Polling threads never spawn: [`Driver::schedule_wake`], `refresh_wake` and
//! `cancel_wake` push a command onto a lock-free [`Ring`] (the Vyukov MPMC ring
//! in `proxima_core::ring`) and return. The single owner thread drains the ring
//! into a fixed-capacity indexed heap (O(log n) insert/remove/pop, O(1) keyed
//! lookup for dedup and cancel), wakes what is due, and parks until the
//! earliest deadline or a submission that beats it.
//!
//! Capacities come from `proxima-core.toml` `[timer]` (`submission_cap`,
//! `table_cap`, `defer_cap`). Overflow policy: a registration the main table
//! cannot hold waits in the deferred table (counted, [`StdThreadDriver::deferred_registrations`])
//! and is promoted as slots free -- its task is not woken, so it cannot spin.
//! Only past table_cap + defer_cap concurrent timers, or a full submission
//! ring, is the task woken at once ([`StdThreadDriver::table_overflows`],
//! [`StdThreadDriver::submission_overflows`]) so it re-polls instead of hanging.
//!
//! The right driver under heavy timer workloads on a per-core runtime is
//! `prime-wheel`; this one keeps `proxima_core::time` usable on a plain host.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering, fence};
use std::sync::{Arc, OnceLock};
use std::task::Waker;
use std::thread::{self, Thread};
use std::time::Duration;

use crate::ring::Ring;
use crate::sized::{TIMER_DEFER_CAP, TIMER_SUBMISSION_CAP, TIMER_TABLE_CAP};
use crate::time::drivers::deadline_table::{DeadlineTable, Entry};
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

struct Shared {
    ring: Ring<Command>,
    owner: OnceLock<Thread>,
    earliest_nanos: AtomicU64,
    live: AtomicUsize,
    closed: AtomicBool,
    submission_overflows: AtomicU64,
    table_overflows: AtomicU64,
    deferred_registrations: AtomicU64,
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
            deferred_registrations: AtomicU64::new(0),
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
        let mut table = DeadlineTable::with_capacity(TIMER_TABLE_CAP);
        let mut deferred = DeadlineTable::with_capacity(TIMER_DEFER_CAP);
        while !self.closed.load(Ordering::Acquire) {
            while let Some(command) = self.ring.dequeue() {
                self.apply(&mut table, &mut deferred, command);
            }
            let now = u64::try_from(epoch.elapsed().as_nanos()).unwrap_or(u64::MAX);
            fire_due(&mut table, now);
            fire_due(&mut deferred, now);
            promote(&mut table, &mut deferred);
            self.live.store(table.len() + deferred.len(), Ordering::Release);
            let earliest = [table.earliest(), deferred.earliest()]
                .into_iter()
                .flatten()
                .map(nanos)
                .min()
                .unwrap_or(u64::MAX);
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

    fn apply(&self, table: &mut DeadlineTable, deferred: &mut DeadlineTable, command: Command) {
        match command {
            Command::Schedule { deadline, waker } => {
                let task = waker.data().addr();
                self.register(table, deferred, deadline, task, waker, true);
            }
            Command::Refresh { deadline, waker } => {
                let task = waker.data().addr();
                self.register(table, deferred, deadline, task, waker, false);
            }
            Command::Cancel { deadline, task } => {
                for tier in [table, deferred] {
                    if let Some(slot) = tier.find(deadline, task) {
                        release(tier, slot);
                        return;
                    }
                }
            }
        }
    }

    // policy: a registration the main table cannot hold waits in the deferred
    // table and is promoted on the owner's next pass as slots free, so its task
    // is not woken (no re-poll churn). only when both tiers are full does the
    // task get an immediate, counted wake.
    fn register(
        &self,
        table: &mut DeadlineTable,
        deferred: &mut DeadlineTable,
        deadline: Instant,
        task: usize,
        waker: Waker,
        counted: bool,
    ) {
        for tier in [&mut *table, &mut *deferred] {
            if let Some(slot) = tier.find(deadline, task) {
                if let Some(entry) = tier.entry_mut(slot) {
                    entry.refs = entry.refs.saturating_add(u32::from(counted));
                    entry.rearm(waker);
                }
                return;
            }
        }
        let Err(entry) = table.insert(Entry::new(deadline, task, waker)) else {
            return;
        };
        match deferred.insert(entry) {
            Ok(()) => {
                self.deferred_registrations.fetch_add(1, Ordering::Relaxed);
            }
            Err(entry) => {
                self.table_overflows.fetch_add(1, Ordering::Relaxed);
                entry.waker.wake();
            }
        }
    }
}

fn release(table: &mut DeadlineTable, slot: u32) {
    let last = table.entry_mut(slot).is_none_or(|entry| {
        entry.refs = entry.refs.saturating_sub(1);
        entry.refs == 0
    });
    if last {
        table.remove(slot);
    }
}

fn promote(table: &mut DeadlineTable, deferred: &mut DeadlineTable) {
    while table.has_room() {
        let Some(entry) = deferred.pop_earliest() else {
            return;
        };
        if let Err(entry) = table.insert(entry) {
            let _ = deferred.insert(entry);
            return;
        }
    }
}

fn fire_due(table: &mut DeadlineTable, now: u64) {
    while table.earliest().is_some_and(|deadline| nanos(deadline) <= now) {
        if let Some(entry) = table.pop_earliest() {
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

    /// Registrations that waited in the deferred table because the main table
    /// was full; their tasks were not woken early.
    #[must_use]
    pub fn deferred_registrations(&self) -> u64 {
        self.shared.get().map_or(0, |shared| {
            shared.deferred_registrations.load(Ordering::Relaxed)
        })
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
