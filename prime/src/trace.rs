//! Default-off runtime trace hooks for Prime reactor wake attribution.
//!
//! This module is intentionally feature-gated behind
//! `runtime-prime-reactor-trace`. It uses `Instant::now()` and global atomics,
//! so it is for diagnostics and benchmarks only, not for the production hot
//! path.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static EPOCH: OnceLock<Instant> = OnceLock::new();

static ACTIVE: AtomicU64 = AtomicU64::new(0);
static READ_PENDING_AT: AtomicU64 = AtomicU64::new(0);
static AFTER_TICK_AT: AtomicU64 = AtomicU64::new(0);
static TIMER_DONE_AT: AtomicU64 = AtomicU64::new(0);
static SPIN_DONE_AT: AtomicU64 = AtomicU64::new(0);
static ARM_WAKEUP_AT: AtomicU64 = AtomicU64::new(0);
static RECHECK_DONE_AT: AtomicU64 = AtomicU64::new(0);
static FIRST_TURN_ENTER_AT: AtomicU64 = AtomicU64::new(0);
static TURN_ENTER_AT: AtomicU64 = AtomicU64::new(0);
static EVENT_READY_AT: AtomicU64 = AtomicU64::new(0);
static WAKER_START_AT: AtomicU64 = AtomicU64::new(0);
static WAKER_END_AT: AtomicU64 = AtomicU64::new(0);
static READY_PUSH_AT: AtomicU64 = AtomicU64::new(0);
static TURN_EXIT_AT: AtomicU64 = AtomicU64::new(0);
static TASK_POLL_START_AT: AtomicU64 = AtomicU64::new(0);

static CYCLES: AtomicU64 = AtomicU64::new(0);
static MISSED: AtomicU64 = AtomicU64::new(0);
static MISSING_TURN_ENTER: AtomicU64 = AtomicU64::new(0);
static MISSING_EVENT_READY: AtomicU64 = AtomicU64::new(0);
static MISSING_WAKER_START: AtomicU64 = AtomicU64::new(0);
static MISSING_WAKER_END: AtomicU64 = AtomicU64::new(0);
static MISSING_READY_PUSH: AtomicU64 = AtomicU64::new(0);
static MISSING_TURN_EXIT: AtomicU64 = AtomicU64::new(0);
static MISSING_TASK_POLL_START: AtomicU64 = AtomicU64::new(0);
static MISSING_WORKER_PHASE: AtomicU64 = AtomicU64::new(0);
static WORKER_PHASE_CYCLES: AtomicU64 = AtomicU64::new(0);
static PENDING_TO_AFTER_TICK_NS: AtomicU64 = AtomicU64::new(0);
static AFTER_TICK_TO_TIMER_DONE_NS: AtomicU64 = AtomicU64::new(0);
static TIMER_DONE_TO_SPIN_DONE_NS: AtomicU64 = AtomicU64::new(0);
static SPIN_DONE_TO_ARM_WAKEUP_NS: AtomicU64 = AtomicU64::new(0);
static ARM_WAKEUP_TO_RECHECK_DONE_NS: AtomicU64 = AtomicU64::new(0);
static RECHECK_DONE_TO_TURN_ENTER_NS: AtomicU64 = AtomicU64::new(0);
static RECHECK_CONTINUES: AtomicU64 = AtomicU64::new(0);
static RECHECK_INBOX_DRAINED: AtomicU64 = AtomicU64::new(0);
static RECHECK_POLLED: AtomicU64 = AtomicU64::new(0);
static RECHECK_FIRED: AtomicU64 = AtomicU64::new(0);
static TURN_ENTERS: AtomicU64 = AtomicU64::new(0);
static PRE_EVENT_EMPTY_TURNS: AtomicU64 = AtomicU64::new(0);
static PRE_EVENT_NONREAD_FIRED: AtomicU64 = AtomicU64::new(0);
static PRE_EVENT_TIMEOUT_TURNS: AtomicU64 = AtomicU64::new(0);
static PRE_EVENT_WAKEUP_EVENTS: AtomicU64 = AtomicU64::new(0);
static PRE_EVENT_IGNORED_READ: AtomicU64 = AtomicU64::new(0);
static PRE_EVENT_IGNORED_WRITE: AtomicU64 = AtomicU64::new(0);
static PRE_EVENT_STALE: AtomicU64 = AtomicU64::new(0);
static PRE_EVENT_UNKNOWN: AtomicU64 = AtomicU64::new(0);
static PENDING_TO_FIRST_TURN_ENTER_NS: AtomicU64 = AtomicU64::new(0);
static FIRST_TURN_ENTER_TO_EVENT_NS: AtomicU64 = AtomicU64::new(0);
static FIRST_TO_LAST_TURN_ENTER_NS: AtomicU64 = AtomicU64::new(0);
static PENDING_TO_TURN_ENTER_NS: AtomicU64 = AtomicU64::new(0);
static TURN_ENTER_TO_EVENT_NS: AtomicU64 = AtomicU64::new(0);
static EVENT_TO_WAKER_START_NS: AtomicU64 = AtomicU64::new(0);
static WAKER_NS: AtomicU64 = AtomicU64::new(0);
static WAKER_TO_READY_PUSH_NS: AtomicU64 = AtomicU64::new(0);
static READY_PUSH_TO_TURN_EXIT_NS: AtomicU64 = AtomicU64::new(0);
static TURN_EXIT_TO_TASK_POLL_NS: AtomicU64 = AtomicU64::new(0);
static TASK_POLL_TO_READ_READY_NS: AtomicU64 = AtomicU64::new(0);
static PENDING_TO_READ_READY_NS: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Default)]
pub struct Snapshot {
    pub cycles: u64,
    pub missed: u64,
    pub missing_turn_enter: u64,
    pub missing_event_ready: u64,
    pub missing_waker_start: u64,
    pub missing_waker_end: u64,
    pub missing_ready_push: u64,
    pub missing_turn_exit: u64,
    pub missing_task_poll_start: u64,
    pub missing_worker_phase: u64,
    pub worker_phase_cycles: u64,
    pub pending_to_after_tick_ns: u64,
    pub after_tick_to_timer_done_ns: u64,
    pub timer_done_to_spin_done_ns: u64,
    pub spin_done_to_arm_wakeup_ns: u64,
    pub arm_wakeup_to_recheck_done_ns: u64,
    pub recheck_done_to_turn_enter_ns: u64,
    pub recheck_continues: u64,
    pub recheck_inbox_drained: u64,
    pub recheck_polled: u64,
    pub recheck_fired: u64,
    pub turn_enters: u64,
    pub pre_event_empty_turns: u64,
    pub pre_event_nonread_fired: u64,
    pub pre_event_timeout_turns: u64,
    pub pre_event_wakeup_events: u64,
    pub pre_event_ignored_read: u64,
    pub pre_event_ignored_write: u64,
    pub pre_event_stale: u64,
    pub pre_event_unknown: u64,
    pub pending_to_first_turn_enter_ns: u64,
    pub first_turn_enter_to_event_ns: u64,
    pub first_to_last_turn_enter_ns: u64,
    pub pending_to_turn_enter_ns: u64,
    pub turn_enter_to_event_ns: u64,
    pub event_to_waker_start_ns: u64,
    pub waker_ns: u64,
    pub waker_to_ready_push_ns: u64,
    pub ready_push_to_turn_exit_ns: u64,
    pub turn_exit_to_task_poll_ns: u64,
    pub task_poll_to_read_ready_ns: u64,
    pub pending_to_read_ready_ns: u64,
}

#[inline]
fn now_ns() -> u64 {
    let epoch = EPOCH.get_or_init(Instant::now);
    epoch.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

#[inline]
fn load(cell: &AtomicU64) -> u64 {
    cell.load(Ordering::Acquire)
}

#[inline]
fn store_once(cell: &AtomicU64) {
    let now = now_ns();
    let _ = cell.compare_exchange(0, now, Ordering::AcqRel, Ordering::Acquire);
}

fn reset_current_cycle() {
    READ_PENDING_AT.store(0, Ordering::Release);
    AFTER_TICK_AT.store(0, Ordering::Release);
    TIMER_DONE_AT.store(0, Ordering::Release);
    SPIN_DONE_AT.store(0, Ordering::Release);
    ARM_WAKEUP_AT.store(0, Ordering::Release);
    RECHECK_DONE_AT.store(0, Ordering::Release);
    FIRST_TURN_ENTER_AT.store(0, Ordering::Release);
    TURN_ENTER_AT.store(0, Ordering::Release);
    EVENT_READY_AT.store(0, Ordering::Release);
    WAKER_START_AT.store(0, Ordering::Release);
    WAKER_END_AT.store(0, Ordering::Release);
    READY_PUSH_AT.store(0, Ordering::Release);
    TURN_EXIT_AT.store(0, Ordering::Release);
    TASK_POLL_START_AT.store(0, Ordering::Release);
}

pub fn reset() {
    ACTIVE.store(0, Ordering::Release);
    reset_current_cycle();
    CYCLES.store(0, Ordering::Release);
    MISSED.store(0, Ordering::Release);
    MISSING_TURN_ENTER.store(0, Ordering::Release);
    MISSING_EVENT_READY.store(0, Ordering::Release);
    MISSING_WAKER_START.store(0, Ordering::Release);
    MISSING_WAKER_END.store(0, Ordering::Release);
    MISSING_READY_PUSH.store(0, Ordering::Release);
    MISSING_TURN_EXIT.store(0, Ordering::Release);
    MISSING_TASK_POLL_START.store(0, Ordering::Release);
    MISSING_WORKER_PHASE.store(0, Ordering::Release);
    WORKER_PHASE_CYCLES.store(0, Ordering::Release);
    PENDING_TO_AFTER_TICK_NS.store(0, Ordering::Release);
    AFTER_TICK_TO_TIMER_DONE_NS.store(0, Ordering::Release);
    TIMER_DONE_TO_SPIN_DONE_NS.store(0, Ordering::Release);
    SPIN_DONE_TO_ARM_WAKEUP_NS.store(0, Ordering::Release);
    ARM_WAKEUP_TO_RECHECK_DONE_NS.store(0, Ordering::Release);
    RECHECK_DONE_TO_TURN_ENTER_NS.store(0, Ordering::Release);
    RECHECK_CONTINUES.store(0, Ordering::Release);
    RECHECK_INBOX_DRAINED.store(0, Ordering::Release);
    RECHECK_POLLED.store(0, Ordering::Release);
    RECHECK_FIRED.store(0, Ordering::Release);
    TURN_ENTERS.store(0, Ordering::Release);
    PRE_EVENT_EMPTY_TURNS.store(0, Ordering::Release);
    PRE_EVENT_NONREAD_FIRED.store(0, Ordering::Release);
    PRE_EVENT_TIMEOUT_TURNS.store(0, Ordering::Release);
    PRE_EVENT_WAKEUP_EVENTS.store(0, Ordering::Release);
    PRE_EVENT_IGNORED_READ.store(0, Ordering::Release);
    PRE_EVENT_IGNORED_WRITE.store(0, Ordering::Release);
    PRE_EVENT_STALE.store(0, Ordering::Release);
    PRE_EVENT_UNKNOWN.store(0, Ordering::Release);
    PENDING_TO_FIRST_TURN_ENTER_NS.store(0, Ordering::Release);
    FIRST_TURN_ENTER_TO_EVENT_NS.store(0, Ordering::Release);
    FIRST_TO_LAST_TURN_ENTER_NS.store(0, Ordering::Release);
    PENDING_TO_TURN_ENTER_NS.store(0, Ordering::Release);
    TURN_ENTER_TO_EVENT_NS.store(0, Ordering::Release);
    EVENT_TO_WAKER_START_NS.store(0, Ordering::Release);
    WAKER_NS.store(0, Ordering::Release);
    WAKER_TO_READY_PUSH_NS.store(0, Ordering::Release);
    READY_PUSH_TO_TURN_EXIT_NS.store(0, Ordering::Release);
    TURN_EXIT_TO_TASK_POLL_NS.store(0, Ordering::Release);
    TASK_POLL_TO_READ_READY_NS.store(0, Ordering::Release);
    PENDING_TO_READ_READY_NS.store(0, Ordering::Release);
}

pub fn snapshot() -> Snapshot {
    Snapshot {
        cycles: load(&CYCLES),
        missed: load(&MISSED),
        missing_turn_enter: load(&MISSING_TURN_ENTER),
        missing_event_ready: load(&MISSING_EVENT_READY),
        missing_waker_start: load(&MISSING_WAKER_START),
        missing_waker_end: load(&MISSING_WAKER_END),
        missing_ready_push: load(&MISSING_READY_PUSH),
        missing_turn_exit: load(&MISSING_TURN_EXIT),
        missing_task_poll_start: load(&MISSING_TASK_POLL_START),
        missing_worker_phase: load(&MISSING_WORKER_PHASE),
        worker_phase_cycles: load(&WORKER_PHASE_CYCLES),
        pending_to_after_tick_ns: load(&PENDING_TO_AFTER_TICK_NS),
        after_tick_to_timer_done_ns: load(&AFTER_TICK_TO_TIMER_DONE_NS),
        timer_done_to_spin_done_ns: load(&TIMER_DONE_TO_SPIN_DONE_NS),
        spin_done_to_arm_wakeup_ns: load(&SPIN_DONE_TO_ARM_WAKEUP_NS),
        arm_wakeup_to_recheck_done_ns: load(&ARM_WAKEUP_TO_RECHECK_DONE_NS),
        recheck_done_to_turn_enter_ns: load(&RECHECK_DONE_TO_TURN_ENTER_NS),
        recheck_continues: load(&RECHECK_CONTINUES),
        recheck_inbox_drained: load(&RECHECK_INBOX_DRAINED),
        recheck_polled: load(&RECHECK_POLLED),
        recheck_fired: load(&RECHECK_FIRED),
        turn_enters: load(&TURN_ENTERS),
        pre_event_empty_turns: load(&PRE_EVENT_EMPTY_TURNS),
        pre_event_nonread_fired: load(&PRE_EVENT_NONREAD_FIRED),
        pre_event_timeout_turns: load(&PRE_EVENT_TIMEOUT_TURNS),
        pre_event_wakeup_events: load(&PRE_EVENT_WAKEUP_EVENTS),
        pre_event_ignored_read: load(&PRE_EVENT_IGNORED_READ),
        pre_event_ignored_write: load(&PRE_EVENT_IGNORED_WRITE),
        pre_event_stale: load(&PRE_EVENT_STALE),
        pre_event_unknown: load(&PRE_EVENT_UNKNOWN),
        pending_to_first_turn_enter_ns: load(&PENDING_TO_FIRST_TURN_ENTER_NS),
        first_turn_enter_to_event_ns: load(&FIRST_TURN_ENTER_TO_EVENT_NS),
        first_to_last_turn_enter_ns: load(&FIRST_TO_LAST_TURN_ENTER_NS),
        pending_to_turn_enter_ns: load(&PENDING_TO_TURN_ENTER_NS),
        turn_enter_to_event_ns: load(&TURN_ENTER_TO_EVENT_NS),
        event_to_waker_start_ns: load(&EVENT_TO_WAKER_START_NS),
        waker_ns: load(&WAKER_NS),
        waker_to_ready_push_ns: load(&WAKER_TO_READY_PUSH_NS),
        ready_push_to_turn_exit_ns: load(&READY_PUSH_TO_TURN_EXIT_NS),
        turn_exit_to_task_poll_ns: load(&TURN_EXIT_TO_TASK_POLL_NS),
        task_poll_to_read_ready_ns: load(&TASK_POLL_TO_READ_READY_NS),
        pending_to_read_ready_ns: load(&PENDING_TO_READ_READY_NS),
    }
}

pub fn record_read_pending() {
    if ACTIVE
        .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        reset_current_cycle();
        READ_PENDING_AT.store(now_ns(), Ordering::Release);
    }
}

pub fn record_after_tick() {
    if load(&ACTIVE) != 0 {
        store_once(&AFTER_TICK_AT);
    }
}

pub fn record_timer_done() {
    if load(&ACTIVE) != 0 {
        store_once(&TIMER_DONE_AT);
    }
}

pub fn record_spin_done() {
    if load(&ACTIVE) != 0 {
        store_once(&SPIN_DONE_AT);
    }
}

pub fn record_arm_wakeup() {
    if load(&ACTIVE) != 0 {
        store_once(&ARM_WAKEUP_AT);
    }
}

pub fn record_recheck_done() {
    if load(&ACTIVE) != 0 {
        store_once(&RECHECK_DONE_AT);
    }
}

pub fn record_recheck_continue(inbox_drained: usize, polled: usize, fired: usize) {
    if load(&ACTIVE) != 0 {
        RECHECK_CONTINUES.fetch_add(1, Ordering::AcqRel);
        RECHECK_INBOX_DRAINED.fetch_add(inbox_drained as u64, Ordering::AcqRel);
        RECHECK_POLLED.fetch_add(polled as u64, Ordering::AcqRel);
        RECHECK_FIRED.fetch_add(fired as u64, Ordering::AcqRel);
    }
}

pub fn record_reactor_timeout() {
    if load(&ACTIVE) != 0 && load(&EVENT_READY_AT) == 0 {
        PRE_EVENT_TIMEOUT_TURNS.fetch_add(1, Ordering::AcqRel);
    }
}

pub fn record_reactor_wakeup_event() {
    if load(&ACTIVE) != 0 && load(&EVENT_READY_AT) == 0 {
        PRE_EVENT_WAKEUP_EVENTS.fetch_add(1, Ordering::AcqRel);
    }
}

pub fn record_reactor_ignored_read() {
    if load(&ACTIVE) != 0 && load(&EVENT_READY_AT) == 0 {
        PRE_EVENT_IGNORED_READ.fetch_add(1, Ordering::AcqRel);
    }
}

pub fn record_reactor_ignored_write() {
    if load(&ACTIVE) != 0 && load(&EVENT_READY_AT) == 0 {
        PRE_EVENT_IGNORED_WRITE.fetch_add(1, Ordering::AcqRel);
    }
}

pub fn record_reactor_stale_event() {
    if load(&ACTIVE) != 0 && load(&EVENT_READY_AT) == 0 {
        PRE_EVENT_STALE.fetch_add(1, Ordering::AcqRel);
    }
}

pub fn record_reactor_unknown_event() {
    if load(&ACTIVE) != 0 && load(&EVENT_READY_AT) == 0 {
        PRE_EVENT_UNKNOWN.fetch_add(1, Ordering::AcqRel);
    }
}

pub fn record_turn_enter() {
    if load(&ACTIVE) != 0 {
        if load(&EVENT_READY_AT) == 0 {
            let now = now_ns();
            let _ =
                FIRST_TURN_ENTER_AT.compare_exchange(0, now, Ordering::AcqRel, Ordering::Acquire);
            TURN_ENTER_AT.store(now, Ordering::Release);
            TURN_ENTERS.fetch_add(1, Ordering::AcqRel);
        }
    }
}

pub fn record_event_ready() {
    if load(&ACTIVE) != 0 {
        store_once(&EVENT_READY_AT);
    }
}

pub fn record_waker_start() {
    if load(&ACTIVE) != 0 {
        store_once(&WAKER_START_AT);
    }
}

pub fn record_waker_end() {
    if load(&ACTIVE) != 0 {
        store_once(&WAKER_END_AT);
    }
}

pub fn record_ready_push() {
    if load(&ACTIVE) != 0 {
        store_once(&READY_PUSH_AT);
    }
}

pub fn record_turn_exit(fired: usize) {
    if load(&ACTIVE) != 0 {
        if load(&EVENT_READY_AT) != 0 {
            store_once(&TURN_EXIT_AT);
        } else if fired == 0 {
            PRE_EVENT_EMPTY_TURNS.fetch_add(1, Ordering::AcqRel);
        } else {
            PRE_EVENT_NONREAD_FIRED.fetch_add(fired as u64, Ordering::AcqRel);
        }
    }
}

pub fn record_task_poll_start() {
    if load(&ACTIVE) != 0 {
        store_once(&TASK_POLL_START_AT);
    }
}

pub fn record_read_ready() {
    if ACTIVE.swap(0, Ordering::AcqRel) == 0 {
        return;
    }

    let ready_at = now_ns();
    let pending = load(&READ_PENDING_AT);
    let after_tick = load(&AFTER_TICK_AT);
    let timer_done = load(&TIMER_DONE_AT);
    let spin_done = load(&SPIN_DONE_AT);
    let arm_wakeup = load(&ARM_WAKEUP_AT);
    let recheck_done = load(&RECHECK_DONE_AT);
    let first_turn_enter = load(&FIRST_TURN_ENTER_AT);
    let turn_enter = load(&TURN_ENTER_AT);
    let event_ready = load(&EVENT_READY_AT);
    let waker_start = load(&WAKER_START_AT);
    let waker_end = load(&WAKER_END_AT);
    let ready_push = load(&READY_PUSH_AT);
    let turn_exit = load(&TURN_EXIT_AT);
    let poll_start = load(&TASK_POLL_START_AT);

    let mut complete = true;
    if first_turn_enter == 0 || first_turn_enter < pending {
        MISSING_TURN_ENTER.fetch_add(1, Ordering::AcqRel);
        complete = false;
    }
    if turn_enter == 0 || turn_enter < first_turn_enter {
        MISSING_TURN_ENTER.fetch_add(1, Ordering::AcqRel);
        complete = false;
    }
    if event_ready == 0 || event_ready < turn_enter {
        MISSING_EVENT_READY.fetch_add(1, Ordering::AcqRel);
        complete = false;
    }
    if waker_start == 0 || waker_start < event_ready {
        MISSING_WAKER_START.fetch_add(1, Ordering::AcqRel);
        complete = false;
    }
    if waker_end == 0 || waker_end < waker_start {
        MISSING_WAKER_END.fetch_add(1, Ordering::AcqRel);
        complete = false;
    }
    if ready_push == 0 || ready_push < waker_start {
        MISSING_READY_PUSH.fetch_add(1, Ordering::AcqRel);
        complete = false;
    }
    if turn_exit == 0 || turn_exit < ready_push {
        MISSING_TURN_EXIT.fetch_add(1, Ordering::AcqRel);
        complete = false;
    }
    if poll_start == 0 || poll_start < turn_exit {
        MISSING_TASK_POLL_START.fetch_add(1, Ordering::AcqRel);
        complete = false;
    }
    if pending == 0 || ready_at < pending {
        complete = false;
    }

    if !complete {
        MISSED.fetch_add(1, Ordering::AcqRel);
        return;
    }

    PENDING_TO_FIRST_TURN_ENTER_NS.fetch_add(first_turn_enter - pending, Ordering::AcqRel);
    FIRST_TURN_ENTER_TO_EVENT_NS.fetch_add(event_ready - first_turn_enter, Ordering::AcqRel);
    FIRST_TO_LAST_TURN_ENTER_NS.fetch_add(turn_enter - first_turn_enter, Ordering::AcqRel);
    PENDING_TO_TURN_ENTER_NS.fetch_add(turn_enter - pending, Ordering::AcqRel);
    TURN_ENTER_TO_EVENT_NS.fetch_add(event_ready - turn_enter, Ordering::AcqRel);
    EVENT_TO_WAKER_START_NS.fetch_add(waker_start - event_ready, Ordering::AcqRel);
    WAKER_NS.fetch_add(waker_end - waker_start, Ordering::AcqRel);
    WAKER_TO_READY_PUSH_NS.fetch_add(ready_push - waker_start, Ordering::AcqRel);
    READY_PUSH_TO_TURN_EXIT_NS.fetch_add(turn_exit - ready_push, Ordering::AcqRel);
    TURN_EXIT_TO_TASK_POLL_NS.fetch_add(poll_start - turn_exit, Ordering::AcqRel);
    TASK_POLL_TO_READ_READY_NS.fetch_add(ready_at - poll_start, Ordering::AcqRel);
    PENDING_TO_READ_READY_NS.fetch_add(ready_at - pending, Ordering::AcqRel);
    CYCLES.fetch_add(1, Ordering::AcqRel);

    if after_tick != 0
        && after_tick >= pending
        && timer_done >= after_tick
        && spin_done >= timer_done
        && arm_wakeup >= spin_done
        && recheck_done >= arm_wakeup
        && first_turn_enter >= recheck_done
    {
        PENDING_TO_AFTER_TICK_NS.fetch_add(after_tick - pending, Ordering::AcqRel);
        AFTER_TICK_TO_TIMER_DONE_NS.fetch_add(timer_done - after_tick, Ordering::AcqRel);
        TIMER_DONE_TO_SPIN_DONE_NS.fetch_add(spin_done - timer_done, Ordering::AcqRel);
        SPIN_DONE_TO_ARM_WAKEUP_NS.fetch_add(arm_wakeup - spin_done, Ordering::AcqRel);
        ARM_WAKEUP_TO_RECHECK_DONE_NS.fetch_add(recheck_done - arm_wakeup, Ordering::AcqRel);
        RECHECK_DONE_TO_TURN_ENTER_NS.fetch_add(first_turn_enter - recheck_done, Ordering::AcqRel);
        WORKER_PHASE_CYCLES.fetch_add(1, Ordering::AcqRel);
    } else {
        MISSING_WORKER_PHASE.fetch_add(1, Ordering::AcqRel);
    }
}
