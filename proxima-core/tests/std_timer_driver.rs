#![cfg(feature = "time-driver-std-thread")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::task::{Context, Poll, Wake, Waker};
use std::time::Duration;

use proxima_core::time::drivers::std_thread::{DRIVER, StdThreadDriver};
use proxima_core::time::{Driver, Instant, sleep, timeout};
use proxima_test::alloc_count::{CountingAllocator, allocations};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

struct Notify {
    index: usize,
    sender: Sender<usize>,
}

impl Wake for Notify {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        let _ = self.sender.send(self.index);
    }
}

struct Counting(AtomicUsize);

impl Wake for Counting {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn notifier(index: usize, sender: &Sender<usize>) -> Waker {
    Waker::from(Arc::new(Notify {
        index,
        sender: sender.clone(),
    }))
}

#[cfg(target_os = "linux")]
fn thread_count() -> usize {
    std::fs::read_dir("/proc/self/task")
        .expect("procfs lists this process's threads")
        .count()
}

#[cfg(target_os = "macos")]
fn thread_count() -> usize {
    let mut info: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
    let size = i32::try_from(std::mem::size_of::<libc::proc_taskinfo>()).expect("struct size fits");
    let written = unsafe {
        libc::proc_pidinfo(
            libc::getpid(),
            libc::PROC_PIDTASKINFO,
            0,
            std::ptr::from_mut(&mut info).cast(),
            size,
        )
    };
    assert_eq!(written, size, "proc_pidinfo filled the task info");
    usize::try_from(info.pti_threadnum).expect("thread count is non-negative")
}

fn after(driver: &StdThreadDriver, delay: Duration) -> Instant {
    driver.now() + delay
}

fn wait_for(receiver: &Receiver<usize>) -> usize {
    receiver.recv().expect("owner thread delivers the wake")
}

#[test]
fn wakes_fire_in_deadline_order_not_submission_order() {
    let driver = StdThreadDriver::new();
    let (sender, receiver) = channel();
    let offsets_millis = [40_u64, 10, 30, 20, 50];

    for (index, offset) in offsets_millis.iter().enumerate() {
        let deadline = after(&driver, Duration::from_millis(*offset));
        driver.schedule_wake(deadline, notifier(index, &sender));
    }
    let fired: Vec<usize> = (0..offsets_millis.len()).map(|_| wait_for(&receiver)).collect();

    assert_eq!(fired, vec![1, 3, 2, 0, 4]);
}

#[test]
fn repeated_refresh_of_one_registration_wakes_once() {
    let driver = StdThreadDriver::new();
    let (sender, receiver) = channel();
    let counter = Arc::new(Counting(AtomicUsize::new(0)));
    let waker = Waker::from(Arc::clone(&counter));
    let deadline = after(&driver, Duration::from_millis(30));

    driver.schedule_wake(deadline, waker.clone());
    for _ in 0..1_000 {
        driver.refresh_wake(deadline, &waker);
    }
    driver.schedule_wake(after(&driver, Duration::from_millis(60)), notifier(7, &sender));
    wait_for(&receiver);

    assert_eq!(counter.0.load(Ordering::SeqCst), 1);
    assert_eq!(driver.pending_timers(), 0);
}

#[test]
fn cancelled_registration_never_wakes() {
    let driver = StdThreadDriver::new();
    let (sender, receiver) = channel();
    let counter = Arc::new(Counting(AtomicUsize::new(0)));
    let waker = Waker::from(Arc::clone(&counter));
    let deadline = after(&driver, Duration::from_millis(30));

    driver.schedule_wake(deadline, waker.clone());
    driver.cancel_wake(deadline, waker.data().addr());
    driver.schedule_wake(after(&driver, Duration::from_millis(60)), notifier(7, &sender));
    wait_for(&receiver);

    assert_eq!(counter.0.load(Ordering::SeqCst), 0);
    assert_eq!(driver.pending_timers(), 0);
}

#[test]
fn dropped_sleep_leaves_no_entry_and_never_wakes() {
    let counter = Arc::new(Counting(AtomicUsize::new(0)));
    let waker = Waker::from(Arc::clone(&counter));
    let mut doomed = sleep(Duration::from_millis(30));
    assert!(Pin::new(&mut doomed).poll(&mut Context::from_waker(&waker)).is_pending());
    drop(doomed);

    let (sender, receiver) = channel();
    let mut sentinel = sleep(Duration::from_millis(60));
    let sentinel_waker = notifier(7, &sender);
    assert!(Pin::new(&mut sentinel).poll(&mut Context::from_waker(&sentinel_waker)).is_pending());
    wait_for(&receiver);

    assert_eq!(counter.0.load(Ordering::SeqCst), 0);
    assert_eq!(DRIVER.pending_timers(), 0);
}

#[test]
fn schedule_refresh_and_cancel_allocate_nothing() {
    let driver = StdThreadDriver::new();
    let (sender, _receiver) = channel();
    let hour = Duration::from_secs(3_600);
    let wakers: Vec<Waker> = (0..1_000).map(|index| notifier(index, &sender)).collect();
    let deadlines: Vec<Instant> = (0..1_000_u64)
        .map(|index| after(&driver, hour + Duration::from_millis(index)))
        .collect();
    let (started_sender, started) = channel();
    driver.schedule_wake(after(&driver, Duration::from_millis(1)), notifier(0, &started_sender));
    wait_for(&started);
    let tasks: Vec<usize> = wakers.iter().map(|waker| waker.data().addr()).collect();
    let before = allocations();

    for (waker, deadline) in wakers.iter().zip(&deadlines) {
        driver.schedule_wake(*deadline, waker.clone());
        driver.refresh_wake(*deadline, waker);
    }
    for (task, deadline) in tasks.iter().zip(&deadlines) {
        driver.cancel_wake(*deadline, *task);
    }

    assert_eq!(allocations() - before, 0);
}

fn poll_once(future: &mut Pin<Box<dyn Future<Output = ()>>>, waker: &Waker) -> Poll<()> {
    future.as_mut().poll(&mut Context::from_waker(waker))
}

#[test]
fn ten_thousand_timers_polled_repeatedly_complete_on_one_owner_thread() {
    const TIMERS: usize = 10_000;
    let baseline = thread_count();
    let (sender, receiver) = channel();
    let wakers: Vec<Waker> = (0..TIMERS).map(|index| notifier(index, &sender)).collect();
    // heterogeneous Sleep/Timeout futures in one table
    let mut futures: Vec<Option<Pin<Box<dyn Future<Output = ()>>>>> = (0..TIMERS)
        .map(|index| {
            let delay = Duration::from_millis(150 + (index % 100) as u64);
            let future: Pin<Box<dyn Future<Output = ()>>> = if index % 2 == 0 {
                Box::pin(sleep(delay))
            } else {
                Box::pin(async move {
                    let outcome = timeout(delay, std::future::pending::<()>()).await;
                    assert!(outcome.is_err(), "pending future loses to the timer");
                })
            };
            Some(future)
        })
        .collect();

    for _round in 0..3 {
        for (slot, waker) in futures.iter_mut().zip(&wakers) {
            let pending = slot.as_mut().map(|future| poll_once(future, waker));
            assert_eq!(pending, Some(Poll::Pending));
        }
    }
    let peak = thread_count();
    let mut completed = 0;
    while completed < TIMERS {
        let index = wait_for(&receiver);
        let Some(future) = futures[index].as_mut() else {
            continue;
        };
        if poll_once(future, &wakers[index]).is_ready() {
            futures[index] = None;
            completed += 1;
        }
    }

    assert!(peak <= baseline + 2, "peak {peak} vs baseline {baseline}");
    assert!(thread_count() <= baseline + 2);
    assert_eq!(completed, TIMERS);
    assert_eq!(DRIVER.submission_overflows(), 0);
    assert_eq!(DRIVER.table_overflows(), 0);
}

#[test]
fn saturated_table_defers_without_early_wakes_and_each_timer_wakes_once() {
    use proxima_core::sized::TIMER_TABLE_CAP;
    let driver = StdThreadDriver::new();
    let (sender, receiver) = channel();
    let deadline = after(&driver, Duration::from_millis(200));
    let total = TIMER_TABLE_CAP + 100;

    for index in 0..total {
        driver.schedule_wake(deadline, notifier(index, &sender));
    }
    let mut wakes_per_timer = vec![0_u32; total];
    let first = wait_for(&receiver);
    let woken_at = driver.now();
    wakes_per_timer[first] += 1;
    for _ in 1..total {
        wakes_per_timer[wait_for(&receiver)] += 1;
    }

    assert!(woken_at >= deadline, "no registration woke before its deadline");
    assert!(wakes_per_timer.iter().all(|wakes| *wakes == 1));
    assert_eq!(driver.deferred_registrations(), 100);
    assert_eq!(driver.table_overflows(), 0);
}

#[test]
fn registrations_beyond_main_and_deferred_capacity_are_counted_and_woken_once_each() {
    use proxima_core::sized::{TIMER_DEFER_CAP, TIMER_TABLE_CAP};
    let driver = StdThreadDriver::new();
    let (sender, receiver) = channel();
    let far = after(&driver, Duration::from_secs(3_600));
    let excess = 10;

    for index in 0..TIMER_TABLE_CAP + TIMER_DEFER_CAP + excess {
        driver.schedule_wake(far, notifier(index, &sender));
    }
    let mut woken: Vec<usize> = (0..excess).map(|_| wait_for(&receiver)).collect();
    woken.sort_unstable();
    woken.dedup();

    assert_eq!(woken.len(), excess);
    assert_eq!(driver.table_overflows(), excess as u64);
}
