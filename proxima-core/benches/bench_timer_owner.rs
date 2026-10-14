//! Owner-thread cost of the std timer driver: wall time from the first
//! submission to a sentinel wake that queues behind n schedules (then n cancels),
//! divided by n. Deadlines are pseudo-random an hour out so none fire.

use std::sync::Arc;
use std::sync::mpsc::{Sender, channel};
use std::task::{Wake, Waker};
use std::time::{Duration, Instant as WallInstant};

use proxima_core::time::Driver;
use proxima_core::time::drivers::std_thread::StdThreadDriver;

struct Notify(Sender<()>);

impl Wake for Notify {
    fn wake(self: Arc<Self>) {
        let _ = self.0.send(());
    }
}

fn nanos_per_op(timers: usize) -> (f64, f64) {
    let driver = StdThreadDriver::new();
    let (sender, receiver) = channel();
    let base = driver.now() + Duration::from_secs(3_600);
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let registrations: Vec<_> = (0..timers)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let deadline = base + Duration::from_micros(state >> 40);
            let (unused, _) = channel::<()>();
            (deadline, Waker::from(Arc::new(Notify(unused))))
        })
        .collect();
    let sentinel = Waker::from(Arc::new(Notify(sender)));

    let started = WallInstant::now();
    for (deadline, waker) in &registrations {
        driver.schedule_wake(*deadline, waker.clone());
    }
    driver.schedule_wake(driver.now(), sentinel.clone());
    if receiver.recv().is_err() {
        return (f64::NAN, f64::NAN);
    }
    let insert = started.elapsed();

    let started = WallInstant::now();
    for (deadline, waker) in &registrations {
        driver.cancel_wake(*deadline, waker.data().addr());
    }
    driver.schedule_wake(driver.now(), sentinel);
    if receiver.recv().is_err() {
        return (f64::NAN, f64::NAN);
    }
    let cancel = started.elapsed();
    let per_op = |elapsed: Duration| elapsed.as_nanos() as f64 / timers as f64;
    (per_op(insert), per_op(cancel))
}

fn report(label: &str, samples: &[f64]) {
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance =
        samples.iter().map(|sample| (sample - mean).powi(2)).sum::<f64>() / samples.len() as f64;
    println!(
        "{label} mean_ns_per_op={mean:.1} cov={:.3}",
        variance.sqrt() / mean
    );
}

fn main() {
    for timers in [1_000, 10_000, 16_000] {
        let runs: Vec<(f64, f64)> = (0..3).map(|_| nanos_per_op(timers)).collect();
        let inserts: Vec<f64> = runs.iter().map(|run| run.0).collect();
        let cancels: Vec<f64> = runs.iter().map(|run| run.1).collect();
        report(&format!("n={timers} insert"), &inserts);
        report(&format!("n={timers} cancel"), &cancels);
    }
}
