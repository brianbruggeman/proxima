#![allow(clippy::expect_used)]
use core::future::Future;
use core::task::{Context, Poll, Waker};

use proxima_primitives::sync::Semaphore;
use proxima_test::alloc_count::{CountingAllocator, allocations, reset};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn measure_uncontended_acquire_allocations() {
    let empty_window_allocations = measure_empty_window();
    assert_eq!(empty_window_allocations, 0);
    emit_record("empty_window_control", 0, empty_window_allocations);

    let boxed_control_allocations = measure_boxed_control();
    assert_eq!(boxed_control_allocations, 1);
    emit_record(
        "boxed_allocation_positive_control",
        1,
        boxed_control_allocations,
    );

    let single_acquire_allocations = measure_acquires(1);
    assert_eq!(single_acquire_allocations, 0);
    emit_record("uncontended_acquire_once", 1, single_acquire_allocations);

    let repeated_acquire_allocations = measure_acquires(16);
    assert_eq!(repeated_acquire_allocations, 0);
    emit_record(
        "uncontended_acquire_repeated",
        16,
        repeated_acquire_allocations,
    );

    let single_try_acquire_allocations = measure_try_acquires(1);
    assert_eq!(single_try_acquire_allocations, 0);
    emit_record(
        "uncontended_try_acquire_once",
        1,
        single_try_acquire_allocations,
    );

    let repeated_try_acquire_allocations = measure_try_acquires(16);
    assert_eq!(repeated_try_acquire_allocations, 0);
    emit_record(
        "uncontended_try_acquire_repeated",
        16,
        repeated_try_acquire_allocations,
    );
}

fn measure_empty_window() -> usize {
    reset();
    allocations()
}

fn measure_boxed_control() -> usize {
    reset();
    let before_allocations = allocations();
    let boxed_value = Box::new(0_u8);
    core::hint::black_box(&boxed_value);
    let measured_allocations = allocations() - before_allocations;
    drop(boxed_value);
    measured_allocations
}

fn measure_acquires(cycles: usize) -> usize {
    let semaphore = Semaphore::new(1);
    let mut context = Context::from_waker(Waker::noop());
    acquire_and_release(&semaphore, &mut context);

    reset();
    let before_allocations = allocations();
    for _ in 0..cycles {
        acquire_and_release(&semaphore, &mut context);
    }
    allocations() - before_allocations
}

fn acquire_and_release(semaphore: &Semaphore, context: &mut Context<'_>) {
    let mut acquire_future = core::pin::pin!(semaphore.acquire());
    match Future::poll(acquire_future.as_mut(), context) {
        Poll::Ready(Ok(permit)) => drop(permit),
        Poll::Ready(Err(_)) => panic!("an open semaphore rejected a permit"),
        Poll::Pending => panic!("an uncontended semaphore acquire returned Pending"),
    }
}

fn measure_try_acquires(cycles: usize) -> usize {
    let semaphore = Semaphore::new(1);
    let warm_permit = semaphore
        .try_acquire()
        .expect("open semaphore has an available permit");
    drop(warm_permit);

    reset();
    let before_allocations = allocations();
    for _ in 0..cycles {
        let permit = semaphore
            .try_acquire()
            .expect("released permit remains available");
        drop(permit);
    }
    allocations() - before_allocations
}

fn emit_record(case_name: &str, cycles: usize, allocation_delta: usize) {
    println!(
        "alloc_record {{\"case\":\"{case_name}\",\"cycles\":{cycles},\"allocation_delta\":{allocation_delta}}}"
    );
}
