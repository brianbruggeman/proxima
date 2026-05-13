//! Stage 7 — cross-thread CPU-bound background work via `BackgroundPool`.
//!
//! Property: a Pipe can dispatch CPU-bound work off the chain
//! runtime (per-core) onto a separate work-stealing pool, await the
//! result, and the chain dispatch latency stays unaffected.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
#![cfg(feature = "rayon")]

use std::sync::Arc;
use std::time::{Duration, Instant};

use proxima::{BackgroundPool, RayonBackgroundPool};

#[proxima::test]
async fn rayon_pool_lets_chain_dispatch_continue_during_cpu_work() {
    let pool: Arc<dyn BackgroundPool> =
        Arc::new(RayonBackgroundPool::with_threads(2).expect("build pool"));

    // Fire a 200ms CPU-bound task on the rayon pool.
    let handle = pool.spawn(Box::new(|| {
        let mut sum: u64 = 0;
        let start = Instant::now();
        while start.elapsed() < Duration::from_millis(200) {
            sum = sum.wrapping_add(1);
        }
        Ok(Box::new(sum) as Box<dyn std::any::Any + Send>)
    }));

    // While that runs on a rayon thread, the chain side (this tokio
    // task) should be able to schedule work without being blocked.
    // We assert by measuring how long a no-op tokio sleep+yield
    // takes — it must complete well before the rayon task does.
    let chain_start = Instant::now();
    let mut chain_completions: u32 = 0;
    while chain_start.elapsed() < Duration::from_millis(150) {
        tokio::time::sleep(Duration::from_millis(5)).await;
        chain_completions += 1;
    }
    let chain_elapsed = chain_start.elapsed();

    // chain side ran ~30 iterations in 150ms — proves it wasn't
    // blocked by the rayon work
    assert!(
        chain_completions >= 20,
        "chain side blocked by background work; only {chain_completions} iters in {chain_elapsed:?}"
    );

    // background result still arrives
    let value = handle.await.expect("background result");
    let _sum = value.downcast::<u64>().expect("downcast");
}

#[proxima::test]
async fn rayon_pool_parallelizes_multiple_tasks() {
    let pool: Arc<dyn BackgroundPool> =
        Arc::new(RayonBackgroundPool::with_threads(4).expect("build pool"));

    let started_at = Instant::now();
    let mut handles = Vec::new();
    for index in 0..4_u32 {
        handles.push(pool.spawn(Box::new(move || {
            std::thread::sleep(Duration::from_millis(100));
            Ok(Box::new(index) as Box<dyn std::any::Any + Send>)
        })));
    }
    for handle in handles {
        let _ = handle.await.expect("result");
    }
    let elapsed = started_at.elapsed();

    // 4 tasks * 100ms each in serial would be ~400ms; in parallel
    // on 4 threads, ~100ms.
    assert!(
        elapsed < Duration::from_millis(250),
        "rayon should have parallelized 4 tasks; took {elapsed:?}"
    );
}
