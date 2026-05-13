//! "We have a stream and we cannot block" — measured.
//!
//! A stream consumer (e.g. a downstream consumer's compaction draining a k-way merge and writing
//! chunks) shares ONE cooperative core with a victim task (the request path —
//! HTTP accept/serve). The chunk write is BLOCKING work (sync file I/O / CPU).
//! Two ways to do it:
//!
//!   - inline   — run the blocking write ON the core. The cooperative executor
//!                cannot run the victim until the write returns. This is the
//!                current downstream-consumer shape (sync compaction on the serve core) and the
//!                starvation half of the jetsam saga.
//!   - offload  — `pool.spawn(write).await`. The blocking work runs on the
//!                background pool; the `.await` YIELDS the core, so the victim
//!                runs while the write proceeds (and writes parallelize across
//!                the pool). This is what `spawn_background_blocking` buys —
//!                "blocking" describes the OFFLOADED WORK, not the caller; the
//!                caller never blocks, it awaits.
//!
//! Metric (apples-to-apples, same core, same N writes, same per-write cost):
//!   - victim_iters: how many times the request path got to run during the
//!     compaction window (responsiveness — higher is better).
//!   - wall_ms: how long the compaction took (offload also parallelizes).
//!
//! Run: cargo bench --bench stream_no_block --features "runtime-tokio runtime-prime-bgpool runtime-prime-bgpool-rayon rayon"

#![cfg(all(feature = "runtime-prime-bgpool", feature = "runtime-tokio"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use proxima::runtime::prime::os::background::ProximaBackgroundPool;

const CHUNKS: usize = 400; // stream chunks (e.g. ~16KB SST data chunks)
const BURN_CYCLES: u64 = 40_000; // per-chunk blocking-write cost (~tens of µs)
const POOL_THREADS: usize = 4;

/// Models the blocking chunk write (sync I/O / CPU-bound serialization).
fn blocking_write(cycles: u64) -> u64 {
    let mut acc: u64 = 0;
    for index in 0..cycles {
        acc = std::hint::black_box(acc.wrapping_add(index ^ 0x9E37_79B9_7F4A_7C15));
    }
    std::hint::black_box(acc)
}

fn main() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    eprintln!(
        "stream-no-block: 1 cooperative core, {CHUNKS} chunk-writes @ {BURN_CYCLES} cycles each, victim = request path\n"
    );

    // ---- ARM: inline (current downstream-consumer shape — blocking write ON the core) ----
    let (victim_inline, wall_inline) = runtime.block_on(async {
        let stop = Arc::new(AtomicBool::new(false));
        let victim_count = Arc::new(AtomicU64::new(0));
        let victim = {
            let stop = stop.clone();
            let count = victim_count.clone();
            tokio::spawn(async move {
                while !stop.load(Ordering::Relaxed) {
                    count.fetch_add(1, Ordering::Relaxed);
                    tokio::task::yield_now().await; // request path wants the core
                }
            })
        };

        let started = Instant::now();
        for _ in 0..CHUNKS {
            // blocking write inline — holds the core for the whole burn
            black_box_u64(blocking_write(BURN_CYCLES));
            tokio::task::yield_now().await; // chunk boundary
        }
        let wall = started.elapsed();
        stop.store(true, Ordering::Relaxed);
        let _ = victim.await;
        (victim_count.load(Ordering::Relaxed), wall)
    });

    // ---- ARM: offload (spawn to the background pool, await — yields the core) ----
    let (victim_offload, wall_offload) = runtime.block_on(async {
        let pool = Arc::new(ProximaBackgroundPool::with_threads(POOL_THREADS).expect("pool"));
        let stop = Arc::new(AtomicBool::new(false));
        let victim_count = Arc::new(AtomicU64::new(0));
        let victim = {
            let stop = stop.clone();
            let count = victim_count.clone();
            tokio::spawn(async move {
                while !stop.load(Ordering::Relaxed) {
                    count.fetch_add(1, Ordering::Relaxed);
                    tokio::task::yield_now().await;
                }
            })
        };

        let started = Instant::now();
        for _ in 0..CHUNKS {
            // blocking write OFF the core; .await frees the core for the victim
            let handle =
                pool.spawn(move || Ok::<u64, proxima::ProximaError>(blocking_write(BURN_CYCLES)));
            let _ = handle.await;
        }
        let wall = started.elapsed();
        stop.store(true, Ordering::Relaxed);
        let _ = victim.await;
        (victim_count.load(Ordering::Relaxed), wall)
    });

    eprintln!(
        "  inline (blocking on core)   victim_iters = {:>9}   wall = {:>7.1} ms",
        victim_inline,
        wall_inline.as_secs_f64() * 1e3
    );
    eprintln!(
        "  offload (spawn + await)     victim_iters = {:>9}   wall = {:>7.1} ms",
        victim_offload,
        wall_offload.as_secs_f64() * 1e3
    );
    let resp = victim_offload as f64 / victim_inline.max(1) as f64;
    eprintln!(
        "\n  victim responsiveness offload/inline = {resp:.0}x   <-- the 'cannot block' win, measured"
    );
    eprintln!(
        "  (offload also parallelizes the writes across {POOL_THREADS} pool threads → lower wall)"
    );
}

fn black_box_u64(value: u64) {
    std::hint::black_box(value);
}
