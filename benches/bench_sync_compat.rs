//! `proxima::sync` vs `tokio::sync` shootout.
//!
//! Compares the hand-rolled / re-backed wrappers in `proxima::sync` to
//! their `tokio::sync` analogues on representative workloads. Each
//! bench arm runs inside a `tokio::runtime::current_thread` so the
//! futures executor is fixed — the only variable is the sync primitive.
//!
//! Arms compared:
//! - mpsc: `tokio::sync::mpsc` vs `proxima_sync_mpsc` (`proxima::sync::mpsc`, async-channel backed)
//! - watch: `tokio::sync::watch` vs `proxima::sync::watch` (hand-rolled over RwLock+Event)
//! - broadcast: `tokio::sync::broadcast` vs `proxima_sync_broadcast` (`proxima::sync::broadcast`, async-broadcast backed)
//! - Notify: `tokio::sync::Notify` vs `proxima::sync::Notify` (hand-rolled over event_listener::Event)
//!
//! Run:
//! ```bash
//! cargo bench -p proxima --bench bench_sync_compat
//! cargo bench -p proxima --bench bench_sync_compat -- mpsc
//! ```

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use std::sync::Arc;
use std::time::Duration;

#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
use std::time::Instant;

use bytes::Bytes;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use tokio::runtime::Builder as TokioBuilder;

#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
use proxima::runtime::{CoreId, PrimeRuntime};

const MPSC_CAPACITY: usize = 32;
const MPSC_MESSAGES: usize = 1024;
const MPSC_PAYLOAD: usize = 8 * 1024;

const BROADCAST_CAPACITY: usize = 1024;
const BROADCAST_MESSAGES: usize = 1024;
const BROADCAST_SUBSCRIBERS: usize = 4;

const BROADCAST_LAG_CAPACITY: usize = 32;
const BROADCAST_LAG_MESSAGES: usize = 1024;
const BROADCAST_LAG_SUBSCRIBERS: usize = 4;
const BROADCAST_LAG_SLOW_DELAY_US: u64 = 50;

const NOTIFY_ROUNDTRIPS: usize = 1024;
const NOTIFY_FANOUT_WAITERS: usize = 100;
const NOTIFY_FANOUT_ITERS: usize = 64;
const WATCH_ROUNDTRIPS: usize = 1024;
const WATCH_FAN_OUT: usize = 16;

fn current_thread_runtime() -> tokio::runtime::Runtime {
    TokioBuilder::new_current_thread()
        .enable_all()
        .build()
        .expect("current thread runtime")
}

fn payload(seed: u8) -> Bytes {
    let mut buf = Vec::with_capacity(MPSC_PAYLOAD);
    buf.resize(MPSC_PAYLOAD, seed);
    Bytes::from(buf)
}

/// Cross-runtime parity helper: drive an async block on a prime
/// worker (`CoreId(0)`) and return the elapsed wall-clock time.
/// Used by the `proxima_on_prime` arms; pairs with criterion's
/// `iter_custom` which expects per-iteration durations.
///
/// **2-core prime configuration.** Producer arms typically run on
/// CoreId(0) and consumer arms on CoreId(1) so they execute on
/// separate OS threads. Single-core prime + cooperative yield-via-
/// waker doesn't reliably let the consumer task run (prime's
/// executor re-polls the just-woken task immediately, starving the
/// consumer). Two cores side-step this with a real thread boundary;
/// the wrappers being tested are all `Send`. Note this is a
/// different shape from the tokio current_thread baseline — the
/// prime arm measures cross-core wrapper cost, not in-task cost.
#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
fn drive_on_prime<MakeFuture, Fut>(prime: &Arc<PrimeRuntime>, make: MakeFuture) -> Duration
where
    MakeFuture: FnOnce(Arc<PrimeRuntime>) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    use proxima::runtime::Runtime;
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    let prime_for_factory = prime.clone();
    let start = Instant::now();
    prime
        .spawn_on_core(
            CoreId(0),
            Box::pin(async move {
                make(prime_for_factory).await;
                let _ = tx.send(());
            }),
        )
        .expect("spawn on prime");
    rx.recv().expect("done");
    start.elapsed()
}

// ---------- mpsc ----------

fn bench_mpsc(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("mpsc_bounded_1p1c");
    // design-favors: neutral (SPSC primitive)
    group.throughput(Throughput::Bytes((MPSC_MESSAGES * MPSC_PAYLOAD) as u64));
    group.measurement_time(Duration::from_secs(3));

    group.bench_function("tokio", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let (sender, mut receiver) = tokio::sync::mpsc::channel::<Bytes>(MPSC_CAPACITY);
                let producer = tokio::spawn(async move {
                    for index in 0..MPSC_MESSAGES {
                        sender
                            .send(payload((index & 0xFF) as u8))
                            .await
                            .expect("send");
                    }
                });
                let mut received = 0usize;
                while let Some(_chunk) = receiver.recv().await {
                    received += 1;
                    if received == MPSC_MESSAGES {
                        break;
                    }
                }
                let _ = producer.await;
            });
        });
    });

    group.bench_function("proxima_sync_mpsc", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let (sender, mut receiver) = proxima::sync::mpsc::channel::<Bytes>(MPSC_CAPACITY);
                let producer = tokio::spawn(async move {
                    for index in 0..MPSC_MESSAGES {
                        sender
                            .send(payload((index & 0xFF) as u8))
                            .await
                            .expect("send");
                    }
                });
                let mut received = 0usize;
                while let Some(_chunk) = receiver.recv().await {
                    received += 1;
                    if received == MPSC_MESSAGES {
                        break;
                    }
                }
                let _ = producer.await;
            });
        });
    });

    #[cfg(all(
        feature = "runtime-prime-executor",
        feature = "runtime-prime-inbox-alloc",
        feature = "runtime-prime-reactor",
        feature = "runtime-prime-bgpool"
    ))]
    {
        use proxima::runtime::Runtime;
        // Single-core prime: producer + consumer both on CoreId(0)
        // (the root drive_on_prime task). mpsc's `Receiver::recv`
        // parks naturally when the channel is empty, and its send
        // path parks when full — so the executor sees real Pending
        // points and interleaves the two halves cleanly. Other
        // primitives (watch / broadcast / Notify) have synchronous
        // send paths and don't yield without a cooperative-yield
        // primitive prime doesn't currently provide; their on_prime
        // arms are deferred for a follow-on pass.
        group.bench_function("proxima_on_prime", |bench| {
            let runtime: Arc<PrimeRuntime> = Arc::new(PrimeRuntime::new(1).expect("prime"));
            bench.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    total += drive_on_prime(&runtime, |prime| async move {
                        let (sender, mut receiver) =
                            proxima::sync::mpsc::channel::<Bytes>(MPSC_CAPACITY);
                        let (done_tx, done_rx) = futures::channel::oneshot::channel::<()>();
                        prime.spawn_on_current_core(Box::pin(async move {
                            for index in 0..MPSC_MESSAGES {
                                sender
                                    .send(payload((index & 0xFF) as u8))
                                    .await
                                    .expect("send");
                            }
                            let _ = done_tx.send(());
                        }));
                        let mut received = 0usize;
                        while let Some(_chunk) = receiver.recv().await {
                            received += 1;
                            if received == MPSC_MESSAGES {
                                break;
                            }
                        }
                        let _ = done_rx.await;
                    });
                }
                total
            });
        });
    }

    group.finish();
}

// ---------- watch (1P -> 1 cursor) ----------

fn bench_watch_roundtrip(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("watch_send_observe_1p1c");
    group.throughput(Throughput::Elements(WATCH_ROUNDTRIPS as u64));
    group.measurement_time(Duration::from_secs(3));

    group.bench_function("tokio", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let (sender, mut receiver) = tokio::sync::watch::channel(0_u64);
                let consumer = tokio::spawn(async move {
                    for _ in 0..WATCH_ROUNDTRIPS {
                        receiver.changed().await.expect("changed");
                        let _value = *receiver.borrow_and_update();
                    }
                });
                for index in 1..=WATCH_ROUNDTRIPS as u64 {
                    sender.send(index).expect("send");
                    tokio::task::yield_now().await;
                }
                consumer.await.expect("consumer join");
            });
        });
    });

    group.bench_function("proxima_hand_rolled", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let (sender, mut receiver) = proxima::sync::watch::channel(0_u64);
                let consumer = tokio::spawn(async move {
                    for _ in 0..WATCH_ROUNDTRIPS {
                        if receiver.changed().await.is_err() {
                            break;
                        }
                        let _value = *receiver.borrow_and_update();
                    }
                });
                for index in 1..=WATCH_ROUNDTRIPS as u64 {
                    sender.send(index).expect("send");
                    tokio::task::yield_now().await;
                }
                consumer.await.expect("consumer join");
            });
        });
    });

    // watch on prime: deferred. Producer's `send` is synchronous;
    // consumer needs the executor to yield between sends but prime's
    // executor re-polls the just-woken task before scheduling other
    // ready tasks under the `waker.wake_by_ref() + Poll::Pending`
    // yield primitive. A 2-core variant (producer on core 0,
    // consumer on core 1) hung on the cross-core async-channel /
    // oneshot wake path — needs a separate debugging pass. Tracked
    // in discipline-proxima-sync.md as a follow-on.

    group.finish();
}

// ---------- watch fanout (1P -> N receivers) ----------

fn bench_watch_fanout(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("watch_fanout_1p_to_n");
    group.throughput(Throughput::Elements(
        (WATCH_FAN_OUT as u64) * (WATCH_ROUNDTRIPS as u64),
    ));
    group.measurement_time(Duration::from_secs(3));

    group.bench_function("tokio", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let (sender, _initial) = tokio::sync::watch::channel(0_u64);
                let mut consumers = Vec::with_capacity(WATCH_FAN_OUT);
                for _ in 0..WATCH_FAN_OUT {
                    let mut receiver = sender.subscribe();
                    consumers.push(tokio::spawn(async move {
                        for _ in 0..WATCH_ROUNDTRIPS {
                            if receiver.changed().await.is_err() {
                                break;
                            }
                            let _value = *receiver.borrow_and_update();
                        }
                    }));
                }
                for index in 1..=WATCH_ROUNDTRIPS as u64 {
                    sender.send(index).expect("send");
                    tokio::task::yield_now().await;
                }
                drop(sender);
                for consumer in consumers {
                    let _ = consumer.await;
                }
            });
        });
    });

    group.bench_function("proxima_hand_rolled", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let (sender, _initial) = proxima::sync::watch::channel(0_u64);
                let mut consumers = Vec::with_capacity(WATCH_FAN_OUT);
                for _ in 0..WATCH_FAN_OUT {
                    let mut receiver = sender.subscribe();
                    consumers.push(tokio::spawn(async move {
                        for _ in 0..WATCH_ROUNDTRIPS {
                            if receiver.changed().await.is_err() {
                                break;
                            }
                            let _value = *receiver.borrow_and_update();
                        }
                    }));
                }
                for index in 1..=WATCH_ROUNDTRIPS as u64 {
                    sender.send(index).expect("send");
                    tokio::task::yield_now().await;
                }
                drop(sender);
                for consumer in consumers {
                    let _ = consumer.await;
                }
            });
        });
    });

    // watch fanout on prime: deferred for the same yield-semantics
    // reason as watch/1p1c. See discipline log.
    #[cfg(any())]
    {
        use proxima::runtime::Runtime;
        group.bench_function("proxima_on_prime", |bench| {
            let runtime: Arc<PrimeRuntime> = Arc::new(PrimeRuntime::new(2).expect("prime"));
            bench.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    total += drive_on_prime(&runtime, |prime| async move {
                        let (sender, _initial) = proxima::sync::watch::channel(0_u64);
                        let (done_tx, done_rx) = futures::channel::oneshot::channel::<()>();
                        let consumer_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
                        let done_signal = Arc::new(std::sync::Mutex::new(Some(done_tx)));
                        for _ in 0..WATCH_FAN_OUT {
                            let mut receiver = sender.subscribe();
                            let counter = consumer_count.clone();
                            let signal_for_consumer = done_signal.clone();
                            prime
                                .spawn_on_core(
                                    CoreId(1),
                                    Box::pin(async move {
                                        let mut observations = 0usize;
                                        while observations < WATCH_ROUNDTRIPS {
                                            if receiver.changed().await.is_err() {
                                                break;
                                            }
                                            let _value = *receiver.borrow_and_update();
                                            observations += 1;
                                        }
                                        let done_now = counter
                                            .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
                                            + 1;
                                        if done_now == WATCH_FAN_OUT {
                                            if let Some(tx) =
                                                signal_for_consumer.lock().unwrap().take()
                                            {
                                                let _ = tx.send(());
                                            }
                                        }
                                    }),
                                )
                                .expect("spawn consumer on core 1");
                        }
                        for index in 1..=WATCH_ROUNDTRIPS as u64 {
                            sender.send(index).expect("send");
                        }
                        drop(sender);
                        let _ = done_rx.await;
                    });
                }
                total
            });
        });
    }

    group.finish();
}

// ---------- broadcast (1P -> N subscribers) ----------

fn bench_broadcast(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("broadcast_1p_to_n");
    group.throughput(Throughput::Elements(
        (BROADCAST_MESSAGES as u64) * (BROADCAST_SUBSCRIBERS as u64),
    ));
    group.measurement_time(Duration::from_secs(3));

    // design-favors: proxima (capacity >= msg count; lag never fires)
    group.bench_function("tokio", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let (sender, _seed) = tokio::sync::broadcast::channel::<u64>(BROADCAST_CAPACITY);
                let mut consumers = Vec::with_capacity(BROADCAST_SUBSCRIBERS);
                for _ in 0..BROADCAST_SUBSCRIBERS {
                    let mut receiver = sender.subscribe();
                    consumers.push(tokio::spawn(async move {
                        let mut delivered = 0usize;
                        while delivered < BROADCAST_MESSAGES {
                            match receiver.recv().await {
                                Ok(_value) => delivered += 1,
                                Err(_) => break,
                            }
                        }
                    }));
                }
                for index in 1..=BROADCAST_MESSAGES as u64 {
                    let _ = sender.send(index);
                    if index.is_multiple_of(64) {
                        tokio::task::yield_now().await;
                    }
                }
                for consumer in consumers {
                    let _ = consumer.await;
                }
            });
        });
    });

    // design-favors: proxima (capacity >= msg count; lag never fires)
    group.bench_function("proxima_sync_broadcast", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let (sender, _seed) = proxima::sync::broadcast::channel::<u64>(BROADCAST_CAPACITY);
                let mut consumers = Vec::with_capacity(BROADCAST_SUBSCRIBERS);
                for _ in 0..BROADCAST_SUBSCRIBERS {
                    let mut receiver = sender.subscribe();
                    consumers.push(tokio::spawn(async move {
                        let mut delivered = 0usize;
                        while delivered < BROADCAST_MESSAGES {
                            match receiver.recv().await {
                                Ok(_value) => delivered += 1,
                                Err(proxima::sync::broadcast::error::RecvError::Lagged(_)) => {}
                                Err(proxima::sync::broadcast::error::RecvError::Closed) => break,
                            }
                        }
                    }));
                }
                for index in 1..=BROADCAST_MESSAGES as u64 {
                    let _ = sender.send(index);
                    if index.is_multiple_of(64) {
                        tokio::task::yield_now().await;
                    }
                }
                drop(sender);
                for consumer in consumers {
                    let _ = consumer.await;
                }
            });
        });
    });

    // broadcast on prime: deferred (same yield-semantics issue as
    // watch — producer's broadcast `send` is sync).
    #[cfg(any())]
    {
        use proxima::runtime::Runtime;
        group.bench_function("proxima_on_prime", |bench| {
            let runtime: Arc<PrimeRuntime> = Arc::new(PrimeRuntime::new(2).expect("prime"));
            bench.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    total += drive_on_prime(&runtime, |prime| async move {
                        let (sender, _seed) =
                            proxima::sync::broadcast::channel::<u64>(BROADCAST_CAPACITY);
                        let (done_tx, done_rx) = futures::channel::oneshot::channel::<()>();
                        let consumer_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
                        let done_signal =
                            Arc::new(std::sync::Mutex::new(Some(done_tx)));
                        for _ in 0..BROADCAST_SUBSCRIBERS {
                            let mut receiver = sender.subscribe();
                            let counter = consumer_count.clone();
                            let signal_for_consumer = done_signal.clone();
                            prime
                                .spawn_on_core(
                                    CoreId(1),
                                    Box::pin(async move {
                                        let mut delivered = 0usize;
                                        while delivered < BROADCAST_MESSAGES {
                                            match receiver.recv().await {
                                                Ok(_value) => delivered += 1,
                                                Err(
                                                    proxima::sync::broadcast::error::RecvError::Lagged(
                                                        _,
                                                    ),
                                                ) => {}
                                                Err(
                                                    proxima::sync::broadcast::error::RecvError::Closed,
                                                ) => break,
                                            }
                                        }
                                        let done_now = counter.fetch_add(
                                            1,
                                            std::sync::atomic::Ordering::AcqRel,
                                        ) + 1;
                                        if done_now == BROADCAST_SUBSCRIBERS {
                                            if let Some(tx) =
                                                signal_for_consumer.lock().unwrap().take()
                                            {
                                                let _ = tx.send(());
                                            }
                                        }
                                    }),
                                )
                                .expect("spawn consumer on core 1");
                        }
                        for index in 1..=BROADCAST_MESSAGES as u64 {
                            let _ = sender.send(index);
                        }
                        drop(sender);
                        let _ = done_rx.await;
                    });
                }
                total
            });
        });
    }

    group.finish();
}

// ---------- broadcast lagging subscriber (lag handling engaged) ----------
//
// 4 subscribers, cap=32, 1024 messages. 3 fast subscribers consume immediately.
// 1 slow subscriber stalls 50µs per message — falls hundreds behind — forcing
// both implementations to detect overflow and surface Lagged/RecvError::Lagged.
// Measures fast-subscriber throughput (the producer must not be back-pressured)
// and counts lag_recovery_count (how many Lagged errors the slow sub received).

fn bench_broadcast_lagging_subscriber(criterion: &mut Criterion) {
    use std::sync::atomic::{AtomicU64, Ordering};

    let mut group = criterion.benchmark_group("broadcast_lagging_subscriber");
    let fast_count = (BROADCAST_LAG_SUBSCRIBERS - 1) as u64;
    group.throughput(Throughput::Elements(
        BROADCAST_LAG_MESSAGES as u64 * fast_count,
    ));
    group.measurement_time(Duration::from_secs(10));

    // design-favors: incumbent (lag handling engaged)
    group.bench_function("tokio", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let lag_count = Arc::new(AtomicU64::new(0));
                let (sender, _seed) =
                    tokio::sync::broadcast::channel::<u64>(BROADCAST_LAG_CAPACITY);
                let mut consumers = Vec::with_capacity(BROADCAST_LAG_SUBSCRIBERS);

                for index in 0..BROADCAST_LAG_SUBSCRIBERS {
                    let mut receiver = sender.subscribe();
                    let lag_counter = lag_count.clone();
                    if index == 0 {
                        consumers.push(tokio::spawn(async move {
                            let mut delivered = 0usize;
                            while delivered < BROADCAST_LAG_MESSAGES {
                                match receiver.recv().await {
                                    Ok(_value) => delivered += 1,
                                    Err(tokio::sync::broadcast::error::RecvError::Lagged(
                                        skipped,
                                    )) => {
                                        lag_counter.fetch_add(skipped, Ordering::Relaxed);
                                    }
                                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                                }
                                tokio::time::sleep(Duration::from_micros(
                                    BROADCAST_LAG_SLOW_DELAY_US,
                                ))
                                .await;
                            }
                        }));
                    } else {
                        consumers.push(tokio::spawn(async move {
                            let mut delivered = 0usize;
                            while delivered < BROADCAST_LAG_MESSAGES {
                                match receiver.recv().await {
                                    Ok(_value) => delivered += 1,
                                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                                }
                            }
                        }));
                    }
                }

                for index in 1..=BROADCAST_LAG_MESSAGES as u64 {
                    let _ = sender.send(index);
                    if index.is_multiple_of(64) {
                        tokio::task::yield_now().await;
                    }
                }
                drop(sender);
                for consumer in consumers {
                    let _ = consumer.await;
                }
                let _lag_events = lag_count.load(Ordering::Relaxed);
            });
        });
    });

    // design-favors: incumbent (lag handling engaged)
    group.bench_function("proxima_sync_broadcast", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let lag_count = Arc::new(AtomicU64::new(0));
                let (sender, _seed) =
                    proxima::sync::broadcast::channel::<u64>(BROADCAST_LAG_CAPACITY);
                let mut consumers = Vec::with_capacity(BROADCAST_LAG_SUBSCRIBERS);

                for index in 0..BROADCAST_LAG_SUBSCRIBERS {
                    let mut receiver = sender.subscribe();
                    let lag_counter = lag_count.clone();
                    if index == 0 {
                        consumers.push(tokio::spawn(async move {
                            let mut delivered = 0usize;
                            while delivered < BROADCAST_LAG_MESSAGES {
                                match receiver.recv().await {
                                    Ok(_value) => delivered += 1,
                                    Err(proxima::sync::broadcast::error::RecvError::Lagged(
                                        skipped,
                                    )) => {
                                        lag_counter.fetch_add(skipped, Ordering::Relaxed);
                                    }
                                    Err(proxima::sync::broadcast::error::RecvError::Closed) => {
                                        break;
                                    }
                                }
                                tokio::time::sleep(Duration::from_micros(
                                    BROADCAST_LAG_SLOW_DELAY_US,
                                ))
                                .await;
                            }
                        }));
                    } else {
                        consumers.push(tokio::spawn(async move {
                            let mut delivered = 0usize;
                            while delivered < BROADCAST_LAG_MESSAGES {
                                match receiver.recv().await {
                                    Ok(_value) => delivered += 1,
                                    Err(proxima::sync::broadcast::error::RecvError::Lagged(_)) => {}
                                    Err(proxima::sync::broadcast::error::RecvError::Closed) => {
                                        break;
                                    }
                                }
                            }
                        }));
                    }
                }

                for index in 1..=BROADCAST_LAG_MESSAGES as u64 {
                    let _ = sender.send(index);
                    if index.is_multiple_of(64) {
                        tokio::task::yield_now().await;
                    }
                }
                drop(sender);
                for consumer in consumers {
                    let _ = consumer.await;
                }
                let _lag_events = lag_count.load(Ordering::Relaxed);
            });
        });
    });

    group.finish();
}

// ---------- Notify roundtrip ----------

fn bench_notify(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("notify_one_roundtrip");
    group.throughput(Throughput::Elements(NOTIFY_ROUNDTRIPS as u64));
    group.measurement_time(Duration::from_secs(3));

    // design-favors: neutral (primitive 1p/1c ping-pong)
    group.bench_function("tokio", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let notify = Arc::new(tokio::sync::Notify::new());
                let consumer_notify = notify.clone();
                let consumer = tokio::spawn(async move {
                    for _ in 0..NOTIFY_ROUNDTRIPS {
                        consumer_notify.notified().await;
                    }
                });
                for _ in 0..NOTIFY_ROUNDTRIPS {
                    notify.notify_one();
                    tokio::task::yield_now().await;
                }
                consumer.await.expect("consumer join");
            });
        });
    });

    // design-favors: neutral (primitive 1p/1c ping-pong)
    group.bench_function("proxima_hand_rolled", |bench| {
        let runtime = current_thread_runtime();
        bench.iter(|| {
            runtime.block_on(async {
                let notify = Arc::new(proxima::sync::Notify::new());
                let consumer_notify = notify.clone();
                let consumer = tokio::spawn(async move {
                    for _ in 0..NOTIFY_ROUNDTRIPS {
                        consumer_notify.notified().await;
                    }
                });
                for _ in 0..NOTIFY_ROUNDTRIPS {
                    notify.notify_one();
                    tokio::task::yield_now().await;
                }
                consumer.await.expect("consumer join");
            });
        });
    });

    // Notify on prime: deferred (same yield-semantics issue —
    // notify_one is sync, producer hammers the consumer's permit
    // without yielding).
    #[cfg(any())]
    {
        use proxima::runtime::Runtime;
        group.bench_function("proxima_on_prime", |bench| {
            let runtime: Arc<PrimeRuntime> = Arc::new(PrimeRuntime::new(2).expect("prime"));
            bench.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    total += drive_on_prime(&runtime, |prime| async move {
                        let notify = Arc::new(proxima::sync::Notify::new());
                        let consumer_notify = notify.clone();
                        let (done_tx, done_rx) = futures::channel::oneshot::channel::<()>();
                        prime
                            .spawn_on_core(
                                CoreId(1),
                                Box::pin(async move {
                                    for _ in 0..NOTIFY_ROUNDTRIPS {
                                        consumer_notify.notified().await;
                                    }
                                    let _ = done_tx.send(());
                                }),
                            )
                            .expect("spawn consumer on core 1");
                        for _ in 0..NOTIFY_ROUNDTRIPS {
                            notify.notify_one();
                        }
                        let _ = done_rx.await;
                    });
                }
                total
            });
        });
    }

    group.finish();
}

// ---------- Notify waiters fan-out (1P -> N waiters, notify_waiters) ----------
//
// Each round: all N consumers signal "ready" via an AtomicUsize counter + a
// tokio Notify; the producer waits for all N before calling notify_waiters().
// This guarantees every notify_waiters() hits all N live waiters — the exact
// fan-out shape tokio's internal queue structure is tuned for.
//
// Multi-thread runtime (4 workers) so consumers run on real OS threads and are
// genuinely parked on the Notify when the producer fires.

fn multi_thread_runtime() -> tokio::runtime::Runtime {
    TokioBuilder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("multi-thread runtime")
}

// Semaphore-gated fanout driver: each consumer calls `notified.enable()` to
// register its waker, THEN adds a permit to the semaphore. Producer acquires
// all N permits before calling notify_waiters() — guaranteeing every waker
// is registered before the broadcast fires. No lost-wakeup risk.
async fn fanout_tokio_body() {
    let notify = Arc::new(tokio::sync::Notify::new());
    let ready = Arc::new(tokio::sync::Semaphore::new(0));
    let mut consumers = Vec::with_capacity(NOTIFY_FANOUT_WAITERS);

    for _ in 0..NOTIFY_FANOUT_WAITERS {
        let consumer_notify = notify.clone();
        let consumer_ready = ready.clone();
        consumers.push(tokio::spawn(async move {
            for _ in 0..NOTIFY_FANOUT_ITERS {
                let mut notified = std::pin::pin!(consumer_notify.notified());
                notified.as_mut().enable();
                consumer_ready.add_permits(1);
                notified.await;
            }
        }));
    }

    for _ in 0..NOTIFY_FANOUT_ITERS {
        let permits = ready
            .acquire_many(NOTIFY_FANOUT_WAITERS as u32)
            .await
            .expect("semaphore acquire");
        permits.forget();
        notify.notify_waiters();
    }

    for consumer in consumers {
        consumer.await.expect("consumer join");
    }
}

async fn fanout_proxima_body() {
    let notify = Arc::new(proxima::sync::Notify::new());
    let ready = Arc::new(tokio::sync::Semaphore::new(0));
    let mut consumers = Vec::with_capacity(NOTIFY_FANOUT_WAITERS);

    for _ in 0..NOTIFY_FANOUT_WAITERS {
        let consumer_notify = notify.clone();
        let consumer_ready = ready.clone();
        consumers.push(tokio::spawn(async move {
            for _ in 0..NOTIFY_FANOUT_ITERS {
                let mut notified = std::pin::pin!(consumer_notify.notified());
                notified.as_mut().enable();
                consumer_ready.add_permits(1);
                notified.await;
            }
        }));
    }

    for _ in 0..NOTIFY_FANOUT_ITERS {
        let permits = ready
            .acquire_many(NOTIFY_FANOUT_WAITERS as u32)
            .await
            .expect("semaphore acquire");
        permits.forget();
        notify.notify_waiters();
    }

    for consumer in consumers {
        consumer.await.expect("consumer join");
    }
}

fn run_tokio_fanout(iters: u64) -> Duration {
    use std::time::Instant;

    let runtime = multi_thread_runtime();
    let mut total = Duration::ZERO;

    for _ in 0..iters {
        let start = Instant::now();
        runtime.block_on(fanout_tokio_body());
        total += start.elapsed();
    }
    total
}

fn run_proxima_fanout(iters: u64) -> Duration {
    use std::time::Instant;

    let runtime = multi_thread_runtime();
    let mut total = Duration::ZERO;

    for _ in 0..iters {
        let start = Instant::now();
        runtime.block_on(fanout_proxima_body());
        total += start.elapsed();
    }
    total
}

fn bench_notify_waiters_fanout_n100(criterion: &mut Criterion) {
    let total_wakeups = (NOTIFY_FANOUT_WAITERS * NOTIFY_FANOUT_ITERS) as u64;
    let mut group = criterion.benchmark_group("notify_waiters_fanout_n100");
    group.throughput(Throughput::Elements(total_wakeups));
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(10));

    // design-favors: incumbent (notify-N home turf — N=100 waiters woken at once)
    group.bench_function("tokio", |bench| {
        bench.iter_custom(run_tokio_fanout);
    });

    // design-favors: incumbent (notify-N home turf — N=100 waiters woken at once)
    group.bench_function("proxima_hand_rolled", |bench| {
        bench.iter_custom(run_proxima_fanout);
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_mpsc,
    bench_watch_roundtrip,
    bench_watch_fanout,
    bench_broadcast,
    bench_broadcast_lagging_subscriber,
    bench_notify,
    bench_notify_waiters_fanout_n100,
);
criterion_main!(benches);
