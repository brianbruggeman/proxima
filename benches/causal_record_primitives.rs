#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

//! Pattern-matched bench: which primitive is right for `CausalIndex`?
//!
//! CausalIndex's access pattern:
//! - record: append a new CausalEdge (monotonically-growing index)
//! - edges/explain: snapshot all edges, walk backward
//! - cross-thread: cheap clone (Arc) threaded through nested
//!   Pipe::call futures; multiple recorders can hit it concurrently
//!
//! Three candidate primitives, benched against THIS exact pattern (not
//! the cycling-string-key registry pattern from `per_core_vs_arcswap`):
//!
//! 1. `ArcSwap<Vec<CausalEdge>>` — the prior implementation. O(N) per
//!    record (full Vec clone-on-write).
//! 2. `Mutex<Vec<CausalEdge>>` — O(1) append. Lock contention scales
//!    with concurrent recorders.
//! 3. `DashMap<u64, CausalEdge>` keyed by atomic seq counter — sharded
//!    write locks; monotonic u64 keys spread across shards. Reads
//!    iterate every shard + sort by seq.
//!
//! Bench scenarios:
//! - single_recorder: uncontested append, growing index.
//! - many_recorders: N concurrent recorder threads + measured recorder.
//! - snapshot_under_writers: read full edge list while recorders run.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

use std::cell::Cell;

use arc_swap::ArcSwap;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dashmap::DashMap;
use smallvec::SmallVec;

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct CausalEdge {
    node_id: String,
    output_start: u32,
    output_end: u32,
    parent_id: String,
    parent_start: u32,
    parent_end: u32,
}

fn fixture_edge() -> CausalEdge {
    CausalEdge {
        node_id: "child".into(),
        output_start: 0,
        output_end: 16,
        parent_id: "parent".into(),
        parent_start: 0,
        parent_end: 16,
    }
}

// --- ArcSwap<Vec> implementation (the prior CausalIndex shape) ---

#[derive(Clone, Default)]
struct ArcSwapIndex {
    edges: Arc<ArcSwap<Vec<CausalEdge>>>,
}

impl ArcSwapIndex {
    fn new() -> Self {
        Self {
            edges: Arc::new(ArcSwap::from_pointee(Vec::new())),
        }
    }
    fn record(&self, edge: CausalEdge) {
        loop {
            let current = self.edges.load_full();
            let mut next: Vec<CausalEdge> = (*current).clone();
            next.push(edge.clone());
            let prev = self.edges.compare_and_swap(&current, Arc::new(next));
            if Arc::ptr_eq(&prev, &current) {
                break;
            }
        }
    }
    fn snapshot(&self) -> Vec<CausalEdge> {
        (*self.edges.load_full()).clone()
    }
}

// --- Mutex<Vec> implementation (lock-based, O(1) append) ---

#[derive(Clone, Default)]
struct MutexIndex {
    edges: Arc<std::sync::Mutex<Vec<CausalEdge>>>,
}

impl MutexIndex {
    fn new() -> Self {
        Self {
            edges: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }
    fn record(&self, edge: CausalEdge) {
        if let Ok(mut guard) = self.edges.lock() {
            guard.push(edge);
        }
    }
    fn snapshot(&self) -> Vec<CausalEdge> {
        self.edges.lock().map(|g| g.clone()).unwrap_or_default()
    }
}

// --- Mutex<SmallVec<[Edge; 4]>> — inline-first-4 then heap-spill ---

#[derive(Clone, Default)]
struct MutexSmallVec4 {
    edges: Arc<std::sync::Mutex<SmallVec<[CausalEdge; 4]>>>,
}

impl MutexSmallVec4 {
    fn new() -> Self {
        Self {
            edges: Arc::new(std::sync::Mutex::new(SmallVec::new())),
        }
    }
    fn record(&self, edge: CausalEdge) {
        if let Ok(mut guard) = self.edges.lock() {
            guard.push(edge);
        }
    }
    fn snapshot(&self) -> Vec<CausalEdge> {
        self.edges
            .lock()
            .map(|g| g.iter().cloned().collect())
            .unwrap_or_default()
    }
}

// --- Mutex<SmallVec<[Edge; 8]>> — inline-first-8 ---

#[derive(Clone, Default)]
struct MutexSmallVec8 {
    edges: Arc<std::sync::Mutex<SmallVec<[CausalEdge; 8]>>>,
}

impl MutexSmallVec8 {
    fn new() -> Self {
        Self {
            edges: Arc::new(std::sync::Mutex::new(SmallVec::new())),
        }
    }
    fn record(&self, edge: CausalEdge) {
        if let Ok(mut guard) = self.edges.lock() {
            guard.push(edge);
        }
    }
    fn snapshot(&self) -> Vec<CausalEdge> {
        self.edges
            .lock()
            .map(|g| g.iter().cloned().collect())
            .unwrap_or_default()
    }
}

// --- Per-core Vec<Mutex<Vec<Edge>>> (Stage 3c shape) ---
//
// Each writer thread is assigned a unique slot index on first
// record (cached in a thread-local). Writers on different threads
// hit different Mutexes; contention is bounded by slot collision
// (writers % num_slots).
//
// snapshot iterates every slot, briefly locking each. In production,
// `current_core()` from `Runtime` would be the slot-assignment
// mechanism — same shape, deterministic per-core slot instead of
// thread-id-hashed.

thread_local! {
    static SLOT_INDEX: Cell<Option<usize>> = const { Cell::new(None) };
}

#[derive(Clone)]
struct PerCoreIndex {
    slots: Arc<Vec<std::sync::Mutex<Vec<CausalEdge>>>>,
}

impl PerCoreIndex {
    fn new(num_slots: usize) -> Self {
        Self {
            slots: Arc::new(
                (0..num_slots)
                    .map(|_| std::sync::Mutex::new(Vec::new()))
                    .collect(),
            ),
        }
    }
    fn record(&self, edge: CausalEdge) {
        let num_slots = self.slots.len();
        let slot_idx = SLOT_INDEX.with(|cell| {
            cell.get().unwrap_or_else(|| {
                let id = std::thread::current().id();
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                std::hash::Hash::hash(&id, &mut hasher);
                let h = std::hash::Hasher::finish(&hasher);
                let idx = (h % num_slots as u64) as usize;
                cell.set(Some(idx));
                idx
            })
        });
        if let Ok(mut guard) = self.slots[slot_idx].lock() {
            guard.push(edge);
        }
    }
    fn snapshot(&self) -> Vec<CausalEdge> {
        let mut all = Vec::new();
        for slot in self.slots.iter() {
            if let Ok(slot) = slot.lock() {
                all.extend(slot.iter().cloned());
            }
        }
        all
    }
}

// --- DashMap<u64, Edge> + atomic seq (sharded, monotonic keys) ---

#[derive(Clone, Default)]
struct DashMapIndex {
    edges: Arc<DashMap<u64, CausalEdge>>,
    seq: Arc<AtomicU64>,
}

impl DashMapIndex {
    fn new() -> Self {
        Self {
            edges: Arc::new(DashMap::new()),
            seq: Arc::new(AtomicU64::new(0)),
        }
    }
    fn record(&self, edge: CausalEdge) {
        let key = self.seq.fetch_add(1, Ordering::Relaxed);
        self.edges.insert(key, edge);
    }
    fn snapshot(&self) -> Vec<CausalEdge> {
        let mut entries: Vec<(u64, CausalEdge)> = self
            .edges
            .iter()
            .map(|entry| (*entry.key(), entry.value().clone()))
            .collect();
        entries.sort_by_key(|(seq, _)| *seq);
        entries.into_iter().map(|(_, edge)| edge).collect()
    }
}

fn single_recorder(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("causal_single_recorder");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));

    let arcswap = ArcSwapIndex::new();
    group.bench_function("arcswap_vec", |bencher| {
        bencher.iter(|| arcswap.record(fixture_edge()));
    });

    let mutex = MutexIndex::new();
    group.bench_function("mutex_vec", |bencher| {
        bencher.iter(|| mutex.record(fixture_edge()));
    });

    let dashmap = DashMapIndex::new();
    group.bench_function("dashmap_u64_seq", |bencher| {
        bencher.iter(|| dashmap.record(fixture_edge()));
    });

    let smallvec4 = MutexSmallVec4::new();
    group.bench_function("mutex_smallvec_4", |bencher| {
        bencher.iter(|| smallvec4.record(fixture_edge()));
    });

    let smallvec8 = MutexSmallVec8::new();
    group.bench_function("mutex_smallvec_8", |bencher| {
        bencher.iter(|| smallvec8.record(fixture_edge()));
    });

    // Per-core (Stage 3c shape): 16 slots, single writer thread
    let percore = PerCoreIndex::new(16);
    group.bench_function("per_core_16_slots", |bencher| {
        bencher.iter(|| percore.record(fixture_edge()));
    });

    group.finish();
}

// Specifically test the "small recording" case the user asked about:
// fresh CausalIndex, push N edges, snapshot. This is the scenario
// where SmallVec eliminates a heap alloc.
fn small_recording_full_lifecycle(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("causal_small_recording_lifecycle");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));

    for edge_count in [1_usize, 4, 8] {
        // Vec<Edge> — heap alloc on first push, doubles thereafter
        group.bench_with_input(
            BenchmarkId::new("mutex_vec", edge_count),
            &edge_count,
            |bencher, &n| {
                bencher.iter(|| {
                    let index = MutexIndex::new();
                    for _ in 0..n {
                        index.record(fixture_edge());
                    }
                    std::hint::black_box(index.snapshot().len());
                });
            },
        );

        // SmallVec<[Edge; 4]> — no heap alloc if N≤4, single heap on spill
        group.bench_with_input(
            BenchmarkId::new("mutex_smallvec_4", edge_count),
            &edge_count,
            |bencher, &n| {
                bencher.iter(|| {
                    let index = MutexSmallVec4::new();
                    for _ in 0..n {
                        index.record(fixture_edge());
                    }
                    std::hint::black_box(index.snapshot().len());
                });
            },
        );

        // SmallVec<[Edge; 8]> — no heap alloc if N≤8
        group.bench_with_input(
            BenchmarkId::new("mutex_smallvec_8", edge_count),
            &edge_count,
            |bencher, &n| {
                bencher.iter(|| {
                    let index = MutexSmallVec8::new();
                    for _ in 0..n {
                        index.record(fixture_edge());
                    }
                    std::hint::black_box(index.snapshot().len());
                });
            },
        );
    }

    group.finish();
}

fn many_recorders(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("causal_many_recorders");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));

    for noise_recorders in [1_usize, 4, 16] {
        // ArcSwap
        let arcswap = ArcSwapIndex::new();
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();
        for _ in 0..noise_recorders {
            let index = arcswap.clone();
            let stop = stop.clone();
            handles.push(thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    index.record(fixture_edge());
                }
            }));
        }
        group.bench_with_input(
            BenchmarkId::new("arcswap_vec", noise_recorders),
            &arcswap,
            |bencher, index| {
                bencher.iter(|| index.record(fixture_edge()));
            },
        );
        stop.store(true, Ordering::Relaxed);
        for handle in handles {
            handle.join().expect("noise recorder joined");
        }

        // Mutex
        let mutex = MutexIndex::new();
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();
        for _ in 0..noise_recorders {
            let index = mutex.clone();
            let stop = stop.clone();
            handles.push(thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    index.record(fixture_edge());
                }
            }));
        }
        group.bench_with_input(
            BenchmarkId::new("mutex_vec", noise_recorders),
            &mutex,
            |bencher, index| {
                bencher.iter(|| index.record(fixture_edge()));
            },
        );
        stop.store(true, Ordering::Relaxed);
        for handle in handles {
            handle.join().expect("noise recorder joined");
        }

        // DashMap
        let dashmap = DashMapIndex::new();
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();
        for _ in 0..noise_recorders {
            let index = dashmap.clone();
            let stop = stop.clone();
            handles.push(thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    index.record(fixture_edge());
                }
            }));
        }
        group.bench_with_input(
            BenchmarkId::new("dashmap_u64_seq", noise_recorders),
            &dashmap,
            |bencher, index| {
                bencher.iter(|| index.record(fixture_edge()));
            },
        );
        stop.store(true, Ordering::Relaxed);
        for handle in handles {
            handle.join().expect("noise recorder joined");
        }

        // Per-core (Stage 3c shape): 16 slots. Writer threads will
        // hash to different slots; lock contention is per-slot, not
        // global. Reset SLOT_INDEX in noise threads via fresh
        // thread::spawn (each new thread gets a fresh thread-local).
        let percore = PerCoreIndex::new(16);
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();
        for _ in 0..noise_recorders {
            let index = percore.clone();
            let stop = stop.clone();
            handles.push(thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    index.record(fixture_edge());
                }
            }));
        }
        group.bench_with_input(
            BenchmarkId::new("per_core_16_slots", noise_recorders),
            &percore,
            |bencher, index| {
                bencher.iter(|| index.record(fixture_edge()));
            },
        );
        stop.store(true, Ordering::Relaxed);
        for handle in handles {
            handle.join().expect("noise recorder joined");
        }
    }

    group.finish();
}

fn snapshot_under_writers(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("causal_snapshot_under_writers");
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(1));

    for writer_threads in [1_usize, 4] {
        // ArcSwap
        let arcswap = ArcSwapIndex::new();
        for _ in 0..1000 {
            arcswap.record(fixture_edge());
        }
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();
        for _ in 0..writer_threads {
            let index = arcswap.clone();
            let stop = stop.clone();
            handles.push(thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    index.record(fixture_edge());
                }
            }));
        }
        group.bench_with_input(
            BenchmarkId::new("arcswap_vec", writer_threads),
            &arcswap,
            |bencher, index| {
                bencher.iter(|| {
                    let snap = index.snapshot();
                    std::hint::black_box(snap.len());
                });
            },
        );
        stop.store(true, Ordering::Relaxed);
        for handle in handles {
            handle.join().expect("writer joined");
        }

        // Mutex
        let mutex = MutexIndex::new();
        for _ in 0..1000 {
            mutex.record(fixture_edge());
        }
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();
        for _ in 0..writer_threads {
            let index = mutex.clone();
            let stop = stop.clone();
            handles.push(thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    index.record(fixture_edge());
                }
            }));
        }
        group.bench_with_input(
            BenchmarkId::new("mutex_vec", writer_threads),
            &mutex,
            |bencher, index| {
                bencher.iter(|| {
                    let snap = index.snapshot();
                    std::hint::black_box(snap.len());
                });
            },
        );
        stop.store(true, Ordering::Relaxed);
        for handle in handles {
            handle.join().expect("writer joined");
        }

        // DashMap
        let dashmap = DashMapIndex::new();
        for _ in 0..1000 {
            dashmap.record(fixture_edge());
        }
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();
        for _ in 0..writer_threads {
            let index = dashmap.clone();
            let stop = stop.clone();
            handles.push(thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    index.record(fixture_edge());
                }
            }));
        }
        group.bench_with_input(
            BenchmarkId::new("dashmap_u64_seq", writer_threads),
            &dashmap,
            |bencher, index| {
                bencher.iter(|| {
                    let snap = index.snapshot();
                    std::hint::black_box(snap.len());
                });
            },
        );
        stop.store(true, Ordering::Relaxed);
        for handle in handles {
            handle.join().expect("writer joined");
        }

        // Per-core: snapshot iterates every slot
        let percore = PerCoreIndex::new(16);
        for _ in 0..1000 {
            percore.record(fixture_edge());
        }
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();
        for _ in 0..writer_threads {
            let index = percore.clone();
            let stop = stop.clone();
            handles.push(thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    index.record(fixture_edge());
                }
            }));
        }
        group.bench_with_input(
            BenchmarkId::new("per_core_16_slots", writer_threads),
            &percore,
            |bencher, index| {
                bencher.iter(|| {
                    let snap = index.snapshot();
                    std::hint::black_box(snap.len());
                });
            },
        );
        stop.store(true, Ordering::Relaxed);
        for handle in handles {
            handle.join().expect("writer joined");
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    single_recorder,
    many_recorders,
    snapshot_under_writers,
    small_recording_full_lifecycle,
);
criterion_main!(benches);
