use bytes::Bytes;
use std::cell::Cell;
use std::future::Future;
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use proxima_core::ProximaError;
use proxima_primitives::pipe::{Pipe, SendPipe};
use proxima_primitives::pipe::handler::{Handler, PipeHandle, ThreadLocalPipeHandle};
use proxima_primitives::pipe::request::{Request, Response};

/// half-open byte interval `[start, end)` keyed to a specific node's
/// emitted byte stream. zero-length ranges are valid (empty bodies).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ByteRange {
    pub start: u64,
    pub end: u64,
}

impl ByteRange {
    #[must_use]
    pub fn contains(&self, offset: u64) -> bool {
        offset >= self.start && offset < self.end
    }
}

/// edge in the causal graph: "node `node_id` emitted bytes `output_range`
/// derived from these `parent_ranges` on upstream nodes." default Handler
/// edges are coarse — whole input maps to whole output. Pipes that know
/// their byte-level mapping can emit finer edges by recording multiple
/// edges per call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEdge {
    pub node_id: String,
    pub output_range: ByteRange,
    pub parent_ranges: Vec<(String, ByteRange)>,
}

// Per-thread slot index cache. Each thread that records into a
// `CausalIndex` is assigned a stable slot index on first record,
// stored here. Keyed only on the thread identity, not on the
// `CausalIndex` instance — multiple `CausalIndex` instances share
// the same per-thread index because their `slots` vectors are the
// same size (currently true: all instances are constructed with
// `default_slot_count`).
thread_local! {
    static THREAD_SLOT: Cell<Option<usize>> = const { Cell::new(None) };
}

fn default_slot_count() -> usize {
    // Slot count scales with host CPU count for proxima per-core
    // deployments — one slot per core means each core's recorder
    // hits an uncontested Mutex.
    //
    // Capped at 64 to bound per-CausalIndex memory on extreme-core
    // hosts (DPDK / NUMA boxes). At 64 slots × ~32 bytes/slot = ~2 KB
    // per empty CausalIndex. CausalIndex instances are per-recording-
    // session (potentially per-request); 2 KB × 100k req/sec = 200 MB/
    // sec of allocation pressure — already at the upper bound. Going
    // higher needs explicit opt-in via `with_slots(N)`.
    //
    // For 128/256/512-core deployments where every core may record
    // into the SAME CausalIndex (e.g., a single long-lived recording
    // span across all cores), call `with_slots(num_cores)` explicitly.
    // The bench data says per-core scales linearly with slot count
    // (thread-local lookup + Mutex acquire are O(1) in slot count);
    // the only cost is memory per CausalIndex and per-snapshot
    // iteration of empty slots.
    std::cmp::min(num_cpus::get(), 64)
}

/// per-run causal store. cheap to clone (Arc internally) so it threads
/// through nested Handler calls without copies. `record` is fire-and-
/// forget; `explain` walks edges backward from an output byte offset.
///
/// implementation: per-thread sharded `Vec<Mutex<Vec<CausalEdge>>>`
/// (Stage 3c). Each writer thread is assigned a stable slot index on
/// first record (via `THREAD_SLOT` thread-local cache, hashed from
/// `thread::current().id()`). Writers on different threads hit
/// different Mutexes → no contention except via slot collision.
/// `edges()` iterates every slot, briefly locking each, and merges.
///
/// chosen via `benches/causal_record_primitives.rs`. Numbers for
/// `record` under N concurrent recorders (Linux host-b):
///
/// | N recorders | ArcSwap<Vec> CoW | Mutex<Vec> | per-core 16 slots |
/// |---|---|---|---|
/// | 0 (uncontested) | 775 µs | 58 ns | **57 ns** (tied) |
/// | 1 noise | 976 µs | 216 ns | **80 ns** (2.7×) |
/// | 4 noise | 1.5 ms | 628 ns | **97 ns** (6.5×) |
/// | 16 noise | 16.8 ms | 2.57 µs | **168 ns** (15×) |
///
/// Per-core wins at every contention level above 0, ties at 0. The
/// merged_edges aggregation property tests (`merged_edges_aggregates_
/// concurrent_recorders` + `explain_walks_chain_across_recorder_
/// threads`, mod tests) pin the contract this implementation must
/// satisfy: cross-thread edges aggregate without drops, per-thread
/// insertion order is preserved within that thread's slot.
///
/// Global cross-thread insertion order is NOT preserved (would
/// require per-record timestamping; the bench shows that's not
/// worth the cost). `explain` walks all slots, so chains crossing
/// thread boundaries still resolve.
#[derive(Clone)]
pub struct CausalIndex {
    slots: Arc<Vec<std::sync::Mutex<Vec<CausalEdge>>>>,
}

impl Default for CausalIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl CausalIndex {
    #[must_use]
    pub fn new() -> Self {
        Self::with_slots(default_slot_count())
    }

    /// Construct with `slot_count` per-thread slots. Use when a
    /// non-default per-core dispatch shape is in play (e.g., 8-core
    /// hosts → 8 slots is enough; 64-core DPDK → 64).
    #[must_use]
    pub fn with_slots(slot_count: usize) -> Self {
        let slot_count = slot_count.max(1);
        Self {
            slots: Arc::new(
                (0..slot_count)
                    .map(|_| std::sync::Mutex::new(Vec::new()))
                    .collect(),
            ),
        }
    }

    pub fn record(&self, edge: CausalEdge) {
        let slots_len = self.slots.len();
        let slot_idx = THREAD_SLOT.with(|cell| {
            cell.get().unwrap_or_else(|| {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                std::thread::current().id().hash(&mut hasher);
                let idx = (hasher.finish() % slots_len as u64) as usize;
                cell.set(Some(idx));
                idx
            })
        });
        // safety: slot_idx is always in range — derived modulo slots_len
        if let Ok(mut guard) = self.slots[slot_idx % slots_len].lock() {
            guard.push(edge);
        }
        // poisoned mutex drops the edge silently — preferable to
        // panicking inside causal record on a process that already
        // had a panic. callers are observability-only.
    }

    /// snapshot of every edge currently recorded. iterates every
    /// slot and concatenates in slot order — within a slot,
    /// insertion order is preserved; across slots, the order is
    /// "slot 0's edges, then slot 1's, etc." (NOT global insertion
    /// order). `explain` doesn't care about order beyond local
    /// chain walks; `write_jsonl` consumers should re-sort if global
    /// order is required.
    #[must_use]
    pub fn edges(&self) -> Vec<CausalEdge> {
        let mut all = Vec::new();
        for slot in self.slots.iter() {
            if let Ok(guard) = slot.lock() {
                all.extend(guard.iter().cloned());
            }
        }
        all
    }

    /// dump every edge to JSONL — one edge per line, suitable for piping
    /// into `proxima explain` after a run.
    pub fn write_jsonl(&self, path: &Path) -> Result<(), ProximaError> {
        let edges = self.edges();
        let mut buffer = String::new();
        for edge in &edges {
            let line = serde_json::to_string(edge)
                .map_err(|err| ProximaError::Encode(format!("causal edge: {err}")))?;
            buffer.push_str(&line);
            buffer.push('\n');
        }
        std::fs::write(path, buffer)?;
        Ok(())
    }

    /// load edges from a previously-written JSONL file. quiet on empty lines.
    pub fn read_jsonl(path: &Path) -> Result<Self, ProximaError> {
        let contents = std::fs::read_to_string(path)?;
        let index = Self::new();
        for line in contents.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let edge: CausalEdge = serde_json::from_str(line)
                .map_err(|err| ProximaError::Decode(format!("causal edge: {err}")))?;
            index.record(edge);
        }
        Ok(index)
    }

    /// walk backward from `(node_id, offset)` to the source. returns the
    /// chain of edges from the queried node up to whichever ancestor has
    /// no recorded parent. order: queried node first, then each parent in
    /// turn. duplicate hits (cycles) terminate the walk to avoid infinite
    /// loops on malformed graphs.
    #[must_use]
    pub fn explain(&self, node_id: &str, offset: u64) -> Vec<CausalEdge> {
        let snapshot = self.edges();
        let mut walked: Vec<CausalEdge> = Vec::new();
        let mut frontier: Option<(String, u64)> = Some((node_id.to_string(), offset));
        while let Some((current_node, current_offset)) = frontier.take() {
            let hit = snapshot.iter().find(|edge| {
                edge.node_id == current_node && edge.output_range.contains(current_offset)
            });
            let Some(edge) = hit else { break };
            if walked.iter().any(|prior| {
                prior.node_id == edge.node_id && prior.output_range == edge.output_range
            }) {
                break;
            }
            walked.push(edge.clone());
            if let Some((parent_node, parent_range)) = edge.parent_ranges.first() {
                frontier = Some((parent_node.clone(), parent_range.start));
            }
        }
        walked
    }
}

/// wrapper Handler that records coarse causal edges around its inner call.
/// pattern: collect the request body, call inner, collect the response
/// body, emit one edge "inner produced response bytes from request bytes".
///
/// for Pipes that need finer-grained edges (codec, transform, etc.),
/// add a more specialized recorder that emits multiple edges per call —
/// the index data structure is the same.
/// Causal-edge recorder. Generic over the inner handle:
/// `Causal<PipeHandle>` impls `Handler`;
/// `Causal<ThreadLocalPipeHandle>` impls `ThreadLocalHandler`.
pub struct Causal<Inner = PipeHandle> {
    pub inner: Inner,
    pub node_id: String,
    pub index: CausalIndex,
}

impl<Inner> Causal<Inner> {
    #[must_use]
    pub fn new(inner: Inner, node_id: impl Into<String>, index: CausalIndex) -> Self {
        Self {
            inner,
            node_id: node_id.into(),
            index,
        }
    }
}

impl<Inner> SendPipe for Causal<Inner>
where
    Inner: Handler + Clone,
{
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let inner = self.inner.clone();
        let node_id = self.node_id.clone();
        let index = self.index.clone();
        async move {
            let (request, request_bytes) = request.body_bytes().await?;
            let parent_node = request
                .context
                .upstream_label
                .as_ref()
                .map(|label| String::from_utf8_lossy(label).into_owned());
            let response = SendPipe::call(&inner, request).await?;
            let status = response.status;
            let headers = response.metadata.clone();
            let response_bytes = response.collect_body().await?;
            let output_range = ByteRange {
                start: 0,
                end: response_bytes.len() as u64,
            };
            let parent_ranges = parent_node
                .into_iter()
                .map(|name| {
                    (
                        name,
                        ByteRange {
                            start: 0,
                            end: request_bytes.len() as u64,
                        },
                    )
                })
                .collect();
            index.record(CausalEdge {
                node_id: node_id.clone(),
                output_range,
                parent_ranges,
            });
            let mut rebuilt = Response::new(status).with_body(response_bytes);
            for (name, value) in headers {
                rebuilt = rebuilt.with_header(name, value);
            }
            Ok(rebuilt)
        }
    }
}


impl Pipe for Causal<ThreadLocalPipeHandle> {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        let inner = self.inner.clone();
        let node_id = self.node_id.clone();
        let index = self.index.clone();
        async move {
            let (request, request_bytes) = request.body_bytes().await?;
            let parent_node = request
                .context
                .upstream_label
                .as_ref()
                .map(|label| String::from_utf8_lossy(label).into_owned());
            let response = Pipe::call(&inner, request).await?;
            let status = response.status;
            let headers = response.metadata.clone();
            let response_bytes = response.collect_body().await?;
            let output_range = ByteRange {
                start: 0,
                end: response_bytes.len() as u64,
            };
            let parent_ranges = parent_node
                .into_iter()
                .map(|name| {
                    (
                        name,
                        ByteRange {
                            start: 0,
                            end: request_bytes.len() as u64,
                        },
                    )
                })
                .collect();
            index.record(CausalEdge {
                node_id: node_id.clone(),
                output_range,
                parent_ranges,
            });
            let mut rebuilt = Response::new(status).with_body(response_bytes);
            for (name, value) in headers {
                rebuilt = rebuilt.with_header(name, value);
            }
            Ok(rebuilt)
        }
    }
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use proxima_primitives::pipe::handler::into_handle;

    fn range(start: u64, end: u64) -> ByteRange {
        ByteRange { start, end }
    }

    #[test]
    fn byte_range_contains_is_half_open() {
        let r = range(10, 20);
        assert!(!r.contains(9));
        assert!(r.contains(10));
        assert!(r.contains(19));
        assert!(!r.contains(20));
    }

    #[test]
    fn explain_walks_back_through_chained_edges() {
        let index = CausalIndex::new();
        index.record(CausalEdge {
            node_id: "source".into(),
            output_range: range(0, 100),
            parent_ranges: Vec::new(),
        });
        index.record(CausalEdge {
            node_id: "transform".into(),
            output_range: range(0, 50),
            parent_ranges: vec![("source".into(), range(0, 100))],
        });
        index.record(CausalEdge {
            node_id: "sink".into(),
            output_range: range(0, 50),
            parent_ranges: vec![("transform".into(), range(0, 50))],
        });
        let chain = index.explain("sink", 25);
        let ids: Vec<&str> = chain.iter().map(|edge| edge.node_id.as_str()).collect();
        assert_eq!(ids, vec!["sink", "transform", "source"]);
    }

    #[test]
    fn explain_returns_empty_when_offset_outside_any_recorded_range() {
        let index = CausalIndex::new();
        index.record(CausalEdge {
            node_id: "alpha".into(),
            output_range: range(0, 10),
            parent_ranges: Vec::new(),
        });
        let chain = index.explain("alpha", 100);
        assert!(chain.is_empty(), "out-of-range offset returns no walk");
    }

    #[test]
    fn explain_terminates_on_cycles() {
        let index = CausalIndex::new();
        index.record(CausalEdge {
            node_id: "a".into(),
            output_range: range(0, 5),
            parent_ranges: vec![("b".into(), range(0, 5))],
        });
        index.record(CausalEdge {
            node_id: "b".into(),
            output_range: range(0, 5),
            parent_ranges: vec![("a".into(), range(0, 5))],
        });
        let chain = index.explain("a", 0);
        // a → b → a (cycle detected, walk stops). two unique edges.
        assert_eq!(chain.len(), 2);
    }

    #[proxima::test]
    async fn causal_pipe_records_edge_around_inner_call() {
        struct Doubler;
        impl SendPipe for Doubler {
            type In = Request<Bytes>;
            type Out = Response<Bytes>;
            type Err = ProximaError;

            fn call(
                &self,
                request: Request<Bytes>,
            ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
                async move {
                    let (_, body) = request.body_bytes().await?;
                    let mut doubled = Vec::with_capacity(body.len() * 2);
                    doubled.extend_from_slice(&body);
                    doubled.extend_from_slice(&body);
                    Ok(Response::ok(Bytes::from(doubled)))
                }
            }
        }

        let index = CausalIndex::new();
        let recorder = Causal::new(into_handle(Doubler), "doubler", index.clone());
        let request = Request::builder()
            .method("POST")
            .path("/x")
            .body("abc")
            .build()
            .expect("request");
        let response = SendPipe::call(&recorder, request).await.expect("call");
        let body = response.collect_body().await.expect("body");
        assert_eq!(&body[..], b"abcabc");
        let edges = index.edges();
        assert_eq!(edges.len(), 1);
        let edge = &edges[0];
        assert_eq!(edge.node_id, "doubler");
        assert_eq!(edge.output_range, range(0, 6));
    }

    /// Property contract for `CausalIndex::edges()` (a.k.a. merged_edges
    /// under the future Stage 3c per-core variant):
    ///
    /// Given N threads each recording M edges concurrently into a
    /// shared `CausalIndex`, `edges()` returns a vec of length N*M
    /// containing every recorded edge exactly once.
    ///
    /// Global insertion order across threads is NOT part of the
    /// contract (per-core implementations can't preserve it without
    /// per-record timestamping). Within a single thread, insertion
    /// order IS preserved — a thread that recorded edges in sequence
    /// A1, A2, A3 will see A1 before A2 before A3 in the merged view.
    ///
    /// This is the property the Stage 3c per-core implementation must
    /// continue to honor. Today (Mutex<Vec>) it's trivially satisfied
    /// because all edges go into one Vec; the test still pins the
    /// contract so a future refactor can't silently drop edges.
    #[test]
    fn merged_edges_aggregates_concurrent_recorders() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::thread;

        const THREADS: usize = 8;
        const EDGES_PER_THREAD: usize = 64;

        let index = Arc::new(CausalIndex::new());
        let started_threads = Arc::new(AtomicUsize::new(0));
        let go = Arc::new(std::sync::Barrier::new(THREADS));

        let mut handles = Vec::with_capacity(THREADS);
        for thread_index in 0..THREADS {
            let index = index.clone();
            let started = started_threads.clone();
            let go = go.clone();
            handles.push(thread::spawn(move || {
                started.fetch_add(1, Ordering::SeqCst);
                go.wait();
                for edge_index in 0..EDGES_PER_THREAD {
                    index.record(CausalEdge {
                        node_id: format!("t{thread_index}-e{edge_index}"),
                        output_range: ByteRange {
                            start: edge_index as u64 * 16,
                            end: edge_index as u64 * 16 + 16,
                        },
                        parent_ranges: Vec::new(),
                    });
                }
            }));
        }
        for handle in handles {
            handle.join().expect("recorder thread joined");
        }

        let merged = index.edges();

        // 1. Count: every edge is present exactly once
        assert_eq!(merged.len(), THREADS * EDGES_PER_THREAD);

        let unique_ids: std::collections::HashSet<&str> =
            merged.iter().map(|edge| edge.node_id.as_str()).collect();
        assert_eq!(unique_ids.len(), THREADS * EDGES_PER_THREAD);

        // 2. Set equality: every (thread, edge) combination is present
        for thread_index in 0..THREADS {
            for edge_index in 0..EDGES_PER_THREAD {
                let expected = format!("t{thread_index}-e{edge_index}");
                assert!(
                    unique_ids.contains(expected.as_str()),
                    "merged edges missing {expected}",
                );
            }
        }

        // 3. Per-thread order preserved: for any thread, its edges
        //    appear in the merged view in the order it recorded them.
        for thread_index in 0..THREADS {
            let prefix = format!("t{thread_index}-");
            let observed: Vec<&str> = merged
                .iter()
                .filter_map(|edge| {
                    if edge.node_id.starts_with(&prefix) {
                        Some(edge.node_id.as_str())
                    } else {
                        None
                    }
                })
                .collect();
            assert_eq!(observed.len(), EDGES_PER_THREAD);
            for (edge_index, node_id) in observed.iter().enumerate().take(EDGES_PER_THREAD) {
                let expected = format!("t{thread_index}-e{edge_index}");
                assert_eq!(
                    *node_id, expected,
                    "thread {thread_index} edge {edge_index} out of order",
                );
            }
        }
    }

    /// Sibling property: `explain()` walks the merged view, so it
    /// must find a chain even when the ancestor edges were recorded
    /// by a different thread than the descendant. This pins
    /// cross-thread `explain` correctness — Stage 3c per-core must
    /// still produce coherent chains across cores.
    #[test]
    fn explain_walks_chain_across_recorder_threads() {
        use std::sync::Arc;
        use std::thread;

        let index = Arc::new(CausalIndex::new());

        // Thread A records the source.
        let source_handle = {
            let index = index.clone();
            thread::spawn(move || {
                index.record(CausalEdge {
                    node_id: "source".into(),
                    output_range: range(0, 100),
                    parent_ranges: Vec::new(),
                });
            })
        };
        source_handle.join().expect("source thread");

        // Thread B records the transform that references the source.
        let transform_handle = {
            let index = index.clone();
            thread::spawn(move || {
                index.record(CausalEdge {
                    node_id: "transform".into(),
                    output_range: range(0, 50),
                    parent_ranges: vec![("source".into(), range(0, 100))],
                });
            })
        };
        transform_handle.join().expect("transform thread");

        // Thread C records the sink that references the transform.
        let sink_handle = {
            let index = index.clone();
            thread::spawn(move || {
                index.record(CausalEdge {
                    node_id: "sink".into(),
                    output_range: range(0, 50),
                    parent_ranges: vec![("transform".into(), range(0, 50))],
                });
            })
        };
        sink_handle.join().expect("sink thread");

        // explain() walks the merged view from main thread. Chain
        // crosses all three recorder-thread boundaries.
        let chain = index.explain("sink", 25);
        let ids: Vec<&str> = chain.iter().map(|edge| edge.node_id.as_str()).collect();
        assert_eq!(ids, vec!["sink", "transform", "source"]);
    }
}
