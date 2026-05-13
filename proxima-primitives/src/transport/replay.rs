use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};

use bytes::Bytes;
use crossbeam_queue::ArrayQueue;
use futures::Stream;
use futures::task::AtomicWaker;

use proxima_core::ProximaError;

use crate::transport::stream::GenericStream;

pub const DEFAULT_REPLAY_CAP_BYTES: usize = 4 * 1024 * 1024;
pub const DEFAULT_SINK_QUEUE: usize = 16;

/// A stream that records items as they flow through, enabling replay and
/// fan-out to additional sinks. `T` is the item type.
///
/// The cap is a "weight" measured by the `weight_fn`: for `Replay<Bytes>`
/// the canonical constructor [`Replay::wrap_bytes`] sets `weight_fn` to
/// `Bytes::len` so the cap is in bytes (matching the old body-replay
/// contract). For any other `T` the default constructors [`Replay::wrap`]
/// and [`Replay::wrap_with`] weight each item as 1 (item-count cap).
pub struct Replay<T: Send + Clone + 'static> {
    inner: Arc<ReplayInner<T>>,
}

struct ReplayInner<T: Send + Clone + 'static> {
    cap_weight: usize,
    sink_queue: usize,
    weight_fn: Arc<dyn Fn(&T) -> usize + Send + Sync + 'static>,
    // WHY Mutex: multi-field state machine atomicity (item log + sink list +
    // pending broadcast + done/error flags). Per-sink fan-out uses
    // ArrayQueue + AtomicWaker (proven 5-34× faster than tokio mpsc). The
    // Mutex guards the multi-field invariant only.
    state: Mutex<ReplayState<T>>,
}

struct ReplayState<T: Send + Clone + 'static> {
    chunks: Vec<T>,
    recorded_weight: usize,
    capped: bool,
    inner_done: bool,
    inner_error: Option<String>,
    inner_stream: Option<GenericStream<T>>,
    sinks: Vec<SinkSlot<T>>,
    pending_event: Option<ReplayEvent<T>>,
}

struct SinkSlot<T: Send + Clone + 'static> {
    queue: Arc<ArrayQueue<ReplayEvent<T>>>,
    consumer_waker: Arc<AtomicWaker>,
    producer_waker: Arc<AtomicWaker>,
    closed: Arc<AtomicBool>,
}

#[derive(Clone)]
pub enum ReplayEvent<T> {
    Item(T),
    End,
    Error(String),
}

impl<T: Send + Clone + 'static> Replay<T> {
    /// Item-count–capped constructor: each item weighs 1.
    #[must_use]
    pub fn wrap<S>(stream: S, cap_items: usize) -> (Self, GenericStream<T>)
    where
        S: Stream<Item = Result<T, ProximaError>> + Send + 'static,
    {
        Self::new_with_weight(stream, cap_items, DEFAULT_SINK_QUEUE, Arc::new(|_| 1))
    }

    /// Item-count–capped constructor with an explicit sink-queue size.
    #[must_use]
    pub fn wrap_with<S>(stream: S, cap_items: usize, sink_queue: usize) -> (Self, GenericStream<T>)
    where
        S: Stream<Item = Result<T, ProximaError>> + Send + 'static,
    {
        Self::new_with_weight(stream, cap_items, sink_queue, Arc::new(|_| 1))
    }

    /// General constructor: caller supplies the weight function and sink-queue.
    #[must_use]
    pub fn new_with_weight<S>(
        stream: S,
        cap_weight: usize,
        sink_queue: usize,
        weight_fn: Arc<dyn Fn(&T) -> usize + Send + Sync + 'static>,
    ) -> (Self, GenericStream<T>)
    where
        S: Stream<Item = Result<T, ProximaError>> + Send + 'static,
    {
        let inner = Arc::new(ReplayInner {
            cap_weight,
            sink_queue: sink_queue.max(1),
            weight_fn,
            state: Mutex::new(ReplayState {
                chunks: Vec::new(),
                recorded_weight: 0,
                capped: false,
                inner_done: false,
                inner_error: None,
                inner_stream: Some(Box::pin(stream)),
                sinks: Vec::new(),
                pending_event: None,
            }),
        });
        let primary: GenericStream<T> = Box::pin(PrimaryStream {
            inner: inner.clone(),
        });
        (Self { inner }, primary)
    }

    pub fn replay(&self) -> Result<GenericStream<T>, ProximaError> {
        let state = lock_state(&self.inner.state);
        if state.capped {
            return Err(ProximaError::Body(format!(
                "tee replay unavailable: exceeded {} weight",
                self.inner.cap_weight
            )));
        }
        if let Some(message) = state.inner_error.as_deref() {
            return Err(ProximaError::Body(format!(
                "tee inner stream errored: {message}"
            )));
        }
        if !state.inner_done {
            return Ok(Box::pin(ReplayStream {
                inner: self.inner.clone(),
                cursor: 0,
                drained: false,
            }));
        }
        let recorded = state.chunks.clone();
        Ok(Box::pin(futures::stream::iter(
            recorded.into_iter().map(Ok::<T, ProximaError>),
        )))
    }

    #[must_use]
    pub fn sink(&self) -> GenericStream<T> {
        let queue = Arc::new(ArrayQueue::new(self.inner.sink_queue));
        let consumer_waker = Arc::new(AtomicWaker::new());
        let producer_waker = Arc::new(AtomicWaker::new());
        let closed = Arc::new(AtomicBool::new(false));
        {
            let mut state = lock_state(&self.inner.state);
            state.sinks.push(SinkSlot {
                queue: queue.clone(),
                consumer_waker: consumer_waker.clone(),
                producer_waker: producer_waker.clone(),
                closed: closed.clone(),
            });
        }
        Box::pin(SinkStream {
            queue,
            consumer_waker,
            producer_waker,
            closed,
        })
    }

    #[must_use]
    pub fn cap_weight(&self) -> usize {
        self.inner.cap_weight
    }

    #[must_use]
    pub fn recorded_weight(&self) -> usize {
        lock_state(&self.inner.state).recorded_weight
    }

    #[must_use]
    pub fn capped(&self) -> bool {
        lock_state(&self.inner.state).capped
    }
}

impl Replay<Bytes> {
    /// Byte-capped constructor for `Replay<Bytes>`: weight = `Bytes::len`.
    /// This is the canonical byte-replay used by HTTP retry — identical
    /// behavior to the old body-specific `Replay` that lived in
    /// the now-dissolved `proxima-graph/src/tee/body.rs` (cap_bytes → weight cap in bytes,
    /// not item count).
    #[must_use]
    pub fn wrap_bytes<S>(stream: S, cap_bytes: usize) -> (Self, GenericStream<Bytes>)
    where
        S: Stream<Item = Result<Bytes, ProximaError>> + Send + 'static,
    {
        Self::wrap_bytes_with(stream, cap_bytes, DEFAULT_SINK_QUEUE)
    }

    #[must_use]
    pub fn wrap_bytes_with<S>(
        stream: S,
        cap_bytes: usize,
        sink_queue: usize,
    ) -> (Self, GenericStream<Bytes>)
    where
        S: Stream<Item = Result<Bytes, ProximaError>> + Send + 'static,
    {
        Self::new_with_weight(stream, cap_bytes, sink_queue, Arc::new(Bytes::len))
    }
}

/// Degenerate alias: byte-chunk replay, the form used by HTTP retry.
pub type BytesReplay = Replay<Bytes>;

struct PrimaryStream<T: Send + Clone + 'static> {
    inner: Arc<ReplayInner<T>>,
}

impl<T: Send + Clone + 'static> Stream for PrimaryStream<T> {
    type Item = Result<T, ProximaError>;

    fn poll_next(self: Pin<&mut Self>, ctx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let mut state = lock_state(&self.inner.state);
        let cap = self.inner.cap_weight;
        let weight_fn = self.inner.weight_fn.clone();
        loop {
            if state.pending_event.is_some() {
                match poll_broadcast(&mut state, ctx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(event) => return primary_yield(event),
                }
            }
            let Some(stream) = state.inner_stream.as_mut() else {
                return Poll::Ready(None);
            };
            match Pin::new(stream).poll_next(ctx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    state.inner_done = true;
                    state.inner_stream = None;
                    state.pending_event = Some(ReplayEvent::End);
                }
                Poll::Ready(Some(Err(error))) => {
                    let message = error.to_string();
                    state.inner_error = Some(message.clone());
                    state.inner_done = true;
                    state.inner_stream = None;
                    state.pending_event = Some(ReplayEvent::Error(message));
                }
                Poll::Ready(Some(Ok(item))) => {
                    record(&mut state, cap, weight_fn.as_ref(), item.clone());
                    state.pending_event = Some(ReplayEvent::Item(item));
                }
            }
        }
    }
}

fn primary_yield<T: Send + Clone + 'static>(
    event: ReplayEvent<T>,
) -> Poll<Option<Result<T, ProximaError>>> {
    match event {
        ReplayEvent::Item(item) => Poll::Ready(Some(Ok(item))),
        ReplayEvent::End => Poll::Ready(None),
        ReplayEvent::Error(message) => Poll::Ready(Some(Err(ProximaError::Body(message)))),
    }
}

struct ReplayStream<T: Send + Clone + 'static> {
    inner: Arc<ReplayInner<T>>,
    cursor: usize,
    drained: bool,
}

impl<T: Send + Clone + 'static> Stream for ReplayStream<T> {
    type Item = Result<T, ProximaError>;

    fn poll_next(self: Pin<&mut Self>, ctx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let cap = this.inner.cap_weight;
        let weight_fn = this.inner.weight_fn.clone();
        let mut state = lock_state(&this.inner.state);
        loop {
            if state.capped {
                return Poll::Ready(Some(Err(ProximaError::Body(format!(
                    "tee replay aborted: exceeded {cap} weight"
                )))));
            }
            if state.pending_event.is_some() {
                match poll_broadcast(&mut state, ctx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(ReplayEvent::Item(item)) => {
                        this.cursor = state.chunks.len();
                        return Poll::Ready(Some(Ok(item)));
                    }
                    Poll::Ready(ReplayEvent::End) => {
                        this.drained = true;
                        return Poll::Ready(None);
                    }
                    Poll::Ready(ReplayEvent::Error(message)) => {
                        return Poll::Ready(Some(Err(ProximaError::Body(message))));
                    }
                }
            }
            if this.cursor < state.chunks.len() {
                let item = state.chunks[this.cursor].clone();
                this.cursor += 1;
                return Poll::Ready(Some(Ok(item)));
            }
            if this.drained || state.inner_done {
                this.drained = true;
                return Poll::Ready(None);
            }
            let Some(stream) = state.inner_stream.as_mut() else {
                return Poll::Ready(None);
            };
            match Pin::new(stream).poll_next(ctx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    state.inner_done = true;
                    state.inner_stream = None;
                    state.pending_event = Some(ReplayEvent::End);
                }
                Poll::Ready(Some(Err(error))) => {
                    let message = error.to_string();
                    state.inner_error = Some(message.clone());
                    state.inner_done = true;
                    state.inner_stream = None;
                    state.pending_event = Some(ReplayEvent::Error(message));
                }
                Poll::Ready(Some(Ok(item))) => {
                    record(&mut state, cap, weight_fn.as_ref(), item.clone());
                    state.pending_event = Some(ReplayEvent::Item(item));
                }
            }
        }
    }
}

struct SinkStream<T: Send + Clone + 'static> {
    queue: Arc<ArrayQueue<ReplayEvent<T>>>,
    consumer_waker: Arc<AtomicWaker>,
    producer_waker: Arc<AtomicWaker>,
    closed: Arc<AtomicBool>,
}

impl<T: Send + Clone + 'static> SinkStream<T> {
    fn yield_event(&self, event: ReplayEvent<T>) -> Poll<Option<Result<T, ProximaError>>> {
        self.producer_waker.wake();
        match event {
            ReplayEvent::Item(item) => Poll::Ready(Some(Ok(item))),
            ReplayEvent::End => Poll::Ready(None),
            ReplayEvent::Error(message) => Poll::Ready(Some(Err(ProximaError::Body(message)))),
        }
    }
}

impl<T: Send + Clone + 'static> Stream for SinkStream<T> {
    type Item = Result<T, ProximaError>;

    fn poll_next(self: Pin<&mut Self>, ctx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(event) = self.queue.pop() {
            return self.yield_event(event);
        }
        self.consumer_waker.register(ctx.waker());
        if let Some(event) = self.queue.pop() {
            return self.yield_event(event);
        }
        Poll::Pending
    }
}

impl<T: Send + Clone + 'static> Drop for SinkStream<T> {
    fn drop(&mut self) {
        self.closed.store(true, Ordering::Release);
        self.producer_waker.wake();
    }
}

fn lock_state<T: Send + Clone + 'static>(
    state: &Mutex<ReplayState<T>>,
) -> std::sync::MutexGuard<'_, ReplayState<T>> {
    state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn record<T: Send + Clone + 'static>(
    state: &mut ReplayState<T>,
    cap_weight: usize,
    weight_fn: &dyn Fn(&T) -> usize,
    item: T,
) {
    if state.capped {
        return;
    }
    let item_weight = weight_fn(&item);
    let next_total = state.recorded_weight.saturating_add(item_weight);
    if next_total > cap_weight {
        state.capped = true;
        state.chunks.clear();
        state.recorded_weight = 0;
        return;
    }
    state.recorded_weight = next_total;
    state.chunks.push(item);
}

fn poll_broadcast<T: Send + Clone + 'static>(
    state: &mut ReplayState<T>,
    ctx: &mut Context<'_>,
) -> Poll<ReplayEvent<T>> {
    let event = match state.pending_event.as_ref() {
        Some(event) => event.clone(),
        None => return Poll::Pending,
    };
    state
        .sinks
        .retain(|slot| !slot.closed.load(Ordering::Acquire));

    let mut idx = 0;
    while idx < state.sinks.len() {
        let slot = &state.sinks[idx];
        match slot.queue.push(event.clone()) {
            Ok(()) => {
                slot.consumer_waker.wake();
                idx += 1;
            }
            Err(_) => {
                slot.producer_waker.register(ctx.waker());
                match slot.queue.push(event.clone()) {
                    Ok(()) => {
                        slot.consumer_waker.wake();
                        idx += 1;
                    }
                    Err(_) => return Poll::Pending,
                }
            }
        }
    }
    state.pending_event = None;
    Poll::Ready(event)
}

/// Tap stream: captures all items and fires a callback on EOF.
#[must_use]
pub fn tap_complete<T, F>(
    stream: GenericStream<T>,
    cap_weight: usize,
    on_complete: F,
) -> GenericStream<T>
where
    T: Send + Clone + 'static,
    F: FnOnce(Vec<T>) + Send + 'static,
{
    tap_complete_with_size(stream, cap_weight, None, on_complete)
}

#[must_use]
pub fn tap_complete_with_size<T, F>(
    stream: GenericStream<T>,
    cap_weight: usize,
    expected_total: Option<usize>,
    on_complete: F,
) -> GenericStream<T>
where
    T: Send + Clone + 'static,
    F: FnOnce(Vec<T>) + Send + 'static,
{
    let state = TapState {
        chunks: Vec::new(),
        captured_items: 0,
        capped: false,
        on_complete: Some(Box::new(on_complete)),
    };
    Box::pin(TapStream {
        inner: stream,
        cap_weight,
        expected_total,
        state,
    })
}

struct TapStream<T: Send + Clone + 'static> {
    inner: GenericStream<T>,
    cap_weight: usize,
    expected_total: Option<usize>,
    state: TapState<T>,
}

struct TapState<T: Send + Clone + 'static> {
    chunks: Vec<T>,
    captured_items: usize,
    capped: bool,
    on_complete: Option<Box<dyn FnOnce(Vec<T>) + Send>>,
}

// all fields of TapStream<T> are Unpin regardless of T:
//   - GenericStream<T> = Pin<Box<...>> which is always Unpin
//   - usize, Option<usize>, Vec<T>, Option<Box<dyn FnOnce(...)>> are all Unpin
// the compiler's auto-derived bound is overly conservative.
impl<T: Send + Clone + 'static> Unpin for TapStream<T> {}

impl<T: Send + Clone + 'static> TapState<T> {
    fn fire_complete(&mut self) {
        if self.capped {
            return;
        }
        if let Some(callback) = self.on_complete.take() {
            let chunks = std::mem::take(&mut self.chunks);
            callback(chunks);
        }
    }
}

impl<T: Send + Clone + 'static> Stream for TapStream<T> {
    type Item = Result<T, ProximaError>;

    fn poll_next(self: Pin<&mut Self>, ctx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match Pin::new(&mut this.inner).poll_next(ctx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => {
                this.state.fire_complete();
                Poll::Ready(None)
            }
            Poll::Ready(Some(Err(error))) => {
                this.state.on_complete = None;
                Poll::Ready(Some(Err(error)))
            }
            Poll::Ready(Some(Ok(chunk))) => {
                if !this.state.capped {
                    let next = this.state.captured_items.saturating_add(1);
                    if next > this.cap_weight {
                        this.state.capped = true;
                        this.state.chunks.clear();
                        this.state.captured_items = 0;
                        this.state.on_complete = None;
                    } else {
                        this.state.captured_items = next;
                        this.state.chunks.push(chunk.clone());
                        if let Some(total) = this.expected_total
                            && this.state.captured_items >= total
                        {
                            this.state.fire_complete();
                        }
                    }
                }
                Poll::Ready(Some(Ok(chunk)))
            }
        }
    }
}

// `#[proxima::test]` and inline `tokio::spawn` pull in the `proxima` /
// `tokio` dev-dependencies, which the loom build keeps out of the graph
// (see `[target.'cfg(not(loom))'.dev-dependencies]` in Cargo.toml); these
// tests are unrelated to the Notify/watch loom protocol.
#[cfg(all(test, not(loom)))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use futures::StreamExt;

    async fn drain_bytes(mut stream: GenericStream<Bytes>) -> Result<Bytes, ProximaError> {
        let mut buffer = bytes::BytesMut::new();
        while let Some(item) = stream.next().await {
            buffer.extend_from_slice(&item?);
        }
        Ok(buffer.freeze())
    }

    async fn drain_chunks(mut stream: GenericStream<Bytes>) -> Result<Vec<Bytes>, ProximaError> {
        let mut chunks = Vec::new();
        while let Some(item) = stream.next().await {
            chunks.push(item?);
        }
        Ok(chunks)
    }

    fn small_body() -> GenericStream<Bytes> {
        Box::pin(futures::stream::iter([
            Ok(Bytes::from_static(b"hel")),
            Ok(Bytes::from_static(b"lo")),
        ]))
    }

    #[proxima::test]
    async fn primary_yields_chunks_in_order() {
        let (_tee, primary) = Replay::wrap_bytes(small_body(), DEFAULT_REPLAY_CAP_BYTES);
        let bytes = drain_bytes(primary).await.expect("collect primary");
        assert_eq!(&bytes[..], b"hello");
    }

    #[proxima::test]
    async fn replay_after_full_consume_returns_same_bytes() {
        let (tee, primary) = Replay::wrap_bytes(small_body(), DEFAULT_REPLAY_CAP_BYTES);
        let first = drain_bytes(primary).await.expect("primary");
        assert_eq!(&first[..], b"hello");
        let replay = tee.replay().expect("replay available");
        let second = drain_bytes(replay).await.expect("replay collect");
        assert_eq!(&second[..], b"hello");
    }

    #[proxima::test]
    async fn replay_drains_inner_when_primary_abandoned() {
        let (tee, primary) = Replay::wrap_bytes(small_body(), DEFAULT_REPLAY_CAP_BYTES);
        drop(primary);
        let replay = tee.replay().expect("replay available");
        let bytes = drain_bytes(replay).await.expect("replay collect");
        assert_eq!(&bytes[..], b"hello");
    }

    #[proxima::test]
    async fn replay_errs_when_byte_cap_exceeded() {
        let (tee, primary) = Replay::wrap_bytes(small_body(), 3);
        let _ = drain_bytes(primary).await.expect("primary collect");
        let outcome = tee.replay();
        assert!(matches!(outcome, Err(ProximaError::Body(_))));
        assert!(tee.capped());
    }

    #[proxima::test]
    async fn sink_receives_chunks_alongside_primary() {
        let (tee, primary) = Replay::wrap_bytes(small_body(), DEFAULT_REPLAY_CAP_BYTES);
        let sink = tee.sink();
        let primary_task = tokio::spawn(async move { drain_bytes(primary).await });
        let sink_chunks = drain_chunks(sink).await.expect("sink chunks");
        let primary_bytes = primary_task.await.expect("join").expect("primary");
        assert_eq!(&primary_bytes[..], b"hello");
        let joined: Vec<u8> = sink_chunks
            .iter()
            .flat_map(|chunk| chunk.to_vec())
            .collect();
        assert_eq!(&joined[..], b"hello");
    }

    #[proxima::test]
    async fn tap_complete_invokes_callback_after_eof() {
        let captured: Arc<std::sync::Mutex<Option<Vec<Bytes>>>> =
            Arc::new(std::sync::Mutex::new(None));
        let captured_for_cb = captured.clone();
        let body = tap_complete(small_body(), DEFAULT_REPLAY_CAP_BYTES, move |chunks| {
            *captured_for_cb.lock().expect("lock") = Some(chunks);
        });
        let bytes = drain_bytes(body).await.expect("collect");
        assert_eq!(&bytes[..], b"hello");
        let chunks = captured
            .lock()
            .expect("lock")
            .take()
            .expect("callback fired");
        assert_eq!(chunks.iter().map(Bytes::len).sum::<usize>(), 5);
    }

    #[proxima::test]
    async fn tap_complete_skips_callback_when_dropped_before_eof() {
        let captured: Arc<std::sync::Mutex<Option<Vec<Bytes>>>> =
            Arc::new(std::sync::Mutex::new(None));
        let captured_for_cb = captured.clone();
        let body = tap_complete(small_body(), DEFAULT_REPLAY_CAP_BYTES, move |chunks| {
            *captured_for_cb.lock().expect("lock") = Some(chunks);
        });
        let mut stream = body;
        let _ = stream.next().await.expect("first").expect("ok");
        drop(stream);
        assert!(
            captured.lock().expect("lock").is_none(),
            "no EOF, no callback"
        );
    }

    #[proxima::test]
    async fn tap_complete_with_size_fires_when_total_reached_without_eof() {
        let captured: Arc<std::sync::Mutex<Option<Vec<Bytes>>>> =
            Arc::new(std::sync::Mutex::new(None));
        let captured_for_cb = captured.clone();
        let body: GenericStream<Bytes> =
            Box::pin(futures::stream::unfold(false, |yielded| async move {
                if yielded {
                    futures::future::pending::<()>().await;
                    None
                } else {
                    Some((Ok(Bytes::from_static(b"hello")), true))
                }
            }));
        let tapped =
            tap_complete_with_size(body, DEFAULT_REPLAY_CAP_BYTES, Some(1), move |chunks| {
                *captured_for_cb.lock().expect("lock") = Some(chunks);
            });
        let mut stream = tapped;
        let chunk = stream.next().await.expect("first chunk").expect("ok");
        assert_eq!(&chunk[..], b"hello");
        let recorded = captured
            .lock()
            .expect("lock")
            .take()
            .expect("callback fired at total");
        assert_eq!(recorded.iter().map(Bytes::len).sum::<usize>(), 5);
    }

    #[proxima::test]
    async fn tap_complete_skips_callback_when_cap_exceeded() {
        let captured: Arc<std::sync::Mutex<Option<Vec<Bytes>>>> =
            Arc::new(std::sync::Mutex::new(None));
        let captured_for_cb = captured.clone();
        let body = tap_complete(small_body(), 1, move |chunks| {
            *captured_for_cb.lock().expect("lock") = Some(chunks);
        });
        let bytes = drain_bytes(body).await.expect("collect");
        assert_eq!(&bytes[..], b"hello");
        assert!(
            captured.lock().expect("lock").is_none(),
            "cap exceeded, no callback"
        );
    }

    #[proxima::test]
    async fn primary_propagates_inner_error() {
        let body: GenericStream<Bytes> = Box::pin(futures::stream::iter([
            Ok(Bytes::from_static(b"part")),
            Err(ProximaError::Body("upstream broke".into())),
        ]));
        let (tee, primary) = Replay::wrap_bytes(body, DEFAULT_REPLAY_CAP_BYTES);
        let mut stream = primary;
        let first = stream.next().await.expect("first chunk").expect("ok");
        assert_eq!(&first[..], b"part");
        let err = stream
            .next()
            .await
            .expect("second item")
            .expect_err("error");
        assert!(matches!(err, ProximaError::Body(_)));
        let outcome = tee.replay();
        assert!(matches!(outcome, Err(ProximaError::Body(_))));
    }

    #[proxima::test]
    async fn bytes_replay_is_byte_capped_not_item_capped() {
        // "hel" (3 bytes) and "lo" (2 bytes) = 5 total bytes.
        // cap at 4 bytes: should cap since 3 + 2 = 5 > 4.
        let (tee, primary) = Replay::wrap_bytes(small_body(), 4);
        let _ = drain_bytes(primary).await.expect("primary collect");
        assert!(tee.capped(), "byte cap (4) exceeded by 5-byte body");
    }
}
