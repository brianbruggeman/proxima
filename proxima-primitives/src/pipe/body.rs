//! Spine payload types after the `Body` collapse (yank-body).
//!
//! The `Pipe` spine carries raw `Bytes` for buffered bodies. Two
//! sibling concerns ride beside the bytes, each its own type so the
//! buffered 80% case pays nothing for their existence:
//!
//! - [`Carry`] — an erased `Arc<dyn Any + Send + Sync>` for in-process
//!   typed payloads (telemetry records) that must NOT serialize to
//!   bytes. Clone is one atomic Arc bump — the zero-copy fan-out.
//! - [`ResponseStream`] / [`RequestStream`] — explicit streaming bodies
//!   built on a `Send`-bounded [`ChunkStream`]. Streaming is a separate
//!   contract, not a body variant; the `+ Send` decision is contained
//!   here in [`ChunkStream`], not smeared across every buffered RPC.
//!
//! Response trailers (RFC 7230 §4.1.2) are stream-completion metadata
//! and live on [`ResponseStream`] behind `std`. Cancellation is
//! per-request and lives on `RequestContext.cancel`, not on the body.

#![cfg(feature = "alloc")]

use core::any::Any;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use bytes::Bytes;
use futures::stream::Stream;
use futures::stream::StreamExt;

#[cfg(feature = "std")]
use alloc::sync::Arc;
#[cfg(feature = "std")]
use proxima_core::signal::Signal;
#[cfg(feature = "std")]
use std::sync::Mutex;

#[cfg(not(feature = "std"))]
use alloc::sync::Arc;

#[cfg(feature = "std")]
use crate::pipe::header_list::HeaderList;
use proxima_core::ProximaError;

/// A `Send`-bounded stream of body chunks. The single named home of the
/// streaming `+ Send` decision (yank-body): a streamed body yields owned
/// `Bytes` chunks, so producer-side `!Send` state stays behind whatever
/// channel feeds this. Reused by [`ResponseStream`] and [`RequestStream`].
pub type ChunkStream = Pin<Box<dyn Stream<Item = Result<Bytes, ProximaError>> + Send>>;

/// Trailers slot — published by a chunked producer at stream end.
/// std-only: needs `Mutex`. Lives on [`ResponseStream`], never on the
/// buffered spine (trailers are meaningless for a complete `Bytes` body).
#[cfg(feature = "std")]
pub type TrailersSlot = Arc<Mutex<Option<HeaderList>>>;

/// In-process typed payload — an erased `Arc<dyn Any + Send + Sync>`.
/// Telemetry records (`SpanRecord`, `LogRecord`, …) ride here without
/// serializing to `Bytes`; the in-process Pipe chain downcasts them via
/// [`Carry::downcast_ref`]. Clone is one atomic Arc increment — the
/// zero-copy fan-out the drainer depends on.
#[derive(Clone)]
pub struct Carry(Arc<dyn Any + Send + Sync>);

impl Carry {
    /// Wrap a value, allocating one `Arc`. Producer side.
    #[must_use]
    pub fn new<T>(value: T) -> Self
    where
        T: Any + Send + Sync + 'static,
    {
        Self(Arc::new(value))
    }

    /// Wrap an already-shared `Arc`. Zero allocation beyond the
    /// `Arc<T> -> Arc<dyn Any>` coercion (a move, not a clone) — used
    /// when the caller already shares the Arc with other consumers
    /// (the drainer emitting one batch into multiple Pipe chains).
    #[must_use]
    pub fn from_arc<T>(value: Arc<T>) -> Self
    where
        T: Any + Send + Sync + 'static,
    {
        Self(value)
    }

    /// Borrow the payload if it is of type `T`. Cost is one `TypeId`
    /// equality check; no clone, no refcount bump.
    #[must_use]
    pub fn downcast_ref<T>(&self) -> Option<&T>
    where
        T: Any + Send + Sync + 'static,
    {
        self.0.downcast_ref::<T>()
    }
}

impl core::fmt::Debug for Carry {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.debug_struct("Carry").finish_non_exhaustive()
    }
}

/// A streamed response body: chunks plus optional trailers. The
/// explicit streaming contract that replaced `Body::Stream`. Ride it on
/// `Response.stream`; the buffered case leaves that field `None`.
pub struct ResponseStream {
    stream: ChunkStream,
    /// Trailers published at stream end. std-only.
    #[cfg(feature = "std")]
    trailers: Option<TrailersSlot>,
}

impl ResponseStream {
    /// Build from any `Send` chunk stream.
    #[must_use]
    pub fn new<S>(stream: S) -> Self
    where
        S: Stream<Item = Result<Bytes, ProximaError>> + Send + 'static,
    {
        Self {
            stream: Box::pin(stream),
            #[cfg(feature = "std")]
            trailers: None,
        }
    }

    /// Build from an already-boxed [`ChunkStream`].
    #[must_use]
    pub fn from_chunk_stream(stream: ChunkStream) -> Self {
        Self {
            stream,
            #[cfg(feature = "std")]
            trailers: None,
        }
    }

    /// One-chunk stream wrapping a complete `Bytes` — the buffered ->
    /// streaming bridge.
    #[must_use]
    pub fn once(bytes: Bytes) -> Self {
        Self::new(futures::stream::once(async move { Ok(bytes) }))
    }

    /// Attach a trailers slot a chunked producer will populate at end.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn with_trailers_slot(mut self, slot: TrailersSlot) -> Self {
        self.trailers = Some(slot);
        self
    }

    /// Attach trailers known up front.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn with_trailers(mut self, trailers: HeaderList) -> Self {
        self.trailers = Some(Arc::new(Mutex::new(Some(trailers))));
        self
    }

    /// Read captured trailers, if any. `None` until a stream-end
    /// producer has populated the slot.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn trailers(&self) -> Option<HeaderList> {
        self.trailers.as_ref()?.lock().ok()?.clone()
    }

    /// Borrow the trailers slot — handed to a producer (e.g. the H1
    /// chunked-decoder pump) before the stream completes.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn trailers_slot(&self) -> Option<&TrailersSlot> {
        self.trailers.as_ref()
    }

    /// Take the underlying chunk stream (drops trailers metadata).
    #[must_use]
    pub fn into_chunk_stream(self) -> ChunkStream {
        self.stream
    }

    /// Borrow the chunk stream mutably (listener pump path).
    #[must_use]
    pub fn chunk_stream_mut(&mut self) -> &mut ChunkStream {
        &mut self.stream
    }

    /// Consume the body stream without materializing any `Bytes` — read and
    /// discard all chunks until the stream ends. The framing decoder still
    /// runs (the caller must exhaust the body to reach the keep-alive
    /// message boundary); only the per-chunk allocation and copy are skipped.
    pub async fn drain(self) -> Result<(), ProximaError> {
        let mut stream = self.stream;
        while let Some(item) = stream.next().await {
            item?;
        }
        Ok(())
    }

    /// Drain the stream to a single `Bytes`. Under std, races the
    /// supplied request-scoped cancel token; the first to resolve wins.
    pub async fn collect(
        self,
        #[cfg(feature = "std")] cancel: Option<&Signal>,
    ) -> Result<Bytes, ProximaError> {
        let mut stream = self.stream;
        let mut chunks: Vec<Bytes> = Vec::new();
        let mut total = 0usize;
        loop {
            #[cfg(feature = "std")]
            let next = match cancel {
                // cancel polls first (select is left-biased); the sticky
                // level makes a fire that raced registration visible here
                Some(token) => match futures::future::select(token.fired(), stream.next()).await {
                    futures::future::Either::Left(((), _)) => {
                        return Err(ProximaError::Body("body cancelled".into()));
                    }
                    futures::future::Either::Right((item, _)) => item,
                },
                None => stream.next().await,
            };
            #[cfg(not(feature = "std"))]
            let next = stream.next().await;
            match next {
                Some(item) => {
                    let chunk = item?;
                    total += chunk.len();
                    chunks.push(chunk);
                }
                None => break,
            }
        }
        if chunks.len() == 1 {
            return Ok(chunks.remove(0));
        }
        let mut buffer = bytes::BytesMut::with_capacity(total);
        for chunk in chunks {
            buffer.extend_from_slice(&chunk);
        }
        Ok(buffer.freeze())
    }
}

impl core::fmt::Debug for ResponseStream {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ResponseStream")
            .finish_non_exhaustive()
    }
}

/// A streamed request body (uploads, WebSocket-inbound relay). Symmetric
/// to [`ResponseStream`]: carries an optional trailers slot a chunked
/// decoder populates at body-end (HTTP/1.1 request trailers, RFC 7230
/// §4.1.2). [`Request::body_bytes`] folds those into `Request.headers`
/// after draining — "request trailers fold into headers at chunked-decode
/// end".
pub struct RequestStream {
    stream: ChunkStream,
    /// Trailers published at stream end. std-only.
    #[cfg(feature = "std")]
    trailers: Option<TrailersSlot>,
}

impl RequestStream {
    #[must_use]
    pub fn new<S>(stream: S) -> Self
    where
        S: Stream<Item = Result<Bytes, ProximaError>> + Send + 'static,
    {
        Self {
            stream: Box::pin(stream),
            #[cfg(feature = "std")]
            trailers: None,
        }
    }

    #[must_use]
    pub fn from_chunk_stream(stream: ChunkStream) -> Self {
        Self {
            stream,
            #[cfg(feature = "std")]
            trailers: None,
        }
    }

    /// Attach a trailers slot a chunked decoder populates at body-end.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn with_trailers_slot(mut self, slot: TrailersSlot) -> Self {
        self.trailers = Some(slot);
        self
    }

    /// Read captured trailers, if any. `None` until a stream-end producer
    /// populates the slot.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn trailers(&self) -> Option<HeaderList> {
        self.trailers.as_ref()?.lock().ok()?.clone()
    }

    /// Borrow the trailers slot — handed to a producer (the chunked
    /// decoder pump) before the stream completes.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn trailers_slot(&self) -> Option<&TrailersSlot> {
        self.trailers.as_ref()
    }

    #[must_use]
    pub fn into_chunk_stream(self) -> ChunkStream {
        self.stream
    }

    #[must_use]
    pub fn chunk_stream_mut(&mut self) -> &mut ChunkStream {
        &mut self.stream
    }

    /// Drain to a single `Bytes`.
    pub async fn collect(self) -> Result<Bytes, ProximaError> {
        let mut stream = self.stream;
        let mut chunks: Vec<Bytes> = Vec::new();
        let mut total = 0usize;
        while let Some(item) = stream.next().await {
            let chunk = item?;
            total += chunk.len();
            chunks.push(chunk);
        }
        if chunks.len() == 1 {
            return Ok(chunks.remove(0));
        }
        let mut buffer = bytes::BytesMut::with_capacity(total);
        for chunk in chunks {
            buffer.extend_from_slice(&chunk);
        }
        Ok(buffer.freeze())
    }
}

impl core::fmt::Debug for RequestStream {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("RequestStream")
            .finish_non_exhaustive()
    }
}

// `#[proxima::test]`, `rstest`, and inline `tokio::spawn`/`tokio::time` pull
// in the `proxima` / `tokio` dev-dependencies, which the loom build keeps
// out of the graph (see `[target.'cfg(not(loom))'.dev-dependencies]` in
// Cargo.toml); these tests are unrelated to the Notify/watch loom protocol.
#[cfg(all(test, not(loom)))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[derive(Debug, PartialEq, Eq)]
    struct DummyRecord {
        name: alloc::string::String,
        count: u32,
    }

    #[test]
    fn carry_new_then_downcast_returns_value() {
        let carry = Carry::new(DummyRecord {
            name: "carried".into(),
            count: 3,
        });
        let view = carry
            .downcast_ref::<DummyRecord>()
            .expect("downcast succeeds");
        assert_eq!(view.name, "carried");
        assert_eq!(view.count, 3);
    }

    #[test]
    fn carry_wrong_type_downcast_returns_none() {
        let carry = Carry::new(DummyRecord {
            name: "x".into(),
            count: 1,
        });
        assert!(carry.downcast_ref::<u64>().is_none());
    }

    #[test]
    fn carry_from_arc_shares_one_allocation() {
        let shared = Arc::new(DummyRecord {
            name: "shared".into(),
            count: 9,
        });
        let strong_before = Arc::strong_count(&shared);
        let carry = Carry::from_arc(Arc::clone(&shared));
        assert_eq!(Arc::strong_count(&shared), strong_before + 1);
        assert_eq!(
            carry
                .downcast_ref::<DummyRecord>()
                .expect("downcast succeeds")
                .count,
            9
        );
    }

    #[test]
    fn carry_clone_is_one_arc_bump_not_a_record_copy() {
        // the fan-out contract: cloning a Carry (Tee/Diff branching) is a
        // single atomic increment of the SAME allocation, never a per-record
        // duplication. Proven by strong_count tracking one shared backing Arc.
        let shared = Arc::new(DummyRecord {
            name: "fanout".into(),
            count: 5,
        });
        let strong_before = Arc::strong_count(&shared);
        let carry = Carry::from_arc(Arc::clone(&shared));
        assert_eq!(Arc::strong_count(&shared), strong_before + 1);
        let branch_a = carry.clone();
        let branch_b = carry.clone();
        assert_eq!(Arc::strong_count(&shared), strong_before + 3);
        assert_eq!(
            branch_a.downcast_ref::<DummyRecord>().expect("a").count,
            branch_b.downcast_ref::<DummyRecord>().expect("b").count
        );
    }

    #[proxima::test]
    async fn response_stream_once_collects_to_original_bytes() {
        let stream = ResponseStream::once(Bytes::from_static(b"hello"));
        let collected = stream
            .collect(
                #[cfg(feature = "std")]
                None,
            )
            .await
            .expect("collect succeeds");
        assert_eq!(&collected[..], b"hello");
    }

    #[proxima::test]
    async fn response_stream_multi_chunk_concatenates_in_order() {
        let stream = ResponseStream::new(futures::stream::iter([
            Ok(Bytes::from_static(b"hel")),
            Ok(Bytes::from_static(b"lo")),
        ]));
        let collected = stream
            .collect(
                #[cfg(feature = "std")]
                None,
            )
            .await
            .expect("collect succeeds");
        assert_eq!(&collected[..], b"hello");
    }

    #[proxima::test]
    async fn response_stream_propagates_inner_error() {
        let stream = ResponseStream::new(futures::stream::iter([Err::<Bytes, _>(
            ProximaError::Body("upstream broken".into()),
        )]));
        let outcome = stream
            .collect(
                #[cfg(feature = "std")]
                None,
            )
            .await;
        assert!(matches!(outcome, Err(ProximaError::Body(_))));
    }

    #[cfg(feature = "std")]
    #[proxima::test]
    async fn response_stream_collect_aborts_on_cancel() {
        use proxima_core::signal::Signal;
        let cancel = Signal::new();
        let pending = futures::stream::poll_fn(|_cx| {
            core::task::Poll::<Option<Result<Bytes, ProximaError>>>::Pending
        });
        let stream = ResponseStream::new(pending);
        let cancel_for_task = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            cancel_for_task.fire();
        });
        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            stream.collect(Some(&cancel)),
        )
        .await
        .expect("collect returns before outer timeout");
        assert!(matches!(outcome, Err(ProximaError::Body(ref msg)) if msg.contains("cancelled")));
    }

    #[cfg(feature = "std")]
    #[proxima::test]
    async fn response_stream_trailers_publish_after_construction() {
        let slot: TrailersSlot = Arc::new(Mutex::new(None));
        let stream =
            ResponseStream::once(Bytes::from_static(b"x")).with_trailers_slot(slot.clone());
        assert!(stream.trailers().is_none());
        let mut headers = HeaderList::new();
        headers.insert(Bytes::from_static(b"X-Done"), Bytes::from_static(b"true"));
        *slot.lock().expect("lock") = Some(headers);
        assert_eq!(
            stream
                .trailers()
                .expect("trailers present")
                .get_str("x-done"),
            Some("true")
        );
    }

    #[rstest]
    #[case::ascii("hello")]
    #[case::unicode("héllo")]
    #[proxima::test]
    async fn request_stream_roundtrips_via_collect(#[case] input: &'static str) {
        let stream = RequestStream::new(futures::stream::once({
            let bytes = Bytes::copy_from_slice(input.as_bytes());
            async move { Ok(bytes) }
        }));
        let collected = stream.collect().await.expect("collect succeeds");
        assert_eq!(&collected[..], input.as_bytes());
    }
}
