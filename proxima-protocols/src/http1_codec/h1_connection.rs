//! HTTP/1.1 connection state machine — zero-copy, alloc-free hot path.
//!
//! Owns one growing read buffer; head + headers + body all borrow
//! into that buffer. Across requests, the buffer is reused via a
//! cursor (pipelined-request bytes don't memcpy). Per-request
//! state (header offset table, response output) is pre-allocated on
//! the Connection and `clear()`'d each cycle — the steady-state
//! request path makes zero allocations.
//!
//! Pattern from the listener:
//!
//! ```ignore
//! let mut conn = Connection::new();
//! let mut out = Vec::with_capacity(8 * 1024);
//! loop {
//!     // single kernel→userspace copy:
//!     conn.feed_bytes(&socket_chunk);
//!     match conn.poll()? {
//!         Poll::NeedInput => continue,
//!         Poll::Close => break,
//!         Poll::RequestReady => {
//!             let head = conn.head().expect("head present");
//!             let body = conn.body();
//!             // ... dispatch via Pipe::call ...
//!             out.clear();
//!             let writer = conn.begin_response(200, "OK", &resp_headers, framing, &mut out);
//!             socket.write_all(&out).await?;
//!             out.clear();
//!             writer.write_chunk(&response_body, &mut out);
//!             socket.write_all(&out).await?;
//!             out.clear();
//!             writer.end_response(&mut out);
//!             socket.write_all(&out).await?;
//!             if conn.keep_alive() {
//!                 conn.reset_for_next_request();
//!             } else {
//!                 break;
//!             }
//!         }
//!     }
//! }
//! ```

use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;

use bumpalo::Bump;
use bytes::Bytes;

use crate::http1_codec::h1::{
    Header, HeaderVec, HttpVersion, ParseError, ParserLimits, RequestHead, StreamingStatus,
    parse_head_streaming,
};
use crate::http1_codec::h1_body::{BodyDecoder, BodyFraming, DecodeError, DecoderLimits, Status as BodyStatus};
use crate::http1_codec::h1_response::{BodyEncoder, write_response_head};

const DEFAULT_BUFFER_BYTES: usize = 8 * 1024;
const DEFAULT_HEADERS_CAPACITY: usize = 32;

/// Auto-stream policy. When attached via
/// `Connection::set_auto_stream_policy`, the connection inspects each
/// request's framing at head-parse time and self-enables streaming
/// mode when the body matches the predicate. Lets the listener stay
/// out of the head-peek business.
#[derive(Debug, Clone, Copy)]
pub struct AutoStreamPolicy {
    /// Auto-stream when `Content-Length` exceeds this byte count.
    /// Smaller bodies stay on the buffered (single-`RequestReady`)
    /// path to avoid the mpsc + spawn cost.
    pub content_length_threshold: u64,
    /// Auto-stream `Transfer-Encoding: chunked` bodies regardless of
    /// declared size.
    pub stream_chunked: bool,
}

impl Default for AutoStreamPolicy {
    fn default() -> Self {
        // 1 MiB threshold + always stream chunked. Mirrors the
        // recommendation in docs/streaming_body_plan.md.
        Self {
            content_length_threshold: 1024 * 1024,
            stream_chunked: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadError {
    Parse(ParseError),
    Decode(DecodeError),
    AmbiguousFraming,
    BadContentLength,
    UnsupportedTransferEncoding,
}

impl From<ParseError> for ReadError {
    fn from(error: ParseError) -> Self {
        Self::Parse(error)
    }
}

impl From<DecodeError> for ReadError {
    fn from(error: DecodeError) -> Self {
        Self::Decode(error)
    }
}

/// Poll result.
///
/// In default (buffered) mode the connection emits the classic
/// `NeedInput` → `RequestReady` → `Close` sequence: the full body is
/// staged in the read buffer before the caller sees the request.
///
/// In streaming mode (`set_streaming(true)`) the head is surfaced
/// independently from the body, and body bytes are exposed
/// chunk-by-chunk via `take_body_chunk()`. The sequence becomes
/// `NeedInput`* → `HeadReady` → (`NeedInput` | `BodyChunk`)* →
/// `BodyEnd`. The listener pumps chunks into a bounded channel
/// feeding the Pipe; the bounded channel is the backpressure
/// signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Poll {
    NeedInput,
    /// Streaming mode only: head parsed, decoder primed, no body
    /// chunks consumed yet. Caller dispatches the request with a
    /// streaming body handle and then continues polling for chunks.
    HeadReady,
    /// Streaming mode only: a body chunk is queued — call
    /// `take_body_chunk()` to consume.
    BodyChunk,
    /// Streaming mode only: body decoder reached End; no more chunks.
    BodyEnd,
    /// Buffered mode: head + full body are present and ready for
    /// dispatch. Never emitted in streaming mode.
    RequestReady,
    /// Client sent `Expect: 100-continue`. The listener must either:
    /// - call `accept_continue(&mut out)` to emit
    ///   `HTTP/1.1 100 Continue\r\n\r\n` and resume normal polling
    ///   (head accessible via `head()` / `header_value()` for an
    ///   informed decision), or
    /// - call `begin_response(...)` to write a final response
    ///   (e.g. 413 Payload Too Large, 417 Expectation Failed) and
    ///   close.
    ///
    /// Emitted exactly once per request, before HeadReady (streaming
    /// mode) or RequestReady (buffered mode). The head is already
    /// parsed when this fires — the listener can inspect headers to
    /// make the decision.
    Expect100Continue,
    Close,
}

#[derive(Debug, PartialEq, Eq)]
enum State {
    ReadingHead,
    ReadingBody,
    AwaitingResponse,
    AfterResponse { keep_alive: bool },
    Closed,
}

/// Cached (start, end) offsets into `Connection::buffer` for each
/// parsed piece. Lets `head()` reconstruct a `RequestHead<'_>` by
/// slicing without re-parsing.
#[derive(Debug, Clone, Copy, Default)]
struct HeadOffsets {
    method: (usize, usize),
    path: (usize, usize),
    version: HttpVersion,
    body_start: usize,
}

impl HeadOffsets {
    fn empty() -> Self {
        Self {
            method: (0, 0),
            path: (0, 0),
            version: HttpVersion::Http11,
            body_start: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct HeaderOffsets {
    name: (usize, usize),
    value: (usize, usize),
}

pub struct Connection {
    state: State,
    /// The one growing read buffer. Single kernel→userspace copy
    /// lands here on `feed_bytes`. Head + body slices borrow into
    /// this buffer.
    buffer: Vec<u8>,
    /// Logical start of the current request. Advances across
    /// requests on `reset_for_next_request` so pipelined bytes
    /// don't memcpy.
    request_start: usize,
    /// Cached head offsets after `parse_head` succeeded. Eliminates
    /// the re-parse on `head()` access.
    head: HeadOffsets,
    /// Reused header offset table; `clear()`'d per request.
    headers: Vec<HeaderOffsets>,
    /// Body decoder for the current request.
    decoder: Option<BodyDecoder>,
    /// `(start, end)` of body bytes inside `buffer`.
    body_span: (usize, usize),
    /// Keep-alive decision derived from head; cached for write_response.
    request_keep_alive: bool,
    /// Response framing for the in-flight response.
    response_encoder: Option<BodyEncoder>,
    /// Streaming-body mode flag. When true, `poll` emits
    /// `HeadReady` after head parse and chunks via `BodyChunk` /
    /// `BodyEnd` instead of buffering the whole body and returning
    /// `RequestReady`.
    streaming_mode: bool,
    /// Set once `HeadReady` has been surfaced for the current
    /// request, so a subsequent `poll` doesn't re-emit it. Reset by
    /// `reset_for_next_request`.
    head_emitted: bool,
    /// Set once the body decoder has signaled End. `poll` drains the
    /// chunks queue first and then emits `BodyEnd`; this flag
    /// disambiguates "no more chunks, more bytes might arrive" from
    /// "no more chunks, decoder is done".
    decoder_ended: bool,
    /// Set once `BodyEnd` has been surfaced for the current request,
    /// to keep `poll` idempotent after the listener has moved on to
    /// response writing.
    body_end_emitted: bool,
    /// Queue of decoded body chunks awaiting `take_body_chunk()` by
    /// the listener. Each chunk is an owned `Bytes` (copy of the
    /// decoder's borrow into the read buffer) so it can travel
    /// across the listener → Pipe body channel without being
    /// invalidated by buffer compaction.
    body_chunks: VecDeque<Bytes>,
    /// Set during head parse when the request carried
    /// `Expect: 100-continue`. While true, `poll` emits
    /// `Expect100Continue` before any other body-phase poll variant
    /// so the listener can decide to accept (write 100, proceed) or
    /// reject (write final response, close). Cleared by
    /// `accept_continue` or `begin_response`.
    expect_continue_pending: bool,
    /// Per-connection auto-stream policy. When set, the head-parse
    /// arm of `poll` evaluates the policy against the request's
    /// framing and flips `streaming_mode` to true automatically.
    auto_stream_policy: Option<AutoStreamPolicy>,
    parser_limits: ParserLimits,
    decoder_limits: DecoderLimits,
    /// Per-request bump arena. The listener uses this to allocate
    /// the typed `proxima::Request` fields (method/path Bytes,
    /// header pairs, response header vec) without hitting the
    /// global allocator on the hot path. `reset_for_next_request`
    /// resets the arena — every per-request allocation is freed in
    /// O(1) by rolling back the bump cursor.
    arena: Bump,
}

impl Default for Connection {
    fn default() -> Self {
        Self::new()
    }
}

impl Connection {
    #[must_use]
    pub fn new() -> Self {
        Self::with_limits(ParserLimits::default(), DecoderLimits::default())
    }

    #[must_use]
    pub fn with_limits(parser_limits: ParserLimits, decoder_limits: DecoderLimits) -> Self {
        Self {
            state: State::ReadingHead,
            buffer: Vec::with_capacity(DEFAULT_BUFFER_BYTES),
            request_start: 0,
            head: HeadOffsets::empty(),
            headers: Vec::with_capacity(DEFAULT_HEADERS_CAPACITY),
            decoder: None,
            body_span: (0, 0),
            request_keep_alive: false,
            response_encoder: None,
            streaming_mode: false,
            head_emitted: false,
            decoder_ended: false,
            body_end_emitted: false,
            body_chunks: VecDeque::new(),
            expect_continue_pending: false,
            auto_stream_policy: None,
            parser_limits,
            decoder_limits,
            // 8 KB initial bump region — covers a typical request's
            // Request fields + response headers without growing.
            arena: Bump::with_capacity(8 * 1024),
        }
    }

    /// Borrow the per-request bump arena. Listener / Pipe-adapter
    /// code allocates per-request scratch here (header tables,
    /// owned-bytes copies that don't outlive the request, etc.)
    /// instead of hitting the global allocator.
    ///
    /// The arena is reset on `reset_for_next_request`; any references
    /// held past that point dangle. Callers must drop all arena
    /// references before reset.
    pub fn request_arena(&self) -> &Bump {
        &self.arena
    }

    pub fn buffered_bytes(&self) -> usize {
        self.buffer.len() - self.request_start
    }

    pub fn feed_bytes(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    /// Enable or disable streaming-body mode. Off by default — the
    /// caller opts in once per connection (typically after seeing
    /// `Transfer-Encoding: chunked` or a large `Content-Length` in
    /// the head). In streaming mode `poll` yields `HeadReady` /
    /// `BodyChunk` / `BodyEnd` instead of waiting for the full body
    /// before returning `RequestReady`.
    ///
    /// Safe to call at any time; the effect is read on each `poll`.
    /// In practice the listener flips it on between seeing the head
    /// (via a tentative head parse / peek at headers) and the first
    /// poll that surfaces it.
    pub fn set_streaming(&mut self, streaming: bool) {
        self.streaming_mode = streaming;
    }

    /// Attach (or clear) the auto-stream policy. Once set, each
    /// request's framing is inspected at head-parse time and
    /// `streaming_mode` flips to true automatically when the policy
    /// matches. Per-request decisions happen inside the connection;
    /// callers just observe `Poll::HeadReady` vs `Poll::RequestReady`.
    pub fn set_auto_stream_policy(&mut self, policy: Option<AutoStreamPolicy>) {
        self.auto_stream_policy = policy;
    }

    #[must_use]
    pub fn is_streaming(&self) -> bool {
        self.streaming_mode
    }

    /// Pop the next pending body chunk if any. Returns `None` if the
    /// queue is empty — the caller should `poll` again to drive more
    /// bytes through the decoder.
    pub fn take_body_chunk(&mut self) -> Option<Bytes> {
        self.body_chunks.pop_front()
    }

    /// True between head parse and `accept_continue` / `begin_response`
    /// when the request carried `Expect: 100-continue`. The listener
    /// inspects this (or `Poll::Expect100Continue`) to decide whether
    /// to write 100 and accept the body or short-circuit with a
    /// final response.
    #[must_use]
    pub fn expects_continue(&self) -> bool {
        self.expect_continue_pending
    }

    /// Drain captured request trailers (RFC 7230 §4.1.2). Valid to
    /// call after body decode completes (RequestReady in buffered
    /// mode, BodyEnd in streaming mode). Returns an empty Vec for
    /// non-chunked bodies and for chunked bodies that didn't carry
    /// trailers — the common case is zero allocations.
    pub fn take_trailers(&mut self) -> Vec<(Bytes, Bytes)> {
        match &mut self.decoder {
            Some(decoder) => decoder.take_trailers(),
            None => Vec::new(),
        }
    }

    /// Emit `HTTP/1.1 100 Continue\r\n\r\n` into `out` and clear the
    /// pending Expect flag so subsequent polls advance to the body
    /// phase normally. Listener composes this into a single write
    /// before resuming body reads. No-op when no Expect is pending.
    pub fn accept_continue(&mut self, out: &mut Vec<u8>) {
        if !self.expect_continue_pending {
            return;
        }
        // The status line is identical for both HTTP/1.0 and 1.1
        // requests at this point; clients that use Expect MUST be
        // HTTP/1.1 (RFC 7231 §5.1.1) so the 1.1 status line is the
        // right reply.
        out.extend_from_slice(b"HTTP/1.1 100 Continue\r\n\r\n");
        self.expect_continue_pending = false;
    }

    pub fn poll(&mut self) -> Result<Poll, ReadError> {
        loop {
            match self.state {
                State::Closed | State::AfterResponse { .. } => return Ok(Poll::Close),
                State::AwaitingResponse => {
                    // Streaming callers already consumed the body via
                    // BodyChunk / BodyEnd — surface BodyEnd once if it
                    // hasn't been seen yet (the decoder may have
                    // finished mid-buffered-decode and flipped state).
                    if self.streaming_mode && !self.body_end_emitted {
                        self.body_end_emitted = true;
                        return Ok(Poll::BodyEnd);
                    }
                    return Ok(Poll::RequestReady);
                }
                State::ReadingHead => {
                    // Hot-path zero-allocation parse: the streaming
                    // parser fires `on_header` per header; we push
                    // offsets straight into the pre-allocated
                    // self.headers buffer (cleared at request start).
                    self.headers.clear();
                    let buffer_base = self.buffer.as_ptr() as usize;
                    let head_buf = &self.buffer[self.request_start..];
                    let header_sink = &mut self.headers;
                    let parsed = parse_head_streaming(
                        head_buf,
                        self.parser_limits,
                        |header: Header<'_>| {
                            header_sink.push(HeaderOffsets {
                                name: offsets_into(buffer_base, header.name()),
                                value: offsets_into(buffer_base, header.value()),
                            });
                        },
                    )?;
                    let (method_off, path_off, version, consumed) = match parsed {
                        StreamingStatus::Partial => return Ok(Poll::NeedInput),
                        StreamingStatus::Complete {
                            method,
                            path,
                            version,
                            consumed,
                        } => (
                            offsets_into(buffer_base, method),
                            offsets_into(buffer_base, path),
                            version,
                            consumed,
                        ),
                    };
                    // Derive framing + keep-alive from the freshly
                    // populated self.headers (offset-based, no
                    // header struct allocation).
                    let framing = body_framing_from_offsets(&self.buffer, &self.headers)?;
                    let keep_alive = keep_alive_from_offsets(&self.buffer, &self.headers, version);
                    let expects_continue =
                        expects_100_continue_from_offsets(&self.buffer, &self.headers);
                    self.head.method = method_off;
                    self.head.path = path_off;
                    self.head.version = version;
                    self.head.body_start = self.request_start + consumed;
                    self.request_keep_alive = keep_alive;
                    self.expect_continue_pending = expects_continue;
                    if let Some(policy) = self.auto_stream_policy
                        && policy_matches(&policy, framing)
                    {
                        self.streaming_mode = true;
                    }
                    self.decoder = Some(BodyDecoder::with_limits(framing, self.decoder_limits));
                    let body_start_abs = self.request_start + consumed;
                    self.body_span = (body_start_abs, body_start_abs);
                    self.state = State::ReadingBody;
                    // Expect100Continue gates everything else — listener
                    // resolves it (accept_continue or begin_response)
                    // before HeadReady / body decode can proceed.
                    if self.expect_continue_pending {
                        return Ok(Poll::Expect100Continue);
                    }
                    if self.streaming_mode && !self.head_emitted {
                        self.head_emitted = true;
                        return Ok(Poll::HeadReady);
                    }
                }
                State::ReadingBody => {
                    // Re-emit Expect100Continue until resolved, even if
                    // the head-parse arm already left it (e.g., listener
                    // looped back to poll without calling accept/begin).
                    if self.expect_continue_pending {
                        return Ok(Poll::Expect100Continue);
                    }
                    // Streaming mode: HeadReady might still be owed
                    // — e.g., we returned Expect100Continue from the
                    // ReadingHead arm before emitting HeadReady, and
                    // the listener accept_continue'd and looped back
                    // to poll. Emit HeadReady here as the deferred
                    // streaming-mode head signal.
                    if self.streaming_mode && !self.head_emitted {
                        self.head_emitted = true;
                        return Ok(Poll::HeadReady);
                    }
                    if self.streaming_mode {
                        // 1. Drain queued chunks one per poll — the
                        //    listener calls take_body_chunk between
                        //    polls and pumps each into the bounded
                        //    body channel (backpressure point).
                        if !self.body_chunks.is_empty() {
                            return Ok(Poll::BodyChunk);
                        }
                        // 2. Decoder already finished + queue is
                        //    empty: advance state, surface BodyEnd
                        //    exactly once.
                        if self.decoder_ended {
                            self.state = State::AwaitingResponse;
                            self.body_end_emitted = true;
                            return Ok(Poll::BodyEnd);
                        }
                    }
                    let Some(decoder) = self.decoder.as_mut() else {
                        return Ok(Poll::NeedInput);
                    };
                    let body_buf = &self.buffer[self.body_span.1..];
                    let body_base_ptr = self.buffer.as_ptr() as usize;
                    let mut chunk_end_abs = self.body_span.1;
                    let streaming = self.streaming_mode;
                    let chunk_sink = &mut self.body_chunks;
                    let (consumed, status) = decoder.feed(body_buf, |chunk| {
                        let chunk_offset = chunk.as_ptr() as usize - body_base_ptr;
                        let chunk_end_in_buffer = chunk_offset + chunk.len();
                        if chunk_end_in_buffer > chunk_end_abs {
                            chunk_end_abs = chunk_end_in_buffer;
                        }
                        if streaming && !chunk.is_empty() {
                            chunk_sink.push_back(Bytes::copy_from_slice(chunk));
                        }
                    })?;
                    self.body_span.1 += consumed;
                    // chunk_end_abs only matters for chunked encoding;
                    // for Content-Length the chunk equals the whole
                    // body region. Use the further-along of the two
                    // so `body()` returns the right span either way.
                    if chunk_end_abs > self.body_span.1 {
                        self.body_span.1 = chunk_end_abs;
                    }
                    let ended = matches!(status, BodyStatus::End);
                    if self.streaming_mode {
                        if ended {
                            self.decoder_ended = true;
                        }
                        if !self.body_chunks.is_empty() {
                            return Ok(Poll::BodyChunk);
                        }
                        if self.decoder_ended {
                            self.state = State::AwaitingResponse;
                            self.body_end_emitted = true;
                            return Ok(Poll::BodyEnd);
                        }
                        return Ok(Poll::NeedInput);
                    }
                    if ended {
                        self.state = State::AwaitingResponse;
                    } else {
                        return Ok(Poll::NeedInput);
                    }
                }
            }
        }
    }

    /// Borrow the method bytes. Empty slice if the head hasn't been
    /// parsed yet. Cheap — direct buffer slice, no allocation.
    pub fn method(&self) -> &[u8] {
        if !self.head_parsed() {
            return &[];
        }
        &self.buffer[self.head.method.0..self.head.method.1]
    }

    /// Borrow the request target.
    pub fn path(&self) -> &[u8] {
        if !self.head_parsed() {
            return &[];
        }
        &self.buffer[self.head.path.0..self.head.path.1]
    }

    fn head_parsed(&self) -> bool {
        matches!(self.state, State::ReadingBody | State::AwaitingResponse)
    }

    pub fn version(&self) -> HttpVersion {
        self.head.version
    }

    /// Iterate over headers without allocating. Caller decides whether
    /// to short-circuit or collect.
    pub fn headers(&self) -> HeadersIter<'_> {
        HeadersIter {
            buffer: &self.buffer,
            offsets: &self.headers,
            cursor: 0,
        }
    }

    /// Find a header value by name (case-insensitive). Linear scan —
    /// for typical header counts beats any indexed structure.
    pub fn header_value(&self, name: &[u8]) -> Option<&[u8]> {
        if !self.head_parsed() {
            return None;
        }
        for offsets in &self.headers {
            let header_name = &self.buffer[offsets.name.0..offsets.name.1];
            if eq_ignore_ascii_case(header_name, name) {
                return Some(&self.buffer[offsets.value.0..offsets.value.1]);
            }
        }
        None
    }

    /// Reconstruct a full `RequestHead<'_>` for callers that want the
    /// typed handle. Allocates only past `sized::HEADER_INLINE_CAP`
    /// headers — NOT on the hot path (`method()` / `path()` /
    /// `headers()` are).
    pub fn head(&self) -> Option<RequestHead<'_>> {
        if !self.head_parsed() {
            return None;
        }
        let headers: HeaderVec<'_> = self.headers().collect();
        Some(RequestHead {
            method: self.method(),
            path: self.path(),
            version: self.head.version,
            headers,
        })
    }

    /// Borrow the decoded body region from the buffer.
    pub fn body(&self) -> &[u8] {
        &self.buffer[self.body_span.0..self.body_span.1]
    }

    #[must_use]
    pub fn keep_alive(&self) -> bool {
        self.request_keep_alive
    }

    /// Start writing the response: emit status line + headers + blank
    /// line into `out`, and return a `ResponseWriter` typestate that
    /// borrows the connection mutably. Subsequent body chunks and the
    /// terminator are written via that handle.
    ///
    /// Typestate guarantee: the borrow ensures the caller cannot
    /// touch the connection (poll, accessors, reset, begin_response
    /// again) while the response is in flight. `write_chunk` exists
    /// only on `ResponseWriter`; `end_response` consumes the
    /// writer so it cannot be called twice. After the writer is
    /// consumed (or dropped), the connection's state has advanced
    /// to `AfterResponse` and the caller can call `keep_alive` /
    /// `reset_for_next_request` / `close`.
    #[must_use = "ResponseWriter must be advanced via end_response or the response is incomplete"]
    pub fn begin_response<'a>(
        &'a mut self,
        status: u16,
        reason: &str,
        headers: &[(String, String)],
        framing: BodyFraming,
        out: &mut Vec<u8>,
    ) -> ResponseWriter<'a> {
        // Writing a final response overrides any pending Expect:
        // 100-continue handshake — the client SHOULD stop sending
        // body bytes upon seeing the final status (RFC 7231).
        self.expect_continue_pending = false;
        write_response_head(out, self.head.version, status, reason, headers, framing);
        self.response_encoder = Some(BodyEncoder::new(framing));
        ResponseWriter {
            connection: self,
            ended: false,
        }
    }

    /// Prepare for the next request on a persistent connection.
    /// Advances the cursor past the current request's body; the next
    /// `poll` parses head from the remaining bytes in-place (no copy).
    /// Resets the per-request bump arena in O(1) — every allocation
    /// the listener / pipe adapter made into it is reclaimed.
    pub fn reset_for_next_request(&mut self) {
        self.request_start = self.body_span.1;
        // If we've consumed the whole buffer, collapse the cursor so
        // the buffer doesn't grow unbounded across keep-alive cycles.
        if self.request_start >= self.buffer.len() {
            self.buffer.clear();
            self.request_start = 0;
        } else if self.request_start > DEFAULT_BUFFER_BYTES {
            // After enough pipelined requests, compact: drain the
            // consumed prefix so the buffer stays bounded.
            self.buffer.drain(..self.request_start);
            self.request_start = 0;
        }
        self.head = HeadOffsets::empty();
        self.headers.clear();
        self.decoder = None;
        self.body_span = (0, 0);
        self.response_encoder = None;
        self.head_emitted = false;
        self.decoder_ended = false;
        self.body_end_emitted = false;
        self.body_chunks.clear();
        self.expect_continue_pending = false;
        self.arena.reset();
        self.state = State::ReadingHead;
    }

    pub fn close(&mut self) {
        self.state = State::Closed;
    }

    /// Drive the state machine and return a typed outcome. Each
    /// non-trivial variant carries a typed handle that gates access
    /// to state-specific accessors at compile time — the listener
    /// cannot call `body()` on a `HeadReady` outcome (body isn't
    /// decoded yet) or `begin_response` twice on the same request
    /// (the handle is consumed).
    ///
    /// Equivalent in semantics to [`Connection::poll`] but typed.
    /// `poll` remains for callers that prefer the bare enum.
    pub fn advance(&mut self) -> Result<Advanced<'_>, ReadError> {
        match self.poll()? {
            Poll::NeedInput => Ok(Advanced::NeedInput),
            Poll::Close => Ok(Advanced::Close),
            Poll::Expect100Continue => {
                Ok(Advanced::Expect100Continue(ExpectGate { connection: self }))
            }
            Poll::HeadReady => Ok(Advanced::HeadReady(HeadReadyHandle { connection: self })),
            Poll::BodyChunk => Ok(Advanced::BodyChunk(BodyChunkHandle { connection: self })),
            Poll::BodyEnd => Ok(Advanced::BodyEnd(BodyEndHandle { connection: self })),
            Poll::RequestReady => Ok(Advanced::RequestReady(BufferedRequestHandle {
                connection: self,
            })),
        }
    }

    /// Drain any bytes buffered past the current request's body.
    /// Used by the listener when handing a socket to an upgrade
    /// handler — a client that eagerly sent tunnel / websocket data
    /// before seeing the server's 101/200 has those bytes here, and
    /// the handler must process them before reading the raw socket.
    /// For most clients (which wait for the server's response) the
    /// returned `Bytes` is empty.
    ///
    /// Truncates the connection buffer to keep `request_start`
    /// consistent; the Connection is at end-of-life after this call
    /// (the listener drops it and hands ownership to the upgrade
    /// handler) so no further `feed_bytes`/`poll` is expected.
    pub fn drain_pipelined_bytes(&mut self) -> Bytes {
        let start = self.body_span.1.max(self.head.body_start);
        if start >= self.buffer.len() {
            return Bytes::new();
        }
        let drained = Bytes::copy_from_slice(&self.buffer[start..]);
        self.buffer.truncate(start);
        drained
    }
}

/// Response-phase typestate handle. Returned by
/// `Connection::begin_response`; the connection is borrowed
/// mutably for the writer's lifetime. The compile-time invariants:
///
/// - You cannot construct a `ResponseWriter` without first calling
///   `begin_response`.
/// - You cannot call `begin_response` twice on the same connection
///   while a writer is alive (borrow check denies it).
/// - You cannot touch the connection (poll, head, body,
///   reset_for_next_request, close) while a writer is alive
///   (same borrow rule).
/// - `end_response` consumes the writer, so you cannot end twice.
/// - `#[must_use]` on `begin_response` flags forgotten ends at
///   compile time; the `Drop` impl below is the runtime fallback
///   that prevents an inconsistent connection state if Drop fires
///   without `end_response` being called.
pub struct ResponseWriter<'a> {
    connection: &'a mut Connection,
    ended: bool,
}

impl<'a> ResponseWriter<'a> {
    /// Frame a body chunk for the wire and append to `out`. For
    /// Content-Length framing the chunk goes through verbatim; for
    /// Chunked framing it gets the `hex(len)\r\n<data>\r\n` wrapper.
    /// Caller is responsible for clearing `out` between socket
    /// writes.
    pub fn write_chunk(&self, data: &[u8], out: &mut Vec<u8>) {
        if let Some(encoder) = self.connection.response_encoder.as_ref() {
            encoder.encode_chunk(data, out);
        }
    }

    /// Emit the body terminator (Chunked: `0\r\n\r\n`; ContentLength
    /// / None: no-op), advance the connection's state to
    /// `AfterResponse`, and consume the writer.
    pub fn end_response(mut self, out: &mut Vec<u8>) {
        if let Some(mut encoder) = self.connection.response_encoder.take() {
            encoder.encode_end(out);
        }
        self.connection.state = State::AfterResponse {
            keep_alive: self.connection.request_keep_alive,
        };
        self.ended = true;
    }

    /// Like `end_response`, but emits trailer headers between the
    /// final 0-length chunk and the terminating CRLF when the
    /// response is chunked-framed. Trailers are ignored for
    /// Content-Length / None framings (no wire slot). Consumes the
    /// writer.
    pub fn end_response_with_trailers(mut self, trailers: &[(&[u8], &[u8])], out: &mut Vec<u8>) {
        if let Some(mut encoder) = self.connection.response_encoder.take() {
            encoder.encode_end_with_trailers(trailers, out);
        }
        self.connection.state = State::AfterResponse {
            keep_alive: self.connection.request_keep_alive,
        };
        self.ended = true;
    }
}

impl Drop for ResponseWriter<'_> {
    fn drop(&mut self) {
        if self.ended {
            return;
        }
        // Forgot to call `end_response`. Defensive recovery: force
        // the state forward to AfterResponse with keep_alive=false
        // so the listener sees the connection as terminal and
        // closes it. The wire is in an incomplete state — the
        // client will get a truncated response — but at least the
        // server-side state machine stays consistent.
        self.connection.response_encoder = None;
        self.connection.state = State::AfterResponse { keep_alive: false };
    }
}

/// Compute the (start, end) byte offsets of `slice` within the buffer
/// whose base pointer is `buffer_base`. `slice` must borrow from that
/// buffer — callers from our parser guarantee this. We use pointer
/// arithmetic instead of position lookups so the cost is O(1).
fn offsets_into(buffer_base: usize, slice: &[u8]) -> (usize, usize) {
    let start = slice.as_ptr() as usize - buffer_base;
    (start, start + slice.len())
}

/// Typed read-side outcome from [`Connection::advance`]. Each
/// variant carries a typed handle that exposes only the accessors
/// valid in that state — the borrow checker prevents callers from
/// touching the connection (or calling `advance` again) while a
/// handle is alive, and method-by-state grouping gives compile-time
/// errors for misuse like calling `body()` before the body is
/// decoded or `begin_response()` twice on the same request.
pub enum Advanced<'a> {
    /// Decoder is hungry. Caller should read more bytes from the
    /// socket and call `feed_bytes` + `advance` again.
    NeedInput,
    /// State machine reached a terminal close (peer closed, server
    /// chose Connection: close, etc.). Drop the connection.
    Close,
    /// Client sent `Expect: 100-continue`. The listener must call
    /// `accept(...)` (emits 100 Continue and resumes polling) or
    /// `reject(...)` (begins a final response without reading body).
    Expect100Continue(ExpectGate<'a>),
    /// Streaming mode: head parsed, no body chunks consumed yet.
    /// Inspect head and either reject (early response) or call
    /// `advance` again to pull chunks.
    HeadReady(HeadReadyHandle<'a>),
    /// Streaming mode: a body chunk is queued. Call `take_chunk`
    /// to consume it.
    BodyChunk(BodyChunkHandle<'a>),
    /// Streaming mode: decoder reached End. Trailers (if any) are
    /// available; begin the response.
    BodyEnd(BodyEndHandle<'a>),
    /// Buffered mode: head + full body are ready. Inspect head /
    /// body / trailers and begin the response.
    RequestReady(BufferedRequestHandle<'a>),
}

/// Handle for the `Expect: 100-continue` decision point. The
/// listener inspects the head and either accepts (emits 100
/// Continue and lets polling resume) or rejects (begins a final
/// response — typically 413).
impl core::fmt::Debug for Advanced<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let variant = match self {
            Self::NeedInput => "NeedInput",
            Self::Close => "Close",
            Self::Expect100Continue(_) => "Expect100Continue",
            Self::HeadReady(_) => "HeadReady",
            Self::BodyChunk(_) => "BodyChunk",
            Self::BodyEnd(_) => "BodyEnd",
            Self::RequestReady(_) => "RequestReady",
        };
        formatter
            .debug_struct("Advanced")
            .field("variant", &variant)
            .finish_non_exhaustive()
    }
}

pub struct ExpectGate<'a> {
    connection: &'a mut Connection,
}

impl<'a> ExpectGate<'a> {
    /// HTTP method bytes — `POST`, `PUT`, etc.
    #[must_use]
    pub fn method(&self) -> &[u8] {
        self.connection.method()
    }

    /// Request target bytes (raw path + query).
    #[must_use]
    pub fn path(&self) -> &[u8] {
        self.connection.path()
    }

    /// Borrow a header value by name (case-insensitive).
    #[must_use]
    pub fn header_value(&self, name: &[u8]) -> Option<&[u8]> {
        self.connection.header_value(name)
    }

    /// Parsed `Content-Length` header value, if present and valid.
    #[must_use]
    pub fn content_length(&self) -> Option<u64> {
        self.header_value(b"content-length")
            .and_then(|raw| core::str::from_utf8(raw).ok())
            .and_then(|text| text.trim().parse().ok())
    }

    /// Emit `HTTP/1.1 100 Continue\r\n\r\n` into `out` and clear
    /// the pending flag. The caller writes `out` to the socket
    /// before resuming `advance` to pull body bytes.
    pub fn accept(self, out: &mut Vec<u8>) {
        self.connection.accept_continue(out);
    }

    /// Begin a final response (e.g. 413 Payload Too Large or 417
    /// Expectation Failed) without reading the body. Consumes the
    /// gate; the returned writer drives the rejection write and
    /// transitions to AfterResponse on completion.
    #[must_use = "the rejection response must be ended via the writer"]
    pub fn reject(
        self,
        status: u16,
        reason: &str,
        headers: &[(String, String)],
        framing: BodyFraming,
        out: &mut Vec<u8>,
    ) -> ResponseWriter<'a> {
        self.connection
            .begin_response(status, reason, headers, framing, out)
    }
}

/// Handle for the streaming-mode `HeadReady` poll outcome. The
/// listener has the parsed head and decides whether to dispatch
/// the streaming body (via subsequent `advance` calls) or to
/// short-circuit with an early response.
pub struct HeadReadyHandle<'a> {
    connection: &'a mut Connection,
}

impl<'a> HeadReadyHandle<'a> {
    #[must_use]
    pub fn method(&self) -> &[u8] {
        self.connection.method()
    }

    #[must_use]
    pub fn path(&self) -> &[u8] {
        self.connection.path()
    }

    #[must_use]
    pub fn version(&self) -> HttpVersion {
        self.connection.version()
    }

    #[must_use]
    pub fn header_value(&self, name: &[u8]) -> Option<&[u8]> {
        self.connection.header_value(name)
    }

    /// Iterate headers without allocating.
    #[must_use]
    pub fn headers(&self) -> HeadersIter<'_> {
        self.connection.headers()
    }

    #[must_use]
    pub fn keep_alive(&self) -> bool {
        self.connection.keep_alive()
    }

    #[must_use]
    pub fn content_length(&self) -> Option<u64> {
        self.header_value(b"content-length")
            .and_then(|raw| core::str::from_utf8(raw).ok())
            .and_then(|text| text.trim().parse().ok())
    }

    /// Construct a full `RequestHead<'_>` (allocates a header
    /// Vec). For hot-path code prefer the zero-alloc accessors
    /// above.
    #[must_use]
    pub fn head(&self) -> Option<RequestHead<'_>> {
        self.connection.head()
    }

    /// Begin an early response (e.g. 413 Payload Too Large) before
    /// reading the body. Consumes the handle.
    #[must_use = "the early response must be ended via the writer"]
    pub fn begin_response(
        self,
        status: u16,
        reason: &str,
        headers: &[(String, String)],
        framing: BodyFraming,
        out: &mut Vec<u8>,
    ) -> ResponseWriter<'a> {
        self.connection
            .begin_response(status, reason, headers, framing, out)
    }
}

/// Handle for the streaming-mode `BodyChunk` poll outcome. A chunk
/// is guaranteed to be present — call `take_chunk` exactly once.
pub struct BodyChunkHandle<'a> {
    connection: &'a mut Connection,
}

impl BodyChunkHandle<'_> {
    #[must_use]
    pub fn method(&self) -> &[u8] {
        self.connection.method()
    }

    #[must_use]
    pub fn path(&self) -> &[u8] {
        self.connection.path()
    }

    /// Take the queued body chunk. By construction at least one
    /// chunk is present when this handle is yielded — callers must
    /// consume it before polling for more.
    #[must_use]
    #[allow(clippy::expect_used)] // typestate guarantees Some; expect is a structural assertion
    pub fn take_chunk(self) -> Bytes {
        self.connection
            .take_body_chunk()
            .expect("BodyChunkHandle constructed without queued chunk")
    }
}

/// Handle for the streaming-mode `BodyEnd` poll outcome. The body
/// is fully decoded. Pull trailers (if any) and begin the response.
pub struct BodyEndHandle<'a> {
    connection: &'a mut Connection,
}

impl<'a> BodyEndHandle<'a> {
    #[must_use]
    pub fn method(&self) -> &[u8] {
        self.connection.method()
    }

    #[must_use]
    pub fn path(&self) -> &[u8] {
        self.connection.path()
    }

    #[must_use]
    pub fn header_value(&self, name: &[u8]) -> Option<&[u8]> {
        self.connection.header_value(name)
    }

    #[must_use]
    pub fn keep_alive(&self) -> bool {
        self.connection.keep_alive()
    }

    /// Captured request trailers. Empty for non-chunked bodies and
    /// chunked bodies without trailers.
    pub fn take_trailers(&mut self) -> Vec<(Bytes, Bytes)> {
        self.connection.take_trailers()
    }

    #[must_use = "the response must be ended via the writer"]
    pub fn begin_response(
        self,
        status: u16,
        reason: &str,
        headers: &[(String, String)],
        framing: BodyFraming,
        out: &mut Vec<u8>,
    ) -> ResponseWriter<'a> {
        self.connection
            .begin_response(status, reason, headers, framing, out)
    }
}

/// Handle for the buffered-mode `RequestReady` poll outcome. Head
/// and body are both present. Begin the response when ready.
pub struct BufferedRequestHandle<'a> {
    connection: &'a mut Connection,
}

impl<'a> BufferedRequestHandle<'a> {
    #[must_use]
    pub fn method(&self) -> &[u8] {
        self.connection.method()
    }

    #[must_use]
    pub fn path(&self) -> &[u8] {
        self.connection.path()
    }

    #[must_use]
    pub fn version(&self) -> HttpVersion {
        self.connection.version()
    }

    #[must_use]
    pub fn header_value(&self, name: &[u8]) -> Option<&[u8]> {
        self.connection.header_value(name)
    }

    #[must_use]
    pub fn headers(&self) -> HeadersIter<'_> {
        self.connection.headers()
    }

    /// Decoded body bytes. Borrowed from the connection's buffer —
    /// zero-copy when the caller doesn't need ownership.
    #[must_use]
    pub fn body(&self) -> &[u8] {
        self.connection.body()
    }

    #[must_use]
    pub fn keep_alive(&self) -> bool {
        self.connection.keep_alive()
    }

    /// Captured request trailers. Same shape as
    /// [`BodyEndHandle::take_trailers`] but for the buffered path.
    pub fn take_trailers(&mut self) -> Vec<(Bytes, Bytes)> {
        self.connection.take_trailers()
    }

    #[must_use]
    pub fn head(&self) -> Option<RequestHead<'_>> {
        self.connection.head()
    }

    #[must_use = "the response must be ended via the writer"]
    pub fn begin_response(
        self,
        status: u16,
        reason: &str,
        headers: &[(String, String)],
        framing: BodyFraming,
        out: &mut Vec<u8>,
    ) -> ResponseWriter<'a> {
        self.connection
            .begin_response(status, reason, headers, framing, out)
    }
}

/// Allocation-free header iterator. Slices each name/value pair
/// out of the connection buffer on demand.
pub struct HeadersIter<'a> {
    buffer: &'a [u8],
    offsets: &'a [HeaderOffsets],
    cursor: usize,
}

impl<'a> Iterator for HeadersIter<'a> {
    type Item = Header<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let entry = self.offsets.get(self.cursor)?;
        self.cursor += 1;
        Some(Header::new(
            &self.buffer[entry.name.0..entry.name.1],
            &self.buffer[entry.value.0..entry.value.1],
        ))
    }
}

/// Decide body framing from the offset-based header table —
/// allocation-free variant of `body_framing_from_head`. Iterates
/// the pre-populated offsets and slices the buffer directly.
fn body_framing_from_offsets(
    buffer: &[u8],
    headers: &[HeaderOffsets],
) -> Result<BodyFraming, ReadError> {
    let mut content_length: Option<u64> = None;
    let mut chunked = false;
    let mut other_encoding = false;
    for offsets in headers {
        let name = &buffer[offsets.name.0..offsets.name.1];
        let value = &buffer[offsets.value.0..offsets.value.1];
        if eq_ignore_ascii_case(name, b"content-length") {
            let raw = core::str::from_utf8(value)
                .map_err(|_| ReadError::BadContentLength)?
                .trim();
            let parsed: u64 = raw.parse().map_err(|_| ReadError::BadContentLength)?;
            content_length = Some(parsed);
        } else if eq_ignore_ascii_case(name, b"transfer-encoding") {
            let text = core::str::from_utf8(value).unwrap_or("");
            for token in text.split(',') {
                let trimmed = token.trim();
                if eq_ignore_ascii_case(trimmed.as_bytes(), b"chunked") {
                    chunked = true;
                } else if !trimmed.is_empty()
                    && !eq_ignore_ascii_case(trimmed.as_bytes(), b"identity")
                {
                    other_encoding = true;
                }
            }
        }
    }
    if chunked && content_length.is_some() {
        return Err(ReadError::AmbiguousFraming);
    }
    if other_encoding {
        return Err(ReadError::UnsupportedTransferEncoding);
    }
    if chunked {
        Ok(BodyFraming::Chunked)
    } else if let Some(length) = content_length {
        Ok(BodyFraming::ContentLength(length))
    } else {
        Ok(BodyFraming::None)
    }
}

fn expects_100_continue_from_offsets(buffer: &[u8], headers: &[HeaderOffsets]) -> bool {
    for offsets in headers {
        let name = &buffer[offsets.name.0..offsets.name.1];
        if !eq_ignore_ascii_case(name, b"expect") {
            continue;
        }
        let value = &buffer[offsets.value.0..offsets.value.1];
        let text = core::str::from_utf8(value).unwrap_or("");
        // RFC 7231 §5.1.1 allows a comma-separated list of expectations.
        for token in text.split(',') {
            if eq_ignore_ascii_case(token.trim().as_bytes(), b"100-continue") {
                return true;
            }
        }
    }
    false
}

fn policy_matches(policy: &AutoStreamPolicy, framing: BodyFraming) -> bool {
    match framing {
        BodyFraming::Chunked => policy.stream_chunked,
        BodyFraming::ContentLength(length) => length > policy.content_length_threshold,
        BodyFraming::None => false,
    }
}

fn keep_alive_from_offsets(buffer: &[u8], headers: &[HeaderOffsets], version: HttpVersion) -> bool {
    let mut header_close = false;
    let mut header_keep_alive = false;
    for offsets in headers {
        let name = &buffer[offsets.name.0..offsets.name.1];
        if !eq_ignore_ascii_case(name, b"connection") {
            continue;
        }
        let value = &buffer[offsets.value.0..offsets.value.1];
        let text = core::str::from_utf8(value).unwrap_or("");
        for token in text.split(',') {
            let trimmed = token.trim();
            if eq_ignore_ascii_case(trimmed.as_bytes(), b"close") {
                header_close = true;
            } else if eq_ignore_ascii_case(trimmed.as_bytes(), b"keep-alive") {
                header_keep_alive = true;
            }
        }
    }
    match version {
        HttpVersion::Http11 => !header_close,
        HttpVersion::Http10 => header_keep_alive,
    }
}

fn eq_ignore_ascii_case(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right.iter())
        .all(|(left_byte, right_byte)| left_byte.eq_ignore_ascii_case(right_byte))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    #[test]
    fn complete_request_ready_with_head_and_body_borrowed_into_buffer() {
        let mut connection = Connection::new();
        connection.feed_bytes(b"POST /v1 HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello");
        assert_eq!(connection.poll().expect("poll"), Poll::RequestReady);
        let head = connection.head().expect("head");
        assert_eq!(head.method, b"POST");
        assert_eq!(head.path, b"/v1");
        assert_eq!(connection.body(), b"hello");
        let buffer_ptr = connection.buffer.as_ptr() as usize;
        let buffer_end = buffer_ptr + connection.buffer.len();
        let method_ptr = head.method.as_ptr() as usize;
        let body_ptr = connection.body().as_ptr() as usize;
        assert!(method_ptr >= buffer_ptr && method_ptr < buffer_end);
        assert!(body_ptr >= buffer_ptr && body_ptr < buffer_end);
    }

    #[test]
    fn partial_input_returns_need_input_until_double_crlf() {
        let mut connection = Connection::new();
        connection.feed_bytes(b"GET / HTTP/1.1\r\nHost:");
        assert_eq!(connection.poll().expect("poll"), Poll::NeedInput);
        connection.feed_bytes(b" example.com\r\n\r\n");
        assert_eq!(connection.poll().expect("poll"), Poll::RequestReady);
    }

    #[test]
    fn write_response_emits_head_then_body_then_terminator() {
        let mut connection = Connection::new();
        connection.feed_bytes(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n");
        assert_eq!(connection.poll().expect("poll"), Poll::RequestReady);
        let headers = vec![("content-length".to_string(), "2".to_string())];
        let mut out = Vec::new();
        let writer =
            connection.begin_response(200, "OK", &headers, BodyFraming::ContentLength(2), &mut out);
        writer.write_chunk(b"ok", &mut out);
        writer.end_response(&mut out);
        let text = core::str::from_utf8(&out).expect("ascii");
        assert!(text.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(text.contains("content-length: 2\r\n"));
        assert!(text.ends_with("\r\n\r\nok"));
    }

    #[test]
    fn keep_alive_default_for_http11_then_close_after_response_with_connection_close() {
        let mut connection = Connection::new();
        connection.feed_bytes(b"GET / HTTP/1.1\r\nConnection: close\r\n\r\n");
        let _ = connection.poll().expect("poll");
        assert!(!connection.keep_alive());
        let mut out = Vec::new();
        let writer = connection.begin_response(200, "OK", &[], BodyFraming::None, &mut out);
        writer.end_response(&mut out);
        assert_eq!(connection.poll().expect("poll"), Poll::Close);
    }

    #[test]
    fn pipelined_request_picked_up_after_reset_without_memcpy() {
        let mut connection = Connection::new();
        connection.feed_bytes(b"GET /a HTTP/1.1\r\n\r\nGET /b HTTP/1.1\r\n\r\n");
        assert_eq!(connection.poll().expect("poll"), Poll::RequestReady);
        assert_eq!(connection.head().expect("head").path, b"/a");
        let mut out = Vec::new();
        let writer = connection.begin_response(200, "OK", &[], BodyFraming::None, &mut out);
        writer.end_response(&mut out);
        connection.reset_for_next_request();
        assert_eq!(connection.poll().expect("poll"), Poll::RequestReady);
        assert_eq!(connection.head().expect("head").path, b"/b");
    }

    #[test]
    fn ambiguous_framing_rejected() {
        let mut connection = Connection::new();
        connection.feed_bytes(
            b"POST / HTTP/1.1\r\nContent-Length: 5\r\nTransfer-Encoding: chunked\r\n\r\n",
        );
        assert_eq!(connection.poll(), Err(ReadError::AmbiguousFraming));
    }

    #[test]
    fn chunked_body_decodes_into_buffer() {
        let mut connection = Connection::new();
        connection.feed_bytes(
            b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n",
        );
        assert_eq!(connection.poll().expect("poll"), Poll::RequestReady);
        let body = connection.body();
        assert!(body.windows(5).any(|window| window == b"hello"));
        assert!(body.windows(6).any(|window| window == b" world"));
    }

    #[test]
    fn reset_after_pipelined_keeps_buffer_bounded() {
        let mut connection = Connection::new();
        // Single request → reset should collapse the buffer back to 0.
        connection.feed_bytes(b"GET /a HTTP/1.1\r\n\r\n");
        assert_eq!(connection.poll().expect("poll"), Poll::RequestReady);
        let mut out = Vec::new();
        let writer = connection.begin_response(200, "OK", &[], BodyFraming::None, &mut out);
        writer.end_response(&mut out);
        connection.reset_for_next_request();
        assert_eq!(connection.buffered_bytes(), 0);
        assert_eq!(connection.request_start, 0);
    }

    #[test]
    fn streaming_mode_emits_head_ready_then_chunks_then_body_end_for_content_length() {
        let mut connection = Connection::new();
        connection.set_streaming(true);
        connection.feed_bytes(b"POST /up HTTP/1.1\r\nContent-Length: 11\r\n\r\nhello ");
        assert_eq!(connection.poll().expect("poll"), Poll::HeadReady);
        assert_eq!(connection.method(), b"POST");
        assert_eq!(connection.path(), b"/up");
        // chunk 1 arrives in the first feed
        assert_eq!(connection.poll().expect("poll"), Poll::BodyChunk);
        let chunk_one = connection.take_body_chunk().expect("chunk1");
        assert_eq!(&chunk_one[..], b"hello ");
        assert_eq!(connection.poll().expect("poll"), Poll::NeedInput);
        // chunk 2 arrives on subsequent feed
        connection.feed_bytes(b"world");
        assert_eq!(connection.poll().expect("poll"), Poll::BodyChunk);
        let chunk_two = connection.take_body_chunk().expect("chunk2");
        assert_eq!(&chunk_two[..], b"world");
        assert_eq!(connection.poll().expect("poll"), Poll::BodyEnd);
        assert!(connection.take_body_chunk().is_none());
    }

    #[test]
    fn streaming_mode_emits_chunked_body_chunk_by_chunk_with_body_end() {
        let mut connection = Connection::new();
        connection.set_streaming(true);
        connection.feed_bytes(b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n");
        assert_eq!(connection.poll().expect("poll"), Poll::HeadReady);
        // Feed first chunk frame.
        connection.feed_bytes(b"5\r\nhello\r\n");
        assert_eq!(connection.poll().expect("poll"), Poll::BodyChunk);
        assert_eq!(&connection.take_body_chunk().expect("chunk")[..], b"hello");
        assert_eq!(connection.poll().expect("poll"), Poll::NeedInput);
        // Feed second chunk + terminator together.
        connection.feed_bytes(b"6\r\n world\r\n0\r\n\r\n");
        assert_eq!(connection.poll().expect("poll"), Poll::BodyChunk);
        assert_eq!(&connection.take_body_chunk().expect("chunk")[..], b" world");
        assert_eq!(connection.poll().expect("poll"), Poll::BodyEnd);
    }

    #[test]
    fn streaming_mode_never_emits_request_ready_for_no_body_request() {
        // Body-less GET — decoder ends immediately. Should jump from
        // HeadReady straight to BodyEnd without a BodyChunk.
        let mut connection = Connection::new();
        connection.set_streaming(true);
        connection.feed_bytes(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n");
        assert_eq!(connection.poll().expect("poll"), Poll::HeadReady);
        assert_eq!(connection.poll().expect("poll"), Poll::BodyEnd);
        assert!(connection.take_body_chunk().is_none());
    }

    #[test]
    fn streaming_mode_off_preserves_buffered_request_ready_path() {
        let mut connection = Connection::new();
        // Default: streaming_mode = false.
        assert!(!connection.is_streaming());
        connection.feed_bytes(b"POST / HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello");
        assert_eq!(connection.poll().expect("poll"), Poll::RequestReady);
        assert_eq!(connection.body(), b"hello");
        assert!(connection.take_body_chunk().is_none());
    }

    #[test]
    fn streaming_mode_then_reset_clears_emission_flags_for_pipelined_request() {
        let mut connection = Connection::new();
        connection.set_streaming(true);
        connection.feed_bytes(b"POST /a HTTP/1.1\r\nContent-Length: 2\r\n\r\nok");
        assert_eq!(connection.poll().expect("poll"), Poll::HeadReady);
        assert_eq!(connection.poll().expect("poll"), Poll::BodyChunk);
        let _ = connection.take_body_chunk().expect("chunk");
        assert_eq!(connection.poll().expect("poll"), Poll::BodyEnd);
        let mut out = Vec::new();
        let writer = connection.begin_response(200, "OK", &[], BodyFraming::None, &mut out);
        writer.end_response(&mut out);
        connection.reset_for_next_request();
        // Streaming flag persists; per-request flags reset → HeadReady fires again.
        connection.feed_bytes(b"GET /b HTTP/1.1\r\nHost: x\r\n\r\n");
        assert_eq!(connection.poll().expect("poll"), Poll::HeadReady);
        assert_eq!(connection.path(), b"/b");
        assert_eq!(connection.poll().expect("poll"), Poll::BodyEnd);
    }

    #[test]
    fn streaming_chunk_outlives_buffer_compaction() {
        // After reset_for_next_request the buffer drains the consumed
        // prefix; any earlier-emitted chunk must still be valid
        // because chunks are owned Bytes copies.
        let mut connection = Connection::new();
        connection.set_streaming(true);
        connection.feed_bytes(b"POST / HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello");
        let _ = connection.poll().expect("poll"); // HeadReady
        let _ = connection.poll().expect("poll"); // BodyChunk
        let chunk = connection.take_body_chunk().expect("chunk");
        let _ = connection.poll().expect("poll"); // BodyEnd
        let mut out = Vec::new();
        let writer = connection.begin_response(200, "OK", &[], BodyFraming::None, &mut out);
        writer.end_response(&mut out);
        connection.reset_for_next_request();
        // Chunk must still be readable post-reset.
        assert_eq!(&chunk[..], b"hello");
    }

    #[test]
    fn auto_stream_policy_flips_streaming_on_chunked_body() {
        let mut connection = Connection::new();
        connection.set_auto_stream_policy(Some(AutoStreamPolicy::default()));
        connection.feed_bytes(b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n");
        assert_eq!(connection.poll().expect("poll"), Poll::HeadReady);
        assert!(connection.is_streaming());
    }

    #[test]
    fn auto_stream_policy_stays_buffered_for_small_content_length() {
        let mut connection = Connection::new();
        connection.set_auto_stream_policy(Some(AutoStreamPolicy::default()));
        // 5 bytes is well under the 1 MiB threshold — buffered path wins.
        connection.feed_bytes(b"POST / HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello");
        assert_eq!(connection.poll().expect("poll"), Poll::RequestReady);
        assert!(!connection.is_streaming());
        assert_eq!(connection.body(), b"hello");
    }

    #[test]
    fn auto_stream_policy_flips_streaming_on_large_content_length() {
        let mut connection = Connection::new();
        // Tiny threshold for the test — anything > 3 bytes triggers streaming.
        connection.set_auto_stream_policy(Some(AutoStreamPolicy {
            content_length_threshold: 3,
            stream_chunked: true,
        }));
        connection.feed_bytes(b"POST / HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello");
        assert_eq!(connection.poll().expect("poll"), Poll::HeadReady);
        assert!(connection.is_streaming());
    }

    #[test]
    fn expect_100_continue_emits_poll_variant_then_accept_writes_status_line() {
        let mut connection = Connection::new();
        connection
            .feed_bytes(b"POST / HTTP/1.1\r\nContent-Length: 5\r\nExpect: 100-continue\r\n\r\n");
        // Expect resolution comes first.
        assert_eq!(connection.poll().expect("poll"), Poll::Expect100Continue);
        assert!(connection.expects_continue());
        // Head is accessible at this point.
        assert_eq!(connection.method(), b"POST");
        assert_eq!(
            connection.header_value(b"content-length"),
            Some(b"5".as_slice())
        );
        // Accept: writes the 100 status line and clears the flag.
        let mut out = Vec::new();
        connection.accept_continue(&mut out);
        assert_eq!(&out[..], b"HTTP/1.1 100 Continue\r\n\r\n");
        assert!(!connection.expects_continue());
        // Now feed the body and poll again — buffered RequestReady.
        connection.feed_bytes(b"hello");
        assert_eq!(connection.poll().expect("poll"), Poll::RequestReady);
        assert_eq!(connection.body(), b"hello");
    }

    #[test]
    fn expect_100_continue_with_streaming_mode_fires_before_head_ready() {
        let mut connection = Connection::new();
        connection.set_streaming(true);
        connection.feed_bytes(
            b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\nExpect: 100-continue\r\n\r\n",
        );
        // Expect blocks even the streaming HeadReady emission.
        assert_eq!(connection.poll().expect("poll"), Poll::Expect100Continue);
        let mut out = Vec::new();
        connection.accept_continue(&mut out);
        assert!(out.starts_with(b"HTTP/1.1 100 Continue"));
        // Next poll surfaces HeadReady (streaming).
        assert_eq!(connection.poll().expect("poll"), Poll::HeadReady);
    }

    #[test]
    fn expect_100_continue_rejection_via_begin_response_clears_flag_and_advances() {
        let mut connection = Connection::new();
        connection.feed_bytes(
            b"POST /big HTTP/1.1\r\nContent-Length: 9999999\r\nExpect: 100-continue\r\n\r\n",
        );
        assert_eq!(connection.poll().expect("poll"), Poll::Expect100Continue);
        // Listener rejects: write 413 directly via begin_response.
        let mut out = Vec::new();
        let writer = connection.begin_response(
            413,
            "Payload Too Large",
            &[("content-length".to_string(), "5".to_string())],
            BodyFraming::ContentLength(5),
            &mut out,
        );
        writer.write_chunk(b"nope!", &mut out);
        writer.end_response(&mut out);
        // Flag was cleared by begin_response. Subsequent poll returns
        // Close (state advanced through AfterResponse).
        assert!(!connection.expects_continue());
        // We didn't read the keep-alive bit (Content-Length present but
        // body never decoded — that's fine for a rejection close).
        assert_eq!(connection.poll().expect("poll"), Poll::Close);
    }

    #[test]
    fn expect_header_with_no_continue_token_does_not_trigger() {
        // RFC allows multiple expectations. Only `100-continue` is the
        // one we react to. An unknown expectation token should fall
        // through to the body phase normally (the listener could
        // reject with 417 if it wanted, but Connection doesn't
        // gate on it).
        let mut connection = Connection::new();
        connection.feed_bytes(
            b"POST / HTTP/1.1\r\nContent-Length: 2\r\nExpect: chocolate-rain\r\n\r\nok",
        );
        assert_eq!(connection.poll().expect("poll"), Poll::RequestReady);
        assert!(!connection.expects_continue());
        assert_eq!(connection.body(), b"ok");
    }

    #[test]
    fn expect_header_case_insensitive_and_comma_separated() {
        // RFC 7231 §5.1.1: header name and token are case-insensitive,
        // and Expect can carry a comma-separated list.
        let mut connection = Connection::new();
        connection.feed_bytes(
            b"POST / HTTP/1.1\r\nContent-Length: 1\r\nexpect: other, 100-Continue\r\n\r\nx",
        );
        assert_eq!(connection.poll().expect("poll"), Poll::Expect100Continue);
        let mut out = Vec::new();
        connection.accept_continue(&mut out);
        assert!(out.starts_with(b"HTTP/1.1 100 Continue"));
    }

    #[test]
    fn expect_pending_resets_between_requests() {
        // First request: Expect, accept, complete. Second request on
        // same connection has no Expect and proceeds normally.
        let mut connection = Connection::new();
        connection
            .feed_bytes(b"POST /a HTTP/1.1\r\nContent-Length: 2\r\nExpect: 100-continue\r\n\r\nok");
        assert_eq!(connection.poll().expect("poll"), Poll::Expect100Continue);
        let mut out = Vec::new();
        connection.accept_continue(&mut out);
        assert_eq!(connection.poll().expect("poll"), Poll::RequestReady);
        out.clear();
        let writer = connection.begin_response(200, "OK", &[], BodyFraming::None, &mut out);
        writer.end_response(&mut out);
        connection.reset_for_next_request();
        assert!(!connection.expects_continue());
        connection.feed_bytes(b"GET /b HTTP/1.1\r\nHost: x\r\n\r\n");
        assert_eq!(connection.poll().expect("poll"), Poll::RequestReady);
        assert!(!connection.expects_continue());
    }

    #[test]
    fn advance_buffered_request_handle_exposes_head_body_and_begins_response() {
        let mut connection = Connection::new();
        connection.feed_bytes(b"POST /v1 HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello");
        let mut out = Vec::new();
        match connection.advance().expect("advance") {
            Advanced::RequestReady(request) => {
                assert_eq!(request.method(), b"POST");
                assert_eq!(request.path(), b"/v1");
                assert_eq!(request.body(), b"hello");
                let writer = request.begin_response(200, "OK", &[], BodyFraming::None, &mut out);
                writer.end_response(&mut out);
            }
            other => panic!("expected RequestReady, got {other:?}"),
        }
        assert!(out.starts_with(b"HTTP/1.1 200 OK\r\n"));
    }

    #[test]
    fn advance_head_ready_handle_in_streaming_mode_lets_listener_inspect_before_body() {
        let mut connection = Connection::new();
        connection.set_auto_stream_policy(Some(AutoStreamPolicy {
            content_length_threshold: 0,
            stream_chunked: true,
        }));
        connection.feed_bytes(b"POST / HTTP/1.1\r\nContent-Length: 5\r\n\r\n");
        match connection.advance().expect("advance") {
            Advanced::HeadReady(head) => {
                assert_eq!(head.method(), b"POST");
                assert_eq!(head.content_length(), Some(5));
                // body() is NOT available on HeadReadyHandle — this would
                // be a compile error if uncommented:
                //   let _ = head.body();
            }
            other => panic!("expected HeadReady, got {other:?}"),
        }
    }

    #[test]
    fn advance_expect_gate_accepts_emits_100_continue_then_proceeds() {
        let mut connection = Connection::new();
        connection
            .feed_bytes(b"POST / HTTP/1.1\r\nContent-Length: 5\r\nExpect: 100-continue\r\n\r\n");
        let mut out = Vec::new();
        match connection.advance().expect("advance") {
            Advanced::Expect100Continue(gate) => {
                assert_eq!(gate.method(), b"POST");
                assert_eq!(gate.content_length(), Some(5));
                gate.accept(&mut out);
            }
            other => panic!("expected Expect100Continue, got {other:?}"),
        }
        assert_eq!(&out[..], b"HTTP/1.1 100 Continue\r\n\r\n");
        // Subsequent advance — body still incoming.
        match connection.advance().expect("advance") {
            Advanced::NeedInput => {}
            other => panic!("expected NeedInput after accept, got {other:?}"),
        }
        connection.feed_bytes(b"hello");
        match connection.advance().expect("advance") {
            Advanced::RequestReady(request) => {
                assert_eq!(request.body(), b"hello");
            }
            other => panic!("expected RequestReady, got {other:?}"),
        }
    }

    #[test]
    fn advance_expect_gate_rejects_with_413_skipping_body() {
        let mut connection = Connection::new();
        connection
            .feed_bytes(b"POST / HTTP/1.1\r\nContent-Length: 9999\r\nExpect: 100-continue\r\n\r\n");
        let mut out = Vec::new();
        match connection.advance().expect("advance") {
            Advanced::Expect100Continue(gate) => {
                let writer = gate.reject(
                    413,
                    "Payload Too Large",
                    &[("content-length".to_string(), "3".to_string())],
                    BodyFraming::ContentLength(3),
                    &mut out,
                );
                writer.write_chunk(b"big", &mut out);
                writer.end_response(&mut out);
            }
            other => panic!("expected Expect100Continue, got {other:?}"),
        }
        assert!(out.starts_with(b"HTTP/1.1 413 Payload Too Large\r\n"));
        // Connection moves to AfterResponse with keep_alive=false after rejection.
        assert!(matches!(
            connection.advance().expect("advance"),
            Advanced::Close
        ));
    }

    #[test]
    fn advance_streaming_chunk_handle_yields_exactly_one_chunk() {
        let mut connection = Connection::new();
        connection.set_auto_stream_policy(Some(AutoStreamPolicy {
            content_length_threshold: 0,
            stream_chunked: true,
        }));
        connection.feed_bytes(b"POST / HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello");
        // HeadReady first.
        match connection.advance().expect("advance") {
            Advanced::HeadReady(_) => {}
            other => panic!("expected HeadReady, got {other:?}"),
        }
        // Then BodyChunk.
        match connection.advance().expect("advance") {
            Advanced::BodyChunk(chunk) => {
                let bytes = chunk.take_chunk();
                assert_eq!(&bytes[..], b"hello");
            }
            other => panic!("expected BodyChunk, got {other:?}"),
        }
        // Then BodyEnd.
        match connection.advance().expect("advance") {
            Advanced::BodyEnd(end) => {
                assert!(end.keep_alive());
            }
            other => panic!("expected BodyEnd, got {other:?}"),
        }
    }

    #[test]
    fn response_writer_drop_without_end_advances_to_after_response_with_close() {
        // Forgetting `end_response` shouldn't leave the connection
        // stuck in Responding. Drop fires the defensive recovery:
        // state advances to AfterResponse, keep_alive=false → close.
        let mut connection = Connection::new();
        connection.feed_bytes(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n");
        let _ = connection.poll().expect("poll");
        let mut out = Vec::new();
        {
            let _writer = connection.begin_response(200, "OK", &[], BodyFraming::None, &mut out);
            // _writer dropped here without end_response being called.
        }
        // After drop, poll should report Close (terminal).
        assert_eq!(connection.poll().expect("poll"), Poll::Close);
    }
}
