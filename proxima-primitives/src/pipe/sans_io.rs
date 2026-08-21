//! The sans-IO byte-stream parser contract — core-tier, no `std`, no
//! `alloc` dependency of its own (implementors own whatever buffer they
//! need; this module only names the shape).
//!
//! # Where this came from
//!
//! Three format parsers landed in this workspace independently
//! (`proxima-gguf::GgufParser`, `proxima-onnx::OnnxParser`,
//! `proxima-safetensors::SafetensorsParser`), each briefed on its own and
//! each inventing its own FSM shape. Two of the three already matched:
//! `GgufParser`/`OnnxParser` both imitated
//! `proxima-protocols::http1_codec::h1_connection::Connection`
//! (`feed_bytes`/`poll`, `&mut self`, one owned growing buffer) by name in
//! their own module docs — that type predates both and is the actual
//! prior art, not a fresh design. `SafetensorsParser` instead imitated
//! `proxima_codec::DelimiterFraming` (`push(self, chunk) -> Self`,
//! self-consuming, cursor folded into the enum variant) — a real,
//! independently-motivated shape in this workspace, but a different one.
//!
//! [`ByteStreamParser`] names the `&mut self` `feed`/`poll` shape as the
//! one contract, because it was *already* established twice
//! (`h1_connection::Connection`, and `GgufParser`/`OnnxParser` copying it)
//! before any of these three parsers existed, and because it is strictly
//! more capable than the self-consuming shape: it lets a caller feed and
//! drain in independent, arbitrarily-interleaved calls (`feed` any number
//! of times, then `poll` any number of times to drain a backlog of
//! events), which a self-consuming `push` cannot do without threading the
//! returned `Self` back through every call site by hand.
//!
//! `DelimiterFraming`'s self-consuming shape is not wrong where it lives —
//! its own doc explains the invariant it buys (the scan cursor can never
//! drift from the buffer it scans, because both live in the same enum
//! variant, moved together). That invariant is about `next_frame`'s
//! *internal* bookkeeping, not about the `&mut self` vs. self-consuming
//! choice at the public boundary: nothing stops a `&mut self` method from
//! reconstructing the same enum internally via `core::mem::replace` with a
//! cheap placeholder, which is exactly how `SafetensorsParser` now
//! implements this trait (`proxima-safetensors/src/parser.rs`) — same
//! variant-folded-cursor invariant, `&mut self` boundary.
//!
//! # Not a [`Pipe`](crate::pipe::Pipe)
//!
//! A parser here is fed bytes and polled for events an unbounded,
//! caller-controlled number of times per `feed` call (see `GgufParser`'s
//! `poll_kv`, which recurses into itself via `self.poll()` after
//! committing each KV pair with no new bytes in between). `Pipe::call` is
//! one `In -> Result<Out, Err>` step; there is no `In` shape that
//! expresses "attempt progress again against what's already buffered,
//! with no new input." This is the same non-`Pipe` argument
//! `proxima-safetensors::header_codec` and `proxima_codec::DelimiterFraming`
//! already made for their own stateful loops — restated here because it
//! applies to the whole trait, not just one impl.

/// One [`ByteStreamParser::poll`] outcome: either not enough buffered
/// bytes to make progress (call [`ByteStreamParser::feed`]), or one unit
/// of progress in the form of `Event`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome<Event> {
    /// Not enough buffered bytes yet — the normal partial-input signal,
    /// never a failure.
    NeedMore,
    Event(Event),
}

/// The sans-IO byte-stream parser contract. Owns its own buffering
/// strategy entirely; this trait only fixes the shape callers drive it
/// through.
///
/// `feed` never inspects or parses the bytes it is given — appending is
/// infallible and cannot fail on malformed input, because malformed input
/// is only discoverable once `poll` actually looks at the buffered bytes.
/// `poll` attempts exactly one unit of progress against whatever is
/// currently buffered and never blocks: [`Outcome::NeedMore`] means "call
/// `feed` again," not "wait." `finish` is a read-only check — `Ok(())`
/// only if the parser has reached a valid terminal state; otherwise the
/// byte stream ended mid-item. It never hands back an owned final value:
/// an impl that has one (e.g. `SafetensorsParser::into_manifest`) surfaces
/// it as a `poll` event or a separate accessor, keeping `finish`'s
/// signature uniform across every impl regardless of what "done" produces.
pub trait ByteStreamParser {
    /// One unit of progress. Declared as a GAT so a lending impl (e.g.
    /// `OnnxParser`, whose events borrow straight into its own
    /// accumulation buffer) and an owning impl (e.g. `GgufParser`, whose
    /// events are fully owned and just ignore `'a`) both satisfy this
    /// trait without either one paying for the other's shape.
    type Event<'a>
    where
        Self: 'a;
    /// This parser's typed error.
    type Error;

    /// Append bytes fed by the caller. Never blocks, never inspects the
    /// bytes.
    fn feed(&mut self, bytes: &[u8]);

    /// Attempt one unit of progress against the currently buffered bytes.
    fn poll(&mut self) -> Result<Outcome<Self::Event<'_>>, Self::Error>;

    /// The caller has no more bytes to feed. `Ok(())` only if the parser
    /// had already reached a valid terminal state.
    fn finish(&self) -> Result<(), Self::Error>;
}

/// Drives any [`ByteStreamParser`] to completion from a chunked byte
/// stream — feeds each chunk, drains every event it produces (looping
/// `poll` until [`Outcome::NeedMore`]) before asking for the next chunk,
/// then calls [`ByteStreamParser::finish`]. `on_event` runs once per
/// emitted event, in order.
///
/// This is the proof the shared contract is load-bearing rather than a
/// relocation: one generic loop drives `GgufParser`, `OnnxParser`, and
/// `SafetensorsParser` alike (see each crate's `sans_io` test module) —
/// before this trait existed, a caller wanting that had to hand-write the
/// same feed/poll loop three times, once per concrete type.
///
/// # Errors
///
/// The first [`ByteStreamParser::Error`] any `feed`d chunk's `poll` or the
/// final `finish` call surfaces.
pub fn drive_to_completion<P, I, F>(parser: &mut P, chunks: I, mut on_event: F) -> Result<(), P::Error>
where
    P: ByteStreamParser,
    I: IntoIterator,
    I::Item: AsRef<[u8]>,
    F: FnMut(P::Event<'_>),
{
    for chunk in chunks {
        parser.feed(chunk.as_ref());
        loop {
            match parser.poll()? {
                Outcome::NeedMore => break,
                Outcome::Event(event) => on_event(event),
            }
        }
    }
    parser.finish()
}
