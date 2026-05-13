use core::future::Future;
use core::time::Duration;

#[cfg(feature = "alloc")]
use alloc::borrow::Cow;

use bytes::Bytes;
use proxima_core::ProximaError;

// ── payload capability traits ─────────────────────────────────────────────────
//
// These traits express what a payload must provide to generic middleware
// skeletons. The skeletons (Retry, etc.) are generic over the inner pipe;
// these traits are the injected seams. HTTP implements them on
// Request/Response in proxima-pipe (orphan rule). Non-HTTP payloads implement
// them in their own crate.
//
// A pass/reject decision is NOT one of these seams: `In -> Result<In, Err>`
// is a pipe (`Ok` = admit, `Err` = reject), composed via `and_then` — not a
// `decide(&In) -> bool` capability with companions to carry the item and the
// rejection back (see `pipe/fan_in.rs`'s `FanInStrategy` doc for the pipe-vs-
// strategy line this used to violate).

/// An input that can be re-fed to the inner pipe for each retry attempt. HTTP
/// forks the body stream through a `Replay` (bounded by `replay_cap_bytes`); a
/// cloneable payload (an event) clones. The skeleton owns neither mechanism.
pub trait Replayable: Sized {
    type Source: Send;
    fn fork(self, replay_cap_bytes: usize) -> (Self, Self::Source);
    fn replay(source: &Self::Source) -> Result<Self, ProximaError>;
}

/// Whether an input is safe to retry (HTTP: idempotent method).
pub trait Idempotent {
    fn is_idempotent(&self) -> bool;
}

/// Outcome classification for retry (HTTP: status codes). A statusless outcome
/// (a plain event) returns `None` and is governed only by error retry.
pub trait Retryable {
    fn retry_status(&self) -> Option<u16>;
    fn is_success(&self) -> bool;
}

/// The monotonic-clock seam the timer-driven executors (`Retry`, and later
/// `Timeout`/`Hedge`) are generic over. `Delay` is an associated future, so the
/// executor holds it inline in its state machine — no boxing, no alloc, no_std.
///
/// Production impls wrap `proxima-time` (its `Sleep` is a concrete `Delay`);
/// deterministic tests supply their own — the core owns neither clock.
pub trait Clock {
    /// A concrete, `Sized` future that resolves once `dur` has elapsed.
    type Delay: Future<Output = ()>;
    /// Monotonic nanoseconds; feeds `RetryController`'s deadline arithmetic.
    fn now_nanos(&self) -> u64;
    fn delay(&self, dur: Duration) -> Self::Delay;
}

/// A value that can apply a sequence of typed ops to itself.
///
/// Implement this on any input or output type to make Transform work over it.
/// HTTP: `Request<Bytes>: ApplyOps<RequestOp>`, `Response<Bytes>: ApplyOps<ResponseOp>`.
pub trait ApplyOps<Op>: Sized {
    fn apply(self, ops: &[Op]) -> Self;
}

/// A value that exposes a mutable byte buffer for fuzzing.
///
/// Implement this on any payload to make `Mutation` / `MutateOp` work over it.
/// HTTP impls it on `Request` and `Response` (their `body`); a non-HTTP payload
/// implements it over its own bytes.
pub trait BytePayload: Sized {
    /// Replace the payload's bytes with `bytes`.
    fn set_bytes(&mut self, bytes: Bytes);
    /// Borrow the payload's current bytes.
    fn bytes(&self) -> &[u8];
}

/// The action taken when a rate-limit bucket is exhausted.
#[derive(Debug, Clone, Copy)]
pub enum ExceededAction {
    Reject429 { retry_after_ms: u64 },
}

/// Extracts a rate-limit bucket key from an input and constructs the rejection
/// output when the bucket is exhausted.
///
/// Implement this on any input type to make `RateLimit` work over it.
/// HTTP: `Request<Bytes>: KeyOf<KeyExtractor>` with `type Rejection = Response<Bytes>`.
///
/// Additive at the alloc tier: the key is borrowed-or-owned (`Cow<[u8]>`), so
/// this seam needs `alloc`. The rest of the vocabulary is core-tier.
#[cfg(feature = "alloc")]
pub trait KeyOf<Extractor>: Sized {
    /// The output value produced when this input is rate-limited.
    type Rejection;

    /// The bucket key for this input given the extractor config.
    fn rate_key<'a>(&'a self, extractor: &'a Extractor) -> Cow<'a, [u8]>;

    /// Constructs the rejection output. Called only when the bucket is empty.
    fn build_rejection(action: &ExceededAction) -> Self::Rejection;
}

/// Outcome of a check: either admitted (pass through to inner) or rejected
/// (short-circuit with a pre-built output).
pub enum CheckOutcome<In, Out> {
    Pass(In),
    Reject(Out),
}

/// A value that can check itself against an op and decide pass/reject.
///
/// Implement this on any input type to make Validate work over it.
/// HTTP: `Request<Bytes>: Checkable<ValidateOp>` with `Out = Response<Bytes>`.
pub trait Checkable<Op>: Sized {
    type Out;

    fn check(
        self,
        op: &Op,
    ) -> impl Future<Output = Result<CheckOutcome<Self, Self::Out>, ProximaError>> + Send;
}
