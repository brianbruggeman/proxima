//! [`AnyProtocol`] — the per-candidate contract the open universal listener
//! ([`crate::any::AnyRegistry`], [`crate::any::Classifier`]) classifies
//! against. Deliberately a NEW trait, not a method bag on
//! [`crate::ListenProtocol`]: `ListenProtocol` owns a bind + an accept loop
//! ("run one socket"); an `AnyProtocol` owns neither — it is asked "is this
//! prefix you?" and, once chosen, "drive this one already-accepted stream."
//! [`crate::handle::Listener::any`]'s single accept loop is the ONE thing
//! that owns the bind; every registered candidate is a peer under it.
//!
//! [`ProbeVerdict`] generalizes [`crate::preface::PrefaceClass`] from "h1 vs
//! h2, a closed set of two" to an open, registry-driven set of arbitrarily
//! many candidates.

use std::any::Any;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use proxima_core::ProximaError;
use proxima_primitives::stream::{PeerInfo, StreamConnection};
use serde_json::Value;

use crate::admission::ConnAdmission;

/// Type-erased per-protocol handler carried through the open universal
/// listener's dispatch path. Different candidates want different handler
/// SHAPES — h1/h2 both want a
/// [`PipeHandle`](proxima_primitives::pipe::handler::PipeHandle)
/// (`Request<Bytes> -> Response<Bytes>`), but a future candidate is under
/// no obligation to share that shape (a redis candidate would want a
/// `Frame -> Frame` handle; a pgwire candidate a typed SQL query engine).
/// There is no one handler type every `AnyProtocol` could be generic over
/// without forcing every candidate through the same shape, so the registry
/// edge (`Listener::builder()`'s `.accept(name, handler)` /
/// `App`'s per-protocol default-handler registry) erases whatever handler a
/// candidate needs behind [`Any`], and that SAME candidate's own
/// [`AnyProtocol::drive`] is the only code that ever downcasts it back —
/// each candidate knows its own concrete handler type; nothing generic
/// needs to.
pub type AnyHandler = Arc<dyn Any + Send + Sync>;

/// Erase a concrete handler value behind [`AnyHandler`]. The registry edge
/// calls this once per binding; a candidate's own [`AnyProtocol::drive`]
/// reverses it via [`downcast_handler`].
pub fn erase_handler<T: Send + Sync + 'static>(handler: T) -> AnyHandler {
    Arc::new(handler)
}

/// Recover the concrete handler type `T` a candidate's own `drive` expects
/// from an [`AnyHandler`], or a named [`ProximaError::Config`] — never a
/// panic, never a silent drop — when the bound handler is the wrong shape
/// for the protocol it was bound to (e.g. a caller bound an h1
/// `PipeHandle` under the name `"h2"` by mistake). Takes `&AnyHandler` (not
/// by value) so a caller doesn't have to give up its own clone just to
/// attempt the downcast; the clone this function makes internally is one
/// cheap `Arc` bump.
pub fn downcast_handler<T: Send + Sync + 'static>(
    protocol_name: &str,
    handler: &AnyHandler,
) -> Result<Arc<T>, ProximaError> {
    handler.clone().downcast::<T>().map_err(|_| {
        ProximaError::Config(format!(
            "any listener: the handler bound to protocol '{protocol_name}' is not the type \
             that protocol expects"
        ))
    })
}

/// Why the open universal listener dropped a connection before any
/// candidate resolved. Passed to the optional reject hook
/// (`AnyListenProtocol`'s `on_reject`, `proxima-http`) so a future
/// deny/DoS-blacklist follow-on has a seam to observe rejections from
/// without re-plumbing the accept loop — this crate ships the seam, not a
/// policy (no deny list, no blacklist logic lives here).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum RejectReason {
    /// Every candidate answered `No` (or an unresolved simultaneous
    /// multi-match) before any one candidate could win outright.
    NoCandidateMatched { bytes_examined: usize },
    /// The accumulated prefix reached the classifier's `global_cap`
    /// without resolving — see [`crate::any::ClassifyOutcome::PrefixBoundExceeded`].
    PrefixBoundExceeded,
}

/// Outcome of handing a candidate the bytes accumulated so far on a fresh
/// connection. Generalizes [`crate::preface::PrefaceClass`]'s three-way
/// split to an open set: a candidate that isn't `Http1`'s hard-coded sibling
/// needs a way to say "not me" without collapsing the field to a fixed enum
/// of known protocols.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProbeVerdict {
    /// This candidate recognizes the prefix as its own wire. `consumed` is
    /// how many leading bytes are this candidate's own fixed framing (e.g.
    /// h2's 24-byte RFC 9113 §3.4 preface) as opposed to bytes that remain
    /// live protocol data once dispatched (e.g. h1's request line, which the
    /// h1 codec re-parses from byte zero). Informational only today — the
    /// listener driving [`crate::any::Classifier`] always replays the FULL
    /// accumulated prefix into [`AnyProtocol::drive`] regardless of this
    /// value (see that method's docs); a future prefix-trimming optimization
    /// may use it to avoid re-parsing bytes the candidate has already
    /// consumed.
    Match { consumed: usize },
    /// Still a live candidate, but not enough bytes have arrived to decide.
    /// The caller reads more and calls [`AnyProtocol::probe`] again with the
    /// larger buffer — bytes already seen are a prefix of the next call's
    /// buffer, nothing is discarded. `at_least` is the total prefix length
    /// (from byte zero) this candidate needs before it can answer `Match`
    /// or `No`.
    NeedMore { at_least: usize },
    /// The accumulated prefix is definitively not this candidate's wire.
    /// The caller drops this candidate from further consideration on this
    /// connection — probing it again with more bytes cannot change the
    /// answer, since a `No` verdict is only reached once the prefix
    /// diverges from the candidate's own framing at a fixed byte offset.
    No,
}

/// One registrable candidate protocol for the open universal listener. A
/// peer of [`crate::ListenProtocol`], not an extension of it:
/// `ListenProtocol` drives one bind's accept loop; `AnyProtocol` is asked to
/// classify a prefix and then drive ONE already-accepted stream — the
/// listener ([`crate::handle::Listener::any`]) owns the socket, the accept
/// loop, and the [`crate::any::Classifier`] that picks among registered
/// candidates.
///
/// Implementors: [`crate::preface`]'s h1/h2 dispatch teaches the shape —
/// `probe` is a pure, sans-IO `&[u8] -> ProbeVerdict` function (no I/O, no
/// allocation beyond what the verdict itself needs), mirroring
/// [`crate::preface::classify_preface`]. `drive` is where the real
/// (allocating, async, runtime-touching) work happens, exactly once, on the
/// stream the listener already accepted.
///
/// # Write your own protocol
///
/// This is the seam a downstream crate uses to teach `Listener::builder()`
/// a wire this workspace has never heard of — kafka/mqtt/amqp/memcached/
/// redis/dns all reach the open universal listener through exactly this
/// trait (see e.g. `proxima_kafka::KafkaAnyProtocol`); nothing about it is
/// first-party-only. Three pieces: (1) a type implementing `AnyProtocol`
/// (`probe` classifies a connection prefix, `drive` serves one accepted
/// connection), (2) a one-line extension trait so callers get a fluent
/// `.yourproto()` instead of the more verbose `.protocol(YourProtocol)`,
/// (3) `use your_crate::YourProtocolExt;` at the call site to light it up —
/// mirroring [`ListenerProtocolExt`](https://docs.rs/proxima/latest/proxima/trait.ListenerProtocolExt.html)'s
/// own `.kafka()`/`.mqtt()`/… shape.
///
/// ```
/// use std::future::Future;
/// use std::pin::Pin;
///
/// use proxima_core::ProximaError;
/// use proxima_listen::admission::ConnAdmission;
/// use proxima_listen::any::{AnyHandler, AnyProtocol, ProbeVerdict};
/// use proxima_primitives::stream::{PeerInfo, StreamConnection};
/// use serde_json::Value;
///
/// // (1) The protocol itself: a trivial `PING\r\n` -> `PONG\r\n` line
/// // protocol, sans-IO classification + one async drive.
/// struct PingProtocol;
///
/// impl AnyProtocol for PingProtocol {
///     fn name(&self) -> &str {
///         "ping"
///     }
///
///     fn max_prefix_bytes(&self) -> usize {
///         6 // len("PING\r\n")
///     }
///
///     fn probe(&self, prefix: &[u8]) -> ProbeVerdict {
///         const TAG: &[u8] = b"PING\r\n";
///         let compare_len = prefix.len().min(TAG.len());
///         if prefix[..compare_len] != TAG[..compare_len] {
///             return ProbeVerdict::No;
///         }
///         if prefix.len() < TAG.len() {
///             return ProbeVerdict::NeedMore { at_least: TAG.len() };
///         }
///         ProbeVerdict::Match { consumed: TAG.len() }
///     }
///
///     fn drive<'a>(
///         &'a self,
///         mut stream: Box<dyn StreamConnection>,
///         _handler: AnyHandler,
///         _spec: &'a Value,
///         _peer: Option<PeerInfo>,
///         _admission: &'a ConnAdmission,
///     ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'a>> {
///         Box::pin(async move {
///             use futures::io::AsyncWriteExt;
///             stream.write_all(b"PONG\r\n").await.map_err(ProximaError::Io)?;
///             Ok(())
///         })
///     }
/// }
///
/// // (2) The one-line extension trait — the same idiom
/// // `proxima::ListenerProtocolExt` uses for `.kafka()`/`.mqtt()`/….
/// trait PingExt: Sized {
///     fn ping(self) -> Self;
/// }
///
/// impl PingExt for proxima::ListenerBuilder {
///     fn ping(self) -> Self {
///         self.protocol(PingProtocol)
///     }
/// }
///
/// # use proxima::{Listener, ListenerBuilderEntry, ListenerProtocolExt, Request, Response, ProximaError as PxError};
/// # use proxima::pipe::into_handle;
/// # use proxima::SendPipe;
/// # use bytes::Bytes;
/// #
/// # struct Dispatch;
/// # impl SendPipe for Dispatch {
/// #     type In = Request<Bytes>;
/// #     type Out = Response<Bytes>;
/// #     type Err = PxError;
/// #     async fn call(&self, _request: Request<Bytes>) -> Result<Response<Bytes>, PxError> {
/// #         Ok(Response::new(404))
/// #     }
/// # }
/// #
/// # #[proxima::main]
/// # async fn main() -> Result<(), PxError> {
/// // (3) `use your_crate::YourProtocolExt;` (here, `PingExt` is local) ->
/// // `.ping()` reads exactly like the first-party `.kafka()`/`.mqtt()`.
/// let bind = std::net::SocketAddr::from(([127, 0, 0, 1], 0));
/// let server = Listener::builder()
///     .bind(bind)
///     .http(bind.to_string())
///     .ping()
///     .handle(into_handle(Dispatch))
///     .serve()
///     .await?;
/// server.stop();
/// # Ok(())
/// # }
/// ```
pub trait AnyProtocol: Send + Sync + 'static {
    /// Registry key and diagnostic label — mirrors
    /// [`crate::ListenProtocol::name`].
    fn name(&self) -> &str;

    /// Precedence among candidates that could still be alive at the same
    /// prefix length. Higher wins. Callers default to `100` (see
    /// [`crate::any::AnyRegistry`]'s doc) when a candidate has no opinion;
    /// candidates sharing a priority are resolved by
    /// [`crate::any::Classifier`]'s current rule (see that type's docs for
    /// the exact — currently provisional — arbitration).
    fn priority(&self) -> u16 {
        100
    }

    /// Upper bound on how many leading bytes this candidate ever needs to
    /// reach a verdict. Lets the listener size its per-connection read
    /// buffer and lets [`crate::any::Classifier`] report
    /// `PrefixBoundExceeded` against a real, bounded expectation rather than
    /// reading forever from a peer that never sends enough to resolve.
    fn max_prefix_bytes(&self) -> usize;

    /// Pure, sans-IO classification of the bytes accumulated so far, from
    /// byte zero of the connection. No I/O, no allocation: mirrors
    /// [`crate::preface::classify_preface`]'s contract exactly, generalized
    /// to an open set of candidates instead of a fixed h1-or-h2 choice.
    #[must_use]
    fn probe(&self, prefix: &[u8]) -> ProbeVerdict;

    /// Drive one already-accepted connection to completion once
    /// [`crate::any::Classifier::advance`] has chosen this candidate.
    /// `stream` carries the FULL accumulated prefix replayed at its front
    /// (via the same `Prepend`-shaped mechanism
    /// `proxima_http`'s former `dispatch_h1_or_h2` used) — the candidate's
    /// own wire parser sees an intact byte stream starting at byte zero, as
    /// if no bytes had ever been sniffed out ahead of it. `handler` is the
    /// [`AnyHandler`] bound to THIS candidate's own name (the listener
    /// looked it up by [`AnyProtocol::name`] before calling `drive`) —
    /// downcast it via [`downcast_handler`] to the concrete type this
    /// candidate expects (e.g. h1/h2 downcast to
    /// [`PipeHandle`](proxima_primitives::pipe::handler::PipeHandle));
    /// pgwire/redis candidates carry their own engine as struct fields and
    /// leave `handler` unused (documented asymmetry — see their own impls).
    ///
    /// `admission` is the listener-wide [`ConnAdmission`] handle: call
    /// [`ConnAdmission::request_admit`] at THIS candidate's own request
    /// boundary (h1 per request, h2 per stream, pgwire per message, redis
    /// per command), dispatch to the handler only on
    /// `RequestAdmit::Admit`, call [`ConnAdmission::request_release`] on
    /// completion, and on `Shed` render this protocol's OWN wire-specific
    /// rejection instead of dispatching — the listener owns the uniform
    /// admission policy (capacity, quiesce, drain); the protocol only
    /// reports boundaries and renders the reply.
    fn drive<'a>(
        &'a self,
        stream: Box<dyn StreamConnection>,
        handler: AnyHandler,
        spec: &'a Value,
        peer: Option<PeerInfo>,
        admission: &'a ConnAdmission,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'a>>;
}
