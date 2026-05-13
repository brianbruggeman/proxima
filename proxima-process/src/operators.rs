//! Operator-applied dispatch units — composites built from inner
//! dispatch units.
//!
//! Operators here:
//! - [`AndThen`] — type-chained sequence (re-exported from the form
//!   family's `proxima_primitives::pipe::AndThen`; `A::Out = B::In`, marker
//!   traits propagate as the intersection — least-permissive wins,
//!   already proven once in the leaf crate).
//! - [`dispatch_match`] — async path-prefix routing for
//!   [`ChildRequest`]. Builds a routing table from
//!   `(&str, &dyn SendDynPipe<ChildRequest, ChildResponse>)`
//!   pairs and dispatches by longest-prefix-first match. The
//!   `match_operator!` macro for exhaustive enum-variant dispatch is
//!   a separate follow-up.
//!
//! Race / All / Tee / Quorum are not implemented in this initial
//! pass — they'll land when a use case requires them.

extern crate alloc;

pub use proxima_primitives::pipe::AndThen;

use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::alloc_tier::SendDynPipe;

/// Hand-written Match dispatch over [`ChildRequest`] by path
/// prefix. Walks `routes` in order; the first whose path matches the
/// request handles it. If nothing matches, dispatches to `fallback`.
///
/// This is `Match (first)` semantics from the operator catalog.
/// `Match (any)` (parallel predicate eval, first match wins) is a
/// follow-up.
///
/// The `match_operator!` macro for exhaustive enum-variant dispatch
/// is a separate construct (lands when the protocol stabilizes);
/// this function is the path-prefix-matched-routing variant.
pub async fn dispatch_match<F>(
    request: super::protocol::ChildRequest,
    routes: &[(
        &str,
        &dyn SendDynPipe<super::protocol::ChildRequest, super::protocol::ChildResponse>,
    )],
    fallback: &F,
) -> Result<super::protocol::ChildResponse, proxima_primitives::pipe::ProximaError>
where
    F: SendPipe<
            In = super::protocol::ChildRequest,
            Out = super::protocol::ChildResponse,
            Err = proxima_primitives::pipe::ProximaError,
        > + Sync,
{
    for (prefix, handler) in routes {
        if request.path().starts_with(prefix) {
            return handler.call_dyn(request).await;
        }
    }
    SendPipe::call(fallback, request).await
}
