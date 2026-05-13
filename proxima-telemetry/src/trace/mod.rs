// T0 — Copy/core enums, no `Tag`, no heap, no proxima-pipe.
pub mod kind;
pub mod status;

// T1 — `clock` re-exports the alloc-tier `crate::clock`; the record types carry
// `Tag` / `SmallVec` / `Bytes` and `TraceId`/`SpanId` (proxima-pipe).
#[cfg(feature = "alloc")]
pub mod clock;
#[cfg(feature = "alloc")]
pub mod event;
#[cfg(feature = "alloc")]
pub mod link;
#[cfg(feature = "alloc")]
pub mod span;
#[cfg(feature = "alloc")]
pub mod tracestate;

#[cfg(test)]
mod tests;

pub use kind::SpanKind;
pub use status::Status;

#[cfg(feature = "alloc")]
pub use clock::{Clock, MonotonicCounter};
#[cfg(feature = "alloc")]
pub use event::{EventBuilder, EventRecord};
#[cfg(feature = "alloc")]
pub use link::{SpanLink, TagInline};
#[cfg(feature = "alloc")]
pub use span::{Span, SpanBuilder, SpanGuard, SpanRecord, SpanSink};
#[cfg(feature = "alloc")]
pub use tracestate::TraceState;

// `SpanId` comes from proxima-pipe (the alloc tier), so the carrier trait rides
// with it.
#[cfg(feature = "alloc")]
use crate::id::SpanId;

/// Trait for envelope types that carry a span context across channel boundaries.
///
/// Implement via `#[derive(SpanCarrier)]` on any struct with a field of type
/// `Option<SpanId>` (named `span_id` or annotated with `#[span_id]`).
#[cfg(feature = "alloc")]
pub trait SpanCarrier {
    fn span_id(&self) -> Option<SpanId>;
    fn set_span_id(&mut self, id: Option<SpanId>);
}

/// Build a `SpanBuilder` with `module_path!` and `file_line!` auto-captured.
///
/// v1 shape: `span_builder!(trace_id, span_id, name)` or
///            `span_builder!(trace_id, span_id, name, "k" = v, ...)`.
///
/// C9 will refactor this once the recorder owns ID generation; at that point
/// trace_id/span_id will be omitted from call sites.
#[macro_export]
macro_rules! span_builder {
    ($trace_id:expr, $span_id:expr, $name:literal $(,)?) => {{
        let __b = $crate::trace::SpanBuilder::new($name, $trace_id, $span_id);
        let __b = __b.module_path(::core::module_path!());
        __b.file_line(::core::line!(), ::core::column!())
    }};
    ($trace_id:expr, $span_id:expr, $name:literal, $($rest:tt)+) => {{
        let __b = $crate::trace::SpanBuilder::new($name, $trace_id, $span_id);
        let __b = __b.module_path(::core::module_path!());
        let mut __b = __b.file_line(::core::line!(), ::core::column!());
        $crate::tag!(__b, $($rest)+);
        __b
    }};
}

/// Append an event to an in-flight `SpanGuard`. The guard owns the clock; no
/// extra clock argument is required at the call site.
///
/// Forms:
/// - `span_event!(guard, "name")` — no attrs
/// - `span_event!(guard, "name", "k" = v, ...)` — with attrs
#[macro_export]
macro_rules! span_event {
    ($span:expr, $name:literal $(,)?) => {{
        if let Some(mut __e) = $span.event($name) {
            __e = __e.module_path(::core::module_path!());
            __e = __e.file_line(::core::line!(), ::core::column!());
            __e.emit();
        }
    }};
    ($span:expr, $name:literal, $($rest:tt)+) => {{
        if let Some(mut __e) = $span.event($name) {
            __e = __e.module_path(::core::module_path!());
            __e = __e.file_line(::core::line!(), ::core::column!());
            $crate::tag!(__e, $($rest)+);
            __e.emit();
        }
    }};
}
