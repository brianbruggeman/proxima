use crate::id::{SpanId, TraceId};
use crate::tag::Tag;

/// P8 opt-sweep: SmallVec<[Tag; 4]> — typical spans have ≤4 attrs, fits inline.
pub type TagInline = smallvec::SmallVec<[Tag; 4]>;

#[derive(Clone, Debug)]
pub struct SpanLink {
    pub trace_id: TraceId,
    pub span_id: SpanId,
    pub attrs: TagInline,
}

impl SpanLink {
    pub fn new(trace_id: TraceId, span_id: SpanId) -> Self {
        Self {
            trace_id,
            span_id,
            attrs: smallvec::SmallVec::new(),
        }
    }
}
