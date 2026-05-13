use crate::id::SpanId;
use crate::tag::{ScalarValue, Tag, TagSink};
use crate::trace::link::TagInline;

#[derive(Clone)]
pub struct EventRecord {
    pub parent_span_id: SpanId,
    pub name: &'static str,
    pub ts_ns: u64,
    pub attrs: TagInline,
    pub module_path: &'static str,
    pub file_line: (u32, u32),
}

pub struct EventBuilder<'span> {
    parent: SpanId,
    name: &'static str,
    attrs: TagInline,
    events: &'span mut smallvec::SmallVec<[EventRecord; 2]>,
    module_path: &'static str,
    file_line: (u32, u32),
    ts_ns: u64,
}

impl<'span> EventBuilder<'span> {
    pub(crate) fn new(
        parent: SpanId,
        name: &'static str,
        ts_ns: u64,
        events: &'span mut smallvec::SmallVec<[EventRecord; 2]>,
    ) -> Self {
        Self {
            parent,
            name,
            attrs: smallvec::SmallVec::new(),
            events,
            module_path: "",
            file_line: (0, 0),
            ts_ns,
        }
    }

    pub fn tag(mut self, key: &'static str, value: impl Into<ScalarValue>) -> Self {
        self.attrs.push(Tag::Scalar {
            key,
            value: value.into(),
        });
        self
    }

    pub fn module_path(mut self, mod_path: &'static str) -> Self {
        self.module_path = mod_path;
        self
    }

    pub fn file_line(mut self, line: u32, col: u32) -> Self {
        self.file_line = (line, col);
        self
    }

    pub fn emit(self) {
        let record = EventRecord {
            parent_span_id: self.parent,
            name: self.name,
            ts_ns: self.ts_ns,
            attrs: self.attrs,
            module_path: self.module_path,
            file_line: self.file_line,
        };
        self.events.push(record);
    }
}

impl TagSink for EventBuilder<'_> {
    fn push_tag(&mut self, tag: Tag) {
        self.attrs.push(tag);
    }
}
