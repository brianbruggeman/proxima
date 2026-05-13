use alloc::vec::Vec;

use crate::tag::{Tag, TagSink};

/// A named instrumentation scope: a label + version + scope-level tags that
/// `Recorder::span_from_scope` stamps onto every span started under it. Carries
/// no rings or clock — the recorder owns those — so a scope is cheap to mint.
pub struct ScopeHandle {
    pub name: &'static str,
    pub version: Option<&'static str>,
    pub tags: Vec<Tag>,
}

impl ScopeHandle {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            version: None,
            tags: Vec::new(),
        }
    }

    pub fn version(mut self, version: &'static str) -> Self {
        self.version = Some(version);
        self
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn tags(&self) -> &[Tag] {
        &self.tags
    }
}

impl TagSink for ScopeHandle {
    fn push_tag(&mut self, tag: Tag) {
        self.tags.push(tag);
    }
}
