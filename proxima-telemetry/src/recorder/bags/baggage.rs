// v1 stub — full propagation lands in C15 (substrate hooks on prime Task).
// Baggage carries per-task ambient context; it is not owned by Recorder.

use alloc::vec::Vec;

use crate::tag::{Tag, TagSink};

pub struct Baggage {
    tags: Vec<Tag>,
}

impl Default for Baggage {
    fn default() -> Self {
        Self::new()
    }
}

impl Baggage {
    pub fn new() -> Self {
        Self { tags: Vec::new() }
    }

    pub fn tags(&self) -> &[Tag] {
        &self.tags
    }
}

impl TagSink for Baggage {
    fn push_tag(&mut self, tag: Tag) {
        self.tags.push(tag);
    }
}
