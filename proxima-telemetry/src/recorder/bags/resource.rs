use core::sync::atomic::{AtomicBool, Ordering};

use alloc::vec::Vec;

use crate::tag::{Tag, TagSink};

pub struct Resource {
    tags: Vec<Tag>,
    frozen: AtomicBool,
}

impl Default for Resource {
    fn default() -> Self {
        Self::new()
    }
}

impl Resource {
    pub fn new() -> Self {
        Self {
            tags: Vec::new(),
            frozen: AtomicBool::new(false),
        }
    }

    pub fn freeze(&self) {
        self.frozen.store(true, Ordering::Release);
    }

    pub fn tags(&self) -> &[Tag] {
        &self.tags
    }

    pub fn is_frozen(&self) -> bool {
        self.frozen.load(Ordering::Acquire)
    }
}

impl TagSink for Resource {
    fn push_tag(&mut self, tag: Tag) {
        if self.frozen.load(Ordering::Acquire) {
            return;
        }
        self.tags.push(tag);
    }
}
