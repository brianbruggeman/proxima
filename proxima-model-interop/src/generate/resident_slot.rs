//! One conversation's device KV left in place by the last request: llama.cpp's slot
//! (`tools/server/server-context.cpp:3198,3423`, upstream `f1ea20621`).
//!
//! Composes [`DeviceKv`] (the buffers), [`RingCheckpoint`] (llama's sliding-window checkpoint) and
//! [`CacheKey`] (staleness). [`DeviceKv::into_slot`] builds one and [`DeviceKv::adopt_resident`]
//! takes it apart. Reach for a `PromptCache` entry for a conversation that is not the live one:
//! entries hold host rows, a slot holds device rows held once.

use super::*;

pub(super) struct ResidentSlot {
    /// Token ids whose rows the device holds, e.g. the 4,080-token prompt plus what was decoded.
    pub(super) ids: Vec<u32>,
    /// The identity the rows were computed under; two requests share a slot only when it matches.
    pub(super) key: CacheKey,
    pub(super) device: DeviceKv,
    /// Sliding-window checkpoints, oldest first, for a rewind the ring rows no longer cover.
    pub(super) checkpoints: Vec<RingCheckpoint>,
}
