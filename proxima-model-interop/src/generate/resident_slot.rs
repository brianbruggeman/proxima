//! One conversation's device KV left in place by the last request: llama.cpp's slot
//! (`tools/server/server-context.cpp:3198,3423`, upstream `f1ea20621`).
//!
//! Composes [`DeviceKv`] (the buffers), [`RingCheckpoint`] (llama's sliding-window checkpoint) and
//! [`CacheKey`] (staleness). [`DeviceKv::into_slot`] builds one and [`DeviceKv::adopt_resident`]
//! takes it apart. Reach for a `PromptCache` entry for a conversation that is not the live one:
//! entries hold host rows, a slot holds device rows held once.

use super::prompt_cache::{entry_is_reusable, longest_common_prefix};
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

impl ResidentSlot {
    /// Device bytes this slot holds, counted by [`DeviceKv::resident_bytes`].
    fn device_bytes(&self) -> usize {
        self.device.resident_bytes()
    }
}

/// What [`ResidentSlots::take`] found for a request's ids.
#[must_use]
pub(super) enum SlotTake {
    /// Nothing is resident.
    Empty,
    /// The slot shares enough of the prompt: rewind it to `keep` with [`DeviceKv::adopt_resident`]
    /// and prefill `ids[keep..]`. `keep` is capped at `ids.len() - 1`, so at least one token is
    /// re-evaluated (llama's `n_past was set to 4068` of 4,069).
    Reused { slot: ResidentSlot, keep: usize },
    /// A slot exists but cannot serve these ids (other [`CacheKey`], or too little shared prefix);
    /// the caller flushes it to a `PromptCache` entry.
    Evicted(ResidentSlot),
}

/// The conversations whose device KV stays in place between requests, least recently put first.
/// The request gate already serialises access, so this holds no lock. Composes [`ResidentSlot`],
/// [`longest_common_prefix`] and [`entry_is_reusable`], the rule a `PromptCache` entry is admitted
/// by; reach for a `PromptCache` entry for conversations that are not resident.
pub(super) struct ResidentSlots {
    slots: Vec<ResidentSlot>,
    capacity: usize,
    budget_bytes: usize,
}

impl ResidentSlots {
    /// `capacity` is `ServingConfig::parallel_sequences` and `budget_bytes` its `resident_kv_bytes`.
    pub(super) const fn new(capacity: usize, budget_bytes: usize) -> Self {
        Self { slots: Vec::new(), capacity, budget_bytes }
    }

    pub(super) fn len(&self) -> usize {
        self.slots.len()
    }

    /// Removes the slot that serves `ids` best and says how to use it, in this order:
    /// - the slot under `key` with the longest shared prefix, when [`entry_is_reusable`] admits it
    ///   at `min_similarity_milli` and at least one token remains to re-evaluate: `Reused`;
    /// - a slot under another key, which can never be reused: `Evicted`, even with room to spare;
    /// - otherwise `Empty` while the table has room (llama fills an idle slot before it evicts
    ///   one), and `Evicted` with the least recently put slot once it is full.
    pub(super) fn take(&mut self, ids: &[u32], key: &CacheKey, min_similarity_milli: u32) -> SlotTake {
        if self.slots.is_empty() {
            return SlotTake::Empty;
        }
        if let Some((index, keep)) = self.reusable(ids, key, min_similarity_milli) {
            return SlotTake::Reused { slot: self.slots.remove(index), keep };
        }
        if let Some(index) = self.slots.iter().position(|slot| slot.key != *key) {
            return SlotTake::Evicted(self.slots.remove(index));
        }
        if self.slots.len() < self.capacity {
            return SlotTake::Empty;
        }
        SlotTake::Evicted(self.slots.remove(0))
    }

    fn reusable(&self, ids: &[u32], key: &CacheKey, min_similarity_milli: u32) -> Option<(usize, usize)> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.key == *key)
            .map(|(index, slot)| (index, slot.ids.len(), longest_common_prefix(&slot.ids, ids)))
            .max_by_key(|(_, _, shared)| *shared)
            .filter(|(_, held, shared)| entry_is_reusable(*shared, *held, ids.len(), min_similarity_milli))
            .map(|(index, _, shared)| (index, shared.min(ids.len().saturating_sub(1))))
            .filter(|(_, keep)| *keep > 0)
    }

    /// Takes `slot` in and returns the one slot the table gives up, if any: the least recently put
    /// when the table holds more than `capacity` slots or more than `budget_bytes` of device KV.
    /// When giving up the oldest alone cannot restore both limits (the new slot is larger than the
    /// budget), the new slot is the one returned, so a single slot is always enough.
    pub(super) fn put(&mut self, slot: ResidentSlot) -> Option<ResidentSlot> {
        self.slots.push(slot);
        if !self.over_limits() {
            return None;
        }
        let oldest = self.slots.remove(0);
        if !self.over_limits() {
            return Some(oldest);
        }
        self.slots.insert(0, oldest);
        self.slots.pop()
    }

    fn over_limits(&self) -> bool {
        self.slots.len() > self.capacity
            || self.slots.iter().map(ResidentSlot::device_bytes).sum::<usize>() > self.budget_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::super::device_kv::tests::grown_slot;
    use super::*;

    const PROMPT_LEN: usize = 4069;
    const CHANGED_TAIL: usize = 12;
    const MIN_SIMILARITY_MILLI: u32 = 500;

    fn prompt_ids() -> Vec<u32> {
        (0..PROMPT_LEN as u32).map(|position| position.wrapping_mul(2_654_435_761) % 262_144).collect()
    }

    fn with_tail_changed(ids: &[u32]) -> Vec<u32> {
        let mut changed = ids.to_vec();
        changed.iter_mut().rev().take(CHANGED_TAIL).for_each(|id| *id += 1);
        changed
    }

    fn key() -> CacheKey {
        CacheKey::of(&ServingConfig::default(), false, RopeScaling::None, 0, 0)
    }

    fn other_key() -> CacheKey {
        CacheKey::of(&ServingConfig::default(), true, RopeScaling::None, 0, 0)
    }

    fn slot_of(ids: &[u32], key: CacheKey) -> ResidentSlot {
        let mut slot = grown_slot(0);
        slot.ids = ids.to_vec();
        slot.key = key;
        slot
    }

    /// Label, keep, and the positions the slot handed back holds.
    type Described = (&'static str, usize, usize);

    struct Case {
        name: &'static str,
        capacity: usize,
        ids: Vec<u32>,
        key: CacheKey,
        expected: Described,
        left: usize,
    }

    fn described(take: &SlotTake) -> Described {
        match take {
            SlotTake::Empty => ("empty", 0, 0),
            SlotTake::Reused { slot, keep } => ("reused", *keep, slot.ids.len()),
            SlotTake::Evicted(slot) => ("evicted", 0, slot.ids.len()),
        }
    }

    #[test]
    fn take_decides_each_shape_of_the_next_request_against_the_resident_prompt() {
        let prompt = prompt_ids();
        let unrelated: Vec<u32> = prompt.iter().map(|id| id + 1).collect();
        let cases = [
            Case { name: "identical repeat re-evaluates one token", capacity: 1, ids: prompt.clone(), key: key(), expected: ("reused", PROMPT_LEN - 1, PROMPT_LEN), left: 0 },
            Case { name: "last 12 ids changed keeps the rest", capacity: 1, ids: with_tail_changed(&prompt), key: key(), expected: ("reused", PROMPT_LEN - CHANGED_TAIL, PROMPT_LEN), left: 0 },
            Case { name: "a different key never shares rows", capacity: 1, ids: prompt.clone(), key: other_key(), expected: ("evicted", 0, PROMPT_LEN), left: 0 },
            Case { name: "no shared prefix on a full table gives up the slot", capacity: 1, ids: unrelated.clone(), key: key(), expected: ("evicted", 0, PROMPT_LEN), left: 0 },
            Case { name: "no shared prefix with room to spare leaves the slot resident", capacity: 2, ids: unrelated, key: key(), expected: ("empty", 0, 0), left: 1 },
            Case { name: "a different key is evicted even with room to spare", capacity: 2, ids: prompt.clone(), key: other_key(), expected: ("evicted", 0, PROMPT_LEN), left: 0 },
        ];
        let mut run = 0;
        for Case { name, capacity, ids, key: request_key, expected, left } in cases {
            let mut table = ResidentSlots::new(capacity, usize::MAX);
            assert!(table.put(slot_of(&prompt, key())).is_none(), "{name}: one slot fits");
            let take = table.take(&ids, &request_key, MIN_SIMILARITY_MILLI);
            assert_eq!(described(&take), expected, "{name}");
            assert_eq!(table.len(), left, "{name}: slots resident afterwards");
            run += 1;
        }
        assert_eq!(run, 6, "every case ran");
    }

    #[test]
    fn take_on_an_empty_table_is_empty() {
        let mut table = ResidentSlots::new(1, usize::MAX);
        let take = table.take(&prompt_ids(), &key(), MIN_SIMILARITY_MILLI);
        assert_eq!(described(&take), ("empty", 0, 0));
    }

    #[test]
    fn put_past_capacity_gives_up_the_least_recently_put_slot() {
        let prompt = prompt_ids();
        let second = with_tail_changed(&prompt);
        let third: Vec<u32> = prompt.iter().map(|id| id + 2).collect();
        let mut table = ResidentSlots::new(2, usize::MAX);
        assert!(table.put(slot_of(&prompt, key())).is_none());
        assert!(table.put(slot_of(&second, key())).is_none());
        let given_up = table.put(slot_of(&third, key())).expect("a third slot overflows a table of two");
        assert_eq!(given_up.ids, prompt, "the first put goes first");
        assert_eq!(table.len(), 2, "the table is back within capacity");
    }

    #[test]
    fn put_past_the_byte_budget_gives_up_a_slot() {
        let prompt = prompt_ids();
        let second = with_tail_changed(&prompt);
        let one_slot = slot_of(&prompt, key()).device_bytes();
        assert!(one_slot > 0, "the fixture holds device rows to count");

        let mut table = ResidentSlots::new(4, one_slot + one_slot / 2);
        assert!(table.put(slot_of(&prompt, key())).is_none(), "one slot is inside the budget");
        let given_up = table.put(slot_of(&second, key())).expect("two slots are over a budget of one and a half");
        assert_eq!(given_up.ids, prompt, "the least recently put goes first");
        assert_eq!(table.len(), 1);

        let mut tight = ResidentSlots::new(4, one_slot - 1);
        let refused = tight.put(slot_of(&prompt, key())).expect("a slot over the whole budget cannot stay");
        assert_eq!(refused.ids, prompt, "the slot just put is the one returned");
        assert_eq!(tight.len(), 0);
    }
}
