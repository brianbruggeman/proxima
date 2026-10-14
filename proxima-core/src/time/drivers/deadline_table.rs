//! Fixed-capacity deadline table for the std timer owner: an indexed binary
//! min-heap (position stored in each entry) plus an open-addressing hash index
//! keyed by (deadline, task). Insert, remove and pop-earliest are O(log n);
//! keyed lookup is O(1) expected. All storage is reserved at construction and
//! never grows -- a full table refuses the insert and hands the entry back.

use std::task::Waker;

use crate::time::Instant;

const EMPTY: u32 = u32::MAX;

pub(super) struct Entry {
    pub(super) deadline: Instant,
    pub(super) task: usize,
    pub(super) waker: Waker,
    pub(super) refs: u32,
    heap_pos: u32,
}

impl Entry {
    pub(super) fn new(deadline: Instant, task: usize, waker: Waker) -> Self {
        Self {
            deadline,
            task,
            waker,
            refs: 1,
            heap_pos: 0,
        }
    }

    pub(super) fn rearm(&mut self, waker: Waker) {
        if !self.waker.will_wake(&waker) {
            self.waker = waker;
        }
    }

    fn order(&self) -> (Instant, usize) {
        (self.deadline, self.task)
    }
}

pub(super) struct DeadlineTable {
    slab: Vec<Option<Entry>>,
    free: Vec<u32>,
    heap: Vec<u32>,
    index: Vec<u32>,
    capacity: usize,
}

fn mix(deadline: Instant, task: usize) -> usize {
    let nanos = u64::try_from(deadline.into_monotonic().as_nanos()).unwrap_or(u64::MAX);
    let folded = nanos ^ (task as u64).rotate_left(32);
    let spread = folded.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (spread >> 32) as usize
}

impl DeadlineTable {
    pub(super) fn with_capacity(capacity: usize) -> Self {
        let buckets = (capacity * 2).next_power_of_two().max(2);
        Self {
            slab: Vec::with_capacity(capacity),
            free: Vec::with_capacity(capacity),
            heap: Vec::with_capacity(capacity),
            index: vec![EMPTY; buckets],
            capacity,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.heap.len()
    }

    pub(super) fn has_room(&self) -> bool {
        self.heap.len() < self.capacity
    }

    pub(super) fn earliest(&self) -> Option<Instant> {
        self.heap
            .first()
            .and_then(|slot| self.entry(*slot))
            .map(|entry| entry.deadline)
    }

    pub(super) fn find(&self, deadline: Instant, task: usize) -> Option<u32> {
        let mask = self.index.len() - 1;
        let mut bucket = mix(deadline, task) & mask;
        loop {
            let slot = self.index[bucket];
            if slot == EMPTY {
                return None;
            }
            if self.entry(slot).is_some_and(|entry| entry.order() == (deadline, task)) {
                return Some(slot);
            }
            bucket = (bucket + 1) & mask;
        }
    }

    pub(super) fn entry_mut(&mut self, slot: u32) -> Option<&mut Entry> {
        self.slab.get_mut(slot as usize).and_then(Option::as_mut)
    }

    pub(super) fn insert(&mut self, entry: Entry) -> Result<(), Entry> {
        if !self.has_room() {
            return Err(entry);
        }
        let (deadline, task) = (entry.deadline, entry.task);
        let slot = match self.free.pop() {
            Some(slot) => {
                self.slab[slot as usize] = Some(entry);
                slot
            }
            None => {
                self.slab.push(Some(entry));
                (self.slab.len() - 1) as u32
            }
        };
        self.index_insert(slot, deadline, task);
        self.heap.push(slot);
        self.set_pos(self.heap.len() - 1);
        self.sift_up(self.heap.len() - 1);
        Ok(())
    }

    pub(super) fn remove(&mut self, slot: u32) -> Option<Entry> {
        let entry = self.slab.get_mut(slot as usize)?.take()?;
        self.index_remove(slot, entry.deadline, entry.task);
        let position = entry.heap_pos as usize;
        let last = self.heap.len() - 1;
        self.heap.swap(position, last);
        self.heap.pop();
        if position < self.heap.len() {
            self.set_pos(position);
            self.sift_down(position);
            self.sift_up(position);
        }
        self.free.push(slot);
        Some(entry)
    }

    pub(super) fn pop_earliest(&mut self) -> Option<Entry> {
        let slot = *self.heap.first()?;
        self.remove(slot)
    }

    fn entry(&self, slot: u32) -> Option<&Entry> {
        self.slab.get(slot as usize).and_then(Option::as_ref)
    }

    fn key_of(&self, heap_position: usize) -> (Instant, usize) {
        self.entry(self.heap[heap_position])
            .map_or((Instant::from_monotonic(core::time::Duration::ZERO), 0), Entry::order)
    }

    fn set_pos(&mut self, heap_position: usize) {
        let slot = self.heap[heap_position];
        if let Some(entry) = self.entry_mut(slot) {
            entry.heap_pos = heap_position as u32;
        }
    }

    fn swap_heap(&mut self, first: usize, second: usize) {
        self.heap.swap(first, second);
        self.set_pos(first);
        self.set_pos(second);
    }

    fn sift_up(&mut self, mut position: usize) {
        while position > 0 {
            let parent = (position - 1) / 2;
            if self.key_of(position) >= self.key_of(parent) {
                break;
            }
            self.swap_heap(position, parent);
            position = parent;
        }
    }

    fn sift_down(&mut self, mut position: usize) {
        loop {
            let left = position * 2 + 1;
            if left >= self.heap.len() {
                break;
            }
            let right = left + 1;
            let child = if right < self.heap.len() && self.key_of(right) < self.key_of(left) {
                right
            } else {
                left
            };
            if self.key_of(child) >= self.key_of(position) {
                break;
            }
            self.swap_heap(position, child);
            position = child;
        }
    }

    fn index_insert(&mut self, slot: u32, deadline: Instant, task: usize) {
        let mask = self.index.len() - 1;
        let mut bucket = mix(deadline, task) & mask;
        while self.index[bucket] != EMPTY {
            bucket = (bucket + 1) & mask;
        }
        self.index[bucket] = slot;
    }

    // backward-shift deletion: no tombstones, so the index never degrades or rehashes.
    fn index_remove(&mut self, slot: u32, deadline: Instant, task: usize) {
        let mask = self.index.len() - 1;
        let mut hole = mix(deadline, task) & mask;
        while self.index[hole] != slot {
            hole = (hole + 1) & mask;
        }
        let mut probe = (hole + 1) & mask;
        while self.index[probe] != EMPTY {
            let moved = self.index[probe];
            let ideal = self
                .entry(moved)
                .map_or(probe, |entry| mix(entry.deadline, entry.task) & mask);
            if (probe.wrapping_sub(ideal) & mask) >= (probe.wrapping_sub(hole) & mask) {
                self.index[hole] = moved;
                hole = probe;
            }
            probe = (probe + 1) & mask;
        }
        self.index[hole] = EMPTY;
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn entry(millis: u64, task: usize) -> Entry {
        let deadline = Instant::from_monotonic(Duration::from_millis(millis));
        Entry::new(deadline, task, Waker::noop().clone())
    }

    fn lcg(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        *state >> 33
    }

    #[test]
    fn pops_in_deadline_order_after_random_inserts_and_removals() {
        let mut table = DeadlineTable::with_capacity(512);
        let mut state = 7_u64;
        let mut keys = Vec::new();
        for task in 0..400 {
            let millis = lcg(&mut state) % 10_000;
            assert!(table.insert(entry(millis, task)).is_ok());
            keys.push((millis, task));
        }
        for (millis, task) in keys.iter().step_by(3) {
            let deadline = Instant::from_monotonic(Duration::from_millis(*millis));
            let slot = table.find(deadline, *task).expect("inserted key is indexed");
            assert!(table.remove(slot).is_some());
        }
        let mut popped = Vec::new();
        while let Some(entry) = table.pop_earliest() {
            popped.push((entry.deadline, entry.task));
        }

        assert_eq!(popped.len(), 400 - keys.iter().step_by(3).count());
        assert!(popped.windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!(table.len(), 0);
    }

    #[test]
    fn full_table_hands_the_entry_back() {
        let mut table = DeadlineTable::with_capacity(2);
        assert!(table.insert(entry(1, 1)).is_ok());
        assert!(table.insert(entry(2, 2)).is_ok());

        let refused = table.insert(entry(3, 3)).expect_err("table is full");

        assert_eq!(refused.task, 3);
        assert_eq!(table.len(), 2);
    }
}
