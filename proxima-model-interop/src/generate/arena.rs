//! A capacity-bounded arena: items in one `Vec`, named by `u32` handles, with
//! a free list so a removed slot is the next one an insert fills
//! (`proxima-tensor/specs/prefix-cache-reuse/SPEC.md` R13).
//!
//! No `Box`, no `Rc`, no allocation per item: the `Vec` grows to the
//! high-water mark of live items and a churn of inserts and removes below it
//! never touches the allocator. It is the storage under
//! [`super::prefix_trie::PrefixTrie`]; reach for a plain `Vec` when nothing
//! removes, and for `proxima_core`'s `ByteArena` when the data is bytes that
//! are only ever appended. Neither frees a slot for reuse, which is the whole
//! reason this one exists.
//!
//! The shape is `no_std + alloc` clean (`Vec`, `NonZeroU32`, nothing from
//! `std`); it sits under the `std`-gated `generate` module only because its
//! one consumer does.

use core::fmt;
use core::marker::PhantomData;
use core::mem::replace;
use core::num::NonZeroU32;

/// The name of one live item. `Option<Handle<_>>` is four bytes: the zero
/// value is the `None`.
pub(super) struct Handle<Item> {
    slot: NonZeroU32,
    item: PhantomData<fn() -> Item>,
}

impl<Item> Handle<Item> {
    fn from_index(index: usize) -> Option<Self> {
        let slot = u32::try_from(index).ok()?.checked_add(1)?;
        Some(Self {
            slot: NonZeroU32::new(slot)?,
            item: PhantomData,
        })
    }

    fn index(self) -> usize {
        (self.slot.get() - 1) as usize
    }

    pub(super) const fn get(self) -> u32 {
        self.slot.get()
    }
}

impl<Item> Clone for Handle<Item> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Item> Copy for Handle<Item> {}

impl<Item> PartialEq for Handle<Item> {
    fn eq(&self, other: &Self) -> bool {
        self.slot == other.slot
    }
}

impl<Item> Eq for Handle<Item> {}

impl<Item> fmt::Debug for Handle<Item> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Handle({})", self.slot)
    }
}

enum Slot<Item> {
    Occupied(Item),
    Vacant { next_free: Option<Handle<Item>> },
}

/// An insert met an arena already holding its capacity of live items.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("arena is full at {capacity} live items")]
pub(super) struct ArenaFull {
    pub(super) capacity: usize,
}

/// At most `capacity` live items. A full arena refuses the insert; what to do
/// then (evict, refuse the work) is the owner's policy, not the arena's.
pub(super) struct Arena<Item> {
    slots: Vec<Slot<Item>>,
    free_head: Option<Handle<Item>>,
    live: usize,
    capacity: usize,
}

impl<Item> Arena<Item> {
    pub(super) const fn new(capacity: usize) -> Self {
        Self {
            slots: Vec::new(),
            free_head: None,
            live: 0,
            capacity,
        }
    }

    pub(super) const fn len(&self) -> usize {
        self.live
    }

    pub(super) const fn capacity(&self) -> usize {
        self.capacity
    }

    pub(super) const fn available(&self) -> usize {
        self.capacity - self.live
    }

    /// Bytes the slot storage holds, live or vacant.
    pub(super) fn byte_len(&self) -> usize {
        self.slots.capacity() * size_of::<Slot<Item>>()
    }

    pub(super) fn insert(&mut self, item: Item) -> Result<Handle<Item>, ArenaFull> {
        let full = ArenaFull {
            capacity: self.capacity,
        };
        if self.live >= self.capacity {
            return Err(full);
        }
        let handle = match self.free_head {
            Some(vacant) => {
                let slot = self.slots.get_mut(vacant.index()).ok_or(full)?;
                let Slot::Vacant { next_free } = replace(slot, Slot::Occupied(item)) else {
                    return Err(full);
                };
                self.free_head = next_free;
                vacant
            }
            None => {
                let handle = Handle::from_index(self.slots.len()).ok_or(full)?;
                self.slots.push(Slot::Occupied(item));
                handle
            }
        };
        self.live += 1;
        Ok(handle)
    }

    /// The item, or `None` for a handle whose item was removed.
    pub(super) fn remove(&mut self, handle: Handle<Item>) -> Option<Item> {
        let slot = self.slots.get_mut(handle.index())?;
        if matches!(slot, Slot::Vacant { .. }) {
            return None;
        }
        let vacated = Slot::Vacant {
            next_free: self.free_head,
        };
        let Slot::Occupied(item) = replace(slot, vacated) else {
            return None;
        };
        self.free_head = Some(handle);
        self.live -= 1;
        Some(item)
    }

    pub(super) fn get(&self, handle: Handle<Item>) -> Option<&Item> {
        match self.slots.get(handle.index())? {
            Slot::Occupied(item) => Some(item),
            Slot::Vacant { .. } => None,
        }
    }

    pub(super) fn get_mut(&mut self, handle: Handle<Item>) -> Option<&mut Item> {
        match self.slots.get_mut(handle.index())? {
            Slot::Occupied(item) => Some(item),
            Slot::Vacant { .. } => None,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::HashMap;

    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest::test_runner::{Config, TestRunner};

    use super::*;
    use crate::generate::alloc_probe::allocations_during;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Stamped {
        stamp: u64,
    }

    fn filled(capacity: usize) -> (Arena<Stamped>, Vec<Handle<Stamped>>) {
        let mut arena = Arena::new(capacity);
        let handles = (0..capacity as u64)
            .map(|stamp| arena.insert(Stamped { stamp }).expect("within capacity"))
            .collect();
        (arena, handles)
    }

    #[test]
    fn an_inserted_item_is_read_back_through_its_handle() {
        let (arena, handles) = filled(4);

        let stamps: Vec<u64> = handles
            .iter()
            .map(|handle| arena.get(*handle).expect("live").stamp)
            .collect();

        assert_eq!(stamps, vec![0, 1, 2, 3]);
        assert_eq!(arena.len(), 4);
    }

    #[test]
    fn an_insert_into_a_full_arena_is_refused_until_a_slot_is_freed() {
        let (mut arena, handles) = filled(3);

        assert_eq!((arena.capacity(), arena.available()), (3, 0));
        let refused = arena.insert(Stamped { stamp: 9 });
        let removed = arena.remove(handles[1]);
        let accepted = arena.insert(Stamped { stamp: 9 });

        assert_eq!(refused.unwrap_err(), ArenaFull { capacity: 3 });
        assert_eq!(removed, Some(Stamped { stamp: 1 }));
        assert_eq!(accepted.expect("a slot was freed"), handles[1]);
        assert_eq!(arena.available(), 0);
        assert_eq!(arena.get(handles[1]), Some(&Stamped { stamp: 9 }));
    }

    #[test]
    fn a_removed_handle_reads_as_absent_and_cannot_be_removed_twice() {
        let (mut arena, handles) = filled(2);

        arena.remove(handles[0]);

        assert!(arena.get(handles[0]).is_none());
        assert!(arena.get_mut(handles[0]).is_none());
        assert!(arena.remove(handles[0]).is_none());
        assert_eq!(arena.len(), 1);
    }

    #[test]
    fn freed_slots_are_reused_last_freed_first_without_growing_the_storage() {
        let (mut arena, handles) = filled(8);
        let storage = arena.byte_len();

        arena.remove(handles[2]);
        arena.remove(handles[5]);
        let first = arena.insert(Stamped { stamp: 20 }).expect("room");
        let second = arena.insert(Stamped { stamp: 21 }).expect("room");

        assert_eq!((first, second), (handles[5], handles[2]));
        assert_eq!(arena.byte_len(), storage);
    }

    #[test]
    fn a_churn_below_the_high_water_mark_never_reaches_the_allocator() {
        let (mut arena, handles) = filled(64);
        handles.iter().skip(8).for_each(|handle| {
            arena.remove(*handle);
        });

        let allocations = allocations_during(|| {
            for round in 0..100_000_u64 {
                let handle = arena.insert(Stamped { stamp: round }).expect("room");
                arena.remove(handle);
            }
        });

        assert_eq!(allocations, 0);
        assert_eq!(arena.len(), 8);
    }

    #[test]
    fn the_arena_agrees_with_a_map_over_10000_generated_operation_sequences() {
        const SEQUENCES: usize = 10_000;
        const CAPACITY: usize = 6;
        let mut runner = TestRunner::new(Config {
            cases: SEQUENCES as u32,
            failure_persistence: None,
            ..Config::default()
        });
        let executed = std::cell::Cell::new(0_usize);

        runner
            .run(&vec((any::<bool>(), 0_usize..CAPACITY), 0..40), |ops| {
                let mut arena: Arena<Stamped> = Arena::new(CAPACITY);
                let mut model: HashMap<u32, u64> = HashMap::new();
                let mut handles: Vec<Handle<Stamped>> = Vec::new();
                for (step, (insert, pick)) in ops.into_iter().enumerate() {
                    let stamp = step as u64;
                    if insert {
                        match arena.insert(Stamped { stamp }) {
                            Ok(handle) => {
                                assert!(model.insert(handle.get(), stamp).is_none());
                                handles.push(handle);
                            }
                            Err(_) => assert_eq!(model.len(), CAPACITY),
                        }
                    } else if !handles.is_empty() {
                        let handle = handles.swap_remove(pick % handles.len());
                        let expected = model.remove(&handle.get());
                        assert_eq!(arena.remove(handle).map(|item| item.stamp), expected);
                    }
                    assert_eq!(arena.len(), model.len());
                }
                for handle in &handles {
                    assert_eq!(
                        arena.get(*handle).map(|item| item.stamp),
                        model.get(&handle.get()).copied()
                    );
                }
                executed.set(executed.get() + 1);
                Ok(())
            })
            .expect("the arena must agree with the map on every sequence");

        assert_eq!(executed.get(), SEQUENCES);
    }
}
