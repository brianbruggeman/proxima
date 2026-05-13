//! Copy-on-write atomic-root-swap crash-consistent update state machine.
//!
//! The region is laid out as `[root: u64][slot0][slot1]`, two equal-size slots
//! plus a single 8-byte aligned root word selecting the live slot (`0` or `1`).
//! An update writes the new value into the currently *dead* slot, persists it,
//! flips the root with one 8-byte atomic store, then persists the root. After a
//! crash, [`CowRoot::recover`] reads only the root and returns the slot it
//! selects — no log, no replay.
//!
//! Why this design: the persistence-reordering crash oracle is the simplest to
//! prove of the candidate techniques (undo-log, redo-log, CoW). `recover` reads
//! one 8-byte word that is power-fail atomic (SNIA/Intel ADR — the same
//! guarantee LMDB uses for its meta-page txnid and ZFS for its uberblock), so
//! the root's crash subset is only `{old, new}`, never torn. The dead slot's
//! partial-persistence states all collapse to one outcome (old) because
//! `recover` never reads the dead slot while the root still points at the live
//! one. The oracle's search space is therefore O(1) in payload size. The full
//! tournament is recorded in `docs/pmem/discipline.md`. The exhaustive oracle is
//! in this module's tests.
//!
//! **Caller precondition for atomicity:** the region's base address must be
//! 8-byte aligned so the root store is a single aligned 8-byte write. Slots may
//! be any length: a torn dead-slot write is harmless because `recover` ignores
//! the dead slot and the root only flips after [`CowRoot`]'s `persist_dead` step.

use crate::pmem::error::PmemError;

/// The root word width: a `u64` slot selector.
pub const ROOT_LEN: usize = 8;

/// A double-buffer with an atomic root selector over a borrowed pmem region.
/// Construct once for a given slot size; drive updates with [`CowRoot::commit`]
/// (or the explicit [`CowRoot::step`] state machine), read with
/// [`CowRoot::read`], recover after a crash with [`CowRoot::recover`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CowRoot {
    slot_len: usize,
}

/// The update state machine. Each `step` consumes the current state and produces
/// the next; the transitions mirror the persistence protocol exactly:
/// `Idle` --write dead slot--> `DeadSlotWritten` --persist (B1)-->
/// `DeadSlotPersisted` --flip root--> `RootFlipped` --persist (B2)-->
/// `Committed`. The two persists are the only ordering barriers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum UpdateState {
    /// nothing written yet; the live slot still holds the committed value
    Idle,
    /// the new value is in the dead slot, not yet persisted
    DeadSlotWritten,
    /// barrier B1 done: the dead slot is durable; safe to flip the root
    DeadSlotPersisted,
    /// the root now selects the new slot, not yet persisted
    RootFlipped,
    /// barrier B2 done: the root is durable; the update is committed
    Committed,
}

/// A validated update plan: which slot to write and the new value. Built by
/// [`CowRoot::prepare`] so [`CowRoot::step`] can stay infallible on the hot path.
#[derive(Debug, Clone, Copy)]
pub struct Commit<'value> {
    dead_index: u64,
    new_value: &'value [u8],
}

impl CowRoot {
    /// Construct a layout for `slot_len`-byte values. Errors on a zero length.
    pub fn new(slot_len: usize) -> Result<Self, PmemError> {
        if slot_len == 0 {
            return Err(PmemError::ZeroSlotLen);
        }
        Ok(Self { slot_len })
    }

    /// The per-value slot length.
    #[must_use]
    pub fn slot_len(&self) -> usize {
        self.slot_len
    }

    /// Bytes the region must hold: the root word plus both slots.
    #[must_use]
    pub fn region_len(&self) -> usize {
        ROOT_LEN + 2 * self.slot_len
    }

    fn ensure_region(&self, region: &[u8]) -> Result<(), PmemError> {
        let need = self.region_len();
        if region.len() < need {
            return Err(PmemError::RegionTooSmall {
                need,
                got: region.len(),
            });
        }
        Ok(())
    }

    fn slot_offset(&self, index: u64) -> usize {
        ROOT_LEN + (index as usize) * self.slot_len
    }

    /// The live slot index per the root word: `0` or `1`. An out-of-range root
    /// (only reachable from a corrupt region) is treated as `0`.
    #[must_use]
    pub fn live_index(&self, region: &[u8]) -> u64 {
        let mut raw = [0u8; ROOT_LEN];
        raw.copy_from_slice(&region[..ROOT_LEN]);
        let root = u64::from_le_bytes(raw);
        if root <= 1 { root } else { 0 }
    }

    /// Read the live value. Steady-state read; the borrow lives as long as the
    /// region.
    pub fn read<'region>(&self, region: &'region [u8]) -> Result<&'region [u8], PmemError> {
        self.ensure_region(region)?;
        let offset = self.slot_offset(self.live_index(region));
        Ok(&region[offset..offset + self.slot_len])
    }

    /// Recover after a crash. Identical to [`CowRoot::read`]: recovery is just
    /// reading the atomic root and returning the slot it selects — there is no
    /// log to replay. Kept as a distinct name so the call site documents intent.
    pub fn recover<'region>(&self, region: &'region [u8]) -> Result<&'region [u8], PmemError> {
        self.read(region)
    }

    /// Initialise a fresh region: write `initial` into slot 0, persist it, set
    /// the root to 0, persist the root.
    pub fn init<P: Fn(&[u8]) + ?Sized>(
        &self,
        region: &mut [u8],
        initial: &[u8],
        persist_fn: &P,
    ) -> Result<(), PmemError> {
        self.ensure_region(region)?;
        if initial.len() != self.slot_len {
            return Err(PmemError::SlotLenMismatch {
                expected: self.slot_len,
                got: initial.len(),
            });
        }
        let offset = self.slot_offset(0);
        region[offset..offset + self.slot_len].copy_from_slice(initial);
        persist_fn(&region[offset..offset + self.slot_len]);
        region[..ROOT_LEN].copy_from_slice(&0u64.to_le_bytes());
        persist_fn(&region[..ROOT_LEN]);
        Ok(())
    }

    /// Validate the region and value and choose the dead slot, producing a plan
    /// for [`CowRoot::step`].
    pub fn prepare<'value>(
        &self,
        region: &[u8],
        new_value: &'value [u8],
    ) -> Result<Commit<'value>, PmemError> {
        self.ensure_region(region)?;
        if new_value.len() != self.slot_len {
            return Err(PmemError::SlotLenMismatch {
                expected: self.slot_len,
                got: new_value.len(),
            });
        }
        let dead_index = 1 - self.live_index(region);
        Ok(Commit {
            dead_index,
            new_value,
        })
    }

    /// Advance the update state machine one transition. Infallible: `plan` was
    /// validated by [`CowRoot::prepare`], so every index is in bounds.
    #[must_use]
    pub fn step<P: Fn(&[u8]) + ?Sized>(
        &self,
        region: &mut [u8],
        plan: &Commit<'_>,
        state: UpdateState,
        persist_fn: &P,
    ) -> UpdateState {
        match state {
            UpdateState::Idle => {
                let offset = self.slot_offset(plan.dead_index);
                region[offset..offset + self.slot_len].copy_from_slice(plan.new_value);
                UpdateState::DeadSlotWritten
            }
            UpdateState::DeadSlotWritten => {
                let offset = self.slot_offset(plan.dead_index);
                persist_fn(&region[offset..offset + self.slot_len]);
                UpdateState::DeadSlotPersisted
            }
            UpdateState::DeadSlotPersisted => {
                region[..ROOT_LEN].copy_from_slice(&plan.dead_index.to_le_bytes());
                UpdateState::RootFlipped
            }
            UpdateState::RootFlipped => {
                persist_fn(&region[..ROOT_LEN]);
                UpdateState::Committed
            }
            UpdateState::Committed => UpdateState::Committed,
        }
    }

    /// Run a full update: prepare, then drive the state machine to `Committed`.
    pub fn commit<P: Fn(&[u8]) + ?Sized>(
        &self,
        region: &mut [u8],
        new_value: &[u8],
        persist_fn: &P,
    ) -> Result<(), PmemError> {
        let plan = self.prepare(region, new_value)?;
        let mut state = UpdateState::Idle;
        while state != UpdateState::Committed {
            state = self.step(region, &plan, state, persist_fn);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use alloc::format;
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;
    use proptest::prelude::*;
    use rstest::rstest;

    fn noop_persist(_bytes: &[u8]) {}

    // the persistence-reordering crash oracle (the pmreorder analog)
    //
    // The FSM's effect on the region is a fixed sequence of stores and barriers.
    // We model that sequence as data, then at every crash point construct every
    // surviving image (durable stores plus an arbitrary subset of pending ones)
    // and assert recover() yields the complete OLD or NEW value, never torn.
    // A >8-byte store can tear at any byte (byte-granular subsets); an 8-byte
    // aligned store is power-fail atomic (whole-or-nothing).

    #[derive(Clone)]
    enum Op {
        Write {
            offset: usize,
            bytes: Vec<u8>,
            atomic: bool,
        },
        Persist {
            offset: usize,
            len: usize,
        },
    }

    #[derive(Clone)]
    struct Pending {
        offset: usize,
        bytes: Vec<u8>,
        atomic: bool,
    }

    // the correct CoW commit, as a persistence-event sequence
    fn commit_ops(layout: &CowRoot, region: &[u8], new_value: &[u8]) -> Vec<Op> {
        let dead = 1 - layout.live_index(region);
        let dead_off = layout.slot_offset(dead);
        vec![
            Op::Write {
                offset: dead_off,
                bytes: new_value.to_vec(),
                atomic: false,
            },
            Op::Persist {
                offset: dead_off,
                len: layout.slot_len(),
            },
            Op::Write {
                offset: 0,
                bytes: dead.to_le_bytes().to_vec(),
                atomic: true,
            },
            Op::Persist {
                offset: 0,
                len: ROOT_LEN,
            },
        ]
    }

    // a BROKEN commit that flips the root before persisting the dead slot — the
    // missing-B1 bug the oracle must catch.
    fn commit_ops_missing_b1(layout: &CowRoot, region: &[u8], new_value: &[u8]) -> Vec<Op> {
        let dead = 1 - layout.live_index(region);
        let dead_off = layout.slot_offset(dead);
        vec![
            Op::Write {
                offset: dead_off,
                bytes: new_value.to_vec(),
                atomic: false,
            },
            Op::Write {
                offset: 0,
                bytes: dead.to_le_bytes().to_vec(),
                atomic: true,
            },
            Op::Persist {
                offset: 0,
                len: ROOT_LEN,
            },
            Op::Persist {
                offset: dead_off,
                len: layout.slot_len(),
            },
        ]
    }

    fn state_at(ops: &[Op], initial: &[u8], crash_after: usize) -> (Vec<u8>, Vec<Pending>) {
        let mut durable = initial.to_vec();
        let mut pending: Vec<Pending> = Vec::new();
        for op in &ops[..crash_after] {
            match op {
                Op::Write {
                    offset,
                    bytes,
                    atomic,
                } => {
                    pending.push(Pending {
                        offset: *offset,
                        bytes: bytes.clone(),
                        atomic: *atomic,
                    });
                }
                Op::Persist { offset, len } => {
                    pending.retain(|pending_write| {
                        let within = pending_write.offset >= *offset
                            && pending_write.offset + pending_write.bytes.len() <= offset + len;
                        if within {
                            for (index, byte) in pending_write.bytes.iter().enumerate() {
                                durable[pending_write.offset + index] = *byte;
                            }
                        }
                        !within
                    });
                }
            }
        }
        (durable, pending)
    }

    // every way a single pending store can survive a crash
    fn write_choices(pending_write: &Pending) -> Vec<Vec<(usize, u8)>> {
        if pending_write.atomic {
            let whole: Vec<(usize, u8)> = pending_write
                .bytes
                .iter()
                .enumerate()
                .map(|(index, byte)| (pending_write.offset + index, *byte))
                .collect();
            return vec![Vec::new(), whole];
        }
        let count = pending_write.bytes.len();
        assert!(
            count <= 16,
            "oracle enumerates byte subsets; keep tearable stores small in tests"
        );
        let mut choices = Vec::new();
        for mask in 0u32..(1u32 << count) {
            let mut subset = Vec::new();
            for index in 0..count {
                if mask & (1u32 << index) != 0 {
                    subset.push((pending_write.offset + index, pending_write.bytes[index]));
                }
            }
            choices.push(subset);
        }
        choices
    }

    // cartesian product of each pending store's survival choices
    fn survival_combos(pending: &[Pending]) -> Vec<Vec<(usize, u8)>> {
        let mut combos: Vec<Vec<(usize, u8)>> = vec![Vec::new()];
        for pending_write in pending {
            let choices = write_choices(pending_write);
            let mut next = Vec::new();
            for base in &combos {
                for choice in &choices {
                    let mut merged = base.clone();
                    merged.extend_from_slice(choice);
                    next.push(merged);
                }
            }
            combos = next;
        }
        combos
    }

    // returns Ok(images_checked) if every crash image recovers to OLD or NEW,
    // else Err(description of the first torn image)
    fn check_oracle(
        layout: &CowRoot,
        initial: &[u8],
        new_value: &[u8],
        ops: &[Op],
    ) -> Result<usize, String> {
        let old_value = layout.read(initial).unwrap().to_vec();
        let mut images = 0usize;
        for crash_after in 0..=ops.len() {
            let (durable, pending) = state_at(ops, initial, crash_after);
            for combo in survival_combos(&pending) {
                let mut image = durable.clone();
                for (offset, byte) in combo {
                    image[offset] = byte;
                }
                let recovered = layout.recover(&image).unwrap();
                images += 1;
                if recovered != old_value.as_slice() && recovered != new_value {
                    return Err(format!(
                        "torn at crash_after={crash_after}: recovered={recovered:02x?} \
                         is neither OLD={old_value:02x?} nor NEW={new_value:02x?}"
                    ));
                }
            }
        }
        Ok(images)
    }

    fn fresh_region(layout: &CowRoot, initial: &[u8]) -> Vec<u8> {
        let mut region = vec![0u8; layout.region_len()];
        layout.init(&mut region, initial, &noop_persist).unwrap();
        region
    }

    // locked worked example (the /algorithm-rigor winner's example)
    // region [root:u64][slot0:8][slot1:8]; OLD=0xAA..AA live in slot0, NEW=0xBB..BB

    const OLD8: [u8; 8] = [0xAA; 8];
    const NEW8: [u8; 8] = [0xBB; 8];

    #[test]
    fn worked_example_full_commit_recovers_new() {
        let layout = CowRoot::new(8).unwrap();
        let mut region = fresh_region(&layout, &OLD8);
        assert_eq!(layout.read(&region).unwrap(), &OLD8);
        layout.commit(&mut region, &NEW8, &noop_persist).unwrap();
        assert_eq!(
            layout.live_index(&region),
            1,
            "root must select the new slot"
        );
        assert_eq!(layout.recover(&region).unwrap(), &NEW8);
    }

    #[test]
    fn worked_example_exhaustive_oracle_is_always_old_or_new() {
        let layout = CowRoot::new(8).unwrap();
        let region = fresh_region(&layout, &OLD8);
        let ops = commit_ops(&layout, &region, &NEW8);
        let checked = check_oracle(&layout, &region, &NEW8, &ops).expect("no torn image");
        // 8-byte tearable dead-slot write => 256 subsets at one crash point, plus
        // the four other crash points (1 + 1 + 2 + 1 images) = 261 total.
        assert_eq!(
            checked, 261,
            "oracle must exhaustively check every crash image"
        );
    }

    // Crash-B from the worked example: the root flip is in the pending subset.
    // ADR atomicity => root is exactly OLD-index or NEW-index, recover is OLD or NEW.
    #[rstest]
    #[case::root_flip_lost(0u64, &OLD8)]
    #[case::root_flip_survived(1u64, &NEW8)]
    fn crash_b_root_flip_atomic_recovers_old_or_new(
        #[case] durable_root: u64,
        #[case] expected: &[u8],
    ) {
        let layout = CowRoot::new(8).unwrap();
        let mut region = fresh_region(&layout, &OLD8);
        // B1 already happened: the dead slot (slot1) is durably NEW
        let slot1 = ROOT_LEN + 8;
        region[slot1..slot1 + 8].copy_from_slice(&NEW8);
        // the atomic root store is in (or not in) the surviving subset
        region[..ROOT_LEN].copy_from_slice(&durable_root.to_le_bytes());
        assert_eq!(layout.recover(&region).unwrap(), expected);
    }

    // Crash-A: dead slot written (even torn) but root not yet flipped -> OLD.
    #[rstest]
    #[case::nothing_survived([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])]
    #[case::first_byte([0xBB, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])]
    #[case::last_byte([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xBB])]
    #[case::fully_written([0xBB, 0xBB, 0xBB, 0xBB, 0xBB, 0xBB, 0xBB, 0xBB])]
    fn crash_a_torn_dead_slot_with_root_unchanged_recovers_old(#[case] torn_slot1: [u8; 8]) {
        let layout = CowRoot::new(8).unwrap();
        let mut region = fresh_region(&layout, &OLD8);
        let slot1 = ROOT_LEN + 8;
        region[slot1..slot1 + 8].copy_from_slice(&torn_slot1);
        assert_eq!(
            layout.live_index(&region),
            0,
            "root still selects the live slot"
        );
        assert_eq!(
            layout.recover(&region).unwrap(),
            &OLD8,
            "dead-slot tearing is invisible"
        );
    }

    // general-payload proof: slots may be arbitrary length
    // Exhaustively oracle commits across several slot sizes and a ping-pong of
    // updates (so both slot directions are exercised).

    #[rstest]
    #[case::one_byte(1)]
    #[case::three_bytes(3)]
    #[case::eight_bytes(8)]
    #[case::twelve_bytes(12)]
    fn general_payload_oracle_holds_for_each_slot_size(#[case] slot_len: usize) {
        let layout = CowRoot::new(slot_len).unwrap();
        let old: Vec<u8> = (0..slot_len).map(|index| 0xA0 + index as u8).collect();
        let new: Vec<u8> = (0..slot_len).map(|index| 0xB0 + index as u8).collect();
        let region = fresh_region(&layout, &old);
        let ops = commit_ops(&layout, &region, &new);
        check_oracle(&layout, &region, &new, &ops)
            .unwrap_or_else(|why| panic!("slot_len={slot_len}: {why}"));
    }

    #[test]
    fn ping_pong_two_commits_alternate_slots_and_recover() {
        let layout = CowRoot::new(4).unwrap();
        let value_a = [0xA1, 0xA2, 0xA3, 0xA4];
        let value_b = [0xB1, 0xB2, 0xB3, 0xB4];
        let value_c = [0xC1, 0xC2, 0xC3, 0xC4];
        let mut region = fresh_region(&layout, &value_a);
        assert_eq!(layout.live_index(&region), 0);

        // each commit's crash window must recover to its own OLD or NEW
        let ops_ab = commit_ops(&layout, &region, &value_b);
        check_oracle(&layout, &region, &value_b, &ops_ab).expect("a->b consistent");
        layout.commit(&mut region, &value_b, &noop_persist).unwrap();
        assert_eq!(layout.live_index(&region), 1, "second slot now live");
        assert_eq!(layout.recover(&region).unwrap(), &value_b);

        let ops_bc = commit_ops(&layout, &region, &value_c);
        check_oracle(&layout, &region, &value_c, &ops_bc).expect("b->c consistent");
        layout.commit(&mut region, &value_c, &noop_persist).unwrap();
        assert_eq!(layout.live_index(&region), 0, "back to the first slot");
        assert_eq!(layout.recover(&region).unwrap(), &value_c);
    }

    // the oracle has teeth: the missing-B1 bug is caught

    #[test]
    fn oracle_catches_missing_b1_barrier() {
        let layout = CowRoot::new(8).unwrap();
        let region = fresh_region(&layout, &OLD8);
        let broken = commit_ops_missing_b1(&layout, &region, &NEW8);
        let result = check_oracle(&layout, &region, &NEW8, &broken);
        assert!(
            result.is_err(),
            "without B1, a surviving root can select a torn slot — oracle must flag it"
        );
    }

    // the op-model must match the real FSM: a no-crash replay equals commit()
    #[test]
    fn op_model_matches_real_commit() {
        let layout = CowRoot::new(8).unwrap();
        let region = fresh_region(&layout, &OLD8);
        let ops = commit_ops(&layout, &region, &NEW8);
        let (replayed, pending) = state_at(&ops, &region, ops.len());
        assert!(
            pending.is_empty(),
            "all stores persisted after a full replay"
        );
        let mut by_commit = region.clone();
        layout.commit(&mut by_commit, &NEW8, &noop_persist).unwrap();
        assert_eq!(replayed, by_commit, "oracle op-model must mirror commit()");
    }

    // sad paths

    #[test]
    fn zero_slot_len_is_rejected() {
        assert_eq!(CowRoot::new(0).unwrap_err(), PmemError::ZeroSlotLen);
    }

    #[test]
    fn region_too_small_is_rejected() {
        let layout = CowRoot::new(8).unwrap();
        let mut region = vec![0u8; layout.region_len() - 1];
        let err = layout.init(&mut region, &OLD8, &noop_persist).unwrap_err();
        assert_eq!(err, PmemError::RegionTooSmall { need: 24, got: 23 });
    }

    #[test]
    fn wrong_value_length_is_rejected() {
        let layout = CowRoot::new(8).unwrap();
        let mut region = fresh_region(&layout, &OLD8);
        let err = layout
            .commit(&mut region, &[0xBB; 4], &noop_persist)
            .unwrap_err();
        assert_eq!(
            err,
            PmemError::SlotLenMismatch {
                expected: 8,
                got: 4
            }
        );
    }

    // property-based proofs (complement the exhaustive small-case oracle)

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        // For random old/new payloads at random slot sizes, the exhaustive crash
        // oracle (every crash point x every surviving subset) recovers old-or-new.
        #[test]
        fn oracle_holds_for_random_payloads(
            (old, new) in (1usize..=8).prop_flat_map(|len| {
                (
                    proptest::collection::vec(any::<u8>(), len),
                    proptest::collection::vec(any::<u8>(), len),
                )
            })
        ) {
            let layout = CowRoot::new(old.len()).unwrap();
            let region = fresh_region(&layout, &old);
            let ops = commit_ops(&layout, &region, &new);
            let result = check_oracle(&layout, &region, &new, &ops);
            prop_assert!(result.is_ok(), "{result:?}");
        }

        // A clean commit of any value (any slot size) is read back exactly.
        #[test]
        fn commit_then_recover_round_trips(
            value in proptest::collection::vec(any::<u8>(), 1usize..=64)
        ) {
            let layout = CowRoot::new(value.len()).unwrap();
            let zero = vec![0u8; value.len()];
            let mut region = fresh_region(&layout, &zero);
            layout.commit(&mut region, &value, &noop_persist).unwrap();
            prop_assert_eq!(layout.recover(&region).unwrap(), value.as_slice());
        }

        // A sequence of commits ping-pongs the slots and always reads back the
        // most recent value; the live index alternates each commit.
        #[test]
        fn sequence_of_commits_reads_back_last_value(
            slot_len in 1usize..=16,
            values in proptest::collection::vec(any::<u8>(), 1usize..=8),
        ) {
            let layout = CowRoot::new(slot_len).unwrap();
            let mut region = fresh_region(&layout, &vec![0u8; slot_len]);
            for (step, fill) in values.iter().enumerate() {
                let value = vec![*fill; slot_len];
                layout.commit(&mut region, &value, &noop_persist).unwrap();
                prop_assert_eq!(layout.recover(&region).unwrap(), value.as_slice());
                prop_assert_eq!(layout.live_index(&region), ((step + 1) % 2) as u64);
            }
        }
    }
}
