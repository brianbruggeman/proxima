//! [`PmemCowStore`]: a durable crash-consistent cell, the leaf's first consumer.
//!
//! Composes [`crate::pmem::CowRoot`] over a [`MappedRegion`], wiring the
//! region's durability (`persist` = the leaf's cache-line flush, plus `msync`
//! for a file-backed mapping) into the FSM's two barriers.

use core::cell::Cell;
use core::slice;
use std::path::Path;

use rustix::mm::{MsyncFlags, msync};

use crate::dax::config::DaxConfig;
use crate::dax::config::PersistMode;
use crate::dax::error::DaxError;
use crate::dax::region::MappedRegion;
use crate::pmem::{CowRoot, PmemError, persist as leaf_persist};

/// A persistent crash-consistent cell holding one `slot_len`-byte value.
pub struct PmemCowStore {
    region: MappedRegion,
    cow: CowRoot,
}

impl PmemCowStore {
    /// Create and initialise a fresh store at `path`, writing `initial` as the
    /// first committed value. The config-free primitive; [`Self::create`] wraps
    /// it for the [`DaxConfig`] surface.
    pub fn create_at(
        path: &Path,
        slot_len: usize,
        mode: PersistMode,
        initial: &[u8],
    ) -> Result<Self, DaxError> {
        let cow = CowRoot::new(slot_len)?;
        let region = MappedRegion::open(path, cow.region_len(), mode)?;
        let mut store = Self { region, cow };
        store.run(|cow, slice, persist| cow.init(slice, initial, persist))?;
        Ok(store)
    }

    /// Open an existing store at `path`. Recovery is implicit: the next
    /// [`Self::read`] / [`Self::recover`] reads the atomic root.
    pub fn open_at(path: &Path, slot_len: usize, mode: PersistMode) -> Result<Self, DaxError> {
        let cow = CowRoot::new(slot_len)?;
        let region = MappedRegion::open(path, cow.region_len(), mode)?;
        Ok(Self { region, cow })
    }

    /// Create and initialise a fresh store from a [`DaxConfig`].
    pub fn create(config: &DaxConfig, initial: &[u8]) -> Result<Self, DaxError> {
        Self::create_at(
            &config.path_buf(),
            config.slot_len,
            config.persist_mode,
            initial,
        )
    }

    /// Open an existing store from a [`DaxConfig`].
    pub fn open(config: &DaxConfig) -> Result<Self, DaxError> {
        Self::open_at(&config.path_buf(), config.slot_len, config.persist_mode)
    }

    /// Commit a new value, crash-consistently. After a crash mid-commit, a
    /// later [`Self::recover`] returns either this value or the prior one.
    pub fn commit(&mut self, value: &[u8]) -> Result<(), DaxError> {
        self.run(|cow, slice, persist| cow.commit(slice, value, persist))
    }

    /// Read the live value (steady state).
    pub fn read(&self) -> Result<&[u8], DaxError> {
        Ok(self.cow.read(self.region.as_slice())?)
    }

    /// Recover the live value after a crash — reads the atomic root, no replay.
    pub fn recover(&self) -> Result<&[u8], DaxError> {
        Ok(self.cow.recover(self.region.as_slice())?)
    }

    /// The configured slot length.
    #[must_use]
    pub fn slot_len(&self) -> usize {
        self.cow.slot_len()
    }

    // Run a leaf op with the region's persist wired in. The persist closure
    // aliases the mapping through a raw pointer (the standard mmap idiom: the
    // syscall reads kernel-visible memory, not a Rust reference), and surfaces an
    // msync failure via `last_err` so the FSM's ordering is preserved while the
    // error still propagates.
    fn run<Op>(&mut self, op: Op) -> Result<(), DaxError>
    where
        Op: FnOnce(&CowRoot, &mut [u8], &dyn Fn(&[u8])) -> Result<(), PmemError>,
    {
        let (base, len, mode) = self.region.raw();
        let last_err: Cell<Option<rustix::io::Errno>> = Cell::new(None);
        let persist = |bytes: &[u8]| {
            leaf_persist(bytes);
            if mode == PersistMode::FileBacked {
                // SAFETY: base..len is the live page-aligned mapping.
                if let Err(err) = unsafe { msync(base.cast(), len, MsyncFlags::SYNC) } {
                    last_err.set(Some(err));
                }
            }
        };
        // SAFETY: base..len is the live mapping; &mut self makes this the only
        // outstanding mutable view for the duration of the op.
        let slice = unsafe { slice::from_raw_parts_mut(base, len) };
        op(&self.cow, slice, &persist)?;
        if let Some(err) = last_err.get() {
            return Err(DaxError::Msync(err));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use tempfile::tempdir;

    const MODE: PersistMode = PersistMode::FileBacked;

    // The end-to-end durability proof the leaf could not give on its own:
    // create -> commit -> DROP (munmap) -> reopen -> recover the committed value.
    #[test]
    fn commit_survives_drop_and_reopen() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("store.pmem");
        let initial = *b"INITIAL_VALUE_00";
        let updated = *b"UPDATED_VALUE_01";

        {
            let mut store = PmemCowStore::create_at(&path, 16, MODE, &initial).expect("create");
            assert_eq!(store.read().expect("read"), &initial);
            store.commit(&updated).expect("commit");
            assert_eq!(store.read().expect("read"), &updated);
        } // store dropped here -> munmap

        let reopened = PmemCowStore::open_at(&path, 16, MODE).expect("reopen");
        assert_eq!(
            reopened.recover().expect("recover"),
            &updated,
            "the committed value must survive munmap + reopen"
        );
    }

    // Multiple commits across reopens always recover the most recent value.
    #[test]
    fn multiple_commits_across_reopens_recover_latest() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("seq.pmem");
        let values = [*b"AAAA", *b"BBBB", *b"CCCC", *b"DDDD"];

        let mut store = PmemCowStore::create_at(&path, 4, MODE, &values[0]).expect("create");
        for value in &values[1..] {
            store.commit(value).expect("commit");
        }
        let last = *values.last().expect("nonempty");
        assert_eq!(store.read().expect("read"), &last);
        drop(store);

        let reopened = PmemCowStore::open_at(&path, 4, MODE).expect("reopen");
        assert_eq!(reopened.recover().expect("recover"), &last);
    }

    #[test]
    fn wrong_value_length_is_rejected() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("badlen.pmem");
        let mut store = PmemCowStore::create_at(&path, 8, MODE, &[0xABu8; 8]).expect("create");
        let err = store
            .commit(&[0xCDu8; 4])
            .expect_err("wrong length must error");
        assert!(matches!(err, DaxError::Pmem(_)));
    }
}
