//! One residency identity rule, shared by every backend's device-buffer cache.
//!
//! A resident weight is a host block a backend uploaded once and serves from
//! device memory afterwards. Whether a later offer may be served from that
//! upload is a question about the HOST block, not about any one device, so the
//! answer lives here and the backends keep only their buffer handles
//! (`MetalBuffer`, `CudaGraphBuffer`).
//!
//! The identity is `(name, host address, byte length, owner generation)`.
//! [`ResidentLedger::lookup`] serves a name only at the identity it was first
//! cached with; any other offer under that name is a typed
//! [`ResidentRefusal`], never a stale serve and never a silent replace.
//! [`ResidentLedger::release_names`] is the one eviction: the owner drops its
//! bytes, releases the names, and the next offer uploads fresh data.
//!
//! The owner generation closes the case the address cannot: an allocator may
//! hand a dropped block's address and length to a new block, and the same name
//! then matches on `(address, length)` alone. An owner that rebinds a name
//! bumps the generation it offers and the stale entry refuses. Owners that do
//! not track one offer [`ResidentIdentity::new`]'s generation 0, and then the
//! release call is their only protection.
//!
//! Composes: `BTreeMap` keyed by name, no new pipe. It is a cache identity, not
//! a dataflow step, so [`Pipe`](proxima_primitives::pipe::Pipe) has nothing to
//! add; the caller-visible gain over each backend's private map is one rule and
//! one error on every backend.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use core::ops::Range;

/// The host block a resident name was bound to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentIdentity {
    pub address: usize,
    pub byte_length: usize,
    pub generation: u64,
}

impl ResidentIdentity {
    /// Identity for an owner that does not track a generation (generation 0).
    #[must_use]
    pub const fn new(address: usize, byte_length: usize) -> Self {
        Self {
            address,
            byte_length,
            generation: 0,
        }
    }

    /// The same block offered by the owner's `generation`-th binding of the name.
    #[must_use]
    pub const fn with_generation(self, generation: u64) -> Self {
        Self { generation, ..self }
    }
}

/// Owner generations: the one epoch per resident name that owners of host
/// bytes advance (through `backend::release_resident_names`) whenever they
/// free or reallocate the bytes under that name, and that every backend reads
/// when it forms an offered [`ResidentIdentity`]. A name never advanced is
/// generation 0.
#[cfg(feature = "std")]
mod owner_generations {
    use std::sync::OnceLock;

    use alloc::collections::BTreeMap;
    use alloc::string::{String, ToString};
    use proxima_primitives::sync::blocking::Mutex;

    static GENERATIONS: OnceLock<Mutex<BTreeMap<String, u64>>> = OnceLock::new();

    fn generations() -> &'static Mutex<BTreeMap<String, u64>> {
        GENERATIONS.get_or_init(|| Mutex::new(BTreeMap::new()))
    }

    pub(super) fn current(name: &str) -> u64 {
        generations().lock().get(name).copied().unwrap_or(0)
    }

    pub(super) fn advance<'name>(names: impl IntoIterator<Item = &'name str>) {
        let mut held = generations().lock();
        for name in names {
            let next = held.get(name).copied().unwrap_or(0) + 1;
            held.insert(name.to_string(), next);
        }
    }
}

/// The generation the owner of `name`'s host bytes currently stands at.
#[cfg(feature = "std")]
#[must_use]
pub fn owner_generation(name: &str) -> u64 {
    owner_generations::current(name)
}

/// Advances these names' owner generation: their previous host bytes are gone.
#[cfg(feature = "std")]
pub fn advance_owner_generation<'name>(names: impl IntoIterator<Item = &'name str>) {
    owner_generations::advance(names);
}

/// A name was offered at an identity other than the one it was cached with.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "resident name {name:?} rebound to a different host block (cached {cached:?}, offered {offered:?})"
)]
pub struct ResidentRefusal {
    pub name: String,
    pub cached: ResidentIdentity,
    pub offered: ResidentIdentity,
}

/// The shared rule: `offered` may be served from `cached` only when equal.
///
/// # Errors
///
/// [`ResidentRefusal`] when any of address, length or generation differ.
pub fn check_resident_identity(
    name: &str,
    cached: ResidentIdentity,
    offered: ResidentIdentity,
) -> Result<(), ResidentRefusal> {
    if cached == offered {
        return Ok(());
    }
    Err(ResidentRefusal {
        name: name.to_string(),
        cached,
        offered,
    })
}

/// Name-keyed resident cache over a backend's own buffer handle.
#[derive(Debug, Clone)]
pub struct ResidentLedger<Handle> {
    entries: BTreeMap<String, (ResidentIdentity, Handle)>,
}

impl<Handle> Default for ResidentLedger<Handle> {
    fn default() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }
}

impl<Handle: Clone> ResidentLedger<Handle> {
    /// `Ok(None)` when `name` was never cached (the caller uploads), the cached
    /// handle when `offered` matches, [`ResidentRefusal`] otherwise.
    ///
    /// # Errors
    ///
    /// [`ResidentRefusal`] when `name` is cached at a different identity.
    pub fn lookup(
        &self,
        name: &str,
        offered: ResidentIdentity,
    ) -> Result<Option<Handle>, ResidentRefusal> {
        let Some((cached, handle)) = self.entries.get(name) else {
            return Ok(None);
        };
        check_resident_identity(name, *cached, offered)?;
        Ok(Some(handle.clone()))
    }

    /// Caches `handle` for `name`. An existing entry at the same identity keeps
    /// its handle; an existing entry at another identity refuses.
    ///
    /// # Errors
    ///
    /// [`ResidentRefusal`] when `name` is cached at a different identity.
    pub fn insert(
        &mut self,
        name: &str,
        identity: ResidentIdentity,
        handle: Handle,
    ) -> Result<(), ResidentRefusal> {
        if let Some((cached, _)) = self.entries.get(name) {
            return check_resident_identity(name, *cached, identity);
        }
        self.entries.insert(name.to_string(), (identity, handle));
        Ok(())
    }
}

impl<Handle> ResidentLedger<Handle> {
    /// Evicts these names; the shared release every backend's cache honors.
    pub fn release_names<'name>(&mut self, names: impl IntoIterator<Item = &'name str>) {
        for name in names {
            self.entries.remove(name);
        }
    }

    /// Evicts every name whose host block starts inside one of `ranges`, for an
    /// owner that frees host bytes and knows addresses rather than names.
    pub fn release_ranges(&mut self, ranges: &[Range<usize>]) {
        self.entries.retain(|_, (identity, _)| {
            !ranges.iter().any(|range| range.contains(&identity.address))
        });
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;

    // no device is needed to prove the rule: the "device buffer" here is a host
    // copy of the uploaded bytes, so a stale serve shows up as stale content.
    type FakeDeviceBuffer = Vec<u8>;

    fn identity_of(block: &[u8]) -> ResidentIdentity {
        ResidentIdentity::new(block.as_ptr() as usize, block.len())
    }

    fn serve(
        ledger: &mut ResidentLedger<FakeDeviceBuffer>,
        name: &str,
        block: &[u8],
        generation: u64,
    ) -> Result<FakeDeviceBuffer, ResidentRefusal> {
        let offered = identity_of(block).with_generation(generation);
        if let Some(buffer) = ledger.lookup(name, offered)? {
            return Ok(buffer);
        }
        ledger.insert(name, offered, block.to_vec())?;
        Ok(block.to_vec())
    }

    #[test]
    fn same_name_same_block_is_served_from_the_first_upload() {
        let mut ledger = ResidentLedger::default();
        let block = vec![1_u8; 64];
        let first = serve(&mut ledger, "w", &block, 0).expect("first upload");
        let second = serve(&mut ledger, "w", &block, 0).expect("cached serve");
        assert_eq!(first, second);
        assert_eq!(ledger.len(), 1);
    }

    #[test]
    fn dropped_block_reallocated_under_the_same_name_never_serves_stale_data() {
        let mut ledger = ResidentLedger::default();
        let old_block = vec![1_u8; 4096];
        serve(&mut ledger, "w", &old_block, 0).expect("first upload");
        drop(old_block);

        let new_block = vec![2_u8; 4096];
        // the owner advances its generation when it reallocates, wherever the allocator put the block.
        let refusal = serve(&mut ledger, "w", &new_block, 1)
            .expect_err("a reallocated block must refuse, never serve the old upload");
        assert_eq!(refusal.name, "w");
    }

    #[test]
    fn recycled_address_with_a_new_owner_generation_is_refused() {
        let mut ledger = ResidentLedger::default();
        let block = vec![1_u8; 256];
        serve(&mut ledger, "w", &block, 0).expect("first upload");

        // same address and length by construction: the case an allocator may produce.
        let refusal = serve(&mut ledger, "w", &block, 1)
            .expect_err("a rebound generation at an identical address must refuse");
        assert_eq!(refusal.cached.address, refusal.offered.address);
        assert_eq!(refusal.cached.byte_length, refusal.offered.byte_length);
        assert_ne!(refusal.cached.generation, refusal.offered.generation);
    }

    #[cfg(feature = "std")]
    #[test]
    fn the_shared_release_advances_the_generation_every_backend_reads() {
        let name = "rb1.shared_release.weight";
        let mut ledger = ResidentLedger::default();
        let block = vec![1_u8; 256];
        let before = crate::resident_identity::owner_generation(name);
        serve(&mut ledger, name, &block, before).expect("first upload");

        // the owner reallocates: the release every owner calls, same address, same length.
        crate::backend::release_resident_names([name]);
        let after = crate::resident_identity::owner_generation(name);
        assert_eq!(
            after,
            before + 1,
            "the release advanced the owner generation"
        );

        let refusal = serve(&mut ledger, name, &block, after)
            .expect_err("a ledger the release did not reach refuses the rebound generation");
        assert_eq!(refusal.cached.address, refusal.offered.address);

        ledger.release_names([name]);
        let rebound = vec![7_u8; 256];
        let buffer = serve(&mut ledger, name, &rebound, after).expect("fresh upload");
        assert_eq!(buffer, rebound, "the evicted name serves the new bytes");
    }

    #[test]
    fn different_length_under_the_same_name_is_refused() {
        let mut ledger = ResidentLedger::default();
        let block = vec![1_u8; 64];
        serve(&mut ledger, "w", &block, 0).expect("first upload");
        let longer = vec![1_u8; 128];
        assert!(serve(&mut ledger, "w", &longer, 0).is_err());
    }

    #[test]
    fn release_names_makes_the_next_offer_a_fresh_upload_with_the_new_bytes() {
        let mut ledger = ResidentLedger::default();
        let block = vec![1_u8; 256];
        serve(&mut ledger, "w", &block, 0).expect("first upload");
        ledger.release_names(["w"]);
        assert!(ledger.is_empty());

        let rebound = vec![9_u8; 256];
        let buffer = serve(&mut ledger, "w", &rebound, 1).expect("fresh upload after release");
        assert_eq!(buffer, rebound);
    }

    #[test]
    fn release_ranges_evicts_only_names_whose_block_starts_in_the_range() {
        let mut ledger = ResidentLedger::default();
        let kept = vec![1_u8; 64];
        let freed = vec![2_u8; 64];
        serve(&mut ledger, "kept", &kept, 0).expect("kept upload");
        serve(&mut ledger, "freed", &freed, 0).expect("freed upload");

        let start = freed.as_ptr() as usize;
        ledger.release_ranges(&[start..start + freed.len()]);
        assert_eq!(ledger.len(), 1);
        assert!(
            ledger
                .lookup("kept", identity_of(&kept))
                .expect("kept stays valid")
                .is_some()
        );
    }
}
