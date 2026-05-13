//! Per-callsite cached emit gate — the mechanism that lets proxima meet (and on
//! the compile-out path, beat) `tracing`'s disabled-callsite fast path.
//!
//! `tracing` caches a per-callsite `Interest`, so a statically-disabled log site
//! is an atomic load + early return and the record is never built. proxima's
//! filter otherwise runs at drain — meaning a disabled record is built, ringed,
//! drained, and only then dropped. A [`CallsiteGate`] moves the decision to the
//! emit site: one `static` gate per callsite caches `(generation, decision)`, so
//! the steady-state cost is two relaxed atomic loads + a branch, and a
//! Drop short-circuits BEFORE the record is constructed (mirroring the existing
//! sampler pre-allocation gate). The expensive [`CompiledEmit::decide`] runs once
//! per callsite per filter generation, not per record.
//!
//! Tier T1 (value T0 — `core` atomics, no alloc).

use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::emit::Decision;

const PRESENT: u64 = 1 << 0;
const KEEP: u64 = 1 << 1;
const GEN_SHIFT: u32 = 32;

/// A monotonically increasing filter generation. Bump it whenever the active
/// filter changes so every callsite gate recomputes on its next hit (the
/// analogue of tracing's `rebuild_interest_cache`).
pub struct FilterGeneration(AtomicU32);

impl FilterGeneration {
    /// Start at generation 1 (0 is reserved as "never cached").
    #[must_use]
    pub const fn new() -> Self {
        Self(AtomicU32::new(1))
    }

    /// The current generation.
    #[inline]
    #[must_use]
    pub fn current(&self) -> u32 {
        self.0.load(Ordering::Relaxed)
    }

    /// Advance the generation, invalidating every cached callsite decision.
    pub fn bump(&self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

impl Default for FilterGeneration {
    fn default() -> Self {
        Self::new()
    }
}

/// A cached keep/drop decision for one callsite. Declare one `static` per emit
/// site; the macro consults it before building the record.
pub struct CallsiteGate {
    /// Packed: `[ generation:32 | .. | KEEP:1 | PRESENT:1 ]`.
    cache: AtomicU64,
}

impl CallsiteGate {
    /// A fresh, uncached gate.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            cache: AtomicU64::new(0),
        }
    }

    /// Decide for this callsite. On a generation hit this is two relaxed loads +
    /// a branch (no `recompute`); on a miss it runs `recompute` once (the
    /// [`CompiledEmit::decide`](crate::emit::CompiledEmit) scan) and caches it.
    #[inline]
    pub fn decide(&self, generation: u32, recompute: impl FnOnce() -> Decision) -> Decision {
        let cached = self.cache.load(Ordering::Relaxed);
        if cached & PRESENT != 0 && (cached >> GEN_SHIFT) as u32 == generation {
            return if cached & KEEP != 0 {
                Decision::Keep
            } else {
                Decision::Drop
            };
        }
        let decision = recompute();
        let mut packed = (u64::from(generation) << GEN_SHIFT) | PRESENT;
        if decision == Decision::Keep {
            packed |= KEEP;
        }
        self.cache.store(packed, Ordering::Relaxed);
        decision
    }

    /// True if this callsite would keep at the given generation, using the cache.
    /// The hot path a macro guards with: `if gate.is_enabled(...) { build + emit }`.
    #[inline]
    pub fn is_enabled(&self, generation: u32, recompute: impl FnOnce() -> Decision) -> bool {
        self.decide(generation, recompute) == Decision::Keep
    }
}

impl Default for CallsiteGate {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default,
        clippy::type_complexity,
        clippy::useless_vec,
        clippy::needless_range_loop,
        clippy::default_constructed_unit_structs
    )]

    use core::cell::Cell;

    use super::{CallsiteGate, FilterGeneration};
    use crate::emit::Decision;

    // the first hit computes; subsequent hits at the same generation are cached
    // (recompute is NOT called again).
    #[test]
    fn caches_after_first_compute() {
        let gate = CallsiteGate::new();
        let computes = Cell::new(0);
        let recompute = || {
            computes.set(computes.get() + 1);
            Decision::Drop
        };

        assert_eq!(gate.decide(1, recompute), Decision::Drop);
        assert_eq!(gate.decide(1, recompute), Decision::Drop);
        assert_eq!(gate.decide(1, recompute), Decision::Drop);
        assert_eq!(
            computes.get(),
            1,
            "recompute ran once; the rest were cached"
        );
    }

    // bumping the generation invalidates the cache (recompute runs again).
    #[test]
    fn generation_bump_invalidates_cache() {
        let generation = FilterGeneration::new();
        let gate = CallsiteGate::new();
        let computes = Cell::new(0);
        let bump_compute = || computes.set(computes.get() + 1);

        let gen0 = generation.current();
        assert!(!gate.is_enabled(gen0, || {
            bump_compute();
            Decision::Drop
        }));
        assert!(!gate.is_enabled(gen0, || {
            bump_compute();
            Decision::Drop
        }));
        assert_eq!(computes.get(), 1, "cached at the same generation");

        generation.bump(); // filter changed
        let gen1 = generation.current();
        assert!(gate.is_enabled(gen1, || {
            bump_compute();
            Decision::Keep
        }));
        assert_eq!(computes.get(), 2, "recomputed after the generation bump");
    }
}
