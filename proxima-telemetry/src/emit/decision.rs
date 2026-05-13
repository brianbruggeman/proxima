//! The per-target threshold that produces a keep/drop decision. `Copy` POD — the
//! hot path is a `Coord` compare plus an optional mask, no allocation. The
//! `verbose_subtree` field is what makes the level hierarchy load-bearing: a
//! target can keep a whole subtree verbosely while the rest of its records stay
//! quiet — something a flat [`crate::level::Level`] floor cannot express.
//!
//! The keep/drop outcome reuses [`crate::sampler::Decision`] — the emit filter is
//! a pre-allocation gate with the same binary outcome as the sampler, so it
//! borrows that primitive rather than defining its own (P1).

use crate::emit::Coord;
use crate::sampler::Decision;

/// A per-target threshold: keep anything at or above `floor`, plus (optionally)
/// keep the whole `verbose_subtree` regardless of floor.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct EmitThreshold {
    /// Minimum coordinate to keep (the flat-style floor; `record >= floor`).
    pub floor: Coord,
    /// An extra subtree kept verbosely even below `floor` — the hierarchy lever.
    pub verbose_subtree: Option<Coord>,
}

impl EmitThreshold {
    /// A plain floor with no verbose subtree (the flat-`Level` equivalent).
    #[must_use]
    pub const fn at(floor: Coord) -> Self {
        Self {
            floor,
            verbose_subtree: None,
        }
    }

    /// A floor plus a verbose subtree kept regardless of the floor.
    #[must_use]
    pub const fn verbose(floor: Coord, subtree: Coord) -> Self {
        Self {
            floor,
            verbose_subtree: Some(subtree),
        }
    }

    /// Resolve one record's coordinate. Hot path: an optional mask + a compare.
    #[inline]
    #[must_use]
    pub fn decide(&self, coord: Coord) -> Decision {
        if self
            .verbose_subtree
            .is_some_and(|subtree| coord.in_subtree_of(subtree))
        {
            return Decision::Keep;
        }
        if coord >= self.floor {
            Decision::Keep
        } else {
            Decision::Drop
        }
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

    use rstest::rstest;

    use super::EmitThreshold;
    use crate::emit::Coord;
    use crate::level::Level;
    use crate::sampler::Decision;

    // a plain floor behaves exactly like the flat FilterByLevelPipe: keep at or
    // above the floor severity, drop below.
    #[rstest]
    #[case::error_kept_at_warn_floor(Level::WARN, Level::ERROR, Decision::Keep)]
    #[case::warn_kept_at_warn_floor(Level::WARN, Level::WARN, Decision::Keep)]
    #[case::info_dropped_at_warn_floor(Level::WARN, Level::INFO, Decision::Drop)]
    #[case::trace_dropped_at_info_floor(Level::INFO, Level::TRACE, Decision::Drop)]
    fn flat_floor_matches_filter_by_level(
        #[case] floor: Level,
        #[case] record: Level,
        #[case] want: Decision,
    ) {
        let threshold = EmitThreshold::at(Coord::from(floor));
        assert_eq!(threshold.decide(Coord::from(record)), want);
    }

    // the verbose subtree keeps a chatty subtree even though its band is below
    // the floor — the thing a flat filter cannot do.
    #[test]
    fn verbose_subtree_keeps_below_floor() {
        // floor = warn (band 13); verbose subtree = the trace-band io tree (1.3)
        let threshold =
            EmitThreshold::verbose(Coord::from(Level::WARN), Coord::parse("1.3").unwrap());

        assert_eq!(
            threshold.decide(Coord::parse("1.3.5").unwrap()),
            Decision::Keep
        ); // in subtree
        assert_eq!(
            threshold.decide(Coord::parse("1.3").unwrap()),
            Decision::Keep
        ); // subtree root
        assert_eq!(
            threshold.decide(Coord::parse("1.4").unwrap()),
            Decision::Drop
        ); // sibling, below floor
        assert_eq!(threshold.decide(Coord::from(Level::ERROR)), Decision::Keep); // above floor
        assert_eq!(threshold.decide(Coord::from(Level::INFO)), Decision::Drop); // below floor, not in subtree
    }
}
