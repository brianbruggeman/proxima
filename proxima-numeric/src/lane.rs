//! The all-lanes mask a lane-parallel fold names: the low `lanes` bits set.
//!
//! A card-C `_sync` shuffle takes that mask as a value, but the width of the value is the
//! dialect's, not the wavefront's: CUDA's intrinsics take an `unsigned int`, while HIP's take a
//! 64-bit `unsigned long long` on every wavefront width. This module owns the numeric half (which
//! lanes are set) so `omega`'s target owns only the dialect half (how the literal is spelled).

/// The mask that marks every lane of a `lanes`-wide warp or wavefront active.
///
/// ```
/// use proxima_numeric::all_lanes_mask;
///
/// assert_eq!(all_lanes_mask(32), 0xffff_ffff);
/// assert_eq!(all_lanes_mask(64), u64::MAX);
/// ```
#[must_use]
pub const fn all_lanes_mask(lanes: u32) -> u64 {
    if lanes >= u64::BITS {
        u64::MAX
    } else {
        (1u64 << lanes) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::all_lanes_mask;

    #[test]
    fn a_warp_and_a_cdna_wavefront_fill_every_lane_they_have() {
        assert_eq!(all_lanes_mask(32), u64::from(u32::MAX));
        assert_eq!(all_lanes_mask(64), u64::MAX);
    }

    #[test]
    fn an_empty_or_oversized_fold_never_shifts_past_the_word() {
        assert_eq!(all_lanes_mask(0), 0);
        assert_eq!(all_lanes_mask(65), u64::MAX);
        assert_eq!(all_lanes_mask(u32::MAX), u64::MAX);
    }
}
