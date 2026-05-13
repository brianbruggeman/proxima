//! Hierarchical level coordinate — the packed integer behind a named level tree.
//!
//! A [`Coord`] is the hot-path representation of a hierarchical level such as
//! `error.auth.token`. The *names* are the user surface (see the manifest layer);
//! the dotted-numeric form (`17.2.1`) is what the names compile to and what gets
//! compared per record. Segment 0 is the flat severity band, so a flat
//! [`crate::level::Level`] is exactly a depth-1 `Coord` — `error` and the whole
//! `error.*` subtree share band 17 and order above every lower band, preserving
//! `Level`'s severity ordering unchanged.
//!
//! The packing buys two things on the hot path: a `u64` compare reproduces
//! component-wise tree order, and a single mask answers "is this coord in that
//! subtree?" in O(1) with no allocation. All of `Coord` is `no_std`/no-alloc.

use core::cmp::Ordering;
use core::fmt;

/// Bits per path segment. 12 bits → each segment is `0..=4095`.
const SEG_BITS: u32 = 12;
/// Maximum path depth. A deeper path is a construction error, never a silent
/// truncation.
const MAX_DEPTH: u32 = 4;
/// Total payload bits = four 12-bit segments.
const PAYLOAD_BITS: u32 = SEG_BITS * MAX_DEPTH;
/// Largest value a single segment can hold.
pub const SEG_MAX: u16 = (1 << SEG_BITS) - 1;
const PAYLOAD_MASK: u64 = (1u64 << PAYLOAD_BITS) - 1;
const BAND_SHIFT: u32 = PAYLOAD_BITS - SEG_BITS;

/// A hierarchical level coordinate, e.g. `17.2.1` (the packing of a name like
/// `error.auth.token`).
///
/// Layout: four 12-bit segments packed into the low 48 bits of `payload`, seg0
/// most-significant; `depth` (`1..=4`) is carried separately so a literal `0`
/// segment never collides with "absent". Seg0 is the flat severity band, so
/// `Coord::from(Level::ERROR)` is the depth-1 coord with band 17.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct Coord {
    payload: u64,
    depth: u8,
}

/// Why a coordinate failed to parse — structured so the config validator can
/// list what went wrong (P15 discoverability), unlike a flat opaque error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoordError {
    /// An empty string, or a path with an empty segment (`1..2`).
    Empty,
    /// More than [`MAX_DEPTH`](self) segments.
    TooDeep { got: usize, max: u32 },
    /// A segment exceeds [`SEG_MAX`].
    SegmentOverflow { index: usize, value: u32, max: u16 },
    /// A segment was not a base-10 number.
    NotNumeric,
}

impl fmt::Display for CoordError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("empty coordinate"),
            Self::TooDeep { got, max } => {
                write!(formatter, "coordinate too deep: {got} segments (max {max})")
            }
            Self::SegmentOverflow { index, value, max } => {
                write!(formatter, "segment {index} = {value} exceeds max {max}")
            }
            Self::NotNumeric => formatter.write_str("non-numeric segment"),
        }
    }
}

impl core::error::Error for CoordError {}

impl Coord {
    /// Build a coordinate from explicit segments. `const` so named levels can be
    /// declared in a `const`/`static` and a bad path is a compile error.
    ///
    /// Panics (compile error in `const` context) on an empty path, more than
    /// four segments, or a segment above [`SEG_MAX`]. The runtime fallible path
    /// is [`Coord::parse`].
    #[must_use]
    pub const fn new(segments: &[u16]) -> Coord {
        assert!(
            !segments.is_empty() && segments.len() <= MAX_DEPTH as usize,
            "coord depth must be 1..=4"
        );
        let mut payload = 0u64;
        let mut index = 0;
        while index < segments.len() {
            assert!(segments[index] <= SEG_MAX, "coord segment must be <= 4095");
            let shift = PAYLOAD_BITS - SEG_BITS * (index as u32 + 1);
            payload |= (segments[index] as u64) << shift;
            index += 1;
        }
        Coord {
            payload,
            depth: segments.len() as u8,
        }
    }

    /// Parse a dotted coordinate (`"17.2.1"`). The power-user / wire form; the
    /// surface is normally a registered name. Errors (never truncates) on depth,
    /// overflow, empties, or non-numeric segments.
    pub fn parse(text: &str) -> Result<Coord, CoordError> {
        if text.is_empty() {
            return Err(CoordError::Empty);
        }
        let mut payload = 0u64;
        let mut depth = 0u32;
        for segment in text.split('.') {
            if depth >= MAX_DEPTH {
                return Err(CoordError::TooDeep {
                    got: text.split('.').count(),
                    max: MAX_DEPTH,
                });
            }
            if segment.is_empty() {
                return Err(CoordError::Empty);
            }
            let value: u32 = segment.parse().map_err(|_| CoordError::NotNumeric)?;
            if value > SEG_MAX as u32 {
                return Err(CoordError::SegmentOverflow {
                    index: depth as usize,
                    value,
                    max: SEG_MAX,
                });
            }
            let shift = PAYLOAD_BITS - SEG_BITS * (depth + 1);
            payload |= (value as u64) << shift;
            depth += 1;
        }
        Ok(Coord {
            payload,
            depth: depth as u8,
        })
    }

    /// A flat severity as a depth-1 coordinate (seg0 = severity). The bridge that
    /// makes a flat [`Level`](crate::level::Level) filter the special case of the
    /// hierarchy: a flat `warn` catches every coord in the warn band (§subtree).
    #[must_use]
    pub const fn from_severity(severity: u8) -> Coord {
        Coord {
            payload: (severity as u64) << BAND_SHIFT,
            depth: 1,
        }
    }

    /// Seg0 — the severity band. A flat `Level` collapses to `band() as u8`.
    #[must_use]
    pub const fn band(self) -> u16 {
        ((self.payload >> BAND_SHIFT) & (SEG_MAX as u64)) as u16
    }

    /// Path depth (`1..=4`).
    #[must_use]
    pub const fn depth(self) -> u8 {
        self.depth
    }

    /// True iff `self` lies in the subtree rooted at `ancestor` (prefix ==
    /// subtree). O(1): one mask + one compare. A shorter path is never inside a
    /// deeper one.
    #[must_use]
    pub fn in_subtree_of(self, ancestor: Coord) -> bool {
        let mask = prefix_mask(ancestor.depth);
        self.depth >= ancestor.depth && (self.payload & mask) == (ancestor.payload & mask)
    }

    /// Append a child segment, deepening the path by one. `None` if already at
    /// max depth or the ordinal overflows a segment. Used to auto-assign
    /// coordinates to a named level tree (so operators name, never number).
    #[must_use]
    pub fn child(self, ordinal: u16) -> Option<Coord> {
        if self.depth as u32 >= MAX_DEPTH || ordinal > SEG_MAX {
            return None;
        }
        let shift = PAYLOAD_BITS - SEG_BITS * (self.depth as u32 + 1);
        Some(Coord {
            payload: self.payload | ((ordinal as u64) << shift),
            depth: self.depth + 1,
        })
    }

    fn segment(self, index: u32) -> u16 {
        let shift = PAYLOAD_BITS - SEG_BITS * (index + 1);
        ((self.payload >> shift) & (SEG_MAX as u64)) as u16
    }
}

/// Mask covering the top `depth` segments of the payload.
#[inline]
const fn prefix_mask(depth: u8) -> u64 {
    let used = SEG_BITS * depth as u32;
    // top `used` bits of the 48-bit payload.
    (!((1u64 << (PAYLOAD_BITS - used)) - 1)) & PAYLOAD_MASK
}

impl Ord for Coord {
    fn cmp(&self, other: &Self) -> Ordering {
        // seg0 (the band) is most-significant, so a payload compare is
        // component-wise path order AND preserves Level's severity ordering
        // (error band > info band). depth breaks ties (a prefix sorts before its
        // own deeper children).
        self.payload
            .cmp(&other.payload)
            .then(self.depth.cmp(&other.depth))
    }
}

impl PartialOrd for Coord {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<crate::level::Level> for Coord {
    fn from(level: crate::level::Level) -> Coord {
        Coord::from_severity(level.severity())
    }
}

impl fmt::Debug for Coord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Coord({self})")
    }
}

impl fmt::Display for Coord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for index in 0..self.depth as u32 {
            if index > 0 {
                formatter.write_str(".")?;
            }
            write!(formatter, "{}", self.segment(index))?;
        }
        Ok(())
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

    use super::{Coord, CoordError, SEG_MAX};
    use crate::level::Level;

    // a more-severe band orders above a less-severe one — Level's contract,
    // preserved through Coord (error > warn > info > debug > trace).
    #[rstest]
    #[case::error_over_warn(Level::ERROR, Level::WARN)]
    #[case::warn_over_info(Level::WARN, Level::INFO)]
    #[case::info_over_debug(Level::INFO, Level::DEBUG)]
    #[case::debug_over_trace(Level::DEBUG, Level::TRACE)]
    fn severity_band_dominates_ordering(#[case] higher: Level, #[case] lower: Level) {
        assert!(Coord::from(higher) > Coord::from(lower));
    }

    // within a band, component-wise order holds and the band still dominates.
    #[test]
    fn component_wise_order_within_and_across_bands() {
        assert!(Coord::parse("17.3.5").unwrap() > Coord::parse("17.3.4").unwrap());
        assert!(Coord::parse("17.4").unwrap() > Coord::parse("17.3.9").unwrap());
        // band dominates sub-segments: a bare error outranks a deep info coord.
        assert!(Coord::from(Level::ERROR) > Coord::parse("9.99.99").unwrap());
    }

    // prefix == subtree, and a flat-band filter catches the whole band subtree.
    #[rstest]
    #[case::leaf_in_subband("17.3.5", "17.3", true)]
    #[case::root_in_self("17.3", "17.3", true)]
    #[case::sibling_excluded("17.4", "17.3", false)]
    #[case::shorter_not_in_deeper("17.3", "17.3.5", false)]
    #[case::deep_in_band("17.2.1", "17", true)]
    #[case::other_band_excluded("13.2.1", "17", false)]
    fn subtree_membership(#[case] candidate: &str, #[case] ancestor: &str, #[case] want: bool) {
        let candidate = Coord::parse(candidate).unwrap();
        let ancestor = Coord::parse(ancestor).unwrap();
        assert_eq!(candidate.in_subtree_of(ancestor), want);
    }

    // the flat-Level bridge: a flat `warn` filter is a depth-1 coord whose
    // subtree is every coord in the warn band.
    #[test]
    fn flat_level_filter_catches_its_band_subtree() {
        let warn = Coord::from(Level::WARN);
        assert!(Coord::parse("13.2").unwrap().in_subtree_of(warn)); // 13 == warn severity
        assert!(Coord::parse("13.7.4").unwrap().in_subtree_of(warn));
        assert!(!Coord::parse("17.1").unwrap().in_subtree_of(warn)); // error band, not warn
    }

    // band round-trips through the flat bridge; depth is correct.
    #[test]
    fn from_severity_round_trips_band_and_depth() {
        for level in [Level::TRACE, Level::INFO, Level::ERROR, Level::FATAL] {
            let coord = Coord::from(level);
            assert_eq!(coord.band(), level.severity() as u16);
            assert_eq!(coord.depth(), 1);
        }
    }

    // parse rejects every malformed shape with a specific, listable error.
    #[rstest]
    #[case::empty("", CoordError::Empty)]
    #[case::empty_segment("1..2", CoordError::Empty)]
    #[case::too_deep("1.2.3.4.5", CoordError::TooDeep { got: 5, max: 4 })]
    #[case::overflow("9999", CoordError::SegmentOverflow { index: 0, value: 9999, max: SEG_MAX })]
    #[case::not_numeric("17.x", CoordError::NotNumeric)]
    fn parse_rejects_malformed(#[case] text: &str, #[case] want: CoordError) {
        assert_eq!(Coord::parse(text).unwrap_err(), want);
    }

    // Display is the inverse of parse for well-formed input (teaching + config).
    #[rstest]
    #[case::flat("17")]
    #[case::two("17.3")]
    #[case::leaf("17.2.1")]
    #[case::zero_segment("5.0.3")] // a literal 0 segment survives (depth, not sentinel)
    fn display_round_trips_parse(#[case] text: &str) {
        let coord = Coord::parse(text).unwrap();
        assert_eq!(coord.to_string(), text);
    }

    // const construction works (named levels declare in a const).
    #[test]
    fn const_new_matches_parse() {
        const AUDIT: Coord = Coord::new(&[17, 2, 1]);
        assert_eq!(AUDIT, Coord::parse("17.2.1").unwrap());
    }

    // layout guard: Coord stays a small POD value (u64 + u8 + pad).
    #[test]
    fn coord_is_small_pod() {
        assert_eq!(core::mem::size_of::<Coord>(), 16);
    }
}
