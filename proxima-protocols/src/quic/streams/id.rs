//! Stream identifier per [RFC 9000 §2.1].
//!
//! Stream IDs are 62-bit varints. The two low bits encode the
//! direction + initiator:
//!
//! | bit pattern | initiator | direction |
//! |-------------|-----------|-----------|
//! | `0b00`      | Client    | Bidi      |
//! | `0b01`      | Server    | Bidi      |
//! | `0b10`      | Client    | Uni       |
//! | `0b11`      | Server    | Uni       |
//!
//! Each side issues stream IDs in ascending order within its
//! initiator + direction class. The first client-bidi stream is
//! `StreamId(0)`; the next is `StreamId(4)`; etc.
//!
//! [RFC 9000 §2.1]: https://www.rfc-editor.org/rfc/rfc9000#section-2.1

use crate::quic::side::Side;

/// Stream direction per RFC 9000 §2.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StreamDirection {
    /// Bidirectional — both halves are open.
    Bidi,
    /// Unidirectional — only the initiator can send.
    Uni,
}

/// 62-bit stream identifier.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StreamId(pub u64);

impl StreamId {
    /// Maximum legal stream ID per RFC 9000 §16 (62-bit varint cap).
    pub const MAX: u64 = (1u64 << 62) - 1;

    /// First-stream IDs per (side, direction).
    const BASE_CLIENT_BIDI: u64 = 0;
    const BASE_SERVER_BIDI: u64 = 1;
    const BASE_CLIENT_UNI: u64 = 2;
    const BASE_SERVER_UNI: u64 = 3;

    /// Direction encoded in the low 2 bits per RFC 9000 §2.1.
    #[must_use]
    pub const fn direction(self) -> StreamDirection {
        if self.0 & 0x2 == 0 {
            StreamDirection::Bidi
        } else {
            StreamDirection::Uni
        }
    }

    /// Initiator encoded in bit 0 per RFC 9000 §2.1.
    #[must_use]
    pub const fn initiator(self) -> Side {
        if self.0 & 0x1 == 0 {
            Side::Client
        } else {
            Side::Server
        }
    }

    /// Compute the next stream ID a side would issue for the given
    /// direction. `prev` is the most-recent ID we've already
    /// issued in this class (or `None` if this is the first).
    #[must_use]
    pub const fn next_local(prev: Option<Self>, side: Side, direction: StreamDirection) -> Self {
        let base = match (side, direction) {
            (Side::Client, StreamDirection::Bidi) => Self::BASE_CLIENT_BIDI,
            (Side::Server, StreamDirection::Bidi) => Self::BASE_SERVER_BIDI,
            (Side::Client, StreamDirection::Uni) => Self::BASE_CLIENT_UNI,
            (Side::Server, StreamDirection::Uni) => Self::BASE_SERVER_UNI,
        };
        match prev {
            Some(Self(value)) => Self(value + 4),
            None => Self(base),
        }
    }

    /// Construct from a wire varint, returning `None` if the value
    /// exceeds the 62-bit limit per RFC 9000 §16.
    #[must_use]
    pub const fn from_varint(value: u64) -> Option<Self> {
        if value > Self::MAX {
            None
        } else {
            Some(Self(value))
        }
    }

    /// Raw u64 value (for serialisation).
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Was this stream opened by the local `side` (as opposed to the
    /// peer)?
    #[must_use]
    pub const fn is_local(self, side: Side) -> bool {
        matches!(
            (self.initiator(), side),
            (Side::Client, Side::Client) | (Side::Server, Side::Server)
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn first_client_bidi_is_zero() {
        let id = StreamId::next_local(None, Side::Client, StreamDirection::Bidi);
        assert_eq!(id, StreamId(0));
        assert_eq!(id.direction(), StreamDirection::Bidi);
        assert_eq!(id.initiator(), Side::Client);
    }

    #[test]
    fn first_server_bidi_is_one() {
        let id = StreamId::next_local(None, Side::Server, StreamDirection::Bidi);
        assert_eq!(id, StreamId(1));
        assert_eq!(id.direction(), StreamDirection::Bidi);
        assert_eq!(id.initiator(), Side::Server);
    }

    #[test]
    fn first_client_uni_is_two() {
        let id = StreamId::next_local(None, Side::Client, StreamDirection::Uni);
        assert_eq!(id, StreamId(2));
        assert_eq!(id.direction(), StreamDirection::Uni);
        assert_eq!(id.initiator(), Side::Client);
    }

    #[test]
    fn first_server_uni_is_three() {
        let id = StreamId::next_local(None, Side::Server, StreamDirection::Uni);
        assert_eq!(id, StreamId(3));
        assert_eq!(id.direction(), StreamDirection::Uni);
        assert_eq!(id.initiator(), Side::Server);
    }

    #[test]
    fn next_within_class_increments_by_four() {
        let prev = StreamId(0);
        let next = StreamId::next_local(Some(prev), Side::Client, StreamDirection::Bidi);
        assert_eq!(next, StreamId(4));
        let third = StreamId::next_local(Some(next), Side::Client, StreamDirection::Bidi);
        assert_eq!(third, StreamId(8));
    }

    #[test]
    fn direction_and_initiator_round_trip_through_class() {
        for stream_id in [0u64, 1, 2, 3, 4, 5, 6, 7, 100, 101, 102, 103] {
            let id = StreamId(stream_id);
            let (expected_side, expected_dir) = match stream_id & 0x3 {
                0 => (Side::Client, StreamDirection::Bidi),
                1 => (Side::Server, StreamDirection::Bidi),
                2 => (Side::Client, StreamDirection::Uni),
                _ => (Side::Server, StreamDirection::Uni),
            };
            assert_eq!(id.initiator(), expected_side, "stream {stream_id}");
            assert_eq!(id.direction(), expected_dir, "stream {stream_id}");
        }
    }

    #[test]
    fn from_varint_rejects_above_max() {
        assert_eq!(
            StreamId::from_varint(StreamId::MAX),
            Some(StreamId(StreamId::MAX))
        );
        assert_eq!(StreamId::from_varint(StreamId::MAX + 1), None);
    }

    #[test]
    fn is_local_matches_initiator_against_side() {
        assert!(StreamId(0).is_local(Side::Client));
        assert!(!StreamId(0).is_local(Side::Server));
        assert!(StreamId(1).is_local(Side::Server));
        assert!(!StreamId(1).is_local(Side::Client));
    }
}
