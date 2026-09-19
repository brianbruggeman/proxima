//! The one source-neutral codec identity: which packed byte layout a
//! quantized/packed buffer holds. Payload-less by design -- a pipe or fn
//! pointer cannot serialize itself to a wire byte, so a sidecar wire format
//! (`proxima-model-interop`'s expert sidecar today) needs a compile-time-
//! exhaustive, serializable tag instead. [`Codec::tag`]/[`Codec::from_tag`]
//! are that wire mapping.
//!
//! This enum is the single identity meant to replace the parallel copies
//! that grew up independently: `proxima-model-interop`'s `PackedOwnedKind`
//! (migrated onto this type), `omega`'s `PackedCodec`, and
//! `proxima-tensor`'s `QuantizedBlock` tag (both still to migrate).

/// Which packed byte layout a codec-tagged buffer holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    Q4K,
    Q5K,
    Q6K,
    Q8_0,
    Q3K,
    Q4_0,
    Float16,
    BFloat16,
    Q2K,
    Q5_1,
    Q5_0,
}

impl Codec {
    /// The wire byte a sidecar tags this codec with. Stable across
    /// releases -- an on-disk sidecar's bytes decode by this mapping.
    #[must_use]
    pub fn tag(self) -> u8 {
        match self {
            Codec::Q2K => 0,
            Codec::Q3K => 1,
            Codec::Q4K => 2,
            Codec::Q5K => 3,
            Codec::Q6K => 4,
            Codec::Q8_0 => 5,
            Codec::Q4_0 => 6,
            Codec::Float16 => 7,
            Codec::BFloat16 => 8,
            Codec::Q5_1 => 9,
            Codec::Q5_0 => 10,
        }
    }

    /// The inverse of [`Self::tag`]. `None` for a byte no codec claims.
    #[must_use]
    pub fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Codec::Q2K),
            1 => Some(Codec::Q3K),
            2 => Some(Codec::Q4K),
            3 => Some(Codec::Q5K),
            4 => Some(Codec::Q6K),
            5 => Some(Codec::Q8_0),
            6 => Some(Codec::Q4_0),
            7 => Some(Codec::Float16),
            8 => Some(Codec::BFloat16),
            9 => Some(Codec::Q5_1),
            10 => Some(Codec::Q5_0),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Codec;

    #[test]
    fn tag_round_trips_every_variant() {
        let variants = [
            Codec::Q4K,
            Codec::Q5K,
            Codec::Q6K,
            Codec::Q8_0,
            Codec::Q3K,
            Codec::Q4_0,
            Codec::Float16,
            Codec::BFloat16,
            Codec::Q2K,
            Codec::Q5_1,
            Codec::Q5_0,
        ];
        for codec in variants {
            assert_eq!(Codec::from_tag(codec.tag()), Some(codec));
        }
    }

    #[test]
    fn from_tag_rejects_unknown_byte() {
        assert_eq!(Codec::from_tag(255), None);
    }
}
