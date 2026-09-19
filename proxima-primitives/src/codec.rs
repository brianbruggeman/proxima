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
    Q4_1,
    Q8_1,
    Q8K,
    Iq1S,
    Iq1M,
    Iq2Xxs,
    Iq2Xs,
    Iq2S,
    Iq3Xxs,
    Iq3S,
    Iq4Nl,
    Iq4Xs,
    Tq10,
    Tq20,
    Mxfp4,
    Nvfp4,
    Q1_0,
    Q2_0,
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
            Codec::Q4_1 => 11,
            Codec::Q8_1 => 12,
            Codec::Q8K => 13,
            Codec::Iq1S => 14,
            Codec::Iq1M => 15,
            Codec::Iq2Xxs => 16,
            Codec::Iq2Xs => 17,
            Codec::Iq2S => 18,
            Codec::Iq3Xxs => 19,
            Codec::Iq3S => 20,
            Codec::Iq4Nl => 21,
            Codec::Iq4Xs => 22,
            Codec::Tq10 => 23,
            Codec::Tq20 => 24,
            Codec::Mxfp4 => 25,
            Codec::Nvfp4 => 26,
            Codec::Q1_0 => 27,
            Codec::Q2_0 => 28,
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
            11 => Some(Codec::Q4_1),
            12 => Some(Codec::Q8_1),
            13 => Some(Codec::Q8K),
            14 => Some(Codec::Iq1S),
            15 => Some(Codec::Iq1M),
            16 => Some(Codec::Iq2Xxs),
            17 => Some(Codec::Iq2Xs),
            18 => Some(Codec::Iq2S),
            19 => Some(Codec::Iq3Xxs),
            20 => Some(Codec::Iq3S),
            21 => Some(Codec::Iq4Nl),
            22 => Some(Codec::Iq4Xs),
            23 => Some(Codec::Tq10),
            24 => Some(Codec::Tq20),
            25 => Some(Codec::Mxfp4),
            26 => Some(Codec::Nvfp4),
            27 => Some(Codec::Q1_0),
            28 => Some(Codec::Q2_0),
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
            Codec::Q4_1,
            Codec::Q8_1,
            Codec::Q8K,
            Codec::Iq1S,
            Codec::Iq1M,
            Codec::Iq2Xxs,
            Codec::Iq2Xs,
            Codec::Iq2S,
            Codec::Iq3Xxs,
            Codec::Iq3S,
            Codec::Iq4Nl,
            Codec::Iq4Xs,
            Codec::Tq10,
            Codec::Tq20,
            Codec::Mxfp4,
            Codec::Nvfp4,
            Codec::Q1_0,
            Codec::Q2_0,
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
