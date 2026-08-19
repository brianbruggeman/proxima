//! The two wire-level enums GGUF stores as `int32_t`: the metadata value
//! type tag and the tensor element type tag.
//!
//! Layout taken from llama.cpp (`/Users/brianbruggeman/repos/others/llama.cpp`):
//! `ggml/include/gguf.h:53-68` (`enum gguf_type`) and
//! `ggml/include/ggml.h:352-391` (`enum ggml_type`).

/// The type tag stored alongside every metadata KV pair (and, for arrays,
/// alongside every element). Wire values match `enum gguf_type` exactly —
/// `gguf.h:53-68`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MetadataType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl MetadataType {
    /// Decode the raw wire tag. `None` for any value outside `gguf_type`'s
    /// range (`GGUF_TYPE_COUNT` in `gguf.h:67` is 13).
    #[must_use]
    pub fn from_wire(raw: u32) -> Option<Self> {
        let value = match raw {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            _ => return None,
        };
        Some(value)
    }
}

/// [`MetadataType`] minus `Array` — the type tag a value or array *element*
/// actually carries once the outer `Array` wrapper (if any) is stripped off.
/// Exists so the scalar reader has an exhaustive match with no `Array` arm
/// to fake a case for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    U64,
    I64,
    F64,
}

impl ScalarType {
    /// `None` when `wire` is `MetadataType::Array` (arrays of arrays are
    /// not a thing this format allows — `gguf.cpp:462`) or unknown.
    #[must_use]
    pub fn from_metadata_type(wire: MetadataType) -> Option<Self> {
        let value = match wire {
            MetadataType::U8 => Self::U8,
            MetadataType::I8 => Self::I8,
            MetadataType::U16 => Self::U16,
            MetadataType::I16 => Self::I16,
            MetadataType::U32 => Self::U32,
            MetadataType::I32 => Self::I32,
            MetadataType::F32 => Self::F32,
            MetadataType::Bool => Self::Bool,
            MetadataType::String => Self::String,
            MetadataType::U64 => Self::U64,
            MetadataType::I64 => Self::I64,
            MetadataType::F64 => Self::F64,
            MetadataType::Array => return None,
        };
        Some(value)
    }
}

/// The tensor element type tag. Wire values match `enum ggml_type` —
/// `ggml.h:352-391`. Only the currently-live variants are named; the file
/// format reserves other `int32_t` values for types ggml has since removed
/// (`Q4_2`/`Q4_3`/the 4-bit-lane `Q4_0_*` variants) and those decode to
/// `None` here, same as an unknown future type would.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
// the ggml/gguf ecosystem's own names (Q4_0, Q5_1, ...) are the identifiers
// every model card and tool documents; renaming them to strict CamelCase
// would make this type harder to cross-reference against the format, not easier.
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    Iq2Xxs,
    Iq2Xs,
    Iq3Xxs,
    Iq1S,
    Iq4Nl,
    Iq3S,
    Iq2S,
    Iq4Xs,
    I8,
    I16,
    I32,
    I64,
    F64,
    Iq1M,
    Bf16,
    Tq10,
    Tq20,
}

/// Per-type block shape: `block_elements` values are packed into
/// `block_bytes` bytes. Contiguous (non-quantized) types have
/// `block_elements == 1`. Sizes verified against the block struct layouts
/// in `ggml/src/ggml-common.h` (`block_q4_0` etc., lines 167-410) and the
/// `type_traits` table in `ggml/src/ggml.c:566-845`; `ggml_half` is
/// `uint16_t` (`ggml-common.h:6`), `QK_K` is 256 (`ggml-common.h:89`),
/// `K_SCALE_SIZE` is 12 (`ggml-common.h:90`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockLayout {
    pub block_elements: u64,
    pub block_bytes: u64,
}

impl GgmlType {
    /// Decode the raw `ggml_type` wire tag (`enum ggml_type`, `ggml.h:352-391`).
    /// `None` for a value outside `[0, GGML_TYPE_COUNT)` (39, `ggml.h:391`)
    /// or one of the retired gap values (4, 5, 31, 32, 33, 36, 37, 38).
    #[must_use]
    pub fn from_wire(raw: i32) -> Option<Self> {
        let value = match raw {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            16 => Self::Iq2Xxs,
            17 => Self::Iq2Xs,
            18 => Self::Iq3Xxs,
            19 => Self::Iq1S,
            20 => Self::Iq4Nl,
            21 => Self::Iq3S,
            22 => Self::Iq2S,
            23 => Self::Iq4Xs,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::Iq1M,
            30 => Self::Bf16,
            34 => Self::Tq10,
            35 => Self::Tq20,
            _ => return None,
        };
        Some(value)
    }

    /// Block layout used to compute a tensor's exact byte footprint, the
    /// same arithmetic as `ggml_nbytes` (`ggml.c`, via `type_traits`
    /// `blck_size`/`type_size`, `ggml.c:1174,1178`). Dequantizing the
    /// packed values themselves stays out of scope; this only sizes them.
    #[must_use]
    pub const fn block_layout(self) -> BlockLayout {
        const fn layout(block_elements: u64, block_bytes: u64) -> BlockLayout {
            BlockLayout {
                block_elements,
                block_bytes,
            }
        }
        match self {
            Self::F32 => layout(1, 4),
            Self::F16 | Self::Bf16 => layout(1, 2),
            Self::I8 => layout(1, 1),
            Self::I16 => layout(1, 2),
            Self::I32 => layout(1, 4),
            Self::I64 | Self::F64 => layout(1, 8),
            Self::Q4_0 | Self::Iq4Nl => layout(32, 18),
            Self::Q4_1 => layout(32, 20),
            Self::Q5_0 => layout(32, 22),
            Self::Q5_1 => layout(32, 24),
            Self::Q8_0 => layout(32, 34),
            Self::Q8_1 => layout(32, 36),
            Self::Q2_K => layout(256, 84),
            Self::Q3_K => layout(256, 110),
            Self::Q4_K => layout(256, 144),
            Self::Q5_K => layout(256, 176),
            Self::Q6_K => layout(256, 210),
            Self::Q8_K => layout(256, 292),
            Self::Iq2Xxs => layout(256, 66),
            Self::Iq2Xs => layout(256, 74),
            Self::Iq3Xxs => layout(256, 98),
            Self::Iq1S => layout(256, 50),
            Self::Iq3S => layout(256, 110),
            Self::Iq2S => layout(256, 82),
            Self::Iq4Xs => layout(256, 136),
            Self::Iq1M => layout(256, 56),
            Self::Tq10 => layout(256, 54),
            Self::Tq20 => layout(256, 66),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_type_round_trips_every_named_wire_value() {
        for raw in 0..=12u32 {
            assert!(MetadataType::from_wire(raw).is_some(), "raw={raw}");
        }
        assert_eq!(MetadataType::from_wire(13), None);
    }

    #[test]
    fn ggml_type_rejects_retired_gap_values() {
        for raw in [4, 5, 31, 32, 33, 36, 37, 38, 39, -1] {
            assert_eq!(GgmlType::from_wire(raw), None, "raw={raw}");
        }
    }

    #[test]
    fn ggml_type_block_layout_matches_known_sizes() {
        assert_eq!(
            GgmlType::F32.block_layout(),
            BlockLayout {
                block_elements: 1,
                block_bytes: 4
            }
        );
        assert_eq!(
            GgmlType::Q4_0.block_layout(),
            BlockLayout {
                block_elements: 32,
                block_bytes: 18
            }
        );
        assert_eq!(
            GgmlType::Q6_K.block_layout(),
            BlockLayout {
                block_elements: 256,
                block_bytes: 210
            }
        );
    }
}
