//! The two wire-level enums GGUF stores as `int32_t`: the metadata value
//! type tag and the tensor element type tag.
//!
//! Layout taken from llama.cpp (`/Users/brianbruggeman/repos/others/llama.cpp`):
//! `ggml/include/gguf.h:53-68` (`enum gguf_type`) and
//! `ggml/include/ggml.h:352-391` (`enum ggml_type`).

use proxima_numeric::NumericFormat;

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
    /// Microscaling FP4: one E8M0 scale shared by 32 E2M1 values.
    Mxfp4,
    /// NVIDIA FP4: four E4M3 sub-block scales over 64 E2M1 values.
    Nvfp4,
    /// Legacy scalar 1-bit and 2-bit formats with layouts distinct from the
    /// K-quant family.
    Q1_0,
    Q2_0,
    /// PrismML's group-128 Q2_0 (`GGML_TYPE_PQ2_0`, id 142): one f16 scale per 128 weights,
    /// the same 2-bit code as [`Self::Q2_0`], 34 bytes per block. Not in [`Self::CURRENT`]:
    /// it is a fork id outside upstream ggml.
    Pq2_0,
    /// PrismML's group-128 ternary (`GGML_TYPE_PTQ1_0`, id 143): TQ1_0's base-3 trit packing with one
    /// f16 scale per 128 weights, 28 bytes per block. Not in [`Self::CURRENT`]: a fork id.
    Ptq1_0,
    /// Proxima-own: 32 weights as one f16 absmax scale plus 32 six-bit E3M2 codes,
    /// 26 bytes per block. Its wire id sits outside ggml's enum range, so a ggml
    /// consumer rejects the file instead of misreading the blocks; it is not in
    /// [`Self::CURRENT`].
    Fp6E3M2,
}

/// First wire id reserved for proxima-own tensor types, far above any id ggml
/// will reach. An own type's id is this base plus its `Codec::tag`.
pub const PROXIMA_OWN_WIRE_BASE: i32 = 0x1_0000;

const FP6_E3M2_WIRE: i32 = PROXIMA_OWN_WIRE_BASE + 32;

/// `GGML_TYPE_PQ2_0` in PrismML-Eng/llama.cpp (`ggml.h`, branch `prism`).
const PQ2_0_WIRE: i32 = 142;

/// `GGML_TYPE_PTQ1_0` in PrismML-Eng/llama.cpp (`ggml.h`, branch `prism`).
const PTQ1_0_WIRE: i32 = 143;

/// Per-type block shape: `block_elements` values are packed into
/// `block_bytes` bytes. Contiguous (non-quantized) types have
/// `block_elements == 1`. Sizes verified against the block struct layouts
/// in `ggml/src/ggml-common.h` (`block_q4_0` etc., lines 167-410) and the
/// `type_traits` table in `ggml/src/ggml.c:566-845`; `ggml_half` is
/// `uint16_t` (`ggml-common.h:6`), `QK_K` is 256 (`ggml-common.h:89`),
/// `K_SCALE_SIZE` is 12 (`ggml-common.h:90`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WireBlockShape {
    pub block_elements: u64,
    pub block_bytes: u64,
}

impl GgmlType {
    /// Every non-retired type in the current ggml wire enum. Keeping this
    /// inventory beside the wire decoder makes format coverage auditable and
    /// gives transforms a single source of truth instead of scattered
    /// `match` lists.
    pub const CURRENT: &'static [Self] = &[
        Self::F32,
        Self::F16,
        Self::Q4_0,
        Self::Q4_1,
        Self::Q5_0,
        Self::Q5_1,
        Self::Q8_0,
        Self::Q8_1,
        Self::Q2_K,
        Self::Q3_K,
        Self::Q4_K,
        Self::Q5_K,
        Self::Q6_K,
        Self::Q8_K,
        Self::Iq2Xxs,
        Self::Iq2Xs,
        Self::Iq3Xxs,
        Self::Iq1S,
        Self::Iq4Nl,
        Self::Iq3S,
        Self::Iq2S,
        Self::Iq4Xs,
        Self::I8,
        Self::I16,
        Self::I32,
        Self::I64,
        Self::F64,
        Self::Iq1M,
        Self::Bf16,
        Self::Tq10,
        Self::Tq20,
        Self::Mxfp4,
        Self::Nvfp4,
        Self::Q1_0,
        Self::Q2_0,
    ];

    /// Whether this type uses a packed block representation rather than one
    /// scalar per element. This is deliberately format-level, not backend
    /// capability-level: a registered type may still need a decoder/kernel.
    #[must_use]
    pub const fn is_quantized(self) -> bool {
        !matches!(
            self,
            Self::F32
                | Self::F16
                | Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
                | Self::F64
                | Self::Bf16
        )
    }

    /// Stable lower-case spelling used by GGUF tools and capability reports.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q8_0 => "q8_0",
            Self::Q8_1 => "q8_1",
            Self::Q2_K => "q2_k",
            Self::Q3_K => "q3_k",
            Self::Q4_K => "q4_k",
            Self::Q5_K => "q5_k",
            Self::Q6_K => "q6_k",
            Self::Q8_K => "q8_k",
            Self::Iq2Xxs => "iq2_xxs",
            Self::Iq2Xs => "iq2_xs",
            Self::Iq3Xxs => "iq3_xxs",
            Self::Iq1S => "iq1_s",
            Self::Iq4Nl => "iq4_nl",
            Self::Iq3S => "iq3_s",
            Self::Iq2S => "iq2_s",
            Self::Iq4Xs => "iq4_xs",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::F64 => "f64",
            Self::Iq1M => "iq1_m",
            Self::Bf16 => "bf16",
            Self::Tq10 => "tq1_0",
            Self::Tq20 => "tq2_0",
            Self::Mxfp4 => "mxfp4",
            Self::Nvfp4 => "nvfp4",
            Self::Q1_0 => "q1_0",
            Self::Q2_0 => "q2_0",
            Self::Pq2_0 => "pq2_0",
            Self::Ptq1_0 => "ptq1_0",
            Self::Fp6E3M2 => "fp6_e3m2",
        }
    }

    /// Decode the raw `ggml_type` wire tag (`enum ggml_type`, `ggml.h`).
    /// `None` for a value outside the current `GGML_TYPE_COUNT` (43) or one
    /// of the retired gap values.
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
            39 => Self::Mxfp4,
            40 => Self::Nvfp4,
            41 => Self::Q1_0,
            42 => Self::Q2_0,
            PQ2_0_WIRE => Self::Pq2_0,
            PTQ1_0_WIRE => Self::Ptq1_0,
            FP6_E3M2_WIRE => Self::Fp6E3M2,
            _ => return None,
        };
        Some(value)
    }

    /// The [`proxima_numeric::NumericFormat`] this wire type stores its elements as: the one
    /// `GgmlType` to format table. `F32` maps to `NumericFormat::Float32`, which has no wire tag of
    /// its own; a caller that wants only packed codecs filters it out. The integer types and `F64`
    /// have no named format constant, so they return `None`, as does a type this table has not met.
    #[must_use]
    pub const fn numeric_format(self) -> Option<NumericFormat> {
        Some(match self {
            Self::F32 => NumericFormat::Float32,
            Self::F16 => NumericFormat::Float16,
            Self::Bf16 => NumericFormat::BFloat16,
            Self::Q4_0 => NumericFormat::Q4_0,
            Self::Q4_1 => NumericFormat::Q4_1,
            Self::Q5_0 => NumericFormat::Q5_0,
            Self::Q5_1 => NumericFormat::Q5_1,
            Self::Q8_0 => NumericFormat::Q8_0,
            Self::Q8_1 => NumericFormat::Q8_1,
            Self::Q2_K => NumericFormat::Q2K,
            Self::Q3_K => NumericFormat::Q3K,
            Self::Q4_K => NumericFormat::Q4K,
            Self::Q5_K => NumericFormat::Q5K,
            Self::Q6_K => NumericFormat::Q6K,
            Self::Q8_K => NumericFormat::Q8K,
            Self::Iq2Xxs => NumericFormat::Iq2Xxs,
            Self::Iq2Xs => NumericFormat::Iq2Xs,
            Self::Iq3Xxs => NumericFormat::Iq3Xxs,
            Self::Iq1S => NumericFormat::Iq1S,
            Self::Iq4Nl => NumericFormat::Iq4Nl,
            Self::Iq3S => NumericFormat::Iq3S,
            Self::Iq2S => NumericFormat::Iq2S,
            Self::Iq4Xs => NumericFormat::Iq4Xs,
            Self::Iq1M => NumericFormat::Iq1M,
            Self::Tq10 => NumericFormat::Tq10,
            Self::Tq20 => NumericFormat::Tq20,
            Self::Mxfp4 => NumericFormat::Mxfp4,
            Self::Nvfp4 => NumericFormat::Nvfp4,
            Self::Q1_0 => NumericFormat::Q1_0,
            Self::Q2_0 => NumericFormat::Q2_0,
            Self::Pq2_0 => NumericFormat::Pq2_0,
            Self::Ptq1_0 => NumericFormat::Ptq1_0,
            Self::Fp6E3M2 => NumericFormat::Fp6E3M2,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 | Self::F64 => return None,
        })
    }

    /// Every variant, including the fork and proxima-own ones [`Self::CURRENT`] leaves out. It is
    /// an inventory, not a mapping: [`Self::from_numeric_format`] searches it through
    /// [`Self::numeric_format`], so the format table stays the only `GgmlType` to format map.
    pub const ALL: &'static [Self] = &[
        Self::F32,
        Self::F16,
        Self::Q4_0,
        Self::Q4_1,
        Self::Q5_0,
        Self::Q5_1,
        Self::Q8_0,
        Self::Q8_1,
        Self::Q2_K,
        Self::Q3_K,
        Self::Q4_K,
        Self::Q5_K,
        Self::Q6_K,
        Self::Q8_K,
        Self::Iq2Xxs,
        Self::Iq2Xs,
        Self::Iq3Xxs,
        Self::Iq1S,
        Self::Iq4Nl,
        Self::Iq3S,
        Self::Iq2S,
        Self::Iq4Xs,
        Self::I8,
        Self::I16,
        Self::I32,
        Self::I64,
        Self::F64,
        Self::Iq1M,
        Self::Bf16,
        Self::Tq10,
        Self::Tq20,
        Self::Mxfp4,
        Self::Nvfp4,
        Self::Q1_0,
        Self::Q2_0,
        Self::Pq2_0,
        Self::Ptq1_0,
        Self::Fp6E3M2,
    ];

    /// The wire type that stores `format`: the reverse of [`Self::numeric_format`], found by
    /// searching [`Self::ALL`] for the variant whose format equals `format`, so no second table
    /// can drift from the forward one. `None` for a format no ggml type stores (`BFloat8`,
    /// `Bf8E5M2`, `Bf4E2M1`, `Q8_1Grouped`). Cold path: it builds one format per probe.
    #[must_use]
    pub fn from_numeric_format(format: &NumericFormat) -> Option<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|candidate| candidate.numeric_format().as_ref() == Some(format))
    }

    /// Encode back to the raw `ggml_type` wire tag — the writer's half of
    /// [`Self::from_wire`]. Kept as an explicit match (not a discriminant
    /// cast) because the wire values have gaps (4, 5, 31-33, 36-38) that
    /// don't line up with this enum's declaration order.
    #[must_use]
    pub const fn to_wire(self) -> i32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2_K => 10,
            Self::Q3_K => 11,
            Self::Q4_K => 12,
            Self::Q5_K => 13,
            Self::Q6_K => 14,
            Self::Q8_K => 15,
            Self::Iq2Xxs => 16,
            Self::Iq2Xs => 17,
            Self::Iq3Xxs => 18,
            Self::Iq1S => 19,
            Self::Iq4Nl => 20,
            Self::Iq3S => 21,
            Self::Iq2S => 22,
            Self::Iq4Xs => 23,
            Self::I8 => 24,
            Self::I16 => 25,
            Self::I32 => 26,
            Self::I64 => 27,
            Self::F64 => 28,
            Self::Iq1M => 29,
            Self::Bf16 => 30,
            Self::Tq10 => 34,
            Self::Tq20 => 35,
            Self::Mxfp4 => 39,
            Self::Nvfp4 => 40,
            Self::Q1_0 => 41,
            Self::Q2_0 => 42,
            Self::Pq2_0 => PQ2_0_WIRE,
            Self::Ptq1_0 => PTQ1_0_WIRE,
            Self::Fp6E3M2 => FP6_E3M2_WIRE,
        }
    }

    /// Block layout used to compute a tensor's exact byte footprint, the
    /// same arithmetic as `ggml_nbytes` (`ggml.c`, via `type_traits`
    /// `blck_size`/`type_size`, `ggml.c:1174,1178`). Dequantizing the
    /// packed values themselves stays out of scope; this only sizes them.
    /// Derived from [`Self::numeric_format`] (`NumericFormat::elements_per_block` and
    /// `bytes_per_block`); only the integer types and `F64`, which have no format, are spelled here.
    #[must_use]
    pub const fn wire_shape(self) -> WireBlockShape {
        let (block_elements, block_bytes) = match self.numeric_format() {
            Some(format) => (format.elements_per_block() as u64, format.bytes_per_block() as u64),
            None => match self {
                Self::I8 => (1, 1),
                Self::I16 => (1, 2),
                Self::I32 => (1, 4),
                Self::I64 | Self::F64 => (1, 8),
                // unreachable for a type with a format; the recorded-table test fails if a new variant lands here
                _ => (1, 1),
            },
        };
        WireBlockShape {
            block_elements,
            block_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The arms of proxima-model-interop's `codec_from_ggml_type` at base commit 994a3d9b
    /// (bind.rs:1241), `Codec::` read as `NumericFormat::`, plus `F32`, the one row the single
    /// table added.
    static BASE_COMMIT_TABLE: [(GgmlType, NumericFormat); 33] = [
        (GgmlType::Q4_K, NumericFormat::Q4K),
        (GgmlType::Q5_K, NumericFormat::Q5K),
        (GgmlType::Q6_K, NumericFormat::Q6K),
        (GgmlType::Q8_0, NumericFormat::Q8_0),
        (GgmlType::Q3_K, NumericFormat::Q3K),
        (GgmlType::Q4_0, NumericFormat::Q4_0),
        (GgmlType::F16, NumericFormat::Float16),
        (GgmlType::Bf16, NumericFormat::BFloat16),
        (GgmlType::Q2_K, NumericFormat::Q2K),
        (GgmlType::Q5_1, NumericFormat::Q5_1),
        (GgmlType::Q5_0, NumericFormat::Q5_0),
        (GgmlType::Q4_1, NumericFormat::Q4_1),
        (GgmlType::Q8_1, NumericFormat::Q8_1),
        (GgmlType::Q8_K, NumericFormat::Q8K),
        (GgmlType::Iq1S, NumericFormat::Iq1S),
        (GgmlType::Iq1M, NumericFormat::Iq1M),
        (GgmlType::Iq2Xxs, NumericFormat::Iq2Xxs),
        (GgmlType::Iq2Xs, NumericFormat::Iq2Xs),
        (GgmlType::Iq2S, NumericFormat::Iq2S),
        (GgmlType::Iq3Xxs, NumericFormat::Iq3Xxs),
        (GgmlType::Iq3S, NumericFormat::Iq3S),
        (GgmlType::Iq4Nl, NumericFormat::Iq4Nl),
        (GgmlType::Iq4Xs, NumericFormat::Iq4Xs),
        (GgmlType::Tq10, NumericFormat::Tq10),
        (GgmlType::Tq20, NumericFormat::Tq20),
        (GgmlType::Mxfp4, NumericFormat::Mxfp4),
        (GgmlType::Nvfp4, NumericFormat::Nvfp4),
        (GgmlType::Q1_0, NumericFormat::Q1_0),
        (GgmlType::Q2_0, NumericFormat::Q2_0),
        (GgmlType::Pq2_0, NumericFormat::Pq2_0),
        (GgmlType::Ptq1_0, NumericFormat::Ptq1_0),
        (GgmlType::Fp6E3M2, NumericFormat::Fp6E3M2),
        (GgmlType::F32, NumericFormat::Float32),
    ];

    #[test]
    fn numeric_format_maps_every_type_exactly_as_the_table_it_replaced() {
        for (ggml_type, expected) in BASE_COMMIT_TABLE {
            assert_eq!(ggml_type.numeric_format(), Some(expected), "{ggml_type:?}");
        }
        for unmapped in [GgmlType::I8, GgmlType::I16, GgmlType::I32, GgmlType::I64, GgmlType::F64] {
            assert_eq!(unmapped.numeric_format(), None, "{unmapped:?}");
        }
    }

    #[test]
    fn the_mapped_types_are_exactly_33() {
        let every_type: Vec<GgmlType> = (0..=160)
            .chain(PROXIMA_OWN_WIRE_BASE..PROXIMA_OWN_WIRE_BASE + 64)
            .filter_map(GgmlType::from_wire)
            .collect();

        let mapped = every_type.iter().filter(|ggml_type| ggml_type.numeric_format().is_some()).count();

        assert_eq!(every_type.len(), 38);
        assert_eq!(mapped, 33);
    }

    #[test]
    fn metadata_type_round_trips_every_named_wire_value() {
        for raw in 0..=12u32 {
            assert!(MetadataType::from_wire(raw).is_some(), "raw={raw}");
        }
        assert_eq!(MetadataType::from_wire(13), None);
    }

    #[test]
    fn ggml_type_rejects_retired_gap_values() {
        for raw in [4, 5, 31, 32, 33, 36, 37, 38, 43, -1] {
            assert_eq!(GgmlType::from_wire(raw), None, "raw={raw}");
        }
    }

    #[test]
    fn ggml_type_to_wire_round_trips_through_from_wire() {
        for raw in [
            0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            26, 27, 28, 29, 30, 34, 35, 39, 40, 41, 42,
        ] {
            let decoded =
                GgmlType::from_wire(raw).unwrap_or_else(|| panic!("raw={raw} should decode"));
            assert_eq!(decoded.to_wire(), raw, "raw={raw}");
        }
    }

    #[test]
    fn proxima_own_fp6_e3m2_wire_id_is_outside_ggml_and_round_trips() {
        let wire = GgmlType::Fp6E3M2.to_wire();
        assert!(wire >= PROXIMA_OWN_WIRE_BASE);
        assert_eq!(GgmlType::from_wire(wire), Some(GgmlType::Fp6E3M2));
        assert!(!GgmlType::CURRENT.contains(&GgmlType::Fp6E3M2));
    }

    #[test]
    fn current_inventory_round_trips_without_gaps() {
        assert_eq!(GgmlType::CURRENT.len(), 35);
        for &kind in GgmlType::CURRENT {
            assert_eq!(GgmlType::from_wire(kind.to_wire()), Some(kind));
            assert!(!kind.name().is_empty());
        }
    }

    #[test]
    fn ggml_type_block_layout_matches_known_sizes() {
        assert_eq!(
            GgmlType::F32.wire_shape(),
            WireBlockShape {
                block_elements: 1,
                block_bytes: 4
            }
        );
        assert_eq!(
            GgmlType::Q4_0.wire_shape(),
            WireBlockShape {
                block_elements: 32,
                block_bytes: 18
            }
        );
        assert_eq!(
            GgmlType::Q6_K.wire_shape(),
            WireBlockShape {
                block_elements: 256,
                block_bytes: 210
            }
        );
    }

    /// (elements, bytes) per block for every `GgmlType::ALL` variant, recorded from the hand-written
    /// `wire_shape` table before it was derived from `numeric_format`.
    static RECORDED_WIRE_SHAPES: [(GgmlType, u64, u64); 38] = [
        (GgmlType::F32, 1, 4),
        (GgmlType::F16, 1, 2),
        (GgmlType::Q4_0, 32, 18),
        (GgmlType::Q4_1, 32, 20),
        (GgmlType::Q5_0, 32, 22),
        (GgmlType::Q5_1, 32, 24),
        (GgmlType::Q8_0, 32, 34),
        (GgmlType::Q8_1, 32, 36),
        (GgmlType::Q2_K, 256, 84),
        (GgmlType::Q3_K, 256, 110),
        (GgmlType::Q4_K, 256, 144),
        (GgmlType::Q5_K, 256, 176),
        (GgmlType::Q6_K, 256, 210),
        (GgmlType::Q8_K, 256, 292),
        (GgmlType::Iq2Xxs, 256, 66),
        (GgmlType::Iq2Xs, 256, 74),
        (GgmlType::Iq3Xxs, 256, 98),
        (GgmlType::Iq1S, 256, 50),
        (GgmlType::Iq4Nl, 32, 18),
        (GgmlType::Iq3S, 256, 110),
        (GgmlType::Iq2S, 256, 82),
        (GgmlType::Iq4Xs, 256, 136),
        (GgmlType::I8, 1, 1),
        (GgmlType::I16, 1, 2),
        (GgmlType::I32, 1, 4),
        (GgmlType::I64, 1, 8),
        (GgmlType::F64, 1, 8),
        (GgmlType::Iq1M, 256, 56),
        (GgmlType::Bf16, 1, 2),
        (GgmlType::Tq10, 256, 54),
        (GgmlType::Tq20, 256, 66),
        (GgmlType::Mxfp4, 32, 17),
        (GgmlType::Nvfp4, 64, 36),
        (GgmlType::Q1_0, 128, 18),
        (GgmlType::Q2_0, 64, 18),
        (GgmlType::Pq2_0, 128, 34),
        (GgmlType::Ptq1_0, 128, 28),
        (GgmlType::Fp6E3M2, 32, 26),
    ];

    #[test]
    fn wire_shape_of_every_variant_matches_the_recorded_table() {
        assert_eq!(GgmlType::ALL.len(), 38);
        assert_eq!(RECORDED_WIRE_SHAPES.len(), GgmlType::ALL.len());
        for (ggml_type, block_elements, block_bytes) in RECORDED_WIRE_SHAPES {
            assert!(GgmlType::ALL.contains(&ggml_type), "{ggml_type:?} missing from ALL");
            assert_eq!(
                ggml_type.wire_shape(),
                WireBlockShape { block_elements, block_bytes },
                "{ggml_type:?}"
            );
        }
    }

    /// no wildcard: adding a variant fails to compile here until `ALL` is revisited.
    fn position_in_all(ggml_type: GgmlType) -> usize {
        match ggml_type {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::Q4_0 => 2,
            GgmlType::Q4_1 => 3,
            GgmlType::Q5_0 => 4,
            GgmlType::Q5_1 => 5,
            GgmlType::Q8_0 => 6,
            GgmlType::Q8_1 => 7,
            GgmlType::Q2_K => 8,
            GgmlType::Q3_K => 9,
            GgmlType::Q4_K => 10,
            GgmlType::Q5_K => 11,
            GgmlType::Q6_K => 12,
            GgmlType::Q8_K => 13,
            GgmlType::Iq2Xxs => 14,
            GgmlType::Iq2Xs => 15,
            GgmlType::Iq3Xxs => 16,
            GgmlType::Iq1S => 17,
            GgmlType::Iq4Nl => 18,
            GgmlType::Iq3S => 19,
            GgmlType::Iq2S => 20,
            GgmlType::Iq4Xs => 21,
            GgmlType::I8 => 22,
            GgmlType::I16 => 23,
            GgmlType::I32 => 24,
            GgmlType::I64 => 25,
            GgmlType::F64 => 26,
            GgmlType::Iq1M => 27,
            GgmlType::Bf16 => 28,
            GgmlType::Tq10 => 29,
            GgmlType::Tq20 => 30,
            GgmlType::Mxfp4 => 31,
            GgmlType::Nvfp4 => 32,
            GgmlType::Q1_0 => 33,
            GgmlType::Q2_0 => 34,
            GgmlType::Pq2_0 => 35,
            GgmlType::Ptq1_0 => 36,
            GgmlType::Fp6E3M2 => 37,
        }
    }

    #[test]
    fn all_lists_every_variant_exactly_once_in_declaration_order() {
        assert_eq!(GgmlType::ALL.len(), 38);
        for (index, ggml_type) in GgmlType::ALL.iter().enumerate() {
            assert_eq!(position_in_all(*ggml_type), index, "{ggml_type:?} position");
        }
    }

    #[test]
    fn numeric_format_is_injective_over_every_variant() {
        let formats: Vec<NumericFormat> = GgmlType::ALL
            .iter()
            .filter_map(|ggml_type| ggml_type.numeric_format())
            .collect();
        for (index, format) in formats.iter().enumerate() {
            assert!(
                !formats[index + 1..].contains(format),
                "two ggml types share the format at position {index}"
            );
        }
    }

    #[test]
    fn from_numeric_format_inverts_numeric_format_for_every_variant_with_a_format() {
        let mut inverted = 0;
        for ggml_type in GgmlType::ALL {
            let Some(format) = ggml_type.numeric_format() else {
                continue;
            };
            assert_eq!(GgmlType::from_numeric_format(&format), Some(*ggml_type), "{ggml_type:?}");
            inverted += 1;
        }
        assert_eq!(inverted, 33, "38 variants, 5 integer or f64 types with no format");
    }

    #[test]
    fn from_numeric_format_maps_float32_to_f32() {
        assert_eq!(GgmlType::from_numeric_format(&NumericFormat::Float32), Some(GgmlType::F32));
    }

    #[test]
    fn from_numeric_format_is_none_for_formats_no_ggml_type_stores() {
        for format in [
            NumericFormat::BFloat8,
            NumericFormat::Bf8E5M2,
            NumericFormat::Bf4E2M1,
            NumericFormat::Q8_1Grouped,
        ] {
            assert_eq!(GgmlType::from_numeric_format(&format), None, "{format:?}");
        }
    }
}
