use proxima_gguf::types::GgmlType;
use serde::{Deserialize, Serialize};

/// the values llama accepts for `--cache-type-k` and `--cache-type-v`, spelled
/// as llama spells them; `as_ggml` is the lowering to the type the serving
/// config holds. mirrored rather than derived on `GgmlType` because that enum
/// is foreign and `#[non_exhaustive]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheType {
    #[serde(rename = "f32")]
    F32,
    #[serde(rename = "f16")]
    F16,
    #[serde(rename = "bf16")]
    Bf16,
    #[serde(rename = "q8_0")]
    Q80,
    #[serde(rename = "q4_0")]
    Q40,
    #[serde(rename = "q4_1")]
    Q41,
    #[serde(rename = "q5_0")]
    Q50,
    #[serde(rename = "q5_1")]
    Q51,
    #[serde(rename = "iq4_nl")]
    Iq4Nl,
}

impl CacheType {
    pub const fn as_ggml(self) -> GgmlType {
        match self {
            Self::F32 => GgmlType::F32,
            Self::F16 => GgmlType::F16,
            Self::Bf16 => GgmlType::Bf16,
            Self::Q80 => GgmlType::Q8_0,
            Self::Q40 => GgmlType::Q4_0,
            Self::Q41 => GgmlType::Q4_1,
            Self::Q50 => GgmlType::Q5_0,
            Self::Q51 => GgmlType::Q5_1,
            Self::Iq4Nl => GgmlType::Iq4Nl,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serving_section_cache_type_round_trips_json() {
        let table = [
            ("f32", CacheType::F32, GgmlType::F32),
            ("f16", CacheType::F16, GgmlType::F16),
            ("bf16", CacheType::Bf16, GgmlType::Bf16),
            ("q8_0", CacheType::Q80, GgmlType::Q8_0),
            ("q4_0", CacheType::Q40, GgmlType::Q4_0),
            ("q4_1", CacheType::Q41, GgmlType::Q4_1),
            ("q5_0", CacheType::Q50, GgmlType::Q5_0),
            ("q5_1", CacheType::Q51, GgmlType::Q5_1),
            ("iq4_nl", CacheType::Iq4Nl, GgmlType::Iq4Nl),
        ];
        assert_eq!(table.len(), 9, "the table must list every cache type");

        for (word, variant, ggml) in table {
            let quoted = format!("\"{word}\"");
            let parsed: CacheType = serde_json::from_str(&quoted)
                .unwrap_or_else(|error| panic!("{quoted} must parse: {error}"));
            assert_eq!(parsed, variant, "{quoted} parses to its variant");
            let written = serde_json::to_string(&variant)
                .unwrap_or_else(|error| panic!("{word} must serialize: {error}"));
            assert_eq!(written, quoted, "{word} serializes back to the same word");
            assert_eq!(variant.as_ggml(), ggml, "{word} lowers to its ggml type");
        }

        for rejected in ["\"q9_9\"", "\"Q8_0\"", "\"q8\""] {
            assert!(
                serde_json::from_str::<CacheType>(rejected).is_err(),
                "{rejected} must be refused"
            );
        }
    }
}
