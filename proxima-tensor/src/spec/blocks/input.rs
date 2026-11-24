use alloc::string::String;

use serde::{Deserialize, Serialize};

use crate::spec::EmbeddingScale;
use proxima_numeric::NumericFormat;

/// What enters the first block: integer token ids looked up in a table of
/// `vocab` rows of `width` values (named `table`, optionally scaled, produced
/// at `activation`), or a dense feature vector of `width` values per position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
// one Input per network, built once at config time; a Box would only add an indirection
#[allow(clippy::large_enum_variant)]
pub enum Input {
    Tokens {
        vocab: u32,
        width: u32,
        table: String,
        #[serde(default)]
        scale: Option<EmbeddingScale>,
        activation: NumericFormat,
    },
    Features {
        width: u32,
    },
}

// conflaguration's skip requires Default; validate rejects this placeholder
impl Default for Input {
    fn default() -> Self {
        Self::Features { width: 0 }
    }
}

#[cfg(feature = "config")]
impl conflaguration::Validate for Input {
    fn validate(&self) -> conflaguration::Result<()> {
        let (field, size) = match self {
            Self::Tokens { vocab, width, .. } if *width > 0 => ("vocab", *vocab),
            Self::Tokens { .. } => ("width", 0),
            Self::Features { width } => ("width", *width),
        };
        if size == 0 {
            return Err(conflaguration::Error::Validation {
                errors: alloc::vec![conflaguration::ValidationMessage::new(
                    field,
                    "must be positive"
                )],
            });
        }
        Ok(())
    }
}
