use serde::{Deserialize, Serialize};

/// What enters the first block: integer token ids looked up in a table of
/// `vocab` rows, or a dense feature vector of `width` values per position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Input {
    Tokens { vocab: u32 },
    Features { width: u32 },
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
        let (field, size) = match *self {
            Self::Tokens { vocab } => ("vocab", vocab),
            Self::Features { width } => ("width", width),
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
