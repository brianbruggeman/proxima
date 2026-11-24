use alloc::string::String;

use serde::{Deserialize, Serialize};

use super::{Norm, NormKind};
use crate::op::ConstantLiteral;

/// What leaves the last block: the final norm and the output projection.
/// `output` is the full tensor name the dialect spells the projection as.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "config", derive(bon::Builder))]
#[serde(deny_unknown_fields)]
pub struct Head {
    /// The norm applied to the final stream before projection.
    pub norm: Norm,
    /// Full tensor name of the vocabulary projection.
    pub output: String,
    /// Logits are divided by this when set (granite's logit scale).
    pub logit_scale: Option<ConstantLiteral>,
}

// a placeholder so `Network`'s skipped field has a Default; validate rejects it
impl Default for Head {
    fn default() -> Self {
        Self {
            norm: Norm {
                kind: NormKind::Rms,
                eps: None,
                weight: None,
                group_width: None,
            },
            output: String::new(),
            logit_scale: None,
        }
    }
}

#[cfg(feature = "config")]
impl conflaguration::Validate for Head {
    fn validate(&self) -> conflaguration::Result<()> {
        conflaguration::Validate::validate(&self.norm)?;
        if self.output.is_empty() {
            return Err(conflaguration::Error::Validation {
                errors: alloc::vec![conflaguration::ValidationMessage::new(
                    "output",
                    "must name the projection tensor",
                )],
            });
        }
        Ok(())
    }
}
