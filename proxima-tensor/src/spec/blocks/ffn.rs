use alloc::collections::BTreeMap;
use alloc::string::String;

use serde::{Deserialize, Serialize};

/// The SwiGLU feed-forward: gate and up projections (separate, or one paired
/// `[2, feed_forward, embedding]` leaf), silu, down projection. The residual
/// add belongs to the enclosing [`Block::Residual`](super::Block::Residual).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "config", derive(bon::Builder))]
#[serde(deny_unknown_fields)]
pub struct DenseFfn {
    /// Hidden width.
    pub feed_forward: u32,
    /// Whether gate and up are read from one paired leaf.
    pub paired_gate_up: bool,
    /// Role key to full tensor name.
    pub leaves: BTreeMap<String, String>,
}

/// The routed mixture-of-experts feed-forward: a router over `expert_count`
/// experts, `expert_used_count` selected per token, softmax gating, silu.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "config", derive(bon::Builder))]
#[serde(deny_unknown_fields)]
pub struct RoutedFfn {
    /// Hidden width of one expert.
    pub feed_forward: u32,
    /// Experts the router chooses among.
    pub expert_count: u32,
    /// Experts selected per token.
    pub expert_used_count: u32,
    /// Role key to full tensor name.
    pub leaves: BTreeMap<String, String>,
}

#[cfg(feature = "config")]
impl conflaguration::Validate for DenseFfn {
    fn validate(&self) -> conflaguration::Result<()> {
        if self.feed_forward == 0 {
            return Err(conflaguration::Error::Validation {
                errors: alloc::vec![conflaguration::ValidationMessage::new(
                    "feed_forward",
                    "must be positive",
                )],
            });
        }
        Ok(())
    }
}

#[cfg(feature = "config")]
impl conflaguration::Validate for RoutedFfn {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = alloc::vec::Vec::new();
        if self.feed_forward == 0 {
            errors.push(conflaguration::ValidationMessage::new(
                "feed_forward",
                "must be positive",
            ));
        }
        if self.expert_used_count == 0 || self.expert_used_count > self.expert_count {
            errors.push(conflaguration::ValidationMessage::new(
                "expert_used_count",
                "must be positive and not exceed expert_count",
            ));
        }
        if errors.is_empty() {
            return Ok(());
        }
        Err(conflaguration::Error::Validation { errors })
    }
}
