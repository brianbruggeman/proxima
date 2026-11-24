use alloc::collections::BTreeMap;
use alloc::string::String;

use serde::{Deserialize, Serialize};

use crate::spec::{AttentionScoreScale, RopePairing};

/// Grouped-query attention over a key/value cache: q/k/v projections (split
/// or fused, with optional biases and per-head QK-norm), rotary embedding,
/// the cached/new two-range softmax and the output projection. The residual
/// add belongs to the enclosing [`Block::Residual`](super::Block::Residual).
///
/// `leaves` maps a role key (`attn_q`, `attn_output`, ...) to the full tensor
/// name the checkpoint dialect spells it as, so lowering reads names from
/// data and never from a block's position in the stack.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "config", derive(bon::Builder))]
#[serde(deny_unknown_fields)]
pub struct Attention {
    /// Query head count.
    pub query_heads: u32,
    /// Key/value head count; divides `query_heads`.
    pub kv_heads: u32,
    /// Width of one head.
    pub head_dim: u32,
    /// How the rotary embedding pairs a head's channels.
    pub rope_pairing: RopePairing,
    /// How raw scores are scaled before the mask.
    pub score_scale: AttentionScoreScale,
    /// Sliding-window width, or None for full causal attention.
    pub window: Option<u32>,
    /// Whether per-head RMS norms run on q and k before the rotation.
    pub qk_norm: bool,
    /// Whether q, k and v carry bias vectors.
    pub biases: bool,
    /// Whether q, k and v come from one fused projection.
    pub fused_qkv: bool,
    /// Role key to full tensor name.
    pub leaves: BTreeMap<String, String>,
}

#[cfg(feature = "config")]
impl conflaguration::Validate for Attention {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = alloc::vec::Vec::new();
        let mut require = |holds: bool, field: &str, message: &str| {
            if !holds {
                errors.push(conflaguration::ValidationMessage::new(field, message));
            }
        };
        require(self.query_heads > 0, "query_heads", "must be positive");
        require(self.head_dim > 0, "head_dim", "must be positive");
        require(
            self.kv_heads > 0 && self.query_heads.is_multiple_of(self.kv_heads),
            "kv_heads",
            "must be positive and divide query_heads",
        );
        if errors.is_empty() {
            return Ok(());
        }
        Err(conflaguration::Error::Validation { errors })
    }
}
