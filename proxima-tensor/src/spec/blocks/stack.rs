use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use super::Block;

/// Blocks applied in order; each block's output width is the next block's input width.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "config", derive(bon::Builder, conflaguration::Settings))]
#[cfg_attr(feature = "config", settings(prefix = "PROXIMA_STACK"))]
#[serde(deny_unknown_fields)]
pub struct Stack {
    /// Applied first to last.
    #[cfg_attr(feature = "config", setting(skip))]
    pub blocks: Vec<Block>,
}

#[cfg(feature = "config")]
impl conflaguration::Validate for Stack {
    fn validate(&self) -> conflaguration::Result<()> {
        self.blocks
            .iter()
            .try_for_each(conflaguration::Validate::validate)
    }
}
