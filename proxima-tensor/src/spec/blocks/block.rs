use serde::{Deserialize, Serialize};

use super::{Attention, DenseFfn, Norm, RoutedFfn, Stack};
use crate::op::ConstantLiteral;

/// One step of a network body, tagged by `kind` in config. A `Block` owns its
/// parameters; the variant set grows as each block kind lands.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Block {
    Norm(Norm),
    Attention(Attention),
    DenseFfn(DenseFfn),
    RoutedFfn(RoutedFfn),
    /// `x + scale * body(x)`: the stream joined to a sublayer's output, the
    /// sublayer scaled first when `scale` is set.
    Residual {
        scale: Option<ConstantLiteral>,
        body: Stack,
    },
}

#[cfg(feature = "config")]
impl conflaguration::Validate for Block {
    fn validate(&self) -> conflaguration::Result<()> {
        match self {
            Self::Norm(norm) => conflaguration::Validate::validate(norm),
            Self::Attention(attention) => conflaguration::Validate::validate(attention),
            Self::DenseFfn(ffn) => conflaguration::Validate::validate(ffn),
            Self::RoutedFfn(ffn) => conflaguration::Validate::validate(ffn),
            Self::Residual { body, .. } => conflaguration::Validate::validate(body),
        }
    }
}
