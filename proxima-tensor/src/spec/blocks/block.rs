use serde::{Deserialize, Serialize};

use super::Norm;

/// One step of a network body, tagged by `kind` in config. A `Block` owns its
/// parameters; the variant set grows as each block kind lands.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Block {
    Norm(Norm),
}

#[cfg(feature = "config")]
impl conflaguration::Validate for Block {
    fn validate(&self) -> conflaguration::Result<()> {
        match self {
            Self::Norm(norm) => conflaguration::Validate::validate(norm),
        }
    }
}
