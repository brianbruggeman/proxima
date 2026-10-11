use std::collections::{BTreeMap, BTreeSet};

use proxima_onnx::census::Census;
use proxima_onnx::messages::{DimensionValue, GraphProto, TypeValue};

/// Why the validate stage refused a graph before paying for lowering.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ValidationError {
    /// The model parsed but its `graph` field is absent.
    #[error("model carries no graph")]
    NoGraph,

    /// Every reason the graph cannot lower, gathered in one pass so a caller
    /// fixes them together instead of one lowering error at a time.
    #[error(
        "graph cannot lower: unlowerable ops (name, nodes) {unlowerable:?}, unpinned dimensions {unpinned:?}"
    )]
    Rejected {
        /// Op types with no lowering arm and their node counts, e.g.
        /// `[("DequantizeLinear", 12)]`.
        unlowerable: Vec<(String, usize)>,
        /// Symbolic input dimension names with no entry in the batch pins,
        /// e.g. `["batch_size", "sequence_length"]`.
        unpinned: Vec<String>,
    },
}

impl ValidationError {
    /// The rejection for a graph whose census found unlowerable ops or whose
    /// symbolic dimensions the pins leave open; `None` when neither holds.
    pub fn rejection(
        graph: &GraphProto<'_>,
        census: &Census<'_>,
        pins: &BTreeMap<String, u64>,
    ) -> Option<Self> {
        let unpinned = unpinned_dimensions(graph, pins);
        if census.unlowerable.is_empty() && unpinned.is_empty() {
            return None;
        }
        let unlowerable = census
            .unlowerable
            .iter()
            .map(|&(op_type, count)| (op_type.to_owned(), count))
            .collect();
        Some(Self::Rejected {
            unlowerable,
            unpinned,
        })
    }
}

fn unpinned_dimensions(graph: &GraphProto<'_>, pins: &BTreeMap<String, u64>) -> Vec<String> {
    graph
        .input
        .iter()
        .filter_map(|input| match input.r#type.as_ref()?.value.as_ref()? {
            TypeValue::Tensor(tensor) => tensor.shape.as_ref(),
            _ => None,
        })
        .flat_map(|shape| shape.dim.iter())
        .filter_map(|dimension| match &dimension.value {
            Some(DimensionValue::Param(name)) if !pins.contains_key(*name) => Some(*name),
            _ => None,
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(str::to_owned)
        .collect()
}
