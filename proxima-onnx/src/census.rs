//! Op coverage census: how many nodes a graph has, which op types they use,
//! and which of those [`crate::lower_graph`] cannot lower.
//!
//! The census reads [`GraphProto`] directly and decides lowerability from
//! [`SUPPORTED_OPS`], the list that sits beside `lower_node`'s match and is
//! tested equal to that match's arms. It lowers nothing, so it is the cheap
//! first question to ask of a model before paying for decode and lowering;
//! compare [`crate::op_coverage`], which renders the same list as a doc.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;

use crate::lower::SUPPORTED_OPS;
use crate::messages::GraphProto;

/// What [`census`] counted in one graph, borrowing its op names from the
/// parsed bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Census<'a> {
    /// Node count of the graph, e.g. 1244 for BAAI/bge-small-en-v1.5.
    pub nodes: usize,
    /// Node count per op type, e.g. `("MatMul", 74)`.
    pub op_counts: BTreeMap<&'a str, usize>,
    /// Op types `lower_node` has no arm for, with their node counts, in
    /// name order, each name once.
    pub unlowerable: Vec<(&'a str, usize)>,
}

impl Census<'_> {
    /// Number of distinct op types in the graph.
    #[must_use]
    pub fn distinct(&self) -> usize {
        self.op_counts.len()
    }
}

/// Counts the nodes and op types of `graph` and names every op type outside
/// [`SUPPORTED_OPS`].
#[must_use]
pub fn census<'a>(graph: &GraphProto<'a>) -> Census<'a> {
    let mut op_counts: BTreeMap<&'a str, usize> = BTreeMap::new();
    for node in &graph.node {
        *op_counts.entry(node.op_type).or_insert(0) += 1;
    }
    let unlowerable = op_counts
        .iter()
        .filter(|(op_type, _)| !SUPPORTED_OPS.contains(op_type))
        .map(|(&op_type, &count)| (op_type, count))
        .collect();
    Census {
        nodes: graph.node.len(),
        op_counts,
        unlowerable,
    }
}

/// Renders `census nodes=<n> distinct=<d> unlowerable=<u>`, then one
/// `unlowerable op=<name> nodes=<count>` line per unlowerable op type.
#[must_use]
pub fn render_census(census: &Census<'_>) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "census nodes={} distinct={} unlowerable={}",
        census.nodes,
        census.distinct(),
        census.unlowerable.len()
    );
    for (op_type, count) in &census.unlowerable {
        let _ = writeln!(out, "unlowerable op={op_type} nodes={count}");
    }
    out
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::{census, render_census};
    use crate::lower::{LowerError, SUPPORTED_OPS, lower_graph};
    use crate::messages::{GraphProto, NodeProto};
    use crate::op_coverage::parse_supported_ops;
    #[cfg(feature = "std")]
    use crate::pipe::parse_complete;

    const LOWER_RS_SOURCE: &str = include_str!("lower.rs");

    fn node(op_type: &'static str, name: &'static str) -> NodeProto<'static> {
        NodeProto {
            input: vec!["x"],
            output: vec!["y"],
            op_type,
            name,
            ..NodeProto::default()
        }
    }

    fn graph_of(nodes: Vec<NodeProto<'static>>) -> GraphProto<'static> {
        GraphProto {
            node: nodes,
            name: "census_graph",
            ..GraphProto::default()
        }
    }

    #[test]
    fn onnx_census_counts_ops_of_a_hand_encoded_graph() {
        let graph = graph_of(vec![
            node("MatMul", "query"),
            node("MatMul", "key"),
            node("Softmax", "scores"),
            node("Add", "residual"),
        ]);

        let counted = census(&graph);

        assert_eq!(counted.nodes, 4);
        assert_eq!(counted.distinct(), 3);
        assert_eq!(counted.op_counts.get("MatMul"), Some(&2));
        assert_eq!(counted.op_counts.get("Softmax"), Some(&1));
        assert_eq!(counted.op_counts.get("Add"), Some(&1));
        assert!(counted.unlowerable.is_empty());
    }

    #[test]
    fn onnx_census_names_every_unlowerable_op_once() {
        let mut nodes: Vec<NodeProto<'static>> = Vec::new();
        for _ in 0..3 {
            nodes.push(node("LSTM", "recurrent"));
            nodes.push(node("NonMaxSuppression", "detector"));
        }
        nodes.push(node("Add", "residual"));

        let counted = census(&graph_of(nodes));

        assert_eq!(
            counted.unlowerable,
            vec![("LSTM", 3), ("NonMaxSuppression", 3)]
        );
        assert_eq!(
            render_census(&counted),
            "census nodes=7 distinct=3 unlowerable=2\n\
             unlowerable op=LSTM nodes=3\n\
             unlowerable op=NonMaxSuppression nodes=3\n"
        );
    }

    #[test]
    fn onnx_census_supported_ops_equals_lower_node_arms() {
        let parsed = parse_supported_ops(LOWER_RS_SOURCE);

        assert_eq!(parsed.as_deref().ok(), Some(SUPPORTED_OPS));
    }

    #[test]
    fn onnx_census_degenerate_control_source_without_the_erf_arm_fails_equality() {
        let without_erf: alloc::string::String = LOWER_RS_SOURCE
            .lines()
            .filter(|line| !line.trim_start().starts_with("\"Erf\" =>"))
            .flat_map(|line| [line, "\n"])
            .collect();

        let parsed = parse_supported_ops(&without_erf);

        let Ok(ops) = parsed else {
            panic!("source without the Erf arm still has both parser markers");
        };
        assert_eq!(ops.len(), SUPPORTED_OPS.len() - 1);
        assert_ne!(ops.as_slice(), SUPPORTED_OPS);
    }

    #[test]
    fn onnx_census_empty_graph_reports_zero() {
        let counted = census(&graph_of(Vec::new()));

        assert_eq!(counted.nodes, 0);
        assert_eq!(counted.distinct(), 0);
        assert_eq!(
            render_census(&counted),
            "census nodes=0 distinct=0 unlowerable=0\n"
        );
    }

    #[test]
    fn onnx_census_unlowerable_agrees_with_lower_graph() {
        let graph = graph_of(vec![node("LSTM", "recurrent"), node("LSTM", "second")]);

        let counted = census(&graph);
        let lowered = lower_graph(&graph);

        let Err(LowerError::UnsupportedOp { op_type, .. }) = lowered else {
            panic!("lower_graph must reject an LSTM graph with UnsupportedOp");
        };
        let census_names: Vec<&str> = counted.unlowerable.iter().map(|(name, _)| *name).collect();
        assert_eq!(census_names, vec![op_type.as_str()]);
    }

    #[cfg(feature = "std")]
    fn census_line_of(path: &str) -> alloc::string::String {
        let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("read {path}: {error}"));
        let Ok(model) = parse_complete(&bytes) else {
            panic!("{path} must parse as a ModelProto");
        };
        let Some(graph) = model.graph else {
            panic!("{path} must carry a graph");
        };
        render_census(&census(&graph))
    }

    #[cfg(feature = "std")]
    fn bge_model_path() -> alloc::string::String {
        std::env::var("BGE_MODEL_PATH").unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_default();
            alloc::format!(
                "{home}/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/onnx/model.onnx"
            )
        })
    }

    #[cfg(feature = "std")]
    #[test]
    fn onnx_census_bge_small_reports_1244_nodes_19_distinct_0_unlowerable() {
        let rendered = census_line_of(&bge_model_path());

        std::println!("{rendered}");
        assert_eq!(rendered, "census nodes=1244 distinct=19 unlowerable=0\n");
    }

    #[cfg(feature = "std")]
    #[test]
    fn onnx_census_mnist_reports_13_nodes_6_distinct_0_unlowerable() {
        let path = std::env::var("MNIST_MODEL").unwrap_or_else(|_| {
            "/Users/brianbruggeman/repos/others/burn/examples/onnx-inference/src/model/mnist.onnx"
                .into()
        });

        let rendered = census_line_of(&path);

        std::println!("{rendered}");
        assert_eq!(rendered, "census nodes=13 distinct=6 unlowerable=0\n");
    }
}
