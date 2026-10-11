use alloc::string::String;
use alloc::vec::Vec;

use crate::op::{NodeId, Op};

/// The result of lowering a parsed model: a self-contained `Op` program, the
/// initializer data it expects bound by name (via
/// `proxima_tensor::cpu::evaluate_named`), the graph's own declared inputs
/// (also bound by name), and its declared outputs as `(name, NodeId)` pairs
/// -- exactly `proxima_tensor::cpu::evaluate_named`'s `outputs` argument shape.
#[derive(Debug)]
pub struct Lowered {
    pub program: Vec<Op>,
    /// `(name, data)` for every initializer this graph declared, decoded to
    /// f32 -- see `decode_numeric_tensor` for the two payload shapes
    /// (`float_data` or little-endian `raw_data`) this pass understands.
    pub initializers: Vec<(String, Vec<f32>)>,
    pub graph_inputs: Vec<String>,
    pub graph_outputs: Vec<(String, NodeId)>,
    /// `(NodeId, onnx output tensor name)` for every ONNX `MatMul` node this
    /// graph declared, in `graph.node` order -- width-gate-decline task
    /// (2026-09-01): a `NodeId` alone ([`NodeId`]'s own doc: "index of an
    /// `Op` in the program slice") carries no ONNX semantics, and
    /// [`Op::Reduce`] never stores one either, so a
    /// per-`NodeId` decline table (`instrument::width_tile_decline_snapshot`)
    /// has no way to say "this NodeId is `encoder.layer.3.attention.self.key`"
    /// without this side table. Restricted to `MatMul` (not every node) since
    /// that is the only op family the decline census names.
    pub matmul_names: Vec<(NodeId, String)>,
}
