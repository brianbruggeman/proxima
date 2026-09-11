//! The sigmoid-gated shared expert every Qwen-MoE-family checkpoint in this
//! crate runs alongside its routed [`proxima_tensor::spec::append_moe_ffn`]
//! FFN: `ffn_shexp = down(silu(gate(x)) * up(x))`, scaled by
//! `sigmoid(x @ gate_inp)` (a per-token scalar gate, the gate weight's own
//! `[embedding]` shape). One copy shared by [`crate::qwen4exp::program`]
//! (`pr27742.diff:3411-3432`) and [`crate::qwen35moe::program`] (confirmed
//! present on the real `qwen3.6:35b-a3b` checkpoint via `strings`,
//! `crate::qwen35moe::bind`'s own doc) rather than two duplicated builders --
//! both checkpoints carry the identical `ffn_gate_inp_shexp`/`ffn_gate_shexp`/
//! `ffn_up_shexp`/`ffn_down_shexp` tensor set.

use proxima_tensor::spec::{elementwise, reduce, sigmoid, silu};
use proxima_tensor::{DType, NodeId, Op, ReduceInit, ScalarOp, TensorError};

/// Composes [`elementwise`]/[`reduce`] for the two dense matvecs,
/// [`silu`]/[`sigmoid`] for the two activations -- every op here is one of
/// proxima's own public `spec` builders (this module's own doc names the
/// primitives), never a new one.
///
/// # Errors
///
/// [`TensorError`] if any composed op fails to lower (a shape mismatch
/// between `x` and the four weight tensors).
#[allow(clippy::too_many_arguments)]
pub fn append_sigmoid_gated_shared_expert(
    program: &mut Vec<Op>,
    x: NodeId,
    gate_inp_shexp: NodeId,
    gate_shexp: NodeId,
    up_shexp: NodeId,
    down_shexp: NodeId,
    one: NodeId,
) -> Result<NodeId, TensorError> {
    let gate_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(x, "sd->sdf"), (gate_shexp, "df->sdf")],
    )?;
    let gate_proj = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        gate_product,
        "sdf->sdf",
        "sf->sdf",
    )?;
    let gate_silu = silu(program, gate_proj, one, "sf->sf")?;

    let up_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(x, "sd->sdf"), (up_shexp, "df->sdf")],
    )?;
    let up_proj = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        up_product,
        "sdf->sdf",
        "sf->sdf",
    )?;

    let hidden = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(gate_silu, "sf->sf"), (up_proj, "sf->sf")],
    )?;

    let down_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(hidden, "sf->sfd"), (down_shexp, "fd->sfd")],
    )?;
    let ffn_shexp = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        down_product,
        "sfd->sfd",
        "sd->sfd",
    )?;

    let gate_logit_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(x, "sd->sd"), (gate_inp_shexp, "d->sd")],
    )?;
    let gate_logit = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        gate_logit_product,
        "sd->sd",
        "s->sd",
    )?;
    let shared_gate = sigmoid(program, gate_logit, one, "s->s")?;

    elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(ffn_shexp, "sd->sd"), (shared_gate, "s->sd")],
    )
}
