//! The cached attention score algebra every two-range engine shares: the cached
//! block (a query range against already-rotated history) and the local block
//! (the same range against this call's own freshly rotated keys), combined by
//! one online softmax with no concatenation.
//!
//! The mask on the cached block is the only thing the engines disagree on, so it
//! is the one argument that selects an arm: `Some((is_masked, neg_infinity))`
//! masks the cached block in the graph, `None` leaves it to the executor's
//! `cached_len` bound (the fused cached attention excludes the bucket's zero
//! padding itself). A descriptor carries that choice as
//! [`CacheMask`](super::CacheMask); the layer builders
//! ([`append_gqa_cached_layer`], [`append_gqa_cached_routed_layer`],
//! [`append_two_range_cached_attention`]) own the projections, norms and
//! output stage in front of and behind this core.

use super::*;

/// Repeats each rotated query half across its key head's group, so one cached
/// or local key head scores `group` query heads: `[s, u, g, i]`.
pub(super) fn group_queries(
    program: &mut Vec<Op>,
    rotated_q_even: NodeId,
    rotated_q_odd: NodeId,
    group: u32,
    group_ones: NodeId,
) -> Result<(NodeId, NodeId), TensorError> {
    let group_map = alloc::format!("s,{group}*u+g,i->sugi");
    let q_even_grouped = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (rotated_q_even, group_map.as_str()),
            (group_ones, "ug->sugi"),
        ],
    )?;
    let q_odd_grouped = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (rotated_q_odd, group_map.as_str()),
            (group_ones, "ug->sugi"),
        ],
    )?;
    Ok((q_even_grouped, q_odd_grouped))
}

/// Scores the query range against every cached key `t` (symbol 1's extent, or
/// the sliding slot's), scaled by `inv_sqrt_head_dim`, then applies `cache_mask`
/// when the graph masks the cached block.
pub(super) fn append_cached_block_scores(
    program: &mut Vec<Op>,
    q_even_grouped: NodeId,
    q_odd_grouped: NodeId,
    k_even_cache: NodeId,
    k_odd_cache: NodeId,
    inv_sqrt_head_dim: NodeId,
    cache_mask: Option<(NodeId, NodeId)>,
) -> Result<NodeId, TensorError> {
    let score_cached_even_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (q_even_grouped, "sugi->stugi"),
            (k_even_cache, "tui->stugi"),
        ],
    )?;
    let score_cached_even = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_cached_even_product,
        "stugi->stugi",
        "stug->stugi",
    )?;
    let score_cached_odd_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(q_odd_grouped, "sugi->stugi"), (k_odd_cache, "tui->stugi")],
    )?;
    let score_cached_odd = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_cached_odd_product,
        "stugi->stugi",
        "stug->stugi",
    )?;
    let score_cached = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[
            (score_cached_even, "stug->stug"),
            (score_cached_odd, "stug->stug"),
        ],
    )?;
    let score_cached_scaled = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(score_cached, "stug->stug"), (inv_sqrt_head_dim, "->stug")],
    )?;
    match cache_mask {
        Some((is_masked, neg_infinity_cached)) => elementwise(
            program,
            DType::Float32,
            ScalarOp::Select,
            &[
                (is_masked, "st->stug"),
                (neg_infinity_cached, "->stug"),
                (score_cached_scaled, "stug->stug"),
            ],
        ),
        None => Ok(score_cached_scaled),
    }
}

/// Scores the local block, combines it with `score_cached` by one online
/// softmax, and returns `(attended, score_max_cached)`: the attended values
/// `[s, u, g, d]` and the cached block's row maximum (the first node of the
/// combine, which the instrumented census brackets).
///
/// `neg_infinity_local` is the caller's `-inf` constant for the causal mask;
/// `None` emits it here, between the local scores and their select, which is
/// where the dense builder always emitted it and where the program digests pin it.
#[expect(
    clippy::too_many_arguments,
    reason = "the combine reads both blocks' operands; folding them into a struct would be a type that only carries arguments"
)]
pub(super) fn append_local_block_and_combine(
    program: &mut Vec<Op>,
    q_even_grouped: NodeId,
    q_odd_grouped: NodeId,
    rotated_k_new_even: NodeId,
    rotated_k_new_odd: NodeId,
    v_new: NodeId,
    v_cache: NodeId,
    inv_sqrt_head_dim: NodeId,
    score_cached_scaled: NodeId,
    is_future_local: NodeId,
    neg_infinity_local: Option<NodeId>,
) -> Result<(NodeId, NodeId), TensorError> {
    let score_new_even_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (q_even_grouped, "sugi->swugi"),
            (rotated_k_new_even, "wui->swugi"),
        ],
    )?;
    let score_new_even = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_new_even_product,
        "swugi->swugi",
        "swug->swugi",
    )?;
    let score_new_odd_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (q_odd_grouped, "sugi->swugi"),
            (rotated_k_new_odd, "wui->swugi"),
        ],
    )?;
    let score_new_odd = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_new_odd_product,
        "swugi->swugi",
        "swug->swugi",
    )?;
    let score_new = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[
            (score_new_even, "swug->swug"),
            (score_new_odd, "swug->swug"),
        ],
    )?;
    let score_new_scaled = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(score_new, "swug->swug"), (inv_sqrt_head_dim, "->swug")],
    )?;
    let neg_infinity = match neg_infinity_local {
        Some(node) => node,
        None => scalar_constant(program, f32::NEG_INFINITY),
    };
    let score_new_masked = elementwise(
        program,
        DType::Float32,
        ScalarOp::Select,
        &[
            (is_future_local, "sw->swug"),
            (neg_infinity, "->swug"),
            (score_new_scaled, "swug->swug"),
        ],
    )?;

    let score_max_cached = reduce(
        program,
        DType::Float32,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        score_cached_scaled,
        "stug->stug",
        "sug->stug",
    )?;
    let score_max_new = reduce(
        program,
        DType::Float32,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        score_new_masked,
        "swug->swug",
        "sug->swug",
    )?;
    let global_max = elementwise(
        program,
        DType::Float32,
        ScalarOp::Maximum,
        &[(score_max_cached, "sug->sug"), (score_max_new, "sug->sug")],
    )?;

    let shifted_cached = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[
            (score_cached_scaled, "stug->stug"),
            (global_max, "sug->stug"),
        ],
    )?;
    let weights_cached = elementwise(
        program,
        DType::Float32,
        ScalarOp::Exponential,
        &[(shifted_cached, "stug->stug")],
    )?;
    let shifted_new = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(score_new_masked, "swug->swug"), (global_max, "sug->swug")],
    )?;
    let weights_new = elementwise(
        program,
        DType::Float32,
        ScalarOp::Exponential,
        &[(shifted_new, "swug->swug")],
    )?;

    let sum_cached = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        weights_cached,
        "stug->stug",
        "sug->stug",
    )?;
    let sum_new = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        weights_new,
        "swug->swug",
        "sug->swug",
    )?;
    let weight_sum = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(sum_cached, "sug->sug"), (sum_new, "sug->sug")],
    )?;
    let inv_weight_sum = elementwise(
        program,
        DType::Float32,
        ScalarOp::Reciprocal,
        &[(weight_sum, "sug->sug")],
    )?;

    let attended_cached_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(weights_cached, "stug->stugd"), (v_cache, "tud->stugd")],
    )?;
    let attended_cached = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        attended_cached_product,
        "stugd->stugd",
        "sugd->stugd",
    )?;
    let attended_new_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(weights_new, "swug->swugd"), (v_new, "wud->swugd")],
    )?;
    let attended_new = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        attended_new_product,
        "swugd->swugd",
        "sugd->swugd",
    )?;
    let attended_sum = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[
            (attended_cached, "sugd->sugd"),
            (attended_new, "sugd->sugd"),
        ],
    )?;
    let attended = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(attended_sum, "sugd->sugd"), (inv_weight_sum, "sug->sugd")],
    )?;
    Ok((attended, score_max_cached))
}
