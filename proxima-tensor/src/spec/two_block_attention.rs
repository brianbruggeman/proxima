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
    let q_even_grouped = elementwise_like(
        program,
        rotated_q_even,
        ScalarOp::Multiply,
        &[
            (rotated_q_even, group_map.as_str()),
            (group_ones, "ug->sugi"),
        ],
    )?;
    let q_odd_grouped = elementwise_like(
        program,
        rotated_q_odd,
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
/// `key_reads` are the two operand maps over the cache, one per half of the pair: two planes read `tui`, one row reads the halves of one leaf.
#[expect(
    clippy::too_many_arguments,
    reason = "the two key reads are the layout; folding them into a struct would be a type that only carries arguments"
)]
fn append_cached_block_scores_with_reads(
    program: &mut Vec<Op>,
    dtype: DType,
    q_even_grouped: NodeId,
    q_odd_grouped: NodeId,
    k_even_cache: NodeId,
    k_odd_cache: NodeId,
    inv_sqrt_head_dim: NodeId,
    cache_mask: Option<(NodeId, NodeId)>,
    key_reads: (&str, &str),
) -> Result<NodeId, TensorError> {
    let score_cached_even_product = elementwise_like(
        program,
        q_even_grouped,
        ScalarOp::Multiply,
        &[(q_even_grouped, "sugi->stugi"), (k_even_cache, key_reads.0)],
    )?;
    let score_cached_even = reduce(
        program,
        dtype,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_cached_even_product,
        "stugi->stugi",
        "stug->stugi",
    )?;
    let score_cached_odd_product = elementwise_like(
        program,
        q_odd_grouped,
        ScalarOp::Multiply,
        &[(q_odd_grouped, "sugi->stugi"), (k_odd_cache, key_reads.1)],
    )?;
    let score_cached_odd = reduce(
        program,
        dtype,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_cached_odd_product,
        "stugi->stugi",
        "stug->stugi",
    )?;
    let score_cached = elementwise(
        program,
        dtype,
        ScalarOp::Add,
        &[
            (score_cached_even, "stug->stug"),
            (score_cached_odd, "stug->stug"),
        ],
    )?;
    let score_cached_scaled = elementwise(
        program,
        dtype,
        ScalarOp::Multiply,
        &[(score_cached, "stug->stug"), (inv_sqrt_head_dim, "->stug")],
    )?;
    match cache_mask {
        Some((is_masked, neg_infinity_cached)) => elementwise(
            program,
            dtype,
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
///
/// `attention_sink` is the layer's `blk.N.attn_sinks.weight` leaf when the
/// schedule sets the knob; `group` is the query-heads-per-key-head ratio the
/// sink's per-query-head index composes with, and is read only when a sink is
/// present.
#[expect(
    clippy::too_many_arguments,
    reason = "the combine reads both blocks' operands; folding them into a struct would be a type that only carries arguments"
)]
pub(super) fn append_local_block_and_combine_with_reads(
    program: &mut Vec<Op>,
    dtype: DType,
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
    neg_infinity_first: bool,
    attention_sink: Option<NodeId>,
    group: u32,
    key_reads: (&str, &str),
) -> Result<(NodeId, NodeId), TensorError> {
    let negative_infinity = |program: &mut Vec<Op>| scalar_constant(program, f32::NEG_INFINITY);
    let neg_infinity_local = match neg_infinity_local {
        None if neg_infinity_first => Some(negative_infinity(program)),
        given => given,
    };
    let score_new_even_product = elementwise_like(
        program,
        q_even_grouped,
        ScalarOp::Multiply,
        &[
            (q_even_grouped, "sugi->swugi"),
            (rotated_k_new_even, key_reads.0),
        ],
    )?;
    let score_new_even = reduce(
        program,
        dtype,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_new_even_product,
        "swugi->swugi",
        "swug->swugi",
    )?;
    let score_new_odd_product = elementwise_like(
        program,
        q_odd_grouped,
        ScalarOp::Multiply,
        &[
            (q_odd_grouped, "sugi->swugi"),
            (rotated_k_new_odd, key_reads.1),
        ],
    )?;
    let score_new_odd = reduce(
        program,
        dtype,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_new_odd_product,
        "swugi->swugi",
        "swug->swugi",
    )?;
    let score_new = elementwise(
        program,
        dtype,
        ScalarOp::Add,
        &[
            (score_new_even, "swug->swug"),
            (score_new_odd, "swug->swug"),
        ],
    )?;
    let score_new_scaled = elementwise(
        program,
        dtype,
        ScalarOp::Multiply,
        &[(score_new, "swug->swug"), (inv_sqrt_head_dim, "->swug")],
    )?;
    let neg_infinity = match neg_infinity_local {
        Some(node) => node,
        None => negative_infinity(program),
    };
    let score_new_masked = elementwise(
        program,
        dtype,
        ScalarOp::Select,
        &[
            (is_future_local, "sw->swug"),
            (neg_infinity, "->swug"),
            (score_new_scaled, "swug->swug"),
        ],
    )?;

    let score_max_cached = reduce(
        program,
        dtype,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        score_cached_scaled,
        "stug->stug",
        "sug->stug",
    )?;
    let score_max_new = reduce(
        program,
        dtype,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        score_new_masked,
        "swug->swug",
        "sug->swug",
    )?;
    let global_max = elementwise(
        program,
        dtype,
        ScalarOp::Maximum,
        &[(score_max_cached, "sug->sug"), (score_max_new, "sug->sug")],
    )?;
    let (global_max, sink_weight) = match attention_sink {
        Some(sink) => {
            let (global_max, sink_weight) =
                append_attention_sink_addend(program, sink, global_max, group)?;
            (global_max, Some(sink_weight))
        }
        None => (global_max, None),
    };

    let shifted_cached = elementwise(
        program,
        dtype,
        ScalarOp::Subtract,
        &[
            (score_cached_scaled, "stug->stug"),
            (global_max, "sug->stug"),
        ],
    )?;
    let weights_cached = elementwise(
        program,
        dtype,
        ScalarOp::Exponential,
        &[(shifted_cached, "stug->stug")],
    )?;
    let shifted_new = elementwise(
        program,
        dtype,
        ScalarOp::Subtract,
        &[(score_new_masked, "swug->swug"), (global_max, "sug->swug")],
    )?;
    let weights_new = elementwise(
        program,
        dtype,
        ScalarOp::Exponential,
        &[(shifted_new, "swug->swug")],
    )?;

    let sum_cached = reduce(
        program,
        dtype,
        ScalarOp::Add,
        ReduceInit::Zero,
        weights_cached,
        "stug->stug",
        "sug->stug",
    )?;
    let sum_new = reduce(
        program,
        dtype,
        ScalarOp::Add,
        ReduceInit::Zero,
        weights_new,
        "swug->swug",
        "sug->swug",
    )?;
    let weight_sum = elementwise(
        program,
        dtype,
        ScalarOp::Add,
        &[(sum_cached, "sug->sug"), (sum_new, "sug->sug")],
    )?;
    let weight_sum = add_attention_sink_weight(program, weight_sum, sink_weight)?;
    let inv_weight_sum = elementwise(
        program,
        dtype,
        ScalarOp::Reciprocal,
        &[(weight_sum, "sug->sug")],
    )?;

    let attended_cached_product = elementwise(
        program,
        dtype,
        ScalarOp::Multiply,
        &[(weights_cached, "stug->stugd"), (v_cache, "tud->stugd")],
    )?;
    let attended_cached = reduce(
        program,
        dtype,
        ScalarOp::Add,
        ReduceInit::Zero,
        attended_cached_product,
        "stugd->stugd",
        "sugd->stugd",
    )?;
    let attended_new_product = elementwise(
        program,
        dtype,
        ScalarOp::Multiply,
        &[(weights_new, "swug->swugd"), (v_new, "wud->swugd")],
    )?;
    let attended_new = reduce(
        program,
        dtype,
        ScalarOp::Add,
        ReduceInit::Zero,
        attended_new_product,
        "swugd->swugd",
        "sugd->swugd",
    )?;
    let attended_sum = elementwise(
        program,
        dtype,
        ScalarOp::Add,
        &[
            (attended_cached, "sugd->sugd"),
            (attended_new, "sugd->sugd"),
        ],
    )?;
    let attended = elementwise_like(
        program,
        q_even_grouped,
        ScalarOp::Multiply,
        &[(attended_sum, "sugd->sugd"), (inv_weight_sum, "sug->sugd")],
    )?;
    Ok((attended, score_max_cached))
}

/// Operand maps reading two key half planes `[t, u, i]` as the cached block.
pub const PLANE_CACHE_READS: (&str, &str) = ("tui->stugi", "tui->stugi");
/// Operand maps reading two key half planes `[w, u, i]` as the local block.
pub const PLANE_NEW_READS: (&str, &str) = ("wui->swugi", "wui->swugi");

/// Attends a query range over the cache and its own new keys with one online
/// softmax: groups the rotated query halves, scores the cached block, then the
/// local block, and returns `(attended, score_max_cached)`.
///
/// `q`, `k_new` and `k_cache` are the rotated (even, odd) halves; `cache_mask`
/// masks the cached block in the graph (`None` leaves it to the executor's
/// `cached_len`); `neg_infinity_local` and `attention_sink` are as in
/// [`append_local_block_and_combine_with_reads`]; `cache_reads` and `new_reads`
/// are the key operand maps over the cache and the new block.
/// `neg_infinity_first` emits a missing `-inf` before the local block's scores
/// instead of between them and their select, the order the single-range MoE
/// graph's digest pins.
#[expect(
    clippy::too_many_arguments,
    reason = "the two blocks read separate planes, masks and layouts; a struct would only carry arguments"
)]
pub fn two_range_attention(
    program: &mut Vec<Op>,
    dtype: DType,
    q: (NodeId, NodeId),
    k_new: (NodeId, NodeId),
    v_new: NodeId,
    k_cache: (NodeId, NodeId),
    v_cache: NodeId,
    inv_sqrt_head_dim: NodeId,
    cache_mask: Option<(NodeId, NodeId)>,
    is_future_local: NodeId,
    neg_infinity_local: Option<NodeId>,
    neg_infinity_first: bool,
    attention_sink: Option<NodeId>,
    group: u32,
    group_ones: NodeId,
    cache_reads: (&str, &str),
    new_reads: (&str, &str),
) -> Result<(NodeId, NodeId), TensorError> {
    let (q_even_grouped, q_odd_grouped) = group_queries(program, q.0, q.1, group, group_ones)?;
    let score_cached_scaled = append_cached_block_scores_with_reads(
        program,
        dtype,
        q_even_grouped,
        q_odd_grouped,
        k_cache.0,
        k_cache.1,
        inv_sqrt_head_dim,
        cache_mask,
        cache_reads,
    )?;
    append_local_block_and_combine_with_reads(
        program,
        dtype,
        q_even_grouped,
        q_odd_grouped,
        k_new.0,
        k_new.1,
        v_new,
        v_cache,
        inv_sqrt_head_dim,
        score_cached_scaled,
        is_future_local,
        neg_infinity_local,
        neg_infinity_first,
        attention_sink,
        group,
        new_reads,
    )
}
