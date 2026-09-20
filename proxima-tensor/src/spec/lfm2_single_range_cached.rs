//! [`lfm2_forward_program_with_experts`]'s single-range-cached counterpart
//! for a schedule of ONLY [`LayerKind::Attention`] entries -- Gemma 4's own
//! shape (no `LayerKind::ShortConv`). Shares that function's own
//! [`build_attention_layer_resources`] pre-pass and
//! [`append_lfm2_layer_ffn`] post-attention/FFN composition verbatim;
//! [`append_lfm2_single_range_cached_attention`] is the one new piece,
//! [`single_range_moe_cached::append_mistral_single_range_cached_layer_with_biases`]'s
//! own merged-cache scoring generalized the same way
//! [`append_attention_mixer`] generalized the prefill mixer -- windowed
//! merged mask ([`causal_mask_merged_windowed`]), [`ValueSource`] (a
//! shared-KV layer has no `attn_v.weight`), [`AttentionScoreScale`], and
//! `value_norm`.
//!
//! Gated one level up (`proxima-model-interop`'s `gemma4-kv-cache`
//! feature, default-off): the functions here always compile, like every
//! other `spec::*_cached` module in this crate, since building a program
//! spec has no execution cost until a caller actually runs it -- only
//! `crate::gemma4::bind::Gemma4Arch::bind`'s own choice of which builder
//! to call is feature-gated.

use super::*;

/// [`append_attention_mixer`]'s single-range-cached counterpart: the same
/// per-layer knobs ([`ValueSource`], [`RopePairing`], `post_attention_norm`,
/// `value_norm`), but scored against a MERGED key/value cache
/// (`k_even_cache`/`k_odd_cache`/`v_cache`, symbol 1 -- every position this
/// call attends, prior positions and this call's own freshly rotated keys
/// NOT yet folded in) instead of this call's own freshly rotated
/// block-local keys, mirroring
/// `single_range_moe_cached::append_mistral_single_range_cached_layer_with_biases`'s
/// own single-softmax scoring shape. Returns `(post_mixer,
/// CachedLayerRoots)` -- this call's own freshly rotated
/// `(k_even, k_odd, v)` for the caller to fold into next call's merged
/// cache, the same 3-wide contract every other single-range cached layer
/// in this crate returns.
#[allow(clippy::too_many_arguments)]
pub fn append_lfm2_single_range_cached_attention(
    program: &mut Vec<Op>,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    inv_sqrt_head_dim: NodeId,
    inv_head_dim: NodeId,
    cos_new: NodeId,
    sin_new: NodeId,
    group_ones: NodeId,
    is_future: NodeId,
    neg_infinity: NodeId,
    group: u32,
    attn_norm_weight: NodeId,
    q_norm_weight: NodeId,
    k_norm_weight: NodeId,
    wq: NodeId,
    wk: NodeId,
    value_source: ValueSource,
    wo: NodeId,
    rope_pairing: RopePairing,
    post_attention_norm_weight: Option<NodeId>,
    value_norm: bool,
    k_even_cache: NodeId,
    k_odd_cache: NodeId,
    v_cache: NodeId,
) -> Result<(NodeId, CachedLayerRoots), TensorError> {
    let normed = rmsnorm(program, x, attn_norm_weight, inv_dim, eps)?;

    let q_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->shdi"), (wq, "ihd->shdi")],
    )?;
    let q_raw = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        q_product,
        "shdi->shdi",
        "shd->shdi",
    )?;
    let q = rmsnorm_per_head(program, q_raw, q_norm_weight, inv_head_dim, eps, "h")?;

    let k_new_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->sudi"), (wk, "iud->sudi")],
    )?;
    let k_new_raw = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        k_new_product,
        "sudi->sudi",
        "sud->sudi",
    )?;
    let k_new = rmsnorm_per_head(program, k_new_raw, k_norm_weight, inv_head_dim, eps, "u")?;

    let v_new_raw = match value_source {
        ValueSource::Projected(wv) => {
            let v_product = elementwise(
                program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(normed, "si->sudi"), (wv, "iud->sudi")],
            )?;
            reduce(
                program,
                DType::Float32,
                ScalarOp::Add,
                ReduceInit::Zero,
                v_product,
                "sudi->sudi",
                "sud->sudi",
            )?
        }
        ValueSource::SharedWithKey => k_new_raw,
        ValueSource::Shared(v) => v,
    };
    // Gemma 4's `v_norm` -- same weightless per-kv-head RMSNorm, no RoPE,
    // [`append_attention_mixer`]'s own doc walks through.
    let v_new = if value_norm {
        rmsnorm_per_head_no_scale(program, v_new_raw, inv_head_dim, eps, "u")?
    } else {
        v_new_raw
    };

    let (rotated_q_even, rotated_q_odd) = fused_rope_pair(program, q, 'h', cos_new, sin_new, rope_pairing)?;
    let (rotated_k_new_even, rotated_k_new_odd) =
        fused_rope_pair(program, k_new, 'u', cos_new, sin_new, rope_pairing)?;

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

    // Single-range: query `s` against the WHOLE merged key range `t`
    // (symbol 1), `k_even_cache`/`k_odd_cache` already carrying every
    // position this query may attend -- this call's own
    // `rotated_k_new_*`/`v_new` are computed above only to be returned as
    // this layer's own `CachedLayerRoots`, never read for THIS call's own
    // score (a query never attends a key that does not exist yet).
    let score_even_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (q_even_grouped, "sugi->stugi"),
            (k_even_cache, "tui->stugi"),
        ],
    )?;
    let score_even = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_even_product,
        "stugi->stugi",
        "stug->stugi",
    )?;
    let score_odd_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(q_odd_grouped, "sugi->stugi"), (k_odd_cache, "tui->stugi")],
    )?;
    let score_odd = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_odd_product,
        "stugi->stugi",
        "stug->stugi",
    )?;
    let scores = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(score_even, "stug->stug"), (score_odd, "stug->stug")],
    )?;
    let scores_scaled = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(scores, "stug->stug"), (inv_sqrt_head_dim, "->stug")],
    )?;
    let scores_masked = elementwise(
        program,
        DType::Float32,
        ScalarOp::Select,
        &[
            (is_future, "st->stug"),
            (neg_infinity, "->stug"),
            (scores_scaled, "stug->stug"),
        ],
    )?;

    let score_max = reduce(
        program,
        DType::Float32,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        scores_masked,
        "stug->stug",
        "sug->stug",
    )?;
    let shifted = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(scores_masked, "stug->stug"), (score_max, "sug->stug")],
    )?;
    let weights = elementwise(
        program,
        DType::Float32,
        ScalarOp::Exponential,
        &[(shifted, "stug->stug")],
    )?;
    let weight_sum = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        weights,
        "stug->stug",
        "sug->stug",
    )?;
    let inv_weight_sum = elementwise(
        program,
        DType::Float32,
        ScalarOp::Reciprocal,
        &[(weight_sum, "sug->sug")],
    )?;
    let probabilities = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(weights, "stug->stug"), (inv_weight_sum, "sug->stug")],
    )?;

    let attended_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(probabilities, "stug->stugd"), (v_cache, "tud->stugd")],
    )?;
    let attended = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        attended_product,
        "stugd->stugd",
        "sugd->stugd",
    )?;

    let wo_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(attended, "sugd->sugdo"), (wo, "ugdo->sugdo")],
    )?;
    let attn_out = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        wo_product,
        "sugdo->sugdo",
        "so->sugdo",
    )?;

    let attn_out = match post_attention_norm_weight {
        Some(gamma) => rmsnorm(program, attn_out, gamma, inv_dim, eps)?,
        None => attn_out,
    };

    let post_mixer = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(attn_out, "sd->sd"), (x, "sd->sd")],
    )?;

    Ok((post_mixer, (rotated_k_new_even, rotated_k_new_odd, v_new)))
}

/// [`causal_mask_merged_windowed`]'s cache-only counterpart:
/// [`append_lfm2_single_range_cached_attention`]'s own doc names the defect
/// this exists to close -- a merged single softmax has no way to tell "this
/// cache row is real history" apart from "this cache row is
/// `KvPadShape`-style zero-padding past `cached_len`, up to whatever bucket
/// boundary the caller rounded to" (`proxima-model-interop`'s
/// `KvPadScratch`'s own doc on that padding), since both read as ordinary
/// `k_even_cache`/`v_cache` values to a single merged score. This mask
/// closes that gap directly: `is_padding` (`key_index(t) > cached_len -
/// 1`, i.e. `t >= cached_len`) excludes every row at or past the real
/// cache boundary UNCONDITIONALLY, independent of the query -- unlike
/// [`causal_mask_merged`]'s own `is_future`, which admits `t ==
/// query_absolute` (this call's own row) precisely because a single merged
/// softmax needs that row real. [`append_lfm2_two_range_cached_attention`]
/// never needs `t == query_absolute` admitted here: that self/local range
/// is scored separately, against this call's own in-graph
/// `rotated_k_new_even`/`rotated_k_new_odd`/`v_new` (never round-tripped
/// through a cache leaf), the same two-block split
/// [`single_range_moe_cached::append_mistral_cached_moe_layer`]'s own
/// `score_cached`/`score_new` combine already established. `window`
/// composes the identical too-old distance check
/// [`causal_mask_merged_windowed`] uses, OR-ed onto `is_padding` by the
/// same [`ScalarOp::Maximum`] convention.
fn causal_mask_cached_windowed(
    program: &mut Vec<Op>,
    cached_len: NodeId,
    window: Option<u32>,
) -> Result<(NodeId, NodeId), TensorError> {
    // `query_index`/`query_absolute` are built unconditionally (not only on
    // the windowed branch): `is_padding` below needs a REAL, "s"-shaped
    // operand to seed the output's `s` axis (shape inference resolves each
    // output axis's extent from a concrete same-letter operand axis, never
    // from a pure scalar `->` broadcast -- two scalar-only operands leave an
    // axis genuinely `UnconstrainedDim`), and `query_index` is the cheapest
    // real "s"-shaped node available. Its VALUE cancels out of
    // `cached_len_ceiling_row` below (`query_absolute - query_index ==
    // cached_len` identically), so `is_padding` stays query-independent.
    let query_index = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Symbolic(0),
        },
    );
    let key_index = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Symbolic(1),
        },
    );
    let query_absolute = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(query_index, "s->s"), (cached_len, "->s")],
    )?;
    let cached_len_row = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(query_absolute, "s->s"), (query_index, "s->s")],
    )?;
    let one = scalar_constant(program, 1.0);
    let cached_len_ceiling_row = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(cached_len_row, "s->s"), (one, "->s")],
    )?;
    let is_padding = elementwise(
        program,
        DType::Float32,
        ScalarOp::Greater,
        &[(key_index, "t->st"), (cached_len_ceiling_row, "s->st")],
    )?;
    let neg_infinity = scalar_constant(program, f32::NEG_INFINITY);
    let Some(window) = window.filter(|&window| window > 0) else {
        return Ok((is_padding, neg_infinity));
    };
    let distance = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(query_absolute, "s->st"), (key_index, "t->st")],
    )?;
    let window_ceiling = scalar_constant(program, window as f32 - 1.0);
    let too_old = elementwise(
        program,
        DType::Float32,
        ScalarOp::Greater,
        &[(distance, "st->st"), (window_ceiling, "->st")],
    )?;
    let is_invalid = elementwise(
        program,
        DType::Float32,
        ScalarOp::Maximum,
        &[(is_padding, "st->st"), (too_old, "st->st")],
    )?;
    Ok((is_invalid, neg_infinity))
}

/// [`append_lfm2_single_range_cached_attention`]'s two-range counterpart:
/// the merged single-softmax read this module's own header doc names as
/// this crate's `gemma4-kv-cache` root cause (a single softmax has no
/// self-consistent way to include this call's own new positions in
/// `k_even_cache`/`k_odd_cache`/`v_cache` before they exist) is closed the
/// way every OTHER production cached engine in this crate already closes
/// it --
/// [`single_range_moe_cached::append_mistral_cached_moe_layer`]'s own
/// two-block online-softmax combine, generalized with the exact same
/// per-layer knobs [`append_lfm2_single_range_cached_attention`] already
/// threads ([`ValueSource`], [`RopePairing`], `post_attention_norm`,
/// `value_norm`, `causal_mask_cached_windowed` for the SWA cache-side
/// bound). The cache block scores ONLY genuine history
/// (`causal_mask_cached_windowed` excludes every padded row at or past
/// `cached_len`); the local block scores this call's own new positions
/// against its own in-graph `rotated_k_new_even`/`rotated_k_new_odd`/
/// `v_new` via the ordinary block-local [`causal_mask_windowed`] mask,
/// never round-tripped through a cache leaf -- so `k_even_cache`/
/// `k_odd_cache`/`v_cache` may be fed exactly what
/// `proxima-model-interop`'s existing growing-cache decode loop already
/// provides (real history for `[0, cached_len)`, zero padding past it),
/// with no caller-side pre-fold and no decode-loop change.
#[allow(clippy::too_many_arguments)]
pub fn append_lfm2_two_range_cached_attention(
    program: &mut Vec<Op>,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    inv_sqrt_head_dim: NodeId,
    inv_head_dim: NodeId,
    cos_new: NodeId,
    sin_new: NodeId,
    group_ones: NodeId,
    is_future_local: NodeId,
    neg_infinity_local: NodeId,
    is_future_cached: NodeId,
    neg_infinity_cached: NodeId,
    group: u32,
    attn_norm_weight: NodeId,
    q_norm_weight: NodeId,
    key_source: KeySource,
    wq: NodeId,
    value_source: ValueSource,
    wo: NodeId,
    rope_pairing: RopePairing,
    post_attention_norm_weight: Option<NodeId>,
    value_norm: bool,
    k_even_cache: NodeId,
    k_odd_cache: NodeId,
    v_cache: NodeId,
) -> Result<(NodeId, CachedLayerRoots), TensorError> {
    let normed = rmsnorm(program, x, attn_norm_weight, inv_dim, eps)?;

    let q_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->shdi"), (wq, "ihd->shdi")],
    )?;
    let q_raw = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        q_product,
        "shdi->shdi",
        "shd->shdi",
    )?;
    let q = rmsnorm_per_head(program, q_raw, q_norm_weight, inv_head_dim, eps, "h")?;

    // `KeySource::Shared` (gemma4 E2B's cross-layer shared-KV,
    // `KeySourceKind::SharedFromLayer(source)`): no `attn_k.weight`/
    // `attn_k_norm.weight` leaf exists for this layer at all, so its `K`
    // is the donor layer's own already-rotated, already-k-norm'd halves,
    // reused verbatim -- `k_new_raw` (unrotated K) is only meaningful for
    // `ValueSource::SharedWithKey` below, which gemma4 E2B's shared layers
    // never combine with `KeySource::Shared` (`KeySourceKind`'s own doc:
    // shared-KV always shares BOTH K and V), so `k_new_raw` need not exist
    // in this arm.
    let (k_new_raw_for_shared_with_key, rotated_k_new_even, rotated_k_new_odd) = match key_source {
        KeySource::Projected { wk, k_norm_weight } => {
            let k_new_product = elementwise(
                program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(normed, "si->sudi"), (wk, "iud->sudi")],
            )?;
            let k_new_raw = reduce(
                program,
                DType::Float32,
                ScalarOp::Add,
                ReduceInit::Zero,
                k_new_product,
                "sudi->sudi",
                "sud->sudi",
            )?;
            let k_new = rmsnorm_per_head(program, k_new_raw, k_norm_weight, inv_head_dim, eps, "u")?;
            let (rotated_even, rotated_odd) =
                fused_rope_pair(program, k_new, 'u', cos_new, sin_new, rope_pairing)?;
            (Some(k_new_raw), rotated_even, rotated_odd)
        }
        KeySource::Shared {
            rotated_even,
            rotated_odd,
        } => (None, rotated_even, rotated_odd),
    };

    let v_new_raw = match value_source {
        ValueSource::Projected(wv) => {
            let v_product = elementwise(
                program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(normed, "si->sudi"), (wv, "iud->sudi")],
            )?;
            reduce(
                program,
                DType::Float32,
                ScalarOp::Add,
                ReduceInit::Zero,
                v_product,
                "sudi->sudi",
                "sud->sudi",
            )?
        }
        ValueSource::SharedWithKey => k_new_raw_for_shared_with_key
            .ok_or(TensorError::SharedWithKeyRequiresProjectedKey)?,
        ValueSource::Shared(v) => v,
    };
    let v_new = if value_norm {
        rmsnorm_per_head_no_scale(program, v_new_raw, inv_head_dim, eps, "u")?
    } else {
        v_new_raw
    };

    let (rotated_q_even, rotated_q_odd) = fused_rope_pair(program, q, 'h', cos_new, sin_new, rope_pairing)?;

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

    // -- cache block: this query range against genuine history only
    // (`is_future_cached` excludes every padded row at or past
    // `cached_len`, and every real row it admits necessarily predates every
    // query in this call, so no additional future check is needed here).
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
    let score_cached_masked = elementwise(
        program,
        DType::Float32,
        ScalarOp::Select,
        &[
            (is_future_cached, "st->stug"),
            (neg_infinity_cached, "->stug"),
            (score_cached_scaled, "stug->stug"),
        ],
    )?;

    // -- local block: this call's own new positions against its own
    // in-graph rotated K/V, never round-tripped through a cache leaf.
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
    let score_new_masked = elementwise(
        program,
        DType::Float32,
        ScalarOp::Select,
        &[
            (is_future_local, "sw->swug"),
            (neg_infinity_local, "->swug"),
            (score_new_scaled, "swug->swug"),
        ],
    )?;

    // -- online-softmax combine, identical shape to
    // `append_mistral_cached_moe_layer`'s own two-block combine.
    let score_max_cached = reduce(
        program,
        DType::Float32,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        score_cached_masked,
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
            (score_cached_masked, "stug->stug"),
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

    let wo_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(attended, "sugd->sugdo"), (wo, "ugdo->sugdo")],
    )?;
    let attn_out = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        wo_product,
        "sugdo->sugdo",
        "so->sugdo",
    )?;

    let attn_out = match post_attention_norm_weight {
        Some(gamma) => rmsnorm(program, attn_out, gamma, inv_dim, eps)?,
        None => attn_out,
    };

    let post_mixer = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(attn_out, "sd->sd"), (x, "sd->sd")],
    )?;

    Ok((post_mixer, (rotated_k_new_even, rotated_k_new_odd, v_new)))
}

/// [`lfm2_forward_program_with_experts`]'s single-range-cached counterpart.
/// `ids`/`rope_cos`/`rope_sin` carry only the NEW positions this call
/// introduces (symbol 0); attention draws on a per-layer already-rotated
/// key/value cache sized by symbol 1 (`kv_cache.{layer}.k_even`/`k_odd`/`v`,
/// the same leaf-naming contract every other single-range cached engine in
/// this crate uses, so `proxima-model-interop`'s existing generic
/// cache-growing decode loop needs no change to drive this builder); the
/// returned roots are `(logits, per_layer_cache_roots, moe_sites)` instead
/// of a bare logits root. `cached_len` (a rank-0 `Op::Input`, zero on the
/// first call) feeds [`causal_mask_merged_windowed`] the same way
/// [`causal_mask_merged`] already does for the dense single-range engine.
///
/// Every `schedule` entry must be [`LayerKind::Attention`] -- a
/// [`LayerKind::ShortConv`] entry has no cache-state contract this builder
/// defines (see this module's own doc on why a conv-cached counterpart is
/// a further step neither this function nor
/// [`lfm2_forward_program_with_experts`] claims) and is rejected with
/// [`TensorError::UnsupportedInBuilder`] rather than silently mishandled.
#[allow(clippy::too_many_arguments)]
pub fn lfm2_single_range_cached_forward_program_with_experts(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    expert_feed_forward: u32,
    query_heads: u32,
    block_count: u32,
    expert_count: u32,
    expert_used_count: u32,
    leading_dense_block_count: u32,
    schedule: &[LayerSchedule],
    embedding_scale: Option<EmbeddingScale>,
    logit_softcap: Option<f32>,
    last_row_only: bool,
) -> Result<(Vec<Op>, NodeId, Vec<CachedLayerRoots>, MoeSites), TensorError> {
    if schedule.len() != block_count as usize {
        return Err(TensorError::LayerScheduleCountMismatch {
            expected: block_count,
            found: schedule.len(),
        });
    }
    if schedule.iter().any(|entry| entry.kind != LayerKind::Attention) {
        return Err(TensorError::UnsupportedInBuilder {
            builder: "lfm2_single_range_cached_forward_program_with_experts",
            feature: "LayerKind::ShortConv (no single-range cache-state contract yet)",
        });
    }

    let mut program = Vec::new();

    let ids = input_leaf(
        &mut program,
        DType::Int32,
        alloc::vec![Extent::Symbolic(0)],
        "ids",
    );
    let table = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(vocab), Extent::Static(embedding)],
        "token_embd.weight",
    );
    let mut x = embedding_lookup(&mut program, table, ids);
    if let Some(scale) = embedding_scale {
        let multiplier = match scale {
            EmbeddingScale::Sqrt => (embedding as f32).sqrt(),
        };
        let multiplier = scalar_constant(&mut program, multiplier);
        x = elementwise(
            &mut program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(x, "sd->sd"), (multiplier, "->sd")],
        )?;
    }

    let inv_dim = scalar_constant(&mut program, 1.0 / embedding as f32);
    let eps = symbolic_leaf(&mut program, DType::Float32, "eps");
    let ones = scalar_constant(&mut program, 1.0);
    let cached_len = input_leaf(&mut program, DType::Float32, Vec::new(), "cached_len");

    let attention_resources =
        build_attention_layer_resources(&mut program, schedule, query_heads, |program, window| {
            let is_future = causal_mask_merged_windowed(program, cached_len, window)?;
            let neg_infinity = scalar_constant(program, f32::NEG_INFINITY);
            Ok((is_future, neg_infinity))
        })?;

    let mut cache_roots: Vec<CachedLayerRoots> = Vec::with_capacity(block_count as usize);
    let mut moe_sites: Vec<MoeSite> = Vec::new();

    for (layer, entry) in schedule.iter().enumerate() {
        let layer = layer as u32;
        let config = &entry.attention;
        let ffn_config = &entry.ffn;
        let resources = &attention_resources[layer as usize];

        let head_dim = config.head_dim;
        let kv_heads = config.kv_heads;
        let pairs = head_dim / 2;
        let group = resources.group;

        let attn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.attn_norm.weight"),
        );
        let ffn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.ffn_norm.weight"),
        );
        let post_attention_norm_weight = if ffn_config.post_attention_norm {
            Some(input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding)],
                &alloc::format!("blk.{layer}.post_attention_norm.weight"),
            ))
        } else {
            None
        };

        let wq = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(embedding),
                Extent::Static(query_heads),
                Extent::Static(head_dim)
            ],
            &alloc::format!("blk.{layer}.attn_q.weight"),
        );
        let wk = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(embedding),
                Extent::Static(kv_heads),
                Extent::Static(head_dim)
            ],
            &alloc::format!("blk.{layer}.attn_k.weight"),
        );
        let value_source = match config.value_source_kind {
            ValueSourceKind::ProjectedV => {
                let wv = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Static(embedding),
                        Extent::Static(kv_heads),
                        Extent::Static(head_dim)
                    ],
                    &alloc::format!("blk.{layer}.attn_v.weight"),
                );
                ValueSource::Projected(wv)
            }
            ValueSourceKind::SharedWithKey => ValueSource::SharedWithKey,
            ValueSourceKind::SharedFromLayer(source) => {
                return Err(TensorError::SharedKvUnsupportedInCachedForward { layer, source_layer: source });
            }
        };
        let wo = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(kv_heads),
                Extent::Static(group),
                Extent::Static(head_dim),
                Extent::Static(embedding),
            ],
            &alloc::format!("blk.{layer}.attn_output.weight"),
        );
        let q_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(head_dim)],
            &alloc::format!("blk.{layer}.attn_q_norm.weight"),
        );
        let k_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(head_dim)],
            &alloc::format!("blk.{layer}.attn_k_norm.weight"),
        );

        let k_even_cache = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Symbolic(1),
                Extent::Static(kv_heads),
                Extent::Static(pairs)
            ],
            &alloc::format!("kv_cache.{layer}.k_even"),
        );
        let k_odd_cache = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Symbolic(1),
                Extent::Static(kv_heads),
                Extent::Static(pairs)
            ],
            &alloc::format!("kv_cache.{layer}.k_odd"),
        );
        let v_cache = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Symbolic(1),
                Extent::Static(kv_heads),
                Extent::Static(head_dim)
            ],
            &alloc::format!("kv_cache.{layer}.v"),
        );

        let (post_mixer, layer_roots) = append_lfm2_single_range_cached_attention(
            &mut program,
            x,
            inv_dim,
            eps,
            resources.inv_sqrt_head_dim,
            resources.inv_head_dim,
            resources.cos,
            resources.sin,
            resources.group_ones,
            resources.is_future,
            resources.neg_infinity,
            group,
            attn_norm_weight,
            q_norm_weight,
            k_norm_weight,
            wq,
            wk,
            value_source,
            wo,
            config.rope_pairing,
            post_attention_norm_weight,
            config.value_norm,
            k_even_cache,
            k_odd_cache,
            v_cache,
        )?;

        x = append_lfm2_layer_ffn(
            &mut program,
            layer,
            post_mixer,
            ffn_norm_weight,
            embedding,
            feed_forward,
            expert_feed_forward,
            expert_count,
            expert_used_count,
            leading_dense_block_count,
            ones,
            inv_dim,
            eps,
            ffn_config,
            // No `LayerKind::Attention`-only cached engine has ever needed
            // PLE (`FfnCombination::Exclusive` is this schedule kind's own
            // shape) -- `lfm2_forward_program_with_experts`'s own preamble
            // is the one caller that builds `ple_dim`/per-layer PLE input.
            None,
            0,
            &mut moe_sites,
        )?;

        cache_roots.push(layer_roots);
    }

    let output_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding)],
        "output_norm.weight",
    );
    let normed_final = rmsnorm(&mut program, x, output_norm_weight, inv_dim, eps)?;

    // Host-supplied gather leaf, not a `gather_last_row`-style in-program
    // reduction -- the same `lm_head_row` contract
    // `mistral_single_range_cached_forward_program`'s own `last_row_only`
    // arm uses, so this builder's decode-time binding stays identical to
    // every other single-range cached engine's.
    let normed_last = if last_row_only {
        let lm_head_row = input_leaf(
            &mut program,
            DType::Int32,
            alloc::vec![Extent::Static(1)],
            "lm_head_row",
        );
        embedding_lookup(&mut program, normed_final, lm_head_row)
    } else {
        normed_final
    };

    let lm_head = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding), Extent::Static(vocab)],
        "output.weight",
    );
    let logits_product = elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed_last, "sd->sdv"), (lm_head, "dv->sdv")],
    )?;
    let logits = reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        logits_product,
        "sdv->sdv",
        "sv->sdv",
    )?;

    let logits = match logit_softcap {
        Some(cap) => {
            let cap_node = scalar_constant(&mut program, cap);
            let scaled = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Divide,
                &[(logits, "sv->sv"), (cap_node, "->sv")],
            )?;
            let tanh = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Tanh,
                &[(scaled, "sv->sv")],
            )?;
            elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(tanh, "sv->sv"), (cap_node, "->sv")],
            )?
        }
        None => logits,
    };

    Ok((program, logits, cache_roots, MoeSites(moe_sites)))
}

/// [`lfm2_two_range_cached_forward_program_with_experts`]'s own per-layer
/// `stored_kv` entry: a donor (real cache-owning) layer's post-rope `K`
/// halves and post-norm `V` (the exact nodes gemma4 E2B's
/// `KeySourceKind::SharedFromLayer(source)`/`ValueSourceKind::SharedFromLayer(source)`
/// needs to reuse verbatim), plus that SAME donor's own already-declared
/// `k_even_cache`/`k_odd_cache`/`v_cache` `Op::Input` nodes -- a shared
/// layer's cache-range score reads these too, in-graph, rather than
/// declaring a second leaf for data the decode loop already feeds once.
#[derive(Debug, Clone, Copy)]
struct StoredSharedKv {
    rotated_k_even: NodeId,
    rotated_k_odd: NodeId,
    v_new: NodeId,
    k_even_cache: NodeId,
    k_odd_cache: NodeId,
    v_cache: NodeId,
}

/// [`lfm2_single_range_cached_forward_program_with_experts`]'s two-range
/// counterpart -- the fix for the divergence that function's OWN merged
/// single softmax cannot express (this module's own header doc): every
/// per-layer attention call here scores this call's own new positions
/// against its own in-graph `rotated_k_new_even`/`rotated_k_new_odd`/
/// `v_new` (see [`append_lfm2_two_range_cached_attention`]), never against
/// a cache leaf that does not hold them yet, so `kv_cache.{layer}.k_even`/
/// `k_odd`/`v` may be fed EXACTLY what `proxima-model-interop`'s existing
/// growing-cache decode loop already provides -- real history for
/// `[0, cached_len)`, `causal_mask_cached_windowed`-excluded padding past
/// it -- with no pre-fold and no decode-loop change. Same schedule/knob
/// contract as the single-range builder (every entry must be
/// [`LayerKind::Attention`]); the returned roots are the same
/// `(logits, per_layer_cache_roots, moe_sites)` shape.
#[allow(clippy::too_many_arguments)]
pub fn lfm2_two_range_cached_forward_program_with_experts(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    expert_feed_forward: u32,
    query_heads: u32,
    block_count: u32,
    expert_count: u32,
    expert_used_count: u32,
    leading_dense_block_count: u32,
    schedule: &[LayerSchedule],
    embedding_scale: Option<EmbeddingScale>,
    logit_softcap: Option<f32>,
    last_row_only: bool,
    // gemma4 E2B/E4B's per-layer-embedding preamble
    // (`lfm2_forward_program_with_experts`'s own `ple_dim` parameter doc) --
    // `None` for every checkpoint with no PLE tensors (12B/26B/31B), so this
    // builder's prior callers (none of whom ever passed a PLE-bearing
    // schedule) see no change in the emitted program.
    ple_dim: Option<u32>,
) -> Result<(Vec<Op>, NodeId, Vec<CachedLayerRoots>, MoeSites), TensorError> {
    if schedule.len() != block_count as usize {
        return Err(TensorError::LayerScheduleCountMismatch {
            expected: block_count,
            found: schedule.len(),
        });
    }
    if schedule.iter().any(|entry| entry.kind != LayerKind::Attention) {
        return Err(TensorError::UnsupportedInBuilder {
            builder: "lfm2_two_range_cached_forward_program_with_experts",
            feature: "LayerKind::ShortConv (no two-range cache-state contract yet)",
        });
    }

    let mut program = Vec::new();

    let ids = input_leaf(
        &mut program,
        DType::Int32,
        alloc::vec![Extent::Symbolic(0)],
        "ids",
    );
    let table = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(vocab), Extent::Static(embedding)],
        "token_embd.weight",
    );
    let mut x = embedding_lookup(&mut program, table, ids);
    if let Some(scale) = embedding_scale {
        let multiplier = match scale {
            EmbeddingScale::Sqrt => (embedding as f32).sqrt(),
        };
        let multiplier = scalar_constant(&mut program, multiplier);
        x = elementwise(
            &mut program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(x, "sd->sd"), (multiplier, "->sd")],
        )?;
    }

    let inv_dim = scalar_constant(&mut program, 1.0 / embedding as f32);
    let eps = symbolic_leaf(&mut program, DType::Float32, "eps");
    let ones = scalar_constant(&mut program, 1.0);
    let cached_len = input_leaf(&mut program, DType::Float32, Vec::new(), "cached_len");

    // Stage A preamble (`lfm2_forward_program_with_experts`'s own doc on
    // `ple_shared`/`ple_layer_inputs`, mirrored verbatim here): `x` is still
    // `h0`, the post-embedding-scale hidden state BEFORE the layer loop
    // below reassigns it -- the real input `append_ple_shared_projections`
    // needs, not any later layer's hidden state.
    let ple_shared = match ple_dim {
        Some(ple_dim) => Some(append_ple_shared_projections(
            &mut program,
            ids,
            x,
            vocab,
            embedding,
            ple_dim,
            ple_dim * block_count,
        )?),
        None => None,
    };
    let mut ple_layer_inputs: Vec<Option<NodeId>> = alloc::vec![None; block_count as usize];

    // Block-local windowed mask (this call's own new-vs-new range) --
    // dedup'd per unique `mask_window` by the shared resource pre-pass, the
    // same way the single-range builder's own `is_future` was.
    let attention_resources =
        build_attention_layer_resources(&mut program, schedule, query_heads, |program, window| {
            causal_mask_windowed(program, window)
        })?;

    // Cache-side mask (genuine history only) -- a second, separate dedup
    // over the same `mask_window` set, since `build_attention_layer_resources`
    // owns exactly one mask slot per layer and the local mask above already
    // claimed it.
    let mut cached_mask_cache: Vec<(Option<u32>, NodeId, NodeId)> = Vec::new();
    let mut cached_masks: Vec<(NodeId, NodeId)> = Vec::with_capacity(schedule.len());
    for entry in schedule {
        let window = entry.attention.mask_window;
        let found = cached_mask_cache
            .iter()
            .find(|(candidate, ..)| *candidate == window)
            .map(|(_, is_future, neg_infinity)| (*is_future, *neg_infinity));
        let pair = match found {
            Some(pair) => pair,
            None => {
                let built = causal_mask_cached_windowed(&mut program, cached_len, window)?;
                cached_mask_cache.push((window, built.0, built.1));
                built
            }
        };
        cached_masks.push(pair);
    }

    let mut cache_roots: Vec<CachedLayerRoots> = Vec::with_capacity(block_count as usize);
    let mut moe_sites: Vec<MoeSite> = Vec::new();
    // One slot per block, populated only for a layer that owns a real
    // `K`/`V` projection and its own `kv_cache.{layer}.*` leaves -- a later
    // `KeySourceKind::SharedFromLayer(source)`/`ValueSourceKind::SharedFromLayer(source)`
    // schedule entry (gemma4 E2B's cross-layer shared-KV) reads
    // `stored_kv[source]` instead of declaring its own leaves at all, the
    // same "never re-project, never re-declare a leaf" contract
    // `lfm2_forward_program_with_experts`'s own `stored_kv` already uses for
    // the cacheless path (`attention_forward.rs`'s own doc on it). See
    // [`StoredSharedKv`] for what each field carries.
    let mut stored_kv: Vec<Option<StoredSharedKv>> = alloc::vec![None; block_count as usize];

    for (layer, entry) in schedule.iter().enumerate() {
        let layer = layer as u32;
        let config = &entry.attention;
        let ffn_config = &entry.ffn;
        let resources = &attention_resources[layer as usize];
        let (is_future_cached, neg_infinity_cached) = cached_masks[layer as usize];

        let head_dim = config.head_dim;
        let kv_heads = config.kv_heads;
        let pairs = head_dim / 2;
        let group = resources.group;

        let attn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.attn_norm.weight"),
        );
        let ffn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.ffn_norm.weight"),
        );
        let post_attention_norm_weight = if ffn_config.post_attention_norm {
            Some(input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding)],
                &alloc::format!("blk.{layer}.post_attention_norm.weight"),
            ))
        } else {
            None
        };
        if let (Some(shared), Some(ple_dim)) = (&ple_shared, ple_dim) {
            ple_layer_inputs[layer as usize] =
                Some(ple_layer_input(&mut program, shared, eps, layer, ple_dim)?);
        }


        let wq = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(embedding),
                Extent::Static(query_heads),
                Extent::Static(head_dim)
            ],
            &alloc::format!("blk.{layer}.attn_q.weight"),
        );
        // gemma4 E2B shares BOTH `K` and `V` from the same donor layer
        // (`KeySourceKind::SharedFromLayer`'s own doc: ollama's own
        // `gemma4.go` reads one donor's `sharedHistory` for both, never `K`
        // alone) -- this builder's own cache-leaf declarations below are
        // gated on `key_source_kind` alone, so a schedule entry naming
        // `SharedFromLayer` on one axis but not the other is rejected here
        // rather than silently misdeclaring (or missing) a leaf.
        let shared_source = match (config.key_source_kind, config.value_source_kind) {
            (KeySourceKind::SharedFromLayer(key_source), ValueSourceKind::SharedFromLayer(value_source)) => {
                if key_source != value_source {
                    return Err(TensorError::SharedKvUnsupportedInCachedForward {
                        layer,
                        source_layer: value_source,
                    });
                }
                Some(key_source)
            }
            (KeySourceKind::SharedFromLayer(source), _) | (_, ValueSourceKind::SharedFromLayer(source)) => {
                return Err(TensorError::SharedKvUnsupportedInCachedForward { layer, source_layer: source });
            }
            (KeySourceKind::ProjectedK, _) => None,
        };

        let (key_source, value_source, k_even_cache, k_odd_cache, v_cache) = match shared_source {
            Some(source) => {
                let donor = stored_kv[source as usize]
                    .ok_or(TensorError::SharedKvSourceNotAvailable { layer, source_layer: source })?;
                (
                    KeySource::Shared {
                        rotated_even: donor.rotated_k_even,
                        rotated_odd: donor.rotated_k_odd,
                    },
                    ValueSource::Shared(donor.v_new),
                    donor.k_even_cache,
                    donor.k_odd_cache,
                    donor.v_cache,
                )
            }
            None => {
                let wk = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Static(embedding),
                        Extent::Static(kv_heads),
                        Extent::Static(head_dim)
                    ],
                    &alloc::format!("blk.{layer}.attn_k.weight"),
                );
                let k_norm_weight = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(head_dim)],
                    &alloc::format!("blk.{layer}.attn_k_norm.weight"),
                );
                let value_source = match config.value_source_kind {
                    ValueSourceKind::ProjectedV => {
                        let wv = input_leaf(
                            &mut program,
                            DType::Float32,
                            alloc::vec![
                                Extent::Static(embedding),
                                Extent::Static(kv_heads),
                                Extent::Static(head_dim)
                            ],
                            &alloc::format!("blk.{layer}.attn_v.weight"),
                        );
                        ValueSource::Projected(wv)
                    }
                    ValueSourceKind::SharedWithKey => ValueSource::SharedWithKey,
                    ValueSourceKind::SharedFromLayer(source) => {
                        return Err(TensorError::SharedKvUnsupportedInCachedForward { layer, source_layer: source });
                    }
                };
                let k_even_cache = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Symbolic(1),
                        Extent::Static(kv_heads),
                        Extent::Static(pairs)
                    ],
                    &alloc::format!("kv_cache.{layer}.k_even"),
                );
                let k_odd_cache = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Symbolic(1),
                        Extent::Static(kv_heads),
                        Extent::Static(pairs)
                    ],
                    &alloc::format!("kv_cache.{layer}.k_odd"),
                );
                let v_cache = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Symbolic(1),
                        Extent::Static(kv_heads),
                        Extent::Static(head_dim)
                    ],
                    &alloc::format!("kv_cache.{layer}.v"),
                );
                (
                    KeySource::Projected { wk, k_norm_weight },
                    value_source,
                    k_even_cache,
                    k_odd_cache,
                    v_cache,
                )
            }
        };

        let wo = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(kv_heads),
                Extent::Static(group),
                Extent::Static(head_dim),
                Extent::Static(embedding),
            ],
            &alloc::format!("blk.{layer}.attn_output.weight"),
        );
        let q_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(head_dim)],
            &alloc::format!("blk.{layer}.attn_q_norm.weight"),
        );

        let (post_mixer, layer_roots) = append_lfm2_two_range_cached_attention(
            &mut program,
            x,
            inv_dim,
            eps,
            resources.inv_sqrt_head_dim,
            resources.inv_head_dim,
            resources.cos,
            resources.sin,
            resources.group_ones,
            resources.is_future,
            resources.neg_infinity,
            is_future_cached,
            neg_infinity_cached,
            group,
            attn_norm_weight,
            q_norm_weight,
            key_source,
            wq,
            value_source,
            wo,
            config.rope_pairing,
            post_attention_norm_weight,
            config.value_norm,
            k_even_cache,
            k_odd_cache,
            v_cache,
        )?;

        if shared_source.is_none() {
            let (rotated_k_even, rotated_k_odd, v_new) = layer_roots;
            stored_kv[layer as usize] = Some(StoredSharedKv {
                rotated_k_even,
                rotated_k_odd,
                v_new,
                k_even_cache,
                k_odd_cache,
                v_cache,
            });
        }

        x = append_lfm2_layer_ffn(
            &mut program,
            layer,
            post_mixer,
            ffn_norm_weight,
            embedding,
            feed_forward,
            expert_feed_forward,
            expert_count,
            expert_used_count,
            leading_dense_block_count,
            ones,
            inv_dim,
            eps,
            ffn_config,
            ple_layer_inputs[layer as usize],
            ple_dim.unwrap_or(0),
            &mut moe_sites,
        )?;

        // A shared-KV layer owns no `kv_cache.{layer}.*` leaves of its own
        // (`stored_kv`'s own doc) -- `cache_roots` stays exactly the set of
        // real cache-owning layers, in layer order, the same "degenerate
        // default for a layer this engine's cache does not cover" shape
        // `descriptor::build_forward`'s own `CacheStrategy::Cacheless` arm
        // already sets for `cache_roots` as a whole.
        if shared_source.is_none() {
            cache_roots.push(layer_roots);
        }
    }

    let output_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding)],
        "output_norm.weight",
    );
    let normed_final = rmsnorm(&mut program, x, output_norm_weight, inv_dim, eps)?;

    let normed_last = if last_row_only {
        let lm_head_row = input_leaf(
            &mut program,
            DType::Int32,
            alloc::vec![Extent::Static(1)],
            "lm_head_row",
        );
        embedding_lookup(&mut program, normed_final, lm_head_row)
    } else {
        normed_final
    };

    let lm_head = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding), Extent::Static(vocab)],
        "output.weight",
    );
    let logits_product = elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed_last, "sd->sdv"), (lm_head, "dv->sdv")],
    )?;
    let logits = reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        logits_product,
        "sdv->sdv",
        "sv->sdv",
    )?;

    let logits = match logit_softcap {
        Some(cap) => {
            let cap_node = scalar_constant(&mut program, cap);
            let scaled = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Divide,
                &[(logits, "sv->sv"), (cap_node, "->sv")],
            )?;
            let tanh = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Tanh,
                &[(scaled, "sv->sv")],
            )?;
            elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(tanh, "sv->sv"), (cap_node, "->sv")],
            )?
        }
        None => logits,
    };

    Ok((program, logits, cache_roots, MoeSites(moe_sites)))
}
