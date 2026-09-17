use super::*;

/// [`append_mistral_cached_layer`]'s single-range counterpart: the SAME
/// function -- same RoPE, same GQA grouping, same `CachedLayerRoots`
/// return -- but scored through ONE softmax over ONE key axis instead of
/// two disjoint ranges combined by hand. `k_even_cache`/`k_odd_cache`/
/// `v_cache` are no longer "everything before this call"; they are the
/// WHOLE merged context this call attends to, prior positions AND this
/// call's own freshly rotated keys already folded in by the caller between
/// calls (`kv_cache.{layer}.*`'s own shape grows from `cached_len` to
/// `cached_len + new_count`, same [`Extent::Symbolic`] slot, no new op).
/// `rotated_k_new_even`/`rotated_k_new_odd`/`v_new` are STILL computed
/// in-graph from `x`, unchanged from [`append_mistral_cached_layer`] --
/// this call's own [`CachedLayerRoots`] the caller folds into next call's
/// merged cache -- they are simply no longer read for THIS call's own
/// score, since this call's own keys are not yet part of the merged range
/// it attends over (a query never attends a key that does not exist until
/// after it is computed).
///
/// Score/softmax/attended here are node-for-node
/// [`append_mistral_layer`]'s own single-range pattern (`score`,
/// `score_max`, `shifted`, `weights`, `weight_sum`, `inv_weight_sum`,
/// `probabilities`, `attended_product`, `attended`) rather than
/// [`append_mistral_cached_layer`]'s two-block online-softmax combine --
/// the entire point of this function existing next to that one.
/// `is_future` here must come from [`causal_mask_merged`], not
/// [`causal_mask`]: shape `[s, t]` with `t` sized by [`Extent::Symbolic`]
/// slot 1 (the merged range), not slot 0.
///
/// `gate_before_up` decides only which of the FFN's two independent matvecs
/// (`ffn_gate.weight`, `ffn_up.weight` -- both read `normed2`, neither reads
/// the other's output) is PUSHED into `program` first; `gate`/`up` are
/// returned bound the same way either way, so every downstream op
/// (`silu_gate`, `ffn_hidden`) is byte-identical regardless of this flag.
/// This exists to measure whether ROW 310/311's `ffn_gate`/`ffn_up`
/// bandwidth asymmetry is positional (first-vs-second in a barrier-free
/// sibling pair, see `omega/src/metal.rs`'s `HazardTracker`) rather than
/// per-kernel -- see this module's own
/// `swapping_gate_and_up_order_keeps_dataflow_identical` test.
///
/// `qk_norm` (ROW 373) is the SAME `Option<(NodeId, NodeId, NodeId)>` shape
/// as [`append_mistral_cached_layer`]'s own parameter of that name --
/// q-norm weight, k-norm weight, `inv_head_dim` -- applied through the same
/// [`rmsnorm_per_head`] calls before RoPE, and selects the same
/// interleaved-vs-split-half pairing that function's doc already derives
/// from `qk_norm.is_some()`. This builder no longer rejects a qk-norm
/// checkpoint; it now builds it, node-for-node the same attention block the
/// two-range sibling would.
#[allow(clippy::too_many_arguments)]
pub fn append_mistral_single_range_cached_layer(
    program: &mut Vec<Op>,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    ones: NodeId,
    inv_sqrt_head_dim: NodeId,
    cos_new: NodeId,
    sin_new: NodeId,
    group_ones: NodeId,
    is_future: NodeId,
    group: u32,
    head_dim: u32,
    attn_norm_weight: NodeId,
    ffn_norm_weight: NodeId,
    wq: NodeId,
    wk: NodeId,
    wv: NodeId,
    wo: NodeId,
    w_gate: NodeId,
    w_up: NodeId,
    w_down: NodeId,
    k_even_cache: NodeId,
    k_odd_cache: NodeId,
    v_cache: NodeId,
    qk_norm: Option<(NodeId, NodeId, NodeId)>,
    gate_before_up: bool,
) -> Result<(NodeId, CachedLayerRoots), TensorError> {
    append_mistral_single_range_cached_layer_with_biases(
        program,
        x,
        inv_dim,
        eps,
        ones,
        inv_sqrt_head_dim,
        cos_new,
        sin_new,
        group_ones,
        is_future,
        group,
        head_dim,
        attn_norm_weight,
        ffn_norm_weight,
        wq,
        wk,
        wv,
        wo,
        w_gate,
        w_up,
        w_down,
        k_even_cache,
        k_odd_cache,
        v_cache,
        qk_norm,
        None,
        gate_before_up,
    )
}

/// Bias-aware single-range dense layer builder. The optional tuple contains
/// Q/K/V projection biases, in the same shapes and order as the two-range
/// builder. Keeping the compatibility wrapper above preserves every existing
/// bias-disabled fixture while allowing checkpoint-driven callers to express
/// the complete projection semantics.
#[allow(clippy::too_many_arguments)]
pub fn append_mistral_single_range_cached_layer_with_biases(
    program: &mut Vec<Op>,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    ones: NodeId,
    inv_sqrt_head_dim: NodeId,
    cos_new: NodeId,
    sin_new: NodeId,
    group_ones: NodeId,
    is_future: NodeId,
    group: u32,
    head_dim: u32,
    attn_norm_weight: NodeId,
    ffn_norm_weight: NodeId,
    wq: NodeId,
    wk: NodeId,
    wv: NodeId,
    wo: NodeId,
    w_gate: NodeId,
    w_up: NodeId,
    w_down: NodeId,
    k_even_cache: NodeId,
    k_odd_cache: NodeId,
    v_cache: NodeId,
    qk_norm: Option<(NodeId, NodeId, NodeId)>,
    qkv_biases: Option<(NodeId, NodeId, NodeId)>,
    gate_before_up: bool,
) -> Result<(NodeId, CachedLayerRoots), TensorError> {
    // Same architecture inputs as `append_mistral_cached_layer`'s own
    // `qk_norm: Option<(NodeId, NodeId, NodeId)>` (q-norm weight, k-norm
    // weight, `inv_head_dim`) -- ROW 373's typed rejection here was a class
    // defect, not a correct omission: this builder's own attention block is
    // otherwise node-for-node the two-range sibling's, so it can carry the
    // same per-head RMSNorm and the same pairing selection
    // (`qk_norm.is_some()`) that sibling already uses, see that function's
    // own `qk_norm` doc.
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

    let v_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->sudi"), (wv, "iud->sudi")],
    )?;
    let v_new = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        v_product,
        "sudi->sudi",
        "sud->sudi",
    )?;

    let (q_raw, k_new_raw, v_new) = match qkv_biases {
        Some((q_bias, k_bias, v_bias)) => (
            elementwise(
                program,
                DType::Float32,
                ScalarOp::Add,
                &[(q_raw, "shd->shd"), (q_bias, "hd->shd")],
            )?,
            elementwise(
                program,
                DType::Float32,
                ScalarOp::Add,
                &[(k_new_raw, "sud->sud"), (k_bias, "ud->sud")],
            )?,
            elementwise(
                program,
                DType::Float32,
                ScalarOp::Add,
                &[(v_new, "sud->sud"), (v_bias, "ud->sud")],
            )?,
        ),
        None => (q_raw, k_new_raw, v_new),
    };

    let (q, k_new) = match qk_norm {
        Some((q_norm_weight, k_norm_weight, inv_head_dim)) => {
            let q = rmsnorm_per_head(program, q_raw, q_norm_weight, inv_head_dim, eps, "h")?;
            let k_new =
                rmsnorm_per_head(program, k_new_raw, k_norm_weight, inv_head_dim, eps, "u")?;
            (q, k_new)
        }
        None => (q_raw, k_new_raw),
    };

    // Pairing selection mirrors `append_mistral_cached_layer`'s own
    // `qk_norm.is_some()` rule (that function's doc walks the NEOX-vs-
    // interleaved reasoning): a checkpoint carrying `attn_q_norm.weight` is
    // the split-half family, everything else stays interleaved.
    let (rotated_q_even, rotated_q_odd, rotated_k_new_even, rotated_k_new_odd) = match qk_norm {
        Some(_) => {
            let pairs = head_dim / 2;
            let (rotated_q_first, rotated_q_second) = fused_rope_pair(
                program,
                q,
                'h',
                cos_new,
                sin_new,
                RopePairing::SplitHalf { pairs },
            )?;
            let (rotated_k_first, rotated_k_second) = fused_rope_pair(
                program,
                k_new,
                'u',
                cos_new,
                sin_new,
                RopePairing::SplitHalf { pairs },
            )?;
            (
                rotated_q_first,
                rotated_q_second,
                rotated_k_first,
                rotated_k_second,
            )
        }
        None => {
            let (rotated_q_even, rotated_q_odd) =
                fused_rope_pair(program, q, 'h', cos_new, sin_new, RopePairing::Interleaved)?;
            let (rotated_k_new_even, rotated_k_new_odd) = fused_rope_pair(
                program,
                k_new,
                'u',
                cos_new,
                sin_new,
                RopePairing::Interleaved,
            )?;
            (
                rotated_q_even,
                rotated_q_odd,
                rotated_k_new_even,
                rotated_k_new_odd,
            )
        }
    };

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

    // single range: query `s` against the WHOLE merged key range `t`
    // (symbol 1's extent), `k_even_cache`/`k_odd_cache` already carrying
    // every position this query may attend -- no second source, no combine.
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
    let neg_infinity = scalar_constant(program, f32::NEG_INFINITY);
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

    let residual1 = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(attn_out, "sd->sd"), (x, "sd->sd")],
    )?;

    let normed2 = rmsnorm(program, residual1, ffn_norm_weight, inv_dim, eps)?;

    let append_gate = |program: &mut Vec<Op>| -> Result<NodeId, TensorError> {
        let gate_product = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(normed2, "sd->sdg"), (w_gate, "dg->sdg")],
        )?;
        reduce(
            program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            gate_product,
            "sdg->sdg",
            "sg->sdg",
        )
    };
    let append_up = |program: &mut Vec<Op>| -> Result<NodeId, TensorError> {
        let up_product = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(normed2, "sd->sdg"), (w_up, "dg->sdg")],
        )?;
        reduce(
            program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            up_product,
            "sdg->sdg",
            "sg->sdg",
        )
    };
    // `gate_before_up` only decides encode ORDER of these two independent
    // matvecs (both read `normed2`, neither reads the other's output) --
    // `gate`/`up` bind identically either way, so every op below is
    // unaffected by which branch ran.
    let (gate, up) = if gate_before_up {
        let gate = append_gate(program)?;
        let up = append_up(program)?;
        (gate, up)
    } else {
        let up = append_up(program)?;
        let gate = append_gate(program)?;
        (gate, up)
    };

    let neg_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Negate,
        &[(gate, "sg->sg")],
    )?;
    let exp_neg_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Exponential,
        &[(neg_gate, "sg->sg")],
    )?;
    let one_plus_exp = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(exp_neg_gate, "sg->sg"), (ones, "->sg")],
    )?;
    let sigmoid_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Reciprocal,
        &[(one_plus_exp, "sg->sg")],
    )?;
    let silu_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(gate, "sg->sg"), (sigmoid_gate, "sg->sg")],
    )?;
    let ffn_hidden = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(silu_gate, "sg->sg"), (up, "sg->sg")],
    )?;

    let down_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(ffn_hidden, "sg->sgd"), (w_down, "gd->sgd")],
    )?;
    let ffn_out = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        down_product,
        "sgd->sgd",
        "sd->sgd",
    )?;

    let x_next = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(ffn_out, "sd->sd"), (residual1, "sd->sd")],
    )?;

    Ok((x_next, (rotated_k_new_even, rotated_k_new_odd, v_new)))
}

/// [`mistral_single_range_cached_forward_program`] is
/// [`mistral_cached_forward_program`]'s single-range counterpart: same
/// per-layer weight inputs, same [`CachedLayerRoots`] contract, one
/// difference -- `causal_mask_merged` in place of `causal_mask`
/// (needs a `cached_len` scalar the plain cache mask does not), and
/// `append_mistral_single_range_cached_layer` in place of
/// `append_mistral_cached_layer` for every layer. Dense-only (no MoE
/// branch): the mixture-of-experts FFN this function's counterpart also
/// supports is orthogonal to the attention-merge this function exists to
/// prove, and duplicating that branch here would test nothing new.
///
/// `qk_norm` (ROW 373) selects the same per-head QK-norm + split-half RoPE
/// pairing [`qwen3_cached_forward_program`] carries on the two-range path --
/// `true` declares `blk.{layer}.attn_q_norm.weight`/`attn_k_norm.weight`
/// inputs per layer and threads them through
/// [`append_mistral_single_range_cached_layer`]'s own
/// `Option<(NodeId, NodeId, NodeId)>` parameter, mirroring
/// [`mistral_cached_forward_program_with_experts`]'s own `inv_head_dim`/
/// `qk_norm_weights` construction below. `false` reproduces today's
/// interleaved, no-norm program node-for-node.
// ROW 326/328 diagnostic: `duplicate_head` mirrors `gate_before_up`'s own
// mechanism (a plain, always-compiled parameter a caller sets, production
// call sites pass a fixed literal) rather than a `#[cfg(test)]` item,
// because `#[cfg(test)]` on a `proxima-tensor` item is invisible
// cross-crate to `omega`/`proxima-model-interop`, which is where the
// Metal measurement this flag exists for actually runs.
// [`DuplicateHeadPosition::Before`]/`After` each append a second,
// identical `output.weight` reduce reusing this call's own `lm_head`
// (against `x`, the raw embedding, for `Before`; against `normed_final`
// for `After`), returned as the 4th tuple element so a caller can add it
// to a `Plan`'s requested outputs -- otherwise graph pruning drops it as
// unreachable dead code, same as any other unread node.
// [`DuplicateHeadPosition::None`] (every production call site) is
// byte-identical to this function's behavior before the flag existed.
//
// `last_row_only` is `mistral_cached_forward_program_with_experts_and_layer_taps`'s
// own flag, reproduced here: `true` gathers `normed_final` to its last row
// through a host-supplied `lm_head_row` leaf (that function's own doc has
// the full mechanism) before the real `output.weight` reduce, so `logits`
// is `[1, vocab]` instead of `[new_count, vocab]`. `false` (every call site
// before this flag existed) is byte-identical to this function's prior
// behavior. The `DuplicateHeadPosition` scratch reduce is unaffected either
// way -- it exists to measure the FULL-width projection's own cost (ROW
// 326/328's own doc), so it keeps reading `x`/`normed_final` directly
// regardless of `last_row_only`.
#[allow(
    clippy::too_many_arguments,
    reason = "one architecture hyperparameter per positional arg, matching every other \
              forward-program builder in this file (see the other `too_many_arguments` \
              call sites above); `last_row_only` is the 9th and last"
)]
pub fn mistral_single_range_cached_forward_program(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    block_count: u32,
    qk_norm: bool,
    duplicate_head: DuplicateHeadPosition,
    last_row_only: bool,
) -> Result<SingleRangeForwardProgram, TensorError> {
    mistral_single_range_cached_forward_program_with_biases(
        vocab,
        embedding,
        feed_forward,
        query_heads,
        kv_heads,
        head_dim,
        block_count,
        qk_norm,
        false,
        duplicate_head,
        last_row_only,
    )
}

/// Bias-aware counterpart of [`mistral_single_range_cached_forward_program`].
/// `qkv_biases` is a graph capability selected by checkpoint metadata; when
/// false, the legacy graph is retained exactly.
#[allow(clippy::too_many_arguments)]
pub fn mistral_single_range_cached_forward_program_with_biases(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    block_count: u32,
    qk_norm: bool,
    qkv_biases: bool,
    duplicate_head: DuplicateHeadPosition,
    last_row_only: bool,
) -> Result<SingleRangeForwardProgram, TensorError> {
    let group = query_heads / kv_heads;
    let pairs = head_dim / 2;

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

    // ROW 328: a SEPARATE, early `output.weight` `Op::Input` -- not a
    // second reference to the one declared before the real head below.
    // `resolve_named_blocks` (`proxima-tensor/src/cpu.rs`) resolves every
    // `Op::Input` node by NAME independently, so two nodes sharing the name
    // `output.weight` both bind to the same weight bytes with no special
    // casing; declaring a second one here (instead of hoisting the single
    // existing declaration) keeps the `None`/`After` program's own node
    // sequence byte-for-byte identical to before this row -- hoisting the
    // one declaration shifted every later `NodeId` by one and changed
    // `cached_attention_single_range_candidates`' fused bound-op count for
    // EVERY position, not just `Before` (`619` -> `523`, a regression this
    // row's own gate caught before it landed).
    let duplicate_head_scratch_before = if duplicate_head == DuplicateHeadPosition::Before {
        let lm_head_before = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding), Extent::Static(vocab)],
            "output.weight",
        );
        Some(duplicate_head_reduce(&mut program, x, lm_head_before)?)
    } else {
        None
    };

    let inv_dim = scalar_constant(&mut program, 1.0 / embedding as f32);
    let eps = symbolic_leaf(&mut program, DType::Float32, "eps");
    let ones = scalar_constant(&mut program, 1.0);
    let inv_sqrt_head_dim = scalar_constant(&mut program, 1.0 / (head_dim as f32).sqrt());
    // only materialized when a layer actually consumes it (`qk_norm`), same
    // guard `mistral_cached_forward_program_with_experts` uses so a dense
    // checkpoint's own node count is unaffected by this feature existing.
    let inv_head_dim = qk_norm.then(|| scalar_constant(&mut program, 1.0 / head_dim as f32));
    let cos_new = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(pairs)],
        "rope_cos",
    );
    let sin_new = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(pairs)],
        "rope_sin",
    );
    let group_ones = op::append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
            value: 1.0,
        },
    );
    let cached_len = input_leaf(&mut program, DType::Float32, Vec::new(), "cached_len");
    let is_future = causal_mask_merged(&mut program, cached_len)?;

    let mut cache_roots: Vec<CachedLayerRoots> = Vec::with_capacity(block_count as usize);

    for layer in 0..block_count {
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
        let w_gate = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding), Extent::Static(feed_forward)],
            &alloc::format!("blk.{layer}.ffn_gate.weight"),
        );
        let w_up = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding), Extent::Static(feed_forward)],
            &alloc::format!("blk.{layer}.ffn_up.weight"),
        );
        let w_down = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(feed_forward), Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.ffn_down.weight"),
        );
        let qk_norm_weights = inv_head_dim.map(|inv_head_dim| {
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
            (q_norm_weight, k_norm_weight, inv_head_dim)
        });
        let qkv_bias_weights = qkv_biases.then(|| {
            let q_bias = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(query_heads), Extent::Static(head_dim)],
                &alloc::format!("blk.{layer}.attn_q.bias"),
            );
            let k_bias = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(kv_heads), Extent::Static(head_dim)],
                &alloc::format!("blk.{layer}.attn_k.bias"),
            );
            let v_bias = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(kv_heads), Extent::Static(head_dim)],
                &alloc::format!("blk.{layer}.attn_v.bias"),
            );
            (q_bias, k_bias, v_bias)
        });

        let (x_next, layer_roots) = append_mistral_single_range_cached_layer_with_biases(
            &mut program,
            x,
            inv_dim,
            eps,
            ones,
            inv_sqrt_head_dim,
            cos_new,
            sin_new,
            group_ones,
            is_future,
            group,
            head_dim,
            attn_norm_weight,
            ffn_norm_weight,
            wq,
            wk,
            wv,
            wo,
            w_gate,
            w_up,
            w_down,
            k_even_cache,
            k_odd_cache,
            v_cache,
            qk_norm_weights,
            qkv_bias_weights,
            true,
        )?;
        x = x_next;
        cache_roots.push(layer_roots);
    }

    let output_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding)],
        "output_norm.weight",
    );
    let normed_final = rmsnorm(&mut program, x, output_norm_weight, inv_dim, eps)?;

    // Same `lm_head_row` leaf and gather
    // `mistral_cached_forward_program_with_experts_and_layer_taps`'s own
    // `last_row_only` arm uses -- see that call site for the full mechanism
    // doc. Only the REAL head narrows to one row; `duplicate_head_scratch`
    // below still reads `x`/`normed_final` directly, since it exists to
    // measure the FULL-width projection's own cost.
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
    let logits = duplicate_head_reduce(&mut program, normed_last, lm_head)?;

    let duplicate_head_scratch = match duplicate_head {
        DuplicateHeadPosition::None => None,
        DuplicateHeadPosition::Before => duplicate_head_scratch_before,
        DuplicateHeadPosition::After => {
            Some(duplicate_head_reduce(&mut program, normed_final, lm_head)?)
        }
    };

    Ok((program, logits, cache_roots, duplicate_head_scratch))
}

/// `sum_d(activation[s, d] * lm_head[d, v])` -- the vocab-projection
/// multiply-reduce pair both the real head and every
/// [`DuplicateHeadPosition`] scratch reduce share, factored out so ROW 328's
/// `Before`/`After` positions differ only in which activation node (`x`
/// pre-layer-0 vs `normed_final` post-layer-31) they read, never in the
/// reduce shape itself.
pub fn duplicate_head_reduce(
    program: &mut Vec<Op>,
    activation: NodeId,
    lm_head: NodeId,
) -> Result<NodeId, TensorError> {
    let product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(activation, "sd->sdv"), (lm_head, "dv->sdv")],
    )?;
    reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "sdv->sdv",
        "sv->sdv",
    )
}

/// [`append_mistral_cached_layer`]'s mixture-of-experts counterpart, the
/// same relationship [`append_mistral_moe_layer`] bears to
/// [`append_mistral_layer`]: cached attention block (RoPE + GQA +
/// online-softmax combine over the cached/new key split, the same shape as
/// [`append_mistral_cached_layer`]'s own, including that function's
/// `qk_norm`-gated per-head Q/K norm and RoPE-pairing switch -- Qwen3-MoE's
/// own checkpoint carries `attn_q_norm.weight`/`attn_k_norm.weight` on every
/// layer, every one of them MoE, so this arm needs the identical switch or
/// every MoE layer silently skips QK-norm and rotates Q/K with the wrong
/// (interleaved, not NEOX split-half) pairing), [`append_moe_ffn`] in place
/// of the dense SwiGLU triple. Kept as a separate function for the same
/// reason [`append_mistral_moe_layer`] is: the dense cached path's own node
/// sequence never changes shape merely because this function exists next to
/// it.
#[allow(clippy::too_many_arguments)]
pub fn append_mistral_cached_moe_layer(
    program: &mut Vec<Op>,
    layer: u32,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    ones: NodeId,
    inv_sqrt_head_dim: NodeId,
    cos_new: NodeId,
    sin_new: NodeId,
    group_ones: NodeId,
    is_future: NodeId,
    group: u32,
    head_dim: u32,
    attn_norm_weight: NodeId,
    ffn_norm_weight: NodeId,
    wq: NodeId,
    wk: NodeId,
    wv: NodeId,
    wo: NodeId,
    gate_inp: NodeId,
    expert_w_gate: NodeId,
    expert_w_up: NodeId,
    expert_w_down: NodeId,
    expert_count: u32,
    expert_used_count: u32,
    k_even_cache: NodeId,
    k_odd_cache: NodeId,
    v_cache: NodeId,
    qk_norm: Option<(NodeId, NodeId, NodeId)>,
) -> Result<(NodeId, CachedLayerRoots, MoeSite), TensorError> {
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

    let (q, k_new) = match qk_norm {
        Some((q_norm_weight, k_norm_weight, inv_head_dim)) => {
            let q = rmsnorm_per_head(program, q_raw, q_norm_weight, inv_head_dim, eps, "h")?;
            let k_new =
                rmsnorm_per_head(program, k_new_raw, k_norm_weight, inv_head_dim, eps, "u")?;
            (q, k_new)
        }
        None => (q_raw, k_new_raw),
    };

    let v_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->sudi"), (wv, "iud->sudi")],
    )?;
    let v_new = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        v_product,
        "sudi->sudi",
        "sud->sudi",
    )?;

    // Same `qk_norm.is_some()` switch as [`append_mistral_cached_layer`]
    // (see that function's own doc): a checkpoint carrying `attn_q_norm.weight`
    // is NEOX-family (Qwen3), whose on-disk Q/K rows stay in HF's native
    // split-half layout, never llama.cpp's converter-permuted interleaved
    // pairing a no-qk_norm (Mistral/LLaMA) checkpoint uses.
    let (rotated_q_even, rotated_q_odd, rotated_k_new_even, rotated_k_new_odd) = match qk_norm {
        Some(_) => {
            let pairs = head_dim / 2;
            let (rotated_q_first, rotated_q_second) = fused_rope_pair(
                program,
                q,
                'h',
                cos_new,
                sin_new,
                RopePairing::SplitHalf { pairs },
            )?;
            let (rotated_k_first, rotated_k_second) = fused_rope_pair(
                program,
                k_new,
                'u',
                cos_new,
                sin_new,
                RopePairing::SplitHalf { pairs },
            )?;
            (
                rotated_q_first,
                rotated_q_second,
                rotated_k_first,
                rotated_k_second,
            )
        }
        None => {
            let (rotated_q_even, rotated_q_odd) =
                fused_rope_pair(program, q, 'h', cos_new, sin_new, RopePairing::Interleaved)?;
            let (rotated_k_new_even, rotated_k_new_odd) = fused_rope_pair(
                program,
                k_new,
                'u',
                cos_new,
                sin_new,
                RopePairing::Interleaved,
            )?;
            (
                rotated_q_even,
                rotated_q_odd,
                rotated_k_new_even,
                rotated_k_new_odd,
            )
        }
    };

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
    let neg_infinity = scalar_constant(program, f32::NEG_INFINITY);
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
            (is_future, "sw->swug"),
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

    let residual1 = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(attn_out, "sd->sd"), (x, "sd->sd")],
    )?;

    let normed2 = rmsnorm(program, residual1, ffn_norm_weight, inv_dim, eps)?;

    let moe_spec = MoeFfnSpec {
        router: MoeRouter::GateInput(gate_inp),
        expert_w_gate,
        expert_w_up,
        expert_w_down,
        expert_count,
        expert_used_count,
        ones,
        gating: ExpertGatingFunc::Softmax,
        expert_bias: None,
        expert_scale: None,
        activation: Activation::Silu,
        strategy: MoeProjectionStrategy::PerRoute,
    };
    let (ffn_out, site) = append_moe_ffn(program, layer, normed2, &moe_spec)?;

    let x_next = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(ffn_out, "sd->sd"), (residual1, "sd->sd")],
    )?;

    Ok((x_next, (rotated_k_new_even, rotated_k_new_odd, v_new), site))
}

/// Which mixer one transformer block runs. LFM2.5-8B-A1B (`general.architecture
/// = "lfm2moe"`) hybridizes short-convolution and attention blocks in the same
/// 24-layer stack, and GGUF carries no `layer_types` metadata key for this
/// architecture (confirmed absent on the real checkpoint's own metadata dump)
/// -- the only ground truth is which tensors a block's own name prefix owns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Attention,
    ShortConv,
}

impl LayerKind {
    /// Derives one block's kind from its own tensor name set: LFM2.5-8B-A1B's
    /// real checkpoint shows every block owns exactly one of
    /// `blk.{layer}.attn_q.weight` or `blk.{layer}.shortconv.conv.weight`,
    /// never neither and never both, so this is a presence check, not a
    /// classifier -- the caller (whoever already has the checkpoint's tensor
    /// directory, e.g. `proxima-model-interop`) walks that directory once per
    /// block and hands this a `layer`-scoped name iterator; this function
    /// never reads a file itself, keeping `proxima-tensor` free of a GGUF
    /// dependency.
    pub fn from_tensor_names<'name>(
        names: impl IntoIterator<Item = &'name str>,
        layer: u32,
    ) -> Result<Self, TensorError> {
        let attention_marker = alloc::format!("blk.{layer}.attn_q.weight");
        let conv_marker = alloc::format!("blk.{layer}.shortconv.conv.weight");
        for name in names {
            if name == attention_marker {
                return Ok(Self::Attention);
            }
            if name == conv_marker {
                return Ok(Self::ShortConv);
            }
        }
        Err(TensorError::UndeterminedLayerKind { layer })
    }
}
