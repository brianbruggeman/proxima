use super::*;

/// [`append_mistral_layer`]'s attention sub-block in isolation (RoPE + GQA +
/// causal mask + residual, no FFN) -- the piece [`lfm2_forward_program_with_experts`]
/// needs on its own, since an attention block there sits beside
/// [`append_lfm2_conv_mixer`] rather than always beside the same FFN choice
/// [`append_mistral_layer`] bundles it with. Node-for-node the same attention
/// graph [`append_mistral_layer`] runs before its own FFN call, extracted
/// rather than shared by refactoring that function, so the dense Mistral/Llama
/// path's own generated program bytes never change shape because this
/// function exists next to it.
#[allow(clippy::too_many_arguments)]
pub fn append_attention_mixer(
    program: &mut Vec<Op>,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    inv_sqrt_head_dim: NodeId,
    inv_head_dim: NodeId,
    cos: NodeId,
    sin: NodeId,
    group_ones: NodeId,
    is_future: NodeId,
    neg_infinity: NodeId,
    group: u32,
    attn_norm_weight: NodeId,
    q_norm_weight: NodeId,
    k_norm_weight: NodeId,
    wq: NodeId,
    wk: NodeId,
    wv: NodeId,
    wo: NodeId,
) -> Result<NodeId, TensorError> {
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
    // [`Lfm2MoeAttention.q_layernorm`]'s own placement
    // (`modeling_lfm2_moe.py:331`): normalizes right after the head
    // reshape, BEFORE `apply_rotary_pos_emb` -- never after.
    let q = rmsnorm_per_head(program, q_raw, q_norm_weight, inv_head_dim, eps, "h")?;

    let k_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->sudi"), (wk, "iud->sudi")],
    )?;
    let k_raw = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        k_product,
        "sudi->sudi",
        "sud->sudi",
    )?;
    let k = rmsnorm_per_head(program, k_raw, k_norm_weight, inv_head_dim, eps, "u")?;

    let v_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->sudi"), (wv, "iud->sudi")],
    )?;
    let v = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        v_product,
        "sudi->sudi",
        "sud->sudi",
    )?;

    let q_even_cos = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(q, "s,h,2*i->shi"), (cos, "si->shi")],
    )?;
    let q_odd_sin = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(q, "s,h,2*i+1->shi"), (sin, "si->shi")],
    )?;
    let rotated_q_even = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(q_even_cos, "shi->shi"), (q_odd_sin, "shi->shi")],
    )?;
    let q_even_sin = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(q, "s,h,2*i->shi"), (sin, "si->shi")],
    )?;
    let q_odd_cos = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(q, "s,h,2*i+1->shi"), (cos, "si->shi")],
    )?;
    let rotated_q_odd = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(q_even_sin, "shi->shi"), (q_odd_cos, "shi->shi")],
    )?;

    let k_even_cos = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(k, "s,u,2*i->sui"), (cos, "si->sui")],
    )?;
    let k_odd_sin = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(k, "s,u,2*i+1->sui"), (sin, "si->sui")],
    )?;
    let rotated_k_even = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(k_even_cos, "sui->sui"), (k_odd_sin, "sui->sui")],
    )?;
    let k_even_sin = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(k, "s,u,2*i->sui"), (sin, "si->sui")],
    )?;
    let k_odd_cos = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(k, "s,u,2*i+1->sui"), (cos, "si->sui")],
    )?;
    let rotated_k_odd = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(k_even_sin, "sui->sui"), (k_odd_cos, "sui->sui")],
    )?;

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

    let score_even_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (q_even_grouped, "sugi->stugi"),
            (rotated_k_even, "tui->stugi"),
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
        &[
            (q_odd_grouped, "sugi->stugi"),
            (rotated_k_odd, "tui->stugi"),
        ],
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
        &[(probabilities, "stug->stugd"), (v, "tud->stugd")],
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

    elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(attn_out, "sd->sd"), (x, "sd->sd")],
    )
}

/// LFM2.5-8B-A1B's hybrid forward pass: `block_count` blocks, each either
/// `append_attention_mixer` or `append_lfm2_conv_mixer` per its own
/// `layer_kinds[layer]` (derived by [`LayerKind::from_tensor_names`] from the
/// real checkpoint's tensor directory, since `layer_types` is not a metadata
/// key this architecture writes), then a shared RMSNorm and
/// `append_moe_ffn`/dense-triple FFN exactly like
/// [`mistral_forward_program`]'s own MoE branch --
/// `leading_dense_block_count` (LFM2.5-8B-A1B: `2`) is threaded per layer
/// rather than a single crate-wide dense/MoE switch, since this checkpoint's
/// first two blocks are dense and the rest are routed.
///
/// Prefill-only: takes the whole prompt as one `[seq, embedding]` pass, the
/// same scope [`mistral_forward_program`] has. A KV-cached and
/// conv-state-cached incremental counterpart (mirroring
/// [`mistral_cached_forward_program_with_experts`]) is a further step this
/// function's own doc does not claim -- `causal_conv1d`'s masked-gather
/// composition only needs the whole sequence to be present at once, which a
/// one-token-at-a-time decode call does not have.
#[allow(clippy::too_many_arguments)]
pub fn lfm2_forward_program_with_experts(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    expert_feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    block_count: u32,
    expert_count: u32,
    expert_used_count: u32,
    leading_dense_block_count: u32,
    l_cache: u32,
    layer_kinds: &[LayerKind],
) -> Result<(Vec<Op>, NodeId, MoeSites), TensorError> {
    if layer_kinds.len() != block_count as usize {
        return Err(TensorError::LayerKindCountMismatch {
            expected: block_count,
            found: layer_kinds.len(),
        });
    }

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

    let inv_dim = scalar_constant(&mut program, 1.0 / embedding as f32);
    let eps = symbolic_leaf(&mut program, DType::Float32, "eps");
    let ones = scalar_constant(&mut program, 1.0);
    let inv_sqrt_head_dim = scalar_constant(&mut program, 1.0 / (head_dim as f32).sqrt());
    let inv_head_dim = scalar_constant(&mut program, 1.0 / head_dim as f32);
    let cos = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(pairs)],
        "rope_cos",
    );
    let sin = input_leaf(
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
    let (is_future, neg_infinity) = causal_mask(&mut program)?;
    let mut moe_sites: Vec<MoeSite> = Vec::new();

    for (layer, kind) in layer_kinds.iter().enumerate() {
        let layer = layer as u32;
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

        let post_mixer = match kind {
            LayerKind::Attention => {
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
                append_attention_mixer(
                    &mut program,
                    x,
                    inv_dim,
                    eps,
                    inv_sqrt_head_dim,
                    inv_head_dim,
                    cos,
                    sin,
                    group_ones,
                    is_future,
                    neg_infinity,
                    group,
                    attn_norm_weight,
                    q_norm_weight,
                    k_norm_weight,
                    wq,
                    wk,
                    wv,
                    wo,
                )?
            }
            LayerKind::ShortConv => {
                // `b_proj`/`c_proj`/`x_proj` are the real checkpoint's single
                // fused `blk.{layer}.shortconv.in_proj.weight`
                // (`[embedding, 3*embedding]`) split three ways -- see
                // `append_lfm2_conv_mixer`'s own doc for why this graph
                // cannot instead slice one fused `Input` by offset. Binding
                // these three names from that one on-disk tensor is a
                // binder-side split this session does not implement; the
                // names here are this program's contract for whoever does.
                let b_proj = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(embedding), Extent::Static(embedding)],
                    &alloc::format!("blk.{layer}.shortconv.in_proj.weight.b"),
                );
                let c_proj = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(embedding), Extent::Static(embedding)],
                    &alloc::format!("blk.{layer}.shortconv.in_proj.weight.c"),
                );
                let x_proj = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(embedding), Extent::Static(embedding)],
                    &alloc::format!("blk.{layer}.shortconv.in_proj.weight.x"),
                );
                // `[embedding, l_cache]`, NOT `[l_cache, embedding]` --
                // `causal_conv1d`'s own doc on its `dl->sld` map explains why:
                // the real on-disk tensor has `l_cache` as its fastest axis,
                // and `row_major_strides` (`bind.rs`) makes the LAST declared
                // shape axis the fastest one.
                let conv_weight = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(embedding), Extent::Static(l_cache)],
                    &alloc::format!("blk.{layer}.shortconv.conv.weight"),
                );
                let out_proj = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(embedding), Extent::Static(embedding)],
                    &alloc::format!("blk.{layer}.shortconv.out_proj.weight"),
                );
                append_lfm2_conv_mixer(
                    &mut program,
                    x,
                    inv_dim,
                    eps,
                    attn_norm_weight,
                    b_proj,
                    c_proj,
                    x_proj,
                    conv_weight,
                    out_proj,
                    l_cache,
                )?
            }
        };

        let normed2 = rmsnorm(&mut program, post_mixer, ffn_norm_weight, inv_dim, eps)?;

        let ffn_out = if layer < leading_dense_block_count {
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
            let gate_product = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(normed2, "sd->sdg"), (w_gate, "dg->sdg")],
            )?;
            let gate = reduce(
                &mut program,
                DType::Float32,
                ScalarOp::Add,
                ReduceInit::Zero,
                gate_product,
                "sdg->sdg",
                "sg->sdg",
            )?;
            let up_product = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(normed2, "sd->sdg"), (w_up, "dg->sdg")],
            )?;
            let up = reduce(
                &mut program,
                DType::Float32,
                ScalarOp::Add,
                ReduceInit::Zero,
                up_product,
                "sdg->sdg",
                "sg->sdg",
            )?;

            let neg_gate = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Negate,
                &[(gate, "sg->sg")],
            )?;
            let exp_neg_gate = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Exponential,
                &[(neg_gate, "sg->sg")],
            )?;
            let one_plus_exp = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Add,
                &[(exp_neg_gate, "sg->sg"), (ones, "->sg")],
            )?;
            let sigmoid_gate = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Reciprocal,
                &[(one_plus_exp, "sg->sg")],
            )?;
            let silu_gate = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(gate, "sg->sg"), (sigmoid_gate, "sg->sg")],
            )?;
            let ffn_hidden = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(silu_gate, "sg->sg"), (up, "sg->sg")],
            )?;

            let down_product = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(ffn_hidden, "sg->sgd"), (w_down, "gd->sgd")],
            )?;
            reduce(
                &mut program,
                DType::Float32,
                ScalarOp::Add,
                ReduceInit::Zero,
                down_product,
                "sgd->sgd",
                "sd->sgd",
            )?
        } else {
            let gate_inp = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(expert_count)],
                &alloc::format!("blk.{layer}.ffn_gate_inp.weight"),
            );
            let expert_w_gate = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(expert_count),
                    Extent::Static(embedding),
                    Extent::Static(expert_feed_forward)
                ],
                &alloc::format!("blk.{layer}.ffn_gate_exps.weight"),
            );
            let expert_w_up = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(expert_count),
                    Extent::Static(embedding),
                    Extent::Static(expert_feed_forward)
                ],
                &alloc::format!("blk.{layer}.ffn_up_exps.weight"),
            );
            let expert_w_down = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(expert_count),
                    Extent::Static(expert_feed_forward),
                    Extent::Static(embedding)
                ],
                &alloc::format!("blk.{layer}.ffn_down_exps.weight"),
            );
            let expert_bias = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(expert_count)],
                &alloc::format!("blk.{layer}.exp_probs_b.bias"),
            );
            let (ffn_out, site) = append_moe_ffn(
                &mut program,
                layer,
                normed2,
                gate_inp,
                expert_w_gate,
                expert_w_up,
                expert_w_down,
                expert_count,
                expert_used_count,
                ones,
                ExpertGatingFunc::Sigmoid,
                Some(expert_bias),
            )?;
            moe_sites.push(site);
            ffn_out
        };

        x = elementwise(
            &mut program,
            DType::Float32,
            ScalarOp::Add,
            &[(ffn_out, "sd->sd"), (post_mixer, "sd->sd")],
        )?;
    }

    let output_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding)],
        "output_norm.weight",
    );
    let normed_final = rmsnorm(&mut program, x, output_norm_weight, inv_dim, eps)?;

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
        &[(normed_final, "sd->sdv"), (lm_head, "dv->sdv")],
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

    Ok((program, logits, MoeSites(moe_sites)))
}

/// [`mistral_forward_program`]'s key/value-cached counterpart: the same
/// architecture, but `ids`/`rope_cos`/`rope_sin` carry only the `new`
/// positions this call introduces (symbol 0), attention also draws on a
/// per-layer already-rotated key/value cache sized by symbol 1
/// (`kv_cache.{layer}.k_even`/`k_odd`/`v`, bound [`Op::Input`]s each layer's
/// own online-softmax attention combines with its freshly computed
/// key/value), and the returned roots are `(logits,
/// per_layer_cache_roots)` instead of one implicit last-node root, since a
/// caller now needs the per-layer [`CachedLayerRoots`] to grow its cache for
/// the next call. A caller passes `symbols = [new_positions, cached_len]`
/// to [`crate::shape::infer`]/[`crate::cpu::evaluate_quantized_named`], and
/// on the very first call binds every `kv_cache.*` name to a zero-length
/// buffer (`cached_len == 0`) -- the cached-block reduces both fold over an
/// empty range, which [`ReduceInit::Zero`]/[`ReduceInit::NegativeInfinity`]
/// already define as identity/`-inf`, so the first call degenerates to
/// plain causal self-attention over the whole prompt with no special case.
///
/// Dense-only: always binds `append_mistral_cached_layer`'s plain
/// `ffn_{gate,up,down}.weight` triple. Delegates to
/// [`mistral_cached_forward_program_with_experts`] with `expert_count = 0`,
/// `expert_used_count = 0` -- that function's own doc explains why those two
/// values select the identical dense program this function has always
/// built. Kept as its own entry point (rather than folding the two extra
/// parameters in here) because this signature already has real callers
/// outside this crate that a dense-only checkpoint never needs to pass an
/// expert config to.
#[allow(clippy::too_many_arguments)]
pub fn mistral_cached_forward_program(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    block_count: u32,
) -> Result<(Vec<Op>, NodeId, Vec<CachedLayerRoots>), TensorError> {
    mistral_cached_forward_program_with_experts(
        vocab,
        embedding,
        feed_forward,
        query_heads,
        kv_heads,
        head_dim,
        block_count,
        0,
        0,
        false,
        false,
        false,
        false,
    )
    .map(|(program, roots, cache_roots, _moe_sites)| (program, roots.logits, cache_roots))
}

/// [`mistral_cached_forward_program`]'s Qwen3 dense-attention counterpart:
/// the identical interleaved-RoPE cached layer, plus per-head QK-norm
/// (Qwen3's own `q_norm`/`k_norm`, `modeling_qwen3.py`'s `Qwen3Attention`)
/// applied to `q`/`k_new` before RoPE -- see
/// `append_mistral_cached_layer`'s `qk_norm` parameter doc for the exact
/// two ops this adds over the plain Mistral layer. Qwen3 has no
/// mixture-of-experts variant this crate has bound yet, so this takes no
/// `expert_count`/`expert_used_count`, the same dense-only shape
/// [`mistral_cached_forward_program`] itself uses.
#[allow(clippy::too_many_arguments)]
pub fn qwen3_cached_forward_program(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    block_count: u32,
) -> Result<(Vec<Op>, NodeId, Vec<CachedLayerRoots>), TensorError> {
    mistral_cached_forward_program_with_experts(
        vocab,
        embedding,
        feed_forward,
        query_heads,
        kv_heads,
        head_dim,
        block_count,
        0,
        0,
        true,
        false,
        false,
        false,
    )
    .map(|(program, roots, cache_roots, _moe_sites)| (program, roots.logits, cache_roots))
}

/// [`mistral_cached_forward_program`]'s mixture-of-experts-capable
/// counterpart, carrying the same `expert_count`/`expert_used_count`
/// parameters [`mistral_forward_program`] already takes. `expert_count == 0`
/// binds every layer through `append_mistral_cached_layer`'s plain
/// `ffn_{gate,up,down}.weight` triple, node-for-node the same program
/// [`mistral_cached_forward_program`] has always built, so a dense
/// checkpoint's generated program is unaffected by this function's
/// existence. `expert_count > 0` routes each layer through
/// `append_mistral_cached_moe_layer` instead, gathering one of
/// `expert_count` experts' weight slabs per token per `append_moe_ffn`'s
/// doc -- the same routed FFN [`mistral_forward_program`]'s own MoE branch
/// already runs, reused rather than reconstructed.
///
/// `paired_gate_up_reduce` is passed straight through to every dense layer's
/// `append_mistral_cached_layer` call (see that parameter's own doc) --
/// `false` at every call site in this crate today; a caller opts in only
/// once its loader has bound `blk.{layer}.ffn_gate_up.weight`
/// (`proxima-model-interop::bind::bind_matmul_weight_paired`). No effect on
/// the `expert_count > 0` branch (MoE's own gate/up weights are a separate
/// per-expert stack this flag does not touch).
///
/// `fused_qkv_reduce` is passed straight through to every dense layer's
/// `append_mistral_cached_layer` call (see that parameter's own doc) --
/// `false` at every call site in this crate today; a caller opts in only
/// once its loader has bound `blk.{layer}.attn_qkv.weight`
/// (`proxima-model-interop::bind::bind_matmul_weight_triple`). Requires
/// `qk_norm == false` (`append_mistral_cached_layer`'s own doc); no effect
/// on the `expert_count > 0` branch (attention projections are untouched by
/// which FFN branch runs).
#[allow(clippy::too_many_arguments)]
pub fn mistral_cached_forward_program_with_experts(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    block_count: u32,
    expert_count: u32,
    expert_used_count: u32,
    qk_norm: bool,
    qkv_biases: bool,
    paired_gate_up_reduce: bool,
    fused_qkv_reduce: bool,
) -> Result<(Vec<Op>, ForwardRoots, Vec<CachedLayerRoots>, MoeSites), TensorError> {
    let (program, roots, cache_roots, _layer_residuals, moe_sites) =
        mistral_cached_forward_program_with_experts_and_layer_taps(
            vocab,
            embedding,
            feed_forward,
            query_heads,
            kv_heads,
            head_dim,
            block_count,
            expert_count,
            expert_used_count,
            qk_norm,
            qkv_biases,
            paired_gate_up_reduce,
            fused_qkv_reduce,
            false,
        )?;
    Ok((program, roots, cache_roots, moe_sites))
}

/// [`mistral_cached_forward_program_with_experts`]'s full implementation,
/// additionally returning one [`NodeId`] per layer -- the residual
/// (`x_next`, the post-MoE-add activation) each block hands the next layer,
/// in layer order, `block_count` entries. A caller bisecting a CPU-vs-Metal
/// divergence requests these as extra program outputs (48 x [seq, embedding]
/// floats for a 48-layer checkpoint, trivially small) to find the first
/// layer whose output disagrees, without materializing every intermediate
/// node in the graph as an output (the CPU evaluator keeps every requested
/// output's full lifetime alive, so requesting ALL nodes is the >130 GB
/// failure mode this narrower request set avoids).
///
/// `last_row_only` gates the vocab-projection matmul's own row count:
/// `true` slices the final-norm activation to its last row before
/// `output.weight` ever multiplies it (a host-supplied `lm_head_row`
/// `Op::Input`, gathered through the same [`IndexMap::Computed`] shape
/// [`embedding_lookup`] already proves correct -- see that leaf's own doc
/// at the call site below for why it is host-supplied rather than
/// in-graph-derived), so the matmul computes one row instead of
/// `new_count`. `false` (every existing caller today) reproduces the prior
/// per-row-logits program unchanged -- a caller genuinely needing every
/// new row's own logits (multi-token verification, prefill scoring,
/// logprobs) opts into that by passing `false`, not by this crate guessing
/// which one a caller wants.
#[allow(clippy::too_many_arguments)]
pub fn mistral_cached_forward_program_with_experts_and_layer_taps(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    block_count: u32,
    expert_count: u32,
    expert_used_count: u32,
    qk_norm: bool,
    qkv_biases: bool,
    paired_gate_up_reduce: bool,
    fused_qkv_reduce: bool,
    last_row_only: bool,
) -> Result<MistralMoeForwardProgramWithLayerTaps, TensorError> {
    let rope_pairing = if qk_norm {
        RopePairing::SplitHalf {
            pairs: head_dim / 2,
        }
    } else {
        RopePairing::Interleaved
    };
    mistral_cached_forward_program_with_experts_and_layer_taps_with_rope_pairing(
        vocab,
        embedding,
        feed_forward,
        query_heads,
        kv_heads,
        head_dim,
        block_count,
        expert_count,
        expert_used_count,
        qk_norm,
        qkv_biases,
        paired_gate_up_reduce,
        fused_qkv_reduce,
        last_row_only,
        rope_pairing,
    )
}

/// Qwen2's dense/MoE graph variant. Qwen2 uses split-half (NEOX) RoPE even
/// though it has no QK-norm weights, so its pairing must be selected from the
/// architecture name rather than inferred from the presence of norm tensors.
#[allow(clippy::too_many_arguments)]
pub fn qwen2_cached_forward_program_with_experts_and_layer_taps(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    block_count: u32,
    expert_count: u32,
    expert_used_count: u32,
    qkv_biases: bool,
    paired_gate_up_reduce: bool,
    fused_qkv_reduce: bool,
    last_row_only: bool,
) -> Result<MistralMoeForwardProgramWithLayerTaps, TensorError> {
    mistral_cached_forward_program_with_experts_and_layer_taps_with_rope_pairing(
        vocab,
        embedding,
        feed_forward,
        query_heads,
        kv_heads,
        head_dim,
        block_count,
        expert_count,
        expert_used_count,
        false,
        qkv_biases,
        paired_gate_up_reduce,
        fused_qkv_reduce,
        last_row_only,
        RopePairing::SplitHalf {
            pairs: head_dim / 2,
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mistral_cached_forward_program_with_experts_and_layer_taps_with_rope_pairing(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    block_count: u32,
    expert_count: u32,
    expert_used_count: u32,
    qk_norm: bool,
    qkv_biases: bool,
    paired_gate_up_reduce: bool,
    fused_qkv_reduce: bool,
    last_row_only: bool,
    rope_pairing: RopePairing,
) -> Result<MistralMoeForwardProgramWithLayerTaps, TensorError> {
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

    let inv_dim = scalar_constant(&mut program, 1.0 / embedding as f32);
    let eps = symbolic_leaf(&mut program, DType::Float32, "eps");
    let ones = scalar_constant(&mut program, 1.0);
    let inv_sqrt_head_dim = scalar_constant(&mut program, 1.0 / (head_dim as f32).sqrt());
    // only materialized when a layer actually consumes it (`qk_norm`), so a
    // dense checkpoint with no QK-norm keeps the identical node count this
    // function has always emitted.
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
    // Only ever consulted by `append_mistral_cached_layer`'s
    // `fused_qkv_reduce` branch (that parameter's own doc) -- built ONLY
    // when the flag is set, so `false` reproduces today's program
    // node-for-node (`cached_attention_rewrite_replaces_the_bound_attention_subgraph`'s
    // own literal bound-op-count fixture is the guard: it caught the
    // unconditional-`Op::Constant` version of this as a real +2 node
    // regression before this comment existed).
    let (head_shape_ones, kv_head_shape_ones) = if fused_qkv_reduce {
        let head_shape_ones = op::append(
            &mut program,
            Op::Constant {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(query_heads), Extent::Static(head_dim)],
                value: 1.0,
            },
        );
        let kv_head_shape_ones = op::append(
            &mut program,
            Op::Constant {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(head_dim)],
                value: 1.0,
            },
        );
        (head_shape_ones, kv_head_shape_ones)
    } else {
        // Never read (`append_mistral_cached_layer`'s `fused_qkv_reduce`
        // branch is the only reader, and it never runs when the flag is
        // `false`) -- `ones` (already built above) is reused as the
        // placeholder rather than adding an `Option` the callee would need
        // to `expect()` out of (this crate's own no-`expect`-in-production
        // rule), or building a real constant no `false` caller ever needs.
        (ones, ones)
    };
    let (is_future, _neg_infinity) = causal_mask(&mut program)?;
    // Rank-0 `Op::Input`, same precedent `eps`/`rope_cos`/`rope_sin` set
    // (`causal_mask_merged`'s own doc), named "cached_len" so
    // `proxima_tensor::bind::cached_attention_candidates` can find it by
    // name -- this crate's own precedent for what a name is for
    // (`Op::Input`'s own doc: "identity, not decoration"). It feeds no
    // arithmetic in this program: the host supplies the REAL `cached_len`
    // every call, independent of `kv_cache.{layer}.*`'s own
    // `Extent::Symbolic(1)` extent (which a caller may round up to a
    // bucket boundary, `ServingConfig::kv_bucket_tokens`, without
    // rebuilding this program), and the fused `BoundOpKind::CachedAttention`
    // reads it as a NINTH, runtime operand at execution time instead --
    // the bucket's own zero-padding is excluded by that bound, never by a
    // mask node in this graph (`BoundOpKind::CachedAttention`'s own doc on
    // the `cached_key_rows != 0` discriminator).
    let _cached_len = input_leaf(&mut program, DType::Float32, Vec::new(), "cached_len");

    let mut cache_roots: Vec<CachedLayerRoots> = Vec::with_capacity(block_count as usize);
    let mut layer_residuals: Vec<NodeId> = Vec::with_capacity(block_count as usize);
    let mut moe_sites: Vec<MoeSite> = Vec::new();

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
        let (wq, wk, wv) = if fused_qkv_reduce {
            let rows = (query_heads + 2 * kv_heads) * head_dim;
            let w_qkv = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(rows), Extent::Static(embedding)],
                &alloc::format!("blk.{layer}.attn_qkv.weight"),
            );
            (w_qkv, w_qkv, w_qkv)
        } else {
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
            (wq, wk, wv)
        };
        let q_bias = qkv_biases.then(|| {
            input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(query_heads), Extent::Static(head_dim)],
                &alloc::format!("blk.{layer}.attn_q.bias"),
            )
        });
        let k_bias = qkv_biases.then(|| {
            input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(kv_heads), Extent::Static(head_dim)],
                &alloc::format!("blk.{layer}.attn_k.bias"),
            )
        });
        let v_bias = qkv_biases.then(|| {
            input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(kv_heads), Extent::Static(head_dim)],
                &alloc::format!("blk.{layer}.attn_v.bias"),
            )
        });
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

        let (x_next, layer_roots) = if expert_count == 0 {
            let (w_gate, w_up) = if paired_gate_up_reduce {
                let w_gate_up = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Static(2),
                        Extent::Static(feed_forward),
                        Extent::Static(embedding)
                    ],
                    &alloc::format!("blk.{layer}.ffn_gate_up.weight"),
                );
                (w_gate_up, w_gate_up)
            } else {
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
                (w_gate, w_up)
            };
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

            append_mistral_cached_layer(
                &mut program,
                x,
                inv_dim,
                eps,
                ones,
                inv_sqrt_head_dim,
                cos_new,
                sin_new,
                group_ones,
                head_shape_ones,
                kv_head_shape_ones,
                is_future,
                group,
                head_dim,
                query_heads,
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
                q_bias,
                k_bias,
                v_bias,
                paired_gate_up_reduce,
                fused_qkv_reduce,
                rope_pairing,
            )?
        } else {
            let gate_inp = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(expert_count)],
                &alloc::format!("blk.{layer}.ffn_gate_inp.weight"),
            );
            let expert_w_gate = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(expert_count),
                    Extent::Static(embedding),
                    Extent::Static(feed_forward)
                ],
                &alloc::format!("blk.{layer}.ffn_gate_exps.weight"),
            );
            let expert_w_up = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(expert_count),
                    Extent::Static(embedding),
                    Extent::Static(feed_forward)
                ],
                &alloc::format!("blk.{layer}.ffn_up_exps.weight"),
            );
            let expert_w_down = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(expert_count),
                    Extent::Static(feed_forward),
                    Extent::Static(embedding)
                ],
                &alloc::format!("blk.{layer}.ffn_down_exps.weight"),
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

            let (next_x, next_roots, site) = append_mistral_cached_moe_layer(
                &mut program,
                layer,
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
                gate_inp,
                expert_w_gate,
                expert_w_up,
                expert_w_down,
                expert_count,
                expert_used_count,
                k_even_cache,
                k_odd_cache,
                v_cache,
                qk_norm_weights,
            )?;
            moe_sites.push(site);
            (next_x, next_roots)
        };
        x = x_next;
        cache_roots.push(layer_roots);
        layer_residuals.push(x_next);
    }

    let output_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding)],
        "output_norm.weight",
    );
    let normed_final = rmsnorm(&mut program, x, output_norm_weight, inv_dim, eps)?;

    // The decode loop only ever samples the LAST row's logits, prefill or
    // not (`proxima-model-interop::generate`'s own `logits[(new_count - 1)
    // * vocab_size..]` slice, every call site) -- greedy sampling needs one
    // row, never the whole prefill. Slicing here, before the vocab-sized
    // `output.weight` matmul, is what turns a 915-row Q6K reduce into a
    // 1-row one on an 850+-token prefill (`docs/discipline.md` ROW 418's
    // own `output.weight` measurement: 14.7s of 43.8s GPU time, 33% of
    // total, on the FULL 915-row projection). `embedding_lookup` is reused
    // verbatim, not a new primitive: it is already exactly `table[ids[s],
    // d]`, the same [`IndexMap::Computed`] gather this needs, just with a
    // 1-entry `lm_head_row` index instead of a `new_count`-entry `ids`.
    // `lm_head_row` is host-supplied (`new_count - 1`, same convention as
    // `cached_len`/`ids` above) rather than derived in-graph from
    // `Extent::Symbolic(0)`: `cpu.rs`'s own
    // `evaluate_typed_names_a_computed_gather_index_node_as_not_yet_supported`
    // test is this crate's own proof that an in-program-computed gather
    // index (an `Op::Iota`/`Op::Reduce` chain, not a caller-supplied
    // `Op::Input` block) is a named `NotLowerable` gap on the typed
    // evaluator, not a silently-guessed execution path -- a host-supplied
    // leaf is the one gather-index shape this crate's gather machinery
    // already proves correct end to end (`embedding_lookup`'s own `ids`).
    // `last_row_only: false` skips this leaf entirely (not merely bypasses
    // it) so the program a `false` caller gets is byte-for-byte the one
    // this function has always built -- no new node, no new required
    // binding, every existing per-position-logits caller unaffected.
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

    Ok((
        program,
        ForwardRoots {
            logits,
            hidden: normed_last,
        },
        cache_roots,
        layer_residuals,
        MoeSites(moe_sites),
    ))
}

/// Per-layer roots [`qwen35_forward_program`]'s own caller threads back in
/// as next-call cache [`Op::Input`]s -- [`Qwen35DenseAttentionRoots`]'s own
/// 4-wide KV-cache shape for a dense-attention layer
/// (`append_qwen35_dense_attention_layer`'s own doc walks through why it
/// is 4-wide, not [`CachedLayerRoots`]'s 3), or `append_qwen35_ssm_mixer`'s
/// own `(qkv_mixed, state_out)` return for an SSM layer. A discriminated
/// enum, not a bool flag riding alongside a fixed-shape tuple: the layer
/// kinds carry genuinely different cache shapes, the same reason
/// [`LayerKind`] exists as its own type rather than a boolean.
///
/// `Attention(CachedLayerRoots)` is [`mistral_cached_forward_program_with_experts`]'s
/// own 3-wide shape, still constructed by that program's caller
/// (`crate::generate::LoadedModel::load`) for every non-qwen35 checkpoint --
/// kept as its own variant rather than folded into `DenseAttention` so that
/// caller's cache-threading loop, and its `LayerCache`, are unaffected by
/// this checkpoint's own partial-rotary gap.
#[derive(Debug, Clone, Copy)]
pub enum Qwen35LayerRoots {
    Attention(CachedLayerRoots),
    DenseAttention(Qwen35DenseAttentionRoots),
    Ssm {
        qkv_mixed: NodeId,
        state_out: NodeId,
    },
}

/// Qwen3.5's whole-model incremental forward program: `full_attention_interval`
/// dense-attention layers (`append_mistral_cached_layer`, the same KV-cache
/// pattern [`mistral_cached_forward_program_with_experts`] already runs)
/// interleaved with gated-DeltaNet layers (`append_qwen35_ssm_mixer`),
/// following llama.cpp's own `hparams.is_recr_impl[i] = (i < n_layer) &&
/// ((i + 1) % full_attention_interval != 0)` (`qwen35.cpp:19-20`) -- layer
/// `full_attention_interval - 1`, `2 * full_attention_interval - 1`, ... are
/// dense attention, every other layer is SSM. Qwen3.5 never routes FFN
/// through experts (`qwen35.cpp:471`, `GGML_ASSERT(model.layers[il].ffn_gate_inp
/// == nullptr)`), so every layer's FFN is the plain dense triple
/// [`mistral_cached_forward_program_with_experts`]'s own `expert_count == 0`
/// branch already builds -- reused here rather than reconstructed.
///
/// `ssm_d_state`/`ssm_dt_rank`/`ssm_n_group`/`ssm_d_inner`/`ssm_d_conv` name
/// the same five hyperparameters `qwen35.cpp:335-343`'s own
/// `build_layer_attn_linear` reads off `hparams`, unpacked into
/// `append_qwen35_ssm_mixer`'s own `key_dim = ssm_d_state * ssm_n_group`,
/// `value_dim = ssm_d_inner`, `kv_heads = ssm_n_group`, `group = ssm_dt_rank
/// / ssm_n_group`, `l_cache = ssm_d_conv` (`head_v_dim = ssm_d_inner /
/// ssm_dt_rank` falls out inside the mixer itself, matching the oracle's own
/// `head_v_dim = d_inner / num_v_heads`). `rms_eps` is
/// `hparams.f_norm_rms_eps` baked as a graph-build-time constant, the same
/// choice this module already makes for `inv_dim`/`inv_sqrt_head_dim`
/// (Rust-side config values, not runtime-bound `Input`s) rather than a fresh
/// runtime-bound tensor shaped to `append_qwen35_ssm_mixer`'s own
/// `head_eps` (`[kv_heads, group]`) -- there is exactly one epsilon value
/// per checkpoint, known at program-build time.
///
/// Dense attention's own layers (`append_qwen35_dense_attention_layer`,
/// not `append_mistral_cached_layer`) run split-half RoPE over the
/// checkpoint's PARTIAL rotary width plus a concatenated-by-sum pass-through
/// remainder, and a per-head sigmoid gate on the attention output --
/// `append_qwen35_dense_attention_layer`'s own doc walks through why the
/// declared 3-section MRoPE (`rope.dimension_sections`) collapses to plain
/// single-section RoPE for this checkpoint's text-only forward program.
#[allow(clippy::too_many_arguments)]
pub fn qwen35_forward_program(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    attn_head_dim: u32,
    block_count: u32,
    full_attention_interval: u32,
    ssm_d_state: u32,
    ssm_dt_rank: u32,
    ssm_n_group: u32,
    ssm_d_inner: u32,
    ssm_d_conv: u32,
    rms_eps: f32,
) -> Result<(Vec<Op>, NodeId, Vec<Qwen35LayerRoots>), TensorError> {
    qwen35_forward_program_with_last_row(
        vocab,
        embedding,
        feed_forward,
        query_heads,
        kv_heads,
        head_dim,
        attn_head_dim,
        block_count,
        full_attention_interval,
        ssm_d_state,
        ssm_dt_rank,
        ssm_n_group,
        ssm_d_inner,
        ssm_d_conv,
        rms_eps,
        false,
    )
}

/// Builds the Qwen35 program while optionally reducing the final vocabulary
/// projection to a host-selected row before the packed weight is read.
#[allow(clippy::too_many_arguments)]
pub fn qwen35_forward_program_with_last_row(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    attn_head_dim: u32,
    block_count: u32,
    full_attention_interval: u32,
    ssm_d_state: u32,
    ssm_dt_rank: u32,
    ssm_n_group: u32,
    ssm_d_inner: u32,
    ssm_d_conv: u32,
    rms_eps: f32,
    last_row_only: bool,
) -> Result<(Vec<Op>, NodeId, Vec<Qwen35LayerRoots>), TensorError> {
    if full_attention_interval == 0 {
        return Err(TensorError::InvalidFullAttentionInterval {
            full_attention_interval,
        });
    }

    let group = query_heads / kv_heads;
    let pairs = head_dim / 2;
    let ssm_group = ssm_dt_rank / ssm_n_group;
    let ssm_key_dim = ssm_d_state * ssm_n_group;

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

    let inv_dim = scalar_constant(&mut program, 1.0 / embedding as f32);
    let eps = symbolic_leaf(&mut program, DType::Float32, "eps");
    let ones = scalar_constant(&mut program, 1.0);
    let one = ones;
    // `head_dim` here is `rope.dimension_count` -- this checkpoint's
    // PARTIAL-rotary width (`rotary_dim`), never the real per-head width.
    // Dense attention's own score scale is `attn_head_dim`-based
    // (`self.scaling = self.head_dim**-0.5` where `self.head_dim` is the
    // real width, `modeling_qwen3_next.py:262,264`), not
    // `rotary_dim`-based.
    let inv_sqrt_attn_head_dim = scalar_constant(&mut program, 1.0 / (attn_head_dim as f32).sqrt());
    let inv_attn_head_dim = scalar_constant(&mut program, 1.0 / attn_head_dim as f32);
    let inv_sqrt_key_dim = scalar_constant(&mut program, 1.0 / (ssm_d_state as f32).sqrt());
    let head_v_dim = ssm_d_inner / ssm_dt_rank;
    let inv_head_v_dim = scalar_constant(&mut program, 1.0 / head_v_dim as f32);
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
    let head_eps = op::append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(ssm_n_group), Extent::Static(ssm_group)],
            value: rms_eps,
        },
    );
    let (is_future, _neg_infinity) = causal_mask(&mut program)?;
    // Same rank-0 leaf [`mistral_cached_forward_program_with_experts`] adds
    // right after its own `causal_mask` call, and for the same reason: named
    // "cached_len" so `bind::cached_attention_candidates`'s `find_named_input`
    // picks it up by NAME on the `Attention` arm's fused `CachedAttention`
    // op. The `DenseAttention` arm has no equivalent fusion, so this same
    // node is ALSO threaded directly into every
    // [`append_qwen35_dense_attention_layer`] call below to mask its own
    // padded cached range (that function's own doc).
    let cached_len = input_leaf(&mut program, DType::Float32, Vec::new(), "cached_len");

    let mut layer_roots: Vec<Qwen35LayerRoots> = Vec::with_capacity(block_count as usize);

    for layer in 0..block_count {
        let attn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.attn_norm.weight"),
        );
        // Named `post_attention_norm.weight` on disk, not `ffn_norm.weight`
        // -- this checkpoint's own GGUF writer names this tensor
        // differently from every other architecture this crate binds
        // (`proxima_model_interop::qwen35`'s own module doc, confirmed via
        // `strings` on the real file: no `blk.N.ffn_norm.weight` key
        // exists anywhere), on both layer kinds.
        let ffn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.post_attention_norm.weight"),
        );

        // `hparams.is_recr_impl[i] = (i + 1) % full_attention_interval != 0`
        // (`qwen35.cpp:19-20`) is TRUE for SSM layers -- dense attention is
        // its negation, `(i + 1) % full_attention_interval == 0`.
        let is_dense_attention = (layer + 1) % full_attention_interval == 0;

        let (x_next, roots) = if is_dense_attention {
            // real per-head width read off metadata (`attention.key_length`,
            // `attn_head_dim` param) rather than `embedding / query_heads`
            // -- the latter is not even an integer on the 27B checkpoint
            // (`5120 / 24 = 213.33`), confirmed wrong against the real file
            // by [`crate::qwen35::qwen35_architecture_from_metadata`]'s own
            // caller-side doc.
            let wq_flat = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(embedding),
                    Extent::Static(query_heads * attn_head_dim * 2)
                ],
                &alloc::format!("blk.{layer}.attn_q.weight"),
            );
            // A pure reshape (multiply by a broadcast-ones constant), the
            // same lossless-reshape donor trick `wk`/`wv` already use below
            // -- NEVER `per_head_channel_slice` on this packed leaf. That
            // per-head WEIGHT-level slice inserted a select-then-reduce
            // between `wq_flat` and the real contraction, which
            // `is_quantized_matmul_operand`/`run_reduce_quantized` (`cpu.rs`)
            // then misidentifies as the whole quantized matmul shape and
            // derives `rows`/`k` from the wrong axis pair -- `q`/`gate` now
            // split on the ACTIVATION side instead, inside
            // [`append_qwen35_dense_attention_only_with_taps`], via
            // [`per_head_channel_range`].
            let qg_head_ones = op::append(
                &mut program,
                Op::Constant {
                    dtype: DType::Float32,
                    shape: alloc::vec![
                        Extent::Static(query_heads),
                        Extent::Static(attn_head_dim * 2)
                    ],
                    value: 1.0,
                },
            );
            let wq_gate = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[
                    (
                        wq_flat,
                        alloc::format!("i,{}*h+c->ihc", attn_head_dim * 2).as_str(),
                    ),
                    (qg_head_ones, "hc->ihc"),
                ],
            )?;
            let wk_flat = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(embedding),
                    Extent::Static(kv_heads * attn_head_dim)
                ],
                &alloc::format!("blk.{layer}.attn_k.weight"),
            );
            // `k` carries no gate and no partial-rotary truncation at the
            // weight level (the split into rotated/pass halves happens on
            // the ACTIVATION inside [`append_qwen35_dense_attention_layer`]
            // now that `q_norm`/`k_norm` need the full width first) -- the
            // same lossless-reshape donor trick `v`/`o` already use below.
            let k_head_ones = op::append(
                &mut program,
                Op::Constant {
                    dtype: DType::Float32,
                    shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(attn_head_dim)],
                    value: 1.0,
                },
            );
            let wk = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[
                    (
                        wk_flat,
                        alloc::format!("i,{attn_head_dim}*u+d->iud").as_str(),
                    ),
                    (k_head_ones, "ud->iud"),
                ],
            )?;
            let v_head_ones = op::append(
                &mut program,
                Op::Constant {
                    dtype: DType::Float32,
                    shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(attn_head_dim)],
                    value: 1.0,
                },
            );
            let o_head_ones = op::append(
                &mut program,
                Op::Constant {
                    dtype: DType::Float32,
                    shape: alloc::vec![
                        Extent::Static(kv_heads),
                        Extent::Static(group),
                        Extent::Static(attn_head_dim)
                    ],
                    value: 1.0,
                },
            );
            let wv_flat = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(embedding),
                    Extent::Static(kv_heads * attn_head_dim)
                ],
                &alloc::format!("blk.{layer}.attn_v.weight"),
            );
            let wv = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[
                    (
                        wv_flat,
                        alloc::format!("i,{attn_head_dim}*u+d->iud").as_str(),
                    ),
                    (v_head_ones, "ud->iud"),
                ],
            )?;
            let wo_flat = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(query_heads * attn_head_dim),
                    Extent::Static(embedding)
                ],
                &alloc::format!("blk.{layer}.attn_output.weight"),
            );
            let wo = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[
                    (
                        wo_flat,
                        alloc::format!("{}*u+{attn_head_dim}*g+d,e->ugde", attn_head_dim * group)
                            .as_str(),
                    ),
                    (o_head_ones, "ugd->ugde"),
                ],
            )?;
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
            let q_norm_weight = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(attn_head_dim)],
                &alloc::format!("blk.{layer}.attn_q_norm.weight"),
            );
            let k_norm_weight = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(attn_head_dim)],
                &alloc::format!("blk.{layer}.attn_k_norm.weight"),
            );
            let pass_dim = attn_head_dim - head_dim;
            let k_first_cache = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Symbolic(1),
                    Extent::Static(kv_heads),
                    Extent::Static(pairs)
                ],
                &alloc::format!("kv_cache.{layer}.k_first"),
            );
            let k_second_cache = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Symbolic(1),
                    Extent::Static(kv_heads),
                    Extent::Static(pairs)
                ],
                &alloc::format!("kv_cache.{layer}.k_second"),
            );
            let k_pass_cache = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Symbolic(1),
                    Extent::Static(kv_heads),
                    Extent::Static(pass_dim)
                ],
                &alloc::format!("kv_cache.{layer}.k_pass"),
            );
            let v_cache = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Symbolic(1),
                    Extent::Static(kv_heads),
                    Extent::Static(attn_head_dim)
                ],
                &alloc::format!("kv_cache.{layer}.v"),
            );

            let (x_next, dense_attention_roots) = append_qwen35_dense_attention_layer(
                &mut program,
                x,
                inv_dim,
                eps,
                ones,
                inv_sqrt_attn_head_dim,
                inv_attn_head_dim,
                cos_new,
                sin_new,
                group_ones,
                is_future,
                cached_len,
                group,
                head_dim,
                attn_head_dim,
                attn_norm_weight,
                ffn_norm_weight,
                q_norm_weight,
                k_norm_weight,
                wq_gate,
                wk,
                wv,
                wo,
                w_gate,
                w_up,
                w_down,
                k_first_cache,
                k_second_cache,
                k_pass_cache,
                v_cache,
            )?;
            (
                x_next,
                Qwen35LayerRoots::DenseAttention(dense_attention_roots),
            )
        } else {
            let qkv_dim = 2 * ssm_key_dim + ssm_d_inner;
            let wqkv = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(qkv_dim)],
                &alloc::format!("blk.{layer}.ssm_in.weight"),
            );
            let wqkv_gate = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(ssm_d_inner)],
                &alloc::format!("blk.{layer}.ssm_gate.weight"),
            );
            let conv_weight = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(qkv_dim), Extent::Static(ssm_d_conv)],
                &alloc::format!("blk.{layer}.ssm_conv1d.weight"),
            );
            let conv_history_in = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(ssm_d_conv - 1), Extent::Static(qkv_dim)],
                &alloc::format!("ssm_cache.{layer}.conv_history"),
            );
            let ssm_beta = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(ssm_dt_rank)],
                &alloc::format!("blk.{layer}.ssm_beta.weight"),
            );
            let ssm_alpha = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(ssm_dt_rank)],
                &alloc::format!("blk.{layer}.ssm_alpha.weight"),
            );
            let ssm_dt_bias = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(ssm_dt_rank)],
                &alloc::format!("blk.{layer}.ssm_dt.bias"),
            );
            let ssm_a = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(ssm_dt_rank)],
                &alloc::format!("blk.{layer}.ssm_a"),
            );
            let ssm_norm_weight = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(head_v_dim)],
                &alloc::format!("blk.{layer}.ssm_norm.weight"),
            );
            let ssm_out = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(ssm_d_inner), Extent::Static(embedding)],
                &alloc::format!("blk.{layer}.ssm_out.weight"),
            );
            let state_in = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(ssm_d_state),
                    Extent::Static(head_v_dim),
                    Extent::Static(ssm_n_group),
                    Extent::Static(ssm_group)
                ],
                &alloc::format!("ssm_cache.{layer}.state"),
            );

            let (mixer_out, qkv_mixed, state_out) = append_qwen35_ssm_mixer(
                &mut program,
                x,
                inv_dim,
                eps,
                head_eps,
                one,
                inv_sqrt_key_dim,
                inv_head_v_dim,
                Some(attn_norm_weight),
                wqkv,
                wqkv_gate,
                conv_weight,
                conv_history_in,
                ssm_beta,
                ssm_alpha,
                ssm_dt_bias,
                ssm_a,
                ssm_norm_weight,
                ssm_out,
                state_in,
                ssm_key_dim,
                ssm_d_inner,
                ssm_n_group,
                ssm_group,
                ssm_d_conv,
                GdnOutputGate::Silu,
            )?;

            // Unlike `append_mistral_cached_layer` (bundles FFN internally),
            // `append_qwen35_ssm_mixer` is mixer-plus-residual only -- the
            // same scope `append_lfm2_conv_mixer` has -- so the SSM branch
            // runs its own dense FFN pass here, matching
            // `mistral_cached_forward_program_with_experts`'s own
            // `expert_count == 0` FFN math exactly (Qwen3.5 never routes FFN
            // through experts, `qwen35.cpp:471`).
            let normed2 = rmsnorm(&mut program, mixer_out, ffn_norm_weight, inv_dim, eps)?;
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
            let gate_product = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(normed2, "sd->sdg"), (w_gate, "dg->sdg")],
            )?;
            let gate = reduce(
                &mut program,
                DType::Float32,
                ScalarOp::Add,
                ReduceInit::Zero,
                gate_product,
                "sdg->sdg",
                "sg->sdg",
            )?;
            let up_product = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(normed2, "sd->sdg"), (w_up, "dg->sdg")],
            )?;
            let up = reduce(
                &mut program,
                DType::Float32,
                ScalarOp::Add,
                ReduceInit::Zero,
                up_product,
                "sdg->sdg",
                "sg->sdg",
            )?;
            let silu_gate = silu(&mut program, gate, one, "sg->sg")?;
            let ffn_hidden = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(silu_gate, "sg->sg"), (up, "sg->sg")],
            )?;
            let down_product = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(ffn_hidden, "sg->sgd"), (w_down, "gd->sgd")],
            )?;
            let ffn_out = reduce(
                &mut program,
                DType::Float32,
                ScalarOp::Add,
                ReduceInit::Zero,
                down_product,
                "sgd->sgd",
                "sd->sgd",
            )?;
            let x_after_ffn = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Add,
                &[(ffn_out, "sd->sd"), (mixer_out, "sd->sd")],
            )?;

            (
                x_after_ffn,
                Qwen35LayerRoots::Ssm {
                    qkv_mixed,
                    state_out,
                },
            )
        };

        x = x_next;
        layer_roots.push(roots);
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

    Ok((program, logits, layer_roots))
}

