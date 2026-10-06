use super::*;

/// The attention shape every layer of a hybrid schedule carries (a recurrent
/// layer holds one too, unread): the recurrent-hybrid engines lower one shape,
/// so every layer must agree with the first except for `kv_heads`, which the
/// routed engine reads per layer. The engines lower the gated attention variant
/// only, so a descriptor that does not set [`ModelDescriptor::gated_attention`]
/// is refused instead of lowered as something else.
fn hybrid_attention<'schedule>(
    descriptor: &'schedule ModelDescriptor,
    builder: &'static str,
) -> Result<&'schedule LayerAttentionConfig, TensorError> {
    let first = descriptor.layers.first().ok_or(TensorError::UnsupportedInBuilder {
        builder,
        feature: "an empty schedule",
    })?;
    refuse_when(
        descriptor
            .layers
            .iter()
            .any(|layer| LayerAttentionConfig { kv_heads: first.attention.kv_heads, ..layer.attention.clone() } != first.attention),
        builder,
        "layers that differ in anything but kv heads",
    )?;
    refuse_when(!descriptor.gated_attention, builder, "attention that is not gated")?;
    Ok(&first.attention)
}

/// The rotating width of the attention shape: this engine rotates the
/// leading `2 * pairs` channels split-half and passes the rest through.
fn rotary_dim(attention: &LayerAttentionConfig, builder: &'static str) -> Result<u32, TensorError> {
    match attention.rope_pairing {
        RopePairing::SplitHalf { pairs } => Ok(pairs * 2),
        RopePairing::Interleaved => Err(TensorError::UnsupportedInBuilder {
            builder,
            feature: "interleaved rope over a partial rotary width",
        }),
    }
}

/// `layers[i].ffn` must be the plain SwiGLU the dense hybrid lowers: the
/// routed-gating fields a dense layer never reads are free, everything else
/// must be the [`LayerFfnConfig::exclusive`] default.
fn refuse_unless_plain_swiglu(layers: &[LayerSchedule], builder: &'static str) -> Result<(), TensorError> {
    let plain = LayerFfnConfig::exclusive();
    refuse_when(
        layers.iter().any(|layer| {
            LayerFfnConfig { routed_gating: plain.routed_gating, routed_expert_bias: plain.routed_expert_bias, ..layer.ffn } != plain
        }),
        builder,
        "an ffn other than plain silu swiglu",
    )
}

/// Gated-DeltaNet layers interleaved with gated dense attention over a dense
/// SwiGLU FFN, lowered from one [`ModelDescriptor`]: the layer kinds come from
/// `layers[i].kind`, the recurrence shape from the `ssm_*` fields, and the
/// attention shape from the first attention layer. This is the program
/// `qwen35_forward_program_with_last_row` has always built, with the
/// `(layer + 1) % full_attention_interval` predicate replaced by the schedule.
pub(super) fn hybrid_dense_forward(descriptor: &ModelDescriptor) -> Result<ForwardProgram, TensorError> {
    const BUILDER: &str = "build_forward(hybrid dense)";
    refuse_when(descriptor.layers.len() != descriptor.block_count as usize, BUILDER, "a schedule whose length is not block_count")?;
    refuse_when(descriptor.layers.iter().any(|layer| layer.kind == LayerKind::ShortConv), BUILDER, "a short convolution layer")?;
    refuse_when(descriptor.expert_count != 0, BUILDER, "routed experts")?;
    refuse_unless_plain_swiglu(&descriptor.layers, BUILDER)?;
    let attention = hybrid_attention(descriptor, BUILDER)?;
    refuse_when(
        descriptor.layers.iter().any(|layer| layer.attention.kv_heads != attention.kv_heads),
        BUILDER,
        "layers that differ in kv heads",
    )?;
    refuse_when(descriptor.ssm_time_step_rank == 0 || descriptor.ssm_group_count == 0 || descriptor.ssm_conv_kernel == 0, BUILDER, "a recurrence shape with a zero extent")?;

    let vocab = descriptor.vocab;
    let embedding = descriptor.embedding;
    let feed_forward = descriptor.feed_forward;
    let query_heads = descriptor.query_heads;
    let kv_heads = attention.kv_heads;
    let head_dim = rotary_dim(attention, BUILDER)?;
    let attn_head_dim = attention.head_dim;
    let score_multiplier = attention.score_scale.multiplier();
    let block_count = descriptor.block_count;
    let ssm_d_state = descriptor.ssm_state_size;
    let ssm_dt_rank = descriptor.ssm_time_step_rank;
    let ssm_n_group = descriptor.ssm_group_count;
    let ssm_d_inner = descriptor.ssm_inner_size;
    let ssm_d_conv = descriptor.ssm_conv_kernel;
    let rms_eps = descriptor.ssm_epsilon;
    let last_row_only = descriptor.last_row_only;

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
    let inv_sqrt_attn_head_dim = scalar_constant(&mut program, score_multiplier);
    let inv_attn_head_dim = scalar_constant(&mut program, 1.0 / attn_head_dim as f32);
    let inv_sqrt_key_dim = scalar_constant(&mut program, 1.0 / libm::sqrtf(ssm_d_state as f32));
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

        let is_dense_attention = descriptor.layers[layer as usize].kind == LayerKind::Attention;

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

    let normed_last = gather_last_row(&mut program, normed_final, last_row_only);
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

    Ok(ForwardProgram {
        program,
        logits,
        layer_roots,
        moe_sites: MoeSites::default(),
        layer_residuals: Vec::new(),
        hidden: None,
        duplicate_head_roots: Vec::new(),
        layer_diagnostics: Vec::new(),
    })
}

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

fn append_qwen35moe_router(
    program: &mut Vec<Op>,
    mixer_out: NodeId,
    post_attention_norm_weight: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    gate_inp: NodeId,
) -> Result<(NodeId, NodeId), TensorError> {
    let normed = rmsnorm(program, mixer_out, post_attention_norm_weight, inv_dim, eps)?;
    let gate_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "sd->sde"), (gate_inp, "de->sde")],
    )?;
    let router_logits = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        gate_product,
        "sde->sde",
        "se->sde",
    )?;
    Ok((normed, router_logits))
}

/// Runs `x`'s post-attention-norm hidden state through the routed
/// [`append_moe_ffn`] plus the gated shared expert, and adds the result back
/// onto `mixer_out` (the pre-FFN residual stream) -- the one FFN sub-block
/// shape every `qwen35moe` layer shares, dense-attention or GDN alike.
#[allow(clippy::too_many_arguments)]
fn append_qwen35moe_ffn(
    program: &mut Vec<Op>,
    layer: u32,
    mixer_out: NodeId,
    post_attention_norm_weight: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    one: NodeId,
    gate_inp: NodeId,
    expert_w_gate: NodeId,
    expert_w_up: NodeId,
    expert_w_down: NodeId,
    expert_count: u32,
    expert_used_count: u32,
    gate_inp_shexp: NodeId,
    gate_shexp: NodeId,
    up_shexp: NodeId,
    down_shexp: NodeId,
) -> Result<(NodeId, MoeSite, NodeId, NodeId, NodeId, NodeId), TensorError> {
    // `append_moe_ffn`'s own leading two ops, shared with the batched GDN
    // prefill route so both paths select experts from the same algebra.
    let (normed, router_logits) = append_qwen35moe_router(
        program,
        mixer_out,
        post_attention_norm_weight,
        inv_dim,
        eps,
        gate_inp,
    )?;

    let moe_spec = MoeFfnSpec {
        router: MoeRouter::Logits(router_logits),
        expert_w_gate,
        expert_w_up,
        expert_w_down,
        expert_count,
        expert_used_count,
        ones: one,
        gating: ExpertGatingFunc::Softmax,
        expert_bias: None,
        expert_scale: None,
        activation: Activation::Silu,
        strategy: MoeProjectionStrategy::PerRoute,
    };
    let (routed_out, moe_site) = append_moe_ffn(program, layer, normed, &moe_spec)?;
    let shared_out = append_sigmoid_gated_shared_expert(
        program,
        normed,
        gate_inp_shexp,
        gate_shexp,
        up_shexp,
        down_shexp,
        one,
    )?;

    let ffn_out = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(routed_out, "sd->sd"), (shared_out, "sd->sd")],
    )?;
    let residual = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(ffn_out, "sd->sd"), (mixer_out, "sd->sd")],
    )?;
    Ok((
        residual,
        moe_site,
        normed,
        router_logits,
        routed_out,
        shared_out,
    ))
}

/// One layer's own diagnostic [`NodeId`]s -- production-neutral: every field
/// is a root the forward program already computes for its normal decode
/// work, just named and returned so a test can request them from
/// `proxima_model_interop::LoadedModel::forward_node_values` without
/// hand-counting `Op::Input`s the way earlier bisection tests in this crate
/// did. `ssm_taps` is `Some` only for [`LayerKind::Gdn`] layers,
/// `dense_attention_taps` only for [`LayerKind::Attention`] layers
/// -- exactly one of the two is `Some` for any given layer.
///
/// `mixer_output` is the mixer's own PRE-residual result (`taps.ssm_out_result`
/// for a GDN layer, `taps.o_proj_out` for a dense-attention layer --
/// [`proxima_tensor::spec::Qwen35DenseAttentionTaps::o_proj_out`], the
/// `o_proj` reduce before its own residual add); `post_mixer_residual` is
/// that result added back onto this layer's `block_input` (what the mixer
/// builders themselves call `mixer_out`/`x_next`/`residual1`).
#[derive(Debug, Clone)]
pub struct Qwen35MoeLayerDiagnostics {
    pub block_input: NodeId,
    pub ssm_taps: Option<SsmMixerTaps>,
    pub dense_attention_taps: Option<Qwen35DenseAttentionTaps>,
    pub mixer_output: NodeId,
    pub post_mixer_residual: NodeId,
    pub post_attention_norm_output: NodeId,
    pub router_logits: NodeId,
    pub routed_output: NodeId,
    pub shared_output: NodeId,
    pub block_output: NodeId,
}

/// Gated-DeltaNet layers interleaved with gated dense attention over a routed
/// FFN plus a sigmoid-gated shared expert, lowered from one [`ModelDescriptor`]:
/// layer kinds from `layers[i].kind`, the recurrence shape from the `ssm_*`
/// fields, the routed experts from `expert_*`, and the shared expert from
/// `expert_shared_feed_forward`. `prefill_width` pins the position axis to a
/// literal extent, which is the only way
/// [`append_qwen35_ssm_mixer_with_taps_and_layout`]'s M > 1 branch (its scan is
/// unrolled in Rust) is reached; the symbolic per-step program takes the
/// single-position path regardless of how many rows resolve at run time.
///
/// Teaching pointer: composes [`embedding_lookup`], [`causal_mask`],
/// [`append_qwen35_dense_attention_only_with_taps`] and
/// [`append_qwen35_ssm_mixer_with_taps_and_layout`] per layer, [`rmsnorm`] for the
/// post-attention norm, [`append_moe_ffn`] for the routed FFN,
/// [`append_sigmoid_gated_shared_expert`] for the shared expert, and
/// [`elementwise`]/[`reduce`] for the residual adds and `lm_head`.
#[allow(clippy::too_many_lines)]
pub(super) fn hybrid_routed_forward(descriptor: &ModelDescriptor) -> Result<ForwardProgram, TensorError> {
    const BUILDER: &str = "build_forward(hybrid routed)";
    refuse_when(descriptor.layers.len() != descriptor.block_count as usize, BUILDER, "a schedule whose length is not block_count")?;
    refuse_when(descriptor.layers.iter().any(|layer| layer.kind == LayerKind::ShortConv), BUILDER, "a short convolution layer")?;
    refuse_when(descriptor.layers.iter().any(|layer| layer.ffn.combination != FfnCombination::RoutedWithSharedExpert), BUILDER, "a layer whose ffn is not routed with a shared expert")?;
    refuse_when(
        descriptor.layers.iter().any(|layer| layer.ffn.routed_expert_bias || layer.ffn.routed_gating != ExpertGatingFunc::Softmax || layer.ffn.activation != Activation::Silu),
        BUILDER,
        "a routed ffn other than softmax gating over silu experts without a bias",
    )?;
    refuse_when(descriptor.ssm_time_step_rank == 0 || descriptor.ssm_group_count == 0 || descriptor.ssm_conv_kernel == 0, BUILDER, "a recurrence shape with a zero extent")?;
    let attention = hybrid_attention(descriptor, BUILDER)?;
    let embedding = descriptor.embedding;
    let attn_head_dim = attention.head_dim;
    let rope_dims = rotary_dim(attention, BUILDER)?;
    let pairs = rope_dims / 2;
    let pass_dim = attn_head_dim - rope_dims;
    let x_extent = match descriptor.prefill_width {
        Some(width) => Extent::Static(width),
        None => Extent::Symbolic(0),
    };

    let mut program = Vec::new();

    let ids = input_leaf(&mut program, DType::Int32, alloc::vec![x_extent], "ids");
    let table = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(descriptor.vocab),
            Extent::Static(embedding),
        ],
        "token_embd.weight",
    );
    let mut x = embedding_lookup(&mut program, table, ids);

    let inv_dim = scalar_constant(&mut program, 1.0 / embedding as f32);
    let eps = input_leaf(&mut program, DType::Float32, alloc::vec![x_extent], "eps");
    let ones = scalar_constant(&mut program, 1.0);
    let one = ones;

    let inv_sqrt_attn_head_dim = scalar_constant(&mut program, attention.score_scale.multiplier());
    let inv_attn_head_dim = scalar_constant(&mut program, 1.0 / attn_head_dim as f32);
    let cos_new = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![x_extent, Extent::Static(pairs)],
        "rope_cos",
    );
    let sin_new = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![x_extent, Extent::Static(pairs)],
        "rope_sin",
    );

    let ssm_group = descriptor.ssm_time_step_rank / descriptor.ssm_group_count.max(1);
    let ssm_key_dim = descriptor.ssm_state_size * descriptor.ssm_group_count;
    let head_v_dim = descriptor.ssm_inner_size / descriptor.ssm_time_step_rank.max(1);
    let inv_sqrt_key_dim = scalar_constant(
        &mut program,
        1.0 / libm::sqrtf(descriptor.ssm_state_size as f32),
    );
    let inv_head_v_dim = scalar_constant(&mut program, 1.0 / head_v_dim.max(1) as f32);
    let head_eps = op::append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![
                Extent::Static(descriptor.ssm_group_count),
                Extent::Static(ssm_group),
            ],
            value: descriptor.ssm_epsilon,
        },
    );

    let (is_future, _neg_infinity) = causal_mask(&mut program)?;
    let cached_len = input_leaf(&mut program, DType::Float32, Vec::new(), "cached_len");

    let mut layer_roots: Vec<Qwen35LayerRoots> = Vec::with_capacity(descriptor.layers.len());
    let mut moe_sites: Vec<MoeSite> = Vec::with_capacity(descriptor.layers.len());
    let mut layer_diagnostics: Vec<Qwen35MoeLayerDiagnostics> =
        Vec::with_capacity(descriptor.layers.len());

    for (layer, schedule) in descriptor.layers.iter().enumerate() {
        let kind = schedule.kind;
        let layer = layer as u32;
        let block_input = x;

        let attn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.attn_norm.weight"),
        );
        let post_attention_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.post_attention_norm.weight"),
        );
        let gate_inp = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(embedding),
                Extent::Static(descriptor.expert_count),
            ],
            &alloc::format!("blk.{layer}.ffn_gate_inp.weight"),
        );

        let (mixer_out, ssm_taps, dense_attention_taps, mixer_output_pre_residual) = match kind {
            LayerKind::Attention => {
                let kv_heads = schedule.attention.kv_heads;
                let group = descriptor.query_heads / kv_heads.max(1);
                let group_ones = op::append(
                    &mut program,
                    Op::Constant {
                        dtype: DType::Float32,
                        shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
                        value: 1.0,
                    },
                );

                // `attn_q.weight` carries `[Q | gate]` per head, doubled
                // width -- the same fused Q-gate `proxima_tensor::spec`'s own
                // real `qwen35` dense checkpoint reads off `attn_q.weight`
                // (`spec.rs:8908-8944`). Reshaped via the same lossless
                // broadcast-multiply-by-ones trick `wk`/`wv` use below, NEVER
                // a per-head WEIGHT-level slice -- splitting a packed
                // quantized weight per head before the real contraction runs
                // breaks `cpu::is_quantized_matmul_operand`'s recognizer,
                // which then derives the packed row length from the wrong
                // axis (`per_head_channel_slice`'s own former call site here).
                // `append_qwen35_dense_attention_only_with_taps` does the
                // real `x_normed @ wq_gate` contraction and narrows to
                // `q`/`gate` per head on the ACTIVATION via
                // `per_head_channel_range`.
                let wq_flat = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Static(embedding),
                        Extent::Static(descriptor.query_heads * attn_head_dim * 2),
                    ],
                    &alloc::format!("blk.{layer}.attn_q.weight"),
                );
                let qg_head_ones = op::append(
                    &mut program,
                    Op::Constant {
                        dtype: DType::Float32,
                        shape: alloc::vec![
                            Extent::Static(descriptor.query_heads),
                            Extent::Static(attn_head_dim * 2),
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

                // `wk`/`wv`/`wo` reshape their own flat matmul-bound leaf via
                // the same lossless broadcast-multiply-by-ones trick
                // proxima's own dense `qwen35_forward_program` uses
                // (`proxima-tensor/src/spec.rs:8777-8853`) rather than
                // declaring the multi-axis shape directly on the `Op::Input`
                // leaf: the leaf's on-disk bytes are `bind_matmul_weight`'s
                // dequantized-and-transposed `[in_dim, out_dim]` buffer
                // (`bind.rs`'s `dense_attention_out_in_dims`), and a flat 2-D
                // leaf declaration is what actually matches that buffer's
                // real byte order -- baking `kv_heads`/`attn_head_dim` into
                // the leaf's own axis list here left the interpreter reading
                // those bytes under the wrong axis order.
                let wk_flat = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Static(embedding),
                        Extent::Static(kv_heads * attn_head_dim),
                    ],
                    &alloc::format!("blk.{layer}.attn_k.weight"),
                );
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
                        (wk_flat, alloc::format!("i,{attn_head_dim}*u+d->iud").as_str()),
                        (k_head_ones, "ud->iud"),
                    ],
                )?;

                let wv_flat = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Static(embedding),
                        Extent::Static(kv_heads * attn_head_dim),
                    ],
                    &alloc::format!("blk.{layer}.attn_v.weight"),
                );
                let v_head_ones = op::append(
                    &mut program,
                    Op::Constant {
                        dtype: DType::Float32,
                        shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(attn_head_dim)],
                        value: 1.0,
                    },
                );
                let wv = elementwise(
                    &mut program,
                    DType::Float32,
                    ScalarOp::Multiply,
                    &[
                        (wv_flat, alloc::format!("i,{attn_head_dim}*u+d->iud").as_str()),
                        (v_head_ones, "ud->iud"),
                    ],
                )?;

                let wo_flat = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Static(descriptor.query_heads * attn_head_dim),
                        Extent::Static(embedding),
                    ],
                    &alloc::format!("blk.{layer}.attn_output.weight"),
                );
                let o_head_ones = op::append(
                    &mut program,
                    Op::Constant {
                        dtype: DType::Float32,
                        shape: alloc::vec![
                            Extent::Static(kv_heads),
                            Extent::Static(group),
                            Extent::Static(attn_head_dim),
                        ],
                        value: 1.0,
                    },
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

                let k_first_cache = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Symbolic(1),
                        Extent::Static(kv_heads),
                        Extent::Static(pairs),
                    ],
                    &alloc::format!("kv_cache.{layer}.k_first"),
                );
                let k_second_cache = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Symbolic(1),
                        Extent::Static(kv_heads),
                        Extent::Static(pairs),
                    ],
                    &alloc::format!("kv_cache.{layer}.k_second"),
                );
                let k_pass_cache = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Symbolic(1),
                        Extent::Static(kv_heads),
                        Extent::Static(pass_dim),
                    ],
                    &alloc::format!("kv_cache.{layer}.k_pass"),
                );
                let v_cache = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Symbolic(1),
                        Extent::Static(kv_heads),
                        Extent::Static(attn_head_dim),
                    ],
                    &alloc::format!("kv_cache.{layer}.v"),
                );

                // `_with_taps` -- byte-identical program to
                // `append_qwen35_dense_attention_only`
                // (`dense_attention_only_and_with_taps_produce_the_same_program`
                // proves it in proxima-tensor), so this is a diagnostic-only
                // change, never a production behaviour change.
                let (residual1, dense_taps) = append_qwen35_dense_attention_only_with_taps(
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
                    rope_dims,
                    attn_head_dim,
                    attn_norm_weight,
                    q_norm_weight,
                    k_norm_weight,
                    wq_gate,
                    wk,
                    wv,
                    wo,
                    k_first_cache,
                    k_second_cache,
                    k_pass_cache,
                    v_cache,
                )?;
                layer_roots.push(Qwen35LayerRoots::DenseAttention((
                    dense_taps.rotated_k_new_first,
                    dense_taps.rotated_k_new_second,
                    dense_taps.k_pass,
                    dense_taps.v_new,
                )));
                let o_proj_out = dense_taps.o_proj_out;
                (residual1, None, Some(dense_taps), o_proj_out)
            }
            LayerKind::ShortConv => {
                return Err(TensorError::UnsupportedInBuilder {
                    builder: BUILDER,
                    feature: "a short convolution layer",
                });
            }
            LayerKind::Gdn => {
                let qkv_dim = 2 * ssm_key_dim + descriptor.ssm_inner_size;
                let wqkv = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(embedding), Extent::Static(qkv_dim)],
                    &alloc::format!("blk.{layer}.attn_qkv.weight"),
                );
                let wqkv_gate = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Static(embedding),
                        Extent::Static(descriptor.ssm_inner_size),
                    ],
                    &alloc::format!("blk.{layer}.attn_gate.weight"),
                );
                let conv_weight = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Static(qkv_dim),
                        Extent::Static(descriptor.ssm_conv_kernel),
                    ],
                    &alloc::format!("blk.{layer}.ssm_conv1d.weight"),
                );
                let conv_history_in = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Static(descriptor.ssm_conv_kernel.saturating_sub(1)),
                        Extent::Static(qkv_dim),
                    ],
                    &alloc::format!("ssm_cache.{layer}.conv_history"),
                );
                let ssm_beta = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Static(embedding),
                        Extent::Static(descriptor.ssm_time_step_rank),
                    ],
                    &alloc::format!("blk.{layer}.ssm_beta.weight"),
                );
                let ssm_alpha = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Static(embedding),
                        Extent::Static(descriptor.ssm_time_step_rank),
                    ],
                    &alloc::format!("blk.{layer}.ssm_alpha.weight"),
                );
                let ssm_dt_bias = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(descriptor.ssm_time_step_rank)],
                    &alloc::format!("blk.{layer}.ssm_dt"),
                );
                let ssm_a = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(descriptor.ssm_time_step_rank)],
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
                    alloc::vec![
                        Extent::Static(descriptor.ssm_inner_size),
                        Extent::Static(embedding),
                    ],
                    &alloc::format!("blk.{layer}.ssm_out.weight"),
                );
                let state_in = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Static(descriptor.ssm_state_size),
                        Extent::Static(head_v_dim),
                        Extent::Static(descriptor.ssm_group_count),
                        Extent::Static(ssm_group),
                    ],
                    &alloc::format!("ssm_cache.{layer}.state"),
                );

                // `_with_taps` -- byte-identical program to
                // `append_qwen35_ssm_mixer` (that wrapper's own doc: "this
                // only reshapes the return value the shared builder already
                // computed"), so this is a diagnostic-only change, never a
                // production behaviour change.
                let (mixer_out, taps) = append_qwen35_ssm_mixer_with_taps_and_layout(
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
                    descriptor.ssm_inner_size,
                    descriptor.ssm_group_count,
                    ssm_group,
                    descriptor.ssm_conv_kernel,
                    GdnOutputGate::Silu,
                    descriptor.v_head_reordered,
                    descriptor.prefill_width,
                )?;
                layer_roots.push(Qwen35LayerRoots::Ssm {
                    qkv_mixed: taps.qkv_mixed,
                    state_out: taps.state_out,
                });
                let ssm_out_result = taps.ssm_out_result;
                (mixer_out, Some(taps), None, ssm_out_result)
            }
        };

        let expert_w_gate = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(descriptor.expert_count),
                Extent::Static(embedding),
                Extent::Static(descriptor.expert_feed_forward),
            ],
            &alloc::format!("blk.{layer}.ffn_gate_exps.weight"),
        );
        let expert_w_up = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(descriptor.expert_count),
                Extent::Static(embedding),
                Extent::Static(descriptor.expert_feed_forward),
            ],
            &alloc::format!("blk.{layer}.ffn_up_exps.weight"),
        );
        let expert_w_down = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(descriptor.expert_count),
                Extent::Static(descriptor.expert_feed_forward),
                Extent::Static(embedding),
            ],
            &alloc::format!("blk.{layer}.ffn_down_exps.weight"),
        );
        let gate_inp_shexp = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.ffn_gate_inp_shexp.weight"),
        );
        let gate_shexp = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(embedding),
                Extent::Static(descriptor.expert_shared_feed_forward),
            ],
            &alloc::format!("blk.{layer}.ffn_gate_shexp.weight"),
        );
        let up_shexp = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(embedding),
                Extent::Static(descriptor.expert_shared_feed_forward),
            ],
            &alloc::format!("blk.{layer}.ffn_up_shexp.weight"),
        );
        let down_shexp = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(descriptor.expert_shared_feed_forward),
                Extent::Static(embedding),
            ],
            &alloc::format!("blk.{layer}.ffn_down_shexp.weight"),
        );

        let (
            residual,
            moe_site,
            post_attention_norm_output,
            router_logits,
            routed_output,
            shared_output,
        ) = append_qwen35moe_ffn(
            &mut program,
            layer,
            mixer_out,
            post_attention_norm_weight,
            inv_dim,
            eps,
            one,
            gate_inp,
            expert_w_gate,
            expert_w_up,
            expert_w_down,
            descriptor.expert_count,
            descriptor.expert_used_count,
            gate_inp_shexp,
            gate_shexp,
            up_shexp,
            down_shexp,
        )?;
        layer_diagnostics.push(Qwen35MoeLayerDiagnostics {
            block_input,
            ssm_taps,
            dense_attention_taps,
            mixer_output: mixer_output_pre_residual,
            post_mixer_residual: mixer_out,
            post_attention_norm_output,
            router_logits,
            routed_output,
            shared_output,
            block_output: residual,
        });
        x = residual;
        moe_sites.push(moe_site);
    }

    let output_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding)],
        "output_norm.weight",
    );
    let normed_final = rmsnorm(&mut program, x, output_norm_weight, inv_dim, eps)?;

    let normed_last = gather_last_row(&mut program, normed_final, descriptor.last_row_only);

    let lm_head = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(embedding),
            Extent::Static(descriptor.vocab),
        ],
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

    Ok(ForwardProgram {
        program,
        logits,
        layer_roots,
        moe_sites: MoeSites(moe_sites),
        layer_residuals: Vec::new(),
        hidden: Some(normed_final),
        duplicate_head_roots: Vec::new(),
        layer_diagnostics,
    })
}
