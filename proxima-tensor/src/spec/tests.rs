use super::*;

#[test]
fn grouped_gathered_expert_product_infers_selected_axis() {
    let mut program = Vec::new();
    let stack = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(3), Extent::Static(4), Extent::Static(2)],
        "stack",
    );
    let route = input_leaf(
        &mut program,
        DType::Int32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(2)],
        "route",
    );
    let activation = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(4)],
        "activation",
    );
    let product = grouped_gathered_expert_product(&mut program, stack, route, activation);
    let shapes = crate::shape::infer(&program, &[1]).expect("grouped gather infers");
    assert_eq!(shapes.of(product), &[1, 2, 4, 2]);

    let stack_values: Vec<f32> = (0..24).map(|value| value as f32).collect();
    let route_values = [2.0_f32, 0.0];
    let activation_values = [1.0_f32, 2.0, 3.0, 4.0];
    let evaluated = crate::cpu::evaluate_quantized(
        &program,
        &[1],
        &[
            crate::cpu::QuantizedBlock::Float32(&stack_values),
            crate::cpu::QuantizedBlock::Float32(&route_values),
            crate::cpu::QuantizedBlock::Float32(&activation_values),
        ],
        &[product],
    )
    .expect("grouped gather evaluates");
    let output = evaluated.root();
    assert_eq!(output.len(), 16);
    for (selected, expert) in [2_usize, 0].into_iter().enumerate() {
        for input in 0..4 {
            for output_index in 0..2 {
                let stack_index = (expert * 8) + input * 2 + output_index;
                let expected = stack_values[stack_index] * activation_values[input];
                let found = output[(selected * 4 + input) * 2 + output_index];
                assert_eq!(found, expected);
            }
        }
    }
}

#[test]
fn selected_scalar_routes_stack_in_token_then_selected_order() {
    let mut program = Vec::new();
    let first = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(2)],
        "first",
    );
    let second = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(2)],
        "second",
    );
    let stacked = stack_selected_routes(&mut program, &[first, second])
        .expect("two selected routes stack");

    let shapes = crate::shape::infer(&program, &[]).expect("route stack infers");
    assert_eq!(shapes.of(stacked), &[2, 2]);
    let first_values = [2.0f32, 0.0];
    let second_values = [0.0f32, 1.0];
    let evaluated = crate::cpu::evaluate_quantized(
        &program,
        &[],
        &[
            crate::cpu::QuantizedBlock::Float32(&first_values),
            crate::cpu::QuantizedBlock::Float32(&second_values),
        ],
        &[stacked],
    )
    .expect("route stack evaluates");
    assert_eq!(evaluated.root(), &[2.0, 0.0, 0.0, 1.0]);
}

#[test]
fn grouped_gate_up_matches_the_per_route_moe_graph_on_cpu() {
    const EXPERT_COUNT: u32 = 3;
    const EXPERT_USED_COUNT: u32 = 2;
    const EMBEDDING: u32 = 2;
    const FEED_FORWARD: u32 = 2;

    let build = |grouped: bool| {
        let mut program = Vec::new();
        let x = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(1), Extent::Static(EMBEDDING)],
            "x",
        );
        let gate_inp = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(EMBEDDING), Extent::Static(EXPERT_COUNT)],
            "gate_inp",
        );
        let gate = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(EXPERT_COUNT),
                Extent::Static(EMBEDDING),
                Extent::Static(FEED_FORWARD),
            ],
            "gate",
        );
        let up = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(EXPERT_COUNT),
                Extent::Static(EMBEDDING),
                Extent::Static(FEED_FORWARD),
            ],
            "up",
        );
        let down = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(EXPERT_COUNT),
                Extent::Static(FEED_FORWARD),
                Extent::Static(EMBEDDING),
            ],
            "down",
        );
        let one = scalar_constant(&mut program, 1.0);
        let appended = if grouped {
            append_moe_ffn_grouped_gate_up
        } else {
            append_moe_ffn
        };
        let (root, _) = appended(
            &mut program,
            0,
            x,
            gate_inp,
            gate,
            up,
            down,
            EXPERT_COUNT,
            EXPERT_USED_COUNT,
            one,
            ExpertGatingFunc::Softmax,
            None,
        )
        .expect("the MoE graph builds");
        (program, root)
    };

    let x = [3.0f32, 2.0];
    let gate_inp = [1.0f32, 0.0, 0.0, 0.0, 1.0, 2.0];
    let gate = [
        1.0f32, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0,
    ];
    let up = [
        1.0f32, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0, 0.0, 0.0, 2.0,
    ];
    let down = [
        1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0,
    ];
    let blocks: [&[f32]; 5] = [&x, &gate_inp, &gate, &up, &down];
    let workers = core::num::NonZeroUsize::new(1).expect("one worker exists");

    let (per_route_program, per_route_root) = build(false);
    let per_route = crate::cpu::evaluate_parallel(
        &per_route_program,
        &[],
        &blocks,
        &[per_route_root],
        workers,
    )
    .expect("per-route graph evaluates");
    let (grouped_program, grouped_root) = build(true);
    let grouped =
        crate::cpu::evaluate_parallel(&grouped_program, &[], &blocks, &[grouped_root], workers)
            .expect("grouped gate/up graph evaluates");

    assert_eq!(grouped.root(), per_route.root());
    let gathered_count = |program: &[Op]| {
        program
            .iter()
            .filter(|operation| {
                matches!(
                    operation,
                    Op::Elementwise { operands, .. }
                        if operands.iter().any(|(_, index_map)| {
                            matches!(index_map, IndexMap::Computed { .. })
                        })
                )
            })
            .count()
    };
    assert_eq!(gathered_count(&per_route_program), 6);
    assert_eq!(gathered_count(&grouped_program), 4);
}

#[test]
fn gather_head_permutation_selects_head_rows_without_reordering_tokens_or_features() {
    let sequence = 2u32;
    let source_heads = 3u32;
    let output_heads = 3u32;
    let head_dim = 2u32;
    let mut program = Vec::new();
    let source = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(sequence),
            Extent::Static(source_heads),
            Extent::Static(head_dim),
        ],
        "source",
    );
    let indices = input_leaf(
        &mut program,
        DType::Int32,
        alloc::vec![Extent::Static(output_heads)],
        "head_permutation",
    );
    let gathered = gather_head_permutation(&mut program, source, indices, head_dim);
    let source_data = [
        10.0f32, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0, 50.0, 51.0, 60.0, 61.0,
    ];
    let permutation = [2i32, 0, 1];
    let evaluated = crate::cpu::evaluate_typed(
        &program,
        &[],
        &[
            crate::cpu::TypedBuffer::Float32(source_data.to_vec()),
            crate::cpu::TypedBuffer::Int32(permutation.to_vec()),
        ],
        &[gathered],
    )
    .expect("head permutation gather evaluates");
    let crate::cpu::TypedBuffer::Float32(output) = &evaluated[0].2 else {
        panic!("head permutation must preserve the source dtype");
    };
    assert_eq!(
        output,
        &[
            30.0, 31.0, 10.0, 11.0, 20.0, 21.0, 60.0, 61.0, 40.0, 41.0, 50.0, 51.0,
        ],
        "permutation must select only the head axis"
    );
}

/// Proves the per-layer builders this module exports as `pub` are
/// actually SUFFICIENT to build a forward program from outside this
/// crate -- composes `embedding_lookup` -> [`append_hyper_connection_mix`]
/// -> [`append_qwen35_ssm_mixer`] (`GdnOutputGate::Sigmoid`, the
/// qwen4exp gate) -> [`append_hyper_connection_combine`] -> a final
/// output mixer (`w_inject: None`, mirroring the doc's own
/// "the final hyper-connection mixer carries [output_norm]") -> the
/// same `rmsnorm` + multiply + reduce lm-head chain
/// [`qwen35_forward_program`] ends every program with. Every dimension
/// is the smallest non-degenerate size that keeps every builder's own
/// einsum maps distinct (`embedding = 1` matches
/// `build_ssm_mixer_test_program`'s own convention below); the
/// assertion is finiteness and shape, not a numeric reference, since
/// this test's job is proving the public surface COMPOSES, not
/// re-proving any one builder's own arithmetic (each builder already
/// has its own f64-reference test for that).
#[test]
fn public_builders_compose_a_one_layer_forward_program() {
    let tokens = 2usize;
    let vocab = 3u32;
    let embedding = 1u32;
    let hc = 2u32;
    let low_rank = 1u32;

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
    let embedded = embedding_lookup(&mut program, table, ids);

    let inv_dim = scalar_constant(&mut program, 1.0 / embedding as f32);
    // `eps` carries its own `s` (token) axis throughout this module's
    // rmsnorm-family ops (`rmsnorm`'s own `(eps, "s->s")`,
    // `append_hyper_connection_mix`'s `(eps, "s->sh")`) -- a per-token
    // leaf, never a rank-0 constant, matching every other builder's own
    // test fixture (`assert_mix_matches_reference`'s own `eps_data`).
    let eps = symbolic_leaf(&mut program, DType::Float32, "eps");
    let inv_hc = scalar_constant(&mut program, 1.0 / hc as f32);
    let one = scalar_constant(&mut program, 1.0);
    let two = scalar_constant(&mut program, 2.0);

    // Broadcast the `[tokens, embedding]` embedding lookup into the
    // `[tokens, hc, embedding]` hyper-connection residual every stream
    // starts identical at layer 0. Neither `embedded` (`si`, no `h`)
    // nor a rank-0 scalar owns the `h` axis, so shape inference cannot
    // size it from either alone -- `hc_ones`, a real `[hc]`-shaped
    // donor, is the same all-ones-donor idiom `group_ones`/`key_head_ones`
    // already use to constrain a read-side axis no other operand names.
    let hc_ones = op::append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(hc)],
            value: 1.0,
        },
    );
    let residual = elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(embedded, "si->shi"), (hc_ones, "h->shi")],
    )
    .expect("broadcast into hc streams lowers");

    let w_norm = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(hc), Extent::Static(embedding)],
        "w_norm",
    );
    let w_down = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(hc),
            Extent::Static(embedding),
            Extent::Static(low_rank)
        ],
        "w_down",
    );
    let w_up = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(low_rank),
            Extent::Static(hc),
            Extent::Static(embedding)
        ],
        "w_up",
    );
    let w_inject = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(hc),
            Extent::Static(embedding),
            Extent::Static(hc)
        ],
        "w_inject",
    );

    let (mixed, inject) = append_hyper_connection_mix(
        &mut program,
        residual,
        inv_dim,
        eps,
        inv_hc,
        one,
        w_norm,
        w_down,
        w_up,
        Some(w_inject),
    )
    .expect("hyper-connection mix lowers");
    let inject = inject.expect("w_inject was Some, so inject must be Some");

    let key_dim = 1u32;
    let value_dim = 2u32;
    let kv_heads = 1u32;
    let group = 2u32;
    let l_cache = 2u32;
    let qkv_dim = 2 * key_dim + value_dim;

    let head_eps = op::append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
            value: 1e-6,
        },
    );
    let inv_sqrt_key_dim = scalar_constant(&mut program, 1.0);
    let inv_head_v_dim = scalar_constant(&mut program, 1.0);
    let attn_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding)],
        "attn_norm_weight",
    );
    let wqkv = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding), Extent::Static(qkv_dim)],
        "wqkv",
    );
    let wqkv_gate = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding), Extent::Static(value_dim)],
        "wqkv_gate",
    );
    let conv_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(qkv_dim), Extent::Static(l_cache)],
        "conv_weight",
    );
    let conv_history_in = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(l_cache - 1), Extent::Static(qkv_dim)],
        "conv_history_in",
    );
    let ssm_beta = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding), Extent::Static(kv_heads * group)],
        "ssm_beta",
    );
    let ssm_alpha = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding), Extent::Static(kv_heads * group)],
        "ssm_alpha",
    );
    let ssm_dt_bias = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(kv_heads * group)],
        "ssm_dt_bias",
    );
    let ssm_a = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(kv_heads * group)],
        "ssm_a",
    );
    let ssm_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding)],
        "ssm_norm_weight",
    );
    let ssm_out = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(value_dim), Extent::Static(embedding)],
        "ssm_out",
    );
    let state_in = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(1),
            Extent::Static(1),
            Extent::Static(kv_heads),
            Extent::Static(group)
        ],
        "state_in",
    );

    let (block_out, _qkv_mixed, _state_out) = append_qwen35_ssm_mixer(
        &mut program,
        mixed,
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
        key_dim,
        value_dim,
        kv_heads,
        group,
        l_cache,
        GdnOutputGate::Sigmoid,
    )
    .expect("qwen4exp's own output gate lowers through the shared ssm mixer builder");

    let residual = append_hyper_connection_combine(
        &mut program,
        residual,
        block_out,
        inject,
        inv_hc,
        one,
        two,
    )
    .expect("hyper-connection combine lowers");

    let final_w_norm = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(hc), Extent::Static(embedding)],
        "final_w_norm",
    );
    let final_w_down = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(hc),
            Extent::Static(embedding),
            Extent::Static(low_rank)
        ],
        "final_w_down",
    );
    let final_w_up = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(low_rank),
            Extent::Static(hc),
            Extent::Static(embedding)
        ],
        "final_w_up",
    );
    let (final_mixed, no_inject) = append_hyper_connection_mix(
        &mut program,
        residual,
        inv_dim,
        eps,
        inv_hc,
        one,
        final_w_norm,
        final_w_down,
        final_w_up,
        None,
    )
    .expect("final output mixer lowers");
    assert!(
        no_inject.is_none(),
        "the final output mixer must pass w_inject: None"
    );

    // The same rmsnorm + multiply + reduce chain `qwen35_forward_program`
    // ends every program with.
    let output_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding)],
        "output_norm.weight",
    );
    let normed_final = rmsnorm(&mut program, final_mixed, output_norm_weight, inv_dim, eps)
        .expect("final rmsnorm lowers");
    let lm_head_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding), Extent::Static(vocab)],
        "output.weight",
    );
    let logits_product = elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed_final, "sd->sdv"), (lm_head_weight, "dv->sdv")],
    )
    .expect("lm_head product lowers");
    let logits = reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        logits_product,
        "sdv->sdv",
        "sv->sdv",
    )
    .expect("lm_head reduce lowers");

    let mut state = 0x1234_5678_9abc_def0u64;
    let mut filled_input = |len: usize| -> Vec<f32> {
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
            })
            .collect()
    };

    // `evaluate_named` binds every input as an `&[f32]` regardless of
    // the `Op::Input::dtype` it was declared with -- `ids`'s own
    // `DType::Int32` only documents intent, since the gather this
    // module's `embedding_lookup` builds reads its index operand
    // through the same f32 buffer as every other input (the real
    // forward-program tests above bind their own `ids` this same way,
    // as `ids_f32`).
    let ids_data: Vec<f32> = (0..tokens as i32)
        .map(|token| (token % vocab as i32) as f32)
        .collect();
    let eps_data = alloc::vec![1e-6f32; tokens];
    let table_data = filled_input(vocab as usize * embedding as usize);
    let w_norm_data = filled_input(hc as usize * embedding as usize);
    let w_down_data = filled_input(hc as usize * embedding as usize * low_rank as usize);
    let w_up_data = filled_input(low_rank as usize * hc as usize * embedding as usize);
    let w_inject_data = filled_input(hc as usize * embedding as usize * hc as usize);
    let attn_norm_data = filled_input(embedding as usize);
    let wqkv_data = filled_input(embedding as usize * qkv_dim as usize);
    let wqkv_gate_data = filled_input(embedding as usize * value_dim as usize);
    let conv_weight_data = filled_input(qkv_dim as usize * l_cache as usize);
    let conv_history_data = filled_input((l_cache as usize - 1) * qkv_dim as usize);
    let ssm_beta_data = filled_input(embedding as usize * (kv_heads * group) as usize);
    let ssm_alpha_data = filled_input(embedding as usize * (kv_heads * group) as usize);
    let ssm_dt_bias_data = filled_input((kv_heads * group) as usize);
    let ssm_a_data = filled_input((kv_heads * group) as usize);
    let ssm_norm_data = filled_input(embedding as usize);
    let ssm_out_data = filled_input(value_dim as usize * embedding as usize);
    let state_in_data = filled_input(kv_heads as usize * group as usize);
    let final_w_norm_data = filled_input(hc as usize * embedding as usize);
    let final_w_down_data = filled_input(hc as usize * embedding as usize * low_rank as usize);
    let final_w_up_data = filled_input(low_rank as usize * hc as usize * embedding as usize);
    let output_norm_data = filled_input(embedding as usize);
    let lm_head_data = filled_input(embedding as usize * vocab as usize);

    let named: Vec<(&str, &[f32])> = alloc::vec![
        ("ids", ids_data.as_slice()),
        ("eps", eps_data.as_slice()),
        ("token_embd.weight", table_data.as_slice()),
        ("w_norm", w_norm_data.as_slice()),
        ("w_down", w_down_data.as_slice()),
        ("w_up", w_up_data.as_slice()),
        ("w_inject", w_inject_data.as_slice()),
        ("attn_norm_weight", attn_norm_data.as_slice()),
        ("wqkv", wqkv_data.as_slice()),
        ("wqkv_gate", wqkv_gate_data.as_slice()),
        ("conv_weight", conv_weight_data.as_slice()),
        ("conv_history_in", conv_history_data.as_slice()),
        ("ssm_beta", ssm_beta_data.as_slice()),
        ("ssm_alpha", ssm_alpha_data.as_slice()),
        ("ssm_dt_bias", ssm_dt_bias_data.as_slice()),
        ("ssm_a", ssm_a_data.as_slice()),
        ("ssm_norm_weight", ssm_norm_data.as_slice()),
        ("ssm_out", ssm_out_data.as_slice()),
        ("state_in", state_in_data.as_slice()),
        ("final_w_norm", final_w_norm_data.as_slice()),
        ("final_w_down", final_w_down_data.as_slice()),
        ("final_w_up", final_w_up_data.as_slice()),
        ("output_norm.weight", output_norm_data.as_slice()),
        ("output.weight", lm_head_data.as_slice()),
    ];

    let evaluated = crate::cpu::evaluate_named(&program, &[tokens as u64], &named, &[logits])
        .expect("the composed program evaluates on cpu");
    let (logits_values, logits_shape) = evaluated.get(logits).expect("logits output present");

    assert_eq!(
        logits_shape,
        [tokens as u64, vocab as u64],
        "logits must be [tokens, vocab] -- the same shape qwen35_forward_program's own lm_head produces"
    );
    assert!(
        logits_values.iter().all(|value| value.is_finite()),
        "every logit must be finite: {logits_values:?}"
    );
}

/// Regression proof for the qk-norm-dropped-on-MoE-layers bug fixed
/// alongside this test: before the fix, `append_mistral_cached_moe_layer`
/// took no `qk_norm` parameter at all, so `mistral_cached_forward_program_with_experts`
/// silently discarded the `qk_norm` argument for every routed (MoE)
/// layer -- flipping it produced the byte-identical program. A
/// Qwen3-MoE-shaped checkpoint (`expert_count > 0`) carries
/// `attn_q_norm.weight`/`attn_k_norm.weight` on every layer, so
/// `qk_norm=true` must now append the extra per-head rmsnorm reduces
/// (and swap RoPE pairing) the dense `qk_norm` path already gets --
/// this asserts the MoE program's own length actually changes with the
/// flag, the exact invariant the bug violated.
#[test]
fn mistral_cached_forward_program_with_experts_qk_norm_changes_the_moe_program() {
    let (qk_norm_off, _, _, _) = mistral_cached_forward_program_with_experts(
        32_000, 256, 128, 4, 2, 64, 1, 4, 1, false, false, false, false,
    )
    .expect("moe program without qk_norm lowers");
    let (qk_norm_on, _, _, _) = mistral_cached_forward_program_with_experts(
        32_000, 256, 128, 4, 2, 64, 1, 4, 1, true, false, false, false,
    )
    .expect("moe program with qk_norm lowers");

    assert_ne!(
        qk_norm_off.len(),
        qk_norm_on.len(),
        "a Qwen3-MoE-shaped forward program (expert_count > 0) must grow when qk_norm \
         flips on -- an unchanged length means the MoE layer builder is still dropping \
         attn_q_norm/attn_k_norm on the floor"
    );
}

/// Qwen2 is NEOX split-half RoPE even without QK-norm tensors. Keep this
/// architecture choice explicit so a future generic-path refactor cannot
/// regress to using QK-norm presence as a pairing proxy.
#[test]
fn qwen2_cached_program_uses_split_half_rope_without_qk_norm() {
    let (qwen2_program, _, _, _, _) = qwen2_cached_forward_program_with_experts_and_layer_taps(
        32_000, 256, 128, 4, 2, 64, 1, 0, 0, false, false, false, false,
    )
    .expect("qwen2 program lowers");
    let (generic_program, _, _, _, _) =
        mistral_cached_forward_program_with_experts_and_layer_taps(
            32_000, 256, 128, 4, 2, 64, 1, 0, 0, false, false, false, false, false,
        )
        .expect("generic program lowers");
    assert!(
        qwen2_program != generic_program,
        "Qwen2's explicit split-half graph must differ from the generic interleaved graph"
    );
}

/// [`mistral_cached_forward_program_with_experts_and_layer_taps`] must
/// build the byte-identical program to its thin-wrapper sibling
/// (`mistral_cached_forward_program_with_experts`'s own doc on that
/// relationship) and return exactly one residual tap per layer, in
/// layer order -- the invariant a caller bisecting CPU-vs-Metal
/// divergence across a 48-layer checkpoint depends on to index
/// `layer_residuals[layer]` directly.
#[test]
fn layer_taps_variant_matches_the_plain_program_and_returns_one_tap_per_layer() {
    let (plain_program, plain_roots, plain_cache_roots, _plain_moe_sites) =
        mistral_cached_forward_program_with_experts(
            32_000, 256, 128, 4, 2, 64, 3, 4, 1, true, false, false, false,
        )
        .expect("plain moe program lowers");
    let (taps_program, taps_roots, taps_cache_roots, layer_residuals, _taps_moe_sites) =
        mistral_cached_forward_program_with_experts_and_layer_taps(
            32_000, 256, 128, 4, 2, 64, 3, 4, 1, true, false, false, false, false,
        )
        .expect("taps moe program lowers");

    assert_eq!(
        plain_program, taps_program,
        "the taps variant must build the identical graph -- it only returns extra \
         NodeIds into the same program, never a structurally different one"
    );
    assert_eq!(plain_roots, taps_roots);
    assert_eq!(plain_cache_roots, taps_cache_roots);
    assert_eq!(
        layer_residuals.len(),
        3,
        "one residual NodeId per layer (block_count=3)"
    );
    assert!(
        layer_residuals.windows(2).all(|pair| pair[0].0 < pair[1].0),
        "residual taps must appear in strictly increasing program order across layers: \
         {layer_residuals:?}"
    );
}

/// [`ForwardRoots::hidden`] must be the pre-`lm_head` activation the
/// vocab-projection multiply actually reads -- proved by walking the
/// graph FROM `logits` backward (`logits`'s own `Reduce::operand` is
/// the `logits_product` elementwise; that node's own operand set must
/// contain `hidden`) rather than assuming any fixed distance between
/// the two `NodeId`s. This is the structural guarantee
/// `LoadedModel::embed` (`proxima-model-interop`) depends on: if a
/// future refactor of this builder ever produced a `hidden` that is
/// NOT actually upstream of `lm_head`'s multiply, this test fails
/// before any real-checkpoint embedding test would even hint at it.
#[test]
fn forward_roots_hidden_is_an_operand_of_the_lm_head_product() {
    let (program, roots, _cache_roots, _moe_sites) =
        mistral_cached_forward_program_with_experts(
            32_002, 4096, 14336, 32, 8, 128, 2, 0, 0, false, false, false, false,
        )
        .expect("the dense cached forward pass lowers to a program");

    let Op::Reduce(logits_reduce) = &program[roots.logits.0 as usize] else {
        panic!("ForwardRoots::logits must name an Op::Reduce (the vocab-projection sum)");
    };
    let logits_product = logits_reduce.operand;

    let Op::Elementwise {
        operands: product_operands,
        ..
    } = &program[logits_product.0 as usize]
    else {
        panic!("logits's own Reduce::operand must name an Op::Elementwise (the multiply)");
    };

    assert!(
        product_operands
            .iter()
            .any(|(operand, _map)| *operand == roots.hidden),
        "ForwardRoots::hidden ({:?}) must be one of the lm_head product's own operands \
         ({:?}), or LoadedModel::embed would pool the wrong tensor",
        roots.hidden,
        product_operands
            .iter()
            .map(|(operand, _)| *operand)
            .collect::<alloc::vec::Vec<_>>(),
    );
}

const MATMUL_TOML: &str = r#"
[[node]]
op = "input"
id = "lhs"
dtype = "float32"
shape = ["?0", 768]

[[node]]
op = "input"
id = "rhs"
dtype = "float32"
shape = [768, 3072]

[[node]]
op = "elementwise"
id = "product"
dtype = "float32"
body = "multiply"
inputs = ["lhs", "rhs"]
maps = ["ik->ijk", "kj->ijk"]

[[node]]
op = "reduce"
id = "sum"
dtype = "float32"
body = "add"
init = "zero"
input = "product"
in_map = "ijk->ijk"
out_map = "ij->ijk"
keep = "reduce"
name = "matmul"
"#;

fn matmul_in_rust() -> Vec<Op> {
    let mut program = Vec::new();
    let lhs = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Symbolic(0), Extent::Static(768)],
            name: None,
        },
    );
    let rhs = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(768), Extent::Static(3072)],
            name: None,
        },
    );
    let product = op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![
                (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
            ],
            name: None,
        },
    );
    op::append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: Some("matmul".into()),
        }),
    );
    program
}

/// The whole reason this module exists: if these two disagree, the claim
/// that the algebra is describable as data is false.
#[test]
fn a_program_written_as_toml_equals_the_same_program_written_in_rust() {
    let spec: ProgramSpec = toml::from_str(MATMUL_TOML).expect("spec parses");
    spec.validate().expect("spec is structurally sound");
    let from_config = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");
    assert_eq!(
        from_config,
        matmul_in_rust(),
        "config and code must produce the same program"
    );
    crate::shape::infer(&from_config, &[512]).expect("the parsed program also infers");
}

const EMBEDDING_TOML: &str = r#"
[[node]]
op = "input"
id = "table"
dtype = "float32"
shape = [50000, 8]

[[node]]
op = "input"
id = "ids"
dtype = "int32"
shape = [4]

[[node]]
op = "elementwise"
id = "gathered"
dtype = "float32"
body = "identity"
inputs = ["table"]
maps = [{ gather = "ids", index_map = "s->sd", map = "d->sd", dim = 0 }]
"#;

fn embedding_lookup_in_rust() -> Vec<Op> {
    let mut program = Vec::new();
    let table = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(50_000), Extent::Static(8)],
            name: None,
        },
    );
    let ids = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Int32,
            shape: alloc::vec![Extent::Static(4)],
            name: None,
        },
    );
    let gathered_map = IndexMap::Computed {
        indices: ids,
        index_map: map::projection(2, &[0]),
        base: IndexPattern {
            iter_rank: 2,
            axes: alloc::vec![
                AxisIndex::default(),
                AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(1)).collect(),
                    offset: 0,
                    len: None,
                },
            ],
        },
        gathered_dim: 0,
    };
    op::append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: alloc::vec![(table, gathered_map)],
            name: None,
        },
    );
    program
}

/// The gather analogue of
/// [`a_program_written_as_toml_equals_the_same_program_written_in_rust`]:
/// an embedding lookup written as TOML must equal the same program built
/// directly, and the parsed program must still pass shape inference.
#[test]
fn an_embedding_lookup_written_as_toml_equals_the_same_program_written_in_rust() {
    let spec: ProgramSpec = toml::from_str(EMBEDDING_TOML).expect("spec parses");
    spec.validate().expect("spec is structurally sound");
    let from_config = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");
    assert_eq!(
        from_config,
        embedding_lookup_in_rust(),
        "config and code must produce the same gather program"
    );
    crate::shape::infer(&from_config, &[]).expect("the parsed gather program also infers");
}

#[test]
fn the_name_survives_the_config_round_trip() {
    let spec: ProgramSpec = toml::from_str(MATMUL_TOML).expect("spec parses");
    let program = Vec::<Op>::try_from(&spec).expect("lowers");
    let root = program.last().expect("root");
    assert_eq!(root.name(), Some("matmul"));
}

#[test]
fn a_symbolic_extent_survives_as_a_symbol() {
    let spec: ProgramSpec = toml::from_str(MATMUL_TOML).expect("spec parses");
    let program = Vec::<Op>::try_from(&spec).expect("lowers");
    let Op::Input { shape, .. } = &program[0] else {
        panic!("first node is a leaf");
    };
    assert_eq!(
        shape[0],
        Extent::Symbolic(0),
        "sequence length stays unresolved"
    );
}

#[test]
fn an_input_name_survives_the_config_round_trip() {
    let named = r#"
[[node]]
op = "input"
id = "x"
dtype = "float32"
shape = [4]
name = "weights.embedding"
"#;
    let spec: ProgramSpec = toml::from_str(named).expect("spec parses");
    let program = Vec::<Op>::try_from(&spec).expect("lowers");
    assert_eq!(program[0].name(), Some("weights.embedding"));
}

#[proxima::test]
#[case::identity("ij->ij", 2, &[0, 1])]
#[case::transpose("ji->ij", 2, &[1, 0])]
#[case::broadcast("j->ij", 2, &[1])]
#[case::contraction_lhs("ik->ijk", 3, &[0, 2])]
#[case::full_reduction("->i", 1, &[])]
async fn projection_notation_reads_like_einsum(
    #[case] notation: &str,
    #[case] rank: u16,
    #[case] projected: &[u16],
) {
    let (found_rank, found) = parse_projection(notation).expect("well-formed");
    assert_eq!(found_rank, rank);
    assert_eq!(found, projected);
}

#[test]
fn a_map_without_an_arrow_is_rejected() {
    assert!(matches!(
        parse_projection("ijk").expect_err("no arrow"),
        TensorError::MalformedMap(_)
    ));
}

#[test]
fn projecting_a_letter_the_iteration_space_lacks_is_rejected() {
    let error = parse_projection("iz->ijk").expect_err("z is not in ijk");
    assert!(
        matches!(error, TensorError::UnknownIndexLetter { letter: 'z', .. }),
        "{error}"
    );
}

/// A shifted, scaled, or multi-term address still honors a trailing
/// `@length` -- the same declared-`len` fact `shape::unify_iteration_space`
/// resolves regardless of how the address term is spelled
/// (`AxisIndex::len_target_axis`'s own doc).
#[proxima::test]
#[case::shifted("s,i+1@2->si")]
#[case::scaled("s,2*i@4->si")]
#[case::multi_term_unit_coefficient("s,2*i+p@2->sip")]
async fn a_len_declaration_parses_regardless_of_address_shape(#[case] notation: &str) {
    let pattern = parse_operand_pattern(notation).expect("a declared len parses");
    let len_bearing = pattern
        .axes
        .iter()
        .find(|axis| axis.len.is_some())
        .expect("one axis in the notation declares len");
    assert!(len_bearing.len_target_axis().is_some());
}

/// `i+j@2` has two equally-plain (`coeff == 1`) terms -- `len` cannot
/// say which one it describes, so this is malformed at parse time
/// rather than accepted and silently ignored (or rejected far later, at
/// shape inference, over a program already built).
#[test]
fn a_len_on_two_unit_coefficient_terms_is_rejected_at_parse_time() {
    let error =
        parse_operand_pattern("s,i+j@2->sij").expect_err("len has no unambiguous target");
    assert!(matches!(error, TensorError::MalformedMap(_)), "{error}");
}

#[test]
fn a_forward_reference_in_config_is_rejected() {
    let forward = r#"
[[node]]
op = "elementwise"
id = "early"
dtype = "float32"
body = "identity"
inputs = ["later"]
maps = ["i->i"]

[[node]]
op = "input"
id = "later"
dtype = "float32"
shape = [4]
"#;
    let spec: ProgramSpec = toml::from_str(forward).expect("parses");
    assert!(
        spec.validate().is_err(),
        "config order mirrors the program's backwards-reference rule"
    );
    assert!(matches!(
        Vec::<Op>::try_from(&spec).expect_err("cannot lower"),
        TensorError::UnknownNode(_)
    ));
}

#[test]
fn a_duplicate_id_is_rejected() {
    let duplicate = r#"
[[node]]
op = "input"
id = "same"
dtype = "float32"
shape = [4]

[[node]]
op = "input"
id = "same"
dtype = "float32"
shape = [8]
"#;
    let spec: ProgramSpec = toml::from_str(duplicate).expect("parses");
    assert!(spec.validate().is_err(), "ids must be unique");
}

#[test]
fn inputs_and_maps_must_agree_in_count() {
    let lopsided = r#"
[[node]]
op = "input"
id = "source"
dtype = "float32"
shape = [4]

[[node]]
op = "elementwise"
id = "bad"
dtype = "float32"
body = "add"
inputs = ["source", "source"]
maps = ["i->i"]
"#;
    let spec: ProgramSpec = toml::from_str(lopsided).expect("parses");
    assert!(spec.validate().is_err());
    assert!(matches!(
        Vec::<Op>::try_from(&spec).expect_err("cannot lower"),
        TensorError::SpecArityMismatch { .. }
    ));
}

#[test]
fn a_malformed_extent_is_rejected() {
    let bad = r#"
[[node]]
op = "input"
id = "source"
dtype = "float32"
shape = ["seq"]
"#;
    let spec: ProgramSpec = toml::from_str(bad).expect("parses");
    assert!(matches!(
        Vec::<Op>::try_from(&spec).expect_err("`seq` is not `?n`"),
        TensorError::MalformedExtent(_)
    ));
}

use crate::test_support::Lcg;

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

/// The claim "a new architecture is a config file, not a PR" is only
/// worth anything if a real architecture fits. A single-head attention
/// block with RMSNorm and a full softmax does, and this checks it
/// evaluates rather than merely parses — a spec that lowers and then
/// produces garbage is still a PR waiting to happen.
///
/// The softmax rows are the assertion that matters: finite output only
/// proves the pipeline ran, whereas rows summing to one prove it computed
/// attention. What this file still leaves as a plain input rather than
/// deriving is a mask, which would need an index-derived tensor no `Op`
/// produces — RoPE's multi-term affine no longer belongs on that list;
/// see the pairwise-rotation test below.
///
/// Inputs are LCG-derived rather than uniform constants: a uniform row
/// makes every q/k/v row identical, so the scores collapse to a uniform
/// distribution regardless of whether the index maps, reduction order,
/// or broadcast are correct. Varied inputs make the softmax rows genuinely
/// non-uniform, so a transposed axis or a wrong reduction actually shows
/// up as a numeric difference instead of vanishing by symmetry.
#[test]
fn an_attention_block_written_as_toml_evaluates() {
    const SEQUENCE: usize = 4;
    const MODEL: usize = 8;

    let text = include_str!("../../specs/attention_block.toml");
    let spec: ProgramSpec = toml::from_str(text).expect("spec parses");
    spec.validate().expect("spec is structurally sound");
    let program = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");

    let symbols = [SEQUENCE as u64];
    crate::shape::infer(&program, &symbols).expect("the block infers");

    let activations = random_vec(1, SEQUENCE * MODEL);
    let inverse_dim = alloc::vec![1.0 / MODEL as f32; SEQUENCE];
    let wq = random_vec(2, MODEL * MODEL);
    let wk = random_vec(3, MODEL * MODEL);
    let wv = random_vec(4, MODEL * MODEL);
    let blocks: [&[f32]; 5] = [&activations, &inverse_dim, &wq, &wk, &wv];

    let probabilities = spec
        .node
        .iter()
        .position(|node| node.id() == "probabilities")
        .expect("the spec defines a probabilities node");
    let probabilities = NodeId(probabilities as u32);
    let root = NodeId(program.len() as u32 - 1);

    let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");
    let evaluated = crate::cpu::evaluate_parallel(
        &program,
        &symbols,
        &blocks,
        &[root, probabilities],
        workers,
    )
    .expect("the block evaluates");

    let output = evaluated.root();
    assert_eq!(
        output.len(),
        SEQUENCE * MODEL,
        "a vacuous output proves nothing"
    );
    assert!(
        output.iter().all(|value| value.is_finite()),
        "output must be finite"
    );

    let (rows, _) = evaluated
        .get(probabilities)
        .expect("probabilities were requested");
    assert_eq!(rows.len(), SEQUENCE * SEQUENCE);
    for row in rows.as_chunks::<SEQUENCE>().0 {
        let total: f32 = row.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "softmax row sums to {total}, not 1.0"
        );
        let max = row.iter().copied().fold(f32::MIN, f32::max);
        let min = row.iter().copied().fold(f32::MAX, f32::min);
        assert!(
            max - min > 1e-3,
            "softmax row {row:?} is uniform (max - min = {}); varied inputs should break score ties",
            max - min
        );
    }
}

/// RoPE's whole reason for existing in this module's doc: `2*i` and
/// `2*i+1` are the multi-term affine that used to have no string
/// spelling. This does not just check the spec parses — a parser that
/// silently addressed the wrong elements would still parse — it checks
/// the *evaluated* output obeys the one property that only holds if the
/// pairwise addressing is right: a rotation preserves each pair's norm.
/// `expected` is computed straight off the raw `x` buffer at the literal
/// indices `2*i` / `2*i+1`, independently of anything the graph did, so
/// an addressing bug (reading `i` instead of `2*i`, or the wrong operand
/// axis order) would read a different pair and very likely a different
/// norm — `x`'s eight values are pairwise distinct for exactly that
/// reason.
#[test]
fn a_rope_pairwise_rotation_written_as_toml_preserves_pair_norm() {
    const SEQUENCE: usize = 2;
    const MODEL: usize = 4;
    const PAIRS: usize = MODEL / 2;

    let text = include_str!("../../specs/rope.toml");
    let spec: ProgramSpec = toml::from_str(text).expect("spec parses");
    spec.validate().expect("spec is structurally sound");
    let program = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");

    let symbols: [u64; 0] = [];
    crate::shape::infer(&program, &symbols).expect("the rotation infers");

    let x: [f32; SEQUENCE * MODEL] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    // two exact Pythagorean triples (3-4-5 and 7-24-25 scaled by 1/25),
    // so cos^2 + sin^2 = 1 exactly and any drift in the assertion below
    // is evaluation error, not a badly chosen rotation.
    let cos: [f32; SEQUENCE * PAIRS] = [0.6, 0.28, 0.6, 0.28];
    let sin: [f32; SEQUENCE * PAIRS] = [0.8, 0.96, 0.8, 0.96];
    let blocks: [&[f32]; 3] = [&x, &cos, &sin];

    let node_id = |id: &str| {
        let position = spec
            .node
            .iter()
            .position(|node| node.id() == id)
            .unwrap_or_else(|| panic!("the spec defines a {id} node"));
        NodeId(position as u32)
    };
    let rotated_even_id = node_id("rotated_even");
    let root = NodeId(program.len() as u32 - 1);
    assert_eq!(root, node_id("rotated_odd"), "rotated_odd is the last node");

    let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");
    let evaluated = crate::cpu::evaluate_parallel(
        &program,
        &symbols,
        &blocks,
        &[root, rotated_even_id],
        workers,
    )
    .expect("the rotation evaluates");

    let rotated_odd = evaluated.root();
    let (rotated_even, _) = evaluated
        .get(rotated_even_id)
        .expect("rotated_even was requested");
    assert_eq!(rotated_even.len(), SEQUENCE * PAIRS);
    assert_eq!(rotated_odd.len(), SEQUENCE * PAIRS);

    for sequence in 0..SEQUENCE {
        for pair in 0..PAIRS {
            let raw_even = x[sequence * MODEL + 2 * pair];
            let raw_odd = x[sequence * MODEL + 2 * pair + 1];
            let expected_norm = raw_even * raw_even + raw_odd * raw_odd;

            let rotated_index = sequence * PAIRS + pair;
            let found_even = rotated_even[rotated_index];
            let found_odd = rotated_odd[rotated_index];
            let found_norm = found_even * found_even + found_odd * found_odd;

            assert!(
                (found_norm - expected_norm).abs() < 1e-3,
                "pair ({raw_even}, {raw_odd}) has norm {expected_norm} but the rotated \
                 pair ({found_even}, {found_odd}) has norm {found_norm}"
            );
        }
    }
}

/// The test that makes `Op::Iota` worth having: `causal_attention.toml`
/// is `attention_block.toml` plus a real causal mask built from two
/// `Iota` leaves, and the property that makes a mask *causal* rather
/// than decorative is checked directly on the evaluated softmax output —
/// not just that the spec parses or that the output is finite.
///
/// Two invariants, over every one of the `SEQUENCE * SEQUENCE` = 16
/// probability cells (`checked` asserts that count, so a loop bug can't
/// silently check zero of them):
/// - every strictly-upper-triangular cell (`key > query`, a key position
///   later than its query) is *exactly* `0.0` — not merely small,
///   because `exp(-infinity)` is exact zero in IEEE-754 and a mask that
///   only suppresses without zeroing is not a causal mask;
/// - every row still sums to `1.0`, the same softmax invariant
///   `an_attention_block_written_as_toml_evaluates` checks, proving the
///   mask did not just zero everything.
///
/// Inputs are LCG-derived, not uniform, for the same reason
/// `an_attention_block_written_as_toml_evaluates` gives: under uniform
/// input every unmasked score in a row is identical, so a mask that
/// masked the wrong cells (or none at all) could still coincidentally
/// leave the *sum* at 1.0 — varied scores make a wrong mask show up as a
/// nonzero cell instead of vanishing by symmetry.
#[test]
fn a_causal_attention_block_written_as_toml_masks_future_positions() {
    const SEQUENCE: usize = 4;
    const MODEL: usize = 8;

    let text = include_str!("../../specs/causal_attention.toml");
    let spec: ProgramSpec = toml::from_str(text).expect("spec parses");
    spec.validate().expect("spec is structurally sound");
    let program = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");

    let symbols = [SEQUENCE as u64];
    crate::shape::infer(&program, &symbols).expect("the causal block infers");

    let activations = random_vec(11, SEQUENCE * MODEL);
    let inverse_dim = alloc::vec![1.0 / MODEL as f32; SEQUENCE];
    let wq = random_vec(12, MODEL * MODEL);
    let wk = random_vec(13, MODEL * MODEL);
    let wv = random_vec(14, MODEL * MODEL);
    let blocks: [&[f32]; 5] = [&activations, &inverse_dim, &wq, &wk, &wv];

    let probabilities = spec
        .node
        .iter()
        .position(|node| node.id() == "probabilities")
        .expect("the spec defines a probabilities node");
    let probabilities = NodeId(probabilities as u32);
    let root = NodeId(program.len() as u32 - 1);

    let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");
    let evaluated = crate::cpu::evaluate_parallel(
        &program,
        &symbols,
        &blocks,
        &[root, probabilities],
        workers,
    )
    .expect("the causal block evaluates");

    let output = evaluated.root();
    assert_eq!(
        output.len(),
        SEQUENCE * MODEL,
        "a vacuous output proves nothing"
    );
    assert!(
        output.iter().all(|value| value.is_finite()),
        "output must be finite"
    );

    let (rows, _) = evaluated
        .get(probabilities)
        .expect("probabilities were requested");
    assert_eq!(rows.len(), SEQUENCE * SEQUENCE);

    let mut checked = 0usize;
    for (query, row) in rows.as_chunks::<SEQUENCE>().0.iter().enumerate() {
        let total: f32 = row.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "softmax row {query} sums to {total}, not 1.0"
        );
        for (key, &probability) in row.iter().enumerate() {
            if key > query {
                assert_eq!(
                    probability, 0.0,
                    "row {query} col {key} is strictly upper-triangular (key {key} > \
                     query {query}) and must be masked to exactly 0.0, found {probability}"
                );
            }
            checked += 1;
        }
    }
    assert_eq!(
        checked,
        SEQUENCE * SEQUENCE,
        "every probability cell must be checked, not a subset"
    );
}

/// A full llama-style block — attention plus its output projection and
/// residual, a second RMSNorm, and a SwiGLU feed-forward with its own
/// residual — built on top of the attention block above. Every addition
/// lowers with the same node kinds and closed `ScalarOp` set the
/// attention block already used; `transformer_block.toml`'s header
/// records why nothing new was needed.
///
/// Two invariants, not just finiteness:
/// - the softmax rows inside it still sum to one, the same evidence
///   `an_attention_block_written_as_toml_evaluates` uses;
/// - a degenerate control: zero every projection weight (Q/K/V, the
///   output projection, and all three FFN matrices) and the block must
///   return its own input unchanged, because both sub-blocks' nonlinear
///   interior gets multiplied away by a zeroed projection before either
///   residual add — only the residual path survives. If this assertion
///   fails, the residual wiring is broken and weakening it to an
///   approximate check would hide that.
#[test]
fn a_transformer_block_written_as_toml_evaluates() {
    const SEQUENCE: usize = 4;
    const MODEL: usize = 8;
    const FFN: usize = 16;

    let text = include_str!("../../specs/transformer_block.toml");
    let spec: ProgramSpec = toml::from_str(text).expect("spec parses");
    spec.validate().expect("spec is structurally sound");
    let program = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");

    let symbols = [SEQUENCE as u64];
    crate::shape::infer(&program, &symbols).expect("the block infers");

    let probabilities = spec
        .node
        .iter()
        .position(|node| node.id() == "probabilities")
        .expect("the spec defines a probabilities node");
    let probabilities = NodeId(probabilities as u32);
    let root = NodeId(program.len() as u32 - 1);
    let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");

    // --- run 1: real weights, evaluates to something finite and the
    // softmax invariant still holds inside the larger block.
    let activations = alloc::vec![0.5f32; SEQUENCE * MODEL];
    let inverse_dim = alloc::vec![1.0 / MODEL as f32; SEQUENCE];
    let ones = alloc::vec![1.0f32; SEQUENCE];
    let square_weights = alloc::vec![0.125f32; MODEL * MODEL];
    let gate_up_weights = alloc::vec![0.0625f32; MODEL * FFN];
    let down_weights = alloc::vec![0.0625f32; FFN * MODEL];
    let real_blocks: [&[f32]; 10] = [
        &activations,
        &inverse_dim,
        &ones,
        &square_weights,
        &square_weights,
        &square_weights,
        &square_weights,
        &gate_up_weights,
        &gate_up_weights,
        &down_weights,
    ];

    let evaluated = crate::cpu::evaluate_parallel(
        &program,
        &symbols,
        &real_blocks,
        &[root, probabilities],
        workers,
    )
    .expect("the block evaluates");

    let output = evaluated.root();
    assert_eq!(
        output.len(),
        SEQUENCE * MODEL,
        "a vacuous output proves nothing"
    );
    assert!(
        output.iter().all(|value| value.is_finite()),
        "output must be finite"
    );

    let (rows, _) = evaluated
        .get(probabilities)
        .expect("probabilities were requested");
    for row in rows.as_chunks::<SEQUENCE>().0 {
        let total: f32 = row.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "softmax row sums to {total}, not 1.0"
        );
    }

    // --- run 2: degenerate control. every projection weight is zero, so
    // attention's contribution and the feed-forward's contribution are
    // each multiplied to exactly zero before their residual add — the
    // block must hand its input straight through.
    let zero_square = alloc::vec![0.0f32; MODEL * MODEL];
    let zero_gate_up = alloc::vec![0.0f32; MODEL * FFN];
    let zero_down = alloc::vec![0.0f32; FFN * MODEL];
    let zeroed_blocks: [&[f32]; 10] = [
        &activations,
        &inverse_dim,
        &ones,
        &zero_square,
        &zero_square,
        &zero_square,
        &zero_square,
        &zero_gate_up,
        &zero_gate_up,
        &zero_down,
    ];

    let evaluated_zeroed =
        crate::cpu::evaluate_parallel(&program, &symbols, &zeroed_blocks, &[root], workers)
            .expect("the zeroed block evaluates");

    let residual_output = evaluated_zeroed.root();
    assert_eq!(residual_output.len(), activations.len());
    for (result, input) in residual_output.iter().zip(activations.iter()) {
        assert!(
            (result - input).abs() < 1e-5,
            "residual did not carry: got {result}, expected input {input}"
        );
    }
}

/// `specs/conv2d.toml`'s whole reason to exist: proves a [`NodeSpec::Reduce`]'s
/// `in_map` can now spell the same multi-term windowing an `Elementwise`
/// operand already could — `Reduce(Add)` over `Elementwise(Multiply)`,
/// this file's own `matmul` shape, but with a two-term spatial axis
/// (`h+y`, `w+x`) in place of a bare projection.
///
/// Two invariants, not just finiteness:
/// - output channel 0's kernel is all zero except a single 1 at the 3x3
///   window's centre, so every output pixel is exactly the padded
///   image's centre-tapped pixel — which, because the image was padded
///   by exactly the kernel's radius, is the *original* unpadded pixel at
///   the same coordinate. Reproducing 25 pixels exactly proves the
///   two-term axis addressed the right element at every position, not
///   merely that evaluation completed;
/// - output channel 1's kernel is all zero, a degenerate control: every
///   one of its 25 pixels must be exactly zero, proving the reduction
///   actually depends on the kernel's weights rather than echoing its
///   windowed input regardless of them.
#[test]
fn a_conv2d_written_as_toml_reproduces_its_input_through_a_center_tap_kernel() {
    const IMAGE: usize = 5;
    const PADDED: usize = IMAGE + 2;
    const KERNEL: usize = 3;
    const CENTRE: usize = KERNEL / 2;

    let text = include_str!("../../specs/conv2d.toml");
    let spec: ProgramSpec = toml::from_str(text).expect("spec parses");
    spec.validate().expect("spec is structurally sound");
    let program = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");

    let symbols: [u64; 0] = [];
    crate::shape::infer(&program, &symbols).expect("the convolution infers");

    // image: a zero border (the materialized padding) around a real,
    // non-constant 5x5 interior, so a transposed axis or a wrong offset
    // reads a different, numerically distinct pixel rather than
    // vanishing by symmetry.
    let interior = random_vec(11, IMAGE * IMAGE);
    let mut image = alloc::vec![0.0f32; PADDED * PADDED];
    for row in 0..IMAGE {
        for col in 0..IMAGE {
            image[(row + 1) * PADDED + (col + 1)] = interior[row * IMAGE + col];
        }
    }

    // kernel: [co, ho, wo, kh, kw] = [2, 5, 5, 3, 3]. Channel 0 is a
    // center-tap identity at every output position; channel 1 stays all
    // zero (the `vec!` default).
    let mut kernel = alloc::vec![0.0f32; 2 * IMAGE * IMAGE * KERNEL * KERNEL];
    for out_row in 0..IMAGE {
        for out_col in 0..IMAGE {
            let index = (((out_row * IMAGE + out_col) * KERNEL) + CENTRE) * KERNEL + CENTRE;
            kernel[index] = 1.0;
        }
    }

    let root = NodeId(program.len() as u32 - 1);
    let blocks: [&[f32]; 2] = [&image, &kernel];
    let evaluated = crate::cpu::evaluate(&program, &symbols, &blocks, &[root])
        .expect("the convolution evaluates");

    let output = evaluated.root();
    assert_eq!(
        output.len(),
        2 * IMAGE * IMAGE,
        "a vacuous output proves nothing"
    );

    let channel_0 = &output[..IMAGE * IMAGE];
    let channel_1 = &output[IMAGE * IMAGE..];

    assert_eq!(
        channel_0,
        interior.as_slice(),
        "channel 0's center-tap kernel must reproduce all {} interior pixels exactly",
        IMAGE * IMAGE
    );
    for (index, value) in channel_1.iter().enumerate() {
        assert_eq!(
            *value, 0.0,
            "channel 1's all-zero kernel must produce exactly zero at pixel {index}, got {value}"
        );
    }
}

/// `row @ matrix`, `row` length `d_in`, `matrix` row-major `[d_in,
/// d_out]` — the reference computation `moe_block.toml`'s own test
/// checks the graph against, independent of anything the graph did.
fn matvec(row: &[f32], matrix: &[f32], d_in: usize, d_out: usize) -> alloc::vec::Vec<f32> {
    (0..d_out)
        .map(|out| {
            (0..d_in)
                .map(|inp| row[inp] * matrix[inp * d_out + out])
                .sum()
        })
        .collect()
}

/// The whole reason `Op::Iota` plus `IndexMap::Computed` together are
/// worth having: a top-1 sparse mixture-of-experts feed-forward, built
/// from gate -> argmax route -> gathered expert weights -> the expert's
/// own linear layer, with zero new `Op`/`ScalarOp` variants over what
/// `causal_attention.toml`'s mask and the embedding-lookup worked
/// example already used. `moe_block.toml`'s own header spells out the
/// argmax construction (`mask * iota`, no `Select`, no synthetic
/// `-infinity`) and why the gather is the same mechanism as an
/// embedding lookup with one more non-gathered axis.
///
/// Two tokens, two experts, wired so token 0's gate logits favor expert
/// 0 (3 vs 1) and token 1's favor expert 1 (4 vs 1). `expected_token0`/
/// `expected_token1` are each computed directly from that token's own
/// `x` row and its *routed* expert's weight matrix via [`matvec`],
/// independently of the graph — if the gather read the wrong expert's
/// slab, or the wrong token's `x` row, or `argmax` picked the wrong
/// index, this is what would catch it, not a shape or finiteness check.
///
/// The degenerate control reruns the identical graph with the gate
/// weights swapped, which flips both tokens' routes (token 0 -> expert
/// 1, token 1 -> expert 0 now — see the swapped-gate arithmetic in the
/// comments below), but both experts' weight slabs set to
/// `matrix_a`. If routing still leaked into the result, the output
/// would differ from `x @ matrix_a` for one or both tokens; since the
/// experts are equal, it must not.
#[test]
fn a_moe_block_written_as_toml_routes_each_token_to_its_own_experts_weights() {
    const SEQUENCE: usize = 2;
    const D_IN: usize = 3;
    const D_OUT: usize = 2;
    const N_EXPERTS: usize = 2;

    let text = include_str!("../../specs/moe_block.toml");
    let spec: ProgramSpec = toml::from_str(text).expect("spec parses");
    spec.validate().expect("spec is structurally sound");
    let program = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");

    let symbols = [SEQUENCE as u64];
    crate::shape::infer(&program, &symbols).expect("the moe block infers");

    let root = NodeId(program.len() as u32 - 1);
    let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");

    // token 0 = [3, 2, 1]: logits = [x[0], x[2]] = [3, 1] -> expert 0.
    // token 1 = [1, 2, 4]: logits = [x[0], x[2]] = [1, 4] -> expert 1.
    let x: [f32; SEQUENCE * D_IN] = [3.0, 2.0, 1.0, 1.0, 2.0, 4.0];
    let gate_w: [f32; D_IN * N_EXPERTS] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let matrix_a: [f32; D_IN * D_OUT] = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let matrix_b: [f32; D_IN * D_OUT] = [2.0, 0.0, 0.0, 2.0, 1.0, -1.0];
    let expert_w: [f32; N_EXPERTS * D_IN * D_OUT] = [
        matrix_a[0],
        matrix_a[1],
        matrix_a[2],
        matrix_a[3],
        matrix_a[4],
        matrix_a[5],
        matrix_b[0],
        matrix_b[1],
        matrix_b[2],
        matrix_b[3],
        matrix_b[4],
        matrix_b[5],
    ];
    let blocks: [&[f32]; 3] = [&x, &gate_w, &expert_w];

    let evaluated =
        crate::cpu::evaluate_parallel(&program, &symbols, &blocks, &[root], workers)
            .expect("the moe block evaluates");
    let output = evaluated.root();
    assert_eq!(
        output.len(),
        SEQUENCE * D_OUT,
        "a vacuous output proves nothing"
    );

    let expected_token0 = matvec(&x[0..D_IN], &matrix_a, D_IN, D_OUT);
    let expected_token1 = matvec(&x[D_IN..2 * D_IN], &matrix_b, D_IN, D_OUT);
    for (found, expected) in output[0..D_OUT].iter().zip(&expected_token0) {
        assert!(
            (found - expected).abs() < 1e-5,
            "token 0 (routed to expert 0): got {found}, expected {expected}"
        );
    }
    for (found, expected) in output[D_OUT..2 * D_OUT].iter().zip(&expected_token1) {
        assert!(
            (found - expected).abs() < 1e-5,
            "token 1 (routed to expert 1): got {found}, expected {expected}"
        );
    }

    // --- degenerate control: swap the gate so routing flips.
    // token 0 = [3, 2, 1]: logits = [x[2], x[0]] = [1, 3] -> expert 1.
    // token 1 = [1, 2, 4]: logits = [x[2], x[0]] = [4, 1] -> expert 0.
    // Both experts' weights are `matrix_a`, so the flipped route must
    // not change the answer from `x @ matrix_a`.
    let swapped_gate_w: [f32; D_IN * N_EXPERTS] = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    let uniform_expert_w: [f32; N_EXPERTS * D_IN * D_OUT] = [
        matrix_a[0],
        matrix_a[1],
        matrix_a[2],
        matrix_a[3],
        matrix_a[4],
        matrix_a[5],
        matrix_a[0],
        matrix_a[1],
        matrix_a[2],
        matrix_a[3],
        matrix_a[4],
        matrix_a[5],
    ];
    let degenerate_blocks: [&[f32]; 3] = [&x, &swapped_gate_w, &uniform_expert_w];
    let evaluated_degenerate =
        crate::cpu::evaluate_parallel(&program, &symbols, &degenerate_blocks, &[root], workers)
            .expect("the degenerate moe block evaluates");
    let degenerate_output = evaluated_degenerate.root();

    let expected_uniform_token0 = matvec(&x[0..D_IN], &matrix_a, D_IN, D_OUT);
    let expected_uniform_token1 = matvec(&x[D_IN..2 * D_IN], &matrix_a, D_IN, D_OUT);
    for (found, expected) in degenerate_output[0..D_OUT]
        .iter()
        .zip(&expected_uniform_token0)
    {
        assert!(
            (found - expected).abs() < 1e-5,
            "degenerate control, token 0: got {found}, expected {expected} \
             (routing flipped but experts are identical, so output must not move)"
        );
    }
    for (found, expected) in degenerate_output[D_OUT..2 * D_OUT]
        .iter()
        .zip(&expected_uniform_token1)
    {
        assert!(
            (found - expected).abs() < 1e-5,
            "degenerate control, token 1: got {found}, expected {expected} \
             (routing flipped but experts are identical, so output must not move)"
        );
    }
}

/// Probe for the harder question `a_moe_block_written_as_toml_...` does
/// not answer: does a *fixed* k > 1 stay expressible with zero new
/// ops, or does top-k genuinely need something this crate lacks
/// (`moe_topk2_probe.toml`'s own header names the boundary: a fixed,
/// unrolled k is fine, a general variable-k `TopK` op is not)?
///
/// Three experts, logits `[2, 5, 3]` by construction (see the spec's
/// gate weights): top-2 must select expert 1 (5) then expert 2 (3) and
/// exclude expert 0 (2). Expert 0's weight is `[100, 100]` — wildly
/// different from experts 1 (`[1, 2]`) and 2 (`[3, 4]`) — so a wrong
/// inclusion is not a rounding error, it is off by roughly 30-100x.
/// `expected = x . expert1_weight + x . expert2_weight`, computed
/// independently of the graph via [`matvec`].
#[test]
fn a_topk2_probe_unrolls_two_argmax_rounds_with_exclusion() {
    const D_IN: usize = 2;
    const D_OUT: usize = 1;
    const N_EXPERTS: usize = 3;

    let text = include_str!("../../specs/moe_topk2_probe.toml");
    let spec: ProgramSpec = toml::from_str(text).expect("spec parses");
    spec.validate().expect("spec is structurally sound");
    let program = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");

    let symbols = [1u64];
    crate::shape::infer(&program, &symbols).expect("the top-2 probe infers");

    let root = NodeId(program.len() as u32 - 1);
    let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");

    // logits = x @ gate_w = [1*1+1*1, 1*2+1*3, 1*1+1*2] = [2, 5, 3].
    let x: [f32; D_IN] = [1.0, 1.0];
    let gate_w: [f32; D_IN * N_EXPERTS] = [1.0, 2.0, 1.0, 1.0, 3.0, 2.0];
    let expert0_weight: [f32; D_IN * D_OUT] = [100.0, 100.0];
    let expert1_weight: [f32; D_IN * D_OUT] = [1.0, 2.0];
    let expert2_weight: [f32; D_IN * D_OUT] = [3.0, 4.0];
    let expert_w: [f32; N_EXPERTS * D_IN * D_OUT] = [
        expert0_weight[0],
        expert0_weight[1],
        expert1_weight[0],
        expert1_weight[1],
        expert2_weight[0],
        expert2_weight[1],
    ];
    let blocks: [&[f32]; 3] = [&x, &gate_w, &expert_w];

    let evaluated =
        crate::cpu::evaluate_parallel(&program, &symbols, &blocks, &[root], workers)
            .expect("the top-2 probe evaluates");
    let output = evaluated.root();
    assert_eq!(output.len(), D_OUT, "a vacuous output proves nothing");

    let expected_expert1 = matvec(&x, &expert1_weight, D_IN, D_OUT);
    let expected_expert2 = matvec(&x, &expert2_weight, D_IN, D_OUT);
    let expected = expected_expert1[0] + expected_expert2[0];
    assert!(
        (output[0] - expected).abs() < 1e-5,
        "got {}, expected {expected} (expert 1's {expected_expert1:?} + expert 2's \
         {expected_expert2:?}); expert 0's [100, 100] weight must never contribute",
        output[0]
    );
}

/// SwiGLU over a raw `f32` slice, independent of the graph
/// [`append_moe_ffn`] builds -- the same role [`matvec`] plays for the
/// bare-linear probes above, just with the real per-layer nonlinearity
/// [`append_mistral_layer`]'s dense FFN also runs.
fn swiglu_ffn(
    x: &[f32],
    gate_w: &[f32],
    up_w: &[f32],
    down_w: &[f32],
    d_in: usize,
    hidden: usize,
) -> alloc::vec::Vec<f32> {
    let gate = matvec(x, gate_w, d_in, hidden);
    let up = matvec(x, up_w, d_in, hidden);
    let activated: alloc::vec::Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&gate_value, &up_value)| {
            let silu = gate_value / (1.0 + (-gate_value).exp());
            silu * up_value
        })
        .collect();
    matvec(&activated, down_w, hidden, d_in)
}

/// Independent top-`k` reference: which experts a token's `logits` route
/// to (descending order, ties broken toward the HIGHER index -- ROW 569,
/// `docs/discipline.md`, corrects this comment's own prior claim of
/// "toward the lower index": `Iterator::max_by` returns the LAST of
/// several equally-maximum elements, and `remaining` is built in
/// ascending order, so a tie resolves to the higher index here, exactly
/// matching [`append_moe_ffn`]'s own `mask * iota -> reduce(Maximum)`
/// construction -- proven by `bind::tests::moe_routing_census`'s own
/// exact-tie fixture) and their softmax shares among only that selected
/// set -- `weight_i = exp(logit_i - max) / sum_selected`, the same shift
/// [`append_moe_ffn`]'s doc names.
fn top_k_routes_and_weights(logits: &[f32], k: usize) -> alloc::vec::Vec<(usize, f32)> {
    let mut remaining: alloc::vec::Vec<usize> = (0..logits.len()).collect();
    let mut routes = alloc::vec::Vec::new();
    for _ in 0..k {
        let winner = *remaining
            .iter()
            .max_by(|&&left, &&right| {
                logits[left]
                    .partial_cmp(&logits[right])
                    .expect("logits are finite")
            })
            .expect("k does not exceed the expert count");
        routes.push(winner);
        remaining.retain(|&candidate| candidate != winner);
    }
    let max_logit = routes
        .iter()
        .map(|&expert| logits[expert])
        .fold(f32::NEG_INFINITY, f32::max);
    let unnormalized: alloc::vec::Vec<f32> = routes
        .iter()
        .map(|&expert| (logits[expert] - max_logit).exp())
        .collect();
    let total: f32 = unnormalized.iter().sum();
    routes
        .into_iter()
        .zip(unnormalized)
        .map(|(expert, weight)| (expert, weight / total))
        .collect()
}

/// End-to-end proof for [`append_moe_ffn`]/[`append_mistral_moe_layer`]:
/// two tokens, three experts, top-2 routing, real SwiGLU per expert
/// (not the bare-linear stand-in the two probes above use) and a real
/// softmax combination weight -- everything [`a_moe_block_written_as_toml_...`]
/// and [`a_topk2_probe_...`] proved the algebra can express, now proven
/// for the actual generated code this crate ships, not just the TOML
/// worked examples.
///
/// Router weights are chosen so token 0 (`[3, 2]`) routes to experts
/// `2, 0` (logits `[3, 2, 4]`) and token 1 (`[1, 4]`) routes to experts
/// `2, 1` (logits `[1, 4, 8]`) -- a different pair per token, so a
/// cross-token routing bug (using token 0's route for token 1 or vice
/// versa) is not masked by both tokens agreeing. `expected` is computed
/// entirely independently: [`top_k_routes_and_weights`] picks the route
/// and softmax shares from the same raw `logits` the graph computes
/// on-the-fly, and [`swiglu_ffn`] runs each selected expert's own
/// weights with no dependency on [`Op`]/[`IndexMap`]/[`append_moe_ffn`]
/// itself.
#[test]
fn a_routed_ffn_built_by_append_moe_ffn_matches_an_independent_topk_swiglu_reference() {
    const SEQUENCE: usize = 2;
    const EMBEDDING: usize = 2;
    const FEED_FORWARD: usize = 2;
    const EXPERT_COUNT: u32 = 3;
    const EXPERT_USED_COUNT: u32 = 2;

    let x: [f32; SEQUENCE * EMBEDDING] = [3.0, 2.0, 1.0, 4.0];
    // gate_inp[d, e]: logits[s, e] = sum_d x[s, d] * gate_inp[d, e].
    let gate_inp: [f32; EMBEDDING * EXPERT_COUNT as usize] = [1.0, 0.0, 0.0, 0.0, 1.0, 2.0];

    let gate_weights: [[f32; EMBEDDING * FEED_FORWARD]; 3] = [
        [1.0, 0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0, 2.0],
        [1.0, 1.0, 1.0, 1.0],
    ];
    let up_weights: [[f32; EMBEDDING * FEED_FORWARD]; 3] = [
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
        [2.0, 0.0, 0.0, 2.0],
    ];
    let down_weights: [[f32; FEED_FORWARD * EMBEDDING]; 3] = [
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
    ];

    let stack_experts =
        |weights: &[[f32; EMBEDDING * FEED_FORWARD]; 3]| -> alloc::vec::Vec<f32> {
            weights.iter().flatten().copied().collect()
        };
    let expert_w_gate = stack_experts(&gate_weights);
    let expert_w_up = stack_experts(&up_weights);
    let expert_w_down: alloc::vec::Vec<f32> = down_weights.iter().flatten().copied().collect();

    let mut program = Vec::new();
    let x_node = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(EMBEDDING as u32)],
        "x",
    );
    let gate_inp_node = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(EMBEDDING as u32),
            Extent::Static(EXPERT_COUNT)
        ],
        "gate_inp",
    );
    let expert_w_gate_node = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(EXPERT_COUNT),
            Extent::Static(EMBEDDING as u32),
            Extent::Static(FEED_FORWARD as u32)
        ],
        "expert_w_gate",
    );
    let expert_w_up_node = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(EXPERT_COUNT),
            Extent::Static(EMBEDDING as u32),
            Extent::Static(FEED_FORWARD as u32)
        ],
        "expert_w_up",
    );
    let expert_w_down_node = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(EXPERT_COUNT),
            Extent::Static(FEED_FORWARD as u32),
            Extent::Static(EMBEDDING as u32)
        ],
        "expert_w_down",
    );
    let ones = scalar_constant(&mut program, 1.0);

    let (root, _site) = append_moe_ffn(
        &mut program,
        0,
        x_node,
        gate_inp_node,
        expert_w_gate_node,
        expert_w_up_node,
        expert_w_down_node,
        EXPERT_COUNT,
        EXPERT_USED_COUNT,
        ones,
        ExpertGatingFunc::Softmax,
        None,
    )
    .expect("the routed ffn lowers");

    let gathered_products = program
        .iter()
        .filter(|operation| {
            matches!(
                operation,
                Op::Elementwise { operands, .. }
                    if operands
                        .iter()
                        .any(|(_, index_map)| matches!(index_map, IndexMap::Computed { .. }))
            )
        })
        .count();
    assert_eq!(
        gathered_products,
        3 * EXPERT_USED_COUNT as usize,
        "each selected route gathers its gate, up, and down expert independently"
    );

    let symbols = [SEQUENCE as u64];
    crate::shape::infer(&program, &symbols).expect("the routed ffn infers");

    let blocks: [&[f32]; 5] = [&x, &gate_inp, &expert_w_gate, &expert_w_up, &expert_w_down];
    let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");
    let evaluated =
        crate::cpu::evaluate_parallel(&program, &symbols, &blocks, &[root], workers)
            .expect("the routed ffn evaluates");
    let output = evaluated.root();
    assert_eq!(
        output.len(),
        SEQUENCE * EMBEDDING,
        "a vacuous output proves nothing"
    );

    for (token, x_row) in x.chunks(EMBEDDING).enumerate() {
        let logits: alloc::vec::Vec<f32> = (0..EXPERT_COUNT as usize)
            .map(|expert| {
                (0..EMBEDDING)
                    .map(|dim| x_row[dim] * gate_inp[dim * EXPERT_COUNT as usize + expert])
                    .sum()
            })
            .collect();
        let routes = top_k_routes_and_weights(&logits, EXPERT_USED_COUNT as usize);
        let mut expected = alloc::vec![0.0f32; EMBEDDING];
        for (expert, weight) in routes {
            let expert_out = swiglu_ffn(
                x_row,
                &gate_weights[expert],
                &up_weights[expert],
                &down_weights[expert],
                EMBEDDING,
                FEED_FORWARD,
            );
            for (accum, value) in expected.iter_mut().zip(&expert_out) {
                *accum += weight * value;
            }
        }
        let found = &output[token * EMBEDDING..(token + 1) * EMBEDDING];
        for (found_value, expected_value) in found.iter().zip(&expected) {
            assert!(
                (found_value - expected_value).abs() < 1e-4,
                "token {token}: got {found:?}, expected {expected:?} (independent top-{EXPERT_USED_COUNT} \
                 softmax-weighted swiglu reference)"
            );
        }
    }
}

/// [`ExpertGatingFunc::Sigmoid`]'s independent reference:
/// `route_tokens_to_experts` (`modeling_lfm2_moe.py:208-220`) selects
/// top-`k` by `sigmoid(logits) + bias`, then weights each selected
/// expert by its OWN unbiased `sigmoid(logits)` value, normalized over
/// only the selected set. Mirrors [`top_k_routes_and_weights`]'s own
/// shape (max-by, retain, normalize) with the two extra steps LFM2's
/// gating needs: a sigmoid instead of a softmax, and a bias that
/// participates in the `max_by` but never in the returned weight.
fn sigmoid_topk_routes_and_weights(
    logits: &[f32],
    bias: &[f32],
    k: usize,
) -> alloc::vec::Vec<(usize, f32)> {
    let scores: alloc::vec::Vec<f32> = logits
        .iter()
        .map(|&logit| 1.0 / (1.0 + (-logit).exp()))
        .collect();
    let selection: alloc::vec::Vec<f32> = scores
        .iter()
        .zip(bias)
        .map(|(&score, &b)| score + b)
        .collect();
    let mut remaining: alloc::vec::Vec<usize> = (0..logits.len()).collect();
    let mut routes = alloc::vec::Vec::new();
    for _ in 0..k {
        let winner = *remaining
            .iter()
            .max_by(|&&left, &&right| {
                selection[left]
                    .partial_cmp(&selection[right])
                    .expect("selection scores are finite")
            })
            .expect("k does not exceed the expert count");
        routes.push(winner);
        remaining.retain(|&candidate| candidate != winner);
    }
    let total: f32 = routes.iter().map(|&expert| scores[expert]).sum();
    routes
        .into_iter()
        .map(|expert| (expert, scores[expert] / total))
        .collect()
}

/// [`sigmoid_topk_routes_and_weights`]'s own contract, proved directly
/// against hand-computed `sigmoid` values before any graph is involved:
/// a bias large enough to overturn one token's ranking changes which
/// two experts are selected (proof the bias drives SELECTION), while
/// the returned weight is always the UNBIASED score (proof the bias
/// never reaches the weight) -- the two halves of `exp_probs_b`'s own
/// contract this session closes.
#[test]
fn sigmoid_topk_reference_lets_bias_change_selection_but_never_the_weight() {
    let logits = [3.0f32, 2.0, 4.0];
    let sigmoid = |value: f32| 1.0 / (1.0 + (-value).exp());

    let unbiased = sigmoid_topk_routes_and_weights(&logits, &[0.0, 0.0, 0.0], 2);
    let mut unbiased_experts: alloc::vec::Vec<usize> =
        unbiased.iter().map(|(expert, _)| *expert).collect();
    unbiased_experts.sort_unstable();
    assert_eq!(
        unbiased_experts,
        alloc::vec![0, 2],
        "unbiased sigmoid ranking matches raw-logit ranking: e2 > e0 > e1"
    );

    // pushes e1 (raw sigmoid ~0.881) above e0 (raw sigmoid ~0.953) for
    // SELECTION only: 0.881 + 0.2 = 1.081 > 0.953, but e2 (~0.982) still
    // wins outright, so the selected PAIR changes from {e0, e2} to {e1, e2}.
    let biased = sigmoid_topk_routes_and_weights(&logits, &[0.0, 0.2, 0.0], 2);
    let mut biased_experts: alloc::vec::Vec<usize> =
        biased.iter().map(|(expert, _)| *expert).collect();
    biased_experts.sort_unstable();
    assert_eq!(
        biased_experts,
        alloc::vec![1, 2],
        "a large-enough bias on e1 must swap it in for e0"
    );

    let e1_weight = biased
        .iter()
        .find(|(expert, _)| *expert == 1)
        .map(|(_, weight)| *weight)
        .expect("e1 was selected");
    let e2_weight = biased
        .iter()
        .find(|(expert, _)| *expert == 2)
        .map(|(_, weight)| *weight)
        .expect("e2 was selected");
    let expected_e1 = sigmoid(2.0) / (sigmoid(2.0) + sigmoid(4.0));
    let expected_e2 = sigmoid(4.0) / (sigmoid(2.0) + sigmoid(4.0));
    assert!(
        (e1_weight - expected_e1).abs() < 1e-6,
        "e1's weight must be its UNBIASED sigmoid share ({expected_e1}), got {e1_weight} -- the bias must never reach the weight"
    );
    assert!(
        (e2_weight - expected_e2).abs() < 1e-6,
        "e2's weight must be its unbiased sigmoid share ({expected_e2}), got {e2_weight}"
    );
}

/// End-to-end proof for [`append_moe_ffn`]'s `Sigmoid` branch, same
/// shape as [`a_routed_ffn_built_by_append_moe_ffn_matches_an_independent_topk_swiglu_reference`]
/// (same `x`/`gate_inp`/expert weights, so the same logits `[3, 2, 4]`/
/// `[1, 4, 8]` this time run through `sigmoid` + a per-expert bias
/// instead of softmax): token 0's bias (`+0.2` on expert 1) flips its
/// selected pair from `{0, 2}` (sigmoid-ranking-only) to `{1, 2}` --
/// proof the graph's OWN `Op::Select` exclusion, not just the
/// hand-rolled reference above, routes by the biased score. Token 1
/// keeps the same pair its unbiased ranking already picked, proving the
/// bias is a per-expert additive term the graph applies uniformly, not
/// a per-token special case.
#[test]
fn a_routed_ffn_built_by_append_moe_ffn_with_sigmoid_gating_and_bias_matches_an_independent_reference()
 {
    const SEQUENCE: usize = 2;
    const EMBEDDING: usize = 2;
    const FEED_FORWARD: usize = 2;
    const EXPERT_COUNT: u32 = 3;
    const EXPERT_USED_COUNT: u32 = 2;

    let x: [f32; SEQUENCE * EMBEDDING] = [3.0, 2.0, 1.0, 4.0];
    let gate_inp: [f32; EMBEDDING * EXPERT_COUNT as usize] = [1.0, 0.0, 0.0, 0.0, 1.0, 2.0];
    let bias: [f32; EXPERT_COUNT as usize] = [0.0, 0.2, 0.0];

    let gate_weights: [[f32; EMBEDDING * FEED_FORWARD]; 3] = [
        [1.0, 0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0, 2.0],
        [1.0, 1.0, 1.0, 1.0],
    ];
    let up_weights: [[f32; EMBEDDING * FEED_FORWARD]; 3] = [
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
        [2.0, 0.0, 0.0, 2.0],
    ];
    let down_weights: [[f32; FEED_FORWARD * EMBEDDING]; 3] = [
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 0.0],
    ];

    let stack_experts =
        |weights: &[[f32; EMBEDDING * FEED_FORWARD]; 3]| -> alloc::vec::Vec<f32> {
            weights.iter().flatten().copied().collect()
        };
    let expert_w_gate = stack_experts(&gate_weights);
    let expert_w_up = stack_experts(&up_weights);
    let expert_w_down: alloc::vec::Vec<f32> = down_weights.iter().flatten().copied().collect();

    let mut program = Vec::new();
    let x_node = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(EMBEDDING as u32)],
        "x",
    );
    let gate_inp_node = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(EMBEDDING as u32),
            Extent::Static(EXPERT_COUNT)
        ],
        "gate_inp",
    );
    let expert_w_gate_node = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(EXPERT_COUNT),
            Extent::Static(EMBEDDING as u32),
            Extent::Static(FEED_FORWARD as u32)
        ],
        "expert_w_gate",
    );
    let expert_w_up_node = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(EXPERT_COUNT),
            Extent::Static(EMBEDDING as u32),
            Extent::Static(FEED_FORWARD as u32)
        ],
        "expert_w_up",
    );
    let expert_w_down_node = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(EXPERT_COUNT),
            Extent::Static(FEED_FORWARD as u32),
            Extent::Static(EMBEDDING as u32)
        ],
        "expert_w_down",
    );
    let bias_node = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(EXPERT_COUNT)],
        "bias",
    );
    let ones = scalar_constant(&mut program, 1.0);

    let (root, _site) = append_moe_ffn(
        &mut program,
        0,
        x_node,
        gate_inp_node,
        expert_w_gate_node,
        expert_w_up_node,
        expert_w_down_node,
        EXPERT_COUNT,
        EXPERT_USED_COUNT,
        ones,
        ExpertGatingFunc::Sigmoid,
        Some(bias_node),
    )
    .expect("the sigmoid-gated routed ffn lowers");

    let symbols = [SEQUENCE as u64];
    crate::shape::infer(&program, &symbols).expect("the sigmoid-gated routed ffn infers");

    let blocks: [&[f32]; 6] = [
        &x,
        &gate_inp,
        &expert_w_gate,
        &expert_w_up,
        &expert_w_down,
        &bias,
    ];
    let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");
    let evaluated =
        crate::cpu::evaluate_parallel(&program, &symbols, &blocks, &[root], workers)
            .expect("the sigmoid-gated routed ffn evaluates");
    let output = evaluated.root();
    assert_eq!(
        output.len(),
        SEQUENCE * EMBEDDING,
        "a vacuous output proves nothing"
    );

    for (token, x_row) in x.chunks(EMBEDDING).enumerate() {
        let logits: alloc::vec::Vec<f32> = (0..EXPERT_COUNT as usize)
            .map(|expert| {
                (0..EMBEDDING)
                    .map(|dim| x_row[dim] * gate_inp[dim * EXPERT_COUNT as usize + expert])
                    .sum()
            })
            .collect();
        let routes =
            sigmoid_topk_routes_and_weights(&logits, &bias, EXPERT_USED_COUNT as usize);
        let mut expected = alloc::vec![0.0f32; EMBEDDING];
        for (expert, weight) in &routes {
            let expert_out = swiglu_ffn(
                x_row,
                &gate_weights[*expert],
                &up_weights[*expert],
                &down_weights[*expert],
                EMBEDDING,
                FEED_FORWARD,
            );
            for (accum, value) in expected.iter_mut().zip(&expert_out) {
                *accum += weight * value;
            }
        }
        let found = &output[token * EMBEDDING..(token + 1) * EMBEDDING];
        for (found_value, expected_value) in found.iter().zip(&expected) {
            assert!(
                (found_value - expected_value).abs() < 1e-4,
                "token {token}: got {found:?}, expected {expected:?} (independent sigmoid+bias top-{EXPERT_USED_COUNT} \
                 swiglu reference); routes={routes:?}"
            );
        }
    }
}

/// Deterministic, dependency-free pseudo-random `f32` row generator for
/// the two packed-`Q4_K` tests below -- no RNG crate needed (principle 1:
/// a `u64` multiply-and-shift is the whole requirement), just enough
/// spread across `[-scale, scale]` that a wrong-expert read (a different
/// `scale`, see [`quantized_moe_ffn_over_a_packed_q4k_expert_stack_matches_the_routed_experts_own_swiglu`]'s
/// `1x`/`5x`/`20x` asymmetric per-expert scales) cannot land on the right
/// answer by coincidence.
fn synth_row(seed: u64, len: usize, scale: f32) -> alloc::vec::Vec<f32> {
    (0..len)
        .map(|index| {
            let mixed = seed
                .wrapping_mul(2_654_435_761)
                .wrapping_add((index as u64).wrapping_mul(40_503));
            let unit = ((mixed >> 16) & 0xFFFF) as f32 / 65_535.0;
            (unit * 2.0 - 1.0) * scale
        })
        .collect()
}

/// Packs `rows` independent `[k]`-length `f32` rows (`k` must be
/// [`proxima_gguf::quant::q4_k::QK_K`] exactly, one super-block per row)
/// into one `Q4_K` byte buffer, row-major -- the same physical layout
/// [`proxima_gguf::restack`] produces when it byte-concatenates a real
/// GGUF checkpoint's per-expert tensors (see [`crate::cpu::run_reduce_quantized`]'s
/// own doc on `per_expert_bytes`).
fn quantize_rows(matrix: &[f32], rows: usize, k: usize) -> alloc::vec::Vec<u8> {
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, quantize};

    let mut packed = alloc::vec![0u8; rows * BLOCK_BYTES];
    for (row, out_block) in matrix
        .chunks_exact(k)
        .zip(packed.as_chunks_mut::<BLOCK_BYTES>().0)
    {
        quantize(row, out_block).expect("k is QK_K by construction");
    }
    packed
}

/// The exact inverse of [`quantize_rows`] -- the "equivalent dequantised
/// f32 experts" the binder hand-off names: what a caller gets by
/// dequantizing the packed bytes back to `f32` before handing them to
/// [`crate::cpu::evaluate_parallel`], the non-quantized evaluator.
fn dequantize_rows(packed: &[u8], rows: usize, k: usize) -> alloc::vec::Vec<f32> {
    use proxima_gguf::quant::q4_k::dequantize;

    let mut matrix = alloc::vec![0.0f32; rows * k];
    dequantize(packed, &mut matrix).expect("packed rows dequantize");
    matrix
}

/// Transposes a `[rows, cols]` row-major matrix into `[cols, rows]`.
/// Needed because a packed `Q4_K` node's `rows`/`k` split
/// ([`crate::cpu::run_reduce_quantized`]'s own derivation: `rows` is
/// whichever axis the weight's `IndexMap` varies over among the
/// reduce's OUTPUT axes, `k` is the reduced axis's own extent) and an
/// `Op::Input`'s *declared* axis order (row-major, last axis fastest --
/// [`matvec`]'s own `matrix[inp * d_out + out]` is the same convention)
/// are two different, unrelated conventions for the SAME node: the
/// packed bytes are whatever a real GGUF file's own native layout is
/// (never read through the ordinary strided path at all), while a
/// plain `f32` binding of the identical node IS read through it. The
/// "equivalent dequantised f32 experts" this session's brief asks for
/// therefore is not simply [`dequantize_rows`]'s own output -- it is
/// that output transposed into the declared-shape convention.
fn transpose_rows(matrix: &[f32], rows: usize, cols: usize) -> alloc::vec::Vec<f32> {
    let mut transposed = alloc::vec![0.0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            transposed[col * rows + row] = matrix[row * cols + col];
        }
    }
    transposed
}

/// Isolates [`append_qwen35_ssm_mixer_with_taps`]'s output-projection
/// reduce (spec.rs `out_weight_split_map`/`cur_product`/`cur`, the piece
/// a downstream model-crate qwen35moe stage-by-stage diagnostic
/// (`qwen35moe_layer0_stage_by_stage_position0_matches_tapped_reference`)
/// found `ssm_out_result` off by `scaled_rel_err=1.252` on real
/// `qwen3.6:35b-a3b` bytes) from the rest of the GDN mixer, at
/// NON-degenerate `u=4,g=2,j=32` -- every prior mixer test
/// (`build_ssm_mixer_test_program`,
/// `public_builders_compose_a_one_layer_forward_program`) used
/// `kv_heads=1` or `head_v_dim=1`, so a wrong (u,g,j) nesting order in
/// [`append_qwen35_ssm_mixer_with_taps`]'s own map could never show up
/// on them. Decides whether a divergence traces to the quantized-matmul
/// fold ([`crate::cpu::run_reduce_quantized`]) mishandling this
/// 3-letter contraction over a declared 4-D leaf, or to the map string
/// computing something other than what it says: `packed` (real Q4_K
/// bytes through [`crate::cpu::evaluate_quantized`]) vs `dequantized`
/// (the SAME bytes dequantized and transposed into the node's own
/// declared axis order, through the ordinary [`crate::cpu::evaluate_named`]
/// path) vs `hand_f64` (a from-scratch sum using the exact formula the
/// map string encodes). `packed` diverging from `dequantized` beyond
/// Q4_K's own quantization noise is the quantized-fold bug; `dequantized`
/// diverging from `hand_f64` is the map string not doing what its own
/// formula says.
#[test]
fn ssm_out_projection_reduce_isolated_packed_matches_dequantized_and_hand_computed() {
    use proxima_gguf::quant::q4_k::QK_K;

    let kv_heads = 4usize;
    let group = 2usize;
    let head_v_dim = 32usize;
    let value_dim = kv_heads * group * head_v_dim;
    assert_eq!(
        value_dim, QK_K,
        "one Q4_K super-block per output row, matching quantize_rows's own convention"
    );
    let embedding = 3usize;

    let mut program = Vec::new();
    let gated_out = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(head_v_dim as u32),
            Extent::Static(kv_heads as u32),
            Extent::Static(group as u32)
        ],
        "gated_out",
    );
    let ssm_out = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(value_dim as u32),
            Extent::Static(embedding as u32)
        ],
        "ssm_out",
    );
    let value_head_ones = op::append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![
                Extent::Static(kv_heads as u32),
                Extent::Static(group as u32),
                Extent::Static(head_v_dim as u32),
            ],
            value: 1.0,
        },
    );
    // exact copy of append_qwen35_ssm_mixer_with_taps's own
    // output-projection maps, spec.rs:7093-7117.
    let out_weight_split_map = alloc::format!("{}*j+{group}*u+g,d->ugjd", kv_heads * group);
    let ssm_out_split = elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (ssm_out, out_weight_split_map.as_str()),
            (value_head_ones, "ugj->ugjd"),
        ],
    )
    .expect("ssm_out_split lowers");
    let cur_product = elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(gated_out, "jug->gujd"), (ssm_out_split, "ugjd->gujd")],
    )
    .expect("cur_product lowers");
    let cur = reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        cur_product,
        "gujd->gujd",
        "d->gujd",
    )
    .expect("cur reduce lowers");

    let gated_data = synth_row(101, head_v_dim * kv_heads * group, 1.0);
    // native GGUF physical layout: `embedding` rows, each `value_dim`
    // contiguous contraction elements -- ne0 = value_dim, ne1 = embedding.
    let weight_native = synth_row(202, embedding * value_dim, 1.0);

    let packed = quantize_rows(&weight_native, embedding, value_dim);

    let packed_evaluated = crate::cpu::evaluate_quantized(
        &program,
        &[],
        &[
            crate::cpu::QuantizedBlock::Float32(&gated_data),
            crate::cpu::QuantizedBlock::Q4K(&packed),
        ],
        &[cur],
    )
    .expect("packed reduce evaluates");
    let packed_cur = packed_evaluated.get(cur).expect("cur present").0.to_vec();

    // dequantize the SAME packed bytes, then transpose into the node's
    // own DECLARED axis order (`[value_dim, embedding]`,
    // last-axis-fastest) -- `transpose_rows`'s own doc: the packed
    // bytes and a plain f32 binding of the same declared node are two
    // different, unrelated conventions.
    let dequantized_native = dequantize_rows(&packed, embedding, value_dim);
    let dequantized_declared = transpose_rows(&dequantized_native, embedding, value_dim);

    let dequantized_evaluated = crate::cpu::evaluate_named(
        &program,
        &[],
        &[
            ("gated_out", gated_data.as_slice()),
            ("ssm_out", dequantized_declared.as_slice()),
        ],
        &[cur],
    )
    .expect("dequantized reduce evaluates");
    let (dequantized_cur, _) = dequantized_evaluated.get(cur).expect("cur present");

    // hand-computed f64 sum, the exact formula the map string above
    // encodes: combined = kv_heads*group*j + group*u + g, weight read
    // native[e*value_dim + combined] (embedding-major, contraction-minor,
    // the real GGUF `ssm_out.weight` convention).
    let mut expected = alloc::vec![0.0f64; embedding];
    for u in 0..kv_heads {
        for g in 0..group {
            for j in 0..head_v_dim {
                let jug_index = (j * kv_heads + u) * group + g;
                let combined = (j * kv_heads + u) * group + g;
                let gated = f64::from(gated_data[jug_index]);
                for d in 0..embedding {
                    expected[d] += gated * f64::from(weight_native[d * value_dim + combined]);
                }
            }
        }
    }

    for d in 0..embedding {
        println!(
            "d={d} packed={:.6e} dequantized={:.6e} hand_f64={:.6e}",
            f64::from(packed_cur[d]),
            f64::from(dequantized_cur[d]),
            expected[d],
        );
    }

    let norm_inf = expected
        .iter()
        .fold(0.0f64, |acc, value| acc.max(value.abs()))
        .max(1e-12);
    let packed_vs_dequantized_scaled = packed_cur
        .iter()
        .zip(dequantized_cur.iter())
        .map(|(packed_value, dequantized_value)| {
            (f64::from(*packed_value) - f64::from(*dequantized_value)).abs()
        })
        .fold(0.0f64, f64::max)
        / norm_inf;
    let dequantized_vs_hand_scaled = dequantized_cur
        .iter()
        .zip(expected.iter())
        .map(|(dequantized_value, expected_value)| {
            (f64::from(*dequantized_value) - expected_value).abs()
        })
        .fold(0.0f64, f64::max)
        / norm_inf;
    println!(
        "packed_vs_dequantized_scaled_rel={packed_vs_dequantized_scaled:.6e} \
         dequantized_vs_hand_scaled_rel={dequantized_vs_hand_scaled:.6e}"
    );

    // `< 2e-2`, not `< 1e-5`: this is `f32` accumulation over 256 terms
    // in a different fold order than the `f64` hand loop, not exactness
    // -- well clear of the `~1.25` scaled_rel_err a real (u,g,j)-role
    // mismatch produces (see the fix this test guards,
    // `ssm_out_weight_layout_sweep` in that same downstream stage-by-stage
    // diagnostic).
    assert!(
        dequantized_vs_hand_scaled < 2e-2,
        "the spec's own affine map must mechanically implement the formula it encodes, \
         got scaled_rel_err={dequantized_vs_hand_scaled:.6e}"
    );
    assert!(
        packed_vs_dequantized_scaled < 1e-2,
        "the packed Q4_K quantized-matmul fold must agree with the same bytes dequantized \
         and run through the ordinary path within Q4_K quantization noise, \
         got scaled_rel_err={packed_vs_dequantized_scaled:.6e}"
    );
}

/// The crate's own packed-`Q4_K` matmul kernel, whichever one
/// [`crate::cpu::run_reduce_quantized`]'s own `q4k-int8-dot`-gated arm
/// would actually call for this build -- comparing the graph's gathered
/// output against the OTHER kernel would fail on that kernel's own lossy
/// `Q8_K` activation quantization, not on a wrong-expert read, exactly
/// [`crate::cpu`]'s own `evaluate_quantized_gathered_moe_weight_matches_the_routed_experts_own_matmul`
/// avoids.
fn matmul_q4k_active(weights: &[u8], rows: usize, activation: &[f32]) -> alloc::vec::Vec<f32> {
    #[cfg(feature = "q4k-int8-dot")]
    {
        crate::cpu::matmul_q4k_q8k_f32(weights, rows, activation)
            .expect("packed q4k matmul evaluates")
    }
    #[cfg(not(feature = "q4k-int8-dot"))]
    {
        crate::cpu::matmul_q4k_f32(weights, rows, activation)
            .expect("packed q4k matmul evaluates")
    }
}

/// The decisive proof this session's hand-off exists for:
/// [`gathered_expert_product`] -- unchanged, no widening, the exact
/// construction [`append_moe_ffn`]'s callers (LFM2, Mistral) already
/// build -- correctly gathers one expert's `[rows, k]` slab out of a
/// stacked packed `Q4_K` buffer through the real evaluator
/// (`evaluate_quantized` -> `run_reduce_quantized`'s gather-resolution
/// branch), not a hand-rolled duplicate of the construction.
///
/// Three tokens, three DISTINCT experts (`route = [2, 0, 1]`, chosen so
/// no token's true route is its own position), asymmetric per-expert
/// scales (`1x`/`5x`/`20x`) so a wrong-expert read is off by that same
/// factor, not a rounding difference -- this is a mechanism check.
///
/// The discriminator this session's brief calls for lives in the same
/// function body: the identical program and packed bytes are evaluated a
/// SECOND time with `route` forced to the constant `[0, 0, 0]` --
/// proving the assertion below is capable of failing, not just capable
/// of passing -- then the real route is restored and re-asserted, so the
/// test ends green.
#[test]
fn gathered_expert_product_over_a_packed_q4k_stack_reads_the_routed_experts_own_bytes() {
    use proxima_gguf::quant::q4_k::QK_K;

    const EXPERT_COUNT: u32 = 3;
    const ROWS: usize = 2;
    const SEQUENCE: usize = 3;
    let k = QK_K;

    let expert_scales = [1.0f32, 5.0, 20.0];
    let expert_matrices: alloc::vec::Vec<alloc::vec::Vec<f32>> = expert_scales
        .iter()
        .enumerate()
        .map(|(expert, &scale)| synth_row(101 + expert as u64, ROWS * k, scale))
        .collect();
    let expert_blocks: alloc::vec::Vec<alloc::vec::Vec<u8>> = expert_matrices
        .iter()
        .map(|matrix| quantize_rows(matrix, ROWS, k))
        .collect();
    let stacked_weight: alloc::vec::Vec<u8> = expert_blocks.iter().flatten().copied().collect();

    let activation: alloc::vec::Vec<f32> = synth_row(211, SEQUENCE * k, 1.0);

    let mut program = Vec::new();
    // The gather source is `[expert, k, rows]`: route selects the first
    // axis, then the product maps `k` and `rows` onto its contraction
    // and output axes.
    let stack_node = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(EXPERT_COUNT),
            Extent::Static(k as u32),
            Extent::Static(ROWS as u32),
        ],
        "expert_stack",
    );
    // `Int32`, not `Float32` -- the same declared dtype
    // `gathered_quantized_matmul_program` (`cpu.rs`'s own test) uses for
    // a gather-indices node: `reject_non_float32`'s `index_node_ids`
    // exemption keys off usage (any node referenced as a gather's
    // `indices`), not this declaration, but the gather-resolution code
    // itself (`run_reduce_quantized`'s `raw_index = *index_buffer.get(..)`)
    // always reads the bound buffer as `f32` regardless -- `Int32` here
    // is shape-inference metadata, not a storage format.
    let route_node = input_leaf(
        &mut program,
        DType::Int32,
        alloc::vec![Extent::Static(SEQUENCE as u32)],
        "route",
    );
    let x_node = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(SEQUENCE as u32), Extent::Static(k as u32)],
        "x",
    );

    let product = gathered_expert_product(&mut program, stack_node, route_node, x_node);
    let sum = op::append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
            // keeps `s` (axis 0) and `rows`/`o` (axis 2); reduces `i`/`k`
            // (axis 1) -- the same "so->sio" shape [`append_moe_ffn`]'s
            // own gate/up reduces use.
            out_map: IndexMap::Affine(map::projection(3, &[0, 2])),
            keep: Keep::Reduce,
            name: Some("gathered_q4k_matmul".into()),
        }),
    );

    let true_route = [2.0f32, 0.0, 1.0];
    let expected: alloc::vec::Vec<f32> = true_route
        .iter()
        .enumerate()
        .flat_map(|(token, &route)| {
            let expert = route as usize;
            let activation_row = &activation[token * k..(token + 1) * k];
            matmul_q4k_active(&expert_blocks[expert], ROWS, activation_row)
        })
        .collect();

    let run = |route_data: &[f32]| -> alloc::vec::Vec<f32> {
        let quantized_blocks = [
            crate::cpu::QuantizedBlock::Q4K(&stacked_weight),
            crate::cpu::QuantizedBlock::Float32(route_data),
            crate::cpu::QuantizedBlock::Float32(&activation),
        ];
        crate::cpu::evaluate_quantized(&program, &[], &quantized_blocks, &[sum])
            .expect("the gathered packed matmul evaluates")
            .root()
            .to_vec()
    };

    let constant_route = [0.0f32, 0.0, 0.0];
    let broken = run(&constant_route);
    assert_ne!(
        broken, expected,
        "forcing every token's route to expert 0 must diverge from the true per-token routing \
         (tokens 0 and 2 route elsewhere) -- if this passed, the assertion below could not be trusted"
    );

    let fixed = run(&true_route);
    assert_eq!(
        fixed, expected,
        "gathered_expert_product over a packed Q4_K stack must read exactly the routed expert's own bytes"
    );
}

/// The headline proof: the full [`append_moe_ffn`] graph -- the same
/// function LFM2's own `blk.{layer}.ffn_gate_exps.weight` call site
/// (`spec.rs`'s own LFM2 builder) invokes, unmodified -- run over a
/// stacked packed `Q4_K` expert block reproduces, per token, to within a
/// single-ULP `f32` rounding bound (see the per-element assertion below
/// for why this is not literal `assert_eq!`), what an independent
/// SwiGLU built from the SAME packed bytes' own matmul kernel produces
/// for that token's TRUE routed expert. Three
/// tokens, three distinct experts (`x` is one-hot per token; `gate_inp`
/// is built so token 0 routes to expert 2, token 1 to expert 0, token 2
/// to expert 1 -- no token routes to its own position, so a routing bug
/// that reused one token's route for another could not hide), top-1
/// selection so the softmax combination weight is always exactly `1.0`
/// and cannot mask a wrong-expert read behind a partial blend.
///
/// This is [`gathered_expert_product_over_a_packed_q4k_stack_reads_the_routed_experts_own_bytes`]'s
/// same discriminating construction (asymmetric `1x`/`5x`/`20x`
/// per-expert scales), lifted to the whole FFN graph `append_moe_ffn`
/// actually builds -- gate, up, SiLU, down, three packed operands, one
/// per projection -- proving the widening this session's brief asked
/// for needs no code change: `expert_w_gate`/`expert_w_up`/`expert_w_down`
/// are declared exactly the way [`append_moe_ffn`]'s real callers
/// already declare them (`DType::Float32`, `[expert_count, ..]`), and
/// `evaluate_quantized` binds a `QuantizedBlock::Q4K` to that same node
/// with no spec-side change at all.
#[test]
fn quantized_moe_ffn_over_a_packed_q4k_expert_stack_matches_the_routed_experts_own_swiglu() {
    use proxima_gguf::quant::q4_k::QK_K;

    const EMBEDDING: usize = QK_K;
    const FEED_FORWARD: usize = QK_K;
    const EXPERT_COUNT: u32 = 3;
    const EXPERT_USED_COUNT: u32 = 1;
    const SEQUENCE: usize = 3;

    let expert_scales = [1.0f32, 5.0, 20.0];
    let gate_matrices: alloc::vec::Vec<alloc::vec::Vec<f32>> = expert_scales
        .iter()
        .enumerate()
        .map(|(expert, &scale)| {
            synth_row(1_001 + expert as u64, FEED_FORWARD * EMBEDDING, scale)
        })
        .collect();
    let up_matrices: alloc::vec::Vec<alloc::vec::Vec<f32>> = expert_scales
        .iter()
        .enumerate()
        .map(|(expert, &scale)| {
            synth_row(2_002 + expert as u64, FEED_FORWARD * EMBEDDING, scale)
        })
        .collect();
    let down_matrices: alloc::vec::Vec<alloc::vec::Vec<f32>> = expert_scales
        .iter()
        .enumerate()
        .map(|(expert, &scale)| {
            synth_row(3_003 + expert as u64, EMBEDDING * FEED_FORWARD, scale)
        })
        .collect();

    let gate_blocks: alloc::vec::Vec<alloc::vec::Vec<u8>> = gate_matrices
        .iter()
        .map(|matrix| quantize_rows(matrix, FEED_FORWARD, EMBEDDING))
        .collect();
    let up_blocks: alloc::vec::Vec<alloc::vec::Vec<u8>> = up_matrices
        .iter()
        .map(|matrix| quantize_rows(matrix, FEED_FORWARD, EMBEDDING))
        .collect();
    let down_blocks: alloc::vec::Vec<alloc::vec::Vec<u8>> = down_matrices
        .iter()
        .map(|matrix| quantize_rows(matrix, EMBEDDING, FEED_FORWARD))
        .collect();

    let stacked_gate: alloc::vec::Vec<u8> = gate_blocks.iter().flatten().copied().collect();
    let stacked_up: alloc::vec::Vec<u8> = up_blocks.iter().flatten().copied().collect();
    let stacked_down: alloc::vec::Vec<u8> = down_blocks.iter().flatten().copied().collect();

    // token s is the one-hot vector at index s; gate_inp's rows 0..3
    // make index 0 favor expert 2, index 1 favor expert 0, index 2
    // favor expert 1 -- every other row is zero, so only the one-hot
    // position drives the route.
    let mut x = alloc::vec![0.0f32; SEQUENCE * EMBEDDING];
    for token in 0..SEQUENCE {
        x[token * EMBEDDING + token] = 1.0;
    }
    let true_route = [2usize, 0, 1];
    let mut gate_inp = alloc::vec![0.0f32; EMBEDDING * EXPERT_COUNT as usize];
    for (token, &expert) in true_route.iter().enumerate() {
        gate_inp[token * EXPERT_COUNT as usize + expert] = 5.0;
    }

    let mut program = Vec::new();
    let x_node = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(EMBEDDING as u32)],
        "x",
    );
    let gate_inp_node = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(EMBEDDING as u32),
            Extent::Static(EXPERT_COUNT)
        ],
        "gate_inp",
    );
    let expert_w_gate_node = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(EXPERT_COUNT),
            Extent::Static(EMBEDDING as u32),
            Extent::Static(FEED_FORWARD as u32),
        ],
        "expert_w_gate",
    );
    let expert_w_up_node = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(EXPERT_COUNT),
            Extent::Static(EMBEDDING as u32),
            Extent::Static(FEED_FORWARD as u32),
        ],
        "expert_w_up",
    );
    let expert_w_down_node = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(EXPERT_COUNT),
            Extent::Static(FEED_FORWARD as u32),
            Extent::Static(EMBEDDING as u32),
        ],
        "expert_w_down",
    );
    let ones = scalar_constant(&mut program, 1.0);

    let (root, _site) = append_moe_ffn(
        &mut program,
        0,
        x_node,
        gate_inp_node,
        expert_w_gate_node,
        expert_w_up_node,
        expert_w_down_node,
        EXPERT_COUNT,
        EXPERT_USED_COUNT,
        ones,
        ExpertGatingFunc::Softmax,
        None,
    )
    .expect("the packed-stack routed ffn lowers");

    let symbols = [SEQUENCE as u64];
    crate::shape::infer(&program, &symbols).expect("the packed-stack routed ffn infers");

    let quantized_blocks = [
        crate::cpu::QuantizedBlock::Float32(&x),
        crate::cpu::QuantizedBlock::Float32(&gate_inp),
        crate::cpu::QuantizedBlock::Q4K(&stacked_gate),
        crate::cpu::QuantizedBlock::Q4K(&stacked_up),
        crate::cpu::QuantizedBlock::Q4K(&stacked_down),
    ];
    let evaluated =
        crate::cpu::evaluate_quantized(&program, &symbols, &quantized_blocks, &[root])
            .expect("the packed-stack routed ffn evaluates over Q4_K");
    let output = evaluated.root();
    assert_eq!(
        output.len(),
        SEQUENCE * EMBEDDING,
        "a vacuous output proves nothing"
    );

    for (token, &expert) in true_route.iter().enumerate() {
        let x_row = &x[token * EMBEDDING..(token + 1) * EMBEDDING];
        let gate = matmul_q4k_active(&gate_blocks[expert], FEED_FORWARD, x_row);
        let up = matmul_q4k_active(&up_blocks[expert], FEED_FORWARD, x_row);
        let hidden: alloc::vec::Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&gate_value, &up_value)| {
                let silu = gate_value / (1.0 + (-gate_value).exp());
                silu * up_value
            })
            .collect();
        let expected = matmul_q4k_active(&down_blocks[expert], EMBEDDING, &hidden);

        let found = &output[token * EMBEDDING..(token + 1) * EMBEDDING];
        // A tight relative bound, not `assert_eq!`: at `rows=EMBEDDING=
        // FEED_FORWARD=256` (mandatory -- `Q4_K` needs a whole
        // `QK_K`-multiple contraction axis on BOTH projections), every
        // one of these matmuls clears `PARALLEL_THRESHOLD` (4096 macs;
        // `crate::sized::PARALLEL_THRESHOLD`), so `quantized_matmul_workers`
        // threads the row batch. This direct reference call's own
        // `session: None` and the graph's own internal session context
        // are not the same value, so they are not guaranteed to pick the
        // identical worker chunking -- rows are still computed
        // independently either way (no cross-row summation), so the
        // measured gap tops out at single-ULP `f32` rounding (~1e-7
        // relative, observed), five orders of magnitude below the
        // 1x/5x/20x scale gap a wrong-routed expert would produce. The
        // discriminator test above already proves THIS reads the right
        // expert; this bound proves it reads it correctly.
        for (found_value, expected_value) in found.iter().zip(expected.iter()) {
            let scale = found_value.abs().max(expected_value.abs()).max(1.0);
            assert!(
                (found_value - expected_value).abs() / scale < 1e-4,
                "token {token} routed to expert {expert}: append_moe_ffn over a packed Q4_K expert \
                 stack ({found_value}) diverges from that expert's own standalone Q4_K swiglu \
                 ({expected_value}) past single-ULP rounding"
            );
        }
    }

    // Secondary, honestly-labeled sanity check, not the primary
    // correctness gate: the SAME graph run over the DEQUANTIZED f32
    // experts (`evaluate_parallel`, no int8 activation path at all)
    // stays close to the packed-quantized run -- a LOOSE bound, since
    // `q4k-int8-dot`'s own `Q8_K` activation quantization is a real,
    // already-measured, expected source of numerical difference from a
    // naive f32 dot on dequantized weights (see `cpu.rs`'s own
    // `relative_max_diff < 0.01` sanity bound for the dense codec path),
    // not a routing defect this test exists to catch.
    let dequantized_gate: alloc::vec::Vec<f32> = gate_blocks
        .iter()
        .flat_map(|bytes| {
            transpose_rows(
                &dequantize_rows(bytes, FEED_FORWARD, EMBEDDING),
                FEED_FORWARD,
                EMBEDDING,
            )
        })
        .collect();
    let dequantized_up: alloc::vec::Vec<f32> = up_blocks
        .iter()
        .flat_map(|bytes| {
            transpose_rows(
                &dequantize_rows(bytes, FEED_FORWARD, EMBEDDING),
                FEED_FORWARD,
                EMBEDDING,
            )
        })
        .collect();
    let dequantized_down: alloc::vec::Vec<f32> = down_blocks
        .iter()
        .flat_map(|bytes| {
            transpose_rows(
                &dequantize_rows(bytes, EMBEDDING, FEED_FORWARD),
                EMBEDDING,
                FEED_FORWARD,
            )
        })
        .collect();
    let f32_blocks: [&[f32]; 5] = [
        &x,
        &gate_inp,
        &dequantized_gate,
        &dequantized_up,
        &dequantized_down,
    ];
    let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");
    let dequantized_evaluated =
        crate::cpu::evaluate_parallel(&program, &symbols, &f32_blocks, &[root], workers)
            .expect("the dequantized-f32 routed ffn evaluates");
    let dequantized_output = dequantized_evaluated.root();
    for (found_value, dequantized_value) in output.iter().zip(dequantized_output.iter()) {
        let scale = found_value.abs().max(dequantized_value.abs()).max(1.0);
        assert!(
            (found_value - dequantized_value).abs() / scale < 0.05,
            "packed-Q4_K output {found_value} and dequantized-f32 output {dequantized_value} diverge \
             past Q8_K activation-quantization's own expected error budget"
        );
    }
}

/// Independent reference for grouped-query attention: a plain
/// `q @ k^T` -> causal softmax -> `@ v` over raw f32 slices, with no
/// dependency on `Op`, `IndexMap`, or anything else the graph under test
/// builds. `q`/`k`/`v` come from a linear projection (`project`) laid
/// out the same row-major way `Input`'s declared `shape` implies
/// (`[dim_in, heads, head_dim]`, slowest axis first) — the one place
/// this function and the spec's `wq`/`wk`/`wv` shapes must agree, and
/// the reason both are documented at the call site.
fn project(
    x: &[f32],
    weight: &[f32],
    sequence: usize,
    dim_in: usize,
    heads: usize,
    head_dim: usize,
) -> alloc::vec::Vec<f32> {
    let mut projected = alloc::vec![0.0f32; sequence * heads * head_dim];
    for position in 0..sequence {
        for head in 0..heads {
            for dim in 0..head_dim {
                let mut accumulator = 0.0f32;
                for input_dim in 0..dim_in {
                    let activation = x[position * dim_in + input_dim];
                    let coefficient =
                        weight[input_dim * heads * head_dim + head * head_dim + dim];
                    accumulator += activation * coefficient;
                }
                projected[(position * heads + head) * head_dim + dim] = accumulator;
            }
        }
    }
    projected
}

/// The six sizes one grouped-query-attention case needs, gathered into
/// one type so `expected_gqa_attended` and `run_gqa_case` each take a
/// handful of arguments instead of one per size.
#[derive(Debug, Clone, Copy)]
struct GqaDims {
    sequence: usize,
    dim_in: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    group: usize,
}

/// `expected[((s*kv_heads+u)*group+g)*head_dim+d]` — the same `sugd`
/// physical order `gqa_attention.toml`'s `attended` reduce declares in
/// its `out_map`. `h = u*group + g` is the property under test, spelled
/// here as plain arithmetic rather than an index map, so the two can
/// disagree if the graph's addressing is wrong.
fn expected_gqa_attended(
    x: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    dims: GqaDims,
) -> alloc::vec::Vec<f32> {
    let GqaDims {
        sequence,
        dim_in,
        query_heads,
        kv_heads,
        head_dim,
        group,
    } = dims;
    let q = project(x, wq, sequence, dim_in, query_heads, head_dim);
    let k = project(x, wk, sequence, dim_in, kv_heads, head_dim);
    let v = project(x, wv, sequence, dim_in, kv_heads, head_dim);

    let mut output = alloc::vec![0.0f32; sequence * kv_heads * group * head_dim];
    for query_position in 0..sequence {
        for kv_head in 0..kv_heads {
            for offset in 0..group {
                let query_head = kv_head * group + offset;
                let mut scores = alloc::vec![f32::NEG_INFINITY; sequence];
                for key_position in 0..=query_position {
                    let mut score = 0.0f32;
                    for dim in 0..head_dim {
                        let query_value =
                            q[(query_position * query_heads + query_head) * head_dim + dim];
                        let key_value = k[(key_position * kv_heads + kv_head) * head_dim + dim];
                        score += query_value * key_value;
                    }
                    scores[key_position] = score;
                }
                let max_score = scores.iter().copied().fold(f32::MIN, f32::max);
                let exponentials: alloc::vec::Vec<f32> = scores
                    .iter()
                    .map(|&score| {
                        if score.is_finite() {
                            (score - max_score).exp()
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let total: f32 = exponentials.iter().sum();
                for dim in 0..head_dim {
                    let mut accumulator = 0.0f32;
                    for key_position in 0..sequence {
                        let probability = exponentials[key_position] / total;
                        let value_value =
                            v[(key_position * kv_heads + kv_head) * head_dim + dim];
                        accumulator += probability * value_value;
                    }
                    let index = ((query_position * kv_heads + kv_head) * group + offset)
                        * head_dim
                        + dim;
                    output[index] = accumulator;
                }
            }
        }
    }
    output
}

/// The property that makes this GQA rather than plain multi-head
/// attention: query heads sharing a kv head must attend against the
/// *same* k/v head, and query heads in different groups must attend
/// against *different* ones. `wk`/`wv` give kv head 0 and kv head 1 a
/// +-10.0 offset on top of independent LCG noise, so a wrong kv-head
/// selection (e.g. every group reading kv head 0) shows up as an
/// order-of-magnitude disagreement, not a rounding error — the same
/// sharpness `a_topk2_probe_...`'s 100-vs-1 weights use.
///
/// `expected_gqa_attended` computes the same arithmetic independently
/// of the graph, in `sugd` order, so it is compared element by element
/// against `attended` (the spec's root) rather than read back from any
/// intermediate the graph produced.
fn run_gqa_case(text: &str, dims: GqaDims, seed: u64) {
    let GqaDims {
        sequence,
        dim_in,
        query_heads,
        kv_heads,
        head_dim,
        group,
    } = dims;

    let spec: ProgramSpec = toml::from_str(text).expect("spec parses");
    spec.validate().expect("spec is structurally sound");
    let program = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");

    let symbols = [sequence as u64];
    crate::shape::infer(&program, &symbols).expect("the gqa block infers");

    let x = random_vec(seed, sequence * dim_in);
    let wq = random_vec(seed + 1, dim_in * query_heads * head_dim);

    let wk_noise = random_vec(seed + 2, dim_in * kv_heads * head_dim);
    let wv_noise = random_vec(seed + 3, dim_in * kv_heads * head_dim);
    let mut wk = alloc::vec![0.0f32; dim_in * kv_heads * head_dim];
    let mut wv = alloc::vec![0.0f32; dim_in * kv_heads * head_dim];
    for input_dim in 0..dim_in {
        for kv_head in 0..kv_heads {
            let bias = if kv_head == 0 { 10.0 } else { -10.0 };
            for dim in 0..head_dim {
                let index = input_dim * kv_heads * head_dim + kv_head * head_dim + dim;
                wk[index] = wk_noise[index] + bias;
                wv[index] = wv_noise[index] + bias;
            }
        }
    }

    // `group_ones` only pins `q_grouped`'s (kv-head, group) extents for
    // `shape::infer` (see `gqa_attention.toml`'s header) — it must stay
    // exactly 1.0 or it would silently rescale every query head's score.
    let group_ones = alloc::vec![1.0f32; kv_heads * group];

    let probabilities = spec
        .node
        .iter()
        .position(|node| node.id() == "probabilities")
        .expect("the spec defines a probabilities node");
    let probabilities = NodeId(probabilities as u32);
    let root = NodeId(program.len() as u32 - 1);

    let blocks: [&[f32]; 5] = [&x, &wq, &wk, &wv, &group_ones];
    let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");
    let evaluated = crate::cpu::evaluate_parallel(
        &program,
        &symbols,
        &blocks,
        &[root, probabilities],
        workers,
    )
    .expect("the gqa block evaluates");

    let output = evaluated.root();
    let expected_len = sequence * kv_heads * group * head_dim;
    assert_eq!(
        output.len(),
        expected_len,
        "a vacuous output proves nothing"
    );
    assert!(
        output.iter().all(|value| value.is_finite()),
        "output must be finite"
    );

    let expected = expected_gqa_attended(&x, &wq, &wk, &wv, dims);
    assert_eq!(expected.len(), expected_len);

    let mut compared = 0usize;
    for (index, (&found, &wanted)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (found - wanted).abs() < 1e-3,
            "element {index}: graph produced {found}, independent reference produced \
             {wanted} — a query head is attending against the wrong kv head"
        );
        compared += 1;
    }
    assert_eq!(
        compared, expected_len,
        "every element must be checked, not a subset"
    );

    let (rows, _) = evaluated
        .get(probabilities)
        .expect("probabilities were requested");
    assert_eq!(rows.len(), sequence * sequence * kv_heads * group);

    let mut checked = 0usize;
    for query_position in 0..sequence {
        for kv_head in 0..kv_heads {
            for offset in 0..group {
                let mut total = 0.0f32;
                for key_position in 0..sequence {
                    let index = ((query_position * sequence + key_position) * kv_heads
                        + kv_head)
                        * group
                        + offset;
                    let probability = rows[index];
                    if key_position > query_position {
                        assert_eq!(
                            probability, 0.0,
                            "query {query_position} kv-head {kv_head} group-offset \
                             {offset} key {key_position} is strictly upper-triangular \
                             and must be masked to exactly 0.0, found {probability}"
                        );
                    }
                    total += probability;
                    checked += 1;
                }
                assert!(
                    (total - 1.0).abs() < 1e-5,
                    "query {query_position} kv-head {kv_head} group-offset {offset} \
                     softmax row sums to {total}, not 1.0"
                );
            }
        }
    }
    assert_eq!(
        checked,
        sequence * sequence * kv_heads * group,
        "every probability cell must be checked, not a subset"
    );
}

#[test]
fn a_gqa_attention_block_groups_query_heads_onto_shared_kv_heads() {
    let text = include_str!("../../specs/gqa_attention.toml");
    let dims = GqaDims {
        sequence: 4,
        dim_in: 4,
        query_heads: 4,
        kv_heads: 2,
        head_dim: 4,
        group: 2,
    };
    run_gqa_case(text, dims, 31);
}

/// `deepseek-coder-33b` is `head_count=56`, `head_count_kv=8` — group 7,
/// not a power of two. This is that shape at a hand-checkable size (6
/// query heads, 2 kv heads, group 3): the only spec change from
/// `gqa_attention.toml` is `wq`'s head extent and the affine
/// coefficient (`2*u+g` -> `3*u+g`), so this test is the check that
/// `coeff=3` behaves identically to `coeff=2`, not an assumption resting
/// on the power-of-two case alone.
#[test]
fn a_gqa_attention_block_with_a_non_power_of_two_group_groups_query_heads_onto_shared_kv_heads()
{
    let text = include_str!("../../specs/gqa_attention_group3.toml");
    let dims = GqaDims {
        sequence: 4,
        dim_in: 4,
        query_heads: 6,
        kv_heads: 2,
        head_dim: 4,
        group: 3,
    };
    run_gqa_case(text, dims, 41);
}

/// The regression this fix exists for: `1/sqrt(head_dim)` missing from
/// `scores` before the mask does not fail `sums to 1.0` — a saturated
/// softmax is still a valid softmax — so that invariant alone cannot
/// catch it. This builds the same score/scale/softmax composition
/// `append_mistral_layer` now runs (`q . k`, multiply by
/// [`scalar_constant`], then max-shift/exp/normalize, no mask
/// — masking is `causal_attention.toml`'s own proven concern, not this
/// one's), at the model's real `head_dim=128`, built TWICE on the same
/// `q`/`k`: once with the scaling step omitted entirely (exactly the
/// pre-fix graph — before this fix `scores` fed the mask directly, which
/// is what `unscaled` below reproduces) and once with it present, using
/// the actual production helper rather than a hand-rolled stand-in.
///
/// `q` is the all-ones vector and key 0 is `0.15 * q` (dot product
/// `0.15 * 128 = 19.2` exactly, no estimation); the other 15 keys are
/// all-zero (dot product `0.0` exactly). Chosen, not randomly sampled,
/// so the separation is provable arithmetic: unscaled, `exp(19.2)` so
/// overwhelms `15 * exp(0)` that key 0 takes essentially the whole
/// distribution; scaled by `1/sqrt(128)`, the same score drops to
/// `1.697`, and `exp(1.697) = 5.46` split against `15 * exp(0) = 15`
/// cannot exceed half the row.
///
/// The assertion a plain "sums to 1.0" check would have missed: the
/// unscaled row's largest weight must be near-one-hot (`> 0.9`) and the
/// scaled row's must not (`< 0.5`) — both rows still sum to `1.0`, so
/// only a degeneracy check, not a normalization check, tells them apart.
#[test]
fn scaling_attention_scores_by_inverse_sqrt_head_dim_prevents_softmax_saturation() {
    const HEAD_DIM: usize = 128;
    const KEYS: usize = 16;
    const KEY_ZERO_WEIGHT: f32 = 0.15;

    fn build(scaled: bool) -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let query = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(0), Extent::Static(HEAD_DIM as u32)],
            "q",
        );
        let key = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(KEYS as u32), Extent::Static(HEAD_DIM as u32)],
            "k",
        );

        let score_product = elementwise(
            &mut program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(query, "sh->sth"), (key, "th->sth")],
        )
        .expect("score product builds");
        let scores = reduce(
            &mut program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            score_product,
            "sth->sth",
            "st->sth",
        )
        .expect("scores reduce builds");
        let scores = if scaled {
            let inv_sqrt_head_dim =
                scalar_constant(&mut program, 1.0 / (HEAD_DIM as f32).sqrt());
            elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(scores, "st->st"), (inv_sqrt_head_dim, "->st")],
            )
            .expect("scaling multiply builds")
        } else {
            scores
        };
        let score_max = reduce(
            &mut program,
            DType::Float32,
            ScalarOp::Maximum,
            ReduceInit::NegativeInfinity,
            scores,
            "st->st",
            "s->st",
        )
        .expect("max reduce builds");
        let shifted = elementwise(
            &mut program,
            DType::Float32,
            ScalarOp::Subtract,
            &[(scores, "st->st"), (score_max, "s->st")],
        )
        .expect("shift builds");
        let weights = elementwise(
            &mut program,
            DType::Float32,
            ScalarOp::Exponential,
            &[(shifted, "st->st")],
        )
        .expect("exponential builds");
        let weight_sum = reduce(
            &mut program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            weights,
            "st->st",
            "s->st",
        )
        .expect("weight sum reduce builds");
        let inv_weight_sum = elementwise(
            &mut program,
            DType::Float32,
            ScalarOp::Reciprocal,
            &[(weight_sum, "s->s")],
        )
        .expect("reciprocal builds");
        let probabilities = elementwise(
            &mut program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(weights, "st->st"), (inv_weight_sum, "s->st")],
        )
        .expect("probabilities multiply builds");
        (program, probabilities)
    }

    let query_vector = alloc::vec![1.0f32; HEAD_DIM];
    let mut key_vectors = alloc::vec![0.0f32; KEYS * HEAD_DIM];
    key_vectors[0..HEAD_DIM].fill(KEY_ZERO_WEIGHT);
    let symbols = [1u64];
    let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");

    let evaluate = |scaled: bool| -> Vec<f32> {
        let (program, probabilities) = build(scaled);
        crate::shape::infer(&program, &symbols)
            .expect("the isolated score/softmax slice infers");
        let root = NodeId(program.len() as u32 - 1);
        assert_eq!(
            root, probabilities,
            "probabilities is the program's own last node"
        );
        let blocks: [&[f32]; 2] = [&query_vector, &key_vectors];
        let evaluated =
            crate::cpu::evaluate_parallel(&program, &symbols, &blocks, &[root], workers)
                .expect("the isolated score/softmax slice evaluates");
        evaluated.root().to_vec()
    };

    let unscaled = evaluate(false);
    let scaled = evaluate(true);

    for (label, row) in [
        ("unscaled (pre-fix)", &unscaled),
        ("scaled (post-fix)", &scaled),
    ] {
        let total: f32 = row.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-4,
            "{label} softmax row sums to {total}, not 1.0"
        );
    }

    let unscaled_max = unscaled[0];
    let scaled_max = scaled[0];
    assert!(
        unscaled_max > 0.9,
        "pre-fix (no scaling) softmax should saturate toward one-hot over head_dim={HEAD_DIM} \
         (key 0's raw score is {} against 15 keys at 0.0), but key 0's weight is only \
         {unscaled_max} — the test data no longer reproduces the bug this regression test \
         exists to catch",
        KEY_ZERO_WEIGHT * HEAD_DIM as f32
    );
    assert!(
        scaled_max < 0.5,
        "post-fix (scaled by 1/sqrt(head_dim)) softmax should blend across {KEYS} keys instead \
         of collapsing to one, but key 0's weight is {scaled_max}, no better than the unscaled \
         {unscaled_max} — 1/sqrt(head_dim) is not doing its job"
    );
}

/// The milestone this crate has been building toward: one real
/// openchat-3.5-1210 / Mistral-7B transformer layer, RoPE + GQA +
/// causal mask composed together, at the model's own dimensions
/// (`embedding_length=4096`, `head_count=32`, `head_count_kv=8`,
/// `head_dim=128`, `feed_forward_length=14336`) — not a toy shrink of
/// them. `mistral_layer.toml`'s header records the one addressing
/// decision the composition forced: the rotated dot product is
/// recovered as even-pairs-plus-odd-pairs rather than re-interleaved,
/// because interleaving needs a write-placement op this crate does not
/// have. No new `Op` or `ScalarOp` was needed here; if one had been,
/// this comment would say so instead.
///
/// Shape inference is cheap — symbolic arithmetic over extents, not
/// data — and runs here at the model's real context length (8192) and
/// again at the small sequence length the evaluation test below uses,
/// proving the same graph types at both.
///
/// Evaluating this spec at its real embedding/feed-forward dimensions
/// was tried and MEASURED, not assumed to be fine: random weight
/// generation (~870MB across `wq`/`wk`/`wv`/`wo`/`w_gate`/`w_up`/
/// `w_down`) took 2.77s, but `evaluate_parallel` itself did not finish
/// inside a 90s budget even at `SEQUENCE=4` — the elementwise nodes
/// feeding `gate_product`/`up_product`/`down_product` materialize a
/// full `seq * embedding * feed_forward` product ahead of their reduce
/// (`seq=4` gives `4 * 4096 * 14336` = 235M elements, ~940MB, per node,
/// three of them), independent of how small `seq` is. So the evaluation
/// test below runs `mistral_layer_small.toml` instead — node-for-node
/// the same file with every non-sequence axis divided down while
/// preserving the real ratios (GQA group stays 4, RoPE still rotates
/// the full head_dim) — see that file's header for the exact numbers.
#[test]
fn a_mistral_layer_written_as_toml_infers_at_its_real_dimensions() {
    const REAL_CONTEXT: u64 = 8192;
    const SMALL_SEQUENCE: u64 = 4;

    let text = include_str!("../../specs/mistral_layer.toml");
    let spec: ProgramSpec = toml::from_str(text).expect("spec parses");
    spec.validate().expect("spec is structurally sound");
    let program = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");

    crate::shape::infer(&program, &[REAL_CONTEXT])
        .expect("the layer infers at its real context length");
    crate::shape::infer(&program, &[SMALL_SEQUENCE])
        .expect("the layer infers at a small sequence length too");
}

/// Wall-clock probe for `bind.rs`'s reduce-fusion cost fix: runs
/// `mistral_layer.toml` at the model's real dimensions
/// (`embedding=4096`, `feed_forward=14336`) at `sequence=4`, the exact
/// configuration the sibling milestone test above found too slow to run
/// unfused (`ffn_out`'s reduce absorbing the whole SwiGLU activation
/// chain recomputed it once per `embedding` element instead of once per
/// its own `seq*feed_forward`). `#[ignore]`d — ~870MB of random weights
/// plus a multi-second real run does not belong in the default
/// `nextest` budget; run explicitly with `--ignored` when re-measuring.
#[test]
#[ignore = "measures the real-dimension mistral layer's wall clock; run explicitly"]
fn a_mistral_layer_written_as_toml_evaluates_at_its_real_dimensions() {
    const SEQUENCE: usize = 4;
    const EMBEDDING: usize = 4096;
    const QUERY_HEADS: usize = 32;
    const KV_HEADS: usize = 8;
    const HEAD_DIM: usize = 128;
    const PAIRS: usize = HEAD_DIM / 2;
    const GROUP: usize = QUERY_HEADS / KV_HEADS;
    const FEED_FORWARD: usize = 14336;

    let text = include_str!("../../specs/mistral_layer.toml");
    let spec: ProgramSpec = toml::from_str(text).expect("spec parses");
    spec.validate().expect("spec is structurally sound");
    let program = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");

    let symbols = [SEQUENCE as u64];
    let shapes = crate::shape::infer(&program, &symbols).expect("the real layer infers");

    let activations = random_vec(101, SEQUENCE * EMBEDDING);
    let epsilon = alloc::vec![1e-5f32; SEQUENCE];
    let wq = random_vec(102, EMBEDDING * QUERY_HEADS * HEAD_DIM);
    let wk = random_vec(103, EMBEDDING * KV_HEADS * HEAD_DIM);
    let wv = random_vec(104, EMBEDDING * KV_HEADS * HEAD_DIM);
    let wo = random_vec(105, KV_HEADS * GROUP * HEAD_DIM * EMBEDDING);
    let w_gate = random_vec(106, EMBEDDING * FEED_FORWARD);
    let w_up = random_vec(107, EMBEDDING * FEED_FORWARD);
    let w_down = random_vec(108, FEED_FORWARD * EMBEDDING);
    let cos = random_vec(109, SEQUENCE * PAIRS);
    let sin = random_vec(110, SEQUENCE * PAIRS);
    let attn_norm_weight = alloc::vec![1.0f32; EMBEDDING];
    let ffn_norm_weight = alloc::vec![1.0f32; EMBEDDING];

    let blocks: [&[f32]; 13] = [
        &activations,
        &epsilon,
        &wq,
        &wk,
        &wv,
        &wo,
        &w_gate,
        &w_up,
        &w_down,
        &cos,
        &sin,
        &attn_norm_weight,
        &ffn_norm_weight,
    ];

    let ffn_out = spec
        .node
        .iter()
        .position(|node| node.id() == "ffn_out")
        .expect("the spec defines an ffn_out node");
    let ffn_out = NodeId(ffn_out as u32);
    let root = NodeId(program.len() as u32 - 1);

    let bound = crate::bind::bind(
        &program,
        &shapes,
        &[root, ffn_out],
        crate::numeric::NumericPolicy::bit_exact(),
    )
    .expect("the real layer binds");
    let ffn_out_body_steps = bound
        .iter()
        .find(|op| op.node == ffn_out)
        .expect("ffn_out is a bound op")
        .element_body()
        .steps
        .len();
    std::println!("ffn_out body_steps={ffn_out_body_steps}");

    let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");
    let wall_start = std::time::Instant::now();
    let evaluated =
        crate::cpu::evaluate_parallel(&program, &symbols, &blocks, &[root], workers)
            .expect("the real mistral layer evaluates");
    let wall = wall_start.elapsed();
    std::println!("wall_clock={wall:?}");

    let output = evaluated.root();
    assert_eq!(
        output.len(),
        SEQUENCE * EMBEDDING,
        "a vacuous output proves nothing"
    );
    assert!(
        output.iter().all(|value| value.is_finite()),
        "output must be finite"
    );
}

/// The evaluation half of the milestone above: `mistral_layer_small.toml`
/// is the same RoPE+GQA+causal-mask composition, small enough to
/// actually run (see the sibling test's doc comment and that file's
/// header for why). Two invariants, not just finiteness: the output is
/// the right shape and every value is finite, and every softmax row —
/// indexed explicitly because `probabilities`'s `(query, key, kv_head,
/// group_offset)` layout makes a key-axis row a strided read, not a
/// contiguous one, the same way `run_gqa_case` above handles it — sums
/// to 1.0.
#[test]
fn a_mistral_layer_written_as_toml_evaluates() {
    const SEQUENCE: usize = 4;
    const EMBEDDING: usize = 16;
    const QUERY_HEADS: usize = 8;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 4;
    const PAIRS: usize = HEAD_DIM / 2;
    const GROUP: usize = QUERY_HEADS / KV_HEADS;
    const FEED_FORWARD: usize = 32;

    let text = include_str!("../../specs/mistral_layer_small.toml");
    let spec: ProgramSpec = toml::from_str(text).expect("spec parses");
    spec.validate().expect("spec is structurally sound");
    let program = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");

    let symbols = [SEQUENCE as u64];
    crate::shape::infer(&program, &symbols).expect("the small layer infers");

    let activations = random_vec(101, SEQUENCE * EMBEDDING);
    let epsilon = alloc::vec![1e-5f32; SEQUENCE];
    let wq = random_vec(102, EMBEDDING * QUERY_HEADS * HEAD_DIM);
    let wk = random_vec(103, EMBEDDING * KV_HEADS * HEAD_DIM);
    let wv = random_vec(104, EMBEDDING * KV_HEADS * HEAD_DIM);
    let wo = random_vec(105, KV_HEADS * GROUP * HEAD_DIM * EMBEDDING);
    let w_gate = random_vec(106, EMBEDDING * FEED_FORWARD);
    let w_up = random_vec(107, EMBEDDING * FEED_FORWARD);
    let w_down = random_vec(108, FEED_FORWARD * EMBEDDING);
    let cos = random_vec(109, SEQUENCE * PAIRS);
    let sin = random_vec(110, SEQUENCE * PAIRS);

    let blocks: [&[f32]; 11] = [
        &activations,
        &epsilon,
        &wq,
        &wk,
        &wv,
        &wo,
        &w_gate,
        &w_up,
        &w_down,
        &cos,
        &sin,
    ];

    let probabilities = spec
        .node
        .iter()
        .position(|node| node.id() == "probabilities")
        .expect("the spec defines a probabilities node");
    let probabilities = NodeId(probabilities as u32);
    let root = NodeId(program.len() as u32 - 1);

    let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");
    let evaluated = crate::cpu::evaluate_parallel(
        &program,
        &symbols,
        &blocks,
        &[root, probabilities],
        workers,
    )
    .expect("the small mistral layer evaluates");

    let output = evaluated.root();
    assert_eq!(
        output.len(),
        SEQUENCE * EMBEDDING,
        "a vacuous output proves nothing"
    );
    assert!(
        output.iter().all(|value| value.is_finite()),
        "output must be finite"
    );

    let (rows, _) = evaluated
        .get(probabilities)
        .expect("probabilities were requested");
    assert_eq!(rows.len(), SEQUENCE * SEQUENCE * KV_HEADS * GROUP);

    // probabilities is laid out `(query, key, kv_head, group_offset)`
    // row-major, so a softmax "row" over the key axis is not a
    // contiguous slice — index it explicitly, the same way
    // `run_gqa_case` above does for the same `stug` iteration order.
    let mut checked = 0usize;
    for query_position in 0..SEQUENCE {
        for kv_head in 0..KV_HEADS {
            for offset in 0..GROUP {
                let mut total = 0.0f32;
                for key_position in 0..SEQUENCE {
                    let index = ((query_position * SEQUENCE + key_position) * KV_HEADS
                        + kv_head)
                        * GROUP
                        + offset;
                    total += rows[index];
                    checked += 1;
                }
                assert!(
                    (total - 1.0).abs() < 1e-4,
                    "query {query_position} kv-head {kv_head} group-offset {offset} \
                     softmax row sums to {total}, not 1.0"
                );
            }
        }
    }
    assert_eq!(
        checked,
        SEQUENCE * SEQUENCE * KV_HEADS * GROUP,
        "every probability cell must be checked, not a subset"
    );
}

/// The whole model, built as a program instead of authored as 32 copies
/// of one TOML file: token embedding lookup, `block_count` layers (each
/// [`append_mistral_layer`], mirroring `specs/mistral_layer.toml`), a
/// final RMSNorm, and the LM head projection to `[seq, vocab]` logits.
/// Shape inference is symbolic arithmetic over extents, not data — cheap
/// enough to run unignored even at the model's real context length,
/// matching `a_mistral_layer_written_as_toml_infers_at_its_real_dimensions`
/// above for one layer.
/// The contract the [`Op::Constant`] variant exists to hold: a literal
/// is a node, so the only names crossing the binding surface are data
/// (`ids`), model weights, position tables (`rope_cos`/`rope_sin`), and
/// the one piece of model metadata this function's `u32` parameters do
/// not carry (`eps`). `inv_dim`, `ones` and `group_ones` were bound
/// `Input`s that `proxima-model-interop`'s `bind.rs` filled with a
/// repeated scalar on every call; each was a name two files had to agree
/// on forever, which is the drift class this asserts is gone.
#[test]
fn no_repeated_scalar_crosses_the_binding_surface() {
    let program = mistral_forward_program(128, 64, 172, 8, 4, 16, 2, 0, 0)
        .expect("the forward pass lowers to a program");

    let bound: Vec<&str> = program
        .iter()
        .filter_map(|expr| match expr {
            Op::Input { .. } => expr.name(),
            _ => None,
        })
        .collect();

    for collapsed in [
        "inv_dim",
        "ones",
        "group_ones",
        "inv_sqrt_head_dim",
        "neg_infinity",
    ] {
        assert!(
            !bound.contains(&collapsed),
            "{collapsed} is a literal and must be an Op::Constant, not a bound Input; \
             bound names are {bound:?}"
        );
    }

    assert!(bound.contains(&"eps"), "eps is model metadata, still bound");
    assert!(bound.contains(&"ids"), "ids is per-call data, still bound");
    assert!(
        bound.contains(&"rope_cos"),
        "rope_cos varies with position, still bound"
    );
}

/// `op = "constant"` is the TOML face of [`Op::Constant`], and
/// `shape = []` is the rank-0 spelling every scalar literal uses.
#[test]
fn a_constant_node_reads_from_toml_with_its_literal_and_shape() {
    const TOML: &str = r#"
[[node]]
op = "constant"
id = "eps"
dtype = "float32"
shape = []
value = 1e-5

[[node]]
op = "constant"
id = "group_ones"
dtype = "float32"
shape = [4, 2]
value = 1.0
"#;
    let spec: ProgramSpec = toml::from_str(TOML).expect("constant nodes parse");
    let program = Vec::<Op>::try_from(&spec).expect("constant nodes lower");

    assert_eq!(
        program[0],
        Op::Constant {
            dtype: DType::Float32,
            shape: Vec::new(),
            value: 1e-5,
        }
    );
    assert_eq!(
        program[1],
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(4), Extent::Static(2)],
            value: 1.0,
        }
    );
}

#[test]
fn the_whole_mistral_forward_pass_infers_at_real_dimensions() {
    const REAL_CONTEXT: u64 = 8192;

    let build_start = std::time::Instant::now();
    let program = mistral_forward_program(32_002, 4096, 14336, 32, 8, 128, 32, 0, 0)
        .expect("the whole forward pass lowers to a program");
    let build_elapsed = build_start.elapsed();

    let infer_start = std::time::Instant::now();
    crate::shape::infer(&program, &[REAL_CONTEXT])
        .expect("the whole forward pass infers at its real context length");
    let infer_elapsed = infer_start.elapsed();

    std::println!(
        "mistral_forward_program: nodes={} build={build_elapsed:?} infer={infer_elapsed:?}",
        program.len()
    );
    assert!(
        program.len() > 2_000,
        "32 layers of dozens of nodes plus embedding/lm-head should be thousands of nodes, not {}",
        program.len()
    );
}

/// Node-count budget for the chunked key/value fold, established
/// BEFORE that fold is built. [`mistral_cached_forward_program`] binds
/// ONE cache buffer per layer sized by `Extent::Symbolic(1)`, so it is
/// already the one-chunk case of an N-chunk fold. Splitting the cache
/// into N fixed chunks replicates, per layer per chunk, the twelve
/// cache-reading nodes in `append_mistral_cached_layer`
/// (`score_cached_even_product`, `score_cached_even`,
/// `score_cached_odd_product`, `score_cached_odd`, `score_cached`,
/// `score_cached_scaled`, `score_max_cached`, `shifted_cached`,
/// `weights_cached`, `sum_cached`, `attended_cached_product`,
/// `attended_cached`), plus that chunk's own three `kv_cache.*`
/// `Op::Input` leaves, plus three combine nodes -- one `Maximum` into
/// `global_max`, one `Add` into `weight_sum`, one `Add` into
/// `attended_sum`. Eighteen nodes per chunk per layer.
///
/// The printed `per_chunk_per_model` figure is what a caller multiplies
/// by its chunk count to decide whether an N-chunk fold can be flat
/// graph nodes at all. It cannot, past a low N: the fold has to iterate
/// chunks inside one reduce rather than have the program name each one.
#[test]
fn the_chunked_cache_fold_node_budget_is_measured_before_it_is_built() {
    let nodes_of = |block_count: u32| {
        mistral_cached_forward_program(32_002, 4096, 14336, 32, 8, 128, block_count)
            .expect("the cached forward pass lowers to a program")
            .0
            .len()
    };
    let per_layer = nodes_of(2) - nodes_of(1);
    let full = nodes_of(32);
    let uncached = mistral_forward_program(32_002, 4096, 14336, 32, 8, 128, 32, 0, 0)
        .expect("the whole forward pass lowers to a program")
        .len();
    // a built `Op` is not an executed op: `crate::bind` fuses
    // elementwise chains, so the graph the evaluator walks is smaller
    // than the program. Both counts are printed because the chunk
    // budget is built in program nodes and paid in bound ops.
    let (cached_program, cached_logits, cached_roots) =
        mistral_cached_forward_program(32_002, 4096, 14336, 32, 8, 128, 32)
            .expect("the cached forward pass lowers to a program");
    let mut cached_outputs = alloc::vec![cached_logits];
    for (even, odd, value) in &cached_roots {
        cached_outputs.extend_from_slice(&[*even, *odd, *value]);
    }
    let cached_shapes = crate::shape::infer(&cached_program, &[1, 71])
        .expect("one new position against a 71-position cache infers");
    let bound = crate::bind::bind(
        &cached_program,
        &cached_shapes,
        &cached_outputs,
        crate::numeric::NumericPolicy::bit_exact(),
    )
    .expect("the cached program binds")
    .len();

    const CACHE_READING_NODES: usize = 12;
    const CACHE_INPUT_LEAVES: usize = 3;
    const COMBINE_NODES: usize = 3;
    const PER_CHUNK_PER_LAYER: usize = CACHE_READING_NODES + CACHE_INPUT_LEAVES + COMBINE_NODES;
    const LAYERS: usize = 32;

    std::println!(
        "cached_fold_budget uncached_nodes={uncached} cached_nodes={full} cached_per_layer={per_layer} bound_ops_at_ctx71={bound} reduces_per_chunk_per_layer=5 per_chunk_per_layer={PER_CHUNK_PER_LAYER} per_chunk_per_model={}",
        PER_CHUNK_PER_LAYER * LAYERS
    );
    for chunks in [1_usize, 4, 16, 64, 256, 1024, 4096] {
        std::println!(
            "cached_fold_budget chunks={chunks} context_at_chunk_256={} added_nodes={} total_nodes={}",
            chunks * 256,
            (chunks - 1) * PER_CHUNK_PER_LAYER * LAYERS,
            full + (chunks - 1) * PER_CHUNK_PER_LAYER * LAYERS
        );
    }

    assert!(
        PER_CHUNK_PER_LAYER < per_layer,
        "a chunk replicates only the cache-reading part of a layer, never the whole {per_layer}-node layer"
    );
}

/// [`the_chunked_cache_fold_node_budget_is_measured_before_it_is_built`]'s
/// single-range counterpart: [`append_mistral_single_range_cached_layer`]
/// deletes the eighteen cache-reading nodes that test documents
/// (`CACHE_READING_NODES=12` plus three combine nodes counted
/// separately there) and replaces them with nothing -- there is no
/// second block left to combine, so the eighteen-node-per-chunk cost
/// this budget exists to warn about does not apply to the single-range
/// path at all. Old->new, per layer: 83 raw `Op`s (baseline, matching
/// [`append_mistral_cached_layer`]'s own doc) -> whatever `per_layer`
/// prints below, deleting the 6-op cached score block
/// (`score_cached_even_product`..`score_cached_scaled`) and the
/// 17-op online-softmax combine (`score_max_cached`..`attended`),
/// adding back an 8-op single-pass softmax
/// (`score_max`,`shifted`,`weights`,`weight_sum`,`inv_weight_sum`,
/// `probabilities`,`attended_product`,`attended`) node-for-node
/// [`append_mistral_layer`]'s own pattern.
#[test]
fn the_single_range_cache_fold_node_budget_is_measured_against_the_two_range_baseline() {
    let two_range_nodes_of = |block_count: u32| {
        mistral_cached_forward_program(32_002, 4096, 14336, 32, 8, 128, block_count)
            .expect("the two-range cached forward pass lowers to a program")
            .0
            .len()
    };
    let single_range_nodes_of = |block_count: u32| {
        mistral_single_range_cached_forward_program(
            32_002,
            4096,
            14336,
            32,
            8,
            128,
            block_count,
            false,
            DuplicateHeadPosition::None,
            false,
        )
        .expect("the single-range cached forward pass lowers to a program")
        .0
        .len()
    };
    let two_range_per_layer = two_range_nodes_of(2) - two_range_nodes_of(1);
    let single_range_per_layer = single_range_nodes_of(2) - single_range_nodes_of(1);

    let (two_range_program, two_range_logits, two_range_roots) =
        mistral_cached_forward_program(32_002, 4096, 14336, 32, 8, 128, 32)
            .expect("the two-range cached forward pass lowers to a program");
    let mut two_range_outputs = alloc::vec![two_range_logits];
    for (even, odd, value) in &two_range_roots {
        two_range_outputs.extend_from_slice(&[*even, *odd, *value]);
    }
    let two_range_shapes = crate::shape::infer(&two_range_program, &[1, 71])
        .expect("one new position against a 71-position cache infers");
    // fusion held explicitly off: `bind`'s default `fuse_cached_attention
    // = true` (under `cached-attention-streaming`) collapses the
    // two-range baseline's online-softmax combine into `CachedAttention`
    // BoundOps but has no candidate to fuse on the single-range path
    // below, so an unpinned `bind` call here compares a fused count
    // against an unfused one instead of the structural raw-bind
    // difference this test names.
    let two_range_bound = crate::bind::bind_with_fusion(
        &two_range_program,
        &two_range_shapes,
        &two_range_outputs,
        false,
        crate::numeric::NumericPolicy::default(),
    )
    .expect("the two-range cached program binds")
    .len();

    let (single_range_program, single_range_logits, single_range_roots, _) =
        mistral_single_range_cached_forward_program(
            32_002,
            4096,
            14336,
            32,
            8,
            128,
            32,
            false,
            DuplicateHeadPosition::None,
            false,
        )
        .expect("the single-range cached forward pass lowers to a program");
    let mut single_range_outputs = alloc::vec![single_range_logits];
    for (even, odd, value) in &single_range_roots {
        single_range_outputs.extend_from_slice(&[*even, *odd, *value]);
    }
    // symbol 1 here is the MERGED length -- 71 total context, matching
    // the two-range baseline's 71-position existing cache plus its own
    // one new position folded in, so both measurements are read at the
    // same total context depth.
    let single_range_shapes = crate::shape::infer(&single_range_program, &[1, 71])
        .expect("one new position against a 71-position merged range infers");
    let single_range_bound = crate::bind::bind_with_fusion(
        &single_range_program,
        &single_range_shapes,
        &single_range_outputs,
        false,
        crate::numeric::NumericPolicy::default(),
    )
    .expect("the single-range cached program binds")
    .len();

    std::println!(
        "single_range_vs_two_range raw_ops_per_layer: before={two_range_per_layer} after={single_range_per_layer} bound_ops_at_ctx71: before={two_range_bound} after={single_range_bound}"
    );

    assert!(
        single_range_per_layer < two_range_per_layer,
        "single-range must emit fewer raw ops per layer than the two-range baseline: before={two_range_per_layer} after={single_range_per_layer}"
    );
    assert!(
        single_range_bound < two_range_bound,
        "single-range must bind fewer ops at ctx71 than the two-range baseline: before={two_range_bound} after={single_range_bound}"
    );
}

/// The falsifiable claim under test: for the SAME weights and the SAME
/// `cached_len`, [`append_mistral_single_range_cached_layer`] must
/// produce the SAME decode-step logits [`append_mistral_cached_layer`]'s
/// own two-range online-softmax combine produces -- PROVIDED its cache
/// input holds what write-placement (`proxima-wt-place`'s
/// `execute_plan_with_placements`) actually hands it at runtime: the
/// `cached_len` prior positions PLUS this call's OWN rotated K/V
/// appended at the tail, sized `cached_len + new_count`. This is not a
/// calling-convention change to the single-range graph -- it is the
/// single-range graph's documented contract
/// (`append_mistral_single_range_cached_layer`'s own doc: "the WHOLE
/// merged context this call attends to ... already folded in by the
/// caller between calls"). `cpu::evaluate` has no write-placement, so
/// this test builds that merged cache by hand: run the two-range oracle
/// first, read this call's own `CachedLayerRoots` back out of its
/// `Evaluated`, and concatenate them onto the prior cache before
/// evaluating the single-range arm -- exactly what write-placement
/// would have left resident. Error is normalized against the two-range
/// oracle's own BATCH PEAK magnitude (never per-row: a per-row relative
/// error explodes at zero crossings and produced a false 872% "bug"
/// report on this codebase before). `cached_len = 0` is covered first
/// because it is the case most likely to be silently wrong -- with no
/// prior cache the merged range is exactly this call's own new key(s),
/// so a query must attend only itself.
type PerLayerCacheColumns = (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>);

#[test]
fn a_single_range_decode_step_matches_the_two_range_decode_step() {
    const VOCAB: usize = 5;
    const EMBEDDING: usize = 4;
    const FEED_FORWARD: usize = 4;
    const QUERY_HEADS: usize = 2;
    const KV_HEADS: usize = 1;
    const HEAD_DIM: usize = 2;
    const PAIRS: usize = HEAD_DIM / 2;
    const GROUP: usize = QUERY_HEADS / KV_HEADS;
    const BLOCK_COUNT: u32 = 2;

    struct LayerWeights {
        attn_norm: Vec<f32>,
        ffn_norm: Vec<f32>,
        wq: Vec<f32>,
        wk: Vec<f32>,
        wv: Vec<f32>,
        wo: Vec<f32>,
        w_gate: Vec<f32>,
        w_up: Vec<f32>,
        w_down: Vec<f32>,
    }

    fn max_error_at(cached_len: usize, new_count: usize) -> (f32, f32) {
        let sequence = cached_len + new_count;
        let ids: Vec<u32> = (0..sequence as u32).map(|id| 1 + id % 3).collect();
        let ids_f32: Vec<f32> = ids.iter().map(|&id| id as f32).collect();

        let table = random_vec(10, VOCAB * EMBEDDING);
        let eps_cached = alloc::vec![1e-5f32; cached_len.max(1)];
        let eps_new = alloc::vec![1e-5f32; new_count];
        let (cos_cached, sin_cached) = rope_angles(0, cached_len.max(1), PAIRS, HEAD_DIM);
        let (cos_new, sin_new) = rope_angles(cached_len, new_count, PAIRS, HEAD_DIM);

        let mut layers = Vec::new();
        let mut seed = 200u64;
        for _ in 0..BLOCK_COUNT {
            layers.push(LayerWeights {
                attn_norm: alloc::vec![1.0f32; EMBEDDING],
                ffn_norm: alloc::vec![1.0f32; EMBEDDING],
                wq: random_vec(seed, EMBEDDING * QUERY_HEADS * HEAD_DIM),
                wk: random_vec(seed + 1, EMBEDDING * KV_HEADS * HEAD_DIM),
                wv: random_vec(seed + 2, EMBEDDING * KV_HEADS * HEAD_DIM),
                wo: random_vec(seed + 3, KV_HEADS * GROUP * HEAD_DIM * EMBEDDING),
                w_gate: random_vec(seed + 4, EMBEDDING * FEED_FORWARD),
                w_up: random_vec(seed + 5, EMBEDDING * FEED_FORWARD),
                w_down: random_vec(seed + 6, FEED_FORWARD * EMBEDDING),
            });
            seed += 7;
        }
        let output_norm = alloc::vec![1.0f32; EMBEDDING];
        let lm_head = random_vec(seed, EMBEDDING * VOCAB);

        let layer_names: Vec<[alloc::string::String; 9]> = (0..BLOCK_COUNT as usize)
            .map(|layer| {
                [
                    alloc::format!("blk.{layer}.attn_norm.weight"),
                    alloc::format!("blk.{layer}.ffn_norm.weight"),
                    alloc::format!("blk.{layer}.attn_q.weight"),
                    alloc::format!("blk.{layer}.attn_k.weight"),
                    alloc::format!("blk.{layer}.attn_v.weight"),
                    alloc::format!("blk.{layer}.attn_output.weight"),
                    alloc::format!("blk.{layer}.ffn_gate.weight"),
                    alloc::format!("blk.{layer}.ffn_up.weight"),
                    alloc::format!("blk.{layer}.ffn_down.weight"),
                ]
            })
            .collect();
        let kv_cache_names: Vec<[alloc::string::String; 3]> = (0..BLOCK_COUNT as usize)
            .map(|layer| {
                [
                    alloc::format!("kv_cache.{layer}.k_even"),
                    alloc::format!("kv_cache.{layer}.k_odd"),
                    alloc::format!("kv_cache.{layer}.v"),
                ]
            })
            .collect();

        let mut common_named: Vec<(&str, &[f32])> =
            alloc::vec![("token_embd.weight", table.as_slice())];
        for (layer_index, weights) in layers.iter().enumerate() {
            let names = &layer_names[layer_index];
            common_named.push((names[0].as_str(), weights.attn_norm.as_slice()));
            common_named.push((names[1].as_str(), weights.ffn_norm.as_slice()));
            common_named.push((names[2].as_str(), weights.wq.as_slice()));
            common_named.push((names[3].as_str(), weights.wk.as_slice()));
            common_named.push((names[4].as_str(), weights.wv.as_slice()));
            common_named.push((names[5].as_str(), weights.wo.as_slice()));
            common_named.push((names[6].as_str(), weights.w_gate.as_slice()));
            common_named.push((names[7].as_str(), weights.w_up.as_slice()));
            common_named.push((names[8].as_str(), weights.w_down.as_slice()));
        }
        common_named.push(("output_norm.weight", output_norm.as_slice()));
        common_named.push(("output.weight", lm_head.as_slice()));

        // -- fold the cache up to `cached_len` via the two-range
        // program's own prefill path, the same mechanism
        // `a_cached_decode_step_matches_the_uncached_forward_pass_exactly`
        // already trusts.
        let (cached_program, _, cache_roots) = mistral_cached_forward_program(
            VOCAB as u32,
            EMBEDDING as u32,
            FEED_FORWARD as u32,
            QUERY_HEADS as u32,
            KV_HEADS as u32,
            HEAD_DIM as u32,
            BLOCK_COUNT,
        )
        .expect("cached forward pass lowers");

        let (k_even_cache, k_odd_cache, v_cache): PerLayerCacheColumns = if cached_len == 0 {
            (
                alloc::vec![Vec::new(); BLOCK_COUNT as usize],
                alloc::vec![Vec::new(); BLOCK_COUNT as usize],
                alloc::vec![Vec::new(); BLOCK_COUNT as usize],
            )
        } else {
            let prefill_cached_len_value = [0.0f32];
            let mut prefill_named = common_named.clone();
            prefill_named.push(("ids", &ids_f32[..cached_len]));
            prefill_named.push(("eps", eps_cached.as_slice()));
            prefill_named.push(("rope_cos", cos_cached.as_slice()));
            prefill_named.push(("rope_sin", sin_cached.as_slice()));
            prefill_named.push(("cached_len", prefill_cached_len_value.as_slice()));
            let empty = Vec::<f32>::new();
            for names in &kv_cache_names {
                prefill_named.push((names[0].as_str(), empty.as_slice()));
                prefill_named.push((names[1].as_str(), empty.as_slice()));
                prefill_named.push((names[2].as_str(), empty.as_slice()));
            }
            let mut prefill_roots = Vec::new();
            for (even, odd, value) in &cache_roots {
                prefill_roots.push(*even);
                prefill_roots.push(*odd);
                prefill_roots.push(*value);
            }
            let prefill_symbols = [cached_len as u64, 0u64];
            let prefill_evaluated = crate::cpu::evaluate_named(
                &cached_program,
                &prefill_symbols,
                &prefill_named,
                &prefill_roots,
            )
            .expect("prefill call evaluates");
            let mut even_out = Vec::with_capacity(BLOCK_COUNT as usize);
            let mut odd_out = Vec::with_capacity(BLOCK_COUNT as usize);
            let mut value_out = Vec::with_capacity(BLOCK_COUNT as usize);
            for (even, odd, value) in &cache_roots {
                even_out.push(prefill_evaluated.get(*even).expect("k_even").0.to_vec());
                odd_out.push(prefill_evaluated.get(*odd).expect("k_odd").0.to_vec());
                value_out.push(prefill_evaluated.get(*value).expect("v").0.to_vec());
            }
            (even_out, odd_out, value_out)
        };

        // -- two-range decode step: the trusted incumbent.
        let two_range_cached_len_value = [cached_len as f32];
        let mut two_range_named = common_named.clone();
        two_range_named.push(("ids", &ids_f32[cached_len..]));
        two_range_named.push(("eps", eps_new.as_slice()));
        two_range_named.push(("rope_cos", cos_new.as_slice()));
        two_range_named.push(("rope_sin", sin_new.as_slice()));
        two_range_named.push(("cached_len", two_range_cached_len_value.as_slice()));
        for (layer_index, names) in kv_cache_names.iter().enumerate() {
            two_range_named.push((names[0].as_str(), k_even_cache[layer_index].as_slice()));
            two_range_named.push((names[1].as_str(), k_odd_cache[layer_index].as_slice()));
            two_range_named.push((names[2].as_str(), v_cache[layer_index].as_slice()));
        }
        let two_range_root = NodeId(cached_program.len() as u32 - 1);
        let two_range_symbols = [new_count as u64, cached_len as u64];
        let mut two_range_roots: Vec<NodeId> = Vec::with_capacity(cache_roots.len() * 3 + 1);
        for (even, odd, value) in &cache_roots {
            two_range_roots.push(*even);
            two_range_roots.push(*odd);
            two_range_roots.push(*value);
        }
        two_range_roots.push(two_range_root);
        let two_range_evaluated = crate::cpu::evaluate_named(
            &cached_program,
            &two_range_symbols,
            &two_range_named,
            &two_range_roots,
        )
        .expect("two-range decode call evaluates");
        let (two_range_logits, two_range_shape) = two_range_evaluated
            .get(two_range_root)
            .expect("two-range logits present");
        assert_eq!(two_range_shape, [new_count as u64, VOCAB as u64]);

        // -- this decode call's own rotated K/V, per layer: exactly
        // what write-placement would leave resident at the cache's
        // tail for the NEXT call. Concatenated onto the prior cache
        // below to build the single-range arm's merged input.
        let mut merged_k_even_cache = k_even_cache.clone();
        let mut merged_k_odd_cache = k_odd_cache.clone();
        let mut merged_v_cache = v_cache.clone();
        for (layer_index, (even, odd, value)) in cache_roots.iter().enumerate() {
            let new_even = two_range_evaluated.get(*even).expect("k_new_even").0;
            let new_odd = two_range_evaluated.get(*odd).expect("k_new_odd").0;
            let new_value = two_range_evaluated.get(*value).expect("v_new").0;
            merged_k_even_cache[layer_index].extend_from_slice(new_even);
            merged_k_odd_cache[layer_index].extend_from_slice(new_odd);
            merged_v_cache[layer_index].extend_from_slice(new_value);
        }

        // -- single-range decode step: the graph under test, fed the
        // MERGED cache (prior positions plus this call's own, folded
        // in by hand the way write-placement would fold them in at
        // runtime).
        let (single_range_program, single_range_root, _, _) =
            mistral_single_range_cached_forward_program(
                VOCAB as u32,
                EMBEDDING as u32,
                FEED_FORWARD as u32,
                QUERY_HEADS as u32,
                KV_HEADS as u32,
                HEAD_DIM as u32,
                BLOCK_COUNT,
                false,
                DuplicateHeadPosition::None,
                false,
            )
            .expect("single-range cached forward pass lowers");
        let cached_len_scalar = alloc::vec![cached_len as f32];
        let mut single_range_named = common_named.clone();
        single_range_named.push(("ids", &ids_f32[cached_len..]));
        single_range_named.push(("eps", eps_new.as_slice()));
        single_range_named.push(("rope_cos", cos_new.as_slice()));
        single_range_named.push(("rope_sin", sin_new.as_slice()));
        single_range_named.push(("cached_len", cached_len_scalar.as_slice()));
        for (layer_index, names) in kv_cache_names.iter().enumerate() {
            single_range_named.push((
                names[0].as_str(),
                merged_k_even_cache[layer_index].as_slice(),
            ));
            single_range_named.push((
                names[1].as_str(),
                merged_k_odd_cache[layer_index].as_slice(),
            ));
            single_range_named
                .push((names[2].as_str(), merged_v_cache[layer_index].as_slice()));
        }
        let single_range_symbols = [new_count as u64, sequence as u64];
        let single_range_evaluated = crate::cpu::evaluate_named(
            &single_range_program,
            &single_range_symbols,
            &single_range_named,
            &[single_range_root],
        )
        .expect("single-range decode call evaluates");
        let (single_range_logits, single_range_shape) = single_range_evaluated
            .get(single_range_root)
            .expect("single-range logits present");
        assert_eq!(single_range_shape, [new_count as u64, VOCAB as u64]);

        let batch_peak = two_range_logits
            .iter()
            .fold(0.0f32, |peak, value| peak.max(value.abs()));
        let max_error = two_range_logits
            .iter()
            .zip(single_range_logits.iter())
            .map(|(oracle, candidate)| (oracle - candidate).abs())
            .fold(0.0f32, f32::max);
        let normalized_error = if batch_peak > 0.0 {
            max_error / batch_peak
        } else {
            max_error
        };
        std::println!(
            "single_range_vs_two_range_decode cached_len={cached_len} new_count={new_count} batch_peak={batch_peak} max_error={max_error} normalized_error={normalized_error} two_range={two_range_logits:?} single_range={single_range_logits:?}"
        );
        (max_error, normalized_error)
    }

    let cases = [(0usize, 1usize), (1usize, 1usize), (6usize, 2usize)];
    let results: Vec<((usize, usize), (f32, f32))> = cases
        .iter()
        .map(|&(cached_len, new_count)| {
            ((cached_len, new_count), max_error_at(cached_len, new_count))
        })
        .collect();
    for (cached_len, new_count) in cases {
        let (max_error, normalized_error) = results
            .iter()
            .find(|(case, _)| *case == (cached_len, new_count))
            .expect("case present")
            .1;
        assert!(
            normalized_error < 1e-4,
            "single-range decode diverged from the two-range decode at cached_len={cached_len} new_count={new_count}: max_error={max_error} normalized_error={normalized_error}"
        );
    }
}

/// ROW 373's own parity check: [`a_single_range_decode_step_matches_the_two_range_decode_step`]'s
/// exact harness, `qk_norm` flipped on for both arms
/// ([`qwen3_cached_forward_program`] as the two-range oracle,
/// [`mistral_single_range_cached_forward_program`]'s `qk_norm: true` as
/// the candidate) and `attn_q_norm.weight`/`attn_k_norm.weight` (random,
/// non-degenerate, so a wrong gamma or a missing normalization is
/// visible) added per layer. Split-half RoPE is exercised by
/// construction -- `qk_norm.is_some()` selects it in both builders, see
/// `append_mistral_cached_layer`'s and
/// `append_mistral_single_range_cached_layer`'s own doc on that rule.
/// Same normalized-error tolerance as the plain arm: the online-softmax
/// combine's own op ordering (two partial reduces, elementwise-summed)
/// vs the single-range one-shot reduce is not required to be 0-ULP, only
/// numerically equivalent -- `a_single_range_decode_step_matches_the_two_range_decode_step`'s
/// own doc already established `< 1e-4` as this codebase's bar for that
/// distinction.
#[test]
fn a_single_range_decode_step_with_qk_norm_matches_the_two_range_decode_step() {
    const VOCAB: usize = 5;
    const EMBEDDING: usize = 4;
    const FEED_FORWARD: usize = 4;
    const QUERY_HEADS: usize = 2;
    const KV_HEADS: usize = 1;
    const HEAD_DIM: usize = 2;
    const PAIRS: usize = HEAD_DIM / 2;
    const GROUP: usize = QUERY_HEADS / KV_HEADS;
    const BLOCK_COUNT: u32 = 2;

    struct LayerWeights {
        attn_norm: Vec<f32>,
        ffn_norm: Vec<f32>,
        wq: Vec<f32>,
        wk: Vec<f32>,
        wv: Vec<f32>,
        wo: Vec<f32>,
        w_gate: Vec<f32>,
        w_up: Vec<f32>,
        w_down: Vec<f32>,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
    }

    fn max_error_at(cached_len: usize, new_count: usize) -> (f32, f32) {
        let sequence = cached_len + new_count;
        let ids: Vec<u32> = (0..sequence as u32).map(|id| 1 + id % 3).collect();
        let ids_f32: Vec<f32> = ids.iter().map(|&id| id as f32).collect();

        let table = random_vec(10, VOCAB * EMBEDDING);
        let eps_cached = alloc::vec![1e-5f32; cached_len.max(1)];
        let eps_new = alloc::vec![1e-5f32; new_count];
        let (cos_cached, sin_cached) = rope_angles(0, cached_len.max(1), PAIRS, HEAD_DIM);
        let (cos_new, sin_new) = rope_angles(cached_len, new_count, PAIRS, HEAD_DIM);

        let mut layers = Vec::new();
        let mut seed = 300u64;
        for _ in 0..BLOCK_COUNT {
            layers.push(LayerWeights {
                attn_norm: alloc::vec![1.0f32; EMBEDDING],
                ffn_norm: alloc::vec![1.0f32; EMBEDDING],
                wq: random_vec(seed, EMBEDDING * QUERY_HEADS * HEAD_DIM),
                wk: random_vec(seed + 1, EMBEDDING * KV_HEADS * HEAD_DIM),
                wv: random_vec(seed + 2, EMBEDDING * KV_HEADS * HEAD_DIM),
                wo: random_vec(seed + 3, KV_HEADS * GROUP * HEAD_DIM * EMBEDDING),
                w_gate: random_vec(seed + 4, EMBEDDING * FEED_FORWARD),
                w_up: random_vec(seed + 5, EMBEDDING * FEED_FORWARD),
                w_down: random_vec(seed + 6, FEED_FORWARD * EMBEDDING),
                q_norm: random_vec(seed + 7, HEAD_DIM),
                k_norm: random_vec(seed + 8, HEAD_DIM),
            });
            seed += 9;
        }
        let output_norm = alloc::vec![1.0f32; EMBEDDING];
        let lm_head = random_vec(seed, EMBEDDING * VOCAB);

        let layer_names: Vec<[alloc::string::String; 11]> = (0..BLOCK_COUNT as usize)
            .map(|layer| {
                [
                    alloc::format!("blk.{layer}.attn_norm.weight"),
                    alloc::format!("blk.{layer}.ffn_norm.weight"),
                    alloc::format!("blk.{layer}.attn_q.weight"),
                    alloc::format!("blk.{layer}.attn_k.weight"),
                    alloc::format!("blk.{layer}.attn_v.weight"),
                    alloc::format!("blk.{layer}.attn_output.weight"),
                    alloc::format!("blk.{layer}.ffn_gate.weight"),
                    alloc::format!("blk.{layer}.ffn_up.weight"),
                    alloc::format!("blk.{layer}.ffn_down.weight"),
                    alloc::format!("blk.{layer}.attn_q_norm.weight"),
                    alloc::format!("blk.{layer}.attn_k_norm.weight"),
                ]
            })
            .collect();
        let kv_cache_names: Vec<[alloc::string::String; 3]> = (0..BLOCK_COUNT as usize)
            .map(|layer| {
                [
                    alloc::format!("kv_cache.{layer}.k_even"),
                    alloc::format!("kv_cache.{layer}.k_odd"),
                    alloc::format!("kv_cache.{layer}.v"),
                ]
            })
            .collect();

        let mut common_named: Vec<(&str, &[f32])> =
            alloc::vec![("token_embd.weight", table.as_slice())];
        for (layer_index, weights) in layers.iter().enumerate() {
            let names = &layer_names[layer_index];
            common_named.push((names[0].as_str(), weights.attn_norm.as_slice()));
            common_named.push((names[1].as_str(), weights.ffn_norm.as_slice()));
            common_named.push((names[2].as_str(), weights.wq.as_slice()));
            common_named.push((names[3].as_str(), weights.wk.as_slice()));
            common_named.push((names[4].as_str(), weights.wv.as_slice()));
            common_named.push((names[5].as_str(), weights.wo.as_slice()));
            common_named.push((names[6].as_str(), weights.w_gate.as_slice()));
            common_named.push((names[7].as_str(), weights.w_up.as_slice()));
            common_named.push((names[8].as_str(), weights.w_down.as_slice()));
            common_named.push((names[9].as_str(), weights.q_norm.as_slice()));
            common_named.push((names[10].as_str(), weights.k_norm.as_slice()));
        }
        common_named.push(("output_norm.weight", output_norm.as_slice()));
        common_named.push(("output.weight", lm_head.as_slice()));

        // -- fold the cache up to `cached_len` via the two-range qk-norm
        // program's own prefill path, same mechanism the plain-layer
        // parity test trusts.
        let (cached_program, _, cache_roots) = qwen3_cached_forward_program(
            VOCAB as u32,
            EMBEDDING as u32,
            FEED_FORWARD as u32,
            QUERY_HEADS as u32,
            KV_HEADS as u32,
            HEAD_DIM as u32,
            BLOCK_COUNT,
        )
        .expect("qk-norm cached forward pass lowers");

        let (k_even_cache, k_odd_cache, v_cache): PerLayerCacheColumns = if cached_len == 0 {
            (
                alloc::vec![Vec::new(); BLOCK_COUNT as usize],
                alloc::vec![Vec::new(); BLOCK_COUNT as usize],
                alloc::vec![Vec::new(); BLOCK_COUNT as usize],
            )
        } else {
            let prefill_cached_len_value = [0.0f32];
            let mut prefill_named = common_named.clone();
            prefill_named.push(("ids", &ids_f32[..cached_len]));
            prefill_named.push(("eps", eps_cached.as_slice()));
            prefill_named.push(("rope_cos", cos_cached.as_slice()));
            prefill_named.push(("rope_sin", sin_cached.as_slice()));
            prefill_named.push(("cached_len", prefill_cached_len_value.as_slice()));
            let empty = Vec::<f32>::new();
            for names in &kv_cache_names {
                prefill_named.push((names[0].as_str(), empty.as_slice()));
                prefill_named.push((names[1].as_str(), empty.as_slice()));
                prefill_named.push((names[2].as_str(), empty.as_slice()));
            }
            let mut prefill_roots = Vec::new();
            for (even, odd, value) in &cache_roots {
                prefill_roots.push(*even);
                prefill_roots.push(*odd);
                prefill_roots.push(*value);
            }
            let prefill_symbols = [cached_len as u64, 0u64];
            let prefill_evaluated = crate::cpu::evaluate_named(
                &cached_program,
                &prefill_symbols,
                &prefill_named,
                &prefill_roots,
            )
            .expect("prefill call evaluates");
            let mut even_out = Vec::with_capacity(BLOCK_COUNT as usize);
            let mut odd_out = Vec::with_capacity(BLOCK_COUNT as usize);
            let mut value_out = Vec::with_capacity(BLOCK_COUNT as usize);
            for (even, odd, value) in &cache_roots {
                even_out.push(prefill_evaluated.get(*even).expect("k_even").0.to_vec());
                odd_out.push(prefill_evaluated.get(*odd).expect("k_odd").0.to_vec());
                value_out.push(prefill_evaluated.get(*value).expect("v").0.to_vec());
            }
            (even_out, odd_out, value_out)
        };

        // -- two-range decode step: the trusted incumbent.
        let two_range_cached_len_value = [cached_len as f32];
        let mut two_range_named = common_named.clone();
        two_range_named.push(("ids", &ids_f32[cached_len..]));
        two_range_named.push(("eps", eps_new.as_slice()));
        two_range_named.push(("rope_cos", cos_new.as_slice()));
        two_range_named.push(("rope_sin", sin_new.as_slice()));
        two_range_named.push(("cached_len", two_range_cached_len_value.as_slice()));
        for (layer_index, names) in kv_cache_names.iter().enumerate() {
            two_range_named.push((names[0].as_str(), k_even_cache[layer_index].as_slice()));
            two_range_named.push((names[1].as_str(), k_odd_cache[layer_index].as_slice()));
            two_range_named.push((names[2].as_str(), v_cache[layer_index].as_slice()));
        }
        let two_range_root = NodeId(cached_program.len() as u32 - 1);
        let two_range_symbols = [new_count as u64, cached_len as u64];
        let mut two_range_roots: Vec<NodeId> = Vec::with_capacity(cache_roots.len() * 3 + 1);
        for (even, odd, value) in &cache_roots {
            two_range_roots.push(*even);
            two_range_roots.push(*odd);
            two_range_roots.push(*value);
        }
        two_range_roots.push(two_range_root);
        let two_range_evaluated = crate::cpu::evaluate_named(
            &cached_program,
            &two_range_symbols,
            &two_range_named,
            &two_range_roots,
        )
        .expect("two-range decode call evaluates");
        let (two_range_logits, two_range_shape) = two_range_evaluated
            .get(two_range_root)
            .expect("two-range logits present");
        assert_eq!(two_range_shape, [new_count as u64, VOCAB as u64]);

        // -- this decode call's own rotated K/V, per layer: exactly
        // what write-placement would leave resident at the cache's
        // tail for the NEXT call. Concatenated onto the prior cache
        // below to build the single-range arm's merged input.
        let mut merged_k_even_cache = k_even_cache.clone();
        let mut merged_k_odd_cache = k_odd_cache.clone();
        let mut merged_v_cache = v_cache.clone();
        for (layer_index, (even, odd, value)) in cache_roots.iter().enumerate() {
            let new_even = two_range_evaluated.get(*even).expect("k_new_even").0;
            let new_odd = two_range_evaluated.get(*odd).expect("k_new_odd").0;
            let new_value = two_range_evaluated.get(*value).expect("v_new").0;
            merged_k_even_cache[layer_index].extend_from_slice(new_even);
            merged_k_odd_cache[layer_index].extend_from_slice(new_odd);
            merged_v_cache[layer_index].extend_from_slice(new_value);
        }

        // -- single-range decode step: the graph under test, `qk_norm:
        // true`, fed the MERGED cache.
        let (single_range_program, single_range_root, _, _) =
            mistral_single_range_cached_forward_program(
                VOCAB as u32,
                EMBEDDING as u32,
                FEED_FORWARD as u32,
                QUERY_HEADS as u32,
                KV_HEADS as u32,
                HEAD_DIM as u32,
                BLOCK_COUNT,
                true,
                DuplicateHeadPosition::None,
                false,
            )
            .expect("single-range qk-norm cached forward pass lowers");
        let cached_len_scalar = alloc::vec![cached_len as f32];
        let mut single_range_named = common_named.clone();
        single_range_named.push(("ids", &ids_f32[cached_len..]));
        single_range_named.push(("eps", eps_new.as_slice()));
        single_range_named.push(("rope_cos", cos_new.as_slice()));
        single_range_named.push(("rope_sin", sin_new.as_slice()));
        single_range_named.push(("cached_len", cached_len_scalar.as_slice()));
        for (layer_index, names) in kv_cache_names.iter().enumerate() {
            single_range_named.push((
                names[0].as_str(),
                merged_k_even_cache[layer_index].as_slice(),
            ));
            single_range_named.push((
                names[1].as_str(),
                merged_k_odd_cache[layer_index].as_slice(),
            ));
            single_range_named
                .push((names[2].as_str(), merged_v_cache[layer_index].as_slice()));
        }
        let single_range_symbols = [new_count as u64, sequence as u64];
        let single_range_evaluated = crate::cpu::evaluate_named(
            &single_range_program,
            &single_range_symbols,
            &single_range_named,
            &[single_range_root],
        )
        .expect("single-range decode call evaluates");
        let (single_range_logits, single_range_shape) = single_range_evaluated
            .get(single_range_root)
            .expect("single-range logits present");
        assert_eq!(single_range_shape, [new_count as u64, VOCAB as u64]);

        let batch_peak = two_range_logits
            .iter()
            .fold(0.0f32, |peak, value| peak.max(value.abs()));
        let max_error = two_range_logits
            .iter()
            .zip(single_range_logits.iter())
            .map(|(oracle, candidate)| (oracle - candidate).abs())
            .fold(0.0f32, f32::max);
        let normalized_error = if batch_peak > 0.0 {
            max_error / batch_peak
        } else {
            max_error
        };
        std::println!(
            "single_range_vs_two_range_decode_qk_norm cached_len={cached_len} new_count={new_count} batch_peak={batch_peak} max_error={max_error} normalized_error={normalized_error} two_range={two_range_logits:?} single_range={single_range_logits:?}"
        );
        (max_error, normalized_error)
    }

    let cases = [(0usize, 1usize), (1usize, 1usize), (6usize, 2usize)];
    let results: Vec<((usize, usize), (f32, f32))> = cases
        .iter()
        .map(|&(cached_len, new_count)| {
            ((cached_len, new_count), max_error_at(cached_len, new_count))
        })
        .collect();
    for (cached_len, new_count) in cases {
        let (max_error, normalized_error) = results
            .iter()
            .find(|(case, _)| *case == (cached_len, new_count))
            .expect("case present")
            .1;
        assert!(
            normalized_error < 1e-4,
            "single-range qk-norm decode diverged from the two-range qk-norm decode at cached_len={cached_len} new_count={new_count}: max_error={max_error} normalized_error={normalized_error}"
        );
    }
}

/// Direct A/B on the SAME single-range program under the SAME data:
/// `crate::bind::bind_with_fusion(.., true)` (fires
/// [`cached_attention_single_range_candidates`], one
/// `BoundOpKind::CachedAttention` per layer) against `bind_with_fusion(..,
/// false)` (the literal `score_even`/`score_odd`/mask/softmax/`attended`
/// chain [`crate::cpu::run_node_into`] would otherwise run node-for-node).
/// [`a_single_range_decode_step_matches_the_two_range_decode_step`]
/// already proves the fused kind agrees with the two-range oracle on
/// real attention math; this test isolates the rewrite itself —
/// same program, same weights, same cache, fused vs not — so a
/// divergence here can only be the fusion transform, never a
/// two-range-specific difference.
#[test]
#[cfg(feature = "cached-attention-streaming")]
fn cached_attention_single_range_fused_matches_the_unfused_program() {
    use core::pin::pin;
    use core::task::{Context, Poll, Waker};

    use crate::bind::{
        BoundOp, BoundOpKind, READY_BATCH_CAPACITY, ReadyBatch, bind_with_fusion,
        block_node_ids,
    };
    use crate::cpu::Interpreter;
    use crate::numeric::NumericPolicy;
    use proxima_primitives::pipe::Pipe;

    const VOCAB: u32 = 5;
    const EMBEDDING: u32 = 4;
    const FEED_FORWARD: u32 = 4;
    const QUERY_HEADS: u32 = 2;
    const KV_HEADS: u32 = 1;
    const HEAD_DIM: u32 = 2;
    const BLOCK_COUNT: u32 = 2;
    const CACHED_LEN: usize = 3;
    const NEW_COUNT: usize = 2;
    const MERGED_LEN: usize = CACHED_LEN + NEW_COUNT;

    fn run_resolved(
        program_len: usize,
        resolved: &[BoundOp],
        inputs: Vec<(NodeId, Vec<f32>)>,
    ) -> Vec<Option<Vec<f32>>> {
        let mut buffers: Vec<Option<Vec<f32>>> = alloc::vec![None; program_len];
        for (node, data) in inputs {
            buffers[node.0 as usize] = Some(data);
        }
        let interpreter = Interpreter::new(&mut buffers);
        for chunk in resolved.chunks(READY_BATCH_CAPACITY) {
            let batch: ReadyBatch = chunk.iter().cloned().collect();
            let waker = Waker::noop();
            let mut context = Context::from_waker(waker);
            let mut future = pin!(interpreter.call(batch));
            match future.as_mut().poll(&mut context) {
                Poll::Ready(result) => result.expect("resolved batch computes"),
                Poll::Pending => unreachable!("cpu pipes never yield: no internal .await"),
            }
        }
        buffers
    }

    // `bucket_padding` reproduces `kv-capacity-bucket`'s KV extent
    // rounding directly on this fixture: the KV leaves grow by
    // `bucket_padding` zero-filled rows past `MERGED_LEN` (exactly what
    // `[merged_len, bucket)` looks like at runtime) while `cached_len`'s
    // own `Op::Input` value never moves, so any divergence this
    // introduces is the fused op reading the wrong band, never a
    // different cache content.
    fn max_error_for_padding(padding: usize, program: &[Op], root: NodeId) -> f32 {
        let pairs = (HEAD_DIM / 2) as usize;
        let group = (QUERY_HEADS / KV_HEADS) as usize;
        let sequence = MERGED_LEN + padding;

        let table = random_vec(1, VOCAB as usize * EMBEDDING as usize);
        let ids: Vec<f32> = (0..NEW_COUNT as u32)
            .map(|id| 1.0 + (id % 3) as f32)
            .collect();
        let eps = alloc::vec![1e-5f32; NEW_COUNT];
        let (cos_new, sin_new) = rope_angles(CACHED_LEN, NEW_COUNT, pairs, HEAD_DIM as usize);
        let cached_len_scalar = alloc::vec![CACHED_LEN as f32];

        let mut owned: Vec<(String, Vec<f32>)> = alloc::vec![
            (String::from("token_embd.weight"), table),
            (String::from("ids"), ids),
            (String::from("eps"), eps),
            (String::from("rope_cos"), cos_new),
            (String::from("rope_sin"), sin_new),
            (String::from("cached_len"), cached_len_scalar),
        ];
        let mut seed = 900u64;
        for layer in 0..BLOCK_COUNT as usize {
            owned.push((
                alloc::format!("blk.{layer}.attn_norm.weight"),
                alloc::vec![1.0f32; EMBEDDING as usize],
            ));
            owned.push((
                alloc::format!("blk.{layer}.ffn_norm.weight"),
                alloc::vec![1.0f32; EMBEDDING as usize],
            ));
            owned.push((
                alloc::format!("blk.{layer}.attn_q.weight"),
                random_vec(
                    seed,
                    EMBEDDING as usize * QUERY_HEADS as usize * HEAD_DIM as usize,
                ),
            ));
            owned.push((
                alloc::format!("blk.{layer}.attn_k.weight"),
                random_vec(
                    seed + 1,
                    EMBEDDING as usize * KV_HEADS as usize * HEAD_DIM as usize,
                ),
            ));
            owned.push((
                alloc::format!("blk.{layer}.attn_v.weight"),
                random_vec(
                    seed + 2,
                    EMBEDDING as usize * KV_HEADS as usize * HEAD_DIM as usize,
                ),
            ));
            owned.push((
                alloc::format!("blk.{layer}.attn_output.weight"),
                random_vec(
                    seed + 3,
                    KV_HEADS as usize * group * HEAD_DIM as usize * EMBEDDING as usize,
                ),
            ));
            owned.push((
                alloc::format!("blk.{layer}.ffn_gate.weight"),
                random_vec(seed + 4, EMBEDDING as usize * FEED_FORWARD as usize),
            ));
            owned.push((
                alloc::format!("blk.{layer}.ffn_up.weight"),
                random_vec(seed + 5, EMBEDDING as usize * FEED_FORWARD as usize),
            ));
            owned.push((
                alloc::format!("blk.{layer}.ffn_down.weight"),
                random_vec(seed + 6, FEED_FORWARD as usize * EMBEDDING as usize),
            ));
            let mut k_even = random_vec(seed + 7, MERGED_LEN * KV_HEADS as usize * pairs);
            k_even.resize(sequence * KV_HEADS as usize * pairs, 0.0);
            let mut k_odd = random_vec(seed + 8, MERGED_LEN * KV_HEADS as usize * pairs);
            k_odd.resize(sequence * KV_HEADS as usize * pairs, 0.0);
            let mut v =
                random_vec(seed + 9, MERGED_LEN * KV_HEADS as usize * HEAD_DIM as usize);
            v.resize(sequence * KV_HEADS as usize * HEAD_DIM as usize, 0.0);
            owned.push((alloc::format!("kv_cache.{layer}.k_even"), k_even));
            owned.push((alloc::format!("kv_cache.{layer}.k_odd"), k_odd));
            owned.push((alloc::format!("kv_cache.{layer}.v"), v));
            seed += 10;
        }
        owned.push((
            String::from("output_norm.weight"),
            alloc::vec![1.0f32; EMBEDDING as usize],
        ));
        owned.push((
            String::from("output.weight"),
            random_vec(seed, EMBEDDING as usize * VOCAB as usize),
        ));

        let shapes = crate::shape::infer(program, &[NEW_COUNT as u64, sequence as u64])
            .expect("single-range fused-vs-unfused fixture infers");

        let inputs_for = |resolved: &[BoundOp]| -> Vec<(NodeId, Vec<f32>)> {
            let _ = resolved;
            block_node_ids(program)
                .into_iter()
                .map(|node| {
                    let name = match &program[node.0 as usize] {
                        Op::Input {
                            name: Some(name), ..
                        } => name.clone(),
                        _ => unreachable!("block_node_ids only ever returns Op::Input nodes"),
                    };
                    let data = owned
                        .iter()
                        .find(|(candidate, _)| *candidate == name)
                        .unwrap_or_else(|| panic!("missing named input {name}"))
                        .1
                        .clone();
                    (node, data)
                })
                .collect()
        };

        let fused = bind_with_fusion(program, &shapes, &[root], true, NumericPolicy::default())
            .expect("fused single-range bind succeeds");
        let unfused =
            bind_with_fusion(program, &shapes, &[root], false, NumericPolicy::default())
                .expect("unfused single-range bind succeeds");

        assert!(
            fused
                .iter()
                .any(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. })),
            "fused bind must produce at least one cached-attention step"
        );
        assert!(
            !unfused
                .iter()
                .any(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. })),
            "unfused bind must never produce a cached-attention step"
        );

        let fused_buffers = run_resolved(program.len(), &fused, inputs_for(&fused));
        let unfused_buffers = run_resolved(program.len(), &unfused, inputs_for(&unfused));

        let fused_logits = fused_buffers[root.0 as usize]
            .as_ref()
            .expect("fused logits present");
        let unfused_logits = unfused_buffers[root.0 as usize]
            .as_ref()
            .expect("unfused logits present");

        assert_eq!(fused_logits.len(), unfused_logits.len());
        let max_error = fused_logits
            .iter()
            .zip(unfused_logits.iter())
            .map(|(fused, unfused)| (fused - unfused).abs())
            .fold(0.0f32, f32::max);
        std::println!(
            "single_range_fused_vs_unfused bucket_padding={padding} max_error={max_error} fused={fused_logits:?} unfused={unfused_logits:?}"
        );
        max_error
    }

    let (program, root, _, _) = mistral_single_range_cached_forward_program(
        VOCAB,
        EMBEDDING,
        FEED_FORWARD,
        QUERY_HEADS,
        KV_HEADS,
        HEAD_DIM,
        BLOCK_COUNT,
        false,
        DuplicateHeadPosition::None,
        false,
    )
    .expect("single-range cached forward pass lowers");

    for bucket_padding in [0usize, 1, 5] {
        let max_error = max_error_for_padding(bucket_padding, &program, root);
        assert!(
            max_error <= 1e-5,
            "fused single-range cached attention diverged from the unfused program at bucket_padding={bucket_padding}: max_error={max_error}"
        );
    }
}

/// CARD 6.1's falsifiable claim: `proxima-model-interop`'s
/// `kv-capacity-bucket` feature rounds the single-range KV extent
/// (`Extent::Symbolic(1)`, bound via `symbols[1]`) up from the true,
/// strictly-increasing `merged_len` to `bucket = ceil(merged_len /
/// KV_BUCKET_TOKENS) * KV_BUCKET_TOKENS`, so the Metal plan-cache key
/// (`(new_count, symbols[1])`) stays constant across a whole bucket of
/// decode steps. This must not move a single bit of the decode step's
/// logits, PROVIDED the padded tail `[merged_len, bucket)` reads as
/// exactly zero: [`causal_mask_merged`]'s existing `key_index >
/// query_absolute` comparison already masks every key index
/// `>= merged_len` as "future" for every query in this call (no
/// query's own absolute position ever reaches a padded key's index,
/// since the last query sits at `merged_len - 1`), so
/// `ScalarOp::Select` picks the constant `neg_infinity` for the whole
/// padded tail without ever reading it -- zero new `Op`/`BoundOpKind`/
/// `ScalarOp`/`IndexMap` variant, exactly `Cargo.toml`'s
/// `kv-capacity-bucket` doc states. Covers the three bucket sizes CARD
/// 6.1 names (8, 32, 256) with `cached_len` set to span each bucket's
/// own boundary (one merged_len below it, exactly on it, one above
/// it) against a one-new-token decode step -- 9 cases.
#[cfg(feature = "kv-capacity-bucket")]
#[test]
fn cpu_mask_zero_ulp() {
    const VOCAB: usize = 5;
    const EMBEDDING: usize = 4;
    const FEED_FORWARD: usize = 4;
    const QUERY_HEADS: usize = 2;
    const KV_HEADS: usize = 1;
    const HEAD_DIM: usize = 2;
    const PAIRS: usize = HEAD_DIM / 2;
    const GROUP: usize = QUERY_HEADS / KV_HEADS;
    const BLOCK_COUNT: u32 = 2;
    const NEW_COUNT: usize = 1;

    fn bucket_of(merged_len: usize, bucket_tokens: usize) -> usize {
        merged_len.div_ceil(bucket_tokens) * bucket_tokens
    }

    // tight (unbucketed, `symbols[1] == merged_len`) vs padded
    // (`symbols[1] == bucket`) logits for the SAME weights, SAME
    // `cached_len`, SAME real KV content in `[0, merged_len)` --
    // built once and sliced/extended, never regenerated per arm, so
    // any divergence is the mask, not a different random draw.
    fn logits_at(cached_len: usize, bucket_tokens: usize) -> (Vec<f32>, Vec<f32>) {
        let merged_len = cached_len + NEW_COUNT;
        let bucket = bucket_of(merged_len, bucket_tokens);
        let ids_f32: Vec<f32> = (0..merged_len as u32)
            .map(|id| (1 + id % 3) as f32)
            .collect();

        let table = random_vec(10, VOCAB * EMBEDDING);
        let eps_new = alloc::vec![1e-5f32; NEW_COUNT];
        let (cos_new, sin_new) = rope_angles(cached_len, NEW_COUNT, PAIRS, HEAD_DIM);

        let layer_names: Vec<[alloc::string::String; 9]> = (0..BLOCK_COUNT as usize)
            .map(|layer| {
                [
                    alloc::format!("blk.{layer}.attn_norm.weight"),
                    alloc::format!("blk.{layer}.ffn_norm.weight"),
                    alloc::format!("blk.{layer}.attn_q.weight"),
                    alloc::format!("blk.{layer}.attn_k.weight"),
                    alloc::format!("blk.{layer}.attn_v.weight"),
                    alloc::format!("blk.{layer}.attn_output.weight"),
                    alloc::format!("blk.{layer}.ffn_gate.weight"),
                    alloc::format!("blk.{layer}.ffn_up.weight"),
                    alloc::format!("blk.{layer}.ffn_down.weight"),
                ]
            })
            .collect();
        let kv_cache_names: Vec<[alloc::string::String; 3]> = (0..BLOCK_COUNT as usize)
            .map(|layer| {
                [
                    alloc::format!("kv_cache.{layer}.k_even"),
                    alloc::format!("kv_cache.{layer}.k_odd"),
                    alloc::format!("kv_cache.{layer}.v"),
                ]
            })
            .collect();

        let mut common_named: Vec<(&str, &[f32])> =
            alloc::vec![("token_embd.weight", table.as_slice())];
        let mut layer_weights: Vec<[Vec<f32>; 9]> = Vec::with_capacity(BLOCK_COUNT as usize);
        let mut seed = 900u64;
        for _ in 0..BLOCK_COUNT {
            layer_weights.push([
                alloc::vec![1.0f32; EMBEDDING],
                alloc::vec![1.0f32; EMBEDDING],
                random_vec(seed, EMBEDDING * QUERY_HEADS * HEAD_DIM),
                random_vec(seed + 1, EMBEDDING * KV_HEADS * HEAD_DIM),
                random_vec(seed + 2, EMBEDDING * KV_HEADS * HEAD_DIM),
                random_vec(seed + 3, KV_HEADS * GROUP * HEAD_DIM * EMBEDDING),
                random_vec(seed + 4, EMBEDDING * FEED_FORWARD),
                random_vec(seed + 5, EMBEDDING * FEED_FORWARD),
                random_vec(seed + 6, FEED_FORWARD * EMBEDDING),
            ]);
            seed += 7;
        }
        for (layer_index, weights) in layer_weights.iter().enumerate() {
            let names = &layer_names[layer_index];
            for (name, data) in names.iter().zip(weights.iter()) {
                common_named.push((name.as_str(), data.as_slice()));
            }
        }
        let output_norm = alloc::vec![1.0f32; EMBEDDING];
        let lm_head = random_vec(seed, EMBEDDING * VOCAB);
        common_named.push(("output_norm.weight", output_norm.as_slice()));
        common_named.push(("output.weight", lm_head.as_slice()));

        // per-layer KV cache, `bucket`-long, real random content in
        // `[0, merged_len)`, EXACT zero in the padded tail
        // `[merged_len, bucket)` -- the tail-zero invariant
        // `run_decode_loop_placed_kv`'s own one-time buffer zero-fill
        // provides at runtime (`omega::metal::zero_placed_buffer`).
        let mut k_even_padded: Vec<Vec<f32>> = Vec::with_capacity(BLOCK_COUNT as usize);
        let mut k_odd_padded: Vec<Vec<f32>> = Vec::with_capacity(BLOCK_COUNT as usize);
        let mut v_padded: Vec<Vec<f32>> = Vec::with_capacity(BLOCK_COUNT as usize);
        let mut kv_seed = 5000u64;
        for _ in 0..BLOCK_COUNT {
            let mut k_even = random_vec(kv_seed, merged_len * KV_HEADS * PAIRS);
            k_even.resize(bucket * KV_HEADS * PAIRS, 0.0);
            let mut k_odd = random_vec(kv_seed + 1, merged_len * KV_HEADS * PAIRS);
            k_odd.resize(bucket * KV_HEADS * PAIRS, 0.0);
            let mut v = random_vec(kv_seed + 2, merged_len * KV_HEADS * HEAD_DIM);
            v.resize(bucket * KV_HEADS * HEAD_DIM, 0.0);
            k_even_padded.push(k_even);
            k_odd_padded.push(k_odd);
            v_padded.push(v);
            kv_seed += 3;
        }

        let (program, root, _, _) = mistral_single_range_cached_forward_program(
            VOCAB as u32,
            EMBEDDING as u32,
            FEED_FORWARD as u32,
            QUERY_HEADS as u32,
            KV_HEADS as u32,
            HEAD_DIM as u32,
            BLOCK_COUNT,
            false,
            DuplicateHeadPosition::None,
            false,
        )
        .expect("single-range cached forward pass lowers");

        let cached_len_scalar = alloc::vec![cached_len as f32];
        let run = |extent: usize, k_even: &[Vec<f32>], k_odd: &[Vec<f32>], v: &[Vec<f32>]| {
            let mut named = common_named.clone();
            named.push(("ids", ids_f32[cached_len..].as_ref()));
            named.push(("eps", eps_new.as_slice()));
            named.push(("rope_cos", cos_new.as_slice()));
            named.push(("rope_sin", sin_new.as_slice()));
            named.push(("cached_len", cached_len_scalar.as_slice()));
            for (layer_index, names) in kv_cache_names.iter().enumerate() {
                named.push((names[0].as_str(), k_even[layer_index].as_slice()));
                named.push((names[1].as_str(), k_odd[layer_index].as_slice()));
                named.push((names[2].as_str(), v[layer_index].as_slice()));
            }
            let symbols = [NEW_COUNT as u64, extent as u64];
            let evaluated = crate::cpu::evaluate_named(&program, &symbols, &named, &[root])
                .expect("single-range decode call evaluates");
            evaluated.get(root).expect("logits present").0.to_vec()
        };

        let tight_k_even: Vec<Vec<f32>> = k_even_padded
            .iter()
            .map(|column| column[..merged_len * KV_HEADS * PAIRS].to_vec())
            .collect();
        let tight_k_odd: Vec<Vec<f32>> = k_odd_padded
            .iter()
            .map(|column| column[..merged_len * KV_HEADS * PAIRS].to_vec())
            .collect();
        let tight_v: Vec<Vec<f32>> = v_padded
            .iter()
            .map(|column| column[..merged_len * KV_HEADS * HEAD_DIM].to_vec())
            .collect();

        let tight_logits = run(merged_len, &tight_k_even, &tight_k_odd, &tight_v);
        let padded_logits = run(bucket, &k_even_padded, &k_odd_padded, &v_padded);
        (tight_logits, padded_logits)
    }

    let cases: Vec<(usize, usize)> = [8usize, 32, 256]
        .into_iter()
        .flat_map(|bucket_tokens| {
            [
                bucket_tokens.saturating_sub(2),
                bucket_tokens.saturating_sub(1),
                bucket_tokens,
            ]
            .into_iter()
            .map(move |cached_len| (bucket_tokens, cached_len))
        })
        .collect();
    assert_eq!(
        cases.len(),
        9,
        "3 bucket sizes x 3 boundary-spanning cached_len values"
    );

    for (bucket_tokens, cached_len) in cases {
        let (tight, padded) = logits_at(cached_len, bucket_tokens);
        std::println!(
            "cpu_mask_zero_ulp bucket_tokens={bucket_tokens} cached_len={cached_len} tight={tight:?} padded={padded:?}"
        );
        assert_eq!(
            tight, padded,
            "bucket_tokens={bucket_tokens} cached_len={cached_len}: bucketed KV extent diverged from the tight extent, 0-ULP required"
        );
    }
}

/// Classifies every [`crate::bind::BoundOp`] a real decode step binds as
/// VARIANT (its resolved shape/layout/body changes when `cached_len`
/// changes) or INVARIANT (it does not), by binding the SAME program
/// twice against two different `cached_len` values and comparing the
/// two `Vec<BoundOp>` positionally. `BoundOp: PartialEq`
/// (`crate::bind::BoundOp`'s own derive) makes this an exact structural
/// diff, not an inference about which ops "should" depend on the cache:
/// any op whose extents, operand layouts, or fused body differ between
/// the two binds is exactly the set a per-step re-resolve exists to
/// recompute; anything unchanged was resolved for nothing.
///
/// This is the count `docs/discipline.md`'s resolve-once row rests on --
/// see that row for the paper estimate (~15%) this either confirms or
/// refutes.
#[test]
fn bound_ops_are_classified_variant_or_invariant_in_cached_len() {
    const VOCAB: u32 = 32_002;
    const EMBEDDING: u32 = 4096;
    const FEED_FORWARD: u32 = 14336;
    const QUERY_HEADS: u32 = 32;
    const KV_HEADS: u32 = 8;
    const HEAD_DIM: u32 = 128;
    const BLOCK_COUNT: u32 = 2;
    const NEW_COUNT: u64 = 1;
    const CACHED_LEN_A: u64 = 50;
    const CACHED_LEN_B: u64 = 51;

    let header_nodes = mistral_cached_forward_program(
        VOCAB,
        EMBEDDING,
        FEED_FORWARD,
        QUERY_HEADS,
        KV_HEADS,
        HEAD_DIM,
        0,
    )
    .expect("a zero-layer program still lowers (embedding lookup plus final norm/lm-head)")
    .0
    .len();
    let one_layer_nodes = mistral_cached_forward_program(
        VOCAB,
        EMBEDDING,
        FEED_FORWARD,
        QUERY_HEADS,
        KV_HEADS,
        HEAD_DIM,
        1,
    )
    .expect("a one-layer program lowers")
    .0
    .len();
    let per_layer_program_nodes = one_layer_nodes - header_nodes;

    let (program, logits_root, cache_roots) = mistral_cached_forward_program(
        VOCAB,
        EMBEDDING,
        FEED_FORWARD,
        QUERY_HEADS,
        KV_HEADS,
        HEAD_DIM,
        BLOCK_COUNT,
    )
    .expect("the two-layer cached forward pass lowers to a program");
    let mut outputs = alloc::vec![logits_root];
    for (even, odd, value) in &cache_roots {
        outputs.extend_from_slice(&[*even, *odd, *value]);
    }

    let shapes_a = crate::shape::infer(&program, &[NEW_COUNT, CACHED_LEN_A])
        .expect("cached_len=50 infers");
    let resolved_a = crate::bind::bind(
        &program,
        &shapes_a,
        &outputs,
        crate::numeric::NumericPolicy::bit_exact(),
    )
    .expect("cached_len=50 binds");
    let shapes_b = crate::shape::infer(&program, &[NEW_COUNT, CACHED_LEN_B])
        .expect("cached_len=51 infers");
    let resolved_b = crate::bind::bind(
        &program,
        &shapes_b,
        &outputs,
        crate::numeric::NumericPolicy::bit_exact(),
    )
    .expect("cached_len=51 binds");

    assert_eq!(
        resolved_a.len(),
        resolved_b.len(),
        "the same program topology must bind to the same bound-op count regardless of cached_len"
    );

    let mut variant_total = 0usize;
    let mut invariant_total = 0usize;
    // layer index -> (variant, invariant); usize::MAX buckets header/lm-head nodes
    let mut per_layer: std::collections::BTreeMap<usize, (usize, usize)> =
        std::collections::BTreeMap::new();

    for (bound_a, bound_b) in resolved_a.iter().zip(resolved_b.iter()) {
        let variant = bound_a != bound_b;
        if variant {
            variant_total += 1;
        } else {
            invariant_total += 1;
        }
        let node_index = bound_a.node.0 as usize;
        let layer = if node_index >= header_nodes {
            let offset = node_index - header_nodes;
            let layer = offset / per_layer_program_nodes;
            if layer < BLOCK_COUNT as usize {
                layer
            } else {
                usize::MAX
            }
        } else {
            usize::MAX
        };
        let entry = per_layer.entry(layer).or_insert((0, 0));
        if variant {
            entry.0 += 1;
        } else {
            entry.1 += 1;
        }
    }

    std::println!(
        "bound_op_classification total_bound_ops={} variant={variant_total} invariant={invariant_total} variant_pct={:.1}",
        resolved_a.len(),
        100.0 * variant_total as f64 / resolved_a.len() as f64
    );
    for (layer, (variant, invariant)) in &per_layer {
        let label = if *layer == usize::MAX {
            "header_or_lm_head".to_string()
        } else {
            alloc::format!("layer_{layer}")
        };
        std::println!(
            "bound_op_classification bucket={label} variant={variant} invariant={invariant}"
        );
    }

    assert!(
        variant_total > 0,
        "the cache-reading ops must be classified variant, or this test cannot distinguish anything"
    );
    assert!(
        invariant_total > 0,
        "if every bound op is variant the resolve-once split has nothing to cache -- report this, do not build the split"
    );
}

/// The interpreter's per-node dispatch floor: how long a node costs
/// when the node does essentially no arithmetic. This is the number the
/// chunked-cache node budget above has to be multiplied by, because a
/// chunk's own cache-reading nodes are tiny -- one 256-position slice
/// of one head -- so what a chunk costs is dispatch, not math.
///
/// Shaped as a balanced `Add` tree over `[1]`-shaped tensors, not a
/// chain: `PROXIMA_CHAIN_DEPTH` below records that a linear chain
/// overflows this evaluator's stack, and a balanced tree is what an
/// N-way associative combine wants anyway.
#[test]
fn the_interpreter_per_node_dispatch_floor_is_measured() {
    const LEAVES: usize = 2_048;
    const REPEATS: usize = 20;

    let mut program = Vec::new();
    let seed = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(1)],
        "seed",
    );
    let mut level: Vec<NodeId> = (0..LEAVES)
        .map(|_| {
            elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Add,
                &[(seed, "a->a"), (seed, "a->a")],
            )
            .expect("a scalar add lowers")
        })
        .collect();
    while level.len() > 1 {
        level = level
            .chunks(2)
            .map(|pair| match pair {
                [left, right] => elementwise(
                    &mut program,
                    DType::Float32,
                    ScalarOp::Add,
                    &[(*left, "a->a"), (*right, "a->a")],
                )
                .expect("a scalar add lowers"),
                [only] => *only,
                _ => unreachable!("chunks(2) yields one or two"),
            })
            .collect();
    }
    let root = level[0];
    let total = program.len();

    let seed_data = alloc::vec![1.0f32];
    let named: [(&str, &[f32]); 1] = [("seed", seed_data.as_slice())];

    let mut samples: Vec<f64> = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let started = std::time::Instant::now();
        crate::cpu::evaluate_named(&program, &[1], &named, &[root])
            .expect("the tree evaluates");
        samples.push(started.elapsed().as_secs_f64() * 1e9 / total as f64);
    }
    samples.sort_by(|left, right| left.partial_cmp(right).expect("no nan timings"));

    std::println!(
        "per_node_floor nodes={total} repeats={REPEATS} median_ns={:.1} min_ns={:.1} max_ns={:.1}",
        samples[REPEATS / 2],
        samples[0],
        samples[REPEATS - 1]
    );
    assert_eq!(samples.len(), REPEATS, "one timing per repeat");
}

/// How deep a dependency chain this evaluator survives. A flat N-chunk
/// cache fold that combines chunks pairwise left-to-right builds a
/// chain exactly N long, so this bounds that shape independently of the
/// node-count budget. Depth comes from `PROXIMA_CHAIN_DEPTH` so a
/// caller can walk it upward across separate processes -- a stack
/// overflow aborts, it does not unwind, so one process cannot bisect it.
#[test]
#[ignore = "probes the evaluator's stack depth; aborts by design past the limit"]
fn the_evaluator_survives_a_dependency_chain_of_a_given_depth() {
    let depth: usize = std::env::var("PROXIMA_CHAIN_DEPTH")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(256);

    let mut program = Vec::new();
    let seed = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(1)],
        "seed",
    );
    let mut tip = seed;
    for _ in 0..depth {
        tip = elementwise(
            &mut program,
            DType::Float32,
            ScalarOp::Add,
            &[(tip, "a->a"), (seed, "a->a")],
        )
        .expect("a scalar add chains");
    }

    let seed_data = alloc::vec![1.0f32];
    let named: [(&str, &[f32]); 1] = [("seed", seed_data.as_slice())];
    let evaluated = crate::cpu::evaluate_named(&program, &[1], &named, &[tip])
        .expect("the chain evaluates");
    let (data, _) = evaluated.get(tip).expect("chain tip present");

    std::println!(
        "chain_depth depth={depth} nodes={} value={}",
        program.len(),
        data[0]
    );
    assert_eq!(data[0], 1.0 + depth as f32, "each link adds one seed");
}

/// Wall-clock probe for the whole forward pass, `SEQUENCE=4`, RANDOM
/// weights, at the model's real dimensions — the 32-layer analogue of
/// `a_mistral_layer_written_as_toml_evaluates_at_its_real_dimensions`
/// above, gated `#[ignore]` for the same reason and then some: ~32
/// layers' worth of real-dimension weights is tens of GB and a
/// multi-second-per-layer run, neither of which belongs in the default
/// `nextest` budget. Run explicitly with `--ignored --release`.
#[test]
#[ignore = "measures the whole real-dimension mistral forward pass's wall clock; run explicitly"]
fn the_whole_mistral_forward_pass_evaluates_at_real_dimensions() {
    const SEQUENCE: usize = 4;
    const VOCAB: usize = 32_002;
    const EMBEDDING: usize = 4096;
    const QUERY_HEADS: usize = 32;
    const KV_HEADS: usize = 8;
    const HEAD_DIM: usize = 128;
    const PAIRS: usize = HEAD_DIM / 2;
    const GROUP: usize = QUERY_HEADS / KV_HEADS;
    const FEED_FORWARD: usize = 14336;
    const BLOCK_COUNT: u32 = 32;

    let program = mistral_forward_program(
        VOCAB as u32,
        EMBEDDING as u32,
        FEED_FORWARD as u32,
        QUERY_HEADS as u32,
        KV_HEADS as u32,
        HEAD_DIM as u32,
        BLOCK_COUNT,
        0,
        0,
    )
    .expect("the whole forward pass lowers to a program");

    let symbols = [SEQUENCE as u64];
    crate::shape::infer(&program, &symbols).expect("the whole forward pass infers");

    // block order mirrors `mistral_forward_program`'s own `Input`
    // emission order exactly: ids, table, eps, cos/sin, then each
    // layer's attn_norm_weight/ffn_norm_weight/wq/wk/wv/wo/w_gate/
    // w_up/w_down, then the lm head. `inv_dim`, `ones`, `group_ones`,
    // `inv_sqrt_head_dim` and `neg_infinity` are `Op::Constant` now, so
    // none of them has a block here — that collapse is what
    // `no_repeated_scalar_crosses_the_binding_surface` asserts.
    // `block_node_ids` (cpu.rs) reads `Input`s positionally, which is
    // why this order is load-bearing, not cosmetic.
    let ids: Vec<f32> = (0..SEQUENCE)
        .map(|position| (position % VOCAB) as f32)
        .collect();
    let table = random_vec(200, VOCAB * EMBEDDING);
    let epsilon = alloc::vec![1e-5f32; SEQUENCE];
    let cos = random_vec(201, SEQUENCE * PAIRS);
    let sin = random_vec(202, SEQUENCE * PAIRS);

    let mut owned: Vec<Vec<f32>> = Vec::new();
    let mut seed = 300u64;
    for _layer in 0..BLOCK_COUNT {
        owned.push(alloc::vec![1.0f32; EMBEDDING]);
        owned.push(alloc::vec![1.0f32; EMBEDDING]);
        owned.push(random_vec(seed, EMBEDDING * QUERY_HEADS * HEAD_DIM));
        seed += 1;
        owned.push(random_vec(seed, EMBEDDING * KV_HEADS * HEAD_DIM));
        seed += 1;
        owned.push(random_vec(seed, EMBEDDING * KV_HEADS * HEAD_DIM));
        seed += 1;
        owned.push(random_vec(seed, KV_HEADS * GROUP * HEAD_DIM * EMBEDDING));
        seed += 1;
        owned.push(random_vec(seed, EMBEDDING * FEED_FORWARD));
        seed += 1;
        owned.push(random_vec(seed, EMBEDDING * FEED_FORWARD));
        seed += 1;
        owned.push(random_vec(seed, FEED_FORWARD * EMBEDDING));
        seed += 1;
    }
    let lm_head = random_vec(seed, EMBEDDING * VOCAB);

    let mut blocks: Vec<&[f32]> = alloc::vec![
        ids.as_slice(),
        table.as_slice(),
        epsilon.as_slice(),
        cos.as_slice(),
        sin.as_slice(),
    ];
    for layer_weights in &owned {
        blocks.push(layer_weights.as_slice());
    }
    blocks.push(lm_head.as_slice());

    let root = NodeId(program.len() as u32 - 1);
    let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");

    let wall_start = std::time::Instant::now();
    let evaluated =
        crate::cpu::evaluate_parallel(&program, &symbols, &blocks, &[root], workers)
            .expect("the whole real-dimension mistral forward pass evaluates");
    let wall = wall_start.elapsed();
    std::println!(
        "whole_forward_pass: wall_clock={wall:?} per_layer={:?}",
        wall / BLOCK_COUNT
    );

    let output = evaluated.root();
    assert_eq!(
        output.len(),
        SEQUENCE * VOCAB,
        "logits must be [seq, vocab]"
    );
    assert!(
        output.iter().all(|value| value.is_finite()),
        "logits must be finite"
    );

    let last_row = &output[(SEQUENCE - 1) * VOCAB..SEQUENCE * VOCAB];
    let argmax = last_row
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .expect("logits row is nonempty");
    assert!(
        argmax < VOCAB,
        "argmax {argmax} must address a real vocab entry, meaningless as it is with random weights"
    );
    std::println!("argmax(last position)={argmax}");
}

/// Absolute-position RoPE angles for `count` positions starting at
/// `start` -- the same formula `bind.rs`'s `build_position_inputs`
/// computes per call, generalized with a start offset so a decode
/// step's lone new position gets its true absolute angle instead of
/// position 0.
fn rope_angles(
    start: usize,
    count: usize,
    pairs: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut cos = alloc::vec![0.0f32; count * pairs];
    let mut sin = alloc::vec![0.0f32; count * pairs];
    for offset in 0..count {
        let position = (start + offset) as f32;
        for pair in 0..pairs {
            let theta = position
                * crate::sized::ROPE_FREQ_BASE_DEFAULT
                    .powf(-((2 * pair) as f32) / (head_dim as f32));
            cos[offset * pairs + pair] = theta.cos();
            sin[offset * pairs + pair] = theta.sin();
        }
    }
    (cos, sin)
}

/// The falsifiable claim under test: a prefill call followed by a
/// one-token decode call through [`mistral_cached_forward_program`]
/// must produce the SAME last-position logits [`mistral_forward_program`]
/// produces evaluating the whole sequence at once, with NO per-step
/// growth in the amount of new work the decode call performs (it binds
/// a fixed `N=1` symbol regardless of how long the cache has grown).
/// This is the acceptance criterion from the task brief, proven here at
/// tiny synthetic dimensions instead of the real 226-tensor checkpoint
/// so a wrong index map fails in milliseconds, not after a 36-second
/// real-model run.
#[test]
fn a_cached_decode_step_matches_the_uncached_forward_pass_exactly() {
    const VOCAB: usize = 5;
    const EMBEDDING: usize = 4;
    const FEED_FORWARD: usize = 4;
    const QUERY_HEADS: usize = 2;
    const KV_HEADS: usize = 1;
    const HEAD_DIM: usize = 2;
    const PAIRS: usize = HEAD_DIM / 2;
    const GROUP: usize = QUERY_HEADS / KV_HEADS;
    const BLOCK_COUNT: u32 = 2;
    const PROMPT_LEN: usize = 2;
    const SEQUENCE: usize = PROMPT_LEN + 1;

    let ids: Vec<u32> = alloc::vec![1, 3, 2];
    let ids_f32: Vec<f32> = ids.iter().map(|&id| id as f32).collect();

    let table = random_vec(10, VOCAB * EMBEDDING);
    let epsilon_full = alloc::vec![1e-5f32; SEQUENCE];
    let epsilon_one = alloc::vec![1e-5f32; 1];
    let epsilon_prompt = alloc::vec![1e-5f32; PROMPT_LEN];
    let (cos_full, sin_full) = rope_angles(0, SEQUENCE, PAIRS, HEAD_DIM);
    let (cos_prompt, sin_prompt) = rope_angles(0, PROMPT_LEN, PAIRS, HEAD_DIM);
    let (cos_decode, sin_decode) = rope_angles(PROMPT_LEN, 1, PAIRS, HEAD_DIM);

    struct LayerWeights {
        attn_norm: Vec<f32>,
        ffn_norm: Vec<f32>,
        wq: Vec<f32>,
        wk: Vec<f32>,
        wv: Vec<f32>,
        wo: Vec<f32>,
        w_gate: Vec<f32>,
        w_up: Vec<f32>,
        w_down: Vec<f32>,
    }

    let mut layers = Vec::new();
    let mut seed = 100u64;
    for _ in 0..BLOCK_COUNT {
        let weights = LayerWeights {
            attn_norm: alloc::vec![1.0f32; EMBEDDING],
            ffn_norm: alloc::vec![1.0f32; EMBEDDING],
            wq: random_vec(seed, EMBEDDING * QUERY_HEADS * HEAD_DIM),
            wk: random_vec(seed + 1, EMBEDDING * KV_HEADS * HEAD_DIM),
            wv: random_vec(seed + 2, EMBEDDING * KV_HEADS * HEAD_DIM),
            wo: random_vec(seed + 3, KV_HEADS * GROUP * HEAD_DIM * EMBEDDING),
            w_gate: random_vec(seed + 4, EMBEDDING * FEED_FORWARD),
            w_up: random_vec(seed + 5, EMBEDDING * FEED_FORWARD),
            w_down: random_vec(seed + 6, FEED_FORWARD * EMBEDDING),
        };
        seed += 7;
        layers.push(weights);
    }
    let output_norm = alloc::vec![1.0f32; EMBEDDING];
    let lm_head = random_vec(seed, EMBEDDING * VOCAB);

    // -- uncached oracle: the whole 3-token sequence in one shot.
    let uncached_program = mistral_forward_program(
        VOCAB as u32,
        EMBEDDING as u32,
        FEED_FORWARD as u32,
        QUERY_HEADS as u32,
        KV_HEADS as u32,
        HEAD_DIM as u32,
        BLOCK_COUNT,
        0,
        0,
    )
    .expect("uncached forward pass lowers");
    // real `blk.{layer}.*` names, built with `alloc::format!` so
    // ownership outlives the `&str` borrows below.
    let layer_names: Vec<[alloc::string::String; 9]> = layers
        .iter()
        .enumerate()
        .map(|(layer, _)| {
            [
                alloc::format!("blk.{layer}.attn_norm.weight"),
                alloc::format!("blk.{layer}.ffn_norm.weight"),
                alloc::format!("blk.{layer}.attn_q.weight"),
                alloc::format!("blk.{layer}.attn_k.weight"),
                alloc::format!("blk.{layer}.attn_v.weight"),
                alloc::format!("blk.{layer}.attn_output.weight"),
                alloc::format!("blk.{layer}.ffn_gate.weight"),
                alloc::format!("blk.{layer}.ffn_up.weight"),
                alloc::format!("blk.{layer}.ffn_down.weight"),
            ]
        })
        .collect();
    let mut uncached_named: Vec<(&str, &[f32])> = alloc::vec![
        ("ids", ids_f32.as_slice()),
        ("token_embd.weight", table.as_slice()),
        ("eps", epsilon_full.as_slice()),
        ("rope_cos", cos_full.as_slice()),
        ("rope_sin", sin_full.as_slice())
    ];
    for (layer_index, weights) in layers.iter().enumerate() {
        let names = &layer_names[layer_index];
        uncached_named.push((names[0].as_str(), weights.attn_norm.as_slice()));
        uncached_named.push((names[1].as_str(), weights.ffn_norm.as_slice()));
        uncached_named.push((names[2].as_str(), weights.wq.as_slice()));
        uncached_named.push((names[3].as_str(), weights.wk.as_slice()));
        uncached_named.push((names[4].as_str(), weights.wv.as_slice()));
        uncached_named.push((names[5].as_str(), weights.wo.as_slice()));
        uncached_named.push((names[6].as_str(), weights.w_gate.as_slice()));
        uncached_named.push((names[7].as_str(), weights.w_up.as_slice()));
        uncached_named.push((names[8].as_str(), weights.w_down.as_slice()));
    }
    uncached_named.push(("output_norm.weight", output_norm.as_slice()));
    uncached_named.push(("output.weight", lm_head.as_slice()));

    let uncached_root = NodeId(uncached_program.len() as u32 - 1);
    let uncached_evaluated = crate::cpu::evaluate_named(
        &uncached_program,
        &[SEQUENCE as u64],
        &uncached_named,
        &[uncached_root],
    )
    .expect("uncached forward pass evaluates");
    let (uncached_logits, uncached_shape) = uncached_evaluated
        .get(uncached_root)
        .expect("uncached logits present");
    assert_eq!(uncached_shape, [SEQUENCE as u64, VOCAB as u64]);
    let uncached_last_position = &uncached_logits[(SEQUENCE - 1) * VOCAB..SEQUENCE * VOCAB];

    // -- cached path: prefill the first PROMPT_LEN positions, then one
    // decode step for the final position, growing the cache in between
    // exactly the way `bind.rs`'s decode loop would.
    let (cached_program, cached_logits_root, cache_roots) = mistral_cached_forward_program(
        VOCAB as u32,
        EMBEDDING as u32,
        FEED_FORWARD as u32,
        QUERY_HEADS as u32,
        KV_HEADS as u32,
        HEAD_DIM as u32,
        BLOCK_COUNT,
    )
    .expect("cached forward pass lowers");

    let empty_k_even = Vec::<f32>::new();
    let empty_k_odd = Vec::<f32>::new();
    let empty_v = Vec::<f32>::new();
    let prefill_cached_len = [0.0f32];
    let mut prefill_named: Vec<(&str, &[f32])> = alloc::vec![
        ("ids", &ids_f32[..PROMPT_LEN]),
        ("token_embd.weight", table.as_slice()),
        ("eps", epsilon_prompt.as_slice()),
        ("rope_cos", cos_prompt.as_slice()),
        ("rope_sin", sin_prompt.as_slice()),
        ("cached_len", prefill_cached_len.as_slice()),
    ];
    for (layer_index, weights) in layers.iter().enumerate() {
        let names = &layer_names[layer_index];
        prefill_named.push((names[0].as_str(), weights.attn_norm.as_slice()));
        prefill_named.push((names[1].as_str(), weights.ffn_norm.as_slice()));
        prefill_named.push((names[2].as_str(), weights.wq.as_slice()));
        prefill_named.push((names[3].as_str(), weights.wk.as_slice()));
        prefill_named.push((names[4].as_str(), weights.wv.as_slice()));
        prefill_named.push((names[5].as_str(), weights.wo.as_slice()));
        prefill_named.push((names[6].as_str(), weights.w_gate.as_slice()));
        prefill_named.push((names[7].as_str(), weights.w_up.as_slice()));
        prefill_named.push((names[8].as_str(), weights.w_down.as_slice()));
    }
    prefill_named.push(("output_norm.weight", output_norm.as_slice()));
    prefill_named.push(("output.weight", lm_head.as_slice()));
    let kv_cache_names: Vec<[alloc::string::String; 3]> = (0..BLOCK_COUNT as usize)
        .map(|layer| {
            [
                alloc::format!("kv_cache.{layer}.k_even"),
                alloc::format!("kv_cache.{layer}.k_odd"),
                alloc::format!("kv_cache.{layer}.v"),
            ]
        })
        .collect();
    for names in &kv_cache_names {
        prefill_named.push((names[0].as_str(), empty_k_even.as_slice()));
        prefill_named.push((names[1].as_str(), empty_k_odd.as_slice()));
        prefill_named.push((names[2].as_str(), empty_v.as_slice()));
    }

    let mut prefill_roots: Vec<NodeId> = alloc::vec![cached_logits_root];
    for (even, odd, value) in &cache_roots {
        prefill_roots.push(*even);
        prefill_roots.push(*odd);
        prefill_roots.push(*value);
    }
    let prefill_symbols = [PROMPT_LEN as u64, 0u64];
    let prefill_evaluated = crate::cpu::evaluate_named(
        &cached_program,
        &prefill_symbols,
        &prefill_named,
        &prefill_roots,
    )
    .expect("prefill call evaluates");

    let mut k_even_cache: Vec<Vec<f32>> = Vec::with_capacity(BLOCK_COUNT as usize);
    let mut k_odd_cache: Vec<Vec<f32>> = Vec::with_capacity(BLOCK_COUNT as usize);
    let mut v_cache: Vec<Vec<f32>> = Vec::with_capacity(BLOCK_COUNT as usize);
    for (even, odd, value) in &cache_roots {
        let (even_data, _) = prefill_evaluated
            .get(*even)
            .expect("prefill k_even present");
        let (odd_data, _) = prefill_evaluated.get(*odd).expect("prefill k_odd present");
        let (value_data, _) = prefill_evaluated.get(*value).expect("prefill v present");
        k_even_cache.push(even_data.to_vec());
        k_odd_cache.push(odd_data.to_vec());
        v_cache.push(value_data.to_vec());
    }

    let decode_cached_len = [PROMPT_LEN as f32];
    let mut decode_named: Vec<(&str, &[f32])> = alloc::vec![
        ("ids", &ids_f32[PROMPT_LEN..]),
        ("token_embd.weight", table.as_slice()),
        ("eps", epsilon_one.as_slice()),
        ("rope_cos", cos_decode.as_slice()),
        ("rope_sin", sin_decode.as_slice()),
        ("cached_len", decode_cached_len.as_slice()),
    ];
    for (layer_index, weights) in layers.iter().enumerate() {
        let names = &layer_names[layer_index];
        decode_named.push((names[0].as_str(), weights.attn_norm.as_slice()));
        decode_named.push((names[1].as_str(), weights.ffn_norm.as_slice()));
        decode_named.push((names[2].as_str(), weights.wq.as_slice()));
        decode_named.push((names[3].as_str(), weights.wk.as_slice()));
        decode_named.push((names[4].as_str(), weights.wv.as_slice()));
        decode_named.push((names[5].as_str(), weights.wo.as_slice()));
        decode_named.push((names[6].as_str(), weights.w_gate.as_slice()));
        decode_named.push((names[7].as_str(), weights.w_up.as_slice()));
        decode_named.push((names[8].as_str(), weights.w_down.as_slice()));
    }
    decode_named.push(("output_norm.weight", output_norm.as_slice()));
    decode_named.push(("output.weight", lm_head.as_slice()));
    for (layer_index, names) in kv_cache_names.iter().enumerate() {
        decode_named.push((names[0].as_str(), k_even_cache[layer_index].as_slice()));
        decode_named.push((names[1].as_str(), k_odd_cache[layer_index].as_slice()));
        decode_named.push((names[2].as_str(), v_cache[layer_index].as_slice()));
    }

    let decode_symbols = [1u64, PROMPT_LEN as u64];
    let decode_evaluated = crate::cpu::evaluate_named(
        &cached_program,
        &decode_symbols,
        &decode_named,
        &[cached_logits_root],
    )
    .expect("decode call evaluates");
    let (decode_logits, decode_shape) = decode_evaluated
        .get(cached_logits_root)
        .expect("decode logits present");
    assert_eq!(decode_shape, [1u64, VOCAB as u64]);

    let max_diff = uncached_last_position
        .iter()
        .zip(decode_logits.iter())
        .map(|(oracle, cached)| (oracle - cached).abs())
        .fold(0.0f32, f32::max);
    std::println!(
        "cached_decode_vs_uncached: oracle={uncached_last_position:?} cached={decode_logits:?} max_diff={max_diff}"
    );
    assert!(
        uncached_last_position
            .iter()
            .any(|&value| value != uncached_last_position[0]),
        "degenerate control: oracle logits are all-equal, this run proves nothing"
    );
    assert!(
        max_diff < 1e-4,
        "cached decode step diverged from the uncached oracle: max_diff={max_diff}"
    );
}

/// [`causal_conv1d`]'s whole reason for existing, checked against
/// arithmetic worked out by hand rather than trusted from the
/// implementation: `l_cache=3`, one channel, `weight = [1, 10, 100]`
/// (tap `l=2` is the current position, `l=0` the furthest lookback --
/// [`append_lfm2_conv_mixer`]'s own convention), `x = [1, 2, 3, 4]`.
/// `out[s] = sum_l valid(s,l) * weight[l] * x[s - 2 + l]`, zero where the
/// window reaches before position 0:
/// - `out[0] = 100*x[0]                               = 100`
/// - `out[1] = 10*x[0]  + 100*x[1]                    = 210`
/// - `out[2] = 1*x[0]   + 10*x[1]  + 100*x[2]         = 321`
/// - `out[3] = 1*x[1]   + 10*x[2]  + 100*x[3]         = 432`
#[proxima::test]
async fn causal_conv1d_matches_a_hand_computed_causal_window() {
    let mut program = Vec::new();
    let x = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Symbolic(0), Extent::Static(1)],
            name: Some("x".into()),
        },
    );
    let weight = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            // `[embedding=1, l_cache=3]`, matching `causal_conv1d`'s own
            // `dl->sld` map -- `l_cache` last/fastest, the real
            // checkpoint's own on-disk axis order (`spec.rs`'s own doc on
            // this map explains why). One channel makes this
            // indistinguishable from the old `[3, 1]` shape byte-for-byte
            // (see `causal_conv1d_catches_a_transposed_multi_channel_weight`
            // below for the shape this single-channel test cannot catch).
            shape: alloc::vec![Extent::Static(1), Extent::Static(3)],
            name: Some("weight".into()),
        },
    );
    let output = causal_conv1d(&mut program, x, weight, 3).expect("causal conv lowers");

    let x_data = [1.0f32, 2.0, 3.0, 4.0];
    let weight_data = [1.0f32, 10.0, 100.0];
    let evaluated = crate::cpu::evaluate_named(
        &program,
        &[4],
        &[("x", &x_data), ("weight", &weight_data)],
        &[output],
    )
    .expect("causal conv evaluates");
    let (result, shape) = evaluated.get(output).expect("conv output present");

    std::println!("causal_conv1d result={result:?} shape={shape:?}");
    assert_eq!(shape, [4u64, 1u64]);
    assert_eq!(result, [100.0, 210.0, 321.0, 432.0]);
}

/// proxima-debugger unit oracle (qwen35moe GDN prefill-scan-vs-sequential
/// divergence), narrowed to the one row this comparison is actually
/// valid for: [`causal_conv1d`] evaluated on a `[13, 4]` batch's row 0
/// MUST match a standalone `[1, 4]` call on that same row's data
/// (the interop crate's own since-removed prefill-scan path fed
/// `causal_conv1d` this same shape). Row 0 alone: for
/// `position >= 1`, a fresh `[1, 4]` call is NOT the same computation as
/// that row read out of the `[13, 4]` batch -- `causal_conv1d`'s own
/// `sequence_index` iota starts a length-1 input at its own position 0,
/// so it has no way to see the earlier rows the batch's causal window
/// legitimately reads; only row 0 (the model's own first token, no
/// history either way) is comparable this way, matching what the real
/// checkpoint's own layer-0/position-0 comparison already established.
/// `l_cache=4`/`embedding=4` and non-uniform `x`/`weight` values (never
/// a repeated constant) are deliberate: a repeated value cannot catch a
/// transposed axis or an off-by-one window offset, only a genuinely
/// varying value per `(position, channel, tap)` can.
#[proxima::test]
async fn causal_conv1d_multi_row_batch_row_zero_matches_a_single_row_call() {
    const POSITIONS: usize = 13;
    const EMBEDDING: usize = 4;
    const L_CACHE: usize = 4;

    let x_data: Vec<f32> = (0..POSITIONS * EMBEDDING)
        .map(|index| (index as f32 + 1.0) * 0.1)
        .collect();
    let weight_data: Vec<f32> = (0..EMBEDDING * L_CACHE)
        .map(|index| (index as f32 + 1.0) * 0.01)
        .collect();

    let mut batch_program = Vec::new();
    let batch_x = op::append(
        &mut batch_program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Symbolic(0), Extent::Static(EMBEDDING as u32)],
            name: Some("x".into()),
        },
    );
    let batch_weight = op::append(
        &mut batch_program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![
                Extent::Static(EMBEDDING as u32),
                Extent::Static(L_CACHE as u32)
            ],
            name: Some("weight".into()),
        },
    );
    let batch_output = causal_conv1d(&mut batch_program, batch_x, batch_weight, L_CACHE as u32)
        .expect("causal conv lowers for the batched program");
    let batch_evaluated = crate::cpu::evaluate_named(
        &batch_program,
        &[POSITIONS as u64],
        &[("x", &x_data), ("weight", &weight_data)],
        &[batch_output],
    )
    .expect("batched causal conv evaluates");
    let (batch_result, batch_shape) = batch_evaluated
        .get(batch_output)
        .expect("batched conv output present");
    assert_eq!(batch_shape, [POSITIONS as u64, EMBEDDING as u64]);

    let mut row_program = Vec::new();
    let row_x = op::append(
        &mut row_program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Symbolic(0), Extent::Static(EMBEDDING as u32)],
            name: Some("x".into()),
        },
    );
    let row_weight = op::append(
        &mut row_program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![
                Extent::Static(EMBEDDING as u32),
                Extent::Static(L_CACHE as u32)
            ],
            name: Some("weight".into()),
        },
    );
    let row_output = causal_conv1d(&mut row_program, row_x, row_weight, L_CACHE as u32)
        .expect("causal conv lowers for a single-row program");
    let row_x_data = &x_data[0..EMBEDDING];
    let row_evaluated = crate::cpu::evaluate_named(
        &row_program,
        &[1],
        &[("x", row_x_data), ("weight", &weight_data)],
        &[row_output],
    )
    .expect("single-row causal conv evaluates");
    let (row_result, row_shape) = row_evaluated
        .get(row_output)
        .expect("single-row conv output present");
    assert_eq!(row_shape, [1u64, EMBEDDING as u64]);

    let batch_row = &batch_result[0..EMBEDDING];
    for (channel, (&batched, &single)) in batch_row.iter().zip(row_result).enumerate() {
        let relative_error = (batched - single).abs() / single.abs().max(1e-6);
        std::println!(
            "causal_conv1d_row_zero_check channel={channel} batched={batched} single={single} relative_error={relative_error}"
        );
        assert!(
            relative_error <= 1e-5,
            "channel {channel}: batched={batched} single={single} relative_error={relative_error} exceeds 1e-5"
        );
    }
}

/// proxima-debugger unit oracle (qwen35moe GDN prefill-scan-vs-sequential
/// divergence), the stage [`causal_conv1d_multi_row_batch_row_zero_matches_a_single_row_call`]
/// clears: [`l2norm_with_eps_map`] on the SAME `[s, u, i]` shape
/// `append_qwen35_ssm_mixer`'s own `query_prefill`/`key_prefill` calls
/// use (`kv_heads=16`, `key_dim=128`, matching the real checkpoint's own
/// GQA head count/head width) -- every row is independent (no causal
/// window, no cross-row recurrence at all), so EVERY position's `[13,
/// u, i]`-batched output must equal that same row's own `[1, u, i]`
/// call, not just row 0. `x` is a non-uniform, deterministic pattern
/// (never a repeated constant) so a transposed axis or a stride bug
/// cannot hide behind symmetric data.
#[proxima::test]
async fn l2norm_multi_row_batch_matches_thirteen_single_row_calls() {
    const POSITIONS: usize = 13;
    const KV_HEADS: usize = 16;
    const KEY_DIM: usize = 128;
    const ROW_ELEMENTS: usize = KV_HEADS * KEY_DIM;

    let x_data: Vec<f32> = (0..POSITIONS * ROW_ELEMENTS)
        .map(|index| ((index % 97) as f32 + 1.0) * 0.01)
        .collect();

    let build_and_run = |symbol: u64, x_slice: &[f32]| -> Vec<f32> {
        let mut program = Vec::new();
        let x = op::append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![
                    Extent::Symbolic(0),
                    Extent::Static(KV_HEADS as u32),
                    Extent::Static(KEY_DIM as u32)
                ],
                name: Some("x".into()),
            },
        );
        // matches every real caller (`append_qwen35_ssm_mixer`'s own
        // `eps`): a rank-1 `[s]`-shaped input, one value per position,
        // never a rank-0 `scalar_constant` -- `l2norm_with_eps_map`'s
        // own `eps_map` ("s->su") reads `eps` through a real `s` axis.
        let eps = symbolic_leaf(&mut program, DType::Float32, "eps");
        let output = l2norm_with_eps_map(&mut program, x, eps, "sui->sui", "su->sui", "s->su")
            .expect("l2norm lowers");
        let eps_data = alloc::vec![1e-6f32; symbol as usize];
        let evaluated = crate::cpu::evaluate_named(
            &program,
            &[symbol],
            &[("x", x_slice), ("eps", &eps_data)],
            &[output],
        )
        .expect("l2norm evaluates");
        let (result, shape) = evaluated.get(output).expect("l2norm output present");
        assert_eq!(
            shape,
            [symbol, KV_HEADS as u64, KEY_DIM as u64],
            "l2norm output shape"
        );
        result.to_vec()
    };

    let batch_result = build_and_run(POSITIONS as u64, &x_data);

    for position in 0..POSITIONS {
        let row_slice = &x_data[position * ROW_ELEMENTS..(position + 1) * ROW_ELEMENTS];
        let row_result = build_and_run(1, row_slice);
        let batch_row = &batch_result[position * ROW_ELEMENTS..(position + 1) * ROW_ELEMENTS];
        let max_relative_error = batch_row
            .iter()
            .zip(&row_result)
            .map(|(&batched, &single)| (batched - single).abs() / single.abs().max(1e-6))
            .fold(0.0f32, f32::max);
        std::println!(
            "l2norm_row_check position={position} max_relative_error={max_relative_error}"
        );
        assert!(
            max_relative_error <= 1e-5,
            "position {position}: max_relative_error={max_relative_error} exceeds 1e-5"
        );
    }
}

/// proxima-debugger unit oracle (qwen35moe GDN prefill-scan-vs-sequential
/// divergence), the next stage after [`l2norm_multi_row_batch_matches_thirteen_single_row_calls`]:
/// [`repeat_kv_heads`] multiplies a genuinely `s`-varying operand
/// (`x`, shape `[s,u,d]`) against a broadcast, `s`-INVARIANT donor
/// (`group_ones`, shape `[u,g]`) -- exactly the shape
/// `docs/discipline.md` ROW 561 named (`neon_tile_plan`'s GEMM tile
/// silently reusing row 0's base address for every later row when the
/// "row-invariant" operand secretly still varies along the leading
/// axis). ROW 561's own fix added the `row_stride_b != 0` decline
/// (`cpu.rs:14557-14562`); this test re-proves that fix holds for THIS
/// call site rather than assuming it, on a `[13, u, d]` batch vs the
/// same 13 rows one at a time.
#[proxima::test]
async fn repeat_kv_heads_multi_row_batch_matches_thirteen_single_row_calls() {
    const POSITIONS: usize = 13;
    const KV_HEADS: u32 = 16;
    const GROUP: u32 = 3;
    const KEY_DIM: usize = 128;
    const ROW_ELEMENTS: usize = KV_HEADS as usize * KEY_DIM;

    let x_data: Vec<f32> = (0..POSITIONS * ROW_ELEMENTS)
        .map(|index| ((index % 89) as f32 + 1.0) * 0.01)
        .collect();

    let build_and_run = |symbol: u64, x_slice: &[f32]| -> Vec<f32> {
        let mut program = Vec::new();
        let x = op::append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![
                    Extent::Symbolic(0),
                    Extent::Static(KV_HEADS),
                    Extent::Static(KEY_DIM as u32)
                ],
                name: Some("x".into()),
            },
        );
        let output = repeat_kv_heads(&mut program, x, KV_HEADS, GROUP)
            .expect("repeat_kv_heads lowers");
        let evaluated =
            crate::cpu::evaluate_named(&program, &[symbol], &[("x", x_slice)], &[output])
                .expect("repeat_kv_heads evaluates");
        let (result, shape) = evaluated.get(output).expect("repeat_kv_heads output present");
        assert_eq!(
            shape,
            [symbol, u64::from(KV_HEADS), u64::from(GROUP), KEY_DIM as u64],
            "repeat_kv_heads output shape"
        );
        result.to_vec()
    };

    let batch_result = build_and_run(POSITIONS as u64, &x_data);
    let output_row_elements = ROW_ELEMENTS * GROUP as usize;

    for position in 0..POSITIONS {
        let row_slice = &x_data[position * ROW_ELEMENTS..(position + 1) * ROW_ELEMENTS];
        let row_result = build_and_run(1, row_slice);
        let batch_row = &batch_result
            [position * output_row_elements..(position + 1) * output_row_elements];
        let max_relative_error = batch_row
            .iter()
            .zip(&row_result)
            .map(|(&batched, &single)| (batched - single).abs() / single.abs().max(1e-6))
            .fold(0.0f32, f32::max);
        std::println!(
            "repeat_kv_heads_row_check position={position} max_relative_error={max_relative_error}"
        );
        assert!(
            max_relative_error <= 1e-5,
            "position {position}: max_relative_error={max_relative_error} exceeds 1e-5"
        );
    }
}

/// proxima-debugger unit oracle (qwen35moe GDN prefill-scan-vs-sequential
/// divergence), the exact chain `query_sequence` itself is defined by
/// (`spec.rs:8698-8717`): [`repeat_kv_heads`] feeding DIRECTLY into the
/// zero-reduced-axis permute `reduce(.., "sugd->sugd", "sdug->sugd")`,
/// requesting ONLY the final node as output so the binder fuses the two
/// exactly as it does in the real graph -- the two prior tests each
/// requested their own op's output directly, which forces
/// materialization and can decline a fusion the real multi-op chain
/// takes. `[13, u, d]` batch vs the same 13 rows one at a time.
#[proxima::test]
async fn repeat_kv_heads_then_permute_reduce_multi_row_matches_single_row() {
    const POSITIONS: usize = 13;
    const KV_HEADS: u32 = 16;
    const GROUP: u32 = 3;
    const KEY_DIM: usize = 128;
    const ROW_ELEMENTS: usize = KV_HEADS as usize * KEY_DIM;

    let x_data: Vec<f32> = (0..POSITIONS * ROW_ELEMENTS)
        .map(|index| ((index % 83) as f32 + 1.0) * 0.01)
        .collect();

    let build_and_run = |symbol: u64, x_slice: &[f32]| -> Vec<f32> {
        let mut program = Vec::new();
        let x = op::append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![
                    Extent::Symbolic(0),
                    Extent::Static(KV_HEADS),
                    Extent::Static(KEY_DIM as u32)
                ],
                name: Some("x".into()),
            },
        );
        let repeated = repeat_kv_heads(&mut program, x, KV_HEADS, GROUP)
            .expect("repeat_kv_heads lowers");
        let output = reduce(
            &mut program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            repeated,
            "sugd->sugd",
            "sdug->sugd",
        )
        .expect("permute reduce lowers");
        let evaluated =
            crate::cpu::evaluate_named(&program, &[symbol], &[("x", x_slice)], &[output])
                .expect("chain evaluates");
        let (result, shape) = evaluated.get(output).expect("chain output present");
        // out_map "sdug->sugd" names the output's own axes `s,d,u,g` in
        // that literal order -- NOT `s,u,g,d` (`repeat_kv_heads`'s own
        // shape); a genuine reduce output-axis permute, confirmed by
        // reading the shape back rather than assumed.
        assert_eq!(
            shape,
            [symbol, KEY_DIM as u64, u64::from(KV_HEADS), u64::from(GROUP)],
            "chain output shape"
        );
        result.to_vec()
    };

    let batch_result = build_and_run(POSITIONS as u64, &x_data);
    let output_row_elements = ROW_ELEMENTS * GROUP as usize;

    for position in 0..POSITIONS {
        let row_slice = &x_data[position * ROW_ELEMENTS..(position + 1) * ROW_ELEMENTS];
        let row_result = build_and_run(1, row_slice);
        let batch_row = &batch_result
            [position * output_row_elements..(position + 1) * output_row_elements];
        let max_relative_error = batch_row
            .iter()
            .zip(&row_result)
            .map(|(&batched, &single)| (batched - single).abs() / single.abs().max(1e-6))
            .fold(0.0f32, f32::max);
        std::println!(
            "repeat_then_permute_row_check position={position} max_relative_error={max_relative_error}"
        );
        assert!(
            max_relative_error <= 1e-5,
            "position {position}: max_relative_error={max_relative_error} exceeds 1e-5"
        );
    }
}

/// proxima-debugger unit oracle (qwen35moe GDN prefill-scan-vs-sequential
/// divergence): the head-split multiply `query_prefill_split` itself is
/// defined by (`spec.rs:8664-8671`) -- a MULTI-TERM read
/// (`"s,{head_k_dim}*u+i->sui"`, decomposing one flat `key_dim` axis
/// into `u,i` via an affine combination, unlike [`repeat_kv_heads`]'s
/// plain per-axis `"sud->sugd"`) against the SAME row-invariant
/// `key_head_ones` donor shape ROW 561 named, chained straight into
/// [`l2norm_with_eps_map`] exactly as the real graph does. `[13,
/// head_k_dim*u]` batch vs the same 13 rows one at a time.
#[proxima::test]
async fn query_prefill_split_then_l2norm_multi_row_matches_single_row() {
    const POSITIONS: usize = 13;
    const KV_HEADS: u32 = 16;
    const HEAD_K_DIM: u32 = 8;
    const KEY_DIM: usize = KV_HEADS as usize * HEAD_K_DIM as usize;

    let x_data: Vec<f32> = (0..POSITIONS * KEY_DIM)
        .map(|index| ((index % 79) as f32 + 1.0) * 0.01)
        .collect();

    let build_and_run = |symbol: u64, x_slice: &[f32]| -> Vec<f32> {
        let mut program = Vec::new();
        let x = op::append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Symbolic(0), Extent::Static(KEY_DIM as u32)],
                name: Some("x".into()),
            },
        );
        let key_head_ones = op::append(
            &mut program,
            Op::Constant {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(KV_HEADS), Extent::Static(HEAD_K_DIM)],
                value: 1.0,
            },
        );
        let split_map = alloc::format!("s,{HEAD_K_DIM}*u+i->sui");
        let split = elementwise(
            &mut program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(x, split_map.as_str()), (key_head_ones, "ui->sui")],
        )
        .expect("head split lowers");
        let eps = symbolic_leaf(&mut program, DType::Float32, "eps");
        let output = l2norm_with_eps_map(&mut program, split, eps, "sui->sui", "su->sui", "s->su")
            .expect("l2norm lowers");
        let eps_data = alloc::vec![1e-6f32; symbol as usize];
        let evaluated = crate::cpu::evaluate_named(
            &program,
            &[symbol],
            &[("x", x_slice), ("eps", &eps_data)],
            &[output],
        )
        .expect("chain evaluates");
        let (result, shape) = evaluated.get(output).expect("chain output present");
        assert_eq!(
            shape,
            [symbol, u64::from(KV_HEADS), u64::from(HEAD_K_DIM)],
            "chain output shape"
        );
        result.to_vec()
    };

    let batch_result = build_and_run(POSITIONS as u64, &x_data);

    for position in 0..POSITIONS {
        let row_slice = &x_data[position * KEY_DIM..(position + 1) * KEY_DIM];
        let row_result = build_and_run(1, row_slice);
        let batch_row = &batch_result[position * KEY_DIM..(position + 1) * KEY_DIM];
        let max_relative_error = batch_row
            .iter()
            .zip(&row_result)
            .map(|(&batched, &single)| (batched - single).abs() / single.abs().max(1e-6))
            .fold(0.0f32, f32::max);
        std::println!(
            "query_prefill_split_row_check position={position} max_relative_error={max_relative_error}"
        );
        assert!(
            max_relative_error <= 1e-5,
            "position {position}: max_relative_error={max_relative_error} exceeds 1e-5"
        );
    }
}

/// proxima-debugger unit oracle (qwen35moe GDN prefill-scan-vs-sequential
/// divergence): [`silu`] at the REAL checkpoint's own width
/// (`qkv_dim = 2*key_dim + value_dim*heads` for `qwen35moe`'s GDN mixer
/// is on the order of `8192`, not the earlier tests' narrow 4/128-wide
/// fixtures) -- an elementwise op's own kernel selection
/// (`run_elementwise_range`'s width-tile path) can differ by SIZE, not
/// just by shape, from the narrower ops already cleared above, so this
/// re-runs the same multi-row-vs-single-row check at production scale.
/// `[13, 8192]` batch vs the same 13 rows one at a time.
#[proxima::test]
async fn silu_multi_row_batch_at_production_width_matches_single_row() {
    const POSITIONS: usize = 13;
    const WIDTH: usize = 8192;

    let x_data: Vec<f32> = (0..POSITIONS * WIDTH)
        .map(|index| (((index % 251) as f32) - 125.0) * 0.037)
        .collect();

    let build_and_run = |symbol: u64, x_slice: &[f32]| -> Vec<f32> {
        let mut program = Vec::new();
        let x = op::append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Symbolic(0), Extent::Static(WIDTH as u32)],
                name: Some("x".into()),
            },
        );
        let one = scalar_constant(&mut program, 1.0);
        let output = silu(&mut program, x, one, "sd->sd").expect("silu lowers");
        let evaluated =
            crate::cpu::evaluate_named(&program, &[symbol], &[("x", x_slice)], &[output])
                .expect("silu evaluates");
        let (result, shape) = evaluated.get(output).expect("silu output present");
        assert_eq!(shape, [symbol, WIDTH as u64], "silu output shape");
        result.to_vec()
    };

    let batch_result = build_and_run(POSITIONS as u64, &x_data);

    for position in 0..POSITIONS {
        let row_slice = &x_data[position * WIDTH..(position + 1) * WIDTH];
        let row_result = build_and_run(1, row_slice);
        let batch_row = &batch_result[position * WIDTH..(position + 1) * WIDTH];
        let max_relative_error = batch_row
            .iter()
            .zip(&row_result)
            .map(|(&batched, &single)| (batched - single).abs() / single.abs().max(1e-6))
            .fold(0.0f32, f32::max);
        std::println!(
            "silu_row_check position={position} max_relative_error={max_relative_error}"
        );
        assert!(
            max_relative_error <= 1e-5,
            "position {position}: max_relative_error={max_relative_error} exceeds 1e-5"
        );
    }
}

/// **The rule census.** For the real Mistral/OpenChat cached-forward
/// program, names every rewrite [`crate::bind::bind`] actually applies
/// and counts how many times each fired, reconciling against
/// `docs/discipline.md`'s own measured split (ROW at line 4780): 1196
/// `BoundOp`s, 225 `reduce_matmul_quantized`, 385 `reduce_f32_dense`,
/// 547 `elementwise`, 37 `constant`, 2 `iota`.
///
/// The 225/385 split within `docs/discipline.md`'s own figure is a
/// RUNTIME classification (`cpu::is_quantized_matmul_operand`'s own
/// discriminator: whether a bound weight buffer's byte length matches
/// its declared `Float32` element count, or is smaller because the
/// checkpoint actually stored it `Q4_K`/`Q5_K`/`Q6_K` packed) -- it is
/// not recoverable from this symbolic program alone, which declares
/// every weight `DType::Float32` (`mistral_cached_forward_program_with_
/// experts`, `spec.rs:4561-4646` and onward: every `input_leaf` weight
/// call passes `DType::Float32`, never a quantized tag). Reproducing it
/// here would require binding the real `openchat-3.5-1210.Q4_K_S.gguf`
/// checkpoint's own weight bytes, out of this round's scope. What IS a
/// structural (graph-topology) property, checked below by an inline
/// mirror of `cpu::reduce_is_gemm_shaped`'s own distinct-operand-count
/// discriminator (a `#[cfg]`-gated private fn, not reachable across this
/// module boundary without changing its visibility): whether a reduce
/// reads one operand (LayerNorm mean/variance, softmax max/sum) or two
/// (every matmul-shaped fold, weight-projection AND attention-score
/// alike) -- a DIFFERENT, coarser partition than quantized/dense, since
/// attention's own `Q@K^T`/`softmax@V`/cached-score folds are also
/// two-operand but never weight-quantized. Measured 417 two-operand /
/// 193 one-operand, summing to the same 610 total the 225+385 figure
/// does -- the total reconciles exactly; the sub-split names a
/// different, real distinction, not the same one.
///
/// Also reads back the two elementwise-fusion decisions inline inside
/// `BoundOpBuilder::push` (`bind.rs:641`, `bind.rs:697`) and the
/// masked-window-reduce elimination (`bind.rs:659`) under the
/// `instrument` feature -- fired/declined, never a bare bool, so a rule
/// that fires zero times on this program is visibly distinct from a
/// rule that does not exist.
#[cfg(feature = "instrument")]
#[test]
fn the_rule_census_reconciles_against_the_measured_mistral_forward_split() {
    crate::instrument::reset_fuse_decline();
    crate::instrument::reset_window_reduce();
    crate::instrument::reset_online_softmax_block_range();

    let (program, logits, roots) =
        mistral_cached_forward_program(32_002, 4096, 14336, 32, 8, 128, 32)
            .expect("the cached forward pass lowers to a program");
    let mut outputs = alloc::vec![logits];
    for (even, odd, value) in &roots {
        outputs.extend_from_slice(&[*even, *odd, *value]);
    }
    let shapes = crate::shape::infer(&program, &[1, 71])
        .expect("one new position against a 71-position cache infers");
    // fusion held explicitly off: this census documents the UNFUSED
    // split, so it must bind that way under every feature combination,
    // including `cached-attention-streaming`, where `bind`'s own
    // default (`bind_with_fusion(.., true)`) would otherwise fuse 32
    // attention chains into `BoundOpKind::CachedAttention` and silently
    // invalidate every count below.
    let bound = crate::bind::bind_with_fusion(
        &program,
        &shapes,
        &outputs,
        false,
        crate::numeric::NumericPolicy::default(),
    )
    .expect("the program binds");
    // `reduce-epilogue-fusion` is a bind-time REWRITE gated only by this
    // crate feature (`bind::bind_with_fusion`'s own doc: it "runs
    // unconditionally after this ... gated only by the crate feature"),
    // not by the `fuse_cached_attention` bool this call passes `false`
    // for -- so `bound` above already carries the epilogue-fused shape
    // whenever this feature is compiled in, and every count below that
    // depends on it must be read per-feature, never pinned to one number.
    #[cfg(feature = "reduce-epilogue-fusion")]
    let epilogued_reduce_count = bound
        .iter()
        .filter(|op| {
            matches!(
                &op.kind,
                crate::bind::BoundOpKind::Reduce { epilogue_operands, .. }
                    if !epilogue_operands.is_empty()
            )
        })
        .count();
    // MEASURED (this test, `reduce-epilogue-fusion` on): 128 = 4 fusions
    // x 32 layers, one per `append_mistral_cached_layer` call
    // (`spec.rs:2378`). Each fusion is an `Op::Elementwise` whose SOLE
    // operand-of-interest is an `Op::Reduce` with no other consumer, read
    // at full identity -- exactly `bind::reduce_epilogue_candidates`'s
    // own three conditions -- so the reduce's own `BoundOp` disappears
    // and its producer becomes the consumer's epilogue instead. The four,
    // in per-layer source order:
    //   1. `global_max` (`spec.rs:2678`, `Elementwise::Maximum` over
    //      `score_max_cached`/`score_max_new`) absorbs `score_max_cached`
    //      (the first, hence first-matching, `Reduce` operand) as its
    //      epilogue -- the online-softmax running-max combine.
    //   2. `residual1` (`spec.rs:2809`, `Elementwise::Add` of `attn_out`
    //      and `x`) absorbs `attn_out`, the attention output-projection
    //      reduce (`spec.rs:2799`).
    //   3. `ffn_hidden` (`spec.rs:2879`, `Elementwise::Multiply` of
    //      `silu_gate` and `up`) absorbs `up`, the FFN up-projection
    //      reduce (`spec.rs:2839`).
    //   4. `x_next` (`spec.rs:2902`, `Elementwise::Add` of `ffn_out` and
    //      `residual1`) absorbs `ffn_out`, the FFN down-projection
    //      reduce (`spec.rs:2892`).
    // Every other `Reduce` in the layer (Q/K/V projections, the two
    // per-range attention-score reduces, the two per-range softmax-sum
    // reduces, the two per-range attended-value reduces) keeps a second
    // real consumer or a non-identity/broadcast one, so none of them
    // qualifies -- this is why the count is 4/layer, not higher.
    #[cfg(feature = "reduce-epilogue-fusion")]
    assert_eq!(
        epilogued_reduce_count,
        4 * 32,
        "reduce-epilogue-fusion must absorb exactly 4 reduces per layer on this \
         32-layer Mistral cached-forward program -- global_max, residual1's attn_out, \
         ffn_hidden's up-projection, and x_next's down-projection reduce"
    );

    let mut elementwise = 0_usize;
    let mut reduce_two_operand = 0_usize;
    let mut reduce_one_operand = 0_usize;
    let mut constant = 0_usize;
    let mut iota = 0_usize;
    // `BoundOpKind::CachedAttention` (landed after this census's own
    // baseline figures were measured) is a bind-time fusion held off
    // explicitly above via `bind_with_fusion(.., false)` -- this
    // program never reaches it here regardless of feature set, so it
    // is counted separately as a proof the unfused bind stayed
    // unfused, not merely a side effect of a feature flag being off.
    // The feature-gated block below re-binds WITH fusion on to assert
    // the fused count directly.
    let mut cached_attention = 0_usize;
    for op in &bound {
        match &op.kind {
            crate::bind::BoundOpKind::CachedAttention { .. } => cached_attention += 1,
            crate::bind::BoundOpKind::Elementwise { .. } => elementwise += 1,
            crate::bind::BoundOpKind::Reduce { .. } => {
                let operands = op.operands();
                let is_two_operand = operands.first().is_some_and(|(first, _, _)| {
                    operands.iter().any(|(node, _, _)| node != first)
                });
                if is_two_operand {
                    reduce_two_operand += 1;
                } else {
                    reduce_one_operand += 1;
                }
            }
            crate::bind::BoundOpKind::Constant { .. } => constant += 1,
            crate::bind::BoundOpKind::Iota => iota += 1,
            // This program never binds a Qwen3.5 GDN mixer, so a
            // `gated-delta-net-fusion` build never produces this kind
            // here -- an arm is still required once the variant exists
            // regardless of which program a given test walks.
            crate::bind::BoundOpKind::GatedDeltaNet { .. } => {
                panic!("this Mistral cached-forward program never binds a GatedDeltaNet op")
            }
            // This program has no MoE routing block at all (Mistral's
            // own dense FFN), so a `moe-topk-fusion` build never
            // produces this kind here either.
            crate::bind::BoundOpKind::MoeTopK { .. } => {
                panic!("this Mistral cached-forward program never binds a MoeTopK op")
            }
        }
    }
    assert_eq!(
        cached_attention, 0,
        "this program/feature-set was expected to never fuse a CachedAttention BoundOp -- \
         the census's four-bucket reconciliation below needs updating if that changed"
    );
    let total = bound.len();
    let reduce_total = reduce_two_operand + reduce_one_operand;

    let (fuse_elementwise_fired, fuse_reduce_fired, fuse_distinct_declines) =
        crate::instrument::fuse_totals();
    let (window_reduce_fired, window_reduce_declined) =
        crate::instrument::window_reduce_totals();

    std::println!(
        "rule_census total={total} reduce_total={reduce_total} reduce_two_operand={reduce_two_operand} reduce_one_operand={reduce_one_operand} elementwise={elementwise} constant={constant} iota={iota}"
    );
    std::println!(
        "rule_census fuse_elementwise_operand_fired={fuse_elementwise_fired} fuse_reduce_operand_fired={fuse_reduce_fired} fuse_distinct_declines={fuse_distinct_declines} window_reduce_fired={window_reduce_fired} window_reduce_declined={window_reduce_declined}"
    );

    // costing-vs-free split (2026-09-02 correction, verified against
    // source): `Op::Input` (`bind.rs:607`, `push`'s own match arm)
    // emits NOTHING -- no `BoundOp`, no dispatch, confirmed by this
    // module's own doc ("`Op::Input` never does -- it is where data
    // enters"). `Op::Constant`/`Op::Iota` DO have a real `push` arm
    // (`bind.rs:621`, and the `Iota` arm above it) and this census's
    // own total already proves both dispatch: `constant=37 iota=2` are
    // real `BoundOp`s. So FREE is `Op::Input` alone; COSTING is
    // `Elementwise`, `Reduce`, `Constant`, or `Iota` -- a decline whose
    // operand is any of those corresponds to a materialized, dispatched
    // buffer, exactly like an `Elementwise` decline does.
    let is_free_operand = |node: u32| matches!(program[node as usize], Op::Input { .. });

    let mut still_live_costing = 0_u64;
    let mut still_live_free = 0_u64;
    let mut non_identity_costing = 0_u64;
    let mut non_identity_free = 0_u64;
    let mut not_held_costing = 0_u64;
    let mut not_held_free = 0_u64;
    for (node, _site, reason, calls) in crate::instrument::fuse_decline_snapshot() {
        let free = is_free_operand(node);
        match (reason, free) {
            (crate::instrument::FuseDeclineReason::StillLive, true) => still_live_free += calls,
            (crate::instrument::FuseDeclineReason::StillLive, false) => {
                still_live_costing += calls;
            }
            (crate::instrument::FuseDeclineReason::NonIdentityProjection, true) => {
                non_identity_free += calls;
            }
            (crate::instrument::FuseDeclineReason::NonIdentityProjection, false) => {
                non_identity_costing += calls;
            }
            (crate::instrument::FuseDeclineReason::NotHeld, true) => not_held_free += calls,
            (crate::instrument::FuseDeclineReason::NotHeld, false) => {
                not_held_costing += calls;
            }
            // `quarantine_broadcast_operands` only ever walks children
            // still `held` (`bind.rs:895`'s own `contains_key` guard),
            // and a `held` node is by construction an `Op::Elementwise`
            // chain, never an `Op::Input` leaf -- so this reason is
            // always costing, counted separately below rather than
            // folded into the free/costing split above.
            (crate::instrument::FuseDeclineReason::BroadcastQuarantined, _) => {}
        }
    }
    let still_live_total = still_live_costing + still_live_free;
    let non_identity_total = non_identity_costing + non_identity_free;
    let not_held_total = not_held_costing + not_held_free;
    std::println!(
        "rule_census fuse_decline_still_live={still_live_total} costing={still_live_costing} free={still_live_free}"
    );
    std::println!(
        "rule_census fuse_decline_non_identity_projection={non_identity_total} costing={non_identity_costing} free={non_identity_free}"
    );
    std::println!(
        "rule_census fuse_decline_not_held={not_held_total} costing={not_held_costing} free={not_held_free}"
    );

    // quarantine-broadcast census (this round's task): `bind.rs:901`'s
    // `quarantine_broadcast_operands` is a genuine fuse/no-fuse decision
    // invisible to the counters above -- a decline here looks identical
    // to a rule that never ran without its own site. `N == 0` is a red
    // gate the same way `total > 0` below is.
    let mut quarantine_broadcast_declines = 0_u64;
    for (_node, site, reason, calls) in crate::instrument::fuse_decline_snapshot() {
        if site == crate::instrument::FuseSite::QuarantineBroadcast
            && reason == crate::instrument::FuseDeclineReason::BroadcastQuarantined
        {
            quarantine_broadcast_declines += calls;
        }
    }
    std::println!("rule_census quarantine_broadcast_declines={quarantine_broadcast_declines}");
    assert!(
        quarantine_broadcast_declines > 0,
        "rule census recorded zero quarantine-broadcast declines -- the site is wired to nothing"
    );
    assert_eq!(
        quarantine_broadcast_declines, 65,
        "quarantine-broadcast declines drifted off the measured Mistral cached-forward count"
    );

    // rope_cos/rope_sin named check: both are `Op::Input` (node 7, 8),
    // so under the corrected rule they must land 100% free, on every
    // reason, not just `StillLive`.
    for (node, label) in [(7_u32, "rope_cos"), (8, "rope_sin")] {
        let mut costing = 0_u64;
        let mut free = 0_u64;
        for (decline_node, _site, _reason, calls) in crate::instrument::fuse_decline_snapshot()
        {
            if decline_node != node {
                continue;
            }
            if is_free_operand(decline_node) {
                free += calls;
            } else {
                costing += calls;
            }
        }
        std::println!(
            "rule_census named_node_check node={node} label={label} costing={costing} free={free}"
        );
        assert_eq!(
            costing, 0,
            "{label} (node {node}) is an Op::Input leaf -- every one of its declines must be free"
        );
    }

    // deliverable #4: is `non_identity_projection=129` here the SAME
    // shape class as `width_tile_plan`'s `AxesShape=129` on node 90 in
    // the CPU train lane (`cpu.rs:9940`), or a bare numeric coincidence?
    // These are declines from THIS test's own bind-time fusion check
    // (`bind.rs:641`/`:697`), a different mechanism on a different
    // program (Mistral cached-forward here, BGE there) -- print which
    // node(s)/op(s) actually produce the 129 here so the comparison can
    // be made on evidence, not on the number alone.
    let mut non_identity_nodes: alloc::vec::Vec<(u32, u64)> = alloc::vec::Vec::new();
    for (node, _site, reason, calls) in crate::instrument::fuse_decline_snapshot() {
        if reason == crate::instrument::FuseDeclineReason::NonIdentityProjection {
            non_identity_nodes.push((node, calls));
        }
    }
    non_identity_nodes.sort_by_key(|entry| core::cmp::Reverse(entry.1));
    for (node, calls) in non_identity_nodes.iter().take(5) {
        let op = &program[*node as usize];
        std::println!(
            "rule_census non_identity_projection_node node={node} calls={calls} op={op:?}"
        );
    }
    std::println!(
        "rule_census non_identity_projection_distinct_nodes={}",
        non_identity_nodes.len()
    );

    // WHICH ops, not just how many (2026-09-02 follow-up): `fuse_decline_
    // snapshot` is now keyed by the STILL-LIVE OPERAND's own `NodeId`
    // (bind.rs's own fix -- it previously keyed by the consuming node,
    // which names WHERE a decline was checked, not WHAT had to
    // materialize). `online_softmax_block_ranges` brackets the combine
    // block (`spec.rs:2596-2726`) by construction -- `score_max_cached`
    // and `attended`, the block's own first/last emitted `NodeId`s,
    // recorded once per `append_mistral_cached_layer` call, never
    // assumed from reading the source alone.
    let block_ranges = crate::instrument::online_softmax_block_ranges();
    assert_eq!(
        block_ranges.len(),
        32,
        "one online-softmax block range per layer on a 32-layer forward"
    );
    let stride = block_ranges[1].0 - block_ranges[0].0;
    for window in block_ranges.windows(2) {
        assert_eq!(
            window[1].0 - window[0].0,
            stride,
            "every layer's block must start the same distance from the \
             previous layer's -- confirms structural periodicity by \
             measurement rather than assuming it from the source"
        );
        assert_eq!(
            window[0].1 - window[0].0,
            window[1].1 - window[1].0,
            "every layer's block must span the same number of nodes"
        );
    }
    let block_span = block_ranges[0].1 - block_ranges[0].0;
    std::println!(
        "rule_census online_softmax_block layers=32 stride={stride} block_span_nodes={} first_layer_range=[{},{}]",
        block_span + 1,
        block_ranges[0].0,
        block_ranges[0].1
    );

    let in_any_block = |node: u32| {
        block_ranges
            .iter()
            .any(|&(first, last)| node >= first && node <= last)
    };

    let mut still_live_costing_in_block = 0_u64;
    let mut still_live_costing_out_of_block = 0_u64;
    let mut still_live_free_in_block = 0_u64;
    let mut still_live_free_out_of_block = 0_u64;
    // phase = this operand's `NodeId` distance from ITS OWN layer's
    // block start, `rem_euclid(stride)` folding every layer onto one
    // canonical 0..stride ruler -- the "position within a layer" the
    // task asked for, measured against the real per-layer stride rather
    // than the raw-Op approximation. Split costing/free so a free
    // (`Op::Input`) repeat-offender like `rope_cos`/`rope_sin` cannot
    // hide inside the same ranking as a real materialize cost.
    let mut costing_phase_totals: alloc::collections::BTreeMap<u32, u64> =
        alloc::collections::BTreeMap::new();
    let mut costing_phase_example_node: alloc::collections::BTreeMap<u32, u32> =
        alloc::collections::BTreeMap::new();
    for (node, _site, reason, calls) in crate::instrument::fuse_decline_snapshot() {
        if reason != crate::instrument::FuseDeclineReason::StillLive {
            continue;
        }
        let free = is_free_operand(node);
        let inside = in_any_block(node);
        match (free, inside) {
            (true, true) => still_live_free_in_block += calls,
            (true, false) => still_live_free_out_of_block += calls,
            (false, true) => still_live_costing_in_block += calls,
            (false, false) => still_live_costing_out_of_block += calls,
        }
        if !free {
            let phase = (node.wrapping_sub(block_ranges[0].0)).rem_euclid(stride);
            *costing_phase_totals.entry(phase).or_insert(0) += calls;
            costing_phase_example_node.entry(phase).or_insert(node);
        }
    }
    let still_live_costing_total =
        still_live_costing_in_block + still_live_costing_out_of_block;
    std::println!(
        "rule_census still_live_costing_in_block={still_live_costing_in_block} still_live_costing_outside_block={still_live_costing_out_of_block} still_live_costing_total={still_live_costing_total}"
    );
    let still_live_costing_per_layer = still_live_costing_total as f64 / 32.0;
    std::println!(
        "rule_census still_live_costing_per_layer={still_live_costing_per_layer:.2} layers=32 program_wide={still_live_costing_total}"
    );
    std::println!(
        "rule_census still_live_free_in_block={still_live_free_in_block} still_live_free_outside_block={still_live_free_out_of_block}"
    );
    assert!(
        still_live_costing_total > 0,
        "rule census recorded zero costing still-live declines -- the split is wired to nothing"
    );
    // deliverable answer: what fraction of the COSTING total is the
    // online-softmax block's 96 (all three of global_max/weights_cached/
    // weights_new are Elementwise, so all 96 are costing by construction
    // -- confirmed below, not assumed).
    let block_fraction_permille = still_live_costing_in_block * 1000 / still_live_costing_total;
    std::println!(
        "rule_census online_softmax_block_share_of_costing calls={still_live_costing_in_block} of={still_live_costing_total} permille={block_fraction_permille}"
    );

    // named by construction order within the block (score_max_cached is
    // phase 0, the block's own first node): global_max is the 3rd node
    // emitted (phase 2), weights_cached the 5th (phase 4), weights_new
    // the 7th (phase 6) -- `spec.rs:2616,2632,2644`, read directly off
    // this test's own doc trace of the block, not guessed. All three
    // are `Op::Elementwise`, so they land in `costing_phase_totals`.
    for (phase, label) in [
        (2_u32, "global_max"),
        (4, "weights_cached"),
        (6, "weights_new"),
    ] {
        let calls = costing_phase_totals.get(&phase).copied().unwrap_or(0);
        std::println!(
            "rule_census still_live_costing_phase={phase} label={label} calls={calls}"
        );
    }

    let mut ranked_costing_phases: alloc::vec::Vec<(u32, u64)> =
        costing_phase_totals.into_iter().collect();
    ranked_costing_phases.sort_by_key(|entry| core::cmp::Reverse(entry.1));
    for (phase, calls) in ranked_costing_phases.iter().take(10) {
        std::println!("rule_census still_live_costing_top_phase phase={phase} calls={calls}");
    }
    for (phase, calls) in ranked_costing_phases.iter().take(6) {
        let node = costing_phase_example_node.get(phase).copied().unwrap_or(0);
        let op = &program[node as usize];
        std::println!(
            "rule_census still_live_costing_top_phase_op phase={phase} calls={calls} node={node} op={op:?}"
        );
    }

    // sanity check: 1196 = 610 reduce + 547 elementwise + 37 constant +
    // 2 iota. A costing decline whose operand is `Op::Elementwise`
    // corresponds to a node that either fuses away for free (never a
    // `BoundOp`) or is forced to materialize as one of the 547
    // elementwise `BoundOp`s -- the distinct count of Elementwise-kind
    // nodes that appear ANYWHERE in the decline snapshot (any of the
    // three reasons) is the direct witness for "forced to materialize
    // at least once", checked against 547 rather than assumed to agree
    // with it.
    let mut raw_elementwise_total = 0_u64;
    let mut raw_reduce_total = 0_u64;
    let mut raw_constant_total = 0_u64;
    let mut raw_iota_total = 0_u64;
    let mut raw_input_total = 0_u64;
    for op in &program {
        match op {
            Op::Elementwise { .. } => raw_elementwise_total += 1,
            Op::Reduce(_) => raw_reduce_total += 1,
            Op::Constant { .. } => raw_constant_total += 1,
            Op::Iota { .. } => raw_iota_total += 1,
            Op::Input { .. } => raw_input_total += 1,
        }
    }
    std::println!(
        "rule_census raw_op_totals elementwise={raw_elementwise_total} reduce={raw_reduce_total} constant={raw_constant_total} iota={raw_iota_total} input={raw_input_total} program_len={}",
        program.len()
    );
    assert_eq!(
        raw_reduce_total, 610,
        "reduces never fuse (build_reduce_op always yields exactly one BoundOp), \
         so the raw Op::Reduce count must equal the bound reduce_total exactly"
    );

    let mut elementwise_declined_nodes: alloc::collections::BTreeSet<u32> =
        alloc::collections::BTreeSet::new();
    for (node, _site, _reason, _calls) in crate::instrument::fuse_decline_snapshot() {
        if matches!(program[node as usize], Op::Elementwise { .. }) {
            elementwise_declined_nodes.insert(node);
        }
    }
    std::println!(
        "rule_census elementwise_declined_distinct_nodes={} elementwise_bound_ops=547 raw_elementwise_total={raw_elementwise_total}",
        elementwise_declined_nodes.len()
    );

    // REMATERIALIZATION sizing (2026-09-02, coordinator's rule -- NOT
    // implemented this round, only sized). Consumer count is computed
    // by a full scan of `program` (every place a NodeId is read as an
    // operand), independent of the decline snapshot -- the snapshot
    // only records DECLINE events, not total readers, so it cannot
    // answer "how many consumers" on its own.
    let mut consumer_count: alloc::collections::BTreeMap<u32, u64> =
        alloc::collections::BTreeMap::new();
    let count_map_indices =
        |map: &IndexMap, counts: &mut alloc::collections::BTreeMap<u32, u64>| {
            if let IndexMap::Computed { indices, .. } = map {
                *counts.entry(indices.0).or_insert(0) += 1;
            }
        };
    for op in &program {
        match op {
            Op::Elementwise { operands, .. } => {
                for (operand_node, map) in operands {
                    *consumer_count.entry(operand_node.0).or_insert(0) += 1;
                    count_map_indices(map, &mut consumer_count);
                }
            }
            Op::Reduce(reduce) => {
                *consumer_count.entry(reduce.operand.0).or_insert(0) += 1;
                count_map_indices(&reduce.in_map, &mut consumer_count);
                count_map_indices(&reduce.out_map, &mut consumer_count);
            }
            Op::Input { .. } | Op::Constant { .. } | Op::Iota { .. } => {}
        }
    }

    // deliverable #1: of the 1086 costing still_live declines, how many
    // have an Op::Elementwise operand (the only rematerialization
    // candidate -- a Reduce always dispatches regardless of fusion, so
    // recomputing one buys nothing). A node that is ITSELF a named
    // graph output is excluded: `live::annotate` never retires an
    // output (it must persist to the end regardless of any single
    // consumer), so it declines StillLive even with as few as one
    // real consumer -- and rematerializing it into that consumer would
    // still leave the required output buffer unmaterialized, so it is
    // not a real candidate, not an undercounted one.
    let output_node_set: alloc::collections::BTreeSet<u32> =
        outputs.iter().map(|node| node.0).collect();
    let mut still_live_elementwise_candidates: alloc::collections::BTreeSet<u32> =
        alloc::collections::BTreeSet::new();
    let mut still_live_elementwise_calls = 0_u64;
    let mut still_live_non_elementwise_costing_calls = 0_u64;
    let mut still_live_output_pinned_calls = 0_u64;
    for (node, _site, reason, calls) in crate::instrument::fuse_decline_snapshot() {
        if reason != crate::instrument::FuseDeclineReason::StillLive || is_free_operand(node) {
            continue;
        }
        if !matches!(program[node as usize], Op::Elementwise { .. }) {
            still_live_non_elementwise_costing_calls += calls;
            continue;
        }
        if output_node_set.contains(&node) {
            still_live_output_pinned_calls += calls;
            continue;
        }
        still_live_elementwise_candidates.insert(node);
        still_live_elementwise_calls += calls;
    }
    std::println!(
        "rule_census rematerialize_candidates distinct_nodes={} decline_calls={still_live_elementwise_calls} non_elementwise_costing_calls={still_live_non_elementwise_costing_calls} output_pinned_calls={still_live_output_pinned_calls} of_costing_still_live={still_live_costing_total}",
        still_live_elementwise_candidates.len()
    );
    assert!(
        !still_live_elementwise_candidates.is_empty(),
        "rule census found zero rematerialization candidates -- either the rule genuinely \
         does not apply here or the measurement is wired to nothing"
    );

    // deliverable #2: consumer-count histogram over the candidates.
    let mut consumer_histogram: alloc::collections::BTreeMap<u64, u64> =
        alloc::collections::BTreeMap::new();
    for &node in &still_live_elementwise_candidates {
        let count = consumer_count.get(&node).copied().unwrap_or(0);
        *consumer_histogram.entry(count).or_insert(0) += 1;
    }
    for (consumers, nodes) in &consumer_histogram {
        std::println!(
            "rule_census rematerialize_histogram consumers={consumers} nodes={nodes}"
        );
    }
    assert!(
        consumer_histogram.keys().all(|&count| count >= 2),
        "a StillLive decline means another consumer exists later -- every candidate must \
         show at least 2 total consumers, or the consumer-count scan disagrees with the \
         decline mechanism itself"
    );

    // deliverable #3: projected dispatch saving at three thresholds.
    for threshold in [2_u64, 3, 4] {
        let saved = still_live_elementwise_candidates
            .iter()
            .filter(|&&node| consumer_count.get(&node).copied().unwrap_or(0) <= threshold)
            .count();
        let projected_elementwise = elementwise - saved;
        let projected_total = total - saved;
        std::println!(
            "rule_census rematerialize_projection threshold={threshold} saved_dispatches={saved} elementwise_547_to={projected_elementwise} total_1196_to={projected_total} fraction_of_1196={:.4}",
            saved as f64 / total as f64
        );
    }

    // deliverable #4: the ALU cost side, honest and unrounded. Element
    // counts come from `shapes` (the same `Shapes` table `bind::bind`
    // itself resolved against), never guessed from rank alone.
    let mut recompute_elements_total = 0_u128;
    let mut saved_dispatches_at_2 = 0_u64;
    for &node in &still_live_elementwise_candidates {
        let consumers = consumer_count.get(&node).copied().unwrap_or(0);
        if consumers > 2 {
            continue;
        }
        saved_dispatches_at_2 += 1;
        let extents = shapes.of(crate::op::NodeId(node));
        let element_count: u128 = extents.iter().map(|&extent| extent as u128).product();
        recompute_elements_total += element_count * u128::from(consumers - 1);
    }
    let dispatch_floor_ns = 4_000_u128; // coordinator's own cited ~4us floor
    let dispatch_saving_ns = u128::from(saved_dispatches_at_2) * dispatch_floor_ns;
    // MEASURED range from `docs/discipline.md`'s own instrument-counter
    // table (elementwise Generic fast=2.31 ns/element, slow=16.18
    // ns/element, a real decode step) -- reported as a range, not a
    // single assumed constant, because which arm a rematerialized body
    // would hit is not measured by this census.
    let alu_cost_fast_ns = recompute_elements_total * 231 / 100;
    let alu_cost_slow_ns = recompute_elements_total * 1618 / 100;
    std::println!(
        "rule_census rematerialize_alu_cost threshold=2 saved_dispatches={saved_dispatches_at_2} dispatch_saving_ns={dispatch_saving_ns} recompute_elements={recompute_elements_total} alu_cost_ns_fast_path={alu_cost_fast_ns} alu_cost_ns_slow_path={alu_cost_slow_ns}"
    );
    assert!(
        recompute_elements_total > 0,
        "rule census found rematerialization candidates but zero recompute-element cost -- \
         the shape lookup is wired to nothing"
    );

    // deliverable #5: verify the 547-482=65 gap is exactly the
    // elementwise-kind nodes in `effective_outputs` (never read as an
    // operand by anything else in `push`, so no decline event exists
    // for them, yet they still materialize as named outputs).
    let materialized_elementwise_nodes: alloc::collections::BTreeSet<u32> = bound
        .iter()
        .filter(|op| matches!(op.kind, crate::bind::BoundOpKind::Elementwise { .. }))
        .map(|op| op.node.0)
        .collect();
    // 419 = 547 - 128: the same 128 reduce-epilogue-fusion absorptions
    // asserted above remove one `BoundOpKind::Elementwise` per fusion --
    // the consumer that used to materialize on its own now IS the
    // epilogued `Reduce`, so it drops out of this `Elementwise`-kind
    // filter entirely.
    #[cfg(not(feature = "reduce-epilogue-fusion"))]
    assert_eq!(
        materialized_elementwise_nodes.len(),
        547,
        "the materialized-elementwise set must have exactly 547 members, matching the bound count"
    );
    #[cfg(feature = "reduce-epilogue-fusion")]
    assert_eq!(
        materialized_elementwise_nodes.len(),
        547 - 4 * 32,
        "419 = 547 unfused elementwise BoundOps minus the 128 reduce-epilogue-fusion \
         absorptions (4/layer x 32 layers) -- see epilogued_reduce_count's own doc above"
    );
    let unexplained_nodes: alloc::vec::Vec<u32> = materialized_elementwise_nodes
        .difference(&elementwise_declined_nodes)
        .copied()
        .collect();
    let output_node_set: alloc::collections::BTreeSet<u32> =
        outputs.iter().map(|node| node.0).collect();
    let unexplained_that_are_outputs = unexplained_nodes
        .iter()
        .filter(|node| output_node_set.contains(node))
        .count();
    let unexplained_that_are_not_outputs: alloc::vec::Vec<u32> = unexplained_nodes
        .iter()
        .filter(|node| !output_node_set.contains(node))
        .copied()
        .collect();
    std::println!(
        "rule_census unexplained_gap total={} outputs={unexplained_that_are_outputs} not_outputs={}",
        unexplained_nodes.len(),
        unexplained_that_are_not_outputs.len()
    );
    for node in unexplained_that_are_not_outputs.iter().take(5) {
        let op = &program[*node as usize];
        std::println!("rule_census unexplained_gap_node node={node} op={op:?}");
    }

    // N == 0 is a red gate, not a quiet pass: every rule this census
    // names either fired or declined at least once on a real 32-layer
    // forward, or the census measured nothing and must fail loudly.
    assert!(total > 0, "rule census processed zero bound ops");
    assert!(
        fuse_elementwise_fired + fuse_reduce_fired > 0,
        "rule census recorded zero fusion firings -- the counters are wired to nothing"
    );
    assert!(
        window_reduce_fired + window_reduce_declined > 0,
        "rule census recorded zero window-reduce attempts -- the counter is wired to nothing"
    );

    // `total` and `elementwise` both shift by exactly the 128
    // reduce-epilogue-fusion absorptions under that feature; every
    // fusion removes one whole `BoundOp` (the standalone `Reduce`
    // disappears, its consumer's `Elementwise` slot is repurposed as the
    // SAME `Reduce`'s epilogue rather than adding a new entry) --
    // `reduce_total`/`constant`/`iota` are untouched because the fused
    // reduce keeps its `BoundOpKind::Reduce` kind, just gains a
    // non-default `epilogue_body`.
    // 1195, not 1196: ROW 541 (`docs/discipline.md`) made `bind_plain`
    // bind only what `outputs` reaches. This program carries exactly one
    // `Op::Constant` no `outputs` entry reads (see `constant`'s own
    // count below, 36 not 37) -- previously bound and left as dead
    // weight in `bound`, now never bound at all.
    #[cfg(not(feature = "reduce-epilogue-fusion"))]
    assert_eq!(
        total, 1195,
        "total BoundOps must match the measured forward"
    );
    #[cfg(feature = "reduce-epilogue-fusion")]
    assert_eq!(
        total,
        1195 - 4 * 32,
        "1067 = 1195 unfused total minus the 128 reduce-epilogue-fusion absorptions"
    );
    assert_eq!(
        reduce_total,
        225 + 385,
        "total reduces must match the measured reduce_matmul_quantized + \
         reduce_f32_dense population, even though this test's own two-operand \
         split is a different partition of that same 610 (see this test's own doc); \
         unaffected by reduce-epilogue-fusion -- an absorbed reduce keeps its \
         BoundOpKind::Reduce kind, it only gains a non-default epilogue"
    );
    #[cfg(not(feature = "reduce-epilogue-fusion"))]
    assert_eq!(
        elementwise, 547,
        "elementwise BoundOps must match the measured forward"
    );
    #[cfg(feature = "reduce-epilogue-fusion")]
    assert_eq!(
        elementwise,
        547 - 4 * 32,
        "419 = 547 unfused elementwise BoundOps minus the 128 reduce-epilogue-fusion \
         absorptions (4/layer x 32 layers) -- see epilogued_reduce_count's own doc above"
    );
    // 36, not 37: the one `Op::Constant` ROW 541's reachability pass
    // (`bind_plain`, `docs/discipline.md`) no longer binds -- see
    // `total`'s own comment above.
    assert_eq!(
        constant, 36,
        "constant BoundOps must match the measured forward"
    );
    assert_eq!(iota, 2, "iota BoundOps must match the measured forward");
    assert_eq!(
        reduce_total + elementwise + constant + iota,
        total,
        "the four BoundOpKind buckets must exhaust the total with no remainder"
    );

    // fused count, measured directly rather than assumed: re-bind the
    // SAME program WITH `cached_attention_candidates` fusion turned on
    // (`bind_with_fusion(.., true)`, `bind.rs:2338`) so this census also
    // states what the fused split looks like under
    // `cached-attention-streaming`, instead of only proving the unfused
    // split held.
    #[cfg(feature = "cached-attention-streaming")]
    {
        let fused = crate::bind::bind_with_fusion(
            &program,
            &shapes,
            &outputs,
            true,
            crate::numeric::NumericPolicy::default(),
        )
        .expect("the program binds with fusion enabled");
        let fused_cached_attention = fused
            .iter()
            .filter(|op| matches!(op.kind, crate::bind::BoundOpKind::CachedAttention { .. }))
            .count();
        std::println!(
            "rule_census fused_total={} fused_cached_attention={fused_cached_attention}",
            fused.len()
        );
        assert_eq!(
            fused_cached_attention, 32,
            "one CachedAttention BoundOp fusion per layer on this 32-layer forward"
        );
        // MEASURED (`rule_census fused_total=620 fused_cached_attention=32`):
        // 1196 unfused - 620 fused = 576 BoundOps absorbed into the 32
        // CachedAttention fusions, 18 per fusion.
        assert_eq!(
            fused.len(),
            620,
            "fused total must be 620 on this program -- 1196 unfused minus 576 BoundOps \
             absorbed across the 32 CachedAttention fusions (18 each); re-measure via the \
             `rule_census fused_total=` println above if the fusion rewrite's own \
             absorption count changes"
        );
    }
}

/// `paired_gate_up_reduce`'s own census, in the SAME relation form as
/// [`the_rule_census_reconciles_against_the_measured_mistral_forward_split`]
/// above -- deltas against the baseline program's own measured counts,
/// never a new absolute literal. Building the `[2, feed_forward,
/// embedding]`-leaf program and binding it exactly as the baseline is
/// bound (`bind_with_fusion(.., false)`, fusion held off) isolates one
/// thing: the paired reduce collapses `gate`'s and `up`'s two
/// `Op::Reduce`s into one, so `reduce_total` must drop by exactly one
/// per layer (32 layers, 32-layer program) relative to the baseline
/// this same test computes fresh -- never re-typed from the other
/// test's own docstring, which could drift.
#[test]
fn paired_gate_up_reduce_removes_one_reduce_per_layer_relative_to_the_baseline() {
    let (baseline_program, baseline_logits, baseline_roots) =
        mistral_cached_forward_program(32_002, 4096, 14336, 32, 8, 128, 32)
            .expect("the baseline cached forward pass lowers to a program");
    let mut baseline_outputs = alloc::vec![baseline_logits];
    for (even, odd, value) in &baseline_roots {
        baseline_outputs.extend_from_slice(&[*even, *odd, *value]);
    }
    let baseline_shapes = crate::shape::infer(&baseline_program, &[1, 71])
        .expect("baseline: one new position against a 71-position cache infers");
    let baseline_bound = crate::bind::bind_with_fusion(
        &baseline_program,
        &baseline_shapes,
        &baseline_outputs,
        false,
        crate::numeric::NumericPolicy::default(),
    )
    .expect("the baseline program binds");
    let baseline_reduce_total = baseline_bound
        .iter()
        .filter(|op| matches!(&op.kind, crate::bind::BoundOpKind::Reduce { .. }))
        .count();

    let (paired_program, paired_roots_bundle, paired_roots, _paired_moe_sites) =
        mistral_cached_forward_program_with_experts(
            32_002, 4096, 14336, 32, 8, 128, 32, 0, 0, false, false, true, false,
        )
        .expect("the paired cached forward pass lowers to a program");
    let paired_logits = paired_roots_bundle.logits;
    let mut paired_outputs = alloc::vec![paired_logits];
    for (even, odd, value) in &paired_roots {
        paired_outputs.extend_from_slice(&[*even, *odd, *value]);
    }
    let paired_shapes = crate::shape::infer(&paired_program, &[1, 71])
        .expect("paired: one new position against a 71-position cache infers");
    let paired_bound = crate::bind::bind_with_fusion(
        &paired_program,
        &paired_shapes,
        &paired_outputs,
        false,
        crate::numeric::NumericPolicy::default(),
    )
    .expect("the paired program binds");
    let paired_reduce_total = paired_bound
        .iter()
        .filter(|op| matches!(&op.kind, crate::bind::BoundOpKind::Reduce { .. }))
        .count();

    std::println!(
        "paired_gate_up_reduce_census baseline_reduce_total={baseline_reduce_total} paired_reduce_total={paired_reduce_total} baseline_bound_total={} paired_bound_total={}",
        baseline_bound.len(),
        paired_bound.len()
    );
    assert_eq!(
        baseline_reduce_total - paired_reduce_total,
        32,
        "paired_gate_up_reduce must remove exactly one Op::Reduce per layer (32 layers) \
         relative to the baseline program's own measured reduce total -- gate's and up's \
         two independent reduces collapsing into one paired reduce"
    );

    #[cfg(feature = "reduce-epilogue-fusion")]
    {
        let baseline_epilogued = baseline_bound
            .iter()
            .filter(|op| {
                matches!(
                    &op.kind,
                    crate::bind::BoundOpKind::Reduce { epilogue_operands, .. }
                        if !epilogue_operands.is_empty()
                )
            })
            .count();
        let paired_epilogued = paired_bound
            .iter()
            .filter(|op| {
                matches!(
                    &op.kind,
                    crate::bind::BoundOpKind::Reduce { epilogue_operands, .. }
                        if !epilogue_operands.is_empty()
                )
            })
            .count();
        std::println!(
            "paired_gate_up_reduce_census baseline_epilogued={baseline_epilogued} paired_epilogued={paired_epilogued}"
        );
        // `ffn_hidden` (`spec.rs`'s own `append_mistral_cached_layer`)
        // never absorbs the paired reduce's `up` slice: `up` is read
        // through a non-identity, base-shifted axis expression (the
        // parity-selecting `"s,0*s+1,g->sg"` map) from the SAME node the
        // `gate` chain also reads through a DIFFERENT map
        // (`"s,0*s+0,g->sg"`) — `find_epilogue_source`'s own decline for
        // two DIFFERENT projections of the same source. That site is
        // still MEASURED at zero fusion in EITHER program.
        //
        // The 32-op (one-per-layer) delta below is NEW, and it is
        // correct, not a leak: `find_epilogue_source`/
        // `resolved_reference_counts` used to also reject a consumer's
        // own REPEATED read of the SAME source through the SAME
        // projection (production SiLU reads `gate` once bare, once
        // inside `exp(-gate)`), so the `silu(gate) * up` tail never fused
        // onto `gate`'s own reduce in EITHER program. Fixing that bug
        // lets the baseline program's `gate`/`up` -- two INDEPENDENT
        // `Op::Reduce`s -- absorb that tail once per layer (`+32`
        // epilogued reduces). The paired program cannot gain the same
        // fusion: its `gate`/`up` share ONE `Op::Reduce`, read through
        // the two DIFFERENT parity-split projections above, so the SAME
        // consumer names that one reduce through two different
        // projections and `find_epilogue_source` correctly declines it,
        // exactly as the un-widened site always did. The 32-op delta is
        // therefore precisely paired_gate_up_reduce's own structural
        // cost: sharing one reduce forecloses an epilogue fusion the
        // unpaired baseline can still take.
        assert_eq!(
            baseline_epilogued - paired_epilogued,
            32,
            "paired_gate_up_reduce's shared reduce must foreclose exactly one SiLU-tail \
             epilogue fusion per layer (32 layers) relative to the baseline's two \
             independent reduces; a different delta means some OTHER site started \
             admitting or rejecting asymmetrically between the two programs"
        );
    }
}

/// `fused_qkv_reduce`'s own census, same relation-form discipline as
/// [`paired_gate_up_reduce_removes_one_reduce_per_layer_relative_to_the_baseline`]
/// above -- deltas against a freshly computed baseline, never a
/// re-typed literal. Q/K/V's three independent reduces collapse into
/// one fused reduce (`-2` `Op::Reduce`/layer, `-64` total), but q's, k's,
/// AND v's rows are three DIFFERENT sizes under GQA (unlike
/// `paired_gate_up_reduce`'s identical-size gate/up), so none of the
/// three can be read back out of the shared flat buffer at zero extra
/// cost the way `paired_gate_up_reduce` reads its parity axis -- the IR
/// cannot split one real axis into two unconstrained virtual sub-axes
/// from a single operand (`append_mistral_cached_layer`'s
/// `fused_qkv_reduce` doc traces the exact `shape::infer`
/// `UnconstrainedDim` this hits and why `ScalarOp::arity` blocks the
/// obvious fix of adding a shape-only operand to an existing binary
/// op). All three of q/k/v need their own small
/// `ScalarOp::Multiply`-by-shape-constant extract instead. Net per
/// layer: 3 reduces removed, 1 fused reduce added, 3 extracts added --
/// this test asserts the CORRECTED count, `+1` dispatch/layer (`+32`
/// total), not the `-2`/layer a bandwidth-only reading of ROW 336 would
/// predict.
#[test]
fn fused_qkv_reduce_adds_one_dispatch_per_layer_relative_to_the_baseline() {
    let (baseline_program, baseline_logits, baseline_roots) =
        mistral_cached_forward_program(32_002, 4096, 14336, 32, 8, 128, 32)
            .expect("the baseline cached forward pass lowers to a program");
    let mut baseline_outputs = alloc::vec![baseline_logits];
    for (even, odd, value) in &baseline_roots {
        baseline_outputs.extend_from_slice(&[*even, *odd, *value]);
    }
    let baseline_shapes = crate::shape::infer(&baseline_program, &[1, 71])
        .expect("baseline: one new position against a 71-position cache infers");
    let baseline_bound = crate::bind::bind_with_fusion(
        &baseline_program,
        &baseline_shapes,
        &baseline_outputs,
        false,
        crate::numeric::NumericPolicy::default(),
    )
    .expect("the baseline program binds");
    let baseline_total = baseline_bound.len();
    let baseline_reduce_total = baseline_bound
        .iter()
        .filter(|op| matches!(&op.kind, crate::bind::BoundOpKind::Reduce { .. }))
        .count();

    let (fused_program, fused_roots_bundle, fused_roots, _fused_moe_sites) =
        mistral_cached_forward_program_with_experts(
            32_002, 4096, 14336, 32, 8, 128, 32, 0, 0, false, false, false, true,
        )
        .expect("the fused-qkv cached forward pass lowers to a program");
    let fused_logits = fused_roots_bundle.logits;
    let mut fused_outputs = alloc::vec![fused_logits];
    for (even, odd, value) in &fused_roots {
        fused_outputs.extend_from_slice(&[*even, *odd, *value]);
    }
    let fused_shapes = crate::shape::infer(&fused_program, &[1, 71])
        .expect("fused: one new position against a 71-position cache infers");
    let fused_bound = crate::bind::bind_with_fusion(
        &fused_program,
        &fused_shapes,
        &fused_outputs,
        false,
        crate::numeric::NumericPolicy::default(),
    )
    .expect("the fused-qkv program binds");
    let fused_total = fused_bound.len();
    let fused_reduce_total = fused_bound
        .iter()
        .filter(|op| matches!(&op.kind, crate::bind::BoundOpKind::Reduce { .. }))
        .count();

    std::println!(
        "fused_qkv_reduce_census baseline_reduce_total={baseline_reduce_total} fused_reduce_total={fused_reduce_total} baseline_bound_total={baseline_total} fused_bound_total={fused_total}"
    );
    assert_eq!(
        baseline_reduce_total - fused_reduce_total,
        64,
        "fused_qkv_reduce must remove exactly two Op::Reduce per layer (32 layers) relative \
         to the baseline -- q's, k's, and v's three independent reduces collapsing into one \
         fused reduce"
    );
    assert_eq!(
        fused_total - baseline_total,
        34,
        "fused_qkv_reduce must ADD exactly one dispatch per layer (32 layers -> +32) plus the \
         two one-time Op::Constant shape hints built once for the whole program (+2), 34 \
         total -- 3 reduces removed, 1 fused reduce added, 3 shape-constant extracts added \
         (q/k/v each need their own, unlike paired_gate_up_reduce's zero-extra-cost parity \
         read) nets +1/layer, not the -2/layer a bandwidth-only reading of the reduce count \
         would predict"
    );
}

/// Proof the new test can fail: perturbing one tap weight must move the
/// affected output positions away from the hand-computed reference.
#[proxima::test]
async fn causal_conv1d_hand_computed_check_actually_detects_a_wrong_weight() {
    let mut program = Vec::new();
    let x = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Symbolic(0), Extent::Static(1)],
            name: Some("x".into()),
        },
    );
    let weight = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            // see the previous test's own doc on why `[1, 3]`, not `[3, 1]`.
            shape: alloc::vec![Extent::Static(1), Extent::Static(3)],
            name: Some("weight".into()),
        },
    );
    let output = causal_conv1d(&mut program, x, weight, 3).expect("causal conv lowers");

    let x_data = [1.0f32, 2.0, 3.0, 4.0];
    // tap l=2 perturbed from 100 to 99: every output except out[0] still
    // matches (out[0]'s only real contribution is tap l=2, so this alone
    // would move it too -- included to show the test is not vacuous).
    let perturbed_weight_data = [1.0f32, 10.0, 99.0];
    let evaluated = crate::cpu::evaluate_named(
        &program,
        &[4],
        &[("x", &x_data), ("weight", &perturbed_weight_data)],
        &[output],
    )
    .expect("causal conv evaluates");
    let (result, _shape) = evaluated.get(output).expect("conv output present");

    std::println!("perturbed causal_conv1d result={result:?}");
    assert_ne!(
        result,
        [100.0, 210.0, 321.0, 432.0],
        "a perturbed tap weight must move the output away from the hand-computed reference \
         (if this assertion cannot fail, the test above proves nothing)"
    );
}

/// `causal_conv1d_matches_a_hand_computed_causal_window`'s own single
/// channel (`embedding=1`) cannot distinguish `weight`'s `[l_cache,
/// embedding]` axis order from `[embedding, l_cache]` -- with one
/// channel, transposing does not move a single byte. This is exactly the
/// gap that let the real checkpoint's own `blk.{layer}.shortconv.conv.weight`
/// (GGUF on-disk `[l_cache=3, embedding=2048]`, `l_cache` the FASTEST
/// axis) get bound with its axes swapped for months: two DIFFERENT
/// per-channel weight patterns, so a transposed read produces
/// hand-verifiably wrong numbers instead of silently-correct ones.
/// Channel 0 reuses the single-channel test's own `weight = [1, 10,
/// 100]`/`x = [1, 2, 3, 4]` (`out = [100, 210, 321, 432]`, worked out
/// there); channel 1 uses `weight = [1000, 1, 1]`/`x = [2, 2, 2, 2]`:
/// - `out[0] = 1*x[0]                       = 1*2                = 2`
/// - `out[1] = 1*x[0]  + 1*x[1]             = 1*2 + 1*2          = 4`
/// - `out[2] = 1000*x[0] + 1*x[1] + 1*x[2]  = 1000*2 + 2 + 2     = 2004`
/// - `out[3] = 1000*x[1] + 1*x[2] + 1*x[3]  = 1000*2 + 2 + 2     = 2004`
#[proxima::test]
async fn causal_conv1d_keeps_channels_independent_and_catches_a_transposed_weight() {
    let mut program = Vec::new();
    let x = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Symbolic(0), Extent::Static(2)],
            name: Some("x".into()),
        },
    );
    let weight = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(2), Extent::Static(3)],
            name: Some("weight".into()),
        },
    );
    let output = causal_conv1d(&mut program, x, weight, 3).expect("causal conv lowers");

    // sequence-major, channel-fastest: `[s0_ch0, s0_ch1, s1_ch0, s1_ch1, ...]`.
    let x_data = [1.0f32, 2.0, 2.0, 2.0, 3.0, 2.0, 4.0, 2.0];
    // channel-major, tap-fastest: `[ch0_l0, ch0_l1, ch0_l2, ch1_l0, ch1_l1, ch1_l2]`
    // -- the real checkpoint's own on-disk axis order.
    let weight_data = [1.0f32, 10.0, 100.0, 1000.0, 1.0, 1.0];
    let evaluated = crate::cpu::evaluate_named(
        &program,
        &[4],
        &[("x", &x_data), ("weight", &weight_data)],
        &[output],
    )
    .expect("causal conv evaluates");
    let (result, shape) = evaluated.get(output).expect("conv output present");

    std::println!("multi-channel causal_conv1d result={result:?} shape={shape:?}");
    assert_eq!(shape, [4u64, 2u64]);
    assert_eq!(
        result,
        [100.0, 2.0, 210.0, 4.0, 321.0, 2004.0, 432.0, 2004.0]
    );
}

/// [`rmsnorm_per_head`] against a hand-computed RMS norm -- one token,
/// two heads, `head_dim = 2`: head 0's raw values `[3, 4]` have
/// `mean_square = (9+16)/2 = 12.5`, `rms = sqrt(12.5) ≈ 3.535534`,
/// `inv_rms ≈ 0.282843`; scaled by `gamma = [2.0, 0.5]` that is
/// `[3*0.282843*2, 4*0.282843*0.5] ≈ [1.697056, 0.565685]`. Head 1's
/// `[1, 1]` have `mean_square = 1`, `rms = 1`, so `gamma` passes
/// through unchanged: `[2.0, 0.5]`. Two heads with DIFFERENT norms in
/// the same call proves the reduce is scoped per-head, not pooled
/// across both (a pooled reduce would give both heads the same
/// `inv_rms`, which is not `[1.697056, ...]` next to `[2.0, ...]`).
#[proxima::test]
async fn rmsnorm_per_head_matches_a_hand_computed_rms_norm() {
    let mut program = Vec::new();
    let x = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Symbolic(0), Extent::Static(2), Extent::Static(2)],
            name: Some("x".into()),
        },
    );
    let gamma = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(2)],
            name: Some("gamma".into()),
        },
    );
    let eps = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Symbolic(0)],
            name: Some("eps".into()),
        },
    );
    let inv_head_dim = scalar_constant(&mut program, 0.5);
    let output = rmsnorm_per_head(&mut program, x, gamma, inv_head_dim, eps, "h")
        .expect("per-head rmsnorm lowers");

    let x_data = [3.0f32, 4.0, 1.0, 1.0];
    let gamma_data = [2.0f32, 0.5];
    let eps_data = [0.0f32];
    let evaluated = crate::cpu::evaluate_named(
        &program,
        &[1],
        &[("x", &x_data), ("gamma", &gamma_data), ("eps", &eps_data)],
        &[output],
    )
    .expect("per-head rmsnorm evaluates");
    let (result, shape) = evaluated.get(output).expect("rmsnorm output present");

    std::println!("rmsnorm_per_head result={result:?} shape={shape:?}");
    assert_eq!(shape, [1u64, 2u64, 2u64]);
    let expected = [1.6970563f32, 0.56568545, 2.0, 0.5];
    for (found, wanted) in result.iter().zip(&expected) {
        assert!(
            (found - wanted).abs() < 1e-5,
            "got {result:?}, expected {expected:?}"
        );
    }
}

/// Proof the new test can fail: perturbing `gamma` must move the
/// output away from the hand-computed reference.
#[proxima::test]
async fn rmsnorm_per_head_hand_computed_check_actually_detects_a_wrong_gamma() {
    let mut program = Vec::new();
    let x = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Symbolic(0), Extent::Static(2), Extent::Static(2)],
            name: Some("x".into()),
        },
    );
    let gamma = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(2)],
            name: Some("gamma".into()),
        },
    );
    let eps = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Symbolic(0)],
            name: Some("eps".into()),
        },
    );
    let inv_head_dim = scalar_constant(&mut program, 0.5);
    let output = rmsnorm_per_head(&mut program, x, gamma, inv_head_dim, eps, "h")
        .expect("per-head rmsnorm lowers");

    let x_data = [3.0f32, 4.0, 1.0, 1.0];
    // gamma[0] perturbed from 2.0 to 3.0.
    let perturbed_gamma_data = [3.0f32, 0.5];
    let eps_data = [0.0f32];
    let evaluated = crate::cpu::evaluate_named(
        &program,
        &[1],
        &[
            ("x", &x_data),
            ("gamma", &perturbed_gamma_data),
            ("eps", &eps_data),
        ],
        &[output],
    )
    .expect("per-head rmsnorm evaluates");
    let (result, _shape) = evaluated.get(output).expect("rmsnorm output present");

    std::println!("perturbed rmsnorm_per_head result={result:?}");
    let unperturbed = [1.6970563f32, 0.56568545, 2.0, 0.5];
    assert!(
        (result[0] - unperturbed[0]).abs() > 1e-3,
        "a perturbed gamma must move the output away from the hand-computed reference \
         (if this assertion cannot fail, the test above proves nothing)"
    );
}

#[proxima::test]
#[case::attention_block(2, "blk.2.attn_q.weight", LayerKind::Attention)]
#[case::conv_block(0, "blk.0.shortconv.conv.weight", LayerKind::ShortConv)]
async fn layer_kind_derives_from_the_real_checkpoints_own_tensor_marker(
    #[case] layer: u32,
    #[case] marker: &str,
    #[case] expected: LayerKind,
) {
    let names = ["token_embd.weight", marker, "output_norm.weight"];
    let derived = LayerKind::from_tensor_names(names, layer)
        .expect("a real block names exactly one marker");
    assert_eq!(derived, expected);
}

#[proxima::test]
async fn layer_kind_names_the_block_when_neither_marker_is_present() {
    let names = ["token_embd.weight", "output_norm.weight"];
    let error = LayerKind::from_tensor_names(names, 7)
        .expect_err("a block with no marker cannot derive a kind");
    assert!(
        matches!(error, TensorError::UndeterminedLayerKind { layer: 7 }),
        "got {error:?}"
    );
}

/// LFM2.5-8B-A1B's real dimensions (24 blocks: 2 leading dense, 22 MoE;
/// 18 short-convolution layers at blocks `{0,1,3,4,5,7,8,9,11,12,13,15,
/// 16,17,19,20,22,23}`, 6 attention layers at `{2,6,10,14,18,21}` -- the
/// real checkpoint's own tensor directory, cross-checked against
/// `lfm2moe.attention.head_count_kv`'s per-layer `[0,0,8,...]` sample in
/// its metadata dump) -- proves the hybrid builder lowers and infers at
/// this checkpoint's actual shapes without needing the 5 GB file itself,
/// the same real-dimensions-without-real-weights convention
/// `the_whole_mistral_forward_pass_infers_at_real_dimensions` already
/// uses above.
#[proxima::test]
async fn the_whole_lfm2_forward_pass_infers_at_real_dimensions() {
    const REAL_CONTEXT: u64 = 8192;
    const ATTENTION_LAYERS: [u32; 6] = [2, 6, 10, 14, 18, 21];

    let layer_kinds: Vec<LayerKind> = (0..24)
        .map(|layer| {
            if ATTENTION_LAYERS.contains(&layer) {
                LayerKind::Attention
            } else {
                LayerKind::ShortConv
            }
        })
        .collect();

    let attention_configs: Vec<LayerAttentionConfig> = (0..24)
        .map(|_| LayerAttentionConfig {
            head_dim: 64,
            kv_heads: 8,
            mask_window: None,
            value_source_kind: ValueSourceKind::ProjectedV,
            rope_table: RopeTableSel {
                cos_name: "rope_cos",
                sin_name: "rope_sin",
            },
            rope_pairing: RopePairing::Interleaved,
        })
        .collect();
    let ffn_configs: Vec<LayerFfnConfig> = (0..24).map(|_| LayerFfnConfig::exclusive()).collect();

    let build_start = std::time::Instant::now();
    let (program, _logits, _moe_sites) = lfm2_forward_program_with_experts(
        128_000,
        2048,
        7168,
        1792,
        32,
        24,
        32,
        4,
        2,
        3,
        &layer_kinds,
        &attention_configs,
        &ffn_configs,
        None,
        None,
    )
    .expect("the hybrid forward pass lowers to a program");
    let build_elapsed = build_start.elapsed();

    let infer_start = std::time::Instant::now();
    crate::shape::infer(&program, &[REAL_CONTEXT])
        .expect("the hybrid forward pass infers at its real context length");
    let infer_elapsed = infer_start.elapsed();

    std::println!(
        "lfm2_forward_program_with_experts: nodes={} build={build_elapsed:?} infer={infer_elapsed:?}",
        program.len()
    );
    assert!(
        program.len() > 1_000,
        "24 hybrid blocks plus embedding/lm-head should be well over a thousand nodes, not {}",
        program.len()
    );
}

#[proxima::test]
async fn lfm2_forward_program_rejects_a_layer_kinds_length_mismatch() {
    let layer_kinds = [LayerKind::Attention, LayerKind::ShortConv];
    let attention_configs = [
        LayerAttentionConfig {
            head_dim: 64,
            kv_heads: 8,
            mask_window: None,
            value_source_kind: ValueSourceKind::ProjectedV,
            rope_table: RopeTableSel {
                cos_name: "rope_cos",
                sin_name: "rope_sin",
            },
            rope_pairing: RopePairing::Interleaved,
        },
        LayerAttentionConfig {
            head_dim: 64,
            kv_heads: 8,
            mask_window: None,
            value_source_kind: ValueSourceKind::ProjectedV,
            rope_table: RopeTableSel {
                cos_name: "rope_cos",
                sin_name: "rope_sin",
            },
            rope_pairing: RopePairing::Interleaved,
        },
    ];
    let ffn_configs = [LayerFfnConfig::exclusive(), LayerFfnConfig::exclusive()];
    let error = lfm2_forward_program_with_experts(
        128_000,
        2048,
        7168,
        1792,
        32,
        24,
        32,
        4,
        2,
        3,
        &layer_kinds,
        &attention_configs,
        &ffn_configs,
        None,
        None,
    )
    .expect_err("2 layer_kinds against block_count=24 must be rejected");
    assert!(
        matches!(
            error,
            TensorError::LayerKindCountMismatch {
                expected: 24,
                found: 2
            }
        ),
        "got {error:?}"
    );
}

#[proxima::test]
async fn causal_conv1d_rejects_a_zero_width_window() {
    let mut program = Vec::new();
    let x = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Symbolic(0), Extent::Static(1)],
            name: Some("x".into()),
        },
    );
    let weight = op::append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(0), Extent::Static(1)],
            name: Some("weight".into()),
        },
    );
    let error = causal_conv1d(&mut program, x, weight, 0)
        .expect_err("l_cache=0 has no window to convolve");
    assert!(
        matches!(error, TensorError::InvalidConvConfig { l_cache: 0 }),
        "got {error:?}"
    );
}

/// [`append_qwen35_delta_net_step`] against a hand-computed single-head,
/// `key_dim = value_dim = 2` delta-rule step -- llama.cpp's own
/// `build_delta_net_autoregressive` traced by hand: `decay = exp(0) =
/// 1`, `state_decayed = state_in` (`[[1,2],[3,4]]`), `v_pred = [1,2]`
/// (`state_decayed^T @ k` with `k = [1,0]`), `residual = v - v_pred =
/// [4,4]` (`v = [5,6]`), `delta = residual * beta = [4,4]` (`beta = 1`),
/// `state_out = state_decayed + k(outer)delta = [[5,6],[3,4]]`,
/// `out = state_out^T @ q_scaled` with `q_scaled = q * 1 = [1,0]` gives
/// `[5,6]`.
#[proxima::test]
async fn qwen35_delta_net_step_matches_a_hand_computed_recurrence() {
    let mut program = Vec::new();
    let shape_ih = alloc::vec![Extent::Static(2), Extent::Static(1)];
    let shape_jh = alloc::vec![Extent::Static(2), Extent::Static(1)];
    let shape_h = alloc::vec![Extent::Static(1)];
    let shape_ijh = alloc::vec![Extent::Static(2), Extent::Static(2), Extent::Static(1)];

    let query = input_leaf(&mut program, DType::Float32, shape_ih.clone(), "query");
    let key = input_leaf(&mut program, DType::Float32, shape_ih, "key");
    let value = input_leaf(&mut program, DType::Float32, shape_jh, "value");
    let gate = input_leaf(&mut program, DType::Float32, shape_h.clone(), "gate");
    let beta = input_leaf(&mut program, DType::Float32, shape_h, "beta");
    let state_in = input_leaf(&mut program, DType::Float32, shape_ijh, "state_in");
    let inv_sqrt_key_dim = scalar_constant(&mut program, 1.0);

    let (out, state_out) = append_qwen35_delta_net_step(
        &mut program,
        query,
        key,
        value,
        gate,
        beta,
        state_in,
        inv_sqrt_key_dim,
        "h",
    )
    .expect("delta net step lowers");

    let query_data = [1.0f32, 0.0];
    let key_data = [1.0f32, 0.0];
    let value_data = [5.0f32, 6.0];
    let gate_data = [0.0f32];
    let beta_data = [1.0f32];
    let state_in_data = [1.0f32, 2.0, 3.0, 4.0];

    let evaluated = crate::cpu::evaluate_named(
        &program,
        &[],
        &[
            ("query", &query_data),
            ("key", &key_data),
            ("value", &value_data),
            ("gate", &gate_data),
            ("beta", &beta_data),
            ("state_in", &state_in_data),
        ],
        &[out, state_out],
    )
    .expect("delta net step evaluates");

    let (out_values, _) = evaluated.get(out).expect("out present");
    let (state_out_values, _) = evaluated.get(state_out).expect("state_out present");

    assert_eq!(out_values, [5.0, 6.0], "read-out uses the UPDATED state");
    assert_eq!(
        state_out_values,
        [5.0, 6.0, 3.0, 4.0],
        "state_out = state_decayed + k(outer)delta"
    );
}

/// The caller-buffered prefill scan drives the same one-position graph
/// transition twice and exposes both per-position outputs plus the final
/// carried matrix state. This is the executable boundary a layer-major
/// prefill path can call after producing [`SsmMixerTaps`]' query/key/
/// value/gate/beta tensors in a batch.
#[proxima::test]
async fn qwen35_gdn_prefill_scan_matches_repeated_graph_steps() {
    let mut program = Vec::new();
    let shape_ih = alloc::vec![Extent::Static(2), Extent::Static(1)];
    let shape_jh = alloc::vec![Extent::Static(2), Extent::Static(1)];
    let shape_h = alloc::vec![Extent::Static(1)];
    let shape_ijh = alloc::vec![Extent::Static(2), Extent::Static(2), Extent::Static(1)];
    let query = input_leaf(&mut program, DType::Float32, shape_ih.clone(), "query");
    let key = input_leaf(&mut program, DType::Float32, shape_ih, "key");
    let value = input_leaf(&mut program, DType::Float32, shape_jh, "value");
    let gate = input_leaf(&mut program, DType::Float32, shape_h.clone(), "gate");
    let beta = input_leaf(&mut program, DType::Float32, shape_h, "beta");
    let state_in = input_leaf(&mut program, DType::Float32, shape_ijh, "state_in");
    let inv_sqrt_key_dim = scalar_constant(&mut program, 1.0);
    let (out, state_out) = append_qwen35_delta_net_step(
        &mut program,
        query,
        key,
        value,
        gate,
        beta,
        state_in,
        inv_sqrt_key_dim,
        "h",
    )
    .expect("delta net step lowers");

    let queries = [1.0_f32, 0.0, 0.0, 1.0];
    let keys = [1.0_f32, 0.0, 0.0, 1.0];
    let values = [5.0_f32, 6.0, 7.0, 8.0];
    let gates = [0.0_f32, 0.0];
    let betas = [1.0_f32, 1.0];
    let initial_state = [1.0_f32, 2.0, 3.0, 4.0];

    let mut repeated_state = initial_state.to_vec();
    let mut repeated_outputs = Vec::new();
    for position in 0..2 {
        let query_row = &queries[position * 2..position * 2 + 2];
        let key_row = &keys[position * 2..position * 2 + 2];
        let value_row = &values[position * 2..position * 2 + 2];
        let evaluated = crate::cpu::evaluate_named(
            &program,
            &[],
            &[
                ("query", query_row),
                ("key", key_row),
                ("value", value_row),
                ("gate", &gates[position..position + 1]),
                ("beta", &betas[position..position + 1]),
                ("state_in", repeated_state.as_slice()),
            ],
            &[out, state_out],
        )
        .expect("one recurrent graph step evaluates");
        repeated_outputs.extend_from_slice(evaluated.get(out).expect("out present").0);
        repeated_state = evaluated
            .get(state_out)
            .expect("state_out present")
            .0
            .to_vec();
    }

    let mut scanned_state = initial_state;
    let mut scanned_outputs = [0.0_f32; 4];
    crate::cpu::run_gdn_prefill_scan(crate::cpu::GdnPrefillScan {
        shape: crate::cpu::GdnPrefillShape {
            positions: 2,
            key_dim: 2,
            value_dim: 2,
            heads: 1,
            kv_heads: 1,
        },
        query: &queries,
        key: &keys,
        // `kv_heads == 1`: only one head, so any stride pair addresses
        // the same single row -- natural (dim-fastest) chosen for
        // consistency with the decode-only bound kind's own convention.
        query_key_head_stride: 2,
        query_key_dim_stride: 1,
        value: &values,
        gate: &gates,
        beta: &betas,
        inv_sqrt_key_dim: 1.0,
        state: &mut scanned_state,
        output: &mut scanned_outputs,
    })
    .expect("caller-buffered prefill scan evaluates");

    assert_eq!(scanned_outputs.as_slice(), repeated_outputs.as_slice());
    assert_eq!(scanned_state.as_slice(), repeated_state.as_slice());
    assert_eq!(scanned_outputs, [5.0, 6.0, 7.0, 8.0]);
    assert_eq!(scanned_state, [5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn qwen35_gdn_prefill_scan_rejects_a_truncated_projection() {
    let mut state = [0.0_f32; 4];
    let mut output = [0.0_f32; 2];
    let error = crate::cpu::run_gdn_prefill_scan(crate::cpu::GdnPrefillScan {
        shape: crate::cpu::GdnPrefillShape {
            positions: 1,
            key_dim: 2,
            value_dim: 2,
            heads: 1,
            kv_heads: 1,
        },
        query: &[1.0],
        key: &[1.0, 0.0],
        query_key_head_stride: 2,
        query_key_dim_stride: 1,
        value: &[1.0, 0.0],
        gate: &[0.0],
        beta: &[1.0],
        inv_sqrt_key_dim: 1.0,
        state: &mut state,
        output: &mut output,
    })
    .expect_err("a truncated query projection must be rejected");
    assert!(matches!(
        error,
        TensorError::GdnPrefillBufferSizeMismatch {
            buffer: "query",
            expected: 2,
            found: 1,
        }
    ));
}

/// Proof the hand-computed test can fail: a nonzero `beta` on a
/// perturbed run must move both `out` and `state_out` away from the
/// `beta = 0` (no update at all) reference.
#[proxima::test]
async fn qwen35_delta_net_step_hand_computed_check_actually_detects_a_wrong_beta() {
    let mut program = Vec::new();
    let shape_ih = alloc::vec![Extent::Static(2), Extent::Static(1)];
    let shape_jh = alloc::vec![Extent::Static(2), Extent::Static(1)];
    let shape_h = alloc::vec![Extent::Static(1)];
    let shape_ijh = alloc::vec![Extent::Static(2), Extent::Static(2), Extent::Static(1)];

    let query = input_leaf(&mut program, DType::Float32, shape_ih.clone(), "query");
    let key = input_leaf(&mut program, DType::Float32, shape_ih, "key");
    let value = input_leaf(&mut program, DType::Float32, shape_jh, "value");
    let gate = input_leaf(&mut program, DType::Float32, shape_h.clone(), "gate");
    let beta = input_leaf(&mut program, DType::Float32, shape_h, "beta");
    let state_in = input_leaf(&mut program, DType::Float32, shape_ijh, "state_in");
    let inv_sqrt_key_dim = scalar_constant(&mut program, 1.0);

    let (out, state_out) = append_qwen35_delta_net_step(
        &mut program,
        query,
        key,
        value,
        gate,
        beta,
        state_in,
        inv_sqrt_key_dim,
        "h",
    )
    .expect("delta net step lowers");

    let query_data = [1.0f32, 0.0];
    let key_data = [1.0f32, 0.0];
    let value_data = [5.0f32, 6.0];
    let gate_data = [0.0f32];
    let beta_data = [0.0f32];
    let state_in_data = [1.0f32, 2.0, 3.0, 4.0];

    let evaluated = crate::cpu::evaluate_named(
        &program,
        &[],
        &[
            ("query", &query_data),
            ("key", &key_data),
            ("value", &value_data),
            ("gate", &gate_data),
            ("beta", &beta_data),
            ("state_in", &state_in_data),
        ],
        &[out, state_out],
    )
    .expect("delta net step evaluates");

    let (out_values, _) = evaluated.get(out).expect("out present");
    let (state_out_values, _) = evaluated.get(state_out).expect("state_out present");

    assert_ne!(
        out_values,
        [5.0, 6.0],
        "beta=0 must move out away from the beta=1 reference"
    );
    assert_ne!(
        state_out_values,
        [5.0, 6.0, 3.0, 4.0],
        "beta=0 must move state_out away from the beta=1 reference"
    );
}

/// [`softplus`] against `log(1 + exp(x))` hand-computed at `x = 0` and
/// `x = 1`: `softplus(0) = ln(2) ≈ 0.6931`, `softplus(1) = ln(1 + e) ≈
/// 1.3133` -- llama.cpp's own `ggml_softplus` input to Qwen3.5's
/// `alpha_softplus` (`qwen35.cpp:370`).
#[proxima::test]
async fn softplus_matches_log_one_plus_exp() {
    let mut program = Vec::new();
    let shape_h = alloc::vec![Extent::Static(2)];
    let x = input_leaf(&mut program, DType::Float32, shape_h, "x");
    let one = scalar_constant(&mut program, 1.0);

    let out = softplus(&mut program, x, one, "h->h").expect("softplus lowers");

    let x_data = [0.0f32, 1.0];
    let evaluated = crate::cpu::evaluate_named(&program, &[], &[("x", &x_data)], &[out])
        .expect("softplus evaluates");
    let (out_values, _) = evaluated.get(out).expect("out present");

    assert!(
        (out_values[0] - core::f32::consts::LN_2).abs() < 1e-4,
        "softplus(0) = ln(2), got {}",
        out_values[0]
    );
    assert!(
        (out_values[1] - 1.313_262).abs() < 1e-4,
        "softplus(1) = ln(1+e), got {}",
        out_values[1]
    );
}

/// [`l2norm`] against a hand-computed `[3, 4]` vector: `norm = sqrt(9 +
/// 16) = 5`, so the normalized output is `[0.6, 0.8]` -- `ggml_l2_norm`'s
/// own contract (`qwen35.cpp:428-429`), no learnable weight and no
/// mean-divide, unlike [`rmsnorm`].
#[proxima::test]
async fn l2norm_matches_a_hand_computed_unit_vector() {
    let mut program = Vec::new();
    let shape_d = alloc::vec![Extent::Static(1), Extent::Static(2)];
    let x = input_leaf(&mut program, DType::Float32, shape_d, "x");
    let eps = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(1)],
        "eps",
    );

    let out = l2norm(&mut program, x, eps, "sd->sd", "s->sd").expect("l2norm lowers");

    let x_data = [3.0f32, 4.0];
    let eps_data = [0.0f32];
    let evaluated = crate::cpu::evaluate_named(
        &program,
        &[],
        &[("x", &x_data), ("eps", &eps_data)],
        &[out],
    )
    .expect("l2norm evaluates");
    let (out_values, _) = evaluated.get(out).expect("out present");

    assert!((out_values[0] - 0.6).abs() < 1e-5, "got {}", out_values[0]);
    assert!((out_values[1] - 0.8).abs() < 1e-5, "got {}", out_values[1]);
}

/// [`append_qwen35_conv_branch`] against a hand-computed depthwise causal
/// conv (kernel 4, three channels `q|k|v` at `key_dim = value_dim = 1`,
/// two steps): `causal_conv1d`'s own doc gives `out[s,d] = sum_l
/// weight[d,l] * x[s+l-3, d]` (zero where the index is negative), so with
/// only taps `l=2,3` ever landing in range for a 2-step sequence:
/// `out[0,d] = weight[d,3]*x[0,d]`, `out[1,d] = weight[d,2]*x[0,d] +
/// weight[d,3]*x[1,d]`. With `x[0] = [1,2,3]`, `x[1] = [4,5,6]` and
/// `weight[.,2..4] = [[0.5,1.0], [1.0,-1.0], [2.0,0.5]]` (q,k,v):
/// `out[0] = [1.0, -2.0, 1.5]`, `out[1] = [4.5, -3.0, 9.0]`. `v_conv`
/// (never normalized) is checked against `silu` of those exactly;
/// `q_conv`/`k_conv` are l2-normalized at width 1, which degenerates to
/// `sign(x)` (`x / sqrt(x^2 + 0) = x / |x|`) -- `q`'s raw values are both
/// positive, `k`'s both negative.
#[proxima::test]
async fn append_qwen35_conv_branch_matches_a_hand_computed_conv_silu_split_and_norm() {
    let key_dim = 1u32;
    let value_dim = 1u32;
    let l_cache = 4u32;
    let qkv_dim = 2 * key_dim + value_dim;

    let mut program = Vec::new();
    let qkv_mixed = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(qkv_dim)],
        "qkv_mixed",
    );
    let conv_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(qkv_dim), Extent::Static(l_cache)],
        "conv_weight",
    );
    let eps = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0)],
        "eps",
    );
    let one = scalar_constant(&mut program, 1.0);

    let (q_conv, k_conv, v_conv) = append_qwen35_conv_branch(
        &mut program,
        qkv_mixed,
        conv_weight,
        eps,
        one,
        key_dim,
        value_dim,
        l_cache,
    )
    .expect("conv branch lowers");

    // sequence-major, channel-fastest: `[s0_q, s0_k, s0_v, s1_q, s1_k, s1_v]`.
    let x_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    // channel-major, tap-fastest: taps 0,1 are unreachable at seq=2, so
    // only taps 2,3 (per channel) carry real weight.
    let weight_data = [
        0.0f32, 0.0, 0.5, 1.0, // q
        0.0, 0.0, 1.0, -1.0, // k
        0.0, 0.0, 2.0, 0.5, // v
    ];
    let eps_data = [0.0f32, 0.0];

    let evaluated = crate::cpu::evaluate_named(
        &program,
        &[2],
        &[
            ("qkv_mixed", &x_data),
            ("conv_weight", &weight_data),
            ("eps", &eps_data),
        ],
        &[q_conv, k_conv, v_conv],
    )
    .expect("conv branch evaluates");

    let (q_values, _) = evaluated.get(q_conv).expect("q_conv present");
    let (k_values, _) = evaluated.get(k_conv).expect("k_conv present");
    let (v_values, v_shape) = evaluated.get(v_conv).expect("v_conv present");

    assert_eq!(v_shape, [2u64, 1u64]);
    assert!(
        (v_values[0] - 1.226_362).abs() < 1e-4,
        "got {}",
        v_values[0]
    );
    assert!((v_values[1] - 8.998_89).abs() < 1e-4, "got {}", v_values[1]);
    assert_eq!(
        q_values,
        [1.0, 1.0],
        "silu(q_raw) is positive at both steps, so l2norm at width 1 is +1"
    );
    assert_eq!(
        k_values,
        [-1.0, -1.0],
        "silu(k_raw) is negative at both steps, so l2norm at width 1 is -1"
    );
}

/// Proof the conv-branch reference above can fail: perturbing one `v`
/// tap weight must move `v_conv` away from the hand-computed reference.
#[proxima::test]
async fn append_qwen35_conv_branch_hand_computed_check_actually_detects_a_wrong_weight() {
    let key_dim = 1u32;
    let value_dim = 1u32;
    let l_cache = 4u32;
    let qkv_dim = 2 * key_dim + value_dim;

    let mut program = Vec::new();
    let qkv_mixed = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(qkv_dim)],
        "qkv_mixed",
    );
    let conv_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(qkv_dim), Extent::Static(l_cache)],
        "conv_weight",
    );
    let eps = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0)],
        "eps",
    );
    let one = scalar_constant(&mut program, 1.0);

    let (_, _, v_conv) = append_qwen35_conv_branch(
        &mut program,
        qkv_mixed,
        conv_weight,
        eps,
        one,
        key_dim,
        value_dim,
        l_cache,
    )
    .expect("conv branch lowers");

    let x_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    // v's tap l=3 perturbed from 0.5 to 0.4 -- moves both v_conv positions,
    // since out[0] and out[1] both read tap l=3.
    let perturbed_weight_data = [
        0.0f32, 0.0, 0.5, 1.0, // q
        0.0, 0.0, 1.0, -1.0, // k
        0.0, 0.0, 2.0, 0.4, // v
    ];
    let eps_data = [0.0f32, 0.0];

    let evaluated = crate::cpu::evaluate_named(
        &program,
        &[2],
        &[
            ("qkv_mixed", &x_data),
            ("conv_weight", &perturbed_weight_data),
            ("eps", &eps_data),
        ],
        &[v_conv],
    )
    .expect("conv branch evaluates");
    let (v_values, _) = evaluated.get(v_conv).expect("v_conv present");

    assert!(
        (v_values[0] - 1.226_362).abs() > 1e-4 || (v_values[1] - 8.998_89).abs() > 1e-4,
        "a perturbed v tap weight must move v_conv away from the hand-computed reference \
         (if this assertion cannot fail, the test above proves nothing), got {v_values:?}"
    );
}

/// [`repeat_kv_heads`] against a hand-computed 2-kv-head, group-3 repeat
/// (`num_v_heads = kv_heads * group = 6`, standing in for the real
/// checkpoint's `16 * 3 = 48`): one token, `head_dim = 1`, kv head 0
/// carries `10.0`, kv head 1 carries `20.0`. Every one of kv head 0's
/// three query-head copies must read `10.0` and every one of kv head 1's
/// three must read `20.0`, in `u`-major, `g`-minor order (`sugd`) --
/// `[10,10,10,20,20,20]`, not interleaved (`[10,20,10,20,10,20]`, the
/// shape a `u`/`g` axis swap would produce) and not collapsed onto one
/// head (`[10,10,10,10,10,10]`, the shape a broadcast-only-`u`
/// -- forgetting to size `g` from `group_ones` -- would produce).
#[proxima::test]
async fn repeat_kv_heads_maps_each_kv_head_to_its_own_three_query_heads() {
    let kv_heads = 2u32;
    let group = 3u32;

    let mut program = Vec::new();
    let x = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Symbolic(0),
            Extent::Static(kv_heads),
            Extent::Static(1)
        ],
        "x",
    );
    let repeated = repeat_kv_heads(&mut program, x, kv_heads, group).expect("repeat lowers");

    let x_data = [10.0f32, 20.0];
    let evaluated = crate::cpu::evaluate_named(&program, &[1], &[("x", &x_data)], &[repeated])
        .expect("repeat evaluates");
    let (values, shape) = evaluated.get(repeated).expect("repeated present");

    assert_eq!(shape, [1u64, 2u64, 3u64, 1u64]);
    assert_eq!(values, [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]);
}

/// Proof the repeat reference above can fail: swapping which kv head
/// carries which value must move the repeated output away from the
/// hand-computed reference.
#[proxima::test]
async fn repeat_kv_heads_hand_computed_check_actually_detects_a_swapped_head() {
    let kv_heads = 2u32;
    let group = 3u32;

    let mut program = Vec::new();
    let x = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Symbolic(0),
            Extent::Static(kv_heads),
            Extent::Static(1)
        ],
        "x",
    );
    let repeated = repeat_kv_heads(&mut program, x, kv_heads, group).expect("repeat lowers");

    // kv head 0 and kv head 1's values swapped relative to the reference.
    let swapped_x_data = [20.0f32, 10.0];
    let evaluated =
        crate::cpu::evaluate_named(&program, &[1], &[("x", &swapped_x_data)], &[repeated])
            .expect("repeat evaluates");
    let (values, _) = evaluated.get(repeated).expect("repeated present");

    assert_ne!(
        values,
        [10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
        "a swapped kv head must move the repeated output away from the hand-computed \
         reference (if this assertion cannot fail, the test above proves nothing)"
    );
}

/// [`append_qwen35_ssm_mixer_with_taps`]'s own `s`-axis guard: a static
/// `s = 0` can never feed the recurrence (there is no position zero to
/// seed `state_out`), so it is refused before any op after the guard
/// runs -- unlike `s > 1`, which the M>1 branch
/// [`qwen35_ssm_mixer_one_evaluation_matches_repeated_single_position_steps`]
/// covers now unrolls rather than rejects. Every non-`x` argument reuses
/// `x` itself: the guard is the function's first statement, so nothing
/// downstream of it ever reads them.
#[proxima::test]
async fn qwen35_ssm_mixer_rejects_a_static_zero_width_step() {
    let mut program = Vec::new();
    let x = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(0), Extent::Static(1)],
        "x",
    );

    let result = append_qwen35_ssm_mixer_with_taps(
        &mut program,
        x,
        x,
        x,
        x,
        x,
        x,
        x,
        None,
        x,
        x,
        x,
        x,
        x,
        x,
        x,
        x,
        x,
        x,
        x,
        1,
        1,
        1,
        1,
        1,
        GdnOutputGate::Silu,
        Some(0),
    );

    match result {
        Err(TensorError::SingleTokenStepOnly { op, s }) => {
            assert_eq!(op, "qwen35_ssm_mixer");
            assert_eq!(s, 0);
        }
        other => panic!("expected SingleTokenStepOnly, got {other:?}"),
    }
}

/// The M>1 branch [`append_qwen35_ssm_mixer_with_taps_and_layout`] takes
/// when `x`'s leading axis is a literal `Extent::Static(width)` above 1
/// (a per-request bound graph once the prompt length is known, per that
/// function's own doc) unrolls the delta-rule recurrence across the
/// prompt in one program, rather than through the caller-driven scan
/// [`qwen35_prefill_scan_and_tail_match_repeated_mixer_steps`] already
/// covers. The oracle is the same shape both tests already trust: one
/// evaluation of a static-`M` program against `M` sequential
/// single-position evaluations of the SAME (unchanged, `M=1`-branch)
/// builder, threading `state_out` into the next call's `state_in` and
/// `qkv_mixed` into the next call's `conv_history_in` exactly the way a
/// real decode loop already does.
#[proxima::test]
async fn qwen35_ssm_mixer_one_evaluation_matches_repeated_single_position_steps() {
    let key_dim = 1u32;
    let value_dim = 2u32;
    let kv_heads = 1u32;
    let group = 2u32;
    let l_cache = 2u32;
    let qkv_dim = 2 * key_dim + value_dim;
    let positions = 3u32;

    let x_data = [1.0_f32, -1.0, 0.5];
    let attn_norm_weight_data = [1.0_f32];
    let wqkv_data = [1.0_f32, 2.0, 3.0, 4.0];
    let wqkv_gate_data = [0.5_f32, -0.25];
    let conv_weight_data = [
        0.5_f32, 1.0, // q
        -0.25, 0.75, // k
        1.5, -0.5, // v0
        0.25, 2.0, // v1
    ];
    let ssm_beta_data = [0.25_f32, -0.5];
    let ssm_alpha_data = [0.75_f32, 0.125];
    let ssm_dt_bias_data = [0.1_f32, -0.2];
    let ssm_a_data = [-0.5_f32, -0.25];
    let ssm_norm_weight_data = [1.0_f32];
    let ssm_out_data = [0.75_f32, -0.5];
    let head_eps_data = [0.0_f32, 0.0];
    let initial_state = [0.0_f32, 0.0];
    let initial_history = [0.0_f32; 4];

    let mut static_program = Vec::new();
    let x = input_leaf(
        &mut static_program,
        DType::Float32,
        alloc::vec![Extent::Static(positions), Extent::Static(1)],
        "x",
    );
    let inv_dim = scalar_constant(&mut static_program, 1.0);
    let eps = input_leaf(
        &mut static_program,
        DType::Float32,
        alloc::vec![Extent::Static(positions)],
        "eps",
    );
    let head_eps = input_leaf(
        &mut static_program,
        DType::Float32,
        alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
        "head_eps",
    );
    let one = scalar_constant(&mut static_program, 1.0);
    let inv_sqrt_key_dim = scalar_constant(&mut static_program, 1.0);
    let inv_head_v_dim = scalar_constant(&mut static_program, 1.0);
    let attn_norm_weight = input_leaf(
        &mut static_program,
        DType::Float32,
        alloc::vec![Extent::Static(1)],
        "attn_norm_weight",
    );
    let wqkv = input_leaf(
        &mut static_program,
        DType::Float32,
        alloc::vec![Extent::Static(1), Extent::Static(qkv_dim)],
        "wqkv",
    );
    let wqkv_gate = input_leaf(
        &mut static_program,
        DType::Float32,
        alloc::vec![Extent::Static(1), Extent::Static(value_dim)],
        "wqkv_gate",
    );
    let conv_weight = input_leaf(
        &mut static_program,
        DType::Float32,
        alloc::vec![Extent::Static(qkv_dim), Extent::Static(l_cache)],
        "conv_weight",
    );
    let conv_history_in = input_leaf(
        &mut static_program,
        DType::Float32,
        alloc::vec![Extent::Static(l_cache - 1), Extent::Static(qkv_dim)],
        "conv_history_in",
    );
    let ssm_beta = input_leaf(
        &mut static_program,
        DType::Float32,
        alloc::vec![Extent::Static(1), Extent::Static(kv_heads * group)],
        "ssm_beta",
    );
    let ssm_alpha = input_leaf(
        &mut static_program,
        DType::Float32,
        alloc::vec![Extent::Static(1), Extent::Static(kv_heads * group)],
        "ssm_alpha",
    );
    let ssm_dt_bias = input_leaf(
        &mut static_program,
        DType::Float32,
        alloc::vec![Extent::Static(kv_heads * group)],
        "ssm_dt_bias",
    );
    let ssm_a = input_leaf(
        &mut static_program,
        DType::Float32,
        alloc::vec![Extent::Static(kv_heads * group)],
        "ssm_a",
    );
    let ssm_norm_weight = input_leaf(
        &mut static_program,
        DType::Float32,
        alloc::vec![Extent::Static(1)],
        "ssm_norm_weight",
    );
    let ssm_out = input_leaf(
        &mut static_program,
        DType::Float32,
        alloc::vec![Extent::Static(value_dim), Extent::Static(1)],
        "ssm_out",
    );
    let state_in = input_leaf(
        &mut static_program,
        DType::Float32,
        alloc::vec![
            Extent::Static(1),
            Extent::Static(1),
            Extent::Static(kv_heads),
            Extent::Static(group)
        ],
        "state_in",
    );
    let (static_mixer_out, static_taps) = append_qwen35_ssm_mixer_with_taps(
        &mut static_program,
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
        key_dim,
        value_dim,
        kv_heads,
        group,
        l_cache,
        GdnOutputGate::Silu,
        Some(positions),
    )
    .expect("the M>1 branch lowers");

    let static_result = crate::cpu::evaluate_named(
        &static_program,
        &[u64::from(positions)],
        &[
            ("x", x_data.as_slice()),
            ("eps", &[0.0_f32; 3]),
            ("head_eps", &head_eps_data),
            ("attn_norm_weight", &attn_norm_weight_data),
            ("wqkv", &wqkv_data),
            ("wqkv_gate", &wqkv_gate_data),
            ("conv_weight", &conv_weight_data),
            ("conv_history_in", &initial_history[..qkv_dim as usize]),
            ("ssm_beta", &ssm_beta_data),
            ("ssm_alpha", &ssm_alpha_data),
            ("ssm_dt_bias", &ssm_dt_bias_data),
            ("ssm_a", &ssm_a_data),
            ("ssm_norm_weight", &ssm_norm_weight_data),
            ("ssm_out", &ssm_out_data),
            ("state_in", &initial_state),
        ],
        &[static_mixer_out, static_taps.state_out],
    )
    .expect("the M>1 program evaluates");
    let static_mixer_rows = static_result
        .get(static_mixer_out)
        .expect("static mixer_out present")
        .0;
    let static_state = static_result
        .get(static_taps.state_out)
        .expect("static state_out present")
        .0;

    let (repeated_program, repeated_mixer_out, repeated_taps) =
        build_ssm_mixer_test_program(GdnOutputGate::Silu);
    let mut sequential_state = initial_state.to_vec();
    let mut sequential_history = initial_history[..qkv_dim as usize].to_vec();
    let mut sequential_mixer_rows = alloc::vec::Vec::new();
    for position in 0..positions as usize {
        let evaluated = crate::cpu::evaluate_named(
            &repeated_program,
            &[1],
            &[
                ("x", &x_data[position..position + 1]),
                ("eps", &[0.0_f32]),
                ("head_eps", &head_eps_data),
                ("attn_norm_weight", &attn_norm_weight_data),
                ("wqkv", &wqkv_data),
                ("wqkv_gate", &wqkv_gate_data),
                ("conv_weight", &conv_weight_data),
                ("conv_history_in", sequential_history.as_slice()),
                ("ssm_beta", &ssm_beta_data),
                ("ssm_alpha", &ssm_alpha_data),
                ("ssm_dt_bias", &ssm_dt_bias_data),
                ("ssm_a", &ssm_a_data),
                ("ssm_norm_weight", &ssm_norm_weight_data),
                ("ssm_out", &ssm_out_data),
                ("state_in", sequential_state.as_slice()),
            ],
            &[
                repeated_mixer_out,
                repeated_taps.state_out,
                repeated_taps.qkv_mixed,
            ],
        )
        .expect("one sequential mixer step evaluates");
        sequential_mixer_rows.extend_from_slice(
            evaluated
                .get(repeated_mixer_out)
                .expect("sequential mixer_out present")
                .0,
        );
        sequential_state = evaluated
            .get(repeated_taps.state_out)
            .expect("sequential state_out present")
            .0
            .to_vec();
        sequential_history = evaluated
            .get(repeated_taps.qkv_mixed)
            .expect("sequential qkv_mixed present")
            .0
            .to_vec();
    }

    assert_relative_rows_match(
        static_mixer_rows,
        &sequential_mixer_rows,
        "mixer_out",
    );
    assert_relative_rows_match(static_state, &sequential_state, "state_out");
}

/// proxima-debugger unit oracle (qwen35moe GDN prefill-scan-vs-sequential
/// divergence): the SAME comparison
/// [`qwen35_ssm_mixer_one_evaluation_matches_repeated_single_position_steps`]
/// makes, at the real checkpoint's own head shape (`kv_heads=16`,
/// `group=2`, `head_k_dim=128`, `head_v_dim=128`, `l_cache=4` --
/// `key_dim`/`value_dim` here are the TOTAL, pre-head-split extents this
/// builder's own parameters take: `key_dim=2048=head_k_dim*kv_heads`,
/// `value_dim=4096=head_v_dim*kv_heads*group`) instead of that test's toy
/// `kv_heads=1`, `head_k_dim=1`. `kv_heads=1` degenerates the `u` axis to
/// a single row, so a bug that only shows up when `u` (kv head) and `g`
/// (group) are BOTH non-degenerate cannot be caught there -- this is that
/// missing case. Every intermediate `SsmMixerTaps` field that is a
/// per-position sequence in the M>1 static graph (`qkv_mixed`,
/// `query_sequence`, `key_sequence`, `value_sequence`, `beta_sequence`,
/// `gate_sequence`, `z_sequence`) is compared row-by-row against that same
/// position's own M=1 single-step tap, in addition to the final
/// `mixer_out`/`state_out` the toy oracle already checks -- the row that
/// first exceeds tolerance names the exact stage the M>1 branch diverges
/// at.
///
/// Parametrized over exactly the arguments
/// [`proxima_model_interop::qwen35moe::program`] passes that this
/// oracle's un-parametrized form did not: `v_head_reordered`, whether
/// `attn_norm_weight` is present, the output gate, and the prefill
/// width -- the full program builds with `Some(attn_norm_weight)`,
/// `GdnOutputGate::Silu`, `architecture.v_head_reordered`, and
/// `M=13`; every other combination is here to bisect which single
/// argument (if any) the M>1 branch mishandles.
#[proxima::test]
#[case::baseline(false, true, GdnOutputGate::Silu, 4u32)]
#[case::v_head_reordered_m4(true, true, GdnOutputGate::Silu, 4u32)]
#[case::v_head_reordered_m13(true, true, GdnOutputGate::Silu, 13u32)]
#[case::no_attn_norm_m4(false, false, GdnOutputGate::Silu, 4u32)]
#[case::sigmoid_gate_m4(false, true, GdnOutputGate::Sigmoid, 4u32)]
#[case::production_args_m13(true, true, GdnOutputGate::Silu, 13u32)]
async fn qwen35_ssm_mixer_one_evaluation_matches_repeated_single_position_steps_at_real_dims(
    #[case] v_head_reordered: bool,
    #[case] attn_norm_present: bool,
    #[case] output_gate: GdnOutputGate,
    #[case] positions: u32,
) {
    let key_dim = 2048u32;
    let value_dim = 4096u32;
    let kv_heads = 16u32;
    let group = 2u32;
    let l_cache = 4u32;
    let embedding = 6u32;
    let qkv_dim = 2 * key_dim + value_dim;
    let num_v_heads = kv_heads * group;
    let head_k_dim = key_dim / kv_heads;
    let head_v_dim = value_dim / num_v_heads;

    let deterministic_wave = |index: usize, modulus: usize, scale: f32| -> f32 {
        ((index % modulus) as f32 + 1.0) * scale
    };

    let x_data: alloc::vec::Vec<f32> = (0..(positions * embedding) as usize)
        .map(|index| deterministic_wave(index, 23, 0.01))
        .collect();
    let attn_norm_weight_data: alloc::vec::Vec<f32> = (0..embedding as usize)
        .map(|index| 1.0 + deterministic_wave(index, 7, 0.02))
        .collect();
    let wqkv_data: alloc::vec::Vec<f32> = (0..(embedding * qkv_dim) as usize)
        .map(|index| deterministic_wave(index, 29, 0.002))
        .collect();
    let wqkv_gate_data: alloc::vec::Vec<f32> = (0..(embedding * value_dim) as usize)
        .map(|index| deterministic_wave(index, 31, 0.003))
        .collect();
    let conv_weight_data: alloc::vec::Vec<f32> = (0..(qkv_dim * l_cache) as usize)
        .map(|index| deterministic_wave(index, 17, 0.01))
        .collect();
    let ssm_beta_data: alloc::vec::Vec<f32> = (0..(embedding * num_v_heads) as usize)
        .map(|index| deterministic_wave(index, 11, 0.02) - 0.1)
        .collect();
    let ssm_alpha_data: alloc::vec::Vec<f32> = (0..(embedding * num_v_heads) as usize)
        .map(|index| deterministic_wave(index, 13, 0.02) - 0.1)
        .collect();
    let ssm_dt_bias_data: alloc::vec::Vec<f32> = (0..num_v_heads as usize)
        .map(|index| deterministic_wave(index, 5, 0.01))
        .collect();
    let ssm_a_data: alloc::vec::Vec<f32> = (0..num_v_heads as usize)
        .map(|index| -deterministic_wave(index, 5, 0.05))
        .collect();
    let ssm_norm_weight_data: alloc::vec::Vec<f32> = (0..head_v_dim as usize)
        .map(|index| 1.0 + deterministic_wave(index, 3, 0.01))
        .collect();
    let ssm_out_data: alloc::vec::Vec<f32> = (0..(value_dim * embedding) as usize)
        .map(|index| deterministic_wave(index, 19, 0.004))
        .collect();
    let head_eps_data = alloc::vec![1e-6_f32; (kv_heads * group) as usize];
    let initial_state = alloc::vec![0.0_f32; (head_k_dim * head_v_dim * kv_heads * group) as usize];
    let initial_history = alloc::vec![0.0_f32; ((l_cache - 1) * qkv_dim) as usize];

    let build_program = |sequence_extent: Extent| -> (Vec<Op>, NodeId, SsmMixerTaps) {
        let mut program = Vec::new();
        let x = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![sequence_extent, Extent::Static(embedding)],
            "x",
        );
        let inv_dim = scalar_constant(&mut program, 1.0 / embedding as f32);
        let eps = input_leaf(&mut program, DType::Float32, alloc::vec![sequence_extent], "eps");
        let head_eps = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
            "head_eps",
        );
        let one = scalar_constant(&mut program, 1.0);
        let inv_sqrt_key_dim = scalar_constant(&mut program, 1.0 / (head_k_dim as f32).sqrt());
        let inv_head_v_dim = scalar_constant(&mut program, 1.0 / head_v_dim as f32);
        let attn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            "attn_norm_weight",
        );
        let wqkv = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding), Extent::Static(qkv_dim)],
            "wqkv",
        );
        let wqkv_gate = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding), Extent::Static(value_dim)],
            "wqkv_gate",
        );
        let conv_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(qkv_dim), Extent::Static(l_cache)],
            "conv_weight",
        );
        let conv_history_in = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(l_cache - 1), Extent::Static(qkv_dim)],
            "conv_history_in",
        );
        let ssm_beta = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding), Extent::Static(num_v_heads)],
            "ssm_beta",
        );
        let ssm_alpha = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding), Extent::Static(num_v_heads)],
            "ssm_alpha",
        );
        let ssm_dt_bias = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(num_v_heads)],
            "ssm_dt_bias",
        );
        let ssm_a = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(num_v_heads)],
            "ssm_a",
        );
        let ssm_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(head_v_dim)],
            "ssm_norm_weight",
        );
        let ssm_out = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(value_dim), Extent::Static(embedding)],
            "ssm_out",
        );
        let state_in = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(head_k_dim),
                Extent::Static(head_v_dim),
                Extent::Static(kv_heads),
                Extent::Static(group)
            ],
            "state_in",
        );
        let (mixer_out, taps) = append_qwen35_ssm_mixer_with_taps_and_layout(
            &mut program,
            x,
            inv_dim,
            eps,
            head_eps,
            one,
            inv_sqrt_key_dim,
            inv_head_v_dim,
            if attn_norm_present { Some(attn_norm_weight) } else { None },
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
            key_dim,
            value_dim,
            kv_heads,
            group,
            l_cache,
            output_gate,
            v_head_reordered,
            match sequence_extent {
                Extent::Static(width) => Some(width),
                Extent::Symbolic(_) => None,
            },
        )
        .expect("the qwen35 ssm mixer lowers at real dims");
        (program, mixer_out, taps)
    };

    let (static_program, static_mixer_out, static_taps) =
        build_program(Extent::Static(positions));
    let static_result = crate::cpu::evaluate_named(
        &static_program,
        &[u64::from(positions)],
        &[
            ("x", x_data.as_slice()),
            ("eps", alloc::vec![1e-6_f32; positions as usize].as_slice()),
            ("head_eps", &head_eps_data),
            ("attn_norm_weight", &attn_norm_weight_data),
            ("wqkv", &wqkv_data),
            ("wqkv_gate", &wqkv_gate_data),
            ("conv_weight", &conv_weight_data),
            ("conv_history_in", &initial_history),
            ("ssm_beta", &ssm_beta_data),
            ("ssm_alpha", &ssm_alpha_data),
            ("ssm_dt_bias", &ssm_dt_bias_data),
            ("ssm_a", &ssm_a_data),
            ("ssm_norm_weight", &ssm_norm_weight_data),
            ("ssm_out", &ssm_out_data),
            ("state_in", &initial_state),
        ],
        &[
            static_mixer_out,
            static_taps.state_out,
            static_taps.qkv_mixed,
            static_taps.query_sequence,
            static_taps.key_sequence,
            static_taps.value_sequence,
            static_taps.beta_sequence,
            static_taps.gate_sequence,
            static_taps.z_sequence,
        ],
    )
    .expect("the static M>1 program evaluates");

    let static_mixer_rows = static_result.get(static_mixer_out).expect("mixer_out present").0;
    let static_state = static_result.get(static_taps.state_out).expect("state_out present").0;
    let static_qkv_mixed = static_result.get(static_taps.qkv_mixed).expect("qkv_mixed present").0;
    let static_query = static_result.get(static_taps.query_sequence).expect("query_sequence present").0;
    let static_key = static_result.get(static_taps.key_sequence).expect("key_sequence present").0;
    let static_value = static_result.get(static_taps.value_sequence).expect("value_sequence present").0;
    let static_beta = static_result.get(static_taps.beta_sequence).expect("beta_sequence present").0;
    let static_gate = static_result.get(static_taps.gate_sequence).expect("gate_sequence present").0;
    let static_z = static_result.get(static_taps.z_sequence).expect("z_sequence present").0;

    let (single_program, single_mixer_out, single_taps) = build_program(Extent::Symbolic(0));
    let mut sequential_state = initial_state.clone();
    let mut sequential_history = initial_history.clone();
    let mut sequential_mixer_rows = alloc::vec::Vec::new();
    let mut failures: alloc::vec::Vec<alloc::string::String> = alloc::vec::Vec::new();

    let row_length = |total: usize| -> usize { total / positions as usize };
    let check_row = |label: &str,
                          position: usize,
                          static_buffer: &[f32],
                          single_row: &[f32],
                          failures: &mut alloc::vec::Vec<alloc::string::String>| {
        let length = row_length(static_buffer.len());
        let static_row = &static_buffer[position * length..(position + 1) * length];
        let mut max_relative_error = 0.0_f32;
        for (found, wanted) in static_row.iter().zip(single_row.iter()) {
            let row_norm = wanted.abs().max(1e-5);
            let relative_error = (found - wanted).abs() / row_norm;
            max_relative_error = max_relative_error.max(relative_error);
        }
        std::println!(
            "qwen35_ssm_mixer_real_dims tap={label} position={position} max_relative_error={max_relative_error}"
        );
        if max_relative_error > 1e-5 {
            failures.push(alloc::format!(
                "tap={label} position={position} max_relative_error={max_relative_error}"
            ));
        }
    };

    for position in 0..positions as usize {
        let evaluated = crate::cpu::evaluate_named(
            &single_program,
            &[1],
            &[
                ("x", &x_data[position * embedding as usize..(position + 1) * embedding as usize]),
                ("eps", &[1e-6_f32]),
                ("head_eps", &head_eps_data),
                ("attn_norm_weight", &attn_norm_weight_data),
                ("wqkv", &wqkv_data),
                ("wqkv_gate", &wqkv_gate_data),
                ("conv_weight", &conv_weight_data),
                ("conv_history_in", sequential_history.as_slice()),
                ("ssm_beta", &ssm_beta_data),
                ("ssm_alpha", &ssm_alpha_data),
                ("ssm_dt_bias", &ssm_dt_bias_data),
                ("ssm_a", &ssm_a_data),
                ("ssm_norm_weight", &ssm_norm_weight_data),
                ("ssm_out", &ssm_out_data),
                ("state_in", sequential_state.as_slice()),
            ],
            &[
                single_mixer_out,
                single_taps.state_out,
                single_taps.qkv_mixed,
                single_taps.query,
                single_taps.key,
                single_taps.value,
                single_taps.beta,
                single_taps.gate,
                single_taps.z_head,
            ],
        )
        .expect("one sequential mixer step evaluates at real dims");

        let single_mixer_row = evaluated.get(single_mixer_out).expect("single mixer_out present").0;
        sequential_mixer_rows.extend_from_slice(single_mixer_row);

        check_row("qkv_mixed", position, static_qkv_mixed, evaluated.get(single_taps.qkv_mixed).expect("qkv_mixed present").0, &mut failures);
        check_row("query", position, static_query, evaluated.get(single_taps.query).expect("query present").0, &mut failures);
        check_row("key", position, static_key, evaluated.get(single_taps.key).expect("key present").0, &mut failures);
        check_row("value", position, static_value, evaluated.get(single_taps.value).expect("value present").0, &mut failures);
        check_row("beta", position, static_beta, evaluated.get(single_taps.beta).expect("beta present").0, &mut failures);
        check_row("gate", position, static_gate, evaluated.get(single_taps.gate).expect("gate present").0, &mut failures);
        check_row("z_head", position, static_z, evaluated.get(single_taps.z_head).expect("z_head present").0, &mut failures);
        check_row("mixer_out", position, static_mixer_rows, single_mixer_row, &mut failures);

        sequential_state = evaluated.get(single_taps.state_out).expect("sequential state_out present").0.to_vec();
        let newest_row = evaluated.get(single_taps.qkv_mixed).expect("sequential qkv_mixed present").0;
        // roll the `[l_cache-1, qkv_dim]` window forward: drop the oldest row,
        // append this position's `qkv_mixed` as the newest (oldest-first layout,
        // matching `conv_history_in`'s own declared shape).
        sequential_history.drain(0..qkv_dim as usize);
        sequential_history.extend_from_slice(newest_row);
    }

    let mut state_relative_error = 0.0_f32;
    for (found, wanted) in static_state.iter().zip(sequential_state.iter()) {
        let row_norm = wanted.abs().max(1e-5);
        state_relative_error = state_relative_error.max((found - wanted).abs() / row_norm);
    }
    std::println!("qwen35_ssm_mixer_real_dims tap=state_out final max_relative_error={state_relative_error}");
    if state_relative_error > 1e-5 {
        failures.push(alloc::format!("tap=state_out final max_relative_error={state_relative_error}"));
    }

    assert!(
        failures.is_empty(),
        "real-dims M>1 vs sequential diverged:\n{}",
        failures.join("\n")
    );
}

/// A row-relative tolerance (`1e-5` of the sequential oracle row's own
/// norm, floored at `1e-5` absolute so an exactly-zero row still
/// tolerates float noise) -- the same shape every other cross-path
/// numeric oracle in this module already asserts, spelled once since
/// this test compares two multi-row buffers position by position.
fn assert_relative_rows_match(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label} row count must match"
    );
    for (index, (found, wanted)) in actual.iter().zip(expected.iter()).enumerate() {
        let row_norm = wanted.abs().max(1e-5);
        let relative_error = (found - wanted).abs() / row_norm;
        assert!(
            relative_error <= 1e-5,
            "{label}[{index}] = {found}, expected {wanted}, relative_error = {relative_error}"
        );
    }
}

/// Builds one [`append_qwen35_ssm_mixer`] decode step at `kv_heads = 1`,
/// `group = 2` (the `u,g` seam, degenerate on `u` but real on `g`),
/// `key_dim = value_dim_per_head = 1`, `l_cache = 2` -- every weight
/// chosen to make the pipeline hand-traceable: `wqkv = 0` and the conv's
/// new-token tap contributes nothing, so the whole conv branch reads
/// straight off `conv_history_v0` (the one value this function
/// parameterizes, for the mutation companion below); `attn_norm_weight
/// = 1`, `x = 1`, `eps = 0` make `rmsnorm(x) = 1` exactly, so every
/// downstream projection equals its own raw weight row; `ssm_beta =
/// ssm_alpha = ssm_dt_bias = ssm_a = 0` collapse `beta` to `sigmoid(0) =
/// 0.5` and `gate` (hence `decay`) to `0`/`1`, reusing
/// [`append_qwen35_delta_net_step`]'s own already-proven `decay = 1`
/// path; `head_v_dim = 1` degenerates the gated RMSNorm's own
/// mean-square to `delta_out^2`, so `normed_out = sign(delta_out)`
/// exactly, the same width-1-l2norm-is-sign identity
/// [`append_qwen35_conv_branch`]'s own test already exploits.
fn build_ssm_mixer_test_program(output_gate: GdnOutputGate) -> (Vec<Op>, NodeId, SsmMixerTaps) {
    let mut program = Vec::new();
    let key_dim = 1u32;
    let value_dim = 2u32;
    let kv_heads = 1u32;
    let group = 2u32;
    let l_cache = 2u32;
    let qkv_dim = 2 * key_dim + value_dim;

    let x = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(1)],
        "x",
    );
    let inv_dim = scalar_constant(&mut program, 1.0);
    let eps = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0)],
        "eps",
    );
    let head_eps = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
        "head_eps",
    );
    let one = scalar_constant(&mut program, 1.0);
    let inv_sqrt_key_dim = scalar_constant(&mut program, 1.0);
    let inv_head_v_dim = scalar_constant(&mut program, 1.0);
    let attn_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(1)],
        "attn_norm_weight",
    );
    let wqkv = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(1), Extent::Static(qkv_dim)],
        "wqkv",
    );
    let wqkv_gate = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(1), Extent::Static(value_dim)],
        "wqkv_gate",
    );
    let conv_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(qkv_dim), Extent::Static(l_cache)],
        "conv_weight",
    );
    let conv_history_in = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(l_cache - 1), Extent::Static(qkv_dim)],
        "conv_history_in",
    );
    let ssm_beta = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(1), Extent::Static(kv_heads * group)],
        "ssm_beta",
    );
    let ssm_alpha = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(1), Extent::Static(kv_heads * group)],
        "ssm_alpha",
    );
    let ssm_dt_bias = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(kv_heads * group)],
        "ssm_dt_bias",
    );
    let ssm_a = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(kv_heads * group)],
        "ssm_a",
    );
    let ssm_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(1)],
        "ssm_norm_weight",
    );
    let ssm_out = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(value_dim), Extent::Static(1)],
        "ssm_out",
    );
    let state_in = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(1),
            Extent::Static(1),
            Extent::Static(kv_heads),
            Extent::Static(group)
        ],
        "state_in",
    );

    let (mixer_out, taps) = append_qwen35_ssm_mixer_with_taps(
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
        key_dim,
        value_dim,
        kv_heads,
        group,
        l_cache,
        output_gate,
        None,
    )
    .expect("ssm mixer lowers");

    (program, mixer_out, taps)
}

/// The recurrent prefill boundary is made from values the graph already
/// computes, rather than a second projection path. Their shapes are the
/// exact one-position contract [`append_qwen35_delta_net_step`] consumes;
/// an executor may therefore cut immediately before `query` through
/// `beta`, batch the prefix, and thread only `state_out` sequentially.
#[test]
fn qwen35_ssm_taps_expose_the_existing_delta_net_step_inputs() {
    let (program, _mixer_out, taps) = build_ssm_mixer_test_program(GdnOutputGate::Silu);
    let shapes = crate::shape::infer(&program, &[1]).expect("mixer shapes infer");

    assert_eq!(shapes.of(taps.query), &[1, 1, 2]);
    assert_eq!(shapes.of(taps.key), &[1, 1, 2]);
    assert_eq!(shapes.of(taps.value), &[1, 1, 2]);
    assert_eq!(shapes.of(taps.gate), &[1, 2]);
    assert_eq!(shapes.of(taps.beta), &[1, 2]);
    assert!(
        [taps.query, taps.key, taps.value, taps.gate, taps.beta]
            .into_iter()
            .all(|input| input.0 < taps.state_out.0),
        "every recurrence input must precede the state output in SSA order"
    );
}

#[proxima::test]
async fn qwen35_prefill_sequence_taps_match_repeated_cached_conv_steps() {
    let (program, _mixer_out, taps) = build_ssm_mixer_test_program(GdnOutputGate::Silu);

    let x_data = [1.0_f32, -1.0];
    let eps_data = [0.0_f32, 0.0];
    let head_eps_data = [0.0_f32, 0.0];
    let attn_norm_weight_data = [1.0_f32];
    let wqkv_data = [1.0_f32, 2.0, 3.0, 4.0];
    let wqkv_gate_data = [0.5_f32, -0.25];
    let conv_weight_data = [
        0.5_f32, 1.0, // q
        -0.25, 0.75, // k
        1.5, -0.5, // v0
        0.25, 2.0, // v1
    ];
    let ssm_beta_data = [0.25_f32, -0.5];
    let ssm_alpha_data = [0.75_f32, 0.125];
    let ssm_dt_bias_data = [0.1_f32, -0.2];
    let ssm_a_data = [-0.5_f32, -0.25];
    let ssm_norm_weight_data = [1.0_f32];
    let ssm_out_data = [1.0_f32, 1.0];
    let state_in_data = [0.0_f32, 0.0];
    let initial_history = [0.0_f32; 4];

    let sequence = crate::cpu::evaluate_named(
        &program,
        &[2],
        &[
            ("x", &x_data),
            ("eps", &eps_data),
            ("head_eps", &head_eps_data),
            ("attn_norm_weight", &attn_norm_weight_data),
            ("wqkv", &wqkv_data),
            ("wqkv_gate", &wqkv_gate_data),
            ("conv_weight", &conv_weight_data),
            ("conv_history_in", &initial_history),
            ("ssm_beta", &ssm_beta_data),
            ("ssm_alpha", &ssm_alpha_data),
            ("ssm_dt_bias", &ssm_dt_bias_data),
            ("ssm_a", &ssm_a_data),
            ("ssm_norm_weight", &ssm_norm_weight_data),
            ("ssm_out", &ssm_out_data),
            ("state_in", &state_in_data),
        ],
        &[
            taps.query_sequence,
            taps.key_sequence,
            taps.value_sequence,
            taps.gate_sequence,
            taps.beta_sequence,
        ],
    )
    .expect("sequence taps evaluate");

    let mut history = initial_history;
    let mut repeated_query = Vec::new();
    let mut repeated_key = Vec::new();
    let mut repeated_value = Vec::new();
    let mut repeated_gate = Vec::new();
    let mut repeated_beta = Vec::new();
    for position in 0..2 {
        let evaluated = crate::cpu::evaluate_named(
            &program,
            &[1],
            &[
                ("x", &x_data[position..position + 1]),
                ("eps", &eps_data[position..position + 1]),
                ("head_eps", &head_eps_data),
                ("attn_norm_weight", &attn_norm_weight_data),
                ("wqkv", &wqkv_data),
                ("wqkv_gate", &wqkv_gate_data),
                ("conv_weight", &conv_weight_data),
                ("conv_history_in", &history),
                ("ssm_beta", &ssm_beta_data),
                ("ssm_alpha", &ssm_alpha_data),
                ("ssm_dt_bias", &ssm_dt_bias_data),
                ("ssm_a", &ssm_a_data),
                ("ssm_norm_weight", &ssm_norm_weight_data),
                ("ssm_out", &ssm_out_data),
                ("state_in", &state_in_data),
            ],
            &[
                taps.qkv_mixed,
                taps.query,
                taps.key,
                taps.value,
                taps.gate,
                taps.beta,
            ],
        )
        .expect("one cached-conv step evaluates");
        repeated_query.extend_from_slice(evaluated.get(taps.query).expect("query").0);
        repeated_key.extend_from_slice(evaluated.get(taps.key).expect("key").0);
        repeated_value.extend_from_slice(evaluated.get(taps.value).expect("value").0);
        repeated_gate.extend_from_slice(evaluated.get(taps.gate).expect("gate").0);
        repeated_beta.extend_from_slice(evaluated.get(taps.beta).expect("beta").0);
        history.copy_from_slice(evaluated.get(taps.qkv_mixed).expect("qkv mixed").0);
    }

    for (sequence_node, repeated, label) in [
        (taps.query_sequence, repeated_query.as_slice(), "query"),
        (taps.key_sequence, repeated_key.as_slice(), "key"),
        (taps.value_sequence, repeated_value.as_slice(), "value"),
        (taps.gate_sequence, repeated_gate.as_slice(), "gate"),
        (taps.beta_sequence, repeated_beta.as_slice(), "beta"),
    ] {
        let sequence_values = sequence.get(sequence_node).expect(label).0;
        assert_eq!(
            sequence_values.len(),
            repeated.len(),
            "{label} lengths must agree"
        );
        for (index, (found, expected)) in
            sequence_values.iter().zip(repeated.iter()).enumerate()
        {
            assert!(
                (found - expected).abs() < 1e-6,
                "{label}[{index}] differs: sequence={found} repeated={expected}"
            );
        }
    }
}

#[proxima::test]
async fn qwen35_prefill_scan_and_tail_match_repeated_mixer_steps() {
    let (program, mixer_out, taps) = build_ssm_mixer_test_program(GdnOutputGate::Silu);

    let x_data = [1.0_f32, -1.0];
    let eps_data = [0.0_f32, 0.0];
    let head_eps_data = [0.0_f32, 0.0];
    let attn_norm_weight_data = [1.0_f32];
    let wqkv_data = [1.0_f32, 2.0, 3.0, 4.0];
    let wqkv_gate_data = [0.5_f32, -0.25];
    let conv_weight_data = [
        0.5_f32, 1.0, // q
        -0.25, 0.75, // k
        1.5, -0.5, // v0
        0.25, 2.0, // v1
    ];
    let ssm_beta_data = [0.25_f32, -0.5];
    let ssm_alpha_data = [0.75_f32, 0.125];
    let ssm_dt_bias_data = [0.1_f32, -0.2];
    let ssm_a_data = [-0.5_f32, -0.25];
    let ssm_norm_weight_data = [1.0_f32];
    let ssm_out_data = [0.75_f32, -0.5];
    let initial_state = [0.0_f32, 0.0];
    let initial_history = [0.0_f32; 4];

    let sequence = crate::cpu::evaluate_named(
        &program,
        &[2],
        &[
            ("x", &x_data),
            ("eps", &eps_data),
            ("head_eps", &head_eps_data),
            ("attn_norm_weight", &attn_norm_weight_data),
            ("wqkv", &wqkv_data),
            ("wqkv_gate", &wqkv_gate_data),
            ("conv_weight", &conv_weight_data),
            ("conv_history_in", &initial_history),
            ("ssm_beta", &ssm_beta_data),
            ("ssm_alpha", &ssm_alpha_data),
            ("ssm_dt_bias", &ssm_dt_bias_data),
            ("ssm_a", &ssm_a_data),
            ("ssm_norm_weight", &ssm_norm_weight_data),
            ("ssm_out", &ssm_out_data),
            ("state_in", &initial_state),
        ],
        &[
            taps.query_sequence,
            taps.key_sequence,
            taps.value_sequence,
            taps.gate_sequence,
            taps.beta_sequence,
            taps.z_sequence,
        ],
    )
    .expect("sequence taps evaluate");

    let mut scanned_state = initial_state;
    let mut scanned_delta = [0.0_f32; 4];
    crate::cpu::run_gdn_prefill_scan(crate::cpu::GdnPrefillScan {
        shape: crate::cpu::GdnPrefillShape {
            positions: 2,
            key_dim: 1,
            value_dim: 1,
            heads: 2,
            kv_heads: 2,
        },
        query: sequence.get(taps.query_sequence).expect("query sequence").0,
        key: sequence.get(taps.key_sequence).expect("key sequence").0,
        // `query_sequence`/`key_sequence`'s own `"sdug->sugd"` reduce
        // (`append_qwen35_ssm_mixer_with_taps_and_layout`'s prefill
        // branch) stores `kv_heads` fastest, `key_dim` next -- `key_dim
        // == 1` here so `query_key_dim_stride`'s exact value never
        // actually advances an index, but `kv_heads` still must.
        query_key_head_stride: 1,
        query_key_dim_stride: 2,
        value: sequence.get(taps.value_sequence).expect("value sequence").0,
        gate: sequence.get(taps.gate_sequence).expect("gate sequence").0,
        beta: sequence.get(taps.beta_sequence).expect("beta sequence").0,
        inv_sqrt_key_dim: 1.0,
        state: &mut scanned_state,
        output: &mut scanned_delta,
    })
    .expect("prefill scan evaluates");

    let mut tail_program = Vec::new();
    let tail_x = input_leaf(
        &mut tail_program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(1)],
        "tail_x",
    );
    let tail_delta = input_leaf(
        &mut tail_program,
        DType::Float32,
        alloc::vec![
            Extent::Symbolic(0),
            Extent::Static(1),
            Extent::Static(1),
            Extent::Static(2),
        ],
        "tail_delta",
    );
    let tail_z = input_leaf(
        &mut tail_program,
        DType::Float32,
        alloc::vec![
            Extent::Symbolic(0),
            Extent::Static(1),
            Extent::Static(2),
            Extent::Static(1),
        ],
        "tail_z",
    );
    let tail_head_eps = input_leaf(
        &mut tail_program,
        DType::Float32,
        alloc::vec![Extent::Static(1), Extent::Static(2)],
        "tail_head_eps",
    );
    let tail_inv_head_v_dim = scalar_constant(&mut tail_program, 1.0);
    let tail_norm_weight = input_leaf(
        &mut tail_program,
        DType::Float32,
        alloc::vec![Extent::Static(1)],
        "tail_norm_weight",
    );
    let tail_out_weight = input_leaf(
        &mut tail_program,
        DType::Float32,
        alloc::vec![Extent::Static(2), Extent::Static(1)],
        "tail_out_weight",
    );
    let tail_mixer_out = append_qwen35_gdn_sequence_tail(
        &mut tail_program,
        Qwen35GdnSequenceTail {
            x: tail_x,
            delta_out: tail_delta,
            z: tail_z,
            head_eps: tail_head_eps,
            inv_head_v_dim: tail_inv_head_v_dim,
            norm_weight: tail_norm_weight,
            out_weight: tail_out_weight,
            head_v_dim: 1,
            kv_heads: 1,
            group: 2,
        },
    )
    .expect("sequence tail lowers");
    let tail = crate::cpu::evaluate_named(
        &tail_program,
        &[2],
        &[
            ("tail_x", &x_data),
            ("tail_delta", &scanned_delta),
            (
                "tail_z",
                sequence.get(taps.z_sequence).expect("z sequence").0,
            ),
            ("tail_head_eps", &head_eps_data),
            ("tail_norm_weight", &ssm_norm_weight_data),
            ("tail_out_weight", &ssm_out_data),
        ],
        &[tail_mixer_out],
    )
    .expect("sequence tail evaluates");

    let mut history = initial_history;
    let mut state = initial_state;
    let mut repeated_delta = Vec::new();
    let mut repeated_mixer = Vec::new();
    for position in 0..2 {
        let evaluated = crate::cpu::evaluate_named(
            &program,
            &[1],
            &[
                ("x", &x_data[position..position + 1]),
                ("eps", &eps_data[position..position + 1]),
                ("head_eps", &head_eps_data),
                ("attn_norm_weight", &attn_norm_weight_data),
                ("wqkv", &wqkv_data),
                ("wqkv_gate", &wqkv_gate_data),
                ("conv_weight", &conv_weight_data),
                ("conv_history_in", &history),
                ("ssm_beta", &ssm_beta_data),
                ("ssm_alpha", &ssm_alpha_data),
                ("ssm_dt_bias", &ssm_dt_bias_data),
                ("ssm_a", &ssm_a_data),
                ("ssm_norm_weight", &ssm_norm_weight_data),
                ("ssm_out", &ssm_out_data),
                ("state_in", &state),
            ],
            &[mixer_out, taps.qkv_mixed, taps.delta_out, taps.state_out],
        )
        .expect("one mixer step evaluates");
        repeated_mixer.extend_from_slice(evaluated.get(mixer_out).expect("mixer out").0);
        repeated_delta.extend_from_slice(evaluated.get(taps.delta_out).expect("delta out").0);
        history.copy_from_slice(evaluated.get(taps.qkv_mixed).expect("qkv mixed").0);
        state.copy_from_slice(evaluated.get(taps.state_out).expect("state out").0);
    }

    for (label, found, expected) in [
        ("delta", scanned_delta.as_slice(), repeated_delta.as_slice()),
        ("state", scanned_state.as_slice(), state.as_slice()),
        (
            "mixer",
            tail.get(tail_mixer_out).expect("tail mixer out").0,
            repeated_mixer.as_slice(),
        ),
    ] {
        assert_eq!(found.len(), expected.len(), "{label} lengths must agree");
        for (index, (found_value, expected_value)) in
            found.iter().zip(expected.iter()).enumerate()
        {
            assert!(
                (found_value - expected_value).abs() < 1e-5,
                "{label}[{index}] differs: scan={found_value} repeated={expected_value}"
            );
        }
    }
}

/// Builds one call into [`append_qwen35_dense_attention_layer`] at the
/// smallest non-degenerate dims that still separate all three fixed
/// defects: `embedding = 1` (so every matmul is a scalar identity,
/// `wq`/`wk`/`wv`/`w_gate_q`/`wo` ARE the per-dim activation), `rotary_dim
/// = 2` (`pairs = 1`, exercises the split-half pairing) with `attn_head_dim
/// = 4` (`pass_dim = 2`, exercises the concatenated-by-sum remainder),
/// `kv_heads = query_heads = 1` (`group = 1`, no GQA broadcast to track
/// by hand), `cached_len = 0` (only the "new" self-attention block).
///
/// Hand computation: `x = [2.0]`, `attn_norm_weight = [1.0]`, `eps = 0`
/// gives `normed = 2 / sqrt(4) = 1`, so every `w*` weight IS its own raw
/// projection. `wq = wk = wv = [1,1,1,1]`, `q_norm_weight =
/// k_norm_weight = [1,1,1,1]` -- `rmsnorm_per_head` of `[1,1,1,1]` is a
/// no-op (`mean_square = 1`), keeping `q = k = [1,1,1,1]` exactly.
/// `cos = [0]`, `sin = [1]` (`theta = pi/2` spelled as literals, no
/// transcendental arithmetic needed): split-half RoPE gives
/// `rotated_first = q[0]*0 - q[1]*1 = -1`, `rotated_second = q[1]*0 +
/// q[0]*1 = 1` for both Q and K -- the ROTATED prefix is `[-1, 1]`, the
/// PASS remainder stays `[1, 1]` (defect 2's own fix: dropped in the old
/// code, present here as `score_..._pass`'s own nonzero contribution).
/// `score = (-1)(-1) + (1)(1) + (1)(1) + (1)(1) = 4`, scaled by
/// `inv_sqrt_attn_head_dim = 0.5` gives `2.0`; one key only (self), so
/// softmax weight is `1.0` and `attended = v = [1,1,1,1]`.
/// `w_gate_q = [0,0,0,0]` gives `sigmoid(gate) = 0.5` uniformly (defect
/// 1's own fix: dropped in the old code, present here as the factor of
/// `2` between `attended` and `attn_out` below): `gated_attended =
/// [0.5,0.5,0.5,0.5]`. `wo = [1,1,1,1]` gives `attn_out = 0.5*4 = 2.0`,
/// `residual1 = 2.0 + 2.0 = 4.0`. `normed2 = 4/|4| = 1` (`d=1` again),
/// `feed_forward = 1`, `w_gate = w_up = w_down = [1]`: `silu(1) =
/// 1 * sigmoid(1) ≈ 0.7310586`, `ffn_hidden = 0.7310586 * 1`,
/// `ffn_out = 0.7310586`, `x_next = 0.7310586 + 4.0 ≈ 4.7310586`.
#[test]
fn append_qwen35_dense_attention_layer_matches_a_hand_computed_gate_and_partial_rotary_concat()
{
    let (x_next, ..) = evaluate_dense_attention_test_program(0.0, [0.0f32, 0.0, 0.0, 0.0]);

    assert!(
        (x_next - 4.731_058_6).abs() < 1e-4,
        "x_next = ffn_out + residual1, hand-computed 4.7310586, got {x_next}"
    );
}

/// Proof the hand-computed test above can fail: a nonzero `w_gate_q`
/// (the sigmoid gate this session's own defect 1 fix applies) MUST move
/// `x_next` away from the `w_gate_q = [0,0,0,0]` reference above --
/// `w_gate_q = [10,10,10,10]` pushes `sigmoid(gate)` from `0.5` toward
/// `1.0`, doubling `attended`'s own contribution to `attn_out` before
/// `wo`. A test that could not fail here would not have caught the old
/// code's dropped gate either.
#[test]
fn append_qwen35_dense_attention_layer_hand_computed_check_actually_detects_a_dropped_gate() {
    let (x_next_no_gate, ..) =
        evaluate_dense_attention_test_program(0.0, [0.0f32, 0.0, 0.0, 0.0]);
    let (x_next_gated, ..) =
        evaluate_dense_attention_test_program(0.0, [10.0f32, 10.0, 10.0, 10.0]);

    assert!(
        (x_next_no_gate - x_next_gated).abs() > 0.1,
        "w_gate_q=[0,0,0,0] (sigmoid=0.5) vs [10,10,10,10] (sigmoid~1.0) must move x_next: \
         {x_next_no_gate} vs {x_next_gated}"
    );
}

/// One call into [`append_qwen35_dense_attention_layer`] at the fixed
/// small dims the two tests above hand-compute against -- `gate_data`
/// is the only knob a caller varies (`w_gate_q`'s own 4 values), so the
/// mutation test above and the base test share every other weight byte
/// for byte.
fn evaluate_dense_attention_test_program(
    eps_value: f32,
    gate_data: [f32; 4],
) -> (f32, f32, f32, f32, f32) {
    let mut program = Vec::new();
    let scalar_shape = alloc::vec![Extent::Symbolic(0), Extent::Static(1)];
    let rotary_shape = alloc::vec![Extent::Symbolic(0), Extent::Static(1)];
    let head4_shape = alloc::vec![Extent::Static(1), Extent::Static(1), Extent::Static(4)];
    let cache4_shape = alloc::vec![Extent::Symbolic(1), Extent::Static(1), Extent::Static(1)];
    let cache_pass_shape =
        alloc::vec![Extent::Symbolic(1), Extent::Static(1), Extent::Static(2)];
    let cache_v_shape = alloc::vec![Extent::Symbolic(1), Extent::Static(1), Extent::Static(4)];
    let norm_shape = alloc::vec![Extent::Static(4)];
    let ffn_shape = alloc::vec![Extent::Static(1), Extent::Static(1)];

    let x = input_leaf(&mut program, DType::Float32, scalar_shape.clone(), "x");
    let inv_dim = scalar_constant(&mut program, 1.0);
    let eps = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0)],
        "eps",
    );
    let ones = scalar_constant(&mut program, 1.0);
    let inv_sqrt_attn_head_dim = scalar_constant(&mut program, 0.5);
    let inv_attn_head_dim = scalar_constant(&mut program, 0.25);
    let cos_new = input_leaf(&mut program, DType::Float32, rotary_shape.clone(), "cos");
    let sin_new = input_leaf(&mut program, DType::Float32, rotary_shape, "sin");
    let group_ones = op::append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(1), Extent::Static(1)],
            value: 1.0,
        },
    );
    let (is_future, _neg_infinity) = causal_mask(&mut program).expect("causal mask lowers");
    let cached_len = input_leaf(&mut program, DType::Float32, Vec::new(), "cached_len");

    let head8_shape = alloc::vec![Extent::Static(1), Extent::Static(1), Extent::Static(8)];
    let attn_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(1)],
        "attn_norm_weight",
    );
    let ffn_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(1)],
        "ffn_norm_weight",
    );
    let q_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        norm_shape.clone(),
        "q_norm_weight",
    );
    let k_norm_weight = input_leaf(&mut program, DType::Float32, norm_shape, "k_norm_weight");
    let wq_gate = input_leaf(&mut program, DType::Float32, head8_shape, "wq_gate");
    let wk = input_leaf(&mut program, DType::Float32, head4_shape.clone(), "wk");
    let wv = input_leaf(&mut program, DType::Float32, head4_shape.clone(), "wv");
    let wo = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(1),
            Extent::Static(1),
            Extent::Static(4),
            Extent::Static(1)
        ],
        "wo",
    );
    let w_gate = input_leaf(&mut program, DType::Float32, ffn_shape.clone(), "w_gate");
    let w_up = input_leaf(&mut program, DType::Float32, ffn_shape.clone(), "w_up");
    let w_down = input_leaf(&mut program, DType::Float32, ffn_shape, "w_down");
    let k_first_cache = input_leaf(
        &mut program,
        DType::Float32,
        cache4_shape.clone(),
        "k_first_cache",
    );
    let k_second_cache =
        input_leaf(&mut program, DType::Float32, cache4_shape, "k_second_cache");
    let k_pass_cache = input_leaf(
        &mut program,
        DType::Float32,
        cache_pass_shape,
        "k_pass_cache",
    );
    let v_cache = input_leaf(&mut program, DType::Float32, cache_v_shape, "v_cache");

    let (x_next, (rotated_k_first, rotated_k_second, k_pass, v_new)) =
        append_qwen35_dense_attention_layer(
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
            1,
            2,
            4,
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
        )
        .expect("dense attention layer lowers");

    let x_data = [2.0f32];
    let eps_data = [eps_value];
    let cos_data = [0.0f32];
    let sin_data = [1.0f32];
    let attn_norm_data = [1.0f32];
    let ffn_norm_data = [1.0f32];
    let q_norm_data = [1.0f32, 1.0, 1.0, 1.0];
    let k_norm_data = [1.0f32, 1.0, 1.0, 1.0];
    let wq_gate_data = [
        1.0f32,
        1.0,
        1.0,
        1.0,
        gate_data[0],
        gate_data[1],
        gate_data[2],
        gate_data[3],
    ];
    let wk_data = [1.0f32, 1.0, 1.0, 1.0];
    let wv_data = [1.0f32, 1.0, 1.0, 1.0];
    let wo_data = [1.0f32, 1.0, 1.0, 1.0];
    let w_gate_data = [1.0f32];
    let w_up_data = [1.0f32];
    let w_down_data = [1.0f32];
    let empty: [f32; 0] = [];
    let cached_len_data = [0.0f32];

    let evaluated = crate::cpu::evaluate_named(
        &program,
        &[1, 0],
        &[
            ("x", &x_data),
            ("eps", &eps_data),
            ("cos", &cos_data),
            ("sin", &sin_data),
            ("cached_len", &cached_len_data),
            ("attn_norm_weight", &attn_norm_data),
            ("ffn_norm_weight", &ffn_norm_data),
            ("q_norm_weight", &q_norm_data),
            ("k_norm_weight", &k_norm_data),
            ("wq_gate", &wq_gate_data),
            ("wk", &wk_data),
            ("wv", &wv_data),
            ("wo", &wo_data),
            ("w_gate", &w_gate_data),
            ("w_up", &w_up_data),
            ("w_down", &w_down_data),
            ("k_first_cache", &empty),
            ("k_second_cache", &empty),
            ("k_pass_cache", &empty),
            ("v_cache", &empty),
        ],
        &[x_next, rotated_k_first, rotated_k_second, k_pass, v_new],
    )
    .expect("dense attention layer evaluates");

    let (x_next_values, _) = evaluated.get(x_next).expect("x_next present");
    let (k_first_values, _) = evaluated.get(rotated_k_first).expect("k_first present");
    let (k_second_values, _) = evaluated.get(rotated_k_second).expect("k_second present");
    let (k_pass_values, _) = evaluated.get(k_pass).expect("k_pass present");
    let (v_new_values, _) = evaluated.get(v_new).expect("v_new present");

    (
        x_next_values[0],
        k_first_values[0],
        k_second_values[0],
        k_pass_values[0],
        v_new_values[0],
    )
}

/// Builds [`append_qwen35_dense_attention_only`]'s (or its `_with_taps`
/// sibling's) own tiny fixture inputs, at the same fixed dims
/// [`evaluate_dense_attention_test_program`] hand-computes against
/// (`embedding = 1`, `attn_head_dim = 4`, `rotary_dim = 2`,
/// `kv_heads = query_heads = 1`), so the same input NodeIds and byte
/// values feed both builders under test.
#[allow(clippy::type_complexity)]
fn dense_attention_only_test_inputs(
    program: &mut Vec<Op>,
) -> (
    NodeId,
    NodeId,
    NodeId,
    NodeId,
    NodeId,
    NodeId,
    NodeId,
    NodeId,
    NodeId,
    NodeId,
    NodeId,
    NodeId,
    NodeId,
    NodeId,
    NodeId,
    NodeId,
    NodeId,
    NodeId,
    NodeId,
    NodeId,
    NodeId,
    NodeId,
) {
    let scalar_shape = alloc::vec![Extent::Symbolic(0), Extent::Static(1)];
    let rotary_shape = alloc::vec![Extent::Symbolic(0), Extent::Static(1)];
    let head4_shape = alloc::vec![Extent::Static(1), Extent::Static(1), Extent::Static(4)];
    let head8_shape = alloc::vec![Extent::Static(1), Extent::Static(1), Extent::Static(8)];
    let cache4_shape = alloc::vec![Extent::Symbolic(1), Extent::Static(1), Extent::Static(1)];
    let cache_pass_shape =
        alloc::vec![Extent::Symbolic(1), Extent::Static(1), Extent::Static(2)];
    let cache_v_shape = alloc::vec![Extent::Symbolic(1), Extent::Static(1), Extent::Static(4)];
    let norm_shape = alloc::vec![Extent::Static(4)];

    let x = input_leaf(program, DType::Float32, scalar_shape, "x");
    let inv_dim = scalar_constant(program, 1.0);
    let eps = input_leaf(
        program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0)],
        "eps",
    );
    let ones = scalar_constant(program, 1.0);
    let inv_sqrt_attn_head_dim = scalar_constant(program, 0.5);
    let inv_attn_head_dim = scalar_constant(program, 0.25);
    let cos_new = input_leaf(program, DType::Float32, rotary_shape.clone(), "cos");
    let sin_new = input_leaf(program, DType::Float32, rotary_shape, "sin");
    let group_ones = op::append(
        program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(1), Extent::Static(1)],
            value: 1.0,
        },
    );
    let (is_future, _neg_infinity) = causal_mask(program).expect("causal mask lowers");
    let cached_len = input_leaf(program, DType::Float32, Vec::new(), "cached_len");

    let attn_norm_weight = input_leaf(
        program,
        DType::Float32,
        alloc::vec![Extent::Static(1)],
        "attn_norm_weight",
    );
    let q_norm_weight =
        input_leaf(program, DType::Float32, norm_shape.clone(), "q_norm_weight");
    let k_norm_weight = input_leaf(program, DType::Float32, norm_shape, "k_norm_weight");
    let wq_gate = input_leaf(program, DType::Float32, head8_shape, "wq_gate");
    let wk = input_leaf(program, DType::Float32, head4_shape.clone(), "wk");
    let wv = input_leaf(program, DType::Float32, head4_shape, "wv");
    let wo = input_leaf(
        program,
        DType::Float32,
        alloc::vec![
            Extent::Static(1),
            Extent::Static(1),
            Extent::Static(4),
            Extent::Static(1)
        ],
        "wo",
    );
    let k_first_cache = input_leaf(
        program,
        DType::Float32,
        cache4_shape.clone(),
        "k_first_cache",
    );
    let k_second_cache = input_leaf(program, DType::Float32, cache4_shape, "k_second_cache");
    let k_pass_cache = input_leaf(program, DType::Float32, cache_pass_shape, "k_pass_cache");
    let v_cache = input_leaf(program, DType::Float32, cache_v_shape, "v_cache");

    (
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
    )
}

/// [`append_qwen35_dense_attention_only_with_taps`] must build the
/// byte-identical program to its thin-wrapper sibling
/// [`append_qwen35_dense_attention_only`] -- the taps variant only
/// returns extra `NodeId`s into the same program, never a structurally
/// different one, mirroring
/// `layer_taps_variant_matches_the_plain_program_and_returns_one_tap_per_layer`'s
/// own invariant for the MoE layer-taps builder.
#[test]
fn dense_attention_only_and_with_taps_produce_the_same_program() {
    let mut plain_program = Vec::new();
    let (
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
    ) = dense_attention_only_test_inputs(&mut plain_program);
    let (plain_residual, plain_roots) = append_qwen35_dense_attention_only(
        &mut plain_program,
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
        1,
        2,
        4,
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
    )
    .expect("plain dense attention program lowers");

    let mut taps_program = Vec::new();
    let (
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
    ) = dense_attention_only_test_inputs(&mut taps_program);
    let (taps_residual, taps) = append_qwen35_dense_attention_only_with_taps(
        &mut taps_program,
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
        1,
        2,
        4,
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
    )
    .expect("taps dense attention program lowers");

    assert_eq!(
        plain_program, taps_program,
        "the taps variant must build the identical graph -- it only returns extra \
         NodeIds into the same program, never a structurally different one"
    );
    assert_eq!(plain_residual, taps_residual);
    assert_eq!(
        plain_roots,
        (
            taps.rotated_k_new_first,
            taps.rotated_k_new_second,
            taps.k_pass,
            taps.v_new
        )
    );
}

/// [`qwen35_forward_program`]'s whole-program wiring, both layer kinds
/// in one small stack (`block_count = 4`, `full_attention_interval =
/// 2`, so layers 0,2 are SSM and layers 1,3 are dense attention, per
/// its own `(layer + 1) % full_attention_interval == 0` doc) --
/// mirroring [`the_whole_mistral_forward_pass_infers_at_real_dimensions`]'s
/// own "lowers, then `shape::infer` succeeds" scope, not a numeric
/// check (the mixer's own hand-computed test below already owns that).
/// `symbols = [1, 0]`: one new decode-step token, an empty dense-attention
/// KV cache -- [`append_mistral_cached_layer`]'s own doc already proves
/// `cached_len == 0` degenerates to plain self-attention with no special
/// case.
#[test]
fn the_whole_qwen35_forward_pass_infers_at_real_dimensions() {
    let (program, logits, roots) =
        qwen35_forward_program(100, 8, 16, 2, 1, 4, 8, 4, 2, 2, 2, 1, 4, 3, 1e-5)
            .expect("the whole qwen35 forward pass lowers to a program");

    assert_eq!(roots.len(), 4, "one root set per block");
    assert!(
        matches!(roots[0], Qwen35LayerRoots::Ssm { .. }),
        "layer 0 is SSM: (0 + 1) % 2 != 0"
    );
    assert!(
        matches!(roots[1], Qwen35LayerRoots::DenseAttention(_)),
        "layer 1 is dense attention: (1 + 1) % 2 == 0"
    );
    assert!(
        matches!(roots[2], Qwen35LayerRoots::Ssm { .. }),
        "layer 2 is SSM: (2 + 1) % 2 != 0"
    );
    assert!(
        matches!(roots[3], Qwen35LayerRoots::DenseAttention(_)),
        "layer 3 is dense attention: (3 + 1) % 2 == 0"
    );

    crate::shape::infer(&program, &[1, 0])
        .expect("the whole qwen35 forward pass infers at a real decode step");
    let _ = logits;
}

#[test]
fn qwen35_last_row_projection_has_one_output_row() {
    let (program, logits, _roots) = qwen35_forward_program_with_last_row(
        100, 8, 16, 2, 1, 4, 8, 4, 2, 2, 2, 1, 4, 3, 1e-5, true,
    )
    .expect("the last-row qwen35 program lowers");
    let shapes = crate::shape::infer(&program, &[3, 0])
        .expect("the last-row qwen35 program infers for a three-token prompt");
    assert_eq!(shapes.of(logits), &[1, 100]);
}

/// Real-checkpoint regression: `qwen35moe` layer 3 (the first
/// full-attention layer), position 0 -- the q/gate projection chain
/// [`append_qwen35_dense_attention_only_with_taps`] builds (`qg_product`
/// -> `qg_raw` reduce -> [`per_head_channel_range`] narrow, spec.rs
/// 4771-4794) used to return UNRELATED row-0 values between a 13-row
/// prefill evaluation and a 1-row evaluation of the identical row, even
/// though the two evaluations' row-0 input is bit-identical --
/// `bind::BoundOpBuilder::quarantine_broadcast_operands`'s own
/// size-only heuristic (`child_extent < reduce_extent`) force-materialized
/// the packed `Q4_K` weight side of the product whenever the reduce's
/// OWN leading (batch) axis made the fused weight subtree's extent
/// smaller than the whole reduce's extent -- which is every multi-row
/// prefill -- undoing the fusion `run_reduce_quantized`'s fast path
/// needs and falling back to `materialize_quantized_weight_output`'s
/// dequantize-without-transpose, read back through the weight's
/// DECLARED (mismatched) axis order. Fixed in `bind.rs` by exempting the
/// packed operand of a `composed_packed_product_activation`-shaped
/// product from quarantine at every recursion depth. Real
/// `qwen3.6:35b-a3b` dims: `embedding = 2048`, `query_heads = 16`,
/// `attn_head_dim = 256` (`omega/tests/cached_attention_partial_rotary_parity.rs`'s
/// own `kv_heads = 2, group = 8 -> 16 query heads`), `attn_q.weight`
/// packed `[Q | gate]` per head, Q4_K quantized, reshaped through the
/// SAME broadcast-multiply-by-ones trick production uses
/// (`proxima-model-interop/src/qwen35moe/program.rs`'s own `wq_flat` ->
/// `wq_gate`) -- the real production shape, not a shrunk stand-in.
#[test]
fn qg_product_qg_raw_per_head_channel_range_matches_between_thirteen_row_and_one_row_eval() {
    use proxima_gguf::quant::q4_k::{BLOCK_BYTES, QK_K, quantize};

    const EMBEDDING: usize = 2048;
    const QUERY_HEADS: usize = 16;
    const ATTN_HEAD_DIM: usize = 256;
    const QG_WIDTH: usize = ATTN_HEAD_DIM * 2;
    const ROWS: usize = 13;

    let normed_data = synth_row(7, ROWS * EMBEDDING, 1.0);
    let weight_native = synth_row(11, EMBEDDING * QUERY_HEADS * QG_WIDTH, 1.0);

    let blocks_per_row = EMBEDDING / QK_K;
    let row_bytes = blocks_per_row * BLOCK_BYTES;
    let mut packed = alloc::vec![0u8; QUERY_HEADS * QG_WIDTH * row_bytes];
    for (row, out_block) in weight_native
        .as_chunks::<EMBEDDING>()
        .0
        .iter()
        .zip(packed.chunks_exact_mut(row_bytes))
    {
        quantize(row, out_block).expect("embedding is a QK_K multiple");
    }

    fn build(program: &mut Vec<Op>, sequence: Extent) -> (NodeId, NodeId) {
        let normed = input_leaf(
            program,
            DType::Float32,
            alloc::vec![sequence, Extent::Static(EMBEDDING as u32)],
            "normed",
        );
        let wq_flat = input_leaf(
            program,
            DType::Float32,
            alloc::vec![
                Extent::Static(EMBEDDING as u32),
                Extent::Static((QUERY_HEADS * QG_WIDTH) as u32)
            ],
            "wq_flat",
        );
        let qg_head_ones = op::append(
            program,
            Op::Constant {
                dtype: DType::Float32,
                shape: alloc::vec![
                    Extent::Static(QUERY_HEADS as u32),
                    Extent::Static(QG_WIDTH as u32)
                ],
                value: 1.0,
            },
        );
        let wq_gate = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[
                (
                    wq_flat,
                    alloc::format!("i,{QG_WIDTH}*h+c->ihc").as_str(),
                ),
                (qg_head_ones, "hc->ihc"),
            ],
        )
        .expect("wq_gate reshape lowers");
        let qg_product = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(normed, "si->shci"), (wq_gate, "ihc->shci")],
        )
        .expect("qg_product lowers");
        let qg_raw = reduce(
            program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            qg_product,
            "shci->shci",
            "shc->shci",
        )
        .expect("qg_raw lowers");
        let q_split =
            per_head_channel_range(program, qg_raw, "h", QG_WIDTH as u32, 0, ATTN_HEAD_DIM as u32)
                .expect("q_split lowers");
        (qg_raw, q_split)
    }

    let mut wide_program = Vec::new();
    let (qg_raw_wide, q_split_wide) = build(&mut wide_program, Extent::Symbolic(0));
    let wide_result = crate::cpu::evaluate_quantized_named(
        &wide_program,
        &[ROWS as u64],
        &[
            ("normed", crate::cpu::QuantizedBlock::Float32(&normed_data)),
            ("wq_flat", crate::cpu::QuantizedBlock::Q4K(&packed)),
        ],
        &[qg_raw_wide, q_split_wide],
    )
    .expect("13-row evaluation lowers and executes");
    let (wide_q_split, _) = wide_result.get(q_split_wide).expect("q_split present");
    let (wide_qg_raw, _) = wide_result.get(qg_raw_wide).expect("qg_raw present");

    let per_row_width = QUERY_HEADS * ATTN_HEAD_DIM;
    let qg_raw_row_width = QUERY_HEADS * QG_WIDTH;

    for row in 0..ROWS {
        let row_input = normed_data[row * EMBEDDING..(row + 1) * EMBEDDING].to_vec();
        let mut single_program = Vec::new();
        let (qg_raw_single, q_split_single) = build(&mut single_program, Extent::Symbolic(0));
        let single_result = crate::cpu::evaluate_quantized_named(
            &single_program,
            &[1u64],
            &[
                ("normed", crate::cpu::QuantizedBlock::Float32(&row_input)),
                ("wq_flat", crate::cpu::QuantizedBlock::Q4K(&packed)),
            ],
            &[qg_raw_single, q_split_single],
        )
        .expect("1-row evaluation lowers and executes");
        let (single_q_split, _) = single_result.get(q_split_single).expect("q_split present");
        let (single_qg_raw, _) = single_result.get(qg_raw_single).expect("qg_raw present");

        // independent ground truth: `weight_native` is already in the
        // NATIVE [row=(h*QG_WIDTH+c)][k=embedding] convention
        // `dequantize_row`'s own physical byte order preserves (row
        // outer, k contiguous) -- computed straight from the
        // PRE-quantization f64 data, so this only carries Q4_K's own
        // quantization noise (~1e-2), never the interpreter's own
        // addressing.
        let expected_first_four: Vec<f64> = (0..4)
            .map(|c| {
                (0..EMBEDDING)
                    .map(|embedding_index| {
                        f64::from(row_input[embedding_index])
                            * f64::from(weight_native[c * EMBEDDING + embedding_index])
                    })
                    .sum::<f64>()
            })
            .collect();

        let wide_qg_raw_row = &wide_qg_raw[row * qg_raw_row_width..(row + 1) * qg_raw_row_width];
        for (channel, expected) in expected_first_four.iter().enumerate() {
            let ground_truth_relative =
                (f64::from(single_qg_raw[channel]) - expected).abs() / expected.abs().max(1.0);
            assert!(
                ground_truth_relative <= 5e-3,
                "row {row} channel {channel}: single_qg_raw={} disagrees with the \
                 independent pre-quantization hand computation expected={expected} beyond \
                 Q4_K's own quantization noise floor (relative={ground_truth_relative})",
                single_qg_raw[channel]
            );
        }

        let qg_raw_l2: f64 = wide_qg_raw_row
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt();
        let qg_raw_max_abs_diff = wide_qg_raw_row
            .iter()
            .zip(single_qg_raw.iter())
            .map(|(wide, single)| f64::from((wide - single).abs()))
            .fold(0.0_f64, f64::max);
        let qg_raw_relative = qg_raw_max_abs_diff / qg_raw_l2.max(1e-12);
        assert!(
            qg_raw_relative <= 1e-5,
            "BISECT qg_raw row {row}: max_abs_diff={qg_raw_max_abs_diff} l2_norm={qg_raw_l2} \
             relative={qg_raw_relative} wide[..4]={:?} single[..4]={:?} -- the divergence is \
             already present at qg_raw, before per_head_channel_range narrows it",
            &wide_qg_raw_row[..4],
            &single_qg_raw[..4]
        );

        let wide_row = &wide_q_split[row * per_row_width..(row + 1) * per_row_width];
        let l2_norm: f64 = wide_row
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt();
        let max_abs_diff = wide_row
            .iter()
            .zip(single_q_split.iter())
            .map(|(wide, single)| f64::from((wide - single).abs()))
            .fold(0.0_f64, f64::max);
        let relative = max_abs_diff / l2_norm.max(1e-12);
        assert!(
            relative <= 1e-5,
            "row {row}: max_abs_diff={max_abs_diff} l2_norm={l2_norm} relative={relative} \
             wide[..4]={:?} single[..4]={:?}",
            &wide_row[..4],
            &single_q_split[..4]
        );
    }
}

/// [`the_whole_qwen35_forward_pass_infers_at_real_dimensions`]'s own
/// `(100, 8, 16, 2, 1, 4, 4, 2, 2, 2, 1, 4, 3, 1e-5)` never caught the
/// `attn_q`/`attn_k`/`attn_v`/`attn_output` shape defect this test is
/// named for -- ROOT CAUSE, proved by direct comparison against the
/// real Qwen3.5-2B-Q4_K_M checkpoint's own on-disk tensor dims
/// (`proxima_model_interop::qwen35`'s own
/// `scratch_debug_real_attn_q_dims`-shaped probe, run against the real
/// file): every dense-attention weight was declared using `head_dim`
/// (`rope.dimension_count`, this checkpoint's PARTIAL-rotary width, `64`)
/// as the per-head PROJECTION width too, but the real per-head
/// projection width is `embedding / query_heads` (`256` on the 2B,
/// matching `attn_q_norm.weight`'s/`attn_k_norm.weight`'s own on-disk
/// width exactly, and `qwen3_next`'s own `Qwen3NextAttention.__init__`,
/// `self.head_dim = hidden_size // num_attention_heads`) -- and
/// `attn_q.weight`'s own on-disk width is DOUBLE that again
/// (`query_heads * 256 * 2 = 4096`, not `query_heads * 64 = 512`): a
/// same-width sigmoid gate fused per head
/// (`modeling_qwen3_next.py:293-326`, `torch.chunk(2, dim=-1)` on each
/// head's own `2 * head_dim`-wide block after `.view(..., heads, 2 *
/// head_dim)`), which this program drops (never applies) rather than
/// implements, an accepted extension of this program's existing
/// single-section-RoPE gap.
///
/// The toy test's own `query_heads = 2`, `embedding = 8` degenerate
/// case makes `embedding / query_heads == 4 == head_dim` BY
/// COINCIDENCE (the toy dims were never chosen to keep those two
/// quantities apart), so the toy program's `attn_q`/`attn_k`/`attn_v`
/// declared shapes matched what [`shape::infer`] expected regardless of
/// which formula built them -- the defect is invisible at any dimension
/// set where `embedding / query_heads == head_dim`, which is every toy
/// dimension set this module's own tests use and no real checkpoint's
/// own numbers. This test is the fix for THAT gap: real per-head
/// dimensions, not toy ones that happen to collide.
#[test]
fn the_whole_qwen35_forward_pass_infers_at_the_2b_checkpoints_real_dimensions() {
    // Qwen3.5-2B-Q4_K_M's own metadata: vocab=151936, embedding=2048,
    // feed_forward=6144, query_heads=8, kv_heads=2, head_dim=64
    // (rope.dimension_count), block_count=24, full_attention_interval=4,
    // ssm_state_size=128, ssm_time_step_rank=16, ssm_group_count=16,
    // ssm_inner_size=2048, ssm_conv_kernel=4, rms_epsilon=1e-6.
    let (program, _logits, roots) = qwen35_forward_program(
        151936, 2048, 6144, 8, 2, 64, 256, 24, 4, 128, 16, 16, 2048, 4, 1e-6,
    )
    .expect("the real 2b's own dimensions lower to a program");

    assert_eq!(roots.len(), 24, "one root set per block");
    assert!(
        matches!(roots[3], Qwen35LayerRoots::DenseAttention(_)),
        "layer 3 is dense attention: (3 + 1) % 4 == 0"
    );
    assert!(
        matches!(roots[0], Qwen35LayerRoots::Ssm { .. }),
        "layer 0 is SSM: (0 + 1) % 4 != 0"
    );

    // prefill (6 new tokens, empty cache) and three decode steps against
    // a growing cache -- the shapes this program actually runs under
    // `proxima_model_interop::generate`, not just a single decode step.
    for (new_count, cached_len) in [(6u64, 0u64), (1, 6), (1, 7), (1, 8)] {
        crate::shape::infer(&program, &[new_count, cached_len]).unwrap_or_else(|err| {
            panic!(
                "the real 2b's own dimensions infer at new_count={new_count} \
                 cached_len={cached_len}: {err:?}"
            )
        });
    }
}

/// [`append_qwen35_ssm_mixer`] against the hand-derivation in
/// [`build_ssm_mixer_test_program`]'s own doc: `history = [q=3, k=-2,
/// v0=1, v1=2]` conv-blends straight through (new-token tap weighted by
/// a zero `qkv_mixed`), `silu` gives `q_raw = 2.85772238`, `k_raw =
/// -0.23840584`, `v_conv = [0.73105858, 1.76159416]`; width-1 `l2norm`
/// collapses `q_conv = 1`, `k_conv = -1`; the recurrence (`decay = 1`,
/// `beta = 0.5`, zero `state_in`) gives `state_out = out = [-0.36552929,
/// -0.88079708]` per group; the gated RMSNorm's width-1 mean-square
/// gives `normed_out = sign(out) = -1` for both groups, so
/// `normed_out_gamma = -2`, `gated_out = normed_out_gamma * silu(z)`
/// with `z = [1, 2]` (same `silu` values as `v`) gives `[-1.46211716,
/// -3.52318831]`, and the `[1, 1]` output weight sums those into `cur =
/// -4.98530547`, `mixer_out = x + cur = -3.98530547`.
#[proxima::test]
async fn qwen35_ssm_mixer_matches_a_hand_computed_decode_step() {
    let (program, mixer_out, taps) = build_ssm_mixer_test_program(GdnOutputGate::Silu);

    let x_data = [1.0f32];
    let eps_data = [0.0f32];
    let head_eps_data = [0.0f32, 0.0];
    let attn_norm_weight_data = [1.0f32];
    let wqkv_data = [0.0f32, 0.0, 0.0, 0.0];
    let wqkv_gate_data = [1.0f32, 2.0];
    let conv_weight_data = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let conv_history_in_data = [3.0f32, -2.0, 1.0, 2.0];
    let ssm_beta_data = [0.0f32, 0.0];
    let ssm_alpha_data = [0.0f32, 0.0];
    let ssm_dt_bias_data = [0.0f32, 0.0];
    let ssm_a_data = [0.0f32, 0.0];
    let ssm_norm_weight_data = [2.0f32];
    let ssm_out_data = [1.0f32, 1.0];
    let state_in_data = [0.0f32, 0.0];

    let evaluated = crate::cpu::evaluate_named(
        &program,
        &[1],
        &[
            ("x", &x_data),
            ("eps", &eps_data),
            ("head_eps", &head_eps_data),
            ("attn_norm_weight", &attn_norm_weight_data),
            ("wqkv", &wqkv_data),
            ("wqkv_gate", &wqkv_gate_data),
            ("conv_weight", &conv_weight_data),
            ("conv_history_in", &conv_history_in_data),
            ("ssm_beta", &ssm_beta_data),
            ("ssm_alpha", &ssm_alpha_data),
            ("ssm_dt_bias", &ssm_dt_bias_data),
            ("ssm_a", &ssm_a_data),
            ("ssm_norm_weight", &ssm_norm_weight_data),
            ("ssm_out", &ssm_out_data),
            ("state_in", &state_in_data),
        ],
        &[
            mixer_out,
            taps.qkv_mixed,
            taps.query,
            taps.key,
            taps.value,
            taps.gate,
            taps.beta,
            taps.delta_out,
            taps.state_out,
        ],
    )
    .expect("ssm mixer evaluates");

    let (mixer_out_values, _) = evaluated.get(mixer_out).expect("mixer_out present");
    let (qkv_mixed_values, _) = evaluated.get(taps.qkv_mixed).expect("qkv_mixed present");
    let (state_out_values, _) = evaluated.get(taps.state_out).expect("state_out present");
    let mut scanned_state = state_in_data;
    let mut scanned_output = [0.0_f32; 2];
    crate::cpu::run_gdn_prefill_scan(crate::cpu::GdnPrefillScan {
        shape: crate::cpu::GdnPrefillShape {
            positions: 1,
            key_dim: 1,
            value_dim: 1,
            heads: 2,
            kv_heads: 2,
        },
        query: evaluated.get(taps.query).expect("query tap present").0,
        key: evaluated.get(taps.key).expect("key tap present").0,
        // `taps.query`/`taps.key` are the SQUEEZED `"dug"`-ordered nodes
        // (`append_qwen35_ssm_mixer_with_taps_and_layout`'s own decode
        // squeeze): `kv_heads` fastest here (`group == 1`), `key_dim ==
        // 1` so its own stride never actually advances an index.
        query_key_head_stride: 1,
        query_key_dim_stride: 2,
        value: evaluated.get(taps.value).expect("value tap present").0,
        gate: evaluated.get(taps.gate).expect("gate tap present").0,
        beta: evaluated.get(taps.beta).expect("beta tap present").0,
        inv_sqrt_key_dim: 1.0,
        state: &mut scanned_state,
        output: &mut scanned_output,
    })
    .expect("production prefill scan evaluates the tapped recurrence");
    let (delta_out_values, _) = evaluated.get(taps.delta_out).expect("delta_out present");

    assert!(
        (mixer_out_values[0] - (-3.985_305_5)).abs() < 1e-4,
        "got {}",
        mixer_out_values[0]
    );
    assert_eq!(
        qkv_mixed_values,
        [0.0, 0.0, 0.0, 0.0],
        "wqkv is zero, so qkv_mixed is zero"
    );
    assert!(
        (state_out_values[0] - (-0.365_529_3)).abs() < 1e-4,
        "got {}",
        state_out_values[0]
    );
    assert!(
        (state_out_values[1] - (-0.880_797_1)).abs() < 1e-4,
        "got {}",
        state_out_values[1]
    );
    assert_eq!(
        scanned_output.as_slice(),
        delta_out_values,
        "the production scan substitution must preserve the graph recurrence output"
    );
    assert_eq!(
        scanned_state.as_slice(),
        state_out_values,
        "the production scan substitution must preserve the carried graph state"
    );
}

/// The opt-in prefill roots retain `s` and use a causal convolution
/// across that axis. With two identical projected rows and unit taps,
/// row zero sees one sample while row one sees both samples.
#[proxima::test]
async fn qwen35_prefill_sequence_taps_preserve_causal_rows() {
    let (program, _, taps) = build_ssm_mixer_test_program(GdnOutputGate::Silu);
    let x_data = [1.0_f32, 1.0];
    let eps_data = [0.0_f32, 0.0];
    let head_eps_data = [0.0_f32, 0.0];
    let attn_norm_weight_data = [1.0_f32];
    let wqkv_data = [1.0_f32, 1.0, 1.0, 1.0];
    let wqkv_gate_data = [1.0_f32, 2.0];
    let conv_weight_data = [1.0_f32; 8];
    let conv_history_in_data = [0.0_f32; 4];
    let ssm_beta_data = [0.0_f32, 0.0];
    let ssm_alpha_data = [0.0_f32, 0.0];
    let ssm_dt_bias_data = [0.0_f32, 0.0];
    let ssm_a_data = [0.0_f32, 0.0];
    let ssm_norm_weight_data = [2.0_f32];
    let ssm_out_data = [1.0_f32, 1.0];
    let state_in_data = [0.0_f32, 0.0];

    let evaluated = crate::cpu::evaluate_named(
        &program,
        &[2],
        &[
            ("x", &x_data),
            ("eps", &eps_data),
            ("head_eps", &head_eps_data),
            ("attn_norm_weight", &attn_norm_weight_data),
            ("wqkv", &wqkv_data),
            ("wqkv_gate", &wqkv_gate_data),
            ("conv_weight", &conv_weight_data),
            ("conv_history_in", &conv_history_in_data),
            ("ssm_beta", &ssm_beta_data),
            ("ssm_alpha", &ssm_alpha_data),
            ("ssm_dt_bias", &ssm_dt_bias_data),
            ("ssm_a", &ssm_a_data),
            ("ssm_norm_weight", &ssm_norm_weight_data),
            ("ssm_out", &ssm_out_data),
            ("state_in", &state_in_data),
        ],
        &[
            taps.query_sequence,
            taps.key_sequence,
            taps.value_sequence,
            taps.gate_sequence,
            taps.beta_sequence,
        ],
    )
    .expect("two-position prefill taps evaluate without the decode recurrence");

    let (queries, query_shape) = evaluated
        .get(taps.query_sequence)
        .expect("query sequence present");
    let (values, value_shape) = evaluated
        .get(taps.value_sequence)
        .expect("value sequence present");
    assert_eq!(query_shape, [2, 1, 1, 2]);
    assert_eq!(value_shape, [2, 1, 1, 2]);
    assert_eq!(queries, [1.0, 1.0, 1.0, 1.0]);

    let first_causal_value = 1.0_f32 / (1.0 + libm::expf(-1.0));
    let second_causal_value = 2.0_f32 / (1.0 + libm::expf(-2.0));
    assert_eq!(
        values,
        [
            first_causal_value,
            first_causal_value,
            second_causal_value,
            second_causal_value,
        ]
    );
}

/// Worked example for the GDN sequence tail. Two rows with distinct
/// residual, recurrence, and gate values must equal evaluating the same
/// algebra once per row and concatenating the two `[1,d]` results. This
/// catches the former `[s,j,u,g] -> [j,u,g]` shape loss: broadcasting
/// either row would disagree with the independently evaluated other row.
#[proxima::test]
async fn qwen35_gdn_sequence_tail_matches_repeated_one_position_graphs() {
    let mut program = Vec::new();
    let x = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(2)],
        "x",
    );
    let delta_out = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Symbolic(0),
            Extent::Static(2),
            Extent::Static(1),
            Extent::Static(1),
        ],
        "delta_out",
    );
    let z = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Symbolic(0),
            Extent::Static(1),
            Extent::Static(1),
            Extent::Static(2),
        ],
        "z",
    );
    let head_eps = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(1), Extent::Static(1)],
        "head_eps",
    );
    let inv_head_v_dim = scalar_constant(&mut program, 0.5);
    let norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(2)],
        "norm_weight",
    );
    let out_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(2), Extent::Static(2)],
        "out_weight",
    );
    let output = append_qwen35_gdn_sequence_tail(
        &mut program,
        Qwen35GdnSequenceTail {
            x,
            delta_out,
            z,
            head_eps,
            inv_head_v_dim,
            norm_weight,
            out_weight,
            head_v_dim: 2,
            kv_heads: 1,
            group: 1,
        },
    )
    .expect("sequence tail lowers");

    let x_data = [1.0_f32, 2.0, 3.0, 4.0];
    let delta_data = [3.0_f32, 4.0, -5.0, 2.0];
    let z_data = [0.5_f32, -1.0, 2.0, 0.25];
    let head_eps_data = [0.01_f32];
    let norm_weight_data = [1.5_f32, 0.5];
    let out_weight_data = [2.0_f32, -1.0, 0.25, 3.0];
    let evaluate = |symbols: &[u64], x: &[f32], delta: &[f32], z: &[f32]| {
        crate::cpu::evaluate_named(
            &program,
            symbols,
            &[
                ("x", x),
                ("delta_out", delta),
                ("z", z),
                ("head_eps", &head_eps_data),
                ("norm_weight", &norm_weight_data),
                ("out_weight", &out_weight_data),
            ],
            &[output],
        )
        .expect("sequence tail evaluates")
        .get(output)
        .expect("sequence output present")
        .0
        .to_vec()
    };

    let batched = evaluate(&[2], &x_data, &delta_data, &z_data);
    let mut repeated = evaluate(&[1], &x_data[..2], &delta_data[..2], &z_data[..2]);
    repeated.extend(evaluate(&[1], &x_data[2..], &delta_data[2..], &z_data[2..]));

    assert_eq!(batched, repeated);
    assert_ne!(batched[..2], batched[2..]);
}

/// Same inputs as the hand-computed decode step above, but with
/// [`GdnOutputGate::Sigmoid`] instead of [`GdnOutputGate::Silu`] --
/// qwen4exp's own gate (reference: PR 27742 line 2895-2897). Proves the
/// flag actually changes the lowered program's output (not silently
/// ignored): `sigmoid(z) != silu(z)` for this test's own `z = [1, 2]`,
/// so `mixer_out` must move away from the Silu path's own hand-computed
/// `-3.985_305_5`.
#[proxima::test]
async fn qwen35_ssm_mixer_sigmoid_gate_moves_the_output_away_from_silu() {
    let (program, mixer_out, _) = build_ssm_mixer_test_program(GdnOutputGate::Sigmoid);

    let x_data = [1.0f32];
    let eps_data = [0.0f32];
    let head_eps_data = [0.0f32, 0.0];
    let attn_norm_weight_data = [1.0f32];
    let wqkv_data = [0.0f32, 0.0, 0.0, 0.0];
    let wqkv_gate_data = [1.0f32, 2.0];
    let conv_weight_data = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let conv_history_in_data = [3.0f32, -2.0, 1.0, 2.0];
    let ssm_beta_data = [0.0f32, 0.0];
    let ssm_alpha_data = [0.0f32, 0.0];
    let ssm_dt_bias_data = [0.0f32, 0.0];
    let ssm_a_data = [0.0f32, 0.0];
    let ssm_norm_weight_data = [2.0f32];
    let ssm_out_data = [1.0f32, 1.0];
    let state_in_data = [0.0f32, 0.0];

    let evaluated = crate::cpu::evaluate_named(
        &program,
        &[1],
        &[
            ("x", &x_data),
            ("eps", &eps_data),
            ("head_eps", &head_eps_data),
            ("attn_norm_weight", &attn_norm_weight_data),
            ("wqkv", &wqkv_data),
            ("wqkv_gate", &wqkv_gate_data),
            ("conv_weight", &conv_weight_data),
            ("conv_history_in", &conv_history_in_data),
            ("ssm_beta", &ssm_beta_data),
            ("ssm_alpha", &ssm_alpha_data),
            ("ssm_dt_bias", &ssm_dt_bias_data),
            ("ssm_a", &ssm_a_data),
            ("ssm_norm_weight", &ssm_norm_weight_data),
            ("ssm_out", &ssm_out_data),
            ("state_in", &state_in_data),
        ],
        &[mixer_out],
    )
    .expect("ssm mixer evaluates");

    let (mixer_out_values, _) = evaluated.get(mixer_out).expect("mixer_out present");
    assert!(
        (mixer_out_values[0] - (-3.985_305_5)).abs() > 1e-3,
        "sigmoid gate must move mixer_out away from the silu path's own -3.985305_5, got {}",
        mixer_out_values[0]
    );
}

/// Proof the mixer reference above can fail: perturbing the conv
/// history's `v0` channel (`1.0 -> -3.0`, a sign flip -- see the data
/// comment below for why a same-sign perturbation alone cannot move this
/// particular degenerate configuration) must move `mixer_out` away from
/// the hand-computed reference -- `v0` feeds `v_split`'s `g = 0` group
/// straight into the recurrence's `value` operand and back out through
/// the gated norm and output projection, so a wrong history value is
/// never silently absorbed.
#[proxima::test]
async fn qwen35_ssm_mixer_hand_computed_check_actually_detects_a_wrong_history_value() {
    let (program, mixer_out, _) = build_ssm_mixer_test_program(GdnOutputGate::Silu);

    let x_data = [1.0f32];
    let eps_data = [0.0f32];
    let head_eps_data = [0.0f32, 0.0];
    let attn_norm_weight_data = [1.0f32];
    let wqkv_data = [0.0f32, 0.0, 0.0, 0.0];
    let wqkv_gate_data = [1.0f32, 2.0];
    let conv_weight_data = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    // v0 perturbed from the reference's 1.0 to -3.0 -- a SIGN flip, not
    // just a magnitude change: `head_v_dim = 1` degenerates the gated
    // RMSNorm's own normalize step to `sign(delta_out)` (the same
    // width-1-l2norm-is-sign identity `q_conv`/`k_conv` already exploit),
    // so a same-sign magnitude perturbation alone never moves
    // `mixer_out` at this degenerate width -- confirmed empirically (a
    // first version of this test perturbed v0 to 2.0 and the assertion
    // could not fail, exactly the failure mode this test's own doc
    // warns against).
    let conv_history_in_data = [3.0f32, -2.0, -3.0, 2.0];
    let ssm_beta_data = [0.0f32, 0.0];
    let ssm_alpha_data = [0.0f32, 0.0];
    let ssm_dt_bias_data = [0.0f32, 0.0];
    let ssm_a_data = [0.0f32, 0.0];
    let ssm_norm_weight_data = [2.0f32];
    let ssm_out_data = [1.0f32, 1.0];
    let state_in_data = [0.0f32, 0.0];

    let evaluated = crate::cpu::evaluate_named(
        &program,
        &[1],
        &[
            ("x", &x_data),
            ("eps", &eps_data),
            ("head_eps", &head_eps_data),
            ("attn_norm_weight", &attn_norm_weight_data),
            ("wqkv", &wqkv_data),
            ("wqkv_gate", &wqkv_gate_data),
            ("conv_weight", &conv_weight_data),
            ("conv_history_in", &conv_history_in_data),
            ("ssm_beta", &ssm_beta_data),
            ("ssm_alpha", &ssm_alpha_data),
            ("ssm_dt_bias", &ssm_dt_bias_data),
            ("ssm_a", &ssm_a_data),
            ("ssm_norm_weight", &ssm_norm_weight_data),
            ("ssm_out", &ssm_out_data),
            ("state_in", &state_in_data),
        ],
        &[mixer_out],
    )
    .expect("ssm mixer evaluates");

    let (mixer_out_values, _) = evaluated.get(mixer_out).expect("mixer_out present");

    assert!(
        (mixer_out_values[0] - (-3.985_305_5)).abs() > 1e-3,
        "a perturbed history v0 must move mixer_out away from the hand-computed reference \
         (if this assertion cannot fail, the test above proves nothing), got {}",
        mixer_out_values[0]
    );
}

/// One `append_mistral_single_range_cached_layer` invocation, minus the
/// `gate_before_up` flag under test -- the minimal single-layer preamble
/// [`mistral_single_range_cached_forward_program`]'s own loop body builds
/// for `block_count = 1`, `query_heads = kv_heads = 1`, `head_dim = 2`,
/// `embedding = feed_forward = 2` (small enough to read by eye, large
/// enough that `w_gate`/`w_up`'s shapes are distinguishable from every
/// other node's).
fn single_range_layer_with_order(
    gate_before_up: bool,
    qk_norm: bool,
) -> Result<(Vec<Op>, NodeId, NodeId), TensorError> {
    let embedding = 2_u32;
    let feed_forward = 2_u32;
    let head_dim = 2_u32;
    let pairs = head_dim / 2;
    let group = 1_u32;

    let mut program = Vec::new();
    let x = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(embedding)],
        "x",
    );
    let inv_dim = scalar_constant(&mut program, 1.0 / embedding as f32);
    let eps = symbolic_leaf(&mut program, DType::Float32, "eps");
    let ones = scalar_constant(&mut program, 1.0);
    let inv_sqrt_head_dim = scalar_constant(&mut program, 1.0 / (head_dim as f32).sqrt());
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
            shape: alloc::vec![Extent::Static(1), Extent::Static(group)],
            value: 1.0,
        },
    );
    let cached_len = input_leaf(&mut program, DType::Float32, Vec::new(), "cached_len");
    let is_future = causal_mask_merged(&mut program, cached_len).expect("causal mask builds");
    let attn_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding)],
        "attn_norm.weight",
    );
    let ffn_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding)],
        "ffn_norm.weight",
    );
    let wq = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(embedding),
            Extent::Static(1),
            Extent::Static(head_dim)
        ],
        "attn_q.weight",
    );
    let wk = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(embedding),
            Extent::Static(1),
            Extent::Static(head_dim)
        ],
        "attn_k.weight",
    );
    let wv = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(embedding),
            Extent::Static(1),
            Extent::Static(head_dim)
        ],
        "attn_v.weight",
    );
    let wo = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Static(1),
            Extent::Static(group),
            Extent::Static(head_dim),
            Extent::Static(embedding),
        ],
        "attn_output.weight",
    );
    let k_even_cache = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Symbolic(1),
            Extent::Static(1),
            Extent::Static(pairs)
        ],
        "kv_cache.k_even",
    );
    let k_odd_cache = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Symbolic(1),
            Extent::Static(1),
            Extent::Static(pairs)
        ],
        "kv_cache.k_odd",
    );
    let v_cache = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![
            Extent::Symbolic(1),
            Extent::Static(1),
            Extent::Static(head_dim)
        ],
        "kv_cache.v",
    );
    let w_gate = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding), Extent::Static(feed_forward)],
        "ffn_gate.weight",
    );
    let w_up = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding), Extent::Static(feed_forward)],
        "ffn_up.weight",
    );
    let w_down = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(feed_forward), Extent::Static(embedding)],
        "ffn_down.weight",
    );
    let qk_norm_weights = qk_norm.then(|| {
        let inv_head_dim = scalar_constant(&mut program, 1.0 / head_dim as f32);
        let q_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(head_dim)],
            "attn_q_norm.weight",
        );
        let k_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(head_dim)],
            "attn_k_norm.weight",
        );
        (q_norm_weight, k_norm_weight, inv_head_dim)
    });

    append_mistral_single_range_cached_layer(
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
        gate_before_up,
    )?;

    Ok((program, w_gate, w_up))
}

/// ROW 373: the single-range builder no longer rejects a qk-norm
/// checkpoint -- it binds the same shape [`append_mistral_cached_layer`]
/// would. This is the acceptance replacement for ROW 372's rejection
/// test: a qk-norm layer must produce the two extra `rmsnorm_per_head`
/// reduces (one per q/k) that a plain interleaved layer does not, and
/// nothing else about its op-kind census should move (same op COUNT per
/// kind elsewhere -- the two-range sibling's own `qk_norm` doc already
/// establishes those two reduces are the ENTIRE cost of this feature).
#[test]
fn single_range_layer_binds_qk_norm_with_two_extra_per_head_norm_reduces() {
    let (plain, _, _) = single_range_layer_with_order(true, false)
        .expect("a plain interleaved layer still builds");
    let (normed, _, _) =
        single_range_layer_with_order(true, true).expect("a qk-norm layer now builds");

    let reduce_count = |program: &[Op]| {
        program
            .iter()
            .filter(|op| matches!(op, Op::Reduce { .. }))
            .count()
    };
    let elementwise_count = |program: &[Op]| {
        program
            .iter()
            .filter(|op| matches!(op, Op::Elementwise { .. }))
            .count()
    };

    assert_eq!(
        reduce_count(&normed),
        reduce_count(&plain) + 2,
        "qk-norm adds exactly the two rmsnorm_per_head reduces (q, k) over the plain layer"
    );
    assert_eq!(
        elementwise_count(&normed),
        elementwise_count(&plain) + 14,
        "rmsnorm_per_head is 7 elementwise ops per call (squared, mean_square, \
         mean_square_eps, rms, inv_rms, normed, gamma-scale), twice (q and k) -- 14 more \
         elementwise ops, nothing else moves"
    );
}

/// The `PROXIMA_ENCODE_ORDER=gate_first|up_first` order-swap knob
/// (`test_support::encode_order_from_env` in
/// `proxima-model-interop`) is honored here at the program-builder
/// level: `append_mistral_single_range_cached_layer`'s `gate_before_up`
/// flag decides only which of the two independent FFN matvecs (both
/// read `normed2`, neither reads the other) is PUSHED first — the node
/// SET and every dependency is unchanged, only their relative order.
#[test]
fn swapping_gate_and_up_order_keeps_dataflow_identical() {
    let (gate_first, w_gate_a, w_up_a) = single_range_layer_with_order(true, false)
        .expect("single-range layer builds under either encode order");
    let (up_first, w_gate_b, w_up_b) = single_range_layer_with_order(false, false)
        .expect("single-range layer builds under either encode order");

    assert_eq!(
        gate_first.len(),
        up_first.len(),
        "swapping encode order must not add or drop a single node"
    );

    let reads_operand = |op: &Op, operand: NodeId| -> bool {
        matches!(op, Op::Elementwise { operands, .. }
            if operands.iter().any(|(id, _)| *id == operand))
    };
    let gate_position = |program: &[Op], w_gate: NodeId| -> usize {
        program
            .iter()
            .position(|op| reads_operand(op, w_gate))
            .expect("a ffn_gate.weight-reading elementwise node must exist")
    };
    let up_position = |program: &[Op], w_up: NodeId| -> usize {
        program
            .iter()
            .position(|op| reads_operand(op, w_up))
            .expect("a ffn_up.weight-reading elementwise node must exist")
    };

    let gate_before_gate_first = gate_position(&gate_first, w_gate_a);
    let up_before_gate_first = up_position(&gate_first, w_up_a);
    assert!(
        gate_before_gate_first < up_before_gate_first,
        "gate_before_up=true must encode ffn_gate ({gate_before_gate_first}) before \
         ffn_up ({up_before_gate_first})"
    );

    let gate_before_up_first = gate_position(&up_first, w_gate_b);
    let up_before_up_first = up_position(&up_first, w_up_b);
    assert!(
        up_before_up_first < gate_before_up_first,
        "gate_before_up=false must encode ffn_up ({up_before_up_first}) before \
         ffn_gate ({gate_before_up_first})"
    );

    // same node SET, different order: every op kind/shape present in one
    // program appears the same number of times in the other, just at a
    // different index -- a multiset comparison over each op's discriminant
    // plus its dtype (position-independent, unlike operand `NodeId`s,
    // which legitimately renumber when the gate/up pair swaps).
    let signature = |op: &Op| -> (core::mem::Discriminant<Op>, DType) {
        let dtype = match op {
            Op::Input { dtype, .. }
            | Op::Elementwise { dtype, .. }
            | Op::Constant { dtype, .. }
            | Op::Iota { dtype, .. } => *dtype,
            Op::Reduce(reduce) => reduce.dtype,
        };
        (core::mem::discriminant(op), dtype)
    };
    let mut gate_first_signatures: Vec<_> = gate_first.iter().map(signature).collect();
    let mut up_first_signatures: Vec<_> = up_first.iter().map(signature).collect();
    gate_first_signatures.sort_by_key(|(discriminant, dtype)| {
        (format!("{discriminant:?}"), format!("{dtype:?}"))
    });
    up_first_signatures.sort_by_key(|(discriminant, dtype)| {
        (format!("{discriminant:?}"), format!("{dtype:?}"))
    });
    assert_eq!(
        gate_first_signatures, up_first_signatures,
        "the two programs must carry the identical multiset of op kinds -- \
         the swap must reorder nodes, never add, drop, or retype one"
    );
}

/// Hand-worked 6-query x 6-key sliding-window causal mask, window = 3: a
/// query at `s` attends keys `s`, `s-1`, `s-2` (clamped at 0) and nothing
/// else. `true` marks a masked (disallowed) cell.
///
/// ```text
///        k=0    k=1    k=2    k=3    k=4    k=5
/// q=0  allow  mask   mask   mask   mask   mask
/// q=1  allow  allow  mask   mask   mask   mask
/// q=2  allow  allow  allow  mask   mask   mask
/// q=3  mask   allow  allow  allow  mask   mask
/// q=4  mask   mask   allow  allow  allow  mask
/// q=5  mask   mask   mask   allow  allow  allow
/// ```
#[test]
fn causal_mask_windowed_matches_the_hand_worked_six_by_six_window_three_table() {
    const SEQUENCE: usize = 6;
    const WINDOW: u32 = 3;
    #[rustfmt::skip]
    let expected_masked: [[bool; SEQUENCE]; SEQUENCE] = [
        [false, true,  true,  true,  true,  true ],
        [false, false, true,  true,  true,  true ],
        [false, false, false, true,  true,  true ],
        [true,  false, false, false, true,  true ],
        [true,  true,  false, false, false, true ],
        [true,  true,  true,  false, false, false],
    ];

    let mut program = Vec::new();
    let (is_masked, _neg_infinity) =
        causal_mask_windowed(&mut program, Some(WINDOW)).expect("windowed causal mask lowers");

    let symbols = [SEQUENCE as u64];
    let blocks: [&[f32]; 0] = [];
    let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");
    let evaluated = crate::cpu::evaluate_parallel(&program, &symbols, &blocks, &[is_masked], workers)
        .expect("the windowed causal mask evaluates");

    let (mask, _shape) = evaluated
        .get(is_masked)
        .expect("the mask node was requested");
    assert_eq!(mask.len(), SEQUENCE * SEQUENCE, "a vacuous mask proves nothing");

    let mut checked = 0usize;
    for (query, row) in mask.as_chunks::<SEQUENCE>().0.iter().enumerate() {
        for (key, &value) in row.iter().enumerate() {
            let masked = value != 0.0;
            assert_eq!(
                masked, expected_masked[query][key],
                "query {query} key {key}: expected masked={}, found masked={masked}",
                expected_masked[query][key]
            );
            checked += 1;
        }
    }
    assert_eq!(checked, SEQUENCE * SEQUENCE, "every cell must be checked, not a subset");
}

/// `window = None` must reproduce [`causal_mask`]'s own program and output
/// byte-for-byte -- the existing full-causal mask is the oracle, and the
/// windowed builder's `None` branch must not diverge from it even by an
/// algebraically-equivalent rewrite.
#[test]
fn causal_mask_windowed_with_no_window_matches_causal_mask_exactly() {
    const SEQUENCE: usize = 6;

    let mut plain_program = Vec::new();
    let (plain_is_future, _plain_neg_infinity) =
        causal_mask(&mut plain_program).expect("causal mask lowers");

    let mut windowed_program = Vec::new();
    let (windowed_is_future, _windowed_neg_infinity) =
        causal_mask_windowed(&mut windowed_program, None).expect("windowed causal mask lowers");

    assert_eq!(
        plain_program, windowed_program,
        "window=None must build the identical program, not merely an equivalent one"
    );
    assert_eq!(plain_is_future, windowed_is_future);

    let symbols = [SEQUENCE as u64];
    let blocks: [&[f32]; 0] = [];
    let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");
    let plain_evaluated = crate::cpu::evaluate_parallel(
        &plain_program,
        &symbols,
        &blocks,
        &[plain_is_future],
        workers,
    )
    .expect("the plain causal mask evaluates");
    let windowed_evaluated = crate::cpu::evaluate_parallel(
        &windowed_program,
        &symbols,
        &blocks,
        &[windowed_is_future],
        workers,
    )
    .expect("the windowed causal mask evaluates");

    let (plain_mask, _) = plain_evaluated
        .get(plain_is_future)
        .expect("the plain mask node was requested");
    let (windowed_mask, _) = windowed_evaluated
        .get(windowed_is_future)
        .expect("the windowed mask node was requested");
    assert_eq!(
        plain_mask, windowed_mask,
        "window=None must evaluate to byte-identical values as causal_mask"
    );
}
