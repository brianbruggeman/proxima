use super::*;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
pub(super) mod hyper_connection_tests {
    use super::*;

    /// Deterministic xorshift64, not a real RNG -- reproducible "random"
    /// f32 inputs without a `rand` dependency in this crate's test code.
    fn next_f32(state: &mut u64) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (((*state >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5) * 2.0
    }

    fn filled(state: &mut u64, len: usize) -> Vec<f32> {
        (0..len).map(|_| next_f32(state)).collect()
    }

    fn sigmoid_f64(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn silu_f64(x: f64) -> f64 {
        x * sigmoid_f64(x)
    }

    /// f64 loop from PR 27742 line 2705-2751's own equations, independent
    /// of every builder above -- the oracle
    /// [`append_hyper_connection_mix`]'s test compares against.
    #[allow(clippy::too_many_arguments)]
    fn hc_mix_f64_reference(
        tokens: usize,
        hc: usize,
        embedding: usize,
        low_rank: usize,
        x: &[f32],
        w_norm: &[f32],
        w_down: &[f32],
        w_up: &[f32],
        w_inject: Option<&[f32]>,
        eps: f64,
    ) -> (Vec<f64>, Option<Vec<f64>>) {
        let at_x = |s: usize, h: usize, i: usize| f64::from(x[(s * hc + h) * embedding + i]);
        let at_w_norm = |h: usize, i: usize| f64::from(w_norm[h * embedding + i]);
        let at_w_down =
            |h: usize, i: usize, r: usize| f64::from(w_down[(h * embedding + i) * low_rank + r]);
        let at_w_up = |r: usize, h: usize, i: usize| f64::from(w_up[(r * hc + h) * embedding + i]);

        let mut xn = alloc::vec![0.0f64; tokens * hc * embedding];
        for s in 0..tokens {
            for h in 0..hc {
                let sum_sq: f64 = (0..embedding).map(|i| at_x(s, h, i).powi(2)).sum();
                let inv_rms = 1.0 / ((sum_sq / embedding as f64) + eps).sqrt();
                for i in 0..embedding {
                    xn[(s * hc + h) * embedding + i] = at_x(s, h, i) * inv_rms * at_w_norm(h, i);
                }
            }
        }
        let at_xn = |s: usize, h: usize, i: usize| xn[(s * hc + h) * embedding + i];

        let mut mixed = alloc::vec![0.0f64; tokens * embedding];
        for s in 0..tokens {
            let lo: Vec<f64> = (0..low_rank)
                .map(|r| {
                    let raw: f64 = (0..hc)
                        .flat_map(|h| (0..embedding).map(move |i| (h, i)))
                        .map(|(h, i)| at_xn(s, h, i) * at_w_down(h, i, r))
                        .sum();
                    silu_f64(raw / hc as f64)
                })
                .collect();
            for h in 0..hc {
                for i in 0..embedding {
                    let up: f64 = (0..low_rank).map(|r| lo[r] * at_w_up(r, h, i)).sum();
                    mixed[s * embedding + i] += at_xn(s, h, i) * sigmoid_f64(up);
                }
            }
            for i in 0..embedding {
                mixed[s * embedding + i] /= hc as f64;
            }
        }

        let inject = w_inject.map(|w_inject| {
            let at_w_inject =
                |h: usize, i: usize, o: usize| f64::from(w_inject[(h * embedding + i) * hc + o]);
            let mut inject = alloc::vec![0.0f64; tokens * hc];
            for s in 0..tokens {
                for o in 0..hc {
                    inject[s * hc + o] = (0..hc)
                        .flat_map(|h| (0..embedding).map(move |i| (h, i)))
                        .map(|(h, i)| at_xn(s, h, i) * at_w_inject(h, i, o))
                        .sum();
                }
            }
            inject
        });

        (mixed, inject)
    }

    fn hc_combine_f64_reference(
        tokens: usize,
        hc: usize,
        embedding: usize,
        residual: &[f32],
        block_out: &[f32],
        inject: &[f64],
    ) -> Vec<f64> {
        let mut result = alloc::vec![0.0f64; tokens * hc * embedding];
        for s in 0..tokens {
            for h in 0..hc {
                let weight = 2.0 * sigmoid_f64(inject[s * hc + h] / hc as f64);
                for i in 0..embedding {
                    let residual_value = f64::from(residual[(s * hc + h) * embedding + i]);
                    let block_out_value = f64::from(block_out[s * embedding + i]);
                    result[(s * hc + h) * embedding + i] =
                        residual_value + block_out_value * weight;
                }
            }
        }
        result
    }

    /// Builds a program exercising just [`append_hyper_connection_mix`] at
    /// `(tokens, hc, embedding, low_rank)`, evaluates it on random f32
    /// inputs, and asserts every output element matches
    /// [`hc_mix_f64_reference`]'s independent f64 loop within `1e-5`.
    fn assert_mix_matches_reference(
        tokens: usize,
        hc: usize,
        embedding: usize,
        low_rank: usize,
        with_inject: bool,
        seed: u64,
    ) {
        let mut state = seed;
        let x_data = filled(&mut state, tokens * hc * embedding);
        let w_norm_data = filled(&mut state, hc * embedding);
        let w_down_data = filled(&mut state, hc * embedding * low_rank);
        let w_up_data = filled(&mut state, low_rank * hc * embedding);
        let w_inject_data = with_inject.then(|| filled(&mut state, hc * embedding * hc));
        let eps_data = alloc::vec![1e-6f32; tokens];

        let mut program = Vec::new();
        let x = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Symbolic(0),
                Extent::Static(hc as u32),
                Extent::Static(embedding as u32)
            ],
            "x",
        );
        let w_norm = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(hc as u32), Extent::Static(embedding as u32)],
            "w_norm",
        );
        let w_down = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(hc as u32),
                Extent::Static(embedding as u32),
                Extent::Static(low_rank as u32)
            ],
            "w_down",
        );
        let w_up = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(low_rank as u32),
                Extent::Static(hc as u32),
                Extent::Static(embedding as u32)
            ],
            "w_up",
        );
        let w_inject = with_inject.then(|| {
            input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(hc as u32),
                    Extent::Static(embedding as u32),
                    Extent::Static(hc as u32)
                ],
                "w_inject",
            )
        });
        let inv_dim = scalar_constant(&mut program, 1.0 / embedding as f32);
        let eps = symbolic_leaf(&mut program, DType::Float32, "eps");
        let inv_hc = scalar_constant(&mut program, 1.0 / hc as f32);
        let one = scalar_constant(&mut program, 1.0);

        let (mixed, inject) = append_hyper_connection_mix(
            &mut program,
            x,
            inv_dim,
            eps,
            inv_hc,
            one,
            w_norm,
            w_down,
            w_up,
            w_inject,
        )
        .expect("hyper-connection mix lowers");

        let mut named: Vec<(&str, &[f32])> = alloc::vec![
            ("x", x_data.as_slice()),
            ("w_norm", w_norm_data.as_slice()),
            ("w_down", w_down_data.as_slice()),
            ("w_up", w_up_data.as_slice()),
            ("eps", eps_data.as_slice()),
        ];
        if let Some(w_inject_data) = w_inject_data.as_deref() {
            named.push(("w_inject", w_inject_data));
        }
        let mut outputs = alloc::vec![mixed];
        if let Some(inject) = inject {
            outputs.push(inject);
        }

        let evaluated = crate::cpu::evaluate_named(&program, &[tokens as u64], &named, &outputs)
            .expect("hyper-connection mix evaluates");

        let (mixed_values, _) = evaluated.get(mixed).expect("mixed output present");
        let (expected_mixed, expected_inject) = hc_mix_f64_reference(
            tokens,
            hc,
            embedding,
            low_rank,
            &x_data,
            &w_norm_data,
            &w_down_data,
            &w_up_data,
            w_inject_data.as_deref(),
            1e-6,
        );
        let max_abs_diff_mixed = mixed_values
            .iter()
            .zip(expected_mixed.iter())
            .map(|(actual, expected)| (f64::from(*actual) - expected).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_abs_diff_mixed <= 1e-5,
            "mixed max-abs diff {max_abs_diff_mixed} exceeds 1e-5"
        );

        if let Some(inject_node) = inject {
            let (inject_values, _) = evaluated.get(inject_node).expect("inject output present");
            let expected_inject =
                expected_inject.expect("reference computed inject when w_inject was Some");
            let max_abs_diff_inject = inject_values
                .iter()
                .zip(expected_inject.iter())
                .map(|(actual, expected)| (f64::from(*actual) - expected).abs())
                .fold(0.0f64, f64::max);
            assert!(
                max_abs_diff_inject <= 1e-5,
                "inject max-abs diff {max_abs_diff_inject} exceeds 1e-5"
            );
        }
    }

    #[test]
    fn mix_matches_f64_reference_at_hc_2() {
        assert_mix_matches_reference(3, 2, 4, 3, true, 0x517c_c1b7_2722_0a95);
    }

    #[test]
    fn mix_matches_f64_reference_at_hc_4() {
        assert_mix_matches_reference(2, 4, 3, 2, true, 0x9e37_79b9_7f4a_7c15);
    }

    /// The final output mixer's own shape (reference: PR 27742 line
    /// 2860-2862): `w_inject = None`, no scatter weight computed.
    #[test]
    fn mix_matches_f64_reference_for_the_final_mixer_form() {
        assert_mix_matches_reference(2, 2, 3, 2, false, 0xd1b5_4a32_d192_ed03);
    }

    #[test]
    fn combine_matches_f64_reference() {
        let tokens = 3usize;
        let hc = 2usize;
        let embedding = 4usize;
        let mut state = 0xbf58_476d_1ce4_e5b9u64;
        let residual_data = filled(&mut state, tokens * hc * embedding);
        let block_out_data = filled(&mut state, tokens * embedding);
        let inject_data = filled(&mut state, tokens * hc);

        let mut program = Vec::new();
        let residual = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Symbolic(0),
                Extent::Static(hc as u32),
                Extent::Static(embedding as u32)
            ],
            "residual",
        );
        let block_out = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(0), Extent::Static(embedding as u32)],
            "block_out",
        );
        let inject = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(0), Extent::Static(hc as u32)],
            "inject",
        );
        let inv_hc = scalar_constant(&mut program, 1.0 / hc as f32);
        let one = scalar_constant(&mut program, 1.0);
        let two = scalar_constant(&mut program, 2.0);

        let combined = append_hyper_connection_combine(
            &mut program,
            residual,
            block_out,
            inject,
            inv_hc,
            one,
            two,
        )
        .expect("hyper-connection combine lowers");

        let named: Vec<(&str, &[f32])> = alloc::vec![
            ("residual", residual_data.as_slice()),
            ("block_out", block_out_data.as_slice()),
            ("inject", inject_data.as_slice()),
        ];
        let evaluated = crate::cpu::evaluate_named(&program, &[tokens as u64], &named, &[combined])
            .expect("combine evaluates");
        let (combined_values, _) = evaluated.get(combined).expect("combined output present");

        let inject_f64: Vec<f64> = inject_data.iter().map(|value| f64::from(*value)).collect();
        let expected = hc_combine_f64_reference(
            tokens,
            hc,
            embedding,
            &residual_data,
            &block_out_data,
            &inject_f64,
        );
        let max_abs_diff = combined_values
            .iter()
            .zip(expected.iter())
            .map(|(actual, expected)| (f64::from(*actual) - expected).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_abs_diff <= 1e-5,
            "combine max-abs diff {max_abs_diff} exceeds 1e-5"
        );
    }
}

/// [`append_mistral_cached_layer`]'s Qwen3.5 dense-attention counterpart --
/// same cached-attention/online-softmax shape, three real differences from
/// the oracle (`modeling_qwen3_next.py`'s `Qwen3NextAttention.forward`,
/// `apply_rotary_pos_emb`; cross-checked against `qwen35.cpp`'s own
/// `build_layer_attn`) `append_mistral_cached_layer` has no room for:
///
/// 1. Q/K carry a real per-head width (`attn_head_dim`, this checkpoint's
///    own `attention.key_length`) wider than the rotary width (`rotary_dim`,
///    `rope.dimension_count`) -- RoPE only touches the first `rotary_dim`
///    columns, the remaining `attn_head_dim - rotary_dim` ("pass") columns
///    are concatenated back untouched (`modeling_qwen3_next.py:204-214`,
///    `q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]` ... `q_embed
///    = torch.cat([q_embed, q_pass], dim=-1)`). This module has no
///    concatenation primitive, so the "pass" half is never physically
///    rejoined to the "rot" half -- instead every dot product that would
///    read the concatenated vector (the attention score) is split into a
///    rot-range term plus a pass-range term and summed, which is
///    mathematically identical (`(a‖b)·(c‖d) = a·c + b·d` for disjoint
///    ranges) and is exactly the same disjoint-sum trick this function's
///    own `score_cached_even + score_cached_odd` already uses one level
///    down, one level up.
/// 2. RoPE itself is split-half (NEOX/IMROPE style, `x_rot -> (x[..d/2],
///    x[d/2..])`, GGML_ROPE_TYPE_IMROPE's own `rotate_pairs(n_dims,
///    n_dims/2, ...)`, `ggml/src/ggml-cpu/ops.cpp:6210-6211`), not
///    [`append_mistral_cached_layer`]'s interleaved `(2*i, 2*i+1)` pairing.
///    The checkpoint's declared 3-section MRoPE (`rope.dimension_sections`)
///    collapses to this same plain single-section schedule for text-only
///    input: `ggml_mrope_cache_init`'s own `theta_t`/`theta_h`/`theta_w`
///    tracks are initialized from the SAME position (`llama-graph.cpp`'s
///    `llm_graph_input_pos::set_input`, "the 3 first dims are the same" for
///    a text ubatch) and advance by the identical `theta_scale` every pair
///    index, so `theta_h`/`theta_w` are byte-identical to `theta_t` at
///    every pair regardless of which section claims that pair
///    (`ggml_mrope_cache_init`, `ops.cpp:6027-6037`) -- the declared
///    `[11, 11, 10, 0]` split is real machinery for image/video position
///    streams this checkpoint's text-only forward program never feeds.
/// 3. Q's own projection is `q_proj` fused with a same-width sigmoid gate
///    (`attn_q.weight`'s on-disk width is `2 * query_heads * attn_head_dim`,
///    `modeling_qwen3_next.py:267-268`, `torch.chunk(..., 2, dim=-1)` on the
///    LAST axis of each head's own block, `:295-298`), applied to the
///    attention output right before `o_proj`
///    (`attn_output = attn_output * torch.sigmoid(gate)`,
///    `:325-328`; `qwen35.cpp:322-328` runs the identical
///    `ggml_mul(cur, ggml_sigmoid(gate))` before `wo`).
///
/// The full-attention layer [`qwen35_forward_program`] calls once per
/// `full_attention_interval`'th layer -- see it there for the worked
/// example of wiring this builder's cache inputs and outputs.
///
/// Attention block only -- everything up to and including the residual add
/// after `o_proj`, no FFN. [`append_qwen35_dense_attention_layer`] is a thin
/// wrapper adding the dense-FFN tail on top of this; a caller whose FFN is
/// NOT dense (a routed-MoE checkpoint such as `qwen35moe`, which carries no
/// `blk.N.ffn_{gate,up,down}.weight` on its attention layers at all) calls
/// this directly and appends its own FFN + residual against the returned
/// node, the same "per-layer builders are pub so a foreign crate can
/// compose them" contract this module's own
/// `public_builders_compose_a_one_layer_forward_program` test proves for
/// [`append_qwen35_ssm_mixer`].
///
/// Thin wrapper over [`append_qwen35_dense_attention_only_with_taps`] for
/// callers that only need the two roots this signature already returned
/// before taps existed -- byte-identical program, since this only reshapes
/// the return value the shared builder already computed
/// (`dense_attention_only_and_with_taps_produce_the_same_program` proves the
/// two builders emit identical `Vec<Op>` for the dense synth fixture).
#[allow(clippy::too_many_arguments)]
pub fn append_qwen35_dense_attention_only(
    program: &mut Vec<Op>,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    ones: NodeId,
    inv_sqrt_attn_head_dim: NodeId,
    inv_attn_head_dim: NodeId,
    cos_new: NodeId,
    sin_new: NodeId,
    group_ones: NodeId,
    is_future: NodeId,
    cached_len: NodeId,
    group: u32,
    rotary_dim: u32,
    attn_head_dim: u32,
    attn_norm_weight: NodeId,
    q_norm_weight: NodeId,
    k_norm_weight: NodeId,
    wq_gate: NodeId,
    wk: NodeId,
    wv: NodeId,
    wo: NodeId,
    k_first_cache: NodeId,
    k_second_cache: NodeId,
    k_pass_cache: NodeId,
    v_cache: NodeId,
) -> Result<(NodeId, Qwen35DenseAttentionRoots), TensorError> {
    let (residual1, taps) = append_qwen35_dense_attention_only_with_taps(
        program,
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
        rotary_dim,
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
    Ok((
        residual1,
        (
            taps.rotated_k_new_first,
            taps.rotated_k_new_second,
            taps.k_pass,
            taps.v_new,
        ),
    ))
}

/// Every intermediate a caller needs to bisect
/// [`append_qwen35_dense_attention_only`]'s attention block against an
/// independent reference, in the order the builder computes them
/// (`spec.rs` just below). Field naming mirrors [`SsmMixerTaps`]'s own
/// convention -- one field per stage, named after the stage, not the local
/// variable that happened to hold it.
///
/// **Split convention for `q_split`/`gate_split`:** this function receives
/// `wq_gate` ALREADY reshaped to expose a head axis (the fused on-disk
/// `blk.N.attn_q.weight`, `2 * query_heads * attn_head_dim` wide, reshaped
/// by the caller to `[embedding, heads, 2*attn_head_dim]` via the same
/// multiply-by-broadcast-ones view [`append_qwen35_dense_attention_only`]'s
/// caller already uses for `wk`/`wv` -- never a per-head WEIGHT-level slice:
/// splitting a packed quantized weight per head before the real contraction
/// runs breaks `cpu::is_quantized_matmul_operand`'s recognizer,
/// which then derives the packed row length from the wrong axis
/// (`per_head_channel_slice`'s own former call site here, `spec.rs`
/// `qwen35moe_layer3_gated_attention_position0_matches_tapped_reference`
/// RED before this fix). This function itself does the ONE real
/// `qg_raw = x_normed @ wq_gate` contraction, then narrows to `q`/`gate` per
/// head via [`per_head_channel_range`] on that (dense float32) activation,
/// using the PER-HEAD INTERLEAVE convention: each head's own contiguous
/// `2*attn_head_dim` block splits into `[0, attn_head_dim)` (`q_split`) and
/// `[attn_head_dim, 2*attn_head_dim)` (`gate_split`). This matches HF
/// Qwen3.5 (`modeling_qwen3_next.py:267-268,295-298`,
/// `torch.chunk(query_states.view(..., heads, 2*head_dim), 2, dim=-1)` --
/// chunking the LAST axis of a view whose second-to-last axis is `heads` is
/// per-head, not a global halves split across the whole flat width), and is
/// mathematically identical to a fused-then-chunked projection since matmul
/// distributes over disjoint output columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35DenseAttentionTaps {
    /// `x_normed`: the RMS-normalized dense-attention input consumed by the
    /// q/gate, key, and value projections.
    pub normed: NodeId,
    /// `q_raw`: `x_normed @ wq_gate`, narrowed to `[0, attn_head_dim)` per
    /// head, pre-qk-norm.
    pub q_split: NodeId,
    /// `gate_raw`: `x_normed @ wq_gate`, narrowed to
    /// `[attn_head_dim, 2*attn_head_dim)` per head, pre-sigmoid.
    pub gate_split: NodeId,
    /// `q` after per-head RMSNorm (`q_norm_weight`), full `attn_head_dim` width.
    pub q_normed: NodeId,
    /// `k` after per-head RMSNorm (`k_norm_weight`), full `attn_head_dim` width.
    pub k_normed: NodeId,
    /// first half of the split-half-RoPE-rotated `q` (`rotary_dim/2` wide).
    pub q_rot_first: NodeId,
    /// second half of the split-half-RoPE-rotated `q` (`rotary_dim/2` wide).
    pub q_rot_second: NodeId,
    /// first half of the split-half-RoPE-rotated new-position `k`.
    pub k_rot_first: NodeId,
    /// second half of the split-half-RoPE-rotated new-position `k`.
    pub k_rot_second: NodeId,
    /// scaled, causal-masked attention score against the NEW (uncached) key
    /// -- at position 0 this is the only score that exists.
    pub score_new: NodeId,
    /// softmax-weighted sum over `v` (cached + new), pre-gate.
    pub attended: NodeId,
    /// `sigmoid(gate_split)`, broadcast from kv-heads to query-heads.
    pub gate_sigmoid: NodeId,
    /// `attended * gate_sigmoid`, the value `o_proj`'s matmul consumes.
    pub gated_attended: NodeId,
    /// `gated_attended @ wo`, reduced, before the residual add.
    pub o_proj_out: NodeId,
    /// new-position rotated-and-passed-through key half, first RoPE half --
    /// [`Qwen35DenseAttentionRoots`]'s own first element, cached for the
    /// next call.
    pub rotated_k_new_first: NodeId,
    /// [`Qwen35DenseAttentionRoots`]'s own second element.
    pub rotated_k_new_second: NodeId,
    /// [`Qwen35DenseAttentionRoots`]'s own third element (untouched pass-through `k`).
    pub k_pass: NodeId,
    /// [`Qwen35DenseAttentionRoots`]'s own fourth element (new-position `v`).
    pub v_new: NodeId,
}

/// [`append_qwen35_dense_attention_only`]'s full implementation, returning
/// every [`Qwen35DenseAttentionTaps`] intermediate alongside the residual
/// output for a caller that needs to bisect the attention block (q/gate
/// split, qk-norm, rotary, scores, gate, `o_proj`) against an independent
/// reference -- a downstream `qwen35moe`-shaped consumer's own layer-3
/// position-0 divergence investigation is exactly that caller.
#[allow(clippy::too_many_arguments)]
pub fn append_qwen35_dense_attention_only_with_taps(
    program: &mut Vec<Op>,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    ones: NodeId,
    inv_sqrt_attn_head_dim: NodeId,
    inv_attn_head_dim: NodeId,
    cos_new: NodeId,
    sin_new: NodeId,
    group_ones: NodeId,
    is_future: NodeId,
    cached_len: NodeId,
    group: u32,
    rotary_dim: u32,
    attn_head_dim: u32,
    attn_norm_weight: NodeId,
    q_norm_weight: NodeId,
    k_norm_weight: NodeId,
    wq_gate: NodeId,
    wk: NodeId,
    wv: NodeId,
    wo: NodeId,
    k_first_cache: NodeId,
    k_second_cache: NodeId,
    k_pass_cache: NodeId,
    v_cache: NodeId,
) -> Result<(NodeId, Qwen35DenseAttentionTaps), TensorError> {
    let pass_dim = attn_head_dim - rotary_dim;

    let normed = rmsnorm(program, x, attn_norm_weight, inv_dim, eps)?;

    // Single full-width `q_proj` reduce against the packed `wq_gate` leaf --
    // `is_quantized_matmul_operand`'s recognizer (`cpu.rs`) only fires
    // correctly when a quantized weight feeds a Multiply directly into an
    // Add-reduce; splitting `wq_gate` PER HEAD first (the old
    // `per_head_channel_slice`-on-the-weight approach) inserted an extra
    // select-then-reduce between the packed leaf and this contraction, which
    // that recognizer folds into the SAME "quantized matmul" shape and then
    // derives `rows`/`k` from the wrong axis pair (`run_reduce_quantized`
    // treats the reduced flat `heads*2*attn_head_dim` axis as the
    // contraction width instead of `embedding`), corrupting every row byte
    // offset. Splitting the ACTIVATION output below instead keeps the
    // packed leaf's only consumer a genuine one-elementwise-then-reduce
    // matmul, then narrows q/gate per head with the same
    // [`per_head_channel_range`] technique `q_pass`/`k_pass` already use on
    // dense float32 activations just below.
    let qg_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->shci"), (wq_gate, "ihc->shci")],
    )?;
    let qg_raw = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        qg_product,
        "shci->shci",
        "shc->shci",
    )?;
    let q_raw = per_head_channel_range(program, qg_raw, "h", attn_head_dim * 2, 0, attn_head_dim)?;
    let gate_raw = per_head_channel_range(
        program,
        qg_raw,
        "h",
        attn_head_dim * 2,
        attn_head_dim,
        attn_head_dim,
    )?;

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

    // `q_norm`/`k_norm` run on the FULL `attn_head_dim` width, before RoPE
    // ever splits it (`modeling_qwen3_next.py:300-301`,
    // `self.q_norm(query_states.view(hidden_shape))` where `hidden_shape`'s
    // last dim is `self.head_dim` = `attn_head_dim`; `qwen35.cpp:308-317`
    // normalizes `Qcur`/`Kcur` before `ggml_rope_multi` runs).
    let q = rmsnorm_per_head(program, q_raw, q_norm_weight, inv_attn_head_dim, eps, "h")?;
    let k = rmsnorm_per_head(program, k_raw, k_norm_weight, inv_attn_head_dim, eps, "u")?;

    let q_pass = per_head_channel_range(program, q, "h", attn_head_dim, rotary_dim, pass_dim)?;
    let k_pass = per_head_channel_range(program, k, "u", attn_head_dim, rotary_dim, pass_dim)?;

    // Qwen3.5 uses interleaved MRoPE, including on QK-norm layers. The two
    // returned planes remain separate cache roots, but each plane contains
    // one member of every adjacent pair (`2*i`, `2*i+1`).
    let (rotated_q_first, rotated_q_second) =
        fused_rope_pair(program, q, 'h', cos_new, sin_new, RopePairing::Interleaved)?;
    let (rotated_k_new_first, rotated_k_new_second) =
        fused_rope_pair(program, k, 'u', cos_new, sin_new, RopePairing::Interleaved)?;

    let group_map_i = alloc::format!("s,{group}*u+g,i->sugi");
    let q_first_grouped = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (rotated_q_first, group_map_i.as_str()),
            (group_ones, "ug->sugi"),
        ],
    )?;
    let q_second_grouped = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (rotated_q_second, group_map_i.as_str()),
            (group_ones, "ug->sugi"),
        ],
    )?;
    let group_map_p = alloc::format!("s,{group}*u+g,p->sugp");
    let q_pass_grouped = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(q_pass, group_map_p.as_str()), (group_ones, "ug->sugp")],
    )?;

    let score_cached_first_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (q_first_grouped, "sugi->stugi"),
            (k_first_cache, "tui->stugi"),
        ],
    )?;
    let score_cached_first = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_cached_first_product,
        "stugi->stugi",
        "stug->stugi",
    )?;
    let score_cached_second_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (q_second_grouped, "sugi->stugi"),
            (k_second_cache, "tui->stugi"),
        ],
    )?;
    let score_cached_second = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_cached_second_product,
        "stugi->stugi",
        "stug->stugi",
    )?;
    let score_cached_pass_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (q_pass_grouped, "sugp->stugp"),
            (k_pass_cache, "tup->stugp"),
        ],
    )?;
    let score_cached_pass = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_cached_pass_product,
        "stugp->stugp",
        "stug->stugp",
    )?;
    let score_cached_rotated = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[
            (score_cached_first, "stug->stug"),
            (score_cached_second, "stug->stug"),
        ],
    )?;
    let score_cached = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[
            (score_cached_rotated, "stug->stug"),
            (score_cached_pass, "stug->stug"),
        ],
    )?;
    let score_cached_scaled = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (score_cached, "stug->stug"),
            (inv_sqrt_attn_head_dim, "->stug"),
        ],
    )?;
    // `k_first_cache`/`k_second_cache`/`k_pass_cache`/`v_cache` are bound to
    // the CALLER's own bucketed `kv_extent`, not the real `cached_len`
    // (`Qwen35DenseAttentionPadScratch::fill`'s own doc) -- rows
    // `[cached_len, bound_extent)` are zero-padding, not history. Unlike the
    // `Attention` arm, which excludes that padding via the fused
    // `BoundOpKind::CachedAttention` op's own `cached_key_rows` runtime
    // bound, this graph has no such fusion, so the padding is masked here
    // exactly the way [`causal_mask_merged`] masks its own merged range:
    // `key_index >= cached_len` is invalid, scored `-inf` before either
    // softmax pass sees it.
    let neg_infinity = scalar_constant(program, f32::NEG_INFINITY);
    let cached_key_index = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Symbolic(1),
        },
    );
    let cached_len_exclusive_bound = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(cached_len, "->"), (ones, "->")],
    )?;
    let is_cached_padding = elementwise(
        program,
        DType::Float32,
        ScalarOp::Greater,
        &[
            (cached_key_index, "t->t"),
            (cached_len_exclusive_bound, "->t"),
        ],
    )?;
    let score_cached_scaled = elementwise(
        program,
        DType::Float32,
        ScalarOp::Select,
        &[
            (is_cached_padding, "t->stug"),
            (neg_infinity, "->stug"),
            (score_cached_scaled, "stug->stug"),
        ],
    )?;

    let score_new_first_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (q_first_grouped, "sugi->swugi"),
            (rotated_k_new_first, "wui->swugi"),
        ],
    )?;
    let score_new_first = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_new_first_product,
        "swugi->swugi",
        "swug->swugi",
    )?;
    let score_new_second_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (q_second_grouped, "sugi->swugi"),
            (rotated_k_new_second, "wui->swugi"),
        ],
    )?;
    let score_new_second = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_new_second_product,
        "swugi->swugi",
        "swug->swugi",
    )?;
    let score_new_pass_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(q_pass_grouped, "sugp->swugp"), (k_pass, "wup->swugp")],
    )?;
    let score_new_pass = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_new_pass_product,
        "swugp->swugp",
        "swug->swugp",
    )?;
    let score_new_rotated = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[
            (score_new_first, "swug->swug"),
            (score_new_second, "swug->swug"),
        ],
    )?;
    let score_new = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[
            (score_new_rotated, "swug->swug"),
            (score_new_pass, "swug->swug"),
        ],
    )?;
    let score_new_scaled = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (score_new, "swug->swug"),
            (inv_sqrt_attn_head_dim, "->swug"),
        ],
    )?;
    let neg_infinity = scalar_constant(program, f32::NEG_INFINITY);
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

    // per-head sigmoid gate, applied to the attention output before `o_proj`
    // (`modeling_qwen3_next.py:325-328`, `qwen35.cpp:322-328`).
    let group_map_d = alloc::format!("s,{group}*u+g,d->sugd");
    let gate_grouped = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(gate_raw, group_map_d.as_str()), (group_ones, "ug->sugd")],
    )?;
    let neg_attn_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Negate,
        &[(gate_grouped, "sugd->sugd")],
    )?;
    let exp_neg_attn_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Exponential,
        &[(neg_attn_gate, "sugd->sugd")],
    )?;
    let one_plus_exp_attn_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(exp_neg_attn_gate, "sugd->sugd"), (ones, "->sugd")],
    )?;
    let sigmoid_attn_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Reciprocal,
        &[(one_plus_exp_attn_gate, "sugd->sugd")],
    )?;
    let gated_attended = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(attended, "sugd->sugd"), (sigmoid_attn_gate, "sugd->sugd")],
    )?;

    let wo_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(gated_attended, "sugd->sugdo"), (wo, "ugdo->sugdo")],
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

    let taps = Qwen35DenseAttentionTaps {
        normed,
        q_split: q_raw,
        gate_split: gate_raw,
        q_normed: q,
        k_normed: k,
        q_rot_first: rotated_q_first,
        q_rot_second: rotated_q_second,
        k_rot_first: rotated_k_new_first,
        k_rot_second: rotated_k_new_second,
        score_new: score_new_masked,
        attended,
        gate_sigmoid: sigmoid_attn_gate,
        gated_attended,
        o_proj_out: attn_out,
        rotated_k_new_first,
        rotated_k_new_second,
        k_pass,
        v_new,
    };

    Ok((residual1, taps))
}

/// [`append_qwen35_dense_attention_only`] plus the dense (non-MoE) SwiGLU
/// FFN tail `qwen35_forward_program`'s own non-routed checkpoints carry on
/// every layer -- see that function for the worked example of wiring this
/// builder's cache inputs and outputs. A caller whose FFN is routed
/// (`qwen35moe`-shaped) calls [`append_qwen35_dense_attention_only`]
/// directly instead of this wrapper.
#[allow(clippy::too_many_arguments)]
pub fn append_qwen35_dense_attention_layer(
    program: &mut Vec<Op>,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    ones: NodeId,
    inv_sqrt_attn_head_dim: NodeId,
    inv_attn_head_dim: NodeId,
    cos_new: NodeId,
    sin_new: NodeId,
    group_ones: NodeId,
    is_future: NodeId,
    cached_len: NodeId,
    group: u32,
    rotary_dim: u32,
    attn_head_dim: u32,
    attn_norm_weight: NodeId,
    ffn_norm_weight: NodeId,
    q_norm_weight: NodeId,
    k_norm_weight: NodeId,
    wq_gate: NodeId,
    wk: NodeId,
    wv: NodeId,
    wo: NodeId,
    w_gate: NodeId,
    w_up: NodeId,
    w_down: NodeId,
    k_first_cache: NodeId,
    k_second_cache: NodeId,
    k_pass_cache: NodeId,
    v_cache: NodeId,
) -> Result<(NodeId, Qwen35DenseAttentionRoots), TensorError> {
    let (residual1, roots) = append_qwen35_dense_attention_only(
        program,
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
        rotary_dim,
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

    let normed2 = rmsnorm(program, residual1, ffn_norm_weight, inv_dim, eps)?;

    let gate_product2 = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed2, "sd->sdg"), (w_gate, "dg->sdg")],
    )?;
    let ffn_gate = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        gate_product2,
        "sdg->sdg",
        "sg->sdg",
    )?;
    let up_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed2, "sd->sdg"), (w_up, "dg->sdg")],
    )?;
    let up = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        up_product,
        "sdg->sdg",
        "sg->sdg",
    )?;

    let neg_ffn_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Negate,
        &[(ffn_gate, "sg->sg")],
    )?;
    let exp_neg_ffn_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Exponential,
        &[(neg_ffn_gate, "sg->sg")],
    )?;
    let one_plus_exp_ffn_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(exp_neg_ffn_gate, "sg->sg"), (ones, "->sg")],
    )?;
    let sigmoid_ffn_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Reciprocal,
        &[(one_plus_exp_ffn_gate, "sg->sg")],
    )?;
    let silu_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(ffn_gate, "sg->sg"), (sigmoid_ffn_gate, "sg->sg")],
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

    Ok((x_next, roots))
}

