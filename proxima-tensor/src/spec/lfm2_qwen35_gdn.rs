use super::*;

/// A fixed-width causal depthwise convolution (`l_cache` taps, one weight per
/// channel per tap, no cross-channel mixing), built from the existing
/// `Input`/`Elementwise`/`Reduce`/`Iota`/`Constant` vocabulary with no new
/// `Op` -- the pipe question this crate's own rule forces before any new
/// type, answered by writing the expression below rather than by arguing for
/// one.
///
/// `specs/conv2d.toml`'s own doc already proved the naive route is closed:
/// windowing `x` directly with a negative-offset `Affine` map
/// (`s-(l_cache-1)+l`) fails `shape::bounds_check` globally, because an
/// iteration axis always starts at 0 and the check is over the *whole*
/// symbolic extent, not per element -- at `s=0, l=0` the window reaches
/// index `-(l_cache-1)`, unconditionally out of bounds regardless of how
/// large the buffer is. `conv2d.toml` closes that gap by pre-padding its
/// input's own data; this crate's op set has no concat/pad primitive to build
/// that padding for an internal (not caller-supplied) tensor, so this
/// function takes a different, still-existing-primitives route: it never
/// forms the negative index at all.
///
/// `raw_position = s + l - (l_cache - 1)` is computed as data (two `Iota`s
/// plus a `Constant` offset, exactly [`causal_mask`]'s own `is_future`
/// composition), `clamped_position = max(raw_position, 0)` (always inside
/// `[0, s_max]`, since `raw_position`'s own maximum, reached at
/// `l = l_cache - 1`, is exactly `s`), and `clamped_position` addresses `x`
/// through [`IndexMap::Computed`] -- the same gather
/// [`gathered_expert_product`] already uses to read a data-dependent row.
/// Taps whose *unclamped* position is negative (real left-padding) are zeroed
/// post-gather via `Select`, mirroring how [`causal_mask`] masks attention
/// scores rather than ever reading an invalid position.
pub fn causal_conv1d(
    program: &mut Vec<Op>,
    x: NodeId,
    weight: NodeId,
    l_cache: u32,
) -> Result<NodeId, TensorError> {
    if l_cache == 0 {
        return Err(TensorError::InvalidConvConfig { l_cache });
    }

    let sequence_index = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Symbolic(0),
        },
    );
    let tap_index = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(l_cache),
        },
    );
    let window_offset = scalar_constant(program, -((l_cache - 1) as f32));

    let sequence_plus_tap = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(sequence_index, "s->sl"), (tap_index, "l->sl")],
    )?;
    let raw_position = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(sequence_plus_tap, "sl->sl"), (window_offset, "->sl")],
    )?;

    // `clamped_position` must be an `Op::Reduce`, not a plain `Elementwise`,
    // even though the fold itself is trivial (`max` over a synthetic 2-wide
    // axis holding `[raw_position, 0]`): `bind::BoundOpBuilder::push`'s
    // `Op::Elementwise` arm only forces materialization for nodes it finds in
    // its own `operands` list, and a `Computed` gather's `indices` reference
    // lives on a *different* node's operand entry -- a lone `Elementwise`
    // referenced only that way can sit `held` (fusion-deferred) past the
    // point a later gather needs its buffer, surfacing as
    // `TensorError::NotLowerable`'s "operand buffer missing at evaluation
    // time" (confirmed empirically: a first version of this function used
    // exactly that shape and hit precisely this). `Op::Reduce`'s own arm
    // always `push_ready`s immediately (`bind.rs`'s `push`, the
    // `Op::Reduce(reduce)` match arm), which is why every existing gather
    // index in this crate (`route` in [`gathered_expert_product`]) is already
    // a `Reduce`, never a bare `Elementwise` -- this mirrors that, rather
    // than being a new exception.
    let candidate_axis = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(2),
        },
    );
    let zero = scalar_constant(program, 0.0);
    // `zero_wide`, not the rank-0 `zero` above, is `is_raw_slot`'s second
    // operand: a rank-0 operand contributes no extent to any axis, so
    // `candidate_axis` alone (which only addresses `c`) would leave `s` and
    // `l` unconstrained on this node and `shape::infer` rejects that
    // (`TensorError::UnconstrainedDim`) -- every other broadcast pair in this
    // crate (e.g. `neg_infinity`/`is_future` in [`causal_mask`]'s callers)
    // always has a same-call sibling operand of the full iteration rank for
    // exactly this reason; `zero_wide`'s declared `[Symbolic(0), l_cache]`
    // shape is that sibling here, still comparing against literal `0.0`.
    let zero_wide = op::append(
        program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Symbolic(0), Extent::Static(l_cache)],
            value: 0.0,
        },
    );
    let is_raw_slot = elementwise(
        program,
        DType::Float32,
        ScalarOp::Equal,
        &[(candidate_axis, "c->slc"), (zero_wide, "sl->slc")],
    )?;
    let candidate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Select,
        &[
            (is_raw_slot, "slc->slc"),
            (raw_position, "sl->slc"),
            (zero, "->slc"),
        ],
    )?;
    let clamped_position = reduce(
        program,
        DType::Int32,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        candidate,
        "slc->slc",
        "sl->slc",
    )?;

    let negative_one = scalar_constant(program, -1.0);
    let is_valid = elementwise(
        program,
        DType::Float32,
        ScalarOp::Greater,
        &[(raw_position, "sl->sl"), (negative_one, "->sl")],
    )?;

    let gathered_map = IndexMap::Computed {
        indices: clamped_position,
        index_map: map::projection(3, &[0, 1]),
        base: IndexPattern {
            iter_rank: 3,
            axes: alloc::vec![
                AxisIndex::default(),
                AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(2)).collect(),
                    offset: 0,
                    len: None,
                },
            ],
        },
        gathered_dim: 0,
    };
    let windowed = op::append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: alloc::vec![(x, gathered_map)],
            name: None,
        },
    );

    // `weight`'s own declared shape is `[embedding, l_cache]` (`d` axis
    // first/outer, `l` axis last/fastest) -- `dl->sld`, not `ld->sld` --
    // matching `row_major_strides`'s (`bind.rs`) own last-axis-fastest
    // convention against the REAL on-disk tensor's physical layout: GGUF's
    // own `ne[0] = l_cache` is ggml's fastest axis (confirmed against
    // `llama.cpp`'s own `create_tensor(.., {n_shortconv_l_cache, n_embd},
    // ..)`), so `l_cache` -- not `embedding` -- is genuinely contiguous per
    // channel on disk. Every caller of this function must declare `weight`'s
    // `Op::Input` shape the same way.
    let tap_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(windowed, "sld->sld"), (weight, "dl->sld")],
    )?;
    let zero_tap = scalar_constant(program, 0.0);
    let masked_tap = elementwise(
        program,
        DType::Float32,
        ScalarOp::Select,
        &[
            (is_valid, "sl->sld"),
            (tap_product, "sld->sld"),
            (zero_tap, "->sld"),
        ],
    )?;

    reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        masked_tap,
        "sld->sld",
        "sd->sld",
    )
}

/// LFM2's gated short-convolution mixer, [`append_mistral_layer`]'s
/// attention-block counterpart for a `LayerKind::ShortConv` block: three
/// separate `embedding x embedding` projections (`b_proj`/`c_proj`/`x_proj`)
/// stand in for the real checkpoint's single fused `blk.N.shortconv.in_proj.weight`
/// (`[embedding, 3*embedding]`, one matmul producing three same-width
/// branches) -- **not** a shape this function chose for its own sake. A
/// single reduce over the fused weight followed by three static-offset
/// slices back out was the first version of this function, and it does not
/// type-check: `shape::unify_iteration_space` (`shape.rs:195-212`) resolves a
/// pure single-term axis's extent from the *sliced operand's own buffer
/// width* regardless of its offset (confirmed empirically --
/// `TensorError::ExtentMismatch` at the first later consumer that expects
/// `embedding`, not `3*embedding`), so an offset-only slice of a fused
/// `[s, 3*embedding]` tensor can never narrow to `[s, embedding]` inside this
/// algebra's current `Affine` grammar -- only a *strided* axis (coefficient
/// != 1, [`append_attention_mixer`]'s own `2*i` RoPE pattern) escapes that
/// branch, and a contiguous 2048-wide slice is not a stride. Splitting into
/// three independently-shaped `Input`s sidesteps the gap entirely, at the
/// cost of pushing the fused-to-three-tensor split to whichever binder loads
/// the real checkpoint (unimplemented this session, same as
/// [`append_mistral_layer`]'s own `wq`/`wk`/`wv` already being separate
/// `Input`s despite some checkpoints fusing QKV on disk).
///
/// `b_proj` gates the ungated `x_proj` branch, [`causal_conv1d`] convolves
/// the gated result causally over `l_cache` taps, `c_proj` gates the
/// convolved result, and `out_proj` projects back to `embedding` width --
/// LiquidAI's published LFM2 short-convolution block, `y = out_proj(C ⊙
/// conv(B ⊙ x))`, no activation function inside the block itself, unlike the
/// SwiGLU FFN every layer still runs after it. This branch assignment and
/// tap direction are read directly off HuggingFace's own reference
/// implementation (`transformers/models/lfm2_moe/modeling_lfm2_moe.py`,
/// `Lfm2MoeShortConv.slow_forward`, lines 434-465 of the checked-out
/// package): `BCx = in_proj(x).transpose(-1,-2)` then `B, C, x =
/// BCx.chunk(3, dim=-2)` -- `B` first, `C` second, ungated `x` third along
/// the packed axis, exactly `b_proj`/`c_proj`/`x_proj`'s declared order
/// below -- `Bx = B * x`, `conv_out = self.conv(Bx)` (an `nn.Conv1d` with
/// `padding = l_cache - 1`, left-only), `y = C * conv_out`,
/// `out_proj(y)`. [`causal_conv1d`]'s own tap convention (`l = l_cache - 1`
/// is the current position, `l = 0` the furthest lookback) matches
/// `nn.Conv1d`'s left-padded-causal convolution exactly: with `K - 1` zeros
/// prepended, `output[t] = sum_k weight[k] * padded_input[t + k]`, so
/// `weight[K-1]` always pairs with `input[t]` and `weight[0]` with
/// `input[t - (K-1)]`, the same pairing this function's own weight map
/// (`ld->sld`) uses.
#[allow(clippy::too_many_arguments)]
pub fn append_lfm2_conv_mixer(
    program: &mut Vec<Op>,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    norm_weight: NodeId,
    b_proj: NodeId,
    c_proj: NodeId,
    x_proj: NodeId,
    conv_weight: NodeId,
    out_proj: NodeId,
    l_cache: u32,
) -> Result<NodeId, TensorError> {
    let normed = rmsnorm(program, x, norm_weight, inv_dim, eps)?;

    let branch_b_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "sd->sdg"), (b_proj, "dg->sdg")],
    )?;
    let branch_b = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        branch_b_product,
        "sdg->sdg",
        "sg->sdg",
    )?;

    let branch_x_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "sd->sdg"), (x_proj, "dg->sdg")],
    )?;
    let branch_x = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        branch_x_product,
        "sdg->sdg",
        "sg->sdg",
    )?;

    let branch_c_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "sd->sdg"), (c_proj, "dg->sdg")],
    )?;
    let branch_c = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        branch_c_product,
        "sdg->sdg",
        "sg->sdg",
    )?;

    let gated_input = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(branch_b, "sg->sg"), (branch_x, "sg->sg")],
    )?;
    let convolved = causal_conv1d(program, gated_input, conv_weight, l_cache)?;
    let gated_output = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(convolved, "sg->sg"), (branch_c, "sg->sg")],
    )?;

    let out_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(gated_output, "sd->sdo"), (out_proj, "do->sdo")],
    )?;
    let mixer_out = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        out_product,
        "sdo->sdo",
        "so->sdo",
    )?;

    elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(mixer_out, "sd->sd"), (x, "sd->sd")],
    )
}

/// One token's worth of the gated-DeltaNet recurrence Qwen3.5's linear
/// attention (SSM) layers run -- llama.cpp's own reference,
/// `llm_build_delta_net_base::build_delta_net_autoregressive`
/// (`src/models/delta-net-base.cpp`, the `n_tokens == 1` path
/// `llm_build_delta_net_base::build_delta_net` dispatches to), transcribed
/// op-for-op onto this crate's existing `Elementwise`/`Reduce` vocabulary --
/// no new [`Op`] variant, per this crate's own reuse-first rule: a
/// state-carrying IIR recurrence over a caller-owned `[key_dim, value_dim,
/// head]` matrix is exactly what an `Input`/`Output` pair already expresses
/// for [`append_mistral_cached_layer`]'s own KV cache, so the persistent
/// state here is a caller-provided `state_in` node returned again as
/// `state_out`, not a new stateful primitive.
///
/// Per head `h`, key axis `i`, value axis `j` (`state[i,j,h]`, `q`/`k`
/// share `i`, `v` shares `j` with `state`'s second axis): `state = state *
/// exp(gate)` (`decay`), `v_pred[j] = sum_i state[i,j] * k[i]`
/// (`llama.cpp:305-306`, `sk = sum_rows(state * k)`), `delta[j] = beta *
/// (v[j] - v_pred[j])` (`:309-311`), `state[i,j] += k[i] * delta[j]`
/// (`:313-317`, the outer-product update), `out[j] = sum_i state[i,j] *
/// q_scaled[i]` (`:322-323`, read-out uses the UPDATED state) -- `q_scaled
/// = q / sqrt(key_dim)` is applied by the caller (`llama.cpp:295`,
/// `q = ggml_scale(ctx0, q, scale)`), matching every other pre-scaled `q`
/// this crate's own attention mixers already take.
///
/// `gate` and `beta` arrive already reduced to one scalar per head per
/// token (`llama.cpp`'s own `softplus(alpha + dt_bias) * ssm_a` and
/// `sigmoid(beta_proj)` respectively) -- this function only runs the
/// recurrence, never the projections that produce its inputs.
///
/// `head` is a format-interpolated run of letters, not a single character,
/// the same widening [`rmsnorm_per_head`] already makes: [`repeat_kv_heads`]'s
/// own doc proves this algebra cannot merge a `u,g` (kv-head, group) split
/// back into one physical head axis, so [`append_qwen35_ssm_mixer`] calls
/// this with `head = "ug"` and every map below (`i{head}`, `{head}`,
/// `ij{head}`) carries both letters through unchanged -- the recurrence
/// itself is per-head and never mixes heads, so nothing in its math depends
/// on the head space being one physical axis.
#[allow(clippy::too_many_arguments)]
pub fn append_qwen35_delta_net_step(
    program: &mut Vec<Op>,
    query: NodeId,
    key: NodeId,
    value: NodeId,
    gate: NodeId,
    beta: NodeId,
    state_in: NodeId,
    inv_sqrt_key_dim: NodeId,
    head: &str,
) -> Result<(NodeId, NodeId), TensorError> {
    let i_head = alloc::format!("i{head}->i{head}");
    let i_head_bcast = alloc::format!("->i{head}");
    let head_head = alloc::format!("{head}->{head}");
    let ij_head = alloc::format!("ij{head}->ij{head}");
    let head_to_ij_head = alloc::format!("{head}->ij{head}");
    let j_head = alloc::format!("j{head}->j{head}");
    let head_to_j_head = alloc::format!("{head}->j{head}");
    let i_head_to_ij_head = alloc::format!("i{head}->ij{head}");
    let j_head_to_ij_head = alloc::format!("j{head}->ij{head}");
    let ij_head_reduce_in = alloc::format!("ij{head}->ij{head}");
    let ij_head_reduce_out = alloc::format!("j{head}->ij{head}");

    let query_scaled = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (query, i_head.as_str()),
            (inv_sqrt_key_dim, i_head_bcast.as_str()),
        ],
    )?;
    let decay = elementwise(
        program,
        DType::Float32,
        ScalarOp::Exponential,
        &[(gate, head_head.as_str())],
    )?;
    let state_decayed = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (state_in, ij_head.as_str()),
            (decay, head_to_ij_head.as_str()),
        ],
    )?;
    // `state_decayed` feeds both the value prediction and the state write.
    // Keep the prediction's shared value, but give the state write its own
    // equivalent producer so chain fusion can inline this multiply into the
    // 2 MiB state update instead of materializing and rereading that buffer.
    let state_decayed_for_update = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (state_in, ij_head.as_str()),
            (decay, head_to_ij_head.as_str()),
        ],
    )?;

    let value_pred_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (state_decayed, ij_head.as_str()),
            (key, i_head_to_ij_head.as_str()),
        ],
    )?;
    let value_pred = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        value_pred_product,
        ij_head_reduce_in.as_str(),
        ij_head_reduce_out.as_str(),
    )?;

    let residual = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(value, j_head.as_str()), (value_pred, j_head.as_str())],
    )?;
    let delta = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(residual, j_head.as_str()), (beta, head_to_j_head.as_str())],
    )?;

    let update = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (key, i_head_to_ij_head.as_str()),
            (delta, j_head_to_ij_head.as_str()),
        ],
    )?;
    let state_out = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[
            (state_decayed_for_update, ij_head.as_str()),
            (update, ij_head.as_str()),
        ],
    )?;

    let out_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (state_out, ij_head.as_str()),
            (query_scaled, i_head_to_ij_head.as_str()),
        ],
    )?;
    let out = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        out_product,
        ij_head_reduce_in.as_str(),
        ij_head_reduce_out.as_str(),
    )?;

    Ok((out, state_out))
}

/// `log(1 + exp(x))`, Qwen3.5's own `alpha_softplus` gate input
/// (`llama.cpp:370`, `ggml_softplus`) -- not a [`ScalarOp`] primitive, so
/// composed from the two that are: [`ScalarOp::Exponential`] then
/// [`ScalarOp::Add`] against a `one` constant then [`ScalarOp::Logarithm`],
/// the same compose-not-mint move [`ExpertGatingFunc::Sigmoid`]'s own
/// `neg -> exp -> +1 -> reciprocal` chain already makes for a activation this
/// crate has no dedicated variant for.
pub fn softplus(
    program: &mut Vec<Op>,
    x: NodeId,
    one: NodeId,
    map: &str,
) -> Result<NodeId, TensorError> {
    let target = map.rsplit("->").next().unwrap_or(map);
    let one_map = alloc::format!("->{target}");
    let exp_x = elementwise(program, DType::Float32, ScalarOp::Exponential, &[(x, map)])?;
    let one_plus_exp = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(exp_x, map), (one, one_map.as_str())],
    )?;
    elementwise(
        program,
        DType::Float32,
        ScalarOp::Logarithm,
        &[(one_plus_exp, map)],
    )
}

/// [`rmsnorm`]'s L2-normalize variant: `x / sqrt(sum(x^2) + eps)`, no
/// mean-divide and no learnable `gamma` -- `ggml_l2_norm`
/// (`qwen35.cpp:428-429`, applied to `q_conv`/`k_conv` with no weight
/// tensor), unlike [`rmsnorm`]'s `mean_square = sum_squares / dim` and its
/// trailing `gamma` multiply. `map`/`sum_map` follow [`rmsnorm_per_head`]'s
/// own per-axis convention so the same function serves whichever axis (head
/// dim here, embedding elsewhere) is being normalized.
pub fn l2norm(
    program: &mut Vec<Op>,
    x: NodeId,
    eps: NodeId,
    map: &str,
    sum_map: &str,
) -> Result<NodeId, TensorError> {
    let reduced = sum_map.split("->").next().unwrap_or(sum_map);
    let reduced_map = alloc::format!("{reduced}->{reduced}");
    l2norm_with_eps_map(program, x, eps, map, sum_map, reduced_map.as_str())
}

pub(super) fn l2norm_with_eps_map(
    program: &mut Vec<Op>,
    x: NodeId,
    eps: NodeId,
    map: &str,
    sum_map: &str,
    eps_map: &str,
) -> Result<NodeId, TensorError> {
    let reduced = sum_map.split("->").next().unwrap_or(sum_map);
    let reduced_map = alloc::format!("{reduced}->{reduced}");
    let squared = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(x, map), (x, map)],
    )?;
    let sum_squares = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        squared,
        map,
        sum_map,
    )?;
    let sum_squares_eps = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(sum_squares, reduced_map.as_str()), (eps, eps_map)],
    )?;
    let norm = elementwise(
        program,
        DType::Float32,
        ScalarOp::SquareRoot,
        &[(sum_squares_eps, reduced_map.as_str())],
    )?;
    let inv_norm = elementwise(
        program,
        DType::Float32,
        ScalarOp::Reciprocal,
        &[(norm, reduced_map.as_str())],
    )?;
    elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(x, map), (inv_norm, sum_map)],
    )
}

/// `x * sigmoid(x)`, `ggml_silu`'s own contract (`qwen35.cpp:391-392`, run on
/// `conv_output_proper` before the q/k/v split) -- composed from
/// [`ScalarOp::Negate`]/[`ScalarOp::Exponential`]/[`ScalarOp::Add`]/[`ScalarOp::Reciprocal`], the same
/// `1/(1+e^-x)` chain [`ExpertGatingFunc::Sigmoid`] already builds, then one
/// more [`ScalarOp::Multiply`] against the un-gated input. No dedicated
/// `Sigmoid`/`Silu` [`ScalarOp`] exists, matching that chain's own precedent
/// for an activation this crate composes rather than mints.
pub fn silu(
    program: &mut Vec<Op>,
    x: NodeId,
    one: NodeId,
    map: &str,
) -> Result<NodeId, TensorError> {
    let target = map.rsplit("->").next().unwrap_or(map);
    let one_map = alloc::format!("->{target}");
    let neg_x = elementwise(program, DType::Float32, ScalarOp::Negate, &[(x, map)])?;
    let exp_neg_x = elementwise(
        program,
        DType::Float32,
        ScalarOp::Exponential,
        &[(neg_x, map)],
    )?;
    let one_plus_exp = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(exp_neg_x, map), (one, one_map.as_str())],
    )?;
    let gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Reciprocal,
        &[(one_plus_exp, map)],
    )?;
    elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(x, map), (gate, map)],
    )
}

/// Reads `width` contiguous channels of `x` (`[s, total_channels]`) starting
/// at `offset`, as a fresh `[s, width]` node -- the piece
/// [`append_qwen35_conv_branch`] needs three times (`q`/`k`/`v` out of one
/// fused conv output) that a plain offset [`AxisIndex`] slice cannot give it:
/// [`append_lfm2_conv_mixer`]'s own doc already proves a *nonzero*-offset
/// slice of a wider operand needs a same-width "donor" operand to escape
/// `shape::unify_iteration_space`'s pure-projection extent rule, and
/// `shape.rs`'s own
/// `an_offset_zero_slice_narrower_than_its_operand_is_still_ambiguous` test
/// proves the donor trick still fails at *zero* offset (`q`'s own case here,
/// `qkv_dim`'s first channel) -- there is no bit in `AxisIndex` that
/// disambiguates "the whole axis" from "a same-origin narrower window".
///
/// The declared-extent form now expresses this directly: one affine identity
/// read (`d+offset@width`) narrows the source axis without constructing a
/// one-hot mask, multiply, and reduction. The `@width` is the extent fact;
/// the offset remains the address fact, so shape inference bounds-checks the
/// window against `total_channels` while the backend can emit a plain view.
pub fn channel_slice(
    program: &mut Vec<Op>,
    x: NodeId,
    total_channels: u32,
    offset: u32,
    width: u32,
) -> Result<NodeId, TensorError> {
    let _ = total_channels;
    elementwise(
        program,
        DType::Float32,
        ScalarOp::Identity,
        &[(x, alloc::format!("s,w+{offset}@{width}->sw").as_str())],
    )
}

/// [`channel_slice`]'s own technique, generalized over a leading per-head
/// axis: `x` is `[s, heads, total_channels]` (`heads` contiguous blocks of
/// `total_channels`, e.g. a fused dense-attention `q`/`gate` chunk pair
/// repeated per head, `qwen3_next`'s own `q_proj(x).view(..., heads,
/// 2 * head_dim)` before its `torch.chunk(2, dim=-1)`), and this reads
/// `width` channels at `offset` within every head's own block
/// independently -- `channel_slice`'s single `(s, d)` mask can only carve
/// one contiguous window out of ONE flat channel axis, which is wrong here
/// because each head's window sits at a different flat offset (`head *
/// total_channels + offset`); the extra `Iota` over `heads` folds that
/// per-head stride into the same `target` the mask compares against, one
/// mask covering every head's window in a single select-then-reduce pass.
pub fn per_head_channel_slice(
    program: &mut Vec<Op>,
    x: NodeId,
    heads: u32,
    total_channels: u32,
    offset: u32,
    width: u32,
) -> Result<NodeId, TensorError> {
    let channel_index = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(heads * total_channels),
        },
    );
    let within_index = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(width),
        },
    );
    let head_index = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(heads),
        },
    );
    let period_const = scalar_constant(program, total_channels as f32);
    let offset_const = scalar_constant(program, offset as f32);
    let head_base = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(head_index, "h->h"), (period_const, "->h")],
    )?;
    let head_start = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(head_base, "h->h"), (offset_const, "->h")],
    )?;
    let target = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(head_start, "h->hw"), (within_index, "w->hw")],
    )?;
    let mask = elementwise(
        program,
        DType::Float32,
        ScalarOp::Equal,
        &[(channel_index, "d->dhw"), (target, "hw->dhw")],
    )?;
    let selected = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(x, "sd->sdhw"), (mask, "dhw->sdhw")],
    )?;
    reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        selected,
        "sdhw->sdhw",
        "shw->sdhw",
    )
}

/// [`channel_slice`]'s own select-then-reduce technique, generalized over an
/// ALREADY-split leading per-head axis: `x` is `[s, head, total_channels]`
/// (a head axis of its own, not [`per_head_channel_slice`]'s flat
/// `heads*total_channels` an activation like [`append_qwen35_dense_attention_layer`]'s
/// own `q`/`k` never has after `rmsnorm_per_head`), and this reads `width`
/// channels at the SAME `offset` uniformly across every head (unlike
/// [`per_head_channel_slice`]'s per-head-varying stride, there needed only
/// because the input axis was still flat). A plain affine projection
/// (`"s,h,i+64->shi"`) cannot express this narrowing on its own -- shape
/// inference unifies every operand touching iteration letter `i` to the
/// SAME extent, so a bare projection off `x`'s own `total_channels`-wide
/// axis pins `i` at `total_channels`, not `width`; only the mask-then-reduce
/// route can produce a genuinely narrower output.
pub fn per_head_channel_range(
    program: &mut Vec<Op>,
    x: NodeId,
    head: &str,
    total_channels: u32,
    offset: u32,
    width: u32,
) -> Result<NodeId, TensorError> {
    let channel_index = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(total_channels),
        },
    );
    let within_index = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(width),
        },
    );
    let offset_const = scalar_constant(program, offset as f32);
    let target = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(within_index, "w->w"), (offset_const, "->w")],
    )?;
    let mask = elementwise(
        program,
        DType::Float32,
        ScalarOp::Equal,
        &[(channel_index, "d->dw"), (target, "w->dw")],
    )?;
    let x_map = alloc::format!("s{head}d->s{head}dw");
    let mask_map = alloc::format!("dw->s{head}dw");
    let selected = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(x, x_map.as_str()), (mask, mask_map.as_str())],
    )?;
    let in_map = alloc::format!("s{head}dw->s{head}dw");
    let out_map = alloc::format!("s{head}w->s{head}dw");
    reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        selected,
        in_map.as_str(),
        out_map.as_str(),
    )
}

/// Qwen3.5's `q|k|v` conv branch -- llama.cpp's own
/// `build_layer_attn_linear` (`qwen35.cpp:385-431`): `causal_conv1d` over
/// the fused `qkv_mixed` (`conv_input`, `:385`), [`silu`] (`:391-392`), a
/// three-way [`channel_slice`] split at `qkv_dim = 2*key_dim + value_dim`
/// (`q` at offset `0`, `k` at `key_dim`, `v` at `2*key_dim` --
/// `q_conv`/`k_conv`/`v_conv`'s own `ggml_view_4d` offsets, `:399-419`),
/// then [`l2norm`] on `q`/`k` only (`:428-429`, `v` is never normalized).
/// The GQA head repeat (`:437-440`) is a separate, independently-testable
/// step -- see [`repeat_kv_heads`].
// unwired: this one specifically, not the whole mixer -- `causal_conv1d`
// windows a whole in-graph sequence with zero-boundary padding, which fits
// a prefill call but not a decode step against a persisted history cache,
// so `append_qwen35_ssm_mixer` reimplements this function's own
// silu/channel_slice/l2norm body against the additive cached-conv split its
// own doc describes, rather than calling this. A prefill-only qwen35
// program (mirroring `lfm2_forward_program_with_experts`'s own prefill-only
// scope) is this function's real caller, not built this session.
#[allow(dead_code, clippy::too_many_arguments)]
pub fn append_qwen35_conv_branch(
    program: &mut Vec<Op>,
    qkv_mixed: NodeId,
    conv_weight: NodeId,
    eps: NodeId,
    one: NodeId,
    key_dim: u32,
    value_dim: u32,
    l_cache: u32,
) -> Result<(NodeId, NodeId, NodeId), TensorError> {
    let (q_raw, k_raw, v_conv) = append_qwen35_conv_raw(
        program,
        qkv_mixed,
        conv_weight,
        one,
        key_dim,
        value_dim,
        l_cache,
    )?;

    let q_conv = l2norm(program, q_raw, eps, "sw->sw", "s->sw")?;
    let k_conv = l2norm(program, k_raw, eps, "sw->sw", "s->sw")?;

    Ok((q_conv, k_conv, v_conv))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn append_qwen35_conv_raw(
    program: &mut Vec<Op>,
    qkv_mixed: NodeId,
    conv_weight: NodeId,
    one: NodeId,
    key_dim: u32,
    value_dim: u32,
    l_cache: u32,
) -> Result<(NodeId, NodeId, NodeId), TensorError> {
    let qkv_dim = 2 * key_dim + value_dim;
    let convolved = causal_conv1d(program, qkv_mixed, conv_weight, l_cache)?;
    let activated = silu(program, convolved, one, "sd->sd")?;

    let q_raw = channel_slice(program, activated, qkv_dim, 0, key_dim)?;
    let k_raw = channel_slice(program, activated, qkv_dim, key_dim, key_dim)?;
    let v_conv = channel_slice(program, activated, qkv_dim, 2 * key_dim, value_dim)?;

    Ok((q_raw, k_raw, v_conv))
}

/// The GQA head repeat `q_conv`/`k_conv` need before
/// [`append_qwen35_delta_net_step`] (`qwen35.cpp:437-440`,
/// `ggml_repeat_4d(.., num_v_heads, ..)`): `num_k_heads` (16) real kv heads
/// broadcast to `num_v_heads` (48) query/value heads, 3-wide groups.
///
/// [`append_attention_mixer`]'s own `group_map`/`group_ones` pair already
/// proves the technique this reuses: reading an operand's real axis while a
/// *new* iteration letter is simply absent from that operand's own map
/// broadcasts across it for free (`rotated_k`'s `"tui->stugi"` there never
/// mentions `g`), and multiplying against an all-ones donor of the new
/// letters' shape (`group_ones`, `"ug->sugi"`) is what makes
/// `shape::unify_iteration_space` resolve `g`'s extent at all -- `x` alone
/// (real axis `u`, no `g` term) leaves `g` unconstrained.
///
/// This never merges `u`/`g` back into one physical `h = group*u+g` axis:
/// `shape::project_output_shape` rejects any `Reduce` `out_map` axis that
/// is not a pure single-term projection ("reduce output maps must be pure
/// projections in v1"), and a plain [`Op::Elementwise`]'s output shape *is*
/// its iteration space, so two loop letters cannot collapse into one output
/// letter in this algebra's current grammar -- the same reason
/// [`append_attention_mixer`] itself never merges them either, keeping every
/// downstream op split as `u,g` through to its own final output. A
/// genuinely single-axis repeat would need a scatter with a data-computed
/// `h = group*u+g` destination (the same [`IndexMap::Computed`] shape
/// [`causal_conv1d`]'s own `clamped_position` gather already uses for a
/// data-computed *source*); not built this session.
pub fn repeat_kv_heads(
    program: &mut Vec<Op>,
    x: NodeId,
    kv_heads: u32,
    group: u32,
) -> Result<NodeId, TensorError> {
    let group_ones = op::append(
        program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
            value: 1.0,
        },
    );
    elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(x, "sud->sugd"), (group_ones, "ug->sugd")],
    )
}

/// `1/(1+e^-x)` -- the exact `negate -> exp -> +1 -> reciprocal` chain
/// [`ExpertGatingFunc::Sigmoid`] already composes inline, factored out once
/// [`append_qwen35_ssm_mixer`] needs it twice (`beta`, the attention gate),
/// the same "worth naming at two callers" threshold [`silu`]/[`softplus`]
/// already crossed for their own chains.
pub fn sigmoid(
    program: &mut Vec<Op>,
    x: NodeId,
    one: NodeId,
    map: &str,
) -> Result<NodeId, TensorError> {
    let target = map.rsplit("->").next().unwrap_or(map);
    let one_map = alloc::format!("->{target}");
    let neg_x = elementwise(program, DType::Float32, ScalarOp::Negate, &[(x, map)])?;
    let exp_neg_x = elementwise(
        program,
        DType::Float32,
        ScalarOp::Exponential,
        &[(neg_x, map)],
    )?;
    let one_plus_exp = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(exp_neg_x, map), (one, one_map.as_str())],
    )?;
    elementwise(
        program,
        DType::Float32,
        ScalarOp::Reciprocal,
        &[(one_plus_exp, map)],
    )
}

/// Qwen3.5's gated-DeltaNet mixer, one decode step (`n_tokens == 1`, the same
/// scope [`append_qwen35_delta_net_step`]'s own doc already commits to) --
/// llama.cpp's own `build_layer_attn_linear` (`qwen35.cpp:335-466`) run
/// op-for-op: `build_qkvz` (`:353-356`, `qkv_mixed`/`z`), `beta`/`gate`
/// (`:358-376`, `sigmoid(ssm_beta @ x)` / `ssm_a * softplus(ssm_alpha @ x +
/// ssm_dt)`), the causal conv + [`silu`] + channel split + [`l2norm`]
/// ([`append_qwen35_conv_branch`]'s own body, `:391-429`, reproduced here
/// against a persisted history window instead of [`causal_conv1d`]'s own
/// zero-boundary window -- see the conv step below), the GQA repeat
/// ([`repeat_kv_heads`], `:437-440`), the recurrence itself
/// ([`append_qwen35_delta_net_step`], `build_delta_net_autoregressive`,
/// `delta-net-base.cpp:289-370`), gated RMSNorm (`build_norm_gated`,
/// `qwen35.cpp:243-250`: `rmsnorm(out) * silu(z)`), and the output
/// projection + residual (`:456-464`, folded into the block-level
/// `ggml_add(cur, inpSA)` at `:180`) -- the pre-mixer `rmsnorm` and the
/// post-mixer residual add both happen INSIDE this function, the same
/// choice [`append_lfm2_conv_mixer`] already makes for its own block.
///
/// The `u,g` seam: [`repeat_kv_heads`]'s own doc proves this algebra can
/// never merge a `u` (kv-head)/`g` (group) split back into one physical head
/// axis -- `shape::project_output_shape` rejects any `Reduce` `out_map`
/// axis that is not a pure single-term projection, and a plain
/// `Elementwise`'s output shape IS its iteration space, so two loop letters
/// cannot collapse into one output letter. [`append_qwen35_delta_net_step`]'s
/// own maps are all per-head (nothing in the recurrence mixes heads), so
/// widening its `head` parameter from one letter to `"ug"` costs nothing but
/// string interpolation -- verified by reading its maps before relying on
/// it, not assumed. `value`/`gate`/`beta` all decompose from their real flat
/// `num_v_heads` axis via the identical `group*u+g` computed read
/// [`append_attention_mixer`]'s own `group_map` already proves; `value`
/// folds the within-head axis `j` into the same expression
/// (`(group*head_v_dim)*u + head_v_dim*g + j`, still one `Affine` axis
/// expression -- `parse_axis_expr` sums an arbitrary run of `+`-joined
/// terms, not just two, confirmed by reading it before relying on it).
///
/// State threading mirrors [`append_mistral_cached_layer`]: both caches
/// (`state_in`/`state_out`, [`append_qwen35_delta_net_step`]'s own contract,
/// and `conv_history_in`, the `l_cache - 1` previous raw `qkv_mixed` rows)
/// are caller-persisted [`Op::Input`]s/return values, never concatenated
/// in-graph -- [`causal_conv1d`]'s own doc already establishes this op set
/// has no concat primitive. Instead of windowing (which would need a real
/// concat), the cached conv is a plain additive split: `conv_out = sum_w
/// weight[.., w] * history[w] + weight[.., l_cache - 1] * qkv_mixed_new`,
/// the same disjoint-source blend [`append_mistral_cached_layer`]'s own
/// `score_cached` + `score_new` split already uses for attention, minus the
/// online-softmax combine step (a linear conv sum splits for free; attention
/// only splits after `Maximum`/`Add` recombine it). The caller is
/// responsible for trimming/appending `qkv_mixed` (this function's second
/// return) into its own persisted history buffer, exactly as
/// [`append_mistral_cached_layer`]'s own [`CachedLayerRoots`] callers manage
/// their KV cache outside the graph -- shift-and-trim lives on the host, not
/// in the graph.
///
/// Which nonlinearity gates [`append_qwen35_ssm_mixer`]'s output norm --
/// `rmsnorm(delta_out) * activation(z)` (reference: PR 27742 line 2896-2899,
/// `build_norm_gated`, whose own comment names this "the one numerical
/// difference from Qwen3.5's GDN: sigmoid output gate, not silu"). Qwen3.5's
/// own checkpoint keeps [`GdnOutputGate::Silu`]; qwen4exp's GDN layers pass
/// [`GdnOutputGate::Sigmoid`] -- a layer-kind flag on the shared builder
/// rather than a duplicated function, since every other line of the mixer
/// (fused QKVZ, causal conv, delta-rule recurrence) is identical between the
/// two checkpoints.
/// [`append_qwen35_ssm_mixer`]'s output-gate selector -- see
/// [`qwen35_forward_program`] for the worked example passing
/// [`GdnOutputGate::Silu`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GdnOutputGate {
    /// `qwen35_forward_program`'s own GDN layers (`qwen35.cpp:243-250`).
    Silu,
    /// qwen4exp's GDN layers (reference: PR 27742 line 2895-2897) -- no
    /// production call site in this crate (that forward-program assembly is
    /// model-specific and lives in its own consuming crate); exercised
    /// today by `qwen35_ssm_mixer_sigmoid_gate_moves_the_output_away_from_silu`.
    Sigmoid,
}

/// [`append_qwen35_ssm_mixer_with_taps`]'s own return shape: every
/// intermediate a caller needs to bisect the mixer's tail against an
/// independent reference, in the order the builder computes them
/// (`spec.rs:6608-6928`). `qkv_mixed` is the fused `wqkv` projection
/// (Q/K/V still concatenated, pre-conv). `query`/`key`/`value`/`gate`/
/// `beta` are the sequence-axis-free inputs to
/// [`append_qwen35_delta_net_step`]; exposing those existing roots gives an
/// executor a typed cut at which it can batch the projections and then drive
/// the matrix-state recurrence position by position. `state_out` is the
/// delta-net recurrence's carried state; `delta_out` is the delta-net read-out
/// (`jug` layout, pre-norm); `z` is the raw output-gate projection
/// BEFORE [`GdnOutputGate`]'s silu/sigmoid split, so a caller can apply
/// either nonlinearity independently of which one this program's own
/// `output_gate` argument baked in; `gated_rmsnorm_out` is the per-head
/// RMSNorm output after its own `ssm_norm_weight` scale (before the `z`
/// gate multiplies in); `gated_value` is that result after the `z` gate
/// multiplies in (still per-head, pre-projection); `ssm_out_result` is
/// the `ssm_out` projection's reduce, before the residual add that
/// produces `mixer_out`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SsmMixerTaps {
    /// Every unrolled position's own recurrence `state_out`, in position
    /// order (`per_position_state_out[0]` is position 0's carried state) --
    /// the M=1 branch below fills this with its own single `state_out` so
    /// callers never need to special-case decode vs. prefill width.
    pub per_position_state_out: Vec<NodeId>,
    pub qkv_mixed: NodeId,
    pub query_sequence: NodeId,
    pub key_sequence: NodeId,
    pub value_sequence: NodeId,
    pub gate_sequence: NodeId,
    pub beta_sequence: NodeId,
    pub z_sequence: NodeId,
    pub query: NodeId,
    pub key: NodeId,
    pub value: NodeId,
    pub gate: NodeId,
    pub beta: NodeId,
    pub state_in: NodeId,
    pub z_head: NodeId,
    pub state_out: NodeId,
    pub delta_out: NodeId,
    pub z: NodeId,
    pub gated_rmsnorm_out: NodeId,
    pub gated_value: NodeId,
    pub ssm_out_result: NodeId,
}

/// Inputs to the sequence-preserving tail of Qwen3.5's GDN mixer.
///
/// The recurrence is deliberately outside this value: a sans-IO executor
/// supplies its caller-owned `[s,j,u,g]` scan output as `delta_out`, while
/// this algebra applies the checkpoint's per-row RMSNorm, output gate,
/// projection, and residual without erasing the sequence axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35GdnSequenceTail {
    pub x: NodeId,
    pub delta_out: NodeId,
    pub z: NodeId,
    pub head_eps: NodeId,
    pub inv_head_v_dim: NodeId,
    pub norm_weight: NodeId,
    pub out_weight: NodeId,
    pub head_v_dim: u32,
    pub kv_heads: u32,
    pub group: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35GdnSequenceTailTaps {
    pub gated_value: NodeId,
    pub projected: NodeId,
    pub output: NodeId,
}

/// Appends the row-preserving algebra after a caller-driven GDN prefill scan.
///
/// This composes the existing elementwise and reduction primitives; it adds
/// no stateful operation. `delta_out` is `[s,j,u,g]`, `z` is `[s,u,g,j]`,
/// and the result is the post-mixer residual `[s,d]` consumed by the MoE
/// router.
pub fn append_qwen35_gdn_sequence_tail(
    program: &mut Vec<Op>,
    tail: Qwen35GdnSequenceTail,
) -> Result<NodeId, TensorError> {
    Ok(append_qwen35_gdn_sequence_tail_with_taps(program, tail)?.output)
}

pub fn append_qwen35_gdn_sequence_tail_with_taps(
    program: &mut Vec<Op>,
    tail: Qwen35GdnSequenceTail,
) -> Result<Qwen35GdnSequenceTailTaps, TensorError> {
    let squared = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (tail.delta_out, "sjug->sjug"),
            (tail.delta_out, "sjug->sjug"),
        ],
    )?;
    let sum_squares = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        squared,
        "sjug->sjug",
        "sug->sjug",
    )?;
    let mean_square = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(sum_squares, "sug->sug"), (tail.inv_head_v_dim, "->sug")],
    )?;
    let mean_square_eps = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(mean_square, "sug->sug"), (tail.head_eps, "ug->sug")],
    )?;
    let rms = elementwise(
        program,
        DType::Float32,
        ScalarOp::SquareRoot,
        &[(mean_square_eps, "sug->sug")],
    )?;
    let inv_rms = elementwise(
        program,
        DType::Float32,
        ScalarOp::Reciprocal,
        &[(rms, "sug->sug")],
    )?;
    let normed = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(tail.delta_out, "sjug->sjug"), (inv_rms, "sug->sjug")],
    )?;
    let normed_gamma = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "sjug->sjug"), (tail.norm_weight, "j->sjug")],
    )?;
    let gated = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed_gamma, "sjug->sguj"), (tail.z, "sugj->sguj")],
    )?;
    let out_weight_map = alloc::format!(
        "{}*j+{}*u+g,d->ugjd",
        tail.kv_heads * tail.group,
        tail.group
    );
    let value_head_ones = op::append(
        program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![
                Extent::Static(tail.kv_heads),
                Extent::Static(tail.group),
                Extent::Static(tail.head_v_dim),
            ],
            value: 1.0,
        },
    );
    let out_weight_split = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (tail.out_weight, out_weight_map.as_str()),
            (value_head_ones, "ugj->ugjd"),
        ],
    )?;
    let product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(gated, "sguj->sgujd"), (out_weight_split, "ugjd->sgujd")],
    )?;
    let projected = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "sgujd->sgujd",
        "sd->sgujd",
    )?;
    let output = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(tail.x, "sd->sd"), (projected, "sd->sd")],
    )?;
    Ok(Qwen35GdnSequenceTailTaps {
        gated_value: gated,
        projected,
        output,
    })
}

/// The GDN (gated delta-net) mixer [`qwen35_forward_program`] calls once
/// per non-attention layer -- see it there for the worked example of
/// wiring this builder's inputs. Returns `(x_next, qkv_mixed, state_out)`.
/// Thin wrapper over [`append_qwen35_ssm_mixer_with_taps`] for callers that
/// only need the three roots this signature already returned before taps
/// existed -- byte-identical program, since this only reshapes the return
/// value the shared builder already computed.
#[allow(clippy::too_many_arguments)]
pub fn append_qwen35_ssm_mixer(
    program: &mut Vec<Op>,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    head_eps: NodeId,
    one: NodeId,
    inv_sqrt_key_dim: NodeId,
    inv_head_v_dim: NodeId,
    attn_norm_weight: Option<NodeId>,
    wqkv: NodeId,
    wqkv_gate: NodeId,
    conv_weight: NodeId,
    conv_history_in: NodeId,
    ssm_beta: NodeId,
    ssm_alpha: NodeId,
    ssm_dt_bias: NodeId,
    ssm_a: NodeId,
    ssm_norm_weight: NodeId,
    ssm_out: NodeId,
    state_in: NodeId,
    key_dim: u32,
    value_dim: u32,
    kv_heads: u32,
    group: u32,
    l_cache: u32,
    output_gate: GdnOutputGate,
) -> Result<(NodeId, NodeId, NodeId), TensorError> {
    let (mixer_out, taps) = append_qwen35_ssm_mixer_with_taps(
        program,
        x,
        inv_dim,
        eps,
        head_eps,
        one,
        inv_sqrt_key_dim,
        inv_head_v_dim,
        attn_norm_weight,
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
    )?;
    Ok((mixer_out, taps.qkv_mixed, taps.state_out))
}

/// [`append_qwen35_ssm_mixer`]'s full implementation, returning every
/// [`SsmMixerTaps`] intermediate alongside `mixer_out` for a caller that
/// needs to bisect the tail (per-head gated RMSNorm, output gate, `ssm_out`
/// projection) against an independent reference.
#[allow(clippy::too_many_arguments)]
pub fn append_qwen35_ssm_mixer_with_taps(
    program: &mut Vec<Op>,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    head_eps: NodeId,
    one: NodeId,
    inv_sqrt_key_dim: NodeId,
    inv_head_v_dim: NodeId,
    attn_norm_weight: Option<NodeId>,
    wqkv: NodeId,
    wqkv_gate: NodeId,
    conv_weight: NodeId,
    conv_history_in: NodeId,
    ssm_beta: NodeId,
    ssm_alpha: NodeId,
    ssm_dt_bias: NodeId,
    ssm_a: NodeId,
    ssm_norm_weight: NodeId,
    ssm_out: NodeId,
    state_in: NodeId,
    key_dim: u32,
    value_dim: u32,
    kv_heads: u32,
    group: u32,
    l_cache: u32,
    output_gate: GdnOutputGate,
) -> Result<(NodeId, SsmMixerTaps), TensorError> {
    append_qwen35_ssm_mixer_with_taps_and_layout(
        program,
        x,
        inv_dim,
        eps,
        head_eps,
        one,
        inv_sqrt_key_dim,
        inv_head_v_dim,
        attn_norm_weight,
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
        false,
    )
}

/// Reads one literal position `p` out of a `[s, ..rest]` sequence-preserving
/// tensor and squeezes the resulting size-1 `s` back off, the same
/// slice-then-sum-of-one technique [`channel_slice`] already uses on a
/// channel axis, applied here to the leading position axis instead: a
/// caller-known-at-build-time `p` (this exists only inside the M>1 branch
/// [`append_qwen35_ssm_mixer_with_taps_and_layout`] unrolls, never on the
/// symbolic architecture-level graph), so it is a plain affine offset, not a
/// gather.
pub(super) fn qwen35_gdn_sequence_position(
    program: &mut Vec<Op>,
    node: NodeId,
    rest_letters: &str,
    position: u32,
) -> Result<NodeId, TensorError> {
    let rest_terms = rest_letters
        .chars()
        .map(|letter| letter.to_string())
        .collect::<alloc::vec::Vec<_>>()
        .join(",");
    let iteration = alloc::format!("s{rest_letters}");
    let sliced = elementwise(
        program,
        DType::Float32,
        ScalarOp::Identity,
        &[(
            node,
            alloc::format!("s+{position}@1,{rest_terms}->{iteration}").as_str(),
        )],
    )?;
    reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        sliced,
        alloc::format!("{iteration}->{iteration}").as_str(),
        alloc::format!("{rest_letters}->{iteration}").as_str(),
    )
}

/// One position's outputs from the M>1 branch
/// [`append_qwen35_ssm_mixer_with_taps_and_layout`] unrolls -- grouped so the
/// state-threading loop and the stacked-`delta_out` accumulation share one
/// call per position instead of two.
pub(super) struct Qwen35GdnRecurrenceStep {
    pub(super) query: NodeId,
    pub(super) key: NodeId,
    pub(super) value: NodeId,
    pub(super) beta: NodeId,
    pub(super) gate: NodeId,
    pub(super) z_head: NodeId,
    pub(super) delta_out: NodeId,
    pub(super) state_out: NodeId,
}

/// Slices position `p` out of every sequence-preserving tap and runs one
/// [`append_qwen35_delta_net_step`] against the caller-threaded `state_in`,
/// the per-position body [`append_qwen35_ssm_mixer_with_taps_and_layout`]'s
/// M>1 branch calls once per prompt position, threading `state_out` into the
/// next call's `state_in` the same way the decode path threads it call to
/// call.
#[allow(clippy::too_many_arguments)]
pub(super) fn qwen35_gdn_recurrence_step(
    program: &mut Vec<Op>,
    query_sequence: NodeId,
    key_sequence: NodeId,
    value_sequence: NodeId,
    beta_split: NodeId,
    gate_split: NodeId,
    z_split: NodeId,
    state_in: NodeId,
    inv_sqrt_key_dim: NodeId,
    position: u32,
) -> Result<Qwen35GdnRecurrenceStep, TensorError> {
    let query = qwen35_gdn_sequence_position(program, query_sequence, "dug", position)?;
    let key = qwen35_gdn_sequence_position(program, key_sequence, "dug", position)?;
    let value = qwen35_gdn_sequence_position(program, value_sequence, "jug", position)?;
    let beta = qwen35_gdn_sequence_position(program, beta_split, "ug", position)?;
    let gate = qwen35_gdn_sequence_position(program, gate_split, "ug", position)?;
    let z_head = qwen35_gdn_sequence_position(program, z_split, "ugj", position)?;
    let (delta_out, state_out) = append_qwen35_delta_net_step(
        program,
        query,
        key,
        value,
        gate,
        beta,
        state_in,
        inv_sqrt_key_dim,
        "ug",
    )?;
    Ok(Qwen35GdnRecurrenceStep {
        query,
        key,
        value,
        beta,
        gate,
        z_head,
        delta_out,
        state_out,
    })
}

/// Writes one position's `[j,u,g]` step output into its own row of the
/// stacked `[s,j,u,g]` sequence, everywhere else zero -- [`stack_selected_routes`]'s
/// own Iota-`Equal`-mask-then-`Add` technique, applied to this function's own
/// `s`/`jug` axes instead of `stack_selected_routes`'s `s`/`k`.
pub(super) fn qwen35_gdn_place_position(
    program: &mut Vec<Op>,
    delta_out_at_position: NodeId,
    position_axis: NodeId,
    position: u32,
) -> Result<NodeId, TensorError> {
    let position_value = op::append(
        program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec::Vec::new(),
            value: position as f32,
        },
    );
    let position_mask = elementwise(
        program,
        DType::Float32,
        ScalarOp::Equal,
        &[(position_axis, "s->s"), (position_value, "->s")],
    )?;
    elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (delta_out_at_position, "jug->sjug"),
            (position_mask, "s->sjug"),
        ],
    )
}

/// Builds the Qwen3.5 SSM mixer while selecting the checkpoint's V-head order.
#[allow(clippy::too_many_arguments)]
pub fn append_qwen35_ssm_mixer_with_taps_and_layout(
    program: &mut Vec<Op>,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    head_eps: NodeId,
    one: NodeId,
    inv_sqrt_key_dim: NodeId,
    inv_head_v_dim: NodeId,
    attn_norm_weight: Option<NodeId>,
    wqkv: NodeId,
    wqkv_gate: NodeId,
    conv_weight: NodeId,
    conv_history_in: NodeId,
    ssm_beta: NodeId,
    ssm_alpha: NodeId,
    ssm_dt_bias: NodeId,
    ssm_a: NodeId,
    ssm_norm_weight: NodeId,
    ssm_out: NodeId,
    state_in: NodeId,
    key_dim: u32,
    value_dim: u32,
    kv_heads: u32,
    group: u32,
    l_cache: u32,
    output_gate: GdnOutputGate,
    v_head_reordered: bool,
) -> Result<(NodeId, SsmMixerTaps), TensorError> {
    // `x`'s leading axis is `s` (sequence position) -- when it is a
    // statically-known extent (a synthetic caller, or a per-request bound
    // graph once the prompt length is resolved; the architecture-level
    // spec itself leaves `s` as `Extent::Symbolic`, see
    // `proxima-model-interop::bind_symbols`), a width of `0` can never
    // feed the recurrence below, so it still rejects here rather than
    // let the loop underflow.
    let prefill_width = match &program[x.0 as usize] {
        Op::Input { shape, .. } | Op::Constant { shape, .. } => match shape.first() {
            Some(Extent::Static(width)) => Some(*width),
            _ => None,
        },
        _ => None,
    };
    if prefill_width == Some(0) {
        return Err(TensorError::SingleTokenStepOnly {
            op: "qwen35_ssm_mixer",
            s: 0,
        });
    }

    let head_k_dim = key_dim / kv_heads;
    let num_v_heads = kv_heads * group;
    let head_v_dim = value_dim / num_v_heads;

    let normed = match attn_norm_weight {
        Some(weight) => rmsnorm(program, x, weight, inv_dim, eps)?,
        None => x,
    };

    let qkv_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->siq"), (wqkv, "iq->siq")],
    )?;
    let qkv_mixed = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        qkv_product,
        "siq->siq",
        "sq->siq",
    )?;

    let z_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->siz"), (wqkv_gate, "iz->siz")],
    )?;
    let z = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        z_product,
        "siz->siz",
        "sz->siz",
    )?;
    let z_gated = match output_gate {
        GdnOutputGate::Silu => silu(program, z, one, "sz->sz")?,
        GdnOutputGate::Sigmoid => sigmoid(program, z, one, "sz->sz")?,
    };

    let beta_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->sin"), (ssm_beta, "in->sin")],
    )?;
    let beta_flat = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        beta_product,
        "sin->sin",
        "sn->sin",
    )?;
    let beta_sigmoid = sigmoid(program, beta_flat, one, "sn->sn")?;

    let alpha_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->sin"), (ssm_alpha, "in->sin")],
    )?;
    let alpha_flat = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        alpha_product,
        "sin->sin",
        "sn->sin",
    )?;
    let alpha_biased = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(alpha_flat, "sn->sn"), (ssm_dt_bias, "n->sn")],
    )?;
    let alpha_softplus = softplus(program, alpha_biased, one, "sn->sn")?;
    let gate_flat = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(alpha_softplus, "sn->sn"), (ssm_a, "n->sn")],
    )?;

    // cached causal conv: `weight`'s declared `[q, l_cache]` layout matches
    // `causal_conv1d`'s own (`l_cache` fastest/contiguous per channel) --
    // history (real axis `w`, no `s`) and this call's own new token (real
    // axis `s`) blend additively, no concat.
    let weight_history = channel_slice(program, conv_weight, l_cache, 0, l_cache - 1)?;
    let weight_new_wide = channel_slice(program, conv_weight, l_cache, l_cache - 1, 1)?;
    let weight_new = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        weight_new_wide,
        "qw->qw",
        "q->qw",
    )?;

    let history_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(conv_history_in, "wq->wq"), (weight_history, "qw->wq")],
    )?;
    let history_term = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        history_product,
        "wq->wq",
        "q->wq",
    )?;

    let new_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(qkv_mixed, "sq->sq"), (weight_new, "q->sq")],
    )?;
    let conv_raw = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(new_product, "sq->sq"), (history_term, "q->sq")],
    )?;

    let activated = silu(program, conv_raw, one, "sq->sq")?;
    let qkv_dim = 2 * key_dim + value_dim;
    let q_raw = channel_slice(program, activated, qkv_dim, 0, key_dim)?;
    let k_raw = channel_slice(program, activated, qkv_dim, key_dim, key_dim)?;
    let v_conv = channel_slice(program, activated, qkv_dim, 2 * key_dim, value_dim)?;

    // A read-side multi-term decomposition (`{coeff}*u+i`) constrains the
    // COMBINED axis, never `u`/`i` individually -- `shape::infer` cannot
    // solve one affine equation for two unknown extents, exactly the reason
    // [`repeat_kv_heads`]'s own `group_ones`/`ug->sugd` donor exists. Each
    // decomposition below pairs with the identical all-ones-donor technique,
    // scoped to whichever letters that decomposition introduces.
    let key_head_ones = op::append(
        program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(head_k_dim)],
            value: 1.0,
        },
    );
    let q_split_map = alloc::format!("s,{head_k_dim}*u+i->sui");
    let q_split_raw = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(q_raw, q_split_map.as_str()), (key_head_ones, "ui->sui")],
    )?;
    let k_split_raw = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(k_raw, q_split_map.as_str()), (key_head_ones, "ui->sui")],
    )?;
    let q_split = l2norm_with_eps_map(program, q_split_raw, eps, "sui->sui", "su->sui", "s->su")?;
    let k_split = l2norm_with_eps_map(program, k_split_raw, eps, "sui->sui", "su->sui", "s->su")?;

    let q_repeated = repeat_kv_heads(program, q_split, kv_heads, group)?;
    let k_repeated = repeat_kv_heads(program, k_split, kv_heads, group)?;

    let value_head_ones = op::append(
        program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![
                Extent::Static(kv_heads),
                Extent::Static(group),
                Extent::Static(head_v_dim)
            ],
            value: 1.0,
        },
    );
    let v_split_map = if v_head_reordered {
        alloc::format!("s,{}*g+{}*u+j->sugj", kv_heads * head_v_dim, head_v_dim)
    } else {
        alloc::format!("s,{}*u+{}*g+j->sugj", group * head_v_dim, head_v_dim)
    };
    let v_split = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (v_conv, v_split_map.as_str()),
            (value_head_ones, "ugj->sugj"),
        ],
    )?;

    let group_ones = op::append(
        program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
            value: 1.0,
        },
    );
    let head_split_map = if v_head_reordered {
        alloc::format!("s,{kv_heads}*g+u->sug")
    } else {
        alloc::format!("s,{group}*u+g->sug")
    };
    let beta_split = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (beta_sigmoid, head_split_map.as_str()),
            (group_ones, "ug->sug"),
        ],
    )?;
    let gate_split = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (gate_flat, head_split_map.as_str()),
            (group_ones, "ug->sug"),
        ],
    )?;
    let z_split = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (z_gated, v_split_map.as_str()),
            (value_head_ones, "ugj->sugj"),
        ],
    )?;

    // The persisted-history branch below is the decode path. Its history
    // term is intentionally position-invariant and therefore cannot supply
    // a multi-position prefill. Keep a separate, root-pruned causal branch
    // for the opt-in prefill executor: with an empty cache it gives every
    // sequence tap the same causal window repeated one-position decode would.
    let (query_prefill_raw, key_prefill_raw, value_prefill) = append_qwen35_conv_raw(
        program,
        qkv_mixed,
        conv_weight,
        one,
        key_dim,
        value_dim,
        l_cache,
    )?;
    let query_prefill_split = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (query_prefill_raw, q_split_map.as_str()),
            (key_head_ones, "ui->sui"),
        ],
    )?;
    let key_prefill_split = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (key_prefill_raw, q_split_map.as_str()),
            (key_head_ones, "ui->sui"),
        ],
    )?;
    let query_prefill = l2norm_with_eps_map(
        program,
        query_prefill_split,
        eps,
        "sui->sui",
        "su->sui",
        "s->su",
    )?;
    let key_prefill = l2norm_with_eps_map(
        program,
        key_prefill_split,
        eps,
        "sui->sui",
        "su->sui",
        "s->su",
    )?;
    let query_prefill_repeated = repeat_kv_heads(program, query_prefill, kv_heads, group)?;
    let key_prefill_repeated = repeat_kv_heads(program, key_prefill, kv_heads, group)?;
    let value_prefill_split = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (value_prefill, v_split_map.as_str()),
            (value_head_ones, "ugj->sugj"),
        ],
    )?;
    let query_sequence = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        query_prefill_repeated,
        "sugd->sugd",
        "sdug->sugd",
    )?;
    let key_sequence = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        key_prefill_repeated,
        "sugd->sugd",
        "sdug->sugd",
    )?;
    let value_sequence = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        value_prefill_split,
        "sugj->sugj",
        "sjug->sugj",
    )?;

    let (mixer_out, taps) = if let Some(width) = prefill_width
        && width > 1
    {
        // A prefill-width recurrence: elementwise/reduce cannot express the
        // state-dependent scan over `s` in one vectorized pass (each
        // position's state depends on every earlier position's decayed
        // state), so this Rust loop unrolls it -- legitimate because `width`
        // is a plain build-time `u32` here, never the symbolic `s` the
        // architecture-level spec carries (see the `prefill_width` guard
        // above). Each iteration threads `state_out` into the next
        // `state_in` exactly as the decode path threads it call to call.
        let position_axis = op::append(
            program,
            Op::Iota {
                dtype: DType::Float32,
                extent: Extent::Static(width),
            },
        );

        let mut last_step = qwen35_gdn_recurrence_step(
            program,
            query_sequence,
            key_sequence,
            value_sequence,
            beta_split,
            gate_split,
            z_split,
            state_in,
            inv_sqrt_key_dim,
            0,
        )?;
        let mut delta_out_stacked =
            qwen35_gdn_place_position(program, last_step.delta_out, position_axis, 0)?;
        let mut per_position_state_out = alloc::vec![last_step.state_out];

        for position in 1..width {
            let step = qwen35_gdn_recurrence_step(
                program,
                query_sequence,
                key_sequence,
                value_sequence,
                beta_split,
                gate_split,
                z_split,
                last_step.state_out,
                inv_sqrt_key_dim,
                position,
            )?;
            let placed = qwen35_gdn_place_position(program, step.delta_out, position_axis, position)?;
            delta_out_stacked = elementwise(
                program,
                DType::Float32,
                ScalarOp::Add,
                &[(delta_out_stacked, "sjug->sjug"), (placed, "sjug->sjug")],
            )?;
            per_position_state_out.push(step.state_out);
            last_step = step;
        }

        let delta_out = delta_out_stacked;
        let state_out = last_step.state_out;

        // `append_qwen35_gdn_sequence_tail_with_taps` is the same
        // RMSNorm-gate-out-proj-residual tail the M=1 branch below computes
        // inline: reused rather than duplicated, per its own doc ("a
        // sans-IO executor supplies its caller-owned `[s,j,u,g]` scan
        // output as `delta_out`") -- this recurrence is exactly that
        // caller.
        let tail = append_qwen35_gdn_sequence_tail_with_taps(
            program,
            Qwen35GdnSequenceTail {
                x,
                delta_out,
                z: z_split,
                head_eps,
                inv_head_v_dim,
                norm_weight: ssm_norm_weight,
                out_weight: ssm_out,
                head_v_dim,
                kv_heads,
                group,
            },
        )?;

        let taps = SsmMixerTaps {
            per_position_state_out,
            qkv_mixed,
            query_sequence,
            key_sequence,
            value_sequence,
            gate_sequence: gate_split,
            beta_sequence: beta_split,
            z_sequence: z_split,
            query: last_step.query,
            key: last_step.key,
            value: last_step.value,
            gate: last_step.gate,
            beta: last_step.beta,
            state_in,
            z_head: last_step.z_head,
            state_out,
            delta_out: last_step.delta_out,
            z,
            gated_rmsnorm_out: tail.gated_value,
            gated_value: tail.gated_value,
            ssm_out_result: tail.projected,
        };
        (tail.output, taps)
    } else {
        // squeeze the size-1 decode-step `s` axis away -- `append_qwen35_delta_net_step`
        // has no `s` letter at all (a single already-selected token per its own
        // doc), and reordering the surviving letters here (`dug`, not `ugd`)
        // doubles as the transpose `append_qwen35_delta_net_step`'s own
        // `i{head}`/`j{head}` maps expect.
        let query = reduce(
            program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            q_repeated,
            "sugd->sugd",
            "dug->sugd",
        )?;
        let key = reduce(
            program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            k_repeated,
            "sugd->sugd",
            "dug->sugd",
        )?;
        let value = reduce(
            program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            v_split,
            "sugj->sugj",
            "jug->sugj",
        )?;
        let beta = reduce(
            program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            beta_split,
            "sug->sug",
            "ug->sug",
        )?;
        let gate = reduce(
            program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            gate_split,
            "sug->sug",
            "ug->sug",
        )?;
        let z_head = reduce(
            program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            z_split,
            "sugj->sugj",
            "ugj->sugj",
        )?;

        let (delta_out, state_out) = append_qwen35_delta_net_step(
            program,
            query,
            key,
            value,
            gate,
            beta,
            state_in,
            inv_sqrt_key_dim,
            "ug",
        )?;

        // gated RMSNorm over the per-head value axis `j`, `head_eps`/`inv_head_v_dim`
        // matched to the surviving `u,g` head space -- `build_norm_gated`
        // (`qwen35.cpp:243-250`): `rmsnorm(out, weight) * output_gate(z)`,
        // `output_gate` per [`GdnOutputGate`] (silu for qwen35, sigmoid for
        // qwen4exp, reference: PR 27742 line 2896-2899).
        let squared = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(delta_out, "jug->jug"), (delta_out, "jug->jug")],
        )?;
        let sum_squares = reduce(
            program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            squared,
            "jug->jug",
            "ug->jug",
        )?;
        let mean_square = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(sum_squares, "ug->ug"), (inv_head_v_dim, "->ug")],
        )?;
        let mean_square_eps = elementwise(
            program,
            DType::Float32,
            ScalarOp::Add,
            &[(mean_square, "ug->ug"), (head_eps, "ug->ug")],
        )?;
        let rms = elementwise(
            program,
            DType::Float32,
            ScalarOp::SquareRoot,
            &[(mean_square_eps, "ug->ug")],
        )?;
        let inv_rms = elementwise(
            program,
            DType::Float32,
            ScalarOp::Reciprocal,
            &[(rms, "ug->ug")],
        )?;
        let normed_out = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(delta_out, "jug->jug"), (inv_rms, "ug->jug")],
        )?;
        let normed_out_gamma = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(normed_out, "jug->jug"), (ssm_norm_weight, "j->jug")],
        )?;
        let gated_out_map = if v_head_reordered {
            "jug->guj"
        } else {
            "jug->jug"
        };
        let gated_gate_map = if v_head_reordered {
            "ugj->guj"
        } else {
            "ugj->jug"
        };
        let gated_product_map = if v_head_reordered {
            "guj->gujd"
        } else {
            "jug->gujd"
        };
        let gated_out = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(normed_out_gamma, gated_out_map), (z_head, gated_gate_map)],
        )?;

        // output projection: `ssm_out`'s declared `[value_dim, n_embd]` layout
        // decomposed the same read-side way `q_split_map`/`v_split_map` already
        // decompose a flat checkpoint axis -- never a write-side merge. UNLIKE
        // `v_split_map`'s own `(u*group+g)*head_v_dim+j` nesting (u outer, g
        // mid, j inner -- verified correct against real Q4_K bytes through
        // `gated_value`, within noise), the checkpoint's real `ssm_out.weight`
        // contraction axis nests `(j*kv_heads+u)*group+g` (j outer, u mid, g
        // inner) -- proven on real `qwen3.6:35b-a3b` bytes (a model-crate
        // `qwen35moe_layer0_stage_by_stage_position0_matches_tapped_reference`'s
        // own `ssm_out_weight_layout_sweep`: this order scores
        // `scaled_rel_err=1.5e-2`, at the Q4_K noise floor, against every other
        // (u,g,j)-role permutation scoring `>=1.2`) -- this weight was saved
        // with a different head/value nesting than the value/`z`/qkv weights,
        // not the same convention reused.
        let out_weight_split_map = alloc::format!("{}*j+{group}*u+g,d->ugjd", kv_heads * group);
        let ssm_out_split = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[
                (ssm_out, out_weight_split_map.as_str()),
                (value_head_ones, "ugj->ugjd"),
            ],
        )?;
        let cur_product = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[
                (gated_out, gated_product_map),
                (ssm_out_split, "ugjd->gujd"),
            ],
        )?;
        let cur = reduce(
            program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            cur_product,
            "gujd->gujd",
            "d->gujd",
        )?;

        let mixer_out = elementwise(
            program,
            DType::Float32,
            ScalarOp::Add,
            &[(x, "sd->sd"), (cur, "d->sd")],
        )?;

        let taps = SsmMixerTaps {
            per_position_state_out: alloc::vec![state_out],
            qkv_mixed,
            query_sequence,
            key_sequence,
            value_sequence,
            gate_sequence: gate_split,
            beta_sequence: beta_split,
            z_sequence: z_split,
            query,
            key,
            value,
            gate,
            beta,
            state_in,
            z_head,
            state_out,
            delta_out,
            z,
            gated_rmsnorm_out: normed_out_gamma,
            gated_value: gated_out,
            ssm_out_result: cur,
        };
        (mixer_out, taps)
    };
    Ok((mixer_out, taps))
}

