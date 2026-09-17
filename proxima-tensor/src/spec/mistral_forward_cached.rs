use super::*;

/// The whole model as one program: token embedding lookup, `block_count`
/// copies of `specs/mistral_layer.toml`'s layer (each with its own weights,
/// same node shape, generated rather than hand-authored — this is the whole
/// reason this function exists, since a 32-layer TOML would repeat one graph
/// 32 times with nothing but the weight names differing), a final RMSNorm,
/// and the LM head projection down to `[seq, vocab]` logits.
///
/// Config is plain `u32` parameters, not a struct — nothing here needs a
/// caller to hold them together as one type, and this crate deleted
/// `TensorExecutionConfig` for being unread rather than reintroduce that
/// shape. Composes this module's own `elementwise`/`reduce` (the exact
/// notation grammar `Vec<Op>::try_from(&ProgramSpec)` above already parses),
/// `embedding_lookup` (`shape.rs`'s `embedding_lookup_program` unit test is
/// the addressing reference), and `append_mistral_layer` (mirrors
/// `specs/mistral_layer.toml` node for node).
///
/// `expert_count == 0` means dense: every layer binds
/// `append_mistral_layer`'s plain `ffn_{gate,up,down}.weight` triple,
/// node-for-node the same program this function has always built, so a
/// dense checkpoint's generated program (and therefore its output) is
/// unaffected by this parameter's existence. `expert_count > 0` routes each
/// layer through `append_mistral_moe_layer` instead, gathering one of
/// `expert_count` experts' weight slabs per token per
/// `append_moe_ffn`'s doc.
#[allow(clippy::too_many_arguments)]
pub fn mistral_forward_program(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    block_count: u32,
    expert_count: u32,
    expert_used_count: u32,
) -> Result<Vec<Op>, TensorError> {
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
    // attention's usual `1/sqrt(d_k)`, the same two IEEE ops the deleted
    // five-node `Iota` derivation performed, at build time instead of once
    // per forward pass.
    let inv_sqrt_head_dim = scalar_constant(&mut program, 1.0 / (head_dim as f32).sqrt());
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
    // the one constant here that is not rank-0: `q_*_grouped`'s `u` and `g`
    // iteration extents have no other operand to come from, so this leaf
    // carries them. Its values are all `1.0` either way.
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
        x = if expert_count == 0 {
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

            append_mistral_layer(
                &mut program,
                x,
                inv_dim,
                eps,
                ones,
                inv_sqrt_head_dim,
                cos,
                sin,
                group_ones,
                is_future,
                neg_infinity,
                group,
                attn_norm_weight,
                ffn_norm_weight,
                wq,
                wk,
                wv,
                wo,
                w_gate,
                w_up,
                w_down,
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
                    Extent::Static(feed_forward),
                ],
                &alloc::format!("blk.{layer}.ffn_gate_exps.weight"),
            );
            let expert_w_up = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(expert_count),
                    Extent::Static(embedding),
                    Extent::Static(feed_forward),
                ],
                &alloc::format!("blk.{layer}.ffn_up_exps.weight"),
            );
            let expert_w_down = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(expert_count),
                    Extent::Static(feed_forward),
                    Extent::Static(embedding),
                ],
                &alloc::format!("blk.{layer}.ffn_down_exps.weight"),
            );

            let (next_x, site) = append_mistral_moe_layer(
                &mut program,
                layer,
                x,
                inv_dim,
                eps,
                ones,
                inv_sqrt_head_dim,
                cos,
                sin,
                group_ones,
                is_future,
                neg_infinity,
                group,
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
            )?;
            moe_sites.push(site);
            next_x
        };
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
    reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        logits_product,
        "sdv->sdv",
        "sv->sdv",
    )?;

    Ok(program)
}

/// One transformer layer's per-position outputs the caller appends into its
/// own key/value cache for the next call: `k_even`/`k_odd` are RoPE-rotated
/// already (the same halves this module's per-layer attention consumes
/// directly, so a later call never re-derives RoPE for a position it has
/// already seen), `v` is the un-rotated projected value. Not a library type
/// outside this module — three [`NodeId`]s a caller collects once per layer,
/// nothing more.
pub type CachedLayerRoots = (NodeId, NodeId, NodeId);

/// [`mistral_cached_forward_program_with_experts`]'s two named roots:
/// `logits` (the vocab-projection reduce, this program's terminal node) and
/// `hidden` (`normed_final` -- the LAST-norm activation `logits` is
/// projected FROM, one layer earlier in the graph). Before this type
/// existed, a caller that needed `hidden` (an embedding pooling the final
/// hidden state rather than decoding a token) had no way to reach it except
/// `NodeId(logits.0 - 3)` -- arithmetic over [`op::append`]'s id-is-index
/// invariant that silently breaks the moment a refactor inserts or removes
/// one node between `hidden` and `logits`. A plain two-field struct instead
/// of widening this builder's return arity again: every existing caller
/// that only wants `logits` destructures `ForwardRoots { logits, .. }` (one
/// pattern, no behavior change); `proxima-model-interop`'s
/// `LoadedModel::embed` is the one caller that reads `hidden` too.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForwardRoots {
    pub logits: NodeId,
    pub hidden: NodeId,
}

/// [`mistral_single_range_cached_forward_program`]'s own return shape:
/// the lowered program, its `logits` root, one [`CachedLayerRoots`] per
/// layer, and ROW 326/328's [`DuplicateHeadPosition`] scratch output
/// (`Some` only when that position is not [`DuplicateHeadPosition::None`]
/// -- see the function's own doc).
pub(super) type SingleRangeForwardProgram =
    (Vec<Op>, NodeId, Vec<CachedLayerRoots>, Option<NodeId>);

/// [`mistral_cached_forward_program_with_experts_and_layer_taps`]'s own
/// return shape: the lowered program, its [`ForwardRoots`], one
/// [`CachedLayerRoots`] per layer, one residual [`NodeId`] per layer
/// (that function's own doc on what the fourth element is for), and one
/// [`MoeSite`] per MoE layer (empty on a dense checkpoint).
pub(super) type MistralMoeForwardProgramWithLayerTaps = (
    Vec<Op>,
    ForwardRoots,
    Vec<CachedLayerRoots>,
    Vec<NodeId>,
    MoeSites,
);

/// Where, if anywhere, the ROW 326/328 diagnostic duplicate `output.weight`
/// reduce is emitted relative to the real head -- ROW 328 turns ROW 326's
/// original bool into this 3-way position to test whether the ~1.8ms head
/// cost follows a fixed slot in program order or the first GPU touch of the
/// `output.weight` range after 4 GB of other layer traffic has streamed
/// through the same no-copy mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DuplicateHeadPosition {
    /// no scratch reduce -- every production checkpoint.
    #[default]
    None,
    /// scratch reduce reads `x`, the raw embedding lookup output, before
    /// layer 0 runs -- first dispatch of the token, `output.weight` is
    /// touched before any other per-layer weight this token.
    Before,
    /// scratch reduce reads `normed_final` immediately after the real
    /// head -- last dispatch of the token, ROW 326's original behavior.
    After,
}

/// `append_qwen35_dense_attention_layer`'s own per-position cache roots --
/// [`CachedLayerRoots`]'s 4-wide counterpart, one extra [`NodeId`] for the
/// partial-rotary remainder [`CachedLayerRoots`] has no room for: `k_first`/
/// `k_second` are this checkpoint's split-half (NEOX/IMROPE-style) RoPE
/// halves of the rotated prefix (`k[..., :rotary_dim]`,
/// `modeling_qwen3_next.py:205-210`), `k_pass` is the untouched remainder
/// (`k[..., rotary_dim:]`, `modeling_qwen3_next.py:206`, concatenated back
/// in the oracle, never dropped), `v` the un-rotated projected value.
pub type Qwen35DenseAttentionRoots = (NodeId, NodeId, NodeId, NodeId);

/// [`append_mistral_layer`]'s key/value-cached counterpart: `x` carries only
/// the `new` positions this call introduces (`s`, sized by symbol 0), and
/// attention blends two disjoint key/value sources instead of one —
/// `k_even_cache`/`k_odd_cache`/`v_cache` (already-rotated positions from
/// every earlier call, bound [`Op::Input`] sized by symbol 1) and this
/// call's own freshly projected/rotated `k_new`/`v_new` (`w`, same size as
/// `s`, computed in-graph). Two [`Op::Reduce`] blocks — one per source —
/// combine through online-softmax arithmetic (`Maximum` for the shared max,
/// `Add` for the shared normalizer) rather than a literal concatenation:
/// [`Reduce::out_map`] must stay a pure projection
/// (`shape::project_output_shape`'s own doc), so nothing upstream of a
/// reduce can splice two tensors into one axis. The masking-only-within-`s,w`
/// asymmetry is what makes this correct without a `cached_len` scalar: a
/// cached key is definitionally in the past of every new query, and
/// `is_future` (built once by [`causal_mask`], sized `[s,w]` since `w` and
/// `s` share symbol 0's extent) already forbids a new query attending a
/// later new key, so the cached block never needs masking at all.
///
/// Returns `(x_next, k_new_even, k_new_odd, v_new)` — `x_next` feeds the next
/// layer (or the final RMSNorm/LM head after the last one), and the other
/// three are this layer's [`CachedLayerRoots`] for the caller to append.
///
/// `qk_norm`, when `Some((q_norm_weight, k_norm_weight, inv_head_dim))`, runs
/// [`rmsnorm_per_head`] on `q`/`k_new` right after their projection and
/// strictly BEFORE RoPE -- Qwen3's own per-head QK-norm
/// (`Qwen3Attention.q_norm`/`.k_norm`, `modeling_qwen3.py`, applied to
/// `query_states`/`key_states` before `apply_rotary_pos_emb`). `None` skips
/// both calls entirely, leaving `q`/`k_new` exactly as
/// [`mistral_cached_forward_program`]'s own dense checkpoints have always
/// computed them -- this one flag is what lets a single layer builder serve
/// both architectures rather than forking a parallel copy for the two extra
/// ops Qwen3 needs.
///
/// `paired_gate_up_reduce`, when `true`, requires the caller to have passed
/// the SAME `NodeId` for both `w_gate` and `w_up` -- a single `[2,
/// feed_forward, embedding]` leaf (gate rows then up rows, one dispatch
/// binds it) rather than two separate `[embedding, feed_forward]` leaves.
/// `gate`/`up` are then read back out of ONE `Op::Reduce`'s `[s, feed_forward]`
/// output via the parity axis fixed at `0`/`1` (the constant-offset
/// axis-expression grammar `spec.rs`'s own module doc already carries,
/// `"s,0*s+K,g->sg"` -- a zero-coefficient term on an unrelated iteration
/// letter selects a compile-time-fixed operand axis without adding an
/// `IndexMap` variant). `false` reproduces today's two independent matvecs
/// byte-for-byte -- every op below this branch is unaffected by which side
/// ran. Default `false` at every production call site; flipping it removes
/// one `Op::Reduce` (and its own kernel dispatch) per layer at load-time
/// cost only when the checkpoint's `ffn_gate`/`ffn_up` tensors are not
/// byte-adjacent (`proxima-model-interop::bind::bind_matmul_weight_paired`).
///
/// `fused_qkv_reduce`, when `true`, requires `query_heads` (this is the ONE
/// extra scalar this branch needs that `paired_gate_up_reduce` did not: q's
/// row count differs from k's/v's under GQA, so the flat row axis cannot
/// be recovered from `group`/`head_dim` alone), `head_shape_ones`/
/// `kv_head_shape_ones` (`[query_heads, head_dim]`/`[kv_heads, head_dim]`
/// constants of `1.0`, this function's own doc on those two parameters
/// below), and the SAME `NodeId` passed for `wq`, `wk`, `wv` -- a single
/// `[query_heads + 2 * kv_heads, head_dim, embedding]`-flattened-to-`[rows,
/// embedding]` leaf (q rows, then k rows, then v rows) rather than three
/// separate `[embedding, heads, head_dim]` leaves.
///
/// The IR CANNOT split the fused reduce's one real `[s, rows]` axis back
/// into two independent virtual sub-axes (`h`/`d` for q, `u`/`d` for k/v)
/// from a single operand alone -- confirmed empirically
/// (`shape::infer`'s own `UnconstrainedDim`, not inferred): a compound
/// axis term like `"{head_dim}*h+d"` needs BOTH `h`'s and `d`'s extents
/// pinned by SOME operand's own real, uncompounded axis, and an
/// `Op::Elementwise`'s operand count is fixed to its `ScalarOp`'s arity
/// (`op::ScalarOp::arity`), so there is no room to add a pure
/// shape-providing operand to an already-binary op (`Multiply(gate,
/// weight)`) the way `paired_gate_up_reduce`'s zero-coefficient parity
/// trick could ride a term that was already a bare constant. This is why
/// `paired_gate_up_reduce` (identical row counts either side of its split)
/// could read `gate`/`up` back with ZERO extra dispatch, and this flag
/// (three DIFFERENT row counts under GQA, so no single shared axis exists
/// to split on) cannot: `q_raw`/`k_new_raw`/`v_new` each need their own
/// `ScalarOp::Multiply`-against-a-ones-shaped-constant extract (arity 2,
/// satisfying shape inference; the ones constant contributes only shape,
/// value `1.0`, so the extracted values are bit-identical to a direct
/// read) -- three small dispatches, not one, added back. Net per layer: 3
/// reduces removed, 1 fused reduce added, 3 small extracts added -- ONE
/// MORE dispatch, not fewer. This corrects this feature's own premise (a
/// bandwidth-only reading of ROW 336 predicted `-2`/layer); the measured
/// win, if any, is per-dispatch bandwidth on the one big reduce, not
/// dispatch count. Requires `qk_norm` be `None` -- QK-norm's
/// `rmsnorm_per_head` call needs `q_raw`/`k_new_raw` as their own
/// full-shape node regardless, which this branch already provides via the
/// same extract, but no call site in this crate combines the two flags
/// today and the combination is untested.
/// Where `append_mistral_cached_layer` reads `q`/`k_new`/`v_new` from --
/// [`Self::Split`] is today's three independent reduces; [`Self::Fused`]
/// carries the one shared flat-row reduce plus the byte offset `v_new`'s
/// rows start at within it (`fused_qkv_reduce`'s own doc on that function).
pub(super) enum QkvSource {
    Split,
    Fused {
        node: NodeId,
        v_offset: u32,
        head_dim: u32,
    },
}

#[allow(clippy::too_many_arguments)]
pub fn append_mistral_cached_layer(
    program: &mut Vec<Op>,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    ones: NodeId,
    inv_sqrt_head_dim: NodeId,
    cos_new: NodeId,
    sin_new: NodeId,
    group_ones: NodeId,
    head_shape_ones: NodeId,
    kv_head_shape_ones: NodeId,
    is_future: NodeId,
    group: u32,
    head_dim: u32,
    query_heads: u32,
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
    q_bias: Option<NodeId>,
    k_bias: Option<NodeId>,
    v_bias: Option<NodeId>,
    paired_gate_up_reduce: bool,
    fused_qkv_reduce: bool,
    rope_pairing: RopePairing,
) -> Result<(NodeId, CachedLayerRoots), TensorError> {
    let normed = rmsnorm(program, x, attn_norm_weight, inv_dim, eps)?;
    let kv_heads = query_heads / group;

    let (q_raw, k_new_raw, v_new_source): (NodeId, NodeId, QkvSource) = if fused_qkv_reduce {
        let qkv_product = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(normed, "si->spi"), (wq, "pi->spi")],
        )?;
        let qkv_reduced = reduce(
            program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            qkv_product,
            "spi->spi",
            "sp->spi",
        )?;
        let q_raw = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[
                (
                    qkv_reduced,
                    alloc::format!("s,{head_dim}*h+d->shd").as_str(),
                ),
                (head_shape_ones, "hd->shd"),
            ],
        )?;
        let k_offset = query_heads * head_dim;
        let k_new_raw = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[
                (
                    qkv_reduced,
                    alloc::format!("s,{head_dim}*u+d+{k_offset}->sud").as_str(),
                ),
                (kv_head_shape_ones, "ud->sud"),
            ],
        )?;
        (
            q_bias.map_or(Ok(q_raw), |bias| {
                elementwise(
                    program,
                    DType::Float32,
                    ScalarOp::Add,
                    &[(q_raw, "shd->shd"), (bias, "hd->shd")],
                )
            })?,
            k_bias.map_or(Ok(k_new_raw), |bias| {
                elementwise(
                    program,
                    DType::Float32,
                    ScalarOp::Add,
                    &[(k_new_raw, "sud->sud"), (bias, "ud->sud")],
                )
            })?,
            QkvSource::Fused {
                node: qkv_reduced,
                v_offset: (query_heads + kv_heads) * head_dim,
                head_dim,
            },
        )
    } else {
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
        (
            q_bias.map_or(Ok(q_raw), |bias| {
                elementwise(
                    program,
                    DType::Float32,
                    ScalarOp::Add,
                    &[(q_raw, "shd->shd"), (bias, "hd->shd")],
                )
            })?,
            k_bias.map_or(Ok(k_new_raw), |bias| {
                elementwise(
                    program,
                    DType::Float32,
                    ScalarOp::Add,
                    &[(k_new_raw, "sud->sud"), (bias, "ud->sud")],
                )
            })?,
            QkvSource::Split,
        )
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

    let v_new = match v_new_source {
        QkvSource::Fused {
            node,
            v_offset,
            head_dim,
        } => {
            let v_raw = elementwise(
                program,
                DType::Float32,
                ScalarOp::Multiply,
                &[
                    (
                        node,
                        alloc::format!("s,{head_dim}*u+d+{v_offset}->sud").as_str(),
                    ),
                    (kv_head_shape_ones, "ud->sud"),
                ],
            )?;
            v_bias.map_or(Ok(v_raw), |bias| {
                elementwise(
                    program,
                    DType::Float32,
                    ScalarOp::Add,
                    &[(v_raw, "sud->sud"), (bias, "ud->sud")],
                )
            })
        }
        QkvSource::Split => {
            let v_product = elementwise(
                program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(normed, "si->sudi"), (wv, "iud->sudi")],
            )?;
            let v_raw = reduce(
                program,
                DType::Float32,
                ScalarOp::Add,
                ReduceInit::Zero,
                v_product,
                "sudi->sudi",
                "sud->sudi",
            )?;
            v_bias.map_or(Ok(v_raw), |bias| {
                elementwise(
                    program,
                    DType::Float32,
                    ScalarOp::Add,
                    &[(v_raw, "sud->sud"), (bias, "ud->sud")],
                )
            })
        }
    };
    let v_new = v_new?;

    // The pairing is an architecture property, not a proxy for whether the
    // checkpoint carries QK-norm weights: Qwen2 uses NEOX split-half RoPE
    // without QK-norm, while ordinary LLaMA/Mistral checkpoints use the
    // converter's interleaved layout.
    // `q`/`k_new` are real, fully materialized `[s,h,d]`/`[s,u,d]` nodes
    // under BOTH `QkvSource` variants (the `Multiply`-by-shape-constant
    // extract above already re-materializes them under `Fused`), so every
    // op below reads them exactly as the split path always has -- zero
    // further changes needed downstream of this point.
    let (rotated_q_even, rotated_q_odd, rotated_k_new_even, rotated_k_new_odd) = match rope_pairing
    {
        RopePairing::SplitHalf { .. } => {
            let (rotated_q_first, rotated_q_second) =
                fused_rope_pair(program, q, 'h', cos_new, sin_new, rope_pairing)?;
            let (rotated_k_first, rotated_k_second) =
                fused_rope_pair(program, k_new, 'u', cos_new, sin_new, rope_pairing)?;

            (
                rotated_q_first,
                rotated_q_second,
                rotated_k_first,
                rotated_k_second,
            )
        }
        RopePairing::Interleaved => {
            let (rotated_q_even, rotated_q_odd) =
                fused_rope_pair(program, q, 'h', cos_new, sin_new, rope_pairing)?;
            let (rotated_k_new_even, rotated_k_new_odd) =
                fused_rope_pair(program, k_new, 'u', cos_new, sin_new, rope_pairing)?;

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

    // cached block: query `s` against every already-rotated cached key `t`
    // (symbol 1's extent, zero on the very first call) -- never masked, a
    // cached position is always in the past of a new query.
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
    // new block: query `s` against this call's own freshly rotated key `w`
    // (symbol 0's extent, same range as `s`) -- causal within the block,
    // reusing `is_future` unchanged since it is already `[s, w]`-shaped.
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

    // online-softmax combine: two disjoint key ranges, one shared max and
    // one shared normalizer, no literal concatenation anywhere.
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

    #[cfg(feature = "instrument")]
    instrument::record_online_softmax_block_range(score_max_cached, attended);

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

    // `paired_gate_up_reduce`: one `Op::Reduce` over `w_gate` (== `w_up`,
    // caller's contract, see this function's own doc) read as `[2,
    // feed_forward, embedding]` replaces the two independent matvecs below.
    // `out_map`'s letter order ("spg", not "sgp") is load-bearing, not
    // stylistic: `proxima_tensor::bind::correct_packed_matmul_layouts`
    // derives a packed weight's native stride per output axis from
    // `output_axes`' LISTED order (last-listed axis lands innermost, closest
    // to the reduced axis) -- "spg" places the broadcast `s` axis outermost
    // (its stride is discarded either way) and `g` innermost-of-features so
    // its native stride comes out `embedding` (one row), leaving `p`'s
    // native stride `feed_forward * embedding` (one whole gate/up half) --
    // exactly the real concatenated checkpoint's byte layout. `"sgp"` derives
    // the opposite (interleaved) stride pair and silently mis-reads the
    // buffer.
    let (gate, up, gate_map, up_map): (NodeId, NodeId, &str, &str) = if paired_gate_up_reduce {
        let paired_product = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(normed2, "sd->sdgp"), (w_gate, "pgd->sdgp")],
        )?;
        let paired_result = reduce(
            program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            paired_product,
            "sdgp->sdgp",
            "spg->sdgp",
        )?;
        (
            paired_result,
            paired_result,
            "s,0*s+0,g->sg",
            "s,0*s+1,g->sg",
        )
    } else {
        let gate_product = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(normed2, "sd->sdg"), (w_gate, "dg->sdg")],
        )?;
        let gate = reduce(
            program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            gate_product,
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
        (gate, up, "sg->sg", "sg->sg")
    };

    let neg_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Negate,
        &[(gate, gate_map)],
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
        &[(gate, gate_map), (sigmoid_gate, "sg->sg")],
    )?;
    let ffn_hidden = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(silu_gate, "sg->sg"), (up, up_map)],
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

/// Hyper-connections replace a single residual stream with `hc` parallel
/// copies (`x`, shape `[tokens, hc, embedding]`, letters `s,h,i`) and mix
/// them down to one stream a token mixer or FFN can consume -- reference:
/// PR 27742 line 2705-2751, `build_hc_mix`. Built entirely from
/// [`elementwise`]/[`reduce`]/[`silu`]/[`sigmoid`], the same primitives
/// every other builder in this module composes -- no new [`Op`] variant
/// (`flash-next-plan.md` §3's own "hyper-connections" row: "fully
/// expressible IN the existing `Op` vocabulary").
///
/// Steps, each named after its `build_hc_mix` counterpart:
/// 1. Grouped RMSNorm (reference line 2717-2722): `x` normalized over the
///    embedding axis `i` *per stream* `h`, then scaled by `w_norm`
///    (`[hc, embedding]`, one gamma per `(stream, channel)` pair -- the
///    checkpoint's own flat `[hc_dim]` gamma reshaped, never a single
///    shared-across-streams gamma the way [`rmsnorm_per_head`]'s `gamma`
///    is shared across heads).
/// 2. Low-rank gate (reference line 2724-2729): `xn` down-projected
///    (`w_down`, `[hc, embedding, low_rank]`) to `[tokens, low_rank]`,
///    scaled by `1/hc`, `silu`'d, up-projected (`w_up`,
///    `[low_rank, hc, embedding]`) back to `[tokens, hc, embedding]`, then
///    `sigmoid`'d into a gate multiplied against `xn`.
/// 3. Mean-collapse (reference line 2732-2743): the gated `[tokens, hc,
///    embedding]` stream summed over `h` and scaled by `1/hc`.
/// 4. Optional inject (reference line 2745-2748): `w_inject`
///    (`[hc, embedding, hc]`) projects `xn` to a `[tokens, hc]` scatter
///    weight [`append_hyper_connection_combine`] consumes -- `None` for the
///    final output mixer (reference line 2860-2862: "there is no
///    output_norm: the final hyper-connection mixer carries it"), `Some`
///    for every per-layer attn/ffn hyper-connection module (reference line
///    2816-2821, 2843-2848).
///
/// `inv_dim` is `1/embedding` (the RMSNorm mean, [`rmsnorm`]'s own
/// parameter); `inv_hc` is `1/hc`, reused for the low-rank gate's scale,
/// the mean-collapse scale, and (when `w_inject` is `Some`) the inject
/// scale the caller's own [`append_hyper_connection_combine`] finishes.
///
/// Returns `(mixed, inject)`, `mixed` shaped `[tokens, embedding]`,
/// `inject` shaped `[tokens, hc]` (`Some` iff `w_inject` was `Some`).
///
/// No `qwen4exp_forward_program` call site lands in this crate (that
/// assembly is model-specific and lives in its own consuming crate); this
/// builder is public so that crate can compose one. See
/// [`qwen35_forward_program`] for this crate's own worked example of
/// wiring per-layer builders like this one into a full program.
#[allow(clippy::too_many_arguments)]
pub fn append_hyper_connection_mix(
    program: &mut Vec<Op>,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    inv_hc: NodeId,
    one: NodeId,
    w_norm: NodeId,
    w_down: NodeId,
    w_up: NodeId,
    w_inject: Option<NodeId>,
) -> Result<(NodeId, Option<NodeId>), TensorError> {
    let squared = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(x, "shi->shi"), (x, "shi->shi")],
    )?;
    let sum_squares = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        squared,
        "shi->shi",
        "sh->shi",
    )?;
    let mean_square = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(sum_squares, "sh->sh"), (inv_dim, "->sh")],
    )?;
    let mean_square_eps = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(mean_square, "sh->sh"), (eps, "s->sh")],
    )?;
    let rms = elementwise(
        program,
        DType::Float32,
        ScalarOp::SquareRoot,
        &[(mean_square_eps, "sh->sh")],
    )?;
    let inv_rms = elementwise(
        program,
        DType::Float32,
        ScalarOp::Reciprocal,
        &[(rms, "sh->sh")],
    )?;
    let normed = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(x, "shi->shi"), (inv_rms, "sh->shi")],
    )?;
    let xn = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "shi->shi"), (w_norm, "hi->shi")],
    )?;

    let down_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(xn, "shi->shir"), (w_down, "hir->shir")],
    )?;
    let down_sum_i = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        down_product,
        "shir->shir",
        "shr->shir",
    )?;
    let lo = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        down_sum_i,
        "shr->shr",
        "sr->shr",
    )?;
    let lo_scaled = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(lo, "sr->sr"), (inv_hc, "->sr")],
    )?;
    let lo_silu = silu(program, lo_scaled, one, "sr->sr")?;

    let up_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(lo_silu, "sr->srhi"), (w_up, "rhi->srhi")],
    )?;
    let up_sum_r = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        up_product,
        "srhi->srhi",
        "shi->srhi",
    )?;
    let gate = sigmoid(program, up_sum_r, one, "shi->shi")?;

    let gated = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(xn, "shi->shi"), (gate, "shi->shi")],
    )?;
    let mixed_sum = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        gated,
        "shi->shi",
        "si->shi",
    )?;
    let mixed = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(mixed_sum, "si->si"), (inv_hc, "->si")],
    )?;

    let inject = match w_inject {
        Some(w_inject) => {
            let inject_product = elementwise(
                program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(xn, "shi->shio"), (w_inject, "hio->shio")],
            )?;
            let inject_sum_i = reduce(
                program,
                DType::Float32,
                ScalarOp::Add,
                ReduceInit::Zero,
                inject_product,
                "shio->shio",
                "sho->shio",
            )?;
            let inject_flat = reduce(
                program,
                DType::Float32,
                ScalarOp::Add,
                ReduceInit::Zero,
                inject_sum_i,
                "sho->sho",
                "so->sho",
            )?;
            Some(inject_flat)
        }
        None => None,
    };

    Ok((mixed, inject))
}

/// The residual side of a hyper-connection module -- reference: PR 27742
/// line 2753-2773, `build_hc_combine`: `2*sigmoid(inject/hc)` centres the
/// per-stream scatter weight on `1`, so a zero injection degenerates to a
/// plain residual add, then `block_out` (`[tokens, embedding]`) broadcasts
/// across every stream, scaled by that weight, and adds into `residual`
/// (`[tokens, hc, embedding]`). Pairs with
/// [`append_hyper_connection_mix`]'s `Some(w_inject)` arm; the final output
/// mixer has no combine call (reference line 2860-2869: the mixed stream
/// feeds `output` directly).
///
/// Returns the updated `[tokens, hc, embedding]` residual.
///
/// Same rationale as [`append_hyper_connection_mix`]'s own doc: no
/// production call site in this crate, public so a foreign architecture
/// crate can compose one. See [`qwen35_forward_program`] for this crate's
/// own worked example of a full per-layer builder chain.
#[allow(clippy::too_many_arguments)]
pub fn append_hyper_connection_combine(
    program: &mut Vec<Op>,
    residual: NodeId,
    block_out: NodeId,
    inject: NodeId,
    inv_hc: NodeId,
    one: NodeId,
    two: NodeId,
) -> Result<NodeId, TensorError> {
    let inject_scaled = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(inject, "sh->sh"), (inv_hc, "->sh")],
    )?;
    let inject_sigmoid = sigmoid(program, inject_scaled, one, "sh->sh")?;
    let weight = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(inject_sigmoid, "sh->sh"), (two, "->sh")],
    )?;

    let broadcast_out = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(block_out, "si->shi"), (weight, "sh->shi")],
    )?;
    elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(residual, "shi->shi"), (broadcast_out, "shi->shi")],
    )
}
