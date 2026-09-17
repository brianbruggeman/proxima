use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn push_packed_row_blocked_body(
    source: &mut String,
    resolved: &BoundOp,
    reduce_op: ScalarOp,
    init: ReduceInit,
    output_axes: &[u16],
    rank: usize,
    quantized: &[Option<PackedCodec>],
    element_type: &str,
    block: &PackedRowBlock,
    epilogue_body: &ComposedBody,
    epilogue_operands: &[(NodeId, Layout, Option<Lookup>)],
    expert_source_mode: bool,
) -> Result<(), EmitError> {
    if packed_row_block_token_total(block, &resolved.extents) > 1 {
        push_packed_row_multi_row_body(
            source,
            resolved,
            reduce_op,
            init,
            rank,
            quantized,
            element_type,
            block,
            epilogue_body,
            epilogue_operands,
            expert_source_mode,
        )?;
        return Ok(());
    }
    let PackedRowBlock {
        weight,
        other,
        reduce_dim,
        codec,
        ..
    } = *block;
    let block_bytes = codec.block_bytes();
    let rank_len = rank.max(1);
    let operand_count = resolved.operands().len();
    // seeded on lane 0 only, exactly as the general cooperative path does:
    // the true seed folds in once and every other lane starts at the
    // algebraic identity, so `simd_*` can combine them unconditionally.
    let (init_expr, _) = fold_init_tokens(init);
    let identity = cooperative_identity_token(resolved.node, reduce_op)?;
    // ROW-BLOCKED PACKED PATH. One SIMD group folds `codec.rows_per_
    // simdgroup()` output rows at once so the activation's run of 8 values
    // is loaded into registers ONCE and reused across all of them — ggml's
    // `float sumf[nr0]` (`N_R0_Q4_K 4`, `N_R0_Q6_K 1`). Combined with the
    // super-block header amortization below, the per-element cost becomes
    // one byte load, one mask, one fma (`docs/discipline.md` ROW 74).
    {
        let run = Q4K_BLOCK_ELEMENTS / SIMD_WIDTH as usize;
        let rows = codec.rows_per_simdgroup();
        source.push_str(&format!("    long group_first = output_index * {rows};\n"));
        source.push_str(&format!("    {element_type} sumf[{rows}];\n"));
        // when the seed and the algebraic identity are the textually same
        // token (true for every op this path actually reaches: Add/Zero,
        // Multiply/One, Maximum/NegativeInfinity, Minimum/PositiveInfinity),
        // every lane's `sumf[q]` starts at the same value regardless of
        // which lane it is -- the per-lane ternary is dead, a select over
        // two identical constants, so emit a plain assignment instead.
        if init_expr == identity {
            source.push_str(&format!(
                "    for (int q = 0; q < {rows}; ++q) {{ sumf[q] = {identity}; }}\n"
            ));
        } else if cfg!(feature = "metal-q4k-split-k") {
            // the true seed folds in exactly ONCE across the WHOLE
            // threadgroup, not once per simdgroup -- at split == 1 `sgitg`
            // is always `0`, so this collapses to the feature-off condition
            // (`lane == 0u`) exactly.
            source.push_str(&format!(
                "    for (int q = 0; q < {rows}; ++q) {{ sumf[q] = (lane == 0u && sgitg == 0u) ? ({init_expr}) : ({identity}); }}\n"
            ));
        } else {
            source.push_str(&format!(
                "    for (int q = 0; q < {rows}; ++q) {{ sumf[q] = (lane == 0u) ? ({init_expr}) : ({identity}); }}\n"
            ));
        }
        source.push_str(&format!("    long weight_base[{rows}];\n"));
        source.push_str(&format!("    long other_base[{rows}];\n"));
        // `coord_q_cache[q]` -- kept past this loop, not a scratch local --
        // is the SAME `flat = group_first + q` decode
        // `push_packed_row_combine_and_write`'s epilogue needs for the
        // output-write coordinate; caching it here instead of re-running the
        // `%`/`/` chain against `u.output_extents` a second time in the
        // epilogue is the fix for the ladder-vs-production kernel-body gap
        // this op's own diff found (`docs/bench-campaigns/2026-09-03-gpu-one-
        // risc/design-2026-09-04/kernel-body-diff.md`): the ladder's hand-
        // written dispatch never had a second uniform-driven coordinate
        // decode to pay, because it never had a first one either.
        source.push_str(&format!("    long coord_q_cache[{rows}][{rank_len}];\n"));
        push_packed_row_group_bases(source, resolved, output_axes, rank, rows, weight, other);
        // STRIDE-FREE SPECIALIZATION (`docs/discipline.md` perf/packed-row-
        // addressing row): ggml's own row-blocked kernel assumes a
        // contiguous activation and addresses it with pure element offsets
        // (`ggml-metal.metal:5132`'s `y4 += 4*QK_K`, no per-element stride
        // multiply anywhere). This op's activation is not always
        // contiguous along the reduce axis, but `resolved`'s `Layout`
        // already carries the answer here at EMIT time -- this function
        // renders one op's kernel text once, not once per dispatch -- so
        // every activation address below drops the runtime `other_stride`
        // multiply entirely in the source text whenever the layout proves
        // it would multiply by 1.
        let other_stride_is_one = resolved.operands()[other].1.stride(reduce_dim as u16) == 1;
        source.push_str(&format!(
            "    long other_stride = u.operand_strides[{other}][{reduce_dim}];\n"
        ));
        // LANE SPREAD, ggml's `ix = tiisg/8`. Putting all 32 lanes on ONE
        // super-block gives each lane 8 of its 256 elements, so the header
        // decode is amortized over 8. Putting EIGHT lanes on a super-block
        // and letting the 32 lanes span FOUR at once gives each lane a whole
        // 32-element sub-block per decode — 4x the amortization, and the
        // sub-block is exactly the granularity the header is constant over.
        //
        // `it` selects the sub-block, so `slot = it * 32` and every one of
        // that lane's 32 elements shares a group and a nibble half. Levels
        // are still pulled 8 at a time (`q4k_run8`) rather than 32, to keep
        // the live register count near ggml's `yl[16]+yh[16]+sumf[4]`.
        // eight lanes per super-block (ggml's `tiisg/8`), so the 32 lanes of
        // a SIMD group span four super-blocks and each lane owns exactly one
        // 32-element sub-block — the granularity the header is constant over.
        let lanes_per_block = 8;
        let sub = Q4K_BLOCK_ELEMENTS / lanes_per_block;
        // Structural, not feature-gated: the paired-nibble/paired-lane body
        // applies to any codec whose block layout has one
        // ([`PackedCodec::supports_pair_dot`]) when the dtype is real
        // `Float32` -- a `DType` match, not the `element_type == "float"`
        // MSL-type-token comparison this replaced, which also admitted
        // `Int32`/`UInt32`/`Bool`/`Int8`/`UInt8` (every dtype `type_token`
        // happens to lower to the same MSL `float` storage type) and would
        // have run the scale/minimum float algebra below on integer data.
        let plain_product = !expert_source_mode
            && codec.supports_pair_dot()
            && resolved.dtype == DType::Float32
            && is_plain_product_reduce(resolved, reduce_op, weight, other);
        // `metal-q4k-single-fetch` (default-off): eliminates the redundant
        // paired-lane load the default `Q4_K` lane assignment below makes --
        // see `push_q4k_single_fetch_body`'s own doc. Checked after
        // `plain_product` (`q4k_pair_dot`'s float-only arm keeps priority --
        // neither is a self-contained kernel body the way this one is, so
        // `q4k_pair_dot` stays the fastest-known path where it applies) and
        // takes priority over everything else below it (the scale-deferred/
        // mask-fma arm, the per-element fallback): `push_q4k_single_fetch_body`
        // is fully self-contained (its own dispatch loop, not routed through
        // the lane-spread preamble this `else` arm builds), so it replaces
        // that whole preamble+match rather than plugging into one arm of it.
        // ALSO requires `metal-q4k-split-k` off: `emit`'s own dispatch-geometry
        // setup (`kernel_dispatch_shape`) computes `sgitg`/`split` purely off
        // that feature flag, unconditionally, for every row-blocked op --
        // `push_q4k_single_fetch_body`'s `ib` loop has no `sgitg`/`split`
        // awareness of its own (measured: under `--all-features` its sum came
        // out ~7x too large, consistent with every one of `split` simdgroups
        // redundantly summing the SAME full reduction instead of a disjoint
        // 1/split slice). Correct fix, not a silent one: `plain_product`/the
        // default/mask-fma `else` arm are already split-K-aware (they read
        // `sgitg`/`split` when the feature is on), so gating single-fetch off
        // here just means split-K wins whenever BOTH features are compiled
        // in, same "not invented to compose" posture as its other arms.
        let use_single_fetch = matches!(codec, PackedCodec::Q4K)
            && !plain_product
            && cfg!(feature = "metal-q4k-single-fetch")
            && !cfg!(feature = "metal-q4k-split-k");
        // `metal-q4k-ggml-port` (default-off): the verbatim ggml transcription,
        // see [`push_q4k_ggml_port_body`]/[`push_q6k_ggml_port_body`]'s own
        // docs. Requires `plain_product` (float-only scale-deferred shape,
        // the same gate `q4k_pair_dot`'s own arm below needs) and takes
        // priority over it -- both target the exact same design point, and
        // when this feature is on it is the one under test. Not
        // `metal-q4k-split-k`-aware, same posture as `push_q4k_single_fetch_body`
        // above and for the identical reason: its `ib` loop has no
        // `sgitg`/`split` stride of its own. ALSO requires the codec be one
        // this file has a verbatim ggml body for (`Q4K`, `Q5K`, `Q6K`): each
        // of `push_q4k_ggml_port_body`/`push_q5k_ggml_port_body`/
        // `push_q6k_ggml_port_body` transcribes a SPECIFIC ggml kernel's own
        // fixed byte offsets and decode shape. `plain_product` alone is not
        // codec-specific (every codec with `supports_pair_dot` reaches this
        // arm) -- without this guard, enabling `metal-q4k-ggml-port` silently
        // routed every packed-row `Q5_K` matmul through the `Q4_K`-shaped
        // body too (found via
        // `metal_matmul_on_packed_q5k_weights_matches_the_dequantized_f32_cpu_path`,
        // relative=0.977, essentially uncorrelated output -- the qh plane was
        // simply never read). `Q5_K` now has its own body
        // ([`push_q5k_ggml_port_body`]) instead of falling through to `Q4_K`'s.
        let use_ggml_port = plain_product
            && matches!(
                codec,
                PackedCodec::Q4K | PackedCodec::Q5K | PackedCodec::Q6K
            )
            && cfg!(feature = "metal-q4k-ggml-port")
            && !cfg!(feature = "metal-q4k-split-k");
        if use_ggml_port && matches!(codec, PackedCodec::Q6K) {
            push_q6k_ggml_port_body(
                source,
                weight,
                other,
                rows,
                block_bytes,
                other_stride_is_one,
            );
        } else if use_ggml_port && matches!(codec, PackedCodec::Q5K) {
            push_q5k_ggml_port_body(
                source,
                weight,
                other,
                rows,
                block_bytes,
                other_stride_is_one,
            );
        } else if use_ggml_port {
            push_q4k_ggml_port_body(
                source,
                weight,
                other,
                rows,
                block_bytes,
                other_stride_is_one,
            );
        } else if use_single_fetch {
            push_q4k_single_fetch_body(
                source,
                resolved,
                reduce_op,
                weight,
                other,
                element_type,
                operand_count,
                rows,
                block_bytes,
            );
        } else {
            source.push_str(&format!("    uint ix = (uint)lane / {lanes_per_block}u;\n"));
            source.push_str(&format!("    uint it = (uint)lane % {lanes_per_block}u;\n"));
            source.push_str(&format!("    uint slot = it * {sub}u;\n"));
            if plain_product {
                source.push_str(
                    "    uint iq = it / 4u; uint ir = it % 4u;\n    float yl[16]; float yh[16];\n",
                );
            } else {
                source.push_str(&format!("    {element_type} acts[{sub}];\n"));
            }
            source.push_str(&format!(
                "    int super_blocks = (int)u.reduction_total / {Q4K_BLOCK_ELEMENTS};\n"
            ));
            let ix_stride = SIMD_WIDTH as usize / lanes_per_block;
            if cfg!(feature = "metal-q4k-split-k") {
                // each simdgroup (`sgitg`, 0 at split == 1) owns a disjoint
                // interleaved slice of super-blocks -- a plain strided loop, so a
                // `super_blocks` not evenly divisible by `split` is handled by
                // construction (some simdgroups simply run one fewer iteration),
                // never a separate ragged-tail branch.
                source.push_str(&format!(
                "    int ib_first = (int)(ix + sgitg * {ix_stride}u);\n    int ib_step = (int)({ix_stride}u * split);\n"
            ));
            } else {
                source.push_str(&format!(
                    "    int ib_first = (int)ix;\n    int ib_step = {ix_stride};\n"
                ));
            }
            // HOIST + POINTER INCREMENT (`docs/discipline.md` perf/packed-row-
            // addressing row): `weight_base[q]/Q4K_BLOCK_ELEMENTS` and the y4
            // lane offset (`64*iq + 8*ir`) are invariant across every `ib` this
            // thread visits -- only `ib` itself varies. The prior form
            // recomputed `(weight_base[q]/256 + ib) * block_bytes` and
            // `ib*256*other_stride` from scratch every iteration (a 64-bit
            // multiply-add per row per iteration); ggml's own row/`y4` pointers
            // instead advance by a CONSTANT per iteration
            // (`ggml-metal.metal:5132,5182`'s `q1 += nb01/2`, `y4 += 4*QK_K`).
            // This computes each row's starting byte pointer and the
            // per-iteration byte step ONCE before the loop, then the loop body
            // only adds.
            source.push_str(&format!(
                "    long blk_step = (long)ib_step * {block_bytes};\n"
            ));
            source.push_str(&format!("    device const uchar *blk_ptr[{rows}];\n"));
            source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
            source.push_str(&format!(
            "        blk_ptr[q] = in{weight} + ((long)((int)weight_base[q] / {Q4K_BLOCK_ELEMENTS}) + (long)ib_first) * {block_bytes};\n"
        ));
            source.push_str("    }\n");
            if plain_product {
                push_q4k_plain_product_y4_address(source, other, other_stride_is_one);
            } else if other_stride_is_one {
                // SAME HOIST, generic (non-plain-product) arm: `elem0 = ib*256 +
                // slot` was rebuilt every `ib` purely to feed `(elem0+j)*
                // other_stride` -- the identical 64-bit multiply-add-per-
                // iteration shape arm1 already removed from the plain-product
                // `y4` pointer above. `other_stride_is_one` additionally drops
                // the multiply itself, same as the `y4` arm just above.
                source.push_str(&format!(
                    "    long acts_step = (long)ib_step * {Q4K_BLOCK_ELEMENTS};\n"
                ));
                source.push_str(&format!("    device const {element_type} *acts_row = in{other} + other_base[0] + (long)ib_first * {Q4K_BLOCK_ELEMENTS} + (long)slot;\n"));
            } else {
                source.push_str(&format!(
                    "    long acts_step = (long)ib_step * {Q4K_BLOCK_ELEMENTS} * other_stride;\n"
                ));
                source.push_str(&format!("    device const {element_type} *acts_row = in{other} + other_base[0] + (long)ib_first * {Q4K_BLOCK_ELEMENTS} * other_stride + (long)slot * other_stride;\n"));
            }
            source.push_str("    for (int ib = ib_first; ib < super_blocks; ib += ib_step) {\n");
            if plain_product && other_stride_is_one {
                source.push_str("        for (uint i = 0u; i < 8u; ++i) { yl[i] = y4[i]; yl[i + 8u] = y4[i + 32u]; yh[i] = y4[i + 128u]; yh[i + 8u] = y4[i + 160u]; }\n");
            } else if plain_product {
                // CORRECTNESS FIX, not part of the stride-free specialization
                // above: this arm's `y4[i]`/`y4[i+32]`/... reads were pure
                // element offsets regardless of `other_stride` before this
                // landing -- correct only by accident, for every caller that
                // happened to hand this path a contiguous activation. Ported
                // from `push_q4k_ggml_port_body`'s own already-stride-aware
                // form (`y4_base + (long)i * other_stride`, below in this
                // file), the one sibling body that already got this right.
                source.push_str("        for (uint i = 0u; i < 8u; ++i) { yl[i] = y4[(long)i * other_stride]; yl[i + 8u] = y4[(long)(i + 32u) * other_stride]; yh[i] = y4[(long)(i + 128u) * other_stride]; yh[i + 8u] = y4[(long)(i + 160u) * other_stride]; }\n");
            } else {
                source.push_str(&format!("        for (int j = 0; j < {sub}; ++j) {{\n"));
                if other_stride_is_one {
                    source.push_str("            acts[j] = acts_row[j];\n");
                } else {
                    source.push_str("            acts[j] = acts_row[(long)j * other_stride];\n");
                }
                source.push_str("        }\n");
            }
            source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{\n"));
            source.push_str("            device const uchar *blk = blk_ptr[q];\n");
            match codec {
                PackedCodec::Q2K => {
                    source.push_str(&format!("            for (int e = 0; e < {sub}; ++e) {{\n"));
                    source.push_str(&format!(
                        "                {element_type} scratch[{}];\n",
                        operand_count.max(1)
                    ));
                    source.push_str(&format!(
                        "                scratch[{weight}] = q2k_element(blk, slot + (uint)e);\n"
                    ));
                    source.push_str(&format!("                scratch[{other}] = acts[e];\n"));
                    let value_expr = push_body_steps(
                        source,
                        resolved.element_body(),
                        "                ",
                        element_type,
                    );
                    source.push_str(&format!(
                        "                {element_type} value = {value_expr};\n"
                    ));
                    let combine_expr = scalar_op_expr(reduce_op, &["sumf[q]", "value"]);
                    source.push_str(&format!("                sumf[q] = {combine_expr};\n"));
                    source.push_str("            }\n");
                }
                PackedCodec::Q3K if plain_product => {
                    // `plain_product` is codec-agnostic (the `yl`/`yh` gather
                    // above is built once, shared by `Q4_K`/`Q3_K` and, when
                    // `metal-q5k-pair-dot` is on, `Q5_K` too) -- no separate
                    // activation load path needed here, same posture as
                    // `Q5_K`'s own `plain_product` arm.
                    source.push_str(
                        "            sumf[q] = sumf[q] + q3k_pair_dot(blk, iq, ir, yl, yh);\n",
                    );
                }
                PackedCodec::Q3K => {
                    // `Q3_K`'s sub-block width (16) is narrower than this
                    // loop's 32-element `sub` slot, unlike `Q5_K`'s matching
                    // 32-element sub-block -- amortizing one header decode
                    // across the whole slot the way the `Q5_K` arm below does
                    // would silently span two different sub-block scales. Each
                    // call to `q3k_element` decodes its own header, the same
                    // posture `Q6_K`'s per-element path takes for a different
                    // reason (its scale bytes are plain, not bit-packed, so the
                    // per-call cost is small either way). A follow-up
                    // optimization (a two-headers-per-slot amortization), not a
                    // correctness gap.
                    source.push_str(&format!("            for (int e = 0; e < {sub}; ++e) {{\n"));
                    source.push_str(&format!(
                        "                {element_type} scratch[{}];\n",
                        operand_count.max(1)
                    ));
                    source.push_str(&format!(
                        "                scratch[{weight}] = q3k_element(blk, slot + (uint)e);\n"
                    ));
                    source.push_str(&format!("                scratch[{other}] = acts[e];\n"));
                    let value_expr = push_body_steps(
                        source,
                        resolved.element_body(),
                        "                ",
                        element_type,
                    );
                    source.push_str(&format!(
                        "                {element_type} value = {value_expr};\n"
                    ));
                    let combine_expr = scalar_op_expr(reduce_op, &["sumf[q]", "value"]);
                    source.push_str(&format!("                sumf[q] = {combine_expr};\n"));
                    source.push_str("            }\n");
                }
                PackedCodec::Q4K => {
                    if plain_product {
                        source.push_str(
                            "            sumf[q] = sumf[q] + q4k_pair_dot(blk, iq, ir, yl, yh);\n",
                        );
                    } else if is_plain_product_reduce(resolved, reduce_op, weight, other) {
                        // SCALE-DEFERRED PATH (`docs/discipline.md` ROW 106).
                        // Accumulate the raw nibble x activation product and the
                        // activation sum UNSCALED across the whole sub-block, then
                        // apply `hdr.scale`/`hdr.minimum` ONCE at the end instead
                        // of once per element — legal here because
                        // `is_plain_product_reduce` already proved reduce_op is
                        // `Add` and the body is exactly `weight * other`, so
                        // `sum_j (scale*nibble_j - min)*act_j == scale*sum(nibble_j
                        // *act_j) - min*sum(act_j)`. Mirrors
                        // `ggml-metal.metal:5157-5175`'s `acc1`/`dall` split.
                        // Two bodies behind `metal-q4k-mask-fma`: off, the
                        // shift-then-mask `q4k_run8` extraction into a `dot`
                        // reduce; on, ggml's actual mask-without-shift technique,
                        // fused with the accumulate -- see
                        // `push_q4k_product_reduce_body`'s own doc.
                        push_q4k_header_decode(source);
                        push_q4k_product_reduce_body(source, sub, run, element_type);
                    } else {
                        source.push_str(&format!(
                            "            for (int c = 0; c < {}; ++c) {{\n",
                            sub / run
                        ));
                        // raw 4-bit levels (0..15) are exact in float regardless of
                        // the kernel's element type; q4k_run8 takes `thread float
                        // *out`, and the narrowing to element_type happens where
                        // levels combine into scratch below, same as every other
                        // operand read.
                        source.push_str(&format!("                float levels[{run}];\n"));
                        source.push_str(&format!(
                            "                q4k_run8(blk, slot + (uint)(c * {run}), levels);\n"
                        ));
                        source.push_str(&format!(
                            "                for (int j = 0; j < {run}; ++j) {{\n"
                        ));
                        source.push_str(&format!(
                            "                    {element_type} scratch[{}];\n",
                            operand_count.max(1)
                        ));
                        source.push_str(&format!(
                        "                    scratch[{weight}] = hdr.scale * levels[j] - hdr.minimum;\n"
                    ));
                        source.push_str(&format!(
                            "                    scratch[{other}] = acts[c * {run} + j];\n"
                        ));
                        let value_expr = push_body_steps(
                            source,
                            resolved.element_body(),
                            "                    ",
                            element_type,
                        );
                        source.push_str(&format!(
                            "                    {element_type} value = {value_expr};\n"
                        ));
                        let combine_expr = scalar_op_expr(reduce_op, &["sumf[q]", "value"]);
                        source
                            .push_str(&format!("                    sumf[q] = {combine_expr};\n"));
                        source.push_str("                }\n");
                        source.push_str("            }\n");
                    }
                }
                PackedCodec::Q5K if plain_product => {
                    // See [`Q5K_PAIR_DOT_MSL`]: the same `yl`/`yh` two-word-load pairing `Q4_K`'s own
                    // `plain_product` arm above uses, extended with `Q5_K`'s `qh`
                    // high-bit plane. Reads the SAME `yl`/`yh` activation gather
                    // this preamble already built for `Q4_K` (`plain_product`
                    // is codec-agnostic there), so no separate activation load
                    // path is needed for this codec.
                    source.push_str(
                        "            sumf[q] = sumf[q] + q5k_pair_dot(blk, iq, ir, yl, yh);\n",
                    );
                }
                PackedCodec::Q5K => {
                    // No `q5k_run8`-style batched unpack yet — `Q5_K`'s `qh`
                    // high-bit plane means each element needs a `qs` nibble AND
                    // a `qh` bit from a DIFFERENT byte, the same shape gap
                    // `Q6_K`'s own arm below documents. `d` and this sub-block's
                    // scale/min/mask ARE decoded once per 32-element run via
                    // `q5k_header_for` (the same granularity `q4k_header_for`
                    // amortizes over) — a follow-up optimization, not a
                    // correctness gap; see this landing's discipline row (ROW
                    // 92) for the measured cost of skipping it. The `plain_product`
                    // arm above replaces this whole per-element loop with the
                    // paired-nibble body whenever the reduce is a plain product.
                    source.push_str("            q5k_header hdr = q5k_header_for(blk, slot);\n");
                    source.push_str(&format!("            for (int e = 0; e < {sub}; ++e) {{\n"));
                    source.push_str(&format!(
                        "                {element_type} scratch[{}];\n",
                        operand_count.max(1)
                    ));
                    source.push_str(&format!(
                        "                scratch[{weight}] = q5k_value(blk, slot + (uint)e, hdr);\n"
                    ));
                    source.push_str(&format!("                scratch[{other}] = acts[e];\n"));
                    let value_expr = push_body_steps(
                        source,
                        resolved.element_body(),
                        "                ",
                        element_type,
                    );
                    source.push_str(&format!(
                        "                {element_type} value = {value_expr};\n"
                    ));
                    let combine_expr = scalar_op_expr(reduce_op, &["sumf[q]", "value"]);
                    source.push_str(&format!("                sumf[q] = {combine_expr};\n"));
                    source.push_str("            }\n");
                }
                PackedCodec::Q6K if plain_product => {
                    // See [`Q6K_PAIR_DOT_MSL`]: the same paired-lane body `Q4_K`/`Q5_K`'s own
                    // `plain_product` arms use above, ported to `Q6_K`'s
                    // ql/qh/signed-scale layout. Reads the SAME `yl`/`yh`
                    // activation gather this preamble already built for
                    // `Q4_K` (`plain_product` is codec-agnostic there), so no
                    // separate activation load path is needed for this codec.
                    source.push_str(
                        "            sumf[q] = sumf[q] + q6k_pair_dot(blk, iq, ir, yl, yh);\n",
                    );
                }
                PackedCodec::Q6K => {
                    // No `q6k_run8`-style batched unpack yet — `Q6_K`'s bit
                    // layout does not reduce to two word loads the way `Q4_K`'s
                    // does (each element needs a `ql` byte, a `qh` byte, AND a
                    // sub-block scale byte, not one nibble out of an
                    // already-loaded word). Correct, one element at a time; `d`
                    // is still decoded ONCE per super-block via
                    // `q6k_header_for` rather than per element. The `plain_product`
                    // arm above replaces this whole per-element loop with the
                    // paired-lane body whenever the reduce is a plain product.
                    source.push_str("            q6k_header hdr = q6k_header_for(blk);\n");
                    source.push_str(&format!("            for (int e = 0; e < {sub}; ++e) {{\n"));
                    source.push_str(&format!(
                        "                {element_type} scratch[{}];\n",
                        operand_count.max(1)
                    ));
                    source.push_str(&format!(
                        "                scratch[{weight}] = q6k_value(blk, slot + (uint)e, hdr);\n"
                    ));
                    source.push_str(&format!("                scratch[{other}] = acts[e];\n"));
                    let value_expr = push_body_steps(
                        source,
                        resolved.element_body(),
                        "                ",
                        element_type,
                    );
                    source.push_str(&format!(
                        "                {element_type} value = {value_expr};\n"
                    ));
                    let combine_expr = scalar_op_expr(reduce_op, &["sumf[q]", "value"]);
                    source.push_str(&format!("                sumf[q] = {combine_expr};\n"));
                    source.push_str("            }\n");
                }
                PackedCodec::Q8_0 => {
                    return Err(EmitError::NonKQuantPackedCodec {
                        node: resolved.node,
                        codec: "q8_0",
                    });
                }
                PackedCodec::Q4_0 => {
                    return Err(EmitError::NonKQuantPackedCodec {
                        node: resolved.node,
                        codec: "q4_0",
                    });
                }
                PackedCodec::Q5_1 => {
                    return Err(EmitError::NonKQuantPackedCodec {
                        node: resolved.node,
                        codec: "q5_1",
                    });
                }
                PackedCodec::Float16 => {
                    return Err(EmitError::NonKQuantPackedCodec {
                        node: resolved.node,
                        codec: "float16",
                    });
                }
                PackedCodec::BFloat16 => {
                    return Err(EmitError::NonKQuantPackedCodec {
                        node: resolved.node,
                        codec: "bfloat16",
                    });
                }
            }
            source.push_str("        }\n");
            source.push_str(&format!(
                "        for (int q = 0; q < {rows}; ++q) {{ blk_ptr[q] += blk_step; }}\n"
            ));
            if plain_product {
                source.push_str("        y4 += y4_step;\n");
            } else {
                source.push_str("        acts_row += acts_step;\n");
            }
            source.push_str("    }\n");
        }
        push_packed_row_combine_and_write(
            source,
            resolved.node,
            reduce_op,
            rows,
            rank,
            output_axes,
            element_type,
            epilogue_body,
            epilogue_operands,
        )?;
    }
    Ok(())
}

/// `metal-q4k-single-fetch` (default-off): the row-blocked packed-`Q4_K`
/// path's `it`/`slot` lane assignment, above, gives lanes `2r` and `2r+1`
/// (`r` = one of the super-block's 4 64-element groups) the SAME 32-byte
/// `qs` range — `q4k_run8` loads it twice, once per lane, differing only in
/// which nibble half each keeps (`shift` 0 vs 4). This function replaces
/// that pairing: `sf_half` (still `it % 2`) now selects a DISTINCT 16-byte
/// half of the group's 32 bytes, and `q4k_run8_dual`
/// extracts BOTH nibble halves from each byte it loads, so the pair's two
/// lanes together read the group's 32 bytes exactly once instead of twice.
///
/// Dispatch geometry is untouched: `ix = lane/8` and the `ib += 4` stride
/// are identical to the duplicate-fetch path, so this is a change to which
/// BYTES a lane owns and what it does with them, not to thread count,
/// simdgroups-per-threadgroup, or `PACKED_ROWS_PER_GROUP`. Not `split-K`
/// aware -- this function owns its own complete dispatch loop rather than
/// plugging into the shared lane-spread preamble `metal-q4k-split-k`
/// modifies, so the two features do not compose (see this feature's own
/// Cargo.toml doc).
///
/// Correctness hazard this function exists to get right: a `qs` byte's low
/// nibble and high nibble belong to DIFFERENT 32-element sub-blocks with
/// DIFFERENT 6-bit `(scale, min)` pairs (`q4_k.rs::dequantize_block`'s own
/// doc — elements land "32 output elements apart", not adjacent). A lane
/// that decodes both nibbles of a byte therefore needs BOTH sub-blocks'
/// headers (`hdr_low`/`hdr_high`), never one. `q4k_header_for(blk,
/// sf_low_base)` and `q4k_header_for(blk, sf_high_base)` resolve to the
/// same two sub-block indices for `sf_half == 0` and `sf_half == 1` alike
/// (`sf_low_base % 64` is `0` or `16`, both `< 32`; `sf_high_base % 64` is
/// `32` or `48`, both `>= 32`), so both this lane's low-half partial sum and
/// its pair-partner's low-half partial sum are scaled by the IDENTICAL
/// `hdr_low`, and summing them via `simd_sum` after the `ib` loop
/// reconstructs the same per-sub-block total the duplicate-fetch path
/// computes — see this function's own algebra note on the scale-deferred
/// arm below.
#[allow(clippy::too_many_arguments)]
pub(super) fn push_q4k_single_fetch_body(
    source: &mut String,
    resolved: &BoundOp,
    reduce_op: ScalarOp,
    weight: usize,
    other: usize,
    element_type: &str,
    operand_count: usize,
    rows: usize,
    block_bytes: usize,
) {
    source.push_str("    uint ix = (uint)lane / 8u;\n");
    source.push_str("    uint it = (uint)lane % 8u;\n");
    source.push_str("    uint sf_region = it / 2u;\n");
    source.push_str("    uint sf_half = it % 2u;\n");
    source.push_str("    uint sf_low_base = sf_region * 64u + sf_half * 16u;\n");
    source.push_str("    uint sf_high_base = sf_low_base + 32u;\n");
    source.push_str(&format!(
        "    int super_blocks = (int)u.reduction_total / {Q4K_BLOCK_ELEMENTS};\n"
    ));
    source.push_str("    for (int ib = (int)ix; ib < super_blocks; ib += 4) {\n");
    source.push_str("        int elem0_low = ib * 256 + (int)sf_low_base;\n");
    source.push_str("        int elem0_high = ib * 256 + (int)sf_high_base;\n");
    source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&format!(
        "            device const uchar *blk = in{weight} + ((int)weight_base[q] / {Q4K_BLOCK_ELEMENTS} + ib) * {block_bytes};\n"
    ));
    source.push_str("            q4k_header hdr_low = q4k_header_for(blk, sf_low_base);\n");
    source.push_str("            q4k_header hdr_high = q4k_header_for(blk, sf_high_base);\n");
    if is_plain_product_reduce(resolved, reduce_op, weight, other) {
        // SCALE-DEFERRED, split across TWO sub-blocks instead of one: this
        // lane covers 16 of sub-block-A's 32 elements (`raw_low`/`act_low`)
        // and 16 of sub-block-B's 32 (`raw_high`/`act_high`) — its pair
        // partner (`sf_half` flipped, same `sf_region`) covers the other 16
        // of each. `sum_j (scale*nibble_j - min)*act_j == scale*sum(nibble_j
        // *act_j) - min*sum(act_j)` (the same identity
        // `is_plain_product_reduce`'s caller already proved licenses) holds
        // per HALF exactly as it holds per whole sub-block, and addition
        // distributes over the two halves, so `simd_sum` over both lanes in
        // a pair reconstructs the identical two sub-block totals the
        // duplicate-fetch path computes in one lane each.
        source.push_str(&format!("            {element_type} raw_low = 0;\n"));
        source.push_str(&format!("            {element_type} act_low = 0;\n"));
        source.push_str(&format!("            {element_type} raw_high = 0;\n"));
        source.push_str(&format!("            {element_type} act_high = 0;\n"));
        source.push_str("            for (int c = 0; c < 2; ++c) {\n");
        source.push_str("                float low_levels[8];\n");
        source.push_str("                float high_levels[8];\n");
        source.push_str(
            "                q4k_run8_dual(blk, sf_low_base + (uint)(c * 8), low_levels, high_levels);\n",
        );
        source.push_str("                for (int j = 0; j < 8; ++j) {\n");
        source.push_str(&format!(
            "                    {element_type} act_l = in{other}[other_base[0] + (long)(elem0_low + c * 8 + j) * other_stride];\n"
        ));
        source.push_str(&format!(
            "                    {element_type} act_h = in{other}[other_base[0] + (long)(elem0_high + c * 8 + j) * other_stride];\n"
        ));
        source.push_str("                    raw_low += low_levels[j] * act_l;\n");
        source.push_str("                    act_low += act_l;\n");
        source.push_str("                    raw_high += high_levels[j] * act_h;\n");
        source.push_str("                    act_high += act_h;\n");
        source.push_str("                }\n");
        source.push_str("            }\n");
        source.push_str(
            "            sumf[q] = sumf[q] + hdr_low.scale * raw_low - hdr_low.minimum * act_low + hdr_high.scale * raw_high - hdr_high.minimum * act_high;\n",
        );
    } else {
        source.push_str("            for (int c = 0; c < 2; ++c) {\n");
        source.push_str("                float low_levels[8];\n");
        source.push_str("                float high_levels[8];\n");
        source.push_str(
            "                q4k_run8_dual(blk, sf_low_base + (uint)(c * 8), low_levels, high_levels);\n",
        );
        source.push_str("                for (int j = 0; j < 8; ++j) {\n");
        source.push_str(&format!(
            "                    {element_type} scratch[{}];\n",
            operand_count.max(1)
        ));
        source.push_str(&format!(
            "                    scratch[{weight}] = hdr_low.scale * low_levels[j] - hdr_low.minimum;\n"
        ));
        source.push_str(&format!(
            "                    scratch[{other}] = in{other}[other_base[0] + (long)(elem0_low + c * 8 + j) * other_stride];\n"
        ));
        let low_value_expr = push_body_steps(
            source,
            resolved.element_body(),
            "                    ",
            element_type,
        );
        source.push_str(&format!(
            "                    {element_type} value = {low_value_expr};\n"
        ));
        let low_combine_expr = scalar_op_expr(reduce_op, &["sumf[q]", "value"]);
        source.push_str(&format!(
            "                    sumf[q] = {low_combine_expr};\n"
        ));
        source.push_str("                }\n");
        source.push_str("                for (int j = 0; j < 8; ++j) {\n");
        source.push_str(&format!(
            "                    {element_type} scratch[{}];\n",
            operand_count.max(1)
        ));
        source.push_str(&format!(
            "                    scratch[{weight}] = hdr_high.scale * high_levels[j] - hdr_high.minimum;\n"
        ));
        source.push_str(&format!(
            "                    scratch[{other}] = in{other}[other_base[0] + (long)(elem0_high + c * 8 + j) * other_stride];\n"
        ));
        let high_value_expr = push_body_steps(
            source,
            resolved.element_body(),
            "                    ",
            element_type,
        );
        source.push_str(&format!(
            "                    {element_type} value = {high_value_expr};\n"
        ));
        let high_combine_expr = scalar_op_expr(reduce_op, &["sumf[q]", "value"]);
        source.push_str(&format!(
            "                    sumf[q] = {high_combine_expr};\n"
        ));
        source.push_str("                }\n");
        source.push_str("            }\n");
    }
    source.push_str("        }\n");
    source.push_str("    }\n");
}

/// The plain-product row-blocked bodies' shared activation address: the
/// `y4` pointer's base and per-iteration byte step. `push_packed_row_blocked_
/// body`'s own `plain_product` arm and `push_q4k_ggml_port_body` both read
/// `weight_base`/`other_base`/`other_stride`/`iq`/`ir` from the same
/// preamble and must have already declared `ib_first`/`ib_step` in scope --
/// this is the ONE place either renders the pointer, so the stride-free
/// specialization (drop the runtime `other_stride` multiply when the layout
/// proves it is 1, see the caller's own `other_stride_is_one` doc) applies
/// identically to both instead of drifting.
pub(super) fn push_q4k_plain_product_y4_address(source: &mut String, other: usize, other_stride_is_one: bool) {
    push_packed_row_plain_product_y4_address(
        source,
        other,
        other_stride_is_one,
        "64u * iq + 8u * ir",
    );
}

/// [`push_q4k_plain_product_y4_address`] generalized to any ggml-port body's
/// own lane-derived activation offset -- [`push_q6k_ggml_port_body`] shares
/// this instead of a second copy of the address arithmetic, passing
/// `"128u * ip + l0"` (llama's `y_offset`, `ggml-metal.metal:5383`) in place
/// of `Q4_K`'s `"64u * iq + 8u * ir"`.
pub(super) fn push_packed_row_plain_product_y4_address(
    source: &mut String,
    other: usize,
    other_stride_is_one: bool,
    lane_offset: &str,
) {
    if other_stride_is_one {
        source.push_str(&format!(
            "    long y4_step = (long)ib_step * {Q4K_BLOCK_ELEMENTS};\n"
        ));
        source.push_str(&format!("    device const float *y4 = in{other} + other_base[0] + (long)ib_first * {Q4K_BLOCK_ELEMENTS} + (long)({lane_offset});\n"));
    } else {
        source.push_str(&format!(
            "    long y4_step = (long)ib_step * {Q4K_BLOCK_ELEMENTS} * other_stride;\n"
        ));
        source.push_str(&format!("    device const float *y4 = in{other} + other_base[0] + (long)ib_first * {Q4K_BLOCK_ELEMENTS} * other_stride + (long)({lane_offset}) * other_stride;\n"));
    }
}

// ggml (llama.cpp, MIT license: https://github.com/ggml-org/llama.cpp/blob/
// master/LICENSE) `kernel_mul_mv_q4_K_f32_impl<4,2,32>`
// (ggml-metal.metal:5086-5193), transcribed line-for-line onto this crate's
// operand-base/stride addressing. See `push_q4k_ggml_port_body`'s own doc for
// what is and is not identical to the upstream source.
//
// Copyright (c) 2023-2024 The ggml authors. MIT-licensed; see THIRD_PARTY.md.
//
/// `metal-q4k-ggml-port` (default-off): a VERBATIM port of ggml's
/// `kernel_mul_mv_q4_K_f32_impl<nr0=4, nsg=2, nw=32>`
/// (`ggml-metal.metal:5086-5193`) -- every prior landing on this path
/// (`q4k_pair_dot`'s `plain_product` arm above, `metal-q4k-mask-fma`,
/// `metal-q4k-single-fetch`) is this crate's own RE-DERIVATION of pieces of
/// ggml's technique through its `q4k_header`/`q4k_run8` abstractions; this
/// function instead transcribes ggml's actual per-thread math with no
/// intermediate abstraction, so the only remaining difference from upstream
/// is address computation (`weight_base[q]`/`other_base[0]`/`other_stride`,
/// this crate's per-axis strided reads, instead of ggml's raw `nb01` pointer
/// walk -- ggml's own `q1 += args.nb01/2` row-advance is behaviorally
/// identical to this function's per-row `blk` recompute for the contiguous
/// packed-row layout this crate always uses). The activation address itself
/// shares `push_q4k_plain_product_y4_address` with
/// `push_packed_row_blocked_body`'s own `plain_product` arm, so when the
/// layout proves the reduce-axis stride is 1 this body drops the runtime
/// multiply the same way ggml's raw pointer walk always did.
///
/// Per-thread split (ggml-metal.metal:5100-5103), unchanged from the
/// existing row-blocked preamble's own lane assignment: `ix = lane/8`
/// (0..3, which of 4 super-blocks in today's `ib` stride this lane owns),
/// `it = lane%8` (0..7), `iq = it/4` (0 or 1, selects `q1` vs `q2`'s 64-byte
/// `qs` half), `ir = it%4` (0..3, a 4-uint16 stride within that half).
///
/// Super-block iteration (ggml-metal.metal:5132,5182): `for (ib = ix; ib <
/// nb; ib += 4)`, `nb = reduction_total / 256`; `y4` (the activation gather
/// base) advances by `4 * QK_K` (1024) elements per iteration, matching
/// ggml's `y4 += 4 * QK_K`.
///
/// Scale/min extraction (ggml-metal.metal:5096-5098,5142-5150): three fixed
/// masks, `kmask1 = 0x3f3f` (two 6-bit scale/min fields), `kmask2 = 0x0f0f`
/// (two 4-bit high-scale/high-min fields), `kmask3 = 0xc0c0` (the two
/// leftover high bits of the LOW fields, shifted into place with `>> 2`) --
/// no shift-then-branch the way this file's own `q4k_scale_min` reads it.
/// `sc16[0..3]` (aliased as 8 bytes `sc8[0..7]`) hold, in order: low-group
/// low-half scale, low-group low-half min, low-group high-half scale,
/// low-group high-half min, high-group low-half scale, high-group low-half
/// min, high-group high-half scale, high-group high-half min.
///
/// Nibble extraction (ggml-metal.metal:5157-5166): FOUR fixed bit-position
/// masks off each raw `uint16_t` word -- `& 0x000F`, `& 0x0F00`, `& 0x00F0`,
/// `& 0xF000` -- no runtime shift at all. The `0x0F00`/`0xF000` masks leave
/// their nibble sitting at bit 8/12, so the corresponding accumulator lane
/// (`acc1[1]`/`acc1[3]`/`acc2[1]`/`acc2[3]`) is 256x too large; `0x00F0`
/// leaves its nibble at bit 4, 16x too large. Both residuals are folded into
/// the FINAL per-sub-block combine below (ggml-metal.metal:5171-5175)
/// rather than corrected per element -- `1.0f/256.0f` on the odd
/// accumulator lanes, `1.0f/16.0f` on the whole second scale/min term --
/// this is the "mask-without-shift" technique `metal-q4k-mask-fma`'s own doc
/// names but only ports for the header decode, not this accumulate.
///
/// SIMD reduction (ggml-metal.metal:5187-5192): unchanged from every other
/// row-blocked arm -- `simd_sum(sumf[row])` combines the 32 lanes of one
/// simdgroup, lane 0 alone writes -- handled by the shared
/// `push_packed_row_combine_and_write` tail this function's caller still
/// invokes after it returns.
///
/// Dispatch geometry: `nr0 = 4` is already this crate's own
/// `PACKED_ROWS_PER_GROUP`; `nsg = 2` is wired separately, in
/// `tiled_gemm_threadgroup_width`'s own `metal-q4k-ggml-port` arm.
#[allow(clippy::too_many_arguments)]
pub(super) fn push_q4k_ggml_port_body(
    source: &mut String,
    weight: usize,
    other: usize,
    rows: usize,
    block_bytes: usize,
    other_stride_is_one: bool,
) {
    source.push_str("    uint ix = (uint)lane / 8u;\n");
    source.push_str("    uint it = (uint)lane % 8u;\n");
    source.push_str("    uint iq = it / 4u;\n");
    source.push_str("    uint ir = it % 4u;\n");
    source.push_str(&format!(
        "    int super_blocks = (int)u.reduction_total / {Q4K_BLOCK_ELEMENTS};\n"
    ));
    source.push_str("    float yl[16];\n");
    source.push_str("    float yh[16];\n");
    // HOIST + POINTER INCREMENT, same as `push_packed_row_blocked_body`'s
    // own arm above and for the identical reason: `weight_base[q]/
    // Q4K_BLOCK_ELEMENTS` and the y4 lane offset are invariant across every
    // `ib` this thread visits (`ix` alone selects the starting super-block,
    // the loop always steps by the fixed `4`), so both this row's byte
    // pointer and the activation base are computed ONCE and advanced by a
    // constant per iteration instead of rebuilt from `ib` every time.
    // `ib_first`/`ib_step` (rather than `ix`/the literal `4`) are the same
    // names `push_packed_row_blocked_body`'s own arm declares, so
    // `push_q4k_plain_product_y4_address` renders identical text in both
    // callers.
    source.push_str("    int ib_first = (int)ix;\n    int ib_step = 4;\n");
    source.push_str(&format!(
        "    long blk_step = (long)ib_step * {block_bytes};\n"
    ));
    source.push_str(&format!("    device const uchar *blk_ptr[{rows}];\n"));
    source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&format!(
        "        blk_ptr[q] = in{weight} + ((long)((int)weight_base[q] / {Q4K_BLOCK_ELEMENTS}) + (long)ib_first) * {block_bytes};\n"
    ));
    source.push_str("    }\n");
    push_q4k_plain_product_y4_address(source, other, other_stride_is_one);
    source.push_str("    for (int ib = ib_first; ib < super_blocks; ib += ib_step) {\n");
    source.push_str(
        "        float sumy0 = 0.0f; float sumy1 = 0.0f; float sumy2 = 0.0f; float sumy3 = 0.0f;\n",
    );
    if other_stride_is_one {
        source.push_str(
            "        for (uint i = 0u; i < 8u; ++i) {\n            yl[i] = y4[i]; sumy0 += yl[i];\n",
        );
        source.push_str("            yl[i + 8u] = y4[i + 32u]; sumy1 += yl[i + 8u];\n");
        source.push_str("            yh[i] = y4[i + 128u]; sumy2 += yh[i];\n");
        source.push_str("            yh[i + 8u] = y4[i + 160u]; sumy3 += yh[i + 8u];\n");
    } else {
        source.push_str(
            "        for (uint i = 0u; i < 8u; ++i) {\n            yl[i] = y4[(long)i * other_stride]; sumy0 += yl[i];\n",
        );
        source.push_str(
            "            yl[i + 8u] = y4[(long)(i + 32u) * other_stride]; sumy1 += yl[i + 8u];\n",
        );
        source
            .push_str("            yh[i] = y4[(long)(i + 128u) * other_stride]; sumy2 += yh[i];\n");
        source.push_str(
            "            yh[i + 8u] = y4[(long)(i + 160u) * other_stride]; sumy3 += yh[i + 8u];\n",
        );
    }
    source.push_str("        }\n");
    source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str("            device const uchar *blk = blk_ptr[q];\n");
    source
        .push_str("            device const ushort *sc = (device const ushort *)(blk + 4) + iq;\n");
    source.push_str("            device const ushort *q1 = (device const ushort *)(blk + 16) + 16u * iq + 4u * ir;\n");
    source.push_str("            device const ushort *q2 = q1 + 32;\n");
    source.push_str("            device const half *dh = (device const half *)blk;\n");
    source.push_str("            ushort sc16_0 = sc[0] & (ushort)0x3f3fu;\n");
    source.push_str("            ushort sc16_1 = sc[2] & (ushort)0x3f3fu;\n");
    source.push_str(
        "            ushort sc16_2 = (ushort)(((sc[4] >> 0) & (ushort)0x0f0fu) | ((sc[0] & (ushort)0xc0c0u) >> 2));\n",
    );
    source.push_str(
        "            ushort sc16_3 = (ushort)(((sc[4] >> 4) & (ushort)0x0f0fu) | ((sc[2] & (ushort)0xc0c0u) >> 2));\n",
    );
    source.push_str(
        "            uchar sc8_0 = (uchar)(sc16_0 & 0xffu); uchar sc8_1 = (uchar)(sc16_0 >> 8);\n",
    );
    source.push_str(
        "            uchar sc8_2 = (uchar)(sc16_1 & 0xffu); uchar sc8_3 = (uchar)(sc16_1 >> 8);\n",
    );
    source.push_str(
        "            uchar sc8_4 = (uchar)(sc16_2 & 0xffu); uchar sc8_5 = (uchar)(sc16_2 >> 8);\n",
    );
    source.push_str(
        "            uchar sc8_6 = (uchar)(sc16_3 & 0xffu); uchar sc8_7 = (uchar)(sc16_3 >> 8);\n",
    );
    source.push_str(
        "            float acc1_0 = 0.0f; float acc1_1 = 0.0f; float acc1_2 = 0.0f; float acc1_3 = 0.0f;\n",
    );
    source.push_str(
        "            float acc2_0 = 0.0f; float acc2_1 = 0.0f; float acc2_2 = 0.0f; float acc2_3 = 0.0f;\n",
    );
    source.push_str("            for (uint i = 0u; i < 4u; ++i) {\n");
    source.push_str("                ushort word1 = q1[i];\n");
    source.push_str("                ushort word2 = q2[i];\n");
    source.push_str(
        "                acc1_0 += yl[2u * i + 0u] * (float)(word1 & (ushort)0x000Fu);\n",
    );
    source.push_str(
        "                acc1_1 += yl[2u * i + 1u] * (float)(word1 & (ushort)0x0F00u);\n",
    );
    source.push_str(
        "                acc1_2 += yl[2u * i + 8u] * (float)(word1 & (ushort)0x00F0u);\n",
    );
    source.push_str(
        "                acc1_3 += yl[2u * i + 9u] * (float)(word1 & (ushort)0xF000u);\n",
    );
    source.push_str(
        "                acc2_0 += yh[2u * i + 0u] * (float)(word2 & (ushort)0x000Fu);\n",
    );
    source.push_str(
        "                acc2_1 += yh[2u * i + 1u] * (float)(word2 & (ushort)0x0F00u);\n",
    );
    source.push_str(
        "                acc2_2 += yh[2u * i + 8u] * (float)(word2 & (ushort)0x00F0u);\n",
    );
    source.push_str(
        "                acc2_3 += yh[2u * i + 9u] * (float)(word2 & (ushort)0xF000u);\n",
    );
    source.push_str("            }\n");
    source.push_str("            float dall = (float)dh[0];\n");
    source.push_str("            float dmin = (float)dh[1];\n");
    source.push_str(
        "            sumf[q] = sumf[q] + dall * ((acc1_0 + (1.0f/256.0f) * acc1_1) * (float)sc8_0 +\n",
    );
    source.push_str(
        "                                       (acc1_2 + (1.0f/256.0f) * acc1_3) * (float)sc8_1 * (1.0f/16.0f) +\n",
    );
    source.push_str(
        "                                       (acc2_0 + (1.0f/256.0f) * acc2_1) * (float)sc8_4 +\n",
    );
    source.push_str(
        "                                       (acc2_2 + (1.0f/256.0f) * acc2_3) * (float)sc8_5 * (1.0f/16.0f)) -\n",
    );
    source.push_str(
        "                      dmin * (sumy0 * (float)sc8_2 + sumy1 * (float)sc8_3 + sumy2 * (float)sc8_6 + sumy3 * (float)sc8_7);\n",
    );
    source.push_str("        }\n");
    source.push_str(&format!(
        "        for (int q = 0; q < {rows}; ++q) {{ blk_ptr[q] += blk_step; }}\n"
    ));
    source.push_str("        y4 += y4_step;\n");
    source.push_str("    }\n");
}

// ggml (llama.cpp, MIT license: https://github.com/ggml-org/llama.cpp/blob/
// master/LICENSE) `kernel_mul_mv_q5_K_f32_impl<2,2,32>`
// (ggml-metal.metal:5209-5324), transcribed line-for-line onto this crate's
// operand-base/stride addressing, the same posture as
// [`push_q4k_ggml_port_body`].
//
// Copyright (c) 2023-2024 The ggml authors. MIT-licensed; see THIRD_PARTY.md.
//
/// `metal-q4k-ggml-port` (default-off, same feature `push_q4k_ggml_port_body`
/// answers to): a VERBATIM port of ggml's `kernel_mul_mv_q5_K_f32_impl<nr0=2,
/// nsg=2, nw=32>` (`ggml-metal.metal:5209-5324`) -- the fix for this crate's
/// own `q5k_pair_dot`, whose non-deferred `(scale*value - minimum)*activation`
/// per lane (`docs/discipline.md` ROW 361/384) does the scale/min multiply-
/// subtract EIGHT times per sub-block; this body applies `dall`/`dmin` ONCE
/// per sub-block over pre-summed `acc1`/`acc2`/`sumy`, the same deferral
/// `push_q4k_ggml_port_body` already ports for `Q4_K`.
///
/// Lane assignment (`ggml-metal.metal:5244-5250`), same shape as
/// [`push_q4k_ggml_port_body`]'s own `tid`/`ix`/`iq`/`ir` (`Q5_K`'s `tid =
/// tiisg/4` divides by 4 where `Q4_K`'s divides by 8, because `Q5_K` has no
/// second `it/4` split -- `iq = tid/4`, `ir = tid%4` read directly off `tid`):
/// `l0 = 8*ir`; `q_offset = 32*iq + l0` (byte offset into `qs`, unused here --
/// folded into the `48 + q_offset` pointer below); `y_offset = 64*iq + l0`,
/// IDENTICAL to `Q4_K`'s own `y4` lane offset, so this body shares
/// [`push_q4k_plain_product_y4_address`] rather than a second copy.
///
/// Super-block iteration (`ggml-metal.metal:5263`): `for (i = ix; i < nb; i
/// += 4)`, `ix = tiisg%4` -- same stride `4` as `Q4_K`'s own `ib_step`
/// despite `Q5_K`'s `nr0=2` (this function still iterates `rows` generically,
/// the same posture [`push_q6k_ggml_port_body`]'s own doc calls out for its
/// `nr0=1`).
///
/// Activation gather (`ggml-metal.metal:5269-5276`): `y2 = y1 + 128`, so
/// `y2[l] == y1[128+l]` and `y2[l+32] == y1[160+l]` -- textually the SAME
/// `yl[i]/yl[i+8]/yh[i]/yh[i+8]` gather from `y4[i]/y4[i+32]/y4[i+128]/
/// y4[i+160]` [`push_q4k_ggml_port_body`] already renders, so this body
/// reuses that exact source text rather than a second copy.
///
/// Scale/min extraction (`ggml-metal.metal:5281-5284`): `Q5_K`'s
/// `scales`/`kmask1`/`kmask2`/`kmask3` bit layout is byte-for-byte `Q4_K`'s
/// own (`sc16_0 = sc[0] & 0x3f3f`, ..., `sc16_2 = ((sc[4]>>0) & 0x0f0f) |
/// ((sc[0]&0xc0c0)>>2)`, ...) at the SAME `blk + 4` byte offset -- restated
/// here rather than shared, same posture `q5k_scale_min` already takes on
/// `q4k_scale_min`.
///
/// High-bit plane (`ggml-metal.metal:5253-5256,5289-5297`, the part `Q4_K`
/// has no analogue for): `hm1 = 1 << (2*iq)`, `hm2 = hm1<<1`, `hm3 = hm1<<4`,
/// `hm4 = hm2<<4`, each fixed per-thread (depends only on `iq`, hoisted once
/// outside the `ib` loop). `qh = blk + 16 + l0`; per lane `l`, `q1 = blk + 48
/// + q_offset`, `q2 = q1 + 64`: `acc1[k]` sums `yl_or_yh[l(+8)]` times the
/// masked nibble byte (low or high half of `q1`/`q2`), `acc2[k]` sums the
/// same activation gated by `qh[l] & hm_k` -- no scale multiply inside this
/// loop at all, unlike `q5k_pair_dot`'s per-element
/// `scale*(nibble+high)-minimum`.
///
/// Final combine (`ggml-metal.metal:5299-5305`): `dall`/`dmin` applied ONCE,
/// `sumf[row] += dall*(sc8_0*(acc1_0+16*acc2_0) + sc8_1*(acc1_1/16 +
/// 16*acc2_1) + sc8_4*(acc1_2+16*acc2_2) + sc8_5*(acc1_3/16+16*acc2_3)) -
/// dmin*(sumy0*sc8_2 + sumy1*sc8_3 + sumy2*sc8_6 + sumy3*sc8_7)` -- the
/// `/16`/`*16` folds are `Q5_K`'s own bit-position residuals (the low
/// sub-block's high-nibble term `q1[l]&0xF0` sits 16x too large), not
/// `Q4_K`'s `1/256` word-packed residual, because this body reads plain
/// `uchar` bytes rather than `Q4_K`'s packed `ushort` words.
///
/// SIMD reduction: unchanged from every other row-blocked arm, handled by
/// the shared `push_packed_row_combine_and_write` tail this function's
/// caller still invokes after it returns.
#[allow(clippy::too_many_arguments)]
pub(super) fn push_q5k_ggml_port_body(
    source: &mut String,
    weight: usize,
    other: usize,
    rows: usize,
    block_bytes: usize,
    other_stride_is_one: bool,
) {
    source.push_str("    uint tid = (uint)lane / 4u;\n");
    source.push_str("    uint ix = (uint)lane % 4u;\n");
    source.push_str("    uint iq = tid / 4u;\n");
    source.push_str("    uint ir = tid % 4u;\n");
    source.push_str("    uint l0 = 8u * ir;\n");
    source.push_str("    uint q_offset = 32u * iq + l0;\n");
    source.push_str("    uchar hm1 = (uchar)(1u << (2u * iq));\n");
    source.push_str("    uchar hm2 = (uchar)(hm1 << 1u);\n");
    source.push_str("    uchar hm3 = (uchar)(hm1 << 4u);\n");
    source.push_str("    uchar hm4 = (uchar)(hm2 << 4u);\n");
    source.push_str(&format!(
        "    int super_blocks = (int)u.reduction_total / {Q4K_BLOCK_ELEMENTS};\n"
    ));
    source.push_str("    float yl[16];\n");
    source.push_str("    float yh[16];\n");
    source.push_str("    int ib_first = (int)ix;\n    int ib_step = 4;\n");
    source.push_str(&format!(
        "    long blk_step = (long)ib_step * {block_bytes};\n"
    ));
    source.push_str(&format!("    device const uchar *blk_ptr[{rows}];\n"));
    source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&format!(
        "        blk_ptr[q] = in{weight} + ((long)((int)weight_base[q] / {Q4K_BLOCK_ELEMENTS}) + (long)ib_first) * {block_bytes};\n"
    ));
    source.push_str("    }\n");
    push_q4k_plain_product_y4_address(source, other, other_stride_is_one);
    source.push_str("    for (int ib = ib_first; ib < super_blocks; ib += ib_step) {\n");
    source.push_str(
        "        float sumy0 = 0.0f; float sumy1 = 0.0f; float sumy2 = 0.0f; float sumy3 = 0.0f;\n",
    );
    if other_stride_is_one {
        source.push_str(
            "        for (uint i = 0u; i < 8u; ++i) {\n            yl[i] = y4[i]; sumy0 += yl[i];\n",
        );
        source.push_str("            yl[i + 8u] = y4[i + 32u]; sumy1 += yl[i + 8u];\n");
        source.push_str("            yh[i] = y4[i + 128u]; sumy2 += yh[i];\n");
        source.push_str("            yh[i + 8u] = y4[i + 160u]; sumy3 += yh[i + 8u];\n");
    } else {
        source.push_str(
            "        for (uint i = 0u; i < 8u; ++i) {\n            yl[i] = y4[(long)i * other_stride]; sumy0 += yl[i];\n",
        );
        source.push_str(
            "            yl[i + 8u] = y4[(long)(i + 32u) * other_stride]; sumy1 += yl[i + 8u];\n",
        );
        source
            .push_str("            yh[i] = y4[(long)(i + 128u) * other_stride]; sumy2 += yh[i];\n");
        source.push_str(
            "            yh[i + 8u] = y4[(long)(i + 160u) * other_stride]; sumy3 += yh[i + 8u];\n",
        );
    }
    source.push_str("        }\n");
    source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str("            device const uchar *blk = blk_ptr[q];\n");
    source
        .push_str("            device const ushort *sc = (device const ushort *)(blk + 4) + iq;\n");
    source.push_str("            device const uchar *qh = blk + 16u + l0;\n");
    source.push_str("            device const uchar *q1 = blk + 48u + q_offset;\n");
    source.push_str("            device const uchar *q2 = q1 + 64u;\n");
    source.push_str("            device const half *dh = (device const half *)blk;\n");
    source.push_str("            ushort sc16_0 = sc[0] & (ushort)0x3f3fu;\n");
    source.push_str("            ushort sc16_1 = sc[2] & (ushort)0x3f3fu;\n");
    source.push_str(
        "            ushort sc16_2 = (ushort)(((sc[4] >> 0) & (ushort)0x0f0fu) | ((sc[0] & (ushort)0xc0c0u) >> 2));\n",
    );
    source.push_str(
        "            ushort sc16_3 = (ushort)(((sc[4] >> 4) & (ushort)0x0f0fu) | ((sc[2] & (ushort)0xc0c0u) >> 2));\n",
    );
    source.push_str(
        "            uchar sc8_0 = (uchar)(sc16_0 & 0xffu); uchar sc8_1 = (uchar)(sc16_0 >> 8);\n",
    );
    source.push_str(
        "            uchar sc8_2 = (uchar)(sc16_1 & 0xffu); uchar sc8_3 = (uchar)(sc16_1 >> 8);\n",
    );
    source.push_str(
        "            uchar sc8_4 = (uchar)(sc16_2 & 0xffu); uchar sc8_5 = (uchar)(sc16_2 >> 8);\n",
    );
    source.push_str(
        "            uchar sc8_6 = (uchar)(sc16_3 & 0xffu); uchar sc8_7 = (uchar)(sc16_3 >> 8);\n",
    );
    source.push_str(
        "            float acc1_0 = 0.0f; float acc1_1 = 0.0f; float acc1_2 = 0.0f; float acc1_3 = 0.0f;\n",
    );
    source.push_str(
        "            float acc2_0 = 0.0f; float acc2_1 = 0.0f; float acc2_2 = 0.0f; float acc2_3 = 0.0f;\n",
    );
    source.push_str("            for (uint l = 0u; l < 8u; ++l) {\n");
    source.push_str("                uchar h = qh[l];\n");
    source.push_str("                acc1_0 += yl[l] * (float)(q1[l] & 0x0Fu);\n");
    source.push_str("                acc1_1 += yl[l + 8u] * (float)(q1[l] & 0xF0u);\n");
    source.push_str("                acc1_2 += yh[l] * (float)(q2[l] & 0x0Fu);\n");
    source.push_str("                acc1_3 += yh[l + 8u] * (float)(q2[l] & 0xF0u);\n");
    source.push_str("                acc2_0 += (h & hm1) != 0u ? yl[l] : 0.0f;\n");
    source.push_str("                acc2_1 += (h & hm2) != 0u ? yl[l + 8u] : 0.0f;\n");
    source.push_str("                acc2_2 += (h & hm3) != 0u ? yh[l] : 0.0f;\n");
    source.push_str("                acc2_3 += (h & hm4) != 0u ? yh[l + 8u] : 0.0f;\n");
    source.push_str("            }\n");
    source.push_str("            float dall = (float)dh[0];\n");
    source.push_str("            float dmin = (float)dh[1];\n");
    source.push_str(
        "            sumf[q] = sumf[q] + dall * ((float)sc8_0 * (acc1_0 + 16.0f * acc2_0) +\n",
    );
    source.push_str(
        "                                       (float)sc8_1 * (acc1_1 * (1.0f/16.0f) + 16.0f * acc2_1) +\n",
    );
    source.push_str(
        "                                       (float)sc8_4 * (acc1_2 + 16.0f * acc2_2) +\n",
    );
    source.push_str(
        "                                       (float)sc8_5 * (acc1_3 * (1.0f/16.0f) + 16.0f * acc2_3)) -\n",
    );
    source.push_str(
        "                      dmin * (sumy0 * (float)sc8_2 + sumy1 * (float)sc8_3 + sumy2 * (float)sc8_6 + sumy3 * (float)sc8_7);\n",
    );
    source.push_str("        }\n");
    source.push_str(&format!(
        "        for (int q = 0; q < {rows}; ++q) {{ blk_ptr[q] += blk_step; }}\n"
    ));
    source.push_str("        y4 += y4_step;\n");
    source.push_str("    }\n");
}

// ggml (llama.cpp, MIT license: https://github.com/ggml-org/llama.cpp/blob/
// master/LICENSE) `kernel_mul_mv_q6_K_f32_impl<1,2,32>`
// (ggml-metal.metal:5340-5433), transcribed line-for-line onto this crate's
// operand-base/stride addressing, the same posture as
// [`push_q4k_ggml_port_body`].
//
// Copyright (c) 2023-2024 The ggml authors. MIT-licensed; see THIRD_PARTY.md.
//
/// `metal-q4k-ggml-port` (default-off, same feature `push_q4k_ggml_port_body`
/// answers to -- ROW 316 named the feature after the first codec it ported,
/// not the mechanism): a VERBATIM port of ggml's
/// `kernel_mul_mv_q6_K_f32_impl<nr0=1, nsg=2, nw=32>`
/// (`ggml-metal.metal:5340-5433`).
///
/// Lane variable mapping, llama -> this function (`ggml-metal.metal:5376-5385`):
/// `tid = tiisg/2` -> `tid = lane/2u`; `ix = tiisg%2` -> `ix = lane%2u`;
/// `ip = tid/8` -> `ip = tid/8u`; `il = tid%8` -> `il = tid%8u`;
/// `l0 = 4*il` -> `l0 = 4u*il`; `y_offset = 128*ip+l0` -> the `lane_offset`
/// this shares with [`push_packed_row_plain_product_y4_address`];
/// `q_offset_l = 64*ip+l0`/`q_offset_h = 32*ip+l0` -> identical names;
/// `is = 8*ip+l0/16` -> identical name. `N_R0_Q6_K = 1`
/// (`ggml-metal-impl.h:38`) collapses ggml's own `row`/`nr0` loop to a
/// single iteration, still written as `for q in 0..rows` so this stays in
/// step with [`PackedCodec::rows_per_simdgroup`] rather than baking `1` in
/// the way `push_q4k_ggml_port_body` bakes `4`.
///
/// Super-block iteration (`ggml-metal.metal:5387`): `for (i = ix; i < nb; i
/// += 2)` -- stride `2` (`N_SIMDWIDTH/2` lanes-per-super-block-half), not
/// `Q4_K`'s stride `4`.
///
/// 6-bit reconstruct (`ggml-metal.metal:5409-5412`): `kmask1=0x03`,
/// `kmask2=0x0C`, `kmask3=0x30`, `kmask4=0xC0`, folded directly into the
/// nibble OR with no runtime shift beyond what each mask already leaves in
/// place, then `- 32` (`Q6_K` has no `dmin` term at all, unlike `Q4_K`/`Q5_K`
/// -- ggml's own `sumf[row] += dall * (...)` has a single scale multiply,
/// no minimum subtraction).
///
/// SIMD reduction: unchanged from every other row-blocked arm, handled by
/// the shared `push_packed_row_combine_and_write` tail this function's
/// caller still invokes after it returns.
#[allow(clippy::too_many_arguments)]
pub(super) fn push_q6k_ggml_port_body(
    source: &mut String,
    weight: usize,
    other: usize,
    rows: usize,
    block_bytes: usize,
    other_stride_is_one: bool,
) {
    source.push_str("    uint tid = (uint)lane / 2u;\n");
    source.push_str("    uint ix = (uint)lane % 2u;\n");
    source.push_str("    uint ip = tid / 8u;\n");
    source.push_str("    uint il = tid % 8u;\n");
    source.push_str("    uint l0 = 4u * il;\n");
    source.push_str("    uint is = 8u * ip + l0 / 16u;\n");
    source.push_str("    uint q_offset_l = 64u * ip + l0;\n");
    source.push_str("    uint q_offset_h = 32u * ip + l0;\n");
    source.push_str(&format!(
        "    int super_blocks = (int)u.reduction_total / {Q4K_BLOCK_ELEMENTS};\n"
    ));
    source.push_str("    float yl[16];\n");
    source.push_str("    int ib_first = (int)ix;\n    int ib_step = 2;\n");
    source.push_str(&format!(
        "    long blk_step = (long)ib_step * {block_bytes};\n"
    ));
    source.push_str(&format!("    device const uchar *blk_ptr[{rows}];\n"));
    source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&format!(
        "        blk_ptr[q] = in{weight} + ((long)((int)weight_base[q] / {Q4K_BLOCK_ELEMENTS}) + (long)ib_first) * {block_bytes};\n"
    ));
    source.push_str("    }\n");
    push_packed_row_plain_product_y4_address(source, other, other_stride_is_one, "128u * ip + l0");
    source.push_str("    for (int ib = ib_first; ib < super_blocks; ib += ib_step) {\n");
    if other_stride_is_one {
        source.push_str("        for (uint l = 0u; l < 4u; ++l) {\n            yl[4u * l + 0u] = y4[l]; yl[4u * l + 1u] = y4[l + 32u]; yl[4u * l + 2u] = y4[l + 64u]; yl[4u * l + 3u] = y4[l + 96u];\n        }\n");
    } else {
        source.push_str("        for (uint l = 0u; l < 4u; ++l) {\n            yl[4u * l + 0u] = y4[(long)l * other_stride]; yl[4u * l + 1u] = y4[(long)(l + 32u) * other_stride]; yl[4u * l + 2u] = y4[(long)(l + 64u) * other_stride]; yl[4u * l + 3u] = y4[(long)(l + 96u) * other_stride];\n        }\n");
    }
    source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str("            device const uchar *blk = blk_ptr[q];\n");
    source.push_str("            device const uchar *ql = blk;\n");
    source.push_str("            device const uchar *qh = blk + 128u;\n");
    source.push_str("            device const uchar *sc = blk + 192u + is;\n");
    source.push_str("            device const half *dh = (device const half *)(blk + 208u);\n");
    source.push_str("            float dall = (float)dh[0];\n");
    source.push_str(
        "            float sums0 = 0.0f; float sums1 = 0.0f; float sums2 = 0.0f; float sums3 = 0.0f;\n",
    );
    source.push_str("            for (uint l = 0u; l < 4u; ++l) {\n");
    source.push_str("                uchar q1l = ql[q_offset_l + l];\n");
    source.push_str("                uchar q2l = ql[q_offset_l + 32u + l];\n");
    source.push_str("                uchar qhl = qh[q_offset_h + l];\n");
    source.push_str(
        "                uint q_lo0 = (uint)(q1l & 0x0Fu) | (((uint)qhl & 0x03u) << 4u);\n",
    );
    source.push_str(
        "                uint q_lo1 = (uint)(q2l & 0x0Fu) | (((uint)qhl & 0x0Cu) << 2u);\n",
    );
    source.push_str("                uint q_hi0 = (uint)(q1l >> 4u) | ((uint)qhl & 0x30u);\n");
    source.push_str(
        "                uint q_hi1 = (uint)(q2l >> 4u) | (((uint)qhl & 0xC0u) >> 2u);\n",
    );
    source.push_str("                sums0 += yl[4u * l + 0u] * (float)((int)q_lo0 - 32);\n");
    source.push_str("                sums1 += yl[4u * l + 1u] * (float)((int)q_lo1 - 32);\n");
    source.push_str("                sums2 += yl[4u * l + 2u] * (float)((int)q_hi0 - 32);\n");
    source.push_str("                sums3 += yl[4u * l + 3u] * (float)((int)q_hi1 - 32);\n");
    source.push_str("            }\n");
    source.push_str(
        "            sumf[q] = sumf[q] + dall * (sums0 * (float)(char)sc[0] + sums1 * (float)(char)sc[2] + sums2 * (float)(char)sc[4] + sums3 * (float)(char)sc[6]);\n",
    );
    source.push_str("        }\n");
    source.push_str(&format!(
        "        for (int q = 0; q < {rows}; ++q) {{ blk_ptr[q] += blk_step; }}\n"
    ));
    source.push_str("        y4 += y4_step;\n");
    source.push_str("    }\n");
}

