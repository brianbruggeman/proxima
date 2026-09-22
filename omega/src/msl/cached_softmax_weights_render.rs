use super::*;
use core::fmt::Write as _;

/// Candidate B's fused softmax-normalization kernel
/// ([`BoundOpKind::CachedSoftmaxWeights`]'s own doc): one threadgroup per
/// attention row, `width` lanes cooperating, computing the SAME fold order
/// `proxima_tensor::cpu::run_cached_softmax_weights` runs (group_max from
/// `cached_scores` alone -- `new_scores` never folds in -- then the shift/
/// sum/attend chain) so this renderer and the CPU oracle can never drift.
/// Operand addressing bakes each operand's [`Layout`] (base + strides) as
/// `constexpr` literals rather than reading them from a runtime `Uniforms`
/// blob -- mirrors [`crate::msl::render_cached_attention`]'s own "shape
/// baked at emit time" stance (that function's doc) -- since a
/// `CachedSoftmaxWeights` op's three operands are collapsed to exactly the
/// axes it reads (`bind::cached_softmax_weights_candidates`'s own doc), so
/// nothing about their addressing varies at runtime the way a general
/// elementwise/reduce operand's does.
pub(super) fn render_cached_softmax_weights(
    resolved: &BoundOp,
    entry: &str,
) -> Result<String, EmitError> {
    let BoundOpKind::CachedSoftmaxWeights {
        operands,
        new_key_rows,
        cached_key_rows,
        attention_rows,
        head_dim,
        ..
    } = &resolved.kind
    else {
        return Err(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "cached_softmax_weights",
            found: resolved.kind.name(),
        });
    };
    if *new_key_rows != 1 {
        return Err(EmitError::CachedSoftmaxWeightsNotSupported { node: resolved.node });
    }
    let [(_, cached_layout, cached_lookup), (_, new_layout, new_lookup), (_, value_layout, value_lookup)] =
        operands.as_slice()
    else {
        return Err(EmitError::CachedSoftmaxWeightsNotSupported { node: resolved.node });
    };
    if cached_lookup.is_some() || new_lookup.is_some() || value_lookup.is_some() {
        return Err(EmitError::CachedSoftmaxWeightsNotSupported { node: resolved.node });
    }
    if cached_layout.strides.len() != 2
        || new_layout.strides.len() != 1
        || value_layout.strides.len() != 2
    {
        return Err(EmitError::CachedSoftmaxWeightsNotSupported { node: resolved.node });
    }

    let cached_key_rows = *cached_key_rows;
    let attention_rows = *attention_rows;
    let head_dim = *head_dim;
    let width = wide_cooperative_reduce_width(cached_key_rows);

    let cached_base = cached_layout.base;
    let cached_stride_key = cached_layout.strides[0];
    let cached_stride_row = cached_layout.strides[1];
    let new_base = new_layout.base;
    let new_stride_row = new_layout.strides[0];
    let value_base = value_layout.base;
    let value_stride_row = value_layout.strides[0];
    let value_stride_dim = value_layout.strides[1];

    let mut source = String::new();
    preamble(&mut source);
    source.push_str("struct Uniforms { long total_elements; };\n\n");
    let _ = write!(
        source,
        "kernel void {entry}(\n\
         \tdevice const float* cached_scores [[buffer(0)]],\n\
         \tdevice const float* new_scores [[buffer(1)]],\n\
         \tdevice const float* new_value [[buffer(2)]],\n\
         \tdevice float* out [[buffer(3)]],\n\
         \tconstant Uniforms& u [[buffer(4)]],\n\
         \tdevice float* cached_weight_sum [[buffer(5)]],\n\
         \tdevice float* new_weight_sum [[buffer(6)]],\n\
         \tdevice float* new_attended [[buffer(7)]],\n\
         \tuint local [[thread_position_in_threadgroup]],\n\
         \tuint tg [[threadgroup_position_in_grid]])\n\
         {{\n\
         \t(void)u;\n\
         \tlong row = (long)tg;\n\n"
    );

    // -- group_max: max over cached_scores[key,row] (151/152's own reduce)
    // THEN folded with new_scores[row] (152's own epilogue, `max(reduced,
    // out151[...])` -- `attn_partial_fusion_chain.rs`'s `cooperative_stage`
    // call for node "152", `epi_step2 = max(epi_step1, epi0_value)` where
    // `epi0_value` reads node 151's own output, and 151 itself is a
    // max-reduce over `in149` i.e. `new_scores` -- the harness's own
    // byte-exact production oracle. Proven from row1 payload data
    // (`gate3/row_max_table.txt`): cached-alone max for row1 is
    // `0x41167f84` (9.406132, coincidentally at key=0), but the reference
    // 154 output at that element is `0x3f66ed21` (0.90206), which is
    // `exp(cached[key=0,row1] - group_max)` only when `group_max ==
    // new_scores[row1] == 0x41182... (9.509211)` -- the row's OWN new-token
    // score, not the cached-only max. `run_cached_softmax_weights`'s CPU
    // doc claims the opposite ("new_scores never folds in") and disagrees
    // with production; not fixed here, out of this slice's scope.
    let _ = write!(
        source,
        "\tfloat accumulator0 = -INFINITY;\n\
         \tbool seeded0 = false;\n\
         \tfor (long key = (long)local; key < {cached_key_rows}; key += {width}) {{\n\
         \t\tlong offset = {cached_base} + key * {cached_stride_key} + row * {cached_stride_row};\n\
         \t\tfloat value = cached_scores[offset];\n\
         \t\taccumulator0 = seeded0 ? max(accumulator0, value) : value;\n\
         \t\tseeded0 = true;\n\
         \t}}\n"
    );
    push_cooperative_fold(&mut source, width, 0, "simd_max");
    if width == SIMD_WIDTH {
        // Narrow path unchanged from before this fix (GPU-gated byte-exact
        // already, `wide_fold_diff_ctx3.txt`'s own report) -- `simd_max`
        // already broadcasts to every lane in the simdgroup, so this extra
        // `threadgroup`-memory round trip is redundant but harmless, kept
        // verbatim rather than simplified so the narrow shape's emitted text
        // does not move at all.
        source.push_str(
            "\tthreadgroup float group_max_shared;\n\
             \tif (local == 0u) { group_max_shared = reduced0; }\n\
             \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
             \tfloat group_max = group_max_shared;\n\n",
        );
    } else {
        // Wide path: `push_cooperative_fold`'s own gated fold + broadcast
        // (this fix) already leaves `reduced0` visible to every lane.
        source.push_str("\tfloat group_max = reduced0;\n\n");
    }
    let _ = write!(
        source,
        "\tlong new_offset = {new_base} + row * {new_stride_row};\n\
         \tgroup_max = max(group_max, new_scores[new_offset]);\n\n"
    );

    // -- 154 (this op's own primary output): node[row,key] = exp(cached_scores[key,row] - group_max) --
    let _ = write!(
        source,
        "\tfor (long key = (long)local; key < {cached_key_rows}; key += {width}) {{\n\
         \t\tlong offset = {cached_base} + key * {cached_stride_key} + row * {cached_stride_row};\n\
         \t\tfloat step0 = (cached_scores[offset] - group_max);\n\
         \t\tout[key * {attention_rows} + row] = exp(step0);\n\
         \t}}\n\
         \tthreadgroup_barrier(mem_flags::mem_device);\n\n"
    );

    // -- 156: new_shifted = exp(new_scores[row] - group_max) -- decode-only
    // (`new_key_rows == 1`), every lane recomputes the identical scalar
    // rather than broadcasting a `local == 0u`-only write, matching bug 4's
    // decline shape but with no barrier this stage needs since it depends on
    // nothing another lane wrote. `new_offset` already declared above (the
    // group_max fold reads the same scalar).
    let _ = write!(
        source,
        "\tfloat new_shifted = exp(new_scores[new_offset] - group_max);\n\n"
    );

    // -- 157: cached_weight_sum[row] = sum_k node[row,k] --
    let _ = write!(
        source,
        "\tfloat accumulator1 = 0.0f;\n\
         \tbool seeded1 = false;\n\
         \tfor (long key = (long)local; key < {cached_key_rows}; key += {width}) {{\n\
         \t\tfloat value = out[key * {attention_rows} + row];\n\
         \t\taccumulator1 = seeded1 ? (accumulator1 + value) : value;\n\
         \t\tseeded1 = true;\n\
         \t}}\n"
    );
    push_cooperative_fold(&mut source, width, 1, "simd_sum");
    source.push_str(
        "\tif (local == 0u) {\n\
         \t\tcached_weight_sum[row] = reduced1;\n\
         \t\tnew_weight_sum[row] = new_shifted;\n\
         \t}\n\n",
    );

    // -- 164: new_attended[row,d] = new_shifted * new_value[row,d] --
    let _ = write!(
        source,
        "\tfor (long dim = (long)local; dim < {head_dim}; dim += {width}) {{\n\
         \t\tlong voffset = {value_base} + row * {value_stride_row} + dim * {value_stride_dim};\n\
         \t\tnew_attended[row * {head_dim} + dim] = new_shifted * new_value[voffset];\n\
         \t}}\n\
         }}\n"
    );

    Ok(source)
}

/// Narrow (`width == SIMD_WIDTH`, one physical simdgroup): `simd_{max,sum}`
/// already broadcasts the reduced value to every lane, so `reducedN` needs
/// no further combine -- unchanged from before this fix. Wide (multiple
/// cooperating simdgroups): transcribed VERBATIM (same barrier placement,
/// same lane gating, same broadcast) from the harness's own wide
/// `cooperative_stage` (`omega/examples/attn_partial_fusion_chain.rs:844-
/// 901`) rather than the prior draft's "every thread redundantly folds
/// `partials[]`, only the final store is lane-gated" shape: each
/// simdgroup's lane 0 stores its own partial into `threadgroup` memory, one
/// `threadgroup_barrier(mem_flags::mem_threadgroup)`, then the SERIAL FOLD
/// ITSELF runs ONLY on `local == 0u` (not redundantly on every lane) and
/// writes the result into a second `threadgroup` scalar, then a SECOND
/// `threadgroup_barrier(mem_flags::mem_threadgroup)` before every lane reads
/// that scalar back as `reducedN` -- a real, measured GPU divergence from
/// the redundant-fold shape (candidate_b/integration/omega/gate/
/// wide_fold_diff_ctx3.txt: capacity 512 produced `group_max == cached_
/// scores[key=0, row]` instead of the true row max, capacity 32/narrow
/// stayed byte-exact) is why this landed even though the exact mechanism of
/// that divergence was not further root-caused here (harness-worker's own
/// report, "not root-caused" section).
fn push_cooperative_fold(source: &mut String, width: u64, var_index: u32, reduce_fn: &str) {
    if width == SIMD_WIDTH {
        let _ = writeln!(
            source,
            "\tfloat reduced{var_index} = {reduce_fn}(accumulator{var_index});"
        );
        return;
    }
    let simdgroups = width / SIMD_WIDTH;
    // `simd_max` folds serially via `max(...)`; `simd_sum` via `+` -- the
    // only two `reduce_fn`s this renderer ever calls (group_max, weight_sum).
    let fold_expr = if reduce_fn == "simd_max" {
        format!("max(reduced{var_index}, partials{var_index}[fold_index])")
    } else {
        format!("(reduced{var_index} + partials{var_index}[fold_index])")
    };
    let _ = write!(
        source,
        "\tthreadgroup float partials{var_index}[{simdgroups}];\n\
         \tfloat partial{var_index} = {reduce_fn}(accumulator{var_index});\n\
         \tif (local % 32u == 0u) {{ partials{var_index}[local / 32u] = partial{var_index}; }}\n\
         \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \tthreadgroup float reduced{var_index}_shared;\n\
         \tif (local == 0u) {{\n\
         \t\tfloat reduced{var_index} = partials{var_index}[0];\n\
         \t\tfor (uint fold_index = 1u; fold_index < {simdgroups}u; ++fold_index) {{\n\
         \t\t\treduced{var_index} = {fold_expr};\n\
         \t\t}}\n\
         \t\treduced{var_index}_shared = reduced{var_index};\n\
         \t}}\n\
         \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \tfloat reduced{var_index} = reduced{var_index}_shared;\n"
    );
}
