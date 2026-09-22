use super::*;

/// verbatim per-node device-function prelude, ported byte-for-byte from
/// `omega/examples/attn_staged_replay.rs`'s own `DEVICE_FUNCTIONS_SOURCE` --
/// each function's own doc-comment there names the exact `R9/nodes/<id>.metal`
/// line range it transcribes and which address computations were replaced by
/// the staged layout's direct `key`/`group`/`lane`/`dim` indices. Kept as one
/// `pub(super)` constant so `render_cached_attention_two_pass` and any future
/// staged-kernel variant share the SAME verbatim text instead of drifting.
///
/// `stage_odd_or_even_fold`, `stage_fold_with_epilogue`, and
/// `stage_fold_with_epilogue_tg` (the `threadgroup`-operand twins of the
/// `device`-operand fold family below) are dead code -- Part C moved every
/// operand these three read to `device` scratch, so only their `_dev`
/// counterparts in [`two_pass_scratch_functions`] are ever called. Left here
/// unchanged (still the fixed-32-lane text) rather than deleted: out of
/// scope for this fix (root cause 2 only reaches the four called `_dev` K-dot
/// folds and the three called `_dev` capacity folds), and deleting live-
/// looking device-function text is a bigger diff than this slice affords.
///
/// `__HALF_HEAD_DIM__`/`__HEAD_DIM__` are placeholder tokens, not Metal
/// identifiers -- the layer-0 fixture this was transcribed from has
/// `head_dim=256` (`half_head_dim=128`), so the original per-key/per-group
/// stride and reduction bound were baked in as the literals `128`/`256`;
/// every other gemma4 layer shape (e.g. the global layers at `head_dim=512`)
/// silently read half the K vector at the wrong stride. `render_cached_attention_two_pass`
/// substitutes both tokens with the real per-`BoundOp` `half_head_dim`/`head_dim`
/// before concatenating this text into the kernel (`HALF_HEAD_DIM`/`HEAD_DIM`
/// can't be referenced by name here: those `constant constexpr` declarations
/// live in `staged_two_pass_source`'s generated text, which this prelude
/// precedes in the final source).
pub(super) const DEVICE_FUNCTIONS_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

__attribute__((noinline))
float stage_identity_element(device const float* in0, uint index) {
    long off0 = (long)index;
    float scratch[1];
    scratch[0] = in0[off0];
    float step0 = scratch[0];
    return step0;
}

__attribute__((noinline))
float stage_odd_or_even_fold(device const float* in0, threadgroup const float* in1, uint key, uint group, uint lane) {
    long stride0 = 1;
    long off0 = (long)key * __HALF_HEAD_DIM__;
    off0 += (long)lane * stride0;
    int walk0 = (int)off0;
    int advance0 = (int)(stride0 * 32);
    long stride1 = 1;
    long off1 = (long)group * __HALF_HEAD_DIM__;
    off1 += (long)lane * stride1;
    int walk1 = (int)off1;
    int advance1 = (int)(stride1 * 32);
    float accumulator;
    bool seeded;
    if (lane == 0u) {
        accumulator = 0.0f;
        seeded = true;
    } else {
        accumulator = 0.0f;
        seeded = true;
    }
    for (int r = (int)lane; r < __HALF_HEAD_DIM__; r += 32) {
        float scratch[2];
        scratch[0] = in0[walk0];
        scratch[1] = in1[walk1];
        float step0 = (scratch[0] * scratch[1]);
        float value = step0;
        accumulator = seeded ? (accumulator + value) : value;
        seeded = true;
        walk0 += advance0;
        walk1 += advance1;
    }
    float reduced = simd_sum(accumulator);
    return reduced;
}

__attribute__((noinline))
float stage_fold_with_epilogue(
    device const float* in0, threadgroup const float* in1,
    device const float* epi0, device const float* epi1, threadgroup const float* epi2,
    uint key, uint group, uint lane, uint mask_index, uint epi2_index)
{
    long stride0 = 1;
    long off0 = (long)key * __HALF_HEAD_DIM__;
    off0 += (long)lane * stride0;
    int walk0 = (int)off0;
    int advance0 = (int)(stride0 * 32);
    long stride1 = 1;
    long off1 = (long)group * __HALF_HEAD_DIM__;
    off1 += (long)lane * stride1;
    int walk1 = (int)off1;
    int advance1 = (int)(stride1 * 32);
    float accumulator;
    bool seeded;
    if (lane == 0u) {
        accumulator = 0.0f;
        seeded = true;
    } else {
        accumulator = 0.0f;
        seeded = true;
    }
    for (int r = (int)lane; r < __HALF_HEAD_DIM__; r += 32) {
        float scratch[2];
        scratch[0] = in0[walk0];
        scratch[1] = in1[walk1];
        float step0 = (scratch[0] * scratch[1]);
        float value = step0;
        accumulator = seeded ? (accumulator + value) : value;
        seeded = true;
        walk0 += advance0;
        walk1 += advance1;
    }
    float reduced = simd_sum(accumulator);
    float epi_scratch[4];
    epi_scratch[0] = epi0[mask_index];
    epi_scratch[1] = epi1[0];
    epi_scratch[2] = epi2[epi2_index];
    epi_scratch[3] = reduced;
    float epi_step0 = epi_scratch[3];
    float epi_step1 = (epi_step0 + epi_scratch[2]);
    float epi_step2 = ((epi_scratch[0] != 0.0f) ? epi_scratch[1] : epi_step1);
    return epi_step2;
}

__attribute__((noinline))
float stage_reduce_trivial(threadgroup const float* in0, uint index, bool is_max) {
    float accumulator = is_max ? -INFINITY : 0.0f;
    bool seeded = true;
    for (long r = 0; r < 1; r++) {
        float scratch[1];
        scratch[0] = in0[index];
        float step0 = scratch[0];
        float value = step0;
        accumulator = seeded ? (is_max ? max(accumulator, value) : (accumulator + value)) : value;
        seeded = true;
    }
    return accumulator;
}

__attribute__((noinline))
float stage_reduce_max_with_epilogue(threadgroup const float* in0, threadgroup const float* epi0, uint group, uint lane) {
    float accumulator;
    bool seeded;
    if (lane == 0u) {
        accumulator = -INFINITY;
        seeded = true;
    } else {
        accumulator = -INFINITY;
        seeded = true;
    }
    long stride0 = 8;
    long off0 = (long)group;
    off0 += (long)lane * stride0;
    int walk0 = (int)off0;
    int advance0 = (int)(stride0 * 32);
    for (int r = (int)lane; r < 32; r += 32) {
        float scratch[1];
        scratch[0] = in0[walk0];
        float step0 = scratch[0];
        float value = step0;
        accumulator = seeded ? max(accumulator, value) : value;
        seeded = true;
        walk0 += advance0;
    }
    float reduced = simd_max(accumulator);
    float epi_scratch[2];
    epi_scratch[0] = epi0[group];
    epi_scratch[1] = reduced;
    float epi_step0 = epi_scratch[1];
    float epi_step1 = max(epi_step0, epi_scratch[0]);
    return epi_step1;
}

__attribute__((noinline))
float stage_reduce_add_cooperative(threadgroup const float* in0, uint group, uint lane) {
    float accumulator;
    bool seeded;
    if (lane == 0u) {
        accumulator = 0.0f;
        seeded = true;
    } else {
        accumulator = 0.0f;
        seeded = true;
    }
    long stride0 = 8;
    long off0 = (long)group;
    off0 += (long)lane * stride0;
    int walk0 = (int)off0;
    int advance0 = (int)(stride0 * 32);
    for (int r = (int)lane; r < 32; r += 32) {
        float scratch[1];
        scratch[0] = in0[walk0];
        float step0 = scratch[0];
        float value = step0;
        accumulator = seeded ? (accumulator + value) : value;
        seeded = true;
        walk0 += advance0;
    }
    float reduced = simd_sum(accumulator);
    return reduced;
}

__attribute__((noinline))
float stage_sub_exp(threadgroup const float* in0, threadgroup const float* in1, uint full_index, uint group_index) {
    float scratch[2];
    scratch[0] = in0[full_index];
    scratch[1] = in1[group_index];
    float step0 = (scratch[0] - scratch[1]);
    float step1 = exp(step0);
    return step1;
}

__attribute__((noinline))
float stage_va_reduce_cooperative(device const float* in0, threadgroup const float* in1, uint dim, uint group, uint lane) {
    float accumulator;
    bool seeded;
    if (lane == 0u) {
        accumulator = 0.0f;
        seeded = true;
    } else {
        accumulator = 0.0f;
        seeded = true;
    }
    long stride0 = __HEAD_DIM__;
    long off0 = (long)dim;
    off0 += (long)lane * stride0;
    int walk0 = (int)off0;
    int advance0 = (int)(stride0 * 32);
    long stride1 = 8;
    long off1 = (long)group;
    off1 += (long)lane * stride1;
    int walk1 = (int)off1;
    int advance1 = (int)(stride1 * 32);
    for (int r = (int)lane; r < 32; r += 32) {
        float scratch[2];
        scratch[0] = in0[walk0];
        scratch[1] = in1[walk1];
        float step0 = (scratch[0] * scratch[1]);
        float value = step0;
        accumulator = seeded ? (accumulator + value) : value;
        seeded = true;
        walk0 += advance0;
        walk1 += advance1;
    }
    float reduced = simd_sum(accumulator);
    return reduced;
}

__attribute__((noinline))
float stage_va_multiply_trivial(device const float* in0, threadgroup const float* in1, uint dim, uint group) {
    float accumulator = 0.0f;
    bool seeded = true;
    for (long r = 0; r < 1; r++) {
        float scratch[2];
        scratch[0] = in0[dim];
        scratch[1] = in1[group];
        float step0 = (scratch[0] * scratch[1]);
        float value = step0;
        accumulator = seeded ? (accumulator + value) : value;
        seeded = true;
    }
    return accumulator;
}

__attribute__((noinline))
float stage_add_reciprocal_add_multiply(threadgroup const float* in0, threadgroup const float* in1, threadgroup const float* in2, threadgroup const float* in3, uint gid, uint group_index) {
    float scratch[4];
    scratch[0] = in0[group_index];
    scratch[1] = in1[group_index];
    scratch[2] = in2[gid];
    scratch[3] = in3[gid];
    float step0 = (scratch[0] + scratch[1]);
    float step1 = (1.0f / step0);
    float step2 = (scratch[2] + scratch[3]);
    float step3 = (step1 * step2);
    return step3;
}

__attribute__((noinline))
float stage_fold_with_epilogue_tg(
    device const float* in0, threadgroup const float* in1,
    threadgroup const float* epi0, float epi1_value, threadgroup const float* epi2,
    uint key, uint group, uint lane, uint mask_index, uint epi2_index)
{
    long stride0 = 1;
    long off0 = (long)key * __HALF_HEAD_DIM__;
    off0 += (long)lane * stride0;
    int walk0 = (int)off0;
    int advance0 = (int)(stride0 * 32);
    long stride1 = 1;
    long off1 = (long)group * __HALF_HEAD_DIM__;
    off1 += (long)lane * stride1;
    int walk1 = (int)off1;
    int advance1 = (int)(stride1 * 32);
    float accumulator;
    bool seeded;
    if (lane == 0u) {
        accumulator = 0.0f;
        seeded = true;
    } else {
        accumulator = 0.0f;
        seeded = true;
    }
    for (int r = (int)lane; r < __HALF_HEAD_DIM__; r += 32) {
        float scratch[2];
        scratch[0] = in0[walk0];
        scratch[1] = in1[walk1];
        float step0 = (scratch[0] * scratch[1]);
        float value = step0;
        accumulator = seeded ? (accumulator + value) : value;
        seeded = true;
        walk0 += advance0;
        walk1 += advance1;
    }
    float reduced = simd_sum(accumulator);
    float epi_scratch[4];
    epi_scratch[0] = epi0[mask_index];
    epi_scratch[1] = epi1_value;
    epi_scratch[2] = epi2[epi2_index];
    epi_scratch[3] = reduced;
    float epi_step0 = epi_scratch[3];
    float epi_step1 = (epi_step0 + epi_scratch[2]);
    float epi_step2 = ((epi_scratch[0] != 0.0f) ? epi_scratch[1] : epi_step1);
    return epi_step2;
}
"#;

/// Loop-body + cross-lane combine shared by every cooperative fold this
/// module generates -- the K-dot fold family (`push_dot_fold`) and the
/// capacity-reduce family (`push_cap_fold`) both call this after emitting
/// their own address arithmetic. Mirrors
/// [`push_cooperative_reduce_tail`](super::push_cooperative_reduce_tail)'s
/// own two-shape split (`simdgroups <= 1` vs `> 1`): at `simdgroups == 1`
/// this is `simd_sum`/`simd_max` and nothing else, byte-identical to the
/// pre-fix text (regression-safe for every layer-0-shaped kernel, where
/// `width` never exceeds `SIMD_WIDTH`); at `simdgroups > 1` it adds the
/// SAME threadgroup-memory partials fold `push_cooperative_reduce_tail`
/// uses in production's own per-node reduce kernels, addressed per-query-
/// group (`partials[group * simdgroups + fold_index]`) since one physical
/// threadgroup here hosts `query_groups` independent folds side by side --
/// a shape production's own reduce kernels never need (one reduction per
/// threadgroup there). The trailing `threadgroup_barrier` (present only in
/// the `simdgroups > 1` arm) is NOT in `push_cooperative_reduce_tail`'s own
/// text -- that emitter's kernel returns immediately after its single
/// reduction, so nothing can race the read; here the SAME `partials` array
/// is reused across repeated calls (once per cached key for the K-dot fold,
/// once per `dim` for the V-cache fold), so every reader must finish before
/// the next call's writers can safely overwrite it.
fn push_cooperative_fold_combine(source: &mut String, simdgroups: u64, combine_fn: &str, partials: &str) {
    if simdgroups <= 1 {
        source.push_str(&format!("        reduced = {combine_fn}(accumulator);\n"));
        source.push_str("    }\n");
        return;
    }
    source.push_str(&format!("        float partial = {combine_fn}(accumulator);\n"));
    source.push_str(&format!(
        "        if (lane == 0u) {{ {partials}[group * {simdgroups}u + (wide_lane / 32u)] = partial; }}\n"
    ));
    source.push_str("    }\n");
    source.push_str("    threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    source.push_str("    if (wide_lane == 0u) {\n");
    source.push_str(&format!("        reduced = {partials}[group * {simdgroups}u];\n"));
    source.push_str(&format!(
        "        for (uint fold_index = 1u; fold_index < {simdgroups}u; ++fold_index) {{\n"
    ));
    source.push_str(&format!(
        "            reduced = (reduced + {partials}[group * {simdgroups}u + fold_index]);\n"
    ));
    source.push_str("        }\n");
    source.push_str("    }\n");
    source.push_str("    threadgroup_barrier(mem_flags::mem_threadgroup);\n");
}

/// K-dot fold family (roles 139/142/146/149): reduces over `HALF_HEAD_DIM`
/// at `width` lanes, `width` chosen by [`two_pass_fold_widths`]'s own
/// `width_half` -- the SAME [`cooperative_reduce_width`](super::cooperative_reduce_width)
/// formula production's real per-node K-dot-fold kernels use
/// (`R9/nodes_l4/895.metal`'s own `advance0 = stride0 * 64` at
/// `half_head_dim=256`). Only lanes `wide_lane < width` participate --
/// `width` is always a multiple of `SIMD_WIDTH`, so this guard is uniform
/// per physical simdgroup, never causing intra-simdgroup divergence.
fn push_dot_fold(source: &mut String, width: u64, simdgroups: u64, in0: &str, in1: &str, partials: &str) {
    source.push_str("    float reduced = 0.0f;\n");
    source.push_str("    float accumulator = 0.0f;\n");
    source.push_str(&format!("    if (wide_lane < {width}u) {{\n"));
    source.push_str("        long stride0 = 1;\n");
    source.push_str("        long off0 = (long)key * __HALF_HEAD_DIM__;\n");
    source.push_str("        off0 += (long)wide_lane * stride0;\n");
    source.push_str("        int walk0 = (int)off0;\n");
    source.push_str(&format!("        int advance0 = (int)(stride0 * {width});\n"));
    source.push_str("        long stride1 = 1;\n");
    source.push_str("        long off1 = (long)group * __HALF_HEAD_DIM__;\n");
    source.push_str("        off1 += (long)wide_lane * stride1;\n");
    source.push_str("        int walk1 = (int)off1;\n");
    source.push_str(&format!("        int advance1 = (int)(stride1 * {width});\n"));
    source.push_str(&format!(
        "        for (int r = (int)wide_lane; r < __HALF_HEAD_DIM__; r += {width}) {{\n"
    ));
    source.push_str("            float scratch[2];\n");
    source.push_str(&format!("            scratch[0] = {in0}[walk0];\n"));
    source.push_str(&format!("            scratch[1] = {in1}[walk1];\n"));
    source.push_str("            float step0 = (scratch[0] * scratch[1]);\n");
    source.push_str("            float value = step0;\n");
    source.push_str("            accumulator = accumulator + value;\n");
    source.push_str("            walk0 += advance0;\n");
    source.push_str("            walk1 += advance1;\n");
    source.push_str("        }\n");
    push_cooperative_fold_combine(source, simdgroups, "simd_sum", partials);
}

/// Capacity-reduce family (roles 151/152/157/158/162): reduces over
/// `CACHED_CAPACITY` at `width` lanes, `width` chosen by
/// [`two_pass_fold_widths`]'s own `width_cap` -- the SAME formula applied to
/// `cached_capacity` instead of `half_head_dim`, so a bucket capacity
/// crossing the widening threshold (`> 128`, e.g. `cached_capacity >= 520`)
/// gets the correct cross-simdgroup topology here too, not just for the
/// K-dot folds. `stride` and `bound` are baked as the real `query_groups`/
/// `cached_capacity` integers (not the placeholder-token scheme
/// `__HALF_HEAD_DIM__` uses) since both are already known Rust-side when
/// this text is generated -- the prior hardcoded `8`/`32` literals here were
/// themselves latent bugs for `query_groups != 8` or `cached_capacity !=
/// 32`, both fixed as a side effect of parameterizing width.
#[allow(clippy::too_many_arguments)]
fn push_cap_fold(
    source: &mut String,
    width: u64,
    simdgroups: u64,
    cached_capacity: u64,
    combine_fn: &str,
    identity: &str,
    combine_expr: &str,
    operands: &[(&str, &str, &str)],
    partials: &str,
) {
    source.push_str("    float reduced = 0.0f;\n");
    source.push_str(&format!("    float accumulator = {identity};\n"));
    source.push_str(&format!("    if (wide_lane < {width}u) {{\n"));
    for (index, (_, base_expr, stride_expr)) in operands.iter().enumerate() {
        source.push_str(&format!("        long stride{index} = {stride_expr};\n"));
        source.push_str(&format!("        long off{index} = (long){base_expr};\n"));
        source.push_str(&format!("        off{index} += (long)wide_lane * stride{index};\n"));
        source.push_str(&format!("        int walk{index} = (int)off{index};\n"));
        source.push_str(&format!("        int advance{index} = (int)(stride{index} * {width});\n"));
    }
    source.push_str(&format!(
        "        for (int r = (int)wide_lane; r < {cached_capacity}; r += {width}) {{\n"
    ));
    source.push_str(&format!("            float scratch[{}];\n", operands.len().max(1)));
    for (index, (name, _, _)) in operands.iter().enumerate() {
        source.push_str(&format!("            scratch[{index}] = {name}[walk{index}];\n"));
    }
    if operands.len() == 2 {
        source.push_str("            float step0 = (scratch[0] * scratch[1]);\n");
    } else {
        source.push_str("            float step0 = scratch[0];\n");
    }
    source.push_str("            float value = step0;\n");
    source.push_str(&format!("            accumulator = {combine_expr};\n"));
    for index in 0..operands.len() {
        source.push_str(&format!("            walk{index} += advance{index};\n"));
    }
    source.push_str("        }\n");
    push_cooperative_fold_combine(source, simdgroups, combine_fn, partials);
}

/// Generated per-shape device-function text for the six `_dev` cooperative
/// folds `staged_two_pass_source` actually calls -- `two_pass_scratch_functions`
/// (was a static const) is now a function of `(width_half, simdgroups_half,
/// width_cap, simdgroups_cap, query_groups, cached_capacity)` because the
/// per-lane loop stride, the partials array width, and the fold-loop trip
/// count are all shape-dependent once `metal-wide-cooperative-reduce`
/// widens past `SIMD_WIDTH` -- see [`push_dot_fold`]/[`push_cap_fold`]'s own
/// docs for why a single static string could no longer express this.
/// `stage_sub_exp_dev` and `stage_add_reciprocal_add_multiply_dev` are
/// unaffected by width (no cooperative reduce in either) and stay literal
/// text, appended unchanged.
#[allow(clippy::too_many_arguments)]
fn two_pass_scratch_functions(
    width_half: u64,
    simdgroups_half: u64,
    width_cap: u64,
    simdgroups_cap: u64,
    query_groups: u64,
    cached_capacity: u64,
) -> String {
    let mut source = String::new();
    let query_groups_str = query_groups.to_string();

    source.push_str(
        "\n__attribute__((noinline))\nfloat stage_odd_or_even_fold_dev(device const float* in0, device const float* in1, threadgroup float* partials_half, uint key, uint group, uint wide_lane, uint lane) {\n",
    );
    push_dot_fold(&mut source, width_half, simdgroups_half, "in0", "in1", "partials_half");
    source.push_str("    return reduced;\n}\n");

    source.push_str(
        "\n__attribute__((noinline))\nfloat stage_fold_with_epilogue_tg_dev_scratch_epi(\n    device const float* in0, device const float* in1, threadgroup float* partials_half,\n    threadgroup const float* epi0, float epi1_value, device const float* epi2,\n    uint key, uint group, uint wide_lane, uint lane, uint mask_index, uint epi2_index)\n{\n",
    );
    push_dot_fold(&mut source, width_half, simdgroups_half, "in0", "in1", "partials_half");
    source.push_str(EPILOGUE_TAIL);

    source.push_str(
        "\n__attribute__((noinline))\nfloat stage_fold_with_epilogue_tg_dev_tg_epi(\n    device const float* in0, device const float* in1, threadgroup float* partials_half,\n    threadgroup const float* epi0, float epi1_value, threadgroup const float* epi2,\n    uint key, uint group, uint wide_lane, uint lane, uint mask_index, uint epi2_index)\n{\n",
    );
    push_dot_fold(&mut source, width_half, simdgroups_half, "in0", "in1", "partials_half");
    source.push_str(EPILOGUE_TAIL);

    source.push_str(
        "\n__attribute__((noinline))\nfloat stage_reduce_max_with_epilogue_dev(device const float* in0, threadgroup float* partials_cap, threadgroup const float* epi0, uint group, uint wide_lane, uint lane) {\n",
    );
    push_cap_fold(
        &mut source,
        width_cap,
        simdgroups_cap,
        cached_capacity,
        "simd_max",
        "-INFINITY",
        "max(accumulator, value)",
        &[("in0", "group", query_groups_str.as_str())],
        "partials_cap",
    );
    source.push_str(
        "    float epi_step1 = -INFINITY;\n    if (wide_lane == 0u) {\n        float epi_scratch[2];\n        epi_scratch[0] = epi0[group];\n        epi_scratch[1] = reduced;\n        float epi_step0 = epi_scratch[1];\n        epi_step1 = max(epi_step0, epi_scratch[0]);\n    }\n    return epi_step1;\n}\n",
    );

    source.push_str(
        "\n__attribute__((noinline))\nfloat stage_reduce_add_cooperative_dev(device const float* in0, threadgroup float* partials_cap, uint group, uint wide_lane, uint lane) {\n",
    );
    push_cap_fold(
        &mut source,
        width_cap,
        simdgroups_cap,
        cached_capacity,
        "simd_sum",
        "0.0f",
        "(accumulator + value)",
        &[("in0", "group", query_groups_str.as_str())],
        "partials_cap",
    );
    source.push_str("    return reduced;\n}\n");

    source.push_str(
        "\n__attribute__((noinline))\nfloat stage_va_reduce_cooperative_dev(device const float* in0, device const float* in1, threadgroup float* partials_cap, uint dim, uint group, uint wide_lane, uint lane) {\n",
    );
    push_cap_fold(
        &mut source,
        width_cap,
        simdgroups_cap,
        cached_capacity,
        "simd_sum",
        "0.0f",
        "(accumulator + value)",
        &[("in0", "dim", "__HEAD_DIM__"), ("in1", "group", query_groups_str.as_str())],
        "partials_cap",
    );
    source.push_str("    return reduced;\n}\n");

    source.push_str(
        r#"
__attribute__((noinline))
float stage_sub_exp_dev(device const float* in0, threadgroup const float* in1, uint full_index, uint group_index) {
    float scratch[2];
    scratch[0] = in0[full_index];
    scratch[1] = in1[group_index];
    float step0 = (scratch[0] - scratch[1]);
    float step1 = exp(step0);
    return step1;
}

__attribute__((noinline))
float stage_add_reciprocal_add_multiply_dev(threadgroup const float* in0, threadgroup const float* in1, device const float* in2, device const float* in3, uint gid, uint group_index) {
    float scratch[4];
    scratch[0] = in0[group_index];
    scratch[1] = in1[group_index];
    scratch[2] = in2[gid];
    scratch[3] = in3[gid];
    float step0 = (scratch[0] + scratch[1]);
    float step1 = (1.0f / step0);
    float step2 = (scratch[2] + scratch[3]);
    float step3 = (step1 * step2);
    return step3;
}
"#,
    );

    source
}

const EPILOGUE_TAIL: &str = "    float epi_step2 = 0.0f;\n    if (wide_lane == 0u) {\n        float epi_scratch[4];\n        epi_scratch[0] = epi0[mask_index];\n        epi_scratch[1] = epi1_value;\n        epi_scratch[2] = epi2[epi2_index];\n        epi_scratch[3] = reduced;\n        float epi_step0 = epi_scratch[3];\n        float epi_step1 = (epi_step0 + epi_scratch[2]);\n        epi_step2 = ((epi_scratch[0] != 0.0f) ? epi_scratch[1] : epi_step1);\n    }\n    return epi_step2;\n}\n";

/// One named region of the `device float* scratch` buffer
/// [`two_pass_scratch_layout`] describes -- `role` is the node id this
/// region's VALUE corresponds to at `omega/examples/attn_staged_replay.rs`'s
/// own layer-0 fixture numbering (134/135/139/142/154/162/164 for the seven
/// base regions, 146/149/151/152/156/157/158 for the seven diagnostic-only
/// scalar regions), stable across every layer/shape -- a caller maps its own
/// real per-layer node id to a region by that role, never by array position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScratchRegion {
    pub role: u32,
    pub offset_elements: u64,
    pub len_elements: u64,
}

/// The `device float* scratch` buffer's layout -- the SINGLE source of truth
/// both [`staged_two_pass_source`]'s own `SCRATCH_*_OFFSET` constexprs and
/// any external reader (a stage-gate harness reading the buffer back after
/// dispatch) compute offsets from, so the two can never drift apart. Seven
/// base regions (134/135/139/142/154/162/164, moved off `threadgroup`
/// because a `QUERY_GROUPS * HALF_HEAD_DIM` or `CACHED_CAPACITY *
/// QUERY_GROUPS` array can outgrow a Metal device's `threadgroup` memory
/// budget at `cached_capacity >= 520` or `head_dim == 512`) plus, when
/// `diag` is set, seven more `QUERY_GROUPS`-sized regions
/// (146/149/151/152/156/157/158 -- the per-`QUERY_GROUPS` scalars that never
/// approach that budget and stay `threadgroup` for the arithmetic itself,
/// diagnostic-only STORE-additional copies appended by
/// [`staged_two_pass_source`] when `diag=true`). Production never sets
/// `diag` -- only a diagnostic caller reading intermediate stage values back
/// does.
#[must_use]
pub fn two_pass_scratch_layout(
    query_groups: u64,
    head_dim: u64,
    cached_capacity: u64,
    diag: bool,
) -> Vec<ScratchRegion> {
    let half_head_dim = head_dim / 2;
    let per_half_dim_array = query_groups * half_head_dim;
    let per_key_array = cached_capacity * query_groups;
    let per_full_dim_array = query_groups * head_dim;
    let mut offset = 0u64;
    let mut regions = Vec::with_capacity(if diag { 14 } else { 7 });
    let mut push = |role: u32, len: u64, regions: &mut Vec<ScratchRegion>| {
        regions.push(ScratchRegion { role, offset_elements: offset, len_elements: len });
        offset += len;
    };
    push(134, per_half_dim_array, &mut regions);
    push(135, per_half_dim_array, &mut regions);
    push(139, per_key_array, &mut regions);
    push(142, per_key_array, &mut regions);
    push(154, per_key_array, &mut regions);
    push(162, per_full_dim_array, &mut regions);
    push(164, per_full_dim_array, &mut regions);
    if diag {
        for role in [146, 149, 151, 152, 156, 157, 158] {
            push(role, query_groups, &mut regions);
        }
    }
    regions
}

fn scratch_role_offset(layout: &[ScratchRegion], role: u32) -> u64 {
    layout
        .iter()
        .find(|region| region.role == role)
        .unwrap_or_else(|| panic!("two_pass_scratch_layout carries no role={role} region"))
        .offset_elements
}

/// Element (f32) count [`render_cached_attention_two_pass`]'s emitted kernel
/// needs in its `device float* scratch` buffer -- the sum of
/// [`two_pass_scratch_layout`]'s own region lengths, `diag` selecting
/// whether the seven diagnostic-only scalar regions are included. A caller
/// sizes its own scratch `MTLBuffer` at `elements * size_of::<f32>()` bytes;
/// production wiring threads this through a `Binding::Scratch` slot the same
/// way the existing single-range split path's own
/// `cached_attention_scratch_len` (`omega/src/metal/arena_encode_dispatch_finish.rs`)
/// sizes its own -- production always passes `diag=false`.
#[must_use]
pub fn two_pass_scratch_elements(query_groups: u64, head_dim: u64, cached_capacity: u64, diag: bool) -> u64 {
    two_pass_scratch_layout(query_groups, head_dim, cached_capacity, diag)
        .iter()
        .map(|region| region.len_elements)
        .sum()
}

/// Per-family cooperative-reduce lane width for
/// [`render_cached_attention_two_pass`]'s own kernel -- the SAME
/// [`cooperative_reduce_width`](super::cooperative_reduce_width) formula
/// production's real per-node reduce kernels use
/// (`omega/src/msl/tiled_gemm_cooperative_scan.rs`'s own doc), applied
/// directly to a reduction_total rather than reading one off a `BoundOp`'s
/// `extents` -- the two-pass kernel bakes every shape constant at emit time
/// instead of packing a `Uniforms::reduction_total` field, so there is no
/// real reduce `BoundOp` to point [`cooperative_reduce_width`] itself at.
/// `half_head_dim` governs the four K-dot fold stages (139/142/146/149);
/// `cached_capacity` governs the three row-reduce stages over cached keys
/// (152/157/162).
#[must_use]
pub fn two_pass_fold_widths(head_dim: u64, cached_capacity: u64) -> (u64, u64) {
    (wide_cooperative_reduce_width(head_dim / 2), wide_cooperative_reduce_width(cached_capacity))
}

/// This kernel's PHYSICAL per-query-group thread block width -- the wider
/// of [`two_pass_fold_widths`]'s two results. A stage needing less width
/// than this just runs its cooperative fold on the first `stage_width / 32`
/// simdgroups of the block and leaves the rest idle
/// (`push_dot_fold`/`push_cap_fold`'s own `wide_lane < width` guards).
/// [`grid_threads`](super::grid_threads)'s `two_pass` arm,
/// [`tiled_gemm_threadgroup_width`](super::tiled_gemm_threadgroup_width)'s
/// `two_pass` arm, and `prepare_uniforms_pack.rs`'s `two_pass`
/// `total_elements` packing all call this SAME function so the dispatched
/// grid, the threadgroup width the driver binds, and the kernel's own
/// in-body bound check can never drift from one another.
#[must_use]
pub fn two_pass_threadgroup_width(head_dim: u64, cached_capacity: u64) -> u64 {
    let (width_half, width_cap) = two_pass_fold_widths(head_dim, cached_capacity);
    width_half.max(width_cap).max(SIMD_WIDTH)
}

/// How many of `query_groups` independent folds run in ONE physical wave of
/// [`render_cached_attention_two_pass`]'s own threadgroup -- the largest
/// divisor of `query_groups` whose physical width (`divisor *
/// two_pass_threadgroup_width(..)`) stays under the real-hardware floor
/// `omega-runtime.toml`'s `[two_pass] max_threadgroup_threads` documents.
/// Only a DIVISOR of `query_groups` is ever returned, so the kernel's own
/// wave loop (`NUM_WAVES = query_groups / groups_per_wave`) never needs a
/// ragged final wave or an out-of-range `group` guard -- see
/// [`two_pass_physical_threadgroup_width`]'s doc for why the review that
/// found this (`resident_nocopy_cache.rs`'s own `dispatch` clamp) called
/// relying on that clamp a correctness hazard, not merely a wasted grid.
#[must_use]
#[cfg(feature = "metal-fuse-attn-decode")]
pub fn two_pass_groups_per_wave(query_groups: u64, head_dim: u64, cached_capacity: u64) -> u64 {
    let width = two_pass_threadgroup_width(head_dim, cached_capacity);
    let max_threads = crate::sized::TWO_PASS_MAX_THREADGROUP_THREADS;
    (1..=query_groups)
        .rev()
        .find(|candidate| query_groups.is_multiple_of(*candidate) && candidate * width <= max_threads)
        .unwrap_or(1)
}

/// Feature off: this kernel is never selected (`render_cached_attention`'s
/// own recognizer gate), so no real caller ever depends on the divisor
/// search above -- every `query_groups` fold still fits one physical
/// threadgroup unchanged, the pre-root-cause-3 shape.
#[must_use]
#[cfg(not(feature = "metal-fuse-attn-decode"))]
pub fn two_pass_groups_per_wave(query_groups: u64, _head_dim: u64, _cached_capacity: u64) -> u64 {
    query_groups
}

/// Physical per-threadgroup thread count [`grid_threads`](super::grid_threads)'s
/// `two_pass` arm, [`tiled_gemm_threadgroup_width`](super::tiled_gemm_threadgroup_width)'s
/// `two_pass` arm, and `prepare_uniforms_pack.rs`'s `two_pass` `total_elements`
/// packing all dispatch -- [`two_pass_groups_per_wave`] folds of
/// [`two_pass_threadgroup_width`] lanes each, never the full `query_groups *
/// two_pass_threadgroup_width` a pre-root-cause-3 build requested (and a
/// real device's `maxTotalThreadsPerThreadgroup` could silently clamp).
#[must_use]
pub fn two_pass_physical_threadgroup_width(query_groups: u64, head_dim: u64, cached_capacity: u64) -> u64 {
    two_pass_groups_per_wave(query_groups, head_dim, cached_capacity) * two_pass_threadgroup_width(head_dim, cached_capacity)
}

/// `two_pass` counterpart of [`super::render_cached_attention`]: the same
/// 15-node gemma4 decode chain `omega/examples/attn_staged_replay.rs`
/// gates byte-exact against `R9/vectors_relaxed`/`R9/vectors`
/// (`R9/stage_gates.log`, `R9/staged_final_gate.log`, both `signature=
/// production9`), moved here so `render_cached_attention` can select it
/// for real `BoundOp`s. Parameters are read from `resolved.kind` instead
/// of taken as free arguments -- `query_groups`, `head_dim`,
/// `cached_key_rows` (the compiled bucket capacity), `new_key_rows`
/// (asserted `1` by the recognizer's `staged_decode_only` decline stage
/// before this op can exist), `window` from `cached_lower_inclusive`
/// (`i64::MIN` -> `None`, else `1 - cached_lower_inclusive`), and `scale`
/// (rejected here as defense-in-depth if not `1.0`, even though
/// `staged_scale_not_unity` already guarantees it upstream).
///
/// Bindings: the nine production inputs (`q_even`, `q_odd`,
/// `k_even_cache`, `k_odd_cache`, `new_k_even`, `new_k_odd`, `v_cache`,
/// `v_new`, `cached_len`), `scratch` (Part C: the six arrays too large for
/// `threadgroup` memory at `cached_capacity >= 520` or `head_dim == 512`,
/// sized by [`two_pass_scratch_elements`]), `out`,
/// `Uniforms{ long total_elements; }` -- twelve buffers total. The extra
/// `scratch` slot means a caller's [`Kernel::bindings`](super::Kernel) now
/// carries one `Binding::Scratch` entry for this shape (`emit_and_classify.rs`'s
/// `bindings` function), one more than the eleven-buffer
/// `two_range_cached_bound` shape it used to match exactly.
pub fn render_cached_attention_two_pass(resolved: &BoundOp, entry: &str, diag: bool) -> Result<String, EmitError> {
    let BoundOpKind::CachedAttention {
        cached_key_rows,
        new_key_rows,
        query_groups,
        head_dim,
        scale,
        cached_lower_inclusive,
        ..
    } = &resolved.kind
    else {
        return Err(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "cached_attention",
            found: resolved.kind.name(),
        });
    };
    if *scale != 1.0 {
        return Err(EmitError::CachedAttentionTwoPassScaleNotUnity {
            node: resolved.node,
            scale_bits: scale.to_bits(),
        });
    }
    let window = (*cached_lower_inclusive != i64::MIN).then(|| (1 - *cached_lower_inclusive) as u64);
    let half_head_dim = (*head_dim / 2).to_string();
    let head_dim_string = head_dim.to_string();
    let device_functions = DEVICE_FUNCTIONS_SOURCE
        .replace("__HALF_HEAD_DIM__", &half_head_dim)
        .replace("__HEAD_DIM__", &head_dim_string);
    let (width_half, width_cap) = two_pass_fold_widths(*head_dim, *cached_key_rows);
    let scratch_functions = two_pass_scratch_functions(
        width_half,
        width_half / SIMD_WIDTH,
        width_cap,
        width_cap / SIMD_WIDTH,
        *query_groups,
        *cached_key_rows,
    )
    .replace("__HALF_HEAD_DIM__", &half_head_dim)
    .replace("__HEAD_DIM__", &head_dim_string);
    Ok(format!(
        "{device_functions}{scratch_functions}{}",
        staged_two_pass_source(*query_groups, *head_dim, *cached_key_rows, *new_key_rows, window, entry, diag)
    ))
}

/// The kernel-text emitter, parameterized per shape -- every model-shape
/// constant (`query_groups`, `head_dim`, `cached_capacity`,
/// `new_key_rows`, `window`) is baked as a `constant constexpr` literal in
/// the emitted text, the way the online-softmax emitter's own per-`BoundOp`
/// uniform packing bakes strides/extents; there is no `Uniforms` struct to
/// carry them at all, since a single fixed-shape kernel text is
/// regenerated per shape rather than parameterized at dispatch time.
/// `new_key_rows != 1` panics -- `render_cached_attention_two_pass`'s only
/// caller is `render_cached_attention`, reached only for ops the
/// recognizer's `staged_decode_only` stage already proved decode-only.
///
/// Part C: 134/135 (`QUERY_GROUPS * HALF_HEAD_DIM` each), 139/142/154
/// (`CACHED_CAPACITY * QUERY_GROUPS` each), and 162/164
/// (`QUERY_GROUPS * HEAD_DIM` each) read/write through one `device float*
/// scratch` buffer at compile-time offsets ([`two_pass_scratch_elements`]'s
/// own layout) rather than `threadgroup` arrays -- a `CACHED_CAPACITY >= 520`
/// bucket or a `HEAD_DIM == 512` model both overflow a Metal device's
/// `threadgroup` budget at this shape's array count. The seven scalars
/// (146, 149, 151, 152, 156, 157, 158) and both causal-mask arrays never
/// approach that budget on their own, so they stay `threadgroup`. Every
/// store/load happens at EXACTLY the same point in the stage order as
/// before (the rounding boundary is the store, `device` or `threadgroup`
/// alike) -- only the pointer type and, where a moved array crosses a
/// `threadgroup_barrier`, that barrier's `mem_flags` change, adding
/// `mem_device` alongside `mem_threadgroup` wherever a barrier makes a
/// `scratch`-resident write from before it visible to a read after it.
///
/// root cause 2 fix: `thread_position_in_threadgroup` splits into `group`
/// (`0..QUERY_GROUPS`) and `wide_lane` (`0..WIDTH`, this kernel's physical
/// per-group thread block -- [`two_pass_threadgroup_width`]'s own value)
/// rather than the fixed 32-lane `group`/`lane` split the pre-fix text
/// used. `lane` (`thread_index % 32`) is kept alongside `wide_lane` for the
/// cooperative folds' own `simd_sum`/`simd_max` combine, which is a
/// hardware-physical-simdgroup operation independent of this kernel's
/// logical width. Every per-group (not per-lane) store that used to guard
/// on `lane == 0u` now guards on `wide_lane == 0u` -- with `WIDTH >
/// SIMD_WIDTH`, multiple physical simdgroups now share one query group's
/// block, so `lane == 0u` alone would fire once per simdgroup and race a
/// redundant store into the same `threadgroup` scalar.
pub(super) fn staged_two_pass_source(
    query_groups: u64,
    head_dim: u64,
    cached_capacity: u64,
    new_key_rows: u64,
    window: Option<u64>,
    entry: &str,
    diag: bool,
) -> String {
    assert!(new_key_rows == 1, "render_cached_attention_two_pass is decode-only: new_key_rows must be 1, got {new_key_rows}");
    let half_head_dim = head_dim / 2;
    let width = two_pass_threadgroup_width(head_dim, cached_capacity);
    let (width_half, width_cap) = two_pass_fold_widths(head_dim, cached_capacity);
    let simdgroups_half = width_half / SIMD_WIDTH;
    let simdgroups_cap = width_cap / SIMD_WIDTH;
    let groups_per_wave = two_pass_groups_per_wave(query_groups, head_dim, cached_capacity);
    let num_waves = query_groups / groups_per_wave;
    let physical_threads = groups_per_wave * width;
    // root cause 5 (global-layer windowing): `window == None` means a
    // GLOBAL layer -- no sliding-window masking at all, every cached key
    // always in range. The prior `unwrap_or(512)` silently imposed a
    // 512-token window on every global layer anyway: harmless while no
    // tested `cached_len` ever reached 512, but once it does, `distance >
    // WINDOW_CEILING` starts masking the OLDEST cached key as `too_old` on
    // a layer that must never window -- real information silently dropped,
    // not a rounding difference. `u64::MAX` keeps `WINDOW_CEILING` (an f32)
    // far above any `distance` a real context length ever produces, so
    // `too_old` never fires for a `None` window regardless of scale.
    let window_value = window.unwrap_or(u64::MAX);
    let window_ceiling = window_value as f32 - 1.0;
    let layout = two_pass_scratch_layout(query_groups, head_dim, cached_capacity, diag);
    let scratch_134_offset = scratch_role_offset(&layout, 134);
    let scratch_135_offset = scratch_role_offset(&layout, 135);
    let scratch_139_offset = scratch_role_offset(&layout, 139);
    let scratch_142_offset = scratch_role_offset(&layout, 142);
    let scratch_154_offset = scratch_role_offset(&layout, 154);
    let scratch_162_offset = scratch_role_offset(&layout, 162);
    let scratch_164_offset = scratch_role_offset(&layout, 164);
    // diagnostic-only: seven extra `SCRATCH_DIAG_<role>_OFFSET` constexprs
    // plus, at each of the seven scalars' EXISTING store point, one extra
    // store-only line copying the already-computed `threadgroup` value into
    // `scratch` -- zero arithmetic change. `diag=false` (production) leaves
    // both interpolations empty (a blank constexpr line, a trailing space at
    // each store point) -- not byte-identical text, but every executable
    // statement is unchanged, so the compiled kernel's BEHAVIOR is
    // unaffected; the diag=true-vs-false attended-output-bits gate below is
    // what actually proves that, not a text diff.
    let diag_roles = [146u32, 149, 151, 152, 156, 157, 158];
    let diag_constexprs = if diag {
        diag_roles
            .iter()
            .map(|role| format!("constant constexpr uint SCRATCH_DIAG_{role}_OFFSET = {}u;\n", scratch_role_offset(&layout, *role)))
            .collect::<String>()
    } else {
        String::new()
    };
    let diag_store: [String; 7] = diag_roles.map(|role| {
        if diag {
            format!(" scratch[SCRATCH_DIAG_{role}_OFFSET + group] = shared{role}[group];")
        } else {
            String::new()
        }
    });
    let [diag_store_146, diag_store_149, diag_store_151, diag_store_152, diag_store_156, diag_store_157, diag_store_158] =
        diag_store;
    let partials_half_decl = if simdgroups_half > 1 {
        format!("    threadgroup float partials_half[QUERY_GROUPS * {simdgroups_half}u];\n")
    } else {
        "    threadgroup float partials_half[1];\n".to_string()
    };
    let partials_cap_decl = if simdgroups_cap > 1 {
        format!("    threadgroup float partials_cap[QUERY_GROUPS * {simdgroups_cap}u];\n")
    } else {
        "    threadgroup float partials_cap[1];\n".to_string()
    };
    format!(
        r#"
// generated by staged_two_pass_source(query_groups={query_groups}, head_dim={head_dim},
// cached_capacity={cached_capacity}, new_key_rows={new_key_rows}, window={window:?},
// width_half={width_half}, width_cap={width_cap}, width={width})
constant constexpr uint QUERY_GROUPS = {query_groups}u;
constant constexpr uint HALF_HEAD_DIM = {half_head_dim}u;
constant constexpr uint HEAD_DIM = {head_dim}u;
constant constexpr uint CACHED_CAPACITY = {cached_capacity}u;
constant constexpr uint GROUPS_PER_WAVE = {groups_per_wave}u;
constant constexpr uint NUM_WAVES = {num_waves}u;
constant constexpr float WINDOW_CEILING = {window_ceiling:.1}f;
constant constexpr uint SCRATCH_134_OFFSET = {scratch_134_offset}u;
constant constexpr uint SCRATCH_135_OFFSET = {scratch_135_offset}u;
constant constexpr uint SCRATCH_139_OFFSET = {scratch_139_offset}u;
constant constexpr uint SCRATCH_142_OFFSET = {scratch_142_offset}u;
constant constexpr uint SCRATCH_154_OFFSET = {scratch_154_offset}u;
constant constexpr uint SCRATCH_162_OFFSET = {scratch_162_offset}u;
constant constexpr uint SCRATCH_164_OFFSET = {scratch_164_offset}u;
{diag_constexprs}
struct Uniforms {{ long total_elements; }};

kernel void {entry}(
    device const float* q_even       [[buffer(0)]],
    device const float* q_odd        [[buffer(1)]],
    device const float* k_even_cache [[buffer(2)]],
    device const float* k_odd_cache  [[buffer(3)]],
    device const float* new_k_even   [[buffer(4)]],
    device const float* new_k_odd    [[buffer(5)]],
    device const float* v_cache      [[buffer(6)]],
    device const float* v_new        [[buffer(7)]],
    device const float* cached_len_buf [[buffer(8)]],
    device float* scratch             [[buffer(9)]],
    device float* out                [[buffer(10)]],
    constant Uniforms& u             [[buffer(11)]],
    uint thread_index [[thread_position_in_threadgroup]])
{{
    if ((long)thread_index >= u.total_elements) {{ return; }}

    threadgroup float shared146[QUERY_GROUPS];
    threadgroup float shared149[QUERY_GROUPS];
    threadgroup float shared151[QUERY_GROUPS];
    threadgroup float shared152[QUERY_GROUPS];
    threadgroup float shared156[QUERY_GROUPS];
    threadgroup float shared157[QUERY_GROUPS];
    threadgroup float shared158[QUERY_GROUPS];
    threadgroup float shared_mask[CACHED_CAPACITY];
    threadgroup float shared_local_mask[1];
{partials_half_decl}{partials_cap_decl}    device float* shared134 = scratch + SCRATCH_134_OFFSET;
    device float* shared135 = scratch + SCRATCH_135_OFFSET;
    device float* shared139 = scratch + SCRATCH_139_OFFSET;
    device float* shared142 = scratch + SCRATCH_142_OFFSET;
    device float* shared154 = scratch + SCRATCH_154_OFFSET;
    device float* shared162 = scratch + SCRATCH_162_OFFSET;
    device float* shared164 = scratch + SCRATCH_164_OFFSET;

    for (uint idx = thread_index; idx < CACHED_CAPACITY; idx += {physical_threads}u) {{
        float cached_len = cached_len_buf[0];
        float key_index = (float)idx;
        float query_absolute = 0.0f + cached_len;
        float cached_len_row = query_absolute - 0.0f;
        float cached_len_ceiling_row = cached_len_row - 1.0f;
        float is_padding = ((key_index > cached_len_ceiling_row) ? 1.0f : 0.0f);
        float distance = query_absolute - key_index;
        float too_old = ((distance > WINDOW_CEILING) ? 1.0f : 0.0f);
        float is_invalid = max(is_padding, too_old);
        shared_mask[idx] = is_invalid;
    }}
    if (thread_index == 0u) {{
        shared_local_mask[0] = 0.0f;
    }}

    for (uint i = thread_index; i < QUERY_GROUPS * HALF_HEAD_DIM; i += {physical_threads}u) {{
        shared134[i] = stage_identity_element(q_even, i);
        shared135[i] = stage_identity_element(q_odd, i);
    }}
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    uint local_group = thread_index / {width}u;
    uint wide_lane = thread_index % {width}u;
    uint lane = thread_index % 32u;
    float neg_inf = as_type<float>(0xff800000u);

    // root cause 3 (launch admission): `QUERY_GROUPS` independent folds no
    // longer share ONE physical threadgroup -- `GROUPS_PER_WAVE` of them run
    // at a time (`GROUPS_PER_WAVE * {width}u` threads, the ACTUAL dispatched
    // physical width), looped `NUM_WAVES` times so every thread hits every
    // barrier in lockstep (`GROUPS_PER_WAVE` is chosen at emission time as a
    // divisor of `QUERY_GROUPS`, so this loop never needs a ragged final
    // wave or an out-of-range `group` guard).
    for (uint wave = 0u; wave < NUM_WAVES; wave++) {{
    uint group = wave * GROUPS_PER_WAVE + local_group;

    for (uint key = 0u; key < CACHED_CAPACITY; key++) {{
        float reduced = stage_odd_or_even_fold_dev(k_odd_cache, shared135, partials_half, key, group, wide_lane, lane);
        if (wide_lane == 0u) {{ shared139[key * QUERY_GROUPS + group] = reduced; }}
    }}
    {{
        float reduced = stage_odd_or_even_fold_dev(new_k_odd, shared135, partials_half, 0u, group, wide_lane, lane);
        if (wide_lane == 0u) {{ shared146[group] = reduced;{diag_store_146} }}
    }}
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    for (uint key = 0u; key < CACHED_CAPACITY; key++) {{
        float value = stage_fold_with_epilogue_tg_dev_scratch_epi(k_even_cache, shared134, partials_half, shared_mask, neg_inf, shared139, key, group, wide_lane, lane, key, key * QUERY_GROUPS + group);
        if (wide_lane == 0u) {{ shared142[key * QUERY_GROUPS + group] = value; }}
    }}
    {{
        float value = stage_fold_with_epilogue_tg_dev_tg_epi(new_k_even, shared134, partials_half, shared_local_mask, neg_inf, shared146, 0u, group, wide_lane, lane, 0u, group);
        if (wide_lane == 0u) {{ shared149[group] = value;{diag_store_149} }}
    }}
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    if (wide_lane == 0u) {{
        shared151[group] = stage_reduce_trivial(shared149, group, true);{diag_store_151}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {{
        float value = stage_reduce_max_with_epilogue_dev(shared142, partials_cap, shared151, group, wide_lane, lane);
        if (wide_lane == 0u) {{ shared152[group] = value;{diag_store_152} }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // root cause 4 (elementwise coverage): every OTHER cap-family stage
    // walks `key` across the full `CACHED_CAPACITY` range at `wide_lane`
    // stride -- this one used the raw hardware `lane` (always 0..31) as
    // both the loop TRIP COUNT and the key index, a latent bug invisible
    // while every tested `CACHED_CAPACITY` stayed <= 32 (`lane == wide_lane`
    // there): keys `32..CACHED_CAPACITY` never got a `shared154` write,
    // read back as untouched device scratch by the two reduce stages below.
    for (uint key = wide_lane; key < CACHED_CAPACITY; key += {width}u) {{
        uint full_index = key * QUERY_GROUPS + group;
        shared154[full_index] = stage_sub_exp_dev(shared142, shared152, full_index, group);
    }}
    if (wide_lane == 0u) {{
        shared156[group] = stage_sub_exp(shared149, shared152, group, group);{diag_store_156}
    }}
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    {{
        float value = stage_reduce_add_cooperative_dev(shared154, partials_cap, group, wide_lane, lane);
        if (wide_lane == 0u) {{ shared157[group] = value;{diag_store_157} }}
    }}
    if (wide_lane == 0u) {{
        shared158[group] = stage_reduce_trivial(shared156, group, false);{diag_store_158}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint dim = 0u; dim < HEAD_DIM; dim++) {{
        float reduced = stage_va_reduce_cooperative_dev(v_cache, shared154, partials_cap, dim, group, wide_lane, lane);
        if (wide_lane == 0u) {{ shared162[group * HEAD_DIM + dim] = reduced; }}
    }}
    for (uint dim = lane; dim < HEAD_DIM; dim += 32u) {{
        shared164[group * HEAD_DIM + dim] = stage_va_multiply_trivial(v_new, shared156, dim, group);
    }}
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    for (uint dim = lane; dim < HEAD_DIM; dim += 32u) {{
        uint gid_local = group * HEAD_DIM + dim;
        out[gid_local] = stage_add_reciprocal_add_multiply_dev(shared157, shared158, shared162, shared164, gid_local, group);
    }}
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }}
}}
"#
    )
}
