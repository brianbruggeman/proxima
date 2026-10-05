use super::*;

const RADIX_BITS: u32 = 8;

pub(super) fn render_top_fraction_select(
    resolved: &BoundOp,
    entry: &str,
) -> Result<String, EmitError> {
    let BoundOpKind::TopFractionSelect {
        operands,
        rows,
        has_keep_rows,
    } = &resolved.kind
    else {
        return Err(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "top_fraction_select",
            found: resolved.kind.name(),
        });
    };
    let output_buffer = operands.len();
    let uniforms_buffer = output_buffer + 1;
    let chunk = rows.div_ceil(SELECTION_THREADGROUP_WIDTH);

    let mut source = String::new();
    preamble(&mut source, false);
    source.push_str("struct Uniforms { long unused; };\n\n");
    source.push_str("static inline uint order_key(float value) {\n");
    source.push_str("    value = (value == 0.0f) ? 0.0f : value;\n");
    source.push_str("    const uint bits = as_type<uint>(value);\n");
    source.push_str("    return bits ^ ((bits >> 31) ? 0xFFFFFFFFu : 0x80000000u);\n}\n\n");
    source.push_str(&format!(
        "kernel void {entry}(device const float* scores [[buffer(0)]], device const float* keep_count [[buffer(1)]],\n"
    ));
    if *has_keep_rows {
        source.push_str("    device const float* keep_rows [[buffer(2)]],\n");
    }
    source.push_str(&format!(
        "    device float* out [[buffer({output_buffer})]], constant Uniforms& u [[buffer({uniforms_buffer})]],\n\
         \tuint tid [[thread_position_in_threadgroup]],\n\
         \tuint sg_id [[simdgroup_index_in_threadgroup]],\n\
         \tuint sg_lane [[thread_index_in_simdgroup]]) {{\n"
    ));
    source.push_str(&format!(
        "    constexpr uint ROWS = {rows}u; constexpr uint THREADS = {SELECTION_THREADGROUP_WIDTH}u; constexpr uint CHUNK = {chunk}u;\n\
         \t(void)u;\n\
         \tconst uint keep = min(uint(keep_count[0]), ROWS);\n"
    ));
    source.push_str(
        "    threadgroup atomic_uint histogram[256];\n\
         \tthreadgroup uint partials[32];\n\
         \tthreadgroup uint shared_prefix;\n\
         \tthreadgroup uint shared_remaining;\n\
         \tif (tid == 0u) { shared_prefix = 0u; shared_remaining = keep; }\n\
         \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n",
    );
    for pass in 0..u32::BITS / RADIX_BITS {
        source.push_str(&radix_pass(pass));
    }
    source.push_str(&tie_walk(*has_keep_rows));
    source.push_str("}\n");
    Ok(source)
}

fn radix_pass(pass: u32) -> String {
    let shift = u32::BITS - RADIX_BITS * (pass + 1);
    let high = shift + RADIX_BITS;
    format!(
        "    // pass {pass}\n\
         \tif (tid < 256u) {{ atomic_store_explicit(&histogram[tid], 0u, memory_order_relaxed); }}\n\
         \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \tfor (uint row = tid; row < ROWS; row += THREADS) {{\n\
         \t\tconst uint key = order_key(scores[row]);\n\
         \t\tif ((ulong(key) >> {high}u) == (ulong(shared_prefix) >> {high}u)) {{ atomic_fetch_add_explicit(&histogram[(key >> {shift}u) & 0xFFu], 1u, memory_order_relaxed); }}\n\
         \t}}\n\
         \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \tif (tid == 0u) {{\n\
         \t\tuint need = shared_remaining;\n\
         \t\tuint digit = 255u;\n\
         \t\twhile (need > atomic_load_explicit(&histogram[digit], memory_order_relaxed)) {{\n\
         \t\t\tneed -= atomic_load_explicit(&histogram[digit], memory_order_relaxed);\n\
         \t\t\tdigit -= 1u;\n\
         \t\t}}\n\
         \t\tshared_prefix |= digit << {shift}u;\n\
         \t\tshared_remaining = need;\n\
         \t}}\n\
         \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n"
    )
}

fn tie_walk(has_keep_rows: bool) -> String {
    let store = if has_keep_rows {
        "out[row] = max(selected ? 1.0f : 0.0f, keep_rows[row]);"
    } else {
        "out[row] = selected ? 1.0f : 0.0f;"
    };
    format!(
        "    const uint threshold = shared_prefix;\n\
         \tconst uint ties_to_take = shared_remaining;\n\
         \tconst uint start = min(tid * CHUNK, ROWS);\n\
         \tconst uint end = min(start + CHUNK, ROWS);\n\
         \tuint equal_count = 0u;\n\
         \tfor (uint row = start; row < end; row++) {{ if (order_key(scores[row]) == threshold) {{ equal_count++; }} }}\n\
         \tconst uint within = simd_prefix_exclusive_sum(equal_count);\n\
         \tconst uint simd_total = simd_sum(equal_count);\n\
         \tif (sg_lane == 0u) {{ partials[sg_id] = simd_total; }}\n\
         \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \tif (tid == 0u) {{\n\
         \t\tuint running = 0u;\n\
         \t\tfor (uint index = 0u; index < 32u; index++) {{ const uint total = partials[index]; partials[index] = running; running += total; }}\n\
         \t}}\n\
         \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \tuint seen = 0u;\n\
         \tconst uint ties_before = partials[sg_id] + within;\n\
         \tfor (uint row = start; row < end; row++) {{\n\
         \t\tconst uint key = order_key(scores[row]);\n\
         \t\tconst bool tied = key == threshold;\n\
         \t\tconst bool selected = key > threshold || (tied && ties_before + seen < ties_to_take);\n\
         \t\tif (tied) {{ seen++; }}\n\
         \t\t{store}\n\
         \t}}\n"
    )
}
