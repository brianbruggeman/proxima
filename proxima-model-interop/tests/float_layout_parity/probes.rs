use omega::msl::layout_render::{layout_load_msl, layout_store_msl};
use proxima_gguf::quant::float_layout::{decode, encode};
use proxima_primitives::float_layout::FloatLayout;

const HEADER: &str = "#include <metal_stdlib>\nusing namespace metal;\n";

pub fn prelude(layout: FloatLayout) -> String {
    format!(
        "{HEADER}{}{}",
        layout_load_msl(layout, "probe_load"),
        layout_store_msl(layout, "probe_store")
    )
}

pub fn decode_source(layout: FloatLayout) -> String {
    format!(
        "{}
kernel void probe(device const uint *codes [[buffer(0)]], device float *values [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{
    values[gid] = probe_load(codes[gid]);
}}
",
        prelude(layout)
    )
}

pub fn encode_source(layout: FloatLayout) -> String {
    format!(
        "{}
kernel void probe(device const float *values [[buffer(0)]], device uint *codes [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{
    codes[gid] = probe_store(values[gid]);
}}
",
        prelude(layout)
    )
}

pub fn unary_source(layout: FloatLayout, expression: &str) -> String {
    format!(
        "{}
kernel void probe(device const uint *codes [[buffer(0)]], device uint *stored [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{
    float value = probe_load(codes[gid]);
    stored[gid] = probe_store({expression});
}}
",
        prelude(layout)
    )
}

pub fn add_source(layout: FloatLayout) -> String {
    format!(
        "{}
kernel void probe(device const uint *left [[buffer(0)]], device const uint *right [[buffer(1)]], device uint *stored [[buffer(2)]], uint gid [[thread_position_in_grid]]) {{
    stored[gid] = probe_store(probe_load(left[gid]) + probe_load(right[gid]));
}}
",
        prelude(layout)
    )
}

pub fn row_sum_source(layout: FloatLayout, columns: usize) -> String {
    format!(
        "{}
kernel void probe(device const uint *codes [[buffer(0)]], device uint *stored [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{
    float accumulator = 0.0f;
    for (uint column = 0u; column < {columns}u; ++column) {{
        accumulator += probe_load(codes[gid * {columns}u + column]);
    }}
    stored[gid] = probe_store(accumulator);
}}
",
        prelude(layout)
    )
}

pub fn matmul_source(layout: FloatLayout, reduction: usize, columns: usize) -> String {
    format!(
        "{}
kernel void probe(device const uint *left [[buffer(0)]], device const uint *right [[buffer(1)]], device uint *stored [[buffer(2)]], uint gid [[thread_position_in_grid]]) {{
    uint row = gid / {columns}u;
    uint column = gid % {columns}u;
    float accumulator = 0.0f;
    for (uint step = 0u; step < {reduction}u; ++step) {{
        accumulator = fma(probe_load(left[row * {reduction}u + step]), probe_load(right[step * {columns}u + column]), accumulator);
    }}
    stored[gid] = probe_store(accumulator);
}}
",
        prelude(layout)
    )
}

pub fn narrow(layout: FloatLayout, values: &[f32]) -> Vec<u32> {
    values.iter().map(|&value| encode(value, layout)).collect()
}

pub fn widen(layout: FloatLayout, codes: &[u32]) -> Vec<f32> {
    codes.iter().map(|&code| decode(code, layout)).collect()
}

pub fn cpu_negate(layout: FloatLayout, codes: &[u32]) -> Vec<u32> {
    codes
        .iter()
        .map(|&code| encode(-decode(code, layout), layout))
        .collect()
}

pub fn cpu_double(layout: FloatLayout, codes: &[u32]) -> Vec<u32> {
    codes
        .iter()
        .map(|&code| encode(decode(code, layout) * 2.0, layout))
        .collect()
}

pub fn cpu_add(layout: FloatLayout, left: &[u32], right: &[u32]) -> Vec<u32> {
    left.iter()
        .zip(right)
        .map(|(&a_code, &b_code)| encode(decode(a_code, layout) + decode(b_code, layout), layout))
        .collect()
}

pub fn cpu_row_sum(layout: FloatLayout, codes: &[u32], columns: usize) -> Vec<u32> {
    codes
        .chunks_exact(columns)
        .map(|row| {
            encode(
                row.iter().map(|&code| decode(code, layout)).sum::<f32>(),
                layout,
            )
        })
        .collect()
}

pub fn cpu_matmul(
    layout: FloatLayout,
    left: &[u32],
    right: &[u32],
    reduction: usize,
    columns: usize,
) -> Vec<u32> {
    let rows = left.len() / reduction;
    (0..rows * columns)
        .map(|index| {
            let (row, column) = (index / columns, index % columns);
            let mut accumulator = 0.0f32;
            for step in 0..reduction {
                accumulator = decode(left[row * reduction + step], layout)
                    .mul_add(decode(right[step * columns + column], layout), accumulator);
            }
            encode(accumulator, layout)
        })
        .collect()
}
