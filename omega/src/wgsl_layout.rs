//! WGSL text for loading and storing a [`FloatLayout`]: the same integer steps as
//! `msl::layout_render`, in WGSL's syntax (`select` for the conditional, typed `let` bindings,
//! `countLeadingZeros`). WGSL has no scalar half or bfloat bit conversion, so every layout but
//! the one equal to binary32 renders as field arithmetic, exact for all of them.
//!
//! Text emission only: no wgpu device runs it in the gate, so no parity run backs it.

use alloc::format;
use alloc::string::String;

use proxima_primitives::float_layout::FloatLayout;

use crate::msl::layout_render::{
    F32_ABS_MASK, F32_FRACTION_BITS, F32_INFINITY_BITS, F32_QUIET_BIT, exponent_ones,
    hardware_type, infinity_code, mantissa_mask, overflow_code, quiet_bit, reserves_all_ones,
    sign_shift,
};

/// `fn name(code: u32) -> f32`: the value of the `layout.width()`-bit `code`.
#[must_use]
pub fn layout_load_wgsl(layout: FloatLayout, name: &str) -> String {
    match hardware_type(&layout) {
        Some("float") => {
            format!("\nfn {name}(code: u32) -> f32 {{\n    return bitcast<f32>(code);\n}}\n")
        }
        _ => field_load(&layout, name),
    }
}

fn field_load_reserved(layout: &FloatLayout) -> String {
    if !reserves_all_ones(layout) {
        return String::new();
    }
    let ones = exponent_ones(layout);
    let fraction_shift = F32_FRACTION_BITS - u32::from(layout.mantissa_bits);
    format!(
        "    if (exponent == {ones:#x}u) {{
        var payload = 0u;
        if (mantissa != 0u) {{
            payload = {F32_QUIET_BIT:#x}u | (mantissa << {fraction_shift}u);
        }}
        return bitcast<f32>(sign | {F32_INFINITY_BITS:#010x}u | payload);
    }}
"
    )
}

fn field_load(layout: &FloatLayout, name: &str) -> String {
    let mantissa_bits = layout.mantissa_bits;
    let mantissa_mask = mantissa_mask(layout);
    let ones = exponent_ones(layout);
    let sign_shift = sign_shift(layout);
    let bias = layout.bias();
    let implicit = 1u64 << mantissa_bits;
    let reserved = field_load_reserved(layout);
    format!(
        "
fn {name}_scaled_bits(significand: u32, scale: i32) -> u32 {{
    if (significand == 0u) {{
        return 0u;
    }}
    let lead = 31 - i32(countLeadingZeros(significand));
    let biased = lead + scale + 127;
    if (biased >= 255) {{
        return {F32_INFINITY_BITS:#010x}u;
    }}
    if (biased >= 1) {{
        return (u32(biased) << {F32_FRACTION_BITS}u) | ((significand << u32({F32_FRACTION_BITS} - lead)) & 0x7fffffu);
    }}
    let shift = scale + 149;
    if (shift >= 0) {{
        return significand << u32(shift);
    }}
    if (-shift >= 32) {{
        return 0u;
    }}
    return significand >> u32(-shift);
}}

fn {name}(code: u32) -> f32 {{
    let mantissa = code & {mantissa_mask:#x}u;
    let exponent = (code >> {mantissa_bits}u) & {ones:#x}u;
    let sign = ((code >> {sign_shift}u) & 1u) << 31u;
{reserved}    let significand = select({implicit:#x}u | mantissa, mantissa, exponent == 0u);
    let scale = select(i32(exponent) - ({bias}) - {mantissa_bits}, 1 - ({bias}) - {mantissa_bits}, exponent == 0u);
    return bitcast<f32>(sign | {name}_scaled_bits(significand, scale));
}}
"
    )
}

/// `fn name(value: f32) -> u32`: the `layout.width()`-bit code nearest `value`, ties to the even
/// code, overflow and NaN under the layout's specials.
#[must_use]
pub fn layout_store_wgsl(layout: FloatLayout, name: &str) -> String {
    match hardware_type(&layout) {
        Some("float") => {
            format!("\nfn {name}(value: f32) -> u32 {{\n    return bitcast<u32>(value);\n}}\n")
        }
        _ => field_store(&layout, name),
    }
}

fn field_store_nan(layout: &FloatLayout) -> String {
    if !reserves_all_ones(layout) {
        return format!(
            "        return {:#x}u;\n",
            crate::msl::layout_render::max_normal_code(layout)
        );
    }
    format!(
        "        return sign | {:#x}u | (((bits & 0x7fffffu) >> {}u) | {:#x}u);\n",
        infinity_code(layout),
        F32_FRACTION_BITS - u32::from(layout.mantissa_bits),
        quiet_bit(layout)
    )
}

fn field_store(layout: &FloatLayout, name: &str) -> String {
    let mantissa_bits = layout.mantissa_bits;
    let sign_shift = sign_shift(layout);
    let bias = layout.bias();
    let ones = exponent_ones(layout);
    let overflow = overflow_code(layout);
    let value_shift = F32_FRACTION_BITS - u32::from(mantissa_bits);
    let nan = field_store_nan(layout);
    let shift_limit = F32_FRACTION_BITS + 1;
    format!(
        "
fn {name}_round_shift(value: u32, shift: u32) -> u32 {{
    if (shift == 0u) {{
        return value;
    }}
    if (shift > {shift_limit}u) {{
        return 0u;
    }}
    let quotient = value >> shift;
    let remainder = value & ((1u << shift) - 1u);
    let halfway = 1u << (shift - 1u);
    if (remainder > halfway || (remainder == halfway && (quotient & 1u) == 1u)) {{
        return quotient + 1u;
    }}
    return quotient;
}}

fn {name}(value: f32) -> u32 {{
    let bits = bitcast<u32>(value);
    let sign = (bits >> 31u) << {sign_shift}u;
    let abs_bits = bits & {F32_ABS_MASK:#x}u;
    if (abs_bits > {F32_INFINITY_BITS:#010x}u) {{
{nan}    }}
    if ((abs_bits >> {F32_FRACTION_BITS}u) == 0xffu) {{
        return sign | {overflow:#x}u;
    }}
    let exponent_field = abs_bits >> {F32_FRACTION_BITS}u;
    let significand = select((abs_bits & 0x7fffffu) | 0x800000u, abs_bits, exponent_field == 0u);
    let source_exponent = select(i32(exponent_field), 1, exponent_field == 0u);
    let target_exponent = source_exponent - 127 + ({bias});
    if (target_exponent > {ones}) {{
        return sign | {overflow:#x}u;
    }}
    let base_exponent = select(u32(target_exponent - 1), 0u, target_exponent < 1);
    let extra_shift = select(0u, u32(1 - target_exponent), target_exponent < 1);
    let code = (base_exponent << {mantissa_bits}u) + {name}_round_shift(significand, {value_shift}u + extra_shift);
    return sign | min(code, {overflow:#x}u);
}}
"
    )
}
