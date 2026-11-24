//! The one MSL emitter of an encode: binary32 rows in, a cache format's bytes out, rendered from
//! the format's description and naming no format.
//!
//! Teaching pointer: [`proxima_numeric::FormatCodec::encode`] is the host algorithm. A scalar
//! format encodes through [`layout_store_msl`] (the layout's own integer steps, nearest ties to
//! even). A block format encodes through its [`EncodeRule`]
//! ([`BlockLayout::encode_rule`]): the statistic that fixes the scale, the divisor, the rounding,
//! the clamp, the stored-width round trip of the scale and the zero code of a degenerate scale are
//! each a field of the rule, and the planes' bit addresses are read from the layout, so the
//! rendered code is the host loop with the constants substituted. A format with no rule has no
//! rendering and [`encode_kernel_refusal`] says which property it lacks.
//!
//! One thread encodes one block, walking its elements in order: the mean statistic is an
//! f32 sum whose rounding depends on that order, so a tree reduction would not be byte-equal to
//! the host. The arithmetic is guarded against reassociation and contraction with the Metal
//! `math_mode(safe)` pragma and uses `precise::divide`, because the host divides and multiplies
//! separately and the codes are a function of those exact roundings.
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_numeric::block_layout::BlockLayout;
use proxima_numeric::encode_rule::{CodeClamp, CodeRounding, EncodeRule, ScaleRounding, ScaleStatistic};
use proxima_numeric::float_layout::{FloatLayout, SignEncoding};
use proxima_numeric::float_layout_codec;
use proxima_numeric::{FormatCodec, NumericFormat};

use super::layout_render::{layout_load_msl, layout_store_msl};
pub use crate::encode_render::RenderedEncode;
use crate::encode_render::{EncodeDialect, wrap_block_kernel};

const F32_EXPONENT_MASK: u32 = 0x7f80_0000;
const F32_MAGNITUDE_MASK: u32 = 0x7fff_ffff;
const SUM_BYTES: usize = 2;

const BYTE: usize = core::mem::size_of::<u8>();
const HALF_WORD: usize = core::mem::size_of::<u16>();
const WORD: usize = core::mem::size_of::<u32>();

/// The Metal spelling of the neutral encode render ([`crate::encode_render`]).
pub struct MslEncode;

impl EncodeDialect for MslEncode {
    fn prelude(&self) -> String {
        String::from("#include <metal_stdlib>\nusing namespace metal;\n")
    }

    fn storage_token(&self, bytes: usize) -> Option<&'static str> {
        match bytes {
            BYTE => Some("uchar"),
            HALF_WORD => Some("ushort"),
            WORD => Some("uint"),
            _ => None,
        }
    }

    fn scalar_store(&self, layout: FloatLayout, name: &str) -> String {
        layout_store_msl(layout, name)
    }

    fn scalar_refusal(&self, _layout: &FloatLayout) -> Option<&'static str> {
        None
    }

    fn kernel_open(&self, name: &str, input: &str, storage: &str) -> String {
        format!("kernel void {name}_kernel(device const {input} *input [[buffer(0)]], device {storage} *output [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{\n")
    }

    fn block_function(&self, layout: &BlockLayout, name: &str) -> Option<String> {
        block_encode_msl(layout, name)
    }

    fn block_refusal(&self) -> Option<&'static str> {
        None
    }
}

/// Why `format` has no rendered encode: the property it lacks. Decided from the description alone.
///
/// # Errors
/// The property of the format that makes it unrenderable.
#[must_use = "the refusal names why a format has no rendered encode"]
pub fn encode_kernel_refusal(format: &NumericFormat) -> Result<(), &'static str> {
    crate::encode_render::encode_kernel_refusal(&MslEncode, format)
}

/// Renders the encode kernel of `format`, entry point `{name}_kernel`.
///
/// # Errors
/// [`encode_kernel_refusal`]'s reason.
#[must_use = "the refusal names why a format has no rendered encode"]
pub fn format_encode_kernel_msl(format: &NumericFormat, name: &str) -> Result<RenderedEncode, &'static str> {
    crate::encode_render::format_encode_kernel(&MslEncode, format, name)
}

/// [`format_encode_kernel_msl`] for a block layout under `rule` instead of the layout's own rule;
/// `None` when [`block_encode_msl_by_rule`] has no rendering.
#[must_use]
pub fn block_encode_kernel_by_rule(layout: &BlockLayout, rule: &EncodeRule, name: &str) -> Option<RenderedEncode> {
    wrap_block_kernel(&MslEncode, layout, name, block_encode_msl_by_rule(layout, rule, name)?)
}

fn code_bits(layout: &BlockLayout) -> u32 {
    u32::from(layout.element.low.bits) + layout.element.high.map_or(0, |plane| u32::from(plane.bits))
}

fn is_twos_complement(layout: &BlockLayout) -> bool {
    layout.element.code.exponent_bits == 0 && matches!(layout.element.code.sign, SignEncoding::TwosComplement)
}

/// Renders `static inline void name(device const float *input, device uchar *block)`: the encode
/// of the `layout.elements` values at `input` into the `layout.block_bytes` bytes at `block`, or
/// `None` when the layout has no [`EncodeRule`].
#[must_use]
pub fn block_encode_msl(layout: &BlockLayout, name: &str) -> Option<String> {
    let rule = layout.encode_rule()?;
    if rule.is_float_code() {
        return float_block_encode_msl(layout, &rule, name);
    }
    let scale_store = layout_store_msl(layout.scale.layout, &format!("{name}_scale_store"));
    let scale_load = layout_load_msl(layout.scale.layout, &format!("{name}_scale_load"));
    let sum_store = if rule.stored_sum {
        layout_store_msl(FloatLayout::F16, &format!("{name}_sum_store"))
    } else {
        String::new()
    };
    let fit = fit_msl(layout, &rule)?;
    let codes = codes_msl(layout, &rule, name)?;
    let scatter = scatter_msl(layout);
    let scale_bytes = scale_bytes_msl(layout, name);
    let sum = sum_msl(layout, &rule, name);
    let bytes = layout.block_bytes;
    Some(format!(
        "{scale_store}{scale_load}{sum_store}
static inline int {name}_to_int(float value) {{
    if ((as_type<uint>(value) & {F32_MAGNITUDE_MASK:#x}u) > {F32_EXPONENT_MASK:#x}u) {{
        return 0;
    }}
    if (value <= -2147483648.0f) {{
        return -2147483647 - 1;
    }}
    if (value >= 2147483648.0f) {{
        return 2147483647;
    }}
    return (int)value;
}}

static inline void {name}(device const float *input, device uchar *block) {{
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
{fit}{codes}    uchar bytes[{bytes}];
    for (uint position = 0u; position < {bytes}u; ++position) {{
        bytes[position] = 0;
    }}
{scatter}{scale_bytes}{sum}    for (uint position = 0u; position < {bytes}u; ++position) {{
        block[position] = bytes[position];
    }}
}}
",
        bytes = bytes,
    ))
}

/// [`block_encode_msl`] for `rule`, which a caller may fill differently from
/// [`BlockLayout::encode_rule`] (the same layout with another scale rounding). `None` unless the
/// rule is a float rule the layout can run, the condition [`FormatCodec::coded`] checks for the host.
#[must_use]
pub fn block_encode_msl_by_rule(layout: &BlockLayout, rule: &EncodeRule, name: &str) -> Option<String> {
    FormatCodec::coded(layout, *rule)?;
    float_block_encode_msl(layout, rule, name)
}

fn grid_literal(layout: &BlockLayout) -> String {
    let count = 1u32 << code_bits(layout);
    (0..count)
        .map(|raw| {
            let value = float_layout_codec::decode(raw, layout.element.code);
            if value.is_finite() { float_literal(value) } else { String::from("INFINITY") }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn float_helpers_msl(layout: &BlockLayout, rule: &EncodeRule, name: &str) -> String {
    let mut text = layout_store_msl(layout.scale.layout, &format!("{name}_scale_store"));
    text.push_str(&layout_load_msl(layout.scale.layout, &format!("{name}_scale_load")));
    if matches!(rule.rounding, CodeRounding::NearestEvenQuotient) {
        text.push_str(&layout_store_msl(layout.element.code, &format!("{name}_code_store")));
    } else {
        text.push_str(&format!(
            "\nconstant float {name}_grid[{}] = {{ {} }};\n",
            1u32 << code_bits(layout),
            grid_literal(layout)
        ));
    }
    text.push_str(&format!(
        "
static inline int {name}_floor_log2(uint bits) {{
    uint exponent = (bits >> 23u) & 0xffu;
    if (exponent == 0u) {{
        return 31 - (int)clz(bits & 0x7fffffu) - 149;
    }}
    return (int)exponent - 127;
}}

static inline float {name}_power_of_two(int exponent) {{
    int clamped = clamp(exponent, -149, 127);
    if (clamped >= -126) {{
        return as_type<float>((uint)(clamped + 127) << 23u);
    }}
    return as_type<float>(1u << (uint)(clamped + 149));
}}

static inline int {name}_ceil_log2(uint bits) {{
    uint fraction = bits & 0x7fffffu;
    bool is_power = ((bits >> 23u) & 0xffu) == 0u ? ((fraction & (fraction - 1u)) == 0u) : (fraction == 0u);
    return {name}_floor_log2(bits) + (is_power ? 0 : 1);
}}
"
    ));
    text
}

fn float_scale_msl(layout: &BlockLayout, rule: &EncodeRule, name: &str, reach: f32) -> String {
    let reach_literal = float_literal(reach);
    let target = match rule.scale_rounding {
        ScaleRounding::Stored => format!("metal::precise::divide(peak, {reach_literal})"),
        ScaleRounding::PowerOfTwoFloor => format!(
            "(peak > 0.0f) ? {name}_power_of_two({name}_floor_log2(as_type<uint>(peak)) - {}) : 0.0f",
            reach_floor_log2(reach)
        ),
        ScaleRounding::PowerOfTwoCeil => format!(
            "(peak > 0.0f) ? {name}_power_of_two({name}_ceil_log2(as_type<uint>(metal::precise::divide(peak, {reach_literal})))) : 0.0f"
        ),
    };
    let width = u32::from(layout.scale.layout.width());
    let mask = u32::MAX >> (32 - width);
    format!(
        "        float target = {target};\n        uint scale_bits = {name}_scale_store(target) & {mask:#x}u;\n        float scale = {name}_scale_load(scale_bits);\n"
    )
}

fn reach_floor_log2(reach: f32) -> i32 {
    let bits = reach.to_bits();
    ((bits >> 23) & 0xff) as i32 - 127
}

fn float_codes_msl(layout: &BlockLayout, rule: &EncodeRule, name: &str) -> String {
    let run_elements = layout.run_elements();
    let count = 1u32 << code_bits(layout);
    match rule.rounding {
        CodeRounding::NearestCodeValue => format!(
            "        for (uint index = 0u; index < {run_elements}u; ++index) {{\n            float value = input[base + index];\n            uint best = 0u;\n            float best_error = fabs({name}_grid[0] * scale - value);\n            for (uint code = 1u; code < {count}u; ++code) {{\n                float error = fabs({name}_grid[code] * scale - value);\n                if (error < best_error) {{\n                    best = code;\n                    best_error = error;\n                }}\n            }}\n            codes[base + index] = (uchar)best;\n        }}\n"
        ),
        _ => format!(
            "        for (uint index = 0u; index < {run_elements}u; ++index) {{\n            float value = input[base + index];\n            float quotient = (scale == 0.0f) ? 0.0f : metal::precise::divide(value, scale);\n            codes[base + index] = (uchar){name}_code_store(quotient);\n        }}\n"
        ),
    }
}

fn float_scale_bytes_msl(layout: &BlockLayout) -> String {
    let bytes = layout.scale.layout.width().div_ceil(8);
    let offset = layout.scale.byte_offset;
    let stride = layout.scale_runs.map_or(0, |runs| runs.stride_bytes);
    let mut text = String::new();
    for step in 0..bytes {
        text.push_str(&format!(
            "        bytes[{offset}u + run * {stride}u + {step}u] |= (uchar)((scale_bits >> {}u) & 0xffu);\n",
            8 * u32::from(step)
        ));
    }
    text
}

/// Renders the encode of a float-coded block: each scale run's peak, the scale the rule's
/// [`ScaleRounding`] makes of it, then each code by the rule's [`CodeRounding`]. The host loop is
/// `proxima_numeric::encode::encode_rule_blocks`; the `f32` operations below are its operations in
/// the same order.
fn float_block_encode_msl(layout: &BlockLayout, rule: &EncodeRule, name: &str) -> Option<String> {
    let reach = float_reach(layout)?;
    let helpers = float_helpers_msl(layout, rule, name);
    let scale = float_scale_msl(layout, rule, name, reach);
    let codes = float_codes_msl(layout, rule, name);
    let scale_bytes = float_scale_bytes_msl(layout);
    let scatter = scatter_msl(layout);
    let (elements, bytes, runs, run_elements) =
        (layout.elements, layout.block_bytes, layout.runs(), layout.run_elements());
    Some(format!(
        "{helpers}
static inline void {name}(device const float *input, device uchar *block) {{
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    uchar bytes[{bytes}];
    for (uint position = 0u; position < {bytes}u; ++position) {{
        bytes[position] = 0;
    }}
    uchar codes[{elements}];
    for (uint run = 0u; run < {runs}u; ++run) {{
        uint base = run * {run_elements}u;
        float peak = 0.0f;
        for (uint index = 0u; index < {run_elements}u; ++index) {{
            float magnitude = as_type<float>(as_type<uint>(input[base + index]) & {F32_MAGNITUDE_MASK:#x}u);
            peak = (peak < magnitude) ? magnitude : peak;
        }}
{scale}{scale_bytes}{codes}    }}
{scatter}    for (uint position = 0u; position < {bytes}u; ++position) {{
        block[position] = bytes[position];
    }}
}}
"
    ))
}

fn float_reach(layout: &BlockLayout) -> Option<f32> {
    proxima_numeric::encode::reach(layout).ok()
}

fn float_literal(value: f32) -> String {
    format!("{value:?}f")
}

fn fit_msl(layout: &BlockLayout, rule: &EncodeRule) -> Option<String> {
    let elements = layout.elements;
    let divisor = float_literal(rule.divisor as f32);
    let (statistic, inverse) = match rule.statistic {
        ScaleStatistic::SignedPeak => (
            format!(
                "    float peak_magnitude = 0.0f;\n    float peak_value = 0.0f;\n    for (uint index = 0u; index < {elements}u; ++index) {{\n        float value = input[index];\n        float magnitude = as_type<float>(as_type<uint>(value) & {F32_MAGNITUDE_MASK:#x}u);\n        if (peak_magnitude < magnitude) {{\n            peak_magnitude = magnitude;\n            peak_value = value;\n        }}\n    }}\n    float scale = metal::precise::divide(peak_value, {divisor});\n"
            ),
            true,
        ),
        ScaleStatistic::PeakMagnitude => (
            format!(
                "    uint top = 0u;\n    for (uint index = 0u; index < {elements}u; ++index) {{\n        top = max(top, as_type<uint>(input[index]) & {F32_MAGNITUDE_MASK:#x}u);\n    }}\n    float peak = as_type<float>(top);\n    if (top > {F32_EXPONENT_MASK:#x}u) {{\n        peak = 0.0f;\n        for (uint index = 0u; index < {elements}u; ++index) {{\n            float magnitude = as_type<float>(as_type<uint>(input[index]) & {F32_MAGNITUDE_MASK:#x}u);\n            peak = (peak > magnitude) ? peak : magnitude;\n        }}\n    }}\n    float scale = metal::precise::divide(peak, {divisor});\n"
            ),
            true,
        ),
        ScaleStatistic::MeanMagnitude => (
            format!(
                "    float total = 0.0f;\n    for (uint index = 0u; index < {elements}u; ++index) {{\n        total += as_type<float>(as_type<uint>(input[index]) & {F32_MAGNITUDE_MASK:#x}u);\n    }}\n    float scale = metal::precise::divide(metal::precise::divide(total, {}), {divisor});\n",
                float_literal(f32::from(layout.elements))
            ),
            false,
        ),
        ScaleStatistic::MinMax(_) => return None,
    };
    let inverse = if inverse {
        "    float inverse = (scale == 0.0f) ? 0.0f : metal::precise::divide(1.0f, scale);\n"
    } else {
        "    float inverse = 0.0f;\n"
    };
    Some(format!("{statistic}{inverse}"))
}

fn codes_msl(layout: &BlockLayout, rule: &EncodeRule, name: &str) -> Option<String> {
    let elements = layout.elements;
    let bias = i32::from(layout.element.bias);
    let degenerate = degenerate_msl(rule, name);
    let zero_code = rule.zero_code;
    let code = match rule.rounding {
        CodeRounding::HalfUpThenTruncate => {
            let clamped = match rule.clamp {
                CodeClamp::Both => format!("clamp(raw, {}, {})", rule.lowest + bias, rule.highest + bias),
                CodeClamp::Upper => format!("min(raw, {})", rule.highest + bias),
                CodeClamp::Neither => String::from("raw"),
            };
            format!(
                "            int raw = {name}_to_int(input[index] * inverse + {offset});\n            code = (uint){clamped} & 0xffu;\n",
                offset = float_literal(bias as f32 + 0.5)
            )
        }
        CodeRounding::NearestAwayFromZero => {
            let clamped = match rule.clamp {
                CodeClamp::Both => format!("clamp(rounded, {}, {})", rule.lowest, rule.highest),
                CodeClamp::Upper => format!("min(rounded, {})", rule.highest),
                CodeClamp::Neither => String::from("rounded"),
            };
            let raw = if is_twos_complement(layout) {
                format!("(uint){clamped} & {:#x}u", (1u32 << code_bits(layout)) - 1)
            } else {
                format!("(uint)({clamped} + {bias})")
            };
            format!(
                "            int rounded = {name}_to_int(round(input[index] * inverse));\n            code = ({raw}) & 0xffu;\n"
            )
        }
        CodeRounding::NonNegativeBit => String::from(
            "            code = ((as_type<uint>(input[index]) & 0x7fffffffu) <= 0x7f800000u && input[index] >= 0.0f) ? 1u : 0u;\n",
        ),
        CodeRounding::NearestEven => return None,
        // a float code never reaches the integer loop: `block_encode_msl` renders it separately
        CodeRounding::NearestCodeValue | CodeRounding::NearestEvenQuotient => String::new(),
    };
    Some(format!(
        "{degenerate}    uchar codes[{elements}];\n    for (uint index = 0u; index < {elements}u; ++index) {{\n        uint code = {zero_code}u;\n        if (!degenerate) {{\n{code}        }}\n        codes[index] = (uchar)code;\n    }}\n"
    ))
}

fn degenerate_msl(rule: &EncodeRule, name: &str) -> String {
    if matches!(rule.rounding, CodeRounding::NonNegativeBit) {
        return String::from("    bool degenerate = false;\n");
    }
    format!(
        "    float stored = {name}_scale_load({name}_scale_store(scale));\n    float stored_inverse = metal::precise::divide(1.0f, stored);\n    bool scale_is_finite = (as_type<uint>(scale) & {F32_EXPONENT_MASK:#x}u) != {F32_EXPONENT_MASK:#x}u;\n    bool inverse_is_finite = (as_type<uint>(stored_inverse) & {F32_EXPONENT_MASK:#x}u) != {F32_EXPONENT_MASK:#x}u;\n    bool degenerate = scale_is_finite && (stored == 0.0f || !inverse_is_finite);\n"
    )
}

fn scatter_msl(layout: &BlockLayout) -> String {
    let element = &layout.element;
    let low_bits = u32::from(element.low.bits);
    let mut planes = alloc::vec![(element.low, 0u32)];
    if let Some(high) = element.high {
        planes.push((high, low_bits));
    }
    let mut text = String::new();
    for index in 0..usize::from(layout.elements) {
        for (plane, lift) in &planes {
            let width = u32::from(plane.bits);
            let address = plane.address(index);
            let (byte, shift) = (address / 8, (address % 8) as u32);
            let mask = (1u32 << width) - 1;
            text.push_str(&format!(
                "    bytes[{byte}] |= (uchar)((((uint)codes[{index}] >> {lift}u) & {mask:#x}u) << {shift}u);\n"
            ));
            if shift + width > 8 {
                text.push_str(&format!(
                    "    bytes[{}] |= (uchar)(((((uint)codes[{index}] >> {lift}u) & {mask:#x}u) << {shift}u) >> 8u);\n",
                    byte + 1
                ));
            }
        }
    }
    text
}

fn scale_bytes_msl(layout: &BlockLayout, name: &str) -> String {
    let width = u32::from(layout.scale.layout.width());
    let bytes = width.div_ceil(8) as usize;
    let offset = usize::from(layout.scale.byte_offset);
    let mask = u32::MAX >> (32 - width);
    let mut text = format!("    uint scale_bits = {name}_scale_store(scale) & {mask:#x}u;\n");
    for step in 0..bytes {
        text.push_str(&format!(
            "    bytes[{}] |= (uchar)((scale_bits >> {}u) & 0xffu);\n",
            offset + step,
            8 * step
        ));
    }
    text
}

fn sum_msl(layout: &BlockLayout, rule: &EncodeRule, name: &str) -> String {
    if !rule.stored_sum {
        return String::new();
    }
    let bias = i32::from(layout.element.bias);
    let at = usize::from(layout.scale.byte_offset) + layout.scale.layout.width().div_ceil(8) as usize;
    let elements = layout.elements;
    let value_of_raw = if is_twos_complement(layout) {
        let spare = 32 - code_bits(layout);
        format!("((int)((uint)codes[index] << {spare}u)) >> {spare}u")
    } else {
        format!("(int)codes[index] - {bias}")
    };
    let mut text = format!(
        "    int total_codes = 0;\n    for (uint index = 0u; index < {elements}u; ++index) {{\n        total_codes += {value_of_raw};\n    }}\n    uint sum_bits = {name}_sum_store((float)total_codes * scale);\n"
    );
    for step in 0..SUM_BYTES {
        text.push_str(&format!(
            "    bytes[{}] = (uchar)((sum_bits >> {}u) & 0xffu);\n",
            at + step,
            8 * step
        ));
    }
    text
}

/// Every name the renderer substitutes, so a caller can see the helpers a rendering defines.
#[must_use]
pub fn encode_helper_names(name: &str) -> Vec<String> {
    ["_to_int", "_scale_store", "_scale_load", "_sum_store"]
        .iter()
        .map(|suffix| format!("{name}{suffix}"))
        .collect()
}
