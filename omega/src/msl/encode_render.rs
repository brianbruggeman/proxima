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
use proxima_numeric::encode_rule::{CodeClamp, CodeRounding, EncodeRule, ScaleStatistic, StoredMin};
use proxima_numeric::float_layout::{FloatLayout, SignEncoding};
use proxima_numeric::{FormatCodec, NumericFormat};

use super::layout_render::{layout_load_msl, layout_store_msl};

const F32_EXPONENT_MASK: u32 = 0x7f80_0000;
const F32_MAGNITUDE_MASK: u32 = 0x7fff_ffff;
const SUM_BYTES: usize = 2;

/// A rendered encode: the source of one kernel and the shape it dispatches over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedEncode {
    /// Metal source defining the kernel named `entry`: buffer 0 is the binary32 input, buffer 1 the
    /// packed output; one thread encodes `elements_per_thread` values into `bytes_per_thread` bytes.
    pub source: String,
    /// The kernel function's name.
    pub entry: String,
    /// Elements one thread consumes: the block's element count, or one for a scalar.
    pub elements_per_thread: usize,
    /// Bytes one thread produces.
    pub bytes_per_thread: usize,
}

/// Why `format` has no rendered encode: the property it lacks. Decided from the description alone.
///
/// # Errors
/// The property of the format that makes it unrenderable.
#[must_use = "the refusal names why a format has no rendered encode"]
pub fn encode_kernel_refusal(format: &NumericFormat) -> Result<(), &'static str> {
    let Some(codec) = FormatCodec::for_format(format) else {
        return Err("no host codec resolves the format");
    };
    if !codec.can_encode() {
        return Err("the host codec has no encode to match");
    }
    match format {
        NumericFormat::Scalar(layout) => match layout.width() {
            8 | 16 | 32 => Ok(()),
            _ => Err("the scalar code does not fill a whole byte, half-word or word"),
        },
        NumericFormat::Block(layout) => match layout.encode_rule() {
            Some(rule) => rule_refusal(layout, &rule),
            None => Err("the layout has no encode rule"),
        },
        NumericFormat::Levels(_) => Err("a level format has no encode rule"),
    }
}

/// Why a layout that has an [`EncodeRule`] still has no rendering: the fit the rule names that this
/// emitter does not write (a searched or stored range, sub-block scales, ties to even), or the
/// layout packs independent scale runs.
fn rule_refusal(layout: &BlockLayout, rule: &EncodeRule) -> Result<(), &'static str> {
    if layout.scale_runs.is_some() {
        return Err("the layout packs independent scale runs");
    }
    if matches!(rule.statistic, ScaleStatistic::MinMax(_)) || rule.stored_min != StoredMin::Absent {
        return Err("the layout fits a stored range, which the emitter does not render");
    }
    if rule.sub.is_some() {
        return Err("the layout carries sub-block scales, which the emitter does not render");
    }
    if matches!(rule.rounding, CodeRounding::NearestEven) {
        return Err("the layout rounds ties to even through a magic-number add, which the emitter does not render");
    }
    Ok(())
}

/// Renders the encode kernel of `format`, entry point `{name}_kernel`.
///
/// # Errors
/// [`encode_kernel_refusal`]'s reason.
#[must_use = "the refusal names why a format has no rendered encode"]
pub fn format_encode_kernel_msl(format: &NumericFormat, name: &str) -> Result<RenderedEncode, &'static str> {
    encode_kernel_refusal(format)?;
    match format {
        NumericFormat::Scalar(layout) => Ok(scalar_encode_kernel(*layout, name)),
        NumericFormat::Block(layout) => block_encode_kernel(layout, name).ok_or("the layout has no encode rule"),
        NumericFormat::Levels(_) => Err("a level format has no encode rule"),
    }
}

fn scalar_encode_kernel(layout: FloatLayout, name: &str) -> RenderedEncode {
    let storage = match layout.width() {
        8 => "uchar",
        16 => "ushort",
        _ => "uint",
    };
    let store = layout_store_msl(layout, &format!("{name}_store"));
    let source = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{store}\nkernel void {name}_kernel(device const float *input [[buffer(0)]], device {storage} *output [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{\n    output[gid] = ({storage}){name}_store(input[gid]);\n}}\n"
    );
    RenderedEncode {
        source,
        entry: format!("{name}_kernel"),
        elements_per_thread: 1,
        bytes_per_thread: usize::from(layout.width()) / 8,
    }
}

fn block_encode_kernel(layout: &BlockLayout, name: &str) -> Option<RenderedEncode> {
    let function = block_encode_msl(layout, name)?;
    let elements = usize::from(layout.elements);
    let bytes = usize::from(layout.block_bytes);
    let source = format!(
        "#include <metal_stdlib>\nusing namespace metal;\n{function}\nkernel void {name}_kernel(device const float *input [[buffer(0)]], device uchar *output [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{\n    {name}(input + gid * {elements}u, output + gid * {bytes}u);\n}}\n"
    );
    Some(RenderedEncode {
        source,
        entry: format!("{name}_kernel"),
        elements_per_thread: elements,
        bytes_per_thread: bytes,
    })
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
                "            int raw = {name}(input[index] * inverse + {offset});\n            code = (uint){clamped} & 0xffu;\n",
                name = format!("{name}_to_int"),
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
                "            int rounded = {name}(round(input[index] * inverse));\n            code = ({raw}) & 0xffu;\n",
                name = format!("{name}_to_int")
            )
        }
        CodeRounding::NonNegativeBit => String::from(
            "            code = ((as_type<uint>(input[index]) & 0x7fffffffu) <= 0x7f800000u && input[index] >= 0.0f) ? 1u : 0u;\n",
        ),
        CodeRounding::NearestEven => return None,
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
