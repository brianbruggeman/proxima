//! The one MSL emitter over a [`FloatLayout`]: load (decode) and store (encode) from the fields.
//!
//! Teaching pointer: `proxima_gguf::quant::float_layout::{decode, encode}` is the CPU algorithm
//! and this module renders the same integer steps as MSL, so the two agree code for code. A layout
//! whose fields equal a Metal hardware type (`float`, `half`, `bfloat`) renders through that type's
//! conversion; every other layout renders as field arithmetic whose only inputs are the numeric
//! constants of the struct. Compose the pieces with [`layout_unpack_msl`] for a flat element
//! accessor, or call [`layout_load_msl`] and [`layout_store_msl`] directly when the caller owns the
//! element addressing (the scaled-block decoders in `scaled_float_render` do).

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_primitives::float_layout::{FloatLayout, SignEncoding, Specials};

pub(crate) const F32_FRACTION_BITS: u32 = 23;
pub(crate) const F32_INFINITY_BITS: u32 = 0x7f80_0000;
pub(crate) const F32_QUIET_BIT: u32 = 1 << 22;
pub(crate) const F32_ABS_MASK: u32 = 0x7fff_ffff;

const HARDWARE_TYPES: [(u8, u8, Specials, i16, &str); 3] = [
    (8, 23, Specials::InfinityOnOverflow, 127, "float"),
    (5, 10, Specials::InfinityOnOverflow, 15, "half"),
    (8, 7, Specials::InfinityOnOverflow, 127, "bfloat"),
];

/// The Metal type whose fields equal `layout`'s, compared field by field and never by name.
#[must_use]
pub fn hardware_type(layout: &FloatLayout) -> Option<&'static str> {
    HARDWARE_TYPES
        .iter()
        .find(|(exponent_bits, mantissa_bits, specials, bias, _)| {
            *exponent_bits == layout.exponent_bits
                && *mantissa_bits == layout.mantissa_bits
                && *specials == layout.specials
                && *bias == layout.bias()
                && layout.sign == SignEncoding::Magnitude
        })
        .map(|entry| entry.4)
}

pub(crate) const fn exponent_ones(layout: &FloatLayout) -> u32 {
    (1u32 << layout.exponent_bits) - 1
}

pub(crate) const fn mantissa_mask(layout: &FloatLayout) -> u32 {
    ((1u64 << layout.mantissa_bits) - 1) as u32
}

pub(crate) const fn sign_shift(layout: &FloatLayout) -> u32 {
    layout.exponent_bits as u32 + layout.mantissa_bits as u32
}

pub(crate) const fn infinity_code(layout: &FloatLayout) -> u32 {
    exponent_ones(layout) << layout.mantissa_bits
}

pub(crate) const fn max_normal_code(layout: &FloatLayout) -> u32 {
    let top = match layout.specials {
        Specials::None => infinity_code(layout),
        Specials::SaturateOnOverflow | Specials::InfinityOnOverflow => {
            infinity_code(layout) - (1u32 << layout.mantissa_bits)
        }
    };
    top | mantissa_mask(layout)
}

pub(crate) const fn overflow_code(layout: &FloatLayout) -> u32 {
    match layout.specials {
        Specials::InfinityOnOverflow => infinity_code(layout),
        Specials::None | Specials::SaturateOnOverflow => max_normal_code(layout),
    }
}

pub(crate) const fn quiet_bit(layout: &FloatLayout) -> u32 {
    (1u32 << layout.mantissa_bits) >> 1
}

pub(crate) const fn reserves_all_ones(layout: &FloatLayout) -> bool {
    !matches!(layout.specials, Specials::None)
}

fn integer_scale(layout: &FloatLayout) -> f32 {
    let exponent = 1 - i32::from(layout.bias()) - i32::from(layout.mantissa_bits);
    f32::from_bits(((exponent + 127).clamp(1, 254) as u32) << 23)
}

fn integer_load_msl(layout: &FloatLayout, name: &str) -> String {
    let mantissa_bits = layout.mantissa_bits;
    let mantissa_mask = (1u64 << mantissa_bits) - 1;
    let scale = integer_scale(layout);
    let value = match layout.sign {
        SignEncoding::Absent => "(float)(long)mantissa".to_string(),
        SignEncoding::Magnitude => format!(
            "((code >> {mantissa_bits}u) & 1u) == 0u ? (float)(long)mantissa : -(float)(long)mantissa"
        ),
        SignEncoding::TwosComplement => format!(
            "(float)((long)mantissa - (long)(((code >> {mantissa_bits}u) & 1u) << {mantissa_bits}u))"
        ),
    };
    format!(
        "
static inline float {name}(uint code) {{
    uint mantissa = code & {mantissa_mask:#x}u;
    return ({value}) * {scale:e}f;
}}
"
    )
}

fn integer_store_msl(layout: &FloatLayout, name: &str) -> String {
    let mantissa_bits = layout.mantissa_bits;
    let mantissa_mask = (1u64 << mantissa_bits) - 1;
    let unscale = 1.0 / integer_scale(layout);
    let body = match layout.sign {
        SignEncoding::Absent => {
            format!("    return (uint)clamp(rint(scaled), 0.0f, {mantissa_mask}.0f);")
        }
        SignEncoding::Magnitude => format!(
            "    uint magnitude = (uint)min(rint(fabs(scaled)), {mantissa_mask}.0f);\n    return magnitude | ((as_type<uint>(scaled) >> 31u) << {mantissa_bits}u);"
        ),
        SignEncoding::TwosComplement => format!(
            "    int clamped = (int)clamp(rint(scaled), -{negative_limit}.0f, {mantissa_mask}.0f);\n    return (uint)clamped & {full_mask:#x}u;",
            negative_limit = 1u64 << mantissa_bits,
            full_mask = (1u64 << (mantissa_bits + 1)) - 1
        ),
    };
    format!(
        "
static inline uint {name}(float value) {{
    float scaled = value * {unscale:e}f;
{body}
}}
"
    )
}

/// Renders `static inline float name(uint code)`: the value of the `layout.width()`-bit `code`.
#[must_use]
pub fn layout_load_msl(layout: FloatLayout, name: &str) -> String {
    if layout.exponent_bits == 0 {
        return integer_load_msl(&layout, name);
    }
    match hardware_type(&layout) {
        Some("float") => format!(
            "\nstatic inline float {name}(uint code) {{\n    return as_type<float>(code);\n}}\n"
        ),
        Some(narrow) => format!(
            "\nstatic inline float {name}(uint code) {{\n    return (float)as_type<{narrow}>((ushort)code);\n}}\n"
        ),
        None => field_load_msl(&layout, name),
    }
}

fn field_load_msl(layout: &FloatLayout, name: &str) -> String {
    let mantissa_bits = layout.mantissa_bits;
    let mantissa_mask = mantissa_mask(layout);
    let exponent_ones = exponent_ones(layout);
    let sign_shift = sign_shift(layout);
    let bias = layout.bias();
    let fraction_shift = F32_FRACTION_BITS - u32::from(mantissa_bits);
    let implicit = 1u64 << mantissa_bits;
    let reserved = field_load_reserved_msl(layout, exponent_ones, fraction_shift);
    format!(
        "
static inline uint {name}_scaled_bits(uint significand, int scale) {{
    if (significand == 0u) {{
        return 0u;
    }}
    int lead = 31 - (int)clz(significand);
    int biased = lead + scale + 127;
    if (biased >= 255) {{
        return {F32_INFINITY_BITS:#010x}u;
    }}
    if (biased >= 1) {{
        return ((uint)biased << {F32_FRACTION_BITS}u) | ((significand << (uint)({F32_FRACTION_BITS} - lead)) & 0x7fffffu);
    }}
    int shift = scale + 149;
    if (shift >= 0) {{
        return significand << (uint)shift;
    }}
    return -shift >= 32 ? 0u : significand >> (uint)(-shift);
}}

static inline float {name}(uint code) {{
    uint mantissa = code & {mantissa_mask:#x}u;
    uint exponent = (code >> {mantissa_bits}u) & {exponent_ones:#x}u;
    uint sign = ((code >> {sign_shift}u) & 1u) << 31u;
{reserved}    uint significand = exponent == 0u ? mantissa : ({implicit:#x}u | mantissa);
    int scale = exponent == 0u ? (1 - ({bias}) - {mantissa_bits}) : ((int)exponent - ({bias}) - {mantissa_bits});
    return as_type<float>(sign | {name}_scaled_bits(significand, scale));
}}
"
    )
}

fn field_load_reserved_msl(
    layout: &FloatLayout,
    exponent_ones: u32,
    fraction_shift: u32,
) -> String {
    if !reserves_all_ones(layout) {
        return String::new();
    }
    format!(
        "    if (exponent == {exponent_ones:#x}u) {{
        return as_type<float>(sign | {F32_INFINITY_BITS:#010x}u | (mantissa == 0u ? 0u : ({F32_QUIET_BIT:#x}u | (mantissa << {fraction_shift}u))));
    }}
"
    )
}

/// Renders `static inline uint name(float value)`: the `layout.width()`-bit code nearest `value`,
/// ties to the even code, overflow and NaN under the layout's [`Specials`].
#[must_use]
pub fn layout_store_msl(layout: FloatLayout, name: &str) -> String {
    if layout.exponent_bits == 0 {
        return integer_store_msl(&layout, name);
    }
    match hardware_type(&layout) {
        Some("float") => format!(
            "\nstatic inline uint {name}(float value) {{\n    return as_type<uint>(value);\n}}\n"
        ),
        Some(narrow) => format!(
            "\nstatic inline uint {name}(float value) {{\n    return (uint)as_type<ushort>(({narrow})value);\n}}\n"
        ),
        None => field_store_msl(&layout, name),
    }
}

fn field_store_nan_msl(layout: &FloatLayout) -> String {
    if !reserves_all_ones(layout) {
        return format!("        return {:#x}u;\n", max_normal_code(layout));
    }
    format!(
        "        return sign | {:#x}u | (((bits & 0x7fffffu) >> {}u) | {:#x}u);\n",
        infinity_code(layout),
        F32_FRACTION_BITS - u32::from(layout.mantissa_bits),
        quiet_bit(layout)
    )
}

fn field_store_msl(layout: &FloatLayout, name: &str) -> String {
    let mantissa_bits = layout.mantissa_bits;
    let sign_shift = sign_shift(layout);
    let bias = layout.bias();
    let exponent_ones = exponent_ones(layout);
    let overflow = overflow_code(layout);
    let value_shift = F32_FRACTION_BITS - u32::from(mantissa_bits);
    let nan = field_store_nan_msl(layout);
    format!(
        "
static inline uint {name}_round_shift(uint value, uint shift) {{
    if (shift == 0u) {{
        return value;
    }}
    if (shift > {shift_limit}u) {{
        return 0u;
    }}
    uint quotient = value >> shift;
    uint remainder = value & ((1u << shift) - 1u);
    uint halfway = 1u << (shift - 1u);
    return (remainder > halfway || (remainder == halfway && (quotient & 1u) == 1u)) ? quotient + 1u : quotient;
}}

static inline uint {name}(float value) {{
    uint bits = as_type<uint>(value);
    uint sign = (bits >> 31u) << {sign_shift}u;
    uint abs_bits = bits & {F32_ABS_MASK:#x}u;
    if (abs_bits > {F32_INFINITY_BITS:#010x}u) {{
{nan}    }}
    if ((abs_bits >> {F32_FRACTION_BITS}u) == 0xffu) {{
        return sign | {overflow:#x}u;
    }}
    uint exponent_field = abs_bits >> {F32_FRACTION_BITS}u;
    uint significand = exponent_field == 0u ? abs_bits : ((abs_bits & 0x7fffffu) | 0x800000u);
    int source_exponent = exponent_field == 0u ? 1 : (int)exponent_field;
    int target = source_exponent - 127 + ({bias});
    if (target > {exponent_ones}) {{
        return sign | {overflow:#x}u;
    }}
    uint base_exponent = target < 1 ? 0u : (uint)(target - 1);
    uint extra_shift = target < 1 ? (uint)(1 - target) : 0u;
    uint code = (base_exponent << {mantissa_bits}u) + {name}_round_shift(significand, {value_shift}u + extra_shift);
    return sign | min(code, {overflow:#x}u);
}}
",
        shift_limit = F32_FRACTION_BITS + 1,
    )
}

/// Renders `static inline float name(device const uchar *payload, uint index)`, the flat element
/// accessor: element `index` of a densely packed array of `layout`, low bits first. `None` when the
/// width neither fills whole bytes nor divides a byte, because such an element has no flat address.
#[must_use]
pub fn layout_unpack_msl(layout: FloatLayout, name: &str) -> Option<String> {
    let width = u32::from(layout.width());
    let value_name = format!("{name}_value");
    let code = if width % 8 == 0 {
        let bytes = width / 8;
        let gathered = (0..bytes)
            .map(|byte| format!("((uint)payload[offset + {byte}u] << {}u)", byte * 8))
            .collect::<Vec<_>>()
            .join(" | ");
        format!("    uint offset = index * {bytes}u;\n    uint code = {gathered};\n")
    } else if 8 % width == 0 {
        let mask = (1u32 << width) - 1;
        format!(
            "    uint bit = index * {width}u;\n    uint code = ((uint)payload[bit / 8u] >> (bit % 8u)) & {mask:#x}u;\n"
        )
    } else {
        return None;
    };
    let value = layout_load_msl(layout, &value_name);
    Some(format!(
        "{value}\nstatic inline float {name}(device const uchar *payload, uint index) {{\n{code}    return {value_name}(code);\n}}\n"
    ))
}
