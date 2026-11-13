//! The one MSL emitter over a [`FloatLayout`]: load (decode) and store (encode) from the fields.
//!
//! Teaching pointer: `proxima_numeric::float_layout_codec::{decode, encode}` is the CPU algorithm
//! and this module renders the same integer steps as MSL, so the two agree code for code. A layout
//! whose fields equal a Metal hardware type (`float`, `half`, `bfloat`) renders through that type's
//! conversion; every other layout renders as field arithmetic whose only inputs are the numeric
//! constants of the struct. Compose the pieces with [`layout_unpack_msl`] for a flat element
//! accessor, or call [`layout_load_msl`] and [`layout_store_msl`] directly when the caller owns the
//! element addressing (the scaled-block decoders in `scaled_float_render` do).

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use proxima_numeric::block_layout::{BlockLayout, CodeRegion, Coded, Field, Plane};
use proxima_numeric::float_layout::{FloatLayout, SignEncoding, Specials};
use proxima_numeric::{FormatCodec, LevelFormat, LevelGeometry, LevelTable, NumericFormat};

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
        Specials::None | Specials::NanOnly => infinity_code(layout),
        Specials::SaturateOnOverflow | Specials::InfinityOnOverflow => {
            infinity_code(layout) - (1u32 << layout.mantissa_bits)
        }
    };
    let mantissa = match layout.specials {
        Specials::NanOnly => mantissa_mask(layout) - 1,
        _ => mantissa_mask(layout),
    };
    top | mantissa
}

pub(crate) const fn overflow_code(layout: &FloatLayout) -> u32 {
    match layout.specials {
        Specials::InfinityOnOverflow => infinity_code(layout),
        Specials::None | Specials::NanOnly | Specials::SaturateOnOverflow => {
            max_normal_code(layout)
        }
    }
}

pub(crate) const fn quiet_bit(layout: &FloatLayout) -> u32 {
    (1u32 << layout.mantissa_bits) >> 1
}

pub(crate) const fn reserves_all_ones(layout: &FloatLayout) -> bool {
    matches!(
        layout.specials,
        Specials::SaturateOnOverflow | Specials::InfinityOnOverflow
    )
}

fn integer_scale(layout: &FloatLayout) -> f32 {
    let exponent = 1 - i32::from(layout.bias()) - i32::from(layout.mantissa_bits);
    f32::from_bits(((exponent + 127).clamp(1, 254) as u32) << 23)
}

fn integer_load_msl(layout: &FloatLayout, name: &str) -> String {
    let mantissa_bits = layout.mantissa_bits;
    let mantissa_mask = (1u64 << mantissa_bits) - 1;
    let scale = integer_scale(layout);
    // 64-bit integer arithmetic here slowed the q8_0 k-cache load on metal, so the wide type is used only when 32 bits cannot hold the signed difference; the q8_0 arm of cached_attention_decode_split_times_a_binary16_k_cache_against_a_q4_0_k_cache (omega/tests/cached_attention_decode_split_parity.rs, feature instrument) prints that load's per-step gpu time, and a_rendered_wide_integer_load_equals_the_host_decode_bit_for_bit_on_boundary_codes (layout_render_tests.rs) pins the long arm
    let signed = if mantissa_bits <= 30 { "int" } else { "long" };
    let value = match layout.sign {
        SignEncoding::Absent => "(float)mantissa".to_string(),
        SignEncoding::Magnitude => format!(
            "((code >> {mantissa_bits}u) & 1u) == 0u ? (float)mantissa : -(float)mantissa"
        ),
        SignEncoding::TwosComplement => format!(
            "(float)(({signed})mantissa - ({signed})(((code >> {mantissa_bits}u) & 1u) << {mantissa_bits}u))"
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
    // no mantissa field means no subnormal range: field 0 is itself the lowest power of two
    // (`FloatLayout::lowest_exponent`, OCP MX v1.0 Table 7), never zero
    let (zero_exponent_significand, zero_exponent_scale) = if mantissa_bits == 0 {
        (String::from("1u"), layout.lowest_exponent().to_string())
    } else {
        (String::from("mantissa"), format!("(1 - ({bias}) - {mantissa_bits})"))
    };
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
{reserved}    uint significand = exponent == 0u ? {zero_exponent_significand} : ({implicit:#x}u | mantissa);
    int scale = exponent == 0u ? {zero_exponent_scale} : ((int)exponent - ({bias}) - {mantissa_bits});
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
    if layout.specials == Specials::NanOnly {
        return format!(
            "    if (exponent == {exponent_ones:#x}u && mantissa == {:#x}u) {{\n        return as_type<float>(sign | {F32_INFINITY_BITS:#010x}u | {F32_QUIET_BIT:#x}u | (mantissa << {fraction_shift}u));\n    }}\n",
            mantissa_mask(layout),
            fraction_shift = F32_FRACTION_BITS - u32::from(layout.mantissa_bits)
        );
    }
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
    if layout.specials == Specials::NanOnly {
        return format!(
            "        return sign | {:#x}u | {:#x}u;\n",
            infinity_code(layout),
            mantissa_mask(layout)
        );
    }
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

fn plane_expression(plane: &Plane, index: &str, binding: &str) -> String {
    let mask = (1u32 << plane.bits) - 1;
    let steps = plane
        .digits
        .iter()
        .filter(|digit| digit.radix > 1)
        .map(|digit| {
            format!(
                "    {binding}_address += ({binding}_rest % {}u) * {}u;\n    {binding}_rest /= {}u;\n",
                digit.radix, digit.stride_bits, digit.radix
            )
        })
        .collect::<String>();
    format!(
        "    uint {binding}_rest = {index};\n    uint {binding}_address = {}u;\n{steps}    uint {binding} = ((uint)block[{binding}_address / 8u] >> ({binding}_address % 8u)) & {mask:#x}u;\n",
        plane.base_bits
    )
}

fn coded_expression(coded: &Coded, index: &str, binding: &str, helper: &str) -> String {
    let (mut text, code) = coded_assembly(coded, index, binding);
    text.push_str(&format!("    float {binding} = {helper}({code});\n"));
    text
}

/// The statements that assemble `coded`'s integer code for element `index`, and the expression that names it.
fn coded_assembly(coded: &Coded, index: &str, binding: &str) -> (String, String) {
    if coded.packed_radix > 1 && coded.region_count > 0 {
        let code_name = format!("{binding}_code");
        let mut text = format!("    uint {code_name} = 0u;\n");
        for (region_index, region) in coded
            .regions
            .iter()
            .take(usize::from(coded.region_count))
            .enumerate()
        {
            text.push_str(&radix_region_expression(
                region,
                coded.packed_radix,
                index,
                &code_name,
                region_index,
                if region_index == 0 { "if" } else { "else if" },
            ));
        }
        return (text, code_name);
    }

    let mut text = plane_expression(&coded.low, index, &format!("{binding}_low"));
    let mut code = format!("{binding}_low");
    if let Some(high) = &coded.high {
        text.push_str(&plane_expression(high, index, &format!("{binding}_high")));
        code = format!("({binding}_low | ({binding}_high << {}u))", coded.low.bits);
    }
    (text, code)
}

fn radix_region_expression(
    region: &CodeRegion,
    radix: u8,
    index: &str,
    binding: &str,
    region_index: usize,
    condition: &str,
) -> String {
    let start = u32::from(region.start);
    let end = start + u32::from(region.elements);
    let local = format!("{binding}_region_{region_index}_index");
    let packed = format!("{binding}_region_{region_index}_packed");
    let address_map = plane_expression(&region.source, &local, &packed);
    let axis = usize::from(region.digit_axis);
    let divisor = region
        .source
        .digits
        .iter()
        .take(axis)
        .fold(1u32, |product, digit| product * u32::from(digit.radix));
    let lane_radix = u32::from(region.source.digits[axis].radix);
    let storage_bits = u32::from(region.source.bits);
    let storage_mask = (1u32 << storage_bits) - 1;
    format!(
        "    {condition} ({index} >= {start}u && {index} < {end}u) {{\n        uint {local} = {index} - {start}u;\n{address_map}        uint {binding}_region_{region_index}_lane = ({local} / {divisor}u) % {lane_radix}u;\n        uint {binding}_region_{region_index}_factor = 1u;\n        for (uint power = 0u; power < {binding}_region_{region_index}_lane; ++power) {{ {binding}_region_{region_index}_factor *= {radix}u; }}\n        uint {binding}_region_{region_index}_residue = ({packed} * {binding}_region_{region_index}_factor) & {storage_mask:#x}u;\n        {binding} = ({binding}_region_{region_index}_residue * {radix}u) >> {storage_bits}u;\n    }}\n"
    )
}

fn field_expression(field: &Field, binding: &str, helper: &str) -> String {
    let offset = field.byte_offset;
    let bytes = usize::from(field.layout.width()) / 8;
    let gathered = (0..bytes)
        .map(|byte| {
            format!(
                "((uint)block[{}u] << {}u)",
                usize::from(offset) + byte,
                byte * 8
            )
        })
        .collect::<Vec<_>>()
        .join(" | ");
    format!("    float {binding} = {helper}({gathered});\n")
}

/// The block layout a cached-attention kernel reads a cache stored as `format` through, `None`
/// when the format is not one. [`block_unpack_msl`] must render the layout whole (it has no
/// per-run scale fields), the host codec must resolve it and have an encode instantiation (the
/// device cache is filled by narrowing f32 rows with `FormatCodec::encode`, so a decode-only codec
/// would be admitted and fail at the first commit), and four consecutive elements must stay inside
/// one block so a four-wide load never straddles two. Decided from the layout's own properties,
/// never from a format name; [`block_k_refusal`] names the property that failed.
#[must_use]
pub fn readable_block_layout(format: &NumericFormat) -> Option<&BlockLayout> {
    block_k_refusal(format).ok()
}

/// The layout and the Metal storage type (`uchar`, `ushort` or `uint`) a cache of `format` is
/// addressed through when it is a scalar whose code fills whole bytes, so a four-wide load is
/// four contiguous codes and [`layout_load_msl`] decodes each. `None` for a block or level
/// format, and for a width with no flat address (a 6-bit code straddles bytes). The host codec
/// must also resolve and encode it, because the device cache is filled by encoding rows.
/// Decided from the layout's width, never from a format name.
#[must_use]
pub fn readable_scalar_cache(format: &NumericFormat) -> Option<(FloatLayout, &'static str)> {
    let NumericFormat::Scalar(layout) = format else {
        return None;
    };
    let storage = match layout.width() {
        8 => "uchar",
        16 => "ushort",
        32 => "uint",
        _ => return None,
    };
    FormatCodec::for_format(format)
        .is_some_and(|codec| codec.can_encode())
        .then_some((*layout, storage))
}

/// [`readable_block_layout`] with the reason: `Err` names the property the format lacks.
///
/// # Errors
/// The property of the format that makes it unreadable as a block K.
#[must_use = "the refusal names why a format cannot be a block K"]
pub fn block_k_refusal(format: &NumericFormat) -> Result<&BlockLayout, &'static str> {
    let NumericFormat::Block(layout) = format else {
        return Err("not a block layout");
    };
    if layout.scale_runs.is_some() {
        return Err("the layout packs independent scale runs, which block_unpack_msl does not render");
    }
    if layout.elements % 4 != 0 {
        return Err("the layout's block is not a whole number of four-wide loads");
    }
    let Some(codec) = FormatCodec::for_format(format) else {
        return Err("no host codec resolves the layout");
    };
    if !codec.can_encode() {
        return Err("the host codec has no encode, so the device cache cannot be filled");
    }
    Ok(layout)
}

/// Renders `static inline float name(device const uchar *block, uint index)`: element `index` of
/// one block of `layout`, assembled from its planes and scaled by its fields and sub-block codes.
/// The helpers `name_*` are the layout loads of the code, scale and min layouts.
#[must_use]
pub fn block_unpack_msl(layout: &BlockLayout, name: &str) -> String {
    let element = block_element_msl(layout, name);
    let run = block_run_msl(layout, name);
    format!("{element}{run}")
}

/// [`block_unpack_msl`] without the four-wide `name_run` accessor: the scalar `name(block, index)`
/// and its `name_*` load helpers only. A format whose callers read one element at a time (the
/// legacy `q4_0_element`, `q5_0_element`, `q5_1_element` accessors) renders this, so no unused
/// four-wide body enters the translation unit.
#[must_use]
pub fn block_element_msl(layout: &BlockLayout, name: &str) -> String {
    element_msl(layout, Codes::Affine, name)
}

/// The property of a level format that keeps [`level_block_element_msl`] from rendering it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("level element renderer cannot render this format: {0}")]
pub struct LevelRenderRefusal(pub &'static str);

/// Renders `static inline float name(device const uchar *block, uint index)` for a level-set
/// format: [`block_element_msl`]'s walk of the block's planes and sub-block codes, with the code
/// resolved to a value through the format's [`LevelTable`] instead of the layout's affine map.
///
/// An explicit ladder (IQ4_NL, IQ4_XS) becomes a constant table indexed by the code; a linear
/// ladder is its `zero_point + step * code`; a signed grid (IQ3_S) becomes a constant table of
/// packed magnitudes indexed by the code, a lane picked by `index % lanes`, and a sign read from the
/// sign plane. The sub-block scale is `scale_multiplier * code - bias`, as `proxima_numeric::decode`
/// computes it. Teaching pointer: this is the renderer the tiled GEMM's `tiled_decode` description
/// and the cooperative kernel's `iq4_xs_element` / `iq3_s_element` accessors both call.
///
/// # Errors
/// The level table or layout property the renderer does not cover (grids with a sign table, 64-bit
/// grids, delta grids, scale runs, a block min).
#[must_use = "the refusal names why the format has no rendered element accessor"]
pub fn level_block_element_msl(
    format: &LevelFormat,
    name: &str,
) -> Result<String, LevelRenderRefusal> {
    let layout = renderable_level_layout(format)?;
    Ok(element_msl(layout, Codes::Levels(&format.levels), name))
}

/// Renders the tiled decode contract `function(device const uchar *block, uint slot, thread half
/// *out)` for a level-set format: the `run` consecutive elements at `slot`, each the same value
/// [`level_block_element_msl`]'s accessor `name` returns for it, with the block scale and the
/// sub-block scale decoded once for the run instead of once per element. Teaching pointer: the
/// accessor stays the scalar reference (the cooperative kernel reads it per element); this is the
/// run form the tiled and expert-grouped GEMM stage weights through, so the scale arithmetic
/// leaves the per-element loop.
///
/// # Errors
/// What [`level_block_element_msl`] refuses, and a layout whose sub-blocks are narrower than
/// `run` (a run would then straddle two sub-block scales).
#[must_use = "the refusal names why the format has no rendered run decode"]
pub fn level_run_decode_msl(
    format: &LevelFormat,
    function: &str,
    name: &str,
    run: u64,
) -> Result<String, LevelRenderRefusal> {
    let layout = renderable_level_layout(format)?;
    let mut head = field_expression(&layout.scale, "scale", &format!("{name}_scale"));
    let mut scale = "scale";
    if let Some(sub) = &layout.sub {
        let span = u64::from(layout.elements / sub.count);
        if span % run != 0 {
            return Err(LevelRenderRefusal("a decode run that straddles two sub-block scales"));
        }
        head.push_str(&format!("    uint sub_block = slot / {span}u;\n"));
        head.push_str(&coded_expression(
            &sub.scale,
            "sub_block",
            "sub_scale_raw",
            &format!("{name}_sub_scale_code"),
        ));
        head.push_str(&format!(
            "    float sub_scale = {} - {}.0f;\n    float effective_scale = scale * sub_scale;\n",
            sub_scale_product(sub.scale_multiplier),
            sub.scale.bias
        ));
        scale = "effective_scale";
    }
    let raw = raw_level_msl(layout, Codes::Levels(&format.levels), name);
    Ok(format!(
        "\nstatic inline void {function}(device const uchar *block, uint slot, thread half *out) {{\n{head}    for (uint run = 0u; run < {run}u; ++run) {{\n        uint index = slot + run;\n{raw}        out[run] = (half)(raw * {scale});\n    }}\n}}\n"
    ))
}

fn sub_scale_product(scale_multiplier: i16) -> String {
    if scale_multiplier == 1 {
        "sub_scale_raw".to_string()
    } else {
        format!("({scale_multiplier}.0f * sub_scale_raw)")
    }
}

fn renderable_level_layout(format: &LevelFormat) -> Result<&BlockLayout, LevelRenderRefusal> {
    let LevelGeometry::Packed(layout) = &format.geometry else {
        return Err(LevelRenderRefusal("codebook geometry has no plane walk"));
    };
    if layout.scale_runs.is_some() || layout.min.is_some() {
        return Err(LevelRenderRefusal("scale runs or a block min"));
    }
    if layout.element.packed_radix > 1 {
        return Err(LevelRenderRefusal("packed-radix codes"));
    }
    match &format.levels {
        LevelTable::Linear { .. } | LevelTable::Explicit(_) => {}
        LevelTable::SignedGrid { sign_table: None, lanes, .. } if *lanes > 0 => {}
        LevelTable::SignedGrid { .. } => {
            return Err(LevelRenderRefusal("a signed grid whose sign is a table index"));
        }
        LevelTable::SignedGrid64 { .. } | LevelTable::DeltaGrid { .. } => {
            return Err(LevelRenderRefusal("64-bit and delta grids"));
        }
    }
    Ok(layout)
}

#[derive(Clone, Copy)]
enum Codes<'a> {
    Affine,
    Levels(&'a LevelTable),
}

fn level_table_msl(levels: &LevelTable, name: &str) -> String {
    match levels {
        LevelTable::Linear { zero_point, step, .. } => format!(
            "\nstatic inline float {name}_code(uint code) {{\n    return (float)(({zero_point}) + ({step}) * (int)code);\n}}\n"
        ),
        LevelTable::Explicit(values) => {
            let entries = values.iter().map(i8::to_string).collect::<Vec<_>>().join(", ");
            format!(
                "\nconstant char {name}_levels[{}] = {{ {entries} }};\nstatic inline float {name}_code(uint code) {{\n    return (float){name}_levels[code];\n}}\n",
                values.len()
            )
        }
        LevelTable::SignedGrid { grid, .. } => {
            let rows = grid
                .chunks(8)
                .map(|row| row.iter().map(|entry| format!("{entry:#010x}u")).collect::<Vec<_>>().join(", "))
                .collect::<Vec<_>>()
                .join(",\n    ");
            format!("\nconstant uint {name}_grid[{}] = {{\n    {rows}\n}};\n", grid.len())
        }
        LevelTable::SignedGrid64 { .. } | LevelTable::DeltaGrid { .. } => String::new(),
    }
}

/// The statements that bind `float raw` (the element's level) for `codes`.
fn raw_level_msl(layout: &BlockLayout, codes: Codes<'_>, name: &str) -> String {
    match codes {
        Codes::Levels(LevelTable::SignedGrid { lanes, sign, unit_shift, .. }) => {
            signed_grid_raw_msl(&layout.element, name, *lanes, sign, *unit_shift)
        }
        Codes::Affine | Codes::Levels(_) => {
            coded_expression(&layout.element, "index", "raw", &format!("{name}_code"))
        }
    }
}

fn signed_grid_raw_msl(coded: &Coded, name: &str, lanes: u8, sign: &Plane, unit_shift: u8) -> String {
    let (mut text, code) = coded_assembly(coded, "index", "raw_code");
    text.push_str(&format!(
        "    uint magnitude = ({name}_grid[{code}] >> (8u * (index % {lanes}u))) & 0xFFu;\n"
    ));
    text.push_str(&plane_expression(sign, "index", "negative"));
    let unit = if unit_shift == 0 {
        String::new()
    } else {
        format!(" * {:e}f", 0.5f32.powi(i32::from(unit_shift)))
    };
    text.push_str(&format!(
        "    float raw = negative != 0u ? -(float)magnitude{unit} : (float)magnitude{unit};\n"
    ));
    text
}

fn element_msl(layout: &BlockLayout, codes: Codes<'_>, name: &str) -> String {
    let mut helpers = match codes {
        Codes::Affine => layout_load_msl(layout.element.code, &format!("{name}_code")),
        Codes::Levels(levels) => level_table_msl(levels, name),
    };
    helpers.push_str(&layout_load_msl(
        layout.scale.layout,
        &format!("{name}_scale"),
    ));
    if let Some(min) = &layout.min {
        helpers.push_str(&layout_load_msl(min.layout, &format!("{name}_min")));
    }
    let mut body = raw_level_msl(layout, codes, name);
    body.push_str(&field_expression(
        &layout.scale,
        "scale",
        &format!("{name}_scale"),
    ));
    let mut scale = "scale".to_string();
    if let Some(sub) = &layout.sub {
        let span = layout.elements / sub.count;
        helpers.push_str(&layout_load_msl(
            sub.scale.code,
            &format!("{name}_sub_scale_code"),
        ));
        body.push_str(&format!("    uint sub_block = index / {span}u;\n"));
        body.push_str(&coded_expression(
            &sub.scale,
            "sub_block",
            "sub_scale_raw",
            &format!("{name}_sub_scale_code"),
        ));
        body.push_str(&format!(
            "    float sub_scale = {} - {}.0f;\n    float effective_scale = scale * sub_scale;\n",
            sub_scale_product(sub.scale_multiplier),
            sub.scale.bias
        ));
        scale = "effective_scale".to_string();
    }
    match codes {
        Codes::Affine => body.push_str(&format!(
            "    float value = (({}.0f * raw) - {}.0f) * {scale};\n",
            layout.multiplier, layout.element.bias
        )),
        Codes::Levels(_) => body.push_str(&format!("    float value = raw * {scale};\n")),
    }
    if let Some(min) = &layout.min {
        body.push_str(&field_expression(min, "minimum", &format!("{name}_min")));
        let mut minimum = "minimum".to_string();
        if let Some(sub_min) = layout.sub.and_then(|sub| sub.min) {
            helpers.push_str(&layout_load_msl(
                sub_min.code,
                &format!("{name}_sub_min_code"),
            ));
            body.push_str(&coded_expression(
                &sub_min,
                "sub_block",
                "sub_min_raw",
                &format!("{name}_sub_min_code"),
            ));
            body.push_str(&format!(
                "    float effective_min = minimum * (sub_min_raw - {}.0f);\n",
                sub_min.bias
            ));
            minimum = "effective_min".to_string();
        }
        body.push_str(&format!(
            "    value = value + ({}.0f * {minimum});\n",
            layout.min_multiplier
        ));
    }
    format!(
        "{helpers}\nstatic inline float {name}(device const uchar *block, uint index) {{\n{body}    return value;\n}}\n"
    )
}

/// Renders `static inline float4 name_run(device const uchar *block, uint index)`: elements
/// `index..index + 4` of one block, with the block scale (and min) decoded once for the run
/// instead of once per element. `index` is a multiple of four, which `block_k_refusal` guarantees
/// stays inside one block. Reuses the `name_*` load helpers [`block_unpack_msl`] renders, and
/// keeps the scalar helper's arithmetic order so the two agree bit for bit.
fn block_run_msl(layout: &BlockLayout, name: &str) -> String {
    let mut body = field_expression(&layout.scale, "scale", &format!("{name}_scale"));
    if let Some(min) = &layout.min {
        body.push_str(&field_expression(min, "minimum", &format!("{name}_min")));
    }
    for lane in 0..4 {
        body.push_str(&run_element_msl(layout, name, lane));
    }
    format!(
        "static inline float4 {name}_run(device const uchar *block, uint index) {{\n{body}    return float4(value0, value1, value2, value3);\n}}\n"
    )
}

fn run_element_msl(layout: &BlockLayout, name: &str, lane: usize) -> String {
    let at = format!("(index + {lane}u)");
    let raw = format!("raw{lane}");
    let mut text = coded_expression(&layout.element, &at, &raw, &format!("{name}_code"));
    let mut scale = "scale".to_string();
    let mut minimum = "minimum".to_string();
    if let Some(sub) = &layout.sub {
        let span = layout.elements / sub.count;
        text.push_str(&format!("    uint sub_block{lane} = {at} / {span}u;\n"));
        text.push_str(&coded_expression(
            &sub.scale,
            &format!("sub_block{lane}"),
            &format!("sub_scale_raw{lane}"),
            &format!("{name}_sub_scale_code"),
        ));
        text.push_str(&format!(
            "    float effective_scale{lane} = scale * (sub_scale_raw{lane} - {}.0f);\n",
            sub.scale.bias
        ));
        scale = format!("effective_scale{lane}");
    }
    text.push_str(&format!(
        "    float value{lane} = (({}.0f * {raw}) - {}.0f) * {scale};\n",
        layout.multiplier, layout.element.bias
    ));
    if layout.min.is_some() {
        if let Some(sub_min) = layout.sub.and_then(|sub| sub.min) {
            text.push_str(&coded_expression(
                &sub_min,
                &format!("sub_block{lane}"),
                &format!("sub_min_raw{lane}"),
                &format!("{name}_sub_min_code"),
            ));
            text.push_str(&format!(
                "    float effective_min{lane} = minimum * (sub_min_raw{lane} - {}.0f);\n",
                sub_min.bias
            ));
            minimum = format!("effective_min{lane}");
        }
        text.push_str(&format!(
            "    value{lane} = value{lane} + ({}.0f * {minimum});\n",
            layout.min_multiplier
        ));
    }
    text
}
