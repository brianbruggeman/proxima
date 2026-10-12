use const_format::formatcp;
use proxima_primitives::float_layout::FloatLayout;

use super::*;

pub(super) const FP6_E3M2_ELEMENTS: usize = Codec::FP6_E3M2_BLOCK_ELEMENTS;
pub(super) const FP6_E3M2_BYTES: usize = Codec::FP6_E3M2_BLOCK_BYTES;
const FIELDS_PER_GROUP: u32 = 4;
const BYTES_PER_GROUP: u32 = FIELDS_PER_GROUP * FloatLayout::E3M2.width() as u32 / 8;
const SCALE_BYTES: usize =
    FP6_E3M2_BYTES - FP6_E3M2_ELEMENTS * FloatLayout::E3M2.width() as usize / 8;
const SCALE_DECODE_MSL: &str = "    ushort scale_bits = (ushort)((uint)block[0] | ((uint)block[1] << 8));\n    float scale = (float)as_type<half>(scale_bits);\n";

pub(super) fn push_scaled_float_decoders(source: &mut String, quantized: &[Option<Codec>]) {
    if quantized.contains(&Some(Codec::Fp6E3M2)) {
        source.push_str(&scaled_e3m2_unpack_msl(FloatLayout::E3M2));
        source.push('\n');
    }
}

/// Renders the e3m2 decoders for `layout`: the sub-block geometry is the codec's
/// (`Codec::Fp6E3M2`, 32 fields of six bits behind an f16 scale), the field decode
/// constants come from `layout` (`FloatLayout::E3M2`, or an override of it).
pub(super) fn scaled_e3m2_unpack_msl(layout: FloatLayout) -> String {
    [
        scaled_e3m2_field_msl(layout),
        scaled_e3m2_group_msl(),
        scaled_e3m2_element_msl(layout),
        scaled_e3m2_super_element_msl(),
        scaled_e3m2_quad_dot_msl(layout),
    ]
    .concat()
}

fn scaled_e3m2_field_msl(layout: FloatLayout) -> String {
    super::layout_render::layout_load_msl(layout, "scaled_e3m2_field")
}

fn scaled_e3m2_group_msl() -> String {
    let word = (0..BYTES_PER_GROUP)
        .map(|byte| format!("((uint)group[{byte}] << {}u)", byte * 8))
        .collect::<Vec<_>>()
        .join(" | ");
    format!(
        "
static inline uint scaled_e3m2_group(device const uchar *group) {{
    return {word};
}}
"
    )
}

fn scaled_e3m2_element_msl(layout: FloatLayout) -> String {
    let width = layout.width();
    let mask = (1u32 << width) - 1;
    format!(
        "
static inline float scaled_e3m2_element(device const uchar *block, uint index) {{
{SCALE_DECODE_MSL}    uint word = scaled_e3m2_group(block + {SCALE_BYTES}u + (index / {FIELDS_PER_GROUP}u) * {BYTES_PER_GROUP}u);
    uint field = (word >> ((index % {FIELDS_PER_GROUP}u) * {width}u)) & {mask:#04x}u;
    return scaled_e3m2_field(field) * scale;
}}
"
    )
}

fn scaled_e3m2_super_element_msl() -> String {
    let (elements, bytes) = (FP6_E3M2_ELEMENTS, FP6_E3M2_BYTES);
    format!(
        "
static inline float scaled_e3m2_super_element(device const uchar *superblock, uint index) {{
    return scaled_e3m2_element(superblock + (index / {elements}u) * {bytes}u, index % {elements}u);
}}
"
    )
}

fn scaled_e3m2_quad_dot_msl(layout: FloatLayout) -> String {
    let width = u32::from(layout.width());
    let mask = (1u32 << width) - 1;
    let groups = FP6_E3M2_ELEMENTS as u32 / FIELDS_PER_GROUP;
    let (elements, bytes) = (FP6_E3M2_ELEMENTS, FP6_E3M2_BYTES);
    let accumulate = (0..FIELDS_PER_GROUP)
        .map(|field| {
            let shift = field * width;
            format!("        result += scaled_e3m2_field((word >> {shift}u) & {mask:#04x}u) * run[{field}];\n")
        })
        .collect::<String>();
    format!(
        "
static inline float scaled_e3m2_quad_dot(device const uchar *superblock, uint slot, thread const float *acts) {{
    device const uchar *block = superblock + (slot / {elements}u) * {bytes}u;
{SCALE_DECODE_MSL}    device const uchar *payload = block + {SCALE_BYTES};
    float result = 0.0f;
    for (uint group = 0u; group < {groups}u; ++group) {{
        uint word = scaled_e3m2_group(payload + group * {BYTES_PER_GROUP}u);
        thread const float *run = acts + group * {FIELDS_PER_GROUP}u;
{accumulate}    }}
    return result * scale;
}}
"
    )
}

const ELEMENT_MASK: u32 = (1u32 << FloatLayout::E3M2.width()) - 1;
const RUN_GROUPS: u32 = DECODE_RUN_ELEMENTS as u32 / FIELDS_PER_GROUP;
const FIELD_WIDTH: u32 = FloatLayout::E3M2.width() as u32;

/// The e3m2 entry of the tiled decode description (`tiled_decode`): the
/// `function(block, slot, out)` contract writing one [`DECODE_RUN_ELEMENTS`] run
/// as `half`. It reads the 3-byte groups through `scaled_e3m2_group` and decodes
/// through `scaled_e3m2_field`, the same definitions the row-blocked quad dot
/// calls (spliced once by [`push_scaled_float_decoders`]); every byte read is a
/// `uchar`, because a 26-byte block puts the payload at +2 and the groups on a
/// 3-byte stride, so no group is ever 4-byte or 2-byte aligned.
pub(super) const SCALED_E3M2_DECODE_MSL: &str = formatcp!(
    "
static inline void scaled_e3m2_decode(device const uchar *block, uint slot, thread half *out) {{
{SCALE_DECODE_MSL}    device const uchar *payload = block + {SCALE_BYTES}u + (slot / {FIELDS_PER_GROUP}u) * {BYTES_PER_GROUP}u;
    for (uint group = 0u; group < {RUN_GROUPS}u; ++group) {{
        uint word = scaled_e3m2_group(payload + group * {BYTES_PER_GROUP}u);
        for (uint field = 0u; field < {FIELDS_PER_GROUP}u; ++field) {{
            out[group * {FIELDS_PER_GROUP}u + field] = (half)(scaled_e3m2_field((word >> (field * {FIELD_WIDTH}u)) & {ELEMENT_MASK:#x}u) * scale);
        }}
    }}
}}
"
);
