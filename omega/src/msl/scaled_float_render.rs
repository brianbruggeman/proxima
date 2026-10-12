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
        scaled_block_group_msl(&E3M2_BLOCK),
        scaled_block_element_msl(&E3M2_BLOCK, layout),
        scaled_block_super_element_msl(&E3M2_BLOCK),
        scaled_e3m2_quad_dot_msl(layout),
    ]
    .concat()
}

fn scaled_e3m2_field_msl(layout: FloatLayout) -> String {
    super::layout_render::layout_load_msl(layout, "scaled_e3m2_field")
}

/// The geometry of one block of fields under an f16 scale at its start: how many elements, how
/// many bytes, and how fields pack into groups of bytes. The block emitters below render the
/// element accessors of any such block from a layout; fp6 e3m2 (four six-bit fields in three
/// bytes) and Q8_0 (one eight-bit integer in one byte) are two instances of it.
#[derive(Clone, Copy)]
pub(super) struct ScaledBlock {
    pub prefix: &'static str,
    pub elements: usize,
    pub block_bytes: usize,
    pub fields_per_group: u32,
    pub bytes_per_group: u32,
}

impl ScaledBlock {
    const fn scale_bytes(&self, layout: &FloatLayout) -> usize {
        self.block_bytes - self.elements * layout.width() as usize / 8
    }
}

pub(super) const E3M2_BLOCK: ScaledBlock = ScaledBlock {
    prefix: "scaled_e3m2",
    elements: FP6_E3M2_ELEMENTS,
    block_bytes: FP6_E3M2_BYTES,
    fields_per_group: FIELDS_PER_GROUP,
    bytes_per_group: BYTES_PER_GROUP,
};

pub(super) const Q8_0_BLOCK: ScaledBlock = ScaledBlock {
    prefix: "q8_0",
    elements: Q8_0_BLOCK_ELEMENTS,
    block_bytes: Q8_0_BLOCK_BYTES,
    fields_per_group: 1,
    bytes_per_group: 1,
};

pub(super) fn scaled_block_group_msl(block: &ScaledBlock) -> String {
    let prefix = block.prefix;
    let word = (0..block.bytes_per_group)
        .map(|byte| format!("((uint)group[{byte}] << {}u)", byte * 8))
        .collect::<Vec<_>>()
        .join(" | ");
    format!(
        "
static inline uint {prefix}_group(device const uchar *group) {{
    return {word};
}}
"
    )
}

pub(super) fn scaled_block_element_msl(block: &ScaledBlock, layout: FloatLayout) -> String {
    let prefix = block.prefix;
    let width = layout.width();
    let mask = (1u64 << width) - 1;
    let scale_bytes = block.scale_bytes(&layout);
    let (fields, group_bytes) = (block.fields_per_group, block.bytes_per_group);
    format!(
        "
static inline float {prefix}_element(device const uchar *block, uint index) {{
{SCALE_DECODE_MSL}    uint word = {prefix}_group(block + {scale_bytes}u + (index / {fields}u) * {group_bytes}u);
    uint field = (word >> ((index % {fields}u) * {width}u)) & {mask:#04x}u;
    return {prefix}_field(field) * scale;
}}
"
    )
}

pub(super) fn scaled_block_super_element_msl(block: &ScaledBlock) -> String {
    let prefix = block.prefix;
    let (elements, bytes) = (block.elements, block.block_bytes);
    format!(
        "
static inline float {prefix}_super_element(device const uchar *superblock, uint index) {{
    return {prefix}_element(superblock + (index / {elements}u) * {bytes}u, index % {elements}u);
}}
"
    )
}

/// The `q8_0_field`, `q8_0_group`, `q8_0_element` and `q8_0_super_element` accessors of a Q8_0
/// block, rendered by the same block emitters as the fp6 e3m2 ones.
pub fn q8_0_unpack_msl() -> String {
    [
        super::layout_render::layout_load_msl(FloatLayout::INT8, "q8_0_field"),
        scaled_block_group_msl(&Q8_0_BLOCK),
        scaled_block_element_msl(&Q8_0_BLOCK, FloatLayout::INT8),
        scaled_block_super_element_msl(&Q8_0_BLOCK),
    ]
    .concat()
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
