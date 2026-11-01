use alloc::format;
use alloc::string::String;

use proxima_primitives::block_layout::{BlockLayout, Coded, Field};
use proxima_primitives::float_layout::{FloatLayout, SignEncoding};

use super::radix_dot::radix_block_dot;
use crate::cuda_layout::layout_load_cuda;

/// Elements one int8 dot step covers: eight `dp4a` words.
pub(super) const DOT_ELEMENTS: usize = 32;

/// The int8 lowering of one [`BlockLayout`]: `<name>_i8x32(block, chunk, words)` writes the 32
/// codes of chunk `chunk` as eight int8x4 words holding `multiplier * code - bias`, and
/// `<name>_scale(block)` loads the block scale, so `scale * dot(words, x)` is the block's
/// contribution. Derived from the layout's fields alone; no format is named.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct BlockDot {
    pub name: String,
    pub helpers: String,
    pub elements: usize,
    pub block_bytes: usize,
}

/// `None` when the layout is neither a radix-packed code nor a single-plane linear code whose
/// 32-element chunks are contiguous bit runs of 1, 2, 4 or 8 bits, or its affine value does not
/// fit int8.
pub(super) fn block_dot_i8(layout: &BlockLayout) -> Option<BlockDot> {
    if layout.element.packed_radix > 1 {
        return radix_block_dot(layout);
    }
    let element = &layout.element;
    let bits = usize::from(element.low.bits);
    let elements = usize::from(layout.elements);
    let linear = layout.sub.is_none()
        && layout.min.is_none()
        && element.high.is_none()
        && element.packed_radix == 0
        && elements.is_multiple_of(DOT_ELEMENTS)
        && matches!(bits, 1 | 2 | 4 | 8);
    if !linear || !values_fit_int8(element, layout.multiplier, bits) {
        return None;
    }
    let chunk_base_bits = contiguous_chunk_base(element, elements, bits)?;
    let name = layout_name(layout);
    let mut helpers = scale_helpers(layout, &name)?;
    helpers.push_str(&words_helper(&name, layout, bits, chunk_base_bits / 8)?);
    Some(BlockDot {
        name,
        helpers,
        elements,
        block_bytes: usize::from(layout.block_bytes),
    })
}

/// `<name>_scale(block)`: the block scale, read through the scale field's layout.
pub(super) fn scale_helpers(layout: &BlockLayout, name: &str) -> Option<String> {
    let scale_read = scale_read(&layout.scale)?;
    let mut helpers = layout_load_cuda(layout.scale.layout, &format!("{name}_scale_code"));
    helpers.push_str(&format!(
        "__device__ __forceinline__ float {name}_scale(const unsigned char *block) {{\n    return {name}_scale_code({scale_read});\n}}\n"
    ));
    Some(helpers)
}

pub(super) fn layout_name(layout: &BlockLayout) -> String {
    let element = &layout.element;
    let scale = &layout.scale;
    format!(
        "blockdot_e{}_b{}_c{}{}r{}_at{}_m{}_z{}_s{}x{}at{}",
        layout.elements,
        layout.block_bytes,
        element.low.bits,
        sign_token(element.code),
        element.packed_radix,
        element.low.base_bits,
        layout.multiplier,
        element.bias,
        scale.layout.exponent_bits,
        scale.layout.mantissa_bits,
        scale.byte_offset
    )
}

fn sign_token(code: FloatLayout) -> &'static str {
    match code.sign {
        SignEncoding::TwosComplement => "s",
        SignEncoding::Magnitude => "m",
        SignEncoding::Absent => "u",
    }
}

fn signed_code(code: FloatLayout, raw: i32, bits: usize) -> Option<i32> {
    match code.sign {
        SignEncoding::Absent => Some(raw),
        SignEncoding::TwosComplement => Some(if raw >= 1 << (bits - 1) { raw - (1 << bits) } else { raw }),
        SignEncoding::Magnitude => None,
    }
}

fn values_fit_int8(element: &Coded, multiplier: i16, bits: usize) -> bool {
    (0..1i32 << bits).all(|raw| {
        signed_code(element.code, raw, bits)
            .map(|code| i32::from(multiplier) * code - i32::from(element.bias))
            .is_some_and(|value| (-128..=127).contains(&value))
    })
}

/// The bit address of chunk 0, when every chunk is a contiguous run starting
/// `DOT_ELEMENTS * bits` after the previous one on a 16-bit boundary.
fn contiguous_chunk_base(element: &Coded, elements: usize, bits: usize) -> Option<usize> {
    let base = element.low.address(0);
    let contiguous = (0..elements).all(|index| element.low.address(index) == base + index * bits);
    (contiguous && base.is_multiple_of(16)).then_some(base)
}

fn scale_read(field: &Field) -> Option<String> {
    let offset = field.byte_offset;
    match field.layout.width() {
        16 if offset.is_multiple_of(2) => Some(format!(
            "(unsigned int)*(const unsigned short *)(block + {offset}u)"
        )),
        32 if offset.is_multiple_of(4) => Some(format!("*(const unsigned int *)(block + {offset}u)")),
        _ => None,
    }
}

/// `x * multiplier - bias` on each byte of `x`, exact because every result fits int8 and no
/// byte product carries into its neighbour.
pub(super) fn affine(expression: &str, multiplier: i16, bias: i16, bits: usize) -> Option<String> {
    if multiplier == 1 && bias == 0 {
        return Some(expression.into());
    }
    let largest = ((1u32 << bits) - 1) * u32::try_from(multiplier).ok()?;
    if largest > 255 {
        return None;
    }
    let bias_bytes = u32::from(bias.to_le_bytes()[0]) * 0x0101_0101;
    Some(format!(
        "(int)__vsub4((unsigned int)({expression}) * {multiplier}u, {bias_bytes:#010x}u)"
    ))
}

fn table_byte(layout: &BlockLayout, code: i32) -> u32 {
    let value = i32::from(layout.multiplier) * code - i32::from(layout.element.bias);
    u32::from(value.to_le_bytes()[0])
}

fn words_helper(name: &str, layout: &BlockLayout, bits: usize, base_bytes: usize) -> Option<String> {
    let multiplier = layout.multiplier;
    let bias = layout.element.bias;
    let chunk_bytes = DOT_ELEMENTS * bits / 8;
    let body = match bits {
        8 => format!(
            "    #pragma unroll\n    for (int word = 0; word < 8; word++) {{\n        unsigned int packed = (unsigned int)halves[2 * word] | ((unsigned int)halves[2 * word + 1] << 16);\n        words[word] = {};\n    }}\n",
            affine("packed", multiplier, bias, bits)?
        ),
        4 => format!(
            "    #pragma unroll\n    for (int pair = 0; pair < 4; pair++) {{\n        unsigned int packed = (unsigned int)halves[2 * pair] | ((unsigned int)halves[2 * pair + 1] << 16);\n        int low = (int)(packed & 0x0F0F0F0Fu);\n        int high = (int)((packed >> 4u) & 0x0F0F0F0Fu);\n        words[2 * pair] = {};\n        words[2 * pair + 1] = {};\n    }}\n",
            affine("__byte_perm(low, high, 0x5140)", multiplier, bias, bits)?,
            affine("__byte_perm(low, high, 0x7362)", multiplier, bias, bits)?
        ),
        2 => {
            let table = (0..4).fold(0u32, |table, code| table | (table_byte(layout, code) << (8 * code)));
            format!(
                "    #pragma unroll\n    for (int quad = 0; quad < 4; quad++) {{\n        int packed = (int)(short)halves[quad];\n        int even = __byte_perm({table:#010x}, {table:#010x}, packed);\n        int odd = __byte_perm({table:#010x}, {table:#010x}, packed >> 2);\n        words[2 * quad] = __byte_perm(even, odd, 0x5140);\n        words[2 * quad + 1] = __byte_perm(even, odd, 0x7362);\n    }}\n"
            )
        }
        1 => format!(
            "    unsigned int packed = (unsigned int)halves[0] | ((unsigned int)halves[1] << 16);\n    #pragma unroll\n    for (int word = 0; word < 8; word++) {{\n        unsigned int nibble = (packed >> (4 * word)) & 0xFu;\n        unsigned int spread = (nibble & 1u) | ((nibble & 2u) << 7) | ((nibble & 4u) << 14) | ((nibble & 8u) << 21);\n        words[word] = {};\n    }}\n",
            affine("spread", multiplier, bias, bits)?
        ),
        _ => return None,
    };
    Some(format!(
        "__device__ __forceinline__ void {name}_i8x32(const unsigned char *block, unsigned int chunk, int *words) {{\n    const unsigned short *halves = (const unsigned short *)(block + {base_bytes}u + chunk * {chunk_bytes}u);\n{body}}}\n"
    ))
}

#[cfg(test)]
mod tests {
    use proxima_primitives::block_layout::BlockLayout;

    use super::block_dot_i8;

    #[test]
    fn single_plane_runs_lower_and_split_or_sub_block_layouts_do_not() {
        for layout in [BlockLayout::Q8_0, BlockLayout::Q1_0, BlockLayout::Q2_0, BlockLayout::PQ2_0] {
            assert!(block_dot_i8(&layout).is_some(), "{layout:?} is a contiguous linear run");
        }
        for layout in [BlockLayout::TQ1_0, BlockLayout::PTQ1_0] {
            assert!(block_dot_i8(&layout).is_some(), "{layout:?} is a radix-packed byte code");
        }
        for layout in [BlockLayout::Q4_0, BlockLayout::Q5_0, BlockLayout::Q4_1, BlockLayout::Q4_K] {
            assert!(block_dot_i8(&layout).is_none(), "{layout:?} is not a single contiguous run");
        }
    }

    #[test]
    fn distinct_layouts_get_distinct_helper_names() {
        let two_bit_64 = block_dot_i8(&BlockLayout::Q2_0).expect("q2_0 lowers");
        let two_bit_128 = block_dot_i8(&BlockLayout::PQ2_0).expect("pq2_0 lowers");
        assert_ne!(two_bit_64.name, two_bit_128.name);
        assert_eq!(two_bit_128.elements, 128);
        assert_eq!(two_bit_128.block_bytes, 34);
    }
}
