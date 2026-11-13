use alloc::format;
use alloc::string::String;

use proxima_numeric::block_layout::BlockLayout;

use super::block_dot::{BlockDot, affine, layout_name, rows_per_warp, scale_helpers};

const NIBBLE_BITS: usize = 4;

/// The int8 lowering of a linear code whose low 4-bit plane is nibble-split (the first half of a
/// 32-element chunk in the low nibbles of 16 bytes, the second half in the high nibbles) and whose
/// optional high plane is one bit per element in a contiguous run. The assembled code is
/// `low | high << 4`. The plane addresses come from `Int8DotShape::NibbleSplit`; no format is named.
pub(super) fn split_block_dot(layout: &BlockLayout, low_base_byte: usize, high_base_byte: Option<usize>) -> Option<BlockDot> {
    let elements = usize::from(layout.elements);
    let name = format!("{}_ns{}", layout_name(layout), high_base_byte.map_or_else(String::new, |byte| format!("h{}", byte * 8)));
    let mut helpers = scale_helpers(layout, &name)?;
    helpers.push_str(&words_helper(&name, layout, low_base_byte, high_base_byte)?);
    Some(BlockDot {
        name,
        helpers,
        elements,
        block_bytes: usize::from(layout.block_bytes),
        rows_per_warp: rows_per_warp(false),
        centering: None,
        chunk_terms: None,
    })
}

const HIGH_LOAD: &str = "    const unsigned int high_bits = (unsigned int)high_halves[0] | ((unsigned int)high_halves[1] << 16);\n";
const HIGH_MERGE: &str = "        code_low |= ((((high_bits >> (4 * quad)) & 0xFu) * 0x00204081u) & 0x01010101u) << 4u;\n        code_high |= ((((high_bits >> (16 + 4 * quad)) & 0xFu) * 0x00204081u) & 0x01010101u) << 4u;\n";

fn words_helper(name: &str, layout: &BlockLayout, low_bytes: usize, high_bytes: Option<usize>) -> Option<String> {
    let code_bits = NIBBLE_BITS + usize::from(high_bytes.is_some());
    let (multiplier, bias) = (layout.multiplier, layout.element.bias);
    let low_word = affine("code_low", multiplier, bias, code_bits)?;
    let high_word = affine("code_high", multiplier, bias, code_bits)?;
    let high_pointer = high_bytes.map_or_else(String::new, |bytes| {
        format!("    const unsigned short *high_halves = (const unsigned short *)(block + {bytes}u + chunk * 4u);\n{HIGH_LOAD}")
    });
    let merge = if high_bytes.is_some() { HIGH_MERGE } else { "" };
    Some(format!(
        "__device__ __forceinline__ void {name}_i8x32(const unsigned char *block, unsigned int chunk, int *words) {{\n    const unsigned short *low_halves = (const unsigned short *)(block + {low_bytes}u + chunk * 16u);\n{high_pointer}    #pragma unroll\n    for (int quad = 0; quad < 4; quad++) {{\n        unsigned int packed = (unsigned int)low_halves[2 * quad] | ((unsigned int)low_halves[2 * quad + 1] << 16);\n        unsigned int code_low = packed & 0x0F0F0F0Fu;\n        unsigned int code_high = (packed >> 4u) & 0x0F0F0F0Fu;\n{merge}        words[quad] = {low_word};\n        words[4 + quad] = {high_word};\n    }}\n}}\n"
    ))
}

#[cfg(test)]
mod tests {
    use proxima_numeric::block_layout::BlockLayout;
    use proxima_numeric::int8_dot::{Int8DotShape, int8_dot_shape};

    use crate::cuda::block_dot::block_dot_i8;

    #[test]
    fn nibble_split_planes_lower_with_and_without_a_high_bit_plane() {
        let q5 = block_dot_i8(&BlockLayout::Q5_0).expect("q5_0 is a nibble-split plane plus a bit run");
        let q4 = block_dot_i8(&BlockLayout::Q4_0).expect("q4_0 is a nibble-split plane");
        assert_eq!((q5.elements, q5.block_bytes), (32, 22));
        assert_eq!((q4.elements, q4.block_bytes), (32, 18));
        assert_ne!(q5.name, q4.name);
        assert!(q5.helpers.contains("high_bits") && !q4.helpers.contains("high_bits"));
    }

    #[test]
    fn min_bearing_and_sub_block_layouts_are_not_nibble_split_codes() {
        for layout in [BlockLayout::Q4_1, BlockLayout::Q5_1, BlockLayout::Q4_K, BlockLayout::Q8_0] {
            let shape = int8_dot_shape(&layout);
            assert!(!matches!(shape, Some(Int8DotShape::NibbleSplit { .. })), "{layout:?} is not a min-free nibble-split code: {shape:?}");
        }
    }
}
