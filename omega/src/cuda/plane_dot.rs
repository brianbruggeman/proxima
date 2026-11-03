use alloc::format;
use alloc::string::String;

use proxima_primitives::block_layout::{BlockLayout, Plane};

use super::block_dot::{BlockDot, DOT_ELEMENTS, affine, layout_name, rows_per_warp, scale_helpers, values_fit_int8};

const NIBBLE_BITS: usize = 4;
const CHUNK_NIBBLE_BITS: usize = DOT_ELEMENTS * NIBBLE_BITS;
const HALF_CHUNK: usize = DOT_ELEMENTS / 2;
const ADDRESS_ALIGNMENT_BITS: usize = 16;

/// The int8 lowering of a linear code whose low 4-bit plane is nibble-split (the first half of a
/// 32-element chunk in the low nibbles of 16 bytes, the second half in the high nibbles) and whose
/// optional high plane is one bit per element in a contiguous run. The assembled code is
/// `low | high << 4`; `multiplier * code - bias` must fit int8. Derived from the planes' address
/// maps alone; no format is named.
pub(super) fn split_block_dot(layout: &BlockLayout) -> Option<BlockDot> {
    let element = &layout.element;
    let elements = usize::from(layout.elements);
    let linear = layout.sub.is_none()
        && layout.min.is_none()
        && layout.scale_runs.is_none()
        && element.packed_radix == 0
        && elements.is_multiple_of(DOT_ELEMENTS);
    let low_base = nibble_split_base(&element.low, elements).filter(|_| linear)?;
    let high_base = match &element.high {
        Some(plane) => Some(bit_run_base(plane, elements)?),
        None => None,
    };
    let code_bits = NIBBLE_BITS + usize::from(high_base.is_some());
    if !values_fit_int8(element, layout.multiplier, code_bits) {
        return None;
    }
    let name = format!("{}_ns{}", layout_name(layout), high_base.map_or_else(String::new, |base| format!("h{base}")));
    let mut helpers = scale_helpers(layout, &name)?;
    helpers.push_str(&words_helper(&name, layout, low_base / 8, high_base.map(|base| base / 8))?);
    Some(BlockDot {
        name,
        helpers,
        elements,
        block_bytes: usize::from(layout.block_bytes),
        rows_per_warp: rows_per_warp(false),
        centering: None,
    })
}

fn nibble_split_offset(position: usize) -> usize {
    (position % HALF_CHUNK) * 8 + (position / HALF_CHUNK) * NIBBLE_BITS
}

fn nibble_split_base(plane: &Plane, elements: usize) -> Option<usize> {
    let base = plane.address(0);
    let split = usize::from(plane.bits) == NIBBLE_BITS
        && (0..elements).all(|index| {
            plane.address(index) == base + index / DOT_ELEMENTS * CHUNK_NIBBLE_BITS + nibble_split_offset(index % DOT_ELEMENTS)
        });
    (split && base.is_multiple_of(ADDRESS_ALIGNMENT_BITS)).then_some(base)
}

fn bit_run_base(plane: &Plane, elements: usize) -> Option<usize> {
    let base = plane.address(0);
    let run = plane.bits == 1 && (0..elements).all(|index| plane.address(index) == base + index);
    (run && base.is_multiple_of(ADDRESS_ALIGNMENT_BITS)).then_some(base)
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
    use proxima_primitives::block_layout::BlockLayout;

    use super::split_block_dot;

    #[test]
    fn nibble_split_planes_lower_with_and_without_a_high_bit_plane() {
        let q5 = split_block_dot(&BlockLayout::Q5_0).expect("q5_0 is a nibble-split plane plus a bit run");
        let q4 = split_block_dot(&BlockLayout::Q4_0).expect("q4_0 is a nibble-split plane");
        assert_eq!((q5.elements, q5.block_bytes), (32, 22));
        assert_eq!((q4.elements, q4.block_bytes), (32, 18));
        assert_ne!(q5.name, q4.name);
        assert!(q5.helpers.contains("high_bits") && !q4.helpers.contains("high_bits"));
    }

    #[test]
    fn min_bearing_and_sub_block_layouts_do_not_lower() {
        for layout in [BlockLayout::Q4_1, BlockLayout::Q5_1, BlockLayout::Q4_K, BlockLayout::Q8_0] {
            assert!(split_block_dot(&layout).is_none(), "{layout:?} is not a min-free nibble-split code");
        }
    }
}
