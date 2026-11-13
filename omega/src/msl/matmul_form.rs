//! The decode matvec for a block layout whose codes are whole bytes under one scale, rendered from
//! the [`BlockLayout`] and a launch geometry and from nothing else.
//!
//! Teaching pointer: `sub_block_matvec_msl` renders the layouts whose sub-blocks carry their own
//! scale codes; this is the same walk for a layout with no sub-blocks, which is one sub-block of
//! unit scale. Its facts come from `BlockLayout::contiguous_byte_codes` (proxima-numeric): the
//! address of code `j` is `base_byte + j * code_bytes`, so the kernel carries constants and no
//! per-format table. Selection is by those layout properties; a layout this does not cover is
//! refused with the property named and stays on the caller's other bodies.
//!
//! Geometry: one lane owns one whole block, lane `l` takes real blocks `l + lanes * iteration`.
//! Fold: the scale multiplies every term (`d * code * act`) and a block's terms are summed in
//! element order before joining the row total, the order the policy `bit_exact` admits.

use alloc::format;
use alloc::string::String;

use proxima_numeric::block_layout::{BlockLayout, ByteCodes};
use proxima_numeric::float_layout::SignEncoding;

use super::sub_block_matvec::{SubBlockMatvecRefusal, require, typed_field_read};

/// The local this body names its code bytes by, unique to the flat-block matvec's emitted source;
/// `PACKED_ROW_BODY_MARKERS` holds it so the profiler classifies this body as packed-row-blocked.
pub(super) const FLAT_BLOCK_LEVELS_MARKER: &str = "flat_block_levels";

/// Where the scale enters the sum, and what the per-row statement is when it is not the plain
/// product.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Fold {
    /// `d * code * act` per term, the block's terms summed in element order.
    ScaleEachTerm,
}

/// The launch the matvec is rendered for: rows per simdgroup, lanes per simdgroup, and which
/// operands are the weight and the activations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct FlatLaunch {
    pub(super) rows: usize,
    pub(super) lanes: usize,
    pub(super) weight: usize,
    pub(super) other: usize,
    pub(super) other_stride_is_one: bool,
}

/// A layout the flat-block matvec covers, with every fact the renderer reads.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct FlatBlockForm {
    elements: usize,
    block_bytes: usize,
    codes: ByteCodes,
    signed: bool,
    scale_read: String,
}

impl FlatBlockForm {
    /// # Errors
    /// The property of `layout` the flat-block matvec does not cover.
    pub(super) fn plan(layout: &BlockLayout) -> Result<Self, SubBlockMatvecRefusal> {
        require(layout.sub.is_none(), "no sub-blocks")?;
        require(layout.scale_runs.is_none(), "one scale for the block")?;
        require(layout.min.is_none(), "no block min")?;
        require(layout.multiplier == 1 && layout.element.bias == 0, "code is the value before the scale")?;
        let codes = layout.contiguous_byte_codes().ok_or(SubBlockMatvecRefusal("whole-byte codes laid end to end"))?;
        require(codes.code_bytes == 1, "one byte per code")?;
        Ok(Self {
            elements: usize::from(layout.elements),
            block_bytes: usize::from(layout.block_bytes),
            codes,
            signed: layout.element.code.sign == SignEncoding::TwosComplement,
            scale_read: typed_field_read(&layout.scale)?,
        })
    }

    /// Renders the body between the preamble that declares `lane`, `weight_base`, `other_base`,
    /// `other_stride` and `sumf` and the shared combine and write tail.
    pub(super) fn matvec_msl(&self, launch: &FlatLaunch, reduction_bound: &str, element_type: &str, fold: Fold) -> String {
        let FlatLaunch { rows, lanes, weight, other, other_stride_is_one } = *launch;
        let (elements, block_bytes) = (self.elements, self.block_bytes);
        let mut source = String::new();
        source.push_str(&format!("    int blocks = (int){reduction_bound} / {elements};\n"));
        source.push_str(&format!("    int ib_first = (int)lane;\n    int ib_step = {lanes};\n"));
        source.push_str(&format!("    long blk_step = (long)ib_step * {block_bytes};\n"));
        source.push_str(&format!("    device const uchar *blk_ptr[{rows}];\n"));
        source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
        source.push_str(&format!(
            "        blk_ptr[q] = in{weight} + ((long)((int)weight_base[q] / {elements}) + (long)ib_first) * {block_bytes};\n"
        ));
        source.push_str("    }\n");
        source.push_str(&format!("    {element_type} acts[{elements}];\n"));
        if other_stride_is_one {
            source.push_str(&format!("    long acts_step = (long)ib_step * {elements};\n"));
            source.push_str(&format!(
                "    device const {element_type} *acts_row = in{other} + other_base[0] + (long)ib_first * {elements};\n"
            ));
        } else {
            source.push_str(&format!("    long acts_step = (long)ib_step * {elements} * other_stride;\n"));
            source.push_str(&format!(
                "    device const {element_type} *acts_row = in{other} + other_base[0] + (long)ib_first * {elements} * other_stride;\n"
            ));
        }
        source.push_str("    for (int ib = ib_first; ib < blocks; ib += ib_step) {\n");
        source.push_str(&format!("        for (int j = 0; j < {elements}; ++j) {{\n"));
        if other_stride_is_one {
            source.push_str("            acts[j] = acts_row[j];\n");
        } else {
            source.push_str("            acts[j] = acts_row[(long)j * other_stride];\n");
        }
        source.push_str("        }\n");
        source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{\n"));
        source.push_str("            device const uchar *blk = blk_ptr[q];\n");
        match fold {
            Fold::ScaleEachTerm => source.push_str(&self.scale_each_term()),
        }
        source.push_str("        }\n");
        source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{ blk_ptr[q] += blk_step; }}\n"));
        source.push_str("        acts_row += acts_step;\n    }\n");
        source
    }

    fn scale_each_term(&self) -> String {
        let cast = if self.signed { "char" } else { "uchar" };
        let (elements, base) = (self.elements, self.codes.base_byte);
        format!(
            "            float d = {};\n\
             \x20           device const uchar *{levels} = blk + {base};\n\
             \x20           float result = 0.0f;\n\
             \x20           for (uint j = 0u; j < {elements}u; ++j) {{\n\
             \x20               result += d * (float)(({cast}){levels}[j]) * acts[j];\n\
             \x20           }}\n\
             \x20           sumf[q] = sumf[q] + result;\n",
            self.scale_read,
            levels = FLAT_BLOCK_LEVELS_MARKER,
        )
    }
}

#[cfg(test)]
mod tests {
    use proxima_numeric::block_layout::BlockLayout;

    use super::{FlatBlockForm, FlatLaunch, Fold};

    fn render(layout: &BlockLayout) -> String {
        FlatBlockForm::plan(layout)
            .expect("the layout has a flat-block form")
            .matvec_msl(
                &FlatLaunch { rows: 4, lanes: 32, weight: 0, other: 1, other_stride_is_one: true },
                "u.reduction_total",
                "float",
                Fold::ScaleEachTerm,
            )
    }

    #[test]
    fn q8_0_renders_from_its_layout_one_block_per_lane() {
        let source = render(&BlockLayout::Q8_0);
        assert!(source.contains("(long)ib_first) * 34;"), "{source}");
        assert!(source.contains("blk + 2;") && source.contains("(char)flat_block_levels[j]"), "{source}");
        assert!(source.contains("int ib_step = 32;"), "{source}");
    }

    #[test]
    fn layouts_with_sub_blocks_or_narrow_codes_are_refused_by_property() {
        for layout in [BlockLayout::Q4_0, BlockLayout::Q5_0, BlockLayout::Q4_1, BlockLayout::Q4_K, BlockLayout::Q6_K] {
            assert!(FlatBlockForm::plan(&layout).is_err(), "{layout:?} has no flat-block form");
        }
    }

    #[test]
    fn moving_the_first_code_one_byte_changes_the_rendered_kernel() {
        let mut moved = BlockLayout::Q8_0;
        moved.element.low.base_bits += 8;
        assert_ne!(render(&BlockLayout::Q8_0), render(&moved));
    }

    #[test]
    fn moving_the_scale_field_changes_the_rendered_kernel() {
        let mut moved = BlockLayout::Q8_0;
        moved.scale.byte_offset = 2;
        assert_ne!(render(&BlockLayout::Q8_0), render(&moved));
    }
}
