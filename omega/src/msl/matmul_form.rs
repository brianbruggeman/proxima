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

use proxima_numeric::block_layout::{BlockLayout, ByteCodes, Coded};
use proxima_numeric::float_layout::{FloatLayout, SignEncoding};

use super::layout_render::{coded_assembly, hardware_type};
use super::sub_block_matvec::{SubBlockMatvecRefusal, require, typed_field_read};

/// The local this body names its code bytes by, unique to the flat-block matvec's emitted source;
/// `PACKED_ROW_BODY_MARKERS` holds it so the profiler classifies this body as packed-row-blocked.
pub(super) const FLAT_BLOCK_LEVELS_MARKER: &str = "flat_block_levels";

/// The block-count line every flat-block matvec opens with, whichever `CodeRead` it renders; the
/// plane read names no levels local, so this is its profiler marker.
pub(super) const FLAT_BLOCK_COUNT_MARKER: &str = "int blocks = (int)";

/// Where the scale enters the sum, and what the per-row statement is when it is not the plain
/// product.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Fold {
    /// `d * code * act` per term, the block's terms summed in element order.
    ScaleEachTerm,
    /// `value * act` per term, a run's terms summed in element order: a scalar float has no scale
    /// to hoist.
    Plain,
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

/// How a block's codes are read: the whole-byte run Q8_0 stores, or the bit planes any other
/// unit-scale layout stores them in (Q4_1, Q5_0, Q5_1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum CodeRead {
    /// Codes are `code_bytes == 1` bytes laid end to end; the value is `d * code`.
    ByteRun { codes: ByteCodes, signed: bool },
    /// Codes are assembled per element by `layout_render::coded_assembly` from the layout's planes;
    /// the value is `((multiplier * code) - bias) * d`, plus `min_multiplier * min` when the layout
    /// has a min, the arithmetic `element_msl` renders with `d` and `min` read once per block.
    Planes { element: Coded, multiplier: i16, min_multiplier: i8 },
    /// A scalar float weight read in lane-owned runs as the Metal hardware type of its layout
    /// (`layout_render::hardware_type`: `half`, `bfloat`). The run pointer is cast from the bound
    /// buffer, so a `uchar` payload (`BFloat16`) and a native `half` one (`Float16`) read alike.
    Scalar { element: &'static str },
}

/// A layout the flat-block matvec covers, with every fact the renderer reads.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct FlatBlockForm {
    elements: usize,
    block_bytes: usize,
    read: CodeRead,
    scale_read: String,
    min_read: Option<String>,
}

impl FlatBlockForm {
    /// The `ByteRun` form: whole-byte codes that are the value before the scale, no min.
    ///
    /// # Errors
    /// The property of `layout` the byte-run form does not cover.
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
            read: CodeRead::ByteRun { codes, signed: layout.element.code.sign == SignEncoding::TwosComplement },
            scale_read: typed_field_read(&layout.scale)?,
            min_read: None,
        })
    }

    /// The `Planes` form: one unit-scale block whose codes are read through the layout's planes.
    ///
    /// # Errors
    /// The property of `layout` the plane read does not cover.
    pub(super) fn plan_planes(layout: &BlockLayout) -> Result<Self, SubBlockMatvecRefusal> {
        require(layout.sub.is_none(), "no sub-blocks")?;
        require(layout.scale_runs.is_none(), "one scale for the block")?;
        require(layout.elements > 0, "a block has elements")?;
        require(layout.element.code.sign != SignEncoding::TwosComplement, "unsigned assembled code")?;
        Ok(Self {
            elements: usize::from(layout.elements),
            block_bytes: usize::from(layout.block_bytes),
            read: CodeRead::Planes { element: layout.element, multiplier: layout.multiplier, min_multiplier: layout.min_multiplier },
            scale_read: typed_field_read(&layout.scale)?,
            min_read: layout.min.as_ref().map(typed_field_read).transpose()?,
        })
    }

    /// The `Scalar` form: one lane owns `run_elements` consecutive weights of one float format, no
    /// scale.
    ///
    /// # Errors
    /// A run that is not whole bytes, or a layout with no Metal hardware type to load it as.
    pub(super) fn plan_scalar(layout: FloatLayout, run_elements: usize) -> Result<Self, SubBlockMatvecRefusal> {
        let width = usize::from(layout.width());
        require(run_elements > 0 && (run_elements * width).is_multiple_of(8), "a scalar run is whole bytes")?;
        let element = hardware_type(&layout).ok_or(SubBlockMatvecRefusal("a scalar float with a hardware type"))?;
        Ok(Self {
            elements: run_elements,
            block_bytes: run_elements * width / 8,
            read: CodeRead::Scalar { element },
            scale_read: String::new(),
            min_read: None,
        })
    }

    /// Renders the body between the preamble that declares `lane`, `weight_base`, `other_base`,
    /// `other_stride` and `sumf` and the shared combine and write tail.
    ///
    /// # Errors
    /// A fold the form's read has no meaning under: a scalar read has no scale to apply, and a
    /// scaled read has no plain product.
    pub(super) fn matvec_msl(&self, launch: &FlatLaunch, reduction_bound: &str, element_type: &str, fold: Fold) -> Result<String, SubBlockMatvecRefusal> {
        match (fold, self.read) {
            (Fold::Plain, CodeRead::Scalar { element }) => Ok(self.scalar_run_msl(launch, reduction_bound, element_type, element)),
            (Fold::Plain, _) => Err(SubBlockMatvecRefusal("a plain fold needs an unscaled scalar read")),
            (Fold::ScaleEachTerm, _) => self.block_msl(launch, reduction_bound, element_type, fold),
        }
    }

    fn scalar_run_msl(&self, launch: &FlatLaunch, reduction_bound: &str, element_type: &str, pointer_type: &str) -> String {
        let FlatLaunch { rows, lanes, weight, other, other_stride_is_one } = *launch;
        let run = self.elements;
        let mut source = String::new();
        source.push_str(&format!("    int blocks = (int){reduction_bound} / {run};\n"));
        source.push_str(&format!("    int ib_first = (int)lane;\n    int ib_step = {lanes};\n"));
        source.push_str(&format!("    long blk_step = (long)ib_step * {run};\n"));
        source.push_str(&format!("    device const {pointer_type} *blk_ptr[{rows}];\n"));
        source.push_str(&format!(
            "    for (int q = 0; q < {rows}; ++q) {{\n        blk_ptr[q] = (device const {pointer_type} *)in{weight} + (long)weight_base[q] + (long)ib_first * {run};\n    }}\n"
        ));
        source.push_str(&format!("    {element_type} acts[{run}];\n"));
        let acts_stride = if other_stride_is_one { String::new() } else { String::from(" * other_stride") };
        source.push_str(&format!("    long acts_step = (long)ib_step * {run}{acts_stride};\n"));
        source.push_str(&format!("    device const {element_type} *acts_row = in{other} + other_base[0] + (long)ib_first * {run}{acts_stride};\n"));
        source.push_str("    for (int ib = ib_first; ib < blocks; ib += ib_step) {\n");
        source.push_str(&format!("        for (int j = 0; j < {run}; ++j) {{\n"));
        source.push_str(if other_stride_is_one { "            acts[j] = acts_row[j];\n" } else { "            acts[j] = acts_row[(long)j * other_stride];\n" });
        source.push_str("        }\n");
        source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{\n            device const {pointer_type} *blk = blk_ptr[q];\n"));
        source.push_str(&format!(
            "            float result = 0.0f;\n            for (uint j = 0u; j < {run}u; ++j) {{\n                result += (float)blk[j] * acts[j];\n            }}\n            sumf[q] = sumf[q] + result;\n        }}\n"
        ));
        source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{ blk_ptr[q] += blk_step; }}\n"));
        source.push_str("        acts_row += acts_step;\n    }\n");
        source
    }

    fn block_msl(&self, launch: &FlatLaunch, reduction_bound: &str, element_type: &str, fold: Fold) -> Result<String, SubBlockMatvecRefusal> {
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
            Fold::ScaleEachTerm => source.push_str(&self.scale_each_term()?),
            Fold::Plain => return Err(SubBlockMatvecRefusal("a plain fold has no scaled block body")),
        }
        source.push_str("        }\n");
        source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{ blk_ptr[q] += blk_step; }}\n"));
        source.push_str("        acts_row += acts_step;\n    }\n");
        Ok(source)
    }

    fn scale_each_term(&self) -> Result<String, SubBlockMatvecRefusal> {
        match self.read {
            CodeRead::ByteRun { codes, signed } => Ok(self.byte_run_terms(codes, signed)),
            CodeRead::Planes { element, multiplier, min_multiplier } => Ok(self.plane_terms(&element, multiplier, min_multiplier)),
            CodeRead::Scalar { .. } => Err(SubBlockMatvecRefusal("a scalar read has no scale to apply")),
        }
    }

    fn plane_terms(&self, element: &Coded, multiplier: i16, min_multiplier: i8) -> String {
        let elements = self.elements;
        let (assembly, code) = coded_assembly(element, "j", "raw");
        let mut source = format!("            float d = {};\n", self.scale_read);
        if let Some(min_read) = &self.min_read {
            source.push_str(&format!("            float minimum = {min_read};\n"));
        }
        source.push_str(&format!(
            "            device const uchar *block = blk;\n            float result = 0.0f;\n            for (uint j = 0u; j < {elements}u; ++j) {{\n{assembly}"
        ));
        source.push_str(&format!("                float value = (({multiplier}.0f * (float){code}) - {}.0f) * d;\n", element.bias));
        if self.min_read.is_some() {
            source.push_str(&format!("                value = value + ({min_multiplier}.0f * minimum);\n"));
        }
        source.push_str("                result += value * acts[j];\n            }\n            sumf[q] = sumf[q] + result;\n");
        source
    }

    fn byte_run_terms(&self, codes: ByteCodes, signed: bool) -> String {
        let cast = if signed { "char" } else { "uchar" };
        let (elements, base) = (self.elements, codes.base_byte);
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
    use proxima_numeric::decode::decode_block;
    use proxima_numeric::float_layout::FloatLayout;
    use proxima_numeric::format::NumericFormat;

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
            .expect("a scaled byte-run form renders under the scale fold")
    }

    #[test]
    fn q8_0_renders_from_its_layout_one_block_per_lane() {
        let source = render(&BlockLayout::Q8_0);
        assert!(source.contains("(long)ib_first) * 34;"), "{source}");
        assert!(source.contains("blk + 2;") && source.contains("(char)flat_block_levels[j]"), "{source}");
        assert!(source.contains("int ib_step = 32;"), "{source}");
    }

    const Q8_0_BODY: &str = "    int blocks = (int)u.reduction_total / 32;
    int ib_first = (int)lane;
    int ib_step = 32;
    long blk_step = (long)ib_step * 34;
    device const uchar *blk_ptr[4];
    for (int q = 0; q < 4; ++q) {
        blk_ptr[q] = in0 + ((long)((int)weight_base[q] / 32) + (long)ib_first) * 34;
    }
    float acts[32];
    long acts_step = (long)ib_step * 32;
    device const float *acts_row = in1 + other_base[0] + (long)ib_first * 32;
    for (int ib = ib_first; ib < blocks; ib += ib_step) {
        for (int j = 0; j < 32; ++j) {
            acts[j] = acts_row[j];
        }
        for (int q = 0; q < 4; ++q) {
            device const uchar *blk = blk_ptr[q];
            float d = (float)(*(device const half *)(blk + 0u));
            device const uchar *flat_block_levels = blk + 2;
            float result = 0.0f;
            for (uint j = 0u; j < 32u; ++j) {
                result += d * (float)((char)flat_block_levels[j]) * acts[j];
            }
            sumf[q] = sumf[q] + result;
        }
        for (int q = 0; q < 4; ++q) { blk_ptr[q] += blk_step; }
        acts_row += acts_step;
    }
";

    #[test]
    fn q8_0_text_is_the_text_rendered_before_code_read_existed() {
        assert_eq!(render(&BlockLayout::Q8_0), Q8_0_BODY);
    }

    fn render_planes(layout: &BlockLayout) -> String {
        FlatBlockForm::plan_planes(layout)
            .expect("the layout has a plane-read form")
            .matvec_msl(
                &FlatLaunch { rows: 4, lanes: 32, weight: 0, other: 1, other_stride_is_one: true },
                "u.reduction_total",
                "float",
                Fold::ScaleEachTerm,
            )
            .expect("a plane-read form renders under the scale fold")
    }

    fn position(source: &str, needle: &str) -> usize {
        source.find(needle).unwrap_or_else(|| panic!("`{needle}` missing from\n{source}"))
    }

    const F16_BYTES: &str = "(float)(*(device const half *)(blk + ";

    #[test]
    fn q5_0_hoists_scale_and_reads_both_planes_with_the_bias() {
        let source = render_planes(&BlockLayout::Q5_0);
        let scale = position(&source, &format!("float d = {F16_BYTES}0u));"));
        assert!(scale < position(&source, "for (uint j = 0u; j < 32u; ++j)"), "{source}");
        assert!(source.contains("uint raw_low_address = 48u;") && source.contains("uint raw_high_address = 16u;"), "{source}");
        assert!(source.contains("(raw_low | (raw_high << 4u))"), "{source}");
        assert!(source.contains("- 16.0f) * d;") && !source.contains("minimum"), "{source}");
        assert!(source.contains("* 22;"), "{source}");
    }

    #[test]
    fn q5_1_hoists_scale_and_min_and_reads_both_planes() {
        let source = render_planes(&BlockLayout::Q5_1);
        let loop_start = position(&source, "for (uint j = 0u; j < 32u; ++j)");
        assert!(position(&source, &format!("float d = {F16_BYTES}0u));")) < loop_start, "{source}");
        assert!(position(&source, &format!("float minimum = {F16_BYTES}2u));")) < loop_start, "{source}");
        assert!(source.contains("uint raw_low_address = 64u;") && source.contains("uint raw_high_address = 32u;"), "{source}");
        assert!(source.contains("value = value + (1.0f * minimum);"), "{source}");
        assert!(source.contains("- 0.0f) * d;") && source.contains("* 24;"), "{source}");
    }

    #[test]
    fn q4_1_hoists_scale_and_min_and_reads_one_plane() {
        let source = render_planes(&BlockLayout::Q4_1);
        let loop_start = position(&source, "for (uint j = 0u; j < 32u; ++j)");
        assert!(position(&source, &format!("float d = {F16_BYTES}0u));")) < loop_start, "{source}");
        assert!(position(&source, &format!("float minimum = {F16_BYTES}2u));")) < loop_start, "{source}");
        assert!(source.contains("uint raw_low_address = 32u;") && !source.contains("raw_high"), "{source}");
        assert!(source.contains("value = value + (1.0f * minimum);") && source.contains("* 20;"), "{source}");
    }

    #[test]
    fn plane_read_refuses_what_has_its_own_body() {
        for layout in [BlockLayout::Q4_K, BlockLayout::Q6_K, BlockLayout::Q8_0] {
            assert!(FlatBlockForm::plan_planes(&layout).is_err(), "{layout:?} is not a plane-read layout");
        }
    }

    fn f16_bytes(bits: u16) -> [u8; 2] {
        bits.to_le_bytes()
    }

    fn qh_bytes() -> [u8; 4] {
        0x8009_8019u32.to_le_bytes()
    }

    fn qs_bytes() -> [u8; 16] {
        core::array::from_fn(|index| ((index as u8) << 4) | (15 - index as u8))
    }

    fn decode(format: &NumericFormat, block: &[u8]) -> Vec<f32> {
        let mut values = vec![0.0f32; 32];
        decode_block(format, block, &mut values).expect("the block decodes");
        values
    }

    /// No host evaluator of MSL exists in this tree, so the arithmetic the body emits is asserted
    /// through proxima-numeric's decoder: the body renders `element_msl`'s expression over the same
    /// `BlockLayout`, and the decoder is the oracle for that expression.
    #[test]
    fn q5_0_design_worked_example_block_decodes_to_the_32_hand_derived_values() {
        let block: Vec<u8> = [&f16_bytes(0x2C00)[..], &qh_bytes()[..], &qs_bytes()[..]].concat();
        assert_eq!(block.iter().map(|byte| format!("{byte:02X}")).collect::<Vec<_>>().join(" "), "00 2C 19 80 09 80 0F 1E 2D 3C 4B 5A 69 78 87 96 A5 B4 C3 D2 E1 F0");
        let expected = [
            0.9375, -0.125, -0.1875, 0.75, 0.6875, -0.375, -0.4375, -0.5, -0.5625, -0.625, -0.6875, -0.75, -0.8125, -0.875, -0.9375, 0.0, 0.0, -0.9375, -0.875, 0.1875,
            -0.75, -0.6875, -0.625, -0.5625, -0.5, -0.4375, -0.375, -0.3125, -0.25, -0.1875, -0.125, 0.9375,
        ];
        assert_eq!(decode(&NumericFormat::Block(BlockLayout::Q5_0), &block), expected);
    }

    #[test]
    fn q5_1_block_decodes_to_code_times_scale_plus_min() {
        let block: Vec<u8> = [&f16_bytes(0x2C00)[..], &f16_bytes(0xB800)[..], &qh_bytes()[..], &qs_bytes()[..]].concat();
        let high_bits = 0x8009_8019u32;
        let expected: Vec<f32> = (0..32u32)
            .map(|index| {
                let low = if index < 16 { 15 - index } else { index - 16 };
                let code = low | (((high_bits >> index) & 1) << 4);
                code as f32 * 0.0625 - 0.5
            })
            .collect();
        assert_eq!(decode(&NumericFormat::Block(BlockLayout::Q5_1), &block), expected);
    }

    #[test]
    fn q4_1_block_decodes_to_code_times_scale_plus_min() {
        let block: Vec<u8> = [&f16_bytes(0x2C00)[..], &f16_bytes(0xB800)[..], &qs_bytes()[..]].concat();
        let expected: Vec<f32> = (0..32u32).map(|index| (if index < 16 { 15 - index } else { index - 16 }) as f32 * 0.0625 - 0.5).collect();
        assert_eq!(decode(&NumericFormat::Block(BlockLayout::Q4_1), &block), expected);
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

    fn render_scalar(layout: FloatLayout) -> String {
        FlatBlockForm::plan_scalar(layout, 8)
            .expect("an eight-element run of a 16-bit float is whole bytes")
            .matvec_msl(
                &FlatLaunch { rows: 4, lanes: 32, weight: 0, other: 1, other_stride_is_one: true },
                "u.reduction_total",
                "float",
                Fold::Plain,
            )
            .expect("an unscaled scalar form renders under the plain fold")
    }

    #[test]
    fn f16_run_body_reads_eight_native_halves_per_lane_with_no_scale() {
        let source = render_scalar(FloatLayout::F16);
        assert!(source.contains("int blocks = (int)u.reduction_total / 8;"), "{source}");
        assert!(source.contains("device const half *blk_ptr[4];"), "{source}");
        assert!(source.contains("blk_ptr[q] = (device const half *)in0 + (long)weight_base[q] + (long)ib_first * 8;"), "{source}");
        assert!(source.contains("result += (float)blk[j] * acts[j];"), "{source}");
        assert!(source.contains("long blk_step = (long)ib_step * 8;"), "{source}");
        assert!(!source.contains("float d = "), "{source}");
    }

    #[test]
    fn bf16_run_body_reads_eight_native_bfloats_per_lane_through_a_cast_run_pointer() {
        let source = render_scalar(FloatLayout::BF16);
        assert!(source.contains("device const bfloat *blk_ptr[4];"), "{source}");
        assert!(source.contains("blk_ptr[q] = (device const bfloat *)in0 + (long)weight_base[q] + (long)ib_first * 8;"), "{source}");
        assert!(source.contains("long blk_step = (long)ib_step * 8;"), "{source}");
        assert!(source.contains("result += (float)blk[j] * acts[j];"), "{source}");
        assert!(!source.contains("bf16_element"), "{source}");
        assert!(!source.contains("uchar"), "{source}");
        assert!(!source.contains("<< 8"), "{source}");
    }

    #[test]
    fn a_fold_the_read_has_no_meaning_under_is_refused() {
        let launch = FlatLaunch { rows: 4, lanes: 32, weight: 0, other: 1, other_stride_is_one: true };
        let scalar = FlatBlockForm::plan_scalar(FloatLayout::F16, 8).expect("f16 run");
        assert!(scalar.matvec_msl(&launch, "u.reduction_total", "float", Fold::ScaleEachTerm).is_err());
        let scaled = FlatBlockForm::plan(&BlockLayout::Q8_0).expect("q8_0 form");
        assert!(scaled.matvec_msl(&launch, "u.reduction_total", "float", Fold::Plain).is_err());
    }

    #[test]
    fn a_run_that_is_not_whole_bytes_is_refused() {
        assert!(FlatBlockForm::plan_scalar(FloatLayout::E2M1, 3).is_err());
        assert!(FlatBlockForm::plan_scalar(FloatLayout::F16, 0).is_err());
    }

    /// design 9.4: eight consecutive weights, little-endian, decoded by the host and compared to the
    /// hand-derived values bitwise; the body reads the same two bytes as `half`, so the host decode
    /// is the oracle for what a lane loads.
    #[test]
    fn f16_design_worked_example_run_decodes_to_the_eight_hand_derived_values() {
        let bytes: [u8; 16] = [0x00, 0x3C, 0x00, 0xBC, 0x00, 0x38, 0x01, 0x00, 0xFF, 0x7B, 0x00, 0xC4, 0x55, 0x35, 0x00, 0x04];
        let mut values = [0.0f32; 8];
        proxima_gguf::quant::f16::dequantize(&bytes, &mut values).expect("eight whole f16 elements");
        let expected = [1.0f32, -1.0, 0.5, 1.0 / 16_777_216.0, 65504.0, -4.0, (1.0 + 341.0 / 1024.0) * 0.25, 1.0 / 16_384.0];
        let bits = |row: &[f32]| row.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&values), bits(&expected));
    }

    #[test]
    fn bf16_design_worked_example_row_decodes_to_the_three_hand_derived_values() {
        let bytes: [u8; 6] = [0x80, 0x3F, 0x00, 0xC0, 0xAB, 0x3E];
        let mut values = [0.0f32; 3];
        proxima_gguf::quant::bf16::dequantize(&bytes, &mut values).expect("three whole bf16 elements");
        let expected = [1.0f32, -2.0, (1.0 + 43.0 / 128.0) * 0.25];
        let bits = |row: &[f32]| row.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&values), bits(&expected));
    }

    #[test]
    fn moving_the_scale_field_changes_the_rendered_kernel() {
        let mut moved = BlockLayout::Q8_0;
        moved.scale.byte_offset = 2;
        assert_ne!(render(&BlockLayout::Q8_0), render(&moved));
    }
}
