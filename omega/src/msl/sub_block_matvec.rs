//! The decode matvec body for a block layout whose sub-blocks carry their own scale and min codes,
//! rendered from the [`BlockLayout`] and a [`SubBlockMatvecGeometry`] and from nothing else.
//!
//! Teaching pointer: `layout_render::block_element_msl` renders one element at a time through the
//! planes' mixed-radix address walk; this renders a whole simdgroup matvec by evaluating that walk
//! on the host. Every address the kernel needs (where a thread's run of codes sits, where each
//! sub-block's scale and min bits sit, which shift puts a code at which bit of a 16-bit word) is
//! `Plane::address` evaluated for the digit values the geometry assigns, so the kernel carries
//! constants and no walk. `cuda::sub_block_dot` is the CUDA sibling: it too reads the sub-block
//! codes by the plane addresses rather than by a format name.
//!
//! Codes are accumulated unshifted: a mask leaves a code at bit `shift` of its word, the
//! accumulator is `2^shift` too large, and the factor `2^-shift` is applied once per sub-block
//! instead of once per element.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_numeric::block_layout::{BlockLayout, Coded, Digit, Field};

use super::layout_render::hardware_type;
use super::sub_block_assembled_matvec::assembled_matvec_msl;

const WORD_BITS: u32 = 16;
const BYTE_BITS: u32 = 8;

/// Where one launch puts its work. Each value is a tunable of the kernel, not a fact of the layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SubBlockMatvecGeometry {
    /// Output rows one simdgroup folds at once: `PACKED_ROWS_PER_GROUP` (emit_and_classify.rs:2676),
    /// ggml's `nr0` (ggml-metal.metal:5086 `kernel_mul_mv_q4_K_f32_impl<4,2,32>`).
    pub rows_per_simdgroup: usize,
    /// Simdgroups per threadgroup: `packed_row_nsg_factor()` (tiled_gemm_cooperative_scan.rs:2100),
    /// ggml's `nsg`. The threadgroup width is sized from it there, so the body only carries it.
    pub simdgroups: usize,
    /// Lanes per simdgroup: `SIMD_WIDTH` (sized.rs:130), ggml's `nw`.
    pub lanes: usize,
    /// Consecutive elements one lane gathers per run: ggml's `y4[i]`, `i < 8`
    /// (ggml-metal.metal:5152-5156).
    pub run_elements: usize,
    /// Lanes that share one block: ggml's `ix = tiisg/8`, `it = tiisg%8` (ggml-metal.metal:5100-5103).
    /// The block count in flight per simdgroup is `lanes / threads_per_block`.
    pub threads_per_block: usize,
    /// Which half of the lane number picks the block and which picks the thread within it. It sets
    /// which lane holds which partial sum, so it fixes the order the simdgroup reduction adds them.
    pub lane_order: LaneOrder,
    /// How wide one read of the element codes is: ggml's Q4_K reads `ushort` pairs
    /// (ggml-metal.metal:5152), its Q5_K reads `uchar` (ggml-metal.metal:5289).
    pub code_read: CodeRead,
}

/// How a lane number splits into the block it works on and the thread slot within that block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LaneOrder {
    /// `ix = lane / threads_per_block`, `it = lane % threads_per_block` (ggml-metal.metal:5100-5103).
    BlocksHigh,
    /// `ix = lane % blocks_in_flight`, `it = lane / blocks_in_flight` (ggml-metal.metal:5244-5250
    /// for Q5_K, :5376-5385 for Q6_K).
    BlocksLow,
}

/// The width of one read of the element codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CodeRead {
    Bytes,
    Words,
}

impl CodeRead {
    const fn lanes(self) -> usize {
        match self {
            Self::Bytes => 1,
            Self::Words => 2,
        }
    }

    const fn type_name(self) -> &'static str {
        match self {
            Self::Bytes => "uchar",
            Self::Words => "ushort",
        }
    }
}

impl SubBlockMatvecGeometry {
    /// ggml's `kernel_mul_mv_q4_K_f32_impl<4,2,32>` launch shape.
    pub(super) const RUN_ELEMENTS: usize = 8;
    pub(super) const THREADS_PER_BLOCK: usize = 8;

    pub(super) const fn decode(rows_per_simdgroup: usize, simdgroups: usize, lanes: usize) -> Self {
        Self {
            rows_per_simdgroup,
            simdgroups,
            lanes,
            run_elements: Self::RUN_ELEMENTS,
            threads_per_block: Self::THREADS_PER_BLOCK,
            lane_order: LaneOrder::BlocksHigh,
            code_read: CodeRead::Words,
        }
    }

    /// ggml's `kernel_mul_mv_q5_K_f32_impl<2,2,32>` launch shape (ggml-metal.metal:5209-5250).
    /// Renders bit-identically to ggml's Q5_K port but measured slower, so only the unit tests select it.
    #[cfg(test)]
    pub(super) const fn decode_high_plane(rows_per_simdgroup: usize, simdgroups: usize, lanes: usize) -> Self {
        Self {
            lane_order: LaneOrder::BlocksLow,
            code_read: CodeRead::Bytes,
            ..Self::decode(rows_per_simdgroup, simdgroups, lanes)
        }
    }

    /// ggml's `kernel_mul_mv_q6_K_f32_impl<1,2,32>` launch shape (ggml-metal.metal:5340-5385):
    /// four elements per lane run, sixteen threads per block.
    pub(super) const fn decode_signed_scales(rows_per_simdgroup: usize, simdgroups: usize, lanes: usize) -> Self {
        Self {
            run_elements: 4,
            threads_per_block: 16,
            lane_order: LaneOrder::BlocksLow,
            code_read: CodeRead::Bytes,
            ..Self::decode(rows_per_simdgroup, simdgroups, lanes)
        }
    }
}

/// The property of the layout or geometry that keeps this renderer from rendering it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("sub-block matvec cannot render this layout: {0}")]
pub(super) struct SubBlockMatvecRefusal(pub &'static str);

pub(super) fn require(condition: bool, reason: &'static str) -> Result<(), SubBlockMatvecRefusal> {
    if condition { Ok(()) } else { Err(SubBlockMatvecRefusal(reason)) }
}

/// How the element index splits into the lane digit, the nibble digit and the group digit, and how
/// the geometry splits those between the thread (owned) and the loop (covered).
struct Walk {
    elements: usize,
    bits: u32,
    element_base_bytes: u32,
    lane: Digit,
    nibble: Digit,
    group: Digit,
    lane_runs: usize,
    owned_groups: usize,
    covered_groups: usize,
    read_lanes: usize,
    span: usize,
    high: Option<HighPlane>,
}

/// The one-bit plane above the low code bits: `weight` is what a set bit adds to the code, the byte
/// of the plane a thread reads is `base_byte + ir * ir_step_bytes`, and `shifts` holds the bit it
/// tests for every `(owned, covered, nibble)`.
struct HighPlane {
    weight: u32,
    base_byte: u32,
    ir_step_bytes: u32,
    shifts: Vec<u32>,
}

impl HighPlane {
    fn shifts_by_owned(&self, walk: &Walk, covered: usize, nibble: usize) -> Vec<u32> {
        (0..walk.owned_groups)
            .map(|owned| self.shifts[(owned * walk.covered_groups + covered) * walk.nibbles() + nibble])
            .collect()
    }
}

impl Walk {
    fn of(layout: &BlockLayout, geometry: &SubBlockMatvecGeometry) -> Result<Self, SubBlockMatvecRefusal> {
        let sub = layout.sub.as_ref().ok_or(SubBlockMatvecRefusal("no sub-blocks"))?;
        require(layout.scale_runs.is_none(), "scale runs")?;
        require(layout.multiplier == 1 && layout.element.bias == 0, "affine element code")?;
        require(layout.element.packed_radix == 0, "plane coded element")?;
        require(sub.scale_multiplier == 1, "sub-block scale multiplier")?;
        require(layout.min.is_some() && sub.min.is_some(), "min and sub-block min")?;
        require(layout.min_multiplier.abs() == 1, "unit min multiplier")?;
        let plane = layout.element.low;
        let [lane, nibble, group, tail] = plane.digits;
        let bits = u32::from(plane.bits);
        require(tail.radix <= 1, "three element digits")?;
        require(u32::from(lane.stride_bits) == BYTE_BITS, "lane digit steps one byte")?;
        require(u32::from(nibble.radix) * bits == BYTE_BITS && u32::from(nibble.stride_bits) == bits, "nibble digit packs a byte")?;
        require(u32::from(group.stride_bits) == u32::from(lane.radix) * BYTE_BITS, "group digit follows the lane bytes")?;
        let run_bits = geometry.run_elements as u32 * BYTE_BITS;
        require(plane.base_bits.is_multiple_of(WORD_BITS) && run_bits.is_multiple_of(WORD_BITS), "run is word aligned")?;
        require(u32::from(group.stride_bits).is_multiple_of(WORD_BITS), "group is word aligned")?;
        let word_lanes = (WORD_BITS / BYTE_BITS) as usize;
        require(geometry.run_elements.is_multiple_of(word_lanes), "run fills whole words")?;
        require(usize::from(lane.radix).is_multiple_of(geometry.run_elements), "run divides the lane digit")?;
        let lane_runs = usize::from(lane.radix) / geometry.run_elements;
        require(geometry.threads_per_block.is_multiple_of(lane_runs), "threads cover the lane runs")?;
        require(geometry.lanes.is_multiple_of(geometry.threads_per_block), "blocks tile the simdgroup")?;
        let owned_groups = geometry.threads_per_block / lane_runs;
        require(usize::from(group.radix).is_multiple_of(owned_groups), "threads divide the group digit")?;
        let covered_groups = usize::from(group.radix) / owned_groups;
        let covered = geometry.run_elements * lane_runs * owned_groups * covered_groups * usize::from(nibble.radix);
        let elements = usize::from(layout.elements);
        require(covered == elements, "the threads cover every element once")?;
        let span = elements / usize::from(sub.count);
        require(span.is_multiple_of(usize::from(lane.radix)), "sub-blocks do not split a lane run")?;
        require(u32::from(nibble.radix) * BYTE_BITS == WORD_BITS, "a sub-block pair is one word")?;
        let mut walk = Self {
            elements,
            bits,
            element_base_bytes: plane.base_bits / BYTE_BITS,
            lane,
            nibble,
            group,
            lane_runs,
            owned_groups,
            covered_groups,
            read_lanes: geometry.code_read.lanes(),
            span,
            high: None,
        };
        walk.high = walk.high_plane(layout, geometry)?;
        Ok(walk)
    }

    /// The element index of the code the thread slot `(owned, covered, nibble)` reads at run
    /// position `lane_index`.
    fn element(&self, owned: usize, covered: usize, nibble: usize, lane_index: usize) -> usize {
        lane_index + usize::from(self.lane.radix) * (nibble + self.nibbles() * (owned + self.owned_groups * covered))
    }

    /// The bit plane above the low one, when the element code has it: where its bit of each tile
    /// element sits, found by evaluating the plane's address walk for every thread slot.
    fn high_plane(&self, layout: &BlockLayout, geometry: &SubBlockMatvecGeometry) -> Result<Option<HighPlane>, SubBlockMatvecRefusal> {
        let Some(plane) = layout.element.high else {
            return Ok(None);
        };
        require(plane.bits == 1, "one-bit high plane")?;
        let lane_digit = plane.digits[0];
        require(lane_digit.radix == self.lane.radix && u32::from(lane_digit.stride_bits) == BYTE_BITS, "high plane lane digit steps one byte")?;
        let base_byte = plane.address(self.element(0, 0, 0, 0)) as u32 / BYTE_BITS;
        let mut shifts = Vec::new();
        for owned in 0..self.owned_groups {
            for covered in 0..self.covered_groups {
                for nibble in 0..self.nibbles() {
                    let at = plane.address(self.element(owned, covered, nibble, 0)) as u32;
                    require(at / BYTE_BITS == base_byte, "high plane bits of a thread share a byte")?;
                    shifts.push(at % BYTE_BITS);
                }
            }
        }
        Ok(Some(HighPlane {
            weight: 1u32 << layout.element.low.bits,
            base_byte,
            ir_step_bytes: geometry.run_elements as u32 * u32::from(lane_digit.stride_bits) / BYTE_BITS,
            shifts,
        }))
    }

    fn nibbles(&self) -> usize {
        usize::from(self.nibble.radix)
    }

    fn nibble_elements(&self) -> usize {
        usize::from(self.lane.radix)
    }

    fn group_elements(&self) -> usize {
        usize::from(self.lane.radix) * self.nibbles()
    }

    fn sub_block(&self, nibble: usize, covered: usize, owned: usize) -> usize {
        (nibble * self.nibble_elements() + (owned + self.owned_groups * covered) * self.group_elements()) / self.span
    }
}

/// A masked piece of one 16-bit word of the block: the word's byte offset for each owned-group value,
/// the shift that lands its bits at their place in the assembled code, and the mask that selects them.
struct Piece {
    offsets: Vec<u32>,
    net_shift: i32,
    mask: u32,
}

fn pieces_of(coded: &Coded, walk: &Walk, covered: usize) -> Result<Vec<Piece>, SubBlockMatvecRefusal> {
    let mut pieces: Vec<Piece> = Vec::new();
    let planes = core::iter::once((coded.low, 0u32)).chain(coded.high.map(|plane| (plane, u32::from(coded.low.bits))));
    for (plane, position) in planes {
        let mut offsets = Vec::new();
        let mut shift = None;
        for owned in 0..walk.owned_groups {
            let first = plane.address(walk.sub_block(0, covered, owned));
            let adjacent = (0..walk.nibbles()).all(|nibble| {
                plane.address(walk.sub_block(nibble, covered, owned)) == first + nibble * BYTE_BITS as usize
            });
            require(adjacent, "sub-block codes sit one byte apart")?;
            let (byte, within) = (first as u32 / BYTE_BITS, first as u32 % BYTE_BITS);
            require(byte.is_multiple_of(WORD_BITS / BYTE_BITS), "sub-block code word is aligned")?;
            require(within + u32::from(plane.bits) <= BYTE_BITS, "sub-block code stays in its byte")?;
            require(shift.is_none_or(|known| known == within), "one in-byte shift")?;
            shift = Some(within);
            offsets.push(byte);
        }
        let net_shift = shift.unwrap_or(0) as i32 - position as i32;
        let positioned = ((1u32 << plane.bits) - 1) << position;
        let mask = (0..walk.nibbles()).fold(0u32, |mask, nibble| mask | (positioned << (BYTE_BITS * nibble as u32)));
        match pieces.iter_mut().find(|piece| piece.offsets == offsets && piece.net_shift == net_shift) {
            Some(known) => known.mask |= mask,
            None => pieces.push(Piece { offsets, net_shift, mask }),
        }
    }
    Ok(pieces)
}

/// The distinct 16-bit words the sub-block codes of a row-block are read from.
#[derive(Default)]
struct Words {
    offsets: Vec<Vec<u32>>,
}

impl Words {
    fn index(&mut self, offsets: &[u32]) -> usize {
        self.offsets.iter().position(|known| known == offsets).unwrap_or_else(|| {
            self.offsets.push(offsets.to_vec());
            self.offsets.len() - 1
        })
    }

    /// The constant byte distance from the first word to word `index` for every owned-group value,
    /// when there is one: such a word is a displacement off the first word's address instead of an
    /// address of its own.
    fn displacement(&self, index: usize) -> Option<i64> {
        let anchor = self.offsets.first()?;
        let first = i64::from(*self.offsets[index].first()?) - i64::from(*anchor.first()?);
        let constant = self.offsets[index]
            .iter()
            .zip(anchor)
            .all(|(offset, base)| i64::from(*offset) - i64::from(*base) == first);
        constant.then_some(first)
    }

    fn preamble(&self) -> String {
        self.offsets
            .iter()
            .enumerate()
            .filter(|(index, _)| *index == 0 || self.displacement(*index).is_none())
            .map(|(index, offsets)| format!("    uint word_offset{index} = {};\n", select_by_owned(offsets)))
            .collect()
    }

    fn loads(&self) -> String {
        let anchor = String::from("            device const uchar *word_base = blk + word_offset0;\n");
        let reads: String = (0..self.offsets.len())
            .map(|index| match (index, self.displacement(index)) {
                (0, _) => String::from("            ushort word0 = *(device const ushort *)word_base;\n"),
                (_, Some(delta)) => format!("            ushort word{index} = *(device const ushort *)(word_base + {delta});\n"),
                (_, None) => format!("            ushort word{index} = *(device const ushort *)(blk + word_offset{index});\n"),
            })
            .collect();
        if self.offsets.is_empty() { String::new() } else { format!("{anchor}{reads}") }
    }
}

fn select_by_owned(values: &[u32]) -> String {
    let Some((last, earlier)) = values.split_last() else {
        return String::from("0u");
    };
    earlier
        .iter()
        .enumerate()
        .rev()
        .fold(format!("{last}u"), |otherwise, (owned, value)| format!("(iq == {owned}u ? {value}u : {otherwise})"))
}

fn assembled(pieces: &[Piece], words: &mut Words) -> String {
    let terms: Vec<String> = pieces
        .iter()
        .map(|piece| {
            let word = format!("word{}", words.index(&piece.offsets));
            let shifted = match piece.net_shift {
                0 => word,
                positive if positive > 0 => format!("({word} >> {positive}u)"),
                negative => format!("({word} << {}u)", -negative),
            };
            format!("({shifted} & (ushort){:#06x}u)", piece.mask)
        })
        .collect();
    format!("(ushort)({})", terms.join(" | "))
}

fn code_value(pair: &str, nibble: usize, last: usize, coded: &Coded) -> String {
    let lane = if nibble == last {
        format!("{pair} >> {}u", nibble as u32 * BYTE_BITS)
    } else {
        format!("{pair} & {:#x}u", (1u32 << BYTE_BITS) - 1)
    };
    let code = format!("(float)(uchar)({lane})");
    if coded.bias == 0 { code } else { format!("({code} - {}.0f)", coded.bias) }
}

pub(super) fn typed_field_read(field: &Field) -> Result<String, SubBlockMatvecRefusal> {
    require(field.pieces.is_none(), "whole stored field")?;
    let native = hardware_type(&field.layout).ok_or(SubBlockMatvecRefusal("field is a hardware type"))?;
    let bytes = u16::from(field.layout.width()) / 8;
    require(field.byte_offset.is_multiple_of(bytes), "field is naturally aligned")?;
    Ok(format!("(float)(*(device const {native} *)(blk + {}u))", field.byte_offset))
}

pub(super) fn lane_split(geometry: &SubBlockMatvecGeometry) -> String {
    let blocks_in_flight = geometry.lanes / geometry.threads_per_block;
    match geometry.lane_order {
        LaneOrder::BlocksHigh => format!(
            "    uint ix = (uint)lane / {}u;\n    uint it = (uint)lane % {}u;\n",
            geometry.threads_per_block, geometry.threads_per_block
        ),
        LaneOrder::BlocksLow => format!("    uint ix = (uint)lane % {blocks_in_flight}u;\n    uint it = (uint)lane / {blocks_in_flight}u;\n"),
    }
}

fn reciprocal(shift: u32) -> String {
    format!("(1.0f/{}.0f)", 1u64 << shift)
}

/// Renders the body of the row-blocked decode matvec: the part between the preamble that declares
/// `lane`, `weight_base`, `other_base`, `other_stride` and `sumf`, and the shared combine and write
/// tail. `reduction_bound` is the token for the reduction length.
///
/// # Errors
/// The property of `layout` or `geometry` this renderer does not cover.
#[must_use = "the refusal names why the layout has no rendered matvec"]
pub(super) fn sub_block_matvec_msl(
    layout: &BlockLayout,
    geometry: &SubBlockMatvecGeometry,
    weight: usize,
    other: usize,
    other_stride_is_one: bool,
    reduction_bound: &str,
) -> Result<String, SubBlockMatvecRefusal> {
    if layout.min.is_none() {
        return assembled_matvec_msl(layout, geometry, weight, other, other_stride_is_one, reduction_bound);
    }
    let walk = Walk::of(layout, geometry)?;
    let sub = layout.sub.as_ref().ok_or(SubBlockMatvecRefusal("no sub-blocks"))?;
    let sub_min = sub.min.as_ref().ok_or(SubBlockMatvecRefusal("no sub-block min"))?;
    let min_field = layout.min.as_ref().ok_or(SubBlockMatvecRefusal("no min"))?;
    let scale_read = typed_field_read(&layout.scale)?;
    let min_read = typed_field_read(min_field)?;
    let rows = geometry.rows_per_simdgroup;
    let run = geometry.run_elements;
    let elements = walk.elements;
    let block_bytes = layout.block_bytes;
    let covered = walk.covered_groups;
    let nibbles = walk.nibbles();
    let ix_stride = geometry.lanes / geometry.threads_per_block;
    let run_offset = format!("{run}u * ir + {}u * iq", walk.group_elements());
    let y_offset = |nibble: usize, group: usize| nibble * walk.nibble_elements() + walk.owned_groups * group * walk.group_elements();
    let mut words = Words::default();
    let mut codes = String::new();
    for (kind, coded) in [("scale", &sub.scale), ("min", sub_min)] {
        for group in 0..covered {
            let pair = assembled(&pieces_of(coded, &walk, group)?, &mut words);
            codes.push_str(&format!("            ushort {kind}_pair{group} = {pair};\n"));
            for nibble in 0..nibbles {
                let value = code_value(&format!("{kind}_pair{group}"), nibble, nibbles - 1, coded);
                codes.push_str(&format!("            float {kind}_{group}_{nibble} = {value};\n"));
            }
        }
    }
    let mut source = String::new();
    source.push_str(&lane_split(geometry));
    source.push_str(&format!("    uint ir = it % {}u;\n    uint iq = it / {}u;\n", walk.lane_runs, walk.lane_runs));
    source.push_str(&format!("    int super_blocks = (int){reduction_bound} / {elements};\n"));
    for group in 0..covered {
        source.push_str(&format!("    float y{group}[{}];\n", nibbles * run));
    }
    source.push_str(&format!("    int ib_first = (int)ix;\n    int ib_step = {ix_stride};\n"));
    source.push_str(&format!("    long blk_step = (long)ib_step * {block_bytes};\n"));
    source.push_str(&format!("    device const uchar *blk_ptr[{rows}];\n"));
    source.push_str(&format!("    for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&format!(
        "        blk_ptr[q] = in{weight} + ((long)((int)weight_base[q] / {elements}) + (long)ib_first) * {block_bytes};\n"
    ));
    source.push_str("    }\n");
    let run_bytes = run as u32 * u32::from(walk.lane.stride_bits) / BYTE_BITS;
    let group_bytes = u32::from(walk.group.stride_bits) / BYTE_BITS;
    source.push_str(&format!(
        "    uint code_offset = {}u + ir * {run_bytes}u + iq * {group_bytes}u;\n",
        walk.element_base_bytes
    ));
    if let Some(high) = &walk.high {
        source.push_str(&format!("    uint high_offset = {}u + ir * {}u;\n", high.base_byte, high.ir_step_bytes));
        for group in 0..covered {
            for nibble in 0..nibbles {
                let shift = select_by_owned(&high.shifts_by_owned(&walk, group, nibble));
                source.push_str(&format!("    uchar hm{group}_{nibble} = (uchar)(1u << {shift});\n"));
            }
        }
    }
    let body = format!("            device const uchar *blk = blk_ptr[q];\n{}{codes}", words.loads());
    source.push_str(&words.preamble());
    if other_stride_is_one {
        source.push_str(&format!("    long y4_step = (long)ib_step * {elements};\n"));
        source.push_str(&format!(
            "    device const float *y4 = in{other} + other_base[0] + (long)ib_first * {elements} + (long)({run_offset});\n"
        ));
    } else {
        source.push_str(&format!("    long y4_step = (long)ib_step * {elements} * other_stride;\n"));
        source.push_str(&format!(
            "    device const float *y4 = in{other} + other_base[0] + (long)ib_first * {elements} * other_stride + (long)({run_offset}) * other_stride;\n"
        ));
    }
    source.push_str("    for (int ib = ib_first; ib < super_blocks; ib += ib_step) {\n");
    for group in 0..covered {
        for nibble in 0..nibbles {
            source.push_str(&format!("        float sumy{group}_{nibble} = 0.0f;\n"));
        }
    }
    source.push_str(&format!("        for (uint i = 0u; i < {run}u; ++i) {{\n"));
    for group in 0..covered {
        for nibble in 0..nibbles {
            let at = y_offset(nibble, group);
            let slot = nibble * run;
            let read = if other_stride_is_one { format!("y4[i + {at}u]") } else { format!("y4[(long)(i + {at}u) * other_stride]") };
            source.push_str(&format!("            y{group}[i + {slot}u] = {read}; sumy{group}_{nibble} += y{group}[i + {slot}u];\n"));
        }
    }
    source.push_str("        }\n");
    source.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&body);
    let read_lanes = walk.read_lanes;
    let read_type = geometry.code_read.type_name();
    let accumulator = |group: usize, nibble: usize, lane: usize| format!("acc{}_{}", group + 1, nibble * read_lanes + lane);
    source.push_str(&format!("            device const {read_type} *codes = (device const {read_type} *)(blk + code_offset);\n"));
    if walk.high.is_some() {
        source.push_str("            device const uchar *high_codes = blk + high_offset;\n");
        for group in 0..covered {
            for nibble in 0..nibbles {
                source.push_str(&format!("            float high_acc{group}_{nibble} = 0.0f;\n"));
            }
        }
    }
    for group in 0..covered {
        for nibble in 0..nibbles {
            for lane in 0..read_lanes {
                source.push_str(&format!("            float {} = 0.0f;\n", accumulator(group, nibble, lane)));
            }
        }
    }
    source.push_str(&format!("            for (uint i = 0u; i < {}u; ++i) {{\n", run / read_lanes));
    if walk.high.is_some() {
        source.push_str("                uchar high_byte = high_codes[i];\n");
    }
    for group in 0..covered {
        let shifted_words = walk.owned_groups as u32 * group as u32 * group_bytes / read_lanes as u32;
        source.push_str(&format!("                {read_type} code_word{group} = codes[i + {shifted_words}u];\n"));
    }
    let nibble_shift = |nibble: usize| nibble as u32 * u32::from(walk.nibble.stride_bits);
    let lane_shift = |lane: usize| lane as u32 * u32::from(walk.lane.stride_bits);
    for group in 0..covered {
        for nibble in 0..nibbles {
            for lane in 0..read_lanes {
                let mask = ((1u32 << walk.bits) - 1) << (lane_shift(lane) + nibble_shift(nibble));
                let mask = if read_lanes == 1 { format!("{mask:#04x}") } else { format!("{mask:#06x}") };
                source.push_str(&format!(
                    "                {} += y{group}[{}u * i + {}u] * (float)(code_word{group} & ({read_type}){mask}u);\n",
                    accumulator(group, nibble, lane),
                    read_lanes,
                    nibble * run + lane
                ));
            }
            if walk.high.is_some() {
                let slot = nibble * run;
                source.push_str(&format!(
                    "                high_acc{group}_{nibble} += (high_byte & hm{group}_{nibble}) != 0u ? y{group}[i + {slot}u] : 0.0f;\n"
                ));
            }
        }
    }
    source.push_str("            }\n");
    source.push_str(&format!("            float dall = {scale_read};\n"));
    source.push_str(&format!("            float dmin = {min_read};\n"));
    let mut scaled = Vec::new();
    for group in 0..covered {
        for nibble in 0..nibbles {
            let folded = (1..read_lanes).fold(accumulator(group, nibble, 0), |sum, lane| {
                format!("{sum} + {} * {}", reciprocal(lane_shift(lane)), accumulator(group, nibble, lane))
            });
            let residual = if nibble_shift(nibble) == 0 { String::new() } else { format!(" * {}", reciprocal(nibble_shift(nibble))) };
            match &walk.high {
                None => scaled.push(format!("({folded}) * scale_{group}_{nibble}{residual}")),
                Some(high) => {
                    let low_term = format!("{folded}{residual}");
                    scaled.push(format!("scale_{group}_{nibble} * ({low_term} + {}.0f * high_acc{group}_{nibble})", high.weight));
                }
            }
        }
    }
    let offsets = (0..covered).flat_map(|group| (0..nibbles).map(move |nibble| format!("sumy{group}_{nibble} * min_{group}_{nibble}")));
    let sign = if layout.min_multiplier < 0 { "-" } else { "+" };
    source.push_str(&body_close(&scaled.join(" +\n                "), &offsets.collect::<Vec<_>>().join(" + "), sign, rows));
    Ok(source)
}

fn body_close(scaled: &str, offsets: &str, sign: &str, rows: usize) -> String {
    let mut text = String::new();
    text.push_str(&format!("            sumf[q] = sumf[q] + dall * ({scaled}) {sign} dmin * ({offsets});\n"));
    text.push_str("        }\n");
    text.push_str(&format!("        for (int q = 0; q < {rows}; ++q) {{ blk_ptr[q] += blk_step; }}\n"));
    text.push_str("        y4 += y4_step;\n    }\n");
    text
}

#[cfg(test)]
mod tests {
    use proxima_numeric::block_layout::BlockLayout;

    use super::{SubBlockMatvecGeometry, sub_block_matvec_msl};

    fn render(layout: &BlockLayout) -> Result<alloc::string::String, super::SubBlockMatvecRefusal> {
        sub_block_matvec_msl(layout, &SubBlockMatvecGeometry::decode(4, 2, 32), 0, 1, true, "u.reduction_total")
    }

    #[test]
    fn q4_k_renders_from_its_layout() {
        let source = render(&BlockLayout::Q4_K).expect("q4_k renders");
        assert!(source.contains("0x0f0f") && source.contains("0x3030") && source.contains("(1.0f/256.0f)"), "{source}");
    }

    #[test]
    fn one_field_of_the_layout_changes_the_rendered_kernel() {
        let mut moved = BlockLayout::Q4_K;
        let sub = moved.sub.as_mut().expect("q4_k has sub-blocks");
        sub.scale.low.base_bits += 16;
        assert_ne!(render(&BlockLayout::Q4_K).expect("q4_k renders"), render(&moved).expect("the moved layout renders"));
    }

    #[test]
    fn layouts_without_paired_sub_block_codes_are_refused() {
        for layout in [BlockLayout::Q4_0, BlockLayout::Q4_1, BlockLayout::Q8_K] {
            assert!(render(&layout).is_err(), "{layout:?} has no rendered matvec");
        }
    }

    #[test]
    fn q5_k_renders_its_high_plane_gate_from_its_layout() {
        let geometry = SubBlockMatvecGeometry::decode_high_plane(2, 2, 32);
        let source = sub_block_matvec_msl(&BlockLayout::Q5_K, &geometry, 0, 1, true, "u.reduction_total").expect("q5_k renders");
        assert!(source.contains("high_acc0_0") && source.contains("16.0f * high_acc1_1") && source.contains("(1.0f/16.0f)"), "{source}");
    }

    #[test]
    fn q6_k_renders_its_assembled_codes_from_its_layout() {
        let geometry = SubBlockMatvecGeometry::decode_signed_scales(1, 2, 32);
        let source = sub_block_matvec_msl(&BlockLayout::Q6_K, &geometry, 0, 1, true, "u.reduction_total").expect("q6_k renders");
        assert!(source.contains("- 32)") && source.contains("(float)(char)scales[6u]") && source.contains("high_codes"), "{source}");
    }

    #[test]
    fn one_field_of_the_q6_k_layout_changes_the_rendered_kernel() {
        let geometry = SubBlockMatvecGeometry::decode_signed_scales(1, 2, 32);
        let mut moved = BlockLayout::Q6_K;
        moved.element.bias = 31;
        let render_with = |layout: &BlockLayout| sub_block_matvec_msl(layout, &geometry, 0, 1, true, "u.reduction_total").expect("q6_k renders");
        assert_ne!(render_with(&BlockLayout::Q6_K), render_with(&moved));
    }
}
