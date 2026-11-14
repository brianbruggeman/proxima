//! The decode matvec body for a block layout whose element code is assembled from a low and a high
//! bit plane and whose sub-blocks carry a signed scale and no min, rendered from the [`BlockLayout`]
//! and a [`SubBlockMatvecGeometry`] and from nothing else.
//!
//! Teaching pointer: [`super::sub_block_matvec`] renders the layouts whose sub-blocks carry a scale
//! and a min and defers the scale multiply per sub-block over unshifted codes. This is its sibling
//! for the layouts that assemble the whole code first (`Q6_K`): every element's code is read from
//! its planes (`Plane::address` evaluated on the host for the element the thread slot names), biased,
//! and multiplied by the activation, and each sub-block's sum is scaled once. The kernel carries
//! constants and a per-thread base; it carries no address walk.
//!
//! A thread owns one run of `run_elements` consecutive lane-digit elements (`ir`) and one value of
//! the top element digit (`iq`); it covers every other digit in a compile-time tile.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use proxima_numeric::FloatLayout;
use proxima_numeric::LevelTable;
use proxima_numeric::block_layout::{BlockLayout, Plane, SubBlocks};

use super::sub_block_matvec::{SubBlockMatvecGeometry, SubBlockMatvecRefusal, lane_split, require, select_by_owned, typed_field_read};
use super::tiled_gemm_cooperative_scan::FULL_UNROLL;

const BYTE_BITS: u32 = 8;

/// Where one plane's bits for a tile element sit, relative to the thread's base of that plane.
struct PlaneReads {
    /// Byte of the plane a thread at `ir = 0`, `iq = 0` reads for the first element of the run.
    base_byte: u32,
    /// Bytes the base moves per `ir` step and per `iq` step.
    ir_bytes: u32,
    owned_bytes: u32,
    /// Per tile slot: byte offset from the base and in-byte shift.
    slots: Vec<(u32, u32)>,
    bits: u32,
    /// Whether the plane's address moves across the thread's run. A grid code plane does not: one
    /// code covers the whole run, so the read drops the run index.
    per_run: bool,
}

struct Tile {
    lane_elements: usize,
    run: usize,
    lane_runs: usize,
    owned: usize,
    covered: usize,
}

impl Tile {
    fn element(&self, ir: usize, owned: usize, slot: usize, run_index: usize) -> usize {
        ir * self.run + run_index + self.lane_elements * (slot + self.covered * owned)
    }

    /// `run_step_bits` is how far the plane's address moves per run element: a byte for a per-element
    /// code, the plane's own width for a sign plane, zero for a grid code shared by the run.
    fn plane_reads(&self, plane: &Plane, run_step_bits: usize) -> Result<PlaneReads, SubBlockMatvecRefusal> {
        let origin = plane.address(self.element(0, 0, 0, 0));
        let ir_bits = plane.address(self.element(1.min(self.lane_runs - 1), 0, 0, 0)) - origin;
        let owned_bits = plane.address(self.element(0, 1.min(self.owned - 1), 0, 0)) - origin;
        require(ir_bits.is_multiple_of(BYTE_BITS as usize) && owned_bits.is_multiple_of(BYTE_BITS as usize), "plane steps are whole bytes")?;
        require(origin.is_multiple_of(BYTE_BITS as usize), "plane base is byte aligned")?;
        let mut slots = Vec::new();
        for slot in 0..self.covered {
            let first = plane.address(self.element(0, 0, slot, 0)) - origin;
            let shift = (first as u32) % BYTE_BITS;
            require(shift + u32::from(plane.bits) <= BYTE_BITS, "a code piece stays in its byte")?;
            for ir in 0..self.lane_runs {
                for owned in 0..self.owned {
                    for run_index in 0..self.run {
                        let at = plane.address(self.element(ir, owned, slot, run_index));
                        let expected = origin + first + ir * ir_bits + owned * owned_bits + run_index * run_step_bits;
                        require(at == expected, "plane address is affine in the thread slot")?;
                    }
                }
            }
            slots.push((first as u32 / BYTE_BITS, shift));
        }
        Ok(PlaneReads {
            base_byte: origin as u32 / BYTE_BITS,
            ir_bytes: ir_bits as u32 / BYTE_BITS,
            owned_bytes: owned_bits as u32 / BYTE_BITS,
            slots,
            bits: u32::from(plane.bits),
            per_run: run_step_bits != 0,
        })
    }

    /// The sign plane's per-slot read: the run's sign bits step by the plane's width, so they sit
    /// contiguously from `shift` and one masked byte read holds all of them.
    fn sign_reads(&self, sign: &Plane) -> Result<PlaneReads, SubBlockMatvecRefusal> {
        let reads = self.plane_reads(sign, usize::from(sign.bits))?;
        let run_bits = self.run as u32 * reads.bits;
        require(reads.slots.iter().all(|(_, shift)| shift + run_bits <= BYTE_BITS), "a run of sign bits stays in its byte")?;
        Ok(reads)
    }
}

fn piece(pointer: &str, reads: &PlaneReads, slot: usize) -> String {
    let (byte, shift) = reads.slots[slot];
    let mask = (1u32 << reads.bits) - 1;
    let index = if reads.per_run { format!("i + {byte}u") } else { format!("{byte}u") };
    format!("(((uint){pointer}[{index}] >> {shift}u) & {mask:#x}u)")
}

/// Renders the body of the row-blocked decode matvec for a plane-assembled layout: the part between
/// the preamble that declares `lane`, `weight_base`, `other_base`, `other_stride` and `sumf`, and the
/// shared combine and write tail.
///
/// # Errors
/// The property of `layout` or `geometry` this renderer does not cover.
#[must_use = "the refusal names why the layout has no rendered matvec"]
pub(super) fn assembled_matvec_msl(
    layout: &BlockLayout,
    geometry: &SubBlockMatvecGeometry,
    weight: usize,
    other: usize,
    other_stride_is_one: bool,
    reduction_bound: &str,
) -> Result<String, SubBlockMatvecRefusal> {
    render(layout, None, geometry, weight, other, other_stride_is_one, reduction_bound)
}

/// How a level format's code becomes a value: the table, and the name of the element accessor whose
/// rendered `{element_fn}_code(uint)` helper (`layout_render::level_block_element_msl`) the kernel
/// preamble already defines for the cooperative path.
pub(super) struct LevelCodes<'a> {
    pub table: &'a LevelTable,
    pub element_fn: &'a str,
}

/// [`assembled_matvec_msl`] for a level-set format (IQ4_XS): the element code is resolved to its
/// value through the level table's helper instead of the layout's affine map, and the sub-block
/// scales are unsigned codes assembled from their planes, `code - bias`.
///
/// # Errors
/// The property of `layout` or `geometry` this renderer does not cover.
#[must_use = "the refusal names why the format has no rendered matvec"]
pub(super) fn level_matvec_msl(
    layout: &BlockLayout,
    codes: &LevelCodes<'_>,
    geometry: &SubBlockMatvecGeometry,
    weight: usize,
    other: usize,
    other_stride_is_one: bool,
    reduction_bound: &str,
) -> Result<String, SubBlockMatvecRefusal> {
    require(level_table_rendered(codes.table), "a level table the matvec renders")?;
    render(layout, Some(codes), geometry, weight, other, other_stride_is_one, reduction_bound)
}

/// Whether the level table is one the rendered level matvec can resolve: an explicit or linear
/// ladder, or a signed grid whose sign is a plane bit. The same set as `layout_render`'s
/// `renderable_level_layout`, so admission and the renderer share one predicate.
pub(super) const fn level_table_rendered(table: &LevelTable) -> bool {
    match table {
        LevelTable::Explicit(_) | LevelTable::Linear { .. } => true,
        LevelTable::SignedGrid { sign_table: None, lanes, .. } => *lanes > 0,
        _ => false,
    }
}

/// Where the thread's sub-block scales sit in the block, evaluated for every thread slot.
enum SubScales {
    /// One signed byte per sub-block (`Q6_K`): the thread reads `scales[...]` off a shared base.
    Bytes,
    /// An unsigned code assembled from planes (`IQ4_XS`): per distinct sub-block of the thread's
    /// slots, the byte and in-byte shift of each plane for every owned-group value.
    Planes(PlaneScales),
}

struct PlaneScales {
    multiplier: i16,
    bias: i16,
    planes: Vec<(u8, u32)>,
    distinct: Vec<Vec<(Vec<u32>, Vec<u32>)>>,
    scale_of_slot: Vec<usize>,
}

impl PlaneScales {
    fn of(sub: &SubBlocks, tile: &Tile, span: usize) -> Result<Self, SubBlockMatvecRefusal> {
        let low = sub.scale.low;
        let mut planes = vec![(low.bits, 0u32)];
        let mut sources = vec![low];
        if let Some(high) = sub.scale.high {
            planes.push((high.bits, u32::from(low.bits)));
            sources.push(high);
        }
        let mut distinct: Vec<Vec<(Vec<u32>, Vec<u32>)>> = Vec::new();
        let mut scale_of_slot = Vec::new();
        for slot in 0..tile.covered {
            let mut per_plane: Vec<(Vec<u32>, Vec<u32>)> = vec![(Vec::new(), Vec::new()); sources.len()];
            for owned in 0..tile.owned {
                let sub_block = tile.element(0, owned, slot, 0) / span;
                let uniform = (0..tile.lane_runs).all(|ir| (0..tile.run).all(|index| tile.element(ir, owned, slot, index) / span == sub_block));
                require(uniform, "a thread slot stays inside one sub-block")?;
                for (plane, (offsets, shifts)) in sources.iter().zip(per_plane.iter_mut()) {
                    let at = plane.address(sub_block) as u32;
                    require(at % BYTE_BITS + u32::from(plane.bits) <= BYTE_BITS, "a scale piece stays in its byte")?;
                    offsets.push(at / BYTE_BITS);
                    shifts.push(at % BYTE_BITS);
                }
            }
            let index = distinct.iter().position(|known| *known == per_plane).unwrap_or_else(|| {
                distinct.push(per_plane);
                distinct.len() - 1
            });
            scale_of_slot.push(index);
        }
        Ok(Self { multiplier: sub.scale_multiplier, bias: sub.scale.bias, planes, distinct, scale_of_slot })
    }

    fn preamble(&self) -> String {
        let mut text = String::new();
        for (scale, per_plane) in self.distinct.iter().enumerate() {
            for (plane, (offsets, shifts)) in per_plane.iter().enumerate() {
                text.push_str(&format!("    uint scale{scale}_offset{plane} = {};\n", select_by_owned(offsets)));
                text.push_str(&format!("    uint scale{scale}_shift{plane} = {};\n", select_by_owned(shifts)));
            }
        }
        text
    }

    fn reads(&self) -> String {
        let mut text = String::new();
        for scale in 0..self.distinct.len() {
            let pieces: Vec<String> = self
                .planes
                .iter()
                .enumerate()
                .map(|(plane, (bits, position))| {
                    let read = format!("(((uint)blk[scale{scale}_offset{plane}] >> scale{scale}_shift{plane}) & {:#x}u)", (1u32 << bits) - 1);
                    if *position == 0 { read } else { format!("({read} << {position}u)") }
                })
                .collect();
            let code = format!("(int)({})", pieces.join(" | "));
            let scaled = if self.multiplier == 1 { code } else { format!("{} * {code}", self.multiplier) };
            text.push_str(&format!("            float scale{scale} = (float)({scaled} - {});\n", self.bias));
        }
        text
    }

    fn term(&self, slot: usize) -> String {
        format!("scale{}", self.scale_of_slot[slot])
    }
}

/// Everything the single-row and the token-group body read from the layout and the geometry, so the
/// two bodies cannot disagree about where a code, a scale or an activation sits.
struct Assembled<'a> {
    layout: &'a BlockLayout,
    codes: Option<&'a LevelCodes<'a>>,
    geometry: &'a SubBlockMatvecGeometry,
    run: usize,
    lane_runs: usize,
    covered: usize,
    elements: usize,
    owned_elements: usize,
    lane_elements: usize,
    span: usize,
    low_reads: PlaneReads,
    high_reads: Option<PlaneReads>,
    /// A signed grid's sign plane: per slot, the byte and shift of the first of the run's sign bits.
    sign_reads: Option<PlaneReads>,
    scale_read: String,
    scales: SubScales,
    scale_base_byte: u32,
}

impl<'a> Assembled<'a> {
    fn of(layout: &'a BlockLayout, codes: Option<&'a LevelCodes<'a>>, geometry: &'a SubBlockMatvecGeometry) -> Result<Self, SubBlockMatvecRefusal> {
        let sub = layout.sub.as_ref().ok_or(SubBlockMatvecRefusal("no sub-blocks"))?;
        require(layout.scale_runs.is_none() && layout.multiplier == 1, "plain code multiplier")?;
        require(layout.min.is_none() && sub.min.is_none(), "no min")?;
        require(layout.element.packed_radix == 0, "plane coded element")?;
        let scale_plane = sub.scale.low;
        let scale_digit = scale_plane.digits[0];
        let signed_bytes = sub.scale.code == FloatLayout::INT8;
        require(!signed_bytes || sub.scale_multiplier == 1, "plain Q6_K scale")?;
        if signed_bytes {
            require(sub.scale.high.is_none() && sub.scale.bias == 0, "whole sub-block scale code")?;
            require(scale_plane.bits == 8 && scale_digit.radix == sub.count && u32::from(scale_digit.stride_bits) == BYTE_BITS, "one scale byte per sub-block")?;
            require(scale_plane.base_bits.is_multiple_of(BYTE_BITS), "scale bytes are aligned")?;
        }
        let low = layout.element.low;
        let lane_digit = low.digits[0];
        let grid_sign = match codes {
            Some(LevelCodes { table: LevelTable::SignedGrid { lanes, sign, .. }, .. }) => {
                require(lane_digit.stride_bits == 0 && lane_digit.radix == u16::from(*lanes), "a grid code covers its lanes")?;
                require(geometry.run_elements == usize::from(*lanes), "a grid run is its lanes")?;
                Some(sign)
            }
            _ => {
                require(u32::from(lane_digit.stride_bits) == BYTE_BITS, "lane digit steps one byte")?;
                None
            }
        };
        let lane_elements = usize::from(lane_digit.radix);
        let run = geometry.run_elements;
        require(lane_elements.is_multiple_of(run), "run divides the lane digit")?;
        let lane_runs = lane_elements / run;
        require(geometry.threads_per_block.is_multiple_of(lane_runs) && geometry.lanes.is_multiple_of(geometry.threads_per_block), "threads tile the block")?;
        let owned = geometry.threads_per_block / lane_runs;
        let elements = usize::from(layout.elements);
        let rest = elements / lane_elements;
        require(rest.is_multiple_of(owned) && owned >= 1, "threads divide the rest digits")?;
        let covered = rest / owned;
        let span = elements / usize::from(sub.count);
        require(run <= span && span.is_multiple_of(run), "sub-blocks hold whole runs")?;
        require(!signed_bytes || lane_elements.is_multiple_of(span), "sub-blocks hold whole lane digits")?;
        let tile = Tile { lane_elements, run, lane_runs, owned, covered };
        let scales = if signed_bytes { SubScales::Bytes } else { SubScales::Planes(PlaneScales::of(sub, &tile, span)?) };
        let code_step = if grid_sign.is_some() { 0 } else { BYTE_BITS as usize };
        let low_reads = tile.plane_reads(&low, code_step)?;
        let high_reads = layout.element.high.as_ref().map(|plane| tile.plane_reads(plane, code_step)).transpose()?;
        let sign_reads = grid_sign.map(|sign| tile.sign_reads(sign)).transpose()?;
        let scale_read = typed_field_read(&layout.scale)?;
        Ok(Self {
            layout,
            codes,
            geometry,
            run,
            lane_runs,
            covered,
            elements,
            owned_elements: lane_elements * covered,
            lane_elements,
            span,
            low_reads,
            high_reads,
            sign_reads,
            scale_read,
            scales,
            scale_base_byte: scale_plane.base_bits / BYTE_BITS,
        })
    }

    /// The lane split, the block walk and the per-thread base offsets, up to and including the scale
    /// offsets: everything both bodies read before they diverge on how the activations are held.
    fn walk_msl(&self, weight: usize, reduction_bound: &str, activation_scratch: &str) -> String {
        let (run, lane_runs, elements) = (self.run, self.lane_runs, self.elements);
        let rows = self.geometry.rows_per_simdgroup;
        let block_bytes = self.layout.block_bytes;
        let ix_stride = self.geometry.lanes / self.geometry.threads_per_block;
        let mut source = String::new();
        source.push_str(&lane_split(self.geometry));
        source.push_str(&format!("    uint ir = it % {lane_runs}u;\n    uint iq = it / {lane_runs}u;\n"));
        source.push_str(&format!("    int super_blocks = (int){reduction_bound} / {elements};\n"));
        source.push_str(activation_scratch);
        source.push_str(&format!("    int ib_first = (int)ix;\n    int ib_step = {ix_stride};\n"));
        source.push_str(&format!("    long blk_step = (long)ib_step * {block_bytes};\n"));
        source.push_str(&format!("    device const uchar *blk_ptr[{rows}];\n"));
        source.push_str(&format!("    {FULL_UNROLL}\n    for (int q = 0; q < {rows}; ++q) {{\n"));
        source.push_str(&format!(
            "        blk_ptr[q] = in{weight} + ((long)((int)weight_base[q] / {elements}) + (long)ib_first) * {block_bytes};\n"
        ));
        source.push_str("    }\n");
        source.push_str(&format!(
            "    uint low_offset = {}u + ir * {}u + iq * {}u;\n",
            self.low_reads.base_byte, self.low_reads.ir_bytes, self.low_reads.owned_bytes
        ));
        if let Some(reads) = &self.high_reads {
            source.push_str(&format!("    uint high_offset = {}u + ir * {}u + iq * {}u;\n", reads.base_byte, reads.ir_bytes, reads.owned_bytes));
        }
        if let Some(reads) = &self.sign_reads {
            source.push_str(&format!("    uint sign_offset = {}u + ir * {}u + iq * {}u;\n", reads.base_byte, reads.ir_bytes, reads.owned_bytes));
        }
        match &self.scales {
            SubScales::Bytes => {
                let (span, sub_per_owned) = (self.span, self.owned_elements / self.span);
                source.push_str(&format!("    uint scale_offset = {}u + (ir * {run}u) / {span}u + iq * {sub_per_owned}u;\n", self.scale_base_byte));
            }
            SubScales::Planes(planes) => source.push_str(&planes.preamble()),
        }
        source
    }

    /// The element's value from its planes for `slot`: the level-table helper for a level format,
    /// else the biased code as a float, what `Plane::address` named.
    fn level_msl(&self, slot: usize) -> String {
        let code = self.code_msl(slot);
        match self.codes {
            Some(LevelCodes { table: LevelTable::SignedGrid { .. }, .. }) => format!("level{slot}"),
            Some(levels) => format!("{}_code((uint)({code}))", levels.element_fn),
            None if self.layout.element.bias == 0 => format!("(float)(int)({code})"),
            None => format!("(float)((int)({code}) - {})", self.layout.element.bias),
        }
    }

    /// The slot's assembled code: the low plane's piece with the high plane's piece above it.
    fn code_msl(&self, slot: usize) -> String {
        let code = piece("low_codes", &self.low_reads, slot);
        match &self.high_reads {
            Some(reads) => format!("{code} | ({} << {}u)", piece("high_codes", reads, slot), self.low_reads.bits),
            None => code,
        }
    }

    /// A signed grid's per-slot reads, hoisted above the run loop: the four magnitudes the slot's code
    /// names and the run's sign bits. Empty for every other table.
    fn slot_prelude_msl(&self, slot: usize) -> String {
        let (Some(LevelCodes { table: LevelTable::SignedGrid { .. }, element_fn }), Some(reads)) = (self.codes, &self.sign_reads) else {
            return String::new();
        };
        let (byte, shift) = reads.slots[slot];
        let mask = (1u32 << self.run) - 1;
        format!(
            "            uint grid{slot} = {element_fn}_grid[{}];\n            uint sign{slot} = ((uint)sign_codes[{byte}u] >> {shift}u) & {mask:#x}u;\n",
            self.code_msl(slot)
        )
    }

    /// The run element's level for a signed grid, mirroring `layout_render::signed_grid_raw_msl`: the
    /// lane's magnitude byte, negated when its sign bit is set, scaled by `0.5^unit_shift`.
    fn level_statements_msl(&self, slot: usize) -> String {
        let Some(LevelCodes { table: LevelTable::SignedGrid { unit_shift, .. }, .. }) = self.codes else {
            return String::new();
        };
        let unit = if *unit_shift == 0 { String::new() } else { format!(" * {:e}f", 0.5f32.powi(i32::from(*unit_shift))) };
        format!(
            "                float level{slot} = (float)((grid{slot} >> (8u * i)) & 0xFFu){unit};\n                level{slot} = ((sign{slot} >> i) & 1u) != 0u ? -level{slot} : level{slot};\n"
        )
    }

    /// The pointers into one block the decode reads, declared at the top of the per-row body.
    fn row_pointers_msl(&self) -> String {
        let mut source = String::new();
        source.push_str("            device const uchar *blk = blk_ptr[q];\n");
        source.push_str("            device const uchar *low_codes = blk + low_offset;\n");
        if self.high_reads.is_some() {
            source.push_str("            device const uchar *high_codes = blk + high_offset;\n");
        }
        if self.sign_reads.is_some() {
            source.push_str("            device const uchar *sign_codes = blk + sign_offset;\n");
        }
        match &self.scales {
            SubScales::Bytes => source.push_str("            device const uchar *scales = blk + scale_offset;\n"),
            SubScales::Planes(planes) => source.push_str(&planes.reads()),
        }
        source.push_str(&format!("            float dall = {};\n", self.scale_read));
        source
    }

    /// The scaled fold of the per-slot sums into the row's running sum: `dall * (sums0 * s0 + ..)`.
    fn fold_msl(&self, accumulator: &str) -> String {
        let terms: Vec<String> = (0..self.covered)
            .map(|slot| match &self.scales {
                SubScales::Bytes => format!("sums{slot} * (float)(char)scales[{}u]", slot * self.lane_elements / self.span),
                SubScales::Planes(planes) => format!("sums{slot} * {}", planes.term(slot)),
            })
            .collect();
        format!("            {accumulator} = {accumulator} + dall * ({});\n", terms.join(" + "))
    }
}

fn render(
    layout: &BlockLayout,
    codes: Option<&LevelCodes<'_>>,
    geometry: &SubBlockMatvecGeometry,
    weight: usize,
    other: usize,
    other_stride_is_one: bool,
    reduction_bound: &str,
) -> Result<String, SubBlockMatvecRefusal> {
    let plan = Assembled::of(layout, codes, geometry)?;
    let (run, covered, elements, lane_elements) = (plan.run, plan.covered, plan.elements, plan.lane_elements);
    let rows = geometry.rows_per_simdgroup;
    let run_offset = format!("{run}u * ir + {}u * iq", plan.owned_elements);
    let mut source = plan.walk_msl(weight, reduction_bound, &format!("    float yl[{}];\n", covered * run));
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
    source.push_str(&format!("        {FULL_UNROLL}\n        for (uint i = 0u; i < {run}u; ++i) {{\n"));
    for slot in 0..covered {
        let at = slot * lane_elements;
        let read = if other_stride_is_one { format!("y4[i + {at}u]") } else { format!("y4[(long)(i + {at}u) * other_stride]") };
        source.push_str(&format!("            yl[i + {}u] = {read};\n", slot * run));
    }
    source.push_str("        }\n");
    source.push_str(&format!("        {FULL_UNROLL}\n        for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&plan.row_pointers_msl());
    for slot in 0..covered {
        source.push_str(&format!("            float sums{slot} = 0.0f;\n"));
        source.push_str(&plan.slot_prelude_msl(slot));
    }
    source.push_str(&format!("            {FULL_UNROLL}\n            for (uint i = 0u; i < {run}u; ++i) {{\n"));
    for slot in 0..covered {
        source.push_str(&plan.level_statements_msl(slot));
        source.push_str(&format!("                sums{slot} += yl[i + {}u] * {};\n", slot * run, plan.level_msl(slot)));
    }
    source.push_str("            }\n");
    source.push_str(&plan.fold_msl("sumf[q]"));
    source.push_str("        }\n");
    source.push_str(&format!("        {FULL_UNROLL}\n        for (int q = 0; q < {rows}; ++q) {{ blk_ptr[q] += blk_step; }}\n"));
    source.push_str("        y4 += y4_step;\n    }\n");
    Ok(source)
}

/// Renders the same decode for a group of `token_cap` activation rows (`sumf[token_cap][rows]`,
/// `other_base[token_cap]`): the part between the preamble that declares them and the shared combine
/// and write tail. Each block's codes are read from their planes once per row and held in registers;
/// every token then folds its own run of activations against them in the order the single-row body
/// folds, so a token's result is its single-row result bit for bit.
///
/// `token_cap` is `packed_row_activation_cap(..)`, the group the dispatch grid and the kernel
/// identity also read. Only layouts with one signed scale byte per sub-block (`Q6_K`) are covered.
///
/// # Errors
/// The property of `layout` or `geometry` this renderer does not cover.
#[must_use = "the refusal names why the layout has no rendered matvec"]
pub(super) fn assembled_multi_row_matvec_msl(
    layout: &BlockLayout,
    geometry: &SubBlockMatvecGeometry,
    weight: usize,
    other: usize,
    other_stride_is_one: bool,
    reduction_bound: &str,
    token_cap: usize,
) -> Result<String, SubBlockMatvecRefusal> {
    let plan = Assembled::of(layout, None, geometry)?;
    require(matches!(plan.scales, SubScales::Bytes), "one signed scale byte per sub-block")?;
    let (run, covered, elements, lane_elements) = (plan.run, plan.covered, plan.elements, plan.lane_elements);
    let rows = geometry.rows_per_simdgroup;
    let stride = if other_stride_is_one { "" } else { " * other_stride" };
    let mut source = plan.walk_msl(weight, reduction_bound, "");
    source.push_str(&format!("    long y_step = (long)ib_step * {elements}{stride};\n"));
    source.push_str(&format!(
        "    long y_offset = (long)ib_first * {elements}{stride} + (long)({run}u * ir + {}u * iq){stride};\n",
        plan.owned_elements
    ));
    source.push_str("    for (int ib = ib_first; ib < super_blocks; ib += ib_step) {\n");
    source.push_str(&format!("        {FULL_UNROLL}\n        for (int q = 0; q < {rows}; ++q) {{\n"));
    source.push_str(&plan.row_pointers_msl());
    for slot in 0..covered {
        source.push_str(&format!("            float level{slot}[{run}];\n"));
    }
    source.push_str(&format!("            {FULL_UNROLL}\n            for (uint i = 0u; i < {run}u; ++i) {{\n"));
    for slot in 0..covered {
        source.push_str(&format!("                level{slot}[i] = {};\n", plan.level_msl(slot)));
    }
    source.push_str("            }\n");
    source.push_str(&format!("            {FULL_UNROLL}\n            for (int s = 0; s < {token_cap}; ++s) {{\n"));
    source.push_str(&format!("                device const float *y4 = in{other} + other_base[s] + y_offset;\n"));
    for slot in 0..covered {
        source.push_str(&format!("                float sums{slot} = 0.0f;\n"));
    }
    source.push_str(&format!("                {FULL_UNROLL}\n                for (uint i = 0u; i < {run}u; ++i) {{\n"));
    for slot in 0..covered {
        let at = slot * lane_elements;
        let read = if other_stride_is_one { format!("y4[i + {at}u]") } else { format!("y4[(long)(i + {at}u) * other_stride]") };
        source.push_str(&format!("                    sums{slot} += {read} * level{slot}[i];\n"));
    }
    source.push_str("                }\n");
    source.push_str(&plan.fold_msl("sumf[s][q]").replacen("            ", "                ", 1));
    source.push_str("            }\n        }\n");
    source.push_str(&format!("        {FULL_UNROLL}\n        for (int q = 0; q < {rows}; ++q) {{ blk_ptr[q] += blk_step; }}\n"));
    source.push_str("        y_offset += y_step;\n    }\n");
    Ok(source)
}

#[cfg(test)]
mod tests {
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;

    use proxima_numeric::LevelTable;
    use proxima_numeric::block_layout::BlockLayout;

    use super::{Assembled, LevelCodes, PlaneReads, SubBlockMatvecGeometry, SubBlockMatvecRefusal, SubScales, level_matvec_msl, level_table_rendered};
    use crate::Codec;

    fn iq4_xs_table() -> LevelTable {
        let Codec::Levels(format) = Codec::Iq4Xs else {
            panic!("iq4_xs is a level-set format");
        };
        format.levels
    }

    fn render(layout: &BlockLayout, table: &LevelTable) -> Result<String, SubBlockMatvecRefusal> {
        let element_fn = if matches!(table, LevelTable::SignedGrid { .. }) { "iq3_s_element" } else { "iq4_xs_element" };
        let codes = LevelCodes { table, element_fn };
        level_matvec_msl(layout, &codes, &SubBlockMatvecGeometry::for_levels(table, 4, 2, 32), 0, 1, true, "u.reduction_total")
    }

    #[test]
    fn iq4_xs_renders_its_level_matvec_from_its_layout() {
        let source = render(&BlockLayout::IQ4_XS, &iq4_xs_table()).expect("iq4_xs renders");
        assert!(source.contains("iq4_xs_element_code((uint)("), "{source}");
        assert!(source.contains("scale0_offset1") && source.contains("- 32)"), "{source}");
        assert!(!source.contains("(char)scales"), "{source}");
    }

    #[test]
    fn one_field_of_the_iq4_xs_layout_changes_the_rendered_kernel() {
        let mut moved = BlockLayout::IQ4_XS;
        let sub = moved.sub.as_mut().expect("iq4_xs has sub-blocks");
        sub.scale.low.base_bits += 64;
        let table = iq4_xs_table();
        assert_ne!(render(&BlockLayout::IQ4_XS, &table).expect("iq4_xs renders"), render(&moved, &table).expect("the moved layout renders"));
    }

    fn level_table_of(codec: Codec) -> LevelTable {
        let Codec::Levels(format) = codec else {
            panic!("{codec:?} is a level-set format");
        };
        format.levels
    }

    #[test]
    fn the_level_table_predicate_accepts_iq3_s_and_refuses_iq3_xxs() {
        let iq3_s = level_table_of(Codec::Iq3S);
        assert!(matches!(iq3_s, LevelTable::SignedGrid { sign_table: None, .. }), "iq3_s is a signed grid with a plane sign");
        assert!(level_table_rendered(&iq3_s));
        let iq3_xxs = level_table_of(Codec::Iq3Xxs);
        assert!(matches!(iq3_xxs, LevelTable::SignedGrid { sign_table: Some(_), .. }), "iq3_xxs signs are a table index");
        assert!(!level_table_rendered(&iq3_xxs));
        assert!(level_table_rendered(&iq4_xs_table()));
    }

    #[test]
    fn iq3_s_renders_its_level_matvec_from_its_layout() {
        let source = render(&BlockLayout::IQ3_S, &level_table_of(Codec::Iq3S)).expect("iq3_s renders");
        assert!(source.contains("iq3_s_element_grid["), "{source}");
        assert!(source.contains("2 * (int)"), "{source}");
        assert!(source.contains("sign_codes"), "{source}");
        assert!(!source.contains("(char)scales"), "{source}");
    }

    #[test]
    fn one_field_of_the_iq3_s_layout_changes_the_rendered_kernel() {
        let mut moved = BlockLayout::IQ3_S;
        moved.sub.as_mut().expect("iq3_s has sub-blocks").scale.low.base_bits += 64;
        let table = level_table_of(Codec::Iq3S);
        assert_ne!(render(&BlockLayout::IQ3_S, &table).expect("iq3_s renders"), render(&moved, &table).expect("the moved layout renders"));
    }

    // twin of FLINT_BLK2_GATE_EXPERT0_BLOCK0 in proxima-gguf/src/quant/iq3_s.rs (private to that crate's tests): flint-UD-Q4_K_XL.gguf blk.2.ffn_gate_exps.weight, expert 0, row 0, block 0
    const FLINT_BLK2_GATE_EXPERT0_BLOCK0: [u8; 110] = [
        0x92, 0x00, 0x0e, 0x80, 0x59, 0x91, 0x0e, 0xc7, 0xf8, 0x85, 0xff, 0x44, 0x18, 0xd0, 0x1e, 0xe0, 0x80, 0x00, 0x0b, 0x9d, 0xe1, 0xa9, 0x02, 0xeb, 0x1e,
        0x19, 0x1e, 0x37, 0x76, 0x95, 0xcb, 0xe8, 0x56, 0x32, 0x3d, 0x3f, 0x81, 0x5b, 0x2a, 0x2e, 0xc8, 0x28, 0xa9, 0x3c, 0x52, 0x5b, 0x95, 0x92, 0x05, 0xfa,
        0x8d, 0x5c, 0x80, 0xaf, 0xe1, 0xee, 0x29, 0xa9, 0x88, 0xc5, 0x7b, 0x59, 0x8a, 0x69, 0x4e, 0x34, 0x09, 0x84, 0xa8, 0xc0, 0x5a, 0x41, 0xc3, 0x54, 0xfc,
        0xd7, 0x94, 0x24, 0x82, 0x05, 0xf4, 0x53, 0x2f, 0x51, 0x58, 0xfa, 0x92, 0xf3, 0x5b, 0x60, 0x5b, 0x5a, 0xb0, 0xed, 0x1d, 0xb1, 0xa4, 0xa1, 0x84, 0xdf,
        0x7a, 0x5f, 0x4a, 0x0e, 0x6f, 0x5c, 0xef, 0xbc, 0xab, 0xea,
    ];

    fn reads_at(block: &[u8], reads: &PlaneReads, owned: usize, slot: usize) -> u32 {
        let (byte, shift) = reads.slots[slot];
        let at = reads.base_byte + owned as u32 * reads.owned_bytes + byte;
        (u32::from(block[at as usize]) >> shift) & ((1u32 << reads.bits) - 1)
    }

    fn sign_run_at(block: &[u8], reads: &PlaneReads, owned: usize, slot: usize, run: u32) -> u32 {
        let (byte, shift) = reads.slots[slot];
        let at = reads.base_byte + owned as u32 * reads.owned_bytes + byte;
        (u32::from(block[at as usize]) >> shift) & ((1u32 << run) - 1)
    }

    fn iq3_s_plan<'a>(codes: &'a LevelCodes<'a>, geometry: &'a SubBlockMatvecGeometry) -> Assembled<'a> {
        Assembled::of(&BlockLayout::IQ3_S, Some(codes), geometry).expect("iq3_s plans")
    }

    #[test]
    fn the_iq3_s_plan_addresses_the_worked_tile_of_the_flint_block() {
        let table = level_table_of(Codec::Iq3S);
        let codes = LevelCodes { table: &table, element_fn: "iq3_s_element" };
        let geometry = SubBlockMatvecGeometry::for_levels(&table, 4, 2, 32);
        let plan = iq3_s_plan(&codes, &geometry);
        assert_eq!((plan.run, plan.lane_runs, plan.covered, plan.span), (4, 1, 8, 32));
        let high = plan.high_reads.as_ref().expect("iq3_s has a high plane");
        let sign = plan.sign_reads.as_ref().expect("iq3_s has a sign plane");
        assert_eq!((plan.low_reads.base_byte, plan.low_reads.owned_bytes, plan.low_reads.per_run), (2, 8, false));
        assert_eq!((high.base_byte, high.owned_bytes, high.per_run), (66, 1, false));
        assert_eq!((sign.base_byte, sign.owned_bytes), (74, 4));
        for slot in 0..8 {
            assert_eq!(plan.low_reads.slots[slot], (slot as u32, 0));
            assert_eq!(high.slots[slot], (0, slot as u32));
            assert_eq!(sign.slots[slot], (slot as u32 / 2, 4 * (slot as u32 % 2)));
        }
        let bytes: Vec<u32> = (0..8).map(|slot| reads_at(&FLINT_BLK2_GATE_EXPERT0_BLOCK0, &plan.low_reads, 0, slot)).collect();
        assert_eq!(bytes, [0x0e, 0x80, 0x59, 0x91, 0x0e, 0xc7, 0xf8, 0x85]);
        let high_bits: Vec<u32> = (0..8).map(|slot| reads_at(&FLINT_BLK2_GATE_EXPERT0_BLOCK0, high, 0, slot)).collect();
        assert_eq!(high_bits, [1, 0, 0, 1, 0, 0, 0, 0]);
        let signs: Vec<u32> = (0..4).map(|slot| sign_run_at(&FLINT_BLK2_GATE_EXPERT0_BLOCK0, sign, 0, slot, 4)).collect();
        assert_eq!(signs, [0xc, 0xf, 0x7, 0xd]);
    }

    #[test]
    fn the_iq3_s_walk_evaluated_on_the_flint_block_reproduces_the_hand_derived_elements() {
        let table = level_table_of(Codec::Iq3S);
        let LevelTable::SignedGrid { grid, .. } = &table else {
            panic!("iq3_s is a signed grid");
        };
        let codes = LevelCodes { table: &table, element_fn: "iq3_s_element" };
        let geometry = SubBlockMatvecGeometry::for_levels(&table, 4, 2, 32);
        let plan = iq3_s_plan(&codes, &geometry);
        let SubScales::Planes(scales) = &plan.scales else {
            panic!("iq3_s sub-block scales are plane coded");
        };
        let delta = half::f16::from_bits(u16::from_le_bytes([FLINT_BLK2_GATE_EXPERT0_BLOCK0[0], FLINT_BLK2_GATE_EXPERT0_BLOCK0[1]])).to_f32();
        let mut walked = vec![0.0f32; 256];
        for owned in 0..8 {
            for slot in 0..plan.covered {
                let per_plane = &scales.distinct[scales.scale_of_slot[slot]];
                let scale_code: u32 = per_plane
                    .iter()
                    .zip(&scales.planes)
                    .map(|((offsets, shifts), (bits, position))| ((u32::from(FLINT_BLK2_GATE_EXPERT0_BLOCK0[offsets[owned] as usize]) >> shifts[owned]) & ((1u32 << bits) - 1)) << position)
                    .sum();
                let scale = i32::from(scales.multiplier) * scale_code as i32 - i32::from(scales.bias);
                let index = reads_at(&FLINT_BLK2_GATE_EXPERT0_BLOCK0, &plan.low_reads, owned, slot) | (reads_at(&FLINT_BLK2_GATE_EXPERT0_BLOCK0, plan.high_reads.as_ref().expect("high plane"), owned, slot) << 8);
                let signs = sign_run_at(&FLINT_BLK2_GATE_EXPERT0_BLOCK0, plan.sign_reads.as_ref().expect("sign plane"), owned, slot, 4);
                for lane in 0..4 {
                    let magnitude = ((grid[index as usize] >> (8 * lane)) & 0xFF) as f32;
                    let signed = if (signs >> lane) & 1 != 0 { -magnitude } else { magnitude };
                    walked[lane + 4 * (slot + 8 * owned)] = delta * scale as f32 * signed;
                }
            }
        }
        let hand_derived: [i32; 16] = [5, 1, -13, -5, -3, -1, -3, -3, -3, -9, -11, 1, -3, 11, -15, -9];
        for (element, multiple) in hand_derived.iter().enumerate() {
            let expected = (4526 * multiple) as f32 / 16_777_216.0;
            assert_eq!(walked[element].to_bits(), expected.to_bits(), "element {element}: walked {} want {expected}", walked[element]);
        }
        let mut decoded = [0.0f32; 256];
        proxima_gguf::quant::iq3_s::dequantize(&FLINT_BLK2_GATE_EXPERT0_BLOCK0, &mut decoded).expect("one whole block decodes");
        for (element, (walk, cpu)) in walked.iter().zip(decoded).enumerate() {
            assert!((walk - cpu).abs() <= cpu.abs() * 1e-6, "element {element}: walked {walk} cpu {cpu}");
        }
    }

    #[test]
    fn a_scale_multiplier_of_two_renders_the_doubled_scale_term() {
        let mut doubled = BlockLayout::IQ4_XS;
        doubled.sub.as_mut().expect("iq4_xs has sub-blocks").scale_multiplier = 2;
        let source = render(&doubled, &iq4_xs_table()).expect("a doubled scale renders");
        assert!(source.contains("float scale0 = (float)(2 * (int)("), "{source}");
        let plain = render(&BlockLayout::IQ4_XS, &iq4_xs_table()).expect("iq4_xs renders");
        assert!(!plain.contains("2 * (int)"), "{plain}");
    }

    #[test]
    fn a_grid_table_on_a_byte_stride_layout_is_refused() {
        let grid = LevelTable::SignedGrid { grid: &[], lanes: 4, sign: BlockLayout::IQ4_XS.element.low, sign_table: None, unit_shift: 0 };
        let refusal = render(&BlockLayout::IQ4_XS, &grid).expect_err("a grid on a one-code-per-byte layout is refused");
        assert_eq!(refusal, SubBlockMatvecRefusal("a grid code covers its lanes"));
    }

    #[test]
    fn a_grid_level_table_is_refused() {
        let grid = level_table_of(Codec::Iq3Xxs);
        assert!(render(&BlockLayout::IQ4_XS, &grid).is_err());
    }
}
