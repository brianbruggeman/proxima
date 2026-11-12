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
use alloc::vec::Vec;

use proxima_numeric::block_layout::{BlockLayout, Plane};

use super::sub_block_matvec::{SubBlockMatvecGeometry, SubBlockMatvecRefusal, lane_split, require, typed_field_read};
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

    fn plane_reads(&self, plane: &Plane) -> Result<PlaneReads, SubBlockMatvecRefusal> {
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
                        let expected = origin + first + ir * ir_bits + owned * owned_bits + run_index * BYTE_BITS as usize;
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
        })
    }
}

fn piece(pointer: &str, reads: &PlaneReads, slot: usize) -> String {
    let (byte, shift) = reads.slots[slot];
    let mask = (1u32 << reads.bits) - 1;
    format!("(((uint){pointer}[i + {byte}u] >> {shift}u) & {mask:#x}u)")
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
    let sub = layout.sub.as_ref().ok_or(SubBlockMatvecRefusal("no sub-blocks"))?;
    require(layout.scale_runs.is_none() && layout.multiplier == 1, "plain code multiplier")?;
    require(layout.min.is_none() && sub.min.is_none(), "no min")?;
    require(layout.element.packed_radix == 0 && sub.scale_multiplier == 1, "plane coded element")?;
    let scale_plane = sub.scale.low;
    let scale_digit = scale_plane.digits[0];
    require(sub.scale.high.is_none() && sub.scale.bias == 0, "whole sub-block scale code")?;
    require(scale_plane.bits == 8 && scale_digit.radix == sub.count && u32::from(scale_digit.stride_bits) == BYTE_BITS, "one scale byte per sub-block")?;
    require(scale_plane.base_bits.is_multiple_of(BYTE_BITS), "scale bytes are aligned")?;
    let low = layout.element.low;
    let lane_digit = low.digits[0];
    require(u32::from(lane_digit.stride_bits) == BYTE_BITS, "lane digit steps one byte")?;
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
    require(run <= span && span.is_multiple_of(run) && lane_elements.is_multiple_of(span), "sub-blocks hold whole runs and lane digits")?;
    let tile = Tile { lane_elements, run, lane_runs, owned, covered };
    let low_reads = tile.plane_reads(&low)?;
    let high_reads = layout.element.high.as_ref().map(|plane| tile.plane_reads(plane)).transpose()?;
    let scale_read = typed_field_read(&layout.scale)?;
    let rows = geometry.rows_per_simdgroup;
    let block_bytes = layout.block_bytes;
    let ix_stride = geometry.lanes / geometry.threads_per_block;
    let owned_elements = lane_elements * covered;
    let run_offset = format!("{run}u * ir + {owned_elements}u * iq");
    let mut source = String::new();
    source.push_str(&lane_split(geometry));
    source.push_str(&format!("    uint ir = it % {lane_runs}u;\n    uint iq = it / {lane_runs}u;\n"));
    source.push_str(&format!("    int super_blocks = (int){reduction_bound} / {elements};\n"));
    source.push_str(&format!("    float yl[{}];\n", covered * run));
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
        low_reads.base_byte, low_reads.ir_bytes, low_reads.owned_bytes
    ));
    if let Some(reads) = &high_reads {
        source.push_str(&format!(
            "    uint high_offset = {}u + ir * {}u + iq * {}u;\n",
            reads.base_byte, reads.ir_bytes, reads.owned_bytes
        ));
    }
    let sub_per_owned = owned_elements / span;
    source.push_str(&format!(
        "    uint scale_offset = {}u + (ir * {run}u) / {span}u + iq * {sub_per_owned}u;\n",
        scale_plane.base_bits / BYTE_BITS
    ));
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
    source.push_str("            device const uchar *blk = blk_ptr[q];\n");
    source.push_str("            device const uchar *low_codes = blk + low_offset;\n");
    if high_reads.is_some() {
        source.push_str("            device const uchar *high_codes = blk + high_offset;\n");
    }
    source.push_str("            device const uchar *scales = blk + scale_offset;\n");
    source.push_str(&format!("            float dall = {scale_read};\n"));
    for slot in 0..covered {
        source.push_str(&format!("            float sums{slot} = 0.0f;\n"));
    }
    source.push_str(&format!("            {FULL_UNROLL}\n            for (uint i = 0u; i < {run}u; ++i) {{\n"));
    for slot in 0..covered {
        let mut code = piece("low_codes", &low_reads, slot);
        if let Some(reads) = &high_reads {
            code = format!("{code} | ({} << {}u)", piece("high_codes", reads, slot), low_reads.bits);
        }
        let biased = if layout.element.bias == 0 { format!("(float)(int)({code})") } else { format!("(float)((int)({code}) - {})", layout.element.bias) };
        source.push_str(&format!("                sums{slot} += yl[i + {}u] * {biased};\n", slot * run));
    }
    source.push_str("            }\n");
    let terms: Vec<String> = (0..covered).map(|slot| format!("sums{slot} * (float)(char)scales[{}u]", slot * lane_elements / span)).collect();
    source.push_str(&format!("            sumf[q] = sumf[q] + dall * ({});\n", terms.join(" + ")));
    source.push_str("        }\n");
    source.push_str(&format!("        {FULL_UNROLL}\n        for (int q = 0; q < {rows}; ++q) {{ blk_ptr[q] += blk_step; }}\n"));
    source.push_str("        y4 += y4_step;\n    }\n");
    Ok(source)
}
