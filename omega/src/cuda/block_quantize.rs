//! `BoundOpKind::BlockQuantize` as a normal kernel: one warp quantizes each 32-element group of
//! the dense f32 source into a `block_q8_1`-style run, written through the output's f32 words.
//! Where the run's scale header and codes sit comes from the block layout's run geometry, so a
//! warp writes only its own header and codes. The output holds `block_bytes / 4` words per block,
//! so the layout's block size must be word-sized.

use alloc::format;
use alloc::string::String;

use proxima_numeric::block_layout::BlockLayout;
use proxima_numeric::float_layout::FloatLayout;
use proxima_tensor::{BoundOp, BoundOpKind};

use super::{CudaGridSpec, EmitError, MAX_GRID_BLOCKS};
use crate::grid::checked_product;

const QUANTIZE_THREADS: u64 = 256;
const GROUP_ELEMENTS: u64 = 32;

fn unsupported(resolved: &BoundOp, kind: &'static str) -> EmitError {
    EmitError::CudaUnsupportedOpKind {
        node: resolved.node,
        kind,
    }
}

fn codes_are_byte_runs(layout: &BlockLayout) -> bool {
    let element = &layout.element;
    let run = usize::from(layout.run_elements());
    element.code == FloatLayout::INT8
        && element.high.is_none()
        && element.low.bits == 8
        && (0..usize::from(layout.elements))
            .all(|index| element.low.address(index) == layout.run_code_byte((index / run) as u16) * 8 + index % run * 8)
}

/// A piece the CUDA render lacks for a block activation layout, with the line that found it
/// missing, so a refusal points at the code that would have to change and never at a format name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Missing {
    pub piece: &'static str,
    pub file: &'static str,
    pub line: u32,
}

/// Int8 codes in 32-element byte runs under an f16 scale, each run's scale followed by the two-byte
/// stored sum the int8 readers fold in: what `quantize_device` writes and `activation_code_words`
/// reads.
fn is_int8_run_layout(layout: &BlockLayout) -> bool {
    layout.run_elements() == GROUP_ELEMENTS as u16
        && layout.scale.layout == FloatLayout::F16
        && layout.min.is_none()
        && layout.sub.is_none()
        && layout.multiplier == 1
        && layout.element.bias == 0
        && codes_are_byte_runs(layout)
        && (0..layout.runs()).all(|run| layout.run_code_byte(run) >= layout.run_scale_byte(run) + 4)
        && (0..layout.runs()).all(|run| layout.run_scale_byte(run) + 4 <= usize::from(layout.block_bytes))
}

/// Why no int8 reader (gemm, staged matvec, mma gemm) can read an activation held in `layout`.
pub(crate) fn consumer_missing(layout: &BlockLayout) -> Result<(), Missing> {
    if is_int8_run_layout(layout) {
        return Ok(());
    }
    Err(Missing {
        piece: "an activation reader for this layout: the int8 gemm and staged matvec unpack int8 byte-run codes under an f16 scale with a stored sum (int8_gemm.rs activation_code_words)",
        file: file!(),
        line: line!(),
    })
}

/// Why `render_block_quantize` cannot produce an activation in `layout`.
pub(crate) fn producer_missing(layout: &BlockLayout) -> Result<(), Missing> {
    if !is_int8_run_layout(layout) {
        return Err(Missing {
            piece: "a CUDA quantize device function rendered from this layout's EncodeRule: quantize_device writes the peak/127 rule for int8 byte runs as literals",
            file: file!(),
            line: line!(),
        });
    }
    // a layout with independent scale runs (the grouped pair) has no single rule; its geometry is
    // the run layout checked above
    match layout.encode_rule() {
        Some(rule) if rule.divisor != 127 || !rule.stored_sum => Err(Missing {
            piece: "a quantize device function for this layout's scale rule: quantize_device divides the peak by 127 and stores the sum",
            file: file!(),
            line: line!(),
        }),
        _ => Ok(()),
    }
}

fn servable_layout(resolved: &BoundOp) -> Result<&BlockLayout, EmitError> {
    let BoundOpKind::BlockQuantize { layout, operands } = &resolved.kind else {
        return Err(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "block_quantize",
            found: resolved.kind.name(),
        });
    };
    let dense_source = operands
        .first()
        .is_some_and(|(_, layout, _)| layout.base == 0 && layout.strides.last() == Some(&1));
    match (producer_missing(layout).is_ok(), layout.block_bytes % 4 == 0, dense_source) {
        (true, true, true) => Ok(layout),
        (false, _, _) => Err(unsupported(resolved, "block_quantize_layout")),
        (_, false, _) => Err(unsupported(resolved, "block_quantize_block_bytes_not_word_sized")),
        (_, _, false) => Err(unsupported(resolved, "block_quantize_source_not_dense")),
    }
}

pub(super) fn block_quantize_groups(resolved: &BoundOp) -> Result<u64, EmitError> {
    let layout = servable_layout(resolved)?;
    let blocks = resolved.extents.iter().product::<u64>() / u64::from(layout.block_bytes);
    Ok(blocks * u64::from(layout.runs()))
}

pub(super) fn block_quantize_grid(resolved: &BoundOp) -> Result<CudaGridSpec, EmitError> {
    let groups = block_quantize_groups(resolved)?;
    let grid = CudaGridSpec {
        threads: checked_product(resolved.node, [groups, GROUP_ELEMENTS])?,
        block_width: Some(QUANTIZE_THREADS),
    };
    if grid.blocks() > MAX_GRID_BLOCKS {
        return Err(EmitError::GridExceedsThreadIndex {
            node: resolved.node,
            threads: grid.threads,
            limit: MAX_GRID_BLOCKS * grid.block_width_or_default(),
        });
    }
    Ok(grid)
}

/// The device function one warp runs on one 32-element group `group` of `layout`: absmax scale,
/// codes, and the scaled code sum beside the scale, at the group's run offsets.
pub(super) fn quantize_device(layout: &BlockLayout) -> String {
    let runs = layout.runs();
    let (scale_base, scale_stride) = (layout.run_scale_byte(0), layout.run_scale_byte(runs.min(2) - 1) - layout.run_scale_byte(0));
    let (code_base, code_stride) = (layout.run_code_byte(0), layout.run_code_byte(runs.min(2) - 1) - layout.run_code_byte(0));
    format!(
        "__device__ __forceinline__ void omega_q8_1_quantize_block(const float *in, float *out, long group, unsigned int lane) {{\n\
         \x20   float x = in[group * 32 + lane];\n\
         \x20   float amax = fabsf(x);\n\
         \x20   for (int offset = 16; offset > 0; offset >>= 1) {{ amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset)); }}\n\
         \x20   float d = amax / 127.0f;\n\
         \x20   int q = amax == 0.0f ? 0 : (int)roundf(x / d);\n\
         \x20   int qsum = q;\n\
         \x20   for (int offset = 16; offset > 0; offset >>= 1) {{ qsum += __shfl_xor_sync(0xffffffffu, qsum, offset); }}\n\
         \x20   unsigned char *bytes = (unsigned char *)out + (group / {runs}) * {block_bytes};\n\
         \x20   const long run = group % {runs};\n\
         \x20   unsigned char *header = bytes + {scale_base} + run * {scale_stride};\n\
         \x20   if (lane == 0) {{\n\
         \x20       unsigned short d_bits = __half_as_ushort(__float2half(d));\n\
         \x20       unsigned short s_bits = __half_as_ushort(__float2half(d * (float)qsum));\n\
         \x20       header[0] = (unsigned char)(d_bits & 0xffu);\n\
         \x20       header[1] = (unsigned char)(d_bits >> 8);\n\
         \x20       header[2] = (unsigned char)(s_bits & 0xffu);\n\
         \x20       header[3] = (unsigned char)(s_bits >> 8);\n\
         \x20   }}\n\
         \x20   bytes[{code_base} + run * {code_stride} + lane] = (unsigned char)(signed char)q;\n\
         }}\n",
        block_bytes = layout.block_bytes,
    )
}

pub(super) fn render_block_quantize(resolved: &BoundOp, entry: &str) -> Result<String, EmitError> {
    let layout = servable_layout(resolved)?;
    let device = quantize_device(layout);
    Ok(format!(
        "#include <cuda_fp16.h>\n{device}\n\
         struct Uniforms {{ long groups; }};\n\n\
         extern \"C\" __global__ void __launch_bounds__({QUANTIZE_THREADS}) {entry}(\n\
         \x20   const float* __restrict__ in0,\n\
         \x20   float* __restrict__ out,\n\
         \x20   const Uniforms* __restrict__ u_ptr\n\
         ) {{\n\
         \x20   const Uniforms u = *u_ptr;\n\
         \x20   const long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;\n\
         \x20   const long group = gid / {GROUP_ELEMENTS};\n\
         \x20   if (group >= u.groups) {{ return; }}\n\
         \x20   omega_q8_1_quantize_block(in0, out, group, (unsigned int)(gid & {lane_mask}));\n\
         }}\n",
        lane_mask = GROUP_ELEMENTS - 1,
    ))
}
