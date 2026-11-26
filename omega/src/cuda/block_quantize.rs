//! `BoundOpKind::BlockQuantize` as a normal kernel: one warp quantizes each 32-element group of
//! the dense f32 source into a `block_q8_1`-style run, written through the output's f32 words.
//! Where the run's scale header and codes sit comes from the block layout's run geometry, so a
//! warp writes only its own header and codes. The output holds `block_bytes / 4` words per block,
//! so the layout's block size must be word-sized.

use alloc::format;
use alloc::string::String;

use proxima_numeric::ScaleArithmetic;
use proxima_numeric::block_layout::BlockLayout;
pub use proxima_tensor::Missing;
use proxima_tensor::{
    ACTIVATION_GROUP_ELEMENTS, ACTIVATION_REACH, BoundOp, BoundOpKind, is_int8_run_layout,
    quantize_rule_is_served,
};

use super::target::Target;
use super::{CudaGridSpec, EmitError, MAX_GRID_BLOCKS, word_c};
use crate::grid::checked_product;

const QUANTIZE_THREADS: u64 = 256;
const GROUP_ELEMENTS: u64 = ACTIVATION_GROUP_ELEMENTS;

fn unsupported(resolved: &BoundOp, kind: &'static str) -> EmitError {
    EmitError::CudaUnsupportedOpKind {
        node: resolved.node,
        kind,
    }
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
    if quantize_rule_is_served(layout) {
        return Ok(());
    }
    Err(Missing {
        piece: "a quantize device function for this layout's scale rule: quantize_device divides the peak by 127 and stores the sum",
        file: file!(),
        line: line!(),
    })
}

fn servable_layout(resolved: &BoundOp) -> Result<(&BlockLayout, ScaleArithmetic), EmitError> {
    let BoundOpKind::BlockQuantize {
        layout,
        operands,
        arithmetic,
    } = &resolved.kind
    else {
        return Err(EmitError::RenderKindMismatch {
            node: resolved.node,
            expected: "block_quantize",
            found: resolved.kind.name(),
        });
    };
    let dense_source = operands
        .first()
        .is_some_and(|(_, layout, _)| layout.base == 0 && layout.strides.last() == Some(&1));
    match (
        producer_missing(layout).is_ok(),
        layout.block_bytes % 4 == 0,
        dense_source,
    ) {
        (true, true, true) => Ok((layout, *arithmetic)),
        (false, _, _) => Err(unsupported(resolved, "block_quantize_layout")),
        (_, false, _) => Err(unsupported(
            resolved,
            "block_quantize_block_bytes_not_word_sized",
        )),
        (_, _, false) => Err(unsupported(resolved, "block_quantize_source_not_dense")),
    }
}

pub(super) fn block_quantize_groups(resolved: &BoundOp) -> Result<u64, EmitError> {
    let (layout, _) = servable_layout(resolved)?;
    let blocks = resolved.extents.iter().product::<u64>() / u64::from(layout.block_bytes);
    Ok(blocks * u64::from(layout.runs()))
}

pub(super) fn block_quantize_grid(resolved: &BoundOp) -> Result<CudaGridSpec, EmitError> {
    let groups = block_quantize_groups(resolved)?;
    let grid = CudaGridSpec::launch(
        resolved.node,
        checked_product(resolved.node, [groups, GROUP_ELEMENTS])?,
        Some(QUANTIZE_THREADS),
        "block_quantize",
    )?;
    if grid.blocks() > MAX_GRID_BLOCKS {
        return Err(EmitError::GridExceedsThreadIndex {
            node: resolved.node,
            threads: grid.threads(),
            limit: MAX_GRID_BLOCKS * grid.block_width_or_default(),
        });
    }
    Ok(grid)
}

/// The lines that turn the group peak `amax` into the scale `d` and the lane code `q`, rendered
/// from the rule's arithmetic and divisor (`reach`).
fn scale_and_code(arithmetic: ScaleArithmetic, reach: i32) -> String {
    match arithmetic {
        ScaleArithmetic::InverseOfScale => format!(
            "float d = amax / {reach}.0f;\n\
             \x20   float d_inv = d == 0.0f ? 0.0f : 1.0f / d;\n\
             \x20   int q = (int)roundf(x * d_inv);\n"
        ),
        ScaleArithmetic::Quotient => format!(
            "float d = amax / {reach}.0f;\n\
             \x20   int q = amax == 0.0f ? 0 : (int)roundf(x / d);\n"
        ),
        ScaleArithmetic::ReciprocalMultiply => format!(
            "float d_inv = {reach}.0f / amax;\n\
             \x20   float d = 1.0f / d_inv;\n\
             \x20   int q = amax == 0.0f ? 0 : (int)roundf(x * d_inv);\n"
        ),
    }
}

/// The device function one warp runs on one 32-element group `group` of `layout`: absmax scale,
/// codes under `arithmetic`, and the scaled code sum beside the scale, at the group's run offsets.
pub(super) fn quantize_device(layout: &BlockLayout, arithmetic: ScaleArithmetic) -> String {
    let reach = layout
        .encode_rule()
        .map_or(ACTIVATION_REACH, |rule| rule.divisor);
    let scale_and_code = scale_and_code(arithmetic, reach);
    let runs = layout.runs();
    let (scale_base, scale_stride) = (
        layout.run_scale_byte(0),
        layout.run_scale_byte(runs.min(2) - 1) - layout.run_scale_byte(0),
    );
    let (code_base, code_stride) = (
        layout.run_code_byte(0),
        layout.run_code_byte(runs.min(2) - 1) - layout.run_code_byte(0),
    );
    format!(
        "__device__ __forceinline__ void omega_q8_1_quantize_block(const float *in, float *out, long long group, unsigned int lane, bool live) {{\n\
         \x20   float x = in[group * 32 + lane];\n\
         \x20   float amax = fabsf(x);\n\
         \x20   for (int offset = 16; offset > 0; offset >>= 1) {{ amax = fmaxf(amax, __shfl_xor_sync(PROXIMA_ALL_LANES, amax, offset)); }}\n\
         \x20   {scale_and_code}\
         \x20   int qsum = q;\n\
         \x20   for (int offset = 16; offset > 0; offset >>= 1) {{ qsum += __shfl_xor_sync(PROXIMA_ALL_LANES, qsum, offset); }}\n\
         \x20   if (!live) {{ return; }}\n\
         \x20   unsigned char *bytes = (unsigned char *)out + (group / {runs}) * {block_bytes};\n\
         \x20   const long long run = group % {runs};\n\
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

pub(super) fn render_block_quantize(
    target: Target,
    resolved: &BoundOp,
    entry: &str,
) -> Result<String, EmitError> {
    let (layout, arithmetic) = servable_layout(resolved)?;
    let device = quantize_device(layout, arithmetic);
    let lane_mask_define = target.lane_mask_define();
    let word_c = word_c(&target)?;
    Ok(format!(
        "#include <{header}>\n{lane_mask_define}{device}\n\
         struct Uniforms {{ {word_c} groups; }};\n\n\
         extern \"C\" __global__ void __launch_bounds__({QUANTIZE_THREADS}) {entry}(\n\
         \x20   const float* __restrict__ in0,\n\
         \x20   float* __restrict__ out,\n\
         \x20   const Uniforms* __restrict__ u_ptr\n\
         ) {{\n\
         \x20   const Uniforms u = *u_ptr;\n\
{entry_wave_guard}\
         \x20   const long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
         \x20   const long long group = gid / {GROUP_ELEMENTS};\n\
         \x20   const bool live = group < u.groups;\n\
         \x20   omega_q8_1_quantize_block(in0, out, live ? group : u.groups - 1, (unsigned int)(gid & {lane_mask}), live);\n\
         }}\n",
        lane_mask = GROUP_ELEMENTS - 1,
        entry_wave_guard = target.entry_wave_guard(),
        header = target.fp16_header,
    ))
}
