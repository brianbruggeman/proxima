//! `BoundOpKind::BlockQuantize` as a normal kernel: one warp quantizes each 32-element group of
//! the dense f32 source into `block_q8_1` bytes, written through the output's f32 words. The
//! output holds `block_bytes / 4` words per group, so the layout's block size must be word-sized.

use alloc::format;
use alloc::string::String;

use proxima_primitives::block_layout::BlockLayout;
use proxima_tensor::{BoundOp, BoundOpKind};

use super::{CudaGridSpec, EmitError, MAX_GRID_BLOCKS, Q8_1_QUANTIZE_DEVICE};
use crate::grid::checked_product;

const QUANTIZE_THREADS: u64 = 256;
const GROUP_ELEMENTS: u64 = 32;

fn unsupported(resolved: &BoundOp, kind: &'static str) -> EmitError {
    EmitError::CudaUnsupportedOpKind {
        node: resolved.node,
        kind,
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
    match (**layout == BlockLayout::Q8_1, layout.block_bytes % 4 == 0, dense_source) {
        (true, true, true) => Ok(layout),
        (false, _, _) => Err(unsupported(resolved, "block_quantize_layout")),
        (_, false, _) => Err(unsupported(resolved, "block_quantize_block_bytes_not_word_sized")),
        (_, _, false) => Err(unsupported(resolved, "block_quantize_source_not_dense")),
    }
}

pub(super) fn block_quantize_groups(resolved: &BoundOp) -> Result<u64, EmitError> {
    let layout = servable_layout(resolved)?;
    Ok(resolved.extents.iter().product::<u64>() / u64::from(layout.block_bytes))
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

pub(super) fn render_block_quantize(resolved: &BoundOp, entry: &str) -> Result<String, EmitError> {
    servable_layout(resolved)?;
    Ok(format!(
        "#include <cuda_fp16.h>\n{Q8_1_QUANTIZE_DEVICE}\n\
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
