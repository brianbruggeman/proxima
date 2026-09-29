//! The flat 2D grid form: what a kernel does when its grid is wider than the
//! 32 bits `uint gid [[thread_position_in_grid]]` can index.
//!
//! Metal encodes a dispatch width in 32 bits, so a 1D grid of `T` threads
//! launches only `T mod 2^32` of them and the outputs past that prefix are
//! never written. [`flat_grid2d`] describes the launch that covers `T` instead
//! (`dispatchThreadgroups` over an x-by-y rectangle of whole threadgroups) and
//! [`widen_thread_index`] rewrites the kernel's own signature to match, so the
//! two are produced from the same [`Grid2DSpec`] decision in `grid2d_for` and
//! cannot disagree. The body is untouched: it keeps reading its `gid` (or `tg`), now
//! rebuilt in 64 bits from `threadgroup_position_in_grid`,
//! `threadgroups_per_grid`, `threads_per_threadgroup` and
//! `thread_index_in_threadgroup`.

use super::*;

/// Whether a grid of `threads` is wider than the 32 bits of a 1D
/// `uint gid [[thread_position_in_grid]]` kernel and so must take the flat form.
pub(super) fn exceeds_linear_grid(threads: u64) -> bool {
    threads > crate::sized::GRID_LINEAR_THREAD_LIMIT
}

/// The launch [`Grid2DForm::FlatThreadgroupIndex`] describes for `threads`
/// threads of a kernel whose threadgroup is `threadgroup_width` wide (`None`:
/// the driver's choice). A pinned width that divides the grid is launched as
/// pinned -- every kernel whose lane or row math depends on it (cooperative
/// reduce, tiled GEMM, cached attention, softmax weights, gated delta net) has
/// a grid that is a whole number of its threadgroups by construction. One that
/// does not (a serial reduce the extras pinned a width for that it never
/// reads, a row-blocked packed matvec at nsg=2 whose addressing is
/// `gid / SIMD_WIDTH`) launches one simdgroup wide, which every such grid is a
/// whole number of, or ends in a short last group that guards on its uniform
/// total. The threadgroups are split into an exact x-by-y rectangle, x the
/// largest divisor of the count not above `GRID_MAX_THREADGROUPS_X`, so no
/// whole threadgroup past the grid is ever launched.
pub(super) fn flat_grid2d(
    node: NodeId,
    threads: u64,
    threadgroup_width: Option<u64>,
) -> Result<Grid2DSpec, EmitError> {
    let width = match threadgroup_width {
        Some(pinned) if threads.is_multiple_of(pinned) => pinned,
        _ => SIMD_WIDTH,
    };
    let groups = threads.div_ceil(width);
    let threadgroups_x = (1..=groups.min(crate::sized::GRID_MAX_THREADGROUPS_X))
        .rev()
        .find(|candidate| groups.is_multiple_of(*candidate))
        .unwrap_or(1);
    let threadgroups_y = groups / threadgroups_x;
    if threadgroups_y > u64::from(u32::MAX) {
        return Err(EmitError::GridExceedsThreadIndex {
            node,
            threads,
            limit: u64::from(u32::MAX) * width,
        });
    }
    #[cfg(feature = "instrument")]
    proxima_telemetry::debug!(
        node = node.0,
        threads,
        threadgroups_x,
        threadgroups_y,
        width,
        "grid past the 32-bit thread index takes the flat form"
    );
    Ok(Grid2DSpec {
        form: Grid2DForm::FlatThreadgroupIndex,
        threadgroups_x,
        threadgroups_y,
        threads_per_threadgroup_x: width,
        threads_per_threadgroup_y: 1,
    })
}


/// A grid attribute a 1D kernel's signature may carry, the definition that
/// gives the body the same name back under the flat form, and any body text
/// that reads the attribute's components directly. A kernel reads a subset:
/// most take `gid`; `CachedAttention`'s split kernel also takes `tgid`; the
/// row-blocked split-K body `tptg`; `CachedSoftmaxWeights` (one threadgroup
/// per row, no `gid` at all) only `tg`; the dense-batched GEMM's 1D form a
/// `uint3` whose `x` is the flat index and `z` the batch.
struct GridAttribute {
    parameter: &'static str,
    definition: &'static str,
    body_reads: &'static [(&'static str, &'static str)],
}

const GRID_ATTRIBUTES: [GridAttribute; 6] = [
    GridAttribute {
        parameter: "uint gid [[thread_position_in_grid]]",
        definition: "    ulong gid = wide_group_index * ulong(wide_width.x) + ulong(wide_lane);\n",
        body_reads: &[],
    },
    GridAttribute {
        parameter: "uint3 dense_batch_gid [[thread_position_in_grid]]",
        definition: "",
        body_reads: &[
            ("(long)dense_batch_gid.x", "(long)(wide_group_index * ulong(wide_width.x) + ulong(wide_lane))"),
            ("(long)dense_batch_gid.z", "(long)wide_group.z"),
        ],
    },
    GridAttribute {
        parameter: "uint tgid [[threadgroup_position_in_grid]]",
        definition: "    ulong tgid = wide_group_index;\n",
        body_reads: &[],
    },
    GridAttribute {
        parameter: "uint tg [[threadgroup_position_in_grid]]",
        definition: "    ulong tg = wide_group_index;\n",
        body_reads: &[],
    },
    GridAttribute {
        parameter: "uint tptg [[threads_per_threadgroup]]",
        definition: "    uint tptg = wide_width.x;\n",
        body_reads: &[],
    },
    GridAttribute {
        parameter: "uint local [[thread_position_in_threadgroup]]",
        definition: "    uint local = wide_lane;\n",
        body_reads: &[],
    },
];

const WIDE_PARAMETERS: &str = "uint3 wide_group [[threadgroup_position_in_grid]],\n    \
    uint3 wide_groups [[threadgroups_per_grid]],\n    \
    uint3 wide_width [[threads_per_threadgroup]],\n    \
    uint wide_lane [[thread_index_in_threadgroup]]";

const WIDE_PROLOGUE: &str = "    ulong wide_group_index = ulong(wide_group.y) * ulong(wide_groups.x) + ulong(wide_group.x);\n";

/// Swaps a rendered 1D kernel's grid attributes ([`GRID_ATTRIBUTES`]) for the
/// four [`flat_grid2d`]'s launch provides and gives the body each name back
/// before it runs: `gid`, `tgid` and `tg` as `ulong`s rebuilt from the flat
/// threadgroup index (a 32-bit `threadgroup_position_in_grid` would hold only
/// the x axis), `tptg` as the `uint` width, and the dense-batched `uint3` by
/// rewriting the two reads of it. Metal wants every grid attribute in one
/// vector width, so the scalar ones cannot stay beside the vector ones.
pub(super) fn widen_thread_index(node: NodeId, source: &str) -> Result<String, EmitError> {
    let mismatch = |expected: &'static str| EmitError::RenderKindMismatch {
        node,
        expected,
        found: "missing",
    };
    let first_attribute = GRID_ATTRIBUTES
        .iter()
        .filter_map(|attribute| source.find(attribute.parameter))
        .min()
        .ok_or_else(|| mismatch("grid attribute parameter"))?;
    let (terminator_offset, terminator_len) = [")\n{\n", ") {\n"]
        .iter()
        .filter_map(|terminator| {
            source[first_attribute..]
                .find(terminator)
                .map(|offset| (offset, terminator.len()))
        })
        .min_by_key(|(offset, _)| *offset)
        .ok_or_else(|| mismatch("kernel signature terminator"))?;
    let parameters_end = first_attribute + terminator_offset;
    let body_start = parameters_end + terminator_len;

    let mut prologue = String::from(WIDE_PROLOGUE);
    let mut widened_parameters = String::from(WIDE_PARAMETERS);
    let mut body = String::from(&source[body_start..]);
    for parameter in source[first_attribute..parameters_end].split(',') {
        let parameter = parameter.trim();
        match GRID_ATTRIBUTES
            .iter()
            .find(|attribute| attribute.parameter == parameter)
        {
            Some(attribute) => {
                prologue.push_str(attribute.definition);
                for (read, replacement) in attribute.body_reads {
                    body = body.replace(read, replacement);
                }
            }
            None => {
                widened_parameters.push_str(",\n    ");
                widened_parameters.push_str(parameter);
            }
        }
    }
    Ok(format!(
        "{}{widened_parameters}{}{prologue}{body}",
        &source[..first_attribute],
        &source[parameters_end..body_start],
    ))
}

/// [`widen_thread_index`] when `grid2d` says the launch is flat, the source
/// untouched for the 1D form and the tile form (whose renderer already reads
/// its own threadgroup coordinates). The one place a kernel's text is made to
/// agree with the [`Grid2DSpec`] `grid2d_for` decided.
pub(super) fn widen_for_grid(
    node: NodeId,
    source: String,
    grid2d: Option<Grid2DSpec>,
) -> Result<String, EmitError> {
    match grid2d {
        Some(Grid2DSpec {
            form: Grid2DForm::FlatThreadgroupIndex,
            ..
        }) => widen_thread_index(node, &source),
        _ => Ok(source),
    }
}

