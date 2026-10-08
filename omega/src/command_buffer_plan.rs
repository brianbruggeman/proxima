//! Where a chunked decode step cuts its program into command buffers.
//!
//! Composes nothing at run time: it is a pure function from an op count to the
//! op indices at which the Metal executor (`metal::placements_execute_named`)
//! commits a command buffer and starts encoding the next. It lives outside the
//! `metal` module so the schedule is testable, and reusable by any backend
//! that encodes ahead of the device, without a GPU. The two policy numbers
//! come from `omega-runtime.toml`'s `[command_buffer]` section through
//! `crate::sized`; callers pass them in.

use alloc::vec::Vec;

const FIXED_POINT_ONE: u128 = 1_000_000_000;
const PERMILLE_ONE: u32 = 1000;
const GROWTH_PERMILLE_CEILING: u32 = 4000;

/// Op indices at which a program of `total_ops` ops is cut into `chunk_count`
/// contiguous command buffers: a head of `head_ops` ops, then the remaining ops
/// split into `chunk_count - 1` chunks whose sizes grow by `growth_permille`
/// per chunk (1000 is an even split, 1500 makes each chunk 1.5x the last).
///
/// A short head starts the device early; the growth keeps each chunk's device
/// time longer than the next chunk's encode time, so the host never stalls the
/// device between buffers. `head_ops == 0`, `head_ops >= total_ops` or
/// `chunk_count <= 2` means no head: all ops split evenly, growth ignored,
/// because without a head there is no short first buffer to cover. Returns the
/// empty vector for `chunk_count <= 1` or `total_ops == 0` without allocating.
/// Boundaries that collapse onto each other (more chunks than ops) are dropped
/// rather than producing an empty chunk.
///
/// Cost: one pass over `chunk_count` terms, once per plan; no per-dispatch
/// work. `growth_permille` is clamped to 1000..=4000.
#[must_use]
pub fn chunk_boundaries(
    total_ops: usize,
    chunk_count: usize,
    head_ops: usize,
    growth_permille: u32,
) -> Vec<usize> {
    if chunk_count <= 1 || total_ops == 0 {
        return Vec::new();
    }
    let headed = head_ops > 0 && head_ops < total_ops && chunk_count > 2;
    let (start, tail_chunks) = if headed {
        (head_ops, chunk_count - 1)
    } else {
        (0, chunk_count)
    };
    let mut boundaries = Vec::with_capacity(chunk_count - 1);
    if headed {
        boundaries.push(head_ops);
    }
    let tail_growth = if headed { growth_permille } else { PERMILLE_ONE };
    push_grown_split(&mut boundaries, start, total_ops, tail_chunks, tail_growth);
    boundaries
}

fn series_total(chunks: usize, growth: u128) -> u128 {
    (0..chunks)
        .scan(FIXED_POINT_ONE, |weight, _| {
            let current = *weight;
            *weight = weight.saturating_mul(growth) / u128::from(PERMILLE_ONE);
            Some(current)
        })
        .fold(0u128, u128::saturating_add)
}

fn push_grown_split(
    boundaries: &mut Vec<usize>,
    start: usize,
    total_ops: usize,
    chunks: usize,
    growth_permille: u32,
) {
    let growth = u128::from(growth_permille.clamp(PERMILLE_ONE, GROWTH_PERMILLE_CEILING));
    let span = (total_ops - start) as u128;
    let series = series_total(chunks, growth);
    let mut weight = FIXED_POINT_ONE;
    let mut cumulative = 0u128;
    let mut previous = start;
    for _ in 1..chunks {
        cumulative = cumulative.saturating_add(weight);
        weight = weight.saturating_mul(growth) / u128::from(PERMILLE_ONE);
        let boundary = start + (span.saturating_mul(cumulative) / series) as usize;
        if boundary > previous && boundary < total_ops {
            boundaries.push(boundary);
            previous = boundary;
        }
    }
}
