//! A scalar transcription of the row-tiled partial kernel
//! (`omega/src/msl/cached_attention_row_tiled.rs`) and of the interleaved
//! merge, statement for statement: the threadgroup grid decoded from `tgid`,
//! the band and split slices, the per-block Q.K^T over 8-key fragments with
//! the per-row window mask, the online softmax with its rescale, P.V over
//! fragments into the f32 output tile, the scalar causal pass over the new
//! range in the last split, and the normalized or interleaved store. Simdgroup
//! parallelism and the barriers between phases are sequential loops here, and
//! each 8x8 `simdgroup_matrix` product is a plain 8-term sum, so the model
//! checks the kernel's algorithm -- indexing, masks, band partition, softmax
//! state, store layout -- on the CPU, against the CPU oracle, without a device.
//! What it cannot check is the Metal text itself.

pub struct Inputs<'a> {
    pub query_even: &'a [f32],
    pub query_odd: &'a [f32],
    pub cached_key_even: &'a [f32],
    pub cached_key_odd: &'a [f32],
    pub new_key_even: &'a [f32],
    pub new_key_odd: &'a [f32],
    pub cached_value: &'a [f32],
    pub new_value: &'a [f32],
}

pub struct Shape {
    pub kv_heads: i64,
    pub query_groups: i64,
    pub head_dim: i64,
    pub rows: i64,
    pub scale: f32,
    pub cached_lower: i64,
    pub new_upper: i64,
    pub live: i64,
}

pub struct Tiling {
    pub tile_rows: i64,
    pub block: i64,
    pub splits: i64,
}

struct Partial {
    output_tile: Vec<f32>,
    row_maximum: Vec<f32>,
    row_sum: Vec<f32>,
}

fn threadgroup(
    inputs: &Inputs,
    shape: &Shape,
    tiling: &Tiling,
    split: i64,
    tile: i64,
    kv_head: i64,
) -> (i64, i64, Partial) {
    let Shape {
        kv_heads,
        query_groups,
        head_dim,
        rows: total_rows,
        scale,
        cached_lower,
        new_upper,
        live,
    } = *shape;
    let Tiling {
        tile_rows,
        block,
        splits,
    } = *tiling;
    let half_dim = head_dim / 2;
    let row0 = tile * tile_rows;
    let rows_here = tile_rows.min(total_rows - row0);
    let vectors = rows_here * query_groups;
    let mut output_tile = vec![0.0f32; (vectors * head_dim) as usize];
    let mut score_tile = vec![0.0f32; (vectors * block) as usize];
    let mut row_maximum = vec![f32::NEG_INFINITY; vectors as usize];
    let mut row_sum = vec![0.0f32; vectors as usize];

    let first_key = (live + cached_lower + row0).max(0) & !7;
    let band = (live - first_key).max(0);
    let slice = (((band + splits - 1) / splits) + block - 1) / block * block;
    let slice_start = first_key + split * slice;
    let slice_end = (slice_start + slice).min(live);
    let cached_blocks = if slice_start < slice_end {
        (slice_end - slice_start + block - 1) / block
    } else {
        0
    };
    let new_blocks = if split == splits - 1 {
        (total_rows + block - 1) / block
    } else {
        0
    };

    for step in 0..cached_blocks + new_blocks {
        let new_block = step >= cached_blocks;
        let key0 = if new_block {
            (step - cached_blocks) * block
        } else {
            slice_start + step * block
        };
        let columns = if new_block {
            block.min(total_rows - key0)
        } else {
            block.min(slice_end - key0)
        };
        if !new_block {
            let fragments = (columns + 7) / 8;
            for key_tile in 0..fragments {
                for vector in 0..vectors {
                    let query_row = row0 + vector / query_groups;
                    let query_index = query_row * (kv_heads * query_groups)
                        + kv_head * query_groups
                        + vector % query_groups;
                    for lane_key in 0..8 {
                        let key = key0 + key_tile * 8 + lane_key;
                        let key_offset = key * (kv_heads * half_dim) + kv_head * half_dim;
                        let mut accumulated = 0.0f32;
                        for depth in 0..half_dim {
                            accumulated += inputs.query_even
                                [(query_index * half_dim + depth) as usize]
                                * inputs.cached_key_even[(key_offset + depth) as usize];
                            accumulated += inputs.query_odd
                                [(query_index * half_dim + depth) as usize]
                                * inputs.cached_key_odd[(key_offset + depth) as usize];
                        }
                        score_tile[(vector * block + key_tile * 8 + lane_key) as usize] =
                            accumulated;
                    }
                }
            }
        } else {
            for vector in 0..vectors {
                let query_row = row0 + vector / query_groups;
                for column in 0..columns {
                    let valid = key0 + column - query_row <= new_upper;
                    if valid {
                        let query_index = query_row * (kv_heads * query_groups)
                            + kv_head * query_groups
                            + vector % query_groups;
                        let key_offset =
                            (key0 + column) * (kv_heads * half_dim) + kv_head * half_dim;
                        let mut partial_score = 0.0f32;
                        for depth in 0..half_dim {
                            partial_score += inputs.new_key_even[(key_offset + depth) as usize]
                                * inputs.query_even[(query_index * half_dim + depth) as usize];
                            partial_score += inputs.new_key_odd[(key_offset + depth) as usize]
                                * inputs.query_odd[(query_index * half_dim + depth) as usize];
                        }
                        score_tile[(vector * block + column) as usize] = partial_score * scale;
                    } else {
                        score_tile[(vector * block + column) as usize] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        for vector in 0..vectors {
            let query_row = row0 + vector / query_groups;
            let mut block_maximum = f32::NEG_INFINITY;
            for column in 0..block {
                let mut raw_score = f32::NEG_INFINITY;
                if new_block {
                    if column < columns {
                        raw_score = score_tile[(vector * block + column) as usize];
                    }
                } else {
                    let key = key0 + column;
                    if key < slice_end && key - live - query_row >= cached_lower {
                        raw_score = score_tile[(vector * block + column) as usize] * scale;
                    }
                }
                score_tile[(vector * block + column) as usize] = raw_score;
                block_maximum = block_maximum.max(raw_score);
            }
            let previous_maximum = row_maximum[vector as usize];
            let next_maximum = previous_maximum.max(block_maximum);
            let rescale = if previous_maximum == f32::NEG_INFINITY {
                0.0
            } else {
                (previous_maximum - next_maximum).exp()
            };
            let mut block_sum = 0.0f32;
            for column in 0..block {
                let raw_score = score_tile[(vector * block + column) as usize];
                let weight = if raw_score == f32::NEG_INFINITY {
                    0.0
                } else {
                    (raw_score - next_maximum).exp()
                };
                score_tile[(vector * block + column) as usize] = weight;
                block_sum += weight;
            }
            row_maximum[vector as usize] = next_maximum;
            row_sum[vector as usize] = row_sum[vector as usize] * rescale + block_sum;
            for dimension in 0..head_dim {
                output_tile[(vector * head_dim + dimension) as usize] *= rescale;
            }
        }
        if !new_block {
            let fragments = (columns + 7) / 8;
            for vector in 0..vectors {
                for dimension in 0..head_dim {
                    for key_tile in 0..fragments {
                        for lane_key in 0..8 {
                            let key = key0 + key_tile * 8 + lane_key;
                            let value = inputs.cached_value[(key * (kv_heads * head_dim)
                                + kv_head * head_dim
                                + dimension)
                                as usize];
                            output_tile[(vector * head_dim + dimension) as usize] += score_tile
                                [(vector * block + key_tile * 8 + lane_key) as usize]
                                * value;
                        }
                    }
                }
            }
        } else {
            for dimension in 0..head_dim {
                for vector in 0..vectors {
                    let mut accumulated = 0.0f32;
                    for column in 0..columns {
                        let value = inputs.new_value[((key0 + column) * (kv_heads * head_dim)
                            + kv_head * head_dim
                            + dimension)
                            as usize];
                        accumulated += score_tile[(vector * block + column) as usize] * value;
                    }
                    output_tile[(vector * head_dim + dimension) as usize] += accumulated;
                }
            }
        }
    }
    (
        row0,
        vectors,
        Partial {
            output_tile,
            row_maximum,
            row_sum,
        },
    )
}

/// The whole op: every threadgroup, then the merge when `splits > 1`. Returns
/// the output rows `[rows, kv_heads, groups, head_dim]`, flat. The cached
/// buffers must hold every row the kernel's 8-key fragments reach, which is
/// up to the next whole fragment past the live count.
pub fn model(inputs: &Inputs, shape: &Shape, tiling: &Tiling) -> Vec<f32> {
    let Shape {
        kv_heads,
        query_groups,
        head_dim,
        rows: total_rows,
        ..
    } = *shape;
    let Tiling {
        tile_rows, splits, ..
    } = *tiling;
    let total_elements = total_rows * kv_heads * query_groups;
    let tiles = (total_rows + tile_rows - 1) / tile_rows;
    let mut output = vec![0.0f32; (total_elements * head_dim) as usize];
    let scratch_values = (total_elements * head_dim * splits) as usize;
    let mut scratch = vec![0.0f32; scratch_values + (total_elements * splits * 2) as usize];

    for tgid in 0..kv_heads * tiles * splits {
        let split = tgid % splits;
        let tile = (tgid / splits) % tiles;
        let kv_head = tgid / (splits * tiles);
        let (row0, vectors, partial) = threadgroup(inputs, shape, tiling, split, tile, kv_head);
        for vector in 0..vectors {
            let query_index = (row0 + vector / query_groups) * (kv_heads * query_groups)
                + kv_head * query_groups
                + vector % query_groups;
            for dimension in 0..head_dim {
                let value = partial.output_tile[(vector * head_dim + dimension) as usize];
                if splits == 1 {
                    let sum = partial.row_sum[vector as usize];
                    output[(query_index * head_dim + dimension) as usize] =
                        if sum == 0.0 { 0.0 } else { value / sum };
                } else {
                    let index =
                        ((query_index * (head_dim / 4) + (dimension >> 2)) * splits + split) * 4
                            + (dimension & 3);
                    scratch[index as usize] = value;
                }
            }
            if splits > 1 {
                let stats_index = scratch_values as i64 + (query_index * splits + split) * 2;
                scratch[stats_index as usize] = partial.row_maximum[vector as usize];
                scratch[stats_index as usize + 1] = partial.row_sum[vector as usize];
            }
        }
    }
    if splits > 1 {
        for query_index in 0..total_elements {
            let stat = |split: i64| -> (f32, f32) {
                let index = scratch_values as i64 + (query_index * splits + split) * 2;
                (scratch[index as usize], scratch[index as usize + 1])
            };
            let global_max = (0..splits)
                .map(|split| stat(split).0)
                .fold(f32::NEG_INFINITY, f32::max);
            let weight = |split: i64| -> f32 {
                let maximum = stat(split).0;
                if maximum == f32::NEG_INFINITY {
                    0.0
                } else {
                    (maximum - global_max).exp()
                }
            };
            let total: f32 = (0..splits).map(|split| stat(split).1 * weight(split)).sum();
            let inverse = if total == 0.0 { 0.0 } else { 1.0 / total };
            for dimension in 0..head_dim {
                let summed: f32 = (0..splits)
                    .map(|split| {
                        let index = ((query_index * (head_dim / 4) + (dimension >> 2)) * splits
                            + split)
                            * 4
                            + (dimension & 3);
                        scratch[index as usize] * weight(split)
                    })
                    .sum();
                output[(query_index * head_dim + dimension) as usize] = summed * inverse;
            }
        }
    }
    output
}
