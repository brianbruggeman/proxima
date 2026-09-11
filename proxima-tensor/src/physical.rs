//! Backend-neutral data for a cached-attention execution step.
//!
//! This module deliberately does not own an execution schedule. The existing
//! `Vec<BoundOp>`/`BoundOpBuilder` pipe remains the schedule; these values only
//! describe the operands and scalar domain needed when one bound step absorbs
//! the attention subgraph.

/// One key/value range (cached or new) for [`stream_cached_attention`].
#[cfg(test)]
pub struct AttentionRange<'buffer> {
    key: &'buffer [f32],
    value: &'buffer [f32],
}

/// Computes causal cached/new attention into a caller-owned output buffer.
///
/// The key ranges are streamed independently; no score, softmax, or
/// probability tensor is materialized. `query` is row-major
/// `[query_rows, head_dim]`, each key/value range is row-major
/// `[rows, head_dim]`, and `output` is `[query_rows, head_dim]`.
/// `cached_key_rows` and `new_key_rows` must agree with the corresponding
/// slice lengths. A `false` result means a shape is inconsistent and leaves
/// the output untouched.
#[must_use]
#[cfg(test)]
pub fn stream_cached_attention(
    query: &[f32],
    cached: AttentionRange<'_>,
    new: AttentionRange<'_>,
    output: &mut [f32],
    extents: AttentionExtents,
    scale: f32,
    causal_band: CausalBand,
) -> bool {
    let AttentionRange {
        key: cached_key,
        value: cached_value,
    } = cached;
    let AttentionRange {
        key: new_key,
        value: new_value,
    } = new;
    let query_width = extents.query_rows.checked_mul(extents.head_dim);
    let cached_key_width = extents.cached_key_rows.checked_mul(extents.head_dim);
    let new_key_width = extents.new_key_rows.checked_mul(extents.head_dim);
    let Some(query_width) = query_width else {
        return false;
    };
    let Some(cached_key_width) = cached_key_width else {
        return false;
    };
    let Some(new_key_width) = new_key_width else {
        return false;
    };
    if query.len() != query_width as usize
        || cached_key.len() != cached_key_width as usize
        || new_key.len() != new_key_width as usize
        || cached_value.len() != cached_key_width as usize
        || new_value.len() != new_key_width as usize
        || output.len() != query_width as usize
        || extents.head_dim == 0
    {
        return false;
    }

    let head_dim = extents.head_dim as usize;
    for query_row in 0..extents.query_rows as usize {
        let query_start = query_row * head_dim;
        let output_row = &mut output[query_start..query_start + head_dim];
        output_row.fill(0.0);
        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0;

        for (range_index, (key_range, values, key_rows)) in [
            (cached_key, cached_value, extents.cached_key_rows as usize),
            (new_key, new_value, extents.new_key_rows as usize),
        ]
        .into_iter()
        .enumerate()
        {
            for key_row in 0..key_rows {
                let key_position = if range_index == 0 {
                    key_row as i64 - extents.cached_key_rows as i64
                } else {
                    key_row as i64
                };
                let relative = key_position - query_row as i64;
                if relative < causal_band.lower_inclusive || relative > causal_band.upper_inclusive
                {
                    continue;
                }
                let key_start = key_row * head_dim;
                let mut score = 0.0;
                for dimension in 0..head_dim {
                    score += query[query_start + dimension] * key_range[key_start + dimension];
                }
                let score = score * scale;
                let value_row = &values[key_start..key_start + head_dim];
                if score > running_max {
                    let old_scale = libm::expf(running_max - score);
                    for dimension in 0..head_dim {
                        output_row[dimension] =
                            output_row[dimension] * old_scale + value_row[dimension];
                    }
                    running_sum = running_sum * old_scale + 1.0;
                    running_max = score;
                } else {
                    let weight = libm::expf(score - running_max);
                    for dimension in 0..head_dim {
                        output_row[dimension] += weight * value_row[dimension];
                    }
                    running_sum += weight;
                }
            }
        }
        if running_sum != 0.0 {
            for value in output_row {
                *value /= running_sum;
            }
        }
    }
    true
}

/// The even/odd query halves for [`stream_cached_attention_split`].
#[cfg(test)]
pub struct SplitQuery<'buffer> {
    even: &'buffer [f32],
    odd: &'buffer [f32],
}

/// One key/value range (cached or new) for [`stream_cached_attention_split`],
/// with the even/odd key halves kept separate the way RoPE leaves them.
#[cfg(test)]
pub struct SplitAttentionRange<'buffer> {
    key_even: &'buffer [f32],
    key_odd: &'buffer [f32],
    value: &'buffer [f32],
}

/// Computes the same streaming attention while consuming the split even/odd
/// RoPE operands used by the cached Mistral/Qwen3 graph. The key/value rows
/// remain contiguous in `head_dim`; the query/key halves are contiguous in
/// `head_dim / 2`. This is the zero-copy form the physical-plan matcher must
/// select for the real graph.
#[must_use]
#[cfg(test)]
pub fn stream_cached_attention_split(
    query: SplitQuery<'_>,
    cached: SplitAttentionRange<'_>,
    new: SplitAttentionRange<'_>,
    output: &mut [f32],
    extents: AttentionExtents,
    scale: f32,
    causal_band: CausalBand,
) -> bool {
    let SplitQuery {
        even: query_even,
        odd: query_odd,
    } = query;
    let SplitAttentionRange {
        key_even: cached_key_even,
        key_odd: cached_key_odd,
        value: cached_value,
    } = cached;
    let SplitAttentionRange {
        key_even: new_key_even,
        key_odd: new_key_odd,
        value: new_value,
    } = new;
    if extents.head_dim == 0 || !extents.head_dim.is_multiple_of(2) {
        return false;
    }
    let pair_dim = extents.head_dim / 2;
    let query_width = extents.query_rows.checked_mul(pair_dim);
    let cached_key_width = extents.cached_key_rows.checked_mul(pair_dim);
    let new_key_width = extents.new_key_rows.checked_mul(pair_dim);
    let value_width = extents.head_dim;
    let Some(query_width) = query_width else {
        return false;
    };
    let Some(cached_key_width) = cached_key_width else {
        return false;
    };
    let Some(new_key_width) = new_key_width else {
        return false;
    };
    let cached_value_width = extents.cached_key_rows.checked_mul(value_width);
    let new_value_width = extents.new_key_rows.checked_mul(value_width);
    let Some(cached_value_width) = cached_value_width else {
        return false;
    };
    let Some(new_value_width) = new_value_width else {
        return false;
    };
    if query_even.len() != query_width as usize
        || query_odd.len() != query_width as usize
        || cached_key_even.len() != cached_key_width as usize
        || cached_key_odd.len() != cached_key_width as usize
        || new_key_even.len() != new_key_width as usize
        || new_key_odd.len() != new_key_width as usize
        || cached_value.len() != cached_value_width as usize
        || new_value.len() != new_value_width as usize
        || output.len() != (extents.query_rows * value_width) as usize
    {
        return false;
    }

    let pair_dim = pair_dim as usize;
    let head_dim = value_width as usize;
    for query_row in 0..extents.query_rows as usize {
        let query_start = query_row * pair_dim;
        let output_start = query_row * head_dim;
        let output_row = &mut output[output_start..output_start + head_dim];
        output_row.fill(0.0);
        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0;

        for (range_index, (keys_even, keys_odd, values, key_rows)) in [
            (
                cached_key_even,
                cached_key_odd,
                cached_value,
                extents.cached_key_rows as usize,
            ),
            (
                new_key_even,
                new_key_odd,
                new_value,
                extents.new_key_rows as usize,
            ),
        ]
        .into_iter()
        .enumerate()
        {
            for key_row in 0..key_rows {
                let key_position = if range_index == 0 {
                    key_row as i64 - extents.cached_key_rows as i64
                } else {
                    key_row as i64
                };
                let relative = key_position - query_row as i64;
                if relative < causal_band.lower_inclusive || relative > causal_band.upper_inclusive
                {
                    continue;
                }
                let key_start = key_row * pair_dim;
                let mut score = 0.0;
                for dimension in 0..pair_dim {
                    score += query_even[query_start + dimension] * keys_even[key_start + dimension]
                        + query_odd[query_start + dimension] * keys_odd[key_start + dimension];
                }
                let score = score * scale;
                let value_start = key_row * head_dim;
                let value_row = &values[value_start..value_start + head_dim];
                if score > running_max {
                    let old_scale = libm::expf(running_max - score);
                    for dimension in 0..head_dim {
                        output_row[dimension] =
                            output_row[dimension] * old_scale + value_row[dimension];
                    }
                    running_sum = running_sum * old_scale + 1.0;
                    running_max = score;
                } else {
                    let weight = libm::expf(score - running_max);
                    for dimension in 0..head_dim {
                        output_row[dimension] += weight * value_row[dimension];
                    }
                    running_sum += weight;
                }
            }
        }
        if running_sum != 0.0 {
            for value in output_row {
                *value /= running_sum;
            }
        }
    }
    true
}

/// GQA form of `stream_cached_attention_split`. Each query row contains
/// `kv_heads * query_groups` heads, while each key/value row contains one
/// vector per KV head. The output is laid out as `[query, kv_head, group,
/// head_dim]`, matching the cached layer's `sugd` domain.
#[must_use]
pub fn stream_cached_attention_split_gqa(
    queries: [&[f32]; 2],
    keys: [[&[f32]; 2]; 2],
    values: [&[f32]; 2],
    output: &mut [f32],
    extents: AttentionExtents,
    scale: f32,
    bands: [CausalBand; 2],
) -> bool {
    if extents.head_dim == 0
        || !extents.head_dim.is_multiple_of(2)
        || extents.kv_heads == 0
        || extents.query_groups == 0
    {
        return false;
    }
    let pair_dim = extents.head_dim / 2;
    let query_count = extents
        .query_rows
        .checked_mul(extents.kv_heads)
        .and_then(|value| value.checked_mul(extents.query_groups));
    let query_width = query_count.and_then(|value| value.checked_mul(pair_dim));
    let cached_key_width = extents.cached_key_rows.checked_mul(extents.kv_heads);
    let new_key_width = extents.new_key_rows.checked_mul(extents.kv_heads);
    let Some(query_count) = query_count else {
        return false;
    };
    let Some(query_width) = query_width else {
        return false;
    };
    let Some(cached_key_width) = cached_key_width else {
        return false;
    };
    let Some(new_key_width) = new_key_width else {
        return false;
    };
    let cached_pair_width = cached_key_width.checked_mul(pair_dim);
    let new_pair_width = new_key_width.checked_mul(pair_dim);
    let cached_value_width = cached_key_width.checked_mul(extents.head_dim);
    let new_value_width = new_key_width.checked_mul(extents.head_dim);
    let Some(cached_pair_width) = cached_pair_width else {
        return false;
    };
    let Some(new_pair_width) = new_pair_width else {
        return false;
    };
    let Some(cached_value_width) = cached_value_width else {
        return false;
    };
    let Some(new_value_width) = new_value_width else {
        return false;
    };
    let output_width = query_count.checked_mul(extents.head_dim);
    let Some(output_width) = output_width else {
        return false;
    };
    // A zero-length cached range (a merged-KV fusion's `cached_key_rows: 0`)
    // never indexes into `keys[0]`/`values[0]` below -- the caller may pass
    // the same buffer it passed for the new range, so its length is not
    // required to match a zero-sized cached extent.
    let cached_range_empty = extents.cached_key_rows == 0;
    if queries
        .iter()
        .any(|query| query.len() != query_width as usize)
        || (!cached_range_empty
            && keys[0]
                .iter()
                .any(|key| key.len() != cached_pair_width as usize))
        || keys[1]
            .iter()
            .any(|key| key.len() != new_pair_width as usize)
        || (!cached_range_empty && values[0].len() != cached_value_width as usize)
        || values[1].len() != new_value_width as usize
        || output.len() != output_width as usize
    {
        return false;
    }

    let pair_dim = pair_dim as usize;
    let head_dim = extents.head_dim as usize;
    let kv_heads = extents.kv_heads as usize;
    let query_groups = extents.query_groups as usize;
    for query_row in 0..extents.query_rows as usize {
        for kv_head in 0..kv_heads {
            for query_group in 0..query_groups {
                let query_head = kv_head * query_groups + query_group;
                let query_start = (query_row * kv_heads * query_groups + query_head) * pair_dim;
                let output_start = (query_row * kv_heads * query_groups + query_head) * head_dim;
                let output_row = &mut output[output_start..output_start + head_dim];
                output_row.fill(0.0);
                let mut running_max = f32::NEG_INFINITY;
                let mut running_sum = 0.0;

                for (range_index, (keys_even, keys_odd, values, key_rows)) in [
                    (
                        keys[0][0],
                        keys[0][1],
                        values[0],
                        extents.cached_key_rows as usize,
                    ),
                    (
                        keys[1][0],
                        keys[1][1],
                        values[1],
                        extents.new_key_rows as usize,
                    ),
                ]
                .into_iter()
                .enumerate()
                {
                    for key_row in 0..key_rows {
                        let key_position = if range_index == 0 {
                            key_row as i64 - extents.cached_key_rows as i64
                        } else {
                            key_row as i64
                        };
                        let relative = key_position - query_row as i64;
                        let band = if range_index == 0 { bands[0] } else { bands[1] };
                        if relative < band.lower_inclusive || relative > band.upper_inclusive {
                            continue;
                        }
                        let key_start = (key_row * kv_heads + kv_head) * pair_dim;
                        let mut score = 0.0;
                        for dimension in 0..pair_dim {
                            score += queries[0][query_start + dimension]
                                * keys_even[key_start + dimension]
                                + queries[1][query_start + dimension]
                                    * keys_odd[key_start + dimension];
                        }
                        let score = score * scale;
                        let value_start = (key_row * kv_heads + kv_head) * head_dim;
                        let value_row = &values[value_start..value_start + head_dim];
                        if score > running_max {
                            let old_scale = libm::expf(running_max - score);
                            for dimension in 0..head_dim {
                                output_row[dimension] =
                                    output_row[dimension] * old_scale + value_row[dimension];
                            }
                            running_sum = running_sum * old_scale + 1.0;
                            running_max = score;
                        } else {
                            let weight = libm::expf(score - running_max);
                            for dimension in 0..head_dim {
                                output_row[dimension] += weight * value_row[dimension];
                            }
                            running_sum += weight;
                        }
                    }
                }
                if running_sum != 0.0 {
                    for value in output_row {
                        *value /= running_sum;
                    }
                }
            }
        }
    }
    true
}

/// The extents needed to describe one attention execution domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionExtents {
    pub query_rows: u64,
    pub cached_key_rows: u64,
    pub new_key_rows: u64,
    pub kv_heads: u64,
    pub query_groups: u64,
    pub head_dim: u64,
}

/// Inclusive key-index offsets permitted for a query row.
///
/// The offsets are relative to the query row's logical position. A cached
/// causal row commonly has a negative lower bound and an upper bound of zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CausalBand {
    pub lower_inclusive: i64,
    pub upper_inclusive: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streams_cached_and_new_ranges_without_materialized_weights() {
        let extents = AttentionExtents {
            query_rows: 1,
            cached_key_rows: 1,
            new_key_rows: 1,
            kv_heads: 1,
            query_groups: 1,
            head_dim: 2,
        };
        let mut output = [0.0; 2];
        assert!(stream_cached_attention(
            &[1.0, 0.0],
            AttentionRange {
                key: &[1.0, 0.0],
                value: &[2.0, 4.0],
            },
            AttentionRange {
                key: &[0.0, 1.0],
                value: &[6.0, 8.0],
            },
            &mut output,
            extents,
            1.0,
            CausalBand {
                lower_inclusive: -1,
                upper_inclusive: 0,
            },
        ));

        let first_weight = 1.0f32.exp();
        let expected_first = (first_weight * 2.0 + 1.0 * 6.0) / (first_weight + 1.0);
        let expected_second = (first_weight * 4.0 + 1.0 * 8.0) / (first_weight + 1.0);
        assert!((output[0] - expected_first).abs() < 1e-6);
        assert!((output[1] - expected_second).abs() < 1e-6);
    }

    #[test]
    fn rejects_inconsistent_attention_buffers() {
        let mut output = [0.0; 2];
        assert!(!stream_cached_attention(
            &[1.0],
            AttentionRange {
                key: &[],
                value: &[],
            },
            AttentionRange {
                key: &[],
                value: &[],
            },
            &mut output,
            AttentionExtents {
                query_rows: 1,
                cached_key_rows: 0,
                new_key_rows: 0,
                kv_heads: 1,
                query_groups: 1,
                head_dim: 2,
            },
            1.0,
            CausalBand {
                lower_inclusive: -1,
                upper_inclusive: 0,
            },
        ));
    }

    #[test]
    fn split_rope_stream_matches_full_vector_attention() {
        let extents = AttentionExtents {
            query_rows: 1,
            cached_key_rows: 1,
            new_key_rows: 1,
            kv_heads: 1,
            query_groups: 1,
            head_dim: 2,
        };
        let mut split_output = [0.0; 2];
        let mut full_output = [0.0; 2];
        let band = CausalBand {
            lower_inclusive: -1,
            upper_inclusive: 0,
        };
        assert!(stream_cached_attention_split(
            SplitQuery {
                even: &[1.0],
                odd: &[0.0],
            },
            SplitAttentionRange {
                key_even: &[1.0],
                key_odd: &[0.0],
                value: &[2.0, 4.0],
            },
            SplitAttentionRange {
                key_even: &[0.0],
                key_odd: &[1.0],
                value: &[6.0, 8.0],
            },
            &mut split_output,
            extents.clone(),
            1.0,
            band,
        ));
        assert!(stream_cached_attention(
            &[1.0, 0.0],
            AttentionRange {
                key: &[1.0, 0.0],
                value: &[2.0, 4.0],
            },
            AttentionRange {
                key: &[0.0, 1.0],
                value: &[6.0, 8.0],
            },
            &mut full_output,
            extents,
            1.0,
            band,
        ));
        assert_eq!(split_output, full_output);
    }

    #[test]
    fn gqa_stream_reuses_each_kv_head_for_query_groups() {
        let extents = AttentionExtents {
            query_rows: 1,
            cached_key_rows: 1,
            new_key_rows: 1,
            kv_heads: 1,
            query_groups: 2,
            head_dim: 2,
        };
        let mut output = [0.0; 4];
        assert!(stream_cached_attention_split_gqa(
            [&[1.0, 0.0][..], &[0.0, 1.0][..]],
            [[&[1.0][..], &[0.0][..]], [&[0.0][..], &[1.0][..]],],
            [&[2.0, 4.0][..], &[6.0, 8.0][..]],
            &mut output,
            extents,
            1.0,
            [
                CausalBand {
                    lower_inclusive: -1,
                    upper_inclusive: 0,
                },
                CausalBand {
                    lower_inclusive: -1,
                    upper_inclusive: 0,
                },
            ],
        ));
        let first_weight = 1.0f32.exp();
        let weighted_first = (first_weight * 2.0 + 6.0) / (first_weight + 1.0);
        let weighted_second = (first_weight * 4.0 + 8.0) / (first_weight + 1.0);
        assert!((output[0] - weighted_first).abs() < 1e-6);
        assert!((output[1] - weighted_second).abs() < 1e-6);
        let reverse_weighted_first = (2.0 + first_weight * 6.0) / (1.0 + first_weight);
        let reverse_weighted_second = (4.0 + first_weight * 8.0) / (1.0 + first_weight);
        assert!((output[2] - reverse_weighted_first).abs() < 1e-6);
        assert!((output[3] - reverse_weighted_second).abs() < 1e-6);
    }

    /// Regression for the merged-KV single-range fusion bug (ROW 366): the
    /// buggy bind duplicated the same bucketed-capacity key/value range into
    /// BOTH slots and neutered the "cached" half with an unreachable
    /// `[i64::MAX, i64::MAX]` band; the fix declares `cached_key_rows: 0` --
    /// an empty range, not a dead one. Both forms must agree exactly, since
    /// the dead band never touched the accumulator either way.
    #[test]
    fn merged_kv_zero_cached_range_matches_the_old_duplicated_capacity_with_a_dead_band() {
        let capacity = 4u64;
        let query_even = [1.0f32];
        let query_odd = [0.0f32];
        let key_even = [1.0f32, 0.0, 1.0, 0.0];
        let key_odd = [0.0f32, 1.0, 0.0, 1.0];
        let value = [2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        let extents_old = AttentionExtents {
            query_rows: 1,
            cached_key_rows: capacity,
            new_key_rows: capacity,
            kv_heads: 1,
            query_groups: 1,
            head_dim: 2,
        };
        let extents_new = AttentionExtents {
            cached_key_rows: 0,
            ..extents_old.clone()
        };
        let bands_old = [
            CausalBand {
                lower_inclusive: i64::MAX,
                upper_inclusive: i64::MAX,
            },
            CausalBand {
                lower_inclusive: i64::MIN,
                upper_inclusive: 1,
            },
        ];
        let bands_new = [
            CausalBand {
                lower_inclusive: i64::MIN,
                upper_inclusive: i64::MAX,
            },
            CausalBand {
                lower_inclusive: i64::MIN,
                upper_inclusive: 1,
            },
        ];
        let mut output_old = [0.0f32; 2];
        let mut output_new = [0.0f32; 2];
        assert!(stream_cached_attention_split_gqa(
            [&query_even[..], &query_odd[..]],
            [[&key_even[..], &key_odd[..]], [&key_even[..], &key_odd[..]]],
            [&value[..], &value[..]],
            &mut output_old,
            extents_old,
            1.0,
            bands_old,
        ));
        assert!(stream_cached_attention_split_gqa(
            [&query_even[..], &query_odd[..]],
            [[&key_even[..], &key_odd[..]], [&key_even[..], &key_odd[..]]],
            [&value[..], &value[..]],
            &mut output_new,
            extents_new,
            1.0,
            bands_new,
        ));
        assert_eq!(output_old, output_new);
    }
}
