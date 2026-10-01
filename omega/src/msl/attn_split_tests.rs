use super::attn_golden_tests::attention_op;
use super::*;

/// Feature-off byte identity: each kernel text, merge included, equals what
/// `main` emitted before `metal-attn-split-decode` existed. The goldens come
/// from `attn_golden_tests::record_main_goldens` run on an unpatched export.
#[cfg(not(feature = "metal-attn-split-decode"))]
mod feature_off {
    use super::super::attn_golden_tests::{golden_cases, golden_dir, rendered};

    #[test]
    #[ignore = "needs the goldens recorded from an unpatched main export (record_main_goldens); fails loudly without them"]
    fn attention_sources_match_the_recorded_main_goldens() {
        let cases = golden_cases();
        assert!(!cases.is_empty(), "zero golden cases would compare nothing");
        for (name, op, policy) in &cases {
            let path = golden_dir().join(format!("main_{name}.msl"));
            let golden = std::fs::read_to_string(&path).unwrap_or_else(|error| {
                panic!(
                    "no recorded main golden at {}: {error}; record it with `cargo test -p omega \
                     --lib record_main_goldens -- --ignored` on an unpatched export of main that \
                     carries attn_golden_tests.rs (RUSTFLAGS=\"--cap-lints warn\")",
                    path.display()
                )
            });
            assert_eq!(
                rendered(op, *policy),
                golden,
                "{name}: the feature-off kernel text must be byte-identical to main's"
            );
        }
    }
}

#[test]
fn split_limit_accepts_the_simd_width_and_rejects_more_with_a_typed_error() {
    let node = NodeId(7);

    validate_attention_split_limit(node, 1).expect("one split always fits one lane");
    validate_attention_split_limit(node, SIMD_WIDTH).expect("a full simdgroup of splits fits");

    for excess in [SIMD_WIDTH + 1, 2 * SIMD_WIDTH, 64] {
        let error = validate_attention_split_limit(node, excess)
            .expect_err("a split ceiling past the simdgroup width must be rejected");
        assert_eq!(
            error,
            EmitError::AttentionSplitsExceedSimdWidth {
                node,
                max: excess,
                limit: SIMD_WIDTH,
            }
        );
        assert!(
            error.to_string().contains(&excess.to_string()),
            "the message must name the offending ceiling: {error}"
        );
    }
}

#[test]
fn the_form_classifier_reads_the_operand_count_and_the_row_discriminator() {
    let relaxed = NumericPolicy::llama_relaxed();
    let static_op = attention_op(8, 4, 64, 32, 1, i64::MIN);
    let single_range = attention_op(9, 1, 8, 0, 256, i64::MIN);
    let two_range = attention_op(9, 8, 8, 512, 1, -511);

    assert_eq!(
        cached_attention_form(&static_op.kind, relaxed),
        Some(CachedAttentionForm::Static)
    );
    assert_eq!(
        cached_attention_form(&single_range.kind, relaxed),
        Some(CachedAttentionForm::SingleRangeDynamic { merge: true }),
        "256 keys past the at-scale knee slice across threadgroups under relaxed"
    );
    assert_eq!(
        cached_attention_form(&single_range.kind, NumericPolicy::bit_exact()),
        Some(CachedAttentionForm::SingleRangeDynamic { merge: false })
    );
    assert_eq!(
        cached_attention_form(&two_range.kind, relaxed),
        Some(CachedAttentionForm::TwoRangeCachedBound),
        "head_dim 8 is not a whole float4 V lane set, so even with the feature on it stays one dispatch"
    );
}

/// The partial-rotary single-range op carries 12 operands (8 base + 3 pass
/// planes + the ninth `cached_len`). Every renderer site already read that as
/// the dynamic form; `entry_name` and the kernel identity read `== 9` only,
/// so the 12-operand op was named as a static op with its bucket extent.
#[test]
fn a_twelve_operand_single_range_op_is_named_as_the_dynamic_form() {
    let mut partial_rotary = attention_op(12, 1, 8, 0, 256, i64::MIN);
    let BoundOpKind::CachedAttention { rotary_dim, .. } = &mut partial_rotary.kind else {
        unreachable!("attention_op always builds a CachedAttention kind");
    };
    *rotary_dim = 4;

    assert_eq!(
        cached_attention_form(&partial_rotary.kind, NumericPolicy::bit_exact()),
        Some(CachedAttentionForm::SingleRangeDynamic { merge: false })
    );
    let name = entry_name(&partial_rotary, NumericPolicy::bit_exact());
    assert!(
        name.contains("_udyn_"),
        "the dynamic form's name carries the `dyn` upper token, not a bucket extent: {name}"
    );
}

#[cfg(feature = "metal-attn-split-decode")]
mod decode_split {
    use super::super::attn_golden_tests::attention_op;
    use super::super::*;

    const SLIDING_LOWER: i64 = -511;

    type Misfit = (&'static str, fn(&mut BoundOp));

    fn relaxed_form(cached_key_rows: u64, head_dim: u64) -> Option<CachedAttentionForm> {
        let op = attention_op(9, 8, head_dim, cached_key_rows, 1, SLIDING_LOWER);
        cached_attention_form(&op.kind, NumericPolicy::llama_relaxed())
    }

    fn assert_default_sizing() {
        assert_eq!(
            crate::sized::ATTENTION_SPLIT_KEYS_PER_SPLIT_DECODE,
            32,
            "the expectations below assume the default [attention_splits] sizing"
        );
        assert_eq!(crate::sized::ATTENTION_SPLIT_MAX, 32);
        assert_eq!(crate::sized::ATTENTION_BLOCK_WIDTH, 32);
        assert_eq!(crate::sized::ATTENTION_CONTEXT_CHUNK_CAP, 4);
    }

    #[test]
    fn the_split_and_chunk_counts_follow_the_bucket_capacity() {
        assert_default_sizing();
        let expected = [(32_u64, 2_u64, 1_u64), (512, 17, 1), (2048, 32, 2)];

        for head_dim in [256_u64, 512] {
            for (cached_key_rows, splits, chunks) in expected {
                assert_eq!(
                    relaxed_form(cached_key_rows, head_dim),
                    Some(CachedAttentionForm::TwoRangeDecodeSplit { splits, chunks }),
                    "head_dim {head_dim}, capacity {}",
                    cached_key_rows + 1
                );
            }
        }
        assert_eq!(
            decode_splits_for(32),
            1,
            "a 32-key capacity is one direct-output split"
        );
        assert_eq!(
            decode_splits_for(1_000_000),
            crate::sized::ATTENTION_SPLIT_MAX
        );
        assert_eq!(
            decode_chunks_for(4097, 256),
            4,
            "llama's nsg rule steps 1, 2, 4"
        );
        assert_eq!(
            decode_chunks_for(1_000_000, 256),
            crate::sized::ATTENTION_CONTEXT_CHUNK_CAP
        );
    }

    #[test]
    fn shapes_and_policies_the_split_cannot_serve_stay_one_dispatch() {
        let two_range = attention_op(9, 8, 256, 512, 1, SLIDING_LOWER);
        assert_eq!(
            cached_attention_form(&two_range.kind, NumericPolicy::bit_exact()),
            Some(CachedAttentionForm::TwoRangeCachedBound),
            "bit_exact withholds both reassociations the split needs"
        );
        let tree_only = NumericPolicy::bit_exact().with_contraction(true);
        assert_eq!(
            cached_attention_form(&two_range.kind, tree_only),
            Some(CachedAttentionForm::TwoRangeCachedBound)
        );

        let mut partial_rotary = attention_op(12, 8, 256, 512, 1, SLIDING_LOWER);
        set_field(&mut partial_rotary, |fields| *fields.rotary_dim = 128);
        assert_eq!(
            cached_attention_form(&partial_rotary.kind, NumericPolicy::llama_relaxed()),
            Some(CachedAttentionForm::TwoRangeCachedBound),
            "the partial-rotary pass plane is not served by the block-staged body"
        );

        let misfits: [Misfit; 3] = [
            ("multi-row prefill", |op| {
                set_field(op, |fields| *fields.query_rows = 4)
            }),
            ("two new keys", |op| {
                set_field(op, |fields| *fields.new_key_rows = 2)
            }),
            ("head_dim not a float4 lane set", |op| {
                set_field(op, |fields| {
                    *fields.head_dim = 24;
                    *fields.rotary_dim = 24;
                });
            }),
        ];
        for (label, mutate) in misfits {
            let mut op = attention_op(9, 8, 256, 512, 1, SLIDING_LOWER);
            mutate(&mut op);
            assert_eq!(
                cached_attention_form(&op.kind, NumericPolicy::llama_relaxed()),
                Some(CachedAttentionForm::TwoRangeCachedBound),
                "{label} must not take the decode split"
            );
        }
    }

    struct Fields<'a> {
        query_rows: &'a mut u64,
        new_key_rows: &'a mut u64,
        head_dim: &'a mut u64,
        rotary_dim: &'a mut u64,
    }

    fn set_field(op: &mut BoundOp, mutate: impl FnOnce(Fields<'_>)) {
        let BoundOpKind::CachedAttention {
            query_rows,
            new_key_rows,
            head_dim,
            rotary_dim,
            ..
        } = &mut op.kind
        else {
            unreachable!("attention_op always builds a CachedAttention kind");
        };
        mutate(Fields {
            query_rows,
            new_key_rows,
            head_dim,
            rotary_dim,
        });
    }

    fn threadgroup_layout(op: &BoundOp) -> (u64, u64) {
        let policy = NumericPolicy::llama_relaxed();
        let no_codecs: Vec<Option<Codec>> = Vec::new();
        let threads = grid_threads(op, &no_codecs, policy, false)
            .expect("a CachedAttention op always has a thread count");
        let width = tiled_gemm_threadgroup_width(op, &no_codecs, policy)
            .expect("a CachedAttention op always has a threadgroup width");
        (threads / width, width)
    }

    /// Design point 7: one threadgroup per (query head, split), 8 heads.
    #[test]
    fn the_partial_dispatches_one_threadgroup_per_head_and_split() {
        assert_default_sizing();
        let expected = [(32_u64, 16_u64, 32_u64), (512, 136, 32), (2048, 256, 64)];

        for head_dim in [256_u64, 512] {
            for (cached_key_rows, threadgroups, width) in expected {
                let op = attention_op(9, 8, head_dim, cached_key_rows, 1, SLIDING_LOWER);
                assert_eq!(
                    threadgroup_layout(&op),
                    (threadgroups, width),
                    "head_dim {head_dim}, capacity {}",
                    cached_key_rows + 1
                );
            }
        }
    }

    #[test]
    fn the_partial_kernel_source_carries_the_split_grid_and_the_interleaved_store() {
        assert_default_sizing();
        for head_dim in [256_u64, 512] {
            for cached_key_rows in [32_u64, 512, 2048] {
                for lower in [SLIDING_LOWER, i64::MIN] {
                    let op = attention_op(9, 8, head_dim, cached_key_rows, 1, lower);
                    let kernel = emit(&op, &PackedOperands::new(), NumericPolicy::llama_relaxed())
                        .expect("the decode split kernel emits");
                    let label =
                        format!("head_dim {head_dim} cached {cached_key_rows} lower {lower}");

                    assert!(kernel.entry.ends_with("_ds"), "{label}: {}", kernel.entry);
                    for required in [
                        "uint tgid [[threadgroup_position_in_grid]]",
                        "ushort simdgroup_slot [[simdgroup_index_in_threadgroup]]",
                        "long split = (long)tgid % splits;",
                        "long query_head = ((long)tgid / splits) % (kv_heads * query_groups);",
                        "long first_key = max(0L, cached_key_rows + cached_lower + query_row);",
                        "long slice_len = (band + splits - 1L) / splits;",
                        "long hi = min(lo + slice_len, last_key + 1L);",
                        "if (splits == 1L) {",
                        "long stats_index = u.total_elements * head_dim * splits + (query_index * splits + split) * 2L;",
                        "attn_scratch[((value_base + (dimension >> 2)) * splits + split) * 4L + (dimension & 3L)]",
                        "long cached_key_rows = (long)in8[0];",
                    ] {
                        assert!(
                            kernel.source.contains(required),
                            "{label}: missing `{required}`"
                        );
                    }
                    assert!(
                        !kernel
                            .source
                            .contains("long cached_key_rows = u.cached_key_rows"),
                        "{label}: the bucket extent must not be a compiled or uniform constant"
                    );

                    let cap = effective_context_chunk_cap(1, head_dim);
                    let block_width = crate::sized::ATTENTION_BLOCK_WIDTH;
                    let threadgroup_bytes = 4 * (cap * head_dim + 2 * cap + cap * block_width);
                    assert!(
                        threadgroup_bytes
                            <= crate::sized::CACHED_ATTENTION_THREADGROUP_MEMORY_BYTES,
                        "{label}: {threadgroup_bytes} bytes of threadgroup memory"
                    );
                }
            }
        }
    }

    #[test]
    fn the_split_binds_scratch_and_a_merge_follows_above_one_split() {
        let policy = NumericPolicy::llama_relaxed();
        let split = attention_op(9, 8, 256, 512, 1, SLIDING_LOWER);
        let direct = attention_op(9, 8, 256, 31, 1, SLIDING_LOWER);

        let split_kernel = emit(&split, &PackedOperands::new(), policy).expect("emits");
        assert!(split_kernel.bindings.contains(&Binding::Scratch));
        assert!(
            !split_kernel
                .bindings
                .iter()
                .any(|binding| matches!(binding, Binding::Output(_)))
        );
        assert_eq!(
            split_kernel.bindings.len(),
            9 + 2,
            "nine operands, scratch, uniforms"
        );
        assert!(cached_attention_merge_needed(&split.kind, policy));
        assert!(
            emit_cached_attention_merge(&split, policy)
                .expect("emits")
                .is_some()
        );

        let direct_kernel = emit(&direct, &PackedOperands::new(), policy).expect("emits");
        assert!(
            direct_kernel
                .bindings
                .contains(&Binding::Output(direct.node))
        );
        assert!(!direct_kernel.bindings.contains(&Binding::Scratch));
        assert!(!cached_attention_merge_needed(&direct.kind, policy));
        assert!(
            emit_cached_attention_merge(&direct, policy)
                .expect("emits")
                .is_none(),
            "a single split is written straight to the output; no merge dispatch"
        );
    }

    /// llama.cpp's reduce shape (`ggml-metal-ops.cpp:3712`): one threadgroup
    /// per output row, `32 * nwg` threads.
    #[test]
    fn the_merge_is_llamas_reduce_eight_groups_of_a_full_simdgroup_per_split() {
        let policy = NumericPolicy::llama_relaxed();
        let op = attention_op(9, 8, 512, 2048, 1, i64::MIN);
        let merge = emit_cached_attention_merge(&op, policy)
            .expect("emits")
            .expect("capacity 2049 needs a merge");

        let width = SIMD_WIDTH * crate::sized::ATTENTION_SPLIT_MAX;
        assert_eq!(merge.grid.threadgroup_width, Some(width));
        assert_eq!(
            merge.grid.threads / width,
            8,
            "one threadgroup per (row, head)"
        );
        assert_eq!(width, 1024, "default sizing is llama's 8 x 1024");
        for required in [
            "uint tg [[threadgroup_position_in_grid]]",
            "ushort simdgroup_count [[simdgroups_per_threadgroup]]",
            "float global_max = simd_max(own_max);",
            "float total = simd_sum(own_sum * weight);",
            "float4 summed = simd_sum(live ? partial4[dim4 * splits + (long)lane] * weight : float4(0.0f));",
            "device const float* stats = in0 + u.total_elements * head_dim * splits;",
        ] {
            assert!(
                merge.source.contains(required),
                "merge source missing `{required}`"
            );
        }
        assert!(
            !merge
                .source
                .contains("for (long split = 0; split < splits; split++)"),
            "the per-dimension serial walk over splits is what this kernel replaces"
        );
    }

    #[test]
    fn the_merge_rejects_a_head_dim_that_is_not_a_whole_float4_lane_set() {
        let mut op = attention_op(9, 8, 256, 512, 1, SLIDING_LOWER);
        let BoundOpKind::CachedAttention { head_dim, .. } = &mut op.kind else {
            unreachable!("attention_op always builds a CachedAttention kind");
        };
        *head_dim = 12;

        let error = render_cached_attention_merge(&op, "entry")
            .expect_err("a head_dim that float4 partial reads cannot tile is rejected");
        assert_eq!(
            error,
            EmitError::AttentionBlockMisaligned {
                node: op.node,
                head_dim: 12
            }
        );
    }

    /// Past `u32::MAX` threads the partial and the interleaved merge must
    /// each launch the flat 2D form and render the matching widened source:
    /// the two come from one `grid2d_for` / `merge_grid` decision.
    #[test]
    fn the_split_partial_and_the_interleaved_merge_past_u32_threads_launch_flat_with_a_widened_source()
     {
        const WIDE_HEADS: u64 = 8_388_608;
        let limit = u64::from(u32::MAX);
        let policy = NumericPolicy::llama_relaxed();
        let wide = attention_op(9, WIDE_HEADS, 256, 512, 1, SLIDING_LOWER);
        let narrow = attention_op(9, 8, 256, 512, 1, SLIDING_LOWER);

        let partial = emit(&wide, &PackedOperands::new(), policy).expect("the wide partial emits");
        let (_, shape) = kernel_dispatch_shape(&wide, &PackedOperands::new(), policy)
            .expect("the wide partial has a dispatch shape");
        assert!(partial.grid.threads > limit, "{}", partial.grid.threads);
        assert_eq!(partial.grid, shape, "emit and the cache-hit shape disagree");
        let partial_spec = partial.grid.grid2d.expect("a wide partial dispatches flat");
        assert_eq!(partial_spec.form, Grid2DForm::FlatThreadgroupIndex);
        assert_eq!(
            partial_spec.threadgroups_x * partial_spec.threadgroups_y,
            partial
                .grid
                .threads
                .div_ceil(partial_spec.threads_per_threadgroup_x)
        );
        assert!(partial.source.contains("ulong wide_group_index"));
        assert!(partial.source.contains("ulong tgid = wide_group_index;"));
        assert!(!partial.source.contains("uint tgid [["));

        let merge = emit_cached_attention_merge(&wide, policy)
            .expect("the wide merge emits")
            .expect("capacity 513 needs a merge");
        let width = SIMD_WIDTH * crate::sized::ATTENTION_SPLIT_MAX;
        assert_eq!(merge.grid.threads, WIDE_HEADS * width);
        assert!(merge.grid.threads > limit);
        let merge_spec = merge.grid.grid2d.expect("a wide merge dispatches flat");
        assert_eq!(merge_spec.form, Grid2DForm::FlatThreadgroupIndex);
        assert_eq!(merge_spec.threads_per_threadgroup_x, width);
        assert_eq!(
            merge_spec.threadgroups_x * merge_spec.threadgroups_y,
            WIDE_HEADS
        );
        assert!(merge.source.contains("ulong wide_group_index"));
        assert!(merge.source.contains("ulong tg = wide_group_index;"));
        assert!(!merge.source.contains("uint tg [["));

        let narrow_partial = emit(&narrow, &PackedOperands::new(), policy).expect("emits");
        let narrow_merge = emit_cached_attention_merge(&narrow, policy)
            .expect("emits")
            .expect("capacity 513 needs a merge");
        for (label, kernel_grid, source) in [
            ("partial", narrow_partial.grid, &narrow_partial.source),
            ("merge", narrow_merge.grid, &narrow_merge.source),
        ] {
            assert_eq!(kernel_grid.grid2d, None, "the narrow {label} stays 1D");
            assert!(!source.contains("ulong wide_group_index"), "{label}");
        }
    }

    /// The single-range split writer and the shared merge agree on the
    /// interleaved layout, so the qwen path and the decode form use one merge.
    #[test]
    fn the_single_range_writer_stores_the_same_interleaved_layout() {
        let policy = NumericPolicy::llama_relaxed();
        let op = attention_op(9, 1, 8, 0, 256, i64::MIN);
        let source = render_cached_attention(&op, "entry", policy).expect("renders");

        assert!(source.contains("constexpr long total_rows = 1 * kv_heads * query_groups;"));
        assert!(source.contains("long stats_index = total_rows * head_dim * splits + (query_index * splits + split) * 2L;"));
        assert!(
            !source
                .contains("long scratch_index = (query_index * splits + split) * (2L + head_dim);")
        );
    }

    fn reference_slice(first_key: i64, last_key: i64, splits: i64, split: i64) -> (i64, i64) {
        let band = last_key + 1 - first_key;
        let slice_len = (band + splits - 1) / splits;
        let lo = first_key + split * slice_len;
        (lo, (lo + slice_len).min(last_key + 1))
    }

    /// Property, stated over the whole domain the sized config admits: the
    /// per-split slices tile `[first_key, last_key]` with no overlap and no
    /// gap, so every live key is scored by exactly one threadgroup.
    #[test]
    fn the_split_slices_partition_the_live_band_exactly_once() {
        let mut checked = 0_u64;
        for splits in 1..=32_i64 {
            for band in 1..=4096_i64 {
                for first_key in [0_i64, 1, 7] {
                    let last_key = first_key + band - 1;
                    let mut covered = 0_i64;
                    let mut next_expected = first_key;
                    for split in 0..splits {
                        let (lo, hi) = reference_slice(first_key, last_key, splits, split);
                        if hi > lo {
                            assert_eq!(
                                lo, next_expected,
                                "splits {splits} band {band}: gap or overlap at split {split}"
                            );
                            next_expected = hi;
                            covered += hi - lo;
                        }
                    }
                    assert_eq!(
                        covered, band,
                        "splits {splits} band {band} first_key {first_key}"
                    );
                    assert_eq!(next_expected, last_key + 1);
                    checked += 1;
                }
            }
        }
        assert_eq!(
            checked,
            32 * 4096 * 3,
            "the property must have run over the whole domain"
        );
    }

    /// The design's 4-key worked example, in f32 arithmetic: two splits
    /// `{0,1}` and `{2,3}` merged with the reduce's rescale equal the
    /// single-pass softmax. Internal consistency of the algebra, not an oracle.
    #[test]
    fn the_split_merge_algebra_matches_single_pass_on_the_four_key_fixture() {
        let scores = [1.0_f32, 2.0, 3.0, 0.0];
        let values = [[1.0_f32, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]];

        let partials = [
            partial(&scores[..2], &values[..2]),
            partial(&scores[2..], &values[2..]),
        ];
        let global_max = partials
            .iter()
            .map(|(max, _, _)| *max)
            .fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f32> = partials
            .iter()
            .map(|(max, _, _)| (max - global_max).exp())
            .collect();
        let total: f32 = partials
            .iter()
            .zip(&weights)
            .map(|((_, sum, _), weight)| sum * weight)
            .sum();
        let merged: Vec<f32> = (0..2)
            .map(|dim| {
                let weighted: f32 = partials
                    .iter()
                    .zip(&weights)
                    .map(|((_, _, acc), weight)| acc[dim] * weight)
                    .sum();
                weighted / total
            })
            .collect();

        let (single_max, single_sum, single_acc) = partial(&scores, &values);
        let single: Vec<f32> = single_acc.iter().map(|value| value / single_sum).collect();
        assert_eq!(single_max, 3.0);

        let expected = [0.795_175_8_f32, 0.880_797_1];
        for (dim, ((merged_value, single_value), expected_value)) in
            merged.iter().zip(&single).zip(&expected).enumerate()
        {
            let tolerance = 4.0 * ulp(*single_value);
            assert!(
                (merged_value - single_value).abs() <= tolerance,
                "dim {dim}: merged {merged_value} single {single_value}"
            );
            assert!(
                (merged_value - expected_value).abs() <= tolerance,
                "dim {dim}: merged {merged_value} expected {expected_value}"
            );
        }
        assert!((partials[0].1 - 1.367_879_4).abs() < 1e-6);
        assert!((partials[1].1 - 1.049_787).abs() < 1e-6);
        assert!((partials[1].2[0] - 1.099_574_1).abs() < 1e-6);

        let row = 0_usize;
        let (head_dim, splits, split, dim) = (32_usize, 2_usize, 1_usize, 0_usize);
        let scalar_index = ((row * (head_dim / 4) + (dim >> 2)) * splits + split) * 4 + (dim & 3);
        assert_eq!(
            scalar_index, 4,
            "split 1's dim-0 value sits at float index 4"
        );
    }

    fn partial(scores: &[f32], values: &[[f32; 2]]) -> (f32, f32, [f32; 2]) {
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        let mut acc = [0.0_f32; 2];
        for (score, value) in scores.iter().zip(values) {
            let weight = (score - max).exp();
            sum += weight;
            acc[0] += weight * value[0];
            acc[1] += weight * value[1];
        }
        (max, sum, acc)
    }

    fn ulp(value: f32) -> f32 {
        f32::from_bits(value.to_bits() + 1) - value
    }
}
