use std::collections::BTreeSet;

use super::attn_golden_tests::{attention_op, attention_rows_op};
use super::*;

#[cfg(feature = "metal-attn-variants")]
#[test]
fn card_12_variant_config_defaults_preserve_the_legacy_axes() {
    assert_eq!(
        AttentionVariant::default(),
        AttentionVariant {
            kv_storage: AttentionKvStorage::F32,
            mma_precision: AttentionMmaPrecision::Legacy,
            kv_reuse: AttentionKvReuse::Legacy,
            tile_height: AttentionTileHeight::Legacy,
            query_parallelism: AttentionQueryParallelism::Legacy,
            simd_topology: AttentionSimdTopology::Legacy,
            prefetch: AttentionPrefetch::Off,
        }
    );
}

#[cfg(feature = "metal-attn-variants")]
#[test]
fn card_12_variant_config_stores_each_axis_independently() {
    let selected = AttentionVariant {
        kv_storage: AttentionKvStorage::Bf8,
        mma_precision: AttentionMmaPrecision::F16,
        kv_reuse: AttentionKvReuse::SharedK,
        tile_height: AttentionTileHeight::Rows8,
        query_parallelism: AttentionQueryParallelism::SimdgroupRows,
        simd_topology: AttentionSimdTopology::PerHead,
        prefetch: AttentionPrefetch::NextBlock,
    };
    let root_export: crate::AttentionVariant = selected;
    assert_eq!(root_export, selected);
    assert_eq!(selected.kv_storage, AttentionKvStorage::Bf8);
    assert_eq!(selected.mma_precision, AttentionMmaPrecision::F16);
    assert_eq!(selected.kv_reuse, AttentionKvReuse::SharedK);
    assert_eq!(selected.tile_height, AttentionTileHeight::Rows8);
    assert_eq!(
        selected.query_parallelism,
        AttentionQueryParallelism::SimdgroupRows
    );
    assert_eq!(selected.simd_topology, AttentionSimdTopology::PerHead);
    assert_eq!(selected.prefetch, AttentionPrefetch::NextBlock);
    let prefetch_only = AttentionVariant {
        prefetch: AttentionPrefetch::NextBlock,
        ..AttentionVariant::default()
    };
    assert_eq!(prefetch_only.prefetch, AttentionPrefetch::NextBlock);
    assert_eq!(prefetch_only.kv_storage, AttentionKvStorage::F32);
    assert_eq!(prefetch_only.mma_precision, AttentionMmaPrecision::Legacy);
    assert_eq!(prefetch_only.kv_reuse, AttentionKvReuse::Legacy);
    assert_eq!(prefetch_only.tile_height, AttentionTileHeight::Legacy);
    assert_eq!(
        prefetch_only.query_parallelism,
        AttentionQueryParallelism::Legacy
    );
    assert_eq!(prefetch_only.simd_topology, AttentionSimdTopology::Legacy);
}

const SLIDING_LOWER: i64 = -511;
const GLOBAL_LOWER: i64 = i64::MIN;
const VERIFY_ROWS: [u64; 4] = [2, 5, 17, 49];
const CAPACITIES: [u64; 3] = [32, 512, 2048];
const THREADGROUP_BUDGET: u64 = 32_768;

fn assert_default_sizing() {
    assert_eq!(crate::sized::ATTENTION_ROWS_KEYS_PER_BLOCK, 128);
    assert_eq!(crate::sized::ATTENTION_ROWS_KEYS_PER_SPLIT, 64);
    assert_eq!(crate::sized::ATTENTION_ROWS_HEAD_DIMS_PER_SIMDGROUP, 64);
    assert_eq!(crate::sized::ATTENTION_ROWS_MIN_SIMDGROUPS, 2);
    assert_eq!(crate::sized::ATTENTION_ROWS_MMA_MIN_QUERY_ROWS, 2);
    assert_eq!(crate::sized::ATTENTION_ROWS_VECTOR_BLOCKS_PER_TILE, 2);
    assert_eq!(crate::sized::ATTENTION_ROWS_ACCUMULATOR_FRAGMENTS, 16);
    assert_eq!(crate::sized::ATTENTION_ROWS_TARGET_SIMDGROUPS, 256);
    const { assert!(!crate::sized::ATTENTION_ROWS_MMA_HALF) };
    assert_eq!(
        crate::sized::CACHED_ATTENTION_THREADGROUP_MEMORY_BYTES,
        THREADGROUP_BUDGET
    );
    assert_eq!(crate::sized::ATTENTION_SPLIT_MAX, 32);
}

fn sliding_op(cached_key_rows: u64, rows: u64) -> BoundOp {
    attention_rows_op(9, 8, 256, cached_key_rows, rows, SLIDING_LOWER)
}

fn global_op(cached_key_rows: u64, rows: u64) -> BoundOp {
    attention_rows_op(9, 8, 512, cached_key_rows, rows, GLOBAL_LOWER)
}

fn relaxed_form(op: &BoundOp) -> Option<CachedAttentionForm> {
    cached_attention_form(&op.kind, NumericPolicy::llama_relaxed())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Chosen {
    Decode,
    RowTiled,
    OneDispatch,
}

fn chosen(form: Option<CachedAttentionForm>) -> Chosen {
    match form {
        Some(CachedAttentionForm::TwoRangeDecodeSplit { .. }) => Chosen::Decode,
        Some(CachedAttentionForm::TwoRangeRowTiled { .. }) => Chosen::RowTiled,
        Some(CachedAttentionForm::TwoRangeCachedBound) => Chosen::OneDispatch,
        other => panic!("not a two-range form: {other:?}"),
    }
}

#[test]
fn the_row_count_the_policy_and_the_head_group_choose_the_form() {
    assert_default_sizing();
    let mut cells = 0_u32;

    for rows in [1_u64, 2, 5, 7, 8, 17, 49, 64, 65, 971] {
        for groups in [8_u64, 2] {
            let head_dim = if groups == 8 { 256 } else { 64 };
            let op = attention_rows_op(9, groups, head_dim, 512, rows, SLIDING_LOWER);
            let relaxed_expected = match (rows, groups) {
                (1, _) => Chosen::Decode,
                (2.., 8) => Chosen::RowTiled,
                (8.., _) => Chosen::RowTiled,
                _ => Chosen::OneDispatch,
            };
            assert_eq!(
                chosen(relaxed_form(&op)),
                relaxed_expected,
                "relaxed, rows {rows}, groups {groups}"
            );
            assert_eq!(
                chosen(cached_attention_form(&op.kind, NumericPolicy::bit_exact())),
                Chosen::OneDispatch,
                "bit_exact withholds both reassociations: rows {rows}, groups {groups}"
            );
            cells += 2;
        }
    }
    assert_eq!(cells, 40, "10 row counts x 2 group counts x 2 policies");
}

#[test]
fn shapes_the_tile_cannot_serve_stay_on_the_one_dispatch_form() {
    let mut misfits: Vec<(&str, BoundOp)> = Vec::new();

    let mut unequal = sliding_op(512, 5);
    set_new_key_rows(&mut unequal, 4);
    misfits.push(("query rows differ from new key rows", unequal));

    let mut partial_rotary = attention_rows_op(12, 8, 256, 512, 5, SLIDING_LOWER);
    set_rotary_dim(&mut partial_rotary, 128);
    misfits.push(("a partial-rotary head", partial_rotary));

    misfits.push((
        "a bucket that is not whole 8-key fragments",
        sliding_op(515, 5),
    ));
    misfits.push((
        "a head_dim whose half is not a whole MMA depth",
        attention_rows_op(9, 8, 24, 512, 5, SLIDING_LOWER),
    ));
    misfits.push((
        "a head_dim not divisible over the simdgroups",
        attention_rows_op(9, 8, 272, 512, 5, SLIDING_LOWER),
    ));
    misfits.push((
        "fewer rows than one fragment of a group count that is not whole 8-row blocks",
        attention_rows_op(9, 2, 64, 512, 5, SLIDING_LOWER),
    ));
    misfits.push((
        "a group count whose eight-row unit exceeds the threadgroup memory",
        attention_rows_op(9, 12, 256, 512, 9, SLIDING_LOWER),
    ));
    misfits.push((
        "a vector block whose output fragments exceed the register budget",
        attention_rows_op(9, 8, 4096, 512, 5, SLIDING_LOWER),
    ));
    misfits.push((
        "a group count whose eight-row unit exceeds the register budget",
        attention_rows_op(9, 4, 256, 512, 9, SLIDING_LOWER),
    ));

    let cells = misfits.len();
    for (label, op) in misfits {
        assert_eq!(chosen(relaxed_form(&op)), Chosen::OneDispatch, "{label}");
    }
    assert_eq!(cells, 9, "every misfit must have been classified");
}

fn set_new_key_rows(op: &mut BoundOp, rows: u64) {
    let BoundOpKind::CachedAttention { new_key_rows, .. } = &mut op.kind else {
        unreachable!("attention_op always builds a CachedAttention kind");
    };
    *new_key_rows = rows;
}

fn set_rotary_dim(op: &mut BoundOp, rotary: u64) {
    let BoundOpKind::CachedAttention { rotary_dim, .. } = &mut op.kind else {
        unreachable!("attention_op always builds a CachedAttention kind");
    };
    *rotary_dim = rotary;
}

/// `(query rows, kv heads, query groups, head dim, bucket capacity, expected
/// rows per threadgroup)`: a prefill takes the full tile height its vector
/// block budget allows, a verify of two rows keeps one unit per tile because a
/// taller tile would leave the GPU short of simdgroups, and four rows of the
/// widest head take two units once the splits still fill it.
const TILE_CASES: [(u64, u64, u64, u64, u64, u64); 10] = [
    (971, 1, 8, 512, 1942, 2),
    (971, 1, 8, 256, 1483, 2),
    (1000, 8, 2, 64, 2024, 8),
    (600, 8, 2, 128, 1240, 8),
    (600, 8, 8, 128, 1240, 2),
    (600, 8, 16, 256, 1240, 1),
    (2, 1, 8, 512, 1634, 1),
    (4, 1, 8, 512, 1636, 2),
    (49, 1, 8, 256, 561, 2),
    (17, 1, 8, 512, 2065, 2),
];

#[test]
fn rows_per_threadgroup_follows_the_reuse_registers_memory_and_occupancy_budgets() {
    assert_default_sizing();
    for (query_rows, kv_heads, groups, head_dim, capacity, rows) in TILE_CASES {
        let cell = format!("rows {query_rows} kv {kv_heads} groups {groups} head_dim {head_dim}");
        assert_eq!(
            rows_per_threadgroup(query_rows, kv_heads, groups, head_dim, capacity),
            rows,
            "{cell}"
        );
        let (unit_rows, unit_blocks) = tile_unit(groups);
        let units = rows / unit_rows;
        assert!(units >= 1 && rows % unit_rows == 0, "{cell}: whole units");
        assert!(
            units * unit_blocks <= crate::sized::ATTENTION_ROWS_VECTOR_BLOCKS_PER_TILE.max(unit_blocks),
            "{cell}: the tile shares more vector blocks than the reuse budget"
        );
        assert!(
            units == 1
                || units * tile_unit_fragments(groups, head_dim)
                    <= crate::sized::ATTENTION_ROWS_ACCUMULATOR_FRAGMENTS,
            "{cell}: {units} units exceed the register budget"
        );
        let bytes = row_tile_bytes(rows, groups, row_tiled_block(head_dim));
        assert!(bytes <= THREADGROUP_BUDGET, "{cell}: {bytes} bytes");
    }
    assert_eq!(
        row_tile_bytes(2, 8, 128),
        4 * 16 * (128 + 6),
        "a score block, three f32 and three i32 per query vector"
    );
}

#[test]
fn the_declared_arrays_stay_inside_the_budget_at_every_admitted_shape() {
    let mut shapes = 0_u32;
    for (groups, head_dim) in [(8_u64, 512_u64), (8, 256), (8, 128), (8, 64), (16, 256), (2, 64), (2, 128)] {
        let op = attention_rows_op(9, groups, head_dim, 512, 971, SLIDING_LOWER);
        let Some(CachedAttentionForm::TwoRangeRowTiled {
            rows_per_threadgroup: rows,
            simdgroups,
            ..
        }) = relaxed_form(&op)
        else {
            panic!("groups {groups} head_dim {head_dim}: not the row-tiled form");
        };
        let block = row_tiled_block(head_dim);
        let vectors = rows * groups;
        let declared = 4 * (vectors * block + 3 * vectors + 3 * vectors);
        assert_eq!(
            declared,
            row_tile_bytes(rows, groups, block),
            "groups {groups} head_dim {head_dim}: the sizing rule must count exactly what the kernel declares"
        );
        assert!(
            declared <= THREADGROUP_BUDGET,
            "groups {groups} head_dim {head_dim}: {declared} bytes declared"
        );
        let kernel = emit(&op, &PackedOperands::new(), NumericPolicy::llama_relaxed())
            .expect("the row-tiled kernel emits");
        for declaration in [
            format!("constexpr long tile_rows = {rows};"),
            format!("constexpr long block = {block};"),
            format!("constexpr long simdgroups = {simdgroups};"),
            format!("constexpr long split_keys = {};", crate::sized::ATTENTION_ROWS_KEYS_PER_SPLIT),
            "threadgroup float score_tile[tile_vectors * block];".to_string(),
            "threadgroup int vector_row[tile_vectors];".to_string(),
        ] {
            assert!(
                kernel.source.contains(&declaration),
                "groups {groups} head_dim {head_dim}: the kernel does not carry `{declaration}`"
            );
        }
        shapes += 1;
    }
    assert_eq!(shapes, 7);
}

#[test]
fn simdgroups_follow_the_reference_head_dim_rule_and_never_fall_below_two() {
    for (head_dim, simdgroups, block) in [
        (64_u64, 2_u64, 64_u64),
        (128, 2, 64),
        (256, 4, 128),
        (512, 8, 128),
        (1024, 8, 128),
        (8, 2, 64),
    ] {
        assert_eq!(
            row_tiled_simdgroups(head_dim),
            simdgroups,
            "head_dim {head_dim}"
        );
        assert_eq!(row_tiled_block(head_dim), block, "head_dim {head_dim}");
    }
}

/// `(rows, [(cached 32), (cached 512), (cached 2048)])`, each cell `(rows per
/// threadgroup, splits, threadgroups)` at one kv head.
type GridRow = (u64, [(u64, u64, u64); 3]);

const GLOBAL_GRID: [GridRow; 4] = [
    (2, [(1, 1, 2), (1, 9, 18), (2, 32, 32)]),
    (5, [(1, 1, 5), (1, 7, 35), (2, 11, 33)]),
    (17, [(1, 1, 17), (2, 4, 36), (2, 4, 36)]),
    (49, [(2, 2, 50), (2, 2, 50), (2, 2, 50)]),
];

const SLIDING_GRID: [GridRow; 4] = [
    (2, [(1, 1, 2), (1, 9, 18), (1, 32, 64)]),
    (5, [(1, 1, 5), (1, 9, 45), (2, 22, 66)]),
    (17, [(1, 1, 17), (2, 8, 72), (2, 8, 72)]),
    (49, [(1, 2, 98), (2, 3, 75), (2, 3, 75)]),
];

#[test]
fn row_tiled_splits_matches_the_grid_table() {
    assert_default_sizing();
    let mut cells = 0_u32;
    for (label, head_dim, grid, build) in [
        (
            "global",
            512_u64,
            &GLOBAL_GRID,
            global_op as fn(u64, u64) -> BoundOp,
        ),
        ("sliding", 256, &SLIDING_GRID, sliding_op),
    ] {
        for (rows, per_capacity) in grid {
            for (capacity, (rows_tile, splits, threadgroups)) in
                CAPACITIES.iter().zip(per_capacity)
            {
                let op = build(*capacity, *rows);
                let tiles = rows.div_ceil(*rows_tile);
                assert_eq!(
                    row_tiled_splits(capacity + rows, 1, tiles, row_tiled_simdgroups(head_dim)),
                    *splits,
                    "{label} rows {rows} cached {capacity}: splits"
                );
                assert_eq!(
                    relaxed_form(&op),
                    Some(CachedAttentionForm::TwoRangeRowTiled {
                        splits: *splits,
                        rows_per_threadgroup: *rows_tile,
                        simdgroups: row_tiled_simdgroups(head_dim),
                    }),
                    "{label} rows {rows} cached {capacity}: form"
                );
                assert_eq!(
                    row_tiled_threadgroups(&op.kind, *rows_tile, *splits),
                    *threadgroups,
                    "{label} rows {rows} cached {capacity}: threadgroups"
                );
                cells += 1;
            }
        }
    }
    assert_eq!(cells, 24, "2 layers x 4 row counts x 3 capacities");
}

/// `(query rows, rows per threadgroup, splits)` of eight kv heads at
/// a thousand cached keys: two simdgroups per threadgroup, so the target is 128
/// threadgroups and every chunk width lands on it exactly.
const EIGHT_KV_HEAD_CHUNKS: [(u64, u64, u64); 5] = [(8, 8, 16), (16, 8, 8), (32, 8, 4), (64, 8, 2), (128, 8, 1)];

#[test]
fn eight_kv_head_chunks_split_the_keys_until_the_target_threadgroups_are_filled() {
    assert_default_sizing();
    let simdgroups = row_tiled_simdgroups(64);
    assert_eq!(row_tiled_target_threadgroups(simdgroups), 128);
    for (rows, rows_tile, splits) in EIGHT_KV_HEAD_CHUNKS {
        let capacity = 1024 + rows;
        let tile = rows_per_threadgroup(rows, 8, 2, 64, capacity);
        let tiles = rows.div_ceil(tile);
        assert_eq!(tile, rows_tile, "rows {rows}: rows per threadgroup");
        assert_eq!(
            row_tiled_splits(capacity, 8, tiles, simdgroups),
            splits,
            "rows {rows}: splits"
        );
        assert_eq!(8 * tiles * splits, 128, "rows {rows}: threadgroups");
    }
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

#[test]
fn the_partial_dispatches_one_threadgroup_per_kv_head_row_tile_and_split() {
    assert_default_sizing();
    let mut cells = 0_u32;
    for (grid, build, width) in [
        (&GLOBAL_GRID, global_op as fn(u64, u64) -> BoundOp, 256_u64),
        (&SLIDING_GRID, sliding_op, 128),
    ] {
        for (rows, per_capacity) in grid {
            for (capacity, (_, _, threadgroups)) in CAPACITIES.iter().zip(per_capacity) {
                let op = build(*capacity, *rows);
                assert_eq!(
                    threadgroup_layout(&op),
                    (*threadgroups, width),
                    "rows {rows} cached {capacity}"
                );
                cells += 1;
            }
        }
    }
    assert_eq!(cells, 24);
}

#[test]
fn the_kernel_source_carries_the_tile_decode_the_band_and_the_interleaved_store() {
    assert_default_sizing();
    let mut sources = BTreeSet::new();
    let mut entries = BTreeSet::new();
    let mut emitted = 0_u32;

    for (label, build, groups_head_dim) in [
        ("global", global_op as fn(u64, u64) -> BoundOp, 512_u64),
        ("sliding", sliding_op, 256),
    ] {
        for rows in VERIFY_ROWS {
            for capacity in CAPACITIES {
                let op = build(capacity, rows);
                let kernel = emit(&op, &PackedOperands::new(), NumericPolicy::llama_relaxed())
                    .expect("the row-tiled kernel emits");
                let cell = format!("{label} rows {rows} cached {capacity}");

                assert!(kernel.entry.ends_with("_rt"), "{cell}: {}", kernel.entry);
                assert!(
                    !kernel.entry.contains(&format!("_q{rows}_"))
                        && !kernel.entry.contains(&format!("_c{capacity}_")),
                    "{cell}: the name carries neither the row count nor the bucket: {}",
                    kernel.entry
                );
                for required in [
                    "uint tgid [[threadgroup_position_in_grid]]",
                    "ushort simdgroup_slot [[simdgroup_index_in_threadgroup]]",
                    "long total_rows = u.total_elements / (kv_heads * query_groups);",
                    "long split = (long)tgid % splits;",
                    "long tile = tiles - 1L - ((long)tgid / splits) % tiles;",
                    "long kv_head = (long)tgid / (splits * tiles);",
                    "long live = (long)in8[0];",
                    "long first_key = max(0L, live + cached_lower + row0) & ~7L;",
                    "long new_blocks = (last_split && mma_end > new_start) ? (mma_end - new_start + block - 1L) / block : 0L;",
                    "long tail_steps = (last_split && new_end > total_aligned && new_first < total_rows) ? 1L : 0L;",
                    "long slice = ((((band + splits - 1L) / splits) + split_keys - 1L) / split_keys) * split_keys;",
                    "simdgroup_multiply_accumulate(scores[group][vector_block], query_even[step_index], key_even_tile[group][step_index], scores[group][vector_block]);",
                    "simdgroup_multiply_accumulate(scores[group][vector_block], query_odd[step_index], key_odd_tile[group][step_index], scores[group][vector_block]);",
                    "(key - live - query_row) >= cached_lower",
                    "relative <= new_upper && relative >= cached_lower",
                    "accumulated[slot][vector_block].thread_elements()[0] *= row_scale;",
                    "long stats_index = u.total_elements * head_dim * splits + (query_index * splits + split) * 2L;",
                    "long base = ((query_index * (head_dim / 4L) + (dimension >> 2)) * splits + split) * 4L + (dimension & 3L);",
                ] {
                    assert!(
                        kernel.source.contains(required),
                        "{cell}: missing `{required}`"
                    );
                }
                assert!(
                    !kernel.source.contains("u.cached_key_rows"),
                    "{cell}: the bucket extent is read from the ninth operand"
                );

                let tile_rows = rows_per_threadgroup(rows, 1, 8, groups_head_dim, capacity + rows);
                let declared = row_tile_bytes(tile_rows, 8, row_tiled_block(groups_head_dim));
                assert!(declared <= THREADGROUP_BUDGET, "{cell}: {declared} bytes");

                sources.insert((groups_head_dim, kernel.source));
                entries.insert((groups_head_dim, kernel.entry));
                emitted += 1;
            }
        }
    }
    assert_eq!(emitted, 24, "2 layers x 4 row counts x 3 capacities");
    assert_eq!(
        sources.len(),
        4,
        "a layer shape compiles one kernel per tile height (one and two rows) and serves every bucket"
    );
    assert_eq!(entries.len(), sources.len(), "one entry name per kernel text");
}

#[test]
fn the_split_binds_scratch_and_the_merge_runs_one_threadgroup_per_row_and_head() {
    assert_default_sizing();
    let policy = NumericPolicy::llama_relaxed();
    let mut cells = 0_u32;

    for (build, grid) in [
        (global_op as fn(u64, u64) -> BoundOp, &GLOBAL_GRID),
        (sliding_op, &SLIDING_GRID),
    ] {
        for (rows, per_capacity) in grid {
            for (capacity, (_, splits, _)) in CAPACITIES.iter().zip(per_capacity) {
                let op = build(*capacity, *rows);
                let cell = format!("rows {rows} cached {capacity}");
                let kernel = emit(&op, &PackedOperands::new(), policy).expect("emits");
                let merge = emit_cached_attention_merge(&op, policy).expect("the merge emits");

                if *splits > 1 {
                    assert!(kernel.bindings.contains(&Binding::Scratch), "{cell}");
                    assert!(
                        !kernel
                            .bindings
                            .iter()
                            .any(|binding| matches!(binding, Binding::Output(_))),
                        "{cell}"
                    );
                    let merge =
                        merge.unwrap_or_else(|| panic!("{cell}: splits {splits} needs a merge"));
                    let width = SIMD_WIDTH * crate::sized::ATTENTION_SPLIT_MAX;
                    assert_eq!(merge.grid.threadgroup_width, Some(width), "{cell}");
                    assert_eq!(merge.grid.threads / width, rows * 8, "{cell}");
                    assert_eq!(
                        cached_attention_live_splits(&op.kind, policy),
                        *splits,
                        "{cell}"
                    );
                } else {
                    assert!(
                        kernel.bindings.contains(&Binding::Output(op.node)),
                        "{cell}"
                    );
                    assert!(!kernel.bindings.contains(&Binding::Scratch), "{cell}");
                    assert!(
                        merge.is_none(),
                        "{cell}: one split writes the output directly"
                    );
                }
                cells += 1;
            }
        }
    }
    assert_eq!(cells, 24);
}

#[test]
fn a_launch_past_u32_threads_takes_the_flat_form_through_the_one_grid_decision() {
    let limit = u64::from(u32::MAX);
    let policy = NumericPolicy::llama_relaxed();
    let mut wide = sliding_op(512, 2);
    let BoundOpKind::CachedAttention { kv_heads, .. } = &mut wide.kind else {
        unreachable!("attention_op always builds a CachedAttention kind");
    };
    *kv_heads = 1 << 25;
    wide.extents = vec![2, 1 << 25, 8, 256];

    let kernel = emit(&wide, &PackedOperands::new(), policy).expect("the wide partial emits");
    let (_, shape) = kernel_dispatch_shape(&wide, &PackedOperands::new(), policy)
        .expect("the wide partial has a dispatch shape");

    assert!(kernel.grid.threads > limit, "{}", kernel.grid.threads);
    assert_eq!(kernel.grid, shape, "emit and the cache-hit shape disagree");
    let spec = kernel.grid.grid2d.expect("a wide partial dispatches flat");
    assert_eq!(spec.form, Grid2DForm::FlatThreadgroupIndex);
    assert_eq!(spec.threads_per_threadgroup_x, 128);
    assert!(kernel.source.contains("ulong tgid = wide_group_index;"));
    assert!(!kernel.source.contains("uint tgid [["));
}

#[test]
fn the_decode_split_baseline_bakes_the_new_key_count() {
    let op = attention_rows_op(9, 8, 256, 512, 5, SLIDING_LOWER);
    let source = render_cached_attention_decode_split(&op, "entry", None).expect("renders");
    assert!(source.contains("constexpr long new_key_rows = 5;"));

    let decode = attention_op(9, 8, 256, 512, 1, SLIDING_LOWER);
    let source = render_cached_attention_decode_split(&decode, "entry", None).expect("renders");
    assert!(source.contains("constexpr long new_key_rows = 1;"));
}

#[test]
fn the_row_tiled_kernel_declines_a_half_precision_op() {
    let mut op = sliding_op(512, 5);
    op.dtype = DType::Float16;
    let error = emit(&op, &PackedOperands::new(), NumericPolicy::llama_relaxed())
        .expect_err("the MMA fragments are f32");
    assert_eq!(
        error,
        EmitError::UnsupportedDType {
            node: op.node,
            dtype: DType::Float16
        }
    );
}

/// The kernel's band arithmetic, transcribed: the aligned first key of a
/// tile and each split's slice, a multiple of the split granule.
fn kernel_slices(
    live: i64,
    lower: i64,
    row0: i64,
    splits: i64,
    granule: i64,
) -> (i64, Vec<(i64, i64)>) {
    let first_key = (live + lower + row0).max(0) & !7;
    let band = (live - first_key).max(0);
    let slice = ((band + splits - 1) / splits + granule - 1) / granule * granule;
    let slices = (0..splits)
        .map(|split| {
            let start = first_key + split * slice;
            (start, (start + slice).min(live))
        })
        .collect();
    (first_key, slices)
}

/// Every cached and every new key a query row may attend to is scored by
/// exactly one split of the tile that owns the row. The model is the kernel's
/// own integer arithmetic (its expressions are pinned in
/// `the_kernel_source_carries_the_tile_decode_the_band_and_the_interleaved_store`).
#[test]
fn every_cached_and_new_key_is_scored_by_exactly_one_split() {
    let granule = i64::try_from(crate::sized::ATTENTION_ROWS_KEYS_PER_SPLIT).expect("small");
    let mut cases = 0_u64;

    for rows in 1..=64_i64 {
        for tile_rows in [1_i64, 2, 3, 8, 16, 24] {
            let tiles = (rows + tile_rows - 1) / tile_rows;
            let mut covered_rows = 0;
            for tile in 0..tiles {
                let row0 = tile * tile_rows;
                covered_rows += tile_rows.min(rows - row0);
            }
            assert_eq!(covered_rows, rows, "tiles partition the {rows} rows");
        }

        for live in (1..=2048_i64).step_by(7) {
            for lower in [-511_i64, -9_223_372_036_854_775_807] {
                for splits in 1..=32_i64 {
                    for row0 in [0_i64, rows - 1] {
                        let (first_key, slices) = kernel_slices(live, lower, row0, splits, granule);
                        let mut next = first_key;
                        for (start, end) in
                            slices.iter().copied().filter(|(start, end)| start < end)
                        {
                            assert_eq!(start, next, "live {live} splits {splits}: gap or overlap");
                            assert_eq!(start % 8, 0, "slice starts stay fragment-aligned");
                            next = end;
                        }
                        assert_eq!(
                            next,
                            live.max(first_key),
                            "live {live} lower {lower} splits {splits}: the slices end at the live bound"
                        );
                        for row in row0..rows.min(row0 + 3) {
                            let row_first = (live + lower + row).max(0);
                            assert!(
                                row_first >= first_key || live <= row_first,
                                "row {row}'s own window starts inside the tile's band"
                            );
                        }
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(
        cases,
        64 * 293 * 2 * 32 * 2,
        "the property must have run over the whole domain"
    );
    println!("every_cached_and_new_key_is_scored_by_exactly_one_split: {cases} cases");
}

/// The kernel's new-range schedule, transcribed: the fragment blocks from the
/// aligned first visible key to the last whole fragment, then the scalar tail.
/// Returns the keys visited as `(fragment keys, tail keys)`.
fn new_range_keys(rows: i64, row0: i64, rows_here: i64, lower: i64, upper: i64, block: i64) -> Vec<i64> {
    let total_aligned = rows & !7;
    let last_row = row0 + rows_here - 1;
    let new_first = (row0 + lower).max(0);
    let new_end = rows.min(last_row + upper + 1);
    let new_start = new_first & !7;
    let mma_end = new_end.min(total_aligned);
    let blocks = if mma_end > new_start {
        (mma_end - new_start + block - 1) / block
    } else {
        0
    };
    let mut keys = Vec::new();
    for step in 0..blocks {
        let key0 = new_start + step * block;
        keys.extend(key0..key0 + block.min(mma_end - key0));
    }
    if new_end > total_aligned && new_first < rows {
        keys.extend(total_aligned..rows);
    }
    keys
}

/// Every new key a row of the tile can see is visited exactly once, whatever
/// the row count, tile height, window and upper bound, so the skip of blocks
/// above the causal diagonal and below the window never drops a visible key and
/// the aligned fragments and the scalar tail never overlap.
#[test]
fn the_new_range_visits_every_key_a_row_of_the_tile_can_see_exactly_once() {
    let block = i64::try_from(crate::sized::ATTENTION_ROWS_KEYS_PER_BLOCK).expect("small");
    let mut cases = 0_u64;
    for rows in 1..=200_i64 {
        for tile_rows in [1_i64, 2, 3, 8, 16, 24] {
            for (lower, upper) in [(i64::MIN / 2, 0_i64), (-511, 0), (-39, 0), (-7, 0), (-3, 2)] {
                for row0 in (0..rows).step_by(tile_rows as usize) {
                    let rows_here = tile_rows.min(rows - row0);
                    let keys = new_range_keys(rows, row0, rows_here, lower, upper, block);
                    let mut sorted = keys.clone();
                    sorted.sort_unstable();
                    sorted.dedup();
                    assert_eq!(sorted.len(), keys.len(), "rows {rows} row0 {row0}: a key is visited twice");
                    for key in 0..rows {
                        let visible = (row0..row0 + rows_here)
                            .any(|row| key - row >= lower && key - row <= upper);
                        if visible {
                            assert!(
                                keys.contains(&key),
                                "rows {rows} tile {tile_rows} row0 {row0} lower {lower} upper {upper}: visible key {key} is skipped"
                            );
                        }
                    }
                    assert!(keys.iter().all(|key| (0..rows).contains(key)), "a key outside the new range");
                    cases += 1;
                }
            }
        }
    }
    assert!(cases > 20_000, "the property must have run over the whole domain: {cases}");
}

/// The K-row `CachedSoftmaxWeights` the recognizer binds for a sliding and global layer schedule at
/// `rows` verify rows: eight groups on one kv head, head_dim 256, with the
/// layouts `bind::softmax_weights_axes` collapses to (dense `stug`/`swug`
/// strides, and a value row broadcast across the groups).
fn softmax_weights_rows_op(rows: u64, cached_key_rows: u64) -> BoundOp {
    let layout = |strides: &[i64]| Layout {
        base: 0,
        strides: strides.into(),
    };
    BoundOp {
        node: NodeId(10),
        dtype: DType::Float32,
        extents: vec![rows, cached_key_rows, 1, 8],
        kind: BoundOpKind::CachedSoftmaxWeights {
            operands: vec![
                (NodeId(0), layout(&[cached_key_rows as i64 * 8, 8, 1]), None),
                (NodeId(1), layout(&[rows as i64 * 8, 8, 1]), None),
                (NodeId(2), layout(&[256, 0, 1]), None),
            ],
            cached_weight_sum: NodeId(11),
            new_weight_sum: NodeId(12),
            new_attended: NodeId(13),
            cached_key_rows,
            new_key_rows: rows,
            query_rows: rows,
            attention_rows: rows * 8,
            head_dim: 256,
        },
    }
}

#[cfg(feature = "metal-wide-cooperative-reduce")]
#[test]
fn the_k_row_softmax_weights_kernel_folds_the_new_range_in_key_order() {
    let mut emitted = 0_u32;
    let mut sources = BTreeSet::new();
    for rows in VERIFY_ROWS {
        for cached_key_rows in CAPACITIES {
            let op = softmax_weights_rows_op(rows, cached_key_rows);
            let kernel = emit(&op, &PackedOperands::new(), NumericPolicy::bit_exact())
                .expect("the K-row softmax weights kernel emits");
            let cell = format!("rows {rows} cached {cached_key_rows}");
            assert!(
                kernel
                    .entry
                    .contains(&format!("_a{}_q{rows}_d256", rows * 8)),
                "{cell}: the name carries the row split: {}",
                kernel.entry
            );
            let width = wide_cooperative_reduce_width(cached_key_rows);
            assert_eq!(kernel.grid.threadgroup_width, Some(width), "{cell}");
            assert_eq!(kernel.grid.threads, rows * 8 * width, "{cell}");
            for required in [
                "long query = row / 8;",
                "long group = row % 8;",
                &format!("for (long new_key = 0; new_key < {rows}; new_key++)"),
                &format!("out[(query * {cached_key_rows} + key) * 8 + group] = exp(step0);"),
                "if (local == 0u) {\n\t\tfloat accumulator2 = 0.0f;",
                "new_weight_sum[row] = accumulator2;",
                "new_attended[row * 256 + dim] = accumulator3;",
                "if (local == 0u) { cached_weight_sum[row] = reduced1; }",
            ] {
                assert!(
                    kernel.source.contains(required),
                    "{cell}: missing `{required}`"
                );
            }
            sources.insert(kernel.source);
            emitted += 1;
        }
    }
    assert_eq!(emitted, 12, "4 row counts x 3 capacities");
    assert_eq!(
        sources.len(),
        12,
        "every K and bucket is its own kernel text"
    );
}

#[test]
fn the_k_row_softmax_weights_kernel_rejects_a_layout_not_collapsed_to_the_row_axes() {
    let mut op = softmax_weights_rows_op(5, 512);
    let BoundOpKind::CachedSoftmaxWeights { operands, .. } = &mut op.kind else {
        unreachable!("softmax_weights_rows_op builds a CachedSoftmaxWeights kind");
    };
    operands[1].1.strides = vec![8, 1].into();

    let error = emit(&op, &PackedOperands::new(), NumericPolicy::bit_exact())
        .expect_err("a two-axis new-score layout cannot address the K-row kernel");
    assert_eq!(
        error,
        EmitError::CachedSoftmaxWeightsNotSupported { node: op.node }
    );
}

#[cfg(feature = "metal-wide-cooperative-reduce")]
#[test]
fn the_one_row_softmax_weights_kernel_is_unchanged_with_the_rows_feature_on() {
    use super::attn_golden_tests::{
        GOLDEN_PREFIX, golden_dir, softmax_weights_golden_cases, softmax_weights_rendered,
    };

    let cases = softmax_weights_golden_cases();
    assert_eq!(
        cases.len(),
        3,
        "the golden set must have run over every case"
    );
    for (name, op) in &cases {
        let path = golden_dir().join(format!("main_{name}.msl"));
        let golden = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("no recorded golden at {}: {error}", path.display()));
        assert_eq!(
            softmax_weights_rendered(op),
            golden,
            "{name}: {GOLDEN_PREFIX} the K=1 kernel text must be byte-identical to main's"
        );
    }
}

fn packed_operands_at(op: &BoundOp, indices: &[usize], codec: Codec) -> PackedOperands {
    indices
        .iter()
        .map(|&index| (op.operands()[index].0, codec))
        .collect()
}

#[test]
fn a_float16_cached_kv_makes_the_decode_split_read_half_pointers_for_the_cached_triple_only() {
    let op = attention_op(9, 8, 256, 512, 1, SLIDING_LOWER);
    let policy = NumericPolicy::llama_relaxed();
    let packed = packed_operands_at(&op, &[2, 3, 6], Codec::Float16);

    let half = emit(&op, &packed, policy).expect("a Float16 cached K/V emits on the decode split");
    let plain = emit(&op, &PackedOperands::new(), policy).expect("the plain decode split emits");

    for index in [2, 3, 6] {
        assert!(
            half.source
                .contains(&format!("device const half* in{index} [[buffer({index})]]")),
            "cached operand {index} must bind as half"
        );
    }
    for index in [0, 1, 4, 5, 7, 8] {
        assert!(
            !half
                .source
                .contains(&format!("device const half* in{index} [[buffer({index})]]")),
            "operand {index} keeps the op's own element type"
        );
    }
    assert!(half.source.contains("kr4_cached"));
    assert!(half.source.contains("if (cached) { value_row[step][slot]"));
    assert!(
        !plain.source.contains("device const half*"),
        "the plain decode split must bind no half pointer"
    );
    assert!(plain.source.contains("(cached ? in2 : in4) + kbase"));
}

#[test]
fn the_row_tiled_form_declines_a_float16_cached_kv_instead_of_reading_it_as_f32() {
    let op = sliding_op(512, 5);
    let packed = packed_operands_at(&op, &[2, 3, 6], Codec::Float16);

    let error = emit(&op, &packed, NumericPolicy::llama_relaxed())
        .expect_err("the MMA row-tiled kernel reads f32 K/V");

    assert!(matches!(
        error,
        EmitError::CachedAttentionKvCodecNotSupported { node, .. } if node == op.node
    ));
}

#[test]
fn a_partly_packed_or_misplaced_codec_is_declined_on_the_decode_split() {
    let op = attention_op(9, 8, 256, 512, 1, SLIDING_LOWER);
    let policy = NumericPolicy::llama_relaxed();

    for (label, indices, codec) in [
        ("only the K even plane", vec![2], Codec::Float16),
        ("the query plane", vec![0], Codec::Float16),
        ("the new V rows", vec![7], Codec::Float16),
        ("a quantized cached triple", vec![2, 3, 6], Codec::Q8_0),
    ] {
        let packed = packed_operands_at(&op, &indices, codec);
        let error = emit(&op, &packed, policy).expect_err(label);
        assert!(
            matches!(error, EmitError::CachedAttentionKvCodecNotSupported { .. }),
            "{label}: {error:?}"
        );
    }
}

fn row_tiled_source(op: &BoundOp) -> String {
    let relaxed = NumericPolicy::bit_exact().with_contraction(true).with_reassociation(true);
    emit(op, &PackedOperands::new(), relaxed)
        .expect("the row-tiled kernel emits")
        .source
}

fn prompt_1000_kernel() -> String {
    row_tiled_source(&attention_rows_op(9, 2, 64, 24, 1000, GLOBAL_LOWER))
}

#[test]
fn the_row_tiled_softmax_is_one_online_pass_per_vector_with_one_max_and_one_sum_reduction() {
    assert_default_sizing();
    let source = prompt_1000_kernel();

    assert_eq!(source.matches("simd_max(").count(), 1, "one block max reduction");
    assert_eq!(source.matches("simd_sum(").count(), 1, "one block sum reduction");
    assert_eq!(
        source.matches("exp(local_scores[item] - next_maximum)").count(),
        1,
        "the weights are written once, in f32, against the final block maximum"
    );
    for required in [
        "for (long vector = (long)simdgroup_slot; vector < tile_vectors; vector += simdgroups) {",
        "exp(previous_maximum - next_maximum)",
    ] {
        assert!(source.contains(required), "missing `{required}`");
    }
    assert!(
        !source.contains("vectors_per_simdgroup"),
        "the vector loop is not unrolled by turn: the unrolled form measured 122 us slower per op at the prompt shape"
    );
}

#[test]
fn the_row_tiled_kernel_stages_the_query_tile_once_and_keeps_its_four_barriers() {
    assert_default_sizing();
    let source = prompt_1000_kernel();

    for required in [
        "constexpr bool stage_query = true;",
        "constexpr long query_stage_stride = half_dim + 8L;",
        "threadgroup float query_stage[stage_query ? tile_blocks * 2L * 8L * query_stage_stride : 1L];",
        "simdgroup_load(query_even[step_index], stage_even, (ulong)query_stage_stride);",
        "constexpr long depth_unroll = ((half_dim / 8) % 2 == 0) ? 2 : 1;",
    ] {
        assert!(source.contains(required), "missing `{required}`");
    }
    assert_eq!(
        source.matches("threadgroup_barrier(").count(),
        4,
        "one after the setup and the staging, three per key block"
    );
    let setup_end = source
        .find("threadgroup_barrier(")
        .expect("the setup barrier is in the kernel");
    let staging = source
        .find("query_stage[(vector_block * 16L + stage_row)")
        .expect("the staging store is in the kernel");
    let block_loop = source
        .find("for (long step = 0L; step < cached_blocks")
        .expect("the key block loop is in the kernel");
    assert!(
        staging < setup_end && setup_end < block_loop,
        "staged before the first barrier, outside the block loop"
    );

    let static_bytes = row_tile_bytes(8, 2, row_tiled_block(64));
    let staged_bytes = query_stage_bytes(8, 2, 64);
    assert_eq!((static_bytes, staged_bytes), (4480, 5120));
    assert!(static_bytes + staged_bytes <= THREADGROUP_BUDGET);
}

#[test]
fn the_query_tile_is_staged_only_while_it_fits_the_threadgroup_budget_and_the_staged_byte_cap() {
    assert_default_sizing();
    let mut cells = 0_u32;
    for (groups, head_dim, rows, staged, staged_bytes) in [
        (2_u64, 64_u64, 1000_u64, true, 5_120_u64),
        (8, 256, 17, false, 17_408),
        (16, 256, 49, false, 17_408),
        (8, 512, 49, false, 0),
    ] {
        let op = attention_rows_op(9, groups, head_dim, 512, rows, GLOBAL_LOWER);
        let Some(CachedAttentionForm::TwoRangeRowTiled {
            rows_per_threadgroup: tile_rows,
            ..
        }) = relaxed_form(&op)
        else {
            panic!("groups {groups} head_dim {head_dim}: not the row-tiled form");
        };
        let block = row_tiled_block(head_dim);
        let decided = query_tile_staged(
            tile_rows,
            groups,
            head_dim,
            block,
            THREADGROUP_BUDGET,
            crate::sized::ATTENTION_ROWS_MAX_STAGED_QUERY_BYTES,
        );
        assert_eq!(decided, staged, "groups {groups} head_dim {head_dim} tile {tile_rows}");
        if staged_bytes != 0 {
            assert_eq!(query_stage_bytes(tile_rows, groups, head_dim), staged_bytes);
        }
        let source = row_tiled_source(&op);
        assert!(
            source.contains(&format!("constexpr bool stage_query = {staged};")),
            "groups {groups} head_dim {head_dim}: the emitted gate disagrees with the rule"
        );
        cells += 1;
    }
    assert_eq!(cells, 4);
}

#[test]
fn the_staged_byte_cap_at_the_threadgroup_budget_reproduces_the_fit_only_rule() {
    assert_eq!(crate::sized::ATTENTION_ROWS_MAX_STAGED_QUERY_BYTES, 8_192);
    for (groups, head_dim, rows, fits) in [
        (2_u64, 64_u64, 1000_u64, true),
        (8, 256, 17, true),
        (16, 256, 49, true),
        (8, 512, 49, false),
    ] {
        let op = attention_rows_op(9, groups, head_dim, 512, rows, GLOBAL_LOWER);
        let Some(CachedAttentionForm::TwoRangeRowTiled {
            rows_per_threadgroup: tile_rows,
            ..
        }) = relaxed_form(&op)
        else {
            panic!("groups {groups} head_dim {head_dim}: not the row-tiled form");
        };
        let uncapped = query_tile_staged(
            tile_rows,
            groups,
            head_dim,
            row_tiled_block(head_dim),
            THREADGROUP_BUDGET,
            THREADGROUP_BUDGET,
        );
        assert_eq!(uncapped, fits, "groups {groups} head_dim {head_dim} tile {tile_rows}");
    }
    assert!(!query_tile_staged(8, 2, 64, row_tiled_block(64), THREADGROUP_BUDGET, 0));
}

#[test]
fn the_runtime_toml_pins_the_staged_query_byte_cap_default() {
    let toml_text = include_str!("../../omega-runtime.toml");
    let section = toml_text
        .split("[attention_rows]")
        .nth(1)
        .and_then(|rest| rest.split("\n[selection]").next())
        .expect("the attention_rows section is in the sizing toml");
    assert!(section.contains("\nmax_staged_query_bytes = 8192\n"));
}

fn rendered_with_operands(op: &BoundOp, half_operands: bool) -> String {
    let Some(CachedAttentionForm::TwoRangeRowTiled {
        rows_per_threadgroup,
        simdgroups,
        ..
    }) = relaxed_form(op)
    else {
        panic!("not the row-tiled form");
    };
    cached_attention_row_tiled::render_cached_attention_row_tiled_with(
        op,
        "omega_cached_attention_probe",
        rows_per_threadgroup,
        simdgroups,
        half_operands,
        None,
        AttentionRowSchedule::legacy(),
    )
    .expect("the row-tiled kernel renders")
}

#[cfg(feature = "metal-attn-variants")]
#[test]
fn card_14_k_reuse_stages_each_k_fragment_and_preserves_row_masks() {
    let operation = attention_rows_op(9, 8, 256, 512, 971, SLIDING_LOWER);
    let packed_operands = PackedOperands::new();
    let policy = NumericPolicy::llama_relaxed();
    let legacy = emit_with_attention_variant(
        &operation,
        &packed_operands,
        policy,
        AttentionVariant {
            mma_precision: AttentionMmaPrecision::F32,
            ..AttentionVariant::default()
        },
    )
    .expect("the legacy K reuse source emits");
    let shared_k = emit_with_attention_variant(
        &operation,
        &packed_operands,
        policy,
        AttentionVariant {
            mma_precision: AttentionMmaPrecision::F32,
            kv_reuse: AttentionKvReuse::SharedK,
            ..AttentionVariant::default()
        },
    )
    .expect("the shared K source emits");
    let shared_k_f16 = emit_with_attention_variant(
        &operation,
        &packed_operands,
        policy,
        AttentionVariant {
            mma_precision: AttentionMmaPrecision::F16,
            kv_reuse: AttentionKvReuse::SharedK,
            ..AttentionVariant::default()
        },
    )
    .expect("the half-width shared K source emits");

    assert_ne!(legacy.entry, shared_k.entry);
    assert_ne!(legacy.source, shared_k.source);
    assert!(shared_k.source.contains("threadgroup float shared_key_even"));
    assert!(shared_k.source.contains("simdgroup_store(key_even_tile"));
    assert!(shared_k.source.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"));
    assert!(shared_k.source.contains("omega_load_shared_float"));
    assert_eq!(shared_k.source.matches("simdgroup_store(key_even_tile").count(), 1);
    assert_eq!(shared_k.source.matches("simdgroup_store(key_odd_tile").count(), 1);
    assert!(shared_k.source.contains(
        "int key_tile = (int)simdgroup_slot + group * (int)simdgroups;"
    ));
    assert!(shared_k.source.contains(
        "for (int key_tile = 0; key_tile < fragments; key_tile++)"
    ));
    assert!(shared_k.source.contains(
        "for (int vector_block = (int)simdgroup_slot; vector_block < (int)tile_blocks; vector_block += (int)simdgroups)"
    ));
    assert!(shared_k.source.contains("int vector_slot = vector_block / (int)simdgroups;"));
    assert!(shared_k.source.contains(
        "simdgroup_store(scores[key_tile][vector_slot], score_tile + vector_block * 8 * (int)block + key_tile * 8"
    ));
    assert!(shared_k.source.contains("relative <= new_upper && relative >= cached_lower"));
    assert!(legacy.source.contains("relative <= new_upper && relative >= cached_lower"));
    assert!(legacy.source.contains("if (false) {"));
    assert!(shared_k.source.contains("if (true) {"));
    assert!(shared_k_f16.source.contains("threadgroup half shared_key_even"));
    assert!(shared_k_f16.source.contains("omega_load_shared_half"));
    assert!(shared_k_f16.entry.ends_with("_mma_f16_kv_shared_k"));
}

#[cfg(feature = "metal-attn-variants")]
#[test]
fn card_14_k_reuse_declines_when_staging_exceeds_threadgroup_budget() {
    let operation = attention_rows_op(9, 8, 512, 512, 4, SLIDING_LOWER);
    let legacy = cached_attention_row_tiled::render_cached_attention_row_tiled_with(
        &operation,
        "omega_card_14_legacy",
        4,
        8,
        false,
        None,
        AttentionRowSchedule::legacy(),
    )
    .expect("the same tile fits before K staging");
    assert!(legacy.contains("shared_key_even[1]"));
    assert!(legacy.contains("if (false) {"));

    let error = cached_attention_row_tiled::render_cached_attention_row_tiled_with(
        &operation,
        "omega_card_14_shared_k",
        4,
        8,
        false,
        None,
        AttentionRowSchedule::shared_k(),
    )
    .expect_err("K staging is rejected when it exceeds threadgroup memory");
    assert!(matches!(
        error,
        EmitError::CachedAttentionKvReuseNotSupported {
            reason: "shared K staging exceeds the configured threadgroup-memory budget",
            ..
        }
    ));
}

#[cfg(feature = "metal-attn-variants")]
#[test]
fn card_15_v_reuse_stages_v_for_distinct_query_owners() {
    let operation = attention_rows_op(9, 2, 64, 512, 8, SLIDING_LOWER);
    let packed_operands = PackedOperands::new();
    let policy = NumericPolicy::llama_relaxed();
    let shared_k = emit_with_attention_variant(
        &operation,
        &packed_operands,
        policy,
        AttentionVariant {
            mma_precision: AttentionMmaPrecision::F32,
            kv_reuse: AttentionKvReuse::SharedK,
            ..AttentionVariant::default()
        },
    )
    .expect("the K-only control emits");
    let shared_kv = emit_with_attention_variant(
        &operation,
        &packed_operands,
        policy,
        AttentionVariant {
            mma_precision: AttentionMmaPrecision::F32,
            kv_reuse: AttentionKvReuse::SharedKv,
            ..AttentionVariant::default()
        },
    )
    .expect("the shared K/V source emits");
    let shared_kv_f16 = emit_with_attention_variant(
        &operation,
        &packed_operands,
        policy,
        AttentionVariant {
            mma_precision: AttentionMmaPrecision::F16,
            kv_reuse: AttentionKvReuse::SharedKv,
            ..AttentionVariant::default()
        },
    )
    .expect("the half-width shared K/V source emits");

    assert_ne!(shared_k.entry, shared_kv.entry);
    assert_ne!(shared_k.source, shared_kv.source);
    assert!(shared_kv.entry.ends_with("_mma_f32_kv_shared_kv"));
    assert!(shared_kv.source.contains("threadgroup float shared_value[512]"));
    assert!(shared_kv.source.contains("simdgroup_store(value_operand, shared_value"));
    assert!(shared_kv.source.contains("dimension_block % (int)simdgroups == (int)simdgroup_slot"));
    assert!(shared_kv.source.contains("omega_load_shared_float"));
    assert!(shared_kv.source.contains("weights[vector_slot]"));
    assert!(shared_kv.source.contains("accumulated[dimension_block][vector_slot]"));
    assert!(shared_kv.source.contains("row_sum[vector]"));
    assert!(shared_kv_f16.source.contains("threadgroup half shared_value[512]"));
    assert!(shared_kv_f16.source.contains("omega_load_shared_half"));

    let bf16 = cached_attention_row_tiled::render_cached_attention_row_tiled_with(
        &operation,
        "omega_card_15_bf16",
        8,
        2,
        false,
        Some(Codec::BFloat16),
        AttentionRowSchedule::shared_kv(),
    )
    .expect("BF16 cached V decodes before shared staging");
    let bf8 = cached_attention_row_tiled::render_cached_attention_row_tiled_with(
        &operation,
        "omega_card_15_bf8",
        8,
        2,
        true,
        Some(Codec::BFloat8),
        AttentionRowSchedule::shared_kv(),
    )
    .expect("BF8 cached V decodes before shared staging");
    assert!(bf16.contains("omega_bf16_load_matrix(value_float"));
    assert!(bf8.contains("omega_bf8_load_matrix(value_float"));
    assert!(bf16.contains("omega_zero_padded_value_rows(value_float"));
    assert!(bf8.contains("omega_zero_padded_value_rows(value_float"));
}

#[cfg(feature = "metal-attn-variants")]
#[test]
fn card_15_v_reuse_declines_when_persistent_accumulators_exceed_budget() {
    let operation = attention_rows_op(9, 8, 512, 512, 4, SLIDING_LOWER);
    let error = cached_attention_row_tiled::render_cached_attention_row_tiled_with(
        &operation,
        "omega_card_15_accumulator_decline",
        4,
        8,
        false,
        None,
        AttentionRowSchedule::shared_kv(),
    )
    .expect_err("the query-owned V path declines above the accumulator budget");
    assert!(matches!(
        error,
        EmitError::CachedAttentionKvReuseNotSupported {
            reason: "shared K/V query ownership exceeds the configured accumulator-fragment budget",
            ..
        }
    ));
}

#[test]
fn the_float_setting_multiplies_float_fragments_and_names_no_half_type() {
    for (groups, head_dim) in [(2_u64, 64_u64), (8, 256), (8, 512)] {
        let op = attention_rows_op(9, groups, head_dim, 512, 971, SLIDING_LOWER);
        let source = rendered_with_operands(&op, false);
        for declaration in [
            "simdgroup_float8x8 key_even_tile[key_tiles_per_group][depth_unroll];",
            "simdgroup_float8x8 key_odd_tile[key_tiles_per_group][depth_unroll];",
            "simdgroup_float8x8 query_even[depth_unroll];",
            "simdgroup_float8x8 weights[tile_blocks];",
            "simdgroup_float8x8 value = value_float;",
            "simdgroup_float8x8 scores[score_key_tiles][score_vectors_per_simdgroup];",
            "simdgroup_float8x8 accumulated[accumulator_dimensions][accumulator_vectors];",
        ] {
            assert!(
                source.contains(declaration),
                "head_dim {head_dim}: the float setting does not declare `{declaration}`"
            );
        }
        assert!(!source.contains("simdgroup_half8x8"), "head_dim {head_dim}");
    }
}

#[test]
fn the_half_setting_narrows_the_operands_and_keeps_scores_accumulators_and_softmax_float() {
    for (groups, head_dim) in [(2_u64, 64_u64), (8, 256), (8, 512)] {
        let op = attention_rows_op(9, groups, head_dim, 512, 971, SLIDING_LOWER);
        let source = rendered_with_operands(&op, true);
        for declaration in [
            "simdgroup_half8x8 key_even_tile[key_tiles_per_group][depth_unroll];",
            "simdgroup_half8x8 key_odd_tile[key_tiles_per_group][depth_unroll];",
            "simdgroup_half8x8 query_even[depth_unroll];",
            "simdgroup_half8x8 query_odd[depth_unroll];",
            "simdgroup_half8x8 weights[tile_blocks];",
            "simdgroup_half8x8 value = narrow_fragment(value_float);",
            "simdgroup_float8x8 scores[score_key_tiles][score_vectors_per_simdgroup];",
            "simdgroup_float8x8 accumulated[accumulator_dimensions][accumulator_vectors];",
            "threadgroup float score_tile[tile_vectors * block];",
            "float local_scores[block / 32];",
        ] {
            assert!(
                source.contains(declaration),
                "head_dim {head_dim}: the half setting does not carry `{declaration}`"
            );
        }
        assert!(source.contains("simdgroup_multiply_accumulate(scores[group][vector_block], query_even"));
        assert!(source.contains("simdgroup_multiply_accumulate(accumulated[slot][vector_block], weights[vector_block], value"));
    }
}

#[test]
fn the_two_settings_differ_only_by_the_operand_type_and_the_narrowing_overloads() {
    for (groups, head_dim) in [(2_u64, 64_u64), (8, 256), (8, 512)] {
        let op = attention_rows_op(9, groups, head_dim, 512, 971, SLIDING_LOWER);
        let float_source = rendered_with_operands(&op, false)
            .replace(cached_attention_row_tiled::SHARED_K_FLOAT_HELPER, "");
        let half_source = rendered_with_operands(&op, true);
        assert!(half_source.contains(cached_attention_row_tiled::HALF_OPERAND_HELPERS));
        let restored = half_source
            .replace(cached_attention_row_tiled::HALF_OPERAND_HELPERS, "")
            .replace("threadgroup half shared_key_even", "threadgroup float shared_key_even")
            .replace("threadgroup half shared_key_odd", "threadgroup float shared_key_odd")
            .replace("threadgroup half shared_value", "threadgroup float shared_value")
            .replace("omega_load_shared_half", "omega_load_shared_float")
            .replace("simdgroup_half8x8", "simdgroup_float8x8")
            .replace("narrow_fragment(value_float)", "value_float")
            .replace("simdgroup_half8x8 key_", "simdgroup_float8x8 key_")
            .replace("simdgroup_half8x8 shared_even", "simdgroup_float8x8 shared_even")
            .replace("simdgroup_half8x8 shared_odd", "simdgroup_float8x8 shared_odd")
            .replace("simdgroup_half8x8 query_", "simdgroup_float8x8 query_")
            .replace("simdgroup_half8x8 weights", "simdgroup_float8x8 weights")
            .replace("narrow_fragment(even_float)", "even_float")
            .replace("narrow_fragment(odd_float)", "odd_float")
            .replace(
                "simdgroup_half8x8 value = narrow_fragment(value_float);",
                "simdgroup_float8x8 value = value_float;",
            )
            .replace(
                "simdgroup_half8x8 value_operand = narrow_fragment(value_float);",
                "simdgroup_float8x8 value_operand = value_float;",
            );
        if restored != float_source {
            let mismatch = restored
                .lines()
                .zip(float_source.lines())
                .enumerate()
                .find(|(_, (restored_line, float_line))| restored_line != float_line);
            panic!("head_dim {head_dim}: first mismatch {mismatch:?}");
        }
    }
}

#[test]
fn the_default_sizing_renders_the_float_setting() {
    let op = attention_rows_op(9, 2, 64, 512, 1000, GLOBAL_LOWER);
    let kernel = emit(&op, &PackedOperands::new(), NumericPolicy::llama_relaxed())
        .expect("the row-tiled kernel emits");
    assert!(kernel.source.contains("simdgroup_float8x8 weights[tile_blocks];"));
    assert!(!kernel.source.contains("simdgroup_half8x8"));
}

#[cfg(feature = "metal-attn-variants")]
#[test]
fn card_16_tile_height_selects_rows16_and_declines_nonintegral_rows4() {
    let capture = include_str!("../../../proxima-tensor/specs/decode-prefill-parity/evidence/attn5/raw/probe/granite.out");
    assert!(capture.lines().any(|line| {
        line.contains("extents=[1000, 8, 2, 64]")
    }));

    let captured_shape = attention_rows_op(9, 2, 64, 512, 1000, SLIDING_LOWER);
    let packed = PackedOperands::new();
    let policy = NumericPolicy::llama_relaxed();
    let legacy = emit(&captured_shape, &packed, policy).expect("legacy shape emits");
    let rows16 = emit_with_attention_variant(
        &captured_shape,
        &packed,
        policy,
        AttentionVariant {
            tile_height: AttentionTileHeight::Rows16,
            ..AttentionVariant::default()
        },
    )
    .expect("captured shape admits rows16");
    assert_ne!(legacy.grid.threads, rows16.grid.threads);
    assert!(rows16.entry.contains("_r16_"));
    assert!(rows16.entry.contains("_tile_rows16"));

    let rows4_error = emit_with_attention_variant(
        &captured_shape,
        &packed,
        policy,
        AttentionVariant {
            tile_height: AttentionTileHeight::Rows4,
            ..AttentionVariant::default()
        },
    )
    .expect_err("rows4 cannot represent the two-group eight-row tile unit");
    assert!(matches!(
        rows4_error,
        EmitError::CachedAttentionTileHeightNotSupported {
            rows: 4,
            reason: "requested rows do not form whole query tile units",
            ..
        }
    ));
}

#[cfg(feature = "metal-attn-variants")]
#[test]
fn card_16_tile_height_declines_the_exact_threadgroup_memory_overflow() {
    let shape = attention_rows_op(9, 8, 64, 512, 16, SLIDING_LOWER);
    let required = row_tile_bytes(16, 8, row_tiled_block(64));
    assert_eq!(required, 35_840);
    assert_eq!(crate::sized::CACHED_ATTENTION_THREADGROUP_MEMORY_BYTES, 32_768);
    let error = emit_with_attention_variant(
        &shape,
        &PackedOperands::new(),
        NumericPolicy::llama_relaxed(),
        AttentionVariant {
            tile_height: AttentionTileHeight::Rows16,
            ..AttentionVariant::default()
        },
    )
    .expect_err("rows16 exceeds the threadgroup-memory budget");
    assert!(matches!(
        error,
        EmitError::CachedAttentionTileHeightNotSupported {
            rows: 16,
            reason: "requested tile exceeds the configured threadgroup-memory budget",
            ..
        }
    ));
}

#[cfg(feature = "metal-attn-variants")]
fn card_17_granite_attention(rows: u64) -> BoundOp {
    let mut op = attention_rows_op(9, 2, 64, 512, rows, SLIDING_LOWER);
    op.extents = vec![rows, 8, 2, 64];
    let BoundOpKind::CachedAttention { kv_heads, .. } = &mut op.kind else {
        panic!("attention_rows_op returns cached attention");
    };
    *kv_heads = 8;
    op
}

#[cfg(feature = "metal-attn-variants")]
#[test]
fn card_17_query_parallelism_assigns_each_granite_query_block_once() {
    let capture = include_str!("../../../proxima-tensor/specs/decode-prefill-parity/evidence/attn5/raw/probe/granite.out");
    assert!(capture.lines().any(|line| {
        line.contains("extents=[1000, 8, 2, 64]")
    }));
    let op = card_17_granite_attention(1000);
    let packed = PackedOperands::new();
    let policy = NumericPolicy::llama_relaxed();
    let legacy = emit_with_attention_variant(
        &op,
        &packed,
        policy,
        AttentionVariant {
            kv_reuse: AttentionKvReuse::SharedKv,
            tile_height: AttentionTileHeight::Rows8,
            ..AttentionVariant::default()
        },
    )
    .expect("the shared K/V legacy-row control emits");
    let parallel = emit_with_attention_variant(
        &op,
        &packed,
        policy,
        AttentionVariant {
            kv_reuse: AttentionKvReuse::SharedKv,
            tile_height: AttentionTileHeight::Rows8,
            query_parallelism: AttentionQueryParallelism::SimdgroupRows,
            ..AttentionVariant::default()
        },
    )
    .expect("the captured shape admits simdgroup row ownership");
    let tile_blocks = (8_u64 / 8) * 2;
    let simdgroups = row_tiled_simdgroups(64);
    let mut coverage = vec![0_u8; 1000 * 8 * 2];
    for kv_head in 0..8_u64 {
        for tile_start in (0..1000_u64).step_by(8) {
            for simdgroup_slot in 0..simdgroups {
                for vector_block in (simdgroup_slot..tile_blocks).step_by(simdgroups as usize) {
                    let query_group = vector_block % 2;
                    for row_within_block in 0..8_u64 {
                        let query_row = tile_start + vector_block / 2 * 8 + row_within_block;
                        let query_index =
                            ((kv_head * 1000 + query_row) * 2 + query_group) as usize;
                        coverage[query_index] += 1;
                    }
                }
            }
        }
    }
    assert!(coverage.iter().all(|count| *count == 1));
    assert_eq!(tile_blocks, simdgroups);
    assert!(parallel.source.contains("constexpr bool query_parallel_rows = true"));
    assert!(parallel.source.contains("constexpr bool query_owner_rows = true"));
    let staged_declarations = |source: &str| -> Vec<String> {
        source
            .lines()
            .filter(|line| line.contains("shared_key_even") || line.contains("shared_value["))
            .map(str::to_owned)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        staged_declarations(&legacy.source),
        staged_declarations(&parallel.source),
        "query ownership preserves the selected K/V staging declarations"
    );
    assert_ne!(legacy.entry, parallel.entry);
}

#[cfg(feature = "metal-attn-variants")]
#[test]
fn card_17_query_parallelism_declines_one_row_and_legacy_keeps_decode_split() {
    let op = attention_op(9, 2, 64, 32, 1, SLIDING_LOWER);
    let packed = PackedOperands::new();
    let policy = NumericPolicy::llama_relaxed();
    let legacy = emit(&op, &packed, policy).expect("one-row legacy decode emits");
    assert!(legacy.entry.ends_with("_ds"), "legacy decode entry: {}", legacy.entry);
    let error = emit_with_attention_variant(
        &op,
        &packed,
        policy,
        AttentionVariant {
            query_parallelism: AttentionQueryParallelism::SimdgroupRows,
            ..AttentionVariant::default()
        },
    )
    .expect_err("one query row cannot be split between simdgroups");
    assert!(matches!(
        error,
        EmitError::CachedAttentionQueryParallelismNotSupported {
            query_rows: 1,
            reason: "simdgroup row ownership requires at least two query rows",
            ..
        }
    ));
}

#[cfg(feature = "metal-attn-variants")]
#[test]
fn card_18_prefetch_stages_the_next_kv_block_in_f16_operands() {
    let op = card_17_granite_attention(1000);
    let parallel = emit_with_attention_variant(
        &op,
        &PackedOperands::new(),
        NumericPolicy::llama_relaxed(),
        AttentionVariant {
            kv_reuse: AttentionKvReuse::SharedKv,
            tile_height: AttentionTileHeight::Rows8,
            query_parallelism: AttentionQueryParallelism::SimdgroupRows,
            mma_precision: AttentionMmaPrecision::F16,
            prefetch: AttentionPrefetch::NextBlock,
            ..AttentionVariant::default()
        },
    )
    .expect("the F16 Granite-shaped K/V prefetch fits the threadgroup budget");
    assert!(parallel.entry.contains("_prefetch_next_block"));
    assert!(parallel.source.contains("constexpr bool prefetch_next_block = true"));
    assert!(parallel.source.contains("threadgroup half prefetched_key_even[2048]"));
    assert!(parallel.source.contains("threadgroup half prefetched_key_odd[2048]"));
    assert!(parallel.source.contains("threadgroup half prefetched_value[4096]"));
    assert!(parallel.source.contains("simdgroup_store(value_operand, prefetched_value"));
    assert!(!parallel.source.contains('@'));

    let f32_error = emit_with_attention_variant(
        &op,
        &PackedOperands::new(),
        NumericPolicy::llama_relaxed(),
        AttentionVariant {
            kv_reuse: AttentionKvReuse::SharedKv,
            tile_height: AttentionTileHeight::Rows8,
            query_parallelism: AttentionQueryParallelism::SimdgroupRows,
            mma_precision: AttentionMmaPrecision::F32,
            prefetch: AttentionPrefetch::NextBlock,
            ..AttentionVariant::default()
        },
    )
    .expect_err("F32 double-buffered K/V exceeds the threadgroup budget");
    assert!(matches!(
        f32_error,
        EmitError::CachedAttentionPrefetchNotSupported {
            reason: "double-buffered K/V staging exceeds the configured threadgroup-memory budget",
            ..
        }
    ));
}

#[cfg(feature = "metal-attn-variants")]
#[test]
fn card_18_prefetch_guards_a_partial_final_cached_block() {
    let op = attention_rows_op(9, 2, 64, 520, 1000, SLIDING_LOWER);
    let kernel = emit_with_attention_variant(
        &op,
        &PackedOperands::new(),
        NumericPolicy::llama_relaxed(),
        AttentionVariant {
            tile_height: AttentionTileHeight::Rows8,
            mma_precision: AttentionMmaPrecision::F16,
            prefetch: AttentionPrefetch::NextBlock,
            ..AttentionVariant::default()
        },
    )
    .expect("a partial final cached block remains admissible");
    assert!(kernel.source.contains("step + 1L < cached_blocks"));
    assert!(kernel.source.contains("long next_columns = min(block, slice_end - next_key0)"));
    assert!(kernel.source.contains("int next_fragments = (int)((next_columns + 7L) / 8L)"));
    assert!(kernel.source.contains("fragment_index < next_fragments * depth_fragments"));
    assert!(kernel.source.contains("next_key0 + (long)key_tile * 8L"));
}

#[cfg(feature = "metal-attn-variants")]
#[test]
fn card_19_simd_topology_assigns_each_granite_query_head_once() {
    let capture = include_str!("../../../proxima-tensor/specs/decode-prefill-parity/evidence/attn5/raw/probe/granite.out");
    assert!(capture.lines().any(|line| {
        line.contains("extents=[1000, 8, 2, 64]")
    }));
    let op = card_17_granite_attention(1000);
    let packed = PackedOperands::new();
    let policy = NumericPolicy::llama_relaxed();
    let per_head = emit_with_attention_variant(
        &op,
        &packed,
        policy,
        AttentionVariant {
            kv_reuse: AttentionKvReuse::SharedKv,
            tile_height: AttentionTileHeight::Rows16,
            simd_topology: AttentionSimdTopology::PerHead,
            ..AttentionVariant::default()
        },
    )
    .expect("per-head topology emits for Granite GQA");
    let grouped = emit_with_attention_variant(
        &op,
        &packed,
        policy,
        AttentionVariant {
            kv_reuse: AttentionKvReuse::SharedKv,
            tile_height: AttentionTileHeight::Rows16,
            simd_topology: AttentionSimdTopology::GroupedQueries,
            ..AttentionVariant::default()
        },
    )
    .expect("grouped-query topology emits for Granite GQA");
    assert!(per_head.source.contains("constexpr bool simd_per_head = true"));
    assert!(grouped.source.contains("constexpr bool simd_per_head = false"));
    assert!(per_head.source.contains("constexpr bool query_parallel_rows = false"));
    assert!(grouped.source.contains("constexpr bool query_parallel_rows = false"));
    assert!(per_head.source.contains("constexpr long tile_rows = 16;"));
    assert!(grouped.source.contains("constexpr long tile_rows = 16;"));
    assert!(per_head.source.contains("block_index % (tile_rows / 8L)"));
    assert!(grouped.source.contains("block_index / query_groups"));
    assert!(per_head.entry.contains("_simd_per_head"));
    assert!(grouped.entry.contains("_simd_grouped_queries"));

    let simdgroups = row_tiled_simdgroups(64);
    for per_head_topology in [false, true] {
        let mut coverage = vec![0_u8; 1000 * 8 * 2];
        for kv_head in 0..8_u64 {
            for tile_start in (0..1000_u64).step_by(16) {
                for simdgroup_slot in 0..simdgroups {
                    for vector_block in (simdgroup_slot..4).step_by(simdgroups as usize) {
                        let row_block = if per_head_topology {
                            vector_block % 2
                        } else {
                            vector_block / 2
                        };
                        let query_group = if per_head_topology {
                            vector_block / 2
                        } else {
                            vector_block % 2
                        };
                        let owned_from = tile_start + row_block * 8;
                        let mapped_start = owned_from.min(1000 - 8);
                        for row_within_block in 0..8_u64 {
                            let query_row = mapped_start + row_within_block;
                            if query_row >= owned_from && query_row < 1000 {
                                let index = ((kv_head * 1000 + query_row) * 2 + query_group) as usize;
                                coverage[index] += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(coverage.iter().all(|owners| *owners == 1));
    }
    assert_ne!(per_head.entry, grouped.entry);
}

#[cfg(feature = "metal-attn-variants")]
#[test]
fn card_19_simd_topology_declines_an_insufficient_head_width() {
    let op = attention_rows_op(9, 2, 48, 512, 1000, SLIDING_LOWER);
    let error = emit_with_attention_variant(
        &op,
        &PackedOperands::new(),
        NumericPolicy::llama_relaxed(),
        AttentionVariant {
            tile_height: AttentionTileHeight::Rows8,
            simd_topology: AttentionSimdTopology::PerHead,
            ..AttentionVariant::default()
        },
    )
    .expect_err("per-head lane mapping needs two 8-wide fragments per simdgroup");
    assert!(matches!(
        error,
        EmitError::CachedAttentionSimdTopologyNotSupported {
            reason: "per-head topology requires at least two 8-wide fragments per simdgroup",
            ..
        }
    ));
}
