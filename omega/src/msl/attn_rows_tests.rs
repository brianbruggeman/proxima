use std::collections::BTreeSet;

use super::attn_golden_tests::{attention_op, attention_rows_op};
use super::*;

const SLIDING_LOWER: i64 = -511;
const GLOBAL_LOWER: i64 = i64::MIN;
const VERIFY_ROWS: [u64; 4] = [2, 5, 17, 49];
const CAPACITIES: [u64; 3] = [32, 512, 2048];
const THREADGROUP_BUDGET: u64 = 32_768;

fn assert_default_sizing() {
    assert_eq!(crate::sized::ATTENTION_ROWS_KEYS_PER_BLOCK, 64);
    assert_eq!(crate::sized::ATTENTION_ROWS_HEAD_DIMS_PER_SIMDGROUP, 64);
    assert_eq!(crate::sized::ATTENTION_ROWS_MMA_MIN_QUERY_ROWS, 2);
    assert_eq!(crate::sized::ATTENTION_ROWS_MAX_QUERY_ROWS, 64);
    assert_eq!(crate::sized::ATTENTION_ROWS_TARGET_THREADGROUPS, 256);
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

#[test]
fn the_row_count_the_policy_and_the_head_group_choose_the_form() {
    assert_default_sizing();
    let tiled = Some(CachedAttentionForm::TwoRangeRowTiled {
        splits: 9,
        rows_per_threadgroup: 3,
        simdgroups: 4,
    });
    let decode = Some(CachedAttentionForm::TwoRangeDecodeSplit {
        splits: 17,
        chunks: 4,
    });
    let one_dispatch = Some(CachedAttentionForm::TwoRangeCachedBound);
    let mut cells = 0_u32;

    for rows in [1_u64, 2, 5, 17, 49, 64, 65] {
        for groups in [8_u64, 4] {
            let op = attention_rows_op(9, groups, 256, 512, rows, SLIDING_LOWER);
            let relaxed_expected = match (rows, groups) {
                (1, _) => decode,
                (2..=64, 8) => tiled,
                _ => one_dispatch,
            };
            assert_eq!(
                relaxed_form(&op),
                relaxed_expected,
                "relaxed, rows {rows}, groups {groups}"
            );
            assert_eq!(
                cached_attention_form(&op.kind, NumericPolicy::bit_exact()),
                one_dispatch,
                "bit_exact withholds both reassociations: rows {rows}, groups {groups}"
            );
            cells += 2;
        }
    }
    assert_eq!(cells, 28, "7 row counts x 2 group counts x 2 policies");
}

#[test]
fn shapes_the_tile_cannot_serve_stay_on_the_one_dispatch_form() {
    let one_dispatch = Some(CachedAttentionForm::TwoRangeCachedBound);
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
        "a group count that is not whole 8-row tiles",
        attention_rows_op(9, 12, 256, 512, 5, SLIDING_LOWER),
    ));
    misfits.push((
        "a row whose tile exceeds the threadgroup memory",
        attention_rows_op(9, 8, 4096, 512, 5, SLIDING_LOWER),
    ));

    let cells = misfits.len();
    for (label, op) in misfits {
        assert_eq!(relaxed_form(&op), one_dispatch, "{label}");
    }
    assert_eq!(cells, 7, "every misfit must have been classified");
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

#[test]
fn rows_per_threadgroup_fits_the_threadgroup_memory_budget() {
    assert_default_sizing();
    let expected = [
        (8_u64, 512_u64, 1_u64),
        (8, 256, 3),
        (8, 128, 5),
        (16, 256, 1),
    ];

    for (groups, head_dim, rows) in expected {
        assert_eq!(
            rows_per_threadgroup(groups, head_dim),
            rows,
            "groups {groups}, head_dim {head_dim}"
        );
        let bytes = row_tile_bytes(groups, head_dim);
        assert!(
            rows * bytes <= THREADGROUP_BUDGET,
            "{rows} rows of {bytes} bytes exceed the budget"
        );
        assert!(
            (rows + 1) * bytes > THREADGROUP_BUDGET,
            "one more row of {bytes} bytes would still fit, so the rule left memory unused"
        );
    }
    assert_eq!(row_tile_bytes(8, 512), 4 * 8 * (512 + 64 + 2));
    assert_eq!(row_tile_bytes(8, 256), 4 * 8 * (256 + 64 + 2));
}

#[test]
fn the_declared_arrays_stay_inside_the_budget_at_every_admitted_shape() {
    let mut shapes = 0_u32;
    for (groups, head_dim) in [(8_u64, 512_u64), (8, 256), (8, 128), (8, 64), (16, 256)] {
        let rows = rows_per_threadgroup(groups, head_dim);
        let block = crate::sized::ATTENTION_ROWS_KEYS_PER_BLOCK;
        let vectors = rows * groups;
        let declared = 4 * (vectors * head_dim + vectors * block + 2 * vectors);
        assert_eq!(
            declared,
            rows * row_tile_bytes(groups, head_dim),
            "groups {groups} head_dim {head_dim}: the sizing rule must count exactly what the kernel declares"
        );
        assert!(
            declared <= THREADGROUP_BUDGET,
            "groups {groups} head_dim {head_dim}: {declared} bytes declared"
        );
        shapes += 1;
    }
    assert_eq!(shapes, 5);
}

#[test]
fn simdgroups_follow_llamas_head_dim_rule() {
    for (head_dim, simdgroups) in [
        (64_u64, 1_u64),
        (128, 2),
        (256, 4),
        (512, 8),
        (1024, 8),
        (8, 1),
    ] {
        assert_eq!(
            row_tiled_simdgroups(head_dim),
            simdgroups,
            "head_dim {head_dim}"
        );
    }
}

/// `(rows, [(cached 32), (cached 512), (cached 2048)])`, each cell `(splits,
/// threadgroups)` at one kv head.
type GridRow = (u64, [(u64, u64); 3]);

const GLOBAL_GRID: [GridRow; 4] = [
    (2, [(1, 2), (9, 18), (32, 64)]),
    (5, [(1, 5), (9, 45), (32, 160)]),
    (17, [(1, 17), (9, 153), (16, 272)]),
    (49, [(2, 98), (6, 294), (6, 294)]),
];

const SLIDING_GRID: [GridRow; 4] = [
    (2, [(1, 1), (9, 9), (32, 32)]),
    (5, [(1, 2), (9, 18), (32, 64)]),
    (17, [(1, 6), (9, 54), (32, 192)]),
    (49, [(2, 34), (9, 153), (16, 272)]),
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
            for (capacity, (splits, threadgroups)) in CAPACITIES.iter().zip(per_capacity) {
                let op = build(*capacity, *rows);
                let rows_tile = rows_per_threadgroup(8, head_dim);
                let tiles = rows.div_ceil(rows_tile);
                assert_eq!(
                    row_tiled_splits(capacity + rows, 1, tiles),
                    *splits,
                    "{label} rows {rows} cached {capacity}: splits"
                );
                assert_eq!(
                    relaxed_form(&op),
                    Some(CachedAttentionForm::TwoRangeRowTiled {
                        splits: *splits,
                        rows_per_threadgroup: rows_tile,
                        simdgroups: row_tiled_simdgroups(head_dim),
                    }),
                    "{label} rows {rows} cached {capacity}: form"
                );
                assert_eq!(
                    row_tiled_threadgroups(&op.kind, rows_tile, *splits),
                    *threadgroups,
                    "{label} rows {rows} cached {capacity}: threadgroups"
                );
                cells += 1;
            }
        }
    }
    assert_eq!(cells, 24, "2 layers x 4 row counts x 3 capacities");
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
            for (capacity, (_, threadgroups)) in CAPACITIES.iter().zip(per_capacity) {
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
                    "long tile = ((long)tgid / splits) % tiles;",
                    "long kv_head = (long)tgid / (splits * tiles);",
                    "long live = (long)in8[0];",
                    "long first_key = max(0L, live + cached_lower + row0) & ~7L;",
                    "long new_blocks = (split == splits - 1L) ? (total_rows + block - 1L) / block : 0L;",
                    "simdgroup_multiply_accumulate(scores[vector_block], query_even, key_even, scores[vector_block]);",
                    "simdgroup_multiply_accumulate(scores[vector_block], query_odd, key_odd, scores[vector_block]);",
                    "(key - live - query_row) >= cached_lower",
                    "(key0 + column - query_row) <= new_upper",
                    "long stats_index = u.total_elements * head_dim * splits + (query_index * splits + split) * 2L;",
                    "attn_scratch[((query_index * (head_dim / 4L) + (dimension >> 2)) * splits + split) * 4L + (dimension & 3L)]",
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

                let tile_rows = rows_per_threadgroup(8, groups_head_dim);
                let vectors = tile_rows * 8;
                let block = crate::sized::ATTENTION_ROWS_KEYS_PER_BLOCK;
                let declared = 4 * (vectors * groups_head_dim + vectors * block + 2 * vectors);
                assert_eq!(declared, tile_rows * row_tile_bytes(8, groups_head_dim), "{cell}");
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
        2,
        "one compiled kernel per layer shape serves every K and every bucket"
    );
    assert_eq!(entries.len(), 2);
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
            for (capacity, (splits, _)) in CAPACITIES.iter().zip(per_capacity) {
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
    let source = render_cached_attention_decode_split(&op, "entry").expect("renders");
    assert!(source.contains("constexpr long new_key_rows = 5;"));

    let decode = attention_op(9, 8, 256, 512, 1, SLIDING_LOWER);
    let source = render_cached_attention_decode_split(&decode, "entry").expect("renders");
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
/// tile, each split's slice, and the blocks of the new range.
fn kernel_slices(
    live: i64,
    lower: i64,
    row0: i64,
    splits: i64,
    block: i64,
) -> (i64, Vec<(i64, i64)>) {
    let first_key = (live + lower + row0).max(0) & !7;
    let band = (live - first_key).max(0);
    let slice = ((band + splits - 1) / splits + block - 1) / block * block;
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
    let block = i64::try_from(crate::sized::ATTENTION_ROWS_KEYS_PER_BLOCK).expect("small");
    let mut cases = 0_u64;

    for rows in 1..=64_i64 {
        for tile_rows in [1_i64, 3] {
            let tiles = (rows + tile_rows - 1) / tile_rows;
            let mut covered_rows = 0;
            for tile in 0..tiles {
                let row0 = tile * tile_rows;
                covered_rows += tile_rows.min(rows - row0);
            }
            assert_eq!(covered_rows, rows, "tiles partition the {rows} rows");
        }
        let new_blocks = (rows + block - 1) / block;
        let mut covered_new = 0;
        for step in 0..new_blocks {
            let key0 = step * block;
            covered_new += block.min(rows - key0);
        }
        assert_eq!(covered_new, rows, "the new-range blocks tile [0, {rows})");

        for live in (1..=2048_i64).step_by(7) {
            for lower in [-511_i64, -9_223_372_036_854_775_807] {
                for splits in 1..=32_i64 {
                    for row0 in [0_i64, rows - 1] {
                        let (first_key, slices) = kernel_slices(live, lower, row0, splits, block);
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

/// The K-row `CachedSoftmaxWeights` the recognizer binds for gemma4-E2B at
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
