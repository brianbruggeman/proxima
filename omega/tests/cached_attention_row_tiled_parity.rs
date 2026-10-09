//! Metal-vs-CPU parity for the row-tiled cached attention form
//! (`CachedAttentionForm::TwoRangeRowTiled`: `omega/src/msl/
//! cached_attention_row_tiled.rs` plus the interleaved merge in
//! `cached_attention_render.rs`). Internal consistency only: the CPU reference
//! is this repo's own unfused chain run on the CPU interpreter, never
//! llama.cpp's; the llama.cpp greedy-id gate is
//! `proxima-model-interop/tests/gemma4_attn_split_decode_oracle.rs`.
//!
//! The program is the gemma4-shaped two-layer two-range cached forward
//! (`gemma4_rows_support`): a sliding layer (head_dim 256, window 512) and a
//! global layer (head_dim 512), eight query heads on one kv head. It is swept
//! over `K in {2, 5, 17, 49}` new rows and the cached lengths
//! `{31, 33, 199, 511, 1099, 2047}` (capacity buckets of 32 to 2048, so every
//! split regime from one direct-output split to the capped 32). 48 cells.
//!
//! Every cell first asserts both attention ops of the bound program take the
//! row-tiled form (entry name `_rt`), so a pass cannot be two copies of the
//! one-dispatch kernel agreeing with each other.
//!
//! Degenerate control: the same bound program with `new_upper_inclusive = 1`
//! forced into every attention op, evaluated on the CPU, must DISAGREE with the
//! Metal result in every cell. Letting each row see one future key moves its
//! output for any `K >= 2`, so a comparison that cannot tell the two masks
//! apart is not measuring the mask at all.
//!
//! Needs a Metal device. The data is generated from the program's own shapes
//! with real rotary angles; checkpoint activations are covered by the
//! model-loading census and oracle gates in `proxima-model-interop`.

#![cfg(all(
    feature = "metal",
    feature = "metal-attn-split-rows",
    target_os = "macos"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use core::pin::pin;
use core::task::{Context, Poll, Waker};

use half::bf16;
use proxima_primitives::pipe::Pipe;
use proxima_tensor::bind::{READY_BATCH_CAPACITY, ReadyBatch};
use proxima_tensor::cpu::QuantizedBlock;
use proxima_tensor::{
    BFloat8, BoundOp, BoundOpKind, Interpreter, NodeId, NumericPolicy, Op, bind_with_fusion,
    block_node_ids, infer,
};

mod gemma4_rows_support;
mod support;
use gemma4_rows_support::{Fixture, Geometry, bucket_for, fixture_with};
use support::{as_named_blocks, production_numeric_policy};

const VERIFY_ROWS: [usize; 4] = [2, 5, 17, 49];
const CACHED_LENGTHS: [usize; 6] = [31, 33, 199, 511, 1099, 2047];
const TOLERANCE: f32 = 1e-4;

fn relative_difference(expected: &[f32], actual: &[f32]) -> f32 {
    assert_eq!(expected.len(), actual.len());
    assert!(!expected.is_empty(), "an empty root compares nothing");
    let magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let difference = expected
        .iter()
        .zip(actual)
        .map(|(want, got)| (want - got).abs())
        .fold(0.0f32, f32::max);
    difference / magnitude.max(f32::MIN_POSITIVE)
}

fn bound_attention(fixture: &Fixture, policy: NumericPolicy) -> Vec<BoundOp> {
    let shapes = infer(&fixture.program, &fixture.symbols).expect("the program infers");
    bind_with_fusion(&fixture.program, &shapes, &[fixture.logits], true, policy)
        .expect("the program binds")
}

/// `resolved` run on the CPU interpreter over the fixture's named inputs.
fn run_resolved_on_cpu(fixture: &Fixture, resolved: &[BoundOp]) -> Vec<f32> {
    run_resolved_root_on_cpu(fixture, resolved, fixture.logits)
}

fn run_resolved_root_on_cpu(fixture: &Fixture, resolved: &[BoundOp], root: NodeId) -> Vec<f32> {
    let mut buffers: Vec<Option<Vec<f32>>> = vec![None; fixture.program.len()];
    for node in block_node_ids(&fixture.program) {
        let Op::Input {
            name: Some(name), ..
        } = &fixture.program[node.0 as usize]
        else {
            unreachable!("block_node_ids only returns named inputs");
        };
        let data = fixture
            .named
            .iter()
            .find(|(candidate, _)| candidate == name)
            .unwrap_or_else(|| panic!("missing named input {name}"))
            .1
            .clone();
        buffers[node.0 as usize] = Some(data);
    }
    let interpreter = Interpreter::new(&mut buffers);
    for chunk in resolved.chunks(READY_BATCH_CAPACITY) {
        let batch: ReadyBatch = chunk.iter().cloned().collect();
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        let mut future = pin!(interpreter.call(batch));
        match future.as_mut().poll(&mut context) {
            Poll::Ready(result) => result.expect("resolved batch computes"),
            Poll::Pending => unreachable!("cpu pipes never yield: no internal .await"),
        }
    }
    buffers[root.0 as usize]
        .clone()
        .expect("the selected cpu root was computed")
}

fn one_cell(label: &str, geometry: Geometry, rows: usize, cached_len: usize) -> (f32, f32) {
    let policy = production_numeric_policy();
    let fixture = fixture_with(rows, cached_len, geometry);
    let named = as_named_blocks(&fixture.named);
    let roots = [fixture.logits];
    let cell = format!(
        "{label}: rows {rows} cached_len {cached_len} bucket {}",
        bucket_for(cached_len)
    );

    let resolved = bound_attention(&fixture, policy);
    let attention: Vec<&BoundOp> = resolved
        .iter()
        .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .collect();
    assert_eq!(
        attention.len(),
        2,
        "{cell}: the sliding and the global layer must both fuse"
    );
    for bound in &attention {
        let kernel = omega::emit(bound, &omega::PackedOperands::new(), policy)
            .expect("the bound attention op emits");
        assert!(
            kernel.entry.ends_with("_rt"),
            "{cell}: the op must take the row-tiled form, got {}",
            kernel.entry
        );
    }

    let shapes = infer(&fixture.program, &fixture.symbols).expect("the program infers");
    let unfused = bind_with_fusion(&fixture.program, &shapes, &roots, false, policy)
        .expect("the unfused program binds");
    let cpu = run_resolved_on_cpu(&fixture, &unfused);
    let plan = omega::plan_named(&fixture.program, &fixture.symbols, &named, &roots, policy)
        .expect("metal plans the program");
    let metal = omega::execute_plan_named(&plan, &named).expect("metal runs the program");
    let relative = relative_difference(&cpu, metal.root());

    let mut wrong_mask = resolved.clone();
    let mut forced = 0_usize;
    for bound in &mut wrong_mask {
        if let BoundOpKind::CachedAttention {
            new_upper_inclusive,
            ..
        } = &mut bound.kind
        {
            *new_upper_inclusive = 1;
            forced += 1;
        }
    }
    assert_eq!(
        forced, 2,
        "{cell}: the control must reach both attention ops"
    );
    let control = run_resolved_on_cpu(&fixture, &wrong_mask);
    let control_relative = relative_difference(&control, metal.root());

    eprintln!(
        "row_tiled parity: {cell} relative={relative} wrong_mask_relative={control_relative}"
    );
    (relative, control_relative)
}

#[test]
fn the_row_tiled_kernels_hold_parity_with_the_cpu_evaluator_across_rows_and_cached_lengths() {
    let mut cells = 0_usize;
    for rows in VERIFY_ROWS {
        for cached_len in CACHED_LENGTHS {
            let (relative, control_relative) = one_cell(
                "one kv head verify",
                Geometry::ONE_KV_HEAD,
                rows,
                cached_len,
            );
            assert!(
                relative < TOLERANCE,
                "rows {rows} cached_len {cached_len}: metal disagrees with cpu: relative={relative}"
            );
            assert!(
                control_relative > 10.0 * TOLERANCE,
                "rows {rows} cached_len {cached_len}: the forced wrong mask must disagree with metal \
                 (relative={control_relative}), or this comparison cannot see the mask"
            );
            cells += 1;
        }
    }
    assert_eq!(
        cells, 24,
        "4 row counts x 6 cached lengths, each program carrying both layers: 48 layer cells"
    );
}

/// Prefill-width cells: rows past the old 64-row limit, a window the new range
/// itself crosses, a ragged tail of new keys past the last whole fragment, and
/// a query-group size of two, whose fragments are eight rows of one head.
#[test]
fn the_row_tiled_kernels_hold_parity_with_the_cpu_evaluator_at_prefill_widths() {
    let one_kv_head = Geometry::ONE_KV_HEAD;
    let two_groups = Geometry::TWO_QUERY_GROUPS;
    let cells = [
        ("one kv head past the old row limit", one_kv_head, 70, 33),
        (
            "one kv head window crosses the new range",
            one_kv_head.with_window(40),
            130,
            199,
        ),
        (
            "one kv head window crosses, ragged tail",
            one_kv_head.with_window(48),
            101,
            1,
        ),
        ("one kv head whole prompt", one_kv_head, 600, 33),
        ("two groups eight rows", two_groups, 8, 33),
        ("two groups ragged rows", two_groups, 37, 100),
        (
            "two groups window crosses the new range",
            two_groups.with_window(24),
            50,
            40,
        ),
        ("two groups whole prompt", two_groups, 600, 33),
    ];
    for (label, geometry, rows, cached_len) in cells {
        let (relative, control_relative) = one_cell(label, geometry, rows, cached_len);
        assert!(
            relative < TOLERANCE,
            "{label} rows {rows} cached_len {cached_len}: metal disagrees with cpu: relative={relative}"
        );
        assert!(
            control_relative > 10.0 * TOLERANCE,
            "{label} rows {rows} cached_len {cached_len}: the forced wrong mask must disagree with \
             metal (relative={control_relative}), or this comparison cannot see the mask"
        );
    }
}

/// Fewer rows than one fragment of a two-group head keep the one-dispatch
/// kernel, held to the same CPU reference with the entry-name check that it is
/// not the row-tiled kernel.
#[test]
fn rows_under_one_fragment_of_a_two_group_head_hold_one_dispatch_parity_with_the_cpu_evaluator() {
    let policy = production_numeric_policy();
    let rows = 5;
    let mut cells = 0_usize;
    for cached_len in [33_usize, 511] {
        let fixture = fixture_with(rows, cached_len, Geometry::TWO_QUERY_GROUPS);
        let named = as_named_blocks(&fixture.named);
        let roots = [fixture.logits];
        let cell = format!("rows {rows} cached_len {cached_len}");

        for bound in bound_attention(&fixture, policy)
            .iter()
            .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        {
            let kernel = omega::emit(bound, &omega::PackedOperands::new(), policy)
                .expect("the bound attention op emits");
            assert!(!kernel.entry.ends_with("_rt"), "{cell}: {}", kernel.entry);
        }

        let shapes = infer(&fixture.program, &fixture.symbols).expect("the program infers");
        let unfused = bind_with_fusion(&fixture.program, &shapes, &roots, false, policy)
            .expect("the unfused program binds");
        let cpu = run_resolved_on_cpu(&fixture, &unfused);
        let plan = omega::plan_named(&fixture.program, &fixture.symbols, &named, &roots, policy)
            .expect("metal plans the program");
        let metal = omega::execute_plan_named(&plan, &named).expect("metal runs the program");
        let relative = relative_difference(&cpu, metal.root());
        assert!(relative < TOLERANCE, "{cell}: relative={relative}");
        cells += 1;
    }
    assert_eq!(cells, 2);
}

fn bf16_cache_payload(fixture: &mut Fixture) -> Vec<(String, Vec<u8>)> {
    let shapes = infer(&fixture.program, &fixture.symbols).expect("the fixture shapes infer");
    for (name, values) in &mut fixture.named {
        if name.contains(".attn_k.weight") || name.contains(".attn_v.weight") {
            values.fill(0.0);
        }
    }
    fixture
        .named
        .iter_mut()
        .filter(|(name, _)| name.starts_with("kv_cache."))
        .map(|(name, values)| {
            let input_index = fixture
                .program
                .iter()
                .enumerate()
                .find_map(|(index, op)| match op {
                    Op::Input {
                        name: Some(input_name),
                        ..
                    } if input_name == name => Some(index),
                    _ => None,
                })
                .expect("each cache block maps to a named input");
            let cache_row_width: usize = shapes
                .of(NodeId(input_index as u32))
                .iter()
                .skip(1)
                .map(|extent| *extent as usize)
                .product();
            for (index, value) in values.iter_mut().enumerate() {
                let row = index / cache_row_width;
                let dimension = index % cache_row_width;
                *value = if name.ends_with(".v") {
                    [0.0, 1.0, -1.0, 0.5][(row + dimension) % 4]
                } else {
                    [0.25, -0.5, 0.75, -1.0][(row + dimension) % 4]
                };
            }
            assert!(
                values
                    .iter()
                    .all(|value| bf16::from_f32(*value).to_f32() == *value)
            );
            let bytes = values
                .iter()
                .flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes())
                .collect();
            (name.clone(), bytes)
        })
        .collect()
}

#[test]
fn card_10_bf16_row_matches_f32_cache_output_bits_with_f32_mma() {
    let policy = production_numeric_policy();
    let mut fixture = fixture_with(8, 31, Geometry::ONE_KV_HEAD);
    let packed_cache = bf16_cache_payload(&mut fixture);
    assert_eq!(
        packed_cache.len(),
        3 * 2,
        "two layers each carry K even, K odd and V"
    );
    let attention = bound_attention(&fixture, policy)
        .into_iter()
        .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .collect::<Vec<_>>();
    let root = attention
        .last()
        .expect("the fixture includes a global attention operation")
        .node;
    let roots = [root];
    let f32_named = as_named_blocks(&fixture.named);
    let bf16_named: Vec<(&str, QuantizedBlock<'_>)> = fixture
        .named
        .iter()
        .map(|(name, values)| {
            match packed_cache
                .iter()
                .find(|(packed_name, _)| packed_name == name)
            {
                Some((_, bytes)) => (
                    name.as_str(),
                    QuantizedBlock::Packed {
                        codec: omega::Codec::BFloat16,
                        bytes,
                    },
                ),
                None => (name.as_str(), QuantizedBlock::Float32(values)),
            }
        })
        .collect();
    let f32_plan = omega::plan_named(
        &fixture.program,
        &fixture.symbols,
        &f32_named,
        &roots,
        policy,
    )
    .expect("the f32 row-tiled cache plans");
    let bf16_plan = omega::plan_named(
        &fixture.program,
        &fixture.symbols,
        &bf16_named,
        &roots,
        policy,
    )
    .expect("the BF16 row-tiled cache plans");
    assert_ne!(
        f32_plan
            .kernel_keys()
            .expect("the F32 plan keys are collected"),
        bf16_plan
            .kernel_keys()
            .expect("the BF16 plan keys are collected"),
        "the BF16 plan must execute its codec-specific row shader"
    );
    let f32_output =
        omega::execute_plan_named(&f32_plan, &f32_named).expect("the f32 row-tiled cache executes");
    let bf16_output = omega::execute_plan_named(&bf16_plan, &bf16_named)
        .expect("the BF16 row-tiled cache executes");
    let f32_bits: Vec<u32> = f32_output
        .root()
        .iter()
        .map(|value| value.to_bits())
        .collect();
    let bf16_bits: Vec<u32> = bf16_output
        .root()
        .iter()
        .map(|value| value.to_bits())
        .collect();
    assert_eq!(
        bf16_bits, f32_bits,
        "BF16-exact K/V values preserve output bits"
    );
}

#[test]
fn card_10_bf16_row_selects_the_bf16_source_and_declines_mixed_cache_codecs() {
    let policy = production_numeric_policy();
    let fixture = fixture_with(8, 31, Geometry::ONE_KV_HEAD);
    let resolved = bound_attention(&fixture, policy);
    let attention: Vec<&BoundOp> = resolved
        .iter()
        .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .collect();
    assert_eq!(
        attention.len(),
        2,
        "the fixture has sliding and global attention"
    );
    for bound in attention {
        let packed = [2, 3, 6]
            .into_iter()
            .map(|index| (bound.operands()[index].0, omega::Codec::BFloat16))
            .collect::<omega::PackedOperands>();
        let emitted = omega::emit(bound, &packed, policy).expect("BF16 row source emits");
        assert!(
            emitted.entry.ends_with("_rt"),
            "row-tiled entry: {}",
            emitted.entry
        );
        assert!(emitted.source.contains("device const ushort* in2"));
        assert!(emitted.source.contains("omega_bf16_load_matrix(even_float"));
        assert!(
            emitted
                .source
                .contains("key_even_tile[group][step_index] = even_float;")
        );
        assert!(emitted.source.contains("simdgroup_float8x8"));
        assert!(!emitted.source.contains("simdgroup_half8x8"));

        let mixed = [
            (bound.operands()[2].0, omega::Codec::BFloat16),
            (bound.operands()[3].0, omega::Codec::BFloat16),
            (bound.operands()[6].0, omega::Codec::Float16),
        ]
        .into_iter()
        .collect::<omega::PackedOperands>();
        let error = omega::emit(bound, &mixed, policy).expect_err("mixed cached codecs decline");
        assert!(matches!(
            error,
            omega::EmitError::CachedAttentionKvCodecNotSupported { .. }
        ));
    }
}

fn bf8_cache_payload(fixture: &mut Fixture) -> Vec<(String, Vec<u8>)> {
    let shapes = infer(&fixture.program, &fixture.symbols).expect("the fixture shapes infer");
    for (name, values) in &mut fixture.named {
        if name.contains(".attn_k.weight") || name.contains(".attn_v.weight") {
            values.fill(0.0);
        }
    }
    fixture
        .named
        .iter_mut()
        .filter(|(name, _)| name.starts_with("kv_cache."))
        .map(|(name, values)| {
            let input_index = fixture
                .program
                .iter()
                .enumerate()
                .find_map(|(index, op)| match op {
                    Op::Input {
                        name: Some(input_name),
                        ..
                    } if input_name == name => Some(index),
                    _ => None,
                })
                .expect("each cache block maps to a named input");
            let cache_row_width: usize = shapes
                .of(NodeId(input_index as u32))
                .iter()
                .skip(1)
                .map(|extent| *extent as usize)
                .product();
            for (index, value) in values.iter_mut().enumerate() {
                let row = index / cache_row_width;
                let dimension = index % cache_row_width;
                *value = if row == 31 {
                    57_344.0
                } else if name.ends_with(".v") {
                    [0.0, 1.0, -1.0, 0.5][(row + dimension) % 4]
                } else {
                    [0.25, -0.5, 0.75, -1.0][(row + dimension) % 4]
                };
                assert_eq!(
                    BFloat8::from_f32(*value).to_f32(),
                    *value,
                    "row {row} dimension {dimension} is exactly BF8"
                );
            }
            let bytes = values
                .iter()
                .map(|value| BFloat8::from_f32(*value).to_bits())
                .collect();
            (name.clone(), bytes)
        })
        .collect()
}

#[test]
fn card_11_bf8_row_matches_f32_cache_output_bits_and_masks_the_padded_row() {
    let policy = production_numeric_policy();
    let mut fixture = fixture_with(8, 31, Geometry::ONE_KV_HEAD);
    let packed_cache = bf8_cache_payload(&mut fixture);
    assert_eq!(
        packed_cache.len(),
        3 * 2,
        "two layers each carry K even, K odd and V"
    );
    let attention = bound_attention(&fixture, policy)
        .into_iter()
        .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .collect::<Vec<_>>();
    let root = attention
        .last()
        .expect("the fixture includes a global attention operation")
        .node;
    let roots = [root];
    let f32_named = as_named_blocks(&fixture.named);
    let bf8_named: Vec<(&str, QuantizedBlock<'_>)> = fixture
        .named
        .iter()
        .map(|(name, values)| {
            match packed_cache
                .iter()
                .find(|(packed_name, _)| packed_name == name)
            {
                Some((_, bytes)) => (
                    name.as_str(),
                    QuantizedBlock::Packed {
                        codec: omega::Codec::BFloat8,
                        bytes,
                    },
                ),
                None => (name.as_str(), QuantizedBlock::Float32(values)),
            }
        })
        .collect();
    let f32_plan = omega::plan_named(
        &fixture.program,
        &fixture.symbols,
        &f32_named,
        &roots,
        policy,
    )
    .expect("the F32 row-tiled cache plans");
    let bf8_plan = omega::plan_named(
        &fixture.program,
        &fixture.symbols,
        &bf8_named,
        &roots,
        policy,
    )
    .expect("the BF8 row-tiled cache plans");
    assert_ne!(
        f32_plan
            .kernel_keys()
            .expect("the F32 plan keys are collected"),
        bf8_plan
            .kernel_keys()
            .expect("the BF8 plan keys are collected"),
        "the BF8 plan must execute its codec-specific row shader"
    );
    let f32_output =
        omega::execute_plan_named(&f32_plan, &f32_named).expect("the F32 row-tiled cache executes");
    let bf8_output =
        omega::execute_plan_named(&bf8_plan, &bf8_named).expect("the BF8 row-tiled cache executes");
    let f32_bits: Vec<u32> = f32_output
        .root()
        .iter()
        .map(|value| value.to_bits())
        .collect();
    let bf8_bits: Vec<u32> = bf8_output
        .root()
        .iter()
        .map(|value| value.to_bits())
        .collect();
    assert_eq!(
        bf8_bits, f32_bits,
        "exact BF8 cache values preserve output bits"
    );

    let cache_bucket = 32_usize;
    let mut invalid_row_infinity = packed_cache.clone();
    for (name, bytes) in &mut invalid_row_infinity {
        if name.ends_with(".v") {
            let row_width = bytes.len() / cache_bucket;
            bytes[31 * row_width..32 * row_width].fill(0x7c);
        }
    }
    let infinity_named: Vec<(&str, QuantizedBlock<'_>)> = fixture
        .named
        .iter()
        .map(|(name, values)| match invalid_row_infinity
            .iter()
            .find(|(packed_name, _)| packed_name == name)
        {
            Some((_, bytes)) => (
                name.as_str(),
                QuantizedBlock::Packed {
                    codec: omega::Codec::BFloat8,
                    bytes,
                },
            ),
            None => (name.as_str(), QuantizedBlock::Float32(values)),
        })
        .collect();
    let infinity_output = omega::execute_plan_named(&bf8_plan, &infinity_named)
        .expect("the BF8 plan executes with infinity in the padded V row");
    let infinity_bits: Vec<u32> = infinity_output
        .root()
        .iter()
        .map(|value| value.to_bits())
        .collect();
    assert_eq!(
        infinity_bits, bf8_bits,
        "padded V row values do not enter the attention output"
    );
}

#[test]
fn card_11_bf8_row_selects_the_bf8_source_and_declines_mixed_cache_codecs() {
    let policy = production_numeric_policy();
    let fixture = fixture_with(8, 31, Geometry::ONE_KV_HEAD);
    let resolved = bound_attention(&fixture, policy);
    let attention: Vec<&BoundOp> = resolved
        .iter()
        .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .collect();
    assert_eq!(
        attention.len(),
        2,
        "the fixture has sliding and global attention"
    );
    for bound in attention {
        let packed = [2, 3, 6]
            .into_iter()
            .map(|index| (bound.operands()[index].0, omega::Codec::BFloat8))
            .collect::<omega::PackedOperands>();
        let emitted = omega::emit(bound, &packed, policy).expect("BF8 row source emits");
        assert!(
            emitted.entry.ends_with("_rt"),
            "row-tiled entry: {}",
            emitted.entry
        );
        assert!(emitted.source.contains("device const uchar* in2"));
        assert!(emitted.source.contains("omega_bf8_load_matrix(even_float"));
        assert!(emitted.source.contains("omega_bf8_to_float"));
        assert!(emitted
            .source
            .contains("omega_zero_padded_value_rows(value_float"));
        assert!(emitted.source.contains("simdgroup_float8x8"));
        assert!(!emitted.source.contains("simdgroup_half8x8"));

        let mixed = [
            (bound.operands()[2].0, omega::Codec::BFloat8),
            (bound.operands()[3].0, omega::Codec::BFloat8),
            (bound.operands()[6].0, omega::Codec::BFloat16),
        ]
        .into_iter()
        .collect::<omega::PackedOperands>();
        let error = omega::emit(bound, &mixed, policy).expect_err("mixed cache codecs decline");
        assert!(matches!(
            error,
            omega::EmitError::CachedAttentionKvCodecNotSupported { .. }
        ));
    }
}

#[test]
fn card_13_mma_precision_selects_f32_and_f16_for_each_cache_storage() {
    let policy = production_numeric_policy();
    let fixture = fixture_with(8, 31, Geometry::ONE_KV_HEAD);
    let attention = bound_attention(&fixture, policy)
        .into_iter()
        .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .collect::<Vec<_>>();
    let row_op = attention
        .last()
        .expect("the fixture includes a global row-tiled attention operation");

    for (storage, codec, type_token, loader_token) in [
        (omega::AttentionKvStorage::F32, None, "float* in2", None),
        (
            omega::AttentionKvStorage::Bf16,
            Some(omega::Codec::BFloat16),
            "ushort* in2",
            Some("omega_bf16_load_matrix"),
        ),
        (
            omega::AttentionKvStorage::Bf8,
            Some(omega::Codec::BFloat8),
            "uchar* in2",
            Some("omega_bf8_load_matrix"),
        ),
    ] {
        let packed = codec.map_or_else(omega::PackedOperands::new, |codec| {
            [2, 3, 6]
                .into_iter()
                .map(|index| (row_op.operands()[index].0, codec))
                .collect()
        });
        for (precision, fragment, conversion) in [
            (
                omega::AttentionMmaPrecision::F32,
                "simdgroup_float8x8 weights[tile_blocks];",
                "= value_float;",
            ),
            (
                omega::AttentionMmaPrecision::F16,
                "simdgroup_half8x8 weights[tile_blocks];",
                "= narrow_fragment(value_float);",
            ),
        ] {
            let variant = omega::AttentionVariant {
                kv_storage: storage,
                mma_precision: precision,
                ..omega::AttentionVariant::default()
            };
            let emitted = omega::emit_with_attention_variant(row_op, &packed, policy, variant)
                .expect("the selected row-tiled MMA variant emits");
            assert!(emitted.source.contains(type_token));
            if let Some(loader_token) = loader_token {
                assert!(emitted.source.contains(loader_token));
            }
            assert!(emitted.source.contains(fragment));
            assert!(emitted.source.contains(conversion));
            assert!(emitted
                .source
                .contains("simdgroup_float8x8 accumulated[accumulator_dimensions][accumulator_vectors];"));
            assert!(emitted.source.contains("threadgroup float score_tile"));
            assert!(emitted.source.contains("float local_scores"));
            assert!(emitted.entry.ends_with(match precision {
                omega::AttentionMmaPrecision::F32 => "_mma_f32",
                omega::AttentionMmaPrecision::F16 => "_mma_f16",
                omega::AttentionMmaPrecision::Legacy => unreachable!(),
            }));
        }
    }

    let decode_fixture = fixture_with(1, 31, Geometry::ONE_KV_HEAD);
    let decode_op = bound_attention(&decode_fixture, policy)
        .into_iter()
        .find(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .expect("the fixture includes decode attention");
    let error = omega::emit_with_attention_variant(
        &decode_op,
        &omega::PackedOperands::new(),
        policy,
        omega::AttentionVariant {
            mma_precision: omega::AttentionMmaPrecision::F16,
            ..omega::AttentionVariant::default()
        },
    )
    .expect_err("F16 MMA declines outside the row-tiled form");
    assert!(matches!(
        error,
        omega::EmitError::CachedAttentionMmaPrecisionNotSupported { .. }
    ));
    let unsupported_storage = omega::emit_with_attention_variant(
        row_op,
        &omega::PackedOperands::new(),
        policy,
        omega::AttentionVariant {
            kv_storage: omega::AttentionKvStorage::Bf8,
            ..omega::AttentionVariant::default()
        },
    )
    .expect_err("unimplemented storage selection declines instead of being ignored");
    assert!(matches!(
        unsupported_storage,
        omega::EmitError::CachedAttentionVariantStorageMismatch {
            selected: "bf8",
            bound: "f32"
        }
    ));
}

#[test]
fn card_13_mma_precision_f16_exact_fixture_preserves_attention_output_bits() {
    let policy = production_numeric_policy();
    let mut fixture = fixture_with(8, 31, Geometry::ONE_KV_HEAD);
    for (name, values) in &mut fixture.named {
        if name == "ids" {
            continue;
        }
        if name == "eps" {
            values.fill(0.0);
        } else if name == "token_embd.weight" {
            values.fill(1.0);
        } else if name.ends_with("attn_q.weight") {
            values.fill(1.0 / 256.0);
        } else if name.ends_with("norm.weight") || name.ends_with("layer_output_scale.weight") {
            values.fill(1.0);
        } else if name.starts_with("rope_cos") {
            values.fill(1.0);
        } else if name.starts_with("rope_sin") {
            values.fill(0.0);
        } else if name.starts_with("kv_cache.") {
            let row_width = values.len() / bucket_for(31);
            values.fill(0.0);
            let row_start = 30 * row_width;
            let row_end = row_start + row_width;
            if name.ends_with(".v") {
                values[row_start..row_end].fill(0.5);
            } else {
                values[row_start..row_end].fill(2.0);
            }
        } else {
            values.fill(0.0);
        }
    }
    let attention = bound_attention(&fixture, policy)
        .into_iter()
        .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .collect::<Vec<_>>();
    let root = attention
        .last()
        .expect("the fixture includes a global attention operation")
        .node;
    let roots = [root];
    let named = as_named_blocks(&fixture.named);
    let mut f32_plan = omega::plan_named(
        &fixture.program,
        &fixture.symbols,
        &named,
        &roots,
        policy,
    )
    .expect("the explicit F32 MMA plan builds");
    let plan_storage_mismatch = f32_plan.set_attention_variant(omega::AttentionVariant {
        kv_storage: omega::AttentionKvStorage::Bf8,
        ..omega::AttentionVariant::default()
    });
    assert!(matches!(
        plan_storage_mismatch,
        Err(omega::EmitError::CachedAttentionVariantStorageMismatch {
            selected: "bf8",
            bound: "f32"
        })
    ));
    f32_plan.set_attention_variant(omega::AttentionVariant {
        mma_precision: omega::AttentionMmaPrecision::F32,
        ..omega::AttentionVariant::default()
    })
    .expect("F32 MMA is an admitted variant");
    let mut f16_plan = omega::plan_named(
        &fixture.program,
        &fixture.symbols,
        &named,
        &roots,
        policy,
    )
    .expect("the explicit F16 MMA plan builds");
    f16_plan.set_attention_variant(omega::AttentionVariant {
        mma_precision: omega::AttentionMmaPrecision::F16,
        ..omega::AttentionVariant::default()
    })
    .expect("F16 MMA is an admitted variant");
    let f32_keys = f32_plan
        .kernel_keys()
        .expect("F32 plan exposes its pipeline identities");
    let f16_keys = f16_plan
        .kernel_keys()
        .expect("F16 plan exposes its pipeline identities");
    assert_ne!(f16_keys, f32_keys, "MMA precision participates in kernel identity");
    let f32_output = omega::execute_plan_named(&f32_plan, &named)
        .expect("the explicit F32 MMA plan executes");
    let f16_output = omega::execute_plan_named(&f16_plan, &named)
        .expect("the explicit F16 MMA plan executes");
    let f32_bits: Vec<u32> = f32_output
        .root()
        .iter()
        .map(|value| value.to_bits())
        .collect();
    let f16_bits: Vec<u32> = f16_output
        .root()
        .iter()
        .map(|value| value.to_bits())
        .collect();
    assert!(f32_output.root().iter().all(|value| *value == 0.5));
    assert_eq!(f16_bits, f32_bits, "F16-exact operands preserve output bits");
}

#[test]
fn card_15_v_reuse_device_keeps_query_rows_independent() {
    let policy = production_numeric_policy();
    let mut fixture = fixture_with(8, 31, Geometry::TWO_QUERY_GROUPS);
    let cache_bucket = bucket_for(31);
    for (name, values) in &mut fixture.named {
        if name.starts_with("kv_cache.") && name.ends_with(".v") {
            let row_width = values.len() / cache_bucket;
            for key_row in 0..cache_bucket {
                let row_value = (key_row as f32 + 1.0) / 64.0;
                values[key_row * row_width..(key_row + 1) * row_width].fill(row_value);
            }
        }
    }

    let attention = bound_attention(&fixture, policy)
        .into_iter()
        .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .collect::<Vec<_>>();
    let root = attention
        .last()
        .expect("the fixture includes a global attention operation")
        .node;
    let roots = [root];
    let named = as_named_blocks(&fixture.named);
    let mut legacy_plan = omega::plan_named(
        &fixture.program,
        &fixture.symbols,
        &named,
        &roots,
        policy,
    )
    .expect("the legacy row-tiled plan builds");
    legacy_plan
        .set_attention_variant(omega::AttentionVariant {
            mma_precision: omega::AttentionMmaPrecision::F32,
            ..omega::AttentionVariant::default()
        })
        .expect("the explicit F32 legacy plan is admitted");
    let mut shared_k_plan = omega::plan_named(
        &fixture.program,
        &fixture.symbols,
        &named,
        &roots,
        policy,
    )
    .expect("the shared K control plan builds");
    shared_k_plan
        .set_attention_variant(omega::AttentionVariant {
            mma_precision: omega::AttentionMmaPrecision::F32,
            kv_reuse: omega::AttentionKvReuse::SharedK,
            ..omega::AttentionVariant::default()
        })
        .expect("the shared K control is admitted");
    let mut shared_kv_plan = omega::plan_named(
        &fixture.program,
        &fixture.symbols,
        &named,
        &roots,
        policy,
    )
    .expect("the shared K/V plan builds");
    shared_kv_plan
        .set_attention_variant(omega::AttentionVariant {
            mma_precision: omega::AttentionMmaPrecision::F32,
            kv_reuse: omega::AttentionKvReuse::SharedKv,
            ..omega::AttentionVariant::default()
        })
        .expect("the shared K/V plan is admitted");
    let mut shared_k_f16_plan = omega::plan_named(
        &fixture.program,
        &fixture.symbols,
        &named,
        &roots,
        policy,
    )
    .expect("the F16 shared K control plan builds");
    shared_k_f16_plan
        .set_attention_variant(omega::AttentionVariant {
            mma_precision: omega::AttentionMmaPrecision::F16,
            kv_reuse: omega::AttentionKvReuse::SharedK,
            ..omega::AttentionVariant::default()
        })
        .expect("the F16 shared K control is admitted");
    let mut shared_kv_f16_plan = omega::plan_named(
        &fixture.program,
        &fixture.symbols,
        &named,
        &roots,
        policy,
    )
    .expect("the F16 shared K/V plan builds");
    shared_kv_f16_plan
        .set_attention_variant(omega::AttentionVariant {
            mma_precision: omega::AttentionMmaPrecision::F16,
            kv_reuse: omega::AttentionKvReuse::SharedKv,
            ..omega::AttentionVariant::default()
        })
        .expect("the F16 shared K/V plan is admitted");

    let legacy_keys = legacy_plan
        .kernel_keys()
        .expect("legacy pipeline identities are collected");
    let shared_k_keys = shared_k_plan
        .kernel_keys()
        .expect("shared K pipeline identities are collected");
    let shared_kv_keys = shared_kv_plan
        .kernel_keys()
        .expect("shared K/V pipeline identities are collected");
    assert_ne!(legacy_keys, shared_k_keys);
    assert_ne!(shared_k_keys, shared_kv_keys);

    let legacy = omega::execute_plan_named(&legacy_plan, &named)
        .expect("the legacy control executes");
    let shared_k = omega::execute_plan_named(&shared_k_plan, &named)
        .expect("the shared K control executes");
    let shared_kv = omega::execute_plan_named(&shared_kv_plan, &named)
        .expect("the shared K/V plan executes");
    let shared_k_f16 = omega::execute_plan_named(&shared_k_f16_plan, &named)
        .expect("the F16 shared K plan executes");
    let shared_kv_f16 = omega::execute_plan_named(&shared_kv_f16_plan, &named)
        .expect("the F16 shared K/V plan executes");
    let expected = run_resolved_root_on_cpu(
        &fixture,
        &bind_with_fusion(
            &fixture.program,
            &infer(&fixture.program, &fixture.symbols).expect("the fixture infers"),
            &roots,
            false,
            policy,
        )
        .expect("the unfused CPU reference binds"),
        root,
    );
    let legacy_output = legacy
        .get(root)
        .expect("the legacy attention output is retained")
        .0;
    let shared_k_output = shared_k
        .get(root)
        .expect("the shared K attention output is retained")
        .0;
    let shared_kv_output = shared_kv
        .get(root)
        .expect("the shared K/V attention output is retained")
        .0;
    let shared_k_f16_output = shared_k_f16
        .get(root)
        .expect("the F16 shared K output is retained")
        .0;
    let shared_kv_f16_output = shared_kv_f16
        .get(root)
        .expect("the F16 shared K/V output is retained")
        .0;
    for (label, output) in [
        ("legacy", legacy_output),
        ("shared K", shared_k_output),
        ("shared K/V", shared_kv_output),
    ] {
        let relative = relative_difference(&expected, output);
        assert!(relative < TOLERANCE, "{label} differs from CPU: {relative}");
    }
    let row_width = shared_kv_output.len() / 8;
    assert_ne!(
        &shared_kv_output[..row_width],
        &shared_kv_output[row_width..2 * row_width],
        "the two query rows retain distinct weighted outputs"
    );
    let shared_k_f16_bits: Vec<u32> = shared_k_f16_output
        .iter()
        .map(|value| value.to_bits())
        .collect();
    let shared_kv_f16_bits: Vec<u32> = shared_kv_f16_output
        .iter()
        .map(|value| value.to_bits())
        .collect();
    assert_eq!(
        shared_kv_f16_bits, shared_k_f16_bits,
        "V sharing preserves outputs at the selected F16 MMA precision"
    );

    let mut infinity_fixture_values = fixture.named.clone();
    for (name, values) in &mut infinity_fixture_values {
        if name.starts_with("kv_cache.") && name.ends_with(".v") {
            let row_width = values.len() / cache_bucket;
            values[(cache_bucket - 1) * row_width..].fill(f32::INFINITY);
        }
    }
    let infinity_named = as_named_blocks(&infinity_fixture_values);
    let infinity_output = omega::execute_plan_named(&shared_kv_plan, &infinity_named)
        .expect("the shared K/V plan executes with infinity in a padded V row");
    let infinity_values = infinity_output
        .get(root)
        .expect("the infinity attention output is retained")
        .0;
    let output_bits: Vec<u32> = shared_kv_output
        .iter()
        .map(|value| value.to_bits())
        .collect();
    let infinity_bits: Vec<u32> = infinity_values
        .iter()
        .map(|value| value.to_bits())
        .collect();
    assert_eq!(infinity_bits, output_bits, "the padded V row stays masked");
}

#[test]
fn card_16_tile_height_plan_dispatches_rows16_and_matches_cpu_payload() {
    let policy = production_numeric_policy();
    let fixture = fixture_with(16, 31, Geometry::TWO_QUERY_GROUPS);
    let resolved = bound_attention(&fixture, policy);
    let expected = run_resolved_on_cpu(&fixture, &resolved);
    let named = as_named_blocks(&fixture.named);
    let roots = [fixture.logits];
    let legacy_plan = omega::plan_named(
        &fixture.program,
        &fixture.symbols,
        &named,
        &roots,
        policy,
    )
    .expect("the legacy plan resolves");
    let mut rows16_plan = omega::plan_named(
        &fixture.program,
        &fixture.symbols,
        &named,
        &roots,
        policy,
    )
    .expect("the rows16 candidate plan resolves");
    rows16_plan
        .set_attention_variant(omega::AttentionVariant {
            tile_height: omega::AttentionTileHeight::Rows16,
            ..omega::AttentionVariant::default()
        })
        .expect("the 16-row tile fits the selected shape");
    let legacy_keys = legacy_plan
        .kernel_keys()
        .expect("legacy keys are collected");
    let rows16_keys = rows16_plan
        .kernel_keys()
        .expect("rows16 keys are collected");
    assert_ne!(legacy_keys, rows16_keys);
    assert!(rows16_keys.iter().any(|key| key.contains("_tile_rows16")));

    let output = omega::execute_plan_named(&rows16_plan, &named)
        .expect("the selected rows16 plan executes");
    assert!(
        relative_difference(&expected, output.root()) <= TOLERANCE,
        "rows16 plan output matches the CPU payload"
    );
}

#[test]
fn card_17_query_parallelism_plan_preserves_shared_kv_and_matches_cpu_payload() {
    let policy = production_numeric_policy();
    let fixture = fixture_with(8, 31, Geometry::TWO_QUERY_GROUPS);
    let resolved = bound_attention(&fixture, policy);
    let expected = run_resolved_on_cpu(&fixture, &resolved);
    let named = as_named_blocks(&fixture.named);
    let roots = [fixture.logits];
    let mut query_parallel_plan = omega::plan_named(
        &fixture.program,
        &fixture.symbols,
        &named,
        &roots,
        policy,
    )
    .expect("the Granite-shaped query-parallel plan resolves");
    query_parallel_plan
        .set_attention_variant(omega::AttentionVariant {
            kv_reuse: omega::AttentionKvReuse::SharedKv,
            tile_height: omega::AttentionTileHeight::Rows8,
            query_parallelism: omega::AttentionQueryParallelism::SimdgroupRows,
            ..omega::AttentionVariant::default()
        })
        .expect("query parallelism composes with shared K/V staging");
    let keys = query_parallel_plan
        .kernel_keys()
        .expect("query-parallel pipeline identities are collected");
    assert!(keys.iter().any(|key| key.contains("_query_simdgroup_rows")));
    assert!(keys.iter().any(|key| key.contains("_kv_shared_kv")));
    assert!(keys.iter().any(|key| key.contains("_tile_rows8")));

    let output = omega::execute_plan_named(&query_parallel_plan, &named)
        .expect("the selected query-parallel plan executes");
    assert!(
        relative_difference(&expected, output.root()) <= TOLERANCE,
        "query-parallel SharedKv output matches the CPU payload"
    );
}
