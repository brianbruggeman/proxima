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
    buffers[fixture.logits.0 as usize]
        .clone()
        .expect("the logits root was computed")
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
