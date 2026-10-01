//! Metal-vs-CPU parity for the row-tiled cached attention form
//! (`CachedAttentionForm::TwoRangeRowTiled`: `omega/src/msl/
//! cached_attention_row_tiled.rs` plus the interleaved merge in
//! `cached_attention_render.rs`). Internal consistency only: the CPU reference
//! is this repo's own unfused chain, never llama.cpp's; the llama.cpp greedy-id
//! gate is `proxima-model-interop/tests/gemma4_attn_split_decode_oracle.rs`.
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

use proxima_primitives::pipe::Pipe;
use proxima_tensor::bind::{READY_BATCH_CAPACITY, ReadyBatch};
use proxima_tensor::cpu::evaluate_quantized_named_with_scratch;
use proxima_tensor::{
    BoundOp, BoundOpKind, Interpreter, NumericPolicy, Op, bind_with_fusion, block_node_ids, infer,
};

mod gemma4_rows_support;
mod support;
use gemma4_rows_support::{Fixture, bucket_for, fixture};
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

fn one_cell(rows: usize, cached_len: usize) -> (f32, f32) {
    let policy = production_numeric_policy();
    let fixture = fixture(rows, cached_len);
    let named = as_named_blocks(&fixture.named);
    let roots = [fixture.logits];
    let cell = format!(
        "rows {rows} cached_len {cached_len} bucket {}",
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

    let mut free_buffers = Vec::new();
    let mut validated = None;
    let cpu = evaluate_quantized_named_with_scratch(
        &fixture.program,
        &fixture.symbols,
        &named,
        &roots,
        &mut free_buffers,
        &mut validated,
    )
    .expect("cpu runs the program");
    let plan = omega::plan_named(&fixture.program, &fixture.symbols, &named, &roots, policy)
        .expect("metal plans the program");
    let metal = omega::execute_plan_named(&plan, &named).expect("metal runs the program");
    let relative = relative_difference(cpu.root(), metal.root());

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
            let (relative, control_relative) = one_cell(rows, cached_len);
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

/// Rows past the tile limit keep the one-dispatch kernel, which the
/// recognizer now reaches for gemma masks it never saw before: hold it to the
/// same CPU reference, with the same entry-name check that it is not the
/// row-tiled kernel.
#[test]
fn rows_past_the_tile_limit_hold_one_dispatch_parity_with_the_cpu_evaluator() {
    let policy = production_numeric_policy();
    let rows = omega::sized::ATTENTION_ROWS_MAX_QUERY_ROWS as usize + 6;
    let mut cells = 0_usize;
    for cached_len in [33_usize, 511] {
        let fixture = fixture(rows, cached_len);
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

        let mut free_buffers = Vec::new();
        let mut validated = None;
        let cpu = evaluate_quantized_named_with_scratch(
            &fixture.program,
            &fixture.symbols,
            &named,
            &roots,
            &mut free_buffers,
            &mut validated,
        )
        .expect("cpu runs the program");
        let plan = omega::plan_named(&fixture.program, &fixture.symbols, &named, &roots, policy)
            .expect("metal plans the program");
        let metal = omega::execute_plan_named(&plan, &named).expect("metal runs the program");
        let relative = relative_difference(cpu.root(), metal.root());
        assert!(relative < TOLERANCE, "{cell}: relative={relative}");
        cells += 1;
    }
    assert_eq!(cells, 2);
}
