//! The device-free half of the row-tiled attention gates: what
//! `cached_attention_row_tiled_parity.rs` assumes about its fixture, proven on
//! the CPU so a Metal run is never the first time they are checked.
//!
//! On the gemma4-shaped two-layer program (`gemma4_rows_support`) at the
//! speculative-verify and prefill shapes, and on granite's two-group shape,
//! with the production policy:
//! - both attention ops bind as `CachedAttention` and emit the row-tiled kernel
//!   (entry `_rt`), with the form's split count and threadgroup count;
//! - the CPU interpreter's relaxed `CachedAttention` over those ops agrees with
//!   the unfused chain it runs when the fusion is off;
//! - the scalar transcription of the kernel (`row_tiled_kernel_model`)
//!   reproduces the CPU oracle on the program's own activations, at the tile
//!   geometry the emitted entry name carries;
//! - forcing `new_upper_inclusive = 1` into those ops moves the logits far past
//!   that agreement, which is what makes the Metal comparison's wrong-mask
//!   control a control.

#![cfg(feature = "metal-attn-split-rows")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use core::pin::pin;
use core::task::{Context, Poll, Waker};

use proxima_primitives::pipe::Pipe;
use proxima_tensor::bind::{READY_BATCH_CAPACITY, ReadyBatch};
use proxima_tensor::{
    BoundOp, BoundOpKind, Interpreter, Op, bind_with_fusion, block_node_ids, infer,
};

mod gemma4_rows_support;
mod row_tiled_kernel_model;
mod support;
use gemma4_rows_support::{Fixture, Geometry, bucket_for, fixture_with};
use row_tiled_kernel_model::{Inputs, Shape, Tiling, model};
use support::production_numeric_policy;

/// `(new rows, live cached rows)`: a short verify over a short cache, and a
/// long verify over a cache past the sliding window.
/// One attention geometry at `rows` new rows over `cached_len` live rows.
struct Cell {
    label: &'static str,
    geometry: Geometry,
    rows: usize,
    cached_len: usize,
}

/// The verify cells, then the prefill cells: rows past the old 64-row limit, a
/// window the new range itself crosses (the sliding layer's window set under
/// `rows`), a row count that leaves a scalar tail of new keys past the last
/// whole fragment, and granite's two query groups, which put eight rows of one
/// head in a fragment.
fn all_cells() -> Vec<Cell> {
    let e2b = Geometry::GEMMA4_E2B;
    let granite = Geometry::GRANITE_MOE;
    let verify = [(2, 33), (5, 199), (17, 511), (49, 1099)].map(|(rows, cached_len)| Cell {
        label: "e2b verify",
        geometry: e2b,
        rows,
        cached_len,
    });
    let prefill = [
        ("e2b past the old row limit", e2b, 70, 33),
        ("e2b window crosses the new range", e2b.with_window(40), 130, 199),
        ("e2b window crosses, ragged tail", e2b.with_window(48), 101, 1),
        ("granite eight rows", granite, 8, 33),
        ("granite ragged rows", granite, 37, 100),
        ("granite window crosses the new range", granite.with_window(24), 50, 40),
    ]
    .map(|(label, geometry, rows, cached_len)| Cell {
        label,
        geometry,
        rows,
        cached_len,
    });
    verify.into_iter().chain(prefill).collect()
}

fn entry_number(entry: &str, marker: &str) -> i64 {
    let start = entry
        .find(marker)
        .unwrap_or_else(|| panic!("entry {entry} has no `{marker}`"))
        + marker.len();
    entry[start..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .unwrap_or_else(|error| panic!("entry {entry}: {error}"))
}
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

fn run_resolved_buffers(fixture: &Fixture, resolved: &[BoundOp]) -> Vec<Option<Vec<f32>>> {
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
    buffers
}

fn run_resolved_on_cpu(fixture: &Fixture, resolved: &[BoundOp]) -> Vec<f32> {
    run_resolved_buffers(fixture, resolved)[fixture.logits.0 as usize]
        .clone()
        .expect("the logits root was computed")
}

fn attention_ops(resolved: &[BoundOp]) -> Vec<&BoundOp> {
    resolved
        .iter()
        .filter(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. }))
        .collect()
}

#[test]
fn the_verify_program_binds_row_tiled_attention_and_the_cpu_oracle_agrees_with_the_unfused_chain() {
    let policy = production_numeric_policy();
    let mut cells = 0_usize;

    for Cell {
        label,
        geometry,
        rows,
        cached_len,
    } in all_cells()
    {
        let fixture = fixture_with(rows, cached_len, geometry);
        let roots = [fixture.logits];
        let cell = format!(
            "{label}: rows {rows} cached_len {cached_len} bucket {}",
            bucket_for(cached_len)
        );

        let shapes = infer(&fixture.program, &fixture.symbols).expect("the program infers");
        let resolved = bind_with_fusion(&fixture.program, &shapes, &roots, true, policy)
            .expect("the program binds");
        let attention = attention_ops(&resolved);
        assert_eq!(attention.len(), 2, "{cell}: both layers fuse");
        for bound in &attention {
            let kernel = omega::emit(bound, &omega::PackedOperands::new(), policy)
                .expect("the bound attention op emits");
            assert!(
                kernel.entry.ends_with("_rt"),
                "{cell}: the op must take the row-tiled form, got {}",
                kernel.entry
            );
            let width = kernel
                .grid
                .threadgroup_width
                .expect("the row-tiled form fixes its threadgroup width");
            assert_eq!(
                width as i64,
                32 * entry_number(&kernel.entry, "_n"),
                "{cell}: the width is the entry's simdgroup count"
            );
        }

        let unfused = bind_with_fusion(&fixture.program, &shapes, &roots, false, policy)
            .expect("the unfused program binds");
        let reference = run_resolved_on_cpu(&fixture, &unfused);
        let fused = run_resolved_on_cpu(&fixture, &resolved);
        let relative = relative_difference(&reference, &fused);
        assert!(
            relative < TOLERANCE,
            "{cell}: the relaxed cached-attention oracle differs from the unfused chain by {relative}"
        );

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
        let control_relative = relative_difference(&reference, &control);
        assert!(
            control_relative > 10.0 * TOLERANCE,
            "{cell}: a mask that lets a row see one future key moved the logits only {control_relative}"
        );
        eprintln!("row_tiled cpu gate: {cell} oracle={relative} wrong_mask={control_relative}");
        cells += 1;
    }
    assert_eq!(cells, all_cells().len(), "every cell must have run");
}

fn finite_garbage(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|index| 1_000.0 * ((index + seed) as f32 * 0.37).sin())
        .collect()
}

/// `buffer` with every row past `live_rows` replaced by large finite values,
/// the contents a capacity bucket's stale tail may hold: the kernel reads
/// these rows inside its last 8-key fragment and must mask them, and a
/// transcription that leaked them would show.
fn with_garbage_tail(buffer: &[f32], row_width: usize, live_rows: usize) -> Vec<f32> {
    let live = live_rows * row_width;
    let mut padded = buffer.to_vec();
    let tail = finite_garbage(padded.len() - live, live);
    padded[live..].copy_from_slice(&tail);
    padded
}

/// The kernel transcription, on the real activations of every attention op of
/// the bound program, reproduces the CPU oracle's output for that op.
#[test]
fn the_kernel_transcription_reproduces_the_cpu_oracle_on_the_programs_own_activations() {
    let policy = production_numeric_policy();
    let mut compared = 0_usize;

    for Cell {
        label,
        geometry,
        rows,
        cached_len,
    } in all_cells()
    {
        let fixture = fixture_with(rows, cached_len, geometry);
        let shapes = infer(&fixture.program, &fixture.symbols).expect("the program infers");
        let resolved = bind_with_fusion(&fixture.program, &shapes, &[fixture.logits], true, policy)
            .expect("the program binds");
        let buffers = run_resolved_buffers(&fixture, &resolved);

        for bound in attention_ops(&resolved) {
            let BoundOpKind::CachedAttention {
                operands,
                query_rows,
                kv_heads,
                query_groups,
                head_dim,
                scale,
                cached_lower_inclusive,
                new_upper_inclusive,
                ..
            } = &bound.kind
            else {
                unreachable!("filtered to CachedAttention");
            };
            let operand = |index: usize| -> &[f32] {
                buffers[operands[index].0.0 as usize]
                    .as_deref()
                    .unwrap_or_else(|| panic!("operand {index} was not computed"))
            };
            let oracle = buffers[bound.node.0 as usize]
                .as_deref()
                .expect("the oracle output was computed");
            let live = operand(8)[0] as i64;
            let half_dim = (*head_dim / 2) as usize;
            let kernel = omega::emit(bound, &omega::PackedOperands::new(), policy)
                .expect("the bound attention op emits");
            let width = kernel.grid.threadgroup_width.expect("a fixed width");
            let tile_rows = entry_number(&kernel.entry, "_r");
            let block = entry_number(&kernel.entry, "_b");
            let tiles = (*query_rows as i64 + tile_rows - 1) / tile_rows;
            let threadgroups = (kernel.grid.threads / width) as i64;
            assert_eq!(threadgroups % (*kv_heads as i64 * tiles), 0);
            let splits = threadgroups / (*kv_heads as i64 * tiles);

            let cached_rows = operand(2).len() / (*kv_heads as usize * half_dim);
            let key_row = *kv_heads as usize * half_dim;
            let value_row = *kv_heads as usize * *head_dim as usize;
            let cached_key_even = with_garbage_tail(operand(2), key_row, live as usize);
            let cached_key_odd = with_garbage_tail(operand(3), key_row, live as usize);
            let cached_value = with_garbage_tail(operand(6), value_row, live as usize);
            assert_eq!(cached_rows % 8, 0, "the bucket is whole 8-key fragments");

            let inputs = Inputs {
                query_even: operand(0),
                query_odd: operand(1),
                cached_key_even: &cached_key_even,
                cached_key_odd: &cached_key_odd,
                new_key_even: operand(4),
                new_key_odd: operand(5),
                cached_value: &cached_value,
                new_value: operand(7),
            };
            let shape = Shape {
                kv_heads: *kv_heads as i64,
                query_groups: *query_groups as i64,
                head_dim: *head_dim as i64,
                rows: *query_rows as i64,
                scale: *scale,
                cached_lower: if *cached_lower_inclusive == i64::MIN {
                    -9_223_372_036_854_775_807
                } else {
                    *cached_lower_inclusive
                },
                new_upper: *new_upper_inclusive,
                live,
            };
            let tiling = Tiling {
                tile_rows,
                block,
                split_keys: omega::sized::ATTENTION_ROWS_KEYS_PER_SPLIT as i64,
                splits,
            };
            let modelled = model(&inputs, &shape, &tiling);
            let relative = relative_difference(oracle, &modelled);
            eprintln!(
                "row_tiled model: {label} rows {rows} cached_len {cached_len} head_dim {head_dim} tile_rows {tile_rows} splits {splits} tiles {tiles} relative={relative}"
            );
            assert!(
                relative < TOLERANCE,
                "{label} rows {rows} cached_len {cached_len} head_dim {head_dim}: the transcription differs from the oracle by {relative}"
            );
            compared += 1;
        }
    }
    assert_eq!(
        compared,
        all_cells().len() * 2,
        "every cell's two attention ops must have been compared"
    );
}

/// Fewer rows than one fragment of a group count that does not fill whole 8-row
/// blocks: the recognizer still fuses the candidate and the form falls back to
/// the one-dispatch kernel; the CPU oracle over that op must still equal the
/// unfused chain, so the fused op carries the right fields at any K.
#[test]
fn rows_under_one_fragment_of_a_two_group_head_keep_the_one_dispatch_kernel_and_the_cpu_oracle_agrees() {
    let policy = production_numeric_policy();
    let rows = 5;
    let cached_len = 33;
    let fixture = fixture_with(rows, cached_len, Geometry::GRANITE_MOE);
    let roots = [fixture.logits];
    let shapes = infer(&fixture.program, &fixture.symbols).expect("the program infers");
    let resolved = bind_with_fusion(&fixture.program, &shapes, &roots, true, policy)
        .expect("the program binds");

    let attention = attention_ops(&resolved);
    assert_eq!(attention.len(), 2, "rows {rows}: both layers fuse");
    for bound in &attention {
        let kernel = omega::emit(bound, &omega::PackedOperands::new(), policy)
            .expect("the bound attention op emits");
        assert!(
            !kernel.entry.ends_with("_rt") && !kernel.entry.ends_with("_ds"),
            "rows {rows}: under eight rows the op keeps the one-dispatch kernel, got {}",
            kernel.entry
        );
    }

    let unfused = bind_with_fusion(&fixture.program, &shapes, &roots, false, policy)
        .expect("the unfused program binds");
    let reference = run_resolved_on_cpu(&fixture, &unfused);
    let fused = run_resolved_on_cpu(&fixture, &resolved);
    let relative = relative_difference(&reference, &fused);
    assert!(
        relative < TOLERANCE,
        "rows {rows}: the relaxed cached-attention oracle differs from the unfused chain by {relative}"
    );
}
