//! Metal-vs-CPU parity for a fused [`proxima_tensor::BoundOpKind::Reduce::
//! epilogue_body`] on the real single-range forward program
//! (`support::real_single_range_forward_fixture_with_padding`, the same
//! fixture `cached_attention_coop_load_parity.rs` uses), run at the same
//! three `kv-capacity-bucket` paddings that test sweeps: 0, 1, and 5 rows
//! past the merged `cached_len + new_count` length.
//!
//! With `reduce-epilogue-fusion` compiled into both `proxima-tensor` (via
//! this crate's own passthrough feature) and `omega`, `bind`'s post-pass
//! fuses at least one `Reduce`'s elementwise consumer into its epilogue on
//! this program (`docs/discipline.md`'s own census: 4/layer -- `global_max`,
//! `residual1`, `ffn_hidden`, `x_next`), so this test also asserts that
//! actually happened rather than silently passing on an unfused program.

#![cfg(all(
    feature = "metal",
    feature = "reduce-epilogue-fusion",
    target_os = "macos"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::cpu::evaluate_quantized_named_with_scratch;
use proxima_tensor::{BoundOpKind, NumericPolicy, bind};

mod support;
use support::{as_named_blocks, real_single_range_forward_fixture_with_padding};

fn epilogued_reduce_count(resolved: &[proxima_tensor::BoundOp]) -> usize {
    resolved
        .iter()
        .filter(|bound| {
            matches!(
                &bound.kind,
                BoundOpKind::Reduce {
                    epilogue_operands, ..
                } if !epilogue_operands.is_empty()
            )
        })
        .count()
}

/// One padding value's worth of the parity check -- same structure as
/// `cached_attention_coop_load_parity.rs`'s own `assert_parity_at_padding`,
/// naming the exact padding a failure happened at.
fn assert_parity_at_padding(padding: u64) {
    const CACHED_LEN: u64 = 5;
    const NEW_COUNT: u64 = 1;

    let (program, symbols, roots, owned) =
        real_single_range_forward_fixture_with_padding(CACHED_LEN, NEW_COUNT, padding);
    let output_roots = [roots[0]];
    let named = as_named_blocks(&owned);

    let shapes =
        proxima_tensor::infer(&program, &symbols).expect("single-range padded fixture infers");
    let resolved =
        bind(&program, &shapes, &output_roots, NumericPolicy::default()).expect("single-range padded fixture binds");
    assert!(
        epilogued_reduce_count(&resolved) > 0,
        "padding={padding}: reduce-epilogue-fusion is compiled in but fused nothing on \
         the real single-range program -- either the fixture no longer carries a fusable \
         reduce+elementwise pair or the fusion pass regressed"
    );

    let mut free_buffers = Vec::new();
    let mut validated = None;
    let cpu = evaluate_quantized_named_with_scratch(
        &program,
        &symbols,
        &named,
        &output_roots,
        &mut free_buffers,
        &mut validated,
    )
    .expect("cpu runs the padded single-range program");

    let plan = omega::plan_named(&program, &symbols, &named, &output_roots, NumericPolicy::default())
        .expect("metal plans the padded single-range program");
    let metal = omega::execute_plan_named(&plan, &named)
        .expect("metal runs the padded single-range program on a real device");

    let expected = cpu.root();
    let actual = metal.root();
    assert_eq!(actual.len(), expected.len(), "padding={padding}");

    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let max_diff = expected
        .iter()
        .zip(actual.iter())
        .map(|(want, got)| (want - got).abs())
        .fold(0.0f32, f32::max);
    let relative = max_diff / max_magnitude.max(f32::MIN_POSITIVE);
    eprintln!(
        "reduce-epilogue-fusion parity: padding={padding} max_diff={max_diff} \
         max_magnitude={max_magnitude} relative={relative}"
    );
    assert!(
        relative < 1e-4,
        "padding={padding}: metal disagrees with cpu on the fused-epilogue root: \
         relative={relative} max_diff={max_diff}"
    );
}

/// `bind::reduce_epilogue_candidates` fuses this real program's own RMSNorm
/// `x * inv_rms` tail (`bind.rs`'s `BoundOpKind::Reduce::epilogue_broadcast_
/// axes`) in addition to the plain epilogues the census above names --
/// `omega::msl::render_reduce`'s broadcast-reduce cooperative-write tail
/// (`push_broadcast_epilogue_write`) is what makes this parity check pass
/// end to end now, not just the plain-epilogue sites.
#[test]
fn the_fused_epilogue_holds_parity_at_every_kv_capacity_bucket_padding() {
    for padding in [0u64, 1, 5] {
        assert_parity_at_padding(padding);
    }
}

/// Same shape as `metal_parity.rs`'s own determinism sweeps: the fused
/// epilogue kernel must produce the SAME bytes every dispatch, not merely
/// numbers within tolerance of each other -- a race in the epilogue's own
/// buffer indexing (a wrong `epi{index}` slot, a stale uniform) would show up
/// as run-to-run jitter before it ever failed the CPU-parity check above.
#[test]
fn the_fused_epilogue_is_byte_identical_across_twenty_dispatches() {
    const CACHED_LEN: u64 = 5;
    const NEW_COUNT: u64 = 1;

    let (program, symbols, roots, owned) =
        real_single_range_forward_fixture_with_padding(CACHED_LEN, NEW_COUNT, 0);
    let output_roots = [roots[0]];
    let named = as_named_blocks(&owned);

    let plan = omega::plan_named(&program, &symbols, &named, &output_roots, NumericPolicy::default())
        .expect("metal plans the single-range program");
    let first = omega::execute_plan_named(&plan, &named)
        .expect("metal runs the single-range program on a real device")
        .root()
        .to_vec();

    for run in 1..20 {
        let repeat = omega::execute_plan_named(&plan, &named)
            .expect("metal re-runs the single-range program on a real device");
        assert_eq!(
            repeat.root(),
            first.as_slice(),
            "run {run}: the fused-epilogue kernel produced different bytes on a repeat dispatch"
        );
    }
}

/// A standalone `[seq=7, dim=4096]` RMSNorm -- real BGE/Mistral hidden
/// width, real-valued input from `Lcg` -- isolating the broadcast-reduce
/// epilogue's own Metal renderer (`push_broadcast_epilogue_write`) from the
/// larger real single-range fixture above: `x * inv_rms * gamma` re-
/// broadcasts the fold's scalar back over the very axis it reduced away,
/// mirroring `bind.rs`'s own `rmsnorm_broadcast_reduce_epilogue_matches_
/// bit_for_bit` (CPU, bit-identical) at the Metal boundary (tolerance, not
/// bit-identical, since Metal's `simd_sum` accumulation order differs from
/// the CPU's serial fold).
#[test]
fn the_rmsnorm_broadcast_epilogue_holds_parity_on_a_real_hidden_width() {
    use proxima_tensor::test_support::Lcg;
    use proxima_tensor::{
        DType, Extent, IndexMap, Keep, Op, Reduce, ReduceInit, ScalarOp, append, projection,
    };

    const SEQ: u32 = 7;
    const DIM: u32 = 4096;

    let mut program: Vec<Op> = Vec::new();
    let full = || IndexMap::Affine(projection(2, &[0, 1]));
    let keep_seq = || IndexMap::Affine(projection(1, &[0]));
    let broadcast_scalar_seq = || IndexMap::Affine(projection(1, &[]));
    let broadcast_seq_over_dim = || IndexMap::Affine(projection(2, &[0]));
    let broadcast_dim_over_seq = || IndexMap::Affine(projection(2, &[1]));

    let x = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(SEQ), Extent::Static(DIM)],
            name: Some("x".into()),
        },
    );
    let gamma = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(DIM)],
            name: Some("gamma".into()),
        },
    );
    let inv_dim = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: Vec::new(),
            name: Some("inv_dim".into()),
        },
    );
    let eps = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: Vec::new(),
            name: Some("eps".into()),
        },
    );
    let squared = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![(x, full()), (x, full())],
            name: None,
        },
    );
    let sum_squares = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: squared,
            in_map: IndexMap::Affine(projection(2, &[0, 1])),
            out_map: IndexMap::Affine(projection(2, &[0])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    let mean_square = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![(sum_squares, keep_seq()), (inv_dim, broadcast_scalar_seq())],
            name: None,
        },
    );
    let mean_square_eps = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            operands: vec![(mean_square, keep_seq()), (eps, broadcast_scalar_seq())],
            name: None,
        },
    );
    let rms = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::SquareRoot,
            operands: vec![(mean_square_eps, keep_seq())],
            name: None,
        },
    );
    let inv_rms = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Reciprocal,
            operands: vec![(rms, keep_seq())],
            name: None,
        },
    );
    let normed = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![(x, full()), (inv_rms, broadcast_seq_over_dim())],
            name: None,
        },
    );
    let scaled = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: vec![(normed, full()), (gamma, broadcast_dim_over_seq())],
            name: None,
        },
    );

    let shapes = proxima_tensor::infer(&program, &[]).expect("rmsnorm program infers");
    let output_roots = [scaled];
    let resolved =
        bind(&program, &shapes, &output_roots, NumericPolicy::default()).expect("rmsnorm program binds with fusion");
    assert!(
        epilogued_reduce_count(&resolved) > 0,
        "reduce-epilogue-fusion is compiled in but fused nothing on the standalone rmsnorm \
         program -- the broadcast-reduce candidate match regressed"
    );

    let mut lcg = Lcg(7 * 97 + 3);
    let x_data: Vec<f32> = (0..(SEQ as u64 * DIM as u64) as usize)
        .map(|_| lcg.next_unit())
        .collect();
    let gamma_data: Vec<f32> = (0..DIM as usize).map(|_| lcg.next_unit()).collect();
    let named: Vec<(String, Vec<f32>)> = vec![
        ("x".to_string(), x_data),
        ("gamma".to_string(), gamma_data),
        ("inv_dim".to_string(), vec![1.0f32 / DIM as f32]),
        ("eps".to_string(), vec![1e-5f32]),
    ];
    let named = as_named_blocks(&named);

    let mut free_buffers = Vec::new();
    let mut validated = None;
    let cpu = evaluate_quantized_named_with_scratch(
        &program,
        &[],
        &named,
        &output_roots,
        &mut free_buffers,
        &mut validated,
    )
    .expect("cpu runs the rmsnorm program");

    let plan = omega::plan_named(&program, &[], &named, &output_roots, NumericPolicy::default())
        .expect("metal plans the rmsnorm program");
    let metal = omega::execute_plan_named(&plan, &named)
        .expect("metal runs the rmsnorm program on a real device");

    let expected = cpu.root();
    let actual = metal.root();
    assert_eq!(actual.len(), expected.len());

    let max_magnitude = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0f32, f32::max);
    let max_diff = expected
        .iter()
        .zip(actual.iter())
        .map(|(want, got)| (want - got).abs())
        .fold(0.0f32, f32::max);
    let relative = max_diff / max_magnitude.max(f32::MIN_POSITIVE);
    eprintln!(
        "rmsnorm broadcast-reduce epilogue parity: max_diff={max_diff} \
         max_magnitude={max_magnitude} relative={relative}"
    );
    assert!(
        relative < 1e-4,
        "metal disagrees with cpu on the rmsnorm broadcast-reduce epilogue: relative={relative} \
         max_diff={max_diff}"
    );
}
