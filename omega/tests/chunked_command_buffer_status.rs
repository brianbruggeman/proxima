//! `PROXIMA_COMMAND_BUFFER_CHUNKS`-shaped (K>1) command-buffer splitting
//! must produce byte-identical output to the unsplit (K=1) path, and every
//! command buffer it commits — not only the last one
//! [`omega::execute_plan_with_placements`] waits on directly — must have its
//! status checked before any output is read back. Before this fix, an
//! intermediate chunk's failure in a non-`instrument` build went unchecked
//! (only `chunk_command_buffers`, `#[cfg(feature = "instrument")]`, ever saw
//! it) and its output was read back as if it had succeeded.

#![cfg(all(feature = "metal-output-placement", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::{
    DType, Evaluated, Extent, IndexMap, NodeId, NumericPolicy, Op, QuantizedBlock, ScalarOp,
    append, evaluate, projection,
};

/// Same shape as `metal_parity.rs`'s own `assert_parity` (1e-6 max-abs-diff
/// tolerance): a GPU run under [`MathMode::default`]'s `Relaxed` mode is
/// legitimately a few ULPs off the CPU's exact libm math, so the CPU
/// cross-check here is a tolerance compare, never `assert_eq!`. The K=1 vs
/// K=3 comparison below stays bitwise -- both are the SAME GPU math path,
/// only the command-buffer split differs, so those two must match exactly.
fn assert_close(case: &str, cpu: &[f32], metal: &[f32]) {
    assert_eq!(cpu.len(), metal.len(), "{case}: length mismatch");
    let max_abs_diff = cpu
        .iter()
        .zip(metal)
        .map(|(reference, actual)| (reference - actual).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs_diff <= 1e-6,
        "{case}: max abs diff {max_abs_diff} exceeds 1e-6 tolerance across {} elements",
        cpu.len()
    );
}

/// Six SIBLING unary dispatches (`Identity`/`Negate`/`Tanh`/`Exponential`/
/// `Erf`/`SquareRoot`), each reading the SAME `Op::Input` directly -- not a
/// linear chain. `bind_with_fusion`'s single-consumer rule
/// (`proxima-tensor/src/bind/builder_compose_window.rs:149-153`) fuses a
/// producer into a consumer only when the producer has no OTHER pending
/// reader; six siblings sharing one input are all "other pending readers"
/// of each other, so none fuse and this compiles to 6 separate `BoundOp`
/// dispatches (`prepared.resolved.len() == 6`), confirmed against this
/// file's own `PROXIMA_COMMAND_BUFFER_CHUNKS` boundary math: a chain of
/// unary ops on the same shape collapses to ONE fused dispatch instead
/// (verified empirically: a 6-deep `Tanh` chain produced `total_ops == 1`),
/// which is why this fixture uses siblings, not a chain.
const OPS: [ScalarOp; 6] = [
    ScalarOp::Identity,
    ScalarOp::Negate,
    ScalarOp::Tanh,
    ScalarOp::Exponential,
    ScalarOp::Erf,
    ScalarOp::SquareRoot,
];

fn sibling_ops_program(extent: u32) -> (Vec<Op>, Vec<NodeId>) {
    let mut program = Vec::new();
    let input = append(
        &mut program,
        Op::Input {
            dtype: DType::Float32,
            shape: vec![Extent::Static(extent)],
            name: None,
        },
    );
    let outputs = OPS
        .iter()
        .map(|body| {
            append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: *body,
                    operands: vec![(input, IndexMap::Affine(projection(1, &[0])))],
                    name: None,
                },
            )
        })
        .collect();
    (program, outputs)
}

/// Concatenates every declared output's data, in `outputs`' own order, into
/// one flat vector -- the shape both the bitwise K=1-vs-K=3 comparison and
/// the tolerance-based CPU cross-check below compare against.
fn gather(evaluated: &Evaluated, outputs: &[NodeId]) -> Vec<f32> {
    outputs
        .iter()
        .flat_map(|node| {
            evaluated
                .get(*node)
                .expect("every declared output must be present in Evaluated")
                .0
                .to_vec()
        })
        .collect()
}

#[test]
fn chunked_command_buffers_match_the_unsplit_run_bitwise_and_check_every_status() {
    const EXTENT: u32 = 32;

    let (program, outputs) = sibling_ops_program(EXTENT);
    // strictly positive and moderate in magnitude: `SquareRoot` needs
    // non-negative input, `Exponential` must not overflow f32 at this range.
    let input: Vec<f32> = (0..EXTENT).map(|index| 0.5 + index as f32 * 0.05).collect();

    let cpu = evaluate(&program, &[], &[&input[..]], &outputs)
        .expect("cpu evaluates all six sibling ops as the independent oracle");
    let cpu_output = gather(&cpu, &outputs);

    let mut plan = omega::plan(
        &program,
        &[],
        &[QuantizedBlock::Float32(&input)],
        &outputs,
        NumericPolicy::default(),
    )
    .expect("plans the six sibling ops once, reused by both the K=1 and K=3 runs below");

    plan.set_command_buffer_chunks(1, true);
    let checks_before_k1 = omega::metal::COMMAND_BUFFER_STATUS_CHECKS.get();
    let unsplit = omega::execute_plan_with_placements(
        &plan,
        &[QuantizedBlock::Float32(&input)],
        &[],
        &[],
        &mut Vec::new(),
    )
    .expect("K=1 (unsplit) run executes on a real device");
    let checks_after_k1 = omega::metal::COMMAND_BUFFER_STATUS_CHECKS.get();
    let unsplit_output = gather(&unsplit, &outputs);
    assert_eq!(
        checks_after_k1 - checks_before_k1,
        1,
        "K=1 must commit and status-check exactly one command buffer"
    );
    assert_close("k1_unsplit_vs_cpu", &cpu_output, &unsplit_output);

    plan.set_command_buffer_chunks(3, true);
    let checks_before_k3 = omega::metal::COMMAND_BUFFER_STATUS_CHECKS.get();
    let split = omega::execute_plan_with_placements(
        &plan,
        &[QuantizedBlock::Float32(&input)],
        &[],
        &[],
        &mut Vec::new(),
    )
    .expect("K=3 (chunked) run executes on a real device");
    let checks_after_k3 = omega::metal::COMMAND_BUFFER_STATUS_CHECKS.get();
    let split_output = gather(&split, &outputs);

    assert_eq!(
        checks_after_k3 - checks_before_k3,
        3,
        "K=3 must commit and status-check all three chunked command buffers, not only the last"
    );
    assert_eq!(
        split_output.iter().map(|value| value.to_bits()).collect::<Vec<u32>>(),
        unsplit_output.iter().map(|value| value.to_bits()).collect::<Vec<u32>>(),
        "splitting 6 sibling dispatches into 3 command buffers must not change a single bit \
         of any output versus the unsplit K=1 run"
    );
    assert_close("k3_split_vs_cpu", &cpu_output, &split_output);
}
