//! CPU-vs-Metal parity for the FULL routed MoE FFN (`spec::append_moe_ffn`)
//! at the real qwen3moe 30B-A3B SHAPE -- 128 experts, top-8
//! argmax-with-exclusion rounds -- not just the gathered-weight product
//! `moe_gather_parity.rs` already covers. Reproduces main 88bd381b's
//! decode-time bug: the real 30B model emits token 0 every step on Metal
//! while CPU decodes coherently.
//!
//! ROOT CAUSE (traced via `RUST_LOG=debug` dumps of `Prepared::resolved`'s
//! own node-id order at each `omega::metal::prepare` stage): the final
//! combine's fused `Reduce` (`reduce-epilogue-fusion` folded the output
//! multiply into the weighted-sum reduce, so this op reads round 1's own
//! softmax gating weight -- `spec.rs`'s `append_moe_ffn` `weight` binding --
//! ONLY through its `epilogue_operands`, never through `operands()`) is also
//! a requested output, so `omega::metal::promote_output_placed_nodes` walks
//! it. That function computed a promoted node's own dependency floor from
//! `BoundOp::operands()` alone (`metal.rs`'s `promote_output_placed_nodes`),
//! never `all_read_sources()`, so it moved the fused combine to right after
//! its LAST plain reduce operand and ignored that its epilogue also reads
//! four more nodes emitted LATER in the list -- among them round 1's own
//! gating weight. The combine dispatched before its own epilogue operand's
//! producer, so `omega::metal::buffer_for` found no buffer for it:
//! `NotLowerable { node: NodeId(35), reason: "operand buffer missing at
//! execution time" }` (`NodeId(15)` for round 0's analogous weight when no
//! extra `MoeSite` outputs are requested). Fixed by reading
//! `all_read_sources()` in `promote_output_placed_nodes` instead of
//! `operands()`.

#![cfg(all(feature = "metal", feature = "instrument", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_lines)]

use proxima_tensor::spec::{append_moe_ffn, input_leaf, scalar_constant, ExpertGatingFunc};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{DType, Extent, NumericPolicy};

fn random_vec(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit() * 2.0 - 1.0).collect()
}

struct MoeFixture {
    program: Vec<proxima_tensor::Op>,
    output: proxima_tensor::NodeId,
    extra_outputs: Vec<proxima_tensor::NodeId>,
    named: Vec<(&'static str, Vec<f32>)>,
}

fn build_fixture(seq: u32, expert_count: u32, expert_used_count: u32) -> MoeFixture {
    let embedding = 32u32;
    let expert_hidden = 16u32;

    let mut program = Vec::new();
    let x = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(seq), Extent::Static(embedding)],
        "x",
    );
    let gate_inp = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(embedding), Extent::Static(expert_count)],
        "gate_inp",
    );
    let expert_w_gate = input_leaf(
        &mut program,
        DType::Float32,
        vec![
            Extent::Static(expert_count),
            Extent::Static(embedding),
            Extent::Static(expert_hidden),
        ],
        "expert_w_gate",
    );
    let expert_w_up = input_leaf(
        &mut program,
        DType::Float32,
        vec![
            Extent::Static(expert_count),
            Extent::Static(embedding),
            Extent::Static(expert_hidden),
        ],
        "expert_w_up",
    );
    let expert_w_down = input_leaf(
        &mut program,
        DType::Float32,
        vec![
            Extent::Static(expert_count),
            Extent::Static(expert_hidden),
            Extent::Static(embedding),
        ],
        "expert_w_down",
    );
    let one = scalar_constant(&mut program, 1.0);

    let (routed_out, moe_site) = append_moe_ffn(
        &mut program,
        0,
        x,
        gate_inp,
        expert_w_gate,
        expert_w_up,
        expert_w_down,
        expert_count,
        expert_used_count,
        one,
        ExpertGatingFunc::Softmax,
        None,
    )
    .expect("routed moe ffn lowers at 128/8");

    let mut extra_outputs = moe_site.selected.clone();
    extra_outputs.extend(moe_site.weights.iter().copied());

    let x_data = random_vec(1_001, seq as usize * embedding as usize);
    let gate_inp_data = random_vec(1_002, embedding as usize * expert_count as usize);
    let expert_w_gate_data =
        random_vec(1_003, expert_count as usize * embedding as usize * expert_hidden as usize);
    let expert_w_up_data =
        random_vec(1_004, expert_count as usize * embedding as usize * expert_hidden as usize);
    let expert_w_down_data =
        random_vec(1_005, expert_count as usize * expert_hidden as usize * embedding as usize);

    let named = vec![
        ("x", x_data),
        ("gate_inp", gate_inp_data),
        ("expert_w_gate", expert_w_gate_data),
        ("expert_w_up", expert_w_up_data),
        ("expert_w_down", expert_w_down_data),
    ];

    MoeFixture {
        program,
        output: routed_out,
        extra_outputs,
        named,
    }
}

/// Sweeps `expert_count in {8, 32, 128}` at `top_k in {2, 4, 8}` (`top_k`
/// clamped to `expert_count`) at `seq=1`, comparing CPU vs Metal on the
/// `append_moe_ffn` output AND every per-round router node
/// (`MoeSite.selected`/`weights`) so a failure names the first divergent
/// round, not just the final output.
#[test]
fn moe_topk_fused_fold_parity_sweep_on_metal() {
    let mut failures = Vec::new();

    for &expert_count in &[8u32, 32, 128] {
        for &requested_k in &[2u32, 4, 8] {
            let expert_used_count = requested_k.min(expert_count);
            let seq = 1u32;

            let fixture = build_fixture(seq, expert_count, expert_used_count);
            let mut outputs = vec![fixture.output];
            outputs.extend(fixture.extra_outputs.iter().copied());

            let symbols: Vec<u64> = Vec::new();
            let named: Vec<(&str, proxima_tensor::QuantizedBlock)> = fixture
                .named
                .iter()
                .map(|(name, data)| (*name, proxima_tensor::QuantizedBlock::Float32(data.as_slice())))
                .collect();

            let mut free_buffers = Vec::new();
            let mut validated = None;
            let cpu = proxima_tensor::cpu::evaluate_quantized_named_with_scratch(
                &fixture.program,
                &symbols,
                &named,
                &outputs,
                &mut free_buffers,
                &mut validated,
            );

            let plan = omega::plan_named(
                &fixture.program,
                &symbols,
                &named,
                &outputs,
                NumericPolicy::default(),
            );
            let metal = plan.and_then(|plan| omega::execute_plan_named(&plan, &named));

            match (cpu, metal) {
                (Ok(cpu), Ok(metal)) => {
                    for (round_index, &node) in outputs.iter().enumerate() {
                        let cpu_values = cpu.get(node).map(|(values, _shape)| values.to_vec());
                        let metal_values = metal.get(node).map(|(values, _shape)| values.to_vec());
                        match (cpu_values, metal_values) {
                            (Some(cpu_values), Some(metal_values)) => {
                                // Relative, not absolute: this fixture's logits sit
                                // around 1e4, where a single-ULP f32 accumulation-order
                                // difference between the CPU scalar reduce and Metal's
                                // GPU reduce is already ~1e-2 absolute -- an artifact of
                                // summation order, not a correctness defect. `1e-6` floors
                                // the denominator so a near-zero logit does not blow up a
                                // tiny absolute difference into a spurious failure.
                                let max_relative_diff = cpu_values
                                    .iter()
                                    .zip(metal_values.iter())
                                    .map(|(&want, &got)| {
                                        let denominator = want.abs().max(got.abs()).max(1e-6);
                                        (want - got).abs() / denominator
                                    })
                                    .fold(0.0f32, f32::max);
                                if max_relative_diff > 1e-3 {
                                    failures.push(format!(
                                        "experts={expert_count} top_k={expert_used_count} node_slot={round_index} node={node:?} max_relative_diff={max_relative_diff} cpu={cpu_values:?} metal={metal_values:?}"
                                    ));
                                }
                            }
                            (cpu_present, metal_present) => {
                                failures.push(format!(
                                    "experts={expert_count} top_k={expert_used_count} node_slot={round_index} node={node:?} presence mismatch: cpu_present={} metal_present={}",
                                    cpu_present.is_some(),
                                    metal_present.is_some()
                                ));
                            }
                        }
                    }
                }
                (Err(cpu_error), metal_result) => {
                    failures.push(format!(
                        "experts={expert_count} top_k={expert_used_count} cpu evaluation itself failed: {cpu_error:?} (metal={:?})",
                        metal_result.is_ok()
                    ));
                }
                (Ok(_), Err(metal_error)) => {
                    failures.push(format!(
                        "experts={expert_count} top_k={expert_used_count} metal evaluation failed: {metal_error:?}"
                    ));
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "moe top-k fused fold parity failures:\n{}",
        failures.join("\n")
    );
}
