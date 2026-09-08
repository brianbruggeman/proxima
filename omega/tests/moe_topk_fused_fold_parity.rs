//! CPU-vs-Metal parity for the FULL routed MoE FFN (`spec::append_moe_ffn`)
//! at the real qwen3moe 30B-A3B SHAPE -- 128 experts, top-8
//! argmax-with-exclusion rounds -- not just the gathered-weight product
//! `moe_gather_parity.rs` already covers. Reproduces main 88bd381b's
//! decode-time bug: the real 30B model emits token 0 every step on Metal
//! while CPU decodes coherently.
//!
//! SMALLEST FAILING SHAPE (measured, not the 128/8 the real checkpoint
//! uses): `expert_count=8, top_k=2, seq=1` already fails, at the SAME
//! `omega::metal::buffer_for` error the 30B model hits --
//! `NotLowerable { node: NodeId(35), reason: "operand buffer missing at
//! execution time" }` (`NodeId(15)` when no extra `MoeSite` outputs are
//! requested -- the round-0 analogue of the same node shape). A
//! `proxima_tensor::bind` dump of this exact fixture (`infer` + `bind`,
//! `BoundOp::kind` printed per node) shows the failing node is round 1's
//! own softmax gating weight, `weight = exp(max_selection_round1 -
//! max_selection_round0)` (`spec.rs`'s `append_moe_ffn`, the `weight`
//! binding inside its `for round in 0..expert_used_count` loop), and that
//! this node is a genuine standalone `BoundOp::Elementwise` in `bind`'s own
//! resolved list (`reduce-epilogue-fusion` does not, and structurally
//! cannot, absorb it away -- only a `Reduce`-kind node is
//! `is_epilogue_fusable_reduce`, and this node is `Elementwise`) with TWO
//! real, independent readers: the `weight_total` accumulation (`sd->sd`
//! `Add`, "the normalizing sum") and the final combine reduce's OWN fused
//! epilogue (`epilogue_operands` naming this node directly, "the divide"
//! that renormalizes `weighted_sum / weight_total`). Both readers, and this
//! node's own `BoundOp`, are ordered correctly in `bind`'s resolved
//! sequence (producer before both consumers) -- narrowed that far with
//! captured data, not walked past that point in the 30-minute window: the
//! next instrumentation to close the gap is a `debug!` inside
//! `omega::metal::encode_op`/the dispatch loop (`metal.rs` around the
//! `for (position, bound) in prepared.resolved.iter().enumerate()` loop)
//! logging every `device_buffers.insert`/`.remove` key against this node's
//! id, to see whether its own dispatch's insert is skipped, mis-keyed, or
//! evicted before the epilogue consumer reads it.

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
                                let max_diff = cpu_values
                                    .iter()
                                    .zip(metal_values.iter())
                                    .map(|(&want, &got)| (want - got).abs())
                                    .fold(0.0f32, f32::max);
                                if max_diff > 1e-3 {
                                    failures.push(format!(
                                        "experts={expert_count} top_k={expert_used_count} node_slot={round_index} node={node:?} max_abs_diff={max_diff} cpu={cpu_values:?} metal={metal_values:?}"
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
