//! ROW 569 (`docs/discipline.md`): the `moe_topk` dispatch ALONE, isolated
//! from every other op in a real forward pass -- the `op_profile_kind`
//! `gpu_ms` figures used earlier this row are INCLUSIVE cumulative sums
//! (they total over 1,200 ms on a ~70 ms token) and are not a per-kernel
//! cost; this is the actual per-dispatch number, read straight off
//! `omega::execute_plan_with_placements_dispatch_timed`'s own
//! `OpGpuTiming::gpu_ns` for a plan whose ONLY compute op is the fused
//! `BoundOpKind::MoeTopK` -- `outputs` names only routing nodes, so
//! `bind`'s own liveness reachability never pulls in the per-round FFN
//! evaluation ops `append_moe_ffn_from_logits` also builds.

#![cfg(all(feature = "metal", feature = "instrument", feature = "moe-topk-fusion", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::spec::{ExpertGatingFunc, append_moe_ffn_from_logits, input_leaf, scalar_constant};
use proxima_tensor::{DType, Extent, NumericPolicy, QuantizedBlock};

const EXPERT_COUNT: u32 = 256;
const EXPERT_USED_COUNT: u32 = 8;
const EMBEDDING: u32 = 8;
const FEED_FORWARD: u32 = 8;

#[test]
fn moe_topk_bare_dispatch_median_of_seven() {
    let mut program = Vec::new();
    let x = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(1), Extent::Static(EMBEDDING)],
        "x",
    );
    let logits = input_leaf(
        &mut program,
        DType::Float32,
        vec![Extent::Static(1), Extent::Static(EXPERT_COUNT)],
        "logits",
    );
    let expert_w_gate = input_leaf(
        &mut program,
        DType::Float32,
        vec![
            Extent::Static(EXPERT_COUNT),
            Extent::Static(EMBEDDING),
            Extent::Static(FEED_FORWARD),
        ],
        "expert_w_gate",
    );
    let expert_w_up = input_leaf(
        &mut program,
        DType::Float32,
        vec![
            Extent::Static(EXPERT_COUNT),
            Extent::Static(EMBEDDING),
            Extent::Static(FEED_FORWARD),
        ],
        "expert_w_up",
    );
    let expert_w_down = input_leaf(
        &mut program,
        DType::Float32,
        vec![
            Extent::Static(EXPERT_COUNT),
            Extent::Static(FEED_FORWARD),
            Extent::Static(EMBEDDING),
        ],
        "expert_w_down",
    );
    let one = scalar_constant(&mut program, 1.0);
    let (_output, site) = append_moe_ffn_from_logits(
        &mut program,
        0,
        x,
        logits,
        expert_w_gate,
        expert_w_up,
        expert_w_down,
        EXPERT_COUNT,
        EXPERT_USED_COUNT,
        one,
        ExpertGatingFunc::Softmax,
        None,
    )
    .expect("real-shape qwen35moe routing block lowers");

    // Routing outputs ONLY -- `bind`'s own liveness never pulls in the
    // per-round FFN evaluation ops (gate/up/down gathers, SwiGLU) that
    // `append_moe_ffn_from_logits` also built, since nothing in the routing
    // chain reads them.
    let mut outputs = site.selected.clone();
    outputs.extend(site.weights.iter().copied());

    let x_data = vec![0.1_f32; EMBEDDING as usize];
    let logits_data: Vec<f32> = (0..EXPERT_COUNT as usize)
        .map(|index| (index as f32) * 0.01)
        .collect();
    let expert_w_gate_data =
        vec![0.0_f32; EXPERT_COUNT as usize * EMBEDDING as usize * FEED_FORWARD as usize];
    let expert_w_up_data = expert_w_gate_data.clone();
    let expert_w_down_data = expert_w_gate_data.clone();

    let blocks = [
        QuantizedBlock::Float32(&x_data),
        QuantizedBlock::Float32(&logits_data),
        QuantizedBlock::Float32(&expert_w_gate_data),
        QuantizedBlock::Float32(&expert_w_up_data),
        QuantizedBlock::Float32(&expert_w_down_data),
    ];

    let plan = omega::plan(&program, &[], &blocks, &outputs, NumericPolicy::default())
        .expect("routing-only plan builds");

    let mut samples = Vec::new();
    for _ in 0..7 {
        let (_evaluated, timings, sampling_mode, _encoder_split_ns) =
            omega::execute_plan_with_placements_dispatch_timed(&plan, &blocks, &[], &[])
                .expect("dispatch-timed execution runs on a real Metal device");
        assert_ne!(
            sampling_mode, "unsupported",
            "this device must report real counter-sampling support for a bare-kernel timing to mean anything"
        );
        let moe_topk_timings: Vec<&omega::metal::OpGpuTiming> = timings
            .iter()
            .filter(|timing| timing.kind == "moe_topk")
            .collect();
        assert_eq!(
            moe_topk_timings.len(),
            1,
            "the routing-only plan must resolve to exactly one dispatch, the fused MoeTopK op \
             itself -- got kinds {:?}",
            timings.iter().map(|timing| timing.kind).collect::<Vec<_>>()
        );
        samples.push(moe_topk_timings[0].gpu_ns);
    }

    samples.sort_unstable();
    let median_ns = samples[samples.len() / 2];
    println!("moe_topk bare dispatch samples (ns): {samples:?}");
    println!("moe_topk bare dispatch median: {median_ns} ns = {} us", median_ns as f64 / 1000.0);
    assert!(median_ns > 0, "a real Metal dispatch must report nonzero gpu_ns");
}
