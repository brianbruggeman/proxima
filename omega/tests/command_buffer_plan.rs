use omega::command_buffer_plan::chunk_boundaries;

const LONG_PLAN_OPS: usize = 842;
const SHORT_PLAN_OPS: usize = 383;
const CHUNKS: usize = 8;
const HEAD_OPS: usize = 24;
const GAP_THRESHOLD_NS: u64 = 30_000;

struct MeasuredStep {
    total_ops: usize,
    head_gpu_ns: u64,
    second_chunk_ops: usize,
    second_chunk_encode_ns: u64,
    today_boundaries: [usize; 7],
}

const LONG_PLAN: MeasuredStep = MeasuredStep {
    total_ops: LONG_PLAN_OPS,
    head_gpu_ns: 262_000,
    second_chunk_ops: 116,
    second_chunk_encode_ns: 823_000,
    today_boundaries: [24, 140, 257, 374, 491, 608, 725],
};

const SHORT_PLAN: MeasuredStep = MeasuredStep {
    total_ops: SHORT_PLAN_OPS,
    head_gpu_ns: 273_000,
    second_chunk_ops: 51,
    second_chunk_encode_ns: 488_000,
    today_boundaries: [24, 75, 126, 177, 229, 280, 331],
};

fn first_gap_ns(step: &MeasuredStep, boundaries: &[usize]) -> u64 {
    let second_chunk = boundaries[1] - boundaries[0];
    let encode_ns = step.second_chunk_encode_ns * second_chunk as u64 / step.second_chunk_ops as u64;
    let head_gpu_ns = step.head_gpu_ns * boundaries[0] as u64 / HEAD_OPS as u64;
    encode_ns.saturating_sub(head_gpu_ns)
}

#[test]
fn growth_of_one_thousand_with_a_24_op_head_is_the_even_tail_schedule_of_both_decode_plans() {
    for step in [&LONG_PLAN, &SHORT_PLAN] {
        let boundaries = chunk_boundaries(step.total_ops, CHUNKS, HEAD_OPS, 1000);
        assert_eq!(boundaries, step.today_boundaries);
    }
}

#[test]
fn a_measured_decode_step_without_growth_leaves_the_gap_the_timeline_recorded() {
    let long_plan = chunk_boundaries(LONG_PLAN.total_ops, CHUNKS, HEAD_OPS, 1000);
    let short_plan = chunk_boundaries(SHORT_PLAN.total_ops, CHUNKS, HEAD_OPS, 1000);
    assert_eq!(first_gap_ns(&LONG_PLAN, &long_plan), 561_000);
    assert_eq!(first_gap_ns(&SHORT_PLAN, &short_plan), 215_000);
}

#[test]
fn growth_in_the_default_range_keeps_the_first_boundary_gap_under_the_threshold() {
    for step in [&LONG_PLAN, &SHORT_PLAN] {
        let boundaries = chunk_boundaries(step.total_ops, CHUNKS, HEAD_OPS, 1350);
        let gap = first_gap_ns(step, &boundaries);
        assert!(gap <= GAP_THRESHOLD_NS, "{boundaries:?} leaves {gap} ns");
    }
}

#[test]
fn grown_schedule_of_the_two_decode_plans_is_pinned() {
    assert_eq!(
        chunk_boundaries(LONG_PLAN_OPS, CHUNKS, HEAD_OPS, 1350),
        [24, 63, 117, 190, 288, 421, 600]
    );
    assert_eq!(
        chunk_boundaries(SHORT_PLAN_OPS, CHUNKS, HEAD_OPS, 1350),
        [24, 41, 65, 97, 140, 198, 276]
    );
}

#[test]
fn grown_chunks_never_shrink_and_cover_every_op_exactly_once() {
    for step in [&LONG_PLAN, &SHORT_PLAN] {
        let boundaries = chunk_boundaries(step.total_ops, CHUNKS, HEAD_OPS, 1350);
        assert_eq!(boundaries.len(), CHUNKS - 1);
        assert!(boundaries.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(boundaries.iter().all(|boundary| *boundary < step.total_ops));
        let widths: Vec<usize> = boundaries.windows(2).map(|pair| pair[1] - pair[0]).collect();
        assert!(widths.windows(2).all(|pair| pair[1] + 1 >= pair[0]), "{widths:?}");
    }
}

#[test]
fn no_head_two_chunks_or_a_plan_shorter_than_the_head_split_evenly_whatever_the_growth() {
    assert_eq!(chunk_boundaries(1150, 2, HEAD_OPS, 1350), vec![575]);
    assert_eq!(chunk_boundaries(20, 4, HEAD_OPS, 1350), vec![5, 10, 15]);
    assert_eq!(chunk_boundaries(1150, 4, 0, 1350), vec![287, 575, 862]);
}

#[test]
fn one_chunk_or_an_empty_plan_has_no_boundaries() {
    assert!(chunk_boundaries(1150, 1, HEAD_OPS, 1350).is_empty());
    assert!(chunk_boundaries(0, 8, HEAD_OPS, 1350).is_empty());
}

#[test]
fn growth_outside_the_supported_range_is_clamped() {
    assert_eq!(
        chunk_boundaries(842, 8, HEAD_OPS, 500),
        chunk_boundaries(842, 8, HEAD_OPS, 1000)
    );
    assert_eq!(
        chunk_boundaries(842, 8, HEAD_OPS, u32::MAX),
        chunk_boundaries(842, 8, HEAD_OPS, 4000)
    );
}

#[test]
fn shipped_policy_is_the_head_and_growth_the_tests_above_measure() {
    assert_eq!(omega::sized::COMMAND_BUFFER_FIRST_CHUNK_OPS as usize, HEAD_OPS);
    assert_eq!(omega::sized::COMMAND_BUFFER_GROWTH_PERMILLE, 1350);
}
