//! The twin-output elementwise on a real Metal device: one dispatch writes both rotated halves of a
//! RoPE pair, for split-half and adjacent pairing, into fresh buffers and into caller-placed
//! KV-cache buffers. The off arm is `PROXIMA_DISABLE_TWIN_ELEMENTWISE_FUSION`, which plans the same
//! program as two dispatches; the twin must reproduce it bit for bit, and the CPU interpreter
//! within the Metal parity tolerance.
//!
//! Every test needs a Metal device; none skips. A missing device surfaces as `MetalError::NoDevice`
//! through the `expect` calls below, a loud failure rather than a green run that executed nothing.

#![cfg(all(
    feature = "metal-output-placement",
    feature = "twin-elementwise-fusion",
    target_os = "macos"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_tensor::spec::{RopePairing, fused_rope_pair, input_leaf};
use proxima_tensor::test_support::Lcg;
use proxima_tensor::{DType, Extent, NodeId, NumericPolicy, Op, QuantizedBlock, evaluate};

const SEQUENCE: u32 = 3;
const HEADS: u32 = 2;
const PAIRS: u32 = 4;
const DISABLE_TWIN: &str = "PROXIMA_DISABLE_TWIN_ELEMENTWISE_FUSION";
const PLACEMENT_ALIGNMENT: usize = 256;

struct RopeCase {
    program: Vec<Op>,
    first: NodeId,
    second: NodeId,
    inputs: [Vec<f32>; 3],
    sequence: u32,
    heads: u32,
    pairs: u32,
}

impl RopeCase {
    fn new(head_dim: u32, pairing: RopePairing) -> Self {
        Self::sized(SEQUENCE, HEADS, PAIRS, head_dim, pairing)
    }

    fn sized(sequence: u32, heads: u32, pairs: u32, head_dim: u32, pairing: RopePairing) -> Self {
        let mut program = Vec::new();
        let source = input_leaf(
            &mut program,
            DType::Float32,
            vec![
                Extent::Static(sequence),
                Extent::Static(heads),
                Extent::Static(head_dim),
            ],
            "x",
        );
        let trig = || vec![Extent::Static(sequence), Extent::Static(pairs)];
        let cosine = input_leaf(&mut program, DType::Float32, trig(), "cos");
        let sine = input_leaf(&mut program, DType::Float32, trig(), "sin");
        let (first, second) = fused_rope_pair(&mut program, source, 'h', cosine, sine, pairing)
            .expect("rope pair builds over a [seq, heads, head_dim] source");
        Self {
            program,
            first,
            second,
            inputs: [
                random_values(0x0e1e_0001, (sequence * heads * head_dim) as usize),
                random_values(0x0e1e_0002, (sequence * pairs) as usize),
                random_values(0x0e1e_0003, (sequence * pairs) as usize),
            ],
            sequence,
            heads,
            pairs,
        }
    }

    fn blocks(&self) -> [QuantizedBlock<'_>; 3] {
        [
            QuantizedBlock::Float32(&self.inputs[0]),
            QuantizedBlock::Float32(&self.inputs[1]),
            QuantizedBlock::Float32(&self.inputs[2]),
        ]
    }

    fn half_len(&self) -> usize {
        (self.sequence * self.heads * self.pairs) as usize
    }

    fn plan(&self) -> omega::Plan {
        omega::plan(
            &self.program,
            &[],
            &self.blocks(),
            &[self.first, self.second],
            NumericPolicy::default(),
        )
        .expect("plans the rope pair")
    }

    fn run_unplaced(&self) -> (Vec<f32>, Vec<f32>) {
        let plan = self.plan();
        let evaluated = omega::execute_plan_with_placements(
            &plan,
            &self.blocks(),
            &[],
            &[],
            &mut Vec::new(),
        )
        .expect("executes the rope pair");
        (
            evaluated.get(self.first).expect("first half").0.to_vec(),
            evaluated.get(self.second).expect("second half").0.to_vec(),
        )
    }

    fn run_two_dispatch(&self) -> (Vec<f32>, Vec<f32>) {
        temp_env::with_var(DISABLE_TWIN, Some("1"), || self.run_unplaced())
    }

    fn cpu_reference(&self) -> (Vec<f32>, Vec<f32>) {
        let blocks: Vec<&[f32]> = self.inputs.iter().map(Vec::as_slice).collect();
        let evaluated = evaluate(&self.program, &[], &blocks, &[self.first, self.second])
            .expect("CPU interpreter evaluates the rope pair");
        (
            evaluated.get(self.first).expect("first half").0.to_vec(),
            evaluated.get(self.second).expect("second half").0.to_vec(),
        )
    }
}

fn random_values(seed: u64, count: usize) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..count).map(|_| lcg.next_unit()).collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn max_abs_difference(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len(), "compared slices must be the same length");
    assert!(!left.is_empty(), "a zero-element comparison proves nothing");
    left.iter()
        .zip(right)
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f32::max)
}

#[proxima::test]
#[case::split_half_full_rotary(8, RopePairing::SplitHalf { pairs: PAIRS })]
#[case::split_half_partial_rotary(16, RopePairing::SplitHalf { pairs: PAIRS })]
#[case::adjacent_pairs(8, RopePairing::Interleaved)]
async fn the_metal_rope_pair_matches_the_cpu_reference_for_each_pairing(
    #[case] head_dim: u32,
    #[case] pairing: RopePairing,
) {
    let case = RopeCase::new(head_dim, pairing);
    let (first, second) = case.run_unplaced();

    let (cpu_first, cpu_second) = case.cpu_reference();
    assert!(max_abs_difference(&first, &cpu_first) <= 1e-6, "first half vs CPU");
    assert!(max_abs_difference(&second, &cpu_second) <= 1e-6, "second half vs CPU");
}

#[proxima::test]
#[case::split_half_full_rotary(8, RopePairing::SplitHalf { pairs: PAIRS })]
#[case::split_half_partial_rotary(16, RopePairing::SplitHalf { pairs: PAIRS })]
#[case::adjacent_pairs(8, RopePairing::Interleaved)]
async fn the_twin_dispatch_reproduces_the_two_dispatch_plan_bit_for_bit(
    #[case] head_dim: u32,
    #[case] pairing: RopePairing,
) {
    let case = RopeCase::new(head_dim, pairing);

    let (twin_first, twin_second) = case.run_unplaced();
    let (plain_first, plain_second) = case.run_two_dispatch();

    assert_eq!(bits(&twin_first), bits(&plain_first), "first rotated half");
    assert_eq!(bits(&twin_second), bits(&plain_second), "second rotated half");
    assert_ne!(
        bits(&twin_first),
        bits(&twin_second),
        "degenerate gate: the two halves must differ or a swapped output would pass"
    );
}

#[proxima::test]
#[case::split_half_at_prefill_width(1000, 16, 32, 64, RopePairing::SplitHalf { pairs: 32 })]
#[case::split_half_kv_heads_at_prefill_width(1000, 8, 32, 64, RopePairing::SplitHalf { pairs: 32 })]
#[case::adjacent_pairs_at_prefill_width(1000, 16, 32, 64, RopePairing::Interleaved)]
async fn a_prefill_wide_rope_pair_decodes_its_coordinates_exactly(
    #[case] sequence: u32,
    #[case] heads: u32,
    #[case] pairs: u32,
    #[case] head_dim: u32,
    #[case] pairing: RopePairing,
) {
    let case = RopeCase::sized(sequence, heads, pairs, head_dim, pairing);
    let (first, second) = case.run_unplaced();

    let (cpu_first, cpu_second) = case.cpu_reference();
    assert_eq!(first.len(), (sequence * heads * pairs) as usize, "first half element count");
    assert!(max_abs_difference(&first, &cpu_first) <= 1e-6, "first half vs CPU");
    assert!(max_abs_difference(&second, &cpu_second) <= 1e-6, "second half vs CPU");
}

#[proxima::test]
#[case::split_half(8, RopePairing::SplitHalf { pairs: PAIRS })]
#[case::adjacent_pairs(8, RopePairing::Interleaved)]
async fn both_halves_land_in_caller_placed_cache_buffers_at_nonzero_offsets(
    #[case] head_dim: u32,
    #[case] pairing: RopePairing,
) {
    let case = RopeCase::new(head_dim, pairing);
    let (expected_first, expected_second) = case.run_two_dispatch();
    let half_bytes = case.half_len() * size_of::<f32>();
    let buffer_bytes = PLACEMENT_ALIGNMENT * 4 + half_bytes;
    let first_cache = omega::allocate_placed_buffer(buffer_bytes).expect("first cache buffer");
    let second_cache = omega::allocate_placed_buffer(buffer_bytes).expect("second cache buffer");
    omega::zero_placed_buffer(&first_cache, buffer_bytes);
    omega::zero_placed_buffer(&second_cache, buffer_bytes);
    let plan = case.plan();

    omega::execute_plan_with_placements(
        &plan,
        &case.blocks(),
        &[],
        &[
            (case.first, &first_cache, PLACEMENT_ALIGNMENT),
            (case.second, &second_cache, PLACEMENT_ALIGNMENT * 2),
        ],
        &mut Vec::new(),
    )
    .expect("executes with both rotated halves placed");

    let placed_first =
        omega::read_placed_buffer_f32(&first_cache, PLACEMENT_ALIGNMENT, case.half_len());
    let placed_second =
        omega::read_placed_buffer_f32(&second_cache, PLACEMENT_ALIGNMENT * 2, case.half_len());
    assert_eq!(bits(&placed_first), bits(&expected_first), "first half in its cache");
    assert_eq!(bits(&placed_second), bits(&expected_second), "second half in its cache");
    let head = omega::read_placed_buffer_f32(&second_cache, 0, PLACEMENT_ALIGNMENT / 4);
    assert!(
        head.iter().all(|value| *value == 0.0),
        "bytes before the placed offset must stay untouched"
    );
}

#[proxima::test]
async fn only_the_second_half_placed_still_writes_the_first_into_the_arena() {
    let case = RopeCase::new(8, RopePairing::SplitHalf { pairs: PAIRS });
    let (expected_first, expected_second) = case.run_two_dispatch();
    let half_bytes = case.half_len() * size_of::<f32>();
    let cache = omega::allocate_placed_buffer(PLACEMENT_ALIGNMENT + half_bytes).expect("cache");
    let plan = case.plan();

    let evaluated = omega::execute_plan_with_placements(
        &plan,
        &case.blocks(),
        &[],
        &[(case.second, &cache, PLACEMENT_ALIGNMENT)],
        &mut Vec::new(),
    )
    .expect("executes with the second half placed");

    let first = evaluated.get(case.first).expect("first half read back").0.to_vec();
    let placed_second = omega::read_placed_buffer_f32(&cache, PLACEMENT_ALIGNMENT, case.half_len());
    assert_eq!(bits(&first), bits(&expected_first));
    assert_eq!(bits(&placed_second), bits(&expected_second));
}
