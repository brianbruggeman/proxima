//! The attention sink's worked example, shared by the CPU literal test
//! (`spec::tests::attention_sinks`) and the Metal parity test in `omega`, so
//! both backends validate ONE hand-derived literal instead of each other
//! (SPEC `serve-flint-through-omega`'s own "one hand-worked literal, not
//! against each other").
//!
//! Gated exactly like `crate::test_support`: compiled for this crate's own
//! tests and for a downstream crate that asks for the `test-support` feature,
//! never in a shipping build.

use super::*;

/// The CPU evaluator's sink term against a literal computed by hand from
/// the definition, not from the graph. The incumbent is llama.cpp
/// `cd959bae9`'s `ggml_compute_forward_soft_max_f32` (`ggml/src/ggml-cpu/
/// ops.cpp:5670-5680`): each head's sink joins the row maximum
/// (`max = MAX(max, sk[i02])`) and its own `exp(sink - max)` joins the
/// denominator, contributing no value.
///
/// The lowered shape is the shared two-block combine every cached engine
/// calls (`append_local_block_and_combine`): one query row against two
/// cached keys and one local key, TWO key heads with two query heads each,
/// value head width 2. Two key heads is what makes the sink's own address
/// `group*u + g` observable: with one key head `u` is always 0 and a
/// dropped coefficient is invisible. Head 0's sink (2.0) dominates its
/// scores, head 3's (3.0) dominates by more, heads 1 and 2 trail theirs,
/// so both sides of the incumbent's `MAX(max, sink)` run.
const SINK_KV_HEADS: u32 = 2;
const SINK_GROUP: u32 = 2;
pub const SINK_QUERY_HEADS: u32 = SINK_KV_HEADS * SINK_GROUP;
const INV_SQRT_HEAD_DIM: f32 = 0.5;
/// cached block scores at `s = 0`, `[t][u][g]`
const CACHED_SCORES: [[[f32; 2]; 2]; 2] = [
    [[0.5, -0.25], [0.125, 0.375]],
    [[-1.0, 0.75], [0.25, -0.5]],
];
/// `[u][g][i]`
const Q_EVEN: [[[f32; 2]; 2]; 2] = [
    [[0.5, -0.25], [1.0, 0.25]],
    [[0.25, 0.75], [-0.5, 1.0]],
];
const Q_ODD: [[[f32; 2]; 2]; 2] = [
    [[0.75, 0.5], [-0.5, 1.5]],
    [[1.25, -0.75], [0.25, 0.5]],
];
/// `[u][i]`
const K_EVEN: [[f32; 2]; 2] = [[1.0, 0.5], [-0.5, 1.5]];
const K_ODD: [[f32; 2]; 2] = [[-0.5, 0.25], [0.75, -1.0]];
/// `[t][u][d]`
const V_CACHE: [[[f32; 2]; 2]; 2] = [
    [[1.0, 0.25], [-0.25, 0.5]],
    [[-0.5, 0.75], [0.75, -1.0]],
];
/// `[u][d]`
const V_NEW: [[f32; 2]; 2] = [[0.5, -0.25], [0.125, 0.875]];
/// one learned scalar per query head, `[h]` with `h = group*u + g`
pub const SINKS: [f32; 4] = [2.0, -1.0, 0.5, 3.0];
pub const SINK_LEAF_NAME: &str = "blk.0.attn_sinks.weight";

pub fn sink_combine_program(attention_sink: bool) -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let query = |program: &mut Vec<Op>, name: &str| {
        input_leaf(
            program,
            DType::FLOAT32,
            alloc::vec![
                Extent::Static(1),
                Extent::Static(SINK_KV_HEADS),
                Extent::Static(SINK_GROUP),
                Extent::Static(2)
            ],
            name,
        )
    };
    let key_head = |program: &mut Vec<Op>, name: &str| {
        input_leaf(
            program,
            DType::FLOAT32,
            alloc::vec![
                Extent::Static(1),
                Extent::Static(SINK_KV_HEADS),
                Extent::Static(2)
            ],
            name,
        )
    };
    let q_even = query(&mut program, "q_even");
    let q_odd = query(&mut program, "q_odd");
    let k_even = key_head(&mut program, "k_even");
    let k_odd = key_head(&mut program, "k_odd");
    let v_new = key_head(&mut program, "v_new");
    let v_cache = input_leaf(
        &mut program,
        DType::FLOAT32,
        alloc::vec![
            Extent::Static(2),
            Extent::Static(SINK_KV_HEADS),
            Extent::Static(2)
        ],
        "v_cache",
    );
    let cached_scores = input_leaf(
        &mut program,
        DType::FLOAT32,
        alloc::vec![
            Extent::Static(1),
            Extent::Static(2),
            Extent::Static(SINK_KV_HEADS),
            Extent::Static(SINK_GROUP)
        ],
        "cached_scores",
    );
    let is_future = input_leaf(
        &mut program,
        DType::FLOAT32,
        alloc::vec![Extent::Static(1), Extent::Static(1)],
        "is_future",
    );
    let sink = attention_sink.then(|| sink_leaf(&mut program));
    let inv_sqrt_head_dim = scalar_constant(&mut program, INV_SQRT_HEAD_DIM);
    let neg_infinity = scalar_constant(&mut program, f32::NEG_INFINITY);
    let (attended, _) = super::two_block_attention::append_local_block_and_combine(
        &mut program,
        q_even,
        q_odd,
        k_even,
        k_odd,
        v_new,
        v_cache,
        inv_sqrt_head_dim,
        cached_scores,
        is_future,
        Some(neg_infinity),
        sink,
        SINK_GROUP,
    )
    .expect("the two-block combine with a sink lowers");
    (program, attended)
}

fn flatten2(values: &[[f32; 2]; 2]) -> Vec<f32> {
    values.iter().flatten().copied().collect()
}

fn flatten3(values: &[[[f32; 2]; 2]; 2]) -> Vec<f32> {
    values
        .iter()
        .flat_map(|outer| outer.iter().flatten().copied())
        .collect()
}

/// Every leaf the combine reads, by the name it is declared under; the
/// sink leaf is bound only when the knob is on, which is the contract the
/// flag-off graph keeps.
pub fn sink_combine_inputs(attention_sink: bool) -> Vec<(&'static str, Vec<f32>)> {
    let mut named = alloc::vec![
        ("q_even", flatten3(&Q_EVEN)),
        ("q_odd", flatten3(&Q_ODD)),
        ("k_even", flatten2(&K_EVEN)),
        ("k_odd", flatten2(&K_ODD)),
        ("v_new", flatten2(&V_NEW)),
        ("v_cache", flatten3(&V_CACHE)),
        ("cached_scores", flatten3(&CACHED_SCORES)),
        ("is_future", alloc::vec![0.0]),
    ];
    if attention_sink {
        named.push((SINK_LEAF_NAME, SINKS.to_vec()));
    }
    named
}

pub fn sink_leaf(program: &mut Vec<Op>) -> NodeId {
    input_leaf(
        program,
        DType::FLOAT32,
        alloc::vec![Extent::Static(SINK_QUERY_HEADS)],
        SINK_LEAF_NAME,
    )
}

/// The literal, worked by hand in f32 in the order the combine sums
/// (cached block, then the local key, then the sink addend), one query
/// head at a time with its sink read at `group*u + g`.
pub fn hand_worked_attended(sink: Option<[f32; 4]>) -> Vec<f32> {
    let mut attended = Vec::new();
    for key_head in 0..SINK_KV_HEADS as usize {
        for group_head in 0..SINK_GROUP as usize {
            let query_head = key_head * SINK_GROUP as usize + group_head;
            let new_score = INV_SQRT_HEAD_DIM
                * ((Q_EVEN[key_head][group_head][0] * K_EVEN[key_head][0]
                    + Q_EVEN[key_head][group_head][1] * K_EVEN[key_head][1])
                    + (Q_ODD[key_head][group_head][0] * K_ODD[key_head][0]
                        + Q_ODD[key_head][group_head][1] * K_ODD[key_head][1]));
            let mut scores = alloc::vec![
                CACHED_SCORES[0][key_head][group_head],
                CACHED_SCORES[1][key_head][group_head],
            ];
            scores.push(new_score);
            let mut maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if let Some(sink) = sink {
                maximum = maximum.max(sink[query_head]);
            }
            let weights: Vec<f32> =
                scores.iter().map(|score| (score - maximum).exp()).collect();
            let sum_cached = weights[0] + weights[1];
            let sum_new = weights[2];
            let mut denominator = sum_cached + sum_new;
            if let Some(sink) = sink {
                denominator += (sink[query_head] - maximum).exp();
            }
            let cached_d0 = weights[0] * V_CACHE[0][key_head][0]
                + weights[1] * V_CACHE[1][key_head][0];
            let cached_d1 = weights[0] * V_CACHE[0][key_head][1]
                + weights[1] * V_CACHE[1][key_head][1];
            attended.push((cached_d0 + weights[2] * V_NEW[key_head][0]) / denominator);
            attended.push((cached_d1 + weights[2] * V_NEW[key_head][1]) / denominator);
        }
    }
    attended
}

/// The one comparison both tests run, so the control exercises the
/// assertion rather than a copy of it.
pub fn attended_agrees(actual: &[f32], expected: &[f32]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() <= 1e-6 + 1e-6 * expected.abs())
}
