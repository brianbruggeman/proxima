//! Von Oswald et al. (ICML 2023), *Transformers Learn In-Context by Gradient
//! Descent* -- an explicit CONSTRUCTION, not a learned result: for context
//! tokens `e_i = (x_i, y_i)`, one **linear** self-attention layer with the
//! paper's stated query/key/value weights computes exactly one step of
//! gradient descent on the least-squares loss, expressed as the change it
//! induces in a query token's prediction.
//!
//! # Derivation, from the loss, checked against the paper's closed form
//!
//! Least-squares loss over the `N` context examples, weight `w`:
//!
//! ```text
//! L(w) = (1 / (2N)) * sum_i ( <w, x_i> - y_i )^2
//! ```
//!
//! Its gradient:
//!
//! ```text
//! grad L(w) = (1 / N) * sum_i ( <w, x_i> - y_i ) * x_i
//! ```
//!
//! One gradient-descent step from `w0`, learning rate `eta`:
//!
//! ```text
//! w1 = w0 - eta * grad L(w0)
//!    = w0 - (eta / N) * sum_i ( <w0, x_i> - y_i ) * x_i
//! ```
//!
//! The change this induces in the PREDICTION at any query `x_q` (`x_q` need
//! not be a context point -- it shares the feature space, nothing else):
//!
//! ```text
//! <w1, x_q> - <w0, x_q>
//!   = <w1 - w0, x_q>                                     (linearity of <_,x_q>)
//!   = <-(eta/N) * sum_i (<w0,x_i>-y_i) * x_i, x_q>
//!   = -(eta/N) * sum_i ( <w0, x_i> - y_i ) * <x_i, x_q>   (linearity again)
//! ```
//!
//! This is exactly the form the prompt states and the paper proves. There is
//! no discrepancy between this derivation and that form -- the two are the
//! same expression, so [`build_layer`] below is built directly against it.
//!
//! # Linear attention, never softmax
//!
//! Every step above is linear in `x_i`, `y_i`, and `w0` -- the whole
//! derivation is nothing but re-associating a sum. A softmax over the
//! `<x_i,x_q>` similarities would replace the uniform `1/N` averaging with a
//! data-dependent, normalized weighting, which is no longer the gradient of
//! `L` for ANY loss with a fixed quadratic form -- softmax is exactly the
//! nonlinearity the equivalence needs absent. Nothing in this file's
//! construction uses [`proxima_tensor::op::ScalarOp::Maximum`] as a running
//! max, an `Exponential`-then-normalize pair, or any other softmax
//! ingredient; every `ScalarOp` used below is `Multiply`, `Subtract`, or
//! `Add`, over `Op::Elementwise` and `Op::Reduce` only.
//!
//! # The construction as a graph
//!
//! One shape, reused for two purposes by varying only how many "queries"
//! ride through it (`q_count` below), never the graph itself:
//!
//! - leaves: `x` `[N,d]` (context features), `y` `[N]` (context labels),
//!   `w0` `[d]` (current weight), `queries` `[Q,d]`, and a rank-0
//!   `Constant` holding `-eta/N`.
//! - `pred[i] = sum_k x[i,k] * w0[k]` -- **an inner product per context
//!   token** (`Elementwise(Multiply)` over iteration `(i,k)` then
//!   `Reduce(Add)` over `k`).
//! - `resid[i] = pred[i] - y[i]` -- **a residual against the label**
//!   (`Elementwise(Subtract)`).
//! - `sim[i,q] = sum_k x[i,k] * queries[q,k]` -- one inner product per
//!   `(context token, query)` pair, batched (`Elementwise(Multiply)` over
//!   `(i,k,q)` then `Reduce(Add)` over `k`).
//! - `total[q] = sum_i resid[i] * sim[i,q]` -- **a weighted sum against the
//!   query** (`Elementwise(Multiply)` over `(i,q)` then `Reduce(Add)` over
//!   `i`).
//! - `delta[q] = total[q] * (-eta/N)`.
//!
//! With `Q = 1` and `queries = [x_q]`, `delta[0]` is exactly
//! `<w1,x_q> - <w0,x_q>` above -- [`single_layer_matches_one_gradient_descent_step`]
//! checks this against the independent reference.
//!
//! With `Q = d` and `queries = Identity(d)`, `<x_i, e_k> = x_i[k]`, so
//! `delta[k] = -(eta/N) * sum_i resid[i] * x_i[k]` is literally the
//! gradient-descent step's own `k`-th component -- adding it to `w0`
//! ([`append_weight_update`]) produces `w1` the vector, not just its
//! projection onto one query, which is what lets
//! [`iterating_the_layer_equals_iterating_gradient_descent`] feed a layer's
//! output back in as the next `w0`.
//!
//! Nothing here calls `proxima_autograd::differentiate`: this file measures
//! whether the FORWARD pass alone performs the optimization step, with no
//! gradient machinery invoked to produce it.
#![allow(clippy::unwrap_used, clippy::expect_used)]

extern crate alloc;

use proxima_tensor::cpu::evaluate_named;
use proxima_tensor::dtype::DType;
use proxima_tensor::error::TensorError;
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::op::{self, Extent, NodeId, Op, ReduceInit, ScalarOp};

fn leaf(program: &mut Vec<Op>, name: &str, shape: Vec<Extent>) -> NodeId {
    op::append(
        program,
        Op::Input {
            dtype: DType::Float32,
            shape,
            name: Some(name.into()),
        },
    )
}

fn elementwise(program: &mut Vec<Op>, body: ScalarOp, operands: Vec<(NodeId, IndexMap)>) -> NodeId {
    op::append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body,
            operands,
            name: None,
        },
    )
}

fn reduce_add(
    program: &mut Vec<Op>,
    operand: NodeId,
    in_map: IndexMap,
    out_map: IndexMap,
) -> NodeId {
    op::append(
        program,
        Op::Reduce(proxima_tensor::op::Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand,
            in_map,
            out_map,
            keep: proxima_tensor::op::Keep::Reduce,
            name: None,
        }),
    )
}

fn proj(iter_rank: u16, axes: &[u16]) -> IndexMap {
    IndexMap::Affine(map::projection(iter_rank, axes))
}

fn identity(rank: u16) -> IndexMap {
    proj(rank, &(0..rank).collect::<Vec<u16>>())
}

fn empty(rank: u16) -> IndexMap {
    proj(rank, &[])
}

/// The construction's own output: `delta[q]`, batched over `q_count`
/// queries. See the module doc's "construction as a graph" section for the
/// per-node derivation.
struct Layer {
    program: Vec<Op>,
    w0: NodeId,
    delta: NodeId,
}

/// Builds the linear-attention gradient-descent-step layer for `n` context
/// tokens of feature width `d`, batched over `q_count` query vectors, at
/// learning rate `eta`.
///
/// `transpose_pred_x_map` exists only for
/// [`transposed_pred_operand_map_is_caught_as_a_shape_mismatch`]: `true`
/// swaps the two axes of `x`'s own map in the `pred` node (`(i,k)` become
/// `(k,i)`), the deliberately-wrong construction that test proves gets
/// rejected rather than silently accepted.
fn build_layer(n: usize, d: usize, q_count: usize, eta: f32, transpose_pred_x_map: bool) -> Layer {
    let mut program = Vec::new();
    let x = leaf(
        &mut program,
        "x",
        vec![Extent::Static(n as u32), Extent::Static(d as u32)],
    );
    let y = leaf(&mut program, "y", vec![Extent::Static(n as u32)]);
    let w0 = leaf(&mut program, "w0", vec![Extent::Static(d as u32)]);
    let queries = leaf(
        &mut program,
        "queries",
        vec![Extent::Static(q_count as u32), Extent::Static(d as u32)],
    );
    let scale = op::append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: Vec::new(),
            value: -eta / n as f32,
        },
    );

    // iter (i, k): pred[i] = sum_k x[i,k] * w0[k]
    let x_pred_map = if transpose_pred_x_map {
        proj(2, &[1, 0])
    } else {
        proj(2, &[0, 1])
    };
    let pred_product = elementwise(
        &mut program,
        ScalarOp::Multiply,
        vec![(x, x_pred_map), (w0, proj(2, &[1]))],
    );
    let pred = reduce_add(&mut program, pred_product, identity(2), proj(2, &[0]));

    // resid[i] = pred[i] - y[i]
    let resid = elementwise(
        &mut program,
        ScalarOp::Subtract,
        vec![(pred, identity(1)), (y, identity(1))],
    );

    // iter (i, k, q): sim[i,q] = sum_k x[i,k] * queries[q,k]
    let sim_product = elementwise(
        &mut program,
        ScalarOp::Multiply,
        vec![(x, proj(3, &[0, 1])), (queries, proj(3, &[2, 1]))],
    );
    let sim = reduce_add(&mut program, sim_product, identity(3), proj(3, &[0, 2]));

    // iter (i, q): weighted[i,q] = resid[i] * sim[i,q]
    let weighted = elementwise(
        &mut program,
        ScalarOp::Multiply,
        vec![(resid, proj(2, &[0])), (sim, identity(2))],
    );

    // total[q] = sum_i weighted[i,q]
    let total = reduce_add(&mut program, weighted, identity(2), proj(2, &[1]));

    // delta[q] = total[q] * (-eta/n)
    let delta = elementwise(
        &mut program,
        ScalarOp::Multiply,
        vec![(total, identity(1)), (scale, empty(1))],
    );

    Layer { program, w0, delta }
}

/// Appends `w1[k] = w0[k] + delta[k]` -- only meaningful when `delta` was
/// produced with `q_count == d` (the weight-update reading of [`Layer`]),
/// never called for the single-query reading.
fn append_weight_update(program: &mut Vec<Op>, w0: NodeId, delta: NodeId) -> NodeId {
    elementwise(
        program,
        ScalarOp::Add,
        vec![(w0, identity(1)), (delta, identity(1))],
    )
}

fn flatten(rows: &[Vec<f32>]) -> Vec<f32> {
    rows.iter().flatten().copied().collect()
}

fn identity_matrix(dimension: usize) -> Vec<f32> {
    (0..dimension)
        .flat_map(|row| (0..dimension).map(move |column| if row == column { 1.0 } else { 0.0 }))
        .collect()
}

/// Independent reference: plain `f32` arithmetic over `Vec<f32>`, no
/// [`Op`], no [`proxima_autograd`] of any kind, deliberately not sharing a
/// single helper with [`build_layer`] above.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(left, right)| left * right)
        .sum()
}

fn one_gradient_step(w0: &[f32], xs: &[Vec<f32>], ys: &[f32], eta: f32) -> Vec<f32> {
    let context_count = xs.len() as f32;
    let residuals: Vec<f32> = xs
        .iter()
        .zip(ys.iter())
        .map(|(x_i, &y_i)| dot(w0, x_i) - y_i)
        .collect();
    let gradient: Vec<f32> = (0..w0.len())
        .map(|feature| {
            residuals
                .iter()
                .zip(xs.iter())
                .map(|(&residual, x_i)| residual * x_i[feature])
                .sum::<f32>()
                / context_count
        })
        .collect();
    w0.iter()
        .zip(gradient.iter())
        .map(|(w, g)| w - eta * g)
        .collect()
}

fn least_squares_loss(w: &[f32], xs: &[Vec<f32>], ys: &[f32]) -> f32 {
    let context_count = xs.len() as f32;
    let sum_of_squares: f32 = xs
        .iter()
        .zip(ys.iter())
        .map(|(x_i, &y_i)| {
            let residual = dot(w, x_i) - y_i;
            residual * residual
        })
        .sum();
    sum_of_squares / (2.0 * context_count)
}

fn within_tolerance(graph_value: f32, reference_value: f32, atol: f32, rtol: f32) -> bool {
    (graph_value - reference_value).abs() <= atol + rtol * reference_value.abs()
}

/// `d = 3`, `N = 5` -- deliberately non-square and non-symmetric (see the
/// module-level ROW-135 note in the crate's fixtures: a construction test
/// where every axis shares one extent cannot see a transposed index map,
/// because the map still type-checks under the wrong axis order). `x_q` is
/// distinct from every context row below (checked by inspection: no row
/// repeats `[1.5, -0.5, 2.0]`).
const CONTEXT_COUNT: usize = 5;
const FEATURE_DIM: usize = 3;
const LEARNING_RATE: f32 = 0.1;

fn context_features() -> Vec<Vec<f32>> {
    vec![
        vec![1.0, 2.0, -1.0],
        vec![0.5, -1.5, 2.0],
        vec![-2.0, 1.0, 0.5],
        vec![3.0, 0.0, -2.5],
        vec![-1.0, -2.0, 1.5],
    ]
}

fn context_labels() -> Vec<f32> {
    vec![2.0, -1.0, 0.5, 1.5, -0.5]
}

fn initial_weight() -> Vec<f32> {
    vec![0.3, -0.2, 0.1]
}

fn held_out_query() -> Vec<f32> {
    vec![1.5, -0.5, 2.0]
}

const SINGLE_STEP_ATOL: f32 = 1e-4;
const SINGLE_STEP_RTOL: f32 = 1e-3;

/// The headline claim: the construction's `delta[0]` for a single held-out
/// query equals `<w1,x_q> - <w0,x_q>` from one independently-computed GD
/// step, both sides accounting for the same 5 context tokens summed in a
/// different order (the graph via nested `Reduce`s, the reference via a
/// flat iterator fold) -- so the tolerance is tight (both are exact analytic
/// sums, not a finite-difference approximation) but not bit-exact.
#[proxima::test]
async fn single_layer_matches_one_gradient_descent_step() {
    let x_rows = context_features();
    let y = context_labels();
    let w0 = initial_weight();
    let x_q = held_out_query();
    let x_flat = flatten(&x_rows);

    let layer = build_layer(CONTEXT_COUNT, FEATURE_DIM, 1, LEARNING_RATE, false);
    let evaluated = evaluate_named(
        &layer.program,
        &[],
        &[("x", &x_flat), ("y", &y), ("w0", &w0), ("queries", &x_q)],
        &[layer.delta],
    )
    .expect("single-query layer lowers and evaluates");
    let graph_delta = evaluated.get(layer.delta).expect("delta requested").0[0];

    let w1 = one_gradient_step(&w0, &x_rows, &y, LEARNING_RATE);
    let reference_delta = dot(&w1, &x_q) - dot(&w0, &x_q);

    let relative_error = (graph_delta - reference_delta).abs() / reference_delta.abs();
    std::eprintln!(
        "single_layer_matches_one_gradient_descent_step: graph delta={graph_delta}, \
         reference <w1,x_q>-<w0,x_q>={reference_delta}, relative error={relative_error}"
    );
    assert!(
        within_tolerance(
            graph_delta,
            reference_delta,
            SINGLE_STEP_ATOL,
            SINGLE_STEP_RTOL
        ),
        "construction delta {graph_delta} disagreed with the independent GD-step reference {reference_delta} \
         beyond atol {SINGLE_STEP_ATOL} + rtol {SINGLE_STEP_RTOL}"
    );
}

const ITERATED_STEPS: usize = 4;
const ITERATION_ATOL: f32 = 1e-4;
const ITERATION_RTOL: f32 = 1e-3;

/// Feeds the layer's own output back in as `w0` for the next call, `L = 4`
/// times, and checks two things per step: the graph's `w1` (via `Q = d`,
/// `queries = Identity(d)`, so `delta` is the full gradient-step vector, not
/// one query's projection of it) matches an independently-iterated GD loop,
/// and the loss the reference computes after each step decreases
/// monotonically -- the property that makes "the forward pass is the
/// optimizer" a real claim about every step, not a one-off coincidence at
/// step one.
#[proxima::test]
async fn iterating_the_layer_equals_iterating_gradient_descent() {
    let x_rows = context_features();
    let y = context_labels();
    let x_flat = flatten(&x_rows);
    let queries_flat = identity_matrix(FEATURE_DIM);

    let mut layer_program = build_layer(
        CONTEXT_COUNT,
        FEATURE_DIM,
        FEATURE_DIM,
        LEARNING_RATE,
        false,
    );
    let w1 = append_weight_update(
        &mut layer_program.program,
        layer_program.w0,
        layer_program.delta,
    );

    let mut graph_weight = initial_weight();
    let mut reference_weight = initial_weight();
    let mut losses = vec![least_squares_loss(&reference_weight, &x_rows, &y)];
    std::eprintln!(
        "iterating_the_layer_equals_iterating_gradient_descent: step 0 loss={}",
        losses[0]
    );

    for step in 1..=ITERATED_STEPS {
        let evaluated = evaluate_named(
            &layer_program.program,
            &[],
            &[
                ("x", &x_flat),
                ("y", &y),
                ("w0", &graph_weight),
                ("queries", &queries_flat),
            ],
            &[w1],
        )
        .expect("weight-update layer lowers and evaluates");
        graph_weight = evaluated.get(w1).expect("w1 requested").0.to_vec();

        reference_weight = one_gradient_step(&reference_weight, &x_rows, &y, LEARNING_RATE);

        for (component, (&graph_value, &reference_value)) in
            graph_weight.iter().zip(reference_weight.iter()).enumerate()
        {
            assert!(
                within_tolerance(graph_value, reference_value, ITERATION_ATOL, ITERATION_RTOL),
                "step {step} component {component}: graph w1={graph_value} vs reference w1={reference_value}"
            );
        }

        let loss = least_squares_loss(&reference_weight, &x_rows, &y);
        std::eprintln!(
            "iterating_the_layer_equals_iterating_gradient_descent: step {step} graph_w={graph_weight:?} \
             reference_w={reference_weight:?} loss={loss}"
        );
        losses.push(loss);
    }

    for window in losses.windows(2) {
        assert!(
            window[1] < window[0],
            "loss must decrease monotonically across GD steps, saw {losses:?}"
        );
    }
}

/// Proves the construction test above actually discriminates: transposing
/// `x`'s own map in the `pred` node swaps which axis reads the context
/// index `i` (extent 5) and which reads the feature index `k` (extent 3).
/// Because `CONTEXT_COUNT != FEATURE_DIM`, this is not merely a wrong
/// answer -- it is ill-typed, and `shape::infer`'s `unify_iteration_space`
/// (`proxima-tensor/src/shape.rs:204-211`) rejects it outright as an
/// [`TensorError::ExtentMismatch`] before a single multiply runs. That is a
/// STRONGER discrimination than "numbers slightly different": a transposed
/// map here cannot silently produce a plausible-looking wrong answer, it
/// cannot produce any answer at all. Reverting the flag confirms the
/// correct construction evaluates cleanly with the SAME data.
#[proxima::test]
async fn transposed_pred_operand_map_is_caught_as_a_shape_mismatch() {
    let x_rows = context_features();
    let y = context_labels();
    let w0 = initial_weight();
    let x_q = held_out_query();
    let x_flat = flatten(&x_rows);

    let transposed = build_layer(CONTEXT_COUNT, FEATURE_DIM, 1, LEARNING_RATE, true);
    let transposed_result = evaluate_named(
        &transposed.program,
        &[],
        &[("x", &x_flat), ("y", &y), ("w0", &w0), ("queries", &x_q)],
        &[transposed.delta],
    );
    std::eprintln!("transposed pred operand map result: {transposed_result:?}");
    assert!(
        matches!(transposed_result, Err(TensorError::ExtentMismatch { .. })),
        "a transposed operand map on non-square dims (N={CONTEXT_COUNT} != d={FEATURE_DIM}) must be rejected as an \
         extent mismatch, not silently accepted: {transposed_result:?}"
    );

    let reverted = build_layer(CONTEXT_COUNT, FEATURE_DIM, 1, LEARNING_RATE, false);
    let reverted_result = evaluate_named(
        &reverted.program,
        &[],
        &[("x", &x_flat), ("y", &y), ("w0", &w0), ("queries", &x_q)],
        &[reverted.delta],
    )
    .expect("reverted (correct) map must evaluate cleanly with the identical data");
    let reverted_delta = reverted_result
        .get(reverted.delta)
        .expect("delta requested")
        .0[0];
    std::eprintln!("reverted pred operand map result: delta={reverted_delta}");
}
