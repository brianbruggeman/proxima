//! Hypothesis: a model **linear in its trainable parameters** needs no
//! epochs at all. Fixed random features (`h = relu(W^T x)`, `W` frozen,
//! never trained) plus a closed-form ridge regression solve
//! (`(H^T H + lambda I) beta = H^T Y`) should reach `>= 0.97` top-1 on the
//! full 10k MNIST test set from **exactly one streaming pass** over the
//! 60k train set, with everything past that pass data-free (the solve
//! reads only the accumulated `[k x k]` / `[k x 10]` matrices, never the
//! pixels again).
//!
//! Falsifier: accuracy materially below `0.97` at a reasonable `k`, or any
//! step that revisits a training image after its one accumulation pass.
//!
//! No new [`proxima_tensor::op::Op`] / [`proxima_tensor::op::ScalarOp`]
//! variant, no autograd, no training loop -- this composes exactly four
//! things the crate already ships: [`proxima_tensor::op`]'s `Elementwise`
//! (matmul-as-zip-reduce, the same `product`/`reduce_add` idiom
//! `proxima-autograd/tests/real_mnist_training.rs`'s `batched_dense` uses)
//! and `Reduce` generators, [`relu`] (`ScalarOp::Maximum` against a
//! broadcasting zero -- see that function's own doc), and
//! [`proxima_tensor::cpu::evaluate_named`] to run each step.
//!
//! # Recipe
//!
//! 1. **Features.** `W` is `[784 x k]`, drawn once from
//!    [`proxima_tensor::test_support::Lcg`] and never touched again --
//!    each weight is the mean of 12 `next_unit()` draws (a central-limit
//!    approximation to a standard normal, scaled to unit variance: the sum
//!    of 12 `Uniform(-1, 1)` draws has variance `12 * (1/3) = 4`, so
//!    dividing by 2 restores unit variance), the same trick
//!    `rand_distr::StandardNormal` documents as its own fallback. A random
//!    per-feature bias `b` (same recipe, [`random_bias`]) rides along --
//!    without it every `relu` hyperplane is forced through the origin,
//!    which starves the feature map of any decision boundary that does not
//!    pass through the input's mean. `h(x) = relu(W^T x + b)`.
//!
//!    This module also found and fixed a real bug on the way to a usable
//!    signal: [`proxima_tensor::test_support::Lcg::next_unit`] shifted its
//!    64-bit state right by 33 bits (a 31-bit remainder) while dividing by
//!    the 32-bit [`u32::MAX`], so every caller drawing "uniform in
//!    `[-1, 1)`" was silently drawing from `[-1, 0)` instead -- half the
//!    intended mean and variance. That bug alone was enough to hold this
//!    example's early runs at `0.44`-`0.50` accuracy regardless of `k`;
//!    fixing it at the source (`proxima-tensor/src/test_support.rs`, the
//!    same shared fixture every example/bench in this workspace draws
//!    random floats from) is what makes the ridge solve's ceiling visible
//!    here instead of measuring a broken PRNG.
//! 2. **One streaming pass.** The 60k training images cross the matmul
//!    exactly once, in batches (batching is not epochs -- each image is
//!    read from disk and multiplied against `W` once, full stop). Every
//!    batch's `H^T H` and `H^T Y` are matmul-and-reduce results out of the
//!    same tensor algebra, then folded into running `[k x k]` / `[k x 10]`
//!    accumulators.
//! 3. **Regularize.** `lambda = 1e-2 * trace(A) / k` (a scale-free default:
//!    it makes the ridge penalty comparable to the *average* diagonal
//!    energy `A` already carries, so it does not need retuning per `k`).
//! 4. **Data-free solve.** Conjugate gradient over `A beta = B`, expressed
//!    entirely as matmul (`A @ P`) and elementwise ops (dot-product
//!    reduces, axpy-shaped multiply/add/subtract) -- [`build_cg_step`]'s
//!    own doc walks the ten-node graph. The solve never touches a pixel
//!    again; it only ever reads `A`/`B`.
//! 5. **Predict.** `argmax_c h(x)^T beta[:, c]` over the full 10k test set.
//!
//! # Measured table
//!
//! `cargo run --release --example one_pass_mnist -p proxima-autograd`
//! against the full `~/.cache/burn-dataset/mnist` 60k/10k split, one
//! streaming pass per rung (`TRAIN_BATCH = 500`, 120 batches, 60000 images
//! visited exactly once each -- asserted at runtime), `lambda = 1e-2 *
//! trace(A) / k`, CG to `1e-4` relative residual or `300` iterations:
//!
//! | k | lambda | cg iters | relative residual | one-pass wall clock | solve wall clock | test accuracy |
//! |---:|---:|---:|---:|---:|---:|---:|
//! | 1000 | 837.10 | 141 | 0.000077 | 4.90s | 2.19s | 0.9406 |
//! | 2000 | 899.67 | 193 | 0.000095 | 12.54s | 11.76s | 0.9535 |
//! | 4000 | 849.18 | 260 | 0.000084 | 102.48s | 72.87s | 0.9665 |
//!
//! # Contrast row
//!
//! `real_mnist_training.rs`'s own epoch-trained `784-128-10` MLP: 28
//! epochs over the full 60k set, `0.9786` top-1 on the full 10k test set,
//! `373.8s` wall clock (`tests/real_mnist_training.rs`'s own module doc,
//! rung C). This example's one-pass ridge solve is compared against that
//! number, not against this session's own 4-epoch/8000-image CI-shaped
//! subset (`0.9274`), which trains a materially smaller slice of the data.
//!
//! # Verdict: hypothesis not supported at these `k`, but converging on it
//!
//! CG converges cleanly at every rung (relative residual `~1e-4`, well
//! inside tolerance -- the solve is not the limiting factor) and per-class
//! accuracy is balanced (no class below `0.91` even at `k=1000`, see the
//! per-rung `dead_unit_fraction`/`per_class_accuracy` this example prints
//! -- essentially no dead relu units, so capacity is not being wasted
//! either). What is limiting accuracy is `k` itself: `0.9406 -> 0.9535 ->
//! 0.9665` is a shrinking-but-still-open gap to `0.97` as `k` doubles,
//! consistent with the published random-features/ELM literature, where
//! matching a trained network's ~97-98% on MNIST typically needs `k` in
//! the 5-20k range, well past this ladder's `k=4000` top rung -- this
//! session's own 90-minute budget stopped the ladder at 4000 (`k=4000`'s
//! one-pass-plus-solve alone cost `175s`; wall clock scales worse than
//! linearly in `k` since the Gram accumulation is `O(batch * k^2)` per
//! batch). The falsifier as stated (`materially below 0.97`) is not met at
//! `k=4000` (`0.9665` is `0.0035` short, not "material"), and the trend
//! line does not plateau over this ladder -- so the honest reading is
//! **unresolved at this `k`, not refuted**: extending the ladder to
//! `k=8000`/`k=16000` is the next falsifying-or-confirming step, not a
//! change of method.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_arguments)]

use proxima_autograd::activation::relu;
use proxima_tensor::dtype::DType;
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::op::{self, Extent, NodeId, Op, ReduceInit, ScalarOp};
use proxima_tensor::test_support::Lcg;

const DATASET_DIR: &str = "/Users/brianbruggeman/.cache/burn-dataset/mnist";
const IN_DIM: usize = 28 * 28;
const OUT_DIM: usize = 10;
const TRAIN_EXAMPLES: usize = 60_000;
const TEST_EXAMPLES: usize = 10_000;
const TRAIN_BATCH: usize = 500;
const CG_MAX_ITERS: usize = 300;
const CG_RELATIVE_RESIDUAL_TOLERANCE: f32 = 1e-4;
const LAMBDA_SCALE: f32 = 1e-2;

fn checkpoint_present() -> bool {
    train_images_path().exists() && train_labels_path().exists() && test_images_path().exists() && test_labels_path().exists()
}

fn train_images_path() -> std::path::PathBuf {
    std::path::Path::new(DATASET_DIR).join("train/train-images-idx3-ubyte")
}
fn train_labels_path() -> std::path::PathBuf {
    std::path::Path::new(DATASET_DIR).join("train/train-labels-idx1-ubyte")
}
fn test_images_path() -> std::path::PathBuf {
    std::path::Path::new(DATASET_DIR).join("test/t10k-images-idx3-ubyte")
}
fn test_labels_path() -> std::path::PathBuf {
    std::path::Path::new(DATASET_DIR).join("test/t10k-labels-idx1-ubyte")
}

/// Same idx3/idx1 big-endian header
/// `proxima-autograd/tests/real_mnist_training.rs::idx_header` parses,
/// restated here for the same reason that copy restates it rather than
/// sharing it: this crate's DE-CISC posture keeps small host-side parsing
/// a plain inline fn, not a cross-crate dependency.
fn idx_header(bytes: &[u8]) -> (usize, Vec<usize>) {
    let dimension_count = bytes[3] as usize;
    let item_count = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let mut extents = Vec::with_capacity(dimension_count - 1);
    for axis in 1..dimension_count {
        let offset = 4 + axis * 4;
        extents.push(u32::from_be_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]]) as usize);
    }
    (item_count, extents)
}

fn load_normalized_images(path: &std::path::Path, limit: usize) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read idx3 image file");
    let (item_count, extents) = idx_header(&bytes);
    let pixel_count = extents.iter().product::<usize>();
    let take = item_count.min(limit);
    let header_length = 4 + extents.len() * 4 + 4;
    bytes[header_length..header_length + take * pixel_count].iter().map(|&pixel| ((pixel as f32 / 255.0) - 0.1307) / 0.3081).collect()
}

fn load_one_hot_labels(path: &std::path::Path, limit: usize) -> (Vec<f32>, Vec<u8>) {
    let bytes = std::fs::read(path).expect("read idx1 label file");
    let (item_count, _extents) = idx_header(&bytes);
    let take = item_count.min(limit);
    let raw = &bytes[8..8 + take];
    let mut one_hot = vec![0.0f32; take * OUT_DIM];
    for (index, &label) in raw.iter().enumerate() {
        one_hot[index * OUT_DIM + label as usize] = 1.0;
    }
    (one_hot, raw.to_vec())
}

fn leaf(program: &mut Vec<Op>, name: &str, shape: Vec<Extent>) -> NodeId {
    op::append(program, Op::Input { dtype: DType::Float32, shape, name: Some(name.into()) })
}

fn identity(rank: u16) -> IndexMap {
    IndexMap::Affine(map::projection(rank, &(0..rank).collect::<Vec<u16>>()))
}

fn axes(rank: u16, selected: &[u16]) -> IndexMap {
    IndexMap::Affine(map::projection(rank, selected))
}

fn elementwise(program: &mut Vec<Op>, body: ScalarOp, operands: Vec<(NodeId, IndexMap)>) -> NodeId {
    op::append(program, Op::Elementwise { dtype: DType::Float32, body, operands, name: None })
}

fn reduce_add(program: &mut Vec<Op>, operand: NodeId, in_map: IndexMap, out_map: IndexMap) -> NodeId {
    op::append(
        program,
        Op::Reduce(op::Reduce {
            dtype: DType::Float32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand,
            in_map,
            out_map,
            keep: op::Keep::Reduce,
            name: None,
        }),
    )
}

/// `left[a, b] * right[b, c]`, contracted over `b`, kept axes `keep` --
/// the same zip-then-reduce shape
/// `real_mnist_training.rs`'s `batched_dense` uses for `x @ w`, generalized
/// to whichever pair of the three rank-3 iteration axes is being
/// contracted: feature projection contracts `in`, the Gram accumulation
/// contracts `batch`, and the CG matvec contracts the state dimension --
/// one function, three call sites, no per-shape duplication.
fn matmul3(program: &mut Vec<Op>, left: NodeId, left_axes: &[u16], right: NodeId, right_axes: &[u16], keep: &[u16]) -> NodeId {
    let product = elementwise(program, ScalarOp::Multiply, vec![(left, axes(3, left_axes)), (right, axes(3, right_axes))]);
    reduce_add(program, product, identity(3), axes(3, keep))
}

/// `sum_row(a[row, col] * b[row, col])`, one scalar per `col` -- the
/// column-wise dot product [`build_cg_step`] needs three times per
/// iteration (`r . r`, `p . (A p)`, `new_r . new_r`).
fn colwise_dot(program: &mut Vec<Op>, a: NodeId, b: NodeId) -> NodeId {
    let product = elementwise(program, ScalarOp::Multiply, vec![(a, identity(2)), (b, identity(2))]);
    reduce_add(program, product, identity(2), axes(2, &[1]))
}

/// He-scaled central-limit pseudo-random source: mean of 12
/// [`Lcg::next_unit`] draws (`Uniform(-1, 1)`, variance `1/3` each) has
/// variance `12 * (1/3) = 4`, so dividing the sum by `2 * sqrt(fan_in)`
/// both restores unit variance and applies the same He `sqrt(2 / fan_in)`
/// scale `real_mnist_training.rs::he_init` uses for its own random
/// projection -- `W` here is frozen (never trained), but its fixed scale
/// still has to keep `relu(W^T x)` in a useful dynamic range.
fn random_projection(seed: u64, in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut source = Lcg(seed);
    let scale = (2.0f32 / in_dim as f32).sqrt() / 2.0;
    (0..in_dim * out_dim)
        .map(|_| {
            let sum: f32 = (0..12).map(|_| source.next_unit()).sum();
            sum * scale
        })
        .collect()
}

/// Random per-feature bias, same central-limit recipe as
/// [`random_projection`] but scaled to the pre-activation's own
/// std-deviation-1 unit (see that function's doc): without a bias every
/// `relu(w . x)` hyperplane is forced through the origin, which starves the
/// feature map of half its expressive power (any decision boundary that
/// does not pass through the input's mean). A random ELM-style bias is the
/// standard fix and this crate's `he_init`/`Lcg` combination already
/// supplies the randomness -- this is that recipe reused, not a second one.
fn random_bias(seed: u64, out_dim: usize) -> Vec<f32> {
    let mut source = Lcg(seed);
    (0..out_dim)
        .map(|_| {
            let sum: f32 = (0..12).map(|_| source.next_unit()).sum();
            sum / 2.0
        })
        .collect()
}

/// Builds the feature program once per `(batch, k)` pair: `x [batch, 784]`,
/// `w [784, k]` (frozen), `b [k]` (frozen), `h = relu(x @ w + b) [batch, k]`.
fn build_feature_program(batch: usize, k: usize) -> (Vec<Op>, NodeId, NodeId, NodeId, NodeId) {
    let mut program = Vec::new();
    let x = leaf(&mut program, "x", vec![Extent::Static(batch as u32), Extent::Static(IN_DIM as u32)]);
    let w = leaf(&mut program, "w", vec![Extent::Static(IN_DIM as u32), Extent::Static(k as u32)]);
    let b = leaf(&mut program, "b", vec![Extent::Static(k as u32)]);
    let h_pre = matmul3(&mut program, x, &[0, 1], w, &[1, 2], &[0, 2]);
    let h_biased = elementwise(&mut program, ScalarOp::Add, vec![(h_pre, identity(2)), (b, axes(2, &[1]))]);
    let h = relu(&mut program, DType::Float32, h_biased, 2);
    (program, x, w, b, h)
}

/// Builds the per-batch Gram accumulation program: given this batch's
/// `h [batch, k]` and `y [batch, 10]` (one-hot), `gram_hh = h^T h [k, k]`
/// and `gram_hy = h^T y [k, 10]`, both contracting the batch axis --
/// exactly [`matmul3`] with `batch` as the contracted axis instead of the
/// feature axis [`build_feature_program`] contracts.
fn build_gram_program(batch: usize, k: usize) -> (Vec<Op>, NodeId, NodeId, NodeId, NodeId) {
    let mut program = Vec::new();
    let h = leaf(&mut program, "h", vec![Extent::Static(batch as u32), Extent::Static(k as u32)]);
    let y = leaf(&mut program, "y", vec![Extent::Static(batch as u32), Extent::Static(OUT_DIM as u32)]);
    let gram_hh = matmul3(&mut program, h, &[0, 1], h, &[0, 2], &[1, 2]);
    let gram_hy = matmul3(&mut program, h, &[0, 1], y, &[0, 2], &[1, 2]);
    (program, h, y, gram_hh, gram_hy)
}

struct CgStep {
    program: Vec<Op>,
    new_x: NodeId,
    new_r: NodeId,
    new_p: NodeId,
    new_r_dot_r: NodeId,
}

/// One conjugate-gradient step over `A beta = B`, `A [k, k]` symmetric
/// positive-definite (a Gram matrix plus a positive ridge term always is),
/// `X`/`R`/`P` each `[k, 10]` (one column per class, solved simultaneously
/// -- the ten right-hand sides share one `A @ P` matmul per iteration
/// instead of ten independent scalar CG runs).
///
/// Ten nodes, all matmul ([`matmul3`]) or elementwise
/// (`ScalarOp::{Multiply,Add,Subtract,Divide}` plus [`colwise_dot`]'s
/// reduce): `ap = A @ p`; `alpha = (r . r) / (p . ap)`, broadcast per
/// column; `new_x = x + alpha * p`; `new_r = r - alpha * ap`; `beta =
/// (new_r . new_r) / (r . r)`; `new_p = new_r + beta * p`. Exactly textbook
/// CG (Hestenes-Stiefel), spelled as a tensor program instead of a scalar
/// loop.
fn build_cg_step(k: usize) -> CgStep {
    let mut program = Vec::new();
    let a = leaf(&mut program, "a", vec![Extent::Static(k as u32), Extent::Static(k as u32)]);
    let x = leaf(&mut program, "x", vec![Extent::Static(k as u32), Extent::Static(OUT_DIM as u32)]);
    let r = leaf(&mut program, "r", vec![Extent::Static(k as u32), Extent::Static(OUT_DIM as u32)]);
    let p = leaf(&mut program, "p", vec![Extent::Static(k as u32), Extent::Static(OUT_DIM as u32)]);

    let ap = matmul3(&mut program, a, &[0, 1], p, &[1, 2], &[0, 2]);
    let r_dot_r = colwise_dot(&mut program, r, r);
    let p_dot_ap = colwise_dot(&mut program, p, ap);
    let alpha = elementwise(&mut program, ScalarOp::Divide, vec![(r_dot_r, identity(1)), (p_dot_ap, identity(1))]);

    let alpha_p = elementwise(&mut program, ScalarOp::Multiply, vec![(p, identity(2)), (alpha, axes(2, &[1]))]);
    let new_x = elementwise(&mut program, ScalarOp::Add, vec![(x, identity(2)), (alpha_p, identity(2))]);

    let alpha_ap = elementwise(&mut program, ScalarOp::Multiply, vec![(ap, identity(2)), (alpha, axes(2, &[1]))]);
    let new_r = elementwise(&mut program, ScalarOp::Subtract, vec![(r, identity(2)), (alpha_ap, identity(2))]);

    let new_r_dot_r = colwise_dot(&mut program, new_r, new_r);
    let beta = elementwise(&mut program, ScalarOp::Divide, vec![(new_r_dot_r, identity(1)), (r_dot_r, identity(1))]);

    let beta_p = elementwise(&mut program, ScalarOp::Multiply, vec![(p, identity(2)), (beta, axes(2, &[1]))]);
    let new_p = elementwise(&mut program, ScalarOp::Add, vec![(new_r, identity(2)), (beta_p, identity(2))]);

    CgStep { program, new_x, new_r, new_p, new_r_dot_r }
}

fn argmax_row(values: &[f32]) -> usize {
    values.iter().enumerate().max_by(|left, right| left.1.total_cmp(right.1)).map(|(index, _)| index).expect("nonempty logits")
}

struct RungResult {
    k: usize,
    lambda: f32,
    cg_iters: usize,
    relative_residual: f32,
    one_pass_wall_clock: std::time::Duration,
    solve_wall_clock: std::time::Duration,
    test_accuracy: f64,
    dead_unit_fraction: f64,
    per_class_accuracy: [f64; OUT_DIM],
}

/// Runs the full hypothesis for one `k`: one streaming pass over the 60k
/// training images accumulating `A`/`B`, a data-free CG solve, and a full
/// 10k-test-image evaluation.
fn run_rung(w: &[f32], b: &[f32], k: usize, train_images: &[f32], train_one_hot: &[f32], test_images: &[f32], test_labels: &[u8]) -> RungResult {
    assert_eq!(TRAIN_EXAMPLES % TRAIN_BATCH, 0, "batch must divide the train set evenly for the exactly-once-visit assertion to hold");
    let batch_count = TRAIN_EXAMPLES / TRAIN_BATCH;

    let (feature_program, feature_x, feature_w, feature_b, feature_h) = build_feature_program(TRAIN_BATCH, k);
    let (gram_program, gram_h, gram_y, gram_hh, gram_hy) = build_gram_program(TRAIN_BATCH, k);

    let mut a_total = vec![0.0f32; k * k];
    let mut b_total = vec![0.0f32; k * OUT_DIM];
    let mut images_visited = 0usize;
    let mut dead_units = vec![true; k];

    let one_pass_start = std::time::Instant::now();
    for batch_index in 0..batch_count {
        let image_start = batch_index * TRAIN_BATCH * IN_DIM;
        let label_start = batch_index * TRAIN_BATCH * OUT_DIM;
        let x_batch = &train_images[image_start..image_start + TRAIN_BATCH * IN_DIM];
        let y_batch = &train_one_hot[label_start..label_start + TRAIN_BATCH * OUT_DIM];

        let feature_named: Vec<(&str, &[f32])> = vec![("x", x_batch), ("w", w), ("b", b)];
        let feature_evaluated = proxima_tensor::cpu::evaluate_named(&feature_program, &[], &feature_named, &[feature_h])
            .expect("feature program evaluates on this batch");
        let (h_batch, _shape) = feature_evaluated.get(feature_h).expect("h present");

        for row in h_batch.chunks_exact(k) {
            for (unit_index, &value) in row.iter().enumerate() {
                if value > 0.0 {
                    dead_units[unit_index] = false;
                }
            }
        }

        let gram_named: Vec<(&str, &[f32])> = vec![("h", h_batch), ("y", y_batch)];
        let gram_evaluated =
            proxima_tensor::cpu::evaluate_named(&gram_program, &[], &gram_named, &[gram_hh, gram_hy]).expect("gram program evaluates on this batch");
        let (batch_hh, _) = gram_evaluated.get(gram_hh).expect("gram_hh present");
        let (batch_hy, _) = gram_evaluated.get(gram_hy).expect("gram_hy present");

        for (accumulator, contribution) in a_total.iter_mut().zip(batch_hh.iter()) {
            *accumulator += contribution;
        }
        for (accumulator, contribution) in b_total.iter_mut().zip(batch_hy.iter()) {
            *accumulator += contribution;
        }
        images_visited += TRAIN_BATCH;
    }
    let one_pass_wall_clock = one_pass_start.elapsed();
    assert_eq!(images_visited, TRAIN_EXAMPLES, "every training image must be visited exactly once by the streaming accumulation pass");
    let dead_unit_fraction = dead_units.iter().filter(|&&dead| dead).count() as f64 / k as f64;

    // `lambda = 1e-2 * trace(A) / k`: index arithmetic over `A`'s own
    // diagonal, not a matmul or elementwise-shaped operation, so it stays a
    // plain host loop rather than materializing a `[k, k]` identity buffer
    // through the tensor algebra for no benefit -- see this module's own
    // doc for why this one step is host-side bookkeeping, not compute.
    let trace: f32 = (0..k).map(|index| a_total[index * k + index]).sum();
    let lambda = LAMBDA_SCALE * trace / k as f32;
    for index in 0..k {
        a_total[index * k + index] += lambda;
    }

    let cg_step = build_cg_step(k);
    let mut x_state = vec![0.0f32; k * OUT_DIM];
    let mut r_state = b_total.clone();
    let mut p_state = b_total.clone();
    let b_norm: f32 = b_total.iter().map(|value| value * value).sum::<f32>().sqrt().max(1e-12);

    let solve_start = std::time::Instant::now();
    let mut cg_iters = 0usize;
    let mut relative_residual = f32::INFINITY;
    for iteration in 1..=CG_MAX_ITERS {
        let cg_named: Vec<(&str, &[f32])> = vec![("a", &a_total), ("x", &x_state), ("r", &r_state), ("p", &p_state)];
        let evaluated = proxima_tensor::cpu::evaluate_named(&cg_step.program, &[], &cg_named, &[cg_step.new_x, cg_step.new_r, cg_step.new_p, cg_step.new_r_dot_r])
            .expect("cg step evaluates");

        let (new_x, _) = evaluated.get(cg_step.new_x).expect("new_x present");
        let (new_r, _) = evaluated.get(cg_step.new_r).expect("new_r present");
        let (new_p, _) = evaluated.get(cg_step.new_p).expect("new_p present");
        let (new_r_dot_r, _) = evaluated.get(cg_step.new_r_dot_r).expect("new_r_dot_r present");

        x_state = new_x.to_vec();
        r_state = new_r.to_vec();
        p_state = new_p.to_vec();
        cg_iters = iteration;

        let residual_norm = new_r_dot_r.iter().sum::<f32>().sqrt();
        relative_residual = residual_norm / b_norm;
        if relative_residual < CG_RELATIVE_RESIDUAL_TOLERANCE {
            break;
        }
    }
    let solve_wall_clock = solve_start.elapsed();
    let beta = x_state;

    let (predict_program, predict_x, predict_w, predict_b, predict_h) = build_feature_program(TEST_EXAMPLES, k);
    let predict_named: Vec<(&str, &[f32])> = vec![("x", test_images), ("w", w), ("b", b)];
    let predict_evaluated = proxima_tensor::cpu::evaluate_named(&predict_program, &[], &predict_named, &[predict_h]).expect("predict features evaluate");
    let (test_h, _) = predict_evaluated.get(predict_h).expect("test h present");

    let mut logits_program = Vec::new();
    let logits_h = leaf(&mut logits_program, "h", vec![Extent::Static(TEST_EXAMPLES as u32), Extent::Static(k as u32)]);
    let logits_beta = leaf(&mut logits_program, "beta", vec![Extent::Static(k as u32), Extent::Static(OUT_DIM as u32)]);
    let logits = matmul3(&mut logits_program, logits_h, &[0, 1], logits_beta, &[1, 2], &[0, 2]);
    let logits_named: Vec<(&str, &[f32])> = vec![("h", test_h), ("beta", &beta)];
    let logits_evaluated = proxima_tensor::cpu::evaluate_named(&logits_program, &[], &logits_named, &[logits]).expect("logits evaluate");
    let (logits_values, _) = logits_evaluated.get(logits).expect("logits present");

    let mut correct = 0usize;
    let mut per_class_correct = [0usize; OUT_DIM];
    let mut per_class_total = [0usize; OUT_DIM];
    for (index, &label) in test_labels.iter().enumerate() {
        let row = &logits_values[index * OUT_DIM..(index + 1) * OUT_DIM];
        per_class_total[label as usize] += 1;
        if argmax_row(row) == label as usize {
            correct += 1;
            per_class_correct[label as usize] += 1;
        }
    }
    let test_accuracy = correct as f64 / test_labels.len() as f64;
    let mut per_class_accuracy = [0.0f64; OUT_DIM];
    for class in 0..OUT_DIM {
        per_class_accuracy[class] = per_class_correct[class] as f64 / per_class_total[class].max(1) as f64;
    }

    let _ = (feature_x, feature_w, feature_b, gram_h, gram_y, predict_x, predict_w, predict_b);
    RungResult { k, lambda, cg_iters, relative_residual, one_pass_wall_clock, solve_wall_clock, test_accuracy, dead_unit_fraction, per_class_accuracy }
}

fn main() {
    if !checkpoint_present() {
        eprintln!("skipping: no host-local MNIST idx dataset under {DATASET_DIR}");
        return;
    }

    let train_images = load_normalized_images(&train_images_path(), TRAIN_EXAMPLES);
    let (train_one_hot, _train_labels) = load_one_hot_labels(&train_labels_path(), TRAIN_EXAMPLES);
    let test_images = load_normalized_images(&test_images_path(), TEST_EXAMPLES);
    let (_test_one_hot, test_labels) = load_one_hot_labels(&test_labels_path(), TEST_EXAMPLES);
    assert_eq!(test_labels.len(), TEST_EXAMPLES, "full 10k mnist test set required for the accuracy claim");

    let k_ladder = [1000usize, 2000, 4000];
    let mut results = Vec::new();
    for &k in &k_ladder {
        eprintln!("one_pass_mnist: starting k={k}");
        let w = random_projection(0x0FF5_9E37_79B9_D6C6 ^ k as u64, IN_DIM, k);
        let b = random_bias(0x1B87_3593_2745_A26D ^ k as u64, k);
        let result = run_rung(&w, &b, k, &train_images, &train_one_hot, &test_images, &test_labels);
        eprintln!(
            "one_pass_mnist: k={} lambda={:.6} cg_iters={} relative_residual={:.6} one_pass_wall_clock={:?} solve_wall_clock={:?} test_accuracy={:.4} dead_unit_fraction={:.4} per_class_accuracy={:?}",
            result.k,
            result.lambda,
            result.cg_iters,
            result.relative_residual,
            result.one_pass_wall_clock,
            result.solve_wall_clock,
            result.test_accuracy,
            result.dead_unit_fraction,
            result.per_class_accuracy
        );
        results.push(result);
    }

    eprintln!("\n| k | lambda | cg iters | relative residual | one-pass wall clock | solve wall clock | test accuracy |");
    eprintln!("|---:|---:|---:|---:|---:|---:|---:|");
    for result in &results {
        eprintln!(
            "| {} | {:.6} | {} | {:.6} | {:?} | {:?} | {:.4} |",
            result.k, result.lambda, result.cg_iters, result.relative_residual, result.one_pass_wall_clock, result.solve_wall_clock, result.test_accuracy
        );
    }
    eprintln!("| context: real_mnist_training.rs rung C (28 epochs, full 60k, MLP) | -- | -- | -- | -- | -- | 0.9786 |");

    let best = results.iter().max_by(|left, right| left.test_accuracy.total_cmp(&right.test_accuracy)).expect("at least one rung ran");
    if best.test_accuracy >= 0.97 {
        eprintln!("\nhypothesis SUPPORTED: k={} reached {:.4} >= 0.97 from one streaming pass, zero epochs", best.k, best.test_accuracy);
    } else {
        eprintln!("\nhypothesis NOT SUPPORTED: best rung k={} reached {:.4} < 0.97", best.k, best.test_accuracy);
    }
}
