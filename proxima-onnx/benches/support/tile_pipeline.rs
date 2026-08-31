//! Line-buffer tile pipeline for the mnist forward (owner thesis, see
//! `proxima-tensor/docs/discipline.md` ROW 155): each conv+relu(+BN) layer
//! is a sans-IO FSM implementing `proxima_primitives::pipe::Pipe`
//! (`In`/`Out` = one row-band), composed with the crate's own `AndThen`
//! combinator, streaming row-bands between layers instead of materializing
//! a whole-layer activation buffer. Conv arithmetic dot-products the
//! gathered activation window against one weight row at a time, K-lane
//! vectorized (`dot_chunked_k4`, below) -- the SAME register-tile FMA idiom
//! `proxima-tensor::cpu::gemm_tile_neon` documents, reimplemented here
//! (not called: the real kernel is `cpu.rs`-module-private and the
//! placement discipline forbids adding anything to the library crates for
//! this experiment). This module's ORIGINAL shape vectorized across
//! `TILE_COLS` output channels instead of across `k` and measured scalar
//! on disassembly (this initiative's own discipline-log row); the K-lane
//! restructure below measures packed `fmla.4s` in its place. FC1 (the
//! largest layer, 11616 elements) is likewise never materialized: its 32
//! outputs are accumulated incrementally as conv3's row-bands stream past,
//! a direct consequence of FC being a linear functional of the flattened
//! activation.
//!
//! Bench/test support module only -- not part of any library crate's
//! public surface (`#[path]`-included from `benches/tile_pipeline.rs` and
//! `tests/tile_pipeline_differential.rs`), gated end-to-end behind the
//! `tile-pipeline-bench` feature.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_arguments, clippy::similar_names)]

use std::cell::RefCell;
use std::collections::VecDeque;
use std::convert::Infallible;
use std::rc::Rc;
use std::task::{Context, Poll, Waker};

use proxima_primitives::pipe::{AndThen, Pipe};

/// Register-tile width this module mirrors from `proxima_tensor::sized`
/// (the SAME sizing constant the landed `gemm_tile_neon` kernel uses, not a
/// fresh magic number -- see ROW 155's magic-const audit note for the ones
/// that are NOT yet routed through a shared constant).
pub const TILE_COLS: usize = proxima_tensor::sized::TILE_COLS;

/// Upper bound on `ci * kh * kw` across mnist's 3 real conv folds
/// (`16*3*3 = 144`, conv3's own shape) -- sizes the gather stack array so
/// no layer's window construction needs a heap allocation.
const MAX_K: usize = 144;

/// One row-band: `rows` consecutive rows, each `channels * width` floats,
/// channel-major within a row (`data[r*channels*width + c*width + w]`).
/// Deliberately the SAME layout a producing stage emits and a consuming
/// stage ingests, so stages compose with zero repacking between them.
#[derive(Clone, Debug)]
pub struct RowBand {
    pub channels: usize,
    pub width: usize,
    pub rows: usize,
    pub data: Vec<f32>,
}

impl RowBand {
    fn row(&self, index: usize) -> &[f32] {
        let stride = self.channels * self.width;
        &self.data[index * stride..(index + 1) * stride]
    }

    fn channel_row(&self, index: usize, channel: usize) -> &[f32] {
        &self.row(index)[channel * self.width..(channel + 1) * self.width]
    }
}

/// Precomputed per-channel affine (`y = x*scale + shift`) folding a
/// `BatchNormalization` node's `(weight, bias, running_mean, running_var,
/// epsilon)` into two multiply-add coefficients -- the same reduction the
/// sealed executor's own constant-folding would reach, done once at stage
/// construction so the hot per-pixel path pays one `mul_add`, not a
/// division and a sqrt.
pub struct BatchNormAffine {
    scale: Vec<f32>,
    shift: Vec<f32>,
}

impl BatchNormAffine {
    pub fn new(weight: &[f32], bias: &[f32], mean: &[f32], var: &[f32], epsilon: f32) -> Self {
        let scale: Vec<f32> = weight.iter().zip(var).map(|(&w, &v)| w / (v + epsilon).sqrt()).collect();
        let shift: Vec<f32> = bias.iter().zip(mean).zip(&scale).map(|((&b, &m), &s)| b - m * s).collect();
        Self { scale, shift }
    }

    fn apply(&self, channel: usize, value: f32) -> f32 {
        value.mul_add(self.scale[channel], self.shift[channel])
    }
}

/// One `Conv(kh x kw, stride 1, pad 0) -> ReLU -> optional BatchNorm` layer
/// as a sans-IO FSM. State is a ring of the last `kh` input rows it has
/// seen (`VecDeque`, FIFO, capped at `kh`) -- exactly the "ring of the `kh`
/// input rows it needs" the task's own design calls for. `Pipe::call` takes
/// `&self`, so the ring lives behind a `RefCell`: single-threaded, local,
/// never shared across a `Send` boundary (`Pipe`'s own doc: no `Send`
/// bound), the same interior-mutability shape a per-core `!Send` worker
/// already uses elsewhere in this algebra.
pub struct ConvReluStage<'weights> {
    channels_in: usize,
    channels_out: usize,
    kernel_height: usize,
    kernel_width: usize,
    input_width: usize,
    output_width: usize,
    weight: &'weights [f32],
    bias: &'weights [f32],
    batch_norm: Option<BatchNormAffine>,
    ring: RefCell<VecDeque<Vec<f32>>>,
}

impl<'weights> ConvReluStage<'weights> {
    pub fn new(
        channels_in: usize,
        channels_out: usize,
        kernel_height: usize,
        kernel_width: usize,
        input_width: usize,
        weight: &'weights [f32],
        bias: &'weights [f32],
        batch_norm: Option<BatchNormAffine>,
    ) -> Self {
        assert_eq!(weight.len(), channels_out * channels_in * kernel_height * kernel_width, "conv weight shape mismatch");
        assert_eq!(bias.len(), channels_out, "conv bias shape mismatch");
        assert!(channels_out.is_multiple_of(TILE_COLS), "output channels must tile evenly by TILE_COLS");
        Self {
            channels_in,
            channels_out,
            kernel_height,
            kernel_width,
            input_width,
            output_width: input_width - kernel_width + 1,
            weight,
            bias,
            batch_norm,
            ring: RefCell::new(VecDeque::with_capacity(kernel_height)),
        }
    }

    /// Gathers `channels_in * kernel_height * kernel_width` contiguous
    /// floats for one output column out of the ring's dense per-row
    /// storage (a small, bounded local materialize -- NOT a whole-layer
    /// buffer, and small enough (<= 144 floats) to stay resident in L1
    /// throughout). Order matches the conv weight's own `[co][ci][kh][kw]`
    /// flattening exactly, so the dot-product below needs no further
    /// permutation.
    fn gather_window(&self, ring: &VecDeque<Vec<f32>>, output_column: usize, scratch: &mut [f32; MAX_K]) {
        for channel in 0..self.channels_in {
            for (kernel_row, source_row) in ring.iter().take(self.kernel_height).enumerate() {
                let start = channel * self.input_width + output_column;
                let destination = (channel * self.kernel_height + kernel_row) * self.kernel_width;
                scratch[destination..destination + self.kernel_width].copy_from_slice(&source_row[start..start + self.kernel_width]);
            }
        }
    }

    /// One output row from a full `kh`-deep ring: for every output column,
    /// gather the window once, then dot it against each of `TILE_COLS`
    /// weight rows in turn via [`dot_chunked_k4`] -- K-lane vectorized per
    /// channel, not across channels (see that function's own doc for why),
    /// reassociated over `k` relative to the sealed executor's own SIMD dot
    /// fold, the same bounded-reassociation category ROW 151's own
    /// differential test already documents for this initiative.
    fn compute_output_row(&self, ring: &VecDeque<Vec<f32>>) -> Vec<f32> {
        let reduction_width = self.channels_in * self.kernel_height * self.kernel_width;
        let mut output_row = vec![0.0_f32; self.channels_out * self.output_width];
        let mut gather = [0.0_f32; MAX_K];
        for output_column in 0..self.output_width {
            self.gather_window(ring, output_column, &mut gather);
            let window = &gather[..reduction_width];
            let mut channel_out = 0;
            while channel_out < self.channels_out {
                for lane in 0..TILE_COLS {
                    let channel = channel_out + lane;
                    let row_start = channel * reduction_width;
                    let weight_row = &self.weight[row_start..row_start + reduction_width];
                    let dot = dot_chunked_k4(window, weight_row);
                    let mut value = (dot + self.bias[channel]).max(0.0);
                    if let Some(batch_norm) = &self.batch_norm {
                        value = batch_norm.apply(channel, value);
                    }
                    output_row[channel * self.output_width + output_column] = value;
                }
                channel_out += TILE_COLS;
            }
        }
        output_row
    }
}

/// Dot product over a shared `window` (the gathered `ci*kh*kw` activation
/// span, contiguous) and one weight row (also contiguous, `weight[co]`'s
/// own span), reduced 4 elements of `k` at a time -- the SIMD-amenable
/// axis, per `gemm_tile_neon`'s own kernel: within one weight row the `k`
/// dimension IS contiguous, unlike the ORIGINAL (already-scalarized, see
/// this initiative's own discipline-log row) attempt to vectorize ACROSS
/// `TILE_COLS` output channels, whose weight rows sit `reduction_width`
/// floats apart and can never share one contiguous SIMD load. `k4` folds
/// `k` in 4-wide groups into 4 independent lane accumulators before a
/// single horizontal sum, the same shape `dot_fold_multi_accumulator_binary`
/// (`proxima_tensor::cpu`, private) already uses -- reimplemented here for
/// the same placement-discipline reason this module's own header comment
/// states.
///
/// Portable safe Rust, no `target_arch` split and no `unsafe`: measured
/// (this session's own `objdump`, see the discipline log) to compile to a
/// packed `fmla.4s` loop on aarch64 -- LLVM auto-vectorizes this shape even
/// though it did NOT auto-vectorize the TILE_COLS-lanes-in-one-array shape
/// this replaced. A hand-written `core::arch::aarch64` intrinsics form was
/// also built and benched (same log row): ~1.5% faster at p50, inside the
/// ~1% CoV noise band of both arms, not worth the `unsafe` surface it would
/// add -- named as a real, measured, deliberately-not-adopted alternative,
/// not hidden.
fn dot_chunked_k4(window: &[f32], weight_row: &[f32]) -> f32 {
    let (window_chunks, window_remainder) = window.as_chunks::<4>();
    let (weight_chunks, weight_remainder) = weight_row.as_chunks::<4>();
    let mut lanes = [0.0_f32; 4];
    for (window_chunk, weight_chunk) in window_chunks.iter().zip(weight_chunks) {
        for ((lane, &window_value), &weight_value) in lanes.iter_mut().zip(window_chunk).zip(weight_chunk) {
            *lane = window_value.mul_add(weight_value, *lane);
        }
    }
    let mut total = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (&window_value, &weight_value) in window_remainder.iter().zip(weight_remainder) {
        total = window_value.mul_add(weight_value, total);
    }
    total
}

impl Pipe for ConvReluStage<'_> {
    type In = RowBand;
    type Out = RowBand;
    type Err = Infallible;

    fn call(&self, input: Self::In) -> impl std::future::Future<Output = Result<Self::Out, Self::Err>> {
        async move {
            let mut ring = self.ring.borrow_mut();
            let mut emitted_rows: Vec<Vec<f32>> = Vec::with_capacity(input.rows);
            debug_assert_eq!(input.channels, self.channels_in);
            debug_assert_eq!(input.width, self.input_width);
            for row_index in 0..input.rows {
                ring.push_back(input.row(row_index).to_vec());
                if ring.len() > self.kernel_height {
                    ring.pop_front();
                }
                if ring.len() == self.kernel_height {
                    emitted_rows.push(self.compute_output_row(&ring));
                }
            }
            let rows = emitted_rows.len();
            let mut data = Vec::with_capacity(rows * self.channels_out * self.output_width);
            for row in emitted_rows {
                data.extend(row);
            }
            Ok(RowBand { channels: self.channels_out, width: self.output_width, rows, data })
        }
    }
}

/// FC1's own running state: bias-seeded accumulator plus how many of the
/// flattened activation's rows have been folded in so far (needed to
/// compute each streamed row's absolute flat offset).
struct FcAccumulatorState {
    accumulator: Vec<f32>,
    rows_seen: usize,
}

/// The terminal sink: accumulates FC1 (`out_features x (channels*height*
/// width)`) incrementally as conv3's row-bands stream past, never
/// materializing the 11616-element flattened activation FC1 would
/// otherwise need whole. `Clone` shares the accumulator via `Rc<RefCell<_>>`
/// so a caller can hold one clone inside the composed `AndThen` chain and
/// another outside it to read the final accumulator back out after the
/// image's rows are exhausted -- `Pipe::call` alone has no "flush" signal,
/// so the read-back is a plain inherent method, deliberately outside the
/// algebra.
#[derive(Clone)]
pub struct FcAccumulateStage<'weights> {
    channels: usize,
    height: usize,
    width: usize,
    weight: &'weights [f32],
    state: Rc<RefCell<FcAccumulatorState>>,
}

impl<'weights> FcAccumulateStage<'weights> {
    pub fn new(channels: usize, height: usize, width: usize, out_features: usize, weight: &'weights [f32], bias: &'weights [f32]) -> Self {
        assert_eq!(weight.len(), out_features * channels * height * width, "fc1 weight shape mismatch");
        assert_eq!(bias.len(), out_features, "fc1 bias shape mismatch");
        Self {
            channels,
            height,
            width,
            weight,
            state: Rc::new(RefCell::new(FcAccumulatorState { accumulator: bias.to_vec(), rows_seen: 0 })),
        }
    }

    /// FC1's output (bias-seeded accumulator, ReLU applied) after every row
    /// of the flattened activation has streamed through. Panics if fewer
    /// than `height` rows were ever fed -- a genuine caller bug, the same
    /// severity `debug_assert!`-guarded invariants elsewhere in this module
    /// carry, promoted to a hard panic here because a partial FC1 result is
    /// silently wrong, never a value worth returning.
    pub fn finalize(&self) -> Vec<f32> {
        let state = self.state.borrow();
        assert_eq!(state.rows_seen, self.height, "fc1 accumulator did not see every row of the flattened activation");
        state.accumulator.iter().map(|&value| value.max(0.0)).collect()
    }
}

impl Pipe for FcAccumulateStage<'_> {
    type In = RowBand;
    type Out = ();
    type Err = Infallible;

    fn call(&self, input: Self::In) -> impl std::future::Future<Output = Result<Self::Out, Self::Err>> {
        async move {
            let mut state = self.state.borrow_mut();
            let plane = self.height * self.width;
            let in_features = self.channels * plane;
            for row_index in 0..input.rows {
                let absolute_row = state.rows_seen;
                for channel in 0..self.channels {
                    let channel_row = input.channel_row(row_index, channel);
                    let base = channel * plane + absolute_row * self.width;
                    for (output_index, accumulator_value) in state.accumulator.iter_mut().enumerate() {
                        let weight_start = output_index * in_features + base;
                        let weight_row = &self.weight[weight_start..weight_start + channel_row.len()];
                        *accumulator_value += dot_chunked_k4(channel_row, weight_row);
                    }
                }
                state.rows_seen += 1;
            }
            Ok(())
        }
    }
}

/// Every stage's future resolves on first poll (no real async waiting
/// anywhere in this pipeline -- pure CPU work) -- this drives a `Pipe`
/// synchronously without pulling in an async runtime, using the stable
/// no-op waker.
pub fn block_on_ready<F: std::future::Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    match future.as_mut().poll(&mut context) {
        Poll::Ready(value) => value,
        Poll::Pending => panic!("tile pipeline stage future did not resolve synchronously"),
    }
}

/// The real mnist checkpoint's weights, borrowed straight out of
/// `lowered.initializers` (no copy) -- field names mirror the ONNX
/// initializer names 1:1, see `proxima-onnx/examples/scratch_graph_dump.rs`
/// output cited in ROW 155 for the provenance of every shape below.
pub struct MnistWeights<'data> {
    pub conv1_weight: &'data [f32],
    pub conv1_bias: &'data [f32],
    pub conv2_weight: &'data [f32],
    pub conv2_bias: &'data [f32],
    pub conv3_weight: &'data [f32],
    pub conv3_bias: &'data [f32],
    pub norm1_weight: &'data [f32],
    pub norm1_bias: &'data [f32],
    pub norm1_running_mean: &'data [f32],
    pub norm1_running_var: &'data [f32],
    pub fc1_weight: &'data [f32],
    pub fc1_bias: &'data [f32],
    pub fc2_weight: &'data [f32],
    pub fc2_bias: &'data [f32],
    pub norm2_weight: &'data [f32],
    pub norm2_bias: &'data [f32],
    pub norm2_running_mean: &'data [f32],
    pub norm2_running_var: &'data [f32],
}

impl<'data> MnistWeights<'data> {
    pub fn from_initializers(initializers: &[(&'data str, &'data [f32])]) -> Self {
        let find = |name: &str| -> &'data [f32] { initializers.iter().find(|(candidate, _)| *candidate == name).unwrap_or_else(|| panic!("missing initializer {name}")).1 };
        Self {
            conv1_weight: find("conv1.weight"),
            conv1_bias: find("conv1.bias"),
            conv2_weight: find("conv2.weight"),
            conv2_bias: find("conv2.bias"),
            conv3_weight: find("conv3.weight"),
            conv3_bias: find("conv3.bias"),
            norm1_weight: find("norm1.weight"),
            norm1_bias: find("norm1.bias"),
            norm1_running_mean: find("norm1.running_mean"),
            norm1_running_var: find("norm1.running_var"),
            fc1_weight: find("fc1.weight"),
            fc1_bias: find("fc1.bias"),
            fc2_weight: find("fc2.weight"),
            fc2_bias: find("fc2.bias"),
            norm2_weight: find("norm2.weight"),
            norm2_bias: find("norm2.bias"),
            norm2_running_mean: find("norm2.running_mean"),
            norm2_running_var: find("norm2.running_var"),
        }
    }
}

const EPSILON: f32 = 1e-5;

fn matvec_bias(weight: &[f32], bias: &[f32], input: &[f32], out_features: usize, in_features: usize) -> Vec<f32> {
    (0..out_features)
        .map(|output_index| {
            let row = &weight[output_index * in_features..(output_index + 1) * in_features];
            let dot: f32 = row.iter().zip(input).fold(0.0_f32, |accumulator, (&weight_value, &input_value)| input_value.mul_add(weight_value, accumulator));
            dot + bias[output_index]
        })
        .collect()
}

fn apply_batch_norm(values: &[f32], weight: &[f32], bias: &[f32], mean: &[f32], var: &[f32]) -> Vec<f32> {
    values
        .iter()
        .enumerate()
        .map(|(index, &value)| (value - mean[index]) / (var[index] + EPSILON).sqrt() * weight[index] + bias[index])
        .collect()
}

fn log_softmax(values: &[f32]) -> [f32; 10] {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = values.iter().map(|&value| (value - max).exp()).sum();
    let log_sum = sum.ln();
    let mut out = [0.0_f32; 10];
    for (destination, &value) in out.iter_mut().zip(values) {
        *destination = value - max - log_sum;
    }
    out
}

/// Which row-band granularity to feed stage 1 with per pipe call -- the
/// task's own band-size sweep axis (1 row / `kh` rows / `2*kh` rows).
/// Downstream stages receive whatever their own upstream neighbour emits
/// per call, which is NOT the same number (it depends on ring fill state),
/// exactly as a real streaming pipeline behaves.
#[derive(Clone, Copy, Debug)]
pub struct BandRows(pub usize);

/// Runs the full mnist forward through the composed tile pipeline for one
/// `28*28` normalized image, feeding `band.0` new rows to stage 1 per pipe
/// call. Returns the 10 log-softmax logits -- bit-comparable to
/// `cpu::evaluate_named`'s own output up to reassociation (see the
/// differential test).
pub fn run_pipeline_forward(image: &[f32], weights: &MnistWeights<'_>, band: BandRows) -> [f32; 10] {
    let batch_norm1 = BatchNormAffine::new(weights.norm1_weight, weights.norm1_bias, weights.norm1_running_mean, weights.norm1_running_var, EPSILON);

    let stage1 = ConvReluStage::new(1, 8, 3, 3, 28, weights.conv1_weight, weights.conv1_bias, None);
    let stage2 = ConvReluStage::new(8, 16, 3, 3, 26, weights.conv2_weight, weights.conv2_bias, None);
    let stage3 = ConvReluStage::new(16, 24, 3, 3, 24, weights.conv3_weight, weights.conv3_bias, Some(batch_norm1));
    let fc_stage = FcAccumulateStage::new(24, 22, 22, 32, weights.fc1_weight, weights.fc1_bias);
    let fc_stage_for_finalize = fc_stage.clone();

    let pipeline = AndThen::new(stage1, AndThen::new(stage2, AndThen::new(stage3, fc_stage)));

    let mut row = 0;
    while row < 28 {
        let take = band.0.min(28 - row);
        let data = image[row * 28..(row + take) * 28].to_vec();
        let input_band = RowBand { channels: 1, width: 28, rows: take, data };
        block_on_ready(pipeline.call(input_band)).expect("tile pipeline stages are infallible");
        row += take;
    }

    let fc1_out = fc_stage_for_finalize.finalize();
    let fc2_out = matvec_bias(weights.fc2_weight, weights.fc2_bias, &fc1_out, 10, 32);
    let bn2_out = apply_batch_norm(&fc2_out, weights.norm2_weight, weights.norm2_bias, weights.norm2_running_mean, weights.norm2_running_var);
    log_softmax(&bn2_out)
}
