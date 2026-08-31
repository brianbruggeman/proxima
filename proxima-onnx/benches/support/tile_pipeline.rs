//! Line-buffer tile pipeline for the mnist forward (owner thesis, see
//! `proxima-tensor/docs/discipline.md` ROW 155): each conv+relu(+BN) layer
//! is a sans-IO FSM implementing `proxima_primitives::pipe::Pipe`
//! (`In`/`Out` = one row-band), composed with the crate's own `AndThen`
//! combinator, streaming row-bands between layers instead of materializing
//! a whole-layer activation buffer. Conv arithmetic dot-products the
//! gathered activation window against weight rows -- via ONE of two forms,
//! selected PER CALL SITE at compile time (ROW 170, extended ROW 171):
//! `dot_chunked_k4_tile_multirow::<ROWS>` (register-blocked across a
//! `TILE_COLS` channel group AND `ROWS` adjacent output columns, one
//! shared weight K-chunk load per lane reused by all `ROWS * TILE_COLS`
//! accumulators) or `dot_chunked_k4` (one row at a time, ROW 158's
//! pre-register-blocking form) -- both the SAME register-tile FMA idiom
//! `proxima-tensor::cpu::gemm_tile_neon` documents, reimplemented here
//! (not called: the real kernel is `cpu.rs`-module-private and the
//! placement discipline forbids adding anything to the library crates for
//! this experiment). ROW 169 measured the blocked form winning at
//! conv2/conv3's large `reduction_width` (amortizing the shared load) but
//! LOSING at conv1/fc1's small `reduction_width`, where the blocked
//! form's larger signature did not auto-inline (confirmed `bl` in
//! disassembly) and call overhead dominated a tiny per-call workload --
//! `dot_chunked_k4`'s smaller signature auto-inlines at those same sites
//! (ROW 158's own disassembly finding, re-confirmed ROW 170). ROW 171
//! measured multi-row blocking (`ROWS = 2`, `4`) winning FURTHER at
//! conv2/conv3, `ROWS = 4` the strongest, by dividing weight K-stream
//! traffic (not window traffic, which is genuinely distinct per column)
//! by `ROWS`. `ConvReluStage<'weights, const BLOCKED: bool, const ROWS:
//! usize>` selects the form as two const generics, monomorphized per
//! stage at its own `new::<..>` call site -- conv1 instantiates `BLOCKED
//! = false` (`ROWS` unused, defaults to 1), conv2/conv3 instantiate
//! `BLOCKED = true, ROWS = 4`; `FcAccumulateStage` (fc1's only
//! instantiation) always calls `dot_chunked_k4` directly, unconditionally.
//! No runtime branch, no config: which function a stage's own compiled
//! code calls, and at what row-block width, is fixed at that stage's own
//! type, forever, the same way its shape (`channels_in`, `kernel_height`,
//! ...) already is. FC1 (the largest layer, 11616 elements) is likewise
//! never materialized: its 32 outputs are accumulated incrementally as
//! conv3's row-bands stream past, a direct consequence of FC being a
//! linear functional of the flattened activation.
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
///
/// `BLOCKED` selects the dot-product form this stage's own
/// `compute_output_row` calls (ROW 170/171, see module doc): `true` for
/// the register-blocked `dot_chunked_k4_tile_multirow::<ROWS>`, `false`
/// for the single-row `dot_chunked_k4` (`ROWS` unused). Fixed per stage
/// at construction's own type parameters, not a runtime field -- the
/// branch below on `BLOCKED` is on a monomorphized compile-time constant,
/// eliminated by the optimizer per
/// instantiation (verified via objdump, see the discipline log).
pub struct ConvReluStage<'weights, const BLOCKED: bool, const ROWS: usize = 1> {
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

impl<'weights, const BLOCKED: bool, const ROWS: usize> ConvReluStage<'weights, BLOCKED, ROWS> {
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

    /// One output row from a full `kh`-deep ring. `BLOCKED = false` walks
    /// one output column at a time via [`dot_chunked_k4`] per lane (ROW
    /// 158's auto-inlining form, `ROWS` unused). `BLOCKED = true` walks
    /// `ROWS` adjacent output columns per K-pass via
    /// [`dot_chunked_k4_tile_multirow`] (ROW 171's multi-row register
    /// blocking, generalizing ROW 169's single-row `dot_chunked_k4_tile`,
    /// which is exactly the `ROWS = 1` instantiation of the same
    /// function): one shared weight K-chunk load per lane is now reused
    /// across `ROWS * TILE_COLS` accumulators instead of `TILE_COLS`,
    /// dividing weight K-stream traffic by `ROWS`. A remainder loop (when
    /// `output_width` is not a multiple of `ROWS`) falls back to the same
    /// function's own `ROWS = 1` instantiation, never a separate code
    /// path. `BLOCKED`/`ROWS` are both fixed per stage instantiation, not
    /// read at runtime. Reassociated over `k` relative to the sealed
    /// executor's own SIMD dot fold, the same bounded-reassociation
    /// category ROW 151's own differential test already documents for
    /// this initiative.
    fn compute_output_row(&self, ring: &VecDeque<Vec<f32>>) -> Vec<f32> {
        let reduction_width = self.channels_in * self.kernel_height * self.kernel_width;
        let mut output_row = vec![0.0_f32; self.channels_out * self.output_width];

        if BLOCKED {
            let mut output_column = 0;
            while output_column + ROWS <= self.output_width {
                let mut gathers = [[0.0_f32; MAX_K]; ROWS];
                for (row, gather) in gathers.iter_mut().enumerate() {
                    self.gather_window(ring, output_column + row, gather);
                }
                let windows: [&[f32]; ROWS] = std::array::from_fn(|row| &gathers[row][..reduction_width]);
                self.emit_blocked_columns::<ROWS>(&mut output_row, windows, output_column, reduction_width);
                output_column += ROWS;
            }
            while output_column < self.output_width {
                let mut gather = [0.0_f32; MAX_K];
                self.gather_window(ring, output_column, &mut gather);
                let windows: [&[f32]; 1] = [&gather[..reduction_width]];
                self.emit_blocked_columns::<1>(&mut output_row, windows, output_column, reduction_width);
                output_column += 1;
            }
        } else {
            let mut gather = [0.0_f32; MAX_K];
            for output_column in 0..self.output_width {
                self.gather_window(ring, output_column, &mut gather);
                let window = &gather[..reduction_width];
                let mut channel_out = 0;
                while channel_out < self.channels_out {
                    let dots: [f32; TILE_COLS] = std::array::from_fn(|lane| {
                        let row_start = (channel_out + lane) * reduction_width;
                        let weight_row = &self.weight[row_start..row_start + reduction_width];
                        dot_chunked_k4(window, weight_row)
                    });
                    for (lane, &dot) in dots.iter().enumerate() {
                        let channel = channel_out + lane;
                        let mut value = (dot + self.bias[channel]).max(0.0);
                        if let Some(batch_norm) = &self.batch_norm {
                            value = batch_norm.apply(channel, value);
                        }
                        output_row[channel * self.output_width + output_column] = value;
                    }
                    channel_out += TILE_COLS;
                }
            }
        }
        output_row
    }

    /// Dots `BLOCK` adjacent output columns' windows against every
    /// `TILE_COLS` channel group, writing straight into `output_row` --
    /// shared by [`compute_output_row`]'s own main (`BLOCK = ROWS`) and
    /// remainder (`BLOCK = 1`) loops so there is exactly one place this
    /// bias/ReLU/batch-norm epilogue is written, not two.
    fn emit_blocked_columns<const BLOCK: usize>(&self, output_row: &mut [f32], windows: [&[f32]; BLOCK], output_column: usize, reduction_width: usize) {
        let mut channel_out = 0;
        while channel_out < self.channels_out {
            let weight_rows: [&[f32]; TILE_COLS] = std::array::from_fn(|lane| {
                let row_start = (channel_out + lane) * reduction_width;
                &self.weight[row_start..row_start + reduction_width]
            });
            let dots = dot_chunked_k4_tile_multirow::<BLOCK>(windows, weight_rows);
            for (row, row_dots) in dots.iter().enumerate() {
                for (lane, &dot) in row_dots.iter().enumerate() {
                    let channel = channel_out + lane;
                    let mut value = (dot + self.bias[channel]).max(0.0);
                    if let Some(batch_norm) = &self.batch_norm {
                        value = batch_norm.apply(channel, value);
                    }
                    output_row[channel * self.output_width + output_column + row] = value;
                }
            }
            channel_out += TILE_COLS;
        }
    }
}

/// Dots a shared `window` (the gathered `ci*kh*kw` activation span,
/// contiguous) against all `TILE_COLS` weight rows of one channel group at
/// once (each row also contiguous, `weight[co]`'s own span) -- the
/// register-blocking lever (ROW 169): a shared `window` chunk load per
/// `k`-step feeds `TILE_COLS` independent lane-accumulator chains
/// (`[[f32; 4]; TILE_COLS]`) before advancing, instead of the earlier shape
/// (ROW 158) that called a single-row dot product `TILE_COLS` times, each
/// call independently re-reading `window` from the top. `k` is the
/// SIMD-amenable axis per `gemm_tile_neon`'s own kernel: within one weight
/// row the `k` dimension IS contiguous, unlike the ORIGINAL (already-
/// scalarized, see this initiative's own discipline-log row) attempt to
/// vectorize ACROSS `TILE_COLS` output channels, whose weight rows sit
/// `reduction_width` floats apart and can never share one contiguous SIMD
/// load.
///
/// Portable safe Rust, no `target_arch` split and no `unsafe`: ROW 158
/// measured (this session's own `objdump`) the single-row form to compile
/// to a packed `fmla.4s` loop on aarch64; ROW 169 re-verifies the same for
/// this tiled form (see the discipline log), AND separately measured that
/// forcing `#[inline(always)]` REGRESSES the whole-forward mean (~561us vs
/// ~529us) -- the non-inlined call was the faster shape for conv2/conv3.
/// `#[inline(never)]` (ROW 170): splitting `ConvReluStage` per BLOCKED
/// value (below) dropped this function to ONE call site
/// (`ConvReluStage<true>::compute_output_row`, conv2/conv3 only, since
/// `FcAccumulateStage` now calls `dot_chunked_k4` directly) -- LLVM's
/// inliner treats a single-call-site function as free to inline and DID,
/// silently re-creating ROW 169's own measured-worse inlined shape
/// (measured, ROW 170: total 593.874us vs the `#[inline(never)]` form's
/// own number below -- see the discipline log). `#[inline(never)]` pins
/// ROW 169's own measured-faster non-inlined choice explicitly rather than
/// leaving it to an inliner heuristic that is no longer stable under this
/// call-site count.
/// Multi-row generalization of ROW 169's `dot_chunked_k4_tile` (ROW 171):
/// dots `ROWS` adjacent output-column windows against `TILE_COLS` weight
/// rows of one channel group, per K-pass loading each weight K-chunk
/// **once** (shared across all `ROWS` windows, not just across the
/// `TILE_COLS` lanes ROW 169 already shared it across) -- `ROWS = 1` is
/// byte-identical in shape to ROW 169's own function, kept as this row's
/// own micro-vetted baseline arm rather than a separate symbol. Weight
/// K-stream traffic divides by `ROWS`; window K-stream traffic does not
/// (each output column's window is genuinely distinct data), matching
/// this task's own framing ("each K-chunk load is then reused across R×4
/// accumulators -- K-stream traffic divides by R", read as the shared
/// weight load specifically, since the window load cannot be shared
/// across positions without a sliding-window rewrite of `gather_window`
/// this row did not attempt).
#[inline(never)]
fn dot_chunked_k4_tile_multirow<const ROWS: usize>(windows: [&[f32]; ROWS], weight_rows: [&[f32]; TILE_COLS]) -> [[f32; TILE_COLS]; ROWS] {
    let weight_chunks: [&[[f32; 4]]; TILE_COLS] = std::array::from_fn(|lane| weight_rows[lane].as_chunks::<4>().0);
    let weight_remainders: [&[f32]; TILE_COLS] = std::array::from_fn(|lane| weight_rows[lane].as_chunks::<4>().1);
    let window_chunks: [&[[f32; 4]]; ROWS] = std::array::from_fn(|row| windows[row].as_chunks::<4>().0);
    let window_remainders: [&[f32]; ROWS] = std::array::from_fn(|row| windows[row].as_chunks::<4>().1);
    let chunk_count = weight_chunks[0].len();
    let mut accumulators = [[[0.0_f32; 4]; TILE_COLS]; ROWS];
    for chunk_index in 0..chunk_count {
        let weight_chunk_values: [[f32; 4]; TILE_COLS] = std::array::from_fn(|lane| weight_chunks[lane][chunk_index]);
        for row in 0..ROWS {
            let window_chunk = window_chunks[row][chunk_index];
            for lane in 0..TILE_COLS {
                for element in 0..4 {
                    accumulators[row][lane][element] = window_chunk[element].mul_add(weight_chunk_values[lane][element], accumulators[row][lane][element]);
                }
            }
        }
    }
    std::array::from_fn(|row| {
        std::array::from_fn(|lane| {
            let accumulator = accumulators[row][lane];
            let mut total = accumulator[0] + accumulator[1] + accumulator[2] + accumulator[3];
            for (&window_value, &weight_value) in window_remainders[row].iter().zip(weight_remainders[lane]) {
                total = window_value.mul_add(weight_value, total);
            }
            total
        })
    })
}

/// Dot product over a shared `window` (the gathered `ci*kh*kw` activation
/// span, contiguous) and ONE weight row (also contiguous, `weight[co]`'s
/// own span), reduced 4 elements of `k` at a time -- ROW 158's own form,
/// restored ROW 170 for conv1/fc1's own call sites: its smaller signature
/// (two slices, not `[&[f32]; TILE_COLS]`) auto-inlines where
/// [`dot_chunked_k4_tile`] does not, which matters most at small
/// `reduction_width` where per-call overhead is the whole cost, not the
/// arithmetic. Portable safe Rust, no `target_arch` split and no `unsafe`:
/// ROW 158 measured (objdump) this shape to compile to a packed `fmla.4s`
/// loop on aarch64 (re-confirmed ROW 170, see the discipline log).
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

impl<const BLOCKED: bool, const ROWS: usize> ConvReluStage<'_, BLOCKED, ROWS> {
    /// The stage's own compute, synchronous -- the SAME body `Pipe::call`
    /// wraps in an immediately-ready `Future`, extracted so ROW 172's
    /// dispatch-floor measurement can invoke it directly (bypassing
    /// `Future`/`AndThen`/`block_on_ready`) without forking the arithmetic:
    /// one body, two call surfaces, byte-identical output either way.
    fn process_band(&self, input: RowBand) -> RowBand {
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
        RowBand { channels: self.channels_out, width: self.output_width, rows, data }
    }

    /// ROW 172's direct-call surface: identical output to `Pipe::call`,
    /// zero `Future`/`Waker`/`Context` machinery. Bench-only, feeds the
    /// dispatch-floor arm.
    pub fn compute_direct(&self, input: RowBand) -> RowBand {
        self.process_band(input)
    }
}

impl<const BLOCKED: bool, const ROWS: usize> Pipe for ConvReluStage<'_, BLOCKED, ROWS> {
    type In = RowBand;
    type Out = RowBand;
    type Err = Infallible;

    fn call(&self, input: Self::In) -> impl std::future::Future<Output = Result<Self::Out, Self::Err>> {
        async move { Ok(self.process_band(input)) }
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

impl FcAccumulateStage<'_> {
    /// The stage's own accumulation, synchronous -- the SAME body
    /// `Pipe::call` wraps in an immediately-ready `Future`, extracted for
    /// ROW 172's dispatch-floor measurement (see `ConvReluStage`'s own
    /// `process_band`/`compute_direct` pair for the identical rationale).
    /// FC1's own accumulation loop calls [`dot_chunked_k4`] directly,
    /// unconditionally (ROW 170: fc1 is `dot_chunked_k4_tile`'s own other
    /// small-`reduction_width` regression site, ROW 169's own measured
    /// +19.07% -- `channel_row.len() <= 22` means per-call overhead
    /// dominates, same mechanism as conv1). No `BLOCKED` parameter here:
    /// `FcAccumulateStage` has exactly one instantiation (fc1), so there is
    /// only ever one function this call site could call.
    fn process_band(&self, input: RowBand) {
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
    }

    /// ROW 172's direct-call surface: identical accumulation to
    /// `Pipe::call`, zero `Future`/`Waker`/`Context` machinery.
    pub fn compute_direct(&self, input: RowBand) {
        self.process_band(input);
    }
}

impl Pipe for FcAccumulateStage<'_> {
    type In = RowBand;
    type Out = ();
    type Err = Infallible;

    fn call(&self, input: Self::In) -> impl std::future::Future<Output = Result<Self::Out, Self::Err>> {
        async move {
            self.process_band(input);
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

    // ROW 170/171: conv1 (`reduction_width` = 9) calls the unblocked,
    // auto-inlining `dot_chunked_k4` form (`BLOCKED = false`); conv2/conv3
    // (`reduction_width` = 72/144) call the register-blocked, 4-row
    // multirow form (`BLOCKED = true, ROWS = 4`), which amortizes its own
    // non-inlined call overhead across 4 output columns at once (ROW 171
    // micro-vet: `ROWS = 4` beat `ROWS = 1`/`2` on both conv2 and conv3).
    // fc1 always calls `dot_chunked_k4` directly (see
    // `FcAccumulateStage::call`'s own doc).
    let stage1 = ConvReluStage::<false>::new(1, 8, 3, 3, 28, weights.conv1_weight, weights.conv1_bias, None);
    let stage2 = ConvReluStage::<true, 4>::new(8, 16, 3, 3, 26, weights.conv2_weight, weights.conv2_bias, None);
    let stage3 = ConvReluStage::<true, 4>::new(16, 24, 3, 3, 24, weights.conv3_weight, weights.conv3_bias, Some(batch_norm1));
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

/// ROW 172's dispatch-floor arm: the SAME 4-stage sequence
/// `run_pipeline_forward` runs, calling each stage's own `compute_direct`
/// (plain synchronous function, same `process_band` body) instead of
/// composing `AndThen` and driving one `Future` through `block_on_ready`.
/// Isolates what `Pipe::call`+`AndThen`+`block_on_ready` cost on top of the
/// identical arithmetic -- never the production surface (that stays
/// `run_pipeline_forward`, `AndThen`-composed, per this module's own sans-IO
/// state-machine design); a measurement-only twin, bench-gated.
pub fn run_pipeline_forward_direct(image: &[f32], weights: &MnistWeights<'_>, band: BandRows) -> [f32; 10] {
    let batch_norm1 = BatchNormAffine::new(weights.norm1_weight, weights.norm1_bias, weights.norm1_running_mean, weights.norm1_running_var, EPSILON);

    let stage1 = ConvReluStage::<false>::new(1, 8, 3, 3, 28, weights.conv1_weight, weights.conv1_bias, None);
    let stage2 = ConvReluStage::<true, 4>::new(8, 16, 3, 3, 26, weights.conv2_weight, weights.conv2_bias, None);
    let stage3 = ConvReluStage::<true, 4>::new(16, 24, 3, 3, 24, weights.conv3_weight, weights.conv3_bias, Some(batch_norm1));
    let fc_stage = FcAccumulateStage::new(24, 22, 22, 32, weights.fc1_weight, weights.fc1_bias);

    let mut row = 0;
    while row < 28 {
        let take = band.0.min(28 - row);
        let data = image[row * 28..(row + take) * 28].to_vec();
        let input_band = RowBand { channels: 1, width: 28, rows: take, data };
        let out1 = stage1.compute_direct(input_band);
        let out2 = stage2.compute_direct(out1);
        let out3 = stage3.compute_direct(out2);
        fc_stage.compute_direct(out3);
        row += take;
    }

    let fc1_out = fc_stage.finalize();
    let fc2_out = matvec_bias(weights.fc2_weight, weights.fc2_bias, &fc1_out, 10, 32);
    let bn2_out = apply_batch_norm(&fc2_out, weights.norm2_weight, weights.norm2_bias, weights.norm2_running_mean, weights.norm2_running_var);
    log_softmax(&bn2_out)
}
