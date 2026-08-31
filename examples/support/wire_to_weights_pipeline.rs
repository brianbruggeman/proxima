//! The wire-to-weights request -> real mnist.onnx forward -> response
//! pipeline. Shared, via `#[path]`, between the runnable demo
//! (`examples/wire_to_weights.rs`) and its smoke test
//! (`tests/wire_to_weights_smoke.rs`) — the same example/support-sharing
//! convention `proxima-onnx/benches/support/tile_pipeline.rs` uses for its
//! own bench + differential-test pair (see
//! `proxima-onnx/tests/tile_pipeline_differential.rs`'s `#[path =
//! "../benches/support/tile_pipeline.rs"]`).
//!
//! # The composition
//!
//! Three [`SendPipe`]s chained through the pipe algebra's own composition
//! law, [`AndThen`](proxima::pipe::AndThen) — `First::Out` feeds
//! `Second::In`, and every stage is pinned to `Err = ProximaError`, so
//! `Second::Err: From<First::Err>` is the trivial identity conversion and
//! no error-mapping glue is needed at the seam:
//!
//! ```text
//! AndThen::new(ParseImage, AndThen::new(Classify, RenderResponse))
//! ```
//!
//! `ParseImage` is `Request<Bytes> -> [f32; 784]` (route + body-shape
//! admission, mirroring [`filter`](proxima::pipe)'s `Decide<In>` shape but
//! folded into one stage rather than composed from the standalone `Filter`
//! combinator — noted honestly below). `Classify` is `[f32; 784] ->
//! (Vec<f32>, usize)`, the real forward pass via
//! [`proxima_tensor::cpu::evaluate_named`] over a checkpoint parsed and
//! lowered once at startup (mirrors
//! `proxima-onnx/tests/real_mnist_accuracy.rs`'s own
//! `parse_complete` -> `lower_graph` -> `evaluate_named` chain). `RenderResponse`
//! is `(Vec<f32>, usize) -> Response<Bytes>`.
//!
//! The resulting chain type already satisfies
//! `SendPipe<In = Request<Bytes>, Out = Response<Bytes>, Err =
//! ProximaError>` — exactly [`Handler`](proxima::pipe::Handler)'s blanket
//! bound — so [`into_handle`] takes it directly with zero adapter code.
//!
//! # The honest seam
//!
//! `proxima_tensor::cpu::evaluate_named` is a synchronous function, not a
//! `Pipe` — the tensor executor is array-in/array-out compute, not
//! request/response dataflow, so there is no pipe form to compose it
//! through; `Classify::call` wraps the synchronous call in an `async move`
//! block to satisfy `SendPipe::call`'s signature, which is the entire
//! adapter. Likewise route admission (`POST /classify` only) is written as
//! a plain guard inside `ParseImage::call` rather than a standalone
//! [`Filter`](proxima::pipe) stage in front of it — a single fixed route
//! has no second candidate for `Filter` to admit-or-reject between, so
//! composing a whole extra pipe stage for one `if` would be decoration,
//! not algebra.

#![allow(dead_code)]

use std::fs;
use std::future::Future;
use std::path::Path;
use std::sync::Arc;

use bytes::Bytes;
use proxima::error::ProximaError;
use proxima::pipe::{AndThen, PipeHandle, SendPipe, into_handle};
use proxima::request::{Request, Response};
use proxima_tensor::{NodeId, Op};

/// 28x28 raw pixels, row-major — the same shape
/// `real_mnist_accuracy.rs::load_normalized_images` produces per image.
pub const INPUT_PIXELS: usize = 28 * 28;
/// mnist.onnx classifies into 10 digit classes (`LogSoftmax` over 10).
pub const OUTPUT_CLASSES: usize = 10;

/// The real `mnist.onnx` checkpoint, parsed and lowered exactly once at
/// startup (never per-request) — the same
/// `parse_complete` -> `lower_graph` split
/// `proxima-onnx/tests/real_mnist_checkpoint.rs` and
/// `real_mnist_accuracy.rs` both exercise, with the model bytes and
/// `ModelProto` borrow dropped once `Lowered` (fully owned) is built.
pub struct ModelState {
    program: Vec<Op>,
    initializers: Vec<(String, Vec<f32>)>,
    graph_input_name: String,
    output_node: NodeId,
}

/// Parse + lower the real, on-disk `mnist.onnx` checkpoint once. Returns
/// `Ok(None)` (not an error) when the host-local checkout is absent — the
/// presence-guard convention `real_mnist_accuracy.rs::checkpoint_present`
/// uses, so a caller without the checkout gets a clean skip, not a panic.
pub fn load_model(path: &Path) -> Result<Option<ModelState>, ProximaError> {
    if !path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(path).map_err(|error| ProximaError::Config(format!("read {}: {error}", path.display())))?;
    let model = proxima_onnx::pipe::parse_complete(&bytes).map_err(|error| ProximaError::Config(format!("parse mnist.onnx: {error}")))?;
    let graph = model.graph.as_ref().ok_or_else(|| ProximaError::Config("mnist.onnx has no graph".into()))?;
    let lowered = proxima_onnx::lower::lower_graph(graph).map_err(|error| ProximaError::Config(format!("lower mnist.onnx: {error}")))?;

    let graph_input_name = lowered
        .graph_inputs
        .first()
        .ok_or_else(|| ProximaError::Config("mnist.onnx declares no graph input".into()))?
        .clone();
    let output_node = lowered
        .graph_outputs
        .first()
        .ok_or_else(|| ProximaError::Config("mnist.onnx declares no graph output".into()))?
        .1;

    Ok(Some(ModelState {
        program: lowered.program,
        initializers: lowered.initializers,
        graph_input_name,
        output_node,
    }))
}

/// Build the served pipe handle from a loaded model — the demo binary and
/// the smoke test both call this so the composition is written exactly
/// once.
#[must_use]
pub fn build_handler(model: Arc<ModelState>) -> PipeHandle {
    into_handle(AndThen::new(ParseImage, AndThen::new(Classify { model }, RenderResponse)))
}

/// `Request<Bytes> -> [f32; INPUT_PIXELS]`. Admits only `POST /classify`
/// carrying exactly `INPUT_PIXELS * 4` bytes of little-endian `f32` pixel
/// data — raw LE floats over a JSON array because the request body is
/// already a fixed-size, dense, numeric buffer; decoding it is a
/// `chunks_exact(4)` walk with no text-parsing allocation, where a JSON
/// array would cost a parse pass plus one `String`/`Vec<Value>` per
/// request for the same 784 numbers.
struct ParseImage;

impl SendPipe for ParseImage {
    type In = Request<Bytes>;
    type Out = [f32; INPUT_PIXELS];
    type Err = ProximaError;

    fn call(&self, request: Request<Bytes>) -> impl Future<Output = Result<[f32; INPUT_PIXELS], ProximaError>> + Send {
        async move {
            if request.method != "POST" || request.path.as_ref() != b"/classify" {
                return Err(ProximaError::NotFound(format!("no route for {:?} {}", request.method, String::from_utf8_lossy(&request.path))));
            }
            let (_, body) = request.body_bytes().await?;
            let expected = INPUT_PIXELS * 4;
            if body.len() != expected {
                return Err(ProximaError::Decode(format!(
                    "expected {expected} raw little-endian f32 bytes ({INPUT_PIXELS} pixels), got {} bytes",
                    body.len()
                )));
            }
            let mut pixels = [0.0_f32; INPUT_PIXELS];
            let (chunks, _remainder) = body.as_chunks::<4>();
            for (chunk, slot) in chunks.iter().zip(pixels.iter_mut()) {
                *slot = f32::from_le_bytes(*chunk);
            }
            Ok(pixels)
        }
    }
}

/// `[f32; INPUT_PIXELS] -> (Vec<f32>, usize)`. The real forward pass: runs
/// the checkpoint's lowered `Op` program via
/// [`proxima_tensor::cpu::evaluate_named`] and returns the 10 log-probabilities
/// plus their argmax digit.
struct Classify {
    model: Arc<ModelState>,
}

impl SendPipe for Classify {
    type In = [f32; INPUT_PIXELS];
    type Out = (Vec<f32>, usize);
    type Err = ProximaError;

    fn call(&self, pixels: [f32; INPUT_PIXELS]) -> impl Future<Output = Result<(Vec<f32>, usize), ProximaError>> + Send {
        async move {
            let mut named: Vec<(&str, &[f32])> =
                self.model.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
            named.push((self.model.graph_input_name.as_str(), pixels.as_slice()));

            let evaluated = proxima_tensor::cpu::evaluate_named(&self.model.program, &[], &named, &[self.model.output_node])
                .map_err(|error| ProximaError::Config(format!("mnist forward failed: {error}")))?;
            let (logits, shape) =
                evaluated.get(self.model.output_node).ok_or_else(|| ProximaError::Config("mnist forward produced no output".into()))?;
            if shape != [1_u64, OUTPUT_CLASSES as u64] {
                return Err(ProximaError::Config(format!("expected a 1x{OUTPUT_CLASSES} logit row, got shape {shape:?}")));
            }
            let predicted = argmax(logits);
            Ok((logits.to_vec(), predicted))
        }
    }
}

/// `(Vec<f32>, usize) -> Response<Bytes>`. Renders the 10 log-probabilities
/// and the argmax digit as a small JSON object, hand-written rather than
/// via `serde_json` — the shape is fixed (an f32 array plus one integer),
/// so a `core::fmt::Write` pass is the whole job and adds no dependency to
/// this default-off demo feature.
struct RenderResponse;

impl SendPipe for RenderResponse {
    type In = (Vec<f32>, usize);
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(&self, (logits, predicted): (Vec<f32>, usize)) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move {
            use std::fmt::Write as _;

            let mut body = String::with_capacity(128);
            body.push_str("{\"log_probs\":[");
            for (index, value) in logits.iter().enumerate() {
                if index > 0 {
                    body.push(',');
                }
                let _ = write!(body, "{value}");
            }
            let _ = write!(body, "],\"digit\":{predicted}}}");

            Ok(Response::ok(body).with_header("content-type", "application/json"))
        }
    }
}

fn argmax(values: &[f32]) -> usize {
    values.iter().enumerate().max_by(|left, right| left.1.total_cmp(right.1)).map(|(index, _)| index).unwrap_or(0)
}
