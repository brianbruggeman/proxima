// workspace denies expect_used; a failed fixture must abort the test with its message
#![allow(clippy::expect_used)]

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use arrayvec::ArrayVec;

use proxima_gguf::value::{MetadataArray, MetadataValue};
use proxima_gguf::{GgmlType, GgufModel, TensorPayload, write_complete};
use proxima_tensor::test_support::Lcg;

use std::sync::Arc;

use proxima_primitives::sync::blocking::{Mutex, MutexGuard};
use proxima_telemetry::emit::{EnvFilter, global};
use proxima_telemetry::export::set_default_recorder;
use proxima_telemetry::pipes::InMemoryPipe;
use proxima_telemetry::recorder::Recorder;
use proxima_telemetry::tag::{ScalarValue, Tag};

use super::{
    BackendRuntime, LayerCacheState, LoadedModel, LogitsSink, NodeValuesSink, PrefixState,
    ServingConfig, ubatch_forgoes_row_kernel, ubatch_prefill_chunks,
};

const EMBEDDING: u64 = 16;
const FEED_FORWARD: u64 = 32;
const VOCAB: u64 = 257;
const FULL_HEAD_DIM: u64 = 512;
const SLIDING_HEAD_DIM: u64 = 256;
const SLIDING_WINDOW: u32 = 8;
const GEMMA4_SLIDING_PATTERN: [bool; 3] = [true, true, false];
const DENSE_LAYERS: u64 = 2;
const MAX_REDUCTION: usize = 512;
const DEPENDENT_REDUCTIONS_PER_LAYER: usize = 14;
const UNIT_ROUNDOFF: f64 = 1.0 / 16_777_216.0;

struct TensorSpec {
    name: String,
    dims: Vec<u64>,
    values: Vec<f32>,
}

fn spec(name: &str, dims: &[u64], scale: f32, offset: f32, rng: &mut Lcg) -> TensorSpec {
    let count: u64 = dims.iter().product();
    TensorSpec {
        name: name.to_string(),
        dims: dims.to_vec(),
        values: (0..count)
            .map(|_| offset + scale * rng.next_unit())
            .collect(),
    }
}

fn tokens() -> MetadataValue {
    let mut names: Vec<String> = (0..=255u8)
        .map(|byte| alloc::format!("<0x{byte:02X}>"))
        .collect();
    names.push(String::from("<eos-marker>"));
    MetadataValue::Array(MetadataArray::String(names))
}

fn encode(metadata: Vec<(String, MetadataValue)>, specs: &[TensorSpec]) -> Vec<u8> {
    let payloads: Vec<Vec<u8>> = specs
        .iter()
        .map(|tensor| {
            tensor
                .values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        })
        .collect();
    let model = GgufModel {
        version: 3,
        metadata,
        tensors: specs
            .iter()
            .zip(&payloads)
            .map(|(tensor, data)| TensorPayload {
                name: tensor.name.clone(),
                dims: tensor.dims.iter().copied().collect::<ArrayVec<u64, 4>>(),
                ggml_type: GgmlType::F32,
                data,
            })
            .collect(),
    };
    write_complete(&model).expect("the synthetic checkpoint encodes")
}

fn key(name: &str, value: MetadataValue) -> (String, MetadataValue) {
    (name.to_string(), value)
}

pub(super) fn gemma4_checkpoint() -> Vec<u8> {
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    let mut specs = alloc::vec![
        spec("token_embd.weight", &[EMBEDDING, VOCAB], 1.0, 0.0, &mut rng),
        spec("output_norm.weight", &[EMBEDDING], 0.1, 1.0, &mut rng),
    ];
    let mut rope_frequencies = spec("rope_freqs.weight", &[256], 0.0, 1.0, &mut rng);
    rope_frequencies.values[64..].fill(1.0e30);
    specs.push(rope_frequencies);
    for (layer, &is_sliding) in GEMMA4_SLIDING_PATTERN.iter().enumerate() {
        let head_dim = if is_sliding { SLIDING_HEAD_DIM } else { FULL_HEAD_DIM };
        let name = |suffix: &str| alloc::format!("blk.{layer}.{suffix}");
        specs.extend([
            spec(&name("attn_norm.weight"), &[EMBEDDING], 0.1, 1.0, &mut rng),
            spec(&name("post_attention_norm.weight"), &[EMBEDDING], 0.1, 1.0, &mut rng),
            spec(&name("attn_q_norm.weight"), &[head_dim], 0.1, 1.0, &mut rng),
            spec(&name("attn_k_norm.weight"), &[head_dim], 0.1, 1.0, &mut rng),
            spec(&name("attn_q.weight"), &[EMBEDDING, head_dim], 0.25, 0.0, &mut rng),
            spec(&name("attn_k.weight"), &[EMBEDDING, head_dim], 0.25, 0.0, &mut rng),
            spec(&name("attn_v.weight"), &[EMBEDDING, head_dim], 0.25, 0.0, &mut rng),
            spec(&name("attn_output.weight"), &[head_dim, EMBEDDING], 0.25, 0.0, &mut rng),
            spec(&name("layer_output_scale.weight"), &[1], 0.0, 1.0, &mut rng),
            spec(&name("ffn_norm.weight"), &[EMBEDDING], 0.1, 1.0, &mut rng),
            spec(&name("ffn_gate.weight"), &[EMBEDDING, FEED_FORWARD], 0.25, 0.0, &mut rng),
            spec(&name("ffn_up.weight"), &[EMBEDDING, FEED_FORWARD], 0.25, 0.0, &mut rng),
            spec(&name("ffn_down.weight"), &[FEED_FORWARD, EMBEDDING], 0.25, 0.0, &mut rng),
            spec(&name("post_ffw_norm.weight"), &[EMBEDDING], 0.1, 1.0, &mut rng),
        ]);
    }
    let metadata = alloc::vec![
        key("general.architecture", MetadataValue::String("gemma4".to_string())),
        key("gemma4.embedding_length", MetadataValue::U32(EMBEDDING as u32)),
        key("gemma4.block_count", MetadataValue::U32(GEMMA4_SLIDING_PATTERN.len() as u32)),
        key("gemma4.feed_forward_length", MetadataValue::U32(FEED_FORWARD as u32)),
        key("gemma4.attention.head_count", MetadataValue::U32(1)),
        key("gemma4.attention.head_count_kv", MetadataValue::U32(1)),
        key("gemma4.attention.key_length", MetadataValue::U32(FULL_HEAD_DIM as u32)),
        key("gemma4.attention.value_length", MetadataValue::U32(FULL_HEAD_DIM as u32)),
        key("gemma4.attention.key_length_swa", MetadataValue::U32(SLIDING_HEAD_DIM as u32)),
        key("gemma4.attention.value_length_swa", MetadataValue::U32(SLIDING_HEAD_DIM as u32)),
        key("gemma4.attention.sliding_window", MetadataValue::U32(SLIDING_WINDOW)),
        key(
            "gemma4.attention.sliding_window_pattern",
            MetadataValue::Array(MetadataArray::Bool(GEMMA4_SLIDING_PATTERN.to_vec())),
        ),
        key("gemma4.rope.freq_base", MetadataValue::F32(1.0e6)),
        key("gemma4.rope.freq_base_swa", MetadataValue::F32(1.0e4)),
        key("gemma4.rope.dimension_count", MetadataValue::U32(FULL_HEAD_DIM as u32)),
        key("gemma4.rope.dimension_count_swa", MetadataValue::U32(SLIDING_HEAD_DIM as u32)),
        key("tokenizer.ggml.model", MetadataValue::String("gpt2".to_string())),
        key("tokenizer.ggml.tokens", tokens()),
        key("tokenizer.ggml.merges", MetadataValue::Array(MetadataArray::String(Vec::new()))),
    ];
    encode(metadata, &specs)
}

fn dense_checkpoint() -> Vec<u8> {
    let mut rng = Lcg(0x2545_f491_4f6c_dd1d);
    let projection = EMBEDDING;
    let mut specs = alloc::vec![
        spec("token_embd.weight", &[EMBEDDING, VOCAB], 1.0, 0.0, &mut rng),
        spec("output_norm.weight", &[EMBEDDING], 0.1, 1.0, &mut rng),
        spec("output.weight", &[EMBEDDING, VOCAB], 0.5, 0.0, &mut rng),
    ];
    for layer in 0..DENSE_LAYERS {
        let name = |suffix: &str| alloc::format!("blk.{layer}.{suffix}");
        specs.extend([
            spec(&name("attn_norm.weight"), &[EMBEDDING], 0.1, 1.0, &mut rng),
            spec(&name("ffn_norm.weight"), &[EMBEDDING], 0.1, 1.0, &mut rng),
            spec(&name("attn_q.weight"), &[EMBEDDING, projection], 0.25, 0.0, &mut rng),
            spec(&name("attn_k.weight"), &[EMBEDDING, projection], 0.25, 0.0, &mut rng),
            spec(&name("attn_v.weight"), &[EMBEDDING, projection], 0.25, 0.0, &mut rng),
            spec(&name("attn_output.weight"), &[projection, EMBEDDING], 0.25, 0.0, &mut rng),
            spec(&name("ffn_gate.weight"), &[EMBEDDING, FEED_FORWARD], 0.25, 0.0, &mut rng),
            spec(&name("ffn_up.weight"), &[EMBEDDING, FEED_FORWARD], 0.25, 0.0, &mut rng),
            spec(&name("ffn_down.weight"), &[FEED_FORWARD, EMBEDDING], 0.25, 0.0, &mut rng),
        ]);
    }
    let metadata = alloc::vec![
        key("general.architecture", MetadataValue::String("llama".to_string())),
        key("llama.embedding_length", MetadataValue::U32(EMBEDDING as u32)),
        key("llama.feed_forward_length", MetadataValue::U32(FEED_FORWARD as u32)),
        key("llama.attention.head_count", MetadataValue::U32(1)),
        key("llama.attention.head_count_kv", MetadataValue::U32(1)),
        key("llama.block_count", MetadataValue::U32(DENSE_LAYERS as u32)),
        key("tokenizer.ggml.model", MetadataValue::String("gpt2".to_string())),
        key("tokenizer.ggml.tokens", tokens()),
        key("tokenizer.ggml.merges", MetadataValue::Array(MetadataArray::String(Vec::new()))),
    ];
    encode(metadata, &specs)
}

pub(super) fn config(ubatch_size: u32) -> ServingConfig<'static> {
    ServingConfig {
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size,
        gpu_layers: 0,
        reasoning_budget: 0,
        ..ServingConfig::default()
    }
}

pub(super) struct Prefilled {
    logits: Vec<f32>,
    pub(super) state: PrefixState,
    evaluation_rows: Vec<usize>,
}

fn prefill(model: &LoadedModel<'_>, prompt: &str, ubatch_size: u32) -> Prefilled {
    prefill_with(model, prompt, &config(ubatch_size))
}

pub(super) fn prefill_with(model: &LoadedModel<'_>, prompt: &str, serving_config: &ServingConfig<'_>) -> Prefilled {
    let evaluation_capture = EvaluationCapture::install();
    let mut runtime = BackendRuntime::new(serving_config);
    let mut collected: Vec<Vec<f32>> = Vec::new();
    let (_ids, _text, _eos, state) = model
        .run_decode_loop_observed_seeded(
            prompt,
            1,
            serving_config,
            &mut runtime,
            None,
            &mut LogitsSink::Collect(&mut collected),
            &mut NodeValuesSink::Discard,
            &mut |_event| core::ops::ControlFlow::Continue(()),
            None,
            true,
            None,
            None,
        )
        .expect("the synthetic checkpoint prefills");
    let logits = collected.pop().expect("the last prefill chunk requested logits");
    Prefilled {
        logits,
        state,
        evaluation_rows: evaluation_capture.rows(),
    }
}

// the ambient recorder and level filter are process-wide, so concurrent `prefill` calls
// would read each other's events; a bare std mutex is fine in test code
static AMBIENT_TELEMETRY: Mutex<()> = Mutex::new(());

const EVALUATION_TARGET: &str = "proxima_model_interop::generate::residency_caches";

/// Routes `BackendRuntime::evaluate`'s `trace!(rows = ..)` event into an in-memory pipe, so the
/// per-evaluation row counts are read from telemetry rather than from production state.
struct EvaluationCapture {
    recorder: Arc<Recorder>,
    pipe: InMemoryPipe,
    _serialized: MutexGuard<'static, ()>,
}

impl EvaluationCapture {
    fn install() -> Self {
        let serialized = AMBIENT_TELEMETRY.lock();
        global::install(EnvFilter::parse(&format!("{EVALUATION_TARGET}=trace")));
        let pipe = InMemoryPipe::new();
        let recorder = Arc::new(
            Recorder::builder()
                .pipe(pipe.clone())
                .core_count(1)
                .start()
                .expect("the in-memory evaluation recorder starts"),
        );
        set_default_recorder(Arc::clone(&recorder));
        Self { recorder, pipe, _serialized: serialized }
    }

    fn rows(&self) -> Vec<usize> {
        while self.recorder.drain() > 0 {}
        self.pipe
            .logs()
            .iter()
            .flat_map(|log| log.attrs.iter())
            .filter_map(|tag| match tag {
                Tag::Scalar { key: "rows", value: ScalarValue::U64(rows) } => usize::try_from(*rows).ok(),
                _ => None,
            })
            .collect()
    }
}

fn cache_rows(state: &PrefixState) -> Vec<f32> {
    state
        .layer_caches
        .iter()
        .flat_map(|cache| match cache {
            LayerCacheState::Attention(layer) => {
                [layer.k_even.as_slice(), layer.k_odd.as_slice(), layer.v.as_slice()].concat()
            }
            _ => Vec::new(),
        })
        .collect()
}

fn chunk_rows(width: u32, prompt_rows: usize) -> Vec<usize> {
    let width = width as usize;
    let mut rows = alloc::vec![width; prompt_rows / width];
    if !prompt_rows.is_multiple_of(width) {
        rows.push(prompt_rows % width);
    }
    rows
}

fn gamma(reductions: usize) -> f64 {
    let scaled = reductions as f64 * UNIT_ROUNDOFF;
    scaled / (1.0 - scaled)
}

fn worst_relative_error(reference: &[f32], candidate: &[f32]) -> f64 {
    assert_eq!(reference.len(), candidate.len(), "both runs hold the same row count");
    let scale = reference.iter().fold(0.0f64, |peak, value| peak.max(f64::from(value.abs())));
    reference
        .iter()
        .zip(candidate)
        .map(|(left, right)| f64::from((left - right).abs()) / scale)
        .fold(0.0, f64::max)
}

pub(super) fn prompt_of(length: usize, last_digit: char) -> String {
    let mut digits: String = (0..length - 1)
        .map(|index| char::from_digit((index % 10) as u32, 10).expect("a base-10 digit"))
        .collect();
    digits.push(last_digit);
    digits
}

/// A prefill cut into `ubatch_size`-row chunks must hold the same KV rows and
/// final logits as one evaluation, up to float summation order. Chunking does
/// not change any weight, activation or reduction operand, only how many
/// key terms each softmax and attention-weighted sum adds and in what
/// association, so every output element differs by at most the
/// forward-error bound of the longest reduction, `gamma_n = n*u/(1-n*u)`
/// with `u = 2^-24` (Higham, Accuracy and Stability of Numerical
/// Algorithms, Thm 3.1), compounded once per dependent reduction on the
/// path from embedding to output: `DEPENDENT_REDUCTIONS_PER_LAYER * layers *
/// gamma_n`, relative to the largest reference magnitude. `n` is
/// `MAX_REDUCTION`, the 512-wide full-attention head, which dominates the
/// prompt's key count. Widths 3, 7, 16 and 33 straddle the 8-row sliding
/// window, so the ring wraps inside a chunk (16, 33) and across chunks (3,
/// 7). The control changes the last prompt digit, which must move the
/// output by more than that same bound, or the bound is measuring nothing.
#[test]
fn chunked_prefill_matches_one_evaluation_on_two_range_gemma4() {
    let bytes = gemma4_checkpoint();
    let parsed = proxima_gguf::pipe::parse_complete(&bytes).expect("parses the gemma4 fixture");
    let model = LoadedModel::load(&parsed, &bytes).expect("loads the gemma4 fixture");
    let prompt = prompt_of(40, '3');
    let reference = prefill(&model, &prompt, 0);
    let prompt_rows = reference.state.len();
    let bound = GEMMA4_SLIDING_PATTERN.len() as f64
        * DEPENDENT_REDUCTIONS_PER_LAYER as f64
        * gamma(MAX_REDUCTION.max(prompt_rows));
    assert!(prompt_rows > 33, "the widest chunk must be shorter than the prompt: {prompt_rows}");
    assert_eq!(reference.evaluation_rows, [prompt_rows], "ubatch 0 is one evaluation");

    for width in [3u32, 7, 16, 33] {
        let chunked = prefill(&model, &prompt, width);
        assert_eq!(chunked.state.len(), prompt_rows, "width {width} caches every row");
        assert_eq!(chunked.evaluation_rows, chunk_rows(width, prompt_rows), "width {width}");
        let rows_error = worst_relative_error(&cache_rows(&reference.state), &cache_rows(&chunked.state));
        let logits_error = worst_relative_error(&reference.logits, &chunked.logits);
        assert!(rows_error <= bound, "width {width}: cached rows differ by {rows_error} > {bound}");
        assert!(logits_error <= bound, "width {width}: logits differ by {logits_error} > {bound}");
    }

    let changed = prefill(&model, &prompt_of(40, '4'), 0);
    let control_error = worst_relative_error(&reference.logits, &changed.logits);
    assert!(control_error > bound, "control moved logits by {control_error}, not above {bound}");
}

/// A dense checkpoint's prefill of `L` rows at `-ub C` is `ceil(L / C)`
/// evaluations, each `C` rows except the last, which carries `L mod C`.
#[test]
fn chunked_prefill_evaluation_count_is_ceil_of_rows_over_ubatch() {
    let bytes = dense_checkpoint();
    let parsed = proxima_gguf::pipe::parse_complete(&bytes).expect("parses the dense fixture");
    let model = LoadedModel::load(&parsed, &bytes).expect("loads the dense fixture");
    let prompt = prompt_of(70, '1');
    let prompt_rows = prefill(&model, &prompt, 0).state.len();

    for width in [16u32, 32, 33] {
        let chunked = prefill(&model, &prompt, width);
        assert_eq!(
            chunked.evaluation_rows,
            chunk_rows(width, prompt_rows),
            "width {width} over {prompt_rows} rows"
        );
        assert_eq!(chunked.evaluation_rows.len(), prompt_rows.div_ceil(width as usize));
    }
}

/// `ubatch_size = 0` is the control: exactly one evaluation of every row,
/// today's behaviour before chunking existed.
#[test]
fn chunked_prefill_ubatch_zero_is_one_evaluation() {
    let bytes = dense_checkpoint();
    let parsed = proxima_gguf::pipe::parse_complete(&bytes).expect("parses the dense fixture");
    let model = LoadedModel::load(&parsed, &bytes).expect("loads the dense fixture");

    let control = prefill(&model, &prompt_of(70, '1'), 0);

    assert_eq!(control.evaluation_rows, [control.state.len()]);
}

/// The chunk count helper's edges: the control, a prompt that fits one
/// micro-batch, an exact multiple, and one row over.
#[test]
fn chunked_prefill_chunk_count_edges() {
    assert_eq!(ubatch_prefill_chunks(0, 7895), 1);
    assert_eq!(ubatch_prefill_chunks(32, 32), 1);
    assert_eq!(ubatch_prefill_chunks(32, 64), 2);
    assert_eq!(ubatch_prefill_chunks(32, 65), 3);
    assert_eq!(ubatch_prefill_chunks(32, 7895), 247);
}

/// The debug event's predicate: only a nonzero `ubatch` under the kernel's
/// row threshold, on a prompt that alone would have cleared it, is the
/// configuration's doing. 160 is `omega-runtime.toml`'s `[tiled_gemm].min_tokens`.
#[test]
fn ubatch_below_the_row_threshold_forgoes_the_kernel_only_for_a_long_enough_prompt() {
    assert!(ubatch_forgoes_row_kernel(32, 971, 160), "-ub 32 on a 971-row prompt");
    assert!(!ubatch_forgoes_row_kernel(512, 971, 160), "-ub 512 clears the threshold");
    assert!(!ubatch_forgoes_row_kernel(160, 971, 160), "-ub equal to the threshold clears it");
    assert!(!ubatch_forgoes_row_kernel(0, 971, 160), "-ub 0 is the single pass");
    assert!(!ubatch_forgoes_row_kernel(32, 100, 160), "a short prompt misses the kernel on its own");
}
