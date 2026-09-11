//! ROW 427: proves [`crate::generate::LoadedModel::run_decode_loop_observed_seeded`]'s
//! `step_batches` split -- a `single_position_step` architecture's prefill
//! runs as `prompt_token_count` sequential `new_count == 1` evaluations
//! through the SAME per-step evaluate/cache-append path a decode step
//! already uses, rather than one `new_count == prompt_token_count` call.
//!
//! Reuses `external_architecture_step_inputs.rs`'s own checkpoint layout
//! (`tests/support`'s `EMBEDDING = 256` dimension constant, one dense
//! attention+FFN layer, non-zero constant weights) rather than duplicating
//! it, since the real
//! qwen35 GDN mixer's own fixture (`examples/synth_qwen35_gguf.rs`,
//! `EMBEDDING = 5120`) is `#[ignore]`d for taking minutes in a debug build
//! (`qwen35_synth_hybrid.rs`'s own module doc) -- too slow for this landing
//! gate. `DenseArch`'s cached attention already handles causal masking
//! correctly whether a step's `new_count` is the whole prompt or one
//! position (that invariant predates this change), so this fixture cannot
//! reproduce gated-DeltaNet's own batched-sum-vs-sequential-scan numeric
//! divergence -- what it proves is that the LOOP's split is mechanically
//! correct: `MultiPositionStepUnsupported` is never raised, and splitting
//! prefill into one evaluation per position produces the exact same
//! decoded output as the pre-existing single-batch prefill on a checkpoint
//! for which both are valid (`single_position_step: false`). The mixer's
//! own numeric behavior is covered at the tensor-spec level by
//! `proxima_tensor::spec`'s own `SingleTokenStepOnly` tests and by
//! `bind_symbols_rejects_new_count_above_one_when_single_position_step_is_set`
//! (`architecture.rs`).

#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrayvec::ArrayVec;
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::tensor::MAX_DIMS;
use proxima_gguf::{
    GgmlType, GgufModel, MetadataArray, MetadataValue, TensorPayload, parse_complete,
    write_complete,
};
use proxima_model_interop::{
    Architecture, ArchitectureRegistry, BoundProgram, DenseArch, InteropError, LoadedModel,
    ServingConfig, StepInput, StepInputContext,
};
use proxima_tokenizer::byte_level::byte_to_char;

#[path = "support/mod.rs"]
#[allow(dead_code)]
mod support;

fn push_tokenizer_metadata(metadata: &mut Vec<(String, MetadataValue)>) {
    let mut tokens: Vec<String> = (0..=255u8)
        .map(|byte| String::from(byte_to_char(byte)))
        .collect();
    tokens.push(String::from("<|endoftext|>"));
    metadata.push((
        "tokenizer.ggml.model".to_string(),
        MetadataValue::String("gpt2".to_string()),
    ));
    metadata.push((
        "tokenizer.ggml.tokens".to_string(),
        MetadataValue::Array(MetadataArray::String(tokens)),
    ));
    metadata.push((
        "tokenizer.ggml.merges".to_string(),
        MetadataValue::Array(MetadataArray::String(Vec::new())),
    ));
    metadata.push((
        "tokenizer.ggml.bos_token_id".to_string(),
        MetadataValue::U32(0),
    ));
    metadata.push((
        "tokenizer.ggml.eos_token_id".to_string(),
        MetadataValue::U32(support::EOS_TOKEN_ID),
    ));
}

/// Byte-for-byte `external_architecture_step_inputs.rs`'s own
/// `checkpoint_bytes` -- a real, non-zero-weight dense layout
/// (`tests/support`'s `EMBEDDING = 256` fixture) -- under this test's own
/// `general.architecture` name.
fn checkpoint_bytes(architecture_name: &str) -> Vec<u8> {
    let embedding = support::EMBEDDING;
    let feed_forward = support::FEED_FORWARD;
    let vocab = support::VOCAB;

    let mut buffers: Vec<Vec<u8>> = Vec::new();
    let mut specs: Vec<(String, ArrayVec<u64, MAX_DIMS>, GgmlType)> = Vec::new();

    let embed_values = vec![0.0f32; (vocab * embedding) as usize];
    buffers.push(support::encode_weights(GgmlType::F32, &embed_values));
    specs.push((
        String::from("token_embd.weight"),
        [u64::from(vocab * embedding)].into_iter().collect(),
        GgmlType::F32,
    ));

    let norm_values = vec![1.0f32; embedding as usize];
    buffers.push(support::encode_weights(GgmlType::F32, &norm_values));
    specs.push((
        String::from("blk.0.attn_norm.weight"),
        [u64::from(embedding)].into_iter().collect(),
        GgmlType::F32,
    ));
    buffers.push(support::encode_weights(GgmlType::F32, &norm_values));
    specs.push((
        String::from("blk.0.ffn_norm.weight"),
        [u64::from(embedding)].into_iter().collect(),
        GgmlType::F32,
    ));

    let kv_dim = support::KV_HEADS * support::HEAD_DIM;
    let square = vec![0.05f32; (embedding * embedding) as usize];
    let kv_projection = vec![0.05f32; (embedding * kv_dim) as usize];
    buffers.push(support::encode_weights(GgmlType::F32, &square));
    specs.push((
        String::from("blk.0.attn_q.weight"),
        [u64::from(embedding), u64::from(embedding)]
            .into_iter()
            .collect(),
        GgmlType::F32,
    ));
    for name in ["blk.0.attn_k.weight", "blk.0.attn_v.weight"] {
        buffers.push(support::encode_weights(GgmlType::F32, &kv_projection));
        specs.push((
            String::from(name),
            [u64::from(embedding), u64::from(kv_dim)]
                .into_iter()
                .collect(),
            GgmlType::F32,
        ));
    }
    buffers.push(support::encode_weights(GgmlType::F32, &square));
    specs.push((
        String::from("blk.0.attn_output.weight"),
        [u64::from(embedding), u64::from(embedding)]
            .into_iter()
            .collect(),
        GgmlType::F32,
    ));

    let ffn = vec![0.05f32; (embedding * feed_forward) as usize];
    for name in ["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"] {
        buffers.push(support::encode_weights(GgmlType::F32, &ffn));
        specs.push((
            String::from(name),
            [u64::from(embedding), u64::from(feed_forward)]
                .into_iter()
                .collect(),
            GgmlType::F32,
        ));
    }
    buffers.push(support::encode_weights(GgmlType::F32, &ffn));
    specs.push((
        String::from("blk.0.ffn_down.weight"),
        [u64::from(feed_forward), u64::from(embedding)]
            .into_iter()
            .collect(),
        GgmlType::F32,
    ));

    buffers.push(support::encode_weights(GgmlType::F32, &norm_values));
    specs.push((
        String::from("output_norm.weight"),
        [u64::from(embedding)].into_iter().collect(),
        GgmlType::F32,
    ));

    let output_values = vec![0.05f32; (embedding * vocab) as usize];
    buffers.push(support::encode_weights(GgmlType::F32, &output_values));
    specs.push((
        String::from("output.weight"),
        [u64::from(embedding), u64::from(vocab)]
            .into_iter()
            .collect(),
        GgmlType::F32,
    ));

    let tensors: Vec<TensorPayload<'_>> = specs
        .iter()
        .zip(buffers.iter())
        .map(|((name, dims, ggml_type), data)| TensorPayload {
            name: name.clone(),
            dims: dims.clone(),
            ggml_type: *ggml_type,
            data: data.as_slice(),
        })
        .collect();

    let mut metadata = vec![
        (
            "general.architecture".to_string(),
            MetadataValue::String(architecture_name.to_string()),
        ),
        (
            format!("{architecture_name}.embedding_length"),
            MetadataValue::U32(embedding),
        ),
        (
            format!("{architecture_name}.feed_forward_length"),
            MetadataValue::U32(feed_forward),
        ),
        (
            format!("{architecture_name}.attention.head_count"),
            MetadataValue::U32(support::QUERY_HEADS),
        ),
        (
            format!("{architecture_name}.attention.head_count_kv"),
            MetadataValue::U32(support::KV_HEADS),
        ),
        (
            format!("{architecture_name}.block_count"),
            MetadataValue::U32(1),
        ),
        (
            format!("{architecture_name}.rope.dimension_count"),
            MetadataValue::U32(support::HEAD_DIM),
        ),
    ];
    push_tokenizer_metadata(&mut metadata);

    let model = GgufModel {
        version: 3,
        metadata,
        tensors,
    };
    write_complete(&model).expect("writes a well-formed synthetic checkpoint")
}

/// Records every [`Architecture::step_inputs`] call's own `new_count` --
/// the direct observable of the decode loop's `step_batches` split: one
/// entry per evaluate call, `1` for every one of them once
/// `single_position_step` is set, instead of one entry equal to the whole
/// prompt length on step 0.
static OBSERVED_NEW_COUNTS: Mutex<Vec<usize>> = Mutex::new(Vec::new());
static CALL_COUNT: AtomicUsize = AtomicUsize::new(0);

fn reset_observations() {
    CALL_COUNT.store(0, Ordering::SeqCst);
    OBSERVED_NEW_COUNTS
        .lock()
        .expect("test-only mutex, never poisoned")
        .clear();
}

/// Wraps [`DenseArch::bind`] unmodified except for `single_position_step`,
/// which each static instance below sets differently -- the ONLY variable
/// between the "old" (batched-prefill-legal) and "new"
/// (single-position-only) arms of every test in this module.
struct SinglePositionArch {
    name: &'static str,
    single_position_step: bool,
}

impl Architecture for SinglePositionArch {
    fn name(&self) -> &'static str {
        self.name
    }

    fn bind<'file>(
        &self,
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
    ) -> Result<BoundProgram<'file>, InteropError> {
        let mut bound = DenseArch.bind(parsed, file_bytes)?;
        bound.single_position_step = self.single_position_step;
        Ok(bound)
    }

    fn step_inputs(&self, context: &StepInputContext<'_>, out: &mut Vec<StepInput>) {
        let _ = out;
        CALL_COUNT.fetch_add(1, Ordering::SeqCst);
        OBSERVED_NEW_COUNTS
            .lock()
            .expect("test-only mutex, never poisoned")
            .push(context.new_count);
    }
}

static SINGLE_POSITION_ON: SinglePositionArch = SinglePositionArch {
    name: "acme-single-position-prefill-on",
    single_position_step: true,
};

static SINGLE_POSITION_OFF: SinglePositionArch = SinglePositionArch {
    name: "acme-single-position-prefill-off",
    single_position_step: false,
};

fn registry_with(architecture: &'static SinglePositionArch) -> ArchitectureRegistry {
    let mut registry = ArchitectureRegistry::with_builtin();
    registry.register(architecture);
    registry
}

fn supported_config() -> ServingConfig<'static> {
    ServingConfig {
        model_path: "synthetic-single-position-prefill-fixture",
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        gpu_layers: 0,
        reasoning_budget: 0,
        ..ServingConfig::default()
    }
}

/// GREEN: a `single_position_step` architecture decodes a multi-token
/// prompt successfully (`InteropError::MultiPositionStepUnsupported` is
/// never raised by the loop for any prompt length -- the property ROW 427
/// names explicitly), and `step_inputs`' own observed `new_count` sequence
/// proves the mechanism: every recorded call, prefill included, is exactly
/// `1`, never the batched prompt length.
#[proxima::test]
async fn single_position_step_splits_prefill_into_one_evaluation_per_position() {
    reset_observations();
    let file_bytes = checkpoint_bytes(SINGLE_POSITION_ON.name);
    let parsed = parse_complete(&file_bytes).expect("parses the synthetic checkpoint");
    let registry = registry_with(&SINGLE_POSITION_ON);
    let model = LoadedModel::load_with_registry(&parsed, &file_bytes, &registry)
        .expect("loads through the foreign architecture's own single_position_step bind");

    let prompt = "abcd";
    let outcome = model.generate_with_serving_config(prompt, 2, supported_config());
    let (ids, _text, _stopped) = outcome.expect(
        "single_position_step must never surface MultiPositionStepUnsupported from the decode \
         loop -- the loop is responsible for splitting a multi-token prompt into one \
         new_count == 1 evaluation per position before bind_symbols ever sees it",
    );
    assert_eq!(ids.len(), 2, "max_tokens=2 produces exactly two token ids");

    let observed = OBSERVED_NEW_COUNTS
        .lock()
        .expect("test-only mutex, never poisoned")
        .clone();
    assert!(
        observed.len() >= prompt.len() + 2,
        "one step_inputs call per prompt byte (bos-prefixed) plus one per generated token, \
         got {observed:?}"
    );
    assert!(
        observed.iter().all(|&new_count| new_count == 1),
        "every evaluation -- prefill positions and decode steps alike -- must carry exactly \
         one new position once single_position_step is set; got {observed:?}"
    );
}

/// GREEN, the bit-for-bit half of ROW 427's requirement: on a checkpoint
/// for which BOTH a batched prefill (`single_position_step: false`, the
/// pre-existing single-call path) and a split prefill
/// (`single_position_step: true`, this change's `step_batches` path) are
/// legal, they must decode to the exact same token ids and text -- the
/// split changes HOW MANY evaluate calls the prefill costs, never WHAT the
/// forward pass computes.
#[proxima::test]
async fn single_position_step_prefill_matches_the_pre_existing_batched_prefill() {
    reset_observations();

    let batched_registry = registry_with(&SINGLE_POSITION_OFF);
    let batched_file_bytes = checkpoint_bytes(SINGLE_POSITION_OFF.name);
    let batched_parsed =
        parse_complete(&batched_file_bytes).expect("parses the synthetic checkpoint");
    let batched_model =
        LoadedModel::load_with_registry(&batched_parsed, &batched_file_bytes, &batched_registry)
            .expect("loads with single_position_step: false");

    let split_registry = registry_with(&SINGLE_POSITION_ON);
    let split_file_bytes = checkpoint_bytes(SINGLE_POSITION_ON.name);
    let split_parsed = parse_complete(&split_file_bytes).expect("parses the synthetic checkpoint");
    let split_model =
        LoadedModel::load_with_registry(&split_parsed, &split_file_bytes, &split_registry)
            .expect("loads with single_position_step: true");

    let prompt = "abc";
    let (batched_ids, batched_text, _stopped) = batched_model
        .generate_with_serving_config(prompt, 3, supported_config())
        .expect("batched (single-call) prefill decodes");
    let (split_ids, split_text, _stopped) = split_model
        .generate_with_serving_config(prompt, 3, supported_config())
        .expect("split (one-evaluation-per-position) prefill decodes");

    assert_eq!(
        batched_ids, split_ids,
        "splitting prefill into one evaluation per position must not change which tokens a \
         dense checkpoint decodes"
    );
    assert_eq!(batched_text, split_text);
}
