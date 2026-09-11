//! Proves the seam [`crate::architecture`]'s own module doc names: a crate
//! that has never sent proxima a PR implements [`Architecture`], registers
//! it against its own [`ArchitectureRegistry`], and loads a checkpoint
//! through [`LoadedModel::load_with_registry`] -- the same public surface
//! this test file's own crate (`proxima-model-interop`) exercises for its
//! own `qwen35`/`dense` builtins, exercised here from OUTSIDE the crate,
//! through nothing but `pub` API.
//!
//! Composes [`proxima_model_interop::DenseArch`]'s own `bind` (the builtin
//! fallback architecture) rather than reassembling `bind_all_weights` +
//! `mistral_cached_forward_program_with_experts` by hand -- a foreign
//! architecture that wants "exactly what dense checkpoints already do,
//! under a different `general.architecture` name" delegates to it instead
//! of duplicating it, same as any other [`Pipe`] composition.

#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::atomic::{AtomicBool, Ordering};

use arrayvec::ArrayVec;
use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::tensor::MAX_DIMS;
use proxima_gguf::{
    GgmlType, GgufModel, MetadataArray, MetadataValue, TensorPayload, parse_complete,
    write_complete,
};
use proxima_model_interop::{
    Architecture, ArchitectureRegistry, BoundProgram, DenseArch, InteropError, LoadedModel,
};
use proxima_primitives::pipe::Pipe;
use proxima_tokenizer::byte_level::byte_to_char;

// this test binary only exercises a handful of `support`'s fixture-building
// helpers (`EMBEDDING`/`VOCAB`/`encode_weights`/...) -- the rest are real,
// consumed by this crate's OTHER integration test binaries that share this
// same file via an identical `#[path]` include (`capability_matrix.rs`,
// `qwen35_synth_hybrid.rs`); each test binary compiles its own copy, so
// dead-code analysis runs per-binary against the whole shared file.
#[path = "support/mod.rs"]
#[allow(dead_code)]
mod support;

const FOREIGN_ARCHITECTURE_NAME: &str = "acme-dense";

/// The same real GPT-2 byte-level BPE vocab `support::tokenizer_metadata`
/// (private to that module) builds, inlined here rather than widening that
/// shared fixture's own visibility for one external-crate test file.
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

/// A checkpoint's whole metadata + tensor set, built the same way
/// `support::checkpoint_bytes` builds a dense `llama` checkpoint, except
/// every `{architecture}.*` metadata key is prefixed with
/// [`FOREIGN_ARCHITECTURE_NAME`] instead of `llama` -- proves the registry
/// seam, not a second copy of the dense fixture's own tensor layout.
fn foreign_checkpoint_bytes() -> Vec<u8> {
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
    let square = vec![0.0f32; (embedding * embedding) as usize];
    let kv_projection = vec![0.0f32; (embedding * kv_dim) as usize];
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

    let ffn = vec![0.0f32; (embedding * feed_forward) as usize];
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

    let output_values = vec![0.0f32; (embedding * vocab) as usize];
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
            MetadataValue::String(FOREIGN_ARCHITECTURE_NAME.to_string()),
        ),
        (
            format!("{FOREIGN_ARCHITECTURE_NAME}.embedding_length"),
            MetadataValue::U32(embedding),
        ),
        (
            format!("{FOREIGN_ARCHITECTURE_NAME}.feed_forward_length"),
            MetadataValue::U32(feed_forward),
        ),
        (
            format!("{FOREIGN_ARCHITECTURE_NAME}.attention.head_count"),
            MetadataValue::U32(support::QUERY_HEADS),
        ),
        (
            format!("{FOREIGN_ARCHITECTURE_NAME}.attention.head_count_kv"),
            MetadataValue::U32(support::KV_HEADS),
        ),
        (
            format!("{FOREIGN_ARCHITECTURE_NAME}.block_count"),
            MetadataValue::U32(1),
        ),
        (
            format!("{FOREIGN_ARCHITECTURE_NAME}.rope.dimension_count"),
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

/// A checkpoint family a foreign crate maintains entirely on its own --
/// [`Architecture::bind`] just wraps [`DenseArch`]'s own bind (this crate's
/// builtin fallback), the same "delegate to an existing arch" shape any
/// dense-shaped foreign checkpoint family would take.
struct ForeignArchitecture;

static FOREIGN_BIND_CALLED: AtomicBool = AtomicBool::new(false);
static FOREIGN: ForeignArchitecture = ForeignArchitecture;

impl Architecture for ForeignArchitecture {
    fn name(&self) -> &'static str {
        FOREIGN_ARCHITECTURE_NAME
    }

    fn bind<'file>(
        &self,
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
    ) -> Result<BoundProgram<'file>, InteropError> {
        FOREIGN_BIND_CALLED.store(true, Ordering::SeqCst);
        DenseArch.bind(parsed, file_bytes)
    }
}

#[proxima::test]
async fn a_foreign_architecture_loads_a_checkpoint_through_load_with_registry() {
    FOREIGN_BIND_CALLED.store(false, Ordering::SeqCst);
    let file_bytes = foreign_checkpoint_bytes();
    let parsed = parse_complete(&file_bytes).expect("parses the foreign checkpoint");

    let mut registry = ArchitectureRegistry::with_builtin();
    registry.register(&FOREIGN);
    let resolved = registry
        .resolve(&parsed)
        .expect("resolves the foreign architecture by its own general.architecture name");
    assert_eq!(resolved.name(), FOREIGN_ARCHITECTURE_NAME);

    let model = LoadedModel::load_with_registry(&parsed, &file_bytes, &registry)
        .expect("loads the checkpoint through the foreign architecture's own bind");
    assert!(
        FOREIGN_BIND_CALLED.load(Ordering::SeqCst),
        "load_with_registry must dispatch through the SAME architecture resolve() picked"
    );

    let (ids, text, _stopped_by_eos) = Pipe::call(&model, ("a".to_string(), 1))
        .await
        .expect("a single greedy decode step runs on the foreign-architecture-bound program");
    assert_eq!(
        ids.len(),
        1,
        "a max_tokens=1 budget produces exactly one token id"
    );
    assert!(
        !text.is_empty(),
        "the decoded id must round-trip to non-empty text"
    );
}

/// [`ArchitectureRegistry::with_builtin`] always sets [`DenseArch`] as its
/// default (`architecture.rs`'s own `with_builtin`), so it can never return
/// [`InteropError::UnknownArchitecture`] on its own -- a checkpoint under
/// ANY `general.architecture` name loads through the name-blind dense
/// fallback. A caller that wants strict architecture matching -- reject
/// anything it has not explicitly registered, rather than silently binding
/// it as dense -- builds a registry with no default, the same shape
/// [`ArchitectureRegistry::resolve`]'s own doc describes for that case.
/// This is the sad path [`Self::load_with_registry`] exposes: without the
/// foreign architecture (or ANY default) registered, this checkpoint's own
/// `general.architecture` name resolves to nothing, and the typed error
/// comes back instead of a silently-wrong bind.
#[proxima::test]
async fn a_strict_registry_with_no_default_and_no_foreign_arch_returns_the_typed_error() {
    let file_bytes = foreign_checkpoint_bytes();
    let parsed = parse_complete(&file_bytes).expect("parses the foreign checkpoint");

    let strict_registry = ArchitectureRegistry::new();

    match LoadedModel::load_with_registry(&parsed, &file_bytes, &strict_registry) {
        Err(InteropError::UnknownArchitecture { name }) => {
            assert_eq!(name, FOREIGN_ARCHITECTURE_NAME);
        }
        Ok(_) => {
            panic!("expected InteropError::UnknownArchitecture for an unregistered strict registry")
        }
        Err(other) => panic!("unexpected error: {other}"),
    }
}
