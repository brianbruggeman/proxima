//! Proves the decode loop seeds cache leaves from what the compiled
//! program actually declares, not from `Qwen35LayerRoots`'s own hand-kept
//! discriminant -- the real-world defect this file reproduces at tiny
//! scale: a foreign `Architecture` binds a hybrid (SSM + full-attention)
//! program via `qwen35_forward_program` (itself `append_qwen35_dense_attention_only`
//! and `append_qwen35_ssm_mixer` interleaved, `proxima-tensor/src/spec.rs`'s
//! own `qwen35_forward_program` doc), but its `bind` tags one layer's
//! `layer_roots` entry with the WRONG cache-shape variant -- the program
//! still declares `kv_cache.{layer}.k_first` as an `Op::Input`, but the old
//! decode loop trusted the (wrong) enum tag to decide which leaf NAMES to
//! feed, so it fed `kv_cache.{layer}.k_even` instead and the real leaf was
//! never bound, surfacing as `MissingStepInput { name: "kv_cache.{layer}.k_first" }`.
//!
//! Two cases: a correctly-tagged hybrid decodes 3 tokens with no missing
//! step input (the mechanism works for a well-behaved foreign arch); a
//! deliberately mistagged one is rejected loudly and specifically
//! (`LayerCacheKindMismatch`) at decode-loop setup, instead of surfacing as
//! a confusing `MissingStepInput` deep in a later step.

#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::{GgmlType, GgufModel, MetadataArray, MetadataValue, TensorPayload, parse_complete, write_complete};
use proxima_model_interop::{
    Architecture, ArchitectureRegistry, BoundProgram, InteropError, LoadedModel, Qwen35Arch, StepState,
};
use proxima_primitives::pipe::Pipe;
use proxima_tensor::spec::Qwen35LayerRoots;

// Tiny hybrid dims -- large enough to satisfy `qwen35_forward_program`'s
// own shape relations (`ssm_key_dim = state_size * group_count`,
// `qkv_dim = 2 * ssm_key_dim + inner_size`, `head_v_dim = inner_size /
// time_step_rank`), small enough that a debug-build decode of 2 layers
// finishes in well under a second -- `examples/synth_qwen35_gguf.rs`'s own
// doc: the real (5120-embedding) fixture takes minutes in debug, which is
// why that one is `#[ignore]`d and this one is not.
const EMBEDDING: u32 = 32;
const FEED_FORWARD: u32 = 32;
const VOCAB: u32 = 8;
const QUERY_HEADS: u32 = 2;
const KV_HEADS: u32 = 1;
const ATTN_HEAD_DIM: u32 = 8;
const ROPE_DIM: u32 = 4;
const BLOCK_COUNT: u32 = 2;
const FULL_ATTENTION_INTERVAL: u32 = 2; // layer 1 dense-attention, layer 0 ssm
const SSM_CONV_KERNEL: u32 = 4;
const SSM_STATE_SIZE: u32 = 2;
const SSM_GROUP_COUNT: u32 = 1;
const SSM_TIME_STEP_RANK: u32 = 2;
const SSM_INNER_SIZE: u32 = 4;

fn lcg_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        out.push((state >> 56) as u8);
    }
    out
}

fn matmul_tensor(name: &str, row_len: u32, rows: u32, seed: u64) -> TensorPayload<'static> {
    let data = lcg_bytes((row_len as usize * rows as usize) * 4, seed).leak();
    TensorPayload {
        name: name.to_string(),
        dims: [u64::from(row_len), u64::from(rows)].into_iter().collect(),
        ggml_type: GgmlType::F32,
        data,
    }
}

fn vector_tensor(name: &str, len: u32, seed: u64) -> TensorPayload<'static> {
    let data = lcg_bytes(len as usize * 4, seed).leak();
    TensorPayload {
        name: name.to_string(),
        dims: [u64::from(len)].into_iter().collect(),
        ggml_type: GgmlType::F32,
        data,
    }
}

fn layer_tensors(layer: u32, is_attention: bool, seed: u64) -> Vec<TensorPayload<'static>> {
    let mut tensors = vec![
        vector_tensor(&format!("blk.{layer}.attn_norm.weight"), EMBEDDING, seed + 1),
        vector_tensor(&format!("blk.{layer}.post_attention_norm.weight"), EMBEDDING, seed + 2),
    ];

    if is_attention {
        tensors.push(matmul_tensor(
            &format!("blk.{layer}.attn_q.weight"),
            EMBEDDING,
            QUERY_HEADS * ATTN_HEAD_DIM * 2,
            seed + 10,
        ));
        tensors.push(matmul_tensor(
            &format!("blk.{layer}.attn_k.weight"),
            EMBEDDING,
            KV_HEADS * ATTN_HEAD_DIM,
            seed + 11,
        ));
        tensors.push(matmul_tensor(
            &format!("blk.{layer}.attn_v.weight"),
            EMBEDDING,
            KV_HEADS * ATTN_HEAD_DIM,
            seed + 12,
        ));
        tensors.push(matmul_tensor(
            &format!("blk.{layer}.attn_output.weight"),
            QUERY_HEADS * ATTN_HEAD_DIM,
            EMBEDDING,
            seed + 13,
        ));
        tensors.push(vector_tensor(&format!("blk.{layer}.attn_q_norm.weight"), ATTN_HEAD_DIM, seed + 14));
        tensors.push(vector_tensor(&format!("blk.{layer}.attn_k_norm.weight"), ATTN_HEAD_DIM, seed + 15));
    } else {
        let ssm_key_dim = SSM_STATE_SIZE * SSM_GROUP_COUNT;
        let qkv_dim = 2 * ssm_key_dim + SSM_INNER_SIZE;
        let head_v_dim = SSM_INNER_SIZE / SSM_TIME_STEP_RANK;

        tensors.push(matmul_tensor(&format!("blk.{layer}.attn_qkv.weight"), EMBEDDING, qkv_dim, seed + 20));
        tensors.push(matmul_tensor(
            &format!("blk.{layer}.attn_gate.weight"),
            EMBEDDING,
            SSM_INNER_SIZE,
            seed + 21,
        ));
        tensors.push(matmul_tensor(
            &format!("blk.{layer}.ssm_alpha.weight"),
            EMBEDDING,
            SSM_TIME_STEP_RANK,
            seed + 22,
        ));
        tensors.push(matmul_tensor(
            &format!("blk.{layer}.ssm_beta.weight"),
            EMBEDDING,
            SSM_TIME_STEP_RANK,
            seed + 23,
        ));
        tensors.push(matmul_tensor(
            &format!("blk.{layer}.ssm_out.weight"),
            SSM_INNER_SIZE,
            EMBEDDING,
            seed + 24,
        ));
        tensors.push(vector_tensor(&format!("blk.{layer}.ssm_conv1d.weight"), qkv_dim * SSM_CONV_KERNEL, seed + 25));
        tensors.push(vector_tensor(&format!("blk.{layer}.ssm_dt"), SSM_TIME_STEP_RANK, seed + 26));
        tensors.push(vector_tensor(&format!("blk.{layer}.ssm_a"), SSM_TIME_STEP_RANK, seed + 27));
        tensors.push(vector_tensor(&format!("blk.{layer}.ssm_norm.weight"), head_v_dim, seed + 28));
    }

    tensors.push(matmul_tensor(&format!("blk.{layer}.ffn_gate.weight"), EMBEDDING, FEED_FORWARD, seed + 30));
    tensors.push(matmul_tensor(&format!("blk.{layer}.ffn_up.weight"), EMBEDDING, FEED_FORWARD, seed + 31));
    tensors.push(matmul_tensor(&format!("blk.{layer}.ffn_down.weight"), FEED_FORWARD, EMBEDDING, seed + 32));

    tensors
}

fn tokenizer_metadata() -> Vec<(String, MetadataValue)> {
    let tokens: Vec<String> = (0..VOCAB).map(|byte| (char::from(byte as u8)).to_string()).collect();
    let scores = vec![0.0f32; VOCAB as usize];
    vec![
        ("tokenizer.ggml.model".to_string(), MetadataValue::String("llama".to_string())),
        ("tokenizer.ggml.tokens".to_string(), MetadataValue::Array(MetadataArray::String(tokens))),
        ("tokenizer.ggml.scores".to_string(), MetadataValue::Array(MetadataArray::F32(scores))),
        ("tokenizer.ggml.bos_token_id".to_string(), MetadataValue::U32(1)),
        ("tokenizer.ggml.eos_token_id".to_string(), MetadataValue::U32(2)),
    ]
}

fn architecture_metadata(architecture_name: &str) -> Vec<(String, MetadataValue)> {
    vec![
        (
            "general.architecture".to_string(),
            MetadataValue::String(architecture_name.to_string()),
        ),
        (format!("{architecture_name}.embedding_length"), MetadataValue::U32(EMBEDDING)),
        (format!("{architecture_name}.feed_forward_length"), MetadataValue::U32(FEED_FORWARD)),
        (format!("{architecture_name}.attention.head_count"), MetadataValue::U32(QUERY_HEADS)),
        (format!("{architecture_name}.attention.head_count_kv"), MetadataValue::U32(KV_HEADS)),
        (format!("{architecture_name}.block_count"), MetadataValue::U32(BLOCK_COUNT)),
        (format!("{architecture_name}.rope.dimension_count"), MetadataValue::U32(ROPE_DIM)),
        (format!("{architecture_name}.attention.key_length"), MetadataValue::U32(ATTN_HEAD_DIM)),
        (
            format!("{architecture_name}.full_attention_interval"),
            MetadataValue::U32(FULL_ATTENTION_INTERVAL),
        ),
        (format!("{architecture_name}.ssm.conv_kernel"), MetadataValue::U32(SSM_CONV_KERNEL)),
        (format!("{architecture_name}.ssm.state_size"), MetadataValue::U32(SSM_STATE_SIZE)),
        (format!("{architecture_name}.ssm.group_count"), MetadataValue::U32(SSM_GROUP_COUNT)),
        (format!("{architecture_name}.ssm.time_step_rank"), MetadataValue::U32(SSM_TIME_STEP_RANK)),
        (format!("{architecture_name}.ssm.inner_size"), MetadataValue::U32(SSM_INNER_SIZE)),
    ]
}

fn checkpoint_bytes(architecture_name: &str) -> Vec<u8> {
    let mut metadata = architecture_metadata(architecture_name);
    metadata.extend(tokenizer_metadata());

    let mut tensors = vec![matmul_tensor("token_embd.weight", EMBEDDING, VOCAB, 1)];
    for layer in 0..BLOCK_COUNT {
        let is_attention = (layer + 1).is_multiple_of(FULL_ATTENTION_INTERVAL);
        tensors.extend(layer_tensors(layer, is_attention, u64::from(layer) * 1000 + 100));
    }
    tensors.push(vector_tensor("output_norm.weight", EMBEDDING, 999_999));

    let model = GgufModel {
        version: 3,
        metadata,
        tensors,
    };
    write_complete(&model).expect("writes a well-formed synthetic hybrid checkpoint")
}

const CORRECT_NAME: &str = "acme-qwen35moe-correct";
const MISTAGGED_NAME: &str = "acme-qwen35moe-mistagged";
const NO_STEP_STATE_NAME: &str = "acme-qwen35moe-no-step-state";

/// Delegates entirely to the builtin [`Qwen35Arch`] -- same tensor bind,
/// same `qwen35_forward_program` compile, same `layer_roots` it derives --
/// registered under a foreign name so this test proves the REGISTRY path
/// (`LoadedModel::load_with_registry`), not the builtin `"qwen35"`
/// fast-path branch in `load_inner`.
struct CorrectHybridArch;

impl Architecture for CorrectHybridArch {
    fn name(&self) -> &'static str {
        CORRECT_NAME
    }

    fn bind<'file>(&self, parsed: &ParsedGguf, file_bytes: &'file [u8]) -> Result<BoundProgram<'file>, InteropError> {
        Qwen35Arch.bind(parsed, file_bytes)
    }

    fn step_state(&self, parsed: &ParsedGguf) -> Result<Option<StepState>, InteropError> {
        Qwen35Arch.step_state(parsed)
    }
}

/// Same bind as [`CorrectHybridArch`], except the one `DenseAttention`
/// layer's [`Qwen35LayerRoots`] entry is re-tagged as `Attention` -- the
/// exact real-world mistake class this file's own doc reproduces: the
/// program still declares `kv_cache.{layer}.k_first`/`k_second`/`k_pass`/`v`
/// as `Op::Input` leaves (`Qwen35Arch::bind` never changes what it
/// compiles), only the SEPARATE, hand-kept `layer_roots` classification
/// disagrees with it.
struct MistaggedHybridArch;

impl Architecture for MistaggedHybridArch {
    fn name(&self) -> &'static str {
        MISTAGGED_NAME
    }

    fn bind<'file>(&self, parsed: &ParsedGguf, file_bytes: &'file [u8]) -> Result<BoundProgram<'file>, InteropError> {
        let mut bound = Qwen35Arch.bind(parsed, file_bytes)?;
        let dense_layer = bound
            .layer_roots
            .iter()
            .position(|roots| matches!(roots, Qwen35LayerRoots::DenseAttention(_)))
            .expect("this fixture's own full_attention_interval guarantees one DenseAttention layer");
        if let Qwen35LayerRoots::DenseAttention((first, second, _pass, value)) = bound.layer_roots[dense_layer] {
            bound.layer_roots[dense_layer] = Qwen35LayerRoots::Attention((first, second, value));
        }
        Ok(bound)
    }

    fn step_state(&self, parsed: &ParsedGguf) -> Result<Option<StepState>, InteropError> {
        Qwen35Arch.step_state(parsed)
    }
}

/// Same bind AND same `ssm_shape`/`ssm_state_bytes` as [`CorrectHybridArch`]
/// -- only `attn_head_dim` comes back `0` instead of the real
/// [`Qwen35Arch`] value, the exact shape measured on `qwen3.6:35b-a3b`
/// through a foreign `Architecture` registered via `load_with_registry`:
/// that bind derives `head_dim` from `attention.key_length` alone
/// (`bind.rs`'s header-only derivation) and never fills a per-layer
/// `attn_head_dim` the way [`Qwen35Arch::step_state`] does, so it comes
/// back `0`/unset on the real checkpoint too. Every other [`StepState`]
/// field stays real so this isolates the ONE broken field -- an SSM
/// layer's own cache (sized from `ssm_shape`, a different [`StepState`]
/// field entirely) is not this file's bug and must stay correctly sized
/// for this test to exercise only the `DenseAttention` path.
///
/// The bound program still declares `kv_cache.{layer}.k_first`/`k_second`/
/// `k_pass`/`v` correctly (`bind` never changes what it compiles from a
/// broken `step_state`), but `LoadedModel::run_decode_loop_observed_seeded`
/// used to read `step_state.map(|state| state.attn_head_dim)` as `0` and
/// size the `DenseAttention` layer's `v`/`k_pass` pad-scratch to ZERO
/// elements, panicking on a later decode step's `copy_from_slice` the
/// moment that layer's real (nonzero) cache tried to copy in
/// (`Qwen35DenseAttentionPadScratch::fill`, `generate.rs`).
struct NoStepStateHybridArch;

impl Architecture for NoStepStateHybridArch {
    fn name(&self) -> &'static str {
        NO_STEP_STATE_NAME
    }

    fn bind<'file>(&self, parsed: &ParsedGguf, file_bytes: &'file [u8]) -> Result<BoundProgram<'file>, InteropError> {
        Qwen35Arch.bind(parsed, file_bytes)
    }

    fn step_state(&self, parsed: &ParsedGguf) -> Result<Option<StepState>, InteropError> {
        let state = Qwen35Arch.step_state(parsed)?;
        Ok(state.map(|real| StepState {
            attn_head_dim: 0,
            ..real
        }))
    }
}

fn registry_with(architecture: &'static dyn Architecture) -> ArchitectureRegistry {
    let mut registry = ArchitectureRegistry::with_builtin();
    registry.register(architecture);
    registry
}

static CORRECT: CorrectHybridArch = CorrectHybridArch;
static MISTAGGED: MistaggedHybridArch = MistaggedHybridArch;
static NO_STEP_STATE: NoStepStateHybridArch = NoStepStateHybridArch;

/// The mechanism this file exists to prove: a correctly-bound hybrid
/// program (one SSM layer, one full-attention layer, registered under a
/// foreign name) greedy-decodes through [`LoadedModel::load_with_registry`]
/// with no [`InteropError::MissingStepInput`] -- the decode loop derives
/// each layer's cache-leaf names from the program's own declared
/// `Op::Input` leaves, so a hybrid checkpoint's mixed cache shapes are all
/// fed correctly in one call.
#[proxima::test]
async fn a_foreign_hybrid_architecture_decodes_with_no_missing_step_input() {
    let file_bytes = checkpoint_bytes(CORRECT_NAME);
    let parsed = parse_complete(&file_bytes).expect("parses the synthetic hybrid checkpoint");
    let registry = registry_with(&CORRECT);

    let model = LoadedModel::load_with_registry(&parsed, &file_bytes, &registry)
        .expect("loads a hybrid program through a foreign registry entry");

    let (generated_ids, _text, _stopped) = Pipe::call(&model, ("ab".to_string(), 3))
        .await
        .expect("greedy decode seeds every layer's cache leaves, ssm and dense-attention alike");
    assert_eq!(generated_ids.len(), 3, "max_tokens=3 produces exactly three token ids");
}

/// The failure mode this file's own doc names: a `layer_roots` entry
/// mistagged relative to what the program actually declares is caught
/// once, loudly, at decode-loop setup (naming the layer and both the
/// declared and bound shapes) instead of surfacing as a `MissingStepInput`
/// on a leaf the decode loop never even tried to feed under the wrong tag.
#[proxima::test]
async fn a_mistagged_layer_roots_entry_is_rejected_before_any_step_runs() {
    let file_bytes = checkpoint_bytes(MISTAGGED_NAME);
    let parsed = parse_complete(&file_bytes).expect("parses the synthetic hybrid checkpoint");
    let registry = registry_with(&MISTAGGED);

    let model = LoadedModel::load_with_registry(&parsed, &file_bytes, &registry)
        .expect("load itself never inspects layer_roots' own internal consistency");

    match Pipe::call(&model, ("a".to_string(), 1)).await {
        Err(InteropError::LayerCacheKindMismatch { declared, bound, .. }) => {
            assert_eq!(declared, "kv_cache.{layer}.{k_first,k_second,k_pass,v}");
            assert_eq!(bound, "kv_cache.{layer}.{k_even,k_odd,v}");
        }
        Ok(_) => panic!("expected LayerCacheKindMismatch: the mistagged layer's roots disagree with its program"),
        Err(other) => panic!("expected LayerCacheKindMismatch, got {other}"),
    }
}

/// The real defect measured on `qwen3.6:35b-a3b` through
/// `LoadedModel::load_with_registry`: a foreign `Architecture` that never
/// overrides `step_state` (the trait's own `Ok(None)` default) used to
/// leave the `DenseAttention` layer's `v`/`k_pass` pad-scratch sized to
/// zero elements, panicking on the first decode step with `range end index
/// .. out of range for slice of length 0` the moment that layer's real
/// cache tried to copy in. The pad-scratch shape now comes from the bound
/// program's own declared `kv_cache.{layer}.*` `Op::Input` extents
/// (`generate.rs`'s `cache_leaf_row_elements`/`layer_pad_row_widths`),
/// never from `Architecture::step_state`, so a hybrid checkpoint decodes
/// correctly even when a foreign bind leaves that hook at its default.
#[proxima::test]
async fn a_foreign_architecture_with_no_step_state_override_still_decodes() {
    let file_bytes = checkpoint_bytes(NO_STEP_STATE_NAME);
    let parsed = parse_complete(&file_bytes).expect("parses the synthetic hybrid checkpoint");
    let registry = registry_with(&NO_STEP_STATE);

    let model = LoadedModel::load_with_registry(&parsed, &file_bytes, &registry)
        .expect("loads a hybrid program through a foreign registry entry with no step_state override");

    let (generated_ids, _text, _stopped) = Pipe::call(&model, ("ab".to_string(), 3))
        .await
        .expect("dense-attention pad-scratch is sized from the program's declared cache leaves, not step_state");
    assert_eq!(generated_ids.len(), 3, "max_tokens=3 produces exactly three token ids");
}
