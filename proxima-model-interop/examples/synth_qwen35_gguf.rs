//! Synthesizes a Qwen3.8-27B-*shaped* GGUF fixture and runs it through
//! [`proxima_model_interop::generate::LoadedModel`]'s `qwen35` hybrid path
//! (`crate::qwen35`) -- no real Qwen3.8-27B checkpoint exists on this box
//! (`/private/tmp/.../scratchpad/qwen38-path.md`'s own leading finding: the
//! only on-disk qwen3.5/3.6 files are `qwen35moe`, MoE + vision, typed-
//! rejected by `architecture_from_metadata`'s heterogeneous-`head_count_kv`
//! guard), so memory-by-class and the lowering census cannot come from a
//! real weight file. Random weights of the RIGHT shape and quant mix are
//! sufficient for both: neither depends on weight VALUES, only on tensor
//! byte layout and the op graph [`proxima_tensor::spec::qwen35_forward_program`]
//! builds from [`Qwen35Architecture`]'s hparams.
//!
//! Composes [`proxima_gguf::writer::write_complete`] (the existing sans-IO
//! GGUF writer -- no new writer needed, `grep GgufWriter` found this one
//! already shipping) with [`Qwen35LayerKind::from_interval`]'s own layer-kind
//! arithmetic to emit exactly the tensor set [`crate::qwen35::bind_qwen35_weights`]
//! expects, at exactly the shapes
//! [`proxima_tensor::spec::qwen35_forward_program`] compiles its op graph
//! against (`proxima-tensor/src/spec.rs:7692` and its per-layer `input_leaf`
//! calls) -- every shape below is read from that function's body, not
//! guessed.
//!
//! LAYER COUNT: 4, not the documented 64
//! (`proxima-model-interop/src/qwen35.rs`'s own test fixture). Owner
//! directive, binding: the real Qwen3.8 checkpoint froze the box under
//! memory pressure once already; this fixture stays small enough to never
//! approach that regime. `full_attention_interval = 4` applied to 4 layers
//! yields 3 Ssm + 1 Attention (`(i+1) % 4 == 0` only at `i == 3`), not an
//! even split -- kept the real interval arithmetic rather than distorting it
//! to force 2+2, since correctness of the hybrid *pattern* (both kinds
//! present, in the checkpoint's own ratio) matters more than a round layer
//! count. This is a 4/64 = 1/16 SCALE of the documented layer count; every
//! per-layer byte/time number below is reported per-layer-KIND so a
//! separate real-checkpoint ladder (owner: "another slice runs it") can
//! extrapolate to 27B without re-deriving these shapes.
//!
//! Every hparam below is either DOCUMENTED (read straight off
//! `qwen35.rs`'s own doc comments, confirmed against real on-disk tensor
//! shapes per that module's comments) or ASSUMED (no real 27B file exists;
//! value chosen as a magnitude stand-in, flagged inline). `vocab` and
//! `feed_forward` are deliberately shrunk far below any real checkpoint's
//! own values -- neither affects which Metal lowering branch a packed
//! matvec/attention/scan op reaches (that classification keys on codec +
//! row length, not on absolute vocab/ffn width), and a huge vocab would
//! only bloat the embedding/head tensor and the fabricated tokenizer
//! metadata for no signal this fixture exists to produce.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_gguf::types::GgmlType;
use proxima_gguf::value::{MetadataArray, MetadataValue};
use proxima_gguf::writer::{GgufModel, TensorPayload, write_complete};

const OUTPUT_PATH: &str = "/tmp/proxima-synth-qwen35.gguf";

// DOCUMENTED (`qwen35.rs:136-137`'s own worked example: "5120 / 24 = 213.33"
// proves embedding=5120, query_heads=24 on the real 27B checkpoint).
pub(crate) const EMBEDDING: u32 = 5120;
pub(crate) const QUERY_HEADS: u32 = 24;
// DOCUMENTED (`qwen35.rs:33-48`: "the real per-head width is 512, not 64").
pub(crate) const ATTN_HEAD_DIM: u32 = 512;
// DOCUMENTED (`qwen35.rs:136`: "rope.dimension_count (`64`), this
// checkpoint's PARTIAL rotary width").
pub(crate) const ROPE_DIM: u32 = 64;
// DOCUMENTED (`qwen35.rs`'s own test: `full_attention_interval = 4`).
pub(crate) const FULL_ATTENTION_INTERVAL: u32 = 4;
// REDUCED per owner directive (see module doc) from the documented 64.
pub(crate) const BLOCK_COUNT: u32 = 4;

// ASSUMED -- no real 27B file's `attention.head_count_kv` is observable;
// stand-in taken from the on-disk `qwen3.6:35b-a3b` file's active per-layer
// value (`qwen38-path.md` (1): the heterogeneous array's non-zero entries
// are `2`).
pub(crate) const KV_HEADS: u32 = 2;
// ASSUMED -- no real file; a magnitude stand-in only (roughly 4x embedding,
// a common dense-FFN ratio), shrunk from a typical ~20K-27K real width
// purely to keep this fixture's byte count small; ffn width does not change
// which Metal dispatch branch any op reaches.
pub(crate) const FEED_FORWARD: u32 = 20_480;
// ASSUMED -- shrunk far below any real vocab (Qwen's real tokenizer is
// ~150K entries); this fixture's tokenizer is fabricated single-byte tokens
// (see `build_vocab_metadata`), so a huge vocab buys nothing but bytes.
pub(crate) const VOCAB: u32 = 128;

// ASSUMED stand-ins, all taken verbatim from the real on-disk
// `qwen3.6:35b-a3b` file's own `ssm.*` metadata
// (`qwen38-path.md` (1)) -- the true 27B file's own ssm hparams are not
// observable (no such file exists), but a real hybrid-checkpoint sibling's
// values are a far better magnitude stand-in than an invented number.
pub(crate) const SSM_CONV_KERNEL: u32 = 4;
pub(crate) const SSM_STATE_SIZE: u32 = 128;
pub(crate) const SSM_GROUP_COUNT: u32 = 16;
pub(crate) const SSM_TIME_STEP_RANK: u32 = 32;
pub(crate) const SSM_INNER_SIZE: u32 = 4096;

/// Deterministic, allocation-cheap fill: an LCG stream reduced to a byte per
/// step, the same "look plausible, cost nothing to generate" shape the
/// repo's own bandwidth-arm fixtures use (`feedback_bandwidth_arm_must_
/// exceed_dispatch_floor` in project memory) -- never real weight values,
/// never claimed to be.
fn lcg_bytes(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        out.push((state >> 56) as u8);
    }
    out
}

fn block_elements(ggml_type: GgmlType) -> u64 {
    ggml_type.block_layout().block_elements
}

fn tensor_bytes(elements: u64, ggml_type: GgmlType) -> u64 {
    let layout = ggml_type.block_layout();
    (elements / layout.block_elements) * layout.block_bytes
}

/// `(row_len, rows)` in GGUF's own `[in_dim, out_dim]` `ne` convention
/// (`qwen35.rs:40-42`'s own doc) -- `row_len` is `dims[0]`, the axis a
/// quantized type's block size must divide.
pub(crate) fn matmul_tensor(
    name: &str,
    row_len: u32,
    rows: u32,
    ggml_type: GgmlType,
    seed: u64,
) -> TensorPayload<'static> {
    assert_eq!(
        u64::from(row_len) % block_elements(ggml_type),
        0,
        "{name}: row_len {row_len} not a multiple of {ggml_type:?}'s block size"
    );
    let elements = u64::from(row_len) * u64::from(rows);
    let data = lcg_bytes(tensor_bytes(elements, ggml_type) as usize, seed).leak();
    TensorPayload {
        name: name.to_string(),
        dims: [u64::from(row_len), u64::from(rows)].into_iter().collect(),
        ggml_type,
        data,
    }
}

pub(crate) fn vector_tensor(name: &str, len: u32, seed: u64) -> TensorPayload<'static> {
    let data = lcg_bytes(len as usize * 4, seed).leak();
    TensorPayload {
        name: name.to_string(),
        dims: [u64::from(len)].into_iter().collect(),
        ggml_type: GgmlType::F32,
        data,
    }
}

/// Every tensor [`proxima_model_interop::qwen35::bind_qwen35_weights`] binds
/// for one layer, at exactly the shapes
/// `proxima_tensor::spec::qwen35_forward_program`'s per-layer `input_leaf`
/// calls declare (`proxima-tensor/src/spec.rs:7808-8087`) -- read from that
/// function's body line by line, not derived independently, so a shape bug
/// here would be a transcription error, not a design guess.
///
/// Quant mix: Q4_K for every big matmul-shaped weight, Q6_K for the
/// tied-embedding table (`token_embd.weight`, also this checkpoint's head),
/// F32 for every norm and every small `ssm_*` vector this checkpoint binds
/// via `bind_dense` -- the production mix the land-brief specifies.
pub(crate) fn layer_tensors(
    layer: u32,
    is_attention: bool,
    seed: u64,
) -> Vec<TensorPayload<'static>> {
    let mut tensors = vec![
        vector_tensor(
            &format!("blk.{layer}.attn_norm.weight"),
            EMBEDDING,
            seed + 1,
        ),
        vector_tensor(
            &format!("blk.{layer}.post_attention_norm.weight"),
            EMBEDDING,
            seed + 2,
        ),
    ];

    if is_attention {
        tensors.push(matmul_tensor(
            &format!("blk.{layer}.attn_q.weight"),
            EMBEDDING,
            QUERY_HEADS * ATTN_HEAD_DIM * 2,
            GgmlType::Q4_K,
            seed + 10,
        ));
        tensors.push(matmul_tensor(
            &format!("blk.{layer}.attn_k.weight"),
            EMBEDDING,
            KV_HEADS * ATTN_HEAD_DIM,
            GgmlType::Q4_K,
            seed + 11,
        ));
        tensors.push(matmul_tensor(
            &format!("blk.{layer}.attn_v.weight"),
            EMBEDDING,
            KV_HEADS * ATTN_HEAD_DIM,
            GgmlType::Q4_K,
            seed + 12,
        ));
        tensors.push(matmul_tensor(
            &format!("blk.{layer}.attn_output.weight"),
            QUERY_HEADS * ATTN_HEAD_DIM,
            EMBEDDING,
            GgmlType::Q4_K,
            seed + 13,
        ));
        tensors.push(vector_tensor(
            &format!("blk.{layer}.attn_q_norm.weight"),
            ATTN_HEAD_DIM,
            seed + 14,
        ));
        tensors.push(vector_tensor(
            &format!("blk.{layer}.attn_k_norm.weight"),
            ATTN_HEAD_DIM,
            seed + 15,
        ));
    } else {
        let ssm_key_dim = SSM_STATE_SIZE * SSM_GROUP_COUNT;
        let qkv_dim = 2 * ssm_key_dim + SSM_INNER_SIZE;
        let head_v_dim = SSM_INNER_SIZE / SSM_TIME_STEP_RANK;

        tensors.push(matmul_tensor(
            &format!("blk.{layer}.attn_qkv.weight"),
            EMBEDDING,
            qkv_dim,
            GgmlType::Q4_K,
            seed + 20,
        ));
        tensors.push(matmul_tensor(
            &format!("blk.{layer}.attn_gate.weight"),
            EMBEDDING,
            SSM_INNER_SIZE,
            GgmlType::Q4_K,
            seed + 21,
        ));
        tensors.push(matmul_tensor(
            &format!("blk.{layer}.ssm_alpha.weight"),
            EMBEDDING,
            SSM_TIME_STEP_RANK,
            GgmlType::Q4_K,
            seed + 22,
        ));
        tensors.push(matmul_tensor(
            &format!("blk.{layer}.ssm_beta.weight"),
            EMBEDDING,
            SSM_TIME_STEP_RANK,
            GgmlType::Q4_K,
            seed + 23,
        ));
        tensors.push(matmul_tensor(
            &format!("blk.{layer}.ssm_out.weight"),
            SSM_INNER_SIZE,
            EMBEDDING,
            GgmlType::Q4_K,
            seed + 24,
        ));
        tensors.push(vector_tensor(
            &format!("blk.{layer}.ssm_conv1d.weight"),
            qkv_dim * SSM_CONV_KERNEL,
            seed + 25,
        ));
        tensors.push(vector_tensor(
            &format!("blk.{layer}.ssm_dt"),
            SSM_TIME_STEP_RANK,
            seed + 26,
        ));
        tensors.push(vector_tensor(
            &format!("blk.{layer}.ssm_a"),
            SSM_TIME_STEP_RANK,
            seed + 27,
        ));
        tensors.push(vector_tensor(
            &format!("blk.{layer}.ssm_norm.weight"),
            head_v_dim,
            seed + 28,
        ));
    }

    tensors.push(matmul_tensor(
        &format!("blk.{layer}.ffn_gate.weight"),
        EMBEDDING,
        FEED_FORWARD,
        GgmlType::Q4_K,
        seed + 30,
    ));
    tensors.push(matmul_tensor(
        &format!("blk.{layer}.ffn_up.weight"),
        EMBEDDING,
        FEED_FORWARD,
        GgmlType::Q4_K,
        seed + 31,
    ));
    tensors.push(matmul_tensor(
        &format!("blk.{layer}.ffn_down.weight"),
        FEED_FORWARD,
        EMBEDDING,
        GgmlType::Q4_K,
        seed + 32,
    ));

    tensors
}

/// A fabricated single-byte-per-token vocabulary: `tokens[i]` is the
/// one-character string `char::from(i as u8)` for `i in 0..VOCAB` (ASCII,
/// so always valid UTF-8), which makes tokenization of any ASCII prompt an
/// unambiguous 1:1 mapping regardless of the (all-zero) unigram scores --
/// this fixture cares about the forward pass running, not about a
/// realistic tokenizer.
pub(crate) fn tokenizer_metadata() -> Vec<(String, MetadataValue)> {
    let tokens: Vec<String> = (0..VOCAB)
        .map(|byte| (char::from(byte as u8)).to_string())
        .collect();
    let scores = vec![0.0f32; VOCAB as usize];
    vec![
        (
            "tokenizer.ggml.model".to_string(),
            MetadataValue::String("llama".to_string()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            MetadataValue::Array(MetadataArray::String(tokens)),
        ),
        (
            "tokenizer.ggml.scores".to_string(),
            MetadataValue::Array(MetadataArray::F32(scores)),
        ),
        (
            "tokenizer.ggml.bos_token_id".to_string(),
            MetadataValue::U32(1),
        ),
        (
            "tokenizer.ggml.eos_token_id".to_string(),
            MetadataValue::U32(2),
        ),
    ]
}

pub(crate) fn architecture_metadata() -> Vec<(String, MetadataValue)> {
    vec![
        (
            "general.architecture".to_string(),
            MetadataValue::String("qwen35".to_string()),
        ),
        (
            "qwen35.embedding_length".to_string(),
            MetadataValue::U32(EMBEDDING),
        ),
        (
            "qwen35.feed_forward_length".to_string(),
            MetadataValue::U32(FEED_FORWARD),
        ),
        (
            "qwen35.attention.head_count".to_string(),
            MetadataValue::U32(QUERY_HEADS),
        ),
        (
            "qwen35.attention.head_count_kv".to_string(),
            MetadataValue::U32(KV_HEADS),
        ),
        (
            "qwen35.block_count".to_string(),
            MetadataValue::U32(BLOCK_COUNT),
        ),
        (
            "qwen35.rope.dimension_count".to_string(),
            MetadataValue::U32(ROPE_DIM),
        ),
        (
            "qwen35.attention.key_length".to_string(),
            MetadataValue::U32(ATTN_HEAD_DIM),
        ),
        (
            "qwen35.full_attention_interval".to_string(),
            MetadataValue::U32(FULL_ATTENTION_INTERVAL),
        ),
        (
            "qwen35.ssm.conv_kernel".to_string(),
            MetadataValue::U32(SSM_CONV_KERNEL),
        ),
        (
            "qwen35.ssm.state_size".to_string(),
            MetadataValue::U32(SSM_STATE_SIZE),
        ),
        (
            "qwen35.ssm.group_count".to_string(),
            MetadataValue::U32(SSM_GROUP_COUNT),
        ),
        (
            "qwen35.ssm.time_step_rank".to_string(),
            MetadataValue::U32(SSM_TIME_STEP_RANK),
        ),
        (
            "qwen35.ssm.inner_size".to_string(),
            MetadataValue::U32(SSM_INNER_SIZE),
        ),
    ]
}

fn main() {
    let mut metadata = architecture_metadata();
    metadata.extend(tokenizer_metadata());

    let mut tensors = vec![matmul_tensor(
        "token_embd.weight",
        EMBEDDING,
        VOCAB,
        GgmlType::Q6_K,
        1,
    )];
    for layer in 0..BLOCK_COUNT {
        let is_attention = (layer + 1).is_multiple_of(FULL_ATTENTION_INTERVAL);
        tensors.extend(layer_tensors(
            layer,
            is_attention,
            u64::from(layer) * 1000 + 100,
        ));
    }
    tensors.push(vector_tensor("output_norm.weight", EMBEDDING, 999_999));

    let attention_layers = (0..BLOCK_COUNT)
        .filter(|layer| (layer + 1).is_multiple_of(FULL_ATTENTION_INTERVAL))
        .count();
    println!(
        "synth_qwen35_gguf: block_count={BLOCK_COUNT} attention_layers={attention_layers} ssm_layers={} \
         (full_attention_interval={FULL_ATTENTION_INTERVAL}, documented real block_count=64, scale=1/16)",
        BLOCK_COUNT as usize - attention_layers
    );

    let model = GgufModel {
        version: 3,
        metadata,
        tensors,
    };

    let written = write_complete(&model).expect("write synthetic qwen35 gguf");
    println!(
        "synth_qwen35_gguf: {} tensors, {} bytes ({:.2} MiB) -> {OUTPUT_PATH}",
        model.tensors.len(),
        written.len(),
        written.len() as f64 / (1024.0 * 1024.0)
    );

    let reparsed = proxima_gguf::parse_complete(&written).expect("written bytes parse back");
    assert_eq!(
        reparsed.tensors.len(),
        model.tensors.len(),
        "tensor count round-trips"
    );

    std::fs::write(OUTPUT_PATH, &written).expect("write synthetic gguf to scratchpad");
}
