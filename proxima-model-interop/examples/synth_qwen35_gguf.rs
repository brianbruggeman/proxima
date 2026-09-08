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

const OUTPUT_PATH: &str = "/private/tmp/claude-501/-Users-brianbruggeman-repos-slot-0/6e203711-bd50-48cc-9ade-409668bdafdd/scratchpad/synth-qwen38-27b.gguf";

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
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
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
pub(crate) fn matmul_tensor(name: &str, row_len: u32, rows: u32, ggml_type: GgmlType, seed: u64) -> TensorPayload<'static> {
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
pub(crate) fn layer_tensors(layer: u32, is_attention: bool, seed: u64) -> Vec<TensorPayload<'static>> {
    let mut tensors = vec![
        vector_tensor(&format!("blk.{layer}.attn_norm.weight"), EMBEDDING, seed + 1),
        vector_tensor(&format!("blk.{layer}.post_attention_norm.weight"), EMBEDDING, seed + 2),
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
        tensors.push(vector_tensor(&format!("blk.{layer}.attn_q_norm.weight"), ATTN_HEAD_DIM, seed + 14));
        tensors.push(vector_tensor(&format!("blk.{layer}.attn_k_norm.weight"), ATTN_HEAD_DIM, seed + 15));
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
        tensors.push(vector_tensor(&format!("blk.{layer}.ssm_conv1d.weight"), qkv_dim * SSM_CONV_KERNEL, seed + 25));
        tensors.push(vector_tensor(&format!("blk.{layer}.ssm_dt.bias"), SSM_TIME_STEP_RANK, seed + 26));
        tensors.push(vector_tensor(&format!("blk.{layer}.ssm_a"), SSM_TIME_STEP_RANK, seed + 27));
        tensors.push(vector_tensor(&format!("blk.{layer}.ssm_norm.weight"), head_v_dim, seed + 28));
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

pub(crate) fn architecture_metadata() -> Vec<(String, MetadataValue)> {
    vec![
        ("general.architecture".to_string(), MetadataValue::String("qwen35".to_string())),
        ("qwen35.embedding_length".to_string(), MetadataValue::U32(EMBEDDING)),
        ("qwen35.feed_forward_length".to_string(), MetadataValue::U32(FEED_FORWARD)),
        ("qwen35.attention.head_count".to_string(), MetadataValue::U32(QUERY_HEADS)),
        ("qwen35.attention.head_count_kv".to_string(), MetadataValue::U32(KV_HEADS)),
        ("qwen35.block_count".to_string(), MetadataValue::U32(BLOCK_COUNT)),
        ("qwen35.rope.dimension_count".to_string(), MetadataValue::U32(ROPE_DIM)),
        ("qwen35.attention.key_length".to_string(), MetadataValue::U32(ATTN_HEAD_DIM)),
        ("qwen35.full_attention_interval".to_string(), MetadataValue::U32(FULL_ATTENTION_INTERVAL)),
        ("qwen35.ssm.conv_kernel".to_string(), MetadataValue::U32(SSM_CONV_KERNEL)),
        ("qwen35.ssm.state_size".to_string(), MetadataValue::U32(SSM_STATE_SIZE)),
        ("qwen35.ssm.group_count".to_string(), MetadataValue::U32(SSM_GROUP_COUNT)),
        ("qwen35.ssm.time_step_rank".to_string(), MetadataValue::U32(SSM_TIME_STEP_RANK)),
        ("qwen35.ssm.inner_size".to_string(), MetadataValue::U32(SSM_INNER_SIZE)),
    ]
}

fn build_qwen35_model() -> GgufModel<'static> {
    let mut metadata = architecture_metadata();
    metadata.extend(tokenizer_metadata());

    let mut tensors = vec![matmul_tensor("token_embd.weight", EMBEDDING, VOCAB, GgmlType::Q6_K, 1)];
    for layer in 0..BLOCK_COUNT {
        let is_attention = (layer + 1).is_multiple_of(FULL_ATTENTION_INTERVAL);
        tensors.extend(layer_tensors(layer, is_attention, u64::from(layer) * 1000 + 100));
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

    GgufModel {
        version: 3,
        metadata,
        tensors,
    }
}

/// Which synthetic architecture [`main`] emits — selected by `--arch`.
enum Arch {
    Qwen35,
    Qwen4Exp,
}

impl Arch {
    fn from_flag(flag: Option<&str>) -> Self {
        match flag {
            None | Some("qwen35") => Self::Qwen35,
            Some("qwen4exp") => Self::Qwen4Exp,
            Some(other) => panic!("unknown --arch {other:?}, expected qwen35 or qwen4exp"),
        }
    }
}

/// `--arch <qwen35|qwen4exp>` (default `qwen35`) and an optional trailing
/// output path, mirroring the two constructors below -- no argument parsing
/// crate needed for two flags.
fn parse_args() -> (Arch, String) {
    let mut arch = None;
    let mut output = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--arch" {
            arch = args.next();
        } else {
            output = Some(arg);
        }
    }
    (Arch::from_flag(arch.as_deref()), output.unwrap_or_else(|| OUTPUT_PATH.to_string()))
}

fn main() {
    let (arch, output_path) = parse_args();

    let model = match arch {
        Arch::Qwen35 => build_qwen35_model(),
        Arch::Qwen4Exp => qwen4exp::build_qwen4exp_model(),
    };

    let written = write_complete(&model).expect("write synthetic gguf");
    println!(
        "synth_qwen35_gguf: {} tensors, {} bytes ({:.2} MiB) -> {output_path}",
        model.tensors.len(),
        written.len(),
        written.len() as f64 / (1024.0 * 1024.0)
    );

    let reparsed = proxima_gguf::parse_complete(&written).expect("written bytes parse back");
    assert_eq!(reparsed.tensors.len(), model.tensors.len(), "tensor count round-trips");

    std::fs::write(&output_path, &written).expect("write synthetic gguf to scratchpad");
}

/// The tiny `qwen4exp` (Qwen3.8-Flash-Next, llama.cpp PR #27742) fixture.
/// Every hparam key and tensor name below is read from the PR's own diff
/// (`scratchpad/pr27742.diff`, cited per item), never invented -- only the
/// magnitudes (layer/expert/head counts) are shrunk to keep this fixture
/// writable in well under a second, per the S1 slice brief.
pub(crate) mod qwen4exp {
    use proxima_gguf::types::GgmlType;
    use proxima_gguf::value::{MetadataArray, MetadataValue};
    use proxima_gguf::writer::{GgufModel, TensorPayload};

    use super::{matmul_tensor, vector_tensor};

    pub(crate) const EMBEDDING: u32 = 64;
    pub(crate) const QUERY_HEADS: u32 = 4;
    pub(crate) const KV_HEADS: u32 = 2;
    pub(crate) const ATTN_HEAD_DIM: u32 = 16;
    pub(crate) const ROPE_DIM: u32 = 16;
    // 2 GDN (gated-delta-net) layers then 1 attention layer -- `(i+1) %
    // FULL_ATTENTION_INTERVAL == 0` only at `i == 2`, matching the PR's own
    // `is_recr_impl` schedule derivation (`pr27742.diff:2579-2586`).
    pub(crate) const BLOCK_COUNT: u32 = 3;
    pub(crate) const FULL_ATTENTION_INTERVAL: u32 = 3;
    pub(crate) const VOCAB: u32 = 256;
    pub(crate) const FEED_FORWARD: u32 = 128;

    // Gated delta net (SSM-shaped) sizing -- same field set qwen35's own
    // fixture uses, since qwen4exp shares Qwen3.5's GDN (plan §2: "shares
    // the Qwen3.5 gated delta net").
    pub(crate) const SSM_CONV_KERNEL: u32 = 4;
    pub(crate) const SSM_STATE_SIZE: u32 = 16;
    pub(crate) const SSM_GROUP_COUNT: u32 = 2;
    pub(crate) const SSM_TIME_STEP_RANK: u32 = 4;
    pub(crate) const SSM_INNER_SIZE: u32 = 32;

    // Hyper-connections (`pr27742.diff:2512-2515`, `hparams.dsv4_hc_mult`
    // and `hparams.hc_low_rank`).
    pub(crate) const HC_COUNT: u32 = 2;
    pub(crate) const HC_LOW_RANK: u32 = 8;

    // Sparse-attention indexer (`pr27742.diff:2517-2523`), DeepSeek-V4
    // lineage keys reused unchanged by qwen4exp.
    pub(crate) const INDEXER_HEAD_COUNT: u32 = 2;
    pub(crate) const INDEXER_KEY_LENGTH: u32 = 8;
    pub(crate) const INDEXER_TOP_K: u32 = 4;
    pub(crate) const INDEXER_COMPRESS_RATIO: u32 = 2;

    // MoE (top-2 of 8, plus a shared expert) -- generic MoE keys
    // (`{architecture}.expert_count`/`expert_used_count`, confirmed against
    // this crate's own `bind.rs:285-298` reader) that qwen4exp does not
    // override.
    pub(crate) const EXPERT_COUNT: u32 = 8;
    pub(crate) const EXPERT_USED_COUNT: u32 = 2;
    pub(crate) const EXPERT_FEED_FORWARD: u32 = 32;
    pub(crate) const EXPERT_SHARED_FEED_FORWARD: u32 = 32;

    // PLE n-gram hash table (`pr27742.diff:2525-2577`), one layer only
    // (`layer 0`, the first GDN layer) -- `n_ple == 1` is asserted by the
    // reference (`pr27742.diff:2534`).
    pub(crate) const PLE_LAYER: u32 = 0;
    pub(crate) const PLE_NGRAM_SIZE: u32 = 3;
    pub(crate) const PLE_HEADS_PER_NGRAM: u32 = 2;
    pub(crate) const PLE_CONV_KERNEL: u32 = 4;
    pub(crate) const PLE_HEAD_DIM: u32 = 32;
    // 64-row table per the S1 slice brief; `ple_n_heads = (ngram_size - 1) *
    // heads_per_ngram = 4` (`pr27742.diff:2551`), so 16 rows/head.
    pub(crate) const PLE_TABLE_ROWS: u32 = 64;
    pub(crate) const PLE_ROWS_PER_HEAD: u32 = PLE_TABLE_ROWS / 4;
    pub(crate) const PLE_EOS_TOKEN_ID: u32 = 2;

    /// Hparam keys, `%s` substituted with `"qwen4exp"` -- every key cited
    /// against the diff line that defines its `LLM_KV_*` mapping
    /// (`pr27742.diff:616-629`) or, for keys this arch reuses unchanged
    /// from an existing arch (rope sections, rms eps, expert counts),
    /// against this crate's own reader that already spells the generic
    /// `{architecture}.*` form (`bind.rs:285-298`, `lfm2.rs:111-138`).
    pub(crate) fn architecture_metadata() -> Vec<(String, MetadataValue)> {
        vec![
            ("general.architecture".to_string(), MetadataValue::String("qwen4exp".to_string())),
            ("qwen4exp.embedding_length".to_string(), MetadataValue::U32(EMBEDDING)),
            ("qwen4exp.feed_forward_length".to_string(), MetadataValue::U32(FEED_FORWARD)),
            ("qwen4exp.attention.head_count".to_string(), MetadataValue::U32(QUERY_HEADS)),
            ("qwen4exp.attention.head_count_kv".to_string(), MetadataValue::U32(KV_HEADS)),
            ("qwen4exp.block_count".to_string(), MetadataValue::U32(BLOCK_COUNT)),
            ("qwen4exp.rope.dimension_count".to_string(), MetadataValue::U32(ROPE_DIM)),
            ("qwen4exp.attention.key_length".to_string(), MetadataValue::U32(ATTN_HEAD_DIM)),
            (
                "qwen4exp.attention.full_attention_interval".to_string(),
                MetadataValue::U32(FULL_ATTENTION_INTERVAL),
            ),
            (
                "qwen4exp.attention.layer_norm_rms_epsilon".to_string(),
                MetadataValue::F32(1e-6),
            ),
            ("qwen4exp.ssm.conv_kernel".to_string(), MetadataValue::U32(SSM_CONV_KERNEL)),
            ("qwen4exp.ssm.state_size".to_string(), MetadataValue::U32(SSM_STATE_SIZE)),
            ("qwen4exp.ssm.group_count".to_string(), MetadataValue::U32(SSM_GROUP_COUNT)),
            ("qwen4exp.ssm.time_step_rank".to_string(), MetadataValue::U32(SSM_TIME_STEP_RANK)),
            ("qwen4exp.ssm.inner_size".to_string(), MetadataValue::U32(SSM_INNER_SIZE)),
            // `pr27742.diff:107-108,619` hyper_connection.count/low_rank.
            ("qwen4exp.hyper_connection.count".to_string(), MetadataValue::U32(HC_COUNT)),
            (
                "qwen4exp.hyper_connection.low_rank".to_string(),
                MetadataValue::U32(HC_LOW_RANK),
            ),
            // `pr27742.diff:111-113,2517-2519` indexer head/key/top-k.
            (
                "qwen4exp.attention.indexer_head_count".to_string(),
                MetadataValue::U32(INDEXER_HEAD_COUNT),
            ),
            (
                "qwen4exp.attention.indexer_key_length".to_string(),
                MetadataValue::U32(INDEXER_KEY_LENGTH),
            ),
            (
                "qwen4exp.attention.indexer_top_k".to_string(),
                MetadataValue::U32(INDEXER_TOP_K),
            ),
            // `pr27742.diff:114-118,2523` compress_ratios: one entry per
            // layer, non-zero only on the (one) attention layer.
            (
                "qwen4exp.attention.compress_ratios".to_string(),
                MetadataValue::Array(MetadataArray::U32(
                    (0..BLOCK_COUNT)
                        .map(|layer| {
                            if is_attention_layer(layer) {
                                INDEXER_COMPRESS_RATIO
                            } else {
                                0
                            }
                        })
                        .collect(),
                )),
            ),
            // Generic MoE keys (`bind.rs:285-298`).
            ("qwen4exp.expert_count".to_string(), MetadataValue::U32(EXPERT_COUNT)),
            (
                "qwen4exp.expert_used_count".to_string(),
                MetadataValue::U32(EXPERT_USED_COUNT),
            ),
            (
                "qwen4exp.expert_feed_forward_length".to_string(),
                MetadataValue::U32(EXPERT_FEED_FORWARD),
            ),
            (
                "qwen4exp.expert_shared_feed_forward_length".to_string(),
                MetadataValue::U32(EXPERT_SHARED_FEED_FORWARD),
            ),
            // PLE n-gram hash (`pr27742.diff:121-142,2529-2577`).
            (
                "qwen4exp.ple.layers".to_string(),
                MetadataValue::Array(MetadataArray::U32(vec![PLE_LAYER])),
            ),
            ("qwen4exp.ple.ngram_size".to_string(), MetadataValue::U32(PLE_NGRAM_SIZE)),
            (
                "qwen4exp.ple.heads_per_ngram".to_string(),
                MetadataValue::U32(PLE_HEADS_PER_NGRAM),
            ),
            ("qwen4exp.ple.conv_kernel".to_string(), MetadataValue::U32(PLE_CONV_KERNEL)),
            (
                "qwen4exp.ple.eos_token_id".to_string(),
                MetadataValue::U32(PLE_EOS_TOKEN_ID),
            ),
            (
                "qwen4exp.embedding_length_per_layer_input".to_string(),
                MetadataValue::U32(PLE_HEAD_DIM),
            ),
            (
                "qwen4exp.ple.layer_multipliers".to_string(),
                MetadataValue::Array(MetadataArray::U64(vec![
                    6_364_136_223_846_793_005,
                    1_442_695_040_888_963_407,
                    2_685_821_657_736_338_717,
                    3_202_034_522_624_059_733,
                ])),
            ),
            (
                "qwen4exp.ple.head_offsets".to_string(),
                MetadataValue::Array(MetadataArray::U64(
                    (0..4u32).map(|head| u64::from(head * PLE_ROWS_PER_HEAD)).collect(),
                )),
            ),
            (
                "qwen4exp.ple.head_vocab_sizes".to_string(),
                MetadataValue::Array(MetadataArray::U64(vec![u64::from(PLE_ROWS_PER_HEAD); 4])),
            ),
        ]
    }

    pub(crate) fn tokenizer_metadata() -> Vec<(String, MetadataValue)> {
        let tokens: Vec<String> = (0..VOCAB).map(|byte| (byte as u8 as char).to_string()).collect();
        let scores = vec![0.0f32; VOCAB as usize];
        vec![
            ("tokenizer.ggml.model".to_string(), MetadataValue::String("llama".to_string())),
            ("tokenizer.ggml.tokens".to_string(), MetadataValue::Array(MetadataArray::String(tokens))),
            ("tokenizer.ggml.scores".to_string(), MetadataValue::Array(MetadataArray::F32(scores))),
            ("tokenizer.ggml.bos_token_id".to_string(), MetadataValue::U32(1)),
            (
                "tokenizer.ggml.eos_token_id".to_string(),
                MetadataValue::U32(PLE_EOS_TOKEN_ID),
            ),
        ]
    }

    fn is_attention_layer(layer: u32) -> bool {
        (layer + 1).is_multiple_of(FULL_ATTENTION_INTERVAL)
    }

    /// One layer's tensors, dispatched on GDN vs. sparse-attention per
    /// `pr27742.diff:2654-2696`. Names cited per tensor below; every one is
    /// either declared directly by the PR (`LLM_TENSOR_HC_*`,
    /// `LLM_TENSOR_PLE_*`, `qwen4exp.py`'s conversion side) or reused
    /// unchanged from an existing arch (`LLM_TENSOR_ATTN_*`,
    /// `LLM_TENSOR_SSM_*`, `LLM_TENSOR_FFN_*_EXPS`/`_SHEXP`, `LLM_TENSOR_
    /// INDEXER_*` -- the last read from the plan's own citation of `gguf-py/
    /// gguf/constants.py`, `flash-next-plan.md:110`, since the indexer
    /// tensor-name strings predate this PR's diff and so never appear as an
    /// added line in it).
    fn layer_tensors(layer: u32, seed: u64) -> Vec<TensorPayload<'static>> {
        let mut tensors = vec![
            // `pr27742.diff:2645-2652` (hc_attn_*/hc_ffn_*, two HC modules
            // per layer).
            vector_tensor(&format!("blk.{layer}.hc_attn_norm.weight"), HC_COUNT * EMBEDDING, seed + 1),
            matmul_tensor(
                &format!("blk.{layer}.hc_attn_down.weight"),
                HC_COUNT * EMBEDDING,
                HC_LOW_RANK,
                GgmlType::F32,
                seed + 2,
            ),
            matmul_tensor(
                &format!("blk.{layer}.hc_attn_up.weight"),
                HC_LOW_RANK,
                HC_COUNT * EMBEDDING,
                GgmlType::F32,
                seed + 3,
            ),
            matmul_tensor(
                &format!("blk.{layer}.hc_attn_inject.weight"),
                HC_COUNT * EMBEDDING,
                HC_COUNT,
                GgmlType::F32,
                seed + 4,
            ),
            vector_tensor(&format!("blk.{layer}.hc_ffn_norm.weight"), HC_COUNT * EMBEDDING, seed + 5),
            matmul_tensor(
                &format!("blk.{layer}.hc_ffn_down.weight"),
                HC_COUNT * EMBEDDING,
                HC_LOW_RANK,
                GgmlType::F32,
                seed + 6,
            ),
            matmul_tensor(
                &format!("blk.{layer}.hc_ffn_up.weight"),
                HC_LOW_RANK,
                HC_COUNT * EMBEDDING,
                GgmlType::F32,
                seed + 7,
            ),
            matmul_tensor(
                &format!("blk.{layer}.hc_ffn_inject.weight"),
                HC_COUNT * EMBEDDING,
                HC_COUNT,
                GgmlType::F32,
                seed + 8,
            ),
        ];

        if is_attention_layer(layer) {
            // `pr27742.diff:2656-2666`: q holds [query|gate] interleaved,
            // plus the indexer q/k projections and norms.
            tensors.push(matmul_tensor(
                &format!("blk.{layer}.attn_q.weight"),
                EMBEDDING,
                ATTN_HEAD_DIM * QUERY_HEADS * 2,
                GgmlType::F32,
                seed + 10,
            ));
            tensors.push(matmul_tensor(
                &format!("blk.{layer}.attn_k.weight"),
                EMBEDDING,
                ATTN_HEAD_DIM * KV_HEADS,
                GgmlType::F32,
                seed + 11,
            ));
            tensors.push(matmul_tensor(
                &format!("blk.{layer}.attn_v.weight"),
                EMBEDDING,
                ATTN_HEAD_DIM * KV_HEADS,
                GgmlType::F32,
                seed + 12,
            ));
            tensors.push(matmul_tensor(
                &format!("blk.{layer}.attn_output.weight"),
                ATTN_HEAD_DIM * QUERY_HEADS,
                EMBEDDING,
                GgmlType::F32,
                seed + 13,
            ));
            tensors.push(vector_tensor(&format!("blk.{layer}.attn_q_norm.weight"), ATTN_HEAD_DIM, seed + 14));
            tensors.push(vector_tensor(&format!("blk.{layer}.attn_k_norm.weight"), ATTN_HEAD_DIM, seed + 15));
            tensors.push(matmul_tensor(
                &format!("blk.{layer}.attn_index_q_proj.weight"),
                EMBEDDING,
                INDEXER_HEAD_COUNT * INDEXER_KEY_LENGTH,
                GgmlType::F32,
                seed + 16,
            ));
            tensors.push(matmul_tensor(
                &format!("blk.{layer}.attn_index_k_proj.weight"),
                EMBEDDING,
                INDEXER_KEY_LENGTH,
                GgmlType::F32,
                seed + 17,
            ));
            tensors.push(vector_tensor(
                &format!("blk.{layer}.attn_index_q_norm.weight"),
                INDEXER_KEY_LENGTH,
                seed + 18,
            ));
            tensors.push(vector_tensor(
                &format!("blk.{layer}.attn_index_k_norm.weight"),
                INDEXER_KEY_LENGTH,
                seed + 19,
            ));
        } else {
            // `pr27742.diff:2667-2676` GDN: fused qkv + separate gate,
            // conv1d, alpha/beta, gated output norm.
            let key_dim = SSM_STATE_SIZE * SSM_GROUP_COUNT;
            let value_dim = SSM_STATE_SIZE * SSM_TIME_STEP_RANK;
            let conv_dim = key_dim * 2 + value_dim;
            let head_v_dim = SSM_STATE_SIZE;

            tensors.push(matmul_tensor(
                &format!("blk.{layer}.attn_qkv.weight"),
                EMBEDDING,
                key_dim * 2 + value_dim,
                GgmlType::F32,
                seed + 20,
            ));
            tensors.push(matmul_tensor(
                &format!("blk.{layer}.attn_gate.weight"),
                EMBEDDING,
                value_dim,
                GgmlType::F32,
                seed + 21,
            ));
            tensors.push(vector_tensor(&format!("blk.{layer}.ssm_conv1d.weight"), conv_dim * SSM_CONV_KERNEL, seed + 22));
            tensors.push(vector_tensor(&format!("blk.{layer}.ssm_dt.bias"), SSM_TIME_STEP_RANK, seed + 23));
            tensors.push(vector_tensor(&format!("blk.{layer}.ssm_a"), SSM_TIME_STEP_RANK, seed + 24));
            tensors.push(matmul_tensor(
                &format!("blk.{layer}.ssm_beta.weight"),
                EMBEDDING,
                SSM_TIME_STEP_RANK,
                GgmlType::F32,
                seed + 25,
            ));
            tensors.push(matmul_tensor(
                &format!("blk.{layer}.ssm_alpha.weight"),
                EMBEDDING,
                SSM_TIME_STEP_RANK,
                GgmlType::F32,
                seed + 26,
            ));
            tensors.push(vector_tensor(&format!("blk.{layer}.ssm_norm.weight"), head_v_dim, seed + 27));
            tensors.push(matmul_tensor(
                &format!("blk.{layer}.ssm_out.weight"),
                value_dim,
                EMBEDDING,
                GgmlType::F32,
                seed + 28,
            ));
        }

        if layer == PLE_LAYER {
            // `pr27742.diff:2679-2685` PLE per-layer key/value LoRA
            // projections, grouped RMSNorm x3, dilated conv1d.
            let hc_dim = HC_COUNT * EMBEDDING;
            tensors.push(matmul_tensor(&format!("blk.{layer}.ple_key.weight"), EMBEDDING, hc_dim, GgmlType::F32, seed + 30));
            tensors.push(matmul_tensor(
                &format!("blk.{layer}.ple_value.weight"),
                EMBEDDING,
                EMBEDDING,
                GgmlType::F32,
                seed + 31,
            ));
            tensors.push(vector_tensor(&format!("blk.{layer}.ple_norm_key.weight"), hc_dim, seed + 32));
            tensors.push(vector_tensor(&format!("blk.{layer}.ple_norm_query.weight"), hc_dim, seed + 33));
            tensors.push(vector_tensor(&format!("blk.{layer}.ple_norm_conv.weight"), hc_dim, seed + 34));
            tensors.push(vector_tensor(&format!("blk.{layer}.ple_conv1d.weight"), PLE_CONV_KERNEL * hc_dim, seed + 35));
        }

        // `pr27742.diff:2688-2695` MoE router + expert stack + shared
        // expert, `append_moe_ffn`'s own tensor-name convention (unchanged
        // by this PR, per the plan's own table).
        tensors.push(matmul_tensor(
            &format!("blk.{layer}.ffn_gate_inp.weight"),
            EMBEDDING,
            EXPERT_COUNT,
            GgmlType::F32,
            seed + 40,
        ));
        tensors.push(matmul_tensor(
            &format!("blk.{layer}.ffn_gate_exps.weight"),
            EMBEDDING,
            EXPERT_FEED_FORWARD * EXPERT_COUNT,
            GgmlType::F32,
            seed + 41,
        ));
        tensors.push(matmul_tensor(
            &format!("blk.{layer}.ffn_up_exps.weight"),
            EMBEDDING,
            EXPERT_FEED_FORWARD * EXPERT_COUNT,
            GgmlType::F32,
            seed + 42,
        ));
        tensors.push(matmul_tensor(
            &format!("blk.{layer}.ffn_down_exps.weight"),
            EXPERT_FEED_FORWARD * EXPERT_COUNT,
            EMBEDDING,
            GgmlType::F32,
            seed + 43,
        ));
        tensors.push(vector_tensor(&format!("blk.{layer}.ffn_gate_inp_shexp.weight"), EMBEDDING, seed + 44));
        tensors.push(matmul_tensor(
            &format!("blk.{layer}.ffn_gate_shexp.weight"),
            EMBEDDING,
            EXPERT_SHARED_FEED_FORWARD,
            GgmlType::F32,
            seed + 45,
        ));
        tensors.push(matmul_tensor(
            &format!("blk.{layer}.ffn_up_shexp.weight"),
            EMBEDDING,
            EXPERT_SHARED_FEED_FORWARD,
            GgmlType::F32,
            seed + 46,
        ));
        tensors.push(matmul_tensor(
            &format!("blk.{layer}.ffn_down_shexp.weight"),
            EXPERT_SHARED_FEED_FORWARD,
            EMBEDDING,
            GgmlType::F32,
            seed + 47,
        ));

        tensors
    }

    /// Assembles the whole fixture: token embed, the final hyper-connection
    /// mixer (`pr27742.diff:2604-2607`, "there is no output_norm"), the PLE
    /// table (`pr27742.diff:2614-2627`), and every layer's tensors.
    pub(crate) fn build_qwen4exp_model() -> GgufModel<'static> {
        let mut metadata = architecture_metadata();
        metadata.extend(tokenizer_metadata());

        let hc_dim = HC_COUNT * EMBEDDING;
        let mut tensors = vec![
            matmul_tensor("token_embd.weight", EMBEDDING, VOCAB, GgmlType::F32, 1),
            vector_tensor("output_hc_norm.weight", hc_dim, 2),
            matmul_tensor("output_hc_down.weight", hc_dim, HC_LOW_RANK, GgmlType::F32, 3),
            matmul_tensor("output_hc_up.weight", HC_LOW_RANK, hc_dim, GgmlType::F32, 4),
            matmul_tensor(
                "per_layer_token_embd.weight",
                PLE_HEAD_DIM,
                PLE_TABLE_ROWS,
                GgmlType::F32,
                5,
            ),
        ];

        for layer in 0..BLOCK_COUNT {
            tensors.extend(layer_tensors(layer, u64::from(layer) * 1000 + 100));
        }

        let attention_layers = (0..BLOCK_COUNT).filter(|layer| is_attention_layer(*layer)).count();
        println!(
            "synth_qwen4exp_gguf: block_count={BLOCK_COUNT} attention_layers={attention_layers} \
             gdn_layers={} expert_count={EXPERT_COUNT} expert_used_count={EXPERT_USED_COUNT} \
             hc_count={HC_COUNT} ple_table_rows={PLE_TABLE_ROWS}",
            BLOCK_COUNT as usize - attention_layers
        );

        GgufModel {
            version: 3,
            metadata,
            tensors,
        }
    }
}
