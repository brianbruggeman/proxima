//! Verifies [`synth_qwen35_gguf`]'s `--arch qwen4exp` fixture parses with
//! [`proxima_gguf`] and carries exactly the hparam keys and tensor names
//! llama.cpp PR #27742's own `qwen4exp` reader expects
//! (`scratchpad/pr27742.diff`, cited per row in the tables below) --
//! S1's "does the fixture look like the real checkpoint's header" gate,
//! never a claim about forward-pass correctness (that is S2+).
//!
//! Reuses the generator via `#[path]` (`examples/synth_qwen35_gguf.rs`),
//! same convention [`qwen35_synth_hybrid.rs`] already established for the
//! qwen35 fixture -- one generator, two consuming tests, no duplication.

#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "../examples/synth_qwen35_gguf.rs"]
#[allow(dead_code)]
mod fixture;

use proxima_gguf::types::{GgmlType, MetadataType};
use proxima_gguf::value::MetadataValue;
use proxima_gguf::writer::write_complete;

fn write_fixture_to_tempdir() -> (tempfile::TempDir, Vec<u8>) {
    let directory = tempfile::tempdir().expect("create tempdir for qwen4exp fixture");
    let model = fixture::qwen4exp::build_qwen4exp_model();
    let bytes = write_complete(&model).expect("write synthetic qwen4exp gguf");
    let path = directory.path().join("qwen4exp-tiny.gguf");
    std::fs::write(&path, &bytes).expect("write fixture bytes to tempdir");
    (directory, bytes)
}

/// One expected hparam key: name, wire type, and the diff line that defines
/// its `LLM_KV_*` mapping (or, for a key qwen4exp reuses unchanged from an
/// existing arch, this crate's own reader that already spells the generic
/// `{architecture}.*` form).
struct ExpectedKey {
    name: &'static str,
    kind: MetadataType,
    cite: &'static str,
}

const EXPECTED_KEYS: &[ExpectedKey] = &[
    ExpectedKey { name: "general.architecture", kind: MetadataType::String, cite: "pr27742.diff:611" },
    ExpectedKey { name: "qwen4exp.embedding_length", kind: MetadataType::U32, cite: "generic embedding_length key" },
    ExpectedKey { name: "qwen4exp.attention.head_count", kind: MetadataType::U32, cite: "generic attention.head_count key" },
    ExpectedKey { name: "qwen4exp.attention.head_count_kv", kind: MetadataType::U32, cite: "generic attention.head_count_kv key" },
    ExpectedKey { name: "qwen4exp.block_count", kind: MetadataType::U32, cite: "generic block_count key" },
    ExpectedKey {
        name: "qwen4exp.attention.full_attention_interval",
        kind: MetadataType::U32,
        cite: "pr27742.diff:2582 LLM_KV_FULL_ATTENTION_INTERVAL",
    },
    ExpectedKey {
        name: "qwen4exp.attention.layer_norm_rms_epsilon",
        kind: MetadataType::F32,
        cite: "pr27742.diff:2499 LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, bind.rs:308",
    },
    ExpectedKey { name: "qwen4exp.ssm.conv_kernel", kind: MetadataType::U32, cite: "pr27742.diff:2503" },
    ExpectedKey { name: "qwen4exp.ssm.state_size", kind: MetadataType::U32, cite: "pr27742.diff:2505" },
    ExpectedKey { name: "qwen4exp.ssm.group_count", kind: MetadataType::U32, cite: "pr27742.diff:2507" },
    ExpectedKey { name: "qwen4exp.ssm.time_step_rank", kind: MetadataType::U32, cite: "pr27742.diff:2506" },
    ExpectedKey { name: "qwen4exp.ssm.inner_size", kind: MetadataType::U32, cite: "pr27742.diff:2504" },
    ExpectedKey {
        name: "qwen4exp.hyper_connection.count",
        kind: MetadataType::U32,
        cite: "pr27742.diff:107,619 LLM_KV_HYPER_CONNECTION_COUNT",
    },
    ExpectedKey {
        name: "qwen4exp.hyper_connection.low_rank",
        kind: MetadataType::U32,
        cite: "pr27742.diff:108,255,619 LLM_KV_HYPER_CONNECTION_LOW_RANK",
    },
    ExpectedKey {
        name: "qwen4exp.attention.indexer_head_count",
        kind: MetadataType::U32,
        cite: "pr27742.diff:111,2517 LLM_KV_ATTENTION_INDEXER_HEAD_COUNT",
    },
    ExpectedKey {
        name: "qwen4exp.attention.indexer_key_length",
        kind: MetadataType::U32,
        cite: "pr27742.diff:112,2518 LLM_KV_ATTENTION_INDEXER_KEY_LENGTH",
    },
    ExpectedKey {
        name: "qwen4exp.attention.indexer_top_k",
        kind: MetadataType::U32,
        cite: "pr27742.diff:113,2519 LLM_KV_ATTENTION_INDEXER_TOP_K",
    },
    ExpectedKey {
        name: "qwen4exp.attention.compress_ratios",
        kind: MetadataType::Array,
        cite: "pr27742.diff:114-118,2523 LLM_KV_ATTENTION_COMPRESS_RATIOS",
    },
    ExpectedKey { name: "qwen4exp.expert_count", kind: MetadataType::U32, cite: "bind.rs:285 generic expert_count key" },
    ExpectedKey {
        name: "qwen4exp.expert_used_count",
        kind: MetadataType::U32,
        cite: "bind.rs:294 generic expert_used_count key",
    },
    ExpectedKey {
        name: "qwen4exp.expert_feed_forward_length",
        kind: MetadataType::U32,
        cite: "pr27742.diff:2497 LLM_KV_EXPERT_FEED_FORWARD_LENGTH",
    },
    ExpectedKey {
        name: "qwen4exp.expert_shared_feed_forward_length",
        kind: MetadataType::U32,
        cite: "pr27742.diff:2498 LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH",
    },
    ExpectedKey { name: "qwen4exp.ple.layers", kind: MetadataType::Array, cite: "pr27742.diff:258,621 LLM_KV_PLE_LAYERS" },
    ExpectedKey {
        name: "qwen4exp.ple.ngram_size",
        kind: MetadataType::U32,
        cite: "pr27742.diff:126,259,622 LLM_KV_PLE_NGRAM_SIZE",
    },
    ExpectedKey {
        name: "qwen4exp.ple.heads_per_ngram",
        kind: MetadataType::U32,
        cite: "pr27742.diff:127,260,623 LLM_KV_PLE_HEADS_PER_NGRAM",
    },
    ExpectedKey {
        name: "qwen4exp.ple.conv_kernel",
        kind: MetadataType::U32,
        cite: "pr27742.diff:128,261,624 LLM_KV_PLE_CONV_KERNEL",
    },
    ExpectedKey {
        name: "qwen4exp.ple.eos_token_id",
        kind: MetadataType::U32,
        cite: "pr27742.diff:129,265,628 LLM_KV_PLE_EOS_TOKEN_ID",
    },
    ExpectedKey {
        name: "qwen4exp.embedding_length_per_layer_input",
        kind: MetadataType::U32,
        cite: "pr27742.diff:2085,2548 LLM_KV_EMBEDDING_LENGTH_PER_LAYER",
    },
    ExpectedKey {
        name: "qwen4exp.ple.layer_multipliers",
        kind: MetadataType::Array,
        cite: "pr27742.diff:138,262,625 LLM_KV_PLE_LAYER_MULTIPLIERS",
    },
    ExpectedKey {
        name: "qwen4exp.ple.head_offsets",
        kind: MetadataType::Array,
        cite: "pr27742.diff:140,263,626 LLM_KV_PLE_HEAD_OFFSETS",
    },
    ExpectedKey {
        name: "qwen4exp.ple.head_vocab_sizes",
        kind: MetadataType::Array,
        cite: "pr27742.diff:142,264,627 LLM_KV_PLE_HEAD_VOCAB_SIZES",
    },
];

/// One expected tensor: name, GGUF `ne` shape (`[dims[0], dims[1]]`, the
/// same `[row_len, rows]` convention `matmul_tensor`/`vector_tensor` write
/// in), and the diff line that creates it.
struct ExpectedTensor {
    name: &'static str,
    dims: &'static [u64],
    cite: &'static str,
}

const EMBEDDING: u64 = fixture::qwen4exp::EMBEDDING as u64;
const HC_DIM: u64 = fixture::qwen4exp::HC_COUNT as u64 * EMBEDDING;
const HC_LOW_RANK: u64 = fixture::qwen4exp::HC_LOW_RANK as u64;

const EXPECTED_TENSORS: &[ExpectedTensor] = &[
    ExpectedTensor { name: "token_embd.weight", dims: &[EMBEDDING, 256], cite: "pr27742.diff:2602" },
    ExpectedTensor { name: "output_hc_norm.weight", dims: &[HC_DIM], cite: "pr27742.diff:2605" },
    ExpectedTensor { name: "output_hc_down.weight", dims: &[HC_DIM, HC_LOW_RANK], cite: "pr27742.diff:2606" },
    ExpectedTensor { name: "output_hc_up.weight", dims: &[HC_LOW_RANK, HC_DIM], cite: "pr27742.diff:2607" },
    ExpectedTensor {
        name: "per_layer_token_embd.weight",
        dims: &[32, 64],
        cite: "pr27742.diff:2616-2627 LLM_TENSOR_PER_LAYER_TOKEN_EMBD",
    },
    // Layer 0 -- GDN, also the PLE layer.
    ExpectedTensor { name: "blk.0.hc_attn_norm.weight", dims: &[HC_DIM], cite: "pr27742.diff:2645" },
    ExpectedTensor { name: "blk.0.hc_attn_down.weight", dims: &[HC_DIM, HC_LOW_RANK], cite: "pr27742.diff:2646" },
    ExpectedTensor { name: "blk.0.hc_attn_up.weight", dims: &[HC_LOW_RANK, HC_DIM], cite: "pr27742.diff:2647" },
    ExpectedTensor { name: "blk.0.hc_attn_inject.weight", dims: &[HC_DIM, 2], cite: "pr27742.diff:2648" },
    ExpectedTensor { name: "blk.0.hc_ffn_norm.weight", dims: &[HC_DIM], cite: "pr27742.diff:2649" },
    ExpectedTensor { name: "blk.0.hc_ffn_down.weight", dims: &[HC_DIM, HC_LOW_RANK], cite: "pr27742.diff:2650" },
    ExpectedTensor { name: "blk.0.hc_ffn_up.weight", dims: &[HC_LOW_RANK, HC_DIM], cite: "pr27742.diff:2651" },
    ExpectedTensor { name: "blk.0.hc_ffn_inject.weight", dims: &[HC_DIM, 2], cite: "pr27742.diff:2652" },
    ExpectedTensor { name: "blk.0.attn_qkv.weight", dims: &[EMBEDDING, 128], cite: "pr27742.diff:2668 LLM_TENSOR_ATTN_QKV" },
    ExpectedTensor { name: "blk.0.attn_gate.weight", dims: &[EMBEDDING, 64], cite: "pr27742.diff:2669 LLM_TENSOR_ATTN_GATE" },
    ExpectedTensor { name: "blk.0.ssm_conv1d.weight", dims: &[512], cite: "pr27742.diff:2670 LLM_TENSOR_SSM_CONV1D" },
    ExpectedTensor { name: "blk.0.ssm_dt.bias", dims: &[4], cite: "pr27742.diff:2671 LLM_TENSOR_SSM_DT" },
    ExpectedTensor { name: "blk.0.ssm_a", dims: &[4], cite: "pr27742.diff:2672 LLM_TENSOR_SSM_A_NOSCAN" },
    ExpectedTensor { name: "blk.0.ssm_beta.weight", dims: &[EMBEDDING, 4], cite: "pr27742.diff:2673 LLM_TENSOR_SSM_BETA" },
    ExpectedTensor { name: "blk.0.ssm_alpha.weight", dims: &[EMBEDDING, 4], cite: "pr27742.diff:2674 LLM_TENSOR_SSM_ALPHA" },
    ExpectedTensor { name: "blk.0.ssm_norm.weight", dims: &[16], cite: "pr27742.diff:2675 LLM_TENSOR_SSM_NORM" },
    ExpectedTensor { name: "blk.0.ssm_out.weight", dims: &[64, EMBEDDING], cite: "pr27742.diff:2676 LLM_TENSOR_SSM_OUT" },
    ExpectedTensor { name: "blk.0.ple_key.weight", dims: &[EMBEDDING, HC_DIM], cite: "pr27742.diff:2680 LLM_TENSOR_PLE_KEY" },
    ExpectedTensor { name: "blk.0.ple_value.weight", dims: &[EMBEDDING, EMBEDDING], cite: "pr27742.diff:2681 LLM_TENSOR_PLE_VALUE" },
    ExpectedTensor { name: "blk.0.ple_norm_key.weight", dims: &[HC_DIM], cite: "pr27742.diff:2682 LLM_TENSOR_PLE_NORM_KEY" },
    ExpectedTensor { name: "blk.0.ple_norm_query.weight", dims: &[HC_DIM], cite: "pr27742.diff:2683 LLM_TENSOR_PLE_NORM_QUERY" },
    ExpectedTensor { name: "blk.0.ple_norm_conv.weight", dims: &[HC_DIM], cite: "pr27742.diff:2684 LLM_TENSOR_PLE_NORM_CONV" },
    ExpectedTensor { name: "blk.0.ple_conv1d.weight", dims: &[4 * HC_DIM], cite: "pr27742.diff:2685 LLM_TENSOR_PLE_CONV1D" },
    // Layer 2 -- the one full-attention (sparse indexer) layer.
    ExpectedTensor { name: "blk.2.attn_q.weight", dims: &[EMBEDDING, 128], cite: "pr27742.diff:2656 create_tensor_qkv" },
    ExpectedTensor { name: "blk.2.attn_k.weight", dims: &[EMBEDDING, 32], cite: "pr27742.diff:2656 create_tensor_qkv" },
    ExpectedTensor { name: "blk.2.attn_v.weight", dims: &[EMBEDDING, 32], cite: "pr27742.diff:2656 create_tensor_qkv" },
    ExpectedTensor { name: "blk.2.attn_output.weight", dims: &[64, EMBEDDING], cite: "pr27742.diff:2657 LLM_TENSOR_ATTN_OUT" },
    ExpectedTensor { name: "blk.2.attn_q_norm.weight", dims: &[16], cite: "pr27742.diff:2659 LLM_TENSOR_ATTN_Q_NORM" },
    ExpectedTensor { name: "blk.2.attn_k_norm.weight", dims: &[16], cite: "pr27742.diff:2660 LLM_TENSOR_ATTN_K_NORM" },
    ExpectedTensor {
        name: "blk.2.attn_index_q_proj.weight",
        dims: &[EMBEDDING, 16],
        cite: "pr27742.diff:2663 LLM_TENSOR_INDEXER_Q_PROJ (name per flash-next-plan.md:110, gguf-py/gguf/constants.py)",
    },
    ExpectedTensor {
        name: "blk.2.attn_index_k_proj.weight",
        dims: &[EMBEDDING, 8],
        cite: "pr27742.diff:2664 LLM_TENSOR_INDEXER_K_PROJ (name per flash-next-plan.md:110)",
    },
    ExpectedTensor {
        name: "blk.2.attn_index_q_norm.weight",
        dims: &[8],
        cite: "pr27742.diff:2665 LLM_TENSOR_INDEXER_Q_NORM (name per flash-next-plan.md:110)",
    },
    ExpectedTensor {
        name: "blk.2.attn_index_k_norm.weight",
        dims: &[8],
        cite: "pr27742.diff:2666 LLM_TENSOR_INDEXER_K_NORM (name per flash-next-plan.md:110)",
    },
    // MoE + shared expert, every layer (checked here on layer 1).
    ExpectedTensor { name: "blk.1.ffn_gate_inp.weight", dims: &[EMBEDDING, 8], cite: "pr27742.diff:2688 LLM_TENSOR_FFN_GATE_INP" },
    ExpectedTensor { name: "blk.1.ffn_gate_exps.weight", dims: &[EMBEDDING, 256], cite: "pr27742.diff:2690 create_tensor_gate_up_exps" },
    ExpectedTensor { name: "blk.1.ffn_up_exps.weight", dims: &[EMBEDDING, 256], cite: "pr27742.diff:2690 create_tensor_gate_up_exps" },
    ExpectedTensor { name: "blk.1.ffn_down_exps.weight", dims: &[256, EMBEDDING], cite: "pr27742.diff:2689 LLM_TENSOR_FFN_DOWN_EXPS" },
    ExpectedTensor { name: "blk.1.ffn_gate_inp_shexp.weight", dims: &[EMBEDDING], cite: "pr27742.diff:2692 LLM_TENSOR_FFN_GATE_INP_SHEXP" },
    ExpectedTensor { name: "blk.1.ffn_gate_shexp.weight", dims: &[EMBEDDING, 32], cite: "pr27742.diff:2693 LLM_TENSOR_FFN_GATE_SHEXP" },
    ExpectedTensor { name: "blk.1.ffn_up_shexp.weight", dims: &[EMBEDDING, 32], cite: "pr27742.diff:2694 LLM_TENSOR_FFN_UP_SHEXP" },
    ExpectedTensor { name: "blk.1.ffn_down_shexp.weight", dims: &[32, EMBEDDING], cite: "pr27742.diff:2695 LLM_TENSOR_FFN_DOWN_SHEXP" },
];

/// The fixture's architecture string is exactly `"qwen4exp"`
/// (`pr27742.diff:313,611` `MODEL_ARCH.QWEN4EXP`/`LLM_ARCH_QWEN4EXP`).
#[test]
fn qwen4exp_fixture_arch_string_matches_pr27742() {
    let (_directory, bytes) = write_fixture_to_tempdir();
    let parsed = proxima_gguf::parse_complete(&bytes).expect("parse qwen4exp fixture");

    let architecture = parsed
        .metadata_value("general.architecture")
        .and_then(MetadataValue::as_str)
        .expect("general.architecture present and a string");

    assert_eq!(architecture, "qwen4exp", "fixture must declare the qwen4exp arch string");
}

/// Every hparam key PR #27742's `load_arch_hparams` reads for `qwen4exp` is
/// present, at the wire type its own `LLM_KV_*` mapping declares.
#[test]
fn qwen4exp_fixture_has_every_expected_hparam_key() {
    let (_directory, bytes) = write_fixture_to_tempdir();
    let parsed = proxima_gguf::parse_complete(&bytes).expect("parse qwen4exp fixture");

    for expected in EXPECTED_KEYS {
        let value = parsed
            .metadata_value(expected.name)
            .unwrap_or_else(|| panic!("missing key {:?} ({})", expected.name, expected.cite));
        assert_eq!(
            value.metadata_type(),
            expected.kind,
            "key {:?} ({}) has wrong wire type",
            expected.name,
            expected.cite
        );
    }
}

/// Every tensor PR #27742's `load_arch_tensors` creates for this fixture's
/// 2-GDN + 1-attention, 8-expert, 1-PLE-layer shape is present, at the
/// exact `[row_len, rows]` shape its own `create_tensor` call declares.
#[test]
fn qwen4exp_fixture_has_every_expected_tensor_shape() {
    let (_directory, bytes) = write_fixture_to_tempdir();
    let parsed = proxima_gguf::parse_complete(&bytes).expect("parse qwen4exp fixture");

    for expected in EXPECTED_TENSORS {
        let tensor = parsed
            .tensors
            .iter()
            .find(|candidate| candidate.name == expected.name)
            .unwrap_or_else(|| panic!("missing tensor {:?} ({})", expected.name, expected.cite));
        assert_eq!(
            tensor.dims.as_slice(),
            expected.dims,
            "tensor {:?} ({}) has wrong shape",
            expected.name,
            expected.cite
        );
        assert_eq!(tensor.ggml_type, GgmlType::F32, "tensor {:?} is written as F32", expected.name);
    }

    assert_eq!(parsed.tensors.len(), 87, "tensor count: 5 model-level + 3 layers of per-layer tensors");
}
