// workspace denies expect_used; a failed fixture must abort the test with its message
#![allow(clippy::expect_used)]

//! The smallest checkpoint a recurrent family loads from, written by the same
//! GGUF writer a real conversion ends in. One physical block runs
//! `ouro.num_loops = 2` times over a 2-wide embedding, so a hand computation
//! reaches every boundary: embeddings `[3, 4]` and `[0, 5]`, identity K and V,
//! zero Q, output and FFN (the branches add nothing, which isolates the
//! per-pass output norm), unit gammas, epsilon one.

use alloc::string::String;
use alloc::vec::Vec;

use proxima_gguf::{
    GgmlType, GgufModel, MetadataArray, MetadataValue, TensorPayload, write_complete,
};

pub(super) const EMBEDDING: [f32; 4] = [3.0, 4.0, 0.0, 5.0];

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn tensor_set() -> Vec<(String, Vec<u64>, Vec<u8>)> {
    let zeros = [0.0f32; 4];
    let identity = [1.0f32, 0.0, 0.0, 1.0];
    let gamma = [1.0f32, 1.0];
    [
        ("token_embd.weight", vec![2, 2], &EMBEDDING[..]),
        ("blk.0.attn_norm.weight", vec![2], &gamma[..]),
        ("blk.0.ffn_norm.weight", vec![2], &gamma[..]),
        ("blk.0.post_attention_norm.weight", vec![2], &gamma[..]),
        ("blk.0.post_ffw_norm.weight", vec![2], &gamma[..]),
        ("blk.0.attn_q.weight", vec![2, 2], &zeros[..]),
        ("blk.0.attn_k.weight", vec![2, 2], &identity[..]),
        ("blk.0.attn_v.weight", vec![2, 2], &identity[..]),
        ("blk.0.attn_output.weight", vec![2, 2], &zeros[..]),
        ("blk.0.ffn_gate.weight", vec![2, 2], &zeros[..]),
        ("blk.0.ffn_up.weight", vec![2, 2], &zeros[..]),
        ("blk.0.ffn_down.weight", vec![2, 2], &zeros[..]),
        ("output_norm.weight", vec![2], &gamma[..]),
        ("output.weight", vec![2, 2], &identity[..]),
    ]
    .into_iter()
    .map(|(name, dims, values)| (name.into(), dims, f32_bytes(values)))
    .collect()
}

/// The tiny recurrent checkpoint as GGUF bytes: `num_loops` passes over one
/// physical block, loop norm on.
pub(super) fn tiny_ouro_gguf(num_loops: u32, skip_loop_final_norm: bool) -> Vec<u8> {
    let payloads = tensor_set();
    let tensors = payloads
        .iter()
        .map(|(name, dims, data)| TensorPayload {
            name: name.clone(),
            dims: dims.iter().copied().collect(),
            ggml_type: GgmlType::F32,
            data,
        })
        .collect();
    write_complete(&GgufModel {
        version: 3,
        metadata: alloc::vec![
            (
                "general.architecture".into(),
                MetadataValue::String("ouro".into())
            ),
            ("ouro.embedding_length".into(), MetadataValue::U32(2)),
            ("ouro.feed_forward_length".into(), MetadataValue::U32(2)),
            ("ouro.attention.head_count".into(), MetadataValue::U32(1)),
            ("ouro.attention.head_count_kv".into(), MetadataValue::U32(1)),
            ("ouro.block_count".into(), MetadataValue::U32(1)),
            ("ouro.rope.dimension_count".into(), MetadataValue::U32(2)),
            (
                "ouro.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::F32(1.0),
            ),
            ("ouro.num_loops".into(), MetadataValue::U32(num_loops)),
            (
                "ouro.skip_loop_final_norm".into(),
                MetadataValue::Bool(skip_loop_final_norm),
            ),
            (
                "tokenizer.ggml.model".into(),
                MetadataValue::String("llama".into())
            ),
            (
                "tokenizer.ggml.tokens".into(),
                MetadataValue::Array(MetadataArray::String(alloc::vec!["a".into(), "b".into()])),
            ),
            (
                "tokenizer.ggml.scores".into(),
                MetadataValue::Array(MetadataArray::F32(alloc::vec![0.0, 0.0])),
            ),
        ],
        tensors,
    })
    .expect("write the tiny recurrent GGUF")
}

/// The same checkpoint as Hugging Face stores it: split-half row order (with a
/// width-2 head the two orders coincide), the four branch norms under their HF
/// names, Q zero, K and V identity, output and FFN zero.
pub(super) fn tiny_ouro_safetensors() -> Vec<u8> {
    let zeros = [0.0f32; 4];
    let identity = [1.0f32, 0.0, 0.0, 1.0];
    let gamma = [1.0f32, 1.0];
    let entries: [(&str, Vec<u64>, &[f32]); 14] = [
        ("model.embed_tokens.weight", vec![2, 2], &EMBEDDING[..]),
        ("model.layers.0.input_layernorm.weight", vec![2], &gamma[..]),
        (
            "model.layers.0.input_layernorm_2.weight",
            vec![2],
            &gamma[..],
        ),
        (
            "model.layers.0.post_attention_layernorm.weight",
            vec![2],
            &gamma[..],
        ),
        (
            "model.layers.0.post_attention_layernorm_2.weight",
            vec![2],
            &gamma[..],
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            vec![2, 2],
            &zeros[..],
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            vec![2, 2],
            &identity[..],
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            vec![2, 2],
            &identity[..],
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            vec![2, 2],
            &zeros[..],
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            vec![2, 2],
            &zeros[..],
        ),
        ("model.layers.0.mlp.up_proj.weight", vec![2, 2], &zeros[..]),
        (
            "model.layers.0.mlp.down_proj.weight",
            vec![2, 2],
            &zeros[..],
        ),
        ("model.norm.weight", vec![2], &gamma[..]),
        ("lm_head.weight", vec![2, 2], &identity[..]),
    ];
    let encoded: Vec<(&str, Vec<u64>, Vec<u8>)> = entries
        .into_iter()
        .map(|(name, shape, values)| (name, shape, f32_bytes(values)))
        .collect();
    let tensors = encoded
        .iter()
        .map(|(name, shape, bytes)| proxima_safetensors::TensorPayload {
            name: (*name).into(),
            dtype: proxima_tensor::DType::Float32,
            shape: shape.clone(),
            data: bytes,
        })
        .collect();
    proxima_safetensors::write_complete(&proxima_safetensors::SafetensorsModel {
        tensors,
        metadata: Default::default(),
    })
    .expect("write the tiny recurrent safetensors")
}

/// Its `config.json`, as the pinned checkpoint spells the keys that matter.
pub(super) const TINY_OURO_HF_CONFIG: &[u8] = br#"{
    "model_type": "ouro", "hidden_act": "silu", "total_ut_steps": 2,
    "early_exit_threshold": 1.0, "hidden_size": 2, "num_attention_heads": 1,
    "num_key_value_heads": 1, "num_hidden_layers": 1, "intermediate_size": 2,
    "layer_types": ["full_attention"], "vocab_size": 2, "head_dim": 2,
    "rms_norm_eps": 1.0, "rope_theta": 1000000.0, "tie_word_embeddings": false
}"#;
