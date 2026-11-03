// workspace denies expect_used; a failed fixture must abort the test with its message
#![allow(clippy::expect_used)]

//! The smallest checkpoint a recurrent family loads from, written by the same
//! GGUF writer a real conversion ends in. One physical block runs
//! `ouro.num_loops = 2` times over a 2-wide embedding, so a hand computation
//! reaches every boundary: embeddings `[3, 4]` and `[0, 5]`, identity K and V,
//! zero Q, output and FFN (the branches add nothing, which isolates the
//! per-pass output norm), unit gammas, epsilon one.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use proxima_tensor::test_support::Lcg;

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
    let geometry = Geometry {
        hidden: 2,
        feed_forward: 2,
        vocabulary: 2,
        epsilon: 1.0,
        heads: 1,
        kv_heads: 1,
    };
    ouro_gguf(&tensor_set(), &geometry, num_loops, skip_loop_final_norm)
}

pub(super) const STATE_VOCABULARY: u32 = 16;

fn seeded(generator: &mut Lcg, count: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..count)
        .map(|_| offset + scale * generator.next_unit())
        .collect()
}

fn state_tensor_set(
    hidden: usize,
    kv_width: usize,
    feed_forward: usize,
    vocabulary: usize,
) -> Vec<(String, Vec<u64>, Vec<u8>)> {
    let mut generator = Lcg(0x5EED_0A0B_0C0D_0E0F);
    let square = hidden * hidden;
    let narrow = hidden * kv_width;
    let entries: Vec<(&str, Vec<u64>, Vec<f32>)> = vec![
        (
            "token_embd.weight",
            vec![hidden as u64, vocabulary as u64],
            seeded(&mut generator, hidden * vocabulary, 1.0, 0.0),
        ),
        (
            "blk.0.attn_norm.weight",
            vec![hidden as u64],
            seeded(&mut generator, hidden, 0.1, 1.0),
        ),
        (
            "blk.0.ffn_norm.weight",
            vec![hidden as u64],
            seeded(&mut generator, hidden, 0.1, 1.0),
        ),
        (
            "blk.0.post_attention_norm.weight",
            vec![hidden as u64],
            seeded(&mut generator, hidden, 0.1, 1.0),
        ),
        (
            "blk.0.post_ffw_norm.weight",
            vec![hidden as u64],
            seeded(&mut generator, hidden, 0.1, 1.0),
        ),
        (
            "blk.0.attn_q.weight",
            vec![hidden as u64, hidden as u64],
            seeded(&mut generator, square, 0.8, 0.0),
        ),
        (
            "blk.0.attn_k.weight",
            vec![hidden as u64, kv_width as u64],
            seeded(&mut generator, narrow, 0.8, 0.0),
        ),
        (
            "blk.0.attn_v.weight",
            vec![hidden as u64, kv_width as u64],
            seeded(&mut generator, narrow, 0.8, 0.0),
        ),
        (
            "blk.0.attn_output.weight",
            vec![hidden as u64, hidden as u64],
            seeded(&mut generator, square, 0.8, 0.0),
        ),
        (
            "blk.0.ffn_gate.weight",
            vec![hidden as u64, feed_forward as u64],
            seeded(&mut generator, hidden * feed_forward, 0.6, 0.0),
        ),
        (
            "blk.0.ffn_up.weight",
            vec![hidden as u64, feed_forward as u64],
            seeded(&mut generator, hidden * feed_forward, 0.6, 0.0),
        ),
        (
            "blk.0.ffn_down.weight",
            vec![feed_forward as u64, hidden as u64],
            seeded(&mut generator, hidden * feed_forward, 0.6, 0.0),
        ),
        (
            "output_norm.weight",
            vec![hidden as u64],
            seeded(&mut generator, hidden, 0.1, 1.0),
        ),
        (
            "output.weight",
            vec![hidden as u64, vocabulary as u64],
            seeded(&mut generator, hidden * vocabulary, 1.0, 0.0),
        ),
    ];
    entries
        .into_iter()
        .map(|(name, dims, values)| (name.into(), dims, f32_bytes(&values)))
        .collect()
}

/// A checkpoint whose logits depend on every earlier position: sixteen tokens,
/// an 8-wide single head, and seeded dense Q, K, V, output, FFN and head weights
/// (the tiny fixture zeroes Q, output and FFN, so stale cache state is invisible).
pub(super) fn state_sensitive_ouro_gguf(num_loops: u32) -> Vec<u8> {
    let geometry = Geometry {
        hidden: 8,
        feed_forward: 16,
        vocabulary: STATE_VOCABULARY,
        epsilon: 1e-5,
        heads: 1,
        kv_heads: 1,
    };
    let tensors = state_tensor_set(8, 8, 16, STATE_VOCABULARY as usize);
    ouro_gguf(&tensors, &geometry, num_loops, false)
}

/// The grouped state-sensitive weights under a llama header: the dense family
/// lowers its attention to one fused `CachedAttention` op, which the recurrent
/// ouro schedule never does, so this is the checkpoint that reaches the CUDA
/// tiled cached-attention body at sixteen or more rows.
#[cfg(feature = "cuda")]
pub(super) fn grouped_llama_gguf() -> Vec<u8> {
    let payloads: Vec<(String, Vec<u64>, Vec<u8>)> = state_tensor_set(256, 128, 64, STATE_VOCABULARY as usize)
        .into_iter()
        .filter(|(name, _, _)| !name.contains("post_attention_norm") && !name.contains("post_ffw_norm"))
        .collect();
    let tokens: Vec<String> = (0..STATE_VOCABULARY).map(|index| format!("t{index}")).collect();
    let scores = alloc::vec![0.0f32; tokens.len()];
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
            ("general.architecture".into(), MetadataValue::String("llama".into())),
            ("llama.embedding_length".into(), MetadataValue::U32(256)),
            ("llama.feed_forward_length".into(), MetadataValue::U32(64)),
            ("llama.attention.head_count".into(), MetadataValue::U32(2)),
            ("llama.attention.head_count_kv".into(), MetadataValue::U32(1)),
            ("llama.block_count".into(), MetadataValue::U32(1)),
            ("llama.context_length".into(), MetadataValue::U32(512)),
            ("llama.rope.dimension_count".into(), MetadataValue::U32(128)),
            ("llama.attention.layer_norm_rms_epsilon".into(), MetadataValue::F32(1e-5)),
            ("tokenizer.ggml.model".into(), MetadataValue::String("llama".into())),
            ("tokenizer.ggml.tokens".into(), MetadataValue::Array(MetadataArray::String(tokens))),
            ("tokenizer.ggml.scores".into(), MetadataValue::Array(MetadataArray::F32(scores))),
        ],
        tensors,
    })
    .expect("write the grouped llama GGUF")
}

struct Geometry {
    hidden: u32,
    feed_forward: u32,
    vocabulary: u32,
    epsilon: f32,
    heads: u32,
    kv_heads: u32,
}

fn ouro_gguf(
    payloads: &[(String, Vec<u64>, Vec<u8>)],
    geometry: &Geometry,
    num_loops: u32,
    skip_loop_final_norm: bool,
) -> Vec<u8> {
    let tokens: Vec<String> = (0..geometry.vocabulary)
        .map(|index| format!("t{index}"))
        .collect();
    let scores = alloc::vec![0.0f32; tokens.len()];
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
            (
                "ouro.embedding_length".into(),
                MetadataValue::U32(geometry.hidden),
            ),
            (
                "ouro.feed_forward_length".into(),
                MetadataValue::U32(geometry.feed_forward),
            ),
            ("ouro.attention.head_count".into(), MetadataValue::U32(geometry.heads)),
            ("ouro.attention.head_count_kv".into(), MetadataValue::U32(geometry.kv_heads)),
            ("ouro.block_count".into(), MetadataValue::U32(1)),
            (
                "ouro.rope.dimension_count".into(),
                MetadataValue::U32(geometry.hidden / geometry.heads),
            ),
            (
                "ouro.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::F32(geometry.epsilon),
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
                MetadataValue::Array(MetadataArray::String(tokens)),
            ),
            (
                "tokenizer.ggml.scores".into(),
                MetadataValue::Array(MetadataArray::F32(scores)),
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
            dtype: proxima_tensor::DType::FLOAT32,
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
