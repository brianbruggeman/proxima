// workspace denies expect_used; a failed fixture must abort the test with its message
#![allow(clippy::expect_used)]

use alloc::string::ToString;
use alloc::vec::Vec;

use proxima_gguf::value::{MetadataArray, MetadataValue};
use proxima_tensor::test_support::Lcg;

use super::chunked_prefill_tests::{TensorSpec, encode_with, key, spec, tokens};

const EMBEDDING: u64 = 64;
const FEED_FORWARD: u64 = 32;
pub(super) const VOCAB: u64 = 257;
pub(super) const HEAD_DIM: u64 = 16;
pub(super) const QUERY_HEADS: u64 = 2;
const INNER: u64 = 96;
const STATE: u64 = 2;
const RANK: u64 = 2;
const CONVOLUTION: u64 = 3;
const EXPERTS: u64 = 2;
pub(super) const LAYERS: u64 = 3;

pub(super) fn recurrent_checkpoint(routed: bool) -> Vec<u8> {
    recurrent_checkpoint_with_eos(routed, None)
}

pub(super) fn recurrent_checkpoint_with_eos(routed: bool, eos: Option<u32>) -> Vec<u8> {
    recurrent_checkpoint_shaped(routed, eos, 1, CONVOLUTION, false, (QUERY_HEADS, 1))
}

pub(super) fn recurrent_checkpoint_shaped(
    routed: bool,
    eos: Option<u32>,
    key_heads: u64,
    convolution: u64,
    quantized: bool,
    attention_heads: (u64, u64),
) -> Vec<u8> {
    let (query_heads, kv_heads) = attention_heads;
    let family = if routed { "qwen35moe" } else { "qwen35" };
    let mut random = Lcg(0x6a09_e667_f3bc_c909);
    let mut tensors = alloc::vec![
        spec(
            "token_embd.weight",
            &[EMBEDDING, VOCAB],
            1.0,
            0.0,
            &mut random
        ),
        spec("output_norm.weight", &[EMBEDDING], 0.1, 1.0, &mut random),
        spec("output.weight", &[EMBEDDING, VOCAB], 0.25, 0.0, &mut random),
    ];
    for layer in 0..LAYERS {
        let name = |suffix: &str| alloc::format!("blk.{layer}.{suffix}");
        tensors.extend([
            spec(
                &name("attn_norm.weight"),
                &[EMBEDDING],
                0.1,
                1.0,
                &mut random,
            ),
            spec(
                &name("post_attention_norm.weight"),
                &[EMBEDDING],
                0.1,
                1.0,
                &mut random,
            ),
        ]);
        if layer % 2 == 0 {
            tensors.extend([
                spec(
                    &name("attn_qkv.weight"),
                    &[EMBEDDING, 2 * STATE * key_heads + INNER],
                    0.25,
                    0.0,
                    &mut random,
                ),
                spec(
                    &name("attn_gate.weight"),
                    &[EMBEDDING, INNER],
                    0.25,
                    0.0,
                    &mut random,
                ),
                spec(
                    &name("ssm_conv1d.weight"),
                    &[convolution, 2 * STATE * key_heads + INNER],
                    0.25,
                    0.0,
                    &mut random,
                ),
                spec(&name("ssm_dt"), &[RANK], 0.1, 0.1, &mut random),
                spec(&name("ssm_a"), &[RANK], 0.1, -0.5, &mut random),
                spec(
                    &name("ssm_beta.weight"),
                    &[EMBEDDING, RANK],
                    0.25,
                    0.0,
                    &mut random,
                ),
                spec(
                    &name("ssm_alpha.weight"),
                    &[EMBEDDING, RANK],
                    0.25,
                    0.0,
                    &mut random,
                ),
                spec(
                    &name("ssm_norm.weight"),
                    &[INNER / RANK],
                    0.1,
                    1.0,
                    &mut random,
                ),
                spec(
                    &name("ssm_out.weight"),
                    &[INNER, EMBEDDING],
                    0.25,
                    0.0,
                    &mut random,
                ),
            ]);
        } else {
            tensors.extend([
                spec(
                    &name("attn_q.weight"),
                    &[EMBEDDING, query_heads * HEAD_DIM * 2],
                    0.25,
                    0.0,
                    &mut random,
                ),
                spec(
                    &name("attn_k.weight"),
                    &[EMBEDDING, kv_heads * HEAD_DIM],
                    0.25,
                    0.0,
                    &mut random,
                ),
                spec(
                    &name("attn_v.weight"),
                    &[EMBEDDING, kv_heads * HEAD_DIM],
                    0.25,
                    0.0,
                    &mut random,
                ),
                spec(
                    &name("attn_output.weight"),
                    &[query_heads * HEAD_DIM, EMBEDDING],
                    0.25,
                    0.0,
                    &mut random,
                ),
                spec(
                    &name("attn_q_norm.weight"),
                    &[HEAD_DIM],
                    0.1,
                    1.0,
                    &mut random,
                ),
                spec(
                    &name("attn_k_norm.weight"),
                    &[HEAD_DIM],
                    0.1,
                    1.0,
                    &mut random,
                ),
            ]);
        }
        if routed {
            tensors.extend([
                spec(
                    &name("ffn_gate_inp.weight"),
                    &[EMBEDDING, EXPERTS],
                    0.25,
                    0.0,
                    &mut random,
                ),
                spec(
                    &name("ffn_gate_exps.weight"),
                    &[EMBEDDING, FEED_FORWARD, EXPERTS],
                    0.25,
                    0.0,
                    &mut random,
                ),
                spec(
                    &name("ffn_up_exps.weight"),
                    &[EMBEDDING, FEED_FORWARD, EXPERTS],
                    0.25,
                    0.0,
                    &mut random,
                ),
                spec(
                    &name("ffn_down_exps.weight"),
                    &[FEED_FORWARD, EMBEDDING, EXPERTS],
                    0.25,
                    0.0,
                    &mut random,
                ),
                spec(
                    &name("ffn_gate_inp_shexp.weight"),
                    &[EMBEDDING],
                    0.25,
                    0.0,
                    &mut random,
                ),
                spec(
                    &name("ffn_gate_shexp.weight"),
                    &[EMBEDDING, FEED_FORWARD],
                    0.25,
                    0.0,
                    &mut random,
                ),
                spec(
                    &name("ffn_up_shexp.weight"),
                    &[EMBEDDING, FEED_FORWARD],
                    0.25,
                    0.0,
                    &mut random,
                ),
                spec(
                    &name("ffn_down_shexp.weight"),
                    &[FEED_FORWARD, EMBEDDING],
                    0.25,
                    0.0,
                    &mut random,
                ),
            ]);
        } else {
            tensors.extend([
                spec(
                    &name("ffn_gate.weight"),
                    &[EMBEDDING, FEED_FORWARD],
                    0.25,
                    0.0,
                    &mut random,
                ),
                spec(
                    &name("ffn_up.weight"),
                    &[EMBEDDING, FEED_FORWARD],
                    0.25,
                    0.0,
                    &mut random,
                ),
                spec(
                    &name("ffn_down.weight"),
                    &[FEED_FORWARD, EMBEDDING],
                    0.25,
                    0.0,
                    &mut random,
                ),
            ]);
        }
    }
    let mut metadata = alloc::vec![
        key(
            "general.architecture",
            MetadataValue::String(family.to_string())
        ),
        key(
            "tokenizer.ggml.model",
            MetadataValue::String("gpt2".to_string())
        ),
        key("tokenizer.ggml.tokens", tokens()),
        key(
            "tokenizer.ggml.merges",
            MetadataValue::Array(MetadataArray::String(Vec::new()))
        ),
    ];
    if let Some(eos) = eos {
        metadata.push(key("tokenizer.ggml.eos_token_id", MetadataValue::U32(eos)));
    }
    for (suffix, value) in [
        ("embedding_length", EMBEDDING),
        ("feed_forward_length", FEED_FORWARD),
        ("block_count", LAYERS),
        ("context_length", 32),
        ("full_attention_interval", 2),
        ("attention.head_count", query_heads),
        ("attention.head_count_kv", kv_heads),
        ("attention.key_length", HEAD_DIM),
        ("attention.value_length", HEAD_DIM),
        ("rope.dimension_count", HEAD_DIM / 2),
        ("ssm.conv_kernel", convolution),
        ("ssm.state_size", STATE),
        ("ssm.group_count", key_heads),
        ("ssm.time_step_rank", RANK),
        ("ssm.inner_size", INNER),
        ("expert_count", EXPERTS),
        ("expert_used_count", 1),
        ("expert_feed_forward_length", FEED_FORWARD),
        ("expert_shared_feed_forward_length", FEED_FORWARD),
    ] {
        metadata.push(key(
            &alloc::format!("{family}.{suffix}"),
            MetadataValue::U32(value as u32),
        ));
    }
    encode_with(metadata, &tensors, |tensor| {
        quantized && holds_q8_0_blocks(tensor)
    })
}

fn holds_q8_0_blocks(tensor: &TensorSpec) -> bool {
    let matrix = tensor.name.ends_with(".weight")
        && !tensor.name.contains("norm")
        && !tensor.name.contains("conv1d");
    matrix && tensor.dims.len() > 1 && tensor.dims[0].is_multiple_of(32)
}
