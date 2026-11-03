use arrayvec::ArrayVec;
use proxima_gguf::{
    GgmlType, GgufModel, MetadataArray, MetadataValue, TensorPayload, write_complete,
};
use proxima_tensor::test_support::Lcg;

const EMBEDDING: u64 = 16;
const FEED_FORWARD: u64 = 32;
const VOCAB: u64 = 257;
const FULL_HEAD_DIM: u64 = 512;
const SLIDING_HEAD_DIM: u64 = 256;
const SLIDING_WINDOW: u32 = 256;
const SLIDING_PATTERN: [bool; 3] = [true, true, false];

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

fn byte_tokens() -> MetadataValue {
    let mut names: Vec<String> = (0..=255u8).map(|byte| format!("<0x{byte:02X}>")).collect();
    names.push(String::from("<eos-marker>"));
    MetadataValue::Array(MetadataArray::String(names))
}

fn key(name: &str, value: MetadataValue) -> (String, MetadataValue) {
    (name.to_string(), value)
}

fn layer_specs(layer: usize, head_dim: u64, rng: &mut Lcg) -> Vec<TensorSpec> {
    let name = |suffix: &str| format!("blk.{layer}.{suffix}");
    let projected = [EMBEDDING, head_dim];
    vec![
        spec(&name("attn_norm.weight"), &[EMBEDDING], 0.1, 1.0, rng),
        spec(&name("post_attention_norm.weight"), &[EMBEDDING], 0.1, 1.0, rng),
        spec(&name("attn_q_norm.weight"), &[head_dim], 0.1, 1.0, rng),
        spec(&name("attn_k_norm.weight"), &[head_dim], 0.1, 1.0, rng),
        spec(&name("attn_q.weight"), &projected, 0.25, 0.0, rng),
        spec(&name("attn_k.weight"), &projected, 0.25, 0.0, rng),
        spec(&name("attn_v.weight"), &projected, 0.25, 0.0, rng),
        spec(&name("attn_output.weight"), &[head_dim, EMBEDDING], 0.25, 0.0, rng),
        spec(&name("layer_output_scale.weight"), &[1], 0.0, 1.0, rng),
        spec(&name("ffn_norm.weight"), &[EMBEDDING], 0.1, 1.0, rng),
        spec(&name("ffn_gate.weight"), &[EMBEDDING, FEED_FORWARD], 0.25, 0.0, rng),
        spec(&name("ffn_up.weight"), &[EMBEDDING, FEED_FORWARD], 0.25, 0.0, rng),
        spec(&name("ffn_down.weight"), &[FEED_FORWARD, EMBEDDING], 0.25, 0.0, rng),
        spec(&name("post_ffw_norm.weight"), &[EMBEDDING], 0.1, 1.0, rng),
    ]
}

fn metadata() -> Vec<(String, MetadataValue)> {
    let count = |value: u64| MetadataValue::U32(u32::try_from(value).expect("fits u32"));
    vec![
        key("general.architecture", MetadataValue::String("gemma4".into())),
        key("gemma4.embedding_length", count(EMBEDDING)),
        key("gemma4.context_length", MetadataValue::U32(4096)),
        key("gemma4.block_count", count(SLIDING_PATTERN.len() as u64)),
        key("gemma4.feed_forward_length", count(FEED_FORWARD)),
        key("gemma4.attention.head_count", MetadataValue::U32(1)),
        key("gemma4.attention.head_count_kv", MetadataValue::U32(1)),
        key("gemma4.attention.key_length", count(FULL_HEAD_DIM)),
        key("gemma4.attention.value_length", count(FULL_HEAD_DIM)),
        key("gemma4.attention.key_length_swa", count(SLIDING_HEAD_DIM)),
        key("gemma4.attention.value_length_swa", count(SLIDING_HEAD_DIM)),
        key("gemma4.attention.sliding_window", MetadataValue::U32(SLIDING_WINDOW)),
        key(
            "gemma4.attention.sliding_window_pattern",
            MetadataValue::Array(MetadataArray::Bool(SLIDING_PATTERN.to_vec())),
        ),
        key("gemma4.rope.freq_base", MetadataValue::F32(1.0e6)),
        key("gemma4.rope.freq_base_swa", MetadataValue::F32(1.0e4)),
        key("gemma4.rope.dimension_count", count(FULL_HEAD_DIM)),
        key("gemma4.rope.dimension_count_swa", count(SLIDING_HEAD_DIM)),
        key("tokenizer.ggml.model", MetadataValue::String("gpt2".into())),
        key("tokenizer.ggml.tokens", byte_tokens()),
        key(
            "tokenizer.ggml.merges",
            MetadataValue::Array(MetadataArray::String(Vec::new())),
        ),
    ]
}

/// A three-layer gemma4 checkpoint (two sliding, one full) with seeded f32 weights, small
/// enough that CPU decode takes milliseconds; the serving state machine does not read weights.
pub fn tiny_gemma4_checkpoint() -> Vec<u8> {
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    let mut specs = vec![
        spec("token_embd.weight", &[EMBEDDING, VOCAB], 1.0, 0.0, &mut rng),
        spec("output_norm.weight", &[EMBEDDING], 0.1, 1.0, &mut rng),
    ];
    let mut rope_frequencies = spec("rope_freqs.weight", &[256], 0.0, 1.0, &mut rng);
    rope_frequencies.values[64..].fill(1.0e30);
    specs.push(rope_frequencies);
    for (layer, &is_sliding) in SLIDING_PATTERN.iter().enumerate() {
        let head_dim = if is_sliding { SLIDING_HEAD_DIM } else { FULL_HEAD_DIM };
        specs.extend(layer_specs(layer, head_dim, &mut rng));
    }
    let payloads: Vec<Vec<u8>> = specs
        .iter()
        .map(|tensor| tensor.values.iter().flat_map(|value| value.to_le_bytes()).collect())
        .collect();
    let model = GgufModel {
        version: 3,
        metadata: metadata(),
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
