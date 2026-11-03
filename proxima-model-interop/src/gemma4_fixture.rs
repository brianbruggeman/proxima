//! The generated checkpoints the unit tests and `tests/` drive models with; compiled only for
//! unit tests or under `test-support`, so a normal build never carries them.
// workspace denies expect_used; a failed fixture must abort the test with its message
#![allow(clippy::expect_used)]

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use arrayvec::ArrayVec;

use proxima_gguf::quant::q8_0;
use proxima_gguf::value::{MetadataArray, MetadataValue};
use proxima_gguf::{GgmlType, GgufModel, TensorPayload, write_complete};
use proxima_tensor::test_support::Lcg;

pub const EMBEDDING: u64 = 16;
pub const FEED_FORWARD: u64 = 32;
pub const VOCAB: u64 = 257;
pub const FULL_HEAD_DIM: u64 = 512;
pub const SLIDING_HEAD_DIM: u64 = 256;
pub const SLIDING_WINDOW: u32 = 8;
pub const GEMMA4_SLIDING_PATTERN: [bool; 3] = [true, true, false];

pub struct TensorSpec {
    pub name: String,
    pub dims: Vec<u64>,
    pub values: Vec<f32>,
}

pub fn spec(name: &str, dims: &[u64], scale: f32, offset: f32, rng: &mut Lcg) -> TensorSpec {
    let count: u64 = dims.iter().product();
    TensorSpec {
        name: name.to_string(),
        dims: dims.to_vec(),
        values: (0..count)
            .map(|_| offset + scale * rng.next_unit())
            .collect(),
    }
}

pub fn tokens() -> MetadataValue {
    let mut names: Vec<String> = (0..=255u8)
        .map(|byte| alloc::format!("<0x{byte:02X}>"))
        .collect();
    names.push(String::from("<eos-marker>"));
    MetadataValue::Array(MetadataArray::String(names))
}

pub fn encode(metadata: Vec<(String, MetadataValue)>, specs: &[TensorSpec]) -> Vec<u8> {
    encode_with(metadata, specs, |_| false)
}

/// `encode` with the tensors `quantize` selects stored as `Q8_0` blocks, the
/// codec the real recurrent checkpoints ship their matrices in; the rest stay
/// `F32`. A selected tensor's innermost dimension must hold whole blocks.
pub fn encode_with(
    metadata: Vec<(String, MetadataValue)>,
    specs: &[TensorSpec],
    quantize: impl Fn(&TensorSpec) -> bool,
) -> Vec<u8> {
    let payloads: Vec<(GgmlType, Vec<u8>)> = specs
        .iter()
        .map(|tensor| {
            if !quantize(tensor) {
                let bytes = tensor
                    .values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect();
                return (GgmlType::F32, bytes);
            }
            let mut bytes =
                alloc::vec![0u8; (tensor.values.len() / q8_0::QK8_0) * q8_0::BLOCK_BYTES];
            q8_0::quantize(&tensor.values, &mut bytes)
                .expect("the synthetic tensor holds whole q8_0 blocks");
            (GgmlType::Q8_0, bytes)
        })
        .collect();
    let model = GgufModel {
        version: 3,
        metadata,
        tensors: specs
            .iter()
            .zip(&payloads)
            .map(|(tensor, (ggml_type, data))| TensorPayload {
                name: tensor.name.clone(),
                dims: tensor.dims.iter().copied().collect::<ArrayVec<u64, 4>>(),
                ggml_type: *ggml_type,
                data,
            })
            .collect(),
    };
    write_complete(&model).expect("the synthetic checkpoint encodes")
}

pub fn key(name: &str, value: MetadataValue) -> (String, MetadataValue) {
    (name.to_string(), value)
}

pub fn gemma4_checkpoint() -> Vec<u8> {
    gemma4_checkpoint_with(SLIDING_WINDOW, None)
}

pub fn gemma4_checkpoint_with(sliding_window: u32, context_length: Option<u32>) -> Vec<u8> {
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    let mut specs = alloc::vec![
        spec("token_embd.weight", &[EMBEDDING, VOCAB], 1.0, 0.0, &mut rng),
        spec("output_norm.weight", &[EMBEDDING], 0.1, 1.0, &mut rng),
    ];
    let mut rope_frequencies = spec("rope_freqs.weight", &[256], 0.0, 1.0, &mut rng);
    rope_frequencies.values[64..].fill(1.0e30);
    specs.push(rope_frequencies);
    for (layer, &is_sliding) in GEMMA4_SLIDING_PATTERN.iter().enumerate() {
        let head_dim = if is_sliding {
            SLIDING_HEAD_DIM
        } else {
            FULL_HEAD_DIM
        };
        let name = |suffix: &str| alloc::format!("blk.{layer}.{suffix}");
        specs.extend([
            spec(&name("attn_norm.weight"), &[EMBEDDING], 0.1, 1.0, &mut rng),
            spec(
                &name("post_attention_norm.weight"),
                &[EMBEDDING],
                0.1,
                1.0,
                &mut rng,
            ),
            spec(&name("attn_q_norm.weight"), &[head_dim], 0.1, 1.0, &mut rng),
            spec(&name("attn_k_norm.weight"), &[head_dim], 0.1, 1.0, &mut rng),
            spec(
                &name("attn_q.weight"),
                &[EMBEDDING, head_dim],
                0.25,
                0.0,
                &mut rng,
            ),
            spec(
                &name("attn_k.weight"),
                &[EMBEDDING, head_dim],
                0.25,
                0.0,
                &mut rng,
            ),
            spec(
                &name("attn_v.weight"),
                &[EMBEDDING, head_dim],
                0.25,
                0.0,
                &mut rng,
            ),
            spec(
                &name("attn_output.weight"),
                &[head_dim, EMBEDDING],
                0.25,
                0.0,
                &mut rng,
            ),
            spec(&name("layer_output_scale.weight"), &[1], 0.0, 1.0, &mut rng),
            spec(&name("ffn_norm.weight"), &[EMBEDDING], 0.1, 1.0, &mut rng),
            spec(
                &name("ffn_gate.weight"),
                &[EMBEDDING, FEED_FORWARD],
                0.25,
                0.0,
                &mut rng,
            ),
            spec(
                &name("ffn_up.weight"),
                &[EMBEDDING, FEED_FORWARD],
                0.25,
                0.0,
                &mut rng,
            ),
            spec(
                &name("ffn_down.weight"),
                &[FEED_FORWARD, EMBEDDING],
                0.25,
                0.0,
                &mut rng,
            ),
            spec(
                &name("post_ffw_norm.weight"),
                &[EMBEDDING],
                0.1,
                1.0,
                &mut rng,
            ),
        ]);
    }
    let mut metadata = alloc::vec![
        key(
            "general.architecture",
            MetadataValue::String("gemma4".to_string())
        ),
        key(
            "gemma4.embedding_length",
            MetadataValue::U32(EMBEDDING as u32)
        ),
        key(
            "gemma4.block_count",
            MetadataValue::U32(GEMMA4_SLIDING_PATTERN.len() as u32)
        ),
        key(
            "gemma4.feed_forward_length",
            MetadataValue::U32(FEED_FORWARD as u32)
        ),
        key("gemma4.attention.head_count", MetadataValue::U32(1)),
        key("gemma4.attention.head_count_kv", MetadataValue::U32(1)),
        key(
            "gemma4.attention.key_length",
            MetadataValue::U32(FULL_HEAD_DIM as u32)
        ),
        key(
            "gemma4.attention.value_length",
            MetadataValue::U32(FULL_HEAD_DIM as u32)
        ),
        key(
            "gemma4.attention.key_length_swa",
            MetadataValue::U32(SLIDING_HEAD_DIM as u32)
        ),
        key(
            "gemma4.attention.value_length_swa",
            MetadataValue::U32(SLIDING_HEAD_DIM as u32)
        ),
        key(
            "gemma4.attention.sliding_window",
            MetadataValue::U32(sliding_window)
        ),
        key(
            "gemma4.attention.sliding_window_pattern",
            MetadataValue::Array(MetadataArray::Bool(GEMMA4_SLIDING_PATTERN.to_vec())),
        ),
        key("gemma4.rope.freq_base", MetadataValue::F32(1.0e6)),
        key("gemma4.rope.freq_base_swa", MetadataValue::F32(1.0e4)),
        key(
            "gemma4.rope.dimension_count",
            MetadataValue::U32(FULL_HEAD_DIM as u32)
        ),
        key(
            "gemma4.rope.dimension_count_swa",
            MetadataValue::U32(SLIDING_HEAD_DIM as u32)
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
    metadata.extend(context_length.map(|length| {
        key("gemma4.context_length", MetadataValue::U32(length))
    }));
    encode(metadata, &specs)
}
