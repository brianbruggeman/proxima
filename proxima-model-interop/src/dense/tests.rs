use super::*;
use arrayvec::ArrayVec;
use proxima_gguf::types::GgmlType;
use proxima_gguf::value::MetadataArray;
use proxima_gguf::{GgufModel, MetadataValue, TensorPayload, parse_complete, write_complete};

fn header_bytes(family: &str, floats: &[(&str, f32)]) -> Vec<u8> {
    let mut metadata = vec![(
        "general.architecture".to_string(),
        MetadataValue::String(family.into()),
    )];
    metadata.extend(
        floats
            .iter()
            .map(|(key, value)| (format!("{family}.{key}"), MetadataValue::F32(*value))),
    );
    let model = GgufModel {
        version: 3,
        metadata,
        tensors: Vec::new(),
    };
    write_complete(&model).expect("a header with no tensors encodes")
}

fn granite_input() -> ModelDescriptor {
    gqa_descriptor_from_shape(
        49155,
        1024,
        512,
        16,
        8,
        64,
        24,
        32,
        8,
        false,
        false,
        false,
        false,
        &family_profile("granitemoe").expect("profile embedded"),
    )
}

fn llama_input() -> ModelDescriptor {
    gqa_descriptor_from_shape(
        32000,
        4096,
        14336,
        32,
        8,
        128,
        32,
        0,
        0,
        false,
        false,
        false,
        false,
        &family_profile("llama").expect("profile embedded"),
    )
}

fn llama_architecture() -> ModelHparams {
    ModelHparams {
        vocab: 32000,
        embedding: 4096,
        feed_forward: 14336,
        expert_feed_forward: 14336,
        query_heads: 32,
        kv_heads: 8,
        kv_heads_by_layer: vec![8; 32],
        head_dim: 128,
        block_count: 32,
        expert_count: 0,
        expert_used_count: 0,
        rope_freq_base: 10_000.0,
        rms_epsilon: 1e-5,
        tied_embeddings: false,
        family: "llama".to_string(),
        sliding_rope: None,
    }
}

#[test]
fn an_architecture_reshaped_by_its_own_descriptor_is_unchanged() {
    let architecture = llama_architecture();

    assert_eq!(architecture.reshaped_by(&llama_input()), architecture);
}

#[test]
fn a_config_that_changes_the_layer_count_and_head_width_reshapes_the_architecture() {
    let mut descriptor = llama_input();
    descriptor.block_count = 2;
    descriptor.layers.truncate(2);
    for layer in &mut descriptor.layers {
        layer.attention.head_dim = 64;
    }

    let reshaped = llama_architecture().reshaped_by(&descriptor);

    assert_eq!(
        (reshaped.block_count, reshaped.head_dim, reshaped.kv_heads),
        (2, 64, 8)
    );
}

#[test]
fn a_non_uniform_schedule_keeps_the_header_head_width() {
    let mut descriptor = llama_input();
    descriptor.layers[0].attention.head_dim = 64;

    let reshaped = llama_architecture().reshaped_by(&descriptor);

    assert_eq!(reshaped.head_dim, 128);
}

#[test]
fn header_scales_reach_the_descriptor_of_a_granite_shaped_header() {
    let bytes = header_bytes(
        "granitemoe",
        &[
            ("embedding_scale", 12.0),
            ("residual_scale", 0.22),
            ("logit_scale", 6.0),
            ("attention.scale", 0.015625),
        ],
    );
    let parsed = parse_complete(&bytes).expect("bytes the encoder just wrote parse");
    let input = granite_input();

    let result = with_header_scales(input.clone(), &parsed, "granitemoe");

    assert_eq!(result.embedding_scale, Some(EmbeddingScale::Factor(12.0)));
    assert_eq!(result.logit_scale, Some(6.0));
    assert_eq!(result.residual_scale, Some(0.22));
    assert_eq!(result.layers.len(), 24);
    assert!(
        result
            .layers
            .iter()
            .all(|layer| layer.attention.score_scale == AttentionScoreScale::Factor(0.015625))
    );
    let restored = ModelDescriptor {
        embedding_scale: input.embedding_scale,
        logit_scale: None,
        residual_scale: None,
        layers: input.layers.clone(),
        ..result.clone()
    };
    assert_eq!(restored, input);
}

#[test]
fn a_header_without_scale_keys_leaves_the_descriptor_unchanged() {
    let bytes = header_bytes("llama", &[]);
    let parsed = parse_complete(&bytes).expect("bytes the encoder just wrote parse");
    let input = llama_input();

    assert_eq!(with_header_scales(input.clone(), &parsed, "llama"), input);
}

fn header_with_window(family: &str, window: u32) -> Vec<u8> {
    let model = GgufModel {
        version: 3,
        metadata: vec![
            (
                "general.architecture".to_string(),
                MetadataValue::String(family.into()),
            ),
            (
                format!("{family}.attention.sliding_window"),
                MetadataValue::U32(window),
            ),
        ],
        tensors: Vec::new(),
    };
    write_complete(&model).expect("a header with no tensors encodes")
}

#[test]
fn a_header_sliding_window_reaches_every_layer_of_the_descriptor() {
    let bytes = header_with_window("llama", 4096);
    let parsed = parse_complete(&bytes).expect("bytes the encoder just wrote parse");
    let input = llama_input();

    let result = with_header_window(input.clone(), &parsed, "llama");

    assert_eq!(result.layers.len(), 32);
    assert!(
        result
            .layers
            .iter()
            .all(|layer| layer.attention.mask_window == Some(4096))
    );
    let restored = ModelDescriptor {
        layers: input.layers.clone(),
        ..result
    };
    assert_eq!(
        restored, input,
        "the window is the only field the header changes"
    );
}

#[test]
fn a_header_without_a_sliding_window_leaves_every_layer_unwindowed() {
    let bytes = header_bytes("llama", &[]);
    let parsed = parse_complete(&bytes).expect("bytes the encoder just wrote parse");
    let input = llama_input();

    assert_eq!(with_header_window(input.clone(), &parsed, "llama"), input);
    let zero = header_with_window("llama", 0);
    let parsed_zero = parse_complete(&zero).expect("bytes the encoder just wrote parse");
    assert_eq!(
        with_header_window(input.clone(), &parsed_zero, "llama"),
        input
    );
}

const HYBRID_EMBEDDING: u64 = 8;
const HYBRID_VOCAB: u64 = 16;

fn hybrid_header(kv_heads: &[u32], layer_tensors: &[&str]) -> Vec<u8> {
    let family = "lfm2moe";
    let metadata = vec![
        (
            "general.architecture".to_string(),
            MetadataValue::String(family.into()),
        ),
        (
            format!("{family}.embedding_length"),
            MetadataValue::U32(HYBRID_EMBEDDING as u32),
        ),
        (
            format!("{family}.feed_forward_length"),
            MetadataValue::U32(32),
        ),
        (
            format!("{family}.expert_feed_forward_length"),
            MetadataValue::U32(12),
        ),
        (
            format!("{family}.attention.head_count"),
            MetadataValue::U32(2),
        ),
        (
            format!("{family}.attention.head_count_kv"),
            MetadataValue::Array(MetadataArray::I32(
                kv_heads.iter().map(|&heads| heads as i32).collect(),
            )),
        ),
        (
            format!("{family}.block_count"),
            MetadataValue::U32(kv_heads.len() as u32),
        ),
        (format!("{family}.expert_count"), MetadataValue::U32(4)),
        (format!("{family}.expert_used_count"), MetadataValue::U32(2)),
        (
            format!("{family}.leading_dense_block_count"),
            MetadataValue::U32(1),
        ),
        (format!("{family}.shortconv.l_cache"), MetadataValue::U32(3)),
    ];
    let table = vec![0u8; (HYBRID_EMBEDDING * HYBRID_VOCAB * 4) as usize];
    let marker = [0u8; 4];
    let mut tensors = vec![TensorPayload {
        name: "token_embd.weight".to_string(),
        dims: ArrayVec::from_iter([HYBRID_EMBEDDING, HYBRID_VOCAB]),
        ggml_type: GgmlType::F32,
        data: &table,
    }];
    tensors.extend(layer_tensors.iter().map(|name| TensorPayload {
        name: (*name).to_string(),
        dims: ArrayVec::from_iter([1u64]),
        ggml_type: GgmlType::F32,
        data: &marker,
    }));
    write_complete(&GgufModel {
        version: 3,
        metadata,
        tensors,
    })
    .expect("a hybrid header with marker tensors encodes")
}

fn hybrid_descriptor(
    kv_heads: &[u32],
    layer_tensors: &[&str],
) -> Result<ModelDescriptor, InteropError> {
    let bytes = hybrid_header(kv_heads, layer_tensors);
    let parsed = parse_complete(&bytes).expect("bytes the encoder just wrote parse");
    let architecture = architecture_from_metadata(&parsed)?;
    descriptor_from_gguf(&parsed, &architecture)
}

#[test]
fn zero_kv_layers_with_a_conv_tensor_become_short_conv_layers_of_a_cached_program() {
    let descriptor = hybrid_descriptor(
        &[0, 2, 0],
        &[
            "blk.0.shortconv.conv.weight",
            "blk.1.attn_q.weight",
            "blk.2.shortconv.conv.weight",
        ],
    )
    .expect("a hybrid header describes itself");

    assert_eq!(
        descriptor
            .layers
            .iter()
            .map(|layer| layer.kind)
            .collect::<Vec<_>>(),
        vec![
            LayerKind::ShortConv,
            LayerKind::Attention,
            LayerKind::ShortConv
        ]
    );
    assert_eq!(
        (descriptor.cache_strategy, descriptor.cache_mask),
        (CacheStrategy::Cached, CacheMask::Padded),
        "the padded two-range builder is the one that keeps shortconv_cache state"
    );
    assert_eq!(
        (descriptor.l_cache, descriptor.leading_dense_block_count),
        (3, 1)
    );
    assert_eq!(
        (descriptor.feed_forward, descriptor.expert_feed_forward),
        (32, 12)
    );
    assert_eq!(descriptor.layers[1].attention.kv_heads, 2);
}

/// `LFM2-350M-Q4_K_M.gguf`'s own `lfm2.*` scalars (block_count 16, embedding_length 1024,
/// feed_forward_length 4608, attention.head_count 16, shortconv.l_cache 3) and its
/// `attention.head_count_kv` array; the header carries no `expert_count`,
/// `expert_used_count` or `leading_dense_block_count` key.
const LFM2_350M_KV_HEADS: [u32; 16] = [0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 8, 0, 8, 0, 8, 0];

fn lfm2_350m_header() -> Vec<u8> {
    let family = "lfm2";
    let metadata = vec![
        (
            "general.architecture".to_string(),
            MetadataValue::String(family.into()),
        ),
        (format!("{family}.block_count"), MetadataValue::U32(16)),
        (format!("{family}.embedding_length"), MetadataValue::U32(1024)),
        (
            format!("{family}.feed_forward_length"),
            MetadataValue::U32(4608),
        ),
        (format!("{family}.attention.head_count"), MetadataValue::U32(16)),
        (
            format!("{family}.attention.head_count_kv"),
            MetadataValue::Array(MetadataArray::I32(
                LFM2_350M_KV_HEADS.iter().map(|&heads| heads as i32).collect(),
            )),
        ),
        (format!("{family}.shortconv.l_cache"), MetadataValue::U32(3)),
    ];
    let table = vec![0u8; (1024 * HYBRID_VOCAB * 4) as usize];
    let marker = [0u8; 4];
    let mut tensors = vec![TensorPayload {
        name: "token_embd.weight".to_string(),
        dims: ArrayVec::from_iter([1024, HYBRID_VOCAB]),
        ggml_type: GgmlType::F32,
        data: &table,
    }];
    tensors.extend((0..16).map(|layer| TensorPayload {
        name: if LFM2_350M_KV_HEADS[layer] == 0 {
            format!("blk.{layer}.shortconv.conv.weight")
        } else {
            format!("blk.{layer}.attn_q.weight")
        },
        dims: ArrayVec::from_iter([1u64]),
        ggml_type: GgmlType::F32,
        data: &marker,
    }));
    write_complete(&GgufModel {
        version: 3,
        metadata,
        tensors,
    })
    .expect("an lfm2-350m header with marker tensors encodes")
}

#[test]
fn a_hybrid_header_with_no_experts_takes_the_dense_ffn_on_every_layer() {
    let bytes = lfm2_350m_header();
    let parsed = parse_complete(&bytes).expect("bytes the encoder just wrote parse");
    let architecture = architecture_from_metadata(&parsed).expect("lfm2-350m header reads");

    let descriptor = descriptor_from_gguf(&parsed, &architecture).expect("lfm2-350m describes itself");

    assert_eq!(
        (architecture.expert_count, descriptor.leading_dense_block_count),
        (0, 16),
        "no expert_count means every one of the 16 blocks is a dense block"
    );
    let forward = proxima_tensor::spec::build_forward(&descriptor)
        .expect("a dense hybrid lowers without the routed builder");
    assert_eq!(
        (descriptor.cache_strategy, descriptor.cache_mask),
        (CacheStrategy::Cached, CacheMask::Padded),
        "lfm2-350m's zero-kv layers are short-conv layers the cached builder supports"
    );
    assert_eq!(
        forward.layer_roots.len(),
        16,
        "every one of the 16 layers owns cache roots: 10 short-conv histories and 6 kv caches"
    );
}

#[test]
fn a_zero_kv_layer_whose_tensors_say_attention_is_refused() {
    let outcome = hybrid_descriptor(&[0, 2], &["blk.0.attn_q.weight", "blk.1.attn_q.weight"]);

    assert!(
        matches!(outcome, Err(InteropError::UnsupportedServingConfig(message)) if message.contains("layer 0"))
    );
}

#[test]
fn a_zero_kv_layer_with_no_mixer_tensor_is_refused() {
    let outcome = hybrid_descriptor(&[0, 2], &["blk.1.attn_q.weight"]);

    assert!(matches!(
        outcome,
        Err(InteropError::Tensor(
            proxima_tensor::TensorError::UndeterminedLayerKind { layer: 0 }
        ))
    ));
}

#[test]
fn attention_layers_that_disagree_on_kv_heads_are_still_refused() {
    let outcome = hybrid_descriptor(
        &[0, 2, 4],
        &[
            "blk.0.shortconv.conv.weight",
            "blk.1.attn_q.weight",
            "blk.2.attn_q.weight",
        ],
    );

    assert!(matches!(
        outcome,
        Err(InteropError::HeterogeneousMetadataArray {
            distinct_values: 2,
            ..
        })
    ));
}

#[test]
fn a_zero_scale_means_unset_like_llama_cpp() {
    let bytes = header_bytes("llama", &[("residual_scale", 0.0), ("logit_scale", 0.0)]);
    let parsed = parse_complete(&bytes).expect("bytes the encoder just wrote parse");
    let input = llama_input();

    assert_eq!(with_header_scales(input.clone(), &parsed, "llama"), input);
}

fn lfm2_moe_architecture() -> ModelHparams {
    ModelHparams {
        vocab: 65536,
        embedding: 2048,
        feed_forward: 1792,
        expert_feed_forward: 1792,
        query_heads: 32,
        kv_heads: 8,
        kv_heads_by_layer: vec![8; 24],
        head_dim: 64,
        block_count: 24,
        expert_count: 32,
        expert_used_count: 4,
        rope_freq_base: 1_000_000.0,
        rms_epsilon: 1e-5,
        tied_embeddings: true,
        family: "lfm2moe".to_string(),
        sliding_rope: None,
    }
}

#[test]
fn a_hybrid_with_wider_dense_blocks_keeps_the_expert_width_in_its_architecture() {
    let architecture = lfm2_moe_architecture();
    let descriptor = ModelDescriptor {
        feed_forward: 7168,
        expert_feed_forward: 1792,
        leading_dense_block_count: 2,
        ..granite_input()
    };
    let descriptor = ModelDescriptor {
        vocab: architecture.vocab,
        embedding: architecture.embedding,
        query_heads: architecture.query_heads,
        block_count: architecture.block_count,
        expert_count: architecture.expert_count,
        expert_used_count: architecture.expert_used_count,
        ..descriptor
    };

    let reshaped = architecture.reshaped_by(&descriptor);

    assert_eq!(
        reshaped.feed_forward, 1792,
        "the expert slab cuts each expert at feed_forward rows, and the checkpoint's expert tensors hold 1792"
    );
}

#[test]
fn a_dense_config_takes_its_feed_forward_width_from_the_descriptor() {
    let descriptor = ModelDescriptor {
        feed_forward: 8192,
        expert_feed_forward: 8192,
        ..llama_input()
    };

    let reshaped = llama_architecture().reshaped_by(&descriptor);

    assert_eq!(reshaped.feed_forward, 8192);
}

#[test]
fn an_expert_config_with_no_separate_expert_width_uses_the_dense_width() {
    let descriptor = ModelDescriptor {
        feed_forward: 640,
        expert_feed_forward: 0,
        ..granite_input()
    };

    let reshaped = lfm2_moe_architecture().reshaped_by(&descriptor);

    assert_eq!(reshaped.feed_forward, 640);
}
