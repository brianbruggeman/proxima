use super::*;
use proxima_primitives::pipe::sans_io::Outcome;
use proxima_safetensors::SafetensorsParser;
use sha2::{Digest, Sha256};

/// The real, on-disk `config.json` this task's own evidence points at --
/// `~/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-MLX-4bit/config.json`,
/// copied verbatim (checked byte-for-byte against `cat` on the real
/// file), not synthesized. A Qwen3 mixture-of-experts checkpoint: this
/// is the fixture that proves `num_experts`/`num_experts_per_tok`/
/// `moe_intermediate_size` are read, not just the dense fields.
const REAL_QWEN3_MOE_CONFIG_JSON: &str = r#"{
    "architectures": ["Qwen3MoeForCausalLM"],
    "attention_bias": false,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "decoder_sparse_step": 1,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 6144,
    "max_position_embeddings": 40960,
    "max_window_layers": 48,
    "mlp_only_layers": [],
    "model_type": "qwen3_moe",
    "moe_intermediate_size": 768,
    "norm_topk_prob": true,
    "num_attention_heads": 32,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 48,
    "num_key_value_heads": 4,
    "output_router_logits": false,
    "quantization": {"group_size": 64, "bits": 4},
    "quantization_config": {"group_size": 64, "bits": 4},
    "rms_norm_eps": 1e-06,
    "rope_scaling": null,
    "rope_theta": 1000000.0,
    "router_aux_loss_coef": 0.001,
    "sliding_window": null,
    "tie_word_embeddings": false,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.51.0",
    "use_cache": true,
    "use_sliding_window": false,
    "vocab_size": 151936
}"#;

#[test]
fn real_qwen3_moe_config_json_parses_every_field_this_crate_reads() {
    let config = parse_hf_config(REAL_QWEN3_MOE_CONFIG_JSON.as_bytes())
        .expect("real qwen3 moe config.json parses");

    assert_eq!(config.model_type, "qwen3_moe");
    assert_eq!(
        config.architectures,
        alloc::vec![String::from("Qwen3MoeForCausalLM")]
    );
    assert_eq!(config.hidden_size, 2048);
    assert_eq!(config.num_attention_heads, 32);
    assert_eq!(config.num_key_value_heads, Some(4));
    assert_eq!(config.num_hidden_layers, 48);
    assert_eq!(config.intermediate_size, 6144);
    assert_eq!(config.moe_intermediate_size, Some(768));
    assert!((config.rms_norm_eps - 1e-6).abs() < 1e-12);
    assert!((config.rope_theta - 1_000_000.0).abs() < 1e-6);
    assert_eq!(config.vocab_size, 151_936);
    assert_eq!(config.head_dim, Some(128));
    assert_eq!(config.num_experts, Some(128));
    assert_eq!(config.num_experts_per_tok, Some(8));
}

#[test]
fn real_qwen3_moe_config_json_derives_the_real_moe_architecture() {
    let config = parse_hf_config(REAL_QWEN3_MOE_CONFIG_JSON.as_bytes())
        .expect("real qwen3 moe config.json parses");

    let architecture = architecture_from_hf_config(&config).expect("qwen3 config derives");

    assert_eq!(
        architecture,
        ModelHparams {
            vocab: 151_936,
            embedding: 2048,
            feed_forward: 768,
            query_heads: 32,
            kv_heads: 4,
            kv_heads_by_layer: vec![4; 48],
            head_dim: 128,
            block_count: 48,
            expert_count: 128,
            expert_used_count: 8,
            rope_freq_base: 1_000_000.0,
            rms_epsilon: 1e-6,
            tied_embeddings: false,
            family: String::from("qwen3_moe"),
            sliding_rope: None,
        },
        "feed_forward must read moe_intermediate_size (768), not intermediate_size (6144), \
         once expert_count is nonzero"
    );
}

/// A dense (non-MoE) config with none of `num_key_value_heads`/
/// `num_experts`/`num_experts_per_tok`/`moe_intermediate_size`/`head_dim`
/// present -- proves every optional field's fallback, not just the
/// MoE-populated real fixture above.
#[test]
fn dense_config_with_every_optional_field_absent_derives_via_fallbacks() {
    let json = r#"{
        "model_type": "llama",
        "hidden_size": 8,
        "num_attention_heads": 2,
        "num_hidden_layers": 4,
        "intermediate_size": 32,
        "vocab_size": 100
    }"#;
    let config = parse_hf_config(json.as_bytes()).expect("minimal dense config.json parses");
    let architecture = architecture_from_hf_config(&config).expect("dense config derives");

    assert_eq!(
        architecture,
        ModelHparams {
            vocab: 100,
            embedding: 8,
            feed_forward: 32,
            query_heads: 2,
            kv_heads: 2,
            kv_heads_by_layer: vec![2; 4],
            head_dim: 4,
            block_count: 4,
            expert_count: 0,
            expert_used_count: 0,
            rope_freq_base: proxima_tensor::sized::ROPE_FREQ_BASE_DEFAULT,
            rms_epsilon: 1e-5,
            tied_embeddings: false,
            family: String::from("llama"),
            sliding_rope: None,
        },
        "kv_heads falls back to query_heads, head_dim to hidden_size/num_attention_heads, \
         expert_count/expert_used_count to 0, rope_theta to the sizing-config default, \
         rms_norm_eps to HF's own schema default"
    );
}

/// Mixtral's own field name (`num_local_experts`, not Qwen's
/// `num_experts`) must resolve to the SAME [`HfConfig::num_experts`]
/// field via `#[serde(alias = ..)]` -- this is the "family destroyed"
/// case the alias exists for: two real checkpoint families spell the
/// identical fact two different ways.
#[test]
fn mixtral_style_num_local_experts_alias_reads_the_same_field_as_qwens_num_experts() {
    let json = r#"{
        "hidden_size": 8,
        "num_attention_heads": 2,
        "num_hidden_layers": 4,
        "intermediate_size": 32,
        "vocab_size": 100,
        "num_local_experts": 8,
        "num_experts_per_tok": 2
    }"#;
    let config = parse_hf_config(json.as_bytes()).expect("mixtral-style config.json parses");

    assert_eq!(
        config.num_experts,
        Some(8),
        "num_local_experts must alias into num_experts"
    );
    assert_eq!(
        architecture_from_hf_config(&config)
            .expect("moe config derives")
            .expert_count,
        8
    );
}

/// The real, on-disk `config.json` this session downloaded --
/// `~/.lmstudio/models/HuggingFaceTB/SmolLM2-135M-Instruct/config.json`,
/// copied verbatim -- a dense, tied-embedding Llama-family checkpoint:
/// the fixture that proves `tie_word_embeddings` is read into
/// [`ModelHparams::tied_embeddings`], not just the MoE fields the
/// Qwen3 fixture above exercises.
const REAL_SMOLLM2_CONFIG_JSON: &str = r#"{
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 576,
    "intermediate_size": 1536,
    "num_hidden_layers": 30,
    "num_attention_heads": 9,
    "num_key_value_heads": 3,
    "rms_norm_eps": 1e-05,
    "rope_theta": 100000,
    "vocab_size": 49152,
    "tie_word_embeddings": true,
    "torch_dtype": "bfloat16",
    "max_position_embeddings": 8192,
    "hidden_act": "silu",
    "attention_bias": false,
    "mlp_bias": false,
    "rope_scaling": null
}"#;

#[test]
fn real_smollm2_config_json_derives_a_tied_dense_architecture() {
    let config = parse_hf_config(REAL_SMOLLM2_CONFIG_JSON.as_bytes())
        .expect("real smollm2 config.json parses");

    assert!(
        config.tie_word_embeddings,
        "smollm2 ships tie_word_embeddings: true"
    );

    let architecture = architecture_from_hf_config(&config).expect("sliding config derives");
    assert_eq!(
        architecture,
        ModelHparams {
            vocab: 49_152,
            embedding: 576,
            feed_forward: 1536,
            query_heads: 9,
            kv_heads: 3,
            kv_heads_by_layer: vec![3; 30],
            head_dim: 64,
            block_count: 30,
            expert_count: 0,
            expert_used_count: 0,
            rope_freq_base: 100_000.0,
            rms_epsilon: 1e-5,
            tied_embeddings: true,
            family: String::from("llama"),
            sliding_rope: None,
        },
        "head_dim must derive as hidden_size/num_attention_heads (576/9=64) since no explicit \
         head_dim key is present, and tied_embeddings must read config's tie_word_embeddings"
    );
}

/// Qwen2-7B's `config.json` scalars (hidden 3584, 28 heads, 4 kv heads,
/// 28 layers, vocab 152064, rope_theta 1e6). The HF path takes its family
/// from `model_type`, so it reads the same qwen2 profile the GGUF path
/// reads from `general.architecture`: split-half RoPE with no QK-norm
/// tensors.
#[cfg(feature = "std")]
#[test]
fn hf_qwen2_config_reads_the_split_half_profile_the_gguf_path_reads() {
    let json = r#"{
        "model_type": "qwen2",
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_attention_heads": 28,
        "num_hidden_layers": 28,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "vocab_size": 152064,
        "tie_word_embeddings": false
    }"#;
    let config = parse_hf_config(json.as_bytes()).expect("qwen2-7b config.json parses");
    let architecture = architecture_from_hf_config(&config).expect("dense config derives");

    let profile =
        crate::profiles::family_profile(&architecture.family).expect("qwen2 profile embedded");

    assert_eq!(architecture.family, "qwen2");
    assert_eq!(
        profile.rope_pairing(architecture.head_dim),
        proxima_tensor::spec::RopePairing::SplitHalf { pairs: 64 }
    );
}

#[test]
fn missing_required_field_is_a_typed_error_not_a_panic() {
    let json = r#"{"hidden_size": 8}"#;
    let outcome = parse_hf_config(json.as_bytes());
    assert!(
        matches!(outcome, Err(InteropError::MalformedHfConfig { .. })),
        "missing required field must surface as a typed error, got {outcome:?}"
    );
}

#[test]
fn malformed_json_is_a_typed_error_not_a_panic() {
    let outcome = parse_hf_config(b"not json at all");
    assert!(
        matches!(outcome, Err(InteropError::MalformedHfConfig { .. })),
        "malformed json must surface as a typed error, got {outcome:?}"
    );
}

#[test]
fn architecture_matrix_lfm_schedule() {
    let config_bytes = include_bytes!(
        "../../../proxima-tensor/specs/small-model-architecture-matrix/fixtures/semantic-pair-configs/lfm_text.json"
    );
    let header_bytes = include_bytes!(
        "../../../proxima-tensor/specs/small-model-architecture-matrix/fixtures/lfm_text.safetensors.header"
    );
    let config = parse_hf_config(config_bytes).expect("pinned LFM config parses");
    assert_eq!(
        std::format!("{:x}", Sha256::digest(header_bytes)),
        "31a589aa36efce50a5450571144834995319bc35d59ddca12c369485daaf658a"
    );
    let mut parser = SafetensorsParser::new();
    parser.feed(header_bytes);
    let manifest = match parser.poll().expect("pinned safetensors header parses") {
        Outcome::Event(manifest) => manifest.clone(),
        Outcome::NeedMore => panic!("complete header fixture must produce a manifest"),
    };

    assert_eq!(manifest.tensors.len(), 148);
    let layer_kinds = lfm_layer_kinds_from_manifest(&config, &manifest)
        .expect("config schedule matches pinned tensor names and shapes");
    let expected = vec![
        LayerKind::ShortConv,
        LayerKind::ShortConv,
        LayerKind::Attention,
        LayerKind::ShortConv,
        LayerKind::ShortConv,
        LayerKind::Attention,
        LayerKind::ShortConv,
        LayerKind::ShortConv,
        LayerKind::Attention,
        LayerKind::ShortConv,
        LayerKind::Attention,
        LayerKind::ShortConv,
        LayerKind::Attention,
        LayerKind::ShortConv,
        LayerKind::Attention,
        LayerKind::ShortConv,
    ];
    assert_eq!(layer_kinds, expected);

    let architecture = architecture_from_hf_config(&config)
        .expect("pinned LFM config derives its mixed layer schedule");
    let expected_kv_heads: Vec<u32> = expected
        .iter()
        .map(|layer_kind| match layer_kind {
            LayerKind::ShortConv => 0_u32,
            LayerKind::Attention => 8_u32,
            LayerKind::Gdn => panic!("pinned LFM schedule has no GDN layers"),
        })
        .collect::<Vec<_>>();
    assert_eq!(architecture.kv_heads_by_layer, expected_kv_heads);
    assert_eq!(architecture.kv_heads, 0);

    let mut incomplete_config = config.clone();
    incomplete_config.layer_types.pop();
    assert!(architecture_from_hf_config(&incomplete_config).is_err());

    let mut unknown_layer_config = config.clone();
    unknown_layer_config.layer_types[0] = String::from("unknown");
    assert!(architecture_from_hf_config(&unknown_layer_config).is_err());

    let mut missing_marker = manifest.clone();
    missing_marker
        .tensors
        .retain(|tensor| tensor.name != "model.layers.2.self_attn.q_proj.weight");
    assert!(lfm_layer_kinds_from_manifest(&config, &missing_marker).is_err());

    let mut conflicting_marker = manifest.clone();
    let mut attention_tensor = conflicting_marker
        .tensor("model.layers.2.self_attn.q_proj.weight")
        .expect("pinned attention marker exists")
        .clone();
    attention_tensor.name = String::from("model.layers.0.self_attn.q_proj.weight");
    conflicting_marker.tensors.push(attention_tensor);
    assert!(lfm_layer_kinds_from_manifest(&config, &conflicting_marker).is_err());

    let mut wrong_shape = manifest.clone();
    wrong_shape
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == "model.layers.0.conv.conv.weight")
        .expect("pinned conv marker exists")
        .shape[2] = 2;
    assert!(lfm_layer_kinds_from_manifest(&config, &wrong_shape).is_err());
    std::println!(
        "layers=16 schedule_matches=16 wrong_marker_rejected=1 missing_marker_rejected=1 wrong_shape_rejected=1 incomplete_config_rejected=1 unknown_layer_rejected=1"
    );
}
