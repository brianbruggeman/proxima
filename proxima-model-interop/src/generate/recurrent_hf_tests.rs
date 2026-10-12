// workspace denies expect_used; a failed fixture must abort the test with its message
#![allow(clippy::expect_used)]

//! The same recurrent family from Hugging Face files: the config and the
//! family profile make the descriptor, the safetensors bind by what that
//! descriptor's layers need, and the logits are the ones the GGUF route
//! computes for the same numbers.

use alloc::sync::Arc;

use proxima_gguf::parse_complete;
use proxima_tensor::LoopCdMode;
use proxima_tensor::spec::{RopePairing, build_forward};
use proxima_tokenizer::gguf::vocab_from_metadata;

use super::LoadedModel;
use super::ouro_fixture::{TINY_OURO_HF_CONFIG, tiny_ouro_gguf, tiny_ouro_safetensors};
use crate::error::InteropError;
use crate::hf_config::parse_hf_config;
use crate::ouro_descriptor_from_hf_config;
use crate::recurrent_config::RecurrentConfig;
use crate::serving::ServingConfig;

fn near(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 0.000_001 + 0.000_001 * expected.abs(),
            "coordinate {index}: actual={actual} expected={expected}"
        );
    }
}

fn guided(iterations: Option<u32>, mode: LoopCdMode, strength: f32) -> RecurrentConfig {
    RecurrentConfig {
        iterations,
        mode,
        strength,
        reference_iteration: 1,
    }
}

fn data_start(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes[..8].try_into().expect("eight header bytes")) + 8
}

#[test]
fn recurrent_mainline_hf_descriptor_reads_the_config_and_the_profile() {
    let config = parse_hf_config(TINY_OURO_HF_CONFIG).expect("parse the HF config");

    let (architecture, descriptor) =
        ouro_descriptor_from_hf_config(&config, RecurrentConfig::neutral())
            .expect("derive the descriptor");

    assert_eq!(
        architecture.block_count, 1,
        "the architecture stays physical"
    );
    assert_eq!(descriptor.block_count, 1);
    let recurrence = descriptor.recurrence.expect("the checkpoint recurs");
    assert_eq!(recurrence.iterations, 2);
    assert!(recurrence.normalize_each_iteration);
    let layer = &descriptor.layers[0];
    assert_eq!(
        layer.attention.rope_pairing,
        RopePairing::SplitHalf { pairs: 1 }
    );
    assert!(!layer.attention.qk_norm);
    assert!(layer.ffn.post_attention_norm && layer.ffn.exclusive_dense_post_norm);
    build_forward(&descriptor).expect("the descriptor lowers");
}

#[test]
fn recurrent_mainline_hf_descriptor_refuses_what_it_cannot_lower() {
    let refuse = |edit: &dyn Fn(&mut crate::hf_config::HfConfig)| {
        let mut config = parse_hf_config(TINY_OURO_HF_CONFIG).expect("parse the HF config");
        edit(&mut config);
        let outcome = ouro_descriptor_from_hf_config(&config, RecurrentConfig::neutral());
        assert!(
            matches!(outcome, Err(InteropError::InvalidRecurrentConfig { .. })),
            "{:?}",
            outcome.err()
        );
    };

    refuse(&|config| config.skip_loop_final_norm = Some(true));
    refuse(&|config| config.early_exit_threshold = Some(0.5));
    refuse(&|config| config.hidden_act = Some("gelu".into()));
    refuse(&|config| config.layer_types = vec!["sliding_attention".into()]);
    refuse(&|config| config.model_type = "llama".into());
    let mut config = parse_hf_config(TINY_OURO_HF_CONFIG).expect("parse the HF config");
    config.total_ut_steps = None;
    assert!(matches!(
        ouro_descriptor_from_hf_config(&config, RecurrentConfig::neutral()),
        Err(InteropError::MissingMetadataKey { .. })
    ));
}

#[test]
fn recurrent_mainline_hf_shared_descriptor_literal() {
    let config = parse_hf_config(TINY_OURO_HF_CONFIG).expect("parse the HF config");
    let (architecture, descriptor) =
        ouro_descriptor_from_hf_config(&config, RecurrentConfig::neutral())
            .expect("derive the descriptor");
    let bytes = tiny_ouro_safetensors();
    let manifest = proxima_safetensors::parse_complete(&bytes).expect("parse the safetensors");
    let gguf = tiny_ouro_gguf(2, false);
    let vocab =
        vocab_from_metadata(&parse_complete(&gguf).expect("parse the GGUF")).expect("vocabulary");

    let model = LoadedModel::load_from_safetensors_with_descriptor(
        &manifest,
        &bytes,
        data_start(&bytes),
        architecture,
        vocab,
        &descriptor,
    )
    .expect("load through the descriptor-aware loader");

    let serving =
        ServingConfig::default().with_recurrent(guided(Some(2), LoopCdMode::Disabled, 0.0));
    let logits = model
        .forward_node_values_from_ids(&[0], &[model.logits_root()], &serving)
        .expect("evaluate the HF-loaded graph")
        .remove(0);
    let scale = (2.0f32 / 52.0).sqrt();
    near(&logits, &[3.0 * scale, 4.0 * scale]);
    assert_eq!(model.layer_count(), 2, "the logical depth");
    assert_eq!(model.kv_layers.len(), 2);
    assert!(model.recurrent_readout_roots().is_some());
}

#[test]
fn recurrent_mainline_hf_sibling_graphs_share_one_binding() {
    let config = parse_hf_config(TINY_OURO_HF_CONFIG).expect("parse the HF config");
    let (architecture, descriptor) =
        ouro_descriptor_from_hf_config(&config, RecurrentConfig::neutral())
            .expect("derive the descriptor");
    let bytes = tiny_ouro_safetensors();
    let manifest = proxima_safetensors::parse_complete(&bytes).expect("parse the safetensors");
    let gguf = tiny_ouro_gguf(2, false);
    let vocab =
        vocab_from_metadata(&parse_complete(&gguf).expect("parse the GGUF")).expect("vocabulary");
    let deep = LoadedModel::load_from_safetensors_with_descriptor(
        &manifest,
        &bytes,
        data_start(&bytes),
        architecture,
        vocab,
        &descriptor,
    )
    .expect("load");

    let shallow_policy = guided(Some(1), LoopCdMode::Disabled, 0.0);
    let (_, shallow_descriptor) = ouro_descriptor_from_hf_config(&config, shallow_policy)
        .expect("derive the one-pass descriptor");
    let shallow = deep
        .with_forward_descriptor(&shallow_descriptor)
        .expect("a sibling graph over the same weights");

    assert!(
        Arc::ptr_eq(&deep.weights, &shallow.weights),
        "one binding, two graphs"
    );
    let first_pass = {
        let scale = (2.0f32 / 27.0).sqrt();
        [3.0 * scale, 4.0 * scale]
    };
    let logits = shallow
        .forward_node_values_from_ids(
            &[0],
            &[shallow.logits_root()],
            &ServingConfig::default().with_recurrent(shallow_policy),
        )
        .expect("evaluate the sibling")
        .remove(0);
    near(&logits, &first_pass);
    assert_eq!(shallow.layer_count(), 1);

    let mut rebound = shallow_descriptor.clone();
    rebound.layers[0].attention.qk_norm = true;
    let refused = deep.with_forward_descriptor(&rebound);
    assert!(
        matches!(refused, Err(InteropError::InvalidRecurrentConfig { .. })),
        "{:?}",
        refused.err()
    );
}
