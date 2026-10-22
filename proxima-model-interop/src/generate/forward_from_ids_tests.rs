// workspace denies expect_used; a failed fixture must abort the test with its message
#![allow(clippy::expect_used)]

use alloc::vec::Vec;
use core::ops::ControlFlow;

use proxima_tensor::DType;
use proxima_tensor::LoopCdMode;
use proxima_tokenizer::Vocab;

use super::{BackendRuntime, LoadedModel, LogitsSink, NodeValuesSink};
use crate::bind::ModelHparams;
use crate::error::InteropError;
use crate::recurrent_config::RecurrentConfig;
use crate::serving::ServingConfig;

fn byte_vocab() -> Vocab {
    let tokens: Vec<String> = (0..=255u8).map(|byte| format!("<0x{byte:02X}>")).collect();
    Vocab::new(tokens, &[], None, None, None).expect("a byte-level vocabulary builds")
}

fn dense_architecture() -> ModelHparams {
    ModelHparams {
        vocab: 2,
        embedding: 2,
        feed_forward: 2,
        query_heads: 1,
        kv_heads: 1,
        kv_heads_by_layer: vec![1],
        head_dim: 2,
        block_count: 1,
        expert_count: 0,
        expert_used_count: 0,
        rope_freq_base: 10_000.0,
        rms_epsilon: 1.0e-5,
        tied_embeddings: false,
        family: "llama".into(),
        sliding_rope: None,
    }
}

fn dense_safetensors() -> Vec<u8> {
    let tensors = [
        ("model.embed_tokens.weight", [3.0f32, 4.0, 0.0, 5.0]),
        (
            "model.layers.0.input_layernorm.weight",
            [1.0, 1.0, 0.0, 0.0],
        ),
        (
            "model.layers.0.post_attention_layernorm.weight",
            [1.0, 1.0, 0.0, 0.0],
        ),
        ("model.layers.0.self_attn.q_proj.weight", [0.0; 4]),
        (
            "model.layers.0.self_attn.k_proj.weight",
            [1.0, 0.0, 0.0, 1.0],
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            [1.0, 0.0, 0.0, 1.0],
        ),
        ("model.layers.0.self_attn.o_proj.weight", [0.0; 4]),
        ("model.layers.0.mlp.gate_proj.weight", [0.0; 4]),
        ("model.layers.0.mlp.up_proj.weight", [0.0; 4]),
        ("model.layers.0.mlp.down_proj.weight", [0.0; 4]),
        ("model.norm.weight", [1.0, 1.0, 0.0, 0.0]),
        ("lm_head.weight", [1.0, 0.0, 0.0, 1.0]),
    ];
    let encoded: Vec<(&str, Vec<u8>)> = tensors
        .iter()
        .map(|(name, values)| {
            let width = if name.ends_with("layernorm.weight") || name.ends_with("norm.weight") {
                2
            } else {
                4
            };
            let bytes = values[..width]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            (*name, bytes)
        })
        .collect();
    let payloads = encoded
        .iter()
        .map(|(name, bytes)| proxima_safetensors::TensorPayload {
            name: (*name).into(),
            dtype: DType::FLOAT32,
            shape: vec![(bytes.len() / 4) as u64],
            data: bytes,
        })
        .collect();
    proxima_safetensors::write_complete(&proxima_safetensors::SafetensorsModel {
        tensors: payloads,
        metadata: Default::default(),
    })
    .expect("write the tiny dense safetensors fixture")
}

fn with_dense_model<Output>(run: impl FnOnce(&LoadedModel<'_>) -> Output) -> Output {
    let bytes = dense_safetensors();
    let manifest = proxima_safetensors::parse_complete(&bytes).expect("parse the fixture");
    let header_length = u64::from_le_bytes(bytes[..8].try_into().expect("eight header bytes"));
    let model = LoadedModel::load_from_safetensors(
        &manifest,
        &bytes,
        header_length + 8,
        dense_architecture(),
        byte_vocab(),
    )
    .expect("load the tiny dense model");
    run(&model)
}

fn guided(iterations: Option<u32>, mode: LoopCdMode, strength: f32) -> RecurrentConfig {
    RecurrentConfig {
        iterations,
        mode,
        strength,
        reference_iteration: 1,
    }
}

#[test]
fn an_ordinary_model_forwards_the_ids_it_is_given() {
    with_dense_model(|model| {
        let rows = model
            .forward_node_values_from_ids(
                &[0, 1],
                &[model.logits_root()],
                &ServingConfig::default(),
            )
            .expect("an ordinary model admits the default policy");

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].len(), 2, "one vocabulary row for the last position");
        assert!(model.recurrent_readout_roots().is_none());
        assert!(model.recurrent_config().is_neutral());
    });
}

#[test]
fn an_ordinary_model_refuses_any_recurrence_before_evaluating() {
    with_dense_model(|model| {
        for requested in [
            guided(Some(2), LoopCdMode::Disabled, 0.0),
            guided(None, LoopCdMode::FixedLogits, 0.5),
            guided(None, LoopCdMode::Disabled, 0.25),
        ] {
            let config = ServingConfig::default().with_recurrent(requested);

            let forward = model.forward_node_values_from_ids(&[0], &[model.logits_root()], &config);
            let generated = model.generate_from_ids(&[0], 1, &config, &mut |_| {
                core::ops::ControlFlow::Continue(())
            });

            assert!(
                matches!(
                    forward,
                    Err(InteropError::UnsupportedRecurrentConfig { .. })
                ),
                "{requested:?}: {forward:?}"
            );
            assert!(
                matches!(
                    generated,
                    Err(InteropError::UnsupportedRecurrentConfig { .. })
                ),
                "{requested:?}: {generated:?}"
            );
        }
    });
}

#[test]
fn a_forward_needs_tokens_the_vocabulary_has() {
    with_dense_model(|model| {
        let config = ServingConfig::default();
        let nodes = [model.logits_root()];

        let empty = model.forward_node_values_from_ids(&[], &nodes, &config);
        let outside = model.forward_node_values_from_ids(&[0, 7], &nodes, &config);

        assert!(
            matches!(empty, Err(InteropError::EmptyForwardTokens)),
            "{empty:?}"
        );
        assert!(
            matches!(
                outside,
                Err(InteropError::InvalidForwardToken {
                    position: 1,
                    token: 7,
                    vocab: 2
                })
            ),
            "{outside:?}"
        );
    });
}

#[test]
fn teacher_forced_rows_equal_a_full_forward_over_each_forced_prefix() {
    with_dense_model(|model| {
        let config = ServingConfig::default();
        let prompt = [0u32, 1];
        let forced = [1u32, 0, 1];

        let rows = model
            .teacher_forced_logits_from_ids(&prompt, &forced, &config)
            .expect("the tiny model teacher-forces three tokens");

        assert_eq!(rows.len(), forced.len(), "one row per forced token");
        for (step, row) in rows.iter().enumerate() {
            let prefix: Vec<u32> = prompt.iter().chain(&forced[..step]).copied().collect();
            let expected = model
                .forward_node_values_from_ids(&prefix, &[model.logits_root()], &config)
                .expect("a full forward over the prefix")
                .remove(0);
            assert_eq!(row.len(), expected.len(), "step {step} row width");
            for (got, want) in row.iter().zip(&expected) {
                assert!((got - want).abs() < 1.0e-5, "step {step}: {got} vs {want}");
            }
        }
    });
}

#[test]
fn teacher_forcing_refuses_recurrence_an_ordinary_model_cannot_run() {
    with_dense_model(|model| {
        let config =
            ServingConfig::default().with_recurrent(guided(Some(2), LoopCdMode::Disabled, 0.0));

        let rows = model.teacher_forced_logits_from_ids(&[0], &[1], &config);

        assert!(
            matches!(rows, Err(InteropError::UnsupportedRecurrentConfig { .. })),
            "{rows:?}"
        );
    });
}

#[test]
fn an_empty_prompt_is_refused_before_any_prefill() {
    with_dense_model(|model| {
        let config = ServingConfig::default();
        let mut runtime = BackendRuntime::new(&config);

        let outcome = model.run_decode_loop_observed_seeded(
            "",
            4,
            &config,
            &mut runtime,
            None,
            &mut LogitsSink::Discard,
            &mut NodeValuesSink::Discard,
            &mut |_event| ControlFlow::Continue(()),
            None,
            false,
            None,
            None,
        );

        assert!(
            matches!(outcome, Err(InteropError::EmptyForwardTokens)),
            "an empty prompt must be refused with EmptyForwardTokens, got {:?}",
            outcome.as_ref().map(|_| ())
        );
    });
}
