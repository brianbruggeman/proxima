//! What a configuration cannot do. Each edit below is a config a person could write; each is
//! refused, and the refusal names the compiled primitive the config would need first. No
//! checkpoint is read.
//!
//! Companion to `proxima-tensor/docs/what-configuration-cannot-do.md`.
//!
//! Usage:
//! `cargo run -p proxima-model-interop --example model_config_limits --features std`

#![allow(clippy::expect_used)]

use std::path::Path;

use proxima_tensor::TensorError;
use proxima_tensor::spec::{CacheMask, LayerAttentionConfig, LayerKind, LayerSchedule, ModelDescriptor, RopePairing, build_forward};

fn granite() -> ModelDescriptor {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/model-configs/granite_moe.toml");
    conflaguration::from_file(&path).expect("the granite config loads")
}

fn with_layers(base: ModelDescriptor, edit: impl Fn(&LayerSchedule) -> LayerSchedule) -> ModelDescriptor {
    let layers = base.layers.iter().map(edit).collect();
    ModelDescriptor { layers, ..base }
}

fn refusal_of(descriptor: &ModelDescriptor) -> String {
    let refusal = build_forward(descriptor).expect_err("the lowering refuses this config");
    assert!(matches!(refusal, TensorError::UnsupportedInBuilder { .. }), "got {refusal:?}");
    refusal.to_string()
}

fn main() {
    let split_half = with_layers(granite(), |layer| LayerSchedule {
        attention: LayerAttentionConfig { rope_pairing: RopePairing::SplitHalf { pairs: layer.attention.head_dim / 2 }, ..layer.attention.clone() },
        ..layer.clone()
    });
    println!("split-half rope on the moe layer: {}", refusal_of(&split_half));

    let padded = ModelDescriptor { cache_mask: CacheMask::Padded, ..granite() };
    let padded_refusal = refusal_of(&padded);
    assert!(padded_refusal.contains("a logit scale"), "{padded_refusal}");
    println!("granite's logit scale on the padded cache engine: {padded_refusal}");

    let no_scales = ModelDescriptor { logit_scale: None, residual_scale: None, ..padded };
    let ops = build_forward(&no_scales).expect("without the scales the padded engine lowers it").program.len();
    assert!(ops > 0, "lowering produced zero ops");
    println!("the same config minus the two scales lowers: {ops} ops");

    let gdn = with_layers(granite(), |layer| LayerSchedule { kind: LayerKind::Gdn, ..layer.clone() });
    let gdn_refusal = build_forward(&gdn).expect_err("the recurrent engine refuses a routed granite schedule");
    println!("every layer kind = Gdn on the granite moe schedule: {gdn_refusal:?}");
    assert!(
        matches!(gdn_refusal, TensorError::UnsupportedInBuilder { builder: "build_forward(hybrid dense)", feature: "routed experts" }),
        "got {gdn_refusal:?}"
    );

    let text = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/model-configs/granite_moe.toml"))
        .expect("reads the granite config");
    let unknown_field = text.replace("vocab = 49155", "vocab = 49155\nsliding_window_size = 4096");
    let field_error = toml::from_str::<ModelDescriptor>(&unknown_field).expect_err("a key no compiled field reads is refused");
    println!("unknown key: {field_error}");

    let unknown_variant = text.replace("activation = \"Silu\"", "activation = \"Mish\"");
    let variant_error = toml::from_str::<ModelDescriptor>(&unknown_variant).expect_err("an activation no compiled variant names is refused");
    println!("unknown activation: {variant_error}");
}
