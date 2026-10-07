//! A model is a configuration: load the hand-written granite moe config, expand its
//! `repeat` and `pattern` entries, and lower it to a program. No checkpoint is read.
//!
//! Companion to `proxima-tensor/docs/a-model-is-a-configuration.md`.
//!
//! Usage:
//! `cargo run -p proxima-model-interop --example model_config_expand --features std`

#![allow(clippy::expect_used)]

use std::path::Path;

use proxima_tensor::spec::{ForwardProgram, ModelDescriptor, build_forward};

const LAYER_MARKER: &str = "[[layers]]\nrepeat = 24\nkind = \"Attention\"\n";

fn config_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/model-configs/granite_moe.toml")
}

fn pattern_entry(layer_body: &str, extra_attention_line: &str) -> String {
    let body = layer_body
        .replace("[layers.", "[layers.pattern.")
        .replace("head_dim = 64\n", &format!("head_dim = 64\n{extra_attention_line}"));
    format!("[[layers.pattern]]\nkind = \"Attention\"\n{body}")
}

fn windowed_then_full_text(text: &str) -> String {
    let (header, layer_body) = text.split_once(LAYER_MARKER).expect("the granite config holds one repeated layer entry");
    let windowed = pattern_entry(layer_body, "mask_window = 512\n");
    let full = pattern_entry(layer_body, "");
    format!("{header}[[layers]]\nrepeat = 12\n{windowed}{full}")
}

fn main() {
    let text = std::fs::read_to_string(config_path()).expect("reads the granite config");
    let descriptor: ModelDescriptor = conflaguration::from_file(&config_path()).expect("the config loads");
    conflaguration::Validate::validate(&descriptor).expect("the config is internally consistent");

    assert_eq!(descriptor.layers.len(), 24, "`repeat = 24` expands to one schedule per block");
    assert!(descriptor.layers.iter().all(|layer| *layer == descriptor.layers[0]));
    println!("repeat: 1 entry -> {} layers, all equal", descriptor.layers.len());

    let expanded = toml::to_string(&descriptor).expect("a descriptor serializes");
    let written_entries = expanded.matches("[[layers]]").count();
    assert_eq!(written_entries, 24, "a descriptor serializes expanded, one entry per block");
    let reloaded: ModelDescriptor = toml::from_str(&expanded).expect("the expanded form loads again");
    assert_eq!(reloaded, descriptor, "expanded and repeated forms are the same descriptor");
    println!("round trip: {written_entries} serialized entries reload to the same descriptor");

    let patterned: ModelDescriptor = toml::from_str(&windowed_then_full_text(&text)).expect("the pattern config loads");
    conflaguration::Validate::validate(&patterned).expect("a pattern config holds one entry per block");
    let windows: Vec<Option<u32>> = patterned.layers.iter().map(|layer| layer.attention.mask_window).collect();
    assert_eq!(windows.len(), 24, "repeat 12 of a two-entry pattern is 24 layers");
    assert!(windows.chunks(2).all(|pair| pair == [Some(512), None]), "the pattern alternates in order: {windows:?}");
    println!("pattern: repeat 12 x [windowed, full] -> {} layers, windows {:?}...", windows.len(), &windows[..4]);

    let ForwardProgram { program, .. } = build_forward(&descriptor).expect("the granite config lowers");
    assert!(!program.is_empty(), "lowering produced zero ops");
    println!("lowering: {} layers -> {} ops", descriptor.layers.len(), program.len());
}
