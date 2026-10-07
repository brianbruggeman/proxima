//! Real, on-disk `LiquidAI/LFM2.5-8B-A1B-GGUF` (`LFM2.5-8B-A1B-Q4_K_M.gguf`,
//! 5,155,564,768 bytes, verified): the header facts [`proxima_model_interop::ShortConvHparams`]
//! reads, checked against `llama.cpp`'s own metadata dump. Generation against
//! llama.cpp's ids is `llama_parity_lfm2` in `arch_data_baseline.rs`.
//! `#[ignore]`d and skips cleanly when the host-local download is absent,
//! same convention as `bind.rs::real_lfm2_hybrid_file`.

#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_model_interop::short_conv_architecture_from_metadata;

const MODEL_PATH: &str =
    "/Users/brianbruggeman/.lmstudio/models/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf";

fn checkpoint_present() -> bool {
    std::path::Path::new(MODEL_PATH).exists()
}

/// The real checkpoint's own hparams, read once and printed -- the same
/// numbers captured independently via `llama-cli`'s own metadata dump
/// (`block_count=24`, `embedding_length=2048`, `attention.head_count=32`,
/// `expert_count=32`, `expert_used_count=4`, `leading_dense_block_count=2`,
/// `shortconv.l_cache=3`), so a divergence here is caught before ever
/// reaching a forward pass.
#[test]
#[ignore = "depends on a ~5 GB host-local lfm2 gguf checkout outside this repo"]
fn lfm2_architecture_from_metadata_matches_the_real_checkpoints_own_llama_cli_dump() {
    if !checkpoint_present() {
        eprintln!("skipping: no host-local lfm2 gguf fixture at {MODEL_PATH}");
        return;
    }
    let file_bytes = std::fs::read(MODEL_PATH).expect("read the real lfm2 gguf checkpoint");
    let parsed = proxima_gguf::pipe::parse_complete(&file_bytes)
        .expect("parse the real lfm2 gguf checkpoint");

    let architecture = short_conv_architecture_from_metadata(&parsed)
        .expect("derive ShortConvHparams from the real checkpoint");
    std::println!("real_lfm2 architecture={architecture:?}");

    assert_eq!(architecture.block_count, 24);
    assert_eq!(architecture.embedding, 2048);
    assert_eq!(architecture.query_heads, 32);
    assert_eq!(architecture.kv_heads, 8);
    assert_eq!(architecture.head_dim, 64);
    assert_eq!(architecture.expert_count, 32);
    assert_eq!(architecture.expert_used_count, 4);
    assert_eq!(architecture.expert_feed_forward, 1792);
    assert_eq!(architecture.leading_dense_block_count, 2);
    assert_eq!(architecture.l_cache, 3);
    assert_eq!(architecture.vocab, 128000);

    let attention_layers = architecture
        .layers
        .iter()
        .filter(|layer| matches!(layer.kind, proxima_tensor::spec::LayerKind::Attention))
        .count();
    let conv_layers = architecture.layers.len() - attention_layers;
    assert_eq!(
        attention_layers, 6,
        "real checkpoint: 6 attention layers (index % 4 == 2)"
    );
    assert_eq!(
        conv_layers, 18,
        "real checkpoint: 18 short-convolution layers"
    );
}
