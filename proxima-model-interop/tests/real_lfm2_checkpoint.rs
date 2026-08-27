//! Real, on-disk `LiquidAI/LFM2.5-8B-A1B-GGUF` (`LFM2.5-8B-A1B-Q4_K_M.gguf`,
//! 5,155,564,768 bytes, verified), run end to end through
//! [`proxima_model_interop::run_lfm2_prefill`] -- the first time this
//! hybrid conv/attention/MoE forward program (`e1fd2c7`) has ever executed
//! against real weights rather than only passing `shape::infer`.
//! `#[ignore]`d and skips cleanly when the host-local download is absent,
//! same convention as `bind.rs::real_lfm2_hybrid_file` and
//! `real_smollm2_checkpoint.rs`.
//!
//! `~/repos/others/llama.cpp/bin/llama-cli` (version 5761, built
//! 2025-06-26) cannot cross-check this output: it refuses the file outright
//! (`error loading model architecture: unknown model architecture:
//! 'lfm2moe'`) -- LFM2 support landed in `llama.cpp` after this checkout, a
//! real, measured blocker, not a guess (see this crate's own task report
//! for the captured stderr).
//!
//! This module does not close three real correctness gaps
//! (`crate::lfm2`'s own doc names them): QK-norm, expert bias, and
//! sigmoid- vs. softmax-gated MoE routing. Generated text below should be
//! read as evidence of those gaps, not as a claim of a correct forward
//! pass.

#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_model_interop::{Lfm2Architecture, lfm2_architecture_from_metadata, lfm2_forward_values, run_lfm2_prefill};

const MODEL_PATH: &str = "/Users/brianbruggeman/.lmstudio/models/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf";

fn checkpoint_present() -> bool {
    std::path::Path::new(MODEL_PATH).exists()
}

fn load_real_vocab(parsed: &proxima_gguf::pipe::ParsedGguf) -> proxima_tokenizer::Vocab {
    proxima_tokenizer::gguf::vocab_from_metadata(parsed).expect("build a vocab from the real lfm2 gguf metadata")
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
    let parsed = proxima_gguf::pipe::parse_complete(&file_bytes).expect("parse the real lfm2 gguf checkpoint");

    let architecture = lfm2_architecture_from_metadata(&parsed).expect("derive Lfm2Architecture from the real checkpoint");
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

    let attention_layers = architecture.layer_kinds.iter().filter(|kind| matches!(kind, proxima_tensor::spec::LayerKind::Attention)).count();
    let conv_layers = architecture.layer_kinds.len() - attention_layers;
    assert_eq!(attention_layers, 6, "real checkpoint: 6 attention layers (index % 4 == 2)");
    assert_eq!(conv_layers, 18, "real checkpoint: 18 short-convolution layers");
}

/// Binds every real weight and runs one prefill pass over a short prompt --
/// the deliverable this test exists for. Prints the literal generated
/// text; does NOT assert it matches a fixed string, since this module's
/// own doc names three unclosed correctness gaps (QK-norm, expert bias,
/// sigmoid MoE gating) that make the exact continuation an open question,
/// not a known-good oracle answer yet.
#[test]
#[ignore = "depends on a ~5 GB host-local lfm2 gguf checkout outside this repo"]
fn runs_one_real_forward_pass_over_the_real_checkpoint() {
    if !checkpoint_present() {
        eprintln!("skipping: no host-local lfm2 gguf fixture at {MODEL_PATH}");
        return;
    }
    let file_bytes = std::fs::read(MODEL_PATH).expect("read the real lfm2 gguf checkpoint");
    let parsed = proxima_gguf::pipe::parse_complete(&file_bytes).expect("parse the real lfm2 gguf checkpoint");
    let architecture: Lfm2Architecture = lfm2_architecture_from_metadata(&parsed).expect("derive Lfm2Architecture");
    let vocab = load_real_vocab(&parsed);

    let prompt = "The capital of France is";
    let outcome = run_lfm2_prefill(&parsed, &file_bytes, &architecture, &vocab, prompt, 8);

    match outcome {
        Ok((ids, text)) => {
            std::println!("prompt={prompt:?} generated_ids={ids:?} generated_text={text:?}");
            assert!(!ids.is_empty(), "at least the prompt's own ids must be present");
        }
        Err(error) => {
            std::println!("real lfm2 forward failed: {error:?}");
            panic!("real lfm2 forward pass failed: {error}");
        }
    }
}

/// [`lfm2_forward_values`]'s one-shot forward must pick the SAME greedy
/// token [`run_lfm2_prefill`]'s own decode loop picks on its first
/// iteration -- both bind the same weights, build the same program, and
/// evaluate over the same prompt ids; a real regression in either one's
/// argmax, binding, or program build would flip this without needing an
/// external oracle to see it.
#[test]
#[ignore = "depends on a ~5 GB host-local lfm2 gguf checkout outside this repo"]
fn forward_values_argmax_matches_run_lfm2_prefills_first_generated_token() {
    if !checkpoint_present() {
        eprintln!("skipping: no host-local lfm2 gguf fixture at {MODEL_PATH}");
        return;
    }
    let file_bytes = std::fs::read(MODEL_PATH).expect("read the real lfm2 gguf checkpoint");
    let parsed = proxima_gguf::pipe::parse_complete(&file_bytes).expect("parse the real lfm2 gguf checkpoint");
    let architecture: Lfm2Architecture = lfm2_architecture_from_metadata(&parsed).expect("derive Lfm2Architecture");
    let vocab = load_real_vocab(&parsed);

    let prompt = "The capital of France is";
    let add_bos = vocab.add_bos_token().unwrap_or(true);
    let ids =
        proxima_tokenizer::encode_with_bos_eos(prompt, &vocab, add_bos, false).expect("tokenize the real prompt");

    let (logits, extras) =
        lfm2_forward_values(&parsed, &file_bytes, &architecture, &ids, &[]).expect("one-shot lfm2 forward");
    assert!(extras.is_empty(), "no extra node ids were requested");

    let mut best_token = 0u32;
    let mut best_logit = f32::NEG_INFINITY;
    for (token, logit) in logits.iter().enumerate() {
        if *logit > best_logit {
            best_logit = *logit;
            best_token = token as u32;
        }
    }

    let (generated_ids, _text) = run_lfm2_prefill(&parsed, &file_bytes, &architecture, &vocab, prompt, 1)
        .expect("run_lfm2_prefill's own one-token decode");
    let first_generated_token = generated_ids[ids.len()];

    assert_eq!(
        best_token, first_generated_token,
        "lfm2_forward_values's own argmax must match run_lfm2_prefill's first generated token"
    );
}
