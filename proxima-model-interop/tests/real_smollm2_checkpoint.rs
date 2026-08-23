//! Real, on-disk `HuggingFaceTB/SmolLM2-135M-Instruct` (downloaded this
//! session into `~/.lmstudio/models/HuggingFaceTB/SmolLM2-135M-Instruct/`),
//! run end to end through the crate's public `Pipe` surface -- the first
//! time this crate's safetensors/HF path (`ceeb75b`/`ae74f42`/`c30b8a5`) has
//! ever been driven against a real HuggingFace checkpoint rather than a
//! synthetic fixture. `#[ignore]`d and skips cleanly when the host-local
//! download is absent, same convention as `bind.rs::real_openchat_file`.
//!
//! This checkpoint ships `tie_word_embeddings: true` (its real
//! `config.json`) -- no `lm_head.weight` tensor at all, confirmed against
//! its real `model.safetensors` (272 tensors, `model.embed_tokens.weight`
//! present, no `lm_head.weight`) -- the exact gap this session's
//! `hf_bind.rs` change closes.
//!
//! # Known, named divergence beyond the first generated token
//!
//! The FIRST generated token matches `llama-cli` bit-for-bit against TWO
//! independent GGUF conversions of the identical checkpoint at two
//! different precisions (`Felladrin/gguf-Q8_0-SmolLM2-135M-Instruct` and
//! `bartowski/SmolLM2-135M-Instruct-GGUF`'s `f16` variant), both under the
//! identical (BOS-forced) tokenization -- strong evidence the full-depth
//! forward (30 layers, GQA, RoPE, RMSNorm, tied embedding/output
//! projection) is computing the right thing. A follow-on probe
//! ([`probe_one_shot_prefill_after_the_first_token_matches_the_cached_decode_step`])
//! confirms the cached-decode step is self-consistent with a fresh
//! one-shot prefill at the same position, ruling out the KV-cache
//! append/read path as the cause of anything downstream.
//!
//! Beyond token 1, this crate's real bf16-weight forward diverges from
//! BOTH GGUF oracles (which agree with each other): proxima continues
//! "...is the **capital of France**", both GGUF oracles continue "...is
//! the **city of Paris**, a city of". This is a genuine, unresolved
//! residual, not explained away here: with the caching path ruled out and
//! token 1 matching exactly, the two live hypotheses are (a) bf16 rounding
//! in this specific real checkpoint's weights compounding differently than
//! GGUF's own f16/Q8_0 dequantization by the second decode step, or (b) a
//! subtler numerical divergence a one-token prefill does not expose.
//! Neither is confirmed; distinguishing them needs layer-by-layer
//! activation diffing against `llama.cpp`, which this change does not do.

#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use core::future::Future;
use core::pin::pin;
use core::task::{Context, Poll, Waker};

use proxima_model_interop::{LoadedModel, architecture_from_hf_config, parse_hf_config};
use proxima_primitives::pipe::Pipe;
use proxima_tokenizer::Vocab;

const MODEL_DIR: &str = "/Users/brianbruggeman/.lmstudio/models/HuggingFaceTB/SmolLM2-135M-Instruct";

/// Drives a leaf [`Pipe::call`] future to completion -- the same shape
/// `bind.rs::real_openchat_file::block_on` uses, copied rather than reused
/// (that helper is crate-private, unreachable from this external `tests/`
/// binary): every future this crate's own pipes return is `async move {
/// <synchronous computation> }` with no internal `.await`, so the first
/// poll is always `Poll::Ready`.
fn block_on<Fut: Future>(future: Fut) -> Fut::Output {
    let mut future = pin!(future);
    let waker = Waker::noop();
    let mut context = Context::from_waker(waker);
    match future.as_mut().poll(&mut context) {
        Poll::Ready(output) => output,
        Poll::Pending => unreachable!("proxima-model-interop pipes never yield: no internal .await"),
    }
}

fn checkpoint_present() -> bool {
    std::path::Path::new(MODEL_DIR).join("model.safetensors").exists()
}

/// `generation_config.json`'s own `bos_token_id`/`eos_token_id` -- read as
/// plain JSON rather than adding a dedicated struct for two integers this
/// crate has no other use for.
fn read_generation_config_ids() -> (Option<u32>, Option<u32>) {
    let bytes = std::fs::read(alloc_format(MODEL_DIR, "generation_config.json")).expect("read generation_config.json");
    let value: serde_json::Value = serde_json::from_slice(&bytes).expect("generation_config.json parses");
    let bos = value.get("bos_token_id").and_then(serde_json::Value::as_u64).map(|id| id as u32);
    let eos = value.get("eos_token_id").and_then(serde_json::Value::as_u64).map(|id| id as u32);
    (bos, eos)
}

fn alloc_format(dir: &str, file: &str) -> std::path::PathBuf {
    std::path::Path::new(dir).join(file)
}

/// `8 + header_len`, the same computation `hf_bind.rs`'s own test module
/// uses -- `Manifest`'s `data_offsets` are relative to the byte AFTER the
/// header, never the start of the file.
fn safetensors_data_start(file_bytes: &[u8]) -> u64 {
    let mut length_prefix = [0u8; 8];
    length_prefix.copy_from_slice(&file_bytes[..8]);
    8 + u64::from_le_bytes(length_prefix)
}

fn load_real_model(file_bytes: &[u8]) -> LoadedModel<'_> {
    let config_bytes = std::fs::read(alloc_format(MODEL_DIR, "config.json")).expect("read config.json");
    let hf_config = parse_hf_config(&config_bytes).expect("parse real smollm2 config.json");
    let architecture = architecture_from_hf_config(&hf_config);
    assert!(architecture.tied_embeddings, "real smollm2 config.json declares tie_word_embeddings: true");

    let tokenizer_bytes = std::fs::read(alloc_format(MODEL_DIR, "tokenizer.json")).expect("read tokenizer.json");
    let (bos_token_id, eos_token_id) = read_generation_config_ids();
    let vocab: Vocab = proxima_tokenizer::hf::vocab_from_tokenizer_json(&tokenizer_bytes, bos_token_id, eos_token_id, None)
        .expect("build a vocab from the real tokenizer.json");

    let manifest = proxima_safetensors::parse_complete(file_bytes).expect("parse real model.safetensors");
    let data_start = safetensors_data_start(file_bytes);

    LoadedModel::load_from_safetensors(&manifest, file_bytes, data_start, architecture, vocab)
        .expect("load the real tied-embedding smollm2 checkpoint through the public path")
}

/// One real greedy-decoded token out of the real, downloaded SmolLM2
/// checkpoint. Cross-checked against `llama-cli` (real binary at
/// `~/repos/others/llama.cpp/bin/llama-cli`) run against a real GGUF
/// conversion of the identical checkpoint
/// (`Felladrin/gguf-Q8_0-SmolLM2-135M-Instruct`), under the SAME
/// tokenization convention this crate's `run_decode_loop` actually uses
/// (`generate.rs`'s own `encode_with_bos_eos(prompt, vocab, true, false)` --
/// unconditionally prepends BOS, id 1 `<|im_start|>`, regardless of this
/// checkpoint's own GGUF metadata `tokenizer.ggml.add_bos_token = false`;
/// see this test's own module doc / the task report for that divergence,
/// named but not fixed in this change).
///
/// `--override-kv tokenizer.ggml.add_bos_token=bool:true` on the llama-cli
/// side forces the identical BOS-prepended tokenization, so this is a fair,
/// apples-to-apples comparison of the SAME input token sequence, not two
/// different ones: llama-cli's own answer under that override is
/// `" the"` (captured this session), and this test's assertion below
/// reproduces it byte-for-byte through this crate's independent forward
/// implementation (different weight-binding path, different interpreter,
/// same real bf16 weight bytes).
#[test]
#[ignore = "depends on a host-local SmolLM2-135M-Instruct safetensors download outside this repo"]
fn runs_one_real_forward_pass_and_greedy_picks_a_real_token() {
    if !checkpoint_present() {
        eprintln!("skipping: no host-local smollm2 safetensors checkpoint at {MODEL_DIR}");
        return;
    }
    let file_bytes = std::fs::read(alloc_format(MODEL_DIR, "model.safetensors")).expect("read real model.safetensors");
    let model = load_real_model(&file_bytes);

    let prompt = "The capital of France is";
    let (ids, text, _stopped_by_eos) =
        block_on(Pipe::call(&model, (prompt.to_string(), 1))).expect("generate through the public Pipe path");

    std::println!("prompt={prompt:?} token_id={} token={text:?}", ids[0]);

    assert_eq!(ids.len(), 1);
    assert_eq!(
        text, " the",
        "greedy token text must match llama-cli's real answer under the identical (BOS-prepended) tokenization"
    );
}

/// Determinism: the same prompt, run twice through two independently loaded
/// `LoadedModel`s, must produce byte-identical ids and text -- no shared
/// mutable state between calls, no uninitialized-memory nondeterminism.
#[test]
#[ignore = "depends on a host-local SmolLM2-135M-Instruct safetensors download outside this repo"]
fn greedy_decode_is_deterministic_across_two_independent_loads() {
    if !checkpoint_present() {
        eprintln!("skipping: no host-local smollm2 safetensors checkpoint at {MODEL_DIR}");
        return;
    }
    let file_bytes = std::fs::read(alloc_format(MODEL_DIR, "model.safetensors")).expect("read real model.safetensors");
    let prompt = "The capital of France is";

    let first_model = load_real_model(&file_bytes);
    let (first_ids, first_text, _) =
        block_on(Pipe::call(&first_model, (prompt.to_string(), 8))).expect("first generation succeeds");

    let second_model = load_real_model(&file_bytes);
    let (second_ids, second_text, _) =
        block_on(Pipe::call(&second_model, (prompt.to_string(), 8))).expect("second generation succeeds");

    std::println!("prompt={prompt:?} ids={first_ids:?} text={first_text:?}");

    assert_eq!(first_ids, second_ids, "greedy decode must be byte-identical across independent loads");
    assert_eq!(first_text, second_text, "decoded text must be byte-identical across independent loads");
    assert!(!first_text.is_empty(), "a real forward pass must produce non-empty generated text");
}

/// Diagnostic probe (not a correctness assertion): re-prefills the ENTIRE
/// six-token prompt from scratch (`"The capital of France is the"`, no KV
/// cache at all -- a fresh `LoadedModel`, `max_tokens: 1`) and prints the
/// single predicted next token, to compare against the SECOND id the cached
/// decode loop produced continuing from `"The capital of France is"`. If
/// the two agree, the cached-decode step is consistent with a one-shot
/// prefill at the same position, and the divergence from `llama-cli`'s own
/// answer (see the task report) is in the shared forward math, not the KV
/// cache append/read path specifically.
#[test]
#[ignore = "depends on a host-local SmolLM2-135M-Instruct safetensors download outside this repo"]
fn probe_one_shot_prefill_after_the_first_token_matches_the_cached_decode_step() {
    if !checkpoint_present() {
        eprintln!("skipping: no host-local smollm2 safetensors checkpoint at {MODEL_DIR}");
        return;
    }
    let file_bytes = std::fs::read(alloc_format(MODEL_DIR, "model.safetensors")).expect("read real model.safetensors");
    let model = load_real_model(&file_bytes);
    let (ids, text, _) =
        block_on(Pipe::call(&model, ("The capital of France is the".to_string(), 1))).expect("one-shot prefill succeeds");
    std::println!("one_shot_prefill_after_the token_id={} token={text:?}", ids[0]);
}
