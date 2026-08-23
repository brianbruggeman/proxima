//! The capability matrix this crate never had: every `GgmlType` the GGUF
//! parser knows, every phase (prefill: a multi-token first forward,
//! `new_count > 1`; decode: a single-token continuation against a
//! non-empty KV cache), CPU and Metal, dense and MoE -- run against a
//! synthetic-but-complete checkpoint built in-process
//! (`support::checkpoint_bytes`), never a 4.1 GiB host-local file. A cell
//! that cannot pass yet is still written and `#[ignore]`d with the exact
//! missing piece named, so `cargo nextest run --ignored` is the honest
//! to-do list, and a silent gap can never again read as green.
//!
//! Every test drives [`proxima_model_interop::LoadedModel`]'s public
//! `Pipe` surface end to end -- parse -> load -> generate -- the same path
//! `bind.rs`'s own `real_openchat_file` module exercises against a real
//! checkpoint, just through a fixture instead of a host-local mmap.

#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod support;

use proxima_gguf::GgmlType;
use proxima_model_interop::{InteropError, LoadedModel};
use proxima_primitives::pipe::Pipe;

/// A real English sentence, not a byte stub -- every one of its bytes
/// round-trips through the fixture's byte-level BPE vocab
/// ([`support::checkpoint_bytes`]'s own doc), the same prompt shape
/// `bind.rs`'s own real-checkpoint tests use.
const PROMPT: &str = "The capital of France is";

/// One token id per step of an unbounded budget is never produced: the
/// generic invariant `run_decode_loop`'s own doc encodes, verified through
/// the public `Pipe` boundary rather than the crate-private loop.
fn assert_respects_budget_and_range(ids: &[u32], text: &str, stopped_by_eos: bool, max_tokens: usize) {
    assert!(ids.len() <= max_tokens, "must never exceed the requested token budget: {ids:?}");
    if stopped_by_eos {
        assert!(ids.len() < max_tokens, "an eos stop must produce strictly fewer ids than the full budget");
    } else {
        assert_eq!(ids.len(), max_tokens, "budget exhaustion must produce exactly one id per step");
    }
    for &id in ids {
        assert!(id < support::VOCAB, "token id {id} must be inside the fixture's own {}-token vocab", support::VOCAB);
    }
    if !ids.is_empty() {
        assert!(!text.is_empty(), "a non-empty id sequence must decode to non-empty text");
    }
}

async fn run_cpu(codec: GgmlType, max_tokens: usize) -> Result<(Vec<u32>, String, bool), InteropError> {
    let file_bytes = support::checkpoint_bytes(codec);
    let parsed = proxima_gguf::parse_complete(&file_bytes).expect("parses the synthetic checkpoint");
    let model = LoadedModel::load(&parsed, &file_bytes).expect("loads the synthetic checkpoint");
    Pipe::call(&model, (PROMPT.to_string(), max_tokens)).await
}

// --- dense, CPU: every codec this crate's bind path can actually run ---

#[proxima::test]
#[case::prefill_only(1)]
#[case::prefill_then_decode(2)]
async fn dense_cpu_f32_forward_produces_a_deterministic_token_sequence(#[case] max_tokens: usize) {
    let (ids, text, stopped_by_eos) = run_cpu(GgmlType::F32, max_tokens).await.expect("f32 forward runs on cpu");
    assert_respects_budget_and_range(&ids, &text, stopped_by_eos, max_tokens);
    // real-value determinism, not just "did not panic": the fixture's LCG
    // weight data is fixed, so greedy decode against it must reproduce the
    // exact same ids on every run -- captured from a real run of this test
    // (see this crate's task report for the perturb-and-fail proof).
    let expected: &[u32] = if max_tokens == 1 { &[0] } else { &[0, 0] };
    assert_eq!(ids, expected, "greedy decode must reproduce the fixture's own deterministic output");
}

#[proxima::test]
#[case::prefill_only(1)]
#[case::prefill_then_decode(2)]
async fn dense_cpu_q8_0_forward_produces_a_deterministic_token_sequence(#[case] max_tokens: usize) {
    let (ids, text, stopped_by_eos) = run_cpu(GgmlType::Q8_0, max_tokens).await.expect("q8_0 forward runs on cpu");
    assert_respects_budget_and_range(&ids, &text, stopped_by_eos, max_tokens);
    let expected: &[u32] = if max_tokens == 1 { &[0] } else { &[0, 0] };
    assert_eq!(ids, expected, "greedy decode must reproduce the fixture's own deterministic output");
}

#[proxima::test]
#[case::prefill_only(1)]
#[case::prefill_then_decode(2)]
async fn dense_cpu_q4_k_forward_produces_a_deterministic_token_sequence(#[case] max_tokens: usize) {
    let (ids, text, stopped_by_eos) = run_cpu(GgmlType::Q4_K, max_tokens).await.expect("q4_k forward runs on cpu");
    assert_respects_budget_and_range(&ids, &text, stopped_by_eos, max_tokens);
    let expected: &[u32] = if max_tokens == 1 { &[0] } else { &[0, 0] };
    assert_eq!(ids, expected, "greedy decode must reproduce the fixture's own deterministic output");
}

#[proxima::test]
#[case::prefill_only(1)]
#[case::prefill_then_decode(2)]
async fn dense_cpu_q5_k_forward_produces_a_deterministic_token_sequence(#[case] max_tokens: usize) {
    let (ids, text, stopped_by_eos) = run_cpu(GgmlType::Q5_K, max_tokens).await.expect("q5_k forward runs on cpu");
    assert_respects_budget_and_range(&ids, &text, stopped_by_eos, max_tokens);
    let expected: &[u32] = if max_tokens == 1 { &[0] } else { &[0, 0] };
    assert_eq!(ids, expected, "greedy decode must reproduce the fixture's own deterministic output");
}

#[proxima::test]
#[case::prefill_only(1)]
#[case::prefill_then_decode(2)]
async fn dense_cpu_q6_k_forward_produces_a_deterministic_token_sequence(#[case] max_tokens: usize) {
    let (ids, text, stopped_by_eos) = run_cpu(GgmlType::Q6_K, max_tokens).await.expect("q6_k forward runs on cpu");
    assert_respects_budget_and_range(&ids, &text, stopped_by_eos, max_tokens);
    let expected: &[u32] = if max_tokens == 1 { &[0] } else { &[0, 0] };
    assert_eq!(ids, expected, "greedy decode must reproduce the fixture's own deterministic output");
}

// --- the codecs this crate's bind path cannot run at all ---

/// [`GgmlType::Q4_0`]/[`GgmlType::Q5_0`]/[`GgmlType::Q2_K`]/[`GgmlType::Q3_K`]
/// have no encoder OR decoder anywhere in `proxima_gguf::quant` (grepped:
/// only `q4_k`/`q5_k`/`q6_k`/`q8_0` exist as modules) -- `bind::gguf_tensor_as_f32`'s
/// own `match` names every one of them as `InteropError::UnrepresentableGgmlType`
/// rather than misreading a codec it has no decoder for. `bind::bind_dense`/
/// `bind::bind_matmul_weight`/`bind::bind_all_weights` now propagate that
/// `Err` with `?` instead of `.unwrap_or_else(|error| panic!(...))`, so
/// `LoadedModel::load` returns it through its own documented `Result`
/// rather than aborting the process on untrusted input.
#[proxima::test]
#[case::q4_0(GgmlType::Q4_0)]
#[case::q5_0(GgmlType::Q5_0)]
#[case::q2_k(GgmlType::Q2_K)]
#[case::q3_k(GgmlType::Q3_K)]
async fn dense_cpu_unrepresentable_codec_load_returns_a_typed_error(#[case] codec: GgmlType) {
    let file_bytes = support::checkpoint_bytes(codec);
    let parsed = proxima_gguf::parse_complete(&file_bytes).expect("parses the synthetic checkpoint");
    let outcome = LoadedModel::load(&parsed, &file_bytes);
    let error = outcome.err().expect("loading an unrepresentable codec must return Err, not Ok");
    assert!(
        matches!(error, InteropError::UnrepresentableGgmlType { ggml_type, .. } if ggml_type == codec),
        "a {codec:?} checkpoint's load must fail with UnrepresentableGgmlType naming {codec:?}: {error:?}"
    );
}

#[proxima::test]
#[ignore = "no Q4_0 codec in proxima_gguf::quant (only q4_k/q5_k/q6_k/q8_0 exist); \
            bind::gguf_tensor_as_f32 rejects Q4_0 with UnrepresentableGgmlType before a forward pass can run"]
async fn dense_cpu_q4_0_forward_prefill_and_decode() {
    let (_ids, _text, _stopped) = run_cpu(GgmlType::Q4_0, 2).await.expect("q4_0 forward runs on cpu");
}

#[proxima::test]
#[ignore = "no Q5_0 codec in proxima_gguf::quant (only q4_k/q5_k/q6_k/q8_0 exist); \
            bind::gguf_tensor_as_f32 rejects Q5_0 with UnrepresentableGgmlType before a forward pass can run"]
async fn dense_cpu_q5_0_forward_prefill_and_decode() {
    let (_ids, _text, _stopped) = run_cpu(GgmlType::Q5_0, 2).await.expect("q5_0 forward runs on cpu");
}

#[proxima::test]
#[ignore = "no Q2_K codec in proxima_gguf::quant (only q4_k/q5_k/q6_k/q8_0 exist); \
            bind::gguf_tensor_as_f32 rejects Q2_K with UnrepresentableGgmlType before a forward pass can run"]
async fn dense_cpu_q2_k_forward_prefill_and_decode() {
    let (_ids, _text, _stopped) = run_cpu(GgmlType::Q2_K, 2).await.expect("q2_k forward runs on cpu");
}

#[proxima::test]
#[ignore = "no Q3_K codec in proxima_gguf::quant (only q4_k/q5_k/q6_k/q8_0 exist); \
            bind::gguf_tensor_as_f32 rejects Q3_K with UnrepresentableGgmlType before a forward pass can run"]
async fn dense_cpu_q3_k_forward_prefill_and_decode() {
    let (_ids, _text, _stopped) = run_cpu(GgmlType::Q3_K, 2).await.expect("q3_k forward runs on cpu");
}

// --- float width: this crate's forward path is f32-only ---

#[proxima::test]
#[ignore = "proxima_tensor::cpu::evaluate_quantized_named_with_scratch (the evaluator LoadedModel::call \
            drives) is f32-only: reject_non_float32 (proxima-tensor/src/cpu.rs) rejects any non-Float32 \
            elementwise node outright, and mistral_cached_forward_program has no f16-typed variant; a \
            non-float32 program must instead route through evaluate_typed, which generate.rs never calls"]
async fn dense_cpu_f16_activation_forward_prefill_and_decode() {
    let (_ids, _text, _stopped) = run_cpu(GgmlType::F16, 2).await.expect("f16-activation forward runs on cpu");
}

// --- architecture: MoE has no support at all ---

#[proxima::test]
#[ignore = "bind_all_weights (proxima-model-interop/src/bind.rs) now binds an expert_count > 0 \
            checkpoint's routed FFN weights (blk.N.ffn_gate_inp.weight, blk.N.{ffn_gate,ffn_up,ffn_down}_exps.weight, \
            via bind_moe_expert_weights/proxima_gguf::restack::discover_experts) -- but LoadedModel::load \
            still calls the OLD mistral_cached_forward_program(vocab, embedding, feed_forward, query_heads, \
            kv_heads, head_dim, block_count) with no expert_count/expert_used_count parameters at all, so \
            the forward program it builds only ever references blk.N.ffn_{gate,up,down}.weight -- names an \
            expert_count > 0 bind never produces. proxima_tensor::spec already has the routed building \
            blocks (append_moe_ffn/append_mistral_moe_layer), just not wired into the one entry point \
            generate.rs calls; a checkpoint with experts loads, but its first forward call would fail on \
            a missing named input, not run a real MoE forward"]
async fn moe_architecture_cpu_forward_prefill_and_decode() {
    unimplemented!("mistral_cached_forward_program has no expert-routed variant reachable from LoadedModel::load yet");
}

// --- backend: Metal parity against the same fixture, same codecs CPU runs ---

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal_backend {
    use proxima_gguf::GgmlType;
    use proxima_model_interop::{GPU_LAYERS_ALL, LoadedModel, ServingConfig};

    use super::{PROMPT, assert_respects_budget_and_range};

    /// The one [`ServingConfig`] shape `apply_serving_config` fully
    /// accepts today (`generate.rs`'s own `supported_serving_config`,
    /// private to that module) -- `ServingConfig::default()` carries the
    /// owner's real `-ctk q8_0 -ctv q8_0` invocation, which
    /// `apply_serving_config` correctly rejects (the cached-attention
    /// reduce's Q8_0 read path is not wired end to end; see that error's
    /// own message), so both backends must be compared under the same
    /// *supported* config -- `gpu_layers` is the only field that differs
    /// between the two.
    fn serving_config(gpu_layers: i32) -> ServingConfig<'static> {
        ServingConfig {
            kv_cache_key_quant: GgmlType::F32,
            kv_cache_value_quant: GgmlType::F32,
            flash_attention: false,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers,
            reasoning_budget: 0,
            ..ServingConfig::default()
        }
    }

    /// Runs the same fixture on both backends and asserts the Metal path
    /// reproduces the CPU path's own greedy ids exactly -- the real parity
    /// gate this workspace already uses at the tensor-graph layer
    /// (`omega/tests/metal_real_forward.rs`), lifted to the loader boundary.
    async fn assert_metal_matches_cpu(codec: GgmlType, max_tokens: usize) {
        let file_bytes = super::support::checkpoint_bytes(codec);
        let parsed = proxima_gguf::parse_complete(&file_bytes).expect("parses the synthetic checkpoint");
        let model = LoadedModel::load(&parsed, &file_bytes).expect("loads the synthetic checkpoint");

        let cpu = model
            .generate_with_serving_config(PROMPT, max_tokens, serving_config(0))
            .expect("cpu forward runs");
        let metal = model
            .generate_with_serving_config(PROMPT, max_tokens, serving_config(GPU_LAYERS_ALL))
            .expect("metal forward runs on a real device");

        assert_respects_budget_and_range(&metal.0, &metal.1, metal.2, max_tokens);
        assert_eq!(metal.0, cpu.0, "metal must reproduce the cpu's own greedy token ids for {codec:?}");
        assert_eq!(metal.1, cpu.1, "metal must reproduce the cpu's own decoded text for {codec:?}");
    }

    #[proxima::test]
    #[case::prefill_only(1)]
    #[case::prefill_then_decode(2)]
    async fn dense_metal_f32_matches_cpu(#[case] max_tokens: usize) {
        assert_metal_matches_cpu(GgmlType::F32, max_tokens).await;
    }

    #[proxima::test]
    #[case::prefill_only(1)]
    #[case::prefill_then_decode(2)]
    async fn dense_metal_q8_0_matches_cpu(#[case] max_tokens: usize) {
        assert_metal_matches_cpu(GgmlType::Q8_0, max_tokens).await;
    }

    #[proxima::test]
    #[case::prefill_only(1)]
    #[case::prefill_then_decode(2)]
    async fn dense_metal_q4_k_matches_cpu(#[case] max_tokens: usize) {
        assert_metal_matches_cpu(GgmlType::Q4_K, max_tokens).await;
    }

    #[proxima::test]
    #[case::prefill_only(1)]
    #[case::prefill_then_decode(2)]
    async fn dense_metal_q5_k_matches_cpu(#[case] max_tokens: usize) {
        assert_metal_matches_cpu(GgmlType::Q5_K, max_tokens).await;
    }

    #[proxima::test]
    #[case::prefill_only(1)]
    #[case::prefill_then_decode(2)]
    async fn dense_metal_q6_k_matches_cpu(#[case] max_tokens: usize) {
        assert_metal_matches_cpu(GgmlType::Q6_K, max_tokens).await;
    }
}
