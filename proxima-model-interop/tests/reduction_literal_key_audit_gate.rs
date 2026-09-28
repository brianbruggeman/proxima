//! `c4-7-reduction-literal.md` AC6, on the real gemma4-E2B decode path: with
//! `PROXIMA_PIPELINE_KEY_AUDIT=1`, every `PIPELINE_CACHE` hit `pipeline_for`
//! sees during a real prompt re-emits its own source and sha256-compares it
//! against the hash recorded at compile time. Run once with
//! `PROXIMA_REDUCTION_LITERAL` unset and once `=1` -- the audited/mismatched
//! counts must show real hits (`audited > 0`, a decode of more than one
//! token always reuses at least one pipeline) and zero disagreements in
//! either arm.
//!
//! Skips (does not fail) when the real blob is not present on this host --
//! same posture as `gemma4_correctness_gate.rs`.

#![cfg(all(
    feature = "metal",
    feature = "instrument",
    feature = "metal-reduction-literal",
    target_os = "macos"
))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::File;

use memmap2::Mmap;
use proxima_gguf::parse_complete;
use proxima_gguf::types::GgmlType;
use proxima_model_interop::{GPU_LAYERS_ALL, LoadedModel, ServingConfig};

const REAL_GEMMA4_E2B_GGUF_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

fn run_one_prompt_and_read_audit_counts(reduction_literal_env: Option<&str>) -> Option<(u64, u64)> {
    let Ok(file) = File::open(REAL_GEMMA4_E2B_GGUF_PATH) else {
        eprintln!("skipping: real gemma4-E2B blob not found at {REAL_GEMMA4_E2B_GGUF_PATH}");
        return None;
    };
    // SAFETY: the checkpoint file is not written or truncated by any other
    // process for the duration of this read-only mapping.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the real gemma4-E2B checkpoint");
    let bytes: &[u8] = &mapping;
    let parsed = parse_complete(bytes).expect("parse the real gemma4-E2B checkpoint header");
    let model = LoadedModel::load(&parsed, bytes).expect("bind the real gemma4-E2B checkpoint");

    let serving_config = ServingConfig {
        gpu_layers: GPU_LAYERS_ALL,
        kv_cache_key_quant: GgmlType::F32,
        kv_cache_value_quant: GgmlType::F32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: 0,
        reasoning_budget: 0,
        ..ServingConfig::default()
    };

    temp_env::with_var("PROXIMA_PIPELINE_KEY_AUDIT", Some("1"), || {
        temp_env::with_var("PROXIMA_REDUCTION_LITERAL", reduction_literal_env, || {
            let (token_ids, _text, _stopped_by_eos) = model
                .generate_with_serving_config("The capital of France is", 16, serving_config)
                .expect("greedy decode succeeds");
            assert!(
                token_ids.len() > 1,
                "degenerate gate: decode produced {} tokens, need more than one to exercise a \
                 pipeline-cache HIT at all",
                token_ids.len()
            );
            Some(omega::pipeline_key_audit_counts())
        })
    })
}

#[proxima::test]
async fn gemma4_pipeline_key_audit_finds_no_mismatches_reduction_literal_unset() {
    let Some((audited, mismatched)) = run_one_prompt_and_read_audit_counts(None) else {
        return;
    };
    eprintln!("PROXIMA_REDUCTION_LITERAL unset: audited={audited} mismatched={mismatched}");
    assert!(
        audited > 0,
        "expected at least one PIPELINE_CACHE hit during a 16-token decode to audit, got 0 -- \
         either the audit hook never fired or the decode thread differs from this test's own"
    );
    assert_eq!(
        mismatched, 0,
        "key-completeness audit found {mismatched} mismatches with PROXIMA_REDUCTION_LITERAL unset"
    );
}

#[proxima::test]
async fn gemma4_pipeline_key_audit_finds_no_mismatches_reduction_literal_on() {
    let Some((audited, mismatched)) = run_one_prompt_and_read_audit_counts(Some("1")) else {
        return;
    };
    eprintln!("PROXIMA_REDUCTION_LITERAL=1: audited={audited} mismatched={mismatched}");
    assert!(
        audited > 0,
        "expected at least one PIPELINE_CACHE hit during a 16-token decode to audit, got 0 -- \
         either the audit hook never fired or the decode thread differs from this test's own"
    );
    assert_eq!(
        mismatched, 0,
        "key-completeness audit found {mismatched} mismatches with PROXIMA_REDUCTION_LITERAL=1"
    );
}

/// `S/nb/prefill/RESULTS.md`'s toggle: `=decode` narrows admission to
/// single-token ops only, but a greedy decode is single-token per step by
/// construction, so this arm's audited/mismatched counts should read the
/// same as the `=1` arm above -- the toggle changes prefill's admission,
/// not decode's.
#[proxima::test]
async fn gemma4_pipeline_key_audit_finds_no_mismatches_reduction_literal_decode() {
    let Some((audited, mismatched)) = run_one_prompt_and_read_audit_counts(Some("decode")) else {
        return;
    };
    eprintln!("PROXIMA_REDUCTION_LITERAL=decode: audited={audited} mismatched={mismatched}");
    assert!(
        audited > 0,
        "expected at least one PIPELINE_CACHE hit during a 16-token decode to audit, got 0 -- \
         either the audit hook never fired or the decode thread differs from this test's own"
    );
    assert_eq!(
        mismatched, 0,
        "key-completeness audit found {mismatched} mismatches with PROXIMA_REDUCTION_LITERAL=decode"
    );
}
