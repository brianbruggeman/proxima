//! End-to-end greedy parity for the flint checkpoint: `LoadedModel::load` plus a
//! temperature-0 generation must reproduce the ids `llama_ids.json` captured
//! from llama.cpp, on the CPU interpreter and on Metal.
//!
//! This is slice 12 of `proxima-tensor/specs/serve-flint-through-omega`, and it
//! is the only test in this spec that exercises the whole stack at once: the
//! header readers, the sliding-pattern descriptor, the routed/shared-expert
//! lowering, the attention sinks, the per-layer rope pattern, the `IQ4_XS` and
//! `IQ3_S` codecs, and the expert slab. Every earlier slice pins one mechanism
//! against a hand-worked literal or a captured array; a mistake that survives
//! all of them shows up here as a first divergent id.
//!
//! The oracle is committed, not recomputed: `llama_ids.json` is llama.cpp
//! b12110's own greedy run over the same prompt, and its bytes are pinned by
//! `oracle_fixtures`' digests. `flint_oracle_control_ids_pass_the_parity_comparator`
//! is the control that the comparator can report a divergence at all.
//!
//! Requires the real 23 GiB checkpoint: `PROXIMA_FLINT_GGUF`, defaulting to the
//! path the capture used.

#![allow(clippy::expect_used)]

use std::fs::File;

use memmap2::Mmap;
use proxima_gguf::parse_complete;

use crate::LoadedModel;
use crate::oracle_fixtures::{captured_greedy_run, first_id_divergence};
use crate::serving::ServingConfig;

/// The two knobs these arms set are `gpu_layers` and the KV dtype; everything
/// else is the production default, so the run exercises the same program the
/// serving path lowers.
fn config(gpu_layers: i32) -> ServingConfig<'static> {
    ServingConfig {
        gpu_layers,
        ..ServingConfig::default()
    }
}

fn flint_path() -> String {
    std::env::var("PROXIMA_FLINT_GGUF").unwrap_or_else(|_| {
        String::from("/Users/brianbruggeman/models/llm/flint-gguf/flint-UD-Q4_K_XL.gguf")
    })
}

/// Loads the real checkpoint, or panics naming the environment variable that
/// points at it. Never skipped: an arm that silently does not run cannot
/// discharge the acceptance criterion it exists for.
fn load_flint() -> LoadedModel<'static> {
    let path = flint_path();
    crate::test_support::require_fixture(&path, Some("PROXIMA_FLINT_GGUF"));
    let file = File::open(&path).expect("open the real flint checkpoint");
    // SAFETY: read-only mapping of a file nothing else writes during the test.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the real flint checkpoint");
    // the model borrows the mapping for the whole test; leaking it keeps the
    // borrow alive without a self-referential struct.
    let bytes: &'static [u8] = Box::leak(Box::new(mapping));
    let parsed = parse_complete(bytes).expect("parse the real flint header");
    LoadedModel::load(&parsed, bytes).expect("bind the real flint checkpoint")
}

/// The whole of AC13's CPU half: the committed llama.cpp ids come back from a
/// `gpu_layers = 0` forward, in order, with no arm skipped.
#[test]
fn flint_greedy_parity_cpu_reproduces_llama_cpps_captured_ids() {
    let oracle = captured_greedy_run();
    let model = load_flint();
    let (token_ids, text, _stopped) = model
        .generate_with_serving_config(&oracle.prompt, oracle.generated_ids.len(), config(0))
        .expect("the CPU interpreter must run the flint program");

    assert_eq!(
        first_id_divergence(&oracle.generated_ids, &token_ids),
        None,
        "cpu: first divergent id from llama.cpp {} (ours {token_ids:?}, theirs {:?}, text {text:?})",
        oracle.llama_build,
        oracle.generated_ids
    );
}

/// AC13's Metal half. `gpu_layers = ALL` on a `metal`-featured build selects the
/// Metal engine; a host without a device fails here with `MetalError::NoDevice`
/// rather than silently running the CPU arm twice.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn flint_greedy_parity_metal_reproduces_llama_cpps_captured_ids() {
    let oracle = captured_greedy_run();
    let model = load_flint();
    let (token_ids, text, _stopped) = model
        .generate_with_serving_config(
            &oracle.prompt,
            oracle.generated_ids.len(),
            config(crate::serving::GPU_LAYERS_ALL),
        )
        .expect("the Metal engine must run the flint program");

    assert_eq!(
        first_id_divergence(&oracle.generated_ids, &token_ids),
        None,
        "metal: first divergent id from llama.cpp {} (ours {token_ids:?}, theirs {:?}, text {text:?})",
        oracle.llama_build,
        oracle.generated_ids
    );
}
