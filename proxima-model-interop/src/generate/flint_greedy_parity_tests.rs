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
    load_flint_with(false)
}

/// `arm_verify` sets the descriptor's `speculative_verify`, which flint's family profile leaves
/// off (only gemma4's verify step was measured to pay), so the verify program is lowered at all.
fn load_flint_with(arm_verify: bool) -> LoadedModel<'static> {
    let path = flint_path();
    crate::test_support::require_fixture(&path, Some("PROXIMA_FLINT_GGUF"));
    let file = File::open(&path).expect("open the real flint checkpoint");
    // SAFETY: read-only mapping of a file nothing else writes during the test.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the real flint checkpoint");
    // the model borrows the mapping for the whole test; leaking it keeps the
    // borrow alive without a self-referential struct.
    let bytes: &'static [u8] = Box::leak(Box::new(mapping));
    let parsed = parse_complete(bytes).expect("parse the real flint header");
    if arm_verify {
        let decode = crate::header_descriptor(&parsed, crate::KvLayout::SlidingRing)
            .expect("read the real flint header descriptor");
        let armed = proxima_tensor::spec::ModelDescriptor {
            speculative_verify: true,
            ..decode
        };
        return LoadedModel::load_with_descriptor(&parsed, bytes, &armed)
            .expect("bind the real flint checkpoint with verify armed");
    }
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

/// Speculation must never change greedy output. Both arms pin a binary16 KV cache
/// explicitly (flint's default), so the speculative arm's verify steps attend several
/// query rows against that cache through the row-tiled kernel; `verify_steps > 0` is the
/// counter proving the verify path ran rather than every step falling back to single-row
/// decode. The ids must equal the speculation-off run and llama.cpp's captured ids.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn speculative_decoding_with_a_binary16_kv_cache_keeps_flint_greedy_ids() {
    use core::ops::ControlFlow;

    use proxima_numeric::NumericFormat;

    use crate::SpeculativeDecodeStats;
    use crate::serving::{SpeculativeConfig, SpeculativeTypeSet};

    // the Gettysburg Address opening quoted twice, the second quote cut mid-sentence: the last
    // 12 prompt tokens then recur earlier, so ngram-simple (size_n 12, size_m 48) drafts the
    // rest of the sentence with no forced width.
    const OPENING: &str = "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.";
    let prompt = format!(
        "{OPENING} Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure.\n\nQuoted again: Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the"
    );
    const MAX_TOKENS: usize = 24;
    let model = load_flint_with(true);
    assert!(
        model.speculative_verify_program.is_some() || model.recurrent_speculative_verify.is_some(),
        "flint armed no verify program, so no step can verify"
    );
    assert!(
        crate::kv_cache_format::row_tiled_reads(
            &NumericFormat::Float16,
            &NumericFormat::Float16,
            u64::from(model.architecture.head_dim),
        ),
        "the row-tiled verify kernel must read a binary16 cache"
    );
    let arm = |speculative: SpeculativeConfig<'static>| ServingConfig {
        kv_cache_key_quant: NumericFormat::Float16,
        kv_cache_value_quant: NumericFormat::Float16,
        speculative,
        ..config(crate::serving::GPU_LAYERS_ALL)
    };

    let off = arm(SpeculativeConfig::none());
    assert_eq!(off.speculative.speculative_types, SpeculativeTypeSet::empty());
    let (off_ids, off_text, _) = model
        .generate_with_serving_config(&prompt, MAX_TOKENS, off)
        .expect("speculation off with a binary16 cache must run on Metal");

    let mut stats = SpeculativeDecodeStats::default();
    let (on_ids, on_text, _) = model
        .generate_streaming_with_speculative_stats(
            &prompt,
            MAX_TOKENS,
            arm(SpeculativeConfig::ngram_simple()),
            &mut |_| ControlFlow::Continue(()),
            &mut stats,
            None,
        )
        .expect("ngram-simple speculation with a binary16 cache must run on Metal");

    eprintln!("speculative stats: {stats:?}");
    assert!(
        stats.verify_steps > 0 && stats.drafted_total > 0,
        "the speculative arm never ran a verify step, so the row-tiled path was not exercised: {stats:?}"
    );
    assert_eq!(
        first_id_divergence(&off_ids, &on_ids),
        None,
        "speculation changed greedy ids (off {off_ids:?}, on {on_ids:?}, off text {off_text:?}, on text {on_text:?})"
    );
}
