#![allow(clippy::expect_used)]

use core::ops::ControlFlow;

#[cfg(feature = "instrument")]
use omega::metal::PIPELINE_MISSES;
use proxima_gguf::parse_complete;
use proxima_numeric::NumericFormat;

use super::chunked_prefill_tests::dense_checkpoint;
use super::resident_plans::{PLAN_BUILDS, PLAN_REFITS};
use super::{LoadedModel, ServingConfig};
use crate::serving::{GPU_LAYERS_ALL, PromptCacheConfig};

const CHUNK_ROWS: usize = 24;
const PROMPT_ROWS: usize = 2 * CHUNK_ROWS - 5;
const NEW_TOKENS: usize = 6;

fn config() -> ServingConfig<'static> {
    ServingConfig {
        gpu_layers: GPU_LAYERS_ALL,
        kv_cache_key_quant: NumericFormat::Float32,
        kv_cache_value_quant: NumericFormat::Float32,
        flash_attention: false,
        batch_size: 0,
        ubatch_size: CHUNK_ROWS as u32,
        reasoning_budget: 0,
        temperature: 0.0,
        prompt_cache: PromptCacheConfig::off(),
        ..ServingConfig::default()
    }
}

fn prompt() -> Vec<u32> {
    (0..PROMPT_ROWS).map(|row| ((row * 11 + 5) % 250 + 1) as u32).collect()
}

fn request(model: &LoadedModel<'_>) -> Vec<u32> {
    request_of(model, &prompt())
}

fn request_of(model: &LoadedModel<'_>, ids: &[u32]) -> Vec<u32> {
    model
        .generate_from_ids(ids, NEW_TOKENS, &config(), &mut |_| ControlFlow::Continue(()))
        .expect("request on the tiny dense checkpoint")
        .0
}

#[test]
fn a_request_after_warm_prefill_matches_the_same_request_on_a_fresh_model() {
    let bytes = dense_checkpoint();
    let parsed = parse_complete(&bytes).expect("parse the dense fixture");
    let fresh = LoadedModel::load(&parsed, &bytes).expect("load a fresh model");
    let warmed = LoadedModel::load(&parsed, &bytes).expect("load the warmed model");

    let rows = warmed.warm_prefill(&config(), CHUNK_ROWS).expect("warm prefill");
    let expected = request(&fresh);
    let actual = request(&warmed);

    assert_eq!(rows, 2 * CHUNK_ROWS, "the throwaway prompt is two full chunks, so its last chunk is chunk-width");
    assert_eq!(expected.len(), NEW_TOKENS, "the reference request generated every token");
    assert_eq!(actual, expected, "warm_prefill changed a later request's token ids");
}

#[cfg(feature = "instrument")]
#[test]
fn warm_prefill_leaves_the_first_request_no_pipeline_to_compile() {
    let bytes = dense_checkpoint();
    let parsed = parse_complete(&bytes).expect("parse the dense fixture");
    let model = LoadedModel::load(&parsed, &bytes).expect("load the model");

    let (rows, warm_misses) = PIPELINE_MISSES.scoped(|| model.warm_prefill(&config(), CHUNK_ROWS).expect("warm prefill"));
    let (ids, request_misses) = PIPELINE_MISSES.scoped(|| request(&model));

    assert_eq!(rows, 2 * CHUNK_ROWS);
    assert_eq!(ids.len(), NEW_TOKENS);
    assert_eq!(request_misses, 0, "the request compiled a pipeline after warm_prefill (warm compiled {warm_misses})");
}

#[test]
fn warm_prefill_stores_no_prompt_cache_entry() {
    let bytes = dense_checkpoint();
    let parsed = parse_complete(&bytes).expect("parse the dense fixture");
    let model = LoadedModel::load(&parsed, &bytes).expect("load the model");
    let cached = ServingConfig { prompt_cache: PromptCacheConfig::default(), ..config() };

    model.warm_prefill(&cached, CHUNK_ROWS).expect("warm prefill");

    assert_eq!(model.prompt_cache.lock().entry_count(), 0, "warm_prefill left a prompt-cache entry");
}

fn footprint(model: &LoadedModel<'_>) -> super::resident_plans::ResidentFootprint {
    proxima_primitives::driver::block_on(model.serving.lock()).footprint()
}

#[test]
fn warm_prefill_leaves_no_more_resident_than_one_real_request_does() {
    let bytes = dense_checkpoint();
    let parsed = parse_complete(&bytes).expect("parse the dense fixture");
    let warmed = LoadedModel::load(&parsed, &bytes).expect("load the warmed model");
    let fresh = LoadedModel::load(&parsed, &bytes).expect("load a fresh model");

    warmed.warm_prefill(&config(), CHUNK_ROWS).expect("warm prefill");
    let after_warm = footprint(&warmed);
    let two_full_chunks: Vec<u32> = (0..2 * CHUNK_ROWS).map(|row| ((row * 11 + 5) % 250 + 1) as u32).collect();
    request_of(&fresh, &two_full_chunks);
    let after_request = footprint(&fresh);
    eprintln!("FOOTPRINT after_warm={after_warm:?}");
    eprintln!("FOOTPRINT after_one_request={after_request:?}");

    assert!(
        after_warm.wide_bytes() <= after_request.wide_bytes(),
        "warm_prefill kept {} wide-plan bytes, one real request of two full chunks keeps {}",
        after_warm.wide_bytes(),
        after_request.wide_bytes()
    );
    assert!(
        after_warm.wide.iter().all(|(new_count, _, _)| *new_count == CHUNK_ROWS),
        "warm_prefill kept a wide plan no request of this chunking refits: {:?}",
        after_warm.wide
    );
    assert_eq!(after_warm.kv_buffers, 0, "warm_prefill kept device KV buffers sized for its short context");
}

#[test]
fn the_first_request_after_warm_prefill_refits_the_warmed_last_chunk_plan_to_its_tail() {
    let bytes = dense_checkpoint();
    let parsed = parse_complete(&bytes).expect("parse the dense fixture");
    let warmed = LoadedModel::load(&parsed, &bytes).expect("load the warmed model");
    let fresh = LoadedModel::load(&parsed, &bytes).expect("load a fresh model");

    warmed.warm_prefill(&config(), CHUNK_ROWS).expect("warm prefill");
    request(&warmed);
    request(&fresh);

    let keys = |model: &LoadedModel<'_>| footprint(model).wide.iter().map(|(new_count, outputs, _)| (*new_count, *outputs)).collect::<Vec<_>>();
    assert_eq!(keys(&warmed), keys(&fresh), "after one request the warmed model holds the plans a fresh model holds, with no plan left from the warm-up");
}

#[test]
fn a_short_tail_after_warm_prefill_refits_the_warmed_plan_and_builds_none() {
    let bytes = dense_checkpoint();
    let parsed = parse_complete(&bytes).expect("parse the dense fixture");
    let fresh = LoadedModel::load(&parsed, &bytes).expect("load a fresh model");
    let warmed = LoadedModel::load(&parsed, &bytes).expect("load the warmed model");
    let tail_prompt: Vec<u32> = (0..3 * CHUNK_ROWS - 5).map(|row| ((row * 13 + 7) % 250 + 1) as u32).collect();
    let run = |model: &LoadedModel<'_>| {
        model
            .generate_from_ids(&tail_prompt, NEW_TOKENS, &config(), &mut |_| ControlFlow::Continue(()))
            .expect("request on the tiny dense checkpoint")
            .0
    };

    warmed.warm_prefill(&config(), CHUNK_ROWS).expect("warm prefill");
    let verdicts = proxima_primitives::driver::block_on(warmed.serving.lock()).wide_template_verdicts();
    assert!(!verdicts.is_empty(), "the warm kept no wide plan");
    assert!(
        verdicts.iter().all(|verdict| verdict.is_some()),
        "a kept wide plan has not learned its row template during the warm: {verdicts:?}"
    );
    let ((actual, builds), refits) = PLAN_REFITS.scoped(|| PLAN_BUILDS.scoped(|| run(&warmed)));
    let expected = run(&fresh);

    assert_eq!(builds, 0, "the request built a plan after warm_prefill");
    assert!(refits >= 1, "the request refit no warmed plan to its {}-row tail", CHUNK_ROWS - 5);
    assert_eq!(actual, expected, "refitting the warmed last-chunk plan changed the request's token ids");
}

fn arenas(model: &LoadedModel<'_>) -> Vec<(usize, usize, Vec<usize>, Option<usize>)> {
    proxima_primitives::driver::block_on(model.serving.lock()).wide_arenas()
}

#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
/// Records `phys_footprint` and the wide plans' arena slots around each warm phase, the way
/// `warm_prefill` runs them (generate, trim, prime), then one short-tail request. Read with
/// `--nocapture`; it asserts only that each phase ran.
#[test]
fn warm_phases_footprint_is_recorded_for_plans_templates_and_a_tail_request() {
    let bytes = dense_checkpoint();
    let parsed = parse_complete(&bytes).expect("parse the dense fixture");
    let model = LoadedModel::load(&parsed, &bytes).expect("load the model");
    let mut warm_config = config();
    warm_config.ubatch_size = CHUNK_ROWS as u32;
    let ids: Vec<u32> = (0..2 * CHUNK_ROWS).map(|row| ((row * 11 + 5) % 250 + 1) as u32).collect();
    let tail_prompt: Vec<u32> = (0..3 * CHUNK_ROWS - 5).map(|row| ((row * 13 + 7) % 250 + 1) as u32).collect();

    let before_plans = super::load_model::phys_footprint_bytes();
    model.generate_from_ids(&ids, 2, &warm_config, &mut |_| ControlFlow::Continue(())).expect("warm generate");
    let after_plans = super::load_model::phys_footprint_bytes();
    measure(format!("MEASURE plans_built before={before_plans} after={after_plans} delta={}", after_plans as i64 - before_plans as i64));
    measure(format!("MEASURE arenas_after_warm_generate={:?}", arenas(&model)));

    let before_trim = super::load_model::phys_footprint_bytes();
    let primed = proxima_primitives::driver::block_on(model.serving.lock()).trim_warm_residue(CHUNK_ROWS).expect("trim and prime");
    let after_prime = super::load_model::phys_footprint_bytes();
    measure(format!("MEASURE trim_and_prime primed={primed} before={before_trim} after={after_prime} delta={}", after_prime as i64 - before_trim as i64));
    measure(format!("MEASURE arenas_after_trim={:?}", arenas(&model)));

    let before_request = super::load_model::phys_footprint_bytes();
    let (out, ..) = model
        .generate_from_ids(&tail_prompt, NEW_TOKENS, &config(), &mut |_| ControlFlow::Continue(()))
        .expect("tail request");
    let after_request = super::load_model::phys_footprint_bytes();
    measure(format!("MEASURE tail_request before={before_request} after={after_request} delta={}", after_request as i64 - before_request as i64));
    measure(format!("MEASURE arenas_after_request={:?}", arenas(&model)));
    assert_eq!(out.len(), NEW_TOKENS);
    assert!(primed >= 1, "no wide plan learned its template");
}

#[cfg(all(feature = "instrument", feature = "metal", target_os = "macos"))]
fn measure(line: String) {
    eprintln!("{line}");
    if let Some(path) = std::env::var_os("PROXIMA_MEASURE_OUT")
        && let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path)
    {
        use std::io::Write;
        let _ = writeln!(file, "{line}");
    }
}

#[test]
fn warm_prefill_leaves_every_kept_wide_plan_on_one_transient_buffer() {
    let bytes = dense_checkpoint();
    let parsed = parse_complete(&bytes).expect("parse the dense fixture");
    let model = LoadedModel::load(&parsed, &bytes).expect("load the model");

    model.warm_prefill(&config(), CHUNK_ROWS).expect("warm prefill");

    let arenas = arenas(&model);
    assert!(arenas.len() >= 2, "the warm kept fewer than two wide plans: {arenas:?}");
    let first = arenas[0].3;
    assert!(first.is_some(), "a kept wide plan leases no transient slot");
    assert!(arenas.iter().all(|arena| arena.3 == first), "kept wide plans hold distinct slot-0 buffers: {arenas:?}");
}
