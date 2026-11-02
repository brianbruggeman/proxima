use core::ops::ControlFlow;

use crate::error::InteropError;
use crate::generate::{BackendRuntime, LoadedModel, LogitsSink, NodeValuesSink};
use crate::serving::ServingConfig;

/// What one probe decode produced: the greedy ids and text, whether the
/// model's own eos ended it, and how many prompt tokens were prefilled.
pub struct DecodeProbe {
    pub ids: Vec<u32>,
    pub text: String,
    pub stopped_by_eos: bool,
    pub prompt_token_count: usize,
}

/// One cached decode over a fresh [`BackendRuntime`], the loop the public
/// `generate_with_serving_config` runs, with the prefill length returned so
/// a TTFT or op-profile probe can state how long a prompt it actually timed.
///
/// # Errors
/// Whatever the decode loop returns: tokenization, bind, or backend failures.
pub fn decode(
    model: &LoadedModel,
    prompt: &str,
    max_tokens: usize,
    serving_config: &ServingConfig,
) -> Result<DecodeProbe, InteropError> {
    let mut runtime = BackendRuntime::new(serving_config);
    let (ids, text, stopped_by_eos, prefix_state) = model.run_decode_loop_observed_seeded(
        prompt,
        max_tokens,
        serving_config,
        &mut runtime,
        None,
        &mut LogitsSink::Discard,
        &mut NodeValuesSink::Discard,
        &mut |_event| ControlFlow::Continue(()),
        None,
        true,
        None,
        None,
    )?;
    Ok(DecodeProbe {
        ids,
        text,
        stopped_by_eos,
        prompt_token_count: prefix_state.len(),
    })
}

/// [`LoadedModel::run_decode_loop`] over a fresh [`BackendRuntime`], the
/// loop a per-op GPU profile arms through `PROXIMA_METAL_OP_PROFILE_STEP`.
///
/// # Errors
/// Whatever the decode loop returns: tokenization, bind, or backend failures.
pub fn decode_plain(
    model: &LoadedModel,
    prompt: &str,
    max_tokens: usize,
    serving_config: &ServingConfig,
) -> Result<(Vec<u32>, String, bool), InteropError> {
    let mut runtime = BackendRuntime::new(serving_config);
    model.run_decode_loop(prompt, max_tokens, serving_config, &mut runtime)
}

/// One rung of [`residency_ladder`]: how long the rung took and which pages
/// of the mapping `mincore` still reports missing afterwards.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub struct LadderStage {
    pub label: &'static str,
    pub elapsed: core::time::Duration,
    pub missing_pages: usize,
    pub total_pages: usize,
    pub contiguous_runs: usize,
    pub first_missing_offset: Option<usize>,
    pub last_missing_offset: Option<usize>,
    pub note: Option<String>,
}

/// Prefault, a second immediate prefault, `MADV_WILLNEED` and `mlock` applied
/// in sequence to one mapping, each timed and `mincore`-checked, so the rung
/// that resolves a hole pattern is visible instead of guessed.
///
/// # Errors
/// The prefault pool failing to build or a chunk failing to report.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn residency_ladder(bytes: &[u8]) -> Result<Vec<LadderStage>, InteropError> {
    use crate::mapping_residency::{
        count_missing_pages, lock_resident, missing_byte_ranges, probe_residency,
    };

    let stage = |label: &'static str, started: std::time::Instant, note: Option<String>| {
        let residency = probe_residency(bytes);
        let ranges = missing_byte_ranges(&residency, omega::metal::page_size(), bytes.len());
        LadderStage {
            label,
            elapsed: started.elapsed(),
            missing_pages: count_missing_pages(&residency),
            total_pages: residency.len(),
            contiguous_runs: ranges.len(),
            first_missing_offset: ranges.first().map(|range| range.0),
            last_missing_offset: ranges.last().map(|range| range.1),
            note,
        }
    };

    let mut stages = Vec::with_capacity(4);
    let started = std::time::Instant::now();
    crate::loader::prefault(bytes)?;
    stages.push(stage("stage 1: prefault", started, None));

    let started = std::time::Instant::now();
    crate::loader::prefault(bytes)?;
    stages.push(stage("stage 2: second immediate prefault", started, None));

    let started = std::time::Instant::now();
    // SAFETY: `bytes` is a live borrow for the duration of this call;
    // `madvise` only advises the kernel and never writes through the pointer.
    let advised = unsafe {
        rustix::mm::madvise(
            bytes.as_ptr().cast::<core::ffi::c_void>().cast_mut(),
            bytes.len(),
            rustix::mm::Advice::WillNeed,
        )
    };
    let advise_failure = advised
        .err()
        .map(|error| format!("madvise(WILLNEED) failed: {error}"));
    stages.push(stage("stage 3: madvise(WILLNEED)", started, advise_failure));

    let started = std::time::Instant::now();
    lock_resident(bytes);
    stages.push(stage("stage 4: mlock", started, None));
    Ok(stages)
}

#[cfg(feature = "test-support")]
pub use crate::generate::prompt_cache_probes::{
    LookupCostRow, lookup_cost as prompt_cache_lookup_cost,
};
