#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::error::InteropError;

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

/// Prefault, a second immediate prefault and `MADV_WILLNEED` applied
/// in sequence to one mapping, each timed and `mincore`-checked, so the rung
/// that resolves a hole pattern is visible instead of guessed.
///
/// # Errors
/// The prefault pool failing to build or a chunk failing to report.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn residency_ladder(bytes: &[u8]) -> Result<Vec<LadderStage>, InteropError> {
    use crate::mapping_residency::{count_missing_pages, missing_byte_ranges, probe_residency};

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

    let mut stages = Vec::with_capacity(3);
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
    Ok(stages)
}

#[cfg(feature = "test-support")]
pub use crate::generate::prompt_cache_probes::{
    LookupCostRow, lookup_cost as prompt_cache_lookup_cost,
};

/// Today's serving profile for `parsed`: the family row, then the tensor-presence
/// and metadata-key overlays and the header's embedding scale that
/// `crate::dense::descriptor_from_gguf` applies, with the expert count and the
/// effective QK-norm. The differential in `tests/local_checkpoints.rs` compares
/// the checkpoint-inferred profile against it.
///
/// # Errors
/// The family has no profile row or arch config, or the header contradicts it.
#[cfg(feature = "std")]
pub fn incumbent_profile(
    parsed: &proxima_gguf::pipe::ParsedGguf,
    architecture: &str,
) -> Result<(proxima_tensor::spec::FamilyProfile, u32, bool), crate::error::InteropError> {
    use proxima_tensor::spec::{EmbeddingScale, FamilyProfile};

    use crate::bind::{checkpoint_has_qk_norm, metadata_f32_optional, metadata_u32_optional};
    use crate::ffn_inference::with_inferred_ffn;
    use crate::key_inference::with_inferred_keys;
    use crate::profiles::family_profile;

    let experts = metadata_u32_optional(parsed, &format!("{architecture}.expert_count"));
    let profile = with_inferred_ffn(family_profile(architecture)?, parsed, experts);
    let profile = with_inferred_keys(profile, parsed, architecture)?;
    let header_scale =
        metadata_f32_optional(parsed, &format!("{architecture}.embedding_scale"), 0.0);
    let embedding_scale = (header_scale != 0.0)
        .then_some(EmbeddingScale::Factor(header_scale))
        .or(profile.embedding_scale);
    let profile = FamilyProfile {
        embedding_scale,
        ..profile
    };
    let qk_norm = checkpoint_has_qk_norm(parsed) && profile.qk_norm;
    Ok((profile, experts, qk_norm))
}
