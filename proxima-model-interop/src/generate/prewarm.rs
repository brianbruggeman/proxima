//! Anticipatory prefill (`proxima-tensor/specs/prefix-cache-reuse/SPEC.md`
//! R10): prefill token ids into the prompt cache before a request needs them.
//!
//! [`LoadedModel::prewarm`] is the caller's entry: a system prompt at load,
//! retrieved documents while a tool call runs, a user's partial input as they
//! type. The end of a generation is the built-in point
//! ([`LoadedModel::set_prewarm_suffix`]): the answer is complete and the
//! device idle while the user reads, so the answer's trailing tokens plus the
//! registered turn-boundary suffix are prefilled with no further call.
//!
//! There is no second matching path. A prewarm takes its entry out through
//! the same [`PromptCache::take_best`] a request uses, extends or rewinds it
//! by the same longest common prefix and checkpoints it by the same
//! [`planned_positions`] -- the prefill itself is
//! [`LoadedModel::prefill_through_stops`], the loop a request's own prefill
//! runs, with chunk boundaries added as extra stops. It then stores the entry
//! back, and the next request's ordinary lookup finds it.
//!
//! Preemption is caller-driven, not a background executor: proxima has none
//! here, and the request entry points are synchronous. `prewarm` runs on the
//! caller's thread, and at every chunk boundary asks [`PrewarmGate`] whether a
//! request is pending; if one is, the rows prefilled so far are stored (usable
//! up to what was prefilled) and `prewarm` returns early with
//! [`PrewarmReport::preempted`] set. A request that arrives mid-chunk waits
//! for that chunk only. Reach for a plain [`LoadedModel::generate_from_ids`]
//! when the prefix is needed now; reach for `prewarm` when the device is idle
//! and the prefix is likely.

use core::ops::ControlFlow;
use std::sync::PoisonError;
use std::time::{Duration, Instant};

use proxima_telemetry::{debug, warn};

use super::prompt_cache::{CacheEntry, log_entry_dropped};
use super::*;

/// Why a prewarm did nothing at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrewarmSkip {
    /// The prompt cache is off, so there is nowhere to put rows.
    CacheOff,
    /// No ids to prefill.
    NoIds,
    /// Another prewarm holds the device.
    Busy,
}

/// What one prewarm did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrewarmReport {
    /// Rows the cache already held for these ids, which the prewarm kept.
    pub reused_tokens: usize,
    /// Tokens this prewarm prefilled.
    pub prefilled_tokens: usize,
    /// Chunk boundaries it reached.
    pub chunks: usize,
    /// Whether a pending request ended it before every id was prefilled.
    pub preempted: bool,
    /// The longest time any one chunk took, the bound on how long a request
    /// arriving mid-prewarm waits.
    pub longest_chunk: Duration,
    /// Why nothing was attempted, when nothing was.
    pub skipped: Option<PrewarmSkip>,
}

impl PrewarmReport {
    const fn idle(skipped: Option<PrewarmSkip>, preempted: bool) -> Self {
        Self {
            reused_tokens: 0,
            prefilled_tokens: 0,
            chunks: 0,
            preempted,
            longest_chunk: Duration::ZERO,
            skipped,
        }
    }
}

/// The positions a prewarm stops at while it prefills `held..target`,
/// ascending: every `chunk_tokens` from `held` (none when `chunk_tokens` is
/// `0`), every checkpoint position in the range, and `target` itself.
pub(super) fn prewarm_stops(
    held: usize,
    target: usize,
    chunk_tokens: usize,
    checkpoints: &[usize],
) -> Vec<usize> {
    let boundaries = (chunk_tokens > 0)
        .then(|| (held + chunk_tokens..target).step_by(chunk_tokens))
        .into_iter()
        .flatten();
    let inside = checkpoints
        .iter()
        .copied()
        .filter(|position| *position > held && *position < target);
    let mut stops: Vec<usize> = boundaries
        .chain(inside)
        .chain(core::iter::once(target))
        .collect();
    stops.sort_unstable();
    stops.dedup();
    stops
}

impl LoadedModel<'_> {
    /// Registers the turn-boundary suffix -- the token ids that follow an
    /// answer in the next request's prompt, for gemma4 its end-of-turn token
    /// and the next user turn's opener -- so that after every generation
    /// through the prompt cache proxima prefills the answer's trailing tokens
    /// plus this suffix with no further call. An empty suffix turns the
    /// trigger off, which is its default. It is a setter and not a
    /// [`crate::PromptCacheConfig`] field because that config is `Copy` and a
    /// token list is not.
    pub fn set_prewarm_suffix(&self, suffix: &[u32]) {
        self.prompt_cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .set_prewarm_suffix(suffix);
    }

    /// Prefills `ids` into the prompt cache without generating, at low
    /// priority: in [`crate::PromptCacheConfig::prewarm_chunk_tokens`] chunks,
    /// yielding to a pending request at every chunk boundary. See the module
    /// doc for how it composes the request path.
    ///
    /// # Errors
    ///
    /// Same as [`Self::generate_with_serving_config`]; a failed forward drops
    /// the entry it was extending, as a failed request does.
    pub fn prewarm(
        &self,
        ids: &[u32],
        serving_config: &ServingConfig,
    ) -> Result<PrewarmReport, InteropError> {
        self.prewarm_with_progress(ids, serving_config, &mut |_position| {})
    }

    /// [`Self::prewarm`], calling `on_chunk` with the position reached at
    /// every chunk boundary before it checks for a waiting request.
    ///
    /// # Errors
    ///
    /// Same as [`Self::prewarm`].
    pub fn prewarm_with_progress(
        &self,
        ids: &[u32],
        serving_config: &ServingConfig,
        on_chunk: &mut dyn FnMut(usize),
    ) -> Result<PrewarmReport, InteropError> {
        let effective = self.effective_serving_config(serving_config)?;
        let mut runtime = BackendRuntime::new(&effective);
        self.prewarm_ids(ids, &effective, &mut runtime, None, on_chunk)
    }

    #[allow(clippy::too_many_lines)] // one flow: gate, lookup, chunked prefill, store, report
    pub(super) fn prewarm_ids(
        &self,
        ids: &[u32],
        serving_config: &ServingConfig,
        runtime: &mut BackendRuntime,
        forced_draft_width: Option<u16>,
        on_chunk: &mut dyn FnMut(usize),
    ) -> Result<PrewarmReport, InteropError> {
        let config = serving_config.prompt_cache;
        if !config.is_enabled() {
            return Ok(PrewarmReport::idle(Some(PrewarmSkip::CacheOff), false));
        }
        if ids.is_empty() {
            return Ok(PrewarmReport::idle(Some(PrewarmSkip::NoIds), false));
        }
        apply_serving_config(serving_config, ids.len())?;
        let Some(_slot) = self.prewarm_gate.try_begin() else {
            return Ok(PrewarmReport::idle(Some(PrewarmSkip::Busy), false));
        };
        if self.prewarm_gate.request_waiting() {
            return Ok(PrewarmReport::idle(None, true));
        }
        let (_, widths) = self.declared_layer_cache_names_and_widths()?;
        let key = self.cache_key(serving_config, runtime, forced_draft_width);
        let (found, _) = self
            .prompt_cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take_for_prewarm(ids, &key, &widths);
        let entry = found.unwrap_or_else(|| CacheEntry::empty(key));
        let held = entry.state.cached_len;
        let checkpoints =
            planned_positions(&entry.checkpoint_positions(), held, ids.len(), &[], &config);
        let stops = prewarm_stops(
            held,
            ids.len(),
            config.prewarm_chunk_tokens as usize,
            &checkpoints,
        );
        let mut chunk_started = Instant::now();
        let mut longest_chunk = Duration::ZERO;
        let mut chunks = 0_usize;
        let mut preempted = false;
        let mut entry = self
            .prefill_through_stops(
                ids,
                entry,
                &stops,
                &checkpoints,
                &widths,
                &config,
                serving_config,
                runtime,
                forced_draft_width,
                &mut |position| {
                    longest_chunk = longest_chunk.max(chunk_started.elapsed());
                    chunks += 1;
                    on_chunk(position);
                    chunk_started = Instant::now();
                    if position < ids.len() && self.prewarm_gate.request_waiting() {
                        preempted = true;
                        return ControlFlow::Break(());
                    }
                    ControlFlow::Continue(())
                },
            )
            .inspect_err(log_entry_dropped)?;
        let reached = entry.state.cached_len;
        entry.mark_prewarmed(held, reached);
        self.prompt_cache_store(entry, &config);
        let report = PrewarmReport {
            reused_tokens: held,
            prefilled_tokens: reached - held,
            chunks,
            preempted,
            longest_chunk,
            skipped: None,
        };
        debug!(
            prewarm_tokens = report.prefilled_tokens as u64,
            prewarm_preempted = report.preempted,
            prewarm_reused_tokens = held as u64,
            prewarm_chunks = chunks as u64,
            prewarm_longest_chunk_ns = u64::try_from(longest_chunk.as_nanos()).unwrap_or(u64::MAX),
            "prewarm finished"
        );
        Ok(report)
    }

    /// The end-of-answer trigger: prefills `ids` + `generated` + the
    /// registered suffix into the entry the request just stored. Does nothing
    /// without a suffix. The request has already succeeded, so a prewarm that
    /// fails is logged, not returned.
    pub(super) fn prewarm_after_answer(
        &self,
        ids: &[u32],
        generated: &[u32],
        serving_config: &ServingConfig,
        runtime: &mut BackendRuntime,
        forced_draft_width: Option<u16>,
    ) {
        let suffix = self
            .prompt_cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .prewarm_suffix()
            .to_vec();
        if suffix.is_empty() {
            return;
        }
        let next_prefix: Vec<u32> = ids
            .iter()
            .chain(generated)
            .chain(&suffix)
            .copied()
            .collect();
        if let Err(error) = self.prewarm_ids(
            &next_prefix,
            serving_config,
            runtime,
            forced_draft_width,
            &mut |_position| {},
        ) {
            warn!(
                prewarm_error = %error,
                "end-of-answer prewarm failed after the request succeeded"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stops_are_every_chunk_then_the_target() {
        assert_eq!(prewarm_stops(0, 10, 4, &[]), vec![4, 8, 10]);
    }

    #[test]
    fn chunks_count_from_the_rows_already_held() {
        assert_eq!(prewarm_stops(5, 14, 4, &[]), vec![9, 13, 14]);
    }

    #[test]
    fn a_zero_chunk_is_one_stop_at_the_target() {
        assert_eq!(prewarm_stops(3, 900, 0, &[]), vec![900]);
    }

    #[test]
    fn checkpoints_inside_the_range_become_stops_without_duplicating_a_boundary() {
        assert_eq!(prewarm_stops(0, 12, 4, &[4, 6, 12, 0]), vec![4, 6, 8, 12]);
    }

    #[test]
    fn a_chunk_larger_than_the_work_is_one_stop() {
        assert_eq!(prewarm_stops(10, 20, 256, &[]), vec![20]);
    }
}
