//! Anticipatory prefill (`proxima-tensor/specs/prefix-cache-reuse/SPEC.md`
//! R10): prefill token ids into the prompt cache before a request needs them.
//!
//! [`LoadedModel::prewarm`] is the caller's entry: a system prompt at load,
//! retrieved documents while a tool call runs, a user's partial input as they
//! type. The end of a generation is the built-in point
//! ([`LoadedModel::set_prewarm_suffix`]): the answer is complete and the
//! device idle while the user reads, so the answer's trailing tokens plus the
//! registered turn-boundary suffix are queued ([`super::prewarm_queue`]) as the
//! request returns, and prefilled by [`LoadedModel::with_prewarm_worker`]'s
//! thread or a [`LoadedModel::run_pending_prewarm`] call -- never on the
//! answer's own return path.
//!
//! There is no second matching path. A prewarm takes its entry out through
//! the same [`PromptCache::take_best`] a request uses, extends or rewinds it
//! by the same longest common prefix and checkpoints it by the same
//! [`planned_positions`] -- the prefill itself is
//! [`LoadedModel::prefill_through_stops`], the loop a request's own prefill
//! runs, with chunk boundaries added as extra stops. It then stores the entry
//! back, and the next request's ordinary lookup finds it.
//!
//! Preemption is chunk-driven, not an executor's: proxima has none here, and
//! the request entry points are synchronous. `prewarm` runs on the thread that
//! calls it (the worker's, for a queued one), and at every chunk boundary asks
//! [`PrewarmGate`] whether a request is pending; if one is, the rows prefilled so far are stored (usable
//! up to what was prefilled) and `prewarm` returns early with
//! [`PrewarmReport::preempted`] set. A request that arrives mid-chunk waits
//! for that chunk only. Reach for a plain [`LoadedModel::generate_from_ids`]
//! when the prefix is needed now; reach for `prewarm` when the device is idle
//! and the prefix is likely.

use core::ops::ControlFlow;
use std::time::{Duration, Instant};

use proxima_telemetry::{debug, warn};

use super::prewarm_queue::PrewarmJob;
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
    /// The queued prefix was built under a cache key the prewarming config
    /// does not derive, so no request could reuse its rows.
    ConfigMismatch,
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

/// The prefix the next turn's prompt shares with this one: `ids`, the
/// answer, then `suffix`. The model ends a sliding-pattern answer with its end-of-turn
/// token, and keeps sampling it until the end-of-sequence token (a real answer
/// came back as `[.., 4443, 106, 106, 106, 106]`), while a client re-renders
/// the turn with that token once, as the first token of the suffix. Counting
/// the generated ones as well would put the boundary in twice and diverge from
/// every real next prompt at the second.
fn next_turn_prefix(ids: &[u32], generated: &[u32], suffix: &[u32]) -> Vec<u32> {
    let answer_len = generated
        .iter()
        .rposition(|id| Some(id) != suffix.first())
        .map_or(0, |last| last + 1);
    ids.iter()
        .chain(&generated[..answer_len])
        .chain(suffix)
        .copied()
        .collect()
}

impl LoadedModel<'_> {
    /// Registers the turn-boundary suffix -- the token ids that follow an
    /// answer in the next request's prompt, for sliding-pattern its end-of-turn token
    /// and the next user turn's opener -- so that after every generation
    /// through the prompt cache proxima prefills the answer's trailing tokens
    /// plus this suffix with no further call. An empty suffix turns the
    /// trigger off, which is its default. It is a setter and not a
    /// [`crate::PromptCacheConfig`] field because that config is `Copy` and a
    /// token list is not.
    pub fn set_prewarm_suffix(&self, suffix: &[u32]) {
        self.prompt_cache
            .lock()
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
            .take_for_prewarm(ids, &key, &widths, config.min_similarity_milli);
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

    /// The end-of-answer trigger: queues `ids` + `generated` + the registered
    /// suffix as the prefix the next turn will share, for
    /// [`Self::with_prewarm_worker`] or [`Self::run_pending_prewarm`] to
    /// prefill into the entry the request just stored. Does nothing without a
    /// suffix. It only queues: the answer's caller must not wait for a
    /// prefill nobody asked it to.
    pub(super) fn queue_prewarm_after_answer(
        &self,
        ids: &[u32],
        generated: &[u32],
        key: CacheKey,
        forced_draft_width: Option<u16>,
    ) {
        let suffix = self
            .prompt_cache
            .lock()
            .prewarm_suffix()
            .to_vec();
        if suffix.is_empty() {
            return;
        }
        let prefix = next_turn_prefix(ids, generated, &suffix);
        let queued_tokens = prefix.len() as u64;
        let replaced = self.prewarm_queue.submit(PrewarmJob {
            prefix,
            key,
            forced_draft_width,
        });
        debug!(
            prewarm_queued_tokens = queued_tokens,
            prewarm_replaced_unserved_job = replaced,
            "end-of-answer prewarm queued"
        );
    }

    /// Runs the end-of-answer prewarm a request queued, on the calling
    /// thread, under `serving_config`; `None` when nothing is queued. For a
    /// caller with no worker thread to hand over: call it when the device is
    /// idle, between requests. [`Self::with_prewarm_worker`] is this in a
    /// loop on a thread of its own.
    ///
    /// A queued prefix is skipped ([`PrewarmSkip::ConfigMismatch`]) when
    /// `serving_config` derives another cache key than the request that
    /// queued it ran under: its rows could not be reused.
    ///
    /// # Errors
    ///
    /// Same as [`Self::prewarm`].
    pub fn run_pending_prewarm(
        &self,
        serving_config: &ServingConfig,
    ) -> Result<Option<PrewarmReport>, InteropError> {
        let effective = self.effective_serving_config(serving_config)?;
        let mut runtime = BackendRuntime::new(&effective);
        self.run_queued_prewarm(&effective, &mut runtime)
    }

    fn run_queued_prewarm(
        &self,
        effective: &ServingConfig,
        runtime: &mut BackendRuntime,
    ) -> Result<Option<PrewarmReport>, InteropError> {
        self.prewarm_queue
            .run_next(|job| self.prewarm_queued(&job, effective, runtime))
            .transpose()
    }

    fn prewarm_queued(
        &self,
        job: &PrewarmJob,
        effective: &ServingConfig,
        runtime: &mut BackendRuntime,
    ) -> Result<PrewarmReport, InteropError> {
        let key = self.cache_key(effective, runtime, job.forced_draft_width);
        if key != job.key {
            return Ok(PrewarmReport::idle(
                Some(PrewarmSkip::ConfigMismatch),
                false,
            ));
        }
        let report = self.prewarm_ids(
            &job.prefix,
            effective,
            runtime,
            job.forced_draft_width,
            &mut |_position| {},
        )?;
        if report.skipped.is_none()
            && !report.preempted
            && let Err(error) = self.follow_up_branches(
                &job.prefix,
                effective,
                runtime,
                job.forced_draft_width,
                &mut |_kept| {},
            )
        {
            warn!(
                follow_up_error = %error,
                "follow-up prewarm failed after the request succeeded"
            );
        }
        Ok(report)
    }

    /// Runs `body` with a worker thread prefilling the end-of-answer prewarms
    /// that requests queue, under `serving_config`, and stops the worker when
    /// `body` returns. The worker waits for a queued prefix, runs it behind
    /// `PrewarmGate` so a request arriving mid-prefill waits one chunk at
    /// most, and goes back to waiting; a request itself returns the moment its
    /// answer is stored. The worker keeps one `BackendRuntime` for every
    /// job: a fresh one costs the first forward it runs about 225 ms on
    /// the E2B checkpoint (317 ms against 93 ms for a 5-token prewarm).
    ///
    /// The thread is scoped, not owned by the model: a [`LoadedModel`] borrows
    /// the checkpoint bytes for `'file`, so no thread that outlives the
    /// caller's borrow may hold it. Wrap the code that serves requests, as
    /// `std::thread::scope` is wrapped around any threaded use of the model.
    /// With no worker and no [`Self::run_pending_prewarm`] caller, a queued
    /// prefix is replaced by the next answer's and never prefilled.
    ///
    /// # Errors
    ///
    /// Same as [`Self::generate_with_serving_config`], before `body` runs.
    pub fn with_prewarm_worker<T>(
        &self,
        serving_config: &ServingConfig,
        body: impl FnOnce() -> T,
    ) -> Result<T, InteropError> {
        let effective = self.effective_serving_config(serving_config)?;
        Ok(std::thread::scope(|scope| {
            let (_attached, life) = self.prewarm_queue.attach_worker();
            scope.spawn(|| {
                let _life = life;
                self.serve_queued_prewarms(&effective);
            });
            body()
        }))
    }

    fn serve_queued_prewarms(&self, effective: &ServingConfig) {
        let mut runtime = BackendRuntime::new(effective);
        while self.prewarm_queue.wait_for_work() {
            if let Err(error) = self.run_queued_prewarm(effective, &mut runtime) {
                warn!(
                    prewarm_error = %error,
                    "end-of-answer prewarm failed after the request succeeded"
                );
            }
        }
    }

    /// Blocks until the prewarm now running, and the one queued behind it for
    /// a worker [`Self::with_prewarm_worker`] attached, have finished: the
    /// point where a user who has read the answer would start typing. Returns
    /// at once when nothing is running, and does not wait for a queued prefix
    /// no worker will run.
    pub fn wait_for_prewarm(&self) {
        self.prewarm_queue.wait_idle();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GEMMA_PROMPT: [u32; 4] = [2, 105, 2364, 107];
    const GEMMA_BOUNDARY: [u32; 5] = [106, 107, 105, 2364, 107];

    #[test]
    fn an_answer_that_repeated_the_end_of_turn_token_keeps_it_once_in_the_prefix() {
        let generated = [4443, 106, 106, 106, 106];

        let prefix = next_turn_prefix(&GEMMA_PROMPT, &generated, &GEMMA_BOUNDARY);

        assert_eq!(
            prefix,
            vec![2, 105, 2364, 107, 4443, 106, 107, 105, 2364, 107]
        );
    }

    #[test]
    fn an_answer_cut_off_by_the_token_budget_gets_the_suffix_appended_whole() {
        let generated = [4443, 5018, 563];

        let prefix = next_turn_prefix(&GEMMA_PROMPT, &generated, &GEMMA_BOUNDARY);

        assert_eq!(
            prefix,
            vec![2, 105, 2364, 107, 4443, 5018, 563, 106, 107, 105, 2364, 107]
        );
    }

    #[test]
    fn an_answer_of_nothing_but_the_end_of_turn_token_is_just_the_suffix() {
        let prefix = next_turn_prefix(&GEMMA_PROMPT, &[106, 106], &GEMMA_BOUNDARY);

        assert_eq!(prefix, vec![2, 105, 2364, 107, 106, 107, 105, 2364, 107]);
    }

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
