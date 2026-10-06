//! Deeper anticipation (`proxima-tensor/specs/prefix-cache-reuse/SPEC.md`
//! R10, AC14): after an answer, the model drafts a few likely next user turns
//! and each is prefilled as a branch entry sharing the answer's prefix.
//!
//! A branch is an ordinary [`CacheEntry`] -- the prompt cache's own
//! longest-common-prefix lookup picks whichever one the next prompt matches,
//! and [`CacheEntry::branch_base`] only marks it so an unused branch is
//! evicted before any entry a request produced. Drafting is the same seeded
//! decode loop a request runs ([`LoadedModel::run_decode_loop_from_ids`]) from
//! a copy of the answer's rows, sampled at
//! [`crate::PromptCacheConfig::follow_up_temperature_milli`] with a seed per
//! branch; the prefill of the draft and its closing tokens is
//! [`LoadedModel::prefill_through_stops`]. Preemption works at the granularity
//! of one decoded token during drafting, and one forward for the closing
//! tokens, both well inside a prewarm chunk.
//!
//! Default off ([`crate::PromptCacheConfig::follow_up_branches`] is `0`):
//! each branch holds a full copy of the entry's rows. Reach for
//! [`LoadedModel::prewarm`] when the caller knows the next prefix; reach for
//! this when the next user turn is the unknown.

use core::ops::ControlFlow;

use proxima_telemetry::{debug, warn};

use super::prompt_cache::CacheEntry;
use super::*;

impl LoadedModel<'_> {
    /// Registers the tokens that close a user turn and open the model's
    /// reply -- for sliding-pattern the end-of-turn token, a newline and the model-turn
    /// opener -- which every drafted follow-up branch ends with so that a real
    /// user turn the draft matches continues straight into the reply. An
    /// empty list turns follow-up drafting off, its default.
    pub fn set_follow_up_closing(&self, closing: &[u32]) {
        self.prompt_cache
            .lock()
            .set_follow_up_closing(closing);
    }

    /// Drafts [`crate::PromptCacheConfig::follow_up_branches`] likely next
    /// user turns after `ids` (the prompt, the answer and the turn-boundary
    /// suffix, already in the cache) and prefills each behind it as a branch
    /// entry. Returns the drafted user-turn token ids, one per branch kept;
    /// empty when drafting is off, `ids` is not cached, or a request got
    /// there first.
    ///
    /// # Errors
    ///
    /// Same as [`Self::prewarm`].
    pub fn prewarm_follow_ups(
        &self,
        ids: &[u32],
        serving_config: &ServingConfig,
    ) -> Result<Vec<Vec<u32>>, InteropError> {
        self.prewarm_follow_ups_with_progress(ids, serving_config, &mut |_kept| {})
    }

    /// [`Self::prewarm_follow_ups`], calling `on_branch` with the number of
    /// branches kept so far after each one is stored.
    ///
    /// # Errors
    ///
    /// Same as [`Self::prewarm`].
    pub fn prewarm_follow_ups_with_progress(
        &self,
        ids: &[u32],
        serving_config: &ServingConfig,
        on_branch: &mut dyn FnMut(usize),
    ) -> Result<Vec<Vec<u32>>, InteropError> {
        let effective = self.effective_serving_config(serving_config)?;
        let mut runtime = BackendRuntime::new(&effective);
        self.follow_up_branches(ids, &effective, &mut runtime, None, on_branch)
    }

    pub(super) fn follow_up_branches(
        &self,
        base_ids: &[u32],
        serving_config: &ServingConfig,
        runtime: &mut BackendRuntime,
        forced_draft_width: Option<u16>,
        on_branch: &mut dyn FnMut(usize),
    ) -> Result<Vec<Vec<u32>>, InteropError> {
        let config = serving_config.prompt_cache;
        let closing = self
            .prompt_cache
            .lock()
            .follow_up_closing()
            .to_vec();
        if !config.is_enabled()
            || config.follow_up_branches == 0
            || closing.is_empty()
            || base_ids.is_empty()
        {
            return Ok(Vec::new());
        }
        apply_serving_config(serving_config, base_ids.len())?;
        let Some(_slot) = self.prewarm_gate.try_begin() else {
            return Ok(Vec::new());
        };
        if self.prewarm_gate.request_waiting() {
            return Ok(Vec::new());
        }
        let (_, widths) = self.declared_layer_cache_names_and_widths()?;
        let key = self.cache_key(serving_config, runtime, forced_draft_width);
        let (found, _) = self
            .prompt_cache
            .lock()
            .take_for_prewarm(base_ids, &key, &widths, config.min_similarity_milli);
        let Some(base) = found else {
            return Ok(Vec::new());
        };
        if base.state.cached_len + 1 != base_ids.len() {
            self.prompt_cache_store(base, &config);
            return Ok(Vec::new());
        }
        let answer_rows = base.state.cached_len;
        let mut drafts: Vec<Vec<u32>> = Vec::new();
        let mut branch_ids_built: Vec<Vec<u32>> = Vec::new();
        for index in 0..config.follow_up_branches {
            if self.prewarm_gate.request_waiting() {
                break;
            }
            let drafted = self.draft_branch(
                &base,
                base_ids,
                &closing,
                index,
                serving_config,
                runtime,
                forced_draft_width,
                &widths,
            );
            match drafted {
                Ok(Some((entry, draft))) if !branch_ids_built.contains(&entry.state.ids) => {
                    branch_ids_built.push(entry.state.ids.clone());
                    self.prompt_cache_store(entry, &config);
                    drafts.push(draft);
                    on_branch(drafts.len());
                }
                Ok(Some(_)) => debug!(
                    follow_up_branch = u64::from(index),
                    "follow-up draft repeats an earlier branch"
                ),
                Ok(None) => break,
                Err(error) => {
                    warn!(follow_up_error = %error, "follow-up draft failed, keeping the answer entry");
                    break;
                }
            }
        }
        let mut restored = self.prefill_through_stops(
            base_ids,
            base,
            &[base_ids.len()],
            &[],
            &widths,
            &config,
            serving_config,
            runtime,
            forced_draft_width,
            &mut |_position| ControlFlow::Continue(()),
        )?;
        restored.mark_prewarmed(answer_rows, base_ids.len());
        self.prompt_cache_store(restored, &config);
        debug!(
            follow_up_branches = drafts.len() as u64,
            follow_up_requested = u64::from(config.follow_up_branches),
            "follow-up branches prefilled"
        );
        Ok(drafts)
    }

    /// One branch: a draft of the next user turn sampled from a copy of
    /// `base`'s rows, then the draft and `closing` prefilled behind
    /// `base_ids`. `None` when a request arrived mid-draft or the draft held
    /// nothing. Returns the entry and the draft alone.
    #[allow(clippy::too_many_arguments)] // the seeded decode's own knobs, minus the sinks it fixes
    fn draft_branch(
        &self,
        base: &CacheEntry,
        base_ids: &[u32],
        closing: &[u32],
        index: u32,
        serving_config: &ServingConfig,
        runtime: &mut BackendRuntime,
        forced_draft_width: Option<u16>,
        widths: &[LayerPadRowWidths],
    ) -> Result<Option<(CacheEntry, Vec<u32>)>, InteropError> {
        let config = serving_config.prompt_cache;
        let base_len = base_ids.len();
        let draft_config = ServingConfig {
            temperature: config.follow_up_temperature_milli as f32 / 1000.0,
            seed: serving_config.seed.wrapping_add(u64::from(index) + 1),
            ..*serving_config
        };
        let mut preempted = false;
        let (drafted, _text, _stopped, drafted_state) = self.run_decode_loop_from_ids(
            vec![base_ids[base_len - 1]],
            config.follow_up_max_tokens as usize,
            &draft_config,
            runtime,
            None,
            &mut LogitsSink::Discard,
            &mut NodeValuesSink::Discard,
            &mut |_event| {
                preempted = self.prewarm_gate.request_waiting();
                if preempted {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            },
            Some(base.state.branch()),
            true,
            None,
            forced_draft_width,
        )?;
        let draft: &[u32] = match drafted.split_last() {
            Some((tail, rest)) if closing.first() == Some(tail) => rest,
            _ => &drafted,
        };
        if preempted || draft.is_empty() {
            return Ok(None);
        }
        let branch_ids: Vec<u32> = base_ids
            .iter()
            .chain(draft)
            .chain(closing)
            .copied()
            .collect();
        let mut state = drafted_state;
        let keep = state.cached_len.min(base_len + draft.len());
        state
            .rewind_to(keep, 0, widths)
            .map_err(|reason| InteropError::PromptCacheStopRewind {
                position: keep,
                reason: reason.as_str(),
            })?;
        if state.ids != branch_ids[..keep] {
            debug!(
                follow_up_branch = u64::from(index),
                "drafted rows do not follow the answer, branch dropped"
            );
            return Ok(None);
        }
        let mut entry = CacheEntry::empty(base.key);
        entry.state = state;
        entry.branch_base = Some(base_len);
        let entry = self.prefill_through_stops(
            &branch_ids,
            entry,
            &[branch_ids.len()],
            &[],
            widths,
            &config,
            serving_config,
            runtime,
            forced_draft_width,
            &mut |_position| ControlFlow::Continue(()),
        )?;
        Ok(Some((entry, draft.to_vec())))
    }
}
