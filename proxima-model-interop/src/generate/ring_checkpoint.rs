//! Sliding-window checkpoints: the ring rows a window needs at one position,
//! kept so a prompt cache entry can rewind past its ring's slack
//! (`proxima-tensor/specs/prefix-cache-reuse/SPEC.md` R4).
//!
//! A full-attention layer stores every position, so rewinding it is a
//! truncate. A ring layer keeps only `window + slack` recent rows and
//! overwrites the rest, so a rewind past the slack finds rows that no longer
//! exist. A [`RingCheckpoint`] holds, per ring layer, the `min(position,
//! window)` rows ending at its position; restoring writes them back and the
//! layer is exactly as it was at that position. llama-server does the same for
//! its sliding-window and recurrent memory: `slot.prompt.checkpoints`, searched
//! newest first for one whose `pos_min` lies below `pos_min_thold` and whose
//! `pos_max` does not exceed the target (`tools/server/server-context.cpp:
//! 3276-3365`, upstream `f1ea20621`); a checkpoint past the target is erased
//! (`:3367-3378`).
//!
//! This composes the ring's own two primitives rather than adding a third:
//! [`LayerCache::unroll_live_rows`] reads the live window oldest first, and
//! [`LayerCache::append_at`] writes a block of rows to their ring slots, so a
//! checkpoint is the first one's output fed to the second. Reach for
//! [`PrefixState::rewind_to`] when the rewind fits the slack; a checkpoint is
//! for the rewind that does not.

use super::*;

/// One ring layer's live rows at a checkpoint position, oldest first.
struct LayerRows {
    layer: usize,
    k_even: Vec<f32>,
    k_odd: Vec<f32>,
    v: Vec<f32>,
}

/// The ring rows every sliding layer needs at `position` tokens.
pub(super) struct RingCheckpoint {
    position: usize,
    layers: Vec<LayerRows>,
}

impl RingCheckpoint {
    /// Tokens the checkpoint covers: restoring it leaves a state at this
    /// `cached_len`.
    pub(super) const fn position(&self) -> usize {
        self.position
    }

    /// Host bytes held, by allocated capacity.
    pub(super) fn byte_len(&self) -> usize {
        self.layers
            .iter()
            .map(|rows| {
                (rows.k_even.capacity() + rows.k_odd.capacity() + rows.v.capacity())
                    * size_of::<f32>()
            })
            .sum()
    }

    /// Snapshots `state` at its current `cached_len`. `None` when a snapshot
    /// could not restore the state: nothing is cached, a layer is recurrent or
    /// dense (their state is not a window of rows), a ring is displaced
    /// ([`KvRing::write_offset`]), or no layer is a ring (a rewind never needs
    /// one).
    pub(super) fn capture(state: &PrefixState) -> Option<Self> {
        let position = state.cached_len;
        let mut layers = Vec::new();
        for (layer, entry) in state.layer_caches.iter().enumerate() {
            match entry {
                LayerCacheState::SharedFromLayer => {}
                LayerCacheState::Attention(cache) => {
                    if let Some(rows) = LayerRows::capture(layer, cache, position)? {
                        layers.push(rows);
                    }
                }
                LayerCacheState::DenseAttention(_) | LayerCacheState::Ssm(_) => return None,
            }
        }
        (position > 0 && !layers.is_empty()).then_some(Self { position, layers })
    }
}

impl LayerRows {
    /// `Some(None)` for a full-attention layer (nothing to keep), `None` when
    /// the ring cannot be read back.
    fn capture(layer: usize, cache: &LayerCache, position: usize) -> Option<Option<Self>> {
        let Some(ring) = cache.ring_geometry() else {
            return Some(None);
        };
        if ring.write_offset != 0 {
            return None;
        }
        let live = ring.live_rows(position);
        let mut k_even = vec![0.0; live * ring.even_odd_row];
        let mut k_odd = vec![0.0; live * ring.even_odd_row];
        let mut v = vec![0.0; live * ring.v_row];
        cache
            .unroll_live_rows(position, &mut k_even, &mut k_odd, &mut v, layer)
            .ok()?;
        Some(Some(Self {
            layer,
            k_even,
            k_odd,
            v,
        }))
    }
}

impl PrefixState {
    /// Rewinds this state to `checkpoint`'s position: full-attention layers
    /// truncate, each ring layer gets its snapshot rows written back, and the
    /// ids and `cached_len` follow. The caller owns the preconditions: the
    /// checkpoint was captured from a state whose first `position` tokens are
    /// this state's, and `position <= cached_len`.
    pub(super) fn restore_checkpoint(
        &mut self,
        checkpoint: &RingCheckpoint,
        widths: &[LayerPadRowWidths],
    ) {
        let position = checkpoint.position;
        for (state, width) in self.layer_caches.iter_mut().zip(widths) {
            if let (
                LayerCacheState::Attention(cache),
                LayerPadRowWidths::Attention {
                    even_odd_row,
                    v_row,
                },
            ) = (state, width)
            {
                cache.truncate(position, *even_odd_row, *v_row);
            }
        }
        for rows in &checkpoint.layers {
            if let Some(LayerCacheState::Attention(cache)) = self.layer_caches.get_mut(rows.layer)
                && let Some(ring) = cache.ring_geometry().copied()
            {
                let first = position - ring.live_rows(position);
                cache.append_at(first, &rows.k_even, &rows.k_odd, &rows.v);
            }
        }
        self.ids.truncate(position);
        self.cached_len = position;
    }
}

/// Checkpoints kept out of `positions` (ascending) under a cap of `max`: the
/// earliest is pinned when there is room for it, because it is the position
/// nearest the start of the conversation (a system prompt boundary) and a
/// rewind to it is the one no later checkpoint can stand in for; of the rest,
/// the oldest are evicted first, the way llama-server drops the front of
/// `slot.prompt.checkpoints` (`server-context.cpp`, `ctx_checkpoints` cap).
pub(super) fn retained_positions(positions: &[usize], max: usize) -> Vec<usize> {
    if positions.len() <= max {
        return positions.to_vec();
    }
    match max {
        0 => Vec::new(),
        1 => positions[positions.len() - 1..].to_vec(),
        _ => positions[..1]
            .iter()
            .chain(&positions[positions.len() - (max - 1)..])
            .copied()
            .collect(),
    }
}

/// The positions a request should snapshot while it prefills `resume_len..
/// prompt_len`, ascending: the point it resumes at (the previous turn's end,
/// which the next request may want to come back to) when nothing already
/// covers it, every caller-marked turn end, and every multiple of `interval`
/// inside the range, keeping only those that survive [`retained_positions`]
/// beside the `existing` checkpoints. A position equal to `prompt_len` is
/// never planned: the request has to forward at least its last token to have
/// logits to sample from.
pub(super) fn planned_positions(
    existing: &[usize],
    resume_len: usize,
    prompt_len: usize,
    marks: &[usize],
    config: &PromptCacheConfig,
) -> Vec<usize> {
    let max = config.max_checkpoints as usize;
    let interval = config.checkpoint_interval as usize;
    if max == 0 || config.ring_rewind_slack == 0 {
        return Vec::new();
    }
    let multiples = (interval > 0)
        .then(|| (resume_len / interval + 1) * interval)
        .into_iter()
        .flat_map(|first| (first..prompt_len).step_by(interval));
    let resume = (resume_len > 0).then_some(resume_len);
    let mut wanted: Vec<usize> = resume
        .into_iter()
        .chain(marks.iter().copied())
        .chain(multiples)
        .filter(|position| *position >= resume_len && *position < prompt_len && *position > 0)
        .filter(|position| !existing.contains(position))
        .collect();
    wanted.sort_unstable();
    wanted.dedup();
    let mut merged: Vec<usize> = existing
        .iter()
        .copied()
        .chain(wanted.iter().copied())
        .collect();
    merged.sort_unstable();
    let kept = retained_positions(&merged, max);
    wanted.retain(|position| kept.contains(position));
    wanted
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const WINDOW: usize = 8;
    const SLACK: usize = 4;
    const EVEN_ODD_ROW: usize = 2;
    const V_ROW: usize = 3;

    fn marker(position: usize) -> f32 {
        position as f32 + 1.0
    }

    fn ring_layer(positions: usize) -> LayerCache {
        let ring = KvRing::new(WINDOW, SLACK, EVEN_ODD_ROW, V_ROW, 0);
        let mut cache = LayerCache::ring(ring, positions);
        (0..positions).for_each(|position| {
            let value = marker(position);
            cache.append_at(
                position,
                &[value; EVEN_ODD_ROW],
                &[value; EVEN_ODD_ROW],
                &[value; V_ROW],
            );
        });
        cache
    }

    fn full_layer(positions: usize) -> LayerCache {
        let mut cache = LayerCache::new();
        (0..positions).for_each(|position| {
            let value = marker(position);
            cache.append(
                &[value; EVEN_ODD_ROW],
                &[value; EVEN_ODD_ROW],
                &[value; V_ROW],
            );
        });
        cache
    }

    fn state(stored_len: usize) -> PrefixState {
        PrefixState {
            ids: (0..stored_len as u32).collect(),
            layer_caches: vec![
                LayerCacheState::Attention(ring_layer(stored_len)),
                LayerCacheState::Attention(full_layer(stored_len)),
                LayerCacheState::SharedFromLayer,
            ],
            cached_len: stored_len,
        }
    }

    fn widths() -> Vec<LayerPadRowWidths> {
        let attention = || LayerPadRowWidths::Attention {
            even_odd_row: EVEN_ODD_ROW,
            v_row: V_ROW,
        };
        vec![attention(), attention(), LayerPadRowWidths::SharedFromLayer]
    }

    fn ring_marker_at(state: &PrefixState, position: usize) -> f32 {
        match &state.layer_caches[0] {
            LayerCacheState::Attention(cache) => {
                let ring = cache.ring_geometry().expect("layer 0 is a ring");
                cache.k_even[(position % ring.capacity) * EVEN_ODD_ROW]
            }
            _ => 0.0,
        }
    }

    fn config(interval: u32, max_checkpoints: u32) -> PromptCacheConfig {
        PromptCacheConfig {
            byte_budget: 1 << 20,
            checkpoint_interval: interval,
            max_checkpoints,
            ..PromptCacheConfig::off()
        }
    }

    /// A checkpoint taken at 20 tokens restores the ring layer to exactly the
    /// window rows 12..20, even after the ring wrote 24 more positions over
    /// them, and truncates the full layer, the ids and `cached_len` to 20.
    #[test]
    fn restoring_a_checkpoint_brings_back_the_window_the_ring_overwrote() {
        let checkpoint = RingCheckpoint::capture(&state(20)).expect("a ring layer is captured");
        let mut later = state(44);
        assert_ne!(ring_marker_at(&later, 12), marker(12));

        later.restore_checkpoint(&checkpoint, &widths());

        assert_eq!(later.cached_len, 20);
        assert_eq!(later.ids.len(), 20);
        assert!(
            (12..20).all(|position| ring_marker_at(&later, position) == marker(position)),
            "every window row ending at position 20 reads back its own marker"
        );
        match &later.layer_caches[1] {
            LayerCacheState::Attention(cache) => {
                assert_eq!(cache.k_even.len(), 20 * EVEN_ODD_ROW);
            }
            _ => panic!("layer 1 is a full-attention layer"),
        }
    }

    #[test]
    fn a_checkpoint_inside_the_first_window_holds_only_the_rows_that_exist() {
        let checkpoint = RingCheckpoint::capture(&state(5)).expect("a ring layer is captured");

        let bytes = (5 * EVEN_ODD_ROW * 2 + 5 * V_ROW) * size_of::<f32>();
        assert_eq!(checkpoint.position(), 5);
        assert_eq!(checkpoint.byte_len(), bytes);
    }

    #[test]
    fn a_checkpoint_is_refused_for_state_a_window_of_rows_cannot_restore() {
        let recurrent = PrefixState {
            ids: vec![1, 2, 3],
            layer_caches: vec![LayerCacheState::Ssm(SsmLayerCache::new(4, 4))],
            cached_len: 3,
        };
        let empty = PrefixState {
            ids: Vec::new(),
            layer_caches: vec![LayerCacheState::Attention(ring_layer(0))],
            cached_len: 0,
        };
        let full_only = PrefixState {
            ids: vec![1, 2],
            layer_caches: vec![LayerCacheState::Attention(full_layer(2))],
            cached_len: 2,
        };

        assert!(RingCheckpoint::capture(&recurrent).is_none());
        assert!(RingCheckpoint::capture(&empty).is_none());
        assert!(RingCheckpoint::capture(&full_only).is_none());
    }

    #[test]
    fn retention_pins_the_earliest_and_evicts_the_oldest_of_the_rest() {
        let positions = [512, 1024, 1536, 2048, 2560];

        assert_eq!(
            retained_positions(&positions, 3),
            vec![512, 2048, 2560],
            "the earliest stays, the two newest fill the rest"
        );
        assert_eq!(retained_positions(&positions, 5), positions.to_vec());
        assert_eq!(retained_positions(&positions, 1), vec![2560]);
        assert!(retained_positions(&positions, 0).is_empty());
        assert_eq!(retained_positions(&positions, 2), vec![512, 2560]);
    }

    /// Prefilling 0..5000 under a 1024 interval snapshots 1024, 2048, 3072
    /// and 4096; a cap of 3 keeps the earliest and the two newest, so the
    /// 2048 snapshot is never taken at all.
    #[test]
    fn planning_skips_the_snapshots_retention_would_evict() {
        let planned = planned_positions(&[], 0, 5000, &[], &config(1024, 3));

        assert_eq!(planned, vec![1024, 3072, 4096]);
    }

    #[test]
    fn caller_marks_and_the_resume_point_join_the_interval_positions() {
        let planned = planned_positions(&[], 1500, 4000, &[1800, 3999], &config(1024, 8));

        assert_eq!(planned, vec![1500, 1800, 2048, 3072, 3999]);
    }

    #[test]
    fn a_position_already_checkpointed_or_outside_the_prefill_range_is_not_planned() {
        let planned = planned_positions(&[1024], 1024, 3000, &[1024, 7000, 400], &config(1024, 8));

        assert_eq!(planned, vec![2048]);
    }

    #[test]
    fn planning_is_empty_when_checkpoints_are_off_or_the_ring_has_no_slack() {
        assert!(planned_positions(&[], 0, 5000, &[2000], &config(1024, 0)).is_empty());
        let no_slack = PromptCacheConfig {
            ring_rewind_slack: 0,
            ..config(1024, 4)
        };
        assert!(planned_positions(&[], 0, 5000, &[2000], &no_slack).is_empty());
    }

    #[test]
    fn marks_alone_plan_when_the_interval_is_zero() {
        let planned = planned_positions(&[], 0, 5000, &[2000, 3500], &config(0, 4));

        assert_eq!(planned, vec![2000, 3500]);
    }
}
