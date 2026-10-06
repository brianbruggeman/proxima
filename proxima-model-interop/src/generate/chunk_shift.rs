//! Chunk reuse after a divergence: a request whose prompt stopped matching an
//! entry's ids still finds runs of its later tokens that the entry holds at
//! another position, and moves those rows instead of recomputing them
//! (`proxima-tensor/specs/prefix-cache-reuse/SPEC.md` R5, llama-server's
//! `n_cache_reuse`).
//!
//! llama-server walks the cache pointer forward from the common prefix and,
//! where a run of at least `n_cache_reuse` tokens matches the prompt at the
//! prompt pointer, calls `seq_rm` over the gap and `seq_add` over the run
//! (`tools/server/server-context.cpp:3217-3264`, upstream `f1ea20621`).
//! `seq_add` only renumbers cell positions; the keys are re-rotated by the
//! position delta in the next K-shift graph, `build_rope_shift` rotating the
//! first `n_rot` dims per layer with that layer's own base and scale
//! (`src/llama-kv-cache.cpp:1924-1964`, `:2003-2056`). Here the rows live
//! host side, so the move is the same rotation applied to the rows
//! ([`rotate_rows`]) and a write at the new position. The prompt pointer in
//! llama.cpp only advances on a match, so a replaced span (a summary where the
//! middle turns were) hides every later run from it; [`plan_runs`] also steps
//! the prompt pointer, so the kept turns after a summary are found.
//!
//! A cached key is already rotated for its old position `p` (`R(p) k`), and
//! rotations compose, so the key a fresh prefill stores at `p + d` is
//! `R(d) (R(p) k)`: one rotation by the delta, with the angle table the model
//! itself builds for a single position `d` ([`build_position_inputs`] for the
//! builtin table, [`sliding_rope_inputs`] for a named one). Which table
//! a layer rotates with is read off the bound program ([`rope_leaves_of`]),
//! not guessed from the layer kind.
//!
//! Ring layers keep only `window + slack` rows, so a chunk's ring rows exist
//! only where the entry's ring still holds them ([`ring_rows_live`]); a run
//! whose rows are gone is left to be prefilled like any other gap. A moved
//! chunk's rows were computed under the entry's older context, so layers past
//! the first full-attention layer see keys that differ from a fresh prefill
//! by what that context contributed; llama-server's reuse has the same
//! property.

use core::ops::ControlFlow;

use proxima_telemetry::debug;

use super::prompt_cache::CacheEntry;
use super::ring_checkpoint::LayerRows;
use super::*;

/// A run of `len` tokens found at `old_start` in an entry's ids and at
/// `new_start` in the prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ChunkRun {
    pub(super) old_start: usize,
    pub(super) new_start: usize,
    pub(super) len: usize,
}

impl ChunkRun {
    pub(super) const fn old_end(&self) -> usize {
        self.old_start + self.len
    }

    pub(super) const fn new_end(&self) -> usize {
        self.new_start + self.len
    }
}

/// A run's rows lifted out of the entry, keys already rotated to the new
/// position, ready to be written once the state has reached
/// `run.new_start`.
pub(super) struct MovedRun {
    pub(super) run: ChunkRun,
    pub(super) ids: Vec<u32>,
    pub(super) layers: Vec<LayerRows>,
}

const WINDOW_BASE: u64 = 0x9E37_79B9_7F4A_7C15;

/// Polynomial hash of every window of `window` ids, in order; empty when
/// `ids` is shorter than one window.
fn window_hashes(ids: &[u32], window: usize) -> Vec<u64> {
    if window == 0 || ids.len() < window {
        return Vec::new();
    }
    let top = (1..window).fold(1_u64, |power, _| power.wrapping_mul(WINDOW_BASE));
    let digit = |id: u32| u64::from(id) + 1;
    let mut hash = ids[..window].iter().fold(0_u64, |hash, id| {
        hash.wrapping_mul(WINDOW_BASE).wrapping_add(digit(*id))
    });
    let mut hashes = Vec::with_capacity(ids.len() - window + 1);
    hashes.push(hash);
    for index in 0..ids.len() - window {
        hash = hash
            .wrapping_sub(digit(ids[index]).wrapping_mul(top))
            .wrapping_mul(WINDOW_BASE)
            .wrapping_add(digit(ids[index + window]));
        hashes.push(hash);
    }
    hashes
}

/// The runs of at least `min_len` tokens that `prompt[from..]` shares with
/// `stored[from..]`, in prompt order, each at a stored position at or past
/// where the previous one ended. A candidate is taken at the first stored
/// position that matches (llama.cpp's walk takes the first too) and extended
/// while the ids keep matching; the last prompt token is never inside a run,
/// because it has to be forwarded to produce logits. Windows are hashed with
/// a rolling hash and every hit is verified against the ids, so a collision
/// costs a comparison, never a wrong run. O(stored + prompt) expected.
pub(super) fn plan_runs(
    stored: &[u32],
    prompt: &[u32],
    from: usize,
    min_len: usize,
) -> Vec<ChunkRun> {
    let prompt_end = prompt.len().saturating_sub(1);
    if min_len == 0 || from >= stored.len() || from + min_len > prompt_end {
        return Vec::new();
    }
    let mut index: Vec<(u64, usize)> = window_hashes(&stored[from..], min_len)
        .into_iter()
        .enumerate()
        .map(|(offset, hash)| (hash, from + offset))
        .collect();
    index.sort_unstable();
    let prompt_hashes = window_hashes(&prompt[from..prompt_end], min_len);
    let mut runs = Vec::new();
    let mut cursor = from;
    let mut floor = from;
    while cursor + min_len <= prompt_end {
        let hash = prompt_hashes[cursor - from];
        let first = index.partition_point(|entry| *entry < (hash, floor));
        let found = index[first..]
            .iter()
            .take_while(|(held, _)| *held == hash)
            .map(|(_, position)| *position)
            .find(|position| {
                stored[*position..*position + min_len] == prompt[cursor..cursor + min_len]
            });
        let Some(position) = found else {
            cursor += 1;
            continue;
        };
        let len = stored[position..]
            .iter()
            .zip(&prompt[cursor..prompt_end])
            .take_while(|(held, wanted)| held == wanted)
            .count();
        runs.push(ChunkRun {
            old_start: position,
            new_start: cursor,
            len,
        });
        cursor += len;
        floor = position + len;
    }
    runs
}

/// Whether a ring still holds the rows of `run` that the window after it
/// reads: the last `min(len, window)` rows ending at `run.old_end()`, in an
/// entry of `stored_len` tokens whose rings were last restored to
/// `restored_at` (the conditions [`ring_rewind_fits`] applies to a window).
fn ring_rows_live(ring: &KvRing, run: &ChunkRun, stored_len: usize, restored_at: usize) -> bool {
    let first_needed = run.old_end() - run.len.min(ring.window);
    ring.write_offset == 0
        && first_needed + ring.capacity >= stored_len
        && first_needed + ring.window >= restored_at
}

/// Applies the angle `(cos, sin)` of one position to every `(even, odd)`
/// pair of `rows` laid out `[row][head][pair]`, the rotation a cached key
/// takes when its position moves by the angle's delta: `even' = even cos -
/// odd sin`, `odd' = odd cos + even sin` (`fused_rope_pair`,
/// `proxima-tensor/src/spec/primitives.rs`). `None` when the table does not
/// tile the rows.
fn rotate_rows(even: &mut [f32], odd: &mut [f32], cos: &[f32], sin: &[f32]) -> Option<()> {
    let pairs = cos.len();
    if pairs == 0
        || sin.len() != pairs
        || even.len() != odd.len()
        || !even.len().is_multiple_of(pairs)
    {
        return None;
    }
    for (even_row, odd_row) in even
        .chunks_exact_mut(pairs)
        .zip(odd.chunks_exact_mut(pairs))
    {
        for (((first, second), cosine), sine) in even_row
            .iter_mut()
            .zip(odd_row.iter_mut())
            .zip(cos)
            .zip(sin)
        {
            let (rotated_first, rotated_second) = (
                *first * cosine - *second * sine,
                *second * cosine + *first * sine,
            );
            *first = rotated_first;
            *second = rotated_second;
        }
    }
    Some(())
}

impl LayerRows {
    /// The rows of `run` that `cache` holds, oldest first: every row of a
    /// full layer's chunk, the last `min(len, window)` rows of a ring.
    pub(super) fn of_run(
        layer: usize,
        cache: &LayerCache,
        run: &ChunkRun,
        even_odd_row: usize,
        v_row: usize,
    ) -> Option<Self> {
        let Some(ring) = cache.ring_geometry() else {
            let range = |width: usize| run.old_start * width..run.old_end() * width;
            return Some(Self {
                layer,
                k_even: cache.k_even.get(range(even_odd_row))?.to_vec(),
                k_odd: cache.k_odd.get(range(even_odd_row))?.to_vec(),
                v: cache.v.get(range(v_row))?.to_vec(),
            });
        };
        let live = ring.live_rows(run.old_end());
        let mut rows = Self {
            layer,
            k_even: vec![0.0; live * ring.even_odd_row],
            k_odd: vec![0.0; live * ring.even_odd_row],
            v: vec![0.0; live * ring.v_row],
        };
        cache
            .unroll_live_rows(
                run.old_end(),
                &mut rows.k_even,
                &mut rows.k_odd,
                &mut rows.v,
                layer,
            )
            .ok()?;
        let skipped = live - run.len.min(ring.window);
        rows.k_even.drain(..skipped * ring.even_odd_row);
        rows.k_odd.drain(..skipped * ring.even_odd_row);
        rows.v.drain(..skipped * ring.v_row);
        Some(rows)
    }
}

/// Lifts `run`'s rows out of `state`, every attention layer's keys rotated
/// by `rotations[layer]` (the `(cos, sin)` of the run's position delta).
/// `None` when a layer's rows or rotation are not there.
pub(super) fn extract_run(
    state: &PrefixState,
    run: ChunkRun,
    widths: &[LayerPadRowWidths],
    rotations: &[Option<(Vec<f32>, Vec<f32>)>],
) -> Option<MovedRun> {
    let mut layers = Vec::new();
    for (layer, ((cache_state, width), rotation)) in state
        .layer_caches
        .iter()
        .zip(widths)
        .zip(rotations)
        .enumerate()
    {
        let (
            LayerCacheState::Attention(cache),
            LayerPadRowWidths::Attention {
                even_odd_row,
                v_row,
            },
        ) = (cache_state, width)
        else {
            continue;
        };
        let (cos, sin) = rotation.as_ref()?;
        let mut rows = LayerRows::of_run(layer, cache, &run, *even_odd_row, *v_row)?;
        rotate_rows(&mut rows.k_even, &mut rows.k_odd, cos, sin)?;
        layers.push(rows);
    }
    let ids = state.ids.get(run.old_start..run.old_end())?.to_vec();
    Some(MovedRun { run, ids, layers })
}

impl PrefixState {
    /// Whether every layer's cache is a window of attention rows, the only
    /// state a chunk can be lifted out of and written back into.
    fn rows_are_movable(&self) -> bool {
        self.layer_caches.iter().all(|state| match state {
            LayerCacheState::Attention(cache) => cache
                .ring_geometry()
                .is_none_or(|ring| ring.write_offset == 0),
            LayerCacheState::SharedFromLayer => true,
            LayerCacheState::DenseAttention(_) | LayerCacheState::Ssm(_) => false,
        })
    }

    /// Writes `moved` at the end of this state, which must hold exactly the
    /// run's `new_start` tokens: a full layer appends the chunk, a ring layer
    /// writes its rows to the slots of their new positions, and the ids and
    /// `cached_len` follow.
    fn append_moved(&mut self, moved: &MovedRun) -> Result<(), &'static str> {
        let run = &moved.run;
        if self.cached_len != run.new_start {
            return Err("the state is not at the start of the run");
        }
        for rows in &moved.layers {
            let Some(LayerCacheState::Attention(cache)) = self.layer_caches.get_mut(rows.layer)
            else {
                return Err("the run holds rows for a layer the state does not cache");
            };
            match cache.ring_geometry().copied() {
                Some(ring) => {
                    let kept = rows.k_even.len() / ring.even_odd_row.max(1);
                    cache.append_at(run.new_end() - kept, &rows.k_even, &rows.k_odd, &rows.v);
                }
                None => {
                    let even_odd_row = rows.k_even.len() / run.len.max(1);
                    let v_row = rows.v.len() / run.len.max(1);
                    cache.truncate(run.new_start, even_odd_row, v_row);
                    if cache.k_even.len() != run.new_start * even_odd_row {
                        return Err("a full layer does not hold exactly the rows before the run");
                    }
                    cache.append(&rows.k_even, &rows.k_odd, &rows.v);
                }
            }
        }
        self.ids.extend_from_slice(&moved.ids);
        self.cached_len = run.new_end();
        Ok(())
    }
}

impl CacheEntry {
    /// Writes `moved` into the entry. A ring layer holds only the last window
    /// of a chunk, so its rows before `new_end - window` are not the chunk's:
    /// the entry is marked restored at the chunk's end, which is how
    /// [`ring_rewind_fits`] learns that a rewind to before it must come from a
    /// checkpoint.
    ///
    /// # Errors
    ///
    /// [`InteropError::PromptCacheShift`] when the state cannot take the run.
    pub(super) fn apply_moved(&mut self, moved: &MovedRun) -> Result<(), InteropError> {
        self.state
            .append_moved(moved)
            .map_err(|reason| InteropError::PromptCacheShift {
                position: moved.run.new_start,
                reason,
            })?;
        self.restored_at = self.restored_at.max(moved.run.new_end());
        Ok(())
    }
}

/// The `(cos, sin)` leaf names and pair count the program rotates layer
/// `layer`'s cached keys with, read off the cache roots: the rotated key's
/// first half is `same * cos - partner * sin` over two products, each of
/// which names its table leaf (`fused_rope_pair`).
fn rope_leaves_of<'program>(
    program: &'program [Op],
    layer_roots: &LayerCacheRoots,
) -> Option<(&'program str, &'program str, usize)> {
    let LayerCacheRoots::Attention((rotated_even, ..)) = layer_roots else {
        return None;
    };
    let operand = |node: NodeId, position: usize| match program.get(node.0 as usize)? {
        Op::Elementwise { operands, .. } => operands.get(position).map(|(operand, _)| *operand),
        _ => None,
    };
    let leaf = |product: NodeId| match program.get(operand(product, 1)?.0 as usize)? {
        Op::Input {
            name: Some(name),
            shape,
            ..
        } => match shape.last()? {
            Extent::Static(pairs) => Some((name.as_str(), usize::try_from(*pairs).ok()?)),
            _ => None,
        },
        _ => None,
    };
    let (cos, pairs) = leaf(operand(*rotated_even, 0)?)?;
    let (sin, _) = leaf(operand(*rotated_even, 1)?)?;
    Some((cos, sin, pairs))
}

impl LoadedModel<'_> {
    /// The `(cos, sin)` of a position delta of `delta` for every layer, in the
    /// table that layer's program rotates its keys with. A negative delta
    /// rotates the other way: cosine is even, sine odd. `None` for a layer
    /// that owns no cache, and for a layer whose table is not one this model
    /// can produce for a single position.
    pub(super) fn delta_rotations(
        &self,
        serving_config: &ServingConfig,
        delta: isize,
    ) -> Vec<Option<(Vec<f32>, Vec<f32>)>> {
        let magnitude = delta.unsigned_abs();
        let scaling = self.effective_rope_scaling(serving_config);
        let builtin = build_position_inputs(
            &[0],
            magnitude,
            self.architecture.head_dim,
            self.architecture.rope_freq_base,
            self.architecture.rms_epsilon,
            rope_freq_factors(&self.weights),
            scaling,
        );
        let attention_factor = scaling.attention_factor();
        let mut named = Vec::new();
        sliding_rope_inputs(&self.architecture, magnitude, 1, &mut named);
        let sign = if delta < 0 { -1.0 } else { 1.0 };
        let table = |name: &str, builtin_values: &[f32], builtin_name: &str, scale: f32| {
            let values = if name == builtin_name {
                builtin_values
            } else {
                named
                    .iter()
                    .find(|input| input.name == name)
                    .map(|input| input.values.as_slice())?
            };
            let divisor = if name == builtin_name {
                attention_factor
            } else {
                1.0
            };
            Some(
                values
                    .iter()
                    .map(|value| value * scale / divisor)
                    .collect::<Vec<f32>>(),
            )
        };
        self.layer_roots
            .iter()
            .map(|roots| {
                let (cos_name, sin_name, pairs) = rope_leaves_of(&self.program, roots)?;
                let cos = table(cos_name, &builtin.cos, "rope_cos", 1.0)?;
                let sin = table(sin_name, &builtin.sin, "rope_sin", sign)?;
                (cos.len() == pairs && sin.len() == pairs).then_some((cos, sin))
            })
            .collect()
    }

    /// The chunks of `entry` that `prompt_ids` shares from `from` on, lifted
    /// out for [`CacheEntry::apply_moved`], in prompt order: the runs
    /// [`plan_runs`] finds, minus those whose ring rows are gone
    /// ([`ring_rows_live`]) or whose rotation this model cannot build. Empty
    /// when the entry's state cannot be moved at all.
    pub(super) fn lift_chunks(
        &self,
        entry: &CacheEntry,
        prompt_ids: &[u32],
        from: usize,
        widths: &[LayerPadRowWidths],
        serving_config: &ServingConfig,
    ) -> Vec<MovedRun> {
        let state = &entry.state;
        let min_len = serving_config.prompt_cache.cache_reuse_min as usize;
        if !state.rows_are_movable() || widths.len() != state.layer_caches.len() {
            return Vec::new();
        }
        let stored = &state.ids[..state.cached_len.min(state.ids.len())];
        let ring = sliding_ring_geometry(&state.layer_caches);
        let planned = plan_runs(stored, prompt_ids, from, min_len);
        let live: Vec<ChunkRun> = planned
            .iter()
            .copied()
            .filter(|run| {
                ring.as_ref()
                    .is_none_or(|ring| ring_rows_live(ring, run, stored.len(), entry.restored_at))
            })
            .collect();
        let moved: Vec<MovedRun> = live
            .iter()
            .filter_map(|run| {
                let delta =
                    isize::try_from(run.new_start).ok()? - isize::try_from(run.old_start).ok()?;
                let rotations = self.delta_rotations(serving_config, delta);
                extract_run(state, *run, widths, &rotations)
            })
            .collect();
        debug!(
            cache_shift_planned_runs = planned.len() as u64,
            cache_shift_live_runs = live.len() as u64,
            cache_shift_lifted_runs = moved.len() as u64,
            cache_shift_min_len = min_len as u64,
            "prompt cache chunk plan"
        );
        moved
    }
}

/// The planned checkpoint positions of one gap, and the stops its prefill
/// runs to (the planned positions, then the position the gap ends at).
type GapStops = (Vec<usize>, Vec<usize>);

/// Splits the planned checkpoint `positions` (ascending) around `runs`
/// (ascending, disjoint): for each run, the planned positions at or before
/// its start and the stops its gap must prefill to (those, then the run's
/// start, which the prefill must reach before the run is written); and the
/// positions at or after the last run's end. A position strictly inside a run
/// is dropped: no prefill reaches it.
fn split_stops(positions: &[usize], runs: &[ChunkRun]) -> (Vec<GapStops>, Vec<usize>) {
    let mut remaining = positions.to_vec();
    let mut gaps = Vec::with_capacity(runs.len());
    for run in runs {
        let split = remaining.partition_point(|position| *position <= run.new_start);
        let planned: Vec<usize> = remaining.drain(..split).collect();
        let mut stops = planned.clone();
        if stops.last() != Some(&run.new_start) {
            stops.push(run.new_start);
        }
        gaps.push((planned, stops));
        remaining.retain(|position| *position >= run.new_end());
    }
    (gaps, remaining)
}

impl LoadedModel<'_> {
    /// [`Self::prefill_through_stops`] over a request whose entry carries
    /// lifted chunks ([`CacheEntry::moved`]): the tokens before each chunk are
    /// prefilled up to the chunk's new start, the chunk is written there, and
    /// the prefill continues behind it. Checkpoint positions inside a chunk
    /// are skipped (no prefill reaches them); one at a chunk's end is taken
    /// from the state after the write. With no chunks this is one
    /// `prefill_through_stops` over `positions`.
    #[allow(clippy::too_many_arguments)] // `prefill_through_stops`'s own, which it forwards to
    pub(super) fn prefill_through_runs(
        &self,
        ids: &[u32],
        mut entry: CacheEntry,
        positions: &[usize],
        widths: &[LayerPadRowWidths],
        config: &PromptCacheConfig,
        serving_config: &ServingConfig,
        runtime: &mut BackendRuntime,
        forced_draft_width: Option<u16>,
    ) -> Result<CacheEntry, InteropError> {
        let moved = core::mem::take(&mut entry.moved);
        let runs: Vec<ChunkRun> = moved.iter().map(|lifted| lifted.run).collect();
        let (gaps, remaining) = split_stops(positions, &runs);
        for (lifted, (planned, stops)) in moved.iter().zip(&gaps) {
            entry = self.prefill_through_stops(
                ids,
                entry,
                stops,
                planned,
                widths,
                config,
                serving_config,
                runtime,
                forced_draft_width,
                &mut |_position| ControlFlow::Continue(()),
            )?;
            entry.apply_moved(lifted)?;
            debug!(
                cache_shift_old_start = lifted.run.old_start as u64,
                cache_shift_new_start = lifted.run.new_start as u64,
                cache_shift_tokens = lifted.run.len as u64,
                "prompt cache chunk moved"
            );
        }
        self.prefill_through_stops(
            ids,
            entry,
            &remaining,
            &remaining,
            widths,
            config,
            serving_config,
            runtime,
            forced_draft_width,
            &mut |_position| ControlFlow::Continue(()),
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::super::prompt_cache::longest_common_prefix;
    use super::*;
    use crate::rope_scaling::RopeScaling;

    /// llama-server's loop (`server-context.cpp:3217-3264`) with the prompt
    /// pointer also stepping on a miss, written the slow way: from each prompt
    /// position try every stored position at or past the floor.
    fn scan_reference(
        stored: &[u32],
        prompt: &[u32],
        from: usize,
        min_len: usize,
    ) -> Vec<ChunkRun> {
        let prompt_end = prompt.len().saturating_sub(1);
        let mut runs = Vec::new();
        let (mut cursor, mut floor) = (from, from);
        while cursor < prompt_end {
            let found = (floor..stored.len()).find_map(|position| {
                let len = stored[position..]
                    .iter()
                    .zip(&prompt[cursor..prompt_end])
                    .take_while(|(held, wanted)| held == wanted)
                    .count();
                (len >= min_len && len > 0).then_some((position, len))
            });
            match found {
                Some((position, len)) => {
                    runs.push(ChunkRun {
                        old_start: position,
                        new_start: cursor,
                        len,
                    });
                    cursor += len;
                    floor = position + len;
                }
                None => cursor += 1,
            }
        }
        runs
    }

    /// Worked example, by hand. A cached conversation of 20 tokens: a
    /// 4-token system prompt (1..=4), three old turns (10 tokens, 21..=30),
    /// the last turn (6 tokens, 41..=46). The new prompt keeps the system
    /// prompt, replaces the old turns with a 3-token summary (90, 91, 92)
    /// and keeps the last turn, then adds a new user token (99). With
    /// `min_len = 4`: the common prefix is 4 tokens (1..=4); from prompt
    /// position 4 the tokens 90, 91, 92 appear nowhere in the stored ids, so
    /// the prompt pointer steps past them; at prompt position 7 the window
    /// 41..=44 matches stored position 14 and extends over 41..=46, a run of
    /// 6 (the stored ids end there), so the run is old 14, new 7, length 6,
    /// the delta is 7 - 14 = -7; 99 is the
    /// last prompt token and is never in a run.
    #[test]
    fn a_kept_turn_after_a_replaced_span_is_found_at_its_new_position() {
        let stored: Vec<u32> = [1, 2, 3, 4]
            .into_iter()
            .chain(21..=30)
            .chain(41..=46)
            .collect();
        let prompt: Vec<u32> = [1, 2, 3, 4, 90, 91, 92]
            .into_iter()
            .chain(41..=46)
            .chain([99])
            .collect();
        assert_eq!(stored.len(), 20);
        assert_eq!(longest_common_prefix(&stored, &prompt), 4);

        let runs = plan_runs(&stored, &prompt, 4, 4);

        assert_eq!(
            runs,
            vec![ChunkRun {
                old_start: 14,
                new_start: 7,
                len: 6
            }]
        );
        assert_eq!(runs[0].old_end(), 20);
        assert_eq!(runs[0].new_end(), 13);
    }

    /// llama.cpp's own case: the middle is dropped with nothing in its place, so
    /// the prompt pointer never has to step. A run shorter than `min_len`
    /// (the 2-token echo 5, 6) is not worth moving.
    #[test]
    fn a_dropped_middle_is_one_run_and_a_short_echo_is_left_to_prefill() {
        let stored: Vec<u32> = (1..=30).collect();
        let prompt: Vec<u32> = (1..=6).chain(21..=30).chain([200, 5, 6, 201]).collect();

        let runs = plan_runs(&stored, &prompt, 6, 4);

        assert_eq!(
            runs,
            vec![ChunkRun {
                old_start: 20,
                new_start: 6,
                len: 10
            }]
        );
    }

    #[test]
    fn the_last_prompt_token_is_never_inside_a_run() {
        let stored: Vec<u32> = (1..=12).collect();
        let prompt: Vec<u32> = [50, 51].into_iter().chain(5..=12).collect();

        let runs = plan_runs(&stored, &prompt, 0, 3);

        assert_eq!(
            runs,
            vec![ChunkRun {
                old_start: 4,
                new_start: 2,
                len: 7
            }]
        );
        assert_eq!(runs[0].new_end(), prompt.len() - 1);
    }

    #[test]
    fn runs_never_move_backwards_in_the_stored_ids() {
        let stored: Vec<u32> = (1..=6).chain(11..=16).collect();
        let prompt: Vec<u32> = (11..=16).chain(1..=6).chain([0]).collect();

        let runs = plan_runs(&stored, &prompt, 0, 4);

        assert_eq!(
            runs,
            vec![ChunkRun {
                old_start: 6,
                new_start: 0,
                len: 6
            }]
        );
    }

    #[test]
    fn no_runs_when_nothing_is_long_enough_or_the_window_is_zero() {
        let stored: Vec<u32> = (1..=10).collect();
        let prompt: Vec<u32> = [7, 8, 9, 100, 3, 4, 101].to_vec();

        assert!(plan_runs(&stored, &prompt, 0, 4).is_empty());
        assert!(plan_runs(&stored, &prompt, 0, 0).is_empty());
        assert!(plan_runs(&stored, &prompt, 10, 2).is_empty());
        assert!(plan_runs(&stored, &[1], 0, 1).is_empty());
    }

    #[test]
    fn the_plan_equals_the_slow_scan_over_10000_generated_pairs() {
        let mut rng = fastrand::Rng::with_seed(0x5EED_5417);
        let mut with_runs = 0_usize;
        for _ in 0..10_000 {
            let alphabet = rng.u32(2..6);
            let stored: Vec<u32> = (0..rng.usize(1..60))
                .map(|_| rng.u32(0..alphabet))
                .collect();
            let prompt: Vec<u32> = (0..rng.usize(1..60))
                .map(|_| rng.u32(0..alphabet))
                .collect();
            let from = longest_common_prefix(&stored, &prompt).min(prompt.len() - 1);
            let min_len = rng.usize(1..8);

            let planned = plan_runs(&stored, &prompt, from, min_len);
            let expected = scan_reference(&stored, &prompt, from, min_len);

            assert_eq!(
                planned, expected,
                "{stored:?} {prompt:?} from {from} min {min_len}"
            );
            with_runs += usize::from(!planned.is_empty());
        }
        assert!(
            with_runs > 1000,
            "only {with_runs} of 10,000 cases found a run"
        );
    }

    /// Worked example, by hand. A head of 2 pairs, `base = 100`, `dim = 4`:
    /// pair 0 turns at 1 rad per position, pair 1 at `100^(-1/2) = 0.1`. A
    /// key whose first pair is `(1, 0)` at position 5 sits at `R(5)`
    /// applied to its unrotated pair `(cos 5, -sin 5)` = `(0.2836622,
    /// 0.9589243)`. Moving it to position 2 is a delta of -3: `cos(-3) =
    /// -0.9899925`, `sin(-3) = -0.1411200`; `even' = 1 * -0.9899925 - 0 *
    /// -0.1411200 = -0.9899925`, `odd' = 0 * -0.9899925 + 1 * -0.1411200 =
    /// -0.1411200`. A fresh rotation of the unrotated pair at position 2
    /// (`cos 2 = -0.4161468`, `sin 2 = 0.9092974`) gives
    /// `even = -0.4161468 * 0.2836622 - 0.9092974 * 0.9589243 = -0.9899925`
    /// and `odd = 0.9092974 * 0.2836622 - 0.4161468 * 0.9589243 = -0.1411200`
    /// (the second product is negative, so the sign is minus). Same pair.
    #[test]
    fn a_key_moved_back_three_positions_equals_a_fresh_rotation_at_the_new_position() {
        let mut even = [1.0_f32, 0.5];
        let mut odd = [0.0_f32, -2.0];
        let unrotated_first = (5.0_f32.cos(), -5.0_f32.sin());

        let delta = -3.0_f32;
        let cos = [delta.cos(), (delta * 0.1).cos()];
        let sin = [delta.sin(), (delta * 0.1).sin()];
        rotate_rows(&mut even, &mut odd, &cos, &sin).expect("the table tiles the row");

        assert!((even[0] - -0.989_992_5).abs() < 1e-6, "{}", even[0]);
        assert!((odd[0] - -0.141_120).abs() < 1e-6, "{}", odd[0]);
        let (fresh_even, fresh_odd) = (
            2.0_f32.cos() * unrotated_first.0 - 2.0_f32.sin() * unrotated_first.1,
            2.0_f32.sin() * unrotated_first.0 + 2.0_f32.cos() * unrotated_first.1,
        );
        assert!((even[0] - fresh_even).abs() < 1e-6);
        assert!((odd[0] - fresh_odd).abs() < 1e-6);
    }

    #[test]
    fn a_rotation_by_a_delta_and_back_restores_the_rows() {
        let original_even: Vec<f32> = (0..12).map(|index| index as f32 * 0.25 - 1.0).collect();
        let original_odd: Vec<f32> = (0..12).map(|index| 2.0 - index as f32 * 0.125).collect();
        let (mut even, mut odd) = (original_even.clone(), original_odd.clone());
        let angles: Vec<f32> = (0..3).map(|pair| 37.0 * 0.5_f32.powi(pair)).collect();
        let cos: Vec<f32> = angles.iter().map(|angle| angle.cos()).collect();
        let sin: Vec<f32> = angles.iter().map(|angle| angle.sin()).collect();
        let back_sin: Vec<f32> = sin.iter().map(|value| -value).collect();

        rotate_rows(&mut even, &mut odd, &cos, &sin).expect("table tiles four rows of three pairs");
        rotate_rows(&mut even, &mut odd, &cos, &back_sin).expect("same table");

        let worst = even
            .iter()
            .zip(&original_even)
            .chain(odd.iter().zip(&original_odd))
            .map(|(rotated, original)| (rotated - original).abs())
            .fold(0.0_f32, f32::max);
        assert!(worst < 1e-5, "worst {worst}");
    }

    #[test]
    fn a_table_that_does_not_tile_the_rows_is_refused() {
        let mut even = [0.0_f32; 5];
        let mut odd = [0.0_f32; 5];

        assert!(rotate_rows(&mut even, &mut odd, &[1.0, 1.0], &[0.0, 0.0]).is_none());
        assert!(rotate_rows(&mut even, &mut odd, &[], &[]).is_none());
        assert!(rotate_rows(&mut even, &mut odd, &[1.0], &[0.0, 0.0]).is_none());
    }

    /// The table [`build_position_inputs`] builds for a delta, with
    /// the sliding-pattern family's full-layer frequency factors (`[1.0]*64 + [1e30]*192`),
    /// rotates a key from position 3000 to position 2500 to the same row a
    /// direct table at position 2500 gives, within f32.
    #[test]
    fn the_models_own_table_at_a_delta_moves_a_gemma4_full_layer_key_to_its_new_position() {
        let head_dim = 512_u32;
        let pairs = head_dim as usize / 2;
        let mut factors = vec![1.0_f32; 64];
        factors.extend(vec![1.0e30_f32; 192]);
        let table = |position: usize| {
            build_position_inputs(
                &[0],
                position,
                head_dim,
                1.0e6,
                1e-6,
                Some(&factors),
                RopeScaling::None,
            )
        };
        let unrotated_even: Vec<f32> = (0..pairs)
            .map(|pair| ((pair * 7 % 13) as f32 - 6.0) * 0.1)
            .collect();
        let unrotated_odd: Vec<f32> = (0..pairs)
            .map(|pair| ((pair * 5 % 11) as f32 - 5.0) * 0.1)
            .collect();
        let at_old = table(3000);
        let at_new = table(2500);
        let toward_new = table(500);
        let negative_sin: Vec<f32> = toward_new.sin.iter().map(|value| -value).collect();
        let (mut moved_even, mut moved_odd) = (unrotated_even.clone(), unrotated_odd.clone());
        rotate_rows(&mut moved_even, &mut moved_odd, &at_old.cos, &at_old.sin).expect("tiles");
        let (mut fresh_even, mut fresh_odd) = (unrotated_even, unrotated_odd);
        rotate_rows(&mut fresh_even, &mut fresh_odd, &at_new.cos, &at_new.sin).expect("tiles");

        rotate_rows(
            &mut moved_even,
            &mut moved_odd,
            &toward_new.cos,
            &negative_sin,
        )
        .expect("tiles");

        let worst = moved_even
            .iter()
            .zip(&fresh_even)
            .chain(moved_odd.iter().zip(&fresh_odd))
            .map(|(moved, fresh)| (moved - fresh).abs())
            .fold(0.0_f32, f32::max);
        assert!(worst < 5e-4, "worst {worst}");
    }

    /// Worked example, by hand. Planned checkpoint positions 100, 300, 450,
    /// 600, 900; runs at new 300..450 and new 700..850. The first gap
    /// prefills to 100, 300 (the planned ones up to the run's start, which is
    /// itself planned). 450 is the first run's end, kept for the next gap.
    /// The second gap holds 450, 600 and then 700, the second run's start,
    /// which is not planned and is added as a stop. 900 lies behind the last
    /// run. Nothing lies strictly inside a run here; 320 would be dropped.
    #[test]
    fn checkpoint_positions_split_around_the_runs_and_those_inside_a_run_are_dropped() {
        let runs = [
            ChunkRun {
                old_start: 1000,
                new_start: 300,
                len: 150,
            },
            ChunkRun {
                old_start: 1400,
                new_start: 700,
                len: 150,
            },
        ];

        let (gaps, behind) = split_stops(&[100, 300, 320, 450, 600, 900], &runs);

        assert_eq!(gaps[0], (vec![100, 300], vec![100, 300]));
        assert_eq!(gaps[1], (vec![450, 600], vec![450, 600, 700]));
        assert_eq!(behind, vec![900]);
    }

    #[test]
    fn with_no_runs_every_position_stays_behind_for_the_one_prefill() {
        let (gaps, behind) = split_stops(&[256, 512], &[]);

        assert!(gaps.is_empty());
        assert_eq!(behind, vec![256, 512]);
    }

    const WINDOW: usize = 8;
    const SLACK: usize = 4;
    const EVEN_ODD_ROW: usize = 4;
    const V_ROW: usize = 3;

    fn marker(position: usize) -> f32 {
        position as f32 + 1.0
    }

    fn fill(cache: &mut LayerCache, positions: usize) {
        (0..positions).for_each(|position| {
            let value = marker(position);
            cache.append_at(
                position,
                &[value; EVEN_ODD_ROW],
                &[value * 0.5; EVEN_ODD_ROW],
                &[value; V_ROW],
            );
        });
    }

    fn state(stored_len: usize) -> PrefixState {
        let ring = KvRing::new(WINDOW, SLACK, EVEN_ODD_ROW, V_ROW, 0);
        let mut ring_layer = LayerCache::ring(ring, stored_len);
        fill(&mut ring_layer, stored_len);
        let mut full_layer = LayerCache::new();
        fill(&mut full_layer, stored_len);
        PrefixState {
            ids: (0..stored_len as u32).collect(),
            layer_caches: vec![
                LayerCacheState::Attention(ring_layer),
                LayerCacheState::Attention(full_layer),
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

    fn identity_rotations() -> Vec<Option<(Vec<f32>, Vec<f32>)>> {
        let table = Some((vec![1.0; EVEN_ODD_ROW], vec![0.0; EVEN_ODD_ROW]));
        vec![table.clone(), table, None]
    }

    fn ring_key(state: &PrefixState, position: usize) -> f32 {
        match &state.layer_caches[0] {
            LayerCacheState::Attention(cache) => {
                let ring = cache.ring_geometry().expect("layer 0 is a ring");
                cache.k_even[(position % ring.capacity) * EVEN_ODD_ROW]
            }
            _ => 0.0,
        }
    }

    fn full_rows(state: &PrefixState) -> usize {
        match &state.layer_caches[1] {
            LayerCacheState::Attention(cache) => cache.k_even.len() / EVEN_ODD_ROW,
            _ => 0,
        }
    }

    /// Worked example, by hand. An entry of 30 tokens, a window of 8 and a
    /// slack of 4 (capacity 12), so its ring holds positions 18..30. A run of
    /// 10 tokens at stored 14..24 would need its last 8 rows, positions
    /// 16..24: position 16 is already overwritten (it needs `16 + 12 >= 30`,
    /// which is false), so the run is not live. A run at stored 20..30 needs
    /// 22..30: `22 + 12 >= 30`, live.
    #[test]
    fn a_run_is_live_only_while_the_ring_still_holds_its_last_window() {
        let ring = KvRing::new(WINDOW, SLACK, EVEN_ODD_ROW, V_ROW, 0);
        let older = ChunkRun {
            old_start: 14,
            new_start: 3,
            len: 10,
        };
        let newest = ChunkRun {
            old_start: 20,
            new_start: 3,
            len: 10,
        };
        let short_older = ChunkRun {
            old_start: 17,
            new_start: 3,
            len: 3,
        };
        let restored_over = ChunkRun {
            old_start: 18,
            new_start: 3,
            len: 4,
        };

        assert!(!ring_rows_live(&ring, &older, 30, 0));
        assert!(ring_rows_live(&ring, &newest, 30, 0));
        assert!(
            !ring_rows_live(&ring, &short_older, 30, 0),
            "rows 17..20 are gone: 17 + 12 < 30"
        );
        assert!(ring_rows_live(&ring, &newest, 30, 29));
        assert!(ring_rows_live(&ring, &newest, 30, 30));
        assert!(
            !ring_rows_live(&ring, &restored_over, 30, 30),
            "a restore at 30 replaced every row before 22"
        );
    }

    /// Moving the stored tokens 20..30 of a 30-token entry to the end of a
    /// 9-token state: the full layer holds the 9 rows then the 10 moved ones
    /// in order, the ring holds the moved chunk's last window at the slots of
    /// its new positions (9 + 10 = 19, window 8: positions 11..19), and the
    /// ids and length follow.
    #[test]
    fn a_lifted_run_lands_at_its_new_position_in_every_layer() {
        let source = state(30);
        let run = ChunkRun {
            old_start: 20,
            new_start: 9,
            len: 10,
        };
        let moved = extract_run(&source, run, &widths(), &identity_rotations())
            .expect("every layer has rows and a rotation");
        let mut target = state(9);
        let mut entry = CacheEntry::new(
            PrefixState {
                ids: target.ids.clone(),
                layer_caches: core::mem::take(&mut target.layer_caches),
                cached_len: 9,
            },
            CacheKey::of(&ServingConfig::default(), false, RopeScaling::None, 0, 0),
        );

        entry
            .apply_moved(&moved)
            .expect("the state is at the run start");

        assert_eq!(entry.state.cached_len, 19);
        assert_eq!(entry.state.ids.len(), 19);
        assert_eq!(entry.state.ids[9..], source.ids[20..30]);
        assert_eq!(full_rows(&entry.state), 19);
        let LayerCacheState::Attention(full) = &entry.state.layer_caches[1] else {
            panic!("layer 1 is attention");
        };
        let moved_keys: Vec<f32> = (9..19).map(|row| full.k_even[row * EVEN_ODD_ROW]).collect();
        let source_keys: Vec<f32> = (20..30).map(marker).collect();
        assert_eq!(moved_keys, source_keys);
        assert!(
            (11..19).all(|position| ring_key(&entry.state, position) == marker(position + 11)),
            "ring slot of new position p holds the key stored at p + 11"
        );
        assert_eq!(entry.restored_at, 19);
    }

    #[test]
    fn a_run_is_refused_when_the_state_is_not_at_its_start() {
        let source = state(30);
        let run = ChunkRun {
            old_start: 20,
            new_start: 9,
            len: 10,
        };
        let moved = extract_run(&source, run, &widths(), &identity_rotations()).expect("lifted");
        let mut wrong = state(12);

        let refusal = wrong.append_moved(&moved);

        assert_eq!(refusal, Err("the state is not at the start of the run"));
    }

    #[test]
    fn a_layer_without_a_rotation_makes_the_run_unliftable() {
        let source = state(30);
        let run = ChunkRun {
            old_start: 20,
            new_start: 9,
            len: 10,
        };
        let mut rotations = identity_rotations();
        rotations[1] = None;

        assert!(extract_run(&source, run, &widths(), &rotations).is_none());
    }

    #[test]
    fn a_recurrent_layer_blocks_any_move() {
        let mut recurrent = state(30);
        recurrent
            .layer_caches
            .push(LayerCacheState::Ssm(SsmLayerCache::new(4, 4)));

        assert!(state(30).rows_are_movable());
        assert!(!recurrent.rows_are_movable());
    }
}
