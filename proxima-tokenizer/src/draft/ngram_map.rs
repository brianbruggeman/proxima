//! A faithful port of llama.cpp's `common_ngram_map` (`common/ngram-map.h`,
//! `common/ngram-map.cpp:121-536`): a stateful n-gram-to-m-gram drafter that,
//! unlike [`crate::draft::ngram_simple`], carries an index across calls
//! (`begin`/`draft`/`accept`) instead of rescanning the whole history every
//! step. `key_only = true` gives llama.cpp's `ngram-map-k` speculation type;
//! `key_only = false` gives `ngram-map-k4v` -- the SAME function in the
//! incumbent (`common_ngram_map_draft`'s own `if (map.key_only) { .. return;
//! }` branch), so this module keeps them one port rather than two, per
//! [`crate::draft`]'s own RISC-reuse framing.
//!
//! # The algorithm, traced to the incumbent
//!
//! [`NgramMap`] mirrors `common_ngram_map` field for field: a growable
//! `keys` list (llama.cpp's `std::vector<common_ngram_map_key>`), a fixed-size
//! hash index `key_map` (llama.cpp's `COMMON_NGRAM_HASH_MAP_SIZE`-entry
//! `std::vector<uint32_t>`, allocated once at construction -- see
//! [`NgramMap::new`]'s own doc for why that allocation is legitimate under
//! this crate's zero-alloc discipline), and the incremental bookkeeping
//! (`size_last_begin`, `idx_last_check`, `key_map_last_idx`,
//! `last_draft_*`) llama.cpp's `common_ngram_map_draft` reads and writes every
//! call.
//!
//! [`ngram_map_begin`] (`common_ngram_map_begin`) is called once per
//! generation with the prompt tokens seen so far; it is the ONLY place stale
//! keys/hashes get pruned (a context that shrank -- e.g. a removed
//! reasoning block -- forces this).
//!
//! [`ngram_map_draft`] (`common_ngram_map_draft`) builds the key n-gram from
//! `history`'s trailing `size_key - 1` tokens plus `sampled` (the same
//! zero-allocation trick [`crate::draft::ngram_simple::ngram_simple_draft`]
//! uses: compare `history`'s own slices directly, never materialize the
//! pattern into its own buffer) and looks for an earlier occurrence three
//! ways, in this exact order: (1) `key_map`'s hash entry for this n-gram,
//! (2) a descending linear scan over the region already covered by
//! `begin()`'s prompt (`size_last_begin`) that `key_map` has not yet
//! indexed, (3) a descending linear scan over tokens generated since
//! `begin()`. Every call also extends `key_map` with hashes for any
//! n-grams in either region it has not yet indexed -- this happens whether
//! or not a match was found, exactly matching the incumbent's own
//! unconditional index-maintenance step.
//!
//! On a match, the matched key n-gram is looked up (or created) in `keys`
//! by CONTENT, not by position -- llama.cpp's own linear scan over
//! `map.keys` comparing token content at each key's `key_idx`. `key_only`
//! mode then drafts directly from `values[0]`'s length cap
//! (`n_accepted`, set by [`ngram_map_accept`] after the PREVIOUS draft using
//! this key). `key4v` mode instead tallies up to
//! [`MAX_VALUES`] distinct m-gram continuations seen after this key since
//! `stat_idx` (llama.cpp's own value-slot content comparison, not position
//! comparison), then applies the tie guard `sum_occur > 0 && max_occur < 2 *
//! sum_occur` (`common/ngram-map.cpp:495-499`): if no single continuation
//! clearly dominates, no draft at all -- this is the ONLY place `ngram-map-k`
//! and `ngram-map-k4v` can produce different output for the same input
//! stream (`README.md`'s own "ngram-map key_only vs k4v divergence" section
//! documents the narrow window this fires in and how the fixture streams
//! force it).
//!
//! [`ngram_map_accept`] (`common_ngram_map_accept`) records how many of the
//! just-drafted tokens the target model actually accepted, capping the NEXT
//! draft from that same value slot to that length -- llama.cpp's own adaptive
//! shrink-on-miss behaviour.

use alloc::vec::Vec;

/// llama.cpp's `COMMON_NGRAM_MAX_VALUES` (`common/ngram-map.h:39`): the number of
/// distinct m-gram continuations tracked per key n-gram in `key4v` mode.
/// `key_only` mode only ever populates slot `0`.
pub const MAX_VALUES: usize = 4;

/// llama.cpp's `COMMON_NGRAM_HASH_MAP_SIZE` (`common/ngram-map.h:42`): the fixed
/// entry count of [`NgramMap`]'s hash index, allocated once at
/// [`NgramMap::new`] and never resized.
const HASH_MAP_SIZE: usize = 262_144;

/// llama.cpp's `COMMON_NGRAM_MAX_VALUE_COUNT` (`common/ngram-map.cpp:119`): the
/// saturation cap on both a key's hit count and a value's occurrence count.
const MAX_VALUE_COUNT: u16 = 16_380;

/// llama.cpp's `LCG_FACTOR` (`common/ngram-map.cpp:11`): the 32-bit LCG
/// multiplier `common_ngram_map_hash` folds every token through.
const LCG_FACTOR: u32 = 2_654_435_761;

/// llama.cpp's `ngram-map` default `size_key` (`README.md`'s own recorded
/// default, `common_params_speculative_ngram_map::size_key`).
pub const DEFAULT_SIZE_KEY: u16 = 12;

/// llama.cpp's `ngram-map` default `size_value`.
pub const DEFAULT_SIZE_VALUE: u16 = 48;

/// llama.cpp's `ngram-map` default `min_hits`.
pub const DEFAULT_MIN_HITS: u16 = 1;

/// llama.cpp's `common_ngram_map` constructor arguments (`common/ngram-map.h:72-77`):
/// `size_key`/`size_value` name the n-gram/m-gram sizes (llama.cpp's own
/// `size_key`/`size_value` fields, not `size_ngram`/`size_mgram` --
/// `ngram-map` and `ngram-simple` use different field names for the same
/// role upstream), `key_only` selects `ngram-map-k` (`true`) vs
/// `ngram-map-k4v` (`false`), and `min_hits` is the minimum key occurrence
/// count `key4v` mode requires before it will examine values at all
/// (`key_only` mode ignores it entirely, matching the incumbent).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramMapConfig {
    /// Size of the key n-gram looked up in history -- llama.cpp's `size_key`.
    pub size_key: u16,
    /// Size of the value m-gram drafted after a key match -- llama.cpp's `size_value`.
    pub size_value: u16,
    /// `true` for `ngram-map-k` (drafts from `values[0]` unconditionally on
    /// a key match), `false` for `ngram-map-k4v` (tallies up to
    /// [`MAX_VALUES`] continuations and applies the dominance guard).
    pub key_only: bool,
    /// Minimum key hit count before `key4v` mode will draft -- ignored in
    /// `key_only` mode.
    pub min_hits: u16,
}

/// llama.cpp's `common_ngram_map_value` (`common/ngram-map.h:45-49`): one
/// tracked continuation after a key n-gram. `value_idx == 0` is the
/// incumbent's own sentinel for "unused slot" (position `0` in the token
/// history can never be a legitimate match position either, since every
/// search here -- like [`crate::draft::ngram_simple`]'s own -- excludes it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NgramMapValue {
    value_idx: usize,
    value_num: u16,
    n_accepted: u16,
}

const EMPTY_VALUE: NgramMapValue = NgramMapValue {
    value_idx: 0,
    value_num: 0,
    n_accepted: 0,
};

/// llama.cpp's `common_ngram_map_key` (`common/ngram-map.h:52-58`): one tracked
/// key n-gram, identified by content (compared against `history` at
/// `key_idx` on every lookup, never by `key_idx` equality itself, since a
/// key's first-seen position is not stable identity -- only its content is).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NgramMapKey {
    key_idx: usize,
    stat_idx: usize,
    key_num: u16,
    values: [NgramMapValue; MAX_VALUES],
}

/// A faithful port of `common_ngram_map` (`common/ngram-map.h:61-95`).
/// Constructed once per generation stream; [`ngram_map_begin`],
/// [`ngram_map_draft`], and [`ngram_map_accept`] are llama.cpp's own
/// `common_ngram_map_begin`/`_draft`/`_accept`, kept as free functions over
/// `&mut NgramMap` rather than methods so the module reads the same shape as
/// the incumbent's own three-function API.
#[derive(Debug, Clone)]
pub struct NgramMap {
    config: NgramMapConfig,
    keys: Vec<NgramMapKey>,
    key_map: Vec<u32>,
    key_map_last_idx: u32,
    size_last_begin: usize,
    idx_last_check: usize,
    last_draft_created: bool,
    last_draft_key_idx: usize,
    last_draft_value_idx: usize,
}

impl NgramMap {
    /// Allocates both `key_map` (the fixed-size `HASH_MAP_SIZE`-entry hash
    /// index, llama.cpp's own `key_map.resize(COMMON_NGRAM_HASH_MAP_SIZE)`) and
    /// `keys` up front, at construction, so [`ngram_map_draft`]'s hot path
    /// allocates nothing -- this crate's zero-per-call-allocation discipline
    /// ([`crate::draft::ngram_simple::ngram_simple_draft`]'s own doc).
    ///
    /// `max_context_len` bounds `keys`' pre-allocated capacity: every key's
    /// `key_idx` is a distinct token position in `inp`
    /// ([`ngram_map_draft`]'s own match-position search), so across the
    /// lifetime of one `NgramMap` no more distinct keys can ever exist than
    /// there are positions in the longest `inp` this instance will ever see
    /// -- the caller's model context window. Sizing `keys` to that bound
    /// up front (`Vec::with_capacity`) makes every later `push` (a genuinely
    /// new key n-gram, `ngram_map_draft`'s `key_offset == map.keys.len()`
    /// branch) capacity-neutral: the incumbent's own `std::vector` grows
    /// amortized and unbounded because C++ has no equivalent up-front
    /// caller-supplied bound to size against; this port has one, so it uses
    /// it, without changing which keys are found or drafted (`keys`'
    /// capacity is never observable in the drafted output, only in whether
    /// filling it reallocates).
    #[must_use]
    pub fn new(config: NgramMapConfig, max_context_len: usize) -> Self {
        Self {
            config,
            keys: Vec::with_capacity(max_context_len),
            key_map: alloc::vec![0u32; HASH_MAP_SIZE],
            key_map_last_idx: 0,
            size_last_begin: 0,
            idx_last_check: 0,
            last_draft_created: false,
            last_draft_key_idx: 0,
            last_draft_value_idx: 0,
        }
    }
}

fn hash_key_tokens(tail: &[u32], sampled: u32) -> u32 {
    let mut hash = 0u32;
    for &token in tail {
        hash = hash.wrapping_mul(LCG_FACTOR).wrapping_add(token);
    }
    hash.wrapping_mul(LCG_FACTOR).wrapping_add(sampled)
}

fn hash_ngram(inp: &[u32], start: usize, len: usize) -> u32 {
    let mut hash = 0u32;
    for &token in &inp[start..start + len] {
        hash = hash.wrapping_mul(LCG_FACTOR).wrapping_add(token);
    }
    hash
}

/// A faithful port of `common_ngram_map_begin` (`common/ngram-map.cpp:121-219`)
/// -- see this module's own doc for the algorithm. `tokens` is llama.cpp's
/// `tokens` argument: the prompt (or reasoning-trimmed history) this
/// generation stream is starting from.
pub fn ngram_map_begin(map: &mut NgramMap, tokens: &[u32]) {
    let size_begin = tokens.len();
    let size_key = usize::from(map.config.size_key);
    let size_value = usize::from(map.config.size_value);

    let mut idx_begin_cleanup = map.size_last_begin;
    if idx_begin_cleanup > size_begin {
        idx_begin_cleanup = if size_begin > size_key + size_value {
            size_begin - size_key - size_value
        } else {
            0
        };
    }

    if !map.key_map.is_empty() && size_begin < map.idx_last_check {
        for entry in &mut map.key_map {
            if *entry != 0 && (*entry as usize) >= idx_begin_cleanup {
                *entry = 0;
            }
        }
        map.key_map_last_idx = if idx_begin_cleanup > 0 {
            (idx_begin_cleanup - 1) as u32
        } else {
            0
        };
    }

    if size_begin < map.idx_last_check && !map.keys.is_empty() {
        let key_only = map.config.key_only;
        map.keys.retain_mut(|key| {
            if key.key_idx >= idx_begin_cleanup {
                return false;
            }
            if key_only {
                return true;
            }
            prune_stale_values(key, idx_begin_cleanup);
            key.values[0].value_idx != 0
        });
    }

    map.idx_last_check = size_begin;
    map.size_last_begin = size_begin;
}

/// llama.cpp's stale-value shift loop inside `common_ngram_map_begin`
/// (`common/ngram-map.cpp:189-203`): clears any value slot whose position
/// fell in the pruned region, compacting the surviving slots to the front
/// exactly as the incumbent's own hand-written array shift does.
fn prune_stale_values(key: &mut NgramMapKey, idx_begin_cleanup: usize) {
    for slot in (0..MAX_VALUES).rev() {
        if key.values[slot].value_idx != 0 && key.values[slot].value_idx >= idx_begin_cleanup {
            for shift in slot..MAX_VALUES - 1 {
                key.values[shift] = key.values[shift + 1];
            }
            key.values[MAX_VALUES - 1].value_idx = 0;
            key.values[MAX_VALUES - 1].value_num = 0;
        }
    }
}

/// A faithful port of `common_ngram_map_draft` (`common/ngram-map.cpp:221-516`)
/// -- see this module's own doc for the algorithm, traced to the incumbent.
/// `inp` is llama.cpp's `inp` (every token generated so far, NOT including
/// `sampled`); `sampled` is the token the caller's own sampler just drew for
/// the position immediately after `inp`. `out` follows this crate's
/// caller-owned buffer discipline
/// ([`crate::draft::ngram_simple::ngram_simple_draft`]'s own doc): cleared
/// then filled, allocating nothing on this call once `out`'s capacity
/// already covers the draft length.
///
/// Requires `map.config.size_key >= 1` (matching the incumbent, which never
/// guards against `size_key == 0` either -- llama.cpp's own `ngram-map`
/// speculation type is never configured that way).
pub fn ngram_map_draft(map: &mut NgramMap, inp: &[u32], sampled: u32, out: &mut Vec<u32>) {
    out.clear();
    map.last_draft_created = false;
    map.last_draft_key_idx = 0;
    map.last_draft_value_idx = 0;

    let cur_len = inp.len();
    let size_key = usize::from(map.config.size_key);
    let size_value = usize::from(map.config.size_value);

    // llama.cpp: `if (cur_len < 2 * n + m) return;`
    if cur_len < 2 * size_key + size_value {
        return;
    }

    map.idx_last_check = cur_len;

    let tail = &inp[cur_len - size_key + 1..cur_len];
    let key_matches_at = |inp: &[u32], pos: usize| -> bool {
        inp[pos..pos + size_key - 1] == *tail && inp[pos + size_key - 1] == sampled
    };

    let region_bound = cur_len - size_key - size_value - 1;
    let mut match_pos = 0usize;

    if !map.key_map.is_empty() {
        let hash = hash_key_tokens(tail, sampled);
        let idx_hash = (hash as usize) % map.key_map.len();
        let idx_key = map.key_map[idx_hash] as usize;
        if idx_key != 0 && idx_key < region_bound && key_matches_at(inp, idx_key) {
            match_pos = idx_key;
        }
    }

    if match_pos == 0 && map.size_last_begin > size_key + size_value + 1 {
        let upper = map.size_last_begin - size_key - size_value - 1;
        let mut candidate = upper;
        while candidate > map.key_map_last_idx as usize {
            if key_matches_at(inp, candidate) {
                match_pos = candidate;
                break;
            }
            candidate -= 1;
        }
    }

    if match_pos == 0 {
        let lower = map.size_last_begin.max(map.key_map_last_idx as usize);
        let mut candidate = region_bound;
        while candidate > lower {
            if key_matches_at(inp, candidate) {
                match_pos = candidate;
                break;
            }
            candidate -= 1;
        }
    }

    if !map.key_map.is_empty() {
        extend_key_map_index(map, inp, cur_len, size_key, size_value);
    }

    if match_pos == 0 {
        return;
    }

    let mut key_offset = map.keys.len();
    for (index, key) in map.keys.iter().enumerate() {
        if key_matches_at(inp, key.key_idx) {
            key_offset = index;
            break;
        }
    }
    if key_offset == map.keys.len() {
        map.keys.push(NgramMapKey {
            key_idx: match_pos,
            stat_idx: 0,
            key_num: 0,
            values: [NgramMapValue {
                n_accepted: size_value as u16,
                ..EMPTY_VALUE
            }; MAX_VALUES],
        });
    }

    map.keys[key_offset].key_num = map.keys[key_offset]
        .key_num
        .saturating_add(1)
        .min(MAX_VALUE_COUNT);

    if map.config.key_only {
        draft_key_only(map, key_offset, inp, match_pos, size_key, size_value, out);
        return;
    }

    if map.keys[key_offset].key_num < map.config.min_hits {
        return;
    }

    draft_k4v(map, key_offset, inp, match_pos, size_key, size_value, out);
}

/// llama.cpp's key_map index-maintenance step (`common/ngram-map.cpp:318-340`):
/// runs unconditionally, whether or not a match was found this call, over
/// both the prompt region and the generated region -- see this module's own
/// doc for why.
fn extend_key_map_index(
    map: &mut NgramMap,
    inp: &[u32],
    cur_len: usize,
    size_key: usize,
    size_value: usize,
) {
    if map.size_last_begin > size_key + size_value + 1 {
        let upper = map.size_last_begin - size_key - size_value - 1;
        let mut candidate = upper;
        while candidate > map.key_map_last_idx as usize {
            index_ngram_at(map, inp, candidate, size_key);
            candidate -= 1;
        }
    }

    let lower = map.size_last_begin.max(map.key_map_last_idx as usize);
    let region_bound = cur_len - size_key - size_value - 1;
    let mut candidate = region_bound;
    while candidate > lower {
        index_ngram_at(map, inp, candidate, size_key);
        candidate -= 1;
    }

    map.key_map_last_idx = map.key_map_last_idx.max(region_bound as u32);
}

fn index_ngram_at(map: &mut NgramMap, inp: &[u32], position: usize, size_key: usize) {
    let hash = hash_ngram(inp, position, size_key);
    let idx_hash = (hash as usize) % map.key_map.len();
    if map.key_map[idx_hash] == 0 {
        map.key_map[idx_hash] = position as u32;
    }
}

/// llama.cpp's `key_only` draft branch (`common/ngram-map.cpp:382-398`): always
/// drafts from `values[0]`, length-capped by that slot's own `n_accepted`
/// from the PREVIOUS [`ngram_map_accept`] call.
fn draft_key_only(
    map: &mut NgramMap,
    key_offset: usize,
    inp: &[u32],
    match_pos: usize,
    size_key: usize,
    size_value: usize,
    out: &mut Vec<u32>,
) {
    let n_draft = usize::from(map.keys[key_offset].values[0].n_accepted).min(size_value);
    out.extend_from_slice(&inp[match_pos + size_key..match_pos + size_key + n_draft]);
    map.last_draft_created = true;
    map.last_draft_key_idx = key_offset;
    map.last_draft_value_idx = 0;
}

/// llama.cpp's `key4v` draft branch (`common/ngram-map.cpp:408-515`): tallies up
/// to [`MAX_VALUES`] distinct continuations seen after this key since
/// `stat_idx`, then drafts from the most frequent one UNLESS the dominance
/// guard `sum_occur > 0 && max_occur < 2 * sum_occur` fires -- this module's
/// own doc names the exact narrow window that guard can diverge from
/// `key_only` in.
fn draft_k4v(
    map: &mut NgramMap,
    key_offset: usize,
    inp: &[u32],
    match_pos: usize,
    size_key: usize,
    size_value: usize,
    out: &mut Vec<u32>,
) {
    let stat_idx = map.keys[key_offset].stat_idx;
    for position in stat_idx..=match_pos {
        if inp[position..position + size_key - 1]
            != inp[match_pos..match_pos + size_key - 1]
            || inp[position + size_key - 1] != inp[match_pos + size_key - 1]
        {
            continue;
        }

        let idx_begin_value_key = position + size_key;
        let mut idx_value: Option<usize> = None;
        for slot in 0..MAX_VALUES {
            let value_idx_v = map.keys[key_offset].values[slot].value_idx;
            if value_idx_v == 0 {
                map.keys[key_offset].values[slot] = NgramMapValue {
                    value_idx: idx_begin_value_key,
                    value_num: 0,
                    n_accepted: size_value as u16,
                };
                idx_value = Some(slot);
                break;
            }
            if inp[idx_begin_value_key..idx_begin_value_key + size_value]
                == inp[value_idx_v..value_idx_v + size_value]
            {
                idx_value = Some(slot);
                break;
            }
        }
        if let Some(slot) = idx_value {
            let value = &mut map.keys[key_offset].values[slot];
            value.value_num = value.value_num.saturating_add(1).min(MAX_VALUE_COUNT);
        }
    }
    map.keys[key_offset].stat_idx = match_pos;

    let values = map.keys[key_offset].values;
    let mut max_occur = 0u16;
    let mut slot_max = 0usize;
    for (index, value) in values.iter().enumerate() {
        if value.value_num > max_occur {
            max_occur = value.value_num;
            slot_max = index;
        }
    }
    let sum_occur: u32 = values
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != slot_max)
        .map(|(_, value)| u32::from(value.value_num))
        .sum();

    if sum_occur > 0 && u32::from(max_occur) < 2 * sum_occur {
        return;
    }

    let n_draft = usize::from(values[slot_max].n_accepted).min(size_value);
    out.extend_from_slice(&inp[match_pos + size_key..match_pos + size_key + n_draft]);
    map.last_draft_created = true;
    map.last_draft_key_idx = key_offset;
    map.last_draft_value_idx = slot_max;
}

/// A faithful port of `common_ngram_map_accept` (`common/ngram-map.cpp:518-536`):
/// records how many of the just-drafted tokens the target model actually
/// accepted, so the NEXT draft from this same key/value pair is capped to
/// that length. A no-op when the previous [`ngram_map_draft`] call produced
/// no draft (`map.last_draft_created == false`), matching the incumbent.
pub fn ngram_map_accept(map: &mut NgramMap, n_accepted: u16) {
    if !map.last_draft_created {
        return;
    }
    let key_idx = map.last_draft_key_idx;
    let value_idx = map.last_draft_value_idx;
    map.keys[key_idx].values[value_idx].n_accepted = n_accepted;
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::collections::BTreeMap;
    use alloc::vec;
    use alloc::vec::Vec;

    use serde::Deserialize;

    use super::{NgramMap, NgramMapConfig, ngram_map_accept, ngram_map_begin, ngram_map_draft};

    #[derive(Debug, Deserialize)]
    struct StreamsFile {
        streams: Vec<Stream>,
    }

    #[derive(Debug, Deserialize)]
    struct Stream {
        id: u32,
        tokens: Vec<u32>,
    }

    #[derive(Debug, Deserialize)]
    struct FixtureFile {
        params: FixtureParams,
        cases: Vec<FixtureCase>,
    }

    #[derive(Debug, Deserialize)]
    struct FixtureParams {
        size_key: u16,
        size_value: u16,
        min_hits: u16,
    }

    #[derive(Debug, Deserialize)]
    struct FixtureCase {
        stream_id: u32,
        position: usize,
        sampled: u32,
        draft: Vec<u32>,
        accepted: u16,
    }

    const STREAMS_JSON: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/llama-ngram/fixtures/streams.json"
    ));
    const NGRAM_MAP_K_JSON: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/llama-ngram/fixtures/ngram_map_k.json"
    ));
    const NGRAM_MAP_K4V_JSON: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/llama-ngram/fixtures/ngram_map_k4v.json"
    ));

    /// Replays one fixture file's cases against a fresh [`NgramMap`] per
    /// stream, exactly following the generator's own protocol
    /// (`tests/fixtures/llama-ngram/generator/main.cpp`'s `replay_ngram_map`,
    /// `README.md`'s schema note): one `begin()` per stream over
    /// `tokens[..prompt_len]` where `prompt_len = min(2*size_key +
    /// size_value + 2, tokens.len() / 2)`, then `draft`/`accept` per case in
    /// position order, feeding each case's own recorded `accepted` count
    /// back into `accept` before the next case -- never re-deriving
    /// `accepted` independently, since the fixture already recorded exactly
    /// what the generator fed back (guiding principle 14).
    fn run_fixture(fixture_json: &str, key_only: bool) -> (usize, usize) {
        let streams: StreamsFile = serde_json::from_str(STREAMS_JSON).expect("streams.json parses");
        let fixture: FixtureFile = serde_json::from_str(fixture_json).expect("fixture parses");

        let config = NgramMapConfig {
            size_key: fixture.params.size_key,
            size_value: fixture.params.size_value,
            key_only,
            min_hits: fixture.params.min_hits,
        };
        let min_len = 2 * usize::from(config.size_key) + usize::from(config.size_value) + 2;

        let mut cases_by_stream: BTreeMap<u32, Vec<&FixtureCase>> = BTreeMap::new();
        for case in &fixture.cases {
            cases_by_stream.entry(case.stream_id).or_default().push(case);
        }

        let mut total = 0usize;
        let mut non_empty = 0usize;
        let mut drafted: Vec<u32> = Vec::new();

        for (stream_id, cases) in &cases_by_stream {
            let stream = streams
                .streams
                .iter()
                .find(|stream| stream.id == *stream_id)
                .unwrap_or_else(|| panic!("stream {stream_id} exists in streams.json"));

            let mut map = NgramMap::new(config, stream.tokens.len());
            let prompt_len = min_len.min(stream.tokens.len() / 2);
            ngram_map_begin(&mut map, &stream.tokens[..prompt_len]);

            for case in cases {
                let history = &stream.tokens[..case.position];
                ngram_map_draft(&mut map, history, case.sampled, &mut drafted);
                total += 1;
                if !case.draft.is_empty() {
                    non_empty += 1;
                }
                assert_eq!(
                    drafted, case.draft,
                    "stream {stream_id} position {}: got {drafted:?}, want {:?}",
                    case.position, case.draft
                );
                ngram_map_accept(&mut map, case.accepted);
            }
        }

        (total, non_empty)
    }

    /// [`ngram_map_draft`] against every case in
    /// `tests/fixtures/llama-ngram/fixtures/ngram_map_k.json`
    /// (SPEC.md), produced by calling llama.cpp's own
    /// `common_ngram_map_begin`/`_draft`/`_accept` with `key_only = true`
    /// directly.
    #[test]
    fn ngram_map_k_matches_llama_fixture() {
        let (cases, non_empty) = run_fixture(NGRAM_MAP_K_JSON, true);
        println!("cases = {cases} non_empty = {non_empty}");
        assert!(cases >= 200, "fixture must carry at least 200 cases per SPEC.md");
    }

    /// [`ngram_map_draft`] against every case in
    /// `tests/fixtures/llama-ngram/fixtures/ngram_map_k4v.json`
    /// (SPEC.md), `key_only = false`. Includes the 24 cases where
    /// `ngram-map-k` and `ngram-map-k4v` draft differently
    /// (`README.md`'s own "key_only vs k4v divergence" section) -- both
    /// fixtures must pass with the SAME port, differing only in
    /// `NgramMapConfig::key_only`.
    #[test]
    fn ngram_map_k4v_matches_llama_fixture() {
        let (cases, non_empty) = run_fixture(NGRAM_MAP_K4V_JSON, false);
        println!("cases = {cases} non_empty = {non_empty}");
        assert!(cases >= 200, "fixture must carry at least 200 cases per SPEC.md");
    }

    /// Happy path, hand-computed key_only: `history`'s first four tokens are
    /// the `begin()` prompt (`size_last_begin = 4`); the key `[1, 2, 3]` ->
    /// `[4, 5, 6]` first occurs strictly AFTER `size_last_begin` (at index
    /// 5) and recurs at the very end -- `common_ngram_map_draft`'s own
    /// generated-region scan (`common/ngram-map.cpp:298-312`) only searches
    /// positions strictly GREATER than `size_last_begin`, never equal to
    /// it, so a match placed exactly at `size_last_begin` is invisible to
    /// the incumbent too (confirmed against this test's own earlier,
    /// rejected construction, which placed the key at index 4 and drafted
    /// nothing).
    #[test]
    fn happy_path_key_only_drafts_the_earlier_occurrences_continuation() {
        let config = NgramMapConfig {
            size_key: 3,
            size_value: 3,
            key_only: true,
            min_hits: 1,
        };
        // history: [0, 0, 0, 0, 9, 1, 2, 3, 4, 5, 6, 9, 9, 1, 2] (len 15).
        // key [1, 2, 3] first occurs at index 5, continuation [4, 5, 6];
        // the trailing [1, 2] + sampled 3 recurs that same key.
        let history = vec![0u32, 0, 0, 0, 9, 1, 2, 3, 4, 5, 6, 9, 9, 1, 2];
        let mut map = NgramMap::new(config, history.len());
        ngram_map_begin(&mut map, &history[..4]);
        let mut drafted = Vec::new();
        ngram_map_draft(&mut map, &history, 3, &mut drafted);
        assert_eq!(
            drafted,
            vec![4u32, 5, 6],
            "key [1, 2, 3] recurs at index 5, whose continuation is [4, 5, 6]"
        );
    }

    /// Sad path: `history` shorter than `2 * size_key + size_value` must
    /// draft nothing, regardless of content.
    #[test]
    fn history_too_short_drafts_nothing() {
        let config = NgramMapConfig {
            size_key: 12,
            size_value: 48,
            key_only: true,
            min_hits: 1,
        };
        let history: Vec<u32> = (0..60u32).collect();
        let mut map = NgramMap::new(config, history.len());
        assert!(history.len() < 2 * 12 + 48);
        ngram_map_begin(&mut map, &history[..30]);
        let mut drafted = Vec::new();
        ngram_map_draft(&mut map, &history, 999, &mut drafted);
        assert_eq!(drafted, Vec::<u32>::new());
    }

    /// Sad path, k4v: a key recurring with two continuations that never
    /// break the `2 * sum_occur` dominance guard (each continuation seen
    /// exactly once, tied) must draft nothing on the tie, unlike
    /// `key_only`, which would draft `values[0]` regardless.
    #[test]
    fn k4v_ties_between_continuations_draft_nothing() {
        let config = NgramMapConfig {
            size_key: 3,
            size_value: 2,
            key_only: false,
            min_hits: 1,
        };
        // key [1, 2, 3] occurs at index 1 -> continuation [4, 5]; occurs
        // again at index 8 -> continuation [7, 8]; both tallied once each,
        // so max_occur (1) < 2 * sum_occur (2) on the second recurrence's
        // OWN scan -- but stat_idx only advances to match_pos, and the
        // draft this call fires from is the SECOND occurrence's match,
        // scanning stat_idx=0..=match_pos=8 which tallies BOTH
        // occurrences at once (this call is the key's first draft
        // attempt, key_num was 0 before this call's increment so min_hits
        // is satisfied at key_num=1).
        let history = vec![0u32, 1, 2, 3, 4, 5, 9, 9, 1, 2, 3, 7, 8, 1, 2];
        let mut map = NgramMap::new(config, history.len());
        ngram_map_begin(&mut map, &history[..2]);
        let mut drafted = Vec::new();
        // sampled completes the key [1, 2, 3] at the trailing window
        // history[13..15] = [1, 2], sampled = 3.
        ngram_map_draft(&mut map, &history, 3, &mut drafted);
        assert_eq!(
            drafted,
            Vec::<u32>::new(),
            "two equally-seen continuations must not draft under the k4v dominance guard"
        );
    }

    /// The buffer-reuse contract every drafter in this crate follows
    /// ([`crate::draft::ngram_simple::ngram_simple_draft`]'s own doc):
    /// calling `ngram_map_draft` with a buffer already holding a PRIOR
    /// draft must leave `out` holding exactly the new draft.
    #[test]
    fn a_reused_buffer_holding_a_stale_draft_is_fully_overwritten() {
        let config = NgramMapConfig {
            size_key: 3,
            size_value: 3,
            key_only: true,
            min_hits: 1,
        };
        let history = vec![0u32, 0, 0, 0, 9, 1, 2, 3, 4, 5, 6, 9, 9, 1, 2];
        let mut map = NgramMap::new(config, history.len());
        ngram_map_begin(&mut map, &history[..4]);
        let mut drafted: Vec<u32> = vec![111, 222, 333, 444, 555];

        ngram_map_draft(&mut map, &history, 3, &mut drafted);
        assert_eq!(drafted, vec![4u32, 5, 6]);

        ngram_map_accept(&mut map, 3);

        ngram_map_draft(&mut map, &history, 9_999, &mut drafted);
        assert_eq!(
            drafted,
            Vec::<u32>::new(),
            "a non-match must leave the buffer empty, not the previous call's draft"
        );
    }
}
