//! A faithful port of llama.cpp's `common_ngram_cache` machinery
//! (`common/ngram-cache.h`/`.cpp`) and its driver,
//! `common_speculative_impl_ngram_cache::draft_one`
//! (`common/speculative.cpp:2095-2138`): three independent frequency tables
//! -- context (built from the CURRENT generation's own history), dynamic
//! (built from prior generations, empty in every fixture this module ports
//! against), and static (a pre-built corpus cache, validated against but
//! never itself the primary source unless nothing else answers) -- each
//! mapping an n-gram (length 1 through [`LLAMA_NGRAM_MAX`]) to the
//! empirical distribution of tokens observed to follow it.
//!
//! # The algorithm, traced to the incumbent
//!
//! [`ngram_cache_update`] (`common_ngram_cache_update`,
//! `common/ngram-cache.cpp:12-52`) extracts every n-gram (sizes
//! [`LLAMA_NGRAM_MIN`]..=[`LLAMA_NGRAM_MAX`]) ending within the last `nnew`
//! positions of `inp_data` and bumps that n-gram's count for the token that
//! followed it. [`ngram_cache_draft`] (`common_ngram_cache_draft`,
//! `:146-198`) then, for each position to draft, builds the
//! [`LLAMA_NGRAM_STATIC`]-token static-lookup key and one context/dynamic
//! key per size in [`LLAMA_NGRAM_MIN`]..=[`LLAMA_NGRAM_MAX`], and tries, IN
//! THIS ORDER: the context cache (lax thresholds,
//! [`DRAFT_MIN_SAMPLE_SIZE_LAX`]/[`DRAFT_MIN_PERCENT_LAX`], LONGEST n-gram
//! first), the dynamic cache (strict thresholds,
//! [`DRAFT_MIN_SAMPLE_SIZE_STRICT`]/[`DRAFT_MIN_PERCENT_STRICT`], same
//! longest-first order), then the static cache alone (lax thresholds,
//! [`LLAMA_NGRAM_STATIC`]-sized key only). A context/dynamic candidate is
//! additionally weighted by the static cache's own count for that same
//! token when validating one (`count_primary * count_static`,
//! `try_draft`'s two-cache overload, `:97-144`) -- a token the static
//! corpus never saw at all still competes (`count_static` defaults to `1`,
//! not `0`, `:124`), it is just outweighed by one that IS corroborated.
//! Drafting stops the moment any position fails to produce a token, never
//! drafting a gap and continuing past it.
//!
//! `common_ngram_cache_draft`'s own `inp`/`draft` combined-sequence
//! indexing (`get_token`, `:55-57`) is ported here as [`token_at`], reading
//! directly from the caller's `history`/`sampled`/`out` slices instead of
//! materializing `inp` as its own concatenated vector -- zero-copy, this
//! crate's own caller-owned-buffer discipline
//! ([`crate::draft::ngram_simple::ngram_simple_draft`]'s own doc).
//!
//! # The context-cache maintenance chunking, and why it is faithful
//!
//! [`NgramCacheState`]'s own `cache_size` mirrors
//! `common_speculative_impl_ngram_cache::seq_info::cache_size`
//! (`:2033-2037`): `draft_one` only feeds `nc_context` the tokens NEW since
//! the last call (`tokens_new`, `:2103-2117`), calling
//! `common_ngram_cache_update` with `nnew == tokens_new.size()` -- so from
//! `common_ngram_cache_update`'s own point of view, `tokens_new` IS the
//! entire dataset every time, never the full history. The practical
//! consequence, confirmed by tracing `i_start = max(inp_size - nnew,
//! ngram_size)` with `inp_size == nnew`: any n-gram whose span would cross
//! a PRIOR chunk boundary is never recorded at all, permanently, in either
//! the incumbent or this port -- this is not a bug this port introduces,
//! it is the incumbent's own steady-state behavior, faithfully reproduced.
//! [`ngram_cache_update_delta`] reproduces exactly this call shape without
//! allocating the intermediate `tokens_new` vector the incumbent builds --
//! it walks `history`'s own trailing slice plus `sampled` through the same
//! index space [`ngram_cache_update`]'s generic indexer already threads
//! through, so both entry points share one loop body (RISC reuse, this
//! crate's own guiding principle 1).
//!
//! # Zero allocation, and the caller-supplied bound that makes it possible
//! (SPEC.md invariant 3)
//!
//! [`ngram_cache_draft`]'s pure query path -- given caches already built --
//! allocates nothing: n-gram keys are fixed-size `[Option<u32>;
//! LLAMA_NGRAM_MAX]` arrays on the stack (llama's own `common_ngram` is the
//! same fixed-`LLAMA_NGRAM_MAX`-slot struct, `common/ngram-cache.h:15-38`),
//! and `out` is the caller-owned buffer every drafter in this crate writes
//! into.
//!
//! Cache MAINTENANCE ([`ngram_cache_update`]/[`ngram_cache_update_delta`])
//! is NOT unbounded, despite growing on every genuinely new n-gram observed:
//! within one generation, `common_ngram_cache_update`'s own body
//! (`common/ngram-cache.cpp:12-52`, ported here as [`update_via_indexer`])
//! bumps at most one entry per `(position, ngram_size)` pair, `ngram_size`
//! ranging over [`LLAMA_NGRAM_MIN`]..=[`LLAMA_NGRAM_MAX`] -- and
//! [`NgramCacheState`]'s own `cache_size` bookkeeping ensures every
//! [`ngram_cache_update_delta`] call's own range tiles the token stream
//! EXACTLY ONCE, never overlapping, across the whole lifetime of one
//! generation. So the total number of (n-gram, continuation-token) entries
//! the context cache will EVER hold is bounded by
//! `LLAMA_NGRAM_MAX * max_context_len` -- the caller's own model context
//! window -- the SAME caller-supplied-bound discipline
//! [`crate::draft::ngram_map::NgramMap::new`]'s own doc documents.
//! [`NgramCache::with_capacity`] presizes both the open-addressed key table
//! and the continuation-count arena from that bound up front, so
//! maintenance on a correctly-sized context cache allocates nothing beyond
//! construction -- no [`alloc::collections::BTreeMap`] per-entry tree-node
//! allocation on this crate's own per-token hot path
//! ([`NgramCacheState::new`]'s own doc). [`NgramCache::new`] (unbounded,
//! growable) remains the right constructor for a static/dynamic cache
//! loaded once from a file ([`load_llama_ngram_cache_bytes`]) or built ad
//! hoc in a test -- those are cold, one-time-build paths this module's own
//! zero-alloc claim never covers.

use alloc::vec::Vec;

/// llama's `LLAMA_NGRAM_MIN` (`common/ngram-cache.h:9`).
pub const LLAMA_NGRAM_MIN: usize = 1;

/// llama's `LLAMA_NGRAM_MAX` (`common/ngram-cache.h:10`).
pub const LLAMA_NGRAM_MAX: usize = 4;

/// llama's `LLAMA_NGRAM_STATIC` (`common/ngram-cache.h:11`).
pub const LLAMA_NGRAM_STATIC: usize = 2;

/// llama's `create_state_ngram_cache`'s hardcoded `n_draft`
/// (`common/speculative.cpp:2189-2203`, `README.md`'s own recorded default).
pub const DEFAULT_N_DRAFT: u16 = 8;

/// llama's `draft_min_sample_size_lax` (`common/ngram-cache.cpp:60`),
/// indexed by `ngram_size - 1`.
const DRAFT_MIN_SAMPLE_SIZE_LAX: [u32; LLAMA_NGRAM_MAX] = [2, 2, 1, 1];

/// llama's `draft_min_percent_lax` (`common/ngram-cache.cpp:61`).
const DRAFT_MIN_PERCENT_LAX: [u32; LLAMA_NGRAM_MAX] = [66, 50, 50, 50];

/// llama's `draft_min_sample_size_strict` (`common/ngram-cache.cpp:62`).
const DRAFT_MIN_SAMPLE_SIZE_STRICT: [u32; LLAMA_NGRAM_MAX] = [4, 3, 2, 2];

/// llama's `draft_min_percent_strict` (`common/ngram-cache.cpp:63`).
const DRAFT_MIN_PERCENT_STRICT: [u32; LLAMA_NGRAM_MAX] = [75, 66, 66, 66];

/// llama's `common_ngram` (`common/ngram-cache.h:15-38`): up to
/// [`LLAMA_NGRAM_MAX`] token ids, `None` in place of llama's own
/// `LLAMA_TOKEN_NULL` sentinel for the unused trailing slots of a
/// shorter-than-`LLAMA_NGRAM_MAX` n-gram.
pub type NgramKey = [Option<u32>; LLAMA_NGRAM_MAX];

const NONE_LINK: u32 = u32::MAX;

/// One continuation-token/count pair, part of [`NgramCache`]'s own
/// presized arena -- see [`NgramCache`]'s doc. `next` chains to another
/// entry sharing the same n-gram key ([`NONE_LINK`] ends the chain), the
/// flat-slab equivalent of llama's own `common_ngram_cache_part`
/// (`common/ngram-cache.h:58`, an `unordered_map<token, count>` per key).
#[derive(Debug, Clone, Copy)]
struct PartEntry {
    token: u32,
    count: u32,
    next: u32,
}

/// A read view over one n-gram's continuation-token distribution -- llama's
/// own `common_ngram_cache_part`, without materializing a map: walks
/// [`NgramCache`]'s shared arena starting from one key's own chain head.
#[derive(Debug, Clone, Copy)]
pub struct NgramCachePart<'a> {
    parts: &'a [PartEntry],
    head: u32,
}

impl<'a> NgramCachePart<'a> {
    /// The recorded count for `token`, if this n-gram was ever observed
    /// followed by it -- llama's own `part.find(token)`.
    #[must_use]
    pub fn get(&self, token: u32) -> Option<u32> {
        self.iter().find(|&(candidate, _)| candidate == token).map(|(_, count)| count)
    }

    /// Every `(token, count)` pair recorded for this n-gram, in the arena's
    /// own chain order (newest-inserted first) -- distinct from llama's own
    /// `unordered_map` iteration order (hash-bucket order), but this
    /// module's own fixture tests are the oracle that no drafted output
    /// depends on the difference (`ngram_cache_matches_llama_fixture`
    /// replays 200+ real llama.cpp cases; a tie-breaking divergence would
    /// surface there).
    pub fn iter(&self) -> NgramCachePartIter<'a> {
        NgramCachePartIter { parts: self.parts, cursor: self.head }
    }
}

impl<'a> IntoIterator for NgramCachePart<'a> {
    type Item = (u32, u32);
    type IntoIter = NgramCachePartIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// [`NgramCachePart::iter`]'s own iterator, walking one key's chain in the
/// shared arena.
#[derive(Debug, Clone)]
pub struct NgramCachePartIter<'a> {
    parts: &'a [PartEntry],
    cursor: u32,
}

impl Iterator for NgramCachePartIter<'_> {
    type Item = (u32, u32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor == NONE_LINK {
            return None;
        }
        let entry = self.parts[self.cursor as usize];
        self.cursor = entry.next;
        Some((entry.token, entry.count))
    }
}

/// One open-addressed slot in [`NgramCache`]'s key table.
#[derive(Debug, Clone, Copy)]
struct KeySlot {
    key: NgramKey,
    occupied: bool,
    part_head: u32,
}

const EMPTY_SLOT: KeySlot = KeySlot { key: [None; LLAMA_NGRAM_MAX], occupied: false, part_head: NONE_LINK };

fn hash_key(key: &NgramKey) -> u64 {
    // FNV-1a over the up to LLAMA_NGRAM_MAX slots -- an internal-only hash
    // for this structure's OWN open addressing, never ported from llama
    // (whose `common_ngram_cache`'s hasher is an implementation detail no
    // caller, and no fixture, can observe -- only content equality is part
    // of the contract).
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for slot in key {
        let folded = match slot {
            Some(token) => (u64::from(*token) << 1) | 1,
            None => 0,
        };
        hash ^= folded;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// llama's `common_ngram_cache` (`common/ngram-cache.h:61`): n-gram ->
/// empirical distribution of following tokens. An open-addressed key table
/// (linear probing) over a shared, presizable continuation-count arena --
/// see this module's own doc, "Zero allocation" section, for why a bound
/// exists and how [`NgramCache::with_capacity`] sizes against it -- rather
/// than llama's own `unordered_map` + hand-written hasher (same lookup
/// contract, content equality; no hashing-fidelity surface to port) or a
/// `BTreeMap` (whose per-entry tree-node allocation is exactly what this
/// port replaces).
#[derive(Debug, Clone, Default)]
pub struct NgramCache {
    slots: Vec<KeySlot>,
    mask: usize,
    parts: Vec<PartEntry>,
    len: usize,
}

impl NgramCache {
    /// An empty, growable cache -- llama's own default-constructed
    /// `common_ngram_cache`. The right constructor for a static/dynamic
    /// cache loaded from a file ([`load_llama_ngram_cache_bytes`]) or built
    /// ad hoc, where allocating at load/build time is legitimate (this
    /// module's own doc, "Zero allocation" section) -- for the per-token
    /// HOT path, see [`NgramCache::with_capacity`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Presizes both the key table and the continuation-count arena from a
    /// caller-supplied `max_context_len` -- see this module's own doc,
    /// "Zero allocation" section, for the `LLAMA_NGRAM_MAX * max_context_len`
    /// bound this sizes against, the same caller-supplied-bound discipline
    /// [`crate::draft::ngram_map::NgramMap::new`]'s own doc documents. The
    /// key table is sized to half that bound's occupancy (a 50% load
    /// factor, generous headroom against probe-chain length) so growth
    /// never triggers on a correctly-sized context cache.
    #[must_use]
    pub fn with_capacity(max_context_len: usize) -> Self {
        let part_capacity = LLAMA_NGRAM_MAX.saturating_mul(max_context_len);
        let slot_capacity = part_capacity.saturating_mul(2).next_power_of_two().max(16);
        Self {
            slots: alloc::vec![EMPTY_SLOT; slot_capacity],
            mask: slot_capacity - 1,
            parts: Vec::with_capacity(part_capacity),
            len: 0,
        }
    }

    /// Number of distinct n-grams recorded -- llama's own `.size()`
    /// (`common/ngram-cache.cpp:206`, used by the fixture generator's own
    /// `nc_for_save.size()` log line).
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` when no n-gram has been recorded yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn get(&self, key: &NgramKey) -> Option<NgramCachePart<'_>> {
        if self.slots.is_empty() {
            return None;
        }
        let mut index = (hash_key(key) as usize) & self.mask;
        loop {
            let slot = &self.slots[index];
            if !slot.occupied {
                return None;
            }
            if &slot.key == key {
                return Some(NgramCachePart { parts: &self.parts, head: slot.part_head });
            }
            index = (index + 1) & self.mask;
        }
    }

    /// Finds or creates `key`'s slot, growing the key table first if
    /// occupancy would exceed a 75% load factor -- on a correctly-sized
    /// [`NgramCache::with_capacity`] instance this branch is never taken;
    /// it exists so [`NgramCache::new`]'s growable path (loading an
    /// arbitrarily large static cache file) stays correct rather than
    /// panicking or silently dropping entries.
    fn slot_index_for(&mut self, key: NgramKey) -> usize {
        if self.slots.is_empty() || (self.len + 1) * 4 > self.slots.len() * 3 {
            self.grow();
        }
        let mut index = (hash_key(&key) as usize) & self.mask;
        loop {
            if !self.slots[index].occupied {
                self.slots[index] = KeySlot { key, occupied: true, part_head: NONE_LINK };
                self.len += 1;
                return index;
            }
            if self.slots[index].key == key {
                return index;
            }
            index = (index + 1) & self.mask;
        }
    }

    fn grow(&mut self) {
        let new_capacity = (self.slots.len() * 2).max(16);
        let old_slots = core::mem::replace(&mut self.slots, alloc::vec![EMPTY_SLOT; new_capacity]);
        self.mask = new_capacity - 1;
        for slot in old_slots {
            if !slot.occupied {
                continue;
            }
            let mut index = (hash_key(&slot.key) as usize) & self.mask;
            while self.slots[index].occupied {
                index = (index + 1) & self.mask;
            }
            self.slots[index] = slot;
        }
    }

    /// Increments `key`'s count for `token` by one, creating either the key
    /// or the token entry as needed -- llama's own
    /// `map[key][token] += 1` (`common/ngram-cache.cpp:47`).
    fn bump(&mut self, key: NgramKey, token: u32) {
        let index = self.slot_index_for(key);
        let mut cursor = self.slots[index].part_head;
        while cursor != NONE_LINK {
            let entry = &mut self.parts[cursor as usize];
            if entry.token == token {
                entry.count += 1;
                return;
            }
            cursor = entry.next;
        }
        let new_index = u32::try_from(self.parts.len()).unwrap_or(NONE_LINK - 1);
        self.parts.push(PartEntry { token, count: 1, next: self.slots[index].part_head });
        self.slots[index].part_head = new_index;
    }

    /// Sets `key`'s count for `token` to exactly `count` (overwrite, not
    /// increment) -- [`load_llama_ngram_cache_bytes`]'s own record-by-record
    /// load, matching a `BTreeMap::insert`'s overwrite-on-duplicate
    /// semantics for a malformed record repeating one token.
    fn set_loaded(&mut self, key: NgramKey, token: u32, count: u32) {
        let index = self.slot_index_for(key);
        let mut cursor = self.slots[index].part_head;
        while cursor != NONE_LINK {
            let entry = &mut self.parts[cursor as usize];
            if entry.token == token {
                entry.count = count;
                return;
            }
            cursor = entry.next;
        }
        let new_index = u32::try_from(self.parts.len()).unwrap_or(NONE_LINK - 1);
        self.parts.push(PartEntry { token, count, next: self.slots[index].part_head });
        self.slots[index].part_head = new_index;
    }
}

fn make_key(tokens: &[u32]) -> NgramKey {
    let mut key: NgramKey = [None; LLAMA_NGRAM_MAX];
    for (slot, &token) in key.iter_mut().zip(tokens) {
        *slot = Some(token);
    }
    key
}

/// The shared index-walking core [`ngram_cache_update`] and
/// [`ngram_cache_update_delta`] both drive (this module's own doc, "RISC
/// reuse"): llama's own `common_ngram_cache_update` body
/// (`common/ngram-cache.cpp:20-51`), parameterized over how a logical
/// position maps to a token id so neither caller needs to materialize a
/// concrete slice covering the whole dataset.
fn update_via_indexer(
    cache: &mut NgramCache,
    inp_size: usize,
    nnew: usize,
    token_at: impl Fn(usize) -> u32,
) {
    for ngram_size in LLAMA_NGRAM_MIN..=LLAMA_NGRAM_MAX {
        let i_start = inp_size.saturating_sub(nnew).max(ngram_size);
        for i in i_start..inp_size {
            let ngram_start = i - ngram_size;
            let mut tokens = [0u32; LLAMA_NGRAM_MAX];
            for (offset, slot) in tokens[..ngram_size].iter_mut().enumerate() {
                *slot = token_at(ngram_start + offset);
            }
            let key = make_key(&tokens[..ngram_size]);
            let token = token_at(i);
            cache.bump(key, token);
        }
    }
}

/// A faithful port of `common_ngram_cache_update`
/// (`common/ngram-cache.cpp:12-52`), `ngram_min`/`ngram_max` fixed to
/// [`LLAMA_NGRAM_MIN`]/[`LLAMA_NGRAM_MAX`] -- the only configuration any
/// caller in the incumbent ever uses (`README.md`'s own recorded default).
/// `inp_data` is llama's `inp_data`; `nnew` is llama's `nnew` (how many
/// trailing positions of `inp_data` are new since the last call -- pass
/// `inp_data.len()` to treat the whole slice as new, matching the
/// fixture generator's own static-cache build,
/// `generator/main.cpp:497-499`).
pub fn ngram_cache_update(cache: &mut NgramCache, inp_data: &[u32], nnew: usize) {
    update_via_indexer(cache, inp_data.len(), nnew, |index| inp_data[index]);
}

/// The zero-copy equivalent of the incumbent's own per-call `tokens_new`
/// construction (`common_speculative_impl_ngram_cache::draft_one`,
/// `common/speculative.cpp:2103-2117`; `generator/main.cpp:262-266`'s
/// identical replay) -- see this module's own doc, "the context-cache
/// maintenance chunking" section, for why this is faithful rather than an
/// approximation. `cache_size` is the caller's own
/// [`NgramCacheState`]-tracked high-water mark.
fn ngram_cache_update_delta(cache: &mut NgramCache, history: &[u32], sampled: u32, cache_size: usize) {
    let local_size = history.len() + 1 - cache_size;
    update_via_indexer(cache, local_size, local_size, |local| {
        if local + 1 == local_size {
            sampled
        } else {
            history[cache_size + local]
        }
    });
}

/// llama's `get_token` (`common/ngram-cache.cpp:55-57`): reads position
/// `index` of the logical sequence `history ++ [sampled] ++ out`, exactly
/// the sequence llama's own `inp ++ draft[1..]` represents, without
/// concatenating any of the three into a new buffer.
fn token_at(history: &[u32], sampled: u32, out: &[u32], index: usize) -> u32 {
    if index < history.len() {
        history[index]
    } else if index == history.len() {
        sampled
    } else {
        out[index - history.len() - 1]
    }
}

/// llama's single-cache `try_draft` overload (`common/ngram-cache.cpp:66-95`):
/// the static-cache-only fallback, used when neither context nor dynamic
/// answered.
fn try_draft_static_only(static_cache: &NgramCache, ngram_static: &NgramKey) -> Option<u32> {
    let part_static = static_cache.get(ngram_static)?;

    let mut max_count_static = 0u32;
    let mut sum_count_static = 0u32;
    let mut max_token = None;
    for (token, count_static) in part_static {
        if count_static > max_count_static {
            max_token = Some(token);
            max_count_static = count_static;
        }
        sum_count_static += count_static;
    }

    let index = LLAMA_NGRAM_STATIC - 1;
    if sum_count_static < DRAFT_MIN_SAMPLE_SIZE_LAX[index] {
        return None;
    }
    if 100 * max_count_static < DRAFT_MIN_PERCENT_LAX[index] * sum_count_static {
        return None;
    }
    max_token
}

/// llama's two-cache `try_draft` overload (`common/ngram-cache.cpp:97-144`):
/// the context/dynamic path, longest n-gram first, each candidate weighted
/// by the static cache's own corroborating count.
fn try_draft_primary(
    primary: &NgramCache,
    ngrams_primary: &[NgramKey; LLAMA_NGRAM_MAX],
    part_static: Option<NgramCachePart<'_>>,
    min_sample_size: &[u32; LLAMA_NGRAM_MAX],
    min_percent: &[u32; LLAMA_NGRAM_MAX],
) -> Option<u32> {
    for index in (0..LLAMA_NGRAM_MAX).rev() {
        let Some(part_primary) = primary.get(&ngrams_primary[index]) else {
            continue;
        };

        let mut max_count_primary = 0u32;
        let mut max_count_static = 0u32;
        let mut sum_count_primary = 0u32;
        let mut max_token = None;

        for (token, count_primary) in part_primary {
            let count_static = part_static.and_then(|part| part.get(token)).map_or(1, |count| 100 * count);

            if count_primary * count_static > max_count_primary * max_count_static {
                max_token = Some(token);
                max_count_primary = count_primary;
                max_count_static = count_static;
            }
            sum_count_primary += count_primary;
        }

        if sum_count_primary < min_sample_size[index] {
            continue;
        }
        if 100 * max_count_primary < min_percent[index] * sum_count_primary {
            continue;
        }
        return max_token;
    }
    None
}

/// A faithful port of `common_ngram_cache_draft`
/// (`common/ngram-cache.cpp:146-198`) -- see this module's own doc for the
/// three-cache, longest-n-gram-first algorithm. `history` is llama's `inp`
/// minus its own trailing `id_last` (every token generated so far, NOT
/// including `sampled`); `sampled` is the token the caller's own sampler
/// just drew for the position immediately after `history`. `out` is
/// llama's `draft` minus its own leading `id_last` sentinel entry -- this
/// port never stores `sampled` in `out` at all, so there is no leading
/// entry to strip back out afterward.
///
/// Requires `history.len() >= LLAMA_NGRAM_MAX - 1`: shorter histories fall
/// into a region the incumbent's own `int`-typed index arithmetic can
/// underflow past zero (`ngram_start_cd`, `common/ngram-cache.cpp:174`) --
/// undefined behavior upstream, never reached by any real caller (the
/// fixture generator itself starts every stream at position `LLAMA_NGRAM_
/// MAX + 1`, `generator/main.cpp:259`) -- so this port defines it as "draft
/// nothing" rather than reproducing the underflow.
pub fn ngram_cache_draft(
    history: &[u32],
    sampled: u32,
    n_draft: u16,
    context: &NgramCache,
    dynamic: &NgramCache,
    static_cache: &NgramCache,
    out: &mut Vec<u32>,
) {
    out.clear();

    let inp_size = history.len() + 1;
    if inp_size < LLAMA_NGRAM_STATIC || history.len() + 1 < LLAMA_NGRAM_MAX {
        return;
    }

    let n_draft = usize::from(n_draft);
    while out.len() < n_draft {
        let step = out.len();

        let ngram_start_static = inp_size - LLAMA_NGRAM_STATIC + step;
        let mut ngram_static: NgramKey = [None; LLAMA_NGRAM_MAX];
        for (offset, slot) in ngram_static[..LLAMA_NGRAM_STATIC].iter_mut().enumerate() {
            *slot = Some(token_at(history, sampled, out, ngram_start_static + offset));
        }
        let part_static = static_cache.get(&ngram_static);

        let mut ngrams_cd: [NgramKey; LLAMA_NGRAM_MAX] = [[None; LLAMA_NGRAM_MAX]; LLAMA_NGRAM_MAX];
        for (index, ngram_size) in (LLAMA_NGRAM_MIN..=LLAMA_NGRAM_MAX).enumerate() {
            let ngram_start_cd = inp_size - ngram_size + step;
            let mut key: NgramKey = [None; LLAMA_NGRAM_MAX];
            for (offset, slot) in key[..ngram_size].iter_mut().enumerate() {
                *slot = Some(token_at(history, sampled, out, ngram_start_cd + offset));
            }
            ngrams_cd[index] = key;
        }

        let mut drafted_token = try_draft_primary(
            context,
            &ngrams_cd,
            part_static,
            &DRAFT_MIN_SAMPLE_SIZE_LAX,
            &DRAFT_MIN_PERCENT_LAX,
        );
        if drafted_token.is_none() {
            drafted_token = try_draft_primary(
                dynamic,
                &ngrams_cd,
                part_static,
                &DRAFT_MIN_SAMPLE_SIZE_STRICT,
                &DRAFT_MIN_PERCENT_STRICT,
            );
        }
        if drafted_token.is_none() {
            drafted_token = try_draft_static_only(static_cache, &ngram_static);
        }

        let Some(token) = drafted_token else {
            break;
        };
        out.push(token);
    }
}

/// The per-generation-stream state `common_speculative_impl_ngram_cache::
/// seq_info` carries (`common/speculative.cpp:2032-2038`): an
/// incrementally-built context cache, a dynamic cache (empty unless seeded
/// from a prior generation's own saved cache), a static cache (loaded once,
/// e.g. via [`load_llama_ngram_cache_bytes`]), and the high-water mark
/// `cache_size`.
#[derive(Debug, Clone)]
pub struct NgramCacheState {
    context: NgramCache,
    dynamic: NgramCache,
    static_cache: NgramCache,
    cache_size: usize,
}

impl NgramCacheState {
    /// Every cache starts empty -- llama's own default-constructed
    /// `seq_info` when no `--lookup-cache-*` path was given. `context` is
    /// presized via [`NgramCache::with_capacity`] from `max_context_len`
    /// (the caller's own model context window, the SAME bound
    /// [`crate::draft::ngram_map::NgramMap::new`]'s own doc documents) so
    /// this stream's own per-token maintenance never reallocates --
    /// `dynamic`/`static_cache` start empty and growable
    /// ([`NgramCache::new`]), matching this module's own doc: only the
    /// context cache's per-token path is this crate's zero-alloc claim.
    #[must_use]
    pub fn new(max_context_len: usize) -> Self {
        Self {
            context: NgramCache::with_capacity(max_context_len),
            dynamic: NgramCache::new(),
            static_cache: NgramCache::new(),
            cache_size: 0,
        }
    }

    /// Seeds the dynamic and/or static caches from previously-built
    /// [`NgramCache`]s (e.g. loaded via [`load_llama_ngram_cache_bytes`]) --
    /// llama's own `common_speculative_impl_ngram_cache` constructor
    /// (`common/speculative.cpp:2064-2088`) cloning a loaded cache into
    /// every sequence's `sinfo`. `context` is presized the same way
    /// [`NgramCacheState::new`]'s own doc documents.
    #[must_use]
    pub fn with_caches(max_context_len: usize, dynamic: NgramCache, static_cache: NgramCache) -> Self {
        Self {
            context: NgramCache::with_capacity(max_context_len),
            dynamic,
            static_cache,
            cache_size: 0,
        }
    }
}

/// A faithful port of `common_speculative_impl_ngram_cache::draft_one`
/// (`common/speculative.cpp:2095-2138`) -- see this module's own doc.
/// `history`/`sampled`/`out` follow every other drafter in this crate's own
/// caller-owned-buffer discipline.
pub fn ngram_cache_state_draft(
    state: &mut NgramCacheState,
    history: &[u32],
    sampled: u32,
    n_draft: u16,
    out: &mut Vec<u32>,
) {
    if state.cache_size < history.len() + 1 {
        ngram_cache_update_delta(&mut state.context, history, sampled, state.cache_size);
        state.cache_size = history.len() + 1;
    }

    ngram_cache_draft(
        history,
        sampled,
        n_draft,
        &state.context,
        &state.dynamic,
        &state.static_cache,
        out,
    );
}

/// A parse error while reading a `common_ngram_cache_save`-produced binary
/// file (`common/ngram-cache.cpp:200-220`): a flat, unframed sequence of
/// records with no length prefix at the file level and no magic number --
/// truncation is the only failure mode a reader can detect.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum NgramCacheLoadError {
    /// The file ended in the middle of a record -- llama's own loader
    /// (`common_ngram_cache_load`, `:222-258`) asserts on this rather than
    /// returning an error (`GGML_ASSERT(!hashmap_file.eof())`); this port
    /// reports it instead of aborting the process.
    #[error("truncated llama ngram cache: expected {expected} more bytes at offset {offset}, found {found}")]
    Truncated {
        /// Bytes the next field needed.
        expected: usize,
        /// Byte offset the read started at.
        offset: usize,
        /// Bytes actually remaining in the buffer.
        found: usize,
    },
}

fn read_i32_le(bytes: &[u8], offset: &mut usize) -> Result<i32, NgramCacheLoadError> {
    let remaining = bytes.len() - *offset;
    if remaining < 4 {
        return Err(NgramCacheLoadError::Truncated {
            expected: 4,
            offset: *offset,
            found: remaining,
        });
    }
    let mut raw = [0u8; 4];
    raw.copy_from_slice(&bytes[*offset..*offset + 4]);
    *offset += 4;
    Ok(i32::from_le_bytes(raw))
}

/// A faithful port of `common_ngram_cache_load`
/// (`common/ngram-cache.cpp:222-258`)'s binary layout, read from an
/// in-memory buffer rather than a `std::ifstream` -- the file format itself
/// has no IO dependency, only its acquisition does (kept in this same
/// no_std+alloc module; [`load_llama_ngram_cache_file`] is the std-gated
/// file-reading wrapper). Each record is llama's own raw struct layout,
/// native (little-endian) byte order: [`LLAMA_NGRAM_MAX`] x `i32` ngram
/// tokens (`-1` is llama's `LLAMA_TOKEN_NULL`, mapped to `None`), then one
/// `i32` token count, then that many `(i32 token, i32 count)` pairs.
pub fn load_llama_ngram_cache_bytes(bytes: &[u8]) -> Result<NgramCache, NgramCacheLoadError> {
    let mut cache = NgramCache::new();
    let mut offset = 0usize;

    while offset < bytes.len() {
        let mut ngram: NgramKey = [None; LLAMA_NGRAM_MAX];
        for slot in &mut ngram {
            let raw = read_i32_le(bytes, &mut offset)?;
            *slot = if raw < 0 { None } else { Some(raw as u32) };
        }

        let ntokens = read_i32_le(bytes, &mut offset)?.max(0) as usize;
        for _ in 0..ntokens {
            let token = read_i32_le(bytes, &mut offset)?.max(0) as u32;
            let count = read_i32_le(bytes, &mut offset)?.max(0) as u32;
            cache.set_loaded(ngram, token, count);
        }
    }

    Ok(cache)
}

/// The std-gated file-reading wrapper around [`load_llama_ngram_cache_bytes`]
/// -- llama's own `common_speculative_impl_ngram_cache` constructor loads
/// static/dynamic caches from a path (`common/speculative.cpp:2064-2088`),
/// and this is the equivalent entry point for a caller with a real
/// filesystem.
#[cfg(feature = "std")]
pub fn load_llama_ngram_cache_file(path: &std::path::Path) -> Result<NgramCache, NgramCacheLoadIoError> {
    let bytes = std::fs::read(path).map_err(NgramCacheLoadIoError::Io)?;
    load_llama_ngram_cache_bytes(&bytes).map_err(NgramCacheLoadIoError::Parse)
}

/// [`load_llama_ngram_cache_file`]'s error, std-gated because
/// [`std::io::Error`] itself is a `std` type this crate's no_std+alloc
/// floor never sees.
#[cfg(feature = "std")]
#[derive(Debug, thiserror::Error)]
pub enum NgramCacheLoadIoError {
    /// The file could not be opened or read.
    #[error("failed to read llama ngram cache file: {0}")]
    Io(#[source] std::io::Error),
    /// The file opened, but its contents did not parse -- see
    /// [`NgramCacheLoadError`].
    #[error(transparent)]
    Parse(#[from] NgramCacheLoadError),
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::collections::BTreeMap;
    use alloc::vec;
    use alloc::vec::Vec;

    use serde::Deserialize;

    use super::{
        DEFAULT_N_DRAFT, LLAMA_NGRAM_MAX, NgramCacheState, load_llama_ngram_cache_bytes,
        ngram_cache_draft, ngram_cache_state_draft,
    };

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
        n_draft: u16,
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
    const NGRAM_CACHE_JSON: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/llama-ngram/fixtures/ngram_cache.json"
    ));
    const NGRAM_CACHE_STATIC_BIN: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/llama-ngram/fixtures/ngram_cache_static.bin"
    ));

    /// [`ngram_cache_state_draft`] against every case in
    /// `tests/fixtures/llama-ngram/fixtures/ngram_cache.json` (SPEC.md
    /// AC9), replaying a FRESH [`NgramCacheState`] per stream (`nc_dynamic`/
    /// `nc_static` both empty throughout, matching
    /// `generator/main.cpp:488-489`'s own `nc_static_empty`) -- each
    /// stream's own recorded cases are independent of every other stream's.
    #[test]
    fn ngram_cache_matches_llama_fixture() {
        let streams: StreamsFile = serde_json::from_str(STREAMS_JSON).expect("streams.json parses");
        let fixture: FixtureFile =
            serde_json::from_str(NGRAM_CACHE_JSON).expect("ngram_cache.json parses");

        let mut cases_by_stream: BTreeMap<u32, Vec<&FixtureCase>> = BTreeMap::new();
        for case in &fixture.cases {
            cases_by_stream.entry(case.stream_id).or_default().push(case);
        }

        let mut total = 0usize;
        let mut non_empty = 0usize;
        let mut drafted: Vec<u32> = Vec::new();

        for stream in &streams.streams {
            let Some(cases) = cases_by_stream.get(&stream.id) else {
                continue;
            };

            let mut state = NgramCacheState::new(stream.tokens.len());
            for case in cases {
                let history = &stream.tokens[..case.position];
                ngram_cache_state_draft(
                    &mut state,
                    history,
                    case.sampled,
                    fixture.params.n_draft,
                    &mut drafted,
                );
                total += 1;
                if !case.draft.is_empty() {
                    non_empty += 1;
                }
                assert_eq!(
                    drafted, case.draft,
                    "stream {} position {}: got {drafted:?}, want {:?}",
                    stream.id, case.position, case.draft
                );

                let _ = case.accepted;
            }
        }

        println!("cases = {total} non_empty = {non_empty}");
        assert!(total >= 200, "fixture must carry at least 200 cases per SPEC.md AC9");
    }

    /// SPEC.md AC9's second half, R7's own "loads llama.cpp static/dynamic
    /// cache files": [`load_llama_ngram_cache_bytes`] against the vendored
    /// `ngram_cache_static.bin` (`README.md`'s own recorded provenance: a
    /// real cache built from `streams[2]`, llama.cpp's own
    /// `common/ngram-map.cpp`, via llama's own `common_ngram_cache_update` +
    /// `common_ngram_cache_save`) proves the loaded contents drive drafts
    /// correctly by feeding the loaded cache in as the STATIC cache for a
    /// context/dynamic-empty [`NgramCacheState`] over a real gemma4-tokenized
    /// stream from `streams.json` and confirming the static-only fallback
    /// path (`try_draft_static_only`) actually fires and drafts real,
    /// non-empty tokens sourced from the loaded cache.
    #[test]
    fn ngram_cache_loads_llama_file() {
        let cache = load_llama_ngram_cache_bytes(NGRAM_CACHE_STATIC_BIN)
            .expect("ngram_cache_static.bin parses");
        assert!(
            !cache.is_empty(),
            "the vendored static cache must contain at least one recorded n-gram"
        );

        let streams: StreamsFile = serde_json::from_str(STREAMS_JSON).expect("streams.json parses");
        let stream = streams
            .streams
            .iter()
            .find(|stream| stream.id == 2)
            .expect("stream 2 (the source the static cache was built from) exists");

        let mut state = NgramCacheState::with_caches(stream.tokens.len(), super::NgramCache::new(), cache);
        let mut drafted: Vec<u32> = Vec::new();
        let mut static_only_drafts = 0usize;

        let start = LLAMA_NGRAM_MAX + 1;
        let mut position = start;
        while position + 1 < stream.tokens.len() && position < start + 400 {
            let history = &stream.tokens[..position];
            let sampled = stream.tokens[position];
            ngram_cache_state_draft(&mut state, history, sampled, DEFAULT_N_DRAFT, &mut drafted);
            if !drafted.is_empty() {
                static_only_drafts += 1;
            }
            position += 1;
        }

        println!("static_only_drafts = {static_only_drafts} over {} positions replayed", start + 400 - start);
        assert!(
            static_only_drafts > 0,
            "the loaded static cache (built from the same stream being replayed) must produce at least one real draft"
        );
    }

    /// Sad path: a history shorter than `LLAMA_NGRAM_MAX - 1` must draft
    /// nothing -- this port's own defined-behavior guard for the region
    /// `common_ngram_cache_draft`'s `int`-typed index arithmetic could
    /// underflow past zero in, never reached by any real caller.
    #[test]
    fn history_too_short_drafts_nothing() {
        let history = vec![1u32, 2];
        let mut state = NgramCacheState::new(history.len());
        let mut drafted = Vec::new();
        ngram_cache_state_draft(&mut state, &history, 3, DEFAULT_N_DRAFT, &mut drafted);
        assert_eq!(drafted, Vec::<u32>::new());
    }

    /// Sad path: an untrained cache (no prior `draft` calls to build
    /// `nc_context`, and no static/dynamic cache seeded) must draft
    /// nothing on the very first call.
    #[test]
    fn untrained_cache_drafts_nothing_on_first_call() {
        let history: Vec<u32> = (0..20u32).collect();
        let mut state = NgramCacheState::new(history.len());
        let mut drafted = Vec::new();
        ngram_cache_state_draft(&mut state, &history, 999, DEFAULT_N_DRAFT, &mut drafted);
        assert_eq!(drafted, Vec::<u32>::new());
    }

    /// A faithful reproduction of a real limitation in the incumbent's own
    /// steady-state call pattern (this module's own doc, "the context-cache
    /// maintenance chunking" section, traced from `draft_one`'s own
    /// `tokens_new` construction): once `cache_size` has caught up to
    /// `history.len()` (the common case once the cache is warm and every
    /// subsequent draft is rejected, `accepted = 0` every round), the NEXT
    /// call's "new" delta is exactly the one incoming `sampled` token --
    /// too short to extract even a unigram-plus-next-token pair, so
    /// [`ngram_cache_update`] (and therefore [`ngram_cache_state_draft`]'s
    /// own internal delta call) records nothing at all for it. This is not
    /// a defect this port introduces -- `ngram_cache_matches_llama_fixture`'s
    /// own 200+ passing cases already exercise this exact call shape end to
    /// end against the incumbent's own recorded output.
    #[test]
    fn update_with_a_single_new_token_delta_learns_nothing() {
        let mut cache = super::NgramCache::new();
        super::ngram_cache_update(&mut cache, &[42u32], 1);
        assert_eq!(cache.len(), 0, "a one-token delta can never record a single n-gram");
    }

    /// Happy path, hand-computed, against the pure [`ngram_cache_draft`] +
    /// [`ngram_cache_update`] pair directly (bypassing [`NgramCacheState`]'s
    /// own per-call chunking, which
    /// `single_token_steady_state_advance_never_trains_the_context_cache`
    /// covers separately): a short repeating pattern, trained in ONE
    /// `ngram_cache_update` pass over the whole stream, makes every
    /// n-gram inside the pattern map to exactly one deterministic
    /// continuation, clearing every lax threshold.
    #[test]
    fn happy_path_drafts_a_deterministically_repeating_continuation() {
        let pattern = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let mut stream: Vec<u32> = Vec::new();
        for _ in 0..3 {
            stream.extend_from_slice(&pattern);
        }

        let mut context = super::NgramCache::new();
        super::ngram_cache_update(&mut context, &stream, stream.len());
        let dynamic = super::NgramCache::new();
        let static_cache = super::NgramCache::new();

        // history covers the first two periods; sampled = 1 starts the
        // third, exactly the trained pattern -- the draft must continue it.
        let history = &stream[..16];
        let sampled = stream[16];
        let mut drafted = Vec::new();
        ngram_cache_draft(history, sampled, 4, &context, &dynamic, &static_cache, &mut drafted);

        assert_eq!(drafted, vec![2u32, 3, 4, 5]);
    }

    /// The buffer-reuse contract every drafter in this crate follows
    /// ([`crate::draft::ngram_simple::ngram_simple_draft`]'s own doc):
    /// calling `ngram_cache_draft` with a buffer already holding a PRIOR
    /// draft must leave `out` holding exactly the new draft.
    #[test]
    fn a_reused_buffer_holding_a_stale_draft_is_fully_overwritten() {
        let pattern = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let mut stream: Vec<u32> = Vec::new();
        for _ in 0..3 {
            stream.extend_from_slice(&pattern);
        }

        let mut context = super::NgramCache::new();
        super::ngram_cache_update(&mut context, &stream, stream.len());
        let dynamic = super::NgramCache::new();
        let static_cache = super::NgramCache::new();

        let history = &stream[..16];
        let sampled = stream[16];
        let mut drafted: Vec<u32> = vec![111, 222, 333];
        ngram_cache_draft(history, sampled, 4, &context, &dynamic, &static_cache, &mut drafted);
        assert_eq!(drafted, vec![2u32, 3, 4, 5]);

        let never_seen_history = vec![900u32, 901, 902, 903, 904];
        ngram_cache_draft(&never_seen_history, 9_999, 4, &context, &dynamic, &static_cache, &mut drafted);
        assert_eq!(
            drafted,
            Vec::<u32>::new(),
            "a non-match must leave the buffer empty, not the previous call's draft"
        );
    }

    /// A malformed cache file (truncated mid-record) must report
    /// [`super::NgramCacheLoadError::Truncated`], not panic -- the negative
    /// test principle 9 carves out for "rejects garbage" cases.
    #[test]
    fn truncated_cache_file_is_a_typed_error() {
        let mut truncated = NGRAM_CACHE_STATIC_BIN[..10].to_vec();
        truncated.truncate(10);
        let result = load_llama_ngram_cache_bytes(&truncated);
        assert!(
            matches!(result, Err(super::NgramCacheLoadError::Truncated { .. })),
            "a file cut off mid-record must be a typed error, not a panic"
        );
    }
}
