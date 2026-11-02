use super::*;
use crate::generate::LayerCacheState;
use crate::serving::PromptCacheConfig;


pub(super) fn base_key() -> CacheKey {
    CacheKey::of(&ServingConfig::default(), false, RopeScaling::None, 0, 0)
}

pub(super) fn state_with_ids(ids: &[u32]) -> CacheEntry {
    CacheEntry::new(
        PrefixState {
            ids: ids.to_vec(),
            layer_caches: vec![LayerCacheState::SharedFromLayer],
            cached_len: ids.len(),
        },
        base_key(),
    )
}

pub(super) fn shared_widths() -> Vec<LayerPadRowWidths> {
    vec![LayerPadRowWidths::SharedFromLayer]
}

pub(super) fn enabled_config() -> PromptCacheConfig {
    PromptCacheConfig {
        byte_budget: 1 << 20,
        ..PromptCacheConfig::off()
    }
}

pub(super) fn naive_common_prefix(left: &[u32], right: &[u32]) -> usize {
    let mut shared = 0;
    while shared < left.len() && shared < right.len() && left[shared] == right[shared] {
        shared += 1;
    }
    shared
}

pub(super) fn indexed_config(block_tokens: u32) -> PromptCacheConfig {
    PromptCacheConfig {
        block_tokens,
        max_entries: 64,
        ..enabled_config()
    }
}

/// The scan `take_best` replaced: every entry's common prefix against the
/// prompt, the reusable ones, the longest, the most recently stored among
/// equals.
pub(super) fn scan_reference(
    stored: &[Vec<u32>],
    prompt: &[u32],
    min_similarity_milli: u32,
) -> Option<(usize, usize)> {
    stored
        .iter()
        .enumerate()
        .map(|(position, ids)| (position, naive_common_prefix(ids, prompt), ids.len()))
        .filter(|&(_, lcp, stored_len)| {
            entry_is_reusable(lcp, stored_len, prompt.len(), min_similarity_milli)
        })
        .max_by_key(|&(position, lcp, _)| (lcp, position))
        .map(|(position, lcp, _)| (position, lcp))
}

pub(super) fn random_conversation(rng: &mut fastrand::Rng, tokens: usize) -> Vec<u32> {
    std::iter::once(2)
        .chain((1..tokens).map(|_| rng.u32(4..262_144)))
        .collect()
}

pub(super) fn median_nanos(mut samples: Vec<u128>) -> u128 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

/// One cache size of [`lookup_cost`]: nanoseconds per lookup, per `take_best`, and per
/// the scan the index replaced, for a prompt that extends a stored conversation
/// (hit) and one that shares nothing past BOS (miss), plus what the index holds.
pub struct LookupCostRow {
    pub shared_header: usize,
    pub entries: usize,
    pub lookup_hit_ns: u128,
    pub lookup_miss_ns: u128,
    pub take_hit_ns: u128,
    pub take_miss_ns: u128,
    pub scan_hit_ns: u128,
    pub scan_miss_ns: u128,
    pub index_bytes: usize,
    pub index_nodes: usize,
    pub bloom_bytes: usize,
}

/// Cost: the time one lookup takes with 4, 64, 256 and 1,024 cached
/// 1,024-token conversations (every one opening with the same BOS token).
/// `lookup` is the candidate search alone, timed over batches so the clock's
/// tick is a small part of each sample; `take` is the whole `take_best`, one
/// call per sample. Median of the samples. `shared_header` is the length of an
/// opening every conversation shares (a system prompt), `0` for none.
pub fn lookup_cost(shared_header: usize) -> Vec<LookupCostRow> {
    let mut rows = Vec::new();
    let header: Vec<u32> = if shared_header == 0 {
        Vec::new()
    } else {
        random_conversation(&mut fastrand::Rng::with_seed(5), shared_header)
    };
    const SAMPLES: usize = 200;
    const BATCH: u128 = 100;
    const TAKES: usize = 2000;
    let key = base_key();
    for entry_count in [4_usize, 64, 256, 1024] {
        let config = PromptCacheConfig {
            byte_budget: 1 << 32,
            max_entries: entry_count as u32,
            ..indexed_config(64)
        };
        let mut rng = fastrand::Rng::with_seed(11);
        let stored: Vec<Vec<u32>> = (0..entry_count)
            .map(|_| {
                let tail = random_conversation(&mut rng, 1024 - shared_header);
                header.iter().copied().chain(tail).collect()
            })
            .collect();
        let mut cache = PromptCache::new();
        stored
            .iter()
            .for_each(|ids| assert!(cache.store(state_with_ids(ids), &config).is_some()));
        let extends: Vec<u32> = stored[entry_count / 2]
            .iter()
            .copied()
            .chain((0..100).map(|_| rng.u32(4..262_144)))
            .collect();
        let unrelated: Vec<u32> = header
            .iter()
            .copied()
            .chain(random_conversation(&mut rng, 1124 - shared_header))
            .collect();
        let timed_take = |cache: &mut PromptCache, prompt: &[u32]| -> u128 {
            median_nanos(
                (0..TAKES)
                    .map(|_| {
                        let started = std::time::Instant::now();
                        let (taken, _) = cache.take_best(prompt, &key, &shared_widths(), 100);
                        let nanos = started.elapsed().as_nanos();
                        if let Some(entry) = taken {
                            let mut whole = entry;
                            whole.state.ids = stored[entry_count / 2].clone();
                            whole.state.cached_len = whole.state.ids.len();
                            cache.store(whole, &config);
                        }
                        nanos
                    })
                    .collect(),
            )
        };
        let batched = |work: &dyn Fn()| -> u128 {
            median_nanos(
                (0..SAMPLES)
                    .map(|_| {
                        let started = std::time::Instant::now();
                        (0..BATCH).for_each(|_| work());
                        started.elapsed().as_nanos() / BATCH
                    })
                    .collect(),
            )
        };
        let lookup = |prompt: &[u32]| {
            batched(&|| {
                std::hint::black_box(cache.best_candidate(prompt, &key, 100, false));
            })
        };
        let scan = |prompt: &[u32]| {
            batched(&|| {
                std::hint::black_box(scan_reference(&stored, prompt, 100));
            })
        };
        let lookup_hit = lookup(&extends);
        let lookup_miss = lookup(&unrelated);
        let scan_hit = scan(&extends);
        let scan_miss = scan(&unrelated);
        let index_bytes = cache.index_byte_len();
        let index_nodes = cache.index_node_count();
        let bloom_bytes: usize = cache
            .entries
            .values()
            .map(|entry| entry.bloom.as_ref().map_or(0, BlockBloom::byte_len))
            .sum();
        let take_hit = timed_take(&mut cache, &extends);
        let take_miss = timed_take(&mut cache, &unrelated);
        rows.push(LookupCostRow {
            shared_header,
            entries: entry_count,
            lookup_hit_ns: lookup_hit,
            lookup_miss_ns: lookup_miss,
            take_hit_ns: take_hit,
            take_miss_ns: take_miss,
            scan_hit_ns: scan_hit,
            scan_miss_ns: scan_miss,
            index_bytes,
            index_nodes,
            bloom_bytes,
        });
    }
    rows
}
