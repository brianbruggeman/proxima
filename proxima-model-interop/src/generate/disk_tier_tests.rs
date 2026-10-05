#![allow(clippy::expect_used)]

use std::path::Path;

use super::prompt_cache::{CacheEntry, PromptCache, thaw_best_cold};
use super::*;
use crate::block_file::MappedBlockFile;
use crate::{ColdTier, InteropError, PrefixState};

#[path = "../../tests/support/block_write.rs"]
mod block_write;
#[path = "../../tests/support/block_keys.rs"]
mod block_keys;
#[path = "../../tests/support/disk_tier.rs"]
mod disk_tier;

use block_keys::chained_keys;
use block_write::write_block_file;
use disk_tier::{DiskTier, spilled_names};

const TRACE_A: [u32; 4] = [818, 5279, 529, 7001];
const TRACE_B: [u32; 4] = [2063, 10779, 78113, 236769];
const TRACE_C: [u32; 4] = [194618, 6081, 86460, 699];
const TRACE_D: [u32; 4] = [3459, 1883, 7733, 1156];
const TRACE_E: [u32; 4] = [5001, 5002, 5003, 5004];

fn base_key() -> CacheKey {
    CacheKey::of(&ServingConfig::default(), false, RopeScaling::None, 0, 0)
}

fn enabled_config() -> PromptCacheConfig {
    PromptCacheConfig {
        byte_budget: 1 << 20,
        ..PromptCacheConfig::off()
    }
}

fn rows_of(ids: &[u32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut even = Vec::new();
    let mut odd = Vec::new();
    let mut value = Vec::new();
    for id in ids {
        let base = *id as f32 * 0.5;
        even.extend([base, base + 0.25]);
        odd.extend([base + 0.5, base + 0.75]);
        value.push(*id as f32);
    }
    (even, odd, value)
}

fn entry_with_rows(ids: &[u32]) -> CacheEntry {
    let (even, odd, value) = rows_of(ids);
    let mut full = LayerCache::new();
    full.append(&even, &odd, &value);
    let mut ring = LayerCache::ring(KvRing::new(2, 1, 2, 1, 0), ids.len());
    ring.append_at(0, &even, &odd, &value);
    CacheEntry::new(
        PrefixState {
            ids: ids.to_vec(),
            layer_caches: vec![
                LayerCacheState::Attention(full),
                LayerCacheState::Attention(ring),
                LayerCacheState::SharedFromLayer,
            ],
            cached_len: ids.len(),
        },
        base_key(),
    )
}

fn disk_cache(directory: &Path, max_entries: usize) -> PromptCache {
    let tier = Arc::new(DiskTier::open(directory, [0x22; 16]).expect("the spill directory opens"));
    let mut cache = PromptCache::new();
    cache.set_cold(tier, max_entries, 1 << 30);
    cache
}

fn two_entry_config() -> PromptCacheConfig {
    PromptCacheConfig {
        max_entries: 2,
        ..enabled_config()
    }
}

fn store_all(cache: &mut PromptCache, traces: &[[u32; 4]]) {
    let config = two_entry_config();
    for trace in traces {
        cache.store(entry_with_rows(trace), &config);
    }
}

fn widths() -> Vec<LayerPadRowWidths> {
    vec![
        LayerPadRowWidths::Attention { even_odd_row: 2, v_row: 1 },
        LayerPadRowWidths::Attention { even_odd_row: 2, v_row: 1 },
        LayerPadRowWidths::SharedFromLayer,
    ]
}

#[test]
fn tier_disk_demote_writes_the_entry_and_keeps_it_cold() {
    let directory = tempfile::tempdir().expect("a temporary spill directory");
    let original = entry_with_rows(&TRACE_A).state.branch();
    let mut cache = disk_cache(directory.path(), 2);

    store_all(&mut cache, &[TRACE_A, TRACE_B, TRACE_C]);

    assert_eq!(cache.held(), vec![(0, true), (1, false), (2, false)]);
    let cold = cache.entry(0).expect("the evicted entry stays in the cache");
    assert!(cold.state.layer_caches.is_empty());
    assert!(cold.checkpoints.is_empty());
    assert_eq!(cold.state.ids, TRACE_A);
    assert_eq!(spilled_names(directory.path()), vec!["0000000000000000.pxkv".to_owned()]);
    let path = directory.path().join("0000000000000000.pxkv");
    let mapped = MappedBlockFile::open(&path).expect("the spilled file maps");
    let key = mapped.view().expect("the spilled file decodes").header.content_key;
    let mut expected = Vec::new();
    original
        .to_block_file([0x22; 16], key, &mut expected)
        .expect("the original state encodes");
    assert_eq!(std::fs::read(&path).expect("the spilled file reads"), expected);
    let mut prompt = TRACE_A.to_vec();
    prompt.push(9);
    let (_, report) = cache.take_best(&prompt, &base_key(), &widths(), 0);
    assert_eq!(report.path, CachePath::Miss);
    assert_eq!(report.reused_tokens, 0);
    assert!(cache.held().contains(&(0, true)));
}

#[test]
fn tier_disk_files_leave_with_their_entries() {
    let cleared = tempfile::tempdir().expect("a temporary spill directory");
    let mut cache = disk_cache(cleared.path(), 2);
    store_all(&mut cache, &[TRACE_A, TRACE_B, TRACE_C]);
    assert_eq!(spilled_names(cleared.path()).len(), 1);

    cache.clear();

    assert!(spilled_names(cleared.path()).is_empty());
    let dropped = tempfile::tempdir().expect("a temporary spill directory");
    let mut cache = disk_cache(dropped.path(), 2);
    store_all(&mut cache, &[TRACE_A, TRACE_B, TRACE_C]);
    assert_eq!(spilled_names(dropped.path()).len(), 1);

    drop(cache);

    assert!(spilled_names(dropped.path()).is_empty());
}

#[test]
fn tier_disk_trims_cold_files_past_the_limit() {
    let directory = tempfile::tempdir().expect("a temporary spill directory");
    let mut cache = disk_cache(directory.path(), 1);

    store_all(&mut cache, &[TRACE_A, TRACE_B, TRACE_C, TRACE_D]);

    assert_eq!(cache.held(), vec![(1, true), (2, false), (3, false)]);
    assert_eq!(spilled_names(directory.path()), vec!["0000000000000001.pxkv".to_owned()]);
}

#[test]
fn tier_disk_drops_an_entry_whose_layers_have_no_row_planes() {
    let directory = tempfile::tempdir().expect("a temporary spill directory");
    let mut cache = disk_cache(directory.path(), 2);
    let config = two_entry_config();
    let ssm = CacheEntry::new(
        PrefixState {
            ids: TRACE_E.to_vec(),
            layer_caches: vec![LayerCacheState::Ssm(SsmLayerCache::new(2, 2))],
            cached_len: TRACE_E.len(),
        },
        base_key(),
    );

    cache.store(ssm, &config);
    cache.store(entry_with_rows(&TRACE_A), &config);
    cache.store(entry_with_rows(&TRACE_B), &config);

    assert!(!cache.held().iter().any(|(stamp, _)| *stamp == 0));
    assert!(spilled_names(directory.path()).is_empty());
}

fn prompt_after(trace: &[u32; 4]) -> Vec<u32> {
    let mut prompt = trace.to_vec();
    prompt.push(9);
    prompt
}

fn encoded(state: &PrefixState) -> Vec<u8> {
    let mut bytes = Vec::new();
    state
        .to_block_file([0x22; 16], 7, &mut bytes)
        .expect("the state encodes");
    bytes
}

#[test]
fn tier_round_trip_entry_survives_host_to_disk_to_host() {
    let directory = tempfile::tempdir().expect("a temporary spill directory");
    let original = entry_with_rows(&TRACE_A).state.branch();
    let config = two_entry_config();
    let shared = Mutex::new(disk_cache(directory.path(), 2));
    let first = "0000000000000000.pxkv".to_owned();
    let second = "0000000000000001.pxkv".to_owned();
    let third = "0000000000000002.pxkv".to_owned();
    let fourth = "0000000000000003.pxkv".to_owned();

    shared.lock().store(entry_with_rows(&TRACE_A), &config);
    assert_eq!(shared.lock().held(), vec![(0, false)]);
    assert!(spilled_names(directory.path()).is_empty());

    shared.lock().store(entry_with_rows(&TRACE_B), &config);
    assert_eq!(shared.lock().held(), vec![(0, false), (1, false)]);
    assert!(spilled_names(directory.path()).is_empty());

    shared.lock().store(entry_with_rows(&TRACE_C), &config);
    assert_eq!(shared.lock().held(), vec![(0, true), (1, false), (2, false)]);
    assert_eq!(spilled_names(directory.path()), vec![first.clone()]);

    let restored = thaw_best_cold(&shared, &prompt_after(&TRACE_A), &base_key(), &config);
    assert!(restored.expect("the disk read succeeds"));
    assert_eq!(shared.lock().held(), vec![(1, true), (2, false), (3, false)]);
    assert_eq!(spilled_names(directory.path()), vec![second.clone()]);
    let hot_bytes = encoded(&shared.lock().entry(3).expect("the restored entry is hot").state);
    assert_eq!(hot_bytes, encoded(&original));

    shared.lock().store(entry_with_rows(&TRACE_D), &config);
    assert_eq!(shared.lock().held(), vec![(1, true), (2, true), (3, false), (4, false)]);
    assert_eq!(spilled_names(directory.path()), vec![second, third.clone()]);

    shared.lock().store(entry_with_rows(&TRACE_E), &config);
    assert_eq!(shared.lock().held(), vec![(2, true), (3, true), (4, false), (5, false)]);
    assert_eq!(spilled_names(directory.path()), vec![third, fourth]);

    let prompt = prompt_after(&TRACE_B);
    let (_, report) = shared.lock().take_best(&prompt, &base_key(), &widths(), 0);
    assert_eq!(report.path, CachePath::Miss);
    let thawed = thaw_best_cold(&shared, &prompt, &base_key(), &config);
    assert!(!thawed.expect("no cold entry reaches the prompt"));
}

#[test]
fn tier_policy_oldest_rule_list_moves_a_different_entry_to_disk() {
    let config = two_entry_config();
    let branch_ids = [818, 5279, 529, 7001, 9, 9];
    let fill = |cache: &mut PromptCache| {
        cache.store(entry_with_rows(&TRACE_A), &config);
        let mut branch = entry_with_rows(&branch_ids);
        branch.branch_base = Some(4);
        cache.store(branch, &config);
        cache.store(entry_with_rows(&TRACE_C), &config);
    };
    let default_directory = tempfile::tempdir().expect("a temporary spill directory");
    let mut default_cache = disk_cache(default_directory.path(), 2);
    let oldest_directory = tempfile::tempdir().expect("a temporary spill directory");
    let mut oldest_cache = disk_cache(oldest_directory.path(), 2);
    oldest_cache
        .set_eviction_rules(&[EvictionRule::Oldest])
        .expect("a list ending in oldest is accepted");

    fill(&mut default_cache);
    fill(&mut oldest_cache);

    assert_eq!(default_cache.held(), vec![(0, false), (1, true), (2, false)]);
    assert_eq!(oldest_cache.held(), vec![(0, true), (1, false), (2, false)]);
}
