#![allow(clippy::expect_used)]

use serde_json::Value;

use super::block_bloom::content_hashes;
use super::prompt_cache::{BloomCandidates, CacheEntry, PromptCache};
use super::prompt_cache_real_model_tests::{
    cached_config, encode_continuation, encode_opening, run_cached, with_model,
};
use super::*;
use crate::serving::{PromptCacheConfig, SpeculativeConfig};

const FOLLOW_UPS: &str = include_str!("../../examples/data/follow_up_transcripts.jsonl");
const SUMMARY: &str = "<turn|>\n<|turn>model\nSo far the user asked a few questions and got answers.<turn|>\n<|turn>user\n";

struct Chat {
    id: String,
    opening: Vec<u32>,
    full: Vec<u32>,
    last_user: Vec<u32>,
}

fn chats() -> Vec<(String, Vec<String>)> {
    FOLLOW_UPS
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("a transcript line is json"))
        .map(|record| {
            let turns = record["turns"]
                .as_array()
                .expect("turns is a list")
                .iter()
                .map(|turn| turn.as_str().expect("a turn is text").to_owned())
                .collect();
            (record["id"].as_str().expect("id").to_owned(), turns)
        })
        .collect()
}

/// A chat answered by the real model, turn by turn, the way a client renders it.
fn converse(model: &LoadedModel<'_>, id: &str, turns: &[String]) -> Chat {
    let config = cached_config(SpeculativeConfig::none());
    let boundary = encode_continuation(model, "<turn|>\n<|turn>user\n");
    let opening = encode_opening(
        model,
        &format!("<|turn>user\n{}<turn|>\n<|turn>model\n", turns[0]),
    );
    let mut full = opening.clone();
    let mut last_user = opening.clone();
    for (turn, user_text) in turns.iter().enumerate() {
        if turn > 0 {
            last_user = encode_continuation(model, &format!("{user_text}<turn|>\n<|turn>model\n"));
            full.extend_from_slice(&last_user);
        }
        let answer = run_cached(model, config, &full);
        let text: Vec<u32> = answer
            .generated
            .iter()
            .copied()
            .filter(|token| *token != 106)
            .collect();
        full.extend_from_slice(&text);
        full.extend_from_slice(&boundary);
    }
    Chat {
        id: id.to_owned(),
        opening,
        full,
        last_user,
    }
}

fn entry_of(ids: &[u32]) -> CacheEntry {
    CacheEntry::new(
        PrefixState {
            ids: ids.to_vec(),
            layer_caches: vec![LayerCacheState::SharedFromLayer],
            cached_len: ids.len(),
        },
        CacheKey::of(&ServingConfig::default(), false, RopeScaling::None, 0, 0),
    )
}

fn holds_block(entry_ids: &[u32], block: &[u32], block_tokens: usize) -> bool {
    entry_ids
        .chunks_exact(block_tokens)
        .any(|held| held == block)
}

/// Prompt positions, from `from`, that start a `window`-token run some entry
/// holds at any offset: the content a chunk shift could move.
fn sliding_windows(prompt: &[u32], from: usize, entries: &[&Chat], window: usize) -> usize {
    (from..prompt.len().saturating_sub(window - 1))
        .filter(|start| {
            entries.iter().any(|chat| {
                chat.full
                    .windows(window)
                    .any(|held| held == &prompt[*start..start + window])
            })
        })
        .count()
}

/// AC18: squash each of the 8 follow-up chats the way a client does -- keep
/// the opening turn and the last user turn, replace what lies between with a
/// short summary -- and ask the cache what is still reusable. The chain finds
/// the kept opening; the bloom filters are asked for the last user turn's
/// blocks and are compared with an exact comparison of every aligned block.
#[test]
#[ignore = "depends on a host-local gemma4-E2B gguf blob outside this repo, and a real Metal device"]
fn squashed_transcripts_against_the_bloom_filters_and_an_exact_block_comparison() {
    with_model(|model| {
        let conversations: Vec<Chat> = chats()
            .iter()
            .map(|(id, turns)| converse(model, id, turns))
            .collect();
        let summary = encode_continuation(model, SUMMARY);
        let key = CacheKey::of(&ServingConfig::default(), false, RopeScaling::None, 0, 0);
        for block_tokens in [64_u32, 32, 16] {
            let config = PromptCacheConfig {
                block_tokens,
                max_entries: 64,
                byte_budget: 1 << 30,
                ..PromptCacheConfig::standard()
            };
            let mut cache = PromptCache::new();
            conversations
                .iter()
                .for_each(|chat| assert!(cache.store(entry_of(&chat.full), &config).is_some()));
            let (mut false_positive_pairs, mut negative_pairs, mut missed_pairs) =
                (0_usize, 0_usize, 0_usize);
            let (mut bloom_blocks, mut exact_blocks, mut sliding, mut chain_blocks) = (0, 0, 0, 0);
            for chat in &conversations {
                let squashed: Vec<u32> = chat
                    .opening
                    .iter()
                    .chain(&summary)
                    .chain(&chat.last_user)
                    .copied()
                    .collect();
                let walk = cache_walk(&cache, &squashed);
                let kept = walk * block_tokens as usize;
                let BloomCandidates {
                    entries: bloom_entries,
                    blocks,
                } = cache.bloom_candidates(&squashed, &key, kept);
                let from_block = kept.div_ceil(block_tokens as usize);
                let prompt_blocks: Vec<&[u32]> =
                    squashed.chunks_exact(block_tokens as usize).collect();
                for (stamp_index, other) in conversations.iter().enumerate() {
                    for (block_index, block) in prompt_blocks.iter().enumerate().skip(from_block) {
                        let truth = holds_block(&other.full, block, block_tokens as usize);
                        let said_yes = blocks.iter().any(|(stamp, found)| {
                            *found == block_index && *stamp as usize == stamp_index
                        });
                        negative_pairs += usize::from(!truth);
                        false_positive_pairs += usize::from(said_yes && !truth);
                        missed_pairs += usize::from(truth && !said_yes);
                    }
                }
                let exact = prompt_blocks
                    .iter()
                    .enumerate()
                    .skip(from_block)
                    .filter(|(_, block)| {
                        conversations
                            .iter()
                            .any(|other| holds_block(&other.full, block, block_tokens as usize))
                    })
                    .count();
                let windows = sliding_windows(
                    &squashed,
                    kept,
                    &conversations.iter().collect::<Vec<_>>(),
                    block_tokens as usize,
                );
                println!(
                    "SQUASH block_tokens={block_tokens} chat={} prompt_tokens={} chain_blocks={walk} bloom_entries={bloom_entries} bloom_blocks={} exact_aligned_blocks={exact} sliding_window_positions={windows} content_hash_blocks={}",
                    chat.id,
                    squashed.len(),
                    blocks.len(),
                    content_hashes(&squashed, block_tokens as usize).len()
                );
                bloom_blocks += blocks.len();
                exact_blocks += exact;
                sliding += windows;
                chain_blocks += walk;
            }
            println!(
                "SQUASH_TOTAL block_tokens={block_tokens} chats={} chain_blocks={chain_blocks} bloom_candidate_blocks={bloom_blocks} exact_aligned_blocks={exact_blocks} sliding_window_positions={sliding} false_positive_pairs={false_positive_pairs} negative_pairs={negative_pairs} fp_rate={:.5} missed_pairs={missed_pairs}",
                conversations.len(),
                false_positive_pairs as f64 / negative_pairs.max(1) as f64
            );
            assert_eq!(
                missed_pairs, 0,
                "a bloom filter must never miss a block it holds"
            );
        }
    });
}

fn cache_walk(cache: &PromptCache, prompt: &[u32]) -> usize {
    cache.matched_blocks(prompt)
}
