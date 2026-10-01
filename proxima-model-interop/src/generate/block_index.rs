//! Finding the cached entry that shares the longest prefix with a prompt
//! without comparing the prompt against every entry
//! (`proxima-tensor/specs/prefix-cache-reuse/SPEC.md` R13, R14).
//!
//! Every cached entry's ids are cut into [`PromptCacheConfig::block_tokens`]
//! blocks, each hashed together with the hash of the block before it, so one
//! hash names the whole prefix up to and including its block (the chained
//! block hashes llama.cpp's prompt cache and vLLM's prefix cache use). A
//! request's blocks are hashed the same way and looked up until the first
//! one no entry holds; the entries still matching there are refined token by
//! token past the last whole block. A hash only nominates: every nominated
//! block is compared against the entry's ids before it counts, so a collision
//! can cost a comparison and never a wrong reuse.
//!
//! Beside the chain, each entry carries a bloom filter of its blocks hashed on
//! content alone, with no parent: it answers "does this entry probably hold a
//! block with this content anywhere", the question a prompt whose history was
//! squashed asks after its prefix stopped matching. [`PromptCache::
//! bloom_candidates`](super::PromptCache) reports the answer; moving the rows
//! (spec R5) is not built.
//!
//! Hashing is `xxh3` (the workspace's `xxhash-rust`), seeded with the parent
//! hash for the chain.

use std::collections::HashMap;

use xxhash_rust::xxh3::xxh3_64_with_seed;

const CONTENT_SEED: u64 = 0;

fn block_hash(scratch: &mut Vec<u8>, block: &[u32], seed: u64) -> u64 {
    scratch.clear();
    scratch.extend(block.iter().flat_map(|token| token.to_le_bytes()));
    xxh3_64_with_seed(scratch, seed)
}

/// The chained hash of every whole block of `ids`; the last partial block has
/// none.
pub(super) fn chained_hashes(ids: &[u32], block_tokens: usize) -> Vec<u64> {
    let mut scratch = Vec::with_capacity(block_tokens * size_of::<u32>());
    let mut parent = CONTENT_SEED;
    ids.chunks_exact(block_tokens)
        .map(|block| {
            parent = block_hash(&mut scratch, block, parent);
            parent
        })
        .collect()
}

/// The content-only hash of every whole block of `ids`.
pub(super) fn content_hashes(ids: &[u32], block_tokens: usize) -> Vec<u64> {
    let mut scratch = Vec::with_capacity(block_tokens * size_of::<u32>());
    ids.chunks_exact(block_tokens)
        .map(|block| block_hash(&mut scratch, block, CONTENT_SEED))
        .collect()
}

/// A fixed-size bloom filter over 64-bit hashes (which are already uniform, so
/// the probes are double hashing over the one value).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct BlockBloom {
    words: Vec<u64>,
    hashes: u32,
}

impl BlockBloom {
    /// The filter over the whole-block content hashes of `ids`.
    pub(super) fn of(ids: &[u32], block_tokens: usize, bits: u32, hashes: u32) -> Self {
        let mut bloom = Self::new(bits, hashes);
        content_hashes(ids, block_tokens)
            .into_iter()
            .for_each(|hash| bloom.insert(hash));
        bloom
    }

    pub(super) fn new(bits: u32, hashes: u32) -> Self {
        Self {
            words: vec![0; (bits as usize).div_ceil(64).max(1)],
            hashes: hashes.max(1),
        }
    }

    fn bit_count(&self) -> u64 {
        self.words.len() as u64 * 64
    }

    fn probes(&self, hash: u64) -> impl Iterator<Item = u64> + use<> {
        let bits = self.bit_count();
        let step = hash.rotate_left(32) | 1;
        (0..u64::from(self.hashes))
            .map(move |round| hash.wrapping_add(round.wrapping_mul(step)) % bits)
    }

    pub(super) fn insert(&mut self, hash: u64) {
        for bit in self.probes(hash) {
            self.words[(bit / 64) as usize] |= 1 << (bit % 64);
        }
    }

    pub(super) fn maybe_contains(&self, hash: u64) -> bool {
        self.probes(hash)
            .all(|bit| self.words[(bit / 64) as usize] & (1 << (bit % 64)) != 0)
    }

    pub(super) fn byte_len(&self) -> usize {
        self.words.capacity() * size_of::<u64>()
    }
}

/// What a walk of the chain found: the entries still matching after each
/// whole block, deepest last.
pub(super) struct ChainWalk {
    /// `levels[depth]` holds the stamps of the entries whose first
    /// `depth + 1` blocks equal the prompt's.
    pub(super) levels: Vec<Vec<u64>>,
}

/// Chained block hash to the stamps of the entries holding that block, and
/// the stamps by first token for prompts that share less than a block.
#[derive(Debug, Default)]
pub(super) struct BlockIndex {
    block_tokens: usize,
    chained: HashMap<u64, Vec<u64>>,
    by_first_token: HashMap<u32, Vec<u64>>,
    short: Vec<u64>,
}

impl BlockIndex {
    pub(super) fn new(block_tokens: usize) -> Self {
        Self {
            block_tokens: block_tokens.max(1),
            chained: HashMap::new(),
            by_first_token: HashMap::new(),
            short: Vec::new(),
        }
    }

    pub(super) const fn block_tokens(&self) -> usize {
        self.block_tokens
    }

    pub(super) fn insert(&mut self, stamp: u64, ids: &[u32]) {
        for hash in chained_hashes(ids, self.block_tokens) {
            self.chained.entry(hash).or_default().push(stamp);
        }
        if let Some(first) = ids.first() {
            self.by_first_token.entry(*first).or_default().push(stamp);
        }
        if ids.len() < self.block_tokens {
            self.short.push(stamp);
        }
    }

    pub(super) fn remove(&mut self, stamp: u64, ids: &[u32]) {
        for hash in chained_hashes(ids, self.block_tokens) {
            Self::forget(&mut self.chained, hash, stamp);
        }
        if let Some(first) = ids.first() {
            Self::forget(&mut self.by_first_token, *first, stamp);
        }
        self.short.retain(|held| *held != stamp);
    }

    fn forget<Key: core::hash::Hash + Eq>(map: &mut HashMap<Key, Vec<u64>>, key: Key, stamp: u64) {
        if let Some(stamps) = map.get_mut(&key) {
            stamps.retain(|held| *held != stamp);
            if stamps.is_empty() {
                map.remove(&key);
            }
        }
    }

    #[cfg(test)]
    pub(super) fn byte_len(&self) -> usize {
        Self::table_bytes::<u64>(&self.chained)
            + Self::table_bytes::<u32>(&self.by_first_token)
            + self.short.capacity() * size_of::<u64>()
    }

    #[cfg(test)]
    fn table_bytes<Key>(map: &HashMap<Key, Vec<u64>>) -> usize {
        map.capacity() * (size_of::<Key>() + size_of::<Vec<u64>>() + 1)
            + map
                .values()
                .map(|stamps| stamps.capacity() * size_of::<u64>())
                .sum::<usize>()
    }

    /// The stamps of entries too short to hold a whole block.
    pub(super) fn shorter_than_a_block(&self) -> &[u64] {
        &self.short
    }

    /// The stamps of entries sharing the prompt's first token.
    pub(super) fn sharing_first_token(&self, prompt_ids: &[u32]) -> &[u64] {
        prompt_ids
            .first()
            .and_then(|first| self.by_first_token.get(first))
            .map_or(&[], Vec::as_slice)
    }

    /// Walks the prompt's chained block hashes until no entry holds the next
    /// block. `holds(stamp, depth)` says whether that entry's ids really
    /// carry the prompt's block `depth` (the hash only nominates).
    pub(super) fn walk(
        &self,
        prompt_ids: &[u32],
        mut holds: impl FnMut(u64, usize) -> bool,
    ) -> ChainWalk {
        let mut levels: Vec<Vec<u64>> = Vec::new();
        for (depth, hash) in chained_hashes(prompt_ids, self.block_tokens)
            .into_iter()
            .enumerate()
        {
            let matching: Vec<u64> = self
                .chained
                .get(&hash)
                .into_iter()
                .flatten()
                .copied()
                .filter(|stamp| {
                    depth == 0
                        || levels
                            .last()
                            .is_some_and(|previous| previous.contains(stamp))
                })
                .filter(|stamp| holds(*stamp, depth))
                .collect();
            if matching.is_empty() {
                break;
            }
            levels.push(matching);
        }
        ChainWalk { levels }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blocks_chained_hash_depends_on_every_block_before_it() {
        let head: Vec<u32> = (0..8).collect();
        let tail: Vec<u32> = (100..108).collect();
        let other_head: Vec<u32> = (50..58).collect();
        let behind_head: Vec<u32> = head.iter().chain(&tail).copied().collect();
        let behind_other: Vec<u32> = other_head.iter().chain(&tail).copied().collect();

        let chained_a = chained_hashes(&behind_head, 8);
        let chained_b = chained_hashes(&behind_other, 8);

        assert_ne!(chained_a[1], chained_b[1], "same block, different parents");
        assert_eq!(
            content_hashes(&behind_head, 8)[1],
            content_hashes(&behind_other, 8)[1],
            "content hashes ignore the parents"
        );
    }

    #[test]
    fn only_whole_blocks_are_hashed() {
        assert_eq!(chained_hashes(&(0..20).collect::<Vec<u32>>(), 8).len(), 2);
        assert!(chained_hashes(&[1, 2, 3], 8).is_empty());
    }

    #[test]
    fn a_bloom_filter_never_forgets_an_inserted_hash() {
        let mut bloom = BlockBloom::new(1024, 4);
        let hashes = content_hashes(&(0..640).collect::<Vec<u32>>(), 64);

        hashes.iter().for_each(|hash| bloom.insert(*hash));

        assert_eq!(hashes.len(), 10);
        assert!(hashes.iter().all(|hash| bloom.maybe_contains(*hash)));
    }
}
