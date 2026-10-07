//! A bloom filter per cached entry over its blocks hashed on content alone
//! (`proxima-tensor/specs/prefix-cache-reuse/SPEC.md`).
//!
//! It answers "does this entry probably hold a block with this content
//! anywhere", the question a prompt whose history was squashed asks after its
//! prefix stopped matching; the prefix itself is found by
//! [`super::prefix_trie::PrefixTrie`], which is exact and position-bound.
//! [`PromptCache::bloom_candidates`](super::PromptCache) reports the answer;
//! moving the rows (the spec) is not built.
//!
//! Hashing is `xxh3` (the workspace's `xxhash-rust`) over the block's token
//! bytes.

use xxhash_rust::xxh3::xxh3_64_with_seed;

const CONTENT_SEED: u64 = 0;

fn block_hash(scratch: &mut Vec<u8>, block: &[u32], seed: u64) -> u64 {
    scratch.clear();
    scratch.extend(block.iter().flat_map(|token| token.to_le_bytes()));
    xxh3_64_with_seed(scratch, seed)
}

/// The content-only hash of every whole block of `ids`; the last partial
/// block has none.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blocks_content_hash_ignores_the_blocks_before_it() {
        let tail: Vec<u32> = (100..108).collect();
        let behind_head: Vec<u32> = (0..8).chain(tail.iter().copied()).collect();
        let behind_other: Vec<u32> = (50..58).chain(tail.iter().copied()).collect();

        let behind_head_hashes = content_hashes(&behind_head, 8);
        let behind_other_hashes = content_hashes(&behind_other, 8);

        assert_eq!(behind_head_hashes[1], behind_other_hashes[1]);
        assert_ne!(behind_head_hashes[0], behind_other_hashes[0]);
    }

    #[test]
    fn only_whole_blocks_are_hashed() {
        assert_eq!(content_hashes(&(0..20).collect::<Vec<u32>>(), 8).len(), 2);
        assert!(content_hashes(&[1, 2, 3], 8).is_empty());
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
