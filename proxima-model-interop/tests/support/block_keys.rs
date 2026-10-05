use xxhash_rust::xxh3::xxh3_64_with_seed;

// a hit must prove the whole prefix equal, so each key folds in the previous key; content_hashes stays position independent for chunk shifting
pub fn chained_keys(ids: &[u32], block_tokens: usize, seed: u64) -> Vec<u64> {
    if block_tokens == 0 {
        return Vec::new();
    }
    let mut buffer: Vec<u8> = Vec::with_capacity(8 + block_tokens * 4);
    let mut keys = Vec::with_capacity(ids.len() / block_tokens);
    for block in ids.chunks_exact(block_tokens) {
        let previous = keys.last().copied();
        buffer.clear();
        buffer.extend(previous.iter().flat_map(|key: &u64| key.to_le_bytes()));
        buffer.extend(block.iter().flat_map(|id| id.to_le_bytes()));
        keys.push(xxh3_64_with_seed(&buffer, seed));
    }
    keys
}
