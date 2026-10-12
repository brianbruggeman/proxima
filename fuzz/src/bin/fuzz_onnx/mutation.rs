use proxima_fuzz_harness::Xorshift64;

const MAX_MUTATIONS_PER_DRAW: usize = 4;
const BITS_PER_BYTE: usize = 8;
const MUTATION_KINDS: usize = 4;

pub fn mutate(base: &[u8], generator: &mut Xorshift64) -> Vec<u8> {
    let mut bytes = base.to_vec();
    let mutations = 1 + generator.next_below(MAX_MUTATIONS_PER_DRAW);
    for _ in 0..mutations {
        apply_one_mutation(&mut bytes, generator);
    }
    bytes
}

fn apply_one_mutation(bytes: &mut Vec<u8>, generator: &mut Xorshift64) {
    let position = generator.next_below(bytes.len());
    let replacement = generator.next_u64().to_le_bytes()[0];
    match generator.next_below(MUTATION_KINDS) {
        0 => flip_bit(bytes, position, generator.next_below(BITS_PER_BYTE)),
        1 => overwrite(bytes, position, replacement),
        2 => bytes.truncate(position),
        _ => bytes.insert(position, replacement),
    }
}

fn flip_bit(bytes: &mut [u8], position: usize, bit: usize) {
    if let Some(byte) = bytes.get_mut(position) {
        *byte ^= 1 << bit;
    }
}

fn overwrite(bytes: &mut [u8], position: usize, replacement: u8) {
    if let Some(byte) = bytes.get_mut(position) {
        *byte = replacement;
    }
}
