//! the Q4_0 weight layout a pair-dot matvec reads, and the permutation that
//! produces it from ggml's interleaved blocks.
//!
//! ggml stores one Q4_0 block as 18 bytes: a 2-byte binary16 scale followed by
//! 16 bytes of packed nibbles. A matvec lane therefore reaches its 8 nibble
//! bytes through four `uint16` loads out of the middle of a block plus a
//! separate 2-byte scale load -- five memory instructions per block, and the
//! scale sits in front of the payload so the nibble stream is only 18-byte
//! aligned and can never be one vector load.
//!
//! this module permutes the SAME 18 bytes per block into 288-byte GROUPS of 16
//! blocks:
//!
//! ```text
//! [ blk0 nibbles (16) | ... | blk15 nibbles ] [ scale0 (2) | ... | scale15 (2) ]
//! <------------------ 256 -------------------> <------------ 32 ------------>
//! ```
//!
//! 16 is not arbitrary: the row-blocked matvec walks exactly 16 blocks per
//! iteration per lane pair (`ib_step = 16`, `[Q4_0_BLOCK_BYTES] * 16` = 288), so
//! a group is one loop iteration and the loop's own byte step is unchanged.
//! A WHOLE-TENSOR nibble plane would put the scales at `total_blocks * 16`,
//! which the kernel cannot know without a new uniform; grouping by the loop's
//! own stride keeps the addressing local to `group * 288`.
//!
//! the only cost is the one-time permutation at load.
//!
//! a lane then reads one 8-byte vector (its `il` half of its own block's
//! nibbles, 8-byte aligned inside the group) plus one 2-byte scale: two
//! instructions instead of five.

#![expect(
    dead_code,
    reason = "no bind-side repack caller yet: the layout and its permutation are \
              implemented and value-verified, but wiring the model bind to supply \
              repacked bytes (and select this body) is the remaining piece"
)]

use alloc::vec;
use alloc::vec::Vec;

use super::kernel_types_identity::{Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMENTS};

/// binary16 scale bytes that lead each interleaved block.
pub(crate) const SCALE_BYTES: usize = 2;

/// packed-nibble bytes an interleaved block carries after its scale.
pub(crate) const NIBBLE_BYTES: usize = Q4_0_BLOCK_BYTES - SCALE_BYTES;

/// blocks one repacked group holds -- the row-blocked matvec's own
/// `ib_step`, so a group is one loop iteration.
pub(crate) const GROUP_BLOCKS: usize = 16;

/// nibble bytes of one group, laid out block 0..15 back to back.
pub(crate) const GROUP_NIBBLE_BYTES: usize = GROUP_BLOCKS * NIBBLE_BYTES;

/// scale bytes of one group, laid out block 0..15 back to back.
pub(crate) const GROUP_SCALE_BYTES: usize = GROUP_BLOCKS * SCALE_BYTES;

/// bytes one whole group occupies: `GROUP_BLOCKS * Q4_0_BLOCK_BYTES`, so a
/// repack never changes a buffer's size.
pub(crate) const GROUP_BYTES: usize = GROUP_NIBBLE_BYTES + GROUP_SCALE_BYTES;

/// whether a tensor's block count can be grouped without padding. a Q4_0 tensor
/// whose reduction is a multiple of `GROUP_BLOCKS * Q4_0_BLOCK_ELEMENTS` (512)
/// always can; anything else stays interleaved rather than growing a buffer.
pub(crate) const fn repackable(block_count: usize) -> bool {
    block_count % GROUP_BLOCKS == 0
}

/// bytes a repacked buffer of `block_count` blocks needs -- equal to the
/// interleaved size for every [`repackable`] count.
pub(crate) const fn repacked_len(block_count: usize) -> usize {
    block_count * Q4_0_BLOCK_BYTES
}

/// interleaved `[scale | nibbles]` blocks into [`GROUP_BLOCKS`]-block groups.
/// panics on a length that is not a whole number of groups, which the caller
/// gates with [`repackable`] rather than discovering here.
pub(crate) fn repack(blocks: &[u8]) -> Vec<u8> {
    assert_eq!(
        blocks.len() % Q4_0_BLOCK_BYTES,
        0,
        "repack needs whole 18-byte blocks; got {} bytes",
        blocks.len(),
    );
    let block_count = blocks.len() / Q4_0_BLOCK_BYTES;
    assert!(
        repackable(block_count),
        "repack needs a multiple of {GROUP_BLOCKS} blocks; got {block_count}",
    );
    let mut groups = vec![0u8; repacked_len(block_count)];
    for group in 0..block_count / GROUP_BLOCKS {
        let group_at = group * GROUP_BYTES;
        for slot in 0..GROUP_BLOCKS {
            let block = group * GROUP_BLOCKS + slot;
            let source = &blocks[block * Q4_0_BLOCK_BYTES..(block + 1) * Q4_0_BLOCK_BYTES];
            let nibble_at = group_at + slot * NIBBLE_BYTES;
            groups[nibble_at..nibble_at + NIBBLE_BYTES].copy_from_slice(&source[SCALE_BYTES..]);
            let scale_at = group_at + GROUP_NIBBLE_BYTES + slot * SCALE_BYTES;
            groups[scale_at..scale_at + SCALE_BYTES].copy_from_slice(&source[..SCALE_BYTES]);
        }
    }
    groups
}

/// the inverse of [`repack`]. the A/B gate uses it to prove the permutation is
/// lossless against the interleaved dequantizer, so the repacked kernel needs no
/// second oracle.
pub(crate) fn unrepack(groups: &[u8], block_count: usize) -> Vec<u8> {
    assert_eq!(
        groups.len(),
        repacked_len(block_count),
        "unrepack needs {} bytes for {block_count} blocks; got {}",
        repacked_len(block_count),
        groups.len(),
    );
    let mut blocks = vec![0u8; block_count * Q4_0_BLOCK_BYTES];
    for group in 0..block_count / GROUP_BLOCKS {
        let group_at = group * GROUP_BYTES;
        for slot in 0..GROUP_BLOCKS {
            let block = group * GROUP_BLOCKS + slot;
            let target = &mut blocks[block * Q4_0_BLOCK_BYTES..(block + 1) * Q4_0_BLOCK_BYTES];
            let nibble_at = group_at + slot * NIBBLE_BYTES;
            target[SCALE_BYTES..].copy_from_slice(&groups[nibble_at..nibble_at + NIBBLE_BYTES]);
            let scale_at = group_at + GROUP_NIBBLE_BYTES + slot * SCALE_BYTES;
            target[..SCALE_BYTES].copy_from_slice(&groups[scale_at..scale_at + SCALE_BYTES]);
        }
    }
    blocks
}

/// whether the Q4_0 matvec should read [`repack`]'s grouped layout rather than
/// ggml's interleaved blocks. default OFF: the unset build emits the identical
/// body it always has, and turning this on without also repacking the bytes at
/// bind reads garbage -- the two switches are one decision, which is why the A/B
/// pairs them.
#[cfg(all(feature = "std", feature = "metal-q4_0-grouped"))]
pub(crate) fn grouped_from_env() -> bool {
    matches!(std::env::var("PROXIMA_Q4_0_GROUPED"), Ok(value) if value.trim() == "1")
}

#[cfg(all(not(feature = "std"), feature = "metal-q4_0-grouped"))]
pub(crate) const fn grouped_from_env() -> bool {
    false
}

/// elements one group covers, for a caller sizing its block index from an
/// element offset.
pub(crate) const GROUP_ELEMENTS: usize = GROUP_BLOCKS * Q4_0_BLOCK_ELEMENTS;

#[cfg(test)]
mod tests {
    use super::*;
    use proxima_gguf::quant::q4_0;

    /// deterministic bytes with no dependence on a rand crate: two
    /// multiplicative steps over a 32-bit state.
    fn bytes(seed: u32, len: usize) -> Vec<u8> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 16) as u8
            })
            .collect()
    }

    fn dequantized_bits(blocks: &[u8]) -> Vec<u32> {
        let mut values = vec![0.0_f32; q4_0::elements_for_blocks(blocks.len() / q4_0::BLOCK_BYTES)];
        q4_0::dequantize(blocks, &mut values).expect("whole blocks dequantize");
        // bit equality, not float equality: arbitrary scale bytes can encode
        // f16 NaN, and a permutation test must still see the two sides as equal
        values.into_iter().map(f32::to_bits).collect()
    }

    #[test]
    fn repack_then_unrepack_returns_the_original_bytes() {
        for block_count in [16_usize, 32, 256, 4096] {
            let original = bytes(block_count as u32, block_count * q4_0::BLOCK_BYTES);
            let restored = unrepack(&repack(&original), block_count);
            assert_eq!(restored, original, "round trip must be byte-exact");
        }
    }

    #[test]
    fn a_repacked_tensor_dequantizes_to_the_same_values_as_the_interleaved_one() {
        // the value-preservation gate: the repacked matvec reuses the SAME
        // dequantizer these bytes feed, so equality here means the kernel's
        // nibble/scale addresses are the only thing the layout changes.
        let block_count = 1024;
        let original = bytes(0x5eed, block_count * q4_0::BLOCK_BYTES);
        let groups = repack(&original);
        assert_eq!(
            dequantized_bits(&unrepack(&groups, block_count)),
            dequantized_bits(&original),
            "a repack must not move a single value",
        );
    }

    #[test]
    fn each_lane_half_of_each_block_is_one_aligned_vector_load() {
        // the reason the layout exists: block slot s of group g starts at
        // g * 288 + s * 16, so both 8-byte halves are 8-byte aligned and sit in
        // one 16-byte run, and the 16 scales are one contiguous 32-byte tail.
        let block_count = GROUP_BLOCKS * 2;
        let original = bytes(11, block_count * q4_0::BLOCK_BYTES);
        let groups = repack(&original);
        for group in 0..block_count / GROUP_BLOCKS {
            let group_at = group * GROUP_BYTES;
            for slot in 0..GROUP_BLOCKS {
                let payload_at = group_at + slot * NIBBLE_BYTES;
                assert_eq!(payload_at % 16, 0, "payload offsets must be 16-byte aligned");
                let block = group * GROUP_BLOCKS + slot;
                assert_eq!(
                    &groups[payload_at..payload_at + NIBBLE_BYTES],
                    &original[block * Q4_0_BLOCK_BYTES + SCALE_BYTES
                        ..(block + 1) * Q4_0_BLOCK_BYTES],
                    "block {block} payload must sit at its own group offset",
                );
                let scale_at = group_at + GROUP_NIBBLE_BYTES + slot * SCALE_BYTES;
                assert_eq!(
                    &groups[scale_at..scale_at + SCALE_BYTES],
                    &original[block * Q4_0_BLOCK_BYTES..block * Q4_0_BLOCK_BYTES + SCALE_BYTES],
                    "block {block} scale must sit in the group's scale tail",
                );
            }
        }
        assert_eq!(GROUP_BYTES, GROUP_BLOCKS * Q4_0_BLOCK_BYTES);
        assert_eq!(GROUP_NIBBLE_BYTES + GROUP_SCALE_BYTES, GROUP_BYTES);
    }

    #[test]
    fn a_repacked_buffer_is_the_same_size_as_the_interleaved_one() {
        // no extra device memory: a repack is a permutation, never a widening.
        for block_count in [16_usize, 512, 1024] {
            assert!(repackable(block_count));
            assert_eq!(
                repacked_len(block_count),
                block_count * q4_0::BLOCK_BYTES,
                "repack must not change the buffer size",
            );
        }
    }

    #[test]
    fn a_block_count_off_the_group_boundary_is_not_repackable() {
        // K not a multiple of 512 (one group's worth of elements) declines
        // rather than padding a buffer: 15 blocks cannot fill a group.
        for block_count in [1_usize, 15, 17, 1001] {
            assert!(!repackable(block_count), "{block_count} must decline");
        }
        assert!(repackable(GROUP_BLOCKS));
        assert_eq!(GROUP_ELEMENTS, GROUP_BLOCKS * q4_0::QK4_0);
    }
}
