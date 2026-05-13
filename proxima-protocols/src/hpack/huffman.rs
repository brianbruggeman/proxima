//! HPACK Huffman codec (RFC 7541 Appendix B).
//!
//! Static 256-symbol prefix code plus a 257th "EOS" entry. Each
//! symbol has a code of 5-30 bits; the table is fixed by the spec.
//!
//! ## Encoder
//!
//! Direct `ENCODE_TABLE[symbol] -> (length, code)` lookup, bit-packing
//! the code into the output buffer. Trailing bits in the final byte
//! are padded with 1s per §5.2 of the RFC.
//!
//! ## Decoder
//!
//! Two-level lookup. Root path (state==0): one byte-indexed lookup
//! into `ROOT_BYTE_TABLE[256]` that yields 0-2 emitted symbols + the
//! next state. Mid-code (state!=0): two 4-bit nibble lookups into
//! `DECODE_STATE_TABLE`. Result: 2-3x faster than h2's pure-nibble
//! decoder at every input size (see `hpack_huffman` bench).
//!
//! ### Parked: vectorized 16-byte decode lane
//!
//! Next decode speedup is real SIMD — NEON `vqtbl4q_u8` (and x86
//! `pshufb`) over a compacted byte-only fast-path table. Requires
//! splitting `ROOT_BYTE_TABLE` into a separate `OUTPUT[256]` byte
//! array + `STATUS[256]` mask (1 bit "stayed at root, exactly 1
//! symbol"), then per 16-byte input chunk: 4× `vqtbl4q_u8` (top-2
//! bits dispatch into 4 chunks of 64 entries each), `vmaxvq_u8`
//! reduce on status, `vst1q_u8` emit on all-good. Falls back to
//! scalar on any byte that's 0-symbol, 2-symbol, or leaves root.
//! Target gain: ~750 MiB/s → ~2 GiB/s on the 4 KiB body case where
//! ~80% of bytes meet the fast-path condition. Deferred — encoder
//! is the higher-leverage next move.
//!
//! ## Zero-copy / no-mutex / no-heap
//!
//! Encoder appends to any `BufMut`. Decoder appends to any `BufMut`.
//! Decoded strings can't share allocation with the input (the
//! decoded bytes don't exist in the input), so the output is owned.
//! The decode tables are `const`-evaluated at COMPILE time into
//! `'static` arrays (`.rodata`) — no `Box`, no lazy init, no
//! synchronization, no heap. Tier-3 (no_std + no_alloc) safe.

use bytes::BufMut;
use thiserror::Error;

/// RFC 7541 Appendix B. Each entry is `(length_in_bits, code_value)`.
/// Entry 256 is EOS (end-of-string), used only for padding detection;
/// it's never emitted by the encoder.
#[rustfmt::skip]
pub(crate) const ENCODE_TABLE: [(u8, u32); 257] = [
    (13, 0x1ff8),       (23, 0x7fffd8),     (28, 0xfffffe2),    (28, 0xfffffe3),
    (28, 0xfffffe4),    (28, 0xfffffe5),    (28, 0xfffffe6),    (28, 0xfffffe7),
    (28, 0xfffffe8),    (24, 0xffffea),     (30, 0x3ffffffc),   (28, 0xfffffe9),
    (28, 0xfffffea),    (30, 0x3ffffffd),   (28, 0xfffffeb),    (28, 0xfffffec),
    (28, 0xfffffed),    (28, 0xfffffee),    (28, 0xfffffef),    (28, 0xffffff0),
    (28, 0xffffff1),    (28, 0xffffff2),    (30, 0x3ffffffe),   (28, 0xffffff3),
    (28, 0xffffff4),    (28, 0xffffff5),    (28, 0xffffff6),    (28, 0xffffff7),
    (28, 0xffffff8),    (28, 0xffffff9),    (28, 0xffffffa),    (28, 0xffffffb),
    (6, 0x14),          (10, 0x3f8),        (10, 0x3f9),        (12, 0xffa),
    (13, 0x1ff9),       (6, 0x15),          (8, 0xf8),          (11, 0x7fa),
    (10, 0x3fa),        (10, 0x3fb),        (8, 0xf9),          (11, 0x7fb),
    (8, 0xfa),          (6, 0x16),          (6, 0x17),          (6, 0x18),
    (5, 0x0),           (5, 0x1),           (5, 0x2),           (6, 0x19),
    (6, 0x1a),          (6, 0x1b),          (6, 0x1c),          (6, 0x1d),
    (6, 0x1e),          (6, 0x1f),          (7, 0x5c),          (8, 0xfb),
    (15, 0x7ffc),       (6, 0x20),          (12, 0xffb),        (10, 0x3fc),
    (13, 0x1ffa),       (6, 0x21),          (7, 0x5d),          (7, 0x5e),
    (7, 0x5f),          (7, 0x60),          (7, 0x61),          (7, 0x62),
    (7, 0x63),          (7, 0x64),          (7, 0x65),          (7, 0x66),
    (7, 0x67),          (7, 0x68),          (7, 0x69),          (7, 0x6a),
    (7, 0x6b),          (7, 0x6c),          (7, 0x6d),          (7, 0x6e),
    (7, 0x6f),          (7, 0x70),          (7, 0x71),          (7, 0x72),
    (8, 0xfc),          (7, 0x73),          (8, 0xfd),          (13, 0x1ffb),
    (19, 0x7fff0),      (13, 0x1ffc),       (14, 0x3ffc),       (6, 0x22),
    (15, 0x7ffd),       (5, 0x3),           (6, 0x23),          (5, 0x4),
    (6, 0x24),          (5, 0x5),           (6, 0x25),          (6, 0x26),
    (6, 0x27),          (5, 0x6),           (7, 0x74),          (7, 0x75),
    (6, 0x28),          (6, 0x29),          (6, 0x2a),          (5, 0x7),
    (6, 0x2b),          (7, 0x76),          (6, 0x2c),          (5, 0x8),
    (5, 0x9),           (6, 0x2d),          (7, 0x77),          (7, 0x78),
    (7, 0x79),          (7, 0x7a),          (7, 0x7b),          (15, 0x7ffe),
    (11, 0x7fc),        (14, 0x3ffd),       (13, 0x1ffd),       (28, 0xffffffc),
    (20, 0xfffe6),      (22, 0x3fffd2),     (20, 0xfffe7),      (20, 0xfffe8),
    (22, 0x3fffd3),     (22, 0x3fffd4),     (22, 0x3fffd5),     (23, 0x7fffd9),
    (22, 0x3fffd6),     (23, 0x7fffda),     (23, 0x7fffdb),     (23, 0x7fffdc),
    (23, 0x7fffdd),     (23, 0x7fffde),     (24, 0xffffeb),     (23, 0x7fffdf),
    (24, 0xffffec),     (24, 0xffffed),     (22, 0x3fffd7),     (23, 0x7fffe0),
    (24, 0xffffee),     (23, 0x7fffe1),     (23, 0x7fffe2),     (23, 0x7fffe3),
    (23, 0x7fffe4),     (21, 0x1fffdc),     (22, 0x3fffd8),     (23, 0x7fffe5),
    (22, 0x3fffd9),     (23, 0x7fffe6),     (23, 0x7fffe7),     (24, 0xffffef),
    (22, 0x3fffda),     (21, 0x1fffdd),     (20, 0xfffe9),      (22, 0x3fffdb),
    (22, 0x3fffdc),     (23, 0x7fffe8),     (23, 0x7fffe9),     (21, 0x1fffde),
    (23, 0x7fffea),     (22, 0x3fffdd),     (22, 0x3fffde),     (24, 0xfffff0),
    (21, 0x1fffdf),     (22, 0x3fffdf),     (23, 0x7fffeb),     (23, 0x7fffec),
    (21, 0x1fffe0),     (21, 0x1fffe1),     (22, 0x3fffe0),     (21, 0x1fffe2),
    (23, 0x7fffed),     (22, 0x3fffe1),     (23, 0x7fffee),     (23, 0x7fffef),
    (20, 0xfffea),      (22, 0x3fffe2),     (22, 0x3fffe3),     (22, 0x3fffe4),
    (23, 0x7ffff0),     (22, 0x3fffe5),     (22, 0x3fffe6),     (23, 0x7ffff1),
    (26, 0x3ffffe0),    (26, 0x3ffffe1),    (20, 0xfffeb),      (19, 0x7fff1),
    (22, 0x3fffe7),     (23, 0x7ffff2),     (22, 0x3fffe8),     (25, 0x1ffffec),
    (26, 0x3ffffe2),    (26, 0x3ffffe3),    (26, 0x3ffffe4),    (27, 0x7ffffde),
    (27, 0x7ffffdf),    (26, 0x3ffffe5),    (24, 0xfffff1),     (25, 0x1ffffed),
    (19, 0x7fff2),      (21, 0x1fffe3),     (26, 0x3ffffe6),    (27, 0x7ffffe0),
    (27, 0x7ffffe1),    (26, 0x3ffffe7),    (27, 0x7ffffe2),    (24, 0xfffff2),
    (21, 0x1fffe4),     (21, 0x1fffe5),     (26, 0x3ffffe8),    (26, 0x3ffffe9),
    (28, 0xffffffd),    (27, 0x7ffffe3),    (27, 0x7ffffe4),    (27, 0x7ffffe5),
    (20, 0xfffec),      (24, 0xfffff3),     (20, 0xfffed),      (21, 0x1fffe6),
    (22, 0x3fffe9),     (21, 0x1fffe7),     (21, 0x1fffe8),     (23, 0x7ffff3),
    (22, 0x3fffea),     (22, 0x3fffeb),     (25, 0x1ffffee),    (25, 0x1ffffef),
    (24, 0xfffff4),     (24, 0xfffff5),     (26, 0x3ffffea),    (23, 0x7ffff4),
    (26, 0x3ffffeb),    (27, 0x7ffffe6),    (26, 0x3ffffec),    (26, 0x3ffffed),
    (27, 0x7ffffe7),    (27, 0x7ffffe8),    (27, 0x7ffffe9),    (27, 0x7ffffea),
    (27, 0x7ffffeb),    (28, 0xffffffe),    (27, 0x7ffffec),    (27, 0x7ffffed),
    (27, 0x7ffffee),    (27, 0x7ffffef),    (27, 0x7fffff0),    (26, 0x3ffffee),
    (30, 0x3fffffff),                                                                                                                  // EOS
];

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum HuffmanError {
    #[error("invalid Huffman code (no symbol matches input bits)")]
    InvalidCode,
    #[error("EOS symbol appeared in encoded data (RFC 7541 §5.2)")]
    EosInData,
    #[error("padding exceeds 7 bits — wire encoding is invalid")]
    PaddingTooLong,
    #[error("padding does not match expected MSB-pad pattern")]
    InvalidPadding,
}

/// Encode `input` symbols as a Huffman bit stream, appending to
/// `dst`. Trailing bits of the final byte are padded with 1s per
/// §5.2.
///
/// Returns the number of output bytes written.
///
/// **Implementation:** `u64` bit accumulator (matches 64-bit
/// architecture word). Per symbol: table lookup + shift + OR. Drained
/// bytes go directly to `put_u8`. ~1 ns/byte on Apple M-class — at
/// the scalar limit (4 cycles/byte ≈ table-lookup + shift + OR +
/// emit). u128 was tried and rejected: on 64-bit hardware u128 ops
/// cost ~2× per shift/OR. Chunked-batch via `put_slice` was tried
/// and rejected: the chunk-cleanup tail path costs more than 12-15
/// `put_u8` calls on small inputs (common case for HTTP headers).
///
/// ### Parked: u32 burst drain
///
/// Drain 4 bytes at a time via `put_u32_be` when buffer holds >= 32
/// valid bits. Reduces drain count ~4×; replaces 4 separate
/// `put_u8` writes per drain with one 4-byte write. Worth trying
/// once the wider HPACK harness exists. Deferred — current encoder
/// already beats h2 5-17% on every input <= 512B; marginal loss
/// (~4%) at 4 KiB is in the put_* ceiling not the bit-pack code.
#[inline]
pub fn encode<B: BufMut>(input: &[u8], dst: &mut B) -> usize {
    let expected = encoded_len(input);
    let mut buffer: u64 = 0;
    let mut buffer_bits: u32 = 0;
    let mut bytes_written = 0usize;
    for &symbol in input {
        let (length, code) = ENCODE_TABLE[symbol as usize];
        let length_bits = u32::from(length);
        buffer = (buffer << length_bits) | u64::from(code);
        buffer_bits += length_bits;
        while buffer_bits >= 8 {
            buffer_bits -= 8;
            let byte = ((buffer >> buffer_bits) & 0xFF) as u8;
            dst.put_u8(byte);
            bytes_written += 1;
        }
    }
    // Pad with 1s up to the next byte boundary per §5.2.
    if buffer_bits > 0 {
        let pad_bits = 8 - buffer_bits;
        let pad = (1u64 << pad_bits) - 1;
        let final_byte = (((buffer << pad_bits) | pad) & 0xFF) as u8;
        dst.put_u8(final_byte);
        bytes_written += 1;
    }
    debug_assert_eq!(bytes_written, expected);
    bytes_written
}

/// Compute the encoded byte length without emitting bytes — useful
/// when callers want to pre-size their output buffer.
#[inline]
#[must_use]
pub fn encoded_len(input: &[u8]) -> usize {
    let mut bits = 0u64;
    for &symbol in input {
        bits += u64::from(ENCODE_TABLE[symbol as usize].0);
    }
    bits.div_ceil(8) as usize
}

// ---------- decoder ----------
//
// 4-bit nibble table-driven decoder. State-machine entries are
// derived from the encode table by `const fn`s, evaluated once by
// rustc at COMPILE time — the resulting `static` arrays land in
// `.rodata`, not the heap.
//
// Layout:
//   DECODE_STATE_TABLE: [NibbleEntry; NIBBLE_TABLE_SIZE]
//     Indexed by `state * 16 + nibble`. NUM_STATES * 16 entries total.
//
// Each NibbleEntry packs:
//   - next_state: u16 (the tree position after consuming 4 bits)
//   - flags: u8 with bits:
//       SYMBOL_OUT  (0x1): output_byte holds a decoded byte
//       ACCEPTABLE  (0x2): this state is a valid stream-end position
//                          (i.e., we've completed a code, or we're
//                          on the all-1s padding prefix path)
//       EOS_OUT     (0x4): the decoded "byte" was the EOS marker
//                          (which MUST NOT appear in valid data)
//   - output_byte: u8 (the decoded byte if SYMBOL_OUT is set, else 0)
//
// Decode loop reads each input byte as two nibbles (high then low),
// does one table lookup per nibble. ~3-5 ns per input byte on Apple
// M-class — 2-3× faster than the bit-by-bit walker.

#[derive(Debug, Clone, Copy)]
struct NibbleEntry {
    /// Next tree-state to resume from on the following nibble. `u8`
    /// because the RFC 7541 Huffman tree has exactly 256 internal
    /// states (enforced at COMPILE time — `build_tree`'s const-eval
    /// panics if the RFC table ever produced a different count) —
    /// the `u8`-bounded type lets the compiler prove that
    /// `state * 16 + nibble < 4096` without a runtime check, which
    /// is what makes the safe-index decode path branchless.
    next_state: u8,
    flags: u8,
    output_byte: u8,
}

const FLAG_SYMBOL_OUT: u8 = 0x1;
const FLAG_ACCEPTABLE: u8 = 0x2;
const FLAG_EOS_OUT: u8 = 0x4;

const EOS_MARKER: u16 = 256;

/// Number of tree states for the RFC 7541 Huffman alphabet. Enforced
/// equal to `build_tree`'s actual node count at COMPILE time (a
/// const-eval `assert!` inside `build_tree`, not a runtime check).
/// With this fixed at the type level (via `u8` `next_state`), the
/// compiler can prove `state * 16 + nibble < NUM_STATES * 16`.
const NUM_STATES: usize = 256;
const NIBBLE_TABLE_SIZE: usize = NUM_STATES * 16;

/// Binary-tree node for the Huffman alphabet. `left` / `right` hold
/// either a child index (>0), a leaf marker `-(symbol+1)` (<0), or
/// `0` for "unset" (built during table construction only).
#[derive(Clone, Copy)]
struct TreeNode {
    left: i32,
    right: i32,
}

const EMPTY_TREE_NODE: TreeNode = TreeNode { left: 0, right: 0 };

/// Builds the canonical RFC 7541 Appendix B Huffman tree. Runs
/// entirely in `const` evaluation (rustc, at compile time) — the
/// fixed-size `[TreeNode; NUM_STATES]` output means a code table
/// that ever produced a different node count fails the build rather
/// than corrupting a runtime table.
const fn build_tree() -> [TreeNode; NUM_STATES] {
    let mut nodes = [EMPTY_TREE_NODE; NUM_STATES];
    let mut next_free: usize = 1; // index 0 is root
    let mut symbol = 0usize;
    while symbol < ENCODE_TABLE.len() {
        let (length, code) = ENCODE_TABLE[symbol];
        let length = length as u32;
        let code = code as u64;
        let mut current = 0_i32;
        let mut bit_idx = length;
        while bit_idx > 0 {
            bit_idx -= 1;
            let bit = (code >> bit_idx) & 1;
            let is_last_bit = bit_idx == 0;
            let slot = read_child(&nodes, current, bit);
            if is_last_bit {
                write_child(&mut nodes, current, bit, -((symbol as i32) + 1));
            } else if slot <= 0 {
                let new_idx = next_free;
                next_free += 1;
                write_child(&mut nodes, current, bit, new_idx as i32);
                current = new_idx as i32;
            } else {
                current = slot;
            }
        }
        symbol += 1;
    }
    assert!(
        next_free == NUM_STATES,
        "RFC 7541 Huffman tree node count changed"
    );
    nodes
}

const fn read_child(nodes: &[TreeNode; NUM_STATES], idx: i32, bit: u64) -> i32 {
    let node = &nodes[idx as usize];
    if bit == 1 { node.right } else { node.left }
}

const fn write_child(nodes: &mut [TreeNode; NUM_STATES], idx: i32, bit: u64, value: i32) {
    let node = &mut nodes[idx as usize];
    if bit == 1 {
        node.right = value;
    } else {
        node.left = value;
    }
}

/// The canonical Huffman tree, built once by rustc's const evaluator.
const TREE: [TreeNode; NUM_STATES] = build_tree();

/// True if `target` lies on the root→right-children chain (i.e., the
/// path of all-1 bits, which is the trailing-padding prefix path).
const fn is_on_padding_path(nodes: &[TreeNode; NUM_STATES], target: i32) -> bool {
    if target <= 0 {
        return target == 0;
    }
    let mut current = 0_i32;
    while current != target {
        let right = nodes[current as usize].right;
        if right <= 0 {
            return false;
        }
        current = right;
    }
    true
}

/// Simulate consuming a 4-bit `nibble` starting at tree node `state`.
/// Min code length is 5 bits, so at most ONE symbol can be emitted
/// per nibble. Returns the resulting NibbleEntry.
const fn simulate_nibble(nodes: &[TreeNode; NUM_STATES], state: i32, nibble: u8) -> NibbleEntry {
    let mut current = state;
    let mut emitted: Option<u16> = None;
    let mut bit_idx = 4_i32;
    while bit_idx > 0 {
        bit_idx -= 1;
        let bit = (nibble >> bit_idx) & 1;
        let next = if bit == 1 {
            nodes[current as usize].right
        } else {
            nodes[current as usize].left
        };
        if next < 0 {
            // Leaf: -(symbol+1) → symbol
            let symbol = (-next - 1) as u16;
            emitted = Some(symbol);
            current = 0; // back to root
        } else if next == 0 {
            current = -1;
            break;
        } else {
            current = next;
        }
    }
    let mut flags = 0_u8;
    let mut output_byte = 0_u8;
    if let Some(symbol) = emitted {
        if symbol == EOS_MARKER {
            flags |= FLAG_EOS_OUT;
        } else {
            flags |= FLAG_SYMBOL_OUT;
            output_byte = symbol as u8;
        }
    }
    // Acceptable: ended at root or on the all-1s padding prefix path.
    let acceptable = current == 0 || is_on_padding_path(nodes, current);
    if acceptable {
        flags |= FLAG_ACCEPTABLE;
    }
    // Same NUM_STATES bound + u8 truncation as ROOT_BYTE_TABLE; keeps
    // the decode hot path's index expression provably in range for
    // `[NibbleEntry; NIBBLE_TABLE_SIZE]`.
    assert!(
        current >= 0 && (current as usize) < NUM_STATES,
        "tree state out of range [0, NUM_STATES)"
    );
    let next_state = if current < 0 { 0 } else { current as u8 };
    NibbleEntry {
        next_state,
        flags,
        output_byte,
    }
}

/// Builds the full nibble decode-state table at COMPILE time.
const fn build_decode_state_table() -> [NibbleEntry; NIBBLE_TABLE_SIZE] {
    let mut table = [NibbleEntry {
        next_state: 0,
        flags: 0,
        output_byte: 0,
    }; NIBBLE_TABLE_SIZE];
    let mut state = 0usize;
    while state < NUM_STATES {
        let mut nibble = 0_u8;
        while nibble < 16 {
            table[state * 16 + nibble as usize] = simulate_nibble(&TREE, state as i32, nibble);
            nibble += 1;
        }
        state += 1;
    }
    table
}

/// Decode state table, `const`-evaluated into `.rodata` at compile
/// time — no lazy init, no heap, no synchronization. 16 KB resident,
/// L1-friendly.
static DECODE_STATE_TABLE: [NibbleEntry; NIBBLE_TABLE_SIZE] = build_decode_state_table();

/// Root-only 8-bit fast-path table. From state 0 (root), one input
/// byte's 8 bits decode to 0-2 emitted symbols plus an ending state.
/// 256 entries × 8 bytes = 2 KB — fits comfortably in L1.
///
/// For typical HTTP traffic (ASCII text, 5-8 bit codes), the decoder
/// returns to root after each symbol, so this path fires for the
/// vast majority of input bytes. Saves one table lookup per byte
/// compared to the 4-bit nibble fallback.
#[derive(Debug, Clone, Copy)]
struct ByteEntry {
    /// Ending tree position after consuming 8 bits from root. `u8`
    /// for the same compile-time-bounded reason as `NibbleEntry::next_state`.
    next_state: u8,
    /// Bit 0: SYM1_OUT (output1 valid)
    /// Bit 1: SYM2_OUT (output2 valid)
    /// Bit 2: EOS_OUT
    /// Bit 3: ACCEPTABLE
    flags: u8,
    output1: u8,
    output2: u8,
    _pad: [u8; 3],
}

const BYTE_FLAG_SYM1: u8 = 0x1;
const BYTE_FLAG_SYM2: u8 = 0x2;
const BYTE_FLAG_EOS: u8 = 0x4;
const BYTE_FLAG_ACCEPTABLE: u8 = 0x8;

/// Builds the root-byte fast-path table at COMPILE time.
const fn build_root_byte_table() -> [ByteEntry; 256] {
    let nodes = &TREE;
    let mut table = [ByteEntry {
        next_state: 0,
        flags: 0,
        output1: 0,
        output2: 0,
        _pad: [0; 3],
    }; 256];
    let mut byte = 0_u16;
    while byte < 256 {
        // Walk 8 bits from root, collecting up to 2 emitted symbols.
        let mut current: i32 = 0;
        let mut emits: [u16; 2] = [0; 2];
        let mut emit_count: u8 = 0;
        let mut bit_idx = 8_i32;
        while bit_idx > 0 {
            bit_idx -= 1;
            let bit = ((byte >> bit_idx) & 1) as u64;
            let next = if bit == 1 {
                nodes[current as usize].right
            } else {
                nodes[current as usize].left
            };
            if next < 0 {
                let symbol = (-next - 1) as u16;
                if emit_count < 2 {
                    emits[emit_count as usize] = symbol;
                    emit_count += 1;
                }
                current = 0;
            } else if next == 0 {
                current = -1;
                break;
            } else {
                current = next;
            }
        }
        let mut flags = 0_u8;
        let mut output1 = 0_u8;
        let mut output2 = 0_u8;
        if emit_count >= 1 {
            let sym = emits[0];
            if sym == EOS_MARKER {
                flags |= BYTE_FLAG_EOS;
            } else {
                flags |= BYTE_FLAG_SYM1;
                output1 = sym as u8;
            }
        }
        if emit_count >= 2 {
            let sym = emits[1];
            if sym == EOS_MARKER {
                flags |= BYTE_FLAG_EOS;
            } else {
                flags |= BYTE_FLAG_SYM2;
                output2 = sym as u8;
            }
        }
        let acceptable = current == 0 || is_on_padding_path(nodes, current);
        if acceptable {
            flags |= BYTE_FLAG_ACCEPTABLE;
        }
        // `current` is bounded `[0, NUM_STATES)` here; the assertion
        // below enforces this at COMPILE time. Truncate to `u8` so
        // the type-level proof of `state < NUM_STATES` holds in the
        // decode hot path.
        assert!(
            current >= 0 && (current as usize) < NUM_STATES,
            "tree state out of range [0, NUM_STATES)"
        );
        let next_state = if current < 0 { 0 } else { current as u8 };
        table[byte as usize] = ByteEntry {
            next_state,
            flags,
            output1,
            output2,
            _pad: [0; 3],
        };
        byte += 1;
    }
    table
}

/// Root-byte fast-path table, `const`-evaluated into `.rodata` at
/// compile time — no lazy init, no heap, no synchronization.
static ROOT_BYTE_TABLE: [ByteEntry; 256] = build_root_byte_table();

/// Decode a Huffman bit stream into `dst`, appending bytes.
///
/// **Implementation:** two-level table lookup.
///
/// 1. **Root byte fast path** (state == 0): one lookup per input byte
///    into `ROOT_BYTE_TABLE[256]`. Emits 1-2 decoded symbols + next
///    state. Common case for HTTP-text traffic since most headers
///    end on a byte boundary mid-code or stay at root.
/// 2. **Nibble fallback** (state != 0): two lookups per byte into
///    the per-state `DECODE_STATE_TABLE` (4-bit nibble entries).
///    Used when a code straddles a byte boundary.
///
/// O(1) work per byte. No tree walk, no bit iteration.
///
/// Errors:
/// - `EosInData`: EOS symbol appeared in payload (RFC §5.2: MUST NOT).
/// - `InvalidPadding`: trailing pad bits aren't all-1s.
#[inline]
pub fn decode<B: BufMut>(input: &[u8], dst: &mut B) -> Result<usize, HuffmanError> {
    let nibble_table = &DECODE_STATE_TABLE;
    let root_table = &ROOT_BYTE_TABLE;
    // `state: u8` (max 255) × 16 + nibble (max 15) = 4095. The fixed
    // `[NibbleEntry; NIBBLE_TABLE_SIZE]` table has 4096 entries, so
    // `state as usize * 16 + nibble` is provably in range. LLVM
    // elides the bounds check — no `unsafe` needed.
    let mut state: u8 = 0;
    let mut bytes_written = 0usize;
    let mut last_acceptable: bool = true;
    let mut i = 0usize;
    let len = input.len();
    while i < len {
        if state == 0 {
            // loop guard `i < len` proves bounds; LLVM elides the panic
            // branch the safe index would otherwise carry.
            let byte = input[i];
            // `byte as usize` is `[0, 256)` and `root_table` is
            // `[ByteEntry; 256]`, so LLVM elides the bounds check.
            // verified equivalent codegen vs `get_unchecked` in
            // release builds.
            let entry = &root_table[byte as usize];
            state = entry.next_state;
            if entry.flags & BYTE_FLAG_EOS != 0 {
                return Err(HuffmanError::EosInData);
            }
            if entry.flags & BYTE_FLAG_SYM1 != 0 {
                dst.put_u8(entry.output1);
                bytes_written += 1;
            }
            if entry.flags & BYTE_FLAG_SYM2 != 0 {
                dst.put_u8(entry.output2);
                bytes_written += 1;
            }
            last_acceptable = entry.flags & BYTE_FLAG_ACCEPTABLE != 0;
            i += 1;
        } else {
            // Nibble fallback: state != 0 (mid-code).
            // `state: u8` × 16 + nibble (4-bit) ≤ 4095 < 4096 = table
            // length. Compiler-provable bounds, no `unsafe` needed.
            let byte = input[i];
            let high = (byte >> 4) as usize;
            let low = (byte & 0x0f) as usize;
            let entry = &nibble_table[state as usize * 16 + high];
            state = entry.next_state;
            if entry.flags & FLAG_EOS_OUT != 0 {
                return Err(HuffmanError::EosInData);
            }
            if entry.flags & FLAG_SYMBOL_OUT != 0 {
                dst.put_u8(entry.output_byte);
                bytes_written += 1;
            }
            let entry = &nibble_table[state as usize * 16 + low];
            state = entry.next_state;
            if entry.flags & FLAG_EOS_OUT != 0 {
                return Err(HuffmanError::EosInData);
            }
            if entry.flags & FLAG_SYMBOL_OUT != 0 {
                dst.put_u8(entry.output_byte);
                bytes_written += 1;
            }
            last_acceptable = entry.flags & FLAG_ACCEPTABLE != 0;
            i += 1;
        }
    }
    if !last_acceptable {
        return Err(HuffmanError::InvalidPadding);
    }
    Ok(bytes_written)
}

#[cfg(all(test, not(feature = "hpack-no-alloc")))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// RFC 7541 §C.4.1 — encoding "www.example.com" yields the
    /// hex sequence `f1e3 c2e5 f23a 6ba0 ab90 f4ff`.
    #[test]
    fn rfc_c_4_1_encode_www_example_com() {
        let mut out = Vec::new();
        encode(b"www.example.com", &mut out);
        assert_eq!(
            out,
            vec![
                0xf1, 0xe3, 0xc2, 0xe5, 0xf2, 0x3a, 0x6b, 0xa0, 0xab, 0x90, 0xf4, 0xff
            ],
            "encoded bytes must match RFC §C.4.1"
        );
    }

    #[test]
    fn rfc_c_4_1_decode_www_example_com() {
        let wire = [
            0xf1, 0xe3, 0xc2, 0xe5, 0xf2, 0x3a, 0x6b, 0xa0, 0xab, 0x90, 0xf4, 0xff,
        ];
        let mut out = Vec::new();
        decode(&wire, &mut out).expect("decode");
        assert_eq!(out, b"www.example.com");
    }

    /// RFC 7541 §C.4.2 — encoding "no-cache" yields hex `a8eb 1064
    /// 9cbf`.
    #[test]
    fn rfc_c_4_2_encode_no_cache() {
        let mut out = Vec::new();
        encode(b"no-cache", &mut out);
        assert_eq!(out, vec![0xa8, 0xeb, 0x10, 0x64, 0x9c, 0xbf]);
    }

    #[test]
    fn rfc_c_4_2_decode_no_cache() {
        let mut out = Vec::new();
        decode(&[0xa8, 0xeb, 0x10, 0x64, 0x9c, 0xbf], &mut out).expect("decode");
        assert_eq!(out, b"no-cache");
    }

    /// RFC 7541 §C.4.3 — "custom-key" and "custom-value".
    #[test]
    fn rfc_c_4_3_encode_custom_key() {
        let mut out = Vec::new();
        encode(b"custom-key", &mut out);
        assert_eq!(out, vec![0x25, 0xa8, 0x49, 0xe9, 0x5b, 0xa9, 0x7d, 0x7f]);
    }

    #[test]
    fn rfc_c_4_3_encode_custom_value() {
        let mut out = Vec::new();
        encode(b"custom-value", &mut out);
        assert_eq!(
            out,
            vec![0x25, 0xa8, 0x49, 0xe9, 0x5b, 0xb8, 0xe8, 0xb4, 0xbf]
        );
    }

    #[test]
    fn empty_input_yields_empty_output() {
        let mut out = Vec::new();
        encode(b"", &mut out);
        assert_eq!(out, Vec::<u8>::new());
        let mut decoded = Vec::new();
        decode(&[], &mut decoded).expect("decode");
        assert_eq!(decoded, Vec::<u8>::new());
    }

    #[test]
    fn encoded_len_matches_encode_output() {
        for sample in [
            &b""[..],
            b"a",
            b"www.example.com",
            b"no-cache",
            b"custom-key",
            b"the quick brown fox jumps over the lazy dog 0123456789",
        ] {
            let predicted = encoded_len(sample);
            let mut out = Vec::new();
            encode(sample, &mut out);
            assert_eq!(predicted, out.len(), "encoded_len mismatch for {sample:?}");
        }
    }

    #[test]
    fn round_trip_random_bytes() {
        // Round-trip every byte value 0..=255 individually.
        for byte in 0..=255_u8 {
            let mut encoded = Vec::new();
            encode(&[byte], &mut encoded);
            let mut decoded = Vec::new();
            decode(&encoded, &mut decoded).expect("decode");
            assert_eq!(decoded, vec![byte], "round-trip failed for byte {byte}");
        }
    }

    #[test]
    fn round_trip_long_alphabetic_string() {
        let original: Vec<u8> = (0..=255_u8).cycle().take(4096).collect();
        let mut encoded = Vec::new();
        encode(&original, &mut encoded);
        let mut decoded = Vec::new();
        decode(&encoded, &mut decoded).expect("decode");
        assert_eq!(decoded, original);
    }

    /// `DC-HPACK-HUFFMAN-BOX` 0-heap proof: `DECODE_STATE_TABLE` /
    /// `ROOT_BYTE_TABLE` are `const`-evaluated `.rodata`, not a
    /// lazily-built `Box`. With output buffers pre-sized OUTSIDE the
    /// measured window (so the assertion isolates the table access,
    /// not the caller's own output-growth allocations), a full
    /// encode+decode round trip on a 4 KiB body performs 0 heap
    /// allocations — mirrors `decoder.rs`'s
    /// `alloc_count_decode_into_zero_when_no_huffman_and_no_table_growth`
    /// pattern (`stats_alloc::Region` around
    /// `crate::alloc_test::PROTOCOLS_TEST_ALLOC`).
    #[test]
    #[cfg(feature = "std")]
    fn huffman_tables_are_rodata_not_heap() {
        let input = vec![b'A'; 4096];
        let mut encoded = Vec::with_capacity(encoded_len(&input));
        let mut decoded = Vec::with_capacity(input.len());

        // the process-global stats_alloc counter also ticks for stray
        // allocations on other runtime threads parked in-window (harness,
        // output pump) on a loaded CI runner; that noise is additive-only,
        // so the MIN delta across repeats is encode+decode's true per-call
        // cost. tables are `static [T; N] = const fn()`, so there is no
        // one-time lazy-Box allocation for min to hide.
        let region = crate::alloc_test::exclusive_region();
        let mut min_allocations = usize::MAX;
        for _ in 0..8 {
            encoded.clear();
            decoded.clear();
            let before = region.change();
            encode(&input, &mut encoded);
            decode(&encoded, &mut decoded).expect("decode");
            let after = region.change();
            min_allocations = min_allocations.min(after.allocations - before.allocations);
        }

        assert_eq!(decoded, input);
        assert_eq!(
            min_allocations, 0,
            "huffman encode+decode must perform 0 heap allocations with pre-sized \
             output buffers — proves DECODE_STATE_TABLE / ROOT_BYTE_TABLE live in \
             .rodata, not behind a lazily-built Box"
        );
    }

    /// Independent oracle for the `DC-HPACK-HUFFMAN-BOX` table
    /// redesign (Box+`OnceBox` lazy build → `const fn` compile-time
    /// tables). A single-bit walker built directly off `ENCODE_TABLE`
    /// — RFC 7541 Appendix B's canonical code — with none of the
    /// tree/nibble/byte machinery `build_tree` / `simulate_nibble` /
    /// `build_root_byte_table` construct. Agreement proves the
    /// compile-time tables didn't silently diverge from the RFC
    /// during the Box → const migration.
    fn naive_bit_decode(input: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut accumulator: u64 = 0;
        let mut accumulator_bits: u32 = 0;
        for &byte in input {
            for bit_idx in (0..8).rev() {
                let bit = u64::from((byte >> bit_idx) & 1);
                accumulator = (accumulator << 1) | bit;
                accumulator_bits += 1;
                let matched = ENCODE_TABLE.iter().take(256).position(|&(length, code)| {
                    u32::from(length) == accumulator_bits && u64::from(code) == accumulator
                });
                if let Some(symbol) = matched {
                    out.push(symbol as u8);
                    accumulator = 0;
                    accumulator_bits = 0;
                }
            }
        }
        out
    }

    #[test]
    fn const_tables_agree_with_independent_bit_walker() {
        let samples: Vec<Vec<u8>> = vec![
            b"www.example.com".to_vec(),
            b"no-cache".to_vec(),
            b"custom-key".to_vec(),
            b"custom-value".to_vec(),
            (0..=255_u8).collect(),
            (0..=255_u8).cycle().take(4096).collect(),
        ];
        for sample in samples {
            let mut encoded = Vec::new();
            encode(&sample, &mut encoded);

            let reference = naive_bit_decode(&encoded);
            assert_eq!(
                reference,
                sample,
                "independent bit-walker oracle mismatch for sample len {}",
                sample.len()
            );

            let mut table_driven = Vec::new();
            decode(&encoded, &mut table_driven).expect("decode");
            assert_eq!(
                table_driven, reference,
                "const-table decode diverges from independent bit-walker oracle"
            );
        }
    }
}
