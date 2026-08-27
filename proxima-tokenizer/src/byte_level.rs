//! The GPT-2 byte-level alphabet (`bytes_to_unicode` in the original GPT-2
//! `encoder.py`, reused verbatim by every llama-family BPE vocab this crate
//! targets — confirmed against `tokenizer.ggml.model = "gpt2"` on the real
//! fixture). Every raw byte maps to exactly one `char`, so any `&[u8]` has a
//! lossless display-domain rendering; the vocab's own token strings
//! (`tokenizer.ggml.tokens`) are spelled in this same domain, which is why
//! merging happens on display chars rather than raw bytes.
//!
//! Printable ASCII/Latin-1 (`!`..=`~`, `¡`..=`¬`, `®`..=`ÿ`) maps to itself;
//! every other byte (control chars, space, DEL, ...) maps to a private
//! codepoint starting at `256`, assigned in byte order.

/// `byte_to_char[byte as usize]` is that byte's display-domain codepoint.
#[must_use]
pub const fn byte_to_char_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut assigned = [false; 256];

    let mut byte = 33u32;
    while byte <= 126 {
        table[byte as usize] = byte;
        assigned[byte as usize] = true;
        byte += 1;
    }
    let mut byte = 161u32;
    while byte <= 172 {
        table[byte as usize] = byte;
        assigned[byte as usize] = true;
        byte += 1;
    }
    let mut byte = 174u32;
    while byte <= 255 {
        table[byte as usize] = byte;
        assigned[byte as usize] = true;
        byte += 1;
    }

    let mut next_private = 256u32;
    let mut byte = 0usize;
    while byte < 256 {
        if !assigned[byte] {
            table[byte] = next_private;
            next_private += 1;
        }
        byte += 1;
    }
    table
}

/// Build-time table: `BYTE_TO_CHAR[b]` is byte `b`'s display codepoint.
pub const BYTE_TO_CHAR: [u32; 256] = byte_to_char_table();

/// Inverse of [`BYTE_TO_CHAR`] as a codepoint lookup: index by
/// `codepoint - 33` (the lowest codepoint ever assigned), `None` for gaps.
/// Highest assigned codepoint is `256 + (256 - 188) - 1 = 323`, so a table
/// spanning `33..=323` (291 entries) covers every possible display char.
const CODEPOINT_TABLE_BASE: u32 = 33;
const CODEPOINT_TABLE_LEN: usize = 291;

#[must_use]
const fn char_to_byte_table() -> [Option<u8>; CODEPOINT_TABLE_LEN] {
    let mut table = [None; CODEPOINT_TABLE_LEN];
    let byte_to_char = byte_to_char_table();
    let mut byte = 0usize;
    while byte < 256 {
        let codepoint = byte_to_char[byte];
        let index = (codepoint - CODEPOINT_TABLE_BASE) as usize;
        table[index] = Some(byte as u8);
        byte += 1;
    }
    table
}

const CHAR_TO_BYTE: [Option<u8>; CODEPOINT_TABLE_LEN] = char_to_byte_table();

/// Maps a raw byte to its display-domain `char`.
#[must_use]
pub fn byte_to_char(byte: u8) -> char {
    // SAFETY-free: every entry in `BYTE_TO_CHAR` is a valid scalar value --
    // the table only ever holds ASCII/Latin-1 codepoints or the private
    // range immediately above them, both comfortably below the surrogate
    // range `char::from_u32` would reject.
    char::from_u32(BYTE_TO_CHAR[byte as usize]).unwrap_or('\u{FFFD}')
}

/// Maps a display-domain `char` back to its raw byte, if it is one this
/// alphabet produces.
#[must_use]
pub fn char_to_byte(character: char) -> Option<u8> {
    let codepoint = character as u32;
    if codepoint < CODEPOINT_TABLE_BASE {
        return None;
    }
    let index = (codepoint - CODEPOINT_TABLE_BASE) as usize;
    CHAR_TO_BYTE.get(index).copied().flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_byte_round_trips_through_the_display_alphabet() {
        for byte in 0..=255u8 {
            let character = byte_to_char(byte);
            let recovered = char_to_byte(character);
            assert_eq!(recovered, Some(byte), "byte {byte} did not round-trip");
        }
    }

    #[test]
    fn printable_ascii_maps_to_itself() {
        assert_eq!(byte_to_char(b'!'), '!');
        assert_eq!(byte_to_char(b'~'), '~');
        assert_eq!(byte_to_char(b'A'), 'A');
    }

    #[test]
    fn space_maps_to_the_known_gpt2_marker() {
        // 0x20 (space) is not in the printable ranges, so it lands in the
        // private range. GPT-2/llama.cpp's well-known marker for it is
        // U+0120 ('Ġ') -- pin the exact codepoint, not just round-trip.
        assert_eq!(byte_to_char(b' '), '\u{0120}');
    }

    #[test]
    fn newline_maps_to_the_known_gpt2_marker() {
        assert_eq!(byte_to_char(b'\n'), '\u{010A}');
    }

    #[test]
    fn char_to_byte_rejects_a_codepoint_outside_the_alphabet() {
        assert_eq!(char_to_byte('\u{4E2D}'), None); // a CJK char, never a display char
    }
}
