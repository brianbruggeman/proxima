//! The LLAMA3/GPT-2-family pretokenizer: splits raw text into pretoken
//! spans *before* byte-level BPE runs on each one independently. Mirrors
//! `LLAMA_VOCAB_PRE_TYPE_LLAMA3`'s regex
//! (`llama.cpp/src/llama-vocab.cpp:282-291`, confirmed as the variant this
//! crate's fixture uses via `tokenizer.ggml.pre = "llama-bpe"`):
//!
//! ```text
//! (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
//!   | [^\r\n\p{L}\p{N}]?\p{L}+
//!   | \p{N}{1,3}
//!   | ' '?[^\s\p{L}\p{N}]+[\r\n]*
//!   | \s*[\r\n]+
//!   | \s+(?!\S)
//!   | \s+
//! ```
//!
//! No regex engine ships in this no_std+alloc crate, so this is a
//! hand-rolled scanner implementing the same alternation order using
//! `char::is_alphabetic`/`is_numeric`/`is_whitespace` (Unicode-aware, and
//! available in `core` -- no allocation, no lookup table of our own)
//! in place of `\p{L}`/`\p{N}`/`\s`. The two differ from PCRE's exact
//! Unicode tables at the margins (grapheme-cluster-aware scripts, some
//! symbol/format codepoints) but agree on every ASCII and common-script
//! case exercised by the round-trip tests. Round-trip correctness
//! (`decode(encode(x)) == x`) never depends on where these boundaries
//! land -- only how closely a produced token id sequence matches another
//! implementation's (e.g. llama.cpp's) does.

use alloc::vec::Vec;

fn is_letter(character: char) -> bool {
    character.is_alphabetic()
}

fn is_digit(character: char) -> bool {
    character.is_numeric()
}

fn is_punct(character: char) -> bool {
    !character.is_whitespace() && !character.is_alphabetic() && !character.is_numeric()
}

fn contraction_len(chars: &[char]) -> Option<usize> {
    if chars.first().copied() != Some('\'') {
        return None;
    }
    let lower = |index: usize| chars.get(index).map(|character| character.to_ascii_lowercase());
    match (lower(1), lower(2)) {
        (Some('r'), Some('e')) | (Some('v'), Some('e')) | (Some('l'), Some('l')) => Some(3),
        (Some('s'), _) | (Some('t'), _) | (Some('m'), _) | (Some('d'), _) => Some(2),
        _ => None,
    }
}

/// Splits `text` into pretoken spans, returned as byte-offset ranges into
/// `text` so callers can slice the original string (and, for encode, its
/// UTF-8 bytes) without an extra allocation per pretoken.
#[must_use]
pub fn pretokenize(text: &str) -> Vec<core::ops::Range<usize>> {
    let chars: Vec<char> = text.chars().collect();
    let byte_offsets: Vec<usize> = char_byte_offsets(text, chars.len());

    let mut spans = Vec::new();
    let mut index = 0usize;
    while index < chars.len() {
        let consumed = match_at(&chars, index);
        let end = index + consumed.max(1);
        spans.push(byte_offsets[index]..byte_offsets[end]);
        index = end;
    }
    spans
}

/// Byte offset of each char boundary in `text`, plus one trailing entry
/// for the end of the string (`char_count + 1` total entries).
fn char_byte_offsets(text: &str, char_count: usize) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(char_count + 1);
    offsets.extend(text.char_indices().map(|(offset, _)| offset));
    offsets.push(text.len());
    offsets
}

/// How many chars, starting at `index`, the next pretoken consumes.
/// Returns `0` only when nothing at `index` matches any rule, in which
/// case the caller must still advance by one char (a single
/// non-matching char becomes its own one-char pretoken -- this only
/// happens for codepoints the six alternatives all reject, which does
/// not occur for well-formed Unicode text but keeps the scanner total).
fn match_at(chars: &[char], index: usize) -> usize {
    let remaining = &chars[index..];

    if let Some(length) = contraction_len(remaining) {
        return length;
    }
    if let Some(length) = match_letters(remaining) {
        return length;
    }
    if let Some(length) = match_digits(remaining) {
        return length;
    }
    if let Some(length) = match_punct(remaining) {
        return length;
    }
    if let Some(length) = match_whitespace_with_newline(remaining) {
        return length;
    }
    if let Some(length) = match_trailing_whitespace(remaining) {
        return length;
    }
    0
}

/// `[^\r\n\p{L}\p{N}]?\p{L}+`
fn match_letters(chars: &[char]) -> Option<usize> {
    let first = *chars.first()?;
    if is_letter(first) {
        let run = chars.iter().take_while(|character| is_letter(**character)).count();
        return Some(run);
    }
    if first != '\r'
        && first != '\n'
        && !is_digit(first)
        && let Some(&second) = chars.get(1)
        && is_letter(second)
    {
        let run = chars[1..].iter().take_while(|character| is_letter(**character)).count();
        return Some(1 + run);
    }
    None
}

/// `\p{N}{1,3}`
fn match_digits(chars: &[char]) -> Option<usize> {
    let first = *chars.first()?;
    if !is_digit(first) {
        return None;
    }
    let run = chars.iter().take_while(|character| is_digit(**character)).count();
    Some(run.min(3))
}

/// `' '?[^\s\p{L}\p{N}]+[\r\n]*`
fn match_punct(chars: &[char]) -> Option<usize> {
    let first = *chars.first()?;
    let lead = if is_punct(first) {
        0
    } else if first == ' ' && chars.get(1).is_some_and(|character| is_punct(*character)) {
        1
    } else {
        return None;
    };
    let punct_run = chars[lead..].iter().take_while(|character| is_punct(**character)).count();
    if punct_run == 0 {
        return None;
    }
    let after_punct = lead + punct_run;
    let newline_run = chars[after_punct..]
        .iter()
        .take_while(|character| **character == '\r' || **character == '\n')
        .count();
    Some(after_punct + newline_run)
}

/// `\s*[\r\n]+`, consuming only through the last newline in the leading
/// whitespace run (matching PCRE's greedy-then-backtrack behavior for
/// this pattern -- see the module doc's derivation).
fn match_whitespace_with_newline(chars: &[char]) -> Option<usize> {
    let first = *chars.first()?;
    if !first.is_whitespace() {
        return None;
    }
    let run = chars.iter().take_while(|character| character.is_whitespace()).count();
    let last_newline = chars[..run].iter().rposition(|character| *character == '\r' || *character == '\n')?;
    Some(last_newline + 1)
}

/// `\s+(?!\S)` falling back to `\s+` -- consumes the whole trailing
/// whitespace run if it reaches end-of-input, otherwise all but its
/// last char (left for the next pretoken to pick up via
/// [`match_letters`]/[`match_punct`]'s optional lead), and always at
/// least one char.
fn match_trailing_whitespace(chars: &[char]) -> Option<usize> {
    let first = *chars.first()?;
    if !first.is_whitespace() {
        return None;
    }
    let run = chars.iter().take_while(|character| character.is_whitespace()).count();
    if run == chars.len() {
        Some(run)
    } else if run > 1 {
        Some(run - 1)
    } else {
        Some(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spans(text: &str) -> Vec<&str> {
        pretokenize(text).into_iter().map(|range| &text[range]).collect()
    }

    #[test]
    fn splits_hello_world_with_leading_space_attached() {
        assert_eq!(spans("Hello world"), ["Hello", " world"]);
    }

    #[test]
    fn contraction_is_its_own_pretoken() {
        assert_eq!(spans("don't"), ["don", "'t"]);
    }

    #[test]
    fn digit_runs_cap_at_three() {
        assert_eq!(spans("3333"), ["333", "3"]);
    }

    #[test]
    fn multi_space_run_leaves_last_space_for_the_word() {
        assert_eq!(spans("a  b"), ["a", " ", " b"]);
    }

    #[test]
    fn trailing_whitespace_run_is_swallowed_whole() {
        assert_eq!(spans("a   "), ["a", "   "]);
    }

    #[test]
    fn newline_run_is_its_own_pretoken() {
        assert_eq!(spans("a\n\nb"), ["a", "\n\n", "b"]);
    }

    #[test]
    fn empty_input_has_no_pretokens() {
        assert!(spans("").is_empty());
    }

    #[test]
    fn punctuation_run_with_leading_space() {
        assert_eq!(spans(" Hello!"), [" Hello", "!"]);
    }

    #[test]
    fn every_span_concatenates_back_to_the_original_text() {
        let text = "Hello, y'all! How are you \u{1F601} ?\u{6211}\u{60F3}\u{5728}apple1314151\u{5929}~";
        let mut rebuilt = alloc::string::String::new();
        for span in spans(text) {
            rebuilt.push_str(span);
        }
        assert_eq!(rebuilt, text);
    }
}
