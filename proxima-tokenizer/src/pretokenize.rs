//! The GPT-2-family pretokenizers: split raw text into pretoken spans
//! *before* byte-level BPE runs on each one independently. The rule is
//! chosen by [`PreType`], which a GGUF carries as `tokenizer.ggml.pre`;
//! each variant mirrors one `LLAMA_VOCAB_PRE_TYPE_*` of llama.cpp f1ea20621
//! (`src/llama-vocab.cpp` regexes, `src/unicode.cpp` matchers):
//!
//! ```text
//! grouped-digits  (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
//!           | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}{1,3}
//!           | ' '?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
//! single-digit   same, with \p{N} (one digit per pretoken) in place of \p{N}{1,3}
//! single-digit-marks  single-digit with [\p{L}\p{M}]+ for \p{L}+ and [^\s\p{L}\p{M}\p{N}]+ for
//!         [^\s\p{L}\p{N}]+
//! ```
//!
//! No regex engine ships in this no_std+alloc crate, so this is a
//! hand-rolled scanner implementing the same alternation order. `\p{L}`,
//! `\p{N}`, `\p{M}` and `\s` are answered by `crate::unicode_tables`, the
//! tables llama.cpp's own matchers read, so the class boundaries agree with
//! it codepoint for codepoint.

use alloc::vec::Vec;

use crate::pretokenize_passes::{
    DEFAULT_PASSES, DIGIT_ISOLATED_GPT2_PASSES, FALCON_PASSES, GPT2_PASSES, Pass, run_passes,
};
use crate::unicode_tables::{CLASS_RANGES, LETTER, MARK, NUMBER, PUNCT, WHITESPACE};

/// Which pre-split rule a vocab's `tokenizer.ggml.pre` selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreType {
    /// `LLAMA_VOCAB_PRE_TYPE_LLAMA3`: digit runs of up to three. The same
    /// regex string serves dbrx, smaug-bpe and chatglm4.
    GroupedDigits,
    /// `LLAMA_VOCAB_PRE_TYPE_QWEN2`: one digit per pretoken. The same regex
    /// string serves stablelm2, hunyuan, solar-open and grok-2.
    SingleDigit,
    /// `LLAMA_VOCAB_PRE_TYPE_QWEN35`: one digit per pretoken, and `\p{M}`
    /// joins `\p{L}` in words.
    SingleDigitMarks,
    /// `LLAMA_VOCAB_PRE_TYPE_DEFAULT`: four successive splits, see
    /// `crate::pretokenize_passes`. What llama.cpp uses for `tokenizer.ggml.pre =
    /// "default"` and, with a warning, when the key is missing.
    Default,
    /// `LLAMA_VOCAB_PRE_TYPE_GPT2`: the single gpt2 word regex (gpt-2, mpt,
    /// olmo, jais, trillion, granite-docling and the other names mapped to it).
    Gpt2,
    /// `LLAMA_VOCAB_PRE_TYPE_STARCODER`: `\p{N}` isolated, then the gpt2 word
    /// regex (starcoder, refact, command-r, smollm, codeshell, exaone,
    /// minerva-7b, mellum2).
    DigitIsolatedGpt2,
    /// `LLAMA_VOCAB_PRE_TYPE_FALCON`: punctuation runs including the backtick,
    /// the gpt2 word regex, then ASCII digit triples.
    Falcon,
}

impl PreType {
    fn passes(self) -> Option<&'static [Pass]> {
        match self {
            Self::Default => Some(&DEFAULT_PASSES),
            Self::Gpt2 => Some(&GPT2_PASSES),
            Self::DigitIsolatedGpt2 => Some(&DIGIT_ISOLATED_GPT2_PASSES),
            Self::Falcon => Some(&FALCON_PASSES),
            Self::GroupedDigits | Self::SingleDigit | Self::SingleDigitMarks => None,
        }
    }

    fn digit_run_cap(self) -> usize {
        match self {
            Self::SingleDigit | Self::SingleDigitMarks => 1,
            _ => 3,
        }
    }

    fn is_word(self, character: char) -> bool {
        match self {
            Self::SingleDigitMarks => is_letter(character) || is_mark(character),
            _ => is_letter(character),
        }
    }

    fn is_punct(self, character: char) -> bool {
        !is_whitespace(character) && !is_digit(character) && !self.is_word(character)
    }
}

pub(crate) fn class_bits(character: char) -> u8 {
    let codepoint = u32::from(character);
    let index = CLASS_RANGES.partition_point(|&(start, _)| start <= codepoint);
    CLASS_RANGES[index - 1].1
}

pub(crate) fn is_letter(character: char) -> bool {
    class_bits(character) & LETTER != 0
}

pub(crate) fn is_digit(character: char) -> bool {
    class_bits(character) & NUMBER != 0
}

fn is_mark(character: char) -> bool {
    class_bits(character) & MARK != 0
}

pub(crate) fn is_punctuation(character: char) -> bool {
    class_bits(character) & PUNCT != 0
}

pub(crate) fn is_whitespace(character: char) -> bool {
    WHITESPACE.binary_search(&u32::from(character)).is_ok()
}

fn contraction_len(chars: &[char]) -> Option<usize> {
    if chars.first().copied() != Some('\'') {
        return None;
    }
    let lower = |index: usize| {
        chars
            .get(index)
            .map(|character| character.to_ascii_lowercase())
    };
    match (lower(1), lower(2)) {
        (Some('r'), Some('e')) | (Some('v'), Some('e')) | (Some('l'), Some('l')) => Some(3),
        (Some('s'), _) | (Some('t'), _) | (Some('m'), _) | (Some('d'), _) => Some(2),
        _ => None,
    }
}

/// Splits `text` into pretoken spans under `pre_type`, returned as
/// byte-offset ranges into `text` so callers can slice the original string
/// (and, for encode, its UTF-8 bytes) without an extra allocation per
/// pretoken.
#[must_use]
pub fn pretokenize(text: &str, pre_type: PreType) -> Vec<core::ops::Range<usize>> {
    let chars: Vec<char> = text.chars().collect();
    let byte_offsets: Vec<usize> = char_byte_offsets(text, chars.len());

    if let Some(passes) = pre_type.passes() {
        return run_passes(&chars, passes)
            .into_iter()
            .map(|(start, end)| byte_offsets[start]..byte_offsets[end])
            .collect();
    }

    let mut spans = Vec::new();
    let mut index = 0usize;
    while index < chars.len() {
        let consumed = match_at(&chars, index, pre_type);
        let end = index + consumed.max(1);
        spans.push(byte_offsets[index]..byte_offsets[end]);
        index = end;
    }
    spans
}

/// The char-level pre-split, `[^\n]+|[\n]+` (`LLAMA_VOCAB_PRE_TYPE_GEMMA4`,
/// `llama-vocab.cpp:528-536`): alternating runs of non-newline and newline
/// characters, as byte-offset ranges into `text`. Nothing else is split --
/// BPE merges run over the whole line, since a char-level vocab's merges are keyed on raw
/// characters and no merge may span a `\n`. Contrast [`pretokenize`], the
/// word splitter the GPT-2 byte-level path uses.
#[must_use]
pub fn pretokenize_newline_runs(text: &str) -> Vec<core::ops::Range<usize>> {
    let mut spans = Vec::new();
    let mut start = 0usize;
    let mut in_newlines = false;
    for (offset, character) in text.char_indices() {
        let is_newline = character == '\n';
        if offset > 0 && is_newline != in_newlines {
            spans.push(start..offset);
            start = offset;
        }
        in_newlines = is_newline;
    }
    if start < text.len() {
        spans.push(start..text.len());
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
/// case the caller must still advance by one char.
fn match_at(chars: &[char], index: usize, pre_type: PreType) -> usize {
    let remaining = &chars[index..];

    if let Some(length) = contraction_len(remaining) {
        return length;
    }
    if let Some(length) = match_words(remaining, pre_type) {
        return length;
    }
    if let Some(length) = match_digits(remaining, pre_type) {
        return length;
    }
    if let Some(length) = match_punct(remaining, pre_type) {
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

/// `[^\r\n\p{L}\p{N}]?\p{L}+`, or `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+` for the single-digit-marks rule
fn match_words(chars: &[char], pre_type: PreType) -> Option<usize> {
    let first = *chars.first()?;
    if pre_type.is_word(first) {
        let run = chars
            .iter()
            .take_while(|character| pre_type.is_word(**character))
            .count();
        return Some(run);
    }
    if first != '\r'
        && first != '\n'
        && !is_digit(first)
        && let Some(&second) = chars.get(1)
        && pre_type.is_word(second)
    {
        let run = chars[1..]
            .iter()
            .take_while(|character| pre_type.is_word(**character))
            .count();
        return Some(1 + run);
    }
    None
}

/// `\p{N}{1,3}` for the grouped-digits rule, `\p{N}` for the single-digit rules
fn match_digits(chars: &[char], pre_type: PreType) -> Option<usize> {
    let first = *chars.first()?;
    if !is_digit(first) {
        return None;
    }
    let run = chars
        .iter()
        .take_while(|character| is_digit(**character))
        .count();
    Some(run.min(pre_type.digit_run_cap()))
}

/// `' '?[^\s\p{L}\p{N}]+[\r\n]*` (with `\p{M}` also excluded for the single-digit-marks rule)
fn match_punct(chars: &[char], pre_type: PreType) -> Option<usize> {
    let first = *chars.first()?;
    let lead = if pre_type.is_punct(first) {
        0
    } else if first == ' '
        && chars
            .get(1)
            .is_some_and(|character| pre_type.is_punct(*character))
    {
        1
    } else {
        return None;
    };
    let punct_run = chars[lead..]
        .iter()
        .take_while(|character| pre_type.is_punct(**character))
        .count();
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
/// this pattern).
fn match_whitespace_with_newline(chars: &[char]) -> Option<usize> {
    let first = *chars.first()?;
    if !is_whitespace(first) {
        return None;
    }
    let run = chars
        .iter()
        .take_while(|character| is_whitespace(**character))
        .count();
    let last_newline = chars[..run]
        .iter()
        .rposition(|character| *character == '\r' || *character == '\n')?;
    Some(last_newline + 1)
}

/// `\s+(?!\S)` falling back to `\s+` -- consumes the whole trailing
/// whitespace run if it reaches end-of-input, otherwise all but its
/// last char (left for the next pretoken to pick up via
/// [`match_words`]/[`match_punct`]'s optional lead), and always at
/// least one char.
fn match_trailing_whitespace(chars: &[char]) -> Option<usize> {
    let first = *chars.first()?;
    if !is_whitespace(first) {
        return None;
    }
    let run = chars
        .iter()
        .take_while(|character| is_whitespace(**character))
        .count();
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
        spans_under(text, PreType::GroupedDigits)
    }

    fn spans_under(text: &str, pre_type: PreType) -> Vec<&str> {
        pretokenize(text, pre_type)
            .into_iter()
            .map(|range| &text[range])
            .collect()
    }

    #[test]
    fn qwen_digits_split_one_per_pretoken() {
        assert_eq!(
            spans_under("$1,299.99", PreType::SingleDigit),
            ["$", "1", ",", "2", "9", "9", ".", "9", "9"]
        );
        assert_eq!(spans_under("3333", PreType::SingleDigitMarks), ["3", "3", "3", "3"]);
    }

    #[test]
    fn llama3_digits_still_group_in_threes_where_qwen_splits_singly() {
        assert_eq!(spans_under("v2024", PreType::GroupedDigits), ["v", "202", "4"]);
        assert_eq!(spans_under("v2024", PreType::SingleDigit), ["v", "2", "0", "2", "4"]);
    }

    #[test]
    fn qwen35_keeps_combining_marks_inside_the_word() {
        let text = "cafe\u{301} au";
        assert_eq!(
            spans_under(text, PreType::SingleDigitMarks),
            ["cafe\u{301}", " au"]
        );
        assert_eq!(
            spans_under(text, PreType::SingleDigit),
            ["cafe", "\u{301}", " au"]
        );
    }

    #[test]
    fn default_pre_split_cuts_digit_runs_into_ascii_triples_after_punctuation_runs() {
        assert_eq!(
            spans_under("x=1234567;", PreType::Default),
            ["x", "=", "123", "456", "7", ";"]
        );
        assert_eq!(
            spans_under("$12.50", PreType::Default),
            ["$", "12", ".", "50"]
        );
    }

    #[test]
    fn default_pre_split_isolates_the_apostrophe_before_contractions_can_match() {
        assert_eq!(spans_under("don't", PreType::Default), ["don", "'", "t"]);
        assert_eq!(
            spans_under("Hello, world!", PreType::Default),
            ["Hello", ",", " world", "!"]
        );
    }

    #[test]
    fn default_pre_split_covers_the_input_contiguously() {
        let text = "  call +1 (415) 555-0132 \u{2162} 1234567\u{0663}\u{0664}  \n";
        assert_eq!(spans_under(text, PreType::Default).concat(), text);
    }

    #[test]
    fn gpt2_pre_split_keeps_digit_runs_whole_and_attaches_the_space() {
        assert_eq!(spans_under("don't 12345", PreType::Gpt2), ["don", "'t", " 12345"]);
    }

    #[test]
    fn digit_isolated_gpt2_pre_split_cuts_every_digit_out_of_words() {
        assert_eq!(
            spans_under("x1 y22", PreType::DigitIsolatedGpt2),
            ["x", "1", " y", "2", "2"]
        );
    }

    #[test]
    fn falcon_pre_split_isolates_backticks_and_cuts_ascii_digit_triples() {
        assert_eq!(
            spans_under("a`b1234", PreType::Falcon),
            ["a", "`", "b", "123", "4"]
        );
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
        let text =
            "Hello, y'all! How are you \u{1F601} ?\u{6211}\u{60F3}\u{5728}apple1314151\u{5929}~";
        let mut rebuilt = alloc::string::String::new();
        for span in spans(text) {
            rebuilt.push_str(span);
        }
        assert_eq!(rebuilt, text);
    }

    fn newline_run_spans(text: &str) -> Vec<&str> {
        pretokenize_newline_runs(text)
            .into_iter()
            .map(|range| &text[range])
            .collect()
    }

    #[test]
    fn newline_runs_split_only_on_newline_boundaries() {
        assert_eq!(newline_run_spans("a\nb"), ["a", "\n", "b"]);
        assert_eq!(newline_run_spans("a\n\nb"), ["a", "\n\n", "b"]);
        assert_eq!(newline_run_spans("\n\n\n"), ["\n\n\n"]);
    }

    #[test]
    fn newline_runs_keep_spaces_carriage_returns_and_multibyte_inside_a_line() {
        assert_eq!(
            newline_run_spans("a  b\r\n\u{201c}q\u{201d} x"),
            ["a  b\r", "\n", "\u{201c}q\u{201d} x"]
        );
    }

    #[test]
    fn newline_runs_of_empty_text_is_empty() {
        assert!(newline_run_spans("").is_empty());
    }

    #[test]
    fn newline_runs_cover_the_input_contiguously() {
        let text = "\nfirst line\n\nsecond \u{1F600} line\n";
        let joined: alloc::string::String = newline_run_spans(text).concat();
        assert_eq!(joined, text);
    }
}
