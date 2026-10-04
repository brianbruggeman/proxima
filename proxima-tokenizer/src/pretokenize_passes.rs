//! The multi-pass pre-splits: llama.cpp pre types whose regex list is several
//! expressions applied one after another, each splitting every piece the
//! previous one produced (`unicode_regex_split`, `unicode.cpp:1190-1390`).
//!
//! ```text
//! Default  [\p{P}\$\+<=>\^~\|]+  |  gpt2  |  \p{N}+  |  [0-9][0-9][0-9]
//! Falcon   [\p{P}\$\+<=>\^~\|`]+ |  gpt2  |  [0-9][0-9][0-9]
//! Gpt2     gpt2
//! DigitIsolatedGpt2   \p{N}  |  gpt2
//! gpt2 = 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
//! ```
//!
//! Class passes keep matches and the gaps between them as pieces
//! (`unicode_regex_split_stl`); the gpt2 pass is llama.cpp's hand-written
//! matcher (`unicode_regex_split_custom_gpt2`, `unicode.cpp:215-331`),
//! mirrored here including its single-char fallback and its treatment of a
//! lone trailing space. The punctuation class reads ASCII through the literal
//! class `unicode_regex_split` builds and non-ASCII through the collapsed
//! category byte, so `\p{P}` is the table for non-ASCII and an explicit list
//! for ASCII.

use alloc::vec::Vec;

use crate::pretokenize::{is_digit, is_letter, is_punctuation, is_whitespace};

const ASCII_PUNCT_CLASS: &str = "!\"#%&'()*,-./:;?@[\\]_{}$+<=>^~|";

type Pieces = Vec<(usize, usize)>;

/// One regex of a pre type's list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pass {
    /// `[\p{P}\$\+<=>\^~\|]+`
    PunctRuns,
    /// `[\p{P}\$\+<=>\^~\|`]+`
    PunctAndBacktickRuns,
    /// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)`
    Gpt2Words,
    /// `\p{N}+`
    DigitRuns,
    /// `\p{N}`
    EachDigit,
    /// `[0-9][0-9][0-9]`
    AsciiDigitTriples,
}

pub(crate) const DEFAULT_PASSES: [Pass; 4] = [
    Pass::PunctRuns,
    Pass::Gpt2Words,
    Pass::DigitRuns,
    Pass::AsciiDigitTriples,
];
pub(crate) const FALCON_PASSES: [Pass; 3] = [
    Pass::PunctAndBacktickRuns,
    Pass::Gpt2Words,
    Pass::AsciiDigitTriples,
];
pub(crate) const GPT2_PASSES: [Pass; 1] = [Pass::Gpt2Words];
pub(crate) const DIGIT_ISOLATED_GPT2_PASSES: [Pass; 2] = [Pass::EachDigit, Pass::Gpt2Words];

pub(crate) fn run_passes(chars: &[char], passes: &[Pass]) -> Pieces {
    if chars.is_empty() {
        return Vec::new();
    }
    let mut pieces = alloc::vec![(0, chars.len())];
    for pass in passes {
        pieces = match pass {
            Pass::PunctRuns => split_runs(chars, &pieces, is_punct_class),
            Pass::PunctAndBacktickRuns => split_runs(chars, &pieces, is_punct_or_backtick_class),
            Pass::Gpt2Words => split_gpt2_words(chars, &pieces),
            Pass::DigitRuns => split_runs(chars, &pieces, is_digit),
            Pass::EachDigit => split_each(chars, &pieces, is_digit),
            Pass::AsciiDigitTriples => split_digit_triples(chars, &pieces),
        };
    }
    pieces
}

fn is_punct_class(character: char) -> bool {
    if character.is_ascii() {
        ASCII_PUNCT_CLASS.contains(character)
    } else {
        !is_whitespace(character) && is_punctuation(character)
    }
}

fn is_punct_or_backtick_class(character: char) -> bool {
    character == '`' || is_punct_class(character)
}

fn split_runs(chars: &[char], pieces: &[(usize, usize)], in_class: fn(char) -> bool) -> Pieces {
    let mut out = Vec::with_capacity(pieces.len());
    for &(start, end) in pieces {
        let mut run_start = start;
        for index in start..end {
            if index > run_start && in_class(chars[index]) != in_class(chars[run_start]) {
                out.push((run_start, index));
                run_start = index;
            }
        }
        out.push((run_start, end));
    }
    out
}

fn split_each(chars: &[char], pieces: &[(usize, usize)], in_class: fn(char) -> bool) -> Pieces {
    let mut out = Vec::with_capacity(pieces.len());
    for &(start, end) in pieces {
        let mut gap_start = start;
        for (index, &character) in chars.iter().enumerate().take(end).skip(start) {
            if in_class(character) {
                if gap_start < index {
                    out.push((gap_start, index));
                }
                out.push((index, index + 1));
                gap_start = index + 1;
            }
        }
        if gap_start < end {
            out.push((gap_start, end));
        }
    }
    out
}

fn split_digit_triples(chars: &[char], pieces: &[(usize, usize)]) -> Pieces {
    let mut out = Vec::with_capacity(pieces.len());
    for &(start, end) in pieces {
        let mut gap_start = start;
        let mut index = start;
        while index + 3 <= end {
            if chars[index..index + 3].iter().all(char::is_ascii_digit) {
                if gap_start < index {
                    out.push((gap_start, index));
                }
                out.push((index, index + 3));
                index += 3;
                gap_start = index;
            } else {
                index += 1;
            }
        }
        if gap_start < end {
            out.push((gap_start, end));
        }
    }
    out
}

fn split_gpt2_words(chars: &[char], pieces: &[(usize, usize)]) -> Pieces {
    let mut out = Vec::with_capacity(pieces.len());
    for &(start, end) in pieces {
        let mut position = start;
        while position < end {
            let length = gpt2_token_len(&chars[position..end]);
            out.push((position, position + length));
            position += length;
        }
    }
    out
}

fn contraction_len(chars: &[char]) -> Option<usize> {
    if chars.first() != Some(&'\'') {
        return None;
    }
    match (chars.get(1), chars.get(2)) {
        (Some('s' | 't' | 'm' | 'd'), _) => Some(2),
        (Some('r' | 'v'), Some('e')) | (Some('l'), Some('l')) => Some(3),
        _ => None,
    }
}

fn run_len(chars: &[char], predicate: fn(char) -> bool) -> usize {
    chars.iter().take_while(|character| predicate(**character)).count()
}

fn is_symbol_run_char(character: char) -> bool {
    !is_whitespace(character) && !is_letter(character) && !is_digit(character)
}

fn gpt2_token_len(chars: &[char]) -> usize {
    if let Some(length) = contraction_len(chars) {
        return length;
    }
    let lead = usize::from(chars[0] == ' ');
    let Some(&probe) = chars.get(lead) else {
        return whitespace_len(chars);
    };
    for predicate in [is_letter, is_digit, is_symbol_run_char] {
        if predicate(probe) {
            return lead + run_len(&chars[lead..], predicate);
        }
    }
    whitespace_len(chars)
}

fn whitespace_len(chars: &[char]) -> usize {
    let run = run_len(chars, is_whitespace);
    if run > 1 && run < chars.len() {
        run - 1
    } else {
        run.max(1)
    }
}
