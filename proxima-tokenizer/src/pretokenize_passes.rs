//! The multi-pass pre-splits: llama.cpp pre types whose regex list is several
//! expressions applied one after another, each splitting every piece the
//! previous one produced (`unicode_regex_split`, `unicode.cpp:1190-1390`).
//!
//! ```text
//! Default  [\p{P}\$\+<=>\^~\|]+  |  gpt2  |  \p{N}+  |  [0-9][0-9][0-9]
//! Falcon   [\p{P}\$\+<=>\^~\|`]+ |  gpt2  |  [0-9][0-9][0-9]
//! Gpt2     gpt2
//! DigitIsolatedGpt2   \p{N}  |  gpt2
//! Flint    \p{N}{1,3}  |  [一-龥぀-ゟ゠-ヿ]+  |  flint-word
//! gpt2 = 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
//! flint-word = [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?
//!            | ?[\p{P}\p{S}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
//! ```
//!
//! The flint rule is `llm_tokenizer_bpe`'s `LLAMA_VOCAB_PRE_TYPE_FLINT` case at
//! llama.cpp cd959bae9 `src/llama-vocab.cpp:285-290`. It has no hand-written
//! matcher (`unicode_regex_split_custom`, `src/unicode.cpp:1219-1256`), so it
//! runs the generic `std::regex`-over-collapsed-text path, where
//! `unicode_regex_split` (`src/unicode.cpp:1491-1512`) collapses `\p{Lu}`,
//! `\p{Lt}`, `\p{Lm}`, `\p{Lo}` and `\p{Ll}` onto one LETTER byte -- which is
//! why the two case-alternatives here are one letter-or-mark class.
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

use crate::pretokenize::{
    contraction_len as possessive_len, is_digit, is_letter, is_mark, is_punctuation, is_symbol,
    is_whitespace, match_trailing_whitespace, match_whitespace_with_newline,
};

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
    /// `\p{N}{1,3}` -- any digit run, in threes counted from the left
    NumberTriples,
    /// `[一-龥぀-ゟ゠-ヿ]+`
    CjkRuns,
    /// The flint word alternation, see this module's doc block.
    FlintWords,
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
pub(crate) const FLINT_PASSES: [Pass; 3] = [Pass::NumberTriples, Pass::CjkRuns, Pass::FlintWords];

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
            Pass::NumberTriples => split_number_triples(chars, &pieces),
            Pass::CjkRuns => split_runs(chars, &pieces, is_cjk_run_char),
            Pass::FlintWords => split_flint_words(chars, &pieces),
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

/// `\p{N}{1,3}`: every digit run cut into threes counted from the left.
fn split_number_triples(chars: &[char], pieces: &[(usize, usize)]) -> Pieces {
    let mut out = Vec::with_capacity(pieces.len());
    for &(start, end) in pieces {
        let mut gap_start = start;
        let mut index = start;
        while index < end {
            if !is_digit(chars[index]) {
                index += 1;
                continue;
            }
            if gap_start < index {
                out.push((gap_start, index));
            }
            let mut chunk_end = index;
            while chunk_end < end && chunk_end - index < 3 && is_digit(chars[chunk_end]) {
                chunk_end += 1;
            }
            out.push((index, chunk_end));
            index = chunk_end;
            gap_start = index;
        }
        if gap_start < end {
            out.push((gap_start, end));
        }
    }
    out
}

/// `[一-龥぀-ゟ゠-ヿ]+`: CJK unified ideographs through U+9FA5, hiragana, katakana.
fn is_cjk_run_char(character: char) -> bool {
    matches!(
        character,
        '\u{4E00}'..='\u{9FA5}' | '\u{3040}'..='\u{309F}' | '\u{30A0}'..='\u{30FF}'
    )
}

fn split_flint_words(chars: &[char], pieces: &[(usize, usize)]) -> Pieces {
    let mut out = Vec::with_capacity(pieces.len());
    for &(start, end) in pieces {
        let mut gap_start = start;
        let mut position = start;
        while position < end {
            match flint_token_len(&chars[position..end]) {
                Some(length) => {
                    if gap_start < position {
                        out.push((gap_start, position));
                    }
                    out.push((position, position + length));
                    position += length;
                    gap_start = position;
                }
                None => position += 1,
            }
        }
        if gap_start < end {
            out.push((gap_start, end));
        }
    }
    out
}

/// The first alternative of the flint rule that matches at the start of
/// `chars`, in llama.cpp's order. `None` means no alternative matches here, so
/// the char joins the gap `unicode_regex_split_stl` keeps between matches.
fn flint_token_len(chars: &[char]) -> Option<usize> {
    flint_word_len(chars)
        .or_else(|| flint_punct_len(chars))
        .or_else(|| match_whitespace_with_newline(chars))
        .or_else(|| match_trailing_whitespace(chars))
}

/// `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+(?:'[sS]|...[dD])?`, with the possessive
/// kept on the word it belongs to rather than split off.
fn flint_word_len(chars: &[char]) -> Option<usize> {
    let first = *chars.first()?;
    let lead = if is_word_char(first) {
        0
    } else if flint_can_lead(first) && chars.get(1).is_some_and(|next| is_word_char(*next)) {
        1
    } else {
        return None;
    };
    let run = chars[lead..]
        .iter()
        .take_while(|character| is_word_char(**character))
        .count();
    if run == 0 {
        return None;
    }
    let word_end = lead + run;
    Some(word_end + possessive_len(&chars[word_end..]).unwrap_or(0))
}

fn is_word_char(character: char) -> bool {
    is_letter(character) || is_mark(character)
}

fn flint_can_lead(character: char) -> bool {
    character != '\r' && character != '\n' && !is_word_char(character) && !is_digit(character)
}

/// ` ?[\p{P}\p{S}]+[\r\n]*`
fn flint_punct_len(chars: &[char]) -> Option<usize> {
    let first = *chars.first()?;
    let lead = if is_punct_or_symbol(first) {
        0
    } else if first == ' ' && chars.get(1).is_some_and(|next| is_punct_or_symbol(*next)) {
        1
    } else {
        return None;
    };
    let run = chars[lead..]
        .iter()
        .take_while(|character| is_punct_or_symbol(**character))
        .count();
    if run == 0 {
        return None;
    }
    let after_punct = lead + run;
    let newline_run = chars[after_punct..]
        .iter()
        .take_while(|character| **character == '\r' || **character == '\n')
        .count();
    Some(after_punct + newline_run)
}

/// `[\p{P}\p{S}]`: ASCII punctuation is the same set either way, so only the
/// non-ASCII range needs the table's two category bits.
fn is_punct_or_symbol(character: char) -> bool {
    if character.is_ascii() {
        character.is_ascii_punctuation()
    } else {
        is_punctuation(character) || is_symbol(character)
    }
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

fn gpt2_contraction_len(chars: &[char]) -> Option<usize> {
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
    chars
        .iter()
        .take_while(|character| predicate(**character))
        .count()
}

fn is_symbol_run_char(character: char) -> bool {
    !is_whitespace(character) && !is_letter(character) && !is_digit(character)
}

fn gpt2_token_len(chars: &[char]) -> usize {
    if let Some(length) = gpt2_contraction_len(chars) {
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
