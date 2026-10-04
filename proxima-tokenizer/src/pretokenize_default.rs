//! `LLAMA_VOCAB_PRE_TYPE_DEFAULT`, the pre-split llama.cpp applies when
//! `tokenizer.ggml.pre` is `"default"` or missing (`llama-vocab.cpp:556-566`).
//! It is four regexes applied one after another, each splitting every piece
//! the previous one produced:
//!
//! ```text
//! 1  [\p{P}\$\+<=>\^~\|]+
//! 2  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
//! 3  \p{N}+
//! 4  [0-9][0-9][0-9]
//! ```
//!
//! Passes 1, 3 and 4 keep matches and the gaps between them as pieces
//! (`unicode_regex_split_stl`); pass 2 is llama.cpp's hand-written matcher
//! (`unicode_regex_split_custom_gpt2`, `unicode.cpp:215-331`), mirrored here
//! including its single-char fallback and its treatment of a lone trailing
//! space. Pass 1 reads ASCII through the literal class `unicode_regex_split`
//! builds and non-ASCII through the collapsed category byte, so `\p{P}` is the
//! table for non-ASCII and an explicit list for ASCII.

use alloc::vec::Vec;

use crate::pretokenize::{is_digit, is_letter, is_punctuation, is_whitespace};

const ASCII_PASS_ONE_CLASS: &str = "!\"#%&'()*,-./:;?@[\\]_{}$+<=>^~|";

type Pieces = Vec<(usize, usize)>;

pub(crate) fn default_spans(chars: &[char]) -> Pieces {
    if chars.is_empty() {
        return Vec::new();
    }
    let pieces = alloc::vec![(0, chars.len())];
    let pieces = split_runs(chars, &pieces, is_pass_one_class);
    let pieces = split_gpt2_words(chars, &pieces);
    let pieces = split_runs(chars, &pieces, is_digit);
    split_digit_triples(chars, &pieces)
}

fn is_pass_one_class(character: char) -> bool {
    if character.is_ascii() {
        ASCII_PASS_ONE_CLASS.contains(character)
    } else {
        !is_whitespace(character) && is_punctuation(character)
    }
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
