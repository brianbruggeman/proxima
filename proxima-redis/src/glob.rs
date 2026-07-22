//! Glob predicate for PSUBSCRIBE pattern matching — a small [`GlobSet`]
//! predicate VALUE plugged into the EXISTING generic
//! `proxima_primitives::pipe::live_filter::LiveFilter<Predicate>`, mirroring
//! how [`proxima_primitives::pipe::IdSet`] plugs into that same generic
//! (`LiveFilter<IdSet<Id>>`): a `BTreeSet`-backed live-swappable set with the
//! same `with`/`without` copy-on-write shape, glob-match instead of exact
//! membership. No new combinator — `LiveFilter` already supplies the
//! live-cell wait-free read + copy-on-write control plane; this file only
//! supplies the predicate VALUE.

use std::collections::BTreeSet;

/// A live-swappable set of glob patterns (Redis PSUBSCRIBE syntax: `*` any
/// run including empty, `?` exactly one byte, `[abc]`/`[^abc]`/`[a-z]` a
/// char class, `\x` escapes `x` literally). [`GlobSet::matching`] answers
/// "which of my patterns match this channel" — the query `RedisBroker`
/// needs on every PUBLISH to find the pattern subscriptions a channel
/// satisfies.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GlobSet {
    patterns: BTreeSet<Vec<u8>>,
}

impl GlobSet {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A set seeded from `patterns`.
    pub fn from_patterns(patterns: impl IntoIterator<Item = Vec<u8>>) -> Self {
        Self {
            patterns: patterns.into_iter().collect(),
        }
    }

    /// A copy with `pattern` added — the copy-on-write step for
    /// `FilterControl::update`.
    #[must_use]
    pub fn with(&self, pattern: Vec<u8>) -> Self {
        let mut patterns = self.patterns.clone();
        patterns.insert(pattern);
        Self { patterns }
    }

    /// A copy with `pattern` removed.
    #[must_use]
    pub fn without(&self, pattern: &[u8]) -> Self {
        let mut patterns = self.patterns.clone();
        patterns.remove(pattern);
        Self { patterns }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    /// Every registered pattern that glob-matches `channel`.
    pub fn matching<'a>(&'a self, channel: &'a [u8]) -> impl Iterator<Item = &'a [u8]> + 'a {
        self.patterns
            .iter()
            .filter(move |pattern| glob_match(pattern, channel))
            .map(Vec::as_slice)
    }
}

/// Redis-syntax glob match. Classic backtracking matcher — pattern length is
/// operator-controlled (a subscribed pattern string), so the backtracking
/// cost is bounded by what an operator subscribed, not by attacker input.
#[must_use]
pub fn glob_match(pattern: &[u8], text: &[u8]) -> bool {
    do_match(pattern, text)
}

fn do_match(mut pattern: &[u8], mut text: &[u8]) -> bool {
    loop {
        match pattern.first() {
            None => return text.is_empty(),
            Some(b'*') => {
                while pattern.first() == Some(&b'*') {
                    pattern = &pattern[1..];
                }
                if pattern.is_empty() {
                    return true;
                }
                return (0..=text.len()).any(|start| do_match(pattern, &text[start..]));
            }
            Some(b'?') => {
                let Some((_, rest)) = text.split_first() else {
                    return false;
                };
                text = rest;
                pattern = &pattern[1..];
            }
            Some(b'[') => {
                let (matched, pattern_used, text_used) = match_class(pattern, text);
                if !matched {
                    return false;
                }
                pattern = &pattern[pattern_used..];
                text = &text[text_used..];
            }
            Some(b'\\') if pattern.len() > 1 => {
                let literal = pattern[1];
                match text.first() {
                    Some(&byte) if byte == literal => {
                        text = &text[1..];
                        pattern = &pattern[2..];
                    }
                    _ => return false,
                }
            }
            Some(&literal) => match text.first() {
                Some(&byte) if byte == literal => {
                    text = &text[1..];
                    pattern = &pattern[1..];
                }
                _ => return false,
            },
        }
    }
}

/// Parses one `[...]` char class starting at `pattern[0] == b'['`. Returns
/// `(matched-against-text's-first-byte, pattern-bytes-consumed,
/// text-bytes-consumed)`. An unterminated class (no closing `]`) falls back
/// to treating `[` as a literal byte.
fn match_class(pattern: &[u8], text: &[u8]) -> (bool, usize, usize) {
    let mut index = 1;
    let negate = pattern.get(index) == Some(&b'^');
    if negate {
        index += 1;
    }
    let class_start = index;
    let mut end = index;
    // a `]` immediately after `[` or `[^` is a literal member, not the
    // terminator (the common glob-class convention).
    if pattern.get(end) == Some(&b']') {
        end += 1;
    }
    while end < pattern.len() && pattern[end] != b']' {
        end += 1;
    }
    if end >= pattern.len() {
        let literal_match = text.first() == Some(&b'[');
        return (literal_match, 1, usize::from(literal_match));
    }
    let class = &pattern[class_start..end];
    let pattern_used = end + 1;
    let Some(&byte) = text.first() else {
        return (false, pattern_used, 0);
    };
    let mut in_class = false;
    let mut cursor = 0;
    while cursor < class.len() {
        if cursor + 2 < class.len() && class[cursor + 1] == b'-' {
            let (low, high) = (class[cursor], class[cursor + 2]);
            if byte >= low && byte <= high {
                in_class = true;
            }
            cursor += 3;
        } else {
            if class[cursor] == byte {
                in_class = true;
            }
            cursor += 1;
        }
    }
    let matched = in_class != negate;
    (matched, pattern_used, usize::from(matched))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn exact_pattern_matches_only_itself() {
        assert!(glob_match(b"news.tech", b"news.tech"));
        assert!(!glob_match(b"news.tech", b"news.sport"));
    }

    #[test]
    fn star_matches_any_run_including_empty() {
        assert!(glob_match(b"news.*", b"news."));
        assert!(glob_match(b"news.*", b"news.tech"));
        assert!(glob_match(b"*", b"anything"));
        assert!(glob_match(b"*", b""));
    }

    #[test]
    fn question_mark_matches_exactly_one_byte() {
        assert!(glob_match(b"h?llo", b"hello"));
        assert!(!glob_match(b"h?llo", b"hllo"));
        assert!(!glob_match(b"h?llo", b"heello"));
    }

    #[test]
    fn char_class_matches_any_member() {
        assert!(glob_match(b"h[ae]llo", b"hello"));
        assert!(glob_match(b"h[ae]llo", b"hallo"));
        assert!(!glob_match(b"h[ae]llo", b"hillo"));
    }

    #[test]
    fn negated_char_class_excludes_members() {
        assert!(glob_match(b"h[^ae]llo", b"hillo"));
        assert!(!glob_match(b"h[^ae]llo", b"hello"));
    }

    #[test]
    fn char_class_range_matches_inclusive_bounds() {
        assert!(glob_match(b"item.[0-9]", b"item.5"));
        assert!(!glob_match(b"item.[0-9]", b"item.x"));
    }

    #[test]
    fn escaped_special_char_is_literal() {
        assert!(glob_match(b"news\\*tech", b"news*tech"));
        assert!(!glob_match(b"news\\*tech", b"newsXtech"));
    }

    #[test]
    fn glob_set_matching_finds_every_satisfied_pattern() {
        let set = GlobSet::from_patterns([b"news.*".to_vec(), b"chat.*".to_vec(), b"news.tech".to_vec()]);
        let matched: Vec<&[u8]> = set.matching(b"news.tech").collect();
        assert_eq!(matched.len(), 2, "both news.* and news.tech match");
        assert!(matched.contains(&b"news.*".as_slice()));
        assert!(matched.contains(&b"news.tech".as_slice()));
    }

    #[test]
    fn glob_set_with_and_without_are_copy_on_write() {
        let empty = GlobSet::new();
        let with_one = empty.with(b"a.*".to_vec());
        assert!(empty.is_empty(), "original set is untouched");
        assert_eq!(with_one.len(), 1);

        let cleared = with_one.without(b"a.*");
        assert!(cleared.is_empty());
        assert_eq!(with_one.len(), 1, "without does not mutate its receiver");
    }
}
