//! AMQP topic-exchange binding-key matching — the `#`/`*` wildcard grammar
//! (AMQP 0-9-1 §3.1.3.3) plugged into a live-swappable set the same shape
//! `proxima_redis::glob::GlobSet` gives PSUBSCRIBE: [`TopicSet::matching`]
//! answers "which of my bound patterns match this routing key" for
//! [`crate::broker::AmqpBroker::publish`] on a `topic`-kind exchange.
//!
//! Unlike redis's byte-glob (`*`/`?`/`[...]`), a topic binding key is
//! `.`-delimited *words*: `*` matches exactly one word, `#` matches zero or
//! more words, anything else matches itself literally
//! (e.g. `orders.*.created` matches `orders.eu.created` but not
//! `orders.eu.region.created`; `orders.#` matches both).

use std::collections::BTreeSet;

/// A live-swappable set of topic binding-key patterns.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TopicSet {
    patterns: BTreeSet<Vec<u8>>,
}

impl TopicSet {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with(&self, pattern: Vec<u8>) -> Self {
        let mut patterns = self.patterns.clone();
        patterns.insert(pattern);
        Self { patterns }
    }

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

    /// Every registered pattern that topic-matches `routing_key`.
    pub fn matching<'a>(&'a self, routing_key: &'a [u8]) -> impl Iterator<Item = &'a [u8]> + 'a {
        self.patterns
            .iter()
            .filter(move |pattern| topic_match(pattern, routing_key))
            .map(Vec::as_slice)
    }
}

/// AMQP topic-exchange match: `.`-delimited words, `*` = exactly one word,
/// `#` = zero or more words, anything else = literal word equality.
#[must_use]
pub fn topic_match(pattern: &[u8], routing_key: &[u8]) -> bool {
    let pattern_words: Vec<&[u8]> = pattern.split(|byte| *byte == b'.').collect();
    let key_words: Vec<&[u8]> = routing_key.split(|byte| *byte == b'.').collect();
    match_words(&pattern_words, &key_words)
}

fn match_words(pattern: &[&[u8]], key: &[&[u8]]) -> bool {
    match pattern.split_first() {
        None => key.is_empty(),
        Some((&b"#", rest)) => (0..=key.len()).any(|split| match_words(rest, &key[split..])),
        Some((&b"*", rest)) => match key.split_first() {
            Some((_, key_rest)) => match_words(rest, key_rest),
            None => false,
        },
        Some((&word, rest)) => match key.split_first() {
            Some((&head, key_rest)) if head == word => match_words(rest, key_rest),
            _ => false,
        },
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn literal_pattern_matches_only_itself() {
        assert!(topic_match(b"orders.eu.created", b"orders.eu.created"));
        assert!(!topic_match(b"orders.eu.created", b"orders.eu.updated"));
    }

    #[test]
    fn star_matches_exactly_one_word() {
        assert!(topic_match(b"orders.*.created", b"orders.eu.created"));
        assert!(!topic_match(
            b"orders.*.created",
            b"orders.eu.region.created"
        ));
        assert!(!topic_match(b"orders.*.created", b"orders.created"));
    }

    #[test]
    fn hash_matches_zero_or_more_words() {
        assert!(topic_match(b"orders.#", b"orders"));
        assert!(topic_match(b"orders.#", b"orders.eu"));
        assert!(topic_match(b"orders.#", b"orders.eu.region.created"));
        assert!(!topic_match(b"orders.#", b"shipments.eu"));
    }

    #[test]
    fn bare_hash_matches_everything() {
        assert!(topic_match(b"#", b"orders.eu.created"));
        assert!(topic_match(b"#", b""));
    }

    #[test]
    fn topic_set_matching_finds_every_satisfied_pattern() {
        let set = TopicSet::default()
            .with(b"orders.*.created".to_vec())
            .with(b"orders.#".to_vec())
            .with(b"shipments.#".to_vec());
        let matched: Vec<&[u8]> = set.matching(b"orders.eu.created").collect();
        assert_eq!(matched.len(), 2);
        assert!(matched.contains(&b"orders.*.created".as_slice()));
        assert!(matched.contains(&b"orders.#".as_slice()));
    }

    #[test]
    fn with_and_without_are_copy_on_write() {
        let empty = TopicSet::new();
        let with_one = empty.with(b"orders.#".to_vec());
        assert!(empty.is_empty());
        assert_eq!(with_one.len(), 1);

        let cleared = with_one.without(b"orders.#");
        assert!(cleared.is_empty());
        assert_eq!(with_one.len(), 1, "without does not mutate its receiver");
    }
}
