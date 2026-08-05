//! AMQP topic-exchange binding-key matching — the `#`/`*` wildcard grammar
//! (AMQP 0-9-1 §3.1.3.3). [`topic_match`] answers "does this binding key
//! cover this routing key" for [`crate::broker::AmqpBroker::publish`] on a
//! `topic`-kind exchange.
//!
//! Unlike redis's byte-glob (`*`/`?`/`[...]`), a topic binding key is
//! `.`-delimited *words*: `*` matches exactly one word, `#` matches zero or
//! more words, anything else matches itself literally
//! (e.g. `orders.*.created` matches `orders.eu.created` but not
//! `orders.eu.region.created`; `orders.#` matches both).

use alloc::vec::Vec;

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
    fn several_binding_keys_can_cover_one_routing_key_at_once() {
        let bindings: [&[u8]; 3] = [b"orders.*.created", b"orders.#", b"shipments.#"];
        let matched: Vec<&[u8]> = bindings
            .into_iter()
            .filter(|binding| topic_match(binding, b"orders.eu.created"))
            .collect();
        assert_eq!(
            matched,
            vec![b"orders.*.created".as_slice(), b"orders.#".as_slice()]
        );
    }
}
