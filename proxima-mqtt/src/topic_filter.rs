//! Topic-filter predicate for `SUBSCRIBE` matching — a small
//! [`TopicFilterSet`] predicate VALUE plugged into the EXISTING generic
//! `proxima_primitives::pipe::live_filter::LiveFilter<Predicate>`, mirroring
//! `proxima_redis::glob::GlobSet`'s identical role for PSUBSCRIBE: a
//! `BTreeSet`-backed live-swappable set with the same `with`/`without`
//! copy-on-write shape, MQTT wildcard match instead of redis glob syntax.
//! No new combinator — `LiveFilter` already supplies the live-cell
//! wait-free read + copy-on-write control plane; this file only supplies
//! the predicate VALUE.
//!
//! MQTT topic filters have exactly two wildcards (RFC-mandated, unlike
//! redis's shell-glob syntax): `+` matches exactly one level, `#` matches
//! its own level and every level after it (only legal as the filter's
//! final level). `/` is the level separator. Per [MQTT-4.7.2-1], a filter
//! starting with a wildcard must never match a topic starting with `$`
//! (the reserved `$SYS/...` namespace).

use std::collections::BTreeSet;

/// A live-swappable set of MQTT topic filters. [`TopicFilterSet::matching`]
/// answers "which of my filters match this topic name" — the query
/// [`crate::broker::MqttBroker`] needs on every `PUBLISH` to find the
/// subscriptions a topic satisfies.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TopicFilterSet {
    filters: BTreeSet<Vec<u8>>,
}

impl TopicFilterSet {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A set seeded from `filters`.
    pub fn from_filters(filters: impl IntoIterator<Item = Vec<u8>>) -> Self {
        Self {
            filters: filters.into_iter().collect(),
        }
    }

    /// A copy with `filter` added — the copy-on-write step for
    /// `FilterControl::update`.
    #[must_use]
    pub fn with(&self, filter: Vec<u8>) -> Self {
        let mut filters = self.filters.clone();
        filters.insert(filter);
        Self { filters }
    }

    /// A copy with `filter` removed.
    #[must_use]
    pub fn without(&self, filter: &[u8]) -> Self {
        let mut filters = self.filters.clone();
        filters.remove(filter);
        Self { filters }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.filters.len()
    }

    /// Every registered filter that matches `topic`.
    pub fn matching<'a>(&'a self, topic: &'a [u8]) -> impl Iterator<Item = &'a [u8]> + 'a {
        self.filters
            .iter()
            .filter(move |filter| topic_matches(filter, topic))
            .map(Vec::as_slice)
    }
}

/// [MQTT-4.7.2-1]: a filter with `#`/`+` in its first level must never
/// match a `$`-prefixed topic (the reserved namespace).
fn crosses_dollar_boundary(filter: &[u8], topic: &[u8]) -> bool {
    topic.first() == Some(&b'$') && matches!(filter.first(), Some(&b'#') | Some(&b'+'))
}

/// Does `filter` (RFC 3.8.3.1 topic filter syntax) match `topic` (a
/// concrete, wildcard-free topic name)?
#[must_use]
pub fn topic_matches(filter: &[u8], topic: &[u8]) -> bool {
    if crosses_dollar_boundary(filter, topic) {
        return false;
    }
    let mut filter_levels = filter.split(|byte| *byte == b'/');
    let mut topic_levels = topic.split(|byte| *byte == b'/');
    loop {
        match (filter_levels.next(), topic_levels.next()) {
            (Some(f), Some(t)) => {
                if f == b"#" {
                    return true;
                }
                if f != b"+" && f != t {
                    return false;
                }
            }
            (Some(b"#"), None) => return true,
            (Some(_), None) | (None, Some(_)) => return false,
            (None, None) => return true,
        }
    }
}

/// Is `filter` well-formed per RFC 3.8.3.1: `#` (if present) is only its
/// own final level, `+` (wherever present) is only a whole level, and the
/// filter is non-empty?
#[must_use]
pub fn is_valid_filter(filter: &[u8]) -> bool {
    if filter.is_empty() {
        return false;
    }
    let levels: Vec<&[u8]> = filter.split(|byte| *byte == b'/').collect();
    for (index, level) in levels.iter().enumerate() {
        let is_last = index + 1 == levels.len();
        if level.contains(&b'#') && (*level != b"#" || !is_last) {
            return false;
        }
        if level.contains(&b'+') && *level != b"+" {
            return false;
        }
    }
    true
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn exact_filter_matches_only_itself() {
        assert!(topic_matches(b"sport/tennis", b"sport/tennis"));
        assert!(!topic_matches(b"sport/tennis", b"sport/soccer"));
    }

    #[test]
    fn plus_matches_exactly_one_level() {
        assert!(topic_matches(b"sport/+", b"sport/tennis"));
        assert!(!topic_matches(b"sport/+", b"sport/tennis/player1"));
        assert!(topic_matches(b"sport/+/player1", b"sport/tennis/player1"));
    }

    #[test]
    fn hash_matches_its_own_level_and_everything_beneath() {
        assert!(topic_matches(b"sport/#", b"sport"));
        assert!(topic_matches(b"sport/#", b"sport/tennis"));
        assert!(topic_matches(b"sport/#", b"sport/tennis/player1"));
        assert!(topic_matches(b"#", b"anything/at/all"));
    }

    #[test]
    fn dollar_topics_are_excluded_from_leading_wildcards() {
        assert!(!topic_matches(b"#", b"$SYS/broker/uptime"));
        assert!(!topic_matches(b"+/broker/uptime", b"$SYS/broker/uptime"));
        assert!(topic_matches(b"$SYS/#", b"$SYS/broker/uptime"), "an explicit $SYS filter still matches");
    }

    #[test]
    fn topic_filter_set_matching_finds_every_satisfied_filter() {
        let set = TopicFilterSet::from_filters([
            b"sport/#".to_vec(),
            b"chat/+".to_vec(),
            b"sport/tennis".to_vec(),
        ]);
        let matched: Vec<&[u8]> = set.matching(b"sport/tennis").collect();
        assert_eq!(matched.len(), 2, "both sport/# and sport/tennis match");
    }

    #[test]
    fn topic_filter_set_with_and_without_are_copy_on_write() {
        let empty = TopicFilterSet::new();
        let with_one = empty.with(b"a/+".to_vec());
        assert!(empty.is_empty(), "original set is untouched");
        assert_eq!(with_one.len(), 1);

        let cleared = with_one.without(b"a/+");
        assert!(cleared.is_empty());
        assert_eq!(with_one.len(), 1, "without does not mutate its receiver");
    }

    #[test]
    fn valid_filter_accepts_well_formed_wildcards() {
        assert!(is_valid_filter(b"sport/tennis"));
        assert!(is_valid_filter(b"sport/+"));
        assert!(is_valid_filter(b"sport/+/player1"));
        assert!(is_valid_filter(b"sport/#"));
        assert!(is_valid_filter(b"#"));
    }

    #[test]
    fn valid_filter_rejects_malformed_wildcards() {
        assert!(!is_valid_filter(b""));
        assert!(!is_valid_filter(b"sport#"), "# must be a whole level");
        assert!(!is_valid_filter(b"sport/#/player1"), "# must be the final level");
        assert!(!is_valid_filter(b"sport+"), "+ must be a whole level");
    }
}
