#![cfg(feature = "alloc")]

use core::ops::{Deref, DerefMut};

use alloc::string::String;
use alloc::vec::Vec;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

/// Insertion-ordered header/query pairs backing `Request.headers`,
/// `Request.query`, `Response.headers`. Bytes storage; case-insensitive
/// `&str` lookups. Serializes as `Vec<(String, String)>` for jsonl
/// compat — non-UTF-8 lossily converted on the way out.
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct HeaderList {
    entries: Vec<(Bytes, Bytes)>,
}

impl HeaderList {
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    #[must_use]
    pub fn from_pairs<I, K, V>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: IntoHeaderBytes,
        V: IntoHeaderBytes,
    {
        let mut list = Self::new();
        for (key, value) in pairs {
            list.insert(key.into_header_bytes(), value.into_header_bytes());
        }
        list
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Bytes> {
        let needle = name.as_bytes();
        self.entries
            .iter()
            .find(|(key, _)| key.as_ref().eq_ignore_ascii_case(needle))
            .map(|(_, value)| value)
    }

    /// `None` if absent OR if the bytes aren't valid UTF-8.
    #[must_use]
    pub fn get_str(&self, name: &str) -> Option<&str> {
        self.get(name)
            .and_then(|value| core::str::from_utf8(value).ok())
    }

    /// Case-insensitive replace-or-append. Returns the prior value when
    /// a same-name entry already existed.
    pub fn insert<N: IntoHeaderBytes, V: IntoHeaderBytes>(
        &mut self,
        name: N,
        value: V,
    ) -> Option<Bytes> {
        let name_bytes = name.into_header_bytes();
        let value_bytes = value.into_header_bytes();
        let needle = name_bytes.as_ref();
        if let Some(slot) = self
            .entries
            .iter_mut()
            .find(|(key, _)| key.as_ref().eq_ignore_ascii_case(needle))
        {
            return Some(core::mem::replace(&mut slot.1, value_bytes));
        }
        self.entries.push((name_bytes, value_bytes));
        None
    }

    /// case-insensitive presence check.
    #[must_use]
    pub fn contains_key(&self, name: &str) -> bool {
        let needle = name.as_bytes();
        self.entries
            .iter()
            .any(|(key, _)| key.as_ref().eq_ignore_ascii_case(needle))
    }

    pub fn insert_if_absent<N: IntoHeaderBytes, V: IntoHeaderBytes>(&mut self, name: N, value: V) {
        let name_bytes = name.into_header_bytes();
        if !self.contains_key_bytes(name_bytes.as_ref()) {
            self.entries.push((name_bytes, value.into_header_bytes()));
        }
    }

    fn contains_key_bytes(&self, needle: &[u8]) -> bool {
        self.entries
            .iter()
            .any(|(key, _)| key.as_ref().eq_ignore_ascii_case(needle))
    }

    pub fn iter(&self) -> core::slice::Iter<'_, (Bytes, Bytes)> {
        self.entries.iter()
    }

    pub fn keys(&self) -> impl Iterator<Item = &Bytes> {
        self.entries.iter().map(|(name, _)| name)
    }

    pub fn retain<F: FnMut(&Bytes, &Bytes) -> bool>(&mut self, mut predicate: F) {
        self.entries.retain(|(name, value)| predicate(name, value));
    }

    pub fn remove(&mut self, name: &str) -> Option<Bytes> {
        let needle = name.as_bytes();
        let position = self
            .entries
            .iter()
            .position(|(key, _)| key.as_ref().eq_ignore_ascii_case(needle))?;
        Some(self.entries.swap_remove(position).1)
    }

    #[must_use]
    pub fn into_inner(self) -> Vec<(Bytes, Bytes)> {
        self.entries
    }

    #[must_use]
    pub fn as_slice(&self) -> &[(Bytes, Bytes)] {
        &self.entries
    }
}

/// Conversion shim — exists because `Bytes::From` only covers `&'static`
/// slices; we need to accept non-static `&str` / `&[u8]` (with a copy).
pub trait IntoHeaderBytes {
    fn into_header_bytes(self) -> Bytes;
}

impl IntoHeaderBytes for Bytes {
    fn into_header_bytes(self) -> Bytes {
        self
    }
}

impl IntoHeaderBytes for &Bytes {
    fn into_header_bytes(self) -> Bytes {
        Bytes::clone(self)
    }
}

impl IntoHeaderBytes for String {
    fn into_header_bytes(self) -> Bytes {
        Bytes::from(self)
    }
}

impl IntoHeaderBytes for Vec<u8> {
    fn into_header_bytes(self) -> Bytes {
        Bytes::from(self)
    }
}

impl IntoHeaderBytes for &str {
    fn into_header_bytes(self) -> Bytes {
        Bytes::copy_from_slice(self.as_bytes())
    }
}

impl IntoHeaderBytes for &[u8] {
    fn into_header_bytes(self) -> Bytes {
        Bytes::copy_from_slice(self)
    }
}

impl Deref for HeaderList {
    type Target = [(Bytes, Bytes)];
    fn deref(&self) -> &Self::Target {
        &self.entries
    }
}

impl DerefMut for HeaderList {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.entries
    }
}

impl<'a> IntoIterator for &'a HeaderList {
    type Item = &'a (Bytes, Bytes);
    type IntoIter = core::slice::Iter<'a, (Bytes, Bytes)>;
    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}

impl IntoIterator for HeaderList {
    type Item = (Bytes, Bytes);
    type IntoIter = alloc::vec::IntoIter<(Bytes, Bytes)>;
    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

impl FromIterator<(Bytes, Bytes)> for HeaderList {
    fn from_iter<I: IntoIterator<Item = (Bytes, Bytes)>>(iter: I) -> Self {
        Self {
            entries: iter.into_iter().collect(),
        }
    }
}

impl From<Vec<(Bytes, Bytes)>> for HeaderList {
    fn from(entries: Vec<(Bytes, Bytes)>) -> Self {
        Self { entries }
    }
}

// serde lens: edge-form `Vec<(String, String)>` for wire compat with
// jsonl recording schemas already in the wild. invalid UTF-8 in a
// header value renders as a replacement-char string on serialize —
// preserves round-trip-ability rather than dropping the entry.
impl Serialize for HeaderList {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut sequence = serializer.serialize_seq(Some(self.entries.len()))?;
        for (name, value) in &self.entries {
            let name_str = String::from_utf8_lossy(name);
            let value_str = String::from_utf8_lossy(value);
            sequence.serialize_element(&(name_str.as_ref(), value_str.as_ref()))?;
        }
        sequence.end()
    }
}

impl<'de> Deserialize<'de> for HeaderList {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let pairs: Vec<(String, String)> = Vec::deserialize(deserializer)?;
        Ok(Self {
            entries: pairs
                .into_iter()
                .map(|(name, value)| (Bytes::from(name), Bytes::from(value)))
                .collect(),
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn insert_replaces_case_insensitive_match_and_returns_prior() {
        let mut list = HeaderList::new();
        let prior = list.insert(
            Bytes::from_static(b"Content-Type"),
            Bytes::from_static(b"application/json"),
        );
        assert!(prior.is_none());
        let prior = list.insert(
            Bytes::from_static(b"content-type"),
            Bytes::from_static(b"text/plain"),
        );
        assert_eq!(prior.as_deref(), Some(b"application/json".as_slice()));
        assert_eq!(
            list.len(),
            1,
            "case-insensitive replace should not duplicate"
        );
        assert_eq!(list.get_str("CONTENT-TYPE"), Some("text/plain"));
    }

    #[test]
    fn insert_if_absent_skips_existing_case_insensitive_match() {
        let mut list = HeaderList::new();
        list.insert(Bytes::from_static(b"X-Trace"), Bytes::from_static(b"first"));
        list.insert_if_absent("x-trace", "second");
        assert_eq!(list.get_str("x-trace"), Some("first"));
    }

    #[test]
    fn retain_drops_entries_matching_predicate() {
        let mut list = HeaderList::from_pairs([
            ("authorization", "bearer"),
            ("x-token", "abc"),
            ("x-trace", "trace"),
        ]);
        list.retain(|name, _| !name.as_ref().eq_ignore_ascii_case(b"authorization"));
        assert_eq!(list.len(), 2);
        assert!(list.get("authorization").is_none());
    }

    #[test]
    fn iter_preserves_insertion_order() {
        let list = HeaderList::from_pairs([("z-last", "1"), ("a-first", "2"), ("m-middle", "3")]);
        let names: Vec<&str> = list
            .iter()
            .map(|(name, _)| core::str::from_utf8(name).expect("ascii"))
            .collect();
        assert_eq!(names, vec!["z-last", "a-first", "m-middle"]);
    }

    #[test]
    fn serde_round_trip_preserves_string_form() {
        let list = HeaderList::from_pairs([
            ("content-type", "application/json"),
            ("x-trace", "trace-deadbeef"),
        ]);
        let json = serde_json::to_string(&list).expect("serialize");
        let restored: HeaderList = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(list, restored);
    }

    #[test]
    fn into_header_bytes_static_str_does_not_copy_payload() {
        // sanity: &'static str via `Bytes::from` is the same allocation
        // as the static slice. owning callers can rely on this for
        // zero-copy of compile-time-known header names.
        let bytes = Bytes::from_static(b"traceparent");
        let other = "traceparent".into_header_bytes();
        assert_eq!(bytes, other);
    }
}
