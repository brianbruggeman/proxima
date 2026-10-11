//! Caller-owned storage for the cassette FSM: [`IndexBuilder`] grows while a
//! cassette loads, [`IndexBuilder::seal`] freezes it into a [`CassetteIndex`]
//! sorted by key for allocation-free binary-search lookup.
//!
//! Alloc tier. Kept outside `replay::fsm` so the FSM states stay `Copy` and
//! allocation-free (guiding-principles P3); the FSM mutates the builder through
//! `Loading::on_event` and never owns it.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::replay::fsm::{SegmentId, SegmentStamp};
use crate::replay::keying::{KeyDigest, RequestDigest, ResponseDigest};

/// One recorded exchange. `request` is the collision witness stored beside
/// `key`: a lookup hit whose live request digest differs is a collision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexEntry<R> {
    pub key: KeyDigest,
    pub request: RequestDigest,
    pub segment: SegmentId,
    pub response: ResponseDigest,
    pub payload: R,
}

/// Position of an entry in one [`CassetteIndex`]. The field is private, so the
/// only way to hold one is to have been handed it by the index; a bare
/// `usize` would let any integer name an entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotId(usize);

#[derive(Debug, Clone)]
pub struct IndexBuilder<R> {
    entries: BTreeMap<KeyDigest, IndexEntry<R>>,
    segments: Vec<SegmentStamp>,
}

/// Frozen, sorted by key. The two `Box<[..]>` are the one sanctioned boxing:
/// a sealed index never grows, so the capacity word of a `Vec` is dead weight.
#[derive(Debug, Clone)]
pub struct CassetteIndex<R> {
    entries: alloc::boxed::Box<[IndexEntry<R>]>,
    segments: alloc::boxed::Box<[SegmentStamp]>,
}

impl<R> Default for IndexBuilder<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R> IndexBuilder<R> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            segments: Vec::new(),
        }
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
    pub fn get(&self, key: KeyDigest) -> Option<&IndexEntry<R>> {
        self.entries.get(&key)
    }

    #[must_use]
    pub fn is_declared(&self, segment: SegmentId) -> bool {
        self.segments
            .binary_search_by_key(&segment, |stamp| stamp.segment)
            .is_ok()
    }

    pub(crate) fn declare(&mut self, stamp: SegmentStamp) {
        self.segments.push(stamp);
    }

    pub(crate) fn put(&mut self, entry: IndexEntry<R>) {
        self.entries.insert(entry.key, entry);
    }

    #[must_use]
    pub fn seal(self) -> CassetteIndex<R> {
        CassetteIndex {
            entries: self.entries.into_values().collect(),
            segments: self.segments.into(),
        }
    }
}

impl<R> CassetteIndex<R> {
    #[must_use]
    pub fn empty() -> Self {
        IndexBuilder::new().seal()
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
    pub fn find(&self, key: KeyDigest) -> Option<SlotId> {
        self.entries
            .binary_search_by_key(&key, |entry| entry.key)
            .ok()
            .map(SlotId)
    }

    #[must_use]
    pub fn get(&self, slot: SlotId) -> Option<&IndexEntry<R>> {
        self.entries.get(slot.0)
    }

    #[must_use]
    pub fn segments(&self) -> &[SegmentStamp] {
        &self.segments
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::fsm::{RecordedAt, UnixMillis};
    use crate::replay::keying::{HeaderAllowlist, request_digest, request_key, response_digest};
    use bytes::Bytes;
    use proxima_primitives::pipe::method::Method;
    use proxima_primitives::pipe::request::Request;

    fn forecast(city: &str) -> Request<Bytes> {
        let mut request = Request::builder()
            .method(Method::Get)
            .path("/v1/forecast")
            .build()
            .expect("a method and path are enough to build a request");
        request.query = alloc::vec![(
            Bytes::from_static(b"city"),
            Bytes::copy_from_slice(city.as_bytes())
        )]
        .into();
        request
    }

    fn entry(city: &str, segment: u32, payload: u32) -> IndexEntry<u32> {
        let allowlist = HeaderAllowlist::new(["accept"]).expect("accept is a valid header name");
        let request = forecast(city);
        IndexEntry {
            key: request_key(&request, &allowlist),
            request: request_digest(&request, &allowlist),
            segment: SegmentId::from_raw(segment),
            response: response_digest(200, &[b"{\"temp\":12}"]),
            payload,
        }
    }

    fn sealed(cities: &[&str]) -> CassetteIndex<u32> {
        let mut builder = IndexBuilder::new();
        builder.declare(SegmentStamp {
            segment: SegmentId::from_raw(0),
            recorded_at: RecordedAt::At(UnixMillis::from_raw(1_790_812_800_000)),
        });
        for (position, city) in cities.iter().enumerate() {
            builder.put(entry(
                city,
                0,
                u32::try_from(position).expect("small index"),
            ));
        }
        builder.seal()
    }

    #[test]
    fn request_digest_is_a_different_fold_than_request_key() {
        let allowlist = HeaderAllowlist::new(["accept"]).expect("accept is a valid header name");
        let request = forecast("oslo");
        assert_ne!(
            request_key(&request, &allowlist).get(),
            request_digest(&request, &allowlist).get()
        );
    }

    #[test]
    fn request_digest_changes_with_the_query_value() {
        let allowlist = HeaderAllowlist::new(["accept"]).expect("accept is a valid header name");
        assert_ne!(
            request_digest(&forecast("oslo"), &allowlist),
            request_digest(&forecast("bergen"), &allowlist)
        );
    }

    #[test]
    fn sealed_index_finds_every_inserted_key_and_returns_its_payload() {
        let index = sealed(&["oslo", "bergen", "tromso"]);
        for (position, city) in ["oslo", "bergen", "tromso"].iter().enumerate() {
            let allowlist = HeaderAllowlist::new(["accept"]).expect("valid header name");
            let slot = index
                .find(request_key(&forecast(city), &allowlist))
                .expect("inserted key must be found");
            let found = index.get(slot).expect("a slot from find is in range");
            assert_eq!(found.payload, u32::try_from(position).expect("small index"));
        }
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn sealed_index_misses_an_unrecorded_key() {
        let index = sealed(&["oslo"]);
        let allowlist = HeaderAllowlist::new(["accept"]).expect("valid header name");
        assert_eq!(
            index.find(request_key(&forecast("stavanger"), &allowlist)),
            None
        );
    }

    #[test]
    fn slot_from_a_larger_index_is_none_in_a_smaller_one() {
        let large = sealed(&["oslo", "bergen", "tromso"]);
        let small = sealed(&["oslo"]);
        let allowlist = HeaderAllowlist::new(["accept"]).expect("valid header name");
        let slots: Vec<SlotId> = ["oslo", "bergen", "tromso"]
            .iter()
            .filter_map(|city| large.find(request_key(&forecast(city), &allowlist)))
            .collect();
        let out_of_range = slots
            .iter()
            .filter(|slot| small.get(**slot).is_none())
            .count();
        assert_eq!(out_of_range, 2);
    }

    #[test]
    fn empty_index_has_no_entries_or_segments() {
        let index = CassetteIndex::<u32>::empty();
        assert!(index.is_empty());
        assert_eq!(index.segments().len(), 0);
    }

    #[test]
    fn put_replaces_an_entry_under_the_same_key() {
        let mut builder = IndexBuilder::new();
        builder.put(entry("oslo", 0, 1));
        builder.put(entry("oslo", 0, 2));
        let held = builder.get(entry("oslo", 0, 0).key).expect("key was put");
        assert_eq!((builder.len(), held.payload), (1, 2));
    }

    #[test]
    fn undeclared_segment_is_not_declared() {
        let mut builder = IndexBuilder::<u32>::new();
        assert!(!builder.is_declared(SegmentId::from_raw(0)));
        builder.declare(SegmentStamp {
            segment: SegmentId::from_raw(0),
            recorded_at: RecordedAt::Unknown,
        });
        assert!(builder.is_declared(SegmentId::from_raw(0)));
    }
}
