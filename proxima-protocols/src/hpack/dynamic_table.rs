//! HPACK dynamic table (RFC 7541 §2.3.2 + §4).
//!
//! A bounded LIFO of `(name, value)` pairs maintained in lockstep by
//! the encoder and decoder on a single HTTP/2 connection. New
//! entries push to the front (low dynamic index); old entries
//! evict off the back when an insertion would otherwise exceed
//! `max_size`.
//!
//! ## Sizing (RFC §4.1)
//!
//! Each entry costs `name.len() + value.len() + 32` bytes regardless
//! of huffman encoding on the wire. The `+ 32` is fixed overhead the
//! RFC requires for internal accounting (covers refs / metadata
//! across implementations).
//!
//! ## Index semantics
//!
//! HPACK indices are 1-based and partitioned:
//! - 1..=61 → static table
//! - 62..=(61 + dynamic_len) → dynamic table, with dynamic index 1
//!   (absolute 62) being the most recently inserted entry.
//!
//! This module deals in *dynamic indices* (1-based, relative to the
//! dynamic table). Callers translate to absolute HPACK indices by
//! adding [`STATIC_TABLE_LAST_INDEX`].
//!
//! ## Zero-copy
//!
//! Names and values are `Bytes`. The decoder builds them from the
//! wire and either keeps the `Bytes::clone` (a cheap Arc bump) or
//! owns them outright depending on encoding (huffman → owned;
//! raw → shared with the source buffer). Either way, no copies
//! when the table is queried.

use alloc::collections::VecDeque;

use bytes::Bytes;

/// Last index of the HPACK static table; dynamic indices start at
/// `STATIC_TABLE_LAST_INDEX + 1`.
pub const STATIC_TABLE_LAST_INDEX: usize = 61;

/// Per-entry overhead in bytes from RFC 7541 §4.1.
pub const ENTRY_OVERHEAD: usize = 32;

/// One `(name, value)` pair in the dynamic table.
#[derive(Debug, Clone)]
pub struct DynamicEntry {
    /// Header name (lowercased ASCII per HTTP/2 §8.1.2).
    pub name: Bytes,
    /// Header value as received on the wire.
    pub value: Bytes,
}

impl DynamicEntry {
    /// Construct an entry from `Bytes` halves.
    #[must_use]
    pub fn new(name: Bytes, value: Bytes) -> Self {
        Self { name, value }
    }

    /// Size attributable to this entry per RFC §4.1.
    #[must_use]
    pub fn size(&self) -> usize {
        self.name.len() + self.value.len() + ENTRY_OVERHEAD
    }
}

/// Bounded LIFO dynamic table. Not `Send` is fine — a connection's
/// HPACK state lives on its own task.
#[derive(Debug)]
pub struct DynamicTable {
    entries: VecDeque<DynamicEntry>,
    current_size: usize,
    max_size: usize,
}

impl DynamicTable {
    /// New empty table with the given `SETTINGS_HEADER_TABLE_SIZE`
    /// (RFC 7540 §6.5.2). RFC default is 4096.
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            current_size: 0,
            max_size,
        }
    }

    /// Insert an entry at the front. Evicts oldest entries as
    /// needed; if the entry alone exceeds `max_size`, clears the
    /// table and inserts nothing (RFC §4.4 — "an attempt to add an
    /// entry larger than the maximum size causes the table to be
    /// emptied of all existing entries, and results in an empty
    /// table").
    pub fn insert(&mut self, entry: DynamicEntry) {
        let entry_size = entry.size();
        if entry_size > self.max_size {
            self.entries.clear();
            self.current_size = 0;
            return;
        }
        while self.current_size + entry_size > self.max_size {
            let Some(evicted) = self.entries.pop_back() else {
                break;
            };
            self.current_size -= evicted.size();
        }
        self.current_size += entry_size;
        self.entries.push_front(entry);
    }

    /// Get an entry by 1-based dynamic index. Index 1 = most recent.
    #[must_use]
    pub fn get(&self, dynamic_index: usize) -> Option<&DynamicEntry> {
        if dynamic_index == 0 {
            return None;
        }
        self.entries.get(dynamic_index - 1)
    }

    /// Reverse lookup for the encoder. Returns `Some((dynamic_index,
    /// value_matched))` if `name` is in the table. `value_matched`
    /// is `true` iff a name AND value match was found; otherwise
    /// the returned index is the most recent name-only match.
    ///
    /// Linear scan — typical dynamic tables hold tens of entries,
    /// not thousands, so a hash sidecar would lose to scan-from-
    /// front in cache locality. Re-bench if table sizes ever grow.
    #[must_use]
    pub fn lookup(&self, name: &[u8], value: &[u8]) -> Option<(usize, bool)> {
        let mut name_hit: Option<usize> = None;
        for (offset, entry) in self.entries.iter().enumerate() {
            if entry.name.as_ref() == name {
                if entry.value.as_ref() == value {
                    return Some((offset + 1, true));
                }
                if name_hit.is_none() {
                    name_hit = Some(offset + 1);
                }
            }
        }
        name_hit.map(|index| (index, false))
    }

    /// Name-only lookup. Returns the most-recent dynamic index for
    /// `name` if present.
    #[must_use]
    pub fn lookup_name(&self, name: &[u8]) -> Option<usize> {
        self.entries
            .iter()
            .position(|entry| entry.name.as_ref() == name)
            .map(|offset| offset + 1)
    }

    /// Update `max_size`, evicting from the back until the new cap
    /// is satisfied. Used to honor a SETTINGS_HEADER_TABLE_SIZE
    /// change (RFC 7541 §4.2).
    pub fn set_max_size(&mut self, new_max: usize) {
        while self.current_size > new_max {
            let Some(evicted) = self.entries.pop_back() else {
                break;
            };
            self.current_size -= evicted.size();
        }
        self.max_size = new_max;
    }

    /// Number of entries currently held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` if no entries are held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Current size in bytes (sum of entry sizes).
    #[must_use]
    pub fn size(&self) -> usize {
        self.current_size
    }

    /// Configured maximum size.
    #[must_use]
    pub fn max_size(&self) -> usize {
        self.max_size
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use alloc::vec;

    fn entry(name: &'static [u8], value: &'static [u8]) -> DynamicEntry {
        DynamicEntry::new(Bytes::from_static(name), Bytes::from_static(value))
    }

    #[test]
    fn empty_table_get_returns_none() {
        let table = DynamicTable::new(4096);
        assert!(table.get(1).is_none());
        assert_eq!(table.len(), 0);
        assert_eq!(table.size(), 0);
    }

    #[test]
    fn single_insert_lookup() {
        let mut table = DynamicTable::new(4096);
        table.insert(entry(b"x-foo", b"bar"));
        assert_eq!(table.len(), 1);
        assert_eq!(table.size(), 5 + 3 + ENTRY_OVERHEAD);
        let stored = table.get(1).expect("entry");
        assert_eq!(stored.name.as_ref(), b"x-foo");
        assert_eq!(stored.value.as_ref(), b"bar");
        assert!(table.get(2).is_none());
        assert!(table.get(0).is_none());
    }

    #[test]
    fn newest_at_index_one_lifo_order() {
        let mut table = DynamicTable::new(4096);
        table.insert(entry(b"x-a", b"1"));
        table.insert(entry(b"x-b", b"2"));
        table.insert(entry(b"x-c", b"3"));
        assert_eq!(table.get(1).unwrap().name.as_ref(), b"x-c");
        assert_eq!(table.get(2).unwrap().name.as_ref(), b"x-b");
        assert_eq!(table.get(3).unwrap().name.as_ref(), b"x-a");
    }

    #[test]
    fn eviction_when_exceeding_max() {
        let entry_size = 3 + 1 + ENTRY_OVERHEAD;
        let mut table = DynamicTable::new(entry_size * 2);
        table.insert(entry(b"x-a", b"1"));
        table.insert(entry(b"x-b", b"2"));
        table.insert(entry(b"x-c", b"3"));
        assert_eq!(table.len(), 2);
        assert_eq!(table.get(1).unwrap().name.as_ref(), b"x-c");
        assert_eq!(table.get(2).unwrap().name.as_ref(), b"x-b");
    }

    #[test]
    fn oversized_entry_clears_table() {
        let mut table = DynamicTable::new(64);
        table.insert(entry(b"x-a", b"keep"));
        let big_value = Bytes::from(vec![b'X'; 200]);
        table.insert(DynamicEntry::new(Bytes::from_static(b"big"), big_value));
        assert_eq!(table.len(), 0);
        assert_eq!(table.size(), 0);
    }

    #[test]
    fn set_max_size_evicts() {
        let mut table = DynamicTable::new(4096);
        for byte in *b"abcd" {
            table.insert(DynamicEntry::new(
                Bytes::from(vec![byte; 4]),
                Bytes::from_static(b"v"),
            ));
        }
        assert_eq!(table.len(), 4);
        table.set_max_size(2 * (4 + 1 + ENTRY_OVERHEAD));
        assert_eq!(table.len(), 2);
        assert_eq!(table.get(1).unwrap().name.as_ref(), b"dddd");
        assert_eq!(table.get(2).unwrap().name.as_ref(), b"cccc");
    }

    #[test]
    fn set_max_size_zero_clears() {
        let mut table = DynamicTable::new(4096);
        table.insert(entry(b"x-a", b"1"));
        table.insert(entry(b"x-b", b"2"));
        table.set_max_size(0);
        assert_eq!(table.len(), 0);
        assert_eq!(table.size(), 0);
        assert_eq!(table.max_size(), 0);
    }

    #[test]
    fn lookup_finds_name_value_match() {
        let mut table = DynamicTable::new(4096);
        table.insert(entry(b"x-foo", b"old"));
        table.insert(entry(b"x-foo", b"new"));
        let (index, matched) = table.lookup(b"x-foo", b"new").expect("match");
        assert_eq!(index, 1);
        assert!(matched);
    }

    #[test]
    fn lookup_falls_back_to_name_only() {
        let mut table = DynamicTable::new(4096);
        table.insert(entry(b"x-foo", b"old"));
        table.insert(entry(b"x-foo", b"new"));
        let (index, matched) = table.lookup(b"x-foo", b"absent").expect("name hit");
        assert_eq!(index, 1, "most-recent name hit");
        assert!(!matched);
    }

    #[test]
    fn lookup_returns_none_for_unknown_name() {
        let mut table = DynamicTable::new(4096);
        table.insert(entry(b"x-foo", b"bar"));
        assert!(table.lookup(b"x-baz", b"qux").is_none());
        assert!(table.lookup_name(b"x-baz").is_none());
    }

    #[test]
    fn lookup_name_returns_most_recent() {
        let mut table = DynamicTable::new(4096);
        table.insert(entry(b"x-foo", b"old"));
        table.insert(entry(b"x-bar", b"between"));
        table.insert(entry(b"x-foo", b"new"));
        assert_eq!(table.lookup_name(b"x-foo"), Some(1));
    }
}
