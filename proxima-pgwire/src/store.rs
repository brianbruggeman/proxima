//! Per-connection prepared-statement and portal stores.
//!
//! A connection is single-threaded, so the store needs no locks; a
//! linear scan over a small bounded slot vector beats a hash map at the
//! statement counts real clients hold (sqlx and tokio-postgres cache
//! tens of statements, not thousands). The capacity cap is the
//! protection against a peer that prepares in a loop.

use proxima_protocols::pgwire_codec::{FormatCode, Oid};

use crate::pipe_contract::{ColumnDesc, SqlValue};

/// One bound parameter value carried by a portal: raw wire bytes plus the
/// format code they were sent in. The driver decodes them to
/// `SqlValue` at Execute time before handing them to the engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundParameter {
    /// raw wire bytes; `None` = SQL NULL
    pub value: Option<Vec<u8>>,
    pub format: FormatCode,
}

/// A statement the connection has Parsed. The engine reported its
/// parameter types and result columns at PARSE; the driver caches them
/// for Bind validation, Describe answers, and Execute encoding.
#[derive(Debug, Clone)]
pub struct PreparedStatement {
    pub sql: String,
    pub parameter_types: Vec<Oid>,
    pub columns: Vec<ColumnDesc>,
    /// empty query string — Execute answers EmptyQueryResponse
    pub is_empty_query: bool,
}

/// Rows the engine returned that an Execute did not drain because
/// `max_rows` capped the batch. A later Execute on the same portal resumes
/// from `cursor` without re-calling the engine (extended-query portal
/// suspension, PostgreSQL's `Execute` with a row limit).
#[derive(Debug, Clone)]
pub struct PendingRows {
    pub columns: Vec<ColumnDesc>,
    pub rows: Vec<Vec<SqlValue>>,
    /// next unsent row index
    pub cursor: usize,
    pub result_formats: Vec<FormatCode>,
    /// rows already streamed across earlier batches; the final
    /// CommandComplete reports the running total, not just the last batch
    pub emitted: u64,
    /// engine's explicit completion tag, if any (overrides `SELECT n`)
    pub command_tag: Option<String>,
}

/// A portal created by Bind, pointing at its source statement.
#[derive(Debug, Clone)]
pub struct Portal {
    pub statement_name: String,
    pub parameters: Vec<BoundParameter>,
    pub result_formats: Vec<FormatCode>,
    /// rows buffered by a suspended Execute; `None` until a batch is
    /// capped, cleared once the portal drains, on Close, and on re-bind
    pub pending: Option<PendingRows>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreError {
    /// a non-empty name was inserted twice without an intervening Close
    Duplicate,
    /// the configured slot cap is exhausted
    Full,
}

/// Bounded name → value slots. The unnamed entry (empty name) follows
/// the protocol's replace-on-write rule; named entries are
/// insert-once-until-closed.
#[derive(Debug)]
pub struct NamedSlots<T> {
    slots: Vec<(String, T)>,
    capacity: usize,
}

impl<T> NamedSlots<T> {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: Vec::new(),
            capacity,
        }
    }

    /// Inserts under the protocol naming rules.
    ///
    /// # Errors
    /// [`StoreError::Duplicate`] for a live non-empty name,
    /// [`StoreError::Full`] at the slot cap.
    pub fn insert(&mut self, name: &str, value: T) -> Result<(), StoreError> {
        if name.is_empty() {
            if let Some(slot) = self
                .slots
                .iter_mut()
                .find(|(slot_name, _)| slot_name.is_empty())
            {
                slot.1 = value;
                return Ok(());
            }
        } else if self.get(name).is_some() {
            return Err(StoreError::Duplicate);
        }
        if self.slots.len() >= self.capacity {
            return Err(StoreError::Full);
        }
        self.slots.push((name.to_string(), value));
        Ok(())
    }

    #[must_use]
    pub fn get(&self, name: &str) -> Option<&T> {
        self.slots
            .iter()
            .find(|(slot_name, _)| slot_name == name)
            .map(|(_, value)| value)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut T> {
        self.slots
            .iter_mut()
            .find(|(slot_name, _)| slot_name == name)
            .map(|(_, value)| value)
    }

    /// Removes by name; absent names are fine (Close on a missing portal
    /// is not a protocol error).
    pub fn remove(&mut self, name: &str) -> bool {
        let before = self.slots.len();
        self.slots.retain(|(slot_name, _)| slot_name != name);
        self.slots.len() != before
    }

    /// Removes every entry matching the predicate (closing a statement
    /// closes the portals constructed from it).
    pub fn remove_where(&mut self, mut predicate: impl FnMut(&T) -> bool) {
        self.slots.retain(|(_, value)| !predicate(value));
    }

    pub fn clear(&mut self) {
        self.slots.clear();
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    fn slots(capacity: usize) -> NamedSlots<String> {
        NamedSlots::new(capacity)
    }

    #[test]
    fn unnamed_insert_replaces_existing_unnamed() {
        let mut store = slots(4);
        store
            .insert("", "first".into())
            .expect("first unnamed must insert");
        store
            .insert("", "second".into())
            .expect("second unnamed must replace");

        assert_eq!(store.get(""), Some(&"second".to_string()));
        assert_eq!(store.len(), 1, "replace must not grow the slot count");
    }

    #[test]
    fn named_duplicate_returns_duplicate_error() {
        let mut store = slots(4);
        store
            .insert("stmt1", "select 1".into())
            .expect("first named must insert");

        let result = store.insert("stmt1", "select 2".into());

        assert_eq!(result, Err(StoreError::Duplicate));
    }

    #[test]
    fn capacity_cap_returns_full_error() {
        let mut store = slots(2);
        store
            .insert("a", "select 1".into())
            .expect("slot a must insert");
        store
            .insert("b", "select 2".into())
            .expect("slot b must insert");

        let result = store.insert("c", "select 3".into());

        assert_eq!(result, Err(StoreError::Full));
    }

    #[test]
    fn unnamed_replace_does_not_count_toward_capacity() {
        let mut store = slots(1);
        store
            .insert("", "first".into())
            .expect("unnamed insert must succeed");
        let result = store.insert("", "second".into());

        assert_eq!(result, Ok(()));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn remove_present_name_returns_true() {
        let mut store = slots(4);
        store
            .insert("stmt1", "select 1".into())
            .expect("insert must succeed");

        assert!(
            store.remove("stmt1"),
            "remove of present name must return true"
        );
        assert_eq!(store.get("stmt1"), None);
    }

    #[test]
    fn remove_absent_name_returns_false() {
        let mut store: NamedSlots<String> = slots(4);

        assert!(
            !store.remove("nonexistent"),
            "remove of absent name must return false"
        );
    }

    #[test]
    fn remove_where_removes_portals_matching_statement_name() {
        let mut store: NamedSlots<Portal> = NamedSlots::new(8);
        store
            .insert(
                "p1",
                Portal {
                    statement_name: "select_users".into(),
                    parameters: vec![],
                    result_formats: vec![],
                    pending: None,
                },
            )
            .expect("p1 insert must succeed");
        store
            .insert(
                "p2",
                Portal {
                    statement_name: "select_orders".into(),
                    parameters: vec![],
                    result_formats: vec![],
                    pending: None,
                },
            )
            .expect("p2 insert must succeed");
        store
            .insert(
                "p3",
                Portal {
                    statement_name: "select_users".into(),
                    parameters: vec![],
                    result_formats: vec![],
                    pending: None,
                },
            )
            .expect("p3 insert must succeed");

        store.remove_where(|portal| portal.statement_name == "select_users");

        assert_eq!(store.len(), 1);
        assert!(store.get("p1").is_none());
        assert!(store.get("p2").is_some());
        assert!(store.get("p3").is_none());
    }

    #[test]
    fn clear_empties_all_slots() {
        let mut store = slots(4);
        store.insert("a", "v1".into()).expect("insert a");
        store.insert("b", "v2".into()).expect("insert b");
        store.insert("", "unnamed".into()).expect("insert unnamed");

        store.clear();

        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn get_on_missing_name_is_none() {
        let store: NamedSlots<String> = slots(4);

        assert_eq!(store.get("missing"), None);
    }

    #[test]
    fn get_returns_correct_value_by_name() {
        let mut store = slots(4);
        store
            .insert("stmt_a", "select id from users".into())
            .expect("insert");

        assert_eq!(
            store.get("stmt_a"),
            Some(&"select id from users".to_string())
        );
    }

    #[test]
    fn empty_store_is_empty_and_len_zero() {
        let store: NamedSlots<i32> = NamedSlots::new(10);

        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn zero_capacity_full_on_first_named_insert() {
        let mut store: NamedSlots<i32> = NamedSlots::new(0);

        assert_eq!(store.insert("x", 1), Err(StoreError::Full));
    }
}
