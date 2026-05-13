//! `Env` — first-class environment-map value type.
//!
//! # Where it composes
//!
//! - [`Command::env_from`](super::command::Command::env_from) —
//!   bulk-set on a std-tier `Command`, replacing any prior
//!   entries. Use after composing the env elsewhere.
//! - [`Command::env_snapshot`](super::command::Command::env_snapshot)
//!   — read the current env back out as `Env` for inspection.
//! - [`CommandConfig`](super::command_config::CommandConfig)
//!   carries `env: Env` directly; serde TOML / JSON round-trips
//!   through the field.
//!
//! # When to use Env vs the std-shape methods
//!
//! - Use [`Command::env(k, v)`](super::command::Command::env) /
//!   [`envs`](super::command::Command::envs) /
//!   [`env_remove`](super::command::Command::env_remove) for
//!   in-flight tweaks (drop-in compat with
//!   `std::process::Command`).
//! - Use `Env` when the env is built somewhere else (config
//!   loader, parent-of-child registry, capability-scoped filter)
//!   and you want to hand it through layers as a value type
//!   instead of replaying a series of incremental updates.
//!
//! # Storage
//!
//! `BTreeMap<String, String>` — deterministic iteration
//! (TOML / JSON serialisation has a stable order), no duplicate
//! keys (last-write-wins via `insert`), trivial `serde` shape
//! (transparent map).
//!
//! # Tier
//!
//! `no_std + alloc` — the type compiles in alloc-only builds.
//! The `From<HashMap<_, _>>` ergonomic gates behind
//! `feature = "std"` because `HashMap` is std-tier.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};

use serde::{Deserialize, Serialize};

/// First-class environment-variable map. Key + value are
/// `String` for serialisability (TOML / JSON / env-loader
/// friendly). Code paths that need OsString fidelity should
/// reach for [`Command::env`](super::command::Command::env)
/// directly with `AsRef<OsStr>` arguments.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Env {
    entries: BTreeMap<String, String>,
}

impl Env {
    /// Empty environment.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or overwrite `key = value`. Returns the previous
    /// value, if any.
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) -> Option<String> {
        self.entries.insert(key.into(), value.into())
    }

    /// Remove `key`, returning its value if present.
    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.entries.remove(key)
    }

    /// Borrow the value for `key`, if set.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries.get(key).map(String::as_str)
    }

    /// Whether `key` has a value set.
    #[must_use]
    pub fn contains_key(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the env carries any entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Iterator over `(key, value)` pairs in deterministic
    /// (key-sorted) order.
    pub fn iter(&self) -> alloc::collections::btree_map::Iter<'_, String, String> {
        self.entries.iter()
    }
}

impl<K, V> FromIterator<(K, V)> for Env
where
    K: Into<String>,
    V: Into<String>,
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut env = Env::new();
        for (key, value) in iter {
            env.insert(key, value);
        }
        env
    }
}

impl<K, V> From<alloc::vec::Vec<(K, V)>> for Env
where
    K: Into<String>,
    V: Into<String>,
{
    fn from(pairs: alloc::vec::Vec<(K, V)>) -> Self {
        Self::from_iter(pairs)
    }
}

impl<K, V> From<BTreeMap<K, V>> for Env
where
    K: ToString,
    V: ToString,
{
    fn from(map: BTreeMap<K, V>) -> Self {
        let entries = map
            .into_iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect();
        Self { entries }
    }
}

#[cfg(feature = "std")]
impl<K, V> From<std::collections::HashMap<K, V>> for Env
where
    K: ToString,
    V: ToString,
{
    fn from(map: std::collections::HashMap<K, V>) -> Self {
        let entries = map
            .into_iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect();
        Self { entries }
    }
}

impl<'a> IntoIterator for &'a Env {
    type Item = (&'a String, &'a String);
    type IntoIter = alloc::collections::btree_map::Iter<'a, String, String>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default,
        clippy::type_complexity,
        clippy::useless_vec,
        clippy::needless_range_loop,
        clippy::default_constructed_unit_structs
    )]

    use super::*;

    #[test]
    fn new_is_empty() {
        let env = Env::new();
        assert!(env.is_empty());
        assert_eq!(env.len(), 0);
    }

    #[test]
    fn insert_overwrites_existing_key() {
        let mut env = Env::new();
        env.insert("LANG", "C");
        let previous = env.insert("LANG", "en_US.UTF-8");
        assert_eq!(previous.as_deref(), Some("C"));
        assert_eq!(env.get("LANG"), Some("en_US.UTF-8"));
        assert_eq!(env.len(), 1);
    }

    #[test]
    fn remove_returns_previous_value() {
        let mut env = Env::new();
        env.insert("LANG", "C");
        env.insert("PATH", "/usr/bin");
        let removed = env.remove("LANG");
        assert_eq!(removed.as_deref(), Some("C"));
        assert!(!env.contains_key("LANG"));
        assert!(env.contains_key("PATH"));
    }

    #[test]
    fn from_iter_builds_from_pairs() {
        let env: Env = [("A", "1"), ("B", "2")].into_iter().collect();
        assert_eq!(env.get("A"), Some("1"));
        assert_eq!(env.get("B"), Some("2"));
    }

    #[cfg(feature = "std")]
    #[test]
    fn from_hashmap_builds_from_map() {
        let mut map = std::collections::HashMap::new();
        map.insert("A", "1");
        map.insert("B", "2");
        let env: Env = map.into();
        assert_eq!(env.len(), 2);
    }

    #[test]
    fn serde_round_trips_via_json() {
        let env: Env = [("LANG", "C"), ("PATH", "/usr/bin")].into_iter().collect();
        let json = serde_json::to_string(&env).expect("serialise");
        let restored: Env = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(restored, env);
    }

    #[test]
    fn iter_order_is_deterministic() {
        let env: Env = [("C", "3"), ("A", "1"), ("B", "2")].into_iter().collect();
        let keys: alloc::vec::Vec<&str> = env.iter().map(|(key, _)| key.as_str()).collect();
        assert_eq!(keys, ["A", "B", "C"]);
    }
}
