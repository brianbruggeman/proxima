//! [`FilterRegistry`] — a named store of live [`IdSet`] subscriptions.
//!
//! The pub/sub view of [`live_filter`](crate::pipe::live_filter): many named
//! subscriptions, each an id-set filter you *subscribe by name* and *reconfigure
//! by name*. It dogfoods [`proxima_core::live::Live`] — the subscription map is
//! itself a live cell, so lookups on the data path are wait-free and
//! subscribe/unsubscribe copy-on-write it. That is the same arc-swap CoW shape
//! as [`proxima_core::FactoryRegistry`], but storing *live handles* rather than
//! factories.
//!
//! ## Config-as-composition
//!
//! [`FilterRegistryConfig`] is the config mirror: a map of subscription name →
//! its id list. A new subscription is a new config entry — config, not a
//! recompile. [`FilterRegistry::from_config`] builds every subscription;
//! [`FilterRegistry::to_config`] mirrors the current (live-mutated) state back.
//! The fluent [`FilterRegistryConfig::subscribe`] and the serde/`conflaguration`
//! surfaces are interchangeable, per the workspace config principle. The layered
//! `conflaguration` file/env loader lives behind the default-off `config`
//! feature (the composition boundary).

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec::Vec;

use proxima_core::live::{Live, LiveControl, live};
use serde::{Deserialize, Serialize};

use crate::pipe::live_filter::{FilterControl, IdSet, LiveFilter, live_filter_ids};

/// One named subscription's two live halves.
struct Subscription<Id: Ord> {
    filter: LiveFilter<IdSet<Id>>,
    control: FilterControl<IdSet<Id>>,
}

impl<Id: Ord> Clone for Subscription<Id> {
    fn clone(&self) -> Self {
        Self {
            filter: self.filter.clone(),
            control: self.control.clone(),
        }
    }
}

type Subscriptions<Id> = BTreeMap<String, Subscription<Id>>;

/// A lock-free registry of named live id-subscriptions. Lookups are wait-free
/// (the map is a [`Live`] cell); subscribe/unsubscribe copy-on-write it.
pub struct FilterRegistry<Id: Ord> {
    read: Live<Subscriptions<Id>>,
    write: LiveControl<Subscriptions<Id>>,
}

impl<Id: Ord + Clone> FilterRegistry<Id> {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        let (read, write) = live(BTreeMap::new());
        Self { read, write }
    }

    /// Create a live subscription under `name` seeded from `ids`, returning its
    /// filter (read) and control (write) halves. Replaces an existing `name`.
    pub fn subscribe(
        &self,
        name: impl Into<String>,
        ids: impl IntoIterator<Item = Id>,
    ) -> (LiveFilter<IdSet<Id>>, FilterControl<IdSet<Id>>) {
        let name = name.into();
        let (filter, control) = live_filter_ids(ids);
        let entry = Subscription {
            filter: filter.clone(),
            control: control.clone(),
        };
        self.write.update(|current| {
            let mut next = current.clone();
            next.insert(name.clone(), entry.clone());
            next
        });
        (filter, control)
    }

    /// Remove a named subscription; returns whether it existed.
    pub fn unsubscribe(&self, name: &str) -> bool {
        let existed = self.read.read(|subs| subs.contains_key(name));
        if existed {
            self.write.update(|current| {
                let mut next = current.clone();
                next.remove(name);
                next
            });
        }
        existed
    }

    /// The filter (read) half of a named subscription, for the data path.
    #[must_use]
    pub fn filter(&self, name: &str) -> Option<LiveFilter<IdSet<Id>>> {
        self.read
            .read(|subs| subs.get(name).map(|entry| entry.filter.clone()))
    }

    /// The control (write) half of a named subscription, for the control plane.
    #[must_use]
    pub fn control(&self, name: &str) -> Option<FilterControl<IdSet<Id>>> {
        self.read
            .read(|subs| subs.get(name).map(|entry| entry.control.clone()))
    }

    /// The registered subscription names, sorted.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.read.read(|subs| subs.keys().cloned().collect())
    }

    /// The names of every subscription whose filter currently matches `id` — the
    /// pub/sub dispatch query. O(subscriptions); a broker fans an event to these.
    #[must_use]
    pub fn matches(&self, id: &Id) -> Vec<String> {
        self.read.read(|subs| {
            subs.iter()
                .filter(|(_, entry)| entry.filter.contains(id))
                .map(|(name, _)| name.clone())
                .collect()
        })
    }

    /// Build a registry from its config — one subscription per entry.
    #[must_use]
    pub fn from_config(config: &FilterRegistryConfig<Id>) -> Self {
        let registry = Self::new();
        for (name, ids) in &config.subscriptions {
            registry.subscribe(name.clone(), ids.iter().cloned());
        }
        registry
    }

    /// Mirror the current (live-mutated) registry state back to config.
    #[must_use]
    pub fn to_config(&self) -> FilterRegistryConfig<Id> {
        self.read.read(|subs| {
            let subscriptions = subs
                .iter()
                .map(|(name, entry)| {
                    let ids = entry.filter.snapshot().iter().cloned().collect();
                    (name.clone(), ids)
                })
                .collect();
            FilterRegistryConfig { subscriptions }
        })
    }
}

impl<Id: Ord + Clone> Default for FilterRegistry<Id> {
    fn default() -> Self {
        Self::new()
    }
}

/// The config mirror of a [`FilterRegistry`]: a map of subscription name → its
/// id set. Serde-transparent, so it round-trips as `{ "sub": ["id", ...] }`.
/// The ids are a set (order-independent), matching the [`IdSet`] they seed.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FilterRegistryConfig<Id: Ord> {
    subscriptions: BTreeMap<String, BTreeSet<Id>>,
}

impl<Id: Ord> FilterRegistryConfig<Id> {
    /// An empty config.
    #[must_use]
    pub fn new() -> Self {
        Self {
            subscriptions: BTreeMap::new(),
        }
    }

    /// Fluently add a subscription — the config builder surface.
    #[must_use]
    pub fn subscribe(mut self, name: impl Into<String>, ids: impl IntoIterator<Item = Id>) -> Self {
        self.subscriptions
            .insert(name.into(), ids.into_iter().collect());
        self
    }
}

#[cfg(feature = "config")]
impl<Id: Ord> conflaguration::Validate for FilterRegistryConfig<Id> {
    fn validate(&self) -> conflaguration::Result<()> {
        let errors: Vec<conflaguration::ValidationMessage> = self
            .subscriptions
            .keys()
            .filter(|name| name.trim().is_empty())
            .map(|_| {
                conflaguration::ValidationMessage::new(
                    "name",
                    "subscription name must be non-empty",
                )
            })
            .collect();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

#[cfg(feature = "config")]
impl<Id: Ord + Serialize + serde::de::DeserializeOwned> FilterRegistryConfig<Id> {
    /// Load through `conflaguration`'s layered loader — the TOML/JSON `path`,
    /// then `validate`. The canonical config-surface entry point.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, proxima_core::ProximaError> {
        conflaguration::ConfigBuilder::<Self>::new()
            .file(path)
            .validate()
            .build()
            .map_err(|err| {
                proxima_core::ProximaError::Registry(format!(
                    "filter registry config load failed: {err}"
                ))
            })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn ids() -> (String, String, String) {
        (
            "req-7f3a9c2e1b4d".to_string(),
            "req-0f43d9c0aa18".to_string(),
            "req-c06db8bb5521".to_string(),
        )
    }

    #[test]
    fn subscribe_then_look_up_by_name() {
        let (first, other, _) = ids();
        let registry: FilterRegistry<String> = FilterRegistry::new();
        registry.subscribe("orders", [first.clone()]);
        let filter = registry.filter("orders").expect("subscription exists");
        assert!(filter.contains(&first));
        assert!(!filter.contains(&other));
        assert!(registry.filter("absent").is_none());
    }

    #[test]
    fn control_by_name_reconfigures_the_named_subscription() {
        let (first, second, _) = ids();
        let registry: FilterRegistry<String> = FilterRegistry::new();
        let (filter, _control) = registry.subscribe("orders", [first.clone()]);
        // a different holder reaches the same subscription's control by name.
        registry
            .control("orders")
            .expect("control")
            .add(second.clone());
        assert!(
            filter.contains(&second),
            "the data-path filter sees the control-plane add"
        );
    }

    #[test]
    fn unsubscribe_removes_a_subscription() {
        let (first, _, _) = ids();
        let registry: FilterRegistry<String> = FilterRegistry::new();
        registry.subscribe("orders", [first]);
        assert!(registry.unsubscribe("orders"));
        assert!(
            !registry.unsubscribe("orders"),
            "second removal reports absent"
        );
        assert!(registry.filter("orders").is_none());
    }

    #[test]
    fn names_lists_every_subscription_sorted() {
        let registry: FilterRegistry<String> = FilterRegistry::new();
        registry.subscribe("beta", ["x".to_string()]);
        registry.subscribe("alpha", ["y".to_string()]);
        assert_eq!(
            registry.names(),
            vec!["alpha".to_string(), "beta".to_string()]
        );
    }

    #[test]
    fn matches_returns_the_subscribers_for_an_id() {
        let (shared, only_a, _) = ids();
        let registry: FilterRegistry<String> = FilterRegistry::new();
        registry.subscribe("a", [shared.clone(), only_a.clone()]);
        registry.subscribe("b", [shared.clone()]);
        assert_eq!(
            registry.matches(&shared),
            vec!["a".to_string(), "b".to_string()]
        );
        assert_eq!(registry.matches(&only_a), vec!["a".to_string()]);
    }

    #[test]
    fn from_config_builds_and_to_config_mirrors() {
        let (first, second, third) = ids();
        let config = FilterRegistryConfig::new()
            .subscribe("orders", [first.clone(), second.clone()])
            .subscribe("audit", [third.clone()]);
        let registry = FilterRegistry::from_config(&config);
        assert!(registry.filter("orders").expect("orders").contains(&first));
        assert!(registry.filter("audit").expect("audit").contains(&third));
        assert_eq!(
            registry.to_config(),
            config,
            "built registry mirrors back to its config"
        );
    }

    #[test]
    fn to_config_reflects_live_mutations() {
        let (first, second, _) = ids();
        let registry: FilterRegistry<String> = FilterRegistry::new();
        registry.subscribe("orders", [first.clone()]);
        registry
            .control("orders")
            .expect("control")
            .add(second.clone());
        let mirrored = registry.to_config();
        let expected = FilterRegistryConfig::new().subscribe("orders", [first, second]);
        assert_eq!(mirrored, expected, "the mirror captures the live add");
    }

    #[test]
    fn config_round_trips_through_serde() {
        let (first, second, _) = ids();
        let config = FilterRegistryConfig::new().subscribe("orders", [first, second]);
        let serialized = serde_json::to_string(&config).unwrap();
        assert!(
            serialized.starts_with('{'),
            "a registry config is a name->ids map"
        );
        let restored: FilterRegistryConfig<String> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(restored, config);
    }

    #[cfg(feature = "config")]
    #[test]
    fn conflaguration_validates_a_seeded_config() {
        use conflaguration::ConfigBuilder;
        let (first, _, _) = ids();
        let config = FilterRegistryConfig::new().subscribe("orders", [first]);
        let built = ConfigBuilder::<FilterRegistryConfig<String>>::new()
            .value(config.clone())
            .validate()
            .build()
            .expect("validated build");
        assert_eq!(
            built, config,
            "value-seeded + validated build returns the config"
        );
    }

    #[cfg(feature = "config")]
    #[test]
    fn conflaguration_rejects_a_blank_name() {
        use conflaguration::Validate;
        let (first, _, _) = ids();
        let config = FilterRegistryConfig::new().subscribe("  ", [first]);
        assert!(
            config.validate().is_err(),
            "a blank subscription name is rejected"
        );
    }
}
