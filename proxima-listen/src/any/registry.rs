//! [`AnyRegistry`] — a name-keyed table of [`AnyProtocol`] candidates,
//! peer of [`crate::ListenRegistry`] and built the identical way
//! (`ArcSwap<BTreeMap<..>>`, copy-on-write `register`, lock-free `get`).
//!
//! Priority model (owner-settled): default priority is `100`
//! ([`AnyProtocol::priority`]'s default), higher wins, and **candidates may
//! share a priority** — registering two candidates both at `100` (or any
//! other value) is not an error. Only a duplicate NAME is rejected, exactly
//! like [`crate::ListenRegistry::register`]. Same-priority candidates are
//! the common case (every protocol this task ships defaults to `100`) and
//! are resolved by whichever candidate's [`AnyProtocol::probe`] narrows to
//! `Match` first — see [`crate::any::Classifier`]'s docs for the current
//! (provisional) arbitration rule.

use std::collections::BTreeMap;
use std::sync::Arc;

use arc_swap::ArcSwap;

use super::probe::AnyProtocol;
use proxima_core::ProximaError;

pub struct AnyRegistry {
    protocols: ArcSwap<BTreeMap<String, Arc<dyn AnyProtocol>>>,
}

impl Default for AnyRegistry {
    fn default() -> Self {
        Self {
            protocols: ArcSwap::from_pointee(BTreeMap::new()),
        }
    }
}

impl AnyRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert `protocol` under its own [`AnyProtocol::name`]. Errors on a
    /// duplicate NAME only — a duplicate PRIORITY is expected and allowed
    /// (see this module's doc).
    pub fn register(&self, protocol: Arc<dyn AnyProtocol>) -> Result<(), ProximaError> {
        let name = protocol.name().to_string();
        loop {
            let current = self.protocols.load_full();
            if current.contains_key(&name) {
                return Err(ProximaError::Registry(format!(
                    "any-protocol '{name}' already registered"
                )));
            }
            let mut next: BTreeMap<String, Arc<dyn AnyProtocol>> = (*current).clone();
            next.insert(name.clone(), protocol.clone());
            let prev = self.protocols.compare_and_swap(&current, Arc::new(next));
            if Arc::ptr_eq(&prev, &current) {
                return Ok(());
            }
        }
    }

    pub fn get(&self, name: &str) -> Result<Arc<dyn AnyProtocol>, ProximaError> {
        self.protocols
            .load_full()
            .get(name)
            .cloned()
            .ok_or_else(|| ProximaError::Registry(format!("no any-protocol named '{name}'")))
    }

    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.protocols.load_full().keys().cloned().collect()
    }

    /// A snapshot of every registered candidate, ordered by
    /// [`AnyProtocol::priority`] descending (ties broken by name, ascending,
    /// for a deterministic candidate order run to run). This is the
    /// `candidates` slice [`crate::any::Classifier::new`] takes: the sort
    /// happens once here, at snapshot time, rather than per connection, so
    /// the (currently provisional) priority-ordered arbitration in
    /// `Classifier::advance` can rely on `candidates` already being in
    /// priority order.
    #[must_use]
    pub fn snapshot(&self) -> Arc<[Arc<dyn AnyProtocol>]> {
        let table = self.protocols.load_full();
        let mut ordered: Vec<Arc<dyn AnyProtocol>> = table.values().cloned().collect();
        ordered.sort_by(|left, right| {
            right
                .priority()
                .cmp(&left.priority())
                .then_with(|| left.name().cmp(right.name()))
        });
        Arc::from(ordered)
    }

    /// Same as [`Self::snapshot`], restricted to the named candidates —
    /// backs `.accepts(&[...])`/`.accept(name)` on the listener builder.
    /// Errors if any requested name is not registered, so a typo surfaces
    /// at `.serve()` time rather than silently accepting everything.
    pub fn snapshot_named(
        &self,
        names: &[String],
    ) -> Result<Arc<[Arc<dyn AnyProtocol>]>, ProximaError> {
        let table = self.protocols.load_full();
        let mut ordered: Vec<Arc<dyn AnyProtocol>> = Vec::with_capacity(names.len());
        for name in names {
            let protocol = table
                .get(name)
                .cloned()
                .ok_or_else(|| ProximaError::Registry(format!("no any-protocol named '{name}'")))?;
            ordered.push(protocol);
        }
        ordered.sort_by(|left, right| {
            right
                .priority()
                .cmp(&left.priority())
                .then_with(|| left.name().cmp(right.name()))
        });
        Ok(Arc::from(ordered))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::any::probe::AnyHandler;
    use crate::any::probe::ProbeVerdict;
    use proxima_primitives::stream::{PeerInfo, StreamConnection};
    use serde_json::Value;
    use std::future::Future;
    use std::pin::Pin;

    struct StubAny {
        registered_name: String,
        priority: u16,
    }

    impl AnyProtocol for StubAny {
        fn name(&self) -> &str {
            &self.registered_name
        }

        fn priority(&self) -> u16 {
            self.priority
        }

        fn max_prefix_bytes(&self) -> usize {
            8
        }

        fn probe(&self, _prefix: &[u8]) -> ProbeVerdict {
            ProbeVerdict::No
        }

        fn drive<'a>(
            &'a self,
            _stream: Box<dyn StreamConnection>,
            _handler: AnyHandler,
            _spec: &'a Value,
            _peer: Option<PeerInfo>,
        ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'a>> {
            Box::pin(async move { Ok(()) })
        }
    }

    #[test]
    fn register_and_lookup_round_trip() {
        let registry = AnyRegistry::new();
        registry
            .register(Arc::new(StubAny {
                registered_name: "stub".into(),
                priority: 100,
            }))
            .expect("register");
        let protocol = registry.get("stub").expect("get");
        assert_eq!(protocol.name(), "stub");
    }

    #[test]
    fn duplicate_name_errors_but_duplicate_priority_is_allowed() {
        let registry = AnyRegistry::new();
        registry
            .register(Arc::new(StubAny {
                registered_name: "a".into(),
                priority: 100,
            }))
            .expect("first register");
        // same priority, different name: allowed.
        registry
            .register(Arc::new(StubAny {
                registered_name: "b".into(),
                priority: 100,
            }))
            .expect("second register at the same priority must succeed");
        // same name: rejected.
        let outcome = registry.register(Arc::new(StubAny {
            registered_name: "a".into(),
            priority: 200,
        }));
        assert!(matches!(outcome, Err(ProximaError::Registry(_))));
    }

    #[test]
    fn snapshot_orders_by_priority_descending_then_name() {
        let registry = AnyRegistry::new();
        registry
            .register(Arc::new(StubAny {
                registered_name: "low".into(),
                priority: 50,
            }))
            .expect("register low");
        registry
            .register(Arc::new(StubAny {
                registered_name: "high".into(),
                priority: 200,
            }))
            .expect("register high");
        registry
            .register(Arc::new(StubAny {
                registered_name: "mid-b".into(),
                priority: 100,
            }))
            .expect("register mid-b");
        registry
            .register(Arc::new(StubAny {
                registered_name: "mid-a".into(),
                priority: 100,
            }))
            .expect("register mid-a");

        let snapshot = registry.snapshot();
        let names: Vec<&str> = snapshot.iter().map(|p| p.name()).collect();
        assert_eq!(names, vec!["high", "mid-a", "mid-b", "low"]);
    }

    #[test]
    fn snapshot_named_errors_on_unknown_name() {
        let registry = AnyRegistry::new();
        registry
            .register(Arc::new(StubAny {
                registered_name: "known".into(),
                priority: 100,
            }))
            .expect("register");
        let outcome = registry.snapshot_named(&["known".to_string(), "missing".to_string()]);
        assert!(matches!(outcome, Err(ProximaError::Registry(_))));
    }
}
