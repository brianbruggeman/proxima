//! A generic, lock-free registry of named factories — the composition-from-config
//! primitive shared by every name→thing registry in the workspace.
//!
//! A "factory" is a named builder selected by a config string: a TOML `type =
//! "http"` row names the factory, the registry looks it up, and the factory's own
//! `build` turns the row's spec into an instance. This is the shape that
//! `PipeFactoryRegistry` / `SchemaRegistry` / `ConfigFormatRegistry` each hand-
//! rolled; it lives here so they (and any new consumer — a UI component registry,
//! a codec table) reuse one primitive. The registry is agnostic to what `build`
//! returns, so it is built per output type by choosing the factory trait whose
//! `build` returns it: `FactoryRegistry<dyn PipeFactory>` builds pipes,
//! `FactoryRegistry<dyn ComponentFactory>` builds UI elements.
//!
//! A [`Factory`] (a [`Named`] whose `build` returns an `Output`) closes the
//! loop: [`FactoryRegistry::build`] walks a whole [`Composition`] tree into
//! instances in one call — the executable half of config-as-composition that
//! every consumer would otherwise hand-roll. Because `Factory: Named`, such a
//! registry needs no `impl Named for dyn ..` bridge.
//!
//! [`Named`] is dependency-free and no_std. The registry + config surface are
//! gated behind the `registry` / `config` features (std + `arc-swap`, tokio-free
//! — so they build for wasm; default-off so the base crate stays lean).

/// A factory the registry can store and look up by name. Each domain's factory
/// trait is `Named` (directly or via an `impl Named for dyn XFactory` bridge),
/// so one registry serves any of them.
pub trait Named: Send + Sync + 'static {
    /// the stable config key this factory is registered + selected under.
    fn name(&self) -> &str;
}

#[cfg(feature = "registry")]
mod registry {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use arc_swap::ArcSwap;

    use super::Named;
    use crate::ProximaError;

    /// A lock-free registry of named factories of trait `F` (a factory trait
    /// object, e.g. `dyn PipeFactory`). Reads are wait-free (`ArcSwap`);
    /// registration is copy-on-write CAS, so a factory may register itself even
    /// while the registry is shared behind an `Arc`/`Weak` (the recursive
    /// client-auth case). Build one per output type by choosing the factory
    /// trait whose `build` returns that type.
    pub struct FactoryRegistry<F: ?Sized> {
        factories: ArcSwap<BTreeMap<String, Arc<F>>>,
    }

    impl<F: ?Sized> Default for FactoryRegistry<F> {
        fn default() -> Self {
            Self {
                factories: ArcSwap::from_pointee(BTreeMap::new()),
            }
        }
    }

    impl<F: Named + ?Sized> FactoryRegistry<F> {
        #[must_use]
        pub fn new() -> Self {
            Self::default()
        }

        /// register a factory under its [`Named::name`]; errors on a duplicate.
        pub fn register(&self, factory: Arc<F>) -> Result<(), ProximaError> {
            let name = factory.name().to_string();
            loop {
                let current = self.factories.load_full();
                if current.contains_key(&name) {
                    return Err(ProximaError::Registry(format!(
                        "factory '{name}' already registered"
                    )));
                }
                let mut next: BTreeMap<String, Arc<F>> = (*current).clone();
                next.insert(name.clone(), factory.clone());
                let prev = self.factories.compare_and_swap(&current, Arc::new(next));
                if Arc::ptr_eq(&prev, &current) {
                    return Ok(());
                }
                // lost the CAS race — retry; a racing writer that took our name
                // is surfaced by the duplicate check on the next iteration.
            }
        }

        /// fluent registration: `FactoryRegistry::new().with(a)?.with(b)?`.
        pub fn with(self, factory: Arc<F>) -> Result<Self, ProximaError> {
            self.register(factory)?;
            Ok(self)
        }

        /// look up a factory by its registered name.
        pub fn get(&self, name: &str) -> Result<Arc<F>, ProximaError> {
            self.factories
                .load_full()
                .get(name)
                .cloned()
                .ok_or_else(|| ProximaError::Registry(format!("no factory named '{name}'")))
        }

        /// the registered factory names, sorted.
        #[must_use]
        pub fn names(&self) -> Vec<String> {
            self.factories.load_full().keys().cloned().collect()
        }
    }

    #[cfg(test)]
    #[allow(clippy::unwrap_used, clippy::expect_used)]
    mod tests {
        use super::*;

        struct StubFactory {
            registered_name: String,
        }
        impl Named for StubFactory {
            fn name(&self) -> &str {
                self.registered_name.as_str()
            }
        }

        #[test]
        fn register_get_and_names_round_trip() {
            let registry: FactoryRegistry<dyn Named> = FactoryRegistry::new();
            registry
                .register(Arc::new(StubFactory {
                    registered_name: "alpha".into(),
                }))
                .expect("register");
            registry
                .register(Arc::new(StubFactory {
                    registered_name: "beta".into(),
                }))
                .expect("register");
            assert_eq!(registry.get("alpha").expect("get").name(), "alpha");
            assert_eq!(
                registry.names(),
                vec!["alpha".to_string(), "beta".to_string()]
            );
        }

        #[test]
        fn duplicate_name_is_a_registry_error() {
            let registry: FactoryRegistry<dyn Named> = FactoryRegistry::new();
            registry
                .register(Arc::new(StubFactory {
                    registered_name: "dup".into(),
                }))
                .expect("first");
            assert!(matches!(
                registry.register(Arc::new(StubFactory {
                    registered_name: "dup".into()
                })),
                Err(ProximaError::Registry(_))
            ));
        }

        #[test]
        fn fluent_with_chains_registration() {
            let registry = FactoryRegistry::<dyn Named>::new()
                .with(Arc::new(StubFactory {
                    registered_name: "a".into(),
                }) as Arc<dyn Named>)
                .expect("with a")
                .with(Arc::new(StubFactory {
                    registered_name: "b".into(),
                }) as Arc<dyn Named>)
                .expect("with b");
            assert_eq!(registry.names(), vec!["a".to_string(), "b".to_string()]);
        }

        #[test]
        fn missing_name_is_a_registry_error() {
            let registry: FactoryRegistry<dyn Named> = FactoryRegistry::new();
            assert!(matches!(
                registry.get("absent"),
                Err(ProximaError::Registry(_))
            ));
        }
    }
}

#[cfg(feature = "registry")]
pub use registry::FactoryRegistry;

#[cfg(feature = "config")]
mod config {
    use std::sync::Arc;

    use bon::Builder;
    use conflaguration::{Validate, ValidationMessage};
    use serde::{Deserialize, Serialize};
    use serde_json::Value;

    use super::{FactoryRegistry, Named};
    use crate::ProximaError;

    /// A [`Named`] factory that builds its `Output` from a spec blob and the
    /// already-built `children` — the executable half of config-as-composition.
    /// Because `Factory: Named`, a `FactoryRegistry<dyn Factory<Output = T>>`
    /// stores it with no `impl Named for dyn ..` bridge, and
    /// [`FactoryRegistry::build`] walks a whole [`Composition`] through it in one
    /// call. A dock registers `dyn Factory<Output = Element>`; a notifier table
    /// `dyn Factory<Output = Notifier>`.
    pub trait Factory: Named {
        /// what this factory's `build` produces (a `Pipe`, a UI `Element`, ...).
        type Output;

        /// build one instance from its `spec` config blob and its already-built
        /// `children` (a container's contents, a wrapper's inner). the registry
        /// builds children depth-first before calling this.
        fn build(
            &self,
            spec: &Value,
            children: Vec<Self::Output>,
        ) -> Result<Self::Output, ProximaError>;
    }

    /// one node of a config-as-composition tree: the registered factory `kind`
    /// (the `type` discriminator the registry looks up), its `spec` config blob
    /// (handed to that factory's `build`), and nested `children` — a wrapping
    /// factory's inner, a container's contents. Recursive, so an arbitrarily
    /// deep composition is config, not a recompile. Built fluently via
    /// [`Self::builder`] or loaded from TOML/JSON; the two forms are identical.
    #[derive(Serialize, Deserialize, Builder)]
    #[builder(on(String, into))]
    pub struct FactorySpec {
        #[serde(rename = "type")]
        pub kind: String,

        #[serde(default)]
        #[builder(default)]
        pub spec: Value,

        #[serde(default)]
        #[builder(default)]
        pub children: Vec<FactorySpec>,
    }

    /// an ordered composition of factory specs — the top-level config. Loadable
    /// through conflaguration's layered [`conflaguration::ConfigBuilder`] (TOML/
    /// JSON file then `validate`) AND composable fluently via bon; a round-trip
    /// proves the two surfaces produce the same value.
    #[derive(Default, Serialize, Deserialize, Builder)]
    pub struct Composition {
        #[serde(default, rename = "factory")]
        #[builder(default)]
        pub factories: Vec<FactorySpec>,
    }

    impl Validate for Composition {
        fn validate(&self) -> conflaguration::Result<()> {
            fn check(spec: &FactorySpec, errors: &mut Vec<ValidationMessage>) {
                if spec.kind.trim().is_empty() {
                    errors.push(ValidationMessage::new(
                        "type",
                        "factory type must be non-empty",
                    ));
                }
                for child in &spec.children {
                    check(child, errors);
                }
            }
            let mut errors: Vec<ValidationMessage> = Vec::new();
            for spec in &self.factories {
                check(spec, &mut errors);
            }
            if errors.is_empty() {
                Ok(())
            } else {
                Err(conflaguration::Error::Validation { errors })
            }
        }
    }

    impl Composition {
        /// load through conflaguration's layered loader — the TOML/JSON `path`,
        /// then `validate`. the canonical config-surface entry point.
        pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, ProximaError> {
            conflaguration::ConfigBuilder::<Self>::new()
                .file(path)
                .validate()
                .build()
                .map_err(|err| ProximaError::Registry(format!("composition load failed: {err}")))
        }
    }

    impl<Output: 'static> FactoryRegistry<dyn Factory<Output = Output>> {
        /// build every top-level spec in a `composition` into its `Output`, each
        /// spec's `children` built depth-first first — the one-call config-as-
        /// composition driver every consumer would otherwise hand-roll.
        pub fn build(&self, composition: &Composition) -> Result<Vec<Output>, ProximaError> {
            composition
                .factories
                .iter()
                .map(|spec| self.build_spec(spec))
                .collect()
        }

        fn build_spec(&self, spec: &FactorySpec) -> Result<Output, ProximaError> {
            let children = spec
                .children
                .iter()
                .map(|child| self.build_spec(child))
                .collect::<Result<Vec<_>, _>>()?;
            self.get(&spec.kind)?.build(&spec.spec, children)
        }

        /// register a concrete factory without the `Arc::new(..) as Arc<dyn ..>`
        /// dance — the registry wraps + unsizes it.
        pub fn register_factory<T>(&self, factory: T) -> Result<(), ProximaError>
        where
            T: Factory<Output = Output> + 'static,
        {
            self.register(Arc::new(factory))
        }

        /// fluent peer of [`Self::register_factory`]:
        /// `reg.with_factory(a)?.with_factory(b)?`.
        pub fn with_factory<T>(self, factory: T) -> Result<Self, ProximaError>
        where
            T: Factory<Output = Output> + 'static,
        {
            self.with(Arc::new(factory))
        }
    }

    #[cfg(test)]
    #[allow(clippy::unwrap_used, clippy::expect_used)]
    mod tests {
        use super::*;

        fn sample() -> Composition {
            Composition::builder()
                .factories(vec![
                    FactorySpec::builder()
                        .kind("tabs")
                        .children(vec![FactorySpec::builder().kind("chat").build()])
                        .build(),
                    FactorySpec::builder()
                        .kind("properties")
                        .spec(serde_json::json!({ "side": "toast" }))
                        .build(),
                ])
                .build()
        }

        // the type carries no Clone/PartialEq/Debug (kept minimal), so identity is
        // proven on the serialized form rather than struct equality.
        #[test]
        fn fluent_and_serde_forms_are_identical() {
            let json = serde_json::to_string(&sample()).expect("serialize");
            let reloaded: Composition = serde_json::from_str(&json).expect("reload");
            assert_eq!(
                json,
                serde_json::to_string(&reloaded).expect("reserialize"),
                "the fluent and serialized forms must be identical"
            );
        }

        #[test]
        fn conflaguration_builder_validates_a_fluent_composition() {
            let built = conflaguration::ConfigBuilder::<Composition>::new()
                .value(sample())
                .validate()
                .build()
                .expect("build");
            assert_eq!(
                serde_json::to_string(&built).expect("serialize built"),
                serde_json::to_string(&sample()).expect("serialize sample"),
                "value-seeded + validated build returns the fluent composition"
            );
        }

        #[test]
        fn validate_rejects_an_empty_kind_anywhere_in_the_tree() {
            let bad = Composition::builder()
                .factories(vec![
                    FactorySpec::builder()
                        .kind("ok")
                        .children(vec![FactorySpec::builder().kind("  ").build()])
                        .build(),
                ])
                .build();
            assert!(
                bad.validate().is_err(),
                "an empty factory type, even nested, is rejected"
            );
        }

        // a `Factory` needs no `impl Named for dyn ..` bridge — the supertrait
        // gives `dyn Factory<Output = String>: Named` for free.
        struct TextFactory;
        impl Named for TextFactory {
            fn name(&self) -> &str {
                "text"
            }
        }
        impl Factory for TextFactory {
            type Output = String;
            fn build(&self, spec: &Value, _children: Vec<String>) -> Result<String, ProximaError> {
                Ok(spec
                    .get("value")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string())
            }
        }

        struct GroupFactory;
        impl Named for GroupFactory {
            fn name(&self) -> &str {
                "group"
            }
        }
        impl Factory for GroupFactory {
            type Output = String;
            fn build(&self, _spec: &Value, children: Vec<String>) -> Result<String, ProximaError> {
                Ok(format!("[{}]", children.join(",")))
            }
        }

        #[test]
        fn build_drives_a_composition_tree_in_one_call() {
            let registry = FactoryRegistry::<dyn Factory<Output = String>>::new()
                .with_factory(TextFactory)
                .expect("text")
                .with_factory(GroupFactory)
                .expect("group");
            let composition = Composition::builder()
                .factories(vec![
                    FactorySpec::builder()
                        .kind("group")
                        .children(vec![
                            FactorySpec::builder()
                                .kind("text")
                                .spec(serde_json::json!({ "value": "ask" }))
                                .build(),
                            FactorySpec::builder()
                                .kind("text")
                                .spec(serde_json::json!({ "value": "browse" }))
                                .build(),
                        ])
                        .build(),
                ])
                .build();
            let rendered = registry.build(&composition).expect("build composition");
            assert_eq!(
                rendered,
                vec!["[ask,browse]".to_string()],
                "children build depth-first, then the group wraps them"
            );
        }

        #[test]
        fn build_surfaces_an_unknown_kind_as_a_registry_error() {
            let registry: FactoryRegistry<dyn Factory<Output = String>> = FactoryRegistry::new();
            let composition = Composition::builder()
                .factories(vec![FactorySpec::builder().kind("absent").build()])
                .build();
            assert!(matches!(
                registry.build(&composition),
                Err(ProximaError::Registry(_))
            ));
        }
    }
}

#[cfg(feature = "config")]
pub use config::{Composition, Factory, FactorySpec};
