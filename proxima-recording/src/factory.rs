use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};

use arc_swap::ArcSwap;
use serde_json::Value;

use crate::source::DynRecordingSource;
use proxima_core::ProximaError;
use proxima_runtime::Runtime;

pub type SourceBuildFuture<'lifetime> =
    Pin<Box<dyn Future<Output = Result<DynRecordingSource, ProximaError>> + Send + 'lifetime>>;

pub trait RecordingSourceFactory: Send + Sync + 'static {
    fn name(&self) -> &str;

    fn build<'lifetime>(
        &'lifetime self,
        spec: &'lifetime Value,
        registry: &'lifetime RecordingSourceRegistry,
    ) -> SourceBuildFuture<'lifetime>;
}

pub type DynRecordingSourceFactory = Arc<dyn RecordingSourceFactory>;

pub struct RecordingSourceRegistry {
    factories: ArcSwap<BTreeMap<String, DynRecordingSourceFactory>>,
    // deferred runtime slot, armed once by the umbrella at App assembly with the
    // same runtime the recording spigot gets. Sources offload their blocking
    // file I/O through this runtime, so a factory `build` must observe it set.
    runtime: OnceLock<Arc<dyn Runtime>>,
}

impl Default for RecordingSourceRegistry {
    fn default() -> Self {
        Self {
            factories: ArcSwap::from_pointee(BTreeMap::new()),
            runtime: OnceLock::new(),
        }
    }
}

impl RecordingSourceRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Arm the registry's runtime once. Idempotent — a re-arm with the same or
    /// a different runtime is ignored (set-once), mirroring the spigot.
    pub fn set_runtime(&self, runtime: Arc<dyn Runtime>) {
        let _ = self.runtime.set(runtime);
    }

    /// The armed runtime, used by source factories to inject the offload pool
    /// into the source they build. Errors if a source is resolved before the
    /// registry was armed (a build-order bug, not a runtime condition).
    pub fn runtime(&self) -> Result<Arc<dyn Runtime>, ProximaError> {
        self.runtime.get().cloned().ok_or_else(|| {
            ProximaError::Registry(
                "recording source registry resolved before its runtime was armed".into(),
            )
        })
    }

    pub fn register(&self, factory: DynRecordingSourceFactory) -> Result<(), ProximaError> {
        let name = factory.name().to_string();
        loop {
            let current = self.factories.load_full();
            if current.contains_key(&name) {
                return Err(ProximaError::Registry(format!(
                    "recording source factory `{name}` already registered"
                )));
            }
            let mut next: BTreeMap<String, DynRecordingSourceFactory> = (*current).clone();
            next.insert(name.clone(), factory.clone());
            let prev = self.factories.compare_and_swap(&current, Arc::new(next));
            if Arc::ptr_eq(&prev, &current) {
                return Ok(());
            }
        }
    }

    pub fn get(&self, name: &str) -> Result<DynRecordingSourceFactory, ProximaError> {
        self.factories
            .load_full()
            .get(name)
            .cloned()
            .ok_or_else(|| ProximaError::Registry(format!("no recording source factory `{name}`")))
    }

    #[must_use]
    pub fn names(&self) -> Vec<String> {
        self.factories.load_full().keys().cloned().collect()
    }

    pub async fn resolve(&self, spec: &Value) -> Result<DynRecordingSource, ProximaError> {
        let kind = spec
            .get("type")
            .and_then(Value::as_str)
            .or_else(|| spec.get("format").and_then(Value::as_str))
            .ok_or_else(|| {
                ProximaError::Config("recording source spec requires `type` (or `format`)".into())
            })?;
        let factory = self.get(kind)?;
        factory.build(spec, self).await
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::source::RecordingSource;
    use futures::StreamExt;
    use serde_json::json;

    #[proxima::test]
    async fn source_registry_accepts_format_alias_for_type() {
        struct StubSource;
        impl RecordingSource for StubSource {
            fn events<'lifetime>(
                &'lifetime self,
            ) -> crate::source::RecordingEventStream<'lifetime> {
                Box::pin(futures::stream::empty())
            }
        }
        struct StubSourceFactory;
        impl RecordingSourceFactory for StubSourceFactory {
            fn name(&self) -> &str {
                "stub"
            }
            fn build<'lifetime>(
                &'lifetime self,
                _spec: &'lifetime Value,
                _registry: &'lifetime RecordingSourceRegistry,
            ) -> SourceBuildFuture<'lifetime> {
                Box::pin(async move {
                    let dyn_source: DynRecordingSource = Arc::new(StubSource);
                    Ok(dyn_source)
                })
            }
        }
        let registry = RecordingSourceRegistry::new();
        registry
            .register(Arc::new(StubSourceFactory))
            .expect("register");
        let source = registry
            .resolve(&json!({"format": "stub", "source": "ignored"}))
            .await
            .expect("resolve via format alias");
        let mut stream = source.events();
        assert!(stream.next().await.is_none());
    }
}
