#![cfg(feature = "alloc")]

use alloc::boxed::Box;
use alloc::sync::Arc;
use core::future::Future;
use core::pin::Pin;

use serde_json::Value;

use crate::pipe::handler::PipeHandle;
use proxima_core::ProximaError;
use proxima_core::factory::Named;

pub trait PipeFactory: Send + Sync + 'static {
    fn name(&self) -> &str;

    /// Build a Pipe from a config spec. `inner` carries the next
    /// stage in the chain for wrapping pipes (auth, retry, validate,
    /// transform, …); terminal pipes (synth, http, kv, …) ignore it.
    /// A wrapping factory called with `inner: None` should error; a
    /// terminal factory called with `inner: Some(_)` may either ignore
    /// or error — convention is ignore.
    fn build(
        &self,
        spec: &Value,
        inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>>;
}

pub type DynPipeFactory = Arc<dyn PipeFactory>;

// bridge `PipeFactory` into the generic factory registry without touching any
// existing `impl PipeFactory` — the registry only needs the factory's name.
impl Named for dyn PipeFactory {
    fn name(&self) -> &str {
        PipeFactory::name(self)
    }
}

/// The pipe registry is the generic [`proxima_core::FactoryRegistry`]
/// specialized to `dyn PipeFactory`. The surface (`new` / `register` / `get` /
/// `names` / `with`) is unchanged — only the implementation is now shared,
/// workspace-wide and wasm-reachable, in `proxima-core`.
#[cfg(feature = "std")]
pub type PipeFactoryRegistry = proxima_core::FactoryRegistry<dyn PipeFactory>;

// `#[proxima::test]` pulls in the `proxima` dev-dependency, which the
// loom build keeps out of the graph (see
// `[target.'cfg(not(loom))'.dev-dependencies]` in Cargo.toml); these
// tests are unrelated to the Notify/watch loom protocol.
#[cfg(all(test, not(loom)))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::pipe::handler::into_handle;
    use crate::pipe::request::{Request, Response};
    use bytes::Bytes;
    use crate::pipe::SendPipe;

    struct StubPipe;

    impl SendPipe for StubPipe {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            async move { Ok(Response::ok("stub")) }
        }
    }

    struct StubFactory {
        registered_name: String,
    }

    impl PipeFactory for StubFactory {
        fn name(&self) -> &str {
            self.registered_name.as_str()
        }

        fn build(
            &self,
            _spec: &Value,
            _inner: Option<PipeHandle>,
        ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
            Box::pin(async move { Ok(into_handle(StubPipe)) })
        }
    }

    #[cfg(feature = "std")]
    #[proxima::test]
    async fn register_and_lookup_round_trip() {
        use alloc::string::ToString;
        use alloc::vec;
        let registry = PipeFactoryRegistry::new();
        registry
            .register(Arc::new(StubFactory {
                registered_name: "test".into(),
            }))
            .expect("register");
        let factory = registry.get("test").expect("get");
        let pipe = factory.build(&Value::Null, None).await.expect("build");
        let request = Request::builder().method("GET").path("/").build().expect("builder");
        let response = crate::pipe::SendPipe::call(&pipe, request).await.expect("call");
        assert_eq!(response.status, 200);
        assert_eq!(registry.names(), vec!["test".to_string()]);
    }

    #[cfg(feature = "std")]
    #[proxima::test]
    async fn duplicate_register_returns_registry_error() {
        let registry = PipeFactoryRegistry::new();
        registry
            .register(Arc::new(StubFactory {
                registered_name: "dup".into(),
            }))
            .expect("first register");
        let outcome = registry.register(Arc::new(StubFactory {
            registered_name: "dup".into(),
        }));
        assert!(matches!(outcome, Err(ProximaError::Registry(_))));
    }

    #[cfg(feature = "std")]
    #[proxima::test]
    async fn missing_name_returns_registry_error() {
        let registry = PipeFactoryRegistry::new();
        let outcome = registry.get("absent");
        assert!(matches!(outcome, Err(ProximaError::Registry(_))));
    }
}
