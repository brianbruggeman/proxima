use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::OnceLock;

use bytes::Bytes;
use dashmap::DashMap;

use crate::error::ProximaError;
use crate::request::{Request, Response};

pub type CallbackFuture<'lifetime> =
    Pin<Box<dyn Future<Output = Result<Response<Bytes>, ProximaError>> + Send + 'lifetime>>;

pub trait CallbackFn: Send + Sync + 'static {
    fn invoke<'lifetime>(&'lifetime self, request: Request<Bytes>) -> CallbackFuture<'lifetime>;
}

pub type DynCallbackFn = Arc<dyn CallbackFn>;

/// Process-wide registry mapping callback name → callable. Both the
/// `callback` upstream and the `transform` middleware look up by name.
pub struct CallbackRegistry {
    callbacks: DashMap<String, DynCallbackFn>,
}

impl CallbackRegistry {
    fn new() -> Self {
        Self {
            callbacks: DashMap::new(),
        }
    }

    #[must_use]
    pub fn global() -> &'static Self {
        static REGISTRY: OnceLock<CallbackRegistry> = OnceLock::new();
        REGISTRY.get_or_init(Self::new)
    }

    pub fn register(&self, name: impl Into<String>, callback: DynCallbackFn) {
        self.callbacks.insert(name.into(), callback);
    }

    pub fn lookup(&self, name: &str) -> Option<DynCallbackFn> {
        self.callbacks.get(name).map(|entry| entry.value().clone())
    }

    pub fn deregister(&self, name: &str) -> Option<DynCallbackFn> {
        self.callbacks.remove(name).map(|(_, callback)| callback)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    struct EchoBody;

    impl CallbackFn for EchoBody {
        fn invoke<'lifetime>(
            &'lifetime self,
            request: Request<Bytes>,
        ) -> CallbackFuture<'lifetime> {
            Box::pin(async move {
                let (_, bytes) = request.body_bytes().await?;
                Ok(Response::ok(bytes))
            })
        }
    }

    #[proxima::test]
    async fn register_lookup_and_invoke_callback() {
        let registry = CallbackRegistry::new();
        registry.register("echo", Arc::new(EchoBody));
        let callback = registry.lookup("echo").expect("registered");
        let request = Request::builder()
            .method("POST")
            .path("/")
            .body("ping")
            .build()
            .expect("builder");
        let response = callback.invoke(request).await.expect("invoke");
        let body = response.collect_body().await.expect("collect");
        assert_eq!(&body[..], b"ping");
    }

    #[test]
    fn deregister_removes_callback() {
        let registry = CallbackRegistry::new();
        registry.register("once", Arc::new(EchoBody));
        let removed = registry.deregister("once");
        assert!(removed.is_some());
        assert!(registry.lookup("once").is_none());
    }
}
