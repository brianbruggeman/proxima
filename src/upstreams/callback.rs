use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use bon::Builder;
use bytes::Bytes;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use proxima_primitives::pipe::SendPipe;

use crate::error::ProximaError;
use crate::pipe::{PipeHandle, into_handle};
use crate::pipe_factory::PipeFactory;
use crate::request::{Request, Response};
use crate::upstreams::callback_registry::{CallbackRegistry, DynCallbackFn};

pub struct CallbackUpstream {
    callback_name: String,
    label: String,
    callback: DynCallbackFn,
}

impl CallbackUpstream {
    pub fn new(
        callback_name: impl Into<String>,
        label: impl Into<String>,
        callback: DynCallbackFn,
    ) -> Self {
        Self {
            callback_name: callback_name.into(),
            label: label.into(),
            callback,
        }
    }

    /// This upstream's label, set at construction (TARGET 3 — served-Pipe
    /// naming now lives at the mount-site label, not the handle).
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }
}

impl SendPipe for CallbackUpstream {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        let callback = self.callback.clone();
        let callback_name = self.callback_name.clone();
        async move {
            let result = callback.invoke(request).await;
            match result {
                Ok(response) => Ok(response),
                Err(error) => Err(ProximaError::Upstream(format!(
                    "callback `{callback_name}` failed: {error}"
                ))),
            }
        }
    }
}


/// Typed config surface for [`CallbackUpstream`]. The `callback` itself is a
/// live Rust function resolved from [`CallbackRegistry::global`] by `name` at
/// materialisation — it is runtime-only, not data, so it never appears here.
#[derive(Debug, Clone, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_CALLBACK")]
#[builder(derive(Clone, Debug), on(String, into))]
pub struct CallbackConfig {
    /// Registered callback name to resolve. Required.
    #[setting(default)]
    #[serde(default)]
    #[builder(default)]
    pub name: String,

    /// Handler label. Defaults to `name` when absent.
    #[setting(default)]
    #[serde(default)]
    pub label: Option<String>,
}

impl Validate for CallbackConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.name.is_empty() {
            errors.push(ValidationMessage::new(
                "name",
                "callback upstream requires `name`",
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl CallbackConfig {
    /// Materialise into a runtime [`CallbackUpstream`], resolving the live
    /// callback from the global registry.
    pub fn from_config(self) -> Result<CallbackUpstream, ProximaError> {
        self.validate()
            .map_err(|err| ProximaError::Config(format!("{err}")))?;
        let callback_name = self.name;
        let label = self.label.unwrap_or_else(|| callback_name.clone());
        let callback = CallbackRegistry::global()
            .lookup(&callback_name)
            .ok_or_else(|| {
                ProximaError::Config(format!(
                    "callback `{callback_name}` not registered; call CallbackRegistry::global().register(...) before load"
                ))
            })?;
        Ok(CallbackUpstream::new(callback_name, label, callback))
    }
}

pub struct CallbackPipeFactory;

impl PipeFactory for CallbackPipeFactory {
    fn name(&self) -> &str {
        "callback"
    }

    fn build(
        &self,
        spec: &Value,
        _inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        Box::pin(async move {
            let config: CallbackConfig = serde_json::from_value(spec)
                .map_err(|err| ProximaError::Config(format!("callback config: {err}")))?;
            Ok(into_handle(config.from_config()?))
        })
    }
}

#[allow(dead_code)]
pub(crate) fn _absorb_unused_imports() -> (BTreeMap<String, String>, Arc<()>) {
    (BTreeMap::new(), Arc::new(()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::upstreams::callback_registry::CallbackFn;

    struct StaticOk;

    impl CallbackFn for StaticOk {
        fn invoke<'lifetime>(
            &'lifetime self,
            _request: Request<Bytes>,
        ) -> crate::upstreams::callback_registry::CallbackFuture<'lifetime> {
            Box::pin(async move { Ok(Response::ok("static")) })
        }
    }

    #[proxima::test]
    async fn unregistered_callback_returns_config_error() {
        let factory = CallbackPipeFactory;
        let outcome = factory
            .build(
                &serde_json::json!({"name": "definitely-not-registered"}),
                None,
            )
            .await;
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    // principle-4 parity: the fluent builder and the config value must lower to
    // identical CallbackUpstream state (callback resolved from the registry).
    #[test]
    fn parity_fluent_builder_and_config_value_match() {
        CallbackRegistry::global().register("parity_static_ok", Arc::new(StaticOk));

        let from_value: CallbackConfig = serde_json::from_value(serde_json::json!({
            "name": "parity_static_ok",
            "label": "my-callback",
        }))
        .expect("from_value");
        let from_value = from_value.from_config().expect("from_config value");

        let from_builder = CallbackConfig::builder()
            .name("parity_static_ok")
            .label("my-callback")
            .build()
            .from_config()
            .expect("from_config builder");

        assert_eq!(from_value.callback_name, from_builder.callback_name);
        assert_eq!(from_value.label, from_builder.label);
        CallbackRegistry::global().deregister("parity_static_ok");
    }

    #[proxima::test]
    async fn registered_callback_dispatches_request() {
        CallbackRegistry::global().register("test_static_ok", Arc::new(StaticOk));
        let factory = CallbackPipeFactory;
        let handle = factory
            .build(&serde_json::json!({"name": "test_static_ok"}), None)
            .await
            .expect("build");
        let request = Request::builder()
            .method("GET")
            .path("/")
            .build()
            .expect("builder");
        let response = SendPipe::call(&handle, request).await.expect("call");
        let body = response.collect_body().await.expect("collect");
        assert_eq!(&body[..], b"static");
        CallbackRegistry::global().deregister("test_static_ok");
    }
}
