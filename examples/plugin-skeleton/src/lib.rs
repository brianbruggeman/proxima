//! proxima-plugin-skeleton
//!
//! Canonical plugin skeleton. Replace `StampHeader` with your own
//! pipe and rename the crate. The composition pattern at the bottom
//! (`register(builder) -> Result<AppBuilder>`) is the convention plugin
//! crates expose so users get one-line composition:
//!
//! ```ignore
//! use proxima::App;
//! let app = my_plugin::register(
//!     App::builder().with_defaults()?
//! )?
//! .build()?;
//! ```

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use bytes::Bytes;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::plugin::PluginRegistry;
use proxima_primitives::pipe::{PipeFactory, PipeHandle, ProximaError, Request, Response, into_handle};
use serde_json::Value;

/// Stamps a configurable header onto every response. Trivial example
/// to keep the skeleton readable; replace with your real logic.
pub struct StampHeader {
    inner: PipeHandle,
    name: String,
    value: String,
}

impl SendPipe for StampHeader {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        let inner = self.inner.clone();
        let header_name = self.name.clone();
        let header_value = self.value.clone();
        async move {
            let response = SendPipe::call(&inner, request).await?;
            Ok(response.with_header(header_name, header_value))
        }
    }
}


pub struct StampHeaderFactory;

impl StampHeaderFactory {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for StampHeaderFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl PipeFactory for StampHeaderFactory {
    fn name(&self) -> &str {
        "stamp_header"
    }

    fn build(
        &self,
        spec: &Value,
        inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let header_name = spec
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("x-stamp")
            .to_string();
        let header_value = spec
            .get("value")
            .and_then(Value::as_str)
            .unwrap_or("set")
            .to_string();
        Box::pin(async move {
            let inner = inner.ok_or_else(|| {
                ProximaError::Config("stamp_header requires an inner pipe".into())
            })?;
            Ok(into_handle(StampHeader {
                inner,
                name: header_name,
                value: header_value,
            }))
        })
    }
}

/// Canonical plugin entry point. Apps depending on this crate compose
/// it into their builder via `my_plugin::register(builder)?`.
pub fn register<R: PluginRegistry>(builder: R) -> Result<R, ProximaError> {
    builder.with_upstream_factory(Arc::new(StampHeaderFactory::new()))
}
