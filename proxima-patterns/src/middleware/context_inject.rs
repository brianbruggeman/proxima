use std::future::Future;
use std::sync::Arc;

use bytes::Bytes;
use tracing::Instrument;

use proxima_core::ProximaError;
use proxima_primitives::pipe::{Pipe, SendPipe};
use proxima_primitives::pipe::handler::{Handler, PipeHandle, ThreadLocalPipeHandle};
use proxima_primitives::pipe::request::{Request, Response};
use proxima_primitives::pipe::telemetry_surface::TelemetryHandle;

/// Telemetry / tracing context injection. Generic over the inner handle:
/// `ContextInjector<PipeHandle>` impls `Handler`;
/// `ContextInjector<ThreadLocalPipeHandle>` impls `ThreadLocalHandler`.
pub struct ContextInjector<Inner = PipeHandle> {
    inner: Inner,
    telemetry: TelemetryHandle,
    pipe_label: Option<Arc<[u8]>>,
}

impl<Inner> ContextInjector<Inner> {
    #[must_use]
    pub fn new(inner: Inner, telemetry: TelemetryHandle) -> Self {
        Self {
            inner,
            telemetry,
            pipe_label: None,
        }
    }

    #[must_use]
    pub fn with_pipe_label(mut self, label: impl AsRef<[u8]>) -> Self {
        self.pipe_label = Some(Arc::from(label.as_ref()));
        self
    }
}

impl<Inner> SendPipe for ContextInjector<Inner>
where
    Inner: Handler + Clone,
{
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        mut request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        request.context.telemetry = self.telemetry.clone();
        if request.context.pipe_label.is_none()
            && let Some(label) = &self.pipe_label
        {
            request.context.pipe_label = Some(label.clone());
        }
        let inner = self.inner.clone();
        // no explicit pipe_label set (TARGET 3 — name lives at the
        // mount-site label now, not on the handle itself; a handle-form
        // mount with no `.with_pipe_label(...)` degrades to "anonymous").
        let span_pipe: String = request
            .context
            .pipe_label
            .as_deref()
            .and_then(|bytes| std::str::from_utf8(bytes).ok())
            .map(str::to_owned)
            .unwrap_or_else(|| "anonymous".to_string());
        let method_view = String::from_utf8_lossy(request.method.as_bytes()).into_owned();
        let path_view = String::from_utf8_lossy(&request.path).into_owned();
        let trace_id_str: String = request
            .context
            .trace_id
            .as_deref()
            .and_then(|bytes| std::str::from_utf8(bytes).ok())
            .map(str::to_owned)
            .unwrap_or_default();
        let span = tracing::info_span!(
            "proxima.request",
            method = %method_view,
            path = %path_view,
            pipe = %span_pipe,
            trace_id = %trace_id_str,
        );
        async move { SendPipe::call(&inner, request).await }.instrument(span)
    }
}


impl Pipe for ContextInjector<ThreadLocalPipeHandle> {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        mut request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        request.context.telemetry = self.telemetry.clone();
        if request.context.pipe_label.is_none()
            && let Some(label) = &self.pipe_label
        {
            request.context.pipe_label = Some(label.clone());
        }
        let inner = self.inner.clone();
        // see the SendPipe impl above: no more handle-level name fallback.
        let span_pipe: String = request
            .context
            .pipe_label
            .as_deref()
            .and_then(|bytes| std::str::from_utf8(bytes).ok())
            .map(str::to_owned)
            .unwrap_or_else(|| "anonymous".to_string());
        let method_view = String::from_utf8_lossy(request.method.as_bytes()).into_owned();
        let path_view = String::from_utf8_lossy(&request.path).into_owned();
        let trace_id_str_tl: String = request
            .context
            .trace_id
            .as_deref()
            .and_then(|bytes| std::str::from_utf8(bytes).ok())
            .map(str::to_owned)
            .unwrap_or_default();
        let span = tracing::info_span!(
            "proxima.request",
            method = %method_view,
            path = %path_view,
            pipe = %span_pipe,
            trace_id = %trace_id_str_tl,
        );
        async move { Pipe::call(&inner, request).await }.instrument(span)
    }
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_primitives::pipe::handler::into_handle;
    use proxima_primitives::pipe::request::RequestContext;
    use proxima_telemetry::{Labels, Metrics};
    use std::sync::Arc;

    struct Echo;

    impl SendPipe for Echo {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            async move {
                request.context.telemetry.counter_inc(
                    "echo_called_total",
                    &request.context.metric_labels(&[]),
                    1,
                );
                Ok(Response::ok(bytes::Bytes::new()))
            }
        }
    }


    #[proxima::test]
    async fn injector_sets_telemetry_so_inner_can_emit() {
        let metrics = Arc::new(Metrics::default());
        let telemetry: TelemetryHandle = metrics.clone();
        let inner: PipeHandle = into_handle(Echo);
        let injector = ContextInjector::new(inner, telemetry).with_pipe_label("svc-x");

        let request = Request::builder()
            .method("GET")
            .path("/")
            .build()
            .expect("builder");
        let _ = SendPipe::call(&injector, request).await.expect("call");

        let labels = Labels::from_pairs(&[("pipe", "svc-x")]);
        assert_eq!(metrics.counter("echo_called_total", &labels), Some(1));
    }

    #[proxima::test]
    async fn injector_does_not_clobber_already_set_pipe_label() {
        let metrics = Arc::new(Metrics::default());
        let telemetry: TelemetryHandle = metrics.clone();
        let inner: PipeHandle = into_handle(Echo);
        let injector = ContextInjector::new(inner, telemetry).with_pipe_label("from-injector");

        let context = RequestContext::default().with_pipe_label("from-caller");
        let request = Request::builder()
            .method("GET")
            .path("/")
            .context(context)
            .build()
            .expect("builder");
        let _ = SendPipe::call(&injector, request).await.expect("call");

        let labels = Labels::from_pairs(&[("pipe", "from-caller")]);
        assert_eq!(metrics.counter("echo_called_total", &labels), Some(1));
    }
}
