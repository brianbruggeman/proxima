//! OTLP exporter face — config-first, transport is a verb not a type.
//!
//! `OtlpClient` lowers to a [`proxima::Client`](crate::Client) spec: the
//! transport (the `"http"` key, prime-native under `http-prime`) and the
//! auth / resilience axes (`headers`, `timeout`, `retry`) are config keys
//! resolved by the client's factory registry. The transport `*Upstream`
//! primitives are composed by that registry, NOT by hand here — the fluent
//! fallback (building `Upstream`s directly) stays available for code that needs
//! it, but the config path never touches one.
//!
//! ```ignore
//! let exporter = proxima::otlp::OtlpClient::http()
//!     .endpoint("http://collector:4318")
//!     .header("authorization", "Bearer …")
//!     .retry(3)
//!     .timeout(std::time::Duration::from_secs(5))
//!     .build().await?;            // -> OtlpClient (a Pipe)
//! ```

use std::time::Duration;

use bytes::Bytes;
use serde_json::{Map, Value};

use proxima_primitives::pipe::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::handler::into_handle;
use proxima_primitives::pipe::request::Response;

use crate::Client;
use crate::telemetry::config::{ExporterChoice, TelemetryConfig};
use crate::telemetry::pipes::{
    NullPipe, OtlpHttpCodec, TelemetryPipeHandle, TelemetryRequest, into_telemetry_handle,
};
use crate::telemetry::recorder::{HasPipe, Recorder, RecorderBuilder};

/// Failure composing an OTLP exporter from config.
#[derive(Debug, thiserror::Error)]
pub enum OtlpError {
    #[error("otlp client requires an endpoint")]
    MissingEndpoint,
    #[error("composing the otlp transport via proxima::Client")]
    Client(#[from] ProximaError),
}

/// Fluent configuration builder for an [`OtlpClient`].
///
/// Each verb is a config key on the lowered `proxima::Client` spec: `.endpoint`
/// -> `http`, `.header` -> `headers.request` (the http factory injects them),
/// `.retry` -> the `retry` middleware, `.timeout` -> the http per-request
/// deadline. `.build` hands the spec to `Client`, which composes the transport.
pub struct OtlpClientBuilder {
    endpoint: Option<String>,
    headers: Vec<(String, String)>,
    max_attempts: Option<u32>,
    timeout: Option<Duration>,
}

impl OtlpClientBuilder {
    /// Set the collector endpoint (`http://host:port`; the OTLP signal path is
    /// added by the codec, so the base authority is what matters).
    #[must_use]
    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Add one outbound header (e.g. `authorization`) — the auth axis.
    #[must_use]
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Add several outbound headers.
    #[must_use]
    pub fn headers<K, V>(mut self, headers: impl IntoIterator<Item = (K, V)>) -> Self
    where
        K: Into<String>,
        V: Into<String>,
    {
        for (name, value) in headers {
            self.headers.push((name.into(), value.into()));
        }
        self
    }

    /// Retry the POST up to `max_attempts` times (the `retry` middleware).
    #[must_use]
    pub fn retry(mut self, max_attempts: u32) -> Self {
        self.max_attempts = Some(max_attempts);
        self
    }

    /// Bound each send attempt with a deadline (the http per-request timeout).
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Lower the config to a `proxima::Client` spec and dispatch the OTLP codec
    /// through it. The client's registry composes the transport (prime
    /// `PrimeTcpUpstream` + `H1ClientUpstream` under `http-prime`) from the
    /// `"http"` key — none are constructed here.
    pub async fn build(self) -> Result<OtlpClient, OtlpError> {
        let endpoint = self.endpoint.ok_or(OtlpError::MissingEndpoint)?;

        let mut spec = Map::new();
        spec.insert("http".into(), Value::String(endpoint));
        if !self.headers.is_empty() {
            let mut request_headers = Map::new();
            for (name, value) in self.headers {
                request_headers.insert(name, Value::String(value));
            }
            let mut headers = Map::new();
            headers.insert("request".into(), Value::Object(request_headers));
            spec.insert("headers".into(), Value::Object(headers));
        }
        if let Some(timeout) = self.timeout {
            spec.insert(
                "timeout".into(),
                Value::String(format!("{}ms", timeout.as_millis())),
            );
        }
        if let Some(max_attempts) = self.max_attempts {
            let mut retry = Map::new();
            retry.insert("max_attempts".into(), Value::from(max_attempts));
            spec.insert("retry".into(), Value::Object(retry));
        }

        // route the wire send through the `proxima::Client` API itself (Client is
        // a Pipe): the codec's downstream IS the client, so every POST goes through
        // Client::dispatch (on/off-worker hop + self-owned runtime) rather than a
        // hand-extracted transport handle. OtlpHttpCodec itself is a
        // TelemetryRequest pipe (not a Request<Bytes> Pipe), so it is erased via
        // into_telemetry_handle, not into_handle.
        let client = Client::from_value(Value::Object(spec))?;
        Ok(OtlpClient {
            pipe: into_telemetry_handle(OtlpHttpCodec::new(into_handle(client))),
        })
    }
}

/// A configured OTLP exporter — a telemetry `Pipe` that encodes drained
/// batches to OTLP protobuf and sends them to a collector through a
/// `proxima::Client`.
pub struct OtlpClient {
    pipe: TelemetryPipeHandle,
}

impl OtlpClient {
    /// Begin configuring an OTLP/HTTP exporter.
    #[must_use]
    pub fn http() -> OtlpClientBuilder {
        OtlpClientBuilder {
            endpoint: None,
            headers: Vec::new(),
            max_attempts: None,
            timeout: None,
        }
    }

    /// The composed pipe handle, for use as a recorder terminal or chain stage.
    #[must_use]
    pub fn handle(&self) -> TelemetryPipeHandle {
        self.pipe.clone()
    }
}

impl SendPipe for OtlpClient {
    type In = TelemetryRequest;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: TelemetryRequest,
    ) -> impl core::future::Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        self.pipe.call(request)
    }
}

/// Compose the terminal exporter pipe for an [`ExporterChoice`] — the config
/// path. `OtlpHttp` delegates to the [`OtlpClient`] fluent builder (which lowers
/// to a `proxima::Client` spec).
pub async fn exporter_pipe(choice: &ExporterChoice) -> Result<TelemetryPipeHandle, OtlpError> {
    match choice {
        ExporterChoice::Noop => Ok(into_telemetry_handle(NullPipe::new())),
        ExporterChoice::OtlpHttp { endpoint } => Ok(OtlpClient::http()
            .endpoint(endpoint)
            .build()
            .await?
            .handle()),
        #[cfg(feature = "otlp-grpc")]
        ExporterChoice::OtlpGrpc { .. } => Err(OtlpError::Client(ProximaError::Config(
            "otlp/grpc transport is not yet a registered client factory".into(),
        ))),
    }
}

/// Build a recorder builder from config with the OTLP transport composed via
/// `proxima::Client`. Same as [`Recorder::from_config`] but resolves a
/// transport-requiring exporter into a working pipe before injecting it, so
/// config alone yields a recorder that actually sends over the wire.
pub async fn recorder_from_config(
    cfg: &TelemetryConfig,
) -> Result<RecorderBuilder<HasPipe>, OtlpError> {
    let pipe = exporter_pipe(&cfg.exporter).await?;
    Ok(Recorder::from_config_with_pipe(cfg, pipe))
}

#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-inbox-alloc"
))]
pub use prime_pump::{PrimePump, spawn_prime_pump};

/// The prime-native managed drain pump — the async-terminal steady state.
///
/// The leaf [`ManagedDrainer`](crate::telemetry::recorder) is a `std::thread`
/// looping the *sync* `drain_pass` (`block_on`): correct for a sync sink, a
/// deadlock for an async network terminal driven from a prime executor thread.
/// This pump is its async counterpart — a detached prime task that `.await`s
/// [`Recorder::drain_async`] on the reactor — so the OTLP-over-prime export
/// runs automatically with no manual drain.
#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-inbox-alloc"
))]
mod prime_pump {
    use core::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use futures::channel::oneshot;

    use crate::runtime::prime::os::core_shard::spawn_on_current_core;
    use crate::telemetry::recorder::Recorder;

    /// Handle to a running prime drain pump (from [`spawn_prime_pump`]).
    ///
    /// Stop it with [`PrimePump::stop`], which drains the recorder to empty and
    /// joins the pump task before returning — so the recorder may then drop
    /// without its teardown flush (a sync `block_on`) running on a non-reactor
    /// thread with records still buffered.
    pub struct PrimePump {
        stop: Arc<AtomicBool>,
        done: oneshot::Receiver<()>,
        recorder: Arc<Recorder>,
    }

    /// Spawn a prime-native drain pump for `recorder` on the CURRENT prime worker.
    ///
    /// The pump is a detached prime task (`spawn_on_current_core` —
    /// `prime::os::core_shard`) that drains on a **size-or-time** trigger, never
    /// blocking the executor thread: under `lossless-backpressure`, each cycle
    /// it waits on [`Recorder::pump_wait`] (the size trigger — a full-ring
    /// producer signals) bounded by `flush_interval` via `proxima_core::time::timeout`
    /// (the time trigger / safety net); without that feature there is no
    /// full-ring producer to unblock (`mark_pump_active` is a no-op), so it
    /// falls back to a plain `flush_interval` sleep. Either path then `.await`s
    /// [`Recorder::drain_async`] +
    /// [`Recorder::drain_instruments_async`]: ring signals AND registry
    /// counters/histograms export over the recorder's terminal pipe on the reactor
    /// with no `block_on` (a `block_on` of network I/O on a prime executor thread
    /// would deadlock it). It first marks the pump active so a full-ring producer
    /// parks for a freed slot instead of self-exporting — the contract an async
    /// terminal needs (see [`Recorder::mark_pump_active`]).
    ///
    /// Must be called on a prime worker thread; `spawn_on_current_core` panics
    /// otherwise. The recorder's transport is whatever the config already wired —
    /// compose it with [`recorder_from_config`](super::recorder_from_config), whose
    /// `OtlpHttp` endpoint lowers to a prime `H1ClientUpstream` via `proxima::Client`.
    #[must_use]
    pub fn spawn_prime_pump(recorder: Arc<Recorder>, flush_interval: Duration) -> PrimePump {
        recorder.mark_pump_active(true);
        let stop = Arc::new(AtomicBool::new(false));
        let (done, done_rx) = oneshot::channel();
        let task_stop = Arc::clone(&stop);
        let stop_recorder = Arc::clone(&recorder);
        spawn_on_current_core(Box::pin(async move {
            while !task_stop.load(Ordering::Acquire) {
                // size-or-time: wake on a full-ring producer signal, bounded by
                // flush_interval (the time trigger / safety net if a wake is
                // missed). either outcome means "drain now". without
                // lossless-backpressure there is no size trigger to race (no
                // producer ever parks), so a plain interval sleep is equivalent.
                #[cfg(feature = "lossless-backpressure")]
                {
                    let _ = proxima_core::time::timeout(flush_interval, recorder.pump_wait()).await;
                }
                #[cfg(not(feature = "lossless-backpressure"))]
                {
                    proxima_core::time::sleep(flush_interval).await;
                }
                // ring signals (spans/events/logs/metric-samples/links) AND the
                // registry instruments (counters/histograms) — both awaited, so a
                // metrics recorder exports over the async terminal too.
                recorder.drain_async().await;
                recorder.drain_instruments_async().await;
            }
            // drain to empty so the recorder's teardown flush (a sync `block_on`
            // in `EmitShared::drop`) finds nothing left to export — the pump owns
            // the only async-safe drain for a network terminal.
            while recorder.drain_async().await + recorder.drain_instruments_async().await > 0 {}
            recorder.mark_pump_active(false);
            let _ = done.send(());
        }));
        PrimePump {
            stop,
            done: done_rx,
            recorder: stop_recorder,
        }
    }

    impl PrimePump {
        /// Signal the pump to stop, then await its final drain-to-empty + exit.
        /// After this returns the rings are empty, so dropping the recorder will
        /// not `block_on` an export on a thread without a reactor.
        pub async fn stop(self) {
            self.stop.store(true, Ordering::Release);
            // wake the pump out of its size-or-time wait so it sees `stop` now,
            // not after the next flush interval.
            self.recorder.signal_pump();
            let _ = self.done.await;
        }
    }
}
