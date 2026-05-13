use std::future::Future;
use std::sync::Arc;

use bytes::Bytes;
use proxima_core::ProximaError;
use proxima_primitives::transport::{DEFAULT_REPLAY_CAP_BYTES, tap_complete_with_size};
use crate::kv::cache_key_for_storage;
use crate::kv::write_back::WriteBackConditions;
use crate::kv::{CacheEntry, KvHandle};
use proxima_primitives::pipe::Method;
use proxima_primitives::pipe::{Pipe, SendPipe};
use proxima_primitives::pipe::handler::{Handler, PipeHandle, ThreadLocalPipeHandle};
use proxima_primitives::pipe::request::{Request, Response};
use proxima_primitives::pipe::telemetry_surface::Labels;

pub struct WriteBackTarget {
    pub backend: Arc<dyn KvHandle>,
    pub conditions: WriteBackConditions,
    pub label: String,
}

impl WriteBackTarget {
    pub fn new(backend: Arc<dyn KvHandle>, label: impl Into<String>) -> Self {
        Self {
            backend,
            conditions: WriteBackConditions::default(),
            label: label.into(),
        }
    }
}

/// Write-back cache middleware. Generic over the inner handle:
/// `WriteBack<PipeHandle>` impls `Handler`;
/// `WriteBack<ThreadLocalPipeHandle>` impls `ThreadLocalHandler`.
pub struct WriteBack<Inner = PipeHandle> {
    pub inner: Inner,
    pub targets: Vec<WriteBackTarget>,
    pub cap_bytes: usize,
}

impl<Inner> WriteBack<Inner> {
    #[must_use]
    pub fn new(inner: Inner, targets: Vec<WriteBackTarget>) -> Self {
        Self {
            inner,
            targets,
            cap_bytes: DEFAULT_REPLAY_CAP_BYTES,
        }
    }

    #[must_use]
    pub fn single(inner: Inner, backend: Arc<dyn KvHandle>) -> Self {
        let label = backend.name().to_string();
        Self {
            inner,
            targets: vec![WriteBackTarget::new(backend, label)],
            cap_bytes: DEFAULT_REPLAY_CAP_BYTES,
        }
    }

    #[must_use]
    pub fn with_cap_bytes(mut self, cap: usize) -> Self {
        self.cap_bytes = cap;
        self
    }
}

impl<Inner> SendPipe for WriteBack<Inner>
where
    Inner: Handler + Clone,
{
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let telemetry = request.context.telemetry.clone();
        let context_labels = request.context.metric_labels(&[]);
        let request_method = request.method.clone();
        let targets: Vec<(Arc<dyn KvHandle>, WriteBackConditions, String, String)> = self
            .targets
            .iter()
            .map(|target| {
                let key = cache_key_for_storage(&request, target.backend.version_tag());
                (
                    target.backend.clone(),
                    target.conditions.clone(),
                    target.label.clone(),
                    key,
                )
            })
            .collect();
        let cap_bytes = self.cap_bytes;
        let inner = self.inner.clone();
        async move {
            let response = SendPipe::call(&inner, request).await?;
            if !targets
                .iter()
                .any(|(_, conditions, _, _)| conditions.applies_to(&response))
            {
                return Ok(response);
            }
            // DELETE: evict from targets instead of populating. Response body is
            // typically empty so the tap-and-populate path would write garbage.
            if request_method == Method::Delete {
                for (backend, conditions, label, key) in &targets {
                    if !conditions.applies_to(&response) {
                        continue;
                    }
                    backend.evict(key);
                    let labels = context_labels_with(&context_labels, label);
                    telemetry.counter_inc("proxima.write_back.evictions_total", &labels, 1);
                    telemetry.gauge_set("proxima.cache.entries", &labels, backend.entries() as i64);
                }
                return Ok(response);
            }
            let status = response.status;
            let header_pairs: Vec<(bytes::Bytes, bytes::Bytes)> = response
                .metadata
                .iter()
                .map(|(name, value)| (bytes::Bytes::clone(name), bytes::Bytes::clone(value)))
                .collect();
            let expected_total = header_pairs
                .iter()
                .find(|(name, _)| name.as_ref().eq_ignore_ascii_case(b"content-length"))
                .and_then(|(_, value)| std::str::from_utf8(value).ok()?.parse::<usize>().ok());
            let body = response.into_chunk_stream();
            let header_pairs_for_cb = header_pairs.clone();
            let targets_for_cb = targets;
            let telemetry_for_cb = telemetry;
            let context_labels_for_cb = context_labels;
            let tapped = tap_complete_with_size(body, cap_bytes, expected_total, move |chunks| {
                for (backend, conditions, label, key) in &targets_for_cb {
                    let stub = Response::new(status);
                    if !conditions.applies_to(&stub) {
                        continue;
                    }
                    let entry =
                        CacheEntry::new(status, header_pairs_for_cb.clone(), chunks.clone(), None);
                    backend.put(key.clone(), entry);
                    let labels = context_labels_with(&context_labels_for_cb, label);
                    telemetry_for_cb.gauge_set(
                        "proxima.cache.entries",
                        &labels,
                        backend.entries() as i64,
                    );
                    telemetry_for_cb.gauge_set(
                        "proxima.cache.bytes",
                        &labels,
                        backend.bytes() as i64,
                    );
                    telemetry_for_cb.counter_inc("proxima.write_back.writes_total", &labels, 1);
                }
            });
            let mut rebuilt = Response::new(status)
                .with_stream(proxima_primitives::pipe::ResponseStream::from_chunk_stream(tapped));
            for (name, value) in header_pairs {
                rebuilt = rebuilt.with_header(name, value);
            }
            Ok(rebuilt)
        }
    }
}


impl Pipe for WriteBack<ThreadLocalPipeHandle> {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        let telemetry = request.context.telemetry.clone();
        let context_labels = request.context.metric_labels(&[]);
        let request_method = request.method.clone();
        let targets: Vec<(Arc<dyn KvHandle>, WriteBackConditions, String, String)> = self
            .targets
            .iter()
            .map(|target| {
                let key = cache_key_for_storage(&request, target.backend.version_tag());
                (
                    target.backend.clone(),
                    target.conditions.clone(),
                    target.label.clone(),
                    key,
                )
            })
            .collect();
        let cap_bytes = self.cap_bytes;
        let inner = self.inner.clone();
        async move {
            let response = Pipe::call(&inner, request).await?;
            if !targets
                .iter()
                .any(|(_, conditions, _, _)| conditions.applies_to(&response))
            {
                return Ok(response);
            }
            if request_method == Method::Delete {
                for (backend, conditions, label, key) in &targets {
                    if !conditions.applies_to(&response) {
                        continue;
                    }
                    backend.evict(key);
                    let labels = context_labels_with(&context_labels, label);
                    telemetry.counter_inc("proxima.write_back.evictions_total", &labels, 1);
                    telemetry.gauge_set("proxima.cache.entries", &labels, backend.entries() as i64);
                }
                return Ok(response);
            }
            let status = response.status;
            let header_pairs: Vec<(bytes::Bytes, bytes::Bytes)> = response
                .metadata
                .iter()
                .map(|(name, value)| (bytes::Bytes::clone(name), bytes::Bytes::clone(value)))
                .collect();
            let expected_total = header_pairs
                .iter()
                .find(|(name, _)| name.as_ref().eq_ignore_ascii_case(b"content-length"))
                .and_then(|(_, value)| std::str::from_utf8(value).ok()?.parse::<usize>().ok());
            let body = response.into_chunk_stream();
            let header_pairs_for_cb = header_pairs.clone();
            let targets_for_cb = targets;
            let telemetry_for_cb = telemetry;
            let context_labels_for_cb = context_labels;
            let tapped = tap_complete_with_size(body, cap_bytes, expected_total, move |chunks| {
                for (backend, conditions, label, key) in &targets_for_cb {
                    let stub = Response::new(status);
                    if !conditions.applies_to(&stub) {
                        continue;
                    }
                    let entry =
                        CacheEntry::new(status, header_pairs_for_cb.clone(), chunks.clone(), None);
                    backend.put(key.clone(), entry);
                    let labels = context_labels_with(&context_labels_for_cb, label);
                    telemetry_for_cb.gauge_set(
                        "proxima.cache.entries",
                        &labels,
                        backend.entries() as i64,
                    );
                    telemetry_for_cb.gauge_set(
                        "proxima.cache.bytes",
                        &labels,
                        backend.bytes() as i64,
                    );
                    telemetry_for_cb.counter_inc("proxima.write_back.writes_total", &labels, 1);
                }
            });
            let mut rebuilt = Response::new(status)
                .with_stream(proxima_primitives::pipe::ResponseStream::from_chunk_stream(tapped));
            for (name, value) in header_pairs {
                rebuilt = rebuilt.with_header(name, value);
            }
            Ok(rebuilt)
        }
    }
}


fn context_labels_with(base: &Labels, target: &str) -> Labels {
    let mut pairs: Vec<(String, String)> = base.entries().to_vec();
    pairs.push(("target".into(), target.into()));
    let pair_refs: Vec<(&str, &str)> = pairs
        .iter()
        .map(|(name, value)| (name.as_str(), value.as_str()))
        .collect();
    Labels::from_pairs(&pair_refs)
}

// integration tests using umbrella's KvCache stay in proxima/rust/tests/
