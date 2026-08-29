use std::future::Future;
use std::sync::Arc;

use crate::kv::cache_key_for_storage;
use crate::kv::write_back::WriteBackConditions;
use crate::kv::{CacheEntry, KvHandle};
use crate::middleware::labels_with;
use bytes::Bytes;
use proxima_core::ProximaError;
use proxima_primitives::pipe::Method;
use proxima_primitives::pipe::handler::{Handler, PipeHandle, ThreadLocalPipeHandle};
use proxima_primitives::pipe::request::{Request, Response};
use proxima_primitives::pipe::{Pipe, SendPipe};
use proxima_primitives::transport::{DEFAULT_REPLAY_CAP_BYTES, tap_complete_with_size};

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
                    let labels = labels_with(&context_labels, "target", label);
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
                    let labels = labels_with(&context_labels_for_cb, "target", label);
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
            let mut rebuilt = Response::new(status).with_stream(
                proxima_primitives::pipe::ResponseStream::from_chunk_stream(tapped),
            );
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
                    let labels = labels_with(&context_labels, "target", label);
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
                    let labels = labels_with(&context_labels_for_cb, "target", label);
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
            let mut rebuilt = Response::new(status).with_stream(
                proxima_primitives::pipe::ResponseStream::from_chunk_stream(tapped),
            );
            for (name, value) in header_pairs {
                rebuilt = rebuilt.with_header(name, value);
            }
            Ok(rebuilt)
        }
    }
}

#[cfg(test)]
// the workspace denies unwrap/expect; tests assert through them.
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    use proxima_primitives::pipe::handler::into_handle;

    use super::*;

    /// In-memory `KvHandle` recording every put/evict, so a test can assert
    /// what the write-back tap actually wrote rather than that it ran.
    #[derive(Default)]
    struct FakeKv {
        entries: Mutex<BTreeMap<String, CacheEntry>>,
        evicted: Mutex<Vec<String>>,
    }

    impl KvHandle for FakeKv {
        fn get(&self, key: &str) -> Option<CacheEntry> {
            self.entries.lock().expect("lock").get(key).cloned()
        }

        fn put(&self, key: String, entry: CacheEntry) {
            self.entries.lock().expect("lock").insert(key, entry);
        }

        fn evict(&self, key: &str) {
            self.entries.lock().expect("lock").remove(key);
            self.evicted.lock().expect("lock").push(key.to_string());
        }

        fn entries(&self) -> usize {
            self.entries.lock().expect("lock").len()
        }

        fn bytes(&self) -> usize {
            self.entries
                .lock()
                .expect("lock")
                .values()
                .map(|entry| entry.size_bytes)
                .sum()
        }

        fn name(&self) -> &str {
            "fake-kv"
        }
    }

    /// Answers with `status` and a fixed body.
    struct Origin {
        status: u16,
    }

    impl SendPipe for Origin {
        type In = Request<Bytes>;
        type Out = Response<Bytes>;
        type Err = ProximaError;

        fn call(
            &self,
            _request: Request<Bytes>,
        ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
            let status = self.status;
            async move {
                Ok(Response::new(status)
                    .with_header("content-type", "text/plain")
                    .with_body(Bytes::from_static(b"hello world")))
            }
        }
    }

    fn request(method: &str) -> Request<Bytes> {
        Request::builder()
            .method(method)
            .path("/things/42")
            .build()
            .expect("request")
    }

    #[proxima::test]
    async fn success_response_body_is_written_back_after_the_body_drains() {
        let backend = Arc::new(FakeKv::default());
        let pipe = WriteBack::single(into_handle(Origin { status: 200 }), backend.clone());

        let response = SendPipe::call(&pipe, request("GET")).await.expect("call");
        assert_eq!(response.status, 200);
        assert_eq!(
            backend.entries(),
            0,
            "the tap fires only once the body ends"
        );

        let body = response.collect_body().await.expect("collect");
        assert_eq!(&body[..], b"hello world");
        assert_eq!(
            backend.entries(),
            1,
            "draining the body populates the cache"
        );

        let key = cache_key_for_storage(&request("GET"), None);
        let cached = backend.get(&key).expect("cached entry");
        assert_eq!(cached.status, 200);
        assert_eq!(cached.size_bytes, b"hello world".len());
        assert!(
            cached
                .headers
                .iter()
                .any(|(name, _)| name.as_ref() == b"content-type"),
            "response headers ride along into the entry"
        );
    }

    #[proxima::test]
    async fn non_success_response_is_passed_through_untouched() {
        let backend = Arc::new(FakeKv::default());
        let pipe = WriteBack::single(into_handle(Origin { status: 500 }), backend.clone());

        let response = SendPipe::call(&pipe, request("GET")).await.expect("call");
        assert_eq!(response.status, 500);
        let body = response.collect_body().await.expect("collect");
        assert_eq!(
            &body[..],
            b"hello world",
            "the body still reaches the caller"
        );
        assert_eq!(backend.entries(), 0, "default conditions are 2xx-only");
    }

    #[proxima::test]
    async fn delete_evicts_the_slot_instead_of_populating_it() {
        let backend = Arc::new(FakeKv::default());
        let key = cache_key_for_storage(&request("DELETE"), None);
        backend.put(
            key.clone(),
            CacheEntry::new(200, vec![], vec![Bytes::from_static(b"stale")], None),
        );

        let pipe = WriteBack::single(into_handle(Origin { status: 200 }), backend.clone());
        let response = SendPipe::call(&pipe, request("DELETE"))
            .await
            .expect("call");

        assert_eq!(response.status, 200);
        assert_eq!(backend.entries(), 0, "the stale slot is gone");
        assert_eq!(backend.evicted.lock().expect("lock").as_slice(), &[key]);
    }

    /// A DELETE the conditions reject must not evict — the guard runs per
    /// target, not once for the whole request.
    #[proxima::test]
    async fn delete_with_a_failing_condition_leaves_the_slot_alone() {
        let backend = Arc::new(FakeKv::default());
        let key = cache_key_for_storage(&request("DELETE"), None);
        backend.put(
            key.clone(),
            CacheEntry::new(200, vec![], vec![Bytes::from_static(b"stale")], None),
        );

        let target = WriteBackTarget {
            backend: backend.clone(),
            conditions: WriteBackConditions {
                only_on_success: true,
                min_status: 204,
                max_status: 204,
            },
            label: "fake-kv".into(),
        };
        let pipe = WriteBack::new(into_handle(Origin { status: 200 }), vec![target]);
        let response = SendPipe::call(&pipe, request("DELETE"))
            .await
            .expect("call");

        assert_eq!(response.status, 200);
        assert_eq!(backend.entries(), 1, "200 is outside the 204..=204 window");
        assert!(backend.evicted.lock().expect("lock").is_empty());
    }
}
