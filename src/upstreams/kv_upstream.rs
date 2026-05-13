use std::future::Future;
use std::sync::Arc;

use bytes::Bytes;

use proxima_primitives::pipe::Method;
use proxima_primitives::pipe::SendPipe;

use crate::body::ResponseStream;
use crate::error::ProximaError;
use crate::request::{Request, Response};
use crate::upstreams::kv_cache::cache_key_for_storage;
use proxima_patterns::kv::{CacheEntry, KvHandle};

pub struct KvUpstream<K: KvHandle> {
    backend: Arc<K>,
    list_mode: bool,
}

impl<K: KvHandle> KvUpstream<K> {
    #[must_use]
    pub fn new(backend: Arc<K>) -> Self {
        Self {
            backend,
            list_mode: false,
        }
    }

    #[must_use]
    pub fn with_list_mode(mut self, enabled: bool) -> Self {
        self.list_mode = enabled;
        self
    }

    #[must_use]
    pub fn backend(&self) -> Arc<K> {
        self.backend.clone()
    }
}

impl<K: KvHandle> SendPipe for KvUpstream<K> {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> {
        let key = cache_key_for_storage(&request, self.backend.version_tag());
        let backend = self.backend.clone();
        let telemetry = request.context.telemetry.clone();
        let backend_name = self.backend.name().to_string();
        let labels = request
            .context
            .metric_labels(&[("cache_name", backend_name.as_str())]);
        let method = request.method.clone();
        // List-mode dispatch is opt-in: enabled by config, triggered when
        // the routed mount bound no path params (e.g. `/todos` rather
        // than `/todos/{id}`). Cache-shape backends leave list_mode off.
        let list_mode = self.list_mode && request.context.path_params.is_empty();
        let content_type = request
            .metadata
            .get_str("content-type")
            .map(|value| Bytes::copy_from_slice(value.as_bytes()));
        async move {
            match method {
                Method::Get | Method::Head if list_mode => {
                    telemetry.counter_inc("proxima.cache.lists_total", &labels, 1);
                    Ok(list_response(backend.as_ref()))
                }
                Method::Get | Method::Head => match backend.get(&key) {
                    Some(entry) => {
                        telemetry.counter_inc("proxima.cache.hits_total", &labels, 1);
                        Ok(entry_to_response(entry))
                    }
                    None => {
                        telemetry.counter_inc("proxima.cache.misses_total", &labels, 1);
                        Err(ProximaError::NoData)
                    }
                },
                Method::Put | Method::Post => {
                    let (_, payload) = request.body_bytes().await?;
                    let mut headers: Vec<(Bytes, Bytes)> = Vec::new();
                    if let Some(value) = content_type.clone() {
                        headers.push((Bytes::from_static(b"content-type"), value));
                    }
                    let entry = CacheEntry::new(200, headers, vec![Bytes::clone(&payload)], None);
                    backend.put(key, entry);
                    telemetry.counter_inc("proxima.cache.writes_total", &labels, 1);
                    let mut response = Response::new(200).with_body(payload);
                    if let Some(value) = content_type {
                        response = response.with_header(Bytes::from_static(b"content-type"), value);
                    }
                    Ok(response)
                }
                Method::Delete => {
                    backend.evict(&key);
                    telemetry.counter_inc("proxima.cache.deletes_total", &labels, 1);
                    Ok(Response::new(204))
                }
                _ => Ok(Response::new(405)),
            }
        }
    }
}


pub fn entry_to_response(entry: CacheEntry) -> Response<Bytes> {
    let chunks: Vec<Bytes> = entry.chunks.iter().cloned().collect();
    let mut response = Response::new(entry.status).with_stream(ResponseStream::new(
        futures::stream::iter(chunks.into_iter().map(Ok)),
    ));
    for (name, value) in entry.headers {
        response = response.with_header(name, value);
    }
    response = response.with_header("x-proxima-cache", "HIT");
    response
}

/// Serialize every stored entry as a JSON array of bodies. Each entry's
/// body is parsed as JSON (best-effort: invalid JSON survives as a
/// string).
fn list_response<K: KvHandle + ?Sized>(backend: &K) -> Response<Bytes> {
    let mut values: Vec<serde_json::Value> = Vec::new();
    for (_key, entry) in backend.iter() {
        let mut buffer: Vec<u8> = Vec::new();
        for chunk in entry.chunks.iter() {
            buffer.extend_from_slice(chunk);
        }
        let parsed = serde_json::from_slice::<serde_json::Value>(&buffer).unwrap_or_else(|_| {
            serde_json::Value::String(String::from_utf8_lossy(&buffer).into_owned())
        });
        values.push(parsed);
    }
    let body_text = serde_json::Value::Array(values).to_string();
    Response::new(200)
        .with_header("content-type", "application/json")
        .with_body(body_text)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::upstreams::kv_cache::KvCache;
    use proxima_patterns::kv::KvCaps;

    fn build_request(path: &str) -> Request<Bytes> {
        Request::builder()
            .method("GET")
            .path(path)
            .build()
            .expect("builder")
    }

    #[proxima::test]
    async fn miss_returns_err_no_data() {
        let backend = KvCache::new("c", None, KvCaps::entries(10)).expect("kv");
        let upstream = KvUpstream::new(backend);
        let outcome = upstream.call(build_request("/missing")).await;
        assert!(matches!(outcome, Err(ProximaError::NoData)));
    }

    fn build_request_with(method: &'static str, path: &str, body: &'static [u8]) -> Request<Bytes> {
        Request::builder()
            .method(method)
            .path(path)
            .body(Bytes::from_static(body))
            .build()
            .expect("builder")
    }

    #[proxima::test]
    async fn put_stores_body_and_get_returns_it() {
        let backend = KvCache::new("c", None, KvCaps::entries(10)).expect("kv");
        let upstream = KvUpstream::new(backend.clone());
        let put = upstream
            .call(build_request_with(
                "PUT",
                "/grab/abc",
                b"{\"hash\":\"abc\"}",
            ))
            .await
            .expect("put");
        assert_eq!(put.status, 200);
        let get = upstream
            .call(build_request("/grab/abc"))
            .await
            .expect("get");
        let body = get.collect_body().await.expect("collect");
        assert_eq!(&body[..], b"{\"hash\":\"abc\"}");
    }

    #[proxima::test]
    async fn delete_evicts_and_subsequent_get_misses() {
        let backend = KvCache::new("c", None, KvCaps::entries(10)).expect("kv");
        let upstream = KvUpstream::new(backend.clone());
        upstream
            .call(build_request_with("PUT", "/grab/abc", b"x"))
            .await
            .expect("put");
        let del = upstream
            .call(build_request_with("DELETE", "/grab/abc", b""))
            .await
            .expect("delete");
        assert_eq!(del.status, 204);
        let outcome = upstream.call(build_request("/grab/abc")).await;
        assert!(matches!(outcome, Err(ProximaError::NoData)));
    }

    fn put_with_param(
        method: &'static str,
        path: &str,
        id: &str,
        body: &'static [u8],
    ) -> Request<Bytes> {
        let mut request = Request::builder()
            .method(method)
            .path(path)
            .body(Bytes::from_static(body))
            .build()
            .expect("builder");
        request.context.path_params.insert("id".into(), id.into());
        request
    }

    #[proxima::test]
    async fn list_mode_returns_all_entries_as_array() {
        let backend = KvCache::new("c", None, KvCaps::entries(10)).expect("kv");
        let upstream = KvUpstream::new(backend).with_list_mode(true);
        upstream
            .call(put_with_param(
                "POST",
                "/todos/a",
                "a",
                b"{\"title\":\"buy milk\"}",
            ))
            .await
            .expect("post a");
        upstream
            .call(put_with_param(
                "POST",
                "/todos/b",
                "b",
                b"{\"title\":\"walk dog\"}",
            ))
            .await
            .expect("post b");
        // GET with no path_params on the request → list-mode
        let list = upstream.call(build_request("/todos")).await.expect("list");
        assert_eq!(list.status, 200);
        let body = list.collect_body().await.expect("collect");
        let parsed: serde_json::Value = serde_json::from_slice(&body).expect("json");
        let array = parsed.as_array().expect("array");
        assert_eq!(array.len(), 2);
    }

    #[proxima::test]
    async fn hit_returns_chunks_marked_with_hit_header() {
        let backend = KvCache::new("c", None, KvCaps::entries(10)).expect("kv");
        let key = crate::upstreams::kv_cache::cache_key_from_request(&build_request("/users/42"));
        backend.put(
            key,
            CacheEntry::new(
                200,
                vec![("content-type".into(), "application/json".into())],
                vec![Bytes::from_static(b"{\"id\":42}")],
                None,
            ),
        );
        let upstream = KvUpstream::new(backend);
        let response = upstream
            .call(build_request("/users/42"))
            .await
            .expect("hit");
        assert_eq!(response.status, 200);
        assert_eq!(response.metadata.get_str("x-proxima-cache"), Some("HIT"));
        let body = response.collect_body().await.expect("collect");
        assert_eq!(&body[..], b"{\"id\":42}");
    }
}
