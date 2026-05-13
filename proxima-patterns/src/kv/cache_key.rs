//! Stable request-hash for KV cache lookups. Lives in `proxima_patterns::kv`
//! so the HTTP cache (`upstreams::kv_cache`) and the write-back middleware
//! (`proxima_patterns::middleware::write_back`) compute identical keys
//! without one depending on the other.
use alloc::format;
use alloc::string::String;

use bytes::Bytes;
use proxima_primitives::pipe::request::Request;

/// HTTP-cache shape: method-sensitive. `GET /foo` and `POST /foo` get
/// different keys.
#[must_use]
pub fn cache_key_from_request(request: &Request<Bytes>) -> String {
    cache_key_with_version(request, None)
}

/// HTTP-cache shape with a version tag prefixed — bump the tag to
/// invalidate every cached entry without walking the store.
#[must_use]
pub fn cache_key_with_version(request: &Request<Bytes>, version: Option<&str>) -> String {
    cache_key_for(request.method.as_bytes(), request, version)
}

/// Storage shape: method-agnostic. `PUT`, `GET`, `DELETE` on the same
/// path hit the same slot. Used by upstream-as-store callers.
#[must_use]
pub fn cache_key_for_storage(request: &Request<Bytes>, version: Option<&str>) -> String {
    cache_key_for(b"GET", request, version)
}

fn cache_key_for(method: &[u8], request: &Request<Bytes>, version: Option<&str>) -> String {
    let mut hasher = xxhash_rust::xxh3::Xxh3::new();
    if let Some(tag) = version {
        hasher.update(tag.as_bytes());
        hasher.update(b"\x01");
    }
    hasher.update(method);
    hasher.update(b"\x00");
    hasher.update(request.path.as_ref());
    hasher.update(b"\x00");
    for (name, value) in &request.query {
        hasher.update(name.as_ref());
        hasher.update(b"=");
        hasher.update(value.as_ref());
        hasher.update(b"&");
    }
    format!("{:x}", hasher.digest128())
}
