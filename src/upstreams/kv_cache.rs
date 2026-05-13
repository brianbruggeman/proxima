use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
// CacheEntry's atomic field is portable_atomic-typed for bare-metal targets.
use dashmap::DashMap;
use portable_atomic::AtomicU64;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::ProximaError;
use crate::pipe::PipeHandle;
use crate::pipe_factory::PipeFactory;
use proxima_patterns::kv::{CacheEntry, EvictionPolicy, KvCaps, KvHandle};

pub struct KvCache {
    store: DashMap<String, CacheEntry>,
    label: String,
    default_ttl: Option<Duration>,
    caps: KvCaps,
    bytes_used: AtomicUsize,
    version_tag: Option<String>,
}

impl KvCache {
    pub fn new(
        label: impl Into<String>,
        default_ttl: Option<Duration>,
        caps: KvCaps,
    ) -> Result<Arc<Self>, ProximaError> {
        caps.require_at_least_one()?;
        Ok(Arc::new(Self {
            store: DashMap::new(),
            label: label.into(),
            default_ttl,
            caps,
            bytes_used: AtomicUsize::new(0),
            version_tag: None,
        }))
    }

    pub fn with_version(self: Arc<Self>, version: impl Into<String>) -> Arc<Self> {
        let label = self.label.clone();
        let default_ttl = self.default_ttl;
        let caps = self.caps.clone();
        Arc::new(Self {
            store: DashMap::new(),
            label,
            default_ttl,
            caps,
            bytes_used: AtomicUsize::new(0),
            version_tag: Some(version.into()),
        })
    }

    pub fn version_tag(&self) -> Option<&str> {
        self.version_tag.as_deref()
    }

    pub fn default_ttl(&self) -> Option<Duration> {
        self.default_ttl
    }

    pub fn caps(&self) -> &KvCaps {
        &self.caps
    }

    fn evict_for_room(&self, incoming_size: usize) {
        if matches!(self.caps.eviction, EvictionPolicy::TtlOnly) {
            self.purge_expired();
            return;
        }
        if let Some(max_entries) = self.caps.max_entries {
            while self.store.len() >= max_entries {
                if let Some(victim) = self.pick_victim() {
                    self.evict(&victim);
                } else {
                    break;
                }
            }
        }
        if let Some(max_bytes) = self.caps.max_bytes {
            while self.bytes_used.load(Ordering::Relaxed) + incoming_size > max_bytes as usize {
                if let Some(victim) = self.pick_victim() {
                    self.evict(&victim);
                } else {
                    break;
                }
            }
        }
    }

    fn purge_expired(&self) {
        let stale_keys: Vec<String> = self
            .store
            .iter()
            .filter(|entry| !entry.value().is_fresh())
            .map(|entry| entry.key().clone())
            .collect();
        for key in stale_keys {
            self.evict(&key);
        }
    }

    fn pick_victim(&self) -> Option<String> {
        let metric_for = |entry: &CacheEntry| match self.caps.eviction {
            EvictionPolicy::Lru => entry.last_access(),
            EvictionPolicy::Fifo | EvictionPolicy::TtlOnly => entry.stored_at_micros as u64,
        };
        let mut victim_key: Option<String> = None;
        let mut victim_metric: Option<u64> = None;
        for entry in self.store.iter() {
            let metric = metric_for(entry.value());
            if victim_metric.is_none_or(|prior| metric < prior) {
                victim_metric = Some(metric);
                victim_key = Some(entry.key().clone());
            }
        }
        victim_key
    }
}

impl KvHandle for KvCache {
    fn get(&self, key: &str) -> Option<CacheEntry> {
        let stale_size = self
            .store
            .get(key)
            .filter(|entry| !entry.is_fresh())
            .map(|entry| entry.size_bytes);
        if let Some(size) = stale_size {
            self.store.remove(key);
            self.bytes_used.fetch_sub(size, Ordering::Relaxed);
            return None;
        }
        let entry = self.store.get(key)?;
        entry.touch();
        Some(CacheEntry {
            status: entry.status,
            headers: entry.headers.clone(),
            chunks: entry.chunks.clone(),
            stored_at_micros: entry.stored_at_micros,
            last_access_micros: AtomicU64::new(entry.last_access()),
            ttl: entry.ttl,
            size_bytes: entry.size_bytes,
        })
    }

    fn put(&self, key: String, mut entry: CacheEntry) {
        if entry.ttl.is_none()
            && let Some(default_ttl) = self.default_ttl
        {
            entry.ttl = Some(default_ttl);
        }
        let size = entry.size_bytes;
        self.evict_for_room(size);
        if let Some((_, removed)) = self.store.remove(&key) {
            self.bytes_used
                .fetch_sub(removed.size_bytes, Ordering::Relaxed);
        }
        self.bytes_used.fetch_add(size, Ordering::Relaxed);
        self.store.insert(key, entry);
    }

    fn evict(&self, key: &str) {
        if let Some((_, removed)) = self.store.remove(key) {
            self.bytes_used
                .fetch_sub(removed.size_bytes, Ordering::Relaxed);
        }
    }

    fn entries(&self) -> usize {
        self.store.len()
    }

    fn bytes(&self) -> usize {
        self.bytes_used.load(Ordering::Relaxed)
    }

    fn name(&self) -> &str {
        &self.label
    }

    fn version_tag(&self) -> Option<&str> {
        self.version_tag.as_deref()
    }

    fn iter(&self) -> Vec<(String, CacheEntry)> {
        self.store
            .iter()
            .filter(|entry| entry.value().is_fresh())
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }
}

/// Serialisable eviction policy — the config mirror of [`EvictionPolicy`]
/// (which is not itself serde). Default `lru` matches the historical
/// hand-parser's `Some("lru") | None` arm.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvictionChoice {
    #[default]
    Lru,
    Fifo,
    TtlOnly,
}

impl From<EvictionChoice> for EvictionPolicy {
    fn from(choice: EvictionChoice) -> Self {
        match choice {
            EvictionChoice::Lru => EvictionPolicy::Lru,
            EvictionChoice::Fifo => EvictionPolicy::Fifo,
            EvictionChoice::TtlOnly => EvictionPolicy::TtlOnly,
        }
    }
}

/// A byte-size value: a bare integer (bytes) or a string like `64MB`. Mirrors
/// the historical `max_bytes` parser that accepted both forms.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SizeValue {
    Bytes(u64),
    Sized(String),
}

impl SizeValue {
    pub fn to_bytes(&self) -> Result<u64, ProximaError> {
        match self {
            SizeValue::Bytes(bytes) => Ok(*bytes),
            SizeValue::Sized(text) => parse_size(text),
        }
    }
}

/// Typed config surface for the `kv:cache` upstream — an in-memory cache backend
/// wrapped in a [`KvUpstream`](crate::upstreams::kv_upstream::KvUpstream).
#[derive(Debug, Clone, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_KV_CACHE")]
#[builder(derive(Clone, Debug), on(String, into))]
pub struct KvCacheConfig {
    /// Pipe / backend label.
    #[setting(default = "kv:cache")]
    #[serde(default = "default_label")]
    #[builder(default = default_label())]
    pub name: String,

    /// Default entry TTL, e.g. `1h` / `300s`. `None` keeps entries until evicted.
    #[setting(default)]
    #[serde(default)]
    pub ttl: Option<String>,

    /// Maximum entry count before eviction kicks in.
    #[setting(default)]
    #[serde(default)]
    pub max_entries: Option<usize>,

    /// Maximum total bytes (integer bytes or a sized string like `64MB`).
    #[setting(skip)]
    #[serde(default)]
    pub max_bytes: Option<SizeValue>,

    /// Eviction policy. Defaults to `lru`.
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub eviction: EvictionChoice,

    /// Optional cache-version tag mixed into the storage key.
    #[setting(default)]
    #[serde(default)]
    pub version: Option<String>,

    /// Serve list-style lookups (multi-key) rather than single-key.
    #[setting(default)]
    #[serde(default)]
    #[builder(default)]
    pub list_mode: bool,
}

fn default_label() -> String {
    "kv:cache".to_string()
}

impl Validate for KvCacheConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if let Some(ttl) = &self.ttl
            && parse_duration(ttl).is_err()
        {
            errors.push(ValidationMessage::new("ttl", "must be like '1h' or '300s'"));
        }
        if let Some(size) = &self.max_bytes
            && size.to_bytes().is_err()
        {
            errors.push(ValidationMessage::new(
                "max_bytes",
                "must be an integer or string like '64MB'",
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl KvCacheConfig {
    /// Build the cache backend (without the [`KvUpstream`] wrapper).
    pub fn into_backend(self) -> Result<Arc<KvCache>, ProximaError> {
        self.validate()
            .map_err(|err| ProximaError::Config(format!("{err}")))?;
        let ttl = match &self.ttl {
            Some(raw) => Some(parse_duration(raw)?),
            None => None,
        };
        let max_bytes = match &self.max_bytes {
            Some(size) => Some(size.to_bytes()?),
            None => None,
        };
        let caps = KvCaps {
            max_entries: self.max_entries,
            max_bytes,
            eviction: self.eviction.into(),
        };
        caps.require_at_least_one()?;
        let backend = KvCache::new(self.name, ttl, caps)?;
        match self.version {
            Some(version) => Ok(backend.with_version(version)),
            None => Ok(backend),
        }
    }

    /// Materialise the full `kv:cache` pipe (backend + [`KvUpstream`] wrapper).
    pub fn from_config(
        self,
    ) -> Result<crate::upstreams::kv_upstream::KvUpstream<KvCache>, ProximaError> {
        let list_mode = self.list_mode;
        let backend = self.into_backend()?;
        Ok(crate::upstreams::kv_upstream::KvUpstream::new(backend).with_list_mode(list_mode))
    }
}

pub struct KvCacheFactory;

impl PipeFactory for KvCacheFactory {
    fn name(&self) -> &str {
        "kv:cache"
    }

    fn build(
        &self,
        spec: &Value,
        _inner: Option<PipeHandle>,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<crate::pipe::PipeHandle, ProximaError>>
                + Send
                + '_,
        >,
    > {
        let spec = spec.clone();
        Box::pin(async move {
            let config: KvCacheConfig = serde_json::from_value(spec)
                .map_err(|err| ProximaError::Config(format!("kv:cache config: {err}")))?;
            Ok(crate::pipe::into_handle(config.from_config()?))
        })
    }
}

/// Build a `kv:cache` backend from a json spec — retained for callers (load.rs)
/// that hold a `serde_json::Value` and want just the backend.
pub fn build_kv_cache(spec: &Value) -> Result<Arc<KvCache>, ProximaError> {
    let config: KvCacheConfig = serde_json::from_value(spec.clone())
        .map_err(|err| ProximaError::Config(format!("kv:cache config: {err}")))?;
    config.into_backend()
}

/// Parse a `max_bytes` json value (integer bytes or a sized string like `64MB`).
/// Retained for [`super::kv_file`], which reuses the same `max_bytes` contract.
pub fn parse_size_value(value: &Value) -> Option<Result<u64, ProximaError>> {
    if let Some(integer) = value.as_u64() {
        return Some(Ok(integer));
    }
    if let Some(text) = value.as_str() {
        return Some(parse_size(text));
    }
    Some(Err(ProximaError::Config(
        "max_bytes must be an integer or string like '64MB'".into(),
    )))
}

pub fn parse_size(raw: &str) -> Result<u64, ProximaError> {
    let trimmed = raw.trim();
    let split = trimmed
        .find(|character: char| character.is_alphabetic())
        .unwrap_or(trimmed.len());
    let (digits, suffix) = trimmed.split_at(split);
    let amount: u64 = digits
        .parse()
        .map_err(|_| ProximaError::Config(format!("invalid size value '{raw}'")))?;
    let multiplier = match suffix.to_ascii_uppercase().as_str() {
        "" | "B" => 1u64,
        "KB" | "K" => 1024,
        "MB" | "M" => 1024 * 1024,
        "GB" | "G" => 1024 * 1024 * 1024,
        "TB" | "T" => 1024_u64.pow(4),
        other => {
            return Err(ProximaError::Config(format!(
                "unknown size suffix '{other}'"
            )));
        }
    };
    Ok(amount * multiplier)
}

pub fn parse_duration(raw: &str) -> Result<Duration, ProximaError> {
    let trimmed = raw.trim();
    let (digits, suffix) = trimmed.split_at(
        trimmed
            .find(|character: char| character.is_alphabetic())
            .unwrap_or(trimmed.len()),
    );
    let amount: u64 = digits
        .parse()
        .map_err(|_| ProximaError::Config(format!("invalid duration value '{raw}'")))?;
    let multiplier_seconds = match suffix {
        "" | "s" | "sec" | "secs" => 1u64,
        "ms" => return Ok(Duration::from_millis(amount)),
        "us" => return Ok(Duration::from_micros(amount)),
        "m" | "min" | "mins" => 60,
        "h" | "hr" | "hrs" => 60 * 60,
        "d" | "day" | "days" => 60 * 60 * 24,
        other => {
            return Err(ProximaError::Config(format!(
                "unknown duration suffix '{other}'"
            )));
        }
    };
    Ok(Duration::from_secs(amount * multiplier_seconds))
}

pub use proxima_patterns::kv::{cache_key_for_storage, cache_key_from_request, cache_key_with_version};

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use rstest::rstest;

    fn build_request(method: &str, path: &str) -> crate::request::Request<Bytes> {
        crate::request::Request::builder()
            .method(method)
            .path(path)
            .build()
            .expect("builder")
    }

    // principle-4 parity: the fluent builder and the config value must lower to
    // identical KvCache backend state (ttl, caps, version).
    #[test]
    fn parity_fluent_builder_and_config_value_match() {
        let from_value: KvCacheConfig = serde_json::from_value(serde_json::json!({
            "name": "edge",
            "ttl": "5m",
            "max_entries": 100,
            "max_bytes": "1MB",
            "eviction": "fifo",
            "version": "v3",
        }))
        .expect("from_value");
        let from_value = from_value.into_backend().expect("into_backend value");

        let from_builder = KvCacheConfig::builder()
            .name("edge")
            .ttl("5m")
            .max_entries(100)
            .max_bytes(SizeValue::Sized("1MB".to_string()))
            .eviction(EvictionChoice::Fifo)
            .version("v3")
            .build()
            .into_backend()
            .expect("into_backend builder");

        assert_eq!(from_value.default_ttl(), from_builder.default_ttl());
        assert_eq!(from_value.version_tag(), from_builder.version_tag());
        assert_eq!(
            from_value.caps().max_entries,
            from_builder.caps().max_entries
        );
        assert_eq!(from_value.caps().max_bytes, from_builder.caps().max_bytes);
        assert_eq!(from_value.caps().eviction, from_builder.caps().eviction);
    }

    #[test]
    fn put_then_get_returns_entry() {
        let cache = KvCache::new("c", None, KvCaps::entries(10)).expect("new");
        cache.put(
            "k".into(),
            CacheEntry::new(200, vec![], vec![Bytes::from_static(b"hello")], None),
        );
        let entry = cache.get("k").expect("present");
        assert_eq!(entry.status, 200);
        assert_eq!(entry.size_bytes, 5);
    }

    #[test]
    fn miss_returns_none_not_204() {
        let cache = KvCache::new("c", None, KvCaps::entries(10)).expect("new");
        assert!(cache.get("absent").is_none());
    }

    #[test]
    fn expired_entry_is_evicted_on_get() {
        let cache = KvCache::new("c", None, KvCaps::entries(10)).expect("new");
        cache.put(
            "k".into(),
            CacheEntry::new(
                200,
                vec![],
                vec![Bytes::from_static(b"x")],
                Some(Duration::from_micros(1)),
            ),
        );
        std::thread::sleep(Duration::from_millis(2));
        assert!(cache.get("k").is_none());
        assert_eq!(cache.entries(), 0);
        assert_eq!(cache.bytes(), 0);
    }

    #[test]
    fn lru_eviction_removes_least_recently_accessed() {
        let cache = KvCache::new("c", None, KvCaps::entries(2)).expect("new");
        cache.put(
            "old".into(),
            CacheEntry::new(200, vec![], vec![Bytes::from_static(b"o")], None),
        );
        std::thread::sleep(Duration::from_millis(2));
        cache.put(
            "newer".into(),
            CacheEntry::new(200, vec![], vec![Bytes::from_static(b"n")], None),
        );
        let _ = cache.get("old");
        cache.put(
            "newest".into(),
            CacheEntry::new(200, vec![], vec![Bytes::from_static(b"x")], None),
        );
        assert!(
            cache.get("old").is_some(),
            "old was touched, should survive"
        );
        assert!(
            cache.get("newer").is_none(),
            "newer was untouched, should be evicted"
        );
        assert!(cache.get("newest").is_some());
    }

    #[test]
    fn max_bytes_cap_enforced() {
        let cache = KvCache::new("c", None, KvCaps::bytes(8)).expect("new");
        cache.put(
            "first".into(),
            CacheEntry::new(200, vec![], vec![Bytes::from_static(b"1234")], None),
        );
        cache.put(
            "second".into(),
            CacheEntry::new(200, vec![], vec![Bytes::from_static(b"5678")], None),
        );
        cache.put(
            "third".into(),
            CacheEntry::new(200, vec![], vec![Bytes::from_static(b"9012")], None),
        );
        assert!(cache.bytes() <= 8, "bytes={}", cache.bytes());
    }

    #[test]
    fn require_at_least_one_cap() {
        let outcome = KvCache::new(
            "c",
            None,
            KvCaps {
                max_entries: None,
                max_bytes: None,
                eviction: EvictionPolicy::Lru,
            },
        );
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[rstest]
    #[case::secs("90s", Duration::from_secs(90))]
    #[case::millis("500ms", Duration::from_millis(500))]
    #[case::hours("2h", Duration::from_secs(7200))]
    fn duration_parse(#[case] input: &str, #[case] expected: Duration) {
        assert_eq!(parse_duration(input).expect("parse"), expected);
    }

    #[rstest]
    #[case::raw_int("1024", 1024)]
    #[case::kb("64KB", 64 * 1024)]
    #[case::mb("16MB", 16 * 1024 * 1024)]
    #[case::gb("2GB", 2 * 1024 * 1024 * 1024)]
    fn size_parse(#[case] input: &str, #[case] expected: u64) {
        assert_eq!(parse_size(input).expect("parse"), expected);
    }

    #[test]
    fn cache_key_changes_with_path() {
        let request_a = build_request("GET", "/users/1");
        let request_b = build_request("GET", "/users/2");
        assert_ne!(
            cache_key_from_request(&request_a),
            cache_key_from_request(&request_b)
        );
    }
}
