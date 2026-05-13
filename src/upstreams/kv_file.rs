use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use bon::Builder;
use bytes::Bytes;
// CacheEntry's atomic field is portable_atomic-typed for bare-metal targets; the
// local IndexSlot keeps the std atomic above.
use conflaguration::{Settings, Validate, ValidationMessage};
use dashmap::DashMap;
use portable_atomic::AtomicU64 as CacheAtomicU64;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::ProximaError;
use crate::pipe::PipeHandle;
use crate::pipe_factory::PipeFactory;
use crate::upstreams::kv_cache::{EvictionChoice, SizeValue, parse_duration};
use proxima_patterns::kv::{CacheEntry, EvictionPolicy, KvCaps, KvHandle};

pub struct KvFile {
    root: PathBuf,
    index: DashMap<String, IndexSlot>,
    label: String,
    default_ttl: Option<Duration>,
    caps: KvCaps,
    bytes_used: AtomicUsize,
    version_tag: Option<String>,
}

#[derive(Debug)]
struct IndexSlot {
    size_bytes: usize,
    stored_at_micros: u128,
    last_access_micros: AtomicU64,
    ttl: Option<Duration>,
}

impl IndexSlot {
    fn from_entry(entry: &CacheEntry) -> Self {
        Self {
            size_bytes: entry.size_bytes,
            stored_at_micros: entry.stored_at_micros,
            last_access_micros: AtomicU64::new(entry.last_access()),
            ttl: entry.ttl,
        }
    }

    fn is_fresh(&self) -> bool {
        match self.ttl {
            Some(ttl) => {
                let now = micros_since_epoch();
                let elapsed = now.saturating_sub(self.stored_at_micros);
                Duration::from_micros(elapsed as u64) < ttl
            }
            None => true,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct OnDiskEntry {
    status: u16,
    headers: Vec<(String, String)>,
    chunks: Vec<Vec<u8>>,
    stored_at_micros_lo: u64,
    stored_at_micros_hi: u64,
    ttl_micros: Option<u64>,
}

impl OnDiskEntry {
    fn from_cache(entry: &CacheEntry) -> Self {
        let stored_at = entry.stored_at_micros;
        Self {
            status: entry.status,
            // on-disk schema is str-typed for postcard compat with
            // existing kv:file caches; convert at the boundary via
            // lossy UTF-8.
            headers: entry
                .headers
                .iter()
                .map(|(name, value)| {
                    (
                        String::from_utf8_lossy(name).into_owned(),
                        String::from_utf8_lossy(value).into_owned(),
                    )
                })
                .collect(),
            chunks: entry.chunks.iter().map(|chunk| chunk.to_vec()).collect(),
            stored_at_micros_lo: (stored_at & 0xFFFF_FFFF_FFFF_FFFF_u128) as u64,
            stored_at_micros_hi: (stored_at >> 64) as u64,
            ttl_micros: entry.ttl.map(|duration| duration.as_micros() as u64),
        }
    }

    fn into_cache(self) -> CacheEntry {
        let stored_at =
            (u128::from(self.stored_at_micros_hi) << 64) | u128::from(self.stored_at_micros_lo);
        let chunks: std::sync::Arc<[Bytes]> = self
            .chunks
            .into_iter()
            .map(Bytes::from)
            .collect::<Vec<_>>()
            .into();
        let size_bytes: usize = chunks.iter().map(Bytes::len).sum();
        CacheEntry {
            status: self.status,
            headers: self
                .headers
                .into_iter()
                .map(|(name, value)| (Bytes::from(name), Bytes::from(value)))
                .collect(),
            chunks,
            stored_at_micros: stored_at,
            last_access_micros: CacheAtomicU64::new(stored_at as u64),
            ttl: self.ttl_micros.map(Duration::from_micros),
            size_bytes,
        }
    }
}

impl KvFile {
    pub fn new(
        label: impl Into<String>,
        root: impl Into<PathBuf>,
        default_ttl: Option<Duration>,
        caps: KvCaps,
    ) -> Result<Arc<Self>, ProximaError> {
        caps.require_at_least_one()?;
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        let store = Self {
            root,
            index: DashMap::new(),
            label: label.into(),
            default_ttl,
            caps,
            bytes_used: AtomicUsize::new(0),
            version_tag: None,
        };
        store.scan_into_index()?;
        Ok(Arc::new(store))
    }

    pub fn with_version(self: Arc<Self>, version: impl Into<String>) -> Arc<Self> {
        let label = self.label.clone();
        let default_ttl = self.default_ttl;
        let caps = self.caps.clone();
        let root = self.root.clone();
        let next = Self {
            root,
            index: DashMap::new(),
            label,
            default_ttl,
            caps,
            bytes_used: AtomicUsize::new(0),
            version_tag: Some(version.into()),
        };
        Arc::new(next)
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn scan_into_index(&self) -> Result<(), ProximaError> {
        let mut total_bytes = 0_usize;
        let outer_iter = match std::fs::read_dir(&self.root) {
            Ok(read) => read,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(err) => return Err(ProximaError::Io(err)),
        };
        for outer in outer_iter.flatten() {
            let outer_path = outer.path();
            if !outer_path.is_dir() {
                continue;
            }
            let mid_iter = match std::fs::read_dir(&outer_path) {
                Ok(read) => read,
                Err(_) => continue,
            };
            for mid in mid_iter.flatten() {
                let mid_path = mid.path();
                if !mid_path.is_dir() {
                    continue;
                }
                let inner_iter = match std::fs::read_dir(&mid_path) {
                    Ok(read) => read,
                    Err(_) => continue,
                };
                for inner in inner_iter.flatten() {
                    let path = inner.path();
                    if !path.is_file() {
                        continue;
                    }
                    let bytes = match std::fs::read(&path) {
                        Ok(payload) => payload,
                        Err(_) => continue,
                    };
                    let on_disk: OnDiskEntry = match postcard::from_bytes(&bytes) {
                        Ok(decoded) => decoded,
                        Err(_) => continue,
                    };
                    let entry = on_disk.into_cache();
                    let key = match path.file_name().and_then(|name| name.to_str()) {
                        Some(name) => name.to_string(),
                        None => continue,
                    };
                    if !entry.is_fresh() {
                        let _ = std::fs::remove_file(&path);
                        continue;
                    }
                    total_bytes = total_bytes.saturating_add(entry.size_bytes);
                    self.index.insert(key, IndexSlot::from_entry(&entry));
                }
            }
        }
        self.bytes_used.store(total_bytes, Ordering::Relaxed);
        Ok(())
    }

    fn key_path(&self, key: &str) -> PathBuf {
        let mut path = self.root.clone();
        let prefix_a = key.get(..2).unwrap_or("__");
        let prefix_b = key.get(2..4).unwrap_or("__");
        path.push(prefix_a);
        path.push(prefix_b);
        path.push(key);
        path
    }

    fn read_entry(&self, key: &str) -> Option<CacheEntry> {
        let path = self.key_path(key);
        let bytes = std::fs::read(&path).ok()?;
        let on_disk: OnDiskEntry = postcard::from_bytes(&bytes).ok()?;
        Some(on_disk.into_cache())
    }

    fn write_entry(&self, key: &str, entry: &CacheEntry) -> Result<(), ProximaError> {
        let path = self.key_path(key);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let on_disk = OnDiskEntry::from_cache(entry);
        let bytes =
            postcard::to_allocvec(&on_disk).map_err(|err| ProximaError::Encode(err.to_string()))?;
        let temp = path.with_extension("tmp");
        std::fs::write(&temp, &bytes)?;
        std::fs::rename(&temp, &path)?;
        Ok(())
    }

    fn remove_entry(&self, key: &str) {
        let path = self.key_path(key);
        let _ = std::fs::remove_file(&path);
    }

    fn evict_for_room(&self, incoming_size: usize) {
        if matches!(self.caps.eviction, EvictionPolicy::TtlOnly) {
            self.purge_expired();
            return;
        }
        if let Some(max_entries) = self.caps.max_entries {
            while self.index.len() >= max_entries {
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
        let stale: Vec<String> = self
            .index
            .iter()
            .filter(|slot| !slot.value().is_fresh())
            .map(|slot| slot.key().clone())
            .collect();
        for key in stale {
            self.evict(&key);
        }
    }

    fn pick_victim(&self) -> Option<String> {
        let metric_for = |slot: &IndexSlot| match self.caps.eviction {
            EvictionPolicy::Lru => slot.last_access_micros.load(Ordering::Relaxed),
            EvictionPolicy::Fifo | EvictionPolicy::TtlOnly => slot.stored_at_micros as u64,
        };
        let mut victim_key: Option<String> = None;
        let mut victim_metric: Option<u64> = None;
        for slot in self.index.iter() {
            let metric = metric_for(slot.value());
            if victim_metric.is_none_or(|prior| metric < prior) {
                victim_metric = Some(metric);
                victim_key = Some(slot.key().clone());
            }
        }
        victim_key
    }
}

impl KvHandle for KvFile {
    fn get(&self, key: &str) -> Option<CacheEntry> {
        let stale_size = self
            .index
            .get(key)
            .filter(|slot| !slot.value().is_fresh())
            .map(|slot| slot.value().size_bytes);
        if let Some(size) = stale_size {
            self.index.remove(key);
            self.bytes_used.fetch_sub(size, Ordering::Relaxed);
            self.remove_entry(key);
            return None;
        }
        let entry = self.read_entry(key)?;
        if let Some(slot) = self.index.get(key) {
            slot.value()
                .last_access_micros
                .store(micros_since_epoch() as u64, Ordering::Relaxed);
        }
        entry.touch();
        Some(entry)
    }

    fn put(&self, key: String, mut entry: CacheEntry) {
        if entry.ttl.is_none()
            && let Some(default_ttl) = self.default_ttl
        {
            entry.ttl = Some(default_ttl);
        }
        let size = entry.size_bytes;
        self.evict_for_room(size);
        if let Some((_, removed)) = self.index.remove(&key) {
            self.bytes_used
                .fetch_sub(removed.size_bytes, Ordering::Relaxed);
            self.remove_entry(&key);
        }
        if let Err(err) = self.write_entry(&key, &entry) {
            tracing::error!(target: "proxima.kv.file", error = %err, key = %key, "kv file write failed");
            return;
        }
        self.bytes_used.fetch_add(size, Ordering::Relaxed);
        self.index.insert(key, IndexSlot::from_entry(&entry));
    }

    fn evict(&self, key: &str) {
        if let Some((_, removed)) = self.index.remove(key) {
            self.bytes_used
                .fetch_sub(removed.size_bytes, Ordering::Relaxed);
            self.remove_entry(key);
        }
    }

    fn entries(&self) -> usize {
        self.index.len()
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
        let keys: Vec<String> = self
            .index
            .iter()
            .filter(|slot| slot.value().is_fresh())
            .map(|slot| slot.key().clone())
            .collect();
        let mut out = Vec::with_capacity(keys.len());
        for key in keys {
            if let Some(entry) = self.read_entry(&key) {
                out.push((key, entry));
            }
        }
        out
    }
}

/// Typed config surface for the `kv:file` upstream — a disk-backed cache.
/// Reuses [`EvictionChoice`] / [`SizeValue`] from [`super::kv_cache`]; differs
/// only by the required on-disk `path`.
#[derive(Debug, Clone, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "PROXIMA_KV_FILE")]
#[builder(derive(Clone, Debug), on(String, into))]
pub struct KvFileConfig {
    /// Pipe / backend label.
    #[setting(default = "kv:file")]
    #[serde(default = "default_label")]
    #[builder(default = default_label())]
    pub name: String,

    /// On-disk root directory for the cache. Required.
    #[setting(default)]
    #[serde(default)]
    #[builder(default)]
    pub path: String,

    /// Default entry TTL, e.g. `1h` / `300s`.
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
    "kv:file".to_string()
}

impl Validate for KvFileConfig {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        if self.path.is_empty() {
            errors.push(ValidationMessage::new("path", "kv:file requires `path`"));
        }
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

impl KvFileConfig {
    /// Build the disk-backed cache (without the [`KvUpstream`] wrapper).
    pub fn into_backend(self) -> Result<Arc<KvFile>, ProximaError> {
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
        let backend = KvFile::new(self.name, PathBuf::from(self.path), ttl, caps)?;
        match self.version {
            Some(version) => Ok(backend.with_version(version)),
            None => Ok(backend),
        }
    }

    /// Materialise the full `kv:file` pipe (backend + [`KvUpstream`] wrapper).
    pub fn from_config(
        self,
    ) -> Result<crate::upstreams::kv_upstream::KvUpstream<KvFile>, ProximaError> {
        let list_mode = self.list_mode;
        let backend = self.into_backend()?;
        Ok(crate::upstreams::kv_upstream::KvUpstream::new(backend).with_list_mode(list_mode))
    }
}

pub struct KvFileFactory;

impl PipeFactory for KvFileFactory {
    fn name(&self) -> &str {
        "kv:file"
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
            let config: KvFileConfig = serde_json::from_value(spec)
                .map_err(|err| ProximaError::Config(format!("kv:file config: {err}")))?;
            Ok(crate::pipe::into_handle(config.from_config()?))
        })
    }
}

/// Build a `kv:file` backend from a json spec — retained for callers (load.rs)
/// that hold a `serde_json::Value`.
pub fn build_kv_file(spec: &Value) -> Result<Arc<KvFile>, ProximaError> {
    let config: KvFileConfig = serde_json::from_value(spec.clone())
        .map_err(|err| ProximaError::Config(format!("kv:file config: {err}")))?;
    config.into_backend()
}

fn micros_since_epoch() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_micros())
        .unwrap_or(0)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use serde_json::json;

    fn fresh_caps() -> KvCaps {
        KvCaps::entries(16)
    }

    // principle-4 parity: the fluent builder and the config value must lower to
    // identical KvFile backend state (root, ttl, caps, version).
    #[test]
    fn parity_fluent_builder_and_config_value_match() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().to_str().expect("utf8 path").to_string();

        let from_value: KvFileConfig = serde_json::from_value(json!({
            "name": "disk",
            "path": root,
            "ttl": "10m",
            "max_entries": 50,
            "eviction": "ttl_only",
            "version": "v1",
        }))
        .expect("from_value");
        let from_value = from_value.into_backend().expect("into_backend value");

        let from_builder = KvFileConfig::builder()
            .name("disk")
            .path(root.clone())
            .ttl("10m")
            .max_entries(50)
            .eviction(EvictionChoice::TtlOnly)
            .version("v1")
            .build()
            .into_backend()
            .expect("into_backend builder");

        assert_eq!(from_value.root(), from_builder.root());
        assert_eq!(from_value.default_ttl, from_builder.default_ttl);
        assert_eq!(from_value.version_tag, from_builder.version_tag);
        assert_eq!(from_value.caps.max_entries, from_builder.caps.max_entries);
        assert_eq!(from_value.caps.eviction, from_builder.caps.eviction);
    }

    #[test]
    fn put_then_get_round_trips_through_disk() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = KvFile::new("c", dir.path(), None, fresh_caps()).expect("kv");
        cache.put(
            "ab12cd34".into(),
            CacheEntry::new(
                200,
                vec![("content-type".into(), "text/plain".into())],
                vec![Bytes::from_static(b"hello world")],
                None,
            ),
        );
        let entry = cache.get("ab12cd34").expect("present");
        assert_eq!(entry.status, 200);
        assert_eq!(entry.size_bytes, 11);
        assert_eq!(
            entry.headers,
            vec![("content-type".into(), "text/plain".into())]
        );
        let combined: Vec<u8> = entry.chunks.iter().flatten().copied().collect();
        assert_eq!(&combined[..], b"hello world");
    }

    #[test]
    fn miss_returns_none() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = KvFile::new("c", dir.path(), None, fresh_caps()).expect("kv");
        assert!(cache.get("absent12").is_none());
    }

    #[test]
    fn rebuilds_index_after_reopen() {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let cache = KvFile::new("c", dir.path(), None, fresh_caps()).expect("kv");
            cache.put(
                "ff00aa11".into(),
                CacheEntry::new(201, vec![], vec![Bytes::from_static(b"persisted")], None),
            );
            assert_eq!(cache.entries(), 1);
            assert_eq!(cache.bytes(), 9);
        }
        let reopened = KvFile::new("c", dir.path(), None, fresh_caps()).expect("kv");
        assert_eq!(reopened.entries(), 1, "index rebuilt from disk");
        assert_eq!(reopened.bytes(), 9);
        let entry = reopened.get("ff00aa11").expect("present");
        assert_eq!(entry.status, 201);
    }

    #[test]
    fn expired_entry_is_evicted_on_get_and_removed_from_disk() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = KvFile::new("c", dir.path(), None, fresh_caps()).expect("kv");
        cache.put(
            "deadbeef".into(),
            CacheEntry::new(
                200,
                vec![],
                vec![Bytes::from_static(b"x")],
                Some(Duration::from_micros(1)),
            ),
        );
        std::thread::sleep(Duration::from_millis(2));
        assert!(cache.get("deadbeef").is_none());
        assert_eq!(cache.entries(), 0);
        assert_eq!(cache.bytes(), 0);
        let path = cache.key_path("deadbeef");
        assert!(!path.exists(), "expired file should be removed");
    }

    #[test]
    fn lru_eviction_removes_least_recently_accessed_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = KvFile::new("c", dir.path(), None, KvCaps::entries(2)).expect("kv");
        cache.put(
            "11111111".into(),
            CacheEntry::new(200, vec![], vec![Bytes::from_static(b"o")], None),
        );
        std::thread::sleep(Duration::from_millis(2));
        cache.put(
            "22222222".into(),
            CacheEntry::new(200, vec![], vec![Bytes::from_static(b"n")], None),
        );
        let _ = cache.get("11111111");
        cache.put(
            "33333333".into(),
            CacheEntry::new(200, vec![], vec![Bytes::from_static(b"x")], None),
        );
        assert!(cache.get("11111111").is_some(), "touched entry survives");
        assert!(cache.get("22222222").is_none(), "untouched entry evicted");
        assert!(!cache.key_path("22222222").exists(), "evicted file removed");
        assert!(cache.get("33333333").is_some());
    }

    #[test]
    fn max_bytes_cap_enforced() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = KvFile::new("c", dir.path(), None, KvCaps::bytes(8)).expect("kv");
        cache.put(
            "aaaa1111".into(),
            CacheEntry::new(200, vec![], vec![Bytes::from_static(b"1234")], None),
        );
        cache.put(
            "bbbb2222".into(),
            CacheEntry::new(200, vec![], vec![Bytes::from_static(b"5678")], None),
        );
        cache.put(
            "cccc3333".into(),
            CacheEntry::new(200, vec![], vec![Bytes::from_static(b"9012")], None),
        );
        assert!(cache.bytes() <= 8, "bytes={}", cache.bytes());
    }

    #[test]
    fn factory_requires_path() {
        let factory = KvFileFactory;
        let outcome = futures::executor::block_on(factory.build(&json!({}), None));
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[test]
    fn factory_builds_via_spec() {
        let dir = tempfile::tempdir().expect("tempdir");
        let factory = KvFileFactory;
        let spec = json!({
            "path": dir.path().to_string_lossy(),
            "max_entries": 8,
            "ttl": "30s",
            "name": "disk",
        });
        let outcome = futures::executor::block_on(factory.build(&spec, None));
        assert!(outcome.is_ok(), "factory should build");
    }

    #[test]
    fn require_at_least_one_cap() {
        let dir = tempfile::tempdir().expect("tempdir");
        let outcome = KvFile::new(
            "c",
            dir.path(),
            None,
            KvCaps {
                max_entries: None,
                max_bytes: None,
                eviction: EvictionPolicy::Lru,
            },
        );
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }
}
