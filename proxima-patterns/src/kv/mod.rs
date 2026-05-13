//! Cache-entry primitives + write-back rules for proxima — CacheEntry,
//! KvHandle, EvictionPolicy, WriteBackRule.
//!
//! Folded from the former `proxima-kv` crate.

#[cfg(feature = "alloc")]
use alloc::string::String;
#[cfg(feature = "alloc")]
use alloc::sync::Arc;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
#[cfg(feature = "alloc")]
use core::time::Duration;
#[cfg(feature = "alloc")]
use portable_atomic::{AtomicU64, Ordering};

#[cfg(feature = "alloc")]
use bytes::Bytes;

#[cfg(feature = "alloc")]
#[derive(Debug)]
pub struct CacheEntry {
    pub status: u16,
    /// Cached header name/value pairs. bytes-internal — header values
    /// can carry binary tokens (auth, signed cookies) that wouldn't
    /// round-trip through `String` cleanly.
    pub headers: Vec<(Bytes, Bytes)>,
    /// Cached body chunks. Stored as `Arc<[Bytes]>` so cloning the
    /// CacheEntry on cache writes is a refcount bump rather than a
    /// per-chunk Bytes clone (Bytes itself is already refcounted, but
    /// the outer Vec was being cloned every put).
    pub chunks: Arc<[Bytes]>,
    pub stored_at_micros: u128,
    pub last_access_micros: AtomicU64,
    pub ttl: Option<Duration>,
    pub size_bytes: usize,
}

#[cfg(feature = "alloc")]
impl Clone for CacheEntry {
    fn clone(&self) -> Self {
        Self {
            status: self.status,
            headers: self.headers.clone(),
            chunks: Arc::clone(&self.chunks),
            stored_at_micros: self.stored_at_micros,
            last_access_micros: AtomicU64::new(self.last_access_micros.load(Ordering::Relaxed)),
            ttl: self.ttl,
            size_bytes: self.size_bytes,
        }
    }
}

#[cfg(feature = "alloc")]
impl CacheEntry {
    // wall-clock reads (stored_at/last_access) are irreducible std; the
    // no_std+alloc tier can still build a CacheEntry via the struct
    // literal (all fields are pub) and supply its own timestamp source.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn new(
        status: u16,
        headers: Vec<(Bytes, Bytes)>,
        chunks: impl Into<Arc<[Bytes]>>,
        ttl: Option<Duration>,
    ) -> Self {
        let chunks: Arc<[Bytes]> = chunks.into();
        let size_bytes: usize = chunks.iter().map(bytes::Bytes::len).sum();
        Self {
            status,
            headers,
            chunks,
            stored_at_micros: micros_since_epoch(),
            last_access_micros: AtomicU64::new(micros_since_epoch() as u64),
            ttl,
            size_bytes,
        }
    }

    #[cfg(feature = "std")]
    #[must_use]
    pub fn is_fresh(&self) -> bool {
        match self.ttl {
            Some(ttl) => {
                let elapsed_micros = micros_since_epoch().saturating_sub(self.stored_at_micros);
                Duration::from_micros(elapsed_micros as u64) < ttl
            }
            None => true,
        }
    }

    #[cfg(feature = "std")]
    pub fn touch(&self) {
        self.last_access_micros
            .store(micros_since_epoch() as u64, Ordering::Relaxed);
    }

    #[must_use]
    pub fn last_access(&self) -> u64 {
        self.last_access_micros.load(Ordering::Relaxed)
    }
}

#[cfg(feature = "std")]
fn micros_since_epoch() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_micros())
        .unwrap_or(0)
}

#[cfg(feature = "alloc")]
pub trait KvHandle: Send + Sync + 'static {
    fn get(&self, key: &str) -> Option<CacheEntry>;
    fn put(&self, key: String, entry: CacheEntry);
    fn evict(&self, key: &str);
    fn entries(&self) -> usize;
    fn bytes(&self) -> usize;
    fn name(&self) -> &str;

    fn version_tag(&self) -> Option<&str> {
        None
    }

    /// Snapshot every (key, entry) pair currently stored. Default impl
    /// returns an empty vec so list-mode is opt-in per backend.
    fn iter(&self) -> Vec<(String, CacheEntry)> {
        Vec::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    Lru,
    Fifo,
    TtlOnly,
}

#[derive(Debug, Clone)]
pub struct KvCaps {
    pub max_entries: Option<usize>,
    pub max_bytes: Option<u64>,
    pub eviction: EvictionPolicy,
}

impl KvCaps {
    #[must_use]
    pub fn entries(max_entries: usize) -> Self {
        Self {
            max_entries: Some(max_entries),
            max_bytes: None,
            eviction: EvictionPolicy::Lru,
        }
    }

    #[must_use]
    pub fn bytes(max_bytes: u64) -> Self {
        Self {
            max_entries: None,
            max_bytes: Some(max_bytes),
            eviction: EvictionPolicy::Lru,
        }
    }

    #[must_use]
    pub fn with_eviction(mut self, eviction: EvictionPolicy) -> Self {
        self.eviction = eviction;
        self
    }

    #[cfg(feature = "alloc")]
    pub fn require_at_least_one(&self) -> Result<(), proxima_core::ProximaError> {
        if self.max_entries.is_none() && self.max_bytes.is_none() {
            return Err(proxima_core::ProximaError::Config(
                "kv backend requires at least one of `max_entries` or `max_bytes`".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn cache_entry_size_sums_chunks() {
        let entry = CacheEntry::new(
            200,
            vec![("content-type".into(), "text/plain".into())],
            vec![Bytes::from_static(b"hello"), Bytes::from_static(b" world")],
            None,
        );
        assert_eq!(entry.size_bytes, 11);
    }

    #[test]
    fn cache_entry_with_ttl_zero_is_immediately_stale() {
        let entry = CacheEntry::new(
            200,
            vec![],
            vec![Bytes::from_static(b"x")],
            Some(Duration::ZERO),
        );
        std::thread::sleep(Duration::from_micros(10));
        assert!(!entry.is_fresh());
    }

    #[test]
    fn cache_entry_without_ttl_is_always_fresh() {
        let entry = CacheEntry::new(200, vec![], vec![Bytes::new()], None);
        assert!(entry.is_fresh());
    }

    #[test]
    fn touch_updates_last_access() {
        let entry = CacheEntry::new(200, vec![], vec![Bytes::new()], None);
        let before = entry.last_access();
        std::thread::sleep(Duration::from_micros(50));
        entry.touch();
        assert!(entry.last_access() > before);
    }

    #[test]
    fn caps_require_at_least_one() {
        let bare = KvCaps {
            max_entries: None,
            max_bytes: None,
            eviction: EvictionPolicy::Lru,
        };
        assert!(bare.require_at_least_one().is_err());
        assert!(KvCaps::entries(100).require_at_least_one().is_ok());
        assert!(KvCaps::bytes(1024).require_at_least_one().is_ok());
    }
}

#[cfg(feature = "alloc")]
pub mod write_back;
#[cfg(feature = "alloc")]
pub use write_back::{WriteBackConditions, WriteBackRule};

#[cfg(feature = "alloc")]
pub mod cache_key;
#[cfg(feature = "alloc")]
pub use cache_key::{cache_key_for_storage, cache_key_from_request, cache_key_with_version};
