//! k8s-style desired-state persistence for the daemon.
//!
//! Owns `${PROXIMA_STATE_DIR:-~/.proxima/state.d/}`. Each pipe lives at
//! `pipes/<name>.toml`, each listener at `listeners/<name>.toml`. The
//! store is the source of truth for desired state; the daemon's running
//! state is reconciled against it on startup and on every mutation.
//!
//! Writes are atomic: write to `.tmp-<random>`, fsync, rename. A failed
//! mutation leaves the store unchanged (no torn files).
//!
//! Tiered: `PipeRecord` / `ListenerRecord` are the alloc-tier schema — no
//! std needed to hold or (de)serialize a record. `StateStore` itself is
//! std-tier — fs, directory walks, and the `toml` on-disk format have no
//! no_std analog.
//!
//! Folded in from the former `proxima-state-store` satellite crate. `store`
//! is the alloc floor (`PipeRecord` / `ListenerRecord`), `store-std` adds
//! `StateStore` itself.

#[cfg(feature = "store")]
use alloc::string::String;
#[cfg(feature = "store")]
use alloc::vec::Vec;

#[cfg(feature = "store")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "store")]
use serde_json::Value;

#[cfg(feature = "store-std")]
use std::collections::BTreeMap;
#[cfg(feature = "store-std")]
use std::fs;
#[cfg(feature = "store-std")]
use std::io::Write;
#[cfg(feature = "store-std")]
use std::path::{Path, PathBuf};

#[cfg(feature = "store-std")]
use proxima_core::ProximaError;

#[cfg(feature = "store-std")]
const PIPES_SUBDIR: &str = "pipes";
#[cfg(feature = "store-std")]
const LISTENERS_SUBDIR: &str = "listeners";

/// Filesystem-backed desired-state store. Holds JSON-shaped pipe and
/// listener specs keyed by name, serialized as TOML on disk.
#[cfg(feature = "store-std")]
#[derive(Debug, Clone)]
pub struct StateStore {
    root: PathBuf,
}

#[cfg(feature = "store")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PipeRecord {
    pub name: String,
    #[serde(flatten)]
    pub spec: Value,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub requires: Vec<String>,
}

#[cfg(feature = "store")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ListenerRecord {
    pub name: String,
    #[serde(flatten)]
    pub spec: Value,
}

#[cfg(feature = "store-std")]
impl StateStore {
    /// Open (or create) a store rooted at `path`. Subdirectories
    /// `pipes/` and `listeners/` are created if absent.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ProximaError> {
        let root = path.as_ref().to_path_buf();
        fs::create_dir_all(root.join(PIPES_SUBDIR))
            .map_err(|err| store_error(&root, "create pipes subdir", err))?;
        fs::create_dir_all(root.join(LISTENERS_SUBDIR))
            .map_err(|err| store_error(&root, "create listeners subdir", err))?;
        Ok(Self { root })
    }

    /// Resolve the default state dir: `$PROXIMA_STATE_DIR` if set,
    /// otherwise `~/.proxima/state.d/`.
    pub fn default_path() -> Result<PathBuf, ProximaError> {
        if let Ok(value) = std::env::var("PROXIMA_STATE_DIR") {
            return Ok(PathBuf::from(value));
        }
        let home = std::env::var("HOME").map_err(|_| {
            ProximaError::Config("state store: HOME not set and PROXIMA_STATE_DIR not set".into())
        })?;
        Ok(PathBuf::from(home).join(".proxima").join("state.d"))
    }

    /// Open the default state store.
    pub fn open_default() -> Result<Self, ProximaError> {
        Self::open(Self::default_path()?)
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn list_pipes(&self) -> Result<BTreeMap<String, PipeRecord>, ProximaError> {
        load_records(&self.root.join(PIPES_SUBDIR))
    }

    pub fn get_pipe(&self, name: &str) -> Result<Option<PipeRecord>, ProximaError> {
        load_record_optional(&pipe_path(&self.root, name))
    }

    pub fn put_pipe(&self, record: &PipeRecord) -> Result<(), ProximaError> {
        if record.name.is_empty() {
            return Err(ProximaError::Config("state store: pipe name empty".into()));
        }
        atomic_write_toml(&pipe_path(&self.root, &record.name), record)
    }

    pub fn remove_pipe(&self, name: &str) -> Result<bool, ProximaError> {
        remove_record(&pipe_path(&self.root, name))
    }

    pub fn list_listeners(&self) -> Result<BTreeMap<String, ListenerRecord>, ProximaError> {
        load_records(&self.root.join(LISTENERS_SUBDIR))
    }

    pub fn get_listener(&self, name: &str) -> Result<Option<ListenerRecord>, ProximaError> {
        load_record_optional(&listener_path(&self.root, name))
    }

    pub fn put_listener(&self, record: &ListenerRecord) -> Result<(), ProximaError> {
        if record.name.is_empty() {
            return Err(ProximaError::Config(
                "state store: listener name empty".into(),
            ));
        }
        atomic_write_toml(&listener_path(&self.root, &record.name), record)
    }

    pub fn remove_listener(&self, name: &str) -> Result<bool, ProximaError> {
        remove_record(&listener_path(&self.root, name))
    }
}

#[cfg(feature = "store-std")]
fn pipe_path(root: &Path, name: &str) -> PathBuf {
    root.join(PIPES_SUBDIR).join(format!("{name}.toml"))
}

#[cfg(feature = "store-std")]
fn listener_path(root: &Path, name: &str) -> PathBuf {
    root.join(LISTENERS_SUBDIR).join(format!("{name}.toml"))
}

#[cfg(feature = "store-std")]
fn store_error(root: &Path, action: &str, err: std::io::Error) -> ProximaError {
    ProximaError::Config(format!(
        "state store at {}: {action}: {err}",
        root.display()
    ))
}

#[cfg(feature = "store-std")]
fn atomic_write_toml<T: Serialize>(target: &Path, value: &T) -> Result<(), ProximaError> {
    let parent = target.parent().ok_or_else(|| {
        ProximaError::Config(format!(
            "state store: target path has no parent: {}",
            target.display()
        ))
    })?;
    let serialized = toml::to_string_pretty(value).map_err(|err| {
        ProximaError::Config(format!(
            "state store: serialize {}: {err}",
            target.display()
        ))
    })?;
    let tmp = parent.join(format!(".tmp-{}", random_suffix()));
    {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .map_err(|err| io_error(&tmp, "create tmp file", err))?;
        file.write_all(serialized.as_bytes())
            .map_err(|err| io_error(&tmp, "write tmp file", err))?;
        file.sync_all()
            .map_err(|err| io_error(&tmp, "fsync tmp file", err))?;
    }
    fs::rename(&tmp, target).map_err(|err| {
        let _ = fs::remove_file(&tmp);
        io_error(target, "rename tmp -> target", err)
    })?;
    Ok(())
}

#[cfg(feature = "store-std")]
fn remove_record(target: &Path) -> Result<bool, ProximaError> {
    match fs::remove_file(target) {
        Ok(()) => Ok(true),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(err) => Err(io_error(target, "remove", err)),
    }
}

#[cfg(feature = "store-std")]
fn load_records<T: serde::de::DeserializeOwned>(
    dir: &Path,
) -> Result<BTreeMap<String, T>, ProximaError> {
    let mut out: BTreeMap<String, T> = BTreeMap::new();
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(err) => return Err(io_error(dir, "read dir", err)),
    };
    for entry in entries {
        let entry = entry.map_err(|err| io_error(dir, "iterate dir", err))?;
        let path = entry.path();
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        if path.extension().and_then(|extension| extension.to_str()) != Some("toml") {
            continue;
        }
        if stem.starts_with(".tmp-") {
            continue;
        }
        if let Some(record) = load_record_optional::<T>(&path)? {
            out.insert(stem.to_string(), record);
        }
    }
    Ok(out)
}

#[cfg(feature = "store-std")]
fn load_record_optional<T: serde::de::DeserializeOwned>(
    target: &Path,
) -> Result<Option<T>, ProximaError> {
    let contents = match fs::read_to_string(target) {
        Ok(contents) => contents,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(io_error(target, "read", err)),
    };
    let value: T = toml::from_str(&contents).map_err(|err| {
        ProximaError::Config(format!("state store: parse {}: {err}", target.display()))
    })?;
    Ok(Some(value))
}

#[cfg(feature = "store-std")]
fn io_error(target: &Path, action: &str, err: std::io::Error) -> ProximaError {
    ProximaError::Config(format!("state store: {action} {}: {err}", target.display()))
}

// 12-hex-char suffix for tmp files. `random` doesn't need to be cryptographic;
// uniqueness across concurrent writers is the only requirement.
#[cfg(feature = "store-std")]
fn random_suffix() -> String {
    let bits = fastrand::u64(..);
    format!("{bits:016x}")
}

#[cfg(all(test, feature = "store-std"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    fn pipe(name: &str, kind: &str) -> PipeRecord {
        PipeRecord {
            name: name.into(),
            spec: json!({ "kind": kind }),
            requires: Vec::new(),
        }
    }

    fn listener(name: &str, bind: &str) -> ListenerRecord {
        ListenerRecord {
            name: name.into(),
            spec: json!({ "bind": bind }),
        }
    }

    #[test]
    fn open_creates_subdirs() {
        let dir = tempdir().expect("tempdir");
        let store = StateStore::open(dir.path()).expect("open");
        assert!(store.root().join(PIPES_SUBDIR).is_dir());
        assert!(store.root().join(LISTENERS_SUBDIR).is_dir());
    }

    #[test]
    fn put_then_get_pipe_round_trips() {
        let dir = tempdir().expect("tempdir");
        let store = StateStore::open(dir.path()).expect("open");
        let record = pipe("echo", "synth");
        store.put_pipe(&record).expect("put");

        let loaded = store.get_pipe("echo").expect("get").expect("present");
        assert_eq!(loaded, record);
    }

    #[test]
    fn list_pipes_returns_all_entries() {
        let dir = tempdir().expect("tempdir");
        let store = StateStore::open(dir.path()).expect("open");
        store.put_pipe(&pipe("a", "synth")).expect("put a");
        store.put_pipe(&pipe("b", "kv")).expect("put b");

        let listed = store.list_pipes().expect("list");
        assert_eq!(listed.len(), 2);
        assert_eq!(listed.get("a").expect("a").name, "a");
        assert_eq!(listed.get("b").expect("b").name, "b");
    }

    #[test]
    fn remove_pipe_returns_true_then_false() {
        let dir = tempdir().expect("tempdir");
        let store = StateStore::open(dir.path()).expect("open");
        store.put_pipe(&pipe("doomed", "synth")).expect("put");
        assert!(store.remove_pipe("doomed").expect("first remove"));
        assert!(!store.remove_pipe("doomed").expect("second remove"));
        assert!(
            store
                .get_pipe("doomed")
                .expect("get after remove")
                .is_none()
        );
    }

    #[test]
    fn put_listener_round_trips_and_lists() {
        let dir = tempdir().expect("tempdir");
        let store = StateStore::open(dir.path()).expect("open");
        let record = listener("public", "0.0.0.0:8080");
        store.put_listener(&record).expect("put");
        let loaded = store.get_listener("public").expect("get").expect("present");
        assert_eq!(loaded, record);
        let listed = store.list_listeners().expect("list");
        assert_eq!(listed.len(), 1);
    }

    #[test]
    fn open_reloads_existing_records() {
        let dir = tempdir().expect("tempdir");
        {
            let store = StateStore::open(dir.path()).expect("open first");
            store.put_pipe(&pipe("persistent", "synth")).expect("put");
        }
        let store_again = StateStore::open(dir.path()).expect("reopen");
        assert!(store_again.get_pipe("persistent").expect("get").is_some());
    }

    #[test]
    fn empty_name_is_rejected() {
        let dir = tempdir().expect("tempdir");
        let store = StateStore::open(dir.path()).expect("open");
        let outcome = store.put_pipe(&PipeRecord {
            name: String::new(),
            spec: json!({}),
            requires: Vec::new(),
        });
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[test]
    fn tmp_files_left_behind_are_ignored_on_list() {
        // simulate a writer that died mid-write — a stray .tmp-* file
        // must not surface as a pipe record on subsequent reads.
        let dir = tempdir().expect("tempdir");
        let store = StateStore::open(dir.path()).expect("open");
        let stray = store
            .root()
            .join(PIPES_SUBDIR)
            .join(".tmp-deadbeef-cafebabe");
        fs::write(&stray, "garbage = junk").expect("write stray");
        let listed = store.list_pipes().expect("list");
        assert!(listed.is_empty(), "stray .tmp-* ignored");
    }

    #[test]
    fn default_path_honors_env_override() {
        temp_env::with_var("PROXIMA_STATE_DIR", Some("/custom/state.d"), || {
            let path = StateStore::default_path().expect("default path");
            assert_eq!(path, PathBuf::from("/custom/state.d"));
        });
    }

    #[test]
    fn corrupted_file_surfaces_typed_error() {
        let dir = tempdir().expect("tempdir");
        let store = StateStore::open(dir.path()).expect("open");
        let path = pipe_path(store.root(), "broken");
        fs::write(&path, "this = is = not = toml =").expect("write garbage");
        let outcome = store.get_pipe("broken");
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }
}
