use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

mod error;

pub use error::StoreError;

const OVERRIDE_VAR: &str = "PROXIMA_CUDA_KERNEL_CACHE_DIR";
const DISABLED_VALUE: &str = "off";
const DEFAULT_RELATIVE_PATH: &str = ".cache/proxima/cuda-ptx";

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

/// Compiled CUDA PTX kept on disk, one file per opaque hex name.
///
/// `Disabled` never touches the filesystem. `Directory` owns every file it
/// writes; a name is only ever a file directly inside that directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PtxStore {
    Disabled,
    Directory(PathBuf),
}

impl PtxStore {
    /// Resolves the store from the process environment.
    ///
    /// # Errors
    ///
    /// [`StoreError::NoHome`] when the override is unset and `HOME` is not set.
    pub fn from_env() -> Result<Self, StoreError> {
        Self::resolve(
            std::env::var_os(OVERRIDE_VAR).as_deref(),
            std::env::var_os("HOME").as_deref(),
        )
    }

    /// Resolves the store from an explicit override and home directory.
    ///
    /// An empty or `off` override disables the store; any other override is
    /// the directory itself; without an override the store lives under `home`.
    ///
    /// # Errors
    ///
    /// [`StoreError::NoHome`] when there is no override and no `home`.
    pub fn resolve(
        override_value: Option<&OsStr>,
        home: Option<&OsStr>,
    ) -> Result<Self, StoreError> {
        match override_value {
            Some(value) if value.is_empty() || value == OsStr::new(DISABLED_VALUE) => {
                Ok(Self::Disabled)
            }
            Some(value) => Ok(Self::Directory(PathBuf::from(value))),
            None => {
                let home = home.ok_or(StoreError::NoHome)?;
                Ok(Self::Directory(
                    PathBuf::from(home).join(DEFAULT_RELATIVE_PATH),
                ))
            }
        }
    }

    /// Reads the bytes stored under `name`; a missing file is `Ok(None)`.
    ///
    /// # Errors
    ///
    /// [`StoreError::InvalidName`] for a name that is not hex, and
    /// [`StoreError::Io`] when the file exists but cannot be read.
    pub fn get(&self, name: &str) -> Result<Option<Vec<u8>>, StoreError> {
        validate_name(name)?;
        let Some(root) = self.directory() else {
            return Ok(None);
        };
        let path = root.join(name);
        match fs::read(&path) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(source) => Err(StoreError::io("read", path, source)),
        }
    }

    /// Stores `bytes` under `name` by writing a sibling temp file and renaming
    /// it into place, so readers see either the old file or the whole new one.
    ///
    /// # Errors
    ///
    /// [`StoreError::InvalidName`] for a name that is not hex, and
    /// [`StoreError::Io`] for any filesystem failure, carrying the path.
    pub fn put(&self, name: &str, bytes: &[u8]) -> Result<(), StoreError> {
        validate_name(name)?;
        let Some(root) = self.directory() else {
            return Ok(());
        };
        fs::create_dir_all(root)
            .map_err(|source| StoreError::io("create", root.to_owned(), source))?;
        let temp = root.join(temp_name(name));
        write_temp(&temp, bytes)?;
        let path = root.join(name);
        fs::rename(&temp, &path).map_err(|source| {
            discard(&temp);
            StoreError::io("rename into the store", path, source)
        })
    }

    fn directory(&self) -> Option<&Path> {
        match self {
            Self::Disabled => None,
            Self::Directory(root) => Some(root),
        }
    }
}

fn validate_name(name: &str) -> Result<(), StoreError> {
    let valid = !name.is_empty() && name.bytes().all(|byte| byte.is_ascii_hexdigit());
    if valid {
        Ok(())
    } else {
        Err(StoreError::InvalidName {
            name: name.to_owned(),
        })
    }
}

fn temp_name(name: &str) -> String {
    let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
    format!("{name}.tmp.{}.{sequence}", std::process::id())
}

fn write_temp(temp: &Path, bytes: &[u8]) -> Result<(), StoreError> {
    let written = File::create(temp).and_then(|mut file| {
        file.write_all(bytes)?;
        file.sync_all()
    });
    written.map_err(|source| {
        discard(temp);
        StoreError::io("write", temp.to_owned(), source)
    })
}

fn discard(temp: &Path) {
    match fs::remove_file(temp) {
        Ok(()) => {}
        Err(source) if source.kind() == io::ErrorKind::NotFound => {}
        Err(source) => {
            proxima_telemetry::warn!(
                path = ?temp,
                error = %source,
                "could not remove a partial kernel cache write"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;
    use std::sync::Barrier;

    use super::*;

    const HEX_NAME: &str = "9f2c41d07ab8e35c";
    const PTX_V1: &[u8] =
        b"//\n// Generated by NVIDIA NVVM Compiler\n.version 8.0\n.target sm_75\n";

    #[test]
    fn put_then_get_returns_the_same_bytes() {
        let dir = tempfile::tempdir().expect("tempdir for store root");
        let store =
            PtxStore::resolve(Some(dir.path().as_os_str()), None).expect("override resolves");

        store.put(HEX_NAME, PTX_V1).expect("put writes the entry");

        assert_eq!(
            store.get(HEX_NAME).expect("get reads"),
            Some(PTX_V1.to_vec())
        );
    }

    #[test]
    fn missing_name_and_missing_root_return_none() {
        let dir = tempfile::tempdir().expect("tempdir for store root");
        let root = dir.path().join("not-created-yet");
        let store = PtxStore::Directory(root.clone());

        assert_eq!(store.get(HEX_NAME).expect("get on absent root"), None);
        store.put(HEX_NAME, PTX_V1).expect("put creates the root");
        assert_eq!(store.get("0000ffff").expect("get other name"), None);
        assert!(root.join(HEX_NAME).is_file(), "put created the entry file");
    }

    #[test]
    fn disabled_override_never_writes_and_never_reads() {
        let home = tempfile::tempdir().expect("tempdir for home");
        let home_os = home.path().as_os_str();
        for disabled in [OsStr::new("off"), OsStr::new("")] {
            temp_env::with_vars(
                [(OVERRIDE_VAR, Some(disabled)), ("HOME", Some(home_os))],
                || {
                    let store = PtxStore::from_env().expect("disabled env resolves");
                    assert_eq!(store, PtxStore::Disabled, "override {disabled:?} disables");
                    store.put(HEX_NAME, PTX_V1).expect("disabled put succeeds");
                    assert_eq!(store.get(HEX_NAME).expect("disabled get"), None);
                },
            );
        }
        assert!(
            !home.path().join(".cache").exists(),
            "disabled store created no directory under HOME"
        );
    }

    #[test]
    fn unset_override_defaults_under_home_cache() {
        let home = tempfile::tempdir().expect("tempdir for home");
        temp_env::with_vars(
            [
                (OVERRIDE_VAR, None::<&OsStr>),
                ("HOME", Some(home.path().as_os_str())),
            ],
            || {
                let store = PtxStore::from_env().expect("default resolves under HOME");
                assert_eq!(
                    store,
                    PtxStore::Directory(home.path().join(DEFAULT_RELATIVE_PATH)),
                );
            },
        );
    }

    #[test]
    fn unset_override_without_home_is_a_typed_error() {
        let error = PtxStore::resolve(None, None).expect_err("no override and no HOME");
        assert!(matches!(error, StoreError::NoHome), "got {error:?}");
    }

    #[test]
    fn non_hex_names_are_rejected_before_any_filesystem_access() {
        let dir = tempfile::tempdir().expect("tempdir for store root");
        let root = dir.path().join("store");
        let store = PtxStore::Directory(root.clone());
        let hostile = ["../escape", "a/b", "..", "", "xyz", "ab.cd", "9f/../x"];

        for name in hostile {
            let put = store.put(name, PTX_V1);
            assert!(
                matches!(put, Err(StoreError::InvalidName { .. })),
                "put {name:?}: {put:?}"
            );
            let get = store.get(name);
            assert!(
                matches!(get, Err(StoreError::InvalidName { .. })),
                "get {name:?}: {get:?}"
            );
        }
        assert!(!root.exists(), "rejected names created the store root");
        assert!(
            !dir.path().join("escape").exists(),
            "rejected names escaped the root"
        );
    }

    #[test]
    fn leftover_temp_file_from_a_crashed_write_is_never_returned() {
        let dir = tempfile::tempdir().expect("tempdir for store root");
        let store = PtxStore::Directory(dir.path().to_owned());
        let partial = &PTX_V1[..10];
        fs::write(dir.path().join(format!("{HEX_NAME}.tmp.999.0")), partial)
            .expect("write a crashed temp file");

        assert_eq!(store.get(HEX_NAME).expect("get ignores temp"), None);

        store.put(HEX_NAME, PTX_V1).expect("put after the crash");
        assert_eq!(
            store.get(HEX_NAME).expect("get complete entry"),
            Some(PTX_V1.to_vec())
        );
    }

    #[test]
    fn concurrent_puts_of_one_name_leave_one_valid_file() {
        let dir = tempfile::tempdir().expect("tempdir for store root");
        let store = PtxStore::Directory(dir.path().to_owned());
        let payload_a = vec![b'a'; 4096];
        let payload_b = vec![b'b'; 8192];
        let barrier = Barrier::new(2);

        std::thread::scope(|scope| {
            for payload in [&payload_a, &payload_b] {
                let store = &store;
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    store
                        .put(HEX_NAME, payload)
                        .expect("concurrent put succeeds");
                });
            }
        });

        let stored = store.get(HEX_NAME).expect("get after both puts");
        assert!(
            stored == Some(payload_a.clone()) || stored == Some(payload_b.clone()),
            "entry is neither payload whole: len {:?}",
            stored.as_ref().map(Vec::len)
        );
        let entries = fs::read_dir(dir.path()).expect("read store root").count();
        assert_eq!(entries, 1, "no temp files remain beside the entry");
    }

    #[test]
    fn root_that_is_a_file_is_a_typed_io_error_with_the_path() {
        let dir = tempfile::tempdir().expect("tempdir for store root");
        let file_root = dir.path().join("occupied");
        fs::write(&file_root, b"not a directory").expect("occupy the root path");
        let store = PtxStore::Directory(file_root.clone());

        let error = store
            .put(HEX_NAME, PTX_V1)
            .expect_err("a file cannot be a root");
        match error {
            StoreError::Io { path, .. } => assert_eq!(path, file_root),
            other => panic!("expected io error at the root, got {other:?}"),
        }
    }
}
