//! Portable store-backed crash-consistent cell — the floor under the pmem-native
//! [`crate::dax::PmemCowStore`] fast tier.
//!
//! [`FileCell`] gives the *same* crash guarantee as `PmemCowStore` (a read after
//! a crash sees the complete old or new value, never torn) using a conventional
//! durable store: write the new value to a temp file, fsync it, atomically rename
//! it over the target, fsync the directory (the LMDB/LevelDB/SQLite atomic-commit
//! discipline). It works on ANY OS — no mmap, no rustix, no pmem hardware — holds
//! ANY size ("big stuff"), and is slower than the pmem-native path because it
//! rewrites the whole value per commit. That is the deliberate trade: a store
//! always works, just slower; real pmem (`PmemCowStore` over a DAX mapping) is
//! the byte-addressable fast tier with the identical guarantee.
//!
//! Mechanism note: the pmem-native path swaps an 8-byte root in place and relies
//! on the SNIA/Intel ADR power-fail atomicity of that store. A plain file gives
//! no such atomicity for an in-place word, so the store backend uses atomic
//! `rename` instead — same old-or-new guarantee, different mechanism.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// A durable single-value cell backed by a file, committed crash-consistently
/// via fsync + atomic rename. Portable (any OS) and unbounded in value size.
#[derive(Debug, Clone)]
pub struct FileCell {
    path: PathBuf,
}

impl FileCell {
    /// Create (or overwrite) a cell at `path` with `initial` as the first
    /// committed value.
    pub fn create(path: impl Into<PathBuf>, initial: &[u8]) -> io::Result<Self> {
        let path = path.into();
        write_atomic(&path, initial)?;
        Ok(Self { path })
    }

    /// Open an existing cell; the value is read on [`Self::read`] / [`Self::recover`].
    pub fn open(path: impl Into<PathBuf>) -> io::Result<Self> {
        let path = path.into();
        if !path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "cell does not exist",
            ));
        }
        Ok(Self { path })
    }

    /// Commit a new value, crash-consistently. After a crash mid-commit a later
    /// [`Self::recover`] returns either this value or the prior one.
    pub fn commit(&self, value: &[u8]) -> io::Result<()> {
        write_atomic(&self.path, value)
    }

    /// Read the live value (steady state).
    pub fn read(&self) -> io::Result<Vec<u8>> {
        fs::read(&self.path)
    }

    /// Recover the live value after a crash. The atomic rename guarantees a
    /// complete old-or-new value, so recovery is a plain read.
    pub fn recover(&self) -> io::Result<Vec<u8>> {
        self.read()
    }

    /// The backing file path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let temp = temp_path(path);
    {
        let mut file = File::create(&temp)?;
        file.write_all(bytes)?;
        // the temp bytes must be durable before the rename publishes them
        file.sync_all()?;
    }
    fs::rename(&temp, path)?;
    sync_parent_dir(path)
}

fn temp_path(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(".tmp");
    path.with_file_name(name)
}

// a renamed directory entry is not durable until the parent dir is fsync'd; a no-op
// where directory handles are not a portable concept (rename durability is the FS's)
#[cfg(unix)]
fn sync_parent_dir(path: &Path) -> io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    File::open(parent)?.sync_all()
}

#[cfg(not(unix))]
fn sync_parent_dir(_path: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn create_then_read_round_trips() {
        let dir = tempdir().expect("tempdir");
        let cell = FileCell::create(dir.path().join("cell.bin"), b"initial value").expect("create");
        assert_eq!(cell.read().expect("read"), b"initial value");
    }

    // the store-backed floor's reason to exist: durable across a process restart on
    // a normal filesystem (Apple included), no pmem hardware.
    #[test]
    fn commit_survives_drop_and_reopen() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("cell.bin");
        {
            let cell = FileCell::create(&path, b"v1").expect("create");
            cell.commit(b"v2-committed").expect("commit");
            assert_eq!(cell.read().expect("read"), b"v2-committed");
        } // dropped

        let reopened = FileCell::open(&path).expect("reopen");
        assert_eq!(reopened.recover().expect("recover"), b"v2-committed");
    }

    // "big stuff": the store backs an arbitrary-size value, unlike a fixed pmem slot.
    #[test]
    fn big_value_round_trips() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("big.bin");
        let big: Vec<u8> = (0..4 * 1024 * 1024)
            .map(|index| (index % 251) as u8)
            .collect();
        let cell = FileCell::create(&path, &big).expect("create");
        cell.commit(&big).expect("commit big");
        drop(cell);
        let reopened = FileCell::open(&path).expect("reopen");
        assert_eq!(reopened.recover().expect("recover"), big);
    }

    #[test]
    fn no_temp_file_remains_after_success() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("cell.bin");
        FileCell::create(&path, b"x").expect("create");
        assert!(
            !temp_path(&path).exists(),
            "the atomic rename must consume the temp file"
        );
    }

    // a leftover temp from a prior crashed commit must not affect the live value;
    // the next commit overwrites it and the live value stays old-or-new, never torn.
    #[test]
    fn stale_temp_does_not_corrupt_the_live_value() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("cell.bin");
        let cell = FileCell::create(&path, b"committed").expect("create");
        std::fs::write(temp_path(&path), b"garbage from a crashed commit")
            .expect("write stale tmp");

        assert_eq!(
            cell.read().expect("read"),
            b"committed",
            "stale temp must not affect the live value"
        );
        cell.commit(b"next").expect("commit over stale tmp");
        assert_eq!(cell.read().expect("read"), b"next");
    }

    #[test]
    fn open_missing_cell_is_not_found() {
        let dir = tempdir().expect("tempdir");
        let err = FileCell::open(dir.path().join("absent.bin")).expect_err("missing cell");
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }
}
