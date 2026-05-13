//! A mmap'd persistent-memory region (Linux), via rustix's `linux_raw` backend.
//!
//! Pure Rust, zero C linked: `mmap`/`munmap` are rustix calls, not libc.

use core::ffi::c_void;
use core::slice;
use std::fs::{File, OpenOptions};
use std::os::fd::AsFd;
use std::path::Path;

use rustix::mm::{MapFlags, ProtFlags, mmap, munmap};

use crate::dax::config::PersistMode;
use crate::dax::error::DaxError;

/// A `len`-byte `MAP_SHARED` mapping plus the `File` keeping its fd alive for the
/// mapping's lifetime. Unmapped on drop.
pub struct MappedRegion {
    base: *mut u8,
    len: usize,
    mode: PersistMode,
    _file: File,
}

impl MappedRegion {
    /// Map `len` bytes of `path`. For [`PersistMode::FileBacked`] the file is
    /// grown to at least `len` first; a `/dev/dax` device has a fixed size and is
    /// mapped as-is.
    pub fn open(path: &Path, len: usize, mode: PersistMode) -> Result<Self, DaxError> {
        // create-if-missing but never truncate: reopening must preserve the
        // committed region so recover() can read it back.
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        if mode == PersistMode::FileBacked && file.metadata()?.len() < len as u64 {
            file.set_len(len as u64)?;
        }
        // SAFETY: a null hint lets the kernel pick the address; len > 0; the fd is
        // a freshly opened read/write file kept alive in `_file`. SHARED so stores
        // reach the backing object.
        let base = unsafe {
            mmap(
                core::ptr::null_mut(),
                len,
                ProtFlags::READ | ProtFlags::WRITE,
                MapFlags::SHARED,
                file.as_fd(),
                0,
            )
        }
        .map_err(DaxError::Mmap)?
        .cast::<u8>();
        Ok(Self {
            base,
            len,
            mode,
            _file: file,
        })
    }

    /// Mapping length in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the mapping is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The durability mode this region was opened with.
    #[must_use]
    pub fn mode(&self) -> PersistMode {
        self.mode
    }

    /// Read-only view of the whole mapping.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: base..len is a live mapping for the lifetime of self.
        unsafe { slice::from_raw_parts(self.base, self.len) }
    }

    /// Mutable view of the whole mapping.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: base..len is a live mapping; &mut self gives exclusive access.
        unsafe { slice::from_raw_parts_mut(self.base, self.len) }
    }

    pub(crate) fn raw(&self) -> (*mut u8, usize, PersistMode) {
        (self.base, self.len, self.mode)
    }
}

impl Drop for MappedRegion {
    fn drop(&mut self) {
        // SAFETY: base..len was returned by mmap above and is unmapped once.
        unsafe {
            let _ = munmap(self.base.cast::<c_void>(), self.len);
        }
    }
}
