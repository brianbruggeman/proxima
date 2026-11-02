//! Directory-backed [`ColdTier`]: one block file per demoted prompt-cache
//! entry, written tmp-then-rename so a crash never leaves a torn file, and
//! promoted back through a read-only mapping.

use std::path::{Path, PathBuf};

use xxhash_rust::xxh3::xxh3_64_with_seed;

use crate::block_file::MappedBlockFile;
use crate::{ColdTier, InteropError, PrefixState};

pub struct DiskColdTier {
    directory: PathBuf,
    descriptor_digest: [u8; 16],
}

impl DiskColdTier {
    pub fn open(directory: &Path, descriptor_digest: [u8; 16]) -> Result<Self, InteropError> {
        std::fs::create_dir_all(directory).map_err(|source| InteropError::BlockFileIo {
            path: directory.to_path_buf(),
            source,
        })?;
        Ok(Self {
            directory: directory.to_path_buf(),
            descriptor_digest,
        })
    }

    fn path_of(&self, stamp: u64) -> PathBuf {
        self.directory.join(format!("{stamp:016x}.pxkv"))
    }
}

impl ColdTier for DiskColdTier {
    fn demote(&self, stamp: u64, ids: &[u32], state: &PrefixState) -> Result<u64, InteropError> {
        let content_key = chained_keys(ids, ids.len(), 0).first().copied().ok_or(
            InteropError::BlockFileMalformed {
                reason: "entry has no ids",
            },
        )?;
        let mut bytes = Vec::new();
        state.to_block_file(self.descriptor_digest, content_key, &mut bytes)?;
        write_block_file(&self.directory, stamp, &bytes)?;
        Ok(bytes.len() as u64)
    }

    fn discard(&self, stamp: u64) -> Result<(), InteropError> {
        let path = self.path_of(stamp);
        match std::fs::remove_file(&path) {
            Err(source) if source.kind() != std::io::ErrorKind::NotFound => {
                Err(InteropError::BlockFileIo { path, source })
            }
            _ => Ok(()),
        }
    }

    fn promote(
        &self,
        stamp: u64,
        ids: &[u32],
        cached_len: usize,
    ) -> Result<PrefixState, InteropError> {
        let mapped = MappedBlockFile::open(&self.path_of(stamp))?;
        let view = mapped.view()?;
        view.require_digest(self.descriptor_digest)?;
        PrefixState::from_block_file(ids.to_vec(), cached_len, &view)
    }
}

pub fn write_block_file(
    directory: &Path,
    file_key: u64,
    encoded: &[u8],
) -> Result<PathBuf, InteropError> {
    let final_path = directory.join(format!("{file_key:016x}.pxkv"));
    let temporary_path = directory.join(format!("{file_key:016x}.pxkv.tmp"));
    std::fs::write(&temporary_path, encoded).map_err(|source| InteropError::BlockFileIo {
        path: temporary_path.clone(),
        source,
    })?;
    std::fs::rename(&temporary_path, &final_path).map_err(|source| InteropError::BlockFileIo {
        path: final_path.clone(),
        source,
    })?;
    Ok(final_path)
}

// a hit must prove the whole prefix equal, so each key folds in the previous key; content_hashes stays position independent for chunk shifting
pub fn chained_keys(ids: &[u32], block_tokens: usize, seed: u64) -> Vec<u64> {
    if block_tokens == 0 {
        return Vec::new();
    }
    let mut buffer: Vec<u8> = Vec::with_capacity(8 + block_tokens * 4);
    let mut keys = Vec::with_capacity(ids.len() / block_tokens);
    for block in ids.chunks_exact(block_tokens) {
        let previous = keys.last().copied();
        buffer.clear();
        buffer.extend(previous.iter().flat_map(|key: &u64| key.to_le_bytes()));
        buffer.extend(block.iter().flat_map(|id| id.to_le_bytes()));
        keys.push(xxh3_64_with_seed(&buffer, seed));
    }
    keys
}
