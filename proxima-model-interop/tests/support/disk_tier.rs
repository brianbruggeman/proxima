use std::path::{Path, PathBuf};

use super::{ColdTier, InteropError, MappedBlockFile, PrefixState, chained_keys, write_block_file};

pub struct DiskTier {
    directory: PathBuf,
    descriptor_digest: [u8; 16],
}

impl DiskTier {
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

impl ColdTier for DiskTier {
    fn demote(&self, stamp: u64, ids: &[u32], state: &PrefixState) -> Result<u64, InteropError> {
        let content_key = chained_keys(ids, ids.len(), 0)
            .first()
            .copied()
            .ok_or(InteropError::BlockFileMalformed {
                reason: "entry has no ids",
            })?;
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

    fn promote(&self, stamp: u64, ids: &[u32], cached_len: usize) -> Result<PrefixState, InteropError> {
        let mapped = MappedBlockFile::open(&self.path_of(stamp))?;
        let view = mapped.view()?;
        view.require_digest(self.descriptor_digest)?;
        PrefixState::from_block_file(ids.to_vec(), cached_len, &view)
    }
}

pub fn spilled_names(directory: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(directory)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}
