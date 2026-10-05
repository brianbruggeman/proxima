use std::path::{Path, PathBuf};

use super::InteropError;

pub fn write_block_file(directory: &Path, file_key: u64, encoded: &[u8]) -> Result<PathBuf, InteropError> {
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
