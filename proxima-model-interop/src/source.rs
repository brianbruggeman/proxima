//! Owned checkpoint mappings for configured local storage paths.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use memmap2::MmapOptions;
use proxima_storage::dax::{MappedRegion, PersistMode};
use thiserror::Error;

use crate::mapped_residency::MappedResidency;
use crate::memory_reclaim::{monotonic_now, release_mapping};
use crate::serving_settings::ServingSettings;
use crate::{InteropError, LoadedModel};

/// Errors opening a checkpoint through the configured storage source.
#[derive(Debug, Error)]
pub enum CheckpointSourceError {
    #[error("checkpoint source io: {0}")]
    Io(#[from] std::io::Error),
    #[error("checkpoint source dax: {0}")]
    Dax(#[from] proxima_storage::dax::DaxError),
    #[error(transparent)]
    Interop(#[from] InteropError),
}

/// An owned, byte-addressable checkpoint mapping.
pub enum CheckpointMapping {
    /// Read-only private mapping backed by the normal filesystem/page cache.
    File { file: File, mapping: memmap2::Mmap },
    /// Mapping supplied by the DAX/PMEM facade.
    #[cfg(target_os = "linux")]
    Pmem(MappedRegion),
}

impl CheckpointMapping {
    /// Open a normal read-only filesystem mapping.
    pub fn open_file(path: &Path) -> Result<Self, CheckpointSourceError> {
        let file = File::open(path)?;
        // SAFETY: the file remains owned by this value and is never mutated by
        // the mapping; the caller receives only the mapping's immutable bytes.
        let mapping = unsafe { MmapOptions::new().map(&file)? };
        Ok(Self::File { file, mapping })
    }

    /// Open a mapping through the Linux DAX/PMEM facade.
    #[cfg(target_os = "linux")]
    pub fn open_pmem(path: &Path) -> Result<Self, CheckpointSourceError> {
        let length = std::fs::metadata(path)?.len();
        let length = usize::try_from(length).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "checkpoint is too large")
        })?;
        let mapping = MappedRegion::open(path, length, PersistMode::Dax)?;
        Ok(Self::Pmem(mapping))
    }

    /// Drops the weight mapping's pages from the resident set; the next read refaults them from the file. A DAX mapping has no page cache to drop.
    pub fn release_pages(&self) -> Result<(), InteropError> {
        match self {
            Self::File { mapping, .. } => release_mapping(mapping),
            #[cfg(target_os = "linux")]
            Self::Pmem(_) => Ok(()),
        }
    }

    /// The file mapping to register with a `MappedResidency`; a DAX mapping has none.
    #[must_use]
    pub fn file_mapping(&self) -> Option<&memmap2::Mmap> {
        match self {
            Self::File { mapping, .. } => Some(mapping),
            #[cfg(target_os = "linux")]
            Self::Pmem(_) => None,
        }
    }

    /// The complete checkpoint byte view used by GGUF parsing and binding.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        match self {
            Self::File { mapping, .. } => mapping,
            #[cfg(target_os = "linux")]
            Self::Pmem(mapping) => mapping.as_slice(),
        }
    }

    /// Loads like [`Self::load`], then installs the cold tier when `settings.reclaim.cold_dir` is set.
    pub fn load_serving(&self, settings: &ServingSettings) -> Result<LoadedModel<'_>, CheckpointSourceError> {
        let residency = Arc::new(MappedResidency::new(settings.reclaim.clone()));
        if let Some(mapping) = self.file_mapping() {
            residency.register_weights(mapping, monotonic_now());
        }
        let model = self.load()?.with_residency(residency);
        model.install_cold_tier(&settings.reclaim)?;
        Ok(model)
    }

    /// Parse and bind this mapping without copying its checkpoint bytes.
    pub fn load(&self) -> Result<LoadedModel<'_>, CheckpointSourceError> {
        let parsed = proxima_gguf::parse_complete(self.bytes()).map_err(InteropError::from)?;
        Ok(LoadedModel::load(&parsed, self.bytes())?)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::CheckpointMapping;

    #[test]
    fn file_mapping_exposes_exact_checkpoint_bytes() {
        let directory = tempdir().expect("create checkpoint source directory");
        let path = directory.path().join("checkpoint.gguf");
        let expected = b"GGUF bytes remain byte-identical through the source seam";
        fs::write(&path, expected).expect("write checkpoint source");

        let source = CheckpointMapping::open_file(&path).expect("open file source");

        assert_eq!(source.bytes(), expected);
    }
}
