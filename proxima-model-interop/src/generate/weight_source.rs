//! Where a [`super::LoadedModel`]'s weights and vocabulary come from: the
//! GGUF the metadata was read from, or a Hugging Face directory whose
//! tensors bind under the family's binding profile while the same GGUF
//! header supplies the descriptor and hyperparameters.
//!
//! Teaching pointer: [`super::LoadedModel::load`] is the `Gguf` arm and
//! [`super::LoadedModel::load_from_hf_directory`] the `Directory` arm; both
//! run the one `load_inner` pipeline and differ only at the calls below.

use proxima_gguf::pipe::ParsedGguf;
use proxima_tensor::spec::ModelDescriptor;

use crate::error::InteropError;
use crate::hf_bind::HfShard;
use crate::lowering::{
    BoundProgram, KvLayout, bind_checkpoint_with_kv_layout, bind_hf_checkpoint,
    bind_speculative_verify,
};

pub(super) enum WeightSource<'shards, 'file> {
    Gguf,
    Directory {
        header_bytes: &'file [u8],
        shards: &'shards [HfShard<'file>],
    },
}

impl<'file> WeightSource<'_, 'file> {
    pub(super) fn bind(
        &self,
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
        kv_layout: KvLayout,
        descriptor: Option<&ModelDescriptor>,
    ) -> Result<BoundProgram<'file>, InteropError> {
        match self {
            Self::Gguf => {
                let bound = bind_checkpoint_with_kv_layout(parsed, file_bytes, kv_layout)?;
                match descriptor {
                    Some(config) => bound.lowered_from(config, parsed, file_bytes),
                    None => Ok(bound),
                }
            }
            Self::Directory {
                header_bytes,
                shards,
                ..
            } => bind_hf_checkpoint(parsed, header_bytes, kv_layout, shards),
        }
    }

    pub(super) fn speculative_verify(
        &self,
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
        kv_layout: KvLayout,
        descriptor: Option<&ModelDescriptor>,
    ) -> Result<Option<BoundProgram<'file>>, InteropError> {
        match self {
            Self::Gguf => bind_speculative_verify(parsed, file_bytes, kv_layout, descriptor),
            Self::Directory { .. } => Ok(None),
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(super) fn mapped_bytes(&self, file_bytes: &[u8]) -> u64 {
        match self {
            Self::Gguf => file_bytes.len() as u64,
            Self::Directory { shards, .. } => shards.iter().map(|shard| shard.1.len() as u64).sum(),
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(super) fn weight_class_bytes(
        &self,
        parsed: &ParsedGguf,
        file_bytes: &[u8],
    ) -> (u64, u64, u64) {
        match self {
            Self::Gguf => crate::bind::tensor_bytes_by_class(parsed),
            Self::Directory { .. } => (self.mapped_bytes(file_bytes), 0, 0),
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(super) fn prove_other_shards_resident(
        &self,
        registered: &[u8],
    ) -> Result<crate::mapping_residency::ResidencyRung, InteropError> {
        let others: &[HfShard<'file>] = match self {
            Self::Gguf => &[],
            Self::Directory { shards, .. } => shards,
        };
        others
            .iter()
            .filter(|shard| !core::ptr::eq(shard.1, registered))
            .try_fold(
                crate::mapping_residency::ResidencyRung::Prefault,
                |rung, shard| {
                    crate::mapping_residency::prove_resident(shard.1)
                        .map(|report| rung.max(report.rung))
                },
            )
    }

    pub(super) fn checkpoint_bytes(&self, file_bytes: &[u8]) -> usize {
        match self {
            Self::Gguf => file_bytes.len(),
            Self::Directory { shards, .. } => shards.iter().map(|shard| shard.1.len()).sum(),
        }
    }
}
