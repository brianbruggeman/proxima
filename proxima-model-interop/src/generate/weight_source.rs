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
    BoundProgram, KvLayout, bind_checkpoint_with_kv_layout_and_precision, bind_hf_checkpoint,
    bind_prefill_row_cut_with_precision, bind_speculative_verify_with_precision,
    recurrent_speculative_verify,
};
use crate::serving::WeightPrecisionRule;

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
        weight_precision: &[WeightPrecisionRule<'_>],
    ) -> Result<BoundProgram<'file>, InteropError> {
        match self {
            Self::Gguf => {
                let bound = bind_checkpoint_with_kv_layout_and_precision(
                    parsed,
                    file_bytes,
                    kv_layout,
                    weight_precision,
                )?;
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
        weight_precision: &[WeightPrecisionRule<'_>],
    ) -> Result<Option<BoundProgram<'file>>, InteropError> {
        match self {
            Self::Gguf => bind_speculative_verify_with_precision(
                parsed,
                file_bytes,
                kv_layout,
                descriptor,
                weight_precision,
            ),
            Self::Directory { .. } => Ok(None),
        }
    }

    pub(super) fn prefill_row_cut(
        &self,
        parsed: &ParsedGguf,
        file_bytes: &'file [u8],
        kv_layout: KvLayout,
        descriptor: Option<&ModelDescriptor>,
        weight_precision: &[WeightPrecisionRule<'_>],
    ) -> Result<Option<BoundProgram<'file>>, InteropError> {
        match self {
            Self::Gguf => bind_prefill_row_cut_with_precision(
                parsed,
                file_bytes,
                kv_layout,
                descriptor,
                weight_precision,
            ),
            Self::Directory { .. } => Ok(None),
        }
    }

    pub(super) fn recurrent_speculative_verify(
        &self,
        parsed: &ParsedGguf,
        kv_layout: KvLayout,
        descriptor: Option<&ModelDescriptor>,
    ) -> Result<Option<ModelDescriptor>, InteropError> {
        match self {
            Self::Gguf => recurrent_speculative_verify(parsed, kv_layout, descriptor),
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
    pub(super) fn advise_other_shards(&self, registered: &[u8]) {
        let others: &[HfShard<'file>] = match self {
            Self::Gguf => &[],
            Self::Directory { shards, .. } => shards,
        };
        others
            .iter()
            .filter(|shard| !core::ptr::eq(shard.1, registered))
            .for_each(|shard| crate::mapping_residency::advise_will_need(shard.1));
    }

    pub(super) fn checkpoint_bytes(&self, file_bytes: &[u8]) -> usize {
        match self {
            Self::Gguf => file_bytes.len(),
            Self::Directory { shards, .. } => shards.iter().map(|shard| shard.1.len()).sum(),
        }
    }
}
