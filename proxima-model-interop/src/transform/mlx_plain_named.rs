//! The plain export of a GGUF as Hugging Face named tensors: every stored
//! leaf through [`super::export_plain`], renamed by [`super::dense_names`],
//! with the halves of a fused HF tensor interleaved per expert into the one
//! `[experts, parts * rows, in]` tensor the loader cuts back with
//! `TensorAlias::Part`.
//!
//! A leaf in the binding profile's `extra` list is a constant table the GGUF
//! carries and a Hugging Face directory does not (`rope_freqs.weight`), so it
//! is not exported.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use proxima_gguf::pipe::ParsedGguf;
use proxima_tensor::DType;

use super::mlx_names::{Layout, MlxNames, dense_names};
use super::mlx_plain::{PlainTensor, export_plain};
use crate::error::InteropError;
use crate::profiles::BindingProfile;

/// Every exportable tensor of `parsed` as `dtype`, named for `layout` under
/// `hf_prefix` (e.g. `model.` for a llama-style decoder).
///
/// # Errors
///
/// [`InteropError::MlxUnmappedLeaves`] when a leaf has no stored tensor and no
/// layout row; whatever [`super::export_plain`] can fail with.
pub fn export_plain_named(
    parsed: &ParsedGguf,
    file_bytes: &[u8],
    dtype: DType,
    binding: &BindingProfile,
    layout: Layout,
    hf_prefix: &str,
) -> Result<Vec<PlainTensor>, InteropError> {
    let directory: Vec<&str> = parsed
        .tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect();
    let leaves: Vec<&str> = directory
        .iter()
        .copied()
        .filter(|name| !binding.extra.iter().any(|extra| extra == name))
        .collect();
    let names = dense_names(&leaves, &directory, binding, layout, hf_prefix);
    if !names.unmapped.is_empty() {
        return Err(InteropError::MlxUnmappedLeaves {
            leaves: names.unmapped.join(", "),
        });
    }
    let mut by_leaf: BTreeMap<_, _> = export_plain(parsed, file_bytes, dtype)?
        .into_iter()
        .map(|tensor| (tensor.name.clone(), tensor))
        .collect();
    let mut named = rename_mapped(&names, &mut by_leaf);
    named.extend(fuse_halves(&names, &mut by_leaf));
    Ok(named)
}

fn rename_mapped(
    names: &MlxNames,
    by_leaf: &mut BTreeMap<alloc::string::String, PlainTensor>,
) -> Vec<PlainTensor> {
    names
        .mapped
        .iter()
        .filter_map(|(leaf, hf_name)| {
            by_leaf.remove(leaf).map(|mut tensor| {
                tensor.name.clone_from(hf_name);
                tensor
            })
        })
        .collect()
}

fn fuse_halves(
    names: &MlxNames,
    by_leaf: &mut BTreeMap<alloc::string::String, PlainTensor>,
) -> Vec<PlainTensor> {
    let targets: BTreeSet<&str> = names.fused.iter().map(|item| item.hf.as_str()).collect();
    targets
        .into_iter()
        .filter_map(|target| {
            let mut group: Vec<_> = names
                .fused
                .iter()
                .filter(|item| item.hf == target)
                .collect();
            group.sort_by_key(|item| item.part);
            let parts: Option<Vec<PlainTensor>> = group
                .iter()
                .map(|item| by_leaf.remove(&item.leaf))
                .collect();
            parts.map(|parts| interleave(target, &parts))
        })
        .collect()
}

fn interleave(name: &str, parts: &[PlainTensor]) -> PlainTensor {
    let experts = usize::try_from(parts[0].shape[0]).unwrap_or(1);
    let chunk = parts[0].bytes.len() / experts;
    let mut bytes = Vec::with_capacity(chunk * experts * parts.len());
    (0..experts).for_each(|expert| {
        parts.iter().for_each(|part| {
            bytes.extend_from_slice(&part.bytes[expert * chunk..(expert + 1) * chunk])
        });
    });
    let mut shape = parts[0].shape.clone();
    shape[1] *= parts.len() as u64;
    PlainTensor {
        name: name.into(),
        shape,
        dtype: parts[0].dtype,
        bytes,
    }
}
