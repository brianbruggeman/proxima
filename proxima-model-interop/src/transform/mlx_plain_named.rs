//! The plain export of a GGUF as Hugging Face named tensors: every stored
//! leaf renamed by [`super::dense_names`], with the halves of a fused HF
//! tensor interleaved per expert into the one `[experts, parts * rows, in]`
//! tensor the loader cuts back with `TensorAlias::Part`.
//!
//! A leaf in the binding profile's `extra` list is a constant table the GGUF
//! carries and a Hugging Face directory does not (`rope_freqs.weight`), so it
//! is not exported. [`export_plain_named`] is
//! [`super::mlx_weight_plan::WeightPlan`] without a quantization, converted
//! eagerly.

use alloc::vec::Vec;

use proxima_gguf::pipe::ParsedGguf;
use proxima_numeric::DType;

use super::mlx_names::{Layout, MlxNames, dense_names};
use super::mlx_plain::PlainTensor;
use super::mlx_weight_plan::{WeightPlan, available_workers};
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
    WeightPlan::new(parsed, file_bytes, dtype, binding, layout, hf_prefix, None)?
        .convert_all(available_workers())
}

pub(super) fn leaf_names(
    parsed: &ParsedGguf,
    binding: &BindingProfile,
    layout: Layout,
    hf_prefix: &str,
) -> Result<MlxNames, InteropError> {
    let directory: Vec<&str> = parsed
        .tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect();
    let leaves: Vec<&str> = directory
        .iter()
        .copied()
        .filter(|name| !binding.claims_extra(name))
        .collect();
    let names = dense_names(&leaves, &directory, binding, layout, hf_prefix);
    match names.unmapped.is_empty() {
        true => Ok(names),
        false => Err(InteropError::MlxUnmappedLeaves {
            leaves: names.unmapped.join(", "),
        }),
    }
}
