//! The files of one MLX model directory for a GGUF source: weight shards and
//! their index, `config.json`, `tokenizer.json` and `tokenizer_config.json`.
//!
//! [`mlx_directory_files`] is the sans-IO half: it composes
//! [`super::export_plain_named`] or [`super::export_affine_named`],
//! [`super::MlxExport::assemble`], [`super::hf_config_bytes_with_overrides`] and
//! [`super::export_tokenizer`] and returns `(file name, bytes)` pairs.
//! [`export_mlx_directory`] is the filesystem edge the CLI calls; compose the
//! pure half directly when the caller owns the destination.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use std::fs::{File, create_dir_all, write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use proxima_gguf::parse_complete;
use proxima_gguf::pipe::ParsedGguf;
use proxima_safetensors::TensorPayload;
use proxima_tensor::DType;

use super::mlx::{MLX_INDEX_FILE_NAME, MLX_MAX_SHARD_BYTES, MlxExport};
use super::mlx_affine::AffineParams;
use super::mlx_config::{family_export, hf_config_bytes_with_overrides};
use super::mlx_packed_named::export_affine_named;
use super::mlx_plain::PlainTensor;
use super::mlx_plain_named::export_plain_named;
use super::mlx_tokenizer::export_tokenizer;
use crate::error::InteropError;
use crate::profiles::binding_profile;

const CONFIG_FILE_NAME: &str = "config.json";
const TOKENIZER_FILE_NAME: &str = "tokenizer.json";
const TOKENIZER_CONFIG_FILE_NAME: &str = "tokenizer_config.json";

/// Every file of the MLX directory for `parsed`, quantized with `quantization`
/// (plain float tensors as `dtype` when `None`).
///
/// # Errors
///
/// [`InteropError::UnknownFamilyConfig`] for a family without an MLX layout;
/// otherwise whatever the composed export, config and tokenizer writers fail
/// with.
pub fn mlx_directory_files(
    parsed: &ParsedGguf,
    file_bytes: &[u8],
    quantization: Option<AffineParams>,
    dtype: DType,
) -> Result<Vec<(String, Vec<u8>)>, InteropError> {
    let (layout, hf_prefix, conventions, hparams) = family_export(parsed)?;
    let binding = binding_profile(&hparams.family)?;
    let tensors = match quantization {
        Some(params) => export_affine_named(
            parsed,
            file_bytes,
            params,
            dtype,
            &binding,
            layout,
            hf_prefix,
            conventions,
        )?,
        None => export_plain_named(parsed, file_bytes, dtype, &binding, layout, hf_prefix)?,
    };
    let overrides = quantization.map_or_else(Vec::new, |params| {
        conventions.overrides(tensors.iter().map(|tensor| tensor.name.as_str()), params)
    });
    let mut files = weight_files(&tensors)?;
    drop(tensors);
    let marker = quantization.map(config_marker).transpose()?;
    files.push((
        CONFIG_FILE_NAME.to_string(),
        hf_config_bytes_with_overrides(&hparams, parsed, marker, &overrides)?,
    ));
    let tokenizer = export_tokenizer(parsed)?;
    files.push((TOKENIZER_FILE_NAME.to_string(), tokenizer.tokenizer_json));
    files.push((
        TOKENIZER_CONFIG_FILE_NAME.to_string(),
        tokenizer.tokenizer_config_json,
    ));
    Ok(files)
}

fn config_marker(params: AffineParams) -> Result<(u8, u32), InteropError> {
    let group_size = params.group_size();
    u32::try_from(group_size)
        .map(|group| (params.bits(), group))
        .map_err(|_| InteropError::AffineGroupSizeUnsupported { group_size })
}

fn weight_files(tensors: &[PlainTensor]) -> Result<Vec<(String, Vec<u8>)>, InteropError> {
    let payloads = tensors
        .iter()
        .map(|tensor| TensorPayload {
            name: tensor.name.clone(),
            dtype: tensor.dtype,
            shape: tensor.shape.clone(),
            data: &tensor.bytes,
        })
        .collect();
    let export = MlxExport::assemble(payloads, MLX_MAX_SHARD_BYTES)?;
    let index = export
        .index
        .map(|bytes| (MLX_INDEX_FILE_NAME.to_string(), bytes));
    let shards = export
        .shards
        .into_iter()
        .map(|shard| (shard.file_name, shard.bytes));
    Ok(shards.chain(index).collect())
}

/// Maps the GGUF at `gguf_path` and writes its MLX directory under `out_dir`
/// (created if absent, existing files overwritten); returns the paths written.
///
/// # Errors
///
/// [`InteropError::MlxDirectoryIo`] naming the path that could not be opened,
/// mapped, created or written; otherwise as [`mlx_directory_files`].
pub fn export_mlx_directory(
    gguf_path: &Path,
    out_dir: &Path,
    quantization: Option<AffineParams>,
    dtype: DType,
) -> Result<Vec<PathBuf>, InteropError> {
    let wrap = |path: &Path| {
        let path = path.to_path_buf();
        move |source| InteropError::MlxDirectoryIo { path, source }
    };
    let file = File::open(gguf_path).map_err(wrap(gguf_path))?;
    // the checkpoint is a read-only blob that is not rewritten during an export
    let mapping = unsafe { Mmap::map(&file) }.map_err(wrap(gguf_path))?;
    let parsed = parse_complete(&mapping)?;
    let files = mlx_directory_files(&parsed, &mapping, quantization, dtype)?;
    create_dir_all(out_dir).map_err(wrap(out_dir))?;
    files
        .into_iter()
        .map(|(name, bytes)| {
            let path = out_dir.join(name);
            write(&path, bytes).map_err(wrap(&path))?;
            Ok(path)
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use proxima_gguf::value::MetadataValue;
    use proxima_gguf::{GgufModel, write_complete};

    use super::*;

    fn gguf_of_family(family: &str) -> Vec<u8> {
        write_complete(&GgufModel {
            version: 3,
            metadata: vec![(
                "general.architecture".to_string(),
                MetadataValue::String(family.to_string()),
            )],
            tensors: Vec::new(),
        })
        .expect("writes the gguf")
    }

    #[test]
    fn mlx_directory_files_refuse_a_family_without_an_mlx_layout() {
        let bytes = gguf_of_family("llama");
        let parsed = parse_complete(&bytes).expect("parses the gguf just written");

        let refusal = mlx_directory_files(&parsed, &bytes, None, DType::BFloat16).unwrap_err();

        assert!(
            matches!(refusal, InteropError::UnknownFamilyConfig { family } if family == "llama")
        );
    }

    #[test]
    fn mlx_directory_export_names_the_missing_gguf_in_the_io_error() {
        let scratch = tempfile::tempdir().expect("tempdir");
        let missing = scratch.path().join("absent.gguf");

        let refusal =
            export_mlx_directory(&missing, &scratch.path().join("out"), None, DType::BFloat16)
                .unwrap_err();

        assert!(matches!(refusal, InteropError::MlxDirectoryIo { path, .. } if path == missing));
    }

    #[test]
    fn mlx_directory_export_writes_nothing_when_the_family_is_refused() {
        let scratch = tempfile::tempdir().expect("tempdir");
        let source = scratch.path().join("llama.gguf");
        std::fs::write(&source, gguf_of_family("llama")).expect("write the gguf");
        let out = scratch.path().join("out");

        let refusal = export_mlx_directory(&source, &out, None, DType::BFloat16).unwrap_err();

        assert!(matches!(refusal, InteropError::UnknownFamilyConfig { .. }));
        assert!(!out.exists());
    }
}
