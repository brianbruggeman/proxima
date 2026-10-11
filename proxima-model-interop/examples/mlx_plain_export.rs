//! Writes the plain bf16 MLX weight and tokenizer files of a GGUF to a
//! directory: `mlx_plain_export <gguf> <out-dir> [standard|sandwich-norm] [hf-prefix]`.
//!
//! Composes `export_plain_named` (dequantize, cast, HF names through the
//! binding profile, fused expert halves interleaved), `export_tokenizer` and
//! `MlxExport::assemble` (shard cut and index).

use std::error::Error;
use std::fs::{File, create_dir_all, write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use proxima_gguf::parse_complete;
use proxima_gguf::value::MetadataValue;
use proxima_model_interop::profiles::binding_profile;
use proxima_model_interop::{
    DEFAULT_PLAIN_DTYPE, Layout, MLX_INDEX_FILE_NAME, MLX_MAX_SHARD_BYTES, MlxExport,
    SANDWICH_NORM_LAYOUT, STANDARD_LAYOUT, export_plain_named, export_tokenizer,
};
use proxima_safetensors::TensorPayload;

const DEFAULT_HF_PREFIX: &str = "model.";

fn layout_named(name: &str) -> Result<Layout, Box<dyn Error>> {
    match name {
        "standard" => Ok(STANDARD_LAYOUT),
        "sandwich-norm" => Ok(SANDWICH_NORM_LAYOUT),
        other => Err(format!("unknown layout {other}: standard or sandwich-norm").into()),
    }
}

fn write_directory(export: &MlxExport, out_dir: &Path) -> Result<(), Box<dyn Error>> {
    create_dir_all(out_dir)?;
    for shard in &export.shards {
        write(out_dir.join(&shard.file_name), &shard.bytes)?;
    }
    if let Some(index) = &export.index {
        write(out_dir.join(MLX_INDEX_FILE_NAME), index)?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let (Some(gguf_path), Some(out_dir)) = (args.next(), args.next().map(PathBuf::from)) else {
        return Err("usage: mlx_plain_export <gguf> <out-dir> [layout] [hf-prefix]".into());
    };
    let layout = layout_named(&args.next().unwrap_or_else(|| "standard".into()))?;
    let prefix = args.next().unwrap_or_else(|| DEFAULT_HF_PREFIX.into());
    // SAFETY: the checkpoint is not written while this process reads it.
    let gguf = unsafe { Mmap::map(&File::open(gguf_path)?)? };
    let parsed = parse_complete(&gguf)?;
    let Some(MetadataValue::String(family)) = parsed.metadata_value("general.architecture") else {
        return Err("gguf carries no general.architecture".into());
    };
    let binding = binding_profile(family)?;
    let tensors = export_plain_named(
        &parsed,
        &gguf,
        DEFAULT_PLAIN_DTYPE,
        &binding,
        layout,
        &prefix,
    )?;
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
    write_directory(&export, &out_dir)?;
    let tokenizer = export_tokenizer(&parsed)?;
    write(out_dir.join("tokenizer.json"), &tokenizer.tokenizer_json)?;
    write(
        out_dir.join("tokenizer_config.json"),
        &tokenizer.tokenizer_config_json,
    )?;
    println!(
        "wrote {} shard(s) to {}",
        export.shards.len(),
        out_dir.display()
    );
    Ok(())
}
