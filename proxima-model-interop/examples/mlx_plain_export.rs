//! Writes the plain bf16 MLX weight files of a GGUF to a directory:
//! `mlx_plain_export <gguf> <out-dir>`.
//!
//! Composes `export_plain` (dequantize and cast), `dense_names` (HF names
//! through the binding profile, fused expert halves reported apart) and
//! `MlxExport::assemble` (shard cut and index). Fused halves are interleaved
//! per expert into one `[experts, parts * rows, in]` tensor, the layout the
//! HF loader cuts back with `TensorAlias::Part`.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{File, create_dir_all, write};
use std::path::PathBuf;

use memmap2::Mmap;
use proxima_gguf::parse_complete;
use proxima_model_interop::profiles::binding_profile;
use proxima_model_interop::{
    DEFAULT_PLAIN_DTYPE, MLX_INDEX_FILE_NAME, MLX_MAX_SHARD_BYTES, MlxExport, MlxNames,
    PlainTensor, STANDARD_LAYOUT, dense_names, export_plain,
};
use proxima_safetensors::TensorPayload;

const FAMILY: &str = "granitemoe";
const HF_PREFIX: &str = "model.";

fn interleave(name: &str, parts: &[PlainTensor]) -> PlainTensor {
    let experts = usize::try_from(parts[0].shape[0]).unwrap_or(1);
    let chunk = parts[0].bytes.len() / experts;
    let bytes = (0..experts)
        .flat_map(|expert| {
            parts
                .iter()
                .flat_map(move |part| part.bytes[expert * chunk..(expert + 1) * chunk].to_vec())
        })
        .collect();
    let mut shape = parts[0].shape.clone();
    shape[1] *= parts.len() as u64;
    PlainTensor {
        name: name.to_string(),
        shape,
        dtype: parts[0].dtype,
        bytes,
    }
}

fn fuse(names: &MlxNames, by_leaf: &mut BTreeMap<String, PlainTensor>) -> Vec<PlainTensor> {
    let targets: BTreeSet<&str> = names.fused.iter().map(|item| item.hf.as_str()).collect();
    targets
        .into_iter()
        .filter_map(|target| {
            let mut group: Vec<_> = names.fused.iter().filter(|i| i.hf == target).collect();
            group.sort_by_key(|item| item.part);
            let parts: Option<Vec<PlainTensor>> = group
                .iter()
                .map(|item| by_leaf.remove(&item.leaf))
                .collect();
            parts.map(|parts| interleave(target, &parts))
        })
        .collect()
}

fn hf_tensors(gguf: &Mmap) -> Result<Vec<PlainTensor>, Box<dyn Error>> {
    let parsed = parse_complete(gguf)?;
    let binding = binding_profile(FAMILY)?;
    let directory: Vec<&str> = parsed.tensors.iter().map(|t| t.name.as_str()).collect();
    let leaves: Vec<&str> = directory
        .iter()
        .copied()
        .filter(|name| !binding.extra.iter().any(|extra| extra == name))
        .collect();
    let names = dense_names(&leaves, &directory, &binding, STANDARD_LAYOUT, HF_PREFIX);
    if !names.unmapped.is_empty() {
        return Err(format!("unmapped leaves: {:?}", names.unmapped).into());
    }
    let mut by_leaf: BTreeMap<String, PlainTensor> =
        export_plain(&parsed, gguf, DEFAULT_PLAIN_DTYPE)?
            .into_iter()
            .map(|tensor| (tensor.name.clone(), tensor))
            .collect();
    let mut named: Vec<PlainTensor> = names
        .mapped
        .iter()
        .filter_map(|(leaf, hf)| {
            by_leaf.remove(leaf).map(|mut tensor| {
                tensor.name.clone_from(hf);
                tensor
            })
        })
        .collect();
    named.extend(fuse(&names, &mut by_leaf));
    Ok(named)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let (Some(gguf_path), Some(out_dir)) = (args.next(), args.next().map(PathBuf::from)) else {
        return Err("usage: mlx_plain_export <gguf> <out-dir>".into());
    };
    // SAFETY: the checkpoint is not written while this process reads it.
    let gguf = unsafe { Mmap::map(&File::open(gguf_path)?)? };
    let tensors = hf_tensors(&gguf)?;
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
    create_dir_all(&out_dir)?;
    for shard in &export.shards {
        write(out_dir.join(&shard.file_name), &shard.bytes)?;
    }
    if let Some(index) = &export.index {
        write(out_dir.join(MLX_INDEX_FILE_NAME), index)?;
    }
    println!(
        "wrote {} shard(s) to {}",
        export.shards.len(),
        out_dir.display()
    );
    Ok(())
}
