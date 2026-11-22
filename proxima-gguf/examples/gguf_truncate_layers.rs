//! Writes a copy of a gguf keeping only the first `layers` transformer blocks, with
//! `{architecture}.block_count` rewritten to match, so two backends can be compared on real
//! weights one layer at a time. An optional `vocab_rows` also cuts `token_embd.weight` and
//! `output.weight` to their first rows, which keeps a float32 oracle of the whole file in memory.
//!
//! Usage: `cargo run --example gguf_truncate_layers -- <src.gguf> <dst.gguf> <layers> [vocab_rows]`

use std::env;
use std::error::Error;
use std::fs::{self, File};

use memmap2::Mmap;
use proxima_gguf::pipe::{ParsedGguf, parse_complete};
use proxima_gguf::tensor::TensorInfo;
use proxima_gguf::value::{MetadataArray, MetadataValue};
use proxima_gguf::writer::{GgufModel, TensorPayload, write_complete};

const VOCAB_TENSORS: [&str; 2] = ["token_embd.weight", "output.weight"];

fn block_index(tensor_name: &str) -> Option<usize> {
    tensor_name
        .strip_prefix("blk.")?
        .split('.')
        .next()?
        .parse()
        .ok()
}

fn truncate_array(array: &MetadataArray, layers: usize) -> MetadataArray {
    let mut array = array.clone();
    match &mut array {
        MetadataArray::U8(values) => values.truncate(layers),
        MetadataArray::I8(values) => values.truncate(layers),
        MetadataArray::U16(values) => values.truncate(layers),
        MetadataArray::I16(values) => values.truncate(layers),
        MetadataArray::U32(values) => values.truncate(layers),
        MetadataArray::I32(values) => values.truncate(layers),
        MetadataArray::F32(values) => values.truncate(layers),
        MetadataArray::Bool(values) => values.truncate(layers),
        MetadataArray::String(values) => values.truncate(layers),
        MetadataArray::U64(values) => values.truncate(layers),
        MetadataArray::I64(values) => values.truncate(layers),
        MetadataArray::F64(values) => values.truncate(layers),
    }
    array
}

fn truncated_metadata(
    parsed: &ParsedGguf,
    layers: u32,
) -> Result<Vec<(String, MetadataValue)>, Box<dyn Error>> {
    let Some(MetadataValue::String(architecture)) = parsed.metadata_value("general.architecture")
    else {
        return Err("general.architecture is missing".into());
    };
    let block_count_key = format!("{architecture}.block_count");
    let original_layers = match parsed.metadata_value(&block_count_key) {
        Some(MetadataValue::U32(count)) => *count as usize,
        _ => return Err(format!("{block_count_key} is missing or not U32").into()),
    };
    let family_prefix = format!("{architecture}.");
    Ok(parsed
        .metadata
        .iter()
        .map(|(key, value)| {
            if key == &block_count_key {
                return (key.clone(), MetadataValue::U32(layers));
            }
            if key.starts_with(&family_prefix)
                && let MetadataValue::Array(array) = value
                && array.len() == original_layers
            {
                return (
                    key.clone(),
                    MetadataValue::Array(truncate_array(array, layers as usize)),
                );
            }
            (key.clone(), value.clone())
        })
        .collect())
}

fn payload<'a>(
    parsed: &ParsedGguf,
    mapping: &'a [u8],
    tensor: &TensorInfo,
    vocab_rows: Option<u64>,
) -> Result<TensorPayload<'a>, Box<dyn Error>> {
    let range = parsed.tensor_data_range(tensor, mapping.len() as u64)?;
    let mut dims = tensor.dims.clone();
    let mut end = range.end;
    if let (Some(rows), true) = (vocab_rows, VOCAB_TENSORS.contains(&tensor.name.as_str())) {
        let row_bytes = (range.end - range.start) / dims[1];
        dims[1] = rows;
        end = range.start + row_bytes * rows;
    }
    Ok(TensorPayload {
        name: tensor.name.clone(),
        dims,
        ggml_type: tensor.ggml_type,
        data: &mapping[range.start as usize..end as usize],
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let usage = "usage: gguf_truncate_layers <src.gguf> <dst.gguf> <layers> [vocab_rows]";
    let source = args.next().ok_or(usage)?;
    let destination = args.next().ok_or(usage)?;
    let layers: u32 = args.next().ok_or(usage)?.parse()?;
    let vocab_rows: Option<u64> = args.next().map(|text| text.parse()).transpose()?;

    let file = File::open(&source)?;
    let mapping = unsafe { Mmap::map(&file) }?;
    let parsed = parse_complete(&mapping)?;
    let metadata = truncated_metadata(&parsed, layers)?;
    let tensors = parsed
        .tensors
        .iter()
        .filter(|tensor| block_index(&tensor.name).is_none_or(|index| index < layers as usize))
        .map(|tensor| payload(&parsed, &mapping, tensor, vocab_rows))
        .collect::<Result<Vec<_>, _>>()?;
    let kept = tensors.len();
    let written = write_complete(&GgufModel {
        version: parsed.version,
        metadata,
        tensors,
    })?;
    fs::write(&destination, &written)?;
    println!("kept {kept} of {} tensors, {} bytes", parsed.tensors.len(), written.len());
    Ok(())
}
