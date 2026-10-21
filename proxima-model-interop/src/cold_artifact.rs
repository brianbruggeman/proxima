//! Streaming cold conversion of a pinned BF16 safetensors checkpoint.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions, create_dir};
use std::io::{Read, Seek, SeekFrom, Write};
use std::mem;
use std::path::{Component, Path, PathBuf};

use proxima_primitives::float_layout::FloatLayout;
use proxima_primitives::pipe::sans_io::Outcome;
use proxima_safetensors::{
    MAX_HEADER_BYTES, Manifest, SafetensorsParser, TensorEntry, parse_sharded_index,
};
use proxima_tensor::DType;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

const SOURCE_METADATA_FILES: [&str; 8] = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
];
const OUTPUT_MANIFEST: &str = "manifest.json";

#[derive(Debug, Error)]
pub enum ColdArtifactError {
    #[error("cold artifact IO: {0}")]
    Io(#[from] std::io::Error),
    #[error("safetensors parse: {0}")]
    Safetensors(#[from] proxima_safetensors::SafetensorsError),
    #[error("cold tensor quantization: {0}")]
    Quant(#[from] proxima_gguf::quant::QuantError),
    #[error("JSON manifest: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported cold float format {format:?}")]
    UnsupportedFormat { format: String },
    #[error("source checkpoint is missing {path:?}")]
    MissingSource { path: PathBuf },
    #[error("shard path {path:?} is not a relative file path")]
    InvalidShardPath { path: String },
    #[error("tensor {tensor:?} has dtype {dtype:?}; cold conversion requires BF16")]
    UnsupportedDtype { tensor: String, dtype: DType },
    #[error("tensor {tensor:?} has a shape or byte length that does not fit BF16")]
    InvalidTensorLength { tensor: String },
    #[error("safetensors shard {path:?} has overlapping or out-of-order data ranges")]
    InvalidTensorRanges { path: PathBuf },
    #[error("chunk element count must be positive and divisible by eight")]
    InvalidChunkElements,
    #[error("output directory already exists: {path:?}")]
    OutputExists { path: PathBuf },
    #[error("output directory {output:?} is inside source checkpoint {source_root:?}")]
    OutputInsideSource {
        source_root: PathBuf,
        output: PathBuf,
    },
    #[error("safetensors shard header did not produce a manifest")]
    MissingManifest,
    #[error("cold artifact manifest has an unsupported schema")]
    InvalidArtifactManifest,
    #[error(
        "checkpoint config identifies model type {found:?}, expected Granite 4 granitemoehybrid"
    )]
    UnsupportedModelType { found: String },
    #[error("tensor {tensor:?} is absent from the sharded index")]
    IndexMissingTensor { tensor: String },
}

#[derive(Debug, Serialize)]
pub struct ColdArtifactManifest {
    pub schema: &'static str,
    pub repository: String,
    pub revision: String,
    pub format: &'static str,
    pub layout: FloatLayout,
    pub scale_mode: &'static str,
    pub chunk_elements: usize,
    pub source_artifacts: BTreeMap<String, String>,
    pub tensors: Vec<ColdTensorRecord>,
    pub source_bytes_processed: u64,
    pub source_bytes_scanned: u64,
    pub output_bytes_written: u64,
}

#[derive(Debug, Serialize)]
pub struct ColdTensorRecord {
    pub name: String,
    pub source_shard: String,
    pub output_file: String,
    pub source_shape: Vec<u64>,
    pub source_dtype: &'static str,
    pub format: &'static str,
    pub layout: FloatLayout,
    pub scale: Option<f32>,
    pub source_sha256: String,
    pub output_sha256: String,
    pub source_bytes_processed: u64,
    pub output_bytes_written: u64,
}

#[derive(Debug, Serialize)]
pub struct ColdComparisonReport {
    pub repository: String,
    pub revision: String,
    pub format: String,
    pub scale_mode: String,
    pub source_values: u64,
    pub changed_values: u64,
    pub maximum_absolute_error: f32,
    pub sum_squared_error: f64,
    pub saturated_values: u64,
    pub zeroed_nonzero_values: u64,
    pub tensors: Vec<ColdTensorComparison>,
}

#[derive(Debug, Serialize)]
pub struct ColdTensorComparison {
    pub name: String,
    pub source_dtype: String,
    pub values: u64,
    pub changed_values: u64,
    pub maximum_absolute_error: f32,
    pub maximum_error_sample: Option<ColdValueSample>,
    pub sum_squared_error: f64,
    pub maximum_source_magnitude: f32,
    pub maximum_reconstructed_magnitude: f32,
    pub saturated_values: u64,
    pub zeroed_nonzero_values: u64,
    pub samples: Vec<ColdValueSample>,
}

#[derive(Debug, Serialize)]
pub struct ColdValueSample {
    pub index: u64,
    pub source: f32,
    pub reconstructed: f32,
}

#[derive(Debug, Deserialize)]
struct ColdArtifactInputManifest {
    schema: String,
    repository: String,
    revision: String,
    format: String,
    #[serde(default)]
    layout: Option<FloatLayout>,
    #[serde(default = "default_scale_mode")]
    scale_mode: String,
    chunk_elements: usize,
    tensors: Vec<ColdTensorInputRecord>,
}

fn default_scale_mode() -> String {
    String::from("none")
}

#[derive(Debug, Deserialize)]
struct ColdTensorInputRecord {
    name: String,
    source_shard: String,
    output_file: String,
    source_shape: Vec<u64>,
    source_dtype: String,
    scale: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColdScaleMode {
    None,
    TensorMaximum,
}

impl ColdScaleMode {
    const fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::TensorMaximum => "tensor_maximum",
        }
    }
}

/// Compares every packed tensor against its original checkpoint values and
/// retains per-tensor error totals plus the first eight raw value pairs.
pub fn compare_checkpoint(
    source_dir: &Path,
    artifact_dir: &Path,
) -> Result<ColdComparisonReport, ColdArtifactError> {
    let manifest_file = File::open(artifact_dir.join(OUTPUT_MANIFEST))?;
    let manifest: ColdArtifactInputManifest = serde_json::from_reader(manifest_file)?;
    if manifest.schema != "proxima.cold-weights.v1" {
        return Err(ColdArtifactError::InvalidArtifactManifest);
    }
    let layout = manifest
        .layout
        .or_else(|| parse_format(&manifest.format).ok())
        .ok_or_else(|| ColdArtifactError::UnsupportedFormat {
            format: manifest.format.clone(),
        })?;
    let tensor_scaled = match manifest.scale_mode.as_str() {
        "none" => false,
        "tensor_maximum" => true,
        _ => return Err(ColdArtifactError::InvalidArtifactManifest),
    };
    if manifest.chunk_elements == 0 || !manifest.chunk_elements.is_multiple_of(8) {
        return Err(ColdArtifactError::InvalidChunkElements);
    }
    let index_path = source_dir.join("model.safetensors.index.json");
    let index_bytes = std::fs::read(&index_path)?;
    let index = parse_sharded_index(&index_bytes)?;
    let mut source_values = 0_u64;
    let mut changed_values = 0_u64;
    let mut maximum_absolute_error = 0.0_f32;
    let mut sum_squared_error = 0.0_f64;
    let mut saturated_values = 0_u64;
    let mut zeroed_nonzero_values = 0_u64;
    let format_maximum = layout.max_normal();
    let mut tensors = Vec::with_capacity(manifest.tensors.len());
    let chunk_elements = manifest.chunk_elements;
    let mut input_buffer = vec![0_u8; chunk_elements.saturating_mul(4)];
    let mut decoded_source = vec![0.0_f32; chunk_elements];
    let mut decoded_cold = vec![0.0_f32; chunk_elements];
    let mut packed_buffer = vec![0_u8; chunk_elements.saturating_mul(8).div_ceil(8)];

    for tensor in manifest.tensors {
        let tensor_scale = match (tensor_scaled, tensor.scale) {
            (false, None) => 1.0,
            (true, Some(scale)) if scale.is_finite() && scale > 0.0 => scale,
            _ => return Err(ColdArtifactError::InvalidArtifactManifest),
        };
        let shard_path = safe_shard_path(source_dir, &tensor.source_shard)?;
        let mut shard_file = File::open(&shard_path)?;
        let (_, shard_manifest, data_start, _) = read_shard_header(&mut shard_file)?;
        let source_tensor = shard_manifest.tensor(&tensor.name).ok_or_else(|| {
            ColdArtifactError::IndexMissingTensor {
                tensor: tensor.name.clone(),
            }
        })?;
        if index.shard_for(&tensor.name) != Some(tensor.source_shard.as_str())
            || source_tensor.shape != tensor.source_shape
        {
            return Err(ColdArtifactError::IndexMissingTensor {
                tensor: tensor.name,
            });
        }
        let source_element_bytes = match tensor.source_dtype.as_str() {
            "BF16" => 2_usize,
            "F32" => 4_usize,
            _ => {
                return Err(ColdArtifactError::UnsupportedDtype {
                    tensor: tensor.name,
                    dtype: source_tensor.dtype,
                });
            }
        };
        let value_count = tensor_element_count(source_tensor)?;
        let byte_offset = data_start
            .checked_add(source_tensor.data_offsets.0)
            .ok_or_else(|| ColdArtifactError::InvalidTensorLength {
                tensor: tensor.name.clone(),
            })?;
        shard_file.seek(SeekFrom::Start(byte_offset))?;
        let mut packed_file = File::open(artifact_dir.join(&tensor.output_file))?;
        let mut tensor_comparison = ColdTensorComparison {
            name: tensor.name,
            source_dtype: tensor.source_dtype,
            values: value_count,
            changed_values: 0,
            maximum_absolute_error: 0.0,
            maximum_error_sample: None,
            sum_squared_error: 0.0,
            maximum_source_magnitude: 0.0,
            maximum_reconstructed_magnitude: 0.0,
            saturated_values: 0,
            zeroed_nonzero_values: 0,
            samples: Vec::with_capacity(8),
        };
        let mut processed_values = 0_u64;
        while processed_values < value_count {
            let elements = (value_count - processed_values).min(chunk_elements as u64) as usize;
            let source_bytes = elements * source_element_bytes;
            shard_file.read_exact(&mut input_buffer[..source_bytes])?;
            if source_element_bytes == 2 {
                proxima_gguf::quant::bf16::dequantize(
                    &input_buffer[..source_bytes],
                    &mut decoded_source[..elements],
                )?;
            } else {
                for (bytes, value) in input_buffer[..source_bytes]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(decoded_source[..elements].iter_mut())
                {
                    *value = f32::from_le_bytes(*bytes);
                }
            }
            let packed_bits = elements * layout.width() as usize;
            let packed_bytes = packed_bits.div_ceil(8);
            packed_file.read_exact(&mut packed_buffer[..packed_bytes])?;
            proxima_gguf::quant::cold_float::dequantize(
                layout,
                &packed_buffer[..packed_bytes],
                &mut decoded_cold[..elements],
            )?;
            for value_index in 0..elements {
                let source_value = decoded_source[value_index];
                let reconstructed_value = decoded_cold[value_index] * tensor_scale;
                let normalized_source_value = source_value / tensor_scale;
                let absolute_error = (source_value - reconstructed_value).abs();
                if source_value.to_bits() != reconstructed_value.to_bits() {
                    tensor_comparison.changed_values += 1;
                    changed_values += 1;
                }
                if absolute_error > tensor_comparison.maximum_absolute_error {
                    tensor_comparison.maximum_absolute_error = absolute_error;
                    tensor_comparison.maximum_error_sample = Some(ColdValueSample {
                        index: processed_values + value_index as u64,
                        source: source_value,
                        reconstructed: reconstructed_value,
                    });
                }
                tensor_comparison.maximum_source_magnitude = tensor_comparison
                    .maximum_source_magnitude
                    .max(source_value.abs());
                tensor_comparison.maximum_reconstructed_magnitude = tensor_comparison
                    .maximum_reconstructed_magnitude
                    .max(reconstructed_value.abs());
                if normalized_source_value.abs() > format_maximum {
                    tensor_comparison.saturated_values += 1;
                    saturated_values += 1;
                }
                if source_value != 0.0 && reconstructed_value == 0.0 {
                    tensor_comparison.zeroed_nonzero_values += 1;
                    zeroed_nonzero_values += 1;
                }
                tensor_comparison.sum_squared_error +=
                    f64::from(absolute_error) * f64::from(absolute_error);
                if tensor_comparison.samples.len() < 8 {
                    tensor_comparison.samples.push(ColdValueSample {
                        index: processed_values + value_index as u64,
                        source: source_value,
                        reconstructed: reconstructed_value,
                    });
                }
            }
            processed_values += elements as u64;
        }
        source_values += tensor_comparison.values;
        maximum_absolute_error =
            maximum_absolute_error.max(tensor_comparison.maximum_absolute_error);
        sum_squared_error += tensor_comparison.sum_squared_error;
        tensors.push(tensor_comparison);
    }

    Ok(ColdComparisonReport {
        repository: manifest.repository,
        revision: manifest.revision,
        format: manifest.format,
        scale_mode: manifest.scale_mode,
        source_values,
        changed_values,
        maximum_absolute_error,
        sum_squared_error,
        saturated_values,
        zeroed_nonzero_values,
        tensors,
    })
}

/// Converts every BF16 tensor in a sharded safetensors checkpoint to a
/// Proxima packed-weight directory without allocating a whole shard or tensor.
pub fn convert_checkpoint(
    source_dir: &Path,
    output_dir: &Path,
    layout: FloatLayout,
    repository: &str,
    revision: &str,
    chunk_elements: usize,
) -> Result<ColdArtifactManifest, ColdArtifactError> {
    convert_checkpoint_with_scale(
        source_dir,
        output_dir,
        layout,
        repository,
        revision,
        chunk_elements,
        ColdScaleMode::None,
    )
}

pub fn convert_checkpoint_with_scale(
    source_dir: &Path,
    output_dir: &Path,
    layout: FloatLayout,
    repository: &str,
    revision: &str,
    chunk_elements: usize,
    scale_mode: ColdScaleMode,
) -> Result<ColdArtifactManifest, ColdArtifactError> {
    if chunk_elements == 0 || !chunk_elements.is_multiple_of(8) {
        return Err(ColdArtifactError::InvalidChunkElements);
    }
    if output_dir.exists() {
        return Err(ColdArtifactError::OutputExists {
            path: output_dir.to_path_buf(),
        });
    }
    let source_root = source_dir.canonicalize()?;
    let output_parent = output_dir
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .canonicalize()?;
    let output_path = output_parent.join(output_dir.file_name().ok_or_else(|| {
        ColdArtifactError::OutputExists {
            path: output_dir.to_path_buf(),
        }
    })?);
    if output_path.starts_with(&source_root) {
        return Err(ColdArtifactError::OutputInsideSource {
            source_root,
            output: output_path,
        });
    }
    create_dir(output_dir)?;

    let index_path = source_dir.join("model.safetensors.index.json");
    let index_bytes = std::fs::read(&index_path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            ColdArtifactError::MissingSource { path: index_path }
        } else {
            ColdArtifactError::Io(error)
        }
    })?;
    let index = parse_sharded_index(&index_bytes)?;
    let config_path = source_dir.join("config.json");
    let config_bytes = std::fs::read(&config_path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            ColdArtifactError::MissingSource { path: config_path }
        } else {
            ColdArtifactError::Io(error)
        }
    })?;
    let config: serde_json::Value = serde_json::from_slice(&config_bytes)?;
    let model_type = config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("missing");
    if model_type != "granitemoehybrid" {
        return Err(ColdArtifactError::UnsupportedModelType {
            found: String::from(model_type),
        });
    }
    let mut source_artifacts = BTreeMap::new();
    for filename in SOURCE_METADATA_FILES {
        let path = source_dir.join(filename);
        let digest = hash_file(&path, chunk_elements.saturating_mul(2))?;
        source_artifacts.insert(String::from(filename), digest);
    }

    let mut tensors = Vec::new();
    let mut source_bytes_processed = 0_u64;
    let mut source_bytes_scanned = 0_u64;
    let mut output_bytes_written = 0_u64;
    let mut tensor_ordinal = 0_usize;
    let mut input_buffer = vec![0_u8; chunk_elements.saturating_mul(4)];
    let mut value_buffer = vec![0.0_f32; chunk_elements];
    let output_buffer_len = chunk_elements
        .saturating_mul(layout.width() as usize)
        .div_ceil(8);
    let mut output_buffer = vec![0_u8; output_buffer_len];

    for shard_name in index.shard_filenames() {
        let shard_path = safe_shard_path(source_dir, shard_name)?;
        let mut shard_file = File::open(&shard_path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                ColdArtifactError::MissingSource {
                    path: shard_path.clone(),
                }
            } else {
                ColdArtifactError::Io(error)
            }
        })?;
        let shard_length = shard_file.metadata()?.len();
        let (mut parser, mut manifest, data_start, mut shard_hasher) =
            read_shard_header(&mut shard_file)?;
        let mut shard_tensors = mem::take(&mut manifest.tensors);
        shard_tensors.sort_by_key(|entry| entry.data_offsets.0);
        let mut data_cursor = 0_u64;

        for tensor in shard_tensors {
            if index.shard_for(&tensor.name) != Some(shard_name) {
                return Err(ColdArtifactError::IndexMissingTensor {
                    tensor: tensor.name,
                });
            }
            if tensor.data_offsets.0 < data_cursor {
                return Err(ColdArtifactError::InvalidTensorRanges {
                    path: shard_path.clone(),
                });
            }
            let gap_bytes = tensor.data_offsets.0 - data_cursor;
            consume_gap(
                &mut shard_file,
                gap_bytes,
                &mut input_buffer,
                &mut parser,
                &mut shard_hasher,
                &mut source_bytes_processed,
            )?;
            data_cursor = tensor.data_offsets.0;
            let (source_element_bytes, source_dtype) = if tensor.dtype == DType::BFloat16 {
                (2_usize, "BF16")
            } else if tensor.dtype == DType::Float32 {
                (4_usize, "F32")
            } else {
                return Err(ColdArtifactError::UnsupportedDtype {
                    tensor: tensor.name,
                    dtype: tensor.dtype,
                });
            };
            let element_count = tensor_element_count(&tensor)?;
            let source_bytes = element_count
                .checked_mul(source_element_bytes as u64)
                .ok_or_else(|| ColdArtifactError::InvalidTensorLength {
                    tensor: tensor.name.clone(),
                })?;
            if source_bytes != tensor.byte_len() {
                return Err(ColdArtifactError::InvalidTensorLength {
                    tensor: tensor.name,
                });
            }
            let tensor_scale = if scale_mode == ColdScaleMode::TensorMaximum {
                source_bytes_scanned += source_bytes;
                measure_tensor_scale(
                    &mut shard_file,
                    data_start,
                    &tensor,
                    source_element_bytes,
                    layout,
                    &mut input_buffer,
                    &mut value_buffer,
                )?
            } else {
                1.0
            };
            let tensor_start = data_start
                .checked_add(tensor.data_offsets.0)
                .ok_or_else(|| ColdArtifactError::InvalidTensorLength {
                    tensor: tensor.name.clone(),
                })?;
            shard_file.seek(SeekFrom::Start(tensor_start))?;
            let output_file_name = format!("tensor-{tensor_ordinal:06}.pxw");
            let output_path = output_dir.join(&output_file_name);
            let mut output_file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(output_path)?;
            let mut source_hasher = Sha256::new();
            let mut output_hasher = Sha256::new();
            let mut tensor_source_bytes = 0_u64;
            let mut tensor_output_bytes = 0_u64;
            let mut remaining_elements = element_count;
            while remaining_elements > 0 {
                let elements_this_chunk =
                    remaining_elements.min(value_buffer.len() as u64) as usize;
                let bytes_this_chunk = elements_this_chunk * source_element_bytes;
                let input_chunk = &mut input_buffer[..bytes_this_chunk];
                shard_file.read_exact(input_chunk)?;
                parser.feed(input_chunk);
                shard_hasher.update(&*input_chunk);
                source_hasher.update(&*input_chunk);
                source_bytes_processed += bytes_this_chunk as u64;
                tensor_source_bytes += bytes_this_chunk as u64;

                let values_this_chunk = &mut value_buffer[..elements_this_chunk];
                if tensor.dtype == DType::BFloat16 {
                    proxima_gguf::quant::bf16::dequantize(input_chunk, values_this_chunk)?;
                } else if tensor.dtype == DType::Float32 {
                    for (chunk, value) in input_chunk
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .zip(values_this_chunk.iter_mut())
                    {
                        *value = f32::from_le_bytes(*chunk);
                    }
                } else {
                    return Err(ColdArtifactError::UnsupportedDtype {
                        tensor: tensor.name,
                        dtype: tensor.dtype,
                    });
                }
                if scale_mode == ColdScaleMode::TensorMaximum {
                    values_this_chunk
                        .iter_mut()
                        .for_each(|value| *value /= tensor_scale);
                }

                let output_chunk_len = elements_this_chunk
                    .saturating_mul(layout.width() as usize)
                    .div_ceil(8);
                let output_chunk = &mut output_buffer[..output_chunk_len];
                proxima_gguf::quant::cold_float::quantize_f32(
                    layout,
                    values_this_chunk,
                    output_chunk,
                )?;
                output_file.write_all(output_chunk)?;
                output_hasher.update(output_chunk);
                tensor_output_bytes += output_chunk_len as u64;
                output_bytes_written += output_chunk_len as u64;
                remaining_elements -= elements_this_chunk as u64;
                data_cursor += bytes_this_chunk as u64;
            }
            output_file.sync_all()?;
            tensors.push(ColdTensorRecord {
                name: tensor.name,
                source_shard: String::from(shard_name),
                output_file: output_file_name,
                source_shape: tensor.shape,
                source_dtype,
                format: layout_name(layout),
                layout,
                scale: (scale_mode == ColdScaleMode::TensorMaximum).then_some(tensor_scale),
                source_sha256: digest_hex(source_hasher.finalize()),
                output_sha256: digest_hex(output_hasher.finalize()),
                source_bytes_processed: tensor_source_bytes,
                output_bytes_written: tensor_output_bytes,
            });
            tensor_ordinal += 1;
        }

        let data_length = shard_length.saturating_sub(data_start);
        if data_cursor > data_length {
            return Err(ColdArtifactError::InvalidTensorRanges {
                path: shard_path.clone(),
            });
        }
        consume_gap(
            &mut shard_file,
            data_length - data_cursor,
            &mut input_buffer,
            &mut parser,
            &mut shard_hasher,
            &mut source_bytes_processed,
        )?;
        parser.finish()?;
        source_artifacts.insert(
            String::from(shard_name),
            digest_hex(shard_hasher.finalize()),
        );
    }

    let manifest = ColdArtifactManifest {
        schema: "proxima.cold-weights.v1",
        repository: String::from(repository),
        revision: String::from(revision),
        format: layout_name(layout),
        layout,
        scale_mode: scale_mode.name(),
        chunk_elements,
        source_artifacts,
        tensors,
        source_bytes_processed,
        source_bytes_scanned,
        output_bytes_written,
    };
    let manifest_file = File::create_new(output_dir.join(OUTPUT_MANIFEST))?;
    serde_json::to_writer_pretty(&manifest_file, &manifest)?;
    manifest_file.sync_all()?;
    Ok(manifest)
}

fn measure_tensor_scale(
    shard_file: &mut File,
    data_start: u64,
    tensor: &TensorEntry,
    source_element_bytes: usize,
    layout: FloatLayout,
    input_buffer: &mut [u8],
    value_buffer: &mut [f32],
) -> Result<f32, ColdArtifactError> {
    let tensor_start = data_start
        .checked_add(tensor.data_offsets.0)
        .ok_or_else(|| ColdArtifactError::InvalidTensorLength {
            tensor: tensor.name.clone(),
        })?;
    shard_file.seek(SeekFrom::Start(tensor_start))?;
    let mut remaining_elements = tensor_element_count(tensor)?;
    let mut maximum_magnitude = 0.0_f32;
    while remaining_elements > 0 {
        let elements = remaining_elements.min(value_buffer.len() as u64) as usize;
        let source_bytes = elements * source_element_bytes;
        let input_chunk = &mut input_buffer[..source_bytes];
        shard_file.read_exact(input_chunk)?;
        let values = &mut value_buffer[..elements];
        if tensor.dtype == DType::BFloat16 {
            proxima_gguf::quant::bf16::dequantize(input_chunk, values)?;
        } else if tensor.dtype == DType::Float32 {
            for (bytes, value) in input_chunk.as_chunks::<4>().0.iter().zip(values.iter_mut()) {
                *value = f32::from_le_bytes(*bytes);
            }
        } else {
            return Err(ColdArtifactError::UnsupportedDtype {
                tensor: tensor.name.clone(),
                dtype: tensor.dtype,
            });
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(ColdArtifactError::InvalidTensorLength {
                tensor: tensor.name.clone(),
            });
        }
        maximum_magnitude = values
            .iter()
            .fold(maximum_magnitude, |maximum, value| maximum.max(value.abs()));
        remaining_elements -= elements as u64;
    }
    if maximum_magnitude == 0.0 {
        return Ok(1.0);
    }
    Ok(maximum_magnitude / layout.max_normal())
}

fn parse_format(format_name: &str) -> Result<FloatLayout, ColdArtifactError> {
    match format_name {
        "e5m2" => Ok(FloatLayout::E5M2_SATURATING),
        "e3m2" => Ok(FloatLayout::E3M2),
        "e2m2" => Ok(FloatLayout::E2M2),
        _ => Err(ColdArtifactError::UnsupportedFormat {
            format: String::from(format_name),
        }),
    }
}

fn layout_name(layout: FloatLayout) -> &'static str {
    if layout.same_format(&FloatLayout::E5M2_SATURATING) {
        "e5m2"
    } else if layout.same_format(&FloatLayout::E3M2) {
        "e3m2"
    } else if layout.same_format(&FloatLayout::E2M2) {
        "e2m2"
    } else {
        "custom"
    }
}

fn safe_shard_path(source_dir: &Path, shard_name: &str) -> Result<PathBuf, ColdArtifactError> {
    if Path::new(shard_name)
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(ColdArtifactError::InvalidShardPath {
            path: String::from(shard_name),
        });
    }
    Ok(source_dir.join(shard_name))
}

fn read_shard_header(
    shard_file: &mut File,
) -> Result<(SafetensorsParser, Manifest, u64, Sha256), ColdArtifactError> {
    let mut prefix = [0_u8; 8];
    shard_file.read_exact(&mut prefix)?;
    let header_length = u64::from_le_bytes(prefix);
    if header_length > MAX_HEADER_BYTES {
        return Err(proxima_safetensors::SafetensorsError::HeaderTooLarge {
            declared: header_length,
            max: MAX_HEADER_BYTES,
        }
        .into());
    }
    let header_length = usize::try_from(header_length).map_err(|_| {
        proxima_safetensors::SafetensorsError::HeaderTooLarge {
            declared: u64::MAX,
            max: MAX_HEADER_BYTES,
        }
    })?;
    let mut header = vec![0_u8; 8_usize.saturating_add(header_length)];
    header[..8].copy_from_slice(&prefix);
    shard_file.read_exact(&mut header[8..])?;

    let mut parser = SafetensorsParser::new();
    parser.feed(&header);
    let manifest = match parser.poll()? {
        Outcome::Event(manifest) => manifest.clone(),
        Outcome::NeedMore => return Err(ColdArtifactError::MissingManifest),
    };
    let data_start = 8_u64.saturating_add(header_length as u64);
    let mut shard_hasher = Sha256::new();
    shard_hasher.update(&header);
    Ok((parser, manifest, data_start, shard_hasher))
}

fn consume_gap(
    input: &mut File,
    mut remaining: u64,
    buffer: &mut [u8],
    parser: &mut SafetensorsParser,
    shard_hasher: &mut Sha256,
    source_bytes_processed: &mut u64,
) -> Result<(), ColdArtifactError> {
    while remaining > 0 {
        let length = remaining.min(buffer.len() as u64) as usize;
        let chunk = &mut buffer[..length];
        input.read_exact(chunk)?;
        parser.feed(chunk);
        shard_hasher.update(chunk);
        *source_bytes_processed += length as u64;
        remaining -= length as u64;
    }
    Ok(())
}

fn tensor_element_count(tensor: &TensorEntry) -> Result<u64, ColdArtifactError> {
    tensor.shape.iter().try_fold(1_u64, |product, axis| {
        product
            .checked_mul(*axis)
            .ok_or_else(|| ColdArtifactError::InvalidTensorLength {
                tensor: tensor.name.clone(),
            })
    })
}

fn hash_file(path: &Path, buffer_length: usize) -> Result<String, ColdArtifactError> {
    let mut file = File::open(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            ColdArtifactError::MissingSource {
                path: path.to_path_buf(),
            }
        } else {
            ColdArtifactError::Io(error)
        }
    })?;
    let mut buffer = vec![0_u8; buffer_length];
    let mut hasher = Sha256::new();
    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    Ok(digest_hex(hasher.finalize()))
}

fn digest_hex(digest: impl AsRef<[u8]>) -> String {
    digest
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        ColdArtifactError, ColdScaleMode, SOURCE_METADATA_FILES, compare_checkpoint,
        convert_checkpoint, convert_checkpoint_with_scale, parse_format, safe_shard_path,
    };
    use proxima_gguf::quant::cold_float::dequantize;
    use proxima_primitives::float_layout::FloatLayout;
    use proxima_safetensors::{SafetensorsModel, TensorPayload, write_complete};
    use proxima_tensor::DType;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::Path;

    #[test]
    fn cold_artifact_format_names_are_explicit() {
        assert!(parse_format("e5m2").is_ok());
        assert!(parse_format("e3m2").is_ok());
        assert!(parse_format("e2m2").is_ok());
        assert!(matches!(
            parse_format("bf6"),
            Err(ColdArtifactError::UnsupportedFormat { .. })
        ));
    }

    #[test]
    fn shard_paths_reject_parent_and_absolute_components() {
        assert!(safe_shard_path(Path::new("/checkpoint"), "model-00001.safetensors").is_ok());
        assert!(matches!(
            safe_shard_path(Path::new("/checkpoint"), "../outside.safetensors"),
            Err(ColdArtifactError::InvalidShardPath { .. })
        ));
        assert!(matches!(
            safe_shard_path(Path::new("/checkpoint"), "/outside.safetensors"),
            Err(ColdArtifactError::InvalidShardPath { .. })
        ));
    }

    #[test]
    fn conversion_rejects_output_inside_the_source_checkpoint() {
        let source_dir = tempfile::tempdir().expect("create source directory");
        assert!(matches!(
            convert_checkpoint(
                source_dir.path(),
                &source_dir.path().join("packed"),
                FloatLayout::E3M2,
                "ibm-granite/granite-4.0-micro-base",
                "98e1d641451818f6c8e6b17c1c359b5b4ac495d0",
                8,
            ),
            Err(ColdArtifactError::OutputInsideSource { .. })
        ));
    }

    #[test]
    fn converts_real_safetensors_wire_bytes_into_hashed_packed_artifact() {
        let source_dir = tempfile::tempdir().expect("create source checkpoint fixture");
        let output_root = tempfile::tempdir().expect("create artifact parent fixture");
        let output_dir = output_root.path().join("packed");
        for filename in SOURCE_METADATA_FILES {
            let contents = match filename {
                "config.json" => br#"{"model_type":"granitemoehybrid"}"#.as_slice(),
                "model.safetensors.index.json" => br#"{"metadata":{"total_size":16},"weight_map":{"layer.weight":"model.safetensors","layer.f32_weight":"model.safetensors"}}"#.as_slice(),
                _ => b"pinned tokenizer fixture".as_slice(),
            };
            fs::write(source_dir.path().join(filename), contents)
                .expect("write pinned metadata fixture");
        }
        let source_values = [0x80, 0x3f, 0x00, 0xc0, 0x00, 0x3e, 0xcc, 0x3e];
        let extra_values = [1.5_f32.to_le_bytes(), (-3.0_f32).to_le_bytes()];
        let extra_input = extra_values.concat();
        let source_model = SafetensorsModel {
            tensors: vec![
                TensorPayload {
                    name: String::from("layer.weight"),
                    dtype: DType::BFloat16,
                    shape: vec![4],
                    data: &source_values,
                },
                TensorPayload {
                    name: String::from("layer.f32_weight"),
                    dtype: DType::Float32,
                    shape: vec![2],
                    data: &extra_input,
                },
            ],
            metadata: BTreeMap::new(),
        };
        let source_shard =
            write_complete(&source_model).expect("write standard safetensors fixture");
        fs::write(source_dir.path().join("model.safetensors"), source_shard)
            .expect("write safetensors shard fixture");

        let manifest = convert_checkpoint(
            source_dir.path(),
            &output_dir,
            FloatLayout::E3M2,
            "ibm-granite/granite-4.0-micro-base",
            "98e1d641451818f6c8e6b17c1c359b5b4ac495d0",
            8,
        )
        .expect("convert indexed BF16 and F32 tensors");

        assert_eq!(manifest.tensors.len(), 2);
        assert_eq!(manifest.source_bytes_processed, 16);
        assert_eq!(manifest.output_bytes_written, 5);
        assert_eq!(manifest.tensors[0].name, "layer.weight");
        assert_eq!(manifest.tensors[0].source_shape, [4]);
        assert_eq!(manifest.tensors[0].source_dtype, "BF16");
        assert_eq!(manifest.layout, FloatLayout::E3M2);
        assert_eq!(manifest.tensors[0].layout, FloatLayout::E3M2);
        let manifest_value: serde_json::Value = serde_json::from_slice(
            &fs::read(output_dir.join("manifest.json")).expect("read layout-bearing manifest"),
        )
        .expect("manifest stores its structural float layout");
        assert_eq!(manifest_value["layout"]["exponent_bits"], 3);
        assert_eq!(manifest_value["layout"]["mantissa_bits"], 2);
        assert_eq!(manifest.tensors[1].name, "layer.f32_weight");
        assert_eq!(manifest.tensors[1].source_shape, [2]);
        assert_eq!(manifest.tensors[1].source_dtype, "F32");
        assert_eq!(manifest.tensors[0].source_sha256.len(), 64);
        assert_eq!(manifest.tensors[0].output_sha256.len(), 64);
        assert_eq!(
            manifest.source_artifacts.len(),
            SOURCE_METADATA_FILES.len() + 1
        );
        let packed = fs::read(output_dir.join("tensor-000000.pxw"))
            .expect("read emitted packed tensor artifact");
        let mut decoded = [0.0_f32; 4];
        dequantize(FloatLayout::E3M2, &packed, &mut decoded).expect("decode emitted E3M2 tensor");
        assert_eq!(decoded, [1.0, -2.0, 0.125, 0.375]);
        let packed_f32 = fs::read(output_dir.join("tensor-000001.pxw"))
            .expect("read emitted F32-derived packed artifact");
        let mut decoded_f32 = [0.0_f32; 2];
        dequantize(FloatLayout::E3M2, &packed_f32, &mut decoded_f32)
            .expect("decode emitted F32-derived E3M2 tensor");
        assert_eq!(decoded_f32, [1.5, -3.0]);
        let comparison = compare_checkpoint(source_dir.path(), &output_dir)
            .expect("compare emitted values with indexed BF16 and F32 source tensors");
        assert_eq!(comparison.source_values, 6);
        assert_eq!(comparison.changed_values, 1);
        assert_eq!(comparison.maximum_absolute_error, 0.0234375);
        assert_eq!(comparison.saturated_values, 0);
        assert_eq!(comparison.zeroed_nonzero_values, 0);
        assert_eq!(
            comparison.tensors[0]
                .maximum_error_sample
                .as_ref()
                .map(|sample| sample.index),
            Some(3)
        );
        assert_eq!(comparison.tensors[0].samples[3].source, 0.3984375);
        assert_eq!(comparison.tensors[0].samples[3].reconstructed, 0.375);
        let scaled_output = output_root.path().join("scaled");
        convert_checkpoint_with_scale(
            source_dir.path(),
            &scaled_output,
            FloatLayout::E3M2,
            "ibm-granite/granite-4.0-micro-base",
            "98e1d641451818f6c8e6b17c1c359b5b4ac495d0",
            8,
            ColdScaleMode::TensorMaximum,
        )
        .expect("convert fixture tensors with per-tensor maximum scaling");
        let scaled_comparison = compare_checkpoint(source_dir.path(), &scaled_output)
            .expect("compare scaled values with indexed source tensors");
        assert_eq!(scaled_comparison.scale_mode, "tensor_maximum");
        assert_eq!(scaled_comparison.saturated_values, 0);
        assert_eq!(scaled_comparison.tensors[0].maximum_source_magnitude, 2.0);
        assert_eq!(
            scaled_comparison.tensors[0].maximum_reconstructed_magnitude,
            2.0
        );
        assert!(output_dir.join("manifest.json").is_file());
    }
}
