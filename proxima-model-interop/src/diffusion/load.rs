//! A checkpoint reader for the diffusion graphs: binds each declared weight
//! leaf from a safetensors file, re-encoding it into the packed codec the
//! backend accelerates and reordering a convolution kernel's axes so a tap's
//! reduce reads its channels contiguously.

use std::path::Path;

use proxima_numeric::codec::Codec;
use proxima_numeric::{DType, FormatCodec};
use proxima_tensor::QuantizedBlock;

use crate::CheckpointMapping;

use super::WeightBinding;

/// One bound leaf's bytes: a packed matrix the backend's matmul paths decode,
/// or the plain scalar vector a per-channel scale or bias binds (a packed
/// rank-1 operand has no matmul reader, so it stays plain).
#[derive(Debug)]
pub enum WeightData {
    Packed(Codec, Vec<u8>),
    Plain(Vec<f32>),
}

/// The bound leaves, each holding the exact bytes the plan reads.
#[derive(Debug)]
pub struct Weights {
    pub codec: Codec,
    pub entries: Vec<(String, WeightData)>,
}

impl Weights {
    /// The `(leaf name, block)` pairs a `plan_named`/`execute_plan_named` call
    /// binds.
    #[must_use]
    pub fn blocks(&self) -> Vec<(&str, QuantizedBlock<'_>)> {
        self.entries
            .iter()
            .map(|(name, data)| {
                let block = match data {
                    WeightData::Packed(codec, bytes) => QuantizedBlock::Packed {
                        codec: *codec,
                        bytes: bytes.as_slice(),
                    },
                    WeightData::Plain(values) => QuantizedBlock::Float32(values),
                };
                (name.as_str(), block)
            })
            .collect()
    }
}

/// One safetensors file's tensor directory over its mapped bytes: the file is
/// addressed in place through [`CheckpointMapping`], never copied to the heap.
pub struct Checkpoint {
    mapping: CheckpointMapping,
    manifest: proxima_safetensors::Manifest,
    data_start: usize,
}

impl Checkpoint {
    /// Maps `path` and parses its header.
    ///
    /// # Errors
    ///
    /// A message naming the file and the mapping or parse failure.
    pub fn open(path: &Path) -> Result<Self, String> {
        let mapping = CheckpointMapping::open_file(path)
            .map_err(|why| format!("{}: {why}", path.display()))?;
        let bytes = mapping.bytes();
        let manifest = proxima_safetensors::parse_complete(bytes)
            .map_err(|why| format!("{}: {why}", path.display()))?;
        if bytes.len() < 8 {
            return Err(format!("{}: truncated header", path.display()));
        }
        let header = u64::from_le_bytes(
            bytes[..8]
                .try_into()
                .map_err(|_| format!("{}: truncated length", path.display()))?,
        );
        Ok(Self {
            mapping,
            manifest,
            data_start: 8 + header as usize,
        })
    }

    /// The tensor directory parsed from the file's header.
    #[must_use]
    pub fn manifest(&self) -> &proxima_safetensors::Manifest {
        &self.manifest
    }

    /// The whole file's mapped bytes, the view a loader borrows weights from.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        self.mapping.bytes()
    }

    /// Where tensor data begins in [`Self::bytes`], past the length prefix and header.
    #[must_use]
    pub fn data_start(&self) -> u64 {
        self.data_start as u64
    }

    /// One tensor's `(dtype, shape, bytes)`, offsets already absolute.
    ///
    /// # Errors
    ///
    /// A message naming the tensor when the header does not carry it.
    pub fn tensor(&self, name: &str) -> Result<(DType, Vec<u64>, &[u8]), String> {
        let entry = self
            .manifest
            .tensor(name)
            .ok_or_else(|| format!("checkpoint has no tensor {name}"))?;
        let start = self.data_start + entry.data_offsets.0 as usize;
        let end = self.data_start + entry.data_offsets.1 as usize;
        Ok((
            entry.dtype,
            entry.shape.clone(),
            &self.mapping.bytes()[start..end],
        ))
    }

    /// Decodes one tensor to `f32`, optionally slicing rows and permuting a
    /// convolution kernel's `[out, in, k, k]` axes to `[out, k, k, in]`.
    ///
    /// # Errors
    ///
    /// A message naming the tensor and the failing step.
    pub fn values(
        &self,
        name: &str,
        rows: Option<(u32, u32)>,
        columns: Option<(u32, u32)>,
        permute_kernel: bool,
    ) -> Result<(Vec<u64>, Vec<f32>), String> {
        let (dtype, shape, bytes) = self.tensor(name)?;
        let format = dtype
            .format()
            .ok_or_else(|| format!("{name}: no numeric format"))?;
        let codec = FormatCodec::for_format(&format)
            .ok_or_else(|| format!("{name}: no codec for its format"))?;
        let per_row: u64 = shape.iter().skip(1).product();
        let row_bytes = bytes.len() / (shape[0].max(1) as usize);
        let (first, last) = rows.unwrap_or((0, shape[0] as u32));
        let available = bytes.len() / row_bytes.max(1);
        let last = (last as usize).min(available);
        let first = (first as usize).min(last);
        let start = first * row_bytes;
        let end = last * row_bytes;
        let count = (last - first) * per_row as usize;
        let mut values = vec![0.0f32; count];
        let mut slots: Vec<core::mem::MaybeUninit<f32>> =
            vec![core::mem::MaybeUninit::uninit(); count];
        codec
            .decode(&bytes[start..end], &mut slots)
            .map_err(|why| format!("{name}: {why}"))?;
        for (slot, value) in slots.into_iter().enumerate() {
            values[slot] = unsafe { value.assume_init() };
        }
        let mut shape = shape;
        shape[0] = (last - first) as u64;
        if let Some((from, to)) = columns {
            let width = shape[1] as usize;
            let selected = (to - from) as usize;
            let mut windowed = Vec::with_capacity(values.len() / width * selected);
            for row in values.chunks(width) {
                windowed.extend_from_slice(&row[from as usize..(from as usize + selected).min(width)]);
            }
            values = windowed;
            shape[1] = selected as u64;
        }
        if permute_kernel && shape.len() == 4 {
            values = permute_kernel_axes(&values, &shape);
            shape = vec![shape[0], shape[2], shape[3], shape[1]];
        }
        Ok((shape, values))
    }
}

/// `[out, in, k, k]` row-major to `[out, k, k, in]`.
fn permute_kernel_axes(values: &[f32], shape: &[u64]) -> Vec<f32> {
    let (out, into, kernel, columns) = (
        shape[0] as usize,
        shape[1] as usize,
        shape[2] as usize,
        shape[3] as usize,
    );
    let mut permuted = vec![0.0f32; values.len()];
    for channel_out in 0..out {
        for channel_in in 0..into {
            for row in 0..kernel {
                for column in 0..columns {
                    let source = ((channel_out * into + channel_in) * kernel + row) * columns + column;
                    let target = ((channel_out * kernel + row) * columns + column) * into + channel_in;
                    permuted[target] = values[source];
                }
            }
        }
    }
    permuted
}

/// Encodes `values` into `codec`'s packed bytes.
///
/// # Errors
///
/// A message naming the format when it has no codec.
pub fn encode(values: &[f32], codec: Codec) -> Result<Vec<u8>, String> {
    let encoder = FormatCodec::for_format(&codec)
        .ok_or_else(|| format!("{codec:?}: no encoder"))?;
    let mut bytes = vec![
        0u8;
        encoder
            .packed_len(values.len())
            .ok_or_else(|| format!("{codec:?}: no packed length"))?
    ];
    encoder
        .encode(values, &mut bytes)
        .map_err(|why| why.to_string())?;
    Ok(bytes)
}

/// Binds every declared weight leaf, `codec`-encoded, with each convolution
/// kernel reordered to `[out, k, k, in]`.
///
/// # Errors
///
/// A message naming the leaf and the failing step.
pub fn load_weights(
    checkpoint: &Checkpoint,
    bindings: &[WeightBinding],
    codec: Codec,
    permute_kernels: bool,
) -> Result<Weights, String> {
    let mut entries = Vec::with_capacity(bindings.len());
    for binding in bindings {
#[allow(unused_mut)]
        let (mut shape, mut values) = checkpoint.values(
            &binding.tensor,
            binding.rows,
            binding.columns,
            permute_kernels
                && (binding.tap.is_some() || binding.taps || binding.tensor.ends_with("conv.weight")),
        )?;
        let (shape, values) = match binding.tap {
            Some((row, column)) => (
                alloc::vec![shape[0], shape[3]],
                kernel_tap(&values, &shape, row, column),
            ),
            None => (shape, values),
        };
        let data = if shape.len() == 1 {
            WeightData::Plain(values)
        } else {
            WeightData::Packed(codec, encode(&values, codec)?)
        };
        entries.push((binding.leaf.clone(), data));
    }
    Ok(Weights { codec, entries })
}

/// `[out, k, k, in]`'s kernel position `(row, column)` as a contiguous
/// `[out, in]` matrix.
fn kernel_tap(values: &[f32], shape: &[u64], row: u32, column: u32) -> Vec<f32> {
    let out = shape[0] as usize;
    let kernel = shape[1] as usize;
    let columns = shape[2] as usize;
    let into = shape[3] as usize;
    let mut tapped = Vec::with_capacity(out * into);
    for channel_out in 0..out {
        let base = ((channel_out * kernel + row as usize) * columns + column as usize) * into;
        tapped.extend_from_slice(&values[base..base + into]);
    }
    tapped
}
