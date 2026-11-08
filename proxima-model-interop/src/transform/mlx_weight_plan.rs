//! Every output tensor of an MLX export, named and shaped from the GGUF
//! directory alone, and the conversion of its weights a block of rows at a
//! time.
//!
//! [`WeightPlan::headers`] fixes each tensor's name, dtype and shape before a
//! byte is converted, so a safetensors header can be written first. Blocks
//! decode, quantize and cast independently, so peak memory is the blocks in
//! flight, never a whole tensor as `f32`. The stored leaves are those of
//! [`super::export_plain_named`]; with a quantization every `.weight` matrix
//! becomes the `weight`, `scales` and `biases` triple of
//! [`super::export_affine`]. A matrix whose last axis is not a multiple of the
//! group size stays float, as in mlx-lm.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;
use std::panic::resume_unwind;
use std::sync::atomic::{AtomicUsize, Ordering};

use proxima_gguf::TensorInfo;
use proxima_gguf::pipe::ParsedGguf;
use proxima_safetensors::TensorHeader;
use proxima_numeric::DType;

use super::mlx_affine::{AffineExport, AffineParams, export_affine};
use super::mlx_conventions::QuantizedConventions;
use super::mlx_names::{FusedName, Layout, MlxNames};
use super::mlx_plain::{PlainTensor, RowReader, cast_row};
use super::mlx_plain_named::leaf_names;
use crate::error::InteropError;
use crate::profiles::BindingProfile;

type Directory<'parsed> = BTreeMap<&'parsed str, &'parsed TensorInfo>;

pub(super) type Quantization = Option<(AffineParams, QuantizedConventions)>;

const BLOCK_ELEMENTS: usize = 1 << 21;

struct Unit<'parsed> {
    name: String,
    parts: Vec<&'parsed TensorInfo>,
    fused: bool,
    shape: Vec<u64>,
    row_elements: usize,
    rows_per_expert: u64,
}

impl<'parsed> Unit<'parsed> {
    fn new(name: String, parts: Vec<&'parsed TensorInfo>, fused: bool) -> Option<Self> {
        let first = parts.first()?;
        let mut shape: Vec<u64> = first.dims.iter().rev().copied().collect();
        if fused && let Some(width) = shape.get_mut(1) {
            *width *= parts.len() as u64;
        }
        let row_elements = first.dims.first().copied().unwrap_or(0);
        let experts = first.dims.last().copied().unwrap_or(1);
        let rows_per_expert = first
            .element_count()
            .checked_div(row_elements)
            .and_then(|rows| rows.checked_div(experts))
            .unwrap_or(1);
        Some(Self {
            name,
            parts,
            fused,
            shape,
            row_elements: usize::try_from(row_elements).unwrap_or(0),
            rows_per_expert,
        })
    }

    fn columns(&self) -> usize {
        self.shape
            .last()
            .map_or(0, |dim| usize::try_from(*dim).unwrap_or(0))
    }

    fn rows(&self) -> u64 {
        self.shape
            .split_last()
            .map_or(0, |(_, lead)| lead.iter().product())
    }

    fn source_rows_per_row(&self) -> u64 {
        (self.columns() / self.row_elements.max(1)).max(1) as u64
    }

    fn source_row(&self, index: u64) -> (usize, u64) {
        if !self.fused {
            return (0, index);
        }
        let per_expert = self.rows_per_expert.max(1);
        let span = per_expert * self.parts.len() as u64;
        let expert = index / span;
        let within = index % span;
        (
            usize::try_from(within / per_expert).unwrap_or(0),
            expert * per_expert + within % per_expert,
        )
    }

    fn blocks(&self, unit: usize, block_elements: usize) -> Vec<WeightBlock> {
        let rows = self.rows();
        let per_block = (block_elements / self.columns().max(1)).max(1);
        let starts: Vec<u64> = (0..rows).step_by(per_block).collect();
        if starts.is_empty() {
            return vec![WeightBlock {
                unit,
                rows: 0..0,
                first: true,
                last: true,
            }];
        }
        starts
            .iter()
            .enumerate()
            .map(|(position, &start)| WeightBlock {
                unit,
                rows: start..(start + per_block as u64).min(rows),
                first: position == 0,
                last: position + 1 == starts.len(),
            })
            .collect()
    }
}

/// A run of output rows of one unit; the unit of parallel work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct WeightBlock {
    unit: usize,
    rows: Range<u64>,
    first: bool,
    last: bool,
}

/// The bytes one [`WeightBlock`] converts to: one entry per output tensor of
/// its unit (`weight`, `scales`, `biases` when quantized), each holding the
/// block's rows of that tensor.
pub(super) struct WeightChunk {
    pub(super) unit: usize,
    pub(super) first: bool,
    pub(super) last: bool,
    pub(super) parts: Vec<Vec<u8>>,
}

pub(super) struct WeightPlan<'parsed> {
    parsed: &'parsed ParsedGguf,
    file_bytes: &'parsed [u8],
    units: Vec<Unit<'parsed>>,
    quantization: Quantization,
    dtype: DType,
    pub(super) block_elements: usize,
}

impl<'parsed> WeightPlan<'parsed> {
    /// # Errors
    ///
    /// [`InteropError::MlxUnmappedLeaves`] when a leaf has no stored tensor and
    /// no layout row.
    pub(super) fn new(
        parsed: &'parsed ParsedGguf,
        file_bytes: &'parsed [u8],
        dtype: DType,
        binding: &BindingProfile,
        layout: Layout,
        hf_prefix: &str,
        quantization: Quantization,
    ) -> Result<Self, InteropError> {
        let leaves = leaf_names(parsed, binding, layout, hf_prefix)?;
        let names = match quantization {
            Some((_, conventions)) => conventions.apply(leaves),
            None => leaves,
        };
        let directory: Directory = parsed
            .tensors
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor))
            .collect();
        let mut units = single_units(&names, &directory);
        units.extend(fused_units(&names, &directory));
        Ok(Self {
            parsed,
            file_bytes,
            units,
            quantization,
            dtype,
            block_elements: BLOCK_ELEMENTS,
        })
    }

    /// The header of every output tensor, in the order the weights are
    /// converted and written.
    ///
    /// # Errors
    ///
    /// [`InteropError::AffineBitsUnsupported`] for a convention outside
    /// mlx-lm's bit grid.
    pub(super) fn headers(&self) -> Result<Vec<TensorHeader>, InteropError> {
        Ok(self.unit_headers_all()?.into_iter().flatten().collect())
    }

    pub(super) fn blocks(&self) -> Vec<WeightBlock> {
        self.units
            .iter()
            .enumerate()
            .flat_map(|(index, unit)| unit.blocks(index, self.block_elements))
            .collect()
    }

    /// Converts `blocks` on `workers` threads; the chunks keep the order of
    /// `blocks`.
    ///
    /// # Errors
    ///
    /// Whatever decoding, casting or packing a block fails with.
    pub(super) fn convert(
        &self,
        blocks: &[WeightBlock],
        workers: usize,
    ) -> Result<Vec<WeightChunk>, InteropError> {
        ordered_map(blocks, workers, |block| self.convert_block(block))
    }

    /// Every output tensor whole: the eager form of [`Self::convert`].
    ///
    /// # Errors
    ///
    /// As [`Self::headers`] and [`Self::convert`].
    pub(super) fn convert_all(&self, workers: usize) -> Result<Vec<PlainTensor>, InteropError> {
        let headers = self.unit_headers_all()?;
        let mut joined: Vec<Vec<Vec<u8>>> = vec![Vec::new(); self.units.len()];
        for chunk in self.convert(&self.blocks(), workers)? {
            let slots = &mut joined[chunk.unit];
            slots.resize_with(chunk.parts.len(), Vec::new);
            slots
                .iter_mut()
                .zip(chunk.parts)
                .for_each(|(slot, part)| slot.extend_from_slice(&part));
        }
        Ok(headers
            .into_iter()
            .flatten()
            .zip(joined.into_iter().flatten())
            .map(|(header, bytes)| PlainTensor {
                name: header.name,
                shape: header.shape,
                dtype: header.dtype,
                bytes,
            })
            .collect())
    }

    fn unit_headers_all(&self) -> Result<Vec<Vec<TensorHeader>>, InteropError> {
        self.units
            .iter()
            .map(|unit| self.unit_headers(unit))
            .collect()
    }

    fn packing(&self, unit: &Unit) -> Result<Option<AffineParams>, InteropError> {
        let Some((base, conventions)) = self.quantization else {
            return Ok(None);
        };
        let params = match unit.fused {
            true => base,
            false => conventions.params_for(&unit.name, base)?,
        };
        let quantizable = unit.name.ends_with(".weight") && unit.shape.len() >= 2;
        let aligned = unit.columns().is_multiple_of(params.group_size());
        Ok((quantizable && aligned).then_some(params))
    }

    fn unit_headers(&self, unit: &Unit) -> Result<Vec<TensorHeader>, InteropError> {
        let Some(params) = self.packing(unit)? else {
            return Ok(vec![TensorHeader {
                name: unit.name.clone(),
                dtype: self.dtype,
                shape: unit.shape.clone(),
            }]);
        };
        let base = unit.name.strip_suffix(".weight").unwrap_or(&unit.name);
        let lead = unit.shape.split_last().map_or(&[][..], |(_, lead)| lead);
        let shaped = |last: usize| [lead, &[last as u64]].concat();
        let columns = unit.columns();
        let groups = columns / params.group_size();
        let float = |suffix: &str| TensorHeader {
            name: format!("{base}.{suffix}"),
            dtype: self.dtype,
            shape: shaped(groups),
        };
        Ok(vec![
            TensorHeader {
                name: format!("{base}.weight"),
                dtype: DType::UINT32,
                shape: shaped(columns * usize::from(params.bits()) / 32),
            },
            float("scales"),
            float("biases"),
        ])
    }

    fn convert_block(&self, block: &WeightBlock) -> Result<WeightChunk, InteropError> {
        let unit = &self.units[block.unit];
        let rows = usize::try_from(block.rows.end - block.rows.start).unwrap_or(0);
        let mut values = vec![0.0_f32; rows * unit.columns()];
        self.decode_rows(unit, &block.rows, &mut values)?;
        let parts = match self.packing(unit)? {
            Some(params) => pack_block(unit, &values, rows, params, self.dtype)?,
            None => vec![cast_block(unit, &values, self.dtype)?],
        };
        Ok(WeightChunk {
            unit: block.unit,
            first: block.first,
            last: block.last,
            parts,
        })
    }

    fn decode_rows(
        &self,
        unit: &Unit,
        rows: &Range<u64>,
        values: &mut [f32],
    ) -> Result<(), InteropError> {
        let readers = unit
            .parts
            .iter()
            .map(|info| RowReader::new(self.parsed, self.file_bytes, info))
            .collect::<Result<Vec<_>, _>>()?;
        let per_row = unit.source_rows_per_row();
        let sources = rows.start * per_row..rows.end * per_row;
        for (index, slot) in sources.zip(values.chunks_exact_mut(unit.row_elements.max(1))) {
            let (part, row) = unit.source_row(index);
            readers[part].decode(row, slot)?;
        }
        Ok(())
    }
}

fn cast_block(unit: &Unit, values: &[f32], dtype: DType) -> Result<Vec<u8>, InteropError> {
    let mut bytes = Vec::with_capacity(values.len() * dtype.size_bytes());
    cast_row(dtype, values, &mut bytes).ok_or_else(|| InteropError::UnrepresentableDType {
        tensor: unit.name.clone(),
        dtype,
    })?;
    Ok(bytes)
}

fn pack_block(
    unit: &Unit,
    values: &[f32],
    rows: usize,
    params: AffineParams,
    dtype: DType,
) -> Result<Vec<Vec<u8>>, InteropError> {
    match export_affine(values, [rows, unit.columns()], params, dtype)? {
        AffineExport::Packed(packed) => Ok(vec![
            packed
                .weight
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect(),
            packed.scales,
            packed.biases,
        ]),
        AffineExport::Float => Err(InteropError::MlxWeightPlanMismatch {
            tensor: unit.name.clone(),
        }),
    }
}

fn single_units<'parsed>(names: &MlxNames, directory: &Directory<'parsed>) -> Vec<Unit<'parsed>> {
    names
        .mapped
        .iter()
        .filter_map(|(leaf, hf)| {
            let info = directory.get(leaf.as_str()).copied()?;
            Unit::new(hf.clone(), vec![info], false)
        })
        .collect()
}

fn fused_units<'parsed>(names: &MlxNames, directory: &Directory<'parsed>) -> Vec<Unit<'parsed>> {
    let targets: BTreeSet<&str> = names.fused.iter().map(|item| item.hf.as_str()).collect();
    targets
        .into_iter()
        .filter_map(|target| {
            let mut group: Vec<&FusedName> = names
                .fused
                .iter()
                .filter(|item| item.hf == target)
                .collect();
            group.sort_by_key(|item| item.part);
            let parts: Option<Vec<&TensorInfo>> = group
                .iter()
                .map(|item| directory.get(item.leaf.as_str()).copied())
                .collect();
            Unit::new(target.into(), parts?, true)
        })
        .collect()
}

pub(super) fn available_workers() -> usize {
    std::thread::available_parallelism().map_or(1, |count| count.get())
}

fn ordered_map<Item: Sync, Output: Send, Failure: Send>(
    items: &[Item],
    workers: usize,
    work: impl Fn(&Item) -> Result<Output, Failure> + Sync,
) -> Result<Vec<Output>, Failure> {
    if workers <= 1 {
        return items.iter().map(&work).collect();
    }
    let next = AtomicUsize::new(0);
    let mut claimed: Vec<(usize, Result<Output, Failure>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers.min(items.len()))
            .map(|_| scope.spawn(|| claim_in_turn(items, &next, &work)))
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().unwrap_or_else(|panic| resume_unwind(panic)))
            .collect()
    });
    claimed.sort_unstable_by_key(|(index, _)| *index);
    claimed.into_iter().map(|(_, result)| result).collect()
}

fn claim_in_turn<Item, Output, Failure>(
    items: &[Item],
    next: &AtomicUsize,
    work: &impl Fn(&Item) -> Result<Output, Failure>,
) -> Vec<(usize, Result<Output, Failure>)> {
    std::iter::from_fn(|| {
        let index = next.fetch_add(1, Ordering::Relaxed);
        items.get(index).map(|item| (index, work(item)))
    })
    .collect()
}
