//! Sequential low-codec expert sidecars for HOBBIT-backed residency.
//!
//! The source is a parsed GGUF directory plus a borrowed file mapping. Each
//! expert is decoded and re-encoded into caller-owned buffers before its
//! bytes are written. The sidecar therefore never constructs a packed
//! whole-model buffer; its descriptor table is the only metadata allocation.

use std::alloc::{Layout, alloc, dealloc};
use std::collections::VecDeque;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::Arc;

use memmap2::Mmap;
use proxima_gguf::{GgmlType, ParsedGguf};

use crate::bind::PackedOwnedKind;
use crate::expert_slab::{ExpertProjection, ExpertSlab};
use crate::residency::{ExpertAddress, ResidencyAction};
use crate::{InteropError, recode_expert_into};
use proxima_tensor::cpu::{ExpertEntry, ExpertPayloadSpan};

const MAGIC: &[u8; 8] = b"PXEXSC01";
const VERSION: u32 = 1;
const MAX_PROJECTION_BYTES: usize = u16::MAX as usize;
const DESCRIPTOR_FIXED_BYTES: u64 = 4 + 4 + 4 + 4 + 1 + 1 + 2 + 8 + 8 + 8;

/// One stacked GGUF expert tensor to include in a sidecar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertStackSpec<'name> {
    pub layer: u32,
    pub projection: &'name str,
    pub tensor_name: &'name str,
    pub expert_count: u32,
    pub out_dim: u32,
    pub in_dim: u32,
    pub target_codec: PackedOwnedKind,
}

/// Descriptor returned for each expert written to the sidecar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertSidecarDescriptor {
    pub layer: u32,
    pub projection: String,
    pub expert: u32,
    pub out_dim: u32,
    pub in_dim: u32,
    pub source_codec: PackedOwnedKind,
    pub target_codec: PackedOwnedKind,
    pub source_offset: u64,
    pub source_bytes: u64,
    pub data_offset: u64,
    pub data_bytes: u64,
}

/// Sidecar result, including its descriptors and total bytes emitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertSidecar {
    pub descriptors: Vec<ExpertSidecarDescriptor>,
    pub data_offset: u64,
    pub total_bytes: u64,
}

/// Parsed sidecar metadata and the read-only mapping that owns its payload.
///
/// Descriptor lookup is a dense `(layer, expert, projection)` table built
/// once at attachment. A router boundary therefore selects all three expert
/// projections in O(1), clones only the mapping handle, and never allocates or
/// copies weight bytes.
#[derive(Debug, Clone)]
pub struct MappedExpertSidecar {
    sidecar: ExpertSidecar,
    mapping: Arc<Mmap>,
    source_file: Option<Arc<File>>,
    checkpoint_file: Option<Arc<File>>,
    expert_count: usize,
    descriptor_indices: Vec<[Option<usize>; 3]>,
    high_bytes_per_expert: u64,
}

/// Reusable owner for the low-codec ranges selected by one routed layer.
#[derive(Debug, Default)]
pub(crate) struct ExpertSidecarReadScratch {
    buffers: Vec<Vec<u8>>,
    arenas: [PageAlignedBytes; 3],
    arena_used: [usize; 3],
    arena_spans: [Vec<Option<ExpertPayloadSpan>>; 3],
    high_cache: VecDeque<HighReadCacheEntry>,
    high_cache_bytes: usize,
    high_cache_limit: usize,
    descriptor_slots: Vec<[Option<usize>; 3]>,
    used_buffers: usize,
    pub(crate) ranges_read: usize,
    pub(crate) bytes_read: usize,
    pub(crate) low_ranges_read: usize,
    pub(crate) low_bytes_read: usize,
    pub(crate) high_ranges_read: usize,
    pub(crate) high_bytes_read: usize,
    pub(crate) high_cache_hits: usize,
    pub(crate) high_cache_misses: usize,
}

#[derive(Debug)]
struct PageAlignedBytes {
    pointer: NonNull<u8>,
    capacity: usize,
    length: usize,
}

impl Default for PageAlignedBytes {
    fn default() -> Self {
        Self {
            pointer: NonNull::dangling(),
            capacity: 0,
            length: 0,
        }
    }
}

impl PageAlignedBytes {
    fn reserve(&mut self, length: usize) -> Result<(), InteropError> {
        if length <= self.capacity {
            self.length = length;
            return Ok(());
        }
        let page_size = 4096usize;
        let capacity = length
            .checked_add(page_size - 1)
            .and_then(|value| value.checked_div(page_size))
            .and_then(|pages| pages.checked_mul(page_size))
            .ok_or(InteropError::SidecarSizeOverflow)?;
        let layout = Layout::from_size_align(capacity, page_size)
            .map_err(|_| InteropError::SidecarSizeOverflow)?;
        // SAFETY: the layout is page-aligned and the allocation is initialized
        // by every bounded pread before a slice is exposed to the tensor graph.
        let pointer =
            NonNull::new(unsafe { alloc(layout) }).ok_or(InteropError::SidecarSizeOverflow)?;
        if self.length != 0 {
            // SAFETY: both allocations are valid for the old initialized
            // length and do not overlap.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    self.pointer.as_ptr(),
                    pointer.as_ptr(),
                    self.length,
                );
            }
        }
        if self.capacity != 0 {
            let old_layout = Layout::from_size_align(self.capacity, page_size)
                .map_err(|_| InteropError::SidecarSizeOverflow)?;
            // SAFETY: pointer/layout are the exact prior allocation pair.
            unsafe { dealloc(self.pointer.as_ptr(), old_layout) };
        }
        self.pointer = pointer;
        self.capacity = capacity;
        self.length = length;
        Ok(())
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `reserve` owns an allocation for `length` bytes.
        unsafe { core::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.length) }
    }

    fn as_slice(&self) -> &[u8] {
        // SAFETY: `reserve` owns an allocation for `length` bytes.
        unsafe { core::slice::from_raw_parts(self.pointer.as_ptr(), self.length) }
    }
}

impl Drop for PageAlignedBytes {
    fn drop(&mut self) {
        if self.capacity != 0 {
            if let Ok(layout) = Layout::from_size_align(self.capacity, 4096) {
                // SAFETY: pointer/layout are the exact allocation pair.
                unsafe { dealloc(self.pointer.as_ptr(), layout) };
            }
        }
    }
}

#[derive(Debug)]
struct HighReadCacheEntry {
    layer: usize,
    expert: usize,
    projection: ExpertProjection,
    offset: u64,
    bytes: Vec<u8>,
}

impl ExpertSidecarReadScratch {
    pub(crate) fn with_high_cache_limit(high_cache_limit: usize) -> Self {
        Self {
            high_cache_limit,
            ..Self::default()
        }
    }

    pub(crate) fn bytes(&self, expert: usize, projection: ExpertProjection) -> Option<&[u8]> {
        if let Some(Some(span)) = self.arena_spans[projection.index()].get(expert) {
            let start = usize::try_from(span.offset).ok()?;
            let end = start.checked_add(usize::try_from(span.length).ok()?)?;
            return self.arenas[projection.index()].as_slice().get(start..end);
        }
        let buffer_index = self.descriptor_slots.get(expert)?[projection.index()]?;
        self.buffers.get(buffer_index).map(Vec::as_slice)
    }

    pub(crate) fn arena(
        &self,
        projection: ExpertProjection,
    ) -> Option<(&[u8], &[Option<ExpertPayloadSpan>])> {
        self.arena_spans[projection.index()]
            .iter()
            .any(Option::is_some)
            .then_some((
                self.arenas[projection.index()].as_slice(),
                self.arena_spans[projection.index()].as_slice(),
            ))
    }
}

impl MappedExpertSidecar {
    /// Parses a complete sidecar while retaining its mmap owner.
    pub fn new(mapping: Arc<Mmap>) -> Result<Self, InteropError> {
        Self::new_with_source_file(mapping, None)
    }

    /// Parses a sidecar while retaining an optional file for bounded reads.
    pub fn new_with_source_file(
        mapping: Arc<Mmap>,
        source_file: Option<Arc<File>>,
    ) -> Result<Self, InteropError> {
        let sidecar = ExpertSidecar::from_bytes(&mapping)?;
        let expert_count = sidecar
            .descriptors
            .iter()
            .map(|descriptor| descriptor.expert as usize + 1)
            .max()
            .unwrap_or(0);
        let layer_count = sidecar
            .descriptors
            .iter()
            .map(|descriptor| descriptor.layer as usize + 1)
            .max()
            .unwrap_or(0);
        let entry_count = layer_count
            .checked_mul(expert_count)
            .ok_or(InteropError::SidecarSizeOverflow)?;
        let mut descriptor_indices = vec![[None; 3]; entry_count];
        for (descriptor_index, descriptor) in sidecar.descriptors.iter().enumerate() {
            let projection = projection_from_name(&descriptor.projection)?;
            let address_index =
                descriptor.layer as usize * expert_count + descriptor.expert as usize;
            descriptor_indices[address_index][projection.index()] = Some(descriptor_index);
        }
        let high_bytes_per_expert = descriptor_indices
            .iter()
            .map(|indices| {
                indices
                    .iter()
                    .flatten()
                    .fold(0_u64, |bytes, &descriptor_index| {
                        bytes.saturating_add(sidecar.descriptors[descriptor_index].source_bytes)
                    })
            })
            .max()
            .unwrap_or(0);
        Ok(Self {
            sidecar,
            mapping,
            source_file,
            checkpoint_file: None,
            expert_count,
            descriptor_indices,
            high_bytes_per_expert,
        })
    }

    /// Opens a sidecar whose selected expert payloads are read from the file.
    pub fn from_file(source_file: File) -> Result<Self, InteropError> {
        let source_file = Arc::new(source_file);
        // SAFETY: the read-only mapping and its file owner are retained by
        // the returned sidecar for the complete attachment lifetime.
        let mapping = unsafe { memmap2::MmapOptions::new().map(source_file.as_ref()) }
            .map_err(InteropError::SidecarIo)?;
        mapping.advise(memmap2::Advice::Random)?;
        Self::new_with_source_file(Arc::new(mapping), Some(source_file))
    }

    #[must_use]
    pub(crate) const fn uses_file_reads(&self) -> bool {
        self.source_file.is_some()
    }

    pub(crate) const fn has_checkpoint_file(&self) -> bool {
        self.checkpoint_file.is_some()
    }

    #[must_use]
    pub(crate) fn mapping_bytes(&self) -> &[u8] {
        &self.mapping
    }

    pub(crate) fn populate_all_low_source<'mapping>(
        &'mapping self,
        layer: usize,
        projection: ExpertProjection,
        entries: &mut Vec<ExpertEntry<'mapping>>,
        spans: &mut Vec<Option<ExpertPayloadSpan>>,
    ) -> Result<Range<usize>, InteropError> {
        let mut arena_start = usize::MAX;
        let mut arena_end = 0usize;
        for expert in 0..self.expert_count {
            let descriptor = self.descriptor(ExpertAddress { layer, expert }, projection)?;
            let range = Self::checked_range(
                descriptor,
                descriptor.data_offset,
                descriptor.data_bytes,
                self.mapping.len(),
                descriptor.target_codec,
            )?;
            arena_start = arena_start.min(range.start);
            arena_end = arena_end.max(range.end);
        }
        if arena_start == usize::MAX {
            return Err(InteropError::ExpertSlabIndexOutOfRange { layer, expert: 0 });
        }

        entries.clear();
        spans.clear();
        spans.resize(self.expert_count, None);
        for expert in 0..self.expert_count {
            let descriptor = self.descriptor(ExpertAddress { layer, expert }, projection)?;
            let range = Self::checked_range(
                descriptor,
                descriptor.data_offset,
                descriptor.data_bytes,
                self.mapping.len(),
                descriptor.target_codec,
            )?;
            let offset = range
                .start
                .checked_sub(arena_start)
                .ok_or(InteropError::SidecarSizeOverflow)?;
            spans[expert] = Some(ExpertPayloadSpan {
                offset: u32::try_from(offset).map_err(|_| InteropError::SidecarSizeOverflow)?,
                length: u32::try_from(range.len())
                    .map_err(|_| InteropError::SidecarSizeOverflow)?,
            });
            entries.push(ExpertEntry {
                block: descriptor.target_codec.as_block(&self.mapping[range]),
                out_dim: descriptor.out_dim,
                in_dim: descriptor.in_dim,
                epoch: 0,
            });
        }
        Ok(arena_start..arena_end)
    }

    /// Retains a checkpoint file for bounded high-codec preads.
    pub(crate) fn with_checkpoint_file(mut self, file: File) -> Self {
        self.checkpoint_file = Some(Arc::new(file));
        self
    }

    /// Reads a range through the source file when available, avoiding mmap
    /// page faults for sparse expert payloads.
    pub fn read_range(&self, range: Range<usize>) -> Result<Vec<u8>, InteropError> {
        let mut bytes = Vec::new();
        self.read_range_into(range, &mut bytes)?;
        Ok(bytes)
    }

    fn read_range_into(
        &self,
        range: Range<usize>,
        bytes: &mut Vec<u8>,
    ) -> Result<(), InteropError> {
        let source =
            self.mapping
                .get(range.clone())
                .ok_or(InteropError::ExpertMappedRangeOutOfBounds {
                    start: range.start,
                    end: range.end,
                    mapping_len: self.mapping.len(),
                })?;
        bytes.resize(range.len(), 0);
        let Some(source_file) = &self.source_file else {
            bytes.copy_from_slice(source);
            return Ok(());
        };
        let mut reader = source_file.try_clone().map_err(InteropError::SidecarIo)?;
        reader
            .seek(SeekFrom::Start(range.start as u64))
            .map_err(InteropError::SidecarIo)?;
        reader.read_exact(bytes).map_err(InteropError::SidecarIo)
    }

    fn read_range_into_slice(
        &self,
        range: Range<usize>,
        bytes: &mut [u8],
    ) -> Result<(), InteropError> {
        let source =
            self.mapping
                .get(range.clone())
                .ok_or(InteropError::ExpertMappedRangeOutOfBounds {
                    start: range.start,
                    end: range.end,
                    mapping_len: self.mapping.len(),
                })?;
        if bytes.len() != range.len() {
            return Err(InteropError::ExpertMappedRangeOutOfBounds {
                start: range.start,
                end: range.end,
                mapping_len: bytes.len(),
            });
        }
        let Some(source_file) = &self.source_file else {
            bytes.copy_from_slice(source);
            return Ok(());
        };
        let mut reader = source_file.try_clone().map_err(InteropError::SidecarIo)?;
        reader
            .seek(SeekFrom::Start(range.start as u64))
            .map_err(InteropError::SidecarIo)?;
        reader.read_exact(bytes).map_err(InteropError::SidecarIo)
    }

    /// Reads routed low or high ranges into reusable bounded storage.
    #[cfg(test)]
    pub(crate) fn read_selected(
        &self,
        layer: usize,
        experts: &[u32],
        slab: &ExpertSlab<'_>,
        scratch: &mut ExpertSidecarReadScratch,
    ) -> Result<(), InteropError> {
        self.read_selected_with_checkpoint(layer, experts, slab, scratch, None)
    }

    /// Reads selected ranges, borrowing checkpoint bytes when supplied.
    pub(crate) fn read_selected_with_checkpoint(
        &self,
        layer: usize,
        experts: &[u32],
        slab: &ExpertSlab<'_>,
        scratch: &mut ExpertSidecarReadScratch,
        checkpoint_mapping: Option<&[u8]>,
    ) -> Result<(), InteropError> {
        scratch.used_buffers = 0;
        scratch.arena_used = [0; 3];
        for spans in &mut scratch.arena_spans {
            spans.clear();
        }
        scratch.ranges_read = 0;
        scratch.bytes_read = 0;
        scratch.low_ranges_read = 0;
        scratch.low_bytes_read = 0;
        scratch.high_ranges_read = 0;
        scratch.high_bytes_read = 0;
        scratch.high_cache_hits = 0;
        scratch.high_cache_misses = 0;
        scratch.descriptor_slots.clear();
        scratch
            .descriptor_slots
            .resize(self.expert_count, [None; 3]);
        for spans in &mut scratch.arena_spans {
            spans.resize(self.expert_count, None);
        }
        let mut arena_totals = [0usize; 3];
        for &expert in experts {
            let address = ExpertAddress {
                layer,
                expert: expert as usize,
            };
            for projection in ExpertProjection::ALL {
                if !slab.uses_mapped_low(layer, address.expert, projection)? {
                    continue;
                }
                let descriptor = self.descriptor(address, projection)?;
                arena_totals[projection.index()] = arena_totals[projection.index()]
                    .checked_add(
                        usize::try_from(descriptor.data_bytes)
                            .map_err(|_| InteropError::SidecarSizeOverflow)?,
                    )
                    .ok_or(InteropError::SidecarSizeOverflow)?;
            }
        }
        for (projection_index, total) in arena_totals.into_iter().enumerate() {
            scratch.arenas[projection_index].reserve(total)?;
        }
        for &expert in experts {
            let address = ExpertAddress {
                layer,
                expert: expert as usize,
            };
            for projection in ExpertProjection::ALL {
                let descriptor = self.descriptor(address, projection)?;
                let mapped_low = slab.uses_mapped_low(layer, address.expert, projection)?;
                let (offset, length, codec) = if mapped_low {
                    (
                        descriptor.data_offset,
                        descriptor.data_bytes,
                        descriptor.target_codec,
                    )
                } else {
                    (
                        descriptor.source_offset,
                        descriptor.source_bytes,
                        descriptor.source_codec,
                    )
                };
                let buffer_index = scratch.used_buffers;
                if buffer_index == scratch.buffers.len() {
                    scratch.buffers.push(Vec::new());
                }
                if mapped_low {
                    let range =
                        Self::checked_range(descriptor, offset, length, self.mapping.len(), codec)?;
                    let projection_index = projection.index();
                    let arena_offset = scratch.arena_used[projection_index];
                    let arena_end = arena_offset
                        .checked_add(range.len())
                        .ok_or(InteropError::SidecarSizeOverflow)?;
                    scratch.arenas[projection_index].reserve(arena_end)?;
                    self.read_range_into_slice(
                        range.clone(),
                        &mut scratch.arenas[projection_index].as_mut_slice()
                            [arena_offset..arena_end],
                    )?;
                    scratch.ranges_read += 1;
                    scratch.bytes_read += range.len();
                    scratch.low_ranges_read += 1;
                    scratch.low_bytes_read += range.len();
                    scratch.arena_used[projection_index] = arena_end;
                    scratch.arena_spans[projection_index][address.expert] =
                        Some(ExpertPayloadSpan {
                            offset: u32::try_from(arena_offset)
                                .map_err(|_| InteropError::SidecarSizeOverflow)?,
                            length: u32::try_from(range.len())
                                .map_err(|_| InteropError::SidecarSizeOverflow)?,
                        });
                } else {
                    let (start, byte_length, end, file_length) =
                        if let Some(bytes) = checkpoint_mapping {
                            let range = Self::checked_range(
                                descriptor,
                                offset,
                                length,
                                bytes.len(),
                                descriptor.source_codec,
                            )?;
                            (range.start, range.len(), range.end, bytes.len())
                        } else {
                            let start = usize::try_from(offset)
                                .map_err(|_| InteropError::SidecarSizeOverflow)?;
                            let byte_length = usize::try_from(length)
                                .map_err(|_| InteropError::SidecarSizeOverflow)?;
                            let end = start
                                .checked_add(byte_length)
                                .ok_or(InteropError::SidecarSizeOverflow)?;
                            let checkpoint_file = self.checkpoint_file.as_ref().ok_or_else(|| {
                                InteropError::PreGatherExecutionUnsupported {
                                    architecture: String::from("qwen35moe"),
                                    reason: String::from(
                                        "high expert staging needs a checkpoint file for bounded reads",
                                    ),
                                }
                            })?;
                            let file_length = checkpoint_file
                                .metadata()
                                .map_err(InteropError::SidecarIo)?
                                .len() as usize;
                            (start, byte_length, end, file_length)
                        };
                    if end > file_length {
                        return Err(InteropError::ExpertMappedRangeOutOfBounds {
                            start,
                            end,
                            mapping_len: file_length,
                        });
                    }
                    let cached = scratch.high_cache.iter().find(|entry| {
                        entry.layer == layer
                            && entry.expert == address.expert
                            && entry.projection == projection
                            && entry.offset == offset
                            && entry.bytes.len() == byte_length
                    });
                    if let Some(cached) = cached {
                        scratch.high_cache_hits += 1;
                        scratch.buffers[buffer_index].clear();
                        scratch.buffers[buffer_index].extend_from_slice(&cached.bytes);
                    } else {
                        scratch.high_cache_misses += 1;
                        scratch.buffers[buffer_index].resize(byte_length, 0);
                        if let Some(bytes) = checkpoint_mapping {
                            scratch.buffers[buffer_index].copy_from_slice(&bytes[start..end]);
                        } else {
                            let checkpoint_file = self.checkpoint_file.as_ref().ok_or_else(|| {
                                InteropError::PreGatherExecutionUnsupported {
                                    architecture: String::from("qwen35moe"),
                                    reason: String::from(
                                        "high expert staging needs a checkpoint file for bounded reads",
                                    ),
                                }
                            })?;
                            let mut reader = checkpoint_file
                                .try_clone()
                                .map_err(InteropError::SidecarIo)?;
                            reader
                                .seek(SeekFrom::Start(offset))
                                .map_err(InteropError::SidecarIo)?;
                            reader
                                .read_exact(&mut scratch.buffers[buffer_index])
                                .map_err(InteropError::SidecarIo)?;
                        }
                        if byte_length <= scratch.high_cache_limit {
                            while scratch.high_cache_bytes + byte_length > scratch.high_cache_limit
                            {
                                let Some(evicted) = scratch.high_cache.pop_front() else {
                                    break;
                                };
                                scratch.high_cache_bytes =
                                    scratch.high_cache_bytes.saturating_sub(evicted.bytes.len());
                            }
                            scratch.high_cache_bytes += byte_length;
                            scratch.high_cache.push_back(HighReadCacheEntry {
                                layer,
                                expert: address.expert,
                                projection,
                                offset,
                                bytes: scratch.buffers[buffer_index].clone(),
                            });
                        }
                    }
                    let expected = codec
                        .byte_len_for(descriptor.out_dim as usize * descriptor.in_dim as usize);
                    if byte_length != expected {
                        return Err(invalid_sidecar(format!(
                            "high expert range has {byte_length} bytes, expected {expected}"
                        )));
                    }
                }
                if !mapped_low {
                    scratch.ranges_read += 1;
                    scratch.bytes_read += scratch.buffers[buffer_index].len();
                    scratch.high_ranges_read += 1;
                    scratch.high_bytes_read += scratch.buffers[buffer_index].len();
                }
                scratch.descriptor_slots[address.expert][projection.index()] = Some(buffer_index);
                scratch.used_buffers += 1;
            }
        }
        Ok(())
    }

    /// Releases resident file pages after a step has finished consuming them.
    /// The mapping remains valid and faults pages back from the sidecar file
    /// on the next routed access; callers opt in because refault cost is
    /// workload-dependent.
    #[cfg(unix)]
    pub fn discard_resident_pages(&self) -> Result<(), InteropError> {
        unsafe {
            self.mapping
                .unchecked_advise(memmap2::UncheckedAdvice::DontNeed)
        }
        .map_err(InteropError::SidecarIo)
    }

    /// Advises the kernel that one expert's low-codec sidecar ranges are no
    /// longer needed after its layer gather. The mapping stays valid and a
    /// later route faults only this expert's ranges back in, rather than
    /// refaulting the entire sidecar mapping.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub(crate) fn discard_expert_low(&self, address: ExpertAddress) -> Result<(), InteropError> {
        for projection in ExpertProjection::ALL {
            let descriptor = self.descriptor(address, projection)?;
            let range = Self::checked_range(
                descriptor,
                descriptor.data_offset,
                descriptor.data_bytes,
                self.mapping.len(),
                descriptor.target_codec,
            )?;
            omega::discard_checkpoint_mmap_range_immediate(&self.mapping[range]).map_err(
                |error| InteropError::PreGatherExecutionUnsupported {
                    architecture: String::from("qwen35moe"),
                    reason: error.to_string(),
                },
            )?;
        }
        Ok(())
    }

    #[must_use]
    pub fn descriptor_count(&self) -> usize {
        self.sidecar.descriptors.len()
    }

    fn descriptor(
        &self,
        address: ExpertAddress,
        projection: ExpertProjection,
    ) -> Result<&ExpertSidecarDescriptor, InteropError> {
        let address_index = address
            .layer
            .checked_mul(self.expert_count)
            .and_then(|base| base.checked_add(address.expert))
            .ok_or(InteropError::SidecarSizeOverflow)?;
        let descriptor_index = self
            .descriptor_indices
            .get(address_index)
            .and_then(|indices| indices[projection.index()])
            .ok_or_else(|| {
                invalid_sidecar(format!(
                    "missing layer {} expert {} projection {}",
                    address.layer,
                    address.expert,
                    projection.name()
                ))
            })?;
        Ok(&self.sidecar.descriptors[descriptor_index])
    }

    fn checked_range(
        descriptor: &ExpertSidecarDescriptor,
        start: u64,
        length: u64,
        mapping_len: usize,
        codec: PackedOwnedKind,
    ) -> Result<core::ops::Range<usize>, InteropError> {
        let start = usize::try_from(start).map_err(|_| InteropError::SidecarSizeOverflow)?;
        let length = usize::try_from(length).map_err(|_| InteropError::SidecarSizeOverflow)?;
        let end = start
            .checked_add(length)
            .ok_or(InteropError::SidecarSizeOverflow)?;
        if end > mapping_len {
            return Err(InteropError::ExpertMappedRangeOutOfBounds {
                start,
                end,
                mapping_len,
            });
        }
        let expected_elements = (descriptor.out_dim as usize)
            .checked_mul(descriptor.in_dim as usize)
            .ok_or(InteropError::SidecarSizeOverflow)?;
        let expected_bytes = codec.byte_len_for(expected_elements);
        if length != expected_bytes {
            return Err(invalid_sidecar(format!(
                "layer {} expert {} projection {} has {length} bytes, expected {expected_bytes}",
                descriptor.layer, descriptor.expert, descriptor.projection
            )));
        }
        Ok(start..end)
    }

    /// Replaces every expert projection with its low-codec sidecar view.
    /// This is a load/attachment operation, not part of the token hot path.
    pub(crate) fn install_low_copies(
        &self,
        slab: &mut ExpertSlab<'_>,
        layer_count: usize,
        expert_count: usize,
    ) -> Result<(), InteropError> {
        for layer in 0..layer_count {
            for expert in 0..expert_count {
                let address = ExpertAddress { layer, expert };
                self.validate_address(slab, address, &[])?;
            }
        }
        for layer in 0..layer_count {
            for expert in 0..expert_count {
                self.restore_low(slab, ExpertAddress { layer, expert })?;
            }
        }
        Ok(())
    }

    fn validate_address(
        &self,
        slab: &ExpertSlab<'_>,
        address: ExpertAddress,
        checkpoint: &[u8],
    ) -> Result<(), InteropError> {
        for projection in ExpertProjection::ALL {
            let descriptor = self.descriptor(address, projection)?;
            slab.projection_site(address.layer, projection)?;
            Self::checked_range(
                descriptor,
                descriptor.data_offset,
                descriptor.data_bytes,
                self.mapping.len(),
                descriptor.target_codec,
            )?;
            if !checkpoint.is_empty() {
                Self::checked_range(
                    descriptor,
                    descriptor.source_offset,
                    descriptor.source_bytes,
                    checkpoint.len(),
                    descriptor.source_codec,
                )?;
            }
        }
        Ok(())
    }

    fn promote_high<'file>(
        &self,
        slab: &mut ExpertSlab<'file>,
        checkpoint: &'file [u8],
        address: ExpertAddress,
    ) -> Result<(), InteropError> {
        self.validate_address(slab, address, checkpoint)?;
        for projection in ExpertProjection::ALL {
            let descriptor = self.descriptor(address, projection)?;
            let range = Self::checked_range(
                descriptor,
                descriptor.source_offset,
                descriptor.source_bytes,
                checkpoint.len(),
                descriptor.source_codec,
            )?;
            let site = slab.projection_site(address.layer, projection)?;
            slab.page_expert_borrowed(
                site,
                address.expert,
                descriptor.source_codec,
                &checkpoint[range],
                descriptor.out_dim,
                descriptor.in_dim,
            )?;
        }
        Ok(())
    }

    fn restore_low(
        &self,
        slab: &mut ExpertSlab<'_>,
        address: ExpertAddress,
    ) -> Result<(), InteropError> {
        self.validate_address(slab, address, &[])?;
        for projection in ExpertProjection::ALL {
            let descriptor = self.descriptor(address, projection)?;
            let range = Self::checked_range(
                descriptor,
                descriptor.data_offset,
                descriptor.data_bytes,
                self.mapping.len(),
                descriptor.target_codec,
            )?;
            let site = slab.projection_site(address.layer, projection)?;
            slab.page_expert_mapped(
                site,
                address.expert,
                descriptor.target_codec,
                Arc::clone(&self.mapping),
                range,
                descriptor.out_dim,
                descriptor.in_dim,
            )?;
        }
        Ok(())
    }

    pub(crate) fn apply_action<'file>(
        &self,
        slab: &mut ExpertSlab<'file>,
        checkpoint: &'file [u8],
        action: ResidencyAction,
    ) -> Result<(), InteropError> {
        if std::env::var_os("PROXIMA_DEBUG_EXPERT_UPLOADS").is_some() {
            let address = match action {
                ResidencyAction::Page(address) | ResidencyAction::Evict(address) => address,
            };
            let codecs = ExpertProjection::ALL.map(|projection| {
                self.descriptor(address, projection)
                    .map(|descriptor| (descriptor.source_codec, descriptor.target_codec))
            });
            eprintln!(
                "qwen35 sidecar action={action:?} layer={} expert={} codecs={codecs:?}",
                address.layer, address.expert,
            );
        }
        match action {
            ResidencyAction::Page(address) => self.promote_high(slab, checkpoint, address),
            ResidencyAction::Evict(address) => {
                self.restore_low(slab, address)?;
                #[cfg(all(feature = "metal", target_os = "macos"))]
                if std::env::var_os("PROXIMA_EXPERT_CHECKPOINT_DISCARD").is_some() {
                    for projection in ExpertProjection::ALL {
                        let descriptor = self.descriptor(address, projection)?;
                        let range = Self::checked_range(
                            descriptor,
                            descriptor.source_offset,
                            descriptor.source_bytes,
                            checkpoint.len(),
                            descriptor.source_codec,
                        )?;
                        omega::discard_checkpoint_mmap_range(&checkpoint[range.clone()]).map_err(
                            |error| InteropError::PreGatherExecutionUnsupported {
                                architecture: String::from("qwen35moe"),
                                reason: error.to_string(),
                            },
                        )?;
                        if std::env::var_os("PROXIMA_EXPERT_CHECKPOINT_RESIDENCY").is_some() {
                            let resident_pages =
                                omega::checkpoint_mmap_resident_pages(&checkpoint[range]).map_err(
                                    |error| InteropError::PreGatherExecutionUnsupported {
                                        architecture: String::from("qwen35moe"),
                                        reason: error.to_string(),
                                    },
                                )?;
                            eprintln!(
                                "expert checkpoint residency layer={} expert={} projection={} resident_pages={}",
                                address.layer,
                                address.expert,
                                projection.name(),
                                resident_pages,
                            );
                        }
                    }
                }
                Ok(())
            }
        }
    }

    /// Conservative bytes charged for one high expert across all projections.
    #[must_use]
    pub const fn high_bytes_per_expert(&self) -> u64 {
        self.high_bytes_per_expert
    }
}

fn projection_from_name(name: &str) -> Result<ExpertProjection, InteropError> {
    match name {
        "ffn_gate" => Ok(ExpertProjection::Gate),
        "ffn_up" => Ok(ExpertProjection::Up),
        "ffn_down" => Ok(ExpertProjection::Down),
        other => Err(invalid_sidecar(format!(
            "projection {other:?} is not one of ffn_gate, ffn_up, ffn_down"
        ))),
    }
}

impl ExpertSidecar {
    /// Parses and validates the descriptor directory in a mapped sidecar.
    /// Payload bytes remain in `mapping`; only descriptor metadata is owned.
    pub fn from_bytes(mapping: &[u8]) -> Result<Self, InteropError> {
        let mut cursor = 0usize;
        let magic = take::<8>(mapping, &mut cursor)?;
        if magic != *MAGIC {
            return Err(invalid_sidecar("magic does not match PXEXSC01"));
        }
        let version = u32::from_le_bytes(take::<4>(mapping, &mut cursor)?);
        if version != VERSION {
            return Err(invalid_sidecar(format!(
                "version {version} is unsupported; expected {VERSION}"
            )));
        }
        let descriptor_count_u64 = u64::from_le_bytes(take::<8>(mapping, &mut cursor)?);
        let descriptor_count =
            usize::try_from(descriptor_count_u64).map_err(|_| InteropError::SidecarSizeOverflow)?;
        let maximum_descriptor_count = mapping
            .len()
            .saturating_sub(cursor)
            .checked_div(DESCRIPTOR_FIXED_BYTES as usize)
            .unwrap_or(0);
        if descriptor_count > maximum_descriptor_count {
            return Err(invalid_sidecar(format!(
                "descriptor count {descriptor_count} exceeds the mapping capacity {maximum_descriptor_count}"
            )));
        }
        let mut descriptors = Vec::with_capacity(descriptor_count);

        for descriptor_index in 0..descriptor_count {
            let descriptor_offset = cursor;
            let layer = u32::from_le_bytes(take::<4>(mapping, &mut cursor)?);
            let expert = u32::from_le_bytes(take::<4>(mapping, &mut cursor)?);
            let out_dim = u32::from_le_bytes(take::<4>(mapping, &mut cursor)?);
            let in_dim = u32::from_le_bytes(take::<4>(mapping, &mut cursor)?);
            let source_codec = codec_from_tag(take::<1>(mapping, &mut cursor)?[0])?;
            let target_codec = codec_from_tag(take::<1>(mapping, &mut cursor)?[0])?;
            let projection_bytes =
                usize::from(u16::from_le_bytes(take::<2>(mapping, &mut cursor)?));
            let projection =
                std::str::from_utf8(take_slice(mapping, &mut cursor, projection_bytes)?)
                    .map_err(|error| {
                        invalid_sidecar(format!(
                            "descriptor {descriptor_index} projection is not utf-8: {error}"
                        ))
                    })?
                    .to_owned();
            let source_offset = u64::from_le_bytes(take::<8>(mapping, &mut cursor)?);
            let source_bytes = u64::from_le_bytes(take::<8>(mapping, &mut cursor)?);
            let recorded_descriptor_offset = u64::from_le_bytes(take::<8>(mapping, &mut cursor)?);
            if recorded_descriptor_offset != descriptor_offset as u64 {
                return Err(invalid_sidecar(format!(
                    "descriptor {descriptor_index} starts at {descriptor_offset}, recorded {recorded_descriptor_offset}"
                )));
            }
            let element_count = usize::try_from(out_dim)
                .ok()
                .and_then(|rows| {
                    usize::try_from(in_dim)
                        .ok()
                        .and_then(|columns| rows.checked_mul(columns))
                })
                .ok_or(InteropError::SidecarSizeOverflow)?;
            let expected_source_bytes = source_codec.byte_len_for(element_count);
            let expected_data_bytes = target_codec.byte_len_for(element_count);
            if source_bytes != expected_source_bytes as u64 {
                return Err(invalid_sidecar(format!(
                    "descriptor {descriptor_index} source bytes {source_bytes} do not match {expected_source_bytes} bytes for {out_dim}x{in_dim}"
                )));
            }
            let data_offset = cursor as u64;
            let data_bytes = u64::try_from(expected_data_bytes)
                .map_err(|_| InteropError::SidecarSizeOverflow)?;
            take_slice(mapping, &mut cursor, expected_data_bytes)?;
            if descriptors
                .iter()
                .any(|existing: &ExpertSidecarDescriptor| {
                    existing.layer == layer
                        && existing.expert == expert
                        && existing.projection == projection
                })
            {
                return Err(invalid_sidecar(format!(
                    "descriptor {descriptor_index} duplicates layer {layer}, expert {expert}, projection {projection:?}"
                )));
            }
            descriptors.push(ExpertSidecarDescriptor {
                layer,
                projection,
                expert,
                out_dim,
                in_dim,
                source_codec,
                target_codec,
                source_offset,
                source_bytes,
                data_offset,
                data_bytes,
            });
        }
        if cursor != mapping.len() {
            return Err(invalid_sidecar(format!(
                "{} trailing bytes follow the descriptor payloads",
                mapping.len() - cursor
            )));
        }
        Ok(Self {
            descriptors,
            data_offset: 20,
            total_bytes: cursor as u64,
        })
    }

    /// Looks up one sidecar expert and returns a zero-copy page view into the
    /// caller-owned mapping. The mapping must outlive the returned page.
    pub fn page<'bytes>(
        &'bytes self,
        mapping: &'bytes [u8],
        layer: u32,
        expert: u32,
        projection: &str,
    ) -> Result<crate::residency::ExpertPage<'bytes>, InteropError> {
        let descriptor = self
            .descriptors
            .iter()
            .find(|candidate| {
                candidate.layer == layer
                    && candidate.expert == expert
                    && candidate.projection == projection
            })
            .ok_or(InteropError::ExpertSlabIndexOutOfRange {
                layer: layer as usize,
                expert: expert as usize,
            })?;
        let start = usize::try_from(descriptor.data_offset)
            .map_err(|_| InteropError::SidecarSizeOverflow)?;
        let length = usize::try_from(descriptor.data_bytes)
            .map_err(|_| InteropError::SidecarSizeOverflow)?;
        let end = start
            .checked_add(length)
            .ok_or(InteropError::SidecarSizeOverflow)?;
        let bytes = mapping
            .get(start..end)
            .ok_or(InteropError::ExpertMappedRangeOutOfBounds {
                start,
                end,
                mapping_len: mapping.len(),
            })?;
        Ok(crate::residency::ExpertPage {
            codec: descriptor.target_codec,
            bytes,
            out_dim: descriptor.out_dim,
            in_dim: descriptor.in_dim,
        })
    }
}

/// Writes low-codec copies of the requested packed GGUF expert stacks.
///
/// `scratch` must hold exactly one expert's decoded elements and `output`
/// must hold exactly one target expert. Both are reused for every expert.
/// `destination` may be a file, a bounded writer over an mmap range, or an
/// LSM segment writer. The destination is truncated nowhere; writing starts
/// at its current position and the function seeks only within its own output.
///
/// The source stack is required to be a native packed codec and evenly
/// divisible by `expert_count`; dimensions are retained in every descriptor
/// so a consumer can build an `ExpertSource` without rediscovering shape.
pub fn write_expert_sidecar<W: Write + Seek>(
    parsed: &ParsedGguf,
    file_bytes: &[u8],
    stacks: &[ExpertStackSpec<'_>],
    target_codec: PackedOwnedKind,
    scratch: &mut [f32],
    output: &mut [u8],
    destination: &mut W,
) -> Result<ExpertSidecar, InteropError> {
    let _ = target_codec;
    let descriptor_count = stacks.iter().try_fold(0u64, |count, spec| {
        count
            .checked_add(u64::from(spec.expert_count))
            .ok_or(InteropError::SidecarSizeOverflow)
    })?;
    // magic (8) + version (4) + descriptor count (8). Records contain a
    // descriptor followed immediately by its payload, so a bounded writer
    // needs no second model-sized staging area.
    let header_bytes = 20u64;
    let _descriptor_bytes = stacks.iter().try_fold(0u64, |total, spec| {
        let name_len = spec.projection.len();
        if name_len > MAX_PROJECTION_BYTES {
            return Err(InteropError::SidecarProjectionTooLong {
                found: name_len,
                max: MAX_PROJECTION_BYTES,
            });
        }
        let one = DESCRIPTOR_FIXED_BYTES
            .checked_add(name_len as u64)
            .ok_or(InteropError::SidecarSizeOverflow)?;
        total
            .checked_add(
                one.checked_mul(u64::from(spec.expert_count))
                    .ok_or(InteropError::SidecarSizeOverflow)?,
            )
            .ok_or(InteropError::SidecarSizeOverflow)
    })?;
    // Records are interleaved (descriptor then payload), so the first record
    // begins immediately after the fixed header. `descriptor_bytes` above
    // remains the checked directory-size calculation used for overflow
    // validation.
    let data_offset = header_bytes;

    destination.write_all(MAGIC)?;
    destination.write_all(&VERSION.to_le_bytes())?;
    destination.write_all(&descriptor_count.to_le_bytes())?;
    let mut descriptors = Vec::with_capacity(descriptor_count as usize);
    let mut data_cursor = data_offset;

    for spec in stacks {
        let tensor = parsed
            .tensors
            .iter()
            .find(|tensor| tensor.name == spec.tensor_name)
            .ok_or_else(|| InteropError::UnknownTensor {
                name: spec.tensor_name.to_owned(),
            })?;
        let source_codec = packed_kind(tensor.ggml_type, spec.tensor_name)?;
        let range = parsed.tensor_data_range(tensor, file_bytes.len() as u64)?;
        let source = &file_bytes[range.start as usize..range.end as usize];
        if spec.expert_count == 0 || !source.len().is_multiple_of(spec.expert_count as usize) {
            return Err(InteropError::ExpertSlabIndexOutOfRange {
                layer: spec.layer as usize,
                expert: spec.expert_count as usize,
            });
        }
        let per_expert_source = source.len() / spec.expert_count as usize;
        let expected_elements = spec.out_dim as usize * spec.in_dim as usize;
        if scratch.len() != expected_elements {
            return Err(InteropError::Quant(
                proxima_gguf::quant::QuantError::OutputSizeMismatch {
                    found: scratch.len(),
                    expected: expected_elements,
                },
            ));
        }
        let expected_output = spec.target_codec.byte_len_for(expected_elements);
        if output.len() < expected_output {
            return Err(InteropError::Quant(
                proxima_gguf::quant::QuantError::OutputSizeMismatch {
                    found: output.len(),
                    expected: expected_output,
                },
            ));
        }
        let output = &mut output[..expected_output];
        let projection_bytes = spec.projection.as_bytes();
        for expert in 0..spec.expert_count {
            let expert_source = &source
                [expert as usize * per_expert_source..(expert as usize + 1) * per_expert_source];
            if source_codec == spec.target_codec {
                output.copy_from_slice(expert_source);
            } else {
                recode_expert_into(
                    source_codec,
                    expert_source,
                    spec.target_codec,
                    scratch,
                    output,
                )?;
            }
            let source_offset = range.start + expert as u64 * per_expert_source as u64;
            let descriptor_bytes = DESCRIPTOR_FIXED_BYTES + projection_bytes.len() as u64;
            let payload_offset = data_cursor
                .checked_add(descriptor_bytes)
                .ok_or(InteropError::SidecarSizeOverflow)?;
            let descriptor = ExpertSidecarDescriptor {
                layer: spec.layer,
                projection: spec.projection.to_owned(),
                expert,
                out_dim: spec.out_dim,
                in_dim: spec.in_dim,
                source_codec,
                target_codec: spec.target_codec,
                source_offset,
                source_bytes: per_expert_source as u64,
                data_offset: payload_offset,
                data_bytes: output.len() as u64,
            };
            destination.write_all(&spec.layer.to_le_bytes())?;
            destination.write_all(&expert.to_le_bytes())?;
            destination.write_all(&spec.out_dim.to_le_bytes())?;
            destination.write_all(&spec.in_dim.to_le_bytes())?;
            destination.write_all(&[codec_tag(source_codec), codec_tag(spec.target_codec)])?;
            destination.write_all(&(projection_bytes.len() as u16).to_le_bytes())?;
            destination.write_all(projection_bytes)?;
            destination.write_all(&source_offset.to_le_bytes())?;
            destination.write_all(&(per_expert_source as u64).to_le_bytes())?;
            destination.write_all(&data_cursor.to_le_bytes())?;
            destination.write_all(output)?;
            data_cursor = payload_offset
                .checked_add(output.len() as u64)
                .ok_or(InteropError::SidecarSizeOverflow)?;
            descriptors.push(descriptor);
        }
    }
    Ok(ExpertSidecar {
        descriptors,
        data_offset,
        total_bytes: data_cursor,
    })
}

fn packed_kind(ggml_type: GgmlType, tensor: &str) -> Result<PackedOwnedKind, InteropError> {
    match ggml_type {
        GgmlType::Q2_K => Ok(PackedOwnedKind::Q2K),
        GgmlType::Q3_K => Ok(PackedOwnedKind::Q3K),
        GgmlType::Q4_K => Ok(PackedOwnedKind::Q4K),
        GgmlType::Q5_K => Ok(PackedOwnedKind::Q5K),
        GgmlType::Q6_K => Ok(PackedOwnedKind::Q6K),
        GgmlType::Q8_0 => Ok(PackedOwnedKind::Q8_0),
        GgmlType::Q4_0 => Ok(PackedOwnedKind::Q4_0),
        GgmlType::F16 => Ok(PackedOwnedKind::Float16),
        GgmlType::Bf16 => Ok(PackedOwnedKind::BFloat16),
        other => Err(InteropError::UnrepresentableGgmlType {
            tensor: tensor.to_owned(),
            ggml_type: other,
        }),
    }
}

fn codec_tag(codec: PackedOwnedKind) -> u8 {
    match codec {
        PackedOwnedKind::Q2K => 0,
        PackedOwnedKind::Q3K => 1,
        PackedOwnedKind::Q4K => 2,
        PackedOwnedKind::Q5K => 3,
        PackedOwnedKind::Q6K => 4,
        PackedOwnedKind::Q8_0 => 5,
        PackedOwnedKind::Q4_0 => 6,
        PackedOwnedKind::Float16 => 7,
        PackedOwnedKind::BFloat16 => 8,
    }
}

fn codec_from_tag(tag: u8) -> Result<PackedOwnedKind, InteropError> {
    match tag {
        0 => Ok(PackedOwnedKind::Q2K),
        1 => Ok(PackedOwnedKind::Q3K),
        2 => Ok(PackedOwnedKind::Q4K),
        3 => Ok(PackedOwnedKind::Q5K),
        4 => Ok(PackedOwnedKind::Q6K),
        5 => Ok(PackedOwnedKind::Q8_0),
        6 => Ok(PackedOwnedKind::Q4_0),
        7 => Ok(PackedOwnedKind::Float16),
        8 => Ok(PackedOwnedKind::BFloat16),
        _ => Err(invalid_sidecar(format!("unknown codec tag {tag}"))),
    }
}

fn take<const LENGTH: usize>(
    bytes: &[u8],
    cursor: &mut usize,
) -> Result<[u8; LENGTH], InteropError> {
    take_slice(bytes, cursor, LENGTH)?
        .try_into()
        .map_err(|_| invalid_sidecar("fixed-width field has the wrong length"))
}

fn take_slice<'bytes>(
    bytes: &'bytes [u8],
    cursor: &mut usize,
    length: usize,
) -> Result<&'bytes [u8], InteropError> {
    let end = cursor
        .checked_add(length)
        .ok_or(InteropError::SidecarSizeOverflow)?;
    let slice = bytes.get(*cursor..end).ok_or_else(|| {
        invalid_sidecar(format!(
            "field at offset {} needs {length} bytes, mapping has {}",
            *cursor,
            bytes.len()
        ))
    })?;
    *cursor = end;
    Ok(slice)
}

fn invalid_sidecar(reason: impl Into<String>) -> InteropError {
    InteropError::InvalidExpertSidecar(reason.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrayvec::ArrayVec;
    use memmap2::MmapOptions;
    use proxima_gguf::{GgmlType, TensorInfo};
    use proxima_tensor::NodeId;
    use std::io::Cursor;
    use tempfile::tempfile;

    #[test]
    fn writes_each_expert_sequentially_with_descriptors_and_reused_buffers() {
        let mut source = vec![0u8; 288];
        let values: [f32; 256] = core::array::from_fn(|index| index as f32 * 0.01 - 1.0);
        proxima_gguf::quant::q4_k::quantize(&values, &mut source[..144])
            .expect("first Q4_K expert encodes");
        let second_values: [f32; 256] = core::array::from_fn(|index| 1.0 - index as f32 * 0.01);
        proxima_gguf::quant::q4_k::quantize(&second_values, &mut source[144..])
            .expect("second Q4_K expert encodes");
        let tensor = TensorInfo {
            name: "blk.0.ffn_gate_exps.weight".into(),
            dims: [256u64, 2u64].into_iter().collect::<ArrayVec<_, 4>>(),
            ggml_type: GgmlType::Q4_K,
            offset: 0,
        };
        let parsed = ParsedGguf {
            version: 3,
            tensor_count: 1,
            kv_count: 0,
            metadata: Vec::new(),
            tensors: vec![tensor],
            data_offset: 0,
            alignment: 32,
        };
        let specification = [ExpertStackSpec {
            layer: 0,
            projection: "ffn_gate",
            tensor_name: "blk.0.ffn_gate_exps.weight",
            expert_count: 2,
            out_dim: 1,
            in_dim: 256,
            target_codec: PackedOwnedKind::Q2K,
        }];
        let mut scratch = [0.0f32; 256];
        let mut output = [0u8; 84];
        let mut sidecar = Cursor::new(Vec::new());
        let result = write_expert_sidecar(
            &parsed,
            &source,
            &specification,
            PackedOwnedKind::Q2K,
            &mut scratch,
            &mut output,
            &mut sidecar,
        )
        .expect("the two experts write to the sidecar");
        assert_eq!(result.descriptors.len(), 2);
        assert_eq!(result.descriptors[0].source_offset, 0);
        assert_eq!(result.descriptors[1].source_offset, 144);
        assert_eq!(result.descriptors[0].data_bytes, 84);
        assert_eq!(
            result.descriptors[1].data_offset,
            result.descriptors[0].data_offset + 84 + 52
        );
        assert_eq!(result.total_bytes as usize, sidecar.get_ref().len());
        assert_eq!(&sidecar.get_ref()[0..8], MAGIC);
        assert_eq!(
            u64::from_le_bytes(sidecar.get_ref()[12..20].try_into().expect("count bytes")),
            2
        );
        assert!(
            sidecar.get_ref()[result.descriptors[0].data_offset as usize
                ..result.descriptors[0].data_offset as usize + 84]
                .iter()
                .any(|byte| *byte != 0)
        );
        let page = result
            .page(sidecar.get_ref(), 0, 1, "ffn_gate")
            .expect("the sidecar exposes a zero-copy page descriptor");
        assert_eq!(page.codec, PackedOwnedKind::Q2K);
        assert_eq!(page.bytes.len(), 84);

        let loaded = ExpertSidecar::from_bytes(sidecar.get_ref())
            .expect("the emitted sidecar descriptor directory validates");
        assert_eq!(loaded, result);
        let loaded_page = loaded
            .page(sidecar.get_ref(), 0, 1, "ffn_gate")
            .expect("the loaded descriptor indexes the second expert");
        assert_eq!(loaded_page.bytes, page.bytes);
    }

    #[test]
    fn low_reads_stop_for_high_promotion_and_resume_after_eviction() {
        let mut source = vec![0_u8; 3 * 144];
        for projection in 0..3 {
            let values: [f32; 256] =
                core::array::from_fn(|index| index as f32 * 0.01 + projection as f32);
            let range = projection * 144..(projection + 1) * 144;
            proxima_gguf::quant::q4_k::quantize(&values, &mut source[range])
                .expect("one Q4_K expert encodes");
        }
        let tensors = [
            ("blk.0.ffn_gate_exps.weight", 0_u64),
            ("blk.0.ffn_up_exps.weight", 144_u64),
            ("blk.0.ffn_down_exps.weight", 288_u64),
        ]
        .into_iter()
        .map(|(name, offset)| TensorInfo {
            name: name.into(),
            dims: [256_u64, 1_u64].into_iter().collect::<ArrayVec<_, 4>>(),
            ggml_type: GgmlType::Q4_K,
            offset,
        })
        .collect();
        let parsed = ParsedGguf {
            version: 3,
            tensor_count: 3,
            kv_count: 0,
            metadata: Vec::new(),
            tensors,
            data_offset: 0,
            alignment: 32,
        };
        let specifications = [
            ExpertStackSpec {
                layer: 0,
                projection: "ffn_gate",
                tensor_name: "blk.0.ffn_gate_exps.weight",
                expert_count: 1,
                out_dim: 1,
                in_dim: 256,
                target_codec: PackedOwnedKind::Q2K,
            },
            ExpertStackSpec {
                layer: 0,
                projection: "ffn_up",
                tensor_name: "blk.0.ffn_up_exps.weight",
                expert_count: 1,
                out_dim: 1,
                in_dim: 256,
                target_codec: PackedOwnedKind::Q2K,
            },
            ExpertStackSpec {
                layer: 0,
                projection: "ffn_down",
                tensor_name: "blk.0.ffn_down_exps.weight",
                expert_count: 1,
                out_dim: 1,
                in_dim: 256,
                target_codec: PackedOwnedKind::Q2K,
            },
        ];
        let mut decode_scratch = [0.0_f32; 256];
        let mut encoded_expert = [0_u8; 84];
        let mut sidecar_bytes = Cursor::new(Vec::new());
        write_expert_sidecar(
            &parsed,
            &source,
            &specifications,
            PackedOwnedKind::Q2K,
            &mut decode_scratch,
            &mut encoded_expert,
            &mut sidecar_bytes,
        )
        .expect("all three projections write to the sidecar");
        let mut sidecar_file = tempfile().expect("creates sidecar file");
        sidecar_file
            .write_all(sidecar_bytes.get_ref())
            .expect("writes sidecar bytes");
        sidecar_file.flush().expect("flushes sidecar bytes");
        let mut checkpoint_file = tempfile().expect("creates checkpoint file");
        checkpoint_file
            .write_all(&source)
            .expect("writes checkpoint bytes");
        checkpoint_file.flush().expect("flushes checkpoint bytes");
        let sidecar = MappedExpertSidecar::from_file(sidecar_file)
            .expect("maps the generated sidecar")
            .with_checkpoint_file(checkpoint_file);

        let mut slab = ExpertSlab::new();
        slab.bind_layer_stack(
            0,
            NodeId(1),
            PackedOwnedKind::Q4K,
            &source[0..144],
            1,
            1,
            256,
        )
        .expect("binds gate stack");
        slab.bind_layer_stack(
            1,
            NodeId(2),
            PackedOwnedKind::Q4K,
            &source[144..288],
            1,
            1,
            256,
        )
        .expect("binds up stack");
        slab.bind_layer_stack(
            2,
            NodeId(3),
            PackedOwnedKind::Q4K,
            &source[288..432],
            1,
            1,
            256,
        )
        .expect("binds down stack");
        slab.register_model_layer_site(0, ExpertProjection::Gate, 0);
        slab.register_model_layer_site(0, ExpertProjection::Up, 1);
        slab.register_model_layer_site(0, ExpertProjection::Down, 2);
        sidecar
            .install_low_copies(&mut slab, 1, 1)
            .expect("installs low copies");

        let mut scratch = ExpertSidecarReadScratch::default();
        sidecar
            .read_selected(0, &[0], &slab, &mut scratch)
            .expect("reads all three low projections");
        assert_eq!(scratch.ranges_read, 3);
        assert_eq!(scratch.bytes_read, 3 * 84);
        assert_eq!(scratch.low_ranges_read, 3);
        assert_eq!(scratch.high_ranges_read, 0);

        sidecar
            .apply_action(
                &mut slab,
                &source,
                ResidencyAction::Page(ExpertAddress {
                    layer: 0,
                    expert: 0,
                }),
            )
            .expect("promotes the selected expert");
        sidecar
            .read_selected(0, &[0], &slab, &mut scratch)
            .expect("reads high ranges for the promoted expert");
        assert_eq!(scratch.ranges_read, 3);
        assert_eq!(scratch.bytes_read, 3 * 144);
        assert_eq!(scratch.low_ranges_read, 0);
        assert_eq!(scratch.high_ranges_read, 3);

        sidecar
            .apply_action(
                &mut slab,
                &source,
                ResidencyAction::Evict(ExpertAddress {
                    layer: 0,
                    expert: 0,
                }),
            )
            .expect("restores the low copy");
        sidecar
            .read_selected(0, &[0], &slab, &mut scratch)
            .expect("reads low projections after eviction");
        assert_eq!(scratch.ranges_read, 3);
        assert_eq!(scratch.bytes_read, 3 * 84);
        assert_eq!(scratch.low_ranges_read, 3);
        assert_eq!(scratch.high_ranges_read, 0);
    }

    #[test]
    fn all_low_sources_alias_every_sidecar_expert_and_reject_promotion() {
        let expert_count = 2_usize;
        let projection_bytes = expert_count * 144;
        let mut checkpoint = vec![0_u8; projection_bytes * 3];
        for projection in 0..3 {
            for expert in 0..expert_count {
                let values: [f32; 256] = core::array::from_fn(|index| {
                    index as f32 * 0.01 + projection as f32 + expert as f32 * 0.25
                });
                let start = projection * projection_bytes + expert * 144;
                proxima_gguf::quant::q4_k::quantize(&values, &mut checkpoint[start..start + 144])
                    .expect("the synthetic expert encodes");
            }
        }
        let tensors = ["ffn_gate", "ffn_up", "ffn_down"]
            .into_iter()
            .enumerate()
            .map(|(index, projection)| TensorInfo {
                name: format!("blk.0.{projection}_exps.weight"),
                dims: [256_u64, expert_count as u64]
                    .into_iter()
                    .collect::<ArrayVec<_, 4>>(),
                ggml_type: GgmlType::Q4_K,
                offset: (index * projection_bytes) as u64,
            })
            .collect::<Vec<_>>();
        let parsed = ParsedGguf {
            version: 3,
            tensor_count: 3,
            kv_count: 0,
            metadata: Vec::new(),
            tensors,
            data_offset: 0,
            alignment: 32,
        };
        let specifications = ["ffn_gate", "ffn_up", "ffn_down"].map(|projection| ExpertStackSpec {
            layer: 0,
            projection,
            tensor_name: match projection {
                "ffn_gate" => "blk.0.ffn_gate_exps.weight",
                "ffn_up" => "blk.0.ffn_up_exps.weight",
                _ => "blk.0.ffn_down_exps.weight",
            },
            expert_count: expert_count as u32,
            out_dim: 1,
            in_dim: 256,
            target_codec: PackedOwnedKind::Q2K,
        });
        let mut decode_scratch = [0.0_f32; 256];
        let mut encoded_expert = [0_u8; 84];
        let mut sidecar_bytes = Cursor::new(Vec::new());
        write_expert_sidecar(
            &parsed,
            &checkpoint,
            &specifications,
            PackedOwnedKind::Q2K,
            &mut decode_scratch,
            &mut encoded_expert,
            &mut sidecar_bytes,
        )
        .expect("every low expert writes to the sidecar");
        let mut sidecar_file = tempfile().expect("creates sidecar file");
        sidecar_file
            .write_all(sidecar_bytes.get_ref())
            .expect("writes sidecar bytes");
        sidecar_file.flush().expect("flushes sidecar bytes");
        let sidecar = MappedExpertSidecar::from_file(sidecar_file)
            .expect("the generated sidecar maps and validates");

        let mut slab = ExpertSlab::new();
        for (site, projection) in ExpertProjection::ALL.into_iter().enumerate() {
            let start = site * projection_bytes;
            slab.bind_layer_stack(
                site,
                NodeId(site as u32 + 1),
                PackedOwnedKind::Q4K,
                &checkpoint[start..start + projection_bytes],
                expert_count,
                1,
                256,
            )
            .expect("the checkpoint projection binds");
            slab.register_model_layer_site(0, projection, site);
        }
        sidecar
            .install_low_copies(&mut slab, 1, expert_count)
            .expect("attachment installs every low projection");

        let mut source_scratch = Vec::new();
        let sources = slab
            .all_low_sources_for_step(&sidecar, &mut source_scratch)
            .expect("the all-low snapshot aliases the sidecar");
        assert_eq!(sources.len(), 3);
        let mapping_start = sidecar.mapping_bytes().as_ptr() as usize;
        let mapping_end = mapping_start + sidecar.mapping_bytes().len();
        for source in sources.values() {
            assert_eq!(source.entries().len(), expert_count);
            assert_eq!(source.selected_expert_ids(), None);
            let arena = source.packed_arena().expect("the source carries one arena");
            let arena_start = arena.bytes().as_ptr() as usize;
            assert!(arena_start >= mapping_start);
            assert!(arena_start + arena.bytes().len() <= mapping_end);
            for (entry, span) in source.entries().iter().zip(arena.spans()) {
                let span = span.expect("every all-low expert has a descriptor span");
                let start = span.offset as usize;
                let end = start + span.length as usize;
                assert_eq!(
                    &arena.bytes()[start..end],
                    entry.block.packed_bytes().expect("low experts are packed")
                );
            }
        }
        drop(sources);

        sidecar
            .apply_action(
                &mut slab,
                &checkpoint,
                ResidencyAction::Page(ExpertAddress {
                    layer: 0,
                    expert: 1,
                }),
            )
            .expect("the second expert promotes between snapshots");
        let mut promoted_scratch = Vec::new();
        let error = slab
            .all_low_sources_for_step(&sidecar, &mut promoted_scratch)
            .expect_err("the all-low arm rejects a promoted expert");
        assert!(matches!(
            error,
            InteropError::ExpertAllLowSourceRequired {
                layer: 0,
                expert: 1,
                projection: "ffn_gate",
            }
        ));
    }

    #[test]
    fn rejects_truncated_and_corrupt_descriptor_data() {
        let truncated = &MAGIC[..];
        assert!(matches!(
            ExpertSidecar::from_bytes(truncated),
            Err(InteropError::InvalidExpertSidecar(_))
        ));

        let mut unknown_codec = Vec::new();
        unknown_codec.extend_from_slice(MAGIC);
        unknown_codec.extend_from_slice(&VERSION.to_le_bytes());
        unknown_codec.extend_from_slice(&1u64.to_le_bytes());
        unknown_codec.extend_from_slice(&0u32.to_le_bytes());
        unknown_codec.extend_from_slice(&0u32.to_le_bytes());
        unknown_codec.extend_from_slice(&1u32.to_le_bytes());
        unknown_codec.extend_from_slice(&256u32.to_le_bytes());
        unknown_codec.extend_from_slice(&[255, codec_tag(PackedOwnedKind::Q2K)]);
        unknown_codec.extend_from_slice(&0u16.to_le_bytes());
        unknown_codec.extend_from_slice(&0u64.to_le_bytes());
        unknown_codec.extend_from_slice(&144u64.to_le_bytes());
        unknown_codec.extend_from_slice(&20u64.to_le_bytes());
        unknown_codec.extend_from_slice(&[0u8; 84]);
        assert!(matches!(
            ExpertSidecar::from_bytes(&unknown_codec),
            Err(InteropError::InvalidExpertSidecar(_))
        ));
    }

    #[test]
    fn mapped_sidecar_switches_all_three_projection_sites_without_copying() {
        let values: [f32; 256] = core::array::from_fn(|index| index as f32 * 0.01 - 1.0);
        let mut checkpoint = vec![0_u8; 144 * 3];
        for projection in checkpoint.chunks_exact_mut(144) {
            proxima_gguf::quant::q4_k::quantize(&values, projection)
                .expect("the synthetic checkpoint projection encodes");
        }
        let tensors = ["ffn_gate", "ffn_up", "ffn_down"]
            .into_iter()
            .enumerate()
            .map(|(index, projection)| TensorInfo {
                name: format!("blk.0.{projection}_exps.weight"),
                dims: [256_u64, 1_u64].into_iter().collect::<ArrayVec<_, 4>>(),
                ggml_type: GgmlType::Q4_K,
                offset: (index * 144) as u64,
            })
            .collect::<Vec<_>>();
        let parsed = ParsedGguf {
            version: 3,
            tensor_count: 3,
            kv_count: 0,
            metadata: Vec::new(),
            tensors,
            data_offset: 0,
            alignment: 32,
        };
        let specifications = [
            ExpertStackSpec {
                layer: 0,
                projection: "ffn_gate",
                tensor_name: "blk.0.ffn_gate_exps.weight",
                expert_count: 1,
                out_dim: 1,
                in_dim: 256,
                target_codec: PackedOwnedKind::Q2K,
            },
            ExpertStackSpec {
                layer: 0,
                projection: "ffn_up",
                tensor_name: "blk.0.ffn_up_exps.weight",
                expert_count: 1,
                out_dim: 1,
                in_dim: 256,
                target_codec: PackedOwnedKind::Q2K,
            },
            ExpertStackSpec {
                layer: 0,
                projection: "ffn_down",
                tensor_name: "blk.0.ffn_down_exps.weight",
                expert_count: 1,
                out_dim: 1,
                in_dim: 256,
                target_codec: PackedOwnedKind::Q2K,
            },
        ];
        let mut scratch = [0.0_f32; 256];
        let mut output = [0_u8; 84];
        let mut bytes = Cursor::new(Vec::new());
        write_expert_sidecar(
            &parsed,
            &checkpoint,
            &specifications,
            PackedOwnedKind::Q2K,
            &mut scratch,
            &mut output,
            &mut bytes,
        )
        .expect("the three low-codec projections write");

        let mut file = tempfile().expect("a temporary mapped sidecar opens");
        file.write_all(bytes.get_ref())
            .expect("the complete sidecar writes");
        file.flush().expect("the sidecar reaches the mapping");
        let mapping =
            Arc::new(unsafe { MmapOptions::new().map(&file) }.expect("the sidecar maps read-only"));
        let sidecar = MappedExpertSidecar::new(mapping).expect("the mapped directory validates");

        let mut slab = ExpertSlab::new();
        for (site, projection) in ExpertProjection::ALL.into_iter().enumerate() {
            slab.bind_layer_stack(
                site,
                proxima_tensor::NodeId(site as u32),
                PackedOwnedKind::Q4K,
                &checkpoint[site * 144..(site + 1) * 144],
                1,
                1,
                256,
            )
            .expect("the checkpoint projection binds");
            slab.register_model_layer_site(0, projection, site);
        }

        sidecar
            .install_low_copies(&mut slab, 1, 1)
            .expect("attachment installs every low projection");
        assert_eq!(slab.memory().owned_bytes, 0);
        assert_eq!(slab.memory().mapped_bytes, 84 * 3);

        let address = ExpertAddress {
            layer: 0,
            expert: 0,
        };
        let mut policy =
            crate::residency::ExpertResidency::<1, 1>::new(crate::residency::ResidencyConfig {
                budget_bytes: sidecar.high_bytes_per_expert(),
                high_bytes_per_expert: sidecar.high_bytes_per_expert(),
                ..crate::residency::ResidencyConfig::default()
            });
        policy
            .observe(
                1,
                0,
                [crate::residency::RoutedExpert {
                    expert: 0,
                    importance: 1.0,
                }],
            )
            .expect("the real routed address enters the fixed policy");
        let actions = policy
            .reconcile::<1>()
            .expect("one promotion fits the fixed action batch");
        policy
            .apply_actions_at_boundary(&mut slab, &actions, |slab, action| {
                sidecar.apply_action(slab, &checkpoint, action)
            })
            .expect("the policy callback promotes all three projections");
        assert_eq!(policy.resident(address), Some(true));
        assert_eq!(slab.memory().mapped_bytes, 0);

        sidecar
            .apply_action(&mut slab, &checkpoint, ResidencyAction::Evict(address))
            .expect("an eviction restores every low projection");
        assert_eq!(slab.memory().owned_bytes, 0);
        assert_eq!(slab.memory().mapped_bytes, 84 * 3);
    }
}
