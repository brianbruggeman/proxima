//! Sequential low-codec expert sidecars for HOBBIT-backed residency.
//!
//! The source is a parsed GGUF directory plus a borrowed file mapping. Each
//! expert is decoded and re-encoded into caller-owned buffers before its
//! bytes are written. The sidecar therefore never constructs a packed
//! whole-model buffer; its descriptor table is the only metadata allocation.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::sync::Arc;

use memmap2::Mmap;
use proxima_gguf::{GgmlType, ParsedGguf};

use crate::bind::PackedOwnedKind;
use crate::expert_slab::{ExpertProjection, ExpertSlab};
use crate::residency::{ExpertAddress, ResidencyAction};
use crate::{InteropError, recode_expert_into};

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
    expert_count: usize,
    descriptor_indices: Vec<[Option<usize>; 3]>,
    high_bytes_per_expert: u64,
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
            expert_count,
            descriptor_indices,
            high_bytes_per_expert,
        })
    }

    /// Reads a range through the source file when available, avoiding mmap
    /// page faults for sparse expert payloads.
    pub fn read_range(&self, range: Range<usize>) -> Result<Vec<u8>, InteropError> {
        let Some(source_file) = &self.source_file else {
            return Ok(self.mapping[range].to_vec());
        };
        let mut bytes = vec![0_u8; range.len()];
        let mut reader = source_file.try_clone().map_err(InteropError::SidecarIo)?;
        reader
            .seek(SeekFrom::Start(range.start as u64))
            .map_err(InteropError::SidecarIo)?;
        reader
            .read_exact(&mut bytes)
            .map_err(InteropError::SidecarIo)?;
        Ok(bytes)
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
            if self.source_file.is_some() {
                let bytes = self.read_range(range)?;
                slab.page_expert(
                    address.layer,
                    address.expert,
                    descriptor.target_codec,
                    &bytes,
                    descriptor.out_dim,
                    descriptor.in_dim,
                )?;
            } else {
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
        }
        Ok(())
    }

    pub(crate) fn apply_action<'file>(
        &self,
        slab: &mut ExpertSlab<'file>,
        checkpoint: &'file [u8],
        action: ResidencyAction,
    ) -> Result<(), InteropError> {
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
