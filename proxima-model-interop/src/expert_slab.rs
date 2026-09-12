//! The model's own per-layer, per-expert weight table — the OWNER of the
//! bytes [`proxima_tensor::cpu::ExpertSource`] borrows for exactly one
//! decode step.
//!
//! Composes [`proxima_tensor::cpu::ExpertEntry`]/[`proxima_tensor::cpu::ExpertSource`]/
//! [`proxima_tensor::cpu::expert_entries_from_stack`]: `ExpertSource`'s own
//! doc names the shape a residency policy needs -- something OUTSIDE
//! `proxima-tensor` that owns expert bytes across steps and hands the
//! evaluator a fresh borrow each step, never mutating bytes a running step
//! already borrowed. `ExpertSlab` is that owner. [`ExpertSlab::bind_layer_stack`]
//! is the default construction every load path uses: it ALIASES the
//! checkpoint's own bound stack bytes exactly the way
//! `expert_entries_from_stack` does, zero copy, so a checkpoint that never
//! calls [`ExpertSlab::page_expert`] evaluates bit-identically to reading
//! the stack directly.
//!
//! [`ExpertSlab::page_expert`]/[`ExpertSlab::evict_expert`] are the
//! step-boundary surface a residency policy outside this crate drives --
//! both reject the call with [`InteropError::ExpertSwapDuringStep`] while
//! [`ExpertSlab::begin_step`] has been called with no matching
//! [`ExpertSlab::end_step`] yet, so a policy can never race a running
//! step's own borrowed [`ExpertSource`] snapshot.

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

use memmap2::Mmap;
use std::sync::Arc;

use proxima_gguf::quant::{bf16, f16, q2_k, q3_k, q4_0, q4_k, q5_k, q6_k, q8_0};
use proxima_tensor::NodeId;
use proxima_tensor::cpu::{ExpertEntry, ExpertPayloadSpan, ExpertSource};

use crate::bind::{PackedOwnedKind, quantize_to_kind};
use crate::error::InteropError;

pub(crate) type AllLowExpertSourceScratch<'mapping> = Vec<(
    NodeId,
    ExpertProjection,
    Range<usize>,
    Vec<ExpertEntry<'mapping>>,
    Vec<Option<ExpertPayloadSpan>>,
)>;

/// One routed FFN projection belonging to a model layer.
///
/// The discriminant is the fixed slot used by [`ExpertSlab`] to translate a
/// DynaExq `(layer, expert)` address into every gathered-weight table that
/// serves that expert. This lookup is O(1) and allocated only when the model
/// is bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpertProjection {
    Gate = 0,
    Up = 1,
    Down = 2,
}

impl ExpertProjection {
    pub(crate) const ALL: [Self; 3] = [Self::Gate, Self::Up, Self::Down];

    pub(crate) const fn index(self) -> usize {
        self as usize
    }

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Gate => "ffn_gate",
            Self::Up => "ffn_up",
            Self::Down => "ffn_down",
        }
    }
}

/// Encodes `rows` (one MoE expert's dequantized `[out_dim, in_dim]` `f32`
/// weights) into `codec`'s packed byte representation, one row block at a
/// time inside [`quantize_to_kind`]'s own encoder -- no whole-model buffer
/// beyond this single expert's own `f32` rows. A residency policy composes
/// this with [`ExpertSlab::page_expert`]: dequantize the expert's current
/// bytes (`proxima_gguf::quant`'s per-codec `dequantize`/`dequantize_block`),
/// optionally mutate them, re-encode here into a smaller/different codec
/// (`Q2_K` is the smallest [`PackedOwnedKind`] this crate's encoders
/// support), then hand the result to [`ExpertSlab::page_expert`] -- the
/// same [`PackedOwnedKind`] tag both calls share is what lets
/// [`ExpertCopy::entry`] read the paged bytes back as the right
/// [`proxima_tensor::cpu::QuantizedBlock`] variant.
///
/// # Errors
/// [`InteropError::Quant`] if `rows.len()` does not equal `out_dim as usize
/// * in_dim as usize`, or is not a whole multiple of `codec`'s own block
/// width -- [`quantize_to_kind`]'s underlying encoder rejects both.
pub fn encode_expert_copy(
    rows: &[f32],
    out_dim: u32,
    in_dim: u32,
    codec: PackedOwnedKind,
) -> Result<Vec<u8>, InteropError> {
    let element_count = out_dim as usize * in_dim as usize;
    let mut encoded = vec![0u8; codec.byte_len_for(element_count)];
    quantize_to_kind(codec, rows, &mut encoded)?;
    Ok(encoded)
}

/// Encodes one expert into caller-owned storage.
///
/// This is the no-allocation companion to [`encode_expert_copy`].  A HOBBIT
/// page-in caller can reuse one fixed scratch/output region or write directly
/// into an mmap/LSM segment, so conversion does not create a second model-sized
/// heap allocation.
pub fn encode_expert_copy_into(
    rows: &[f32],
    out_dim: u32,
    in_dim: u32,
    codec: PackedOwnedKind,
    output: &mut [u8],
) -> Result<(), InteropError> {
    let element_count = out_dim as usize * in_dim as usize;
    let expected_bytes = codec.byte_len_for(element_count);
    if output.len() != expected_bytes {
        return Err(InteropError::Quant(
            proxima_gguf::quant::QuantError::OutputSizeMismatch {
                found: output.len(),
                expected: expected_bytes,
            },
        ));
    }
    quantize_to_kind(codec, rows, output).map_err(InteropError::Quant)
}

/// Re-encodes one already-packed expert into caller-owned storage.
///
/// This is the low-copy HOBBIT path: the source is decoded into `scratch`
/// one expert at a time and the target bytes are written into `output`. It
/// never allocates and never materializes a model-sized `Vec`; a caller may
/// provide a fixed stack buffer, an arena slot, or an mmap-backed output
/// range. The returned length is the number of target bytes written.
///
/// `source_bytes` must contain whole blocks of `source_codec`, and
/// `scratch` must be exactly large enough for those decoded elements.
/// `output` must be exactly the target codec's packed size for those
/// elements. `Q2K` is intentionally supported as a target, while source
/// codecs are limited to codecs for which this crate has a decoder.
pub fn recode_expert_into(
    source_codec: PackedOwnedKind,
    source_bytes: &[u8],
    target_codec: PackedOwnedKind,
    scratch: &mut [f32],
    output: &mut [u8],
) -> Result<usize, InteropError> {
    let source_layout = source_codec.to_ggml_type().block_layout();
    let block_bytes = source_layout.block_bytes as usize;
    let block_elements = source_layout.block_elements as usize;
    if !source_bytes.len().is_multiple_of(block_bytes) {
        return Err(InteropError::Quant(
            proxima_gguf::quant::QuantError::InputNotBlockMultiple {
                codec: source_codec_name(source_codec),
                found: source_bytes.len(),
                block_bytes,
            },
        ));
    }
    let element_count = source_bytes.len() / block_bytes * block_elements;
    if scratch.len() != element_count {
        return Err(InteropError::Quant(
            proxima_gguf::quant::QuantError::OutputSizeMismatch {
                found: scratch.len(),
                expected: element_count,
            },
        ));
    }
    let target_bytes = target_codec.byte_len_for(element_count);
    if output.len() != target_bytes {
        return Err(InteropError::Quant(
            proxima_gguf::quant::QuantError::OutputSizeMismatch {
                found: output.len(),
                expected: target_bytes,
            },
        ));
    }
    dequantize_expert(source_codec, source_bytes, scratch)?;
    quantize_to_kind(target_codec, scratch, output)?;
    Ok(target_bytes)
}

fn source_codec_name(codec: PackedOwnedKind) -> &'static str {
    match codec {
        PackedOwnedKind::Q2K => "q2_k",
        PackedOwnedKind::Q3K => "q3_k",
        PackedOwnedKind::Q4K => "q4_k",
        PackedOwnedKind::Q5K => "q5_k",
        PackedOwnedKind::Q6K => "q6_k",
        PackedOwnedKind::Q8_0 => "q8_0",
        PackedOwnedKind::Q4_0 => "q4_0",
        PackedOwnedKind::Float16 => "f16",
        PackedOwnedKind::BFloat16 => "bf16",
    }
}

fn dequantize_expert(
    codec: PackedOwnedKind,
    source: &[u8],
    output: &mut [f32],
) -> Result<(), InteropError> {
    match codec {
        PackedOwnedKind::Q2K => q2_k::dequantize(source, output),
        PackedOwnedKind::Q3K => q3_k::dequantize(source, output),
        PackedOwnedKind::Q4K => q4_k::dequantize(source, output),
        PackedOwnedKind::Q5K => q5_k::dequantize(source, output),
        PackedOwnedKind::Q6K => q6_k::dequantize(source, output),
        PackedOwnedKind::Q8_0 => q8_0::dequantize(source, output),
        PackedOwnedKind::Q4_0 => q4_0::dequantize(source, output),
        PackedOwnedKind::Float16 => f16::dequantize(source, output),
        PackedOwnedKind::BFloat16 => bf16::dequantize(source, output),
    }
    .map_err(InteropError::from)
}

/// One expert's backing bytes. The mapped variant owns its mapping handle and
/// byte range rather than a slice into itself, avoiding a self-referential
/// owner while keeping the range valid for every borrowed [`ExpertSource`]
/// snapshot it produces.
#[derive(Debug, Clone)]
enum ExpertBytes<'file> {
    Borrowed(&'file [u8]),
    Owned(Vec<u8>),
    Mapped {
        mapping: Arc<Mmap>,
        range: Range<usize>,
    },
}

impl<'file> ExpertBytes<'file> {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Borrowed(bytes) => bytes,
            Self::Owned(bytes) => bytes,
            Self::Mapped { mapping, range } => &mapping[range.clone()],
        }
    }

    const fn owned_bytes(&self) -> usize {
        match self {
            Self::Owned(bytes) => bytes.len(),
            Self::Borrowed(_) | Self::Mapped { .. } => 0,
        }
    }

    fn mapped_bytes(&self) -> usize {
        match self {
            Self::Mapped { range, .. } => range.len(),
            Self::Borrowed(_) | Self::Owned(_) => 0,
        }
    }

    const fn is_mapped(&self) -> bool {
        matches!(self, Self::Mapped { .. })
    }
}

/// One expert's own weight bytes for one gathered-reduce projection --
/// either ALIASING the checkpoint's own loaded bytes (every
/// [`ExpertSlab::bind_layer_stack`] entry starts this way), an OWNED copy
/// [`ExpertSlab::page_expert`] wrote in between two decode steps, or a range
/// of a live mmap [`ExpertSlab::page_expert_mapped`] retained.
#[derive(Debug, Clone)]
struct ExpertCopy<'file> {
    codec: PackedOwnedKind,
    bytes: ExpertBytes<'file>,
    out_dim: u32,
    in_dim: u32,
    epoch: u64,
}

impl<'file> ExpertCopy<'file> {
    fn entry(&self) -> ExpertEntry<'_> {
        self.entry_with_bytes(self.bytes.as_slice())
    }

    fn entry_with_bytes<'bytes>(&self, bytes: &'bytes [u8]) -> ExpertEntry<'bytes> {
        ExpertEntry {
            block: self.codec.as_block(bytes),
            out_dim: self.out_dim,
            in_dim: self.in_dim,
            epoch: self.epoch,
        }
    }
}

/// Memory retained directly by an [`ExpertSlab`].
///
/// `owned_bytes` is the heap payload the slab itself keeps resident after a
/// copying page operation. `mapped_bytes` is the logical expert range held by
/// mmap handles: it is address space, not a claim that the operating system
/// has faulted those pages into RSS. This distinction keeps a HOBBIT/DynaExq
/// budget from treating a 22.7GB checkpoint mapping as a 22.7GB heap copy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpertSlabMemory {
    pub owned_bytes: usize,
    pub mapped_bytes: usize,
}

/// One forward-program layer's own expert table. `experts[e]` is `None`
/// only after [`ExpertSlab::evict_expert`] removes it with no fallback
/// bytes yet re-paged. [`ExpertSlab::sources_for_step`] rejects that
/// incomplete table before a gather can observe renumbered expert indices.
#[derive(Debug, Clone, Default)]
struct LayerSlab<'file> {
    experts: Vec<Option<ExpertCopy<'file>>>,
    weight_node: Option<NodeId>,
    model_layer: Option<usize>,
}

/// The model's own owned/aliased expert bytes, one `LayerSlab` (private,
/// this module) per forward-program layer carrying a routed MoE weight.
/// See the module doc for the ownership and step-boundary contract.
#[derive(Debug, Clone, Default)]
pub struct ExpertSlab<'file> {
    layers: Vec<LayerSlab<'file>>,
    model_layer_sites: Vec<[Option<usize>; 3]>,
    selected_experts: Vec<Vec<u32>>,
    step_in_progress: bool,
}

impl<'file> ExpertSlab<'file> {
    /// An empty slab -- every layer starts unbound, so
    /// [`Self::sources_for_step`] returns an empty table and every gather
    /// read falls through to the plain contiguous-stack path unchanged
    /// (`run_reduce_with_quantized_weights`'s own `None` default).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records the routed expert IDs for the immediately following gather.
    pub fn set_selected_experts(&mut self, layer: usize, experts: &[u32]) {
        if self.selected_experts.len() <= layer {
            self.selected_experts.resize_with(layer + 1, Vec::new);
        }
        self.selected_experts[layer].clear();
        self.selected_experts[layer].extend_from_slice(experts);
    }

    /// Adds routed expert IDs to the current step's union without replacing
    /// routes selected by earlier rows in the same prefill segment.
    pub fn add_selected_experts(&mut self, layer: usize, experts: &[u32]) {
        if self.selected_experts.len() <= layer {
            self.selected_experts.resize_with(layer + 1, Vec::new);
        }
        for &expert in experts {
            if !self.selected_experts[layer].contains(&expert) {
                self.selected_experts[layer].push(expert);
            }
        }
    }

    /// Starts a fresh selection union for one layer's next segmented
    /// evaluation. The decode loop calls this once before routing rows; the
    /// subsequent gather snapshot then contains every expert any row needs.
    pub fn clear_selected_experts(&mut self, layer: usize) {
        if self.selected_experts.len() <= layer {
            self.selected_experts.resize_with(layer + 1, Vec::new);
        }
        self.selected_experts[layer].clear();
    }

    /// Clears the route union at a token-step boundary while retaining the
    /// fixed per-layer capacity. Prompt rows in one recurrent prefill step
    /// then accumulate their routes instead of staging the same projection
    /// table once per row.
    pub fn clear_selected_experts_for_step(&mut self) {
        for experts in &mut self.selected_experts {
            experts.clear();
        }
    }

    pub(crate) fn selected_experts(&self, layer: usize) -> &[u32] {
        self.selected_experts.get(layer).map_or(&[], Vec::as_slice)
    }

    /// Registers layer `layer`'s own contiguous expert stack under
    /// `weight_node` -- the SAME [`NodeId`] the forward program's gathered
    /// reduce resolves as its quantized operand, so
    /// `run_reduce_with_quantized_weights` finds this layer's table by the
    /// key it already has in hand. Slices `stack` exactly as
    /// [`proxima_tensor::cpu::expert_entries_from_stack`] does -- zero copy,
    /// bit-identical to reading `stack` directly -- and ALIASES the bytes
    /// (`ExpertBytes::Borrowed`), never copying the checkpoint's own
    /// weights.
    ///
    /// # Errors
    /// [`InteropError::ExpertSlabIndexOutOfRange`] if `stack`'s length is
    /// not a whole multiple of `expert_count`.
    // one argument per real degree of freedom a stacked MoE tensor carries
    // (layer, node, codec, bytes, count, and the two declared dims) -- the
    // same shape `proxima_tensor::cpu::expert_entries_from_stack` takes.
    #[allow(clippy::too_many_arguments)]
    pub fn bind_layer_stack(
        &mut self,
        layer: usize,
        weight_node: NodeId,
        codec: PackedOwnedKind,
        stack: &'file [u8],
        expert_count: usize,
        out_dim: u32,
        in_dim: u32,
    ) -> Result<(), InteropError> {
        if expert_count == 0 || !stack.len().is_multiple_of(expert_count) {
            return Err(InteropError::ExpertSlabIndexOutOfRange {
                layer,
                expert: expert_count,
            });
        }
        let per_expert_bytes = stack.len() / expert_count;
        let experts = stack
            .chunks_exact(per_expert_bytes)
            .map(|chunk| {
                Some(ExpertCopy {
                    codec,
                    bytes: ExpertBytes::Borrowed(chunk),
                    out_dim,
                    in_dim,
                    epoch: 0,
                })
            })
            .collect();
        if self.layers.len() <= layer {
            self.layers.resize(layer + 1, LayerSlab::default());
        }
        self.layers[layer] = LayerSlab {
            experts,
            weight_node: Some(weight_node),
            model_layer: None,
        };
        Ok(())
    }

    /// Records the O(1) model-layer coordinate of an already-bound site.
    pub(crate) fn register_model_layer_site(
        &mut self,
        model_layer: usize,
        projection: ExpertProjection,
        site: usize,
    ) {
        if self.model_layer_sites.len() <= model_layer {
            self.model_layer_sites.resize(model_layer + 1, [None; 3]);
        }
        self.model_layer_sites[model_layer][projection.index()] = Some(site);
        if self.layers.len() <= site {
            self.layers.resize(site + 1, LayerSlab::default());
        }
        self.layers[site].model_layer = Some(model_layer);
    }

    pub(crate) fn projection_site(
        &self,
        model_layer: usize,
        projection: ExpertProjection,
    ) -> Result<usize, InteropError> {
        self.model_layer_sites
            .get(model_layer)
            .and_then(|sites| sites[projection.index()])
            .ok_or(InteropError::ExpertSlabIndexOutOfRange {
                layer: model_layer,
                expert: projection.index(),
            })
    }

    /// Returns whether a routed projection still uses its mapped low copy.
    /// High promotions borrow checkpoint bytes, so a caller can skip a
    /// redundant low-sidecar read before constructing the source snapshot.
    pub(crate) fn uses_mapped_low(
        &self,
        model_layer: usize,
        expert: usize,
        projection: ExpertProjection,
    ) -> Result<bool, InteropError> {
        let site = self.projection_site(model_layer, projection)?;
        let copy = self
            .layers
            .get(site)
            .and_then(|layer| layer.experts.get(expert))
            .ok_or(InteropError::ExpertSlabIndexOutOfRange {
                layer: model_layer,
                expert,
            })?;
        Ok(copy.as_ref().is_some_and(|value| value.bytes.is_mapped()))
    }

    /// Marks a decode step as started -- [`Self::page_expert`]/
    /// [`Self::evict_expert`] reject any call until the matching
    /// [`Self::end_step`] runs, so a paging call from inside a
    /// `step_inputs` callback (this crate's own decode-loop hook) is caught
    /// as [`InteropError::ExpertSwapDuringStep`] rather than mutating bytes
    /// a snapshot already borrowed for this step.
    pub fn begin_step(&mut self) {
        self.step_in_progress = true;
    }

    /// Marks the current decode step as finished -- see [`Self::begin_step`].
    pub fn end_step(&mut self) {
        self.step_in_progress = false;
    }

    /// Copies `bytes` in as expert `expert`'s new weight for layer `layer`,
    /// bumping and returning its epoch. Only legal between steps -- see the
    /// module doc.
    ///
    /// # Errors
    /// [`InteropError::ExpertSwapDuringStep`] if called while a step is in
    /// progress; [`InteropError::ExpertSlabIndexOutOfRange`] if `layer` or
    /// `expert` is out of range for this checkpoint's slab.
    pub fn page_expert(
        &mut self,
        layer: usize,
        expert: usize,
        codec: PackedOwnedKind,
        bytes: &[u8],
        out_dim: u32,
        in_dim: u32,
    ) -> Result<u64, InteropError> {
        self.page_expert_with_bytes(
            layer,
            expert,
            codec,
            ExpertBytes::Owned(bytes.to_vec()),
            out_dim,
            in_dim,
        )
    }

    /// Pages an expert from a stable mapped region without copying its bytes.
    ///
    /// HOBBIT's high-precision source is normally an mmap of the checkpoint or
    /// an LSM segment.  The caller retains ownership of that mapping for the
    /// returned slab's lifetime; the slab only records a borrow, so a
    /// promotion does not transiently duplicate an expert in heap memory.
    /// Like [`Self::page_expert`], this is legal only between decode steps.
    ///
    /// # Errors
    /// [`InteropError::ExpertSwapDuringStep`] while a step is running, or
    /// [`InteropError::ExpertSlabIndexOutOfRange`] for an unknown slot.
    pub fn page_expert_borrowed(
        &mut self,
        layer: usize,
        expert: usize,
        codec: PackedOwnedKind,
        bytes: &'file [u8],
        out_dim: u32,
        in_dim: u32,
    ) -> Result<u64, InteropError> {
        self.page_expert_with_bytes(
            layer,
            expert,
            codec,
            ExpertBytes::Borrowed(bytes),
            out_dim,
            in_dim,
        )
    }

    /// Re-encodes one mapped high-precision expert into caller-owned storage
    /// and pages that storage without a heap copy.  The caller supplies the
    /// fixed decode scratch and the stable target range (normally an mmap or
    /// LSM segment); this method only composes [`recode_expert_into`] with the
    /// existing boundary-checked page operation.
    pub fn page_expert_recode_borrowed(
        &mut self,
        layer: usize,
        expert: usize,
        source_codec: PackedOwnedKind,
        source_bytes: &[u8],
        target_codec: PackedOwnedKind,
        scratch: &mut [f32],
        target_bytes: &'file mut [u8],
        out_dim: u32,
        in_dim: u32,
    ) -> Result<u64, InteropError> {
        let written = recode_expert_into(
            source_codec,
            source_bytes,
            target_codec,
            scratch,
            target_bytes,
        )?;
        if written != target_bytes.len() {
            return Err(InteropError::Quant(
                proxima_gguf::quant::QuantError::OutputSizeMismatch {
                    found: written,
                    expected: target_bytes.len(),
                },
            ));
        }
        self.page_expert_borrowed(layer, expert, target_codec, target_bytes, out_dim, in_dim)
    }

    /// Pages one expert from `mapping[range]` without copying the expert or
    /// retaining a borrow from an external owner. The slab keeps an
    /// [`Arc<Mmap>`] until this expert is replaced or evicted, so every
    /// [`ExpertSource`] snapshot built between those transitions sees stable
    /// bytes even after the caller drops its own mapping handle.
    ///
    /// This is the concrete HOBBIT high-precision source seam: the residency
    /// policy passes the selected expert's byte range from a checkpoint or LSM
    /// mapping at a decode-step boundary. No `Vec<u8>` proportional to the
    /// expert count is created.
    ///
    /// # Errors
    /// [`InteropError::ExpertMappedRangeOutOfBounds`] when `range` does not
    /// fit `mapping`, plus the same step/index errors as
    /// [`Self::page_expert`].
    pub fn page_expert_mapped(
        &mut self,
        layer: usize,
        expert: usize,
        codec: PackedOwnedKind,
        mapping: Arc<Mmap>,
        range: Range<usize>,
        out_dim: u32,
        in_dim: u32,
    ) -> Result<u64, InteropError> {
        if mapping.get(range.clone()).is_none() {
            return Err(InteropError::ExpertMappedRangeOutOfBounds {
                start: range.start,
                end: range.end,
                mapping_len: mapping.len(),
            });
        }
        self.page_expert_with_bytes(
            layer,
            expert,
            codec,
            ExpertBytes::Mapped { mapping, range },
            out_dim,
            in_dim,
        )
    }

    fn page_expert_with_bytes(
        &mut self,
        layer: usize,
        expert: usize,
        codec: PackedOwnedKind,
        bytes: ExpertBytes<'file>,
        out_dim: u32,
        in_dim: u32,
    ) -> Result<u64, InteropError> {
        if self.step_in_progress {
            return Err(InteropError::ExpertSwapDuringStep { layer, expert });
        }
        let slot = self
            .layers
            .get_mut(layer)
            .and_then(|layer_slab| layer_slab.experts.get_mut(expert))
            .ok_or(InteropError::ExpertSlabIndexOutOfRange { layer, expert })?;
        let epoch = slot.as_ref().map_or(0, |copy| copy.epoch) + 1;
        *slot = Some(ExpertCopy {
            codec,
            bytes,
            out_dim,
            in_dim,
            epoch,
        });
        Ok(epoch)
    }

    /// Removes expert `expert` of layer `layer`'s currently-bound bytes --
    /// [`Self::sources_for_step`] rejects the incomplete table until a later
    /// [`Self::page_expert`] re-binds it.
    ///
    /// # Errors
    /// Same as [`Self::page_expert`].
    pub fn evict_expert(&mut self, layer: usize, expert: usize) -> Result<(), InteropError> {
        if self.step_in_progress {
            return Err(InteropError::ExpertSwapDuringStep { layer, expert });
        }
        let slot = self
            .layers
            .get_mut(layer)
            .and_then(|layer_slab| layer_slab.experts.get_mut(expert))
            .ok_or(InteropError::ExpertSlabIndexOutOfRange { layer, expert })?;
        *slot = None;
        Ok(())
    }

    /// `expert`'s current epoch for `layer`, or `None` if either index is
    /// out of range or the expert is currently evicted.
    #[must_use]
    pub fn expert_epoch(&self, layer: usize, expert: usize) -> Option<u64> {
        self.layers
            .get(layer)?
            .experts
            .get(expert)?
            .as_ref()
            .map(|copy| copy.epoch)
    }

    /// Returns the slab-owned heap bytes and mmap-backed expert ranges for
    /// the current table. See [`ExpertSlabMemory`] for why these are separate
    /// quantities rather than one misleading "resident bytes" total.
    #[must_use]
    pub fn memory(&self) -> ExpertSlabMemory {
        self.layers
            .iter()
            .flat_map(|layer_slab| layer_slab.experts.iter().flatten())
            .fold(ExpertSlabMemory::default(), |mut memory, expert| {
                memory.owned_bytes += expert.bytes.owned_bytes();
                memory.mapped_bytes += expert.bytes.mapped_bytes();
                memory
            })
    }

    /// The lowest layer index carrying at least one evicted expert with no
    /// paged replacement yet, or `None` if every bound layer is fully
    /// populated. [`Self::sources_for_step`] rejects an incomplete layer
    /// before it can construct a shorter table with shifted indices. Metal's
    /// current experimental arm consumes only uniform packed-codec tables;
    /// mixed-codec HOBBIT remains a typed backend error until its
    /// codec-tagged kernel is added.
    #[must_use]
    pub fn first_incomplete_layer(&self) -> Option<usize> {
        self.layers
            .iter()
            .position(|layer_slab| layer_slab.experts.iter().any(Option::is_none))
    }

    /// The lowest layer whose expert table no longer aliases the original
    /// checkpoint stack exactly. A backend that cannot consume
    /// [`ExpertSource`] must reject both an eviction and a paged replacement:
    /// silently falling back to the original stack would undo either policy
    /// decision while still returning a plausible model result.
    #[must_use]
    pub fn first_modified_layer(&self) -> Option<usize> {
        self.layers.iter().position(|layer_slab| {
            layer_slab
                .experts
                .iter()
                .any(|expert| expert.as_ref().is_none_or(|copy| copy.epoch != 0))
        })
    }

    /// Snapshots every bound layer's own [`ExpertSource`], keyed by that
    /// layer's gathered-reduce weight [`NodeId`] -- the table
    /// `run_reduce_with_quantized_weights` resolves through. `entries` is
    /// caller-owned scratch (never allocated by this call beyond the one
    /// `Vec` push per bound layer) that must outlive the returned map, since
    /// every [`ExpertSource`] borrows into it.
    ///
    /// An evicted slot is rejected rather than omitted: omission would
    /// renumber every following expert and let a gather read the wrong bytes.
    /// The residency policy must page a replacement before the next snapshot.
    #[must_use]
    pub fn sources_for_step<'scratch>(
        &'scratch self,
        entries: &'scratch mut Vec<(NodeId, Vec<ExpertEntry<'scratch>>)>,
    ) -> Result<BTreeMap<NodeId, ExpertSource<'scratch>>, InteropError> {
        entries.clear();
        for (layer, layer_slab) in self.layers.iter().enumerate() {
            let Some(weight_node) = layer_slab.weight_node else {
                continue;
            };
            if let Some(expert) = layer_slab.experts.iter().position(Option::is_none) {
                return Err(InteropError::ExpertSlabIndexOutOfRange { layer, expert });
            }
            let layer_entries = layer_slab
                .experts
                .iter()
                .filter_map(|expert| expert.as_ref().map(ExpertCopy::entry))
                .collect();
            entries.push((weight_node, layer_entries));
        }
        Ok(entries
            .iter()
            .zip(
                self.layers
                    .iter()
                    .filter(|layer| layer.weight_node.is_some()),
            )
            .map(|((node, entries), layer_slab)| {
                let source = layer_slab
                    .model_layer
                    .and_then(|model_layer| self.selected_experts.get(model_layer))
                    .filter(|experts| !experts.is_empty())
                    .map_or_else(
                        || ExpertSource::new(entries),
                        |experts| ExpertSource::with_selected_expert_ids(entries, experts),
                    );
                (*node, source)
            })
            .collect())
    }

    /// Snapshots exactly one layer's expert tables. Segment-local Qwen35
    /// programs reuse small node IDs across layers, so a whole-model map
    /// would overwrite one layer's source with another's.
    #[cfg(test)]
    pub(crate) fn sources_for_layer<'scratch>(
        &'scratch self,
        layer: usize,
        entries: &'scratch mut Vec<(NodeId, Vec<ExpertEntry<'scratch>>)>,
    ) -> Result<BTreeMap<NodeId, ExpertSource<'scratch>>, InteropError> {
        entries.clear();
        for (_site, layer_slab) in self
            .layers
            .iter()
            .enumerate()
            .filter(|(_, candidate)| candidate.model_layer == Some(layer))
        {
            let Some(weight_node) = layer_slab.weight_node else {
                continue;
            };
            let layer_entries = layer_slab
                .experts
                .iter()
                .filter_map(|expert| expert.as_ref().map(ExpertCopy::entry))
                .collect();
            entries.push((weight_node, layer_entries));
        }
        Ok(entries
            .iter()
            .map(|(weight_node, layer_entries)| {
                let source = self
                    .selected_experts
                    .get(layer)
                    .filter(|experts| !experts.is_empty())
                    .map_or_else(
                        || ExpertSource::new(layer_entries),
                        |experts| ExpertSource::with_selected_expert_ids(layer_entries, experts),
                    );
                (*weight_node, source)
            })
            .collect())
    }

    pub(crate) fn sources_for_layer_with_sidecar<'scratch>(
        &'scratch self,
        layer: usize,
        sidecar: Option<&'scratch crate::expert_sidecar::ExpertSidecarReadScratch>,
        entries: &'scratch mut Vec<(NodeId, ExpertProjection, Vec<ExpertEntry<'scratch>>)>,
    ) -> Result<BTreeMap<NodeId, ExpertSource<'scratch>>, InteropError> {
        entries.clear();
        if !self
            .layers
            .iter()
            .any(|candidate| candidate.model_layer == Some(layer))
        {
            return Err(InteropError::ExpertSlabIndexOutOfRange { layer, expert: 0 });
        }
        for (site, layer_slab) in self
            .layers
            .iter()
            .enumerate()
            .filter(|(_, candidate)| candidate.model_layer == Some(layer))
        {
            let Some(weight_node) = layer_slab.weight_node else {
                continue;
            };
            if let Some(expert) = layer_slab.experts.iter().position(Option::is_none) {
                return Err(InteropError::ExpertSlabIndexOutOfRange { layer, expert });
            }
            let projection = self
                .model_layer_sites
                .get(layer)
                .and_then(|sites| sites.iter().position(|candidate| *candidate == Some(site)))
                .and_then(|index| ExpertProjection::ALL.get(index).copied());
            let layer_entries = layer_slab
                .experts
                .iter()
                .enumerate()
                .filter_map(|(expert_index, expert)| {
                    expert.as_ref().map(|expert| {
                        sidecar
                            .and_then(|scratch| {
                                projection
                                    .and_then(|projection| scratch.bytes(expert_index, projection))
                            })
                            .map_or_else(|| expert.entry(), |bytes| expert.entry_with_bytes(bytes))
                    })
                })
                .collect();
            let projection = projection
                .ok_or_else(|| InteropError::ExpertSlabIndexOutOfRange { layer, expert: 0 })?;
            entries.push((weight_node, projection, layer_entries));
        }
        entries
            .iter()
            .map(|(weight_node, projection, layer_entries)| {
                let selected = self
                    .selected_experts
                    .get(layer)
                    .filter(|experts| !experts.is_empty());
                let source = if let (Some(scratch), Some(selected)) = (sidecar, selected) {
                    let arena = scratch.arena(*projection).filter(|(_, spans)| {
                        selected.iter().all(|expert| {
                            usize::try_from(*expert)
                                .ok()
                                .and_then(|index| spans.get(index))
                                .is_some_and(Option::is_some)
                        })
                    });
                    if let Some((bytes, spans)) = arena {
                        ExpertSource::with_selected_expert_arena(
                            layer_entries,
                            selected,
                            bytes,
                            spans,
                        )
                        .map_err(|error| {
                            InteropError::PreGatherExecutionUnsupported {
                                architecture: String::from("qwen35moe"),
                                reason: error.to_string(),
                            }
                        })?
                    } else {
                        ExpertSource::with_selected_expert_ids(layer_entries, selected)
                    }
                } else if let Some(selected) = selected {
                    ExpertSource::with_selected_expert_ids(layer_entries, selected)
                } else {
                    ExpertSource::new(layer_entries)
                };
                Ok((*weight_node, source))
            })
            .collect()
    }

    pub(crate) fn all_low_sources_for_step<'mapping, 'scratch>(
        &self,
        sidecar: &'mapping crate::expert_sidecar::MappedExpertSidecar,
        scratch: &'scratch mut AllLowExpertSourceScratch<'mapping>,
    ) -> Result<BTreeMap<NodeId, ExpertSource<'scratch>>, InteropError>
    where
        'mapping: 'scratch,
    {
        scratch.clear();
        for (site, layer_slab) in self.layers.iter().enumerate() {
            let (Some(weight_node), Some(model_layer)) =
                (layer_slab.weight_node, layer_slab.model_layer)
            else {
                continue;
            };
            let projection = self
                .model_layer_sites
                .get(model_layer)
                .and_then(|sites| sites.iter().position(|candidate| *candidate == Some(site)))
                .and_then(|index| ExpertProjection::ALL.get(index).copied())
                .ok_or(InteropError::ExpertSlabIndexOutOfRange {
                    layer: model_layer,
                    expert: 0,
                })?;
            for expert in 0..layer_slab.experts.len() {
                if !self.uses_mapped_low(model_layer, expert, projection)? {
                    return Err(InteropError::ExpertAllLowSourceRequired {
                        layer: model_layer,
                        expert,
                        projection: projection.name(),
                    });
                }
            }
            let mut entries = Vec::new();
            let mut spans = Vec::new();
            let arena = sidecar.populate_all_low_source(
                model_layer,
                projection,
                &mut entries,
                &mut spans,
            )?;
            scratch.push((weight_node, projection, arena, entries, spans));
        }
        scratch
            .iter()
            .map(|(weight_node, _projection, arena, entries, spans)| {
                ExpertSource::with_all_expert_arena(
                    entries,
                    &sidecar.mapping_bytes()[arena.clone()],
                    spans,
                )
                .map(|source| (*weight_node, source))
                .map_err(|error| InteropError::PreGatherExecutionUnsupported {
                    architecture: String::from("qwen35moe"),
                    reason: error.to_string(),
                })
            })
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::io::Write;

    fn q4k_stack_bytes(expert_count: usize) -> Vec<u8> {
        // one Q4_K super-block per expert: 144 bytes/block
        // (`proxima_gguf::quant::q4_k`'s own on-disk layout), filled with a
        // per-expert byte pattern so a wrong slice is trivially detectable.
        (0..expert_count)
            .flat_map(|expert| core::iter::repeat_n(expert as u8, 144))
            .collect()
    }

    #[test]
    fn bind_layer_stack_aliases_the_checkpoint_bytes_zero_copy() {
        let stack = q4k_stack_bytes(4);
        let mut slab = ExpertSlab::new();

        slab.bind_layer_stack(0, NodeId(7), PackedOwnedKind::Q4K, &stack, 4, 32, 32)
            .expect("a 4-expert Q4_K stack binds");

        let mut entries = Vec::new();
        let sources = slab
            .sources_for_step(&mut entries)
            .expect("a fully populated slab snapshots");
        let source = sources
            .get(&NodeId(7))
            .expect("layer 0's source is present");
        let entry = source
            .entry(NodeId(7), 2, 32, 32)
            .expect("expert 2 resolves");
        assert_eq!(entry.epoch, 0, "an aliased entry starts at epoch 0");
    }

    #[test]
    fn sources_for_layer_includes_every_projection_site() {
        let stack = q4k_stack_bytes(1);
        let mut slab = ExpertSlab::new();
        slab.bind_layer_stack(0, NodeId(10), PackedOwnedKind::Q4K, &stack, 1, 32, 32)
            .expect("gate site binds");
        slab.bind_layer_stack(1, NodeId(11), PackedOwnedKind::Q4K, &stack, 1, 32, 32)
            .expect("up site binds");
        slab.bind_layer_stack(2, NodeId(12), PackedOwnedKind::Q4K, &stack, 1, 32, 32)
            .expect("down site binds");
        slab.register_model_layer_site(4, ExpertProjection::Gate, 0);
        slab.register_model_layer_site(4, ExpertProjection::Up, 1);
        slab.register_model_layer_site(4, ExpertProjection::Down, 2);

        let mut entries = Vec::new();
        let sources = slab
            .sources_for_layer(4, &mut entries)
            .expect("all three sites map to transformer layer four");
        assert_eq!(sources.len(), 3);
        assert!(sources.contains_key(&NodeId(10)));
        assert!(sources.contains_key(&NodeId(11)));
        assert!(sources.contains_key(&NodeId(12)));
    }

    #[test]
    fn recode_q4k_expert_into_q2k_uses_only_caller_storage() {
        let source_values: [f32; 256] =
            core::array::from_fn(|index| (index as f32 - 127.5) * 0.03125);
        let mut source_bytes = [0u8; 144];
        q4_k::quantize(&source_values, &mut source_bytes)
            .expect("the synthetic expert fits one Q4_K block");
        let mut scratch = [0.0f32; 256];
        let mut target_bytes = [0u8; 84];

        let written = recode_expert_into(
            PackedOwnedKind::Q4K,
            &source_bytes,
            PackedOwnedKind::Q2K,
            &mut scratch,
            &mut target_bytes,
        )
        .expect("Q4_K source can be streamed through Q2_K");

        assert_eq!(written, 84, "one Q2_K block is 84 bytes");
        assert_eq!(target_bytes.len(), 84);
        assert!(scratch.iter().all(|value| value.is_finite()));
        let mut decoded = [0.0f32; 256];
        q2_k::dequantize(&target_bytes, &mut decoded)
            .expect("the caller-owned Q2_K output is a valid block");
        assert!(decoded.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn page_expert_bumps_the_epoch_and_swaps_the_bytes() {
        let stack = q4k_stack_bytes(2);
        let mut slab = ExpertSlab::new();
        slab.bind_layer_stack(0, NodeId(1), PackedOwnedKind::Q4K, &stack, 2, 32, 32)
            .expect("a 2-expert Q4_K stack binds");

        let repaged = vec![9u8; 144];
        let epoch = slab
            .page_expert(0, 1, PackedOwnedKind::Q4K, &repaged, 32, 32)
            .expect("paging between steps succeeds");

        assert_eq!(epoch, 1, "the first page bumps epoch 0 -> 1");
        assert_eq!(slab.expert_epoch(0, 1), Some(1));
        assert_eq!(slab.first_modified_layer(), Some(0));

        slab.evict_expert(0, 1)
            .expect("eviction between steps succeeds");
        assert_eq!(slab.first_incomplete_layer(), Some(0));
    }

    #[test]
    fn page_expert_borrowed_keeps_mapped_bytes_without_copying() {
        let stack = q4k_stack_bytes(1);
        let mapped_copy = vec![7u8; 144];
        let mapped_address = mapped_copy.as_ptr();
        let mut slab = ExpertSlab::new();
        slab.bind_layer_stack(0, NodeId(1), PackedOwnedKind::Q4K, &stack, 1, 32, 32)
            .expect("a 1-expert Q4_K stack binds");

        slab.page_expert_borrowed(0, 0, PackedOwnedKind::Q4K, &mapped_copy, 32, 32)
            .expect("a mapped promotion succeeds");

        let mut entries = Vec::new();
        let sources = slab
            .sources_for_step(&mut entries)
            .expect("a fully populated slab snapshots");
        let entry = sources
            .get(&NodeId(1))
            .expect("the mapped layer remains present")
            .entry(NodeId(1), 0, 32, 32)
            .expect("the mapped expert resolves");
        assert_eq!(
            entry
                .block
                .packed_bytes()
                .expect("the Q4_K entry retains packed bytes")
                .as_ptr(),
            mapped_address
        );
        assert_eq!(entry.epoch, 1);
    }

    #[test]
    fn page_expert_mapped_retains_the_mapping_and_reports_no_owned_payload_bytes() {
        let stack = q4k_stack_bytes(1);
        let mut source_file =
            tempfile::tempfile().expect("creates the isolated expert source file");
        source_file
            .write_all(&[7u8; 144])
            .expect("writes one Q4_K-sized expert payload");
        source_file
            .flush()
            .expect("flushes the mapped expert source payload");
        // The temporary file remains open and immutable for this map's whole
        // lifetime, satisfying memmap2's mapping precondition.
        let mapping = Arc::new(
            unsafe { Mmap::map(&source_file) }
                .expect("maps the real expert source without reading it into a Vec"),
        );
        let mapped_address = mapping.as_ptr();
        let mut slab = ExpertSlab::new();
        slab.bind_layer_stack(0, NodeId(1), PackedOwnedKind::Q4K, &stack, 1, 32, 32)
            .expect("a 1-expert Q4_K stack binds");

        slab.page_expert_mapped(
            0,
            0,
            PackedOwnedKind::Q4K,
            Arc::clone(&mapping),
            0..144,
            32,
            32,
        )
        .expect("the mapped expert replaces the checkpoint alias between steps");
        drop(mapping);

        let memory = slab.memory();
        assert_eq!(
            memory.owned_bytes, 0,
            "mmap paging retains no heap copy of the 144-byte expert payload"
        );
        assert_eq!(
            memory.mapped_bytes, 144,
            "the resident-set policy can account for exactly this mapped expert range"
        );

        let mut entries = Vec::new();
        let sources = slab
            .sources_for_step(&mut entries)
            .expect("a fully populated slab snapshots");
        let entry = sources
            .get(&NodeId(1))
            .expect("the mapped layer remains present")
            .entry(NodeId(1), 0, 32, 32)
            .expect("the mapped expert resolves after the caller drops its Arc");
        assert_eq!(
            entry
                .block
                .packed_bytes()
                .expect("the Q4_K entry retains packed bytes")
                .as_ptr(),
            mapped_address,
            "the ExpertSource points into the mmap, not into a copied expert buffer"
        );
    }

    #[test]
    fn page_expert_during_a_step_is_rejected() {
        let stack = q4k_stack_bytes(1);
        let mut slab = ExpertSlab::new();
        slab.bind_layer_stack(0, NodeId(1), PackedOwnedKind::Q4K, &stack, 1, 32, 32)
            .expect("a 1-expert Q4_K stack binds");
        slab.begin_step();

        let result = slab.page_expert(0, 0, PackedOwnedKind::Q4K, &[0u8; 144], 32, 32);

        assert!(
            matches!(
                result,
                Err(InteropError::ExpertSwapDuringStep {
                    layer: 0,
                    expert: 0
                })
            ),
            "paging mid-step must be rejected, got {result:?}"
        );
    }

    #[test]
    fn evict_expert_removes_it_from_the_next_snapshot() {
        let stack = q4k_stack_bytes(2);
        let mut slab = ExpertSlab::new();
        slab.bind_layer_stack(0, NodeId(1), PackedOwnedKind::Q4K, &stack, 2, 32, 32)
            .expect("a 2-expert Q4_K stack binds");

        slab.evict_expert(0, 1)
            .expect("evicting a bound expert succeeds");

        assert_eq!(
            slab.expert_epoch(0, 1),
            None,
            "an evicted expert has no epoch"
        );
        let mut entries = Vec::new();
        let error = slab
            .sources_for_step(&mut entries)
            .expect_err("an incomplete table must not renumber expert indices");
        assert!(
            matches!(
                error,
                InteropError::ExpertSlabIndexOutOfRange {
                    layer: 0,
                    expert: 1
                }
            ),
            "the error must identify the missing slot: {error:?}"
        );
    }

    #[test]
    fn modified_layer_detects_both_eviction_and_paged_replacement() {
        let stack = q4k_stack_bytes(2);
        let mut evicted = ExpertSlab::new();
        evicted
            .bind_layer_stack(0, NodeId(1), PackedOwnedKind::Q4K, &stack, 2, 32, 32)
            .expect("a 2-expert Q4_K stack binds");
        assert_eq!(evicted.first_modified_layer(), None);
        evicted.evict_expert(0, 1).expect("eviction succeeds");
        assert_eq!(evicted.first_modified_layer(), Some(0));

        let mut paged = ExpertSlab::new();
        paged
            .bind_layer_stack(0, NodeId(1), PackedOwnedKind::Q4K, &stack, 2, 32, 32)
            .expect("a 2-expert Q4_K stack binds");
        paged
            .page_expert(0, 1, PackedOwnedKind::Q4K, &[9u8; 144], 32, 32)
            .expect("paging succeeds");
        assert_eq!(paged.first_modified_layer(), Some(0));
    }

    #[test]
    fn recode_page_replaces_the_snapshot_without_owned_payload() {
        let source_values: [f32; 256] =
            core::array::from_fn(|index| (index as f32 - 127.5) * 0.03125);
        let mut source_bytes = [0u8; 144];
        q4_k::quantize(&source_values, &mut source_bytes)
            .expect("the synthetic source fits one Q4_K block");
        let mut scratch = [0.0f32; 256];
        let mut target_bytes = [0u8; 84];
        let mut slab = ExpertSlab::new();
        slab.bind_layer_stack(0, NodeId(7), PackedOwnedKind::Q4K, &source_bytes, 1, 32, 32)
            .expect("the source stack binds");
        let epoch = slab
            .page_expert_recode_borrowed(
                0,
                0,
                PackedOwnedKind::Q4K,
                &source_bytes,
                PackedOwnedKind::Q2K,
                &mut scratch,
                &mut target_bytes,
                32,
                32,
            )
            .expect("the recoded expert pages at the boundary");
        assert_eq!(epoch, 1);
        assert_eq!(slab.expert_epoch(0, 0), Some(1));
        assert_eq!(slab.memory(), ExpertSlabMemory::default());
        let mut entries = Vec::new();
        let sources = slab
            .sources_for_step(&mut entries)
            .expect("a fully populated slab snapshots");
        assert_eq!(
            sources.get(&NodeId(7)).map(|source| source.entries().len()),
            Some(1)
        );
    }
}
