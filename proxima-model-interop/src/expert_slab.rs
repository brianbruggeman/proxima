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

use alloc::borrow::Cow;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use proxima_tensor::NodeId;
use proxima_tensor::cpu::{ExpertEntry, ExpertSource};

use crate::bind::{PackedOwnedKind, quantize_to_kind};
use crate::error::InteropError;

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

/// One expert's own weight bytes for one gathered-reduce projection --
/// either ALIASING the checkpoint's own loaded bytes (every
/// [`ExpertSlab::bind_layer_stack`] entry starts this way) or an OWNED copy
/// [`ExpertSlab::page_expert`] wrote in between two decode steps.
#[derive(Debug, Clone)]
struct ExpertCopy<'file> {
    codec: PackedOwnedKind,
    bytes: Cow<'file, [u8]>,
    out_dim: u32,
    in_dim: u32,
    epoch: u64,
}

impl<'file> ExpertCopy<'file> {
    fn entry(&self) -> ExpertEntry<'_> {
        ExpertEntry {
            block: self.codec.as_block(&self.bytes),
            out_dim: self.out_dim,
            in_dim: self.in_dim,
            epoch: self.epoch,
        }
    }
}

/// One forward-program layer's own expert table. `experts[e]` is `None`
/// only after [`ExpertSlab::evict_expert`] removes it with no fallback
/// bytes yet re-paged -- [`ExpertSlab::sources_for_step`] simply omits an
/// absent expert from that layer's [`ExpertSource`], which surfaces to a
/// gather that selects it as [`proxima_tensor::TensorError::GatherIndexOutOfRange`],
/// the same typed error an out-of-range expert index gets today.
#[derive(Debug, Clone, Default)]
struct LayerSlab<'file> {
    experts: Vec<Option<ExpertCopy<'file>>>,
    weight_node: Option<NodeId>,
}

/// The model's own owned/aliased expert bytes, one `LayerSlab` (private,
/// this module) per forward-program layer carrying a routed MoE weight.
/// See the module doc for the ownership and step-boundary contract.
#[derive(Debug, Clone, Default)]
pub struct ExpertSlab<'file> {
    layers: Vec<LayerSlab<'file>>,
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

    /// Registers layer `layer`'s own contiguous expert stack under
    /// `weight_node` -- the SAME [`NodeId`] the forward program's gathered
    /// reduce resolves as its quantized operand, so
    /// `run_reduce_with_quantized_weights` finds this layer's table by the
    /// key it already has in hand. Slices `stack` exactly as
    /// [`proxima_tensor::cpu::expert_entries_from_stack`] does -- zero copy,
    /// bit-identical to reading `stack` directly -- and ALIASES the bytes
    /// (`Cow::Borrowed`), never copying the checkpoint's own weights.
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
            return Err(InteropError::ExpertSlabIndexOutOfRange { layer, expert: expert_count });
        }
        let per_expert_bytes = stack.len() / expert_count;
        let experts = stack
            .chunks_exact(per_expert_bytes)
            .map(|chunk| {
                Some(ExpertCopy { codec, bytes: Cow::Borrowed(chunk), out_dim, in_dim, epoch: 0 })
            })
            .collect();
        if self.layers.len() <= layer {
            self.layers.resize(layer + 1, LayerSlab::default());
        }
        self.layers[layer] = LayerSlab { experts, weight_node: Some(weight_node) };
        Ok(())
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
            bytes: Cow::Owned(bytes.to_vec()),
            out_dim,
            in_dim,
            epoch,
        });
        Ok(epoch)
    }

    /// Removes expert `expert` of layer `layer`'s currently-bound bytes --
    /// [`Self::sources_for_step`] omits it from that layer's
    /// [`ExpertSource`] until a later [`Self::page_expert`] re-binds it.
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
        self.layers.get(layer)?.experts.get(expert)?.as_ref().map(|copy| copy.epoch)
    }

    /// Snapshots every bound layer's own [`ExpertSource`], keyed by that
    /// layer's gathered-reduce weight [`NodeId`] -- the table
    /// `run_reduce_with_quantized_weights` resolves through. `entries` is
    /// caller-owned scratch (never allocated by this call beyond the one
    /// `Vec` push per bound layer) that must outlive the returned map, since
    /// every [`ExpertSource`] borrows into it.
    #[must_use]
    pub fn sources_for_step<'scratch>(
        &'scratch self,
        entries: &'scratch mut Vec<(NodeId, Vec<ExpertEntry<'scratch>>)>,
    ) -> BTreeMap<NodeId, ExpertSource<'scratch>> {
        entries.clear();
        for layer_slab in &self.layers {
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
        entries
            .iter()
            .map(|(node, entries)| (*node, ExpertSource::new(entries)))
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

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
        let sources = slab.sources_for_step(&mut entries);
        let source = sources.get(&NodeId(7)).expect("layer 0's source is present");
        let entry = source
            .entry(NodeId(7), 2, 32, 32)
            .expect("expert 2 resolves");
        assert_eq!(entry.epoch, 0, "an aliased entry starts at epoch 0");
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
            matches!(result, Err(InteropError::ExpertSwapDuringStep { layer: 0, expert: 0 })),
            "paging mid-step must be rejected, got {result:?}"
        );
    }

    #[test]
    fn evict_expert_removes_it_from_the_next_snapshot() {
        let stack = q4k_stack_bytes(2);
        let mut slab = ExpertSlab::new();
        slab.bind_layer_stack(0, NodeId(1), PackedOwnedKind::Q4K, &stack, 2, 32, 32)
            .expect("a 2-expert Q4_K stack binds");

        slab.evict_expert(0, 1).expect("evicting a bound expert succeeds");

        assert_eq!(slab.expert_epoch(0, 1), None, "an evicted expert has no epoch");
        let mut entries = Vec::new();
        let sources = slab.sources_for_step(&mut entries);
        let source = sources.get(&NodeId(1)).expect("layer 0's source is still present");
        assert!(
            source.entry(NodeId(1), 1, 32, 32).is_err(),
            "the evicted slot's index must no longer resolve"
        );
    }
}
