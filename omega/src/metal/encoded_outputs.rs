//! Output nodes a plan stores into a caller's buffer in the node's own numeric format.
//!
//! Teaching pointer: [`execute_plan_with_placements`] routes an output node's binary32 rows into a
//! buffer the caller owns. A node registered here through [`Plan::set_encoded_outputs`] is routed
//! the same way but narrower: the producing op writes its rows where any unplaced node's go (the
//! plan's arena), and a dispatch rendered by [`crate::msl::encode_render`] from the format's
//! description encodes them into the caller's buffer, in the same encoder, right after the
//! producer. Nothing is read back and no host codec runs; the caller's `(node, buffer, offset)`
//! triple keeps its meaning, with `offset` the byte offset in the buffer's own format.

use super::*;
use crate::msl::encode_render::{RenderedEncode, format_encode_kernel_msl};
use crate::encode_render::EncodeRows;
use proxima_numeric::NumericFormat;

/// A compiled encode for one registered output node.
pub(super) struct EncodePipeline {
    pub(super) pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(super) elements_per_thread: usize,
}

fn refuse(node: NodeId, reason: &'static str) -> MetalError {
    MetalError::InputContractChanged { node, reason }
}

impl Plan {
    /// Registers `outputs` as nodes whose placed rows are stored in the given format. Each format
    /// must have a rendered encode ([`crate::msl::encode_render::encode_kernel_refusal`]).
    ///
    /// # Errors
    /// [`MetalError::InputContractChanged`] naming the node of a format with no rendered encode.
    pub fn set_encoded_outputs(
        &mut self,
        outputs: impl IntoIterator<Item = (NodeId, NumericFormat)>,
    ) -> Result<(), MetalError> {
        for (node, format) in outputs {
            crate::msl::encode_render::encode_kernel_refusal(&format).map_err(|reason| refuse(node, reason))?;
            self.encoded_outputs.insert(node, format);
        }
        self.encode_pipelines.borrow_mut().clear();
        Ok(())
    }

    /// Refuses a plan with encoded outputs: the profiling executors dispatch per op and would
    /// write binary32 rows into a narrower buffer instead of encoding them.
    #[cfg(feature = "instrument")]
    pub(super) fn reject_encoded_outputs(&self) -> Result<(), MetalError> {
        match self.encoded_outputs.keys().next() {
            Some(node) => Err(refuse(*node, "the per-op timed executors do not encode outputs")),
            None => Ok(()),
        }
    }

    /// Whether `node` is registered as an encoded output.
    #[must_use]
    pub fn is_encoded_output(&self, node: NodeId) -> bool {
        self.encoded_outputs.contains_key(&node)
    }
}

/// Compiles (or finds) the encode pipeline of every registered node.
pub(super) fn resolve_encode_pipelines(
    device: &ProtocolObject<dyn MTLDevice>,
    plan: &Plan,
) -> Result<(), MetalError> {
    let mut pipelines = plan.encode_pipelines.borrow_mut();
    for (node, format) in &plan.encoded_outputs {
        if pipelines.contains_key(node) {
            continue;
        }
        let rendered: RenderedEncode =
            format_encode_kernel_msl(format, "kv_encode").map_err(|reason| refuse(*node, reason))?;
        let kernel = Kernel {
            source: rendered.source,
            entry: rendered.entry,
            bindings: Vec::new(),
            grid: GridSpec { threads: 0, threadgroup_width: None, depth: 1, grid2d: None },
        };
        let key = format!("encode_render:{format:?}");
        let pipeline = pipeline_for_kernel(device, &kernel, &key, plan.math_mode)?;
        pipelines.insert(
            *node,
            EncodePipeline {
                pipeline,
                elements_per_thread: rendered.elements_per_thread,
            },
        );
    }
    Ok(())
}

/// The bytes the caller's buffer must hold past its offset for `node`'s rows.
pub(super) fn encoded_bytes(plan: &Plan, node: NodeId, element_count: u64) -> Result<usize, MetalError> {
    let format = plan.encoded_outputs.get(&node).ok_or_else(|| refuse(node, "not an encoded output"))?;
    usize::try_from(element_count)
        .ok()
        .and_then(|count| EncodeRows { element_count: count, byte_offset: 0 }.byte_len(format))
        .ok_or_else(|| refuse(node, "the node's rows are not whole blocks of the cache format"))
}

/// Encodes `element_count` binary32 values of `source` into `destination` in the same encoder.
/// With a hazard tracker the source's pending write is ordered first by the tracker's own rule (a
/// barrier only when the source was written since the last one); the destination is not recorded
/// as written, because nothing else in the step reads the rows being added.
pub(super) fn encode_rows(
    encoder: &EncoderGuard,
    compiled: &EncodePipeline,
    source: &DeviceBuffer,
    destination: &MetalBuffer,
    rows: EncodeRows,
    tracker: Option<&mut HazardTracker<HazardId>>,
) {
    if let Some(tracker) = tracker {
        let source_region = hazard_region(&source.0, source.1, rows.element_count * core::mem::size_of::<f32>());
        if tracker.needs_barrier(&[source_region], None) {
            tracker.reset();
            encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        }
        tracker.record(&[source_region], None);
    }
    let threads = rows.threads(compiled.elements_per_thread);
    let width = compiled.pipeline.maxTotalThreadsPerThreadgroup().min(32);
    encoder.setComputePipelineState(&compiled.pipeline);
    // SAFETY: both buffers outlive the encoder (the plan's arena and the caller's placed buffer),
    // slots 0 and 1 are the rendered kernel's two buffers, and the caller-validated destination
    // range holds `threads * bytes_per_thread` bytes past its offset.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&source.0), source.1, 0);
        encoder.setBuffer_offset_atIndex(Some(destination), rows.byte_offset, 1);
    }
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize { width: threads, height: 1, depth: 1 },
        MTLSize { width, height: 1, depth: 1 },
    );
}
