//! The renderer's part of the quantized activation decision: whether a configured activation storage
//! format has a CUDA producer and readers. Which readers get a `BlockQuantize` node is decided by
//! `proxima_tensor::derive_activation_quantize`; [`derive_through_tensor_pass`] runs it for a plan.

use proxima_numeric::{NumericFormat, NumericPolicy};
use proxima_tensor::{BoundOp, Missing, Op, Shapes, derive_activation_quantize};

use crate::cuda::target::Target;
use crate::cuda::{consumer_missing, executes, producer_missing};
use crate::lowering::{Codec, PackedOperands};

/// A configured activation storage format the plan cannot honour, and the render pieces it lacks.
/// The plan refuses instead of quietly keeping the activation in binary32.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("activation storage {format:?} is not servable: producer {producer:?}, consumer {consumer:?}")]
pub struct ActivationRefusal {
    pub format: NumericFormat,
    /// The CUDA quantize render's missing piece, with the file and line that found it missing.
    pub producer: Option<Missing>,
    /// The activation readers' missing piece (int8 gemm, staged matvec), likewise.
    pub consumer: Option<Missing>,
}

/// Why a configured format cannot be the plan's activation storage, `None` when the producer and
/// the readers can serve it. A non-block format (a scalar) is not a BlockQuantize concern.
fn storage_refusal(format: NumericFormat) -> Option<ActivationRefusal> {
    let Some(layout) = format.block_layout() else {
        return Some(ActivationRefusal { format, producer: None, consumer: None });
    };
    let producer = producer_missing(&layout).err();
    let consumer = consumer_missing(&layout).err();
    (producer.is_some() || consumer.is_some()).then_some(ActivationRefusal { format, producer, consumer })
}

/// Whether the plan can store activations in the configured block `format` (a descriptor's
/// `activation_storage`); `None` format is the default Q8_1 pair, always servable.
///
/// # Errors
/// [`ActivationRefusal`] naming the producer and reader pieces the format lacks, so a configured
/// format is never silently left in binary32. Boxed: the refusal carries a 700-byte format and is
/// built once per plan, never on a hot path.
pub(crate) fn validate_storage(format: Option<NumericFormat>) -> Result<(), Box<ActivationRefusal>> {
    match format.and_then(storage_refusal) {
        Some(refusal) => Err(Box::new(refusal)),
        None => Ok(()),
    }
}

/// Binds the plan's quantized activations through the tensor pass against what CUDA executes, and
/// records each held format in `packed`. Entries at or past `program.len()` are dropped first, so a
/// refresh never sees the previous plan's derived nodes. The caller has passed `storage` through
/// [`validate_storage`].
pub(crate) fn derive_through_tensor_pass(
    resolved: Vec<BoundOp>,
    program: &[Op],
    shapes: &Shapes,
    packed: &mut PackedOperands,
    policy: NumericPolicy,
    storage: Option<Codec>,
) -> Vec<BoundOp> {
    packed.retain(|node, _| (node.0 as usize) < program.len());
    let (bound, held) = derive_activation_quantize(resolved, program, shapes, packed, policy, &executes(&Target::CUDA), storage);
    packed.extend(held);
    bound
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_float_coded_block_format_is_refused_naming_producer_and_reader_pieces_with_lines() {
        let refusal = validate_storage(Some(NumericFormat::Mxfp4)).expect_err("mxfp4 has no cuda render yet");
        let producer = refusal.producer.expect("the quantize render lacks a piece");
        let consumer = refusal.consumer.expect("the int8 readers lack a piece");
        assert!(producer.file.ends_with("cuda/block_quantize.rs") && producer.line > 0, "{producer:?}");
        assert!(consumer.file.ends_with("cuda/block_quantize.rs") && consumer.line > 0, "{consumer:?}");
    }

    #[test]
    fn a_weight_only_int8_layout_without_the_stored_sum_is_refused_not_misrendered() {
        assert!(validate_storage(Some(NumericFormat::Q8_0)).is_err());
    }

    #[test]
    fn the_default_activation_formats_are_servable() {
        assert!(validate_storage(None).is_ok());
        assert!(validate_storage(Some(NumericFormat::Q8_1Grouped)).is_ok());
    }
}
