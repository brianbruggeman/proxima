use proxima_tensor::{Executes, Form};

#[cfg(feature = "metal-tiled-gemm")]
use crate::msl::with_contraction_grants;

/// What the Metal renderer executes, as the value the binder admits fusions by.
///
/// Composes [`proxima_tensor::Executes`], the same value `crate::cuda::executes`
/// builds for CUDA, so the binder reads one shape for every backend. Each grant
/// mirrors an arm of `crate::msl::emit_and_classify`'s kind match:
/// `CachedAttention`, `WholeAttention` (the per-row static and single-range
/// forms; `CachedAttentionWholePlanesNotSupported` refuses the rest),
/// `CachedSoftmaxWeights`, `ElementwiseTwin`,
/// `GatedDeltaNet`, `MoeTopK` and `TopFractionSelect` have renderers
/// unconditionally; `RoundBatchedReduce` renders only with
/// `metal-moe-mul-mat-id` (the other arm returns `EpilogueNotSupported`);
/// `BlockQuantize` returns `BlockQuantizeNotSupported`, so it is not granted.
/// The reduce epilogue forms are rendered inside `render_reduce`.
#[must_use]
pub fn executes() -> Executes {
    let granted = Executes::BASE
        .with(Form::ElementwiseTwin)
        .with(Form::ReduceEpilogue)
        .with(Form::ReduceBroadcastEpilogue)
        .with(Form::CachedAttention)
        .with(Form::WholeAttention)
        .with(Form::CachedSoftmaxWeights)
        .with(Form::AttentionSink)
        .with(Form::GatedDeltaNet)
        .with(Form::MoeTopK)
        .with(Form::TopFractionSelect);
    let granted = if cfg!(feature = "metal-moe-mul-mat-id") {
        granted.with(Form::RoundBatchedReduce)
    } else {
        granted
    };
    #[cfg(feature = "metal-tiled-gemm")]
    {
        with_contraction_grants(granted)
    }
    #[cfg(not(feature = "metal-tiled-gemm"))]
    {
        granted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_grants_the_moe_top_k_form_it_fused_before_admission_was_a_value() {
        assert!(executes().forms.contains(Form::MoeTopK));
    }

    #[test]
    fn metal_grants_whole_attention_because_its_per_row_kernels_read_whole_planes() {
        assert!(executes().forms.contains(Form::WholeAttention));
    }

    #[test]
    fn metal_refuses_block_quantize_because_its_emitter_returns_not_supported() {
        assert!(!executes().forms.contains(Form::BlockQuantize));
    }
}
