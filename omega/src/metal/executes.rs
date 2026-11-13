use proxima_tensor::{Executes, Form};

/// What the Metal renderer executes, as the value the binder admits fusions by.
///
/// Composes [`proxima_tensor::Executes`], the same value `crate::cuda::executes`
/// builds for CUDA, so the binder reads one shape for every backend. Each grant
/// mirrors an arm of `crate::msl::emit_and_classify`'s kind match:
/// `CachedAttention`, `CachedSoftmaxWeights`, `ElementwiseTwin`,
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
        .with(Form::CachedSoftmaxWeights)
        .with(Form::AttentionSink)
        .with(Form::GatedDeltaNet)
        .with(Form::MoeTopK)
        .with(Form::TopFractionSelect);
    if cfg!(feature = "metal-moe-mul-mat-id") {
        granted.with(Form::RoundBatchedReduce)
    } else {
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
    fn metal_refuses_block_quantize_because_its_emitter_returns_not_supported() {
        assert!(!executes().forms.contains(Form::BlockQuantize));
    }
}
