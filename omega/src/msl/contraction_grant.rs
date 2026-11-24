use proxima_tensor::{ContractionLimits, Executes, Form};

/// `base` plus the contraction forms the Metal renderer executes, with the token
/// minimum from `[tiled_gemm] min_tokens`. The renderer's classifier and
/// `crate::metal::executes` both read this one value, so what the binder admits and
/// what the emitter renders cannot drift.
#[must_use]
pub(crate) fn with_contraction_grants(base: Executes) -> Executes {
    let granted = base.with(Form::TiledContraction).with_contraction(ContractionLimits {
        tiled_min_tokens: crate::sized::TILED_GEMM_MIN_TOKENS,
        grouped_min_tokens: crate::sized::TILED_GEMM_MIN_TOKENS,
    });
    if cfg!(feature = "metal-grouped-gemm") {
        granted.with(Form::GroupedContraction)
    } else {
        granted
    }
}
