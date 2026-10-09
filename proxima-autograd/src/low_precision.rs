//! Low-precision weight views over FP32 master parameters.

use alloc::vec::Vec;

use proxima_gguf::quant::{QuantError, bf4_e2m1, bf8_e5m2};

/// Weight operand format used by a training forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    Fp32,
    Bf8E5M2,
    Bf4E2M1,
}

/// Builds the dequantized operand view used by the forward graph.
///
/// The caller retains and updates `master_weights`; gradients computed with
/// this view are applied directly to those FP32 master values, implementing
/// the identity straight-through estimator.
pub fn weight_view(master_weights: &[f32], format: WeightFormat) -> Result<Vec<f32>, QuantError> {
    match format {
        WeightFormat::Fp32 => Ok(master_weights.into()),
        WeightFormat::Bf8E5M2 => Ok(master_weights
            .iter()
            .map(|value| bf8_e5m2::decode(bf8_e5m2::encode(*value)))
            .collect()),
        WeightFormat::Bf4E2M1 => {
            let mut view = Vec::with_capacity(master_weights.len());
            for pair in master_weights.chunks(2) {
                let first = pair[0];
                let second = pair.get(1).copied().unwrap_or(0.0);
                let packed = bf4_e2m1::pack_pair(first, second)?;
                let decoded = bf4_e2m1::unpack_pair(packed);
                view.extend_from_slice(&decoded[..pair.len()]);
            }
            Ok(view)
        }
    }
}
