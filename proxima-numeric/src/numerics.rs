//! The run-level numeric decisions: what a model's weights are stored as, what its activations
//! and accumulators are computed in, which rewrite permissions the plan holds.
//!
//! This is the type a serving configuration carries so that "what is this value" is answered in
//! one place. Every field is a [`NumericFormat`], so a codec identity, an activation dtype and an
//! accumulator dtype are the same kind of value and a run that mixes them cannot compile its way
//! past a mismatch.

use crate::decode::decode_blocks;
use crate::error::NumericError;
use crate::float_layout::FloatLayout;
use crate::format::NumericFormat;
use crate::numeric::NumericPolicy;

/// One run's numeric decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Numerics {
    /// How stored weights are packed, e.g. [`NumericFormat::Q4K`].
    pub weights: NumericFormat,
    /// How activations are stored between ops, e.g. [`NumericFormat::Float16`].
    pub activation: NumericFormat,
    /// How a reduction accumulates, e.g. [`NumericFormat::Scalar`] over binary32.
    pub accumulator: NumericFormat,
    /// The dtype a fused epilogue reads its operand at, `None` when no epilogue is fused.
    pub epilogue: Option<NumericFormat>,
    /// Which bit-changing rewrites the plan may apply.
    pub policy: NumericPolicy,
}

impl Numerics {
    /// A bit-exact f32 run over f32-stored weights: the reference configuration every backend is
    /// compared against.
    pub const REFERENCE: Self = Self {
        weights: NumericFormat::Scalar(FloatLayout::BF32),
        activation: NumericFormat::Scalar(FloatLayout::BF32),
        accumulator: NumericFormat::Scalar(FloatLayout::BF32),
        epilogue: None,
        policy: NumericPolicy::bit_exact(),
    };

    /// A run whose weights are one packed format and whose arithmetic is binary32 throughout.
    #[must_use]
    pub const fn with_weights(weights: NumericFormat) -> Self {
        Self { weights, ..Self::REFERENCE }
    }

    /// Decode a weight payload into caller-owned storage under this run's weight format.
    ///
    /// # Errors
    /// Whatever [`decode_blocks`] reports for the weight format.
    pub fn decode_weights(&self, data: &[u8], out: &mut [f32]) -> Result<usize, NumericError> {
        decode_blocks(&self.weights, data, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_layout::BlockLayout;

    #[test]
    fn a_run_names_every_format_it_computes_in() {
        let run = Numerics::with_weights(NumericFormat::Q4K);
        assert_eq!(run.weights, NumericFormat::Q4K);
        assert_eq!(run.activation, Numerics::REFERENCE.activation);
        assert_eq!(run.accumulator, Numerics::REFERENCE.accumulator);
        assert_eq!(run.policy, NumericPolicy::bit_exact());
        assert_eq!(run.epilogue, None);
    }

    #[test]
    fn decoding_weights_goes_through_the_run_format() {
        let run = Numerics::with_weights(NumericFormat::Q4_0);
        let mut block = [0_u8; 18];
        block[..2].copy_from_slice(
            &crate::float_layout_codec::encode(1.0, FloatLayout::F16).to_le_bytes()[..2],
        );
        let mut out = [0.0_f32; 32];
        let blocks = run.decode_weights(&block, &mut out).expect("q4_0 decodes");
        assert_eq!(blocks, 1);
        assert_eq!(&out[..2], &[-8.0, -8.0]);
        assert_eq!(BlockLayout::Q4_0.elements, 32);
    }
}
