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
///
/// With the `config` feature this is a conflaguration `Settings` (env prefix `NUMERICS`, e.g.
/// `NUMERICS_WEIGHTS=q4_k`, `NUMERICS_POLICY_CONTRACTION=true`) and a serde struct, and the
/// defaults are [`Numerics::REFERENCE`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "config", derive(conflaguration::Settings))]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "config", settings(prefix = "NUMERICS"))]
#[cfg_attr(feature = "config", serde(default, deny_unknown_fields))]
pub struct Numerics {
    /// How stored weights are packed, e.g. [`NumericFormat::Q4K`].
    #[cfg_attr(feature = "config", setting(default_str = "float32"))]
    pub weights: NumericFormat,
    /// How activations are stored between ops, e.g. [`NumericFormat::Float16`].
    #[cfg_attr(feature = "config", setting(default_str = "float32"))]
    pub activation: NumericFormat,
    /// How a reduction accumulates, e.g. [`NumericFormat::Scalar`] over binary32.
    #[cfg_attr(feature = "config", setting(default_str = "float32"))]
    pub accumulator: NumericFormat,
    /// The dtype a fused epilogue reads its operand at, `None` when no epilogue is fused.
    pub epilogue: Option<NumericFormat>,
    /// Which bit-changing rewrites the plan may apply.
    #[cfg_attr(feature = "config", setting(nested))]
    pub policy: NumericPolicy,
}

impl Default for Numerics {
    fn default() -> Self {
        Self::REFERENCE
    }
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

#[cfg(all(test, feature = "config"))]
mod config_tests {
    use super::*;

    #[test]
    fn the_settings_default_is_the_reference_run() {
        assert_eq!(Numerics::default(), Numerics::REFERENCE);
        let defaulted: Numerics = toml::from_str("").expect("an empty table is the default");
        assert_eq!(defaulted, Numerics::REFERENCE);
    }

    #[test]
    fn builder_env_and_toml_agree_for_a_mixed_run() {
        use conflaguration::Settings;

        let expected = Numerics {
            weights: NumericFormat::Q4K,
            activation: NumericFormat::BFloat16,
            accumulator: NumericFormat::Scalar(FloatLayout::F64),
            epilogue: Some(NumericFormat::Float16),
            policy: NumericPolicy::bit_exact().with_contraction(true),
        };
        let from_toml: Numerics = toml::from_str(
            "weights = 'q4_k'\nactivation = 'bf16'\naccumulator = 'float64'\n\
             epilogue = 'float16'\n[policy]\ncontraction = true\n",
        )
        .expect("numerics TOML deserializes");
        let from_env = temp_env::with_vars(
            [
                ("NUMERICS_WEIGHTS", Some("q4_k")),
                ("NUMERICS_ACTIVATION", Some("bf16")),
                ("NUMERICS_ACCUMULATOR", Some("float64")),
                ("NUMERICS_EPILOGUE", Some("float16")),
                ("NUMERICS_POLICY_CONTRACTION", Some("true")),
            ],
            || Numerics::from_env().expect("numerics env parses"),
        );
        let from_fields = Numerics::with_weights(NumericFormat::Q4K);
        let from_fields = Numerics {
            activation: NumericFormat::BFloat16,
            accumulator: NumericFormat::Scalar(FloatLayout::F64),
            epilogue: Some(NumericFormat::Float16),
            policy: NumericPolicy::bit_exact().with_contraction(true),
            ..from_fields
        };

        assert_eq!(from_toml, expected);
        assert_eq!(from_env, expected);
        assert_eq!(from_fields, expected);
    }

    #[test]
    fn an_unset_environment_resolves_to_the_reference_run() {
        use conflaguration::Settings;

        let cleared = [
            "NUMERICS_WEIGHTS",
            "NUMERICS_ACTIVATION",
            "NUMERICS_ACCUMULATOR",
            "NUMERICS_EPILOGUE",
            "NUMERICS_POLICY_CONTRACTION",
        ]
        .map(|name| (name, None::<&str>));
        let resolved = temp_env::with_vars(cleared, || Numerics::from_env().expect("resolves"));

        assert_eq!(resolved, Numerics::REFERENCE);
    }
}
