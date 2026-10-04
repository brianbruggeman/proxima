//! `RoPE` frequency scaling: how a checkpoint (or a caller) stretches the
//! position table past the context it was trained at.
//!
//! [`RopeScaling`] is plain data the way [`crate::serving::ServingConfig`]
//! is: it is read from the GGUF's `{arch}.rope.scaling.*` keys
//! (`RopeScaling::from_gguf`) or supplied per call through
//! `ServingConfig::rope_scaling`, and consumed once per step by
//! `crate::generate`'s `build_position_inputs`, which composes
//! `RopeScaling::inv_frequencies` and `RopeScaling::attention_factor`
//! into the same `rope_cos`/`rope_sin` table every architecture already
//! binds -- no kernel change, no new program input. The arithmetic follows
//! `transformers`' `_compute_yarn_parameters`
//! (`proxima-tensor/specs/long-context/yarn-worked-example.md` walks it
//! by hand).

#[cfg(feature = "std")]
use alloc::format;
#[cfg(feature = "std")]
use alloc::string::ToString;
#[cfg(feature = "std")]
use alloc::vec::Vec;

#[cfg(feature = "std")]
use proxima_gguf::pipe::ParsedGguf;
#[cfg(feature = "std")]
use proxima_gguf::value::MetadataValue;

#[cfg(feature = "std")]
use crate::bind::{metadata_str, metadata_u32};
#[cfg(feature = "std")]
use crate::error::InteropError;

/// `beta_fast` / `beta_slow` defaults of the `YaRN` paper (arXiv 2309.00071)
/// and of `transformers`.
#[cfg(feature = "std")]
const YARN_BETA_FAST_DEFAULT: f32 = 32.0;
#[cfg(feature = "std")]
const YARN_BETA_SLOW_DEFAULT: f32 = 1.0;

/// The scaling law applied to `RoPE`'s inverse frequencies.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RopeScaling {
    /// Unscaled: every pair rotates at `base^(-2i/dim)`.
    #[default]
    None,
    /// Position interpolation: every inverse frequency divided by `factor`.
    Linear { factor: f32 },
    /// `YaRN`: per-pair blend of interpolation and extrapolation, plus a
    /// logit-temperature `attention_factor` applied to cos and sin.
    Yarn {
        /// Context stretch, e.g. `4.0` for Qwen3-8B's 32768 -> 131072.
        factor: f32,
        /// The context the checkpoint was trained at before scaling, e.g.
        /// `32768` for Qwen3-8B (not the GGUF's `context_length` 40960).
        original_context: u32,
        /// Multiplier on the extrapolation share of the blend (llama.cpp's
        /// `yarn_ext_factor`); `1.0` reproduces `transformers`.
        extrapolation_factor: f32,
        /// Multiplies both cos and sin, so attention logits scale by its
        /// square; `0.1 * ln(factor) + 1` unless the checkpoint states one.
        attention_factor: f32,
        /// Rotations at or above which a pair is left unscaled, e.g. `32.0`.
        beta_fast: f32,
        /// Rotations at or below which a pair is fully interpolated, e.g. `1.0`.
        beta_slow: f32,
    },
}

impl RopeScaling {
    /// The longest context this scaling admits for a checkpoint trained at
    /// `trained`: `original_context * factor` under `YaRN` (the scaling
    /// config's own base, not the GGUF value), `trained * factor` under
    /// linear interpolation, `trained` unscaled.
    #[must_use]
    pub fn limit(&self, trained: u32) -> u32 {
        match *self {
            Self::None => trained,
            Self::Linear { factor } => scaled_limit(trained, factor),
            Self::Yarn {
                factor,
                original_context,
                ..
            } => scaled_limit(original_context, factor),
        }
    }
}

// clamped into u32's range first, so the cast neither truncates nor wraps; no safe f64 -> u32 exists
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn scaled_limit(base: u32, factor: f32) -> u32 {
    let scaled = f64::from(base) * f64::from(factor);
    scaled.clamp(0.0, f64::from(u32::MAX)) as u32
}

// the one deliberate f64 -> f32 rounding; no safe conversion exists
#[cfg(feature = "std")]
#[allow(clippy::cast_possible_truncation)]
fn round_to_f32(value: f64) -> f32 {
    value as f32
}

/// `value` as the nearest `f32`, the same rounding `value as f32` performs;
/// the lint-clean spelling for the `u32` token ids and head indices the
/// position table converts.
#[cfg(feature = "std")]
#[must_use]
pub(crate) fn f32_from_u32(value: u32) -> f32 {
    round_to_f32(f64::from(value))
}

#[cfg(feature = "std")]
impl RopeScaling {
    /// A `YaRN` scaling with the published defaults: extrapolation factor
    /// `1.0`, `beta_fast = 32`, `beta_slow = 1`, and the derived attention
    /// factor `0.1 * ln(factor) + 1`.
    #[must_use]
    pub fn yarn(factor: f32, original_context: u32) -> Self {
        Self::Yarn {
            factor,
            original_context,
            extrapolation_factor: 1.0,
            attention_factor: 0.1 * factor.ln() + 1.0,
            beta_fast: YARN_BETA_FAST_DEFAULT,
            beta_slow: YARN_BETA_SLOW_DEFAULT,
        }
    }

    /// Reads `{arch}.rope.scaling.*` off `parsed`, using llama.cpp's key
    /// names: `type` (`none`, `linear`, `yarn`; absent means `none`),
    /// `factor`, `original_context_length`, `attn_factor`, and
    /// `yarn_ext_factor` / `yarn_beta_fast` / `yarn_beta_slow`. An absent
    /// `attn_factor` derives `0.1 * ln(factor) + 1`; a present one is used
    /// verbatim.
    ///
    /// # Errors
    ///
    /// [`InteropError::UnknownRopeScalingType`] for any other `type`;
    /// [`InteropError::MissingMetadataKey`] when `linear`/`yarn` lack
    /// `factor` (or `yarn` lacks `original_context_length`).
    pub fn from_gguf(parsed: &ParsedGguf) -> Result<Self, InteropError> {
        let family = metadata_str(parsed, "general.architecture")?;
        let key = |name: &str| format!("{family}.rope.scaling.{name}");

        let scaling_type = parsed
            .metadata_value(&key("type"))
            .and_then(MetadataValue::as_str)
            .unwrap_or("none");
        match scaling_type {
            "none" => Ok(Self::None),
            "linear" => Ok(Self::Linear {
                factor: required_f32(parsed, &key("factor"))?,
            }),
            "yarn" => Ok(Self::Yarn {
                factor: required_f32(parsed, &key("factor"))?,
                original_context: metadata_u32(parsed, &key("original_context_length"))?,
                extrapolation_factor: optional_f32(parsed, &key("yarn_ext_factor")).unwrap_or(1.0),
                attention_factor: match optional_f32(parsed, &key("attn_factor")) {
                    Some(attention_factor) => attention_factor,
                    None => 0.1 * required_f32(parsed, &key("factor"))?.ln() + 1.0,
                },
                beta_fast: optional_f32(parsed, &key("yarn_beta_fast"))
                    .unwrap_or(YARN_BETA_FAST_DEFAULT),
                beta_slow: optional_f32(parsed, &key("yarn_beta_slow"))
                    .unwrap_or(YARN_BETA_SLOW_DEFAULT),
            }),
            found => Err(InteropError::UnknownRopeScalingType {
                found: found.to_string(),
            }),
        }
    }

    /// The factor `build_position_inputs` multiplies into both cos
    /// and sin: the `YaRN` `attention_factor`, `1.0` for every other scaling.
    #[must_use]
    pub fn attention_factor(&self) -> f32 {
        match *self {
            Self::Yarn {
                attention_factor, ..
            } => attention_factor,
            Self::None | Self::Linear { .. } => 1.0,
        }
    }

    /// One inverse frequency per rotary pair (`head_dim / 2` of them), or
    /// `None` for [`Self::None`] so the caller keeps its own unscaled
    /// expression bit for bit. Computed in `f64` and rounded once to `f32`,
    /// so the table is the correctly rounded value the reference walk
    /// produces; the angle itself (`position * inv_freq`) stays `f32`, as
    /// in the reference.
    #[must_use]
    pub fn inv_frequencies(&self, head_dim: u32, rope_freq_base: f32) -> Option<Vec<f32>> {
        let pairs = head_dim / 2;
        let dimension = f64::from(head_dim);
        let base = f64::from(rope_freq_base);
        let position_frequency = |pair: u32| base.powf(f64::from(2 * pair) / dimension);
        match *self {
            Self::None => None,
            Self::Linear { factor } => Some(
                (0..pairs)
                    .map(|pair| round_to_f32(1.0 / (f64::from(factor) * position_frequency(pair))))
                    .collect(),
            ),
            Self::Yarn {
                factor,
                original_context,
                extrapolation_factor,
                beta_fast,
                beta_slow,
                ..
            } => {
                let (low, high) =
                    yarn_correction_range(beta_fast, beta_slow, original_context, base, dimension);
                Some(
                    (0..pairs)
                        .map(|pair| {
                            let ramp = ((f64::from(pair) - low) / (high - low)).clamp(0.0, 1.0);
                            let extrapolation_share =
                                (1.0 - ramp) * f64::from(extrapolation_factor);
                            let extrapolated = 1.0 / position_frequency(pair);
                            let interpolated = 1.0 / (f64::from(factor) * position_frequency(pair));
                            round_to_f32(
                                interpolated * (1.0 - extrapolation_share)
                                    + extrapolated * extrapolation_share,
                            )
                        })
                        .collect(),
                )
            }
        }
    }
}

/// `find_correction_range` of `_compute_yarn_parameters` with its default
/// `truncate=True`: the pair indices where the blend starts and ends,
/// clamped to `0..=dim-1`, with the same `low == high` widening so the ramp
/// never divides by zero.
#[cfg(feature = "std")]
fn yarn_correction_range(
    beta_fast: f32,
    beta_slow: f32,
    original_context: u32,
    base: f64,
    dimension: f64,
) -> (f64, f64) {
    let correction_dimension = |rotations: f32| {
        dimension
            * (f64::from(original_context) / (f64::from(rotations) * 2.0 * core::f64::consts::PI))
                .ln()
            / (2.0 * base.ln())
    };
    let low = correction_dimension(beta_fast).floor().max(0.0);
    let high = correction_dimension(beta_slow).ceil().min(dimension - 1.0);
    if (high - low).abs() <= f64::EPSILON {
        (low, high + 0.001)
    } else {
        (low, high)
    }
}

#[cfg(feature = "std")]
fn optional_f32(parsed: &ParsedGguf, key: &str) -> Option<f32> {
    match parsed.metadata_value(key) {
        Some(MetadataValue::F32(value)) => Some(*value),
        Some(MetadataValue::F64(value)) => Some(round_to_f32(*value)),
        _ => None,
    }
}

#[cfg(feature = "std")]
fn required_f32(parsed: &ParsedGguf, key: &str) -> Result<f32, InteropError> {
    optional_f32(parsed, key).ok_or_else(|| InteropError::MissingMetadataKey { key: key.into() })
}

#[cfg(all(test, feature = "std"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use proxima_gguf::value::MetadataValue;

    use super::RopeScaling;
    use crate::error::InteropError;
    use crate::serving::{ContextLength, ServingConfig, resolve_context_length};
    use crate::test_support::parsed_header;

    const QWEN3_TRAINED_CONTEXT: u32 = 40_960;

    fn qwen3_header(scaling_type: &str) -> proxima_gguf::pipe::ParsedGguf {
        parsed_header(vec![
            (
                "general.architecture",
                MetadataValue::String("qwen3".to_string()),
            ),
            (
                "qwen3.context_length",
                MetadataValue::U32(QWEN3_TRAINED_CONTEXT),
            ),
            ("qwen3.rope.freq_base", MetadataValue::F32(1_000_000.0)),
            (
                "qwen3.rope.scaling.type",
                MetadataValue::String(scaling_type.to_string()),
            ),
            ("qwen3.rope.scaling.factor", MetadataValue::F32(4.0)),
            (
                "qwen3.rope.scaling.original_context_length",
                MetadataValue::U32(32_768),
            ),
        ])
    }

    #[proxima::test]
    #[case::none("none", Some(RopeScaling::None))]
    #[case::linear("linear", Some(RopeScaling::Linear { factor: 4.0 }))]
    #[case::yarn("yarn", Some(RopeScaling::yarn(4.0, 32_768)))]
    #[case::unknown_type_rejected("longrope", None)]
    async fn rope_scaling_from_gguf(
        #[case] scaling_type: &'static str,
        #[case] expected: Option<RopeScaling>,
    ) {
        let parsed = qwen3_header(scaling_type);

        let scaling = RopeScaling::from_gguf(&parsed);

        match expected {
            Some(expected_scaling) => assert_eq!(
                scaling.expect("a recognized scaling type must parse"),
                expected_scaling
            ),
            None => assert!(matches!(
                scaling,
                Err(InteropError::UnknownRopeScalingType { ref found }) if found == scaling_type
            )),
        }
    }

    #[test]
    fn rope_scaling_override_replaces_the_gguf_value() {
        let gguf_scaling =
            RopeScaling::from_gguf(&qwen3_header("yarn")).expect("yarn header must parse");
        let linear = RopeScaling::Linear { factor: 2.0 };
        let overridden = ServingConfig {
            rope_scaling: Some(linear),
            ..ServingConfig::default()
        };
        let inherited = ServingConfig::default();

        let effective_overridden = overridden.rope_scaling.unwrap_or(gguf_scaling);
        let effective_inherited = inherited.rope_scaling.unwrap_or(gguf_scaling);

        assert_eq!(effective_overridden, linear);
        assert_eq!(effective_inherited, gguf_scaling);
        assert_eq!(
            resolve_context_length(
                ContextLength::Native,
                QWEN3_TRAINED_CONTEXT,
                effective_overridden
            )
            .expect("no explicit request always resolves to the limit"),
            81_920,
            "the override, not the gguf's yarn, must set the limit (40960 x 2)"
        );
    }

    const QWEN3_ORIGINAL_CONTEXT: u32 = 32_768;
    const QWEN3_HEAD_DIM: u32 = 128;
    const QWEN3_ROPE_BASE: f32 = 1_000_000.0;

    fn assert_relative(actual: f32, expected: f64, tolerance: f64, label: &str) {
        let relative = (f64::from(actual) - expected).abs() / expected.abs();
        assert!(
            relative <= tolerance,
            "{label}: got {actual}, expected {expected}, relative error {relative}"
        );
    }

    /// `yarn-worked-example.md`'s walk for Qwen3-8B's recommended `YaRN`
    /// (base 1e6, dim 128, factor 4, original 32768): pair 0 and 20 are
    /// pure extrapolation, 30 sits inside the ramp (7/17), 40 and 63 are
    /// pure interpolation. Expected values are `bc -l` at scale 22 (the
    /// worked example's 12-decimal figure for pair 63 has only six
    /// significant digits, so it cannot itself resolve 1e-6 relative).
    #[test]
    fn yarn_inv_freq_matches_the_worked_example_pairs() {
        let scaling = RopeScaling::yarn(4.0, QWEN3_ORIGINAL_CONTEXT);
        let expected = [
            (0usize, 1.0_f64),
            (20, 0.013_335_214_321_633_24),
            (30, 0.001_064_360_981_247_001_8),
            (40, 0.000_044_456_985_250_973_07),
            (63, 0.000_000_310_234_440_187_92),
        ];

        let frequencies = scaling
            .inv_frequencies(QWEN3_HEAD_DIM, QWEN3_ROPE_BASE)
            .expect("yarn always yields a table");

        assert_eq!(frequencies.len(), 64);
        for (pair, value) in expected {
            assert_relative(frequencies[pair], value, 1e-6, &format!("pair {pair}"));
        }
    }

    /// `beta_fast == beta_slow == 6000` puts both correction dimensions at
    /// -0.65, so `low` clamps up to 0.0 and `high = ceil(-0.65) = -0.0`:
    /// `low == high` and, without the `+ 0.001` guard, pair 0's ramp is
    /// `0.0 / -0.0 = NaN`. (The example's "same integer" construction does
    /// not occur for a non-integer correction dimension: `floor` and `ceil`
    /// differ by one.)
    #[test]
    fn yarn_inv_freq_low_equals_high_guard_keeps_the_ramp_finite() {
        let scaling = RopeScaling::Yarn {
            factor: 4.0,
            original_context: QWEN3_ORIGINAL_CONTEXT,
            extrapolation_factor: 1.0,
            attention_factor: 1.0,
            beta_fast: 6000.0,
            beta_slow: 6000.0,
        };

        let frequencies = scaling
            .inv_frequencies(QWEN3_HEAD_DIM, QWEN3_ROPE_BASE)
            .expect("yarn always yields a table");

        assert!(frequencies.iter().all(|value| value.is_finite()));
        assert_relative(frequencies[0], 1.0, 1e-6, "pair 0 stays extrapolated");
        let interpolated_pair_one = 1_000_000f64.powf(-2.0 / 128.0) / 4.0;
        assert_relative(
            frequencies[1],
            interpolated_pair_one,
            1e-6,
            "pair 1 is fully interpolated",
        );
    }

    #[test]
    fn yarn_attention_factor_is_point_one_ln_factor_plus_one() {
        let scaling = RopeScaling::yarn(4.0, QWEN3_ORIGINAL_CONTEXT);

        assert_relative(
            scaling.attention_factor(),
            1.138_629_436_111_989,
            1e-6,
            "0.1 * ln(4) + 1",
        );
    }
}
