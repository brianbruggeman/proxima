//! Profile facts a checkpoint states itself, through which GGUF metadata keys
//! exist and what they hold, so a never-tried model does not need a per-family
//! TOML row to say them (PROXNUM-424 design section 3, card 3). Sibling of
//! [`crate::ffn_inference`], which reads the tensor directory instead.
//!
//! Each rule reads `parsed`'s metadata only, never the architecture name, and
//! emits one structured `debug!` event with its key evidence:
//! - [`infer_routed_gating`]: `{arch}.expert_gating_func` (llama.cpp
//!   `llama_expert_gating_func_type`, `llama-hparams.h:17-23`: 1 softmax, 2
//!   sigmoid); absent or `0` states nothing, so the profile's default stands
//!   (llama.cpp hardcodes softmax per arch graph, `models/llama.cpp:212`)
//! - [`infer_schedule_source`]: which header reader spells this layer schedule,
//!   decided from key shape before any reader runs: a
//!   `{arch}.attention.sliding_window_pattern` array is
//!   [`ScheduleSource::SlidingPattern`]; else a `{arch}.full_attention_interval`
//!   is [`ScheduleSource::RecurrentInterval`], or
//!   [`ScheduleSource::RecurrentRoutedInterval`] when `{arch}.expert_count` is
//!   nonzero; else [`ScheduleSource::Uniform`] (scalar keys, short-conv layers
//!   marked by zeros in `head_count_kv`). Vestigial keys such as granite-4.0's
//!   `granite.ssm.*` are not consulted.
//! - [`infer_score_scale`]: `{arch}.attention.scale` when present and nonzero,
//!   else the `1/sqrt(head_dim)` default (`models/llama.cpp:122`,
//!   `models/granite.cpp:235`). The default is not an overlay: an architecture
//!   whose graph hardcodes another scale (gemma4, `models/gemma4.cpp:11`) states
//!   that in its profile, and an absent key leaves the profile alone.
//!
//! [`with_inferred_keys`] is the wiring point that overlays the two facts a
//! [`FamilyProfile`] can carry; [`infer_score_scale`] feeds
//! `crate::dense`'s header scales, since a profile holds only a bool for it.

use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::value::MetadataValue;
use proxima_telemetry::debug;
use proxima_tensor::spec::{ExpertGatingFunc, FamilyProfile, ScheduleSource, expert_gating_func};

use crate::bind::{metadata_f32_optional, metadata_u32_optional};
use crate::error::InteropError;

/// The attention score scale the header states, or the documented default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum InferredScoreScale {
    /// `{arch}.attention.scale`, present and nonzero.
    Factor(f32),
    /// Key absent (or zero): `1/sqrt(head_dim)`.
    InverseSqrtHeadDim,
}

/// The header's `expert_gating_func`, `None` when absent or `0`.
///
/// # Errors
///
/// [`InteropError::Tensor`] wrapping `GgufUnsupportedExpertGating` for values
/// this engine has no routing function for (`3`, `4`): serving the profile's
/// function instead would route with different math than the checkpoint's.
pub(crate) fn infer_routed_gating(
    parsed: &ParsedGguf,
    family: &str,
) -> Result<Option<ExpertGatingFunc>, InteropError> {
    let key = format!("{family}.expert_gating_func");
    let value = metadata_u32_optional(parsed, &key);
    let gating = expert_gating_func(value)?;
    debug!(
        field = "routed_gating",
        key = ?key,
        present = parsed.metadata_value(&key).is_some(),
        raw = value,
        value = ?gating,
        "key fact inferred from metadata"
    );
    Ok(gating)
}

/// The header reader this header's key shape selects.
#[must_use]
pub(crate) fn infer_schedule_source(parsed: &ParsedGguf, family: &str) -> ScheduleSource {
    let pattern_key = format!("{family}.attention.sliding_window_pattern");
    let interval_key = format!("{family}.full_attention_interval");
    let sliding = matches!(
        parsed.metadata_value(&pattern_key),
        Some(MetadataValue::Array(_))
    );
    let interval = parsed.metadata_value(&interval_key).is_some();
    let expert_count = metadata_u32_optional(parsed, &format!("{family}.expert_count"));
    let source = if sliding {
        ScheduleSource::SlidingPattern
    } else if interval && expert_count > 0 {
        ScheduleSource::RecurrentRoutedInterval
    } else if interval {
        ScheduleSource::RecurrentInterval
    } else {
        ScheduleSource::Uniform
    };
    debug!(
        field = "schedule_source",
        value = ?source,
        sliding_window_pattern_array = sliding,
        full_attention_interval = interval,
        expert_count = expert_count,
        "key fact inferred from metadata"
    );
    source
}

/// `{family}.attention.scale` when the header states one (zero reads as
/// absent, the existing `crate::dense` header-scale convention).
#[must_use]
pub(crate) fn infer_score_scale(parsed: &ParsedGguf, family: &str) -> InferredScoreScale {
    let key = format!("{family}.attention.scale");
    let value = metadata_f32_optional(parsed, &key, 0.0);
    let scale = if value == 0.0 {
        InferredScoreScale::InverseSqrtHeadDim
    } else {
        InferredScoreScale::Factor(value)
    };
    debug!(
        field = "score_scale",
        key = ?key,
        value = ?scale,
        "key fact inferred from metadata, default when key absent"
    );
    scale
}

/// `profile` with `routed_gating` replaced when the header states
/// `expert_gating_func`, and `schedule_source` replaced by the key-shape rule.
///
/// Composes [`infer_routed_gating`] and [`infer_schedule_source`].
///
/// # Errors
///
/// Whatever [`infer_routed_gating`] refuses.
pub(crate) fn with_inferred_keys(
    profile: FamilyProfile,
    parsed: &ParsedGguf,
    family: &str,
) -> Result<FamilyProfile, InteropError> {
    let routed_gating = infer_routed_gating(parsed, family)?.unwrap_or(profile.ffn.routed_gating);
    let schedule_source = infer_schedule_source(parsed, family);
    Ok(FamilyProfile {
        ffn: proxima_tensor::spec::LayerFfnConfig {
            routed_gating,
            ..profile.ffn
        },
        schedule_source,
        ..profile
    })
}

#[cfg(test)]
mod tests {
    use proxima_gguf::value::MetadataArray;

    use super::*;
    use crate::profiles::family_profile;

    const FIXTURES: [(&str, Option<&str>, u32, &str); 8] = [
        ("gemma4", Some("gemma4"), 0, include_str!("../tests/fixtures/header_keys/gemma4.txt")),
        (
            "granite3.1-moe",
            Some("granitemoe"),
            32,
            include_str!("../tests/fixtures/header_keys/granitemoe.txt"),
        ),
        ("flint", Some("flint"), 256, include_str!("../tests/fixtures/header_keys/flint.txt")),
        ("Llama-3.2", Some("llama"), 0, include_str!("../tests/fixtures/header_keys/llama.txt")),
        ("granite-4.0", None, 0, include_str!("../tests/fixtures/header_keys/granite4.txt")),
        ("LFM2", Some("lfm2"), 0, include_str!("../tests/fixtures/header_keys/lfm2.txt")),
        ("qwen2.5", Some("qwen2"), 0, include_str!("../tests/fixtures/header_keys/qwen25.txt")),
        ("qwen3", Some("qwen3"), 0, include_str!("../tests/fixtures/header_keys/qwen3.txt")),
    ];

    fn payload<'text>(value: &'text str, wrapper: &str) -> &'text str {
        value
            .strip_prefix(wrapper)
            .and_then(|rest| rest.strip_suffix(')'))
            .expect("fixture value has the wrapper it names")
    }

    fn list<'text>(value: &'text str, wrapper: &str) -> impl Iterator<Item = &'text str> {
        payload(value, wrapper)
            .trim_matches(|ch| ch == '[' || ch == ']')
            .split(", ")
    }

    fn metadata_value(value: &str) -> MetadataValue {
        if let Some(rest) = value.strip_prefix("String(") {
            let text = rest.strip_suffix(')').expect("string closes");
            MetadataValue::String(text.trim_matches('"').into())
        } else if value.starts_with("U32(") {
            MetadataValue::U32(payload(value, "U32(").parse().expect("u32 value"))
        } else if value.starts_with("F32(") {
            MetadataValue::F32(payload(value, "F32(").parse().expect("f32 value"))
        } else if value.starts_with("Array(Bool(") {
            let inner = payload(value, "Array(");
            let flags = list(inner, "Bool(").map(|flag| flag == "true").collect();
            MetadataValue::Array(MetadataArray::Bool(flags))
        } else if value.starts_with("Array(I32(") {
            let inner = payload(value, "Array(");
            let counts = list(inner, "I32(").map(|count| count.parse().expect("i32 entry")).collect();
            MetadataValue::Array(MetadataArray::I32(counts))
        } else {
            panic!("fixture value {value:.40} has no parser")
        }
    }

    fn header(text: &str) -> ParsedGguf {
        let metadata: Vec<(&str, MetadataValue)> = text
            .lines()
            .map(|line| {
                let (key, value) = line.split_once('=').expect("fixture line is key=value");
                (key, metadata_value(value))
            })
            .collect();
        crate::test_support::parsed_header_with_extra_tensors(metadata, 4, 4, &[])
    }

    fn family_of(text: &str) -> String {
        let line = text.lines().next().expect("fixture has an architecture line");
        line.strip_prefix("general.architecture=String(\"")
            .and_then(|rest| rest.strip_suffix("\")"))
            .expect("first line names the architecture")
            .to_string()
    }

    #[test]
    fn fixtures_carry_the_keys_captured_from_each_headers_dump() {
        let lines: Vec<usize> = FIXTURES.iter().map(|row| row.3.lines().count()).collect();

        assert_eq!(lines, [3, 4, 5, 2, 4, 2, 2, 2]);
    }

    #[test]
    fn inferred_key_facts_equal_the_profile_for_each_local_header() {
        let mut table = Vec::new();
        let mut disagreements = Vec::new();
        let mut compared = 0;
        for (model, profiled, expert_count, text) in FIXTURES {
            let parsed = header(text);
            let family = family_of(text);
            let schedule = infer_schedule_source(&parsed, &family);
            let scale = infer_score_scale(&parsed, &family);
            let gating = infer_routed_gating(&parsed, &family).expect("local headers state supported gating");
            let (expected_schedule, profile_scale_inverse_sqrt, profile_gating) = match profiled {
                Some(name) => {
                    let profile = family_profile(name).expect("profile embedded");
                    (
                        profile.schedule_source,
                        Some(profile.score_scale_inverse_sqrt_head_dim),
                        Some(profile.ffn.routed_gating),
                    )
                }
                None => (ScheduleSource::Uniform, None, None),
            };
            compared += 1;
            if schedule != expected_schedule {
                disagreements.push(format!("{model}.schedule_source"));
            }
            match (scale, profile_scale_inverse_sqrt) {
                (InferredScoreScale::InverseSqrtHeadDim, Some(false)) => {
                    disagreements.push(format!("{model}.score_scale"));
                }
                (InferredScoreScale::InverseSqrtHeadDim, _) | (InferredScoreScale::Factor(_), _) => {}
            }
            compared += 1;
            if expert_count > 0 {
                let profile_value = profile_gating.expect("expert headers are profiled");
                if gating.unwrap_or(ExpertGatingFunc::Softmax) != profile_value {
                    disagreements.push(format!("{model}.routed_gating"));
                }
                compared += 1;
            }
            table.push(format!(
                "{model:16} schedule={schedule:?}/{expected_schedule:?} scale={scale:?}/profile_inverse_sqrt={profile_scale_inverse_sqrt:?} gating={gating:?}/{profile_gating:?}"
            ));
        }
        println!("{}", table.join("\n"));

        assert_eq!(
            disagreements,
            ["gemma4.score_scale"],
            "gemma4's header states no attention.scale so the 1/sqrt default disagrees with its unscaled profile; \
             llama.cpp models/gemma4.cpp:11 sets f_attention_scale = 1.0, so the profile is right"
        );
        assert_eq!(compared, 8 + 8 + 2, "schedule and scale on 8 headers, gating on 2 expert headers");
    }

    #[test]
    fn granite_4_0_states_its_scale_and_a_uniform_schedule_despite_vestigial_ssm_keys() {
        let text = FIXTURES[4].3;
        let parsed = header(text);

        assert_eq!(infer_score_scale(&parsed, "granite"), InferredScoreScale::Factor(0.015_625));
        assert_eq!(infer_schedule_source(&parsed, "granite"), ScheduleSource::Uniform);
    }

    #[test]
    fn a_short_conv_header_with_zeros_in_head_count_kv_stays_uniform() {
        let parsed = header(FIXTURES[5].3);

        assert_eq!(infer_schedule_source(&parsed, "lfm2"), ScheduleSource::Uniform);
    }

    #[test]
    fn a_full_attention_interval_selects_the_recurrent_reader_by_expert_count() {
        let dense = crate::test_support::parsed_header_with_extra_tensors(
            vec![("qwen35.full_attention_interval", MetadataValue::U32(4))],
            4,
            4,
            &[],
        );
        let routed = crate::test_support::parsed_header_with_extra_tensors(
            vec![
                ("qwen35moe.full_attention_interval", MetadataValue::U32(4)),
                ("qwen35moe.expert_count", MetadataValue::U32(256)),
            ],
            4,
            4,
            &[],
        );

        assert_eq!(infer_schedule_source(&dense, "qwen35"), ScheduleSource::RecurrentInterval);
        assert_eq!(
            infer_schedule_source(&routed, "qwen35moe"),
            ScheduleSource::RecurrentRoutedInterval
        );
    }

    #[test]
    fn an_unsupported_gating_value_is_refused_not_defaulted() {
        let parsed = crate::test_support::parsed_header_with_extra_tensors(
            vec![("phi9.expert_gating_func", MetadataValue::U32(4))],
            4,
            4,
            &[],
        );

        let refusal = infer_routed_gating(&parsed, "phi9");

        assert!(matches!(refusal, Err(InteropError::Tensor(_))), "{refusal:?}");
    }

    #[test]
    fn an_absent_gating_key_keeps_the_profile_function_and_a_stated_one_replaces_it() {
        let flint = header(FIXTURES[2].3);
        let granitemoe = header(FIXTURES[1].3);
        let softmax_profile = family_profile("granitemoe").expect("profile embedded");

        let kept = with_inferred_keys(softmax_profile, &granitemoe, "granitemoe").expect("overlay");
        let replaced = with_inferred_keys(softmax_profile, &flint, "flint").expect("overlay");

        assert_eq!(kept.ffn.routed_gating, ExpertGatingFunc::Softmax);
        assert_eq!(replaced.ffn.routed_gating, ExpertGatingFunc::Sigmoid);
        assert_eq!(replaced.schedule_source, ScheduleSource::SlidingPattern);
    }
}
