//! Family profiles: the values a checkpoint's GGUF header and HF `config.json`
//! do not carry, one TOML file per family, embedded with `include_str!` so no
//! runtime file IO is needed.
//!
//! This directory is the only place a family name appears. The loader is keyed
//! by the string the checkpoint itself declares (`general.architecture` on the
//! GGUF path, `model_type` on the HF path), so both paths read the same file.
//! A family with no file is an error naming it: [`family_profile`] never falls
//! back to another family's values.
//!
//! The parsed value is a [`proxima_tensor::spec::FamilyProfile`], built from
//! the descriptor's own enums; `proxima_tensor::spec::sliding_pattern_descriptor_from_gguf`
//! and `proxima_tensor::spec::gqa_descriptor_from_shape` consume it as
//! plain fields. TOML parsing stays here because `proxima-tensor`'s alloc tier
//! carries no parser.

use proxima_tensor::spec::FamilyProfile;

use crate::error::InteropError;

mod binding;

pub use binding::{BindingProfile, TensorAlias, binding_profile};

const FAMILY_PROFILES: &[(&str, &str)] = &[
    ("gemma4", include_str!("gemma4.toml")),
    ("llama", include_str!("llama.toml")),
    ("mistral", include_str!("mistral.toml")),
    ("mixtral", include_str!("mixtral.toml")),
    ("granitemoe", include_str!("granitemoe.toml")),
    ("qwen2", include_str!("qwen2.toml")),
    ("qwen3", include_str!("qwen3.toml")),
    ("qwen3moe", include_str!("qwen3moe.toml")),
    ("qwen3_moe", include_str!("qwen3moe.toml")),
    ("qwen35", include_str!("qwen35.toml")),
    ("qwen35moe", include_str!("qwen35moe.toml")),
    ("lfm2", include_str!("lfm2moe.toml")),
    ("lfm2moe", include_str!("lfm2moe.toml")),
];

/// The [`FamilyProfile`] for `family`: a checkpoint's `general.architecture`
/// (GGUF) or `model_type` (HF `config.json`).
///
/// # Errors
///
/// [`InteropError::MissingFamilyProfile`] when no profile is embedded for
/// `family`; [`InteropError::InvalidFamilyProfile`] when the embedded file does
/// not parse.
pub fn family_profile(family: &str) -> Result<FamilyProfile, InteropError> {
    let (_, text) = FAMILY_PROFILES
        .iter()
        .find(|(name, _)| *name == family)
        .ok_or_else(|| InteropError::MissingFamilyProfile {
            family: family.into(),
        })?;
    toml::from_str(text).map_err(|error| InteropError::InvalidFamilyProfile {
        family: family.into(),
        message: error.to_string(),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_tensor::spec::{
        Activation, AttentionScoreScale, EmbeddingScale, ExpertGatingFunc, FfnCombination, FfnRouting,
        KvCacheShape, RopePairing, ScheduleSource,
    };

    #[test]
    fn every_embedded_profile_parses() {
        assert_eq!(FAMILY_PROFILES.len(), 13, "one row per embedded family string");
        for (family, _) in FAMILY_PROFILES {
            family_profile(family).unwrap_or_else(|error| panic!("{family}: {error}"));
        }
    }

    #[test]
    fn gemma4_profile_carries_the_values_gguf_does_not() {
        let profile = family_profile("gemma4").expect("gemma4 profile embedded");

        assert_eq!(profile.embedding_scale, Some(EmbeddingScale::Sqrt));
        assert_eq!(profile.ffn.activation, Activation::GeluTanh);
        assert_eq!(profile.score_scale(256), AttentionScoreScale::Unscaled);
        assert_eq!(profile.rope_pairing(256), RopePairing::SplitHalf { pairs: 128 });
        assert!(profile.value_norm);
        assert!(matches!(
            profile.layer_ffn(128).combination,
            FfnCombination::ParallelDenseMoe(_)
        ));
        assert_eq!(profile.layer_ffn(0).combination, FfnCombination::Exclusive);
    }

    #[test]
    fn dense_profile_scales_scores_and_pairs_rope_adjacent() {
        let profile = family_profile("llama").expect("llama profile embedded");

        assert_eq!(profile.embedding_scale, None);
        assert_eq!(profile.ffn.activation, Activation::Silu);
        assert_eq!(profile.score_scale(128), AttentionScoreScale::InverseSqrtQueryPreAttnScalar(128));
        assert_eq!(profile.rope_pairing(128), RopePairing::Interleaved);
    }

    #[test]
    fn granitemoe_profile_is_adjacent_rope_softmax_routed_and_silu() {
        let profile = family_profile("granitemoe").expect("granitemoe profile embedded");

        assert_eq!(profile.rope_pairing(64), RopePairing::Interleaved);
        assert_eq!(profile.ffn.routed_gating, ExpertGatingFunc::Softmax);
        assert_eq!(profile.ffn.activation, Activation::Silu);
        assert_eq!(profile.ffn.combination, FfnCombination::Exclusive);
        assert_eq!(profile.embedding_scale, None);
    }

    #[test]
    fn only_the_family_whose_verify_step_was_measured_to_pay_arms_it_by_default() {
        for (family, _) in FAMILY_PROFILES {
            let armed = family_profile(family).expect("profile embedded").speculative_verify;

            assert_eq!(armed, *family == "gemma4", "{family}");
        }
    }

    #[test]
    fn each_profile_names_its_header_reader_cache_shape_routing_and_chunk_default() {
        let table = [
            ("llama", ScheduleSource::Uniform, KvCacheShape::Uniform, FfnRouting::Dense, 1),
            ("mixtral", ScheduleSource::Uniform, KvCacheShape::Uniform, FfnRouting::Dense, 1),
            ("granitemoe", ScheduleSource::Uniform, KvCacheShape::Uniform, FfnRouting::Dense, 8),
            ("qwen2", ScheduleSource::Uniform, KvCacheShape::Uniform, FfnRouting::Dense, 1),
            ("qwen3", ScheduleSource::Uniform, KvCacheShape::Uniform, FfnRouting::Dense, 1),
            ("lfm2moe", ScheduleSource::Uniform, KvCacheShape::Uniform, FfnRouting::Dense, 1),
            ("gemma4", ScheduleSource::SlidingPattern, KvCacheShape::Custom, FfnRouting::Dense, 8),
            ("qwen35", ScheduleSource::RecurrentInterval, KvCacheShape::Custom, FfnRouting::Dense, 1),
            (
                "qwen35moe",
                ScheduleSource::RecurrentRoutedInterval,
                KvCacheShape::Monolithic,
                FfnRouting::Routed,
                1,
            ),
        ];
        for (family, source, shape, routing, chunks) in table {
            let profile = family_profile(family).expect("profile embedded");

            assert_eq!(profile.schedule_source, source, "{family} schedule source");
            assert_eq!(profile.kv_cache_shape, shape, "{family} cache shape");
            assert_eq!(profile.ffn_routing, routing, "{family} routing");
            assert_eq!(profile.command_buffer_chunks, chunks, "{family} chunk default");
        }
    }

    #[test]
    fn qwen2_profile_is_split_half_from_its_own_data() {
        let profile = family_profile("qwen2").expect("qwen2 profile embedded");

        assert_eq!(profile.rope_pairing(128), RopePairing::SplitHalf { pairs: 64 });
    }

    /// llama.cpp `llama_model_rope_type` (`src/llama-model.cpp:2968-3140`, f1ea20621):
    /// LLAMA/MISTRAL3 are NORM (adjacent); QWEN2, QWEN3, QWEN3MOE, GEMMA4, LFM2, LFM2MOE are NEOX
    /// (split-half). QWEN35/QWEN35MOE lower through the recurrent-hybrid engine, which rotates
    /// split-half over the partial rotary width.
    /// mixtral GGUFs declare `general.architecture = llama`.
    /// GRANITE_MOE is NORM (`src/llama-model.cpp:3019`, the group returns at :3039).
    #[test]
    fn every_profile_pairs_rope_as_llama_cpp_rope_type_says() {
        let table = [
            ("llama", false),
            ("mistral", false),
            ("mixtral", false),
            ("granitemoe", false),
            ("qwen2", true),
            ("qwen3", true),
            ("qwen3moe", true),
            ("qwen3_moe", true),
            ("gemma4", true),
            ("qwen35", true),
            ("qwen35moe", true),
            ("lfm2", true),
            ("lfm2moe", true),
        ];
        assert_eq!(table.len(), FAMILY_PROFILES.len(), "one expectation per embedded family string");
        for (family, split_half) in table {
            let expected = if split_half {
                RopePairing::SplitHalf { pairs: 64 }
            } else {
                RopePairing::Interleaved
            };
            let profile = family_profile(family).expect("profile embedded");
            assert_eq!(profile.rope_pairing(128), expected, "{family}");
        }
    }

    fn fixture_kv(fixture: &str) -> Vec<(String, String)> {
        let path = format!("{}/tests/fixtures/llama-parity/{fixture}/gguf_kv.txt", env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{path}: {error}"));
        let entries: Vec<(String, String)> = text
            .lines()
            .filter_map(|line| line.splitn(3, " | ").nth(2))
            .filter_map(|entry| entry.split_once(" = "))
            .map(|(key, value)| (key.trim().to_string(), value.trim().trim_matches('\'').to_string()))
            .collect();
        assert!(!entries.is_empty(), "{path} yielded no key/value rows");
        entries
    }

    fn kv_value(kv: &[(String, String)], key: &str) -> Option<String> {
        kv.iter().find(|(name, _)| name == key).map(|(_, value)| value.clone())
    }

    #[test]
    fn real_header_metadata_selects_the_pairing_and_the_rotating_width() {
        let cases = [
            ("gemma4_e2b", RopePairing::SplitHalf { pairs: 256 }),
            ("gemma4_26b", RopePairing::SplitHalf { pairs: 256 }),
            ("openchat", RopePairing::Interleaved),
            ("qwen2", RopePairing::SplitHalf { pairs: 64 }),
            ("qwen3", RopePairing::SplitHalf { pairs: 64 }),
            ("granite_moe", RopePairing::Interleaved),
        ];
        for (fixture, expected) in cases {
            let kv = fixture_kv(fixture);
            let family = kv_value(&kv, "general.architecture").expect("header names its architecture");
            let width = kv_value(&kv, &format!("{family}.rope.dimension_count"))
                .or_else(|| kv_value(&kv, &format!("{family}.attention.key_length")))
                .map_or(128, |text| text.parse::<u32>().expect("integer width"));
            let profile = family_profile(&family).expect("profile embedded for the header's family");
            assert_eq!(profile.rope_pairing(width), expected, "{fixture} ({family})");
        }
    }

    #[test]
    fn qwen35_headers_carry_the_partial_rotary_width_the_profile_pairs_split_half() {
        for fixture in ["qwen35", "qwen35moe"] {
            let kv = fixture_kv(fixture);
            let family = kv_value(&kv, "general.architecture").expect("header names its architecture");

            assert_eq!(kv_value(&kv, &format!("{family}.rope.dimension_count")).as_deref(), Some("64"), "{fixture}");
            assert_eq!(kv_value(&kv, &format!("{family}.attention.key_length")).as_deref(), Some("256"), "{fixture}");
            assert_eq!(
                kv_value(&kv, &format!("{family}.rope.dimension_sections")).as_deref(),
                Some("[11, 11, 10]"),
                "{fixture}"
            );
            let profile = family_profile(&family).expect("the recurrent-hybrid families have a profile");
            assert_eq!(profile.rope_pairing(64), RopePairing::SplitHalf { pairs: 32 }, "{fixture}");
        }
    }

    #[test]
    fn the_shared_expert_family_names_its_ffn_and_the_dense_hybrid_keeps_the_plain_one() {
        let routed = family_profile("qwen35moe").expect("qwen35moe profile embedded");
        let dense = family_profile("qwen35").expect("qwen35 profile embedded");

        assert_eq!(routed.layer_ffn(256).combination, FfnCombination::RoutedWithSharedExpert);
        assert_eq!(dense.layer_ffn(0).combination, FfnCombination::Exclusive);
    }

    #[test]
    fn lfm2_profile_routes_with_sigmoid_gating_and_a_selection_bias() {
        let kv = fixture_kv("lfm2");
        let family = kv_value(&kv, "general.architecture").expect("header names its architecture");
        assert_eq!(kv_value(&kv, &format!("{family}.expert_gating_func")).as_deref(), Some("2"));
        let profile = family_profile(&family).expect("lfm2moe profile embedded");

        assert_eq!(profile.ffn.routed_gating, ExpertGatingFunc::Sigmoid);
        assert!(profile.ffn.routed_expert_bias);
        assert_eq!(profile.ffn.combination, FfnCombination::Exclusive);
    }

    #[test]
    fn a_family_with_no_profile_is_an_error_naming_it() {
        let error = family_profile("phi9").expect_err("phi9 has no embedded profile");

        assert!(matches!(
            &error,
            InteropError::MissingFamilyProfile { family } if family == "phi9"
        ));
        assert!(error.to_string().contains("phi9"));
    }

    #[test]
    fn hf_and_gguf_spellings_of_one_family_read_the_same_profile() {
        assert_eq!(
            family_profile("qwen3_moe").expect("hf spelling"),
            family_profile("qwen3moe").expect("gguf spelling")
        );
    }
}
