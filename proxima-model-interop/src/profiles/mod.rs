//! Family profiles: the values a checkpoint's GGUF header and HF `config.json`
//! do not carry, one TOML file per family, embedded with `include_str!` so no
//! runtime file IO is needed.
//!
//! This directory is the only place a family name is DATA: the values a
//! family's checkpoint does not state, keyed by the string the checkpoint itself
//! declares (`general.architecture` on the GGUF path, `model_type` on the HF
//! path), so both paths read the same file. A family with no file is an error
//! naming it: [`family_profile`] never falls back to another family's values.
//!
//! The parsed value is a [`proxima_tensor::spec::FamilyProfile`], built from
//! the descriptor's own enums; `proxima_tensor::spec::sliding_pattern_descriptor_from_gguf`
//! and `proxima_tensor::spec::gqa_descriptor_from_shape` consume it as
//! plain fields. TOML parsing stays here because `proxima-tensor`'s alloc tier
//! carries no parser.

use proxima_tensor::spec::FamilyProfile;

use crate::error::InteropError;

mod binding;
mod unknown;

pub use unknown::ArchConfigSettings;
use unknown::unknown_family_profile;

#[cfg(test)]
mod scan;

pub use binding::{BindingProfile, TensorAlias, binding_profile};

pub(crate) const FAMILY_PROFILES: &[(&str, &str)] = &[
    ("gemma4", include_str!("gemma4.toml")),
    ("flint", include_str!("flint.toml")),
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
    ("ouro", include_str!("ouro.toml")),
];

/// The [`FamilyProfile`] for `family`: a checkpoint's `general.architecture`
/// (GGUF) or `model_type` (HF `config.json`).
///
/// A family with no embedded row is built from the dense defaults and the
/// run-time arch config file it names ([`ArchConfigSettings`]).
///
/// # Errors
///
/// [`InteropError::MissingArchField`] when an unlisted family's arch config is
/// absent or lacks required fields; [`InteropError::InvalidFamilyProfile`] when
/// a file does not parse.
pub fn family_profile(family: &str) -> Result<FamilyProfile, InteropError> {
    let Some((_, text)) = FAMILY_PROFILES.iter().find(|(name, _)| *name == family) else {
        return unknown_family_profile(family);
    };
    conflaguration::from_toml_str(text).map_err(|error| InteropError::InvalidFamilyProfile {
        family: family.into(),
        message: error.to_string(),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::path::Path;
    use proxima_tensor::spec::{
        Activation, AttentionScoreScale, EmbeddingScale, ExpertGatingFunc, FfnCombination,
        KvCacheShape, RopePairing, ScheduleSource,
    };

    #[test]
    fn every_embedded_profile_parses() {
        assert_eq!(
            FAMILY_PROFILES.len(),
            15,
            "one row per embedded family string"
        );
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
        assert_eq!(
            profile.rope_pairing(256),
            RopePairing::SplitHalf { pairs: 128 }
        );
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
        assert_eq!(
            profile.score_scale(128),
            AttentionScoreScale::InverseSqrtQueryPreAttnScalar(128)
        );
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

    /// The family's non-header values are data, and each one the capture itself
    /// can speak to is checked against the capture: the tensor directory states
    /// which norm and scale leaves the checkpoint carries. Provenance for the
    /// rest is the incumbent's own model file, llama.cpp `src/models/flint.cpp`
    /// at `origin/master` (fetched 2026-11-08): it applies RMSNorm to Q and K
    /// before RoPE, scales scores by `1/sqrt(n_embd_head)`, passes V through
    /// unnormalized, runs SiLU on the dense branch, the routed branch and the
    /// shared expert, and adds the shared expert unscaled.
    #[test]
    fn flint_profile_is_data_carrying_every_value_the_capture_cannot_state() {
        let profile = family_profile("flint").expect("flint profile embedded");
        let rows = crate::oracle_fixtures::parse_tensor_rows(
            &crate::oracle_fixtures::read_fixture("tensor_dir.txt"),
        );
        let carries = |suffix: &str| rows.iter().any(|row| row.name.ends_with(suffix));

        assert!(
            carries("blk.0.attn_q_norm.weight") && carries("blk.0.attn_k_norm.weight"),
            "the capture carries per-head Q and K norm weights, which is qk_norm"
        );
        assert!(
            !carries("ffn_exp_probs_b.bias"),
            "the capture carries no routing bias, so the profile must not declare one"
        );
        assert!(
            !carries("post_attention_norm.weight")
                && !carries("layer_output_scale.weight")
                && !carries("post_ffw_norm.weight"),
            "the capture carries no post-attention norm, layer scale or dense post norm"
        );

        assert!(profile.qk_norm);
        assert!(!profile.value_norm);
        assert_eq!(profile.embedding_scale, None);
        assert_eq!(
            profile.rope_pairing(128),
            RopePairing::SplitHalf { pairs: 64 }
        );
        assert_eq!(
            profile.score_scale(128),
            AttentionScoreScale::InverseSqrtQueryPreAttnScalar(128)
        );
        assert_eq!(profile.schedule_source, ScheduleSource::SlidingPattern);
        assert_eq!(profile.kv_cache_shape, KvCacheShape::Custom);
        assert_eq!(
            profile.layer_ffn(256).combination,
            FfnCombination::RoutedWithSharedExpert
        );
        assert_eq!(
            profile.layer_ffn(0).combination,
            FfnCombination::RoutedWithSharedExpert
        );
        assert!(profile.parallel_dense_moe.is_none());
        assert_eq!(profile.ffn.activation, Activation::Silu);
        assert_eq!(profile.ffn.routed_gating, ExpertGatingFunc::Sigmoid);
        assert!(!profile.ffn.routed_expert_bias);
        assert!(
            !profile.ffn.shared_expert_gate,
            "the capture carries no ffn_gate_inp_shexp.weight, so the shared expert is unscaled"
        );
        assert!(!profile.ffn.post_attention_norm);
        assert!(!profile.ffn.output_scale);
        assert!(!profile.ffn.exclusive_dense_post_norm);
        assert!(profile.recurrence.is_none());
        assert!(!profile.speculative_verify);
        assert_eq!(profile.command_buffer_chunks, 8);
    }

    #[test]
    fn only_the_family_whose_verify_step_was_measured_to_pay_arms_it_by_default() {
        for (family, _) in FAMILY_PROFILES {
            let armed = family_profile(family)
                .expect("profile embedded")
                .speculative_verify;

            assert_eq!(armed, *family == "gemma4", "{family}");
        }
    }

    #[test]
    fn each_profile_names_its_header_reader_cache_shape_and_chunk_default() {
        let table = [
            ("llama", ScheduleSource::Uniform, KvCacheShape::Uniform, 1),
            ("mixtral", ScheduleSource::Uniform, KvCacheShape::Uniform, 1),
            ("granitemoe", ScheduleSource::Uniform, KvCacheShape::Uniform, 8),
            ("qwen2", ScheduleSource::Uniform, KvCacheShape::Uniform, 1),
            ("qwen3", ScheduleSource::Uniform, KvCacheShape::Uniform, 1),
            ("lfm2moe", ScheduleSource::Uniform, KvCacheShape::Uniform, 1),
            ("gemma4", ScheduleSource::SlidingPattern, KvCacheShape::Custom, 8),
            ("flint", ScheduleSource::SlidingPattern, KvCacheShape::Custom, 8),
            ("qwen35", ScheduleSource::RecurrentInterval, KvCacheShape::Custom, 1),
            (
                "qwen35moe",
                ScheduleSource::RecurrentRoutedInterval,
                KvCacheShape::Monolithic,
                1,
            ),
        ];
        for (family, source, shape, chunks) in table {
            let profile = family_profile(family).expect("profile embedded");

            assert_eq!(profile.schedule_source, source, "{family} schedule source");
            assert_eq!(profile.kv_cache_shape, shape, "{family} cache shape");
            assert_eq!(
                profile.command_buffer_chunks, chunks,
                "{family} chunk default"
            );
        }
    }

    #[test]
    fn qwen2_profile_is_split_half_from_its_own_data() {
        let profile = family_profile("qwen2").expect("qwen2 profile embedded");

        assert_eq!(
            profile.rope_pairing(128),
            RopePairing::SplitHalf { pairs: 64 }
        );
    }

    /// llama.cpp `llama_model_rope_type` (`src/llama-model.cpp:2968-3140`, f1ea20621):
    /// LLAMA/MISTRAL3 are NORM (adjacent); QWEN2, QWEN3, QWEN3MOE, GEMMA4, LFM2, LFM2MOE are NEOX
    /// (split-half). QWEN35/QWEN35MOE lower through the recurrent-hybrid engine, which rotates
    /// split-half over the partial rotary width.
    /// mixtral GGUFs declare `general.architecture = llama`.
    /// GRANITE_MOE is NORM (`src/llama-model.cpp:3019`, the group returns at :3039).
    /// OURO is NORM (the pinned llama.cpp selects `ROPE_TYPE_NORM`, and its converter permutes the
    /// Q/K rows into the adjacent layout).
    /// FLINT is in the NEOX group (`src/llama-model.cpp`'s `llama_model_rope_type`, fetched
    /// 2026-11-08).
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
            ("flint", true),
            ("qwen35", true),
            ("qwen35moe", true),
            ("lfm2", true),
            ("lfm2moe", true),
            ("ouro", false),
        ];
        assert_eq!(
            table.len(),
            FAMILY_PROFILES.len(),
            "one expectation per embedded family string"
        );
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

    fn gguf_dump_entry(line: &str) -> Option<(&str, &str)> {
        line.splitn(3, " | ").nth(2)?.split_once(" = ")
    }

    fn model_loader_entry(line: &str) -> Option<(&str, &str)> {
        let (_, entry) = line.split_once(": - kv ")?.1.split_once(':')?;
        let (typed_key, value) = entry.split_once(" = ")?;
        Some((typed_key.split_whitespace().next()?, value))
    }

    fn fixture_kv(fixture: &str) -> Vec<(String, String)> {
        let directory = format!("{}/tests/fixtures/llama-parity/{fixture}", env!("CARGO_MANIFEST_DIR"));
        let gguf_dump = format!("{directory}/gguf_kv.txt");
        let from_gguf_dump = Path::new(&gguf_dump).is_file();
        let path = if from_gguf_dump {
            gguf_dump
        } else {
            format!("{directory}/llama_model_loader_kv.txt")
        };
        let text = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{path}: {error}"));
        let entries: Vec<(String, String)> = text
            .lines()
            .filter_map(|line| {
                if from_gguf_dump {
                    gguf_dump_entry(line)
                } else {
                    model_loader_entry(line)
                }
            })
            .map(|(key, value)| {
                (
                    key.trim().to_string(),
                    value.trim().trim_matches('\'').to_string(),
                )
            })
            .collect();
        assert!(!entries.is_empty(), "{path} yielded no key/value rows");
        entries
    }

    fn kv_value(kv: &[(String, String)], key: &str) -> Option<String> {
        kv.iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value.clone())
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
            let family =
                kv_value(&kv, "general.architecture").expect("header names its architecture");
            let width = kv_value(&kv, &format!("{family}.rope.dimension_count"))
                .or_else(|| kv_value(&kv, &format!("{family}.attention.key_length")))
                .map_or(128, |text| text.parse::<u32>().expect("integer width"));
            let profile =
                family_profile(&family).expect("profile embedded for the header's family");
            assert_eq!(
                profile.rope_pairing(width),
                expected,
                "{fixture} ({family})"
            );
        }
    }

    #[test]
    fn qwen35_headers_carry_the_partial_rotary_width_the_profile_pairs_split_half() {
        for (fixture, sections) in [("qwen35", "[11, 11, 10, 0]"), ("qwen35moe", "[11, 11, 10]")] {
            let kv = fixture_kv(fixture);
            let family =
                kv_value(&kv, "general.architecture").expect("header names its architecture");

            assert_eq!(
                kv_value(&kv, &format!("{family}.rope.dimension_count")).as_deref(),
                Some("64"),
                "{fixture}"
            );
            assert_eq!(
                kv_value(&kv, &format!("{family}.attention.key_length")).as_deref(),
                Some("256"),
                "{fixture}"
            );
            assert_eq!(
                kv_value(&kv, &format!("{family}.rope.dimension_sections")).as_deref(),
                Some(sections),
                "{fixture}"
            );
            let profile =
                family_profile(&family).expect("the recurrent-hybrid families have a profile");
            assert_eq!(
                profile.rope_pairing(64),
                RopePairing::SplitHalf { pairs: 32 },
                "{fixture}"
            );
        }
    }

    #[test]
    fn the_shared_expert_family_names_its_ffn_and_the_dense_hybrid_keeps_the_plain_one() {
        let routed = family_profile("qwen35moe").expect("qwen35moe profile embedded");
        let dense = family_profile("qwen35").expect("qwen35 profile embedded");

        assert_eq!(
            routed.layer_ffn(256).combination,
            FfnCombination::RoutedWithSharedExpert
        );
        assert_eq!(dense.layer_ffn(0).combination, FfnCombination::Exclusive);
    }

    #[test]
    fn lfm2_profile_routes_with_sigmoid_gating_and_a_selection_bias() {
        let kv = fixture_kv("lfm2");
        let family = kv_value(&kv, "general.architecture").expect("header names its architecture");
        assert_eq!(
            kv_value(&kv, &format!("{family}.expert_gating_func")).as_deref(),
            Some("2")
        );
        let profile = family_profile(&family).expect("lfm2moe profile embedded");

        assert_eq!(profile.ffn.routed_gating, ExpertGatingFunc::Sigmoid);
        assert!(profile.ffn.routed_expert_bias);
        assert_eq!(profile.ffn.combination, FfnCombination::Exclusive);
    }

    #[test]
    fn a_family_with_no_profile_or_arch_config_is_an_error_naming_it() {
        let empty_home = tempfile::tempdir().expect("empty home and xdg config home");
        let empty_home_text = empty_home.path().to_str().expect("tempdir path is utf-8");
        let error = temp_env::with_vars(
            [
                ("PROXIMA_ARCH_CONFIG_DIR", None),
                ("XDG_CONFIG_HOME", Some(empty_home_text)),
                ("HOME", Some(empty_home_text)),
            ],
            || family_profile("phi9"),
        )
        .expect_err("phi9 has no profile or arch config");

        assert!(matches!(
            &error,
            InteropError::MissingArchField { architecture, .. } if architecture == "phi9"
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
