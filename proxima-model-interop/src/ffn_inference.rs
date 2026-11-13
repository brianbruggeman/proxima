//! FFN-block facts a checkpoint states itself, through the presence of named
//! tensors, so a never-tried model does not need a per-family TOML row to say
//! them (PROXNUM-424 design, card 1).
//!
//! Same method as [`crate::bind::checkpoint_has_qk_norm`]: read the tensor
//! directory, never the architecture name. [`infer_ffn_from_tensor_names`] is
//! the pure rule over names (the unit under test); [`with_inferred_ffn`] is the
//! one wiring point that overlays the result onto a [`FamilyProfile`] so
//! `proxima_tensor::spec::gqa_descriptor_from_shape` and
//! `proxima_tensor::spec::sliding_pattern_descriptor_from_gguf` build their
//! `LayerFfnConfig` from the inferred value. The profile field stays until the
//! cutover card deletes it.
//!
//! Rules, each over `blk.{N}.<leaf>` on any block:
//! - `post_attention_norm`: `post_attention_norm.weight` present
//! - `output_scale`: `layer_output_scale.weight` present
//! - `routed_expert_bias`: `exp_probs_b.bias` present
//! - `exclusive_dense_post_norm`: `post_ffw_norm.weight` present and neither
//!   `post_ffw_norm_1.weight` nor `post_ffw_norm_2.weight`
//! - `shared_expert_gate`: `ffn_gate_inp_shexp.weight` present
//! - `combination`: all of `ffn_{gate,up,down}_shexp.weight` present is
//!   [`InferredCombination::RoutedWithSharedExpert`]; else both
//!   `post_ffw_norm_1.weight` and `post_ffw_norm_2.weight` present is
//!   [`InferredCombination::ParallelDenseMoe`]; else
//!   [`InferredCombination::Exclusive`]

use proxima_gguf::pipe::ParsedGguf;
use proxima_tensor::spec::{FamilyProfile, FfnCombination, LayerFfnConfig};
use proxima_telemetry::debug;

/// Which FFN combination the tensor directory describes. The
/// [`ParallelDenseMoe`](InferredCombination::ParallelDenseMoe) payload (six
/// sub-norm flags) is not inferred here; it stays profile-sourced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InferredCombination {
    Exclusive,
    ParallelDenseMoe,
    RoutedWithSharedExpert,
}

/// The tensor directory's answer for each FFN fact, plus the per-pattern
/// tensor counts that are the evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InferredFfn {
    pub post_attention_norm: bool,
    pub output_scale: bool,
    pub routed_expert_bias: bool,
    pub exclusive_dense_post_norm: bool,
    pub shared_expert_gate: bool,
    pub combination: InferredCombination,
    pub counts: PatternCounts,
}

/// How many `blk.*` tensors matched each leaf the rules read.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct PatternCounts {
    pub post_attention_norm: u32,
    pub layer_output_scale: u32,
    pub exp_probs_b: u32,
    pub post_ffw_norm: u32,
    pub post_ffw_norm_1: u32,
    pub post_ffw_norm_2: u32,
    pub ffn_gate_inp_shexp: u32,
    pub ffn_gate_shexp: u32,
    pub ffn_up_shexp: u32,
    pub ffn_down_shexp: u32,
}

/// Applies the FFN rules to a tensor-name list (one pass, no allocation).
pub(crate) fn infer_ffn_from_tensor_names<'a>(
    names: impl IntoIterator<Item = &'a str>,
) -> InferredFfn {
    let mut counts = PatternCounts::default();
    for name in names {
        let Some(rest) = name.strip_prefix("blk.") else {
            continue;
        };
        let slot = if rest.ends_with(".post_attention_norm.weight") {
            &mut counts.post_attention_norm
        } else if rest.ends_with(".layer_output_scale.weight") {
            &mut counts.layer_output_scale
        } else if rest.ends_with(".exp_probs_b.bias") {
            &mut counts.exp_probs_b
        } else if rest.ends_with(".post_ffw_norm.weight") {
            &mut counts.post_ffw_norm
        } else if rest.ends_with(".post_ffw_norm_1.weight") {
            &mut counts.post_ffw_norm_1
        } else if rest.ends_with(".post_ffw_norm_2.weight") {
            &mut counts.post_ffw_norm_2
        } else if rest.ends_with(".ffn_gate_inp_shexp.weight") {
            &mut counts.ffn_gate_inp_shexp
        } else if rest.ends_with(".ffn_gate_shexp.weight") {
            &mut counts.ffn_gate_shexp
        } else if rest.ends_with(".ffn_up_shexp.weight") {
            &mut counts.ffn_up_shexp
        } else if rest.ends_with(".ffn_down_shexp.weight") {
            &mut counts.ffn_down_shexp
        } else {
            continue;
        };
        *slot += 1;
    }
    let parallel_norms = counts.post_ffw_norm_1 > 0 && counts.post_ffw_norm_2 > 0;
    let shared_expert =
        counts.ffn_gate_shexp > 0 && counts.ffn_up_shexp > 0 && counts.ffn_down_shexp > 0;
    let combination = if shared_expert {
        InferredCombination::RoutedWithSharedExpert
    } else if parallel_norms {
        InferredCombination::ParallelDenseMoe
    } else {
        InferredCombination::Exclusive
    };
    InferredFfn {
        post_attention_norm: counts.post_attention_norm > 0,
        output_scale: counts.layer_output_scale > 0,
        routed_expert_bias: counts.exp_probs_b > 0,
        exclusive_dense_post_norm: counts.post_ffw_norm > 0
            && counts.post_ffw_norm_1 == 0
            && counts.post_ffw_norm_2 == 0,
        shared_expert_gate: counts.ffn_gate_inp_shexp > 0,
        combination,
        counts,
    }
}

/// `profile` with its FFN facts replaced by what `parsed`'s tensor directory
/// states. `routed_expert_bias` is only meaningful when the checkpoint has
/// experts, so a dense checkpoint (`expert_count == 0`) keeps the profile's
/// value for it; `shared_expert_gate` is only meaningful under
/// [`InferredCombination::RoutedWithSharedExpert`], elsewhere the profile's
/// value is kept. A [`InferredCombination::ParallelDenseMoe`] keeps the
/// profile's six-flag payload; any other combination clears it so
/// [`FamilyProfile::layer_ffn`] takes the inferred combination.
///
/// Composes [`infer_ffn_from_tensor_names`] over [`ParsedGguf::tensors`]; each
/// inferred value is one structured `debug!` event with its tensor evidence.
#[must_use]
pub(crate) fn with_inferred_ffn(
    profile: FamilyProfile,
    parsed: &ParsedGguf,
    expert_count: u32,
) -> FamilyProfile {
    let inferred = infer_ffn_from_tensor_names(parsed.tensors.iter().map(|tensor| tensor.name.as_str()));
    log_inferred(&inferred, expert_count);
    let combination = match inferred.combination {
        InferredCombination::Exclusive => FfnCombination::Exclusive,
        InferredCombination::RoutedWithSharedExpert => FfnCombination::RoutedWithSharedExpert,
        InferredCombination::ParallelDenseMoe => profile.ffn.combination,
    };
    let parallel_dense_moe = match inferred.combination {
        InferredCombination::ParallelDenseMoe => profile.parallel_dense_moe,
        _ => None,
    };
    let routed_expert_bias = if expert_count > 0 {
        inferred.routed_expert_bias
    } else {
        profile.ffn.routed_expert_bias
    };
    let shared_expert_gate = if inferred.combination == InferredCombination::RoutedWithSharedExpert
    {
        inferred.shared_expert_gate
    } else {
        profile.ffn.shared_expert_gate
    };
    FamilyProfile {
        ffn: LayerFfnConfig {
            post_attention_norm: inferred.post_attention_norm,
            output_scale: inferred.output_scale,
            exclusive_dense_post_norm: inferred.exclusive_dense_post_norm,
            shared_expert_gate,
            routed_expert_bias,
            combination,
            ..profile.ffn
        },
        parallel_dense_moe,
        ..profile
    }
}

fn log_inferred(inferred: &InferredFfn, expert_count: u32) {
    let counts = &inferred.counts;
    debug!(
        field = "post_attention_norm",
        value = inferred.post_attention_norm,
        tensor = "blk.*.post_attention_norm.weight",
        count = counts.post_attention_norm,
        "ffn fact inferred from tensor presence"
    );
    debug!(
        field = "output_scale",
        value = inferred.output_scale,
        tensor = "blk.*.layer_output_scale.weight",
        count = counts.layer_output_scale,
        "ffn fact inferred from tensor presence"
    );
    debug!(
        field = "routed_expert_bias",
        value = inferred.routed_expert_bias,
        applied = expert_count > 0,
        tensor = "blk.*.exp_probs_b.bias",
        count = counts.exp_probs_b,
        "ffn fact inferred from tensor presence"
    );
    debug!(
        field = "exclusive_dense_post_norm",
        value = inferred.exclusive_dense_post_norm,
        tensor = "blk.*.post_ffw_norm.weight without _1/_2",
        count = counts.post_ffw_norm,
        count_norm_1 = counts.post_ffw_norm_1,
        count_norm_2 = counts.post_ffw_norm_2,
        "ffn fact inferred from tensor presence"
    );
    debug!(
        field = "shared_expert_gate",
        value = inferred.shared_expert_gate,
        tensor = "blk.*.ffn_gate_inp_shexp.weight",
        count = counts.ffn_gate_inp_shexp,
        "ffn fact inferred from tensor presence"
    );
    debug!(
        field = "combination",
        value = ?inferred.combination,
        tensor = "blk.*.ffn_{gate,up,down}_shexp.weight, blk.*.post_ffw_norm_{1,2}.weight",
        count_gate_shexp = counts.ffn_gate_shexp,
        count_up_shexp = counts.ffn_up_shexp,
        count_down_shexp = counts.ffn_down_shexp,
        count_norm_1 = counts.post_ffw_norm_1,
        count_norm_2 = counts.post_ffw_norm_2,
        "ffn fact inferred from tensor presence"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiles::family_profile;

    const FIXTURES: [(&str, Option<&str>, u32, &str); 8] = [
        (
            "gemma4",
            Some("gemma4"),
            0,
            include_str!("../tests/fixtures/tensor_names/gemma4.txt"),
        ),
        (
            "granite3.1-moe",
            Some("granitemoe"),
            32,
            include_str!("../tests/fixtures/tensor_names/granitemoe.txt"),
        ),
        (
            "flint",
            Some("flint"),
            256,
            include_str!("../tests/fixtures/tensor_names/flint.txt"),
        ),
        (
            "Llama-3.2",
            Some("llama"),
            0,
            include_str!("../tests/fixtures/tensor_names/llama.txt"),
        ),
        (
            "granite-4.0",
            None,
            0,
            include_str!("../tests/fixtures/tensor_names/granite4.txt"),
        ),
        (
            "LFM2",
            Some("lfm2"),
            0,
            include_str!("../tests/fixtures/tensor_names/lfm2.txt"),
        ),
        (
            "qwen2.5",
            Some("qwen2"),
            0,
            include_str!("../tests/fixtures/tensor_names/qwen25.txt"),
        ),
        (
            "qwen3",
            Some("qwen3"),
            0,
            include_str!("../tests/fixtures/tensor_names/qwen3.txt"),
        ),
    ];

    fn names(text: &str) -> impl Iterator<Item = &str> {
        text.lines()
            .map(|line| line.strip_prefix("tensor name=").expect("fixture line is a tensor name"))
    }

    fn inferred_combination(inferred: &InferredFfn, profile: &FamilyProfile) -> FfnCombination {
        match inferred.combination {
            InferredCombination::Exclusive => FfnCombination::Exclusive,
            InferredCombination::RoutedWithSharedExpert => FfnCombination::RoutedWithSharedExpert,
            InferredCombination::ParallelDenseMoe => profile.ffn.combination,
        }
    }

    #[test]
    fn fixtures_carry_each_headers_full_tensor_directory() {
        let counts: Vec<usize> = FIXTURES.iter().map(|row| names(row.3).count()).collect();

        assert_eq!(counts, [541, 242, 779, 147, 254, 148, 290, 311]);
    }

    #[test]
    fn inferred_ffn_equals_the_profile_for_each_local_header() {
        let mut compared = 0;
        let mut disagreements = Vec::new();
        for (model, family, expert_count, text) in FIXTURES {
            let Some(family) = family else { continue };
            let profile = family_profile(family).expect("family profile embedded");
            let inferred = infer_ffn_from_tensor_names(names(text));
            let ffn = profile.layer_ffn(expert_count);
            let mut facts = vec![
                ("post_attention_norm", inferred.post_attention_norm == ffn.post_attention_norm),
                ("output_scale", inferred.output_scale == ffn.output_scale),
                (
                    "exclusive_dense_post_norm",
                    inferred.exclusive_dense_post_norm == ffn.exclusive_dense_post_norm,
                ),
                ("combination", inferred_combination(&inferred, &profile) == ffn.combination),
            ];
            if inferred.combination == InferredCombination::RoutedWithSharedExpert {
                facts.push((
                    "shared_expert_gate",
                    inferred.shared_expert_gate == ffn.shared_expert_gate,
                ));
            }
            if expert_count > 0 {
                facts.push((
                    "routed_expert_bias",
                    inferred.routed_expert_bias == ffn.routed_expert_bias,
                ));
            }
            for (field, agrees) in facts {
                compared += 1;
                if !agrees {
                    disagreements.push(format!("{model}.{field}"));
                }
            }
        }

        assert_eq!(disagreements, Vec::<String>::new(), "inferred != profile");
        assert_eq!(
            compared,
            7 * 4 + 2 + 1,
            "7 models x 4 always-meaningful fields, routed_expert_bias on 2 expert models, shared_expert_gate on flint"
        );
    }

    #[test]
    fn granite_4_0_without_a_profile_infers_a_plain_exclusive_ffn() {
        let text = FIXTURES[4].3;

        let inferred = infer_ffn_from_tensor_names(names(text));

        assert!(!inferred.post_attention_norm);
        assert!(!inferred.output_scale);
        assert!(!inferred.routed_expert_bias);
        assert!(!inferred.exclusive_dense_post_norm);
        assert!(!inferred.shared_expert_gate);
        assert_eq!(inferred.combination, InferredCombination::Exclusive);
    }

    #[test]
    fn gemma4_header_states_its_sandwich_norms_and_flint_its_shared_expert() {
        let gemma4 = infer_ffn_from_tensor_names(names(FIXTURES[0].3));
        let flint = infer_ffn_from_tensor_names(names(FIXTURES[2].3));

        assert_eq!(gemma4.counts.post_attention_norm, 35);
        assert_eq!(gemma4.counts.layer_output_scale, 35);
        assert!(gemma4.exclusive_dense_post_norm);
        assert_eq!(flint.combination, InferredCombination::RoutedWithSharedExpert);
        assert_eq!(flint.counts.ffn_gate_shexp + flint.counts.ffn_up_shexp + flint.counts.ffn_down_shexp, 141);
        assert!(!flint.shared_expert_gate);
    }

    #[test]
    fn parallel_sub_norms_make_a_parallel_dense_moe_and_clear_the_exclusive_norm() {
        let names = [
            "blk.0.post_ffw_norm.weight",
            "blk.0.post_ffw_norm_1.weight",
            "blk.0.post_ffw_norm_2.weight",
            "blk.0.ffn_gate_inp.weight",
        ];

        let inferred = infer_ffn_from_tensor_names(names);

        assert_eq!(inferred.combination, InferredCombination::ParallelDenseMoe);
        assert!(!inferred.exclusive_dense_post_norm);
    }
}
