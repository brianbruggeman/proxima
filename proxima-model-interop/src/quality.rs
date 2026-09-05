//! Decode-quality harness: how far a VARIANT decode configuration's
//! per-token logits drift from a full-precision REFERENCE's, on a
//! held-out prompt set -- the gate the next campaign (multi-token passes,
//! dynamic row elision, lower-bit codecs) needs in place of "generated text
//! is byte-identical", which stops distinguishing a real regression from
//! noise the moment either technique legitimately changes which near-tied
//! token wins an argmax.
//!
//! Composes exactly two primitives this crate already has, through their
//! existing public surface:
//!
//! - [`LoadedModel::generate_with_serving_config`] -- walked forward one
//!   token at a time to read off the REFERENCE's own real greedy
//!   trajectory, so every comparison below is made at a context the
//!   reference model actually decided to reach (teacher forcing on the
//!   reference's own path, not the variant's -- the standard way a decode
//!   approximation's per-step drift is scored without a divergent variant
//!   trajectory confounding the comparison with "wrong context" as well as
//!   "wrong logits").
//! - [`LoadedModel::forward_logits_on_backend`] -- called at each of those
//!   contexts against BOTH `reference` and `variant`, so every metric below
//!   compares two logit vectors computed over identical input tokens.
//!
//! [`quality_report`] is deliberately the only entry point: a future
//! variant (a lower-bit codec, a different `gpu_layers` backend, a
//! differently-featured build) plugs in by handing a different `variant`
//! [`LoadedModel`]/`variant_gpu_layers` pair, never by touching this
//! module's comparison loop.
//!
//! [`print_quality_report`] is this module's [`crate::generate`]-style
//! machine-parseable output -- gated behind `instrument` and printed with
//! `std::println!`, the same convention `crate::generate`'s own
//! `print_token_breakdown`/`print_token_breakdown_metal` use for exactly
//! the same reason (a bench/acceptance-test-facing line, not a
//! production log event).

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use serde::Deserialize;

use crate::error::InteropError;
use crate::generate::{LoadedModel, supported_serving_config};

/// One held-out prompt the quality harness scores a variant decode
/// configuration against. `source` names where `text` came from --
/// `"authored"` for a prompt this crate's own fixture wrote rather than
/// quoting a public dataset verbatim (guiding-principle 9: real-shaped
/// data over a byte stub, never a fabricated citation) -- see
/// `fixtures/quality_prompts.jsonl`'s own header comment for the full
/// per-prompt breakdown.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Prompt {
    /// Stable short identifier (`"code-01"`, `"math-03"`) -- carried
    /// through to [`PromptQuality::prompt_id`] so a per-prompt regression
    /// is traceable back to the exact fixture line without relying on
    /// array position.
    pub id: String,
    /// Coarse task category (`"code"`, `"math"`, `"summarization"`,
    /// `"factual_qa"`, `"multilingual"`) -- not read by [`quality_report`]
    /// itself, carried through so a caller can slice the report by
    /// category without re-reading the fixture.
    pub category: String,
    /// Where `text` came from -- `"authored"`, or a named public dataset
    /// this crate can actually cite.
    pub source: String,
    /// The prompt text itself, fed to both `reference` and `variant`
    /// verbatim (no chat template applied -- [`quality_report`] hands this
    /// straight to [`LoadedModel::generate_with_serving_config`], the same
    /// way `bind.rs`'s own `real_openchat_file::default_prompt` already
    /// pre-renders any template it needs before reaching that call).
    pub text: String,
}

/// One prompt's greedy-decode comparison between `reference` and
/// `variant`, teacher-forced on `reference`'s own generated trajectory
/// (this module's own doc explains why).
#[derive(Debug, Clone, PartialEq)]
pub struct PromptQuality {
    /// [`Prompt::id`] this row scores.
    pub prompt_id: String,
    /// How many of the requested `max_tokens` steps were actually
    /// compared -- fewer than `max_tokens` only when `reference`'s own
    /// greedy decode hit its eos token first (`generate_with_serving_config`'s
    /// own `stopped_by_eos`), in which case there is no reference token
    /// left to condition a further step on.
    pub tokens_compared: usize,
    /// The first step (0-indexed) at which `variant`'s own top-1 token
    /// disagreed with `reference`'s -- [`None`] if every compared step
    /// agreed.
    pub first_divergence: Option<usize>,
    /// `first_divergence.unwrap_or(tokens_compared) / tokens_compared` --
    /// the fraction of the leading run before the first disagreement,
    /// `1.0` exactly when [`Self::first_divergence`] is [`None`].
    pub exact_match_rate: f64,
    /// Fraction of ALL compared steps (not just the leading run) where
    /// `variant`'s top-1 token equaled `reference`'s -- distinct from
    /// [`Self::exact_match_rate`] because a variant can disagree once and
    /// then agree again on every later step (still conditioned on
    /// `reference`'s own trajectory), which the leading-run metric alone
    /// cannot show.
    pub top1_agreement_rate: f64,
    /// Mean over compared steps of `KL(softmax(reference) ||
    /// softmax(variant))`, nats.
    pub kl_mean: f64,
    /// Max over compared steps of that same per-step KL divergence -- the
    /// single worst step, where [`Self::kl_mean`] is the average one.
    pub kl_max: f64,
    /// Max, over every compared step and every vocabulary entry, of
    /// `|reference_logit - variant_logit|` -- the rawest of the four
    /// metrics: unlike the other three, this is computed before any
    /// softmax normalization, so it also catches a variant whose logits
    /// are uniformly rescaled in a way softmax would otherwise cancel out.
    pub max_abs_logit_delta: f32,
}

/// The whole prompt set's aggregate, plus every [`PromptQuality`] row that
/// produced it -- see [`quality_report`]'s own doc for how each aggregate
/// is weighted across prompts.
#[derive(Debug, Clone, PartialEq)]
pub struct QualityReport {
    pub per_prompt: Vec<PromptQuality>,
    /// `per_prompt.len()`.
    pub prompts: usize,
    /// Sum of every row's [`PromptQuality::tokens_compared`].
    pub tokens: usize,
    /// Token-weighted mean of [`PromptQuality::exact_match_rate`]'s own
    /// numerator (matched tokens) over [`Self::tokens`] -- NOT a plain
    /// mean of the per-prompt rates, so a longer prompt's leading run
    /// contributes proportionally more tokens to the aggregate than a
    /// short one's.
    pub exact_match_rate: f64,
    /// Token-weighted mean of [`PromptQuality::top1_agreement_rate`], same
    /// weighting rationale as [`Self::exact_match_rate`].
    pub top1_agreement_rate: f64,
    /// Token-weighted mean of [`PromptQuality::kl_mean`].
    pub kl_mean: f64,
    /// Max over every row's [`PromptQuality::kl_max`] -- the single worst
    /// step across the entire prompt set.
    pub kl_max: f64,
    /// Max over every row's [`PromptQuality::max_abs_logit_delta`].
    pub max_abs_logit_delta: f32,
}

/// `ln(probability)` floor before it enters a KL term's denominator --
/// [`kl_divergence`]'s own doc explains why a hard zero would make the
/// divergence infinite (and therefore useless to average) the moment a
/// variant assigns a reference-favored token a computed probability of
/// exactly zero, which single-precision underflow makes unremarkable
/// rather than exceptional.
const MIN_PROBABILITY: f64 = 1e-12;

/// Numerically-stable softmax over one step's raw logits, computed in
/// `f64` throughout -- `logits` are `f32` (matching every other tensor
/// value this crate's forward pass produces), but a vocabulary-wide KL sum
/// accumulates thousands of `p * ln(p / q)` terms, so the softmax feeding
/// it is computed at the wider precision to keep that sum's own rounding
/// error out of the reported number.
fn softmax_f64(logits: &[f32]) -> Vec<f64> {
    let max_logit: f64 = f64::from(logits.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    let exps: Vec<f64> = logits
        .iter()
        .map(|&logit| (f64::from(logit) - max_logit).exp())
        .collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&value| value / sum).collect()
}

/// `argmax(logits)` -- the greedy pick both [`quality_report`]'s top-1
/// agreement and exact-match metrics compare, ties broken toward the
/// lower index (matching `proxima_tokenizer::sample::sample_next_token`'s
/// own greedy tie-break, so this reads the same winner that path's own
/// `temperature <= 0.0` branch would pick).
fn argmax(logits: &[f32]) -> usize {
    let mut best_index = 0;
    let mut best_value = f32::NEG_INFINITY;
    for (index, &value) in logits.iter().enumerate() {
        if value > best_value {
            best_value = value;
            best_index = index;
        }
    }
    best_index
}

/// `KL(reference || variant)` in nats over one step's full vocabulary,
/// both distributions [`softmax_f64`]'d from their own raw logits first.
/// [`MIN_PROBABILITY`] floors `variant`'s own probability before it enters
/// the denominator -- see that constant's own doc for why a hard zero is
/// the wrong floor here.
fn kl_divergence(reference_logits: &[f32], variant_logits: &[f32]) -> f64 {
    let reference_probabilities = softmax_f64(reference_logits);
    let variant_probabilities = softmax_f64(variant_logits);
    reference_probabilities
        .iter()
        .zip(variant_probabilities.iter())
        .map(|(&reference_probability, &variant_probability)| {
            if reference_probability <= 0.0 {
                0.0
            } else {
                reference_probability
                    * (reference_probability / variant_probability.max(MIN_PROBABILITY)).ln()
            }
        })
        .sum()
}

/// `max(|reference_logits[i] - variant_logits[i]|)` across the shared
/// vocabulary -- the one metric [`quality_report`] computes on the raw
/// logits rather than a softmax of them ([`PromptQuality::max_abs_logit_delta`]'s
/// own doc explains why).
fn max_abs_logit_delta(reference_logits: &[f32], variant_logits: &[f32]) -> f32 {
    reference_logits
        .iter()
        .zip(variant_logits.iter())
        .map(|(&reference_logit, &variant_logit)| (reference_logit - variant_logit).abs())
        .fold(0.0, f32::max)
}

/// One prompt's teacher-forced comparison: `reference_gpu_layers`/
/// `variant_gpu_layers` reach [`LoadedModel::forward_logits_on_backend`]
/// directly (`0` for CPU, [`crate::serving::GPU_LAYERS_ALL`] for Metal on a
/// `metal`-featured build) -- [`quality_report`]'s own doc names why this
/// is a per-side knob rather than baked into `reference`/`variant`
/// themselves ([`ServingConfig`] is call-time state, not part of a loaded
/// checkpoint).
fn score_prompt(
    reference: &LoadedModel,
    reference_gpu_layers: i32,
    variant: &LoadedModel,
    variant_gpu_layers: i32,
    prompt: &Prompt,
    max_tokens: usize,
) -> Result<PromptQuality, InteropError> {
    let reference_config = supported_serving_config(reference_gpu_layers);

    let mut tokens_compared = 0usize;
    let mut first_divergence = None;
    let mut top1_matches = 0usize;
    let mut kl_sum = 0.0f64;
    let mut kl_max = 0.0f64;
    let mut logit_delta_max = 0.0f32;

    for step in 0..max_tokens {
        let context_text = if step == 0 {
            prompt.text.clone()
        } else {
            let (reference_ids, continuation, _stopped_by_eos) =
                reference.generate_with_serving_config(&prompt.text, step, reference_config)?;
            if reference_ids.len() < step {
                // `reference`'s own greedy decode already hit its eos token
                // before reaching this step -- no further reference context
                // exists to condition `variant` on, so this prompt's
                // comparison stops here rather than manufacturing one.
                break;
            }
            format!("{}{continuation}", prompt.text)
        };

        let reference_logits = reference.forward_logits_on_backend(&context_text, reference_gpu_layers)?;
        let variant_logits = variant.forward_logits_on_backend(&context_text, variant_gpu_layers)?;

        let reference_top1 = argmax(&reference_logits);
        let variant_top1 = argmax(&variant_logits);
        if reference_top1 == variant_top1 {
            top1_matches += 1;
        } else if first_divergence.is_none() {
            first_divergence = Some(step);
        }

        let step_kl = kl_divergence(&reference_logits, &variant_logits);
        kl_sum += step_kl;
        kl_max = kl_max.max(step_kl);
        logit_delta_max = logit_delta_max.max(max_abs_logit_delta(&reference_logits, &variant_logits));

        tokens_compared += 1;
    }

    let matched_leading_tokens = first_divergence.unwrap_or(tokens_compared);
    let exact_match_rate = if tokens_compared == 0 {
        1.0
    } else {
        matched_leading_tokens as f64 / tokens_compared as f64
    };
    let top1_agreement_rate = if tokens_compared == 0 {
        1.0
    } else {
        top1_matches as f64 / tokens_compared as f64
    };
    let kl_mean = if tokens_compared == 0 {
        0.0
    } else {
        kl_sum / tokens_compared as f64
    };

    Ok(PromptQuality {
        prompt_id: prompt.id.clone(),
        tokens_compared,
        first_divergence,
        exact_match_rate,
        top1_agreement_rate,
        kl_mean,
        kl_max,
        max_abs_logit_delta: logit_delta_max,
    })
}

/// Scores `variant` against `reference` over `prompts`, `max_tokens` steps
/// each, teacher-forced on `reference`'s own greedy trajectory -- this
/// module's own doc for the full rationale and [`PromptQuality`]/
/// [`QualityReport`] for what each field means.
///
/// `reference_gpu_layers`/`variant_gpu_layers` select each side's backend
/// (`0` for CPU, [`crate::serving::GPU_LAYERS_ALL`] for Metal on a
/// `metal`-featured build) -- passing the reference's own value on both
/// sides against the SAME [`LoadedModel`] is the degenerate control this
/// module's own tests use (`exact_match_rate: 1.0`, every KL term `0.0`
/// exactly, since both sides then compute the identical forward).
///
/// # Errors
///
/// Whatever [`LoadedModel::generate_with_serving_config`] or
/// [`LoadedModel::forward_logits_on_backend`] can fail with, on either
/// `reference` or `variant`.
pub fn quality_report(
    reference: &LoadedModel,
    reference_gpu_layers: i32,
    variant: &LoadedModel,
    variant_gpu_layers: i32,
    prompts: &[Prompt],
    max_tokens: usize,
) -> Result<QualityReport, InteropError> {
    let mut per_prompt = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        per_prompt.push(score_prompt(
            reference,
            reference_gpu_layers,
            variant,
            variant_gpu_layers,
            prompt,
            max_tokens,
        )?);
    }

    let tokens: usize = per_prompt.iter().map(|row| row.tokens_compared).sum();
    let matched_tokens: usize = per_prompt
        .iter()
        .map(|row| row.first_divergence.unwrap_or(row.tokens_compared))
        .sum();
    let top1_matched_tokens: f64 = per_prompt
        .iter()
        .map(|row| row.top1_agreement_rate * row.tokens_compared as f64)
        .sum();
    let kl_weighted_sum: f64 = per_prompt
        .iter()
        .map(|row| row.kl_mean * row.tokens_compared as f64)
        .sum();
    let kl_max = per_prompt
        .iter()
        .map(|row| row.kl_max)
        .fold(0.0, f64::max);
    let max_abs_logit_delta = per_prompt
        .iter()
        .map(|row| row.max_abs_logit_delta)
        .fold(0.0, f32::max);

    let (exact_match_rate, top1_agreement_rate, kl_mean) = if tokens == 0 {
        (1.0, 1.0, 0.0)
    } else {
        (
            matched_tokens as f64 / tokens as f64,
            top1_matched_tokens / tokens as f64,
            kl_weighted_sum / tokens as f64,
        )
    };

    Ok(QualityReport {
        prompts: per_prompt.len(),
        tokens,
        exact_match_rate,
        top1_agreement_rate,
        kl_mean,
        kl_max,
        max_abs_logit_delta,
        per_prompt,
    })
}

/// Parses [`Prompt`] rows out of a `\n`-delimited JSON-lines byte buffer
/// (`fixtures/quality_prompts.jsonl`'s own shape) -- blank lines skipped,
/// every other line must deserialize to a complete [`Prompt`] or this
/// returns [`InteropError::MalformedQualityPrompt`] naming the 1-based
/// line number and `serde_json`'s own reason.
///
/// # Errors
///
/// [`InteropError::MalformedQualityPrompt`] at the first line that is not
/// valid JSON or is missing one of [`Prompt`]'s required fields.
pub fn parse_prompts_jsonl(bytes: &[u8]) -> Result<Vec<Prompt>, InteropError> {
    let text = core::str::from_utf8(bytes).map_err(|error| InteropError::MalformedQualityPrompt {
        line_number: 0,
        reason: error.to_string(),
    })?;
    text.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            serde_json::from_str::<Prompt>(line).map_err(|error| InteropError::MalformedQualityPrompt {
                line_number: index + 1,
                reason: error.to_string(),
            })
        })
        .collect()
}

/// [`QualityReport`]'s machine-parseable output, `crate::generate`'s own
/// `print_token_breakdown` convention: one `quality_summary` line (the
/// aggregate), then one `quality_prompt` line per [`PromptQuality`] row --
/// `std::println!`, gated behind `instrument` for the same reason that
/// module's own print functions are (a bench/acceptance-test-facing line,
/// not a production log event this crate's `warn!`/`error!` telemetry
/// macros are for).
#[cfg(feature = "instrument")]
pub fn print_quality_report(report: &QualityReport) {
    std::println!(
        "quality_summary prompts={} tokens={} exact_match={:.6} top1={:.6} kl_mean={:.6} kl_max={:.6} max_abs_logit_delta={:.6}",
        report.prompts,
        report.tokens,
        report.exact_match_rate,
        report.top1_agreement_rate,
        report.kl_mean,
        report.kl_max,
        report.max_abs_logit_delta,
    );
    for row in &report.per_prompt {
        std::println!(
            "quality_prompt id={} tokens_compared={} first_divergence={} exact_match={:.6} top1={:.6} kl_mean={:.6} kl_max={:.6} max_abs_logit_delta={:.6}",
            row.prompt_id,
            row.tokens_compared,
            row.first_divergence
                .map_or_else(|| "none".to_string(), |index| index.to_string()),
            row.exact_match_rate,
            row.top1_agreement_rate,
            row.kl_mean,
            row.kl_max,
            row.max_abs_logit_delta,
        );
    }
}

#[cfg(all(test, feature = "std"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::string::ToString;
    use alloc::vec;

    use super::{Prompt, argmax, kl_divergence, max_abs_logit_delta, parse_prompts_jsonl};

    /// [`argmax`]'s own contract: the index of the largest logit, ties
    /// broken toward the lower index (matching `sample_next_token`'s
    /// greedy tie-break).
    #[test]
    fn argmax_picks_the_lower_index_on_a_tie() {
        let logits = [0.5_f32, 1.0, 1.0, 0.25];

        assert_eq!(argmax(&logits), 1, "first occurrence of the max wins ties");
    }

    /// Two identical logit vectors carry zero divergence -- the
    /// distribution-level counterpart of [`crate::quality`]'s own
    /// default-vs-default control (`generate.rs` module's own doc for why
    /// that control matters).
    #[test]
    fn kl_divergence_of_identical_logits_is_zero() {
        let logits = [1.0_f32, 2.0, -1.0, 0.5, 3.0];

        let divergence = kl_divergence(&logits, &logits);

        assert!(
            divergence.abs() < 1e-9,
            "identical distributions must carry ~0 KL, got {divergence}"
        );
    }

    /// A real, asymmetric divergence: `variant` is confident in a DIFFERENT
    /// token than `reference`, so `KL(reference || variant)` must be
    /// strictly positive.
    #[test]
    fn kl_divergence_is_positive_when_variant_disagrees() {
        let reference_logits = [5.0_f32, 0.0, 0.0];
        let variant_logits = [0.0_f32, 5.0, 0.0];

        let divergence = kl_divergence(&reference_logits, &variant_logits);

        assert!(
            divergence > 1.0,
            "a variant confidently favoring a different token must diverge sharply, got {divergence}"
        );
    }

    /// [`max_abs_logit_delta`]'s own contract: the largest per-entry
    /// absolute difference, not a mean or a sum.
    #[test]
    fn max_abs_logit_delta_finds_the_single_largest_gap() {
        let reference_logits = [1.0_f32, 2.0, 3.0];
        let variant_logits = [1.1_f32, 2.0, 30.0];

        let delta = max_abs_logit_delta(&reference_logits, &variant_logits);

        assert!(
            (delta - 27.0).abs() < 1e-6,
            "expected the |3.0 - 30.0| outlier, got {delta}"
        );
    }

    /// [`parse_prompts_jsonl`]'s happy path: one line per [`Prompt`], blank
    /// lines skipped, field values carried through verbatim.
    #[test]
    fn parse_prompts_jsonl_reads_one_prompt_per_nonblank_line() {
        let jsonl = concat!(
            "{\"id\":\"code-01\",\"category\":\"code\",\"source\":\"authored\",\"text\":\"Write a function.\"}\n",
            "\n",
            "{\"id\":\"math-01\",\"category\":\"math\",\"source\":\"authored\",\"text\":\"What is 2+2?\"}\n",
        );

        let prompts = parse_prompts_jsonl(jsonl.as_bytes()).expect("valid jsonl fixture parses");

        assert_eq!(
            prompts,
            vec![
                Prompt {
                    id: "code-01".to_string(),
                    category: "code".to_string(),
                    source: "authored".to_string(),
                    text: "Write a function.".to_string(),
                },
                Prompt {
                    id: "math-01".to_string(),
                    category: "math".to_string(),
                    source: "authored".to_string(),
                    text: "What is 2+2?".to_string(),
                },
            ]
        );
    }

    /// The real fixture this crate ships parses cleanly and carries at
    /// least one prompt from every category the task's campaign needs
    /// covered -- a fixture-shape regression (a stray trailing comma, a
    /// renamed field) surfaces here instead of only at the ignored
    /// real-checkpoint test that actually loads a model.
    #[test]
    fn ships_fixture_parses_and_covers_every_required_category() {
        let bytes = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/quality_prompts.jsonl"
        ))
        .expect("quality_prompts.jsonl fixture ships in-tree");

        let prompts = parse_prompts_jsonl(&bytes).expect("shipped fixture is well-formed jsonl");

        assert_eq!(prompts.len(), 32, "fixture is specified as 32 prompts");
        for category in ["code", "math", "summarization", "factual_qa", "multilingual"] {
            assert!(
                prompts.iter().any(|prompt| prompt.category == category),
                "fixture must cover category {category:?}"
            );
        }
        for prompt in &prompts {
            assert!(!prompt.text.trim().is_empty(), "{:?} has empty text", prompt.id);
            assert!(!prompt.source.trim().is_empty(), "{:?} has empty source", prompt.id);
        }
    }
}
