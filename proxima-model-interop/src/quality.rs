//! Decode-quality harness: how far a VARIANT decode configuration's
//! per-token logits drift from a full-precision REFERENCE's, on a
//! held-out prompt set -- the gate the next campaign (multi-token passes,
//! dynamic row elision, lower-bit codecs) needs in place of "generated text
//! is byte-identical", which stops distinguishing a real regression from
//! noise the moment either technique legitimately changes which near-tied
//! token wins an argmax.
//!
//! Composes exactly one primitive this crate already has, run twice --
//! once per side, through its existing crate-internal surface:
//!
//! - [`LoadedModel::run_decode_loop_observed`] -- the SAME cached decode
//!   loop [`LoadedModel::generate_with_serving_config`] runs for real
//!   generation. `reference` runs it unforced to read off its own real
//!   greedy trajectory AND that trajectory's own per-step logits in one
//!   pass (teacher forcing on the reference's own path, not the variant's
//!   -- the standard way a decode approximation's per-step drift is scored
//!   without a divergent variant trajectory confounding the comparison
//!   with "wrong context" as well as "wrong logits"); `variant` then runs
//!   the identical loop forced onto `reference`'s own emitted tokens, so
//!   every metric below compares two logit vectors the SAME cached loop
//!   computed over identical input tokens -- never a second, uncached
//!   forward pass.
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

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use serde::Deserialize;

use crate::error::InteropError;
use crate::generate::{BackendRuntime, LoadedModel, supported_serving_config};

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

/// One prompt's teacher-forced comparison, both sides driven through
/// [`LoadedModel::run_decode_loop_observed`] -- the SAME cached decode loop
/// [`LoadedModel::generate_with_serving_config`] runs for real generation,
/// never a second, uncached forward. `reference` runs it unforced
/// (`token_override: None`) to read off its own real greedy trajectory and
/// that trajectory's own per-step logits in one pass; `variant` then runs
/// the identical loop forced onto `reference`'s own emitted token ids
/// (teacher forcing), so every compared step feeds both sides the exact
/// same input tokens, and each side's `logits_sink` callback -- called
/// with that step's own last-position logits, the same slice the loop
/// already slices out to sample from -- is this module's only source of
/// logits. `reference_gpu_layers`/`variant_gpu_layers` select each side's
/// backend via [`supported_serving_config`] (`0` for CPU,
/// [`crate::serving::GPU_LAYERS_ALL`] for Metal on a `metal`-featured
/// build) -- [`quality_report`]'s own doc names why this is a per-side knob
/// rather than baked into `reference`/`variant` themselves ([`ServingConfig`]
/// is call-time state, not part of a loaded checkpoint).
fn score_prompt(
    reference: &LoadedModel,
    reference_gpu_layers: i32,
    variant: &LoadedModel,
    variant_gpu_layers: i32,
    prompt: &Prompt,
    max_tokens: usize,
) -> Result<PromptQuality, InteropError> {
    let reference_config = supported_serving_config(reference_gpu_layers);
    let variant_config = supported_serving_config(variant_gpu_layers);

    let mut reference_logits: Vec<Vec<f32>> = Vec::with_capacity(max_tokens);
    let mut reference_runtime = BackendRuntime::new(&reference_config);
    let (reference_ids, _reference_text, _reference_stopped_by_eos) = reference
        .run_decode_loop_observed(
            &prompt.text,
            max_tokens,
            &reference_config,
            &mut reference_runtime,
            None,
            &mut |_step, logits| reference_logits.push(logits.to_vec()),
        )?;

    // `reference_logits.len()` already includes the eos-triggering step's
    // own logits (`decode_until_stop_or_budget` runs the closure before
    // deciding whether to push that step's token), but `reference_ids`
    // never carries that token -- capping at `reference_ids.len()` drops
    // that trailing row so `variant` is never teacher-forced onto a token
    // that never entered `reference`'s own generated sequence.
    let tokens_compared = reference_ids.len().min(reference_logits.len());
    reference_logits.truncate(tokens_compared);

    let mut variant_logits: Vec<Vec<f32>> = Vec::with_capacity(tokens_compared);
    if tokens_compared > 0 {
        let mut variant_runtime = BackendRuntime::new(&variant_config);
        variant.run_decode_loop_observed(
            &prompt.text,
            tokens_compared,
            &variant_config,
            &mut variant_runtime,
            Some(&reference_ids[..tokens_compared]),
            &mut |_step, logits| variant_logits.push(logits.to_vec()),
        )?;
    }

    let mut first_divergence = None;
    let mut top1_matches = 0usize;
    let mut kl_sum = 0.0f64;
    let mut kl_max = 0.0f64;
    let mut logit_delta_max = 0.0f32;

    for step in 0..tokens_compared {
        let step_reference_logits = &reference_logits[step];
        let step_variant_logits = &variant_logits[step];

        let reference_top1 = argmax(step_reference_logits);
        let variant_top1 = argmax(step_variant_logits);
        if reference_top1 == variant_top1 {
            top1_matches += 1;
        } else if first_divergence.is_none() {
            first_divergence = Some(step);
        }

        let step_kl = kl_divergence(step_reference_logits, step_variant_logits);
        kl_sum += step_kl;
        kl_max = kl_max.max(step_kl);
        logit_delta_max = logit_delta_max.max(max_abs_logit_delta(step_reference_logits, step_variant_logits));
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

    /// [`parse_prompts_jsonl`]'s sad path: a line that is present but is
    /// not valid JSON must surface [`InteropError::MalformedQualityPrompt`]
    /// naming the 1-based line number and `serde_json`'s own reason, not a
    /// silently-skipped row or a panic -- the same contract a malformed
    /// fixture line in `fixtures/quality_prompts.jsonl` would exercise for
    /// real.
    #[test]
    fn parse_prompts_jsonl_rejects_malformed_json_and_names_the_line() {
        let jsonl = concat!(
            "{\"id\":\"code-01\",\"category\":\"code\",\"source\":\"authored\",\"text\":\"Write a function.\"}\n",
            "{not valid json at all\n",
        );

        let error = parse_prompts_jsonl(jsonl.as_bytes())
            .expect_err("second line is not valid json and must be rejected");

        match error {
            crate::error::InteropError::MalformedQualityPrompt { line_number, reason } => {
                assert_eq!(line_number, 2, "the malformed line is 1-based line 2");
                assert!(!reason.is_empty(), "the serde_json reason must be carried through");
            }
            other => panic!("expected MalformedQualityPrompt, got {other:?}"),
        }
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

// -- Real-data proof: run `quality_report` against the actual host-local
// openchat-3.5 checkpoint `bind.rs`'s own `real_openchat_file` module
// already loads for its acceptance tests, on both a degenerate control
// (same model, same backend on both sides) and a real cross-backend
// comparison (CPU reference vs Metal variant). Same convention:
// `#[ignore]`d, mmaps the fixture instead of copying it, and skips
// cleanly when the host-local model cache is absent.
#[cfg(all(test, feature = "metal"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod real_openchat_file {
    use core::ffi::c_void;
    use std::os::fd::AsFd;

    use alloc::vec::Vec;

    use crate::generate::LoadedModel;
    use crate::serving::{GPU_LAYERS_ALL, ServingConfig};

    use super::{Prompt, parse_prompts_jsonl, quality_report};
    #[cfg(feature = "instrument")]
    use super::print_quality_report;

    /// Same read-only `mmap` of the fixture file `bind.rs`'s own
    /// `real_openchat_file::MappedGguf` uses, for the same reason (the
    /// byte range GGUF already stored is the buffer [`LoadedModel::load`]
    /// reads, with no owned copy in between). Kept test-local here rather
    /// than shared with `bind.rs`'s private copy: opening/mapping a file
    /// is exactly the IO step this crate's own module docs disclaim.
    struct MappedGguf {
        base: *mut u8,
        len: usize,
        _file: std::fs::File,
    }

    impl MappedGguf {
        fn open(path: &std::path::Path) -> std::io::Result<Self> {
            let file = std::fs::File::open(path)?;
            let len =
                usize::try_from(file.metadata()?.len()).expect("fixture file length fits in usize");
            // SAFETY: `len` matches the just-opened file's own length; `file`
            // is kept alive in `_file` for as long as `base` is used, and the
            // mapping is read-only/private so no writer can observe or race it.
            let base = unsafe {
                rustix::mm::mmap(
                    core::ptr::null_mut(),
                    len,
                    rustix::mm::ProtFlags::READ,
                    rustix::mm::MapFlags::PRIVATE,
                    file.as_fd(),
                    0,
                )
            }
            .expect("mmap host-local openchat gguf fixture")
            .cast::<u8>();
            Ok(Self {
                base,
                len,
                _file: file,
            })
        }

        fn as_slice(&self) -> &[u8] {
            // SAFETY: `base` points at `len` bytes mapped for `self`'s whole
            // lifetime; this borrows `self` immutably, so nothing can unmap
            // the region while the returned slice is alive.
            unsafe { core::slice::from_raw_parts(self.base, self.len) }
        }
    }

    impl Drop for MappedGguf {
        fn drop(&mut self) {
            // SAFETY: `base`/`len` are exactly what `open`'s `mmap` call
            // returned; nothing else unmaps this region.
            let _ = unsafe { rustix::mm::munmap(self.base.cast::<c_void>(), self.len) };
        }
    }

    /// `PROXIMA_MAX_TOKENS` overrides how many teacher-forced steps
    /// [`quality_report`] compares per prompt, same env var and same
    /// default-on-unparsable convention as `bind.rs`'s own
    /// `real_openchat_file::decode_loop_max_tokens` -- kept small (8)
    /// because this harness runs one CPU-and-one-Metal forward pass EVERY
    /// step, for EVERY prompt, not one decode loop total.
    fn quality_max_tokens() -> usize {
        std::env::var("PROXIMA_MAX_TOKENS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(8)
    }

    /// `PROXIMA_QUALITY_PROMPTS` caps how many of the shipped fixture's 32
    /// prompts this run scores, front-to-back -- the direct knob the next
    /// slice sizes a full-fixture CPU run against (this slice's own report
    /// names the observed per-prompt wall clock for exactly that reason).
    fn quality_prompt_count() -> usize {
        std::env::var("PROXIMA_QUALITY_PROMPTS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(8)
    }

    /// Loads the shipped fixture and truncates it to
    /// [`quality_prompt_count`] prompts, front-to-back -- the same fixture
    /// [`super::tests::ships_fixture_parses_and_covers_every_required_category`]
    /// already proves is well-formed, so a truncation here can only ever
    /// shrink the prompt set, never surface a new parse failure.
    fn load_quality_prompts() -> Vec<Prompt> {
        let bytes = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/quality_prompts.jsonl"
        ))
        .expect("quality_prompts.jsonl fixture ships in-tree");
        let mut prompts = parse_prompts_jsonl(&bytes).expect("shipped fixture is well-formed jsonl");
        prompts.truncate(quality_prompt_count());
        prompts
    }

    /// Opens the host-local checkpoint the same way `bind.rs`'s own
    /// `real_openchat_file` tests do, or returns [`None`] and prints why
    /// this test is skipping -- callers pattern-match to `return` early on
    /// [`None`] rather than failing a run with no host-local model cache.
    fn open_model(mapped: &MappedGguf) -> LoadedModel<'_> {
        let file_bytes = mapped.as_slice();
        let parsed = proxima_gguf::pipe::parse_complete(file_bytes)
            .expect("parse host-local openchat gguf fixture");
        LoadedModel::load(&parsed, file_bytes)
            .expect("load real openchat checkpoint through the public path")
    }

    /// The degenerate control [`quality_report`]'s own doc names: `variant`
    /// is the SAME [`LoadedModel`] as `reference`, on the SAME backend
    /// (Metal, [`GPU_LAYERS_ALL`]) -- both sides then compute the identical
    /// forward at every compared step, so `exact_match_rate` must be
    /// exactly `1.0`. `kl_mean`/`kl_max` are asserted near-zero rather than
    /// bit-exact `0.0` -- a real run measured `kl_mean = -6.47e-11`, not
    /// `0.0`, because two independent calls into the reduce-quantized
    /// matmul's own worker threads are not guaranteed to sum partial
    /// products in the same order, and floating-point addition is not
    /// associative. A harness that cannot pass its own near-zero degenerate
    /// control cannot be trusted on a real comparison.
    #[test]
    #[ignore = "depends on a host-local openchat gguf checkout outside this repo, and a real Metal device"]
    fn default_vs_default_is_the_degenerate_control() {
        let path = std::path::Path::new(ServingConfig::default().model_path);
        if !path.exists() {
            eprintln!(
                "skipping: no host-local openchat gguf fixture at {}",
                ServingConfig::default().model_path
            );
            return;
        }

        let mapped = MappedGguf::open(path).expect("mmap host-local openchat gguf fixture");
        let model = open_model(&mapped);
        let prompts = load_quality_prompts();
        let max_tokens = quality_max_tokens();

        let report = quality_report(&model, GPU_LAYERS_ALL, &model, GPU_LAYERS_ALL, &prompts, max_tokens)
            .expect("quality_report against the same model and backend on both sides");

        #[cfg(feature = "instrument")]
        print_quality_report(&report);

        assert_eq!(report.prompts, prompts.len(), "every prompt in the set must produce a row");
        assert_eq!(
            report.exact_match_rate, 1.0,
            "identical model and backend on both sides must match every compared token exactly"
        );
        // Not bit-exact zero: two independent forward calls against the SAME
        // model/backend measurably differ by ~1e-10 nats on real hardware
        // (`quality_summary kl_mean=-0.000000` against a raw `report.kl_mean`
        // of -6.47e-11 in this slice's own recorded run) -- the reduce-quantized
        // matmul's own worker-thread scheduling is not required to visit
        // partial sums in the same order every call, so an addition that is
        // mathematically associative is not bit-identical across two runs.
        // The tolerance below is four orders of magnitude above that observed
        // floor, so it stays a meaningful "near enough to zero" gate rather
        // than a bit-exact one this backend cannot actually satisfy.
        assert!(
            report.kl_mean.abs() < 1e-6,
            "identical model and backend on both sides must carry ~zero KL divergence, got {}",
            report.kl_mean
        );
        assert!(
            report.kl_max.abs() < 1e-6,
            "identical model and backend on both sides must carry ~zero worst-step KL divergence, got {}",
            report.kl_max
        );
    }

    /// The real cross-backend comparison: `reference` is the CPU forward
    /// (`gpu_layers: 0`), `variant` is the Metal forward
    /// ([`GPU_LAYERS_ALL`]), both against the SAME loaded checkpoint --
    /// this is the finding, not a pass/fail on a specific drift number, so
    /// the only assertions are that every prompt produced a row and every
    /// reported metric is finite. The printed `quality_summary` line
    /// itself is the result this slice reports.
    #[test]
    #[ignore = "depends on a host-local openchat gguf checkout outside this repo, and a real Metal device"]
    fn metal_vs_cpu_reports_real_drift() {
        let path = std::path::Path::new(ServingConfig::default().model_path);
        if !path.exists() {
            eprintln!(
                "skipping: no host-local openchat gguf fixture at {}",
                ServingConfig::default().model_path
            );
            return;
        }

        let mapped = MappedGguf::open(path).expect("mmap host-local openchat gguf fixture");
        let model = open_model(&mapped);
        let prompts = load_quality_prompts();
        let max_tokens = quality_max_tokens();

        let report = quality_report(&model, 0, &model, GPU_LAYERS_ALL, &prompts, max_tokens)
            .expect("quality_report against a CPU reference and a Metal variant");

        #[cfg(feature = "instrument")]
        print_quality_report(&report);

        assert_eq!(report.prompts, prompts.len(), "every prompt in the set must produce a row");
        assert!(report.exact_match_rate.is_finite(), "exact_match_rate must be a real number");
        assert!(report.top1_agreement_rate.is_finite(), "top1_agreement_rate must be a real number");
        assert!(report.kl_mean.is_finite(), "kl_mean must be a real number");
        assert!(report.kl_max.is_finite(), "kl_max must be a real number");
        assert!(report.max_abs_logit_delta.is_finite(), "max_abs_logit_delta must be a real number");
    }
}
