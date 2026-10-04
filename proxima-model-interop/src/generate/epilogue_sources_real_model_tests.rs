#![allow(clippy::expect_used)]

//! Real-checkpoint drift gate for `NumericPolicy::epilogue_sources`: the
//! widened reduce-epilogue pass folds each RMSNorm apply into its
//! sum-of-squares reduce (170 dispatches per gemma4-E2B decode token), and
//! this test measures what that does to the logits on the Metal backend, per
//! decode step, against the unfused plan on the same loaded checkpoint.
//!
//! Skips with a loud message when the checkpoint is absent, so an explicit
//! run never passes having executed nothing.

use core::ops::ControlFlow;
use std::fs::File;

use memmap2::Mmap;
use omega::MathMode;
use proxima_gguf::parse_complete;
use proxima_tensor::NumericPolicy;

use super::prefix_resume_long_prompt_tests::greedy_config;
use super::{BackendRuntime, LogitsSink, NodeValuesSink};
use crate::LoadedModel;
use crate::serving::ServingConfig;

const GEMMA4_E2B_GGUF_ENV: &str = "PROXIMA_GEMMA4_E2B_GGUF";
const DECODE_STEPS: usize = 24;
const ROW_NORM_RELATIVE_BOUND: f32 = 1e-3;

fn chat_prompt(user_turn: &str) -> String {
    format!("<|turn>user\n{user_turn}<turn|>\n<|turn>model\n")
}

fn arm_config(epilogue_sources: bool) -> ServingConfig<'static> {
    ServingConfig {
        numeric_policy: NumericPolicy::llama_relaxed().with_epilogue_sources(epilogue_sources),
        math_mode: MathMode::Relaxed,
        ..greedy_config()
    }
}

fn collect_logits(
    model: &LoadedModel<'_>,
    prompt: &str,
    serving_config: &ServingConfig<'_>,
) -> Vec<Vec<f32>> {
    let mut runtime = BackendRuntime::new(serving_config);
    let mut collected: Vec<Vec<f32>> = Vec::new();
    model
        .run_decode_loop_observed_seeded(
            prompt,
            DECODE_STEPS,
            serving_config,
            &mut runtime,
            None,
            &mut LogitsSink::Collect(&mut collected),
            &mut NodeValuesSink::Discard,
            &mut |_event| ControlFlow::Continue(()),
            None,
            false,
            None,
            None,
        )
        .expect("the gemma4-E2B checkpoint decodes");
    collected
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0)))
        .map(|(index, _)| index)
        .expect("a logits row is never empty")
}

struct StepDiff {
    max_abs: f32,
    row_norm_relative: f32,
    bit_diff_count: usize,
    argmax_matches: bool,
}

fn diff_step(unfused: &[f32], fused: &[f32]) -> StepDiff {
    let max_abs = unfused
        .iter()
        .zip(fused)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    let row_norm = unfused
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1e-6);
    StepDiff {
        max_abs,
        row_norm_relative: max_abs / row_norm,
        bit_diff_count: unfused
            .iter()
            .zip(fused)
            .filter(|(left, right)| left.to_bits() != right.to_bits())
            .count(),
        argmax_matches: argmax(unfused) == argmax(fused),
    }
}

struct Drift {
    worst_row_norm_relative: f32,
    steps: usize,
}

fn measure_drift() -> Drift {
    let path = std::env::var(GEMMA4_E2B_GGUF_ENV)
        .unwrap_or_else(|_| panic!("set {GEMMA4_E2B_GGUF_ENV} to the gemma4-E2B checkpoint path"));
    let file = File::open(&path).expect("open the gemma4-E2B checkpoint");
    // SAFETY: the checkpoint is not written or truncated while this read-only mapping lives.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the gemma4-E2B checkpoint");
    let parsed = parse_complete(&mapping).expect("parse the gemma4-E2B header");
    let model = LoadedModel::load(&parsed, &mapping).expect("bind the gemma4-E2B checkpoint");
    let prompts = [
        chat_prompt(
            "Which of these is smaller in size: a hippopotamus or a large office building?",
        ),
        chat_prompt("Which is bigger, an ant or a briefcase?"),
        "The capital of France is".to_string(),
    ];
    let unfused_config = arm_config(false);
    let fused_config = arm_config(true);

    let mut drift = Drift {
        worst_row_norm_relative: 0.0,
        steps: 0,
    };
    for prompt in &prompts {
        let unfused = collect_logits(&model, prompt, &unfused_config);
        let fused = collect_logits(&model, prompt, &fused_config);
        assert_eq!(
            unfused.len(),
            fused.len(),
            "both arms decode the same step count"
        );
        for (step, (left, right)) in unfused.iter().zip(&fused).enumerate() {
            let diff = diff_step(left, right);
            eprintln!(
                "epilogue_sources drift prompt={:?} step={step} max_abs={:e} \
                 row_norm_relative={:e} bit_diff={}/{} argmax_matches={}",
                &prompt[..prompt.len().min(40)],
                diff.max_abs,
                diff.row_norm_relative,
                diff.bit_diff_count,
                left.len(),
                diff.argmax_matches,
            );
            assert!(
                diff.argmax_matches,
                "step {step} argmax moved under epilogue_sources"
            );
            drift.worst_row_norm_relative =
                drift.worst_row_norm_relative.max(diff.row_norm_relative);
            drift.steps += 1;
        }
    }
    assert!(drift.steps > 0, "the decode loop produced no logits rows");
    drift
}

#[test]
#[ignore = "needs the host-local gemma4-E2B gguf and a Metal device"]
fn epilogue_sources_drift_under_relaxed_math_stays_inside_the_logit_bound() {
    let drift = measure_drift();

    assert!(
        drift.worst_row_norm_relative < ROW_NORM_RELATIVE_BOUND,
        "worst row-norm-relative logit drift {:e} exceeds {ROW_NORM_RELATIVE_BOUND:e}",
        drift.worst_row_norm_relative
    );
}
