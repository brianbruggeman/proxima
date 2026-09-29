//! Closes the residual `PROXIMA_LOGITS_DIAG` (`src/generate/decode.rs:5019,
//! 5207-5212`) left open: that diagnostic FNV-hashes the step-0 logits
//! vector and found the four `PROXIMA_TILED_GEMM_{Q4_0,DENSE,WIDE_ACT_LOAD,
//! SLIM_TGMEM}` switches (default ON since `omega/Cargo.toml`'s own
//! `metal` feature list, `omega/Cargo.toml:189`) change the hash from decode
//! step 0 on every prompt whose prefill is `>= TILED_GEMM_MIN_TOKENS`
//! (`omega/omega-runtime.toml:51`, value 8) -- argmax never changed,
//! argmax-element diff measured 1e-5..4e-5, but the FULL-VECTOR max-abs diff
//! across all 262144 vocab elements was never measured. This file measures
//! it, for gemma4-E2B on the real checkpoint, via the same one-shot forward
//! [`LoadedModel::forward_logits_on_backend`] `examples/compare_local.rs`
//! (`:257-286`) already uses for CPU-vs-Metal logit comparison, applied here
//! to Metal-defaults-vs-Metal-all-off instead.
//!
//! Both arms and a CPU f32 oracle run inside ONE process on the SAME loaded
//! checkpoint: the four switches are read fresh via `std::env::var` on every
//! call (`omega/src/msl/kernel_types_identity.rs:1828,1868,1906,1941` --
//! none of the four cache the decision itself, only `slim_tgmem`'s own debug
//! log line is memoized via `OnceLock`, which is logging-only and does not
//! gate `active`), and the compiled-kernel cache key folds each switch's
//! state into a distinct suffix (`_tgq0`/`_dbg`/`_wal`/`_slim`,
//! `omega/src/identity.rs:746,762,772,774`), so flipping the four env vars
//! with `temp_env::with_vars` between calls exercises two genuinely
//! different kernel selections rather than one cached compilation replayed
//! twice.
//!
//! The CPU oracle (`forward_logits_on_backend(prompt, 0)`) is unaffected by
//! any of the four switches -- they only gate Metal kernel-emission
//! admission (`omega/src/msl/kernel_types_identity.rs`) -- so it is a fixed
//! third point each Metal arm is also checked against, closing this file's
//! own "compare both arms against an f32 CPU oracle" requirement with the
//! SAME oracle path `compare_local.rs` already established, not a new one.
//! Measured to only be practical at this file's own time budget for the
//! shortest (5-token) prompt -- this crate's CPU forward path is not
//! BLAS-accelerated, and a 26-token real-checkpoint CPU oracle alone ran
//! past 5 CPU-minutes with no output (see the per-prompt comment where the
//! oracle is invoked); it is skipped for the two longer prompts, which
//! still get the full defaults-vs-all-off Metal comparison.
//!
//! Skips (does not fail) when the real blob is not present on this host --
//! same posture as `gemma4_correctness_gate.rs`.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::File;

use memmap2::Mmap;
use proxima_gguf::parse_complete;
use proxima_model_interop::{GPU_LAYERS_ALL, LoadedModel};

const REAL_GEMMA4_E2B_GGUF_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

fn results_dir() -> std::path::PathBuf {
    std::env::temp_dir().join("proxima-gemma4-tiled-gemm-logits")
}

const TILED_GEMM_SWITCH_NAMES: [&str; 4] = [
    "PROXIMA_TILED_GEMM_Q4_0",
    "PROXIMA_TILED_GEMM_DENSE",
    "PROXIMA_TILED_GEMM_WIDE_ACT_LOAD",
    "PROXIMA_TILED_GEMM_SLIM_TGMEM",
];

/// The exact literal text `tests_all.rs`'s own long-prefill fixture already
/// uses (`proxima-model-interop/src/generate/tests_all.rs:2670`, real
/// Sherlock Holmes prose, per guiding-principles #9's "real-world data in
/// tests" -- reused verbatim so this file adds no second long-prompt
/// corpus), repeated to approximate the discipline log's own "prompt5"
/// (510 real tokens, `docs/model-interop/c4-7-reduction-literal.md:213`)
/// without hard-coding a token count this crate's own tokenizer might not
/// reproduce exactly for a different (Sherlock-Holmes vs whatever prompt5's
/// own corpus used) text.
const LONG_PREFIX: &str = "To Sherlock Holmes she is always THE woman. I have seldom heard him mention her under any other name. In his eyes she eclipses and predominates the whole of her sex. It was not that he felt any emotion akin to love for Irene Adler. All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind. He was, I take it, the most perfect reasoning and observing machine that the world has seen, but as a lover he would have placed himself in a false position. He never spoke of the softer passions, save with a gibe and a sneer. They were admirable things for the observer\u{2014}excellent for drawing the veil from men's motives and actions. But for the trained reasoner to admit such intrusions into his own delicate and finely adjusted temperament was to introduce a distracting factor which might throw a doubt upon all his mental results.\n";

fn chat_prompt(user_turn: &str) -> String {
    format!("<|turn>user\n{user_turn}<turn|>\n<|turn>model\n")
}

/// Repeats [`LONG_PREFIX`] until this checkpoint's own tokenizer reports at
/// least 500 tokens (`proxima_tokenizer::encode`, the same public entry
/// point `examples/decode_gbps_baseline.rs:216` uses for prompt-length
/// accounting) -- measured against gemma4's real vocab rather than assumed
/// from a different tokenizer's token count for the same English text.
fn long_prompt_near_510_tokens(vocab: &proxima_tokenizer::Vocab) -> (String, usize) {
    for repeats in 1..=6usize {
        let prompt = LONG_PREFIX.repeat(repeats);
        let token_count = proxima_tokenizer::encode(&prompt, vocab)
            .expect("tokenize the long-prefill fixture with gemma4's real vocab")
            .len();
        if token_count >= 500 {
            return (prompt, token_count);
        }
    }
    let prompt = LONG_PREFIX.repeat(6);
    let token_count = proxima_tokenizer::encode(&prompt, vocab)
        .expect("tokenize the long-prefill fixture with gemma4's real vocab")
        .len();
    (prompt, token_count)
}

fn top5(logits: &[f32]) -> Vec<(usize, f32)> {
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked.truncate(5);
    ranked
}

struct LogitsDiff {
    max_abs: f32,
    mean_abs: f64,
    max_relative: f32,
    row_norm_relative: f32,
    bit_diff_count: usize,
    top5_order_matches: bool,
}

/// Two distinct relative metrics, both reused from existing precedent
/// rather than invented fresh (guiding-principles #14):
///
/// `max_relative` is per-element `|a-b| / max(|a|, |b|, 1e-6)`, the same
/// shape `q4_0_tiled_gemm_batched_run8_parity.rs:190-195` uses -- measured
/// here to explode on a real logits vector (most of the 262144 vocab
/// entries sit near zero, so a ~1e-5 absolute float32 noise floor divides
/// into a huge relative number there) even when the ranked, meaningful
/// entries agree; reported for completeness, NOT gated on.
///
/// `row_norm_relative` is `max|a-b| / ||a||_2` over the whole row --
/// `gemma4_program_metal_cpu_parity.rs`'s own `relative_error_at_last_position`
/// (`:228-244`), the established metric for THIS exact shape (one
/// `vocab_size`-wide logits row), immune to the near-zero-tail blowup
/// above because it normalizes by the row's overall scale, not each
/// element's own tiny magnitude. This is what this file's assertion gates
/// on, at that same file's own `1e-3` bound (`:313-317`).
fn diff_logits(arm_a: &[f32], arm_b: &[f32]) -> LogitsDiff {
    assert_eq!(arm_a.len(), arm_b.len(), "logits vector width mismatch");
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut max_relative = 0.0f32;
    let mut bit_diff_count = 0usize;
    for (left, right) in arm_a.iter().zip(arm_b.iter()) {
        let abs_diff = (left - right).abs();
        max_abs = max_abs.max(abs_diff);
        sum_abs += f64::from(abs_diff);
        if left.to_bits() != right.to_bits() {
            bit_diff_count += 1;
        }
        let denominator = left.abs().max(right.abs()).max(1e-6);
        max_relative = max_relative.max(abs_diff / denominator);
    }
    let row_norm: f32 = arm_a
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1e-6);
    let row_norm_relative = max_abs / row_norm;
    let top5_a = top5(arm_a);
    let top5_b = top5(arm_b);
    let top5_order_matches = top5_a
        .iter()
        .map(|(id, _)| *id)
        .eq(top5_b.iter().map(|(id, _)| *id));
    LogitsDiff {
        max_abs,
        mean_abs: sum_abs / arm_a.len() as f64,
        max_relative,
        row_norm_relative,
        bit_diff_count,
        top5_order_matches,
    }
}

fn format_top5(logits: &[f32]) -> String {
    top5(logits)
        .into_iter()
        .map(|(id, value)| format!("{id}:{value:.6}"))
        .collect::<Vec<_>>()
        .join(", ")
}

#[test]
fn gemma4_e2b_tiled_gemm_defaults_vs_all_off_full_logit_vector_diff() {
    let Ok(file) = File::open(REAL_GEMMA4_E2B_GGUF_PATH) else {
        eprintln!("skipping: real gemma4-E2B blob not found at {REAL_GEMMA4_E2B_GGUF_PATH}");
        return;
    };
    // SAFETY: the checkpoint file is not written or truncated by any other
    // process for the duration of this read-only mapping.
    let mapping = unsafe { Mmap::map(&file) }.expect("mmap the real gemma4-E2B checkpoint");
    let bytes: &[u8] = &mapping;
    let parsed = parse_complete(bytes).expect("parse the real gemma4-E2B checkpoint header");
    let model = LoadedModel::load(&parsed, bytes).expect("bind the real gemma4-E2B checkpoint");
    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed)
        .expect("builds vocab via the current gemma4 dispatch");

    let hippo_prompt = chat_prompt(
        "Which of these is smaller in size: a hippopotamus or a large office building?",
    );
    let (long_prompt, long_prompt_tokens) = long_prompt_near_510_tokens(&vocab);

    let prompts: Vec<(&str, String)> = vec![
        ("p1_capital_5tok", "The capital of France is".to_string()),
        ("p4_hippo_25tok", hippo_prompt),
        ("p5_long_510tok", long_prompt),
    ];

    let results_dir = results_dir();
    std::fs::create_dir_all(&results_dir).expect("create the logits results directory");

    let mut markdown = String::new();
    markdown.push_str("# gemma4-E2B tiled-GEMM defaults vs all-off: full logit-vector diff\n\n");
    markdown.push_str(&format!(
        "Method: `LoadedModel::forward_logits_on_backend` \
         (`proxima-model-interop/src/generate/decode.rs:6294`), one-shot \
         forward over the whole prompt, last-position logits (full \
         `vocab_size`-wide vector, see the per-prompt table's `bit_diff` \
         denominator below), all three arms run in one process on the SAME \
         mapped checkpoint via `temp_env::with_vars` toggling \
         {TILED_GEMM_SWITCH_NAMES:?}.\n\n",
    ));
    markdown.push_str(
        "Arms: `defaults` = all four switches unset (production default, \
         `metal` feature ON since `omega/Cargo.toml:189`); `all_off` = all \
         four switches set to `\"0\"`; `cpu_oracle` = \
         `forward_logits_on_backend(prompt, 0)`, an independent f32 CPU \
         evaluation unaffected by any of the four switches -- the same \
         oracle path `examples/compare_local.rs:257-266` already \
         establishes for CPU-vs-Metal comparison, reused here as the \
         third fixed point.\n\n",
    );
    markdown.push_str(
        "| prompt | tokens | defaults_vs_all_off max_abs | mean_abs | per-elem max_rel | row-norm rel | bit_diff/262144 | top5 match | argmax match | defaults_vs_cpu row-norm rel | all_off_vs_cpu row-norm rel |\n",
    );
    markdown.push_str("|---|---|---|---|---|---|---|---|---|---|---|\n");

    let mut csv = String::from(
        "prompt,tokens,arm_a_vs_arm_b_max_abs,mean_abs,max_relative_per_element,row_norm_relative,bit_diff_count,top5_order_matches,argmax_matches,defaults_vs_cpu_row_norm_relative,all_off_vs_cpu_row_norm_relative\n",
    );

    let mut worst_row_norm_relative_defaults_vs_off = 0.0f32;
    let mut every_argmax_matches = true;

    for (name, prompt) in &prompts {
        let token_count = if *name == "p5_long_510tok" {
            long_prompt_tokens
        } else {
            proxima_tokenizer::encode(prompt, &vocab)
                .expect("tokenize prompt for the results table")
                .len()
        };

        // `forward_logits_on_backend(_, 0)` runs the WHOLE dense f32 CPU
        // forward over every prompt position (`forward_node_values_on_backend`'s
        // own doc: "always a one-shot forward from an empty cache over the
        // WHOLE prompt") to produce even just the last row -- MEASURED on
        // this host: the 5-token prompt's CPU oracle completes in seconds,
        // but the 26-token prompt's CPU oracle alone ran past 5 minutes of
        // CPU time with no output yet (killed before completion; this f32
        // CPU path is not BLAS-accelerated the way the Metal path is, and
        // the real E2B checkpoint's per-layer width dwarfs the small
        // synthetic dims `gemma4_program_metal_cpu_parity.rs` CPU-checks).
        // The CPU oracle therefore only runs for the trivial 5-token
        // control prompt here, on the same time-budget grounds
        // `tests_all.rs:4267-4270` already names for its own real-checkpoint
        // p5/p6 prompts. Metal-vs-Metal (the primary measurement this file
        // exists for) still runs at every token count including 510.
        let cpu_oracle = if *name == "p1_capital_5tok" {
            Some(
                model
                    .forward_logits_on_backend(prompt, 0)
                    .unwrap_or_else(|error| panic!("{name}: CPU f32 oracle forward failed: {error}")),
            )
        } else {
            None
        };

        let unset_pairs: Vec<(&str, Option<&str>)> = TILED_GEMM_SWITCH_NAMES
            .iter()
            .map(|switch| (*switch, None))
            .collect();
        let defaults = temp_env::with_vars(unset_pairs, || {
            model
                .forward_logits_on_backend(prompt, GPU_LAYERS_ALL)
                .unwrap_or_else(|error| panic!("{name}: defaults-arm Metal forward failed: {error}"))
        });

        let off_pairs: Vec<(&str, Option<&str>)> = TILED_GEMM_SWITCH_NAMES
            .iter()
            .map(|switch| (*switch, Some("0")))
            .collect();
        let all_off = temp_env::with_vars(off_pairs, || {
            model
                .forward_logits_on_backend(prompt, GPU_LAYERS_ALL)
                .unwrap_or_else(|error| panic!("{name}: all-off-arm Metal forward failed: {error}"))
        });

        let defaults_vs_off = diff_logits(&defaults, &all_off);
        let defaults_vs_cpu = cpu_oracle.as_ref().map(|cpu| diff_logits(&defaults, cpu));
        let off_vs_cpu = cpu_oracle.as_ref().map(|cpu| diff_logits(&all_off, cpu));

        let argmax_defaults = top5(&defaults)[0].0;
        let argmax_off = top5(&all_off)[0].0;
        let argmax_matches = argmax_defaults == argmax_off;
        every_argmax_matches &= argmax_matches;
        worst_row_norm_relative_defaults_vs_off =
            worst_row_norm_relative_defaults_vs_off.max(defaults_vs_off.row_norm_relative);

        let defaults_vs_cpu_display = defaults_vs_cpu
            .as_ref()
            .map_or_else(|| "skipped(time budget)".to_string(), |diff| format!("{:e}", diff.row_norm_relative));
        let off_vs_cpu_display = off_vs_cpu
            .as_ref()
            .map_or_else(|| "skipped(time budget)".to_string(), |diff| format!("{:e}", diff.row_norm_relative));

        println!(
            "{name} tokens={token_count} defaults_vs_all_off max_abs={} mean_abs={} \
             max_relative_per_element={} row_norm_relative={} bit_diff_count={}/{} \
             top5_order_matches={} argmax_matches={} defaults_top5=[{}] all_off_top5=[{}] \
             defaults_vs_cpu_row_norm_relative={defaults_vs_cpu_display} \
             all_off_vs_cpu_row_norm_relative={off_vs_cpu_display}",
            defaults_vs_off.max_abs,
            defaults_vs_off.mean_abs,
            defaults_vs_off.max_relative,
            defaults_vs_off.row_norm_relative,
            defaults_vs_off.bit_diff_count,
            defaults.len(),
            defaults_vs_off.top5_order_matches,
            argmax_matches,
            format_top5(&defaults),
            format_top5(&all_off),
        );

        markdown.push_str(&format!(
            "| {name} | {token_count} | {:.6e} | {:.6e} | {:.6e} | {:.6e} | {}/{} | {} | {} | {defaults_vs_cpu_display} | {off_vs_cpu_display} |\n",
            defaults_vs_off.max_abs,
            defaults_vs_off.mean_abs,
            defaults_vs_off.max_relative,
            defaults_vs_off.row_norm_relative,
            defaults_vs_off.bit_diff_count,
            defaults.len(),
            defaults_vs_off.top5_order_matches,
            argmax_matches,
        ));
        csv.push_str(&format!(
            "{name},{token_count},{},{},{},{},{},{},{},{defaults_vs_cpu_display},{off_vs_cpu_display}\n",
            defaults_vs_off.max_abs,
            defaults_vs_off.mean_abs,
            defaults_vs_off.max_relative,
            defaults_vs_off.row_norm_relative,
            defaults_vs_off.bit_diff_count,
            defaults_vs_off.top5_order_matches,
            argmax_matches,
        ));

        markdown.push_str(&format!(
            "\ndefaults top-5 ({name}): {}\n\nall_off top-5 ({name}): {}\n\n",
            format_top5(&defaults),
            format_top5(&all_off),
        ));
    }

    markdown.push_str(&format!(
        "\nworst row-norm-relative error (defaults vs all_off) across all three prompts: {worst_row_norm_relative_defaults_vs_off:.6e}\n",
    ));
    markdown.push_str(
        "\nNo gemma4-specific full-vocab f32-oracle parity test existed prior to this file \
         (grepped `oracle` under `proxima-model-interop/tests/`: hits are the CPU-vs-Metal \
         `forward_logits_on_backend(_, 0)` path this file also uses, and the dense/Q4_0 \
         tiled-GEMM synthetic-dims parity suite in `omega/tests/`, neither of which runs the \
         real gemma4-E2B checkpoint end to end) -- the CPU oracle here is that same \
         `forward_logits_on_backend(_, 0)` path, applied to the real checkpoint for the first \
         time in this comparison.\n",
    );

    std::fs::write(results_dir.join("RESULTS.md"), &markdown).expect("write logits/RESULTS.md");
    std::fs::write(results_dir.join("summary.csv"), &csv).expect("write logits/summary.csv");

    assert!(
        every_argmax_matches,
        "tiled-gemm defaults vs all-off disagree on argmax for at least one prompt -- \
         PROXIMA_LOGITS_DIAG's own prior finding (argmax never changed) did not hold this run"
    );
    // `gemma4_program_metal_cpu_parity.rs:228-244,313-317` establishes this
    // exact metric (max-abs diff over a `vocab_size`-wide logits row,
    // normalized by that row's own L2 norm) and its `1e-3` acceptance bound
    // for gemma4's own dual-RoPE forward program on Metal vs CPU; reused
    // here unchanged for defaults-vs-all-off (two Metal arms differing only
    // in kernel selection).
    assert!(
        worst_row_norm_relative_defaults_vs_off < 1e-3,
        "tiled-gemm defaults vs all-off full-logit-vector row_norm_relative={worst_row_norm_relative_defaults_vs_off:e} \
         exceeds the 1e-3 bound `gemma4_program_metal_cpu_parity.rs` establishes for this shape"
    );
}
