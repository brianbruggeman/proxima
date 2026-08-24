// a diagnostic binary, not library surface: every `.expect()` below is a
// setup precondition (real checkpoint present, oracle dump present, model
// loads) whose only correct response is to panic with the failing step
// named, matching this crate's own sibling examples (`any_listener.rs` and
// friends carry the identical allow for the identical reason).
#![allow(clippy::expect_used)]

//! Cross-oracle logit diff for the real, downloaded SmolLM2-135M-Instruct
//! checkpoint: this crate's own [`LoadedModel::forward_logits`] against a
//! raw `f32` dump of `llama.cpp`'s `llama_get_logits_ith` for the identical
//! (BOS-forced) token sequence, evaluated at the same position.
//!
//! The oracle dump is produced OUTSIDE this repo (a small `llama.h`-linked
//! C++ program built against the real `~/repos/others/llama.cpp` checkout,
//! not committed here) -- this example's job is only the comparison, on
//! real execution data from both sides, never a synthetic fixture. Skips
//! cleanly (prints why, exits 0) when either input is absent, matching this
//! crate's own `tests/real_smollm2_checkpoint.rs` convention for a
//! host-local-only artifact.
//!
//! A decoded token is an argmax and destroys exactly the information this
//! tool needs: whether a divergence is a near-tied bf16-vs-f16 rounding gap
//! or a gross defect. Findings are emitted as structured `debug!`/`warn!`
//! records into a file-sink `Exporter` (never a hand-rolled dump), the same
//! pattern `proxima-tensor/examples/q4k_ggml_fidelity.rs` established.

use std::env;
use std::fs;
use std::path::PathBuf;

use proxima_model_interop::{LoadedModel, architecture_from_hf_config, parse_hf_config};
use proxima_telemetry::export::{Exporter, Formatter};
use proxima_telemetry::level::Level;
use proxima_telemetry::recorder::Recorder;

/// One vocab row's rank in a logit vector, sorted by [`f32::total_cmp`] with
/// the token id as tiebreaker -- hash-iteration nondeterminism has no way in
/// here since this only ever sorts a plain, already-materialized `Vec`, but
/// a raw `partial_cmp` on ties would still make "which token is #1" flicker
/// between runs when two logits land bit-identical (a real occurrence: see
/// this run's own top-10 table).
fn ranked_indices(logits: &[f32]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|left, right| logits[*right].total_cmp(&logits[*left]).then_with(|| left.cmp(right)));
    indices
}

fn read_oracle_logits(path: &PathBuf) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read oracle logits at {path:?}: {error}"));
    bytes.as_chunks::<4>().0.iter().map(|chunk| f32::from_le_bytes(*chunk)).collect()
}

fn main() {
    let model_dir = env::args().nth(1).unwrap_or_else(|| {
        "/Users/brianbruggeman/.lmstudio/models/HuggingFaceTB/SmolLM2-135M-Instruct".to_string()
    });
    let oracle_path = env::args().nth(2).unwrap_or_else(|| {
        "/private/tmp/claude-501/-Users-brianbruggeman-repos-slot-0/6cd9e134-c1a3-450a-be93-76dd95389bf4/scratchpad/diverge/dump_f16/final_logits.f32".to_string()
    });
    let prompt = env::args().nth(3).unwrap_or_else(|| "The capital of France is the".to_string());
    let log_path = env::args().nth(4).unwrap_or_else(|| "smollm2_logit_diff.jsonl".to_string());

    let model_path = PathBuf::from(&model_dir).join("model.safetensors");
    let oracle_path = PathBuf::from(&oracle_path);
    if !model_path.exists() {
        println!("skipping: no host-local smollm2 safetensors checkpoint at {model_path:?}");
        return;
    }
    if !oracle_path.exists() {
        println!("skipping: no oracle logit dump at {oracle_path:?}");
        return;
    }

    let recorder = Recorder::builder()
        .export(Exporter::file(&log_path).format(Formatter::Text))
        .expect("file exporter")
        .install()
        .expect("recorder");

    let config_bytes = fs::read(PathBuf::from(&model_dir).join("config.json")).expect("read config.json");
    let hf_config = parse_hf_config(&config_bytes).expect("parse config.json");
    let architecture = architecture_from_hf_config(&hf_config);

    let tokenizer_bytes = fs::read(PathBuf::from(&model_dir).join("tokenizer.json")).expect("read tokenizer.json");
    let vocab = proxima_tokenizer::hf::vocab_from_tokenizer_json(&tokenizer_bytes, Some(1), None, None)
        .expect("build vocab from tokenizer.json");

    let file_bytes = fs::read(&model_path).expect("read model.safetensors");
    let manifest = proxima_safetensors::parse_complete(&file_bytes).expect("parse model.safetensors");
    let mut length_prefix = [0u8; 8];
    length_prefix.copy_from_slice(&file_bytes[..8]);
    let data_start = 8 + u64::from_le_bytes(length_prefix);

    let model = LoadedModel::load_from_safetensors(&manifest, &file_bytes, data_start, architecture, vocab)
        .expect("load real smollm2 checkpoint");

    let ours = model.forward_logits(&prompt).expect("compute our own forward logits");
    let theirs = read_oracle_logits(&oracle_path);

    assert_eq!(ours.len(), theirs.len(), "vocab size mismatch between our logits and the oracle dump");

    let our_ranked = ranked_indices(&ours);
    let their_ranked = ranked_indices(&theirs);

    let mut max_abs_diff = 0f32;
    let mut worst_index = 0usize;
    for index in 0..ours.len() {
        let diff = (ours[index] - theirs[index]).abs();
        if diff > max_abs_diff {
            max_abs_diff = diff;
            worst_index = index;
        }
    }

    let argmax_matches = our_ranked[0] == their_ranked[0];
    let our_top1 = ours[our_ranked[0]];
    let our_top2 = ours[our_ranked[1]];
    let their_top1 = theirs[their_ranked[0]];
    let their_top2 = theirs[their_ranked[1]];
    let our_top1_top2_gap = our_top1 - our_top2;

    if argmax_matches && max_abs_diff < 1e-1 {
        recorder
            .log()
            .level(Level::DEBUG)
            .message("our logits match the oracle within precision-scale tolerance")
            .module_path(module_path!())
            .tag("argmax_matches", argmax_matches)
            .tag("max_abs_diff", f64::from(max_abs_diff))
            .tag("worst_index", worst_index as u64)
            .emit();
    } else {
        recorder
            .log()
            .level(Level::WARN)
            .message("our logits diverge from the oracle beyond precision-scale tolerance")
            .module_path(module_path!())
            .tag("argmax_matches", argmax_matches)
            .tag("max_abs_diff", f64::from(max_abs_diff))
            .tag("worst_index", worst_index as u64)
            .tag("our_top1_token", our_ranked[0] as u64)
            .tag("their_top1_token", their_ranked[0] as u64)
            .emit();
    }
    while recorder.drain() > 0 {}

    println!("prompt={prompt:?} vocab_size={}", ours.len());
    println!("argmax_matches={argmax_matches} max_abs_diff={max_abs_diff:e} worst_index={worst_index}");
    println!(
        "our_top1_token={} our_top1_logit={:.6} our_top2_logit={:.6} our_top1_top2_gap={:.6}",
        our_ranked[0], our_top1, our_top2, our_top1_top2_gap
    );
    println!("their_top1_token={} their_top1_logit={:.6} their_top2_logit={:.6}", their_ranked[0], their_top1, their_top2);
    println!("ours[worst_index]={:.6} theirs[worst_index]={:.6}", ours[worst_index], theirs[worst_index]);

    println!("\nrank ours_token ours_logit theirs_token theirs_logit");
    for rank in 0..10 {
        println!(
            "{rank} {} {:.6} {} {:.6}",
            our_ranked[rank], ours[our_ranked[rank]], their_ranked[rank], theirs[their_ranked[rank]]
        );
    }
}
