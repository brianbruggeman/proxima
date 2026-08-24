//! Cross-oracle logit diff for the real, downloaded
//! `LFM2.5-8B-A1B-Q4_K_M.gguf` checkpoint: this crate's own
//! [`lfm2_forward_values`] against a raw `f32` dump of `llama.cpp`'s
//! `llama_get_logits_ith` for the identical (BOS-forced) token sequence --
//! [`smollm2_logit_oracle_diff`]'s own shape, reused rather than
//! reinvented, adjusted only for this checkpoint's GGUF-bind, hybrid
//! (conv/attention/MoE) forward path in place of the dense safetensors one.
//!
//! The oracle dump is produced OUTSIDE this repo by a small `llama.h`-linked
//! C++ probe (`oracle-dump`, built against a current llama.cpp checkout that
//! actually knows the `lfm2moe` architecture -- the repo-committed
//! `~/repos/others/llama.cpp` checkout at the time this was written does
//! not), not committed here -- same convention as the smollm2 probe.
//! Skips cleanly (prints why, exits 0) when either input is absent.
//!
//! Compares logits, never tokens: a decoded token is an argmax and destroys
//! exactly the information this tool needs to tell a near-tied rounding gap
//! from a gross defect.

use std::env;
use std::fs;
use std::path::PathBuf;

use proxima_gguf::pipe::parse_complete;
use proxima_model_interop::{Lfm2Architecture, lfm2_architecture_from_metadata, lfm2_forward_values};
use proxima_telemetry::export::{Exporter, Formatter};
use proxima_telemetry::level::Level;
use proxima_telemetry::recorder::Recorder;

/// One vocab row's rank in a logit vector, sorted by [`f32::total_cmp`] with
/// the token id as tiebreaker -- [`smollm2_logit_oracle_diff`]'s own helper,
/// unchanged: which token is #1 must not flicker between runs when two
/// logits land bit-identical.
fn ranked_indices(logits: &[f32]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|left, right| logits[*right].total_cmp(&logits[*left]).then_with(|| left.cmp(right)));
    indices
}

fn read_oracle_logits(path: &PathBuf) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|error| panic!("read oracle logits at {path:?}: {error}"));
    bytes.chunks_exact(4).map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect()
}

fn main() {
    let model_path = env::args().nth(1).unwrap_or_else(|| {
        "/Users/brianbruggeman/.lmstudio/models/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf".to_string()
    });
    let oracle_path = env::args().nth(2).unwrap_or_else(|| {
        "/private/tmp/claude-501/-Users-brianbruggeman-repos-slot-0/6cd9e134-c1a3-450a-be93-76dd95389bf4/scratchpad/oracle/dump_lfm2/final_logits.f32".to_string()
    });
    let prompt = env::args().nth(3).unwrap_or_else(|| "The capital of France is".to_string());
    let log_path = env::args().nth(4).unwrap_or_else(|| "lfm2_logit_diff.jsonl".to_string());

    let model_path = PathBuf::from(&model_path);
    let oracle_path = PathBuf::from(&oracle_path);
    if !model_path.exists() {
        println!("skipping: no host-local lfm2 gguf checkpoint at {model_path:?}");
        return;
    }
    if !oracle_path.exists() {
        println!("skipping: no oracle logit dump at {oracle_path:?}");
        return;
    }

    let recorder = Recorder::builder().export(Exporter::file(&log_path).format(Formatter::Text)).expect("file exporter").install().expect("recorder");

    let file_bytes = fs::read(&model_path).expect("read lfm2 gguf checkpoint");
    let parsed = parse_complete(&file_bytes).expect("parse lfm2 gguf checkpoint");
    let architecture: Lfm2Architecture = lfm2_architecture_from_metadata(&parsed).expect("derive lfm2 architecture from gguf metadata");

    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed).expect("build vocab from gguf metadata");
    let add_bos = vocab.add_bos_token().unwrap_or(true);
    let ids = proxima_tokenizer::encode_with_bos_eos(&prompt, &vocab, add_bos, false).expect("tokenize prompt");

    let (ours, _extras) = lfm2_forward_values(&parsed, &file_bytes, &architecture, &ids, &[]).expect("compute our own forward logits");
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

    if argmax_matches && max_abs_diff < 1e-1 {
        recorder
            .log()
            .level(Level::DEBUG)
            .message("our lfm2 logits match the oracle within precision-scale tolerance")
            .module_path(module_path!())
            .tag("argmax_matches", argmax_matches)
            .tag("max_abs_diff", f64::from(max_abs_diff))
            .tag("worst_index", worst_index as u64)
            .emit();
    } else {
        recorder
            .log()
            .level(Level::WARN)
            .message("our lfm2 logits diverge from the oracle beyond precision-scale tolerance")
            .module_path(module_path!())
            .tag("argmax_matches", argmax_matches)
            .tag("max_abs_diff", f64::from(max_abs_diff))
            .tag("worst_index", worst_index as u64)
            .tag("our_top1_token", our_ranked[0] as u64)
            .tag("their_top1_token", their_ranked[0] as u64)
            .emit();
    }
    while recorder.drain() > 0 {}

    println!("prompt={prompt:?} ids={ids:?} vocab_size={}", ours.len());
    println!("argmax_matches={argmax_matches} max_abs_diff={max_abs_diff:e} worst_index={worst_index}");
    println!("our_top1_token={} our_top1_logit={:.6} our_top2_logit={:.6}", our_ranked[0], our_top1, our_top2);
    println!("their_top1_token={} their_top1_logit={:.6} their_top2_logit={:.6}", their_ranked[0], their_top1, their_top2);
    println!("ours[worst_index]={:.6} theirs[worst_index]={:.6}", ours[worst_index], theirs[worst_index]);

    println!("\nrank ours_token ours_logit theirs_token theirs_logit");
    for rank in 0..10 {
        println!("{rank} {} {:.6} {} {:.6}", our_ranked[rank], ours[our_ranked[rank]], their_ranked[rank], theirs[their_ranked[rank]]);
    }
}
