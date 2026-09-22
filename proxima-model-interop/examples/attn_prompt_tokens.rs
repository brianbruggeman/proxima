//! Prompt-token counter for the gemma4-E2B checkpoint, so the coverage
//! campaign can hand the GPU slice exact prompt strings instead of having
//! it burn budget searching for lengths -- reads one prompt per run from
//! stdin, prints its token count under the SAME convention
//! `run_decode_loop_observed_seeded` (`generate/decode.rs`) uses for
//! `cached_len`: `proxima_tokenizer::encode_with_bos_eos`, BOS requested
//! per `wants_bos` (`generate/residency_caches.rs`'s own doc: `vocab
//! .add_bos_token().unwrap_or_else(|| vocab.bos_token_id().is_some())`),
//! EOS per `vocab.add_eos_token().unwrap_or(false)` -- NOT the plain
//! `proxima_tokenizer::encode` the other CPU examples log for a human
//! (`decode_gbps_baseline.rs`/`attn_parity_probe.rs` never add BOS to the
//! number they print).
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::fs::File;
use std::io::Read;

use memmap2::{Mmap, MmapOptions};
use proxima_gguf::parse_complete;
use proxima_tokenizer::vocab::Vocab;

const MODEL_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/\
sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

// mirrors `proxima-model-interop::generate::residency_caches::wants_bos`,
// which is crate-private and unreachable from an example binary
fn wants_bos(vocab: &Vocab) -> bool {
    vocab
        .add_bos_token()
        .unwrap_or_else(|| vocab.bos_token_id().is_some())
}

fn main() {
    let mut prompt = String::new();
    std::io::stdin()
        .read_to_string(&mut prompt)
        .expect("read prompt from stdin");
    let prompt = prompt.trim_end_matches('\n');

    let file = File::open(MODEL_PATH).expect("open gemma4-E2B blob");
    // SAFETY: `file` is dropped at the end of this scope, but the mapping
    // stays valid past that -- POSIX `mmap`/`munmap` semantics, same
    // pattern every real-checkpoint example in this crate already uses.
    let bytes: Mmap = unsafe { MmapOptions::new().map(&file) }.expect("map gemma4-E2B blob");
    let parsed = parse_complete(&bytes).expect("parse gemma4-E2B header");
    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed)
        .expect("builds vocab via the current gemma4 dispatch");

    let add_bos = wants_bos(&vocab);
    let add_eos = vocab.add_eos_token().unwrap_or(false);
    let ids = proxima_tokenizer::encode_with_bos_eos(prompt, &vocab, add_bos, add_eos)
        .expect("tokenize prompt under the cached_len convention");

    let first_six: Vec<u32> = ids.iter().take(6).copied().collect();
    println!("count={} add_bos={add_bos} add_eos={add_eos} first_six_ids={first_six:?}", ids.len());
}
