//! Print the checkpoint-owned tokenizer result for a raw prompt.
//!
//! This is an evidence probe, not an inference path.  It makes prompt
//! rendering and BOS/EOS policy visible before logits are compared with an
//! independent runtime.
//!
//! Usage:
//! `cargo run -p proxima-model-interop --example tokenize_local --features std -- \
//!   /path/to/model.gguf "What is the capital of France?"`

#![allow(clippy::expect_used)]

use std::env;
use std::fs::File;

use memmap2::{Mmap, MmapOptions};
use proxima_gguf::parse_complete;
use proxima_tokenizer::{decode, encode_with_bos_eos};

fn main() {
    let mut args = env::args().skip(1);
    let model_path = args.next().expect("model path");
    let prompt = args.next().expect("raw prompt");
    let add_bos = args.next().is_some_and(|value| value == "--add-bos");
    assert!(args.next().is_none(), "unexpected argument");

    let file = File::open(&model_path).expect("open model");
    // SAFETY: `file` remains alive while the read-only mapping is borrowed.
    let bytes: Mmap = unsafe { MmapOptions::new().map(&file) }.expect("map model");
    let parsed = parse_complete(&bytes).expect("parse model");
    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed)
        .expect("build checkpoint tokenizer");
    let ids = encode_with_bos_eos(&prompt, &vocab, add_bos, false).expect("tokenize prompt");
    let roundtrip = decode(&ids, &vocab).expect("decode token IDs");
    let architecture = parsed
        .metadata
        .iter()
        .find(|(key, _)| key == "general.architecture")
        .map(|(_, value)| format!("{value:?}"));
    let execution_metadata: Vec<_> = parsed
        .metadata
        .iter()
        .filter(|(key, _)| {
            key.contains("attention")
                || key.contains("rope")
                || key.ends_with("embedding_length")
                || key.ends_with("head_count")
        })
        .map(|(key, value)| (key.clone(), format!("{value:?}")))
        .collect();
    let architecture_json = serde_json::to_string(&architecture).expect("serialize architecture");
    let metadata_json = serde_json::to_string(&execution_metadata).expect("serialize metadata");
    let model_json = serde_json::to_string(&model_path).expect("serialize model path");
    let prompt_json = serde_json::to_string(&prompt).expect("serialize prompt");
    let roundtrip_json = serde_json::to_string(&roundtrip).expect("serialize roundtrip");

    println!(
        "{{\"model\":{model_json},\"bytes\":{},\"architecture\":{architecture_json},\"execution_metadata\":{metadata_json},\"add_bos\":{add_bos},\"prompt\":{prompt_json},\"ids\":{},\"roundtrip\":{roundtrip_json}}}",
        bytes.len(),
        format_ids(&ids),
    );
}

fn format_ids(ids: &[u32]) -> String {
    let mut output = String::from("[");
    for (index, id) in ids.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        output.push_str(&id.to_string());
    }
    output.push(']');
    output
}
