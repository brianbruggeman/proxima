//! Prints how many tensors a GGUF file holds per storage type, and the layer-0 tensors by name.
//!
//! Usage: `gguf_codec_inventory <model.gguf>`
//!
//! The per-type counts are the codec list a backend must decode to run the model; the
//! `blk.0.` rows name which weight sits behind each type.

#![allow(clippy::expect_used)]

use std::collections::BTreeMap;
use std::env;
use std::fs::File;

use memmap2::MmapOptions;
use proxima_gguf::parse_complete;

fn main() {
    let model_path = env::args().nth(1).expect("model path");
    let file = File::open(&model_path).expect("open model");
    // SAFETY: `file` remains alive while the read-only mapping is borrowed.
    let bytes = unsafe { MmapOptions::new().map(&file) }.expect("map model");
    let parsed = parse_complete(&bytes).expect("parse model");
    let mut by_type: BTreeMap<String, usize> = BTreeMap::new();
    for tensor in &parsed.tensors {
        *by_type.entry(format!("{:?}", tensor.ggml_type)).or_default() += 1;
    }
    for (ggml_type, count) in &by_type {
        println!("{{\"type\":{ggml_type:?},\"tensors\":{count}}}");
    }
    for tensor in parsed.tensors.iter().filter(|tensor| tensor.name.starts_with("blk.0.") || !tensor.name.starts_with("blk.")) {
        println!(
            "{{\"name\":{:?},\"type\":\"{:?}\",\"dims\":{:?}}}",
            tensor.name, tensor.ggml_type, tensor.dims
        );
    }
}
