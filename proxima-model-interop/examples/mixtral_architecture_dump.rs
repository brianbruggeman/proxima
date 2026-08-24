//! Prints the real Mixtral checkpoint's derived [`ModelArchitecture`] and
//! flags whether `blk.0.ffn_gate_exps.weight` (the native stacked layout
//! [`bind::bind_moe_stacked_experts`] can bind zero-copy) exists, or only
//! the per-expert-tensor layout does -- the shape that decides whether the
//! MoE FFN binds packed or must dequantize every expert to owned `f32`.
//!
//! Scratch tool for sizing the real forward pass, not part of the crate's
//! public surface.

use std::env;
use std::fs;

use proxima_gguf::pipe::parse_complete;
use proxima_model_interop::architecture_from_metadata;

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| {
        "/Users/brianbruggeman/.lmstudio/models/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF/Nous-Hermes-2-Mixtral-8x7B-DPO.Q4_K_S.gguf".to_string()
    });
    let file_bytes = fs::read(&path).expect("read mixtral gguf");
    let parsed = parse_complete(&file_bytes).expect("parse mixtral gguf");
    let architecture = architecture_from_metadata(&parsed).expect("derive architecture");
    println!("{architecture:#?}");

    let stacked_present = parsed.tensors.iter().any(|tensor| tensor.name == "blk.0.ffn_gate_exps.weight");
    println!("blk.0.ffn_gate_exps.weight present={stacked_present}");
    for tensor in &parsed.tensors {
        if tensor.name.starts_with("blk.0.ffn_gate") || tensor.name.starts_with("blk.0.ffn_up") || tensor.name.starts_with("blk.0.ffn_down") {
            println!("{} type={:?} dims={:?}", tensor.name, tensor.ggml_type, tensor.dims);
        }
    }
}
