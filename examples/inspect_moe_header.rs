//! Header-only dump of qwen3moe hparams and the raw GGUF `dims` order for
//! one layer's `_exps` tensors, plus `ffn_gate_inp`. Used to root-cause
//! `InteropError::MoeExpertShapeMismatch` at bind -- reads only the
//! directory, never the tensor payload, via `proxima_gguf::pipe::parse_complete`.
//!
//! Run: `cargo run --example inspect_moe_header -- <gguf-path>`

use std::env;
use std::error::Error;
use std::fs::File;

use memmap2::Mmap;
use proxima_gguf::pipe::parse_complete;
use proxima_gguf::value::MetadataValue;

fn print_metadata_u32(parsed: &proxima_gguf::pipe::ParsedGguf, key: &str) {
    let value = parsed.metadata_value(key);
    match value {
        Some(MetadataValue::U32(inner)) => println!("{key} = {inner}"),
        Some(MetadataValue::U64(inner)) => println!("{key} = {inner}"),
        Some(other) => println!("{key} = {other:?} (unexpected type)"),
        None => println!("{key} = <missing>"),
    }
}

fn print_tensor(parsed: &proxima_gguf::pipe::ParsedGguf, name: &str) {
    match parsed.tensors.iter().find(|tensor| tensor.name == name) {
        Some(tensor) => println!(
            "{name}: dims={:?} ggml_type={:?} element_count={}",
            tensor.dims,
            tensor.ggml_type,
            tensor.element_count(),
        ),
        None => println!("{name}: <missing>"),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = env::args()
        .nth(1)
        .ok_or("usage: inspect_moe_header <gguf-path>")?;
    let file = File::open(&path)?;
    // SAFETY: read-only mapping of a file we hold open for the mmap's lifetime.
    let mmap = unsafe { Mmap::map(&file) }?;
    let parsed = parse_complete(&mmap)?;

    println!(
        "general.architecture = {:?}",
        parsed.metadata_value("general.architecture")
    );
    for key in [
        "qwen3moe.expert_count",
        "qwen3moe.expert_used_count",
        "qwen3moe.expert_feed_forward_length",
        "qwen3moe.feed_forward_length",
        "qwen3moe.embedding_length",
        "qwen3moe.block_count",
    ] {
        print_metadata_u32(&parsed, key);
    }
    println!(
        "qwen3moe.expert_weights_norm = {:?}",
        parsed.metadata_value("qwen3moe.expert_weights_norm")
    );
    println!(
        "qwen3moe.expert_gating_func = {:?}",
        parsed.metadata_value("qwen3moe.expert_gating_func")
    );
    println!(
        "qwen3moe.expert_weights_scale = {:?}",
        parsed.metadata_value("qwen3moe.expert_weights_scale")
    );

    for name in [
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
        "blk.0.ffn_gate_inp.weight",
        "blk.0.ssm_out.weight",
    ] {
        print_tensor(&parsed, name);
    }

    let mut expert_bytes = 0_u64;
    let mut dense_bytes = 0_u64;
    for tensor in &parsed.tensors {
        let range = parsed.tensor_data_range(tensor, mmap.len() as u64)?;
        if tensor.name.contains("_exps.weight") {
            expert_bytes += (range.end - range.start) as u64;
        } else {
            dense_bytes += (range.end - range.start) as u64;
        }
    }
    println!("checkpoint_expert_payload_bytes = {expert_bytes}");
    println!("checkpoint_nonexpert_payload_bytes = {dense_bytes}");

    Ok(())
}
