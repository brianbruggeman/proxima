//! Emit a low-codec HOBBIT sidecar for the real qwen35moe checkpoint.

use std::env;
use std::error::Error;
use std::fs::File;
use std::io::BufWriter;

use memmap2::Mmap;
use proxima_gguf::pipe::parse_complete;
use proxima_model_interop::{ExpertStackSpec, PackedOwnedKind, write_expert_sidecar};

fn main() -> Result<(), Box<dyn Error>> {
    let model_path = env::args()
        .nth(1)
        .ok_or("usage: write_qwen35_sidecar <gguf> <output>")?;
    let output_path = env::args()
        .nth(2)
        .ok_or("usage: write_qwen35_sidecar <gguf> <output>")?;
    let file = File::open(model_path)?;
    // SAFETY: the mapping and its file remain alive for the complete write.
    let mapping = unsafe { Mmap::map(&file)? };
    let parsed = parse_complete(&mapping)?;
    let mut tensor_names = Vec::with_capacity(40 * 3);
    for layer in 0..40_u32 {
        tensor_names.push(format!("blk.{layer}.ffn_gate_exps.weight"));
        tensor_names.push(format!("blk.{layer}.ffn_up_exps.weight"));
        tensor_names.push(format!("blk.{layer}.ffn_down_exps.weight"));
    }
    let mut specifications = Vec::with_capacity(40 * 3);
    for layer in 0..40_u32 {
        let base = layer as usize * 3;
        specifications.push(ExpertStackSpec {
            layer,
            projection: "ffn_gate",
            tensor_name: tensor_names[base].as_str(),
            expert_count: 256,
            out_dim: 512,
            in_dim: 2048,
            target_codec: PackedOwnedKind::Q3K,
        });
        specifications.push(ExpertStackSpec {
            layer,
            projection: "ffn_up",
            tensor_name: tensor_names[base + 1].as_str(),
            expert_count: 256,
            out_dim: 512,
            in_dim: 2048,
            target_codec: PackedOwnedKind::Q3K,
        });
        specifications.push(ExpertStackSpec {
            layer,
            projection: "ffn_down",
            tensor_name: tensor_names[base + 2].as_str(),
            expert_count: 256,
            out_dim: 2048,
            in_dim: 512,
            target_codec: PackedOwnedKind::Q4K,
        });
    }
    let maximum_elements = 2048 * 512;
    let mut scratch = vec![0.0_f32; maximum_elements];
    let mut output = vec![0_u8; maximum_elements];
    let destination_file = File::create(output_path)?;
    let mut destination = BufWriter::new(destination_file);
    let sidecar = write_expert_sidecar(
        &parsed,
        &mapping,
        &specifications,
        PackedOwnedKind::Q3K,
        &mut scratch,
        &mut output,
        &mut destination,
    )?;
    println!("descriptors = {}", sidecar.descriptors.len());
    println!("bytes = {}", sidecar.total_bytes);
    Ok(())
}
