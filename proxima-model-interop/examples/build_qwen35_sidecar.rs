//! Build a Q2_K HOBBIT sidecar beside a Qwen35 GGUF without materializing the
//! checkpoint. The source remains mmap-backed and one expert is recoded at a
//! time.

use std::env;
use std::fs::File;
use std::io::BufWriter;

use memmap2::MmapOptions;
use proxima_gguf::parse_complete;
use proxima_model_interop::{ExpertStackSpec, PackedOwnedKind, write_expert_sidecar};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args().skip(1);
    let input_path = arguments.next().ok_or("argv[1]: qwen35 gguf path")?;
    let output_path = arguments.next().ok_or("argv[2]: sidecar output path")?;
    let input = File::open(&input_path)?;
    let mapping = unsafe { MmapOptions::new().map(&input)? };
    let parsed = parse_complete(&mapping)?;

    let codec_name = env::var("PROXIMA_SIDECAR_CODEC").unwrap_or_else(|_| String::from("q2"));
    let target_codec = match codec_name.as_str() {
        "q2_k" | "q2" => PackedOwnedKind::Q2K,
        "mixed" => PackedOwnedKind::Q4K,
        "q4_k" | "q4" => PackedOwnedKind::Q4K,
        "q6_k" | "q6" => PackedOwnedKind::Q6K,
        other => return Err(format!("unsupported PROXIMA_SIDECAR_CODEC={other:?}").into()),
    };
    let mut specifications = Vec::new();
    for layer in 0..parsed
        .metadata_value("qwen3moe.block_count")
        .and_then(|value| value.as_u32())
        .unwrap_or(40)
    {
        for projection in ["ffn_gate", "ffn_up", "ffn_down"] {
            let tensor_name = format!("blk.{layer}.{projection}_exps.weight");
            let Some(tensor) = parsed
                .tensors
                .iter()
                .find(|tensor| tensor.name == tensor_name)
            else {
                continue;
            };
            let expert_count = *tensor
                .dims
                .last()
                .ok_or("expert tensor has no expert axis")? as u32;
            let in_dim = tensor.dims[0] as u32;
            let out_dim = tensor.dims[1] as u32;
            let projection_codec = if codec_name == "mixed" && projection == "ffn_down" {
                PackedOwnedKind::Q6K
            } else {
                target_codec
            };
            specifications.push(ExpertStackSpec {
                layer,
                projection,
                tensor_name: &tensor.name,
                expert_count,
                out_dim,
                in_dim,
                target_codec: projection_codec,
            });
        }
    }
    if specifications.is_empty() {
        return Err("no Qwen35 expert stacks found".into());
    }
    let max_elements = specifications
        .iter()
        .map(|spec| spec.out_dim as usize * spec.in_dim as usize)
        .max()
        .ok_or("no expert dimensions")?;
    let mut scratch = vec![0.0_f32; max_elements];
    let max_output = specifications
        .iter()
        .map(|spec| {
            let layout = match spec.target_codec {
                PackedOwnedKind::Q2K => proxima_gguf::types::GgmlType::Q2_K.block_layout(),
                PackedOwnedKind::Q4K => proxima_gguf::types::GgmlType::Q4_K.block_layout(),
                PackedOwnedKind::Q6K => proxima_gguf::types::GgmlType::Q6_K.block_layout(),
                _ => unreachable!("target codec is selected from the three packed codecs"),
            };
            (spec.out_dim as usize * spec.in_dim as usize / layout.block_elements as usize)
                * layout.block_bytes as usize
        })
        .max()
        .ok_or("no output dimensions")?;
    let mut output = vec![0_u8; max_output];
    let mut destination = BufWriter::new(File::create(output_path)?);
    let sidecar = write_expert_sidecar(
        &parsed,
        &mapping,
        &specifications,
        target_codec,
        &mut scratch[..max_elements],
        &mut output[..max_output],
        &mut destination,
    )?;
    println!(
        "expert_sidecar descriptors={} bytes={}",
        sidecar.descriptors.len(),
        sidecar.total_bytes
    );
    Ok(())
}
