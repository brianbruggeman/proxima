// scratch diagnostic: dumps the real gemma4 checkpoint header for SLICE 1
// hparams/tensor-name verification. Not library surface.
#![allow(clippy::expect_used)]

use std::env;
use std::fs::File;

fn main() {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "/Users/brianbruggeman/.ollama/models/blobs/sha256-ea549b7688d4c95019754880c21e3f29c58c985a7a1c3b37b9eebd0a95224129".to_string());
    let file = File::open(&path).expect("open gemma4 gguf");
    let mapping = unsafe { memmap2::Mmap::map(&file) }.expect("mmap gemma4 gguf");
    let file_bytes: &[u8] = &mapping;
    let parsed = proxima_gguf::parse_complete(file_bytes).expect("parse gemma4 header");

    println!("metadata entries = {}", parsed.metadata.len());
    for (key, value) in &parsed.metadata {
        if key.starts_with("gemma4.") || key == "general.architecture" {
            match value {
                proxima_gguf::value::MetadataValue::Array(array) => {
                    println!("{key} = array[{}] first10={:?}", array.len(), array);
                }
                other => println!("{key} = {other:?}"),
            }
        }
    }

    println!("total tensors = {}", parsed.tensors.len());
    let mut layer0_names: Vec<&str> = parsed
        .tensors
        .iter()
        .filter(|tensor| tensor.name.starts_with("blk.0."))
        .map(|tensor| tensor.name.as_str())
        .collect();
    layer0_names.sort_unstable();
    println!(
        "blk.0.* tensors ({}): {:#?}",
        layer0_names.len(),
        layer0_names
    );

    let mut layer5_names: Vec<&str> = parsed
        .tensors
        .iter()
        .filter(|tensor| tensor.name.starts_with("blk.5."))
        .map(|tensor| tensor.name.as_str())
        .collect();
    layer5_names.sort_unstable();
    println!(
        "blk.5.* tensors ({}): {:#?}",
        layer5_names.len(),
        layer5_names
    );

    let global_names: Vec<&str> = parsed
        .tensors
        .iter()
        .filter(|tensor| !tensor.name.starts_with("blk."))
        .map(|tensor| tensor.name.as_str())
        .collect();
    println!(
        "global tensors ({}): {:#?}",
        global_names.len(),
        global_names
    );

    for name in [
        "blk.0.ffn_gate_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
        "blk.0.ffn_down_exps.scale",
        "blk.0.ffn_gate_inp.scale",
        "blk.0.ffn_gate_inp.weight",
        "token_embd.weight",
        "output_norm.weight",
    ] {
        let tensor = parsed
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .expect("axis-confirmation tensor present");
        let layout = tensor.ggml_type.block_layout();
        println!(
            "{name}: dims={:?} ggml_type={:?} block_elements={} block_bytes={}",
            tensor.dims, tensor.ggml_type, layout.block_elements, layout.block_bytes
        );
    }

    dump_fused_gate_up_layout(&parsed, file_bytes);
}

/// Independently dequantizes the real `blk.0.ffn_gate_up_exps.weight` (Q3_K)
/// and prints expert-0 row 0 / row 704 values plus a correlation probe
/// between candidate gate/up row pairings, so the packed-byte split
/// arithmetic in `bind_gemma4_fused_gate_up_experts` can be checked against
/// the ACTUAL on-disk layout rather than assumed from the GGUF dim order.
fn dump_fused_gate_up_layout(parsed: &proxima_gguf::ParsedGguf, file_bytes: &[u8]) {
    let name = "blk.0.ffn_gate_up_exps.weight";
    let tensor = parsed
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .expect("fused gate/up tensor present");
    let dims = &tensor.dims;
    println!("\n=== {name} layout probe ===");
    println!("dims={dims:?} ggml_type={:?}", tensor.ggml_type);

    let ne0 = dims[0] as usize; // 2816 embedding/contraction
    let ne1 = dims[1] as usize; // 1408 = 2 * expert_feed_forward
    let ne2 = dims[2] as usize; // 128 experts
    let expert_feed_forward = ne1 / 2;

    let layout = tensor.ggml_type.block_layout();
    let block_elements = layout.block_elements as usize;
    let block_bytes = layout.block_bytes as usize;
    let bytes_per_row = (ne0 / block_elements) * block_bytes;
    let per_expert_bytes = ne1 * bytes_per_row;

    println!(
        "ne0={ne0} ne1={ne1} ne2={ne2} expert_feed_forward={expert_feed_forward} \
         bytes_per_row={bytes_per_row} per_expert_bytes={per_expert_bytes}"
    );

    let range = parsed
        .tensor_data_range(tensor, file_bytes.len() as u64)
        .expect("tensor data range");
    let source = &file_bytes[range.start as usize..range.end as usize];
    println!(
        "source.len()={} expected={}",
        source.len(),
        per_expert_bytes * ne2
    );

    let mut row_f32 = vec![0.0f32; ne0];
    let dequant_row = |source: &[u8], expert: usize, row: usize, out: &mut [f32]| {
        let row_start = expert * per_expert_bytes + row * bytes_per_row;
        let row_bytes = &source[row_start..row_start + bytes_per_row];
        proxima_gguf::quant::q3_k::dequantize(row_bytes, out).expect("dequant row");
    };

    dequant_row(source, 0, 0, &mut row_f32);
    println!("expert0 row0   [0..8]  = {:?}", &row_f32[0..8]);
    println!(
        "expert0 row0   [0..8] abs_sum = {}",
        row_f32.iter().map(|value| value.abs()).sum::<f32>()
    );

    dequant_row(source, 0, expert_feed_forward, &mut row_f32);
    println!(
        "expert0 row{expert_feed_forward} [0..8]  = {:?}",
        &row_f32[0..8]
    );
    println!(
        "expert0 row{expert_feed_forward} abs_sum = {}",
        row_f32.iter().map(|value| value.abs()).sum::<f32>()
    );

    dequant_row(source, 0, 1, &mut row_f32);
    println!("expert0 row1   [0..8]  = {:?}", &row_f32[0..8]);

    dequant_row(source, 0, ne1 - 1, &mut row_f32);
    println!("expert0 row{} [0..8]  = {:?}", ne1 - 1, &row_f32[0..8]);

    // expert 1 row 0: proves the per-expert stride is correct if this is
    // NOT byte-identical to expert 0 row 0 and NOT equal to expert 0's
    // row 704 either (which would indicate a stride-off-by-one).
    dequant_row(source, 1, 0, &mut row_f32);
    println!("expert1 row0   [0..8]  = {:?}", &row_f32[0..8]);

    // correlation probe: for row r in [0, expert_feed_forward), compare
    // gate-row r's f32 stats against up-row r (r + expert_feed_forward) and
    // against a plausible interleaved partner (2r, 2r+1) to see which
    // pairing looks like a matched gate/up pair (same "shape" of magnitude
    // distribution) rather than two unrelated experts' rows.
    let mut gate_row = vec![0.0f32; ne0];
    let mut up_row = vec![0.0f32; ne0];
    for probe_row in [0usize, 100, 703] {
        dequant_row(source, 0, probe_row, &mut gate_row);
        dequant_row(source, 0, probe_row + expert_feed_forward, &mut up_row);
        let gate_abs_mean = gate_row.iter().map(|value| value.abs()).sum::<f32>() / ne0 as f32;
        let up_abs_mean = up_row.iter().map(|value| value.abs()).sum::<f32>() / ne0 as f32;
        println!(
            "blocked-pair probe row={probe_row}/{}: gate_abs_mean={gate_abs_mean:.6} up_abs_mean={up_abs_mean:.6}",
            probe_row + expert_feed_forward
        );
    }
}
