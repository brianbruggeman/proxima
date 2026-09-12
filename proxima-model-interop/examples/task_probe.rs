//! Inspect a GGUF header and route it to a model-task family without loading
//! tensor payloads or attempting execution.
//!
//! Usage: `task_probe <model.gguf> [model.gguf ...]`

use std::env;
use std::fs::File;

use memmap2::MmapOptions;
use proxima_gguf::parse_complete;
use proxima_model_interop::classify_task;

fn main() {
    let paths: Vec<_> = env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: task_probe <model.gguf> [model.gguf ...]");
        std::process::exit(2);
    }
    for path in paths {
        let file = File::open(&path).unwrap_or_else(|error| panic!("open {path}: {error}"));
        // SAFETY: the file remains open for the parse call and this example
        // retains no references after the loop iteration.
        let mapping = unsafe { MmapOptions::new().map(&file) }
            .unwrap_or_else(|error| panic!("map {path}: {error}"));
        let parsed =
            parse_complete(&mapping).unwrap_or_else(|error| panic!("parse {path}: {error}"));
        let profile = classify_task(&parsed);
        let path_json = serde_json::to_string(&path).expect("serialize path");
        let architecture_json =
            serde_json::to_string(&profile.architecture).expect("serialize architecture");
        let task_json = serde_json::to_string(profile.task.name()).expect("serialize task");
        let evidence_json = serde_json::to_string(&profile.evidence).expect("serialize evidence");
        let head_tensors: Vec<_> = parsed
            .tensors
            .iter()
            .filter(|tensor| !tensor.name.starts_with("blk.") && tensor.name != "token_embd.weight")
            .map(|tensor| tensor.name.as_str())
            .collect();
        let heads_json = serde_json::to_string(&head_tensors).expect("serialize head tensors");
        let shape_probe: Vec<_> = parsed
            .tensors
            .iter()
            .filter(|tensor| matches!(tensor.name.as_str(), "blk.0.attn_q.weight" | "blk.0.attn_k.weight" | "blk.0.attn_v.weight" | "output.weight" | "output_norm.weight"))
            .map(|tensor| {
                serde_json::json!({"name": tensor.name, "dims": tensor.dims.iter().copied().collect::<Vec<_>>(), "type": format!("{:?}", tensor.ggml_type)})
            })
            .collect();
        let shape_probe_json = serde_json::to_string(&shape_probe).expect("serialize shape probe");
        println!(
            "{{\"path\":{path_json},\"architecture\":{architecture_json},\"task\":{task_json},\"generation_supported\":{supported},\"tensor_count\":{tensor_count},\"head_tensors\":{heads_json},\"shape_probe\":{shape_probe_json},\"evidence\":{evidence_json}}}",
            supported = profile.generation_supported,
            tensor_count = parsed.tensor_count,
        );
    }
}
