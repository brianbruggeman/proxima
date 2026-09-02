//! ROW 209 pre-flight: does `ACCELERATE_GEMM_ENABLED` (ROW 188/189's valve)
//! ever engage on a real BGE-small-en-v1.5 forward pass? BGE's 96 MatMuls
//! route through `try_run_width_tile` (`fast_path`), which returns before
//! `run_reduce` ever reaches the dot-tile (`reduction_fast_path`) or
//! conv-gemm gates the Accelerate route is wired to (`cpu.rs:6941`,
//! `cpu.rs:6873`) -- so the claim under test is `accelerate_gemm_totals()
//! == (0, 0)` with the valve ON, for all three real sentences.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::env;
use std::fs;
use std::path::Path;

use proxima_tensor::cpu;

const MODEL_PATH_ENV: &str = "BGE_MODEL_PATH";

fn sentences() -> [(&'static str, Vec<i64>); 3] {
    [
        (
            "the cat sat on the mat",
            vec![101, 1996, 4937, 2938, 2006, 1996, 13523, 102],
        ),
        (
            "a cat is sitting on a mat",
            vec![101, 1037, 4937, 2003, 3564, 2006, 1037, 13523, 102],
        ),
        (
            "quantum physics explains atomic energy",
            vec![101, 8559, 5584, 7607, 9593, 2943, 102],
        ),
    ]
}

fn embed(
    lowered_program: &[proxima_tensor::Op],
    graph_inputs: &[String],
    initializers: &[(String, Vec<f32>)],
    output: proxima_tensor::NodeId,
    tokens: &[i64],
) -> Vec<f32> {
    let sequence_length = tokens.len();
    let input_ids: Vec<f32> = tokens.iter().map(|&id| id as f32).collect();
    let attention_mask = vec![1.0f32; sequence_length];
    let token_type_ids = vec![0.0f32; sequence_length];

    let mut named: Vec<(&str, &[f32])> = initializers
        .iter()
        .map(|(name, data)| (name.as_str(), data.as_slice()))
        .collect();
    for name in graph_inputs {
        let data: &[f32] = match name.as_str() {
            "input_ids" => &input_ids,
            "attention_mask" => &attention_mask,
            "token_type_ids" => &token_type_ids,
            other => panic!("unexpected graph input {other:?}"),
        };
        named.push((name.as_str(), data));
    }
    let evaluated = proxima_tensor::cpu::evaluate_named(lowered_program, &[], &named, &[output])
        .expect("evaluate BGE-small on the generic executor");
    let (data, shape) = evaluated.get(output).expect("last_hidden_state present");
    assert_eq!(
        shape,
        &[1u64, sequence_length as u64, 384u64],
        "unexpected last_hidden_state shape"
    );
    data[0..384].to_vec()
}

fn main() {
    let Ok(model_path) = env::var(MODEL_PATH_ENV) else {
        eprintln!(
            "skipping: set {MODEL_PATH_ENV} to a local BGE-small-en-v1.5 model.onnx checkout"
        );
        return;
    };
    if !Path::new(&model_path).exists() {
        eprintln!("skipping: {MODEL_PATH_ENV}={model_path:?} does not exist");
        return;
    }
    let bytes = fs::read(&model_path).expect("read bge model.onnx");
    let model = proxima_onnx::pipe::parse_complete(&bytes).expect("parse");
    let graph = model.graph.as_ref().expect("graph");

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    cpu::set_accelerate_gemm_enabled(true);

    let before_width = cpu::width_tile_counters();
    for (name, tokens) in sentences().iter() {
        let sequence_length = tokens.len();
        let mut pins = std::collections::BTreeMap::new();
        pins.insert("batch_size", 1u64);
        pins.insert("sequence_length", sequence_length as u64);
        let lowered = proxima_onnx::lower::lower_graph_pinned(graph, &pins)
            .expect("lower BGE-small with pinned symbolic axes");
        let output = lowered
            .graph_outputs
            .first()
            .expect("last_hidden_state output")
            .1;
        let embedding = embed(
            &lowered.program,
            &lowered.graph_inputs,
            &lowered.initializers,
            output,
            tokens,
        );
        println!("{name:?}: embedding[:4]={:?}", &embedding[0..4]);
    }
    let after_width = cpu::width_tile_counters();

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let (hits, declined) = cpu::accelerate_gemm_totals();
        println!(
            "ACCELERATE_GEMM_ENABLED=true, accelerate_gemm_totals() = (hits={hits}, declined={declined})"
        );
        println!(
            "width_tile_counters() delta = (bytes={}, invocations={}, fallback={})",
            after_width.0.saturating_sub(before_width.0),
            after_width.1.saturating_sub(before_width.1),
            after_width.2.saturating_sub(before_width.2)
        );
        assert_eq!(
            hits, 0,
            "claim under test: accelerate never fires on BGE's width-tile-routed GEMMs"
        );
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = (before_width, after_width);
        println!("non-aarch64-macos host: ACCELERATE_GEMM_ENABLED does not exist on this target");
    }
}
