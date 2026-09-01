//! BGE-small-en-v1.5 end-to-end acceptance run: parse -> lower (with pinned
//! symbolic batch/sequence axes) -> evaluate on the generic executor ->
//! CLS-pooled, L2-normalized sentence embeddings -> cosine-similarity
//! sanity check.
//!
//! Tokenization: `proxima-tokenizer` is a byte-level BPE tokenizer (see its
//! own crate doc); BGE's `tokenizer.json` declares `BertNormalizer` /
//! `BertPreTokenizer` / WordPiece, a different scheme this crate does not
//! implement. Per this task's own acceptance note, the fallback is a
//! hand-built token-id array using the model's real `vocab.txt` (a plain
//! line-numbered id table, `id = line_number - 1`) and BERT's own
//! `[CLS]=101 ... [SEP]=102` convention -- documented here, not asserted as
//! tokenizer support this crate does not have.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

/// This crate never hardcodes a path onto another repo's checkout -- the
/// model lives on whichever host happens to have it cached, named by
/// `BGE_MODEL_PATH` so this example stays runnable (and skips cleanly
/// when absent) without embedding that path in source.
const MODEL_PATH_ENV: &str = "BGE_MODEL_PATH";

/// `[CLS]`, three sentences, `[SEP]` -- ids read directly off the real
/// `vocab.txt` shipped next to the model (`grep -n` for each whole word,
/// `id = line - 1`): sentence A/B are paraphrases of the same claim
/// (cat on a mat), sentence C is topically unrelated (quantum physics).
fn sentences() -> [(&'static str, Vec<i64>); 3] {
    [
        ("the cat sat on the mat", vec![101, 1996, 4937, 2938, 2006, 1996, 13523, 102]),
        ("a cat is sitting on a mat", vec![101, 1037, 4937, 2003, 3564, 2006, 1037, 13523, 102]),
        ("quantum physics explains atomic energy", vec![101, 8559, 5584, 7607, 9593, 2943, 102]),
    ]
}

fn embed(lowered_program: &[proxima_tensor::Op], graph_inputs: &[String], initializers: &[(String, Vec<f32>)], output: proxima_tensor::NodeId, tokens: &[i64]) -> Vec<f32> {
    let sequence_length = tokens.len();
    let input_ids: Vec<f32> = tokens.iter().map(|&id| id as f32).collect();
    let attention_mask = vec![1.0f32; sequence_length];
    let token_type_ids = vec![0.0f32; sequence_length];

    let mut named: Vec<(&str, &[f32])> = initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
    for name in graph_inputs {
        let data: &[f32] = match name.as_str() {
            "input_ids" => &input_ids,
            "attention_mask" => &attention_mask,
            "token_type_ids" => &token_type_ids,
            other => panic!("unexpected graph input {other:?}"),
        };
        named.push((name.as_str(), data));
    }

    let evaluated = proxima_tensor::cpu::evaluate_named(lowered_program, &[], &named, &[output]).expect("evaluate BGE-small on the generic executor");
    let (data, shape) = evaluated.get(output).expect("last_hidden_state present");
    assert_eq!(shape, &[1u64, sequence_length as u64, 384u64], "unexpected last_hidden_state shape");
    assert!(data.iter().all(|value| value.is_finite()), "non-finite value in last_hidden_state");

    // CLS pooling (BGE's own documented usage: sentence embedding is the
    // first token's hidden state), then L2-normalize.
    let cls = &data[0..384];
    let norm = cls.iter().map(|value| value * value).sum::<f32>().sqrt();
    cls.iter().map(|&value| value / norm).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// `docs/discipline.md` ROW 190's own measurement harness: runs the full
/// sentence set once with the `LayerNorm` epilogue fusion at whatever state
/// `proxima_tensor::cpu::set_epilogue_fuse_enabled` last left it, returning
/// per-sentence eval durations, the fusion engagement counters
/// (`epilogue_fuse_totals`, reset first so this call's own count is
/// isolated), and the resulting embeddings for a bit-identity/cosine
/// comparison between the fused and unfused arms.
/// `docs/discipline.md` ROW 204 adds the cluster-fusion counters
/// (`layer_norm_cluster_totals`) alongside ROW 190's own single-hop
/// counters -- both reset first, so this call's own count is isolated from
/// any earlier pass.
type RunPassResult = (Vec<std::time::Duration>, Vec<Vec<f32>>, (u64, u64, u64), (u64, u64, u64));

fn run_pass(graph: &proxima_onnx::messages::GraphProto<'_>, items: &[(&str, Vec<i64>)]) -> RunPassResult {
    proxima_tensor::cpu::epilogue_fuse_reset();
    proxima_tensor::cpu::layer_norm_cluster_reset();
    let mut durations = Vec::new();
    let mut embeddings = Vec::new();
    for (_, tokens) in items {
        let mut pins = std::collections::BTreeMap::new();
        pins.insert("batch_size", 1u64);
        pins.insert("sequence_length", tokens.len() as u64);
        let lowered = proxima_onnx::lower::lower_graph_pinned(graph, &pins).expect("lower BGE-small with pinned symbolic axes");
        let output = lowered.graph_outputs.first().expect("last_hidden_state output").1;
        let eval_start = Instant::now();
        let embedding = embed(&lowered.program, &lowered.graph_inputs, &lowered.initializers, output, tokens);
        durations.push(eval_start.elapsed());
        embeddings.push(embedding);
    }
    (durations, embeddings, proxima_tensor::cpu::epilogue_fuse_totals(), proxima_tensor::cpu::layer_norm_cluster_totals())
}

fn main() {
    let Ok(model_path) = env::var(MODEL_PATH_ENV) else {
        eprintln!("skipping: set {MODEL_PATH_ENV} to a local BGE-small-en-v1.5 model.onnx checkout");
        return;
    };
    if !Path::new(&model_path).exists() {
        eprintln!("skipping: {MODEL_PATH_ENV}={model_path:?} does not exist");
        return;
    }
    let bytes = fs::read(&model_path).expect("read bge model.onnx");
    let model = proxima_onnx::pipe::parse_complete(&bytes).expect("parse");
    let graph = model.graph.as_ref().expect("graph");

    let items = sentences();
    let runs: usize = env::var("BGE_EVAL_RUNS").ok().and_then(|value| value.parse().ok()).unwrap_or(5);

    let mut fused_run_means = Vec::new();
    let mut unfused_run_means = Vec::new();
    let mut fused_embeddings_last = Vec::new();
    let mut unfused_embeddings_last = Vec::new();
    let mut fused_totals = (0u64, 0u64, 0u64);
    let mut cluster_totals = (0u64, 0u64, 0u64);

    for run in 0..runs {
        proxima_tensor::cpu::set_epilogue_fuse_enabled(true);
        let (durations, embeddings, totals, cluster) = run_pass(graph, &items);
        let mean: std::time::Duration = durations.iter().sum::<std::time::Duration>() / durations.len() as u32;
        println!(
            "run {run} fused: per-sentence={durations:?} mean={mean:?} fuse_hits={} fuse_elements={} fuse_nanos={} ln_cluster_hits={} ln_cluster_elements={} ln_cluster_nanos={}",
            totals.0, totals.1, totals.2, cluster.0, cluster.1, cluster.2
        );
        fused_run_means.push(mean.as_secs_f64() * 1000.0);
        fused_embeddings_last = embeddings;
        fused_totals = totals;
        cluster_totals = cluster;

        proxima_tensor::cpu::set_epilogue_fuse_enabled(false);
        let (durations, embeddings, totals, cluster) = run_pass(graph, &items);
        let mean: std::time::Duration = durations.iter().sum::<std::time::Duration>() / durations.len() as u32;
        println!(
            "run {run} unfused: per-sentence={durations:?} mean={mean:?} fuse_hits={} fuse_elements={} fuse_nanos={} ln_cluster_hits={} ln_cluster_elements={} ln_cluster_nanos={}",
            totals.0, totals.1, totals.2, cluster.0, cluster.1, cluster.2
        );
        unfused_run_means.push(mean.as_secs_f64() * 1000.0);
        unfused_embeddings_last = embeddings;
        proxima_tensor::cpu::set_epilogue_fuse_enabled(true);
    }

    let bit_identical = fused_embeddings_last.len() == unfused_embeddings_last.len()
        && fused_embeddings_last.iter().zip(unfused_embeddings_last.iter()).all(|(fused, unfused)| fused.len() == unfused.len() && fused.iter().zip(unfused.iter()).all(|(&left, &right)| left.to_bits() == right.to_bits()));

    let fused_mean = fused_run_means.iter().sum::<f64>() / fused_run_means.len() as f64;
    let fused_cov = coefficient_of_variation(&fused_run_means, fused_mean);
    let unfused_mean = unfused_run_means.iter().sum::<f64>() / unfused_run_means.len() as f64;
    let unfused_cov = coefficient_of_variation(&unfused_run_means, unfused_mean);

    println!("=== ROW 190 summary ===");
    println!("runs={runs}");
    println!("fused engagement (last run): hits={} elements={} nanos={}", fused_totals.0, fused_totals.1, fused_totals.2);
    println!("ln_cluster engagement (last run): hits={} elements={} nanos={}", cluster_totals.0, cluster_totals.1, cluster_totals.2);
    println!("fused mean per-sentence ms across runs: {fused_run_means:?} mean={fused_mean:.4} CoV={fused_cov:.4}");
    println!("unfused mean per-sentence ms across runs: {unfused_run_means:?} mean={unfused_mean:.4} CoV={unfused_cov:.4}");
    println!("bit_identical(fused vs unfused, last run's embeddings)={bit_identical}");

    let similar = cosine(&fused_embeddings_last[0], &fused_embeddings_last[1]);
    let dissimilar_a = cosine(&fused_embeddings_last[0], &fused_embeddings_last[2]);
    let dissimilar_b = cosine(&fused_embeddings_last[1], &fused_embeddings_last[2]);
    println!("cosine(A,B similar)={similar:.6}");
    println!("cosine(A,C dissimilar)={dissimilar_a:.6}");
    println!("cosine(B,C dissimilar)={dissimilar_b:.6}");
    for (name, embedding) in ["A", "B", "C"].iter().zip(fused_embeddings_last.iter()) {
        println!("embedding[{name}][:8]={:?}", &embedding[0..8]);
    }
    assert!(similar > dissimilar_a, "similar pair should score higher than dissimilar pair A");
    assert!(similar > dissimilar_b, "similar pair should score higher than dissimilar pair B");
    println!("sanity check passed: similar sentence pair scores higher than dissimilar pairs");
}

fn coefficient_of_variation(samples: &[f64], mean: f64) -> f64 {
    if samples.len() < 2 || mean == 0.0 {
        return 0.0;
    }
    let variance = samples.iter().map(|&value| (value - mean).powi(2)).sum::<f64>() / samples.len() as f64;
    variance.sqrt() / mean
}
