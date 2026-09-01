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
    let mut embeddings: Vec<Vec<f32>> = Vec::new();
    let mut total_elapsed = std::time::Duration::ZERO;

    for (text, tokens) in &items {
        let mut pins = std::collections::BTreeMap::new();
        pins.insert("batch_size", 1u64);
        pins.insert("sequence_length", tokens.len() as u64);

        let lower_start = Instant::now();
        let lowered = proxima_onnx::lower::lower_graph_pinned(graph, &pins).expect("lower BGE-small with pinned symbolic axes");
        let lower_elapsed = lower_start.elapsed();

        let output = lowered.graph_outputs.first().expect("last_hidden_state output").1;

        let eval_start = Instant::now();
        let embedding = embed(&lowered.program, &lowered.graph_inputs, &lowered.initializers, output, tokens);
        let eval_elapsed = eval_start.elapsed();
        total_elapsed += eval_elapsed;

        println!("{text:?}: tokens={} lower={:?} eval={:?} dims={} finite={}", tokens.len(), lower_elapsed, eval_elapsed, embedding.len(), embedding.iter().all(|value| value.is_finite()));
        embeddings.push(embedding);
    }

    let similar = cosine(&embeddings[0], &embeddings[1]);
    let dissimilar_a = cosine(&embeddings[0], &embeddings[2]);
    let dissimilar_b = cosine(&embeddings[1], &embeddings[2]);
    println!("cosine(A,B similar)={similar:.6}");
    println!("cosine(A,C dissimilar)={dissimilar_a:.6}");
    println!("cosine(B,C dissimilar)={dissimilar_b:.6}");
    println!("mean wall-clock per inference (eval only, 3 runs): {:?}", total_elapsed / 3);
    assert!(similar > dissimilar_a, "similar pair should score higher than dissimilar pair A");
    assert!(similar > dissimilar_b, "similar pair should score higher than dissimilar pair B");
    println!("sanity check passed: similar sentence pair scores higher than dissimilar pairs");
}
