//! onnxruntime 1.20.1 CPU-EP arm for the BGE-small lane, same protocol as the 2026-09 python harness
//! (commit 36745c59, `scripts/onnx_reference/bench.py`) and `proxima-onnx/examples/bge_eval.rs`:
//! 3 sentences, 1 warmup embed of sentence 0, 5 runs, per-run mean, `CoV` across run means.
//!
//! env: `BGE_MODEL_PATH` (required), `ORT_LIB_DIR`, `BGE_ORT_RUNS` (default 5), `BGE_ORT_EMB_DIR`.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ort_arm::{EmbedSession, OrtRuntime, coefficient_of_variation, cosine, lib_dir_from_env, mean, sha256_hex};

const MODEL_PATH_ENV: &str = "BGE_MODEL_PATH";
const RUNS_ENV: &str = "BGE_ORT_RUNS";
const EMB_DIR_ENV: &str = "BGE_ORT_EMB_DIR";
const DEFAULT_EMB_DIR: &str = "/Users/brianbruggeman/.cache/kvgate/proxnum71";
const WARMUP_PASSES: usize = 1;

/// token ids copied from `proxima-onnx/examples/bge_eval.rs::sentences()`.
const SENTENCES: [(&str, &[i64]); 3] = [
    ("the cat sat on the mat", &[101, 1996, 4937, 2938, 2006, 1996, 13523, 102]),
    ("a cat is sitting on a mat", &[101, 1037, 4937, 2003, 3564, 2006, 1037, 13523, 102]),
    ("quantum physics explains atomic energy", &[101, 8559, 5584, 7607, 9593, 2943, 102]),
];

fn load_average() -> String {
    let mut loads = [0.0_f64; 3];
    // safety: getloadavg writes at most 3 doubles into the array
    let written = unsafe { getloadavg(loads.as_mut_ptr(), 3) };
    format!("{written} samples 1m={:.2} 5m={:.2} 15m={:.2}", loads[0], loads[1], loads[2])
}

unsafe extern "C" {
    fn getloadavg(loadavg: *mut f64, nelem: i32) -> i32;
}

fn main() -> Result<()> {
    let model_path = PathBuf::from(env::var(MODEL_PATH_ENV).with_context(|| format!("set {MODEL_PATH_ENV}"))?);
    let runs: usize = env::var(RUNS_ENV).ok().and_then(|value| value.parse().ok()).unwrap_or(5);
    let emb_dir = env::var_os(EMB_DIR_ENV).map_or_else(|| PathBuf::from(DEFAULT_EMB_DIR), PathBuf::from);
    let model_bytes = fs::metadata(&model_path).with_context(|| format!("stat {}", model_path.display()))?.len();

    let runtime = OrtRuntime::load(&lib_dir_from_env())?;
    println!("ort dylib sha256={} version={}", runtime.report.sha256, runtime.report.version);
    println!("model={} bytes={model_bytes} sha256={}", model_path.display(), sha256_hex(&model_path)?);
    println!("config: CPU EP intra_op=1 inter_op=1 sequential batch=1 warmup={WARMUP_PASSES} runs={runs}");
    println!("loadavg start: {}", load_average());

    let session = EmbedSession::new(&runtime, &model_path)?;
    for _ in 0..WARMUP_PASSES {
        session.embed(SENTENCES[0].1)?;
    }

    let mut per_sentence_ms = vec![Vec::with_capacity(runs); SENTENCES.len()];
    let mut run_means = Vec::with_capacity(runs);
    let mut last_embeddings = Vec::new();
    for run in 0..runs {
        let mut embeddings = Vec::with_capacity(SENTENCES.len());
        let mut timings = Vec::with_capacity(SENTENCES.len());
        for (index, (_, tokens)) in SENTENCES.iter().enumerate() {
            let start = Instant::now();
            let embedding = session.embed(tokens)?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            per_sentence_ms[index].push(elapsed_ms);
            timings.push(elapsed_ms);
            embeddings.push(embedding);
        }
        let run_mean = mean(&timings);
        println!("run {run}: per-sentence ms={timings:.4?} mean_ms={run_mean:.4}");
        run_means.push(run_mean);
        last_embeddings = embeddings;
    }
    println!("loadavg end: {}", load_average());

    for ((text, tokens), samples) in SENTENCES.iter().zip(&per_sentence_ms) {
        let sentence_mean = mean(samples);
        let cov = coefficient_of_variation(samples, sentence_mean) * 100.0;
        println!("sentence M={} {text:?}: mean_ms={sentence_mean:.4} CoV%={cov:.2} samples={samples:.4?}", tokens.len());
    }
    let overall = mean(&run_means);
    let overall_cov = coefficient_of_variation(&run_means, overall) * 100.0;
    println!("onnxruntime per-run means={run_means:.4?} mean_ms/sentence={overall:.4} CoV%={overall_cov:.2}");

    if last_embeddings.len() != SENTENCES.len() {
        bail!("N==0: no embeddings produced");
    }
    println!(
        "cosine(A,B)={:.6} cosine(A,C)={:.6} cosine(B,C)={:.6}",
        cosine(&last_embeddings[0], &last_embeddings[1]),
        cosine(&last_embeddings[0], &last_embeddings[2]),
        cosine(&last_embeddings[1], &last_embeddings[2]),
    );
    fs::create_dir_all(&emb_dir).with_context(|| format!("create {}", emb_dir.display()))?;
    for (index, embedding) in last_embeddings.iter().enumerate() {
        let bytes: Vec<u8> = embedding.iter().flat_map(|value| value.to_le_bytes()).collect();
        let path = emb_dir.join(format!("ort_emb_{index}.f32"));
        fs::write(&path, &bytes).with_context(|| format!("write {}", path.display()))?;
        println!("wrote {} ({} bytes)", path.display(), bytes.len());
    }
    Ok(())
}
