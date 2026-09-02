//! ROW 209 MILLI rung: the real BGE-small-en-v1.5 graph, all 3 real
//! sentences, paired interleaved Accelerate-on/off arms, mirroring
//! `bge_epilogue_profile_pack.rs`'s own harness shape (`BGE_MODEL_PATH`,
//! `StaticArena`, 3-call warm-up excluded, 60 measured calls/arm/run, 5
//! interleaved runs). Also folds in this session's own correctness check
//! (ROW 195's cosine oracle table) and the engagement proof
//! (`accelerate_gemm_totals()` delta per accelerate call, asserted > 0).
//!
//! The Accelerate valve (`ACCELERATE_GEMM_ENABLED`) is a per-call runtime
//! read inside `run_reduce`, never baked into the arena at build time
//! (unlike packing's own `PACK_AT_PLAN_TIME_ENABLED`) -- so ONE arena per
//! sentence serves both arms; only `set_accelerate_gemm_enabled` toggles
//! between calls.
//!
//! PRE-REGISTRATION (recorded before this file was ever run, ONE rung ahead
//! of MICRO's own cells only -- not a derived e2e claim):
//! `bge_accelerate_engagement.rs`'s own pre-flight measured `reduce_f32_dense_gemm`
//! (the `Op::MatMul`-shaped class, count=96) at 85.14%-90.55% of forward
//! wall time (`pct_of_forward`), and MICRO's own real-shape cells at BGE's
//! M in {7,8,9} measured accelerate/neon GEMM-only ratios of 0.308x-0.446x
//! (M=1 cells: 0.631x-0.769x). Unlike packing's own 72/96-eligible split
//! (packing needs a compile-time-constant `b`), the Accelerate valve reads
//! whatever buffer `raw[plan.b_operand]` points to at CALL time -- it does
//! not care whether `b` is a graph constant or another node's runtime
//! output -- so it is not restricted to the 72/96 packable subset; this
//! file's own engagement counter is the proof of how many of the 96 it
//! actually reaches. Scaled prediction, weighting the real per-sentence `M`
//! values (7, 8, 9) toward MICRO's own M=7-9 band: milli step-time ratio
//! should land around `(1 - S_gemm) + S_gemm * 0.36` (0.36 = MICRO's own
//! M=7-9 mean ratio), i.e. roughly 0.40x-0.65x, wider than MICRO's own band
//! to absorb per-call dispatch/elementwise overhead MICRO's single-node
//! synthetic graph does not carry.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::env;
use std::fs;
use std::path::Path;

use proxima_tensor::cpu::{self, StaticArena, build_static_arena, evaluate_named_with_arena};

const MODEL_PATH_ENV: &str = "BGE_MODEL_PATH";
const WARMUP_CALLS: usize = 3;
const MEASURED_CALLS: usize = 60;
const RUNS: usize = 5;

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

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a = a.iter().map(|value| value * value).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|value| value * value).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}

type NamedInputs<'a> = Vec<(&'a str, &'a [f32])>;

fn timed_arm(
    arena: &mut StaticArena,
    named: &NamedInputs<'_>,
    output: proxima_tensor::NodeId,
    accelerate: bool,
) -> (f64, u64, Vec<f32>) {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    cpu::set_accelerate_gemm_enabled(accelerate);
    let _ = accelerate;

    for _ in 0..WARMUP_CALLS {
        let evaluated = evaluate_named_with_arena(arena, named).expect("warm-up eval");
        std::hint::black_box(&evaluated);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let (hits_before, _) = cpu::accelerate_gemm_totals();
    let start = std::time::Instant::now();
    let mut last_embedding = Vec::new();
    for _ in 0..MEASURED_CALLS {
        let evaluated = evaluate_named_with_arena(arena, named).expect("timed eval");
        let (data, _shape) = evaluated.get(output).expect("last_hidden_state present");
        last_embedding.clear();
        last_embedding.extend_from_slice(&data[0..384]);
        std::hint::black_box(&evaluated);
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0 / MEASURED_CALLS as f64;
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let (hits_after, _) = cpu::accelerate_gemm_totals();
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let engagement = hits_after - hits_before;
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    let engagement = 0u64;

    let norm: f32 = last_embedding
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    for value in &mut last_embedding {
        *value /= norm;
    }
    (elapsed_ms, engagement, last_embedding)
}

fn mean_cov(samples: &[f64]) -> (f64, f64) {
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = samples
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / samples.len() as f64;
    (mean, variance.sqrt() / mean * 100.0)
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

    println!(
        "bge_width_tile_accelerate_milli: ROW 209 MILLI rung -- real BGE graph, {RUNS} interleaved runs, {WARMUP_CALLS}-call warm-up excluded, {MEASURED_CALLS} measured calls/arm/run"
    );
    println!(
        "PRE-REGISTRATION: see file doc comment -- predicted step-time ratio ~0.40x-0.65x, engagement > 0 required every accelerate arm."
    );

    let mut embeddings_neon = Vec::new();
    let mut embeddings_accelerate = Vec::new();

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

        let input_ids: Vec<f32> = tokens.iter().map(|&id| id as f32).collect();
        let attention_mask = vec![1.0f32; sequence_length];
        let token_type_ids = vec![0.0f32; sequence_length];
        let mut input_names: [String; 3] = [String::new(), String::new(), String::new()];
        for graph_input in &lowered.graph_inputs {
            match graph_input.as_str() {
                "input_ids" => input_names[0] = graph_input.clone(),
                "attention_mask" => input_names[1] = graph_input.clone(),
                "token_type_ids" => input_names[2] = graph_input.clone(),
                other => panic!("unexpected graph input {other:?}"),
            }
        }
        let mut named: NamedInputs<'_> = lowered
            .initializers
            .iter()
            .map(|(weight_name, data)| (weight_name.as_str(), data.as_slice()))
            .collect();
        named.push((input_names[0].as_str(), &input_ids));
        named.push((input_names[1].as_str(), &attention_mask));
        named.push((input_names[2].as_str(), &token_type_ids));

        let mut arena = build_static_arena(&lowered.program, &[], &[output]).expect("build arena");

        println!("--- {name:?} (M={sequence_length}) ---");
        let mut neon_ms = Vec::with_capacity(RUNS);
        let mut accelerate_ms = Vec::with_capacity(RUNS);
        let mut neon_embedding = Vec::new();
        let mut accelerate_embedding = Vec::new();

        for run in 0..RUNS {
            let neon_first = run % 2 == 0;
            if neon_first {
                let (elapsed, _engagement, embedding) =
                    timed_arm(&mut arena, &named, output, false);
                neon_ms.push(elapsed);
                neon_embedding = embedding;
                let (elapsed, engagement, embedding) = timed_arm(&mut arena, &named, output, true);
                accelerate_ms.push(elapsed);
                accelerate_embedding = embedding;
                println!(
                    "  run {run} (neon, accelerate): neon={:.4}ms accelerate={:.4}ms (engagement-hits={engagement})",
                    neon_ms[run], accelerate_ms[run]
                );
                assert!(
                    engagement > 0,
                    "engagement proof: accelerate arm must record at least one accelerate_gemm_totals() hit, got 0"
                );
            } else {
                let (elapsed, engagement, embedding) = timed_arm(&mut arena, &named, output, true);
                accelerate_ms.push(elapsed);
                accelerate_embedding = embedding;
                let (elapsed, _engagement, embedding) =
                    timed_arm(&mut arena, &named, output, false);
                neon_ms.push(elapsed);
                neon_embedding = embedding;
                println!(
                    "  run {run} (accelerate, neon): accelerate={:.4}ms neon={:.4}ms (engagement-hits={engagement})",
                    accelerate_ms[run], neon_ms[run]
                );
                assert!(
                    engagement > 0,
                    "engagement proof: accelerate arm must record at least one accelerate_gemm_totals() hit, got 0"
                );
            }
        }

        let (neon_mean, neon_cov) = mean_cov(&neon_ms);
        let (accelerate_mean, accelerate_cov) = mean_cov(&accelerate_ms);
        let ratio = accelerate_mean / neon_mean;
        println!(
            "  neon:       mean={neon_mean:.4}ms CoV={neon_cov:.2}% samples={:?}",
            neon_ms
                .iter()
                .map(|value| format!("{value:.3}"))
                .collect::<Vec<_>>()
        );
        println!(
            "  accelerate: mean={accelerate_mean:.4}ms CoV={accelerate_cov:.2}% samples={:?}",
            accelerate_ms
                .iter()
                .map(|value| format!("{value:.3}"))
                .collect::<Vec<_>>()
        );
        println!(
            "  -> accelerate/neon step-time ratio: {ratio:.4}x  ({:.2}% delta)",
            (ratio - 1.0) * 100.0
        );
        let outcome = if (0.30..=0.75).contains(&ratio) {
            "HIT"
        } else {
            "MISS"
        };
        println!(
            "  -> pre-registered prediction: ~0.40x-0.65x (widened to 0.30x-0.75x for gate purposes) -> {outcome}"
        );
        if neon_cov > 5.0 || accelerate_cov > 5.0 {
            println!(
                "  -> CoV above 5% trust line on at least one arm -- report the RANGE, not the point estimate, for this sentence."
            );
        }

        let max_abs_diff = neon_embedding
            .iter()
            .zip(accelerate_embedding.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_rel_diff = neon_embedding
            .iter()
            .zip(accelerate_embedding.iter())
            .map(|(&a, &b)| (a - b).abs() / a.abs().max(1e-6))
            .fold(0.0f32, f32::max);
        println!(
            "  -> correctness: max_abs_diff(neon, accelerate)={max_abs_diff:.8} max_rel_diff={max_rel_diff:.8}"
        );
        assert!(
            max_abs_diff < 1e-2,
            "accelerate embedding diverged from neon embedding beyond f32 reorder tolerance: max_abs_diff={max_abs_diff}"
        );

        embeddings_neon.push(neon_embedding);
        embeddings_accelerate.push(accelerate_embedding);
    }

    println!("\n=== correctness: ROW 195 cosine oracle table, accelerate arm ===");
    let similar = cosine(&embeddings_accelerate[0], &embeddings_accelerate[1]);
    let dissimilar_a = cosine(&embeddings_accelerate[0], &embeddings_accelerate[2]);
    let dissimilar_b = cosine(&embeddings_accelerate[1], &embeddings_accelerate[2]);
    println!("cosine(A,B similar)={similar:.6}   (ROW 195 oracle: 0.936311)");
    println!("cosine(A,C dissimilar)={dissimilar_a:.6}   (ROW 195 oracle: 0.378777)");
    println!("cosine(B,C dissimilar)={dissimilar_b:.6}   (ROW 195 oracle: 0.334176)");

    println!("\n=== correctness: ROW 195 cosine oracle table, neon arm ===");
    let similar_neon = cosine(&embeddings_neon[0], &embeddings_neon[1]);
    let dissimilar_a_neon = cosine(&embeddings_neon[0], &embeddings_neon[2]);
    let dissimilar_b_neon = cosine(&embeddings_neon[1], &embeddings_neon[2]);
    println!("cosine(A,B similar)={similar_neon:.6}");
    println!("cosine(A,C dissimilar)={dissimilar_a_neon:.6}");
    println!("cosine(B,C dissimilar)={dissimilar_b_neon:.6}");

    assert!(
        similar > dissimilar_a,
        "accelerate arm: similar pair should score higher than dissimilar pair A"
    );
    assert!(
        similar > dissimilar_b,
        "accelerate arm: similar pair should score higher than dissimilar pair B"
    );
    println!(
        "sanity check passed on accelerate arm: similar sentence pair scores higher than dissimilar pairs"
    );
}
