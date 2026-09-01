//! Four-arm measurement for the arena/fusion unification
//! (`proxima-tensor/src/cpu.rs`'s `run_resolved_nodes_in_arena` now carries
//! `run_rewrite_worklist`'s fusion dispatch alongside arena reuse and law
//! 6∘5 packing). Every prior BGE number measured at most two of the three
//! landed optimizations at once -- this cell is the first to report all
//! four combinations, interleaved, on the real BGE-small-en-v1.5 model and
//! the same three real sentences `bge_eval.rs`'s own oracle uses.
//!
//! Arm A: `evaluate_named` (fusion only) -- today's sealed baseline.
//! Arm B: arena + packing, no fusion -- ROW 206's own arm.
//! Arm C: arena + packing + fusion -- the new unified path.
//! Arm D: arena + fusion, no packing -- isolates packing's own contribution
//!        under fusion (arm C vs arm D).
//!
//! Every arena is built exactly once per (arm, sentence) pair, OFF the
//! timed loop -- `build_counts` below is the engagement proof that a
//! distinct pinned shape gets exactly one build, never one per call.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use proxima_tensor::cpu::{
    StaticArena, arena_packed_node_count, build_static_arena, build_static_arena_with_constants, evaluate_named, evaluate_named_with_arena,
    layer_norm_cluster_reset, layer_norm_cluster_totals, rewrite_engine_depth_fires, rewrite_engine_reset, set_epilogue_fuse_enabled,
};

const MODEL_PATH_ENV: &str = "BGE_MODEL_PATH";
const RUNS: usize = 6;

fn sentences() -> [(&'static str, Vec<i64>); 3] {
    [
        ("the cat sat on the mat", vec![101, 1996, 4937, 2938, 2006, 1996, 13523, 102]),
        ("a cat is sitting on a mat", vec![101, 1037, 4937, 2003, 3564, 2006, 1037, 13523, 102]),
        ("quantum physics explains atomic energy", vec![101, 8559, 5584, 7607, 9593, 2943, 102]),
    ]
}

struct SentenceGraph {
    label: &'static str,
    program: Vec<proxima_tensor::Op>,
    initializers: Vec<(String, Vec<f32>)>,
    graph_inputs: Vec<String>,
    output: proxima_tensor::NodeId,
    input_ids: Vec<f32>,
    attention_mask: Vec<f32>,
    token_type_ids: Vec<f32>,
}

impl SentenceGraph {
    fn named(&self) -> Vec<(&str, &[f32])> {
        let mut named: Vec<(&str, &[f32])> = self.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect();
        for name in &self.graph_inputs {
            let data: &[f32] = match name.as_str() {
                "input_ids" => &self.input_ids,
                "attention_mask" => &self.attention_mask,
                "token_type_ids" => &self.token_type_ids,
                other => panic!("unexpected graph input {other:?}"),
            };
            named.push((name.as_str(), data));
        }
        named
    }

    fn constant_inputs(&self) -> Vec<(&str, &[f32])> {
        self.initializers.iter().map(|(name, data)| (name.as_str(), data.as_slice())).collect()
    }
}

fn build_sentence_graphs(graph: &proxima_onnx::messages::GraphProto<'_>, items: &[(&'static str, Vec<i64>)]) -> Vec<SentenceGraph> {
    items
        .iter()
        .map(|(label, tokens)| {
            let mut pins = BTreeMap::new();
            pins.insert("batch_size", 1u64);
            pins.insert("sequence_length", tokens.len() as u64);
            let lowered = proxima_onnx::lower::lower_graph_pinned(graph, &pins).expect("lower BGE-small with pinned symbolic axes");
            let output = lowered.graph_outputs.first().expect("last_hidden_state output").1;
            let sequence_length = tokens.len();
            SentenceGraph {
                label,
                program: lowered.program,
                initializers: lowered.initializers,
                graph_inputs: lowered.graph_inputs,
                output,
                input_ids: tokens.iter().map(|&id| id as f32).collect(),
                attention_mask: vec![1.0f32; sequence_length],
                token_type_ids: vec![0.0f32; sequence_length],
            }
        })
        .collect()
}

fn cls_normalize(data: &[f32]) -> Vec<f32> {
    let cls = &data[0..384];
    let norm = cls.iter().map(|value| value * value).sum::<f32>().sqrt();
    cls.iter().map(|&value| value / norm).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// `pgrep -fl 'cargo|rustc|nextest|python'` filtered to drop `cdb-daemon`
/// and `sccache`, checked twice 60s apart, retried up to 15 times before
/// giving up and labeling the host loaded -- the same quiet-gate contract
/// every timed cell in this measurement session uses.
fn quiet_gate() -> bool {
    let matches = |output: &str| -> usize {
        output
            .lines()
            .filter(|line| !line.contains("cdb-daemon") && !line.contains("sccache") && !line.contains("bge_arena_fusion_unify"))
            .count()
    };
    for attempt in 0..15 {
        let first = Command::new("pgrep").args(["-fl", "cargo|rustc|nextest|python"]).output();
        let first_count = first.ok().map(|output| matches(&String::from_utf8_lossy(&output.stdout))).unwrap_or(0);
        if first_count == 0 {
            std::thread::sleep(Duration::from_secs(60));
            let second = Command::new("pgrep").args(["-fl", "cargo|rustc|nextest|python"]).output();
            let second_count = second.ok().map(|output| matches(&String::from_utf8_lossy(&output.stdout))).unwrap_or(0);
            if second_count == 0 {
                return true;
            }
        }
        eprintln!("quiet_gate: host busy, retry {}/15", attempt + 1);
    }
    false
}

fn coefficient_of_variation(samples: &[f64], mean: f64) -> f64 {
    if samples.len() < 2 || mean == 0.0 {
        return 0.0;
    }
    let variance = samples.iter().map(|&value| (value - mean).powi(2)).sum::<f64>() / samples.len() as f64;
    variance.sqrt() / mean * 100.0
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
    let sentence_graphs = build_sentence_graphs(graph, &items);

    let quiet = if env::var("BGE_UNIFY_SKIP_QUIET_GATE").is_ok() { true } else { quiet_gate() };
    let host_label = if quiet { "quiet" } else { "loaded-host" };
    println!("host quiet-gate: {host_label}");

    // Arm B: arena + packing, no fusion.
    set_epilogue_fuse_enabled(false);
    let mut arm_b_arenas: Vec<StaticArena> = Vec::new();
    let mut build_count_b = 0usize;
    for entry in &sentence_graphs {
        arm_b_arenas.push(build_static_arena_with_constants(&entry.program, &[], &[entry.output], &entry.constant_inputs()).expect("build arm B arena"));
        build_count_b += 1;
    }

    // Arm C: arena + packing + fusion.
    set_epilogue_fuse_enabled(true);
    rewrite_engine_reset();
    let mut arm_c_arenas: Vec<StaticArena> = Vec::new();
    let mut build_count_c = 0usize;
    for entry in &sentence_graphs {
        arm_c_arenas.push(build_static_arena_with_constants(&entry.program, &[], &[entry.output], &entry.constant_inputs()).expect("build arm C arena"));
        build_count_c += 1;
    }
    let (arm_c_depth1, arm_c_depth2) = rewrite_engine_depth_fires();
    let arm_c_packed_nodes: usize = arm_c_arenas.iter().map(arena_packed_node_count).sum();

    // Arm D: arena + fusion, no packing.
    let mut arm_d_arenas: Vec<StaticArena> = Vec::new();
    let mut build_count_d = 0usize;
    for entry in &sentence_graphs {
        arm_d_arenas.push(build_static_arena(&entry.program, &[], &[entry.output]).expect("build arm D arena"));
        build_count_d += 1;
    }

    // Runtime engagement: arm C's own arenas, evaluated once, off the timed
    // loop, to assert law 2's cluster fusion actually fires (not just
    // admitted at build time) before any timed number is trusted. This
    // pass also doubles as arm C's own warm-up (see below).
    layer_norm_cluster_reset();
    for (index, entry) in sentence_graphs.iter().enumerate() {
        let named = entry.named();
        let _ = evaluate_named_with_arena(&mut arm_c_arenas[index], &named).expect("arm C engagement warm-up eval");
    }
    let (arm_c_ln_hits, ..) = layer_norm_cluster_totals();

    // Every other arm gets the SAME one-pass warm-up, off the timed loop --
    // without it, arm B/D's own first timed sample carries a cold-cache
    // penalty arm C's engagement pass above already absorbed, which would
    // bias CoV% between arms rather than measuring the steady-state
    // difference this table exists to isolate.
    for (index, entry) in sentence_graphs.iter().enumerate() {
        let named = entry.named();
        let _ = evaluate_named(&entry.program, &[], &named, &[entry.output]).expect("arm A warm-up eval");
        let _ = evaluate_named_with_arena(&mut arm_b_arenas[index], &named).expect("arm B warm-up eval");
        let _ = evaluate_named_with_arena(&mut arm_d_arenas[index], &named).expect("arm D warm-up eval");
    }

    println!("=== ENGAGEMENT PROOF ===");
    println!("1. fusion fires (arena path, build-time admission): depth1(law1_2_epilogue_absorption)={arm_c_depth1} depth2(law2_layer_norm_cluster_upgrade)={arm_c_depth2}; runtime layer_norm_cluster_hits(one warm-up pass, 3 sentences)={arm_c_ln_hits}");
    println!("2. packed node count (arm C, summed over 3 sentence arenas)={arm_c_packed_nodes}");
    println!(
        "3. arena builds: arm B={build_count_b} arm C={build_count_c} arm D={build_count_d} (expected {} each -- one per distinct pinned sentence length, never per call)",
        items.len()
    );
    assert!(arm_c_depth1 > 0, "engagement N==0 is RED: law 1/2 admission never fired at arena-build time");
    assert!(arm_c_depth2 > 0, "engagement N==0 is RED: law 2 cluster upgrade never fired at arena-build time");
    assert!(arm_c_ln_hits > 0, "engagement N==0 is RED: layer-norm cluster fusion never fired at runtime in the arena path");
    assert!(arm_c_packed_nodes > 0, "engagement N==0 is RED: no width-tile node was packed on the real BGE graph");
    assert_eq!(build_count_b, items.len());
    assert_eq!(build_count_c, items.len());
    assert_eq!(build_count_d, items.len());

    // `set_epilogue_fuse_enabled` stays `true` for the remainder of this
    // process -- arm A (`evaluate_named`) re-derives its own fusion plan
    // every call and needs the flag live at call time; arms B/C/D already
    // baked their own plan into their arena at build time above, so the
    // live flag value has no further effect on them.
    set_epilogue_fuse_enabled(true);

    // ms samples per (arm, sentence), one entry per repeat -- interleaved:
    // each repeat visits every arm, every sentence, before the next repeat
    // starts, so no arm/sentence pair accumulates all its samples back to
    // back.
    let mut samples: BTreeMap<(&'static str, &'static str), Vec<f64>> = BTreeMap::new();
    let mut arm_a_embeddings: Vec<Vec<f32>> = Vec::new();
    let mut arm_c_embeddings: Vec<Vec<f32>> = Vec::new();

    for run in 0..RUNS {
        for (index, entry) in sentence_graphs.iter().enumerate() {
            let named = entry.named();

            let start = Instant::now();
            let evaluated = evaluate_named(&entry.program, &[], &named, &[entry.output]).expect("arm A eval");
            let elapsed = start.elapsed();
            let (data, _) = evaluated.get(entry.output).expect("arm A output");
            samples.entry(("A_evaluate_named_fusion_only", entry.label)).or_default().push(elapsed.as_secs_f64() * 1000.0);
            if run == RUNS - 1 {
                arm_a_embeddings.push(cls_normalize(data));
            }

            let start = Instant::now();
            let evaluated = evaluate_named_with_arena(&mut arm_b_arenas[index], &named).expect("arm B eval");
            let elapsed = start.elapsed();
            let _ = evaluated.get(entry.output).expect("arm B output");
            samples.entry(("B_arena_packing_no_fusion", entry.label)).or_default().push(elapsed.as_secs_f64() * 1000.0);

            let start = Instant::now();
            let evaluated = evaluate_named_with_arena(&mut arm_c_arenas[index], &named).expect("arm C eval");
            let elapsed = start.elapsed();
            let (data, _) = evaluated.get(entry.output).expect("arm C output");
            samples.entry(("C_arena_packing_fusion", entry.label)).or_default().push(elapsed.as_secs_f64() * 1000.0);
            if run == RUNS - 1 {
                arm_c_embeddings.push(cls_normalize(data));
            }

            let start = Instant::now();
            let evaluated = evaluate_named_with_arena(&mut arm_d_arenas[index], &named).expect("arm D eval");
            let elapsed = start.elapsed();
            let _ = evaluated.get(entry.output).expect("arm D output");
            samples.entry(("D_arena_fusion_no_packing", entry.label)).or_default().push(elapsed.as_secs_f64() * 1000.0);
        }
    }

    println!("\n=== FOUR-ARM TABLE (ms/sentence, {RUNS} runs interleaved) ===");
    let arms = ["A_evaluate_named_fusion_only", "B_arena_packing_no_fusion", "C_arena_packing_fusion", "D_arena_fusion_no_packing"];
    let mut arm_means: BTreeMap<&'static str, f64> = BTreeMap::new();
    for arm in arms {
        let mut arm_total = 0.0;
        let mut arm_total_n = 0usize;
        for (label, _) in &items {
            let values = samples.get(&(arm, *label)).expect("samples present");
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let cov = coefficient_of_variation(values, mean);
            println!("{arm:<32} sentence={label:<40} mean_ms={mean:>9.4} CoV%={cov:>6.2} n={} samples={values:?}", values.len());
            arm_total += mean;
            arm_total_n += 1;
        }
        let arm_mean = arm_total / arm_total_n as f64;
        arm_means.insert(arm, arm_mean);
        println!("{arm:<32} MEAN across 3 sentences = {arm_mean:.4} ms");
    }

    let ratio_c_over_a = arm_means["C_arena_packing_fusion"] / arm_means["A_evaluate_named_fusion_only"];
    println!("\nratio C/A (arena+packing+fusion / evaluate_named fusion-only) = {ratio_c_over_a:.4}");

    let bit_identical = arm_a_embeddings.len() == arm_c_embeddings.len()
        && arm_a_embeddings
            .iter()
            .zip(arm_c_embeddings.iter())
            .all(|(left, right)| left.len() == right.len() && left.iter().zip(right.iter()).all(|(&a, &b)| a.to_bits() == b.to_bits()));
    println!("bit_identical(arm A vs arm C, last run's embeddings) = {bit_identical}");

    let similar = cosine(&arm_c_embeddings[0], &arm_c_embeddings[1]);
    let dissimilar_a = cosine(&arm_c_embeddings[0], &arm_c_embeddings[2]);
    let dissimilar_b = cosine(&arm_c_embeddings[1], &arm_c_embeddings[2]);
    println!("arm C cosine(A,B)={similar:.6} cosine(A,C)={dissimilar_a:.6} cosine(B,C)={dissimilar_b:.6}");

    assert!(bit_identical, "arm C (arena+packing+fusion) must be bit-identical to arm A (evaluate_named fusion-only)");
    assert!((similar - 0.936311).abs() < 1e-5, "cosine(A,B) drifted from the sealed oracle");
    assert!((dissimilar_a - 0.378777).abs() < 1e-5, "cosine(A,C) drifted from the sealed oracle");
    assert!((dissimilar_b - 0.334176).abs() < 1e-5, "cosine(B,C) drifted from the sealed oracle");
    println!("\nall assertions passed: bit-identity vs arm A, cosine oracle reproduced, all three engagement counts nonzero.");
}
