//! Proof that unifying fusion into `StaticArena`'s own execution loop
//! (`proxima-tensor/src/cpu.rs`'s `run_resolved_nodes_in_arena`) produces
//! the SAME arithmetic as the sealed `evaluate_named` fused path -- both now
//! apply `run_rewrite_worklist`'s law 1/2 (epilogue absorption + layer-norm
//! cluster upgrade) over the identical `resolved` graph, in the identical
//! order, so the two arms must agree bit-for-bit on the real BGE-small-en-v1.5
//! model. Packing (law 6∘5) only reorders WHERE a weight's bytes live, never
//! the arithmetic (`pack_at_plan_time.rs`'s own bit-identity proof), so
//! arm C (arena + packing + fusion) carries all three landed optimizations
//! at once and must still match arm A exactly.
//!
//! Skips cleanly (like `bge_eval.rs`) when `BGE_MODEL_PATH` is unset or the
//! path does not exist -- this crate never hardcodes another repo's model
//! path.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;

use proxima_tensor::cpu::{
    arena_packed_node_count, build_static_arena_with_constants, epilogue_fuse_reset,
    evaluate_named, evaluate_named_with_arena, layer_norm_cluster_reset, layer_norm_cluster_totals,
    rewrite_engine_depth_fires, rewrite_engine_reset,
};

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

fn named_inputs<'a>(
    initializers: &'a [(String, Vec<f32>)],
    graph_inputs: &'a [String],
    input_ids: &'a [f32],
    attention_mask: &'a [f32],
    token_type_ids: &'a [f32],
) -> Vec<(&'a str, &'a [f32])> {
    let mut named: Vec<(&str, &[f32])> = initializers
        .iter()
        .map(|(name, data)| (name.as_str(), data.as_slice()))
        .collect();
    for name in graph_inputs {
        let data: &[f32] = match name.as_str() {
            "input_ids" => input_ids,
            "attention_mask" => attention_mask,
            "token_type_ids" => token_type_ids,
            other => panic!("unexpected graph input {other:?}"),
        };
        named.push((name.as_str(), data));
    }
    named
}

fn cls_normalize(data: &[f32]) -> Vec<f32> {
    let cls = &data[0..384];
    let norm = cls.iter().map(|value| value * value).sum::<f32>().sqrt();
    cls.iter().map(|&value| value / norm).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[test]
fn arena_fusion_matches_evaluate_named_fused_bit_for_bit_on_real_bge() {
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

    let items = sentences();
    let mut arena_builds = 0usize;
    let mut arm_a_embeddings = Vec::new();
    let mut arm_c_embeddings = Vec::new();
    let mut any_packed = false;
    let mut depth1_fires_total = 0u64;
    let mut depth2_fires_total = 0u64;
    let mut any_ln_cluster_hits = 0u64;

    for (label, tokens) in &items {
        let mut pins = BTreeMap::new();
        pins.insert("batch_size", 1u64);
        pins.insert("sequence_length", tokens.len() as u64);
        let lowered = proxima_onnx::lower::lower_graph_pinned(graph, &pins)
            .expect("lower BGE-small with pinned symbolic axes");
        let output = lowered
            .graph_outputs
            .first()
            .expect("last_hidden_state output")
            .1;

        let sequence_length = tokens.len();
        let input_ids: Vec<f32> = tokens.iter().map(|&id| id as f32).collect();
        let attention_mask = vec![1.0f32; sequence_length];
        let token_type_ids = vec![0.0f32; sequence_length];
        let named = named_inputs(
            &lowered.initializers,
            &lowered.graph_inputs,
            &input_ids,
            &attention_mask,
            &token_type_ids,
        );

        // Arm A: `evaluate_named` -- today's sealed fused-only baseline.
        let evaluated_a = evaluate_named(&lowered.program, &[], &named, &[output])
            .expect("evaluate_named on the real BGE graph");
        let (data_a, shape_a) = evaluated_a
            .get(output)
            .expect("arm A last_hidden_state present");
        assert_eq!(
            shape_a,
            &[1u64, sequence_length as u64, 384u64],
            "{label}: unexpected arm A shape"
        );

        // Arm C: arena + packing (law 6∘5) + fusion (law 1/2), built once
        // for this sentence's own pinned shape, engagement counters reset
        // immediately before the build/eval pair so this sentence's own
        // counts are isolated from any earlier sentence's.
        rewrite_engine_reset();
        epilogue_fuse_reset();
        layer_norm_cluster_reset();
        let constant_inputs: Vec<(&str, &[f32])> = lowered
            .initializers
            .iter()
            .map(|(name, data)| (name.as_str(), data.as_slice()))
            .collect();
        let mut arena =
            build_static_arena_with_constants(&lowered.program, &[], &[output], &constant_inputs)
                .expect("build packed+fused arena");
        arena_builds += 1;
        any_packed |= arena_packed_node_count(&arena) > 0;
        // Build-time admission evidence -- `run_rewrite_worklist` runs
        // exactly once, inside `build_static_arena_with_constants` above,
        // never inside the `evaluate_named_with_arena` call below.
        let (depth1, depth2) = rewrite_engine_depth_fires();
        depth1_fires_total += depth1;
        depth2_fires_total += depth2;

        let evaluated_c = evaluate_named_with_arena(&mut arena, &named)
            .expect("evaluate_named_with_arena on the real BGE graph");
        // Runtime firing evidence: BGE's own 25 `LayerNorm` sites are law 2
        // cluster upgrades (`docs/discipline.md` ROW 204) -- every one of
        // them is REMOVED from the single-hop `epilogue_fuse_fire_at` map
        // once upgraded (`cpu.rs`'s own "superseded by a cluster upgrade"
        // comment), so `epilogue_fuse_totals`'s runtime hit count is
        // legitimately zero on this graph; `layer_norm_cluster_totals` is
        // the runtime counter that must be nonzero instead -- the exact
        // mirror of `real_mnist_accuracy.rs`'s own assertion in the other
        // direction (`ln_cluster_hits == 0` on mnist, which has no
        // `LayerNorm` site at all).
        let (ln_hits, ..) = layer_norm_cluster_totals();
        any_ln_cluster_hits += ln_hits;
        let (data_c, shape_c) = evaluated_c
            .get(output)
            .expect("arm C last_hidden_state present");
        assert_eq!(
            shape_c,
            &[1u64, sequence_length as u64, 384u64],
            "{label}: unexpected arm C shape"
        );

        assert_eq!(
            data_a.len(),
            data_c.len(),
            "{label}: arm A and arm C last_hidden_state length must match"
        );
        assert!(
            data_a
                .iter()
                .zip(data_c.iter())
                .all(|(&left, &right)| left.to_bits() == right.to_bits()),
            "{label}: arena fusion path (arm C) diverged from evaluate_named fusion path (arm A) -- not bit-identical"
        );

        arm_a_embeddings.push(cls_normalize(data_a));
        arm_c_embeddings.push(cls_normalize(data_c));
    }

    assert_eq!(
        arena_builds,
        items.len(),
        "expected exactly one arena build per distinct pinned sentence length, not per call"
    );
    assert!(
        any_packed,
        "engagement N==0 is RED: no width-tile node was packed on the real BGE graph"
    );
    assert!(
        depth1_fires_total > 0,
        "engagement N==0 is RED: law 1/2 admission never fired at arena-build time"
    );
    assert!(
        depth2_fires_total > 0,
        "engagement N==0 is RED: law 2's layer-norm cluster upgrade never fired at arena-build time"
    );
    assert!(
        any_ln_cluster_hits > 0,
        "engagement N==0 is RED: layer-norm cluster fusion never fired at runtime in the arena path"
    );

    let similar = cosine(&arm_c_embeddings[0], &arm_c_embeddings[1]);
    let dissimilar_a = cosine(&arm_c_embeddings[0], &arm_c_embeddings[2]);
    let dissimilar_b = cosine(&arm_c_embeddings[1], &arm_c_embeddings[2]);
    println!(
        "arena-path cosine(A,B)={similar:.6} cosine(A,C)={dissimilar_a:.6} cosine(B,C)={dissimilar_b:.6}"
    );
    assert!(
        (similar - 0.936311).abs() < 1e-5,
        "cosine(A,B) drifted from the sealed oracle: {similar:.6}"
    );
    assert!(
        (dissimilar_a - 0.378777).abs() < 1e-5,
        "cosine(A,C) drifted from the sealed oracle: {dissimilar_a:.6}"
    );
    assert!(
        (dissimilar_b - 0.334176).abs() < 1e-5,
        "cosine(B,C) drifted from the sealed oracle: {dissimilar_b:.6}"
    );
}
