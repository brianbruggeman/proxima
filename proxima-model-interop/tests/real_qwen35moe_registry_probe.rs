//! `#[ignore]`d, real-blob probe: does the BUILTIN registry
//! (`ArchitectureRegistry::with_builtin`) know what to do with a real
//! `qwen3.6:35b-a3b` (`general.architecture = qwen35moe`) checkpoint reaches
//! its dedicated registry entry and preserves the header's per-layer KV-head
//! configuration. This is intentionally header-only: parse_complete reads the
//! directory while the mapping remains demand-paged, and resolve never asks
//! the 22 GiB expert payload to become resident.
//!
//! Gated on `PROXIMA_QWEN35MOE_GGUF` (absolute path to the real blob);
//! skips with a clear message when unset, never a false pass. Mmaps the
//! real file read-only -- resolve + bind only touch the metadata header and
//! the (small, non-expert) tensors `DenseArch::bind` reads before its own
//! typed rejection fires, so this never faults in the multi-GB expert
//! tensor pages.

#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::File;

use proxima_gguf::parse_complete;
use proxima_model_interop::{ArchitectureRegistry, LoadedModel, architecture_from_metadata};

#[proxima::test]
#[ignore = "requires a real, local qwen3.6:35b-a3b GGUF blob; set PROXIMA_QWEN35MOE_GGUF"]
async fn builtin_registry_routes_real_qwen35moe_header_with_per_layer_kv_configuration() {
    let Ok(path) = std::env::var("PROXIMA_QWEN35MOE_GGUF") else {
        eprintln!("skipping: PROXIMA_QWEN35MOE_GGUF not set");
        return;
    };
    let file = File::open(&path).unwrap_or_else(|error| panic!("open {path}: {error}"));
    let mapping = unsafe { memmap2::Mmap::map(&file) }.expect("mmap the real checkpoint read-only");
    let file_bytes: &[u8] = &mapping;

    let parsed = parse_complete(file_bytes).expect("parses the real checkpoint's own GGUF header");
    let general_architecture = parsed
        .metadata
        .iter()
        .find_map(|(key, value)| (key == "general.architecture").then_some(value.clone()));
    eprintln!("real blob general.architecture = {general_architecture:?}");
    for (key, value) in &parsed.metadata {
        if key.starts_with("qwen35moe.rope.") || key.starts_with("qwen35moe.ssm.v_head") {
            eprintln!("real blob {key} = {value:?}");
        }
    }

    let registry = ArchitectureRegistry::with_builtin();
    let route = registry
        .resolve(&parsed)
        .expect("builtin registry resolves the real header");
    assert_eq!(route.name(), "qwen35moe");

    let architecture = architecture_from_metadata(&parsed)
        .expect("the real header's per-layer KV-head array is configuration, not a parse error");
    let moe_architecture = proxima_model_interop::qwen35moe::from_metadata(&parsed)
        .expect("qwen35moe hparams preserve the hybrid layer configuration");
    assert_eq!(architecture.block_count, 40);
    assert_eq!(architecture.kv_heads_by_layer.len(), 40);
    for (layer, &kv_heads) in architecture.kv_heads_by_layer.iter().enumerate() {
        let expected = if (layer + 1).is_multiple_of(4) { 2 } else { 0 };
        assert_eq!(
            kv_heads, expected,
            "layer {layer} keeps its GGUF KV-head entry"
        );
    }
    assert!(
        architecture.uniform_kv_heads().is_err(),
        "a uniform dense builder must not erase the real SSM/attention distinction"
    );
    assert_eq!(moe_architecture.layer_kinds.len(), 40);
    assert_eq!(moe_architecture.expert_count, 256);

    let bound = route
        .bind(&parsed, file_bytes)
        .unwrap_or_else(|error| panic!("real qwen35moe bind failed: {error}"));
    assert_eq!(
        bound.router_roots.len(),
        40,
        "each real qwen35moe layer exposes the router node already used by its gather"
    );
    assert_eq!(
        bound.qwen35moe_layer_diagnostics.len(),
        40,
        "the bound program exposes every qwen35moe layer boundary"
    );
    eprintln!(
        "real qwen35moe bound: ops={} layers={} packed_weights={} owned_weights={} resident_bytes={}",
        bound.program.len(),
        bound.layer_roots.len(),
        bound.weights.packed().len(),
        bound.weights.owned().len(),
        bound.weights.resident_bytes(),
    );
    drop(bound);

    let loaded = LoadedModel::load_with_registry(&parsed, file_bytes, &registry)
        .expect("the real qwen35moe checkpoint loads through the builtin registry");
    assert_eq!(
        loaded.qwen35moe_layer_diagnostics().len(),
        40,
        "LoadedModel preserves every bound qwen35moe layer boundary"
    );
}

/// Header-only static-plan probe for ROW 591's `NodeId(6540)`
/// `"operand buffer missing at evaluation time"` failure: builds the SAME
/// width-13 one-evaluation prefill symbolic program
/// `decode.rs`'s `one_evaluation_prefill_programs` builds
/// (`qwen35moe_forward_program_at_width`, purely symbolic -- `input_leaf`
/// placeholders, no tensor bytes) and runs `bind::bind` +
/// `dead_resolved_nodes` + `node_retirement` + `node_last_reader` over it
/// directly, the exact admission pipeline `evaluate_named`/`quantized_eval`
/// run before ever touching a weight byte. Never faults the multi-GB expert
/// pages -- same header-only contract as the test above.
#[proxima::test]
#[ignore = "requires a real, local qwen3.6:35b-a3b GGUF blob; set PROXIMA_QWEN35MOE_GGUF"]
async fn real_qwen35moe_width_13_plan_names_node_6540() {
    let Ok(path) = std::env::var("PROXIMA_QWEN35MOE_GGUF") else {
        eprintln!("skipping: PROXIMA_QWEN35MOE_GGUF not set");
        return;
    };
    let file = File::open(&path).unwrap_or_else(|error| panic!("open {path}: {error}"));
    let mapping = unsafe { memmap2::Mmap::map(&file) }.expect("mmap the real checkpoint read-only");
    let file_bytes: &[u8] = &mapping;

    let parsed = parse_complete(file_bytes).expect("parses the real checkpoint's own GGUF header");
    let architecture = proxima_model_interop::qwen35moe::from_metadata(&parsed)
        .expect("qwen35moe hparams preserve the hybrid layer configuration");

    let (program, roots, _layer_roots, _moe_sites, _diagnostics) =
        proxima_model_interop::qwen35moe::qwen35moe_forward_program_at_width(
            &architecture,
            Some(13),
        )
        .expect("width-13 one-evaluation prefill program builds from header-only hparams");
    eprintln!(
        "width-13 program: ops={} logits_root={:?}",
        program.len(),
        roots.logits
    );

    let second_target = proxima_tensor::NodeId(6943);
    eprintln!(
        "node 6943 raw op = {:?}",
        program.get(second_target.0 as usize)
    );

    let target = proxima_tensor::NodeId(6540);
    assert!(
        (target.0 as usize) < program.len(),
        "node 6540 must exist in a {}-op program",
        program.len()
    );
    eprintln!("node 6540 raw op = {:?}", program[target.0 as usize]);

    let symbols = [13u64, 13u64];
    let shapes = proxima_tensor::shape::infer(&program, &symbols)
        .expect("shape inference over the real-width program");
    let resolved = proxima_tensor::bind::bind(
        &program,
        &shapes,
        &[roots.logits],
        proxima_tensor::NumericPolicy::bit_exact(),
    )
    .expect("bind admits the real width-13 program");
    eprintln!("resolved.len() = {}", resolved.len());

    let bound_position = resolved.iter().position(|computed| computed.node == target);
    match bound_position {
        Some(position) => {
            eprintln!(
                "node 6540 IS materialized at resolved position {position}: kind={:?} operands={:?}",
                resolved[position].kind,
                resolved[position]
                    .operands()
                    .iter()
                    .map(|(source, ..)| source.0)
                    .collect::<Vec<_>>()
            );
        }
        None => {
            eprintln!("node 6540 is NEVER materialized in `resolved` (absorbed or dead)");
        }
    }

    let dead = proxima_tensor::dead_resolved_nodes(&resolved, &[roots.logits]);
    eprintln!(
        "node 6540 in dead_resolved_nodes = {}",
        dead.contains(&target)
    );

    let last_reader = proxima_tensor::node_last_reader(&resolved, program.len());
    let reader_position = last_reader[target.0 as usize];
    eprintln!(
        "node 6540 last-reader resolved position = {} (u32::MAX means never read)",
        reader_position
    );
    if reader_position != u32::MAX {
        let reader = &resolved[reader_position as usize];
        eprintln!(
            "reader node={:?} kind_variant={}",
            reader.node,
            match &reader.kind {
                proxima_tensor::bind::BoundOpKind::Elementwise { .. } => "Elementwise",
                proxima_tensor::bind::BoundOpKind::Reduce { .. } => "Reduce",
                proxima_tensor::bind::BoundOpKind::RoundBatchedReduce { .. } => {
                    "RoundBatchedReduce"
                }
                proxima_tensor::bind::BoundOpKind::CachedAttention { .. } => "CachedAttention",
                proxima_tensor::bind::BoundOpKind::GatedDeltaNet { .. } => "GatedDeltaNet",
                proxima_tensor::bind::BoundOpKind::MoeTopK { .. } => "MoeTopK",
                proxima_tensor::bind::BoundOpKind::Iota => "Iota",
                proxima_tensor::bind::BoundOpKind::Constant { .. } => "Constant",
            }
        );
        if let proxima_tensor::bind::BoundOpKind::Reduce {
            epilogue_operands,
            operands,
            ..
        } = &reader.kind
        {
            eprintln!(
                "reader operands={:?} epilogue_operands={:?}",
                operands.iter().map(|(node, ..)| node.0).collect::<Vec<_>>(),
                epilogue_operands
                    .iter()
                    .map(|(node, ..)| node.0)
                    .collect::<Vec<_>>(),
            );
        }
    }

    let mut producer_position = None;
    let mut consumer_positions = Vec::new();
    for (position, computed) in resolved.iter().enumerate() {
        if computed.node == target {
            producer_position = Some(position);
        }
        if computed
            .operands()
            .iter()
            .any(|(source, ..)| *source == target)
        {
            consumer_positions.push(position);
        }
    }
    eprintln!(
        "node 6540 producer_position={producer_position:?} consumer_positions={consumer_positions:?}"
    );
}
