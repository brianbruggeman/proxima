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
