//! `#[ignore]`d, real-blob probe for SLICE 1 of the `gemma4` architecture
//! handler: does the BUILTIN registry (`ArchitectureRegistry::with_builtin`)
//! route a real `gemma4` checkpoint to its own entry, does
//! `gemma4::from_metadata` preserve the header's per-layer KV-head and
//! sliding-window-pattern arrays, and does `gemma4_tensor_names` produce
//! EXACTLY the real checkpoint's own tensor directory. Header-only, same
//! contract as `real_qwen35moe_registry_probe.rs`: `parse_complete` reads
//! the directory while the mapping stays demand-paged, and nothing here
//! forces the multi-GB expert payload resident.
//!
//! Gated on `PROXIMA_GEMMA4_GGUF` (absolute path to the real blob); skips
//! with a clear message when unset, never a false pass.

#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeSet;
use std::fs::File;

use proxima_gguf::parse_complete;
use proxima_model_interop::ArchitectureRegistry;
use proxima_model_interop::gemma4::{from_metadata, gemma4_tensor_names};

#[proxima::test]
#[ignore = "requires a real, local gemma4 GGUF blob; set PROXIMA_GEMMA4_GGUF"]
async fn builtin_registry_routes_real_gemma4_header_with_exact_tensor_directory() {
    let Ok(path) = std::env::var("PROXIMA_GEMMA4_GGUF") else {
        eprintln!("skipping: PROXIMA_GEMMA4_GGUF not set");
        return;
    };
    let file = File::open(&path).unwrap_or_else(|error| panic!("open {path}: {error}"));
    let mapping = unsafe { memmap2::Mmap::map(&file) }.expect("mmap the real checkpoint read-only");
    let file_bytes: &[u8] = &mapping;

    let parsed = parse_complete(file_bytes).expect("parses the real checkpoint's own GGUF header");

    let registry = ArchitectureRegistry::with_builtin();
    let route = registry
        .resolve(&parsed)
        .expect("builtin registry resolves the real gemma4 header");
    assert_eq!(route.name(), "gemma4", "gemma4 must not fall back to dense");

    let architecture =
        from_metadata(&parsed).expect("gemma4 hparams parse from the real checkpoint header");

    assert_eq!(architecture.block_count, 30);
    assert_eq!(architecture.expert_count, 128);
    assert_eq!(architecture.expert_used_count, 8);
    assert_eq!(architecture.embedding, 2816);
    assert_eq!(architecture.feed_forward, 2112);
    assert_eq!(architecture.expert_feed_forward, 704);
    assert_eq!(architecture.head_count, 16);
    assert_eq!(architecture.rms_epsilon, 1e-6);
    assert_eq!(architecture.key_length, 512);
    assert_eq!(architecture.value_length, 512);
    assert_eq!(architecture.sliding_window, 1024);
    assert_eq!(architecture.key_length_swa, 256);
    assert_eq!(architecture.value_length_swa, 256);
    assert_eq!(architecture.rope_freq_base, 1_000_000.0);
    assert_eq!(architecture.rope_freq_base_swa, 10_000.0);
    assert_eq!(architecture.rope_dimension_count, 512);
    assert_eq!(architecture.rope_dimension_count_swa, 256);
    assert_eq!(architecture.final_logit_softcapping, 30.0);

    let expected_kv_heads: Vec<u32> = (0..30)
        .map(|layer: u32| if (layer + 1).is_multiple_of(6) { 2 } else { 8 })
        .collect();
    assert_eq!(
        architecture.kv_heads_by_layer, expected_kv_heads,
        "real header alternates 8/8/8/8/8/2 kv-heads per 6-layer block"
    );

    let expected_sliding_window_pattern: Vec<bool> = (0..30)
        .map(|layer: u32| !(layer + 1).is_multiple_of(6))
        .collect();
    assert_eq!(
        architecture.sliding_window_pattern, expected_sliding_window_pattern,
        "real header marks every 6th layer (5, 11, 17, 23, 29) as full attention"
    );
    let computed_names: BTreeSet<String> =
        gemma4_tensor_names(&architecture).into_iter().collect();
    let real_names: BTreeSet<String> = parsed
        .tensors
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect();

    let missing: Vec<&String> = real_names.difference(&computed_names).collect();
    let extra: Vec<&String> = computed_names.difference(&real_names).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "gemma4_tensor_names must exactly match the real tensor directory: missing={missing:?} extra={extra:?}"
    );
    assert_eq!(
        parsed.tensors.len(),
        658,
        "real gemma4 checkpoint declares 658 tensors"
    );
    assert_eq!(computed_names.len(), 658);
}
