//! `#[ignore]`d, real-blob probes for the `gemma4` architecture handler:
//! does the BUILTIN registry (`ArchitectureRegistry::with_builtin`) route a
//! real `gemma4` checkpoint to its own entry, does `gemma4::from_metadata`
//! preserve the header's per-layer KV-head and sliding-window-pattern
//! arrays, and does the header declare the tensor count the checkpoint's own
//! tensor directory holds. Header-only, same contract as
//! `real_qwen35moe_registry_probe.rs`: `parse_complete` reads the directory
//! while the mapping stays demand-paged, and nothing here forces the
//! multi-GB expert payload resident.
//!
//! `gemma4` names TWO distinct real checkpoints on this host, resolved by
//! manifest identity (`~/.ollama/models/manifests/registry.ollama.ai/...`),
//! never by "first gemma4 thing found":
//!
//! - `batiai/gemma4-26b:latest` -- the MoE 26B-A4B checkpoint
//!   (`general.architecture = gemma4`, `expert_count = 128`), blob digest
//!   `sha256-ea549b7688d4c95019754880c21e3f29c58c985a7a1c3b37b9eebd0a95224129`
//!   (13,286,728,576 bytes). [`GEMMA4_MOE_DEFAULT_PATH`] hardcodes that
//!   digest's blob path the same way `test_support.rs`'s own
//!   `qwen35moe_gguf_path` hardcodes its real blob; `PROXIMA_GEMMA4_GGUF`
//!   overrides it.
//! - `library/gemma4:e2b-it-qat` -- the dense, matformer-style E2B
//!   checkpoint (`expert_count = 0`, per-layer `feed_forward_length`
//!   array), blob digest
//!   `sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd`
//!   (3,349,514,112 bytes). [`GEMMA4_E2B_DEFAULT_PATH`] hardcodes that
//!   digest's blob path; `PROXIMA_GEMMA4_E2B_GGUF` overrides it.
//!
//! Each test resolves its own env var and, when neither the override nor the
//! default path exists on this host, fails loudly naming the env var and the
//! path ([`require_fixture`], the integration-test copy of the crate-private
//! `src/test_support.rs::require_fixture` that `f412eefe` introduced): an
//! `#[ignore]`d test only runs when explicitly asked for, so returning
//! success having executed nothing would be indistinguishable from a pass.

#![cfg(feature = "std")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::File;

use proxima_gguf::parse_complete;
use proxima_model_interop::ArchitectureRegistry;
use proxima_model_interop::gemma4::from_metadata;

/// `batiai/gemma4-26b:latest` manifest's own model layer digest -- see this
/// module's own doc for the manifest path and byte count.
const GEMMA4_MOE_DEFAULT_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/\
     sha256-ea549b7688d4c95019754880c21e3f29c58c985a7a1c3b37b9eebd0a95224129";

/// `library/gemma4:e2b-it-qat` manifest's own model layer digest -- see this
/// module's own doc for the manifest path and byte count.
const GEMMA4_E2B_DEFAULT_PATH: &str = "/Users/brianbruggeman/.ollama/models/blobs/\
     sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";

fn gemma4_moe_gguf_path() -> String {
    std::env::var("PROXIMA_GEMMA4_GGUF").unwrap_or_else(|_| GEMMA4_MOE_DEFAULT_PATH.to_string())
}

fn gemma4_e2b_gguf_path() -> String {
    std::env::var("PROXIMA_GEMMA4_E2B_GGUF")
        .unwrap_or_else(|_| GEMMA4_E2B_DEFAULT_PATH.to_string())
}

fn require_fixture(path: &str, env_var: &str) {
    assert!(
        std::path::Path::new(path).exists(),
        "no host-local gguf fixture at {path}: set {env_var} to a valid checkpoint path, or stage one at this default path"
    );
}

fn parse_real_blob(path: &str) -> (memmap2::Mmap, proxima_gguf::pipe::ParsedGguf) {
    let file = File::open(path).unwrap_or_else(|error| panic!("open {path}: {error}"));
    let mapping = unsafe { memmap2::Mmap::map(&file) }.expect("mmap the real checkpoint read-only");
    let parsed =
        parse_complete(&mapping).expect("parses the real checkpoint's own GGUF header");
    (mapping, parsed)
}

/// The MoE 26B-A4B checkpoint (`batiai/gemma4-26b:latest`) -- the variant
/// this test was originally written for. Skips when that specific manifest's
/// blob is absent; never silently routes to a different `gemma4` variant.
#[proxima::test]
#[ignore = "requires the real, local gemma4 26B-A4B MoE GGUF blob (batiai/gemma4-26b:latest); set PROXIMA_GEMMA4_GGUF"]
async fn builtin_registry_routes_real_gemma4_header_with_its_tensor_count() {
    let path = gemma4_moe_gguf_path();
    require_fixture(&path, "PROXIMA_GEMMA4_GGUF");
    let (_mapping, parsed) = parse_real_blob(&path);

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

    assert_eq!(
        parsed.tensors.len(),
        658,
        "real gemma4 26B-A4B checkpoint declares 658 tensors"
    );
}

/// The dense, matformer-style E2B checkpoint (`library/gemma4:e2b-it-qat`)
/// -- present on this host even when the MoE 26B-A4B blob above is not.
/// Same assertion shape as the MoE test: registry routing, every header
/// field `gemma4::from_metadata` derives, and the tensor count -- with
/// values read from this checkpoint's own real GGUF header (dumped via a
/// throwaway instrumented run of this same parse path; see this test's own
/// assertions below for the values that run captured).
#[proxima::test]
#[ignore = "requires the real, local gemma4 E2B dense GGUF blob (library/gemma4:e2b-it-qat); set PROXIMA_GEMMA4_E2B_GGUF"]
async fn builtin_registry_routes_real_gemma4_e2b_dense_header() {
    let path = gemma4_e2b_gguf_path();
    require_fixture(&path, "PROXIMA_GEMMA4_E2B_GGUF");
    let (_mapping, parsed) = parse_real_blob(&path);

    let registry = ArchitectureRegistry::with_builtin();
    let route = registry
        .resolve(&parsed)
        .expect("builtin registry resolves the real gemma4 E2B header");
    assert_eq!(
        route.name(),
        "gemma4",
        "the dense E2B checkpoint is still general.architecture = gemma4, not a separate route"
    );

    let architecture =
        from_metadata(&parsed).expect("gemma4 hparams parse from the real E2B checkpoint header");

    assert_eq!(architecture.block_count, 35);
    assert_eq!(
        architecture.expert_count, 0,
        "the dense E2B checkpoint carries no expert_count key at all"
    );
    assert_eq!(
        architecture.expert_used_count, 0,
        "the dense E2B checkpoint carries no expert_used_count key at all"
    );
    assert_eq!(
        architecture.expert_feed_forward, 0,
        "the dense E2B checkpoint carries no expert_feed_forward_length key at all"
    );
    assert_eq!(architecture.embedding, 1536);
    assert_eq!(architecture.head_count, 8);
    assert_eq!(architecture.rms_epsilon, 1e-6);
    assert_eq!(architecture.key_length, 512);
    assert_eq!(architecture.value_length, 512);
    assert_eq!(architecture.sliding_window, 512);
    assert_eq!(architecture.key_length_swa, 256);
    assert_eq!(architecture.value_length_swa, 256);
    assert_eq!(architecture.rope_freq_base, 1_000_000.0);
    assert_eq!(architecture.rope_freq_base_swa, 10_000.0);
    assert_eq!(architecture.rope_dimension_count, 512);
    assert_eq!(architecture.rope_dimension_count_swa, 256);
    assert_eq!(architecture.final_logit_softcapping, 30.0);
    assert_eq!(
        architecture.shared_kv_layers, 20,
        "the real E2B header's trailing 20 layers share KV with an earlier own-KV layer"
    );
    assert_eq!(
        architecture.ple_dim, 256,
        "the real E2B header carries a 256-wide per-layer-embedding (PLE) input"
    );

    let expected_kv_heads: Vec<u32> = std::iter::repeat_n(1u32, 35).collect();
    assert_eq!(
        architecture.kv_heads_by_layer, expected_kv_heads,
        "the real E2B header carries a uniform head_count_kv = 1 across every layer"
    );

    let expected_sliding_window_pattern: Vec<bool> = (0..35)
        .map(|layer: u32| !(layer + 1).is_multiple_of(5))
        .collect();
    assert_eq!(
        architecture.sliding_window_pattern, expected_sliding_window_pattern,
        "the real E2B header marks every 5th layer (4, 9, 14, ..., 34) as full attention"
    );

    let expected_feed_forward_by_layer: Vec<u32> = (0..35)
        .map(|layer: u32| if layer < 15 { 6144 } else { 12288 })
        .collect();
    assert_eq!(
        architecture.feed_forward_by_layer, expected_feed_forward_by_layer,
        "the real E2B header's matformer feed_forward_length array widens from 6144 to 12288 at layer 15"
    );

    assert_eq!(
        parsed.tensors.len(),
        541,
        "real gemma4 E2B dense checkpoint declares 541 tensors"
    );
}
