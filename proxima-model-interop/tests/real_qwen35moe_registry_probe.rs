//! `#[ignore]`d, real-blob probe: does the BUILTIN registry
//! (`ArchitectureRegistry::with_builtin`) know what to do with a real
//! `qwen3.6:35b-a3b` (`general.architecture = qwen35moe`) checkpoint, or
//! does it fall through to `DenseArch`'s default and reject it?
//! `with_builtin`'s own doc: unmatched names fall back to `DenseArch`
//! (`crate::dense::DENSE`), not `Qwen35Arch` (registered under the exact
//! name `"qwen35"`, which `"qwen35moe"` does not match) -- this test names
//! which typed error that fallback produces against the real file, rather
//! than asserting it from reading the code alone.
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
use proxima_model_interop::{ArchitectureRegistry, LoadedModel};

#[proxima::test]
#[ignore = "requires a real, local qwen3.6:35b-a3b GGUF blob; set PROXIMA_QWEN35MOE_GGUF"]
async fn builtin_registry_reports_a_typed_rejection_for_a_real_qwen35moe_blob() {
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

    let registry = ArchitectureRegistry::with_builtin();
    match LoadedModel::load_with_registry(&parsed, file_bytes, &registry) {
        Err(error) => {
            eprintln!(
                "builtin registry rejects this checkpoint (Qwen35Arch is registered under \
                 \"qwen35\", not \"qwen35moe\", so resolve() falls back to DenseArch): {error}"
            );
        }
        Ok(_) => panic!(
            "expected the DenseArch fallback to reject a qwen35moe checkpoint (heterogeneous \
             head_count_kv / expert-routed FFN, neither of which DenseArch's own dense-only \
             tensor set expects); got Ok instead -- either this checkpoint's own metadata \
             changed, or the fallback silently started accepting it"
        ),
    }
}
