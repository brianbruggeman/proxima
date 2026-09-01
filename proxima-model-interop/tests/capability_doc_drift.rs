//! The point of `docs/compatibility.md` existing at all: fail loudly the
//! moment `src/capability.rs`'s tables say something the committed doc
//! doesn't. `capability_doc_content_matches_the_generated_tables` re-derives
//! the doc body in-memory from the exact same tables
//! `examples/generate_compatibility_doc.rs` reads and asserts byte equality
//! against the committed file -- if a codec's status changes in
//! `src/capability.rs` without regenerating the doc, this test goes red
//! with the regen command named in its own failure message, not a stale
//! green.

use proxima_model_interop::capability::render_ggml_matrix_markdown;
#[cfg(feature = "metal")]
use proxima_model_interop::capability::quant_format::render_markdown as render_quant_format_markdown;

const REGEN_COMMAND: &str =
    "cargo run -p proxima-model-interop --example generate_compatibility_doc --features metal";

const COMMITTED_DOC: &str = include_str!("../docs/compatibility.md");

/// The always-buildable half of the drift guard: the GGML codec/topology/
/// backend section renders identically regardless of which optional
/// features this test binary itself was built with.
#[test]
fn ggml_matrix_section_matches_committed_doc() {
    let rendered = render_ggml_matrix_markdown();
    assert!(
        COMMITTED_DOC.contains(&rendered),
        "docs/compatibility.md's GGML matrix section does not match src/capability.rs's \
         GGML_CAPABILITY_TABLE. Regenerate with: {REGEN_COMMAND}"
    );
}

/// The `metal`-gated half: needs `omega::msl::PackedCodec`, so it can only
/// run in the same feature configuration the doc itself was generated
/// under.
#[cfg(feature = "metal")]
#[test]
fn quant_format_section_matches_committed_doc() {
    let rendered = render_quant_format_markdown();
    assert!(
        COMMITTED_DOC.contains(&rendered),
        "docs/compatibility.md's quantized-packed-format section does not match \
         src/capability.rs's quant_format table. Regenerate with: {REGEN_COMMAND}"
    );
}

/// Full-document byte equality, only checkable under the same feature set
/// the committed file was generated with (`metal`) -- the strongest form of
/// this guard, run whenever that feature is on.
#[cfg(feature = "metal")]
#[test]
fn full_document_is_byte_identical_to_the_generator_output() {
    let mut expected = String::new();
    expected.push_str("# proxima-model-interop compatibility matrix\n\n");
    expected.push_str(
        "Generated from `src/capability.rs`'s own tables by \
         `examples/generate_compatibility_doc.rs` -- do not hand-edit. \
         Regenerate with `cargo run -p proxima-model-interop --example \
         generate_compatibility_doc --features metal` after any change to \
         `src/capability.rs` or `tests/capability_matrix.rs`'s cell set. \
         `tests/capability_doc_drift.rs` fails the build if this file falls \
         out of sync.\n\n",
    );
    expected.push_str("## GGML codec x topology x backend\n\n");
    expected.push_str(&render_ggml_matrix_markdown());
    expected.push_str("\n## Quantized packed-format coverage\n\n");
    expected.push_str(&render_quant_format_markdown());

    assert_eq!(
        COMMITTED_DOC, expected,
        "docs/compatibility.md is not byte-identical to what src/capability.rs's tables \
         produce today. Regenerate with: {REGEN_COMMAND}"
    );
}
