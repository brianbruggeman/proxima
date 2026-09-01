//! Regenerates `docs/compatibility.md` from
//! [`proxima_model_interop::capability`]'s tables -- the same data
//! `tests/capability_doc_drift.rs`'s drift guard re-derives on every test
//! run. Run it after any change to `src/capability.rs` or
//! `tests/capability_matrix.rs`'s own cell set:
//!
//! ```sh
//! cargo run -p proxima-model-interop --example generate_compatibility_doc --features metal
//! ```
//!
//! `--features metal` is required: the quantized-packed-format section
//! reads `omega::msl::PackedCodec`, which only this crate's `metal` feature
//! pulls in (`Cargo.toml`'s own `metal` feature comment explains why).
#![allow(clippy::expect_used)]

use std::fs;
use std::path::Path;

use proxima_model_interop::capability::render_ggml_matrix_markdown;
#[cfg(feature = "metal")]
use proxima_model_interop::capability::quant_format::render_markdown as render_quant_format_markdown;

fn document_body() -> String {
    let mut body = String::new();
    body.push_str("# proxima-model-interop compatibility matrix\n\n");
    body.push_str(
        "Generated from `src/capability.rs`'s own tables by \
         `examples/generate_compatibility_doc.rs` -- do not hand-edit. \
         Regenerate with `cargo run -p proxima-model-interop --example \
         generate_compatibility_doc --features metal` after any change to \
         `src/capability.rs` or `tests/capability_matrix.rs`'s cell set. \
         `tests/capability_doc_drift.rs` fails the build if this file falls \
         out of sync.\n\n",
    );
    body.push_str("## GGML codec x topology x backend\n\n");
    body.push_str(&render_ggml_matrix_markdown());
    body.push_str("\n## Quantized packed-format coverage\n\n");
    #[cfg(feature = "metal")]
    body.push_str(&render_quant_format_markdown());
    #[cfg(not(feature = "metal"))]
    body.push_str("(this section needs `--features metal`; not rendered by this build)\n");
    body
}

fn main() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let output_path = Path::new(manifest_dir).join("docs").join("compatibility.md");
    fs::create_dir_all(output_path.parent().expect("docs dir has a parent")).expect("create docs dir");
    fs::write(&output_path, document_body()).expect("write compatibility.md");
    println!("wrote {}", output_path.display());
}
