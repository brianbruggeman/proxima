//! Regenerates `docs/op_coverage.md` from
//! [`proxima_onnx::op_coverage::render_markdown`]'s parse of
//! `src/lower.rs`'s own `lower_node` match arms. Run after adding, removing,
//! or renaming an op arm in that match:
//!
//! ```sh
//! cargo run -p proxima-onnx --example generate_op_coverage_doc
//! ```
#![allow(clippy::expect_used)]

use std::fs;
use std::path::Path;

use proxima_onnx::op_coverage::render_markdown;

const LOWER_RS_SOURCE: &str = include_str!("../src/lower.rs");

fn main() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let output_path = Path::new(manifest_dir).join("docs").join("op_coverage.md");
    fs::create_dir_all(output_path.parent().expect("docs dir has a parent")).expect("create docs dir");
    let body = render_markdown(LOWER_RS_SOURCE).expect("lower.rs still has both parser markers");
    fs::write(&output_path, body).expect("write op_coverage.md");
    println!("wrote {}", output_path.display());
}
