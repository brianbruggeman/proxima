#![allow(clippy::expect_used)]

//! Re-parses `src/lower.rs`'s own `lower_node` match arms
//! ([`proxima_onnx::op_coverage::render_markdown`]) on every test run and
//! asserts it is byte-identical to the committed `docs/op_coverage.md`.
//! An op added, removed, or renamed in that match without regenerating the
//! doc fails this test with the regen command in its own message.

use proxima_onnx::op_coverage::render_markdown;

const LOWER_RS_SOURCE: &str = include_str!("../src/lower.rs");
const COMMITTED_DOC: &str = include_str!("../docs/op_coverage.md");

#[test]
fn op_coverage_doc_matches_lower_node_match_arms() {
    let rendered = render_markdown(LOWER_RS_SOURCE).expect("lower.rs still has both parser markers");
    assert_eq!(
        COMMITTED_DOC, rendered,
        "docs/op_coverage.md no longer matches src/lower.rs's lower_node match arms. \
         Regenerate with: cargo run -p proxima-onnx --example generate_op_coverage_doc"
    );
}
