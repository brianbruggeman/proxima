//! Parses `lower_node`'s own match arms out of `src/lower.rs`'s source
//! text and renders them to `docs/op_coverage.md` -- no second, hand-typed
//! op list to fall out of sync with the one `lower_node` actually
//! dispatches on (`lower_node` itself is private to [`crate::lower`], the
//! only entry point a caller reaches it through is
//! [`crate::lower_graph`]). `examples/generate_op_coverage_doc.rs` writes
//! the doc; `tests/op_coverage_doc_drift.rs` re-parses the same source on
//! every test run and fails if the committed doc no longer matches what
//! `lower_node` supports today.
//!
//! Parsing the source instead of hand-copying the op list is the point:
//! `lower_node`'s match is one function, one file (`src/lower.rs:328-378`
//! at the time this module was written) -- restructuring it into a
//! caller-iterable table (the shape `proxima-model-interop`'s
//! `capability.rs` uses) would mean threading a runtime dispatch table
//! through every `lower_*` helper's distinct signature, which is a
//! materially larger change than this doc generator needs. Reading the
//! match's own arms back out of its source text gets the same
//! cannot-drift guarantee without touching `lower.rs` at all.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;

use thiserror::Error;

const START_MARKER: &str = "fn lower_node(";
const END_MARKER: &str = "other => Err(LowerError::UnsupportedOp";

/// [`parse_supported_ops`] found a `lower.rs` shape it has never seen --
/// this parser is out of date with `lower_node`'s own source, not a
/// runtime input the caller supplied wrongly.
#[derive(Debug, Error)]
pub enum OpCoverageParseError {
    #[error("lower.rs no longer defines `{START_MARKER}` -- op_coverage.rs's parser is out of date")]
    MissingStartMarker,
    #[error("lower.rs's lower_node no longer ends its match on `{END_MARKER}` -- op_coverage.rs's parser is out of date")]
    MissingEndMarker,
}

/// Every ONNX op name `lower_node`'s match dispatches on, in source
/// order -- parsed from `        "OpName" => ...` lines between the
/// `fn lower_node(` marker and the `other => Err(LowerError::UnsupportedOp`
/// arm that closes its match.
pub fn parse_supported_ops(lower_rs_source: &str) -> Result<Vec<&str>, OpCoverageParseError> {
    let after_start = lower_rs_source.split_once(START_MARKER).map(|(_, rest)| rest).ok_or(OpCoverageParseError::MissingStartMarker)?;
    let body = after_start.split_once(END_MARKER).map(|(body, _)| body).ok_or(OpCoverageParseError::MissingEndMarker)?;

    Ok(body
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            let rest = trimmed.strip_prefix('"')?;
            let (op_name, rest) = rest.split_once('"')?;
            rest.trim_start().starts_with("=>").then_some(op_name)
        })
        .collect())
}

/// Renders [`parse_supported_ops`]'s output to the exact markdown body
/// `docs/op_coverage.md` commits.
///
/// # Errors
///
/// [`OpCoverageParseError`] if `lower_rs_source` no longer has the shape
/// this parser expects.
pub fn render_markdown(lower_rs_source: &str) -> Result<String, OpCoverageParseError> {
    let ops = parse_supported_ops(lower_rs_source)?;
    let mut out = String::new();
    let _ = writeln!(out, "# proxima-onnx op lowering coverage\n");
    let _ = writeln!(
        out,
        "Generated from `src/lower.rs`'s own `lower_node` match arms by \
         `examples/generate_op_coverage_doc.rs` -- do not hand-edit. \
         Regenerate with `cargo run -p proxima-onnx --example \
         generate_op_coverage_doc`. `tests/op_coverage_doc_drift.rs` fails \
         the build if this file falls out of sync.\n"
    );
    let _ = writeln!(out, "{} ops lower today; any ONNX op not in this list hits `LowerError::UnsupportedOp`.\n", ops.len());
    out.push_str("| onnx op |\n");
    out.push_str("| --- |\n");
    for op in ops {
        let _ = writeln!(out, "| {op} |");
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::parse_supported_ops;

    const SAMPLE: &str = "fn lower_node(program: &mut Vec<Op>) -> Result<(), LowerError> {\n    match node.op_type {\n        \"Add\" => lower_binary(program, values, node, ScalarOp::Add),\n        \"Sub\" => lower_binary(program, values, node, ScalarOp::Subtract),\n        other => Err(LowerError::UnsupportedOp { name: node.name.to_string(), op_type: other.to_string() }),\n    }\n}\n";

    #[test]
    fn parses_every_quoted_arm_between_the_two_markers() {
        assert_eq!(parse_supported_ops(SAMPLE).expect("sample source has both markers"), alloc::vec!["Add", "Sub"]);
    }

    #[test]
    fn parses_the_real_lower_rs_and_finds_at_least_the_forty_one_documented_ops() {
        let real_source = include_str!("lower.rs");
        let ops = parse_supported_ops(real_source).expect("real lower.rs has both markers");
        assert!(ops.len() >= 41, "expected at least 41 supported ops in lower_node's match, found {}: {ops:?}", ops.len());
    }
}
