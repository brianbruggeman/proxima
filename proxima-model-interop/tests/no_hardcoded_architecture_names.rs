//! guard: the forward / decode / fusion pipeline dispatches on model
//! CAPABILITY read from the descriptor, never on a hardcoded architecture-name
//! literal. the one legitimate place a name string drives a choice is the
//! registry that maps `general.architecture` -> impl at bind time
//! (architecture.rs); everything downstream asks the descriptor what the model
//! DOES, not which model it is. a name literal reappearing in a pipeline file
//! is the regression this test exists to catch.

use std::path::Path;

// model-name tokens that must not drive control flow in the forward path.
const ARCH_NAME_LITERALS: &[&str] = &["qwen2", "qwen35moe", "qwen35", "gemma4"];

// files that compose / execute the forward pass. paths are crate-relative.
const PIPELINE_FILES: &[&str] = &[
    "src/dense.rs",
    "src/generate/pregather.rs",
    "src/generate/decode.rs",
];

// a documented, justified exception is a single row here with a why. the target
// is an empty allowlist; anything added is reviewed as a principle exception.
const ALLOWLIST: &[(&str, u32, &str)] = &[];

/// `true` when `line` shapes a CONTROL-FLOW decision off the quoted literal
/// (an equality/inequality comparison, a `matches!` pattern, or a
/// `.name()` read feeding either) rather than merely carrying the name as
/// inert payload -- e.g. `InteropError::PreGatherExecutionUnsupported`'s
/// own `architecture: String::from("qwen35moe")` field, built only AFTER a
/// capability check upstream has already gated entry to that arm, reports
/// which family a qwen35moe-only operation requires without itself
/// deciding anything. The invariant this test enforces is "dispatch on
/// capability, never on name" -- a name string that never reaches a branch
/// condition is not a dispatch, so it is not the defect this test exists to
/// catch.
fn shapes_control_flow(line: &str) -> bool {
    line.contains("==") || line.contains("!=") || line.contains("matches!") || line.contains(".name()")
}

/// `true` once `line` is a `mod` item declaration -- the point past which a
/// preceding run of `#[cfg(...)]` attributes containing `test` marks a
/// trailing inline unit-test module (this crate's own convention: every
/// pipeline file's unit tests are one `mod` block at the end of the file,
/// gated `#[cfg(test)]` or a compound `#[cfg(all(test, ...))]`), not a
/// single production item that merely carries a `#[cfg(any(test, ...))]`
/// reachability gate (e.g. a helper also exercised by a benchmark feature).
fn declares_a_module(line: &str) -> bool {
    line.trim_start().starts_with("mod ")
}

#[test]
fn pipeline_dispatches_on_capability_not_name() {
    let crate_root = env!("CARGO_MANIFEST_DIR");
    let mut violations = Vec::new();

    for rel in PIPELINE_FILES {
        let path = Path::new(crate_root).join(rel);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("read {rel}: {err}"));

        let mut pending_attribute_names_test = false;
        for (index, line) in src.lines().enumerate() {
            let line_number = (index as u32) + 1;
            let trimmed = line.trim_start();

            if trimmed.starts_with("#[") {
                if trimmed.contains("test") {
                    pending_attribute_names_test = true;
                }
                continue;
            }
            if declares_a_module(trimmed) && pending_attribute_names_test {
                // a test-gated module, by this crate's own trailing-module
                // convention, legitimately names real models in fixtures --
                // stop scanning the rest of this file.
                break;
            }
            if !trimmed.is_empty() {
                pending_attribute_names_test = false;
            }
            if trimmed.starts_with("//") || trimmed.starts_with('*') {
                continue;
            }
            if !shapes_control_flow(line) {
                continue;
            }
            for name in ARCH_NAME_LITERALS {
                if !line.contains(&format!("\"{name}\"")) {
                    continue;
                }
                if ALLOWLIST
                    .iter()
                    .any(|(file, row, _)| file == rel && *row == line_number)
                {
                    continue;
                }
                violations.push(format!("{rel}:{line_number}: hardcoded \"{name}\" -> {trimmed}"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "pipeline must dispatch on config-derived capability, not architecture name; \
         interpret the name once at bind and read a descriptor field downstream:\n{}",
        violations.join("\n")
    );
}
