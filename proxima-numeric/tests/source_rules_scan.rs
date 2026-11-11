//! Source scans over production text across the crates that consume `proxima-numeric`.
//!
//! Two rules are asserted from the text itself, the way `omega/src/msl/layout_render_tests.rs`
//! scans the emitter source for layout names: no crate outside this one defines a numeric format
//! (PROXNUM-1 AC2), and the files the scalar codec card touched carry no `allow`, `panic!`,
//! `todo!`, `unimplemented!`, `unwrap` or `expect` and justify every `unsafe` block on one line
//! (PROXNUM-1 AC10). Production text is everything before the first `#[cfg(test)]` of a file whose
//! name does not end in `tests.rs`.

// the helpers outside #[test] fns state their failures through expect messages
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs;
use std::path::{Path, PathBuf};

const FOREIGN_SOURCE_DIRS: [&str; 3] = ["omega/src", "proxima-model-interop/src", "proxima-gguf/src"];

const CARD_FILES: [&str; 9] = [
    "proxima-numeric/src/scalar_codec.rs",
    "proxima-numeric/src/float_layout_codec.rs",
    "proxima-numeric/src/float_layout.rs",
    "proxima-numeric/src/format.rs",
    "proxima-numeric/src/encode.rs",
    "proxima-numeric/src/decode.rs",
    "omega/src/metal/execute_and_hazards.rs",
    "omega/src/msl/cached_attention_render.rs",
    "proxima-model-interop/src/generate/device_kv.rs",
];

const FORBIDDEN_IN_PRODUCTION: [&str; 6] = ["#[allow(", "panic!(", "todo!(", "unimplemented!(", ".unwrap()", ".expect("];

const FORMAT_TYPES: [&str; 3] = ["NumericFormat", "FloatLayout", "Codec"];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate sits one level under the workspace root")
        .to_path_buf()
}

fn rust_files_under(directory: &Path, found: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(directory).expect("a scanned source directory is readable");
    for entry in entries {
        let path = entry.expect("a directory entry is readable").path();
        if path.is_dir() {
            rust_files_under(&path, found);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            found.push(path);
        }
    }
}

fn is_test_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with("tests.rs"))
}

fn production_text(path: &Path) -> String {
    let text = fs::read_to_string(path).expect("a scanned source file is readable");
    match text.find("#[cfg(test)]") {
        Some(end) => text[..end].to_string(),
        None => text,
    }
}

fn code_lines(text: &str) -> impl Iterator<Item = (usize, &str)> {
    text.lines()
        .enumerate()
        .map(|(index, line)| (index + 1, line))
        .filter(|(_, line)| !line.trim_start().starts_with("//"))
}

fn defines_a_format(line: &str) -> bool {
    let trimmed = line.trim_start();
    let is_item = ["const ", "pub const ", "pub(crate) const ", "static ", "pub static ", "pub struct ", "struct ", "pub enum ", "enum "]
        .iter()
        .any(|prefix| trimmed.starts_with(prefix));
    let typed_constant = FORMAT_TYPES.iter().any(|name| trimmed.contains(&format!(": {name} =")));
    let named_type = FORMAT_TYPES.iter().any(|name| {
        ["struct", "enum"].iter().any(|keyword| {
            trimmed.split_once(&format!("{keyword} {name}")).is_some_and(|(_, after)| !after.starts_with(|next: char| next.is_alphanumeric() || next == '_'))
        })
    });
    let signature = trimmed.contains("->") || trimmed.starts_with("impl") || trimmed.contains("fn ");
    let literal = !signature
        && ["NumericFormat {", "FloatLayout {"].iter().any(|name| {
            trimmed.match_indices(name).any(|(at, _)| !trimmed[..at].ends_with(|before: char| before.is_alphanumeric() || before == '_'))
        });
    (is_item && (typed_constant || named_type)) || literal
}

fn foreign_production_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for directory in FOREIGN_SOURCE_DIRS {
        rust_files_under(&root.join(directory), &mut files);
    }
    files.retain(|path| !is_test_file(path));
    files.sort();
    files
}

#[test]
fn no_numeric_format_or_float_layout_is_defined_outside_proxima_numeric() {
    let root = workspace_root();
    let files = foreign_production_files(&root);
    let mut hits = Vec::new();

    for file in &files {
        let text = production_text(file);
        for (number, line) in code_lines(&text).filter(|(_, line)| defines_a_format(line)) {
            hits.push(format!("{}:{number}: {}", file.display(), line.trim()));
        }
    }

    assert!(files.len() > 100, "the scan selected {} files; it must reach the three crates", files.len());
    assert_eq!(hits, Vec::<String>::new());
}

#[test]
fn the_format_definition_detector_flags_a_constant_and_a_struct_literal() {
    assert!(defines_a_format("const MY_FORMAT: FloatLayout = FloatLayout::BF16;"));
    assert!(defines_a_format("    let layout = FloatLayout { exponent_bits: 5, ..other };"));
    assert!(defines_a_format("pub struct NumericFormat;"));
    assert!(!defines_a_format("impl NumericFormat {"));
    assert!(!defines_a_format("struct CodecShape {"));
    assert!(!defines_a_format(") -> FloatLayout {"));
    assert!(!defines_a_format("    let format = NumericFormat::Float32;"));
}

#[test]
fn the_card_files_carry_no_allow_panic_todo_unimplemented_unwrap_or_expect_in_production() {
    let root = workspace_root();
    let mut hits = Vec::new();

    for file in CARD_FILES {
        let text = production_text(&root.join(file));
        for (number, line) in code_lines(&text) {
            for marker in FORBIDDEN_IN_PRODUCTION.iter().filter(|marker| line.contains(**marker)) {
                hits.push(format!("{file}:{number}: {marker}"));
            }
        }
    }

    assert_eq!(CARD_FILES.len(), 9);
    assert_eq!(hits, Vec::<String>::new());
}

#[test]
fn every_unsafe_block_in_the_card_files_has_a_one_line_safety_comment_directly_above() {
    let root = workspace_root();
    let mut unsafe_blocks = 0;
    let mut unjustified = Vec::new();

    for file in CARD_FILES {
        let text = production_text(&root.join(file));
        let lines: Vec<&str> = text.lines().collect();
        for index in (0..lines.len()).filter(|index| lines[*index].contains("unsafe {") && !lines[*index].trim_start().starts_with("//")) {
            unsafe_blocks += 1;
            let above = index.checked_sub(1).map(|previous| lines[previous].trim_start());
            let above_that = index.checked_sub(2).map(|previous| lines[previous].trim_start());
            let one_line = above.is_some_and(|text| text.starts_with("// SAFETY:"))
                && !above_that.is_some_and(|text| text.starts_with("//"));
            if !one_line {
                unjustified.push(format!("{file}:{}", index + 1));
            }
        }
    }

    assert!(unsafe_blocks > 0, "the card files hold unsafe blocks; a scan that sees none selected nothing");
    assert_eq!(unjustified, Vec::<String>::new());
}
