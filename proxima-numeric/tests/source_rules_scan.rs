//! Source scans over production text across the crates that consume `proxima-numeric`.
//!
//! Two rules are asserted from the text itself, the way `omega/src/msl/layout_render_tests.rs`
//! scans the emitter source for layout names: no crate outside this one defines a numeric format
//! (PROXNUM-1 AC2), and the files the scalar codec card touched carry no `allow`, `panic!`,
//! `todo!`, `unimplemented!`, `unwrap` or `expect` and justify every `unsafe` block on one line
//! (PROXNUM-1 AC10). Production text is everything before the `#[cfg(test)]` that gates a module, of
//! a file whose name does not end in `tests.rs`; a `#[cfg(test)]` fn among production code is
//! skipped alone.

// the helpers outside #[test] fns state their failures through expect messages
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs;
use std::path::{Path, PathBuf};

const FOREIGN_SOURCE_DIRS: [&str; 3] = ["omega/src", "proxima-model-interop/src", "proxima-gguf/src"];

const CARD_FILES: [&str; 9] = [
    "proxima-numeric/src/format_codec.rs",
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

include!("support/production_text.rs");

fn production_text(path: &Path) -> String {
    production_of(&fs::read_to_string(path).expect("a scanned source file is readable"))
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
            trimmed
                .match_indices(name)
                .any(|(offset, _)| !trimmed[..offset].ends_with(|before: char| before.is_alphanumeric() || before == '_'))
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

#[test]
fn a_cfg_test_fn_among_production_code_does_not_end_the_production_region() {
    let source = "\
fn first() {}

#[cfg(test)]
#[inline]
pub(super) fn only_for_tests(value: u32) -> u32 {
    value.checked_add(1).unwrap()
}

fn second() -> u32 {
    Some(2).unwrap()
}

#[cfg(test)]
mod tests {
    fn third() {}
}
";

    let production = production_of(source);

    assert!(production.contains("fn second()"), "code after a gated fn is still production");
    assert!(production.contains("Some(2).unwrap()"));
    assert!(!production.contains("only_for_tests"), "the gated fn is skipped alone");
    assert!(!production.contains("fn third()"), "the gated module ends the region");
    assert_eq!(production.lines().nth(8), Some("fn second() -> u32 {"), "skipped lines keep their line numbers");
}

#[test]
fn a_cfg_all_test_module_ends_the_production_region_like_a_cfg_test_module() {
    let source = "fn kept() {}\n\n#[cfg(all(test, feature = \"metal\"))]\n#[allow(clippy::unwrap_used)]\nmod placed_tests {\n    fn dropped() {}\n}\n";

    let production = production_of(source);

    assert!(production.contains("fn kept()"));
    assert!(!production.contains("dropped"));
}

#[test]
fn the_omega_cached_attention_renderer_is_scanned_past_its_cfg_test_helper() {
    let root = workspace_root();
    let text = production_text(&root.join("omega/src/msl/cached_attention_render.rs"));

    assert!(text.contains("fn cached_attention_kv_codec_for"), "the scan must reach code after the gated helper");
}

fn workspace_members(root: &Path) -> Vec<String> {
    let manifest = fs::read_to_string(root.join("Cargo.toml")).expect("the workspace manifest is readable");
    let list = manifest
        .split_once("members = [")
        .and_then(|(_, after)| after.split_once(']'))
        .map(|(list, _)| list)
        .expect("the workspace manifest lists its members");
    list.split('"').skip(1).step_by(2).map(str::to_string).collect()
}

fn defines_a_format_constant(line: &str) -> bool {
    let trimmed = line.trim_start();
    let is_item = ["const ", "pub const ", "pub(crate) const ", "pub(super) const ", "static ", "pub static "]
        .iter()
        .any(|prefix| trimmed.starts_with(prefix));
    let declared_type = trimmed.split_once(':').and_then(|(_, after)| after.split_once('=')).map(|(declared, _)| declared);
    is_item
        && declared_type.is_some_and(|declared| {
            let bare = declared.trim().trim_start_matches('&').trim_start_matches("'static ");
            bare == "NumericFormat" || bare == "Codec"
        })
}

#[test]
fn the_format_constant_detector_flags_every_spelling_of_a_typed_definition() {
    assert!(defines_a_format_constant("pub const KV: NumericFormat = NumericFormat::Q4_0;"));
    assert!(defines_a_format_constant("    const CACHE: Codec = Codec::Q8_0;"));
    assert!(defines_a_format_constant("const BORROWED: &NumericFormat = &NumericFormat::Q4_0;"));
    assert!(defines_a_format_constant("const SHARED: &'static Codec = &Codec::Q8_0;"));
    assert!(!defines_a_format_constant("static FORMATS: [NumericFormat; 2] = [a, b];"), "a table of formats defines none");
    assert!(!defines_a_format_constant("const COUNT: usize = 2;"));
    assert!(!defines_a_format_constant("let format: NumericFormat = NumericFormat::Q4_0;"));
    assert!(!defines_a_format_constant("const CODEC_NAME: &str = \"codec\";"));
}

#[test]
fn no_numeric_format_or_codec_constant_is_defined_in_any_workspace_member_outside_proxima_numeric() {
    let root = workspace_root();
    let members = workspace_members(&root);
    let mut files = Vec::new();
    for member in members.iter().filter(|member| member.as_str() != "proxima-numeric") {
        let source_directory = root.join(member).join("src");
        if source_directory.is_dir() {
            rust_files_under(&source_directory, &mut files);
        }
    }
    let mut hits = Vec::new();

    for file in &files {
        let text = fs::read_to_string(file).expect("a scanned source file is readable");
        for (number, line) in code_lines(&text).filter(|(_, line)| defines_a_format_constant(line)) {
            hits.push(format!("{}:{number}: {}", file.display(), line.trim()));
        }
    }

    assert!(members.len() > 40, "the manifest parse found {} members", members.len());
    assert!(files.len() > 1000, "the scan selected {} files; it must reach every member's src", files.len());
    assert_eq!(hits, Vec::<String>::new());
}

fn body_of_fn(text: &str, signature: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let start = lines
        .iter()
        .position(|line| line.contains(signature))
        .unwrap_or_else(|| panic!("{signature} is defined in the scanned text"));
    lines[start..end_of_item(&lines, start)].join("\n")
}

fn maps_a_ggml_type_to_something(line: &str) -> bool {
    let trimmed = line.trim_start();
    !trimmed.starts_with("//")
        && trimmed.contains("GgmlType::")
        && (trimmed.contains("=>") || trimmed.starts_with("GgmlType::") || trimmed.starts_with("| GgmlType::"))
}

#[test]
fn the_ggml_type_to_format_mapping_lives_only_in_gguf_numeric_format() {
    let root = workspace_root();
    let bind = production_text(&root.join("proxima-model-interop/src/bind.rs"));
    let example = production_text(&root.join("proxima-model-interop/examples/plan_rows_diff.rs"));
    let derivations = [
        body_of_fn(&bind, "fn codec_from_ggml_type("),
        body_of_fn(&bind, "fn packed_block_codec("),
        body_of_fn(&example, "fn codec_of("),
    ];
    let entry = body_of_fn(&bind, "fn gguf_tensor_as_packed_block<");

    for body in &derivations {
        assert!(body.contains(".numeric_format()"), "a derivation site asks GgmlType::numeric_format:\n{body}");
        let arms: Vec<&str> = body.lines().filter(|line| maps_a_ggml_type_to_something(line)).collect();
        assert_eq!(arms, Vec::<&str>::new(), "a derivation site holds no GgmlType arm of its own");
    }
    let entry_arms: Vec<&str> = entry
        .lines()
        .filter(|line| maps_a_ggml_type_to_something(line) && !line.contains("GgmlType::F32 =>"))
        .collect();
    assert!(entry.contains("packed_block_codec("), "the packed-block entry derives its codec through packed_block_codec");
    assert_eq!(entry_arms, Vec::<&str>::new(), "only the aligned f32 view is matched by type");
}

#[test]
fn the_gguf_mapping_scan_flags_a_hand_written_match() {
    let planted = "fn codec_from_ggml_type(ggml_type: GgmlType) -> Option<Codec> {\n    match ggml_type {\n        GgmlType::Q4_0 => Some(Codec::Q4_0),\n        _ => None,\n    }\n}\n";

    let body = body_of_fn(planted, "fn codec_from_ggml_type(");

    assert!(!body.contains(".numeric_format()"));
    assert!(body.lines().any(maps_a_ggml_type_to_something));
}
