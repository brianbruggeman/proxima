//! Emits `src/unicode_tables.rs` from llama.cpp's `src/unicode-data.cpp`
//! (`unicode_ranges_flags` and `unicode_set_whitespace`), the tables its
//! pre-tokenizers read for `\p{N}`, `\p{L}`, `\p{M}`, `\p{P}` and `\s`.
//!
//! ```text
//! cargo run -p proxima-tokenizer --example gen_unicode_tables -- \
//!     <llama.cpp>/src/unicode-data.cpp <llama.cpp commit> proxima-tokenizer/src/unicode_tables.rs
//! ```

use std::fmt::Write as _;
use std::fs;

const FLAG_NUMBER: u16 = 0x0002;
const FLAG_LETTER: u16 = 0x0004;
const FLAG_MARK: u16 = 0x0010;
const FLAG_PUNCT: u16 = 0x0020;
const RANGES_HEADER: &str = "unicode_ranges_flags";
const WHITESPACE_HEADER: &str = "unicode_set_whitespace";

type ClassRange = (u32, u8);

enum Section {
    None,
    Ranges,
    Whitespace,
}

fn parse_hex(text: &str) -> Result<u32, String> {
    u32::from_str_radix(text.trim().trim_start_matches("0x"), 16)
        .map_err(|error| format!("bad hex {text:?}: {error}"))
}

fn parse_range_line(line: &str) -> Result<(u32, u16), String> {
    let inner = line.trim().trim_start_matches('{').trim_end_matches("},");
    let (start, flags) = inner
        .split_once(',')
        .ok_or_else(|| format!("range line without a comma: {line:?}"))?;
    let flags = u16::try_from(parse_hex(flags)?).map_err(|error| format!("flags {line:?}: {error}"))?;
    Ok((parse_hex(start)?, flags))
}

fn class_bits(flags: u16) -> u8 {
    let mut bits = 0u8;
    for (flag, bit) in [
        (FLAG_NUMBER, 1u8),
        (FLAG_LETTER, 2),
        (FLAG_MARK, 4),
        (FLAG_PUNCT, 8),
    ] {
        if flags & flag != 0 {
            bits |= bit;
        }
    }
    bits
}

fn parse(source: &str) -> Result<(Vec<ClassRange>, Vec<u32>), String> {
    let mut ranges: Vec<(u32, u8)> = Vec::new();
    let mut whitespace: Vec<u32> = Vec::new();
    let mut section = Section::None;
    for line in source.lines() {
        if line.contains(RANGES_HEADER) && line.starts_with("const") {
            section = Section::Ranges;
        } else if line.contains(WHITESPACE_HEADER) && line.starts_with("const") {
            section = Section::Whitespace;
        } else if line.starts_with("};") {
            section = Section::None;
        } else {
            match section {
                Section::Ranges => {
                    let (start, flags) = parse_range_line(line)?;
                    let bits = class_bits(flags);
                    if ranges.last().map(|last| last.1) != Some(bits) {
                        ranges.push((start, bits));
                    }
                }
                Section::Whitespace => whitespace.push(parse_hex(line.trim().trim_end_matches(','))?),
                Section::None => {}
            }
        }
    }
    whitespace.sort_unstable();
    Ok((ranges, whitespace))
}

fn render(commit: &str, ranges: &[(u32, u8)], whitespace: &[u32]) -> Result<String, std::fmt::Error> {
    let mut out = String::new();
    writeln!(out, "//! Generated from llama.cpp {commit} `src/unicode-data.cpp` by")?;
    writeln!(out, "//! `cargo run -p proxima-tokenizer --example gen_unicode_tables`: the exact")?;
    writeln!(out, "//! `\\p{{N}}` / `\\p{{L}}` / `\\p{{M}}` / `\\p{{P}}` / `\\s` tables llama.cpp's")?;
    writeln!(out, "//! pre-tokenizers read.\n")?;
    writeln!(out, "pub(super) const NUMBER: u8 = 1;")?;
    writeln!(out, "pub(super) const LETTER: u8 = 2;")?;
    writeln!(out, "pub(super) const MARK: u8 = 4;")?;
    writeln!(out, "pub(super) const PUNCT: u8 = 8;\n")?;
    writeln!(out, "pub(super) static CLASS_RANGES: [(u32, u8); {}] = [", ranges.len())?;
    for chunk in ranges.chunks(6) {
        let items: Vec<String> = chunk
            .iter()
            .map(|(start, bits)| format!("(0x{start:06X}, {bits})"))
            .collect();
        writeln!(out, "    {},", items.join(", "))?;
    }
    writeln!(out, "];\n")?;
    writeln!(out, "pub(super) static WHITESPACE: [u32; {}] = [", whitespace.len())?;
    for chunk in whitespace.chunks(8) {
        let items: Vec<String> = chunk.iter().map(|codepoint| format!("0x{codepoint:06X}")).collect();
        writeln!(out, "    {},", items.join(", "))?;
    }
    writeln!(out, "];")?;
    Ok(out)
}

fn main() -> Result<(), String> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let [source_path, commit, output_path] = arguments.as_slice() else {
        return Err("usage: gen_unicode_tables <unicode-data.cpp> <llama.cpp commit> <output.rs>".to_owned());
    };
    let source = fs::read_to_string(source_path).map_err(|error| format!("read {source_path}: {error}"))?;
    let (ranges, whitespace) = parse(&source)?;
    if ranges.is_empty() || whitespace.is_empty() {
        return Err(format!(
            "parsed {} ranges and {} whitespace codepoints from {source_path}",
            ranges.len(),
            whitespace.len()
        ));
    }
    let rendered = render(commit, &ranges, &whitespace).map_err(|error| error.to_string())?;
    fs::write(output_path, rendered).map_err(|error| format!("write {output_path}: {error}"))?;
    println!("wrote {output_path}: {} ranges, {} whitespace codepoints", ranges.len(), whitespace.len());
    Ok(())
}
