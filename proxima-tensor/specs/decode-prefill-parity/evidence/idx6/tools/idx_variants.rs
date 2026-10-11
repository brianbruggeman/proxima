use std::fs;
use std::path::{Path, PathBuf};

const GATHER_DIVIDE: &str = "long expert_slot = group_first / group_out_extent;";
const GATHER_REMAINDER: &str = "long out_row_base = group_first % group_out_extent;";

fn is_word_char(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn replace_word(text: &str, word: &str, replacement: &str) -> String {
    let bytes = text.as_bytes();
    let mut output = String::with_capacity(text.len());
    let mut cursor = 0;
    while let Some(found) = text[cursor..].find(word) {
        let start = cursor + found;
        let end = start + word.len();
        let before_ok = start == 0 || !is_word_char(bytes[start - 1]);
        let after_ok = end >= bytes.len() || !is_word_char(bytes[end]);
        output.push_str(&text[cursor..start]);
        output.push_str(if before_ok && after_ok { replacement } else { word });
        cursor = end;
    }
    output.push_str(&text[cursor..]);
    output
}

fn strip_literal_suffix(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut output = String::with_capacity(text.len());
    for (index, &byte) in bytes.iter().enumerate() {
        let literal_suffix = byte == b'L'
            && index > 0
            && bytes[index - 1].is_ascii_digit()
            && (index + 1 >= bytes.len() || !is_word_char(bytes[index + 1]))
            && !preceded_by_identifier_digits(bytes, index);
        if !literal_suffix {
            output.push(byte as char);
        }
    }
    output
}

fn preceded_by_identifier_digits(bytes: &[u8], suffix_index: usize) -> bool {
    let mut cursor = suffix_index;
    while cursor > 0 && bytes[cursor - 1].is_ascii_digit() {
        cursor -= 1;
    }
    cursor > 0 && (bytes[cursor - 1].is_ascii_alphabetic() || bytes[cursor - 1] == b'_')
}

fn uniform_long_members(source: &str) -> Vec<String> {
    let start = source.find("struct Uniforms {").expect("uniforms struct");
    let block = &source[start..];
    let end = block.find("};").expect("uniforms end");
    block[..end]
        .lines()
        .filter_map(|line| line.trim().strip_prefix("long "))
        .map(|rest| rest.split(['[', ';']).next().unwrap().trim().to_string())
        .collect()
}

fn cast_uniform_reads(body: &str, members: &[String]) -> String {
    let bytes = body.as_bytes();
    let mut output = String::with_capacity(body.len() + 256);
    let mut cursor = 0;
    while cursor < bytes.len() {
        let member = (bytes[cursor] == b'u' && cursor + 2 <= bytes.len() && bytes.get(cursor + 1) == Some(&b'.'))
            .then(|| members.iter().find(|name| uniform_member_at(body, cursor + 2, name)))
            .flatten();
        let boundary_ok = cursor == 0 || !is_word_char(bytes[cursor - 1]);
        match member {
            Some(name) if boundary_ok => {
                let mut end = cursor + 2 + name.len();
                while bytes.get(end) == Some(&b'[') {
                    end = matching_bracket(bytes, end) + 1;
                }
                output.push_str(&format!("((int){})", &body[cursor..end]));
                cursor = end;
            }
            _ => {
                output.push(bytes[cursor] as char);
                cursor += 1;
            }
        }
    }
    output
}

fn uniform_member_at(body: &str, at: usize, name: &str) -> bool {
    let tail = &body[at..];
    tail.starts_with(name) && tail.as_bytes().get(name.len()).is_none_or(|byte| !is_word_char(*byte))
}

fn matching_bracket(bytes: &[u8], open: usize) -> usize {
    let mut depth = 0;
    for (offset, &byte) in bytes[open..].iter().enumerate() {
        match byte {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return open + offset;
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced bracket");
}

fn int32_body(source: &str) -> String {
    let members = uniform_long_members(source);
    let at = source.find("kernel void ").expect("kernel entry");
    let (head, body) = source.split_at(at);
    let body = cast_uniform_reads(body, &members);
    let body = replace_word(&body, "long", "int");
    let body = strip_literal_suffix(&body);
    format!("{head}{body}")
}

fn gathered_prologue32(source: &str) -> Option<String> {
    if !source.contains(GATHER_DIVIDE) || !source.contains(GATHER_REMAINDER) {
        return None;
    }
    let narrowed_divide = "long expert_slot = (long)((uint)group_first / (uint)group_out_extent);";
    let narrowed_remainder = "long out_row_base = (long)((uint)group_first % (uint)group_out_extent);";
    Some(source.replace(GATHER_DIVIDE, narrowed_divide).replace(GATHER_REMAINDER, narrowed_remainder))
}

fn step_counters32(source: &str) -> Option<String> {
    let mut changed = source.to_string();
    let mut touched = false;
    for name in ["blk_step", "y4_step", "acts_step", "y_step"] {
        let marker = format!("    long {name} = (long)ib_step * ");
        if let Some(at) = changed.find(&marker) {
            let end = at + changed[at..].find(";\n").expect("statement end");
            let statement = changed[at..end].replace("long ", "int ").replace("(long)ib_step", "ib_step");
            changed.replace_range(at..end, &statement);
            touched = true;
        }
    }
    touched.then_some(changed)
}

fn block_decode32(source: &str) -> Option<String> {
    let divide = source.find("(off0 / ")?;
    let divisor_end = divide + 8 + source[divide + 8..].find(')')?;
    let divisor = &source[divide + 8..divisor_end];
    let remainder_marker = format!("(uint)(off0 % {divisor})");
    if !source.contains(&remainder_marker) {
        return None;
    }
    let wide_divide = format!("(off0 / {divisor}) * ");
    let narrow_divide = format!("(long)((uint)off0 / {divisor}u) * ");
    let narrow_remainder = format!("((uint)off0 % {divisor}u)");
    Some(source.replace(&wide_divide, &narrow_divide).replace(&remainder_marker, &narrow_remainder))
}

fn carries_epilogue_or_gather(source: &str) -> bool {
    source.contains("epilogue_operand_base") || source.contains("gather_index_base")
}

fn packed_pointers(source: &str) -> bool {
    source.contains("device const uchar* in") || source.contains("device const uchar *in")
}

fn write_variant(directory: &Path, sha: &str, tag: &str, text: &str) {
    fs::write(directory.join(format!("{sha}.{tag}.metal")), text).expect("write variant");
}

fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    let [_, describe, caps, directory] = &arguments[..] else {
        panic!("usage: idx_variants <describe.out> <caps dir> <variant dir>");
    };
    let directory = PathBuf::from(directory);
    fs::create_dir_all(&directory).expect("variant dir");
    let listing = fs::read_to_string(describe).expect("describe");
    let mut seen: Vec<String> = Vec::new();
    let mut counts = [0usize; 5];
    for line in listing.lines().filter_map(|line| line.strip_prefix("ab describe sha=")) {
        let sha = line[..16].to_string();
        if seen.contains(&sha) {
            continue;
        }
        seen.push(sha.clone());
        let source = fs::read_to_string(Path::new(caps).join(format!("pipeline_{sha}.metal"))).expect("source");
        if !packed_pointers(&source) {
            write_variant(&directory, &sha, "i32", &int32_body(&source));
            counts[0] += 1;
            if carries_epilogue_or_gather(&source) {
                write_variant(&directory, &sha, "w32", &int32_body(&source));
                counts[3] += 1;
            }
        }
        if let Some(text) = block_decode32(&source) {
            write_variant(&directory, &sha, "gq32", &text);
            counts[4] += 1;
        }
        if let Some(text) = gathered_prologue32(&source) {
            write_variant(&directory, &sha, "gp32", &text);
            counts[1] += 1;
        }
        if let Some(text) = step_counters32(&source) {
            write_variant(&directory, &sha, "st32", &text);
            counts[2] += 1;
        }
    }
    eprintln!("variants written: i32={} gp32={} st32={} w32={} gq32={} over {} kernels", counts[0], counts[1], counts[2], counts[3], counts[4], seen.len());
}
