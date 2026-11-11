// shared by `tests/source_rules_scan.rs` and the format codec's own scan test through `include!`,
// so both cut a source file into production and test text the same way.

// a cfg(test) helper fn or out-of-line module sits between production fns, so only a gated inline
// module may end the region; blanked lines keep the line numbers of the hits honest
fn production_of(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut kept = String::with_capacity(text.len());
    let mut index = 0;

    while index < lines.len() {
        let gate_end = attribute_end(&lines, index);
        if !requires_test(&attribute_text(&lines[index..=gate_end])) {
            kept.push_str(lines[index]);
            kept.push('\n');
            index += 1;
            continue;
        }
        let Some(item) = gated_item_line(&lines, gate_end) else {
            break;
        };
        if is_module_line(lines[item]) && !lines[item].trim_end().ends_with(';') {
            break;
        }
        let end = end_of_item(&lines, item);
        kept.push_str(&"\n".repeat(end - index));
        index = end;
    }
    kept
}

// the last line of the attribute that starts at `start`; a line that starts no attribute ends itself
fn attribute_end(lines: &[&str], start: usize) -> usize {
    if !lines[start].trim_start().starts_with("#[") {
        return start;
    }
    let mut depth = 0_i32;
    for (index, line) in lines.iter().enumerate().skip(start) {
        for character in line.chars() {
            match character {
                '[' => depth += 1,
                ']' => depth -= 1,
                _ => {}
            }
        }
        if depth <= 0 {
            return index;
        }
    }
    lines.len() - 1
}

fn attribute_text(attribute_lines: &[&str]) -> String {
    attribute_lines.iter().map(|line| line.trim()).collect::<Vec<_>>().join(" ")
}

// an item is test-only when its cfg requires `test`: `cfg(test)` or `test` a direct argument of
// `all(..)`; `any(.., test)` and `not(test)` still build in production
fn requires_test(attribute: &str) -> bool {
    attribute
        .strip_prefix("#[cfg(")
        .and_then(|rest| rest.strip_suffix(")]"))
        .is_some_and(predicate_requires_test)
}

fn predicate_requires_test(predicate: &str) -> bool {
    let predicate = predicate.trim();
    if predicate == "test" {
        return true;
    }
    predicate
        .strip_prefix("all(")
        .and_then(|rest| rest.strip_suffix(')'))
        .is_some_and(|arguments| split_arguments(arguments).iter().any(|argument| predicate_requires_test(argument)))
}

fn split_arguments(arguments: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut depth = 0_i32;
    let mut quoted = false;
    let mut start = 0;
    for (offset, character) in arguments.char_indices() {
        match character {
            '"' => quoted = !quoted,
            '(' if !quoted => depth += 1,
            ')' if !quoted => depth -= 1,
            ',' if !quoted && depth == 0 => {
                pieces.push(&arguments[start..offset]);
                start = offset + 1;
            }
            _ => {}
        }
    }
    pieces.push(&arguments[start..]);
    pieces
}

fn gated_item_line(lines: &[&str], attribute_end_line: usize) -> Option<usize> {
    let mut index = attribute_end_line + 1;
    while index < lines.len() {
        let trimmed = lines[index].trim();
        if trimmed.is_empty() {
            index += 1;
        } else if trimmed.starts_with("#[") {
            index = attribute_end(lines, index) + 1;
        } else {
            return Some(index);
        }
    }
    None
}

fn is_module_line(line: &str) -> bool {
    line.split_whitespace()
        .find(|word| *word != "pub" && !word.starts_with("pub("))
        .is_some_and(|word| word == "mod")
}

fn end_of_item(lines: &[&str], start: usize) -> usize {
    let mut depth = 0_i32;
    let mut opened = false;
    for (index, line) in lines.iter().enumerate().skip(start) {
        for character in line.chars() {
            match character {
                '{' => {
                    depth += 1;
                    opened = true;
                }
                '}' => depth -= 1,
                _ => {}
            }
        }
        if (opened && depth <= 0) || (!opened && line.trim_end().ends_with(';')) {
            return index + 1;
        }
    }
    lines.len()
}
