// shared by `tests/source_rules_scan.rs` and the format codec's own scan test through `include!`,
// so both cut a source file into production and test text the same way.

// a cfg(test) helper fn sits between production fns, so only a gated module may end the region;
// blanked lines keep the line numbers of the hits honest
fn production_of(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut kept = String::with_capacity(text.len());
    let mut index = 0;

    while index < lines.len() {
        if !is_test_gate(lines[index]) {
            kept.push_str(lines[index]);
            kept.push('\n');
            index += 1;
            continue;
        }
        let Some(item) = gated_item_line(&lines, index) else {
            break;
        };
        if is_module_line(lines[item]) {
            break;
        }
        let end = end_of_item(&lines, item);
        kept.push_str(&"\n".repeat(end - index));
        index = end;
    }
    kept
}

// `cfg(all(test, feature = ..))` gates a test module just as `cfg(test)` does
fn is_test_gate(line: &str) -> bool {
    let trimmed = line.trim();
    let words: Vec<&str> = trimmed
        .split(|character: char| !character.is_alphanumeric() && character != '_')
        .collect();
    trimmed.starts_with("#[cfg(") && words.contains(&"test") && !words.contains(&"not")
}

fn gated_item_line(lines: &[&str], attribute: usize) -> Option<usize> {
    lines
        .iter()
        .enumerate()
        .skip(attribute + 1)
        .find(|(_, line)| {
            let trimmed = line.trim();
            !trimmed.is_empty() && !trimmed.starts_with("#[")
        })
        .map(|(index, _)| index)
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
