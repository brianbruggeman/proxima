//! Stale-citation scan over `docs/omega/*.md`.
//!
//! The omega pages cite source as backticked `path/to/file.rs:N` or `:A-B`, and a bare `:N` that
//! continues the last path cited in the same paragraph or table row. Nothing used to fail when a
//! commit shifted those lines (commit 723a16f12f4 moved 55 of them in `08-the-ir.md`). This scan
//! asserts three things from the text itself: the cited file exists and has that many lines, a
//! bare citation has a path to continue, and a backticked identifier right before a citation
//! appears within lines N-1 to N+2 of the cited file.
//!
//! A path without a `/` (`backend.rs:490`) is looked up by file name across the workspace and
//! must match exactly one file. A table row with no path of its own continues the path that was
//! last cited before the table began, because the page introduces the table's file in the
//! sentence above it.

// the helpers outside #[test] fns state their failures through expect messages
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

const CITED_EXTENSIONS: [&str; 8] = ["rs", "toml", "md", "sh", "json", "jsonl", "yaml", "yml"];

const IDENTIFIER_GAPS: [&str; 6] = [" (", " at ", ", ", " in ", " on ", " | "];

const SKIPPED_DIRECTORIES: [&str; 2] = ["target", "node_modules"];

const PREFERRED_ROOTS: [&str; 3] = ["omega/src", "proxima-tensor/src", "proxima-numeric/src"];

const MINIMUM_CITATIONS: usize = 100;

#[derive(Debug, PartialEq)]
struct Citation {
    path: Option<String>,
    first: usize,
    last: usize,
}

#[derive(Debug, Default)]
struct Report {
    checked: usize,
    failures: Vec<String>,
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate sits one level under the workspace root")
        .to_path_buf()
}

fn index_by_file_name(directory: &Path, index: &mut HashMap<String, Vec<PathBuf>>) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if path.is_dir() {
            if !name.starts_with('.') && !SKIPPED_DIRECTORIES.contains(&name.as_str()) {
                index_by_file_name(&path, index);
            }
        } else {
            index.entry(name).or_default().push(path);
        }
    }
}

fn is_cited_extension(name: &str) -> bool {
    name.rsplit_once('.')
        .is_some_and(|(_, extension)| CITED_EXTENSIONS.contains(&extension))
}

fn parse_citation(content: &str) -> Option<Citation> {
    let (left, right) = content.rsplit_once(':')?;
    let (first, last) = match right.split_once('-') {
        Some((from, to)) => (from.parse::<usize>().ok()?, to.parse::<usize>().ok()?),
        None => {
            let line = right.parse::<usize>().ok()?;
            (line, line)
        }
    };
    if left.is_empty() {
        return Some(Citation {
            path: None,
            first,
            last,
        });
    }
    let plain = !left.contains(char::is_whitespace) && !left.contains(':');
    (plain && is_cited_extension(left)).then(|| Citation {
        path: Some(left.to_string()),
        first,
        last,
    })
}

// a path with a directory and no line (`crate/src/x.rs`) is what a following bare `:N` continues
fn names_a_path(content: &str) -> bool {
    content.contains('/')
        && !content.contains(char::is_whitespace)
        && !content.contains(':')
        && is_cited_extension(content)
}

fn backtick_spans(line: &str) -> Vec<(usize, usize)> {
    let ticks: Vec<usize> = line.match_indices('`').map(|(index, _)| index).collect();
    ticks
        .iter()
        .step_by(2)
        .zip(ticks.iter().skip(1).step_by(2))
        .map(|(open, close)| (*open, *close))
        .collect()
}

fn identifier_of(content: &str) -> Option<&str> {
    let name = content.strip_suffix("()").unwrap_or(content);
    let name = name.rsplit("::").next()?;
    let mut chars = name.chars();
    let starts_well = chars
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic() || first == '_');
    (starts_well && chars.all(|next| next.is_ascii_alphanumeric() || next == '_')).then_some(name)
}

fn contains_word(text: &str, word: &str) -> bool {
    let is_word_char = |character: char| character.is_ascii_alphanumeric() || character == '_';
    // case-insensitive so a toml node kind (`reduce`) finds its variant (`Reduce`)
    let lowered = text.to_ascii_lowercase();
    lowered
        .match_indices(&word.to_ascii_lowercase())
        .any(|(start, _)| {
            let before = lowered[..start].chars().next_back();
            let after = lowered[start + word.len()..].chars().next();
            !before.is_some_and(is_word_char) && !after.is_some_and(is_word_char)
        })
}

fn identifier_before<'a>(
    line: &'a str,
    spans: &[(usize, usize)],
    position: usize,
) -> Option<&'a str> {
    let (previous_open, previous_close) = *spans.get(position.checked_sub(1)?)?;
    let (open, _) = spans[position];
    let gap = &line[previous_close + 1..open];
    let previous = &line[previous_open + 1..previous_close];
    (IDENTIFIER_GAPS.contains(&gap) && parse_citation(previous).is_none())
        .then(|| identifier_of(previous))
        .flatten()
}

fn check_lines<R>(
    resolve: &R,
    citation: &Citation,
    path: &str,
    identifier: Option<&str>,
) -> Result<(), String>
where
    R: Fn(&str) -> Result<String, String>,
{
    let content = resolve(path)?;
    let line_count = content.lines().count();
    let highest = citation.last.max(citation.first);
    if citation.first == 0 || highest > line_count {
        return Err(format!(
            "{path} has {line_count} lines, citation reaches {highest}"
        ));
    }
    let Some(name) = identifier else {
        return Ok(());
    };
    let from = citation.first.saturating_sub(2);
    let window: Vec<&str> = content
        .lines()
        .skip(from)
        .take(citation.first + 2 - from)
        .collect();
    let found = window.iter().any(|line| contains_word(line, name));
    let range = format!("{}-{}", from + 1, citation.first + 2);
    let elsewhere: Vec<String> = content
        .lines()
        .enumerate()
        .filter(|(_, line)| contains_word(line, name))
        .map(|(index, _)| (index + 1).to_string())
        .take(4)
        .collect();
    found.then_some(()).ok_or_else(|| {
        format!(
            "`{name}` is not in lines {range} of {path} (it appears at lines {})",
            elsewhere.join(", ")
        )
    })
}

// a table row or list item is its own scope that starts from the path cited above the block
fn starts_a_row(trimmed: &str) -> bool {
    let numbered = trimmed.split_once(". ").is_some_and(|(number, _)| {
        !number.is_empty() && number.chars().all(|digit| digit.is_ascii_digit())
    });
    trimmed.starts_with('|') || trimmed.starts_with("- ") || trimmed.starts_with("* ") || numbered
}

fn check_document<R>(doc: &str, text: &str, resolve: &R) -> Report
where
    R: Fn(&str) -> Result<String, String>,
{
    let mut report = Report::default();
    let mut fenced = false;
    let mut last_path: Option<String> = None;
    let mut carried: Option<String> = None;
    let mut table_path: Option<String> = None;
    let mut in_table = false;
    for (index, line) in text.lines().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            fenced = !fenced;
            continue;
        }
        if fenced {
            continue;
        }
        let continues_item = in_table && line.starts_with(char::is_whitespace);
        let is_row = !continues_item && starts_a_row(trimmed);
        if trimmed.is_empty() || trimmed.starts_with('#') {
            if !in_table {
                carried = last_path.take().or(carried);
            }
            last_path = None;
            in_table = false;
            continue;
        }
        if is_row && !in_table {
            last_path = carried.clone();
            table_path = last_path.clone();
        }
        if is_row && trimmed.starts_with('|') {
            last_path = table_path.clone();
        }
        in_table = is_row || continues_item;
        scan_line(doc, index + 1, line, resolve, &mut last_path, &mut report);
    }
    report
}

fn scan_line<R>(
    doc: &str,
    number: usize,
    line: &str,
    resolve: &R,
    last_path: &mut Option<String>,
    report: &mut Report,
) where
    R: Fn(&str) -> Result<String, String>,
{
    let spans = backtick_spans(line);
    for (position, (open, close)) in spans.iter().enumerate() {
        let content = &line[open + 1..*close];
        let Some(citation) = parse_citation(content) else {
            if names_a_path(content) {
                *last_path = Some(content.to_string());
            }
            continue;
        };
        report.checked += 1;
        if let Some(path) = &citation.path {
            *last_path = Some(path.clone());
        }
        let Some(path) = last_path.clone() else {
            report.failures.push(format!("{doc}:{number}: `{content}` is a bare citation with no earlier path in its paragraph or row"));
            continue;
        };
        let identifier = identifier_before(line, &spans, position);
        if let Err(problem) = check_lines(resolve, &citation, &path, identifier) {
            report
                .failures
                .push(format!("{doc}:{number}: `{content}` -> {problem}"));
        }
    }
}

// a file name shared by several crates resolves to the one under the first source root that has exactly one
fn preferred_match(root: &Path, candidates: &[PathBuf]) -> Option<PathBuf> {
    PREFERRED_ROOTS.iter().find_map(|preferred| {
        let top_level = candidates
            .iter()
            .filter(|candidate| candidate.parent() == Some(root.join(preferred).as_path()))
            .collect::<Vec<_>>();
        if let [only] = top_level.as_slice() {
            return Some((*only).clone());
        }
        let under: Vec<&PathBuf> = candidates
            .iter()
            .filter(|candidate| candidate.starts_with(root.join(preferred)))
            .collect();
        match under.as_slice() {
            [only] => Some((*only).clone()),
            _ => None,
        }
    })
}

fn workspace_resolver(root: &Path) -> impl Fn(&str) -> Result<String, String> {
    let mut index = HashMap::new();
    index_by_file_name(root, &mut index);
    let root = root.to_path_buf();
    move |path: &str| {
        let name = path.rsplit('/').next().unwrap_or(path);
        let full = if root.join(path).is_file() {
            root.join(path)
        } else if path.contains('/') {
            let tails: Vec<&PathBuf> = index
                .get(name)
                .map(|found| found.iter().filter(|file| file.ends_with(path)).collect())
                .unwrap_or_default();
            match tails.as_slice() {
                [only] => (*only).clone(),
                _ => return Err(format!("{path} names no file in the workspace")),
            }
        } else {
            match index.get(path).map(Vec::as_slice) {
                Some([only]) => only.clone(),
                Some(many) => match preferred_match(&root, many) {
                    Some(found) => found,
                    None => {
                        return Err(format!(
                            "{path} names {} files in the workspace; cite a path with a directory",
                            many.len()
                        ));
                    }
                },
                None => return Err(format!("{path} names no file in the workspace")),
            }
        };
        fs::read_to_string(&full)
            .map_err(|problem| format!("{path} is not a readable file: {problem}"))
    }
}

#[test]
fn every_omega_doc_citation_names_a_real_line() {
    let root = workspace_root();
    let resolve = workspace_resolver(&root);
    let mut pages: Vec<PathBuf> = fs::read_dir(root.join("docs/omega"))
        .expect("docs/omega is readable")
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "md"))
        .collect();
    pages.sort();
    let mut checked = 0;
    let mut failures = Vec::new();
    for page in &pages {
        let text = fs::read_to_string(page).expect("an omega page is readable");
        let name = page
            .file_name()
            .expect("a page has a name")
            .to_string_lossy()
            .into_owned();
        let report = check_document(&format!("docs/omega/{name}"), &text, &resolve);
        checked += report.checked;
        failures.extend(report.failures);
    }
    println!(
        "doc_citations: checked {checked} citations across {} pages",
        pages.len()
    );
    assert!(
        failures.is_empty(),
        "{} stale citations:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(
        checked > MINIMUM_CITATIONS,
        "only {checked} citations found; the scan lost its subject"
    );
}

fn fixture_resolver(path: &str) -> Result<String, String> {
    match path {
        "crate/src/shape.rs" => Ok("// header\n\npub struct Shape {\n    rank: usize,\n}\n\nimpl Shape {\n    pub fn rank(&self) -> usize {\n        self.rank\n    }\n}\n".to_string()),
        other => Err(format!("{other} names no file in the workspace")),
    }
}

fn failures_of(text: &str) -> Vec<String> {
    check_document("fixture.md", text, &fixture_resolver).failures
}

#[test]
fn citation_at_the_right_line_passes() {
    let report = check_document(
        "fixture.md",
        "`Shape` (`crate/src/shape.rs:3`) owns the rank.",
        &fixture_resolver,
    );
    assert_eq!((report.checked, report.failures), (1, Vec::new()));
}

#[test]
fn citation_with_a_wrong_line_number_fails() {
    let failures = failures_of("`Shape` (`crate/src/shape.rs:9`) owns the rank.");
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
        failures[0].contains("`Shape` is not in lines 8-11"),
        "{failures:?}"
    );
}

#[test]
fn citation_past_the_end_of_the_file_fails() {
    let failures = failures_of("see `crate/src/shape.rs:40-44` for the rest.");
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
        failures[0].contains("has 11 lines, citation reaches 44"),
        "{failures:?}"
    );
}

#[test]
fn citation_of_a_missing_file_fails() {
    let failures = failures_of("see `crate/src/gone.rs:3`.");
    assert!(failures[0].contains("names no file"), "{failures:?}");
}

#[test]
fn bare_citation_continues_the_path_cited_earlier_in_the_paragraph() {
    let text = "`Shape` (`crate/src/shape.rs:3`), whose `rank` method (`:8`) returns it.";
    let report = check_document("fixture.md", text, &fixture_resolver);
    assert_eq!((report.checked, report.failures), (2, Vec::new()));
}

#[test]
fn bare_citation_with_a_wrong_line_number_fails() {
    let failures = failures_of("`Shape` (`crate/src/shape.rs:3`), whose `rank` (`:1`) returns it.");
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
        failures[0].contains("`rank` is not in lines"),
        "{failures:?}"
    );
}

#[test]
fn bare_citation_without_an_earlier_path_fails() {
    let failures = failures_of("`Shape` (`:3`) owns the rank.");
    assert!(
        failures[0].contains("bare citation with no earlier path"),
        "{failures:?}"
    );
}

#[test]
fn bare_citation_does_not_cross_a_paragraph_break() {
    let failures =
        failures_of("`Shape` (`crate/src/shape.rs:3`).\n\nA second paragraph cites `:3`.");
    assert!(
        failures[0].contains("bare citation with no earlier path"),
        "{failures:?}"
    );
}

#[test]
fn table_rows_continue_the_path_cited_above_the_table() {
    let text = "The shape lives in `crate/src/shape.rs:3`.\n\n| type | where |\n|---|---|\n| `Shape` | `:3` |\n| `rank` | `:8` |\n";
    let report = check_document("fixture.md", text, &fixture_resolver);
    assert_eq!((report.checked, report.failures), (3, Vec::new()));
}

#[test]
fn citations_inside_code_fences_are_not_scanned() {
    let report = check_document(
        "fixture.md",
        "```\n`crate/src/gone.rs:3`\n```\n",
        &fixture_resolver,
    );
    assert_eq!((report.checked, report.failures), (0, Vec::new()));
}
