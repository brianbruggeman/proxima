//! The mechanical form of "the family is data": its name may appear in the
//! profile file it labels, in test code, and nowhere else the model graph is
//! built. This reads source text, drops every item a release build never
//! compiles, and reports what is left.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

/// The families this crate carries a real header capture for: a directory under
/// `tests/fixtures/` that holds `gguf_kv_loader.txt`. The rule below binds on
/// the captured families, so a new capture joins the scan with no edit here and
/// a family with no capture (whose doc prose names it, as the registry's other
/// rows do) is not held to it.
fn captured_families() -> Vec<String> {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut names: Vec<String> = fs::read_dir(&directory)
        .unwrap_or_else(|error| panic!("{}: {error}", directory.display()))
        .filter_map(|entry| {
            let path = entry
                .unwrap_or_else(|error| panic!("{}: {error}", directory.display()))
                .path();
            let carries_a_header = path.join("gguf_kv_loader.txt").is_file();
            carries_a_header.then(|| {
                path.file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default()
            })
        })
        .collect();
    names.sort();
    names
}

/// One production source line that names the family.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Mention {
    pub(super) path: PathBuf,
    pub(super) line: usize,
    pub(super) text: String,
}

impl fmt::Display for Mention {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}:{}: {}",
            self.path.display(),
            self.line,
            self.text
        )
    }
}

/// Every production mention of `literal` under `roots`, in path order.
///
/// Three things are test code, not production: a path under `data_home` (the
/// family profiles directory is where the name is data), a path named
/// `tests.rs`/`*_tests.rs` or living in a `tests` directory, and a file whose
/// own inner `#![cfg(...)]` cannot hold without `test`. Inside a file, every
/// item a release build cannot compile is dropped, and an out-of-line module
/// declared under one drops its file too -- otherwise `#[cfg(test)] mod
/// fixtures;` would hide test-only bytes from the scan that the compiler never
/// sees either.
pub(super) fn mentions(roots: &[PathBuf], data_home: &Path, literal: &str) -> Vec<Mention> {
    let mut files = Vec::new();
    for root in roots {
        collect_rust_files(root, &mut files);
    }
    files.sort();
    let mut excluded: Vec<PathBuf> = files
        .iter()
        .filter(|path| is_test_path(path) || path.starts_with(data_home))
        .cloned()
        .collect();
    let mut splits = Vec::with_capacity(files.len());
    for path in files {
        let source =
            fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        if is_test_only_file(&source) {
            excluded.push(path);
            continue;
        }
        let split = split_test_gated(&source);
        excluded.extend(gated_module_files(&path, &split.gated));
        splits.push((path, split.kept));
    }
    let needle = literal.to_lowercase();
    let mut found = Vec::new();
    for (path, kept) in splits {
        if excluded.contains(&path) {
            continue;
        }
        for (index, line) in kept.lines().enumerate() {
            if line.to_lowercase().contains(&needle) {
                found.push(Mention {
                    path: path.clone(),
                    line: index + 1,
                    text: line.trim().to_string(),
                });
            }
        }
    }
    found
}

fn is_test_path(path: &Path) -> bool {
    let named_after_tests = path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == "tests.rs" || name.ends_with("_tests.rs"));
    named_after_tests
        || path
            .components()
            .any(|component| component.as_os_str() == "tests")
}

/// A module file can gate itself whole: `#![cfg(all(test, feature = "std"))]`
/// at the top of a file compiles nothing in a release build, so its bytes are
/// test bytes even though no declaring item says so.
fn is_test_only_file(source: &str) -> bool {
    let bytes = source.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        index = match skip_whitespace_and_comments(bytes, index) {
            Some(next) => next,
            None => return false,
        };
        if bytes.get(index) != Some(&b'#') || bytes.get(index + 1) != Some(&b'!') {
            return false;
        }
        let Some(end) = attribute_end(bytes, index + 1) else {
            return false;
        };
        if is_test_only_cfg(&source[index..=end]) {
            return true;
        }
        index = end + 1;
    }
    false
}

fn collect_rust_files(directory: &Path, files: &mut Vec<PathBuf>) {
    let entries =
        fs::read_dir(directory).unwrap_or_else(|error| panic!("{}: {error}", directory.display()));
    for entry in entries {
        let path = entry
            .unwrap_or_else(|error| panic!("{}: {error}", directory.display()))
            .path();
        if path.is_dir() {
            collect_rust_files(&path, files);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
}

/// A source file as a release build sees it: `kept` is the text with every
/// test-gated item blanked out to preserve line numbers, `gated` holds those
/// items' own text so their out-of-line modules can be dropped too.
struct Split {
    kept: String,
    gated: Vec<String>,
}

fn split_test_gated(source: &str) -> Split {
    let bytes = source.as_bytes();
    let mut kept = String::with_capacity(source.len());
    let mut gated = Vec::new();
    let mut copied = 0;
    let mut index = 0;
    while index < bytes.len() {
        if let Some(end) = skip_literal(bytes, index) {
            index = end;
            continue;
        }
        if bytes[index] != b'#' || bytes.get(index + 1) != Some(&b'[') {
            index += 1;
            continue;
        }
        let Some(attribute) = attribute_end(bytes, index) else {
            index += 1;
            continue;
        };
        if !is_test_only_cfg(&source[index..=attribute]) {
            index = attribute + 1;
            continue;
        }
        let Some(item_end) = gated_item_end(bytes, attribute + 1) else {
            index = attribute + 1;
            continue;
        };
        kept.push_str(&source[copied..index]);
        kept.extend(
            source[index..=item_end]
                .chars()
                .filter(|character| *character == '\n'),
        );
        gated.push(source[index..=item_end].to_string());
        copied = item_end + 1;
        index = item_end + 1;
    }
    kept.push_str(&source[copied..]);
    Split { kept, gated }
}

/// The end of the item a test-gating attribute applies to: every attribute
/// stacked on it, then the item's own body, which ends at a top-level `;` or a
/// matching `}`.
fn gated_item_end(bytes: &[u8], mut index: usize) -> Option<usize> {
    loop {
        index = skip_whitespace_and_comments(bytes, index)?;
        if bytes.get(index) == Some(&b'#') && bytes.get(index + 1) == Some(&b'[') {
            index = attribute_end(bytes, index)? + 1;
            continue;
        }
        break;
    }
    let mut depth = 0usize;
    let mut opened = false;
    while index < bytes.len() {
        if let Some(end) = skip_literal(bytes, index) {
            index = end;
            continue;
        }
        match bytes[index] {
            b'{' | b'(' | b'[' => {
                depth += 1;
                opened = true;
            }
            b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 && opened {
                    return Some(index);
                }
            }
            b')' | b']' => depth = depth.saturating_sub(1),
            b';' if depth == 0 => return Some(index),
            _ => {}
        }
        index += 1;
    }
    None
}

/// The byte after the `]` closing the attribute that opens at `start`.
fn attribute_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut index = start + 1;
    while index < bytes.len() {
        if let Some(end) = skip_literal(bytes, index) {
            index = end;
            continue;
        }
        match bytes[index] {
            b'[' | b'(' => depth += 1,
            b']' | b')' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
        index += 1;
    }
    None
}

/// True when `cfg` cannot hold without `test`, so a release build drops the
/// item. `cfg(any(test, ...))` is NOT test-only: the other arm compiles it.
fn is_test_only_cfg(attribute: &str) -> bool {
    let inner = attribute
        .trim_start_matches('#')
        .trim_start_matches('!')
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim();
    let Some(predicate) = inner.strip_prefix("cfg") else {
        return false;
    };
    let predicate = predicate.trim();
    let Some(predicate) = predicate.strip_prefix('(') else {
        return false;
    };
    let predicate = predicate.strip_suffix(')').unwrap_or(predicate);
    !Predicate::new(predicate).evaluate(false).can_be_true
}

/// What a `cfg` predicate can evaluate to when `test` is fixed and every other
/// atom is free to be either value -- the shortest statement of "does a release
/// build compile this".
#[derive(Clone, Copy)]
struct Possible {
    can_be_false: bool,
    can_be_true: bool,
}

impl Possible {
    const FREE: Self = Self {
        can_be_false: true,
        can_be_true: true,
    };
    const TRUE: Self = Self {
        can_be_false: false,
        can_be_true: true,
    };
    const FALSE: Self = Self {
        can_be_false: true,
        can_be_true: false,
    };

    fn and(self, other: Self) -> Self {
        Self {
            can_be_false: self.can_be_false || other.can_be_false,
            can_be_true: self.can_be_true && other.can_be_true,
        }
    }

    fn or(self, other: Self) -> Self {
        Self {
            can_be_false: self.can_be_false && other.can_be_false,
            can_be_true: self.can_be_true || other.can_be_true,
        }
    }

    fn negate(self) -> Self {
        Self {
            can_be_false: self.can_be_true,
            can_be_true: self.can_be_false,
        }
    }
}

struct Predicate<'a> {
    text: &'a [u8],
    index: usize,
}

impl<'a> Predicate<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            text: text.as_bytes(),
            index: 0,
        }
    }

    fn evaluate(&mut self, test: bool) -> Possible {
        self.space();
        let word = self.word();
        match word.as_str() {
            "all" if self.peek() == Some(b'(') => {
                self.index += 1;
                let mut value = Possible::TRUE;
                loop {
                    value = value.and(self.evaluate(test));
                    self.space();
                    match self.peek() {
                        Some(b',') => self.index += 1,
                        Some(b')') => {
                            self.index += 1;
                            break;
                        }
                        _ => break,
                    }
                }
                value
            }
            "any" if self.peek() == Some(b'(') => {
                self.index += 1;
                let mut value = Possible::FALSE;
                loop {
                    value = value.or(self.evaluate(test));
                    self.space();
                    match self.peek() {
                        Some(b',') => self.index += 1,
                        Some(b')') => {
                            self.index += 1;
                            break;
                        }
                        _ => break,
                    }
                }
                value
            }
            "not" if self.peek() == Some(b'(') => {
                self.index += 1;
                let value = self.evaluate(test);
                self.space();
                if self.peek() == Some(b')') {
                    self.index += 1;
                }
                value.negate()
            }
            "test" => {
                if test {
                    Possible::TRUE
                } else {
                    Possible::FALSE
                }
            }
            _ => {
                self.skip_atom();
                Possible::FREE
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.text.get(self.index).copied()
    }

    fn space(&mut self) {
        while self.peek().is_some_and(|byte| byte.is_ascii_whitespace()) {
            self.index += 1;
        }
    }

    fn word(&mut self) -> String {
        let start = self.index;
        while self
            .peek()
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        {
            self.index += 1;
        }
        String::from_utf8_lossy(&self.text[start..self.index]).into_owned()
    }

    /// A non-`test` atom: `feature = "metal"`, `version("1.80")`, `target_os`.
    fn skip_atom(&mut self) {
        self.space();
        match self.peek() {
            Some(b'=') => {
                self.index += 1;
                self.space();
                if self.peek() == Some(b'"') {
                    self.skip_string();
                } else {
                    self.word();
                }
            }
            Some(b'(') => {
                let mut depth = 0usize;
                while let Some(byte) = self.peek() {
                    if byte == b'"' {
                        self.skip_string();
                        continue;
                    }
                    match byte {
                        b'(' => depth += 1,
                        b')' => {
                            depth -= 1;
                            if depth == 0 {
                                self.index += 1;
                                return;
                            }
                        }
                        _ => {}
                    }
                    self.index += 1;
                }
            }
            _ => {}
        }
    }

    fn skip_string(&mut self) {
        self.index += 1;
        while let Some(byte) = self.peek() {
            self.index += 1;
            match byte {
                b'\\' => self.index += 1,
                b'"' => return,
                _ => {}
            }
        }
    }
}

/// The files a test-gated item's own `mod NAME;` declarations resolve to, so a
/// test-only module file is not mistaken for production source.
fn gated_module_files(declaring: &Path, gated: &[String]) -> Vec<PathBuf> {
    let base = out_of_line_module_base(declaring);
    let mut files = Vec::new();
    for item in gated {
        if let Some(target) = declared_path(item) {
            files.push(base.join(target));
        }
        if let Some(name) = declared_module(item) {
            files.push(base.join(format!("{name}.rs")));
            files.push(base.join(&name).join("mod.rs"));
        }
    }
    files
}

fn out_of_line_module_base(declaring: &Path) -> PathBuf {
    let parent = declaring.parent().unwrap_or_else(|| Path::new(""));
    let name = declaring
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    if name == "lib.rs" || name == "main.rs" || name == "mod.rs" {
        parent.to_path_buf()
    } else {
        parent.join(declaring.file_stem().unwrap_or_default())
    }
}

fn declared_path(item: &str) -> Option<PathBuf> {
    let rest = item.split("#[path").nth(1)?;
    let rest = rest.split('"').nth(1)?;
    Some(PathBuf::from(rest))
}

fn declared_module(item: &str) -> Option<String> {
    let rest = item.split("mod ").nth(1)?;
    let name: String = rest
        .chars()
        .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
        .collect();
    rest.trim_start_matches(&name)
        .trim_start()
        .starts_with(';')
        .then_some(name)
}

/// Comments, string literals, raw strings and char literals are not code: an
/// attribute marker inside one is not an attribute.
fn skip_literal(bytes: &[u8], index: usize) -> Option<usize> {
    match (bytes[index], bytes.get(index + 1).copied()) {
        (b'/', Some(b'/')) => {
            let end = bytes[index..]
                .iter()
                .position(|byte| *byte == b'\n')
                .map_or(bytes.len(), |offset| index + offset + 1);
            Some(end)
        }
        (b'/', Some(b'*')) => {
            let mut depth = 0usize;
            let mut cursor = index;
            while cursor + 1 < bytes.len() {
                match (bytes[cursor], bytes[cursor + 1]) {
                    (b'/', b'*') => {
                        depth += 1;
                        cursor += 2;
                    }
                    (b'*', b'/') => {
                        depth -= 1;
                        cursor += 2;
                        if depth == 0 {
                            return Some(cursor);
                        }
                    }
                    _ => cursor += 1,
                }
            }
            Some(bytes.len())
        }
        (b'"', _) => {
            let mut cursor = index + 1;
            while cursor < bytes.len() {
                match bytes[cursor] {
                    b'\\' => cursor += 2,
                    b'"' => return Some(cursor + 1),
                    _ => cursor += 1,
                }
            }
            Some(bytes.len())
        }
        (b'r', Some(b'"')) | (b'r', Some(b'#')) => raw_string_end(bytes, index),
        (b'\'', _) => char_literal_end(bytes, index),
        _ => None,
    }
}

fn raw_string_end(bytes: &[u8], index: usize) -> Option<usize> {
    let mut cursor = index + 1;
    let mut hashes = 0usize;
    while bytes.get(cursor) == Some(&b'#') {
        hashes += 1;
        cursor += 1;
    }
    if bytes.get(cursor) != Some(&b'"') {
        return None;
    }
    cursor += 1;
    while cursor < bytes.len() {
        if bytes[cursor] == b'"'
            && bytes[cursor + 1..]
                .iter()
                .take(hashes)
                .filter(|byte| **byte == b'#')
                .count()
                == hashes
        {
            return Some(cursor + 1 + hashes);
        }
        cursor += 1;
    }
    Some(bytes.len())
}

/// `'x'`, `'\n'`, `'\u{..}'`; a lifetime such as `'a` is code, not a literal.
fn char_literal_end(bytes: &[u8], index: usize) -> Option<usize> {
    match bytes.get(index + 1) {
        Some(b'\\') => bytes[index + 2..]
            .iter()
            .position(|byte| *byte == b'\'')
            .map(|offset| index + 3 + offset),
        Some(_) if bytes.get(index + 2) == Some(&b'\'') => Some(index + 3),
        _ => None,
    }
}

fn skip_whitespace_and_comments(bytes: &[u8], mut index: usize) -> Option<usize> {
    while index < bytes.len() {
        if bytes[index].is_ascii_whitespace() {
            index += 1;
            continue;
        }
        match skip_literal(bytes, index) {
            Some(end) if bytes[index] == b'/' => index = end,
            _ => return Some(index),
        }
    }
    Some(index)
}

#[cfg(test)]
mod tests {
    use super::{Mention, captured_families, mentions, split_test_gated};
    use std::fs;
    use std::path::{Path, PathBuf};

    /// The control plants the captured family's own name, so a planted literal
    /// is the same bytes the scan above is looking for.
    const FAMILY: &str = "flint";

    fn crate_roots() -> Vec<PathBuf> {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        ["src", "../proxima-tensor/src", "../omega/src"]
            .iter()
            .map(|relative| manifest.join(relative))
            .collect()
    }

    /// The family's data home: the one directory in the scanned tree where the
    /// name is a key rather than a branch.
    fn data_home() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/profiles")
    }

    fn write(path: &Path, text: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .unwrap_or_else(|error| panic!("{}: {error}", parent.display()));
        }
        fs::write(path, text).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    }

    /// R8's mechanical half: a captured family's name is data, so it lives in
    /// the profile file it labels and nowhere a release build compiles -- doc
    /// comments included, unlike the quoted-literal guard in `tests/`, which
    /// reads dispatch out of code. The three roots here are the crates that
    /// build a model graph from a `general.architecture` string;
    /// `proxima-tokenizer`'s `PreType` table is keyed on the checkpoint's own
    /// `tokenizer.ggml.pre` name and is a table of pre-split rules, not a family.
    #[test]
    fn non_test_source_outside_the_profiles_never_names_a_captured_family() {
        let roots = crate_roots();
        for root in &roots {
            assert!(
                root.is_dir(),
                "{} is not a directory: a root that moved must fail here, not scan nothing",
                root.display()
            );
        }
        let families = captured_families();
        assert!(
            !families.is_empty(),
            "no captured family found under tests/fixtures: the scan would pass vacuously"
        );

        let mut found = Vec::new();
        for family in &families {
            found.extend(mentions(&roots, &data_home(), family));
        }

        assert!(
            found.is_empty(),
            "a captured family's name belongs in `profiles/`, not in these {} production \
             lines:\n{}",
            found.len(),
            found
                .iter()
                .map(Mention::to_string)
                .collect::<Vec<String>>()
                .join("\n")
        );
    }

    /// AC28: the scan is not vacuous. A planted production mention is found, and
    /// no mention the release build never compiles is -- otherwise a scanner
    /// that reported everything, or nothing, would pass the test above.
    #[test]
    fn control_finds_planted_literal() {
        let tree = tempfile::tempdir().expect("a temporary tree for the planted sources");
        let root = tree.path().join("src");
        let planted_data_home = root.join("profiles");
        write(
            &root.join("lib.rs"),
            "mod planted;\n#[cfg(test)]\nmod declared_tests;\n",
        );
        write(
            &root.join("planted.rs"),
            &format!("pub const FAMILY: &str = \"{FAMILY}\";\n"),
        );
        write(
            &root.join("declared_tests.rs"),
            &format!("pub const FAMILY: &str = \"{FAMILY}\";\n"),
        );
        write(
            &root.join("gated_inline.rs"),
            &format!(
                "pub const KEPT: u8 = 1;\n#[cfg(all(test, feature = \"std\"))]\nmod hidden {{\n    pub const FAMILY: &str = \"{FAMILY}\";\n}}\n"
            ),
        );
        write(
            &root.join("release_only.rs"),
            &format!("#[cfg(not(test))]\npub const FAMILY: &str = \"{FAMILY}\";\n"),
        );
        write(
            &root.join("any_gate.rs"),
            &format!(
                "#[cfg(any(test, feature = \"std\"))]\nmod compiled_in_release {{\n    pub const FAMILY: &str = \"{FAMILY}\";\n}}\n"
            ),
        );
        write(
            &planted_data_home.join("planted.rs"),
            &format!("pub const FAMILY: &str = \"{FAMILY}\";\n"),
        );
        write(
            &root.join("gated_whole.rs"),
            &format!(
                "#![cfg(all(test, feature = \"std\"))]\npub const FAMILY: &str = \"{FAMILY}\";\n"
            ),
        );
        write(
            &root.join("named_tests.rs"),
            &format!("pub const FAMILY: &str = \"{FAMILY}\";\n"),
        );

        let found = mentions(&[root], &planted_data_home, FAMILY);
        let mut found: Vec<String> = found
            .iter()
            .map(|mention| {
                format!(
                    "{}:{}",
                    mention
                        .path
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    mention.line
                )
            })
            .collect();
        found.sort();

        assert_eq!(
            found,
            [
                "any_gate.rs:3".to_string(),
                "planted.rs:1".to_string(),
                "release_only.rs:2".to_string(),
            ],
            "the production mentions are found and the test-only ones are not"
        );
    }

    /// The rule the two tests above rest on, stated on its own: `all(test, ..)`
    /// is test-only, `not(test)` and `any(test, ..)` are release code.
    #[test]
    fn a_release_build_keeps_every_gate_that_any_arm_can_satisfy_without_test() {
        let source = concat!(
            "#[cfg(test)]\nmod a { const X: &str = \"kept out\"; }\n",
            "#[cfg(all(test, feature = \"std\"))]\nmod b { const X: &str = \"kept out\"; }\n",
            "#[cfg(any(test, feature = \"std\"))]\nmod c { const X: &str = \"kept\"; }\n",
            "#[cfg(not(test))]\nmod d { const X: &str = \"kept\"; }\n",
        );

        let split = split_test_gated(source);

        assert_eq!(
            split.gated.len(),
            2,
            "two items cannot compile without test"
        );
        assert!(!split.kept.contains("kept out"));
        assert_eq!(split.kept.matches("\"kept\"").count(), 2);
    }
}
