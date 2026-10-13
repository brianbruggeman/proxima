//! guard: production source in the three crates that carry model-family
//! knowledge (`proxima-model-interop`, `proxima-tensor`, `omega`) must not quote
//! a family name. the registry is `src/profiles/` (the `FAMILY_PROFILES` rows and
//! the binding rows): the one place a family name is data. everything else asks
//! the descriptor what the model DOES, not which model it IS. a quoted name in
//! production code is the regression this test exists to catch.

use std::collections::{HashMap, HashSet};
use std::io;
use std::path::{Path, PathBuf};

const SCAN_ROOTS: &[&str] = &["proxima-model-interop/src", "proxima-tensor/src", "omega/src"];

const REGISTRY_FILES: &[&str] = &["src/profiles/mod.rs", "src/profiles/binding.rs"];

const EXTRA_NAMES: &[&str] = &["granitemoehybrid", "granite-embed-multi-311m"];

// (file relative to the workspace root, quoted literal, why or remove-in tag)
const ALLOWLIST: &[(&str, &str, &str)] = &[
    ("proxima-model-interop/src/profiles/mod.rs", "gemma4", "boundary: FAMILY_PROFILES row"),
    ("proxima-model-interop/src/profiles/mod.rs", "llama", "boundary: FAMILY_PROFILES row"),
    ("proxima-model-interop/src/profiles/mod.rs", "mistral", "boundary: FAMILY_PROFILES row"),
    ("proxima-model-interop/src/profiles/mod.rs", "mixtral", "boundary: FAMILY_PROFILES row"),
    ("proxima-model-interop/src/profiles/mod.rs", "granitemoe", "boundary: FAMILY_PROFILES row"),
    ("proxima-model-interop/src/profiles/mod.rs", "qwen2", "boundary: FAMILY_PROFILES row"),
    ("proxima-model-interop/src/profiles/mod.rs", "qwen3", "boundary: FAMILY_PROFILES row"),
    ("proxima-model-interop/src/profiles/mod.rs", "qwen3moe", "boundary: FAMILY_PROFILES row"),
    ("proxima-model-interop/src/profiles/mod.rs", "qwen3_moe", "boundary: FAMILY_PROFILES row"),
    ("proxima-model-interop/src/profiles/mod.rs", "qwen35", "boundary: FAMILY_PROFILES row"),
    ("proxima-model-interop/src/profiles/mod.rs", "qwen35moe", "boundary: FAMILY_PROFILES row"),
    ("proxima-model-interop/src/profiles/mod.rs", "lfm2", "boundary: FAMILY_PROFILES row"),
    ("proxima-model-interop/src/profiles/mod.rs", "lfm2moe", "boundary: FAMILY_PROFILES row"),
    ("proxima-model-interop/src/profiles/mod.rs", "ouro", "boundary: FAMILY_PROFILES row"),
    ("proxima-model-interop/src/profiles/binding.rs", "gemma4", "boundary: binding row (B3 may fold into mod.rs)"),
    ("proxima-model-interop/src/profiles/binding.rs", "qwen35", "boundary: binding row (B3 may fold into mod.rs)"),
    ("proxima-model-interop/src/profiles/binding.rs", "qwen35moe", "boundary: binding row (B3 may fold into mod.rs)"),
    ("proxima-model-interop/src/profiles/binding.rs", "lfm2", "boundary: binding row (B3 may fold into mod.rs)"),
    ("proxima-model-interop/src/profiles/binding.rs", "lfm2moe", "boundary: binding row (B3 may fold into mod.rs)"),
    ("proxima-model-interop/src/profiles/binding.rs", "granitemoe", "boundary: binding row (B3 may fold into mod.rs)"),
    ("proxima-model-interop/src/hf_config.rs", "lfm2", "remove-in: D1, D2"),
    ("proxima-model-interop/src/hf_descriptor.rs", "ouro", "remove-in: D3"),
    ("proxima-model-interop/src/cold_artifact.rs", "granitemoehybrid", "remove-in: B2"),
    ("proxima-model-interop/src/draft_model_compatibility.rs", "llama", "remove-in: B1"),
    ("proxima-model-interop/src/draft_model_compatibility.rs", "gemma4", "remove-in: B1"),
    ("proxima-model-interop/src/draft_model_compatibility.rs", "lfm2", "remove-in: B1"),
    ("proxima-model-interop/src/draft_model_compatibility.rs", "granite-embed-multi-311m", "remove-in: B1"),
    ("proxima-model-interop/src/transform/mlx_config.rs", "granitemoe", "boundary: MLX FAMILIES rows"),
    ("proxima-model-interop/src/transform/mlx_config.rs", "gemma4", "boundary: MLX FAMILIES rows; String::from at :356 removed in M1"),
];

#[derive(Debug, PartialEq, Eq)]
struct Hit {
    path: String,
    line: usize,
    literal: String,
    text: String,
}

struct Scan {
    files: usize,
    hits: Vec<Hit>,
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn family_names(crate_dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = REGISTRY_FILES
        .iter()
        .flat_map(|rel| {
            let text = std::fs::read_to_string(crate_dir.join(rel))
                .unwrap_or_else(|err| panic!("read {rel}: {err}"));
            registry_row_names(&text)
        })
        .collect();
    names.extend(EXTRA_NAMES.iter().map(|name| (*name).to_owned()));
    names.sort();
    names.dedup();
    names
}

fn registry_row_names(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| {
            let row = line.trim_start().strip_prefix("(\"")?;
            let (name, rest) = row.split_once('"')?;
            rest.starts_with(", include_str!").then(|| name.to_owned())
        })
        .collect()
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            if path.file_name().and_then(|name| name.to_str()) != Some("tests") {
                rust_files(&path, out)?;
            }
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            out.push(path);
        }
    }
    Ok(())
}

fn is_test_cfg(attribute: &str) -> bool {
    let compact: String = attribute.split_whitespace().collect();
    compact.starts_with("#[cfg(test)]") || compact.starts_with("#[cfg(all(test,")
}

fn item_head(trimmed: &str) -> &str {
    match trimmed.strip_prefix("pub") {
        Some(rest) if rest.starts_with('(') => rest.split_once(") ").map_or(trimmed, |(_, head)| head),
        Some(rest) if rest.starts_with(' ') => rest.trim_start(),
        _ => trimmed,
    }
}

fn file_mod_name(trimmed: &str) -> Option<&str> {
    item_head(trimmed)
        .strip_prefix("mod ")?
        .strip_suffix(';')
        .map(str::trim)
}

fn inline_mod_opens(trimmed: &str) -> bool {
    item_head(trimmed).starts_with("mod ") && trimmed.ends_with('{')
}

fn bracket_delta(text: &str) -> isize {
    text.chars()
        .map(|c| match c {
            '[' => 1,
            ']' => -1,
            _ => 0,
        })
        .sum()
}

fn logical_lines(text: &str) -> Vec<(usize, String)> {
    let mut lines = Vec::new();
    let mut open: Option<(usize, String, isize)> = None;
    for (index, line) in text.lines().enumerate() {
        let number = index + 1;
        match open.take() {
            Some((start, mut joined, depth)) => {
                joined.push_str(line.trim());
                let depth = depth + bracket_delta(line);
                if depth > 0 {
                    open = Some((start, joined, depth));
                } else {
                    lines.push((start, joined));
                }
            }
            None => {
                let trimmed = line.trim_start();
                let depth = bracket_delta(trimmed);
                if trimmed.starts_with("#[") && depth > 0 {
                    open = Some((number, trimmed.to_owned(), depth));
                } else {
                    lines.push((number, line.to_owned()));
                }
            }
        }
    }
    lines
}

fn child_candidates(file: &Path, name: &str) -> [PathBuf; 2] {
    let parent = file.parent().unwrap_or(Path::new(""));
    let stem = file.file_stem().and_then(|stem| stem.to_str()).unwrap_or_default();
    let dir = if matches!(stem, "mod" | "lib" | "main") {
        parent.to_path_buf()
    } else {
        parent.join(stem)
    };
    [dir.join(format!("{name}.rs")), dir.join(name).join("mod.rs")]
}

fn module_children(file: &Path, text: &str) -> Vec<(PathBuf, bool)> {
    let mut children = Vec::new();
    let mut gated = false;
    for (_, line) in logical_lines(text) {
        let trimmed = line.trim_start();
        if is_test_cfg(trimmed) {
            gated = true;
            continue;
        }
        if trimmed.starts_with("#[") || trimmed.is_empty() {
            continue;
        }
        if let Some(name) = file_mod_name(trimmed) {
            children.extend(child_candidates(file, name).into_iter().map(|path| (path, gated)));
        }
        gated = false;
    }
    children
}

fn test_only_files(sources: &[(PathBuf, String)]) -> HashSet<PathBuf> {
    let texts: HashMap<&Path, &str> = sources.iter().map(|(path, text)| (path.as_path(), text.as_str())).collect();
    let mut frontier: Vec<PathBuf> = sources
        .iter()
        .flat_map(|(path, text)| module_children(path, text))
        .filter_map(|(path, gated)| gated.then_some(path))
        .collect();
    let mut excluded = HashSet::new();
    while let Some(next) = frontier.pop() {
        let Some(text) = texts.get(next.as_path()) else {
            continue;
        };
        if !excluded.insert(next.clone()) {
            continue;
        }
        frontier.extend(module_children(&next, text).into_iter().map(|(path, _)| path));
    }
    excluded
}

fn brace_delta(line: &str) -> isize {
    let mut delta = 0;
    let mut in_string = false;
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        match (in_string, c) {
            (true, '\\') => {
                chars.next();
            }
            (true, '"') => in_string = false,
            (true, _) => {}
            (false, '"') => in_string = true,
            (false, '/') if chars.clone().next() == Some('/') => break,
            (false, '\'') => {
                let mut lookahead = chars.clone();
                if let (Some(_), Some('\'')) = (lookahead.next(), lookahead.next()) {
                    chars.next();
                    chars.next();
                }
            }
            (false, '{') => delta += 1,
            (false, '}') => delta -= 1,
            (false, _) => {}
        }
    }
    delta
}

fn production_hits(rel: &str, text: &str, names: &[String]) -> Vec<Hit> {
    let mut hits = Vec::new();
    let mut gated = false;
    let mut skip_depth: Option<isize> = None;
    for (number, line) in logical_lines(text) {
        let trimmed = line.trim_start();
        if let Some(depth) = skip_depth {
            let depth = depth + brace_delta(&line);
            skip_depth = (depth > 0).then_some(depth);
            continue;
        }
        if is_test_cfg(trimmed) {
            gated = true;
            continue;
        }
        if trimmed.starts_with("#[") || trimmed.is_empty() {
            continue;
        }
        let opens_test_mod = gated && inline_mod_opens(trimmed);
        gated = false;
        if opens_test_mod {
            let depth = brace_delta(&line);
            skip_depth = (depth > 0).then_some(depth);
            continue;
        }
        if trimmed.starts_with("//") {
            continue;
        }
        for name in names {
            if line.contains(&format!("\"{name}\"")) {
                hits.push(Hit {
                    path: rel.to_owned(),
                    line: number,
                    literal: name.clone(),
                    text: trimmed.to_owned(),
                });
            }
        }
    }
    hits
}

fn scan_root(workspace: &Path, label: &str, names: &[String]) -> io::Result<Scan> {
    let base = workspace.join(label);
    let mut paths = Vec::new();
    rust_files(&base, &mut paths)?;
    paths.sort();
    let sources = paths
        .into_iter()
        .map(|path| std::fs::read_to_string(&path).map(|text| (path, text)))
        .collect::<io::Result<Vec<_>>>()?;
    let excluded = test_only_files(&sources);
    let mut scan = Scan { files: 0, hits: Vec::new() };
    for (path, text) in &sources {
        if excluded.contains(path) {
            continue;
        }
        scan.files += 1;
        let relative = path.strip_prefix(&base).map_err(io::Error::other)?;
        let rel = format!("{label}/{}", relative.display());
        scan.hits.extend(production_hits(&rel, text, names));
    }
    Ok(scan)
}

fn unlisted<'a>(hits: &'a [Hit], allowlist: &[(&str, &str, &str)]) -> Vec<&'a Hit> {
    hits.iter()
        .filter(|hit| {
            !allowlist
                .iter()
                .any(|(path, literal, _)| *path == hit.path && *literal == hit.literal)
        })
        .collect()
}

#[test]
fn pipeline_dispatches_on_capability_not_name() {
    let workspace = workspace_root();
    let names = family_names(Path::new(env!("CARGO_MANIFEST_DIR")));
    let mut violations = Vec::new();
    let mut scanned = 0;
    for label in SCAN_ROOTS {
        let scan = scan_root(&workspace, label, &names).expect("walk scan root");
        scanned += scan.files;
        for hit in unlisted(&scan.hits, ALLOWLIST) {
            violations.push(format!("{}:{}: quoted \"{}\" -> {}", hit.path, hit.line, hit.literal, hit.text));
        }
    }
    assert!(
        violations.is_empty(),
        "production source must dispatch on config-derived capability, not a quoted family name \
         (scanned {scanned} files, {} names); interpret the name once in src/profiles/ and read a \
         descriptor field downstream:\n{}",
        names.len(),
        violations.join("\n")
    );
}

#[test]
fn guard_scans_a_nonzero_file_set_per_root() {
    let names = family_names(Path::new(env!("CARGO_MANIFEST_DIR")));
    assert!(names.iter().any(|name| name == "qwen35moe"), "name set lost the qwen35moe registry row: {names:?}");
    let workspace = workspace_root();
    for label in SCAN_ROOTS {
        let scan = scan_root(&workspace, label, &names).expect("walk scan root");
        assert!(scan.files > 0, "{label} scanned 0 production files; the guard would pass vacuously");
    }
}

#[test]
fn guard_flags_a_planted_family_literal_in_a_temp_tree() {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR")).join("arch_g1_control");
    let _ = std::fs::remove_dir_all(&root);
    let src = root.join("planted/src");
    std::fs::create_dir_all(src.join("prod")).expect("create planted tree");
    std::fs::write(
        src.join("prod.rs"),
        "fn tier(name: &str) -> u8 {\n    match name {\n        \"qwen35\" => 1,\n        _ => 0,\n    }\n}\n#[cfg(test)]\nmod tests;\n",
    )
    .expect("write prod.rs");
    std::fs::write(src.join("prod/tests.rs"), "const FIXTURE: &str = \"qwen35\";\n").expect("write tests.rs");
    std::fs::write(
        src.join("inline.rs"),
        "#[cfg(test)]\nmod tests {\n    const FIXTURE: &str = \"qwen35\";\n}\nfn after() -> &'static str {\n    \"qwen35\"\n}\n",
    )
    .expect("write inline.rs");

    let names = vec!["qwen35".to_owned()];
    let scan = scan_root(&root, "planted/src", &names).expect("walk planted tree");
    assert_eq!(scan.files, 2, "prod.rs and inline.rs are production; prod/tests.rs is cfg(test)");
    let flagged: Vec<(String, usize)> = unlisted(&scan.hits, &[])
        .iter()
        .map(|hit| (hit.path.clone(), hit.line))
        .collect();
    assert_eq!(
        flagged,
        vec![("planted/src/inline.rs".to_owned(), 6), ("planted/src/prod.rs".to_owned(), 3)],
    );
}
