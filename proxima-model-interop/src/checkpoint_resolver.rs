//! Where the real-checkpoint tests find their GGUF files, shared by this crate's unit tests and
//! its `tests/`: compiled only for unit tests or under `test-support`, never in a normal build.

use std::env::{split_paths, var, var_os};
use std::fs::read_dir;
use std::path::{Path, PathBuf};

use serde_json::Value;

const MODEL_DIRS_ENV: &str = "PROXIMA_MODEL_DIRS";
const FIXTURES_ENV: &str = "PROXIMA_FIXTURES";
const MANIFEST: &str = "manifest.jsonl";
const DEFAULT_MODEL_DIRS: [&str; 4] = [
    "/mnt/data3/models",
    "/mnt/data5/models",
    "/mnt/data5/models/infer-fixtures",
    "/mnt/data5/ollama/blobs",
];

fn search_dirs() -> Vec<PathBuf> {
    if let Some(configured) = var_os(MODEL_DIRS_ENV) {
        return split_paths(&configured).collect();
    }
    let home_models = var("HOME").map(|home| Path::new(&home).join("models"));
    home_models
        .into_iter()
        .chain(DEFAULT_MODEL_DIRS.iter().map(PathBuf::from))
        .collect()
}

fn file_with_length(directory: &Path, length: u64) -> Option<PathBuf> {
    read_dir(directory)
        .ok()?
        .filter_map(Result::ok)
        .find(|entry| {
            entry
                .metadata()
                .is_ok_and(|info| info.is_file() && info.len() == length)
        })
        .map(|entry| entry.path())
}

/// Resolution order: the named env var, the recorded path, a file in the search dirs with the
/// recorded file name (ollama blobs are named by their own sha256), then any file of the recorded
/// byte length in `PROXIMA_MODEL_DIRS` (default `$HOME/models` and the shared model mounts). The
/// length is the identity; a wrong file still fails its content fixtures.
pub fn resolve_checkpoint(env_name: &str, recorded: &str, length: u64) -> PathBuf {
    if let Ok(path) = var(env_name) {
        return PathBuf::from(path);
    }
    if Path::new(recorded).exists() {
        return PathBuf::from(recorded);
    }
    let dirs = search_dirs();
    let named = Path::new(recorded).file_name().map(PathBuf::from);
    let by_name =
        named.and_then(|name| dirs.iter().map(|dir| dir.join(&name)).find(|path| path.is_file()));
    if let Some(path) = by_name {
        return path;
    }
    if length == 0 {
        return PathBuf::from(recorded);
    }
    dirs.iter()
        .find_map(|directory| file_with_length(directory, length))
        .unwrap_or_else(|| PathBuf::from(recorded))
}

fn fixtures_root() -> Result<PathBuf, String> {
    if let Ok(root) = var(FIXTURES_ENV) {
        return Ok(PathBuf::from(root));
    }
    search_dirs()
        .into_iter()
        .find(|dir| dir.join(MANIFEST).is_file())
        .ok_or_else(|| {
            format!("{FIXTURES_ENV} is not set and no {MODEL_DIRS_ENV} entry holds a {MANIFEST}")
        })
}

/// The GGUF a manifest row names, under `PROXIMA_FIXTURES` or the first model dir holding a
/// `manifest.jsonl`.
pub fn fixture_path(row: u8) -> Result<PathBuf, String> {
    let root = fixtures_root().map_err(|error| format!("row {row}: {error}"))?;
    let manifest = root.join(MANIFEST);
    let text = std::fs::read_to_string(&manifest)
        .map_err(|error| format!("row {row}: cannot read {}: {error}", manifest.display()))?;
    for line in text.lines() {
        let entry: Value = serde_json::from_str(line)
            .map_err(|error| format!("row {row}: bad line in {}: {error}", manifest.display()))?;
        if entry["row"].as_u64() == Some(u64::from(row)) {
            let file = entry["file"]
                .as_str()
                .ok_or_else(|| format!("row {row}: no `file` key in {}", manifest.display()))?;
            return Ok(root.join(file));
        }
    }
    Err(format!("row {row}: not present in {}", manifest.display()))
}

/// The manifest file of the fixture directory, for tests that read it directly.
pub fn fixture_manifest() -> Result<PathBuf, String> {
    fixtures_root().map(|root| root.join(MANIFEST))
}
