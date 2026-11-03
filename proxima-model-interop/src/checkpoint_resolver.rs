//! Where the real-checkpoint tests find their GGUF files, shared by this crate's unit tests and
//! its `tests/`: compiled only for unit tests or under `test-support`, never in a normal build.
//! The checkpoint resolver itself lives in `proxima_gguf::checkpoint_resolver`, which every
//! crate with real-checkpoint tests shares.

use std::env::var;
use std::path::PathBuf;

use serde_json::Value;

pub use proxima_gguf::checkpoint_resolver::{MODEL_DIRS_ENV, model_search_dirs, resolve_checkpoint};

const FIXTURES_ENV: &str = "PROXIMA_FIXTURES";
const MANIFEST: &str = "manifest.jsonl";

fn fixtures_root() -> Result<PathBuf, String> {
    if let Ok(root) = var(FIXTURES_ENV) {
        return Ok(PathBuf::from(root));
    }
    model_search_dirs()
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
