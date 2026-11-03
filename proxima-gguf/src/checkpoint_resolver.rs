//! Where real-checkpoint tests find their GGUF files, shared by every crate whose tests read one:
//! compiled only for unit tests or under `test-support`, never in a normal build.

use std::env::{split_paths, var, var_os};
use std::fs::read_dir;
use std::path::{Path, PathBuf};

pub const MODEL_DIRS_ENV: &str = "PROXIMA_MODEL_DIRS";
const DEFAULT_MODEL_DIRS: [&str; 5] = [
    "/mnt/data3/models",
    "/mnt/data5/models",
    "/mnt/data5/models/infer-fixtures",
    "/mnt/data5/ollama/blobs",
    "/home/bix/repos/llama.cpp-incumbent/models",
];

/// `PROXIMA_MODEL_DIRS` when set, else `$HOME/models` followed by the shared model mounts.
#[must_use]
pub fn model_search_dirs() -> Vec<PathBuf> {
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
/// byte length in the search dirs. The length is the identity; a wrong file still fails its
/// content fixtures. A recorded length of zero skips the length probe.
#[must_use]
pub fn resolve_checkpoint(env_name: &str, recorded: &str, length: u64) -> PathBuf {
    if let Ok(path) = var(env_name) {
        return PathBuf::from(path);
    }
    if Path::new(recorded).exists() {
        return PathBuf::from(recorded);
    }
    let dirs = model_search_dirs();
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

/// [`resolve_checkpoint`] that fails the calling test loudly when nothing resolves, so a missing
/// fixture is a red test rather than a silent pass.
///
/// # Panics
/// If the resolved path does not exist.
#[must_use]
pub fn require_checkpoint(env_name: &str, recorded: &str, length: u64) -> PathBuf {
    let path = resolve_checkpoint(env_name, recorded, length);
    assert!(
        path.exists(),
        "no gguf fixture at {}: set {env_name} or stage the blob (recorded {recorded})",
        path.display()
    );
    path
}
