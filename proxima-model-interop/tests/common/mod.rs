use std::env::{split_paths, var, var_os};
use std::fs::read_dir;
use std::path::{Path, PathBuf};

const MODEL_DIRS_ENV: &str = "PROXIMA_MODEL_DIRS";
const DEFAULT_MODEL_DIRS: [&str; 2] = ["/mnt/data3/models", "/mnt/data5/models"];

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

fn gguf_with_length(directory: &Path, length: u64) -> Option<PathBuf> {
    read_dir(directory)
        .ok()?
        .filter_map(Result::ok)
        .find(|entry| entry.metadata().is_ok_and(|info| info.is_file() && info.len() == length))
        .map(|entry| entry.path())
}

/// Resolution order: the named env var, the recorded path, then any file of the
/// recorded byte length in `PROXIMA_MODEL_DIRS` (default `$HOME/models` and the
/// shared model mounts). The length is the identity; a wrong file still fails
/// its content fixtures.
pub fn resolve_checkpoint(env_name: &str, recorded: &str, length: u64) -> PathBuf {
    if let Ok(path) = var(env_name) {
        return PathBuf::from(path);
    }
    if Path::new(recorded).exists() {
        return PathBuf::from(recorded);
    }
    if length == 0 {
        return PathBuf::from(recorded);
    }
    search_dirs()
        .iter()
        .find_map(|directory| gguf_with_length(directory, length))
        .unwrap_or_else(|| PathBuf::from(recorded))
}
