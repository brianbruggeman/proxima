//! Where real-checkpoint tests find their GGUF files, shared by every crate whose tests read one:
//! compiled only for unit tests or under `test-support`, never in a normal build.

use std::env::{split_paths, var, var_os};
use std::fs::read_dir;
use std::path::{Path, PathBuf};

pub const MODEL_DIRS_ENV: &str = "PROXIMA_MODEL_DIRS";
pub const LLAMA_CPP_DIR_ENV: &str = "PROXIMA_LLAMA_CPP_DIR";
const DEFAULT_LLAMA_CPP_CHECKOUT: &str = "repos/llama.cpp-incumbent";
const DEFAULT_MODEL_DIRS: [&str; 4] = [
    "/mnt/data3/models",
    "/mnt/data5/models",
    "/mnt/data5/models/infer-fixtures",
    "/mnt/data5/ollama/blobs",
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
        .chain(llama_cpp_models_dir())
        .collect()
}

/// `models/` under `PROXIMA_LLAMA_CPP_DIR`, else under `$HOME/repos/llama.cpp-incumbent`; llama.cpp ships its vocab-only ggufs there.
fn llama_cpp_models_dir() -> Option<PathBuf> {
    let checkout = var_os(LLAMA_CPP_DIR_ENV).map(PathBuf::from).or_else(|| {
        var("HOME")
            .ok()
            .map(|home| Path::new(&home).join(DEFAULT_LLAMA_CPP_CHECKOUT))
    })?;
    Some(checkout.join("models"))
}

fn has_length(path: &Path, length: u64) -> bool {
    length == 0
        || path
            .metadata()
            .is_ok_and(|info| info.is_file() && info.len() == length)
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
/// content fixtures. Every candidate must match the recorded length, and require_checkpoint rejects an env path of another length; a recorded length of zero skips the length probe.
#[must_use]
pub fn resolve_checkpoint(env_name: &str, recorded: &str, length: u64) -> PathBuf {
    if let Ok(path) = var(env_name) {
        return PathBuf::from(path);
    }
    if has_length(Path::new(recorded), length) && Path::new(recorded).exists() {
        return PathBuf::from(recorded);
    }
    let dirs = model_search_dirs();
    let named = Path::new(recorded).file_name().map(PathBuf::from);
    let by_name = named.and_then(|name| {
        dirs.iter()
            .map(|dir| dir.join(&name))
            .find(|path| path.is_file() && has_length(path, length))
    });
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
/// If the resolved path does not exist, or exists with a byte length other than the recorded one.
#[must_use]
pub fn require_checkpoint(env_name: &str, recorded: &str, length: u64) -> PathBuf {
    let path = resolve_checkpoint(env_name, recorded, length);
    assert!(
        path.exists(),
        "no gguf fixture at {}: set {env_name} or stage the blob (recorded {recorded})",
        path.display()
    );
    if length != 0 {
        let actual = path.metadata().map_or(0, |info| info.len());
        assert_eq!(
            actual,
            length,
            "gguf fixture {} is {actual} bytes, expected {length} (truncated download or wrong file; env {env_name}, recorded {recorded})",
            path.display()
        );
    }
    path
}

#[cfg(test)]
mod tests {
    use std::fs::{create_dir_all, write};

    use tempfile::tempdir;

    use super::*;

    const ENV_NAME: &str = "PROXIMA_CHECKPOINT_RESOLVER_TEST_GGUF";

    #[test]
    fn truncated_file_does_not_satisfy_the_recorded_length() {
        let directory = tempdir().expect("temp dir");
        let truncated = directory.path().join("model.gguf");
        write(&truncated, [0u8; 10]).expect("write truncated file");

        assert!(!has_length(&truncated, 100));
        assert!(has_length(&truncated, 10));
        assert!(has_length(&truncated, 0));
    }

    #[test]
    #[should_panic(expected = "is 10 bytes, expected 100")]
    fn require_rejects_a_truncated_recorded_file() {
        let directory = tempdir().expect("temp dir");
        let truncated = directory.path().join("model.gguf");
        write(&truncated, [0u8; 10]).expect("write truncated file");

        temp_env::with_vars([(ENV_NAME, None::<&str>)], || {
            let _ = require_checkpoint(ENV_NAME, truncated.to_str().expect("utf8 path"), 100);
        });
    }

    #[test]
    #[should_panic(expected = "is 10 bytes, expected 100")]
    fn require_rejects_a_truncated_env_path() {
        let directory = tempdir().expect("temp dir");
        let truncated = directory.path().join("model.gguf");
        write(&truncated, [0u8; 10]).expect("write truncated file");

        temp_env::with_vars([(ENV_NAME, truncated.to_str())], || {
            let _ = require_checkpoint(ENV_NAME, "/nonexistent/model.gguf", 100);
        });
    }

    #[test]
    fn full_file_in_a_search_dir_beats_a_truncated_recorded_one() {
        let directory = tempdir().expect("temp dir");
        let search = directory.path().join("search");
        create_dir_all(&search).expect("create search dir");
        let truncated = directory.path().join("model.gguf");
        write(&truncated, [0u8; 10]).expect("write truncated file");
        write(search.join("model.gguf"), [0u8; 100]).expect("write full file");

        let resolved = temp_env::with_vars(
            [
                (ENV_NAME, None),
                (MODEL_DIRS_ENV, Some(search.to_str().expect("utf8 path"))),
            ],
            || require_checkpoint(ENV_NAME, truncated.to_str().expect("utf8 path"), 100),
        );

        assert_eq!(resolved, search.join("model.gguf"));
    }
}
