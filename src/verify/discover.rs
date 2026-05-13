//! Project-root discovery for spec / recording / policy files.
//! Implements `proxima verify` and `proxima replay` zero-arg
//! invocation — the user runs the CLI at a project root and the
//! conventional file locations get picked up automatically.
//!
//! Rules:
//!
//! - Spec: first match of `./proxima.toml`, `./spec.toml`,
//!   `./proxima.json`, `./.proxima/spec.toml`.
//! - Policy: first match of `./proxima.policy.toml`,
//!   `./.proxima/policy.toml`.
//! - Recording: newest file matching `./.proxima/recordings/*.bin`
//!   or `./recordings/*.bin`.

use std::path::{Path, PathBuf};

use crate::error::ProximaError;

const SPEC_CANDIDATES: &[&str] = &[
    "proxima.toml",
    "spec.toml",
    "proxima.json",
    ".proxima/spec.toml",
];

const POLICY_CANDIDATES: &[&str] = &["proxima.policy.toml", ".proxima/policy.toml"];

const RECORDING_DIRS: &[&str] = &[".proxima/recordings", "recordings"];

/// Discover the spec file at the given root. Returns the first
/// candidate that exists.
pub fn discover_spec(root: &Path) -> Option<PathBuf> {
    first_existing(root, SPEC_CANDIDATES)
}

/// Discover the policy file at the given root. Returns the first
/// candidate that exists.
pub fn discover_policy(root: &Path) -> Option<PathBuf> {
    first_existing(root, POLICY_CANDIDATES)
}

/// Discover the newest recording at the given root. Walks both
/// recording-dir candidates; returns the newest `.bin` or `.jsonl`
/// file across either dir.
pub fn discover_newest_recording(root: &Path) -> Result<Option<PathBuf>, ProximaError> {
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for dir_name in RECORDING_DIRS {
        let dir = root.join(dir_name);
        if !dir.is_dir() {
            continue;
        }
        let entries = std::fs::read_dir(&dir).map_err(ProximaError::Io)?;
        for entry in entries {
            let entry = entry.map_err(ProximaError::Io)?;
            let path = entry.path();
            let Some(extension) = path.extension().and_then(|raw| raw.to_str()) else {
                continue;
            };
            if extension != "bin" && extension != "jsonl" {
                continue;
            }
            let modified = entry
                .metadata()
                .map_err(ProximaError::Io)?
                .modified()
                .map_err(ProximaError::Io)?;
            match &best {
                Some((current, _)) if *current >= modified => {}
                _ => best = Some((modified, path)),
            }
        }
    }
    Ok(best.map(|(_, path)| path))
}

fn first_existing(root: &Path, candidates: &[&str]) -> Option<PathBuf> {
    candidates
        .iter()
        .map(|relative| root.join(relative))
        .find(|path| path.is_file())
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default,
        clippy::type_complexity,
        clippy::useless_vec,
        clippy::needless_range_loop,
        clippy::default_constructed_unit_structs
    )]
    use super::*;
    use std::fs;
    use std::io::Write;

    #[test]
    fn discover_spec_picks_first_match_in_priority_order() {
        let temp = tempfile::tempdir().expect("tempdir");
        let proxima_toml = temp.path().join("proxima.toml");
        let spec_toml = temp.path().join("spec.toml");
        fs::write(&proxima_toml, "").expect("write proxima.toml");
        fs::write(&spec_toml, "").expect("write spec.toml");

        let discovered = discover_spec(temp.path()).expect("discover");
        assert_eq!(
            discovered, proxima_toml,
            "proxima.toml ranks above spec.toml"
        );
    }

    #[test]
    fn discover_spec_falls_through_when_top_missing() {
        let temp = tempfile::tempdir().expect("tempdir");
        let spec_toml = temp.path().join("spec.toml");
        fs::write(&spec_toml, "").expect("write spec.toml");
        let discovered = discover_spec(temp.path()).expect("discover");
        assert_eq!(discovered, spec_toml);
    }

    #[test]
    fn discover_spec_finds_dot_proxima_subdir() {
        let temp = tempfile::tempdir().expect("tempdir");
        let subdir = temp.path().join(".proxima");
        fs::create_dir_all(&subdir).expect("mkdir .proxima");
        let spec = subdir.join("spec.toml");
        fs::write(&spec, "").expect("write");
        let discovered = discover_spec(temp.path()).expect("discover");
        assert_eq!(discovered, spec);
    }

    #[test]
    fn discover_spec_returns_none_when_no_candidate() {
        let temp = tempfile::tempdir().expect("tempdir");
        assert!(discover_spec(temp.path()).is_none());
    }

    #[test]
    fn discover_policy_finds_top_level_first() {
        let temp = tempfile::tempdir().expect("tempdir");
        let top = temp.path().join("proxima.policy.toml");
        fs::write(&top, "").expect("write");
        let discovered = discover_policy(temp.path()).expect("discover");
        assert_eq!(discovered, top);
    }

    #[test]
    fn discover_newest_recording_returns_the_latest_file() {
        let temp = tempfile::tempdir().expect("tempdir");
        let dir = temp.path().join(".proxima").join("recordings");
        fs::create_dir_all(&dir).expect("mkdir recordings");

        let older = dir.join("a.bin");
        let newer = dir.join("b.bin");
        let mut file_a = fs::File::create(&older).expect("create a.bin");
        file_a.write_all(b"x").expect("write");
        drop(file_a);
        // ensure modification times differ even on coarse filesystems
        std::thread::sleep(std::time::Duration::from_millis(10));
        let mut file_b = fs::File::create(&newer).expect("create b.bin");
        file_b.write_all(b"y").expect("write");
        drop(file_b);

        let discovered = discover_newest_recording(temp.path())
            .expect("discover")
            .expect("found");
        assert_eq!(discovered, newer);
    }

    #[test]
    fn discover_newest_recording_skips_non_bin_files() {
        let temp = tempfile::tempdir().expect("tempdir");
        let dir = temp.path().join("recordings");
        fs::create_dir_all(&dir).expect("mkdir recordings");
        let non_bin = dir.join("note.txt");
        fs::write(&non_bin, "").expect("write");
        let discovered = discover_newest_recording(temp.path()).expect("discover");
        assert!(discovered.is_none(), "txt should not match");
    }

    #[test]
    fn discover_newest_recording_returns_none_with_no_dir() {
        let temp = tempfile::tempdir().expect("tempdir");
        let discovered = discover_newest_recording(temp.path()).expect("discover");
        assert!(discovered.is_none());
    }
}
