//! Scenario file discovery for `proxima load`. Mirrors the
//! [`crate::verify::discover`] pattern: zero-arg invocation looks at
//! conventional project-root paths; `--name <name>` invocation walks
//! a tiered list of search roots (cwd → XDG user → system → env-var
//! override) and returns every matching file by stem.
//!
//! Tier order for `--name` lookup, highest priority first:
//!
//! 1. paths from `PROXIMA_SCENARIO_PATH` env var (colon-separated)
//! 2. `./scenarios/` and `./.proxima/scenarios/` (cwd-relative)
//! 3. `$XDG_CONFIG_HOME/proxima/scenarios/` (or `$HOME/.config/...`)
//! 4. `/etc/proxima/scenarios/` (system)
//!
//! Fuzzy match + interactive picker for ambiguous `--name` lookups
//! is intentionally not in this MVP; callers receive every match and
//! decide.

use std::path::{Path, PathBuf};

/// File names checked at the project root for zero-arg invocation,
/// in priority order. First existing match wins.
const SCENARIO_CANDIDATES: &[&str] = &[
    "proxima.scenario.toml",
    "scenario.toml",
    ".proxima/scenario.toml",
];

/// Directories inside `cwd` that hold named scenario files for
/// `--name` lookup. Ordered: highest priority first.
const CWD_SCENARIO_DIRS: &[&str] = &["scenarios", ".proxima/scenarios"];

/// System-wide scenario directory. Lowest priority tier; only
/// consulted when no closer match exists.
const SYSTEM_SCENARIO_DIR: &str = "/etc/proxima/scenarios";

/// Env-var override for additional search roots. Colon-separated
/// list, prepended at the top of the tier order.
const PROXIMA_SCENARIO_PATH: &str = "PROXIMA_SCENARIO_PATH";

/// Discover the conventional scenario file at the given root for
/// `proxima load` with no args. Returns the first candidate that
/// exists or `None` if none match.
#[must_use]
pub fn discover_scenario(root: &Path) -> Option<PathBuf> {
    SCENARIO_CANDIDATES
        .iter()
        .map(|relative| root.join(relative))
        .find(|path| path.is_file())
}

/// Return the ordered list of search-root directories for `--name`
/// lookup. Highest priority first. Non-existent roots are included
/// so callers can see the full search path in error messages.
#[must_use]
pub fn name_search_roots(cwd: &Path) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(raw) = std::env::var(PROXIMA_SCENARIO_PATH) {
        for entry in raw.split(':').filter(|each| !each.is_empty()) {
            roots.push(PathBuf::from(entry));
        }
    }
    for relative in CWD_SCENARIO_DIRS {
        roots.push(cwd.join(relative));
    }
    if let Some(xdg) = xdg_scenarios_dir() {
        roots.push(xdg);
    }
    roots.push(PathBuf::from(SYSTEM_SCENARIO_DIR));
    roots
}

/// Look up scenario files by stem name across the tiered search
/// roots. Returns every match in priority order; the CLI can show
/// them and ask the user to disambiguate, or pick the first.
#[must_use]
pub fn discover_by_name(name: &str, cwd: &Path) -> Vec<PathBuf> {
    let mut matches = Vec::new();
    let filename = format!("{name}.toml");
    for root in name_search_roots(cwd) {
        let candidate = root.join(&filename);
        if candidate.is_file() {
            matches.push(candidate);
        }
    }
    matches
}

fn xdg_scenarios_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join(".config"))
        })?;
    Some(base.join("proxima").join("scenarios"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn discover_scenario_picks_first_match_in_priority_order() {
        let temp = tempfile::tempdir().expect("tempdir");
        let proxima_scenario = temp.path().join("proxima.scenario.toml");
        let scenario = temp.path().join("scenario.toml");
        fs::write(&proxima_scenario, "").expect("write proxima.scenario.toml");
        fs::write(&scenario, "").expect("write scenario.toml");

        let discovered = discover_scenario(temp.path()).expect("discover");
        assert_eq!(
            discovered, proxima_scenario,
            "proxima.scenario.toml ranks above scenario.toml"
        );
    }

    #[test]
    fn discover_scenario_falls_through_when_top_missing() {
        let temp = tempfile::tempdir().expect("tempdir");
        let scenario = temp.path().join("scenario.toml");
        fs::write(&scenario, "").expect("write");
        let discovered = discover_scenario(temp.path()).expect("discover");
        assert_eq!(discovered, scenario);
    }

    #[test]
    fn discover_scenario_finds_dot_proxima_subdir() {
        let temp = tempfile::tempdir().expect("tempdir");
        let subdir = temp.path().join(".proxima");
        fs::create_dir_all(&subdir).expect("mkdir .proxima");
        let scenario = subdir.join("scenario.toml");
        fs::write(&scenario, "").expect("write");
        let discovered = discover_scenario(temp.path()).expect("discover");
        assert_eq!(discovered, scenario);
    }

    #[test]
    fn discover_scenario_returns_none_when_no_candidate() {
        let temp = tempfile::tempdir().expect("tempdir");
        assert!(discover_scenario(temp.path()).is_none());
    }

    #[test]
    fn name_search_roots_orders_env_above_cwd_above_system() {
        let temp = tempfile::tempdir().expect("tempdir");
        let extra_root = temp.path().join("extra");
        let raw_path = extra_root.to_string_lossy().into_owned();

        let cwd = temp.path().join("project");
        fs::create_dir_all(&cwd).expect("mkdir project");

        temp_env::with_var(PROXIMA_SCENARIO_PATH, Some(raw_path.as_str()), || {
            let roots = name_search_roots(&cwd);
            assert!(
                roots.first().map(PathBuf::as_path) == Some(extra_root.as_path()),
                "env var path must come first; roots={roots:?}"
            );
            assert!(
                roots.contains(&cwd.join("scenarios")),
                "cwd-relative scenarios dir must be present"
            );
            assert!(
                roots.last().map(PathBuf::as_path) == Some(Path::new(SYSTEM_SCENARIO_DIR)),
                "/etc path must come last; roots={roots:?}"
            );
        });
    }

    #[test]
    fn discover_by_name_finds_match_in_cwd_scenarios_dir() {
        let temp = tempfile::tempdir().expect("tempdir");
        let scenarios_dir = temp.path().join("scenarios");
        fs::create_dir_all(&scenarios_dir).expect("mkdir scenarios");
        let h2_soak = scenarios_dir.join("h2-soak.toml");
        fs::write(&h2_soak, "").expect("write");

        temp_env::with_var(PROXIMA_SCENARIO_PATH, None::<&str>, || {
            let matches = discover_by_name("h2-soak", temp.path());
            assert_eq!(matches, vec![h2_soak.clone()]);
        });
    }

    #[test]
    fn discover_by_name_returns_empty_when_no_match() {
        let temp = tempfile::tempdir().expect("tempdir");
        temp_env::with_var(PROXIMA_SCENARIO_PATH, None::<&str>, || {
            let matches = discover_by_name("nonexistent", temp.path());
            assert!(matches.is_empty());
        });
    }

    #[test]
    fn discover_by_name_returns_env_match_before_cwd_match() {
        let temp = tempfile::tempdir().expect("tempdir");
        let env_dir = temp.path().join("env-scenarios");
        fs::create_dir_all(&env_dir).expect("mkdir env-scenarios");
        let env_match = env_dir.join("h2-soak.toml");
        fs::write(&env_match, "").expect("write env");

        let cwd_dir = temp.path().join("scenarios");
        fs::create_dir_all(&cwd_dir).expect("mkdir cwd-scenarios");
        let cwd_match = cwd_dir.join("h2-soak.toml");
        fs::write(&cwd_match, "").expect("write cwd");

        let env_path_string = env_dir.to_string_lossy().into_owned();
        temp_env::with_var(
            PROXIMA_SCENARIO_PATH,
            Some(env_path_string.as_str()),
            || {
                let matches = discover_by_name("h2-soak", temp.path());
                assert_eq!(matches[0], env_match, "env match comes first");
                assert!(
                    matches.contains(&cwd_match),
                    "cwd match also present in list"
                );
            },
        );
    }
}
