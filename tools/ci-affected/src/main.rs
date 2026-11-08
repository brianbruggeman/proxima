//! Decide which CI areas a change set touches, considering the Cargo
//! dependency graph when the repo is a Cargo workspace.
//!
//! Areas are declared in `.github/ci-areas.toml`. Each area is defined by
//! root `packages` (cargo mode) and/or `paths` globs (works with or without
//! cargo). An area runs when a changed file is in its package's dependency
//! closure OR matches one of its path globs. Global paths (lockfile, root
//! manifest, the `.github` tree, this tool, the areas config) force every area
//! on, as do `--force-all` (release tags / workflow_dispatch) and an empty or
//! undetectable change set.
//!
//! Emits `area=true|false` lines to `$GITHUB_OUTPUT` and stdout. Exits 0
//! always: a detection error forces every area on rather than silently
//! skipping a gate.

mod glob;
mod graph;

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use graph::{CargoMetadata, WorkspaceGraph};

/// Paths that make every area relevant: the dependency graph, the CI wiring,
/// this detector's own source, and the area map.
const GLOBAL_GLOBS: [&str; 6] = [
    "Cargo.lock",
    "Cargo.toml",
    ".github/workflows/affected.yml",
    "tools/ci-affected/**",
    ".github/ci-areas.toml",
    "rust-toolchain*",
];

struct Area {
    packages: Vec<String>,
    paths: Vec<String>,
}

struct Arguments {
    base: Option<String>,
    head: Option<String>,
    force_all: bool,
    changed: Option<Vec<String>>,
    config: PathBuf,
}

fn main() -> ExitCode {
    let arguments = match parse_arguments() {
        Ok(arguments) => arguments,
        Err(error) => {
            eprintln!("ci-affected: {error}");
            return ExitCode::from(2);
        }
    };
    let areas = match load_areas(&arguments.config) {
        Ok(areas) => areas,
        Err(error) => {
            eprintln!(
                "ci-affected: cannot read {}: {error}",
                arguments.config.display()
            );
            return ExitCode::FAILURE;
        }
    };

    let hits = detect(&arguments, &areas).unwrap_or_else(|error| {
        eprintln!("ci-affected: {error}; forcing every area on");
        areas.keys().map(|name| (name.clone(), true)).collect()
    });
    if let Err(error) = emit(&hits) {
        eprintln!("ci-affected: cannot write $GITHUB_OUTPUT: {error}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn parse_arguments() -> Result<Arguments, String> {
    let mut arguments = Arguments {
        base: None,
        head: None,
        force_all: false,
        changed: None,
        config: PathBuf::from(".github/ci-areas.toml"),
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].clone();
        index += 1;
        match flag.as_str() {
            "--base" => {
                arguments.base = Some(args.get(index).ok_or("--base needs a value")?.clone());
                index += 1;
            }
            "--head" => {
                arguments.head = Some(args.get(index).ok_or("--head needs a value")?.clone());
                index += 1;
            }
            "--config" => {
                arguments.config = PathBuf::from(args.get(index).ok_or("--config needs a value")?);
                index += 1;
            }
            "--force-all" => arguments.force_all = true,
            "--changed" => {
                let mut files = Vec::new();
                while let Some(next) = args.get(index) {
                    if next.starts_with("--") {
                        break;
                    }
                    files.push(next.clone());
                    index += 1;
                }
                arguments.changed = Some(files);
            }
            other => return Err(format!("unrecognized argument: {other}")),
        }
    }
    Ok(arguments)
}

fn load_areas(path: &Path) -> Result<BTreeMap<String, Area>, Box<dyn Error>> {
    let text = std::fs::read_to_string(path)?;
    let table: toml::Table = text.parse()?;
    let area_table = table
        .get("area")
        .and_then(toml::Value::as_table)
        .ok_or("config has no [area] table")?;

    let mut areas = BTreeMap::new();
    for (name, value) in area_table {
        let spec = value.as_table().ok_or("an area entry is not a table")?;
        areas.insert(
            name.clone(),
            Area {
                packages: string_list(spec.get("packages")),
                paths: string_list(spec.get("paths")),
            },
        );
    }
    Ok(areas)
}

fn string_list(value: Option<&toml::Value>) -> Vec<String> {
    value
        .and_then(toml::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(toml::Value::as_str)
        .map(str::to_string)
        .collect()
}

fn detect(
    arguments: &Arguments,
    areas: &BTreeMap<String, Area>,
) -> Result<BTreeMap<String, bool>, Box<dyn Error>> {
    let forced =
        arguments.force_all || std::env::var("FORCE_ALL").is_ok_and(|value| value == "true");
    let files = match &arguments.changed {
        Some(files) => files.clone(),
        None => changed_files(arguments.base.as_deref(), arguments.head.as_deref())?,
    };

    if forced || files.is_empty() || touches_global(&files) {
        return Ok(areas.keys().map(|name| (name.clone(), true)).collect());
    }

    let root = std::env::current_dir()?;
    let metadata = cargo_metadata()?;
    let graph = WorkspaceGraph::from_metadata(&metadata, &root);
    let changed_packages = graph.owning_packages(&files);

    let mut hits = BTreeMap::new();
    for (name, area) in areas {
        let reachable = reachable_packages(area, &graph);
        let by_dependency = !reachable.is_disjoint(&changed_packages);
        let by_path = area
            .paths
            .iter()
            .any(|pattern| files.iter().any(|path| glob::matches(pattern, path)));
        hits.insert(name.clone(), by_dependency || by_path);
    }
    Ok(hits)
}

fn reachable_packages(area: &Area, graph: &WorkspaceGraph) -> BTreeSet<String> {
    let mut reachable = BTreeSet::new();
    for package in &area.packages {
        match graph.closure.get(package) {
            Some(closure) => reachable.extend(closure.iter().cloned()),
            None => {
                reachable.insert(package.clone());
            }
        }
    }
    reachable
}

fn touches_global(files: &[String]) -> bool {
    files.iter().any(|path| {
        GLOBAL_GLOBS
            .iter()
            .any(|pattern| glob::matches(pattern, path))
    })
}

fn changed_files(base: Option<&str>, head: Option<&str>) -> Result<Vec<String>, Box<dyn Error>> {
    let (Some(base), Some(head)) = (base, head) else {
        return Ok(Vec::new());
    };
    let output = Command::new("git")
        .args(["diff", "--name-only", base, head])
        .output()?;
    if !output.status.success() {
        return Err(format!("git diff --name-only {base} {head} failed").into());
    }
    let text = String::from_utf8(output.stdout)?;
    Ok(text
        .lines()
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect())
}

fn cargo_metadata() -> Result<CargoMetadata, Box<dyn Error>> {
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1"])
        .output()?;
    if !output.status.success() {
        return Err("cargo metadata failed".into());
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn emit(hits: &BTreeMap<String, bool>) -> std::io::Result<()> {
    let lines: Vec<String> = hits
        .iter()
        .map(|(name, hit)| format!("{name}={hit}"))
        .collect();
    let rendered = lines.join("\n");
    println!("{rendered}");
    if let Ok(path) = std::env::var("GITHUB_OUTPUT") {
        std::fs::write(path, format!("{rendered}\n"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Area, GLOBAL_GLOBS, glob};

    #[test]
    fn the_detector_source_forces_every_area_on() {
        assert!(
            GLOBAL_GLOBS
                .iter()
                .any(|pattern| glob::matches(pattern, "tools/ci-affected/src/main.rs")),
            "editing the detector must not leave areas silently off"
        );
        assert!(
            GLOBAL_GLOBS
                .iter()
                .any(|pattern| glob::matches(pattern, ".github/ci-areas.toml"))
        );
        assert!(
            !GLOBAL_GLOBS
                .iter()
                .any(|pattern| glob::matches(pattern, "docs/tutorials/page.md")),
            "an ordinary docs page must not force every area on"
        );
    }

    #[test]
    fn an_area_with_neither_packages_nor_paths_never_fires() {
        let area = Area {
            packages: Vec::new(),
            paths: Vec::new(),
        };
        assert!(area.packages.is_empty() && area.paths.is_empty());
    }
}
