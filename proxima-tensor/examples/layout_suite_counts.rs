use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

const SUITES: [(&str, [&str; 2]); 4] = [
    ("proxima-tensor", ["--features", "float-layout-dtype"]),
    (
        "proxima-autograd",
        ["--features", "proxima-tensor/float-layout-dtype"],
    ),
    (
        "proxima-onnx",
        ["--features", "proxima-tensor/float-layout-dtype"],
    ),
    (
        "proxima-model-interop",
        ["--features", "std,float-layout-dtype"],
    ),
];

#[derive(Debug, thiserror::Error)]
enum SuiteError {
    #[error("usage: layout_suite_counts --baseline-dir <dir> [--write-baseline]")]
    Usage,
    #[error("cannot run cargo for {crate_name}")]
    Spawn {
        crate_name: &'static str,
        #[source]
        source: io::Error,
    },
    #[error("{crate_name} printed no nextest summary line; last lines:\n{tail}")]
    NoSummary {
        crate_name: &'static str,
        tail: String,
    },
    #[error("{crate_name} ran {run} tests and passed {passed}: the suite must pass whole")]
    Failing {
        crate_name: &'static str,
        run: usize,
        passed: usize,
    },
    #[error("cannot read or write {path}")]
    Count {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("{path} does not hold a test count")]
    BadCount { path: PathBuf },
}

fn flag_value(arguments: &[String], flag: &str) -> Option<String> {
    arguments
        .iter()
        .position(|argument| argument == flag)
        .and_then(|index| arguments.get(index + 1))
        .cloned()
}

fn summary_counts(output: &str) -> Option<(usize, usize)> {
    let line = output
        .lines()
        .rev()
        .find(|line| line.contains(" tests run: "))?;
    let after = line.split(" tests run: ").collect::<Vec<_>>();
    let run = after.first()?.split_whitespace().last()?.parse().ok()?;
    let passed = after.get(1)?.split(" passed").next()?.trim().parse().ok()?;
    Some((run, passed))
}

fn passed_count(crate_name: &'static str, features: &[&str; 2]) -> Result<usize, SuiteError> {
    let output = Command::new("cargo")
        .args([
            "nextest",
            "run",
            "-p",
            crate_name,
            "--lib",
            "--no-fail-fast",
        ])
        .args(features)
        .output()
        .map_err(|source| SuiteError::Spawn { crate_name, source })?;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let (run, passed) = summary_counts(&text).ok_or_else(|| SuiteError::NoSummary {
        crate_name,
        tail: text.lines().rev().take(12).collect::<Vec<_>>().join("\n"),
    })?;
    if run != passed || passed == 0 {
        return Err(SuiteError::Failing {
            crate_name,
            run,
            passed,
        });
    }
    Ok(passed)
}

fn baseline_path(directory: &Path, crate_name: &str) -> PathBuf {
    directory.join(format!("{crate_name}.count"))
}

fn read_baseline(directory: &Path, crate_name: &str) -> Result<usize, SuiteError> {
    let path = baseline_path(directory, crate_name);
    fs::read_to_string(&path)
        .map_err(|source| SuiteError::Count {
            path: path.clone(),
            source,
        })?
        .trim()
        .parse()
        .map_err(|_| SuiteError::BadCount { path })
}

fn write_baseline(directory: &Path, crate_name: &str, passed: usize) -> Result<(), SuiteError> {
    fs::create_dir_all(directory).map_err(|source| SuiteError::Count {
        path: directory.to_path_buf(),
        source,
    })?;
    let path = baseline_path(directory, crate_name);
    fs::write(&path, format!("{passed}\n")).map_err(|source| SuiteError::Count { path, source })
}

fn run(arguments: &[String]) -> Result<(), SuiteError> {
    let directory = flag_value(arguments, "--baseline-dir").ok_or(SuiteError::Usage)?;
    let directory = Path::new(&directory);
    let writing = arguments
        .iter()
        .any(|argument| argument == "--write-baseline");
    for (crate_name, features) in &SUITES {
        let passed = passed_count(crate_name, features)?;
        if writing {
            write_baseline(directory, crate_name, passed)?;
        }
        let baseline = read_baseline(directory, crate_name)?;
        println!("suite crate={crate_name} passed={passed} baseline={baseline}");
    }
    Ok(())
}

fn main() -> Result<(), SuiteError> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    run(&arguments)
}
