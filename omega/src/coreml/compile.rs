use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

use proxima_mil::writer::MilPackage;
use sha2::{Digest, Sha256};

const TOOL_NAME: &str = "coremlcompiler";
const COMPILED_MIL: &str = "model.mil";
const CACHE_RELATIVE_PATH: &str = ".cache/proxima/coreml-lane";

/// Why a package did not become a compiled model.
///
/// `Rejected` carries the compiler's own diagnostic verbatim, the text the
/// typed refusals of the Verify stage are parsed from.
#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    #[error("xcrun could not find {TOOL_NAME}: {message}")]
    ToolNotFound { message: String },

    #[error("could not start {program}: {source}")]
    Spawn {
        program: String,
        #[source]
        source: io::Error,
    },

    #[error("{operation} {path} failed: {source}")]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("{TOOL_NAME} exited with {status}: {message}")]
    Rejected { status: String, message: String },

    #[error("{TOOL_NAME} wrote no compiled model at {expected}")]
    NoCompiledModel { expected: PathBuf },

    #[error("HOME is not set, so the default compile cache directory is unknown")]
    NoHome,
}

/// A compiled model on disk: the `.mlmodelc` directory the Core ML framework
/// loads, named by the sha256 of the package it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Compiled {
    /// The `<sha256>.mlmodelc` directory inside the cache.
    pub directory: PathBuf,
    /// [`package_sha256`] of the package this was compiled from.
    pub sha256: String,
    /// True when the cache already held this package and no process ran.
    pub cache_hit: bool,
}

impl Compiled {
    /// The text the compiler wrote to `model.mil`: the program it kept.
    ///
    /// # Errors
    ///
    /// [`CompileError::Io`] when the file cannot be read.
    pub fn model_mil(&self) -> Result<String, CompileError> {
        let path = self.directory.join(COMPILED_MIL);
        fs::read_to_string(&path).map_err(|source| CompileError::Io {
            operation: "read",
            path,
            source,
        })
    }
}

/// The edge that runs `xcrun coremlcompiler compile` and caches by content.
///
/// It consumes the [`MilPackage`] that [`proxima_mil::writer::MilWriter`]
/// produced and is what the door's Verify stage calls once the `coreml-lane`
/// feature is on. The cache key is [`package_sha256`], so a changed program
/// recompiles and an unchanged one spawns nothing; compose it directly when
/// you hold a package and want the `.mlmodelc`, or through the door when you
/// hold a model file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Compiler {
    tool: PathBuf,
    cache: PathBuf,
}

impl Compiler {
    /// Locates the compiler with `xcrun --find coremlcompiler`.
    ///
    /// # Errors
    ///
    /// [`CompileError::ToolNotFound`] when xcrun cannot find it,
    /// [`CompileError::Spawn`] when xcrun does not start.
    pub fn locate(cache: impl Into<PathBuf>) -> Result<Self, CompileError> {
        Ok(Self::with_tool(locate_tool()?, cache))
    }

    /// A compiler at a known path; nothing is spawned until a cache miss.
    #[must_use]
    pub fn with_tool(tool: impl Into<PathBuf>, cache: impl Into<PathBuf>) -> Self {
        Self {
            tool: tool.into(),
            cache: cache.into(),
        }
    }

    /// Compiles `package`, or returns the cached `.mlmodelc` for its sha256.
    ///
    /// # Errors
    ///
    /// [`CompileError::Rejected`] with the compiler's message when it refuses
    /// the package; [`CompileError::Io`] and [`CompileError::Spawn`] for the
    /// filesystem and process edges.
    pub fn compile(&self, package: &MilPackage) -> Result<Compiled, CompileError> {
        let sha256 = package_sha256(package);
        let directory = self.cache.join(format!("{sha256}.mlmodelc"));
        if directory.join(COMPILED_MIL).is_file() {
            return Ok(Compiled {
                directory,
                sha256,
                cache_hit: true,
            });
        }
        let staging = self
            .cache
            .join(format!(".staging-{sha256}-{}", std::process::id()));
        let built = self.build_in(&staging, package, &sha256, &directory);
        let cleaned = remove_tree(&staging);
        built.and(cleaned).map(|()| Compiled {
            directory,
            sha256,
            cache_hit: false,
        })
    }

    fn build_in(
        &self,
        staging: &Path,
        package: &MilPackage,
        sha256: &str,
        directory: &Path,
    ) -> Result<(), CompileError> {
        let package_root = staging.join(format!("{sha256}.mlpackage"));
        let output = staging.join("out");
        write_package(&package_root, package)?;
        create_dir(&output)?;
        run_compiler(&self.tool, &package_root, &output)?;
        let produced = output.join(format!("{sha256}.mlmodelc"));
        if !produced.join(COMPILED_MIL).is_file() {
            return Err(CompileError::NoCompiledModel { expected: produced });
        }
        fs::rename(&produced, directory).map_err(|source| CompileError::Io {
            operation: "rename into the cache",
            path: directory.to_owned(),
            source,
        })
    }
}

/// `$HOME/.cache/proxima/coreml-lane`, where compiled models are kept unless
/// the door's `model_cache_dir` names another directory.
///
/// # Errors
///
/// [`CompileError::NoHome`] when `HOME` is not set.
pub fn default_cache_dir() -> Result<PathBuf, CompileError> {
    let home = std::env::var_os("HOME").ok_or(CompileError::NoHome)?;
    Ok(PathBuf::from(home).join(CACHE_RELATIVE_PATH))
}

/// The sha256, lowercase hex, over the (relative path, bytes) pairs of the
/// package's files sorted by path.
#[must_use]
pub fn package_sha256(package: &MilPackage) -> String {
    let mut files = package.files().to_vec();
    files.sort_by_key(|(path, _)| *path);
    let mut digest = Sha256::new();
    for (path, bytes) in files {
        digest.update(path.as_bytes());
        digest.update(bytes);
    }
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// How many operations of each type a compiled `model.mil` holds, constants
/// included: every statement is `<type> <name> = <operator>(...)`.
#[must_use]
pub fn operator_counts(model_mil: &str) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for operator in model_mil.lines().filter_map(statement_operator) {
        *counts.entry(operator.to_owned()).or_insert(0) += 1;
    }
    counts
}

fn statement_operator(line: &str) -> Option<&str> {
    let (_, call) = line.split_once(" = ")?;
    let (name, _) = call.split_once('(')?;
    let valid = !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_');
    valid.then_some(name)
}

fn locate_tool() -> Result<PathBuf, CompileError> {
    let output = Command::new("xcrun")
        .args(["--find", TOOL_NAME])
        .output()
        .map_err(|source| CompileError::Spawn {
            program: "xcrun".to_owned(),
            source,
        })?;
    match output.status.success() {
        true => Ok(PathBuf::from(
            String::from_utf8_lossy(&output.stdout).trim(),
        )),
        false => Err(CompileError::ToolNotFound {
            message: diagnostic(&output),
        }),
    }
}

fn run_compiler(tool: &Path, package: &Path, output_dir: &Path) -> Result<(), CompileError> {
    let output = Command::new(tool)
        .arg("compile")
        .arg(package)
        .arg(output_dir)
        .output()
        .map_err(|source| CompileError::Spawn {
            program: tool.display().to_string(),
            source,
        })?;
    match output.status.success() {
        true => Ok(()),
        false => Err(CompileError::Rejected {
            status: output.status.to_string(),
            message: diagnostic(&output),
        }),
    }
}

fn diagnostic(output: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    format!("{} {}", stderr.trim(), stdout.trim())
        .trim()
        .to_owned()
}

fn write_package(root: &Path, package: &MilPackage) -> Result<(), CompileError> {
    package.files().iter().try_for_each(|(relative, bytes)| {
        let path = root.join(relative);
        if let Some(parent) = path.parent() {
            create_dir(parent)?;
        }
        fs::write(&path, bytes).map_err(|source| CompileError::Io {
            operation: "write",
            path,
            source,
        })
    })
}

fn create_dir(path: &Path) -> Result<(), CompileError> {
    fs::create_dir_all(path).map_err(|source| CompileError::Io {
        operation: "create",
        path: path.to_owned(),
        source,
    })
}

fn remove_tree(path: &Path) -> Result<(), CompileError> {
    match fs::remove_dir_all(path) {
        Err(source) if source.kind() != io::ErrorKind::NotFound => Err(CompileError::Io {
            operation: "remove",
            path: path.to_owned(),
            source,
        }),
        _ => Ok(()),
    }
}
