use std::ffi::OsStr;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::thread;

use thiserror::Error;

pub const PINNED_VERSION: &str = "21.1.8";
pub const DEFAULT_BIN: &str = "/opt/homebrew/opt/llvm/bin";
pub const RUNNER_LIBRARIES: [&str; 2] =
    ["libmlir_runner_utils.dylib", "libmlir_c_runner_utils.dylib"];
pub const LOWERING_PASSES: [&str; 8] = [
    "--one-shot-bufferize=bufferize-function-boundaries",
    "--convert-linalg-to-loops",
    "--convert-scf-to-cf",
    "--expand-strided-metadata",
    "--finalize-memref-to-llvm",
    "--convert-func-to-llvm",
    "--convert-arith-to-llvm",
    "--convert-cf-to-llvm",
];
pub const RECONCILE_PASS: &str = "--reconcile-unrealized-casts";

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("{tool} is not a file at {path}")]
    Missing { tool: &'static str, path: PathBuf },
    #[error("{tool} at {path} could not run: {source}")]
    Spawn {
        tool: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    #[error("{tool} at {path} printed no llvm version line: {output}")]
    NoVersion {
        tool: &'static str,
        path: PathBuf,
        output: String,
    },
    #[error("{tool} at {path} reports llvm {found}, pinned {expected}")]
    WrongVersion {
        tool: &'static str,
        path: PathBuf,
        found: String,
        expected: &'static str,
    },
    #[error("{tool} exited with {status}: {stderr}")]
    Failed {
        tool: &'static str,
        status: String,
        stderr: String,
    },
    #[error("{tool} did not take its input: {source}")]
    Stdin {
        tool: &'static str,
        source: io::Error,
    },
    #[error("runner library {name} is not a file at {path}")]
    MissingLibrary { name: &'static str, path: PathBuf },
    #[error("no printed memref data in runner output: {output}")]
    NoMemref { output: String },
    #[error("printed memref token {token} is not a number")]
    BadNumber { token: String },
}

#[derive(Debug, Clone)]
pub struct Tool {
    name: &'static str,
    path: PathBuf,
}

impl Tool {
    pub fn at(name: &'static str, path: impl Into<PathBuf>) -> Self {
        Self {
            name,
            path: path.into(),
        }
    }

    pub fn resolve(name: &'static str, override_variable: &str) -> Self {
        let path = std::env::var_os(override_variable)
            .map_or_else(|| bin_directory().join(name), PathBuf::from);
        Self::at(name, path)
    }

    pub fn pinned(self) -> Result<Self, ToolError> {
        if !self.path.is_file() {
            return Err(ToolError::Missing {
                tool: self.name,
                path: self.path,
            });
        }
        let output = self.run(["--version"], "")?;
        check_version(self.name, &self.path, &output)?;
        Ok(self)
    }

    pub fn run<I, S>(&self, args: I, input: &str) -> Result<String, ToolError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let spawn_error = |source| ToolError::Spawn {
            tool: self.name,
            path: self.path.clone(),
            source,
        };
        let child = Command::new(&self.path)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(spawn_error)?;
        let (written, output) = communicate(child, input);
        self.finish(written, &output.map_err(spawn_error)?)
    }

    fn finish(&self, written: Written, output: &Output) -> Result<String, ToolError> {
        if !output.status.success() {
            return Err(ToolError::Failed {
                tool: self.name,
                status: output.status.to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }
        if let Ok(Err(source)) = written {
            return Err(ToolError::Stdin {
                tool: self.name,
                source,
            });
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }

    pub fn runner_libraries(&self) -> Result<Vec<PathBuf>, ToolError> {
        let library_directory = self
            .path
            .parent()
            .and_then(Path::parent)
            .map_or_else(|| PathBuf::from("lib"), |prefix| prefix.join("lib"));
        RUNNER_LIBRARIES
            .iter()
            .map(|name| library_path(&library_directory, name))
            .collect()
    }
}

type Written = thread::Result<io::Result<()>>;

fn communicate(mut child: Child, input: &str) -> (Written, io::Result<Output>) {
    let pipe = child.stdin.take();
    thread::scope(|scope| {
        let feeder =
            scope.spawn(move || pipe.map_or(Ok(()), |mut open| open.write_all(input.as_bytes())));
        let output = child.wait_with_output();
        (feeder.join(), output)
    })
}

fn library_path(directory: &Path, name: &'static str) -> Result<PathBuf, ToolError> {
    let path = directory.join(name);
    if path.is_file() {
        Ok(path)
    } else {
        Err(ToolError::MissingLibrary { name, path })
    }
}

fn bin_directory() -> PathBuf {
    std::env::var_os("MLIR_BIN").map_or_else(|| PathBuf::from(DEFAULT_BIN), PathBuf::from)
}

pub fn parse_version(output: &str) -> Option<&str> {
    output
        .lines()
        .find_map(|line| line.split_once("LLVM version "))
        .and_then(|(_, rest)| rest.split_whitespace().next())
}

pub fn check_version(tool: &'static str, path: &Path, output: &str) -> Result<(), ToolError> {
    match parse_version(output) {
        None => Err(ToolError::NoVersion {
            tool,
            path: path.to_path_buf(),
            output: output.to_owned(),
        }),
        Some(found) if found != PINNED_VERSION => Err(ToolError::WrongVersion {
            tool,
            path: path.to_path_buf(),
            found: found.to_owned(),
            expected: PINNED_VERSION,
        }),
        Some(_) => Ok(()),
    }
}

pub fn mlir_opt() -> Result<Tool, ToolError> {
    Tool::resolve("mlir-opt", "MLIR_OPT").pinned()
}

pub fn mlir_runner() -> Result<Tool, ToolError> {
    Tool::resolve("mlir-runner", "MLIR_RUNNER").pinned()
}

pub fn lower_to_llvm(opt: &Tool, source: &str) -> Result<String, ToolError> {
    let arguments = LOWERING_PASSES.iter().chain([&RECONCILE_PASS]);
    opt.run(arguments, source)
}

pub fn run_main(runner: &Tool, llvm_text: &str) -> Result<String, ToolError> {
    let libraries = runner.runner_libraries()?;
    let joined = libraries
        .iter()
        .map(|library| library.to_string_lossy())
        .collect::<Vec<_>>()
        .join(",");
    let arguments = [
        "-e".to_owned(),
        "main".to_owned(),
        "-entry-point-result=void".to_owned(),
        format!("-shared-libs={joined}"),
    ];
    runner.run(arguments, llvm_text)
}

pub fn parse_printed_memref(output: &str) -> Result<Vec<f64>, ToolError> {
    let (_, rest) = output
        .split_once("data =")
        .ok_or_else(|| ToolError::NoMemref {
            output: output.to_owned(),
        })?;
    rest.split(|byte: char| matches!(byte, '[' | ']' | ',') || byte.is_whitespace())
        .filter(|token| !token.is_empty())
        .map(|token| {
            token.parse::<f64>().map_err(|_| ToolError::BadNumber {
                token: token.to_owned(),
            })
        })
        .collect()
}
