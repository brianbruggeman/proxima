//! Prints the op coverage census of one `.onnx` file:
//!
//! ```sh
//! cargo run -p proxima-onnx --example onnx_census -- path/to/model.onnx
//! ```
//!
//! The first line is `census nodes=<n> distinct=<d> unlowerable=<u>`; one
//! `unlowerable op=<name> nodes=<count>` line follows per op type
//! [`proxima_onnx::lower_graph`] cannot lower. Built from
//! [`proxima_onnx::parse_complete`], [`proxima_onnx::census::census`] and
//! [`proxima_onnx::census::render_census`].

use std::env;
use std::fs;
use std::process::ExitCode;

use proxima_onnx::census::{census, render_census};
use proxima_onnx::parse_complete;

fn run(path: &str) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {path}: {error}"))?;
    let model = parse_complete(&bytes).map_err(|error| format!("parse {path}: {error}"))?;
    let graph = model
        .graph
        .ok_or_else(|| format!("{path} carries no graph"))?;
    Ok(render_census(&census(&graph)))
}

fn main() -> ExitCode {
    let Some(path) = env::args().nth(1) else {
        eprintln!("usage: onnx_census <model.onnx>");
        return ExitCode::from(2);
    };
    match run(&path) {
        Ok(rendered) => {
            print!("{rendered}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}
