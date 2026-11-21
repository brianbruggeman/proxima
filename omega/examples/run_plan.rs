//! Run a `ProgramSpec` plan on omega, synthesising every bound input, so a
//! plan can be exercised from a shell with no data files.
//!
//! ```text
//! cargo run -p proxima-omega --example run_plan -- \
//!     proxima-tensor/src/spec/fixtures/matmul_silu_softmax.toml --bind x=2x4 --bind w=4x3
//! ```

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use omega::backend::{execute_plan_named, plan_named_with_config};
use omega::config::RuntimeConfig;
use proxima_tensor::QuantizedBlock;
use proxima_tensor::spec::ProgramSpec;

const DEFAULT_RUNTIME: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/examples/omega_hello_tensor.toml"
);
const PRINTED_VALUES: usize = 8;

struct Request {
    plan: String,
    runtime: String,
    bindings: Vec<(String, Vec<u64>)>,
}

fn main() -> Result<()> {
    let request = parse_args()?;
    let text = fs::read_to_string(&request.plan)
        .with_context(|| format!("read plan {}", request.plan))?;
    let spec: ProgramSpec = toml::from_str(&text)
        .with_context(|| format!("{} is not a tensor program", request.plan))?;
    let config = RuntimeConfig::from_file(&request.runtime)
        .with_context(|| format!("load runtime config {}", request.runtime))?;

    let supplied: Vec<(String, Vec<f32>)> = request
        .bindings
        .iter()
        .map(|(name, shape)| (name.clone(), synthesise(shape)))
        .collect();
    let inputs: Vec<(&str, &[f32])> = supplied
        .iter()
        .map(|(name, data)| (name.as_str(), data.as_slice()))
        .collect();

    let (program, bound, outputs) = spec.plan_parts(&inputs)?;
    let named: Vec<(&str, QuantizedBlock<'_>)> = bound
        .iter()
        .map(|(name, data)| (*name, QuantizedBlock::Float32(data)))
        .collect();
    let (mut plan, report) = plan_named_with_config(&config, &program, &[], &named, &outputs)?;
    let evaluated = execute_plan_named(&mut plan, &named)?;

    println!("backend = {}", report[0].device_identity);
    for (name, shape, values) in spec.collect_outputs(&evaluated)? {
        println!("{name} {shape:?} {}", summarise(&values));
    }
    Ok(())
}

fn parse_args() -> Result<Request> {
    let mut request = Request {
        plan: String::new(),
        runtime: DEFAULT_RUNTIME.to_string(),
        bindings: Vec::new(),
    };
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--runtime" => {
                request.runtime = args.next().context("--runtime needs a path")?;
            }
            "--bind" => {
                let binding = args.next().context("--bind needs name=rowsxcols")?;
                let (name, shape) = binding
                    .split_once('=')
                    .with_context(|| format!("{binding} is not name=rowsxcols"))?;
                let extent = shape
                    .split('x')
                    .map(|dim| dim.parse::<u64>())
                    .collect::<Result<Vec<u64>, _>>()
                    .with_context(|| format!("{shape} is not rowsxcols"))?;
                request.bindings.push((name.to_string(), extent));
            }
            flag if flag.starts_with("--") => bail!("unknown flag {flag}"),
            path => request.plan = path.to_string(),
        }
    }
    if request.plan.is_empty() {
        bail!("usage: run_plan <plan.toml> [--runtime <toml>] [--bind name=rowsxcols]...");
    }
    if !Path::new(&request.plan).is_file() {
        bail!("no such plan: {}", request.plan);
    }
    Ok(request)
}

/// Deterministic values in `[-0.5, 0.5)`: a plan's numbers are reproducible
/// from its shape alone, so a doc can paste the output it actually printed.
fn synthesise(shape: &[u64]) -> Vec<f32> {
    let count = shape.iter().product::<u64>();
    (0..count)
        .map(|index| ((index * 37) % 97) as f32 / 97.0 - 0.5)
        .collect()
}

fn summarise(values: &[f32]) -> String {
    let shown = values.len().min(PRINTED_VALUES);
    let head = values[..shown]
        .iter()
        .map(|value| format!("{value:.4}"))
        .collect::<Vec<_>>()
        .join(", ");
    if shown == values.len() {
        format!("[{head}]")
    } else {
        format!("[{head}, ... {} values]", values.len())
    }
}
