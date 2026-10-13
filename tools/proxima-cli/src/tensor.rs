//! `proxima tensor run` — post a tensor plan and its inputs to a running
//! proximad's `/tensor/run` route and print or save the evaluated outputs.
//! Reuses the UDS transport from `pipeline` and the safetensors codec from
//! `tensor_wire`; the server half is `tensor_run`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Subcommand};
use proxima::ProximaError;
use proxima_cli::tensor_wire::{TensorRecord, decode_response, encode_request};

use crate::pipeline::PipelineTransport;

const TENSOR_RUN_PATH: &str = "/tensor/run";
const SAFETENSORS_CONTENT_TYPE: &str = "application/x-safetensors";

#[derive(Args, Debug)]
pub struct TensorArgs {
    #[command(subcommand)]
    action: TensorAction,
}

#[derive(Subcommand, Debug)]
pub enum TensorAction {
    /// Evaluate a tensor plan on a running `proximad` over its UDS.
    /// Prints `<name> <shape> <n> floats` for each output.
    Run(RunArgs),
}

#[derive(Args, Debug)]
pub struct RunArgs {
    /// Path to the running proximad's UDS.
    #[arg(long, value_name = "PATH", env = "PROXIMAD_SOCKET")]
    socket: PathBuf,

    /// Tensor plan (TOML `ProgramSpec`), e.g. `matmul_silu_softmax.toml`.
    plan: PathBuf,

    /// Safetensors file holding one F32 tensor per `[[bind]]` name.
    #[arg(long, value_name = "PATH")]
    inputs: Option<PathBuf>,

    /// Write the evaluated outputs as a safetensors file.
    #[arg(long, value_name = "PATH")]
    output: Option<PathBuf>,
}

pub async fn run(args: TensorArgs) -> ExitCode {
    let outcome = match args.action {
        TensorAction::Run(run_args) => run_plan(run_args).await,
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("proxima tensor run: {error}");
            ExitCode::from(1)
        }
    }
}

async fn run_plan(args: RunArgs) -> Result<(), ProximaError> {
    let plan = read_text(&args.plan).await?;
    let inputs = match &args.inputs {
        Some(path) => decode_response(&read_bytes(path).await?)?,
        None => Vec::new(),
    };
    let body = encode_request(&plan, &borrow_inputs(&inputs))?;
    let transport = PipelineTransport::LocalUds(args.socket);
    let response = transport
        .connect()
        .await?
        .http_call(
            "POST",
            TENSOR_RUN_PATH,
            Some((SAFETENSORS_CONTENT_TYPE, body)),
        )
        .await?;
    if response.status != 200 {
        return Err(ProximaError::Upstream(format!(
            "tensor run failed: HTTP {} — {}",
            response.status,
            String::from_utf8_lossy(&response.body)
        )));
    }
    let outputs = decode_response(&response.body)?;
    print_summary(&outputs);
    if let Some(path) = &args.output {
        tokio::fs::write(path, &response.body)
            .await
            .map_err(|err| io_error(format!("write output {path:?}: {err}")))?;
    }
    Ok(())
}

fn borrow_inputs(inputs: &[TensorRecord]) -> Vec<(&str, &[u64], &[f32])> {
    inputs
        .iter()
        .map(|(name, shape, values)| (name.as_str(), shape.as_slice(), values.as_slice()))
        .collect()
}

fn print_summary(outputs: &[TensorRecord]) {
    for (name, shape, values) in outputs {
        println!("{name} {shape:?} {} floats", values.len());
    }
}

async fn read_text(path: &Path) -> Result<String, ProximaError> {
    tokio::fs::read_to_string(path)
        .await
        .map_err(|err| io_error(format!("read plan {path:?}: {err}")))
}

async fn read_bytes(path: &Path) -> Result<Vec<u8>, ProximaError> {
    tokio::fs::read(path)
        .await
        .map_err(|err| io_error(format!("read inputs {path:?}: {err}")))
}

fn io_error(message: String) -> ProximaError {
    ProximaError::Io(std::io::Error::other(message))
}
