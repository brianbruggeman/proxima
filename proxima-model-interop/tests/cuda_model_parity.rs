//! Model-level CUDA parity: greedy decode on the CUDA backend must produce the same token ids
//! as the CPU path on a real checkpoint. The CPU path is the oracle here, so this is a
//! determinism check between backends, not an external-oracle check (that is
//! `gemma4_correctness_gate.rs`). The expected ids are a fact about the checkpoint: `CpuOracle`
//! is the recorded upstream, so a plain run replays `tests/cassettes/*.jsonl` and decodes on CUDA
//! only; `PROXIMA_CASSETTE=record` reruns the CPU path and rewrites the cassette. Fixtures are
//! env-selected like `arch_data_baseline.rs`.

#![cfg(feature = "cuda")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs::File;
use std::future::Future;
use std::path::Path;

use memmap2::Mmap;
use proxima::request::{Request, Response};
use proxima::{Bytes, PipeHandle, ProximaError};
use proxima_gguf::parse_complete;
use proxima_model_interop::{GPU_LAYERS_ALL, LoadedModel, ServingConfig};
use proxima_primitives::pipe::SendPipe;
use serde::{Deserialize, Serialize};

const GEMMA4_E2B_ENV: &str = "PROXIMA_ARCH_GEMMA4_E2B_GGUF";
const GRANITE_MOE_ENV: &str = "PROXIMA_ARCH_GRANITE_MOE_GGUF";

const PROMPT: &str = "The capital of France is";
const TOKENS: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Greedy {
    text: String,
    ids: Vec<u32>,
}

#[derive(Clone)]
struct CpuOracle {
    name: &'static str,
    env: &'static str,
}

fn config(gpu_layers: i32) -> ServingConfig<'static> {
    ServingConfig {
        gpu_layers,
        ..ServingConfig::default()
    }
}

fn greedy_on(name: &str, env: &str, gpu_layers: i32) -> Result<Greedy, ProximaError> {
    let path = std::env::var(env).map_err(|_| {
        ProximaError::Record(format!(
            "{name} checkpoint env var {env} is unset: set it to the path of a {name} gguf"
        ))
    })?;
    if !Path::new(&path).exists() {
        return Err(ProximaError::Record(format!(
            "{name} checkpoint is missing at {path} (from {env}): stage the gguf there"
        )));
    }
    let fail = |step: &str, error: &dyn std::fmt::Display| {
        ProximaError::Record(format!("{name} {step} {path}: {error}"))
    };
    let file = File::open(&path).map_err(|error| fail("open", &error))?;
    // SAFETY: the checkpoint file is not written or truncated while this read-only mapping lives.
    let mapping = unsafe { Mmap::map(&file) }.map_err(|error| fail("mmap", &error))?;
    let bytes: &[u8] = &mapping;
    let parsed = parse_complete(bytes).map_err(|error| fail("parse", &error))?;
    let model = LoadedModel::load(&parsed, bytes).map_err(|error| fail("bind", &error))?;
    let (ids, text, _) = model
        .generate_with_serving_config(PROMPT, TOKENS, config(gpu_layers))
        .map_err(|error| fail("greedy decode", &error))?;
    Ok(Greedy { text, ids })
}

impl SendPipe for CpuOracle {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let oracle = self.clone();
        async move {
            let greedy = greedy_on(oracle.name, oracle.env, 0)?;
            let body = serde_json::to_vec(&greedy)
                .map_err(|error| ProximaError::Record(format!("encode cpu oracle: {error}")))?;
            Ok(Response::new(200).with_body(Bytes::from(body)))
        }
    }
}

async fn recorded_cpu_ids(name: &str, oracle: &PipeHandle) -> Greedy {
    let request = Request::builder()
        .method("GET")
        .path("/cpu_greedy")
        .query_param("model", name)
        .query_param("prompt", PROMPT)
        .query_param("n", TOKENS.to_string())
        .build()
        .expect("build the cpu oracle request");
    let response = SendPipe::call(oracle, request)
        .await
        .unwrap_or_else(|error| panic!("{name} cpu oracle: {error}"));
    let body = response
        .collect_body()
        .await
        .expect("collect the cpu oracle body");
    serde_json::from_slice(&body).expect("decode the cpu oracle greedy ids")
}

async fn assert_cuda_matches_cpu(name: &str, env: &str, oracle: &PipeHandle) {
    let cpu = recorded_cpu_ids(name, oracle).await;
    let cuda = greedy_on(name, env, GPU_LAYERS_ALL)
        .unwrap_or_else(|error| panic!("{name} cuda greedy decode failed: {error}"));

    println!(
        "{name} tokens={} cpu={:?} cuda={:?} cpu_text={:?} cuda_text={:?}",
        cpu.ids.len(),
        cpu.ids,
        cuda.ids,
        cpu.text,
        cuda.text
    );
    assert!(!cpu.ids.is_empty(), "{name}: cpu oracle produced no tokens");
    assert_eq!(
        cuda.ids, cpu.ids,
        "{name}: cuda greedy token ids diverge from the cpu path"
    );
}

#[proxima::test(cassette = "cuda_parity_gemma4_e2b")]
async fn cuda_greedy_decode_matches_cpu_on_gemma4_e2b(
    #[cassette(CpuOracle { name: "gemma4_e2b", env: GEMMA4_E2B_ENV })] oracle: PipeHandle,
) {
    assert_cuda_matches_cpu("gemma4_e2b", GEMMA4_E2B_ENV, &oracle).await;
}

#[proxima::test(cassette = "cuda_parity_granite_moe")]
async fn cuda_greedy_decode_matches_cpu_on_granite_moe(
    #[cassette(CpuOracle { name: "granite_moe", env: GRANITE_MOE_ENV })] oracle: PipeHandle,
) {
    assert_cuda_matches_cpu("granite_moe", GRANITE_MOE_ENV, &oracle).await;
}
