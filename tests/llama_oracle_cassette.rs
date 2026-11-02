//! The llama.cpp incumbent as a recorded oracle. An external process is IO,
//! IO is a pipe: `LlamaOracle` is a `Request -> Response` handler whose call
//! spawns `llama-tokenize` / `llama-completion`. `#[cassette(..)]` records its
//! traffic once; every later run replays it with the binaries absent.
//!
//! Record on a host that has the incumbent:
//! `PROXIMA_CASSETTE=record cargo val -j 1 -p proxima --test llama_oracle_cassette`

#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::future::Future;
use std::path::PathBuf;
use std::process::Command;

use bytes::Bytes;
use serde::{Deserialize, Serialize};

use proxima::request::{Request, Response};
use proxima::{PipeHandle, ProximaError};
use proxima_primitives::pipe::SendPipe;

const INCUMBENT_BIN_DIR: &str = "/home/bix/repos/llama.cpp-incumbent/build/bin";
const CUDA_RUNTIME_DIR: &str = "/usr/local/lib/ollama.stale-0.5.7.bak";
const MODEL: &str = "/home/bix/models/qwen3-0.6b-q8_0.gguf";
const ABSENT_BIN_DIR: &str = "/nonexistent/llama-bin";
const PROMPT: &str = "The capital of France is";
const GREEDY_TOKENS: usize = 6;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct TokenIds {
    ids: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Greedy {
    text: String,
    ids: Vec<u32>,
}

#[derive(Clone)]
struct LlamaOracle {
    bin_dir: PathBuf,
}

impl LlamaOracle {
    fn incumbent() -> Self {
        Self {
            bin_dir: PathBuf::from(INCUMBENT_BIN_DIR),
        }
    }

    fn absent() -> Self {
        Self {
            bin_dir: PathBuf::from(ABSENT_BIN_DIR),
        }
    }

    fn run(&self, binary: &str, args: &[&str]) -> Result<String, ProximaError> {
        let program = self.bin_dir.join(binary);
        let library_path = format!("{}:{CUDA_RUNTIME_DIR}", self.bin_dir.display());
        let output = Command::new(&program)
            .args(args)
            .env("LD_LIBRARY_PATH", library_path)
            .stdin(std::process::Stdio::null())
            .output()
            .map_err(|error| {
                ProximaError::Record(format!("spawn {}: {error}", program.display()))
            })?;
        if !output.status.success() {
            return Err(ProximaError::Record(format!(
                "{} exited {}: {}",
                program.display(),
                output.status,
                String::from_utf8_lossy(&output.stderr)
                    .lines()
                    .last()
                    .unwrap_or_default()
            )));
        }
        String::from_utf8(output.stdout)
            .map_err(|error| ProximaError::Record(format!("{binary} stdout is not utf-8: {error}")))
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>, ProximaError> {
        let stdout = self.run(
            "llama-tokenize",
            &[
                "-m",
                MODEL,
                "--ids",
                "--no-bos",
                "--log-disable",
                "-p",
                text,
            ],
        )?;
        serde_json::from_str(stdout.trim())
            .map_err(|error| ProximaError::Record(format!("parse ids `{stdout}`: {error}")))
    }

    fn greedy(&self, prompt: &str, count: usize) -> Result<Greedy, ProximaError> {
        let count_arg = count.to_string();
        let stdout = self.run(
            "llama-completion",
            &[
                "-m", MODEL, "-p", prompt, "-n", &count_arg, "--temp", "0", "-ngl", "0", "-no-cnv",
                "-s", "1",
            ],
        )?;
        let generated = stdout
            .strip_prefix(prompt)
            .ok_or_else(|| {
                ProximaError::Record(format!("stdout lacks the prompt echo: {stdout:?}"))
            })?
            .trim_end_matches('\n')
            .to_string();
        let ids = self.tokenize(&generated)?;
        if ids.len() != count {
            return Err(ProximaError::Record(format!(
                "generated text {generated:?} retokenizes to {} ids, expected {count}",
                ids.len()
            )));
        }
        Ok(Greedy {
            text: generated,
            ids,
        })
    }

    fn answer(&self, request: &Request<Bytes>) -> Result<Vec<u8>, ProximaError> {
        let prompt = request
            .query
            .get_str("prompt")
            .ok_or_else(|| ProximaError::Record("request lacks a `prompt` query".into()))?
            .to_string();
        match &request.path[..] {
            b"/tokenize" => encode(&TokenIds {
                ids: self.tokenize(&prompt)?,
            }),
            b"/greedy" => {
                let count = request
                    .query
                    .get_str("n")
                    .and_then(|raw| raw.parse().ok())
                    .ok_or_else(|| {
                        ProximaError::Record("request lacks a numeric `n` query".into())
                    })?;
                encode(&self.greedy(&prompt, count)?)
            }
            other => Err(ProximaError::Record(format!(
                "unknown oracle route {}",
                String::from_utf8_lossy(other)
            ))),
        }
    }
}

fn encode<Value: Serialize>(value: &Value) -> Result<Vec<u8>, ProximaError> {
    serde_json::to_vec(value)
        .map_err(|error| ProximaError::Record(format!("encode oracle response: {error}")))
}

impl SendPipe for LlamaOracle {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        let oracle = self.clone();
        async move {
            let body = oracle.answer(&request)?;
            Ok(Response::new(200).with_body(Bytes::from(body)))
        }
    }
}

async fn ask<Answer: for<'de> Deserialize<'de>>(
    pipe: &PipeHandle,
    path: &str,
    count: Option<usize>,
) -> Result<Answer, ProximaError> {
    let mut builder = Request::builder()
        .method("GET")
        .path(path)
        .query_param("prompt", PROMPT);
    if let Some(count) = count {
        builder = builder.query_param("n", count.to_string());
    }
    let response = SendPipe::call(pipe, builder.build()?).await?;
    let body = response.collect_body().await?;
    serde_json::from_slice(&body)
        .map_err(|error| ProximaError::Record(format!("decode oracle response: {error}")))
}

fn france_ids() -> TokenIds {
    TokenIds {
        ids: vec![785, 6722, 315, 9625, 374],
    }
}

#[proxima::test(cassette = "llama_tokenize")]
async fn tokenize_matches_the_incumbent(#[cassette(LlamaOracle::incumbent())] pipe: PipeHandle) {
    let answer: TokenIds = ask(&pipe, "/tokenize", None).await.expect("tokenize");

    assert_eq!(answer, france_ids());
}

#[proxima::test(cassette = "llama_tokenize")]
async fn tokenize_replays_with_the_binary_absent(
    #[cassette(LlamaOracle::absent())] pipe: PipeHandle,
) {
    let answer: TokenIds = ask(&pipe, "/tokenize", None).await.expect("tokenize");

    assert_eq!(answer, france_ids());
}

#[proxima::test(cassette = "llama_greedy")]
async fn greedy_ids_match_the_incumbent(#[cassette(LlamaOracle::incumbent())] pipe: PipeHandle) {
    let answer: Greedy = ask(&pipe, "/greedy", Some(GREEDY_TOKENS))
        .await
        .expect("greedy");

    assert_eq!(answer.text, " Paris. The capital of France");
    assert_eq!(answer.ids.len(), GREEDY_TOKENS);
}

#[proxima::test(cassette = "llama_greedy")]
async fn greedy_replays_with_the_binary_absent(
    #[cassette(LlamaOracle::absent())] pipe: PipeHandle,
) {
    let answer: Greedy = ask(&pipe, "/greedy", Some(GREEDY_TOKENS))
        .await
        .expect("greedy");

    assert_eq!(answer.text, " Paris. The capital of France");
    assert_eq!(answer.ids.len(), GREEDY_TOKENS);
}

#[proxima::test(cassette = "llama_tokenize")]
async fn an_unrecorded_request_is_a_replay_miss(
    #[cassette(LlamaOracle::absent())] pipe: PipeHandle,
) {
    let request = Request::builder()
        .method("GET")
        .path("/tokenize")
        .query_param("prompt", "a prompt nobody recorded")
        .build()
        .expect("request");

    let outcome = SendPipe::call(&pipe, request).await;

    assert!(
        matches!(outcome, Err(ProximaError::ReplayMiss { .. })),
        "{outcome:?}"
    );
}
