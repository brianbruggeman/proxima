//! The llama.cpp incumbent as a recorded oracle. An external process is IO,
//! IO is a pipe: `LlamaOracle` is a `Request -> Response` handler whose call
//! runs `llama-tokenize` and a one-shot `llama-server` `/completion` with
//! `return_tokens`, so greedy ids are the generator's own. `#[cassette(..)]`
//! records its traffic once; every later run replays it with no binaries.
//!
//! Record with a CPU build (cmake -DGGML_CUDA=OFF -DBUILD_SHARED_LIBS=OFF,
//! targets llama-tokenize llama-server):
//! `PROXIMA_LLAMA_BIN_DIR=<build>/bin PROXIMA_CASSETTE=record cargo val -p proxima-lib --features test-support --test llama_oracle_cassette`

#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::future::Future;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use bytes::Bytes;
use serde::{Deserialize, Serialize};

use proxima::request::{Request, Response};
use proxima::{PipeHandle, ProximaError};
use proxima_primitives::pipe::SendPipe;

const BIN_DIR_VAR: &str = "PROXIMA_LLAMA_BIN_DIR";
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
    bin_dir: Option<PathBuf>,
}

fn free_port() -> Result<u16, ProximaError> {
    TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .map(|address| address.port())
        .map_err(|error| ProximaError::Record(format!("reserve a port: {error}")))
}

fn wait_listening(server: &mut Child) -> Result<(), ProximaError> {
    let stderr = server
        .stderr
        .take()
        .ok_or_else(|| ProximaError::Record("llama-server stderr was not piped".into()))?;
    let mut last = String::new();
    for line in BufReader::new(stderr).lines() {
        last = line.map_err(|error| ProximaError::Record(format!("read llama-server: {error}")))?;
        if last.contains("listening on") {
            return Ok(());
        }
    }
    Err(ProximaError::Record(format!(
        "llama-server exited before listening; last line: {last}"
    )))
}

fn complete(port: u16, prompt: &str, count: usize) -> Result<Greedy, ProximaError> {
    let payload = serde_json::json!({
        "prompt": prompt,
        "n_predict": count,
        "temperature": 0,
        "seed": 1,
        "return_tokens": true,
        "cache_prompt": false,
    })
    .to_string();
    let reply = post(port, "/completion", &payload)?;
    let parsed: ServerCompletion = serde_json::from_str(&reply)
        .map_err(|error| ProximaError::Record(format!("parse completion `{reply}`: {error}")))?;
    Ok(Greedy {
        text: parsed.content,
        ids: parsed.tokens,
    })
}

fn post(port: u16, path: &str, payload: &str) -> Result<String, ProximaError> {
    let fail = |error: std::io::Error| ProximaError::Record(format!("llama-server http: {error}"));
    let mut stream = TcpStream::connect(("127.0.0.1", port)).map_err(fail)?;
    let head = format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n",
        payload.len()
    );
    stream.write_all(head.as_bytes()).map_err(fail)?;
    stream.write_all(payload.as_bytes()).map_err(fail)?;
    let mut raw = String::new();
    stream.read_to_string(&mut raw).map_err(fail)?;
    raw.split_once("\r\n\r\n")
        .map(|(_, body)| body.to_string())
        .ok_or_else(|| ProximaError::Record(format!("malformed http reply: {raw}")))
}

#[derive(Deserialize)]
struct ServerCompletion {
    content: String,
    tokens: Vec<u32>,
}

impl LlamaOracle {
    fn incumbent() -> Self {
        Self {
            bin_dir: std::env::var_os(BIN_DIR_VAR).map(PathBuf::from),
        }
    }

    fn absent() -> Self {
        Self {
            bin_dir: Some(PathBuf::from(ABSENT_BIN_DIR)),
        }
    }

    fn bin_dir(&self) -> Result<&PathBuf, ProximaError> {
        self.bin_dir.as_ref().ok_or_else(|| {
            ProximaError::Record(format!(
                "{BIN_DIR_VAR} is not set; recording needs a CPU build of llama-tokenize and \
                 llama-server, point {BIN_DIR_VAR} at its bin directory"
            ))
        })
    }

    fn run(&self, binary: &str, args: &[&str]) -> Result<String, ProximaError> {
        let program = self.bin_dir()?.join(binary);
        let output = Command::new(&program)
            .args(args)
            .stdin(Stdio::null())
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
        let port = free_port()?;
        let mut server = self.spawn_server(port)?;
        let outcome = wait_listening(&mut server).and_then(|()| complete(port, prompt, count));
        let reaped = server.kill().and_then(|()| server.wait());
        let greedy = outcome?;
        reaped.map_err(|error| ProximaError::Record(format!("reap llama-server: {error}")))?;
        if greedy.ids.len() != count {
            return Err(ProximaError::Record(format!(
                "llama-server returned {} token ids for n_predict {count}",
                greedy.ids.len()
            )));
        }
        Ok(greedy)
    }

    fn spawn_server(&self, port: u16) -> Result<Child, ProximaError> {
        let program = self.bin_dir()?.join("llama-server");
        Command::new(&program)
            .args(["-m", MODEL, "--host", "127.0.0.1", "--no-webui", "-np", "1"])
            .args(["-c", "512", "--port", &port.to_string()])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| ProximaError::Record(format!("spawn {}: {error}", program.display())))
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
    assert_eq!(answer.ids, vec![12095, 13, 576, 6722, 315, 9625]);
}

#[proxima::test(cassette = "llama_greedy")]
async fn greedy_replays_with_the_binary_absent(
    #[cassette(LlamaOracle::absent())] pipe: PipeHandle,
) {
    let answer: Greedy = ask(&pipe, "/greedy", Some(GREEDY_TOKENS))
        .await
        .expect("greedy");

    assert_eq!(answer.text, " Paris. The capital of France");
    assert_eq!(answer.ids, vec![12095, 13, 576, 6722, 315, 9625]);
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
