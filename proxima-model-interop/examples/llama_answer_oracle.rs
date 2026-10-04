//! Side-by-side answer text, proxima vs llama-server, on the ids in a
//! `tests/fixtures/llama-parity/<model>/llama_ids.json` oracle file.
//!
//! proxima generates first (real checkpoint, `generate_from_ids`, prompt cache
//! off, greedy), then a llama-server child is spawned on the same GGUF. For
//! every prompt the output holds: proxima ids and text, llama's fixture ids,
//! llama's live ids, llama's `content`, `/detokenize` of both id lists, the
//! first differing index with both ids decoded by both detokenizers, and
//! llama's top-5 logprobs at every generated step (`/completion` with
//! `n_probs: 5`, `top_k: 1`, `cache_prompt: false`).
//!
//! `cargo run --release --example llama_answer_oracle --features std,metal -- \
//!    --model <gguf> --fixture <llama_ids.json> --llama-server-bin <path> --out <json>`

#![allow(clippy::expect_used, clippy::unwrap_used)]

use core::ops::ControlFlow;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use proxima_gguf::parse_complete;
use proxima_model_interop::{LoadedModel, PromptCacheConfig, ServingConfig};
use proxima_tokenizer::vocab::Vocab;
use serde_json::{Value, json};

struct Arguments {
    model: PathBuf,
    fixture: PathBuf,
    llama_server_bin: PathBuf,
    out: PathBuf,
    port: u16,
    ollama_model: Option<String>,
    llama_ngl: String,
    skip_proxima: bool,
    n_probs: usize,
    llama_extra: Vec<String>,
}

fn parse_arguments() -> Arguments {
    let mut values = BTreeMap::new();
    let mut tokens = std::env::args().skip(1);
    while let Some(flag) = tokens.next() {
        let value = tokens
            .next()
            .unwrap_or_else(|| panic!("{flag} needs a value"));
        values.insert(flag, value);
    }
    let required = |flag: &str| {
        values
            .get(flag)
            .unwrap_or_else(|| panic!("missing {flag}"))
            .clone()
    };
    Arguments {
        model: required("--model").into(),
        fixture: required("--fixture").into(),
        llama_server_bin: required("--llama-server-bin").into(),
        out: required("--out").into(),
        port: values
            .get("--port")
            .map_or(18211, |text| text.parse().expect("--port is a u16")),
        ollama_model: values.get("--ollama-model").cloned(),
        llama_ngl: values
            .get("--llama-ngl")
            .cloned()
            .unwrap_or_else(|| "99".to_owned()),
        skip_proxima: values.contains_key("--skip-proxima"),
        llama_extra: values.get("--llama-extra").map_or_else(Vec::new, |text| {
            text.split_whitespace().map(str::to_owned).collect()
        }),
        n_probs: values
            .get("--n-probs")
            .map_or(5, |text| text.parse().expect("--n-probs is a usize")),
    }
}

struct LlamaServer {
    child: Child,
    port: u16,
}

impl LlamaServer {
    fn spawn(arguments: &Arguments) -> Self {
        let mut command = Command::new(&arguments.llama_server_bin);
        if let Some(lib_dir) = arguments.llama_server_bin.parent() {
            command.env("DYLD_LIBRARY_PATH", lib_dir);
        }
        let mut child = command
            .arg("-m")
            .arg(&arguments.model)
            .args([
                "-ngl",
                &arguments.llama_ngl,
                "-c",
                "4096",
                "--host",
                "127.0.0.1",
                "--no-webui",
            ])
            .args(["--port", &arguments.port.to_string()])
            .args(&arguments.llama_extra)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn llama-server");
        let stderr = child.stderr.take().expect("llama-server stderr is piped");
        let mut lines = BufReader::new(stderr).lines();
        let server = Self {
            child,
            port: arguments.port,
        };
        let ready = lines
            .by_ref()
            .map_while(Result::ok)
            .any(|line| line.contains("listening on"));
        assert!(ready, "llama-server exited before it reported listening");
        std::thread::spawn(move || lines.for_each(drop));
        let health = server.request("GET", "/health", None);
        assert_eq!(health["status"], "ok", "llama-server /health: {health}");
        server
    }

    fn request(&self, method: &str, path: &str, body: Option<&Value>) -> Value {
        http_json(self.port, method, path, body)
    }

    fn detokenize(&self, ids: &[u32]) -> String {
        let reply = self.request("POST", "/detokenize", Some(&json!({ "tokens": ids })));
        reply["content"]
            .as_str()
            .expect("detokenize content")
            .to_owned()
    }

    fn complete(&self, prompt_ids: &[u32], n_predict: usize, n_probs: usize) -> Value {
        let body = json!({
            "prompt": prompt_ids,
            "n_predict": n_predict,
            "temperature": 0,
            "top_k": 1,
            "n_probs": n_probs,
            "cache_prompt": false,
            "return_tokens": true,
        });
        self.request("POST", "/completion", Some(&body))
    }
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn dechunk(mut body: &[u8]) -> Vec<u8> {
    let mut decoded = Vec::new();
    loop {
        let line_end = body
            .windows(2)
            .position(|window| window == b"\r\n")
            .expect("chunk size line");
        let size = usize::from_str_radix(String::from_utf8_lossy(&body[..line_end]).trim(), 16)
            .expect("chunk size is hex");
        if size == 0 {
            return decoded;
        }
        let start = line_end + 2;
        decoded.extend_from_slice(&body[start..start + size]);
        body = &body[start + size + 2..];
    }
}

fn http_json(port: u16, method: &str, path: &str, body: Option<&Value>) -> Value {
    let payload = body.map(|value| serde_json::to_vec(value).expect("json body"));
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect http server");
    let content_length = payload.as_ref().map_or(0, Vec::len);
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {content_length}\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(head.as_bytes()).expect("write head");
    if let Some(payload) = &payload {
        stream.write_all(payload).expect("write body");
    }
    let mut response = Vec::new();
    stream.read_to_end(&mut response).expect("read response");
    let split = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .expect("header/body separator");
    let head_text = String::from_utf8_lossy(&response[..split]).to_ascii_lowercase();
    assert!(
        head_text.starts_with("http/1.1 200"),
        "{method} {path}: {head_text}"
    );
    let raw_body = &response[split + 4..];
    let body_bytes = if head_text.contains("transfer-encoding: chunked") {
        dechunk(raw_body)
    } else {
        raw_body.to_vec()
    };
    serde_json::from_slice(&body_bytes).expect("json response")
}

const OLLAMA_PORT: u16 = 11434;

fn ollama_generate(model: &str, prompt: &str, n_predict: usize, unload: bool) -> Value {
    let body = json!({
        "model": model,
        "prompt": prompt,
        "raw": true,
        "stream": false,
        "logprobs": true,
        "top_logprobs": 5,
        "keep_alive": if unload { json!(0) } else { json!("5m") },
        "options": { "temperature": 0, "top_k": 1, "num_predict": n_predict, "seed": 1 },
    });
    http_json(OLLAMA_PORT, "POST", "/api/generate", Some(&body))
}

fn ids_of(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .expect("ids are a json array")
        .iter()
        .map(|id| u32::try_from(id.as_u64().expect("unsigned id")).expect("id fits u32"))
        .collect()
}

fn first_divergence(expected: &[u32], actual: &[u32]) -> Option<usize> {
    let shared = expected.len().min(actual.len());
    (0..shared)
        .find(|&index| expected[index] != actual[index])
        .or_else(|| (expected.len() != actual.len()).then_some(shared))
}

struct Case {
    prompt: String,
    prompt_ids: Vec<u32>,
    llama_fixture_ids: Vec<u32>,
}

fn load_cases(arguments: &Arguments) -> Vec<Case> {
    let text = std::fs::read_to_string(&arguments.fixture).expect("read fixture");
    let records: Vec<Value> = serde_json::from_str(&text).expect("fixture json");
    records
        .iter()
        .map(|record| Case {
            prompt: record["prompt"].as_str().expect("prompt").to_owned(),
            prompt_ids: ids_of(&record["prompt_ids"]),
            llama_fixture_ids: ids_of(&record["generated_ids"]),
        })
        .collect()
}

fn proxima_generate(arguments: &Arguments, cases: &[Case]) -> (Vec<Vec<u32>>, Vocab) {
    let file = File::open(&arguments.model).expect("open model");
    let mapping = unsafe { memmap2::Mmap::map(&file) }.expect("mmap model");
    let bytes: &[u8] = &mapping;
    let parsed = parse_complete(bytes).expect("parse gguf header");
    let vocab = proxima_tokenizer::gguf::vocab_from_metadata(&parsed).expect("vocab");
    if arguments.skip_proxima {
        let echoed = cases
            .iter()
            .map(|case| case.llama_fixture_ids.clone())
            .collect();
        return (echoed, vocab);
    }
    let model = LoadedModel::load(&parsed, bytes).expect("load model");
    let config = ServingConfig {
        prompt_cache: PromptCacheConfig::off(),
        ..ServingConfig::default()
    };
    let generated = cases
        .iter()
        .map(|case| {
            let (ids, _text, _stopped) = model
                .generate_from_ids(
                    &case.prompt_ids,
                    case.llama_fixture_ids.len(),
                    &config,
                    &mut |_event| ControlFlow::Continue(()),
                )
                .expect("proxima generate_from_ids");
            ids
        })
        .collect();
    (generated, vocab)
}

fn record_for(
    case: &Case,
    proxima_ids: &[u32],
    vocab: &Vocab,
    llama: &LlamaServer,
    n_probs: usize,
) -> Value {
    let proxima_decode =
        |ids: &[u32]| proxima_tokenizer::decode(ids, vocab).expect("proxima detokenize");
    let completion = llama.complete(&case.prompt_ids, case.llama_fixture_ids.len(), n_probs);
    let live_ids = ids_of(&completion["tokens"]);
    let divergence = first_divergence(&case.llama_fixture_ids, proxima_ids);
    let differing = divergence.map(|index| {
        let proxima_id = proxima_ids.get(index).copied();
        let llama_id = case.llama_fixture_ids.get(index).copied();
        json!({
            "index": index,
            "proxima_id": proxima_id,
            "llama_id": llama_id,
            "proxima_id_by_proxima_decode": proxima_id.map(|id| proxima_decode(&[id])),
            "proxima_id_by_llama_detokenize": proxima_id.map(|id| llama.detokenize(&[id])),
            "llama_id_by_proxima_decode": llama_id.map(|id| proxima_decode(&[id])),
            "llama_id_by_llama_detokenize": llama_id.map(|id| llama.detokenize(&[id])),
        })
    });
    json!({
        "prompt": case.prompt,
        "prompt_ids": case.prompt_ids,
        "proxima_ids": proxima_ids,
        "proxima_text_by_proxima_decode": proxima_decode(proxima_ids),
        "proxima_text_by_llama_detokenize": llama.detokenize(proxima_ids),
        "llama_fixture_ids": case.llama_fixture_ids,
        "llama_live_ids": live_ids,
        "llama_live_equals_fixture": live_ids == case.llama_fixture_ids,
        "llama_text_by_llama_content": completion["content"],
        "llama_text_by_llama_detokenize": llama.detokenize(&case.llama_fixture_ids),
        "llama_text_by_proxima_decode": proxima_decode(&case.llama_fixture_ids),
        "first_divergence": differing,
        "llama_top5_per_step": completion["completion_probabilities"],
    })
}

fn main() {
    let arguments = parse_arguments();
    let cases = load_cases(&arguments);
    let (proxima_ids, vocab) = proxima_generate(&arguments, &cases);
    let llama = LlamaServer::spawn(&arguments);
    let mut records: Vec<Value> = cases
        .iter()
        .zip(&proxima_ids)
        .map(|(case, ids)| record_for(case, ids, &vocab, &llama, arguments.n_probs))
        .collect();
    drop(llama);
    if let Some(model) = &arguments.ollama_model {
        let last = cases.len() - 1;
        for (index, (case, record)) in cases.iter().zip(&mut records).enumerate() {
            let reply = ollama_generate(
                model,
                &case.prompt,
                case.llama_fixture_ids.len(),
                index == last,
            );
            record["ollama_text"] = reply["response"].clone();
            record["ollama_logprobs"] = reply["logprobs"].clone();
            record["ollama_eval_counts"] = json!({ "prompt_eval_count": reply["prompt_eval_count"], "eval_count": reply["eval_count"] });
        }
    }
    let rendered = serde_json::to_string_pretty(&records).expect("render records");
    std::fs::write(&arguments.out, &rendered).expect("write out");
    println!("{rendered}");
}
