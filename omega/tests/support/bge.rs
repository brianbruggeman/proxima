use std::env;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use moe_test_json::Value;
use sha2::{Digest, Sha256};

const MODEL_PATH_ENV: &str = "BGE_MODEL_PATH";
const MODEL_RELATIVE_PATH: &str = ".cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/onnx/model.onnx";

pub struct CorpusRow {
    pub text: String,
    pub ids: Vec<i64>,
}

pub fn model_path() -> Result<PathBuf> {
    match env::var_os(MODEL_PATH_ENV) {
        Some(path) => Ok(PathBuf::from(path)),
        None => Ok(
            PathBuf::from(env::var_os("HOME").context("HOME is not set")?)
                .join(MODEL_RELATIVE_PATH),
        ),
    }
}

pub fn read_model() -> Result<Vec<u8>> {
    let path = model_path()?;
    fs::read(&path).with_context(|| format!("read model {}", path.display()))
}

pub fn read_fixture(name: &str) -> Result<Vec<u8>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/bge-small")
        .join(name);
    fs::read(&path).with_context(|| format!("read fixture {}", path.display()))
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub fn recorded_sha256() -> Result<String> {
    let record = String::from_utf8(read_fixture("model.sha256")?)?;
    let hash = record
        .split_whitespace()
        .next()
        .context("model.sha256 is empty")?;
    Ok(hash.to_owned())
}

pub fn read_corpus() -> Result<Vec<CorpusRow>> {
    let text = String::from_utf8(read_fixture("corpus.jsonl")?)?;
    text.lines().map(parse_row).collect()
}

fn parse_row(line: &str) -> Result<CorpusRow> {
    let value: Value = moe_test_json::from_str(line)?;
    let text = value["text"]
        .as_str()
        .context("row has no text")?
        .to_owned();
    let ids = value["ids"]
        .as_array()
        .context("row has no ids")?
        .iter()
        .map(|id| {
            id.as_i64()
                .ok_or_else(|| anyhow!("id is not an integer: {id}"))
        })
        .collect::<Result<_>>()?;
    Ok(CorpusRow { text, ids })
}
