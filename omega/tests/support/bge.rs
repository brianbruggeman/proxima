use std::env;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, ensure};
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

pub struct RecordBatch {
    pub seq: usize,
    pub input_ids: Vec<Vec<i64>>,
    pub attention_mask: Vec<Vec<i64>>,
    pub token_type_ids: Vec<Vec<i64>>,
    pub cls: Vec<Vec<f32>>,
}

pub struct Record {
    pub header: Value,
    pub batches: Vec<RecordBatch>,
}

pub fn read_record(cell: &str) -> Result<Record> {
    let text = String::from_utf8(read_fixture(&format!("tract_{cell}.jsonl"))?)?;
    let mut lines = text.lines();
    let header = moe_test_json::from_str(lines.next().context("record is empty")?)?;
    let batches = lines.map(parse_batch).collect::<Result<_>>()?;
    Ok(Record { header, batches })
}

fn parse_batch(line: &str) -> Result<RecordBatch> {
    let value: Value = moe_test_json::from_str(line)?;
    Ok(RecordBatch {
        seq: usize::try_from(value["seq"].as_u64().context("batch has no seq")?)?,
        input_ids: integer_rows(&value, "input_ids")?,
        attention_mask: integer_rows(&value, "attention_mask")?,
        token_type_ids: integer_rows(&value, "token_type_ids")?,
        cls: float_rows(&value, "cls")?,
    })
}

fn rows_of<'a>(value: &'a Value, key: &str) -> Result<&'a Vec<Value>> {
    value[key]
        .as_array()
        .with_context(|| format!("batch has no {key}"))
}

fn integer_rows(value: &Value, key: &str) -> Result<Vec<Vec<i64>>> {
    rows_of(value, key)?
        .iter()
        .map(|row| {
            row.as_array()
                .context("row is not an array")?
                .iter()
                .map(|item| item.as_i64().context("item is not an integer"))
                .collect()
        })
        .collect()
}

fn float_rows(value: &Value, key: &str) -> Result<Vec<Vec<f32>>> {
    rows_of(value, key)?
        .iter()
        .map(|row| {
            row.as_array()
                .context("row is not an array")?
                .iter()
                .map(|item| {
                    item.as_f64()
                        .map(|float| float as f32)
                        .context("item is not a float")
                })
                .collect()
        })
        .collect()
}

pub fn l2_normalize(row: &[f32]) -> Vec<f32> {
    let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
    row.iter().map(|value| value / norm).collect()
}

pub fn max_abs_difference(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max)
}

pub fn cosine(left: &[f32], right: &[f32]) -> f64 {
    let (left, right) = (l2_normalize(left), l2_normalize(right));
    left.iter()
        .zip(&right)
        .map(|(left, right)| f64::from(left * right))
        .sum()
}

pub fn all_within(actual: &[f64], expected: &[f64], tolerance: f64) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() <= tolerance)
}

pub const EMBEDDING_DIMENSION: usize = 384;

pub fn cls_rows(values: &[f32], rows: usize) -> Result<Vec<Vec<f32>>> {
    ensure!(
        rows > 0 && values.len().is_multiple_of(rows),
        "{} values do not split into {rows} rows",
        values.len()
    );
    let stride = values.len() / rows;
    ensure!(
        stride >= EMBEDDING_DIMENSION,
        "row stride {stride} is below {EMBEDDING_DIMENSION}"
    );
    Ok(values
        .chunks_exact(stride)
        .map(|row| row[..EMBEDDING_DIMENSION].to_vec())
        .collect())
}

pub fn flatten_as_float(rows: &[Vec<i64>]) -> Vec<f32> {
    rows.iter().flatten().map(|&value| value as f32).collect()
}

pub fn sentence_differences(actual: &[Vec<f32>], expected: &[Vec<f32>]) -> Result<Vec<f32>> {
    ensure!(
        actual.len() == expected.len(),
        "{} actual rows against {} expected rows",
        actual.len(),
        expected.len()
    );
    Ok(actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| max_abs_difference(actual, expected))
        .collect())
}

pub fn norm(row: &[f32]) -> f32 {
    row.iter().map(|value| value * value).sum::<f32>().sqrt()
}

pub struct TimingPass {
    pub cell: String,
    pub phase: String,
    pub load_before: f64,
}

pub struct Timing {
    pub header: Value,
    pub passes: Vec<TimingPass>,
}

pub fn read_timing() -> Result<Timing> {
    let text = String::from_utf8(read_fixture("tract_timing.jsonl")?)?;
    let mut lines = text.lines();
    let header = moe_test_json::from_str(lines.next().context("timing record is empty")?)?;
    let passes = lines.map(parse_pass).collect::<Result<_>>()?;
    Ok(Timing { header, passes })
}

fn parse_pass(line: &str) -> Result<TimingPass> {
    let value: Value = moe_test_json::from_str(line)?;
    let text = |key: &str| {
        value[key]
            .as_str()
            .map(str::to_owned)
            .with_context(|| format!("pass has no {key}"))
    };
    Ok(TimingPass {
        cell: text("cell")?,
        phase: text("phase")?,
        load_before: value["load_before"]
            .as_f64()
            .context("pass has no load_before")?,
    })
}

pub fn passes_in(timing: &Timing, cell: &str, phase: &str) -> usize {
    timing
        .passes
        .iter()
        .filter(|pass| pass.cell == cell && pass.phase == phase)
        .count()
}

pub fn all_started_at_or_below(passes: &[TimingPass], limit: f64) -> bool {
    passes.iter().all(|pass| pass.load_before <= limit)
}
