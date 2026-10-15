use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use proxima_gguf::parse_complete;
use proxima_gguf::value::MetadataValue;
use serde_json::{Value, json};

const HEADER_BYTES: u64 = 33_554_432;

fn main() -> ExitCode {
    match run() {
        Ok(rows) => {
            eprintln!("descriptor_census: {rows} rows parsed");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("descriptor_census: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<usize, String> {
    let (manifest, out) = parse_args()?;
    let text = std::fs::read_to_string(&manifest)
        .map_err(|error| format!("manifest {}: {error}", manifest.display()))?;
    let directory = manifest.parent().unwrap_or(Path::new("."));
    let mut output = String::new();
    for (index, line) in text.lines().enumerate() {
        let record = census_line(directory, line)
            .map_err(|error| format!("manifest line {}: {error}", index + 1))?;
        output.push_str(&record.to_string());
        output.push('\n');
    }
    std::fs::write(&out, &output).map_err(|error| format!("out {}: {error}", out.display()))?;
    Ok(output.lines().count())
}

fn parse_args() -> Result<(PathBuf, PathBuf), String> {
    let args: Vec<String> = std::env::args().collect();
    let value_of = |flag: &str| {
        args.iter()
            .position(|arg| arg == flag)
            .and_then(|position| args.get(position + 1))
            .map(PathBuf::from)
            .ok_or_else(|| format!("missing {flag} <path>"))
    };
    Ok((value_of("--manifest")?, value_of("--out")?))
}

fn census_line(directory: &Path, line: &str) -> Result<Value, String> {
    let entry: Value = serde_json::from_str(line).map_err(|error| format!("not json: {error}"))?;
    let row = entry["row"].as_u64().ok_or("no `row` key")?;
    let file = entry["file"]
        .as_str()
        .ok_or_else(|| format!("row {row}: no `file` key"))?;
    let sha256 = entry["sha256"]
        .as_str()
        .ok_or_else(|| format!("row {row}: no `sha256` key"))?;
    let path = directory.join(file);
    let bytes = std::fs::metadata(&path)
        .map_err(|error| format!("row {row}: {}: {error}", path.display()))?
        .len();
    let header = read_header(&path, bytes)
        .map_err(|error| format!("row {row}: {}: {error}", path.display()))?;
    let parsed = parse_complete(&header)
        .map_err(|error| format!("row {row}: {file} header does not parse: {error:?}"))?;
    let architecture = match parsed.metadata_value("general.architecture") {
        Some(MetadataValue::String(name)) => name.clone(),
        other => return Err(format!("row {row}: general.architecture is {other:?}")),
    };
    Ok(json!({
        "row": row,
        "file": file,
        "bytes": bytes,
        "sha256": sha256,
        "architecture": architecture,
        "tensor_count": parsed.tensors.len(),
        "kv_count": parsed.metadata.len(),
        "parsed": true,
    }))
}

fn read_header(path: &Path, bytes: u64) -> std::io::Result<Vec<u8>> {
    let mut header = Vec::new();
    File::open(path)?
        .take(bytes.min(HEADER_BYTES))
        .read_to_end(&mut header)?;
    Ok(header)
}
