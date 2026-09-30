//! The Ollama arm: request body (R16c1, R16c2), a blocking HTTP client, and
//! the digest lookup that maps a GGUF blob to the model name Ollama serves.

use std::fs;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::NiahError;

const OLLAMA_ADDRESS: &str = "localhost:11434";
const MODEL_LAYER: &str = "application/vnd.ollama.image.model";
const DEFAULT_REGISTRY: &str = "registry.ollama.ai";
const DEFAULT_NAMESPACE: &str = "library";

/// One installed Ollama model: its served name and the blob file that holds
/// its GGUF, e.g. `gemma4:e2b-it-qat` and `sha256-3646b4c1...`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ManifestEntry {
    pub(crate) name: String,
    pub(crate) blob: String,
}

/// The `/api/generate` body: the exact proxima prompt bytes, raw, unstreamed,
/// greedy, at the same `num_ctx` the proxima arm serves.
pub(crate) fn request_body(model: &str, prompt: &str, context: u32, max_new: usize) -> Value {
    json!({
        "model": model,
        "prompt": prompt,
        "raw": true,
        "stream": false,
        "options": {"temperature": 0, "num_ctx": context, "num_predict": max_new},
    })
}

pub(crate) fn answer(body: &Value) -> Result<String, NiahError> {
    let payload = http_post_json("/api/generate", &body.to_string())?;
    let reply: Value = serde_json::from_slice(&payload)?;
    reply
        .get("response")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| NiahError::Failed("ollama reply has no response field".to_string()))
}

fn decode_chunked(body: &[u8]) -> Result<Vec<u8>, NiahError> {
    let mut decoded = Vec::new();
    let mut rest = body;
    loop {
        let line_end = rest
            .windows(2)
            .position(|pair| pair == b"\r\n")
            .ok_or_else(|| NiahError::Failed("chunked body: missing size line".to_string()))?;
        let size_text = String::from_utf8_lossy(&rest[..line_end]);
        let size = usize::from_str_radix(size_text.split(';').next().unwrap_or("").trim(), 16)
            .map_err(|error| NiahError::Failed(format!("chunked body: bad size: {error}")))?;
        let start = line_end + 2;
        if size == 0 {
            return Ok(decoded);
        }
        let chunk = rest
            .get(start..start + size)
            .ok_or_else(|| NiahError::Failed("chunked body: truncated chunk".to_string()))?;
        decoded.extend_from_slice(chunk);
        rest = rest
            .get(start + size + 2..)
            .ok_or_else(|| NiahError::Failed("chunked body: missing chunk end".to_string()))?;
    }
}

fn http_post_json(path: &str, body: &str) -> Result<Vec<u8>, NiahError> {
    let mut stream = TcpStream::connect(OLLAMA_ADDRESS)
        .map_err(|source| NiahError::io(OLLAMA_ADDRESS, source))?;
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {OLLAMA_ADDRESS}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(request.as_bytes())
        .map_err(|source| NiahError::io("ollama write", source))?;
    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .map_err(|source| NiahError::io("ollama read", source))?;
    let header_end = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| NiahError::Failed("ollama: response has no header end".to_string()))?;
    let header = String::from_utf8_lossy(&raw[..header_end]).to_lowercase();
    let payload = &raw[header_end + 4..];
    if !header.starts_with("http/1.1 200") {
        return Err(NiahError::Failed(format!(
            "ollama: {} body={}",
            header.lines().next().unwrap_or(""),
            String::from_utf8_lossy(payload)
        )));
    }
    if header.contains("transfer-encoding: chunked") {
        decode_chunked(payload)
    } else {
        Ok(payload.to_vec())
    }
}

pub(crate) fn models_root() -> Result<PathBuf, NiahError> {
    if let Some(root) = std::env::var_os("OLLAMA_MODELS") {
        return Ok(PathBuf::from(root));
    }
    let home = std::env::var_os("HOME")
        .ok_or_else(|| NiahError::Failed("HOME is not set and OLLAMA_MODELS is unset".into()))?;
    Ok(PathBuf::from(home).join(".ollama").join("models"))
}

fn manifest_files(manifests: &Path) -> Result<Vec<PathBuf>, NiahError> {
    let mut files = Vec::new();
    let mut pending = vec![manifests.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let listing = fs::read_dir(&directory)
            .map_err(|source| NiahError::io(&directory.display().to_string(), source))?;
        for entry in listing {
            let path = entry
                .map_err(|source| NiahError::io(&directory.display().to_string(), source))?
                .path();
            if path.is_dir() {
                pending.push(path);
            } else {
                files.push(path);
            }
        }
    }
    Ok(files)
}

fn served_name(manifests: &Path, file: &Path) -> Option<String> {
    let parts: Vec<&str> = file
        .strip_prefix(manifests)
        .ok()?
        .iter()
        .filter_map(|part| part.to_str())
        .collect();
    let [registry, namespace, model, tag] = parts.as_slice() else {
        return None;
    };
    let host = if *registry == DEFAULT_REGISTRY {
        String::new()
    } else {
        format!("{registry}/")
    };
    let scope = if *namespace == DEFAULT_NAMESPACE {
        String::new()
    } else {
        format!("{namespace}/")
    };
    Some(format!("{host}{scope}{model}:{tag}"))
}

fn model_blob(manifest: &Value) -> Option<String> {
    manifest
        .get("layers")?
        .as_array()?
        .iter()
        .find(|layer| layer.get("mediaType").and_then(Value::as_str) == Some(MODEL_LAYER))?
        .get("digest")?
        .as_str()
        .map(|digest| digest.replace(':', "-"))
}

/// Every model installed under `<models_root>/manifests`, by walking the
/// registry/namespace/model/tag tree Ollama writes.
pub(crate) fn manifest_entries(models_root: &Path) -> Result<Vec<ManifestEntry>, NiahError> {
    let manifests = models_root.join("manifests");
    manifest_files(&manifests)?
        .into_iter()
        .filter_map(|file| {
            let name = served_name(&manifests, &file)?;
            Some((file, name))
        })
        .map(|(file, name)| {
            let text = fs::read_to_string(&file)
                .map_err(|source| NiahError::io(&file.display().to_string(), source))?;
            let manifest: Value = serde_json::from_str(&text)?;
            Ok(model_blob(&manifest).map(|blob| ManifestEntry { name, blob }))
        })
        .filter_map(Result::transpose)
        .collect()
}

/// The Ollama model name whose GGUF blob is `model_path` (a blob path, or a
/// symlink to one).
pub(crate) fn name_for_blob(models_root: &Path, model_path: &Path) -> Result<String, NiahError> {
    let resolved = fs::canonicalize(model_path)
        .map_err(|source| NiahError::io(&model_path.display().to_string(), source))?;
    let blob = resolved
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| NiahError::OllamaModelUnknown(resolved.display().to_string()))?;
    manifest_entries(models_root)?
        .into_iter()
        .find(|entry| entry.blob == blob)
        .map(|entry| entry.name)
        .ok_or_else(|| NiahError::OllamaModelUnknown(blob.to_string()))
}
