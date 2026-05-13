//! Pure rewrites: sugary spec → primitive spec. Folded in from the former
//! `proxima-sugar` satellite crate.
//!
//! | sugar key | desugars to |
//! | --- | --- |
//! | `cache = true` | `kv:cache` upstream + fallthrough + write_back |
//! | `cache = { max_entries = N, ttl = "1h" }` | same with configured kv settings |
//! | `mock = { status = ..., body = ... }` | `synth` upstream |
//! | `replay = "fixture.jsonl"` | `replay` upstream over jsonl |
//! | `mcp = { command = "...", args = [...] }` | `process_rpc` upstream |
//! | `record = "trace.jsonl"` | `type = "record"` wrapping the sibling origin |
//!
//! [`desugar`] is the config half. [`builder`] is the fluent half: a
//! [`SpecBuilder`] seam + axis traits ([`ProtocolSugar`], [`TransportSugar`])
//! that accumulate the same spec `Value` — the fluent builder and the config
//! file meet on one spec (one door, free parity).

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;

use serde_json::{Map, Value, json};

use proxima_core::ProximaError;

pub mod builder;
pub use builder::{ProtocolSugar, SpecBuilder, TransportSugar};

pub fn desugar(value: Value) -> Result<Value, ProximaError> {
    let Value::Object(map) = value else {
        return Ok(value);
    };
    let has_cache = map.contains_key("cache");
    let has_mock = map.contains_key("mock");
    let has_replay_string = matches!(map.get("replay"), Some(Value::String(_)));
    let has_mcp = map.contains_key("mcp");
    let has_record_string = matches!(map.get("record"), Some(Value::String(_)));
    if !has_cache && !has_mock && !has_replay_string && !has_mcp && !has_record_string {
        return Ok(Value::Object(map));
    }
    let mut working = map;
    if working.contains_key("mock") {
        working = desugar_mock(working)?;
    }
    if matches!(working.get("replay"), Some(Value::String(_))) {
        working = desugar_replay_string(working)?;
    }
    if working.contains_key("mcp") {
        working = desugar_mcp(working)?;
    }
    // record runs after mcp/mock/replay so it wraps the resolved origin.
    if matches!(working.get("record"), Some(Value::String(_))) {
        working = desugar_record_string(working)?;
    }
    if working.contains_key("cache") {
        working = desugar_cache(working)?;
    }
    Ok(Value::Object(working))
}

fn desugar_cache(mut map: Map<String, Value>) -> Result<Map<String, Value>, ProximaError> {
    let cache_value = map.remove("cache").unwrap_or(Value::Bool(false));
    let kv_entry = match cache_value {
        Value::Bool(true) => json!({
            "name": "cache",
            "kv": "cache",
            "max_entries": 1024,
        }),
        Value::Bool(false) => return Ok(map),
        Value::Object(inner) => {
            let mut entry = serde_json::Map::new();
            entry.insert("name".into(), Value::String("cache".into()));
            entry.insert("kv".into(), Value::String("cache".into()));
            for (key, value) in inner {
                entry.insert(key, value);
            }
            entry
                .entry("max_entries".to_string())
                .or_insert_with(|| Value::Number(1024.into()));
            Value::Object(entry)
        }
        other => {
            return Err(ProximaError::Config(format!(
                "`cache` must be bool or object, got {other:?}"
            )));
        }
    };

    let mut origin_entry = serde_json::Map::new();
    origin_entry.insert("name".into(), Value::String("origin".into()));
    let mut origin_added = false;
    for key in [
        "http",
        "synth",
        "replay",
        "callback",
        "process",
        "process_rpc",
        "kv",
    ] {
        if let Some(value) = map.remove(key) {
            origin_entry.insert(key.into(), value);
            origin_added = true;
            break;
        }
    }
    if !origin_added {
        return Err(ProximaError::Config(
            "`cache = ...` requires a sibling origin (http / synth / replay / callback / process / process_rpc / kv)"
                .into(),
        ));
    }

    map.insert(
        "upstreams".into(),
        Value::Array(vec![
            Value::Object(
                kv_entry
                    .as_object()
                    .unwrap_or(&serde_json::Map::new())
                    .clone(),
            ),
            Value::Object(origin_entry),
        ]),
    );
    map.entry("select".to_string())
        .or_insert_with(|| json!({ "algorithm": "fallthrough", "miss_on": ["no_data"] }));
    map.entry("write_back".to_string())
        .or_insert_with(|| json!([["origin", "cache"]]));
    map.entry("name".to_string())
        .or_insert_with(|| Value::String("proxima".into()));
    Ok(map)
}

fn desugar_mock(mut map: Map<String, Value>) -> Result<Map<String, Value>, ProximaError> {
    let mock_value = map.remove("mock").unwrap_or(Value::Null);
    let synth_spec = match mock_value {
        Value::Object(inner) => Value::Object(inner),
        Value::String(text) => json!({ "status": 200, "body": text }),
        other => {
            return Err(ProximaError::Config(format!(
                "`mock` must be string or object, got {other:?}"
            )));
        }
    };
    map.insert("synth".into(), synth_spec);
    Ok(map)
}

fn desugar_mcp(mut map: Map<String, Value>) -> Result<Map<String, Value>, ProximaError> {
    let mcp_value = map.remove("mcp").unwrap_or(Value::Null);
    let process_rpc_spec = match mcp_value {
        Value::Object(inner) => Value::Object(inner),
        Value::String(text) => json!({ "command": text, "args": [] }),
        other => {
            return Err(ProximaError::Config(format!(
                "`mcp` must be string or object, got {other:?}"
            )));
        }
    };
    map.insert("process_rpc".into(), process_rpc_spec);
    Ok(map)
}

fn desugar_record_string(mut map: Map<String, Value>) -> Result<Map<String, Value>, ProximaError> {
    let path = match map.remove("record") {
        Some(Value::String(text)) => text,
        other => {
            return Err(ProximaError::Config(format!(
                "expected `record = \"path\"` (string), got {other:?}"
            )));
        }
    };
    let mut inner = serde_json::Map::new();
    let mut origin_added = false;
    for key in [
        "http",
        "synth",
        "replay",
        "callback",
        "process",
        "process_rpc",
        "kv",
        "upstreams",
    ] {
        if let Some(value) = map.remove(key) {
            inner.insert(key.into(), value);
            origin_added = true;
            if key != "upstreams" {
                break;
            }
        }
    }
    if !origin_added {
        return Err(ProximaError::Config(
            "`record = \"...\"` requires a sibling origin (http / synth / replay / callback / process / process_rpc / kv / upstreams)"
                .into(),
        ));
    }
    let format = if path.ends_with(".bin") {
        "bin"
    } else {
        "jsonl"
    };
    map.insert("type".into(), Value::String("record".into()));
    map.insert("sink".into(), json!({ "type": format, "path": path }));
    map.insert("inner".into(), Value::Object(inner));
    Ok(map)
}

fn desugar_replay_string(mut map: Map<String, Value>) -> Result<Map<String, Value>, ProximaError> {
    let path = match map.remove("replay") {
        Some(Value::String(text)) => text,
        other => {
            return Err(ProximaError::Config(format!(
                "expected `replay = \"path\"` (string), got {other:?}"
            )));
        }
    };
    let format = if path.ends_with(".bin") {
        "bin"
    } else {
        "jsonl"
    };
    map.insert("replay".into(), json!({ "source": path, "format": format }));
    Ok(map)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn passthrough_when_no_sugar_keys() {
        let raw = json!({"http": "https://example.com"});
        let desugared = desugar(raw.clone()).expect("desugar");
        assert_eq!(desugared, raw);
    }

    #[test]
    fn cache_true_with_http_origin_emits_fallthrough_kv_writeback() {
        let raw = json!({"http": "https://api.example.com", "cache": true});
        let desugared = desugar(raw).expect("desugar");
        let upstreams = desugared
            .get("upstreams")
            .and_then(Value::as_array)
            .expect("upstreams");
        assert_eq!(upstreams.len(), 2);
        assert_eq!(upstreams[0]["name"], "cache");
        assert_eq!(upstreams[0]["kv"], "cache");
        assert_eq!(upstreams[0]["max_entries"], 1024);
        assert_eq!(upstreams[1]["name"], "origin");
        assert_eq!(upstreams[1]["http"], "https://api.example.com");
        assert_eq!(
            desugared["select"],
            json!({"algorithm": "fallthrough", "miss_on": ["no_data"]}),
        );
        assert_eq!(desugared["write_back"], json!([["origin", "cache"]]));
    }

    #[test]
    fn cache_object_overrides_max_entries() {
        let raw = json!({
            "http": "https://api.example.com",
            "cache": { "max_entries": 64 },
        });
        let desugared = desugar(raw).expect("desugar");
        let upstreams = desugared["upstreams"].as_array().expect("upstreams");
        assert_eq!(upstreams[0]["max_entries"], 64);
    }

    #[test]
    fn cache_without_origin_returns_config_error() {
        let raw = json!({ "cache": true });
        let outcome = desugar(raw);
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[test]
    fn mock_object_desugars_to_synth() {
        let raw = json!({ "mock": { "status": 200, "body": "hi" } });
        let desugared = desugar(raw).expect("desugar");
        assert_eq!(desugared["synth"]["status"], 200);
        assert_eq!(desugared["synth"]["body"], "hi");
    }

    #[test]
    fn mock_string_desugars_to_synth_200_with_body() {
        let raw = json!({ "mock": "hello" });
        let desugared = desugar(raw).expect("desugar");
        assert_eq!(desugared["synth"]["status"], 200);
        assert_eq!(desugared["synth"]["body"], "hello");
    }

    #[test]
    fn replay_string_desugars_to_jsonl_source_object() {
        let raw = json!({ "replay": "./fixture.jsonl" });
        let desugared = desugar(raw).expect("desugar");
        assert_eq!(desugared["replay"]["source"], "./fixture.jsonl");
        assert_eq!(desugared["replay"]["format"], "jsonl");
    }

    #[test]
    fn replay_string_with_bin_extension_picks_bin_format() {
        let raw = json!({ "replay": "./fixture.bin" });
        let desugared = desugar(raw).expect("desugar");
        assert_eq!(desugared["replay"]["format"], "bin");
    }

    #[test]
    fn mcp_object_desugars_to_process_rpc() {
        let raw = json!({
            "mcp": { "command": "agent", "args": ["serve", "--mcp"] },
        });
        let desugared = desugar(raw).expect("desugar");
        let rpc = desugared.get("process_rpc").expect("process_rpc");
        assert_eq!(rpc["command"], "agent");
        assert_eq!(rpc["args"], json!(["serve", "--mcp"]));
    }

    #[test]
    fn mcp_string_desugars_to_process_rpc_with_no_args() {
        let raw = json!({ "mcp": "agent" });
        let desugared = desugar(raw).expect("desugar");
        let rpc = desugared.get("process_rpc").expect("process_rpc");
        assert_eq!(rpc["command"], "agent");
        assert_eq!(rpc["args"], json!([]));
    }

    #[test]
    fn mcp_plus_cache_composes_through_process_rpc_origin() {
        let raw = json!({
            "mcp": { "command": "agent", "args": ["serve", "--mcp"] },
            "cache": true,
        });
        let desugared = desugar(raw).expect("desugar");
        let upstreams = desugared["upstreams"].as_array().expect("upstreams");
        assert_eq!(upstreams[1]["name"], "origin");
        assert_eq!(upstreams[1]["process_rpc"]["command"], "agent");
    }

    #[test]
    fn record_string_wraps_sibling_http_origin() {
        let raw = json!({
            "http": "https://api.example.com",
            "record": "./trace.jsonl",
        });
        let desugared = desugar(raw).expect("desugar");
        assert_eq!(desugared["type"], "record");
        assert_eq!(desugared["sink"]["type"], "jsonl");
        assert_eq!(desugared["sink"]["path"], "./trace.jsonl");
        assert_eq!(desugared["inner"]["http"], "https://api.example.com");
        assert!(desugared.get("http").is_none(), "http moved into inner");
    }

    #[test]
    fn record_bin_extension_picks_bin_sink() {
        let raw = json!({
            "synth": { "status": 200, "body": "x" },
            "record": "./trace.bin",
        });
        let desugared = desugar(raw).expect("desugar");
        assert_eq!(desugared["sink"]["type"], "bin");
    }

    #[test]
    fn record_without_origin_returns_config_error() {
        let raw = json!({ "record": "./trace.jsonl" });
        let outcome = desugar(raw);
        assert!(matches!(outcome, Err(ProximaError::Config(_))));
    }

    #[test]
    fn mock_plus_cache_composes_through_synth_origin() {
        let raw = json!({
            "mock": { "status": 200, "body": "fake" },
            "cache": true,
        });
        let desugared = desugar(raw).expect("desugar");
        let upstreams = desugared["upstreams"].as_array().expect("upstreams");
        assert_eq!(upstreams[1]["name"], "origin");
        assert_eq!(upstreams[1]["synth"]["body"], "fake");
    }
}
