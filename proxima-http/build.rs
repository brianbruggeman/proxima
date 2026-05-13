//! Build-time sizing for proxima-listeners-http.
//!
//! Reads `proxima-listeners-http.toml` at the crate root + applies optional
//! per-system env-var overrides, emitting a generated constant module into
//! `OUT_DIR/proxima_listeners_http_sized.rs`. The library `include!`s the
//! generated file inside a `sized` module.
//!
//! These are the compile-time DEFAULTS. The per-process surface is the
//! runtime conflaguration `HttpListenerConfig`, which overrides them via
//! config file / env / fluent builder.
//!
//! Env-var override naming: `PROXIMA_LISTENERS_HTTP_<SECTION>_<KEY>`
//! (e.g. `PROXIMA_LISTENERS_HTTP_LISTENER_DRAIN_TIMEOUT_MS=30000`).
//! Downstream consumers that re-brand their own surface map their config
//! onto these via the runtime `HttpListenerConfig` API, not by setting
//! these build vars directly.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use toml::Value;

fn env_name(section: &str, key: &str) -> String {
    format!(
        "PROXIMA_LISTENERS_HTTP_{}_{}",
        section.to_ascii_uppercase(),
        key.to_ascii_uppercase()
    )
}

fn get_str<'table>(table: &'table Value, section: &str, key: &str) -> &'table str {
    table
        .get(section)
        .and_then(|sec| sec.get(key))
        .and_then(Value::as_str)
        .unwrap_or_else(|| {
            panic!("proxima-listeners-http.toml: missing or non-string [{section}].{key}")
        })
}

fn get_int(table: &Value, section: &str, key: &str) -> u64 {
    let raw = table
        .get(section)
        .and_then(|sec| sec.get(key))
        .and_then(Value::as_integer)
        .unwrap_or_else(|| {
            panic!("proxima-listeners-http.toml: missing or non-integer [{section}].{key}")
        });
    u64::try_from(raw).unwrap_or_else(|_| panic!("[{section}].{key} = {raw} must be non-negative"))
}

fn get_bool(table: &Value, section: &str, key: &str) -> bool {
    table
        .get(section)
        .and_then(|sec| sec.get(key))
        .and_then(Value::as_bool)
        .unwrap_or_else(|| {
            panic!("proxima-listeners-http.toml: missing or non-bool [{section}].{key}")
        })
}

/// Read a string `(section, key)` from the TOML, then apply the optional
/// `PROXIMA_LISTENERS_HTTP_<SECTION>_<KEY>` env-var override.
fn resolve_str(table: &Value, section: &str, key: &str) -> String {
    let name = env_name(section, key);
    println!("cargo:rerun-if-env-changed={name}");
    env::var(&name).unwrap_or_else(|_| get_str(table, section, key).to_string())
}

/// Read an integer `(section, key)` from the TOML, then apply the optional
/// `PROXIMA_LISTENERS_HTTP_<SECTION>_<KEY>` env-var override.
fn resolve_int(table: &Value, section: &str, key: &str) -> u64 {
    let name = env_name(section, key);
    println!("cargo:rerun-if-env-changed={name}");
    if let Ok(raw) = env::var(&name) {
        return raw
            .parse()
            .unwrap_or_else(|err| panic!("{name} = {raw}: {err}"));
    }
    get_int(table, section, key)
}

/// Read a bool `(section, key)` from the TOML, then apply the optional
/// `PROXIMA_LISTENERS_HTTP_<SECTION>_<KEY>` env-var override.
fn resolve_bool(table: &Value, section: &str, key: &str) -> bool {
    let name = env_name(section, key);
    println!("cargo:rerun-if-env-changed={name}");
    if let Ok(raw) = env::var(&name) {
        return raw
            .parse()
            .unwrap_or_else(|err| panic!("{name} = {raw}: {err}"));
    }
    get_bool(table, section, key)
}

fn require_u16(name: &str, value: u64) -> u16 {
    u16::try_from(value).unwrap_or_else(|_| panic!("{name} = {value} overflows u16"))
}

fn require_nonempty(name: &str, value: &str) -> String {
    assert!(!value.is_empty(), "{name} must be non-empty");
    value.to_string()
}

#[allow(clippy::expect_used)]
fn emit_sizing_consts(out_dir: &Path) {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set by cargo");
    let toml_path = PathBuf::from(&manifest_dir).join("proxima-listeners-http.toml");
    println!("cargo:rerun-if-changed=proxima-listeners-http.toml");

    let text = fs::read_to_string(&toml_path)
        .unwrap_or_else(|err| panic!("read {}: {err}", toml_path.display()));
    let root: Value = text
        .parse()
        .unwrap_or_else(|err| panic!("parse {}: {err}", toml_path.display()));

    let name = require_nonempty("listener.name", &resolve_str(&root, "listener", "name"));
    let drain_timeout_ms = resolve_int(&root, "listener", "drain_timeout_ms");
    let quiesce_status = require_u16(
        "listener.quiesce_status",
        resolve_int(&root, "listener", "quiesce_status"),
    );
    let quiesce_retry_after = require_nonempty(
        "listener.quiesce_retry_after",
        &resolve_str(&root, "listener", "quiesce_retry_after"),
    );
    let proxy_protocol_enabled = resolve_bool(&root, "listener", "proxy_protocol_enabled");
    let encode_buffer_headroom = resolve_int(&root, "serve_pipe", "encode_buffer_headroom");

    let out = format!(
        "// AUTO-GENERATED by build.rs from proxima-listeners-http.toml. DO NOT EDIT.\n\
         /// Fallback telemetry label for a listener whose spec doesn't set\n\
         /// `name` (default). At std this seeds `HttpListenerConfig::name`.\n\
         pub const LISTENER_NAME_DEFAULT: &str = {name:?};\n\
         /// Fallback in-flight-drain timeout (ms, default). At\n\
         /// no_std+no_alloc this const IS the drain window; at std it seeds\n\
         /// `HttpListenerConfig::drain_timeout_ms`.\n\
         pub const LISTENER_DRAIN_TIMEOUT_MS_DEFAULT: u64 = {drain_timeout_ms};\n\
         /// Fallback HTTP status returned while quiescing (default). At std\n\
         /// this seeds `HttpListenerConfig::quiesce_status`.\n\
         pub const LISTENER_QUIESCE_STATUS_DEFAULT: u16 = {quiesce_status};\n\
         /// Fallback `Retry-After` header value (seconds, default). At std\n\
         /// this seeds `HttpListenerConfig::quiesce_retry_after`.\n\
         pub const LISTENER_QUIESCE_RETRY_AFTER_DEFAULT: &str = {quiesce_retry_after:?};\n\
         /// Fallback PROXY-protocol requirement (default). At std this seeds\n\
         /// `HttpListenerConfig::proxy_protocol_enabled`.\n\
         pub const LISTENER_PROXY_PROTOCOL_ENABLED_DEFAULT: bool = {proxy_protocol_enabled};\n\
         /// Fallback headroom (bytes, default) added to the response\n\
         /// body length when pre-sizing the h1 pipe-serve encode\n\
         /// buffer's initial capacity (`http1::serve_pipe`'s\n\
         /// `encode_response` / `encode_error_response`).\n\
         pub const SERVE_PIPE_ENCODE_BUFFER_HEADROOM_DEFAULT: usize = {encode_buffer_headroom};\n",
    );

    let out_path = out_dir.join("proxima_listeners_http_sized.rs");
    fs::write(&out_path, out).unwrap_or_else(|err| panic!("write {}: {err}", out_path.display()));
}

#[allow(clippy::expect_used)]
fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR set by cargo"));
    emit_sizing_consts(&out_dir);
}
