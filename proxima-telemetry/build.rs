//! Build-time sizing for proxima-telemetry.
//!
//! Reads `proxima-telemetry.toml` at the crate root + applies optional
//! per-system env-var overrides, emitting a generated constant module into
//! `OUT_DIR/proxima_telemetry_sized.rs`. The library `include!`s the generated
//! file inside a `sized` module.
//!
//! These are the compile-time DEFAULTS. The per-process surface is the runtime
//! conflaguration `TelemetryConfig` (`drain_batch` / `assist_batch`), which
//! overrides them via config file / env / fluent builder.
//!
//! Env-var override naming: `PROXIMA_TELEMETRY_<SECTION>_<KEY>`
//! (e.g. `PROXIMA_TELEMETRY_DRAIN_ASSIST_BATCH=32`). Downstream consumers that
//! re-brand their own surface map their config onto these via the runtime
//! `TelemetryConfig` API, not by setting these build vars directly.
//!
//! Also emits the `log_buffer` module's sizing (folded in from the former
//! `proxima-log-buffer` satellite crate): reads `proxima-log-buffer.toml` +
//! `PROXIMA_LOG_BUFFER_<SECTION>_<KEY>` env overrides into
//! `OUT_DIR/proxima_log_buffer_sized.rs`, unchanged from that crate's own
//! build-time contract.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use toml::Value;

fn get_int(table: &Value, section: &str, key: &str) -> u64 {
    let raw = table
        .get(section)
        .and_then(|sec| sec.get(key))
        .and_then(Value::as_integer)
        .unwrap_or_else(|| {
            panic!("proxima-telemetry.toml: missing or non-integer [{section}].{key}")
        });
    u64::try_from(raw).unwrap_or_else(|_| panic!("[{section}].{key} = {raw} must be non-negative"))
}

/// Read `(section, key)` from the TOML, then apply the optional
/// `PROXIMA_TELEMETRY_<SECTION>_<KEY>` env-var override.
fn resolve(table: &Value, section: &str, key: &str) -> u64 {
    let env_name = format!(
        "PROXIMA_TELEMETRY_{}_{}",
        section.to_ascii_uppercase(),
        key.to_ascii_uppercase()
    );
    println!("cargo:rerun-if-env-changed={env_name}");
    if let Ok(raw) = env::var(&env_name) {
        return raw
            .parse()
            .unwrap_or_else(|err| panic!("{env_name} = {raw}: {err}"));
    }
    get_int(table, section, key)
}

fn require_nonzero(name: &str, value: u64) -> u64 {
    assert!(value > 0, "{name} must be non-zero; got {value}");
    value
}

fn require_usize(name: &str, value: u64) -> usize {
    usize::try_from(value).unwrap_or_else(|_| panic!("{name} = {value} overflows usize"))
}

/// Resolve the compile-time emit floor: `[emit] max_level` (a level name) with a
/// `PROXIMA_TELEMETRY_EMIT_MAX_LEVEL` env override, mapped to proxima severity.
/// `trace` (1) keeps everything; `off` (255) compiles out all emits.
fn resolve_emit_floor(table: &Value) -> u8 {
    let env_name = "PROXIMA_TELEMETRY_EMIT_MAX_LEVEL";
    println!("cargo:rerun-if-env-changed={env_name}");
    let name = env::var(env_name).ok().or_else(|| {
        table
            .get("emit")
            .and_then(|section| section.get("max_level"))
            .and_then(Value::as_str)
            .map(str::to_string)
    });
    match name
        .unwrap_or_else(|| "trace".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "trace" => 1,
        "debug" => 5,
        "info" => 9,
        "warn" => 13,
        "error" => 17,
        "fatal" => 21,
        "off" => u8::MAX,
        other => {
            panic!("[emit].max_level = {other:?}: expected trace|debug|info|warn|error|fatal|off")
        }
    }
}

#[allow(clippy::expect_used)]
fn emit_sizing_consts(out_dir: &Path) {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set by cargo");
    let toml_path = PathBuf::from(&manifest_dir).join("proxima-telemetry.toml");
    println!("cargo:rerun-if-changed=proxima-telemetry.toml");

    let text = fs::read_to_string(&toml_path)
        .unwrap_or_else(|err| panic!("read {}: {err}", toml_path.display()));
    let root: Value = text
        .parse()
        .unwrap_or_else(|err| panic!("parse {}: {err}", toml_path.display()));

    let drain_batch = require_usize(
        "drain.batch",
        require_nonzero("drain.batch", resolve(&root, "drain", "batch")),
    );
    let assist_batch = require_usize(
        "drain.assist_batch",
        require_nonzero(
            "drain.assist_batch",
            resolve(&root, "drain", "assist_batch"),
        ),
    );
    let flush_interval_micros = require_nonzero(
        "pump.flush_interval_micros",
        resolve(&root, "pump", "flush_interval_micros"),
    );
    let emit_floor = resolve_emit_floor(&root);
    // unified-instrument compile-time defaults (the no_std+no_alloc config floor).
    let instrument_metrics = resolve(&root, "instrument", "metrics") != 0;
    let instrument_default_budget_micros = resolve(&root, "instrument", "default_budget_micros");
    let sink_capacity_bytes = require_usize(
        "sink.capacity_bytes",
        require_nonzero(
            "sink.capacity_bytes",
            resolve(&root, "sink", "capacity_bytes"),
        ),
    );
    // error-elevation caps (feature `elevation`): bound the per-trace replay
    // buffer's memory. Always emitted; consumed only under the feature.
    let elevation_max_traces = require_usize(
        "elevation.max_traces",
        require_nonzero("elevation.max_traces", resolve(&root, "elevation", "max_traces")),
    );
    let elevation_per_trace_ring = require_usize(
        "elevation.per_trace_ring",
        require_nonzero(
            "elevation.per_trace_ring",
            resolve(&root, "elevation", "per_trace_ring"),
        ),
    );

    let out = format!(
        "// AUTO-GENERATED by build.rs from proxima-telemetry.toml. DO NOT EDIT.\n\
         /// Records a background drain pass moves+exports per ring per pass (default cap).\n\
         pub const DRAIN_BATCH: usize = {drain_batch};\n\
         /// Records a producer drains+exports on a full ring before retrying its push.\n\
         pub const DRAIN_ASSIST_BATCH: usize = {assist_batch};\n\
         /// Managed pump time-trigger flush interval in microseconds (default).\n\
         pub const FLUSH_INTERVAL_MICROS: u64 = {flush_interval_micros};\n\
         /// Compile-time emit severity floor (build.rs `[emit] max_level`): a record\n\
         /// below this severity is statically disabled. 1=trace .. 21=fatal, 255=off.\n\
         pub const EMIT_COMPILE_FLOOR: u8 = {emit_floor};\n\
         /// Unified-instrument default: does a recorder consume the span-duration\n\
         /// metric by default (the consumer gate)? At no_std+no_alloc this const IS\n\
         /// the config; at std it seeds `InstrumentConfig`.\n\
         pub const INSTRUMENT_METRICS_DEFAULT: bool = {instrument_metrics};\n\
         /// Unified-instrument default tail-sampling budget (microseconds) for spans\n\
         /// without an explicit `#[span(budget)]`. 0 = none.\n\
         pub const INSTRUMENT_DEFAULT_BUDGET_MICROS: u64 = {instrument_default_budget_micros};\n\
         /// Preallocation (bytes) for the terminal batch buffer. At no_std+no_alloc\n\
         /// this const IS the reserved buffer size; at std it is the build-time default.\n\
         pub const SINK_CAPACITY_BYTES: usize = {sink_capacity_bytes};\n\
         /// Error-elevation hard count-cap: max concurrently-buffered traces before\n\
         /// the least-recently-touched trace's replay buffer is evicted (OOM backstop).\n\
         pub const ELEVATION_MAX_TRACES: usize = {elevation_max_traces};\n\
         /// Error-elevation per-trace replay ring capacity (records); DropOldest past it.\n\
         pub const ELEVATION_PER_TRACE_RING: usize = {elevation_per_trace_ring};\n",
    );

    let out_path = out_dir.join("proxima_telemetry_sized.rs");
    fs::write(&out_path, out).unwrap_or_else(|err| panic!("write {}: {err}", out_path.display()));
}

fn log_buffer_get_int(table: &Value, section: &str, key: &str) -> u64 {
    let raw = table
        .get(section)
        .and_then(|sec| sec.get(key))
        .and_then(Value::as_integer)
        .unwrap_or_else(|| {
            panic!("proxima-log-buffer.toml: missing or non-integer [{section}].{key}")
        });
    u64::try_from(raw).unwrap_or_else(|_| panic!("[{section}].{key} = {raw} must be non-negative"))
}

/// Read `(section, key)` from the log-buffer TOML, then apply the optional
/// `PROXIMA_LOG_BUFFER_<SECTION>_<KEY>` env-var override.
fn log_buffer_resolve(table: &Value, section: &str, key: &str) -> u64 {
    let env_name = format!(
        "PROXIMA_LOG_BUFFER_{}_{}",
        section.to_ascii_uppercase(),
        key.to_ascii_uppercase()
    );
    println!("cargo:rerun-if-env-changed={env_name}");
    if let Ok(raw) = env::var(&env_name) {
        return raw
            .parse()
            .unwrap_or_else(|err| panic!("{env_name} = {raw}: {err}"));
    }
    log_buffer_get_int(table, section, key)
}

#[allow(clippy::expect_used)]
fn emit_log_buffer_sizing_consts(out_dir: &Path) {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set by cargo");
    let toml_path = PathBuf::from(&manifest_dir).join("proxima-log-buffer.toml");
    println!("cargo:rerun-if-changed=proxima-log-buffer.toml");

    let text = fs::read_to_string(&toml_path)
        .unwrap_or_else(|err| panic!("read {}: {err}", toml_path.display()));
    let root: Value = text
        .parse()
        .unwrap_or_else(|err| panic!("parse {}: {err}", toml_path.display()));

    let capacity = require_usize(
        "buffer.capacity",
        require_nonzero(
            "buffer.capacity",
            log_buffer_resolve(&root, "buffer", "capacity"),
        ),
    );
    let live_tail_channel_capacity = require_usize(
        "live_tail.channel_capacity",
        require_nonzero(
            "live_tail.channel_capacity",
            log_buffer_resolve(&root, "live_tail", "channel_capacity"),
        ),
    );

    let out = format!(
        "// AUTO-GENERATED by build.rs from proxima-log-buffer.toml. DO NOT EDIT.\n\
         /// Retained ring-buffer capacity (lines) for a `LogBuffer` (default cap).\n\
         /// At no_std+no_alloc this const IS the ring's reserved size; at std it\n\
         /// seeds `LogBufferConfig::capacity`.\n\
         pub const LOG_BUFFER_CAPACITY_DEFAULT: usize = {capacity};\n\
         /// Per-subscriber live-tail queue capacity (default cap). At\n\
         /// no_std+no_alloc this const IS the queue's reserved size; at std it\n\
         /// seeds `LogBufferConfig::live_tail_channel_capacity`.\n\
         pub const LIVE_TAIL_CHANNEL_CAPACITY_DEFAULT: usize = {live_tail_channel_capacity};\n",
    );

    let out_path = out_dir.join("proxima_log_buffer_sized.rs");
    fs::write(&out_path, out).unwrap_or_else(|err| panic!("write {}: {err}", out_path.display()));
}

#[allow(clippy::expect_used)]
fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR set by cargo"));
    emit_sizing_consts(&out_dir);
    emit_log_buffer_sizing_consts(&out_dir);
}
