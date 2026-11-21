#![allow(clippy::expect_used)]

use std::fmt::Debug;

use crate::upstreams::callback::CallbackConfig;
use crate::upstreams::fs::FsConfig;
#[cfg(all(
    feature = "http-prime",
    feature = "http2",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
use crate::upstreams::grpc_h2::GrpcConfig;
#[cfg(all(
    feature = "h3-native-upstream",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
use crate::upstreams::h3_native::H3NativeConfig;
use crate::upstreams::kv_cache::KvCacheConfig;
use crate::upstreams::kv_file::KvFileConfig;
#[cfg(feature = "tokio")]
use crate::upstreams::process::ProcessConfig;
#[cfg(unix)]
use crate::upstreams::process_rpc::ProcessRpcConfig;
use crate::upstreams::synth::SynthConfig;
use proxima_config::{ConfigFormatFactory, TomlConfigFormat};
#[cfg(feature = "http-prime-deps")]
use proxima_http::http1::HttpConfig;
use proxima_patterns::middleware::auth::AuthConfig;
use proxima_patterns::middleware::client_auth::ClientAuthConfig;
use proxima_primitives::pipe::delay::DelayConfig;
use proxima_primitives::pipe::filter::FilterConfig;
use proxima_primitives::pipe::isolate::IsolateConfig;
use proxima_primitives::pipe::rate_limit::RateLimitConfig;
use proxima_primitives::pipe::retry::RetryConfig;
use proxima_primitives::pipe::transform::TransformConfig;
use proxima_primitives::pipe::validate::ValidateConfig;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

fn toml_value(member: &str, raw: &str) -> Value {
    TomlConfigFormat
        .parse(raw)
        .unwrap_or_else(|err| panic!("{member}: toml parse failed: {err}"))
}

fn expected_value(member: &str, raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|err| panic!("{member}: expected.json invalid: {err}"))
}

fn today_read<T: DeserializeOwned>(member: &str, raw: &str) -> T {
    serde_json::from_value::<T>(toml_value(member, raw))
        .unwrap_or_else(|err| panic!("{member}: from_value failed: {err}"))
}

fn assert_serialized_twin<T: DeserializeOwned + Serialize>(
    member: &str,
    toml_raw: &str,
    expected_raw: &str,
) {
    let config: T = today_read(member, toml_raw);
    let actual = serde_json::to_value(&config).expect("config serializes to a value");
    assert_eq!(
        actual,
        expected_value(member, expected_raw),
        "{member}: to_value differs from expected.json"
    );
}

fn assert_typed_twin<T: DeserializeOwned + PartialEq + Debug>(
    member: &str,
    toml_raw: &str,
    expected_raw: &str,
) {
    let actual: T = today_read(member, toml_raw);
    let expected: T = serde_json::from_value(expected_value(member, expected_raw))
        .unwrap_or_else(|err| panic!("{member}: expected.json does not read as the type: {err}"));
    assert_eq!(
        actual, expected,
        "{member}: typed value differs from expected.json"
    );
}

macro_rules! fixture {
    ($name:literal, $ext:literal) => {
        include_str!(concat!(
            "../tests/fixtures/config_formats/types/",
            $name,
            ".",
            $ext
        ))
    };
}

#[test]
fn ctl_format_twin_retry_config() {
    assert_serialized_twin::<RetryConfig>(
        "retry_config",
        fixture!("retry_config", "toml"),
        fixture!("retry_config", "expected.json"),
    );
}

#[test]
fn ctl_format_twin_transform_config() {
    assert_serialized_twin::<TransformConfig>(
        "transform_config",
        fixture!("transform_config", "toml"),
        fixture!("transform_config", "expected.json"),
    );
}

#[test]
fn ctl_format_twin_rate_limit_config() {
    assert_serialized_twin::<RateLimitConfig>(
        "rate_limit_config",
        fixture!("rate_limit_config", "toml"),
        fixture!("rate_limit_config", "expected.json"),
    );
}

#[test]
fn ctl_format_twin_client_auth_config() {
    assert_serialized_twin::<ClientAuthConfig>(
        "client_auth_config",
        fixture!("client_auth_config", "toml"),
        fixture!("client_auth_config", "expected.json"),
    );
}

#[test]
fn ctl_format_twin_auth_config() {
    assert_serialized_twin::<AuthConfig>(
        "auth_config",
        fixture!("auth_config", "toml"),
        fixture!("auth_config", "expected.json"),
    );
}

#[test]
fn ctl_format_twin_validate_config() {
    assert_serialized_twin::<ValidateConfig>(
        "validate_config",
        fixture!("validate_config", "toml"),
        fixture!("validate_config", "expected.json"),
    );
}

#[test]
fn ctl_format_twin_filter_config() {
    assert_typed_twin::<FilterConfig>(
        "filter_config",
        fixture!("filter_config", "toml"),
        fixture!("filter_config", "expected.json"),
    );
}

#[test]
fn ctl_format_twin_delay_config() {
    assert_typed_twin::<DelayConfig>(
        "delay_config",
        fixture!("delay_config", "toml"),
        fixture!("delay_config", "expected.json"),
    );
}

#[test]
fn ctl_format_twin_isolate_config() {
    assert_serialized_twin::<IsolateConfig>(
        "isolate_config",
        fixture!("isolate_config", "toml"),
        fixture!("isolate_config", "expected.json"),
    );
}

#[test]
#[cfg(feature = "http-prime-deps")]
fn ctl_format_twin_http_config() {
    assert_serialized_twin::<HttpConfig>(
        "http_config",
        fixture!("http_config", "toml"),
        fixture!("http_config", "expected.json"),
    );
}

#[test]
#[cfg(all(
    feature = "h3-native-upstream",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
fn ctl_format_twin_h3_native_config() {
    assert_serialized_twin::<H3NativeConfig>(
        "h3_native_config",
        fixture!("h3_native_config", "toml"),
        fixture!("h3_native_config", "expected.json"),
    );
}

#[test]
#[cfg(all(
    feature = "http-prime",
    feature = "http2",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
fn ctl_format_twin_grpc_config() {
    assert_serialized_twin::<GrpcConfig>(
        "grpc_config",
        fixture!("grpc_config", "toml"),
        fixture!("grpc_config", "expected.json"),
    );
}

#[test]
fn ctl_format_twin_synth_config() {
    assert_serialized_twin::<SynthConfig>(
        "synth_config",
        fixture!("synth_config", "toml"),
        fixture!("synth_config", "expected.json"),
    );
}

#[test]
fn ctl_format_twin_callback_config() {
    assert_serialized_twin::<CallbackConfig>(
        "callback_config",
        fixture!("callback_config", "toml"),
        fixture!("callback_config", "expected.json"),
    );
}

#[test]
#[cfg(feature = "tokio")]
fn ctl_format_twin_process_config() {
    assert_serialized_twin::<ProcessConfig>(
        "process_config",
        fixture!("process_config", "toml"),
        fixture!("process_config", "expected.json"),
    );
}

#[test]
#[cfg(unix)]
fn ctl_format_twin_process_rpc_config() {
    assert_serialized_twin::<ProcessRpcConfig>(
        "process_rpc_config",
        fixture!("process_rpc_config", "toml"),
        fixture!("process_rpc_config", "expected.json"),
    );
}

#[test]
fn ctl_format_twin_fs_config() {
    assert_serialized_twin::<FsConfig>(
        "fs_config",
        fixture!("fs_config", "toml"),
        fixture!("fs_config", "expected.json"),
    );
}

#[test]
fn ctl_format_twin_kv_cache_config() {
    assert_serialized_twin::<KvCacheConfig>(
        "kv_cache_config",
        fixture!("kv_cache_config", "toml"),
        fixture!("kv_cache_config", "expected.json"),
    );
}

#[test]
fn ctl_format_twin_kv_file_config() {
    assert_serialized_twin::<KvFileConfig>(
        "kv_file_config",
        fixture!("kv_file_config", "toml"),
        fixture!("kv_file_config", "expected.json"),
    );
}
