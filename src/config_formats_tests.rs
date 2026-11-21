#![allow(clippy::expect_used)]

use std::fmt::Debug;

use proxima_config::{ConfigFormatFactory, TomlConfigFormat};
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
