#![allow(clippy::expect_used)]

use std::fmt::Debug;

#[cfg(feature = "tokio")]
use crate::pipelines::spec::StageSpec;
use crate::schema::Schema;
use crate::settings::ProximaSettings;
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
#[cfg(feature = "recording")]
use crate::upstreams::record::RecordConfig;
#[cfg(any(feature = "tcp", feature = "unix"))]
use crate::upstreams::stream_passthrough::StreamPassthroughSettings;
use crate::upstreams::synth::SynthConfig;
#[cfg(feature = "vm")]
use crate::upstreams::vm::VmConfig;
#[cfg(all(
    feature = "amqp-client",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
use proxima_amqp::AmqpClientConfig;
use proxima_config::{ConfigFormatFactory, TomlConfigFormat};
#[cfg(all(
    feature = "dns-client",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
use proxima_dns::DnsResolverConfig;
#[cfg(feature = "http1")]
use proxima_http::http1::HttpConfig;
#[cfg(all(
    feature = "kafka-client",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
use proxima_kafka::KafkaClientConfig;
#[cfg(all(
    feature = "memcached-client",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
use proxima_memcached::MemcachedClientConfig;
#[cfg(all(
    feature = "mqtt-client",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
use proxima_mqtt::MqttClientConfig;
use proxima_patterns::middleware::auth::AuthConfig;
use proxima_patterns::middleware::client_auth::ClientAuthConfig;
#[cfg(all(
    feature = "pgwire-client",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
use proxima_pgwire::PgClientConfig;
use proxima_primitives::pipe::delay::DelayConfig;
use proxima_primitives::pipe::filter::FilterConfig;
use proxima_primitives::pipe::isolate::IsolateConfig;
use proxima_primitives::pipe::rate_limit::RateLimitConfig;
use proxima_primitives::pipe::retry::RetryConfig;
use proxima_primitives::pipe::transform::TransformConfig;
use proxima_primitives::pipe::validate::ValidateConfig;
#[cfg(all(
    feature = "redis-client",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
use proxima_redis::RedisClientConfig;
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
fn ctl_format_member_retry_config() {
    assert_serialized_twin::<RetryConfig>(
        "retry_config",
        fixture!("retry_config", "toml"),
        fixture!("retry_config", "expected.json"),
    );
}

#[test]
fn ctl_format_member_transform_config() {
    assert_serialized_twin::<TransformConfig>(
        "transform_config",
        fixture!("transform_config", "toml"),
        fixture!("transform_config", "expected.json"),
    );
}

#[test]
fn ctl_format_member_rate_limit_config() {
    assert_serialized_twin::<RateLimitConfig>(
        "rate_limit_config",
        fixture!("rate_limit_config", "toml"),
        fixture!("rate_limit_config", "expected.json"),
    );
}

#[test]
fn ctl_format_member_client_auth_config() {
    assert_serialized_twin::<ClientAuthConfig>(
        "client_auth_config",
        fixture!("client_auth_config", "toml"),
        fixture!("client_auth_config", "expected.json"),
    );
}

#[test]
fn ctl_format_member_auth_config() {
    assert_serialized_twin::<AuthConfig>(
        "auth_config",
        fixture!("auth_config", "toml"),
        fixture!("auth_config", "expected.json"),
    );
}

#[test]
fn ctl_format_member_validate_config() {
    assert_serialized_twin::<ValidateConfig>(
        "validate_config",
        fixture!("validate_config", "toml"),
        fixture!("validate_config", "expected.json"),
    );
}

#[test]
fn ctl_format_member_filter_config() {
    assert_typed_twin::<FilterConfig>(
        "filter_config",
        fixture!("filter_config", "toml"),
        fixture!("filter_config", "expected.json"),
    );
}

#[test]
fn ctl_format_member_delay_config() {
    assert_typed_twin::<DelayConfig>(
        "delay_config",
        fixture!("delay_config", "toml"),
        fixture!("delay_config", "expected.json"),
    );
}

#[test]
fn ctl_format_member_isolate_config() {
    assert_serialized_twin::<IsolateConfig>(
        "isolate_config",
        fixture!("isolate_config", "toml"),
        fixture!("isolate_config", "expected.json"),
    );
}

#[test]
#[cfg(feature = "http1")]
fn ctl_format_member_http_config() {
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
fn ctl_format_member_h3_native_config() {
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
fn ctl_format_member_grpc_config() {
    assert_serialized_twin::<GrpcConfig>(
        "grpc_config",
        fixture!("grpc_config", "toml"),
        fixture!("grpc_config", "expected.json"),
    );
}

#[test]
fn ctl_format_member_synth_config() {
    assert_serialized_twin::<SynthConfig>(
        "synth_config",
        fixture!("synth_config", "toml"),
        fixture!("synth_config", "expected.json"),
    );
}

#[test]
fn ctl_format_member_callback_config() {
    assert_serialized_twin::<CallbackConfig>(
        "callback_config",
        fixture!("callback_config", "toml"),
        fixture!("callback_config", "expected.json"),
    );
}

#[test]
#[cfg(feature = "tokio")]
fn ctl_format_member_process_config() {
    assert_serialized_twin::<ProcessConfig>(
        "process_config",
        fixture!("process_config", "toml"),
        fixture!("process_config", "expected.json"),
    );
}

#[test]
#[cfg(unix)]
fn ctl_format_member_process_rpc_config() {
    assert_serialized_twin::<ProcessRpcConfig>(
        "process_rpc_config",
        fixture!("process_rpc_config", "toml"),
        fixture!("process_rpc_config", "expected.json"),
    );
}

#[test]
fn ctl_format_member_fs_config() {
    assert_serialized_twin::<FsConfig>(
        "fs_config",
        fixture!("fs_config", "toml"),
        fixture!("fs_config", "expected.json"),
    );
}

#[test]
fn ctl_format_member_kv_cache_config() {
    assert_serialized_twin::<KvCacheConfig>(
        "kv_cache_config",
        fixture!("kv_cache_config", "toml"),
        fixture!("kv_cache_config", "expected.json"),
    );
}

#[test]
fn ctl_format_member_kv_file_config() {
    assert_serialized_twin::<KvFileConfig>(
        "kv_file_config",
        fixture!("kv_file_config", "toml"),
        fixture!("kv_file_config", "expected.json"),
    );
}

#[test]
#[cfg(any(feature = "tcp", feature = "unix"))]
fn ctl_format_member_stream_passthrough_settings() {
    assert_typed_twin::<StreamPassthroughSettings>(
        "stream_passthrough_settings",
        fixture!("stream_passthrough_settings", "toml"),
        fixture!("stream_passthrough_settings", "expected.json"),
    );
}

#[test]
#[cfg(feature = "vm")]
fn ctl_format_member_vm_config() {
    assert_typed_twin::<VmConfig>(
        "vm_config",
        fixture!("vm_config", "toml"),
        fixture!("vm_config", "expected.json"),
    );
}

#[test]
#[cfg(feature = "recording")]
fn ctl_format_member_record_config() {
    assert_serialized_twin::<RecordConfig>(
        "record_config",
        fixture!("record_config", "toml"),
        fixture!("record_config", "expected.json"),
    );
}

#[test]
#[cfg(all(
    feature = "mqtt-client",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
fn ctl_format_member_mqtt_client_config() {
    assert_typed_twin::<MqttClientConfig>(
        "mqtt_client_config",
        fixture!("mqtt_client_config", "toml"),
        fixture!("mqtt_client_config", "expected.json"),
    );
}

#[test]
#[cfg(all(
    feature = "dns-client",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
fn ctl_format_member_dns_resolver_config() {
    assert_typed_twin::<DnsResolverConfig>(
        "dns_resolver_config",
        fixture!("dns_resolver_config", "toml"),
        fixture!("dns_resolver_config", "expected.json"),
    );
}

#[test]
#[cfg(all(
    feature = "pgwire-client",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
fn ctl_format_member_pg_client_config() {
    assert_typed_twin::<PgClientConfig>(
        "pg_client_config",
        fixture!("pg_client_config", "toml"),
        fixture!("pg_client_config", "expected.json"),
    );
}

#[test]
#[cfg(all(
    feature = "amqp-client",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
fn ctl_format_member_amqp_client_config() {
    assert_typed_twin::<AmqpClientConfig>(
        "amqp_client_config",
        fixture!("amqp_client_config", "toml"),
        fixture!("amqp_client_config", "expected.json"),
    );
}

#[test]
#[cfg(all(
    feature = "redis-client",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
fn ctl_format_member_redis_client_config() {
    assert_typed_twin::<RedisClientConfig>(
        "redis_client_config",
        fixture!("redis_client_config", "toml"),
        fixture!("redis_client_config", "expected.json"),
    );
}

#[test]
#[cfg(all(
    feature = "memcached-client",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
fn ctl_format_member_memcached_client_config() {
    assert_typed_twin::<MemcachedClientConfig>(
        "memcached_client_config",
        fixture!("memcached_client_config", "toml"),
        fixture!("memcached_client_config", "expected.json"),
    );
}

#[test]
#[cfg(all(
    feature = "kafka-client",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
fn ctl_format_member_kafka_client_config() {
    assert_typed_twin::<KafkaClientConfig>(
        "kafka_client_config",
        fixture!("kafka_client_config", "toml"),
        fixture!("kafka_client_config", "expected.json"),
    );
}

#[test]
#[cfg(feature = "tokio")]
fn ctl_format_member_stage_spec() {
    assert_typed_twin::<StageSpec>(
        "stage_spec",
        fixture!("stage_spec", "toml"),
        fixture!("stage_spec", "expected.json"),
    );
}

#[test]
fn ctl_format_member_schema() {
    assert_typed_twin::<Schema>(
        "schema",
        fixture!("schema", "toml"),
        fixture!("schema", "expected.json"),
    );
}

#[test]
fn ctl_format_member_proxima_settings() {
    assert_serialized_twin::<ProximaSettings>(
        "proxima_settings",
        fixture!("proxima_settings", "toml"),
        fixture!("proxima_settings", "expected.json"),
    );
}

#[test]
fn ctl_format_read_max_bytes_typed() {
    let value = toml_value(
        "max_bytes",
        include_str!("../tests/fixtures/config_formats/reads/max_bytes_typed.toml"),
    );
    let parsed = crate::upstreams::kv_cache::parse_size_value(&value["max_bytes"]);
    assert_eq!(parsed.expect("some").expect("ok"), 67_108_864);
}

// the built source hides its period, so this control observes only that
// `period_ms = 250` builds
#[test]
fn ctl_format_read_period_ms_typed() {
    use proxima_primitives::pipe::SourceFactory;

    let value = toml_value(
        "period_ms",
        include_str!("../tests/fixtures/config_formats/reads/period_ms_typed.toml"),
    );
    let inner =
        crate::pipe::into_handle(crate::upstreams::synth::SynthUpstream::new("tick", 200, ""));
    let factory =
        proxima_primitives::pipe::interval_pipe::IntervalSourceFactory::new("tick", inner);
    factory
        .build(&value)
        .map(|_| ())
        .expect("period_ms = 250 builds an interval source");
}

#[proxima::test]
async fn ctl_format_client_toml() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/client_config/client.toml");
    let client = crate::client::Client::from_path(path)
        .await
        .expect("client.toml loads");
    let response = client.get("/").send().await.expect("one request");
    assert_eq!(response.status(), 201);
    let body = response.bytes().await.expect("body");
    assert_eq!(&body[..], b"created");
}
