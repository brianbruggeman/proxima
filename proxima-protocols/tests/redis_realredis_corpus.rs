//! Vendored real-Redis byte corpus (principle 16: vectors live in the repo).
//!
//! The fixtures under `tests/fixtures/realredis/*.bin` are the *actual* bytes a
//! real `redis:7` server sent proxima's own client (captured by the
//! `capture_realredis` example). This test re-proves on every build, with no
//! server required, that:
//!
//! 1. our parser fully consumes each real server stream (principle 14: the
//!    canonical incumbent is the oracle), and
//! 2. an independent implementation — the `redis-protocol` crate — agrees on the
//!    framing (same frame count, same total bytes consumed). Two implementations
//!    agreeing on real bytes is the parity bar.
//!
//! CI re-captures from a live `redis:7` / `valkey` service into a fresh dir and
//! re-runs this test with `REDIS_REAL_FIXTURES` pointed at it, so a structural
//! drift in a newer point release surfaces as a version-drift canary.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::PathBuf;

use proxima_protocols::redis::{RespValue, parse};

fn fixture_dir() -> PathBuf {
    std::env::var("REDIS_REAL_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/realredis")
        })
}

fn read(name: &str) -> Vec<u8> {
    let path = fixture_dir().join(format!("{name}.bin"));
    std::fs::read(&path).unwrap_or_else(|err| panic!("read fixture {}: {err}", path.display()))
}

/// Parse every frame in `bytes` with proxima, asserting the stream is fully
/// consumed (no trailing partial frame), and return the owned values.
fn parse_all(bytes: &[u8]) -> Vec<RespValue> {
    let mut values = Vec::new();
    let mut cursor = 0;
    while cursor < bytes.len() {
        let (frame, used) = parse(&bytes[cursor..])
            .unwrap_or_else(|err| panic!("proxima parse at offset {cursor}: {err:?}"));
        values.push(RespValue::from_frame(&frame));
        cursor += used;
    }
    assert_eq!(cursor, bytes.len(), "proxima must consume the whole stream");
    values
}

/// Count frames + total bytes the `redis-protocol` crate consumes, as an
/// independent framing oracle. RESP3 fixtures decode with the resp3 codec;
/// RESP2 fixtures (legacy `$-1` null) with the resp2 codec.
fn redis_protocol_frames(bytes: &[u8], resp3: bool) -> (usize, usize) {
    let mut count = 0;
    let mut cursor = 0;
    while cursor < bytes.len() {
        let used = if resp3 {
            let (_, used) = redis_protocol::resp3::decode::complete::decode(&bytes[cursor..])
                .expect("redis-protocol resp3 decode")
                .expect("redis-protocol resp3 needs more (corpus is complete)");
            used
        } else {
            let (_, used) = redis_protocol::resp2::decode::decode(&bytes[cursor..])
                .expect("redis-protocol resp2 decode")
                .expect("redis-protocol resp2 needs more (corpus is complete)");
            used
        };
        count += 1;
        cursor += used;
    }
    (count, cursor)
}

/// Both implementations agree on the framing of a real server stream.
fn assert_parity(name: &str, resp3: bool) {
    let bytes = read(name);
    let proxima = parse_all(&bytes);
    let (oracle_count, oracle_consumed) = redis_protocol_frames(&bytes, resp3);
    assert_eq!(
        proxima.len(),
        oracle_count,
        "{name}: frame count must match the oracle"
    );
    assert_eq!(
        oracle_consumed,
        bytes.len(),
        "{name}: oracle must consume the whole stream"
    );
}

#[test]
fn hello3_is_a_real_server_property_map() {
    let values = parse_all(&read("hello3"));
    assert_eq!(values.len(), 1, "HELLO 3 answers with exactly one map");
    let RespValue::Map(pairs) = &values[0] else {
        panic!("expected a RESP3 map, got {:?}", values[0]);
    };
    let server = pairs
        .iter()
        .find_map(|(key, value)| (key.as_str() == Some("server")).then_some(value));
    assert_eq!(server.and_then(RespValue::as_str), Some("redis"));
    assert_parity("hello3", true);
}

#[test]
fn string_ops_set_get_then_miss() {
    let values = parse_all(&read("string_ops"));
    assert_eq!(
        values[0],
        RespValue::SimpleString("OK".into()),
        "SET -> +OK"
    );
    assert_eq!(
        values[1],
        RespValue::BulkString(b"hello".to_vec()),
        "GET hit -> $hello"
    );
    assert!(values[2].is_null(), "GET miss -> null ($-1)");
    assert_parity("string_ops", false);
}

#[test]
fn numeric_integer_replies() {
    let values = parse_all(&read("numeric"));
    assert_eq!(
        values,
        vec![
            RespValue::Integer(0),
            RespValue::Integer(1),
            RespValue::Integer(42)
        ]
    );
    assert_parity("numeric", false);
}

#[test]
fn list_array_reply() {
    let values = parse_all(&read("list"));
    let array = values
        .last()
        .and_then(RespValue::as_array)
        .expect("LRANGE -> array");
    assert_eq!(array.len(), 3);
    assert_eq!(array[0], RespValue::BulkString(b"a".to_vec()));
    assert_eq!(array[2], RespValue::BulkString(b"c".to_vec()));
    assert_parity("list", false);
}

#[test]
fn error_reply_is_surfaced() {
    let values = parse_all(&read("error"));
    assert_eq!(values.len(), 1);
    assert!(
        values[0]
            .as_error()
            .expect("error reply")
            .contains("unknown command")
    );
    assert_parity("error", false);
}
