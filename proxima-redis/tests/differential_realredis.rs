//! Live differential parity against the canonical incumbents — real `redis:7`
//! AND real `valkey` — driven by proxima's OWN client
//! ([`proxima_redis::RedisClient`]), no `redis` crate. The same client running
//! the same command script against both servers must observe an identical RESP
//! contract (principle 14: parity vs the canonical incumbent, and Valkey is the
//! same wire protocol — one client covers both).
//!
//! Requires reachable servers (principle 15 legitimate-deferral cat. 2: external
//! infra). Set `REDIS_REAL_HOST` (+ `_PORT`) and `VALKEY_REAL_HOST` (+ `_PORT`);
//! CI provides both via service containers. Absent either, the test logs why it
//! skips and returns; it is never `#[ignore]`'d.

#![cfg(feature = "client")]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::net::TcpStream;

use proxima_redis::{RedisClient, RedisClientConfig, RespValue};

/// A fixed, deterministic command script (the keys are DEL'd first so the
/// replies do not depend on prior state). Returns the reply to each command.
fn probe_script(host: &str, port: u16) -> Vec<RespValue> {
    let config = RedisClientConfig::builder()
        .host(host.to_string())
        .port(port)
        .resp3(false)
        .build();
    let stream = TcpStream::connect((host, port)).expect("connect");
    stream.set_nodelay(true).expect("nodelay");
    let mut client = RedisClient::connect(stream, &config).expect("handshake");

    // prelude: clean state so INCR/RPUSH replies are deterministic (uncompared).
    client
        .command(&[
            b"DEL",
            b"proxima:diff:s",
            b"proxima:diff:n",
            b"proxima:diff:l",
        ])
        .expect("del");

    let script: &[&[&[u8]]] = &[
        &[b"SET", b"proxima:diff:s", b"hello"],
        &[b"GET", b"proxima:diff:s"],
        &[b"GET", b"proxima:diff:absent"],
        &[b"INCR", b"proxima:diff:n"],
        &[b"INCRBY", b"proxima:diff:n", b"41"],
        &[b"RPUSH", b"proxima:diff:l", b"a", b"b", b"c"],
        &[b"LRANGE", b"proxima:diff:l", b"0", b"-1"],
        &[b"TYPE", b"proxima:diff:l"],
    ];
    let replies = script
        .iter()
        .map(|argv| client.command(argv).expect("command"))
        .collect();
    let _ = client.close();
    replies
}

fn error_reply(host: &str, port: u16) -> RespValue {
    let config = RedisClientConfig::builder()
        .host(host.to_string())
        .port(port)
        .resp3(false)
        .build();
    let stream = TcpStream::connect((host, port)).expect("connect");
    let mut client = RedisClient::connect(stream, &config).expect("handshake");
    let reply = client
        .command(&[b"FLIBBERTIGIBBET"])
        .expect("unknown command");
    let _ = client.close();
    reply
}

fn endpoints() -> Option<((String, u16), (String, u16))> {
    let redis_host = std::env::var("REDIS_REAL_HOST")
        .ok()
        .filter(|host| !host.is_empty())?;
    let valkey_host = std::env::var("VALKEY_REAL_HOST")
        .ok()
        .filter(|host| !host.is_empty())?;
    let redis_port = std::env::var("REDIS_REAL_PORT")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(6379);
    let valkey_port = std::env::var("VALKEY_REAL_PORT")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(6379);
    Some(((redis_host, redis_port), (valkey_host, valkey_port)))
}

#[test]
fn redis_and_valkey_answer_identically() {
    let Some(((redis_host, redis_port), (valkey_host, valkey_port))) = endpoints() else {
        eprintln!(
            "skipping differential_realredis: REDIS_REAL_HOST / VALKEY_REAL_HOST unset (no servers). \
             CI provides redis:7 + valkey services; locally run both in docker and set them."
        );
        return;
    };

    let redis = probe_script(&redis_host, redis_port);
    let valkey = probe_script(&valkey_host, valkey_port);
    assert_eq!(
        redis, valkey,
        "redis and valkey must answer the deterministic script identically via our client"
    );

    // sanity: the script actually exercised the value shapes we claim.
    assert_eq!(redis[0], RespValue::SimpleString("OK".into()));
    assert_eq!(redis[1], RespValue::BulkString(b"hello".to_vec()));
    assert!(redis[2].is_null());
    assert_eq!(redis[3], RespValue::Integer(1));
    assert_eq!(redis[4], RespValue::Integer(42));
    assert_eq!(redis[6].as_array().map(<[_]>::len), Some(3));

    // both surface an error for an unknown command (the exact prose can drift
    // between server families, so parity here is "both error on it").
    assert!(error_reply(&redis_host, redis_port).as_error().is_some());
    assert!(error_reply(&valkey_host, valkey_port).as_error().is_some());
}
