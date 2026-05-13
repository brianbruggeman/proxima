//! The redis client `Pipe` ([`RedisClientUpstream`]) driven over a real
//! `StreamUpstream` (tokio) against real Redis/Valkey — proving the async client
//! path that `proxima::Client` reaches through the `PipeFactory`. Mirrors
//! pgwire's `client_pipe_realpg`. Env-gated on a reachable server; skips with a
//! logged reason locally, never `#[ignore]`'d.

#![cfg(feature = "client")]
#![allow(clippy::expect_used)]

use std::net::SocketAddr;

use proxima_net::tokio::tokio_stream_upstream::TokioTcpUpstream;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::request::Request;
use proxima_redis::{RedisClientConfig, RedisClientUpstream, RespValue, verb};

fn decode_reply(payload: &[u8]) -> RespValue {
    let (frame, _consumed) =
        proxima_redis::parse(payload).expect("response payload must be valid RESP");
    RespValue::from_frame(&frame)
}

#[proxima::test(flavor = "multi_thread", worker_threads = 2)]
async fn client_pipe_set_get_against_real_redis() {
    let host = match std::env::var("REDIS_REAL_HOST") {
        Ok(host) if !host.is_empty() => host,
        _ => {
            eprintln!(
                "skipping client_pipe_realredis: REDIS_REAL_HOST unset (no server). \
                 CI provides a redis service; locally run docker redis:7 and set it."
            );
            return;
        }
    };
    let port: u16 = std::env::var("REDIS_REAL_PORT")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(6379);
    let addr: SocketAddr = format!("{host}:{port}").parse().expect("ip:port");
    let config = RedisClientConfig::builder().host(host).port(port).build();
    let client = RedisClientUpstream::new(TokioTcpUpstream::new(addr), config);

    // SET via NUL-delimited body args (verb in method, args in body).
    let set = Request::builder()
        .method(verb::SET)
        .path("")
        .body("proxima:pipe\0world")
        .build()
        .expect("request");
    let response = client.call(set).await.expect("set");
    assert_eq!(
        decode_reply(&response.payload),
        RespValue::SimpleString("OK".into())
    );

    // the cached connection is reused (no re-handshake): GET via the body arg.
    let get = Request::builder()
        .method(verb::GET)
        .path("")
        .body("proxima:pipe")
        .build()
        .expect("request");
    let response = client.call(get).await.expect("get");
    assert_eq!(
        decode_reply(&response.payload),
        RespValue::BulkString(b"world".to_vec())
    );

    // a server error reply stays transport-Ok and the connection survives.
    let bad = Request::builder()
        .method("FLIBBERTIGIBBET")
        .path("")
        .build()
        .expect("request");
    let response = client.call(bad).await.expect("error is transport-ok");
    assert!(
        decode_reply(&response.payload).as_error().is_some(),
        "unknown command -> error reply"
    );

    // still usable after the error.
    let get = Request::builder()
        .method(verb::GET)
        .path("")
        .body("proxima:pipe")
        .build()
        .expect("request");
    let response = client.call(get).await.expect("get after error");
    assert_eq!(
        decode_reply(&response.payload),
        RespValue::BulkString(b"world".to_vec())
    );
}
