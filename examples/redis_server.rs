//! P17 sans-IO walkthrough: a real client conversation exercising every
//! legal `Connection` FSM transition —
//!
//! `PING` -> `SUBSCRIBE` (enters `Subscriber` mode) -> a gated `GET` rejected
//! while subscribed -> `PUBLISH` from a SECOND connection -> the pushed
//! `message` frame delivered on the first -> `UNSUBSCRIBE` (back to
//! `Command` mode) -> `GET` succeeding again.
//!
//! The server is `Listener::builder().redis(handler).bind(addr).serve()` —
//! the `.redis(handler)` axis resolves onto a single-candidate
//! `AnyListenProtocol` wrapping `RedisAnyProtocol`, the SAME bind + accept
//! loop (real `ListenerCore`/`ConnAdmission` admission, graceful drain)
//! every other TCP-stream listener now shares. The business handler is a
//! tiny in-memory KV store — GET/SET/DEL — everything else (PING/
//! SUBSCRIBE/PSUBSCRIBE/UNSUBSCRIBE/PUBLISH) is answered by the driver
//! itself, never reaching the handler.
//!
//! Run with: `cargo run -p proxima --example redis_server --features redis-listener`

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use proxima::error::ProximaError;
use proxima::listeners::redis::RespValue;
use proxima::{Listener, ListenerBuilderEntry, ListenerProtocolExt};
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::request::Response;
use proxima_redis::{RedisPipeReply, RedisPipeRequest};

/// A tiny in-memory KV store — the business handler the driver dispatches
/// every non-pub/sub command to. GET/SET/DEL only; anything else answers
/// `ERR unknown command`. Lock poisoning recovers rather than panicking
/// (`unwrap_or_else(PoisonError::into_inner)`) — an example still follows
/// the no-`unwrap`/no-`expect` house rule.
#[derive(Default, Clone)]
struct KvStore {
    data: Arc<Mutex<HashMap<Vec<u8>, Vec<u8>>>>,
}

impl SendPipe for KvStore {
    type In = RedisPipeRequest;
    type Out = RedisPipeReply;
    type Err = ProximaError;

    fn call(
        &self,
        request: RedisPipeRequest,
    ) -> impl core::future::Future<Output = Result<RedisPipeReply, ProximaError>> + Send {
        let store = self.data.clone();
        async move {
            let args = &request.payload.args;
            let reply = match request.method.as_bytes() {
                b"GET" => {
                    let key = args.first().cloned().unwrap_or_default();
                    let guard = store.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                    match guard.get(&key) {
                        Some(value) => RespValue::BulkString(value.clone()),
                        None => RespValue::Null,
                    }
                }
                b"SET" => {
                    let key = args.first().cloned().unwrap_or_default();
                    let value = args.get(1).cloned().unwrap_or_default();
                    let mut guard = store.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                    guard.insert(key, value);
                    RespValue::SimpleString("OK".to_string())
                }
                b"DEL" => {
                    let key = args.first().cloned().unwrap_or_default();
                    let mut guard = store.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                    let removed = guard.remove(&key).is_some();
                    RespValue::Integer(i64::from(removed))
                }
                other => RespValue::Error(format!(
                    "ERR unknown command '{}'",
                    String::from_utf8_lossy(other)
                )),
            };
            Ok(Response::typed(200, reply))
        }
    }
}

async fn send_command(stream: &mut TcpStream, args: &[&[u8]]) -> Result<(), ProximaError> {
    let mut wire = Vec::new();
    proxima_redis::encode_command(args, &mut wire);
    stream.write_all(&wire).await?;
    Ok(())
}

/// Reads bytes off `stream` into `buffered` until one full RESP frame is
/// available, then returns its owned `RespValue` and drains the consumed
/// prefix.
async fn read_reply(
    stream: &mut TcpStream,
    buffered: &mut Vec<u8>,
) -> Result<RespValue, ProximaError> {
    loop {
        match proxima_redis::parse(buffered) {
            Ok((frame, consumed)) => {
                let value = RespValue::from_frame(&frame);
                buffered.drain(..consumed);
                return Ok(value);
            }
            Err(proxima_redis::ParseError::NeedMore) => {
                let mut chunk = [0_u8; 4096];
                let read = stream.read(&mut chunk).await?;
                if read == 0 {
                    return Err(ProximaError::Upstream(
                        "server closed the connection unexpectedly".into(),
                    ));
                }
                buffered.extend_from_slice(&chunk[..read]);
            }
            Err(error) => {
                return Err(ProximaError::Decode(format!(
                    "malformed reply from server: {error}"
                )));
            }
        }
    }
}

fn expect_reply(actual: RespValue, expected: &RespValue, step: &str) -> Result<(), ProximaError> {
    if &actual == expected {
        Ok(())
    } else {
        Err(ProximaError::Upstream(format!(
            "{step}: expected {expected:?}, got {actual:?}"
        )))
    }
}

#[tokio::main]
async fn main() -> Result<(), ProximaError> {
    let bind_addr: std::net::SocketAddr = "127.0.0.1:0".parse().expect("parse bind addr");
    let handler = proxima_redis::into_redis_handle(KvStore::default());

    // bind to a free port first, then drop it so .serve() can claim it —
    // `.bind(addr)` with port 0 hides the ephemeral port the OS chose.
    let probe = tokio::net::TcpListener::bind(bind_addr).await?;
    let addr = probe.local_addr()?;
    drop(probe);

    let server = Listener::builder()
        .redis(handler)
        .bind(addr)
        .handle(proxima::pipe::into_handle(NullHandle))
        .serve()
        .await?;

    let mut subscriber = TcpStream::connect(addr).await?;
    let mut subscriber_buf = Vec::new();

    println!("PING -> expect PONG");
    send_command(&mut subscriber, &[b"PING"]).await?;
    expect_reply(
        read_reply(&mut subscriber, &mut subscriber_buf).await?,
        &RespValue::SimpleString("PONG".to_string()),
        "PING",
    )?;

    println!("SUBSCRIBE news -> enters Subscriber mode");
    send_command(&mut subscriber, &[b"SUBSCRIBE", b"news"]).await?;
    expect_reply(
        read_reply(&mut subscriber, &mut subscriber_buf).await?,
        &RespValue::Array(vec![
            RespValue::BulkString(b"subscribe".to_vec()),
            RespValue::BulkString(b"news".to_vec()),
            RespValue::Integer(1),
        ]),
        "SUBSCRIBE ack",
    )?;

    println!("GET while subscribed -> gated: a normal error reply, connection stays alive");
    send_command(&mut subscriber, &[b"GET", b"k"]).await?;
    match read_reply(&mut subscriber, &mut subscriber_buf).await? {
        RespValue::Error(message) => {
            if !message.contains("SUBSCRIBE") {
                return Err(ProximaError::Upstream(format!(
                    "gate rejection should name the allowed commands: {message}"
                )));
            }
            println!("  rejected as expected: {message}");
        }
        other => {
            return Err(ProximaError::Upstream(format!(
                "expected a gated error reply, got {other:?}"
            )));
        }
    }

    println!("PUBLISH from a second connection -> delivered to the subscriber");
    let mut publisher = TcpStream::connect(addr).await?;
    let mut publisher_buf = Vec::new();
    send_command(
        &mut publisher,
        &[b"PUBLISH", b"news", b"hello subscribers"],
    )
    .await?;
    expect_reply(
        read_reply(&mut publisher, &mut publisher_buf).await?,
        &RespValue::Integer(1),
        "PUBLISH reply",
    )?;

    let pushed = read_reply(&mut subscriber, &mut subscriber_buf).await?;
    expect_reply(
        pushed.clone(),
        &RespValue::Array(vec![
            RespValue::BulkString(b"message".to_vec()),
            RespValue::BulkString(b"news".to_vec()),
            RespValue::BulkString(b"hello subscribers".to_vec()),
        ]),
        "pushed message",
    )?;
    println!("  push delivered: {pushed:?}");

    println!("UNSUBSCRIBE news -> back to Command mode");
    send_command(&mut subscriber, &[b"UNSUBSCRIBE", b"news"]).await?;
    expect_reply(
        read_reply(&mut subscriber, &mut subscriber_buf).await?,
        &RespValue::Array(vec![
            RespValue::BulkString(b"unsubscribe".to_vec()),
            RespValue::BulkString(b"news".to_vec()),
            RespValue::Integer(0),
        ]),
        "UNSUBSCRIBE ack",
    )?;

    println!("GET again -> Command mode restored, reaches the KV store handler");
    send_command(&mut subscriber, &[b"GET", b"missing-key"]).await?;
    expect_reply(
        read_reply(&mut subscriber, &mut subscriber_buf).await?,
        &RespValue::Null,
        "GET missing key",
    )?;

    println!("SET then GET -> the business handler round-trips a value");
    send_command(&mut subscriber, &[b"SET", b"greeting", b"hi"]).await?;
    expect_reply(
        read_reply(&mut subscriber, &mut subscriber_buf).await?,
        &RespValue::SimpleString("OK".to_string()),
        "SET",
    )?;
    send_command(&mut subscriber, &[b"GET", b"greeting"]).await?;
    expect_reply(
        read_reply(&mut subscriber, &mut subscriber_buf).await?,
        &RespValue::BulkString(b"hi".to_vec()),
        "GET greeting",
    )?;

    println!("walkthrough complete: every legal Connection FSM transition exercised cleanly");
    server.stop();
    Ok(())
}

/// `.redis(handler)` carries its own typed engine, so the generic
/// `.handle(pipe)` this builder still requires (the one uniform
/// validation path every listener axis shares) is never actually
/// dispatched to.
struct NullHandle;

impl SendPipe for NullHandle {
    type In = proxima::request::Request<bytes::Bytes>;
    type Out = proxima::request::Response<bytes::Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        _request: proxima::request::Request<bytes::Bytes>,
    ) -> impl core::future::Future<Output = Result<proxima::request::Response<bytes::Bytes>, ProximaError>>
    + Send {
        async move { Ok(proxima::request::Response::new(404)) }
    }
}
