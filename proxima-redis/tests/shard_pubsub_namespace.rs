#![allow(clippy::unwrap_used, clippy::expect_used)]
//! PC3-redis-COMPLIANCE proof: sharded pub/sub (`SSUBSCRIBE`/`SUNSUBSCRIBE`/
//! `SPUBLISH`) lives in a channel namespace distinct from regular pub/sub
//! (`SUBSCRIBE`/`UNSUBSCRIBE`/`PUBLISH`), matching real Redis (7.0+) — an
//! `SPUBLISH foo` reaches only `SSUBSCRIBE foo` subscribers, never a
//! `SUBSCRIBE foo` one, and vice versa, even though both share the same
//! channel name `foo` and proxima is single-node. Same two-socket-through-
//! the-real-driver shape as `pubsub_round_trip.rs` (`serve_connection` +
//! `RedisBroker`, not just the in-process broker unit tests).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use proxima_core::ProximaError;
use proxima_listen::admission::ConnAdmission;
use proxima_net::tokio::tokio_stream_listener::TokioTcpConnection;
use proxima_primitives::pipe::SendPipe;
use proxima_protocols::redis::{ParseError, RedisRequest, RespValue, encode_command, parse};
use proxima_redis::{RedisBroker, RedisServerConfig, serve_connection};

#[derive(Default, Clone)]
struct KvStore {
    data: Arc<Mutex<HashMap<Vec<u8>, Vec<u8>>>>,
}

impl SendPipe for KvStore {
    type In = RedisRequest;
    type Out = RespValue;
    type Err = ProximaError;

    fn call(
        &self,
        request: RedisRequest,
    ) -> impl core::future::Future<Output = Result<RespValue, ProximaError>> + Send {
        let store = self.data.clone();
        async move {
            let RedisRequest::Command { verb, args } = request else {
                return Ok(RespValue::Error("ERR unknown command".to_string()));
            };
            let reply = match verb.as_slice() {
                b"GET" => {
                    let key = args.first().cloned().unwrap_or_default();
                    match store.lock().expect("kv lock").get(&key) {
                        Some(value) => RespValue::BulkString(value.clone()),
                        None => RespValue::Null,
                    }
                }
                other => RespValue::Error(format!(
                    "ERR unknown command '{}'",
                    String::from_utf8_lossy(other)
                )),
            };
            Ok(reply)
        }
    }
}

async fn spawn_server() -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let bind_addr = listener.local_addr().expect("local_addr");
    let handler = proxima_redis::into_redis_handle(KvStore::default());
    let broker = Arc::new(RedisBroker::new());
    let config = Arc::new(RedisServerConfig::default());
    tokio::spawn(async move {
        loop {
            let Ok((stream, _peer)) = listener.accept().await else {
                return;
            };
            let connection = TokioTcpConnection::from_tokio(stream);
            let handler = handler.clone();
            let broker = Arc::clone(&broker);
            let config = Arc::clone(&config);
            tokio::spawn(async move {
                let (_shutdown_tx, shutdown_rx) = futures::channel::oneshot::channel();
                let _ = serve_connection(
                    connection,
                    handler,
                    broker,
                    &config,
                    shutdown_rx,
                    ConnAdmission::unbounded(),
                )
                .await;
            });
        }
    });
    bind_addr
}

async fn send_command(stream: &mut TcpStream, args: &[&[u8]]) {
    let mut wire = Vec::new();
    encode_command(args, &mut wire);
    stream.write_all(&wire).await.expect("write command");
}

async fn read_reply(stream: &mut TcpStream, buffered: &mut Vec<u8>) -> RespValue {
    loop {
        match parse(buffered) {
            Ok((frame, consumed)) => {
                let value = RespValue::from_frame(&frame);
                buffered.drain(..consumed);
                return value;
            }
            Err(ParseError::NeedMore) => {
                let mut chunk = [0_u8; 4096];
                let read = stream.read(&mut chunk).await.expect("read reply");
                assert!(read > 0, "server closed the connection unexpectedly");
                buffered.extend_from_slice(&chunk[..read]);
            }
            Err(error) => panic!("malformed reply from server: {error}"),
        }
    }
}

/// A frame never arrives within `timeout` — proves cross-namespace
/// non-delivery without racing a fixed sleep against the push channel: a
/// short deadline is enough because the driver's push race
/// (`proxima_listen::wait_for_wire_event`) delivers same-process traffic
/// near-instantly when it IS going to arrive at all.
async fn assert_no_frame_arrives(stream: &mut TcpStream, buffered: &mut Vec<u8>, timeout: Duration) {
    match parse(buffered) {
        Ok((frame, _)) => panic!("unexpected frame already buffered: {frame:?}"),
        Err(ParseError::Malformed(reason)) => panic!("malformed bytes already buffered: {reason}"),
        Err(ParseError::NeedMore) => {}
    }
    let mut chunk = [0_u8; 4096];
    match tokio::time::timeout(timeout, stream.read(&mut chunk)).await {
        Err(_elapsed) => {}
        Ok(Ok(0)) => panic!("server closed the connection unexpectedly"),
        Ok(Ok(read)) => {
            buffered.extend_from_slice(&chunk[..read]);
            match parse(buffered) {
                Ok((frame, _)) => panic!("cross-namespace delivery: unexpected frame {frame:?}"),
                Err(_) => panic!("cross-namespace delivery: unexpected bytes on the wire"),
            }
        }
        Ok(Err(error)) => panic!("read error: {error}"),
    }
}

#[proxima::test(runtime = "tokio")]
async fn spublish_is_delivered_to_an_ssubscribe_connection_as_smessage() {
    let bind_addr = spawn_server().await;
    let mut subscriber = TcpStream::connect(bind_addr).await.expect("connect subscriber");
    let mut subscriber_buf = Vec::new();

    send_command(&mut subscriber, &[b"SSUBSCRIBE", b"orders"]).await;
    assert_eq!(
        read_reply(&mut subscriber, &mut subscriber_buf).await,
        RespValue::Array(vec![
            RespValue::BulkString(b"ssubscribe".to_vec()),
            RespValue::BulkString(b"orders".to_vec()),
            RespValue::Integer(1),
        ])
    );

    let mut publisher = TcpStream::connect(bind_addr).await.expect("connect publisher");
    let mut publisher_buf = Vec::new();
    send_command(&mut publisher, &[b"SPUBLISH", b"orders", b"shipped"]).await;
    assert_eq!(
        read_reply(&mut publisher, &mut publisher_buf).await,
        RespValue::Integer(1)
    );

    assert_eq!(
        read_reply(&mut subscriber, &mut subscriber_buf).await,
        RespValue::Array(vec![
            RespValue::BulkString(b"smessage".to_vec()),
            RespValue::BulkString(b"orders".to_vec()),
            RespValue::BulkString(b"shipped".to_vec()),
        ])
    );
}

/// The compliance proof: `SPUBLISH orders` must NOT reach a connection that
/// did the REGULAR `SUBSCRIBE orders` — the two channel namespaces do not
/// cross, even though they share the same channel name.
#[proxima::test(runtime = "tokio")]
async fn spublish_does_not_cross_into_a_regular_subscribe_connection() {
    let bind_addr = spawn_server().await;
    let mut subscriber = TcpStream::connect(bind_addr).await.expect("connect subscriber");
    let mut subscriber_buf = Vec::new();

    send_command(&mut subscriber, &[b"SUBSCRIBE", b"orders"]).await;
    let _ack = read_reply(&mut subscriber, &mut subscriber_buf).await;

    let mut publisher = TcpStream::connect(bind_addr).await.expect("connect publisher");
    let mut publisher_buf = Vec::new();
    send_command(&mut publisher, &[b"SPUBLISH", b"orders", b"shipped"]).await;
    assert_eq!(
        read_reply(&mut publisher, &mut publisher_buf).await,
        RespValue::Integer(0),
        "SPUBLISH must report zero shard subscribers even though a regular SUBSCRIBE exists"
    );

    assert_no_frame_arrives(&mut subscriber, &mut subscriber_buf, Duration::from_millis(200)).await;
}

/// The reverse direction: a regular `PUBLISH orders` must NOT reach a
/// connection that did `SSUBSCRIBE orders`.
#[proxima::test(runtime = "tokio")]
async fn publish_does_not_cross_into_an_ssubscribe_connection() {
    let bind_addr = spawn_server().await;
    let mut subscriber = TcpStream::connect(bind_addr).await.expect("connect subscriber");
    let mut subscriber_buf = Vec::new();

    send_command(&mut subscriber, &[b"SSUBSCRIBE", b"orders"]).await;
    let _ack = read_reply(&mut subscriber, &mut subscriber_buf).await;

    let mut publisher = TcpStream::connect(bind_addr).await.expect("connect publisher");
    let mut publisher_buf = Vec::new();
    send_command(&mut publisher, &[b"PUBLISH", b"orders", b"regular"]).await;
    assert_eq!(
        read_reply(&mut publisher, &mut publisher_buf).await,
        RespValue::Integer(0),
        "PUBLISH must report zero regular subscribers even though an SSUBSCRIBE exists"
    );

    assert_no_frame_arrives(&mut subscriber, &mut subscriber_buf, Duration::from_millis(200)).await;
}

/// Regression: the regular pub/sub path stays unbroken — `PUBLISH` still
/// reaches a `SUBSCRIBE` connection, framed `message`, exactly as before this
/// change.
#[proxima::test(runtime = "tokio")]
async fn publish_still_reaches_a_regular_subscribe_connection_as_message() {
    let bind_addr = spawn_server().await;
    let mut subscriber = TcpStream::connect(bind_addr).await.expect("connect subscriber");
    let mut subscriber_buf = Vec::new();

    send_command(&mut subscriber, &[b"SUBSCRIBE", b"orders"]).await;
    let _ack = read_reply(&mut subscriber, &mut subscriber_buf).await;

    let mut publisher = TcpStream::connect(bind_addr).await.expect("connect publisher");
    let mut publisher_buf = Vec::new();
    send_command(&mut publisher, &[b"PUBLISH", b"orders", b"regular"]).await;
    assert_eq!(
        read_reply(&mut publisher, &mut publisher_buf).await,
        RespValue::Integer(1)
    );

    assert_eq!(
        read_reply(&mut subscriber, &mut subscriber_buf).await,
        RespValue::Array(vec![
            RespValue::BulkString(b"message".to_vec()),
            RespValue::BulkString(b"orders".to_vec()),
            RespValue::BulkString(b"regular".to_vec()),
        ])
    );
}

/// `SSUBSCRIBE` enters subscriber mode: a subsequent non-safe command is
/// rejected by `Connection::admits()` while shard-subscribed, same gate
/// `SUBSCRIBE` drives.
#[proxima::test(runtime = "tokio")]
async fn ssubscribe_enters_subscriber_mode_and_gates_a_non_safe_command() {
    let bind_addr = spawn_server().await;
    let mut client = TcpStream::connect(bind_addr).await.expect("connect");
    let mut buffered = Vec::new();

    send_command(&mut client, &[b"SSUBSCRIBE", b"orders"]).await;
    let _ack = read_reply(&mut client, &mut buffered).await;

    send_command(&mut client, &[b"GET", b"k"]).await;
    match read_reply(&mut client, &mut buffered).await {
        RespValue::Error(message) => assert!(
            message.contains("SUBSCRIBE"),
            "gate rejection should name the allowed commands: {message}"
        ),
        other => panic!("expected a gated error reply, got {other:?}"),
    }
}

/// Bare `SUNSUBSCRIBE` (no args) removes every shard subscription and, with
/// no regular subscriptions left, returns the connection to `ConnMode::Command`
/// — the previously-gated `GET` is admitted again.
#[proxima::test(runtime = "tokio")]
async fn bare_sunsubscribe_clears_every_shard_subscription_and_exits_subscriber_mode() {
    let bind_addr = spawn_server().await;
    let mut client = TcpStream::connect(bind_addr).await.expect("connect");
    let mut buffered = Vec::new();

    send_command(&mut client, &[b"SSUBSCRIBE", b"orders", b"payments"]).await;
    let _first_ack = read_reply(&mut client, &mut buffered).await;
    let _second_ack = read_reply(&mut client, &mut buffered).await;

    send_command(&mut client, &[b"SUNSUBSCRIBE"]).await;
    let mut acked: Vec<RespValue> = Vec::new();
    for _ in 0..2 {
        acked.push(read_reply(&mut client, &mut buffered).await);
    }
    assert!(
        acked.iter().all(|value| matches!(
            value,
            RespValue::Array(items) if items[0] == RespValue::BulkString(b"sunsubscribe".to_vec())
        )),
        "every ack must be a sunsubscribe frame: {acked:?}"
    );

    send_command(&mut client, &[b"GET", b"k"]).await;
    assert_eq!(read_reply(&mut client, &mut buffered).await, RespValue::Null);
}

/// A live regular subscription keeps the connection gated even after every
/// shard subscription is gone — `ConnMode` only falls back to `Command` once
/// ALL three families (exact, pattern, shard) are empty.
#[proxima::test(runtime = "tokio")]
async fn a_live_regular_subscription_keeps_the_connection_gated_after_sunsubscribe() {
    let bind_addr = spawn_server().await;
    let mut client = TcpStream::connect(bind_addr).await.expect("connect");
    let mut buffered = Vec::new();

    send_command(&mut client, &[b"SUBSCRIBE", b"news"]).await;
    let _subscribe_ack = read_reply(&mut client, &mut buffered).await;
    send_command(&mut client, &[b"SSUBSCRIBE", b"orders"]).await;
    let _ssubscribe_ack = read_reply(&mut client, &mut buffered).await;

    send_command(&mut client, &[b"SUNSUBSCRIBE"]).await;
    let _sunsubscribe_ack = read_reply(&mut client, &mut buffered).await;

    send_command(&mut client, &[b"GET", b"k"]).await;
    match read_reply(&mut client, &mut buffered).await {
        RespValue::Error(message) => assert!(
            message.contains("SUBSCRIBE"),
            "the live regular subscription must still gate GET: {message}"
        ),
        other => panic!("expected a gated error reply, got {other:?}"),
    }
}
