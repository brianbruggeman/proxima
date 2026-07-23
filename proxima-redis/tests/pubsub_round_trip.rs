#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Pub/sub round trip across TWO connections: SUBSCRIBE on one, PUBLISH
//! from the other, the pushed `message`/`pmessage` frame delivered — proves
//! `KeyedFanOut`/`RedisBroker` end to end through the real driver
//! (`serve_connection`), not just the in-process `RedisBroker` unit tests.
//! Also proves the subscriber-mode admission gate: a gated command is
//! rejected while subscribed and admitted again after UNSUBSCRIBE.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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

#[proxima::test(runtime = "tokio")]
async fn publish_delivers_to_a_subscribed_connection_across_two_sockets() {
    let bind_addr = spawn_server().await;
    let mut subscriber = TcpStream::connect(bind_addr).await.expect("connect subscriber");
    let mut subscriber_buf = Vec::new();

    send_command(&mut subscriber, &[b"SUBSCRIBE", b"news"]).await;
    assert_eq!(
        read_reply(&mut subscriber, &mut subscriber_buf).await,
        RespValue::Array(vec![
            RespValue::BulkString(b"subscribe".to_vec()),
            RespValue::BulkString(b"news".to_vec()),
            RespValue::Integer(1),
        ])
    );

    let mut publisher = TcpStream::connect(bind_addr).await.expect("connect publisher");
    let mut publisher_buf = Vec::new();
    send_command(&mut publisher, &[b"PUBLISH", b"news", b"hi"]).await;
    assert_eq!(
        read_reply(&mut publisher, &mut publisher_buf).await,
        RespValue::Integer(1)
    );

    assert_eq!(
        read_reply(&mut subscriber, &mut subscriber_buf).await,
        RespValue::Array(vec![
            RespValue::BulkString(b"message".to_vec()),
            RespValue::BulkString(b"news".to_vec()),
            RespValue::BulkString(b"hi".to_vec()),
        ])
    );
}

#[proxima::test(runtime = "tokio")]
async fn psubscribe_pattern_receives_a_pmessage_frame() {
    let bind_addr = spawn_server().await;
    let mut subscriber = TcpStream::connect(bind_addr).await.expect("connect subscriber");
    let mut subscriber_buf = Vec::new();

    send_command(&mut subscriber, &[b"PSUBSCRIBE", b"news.*"]).await;
    let _ack = read_reply(&mut subscriber, &mut subscriber_buf).await;

    let mut publisher = TcpStream::connect(bind_addr).await.expect("connect publisher");
    let mut publisher_buf = Vec::new();
    send_command(&mut publisher, &[b"PUBLISH", b"news.tech", b"hi"]).await;
    assert_eq!(
        read_reply(&mut publisher, &mut publisher_buf).await,
        RespValue::Integer(1)
    );

    assert_eq!(
        read_reply(&mut subscriber, &mut subscriber_buf).await,
        RespValue::Array(vec![
            RespValue::BulkString(b"pmessage".to_vec()),
            RespValue::BulkString(b"news.*".to_vec()),
            RespValue::BulkString(b"news.tech".to_vec()),
            RespValue::BulkString(b"hi".to_vec()),
        ])
    );
}

#[proxima::test(runtime = "tokio")]
async fn gated_command_is_rejected_while_subscribed_then_admitted_after_unsubscribe() {
    let bind_addr = spawn_server().await;
    let mut client = TcpStream::connect(bind_addr).await.expect("connect");
    let mut buffered = Vec::new();

    send_command(&mut client, &[b"SUBSCRIBE", b"chan"]).await;
    let _subscribe_ack = read_reply(&mut client, &mut buffered).await;

    send_command(&mut client, &[b"GET", b"k"]).await;
    match read_reply(&mut client, &mut buffered).await {
        RespValue::Error(message) => assert!(
            message.contains("SUBSCRIBE"),
            "gate rejection should name the allowed commands: {message}"
        ),
        other => panic!("expected a gated error reply, got {other:?}"),
    }

    send_command(&mut client, &[b"UNSUBSCRIBE", b"chan"]).await;
    assert_eq!(
        read_reply(&mut client, &mut buffered).await,
        RespValue::Array(vec![
            RespValue::BulkString(b"unsubscribe".to_vec()),
            RespValue::BulkString(b"chan".to_vec()),
            RespValue::Integer(0),
        ])
    );

    send_command(&mut client, &[b"GET", b"k"]).await;
    assert_eq!(read_reply(&mut client, &mut buffered).await, RespValue::Null);
}

#[proxima::test(runtime = "tokio")]
async fn unsubscribed_channel_no_longer_receives_publishes() {
    let bind_addr = spawn_server().await;
    let mut subscriber = TcpStream::connect(bind_addr).await.expect("connect subscriber");
    let mut subscriber_buf = Vec::new();

    send_command(&mut subscriber, &[b"SUBSCRIBE", b"chan"]).await;
    let _ack = read_reply(&mut subscriber, &mut subscriber_buf).await;
    send_command(&mut subscriber, &[b"UNSUBSCRIBE", b"chan"]).await;
    let _unsub_ack = read_reply(&mut subscriber, &mut subscriber_buf).await;

    let mut publisher = TcpStream::connect(bind_addr).await.expect("connect publisher");
    let mut publisher_buf = Vec::new();
    send_command(&mut publisher, &[b"PUBLISH", b"chan", b"nobody-home"]).await;
    assert_eq!(
        read_reply(&mut publisher, &mut publisher_buf).await,
        RespValue::Integer(0),
        "no subscribers remain after UNSUBSCRIBE"
    );
}
