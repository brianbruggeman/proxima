#![cfg(all(feature = "client", feature = "listen"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Runtime proof of the blocking client's `RedisClient<S, Active|Subscribed>`
//! typestate FSM (PC3-redis-b) — a full `Active -> subscribe -> Subscribed ->
//! next_push -> unsubscribe_all -> Active` round trip against the same
//! in-process broker/`serve_connection` harness `pubsub_round_trip.rs` uses,
//! driven this time through [`proxima_redis::RedisClient`] instead of a raw
//! socket. Proves the transitions work at runtime, not just at the type
//! level (the compile-time half is the `compile_fail` doctest on
//! [`proxima_redis::client::Subscribed`]).

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::net::TcpListener;

use proxima_core::ProximaError;
use proxima_listen::admission::ConnAdmission;
use proxima_net::tokio::tokio_stream_listener::TokioTcpConnection;
use proxima_primitives::pipe::SendPipe;
use proxima_protocols::redis::{RedisRequest, RespValue};
use proxima_redis::{
    ClientError, RedisBroker, RedisClient, RedisClientConfig, RedisServerConfig, serve_connection,
};

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

/// RESP2 sidesteps `HELLO` (the in-process `KvStore` handler above only
/// answers `GET`) so the blocking client's startup handshake completes
/// without a server-side `HELLO` implementation — subscribe acks and pushed
/// frames are `RespValue::Array` either way (`subscribe_ack` in
/// `proxima-redis/src/connection.rs` never upgrades to a RESP3 `Push` type).
fn client_config() -> RedisClientConfig {
    RedisClientConfig::builder().resp3(false).build()
}

// `flavor = "multi_thread"`: the blocking `RedisClient` calls below run
// synchronous `std::net::TcpStream` I/O on the test's own task — a
// single-thread (default) runtime would starve the in-process server's
// spawned connection task on that same thread and deadlock.
#[proxima::test(runtime = "tokio", flavor = "multi_thread")]
async fn active_subscribe_next_push_unsubscribe_all_round_trips_through_every_state() {
    let bind_addr = spawn_server().await;
    let config = client_config();

    let subscriber_stream = TcpStream::connect(bind_addr).expect("connect subscriber");
    let active = RedisClient::connect(subscriber_stream, &config).expect("handshake");

    // Active -> Subscribed: `subscribe` consumes the Active client and
    // returns a `RedisClient<_, Subscribed>` — `.command(..)` is no longer a
    // method that exists on the returned value (see the compile_fail
    // doctest on `Subscribed`).
    let mut subscribed = active.subscribe(&[b"news"]).expect("subscribe");

    let publisher_stream = TcpStream::connect(bind_addr).expect("connect publisher");
    let mut publisher = RedisClient::connect(publisher_stream, &config).expect("handshake");
    let publish_reply = publisher
        .command(&[b"PUBLISH", b"news", b"hi"])
        .expect("publish");
    assert_eq!(publish_reply, RespValue::Integer(1));

    let pushed = subscribed.next_push().expect("next_push");
    assert_eq!(
        pushed,
        RespValue::Array(vec![
            RespValue::BulkString(b"message".to_vec()),
            RespValue::BulkString(b"news".to_vec()),
            RespValue::BulkString(b"hi".to_vec()),
        ])
    );

    // Subscribed -> Active: `unsubscribe_all` consumes the Subscribed client
    // and returns a `RedisClient<_, Active>` — `command` is available again.
    let mut active_again = subscribed.unsubscribe_all().expect("unsubscribe_all");
    let get_reply = active_again.command(&[b"GET", b"k"]).expect("get");
    assert_eq!(get_reply, RespValue::Null);

    let publish_after_unsubscribe = publisher
        .command(&[b"PUBLISH", b"news", b"nobody-home"])
        .expect("publish after unsubscribe");
    assert_eq!(
        publish_after_unsubscribe,
        RespValue::Integer(0),
        "no subscribers remain once unsubscribe_all returns to Active"
    );
}

#[proxima::test(runtime = "tokio", flavor = "multi_thread")]
async fn psubscribe_round_trips_through_pattern_state_too() {
    let bind_addr = spawn_server().await;
    let config = client_config();

    let subscriber_stream = TcpStream::connect(bind_addr).expect("connect subscriber");
    let active = RedisClient::connect(subscriber_stream, &config).expect("handshake");
    let mut subscribed = active.psubscribe(&[b"news.*"]).expect("psubscribe");

    let publisher_stream = TcpStream::connect(bind_addr).expect("connect publisher");
    let mut publisher = RedisClient::connect(publisher_stream, &config).expect("handshake");
    publisher
        .command(&[b"PUBLISH", b"news.tech", b"hi"])
        .expect("publish");

    let pushed = subscribed.next_push().expect("next_push");
    assert_eq!(
        pushed,
        RespValue::Array(vec![
            RespValue::BulkString(b"pmessage".to_vec()),
            RespValue::BulkString(b"news.*".to_vec()),
            RespValue::BulkString(b"news.tech".to_vec()),
            RespValue::BulkString(b"hi".to_vec()),
        ])
    );

    let _active_again = subscribed.unsubscribe_all().expect("unsubscribe_all");
}

#[proxima::test(runtime = "tokio", flavor = "multi_thread")]
async fn ssubscribe_round_trips_through_the_shard_namespace_too() {
    let bind_addr = spawn_server().await;
    let config = client_config();

    let subscriber_stream = TcpStream::connect(bind_addr).expect("connect subscriber");
    let active = RedisClient::connect(subscriber_stream, &config).expect("handshake");
    let mut subscribed = active.ssubscribe(&[b"orders"]).expect("ssubscribe");

    let publisher_stream = TcpStream::connect(bind_addr).expect("connect publisher");
    let mut publisher = RedisClient::connect(publisher_stream, &config).expect("handshake");
    let publish_reply = publisher
        .command(&[b"SPUBLISH", b"orders", b"shipped"])
        .expect("spublish");
    assert_eq!(publish_reply, RespValue::Integer(1));

    let pushed = subscribed.next_push().expect("next_push");
    assert_eq!(
        pushed,
        RespValue::Array(vec![
            RespValue::BulkString(b"smessage".to_vec()),
            RespValue::BulkString(b"orders".to_vec()),
            RespValue::BulkString(b"shipped".to_vec()),
        ])
    );

    let mut active_again = subscribed.unsubscribe_all().expect("unsubscribe_all");
    let spublish_after_unsubscribe = active_again
        .command(&[b"SPUBLISH", b"orders", b"nobody-home"])
        .expect("spublish after unsubscribe_all");
    assert_eq!(
        spublish_after_unsubscribe,
        RespValue::Integer(0),
        "no shard subscribers remain once unsubscribe_all returns to Active"
    );
}

/// A duplicate channel name in one `SUBSCRIBE` call: the server acks every
/// argument unconditionally (one frame per loop iteration — `subscribe`'s own
/// drain of `channels.len()` acks is unaffected), but its per-connection
/// bookkeeping (`SubscriberState`/`Connection::subscribe`, both deduplicating
/// sets) tracks only ONE distinct channel — so a bare-count client-side
/// tracker would drain 2 acks from `unsubscribe_all` while the server only
/// ever sends 1, hanging forever on the second read. The set-based
/// bookkeeping (`RedisClient`'s `channels: BTreeSet<Vec<u8>>`) dedupes the
/// same way the server does, so exactly 1 ack is drained. The read timeout
/// below turns a regression back into this bug into a loud, fast test
/// failure instead of an indefinite CI hang.
#[proxima::test(runtime = "tokio", flavor = "multi_thread")]
async fn subscribe_with_a_duplicate_channel_name_still_returns_cleanly_to_active() {
    let bind_addr = spawn_server().await;
    let config = client_config();

    let subscriber_stream = TcpStream::connect(bind_addr).expect("connect subscriber");
    subscriber_stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .expect("set_read_timeout");
    let active = RedisClient::connect(subscriber_stream, &config).expect("handshake");

    let subscribed = active
        .subscribe(&[b"news", b"news"])
        .expect("subscribe with a duplicate channel name");

    let mut active_again = subscribed
        .unsubscribe_all()
        .expect("unsubscribe_all must drain exactly the distinct-channel ack count, not hang");

    let get_reply = active_again
        .command(&[b"GET", b"k"])
        .expect("command after unsubscribe_all");
    assert_eq!(get_reply, RespValue::Null);
}

/// The shard-namespace variant of the same historically-bitten pattern
/// (`subscribe_with_a_duplicate_channel_name_still_returns_cleanly_to_active`
/// above): a duplicate shard-channel name in one `SSUBSCRIBE` call acks every
/// argument unconditionally, but the server's per-connection bookkeeping
/// (`SubscriberState::shard_channels`/`Connection::subscribe_shard`, both
/// deduplicating sets) tracks only ONE distinct shard channel — so a
/// bare-count client-side tracker would drain 2 `SUNSUBSCRIBE` acks from
/// `unsubscribe_all` while the server only ever sends 1, hanging forever on
/// the second read. `RedisClient`'s `shard_channels: BTreeSet<Vec<u8>>`
/// dedupes the same way the server does, so exactly 1 ack is drained. The
/// read timeout below turns a regression back into this bug into a loud,
/// fast test failure instead of an indefinite CI hang.
#[proxima::test(runtime = "tokio", flavor = "multi_thread")]
async fn ssubscribe_with_a_duplicate_channel_name_still_returns_cleanly_to_active() {
    let bind_addr = spawn_server().await;
    let config = client_config();

    let subscriber_stream = TcpStream::connect(bind_addr).expect("connect subscriber");
    subscriber_stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .expect("set_read_timeout");
    let active = RedisClient::connect(subscriber_stream, &config).expect("handshake");

    let subscribed = active
        .ssubscribe(&[b"news", b"news"])
        .expect("ssubscribe with a duplicate shard-channel name");

    let mut active_again = subscribed
        .unsubscribe_all()
        .expect("unsubscribe_all must drain exactly the distinct-shard-channel ack count, not hang");

    let get_reply = active_again
        .command(&[b"GET", b"k"])
        .expect("command after unsubscribe_all");
    assert_eq!(get_reply, RespValue::Null);
}

/// A fake `Read + Write` transport that panics on any read and records every
/// byte written — used to prove `subscribe(&[])`/`psubscribe(&[])` reject
/// before touching the transport at all (no real socket or server needed;
/// deterministic, no I/O timing involved).
struct NeverStream {
    written: Arc<Mutex<Vec<u8>>>,
}

impl Read for NeverStream {
    fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
        panic!("an empty subscribe/psubscribe must reject before ever reading from the transport");
    }
}

impl Write for NeverStream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.written.lock().expect("written lock").extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn subscribe_and_psubscribe_reject_an_empty_target_list_without_sending_anything() {
    let config = RedisClientConfig::builder().resp3(false).build();

    let channel_writes = Arc::new(Mutex::new(Vec::new()));
    let channel_client = RedisClient::connect(
        NeverStream {
            written: Arc::clone(&channel_writes),
        },
        &config,
    )
    .expect("handshake never touches the transport for resp2/no-auth");
    match channel_client.subscribe(&[]) {
        Err(ClientError::Protocol(_)) => {}
        Err(other) => panic!("expected ClientError::Protocol for an empty subscribe, got {other:?}"),
        Ok(_) => panic!("expected an empty subscribe to be rejected, not to transition to Subscribed"),
    }
    assert!(
        channel_writes.lock().expect("written lock").is_empty(),
        "an empty subscribe must not write a bare SUBSCRIBE onto the wire"
    );

    let pattern_writes = Arc::new(Mutex::new(Vec::new()));
    let pattern_client = RedisClient::connect(
        NeverStream {
            written: Arc::clone(&pattern_writes),
        },
        &config,
    )
    .expect("handshake never touches the transport for resp2/no-auth");
    match pattern_client.psubscribe(&[]) {
        Err(ClientError::Protocol(_)) => {}
        Err(other) => panic!("expected ClientError::Protocol for an empty psubscribe, got {other:?}"),
        Ok(_) => panic!("expected an empty psubscribe to be rejected, not to transition to Subscribed"),
    }
    assert!(
        pattern_writes.lock().expect("written lock").is_empty(),
        "an empty psubscribe must not write a bare PSUBSCRIBE onto the wire"
    );
}

/// End-to-end confirmation that rejecting `subscribe(&[])` before sending
/// anything leaves the SERVER side undisturbed: a sibling connection PUBLISHes
/// right afterward and sees zero subscribers (nothing was ever admitted into
/// `RedisBroker`), and a plain command still round-trips normally.
#[proxima::test(runtime = "tokio", flavor = "multi_thread")]
async fn empty_subscribe_leaves_the_broker_and_server_untouched() {
    let bind_addr = spawn_server().await;
    let config = client_config();

    let rejected_stream = TcpStream::connect(bind_addr).expect("connect");
    let rejected = RedisClient::connect(rejected_stream, &config).expect("handshake");
    match rejected.subscribe(&[]) {
        Err(ClientError::Protocol(_)) => {}
        Err(other) => panic!("expected ClientError::Protocol for an empty subscribe, got {other:?}"),
        Ok(_) => panic!("expected an empty subscribe to be rejected, not to transition to Subscribed"),
    }

    let sibling_stream = TcpStream::connect(bind_addr).expect("connect sibling");
    let mut sibling = RedisClient::connect(sibling_stream, &config).expect("handshake");
    let publish_reply = sibling
        .command(&[b"PUBLISH", b"nobody-subscribed", b"hi"])
        .expect("publish");
    assert_eq!(
        publish_reply,
        RespValue::Integer(0),
        "the rejected empty subscribe must never have reached the broker"
    );

    let get_reply = sibling.command(&[b"GET", b"k"]).expect("command still works");
    assert_eq!(get_reply, RespValue::Null);
}
