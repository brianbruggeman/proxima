#![allow(clippy::unwrap_used, clippy::expect_used)]
//! Pub/sub round trip across TWO connections: SUBSCRIBE on one, PUBLISH
//! from the other, the pushed `PUBLISH` frame delivered — proves
//! `KeyedFanOut`/`MqttBroker` end to end through the real driver
//! (`serve_connection`), not just the in-process `MqttBroker` unit tests.
//! Also proves the CONNECT auth hook: a handler that forbids a client id
//! gets a non-zero `CONNACK` and the connection closes.

use std::sync::Arc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use proxima_core::ProximaError;
use proxima_listen::admission::ConnAdmission;
use proxima_net::tokio::tokio_stream_listener::TokioTcpConnection;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::request::Response;
use proxima_protocols::mqtt::encode::{encode_connect, encode_publish, encode_subscribe, encode_unsubscribe};
use proxima_protocols::mqtt::{MqttReply, MqttRequest, ParseError, Packet, parse_packet};
use proxima_mqtt::{MqttBroker, MqttPipeReply, MqttPipeRequest, MqttServerConfig, serve_connection};

#[derive(Clone)]
struct AllowAllExcept {
    forbidden_client_id: Option<Vec<u8>>,
}

impl SendPipe for AllowAllExcept {
    type In = MqttPipeRequest;
    type Out = MqttPipeReply;
    type Err = ProximaError;

    fn call(
        &self,
        request: MqttPipeRequest,
    ) -> impl core::future::Future<Output = Result<MqttPipeReply, ProximaError>> + Send {
        let forbidden = self.forbidden_client_id.clone();
        async move {
            if let MqttRequest::Connect { client_id, .. } = &request.payload
                && Some(client_id.clone()) == forbidden
            {
                return Err(ProximaError::Forbidden("client id is blocked".into()));
            }
            Ok(Response::typed(200, MqttReply::ConnAck { session_present: false, return_code: 0 }))
        }
    }
}

async fn spawn_server(handler_pipe: AllowAllExcept) -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let bind_addr = listener.local_addr().expect("local_addr");
    let handler = proxima_mqtt::into_mqtt_handle(handler_pipe);
    let broker = Arc::new(MqttBroker::new());
    let config = Arc::new(MqttServerConfig::default());
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

#[derive(Debug)]
enum Inbound {
    ConnAck { return_code: u8 },
    SubAck { granted: Vec<u8> },
    UnsubAck,
    Publish { topic: Vec<u8>, payload: Vec<u8> },
}

async fn read_one(stream: &mut TcpStream, buffered: &mut Vec<u8>) -> Inbound {
    loop {
        match parse_packet(buffered) {
            Ok((packet, consumed)) => {
                let inbound = match packet {
                    Packet::ConnAck { return_code, .. } => Inbound::ConnAck { return_code },
                    Packet::SubAck { return_codes, .. } => Inbound::SubAck { granted: return_codes.to_vec() },
                    Packet::Ack { .. } => Inbound::UnsubAck,
                    Packet::Publish { topic, payload, .. } => {
                        Inbound::Publish { topic: topic.to_vec(), payload: payload.to_vec() }
                    }
                    other => panic!("unexpected packet: {other:?}"),
                };
                buffered.drain(..consumed);
                return inbound;
            }
            Err(ParseError::Short | ParseError::PartialPacket(_)) => {
                let mut chunk = [0_u8; 4096];
                let read = stream.read(&mut chunk).await.expect("read reply");
                assert!(read > 0, "server closed the connection unexpectedly");
                buffered.extend_from_slice(&chunk[..read]);
            }
            Err(error) => panic!("malformed packet from server: {error}"),
        }
    }
}

async fn connect(stream: &mut TcpStream, buffered: &mut Vec<u8>, client_id: &str) -> u8 {
    let mut wire = Vec::new();
    encode_connect(client_id.as_bytes(), true, 30, None, None, &mut wire);
    stream.write_all(&wire).await.expect("write CONNECT");
    match read_one(stream, buffered).await {
        Inbound::ConnAck { return_code } => return_code,
        other => panic!("expected CONNACK, got {other:?}"),
    }
}

#[proxima::test(runtime = "tokio")]
async fn publish_delivers_to_a_subscribed_connection_across_two_sockets() {
    let bind_addr = spawn_server(AllowAllExcept { forbidden_client_id: None }).await;

    let mut subscriber = TcpStream::connect(bind_addr).await.expect("connect subscriber");
    let mut subscriber_buf = Vec::new();
    assert_eq!(connect(&mut subscriber, &mut subscriber_buf, "subscriber").await, 0);

    let mut sub_wire = Vec::new();
    encode_subscribe(1, &[(b"news/#", 0)], &mut sub_wire);
    subscriber.write_all(&sub_wire).await.expect("write SUBSCRIBE");
    match read_one(&mut subscriber, &mut subscriber_buf).await {
        Inbound::SubAck { granted } => assert_eq!(granted, vec![0]),
        other => panic!("expected SUBACK, got {other:?}"),
    }

    let mut publisher = TcpStream::connect(bind_addr).await.expect("connect publisher");
    let mut publisher_buf = Vec::new();
    assert_eq!(connect(&mut publisher, &mut publisher_buf, "publisher").await, 0);

    let mut pub_wire = Vec::new();
    encode_publish(b"news/tech", None, b"hi", 0, false, false, &mut pub_wire);
    publisher.write_all(&pub_wire).await.expect("write PUBLISH");

    match read_one(&mut subscriber, &mut subscriber_buf).await {
        Inbound::Publish { topic, payload } => {
            assert_eq!(topic, b"news/tech");
            assert_eq!(payload, b"hi");
        }
        other => panic!("expected a pushed PUBLISH, got {other:?}"),
    }
}

#[proxima::test(runtime = "tokio")]
async fn unsubscribed_topic_no_longer_receives_publishes() {
    let bind_addr = spawn_server(AllowAllExcept { forbidden_client_id: None }).await;

    let mut subscriber = TcpStream::connect(bind_addr).await.expect("connect subscriber");
    let mut subscriber_buf = Vec::new();
    assert_eq!(connect(&mut subscriber, &mut subscriber_buf, "subscriber").await, 0);

    let mut sub_wire = Vec::new();
    encode_subscribe(1, &[(b"chan", 0)], &mut sub_wire);
    subscriber.write_all(&sub_wire).await.expect("write SUBSCRIBE");
    let _ack = read_one(&mut subscriber, &mut subscriber_buf).await;

    let mut unsub_wire = Vec::new();
    encode_unsubscribe(2, &[b"chan"], &mut unsub_wire);
    subscriber.write_all(&unsub_wire).await.expect("write UNSUBSCRIBE");
    match read_one(&mut subscriber, &mut subscriber_buf).await {
        Inbound::UnsubAck => {}
        other => panic!("expected UNSUBACK, got {other:?}"),
    }

    let mut publisher = TcpStream::connect(bind_addr).await.expect("connect publisher");
    let mut publisher_buf = Vec::new();
    assert_eq!(connect(&mut publisher, &mut publisher_buf, "publisher").await, 0);
    let mut pub_wire = Vec::new();
    encode_publish(b"chan", None, b"nobody-home", 0, false, false, &mut pub_wire);
    publisher.write_all(&pub_wire).await.expect("write PUBLISH");

    // no delivery is expected; proven by a follow-up PING/PINGRESP round
    // trip on the subscriber's own connection completing cleanly with no
    // stray bytes queued ahead of the PINGRESP.
    let mut ping_wire = Vec::new();
    proxima_protocols::mqtt::encode::encode_pingreq(&mut ping_wire);
    subscriber.write_all(&ping_wire).await.expect("write PINGREQ");
    let mut chunk = [0_u8; 16];
    let read = subscriber.read(&mut chunk).await.expect("read PINGRESP");
    assert_eq!(&chunk[..read], &[0xD0, 0x00], "only the PINGRESP arrived, no leaked PUBLISH");
}

#[proxima::test(runtime = "tokio")]
async fn connect_is_rejected_with_a_non_zero_connack_when_the_handler_forbids_the_client_id() {
    let bind_addr = spawn_server(AllowAllExcept {
        forbidden_client_id: Some(b"blocked".to_vec()),
    })
    .await;

    let mut client = TcpStream::connect(bind_addr).await.expect("connect");
    let mut buffered = Vec::new();
    assert_eq!(connect(&mut client, &mut buffered, "blocked").await, 5);

    // the broker closes the connection after a non-zero CONNACK.
    let mut chunk = [0_u8; 16];
    let read = client.read(&mut chunk).await.expect("read after rejection");
    assert_eq!(read, 0, "connection must close after a non-zero CONNACK");
}
