#![allow(clippy::unwrap_used, clippy::expect_used)]
//! End-to-end coverage of `KafkaAnyProtocol::drive` over a REAL loopback
//! socket — not just `framed_app`'s in-process `KafkaFramedApp::call`
//! unit tests. Reshaped from the deleted `connection.rs`'s
//! `serve_connection` test suite (a scripted-socket `drive(wire)`
//! helper) plus `pipe.rs`'s CONNECT/upgrade tests — those two files, and
//! the CONNECT/upgrade indirection they existed to bridge, are gone:
//! `KafkaAnyProtocol` now drives a `proxima_listen::any::FramedAny`
//! directly, so this suite asserts the SAME wire-observable behavior
//! (ApiVersions/Produce/unsupported-version/protocol-violation/
//! message-too-large/admission-shed) through the new driver instead.

use std::net::{Ipv4Addr, SocketAddr};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use proxima_core::ProximaError;
use proxima_listen::admission::ConnAdmission;
use proxima_listen::any::{AnyProtocol, erase_handler};
use proxima_net::tokio::tokio_stream_listener::TokioTcpConnection;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::stream::StreamConnection;

use proxima_kafka::wire::{ApiKey, ProduceResponse, RequestBody, ResponseBody};
use proxima_kafka::{KafkaAnyProtocol, KafkaPipeHandle, KafkaServerConfig, into_kafka_handle};

fn encode_request(api_key: i16, api_version: i16, correlation_id: i32, body: &[u8]) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&api_key.to_be_bytes());
    payload.extend_from_slice(&api_version.to_be_bytes());
    payload.extend_from_slice(&correlation_id.to_be_bytes());
    payload.extend_from_slice(&(-1_i16).to_be_bytes()); // null client_id
    payload.extend_from_slice(body);

    let mut wire = Vec::new();
    wire.extend_from_slice(&(payload.len() as i32).to_be_bytes());
    wire.extend_from_slice(&payload);
    wire
}

fn api_versions_request(correlation_id: i32) -> Vec<u8> {
    encode_request(ApiKey::ApiVersions.to_i16(), 0, correlation_id, b"")
}

/// A Produce v0 body with zero topics — the smallest well-formed request
/// this facade's `decode_request` accepts.
fn empty_produce_body() -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&1_i16.to_be_bytes()); // acks
    body.extend_from_slice(&100_i32.to_be_bytes()); // timeout_ms
    body.extend_from_slice(&0_i32.to_be_bytes()); // topics: empty array
    body
}

fn read_correlation_id(reply: &[u8]) -> i32 {
    i32::from_be_bytes([reply[4], reply[5], reply[6], reply[7]])
}

struct HandlerThatMustNotBeCalled;

impl SendPipe for HandlerThatMustNotBeCalled {
    type In = RequestBody;
    type Out = ResponseBody;
    type Err = ProximaError;

    async fn call(&self, _request: RequestBody) -> Result<ResponseBody, ProximaError> {
        panic!("this request must never reach the handler pipe");
    }
}

struct EchoProduceHandler;

impl SendPipe for EchoProduceHandler {
    type In = RequestBody;
    type Out = ResponseBody;
    type Err = ProximaError;

    async fn call(&self, request: RequestBody) -> Result<ResponseBody, ProximaError> {
        match request {
            RequestBody::Produce(_) => Ok(ResponseBody::Produce(ProduceResponse::default())),
            _ => Err(ProximaError::Upstream("unexpected api".into())),
        }
    }
}

fn handler_that_must_not_be_called() -> KafkaPipeHandle {
    into_kafka_handle(HandlerThatMustNotBeCalled)
}

fn echo_produce_handler() -> KafkaPipeHandle {
    into_kafka_handle(EchoProduceHandler)
}

async fn bind_loopback() -> (TcpListener, SocketAddr) {
    let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    (listener, addr)
}

/// Accepts exactly one connection and drives it to completion through
/// `protocol`, returning once `drive` returns (clean close or error).
async fn accept_and_drive(
    listener: TcpListener,
    protocol: KafkaAnyProtocol,
    admission: ConnAdmission,
) -> Result<(), ProximaError> {
    let (stream, _peer) = listener.accept().await.expect("accept");
    let connection: Box<dyn StreamConnection> = Box::new(TokioTcpConnection::from_tokio(stream));
    let handler = erase_handler(());
    protocol
        .drive(connection, handler, &serde_json::Value::Null, None, &admission)
        .await
}

/// `ApiVersions` is answered protocol-level and never reaches the
/// handler pipe — mirrors the deleted `connection.rs`'s
/// `api_versions_is_answered_protocol_level_without_reaching_the_handler`.
#[tokio::test]
async fn api_versions_is_answered_without_reaching_the_handler() {
    let (listener, addr) = bind_loopback().await;
    let protocol = KafkaAnyProtocol::new("kafka", handler_that_must_not_be_called());
    let server = tokio::spawn(accept_and_drive(listener, protocol, ConnAdmission::unbounded()));

    let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
    stream.write_all(&api_versions_request(1)).await.expect("write");
    // ApiVersions keeps the connection open (`Reply`, not `CloseWithReply`);
    // half-close the write side so the server's next read sees EOF and
    // returns cleanly.
    stream.shutdown().await.expect("shutdown write half");
    let mut reply = Vec::new();
    stream.read_to_end(&mut reply).await.expect("read to eof");
    assert!(reply.len() > 8);
    assert_eq!(read_correlation_id(&reply), 1);
    server.await.expect("server task").expect("drive");
}

/// A well-formed Produce request reaches the handler pipe and its reply
/// rides back under the same `correlation_id`.
#[tokio::test]
async fn a_produce_request_reaches_the_handler_and_replies() {
    let (listener, addr) = bind_loopback().await;
    let protocol = KafkaAnyProtocol::new("kafka", echo_produce_handler());
    let server = tokio::spawn(accept_and_drive(listener, protocol, ConnAdmission::unbounded()));

    let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
    let request = encode_request(ApiKey::Produce.to_i16(), 0, 42, &empty_produce_body());
    stream.write_all(&request).await.expect("write");
    stream.shutdown().await.expect("shutdown write half");
    let mut reply = Vec::new();
    stream.read_to_end(&mut reply).await.expect("read to eof");
    // [len:4][correlation_id:4][topics array count:4] — ProduceResponse::default().
    assert_eq!(reply.len(), 12);
    assert_eq!(read_correlation_id(&reply), 42);
    assert_eq!(&reply[8..12], &0_i32.to_be_bytes());
    server.await.expect("server task").expect("drive");
}

/// An unsupported `api_version` for a recognized `api_key` answers a
/// well-formed, data-free reply under the request's own `correlation_id`
/// WITHOUT reaching the handler — mirrors the deleted `connection.rs`'s
/// `unsupported_version_gets_a_well_formed_empty_reply_not_a_dropped_connection`.
#[tokio::test]
async fn unsupported_version_replies_without_reaching_the_handler() {
    let (listener, addr) = bind_loopback().await;
    let protocol = KafkaAnyProtocol::new("kafka", handler_that_must_not_be_called());
    let server = tokio::spawn(accept_and_drive(listener, protocol, ConnAdmission::unbounded()));

    let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
    let request = encode_request(ApiKey::Produce.to_i16(), 9, 3, b"");
    stream.write_all(&request).await.expect("write");
    stream.shutdown().await.expect("shutdown write half");
    let mut reply = Vec::new();
    stream.read_to_end(&mut reply).await.expect("read to eof");
    assert!(!reply.is_empty(), "connection must reply, not silently close");
    assert_eq!(read_correlation_id(&reply), 3);
    server.await.expect("server task").expect("drive");
}

/// A malformed request header (too short to even carry `api_key`/
/// `api_version`/`correlation_id`/`client_id`) has no trustworthy
/// `correlation_id` to answer against — the connection closes with no
/// reply at all, mirroring the deleted `Advanced::ProtocolError` arm.
#[tokio::test]
async fn a_malformed_header_closes_the_connection_with_no_reply() {
    let (listener, addr) = bind_loopback().await;
    let protocol = KafkaAnyProtocol::new("kafka", handler_that_must_not_be_called());
    let server = tokio::spawn(accept_and_drive(listener, protocol, ConnAdmission::unbounded()));

    let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
    let mut malformed = Vec::new();
    malformed.extend_from_slice(&3_i32.to_be_bytes()); // declares 3 bytes, too short for a v0 header
    malformed.extend_from_slice(&[0_u8, 1, 2]);
    stream.write_all(&malformed).await.expect("write");
    let mut reply = Vec::new();
    stream.read_to_end(&mut reply).await.expect("read to eof");
    assert_eq!(reply, b"", "a malformed header must close with no reply at all");
    server.await.expect("server task").expect("drive");
}

/// A still-incomplete frame whose declared size already exceeds the
/// configured cap closes with no reply (there is no complete header yet
/// to answer against) — mirrors the deleted `Advanced::MessageTooLarge`.
#[tokio::test]
async fn an_oversized_declared_frame_closes_with_no_reply() {
    let (listener, addr) = bind_loopback().await;
    let config = KafkaServerConfig::builder().max_message_bytes(10).build();
    let protocol = KafkaAnyProtocol::new("kafka", handler_that_must_not_be_called()).with_config(config);
    let server = tokio::spawn(accept_and_drive(listener, protocol, ConnAdmission::unbounded()));

    let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
    // declares a 1000-byte payload but supplies none of it.
    stream.write_all(&1000_i32.to_be_bytes()).await.expect("write");
    let mut reply = Vec::new();
    stream.read_to_end(&mut reply).await.expect("read to eof");
    assert_eq!(reply, b"", "an oversized declared frame must close with no reply");
    server.await.expect("server task").expect("drive");
}

/// The listener's admission policy — not the business handler — decides
/// whether a request reaches the engine while quiescing. `FramedAny`'s
/// generic `AdmittedApp` checks admission on EVERY frame uniformly, so a
/// handler that would panic if ever called proves the shed path never
/// dispatched to it.
#[tokio::test]
async fn a_business_request_is_shed_with_an_empty_reply_while_admission_is_quiescing() {
    let (listener, addr) = bind_loopback().await;
    let protocol = KafkaAnyProtocol::new("kafka", handler_that_must_not_be_called());
    let admission = ConnAdmission::unbounded();
    admission.begin_quiesce();
    let server = tokio::spawn(accept_and_drive(listener, protocol, admission));

    let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
    let request = encode_request(ApiKey::Produce.to_i16(), 0, 7, &empty_produce_body());
    stream.write_all(&request).await.expect("write");
    stream.shutdown().await.expect("shutdown write half");
    let mut reply = Vec::new();
    stream.read_to_end(&mut reply).await.expect("read to eof");
    assert!(!reply.is_empty(), "a shed request must still reply, not drop the connection");
    assert_eq!(read_correlation_id(&reply), 7);
    server.await.expect("server task").expect("drive");
}

/// `ApiVersions` is exempt from admission shedding — the deleted
/// driver's own admission check ran only AFTER `ApiVersions` had already
/// short-circuited to its answer, so it was never actually sheddable;
/// `KafkaFramedApp`'s `shed_reply` reproduces that exemption.
#[tokio::test]
async fn api_versions_bypasses_admission_shedding() {
    let (listener, addr) = bind_loopback().await;
    let protocol = KafkaAnyProtocol::new("kafka", handler_that_must_not_be_called());
    let admission = ConnAdmission::unbounded();
    admission.begin_quiesce();
    let server = tokio::spawn(accept_and_drive(listener, protocol, admission));

    let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
    stream.write_all(&api_versions_request(5)).await.expect("write");
    stream.shutdown().await.expect("shutdown write half");
    let mut reply = Vec::new();
    stream.read_to_end(&mut reply).await.expect("read to eof");
    assert!(reply.len() > 8);
    assert_eq!(read_correlation_id(&reply), 5);
    server.await.expect("server task").expect("drive");
}
