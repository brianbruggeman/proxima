#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]

use proxima_protocols::http3_codec::client::{ClientConnection, H3ClientEvent};
use proxima_protocols::http3_codec::frame::{self, H3Frame};
use proxima_protocols::http3_codec::qpack;
use proxima_protocols::http3_codec::server::{H3ServerEvent, ServerConnection, StreamId as ServerStreamId};
use proxima_protocols::http3_codec::settings::Settings;

const GREASE_FRAME_TYPE: u64 = 0x21;
const HTTP2_RESERVED_PRIORITY_FRAME_TYPE: u64 = 0x02;

fn encode_frame(frame_value: &H3Frame<'_>) -> Vec<u8> {
    let mut output = vec![0_u8; 512];
    let written = frame::encode(frame_value, &mut output).expect("encode h3 test frame");
    output.truncate(written);
    output
}

fn encode_headers(headers: &[(&[u8], &[u8])]) -> Vec<u8> {
    let mut header_block = Vec::new();
    qpack::encoder::encode_refs(headers.iter().copied(), &mut header_block)
        .expect("encode qpack test headers");
    encode_frame(&H3Frame::Headers {
        header_block: &header_block,
    })
}

fn empty_settings() -> Vec<u8> {
    encode_frame(&H3Frame::Settings { payload: &[] })
}

fn grease_frame() -> Vec<u8> {
    encode_frame(&H3Frame::Reserved {
        frame_type: GREASE_FRAME_TYPE,
        payload: b"grease",
    })
}

fn http2_reserved_priority_frame() -> Vec<u8> {
    encode_frame(&H3Frame::Reserved {
        frame_type: HTTP2_RESERVED_PRIORITY_FRAME_TYPE,
        payload: &[],
    })
}

#[test]
fn server_ignores_grease_frame_after_request_headers() {
    let mut server = ServerConnection::new(Settings::default());
    server
        .feed_control(&empty_settings())
        .expect("establish peer settings");
    let _ = server.poll_event().expect("settings event");

    let mut request = encode_headers(&[
        (b":method", b"GET"),
        (b":scheme", b"https"),
        (b":authority", b"localhost"),
        (b":path", b"/"),
    ]);
    request.extend_from_slice(&grease_frame());

    server
        .feed_request(ServerStreamId(0), &request, true)
        .expect("request stream GREASE must be ignored");

    assert!(matches!(
        server.poll_event(),
        Some(H3ServerEvent::RequestHeaders { .. })
    ));
    assert!(matches!(
        server.poll_event(),
        Some(H3ServerEvent::RequestFinished { .. })
    ));
}

#[test]
fn client_ignores_grease_frame_after_response_headers() {
    let mut client = ClientConnection::new(Settings::default());
    client
        .feed_control(&empty_settings())
        .expect("establish peer settings");
    let _ = client.poll_event().expect("settings event");

    let stream_id = client
        .open_request(&[(b":method", b"GET"), (b":path", b"/")])
        .expect("open request");
    let mut response = encode_headers(&[(b":status", b"200")]);
    response.extend_from_slice(&grease_frame());

    client
        .feed_response(stream_id, &response, true)
        .expect("response stream GREASE must be ignored");

    assert!(matches!(
        client.poll_event(),
        Some(H3ClientEvent::ResponseHeaders { .. })
    ));
    assert!(matches!(
        client.poll_event(),
        Some(H3ClientEvent::ResponseFinished { .. })
    ));
}

#[test]
fn server_rejects_http2_reserved_frame_on_control_stream() {
    let mut server = ServerConnection::new(Settings::default());
    server
        .feed_control(&empty_settings())
        .expect("establish peer settings");

    server
        .feed_control(&http2_reserved_priority_frame())
        .expect_err("HTTP/2 PRIORITY frame must be rejected on HTTP/3 control stream");
}

#[test]
fn client_rejects_http2_reserved_frame_on_control_stream() {
    let mut client = ClientConnection::new(Settings::default());
    client
        .feed_control(&empty_settings())
        .expect("establish peer settings");

    client
        .feed_control(&http2_reserved_priority_frame())
        .expect_err("HTTP/2 PRIORITY frame must be rejected on HTTP/3 control stream");
}
