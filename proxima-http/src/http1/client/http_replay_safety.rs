use super::{H1ClientUpstream, ResponseBodyMode};
use bytes::Bytes;
use proxima_core::ProximaError;
use proxima_net::tokio::tokio_stream_upstream::TokioTcpUpstream;
use proxima_primitives::pipe::capabilities::Idempotent;
use proxima_primitives::pipe::{Method, Request, SendPipe};
use std::net::{Ipv4Addr, SocketAddr};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;

const CHARGE_BODY: &[u8] = br#"{"task":"charge1"}"#;
const RESPONSE: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}";

async fn read_full_request(socket: &mut TcpStream) -> Option<Vec<u8>> {
    let mut wire = Vec::new();
    let mut single = [0_u8; 1];
    while !wire.ends_with(b"\r\n\r\n") {
        let count = socket.read(&mut single).await.expect("peer reads head");
        if count == 0 {
            assert!(wire.is_empty(), "peer received truncated head: {wire:?}");
            return None;
        }
        wire.extend_from_slice(&single);
    }
    let head = String::from_utf8(wire.clone()).expect("fixture head is utf8");
    let content_length = head
        .split("\r\n")
        .filter_map(|line| line.split_once(':'))
        .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        .map_or(0, |(_, value)| {
            value.trim().parse::<usize>().expect("length")
        });
    let head_length = wire.len();
    wire.resize(head_length + content_length, 0);
    socket
        .read_exact(&mut wire[head_length..])
        .await
        .expect("peer receives exact complete content-length body before closing");
    Some(wire)
}

async fn send(
    client: &H1ClientUpstream<TokioTcpUpstream>,
    raw: bool,
    method: &str,
    path: &str,
    body: &[u8],
) -> Result<u16, ProximaError> {
    if raw {
        let mut wire = format!(
            "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n",
            body.len()
        )
        .into_bytes();
        wire.extend_from_slice(body);
        client.send_raw(&wire).await
    } else {
        let request = Request::builder()
            .method(method)
            .path(Bytes::copy_from_slice(path.as_bytes()))
            .body(Bytes::copy_from_slice(body))
            .build()
            .expect("literal request builds");
        let response = SendPipe::call(client, request).await?;
        let status = response.status;
        assert_eq!(
            response.collect_body().await.expect("complete response"),
            "{}"
        );
        Ok(status)
    }
}

async fn scenario(
    raw: bool,
    method: &str,
    reused: bool,
    close_targets: usize,
) -> (Result<u16, ProximaError>, Vec<(usize, Vec<u8>)>, bool) {
    assert_eq!(CHARGE_BODY.len(), 18, "paper body framing is independent");
    let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
        .await
        .expect("bind peer");
    let address = listener.local_addr().expect("peer address");
    let (finish, mut finished) = oneshot::channel::<()>();
    let peer = tokio::spawn(async move {
        let mut records = Vec::new();
        let mut ordinal = 0;
        let mut targets = 0;
        loop {
            let mut socket = tokio::select! {
                _ = &mut finished => break,
                accepted = listener.accept() => accepted.expect("accept peer connection").0,
            };
            ordinal += 1;
            loop {
                let wire = tokio::select! {
                    _ = &mut finished => return records,
                    request = read_full_request(&mut socket) => match request {
                        Some(wire) => wire,
                        None => break,
                    },
                };
                let is_target = wire
                    .windows(b"/v1/clef/decide".len())
                    .any(|part| part == b"/v1/clef/decide")
                    || wire
                        .windows(b"GET /status ".len())
                        .any(|part| part == b"GET /status ");
                let head_end = wire
                    .windows(4)
                    .position(|part| part == b"\r\n\r\n")
                    .expect("full head terminator")
                    + 4;
                println!(
                    "replay_peer {}",
                    serde_json::json!({
                        "connection": ordinal,
                        "head": String::from_utf8_lossy(&wire[..head_end]),
                        "body": String::from_utf8_lossy(&wire[head_end..]),
                        "wire": wire,
                        "peer_read_complete": true,
                        "phase": if is_target && targets < close_targets {
                            "complete_request_then_eof_before_response_head"
                        } else { "complete_request_then_response" },
                    })
                );
                records.push((ordinal, wire));
                if is_target {
                    targets += 1;
                    if targets <= close_targets {
                        break;
                    }
                }
                socket.write_all(RESPONSE).await.expect("peer response");
            }
        }
        records
    });
    let client = H1ClientUpstream::new(TokioTcpUpstream::new(address), "localhost", "replay")
        .with_response_body(ResponseBodyMode::Collect);
    if reused {
        assert_eq!(
            send(&client, raw, "POST", "/v1/clef/preflight", b"{}")
                .await
                .expect("preflight fully consumed"),
            200
        );
        assert!(
            client.idle.len() == 1,
            "preflight cached connection"
        );
    }
    let body = if method == "GET" {
        b"".as_slice()
    } else {
        CHARGE_BODY
    };
    let path = if method == "GET" {
        "/status"
    } else {
        "/v1/clef/decide"
    };
    let outcome = send(&client, raw, method, path, body).await;
    let discarded = client.idle.is_empty();
    println!(
        "replay_client {}",
        serde_json::json!({
            "raw": raw, "method": method, "reused": reused,
            "target_outcome": format!("{outcome:?}"), "connection_discarded": discarded,
        })
    );
    assert_eq!(
        send(&client, raw, "POST", "/independent", b"new-body")
            .await
            .expect("independent subsequent request succeeds"),
        200
    );
    finish.send(()).expect("finish peer");
    let records = peer.await.expect("peer task");
    let independent: Vec<_> = records
        .iter()
        .filter(|(_, wire)| {
            wire.windows(b"/independent".len())
                .any(|part| part == b"/independent")
        })
        .collect();
    assert_eq!(independent.len(), 1, "one independent request");
    assert!(
        independent[0].1.ends_with(b"\r\n\r\nnew-body"),
        "new bytes without old tail"
    );
    (outcome, records, discarded)
}

fn targets(records: &[(usize, Vec<u8>)]) -> Vec<&(usize, Vec<u8>)> {
    records
        .iter()
        .filter(|(_, wire)| {
            wire.windows(b"/v1/clef/decide".len())
                .any(|part| part == b"/v1/clef/decide")
                || wire
                    .windows(b"GET /status ".len())
                    .any(|part| part == b"GET /status ")
        })
        .collect()
}

fn original_eof(outcome: Result<u16, ProximaError>) {
    match &outcome {
        Err(ProximaError::Upstream(message)) => {
            assert_eq!(message, "connection closed before response head")
        }
        _ => panic!("original response-head EOF must survive: {outcome:?}"),
    }
}

// locked paper: proxima-tensor/specs/http1-replay-safety/SPEC.md and external proxima-replay-paper.md.
#[proxima::test(runtime = "tokio")]
async fn ambiguous_post_not_replayed() {
    let (outcome, records, discarded) = scenario(false, "POST", true, 1).await;
    let received = targets(&records);
    assert_eq!(
        received.len(),
        1,
        "one fully consumed POST, no ambiguous replay"
    );
    assert_eq!(received[0].0, 1);
    assert!(received[0].1.ends_with(CHARGE_BODY));
    original_eof(outcome);
    assert!(discarded, "failed exchange discards pooled connection");
}

#[proxima::test(runtime = "tokio")]
async fn idempotent_recovery_bounded() {
    for (token, expected) in [
        ("GET", true),
        ("HEAD", true),
        ("OPTIONS", true),
        ("TRACE", true),
        ("PUT", true),
        ("DELETE", true),
        ("POST", false),
        ("PATCH", false),
        ("CONNECT", false),
        ("CHARGE", false),
        ("get", false),
        ("GETTING", false),
        ("", false),
    ] {
        let method = Method::from(token);
        assert_eq!(method.is_idempotent(), expected, "method {token:?}");
        assert_eq!(
            Method::is_idempotent_token(token.as_bytes()),
            expected,
            "token {token:?}"
        );
        let request = Request::builder()
            .method(token)
            .path("/")
            .build()
            .expect("request");
        assert_eq!(
            Idempotent::is_idempotent(&request),
            expected,
            "capability {token:?}"
        );
    }
    assert!(!Method::Other(Bytes::from_static(b"GET")).is_idempotent());
    for close_targets in [1, 2] {
        let (outcome, records, discarded) = scenario(false, "GET", true, close_targets).await;
        let received = targets(&records);
        assert_eq!(received.len(), 2, "one initial request and one recovery");
        assert_eq!(received[0].1, received[1].1, "byte-identical recovery");
        assert_eq!((received[0].0, received[1].0), (1, 2));
        assert!(received[0].1.starts_with(b"GET /status HTTP/1.1\r\n"));
        if close_targets == 1 {
            assert_eq!(outcome.expect("idempotent recovery returns response"), 200);
            assert!(!discarded);
        } else {
            original_eof(outcome);
            assert!(discarded, "failed automatic retry discards connection");
        }
    }
}

#[proxima::test(runtime = "tokio")]
async fn raw_unknown_not_replayed() {
    for method in [
        "POST",
        "PATCH",
        "CONNECT",
        "CHARGE",
        "get",
        "GETTING",
        "",
        "GET\t",
        "GET/status",
    ] {
        let (outcome, records, discarded) = scenario(true, method, true, 1).await;
        let received = targets(&records);
        assert_eq!(received.len(), 1, "raw method {method:?} never replays");
        let mut supplied = format!(
            "{method} /v1/clef/decide HTTP/1.1\r\nHost: localhost\r\nContent-Length: 18\r\n\r\n"
        )
        .into_bytes();
        supplied.extend_from_slice(CHARGE_BODY);
        assert_eq!(
            received[0].1, supplied,
            "peer consumed exact supplied bytes"
        );
        original_eof(outcome);
        assert!(discarded);
    }
    for close_targets in [1, 2] {
        let (outcome, records, discarded) = scenario(true, "GET", true, close_targets).await;
        let received = targets(&records);
        assert_eq!(received.len(), 2, "raw GET recovery bounded once");
        assert_eq!(received[0].1, received[1].1);
        assert_eq!((received[0].0, received[1].0), (1, 2));
        if close_targets == 1 {
            assert_eq!(outcome.expect("raw GET recovers"), 200);
            assert!(!discarded);
        } else {
            original_eof(outcome);
            assert!(discarded);
        }
    }
}

#[proxima::test(runtime = "tokio")]
async fn fresh_failure_not_replayed() {
    for raw in [false, true] {
        for method in ["GET", "POST"] {
            let (outcome, records, discarded) = scenario(raw, method, false, 1).await;
            let received = targets(&records);
            assert_eq!(received.len(), 1, "fresh {method} raw={raw} sends once");
            assert_eq!(received[0].0, 1);
            original_eof(outcome);
            assert!(discarded, "fresh failure discards connection");
        }
    }
}
