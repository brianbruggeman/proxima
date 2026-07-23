//! `DnsAnyProtocol` — DNS-over-TCP (RFC 1035 §4.2.2: each message prefixed
//! by a 2-byte big-endian length) as an [`AnyProtocol`] candidate for the
//! open universal listener, mirroring `RedisAnyProtocol` /
//! `proxima_pgwire`'s own `AnyProtocol` candidates.
//!
//! **Gap this module fills, not one it hides:** `proxima_codec`'s only
//! length-delimited framer, [`proxima_codec::LengthDelimitedCodec`], is
//! hard-coded to a **4-byte** big-endian prefix (`[u32 BE len][payload]` —
//! see its own doc comment). RFC 1035 §4.2.2's TCP framing is a **2-byte**
//! prefix — a different width the shared codec cannot express, so this
//! module reads/writes the 2-byte prefix itself, directly against
//! [`proxima_protocols::dns::codec_trait::parse_message`] and
//! [`proxima_protocols::dns::encode::encode_response`] (the message-body
//! codec the UDP path already uses). This is the same shape
//! `RedisConnectionPipe` and pgwire's connection driver use — a bespoke,
//! small connection loop owning its own framing — not an extension to the
//! shared codec crate; widening `LengthDelimitedCodec` to a generic prefix
//! width is future work tracked here as the named gap, not solved inline.
//!
//! Positive-match probe: the 2-byte length prefix plus a plausible header
//! (`QDCOUNT == 1`, matching this crate's single-question contract — see
//! [`crate::pipes`]'s module doc) is enough signal for a **single-candidate
//! registration** (`Listener::builder().accept("dns-tcp")`), the same
//! "good enough because nothing else is registered under this name" positive
//! -match reasoning [`proxima_redis::RedisAnyProtocol`]'s own doc lays out —
//! DNS query IDs are arbitrary 16-bit values, so there's no sigil byte to
//! sniff the way RESP's `*` or a fixed magic number would give.

use std::future::Future;
use std::pin::Pin;

use futures::io::{AsyncReadExt, AsyncWriteExt};
use serde_json::Value;

use proxima_core::ProximaError;
use proxima_listen::admission::{ConnAdmission, RequestAdmit};
use proxima_listen::any::{AnyHandler, AnyProtocol, ProbeVerdict};
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::Method;
use proxima_primitives::pipe::header_list::HeaderList;
use proxima_primitives::pipe::request::{Request, RequestContext};
use proxima_primitives::stream::{PeerInfo, StreamConnection};
use proxima_protocols::dns::codec_trait::parse_message;
use proxima_telemetry::warn;

use crate::config::DnsServerConfig;
use crate::pipes::{DnsAnswer, DnsPipeHandle};
use crate::wire::{answer_to_wire, message_to_query};

/// RFC 1035 §4.2.2 length-prefix width, in bytes.
const TCP_LENGTH_PREFIX_BYTES: usize = 2;
/// Smallest a framed TCP message can legally be: the 2-byte length prefix
/// plus the 12-byte fixed DNS header (RFC 1035 §4.1.1) it must at least
/// declare.
const MIN_TCP_FRAME_PREFIX: usize = TCP_LENGTH_PREFIX_BYTES + 12;
const METHOD_LABEL: &[u8] = b"DNS-TCP";

/// DNS-over-TCP wire candidate for the open universal listener.
pub struct DnsAnyProtocol {
    label: String,
    handler: DnsPipeHandle,
    config: DnsServerConfig,
}

impl DnsAnyProtocol {
    #[must_use]
    pub fn new(label: impl Into<String>, handler: DnsPipeHandle) -> Self {
        Self {
            label: label.into(),
            handler,
            config: DnsServerConfig::default(),
        }
    }

    /// Replaces the default [`DnsServerConfig`]; a `dns` object in the
    /// listener spec still wins at drive time.
    #[must_use]
    pub fn with_config(mut self, config: DnsServerConfig) -> Self {
        self.config = config;
        self
    }
}

fn resolve_config(base: &DnsServerConfig, spec: &Value) -> Result<DnsServerConfig, ProximaError> {
    match spec.get("dns") {
        None => Ok(base.clone()),
        Some(overrides) => serde_json::from_value(overrides.clone())
            .map_err(|error| ProximaError::Config(format!("dns spec: {error}"))),
    }
}

impl AnyProtocol for DnsAnyProtocol {
    fn name(&self) -> &str {
        &self.label
    }

    fn max_prefix_bytes(&self) -> usize {
        MIN_TCP_FRAME_PREFIX
    }

    fn probe(&self, prefix: &[u8]) -> ProbeVerdict {
        if prefix.len() < MIN_TCP_FRAME_PREFIX {
            return ProbeVerdict::NeedMore {
                at_least: MIN_TCP_FRAME_PREFIX,
            };
        }
        let declared_len = usize::from(u16::from_be_bytes([prefix[0], prefix[1]]));
        if declared_len < 12 {
            // no framed DNS message is ever shorter than its own header.
            return ProbeVerdict::No;
        }
        // header's QDCOUNT sits at header-relative offset 4..6, i.e.
        // prefix-relative 2+4..2+6 — see RFC 1035 §4.1.1's field layout.
        let qdcount = u16::from_be_bytes([prefix[6], prefix[7]]);
        if qdcount == 1 {
            ProbeVerdict::Match { consumed: 0 }
        } else {
            ProbeVerdict::No
        }
    }

    fn drive<'a>(
        &'a self,
        stream: Box<dyn StreamConnection>,
        _handler: AnyHandler,
        spec: &'a Value,
        _peer: Option<PeerInfo>,
        admission: &'a ConnAdmission,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'a>> {
        Box::pin(async move {
            let config = resolve_config(&self.config, spec)?;
            serve_tcp_connection(stream, &self.handler, &config, admission, &self.label).await
        })
    }
}

/// Drive one accepted TCP connection to completion: read a 2-byte length
/// prefix, read exactly that many bytes, parse/dispatch/encode/write the
/// reply, and loop — RFC 1035 §4.2.2 pipelining lets one connection carry
/// many queries. Returns `Ok(())` on a clean EOF between messages (the
/// client closed); a read that stops mid-message is
/// [`crate::error::DnsServeError::UnexpectedEof`], surfaced as
/// [`ProximaError`] at this function's boundary — one connection's own
/// framing error never touches any other connection.
async fn serve_tcp_connection(
    mut stream: Box<dyn StreamConnection>,
    handler: &DnsPipeHandle,
    config: &DnsServerConfig,
    admission: &ConnAdmission,
    label: &str,
) -> Result<(), ProximaError> {
    let peer = stream.peer();
    loop {
        let mut length_prefix = [0u8; TCP_LENGTH_PREFIX_BYTES];
        match stream.read_exact(&mut length_prefix).await {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(error) => return Err(ProximaError::Io(error)),
        }
        let message_len = usize::from(u16::from_be_bytes(length_prefix));
        if message_len > config.max_message_bytes {
            warn!(
                label = %label,
                ?peer,
                len = message_len,
                limit = config.max_message_bytes,
                "dns-tcp message exceeds message limit; closing connection"
            );
            return Ok(());
        }

        let mut body = vec![0u8; message_len];
        stream
            .read_exact(&mut body)
            .await
            .map_err(ProximaError::Io)?;

        let reply_bytes = match handle_one_message(&body, handler, admission, label, peer.clone()).await {
            Some(bytes) => bytes,
            None => continue,
        };

        let reply_len = u16::try_from(reply_bytes.len()).map_err(|_| {
            ProximaError::Config(format!(
                "{label}: encoded dns-tcp reply of {} bytes exceeds the u16 length prefix",
                reply_bytes.len()
            ))
        })?;
        stream
            .write_all(&reply_len.to_be_bytes())
            .await
            .map_err(ProximaError::Io)?;
        stream
            .write_all(&reply_bytes)
            .await
            .map_err(ProximaError::Io)?;
        stream.flush().await.map_err(ProximaError::Io)?;
    }
}

/// Parse, admit, dispatch, and encode one framed message. `None` means
/// "nothing to send back" (malformed input, or the handler failed) — those
/// are logged and the connection carries on to the next frame, mirroring
/// the UDP listener's "one bad message must not tear down the
/// connectionless-equivalent loop" contract; a TCP connection is the
/// analogous per-connection unit here.
async fn handle_one_message(
    body: &[u8],
    handler: &DnsPipeHandle,
    admission: &ConnAdmission,
    label: &str,
    peer: Option<PeerInfo>,
) -> Option<Vec<u8>> {
    let message = match parse_message(body) {
        Ok(message) => message,
        Err(error) => {
            warn!(label = %label, ?peer, ?error, "dns-tcp message failed to parse; skipping");
            return None;
        }
    };
    let query = match message_to_query(&message) {
        Some(query) => query,
        None => {
            warn!(label = %label, ?peer, "dns-tcp message is not exactly one question; skipping");
            return None;
        }
    };

    let answer = match admission.request_admit() {
        RequestAdmit::Admit => {
            let request = Request {
                method: Method::from_wire(bytes::Bytes::from_static(METHOD_LABEL)),
                path: bytes::Bytes::from_static(b"/"),
                query: HeaderList::new(),
                metadata: HeaderList::new(),
                payload: query.clone(),
                stream: None,
                context: RequestContext::default(),
            };
            let outcome = SendPipe::call(handler, request).await;
            admission.request_release();
            match outcome {
                Ok(reply) => reply.payload,
                Err(error) => {
                    warn!(label = %label, ?peer, ?error, "dns-tcp handler pipe failed; skipping");
                    return None;
                }
            }
        }
        // SERVFAIL (RFC 1035 §4.1.1 RCODE 2) — the listener's uniform
        // admission policy sheds the request; DNS's own wire-specific
        // rejection is a server-failure answer, not a dropped connection.
        RequestAdmit::Shed { reason } => {
            warn!(label = %label, ?peer, ?reason, "dns-tcp request shed; replying servfail");
            DnsAnswer {
                rcode: 2,
                authoritative: false,
                recursion_available: true,
                records: Vec::new(),
            }
        }
    };

    let mut out = Vec::new();
    if let Err(error) = answer_to_wire(&query, &answer, &mut out) {
        warn!(label = %label, ?peer, ?error, "dns-tcp answer failed to encode; skipping");
        return None;
    }
    Some(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_protocols::dns::encode;

    fn framed_query(id: u16) -> Vec<u8> {
        let mut message = Vec::new();
        encode::encode_query(
            id,
            true,
            encode::EncodeQuestion {
                name: "example.com.",
                qtype: 1,
                qclass: 1,
            },
            &mut message,
        )
        .unwrap();
        let mut framed = Vec::new();
        framed.extend_from_slice(&u16::try_from(message.len()).unwrap().to_be_bytes());
        framed.extend_from_slice(&message);
        framed
    }

    fn handler() -> DnsPipeHandle {
        struct EchoAnswer;
        impl SendPipe for EchoAnswer {
            type In = crate::pipes::DnsPipeRequest;
            type Out = crate::pipes::DnsPipeReply;
            type Err = ProximaError;

            async fn call(&self, _request: Self::In) -> Result<Self::Out, ProximaError> {
                Ok(crate::pipes::DnsPipeReply::typed(200, DnsAnswer::ok(Vec::new())))
            }
        }
        crate::pipes::into_dns_handle(EchoAnswer)
    }

    #[test]
    fn probe_needs_more_below_the_minimum_frame() {
        let protocol = DnsAnyProtocol::new("dns-tcp", handler());
        assert_eq!(
            protocol.probe(&[0u8; 4]),
            ProbeVerdict::NeedMore {
                at_least: MIN_TCP_FRAME_PREFIX
            }
        );
    }

    #[test]
    fn probe_matches_a_well_formed_single_question_frame() {
        let protocol = DnsAnyProtocol::new("dns-tcp", handler());
        let framed = framed_query(1234);
        assert_eq!(
            protocol.probe(&framed[..MIN_TCP_FRAME_PREFIX]),
            ProbeVerdict::Match { consumed: 0 }
        );
    }

    #[test]
    fn probe_rejects_a_multi_question_header() {
        let protocol = DnsAnyProtocol::new("dns-tcp", handler());
        let mut framed = framed_query(1234);
        // bump QDCOUNT (prefix-relative bytes 6..8) to 2.
        framed[7] = 2;
        assert_eq!(protocol.probe(&framed[..MIN_TCP_FRAME_PREFIX]), ProbeVerdict::No);
    }

    #[proxima::test]
    async fn handle_one_message_returns_a_framed_reply_body() {
        let admission = ConnAdmission::unbounded();
        let framed = framed_query(1234);
        let body = &framed[TCP_LENGTH_PREFIX_BYTES..];
        let reply = handle_one_message(body, &handler(), &admission, "dns-tcp-test", None)
            .await
            .expect("well-formed query yields a reply");
        let message = parse_message(&reply).unwrap();
        assert_eq!(message.header.id, 1234);
        assert!(message.header.flags.is_response());
    }

    #[proxima::test]
    async fn handle_one_message_skips_malformed_input() {
        let admission = ConnAdmission::unbounded();
        let reply = handle_one_message(&[0u8; 4], &handler(), &admission, "dns-tcp-test", None).await;
        assert!(reply.is_none());
    }
}
