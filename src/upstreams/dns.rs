//! `PipeFactory` + [`ClientProtocol`] for the `dns` protocol terminal — a
//! `proxima::Client` transport that issues DNS queries.
//!
//! Reached via the `type` discriminator (`{"type":"dns", "dsn":
//! "dns://9.9.9.9:53"}` or the field form) or the `.dns(dsn)` builder sugar
//! (`ClientProtocolExt`), which lowers to `.protocol(DnsClientProtocol::dsn(dsn))`.
//! Composes [`DnsClientUpstream`](proxima_dns::DnsClientUpstream) over the
//! prime UDP transport ([`PrimeDatagramFactory`](proxima_net::prime::PrimeDatagramFactory)).
//!
//! Unlike the TCP-shaped `kafka`/`mqtt`/`amqp`/`memcached`/`redis`/`pgwire`
//! terminals, `DnsClientUpstream` speaks the crate's own typed
//! `DnsPipeRequest`/`DnsPipeReply` (`Request<DnsQuery>`/`Response<DnsAnswer>`),
//! not `Request<Bytes>`/`Response<Bytes>` — so [`DnsClientPipe`] translates
//! at the boundary, the same role `PgwireClientPipe` plays for pgwire.
//! Convention (this facade's own, there being no wire-level convention to
//! borrow): `request.path` (leading `/` stripped) is the query name;
//! `request.query`'s `type` key (e.g. `"AAAA"`) picks the record type,
//! defaulting to `"A"`; `class` defaults to `IN`.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use bytes::Bytes;
use serde_json::Value;

use proxima_dns::{DnsClientUpstream, DnsPipeRequest, DnsQuery, DnsResolverConfig};
use proxima_net::prime::PrimeDatagramFactory;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::handler::{PipeHandle, into_handle};
use proxima_primitives::pipe::header_list::HeaderList;
use proxima_primitives::pipe::pipe_factory::PipeFactory;
use proxima_primitives::pipe::request::{Request, RequestContext, Response};

use crate::client::handle::ClientProtocol;
use crate::error::ProximaError;

/// A [`PipeFactory`] for the `dns` key. Builds a client `Pipe` from a
/// [`DnsResolverConfig`] parsed out of the spec.
#[derive(Debug, Default)]
pub struct DnsPipeFactory;

impl DnsPipeFactory {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl PipeFactory for DnsPipeFactory {
    fn name(&self) -> &str {
        "dns"
    }

    fn build(
        &self,
        spec: &Value,
        _inner: Option<PipeHandle>,
    ) -> Pin<Box<dyn Future<Output = Result<PipeHandle, ProximaError>> + Send + '_>> {
        let spec = spec.clone();
        Box::pin(async move {
            let config = config_from_spec(&spec)?;
            let upstream = DnsClientUpstream::new(Arc::new(PrimeDatagramFactory), config);
            Ok(into_handle(DnsClientPipe::new(upstream)))
        })
    }
}

/// Parse a [`DnsResolverConfig`] from the spec: prefer a `dsn` string, else
/// deserialize the field form (serde ignores the `type` discriminator).
fn config_from_spec(spec: &Value) -> Result<DnsResolverConfig, ProximaError> {
    if let Some(dsn) = spec.get("dsn").and_then(Value::as_str) {
        return DnsResolverConfig::from_dsn(dsn)
            .map_err(|err| ProximaError::Config(format!("dns dsn: {err}")));
    }
    serde_json::from_value(spec.clone())
        .map_err(|err| ProximaError::Config(format!("dns config: {err}")))
}

/// Translates the generic `Request<Bytes>`/`Response<Bytes>` boundary every
/// `PipeHandle` shares into [`DnsClientUpstream`]'s typed
/// `DnsPipeRequest`/`DnsPipeReply` — see the module doc for the request
/// convention.
struct DnsClientPipe {
    inner: DnsClientUpstream,
}

impl DnsClientPipe {
    fn new(inner: DnsClientUpstream) -> Self {
        Self { inner }
    }
}

impl SendPipe for DnsClientPipe {
    type In = Request<Bytes>;
    type Out = Response<Bytes>;
    type Err = ProximaError;

    fn call(
        &self,
        request: Request<Bytes>,
    ) -> impl Future<Output = Result<Response<Bytes>, ProximaError>> + Send {
        async move {
            let name = String::from_utf8_lossy(&request.path)
                .trim_start_matches('/')
                .to_string();
            let qtype = request
                .query
                .get_str("type")
                .map(record_type_to_qtype)
                .unwrap_or(1);
            let qclass = request
                .query
                .get_str("class")
                .and_then(|value| value.parse::<u16>().ok())
                .unwrap_or(1);
            let dns_request: DnsPipeRequest = Request {
                method: request.method,
                path: Bytes::new(),
                query: HeaderList::new(),
                metadata: HeaderList::new(),
                payload: DnsQuery {
                    id: 0,
                    recursion_desired: true,
                    name,
                    qtype,
                    qclass,
                },
                stream: None,
                context: RequestContext::default(),
            };
            let reply = self.inner.call(dns_request).await?;
            Ok(dns_reply_to_bytes(reply))
        }
    }
}

/// `A`/`AAAA`/`CNAME`/`MX`/`TXT`/`NS`/`SOA`/`PTR`/`SRV` mnemonics (RFC 1035
/// §3.2.2 / RFC 3596) to their numeric qtype, defaulting to `A` (1) for
/// anything unrecognized rather than erroring — a caller who wants an exotic
/// qtype passes the number directly (`"28"` parses the same as `"AAAA"`).
fn record_type_to_qtype(value: &str) -> u16 {
    match value.to_ascii_uppercase().as_str() {
        "A" => 1,
        "NS" => 2,
        "CNAME" => 5,
        "SOA" => 6,
        "PTR" => 12,
        "MX" => 15,
        "TXT" => 16,
        "AAAA" => 28,
        "SRV" => 33,
        other => other.parse().unwrap_or(1),
    }
}

fn dns_reply_to_bytes(reply: proxima_dns::DnsPipeReply) -> Response<Bytes> {
    let Response {
        status,
        metadata,
        payload,
        stream,
        upgrade,
    } = reply;
    let body = serde_json::to_vec(&dns_answer_to_json(&payload))
        .unwrap_or_else(|err| format!("{{\"encode_error\":\"{err}\"}}").into_bytes());
    let mut response = Response::new(status).with_payload(body);
    response.metadata = metadata;
    response.stream = stream;
    response.upgrade = upgrade;
    response
}

fn dns_answer_to_json(answer: &proxima_dns::DnsAnswer) -> Value {
    serde_json::json!({
        "rcode": answer.rcode,
        "authoritative": answer.authoritative,
        "recursion_available": answer.recursion_available,
        "records": answer.records.iter().map(|record| serde_json::json!({
            "name": record.name,
            "rtype": record.rtype,
            "rclass": record.rclass,
            "ttl": record.ttl,
            "rdata": record.rdata,
        })).collect::<Vec<_>>(),
    })
}

/// The out-of-crate [`ClientProtocol`] a `.dns(dsn)` builder call merges.
pub struct DnsClientProtocol {
    dsn: String,
}

impl DnsClientProtocol {
    /// Point at a resolver by DSN (`dns://resolver_ip[:port]`).
    #[must_use]
    pub fn dsn(dsn: impl Into<String>) -> Self {
        Self { dsn: dsn.into() }
    }
}

impl ClientProtocol for DnsClientProtocol {
    fn spec(&self) -> Value {
        serde_json::json!({"type": "dns", "dsn": self.dsn})
    }

    fn factory(&self) -> Arc<dyn PipeFactory> {
        Arc::new(DnsPipeFactory::new())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn config_from_dsn_spec() {
        let spec = serde_json::json!({ "type": "dns", "dsn": "dns://9.9.9.9:5353" });
        let config = config_from_spec(&spec).expect("config");
        assert_eq!(config.resolver_ip, "9.9.9.9");
        assert_eq!(config.port, 5353);
    }

    #[test]
    fn factory_name_is_the_spec_key() {
        assert_eq!(DnsPipeFactory::new().name(), "dns");
    }

    #[test]
    fn client_protocol_lowers_to_the_type_and_dsn_spec() {
        let protocol = DnsClientProtocol::dsn("dns://9.9.9.9:53");
        let spec = protocol.spec();
        assert_eq!(spec["type"], "dns");
        assert_eq!(spec["dsn"], "dns://9.9.9.9:53");
        assert_eq!(protocol.factory().name(), "dns");
    }

    #[test]
    fn record_type_mnemonics_map_to_the_rfc_numeric_qtype() {
        assert_eq!(record_type_to_qtype("A"), 1);
        assert_eq!(record_type_to_qtype("aaaa"), 28);
        assert_eq!(record_type_to_qtype("MX"), 15);
        assert_eq!(record_type_to_qtype("28"), 28);
        assert_eq!(record_type_to_qtype("bogus"), 1);
    }
}
