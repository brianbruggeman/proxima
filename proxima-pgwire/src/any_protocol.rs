//! `PgWireAnyProtocol` — pgwire as an [`AnyProtocol`] candidate for the open
//! universal listener (`Listener::builder().accept("pgwire")` /
//! `AnyListenProtocol`), replacing the standalone `PgWireListenProtocol`
//! bind+accept loop it used to own.
//!
//! Positive-match probe: a real PostgreSQL wire connection opens with an
//! 8-byte prefix — a 4-byte big-endian message length, then a 4-byte
//! protocol code — either the v3.0 `StartupMessage` code (`196608`) or the
//! `SSLRequest` code (`80877103`, RFC-shaped: PostgreSQL's own magic
//! constant, not a real protocol version). Both are legitimate "this is
//! pgwire" signals; RESP inline commands and every other candidate's own
//! wire never produce either 4-byte value at that offset.
//!
//! `drive` carries its own engine (`query`, `auth`, `config`, `registry`)
//! as struct fields — the same asymmetry `PgWireListenProtocol` always
//! had (the generic `AnyHandler` parameter is unused here; downcasting it
//! would just fail since pgwire's handler shape is a typed
//! [`PgPipeHandle`], not a [`proxima_primitives::pipe::handler::PipeHandle`]).
//! Each accepted connection builds a FRESH [`PgWireConnectionPipe`] carrying
//! THIS connection's [`ConnAdmission`] clone (the shared listener-wide
//! counter itself; cloning is one `Arc` bump), erases it, and hands it to
//! [`proxima_listen::serve_pipe::handle_connection`] — the ONE
//! CONNECT-request/upgrade-handler driver every connection-as-Pipe
//! protocol (pgwire, redis) now shares instead of each carrying its own
//! byte-identical copy.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use proxima_core::ProximaError;
use proxima_listen::admission::ConnAdmission;
use proxima_listen::any::{AnyHandler, AnyProtocol, ProbeVerdict};
use proxima_primitives::pipe::handler::into_handle;
use proxima_primitives::stream::{PeerInfo, StreamConnection};

use crate::auth::PgAuth;
use crate::config::PgServerConfig;
use crate::connection::CancelRegistry;
use crate::error::ServeError;
use crate::pipe::PgWireConnectionPipe;
use crate::pipes::PgPipeHandle;

#[cfg(feature = "tls")]
type TlsAcceptor = futures_rustls::TlsAcceptor;
#[cfg(not(feature = "tls"))]
type TlsAcceptor = ();

/// Length-prefixed protocol code a real PostgreSQL wire connection's first
/// 8 bytes carry: v3.0 `StartupMessage` (`196608` = `3 << 16`).
const STARTUP_PROTOCOL_V3: i32 = 196_608;
/// `SSLRequest`'s magic code (`1234 << 16 | 5679`) — PostgreSQL's own
/// constant, sent in place of a real protocol version to request TLS
/// before the real startup packet.
const SSL_REQUEST_CODE: i32 = 80_877_103;

/// PostgreSQL wire candidate for the open universal listener. Mounts via
/// `Listener::builder().any_handler("pgwire", ..)` /
/// `.accept("pgwire")` — a single-candidate registration, since RESP
/// inline commands (no length-prefixed classifier signal at all) put
/// redis's own reachable-by-classifier surface out of scope for the
/// shared open registry the same way pgwire's positive 8-byte match does
/// for everyone else.
pub struct PgWireAnyProtocol {
    label: String,
    query: PgPipeHandle,
    config: PgServerConfig,
    auth_override: Option<PgAuth>,
    registry: Arc<CancelRegistry>,
    /// Built ONCE here, not per connection — `drive` installs this SAME
    /// `Arc` onto every fresh per-connection `PgWireConnectionPipe` it
    /// builds (see `PgWireConnectionPipe::with_broker`'s doc for why a
    /// fresh broker per connection would silently break LISTEN/NOTIFY
    /// across connections).
    broker: Arc<crate::broker::NotifyBroker>,
}

impl PgWireAnyProtocol {
    #[must_use]
    pub fn new(label: impl Into<String>, query: PgPipeHandle) -> Self {
        Self {
            label: label.into(),
            query,
            config: PgServerConfig::default(),
            auth_override: None,
            registry: Arc::new(CancelRegistry::new()),
            broker: Arc::new(crate::broker::NotifyBroker::new()),
        }
    }

    /// Replaces the default [`PgServerConfig`]; a `pgwire` object in the
    /// listener spec still wins at drive time.
    #[must_use]
    pub fn with_config(mut self, config: PgServerConfig) -> Self {
        self.config = config;
        self
    }

    /// Installs an authentication policy directly, overriding the config's
    /// auth section.
    #[must_use]
    pub fn with_auth(mut self, auth: PgAuth) -> Self {
        self.auth_override = Some(auth);
        self
    }
}

fn resolve_config(base: &PgServerConfig, spec: &Value) -> Result<PgServerConfig, ProximaError> {
    match spec.get("pgwire") {
        None => Ok(base.clone()),
        Some(overrides) => serde_json::from_value(overrides.clone())
            .map_err(|error| ProximaError::Config(format!("pgwire spec: {error}"))),
    }
}

#[cfg(feature = "tls")]
fn resolve_tls(spec: &Value) -> Result<Option<TlsAcceptor>, ProximaError> {
    let config = proxima_tls::config_from_spec_value(spec.get(proxima_tls::SPEC_KEY))?;
    config
        .map(|config| proxima_tls::build_acceptor_futures_io(&config))
        .transpose()
}

#[cfg(not(feature = "tls"))]
fn resolve_tls(_spec: &Value) -> Result<Option<TlsAcceptor>, ProximaError> {
    Ok(None)
}

impl AnyProtocol for PgWireAnyProtocol {
    fn name(&self) -> &str {
        &self.label
    }

    /// 4-byte length + 4-byte protocol code.
    fn max_prefix_bytes(&self) -> usize {
        8
    }

    fn probe(&self, prefix: &[u8]) -> ProbeVerdict {
        if prefix.len() < 8 {
            return ProbeVerdict::NeedMore { at_least: 8 };
        }
        let code = i32::from_be_bytes([prefix[4], prefix[5], prefix[6], prefix[7]]);
        if code == STARTUP_PROTOCOL_V3 || code == SSL_REQUEST_CODE {
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
            let auth = match &self.auth_override {
                Some(auth) => auth.clone(),
                None => config
                    .build_auth()
                    .map_err(|error: ServeError| ProximaError::Config(error.to_string()))?,
            };
            let tls = resolve_tls(spec)?;
            let connection_pipe = PgWireConnectionPipe::new(
                self.label.clone(),
                self.query.clone(),
                auth,
                Arc::new(config),
                Arc::clone(&self.registry),
            )
            .with_broker(Arc::clone(&self.broker))
            .with_admission(admission.clone());
            #[cfg(feature = "tls")]
            let connection_pipe = connection_pipe.with_tls(tls);
            #[cfg(not(feature = "tls"))]
            let _ = tls;
            let pipe = into_handle(connection_pipe);
            proxima_listen::serve_pipe::handle_connection(stream, pipe).await
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn probe_matches_startup_v3_and_ssl_request_and_rejects_short_or_unknown() {
        let protocol = PgWireAnyProtocol::new("pgwire", crate::pipes::into_pg_handle(NeverCalled));

        let mut startup = vec![0_u8, 0, 0, 8];
        startup.extend_from_slice(&STARTUP_PROTOCOL_V3.to_be_bytes());
        assert_eq!(
            protocol.probe(&startup),
            ProbeVerdict::Match { consumed: 0 }
        );

        let mut ssl_request = vec![0_u8, 0, 0, 8];
        ssl_request.extend_from_slice(&SSL_REQUEST_CODE.to_be_bytes());
        assert_eq!(
            protocol.probe(&ssl_request),
            ProbeVerdict::Match { consumed: 0 }
        );

        assert_eq!(protocol.probe(b"short"), ProbeVerdict::NeedMore { at_least: 8 });

        let unknown = vec![0_u8, 0, 0, 8, 0, 0, 0, 1];
        assert_eq!(protocol.probe(&unknown), ProbeVerdict::No);
    }

    struct NeverCalled;

    impl proxima_primitives::pipe::SendPipe for NeverCalled {
        type In = crate::pipes::PgRequest;
        type Out = crate::pipes::PgResponse;
        type Err = ProximaError;

        async fn call(&self, _request: Self::In) -> Result<Self::Out, ProximaError> {
            unreachable!("probe tests never dispatch to the engine")
        }
    }
}
