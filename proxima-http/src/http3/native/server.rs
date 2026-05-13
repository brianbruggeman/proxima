//! Runtime-agnostic HTTP/3 server facade.
//!
//! Owns the QUIC endpoint + a per-connection
//! [`proxima_protocols::http3_codec::server::ServerConnection`] state machine.
//! Drives both via `poll_*` so any executor can schedule.
//!
//! # Scope
//!
//! C39 minimal viable: single-connection server (one accepted QUIC
//! peer per [`Server`]). Multi-connection accept + per-connection
//! task fan-out lands at the consumer layer (`proxima::listeners`)
//! during the Phase D2 cutover, using the same `poll_*` shape.

use std::task::{Context, Poll};

use proxima_protocols::http3_codec::server::{H3ServerEvent, ServerConnection, ServerState, StreamId};
use proxima_protocols::http3_codec::settings::Settings;
use proxima_quic::native::{Endpoint, EndpointError};
use proxima_protocols::quic::connection::ConnectionState;
use proxima_protocols::quic::time::Instant;
use proxima_protocols::quic::tls::TlsProvider;

use super::config::ServerConfig;
use super::driver::{DriverState, drive_server_step};

/// Server-side facade error.
#[derive(Debug)]
#[non_exhaustive]
pub enum ServerError {
    /// QUIC-layer error from the endpoint.
    Endpoint(EndpointError),
    /// H3-layer error from the sans-IO state machine.
    H3(proxima_protocols::http3_codec::server::ServerError),
    /// Per-connection driver step failed (QUIC stream-routing layer).
    Driver(proxima_protocols::quic::connection::ConnectionError),
    /// Method called outside the legal state.
    IllegalInState {
        state: &'static str,
        method: &'static str,
    },
}

impl core::fmt::Display for ServerError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Endpoint(err) => write!(f, "endpoint: {err}"),
            Self::H3(err) => write!(f, "h3: {err:?}"),
            Self::Driver(err) => write!(f, "driver: {err:?}"),
            Self::IllegalInState { state, method } => {
                write!(f, "illegal in state {state}: method {method}")
            }
        }
    }
}

impl std::error::Error for ServerError {}

impl From<EndpointError> for ServerError {
    fn from(err: EndpointError) -> Self {
        Self::Endpoint(err)
    }
}

impl From<proxima_protocols::http3_codec::server::ServerError> for ServerError {
    fn from(err: proxima_protocols::http3_codec::server::ServerError) -> Self {
        Self::H3(err)
    }
}

/// HTTP/3 server facade — one accepted QUIC connection serving many
/// concurrent H3 requests.
pub struct Server<P: TlsProvider> {
    config: ServerConfig,
    endpoint: Endpoint<P>,
    h3: ServerConnection,
    /// Per-connection stream-routing state for [`drive_server_step`].
    /// Owned by the facade so callers don't need to thread it through
    /// every send/recv call.
    driver_state: DriverState,
}

impl<P: TlsProvider> Server<P> {
    /// Wrap an already-bound [`Endpoint`] + construct the H3
    /// connection state machine seeded with the local SETTINGS values
    /// derived from `config`.
    #[must_use]
    pub fn new(config: ServerConfig, endpoint: Endpoint<P>) -> Self {
        let settings = config.to_h3_settings();
        let h3 = ServerConnection::new(settings);
        Self {
            config,
            endpoint,
            h3,
            driver_state: DriverState::new(),
        }
    }

    /// Run one driver pass: route any newly-readable QUIC stream bytes
    /// through the H3 state machine and ship any H3-queued bytes back
    /// to QUIC. Idempotent; callers may invoke at any point in their
    /// event loop. The QUIC connection must already be Established
    /// (the driver opens the local control stream lazily on first
    /// successful call).
    ///
    /// # Errors
    ///
    /// Bubbles QUIC / H3 errors from [`drive_server_step`].
    pub fn drive(&mut self) -> Result<(), ServerError> {
        if !matches!(
            self.endpoint.connection().state(),
            ConnectionState::Established(_)
        ) {
            // Pre-Established → nothing for the H3 state machine to do.
            return Ok(());
        }
        drive_server_step(
            self.endpoint.connection_mut(),
            &mut self.h3,
            &mut self.driver_state,
        )
        .map_err(ServerError::Driver)
    }

    /// Borrow the loaded config (for diagnostics).
    #[must_use]
    pub fn config(&self) -> &ServerConfig {
        &self.config
    }

    /// Borrow the H3 state machine for in-flight introspection.
    pub fn h3(&self) -> &ServerConnection {
        &self.h3
    }

    /// Mutably borrow the H3 state machine.
    pub fn h3_mut(&mut self) -> &mut ServerConnection {
        &mut self.h3
    }

    /// Current H3 state.
    #[must_use]
    pub fn state(&self) -> &ServerState {
        self.h3.state()
    }

    /// Drain the next pending H3 event.
    #[must_use]
    pub fn poll_event(&mut self) -> Option<H3ServerEvent> {
        self.h3.poll_event()
    }

    /// Borrow the peer's negotiated SETTINGS (None until SETTINGS
    /// exchange completes).
    #[must_use]
    pub fn peer_settings(&self) -> Option<&Settings> {
        self.h3.peer_settings()
    }

    /// Drive at most one outbound datagram from the QUIC layer to the
    /// socket. Runs [`Self::drive`] FIRST so any H3-queued bytes get
    /// staged onto the QUIC layer before the datagram is built; mirrors
    /// [`Endpoint::poll_send`] for the rest.
    ///
    /// # Errors
    ///
    /// See [`ServerError`].
    pub fn poll_send(
        &mut self,
        cx: &mut Context<'_>,
        now: Instant,
    ) -> Poll<Result<bool, ServerError>> {
        if let Err(err) = self.drive() {
            return Poll::Ready(Err(err));
        }
        match self.endpoint.poll_send(cx, now) {
            Poll::Ready(Ok(sent)) => Poll::Ready(Ok(sent)),
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(err)) => Poll::Ready(Err(err.into())),
        }
    }

    /// Drive at most one inbound datagram from the socket into the
    /// QUIC state machine, then run [`Self::drive`] so any newly-
    /// readable stream bytes route through the H3 state machine before
    /// the caller next polls events.
    ///
    /// # Errors
    ///
    /// See [`ServerError`].
    pub fn poll_recv(
        &mut self,
        cx: &mut Context<'_>,
        now: Instant,
    ) -> Poll<Result<bool, ServerError>> {
        let received = match self.endpoint.poll_recv(cx, now) {
            Poll::Ready(Ok(received)) => received,
            Poll::Pending => return Poll::Pending,
            Poll::Ready(Err(err)) => return Poll::Ready(Err(err.into())),
        };
        if let Err(err) = self.drive() {
            return Poll::Ready(Err(err));
        }
        Poll::Ready(Ok(received))
    }

    /// Send a response on `stream_id`.
    ///
    /// # Errors
    ///
    /// See [`ServerError::H3`].
    pub fn send_response_headers(
        &mut self,
        stream_id: StreamId,
        headers: &[(&[u8], &[u8])],
    ) -> Result<(), ServerError> {
        self.h3
            .send_response_headers(stream_id, headers)
            .map_err(Into::into)
    }

    /// Append body bytes to a response.
    ///
    /// # Errors
    ///
    /// See [`ServerError::H3`].
    pub fn send_response_data(
        &mut self,
        stream_id: StreamId,
        bytes: &[u8],
    ) -> Result<(), ServerError> {
        self.h3
            .send_response_data(stream_id, bytes)
            .map_err(Into::into)
    }

    /// Mark the response complete (sets FIN on next stream write).
    ///
    /// # Errors
    ///
    /// See [`ServerError::H3`].
    pub fn finish_response(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        self.h3.finish_response(stream_id).map_err(Into::into)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn server_error_displays_endpoint_variant() {
        let err = ServerError::Endpoint(EndpointError::UnconfiguredSide);
        let formatted = format!("{err}");
        assert!(formatted.contains("endpoint:"));
    }
}
