//! Runtime-agnostic HTTP/3 client facade. Symmetric to
//! [`super::server::Server`].

use std::task::{Context, Poll};

use proxima_protocols::http3_codec::client::{ClientConnection, ClientState, H3ClientEvent};
use proxima_protocols::http3_codec::server::StreamId;
use proxima_protocols::http3_codec::settings::Settings;
use proxima_quic::native::{Endpoint, EndpointError};
use proxima_protocols::quic::connection::{ConnectionError, ConnectionState, TimerOutcome};
use proxima_protocols::quic::streams::StreamDirection;
use proxima_protocols::quic::time::Instant;
use proxima_protocols::quic::tls::TlsProvider;

use super::config::ClientConfig;
use super::driver::{DriverState, drive_client_step};

/// Client-side facade error.
#[derive(Debug)]
#[non_exhaustive]
pub enum ClientError {
    Endpoint(EndpointError),
    H3(proxima_protocols::http3_codec::client::ClientError),
    Driver(proxima_protocols::quic::connection::ConnectionError),
    IllegalInState {
        state: &'static str,
        method: &'static str,
    },
    /// All concurrent QUIC bidi stream slots are in use, or the peer's
    /// cumulative MAX_STREAMS limit (RFC 9000 §4.6) prevents opening
    /// another stream. The connection is healthy. Caller should wait for
    /// in-flight streams to complete and for the peer to issue a
    /// MAX_STREAMS frame, then retry [`Client::open_request`].
    StreamCreditExhausted,
}

impl core::fmt::Display for ClientError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Endpoint(err) => write!(f, "endpoint: {err}"),
            Self::H3(err) => write!(f, "h3: {err:?}"),
            Self::Driver(err) => write!(f, "driver: {err:?}"),
            Self::IllegalInState { state, method } => {
                write!(f, "illegal in state {state}: method {method}")
            }
            Self::StreamCreditExhausted => f.write_str(
                "stream credit exhausted: all bidi slots in use or peer MAX_STREAMS limit reached",
            ),
        }
    }
}

impl std::error::Error for ClientError {}

impl From<EndpointError> for ClientError {
    fn from(err: EndpointError) -> Self {
        Self::Endpoint(err)
    }
}

impl From<proxima_protocols::http3_codec::client::ClientError> for ClientError {
    fn from(err: proxima_protocols::http3_codec::client::ClientError) -> Self {
        Self::H3(err)
    }
}

/// HTTP/3 client facade.
pub struct Client<P: TlsProvider> {
    config: ClientConfig,
    endpoint: Endpoint<P>,
    h3: ClientConnection,
    driver_state: DriverState,
}

impl<P: TlsProvider> Client<P> {
    /// Construct from a connected QUIC [`Endpoint`].
    #[must_use]
    pub fn new(config: ClientConfig, endpoint: Endpoint<P>) -> Self {
        let settings = config.to_h3_settings();
        let h3 = ClientConnection::new(settings);
        Self {
            config,
            endpoint,
            h3,
            driver_state: DriverState::new(),
        }
    }

    /// Run one driver pass — symmetric to
    /// [`super::server::Server::drive`].
    ///
    /// # Errors
    ///
    /// Bubbles QUIC / H3 errors from [`drive_client_step`].
    pub fn drive(&mut self) -> Result<(), ClientError> {
        if !matches!(
            self.endpoint.connection().state(),
            ConnectionState::Established(_)
        ) {
            return Ok(());
        }
        drive_client_step(
            self.endpoint.connection_mut(),
            &mut self.h3,
            &mut self.driver_state,
        )
        .map_err(ClientError::Driver)
    }

    /// Advance the QUIC connection's timers to `now` — fires ACK-delay,
    /// PTO retransmit, idle, close, and drain timers. A driver loop MUST
    /// call this (a select against [`Self::next_timeout`], or a periodic
    /// tick) or the handshake stalls: without it the client never flushes
    /// delayed ACKs and never retransmits a lost flight.
    ///
    /// # Errors
    ///
    /// Bubbles QUIC errors from the endpoint.
    pub fn handle_timeout(&mut self, now: Instant) -> Result<TimerOutcome, ClientError> {
        Ok(self.endpoint.handle_timeout(now)?)
    }

    /// Whether the underlying QUIC connection has reached Established —
    /// diagnostic introspection for driver loops + tests.
    #[must_use]
    pub fn quic_established(&self) -> bool {
        matches!(
            self.endpoint.connection().state(),
            ConnectionState::Established(_)
        )
    }

    /// Name of the underlying QUIC connection state — diagnostic only.
    #[must_use]
    pub fn quic_state_name(&self) -> &'static str {
        match self.endpoint.connection().state() {
            ConnectionState::Initial(_) => "initial",
            ConnectionState::Handshake(_) => "handshake",
            ConnectionState::Established(_) => "established",
            ConnectionState::Closing(_) => "closing",
            ConnectionState::Draining(_) => "draining",
            ConnectionState::Closed => "closed",
            _ => "other",
        }
    }

    /// Next timer deadline (PTO / ACK-delay / idle / close / drain), or
    /// `None` when no timer is armed. Drive a sleep against this and call
    /// [`Self::handle_timeout`] when it elapses.
    #[must_use]
    pub fn next_timeout(&self) -> Option<Instant> {
        self.endpoint.next_timeout()
    }

    /// Borrow the loaded config.
    #[must_use]
    pub fn config(&self) -> &ClientConfig {
        &self.config
    }

    /// Borrow the H3 state machine.
    pub fn h3(&self) -> &ClientConnection {
        &self.h3
    }

    /// Mutably borrow the H3 state machine.
    pub fn h3_mut(&mut self) -> &mut ClientConnection {
        &mut self.h3
    }

    /// Current H3 state.
    #[must_use]
    pub fn state(&self) -> &ClientState {
        self.h3.state()
    }

    /// Peer's negotiated SETTINGS.
    #[must_use]
    pub fn peer_settings(&self) -> Option<&Settings> {
        self.h3.peer_settings()
    }

    /// Drain pending events.
    #[must_use]
    pub fn poll_event(&mut self) -> Option<H3ClientEvent> {
        self.h3.poll_event()
    }

    /// Drain outbound proto datagrams to the socket. Runs
    /// [`Self::drive`] first so any H3-queued bytes get staged before
    /// the next datagram is built.
    ///
    /// # Errors
    ///
    /// See [`ClientError`].
    pub fn poll_send(
        &mut self,
        cx: &mut Context<'_>,
        now: Instant,
    ) -> Poll<Result<bool, ClientError>> {
        if let Err(err) = self.drive() {
            return Poll::Ready(Err(err));
        }
        match self.endpoint.poll_send(cx, now) {
            Poll::Ready(Ok(sent)) => Poll::Ready(Ok(sent)),
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(err)) => Poll::Ready(Err(err.into())),
        }
    }

    /// Drain inbound socket datagrams into the proto state machine,
    /// then run [`Self::drive`] so any newly-readable stream bytes
    /// route into the H3 layer before the caller's next poll_event.
    ///
    /// # Errors
    ///
    /// See [`ClientError`].
    pub fn poll_recv(
        &mut self,
        cx: &mut Context<'_>,
        now: Instant,
    ) -> Poll<Result<bool, ClientError>> {
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

    /// Batch-recv wrapper: drain up to `CLIENT_RECV_BATCH_CAP` inbound
    /// datagrams in one `recvmmsg` call, feed them all to the proto, then
    /// run [`Self::drive`]. Replaces the `now_or_never(poll_recv)` loop in
    /// the multiplexed bench driver — one syscall per wakeup instead of N.
    ///
    /// # Errors
    ///
    /// See [`ClientError`].
    pub fn poll_recv_batch(
        &mut self,
        cx: &mut Context<'_>,
        now: Instant,
    ) -> Poll<Result<usize, ClientError>> {
        let count = match self.endpoint.poll_recv_batch(cx, now) {
            Poll::Ready(Ok(count)) => count,
            Poll::Pending => return Poll::Pending,
            Poll::Ready(Err(err)) => return Poll::Ready(Err(err.into())),
        };
        if count > 0
            && let Err(err) = self.drive()
        {
            return Poll::Ready(Err(err));
        }
        Poll::Ready(Ok(count))
    }

    /// Batch-send wrapper: drain ALL pending proto datagrams into one
    /// `sendmmsg` call. Replaces the `loop { poll_send }` in the multiplexed
    /// bench driver — one syscall per pass instead of one per datagram.
    ///
    /// # Errors
    ///
    /// See [`ClientError`].
    pub fn poll_send_batch(
        &mut self,
        cx: &mut Context<'_>,
        now: Instant,
    ) -> Poll<Result<usize, ClientError>> {
        if !matches!(
            self.endpoint.connection().state(),
            ConnectionState::Established(_)
        ) {
            return Poll::Ready(Ok(0));
        }
        if let Err(err) = self.drive() {
            return Poll::Ready(Err(err));
        }
        match self.endpoint.poll_send_batch(cx, now) {
            Poll::Ready(Ok(sent)) => Poll::Ready(Ok(sent)),
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(err)) => Poll::Ready(Err(err.into())),
        }
    }

    /// Open a request — opens a QUIC bidi stream and queues the H3 HEADERS.
    ///
    /// The QUIC stream is opened BEFORE any H3 state is mutated so the
    /// call is fully atomic: either both succeed or neither happens. This
    /// prevents the connection from being poisoned by a QUIC stream that
    /// the H3 layer accepted but QUIC could not back (RFC 9000 §4.6 limit
    /// exhausted or concurrent stream-table cap reached).
    ///
    /// # Errors
    ///
    /// [`ClientError::StreamCreditExhausted`] when all concurrent bidi
    /// QUIC stream slots are occupied or the peer's cumulative MAX_STREAMS
    /// limit (RFC 9000 §4.6) blocks the open. The connection is healthy;
    /// caller should retry after existing streams complete and the server
    /// issues a MAX_STREAMS frame.
    ///
    /// [`ClientError::H3`] for H3 framing / state-machine errors.
    pub fn open_request(&mut self, headers: &[(&[u8], &[u8])]) -> Result<StreamId, ClientError> {
        self.endpoint
            .connection_mut()
            .open_stream(StreamDirection::Bidi)
            .map_err(|err| match err {
                ConnectionError::PeerStreamLimitExhausted => ClientError::StreamCreditExhausted,
                other => ClientError::Driver(other),
            })?;
        self.h3.open_request(headers).map_err(Into::into)
    }

    /// Append body bytes to a request.
    ///
    /// # Errors
    ///
    /// See [`ClientError::H3`].
    pub fn send_request_data(
        &mut self,
        stream_id: StreamId,
        bytes: &[u8],
    ) -> Result<(), ClientError> {
        self.h3
            .send_request_data(stream_id, bytes)
            .map_err(Into::into)
    }

    /// Mark the request complete (FIN on next stream write).
    ///
    /// # Errors
    ///
    /// See [`ClientError::H3`].
    pub fn finish_request(&mut self, stream_id: StreamId) -> Result<(), ClientError> {
        self.h3.finish_request(stream_id).map_err(Into::into)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn client_error_displays_endpoint_variant() {
        let err = ClientError::Endpoint(EndpointError::UnconfiguredSide);
        let formatted = format!("{err}");
        assert!(formatted.contains("endpoint:"));
    }
}
