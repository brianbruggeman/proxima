//! Public configuration surface for the native facade.
//!
//! Each type round-trips through `serde` for conflaguration loading +
//! exposes plain pub fields for fluent construction. (The bon-derived
//! Builder + Settings + Validate + ConfigDisplay derives compose at the
//! consumer crate; this crate's surface stays minimal to avoid pulling
//! the entire conflaguration tree into the proto-facing facade.)
//!
//! Per principle 4: `Default::default()` ≡ canonical RFC defaults so
//! `EndpointConfig::default()` produces a useful endpoint with no
//! tuning required.

use std::net::SocketAddr;

use serde::{Deserialize, Serialize};

/// Top-level endpoint configuration — the bind address + the per-side
/// configs the endpoint switches between when accepting vs. opening
/// connections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointConfig {
    /// IPv4/IPv6 socket address the endpoint binds to. Use port 0 to
    /// let the OS choose an ephemeral port.
    pub bind: SocketAddr,
    /// Default client config used by `Endpoint::connect`. `None`
    /// disables client-side use.
    pub client: Option<ClientConfig>,
    /// Default server config used by `Endpoint::accept`. `None`
    /// disables server-side accept (client-only endpoint).
    pub server: Option<ServerConfig>,
}

impl EndpointConfig {
    /// Construct a client-only endpoint bound to `addr`.
    #[must_use]
    pub fn client_only(addr: SocketAddr, client: ClientConfig) -> Self {
        Self {
            bind: addr,
            client: Some(client),
            server: None,
        }
    }

    /// Construct a server-only endpoint bound to `addr`.
    #[must_use]
    pub fn server_only(addr: SocketAddr, server: ServerConfig) -> Self {
        Self {
            bind: addr,
            client: None,
            server: Some(server),
        }
    }
}

impl Default for EndpointConfig {
    fn default() -> Self {
        use std::net::{IpAddr, Ipv4Addr};
        Self {
            bind: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 0),
            client: Some(ClientConfig::default()),
            server: None,
        }
    }
}

/// Client-side configuration.
///
/// Carries the locally-advertised transport parameters (encoded once
/// during construction). ALPN + cert verification are TLS-provider
/// concerns and travel via the `tls_alpn` field below — the actual
/// rustls-backed provider lives in the proto crate's `tls` module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientConfig {
    /// ALPN protocols offered on the client hello, in preferred-first
    /// order. Empty = no ALPN.
    pub tls_alpn: Vec<Vec<u8>>,
    /// Max idle timeout (milliseconds). 0 = disabled per RFC 9000 §18.2.
    pub max_idle_timeout_ms: u64,
    /// Initial connection-level flow-control credit (bytes).
    pub initial_max_data: u64,
    /// Initial per-stream credit for locally-opened bidi streams
    /// (peer's TP 0x05).
    pub initial_max_stream_data_bidi_local: u64,
    /// Initial per-stream credit for peer-opened bidi streams
    /// (peer's TP 0x06).
    pub initial_max_stream_data_bidi_remote: u64,
    /// Initial per-stream credit for uni streams (peer's TP 0x07).
    pub initial_max_stream_data_uni: u64,
    /// Cap on concurrent peer-initiated bidi streams.
    pub initial_max_streams_bidi: u64,
    /// Cap on concurrent peer-initiated uni streams.
    pub initial_max_streams_uni: u64,
    /// Local cap on multipath path IDs we accept. 0 disables multipath.
    pub initial_max_path_id: u64,
}

impl Default for ClientConfig {
    fn default() -> Self {
        // RFC 9000 §18.2 defaults, plus reasonable production caps.
        Self {
            tls_alpn: Vec::new(),
            max_idle_timeout_ms: 30_000,
            initial_max_data: 1_048_576,
            initial_max_stream_data_bidi_local: 65_536,
            initial_max_stream_data_bidi_remote: 65_536,
            initial_max_stream_data_uni: 65_536,
            initial_max_streams_bidi: 100,
            initial_max_streams_uni: 100,
            initial_max_path_id: 0,
        }
    }
}

/// Server-side configuration. Symmetric to [`ClientConfig`] but adds
/// the cert chain + key the TLS provider hands the server-side
/// handshake.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// ALPN protocols the server will accept, in preferred-first order.
    pub tls_alpn: Vec<Vec<u8>>,
    /// PEM-encoded cert chain (leaf first).
    pub cert_chain_pem: Vec<u8>,
    /// PEM-encoded private key (PKCS#8).
    pub private_key_pem: Vec<u8>,
    /// Max idle timeout.
    pub max_idle_timeout_ms: u64,
    /// Initial connection-level flow-control credit (bytes).
    pub initial_max_data: u64,
    /// Per-stream initial credits — see [`ClientConfig`].
    pub initial_max_stream_data_bidi_local: u64,
    pub initial_max_stream_data_bidi_remote: u64,
    pub initial_max_stream_data_uni: u64,
    pub initial_max_streams_bidi: u64,
    pub initial_max_streams_uni: u64,
    pub initial_max_path_id: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            tls_alpn: Vec::new(),
            cert_chain_pem: Vec::new(),
            private_key_pem: Vec::new(),
            max_idle_timeout_ms: 30_000,
            initial_max_data: 1_048_576,
            initial_max_stream_data_bidi_local: 65_536,
            initial_max_stream_data_bidi_remote: 65_536,
            initial_max_stream_data_uni: 65_536,
            initial_max_streams_bidi: 100,
            initial_max_streams_uni: 100,
            initial_max_path_id: 0,
        }
    }
}

impl ClientConfig {
    /// Encode the transport parameters to a byte vector ready for the
    /// TLS provider's `local_transport_parameters_wire` argument.
    ///
    /// NOTE: this omits `initial_source_connection_id`, which a real
    /// handshake REQUIRES (RFC 9000 §18.2 / §7.3) — use
    /// [`Self::encode_transport_parameters_with_source_cid`] for the live
    /// dial path. This bare form is retained for policy-only / test use.
    ///
    /// # Errors
    ///
    /// Bubbles up [`proxima_protocols::quic::transport_parameters::EncodeError`]
    /// if the values overflow the wire format (varint cap).
    pub fn encode_transport_parameters(
        &self,
    ) -> Result<Vec<u8>, proxima_protocols::quic::transport_parameters::EncodeError> {
        self.encode_transport_parameters_inner(None)
    }

    /// Like [`Self::encode_transport_parameters`] but also advertises
    /// `initial_source_connection_id` = `source_cid` (RFC 9000 §18.2 /
    /// §7.3). The peer validates the client's ISCID against the Source
    /// CID of the client's Initial and rejects the connection if it's
    /// absent. `source_cid` is the SCID the client minted for
    /// [`Connection::new_client`](proxima_protocols::quic::connection::Connection::new_client).
    ///
    /// # Errors
    ///
    /// Bubbles up [`proxima_protocols::quic::transport_parameters::EncodeError`]
    /// if the values overflow the wire format (varint cap).
    pub fn encode_transport_parameters_with_source_cid(
        &self,
        source_cid: &[u8],
    ) -> Result<Vec<u8>, proxima_protocols::quic::transport_parameters::EncodeError> {
        self.encode_transport_parameters_inner(Some(source_cid))
    }

    fn encode_transport_parameters_inner(
        &self,
        source_cid: Option<&[u8]>,
    ) -> Result<Vec<u8>, proxima_protocols::quic::transport_parameters::EncodeError> {
        let tp = proxima_protocols::quic::transport_parameters::TransportParameters {
            initial_max_data: Some(self.initial_max_data),
            max_idle_timeout_ms: Some(self.max_idle_timeout_ms),
            initial_max_stream_data_bidi_local: Some(self.initial_max_stream_data_bidi_local),
            initial_max_stream_data_bidi_remote: Some(self.initial_max_stream_data_bidi_remote),
            initial_max_stream_data_uni: Some(self.initial_max_stream_data_uni),
            initial_max_streams_bidi: Some(self.initial_max_streams_bidi),
            initial_max_streams_uni: Some(self.initial_max_streams_uni),
            initial_source_connection_id: source_cid,
            // advertise the size we actually buffer for (RFC 9000 §18.2). The
            // peer MUST NOT send a larger datagram; sized from the same const
            // as the recv buffers so the two can never drift and truncate.
            max_udp_payload_size: Some(proxima_protocols::quic::endpoint::MAX_UDP_PAYLOAD_SIZE as u64),
            initial_max_path_id: if self.initial_max_path_id > 0 {
                Some(self.initial_max_path_id)
            } else {
                None
            },
            ..Default::default()
        };
        let mut buffer = vec![0u8; 256];
        let written = tp.encode(&mut buffer)?;
        buffer.truncate(written);
        Ok(buffer)
    }
}

impl ServerConfig {
    /// Symmetric to [`ClientConfig::encode_transport_parameters`].
    ///
    /// # Errors
    ///
    /// Same as the client variant.
    pub fn encode_transport_parameters(
        &self,
    ) -> Result<Vec<u8>, proxima_protocols::quic::transport_parameters::EncodeError> {
        let tp = proxima_protocols::quic::transport_parameters::TransportParameters {
            initial_max_data: Some(self.initial_max_data),
            max_idle_timeout_ms: Some(self.max_idle_timeout_ms),
            initial_max_stream_data_bidi_local: Some(self.initial_max_stream_data_bidi_local),
            initial_max_stream_data_bidi_remote: Some(self.initial_max_stream_data_bidi_remote),
            initial_max_stream_data_uni: Some(self.initial_max_stream_data_uni),
            initial_max_streams_bidi: Some(self.initial_max_streams_bidi),
            initial_max_streams_uni: Some(self.initial_max_streams_uni),
            // advertise the size we actually buffer for (RFC 9000 §18.2). The
            // peer MUST NOT send a larger datagram; sized from the same const
            // as the recv buffers so the two can never drift and truncate.
            max_udp_payload_size: Some(proxima_protocols::quic::endpoint::MAX_UDP_PAYLOAD_SIZE as u64),
            initial_max_path_id: if self.initial_max_path_id > 0 {
                Some(self.initial_max_path_id)
            } else {
                None
            },
            ..Default::default()
        };
        let mut buffer = vec![0u8; 256];
        let written = tp.encode(&mut buffer)?;
        buffer.truncate(written);
        Ok(buffer)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn client_config_defaults_round_trip_through_serde() {
        let config = ClientConfig::default();
        let toml = toml::to_string(&config).expect("serialize");
        let back: ClientConfig = toml::from_str(&toml).expect("deserialize");
        assert_eq!(back.initial_max_data, config.initial_max_data);
        assert_eq!(back.max_idle_timeout_ms, config.max_idle_timeout_ms);
    }

    #[test]
    fn server_config_defaults_round_trip_through_serde() {
        let config = ServerConfig::default();
        let toml = toml::to_string(&config).expect("serialize");
        let back: ServerConfig = toml::from_str(&toml).expect("deserialize");
        assert_eq!(back.initial_max_data, config.initial_max_data);
    }

    #[test]
    fn client_config_encode_transport_parameters_parses_back() {
        let config = ClientConfig::default();
        let bytes = config.encode_transport_parameters().expect("encode");
        let parsed = proxima_protocols::quic::transport_parameters::parse(&bytes).expect("parse");
        assert_eq!(parsed.initial_max_data, Some(config.initial_max_data));
        assert_eq!(parsed.max_idle_timeout_ms, Some(config.max_idle_timeout_ms));
    }

    #[test]
    fn endpoint_config_default_is_client_only_loopback() {
        let config = EndpointConfig::default();
        assert!(config.client.is_some());
        assert!(config.server.is_none());
        assert!(config.bind.ip().is_loopback());
    }

    /// Principle 4 + C30 parity test — a config loaded from TOML matches
    /// a config constructed via plain struct literal, field-for-field.
    /// (The bon-derived Builder + conflaguration `Settings` derives
    /// land at the consumer layer; this surface verifies the
    /// serde-round-trip half of the parity contract.)
    #[test]
    fn endpoint_config_toml_parity_with_literal_construction() {
        use std::net::{IpAddr, Ipv4Addr, SocketAddr};
        let from_literal = EndpointConfig {
            bind: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), 4433),
            client: Some(ClientConfig {
                tls_alpn: vec![b"h3".to_vec()],
                max_idle_timeout_ms: 60_000,
                initial_max_data: 16_777_216,
                initial_max_stream_data_bidi_local: 1_048_576,
                initial_max_stream_data_bidi_remote: 1_048_576,
                initial_max_stream_data_uni: 1_048_576,
                initial_max_streams_bidi: 1024,
                initial_max_streams_uni: 1024,
                initial_max_path_id: 0,
            }),
            server: None,
        };

        let toml_str = r#"
bind = "0.0.0.0:4433"

[client]
tls_alpn = [[104, 51]]
max_idle_timeout_ms = 60000
initial_max_data = 16777216
initial_max_stream_data_bidi_local = 1048576
initial_max_stream_data_bidi_remote = 1048576
initial_max_stream_data_uni = 1048576
initial_max_streams_bidi = 1024
initial_max_streams_uni = 1024
initial_max_path_id = 0
"#;
        let from_toml: EndpointConfig = toml::from_str(toml_str).expect("toml parse");

        assert_eq!(from_literal.bind, from_toml.bind);
        let lhs = from_literal.client.as_ref().unwrap();
        let rhs = from_toml.client.as_ref().unwrap();
        assert_eq!(lhs.tls_alpn, rhs.tls_alpn);
        assert_eq!(lhs.max_idle_timeout_ms, rhs.max_idle_timeout_ms);
        assert_eq!(lhs.initial_max_data, rhs.initial_max_data);
        assert_eq!(
            lhs.initial_max_stream_data_bidi_local,
            rhs.initial_max_stream_data_bidi_local
        );
        assert_eq!(
            lhs.initial_max_stream_data_bidi_remote,
            rhs.initial_max_stream_data_bidi_remote
        );
        assert_eq!(
            lhs.initial_max_stream_data_uni,
            rhs.initial_max_stream_data_uni
        );
        assert_eq!(lhs.initial_max_streams_bidi, rhs.initial_max_streams_bidi);
        assert_eq!(lhs.initial_max_streams_uni, rhs.initial_max_streams_uni);
        assert_eq!(lhs.initial_max_path_id, rhs.initial_max_path_id);
        assert!(from_toml.server.is_none());
    }
}
