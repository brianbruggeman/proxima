//! Public configuration surface for the native facade.
//!
//! Each type round-trips through serde for conflaguration loading
//! (principle 4) — the consumer crate adds the bon Builder +
//! conflaguration Settings derives on top.

use std::time::Duration;

use proxima_protocols::quic::connection::HandshakeLimits;
use serde::{Deserialize, Serialize};

/// Default budget for completing the QUIC + HTTP/3 handshake before a client
/// connect attempt is abandoned (microseconds). Single source of truth for the
/// 30 s default; the runtime override is [`ClientConfig::handshake_timeout_micros`]
/// (conflaguration-loadable via serde), resolved by
/// [`ClientConfig::handshake_timeout`]. Tune for high-RTT / CPU-loaded peers.
pub const DEFAULT_HANDSHAKE_TIMEOUT_MICROS: u64 = 30_000_000;

/// Server-side HTTP/3 configuration.
///
/// Carries the HTTP/3 SETTINGS values the server advertises +
/// ALPN protocols. The QUIC-layer config (cert chain, transport
/// parameters, etc.) is owned by [`proxima_quic::native::ServerConfig`]
/// — the H3 facade composes the two at construction time.
///
/// The four `handshake_*` fields are the runtime-configurable override
/// surface for [`HandshakeLimits`]: absent or `null` → build-time floor
/// from `proxima-quic-proto.toml`; set → runtime override. Convert to
/// the proto-layer type via [`ServerConfig::to_handshake_limits`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// ALPN values to advertise. Defaults to `["h3"]`.
    pub alpn: Vec<Vec<u8>>,
    /// QPACK encoder's max dynamic-table capacity to advertise.
    pub qpack_max_table_capacity: u64,
    /// QPACK blocked-streams cap.
    pub qpack_blocked_streams: u64,
    /// Max decompressed field-section size we'll accept.
    pub max_field_section_size: u64,
    /// Advertise RFC 9297 H3-Datagram support.
    pub h3_datagram: bool,
    /// Advertise RFC 9220 Extended CONNECT support.
    pub enable_connect_protocol: bool,
    /// Runtime override for the early-data byte budget; `None` uses the
    /// build-time floor from `proxima-quic-proto.toml`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handshake_early_data_max_bytes: Option<usize>,
    /// Runtime override for the early-data datagram-count budget.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handshake_early_data_max_datagrams: Option<usize>,
    /// Runtime override for the early-data hold window (µs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handshake_early_data_hold_micros: Option<u64>,
    /// Runtime override for the half-open handshake expiry (µs). Lower
    /// for high-churn servers; raise for high-RTT peers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handshake_completion_micros: Option<u64>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            alpn: vec![b"h3".to_vec()],
            qpack_max_table_capacity: 0,
            qpack_blocked_streams: 0,
            // 64 KiB matches the h3-proto Settings::default() and the
            // mainstream stacks (quinn-h3, h2). i64::MAX advertised
            // "no limit" but combined with per-frame buffering in the
            // driver, a legal large POST / HEADERS could grow heap
            // until exhaustion.
            max_field_section_size: 65_536,
            h3_datagram: false,
            enable_connect_protocol: false,
            handshake_early_data_max_bytes: None,
            handshake_early_data_max_datagrams: None,
            handshake_early_data_hold_micros: None,
            handshake_completion_micros: None,
        }
    }
}

impl ServerConfig {
    /// Convert to a [`proxima_protocols::http3_codec::settings::Settings`] value.
    #[must_use]
    pub fn to_h3_settings(&self) -> proxima_protocols::http3_codec::settings::Settings {
        proxima_protocols::http3_codec::settings::Settings {
            qpack_max_table_capacity: self.qpack_max_table_capacity,
            max_field_section_size: self.max_field_section_size,
            qpack_blocked_streams: self.qpack_blocked_streams,
            h3_datagram: self.h3_datagram,
            enable_connect_protocol: self.enable_connect_protocol,
        }
    }

    /// Produce a [`HandshakeLimits`] from this config, falling back to the
    /// build-time floor for any field not explicitly overridden.
    #[must_use]
    pub fn to_handshake_limits(&self) -> HandshakeLimits {
        let defaults = HandshakeLimits::default();
        HandshakeLimits {
            early_data_max_bytes: self
                .handshake_early_data_max_bytes
                .unwrap_or(defaults.early_data_max_bytes),
            early_data_max_datagrams: self
                .handshake_early_data_max_datagrams
                .unwrap_or(defaults.early_data_max_datagrams),
            early_data_hold_micros: self
                .handshake_early_data_hold_micros
                .unwrap_or(defaults.early_data_hold_micros),
            handshake_completion_micros: self
                .handshake_completion_micros
                .unwrap_or(defaults.handshake_completion_micros),
        }
    }
}

/// Client-side HTTP/3 configuration. Symmetric to [`ServerConfig`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientConfig {
    pub alpn: Vec<Vec<u8>>,
    pub qpack_max_table_capacity: u64,
    pub qpack_blocked_streams: u64,
    pub max_field_section_size: u64,
    pub h3_datagram: bool,
    pub enable_connect_protocol: bool,
    /// Runtime override for the early-data byte budget.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handshake_early_data_max_bytes: Option<usize>,
    /// Runtime override for the early-data datagram-count budget.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handshake_early_data_max_datagrams: Option<usize>,
    /// Runtime override for the early-data hold window (µs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handshake_early_data_hold_micros: Option<u64>,
    /// Runtime override for the half-open handshake expiry (µs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handshake_completion_micros: Option<u64>,
    /// Runtime override for the connect-side handshake timeout (µs): how long
    /// `establish_connection` drives the QUIC+H3 handshake before giving up.
    /// Absent → [`DEFAULT_HANDSHAKE_TIMEOUT_MICROS`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handshake_timeout_micros: Option<u64>,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            alpn: vec![b"h3".to_vec()],
            qpack_max_table_capacity: 0,
            qpack_blocked_streams: 0,
            // 64 KiB matches the h3-proto Settings::default() and the
            // mainstream stacks (quinn-h3, h2). i64::MAX advertised
            // "no limit" but combined with per-frame buffering in the
            // driver, a legal large POST / HEADERS could grow heap
            // until exhaustion.
            max_field_section_size: 65_536,
            h3_datagram: false,
            enable_connect_protocol: false,
            handshake_early_data_max_bytes: None,
            handshake_early_data_max_datagrams: None,
            handshake_early_data_hold_micros: None,
            handshake_completion_micros: None,
            handshake_timeout_micros: None,
        }
    }
}

impl ClientConfig {
    /// Resolve the connect-side handshake timeout, falling back to
    /// [`DEFAULT_HANDSHAKE_TIMEOUT_MICROS`] when not overridden.
    #[must_use]
    pub fn handshake_timeout(&self) -> Duration {
        Duration::from_micros(
            self.handshake_timeout_micros
                .unwrap_or(DEFAULT_HANDSHAKE_TIMEOUT_MICROS),
        )
    }

    /// Convert to a [`proxima_protocols::http3_codec::settings::Settings`] value.
    #[must_use]
    pub fn to_h3_settings(&self) -> proxima_protocols::http3_codec::settings::Settings {
        proxima_protocols::http3_codec::settings::Settings {
            qpack_max_table_capacity: self.qpack_max_table_capacity,
            max_field_section_size: self.max_field_section_size,
            qpack_blocked_streams: self.qpack_blocked_streams,
            h3_datagram: self.h3_datagram,
            enable_connect_protocol: self.enable_connect_protocol,
        }
    }

    /// Produce a [`HandshakeLimits`] from this config, falling back to the
    /// build-time floor for any field not explicitly overridden.
    #[must_use]
    pub fn to_handshake_limits(&self) -> HandshakeLimits {
        let defaults = HandshakeLimits::default();
        HandshakeLimits {
            early_data_max_bytes: self
                .handshake_early_data_max_bytes
                .unwrap_or(defaults.early_data_max_bytes),
            early_data_max_datagrams: self
                .handshake_early_data_max_datagrams
                .unwrap_or(defaults.early_data_max_datagrams),
            early_data_hold_micros: self
                .handshake_early_data_hold_micros
                .unwrap_or(defaults.early_data_hold_micros),
            handshake_completion_micros: self
                .handshake_completion_micros
                .unwrap_or(defaults.handshake_completion_micros),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn server_config_defaults_round_trip_through_toml() {
        let config = ServerConfig::default();
        let toml_str = toml::to_string(&config).expect("serialize");
        let back: ServerConfig = toml::from_str(&toml_str).expect("deserialize");
        assert_eq!(back.alpn, config.alpn);
        assert_eq!(back.max_field_section_size, config.max_field_section_size);
    }

    #[test]
    fn server_config_to_h3_settings_preserves_values() {
        let config = ServerConfig {
            qpack_max_table_capacity: 4096,
            max_field_section_size: 16_384,
            h3_datagram: true,
            ..Default::default()
        };
        let settings = config.to_h3_settings();
        assert_eq!(settings.qpack_max_table_capacity, 4096);
        assert_eq!(settings.max_field_section_size, 16_384);
        assert!(settings.h3_datagram);
    }

    /// C40 parity test: TOML-loaded ServerConfig matches literal-
    /// constructed field-for-field. Per principle 4.
    #[test]
    fn server_config_toml_parity_with_literal() {
        let literal = ServerConfig {
            alpn: vec![b"h3".to_vec()],
            qpack_max_table_capacity: 4096,
            qpack_blocked_streams: 100,
            max_field_section_size: 65_536,
            h3_datagram: true,
            enable_connect_protocol: true,
            ..Default::default()
        };
        let toml_str = r#"
alpn = [[104, 51]]
qpack_max_table_capacity = 4096
qpack_blocked_streams = 100
max_field_section_size = 65536
h3_datagram = true
enable_connect_protocol = true
"#;
        let from_toml: ServerConfig = toml::from_str(toml_str).expect("toml");
        assert_eq!(literal.alpn, from_toml.alpn);
        assert_eq!(
            literal.qpack_max_table_capacity,
            from_toml.qpack_max_table_capacity
        );
        assert_eq!(
            literal.qpack_blocked_streams,
            from_toml.qpack_blocked_streams
        );
        assert_eq!(
            literal.max_field_section_size,
            from_toml.max_field_section_size
        );
        assert_eq!(literal.h3_datagram, from_toml.h3_datagram);
        assert_eq!(
            literal.enable_connect_protocol,
            from_toml.enable_connect_protocol
        );
    }

    #[test]
    fn client_config_round_trip() {
        let config = ClientConfig::default();
        let toml_str = toml::to_string(&config).expect("serialize");
        let back: ClientConfig = toml::from_str(&toml_str).expect("deserialize");
        assert_eq!(back.alpn, config.alpn);
    }

    #[test]
    fn client_handshake_timeout_defaults_to_sized_const() {
        assert_eq!(
            ClientConfig::default().handshake_timeout(),
            Duration::from_micros(DEFAULT_HANDSHAKE_TIMEOUT_MICROS)
        );
    }

    #[test]
    fn client_handshake_timeout_honours_conflaguration_override() {
        // a deployment dialing a high-RTT peer raises the budget via config;
        // prove it survives the serde round-trip conflaguration loads through.
        let config = ClientConfig {
            handshake_timeout_micros: Some(5_000_000),
            ..Default::default()
        };
        let toml_str = toml::to_string(&config).expect("serialize");
        let back: ClientConfig = toml::from_str(&toml_str).expect("deserialize");
        assert_eq!(back.handshake_timeout(), Duration::from_secs(5));
    }

    #[test]
    fn server_config_default_to_handshake_limits_matches_sized_floor() {
        use proxima_protocols::quic::connection::HandshakeLimits;
        let limits = ServerConfig::default().to_handshake_limits();
        let floor = HandshakeLimits::default();
        assert_eq!(
            limits.handshake_completion_micros,
            floor.handshake_completion_micros
        );
        assert_eq!(limits.early_data_max_bytes, floor.early_data_max_bytes);
        assert_eq!(
            limits.early_data_max_datagrams,
            floor.early_data_max_datagrams
        );
        assert_eq!(limits.early_data_hold_micros, floor.early_data_hold_micros);
    }

    #[test]
    fn server_config_override_handshake_completion_propagates() {
        let config = ServerConfig {
            handshake_completion_micros: Some(1_000),
            ..Default::default()
        };
        let limits = config.to_handshake_limits();
        assert_eq!(limits.handshake_completion_micros, 1_000);
    }

    #[test]
    fn server_config_handshake_fields_round_trip_through_toml() {
        let config = ServerConfig {
            handshake_completion_micros: Some(5_000_000),
            handshake_early_data_max_bytes: Some(32_768),
            ..Default::default()
        };
        let toml_str = toml::to_string(&config).expect("serialize");
        let back: ServerConfig = toml::from_str(&toml_str).expect("deserialize");
        assert_eq!(back.handshake_completion_micros, Some(5_000_000));
        assert_eq!(back.handshake_early_data_max_bytes, Some(32_768));
        assert_eq!(back.handshake_early_data_max_datagrams, None);
    }

    #[test]
    fn client_config_to_handshake_limits_override_propagates() {
        let config = ClientConfig {
            handshake_completion_micros: Some(2_000),
            handshake_early_data_hold_micros: Some(50_000),
            ..Default::default()
        };
        let limits = config.to_handshake_limits();
        assert_eq!(limits.handshake_completion_micros, 2_000);
        assert_eq!(limits.early_data_hold_micros, 50_000);
    }
}
