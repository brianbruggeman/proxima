//! Listener-spec resolution shared by both pgwire mounts.
//!
//! A listener hands the protocol a `serde_json::Value` spec at serve time;
//! pgwire reads two keys out of it — its own `pgwire` object (a full
//! [`PgServerConfig`] overriding whatever the mount was built with) and the
//! workspace-wide `proxima_tls::SPEC_KEY`. Both [`crate::listen`] and
//! [`crate::any_protocol`] mount the same connection pipe, so they read the
//! spec the same way; this is that reading, written once.

use serde_json::Value;

use proxima_core::ProximaError;

use crate::config::PgServerConfig;

#[cfg(feature = "tls")]
pub(crate) type TlsAcceptor = futures_rustls::TlsAcceptor;
#[cfg(not(feature = "tls"))]
pub(crate) type TlsAcceptor = ();

/// A `pgwire` object in the spec replaces the mount's own config wholesale;
/// absent, the mount's config stands.
pub(crate) fn resolve_config(
    base: &PgServerConfig,
    spec: &Value,
) -> Result<PgServerConfig, ProximaError> {
    match spec.get("pgwire") {
        None => Ok(base.clone()),
        Some(overrides) => serde_json::from_value(overrides.clone())
            .map_err(|error| ProximaError::Config(format!("pgwire spec: {error}"))),
    }
}

/// Builds the SSLRequest acceptor from the listener's TLS section, exactly
/// as the HTTP listeners do. `None` means the mount serves plaintext and
/// refuses SSL.
#[cfg(feature = "tls")]
pub(crate) fn resolve_tls(spec: &Value) -> Result<Option<TlsAcceptor>, ProximaError> {
    let config = proxima_tls::config_from_spec_value(spec.get(proxima_tls::SPEC_KEY))?;
    config
        .map(|config| proxima_tls::build_acceptor_futures_io(&config))
        .transpose()
}

#[cfg(not(feature = "tls"))]
pub(crate) fn resolve_tls(_spec: &Value) -> Result<Option<TlsAcceptor>, ProximaError> {
    Ok(None)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use serde_json::json;

    use super::*;

    #[test]
    fn spec_without_a_pgwire_object_keeps_the_mount_config() {
        let base = PgServerConfig::builder().max_portals(9).build();

        let resolved =
            resolve_config(&base, &json!({"other": 1})).expect("an absent key must not fail");

        assert_eq!(resolved.max_portals, 9);
    }

    #[test]
    fn pgwire_object_in_the_spec_replaces_the_mount_config() {
        let base = PgServerConfig::builder().max_portals(9).build();

        let resolved = resolve_config(&base, &json!({"pgwire": {"max_portals": 4}}))
            .expect("a well-formed override must parse");

        assert_eq!(resolved.max_portals, 4);
    }

    #[test]
    fn malformed_pgwire_object_is_a_config_error() {
        let base = PgServerConfig::default();

        let outcome = resolve_config(&base, &json!({"pgwire": {"max_portals": "lots"}}));

        assert!(
            matches!(outcome, Err(ProximaError::Config(_))),
            "a non-numeric slot count must surface as a config error"
        );
    }

    #[test]
    fn spec_without_a_tls_section_yields_no_acceptor() {
        let acceptor = resolve_tls(&json!({})).expect("an absent tls section must not fail");

        assert!(acceptor.is_none(), "plaintext mount refuses SSL");
    }
}
