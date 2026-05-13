use std::path::Path;
use std::sync::Arc;

use proxima_core::ProximaError;
use rcgen::{
    BasicConstraints, CertificateParams, CertifiedIssuer, DistinguishedName, DnType,
    ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair, KeyUsagePurpose, PKCS_ECDSA_P256_SHA256,
};
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use time::{Duration, OffsetDateTime};

/// Sane validity window starting an hour ago. rcgen's default not_before/not_after
/// are sentinel extremes (1975 / 4096) that Chromium's cert parser rejects outright
/// ("failed parsing extensions") — Node is lenient, Chromium is not, so without this
/// no Chromium-stack (renderer) request is interceptable.
fn cert_validity(days_valid: i64) -> (OffsetDateTime, OffsetDateTime) {
    let now = OffsetDateTime::now_utc();
    (now - Duration::hours(1), now + Duration::days(days_valid))
}

// cakeypair lives behind Arc, constructed once per process; size delta is cold
#[allow(clippy::large_enum_variant)]
pub enum CaKeyPair {
    Generated {
        params: CertificateParams,
        key_pair: KeyPair,
    },
    Loaded {
        cert_pem: String,
        key_pair: KeyPair,
    },
}

pub fn generate_ca() -> Result<CaKeyPair, ProximaError> {
    let mut distinguished_name = DistinguishedName::new();
    distinguished_name.push(DnType::CommonName, "proxima intercept ca");
    distinguished_name.push(DnType::OrganizationName, "proxima");

    let key_pair = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256)
        .map_err(|err| ProximaError::Config(format!("ca: generate keypair: {err}")))?;

    let mut params = CertificateParams::new(Vec::<String>::new())
        .map_err(|err| ProximaError::Config(format!("ca: cert params: {err}")))?;
    params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    params.distinguished_name = distinguished_name;
    params.key_usages.push(KeyUsagePurpose::KeyCertSign);
    params.key_usages.push(KeyUsagePurpose::CrlSign);
    let (not_before, not_after) = cert_validity(3650);
    params.not_before = not_before;
    params.not_after = not_after;

    Ok(CaKeyPair::Generated { params, key_pair })
}

pub fn generate_domain_cert(
    ca: &CaKeyPair,
    domain: &str,
) -> Result<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>), ProximaError> {
    let mut domain_params = CertificateParams::new(vec![domain.to_string()])
        .map_err(|err| ProximaError::Config(format!("domain cert params: {err}")))?;
    // chrome requires EKU serverAuth on TLS leaf certs; without it the renderer
    // rejects the forged cert even once the CA is trusted.
    domain_params
        .extended_key_usages
        .push(ExtendedKeyUsagePurpose::ServerAuth);
    let (not_before, not_after) = cert_validity(397);
    domain_params.not_before = not_before;
    domain_params.not_after = not_after;

    let domain_key = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256)
        .map_err(|err| ProximaError::Config(format!("domain: generate keypair: {err}")))?;

    match ca {
        CaKeyPair::Generated { params, key_pair } => {
            let issuer = CertifiedIssuer::self_signed(params.clone(), key_pair)
                .map_err(|err| ProximaError::Config(format!("ca: certified issuer: {err}")))?;

            let domain_cert = domain_params
                .signed_by(&domain_key, &issuer)
                .map_err(|err| ProximaError::Config(format!("domain: sign cert: {err}")))?;

            let ca_cert_ref: &rcgen::Certificate = issuer.as_ref();
            let cert_der = CertificateDer::from(domain_cert.der().to_vec());
            let ca_der = CertificateDer::from(ca_cert_ref.der().to_vec());
            let key_der = PrivateKeyDer::Pkcs8(domain_key.serialize_der().into());

            Ok((vec![cert_der, ca_der], key_der))
        }
        CaKeyPair::Loaded { cert_pem, key_pair } => {
            let issuer = Issuer::from_ca_cert_pem(cert_pem, key_pair)
                .map_err(|err| ProximaError::Config(format!("ca: issuer from pem: {err}")))?;

            let domain_cert = domain_params
                .signed_by(&domain_key, &issuer)
                .map_err(|err| ProximaError::Config(format!("domain: sign cert: {err}")))?;

            let ca_der_raw = pem::parse(cert_pem)
                .map_err(|err| ProximaError::Config(format!("ca: parse pem for chain: {err}")))?;

            let cert_der = CertificateDer::from(domain_cert.der().to_vec());
            let ca_der = CertificateDer::from(ca_der_raw.contents().to_vec());
            let key_der = PrivateKeyDer::Pkcs8(domain_key.serialize_der().into());

            Ok((vec![cert_der, ca_der], key_der))
        }
    }
}

pub fn load_ca(cert_path: &Path, key_path: &Path) -> Result<CaKeyPair, ProximaError> {
    let key_pem = std::fs::read_to_string(key_path).map_err(|err| {
        ProximaError::Config(format!("ca: read key {}: {err}", key_path.display()))
    })?;

    let key_pair = KeyPair::from_pem(&key_pem)
        .map_err(|err| ProximaError::Config(format!("ca: parse key pem: {err}")))?;

    let cert_pem = std::fs::read_to_string(cert_path).map_err(|err| {
        ProximaError::Config(format!("ca: read cert {}: {err}", cert_path.display()))
    })?;

    Ok(CaKeyPair::Loaded { cert_pem, key_pair })
}

pub fn ca_cert_pem(ca: &CaKeyPair) -> Result<String, ProximaError> {
    match ca {
        CaKeyPair::Generated { params, key_pair } => {
            let cert = params
                .self_signed(key_pair)
                .map_err(|err| ProximaError::Config(format!("ca: self-sign for pem: {err}")))?;
            Ok(cert.pem())
        }
        CaKeyPair::Loaded { cert_pem, .. } => Ok(cert_pem.clone()),
    }
}

pub fn ca_key_pem(ca: &CaKeyPair) -> String {
    match ca {
        CaKeyPair::Generated { key_pair, .. } | CaKeyPair::Loaded { key_pair, .. } => {
            key_pair.serialize_pem()
        }
    }
}

/// SNI-driven forging cert resolver: rustls hands it the ClientHello's server
/// name, it mints (and caches) a per-host leaf signed by our CA on the fly. This
/// is the MITM keystone for any TLS surface that resolves cert by SNI rather than
/// pre-binding one host — notably QUIC/h3 (proxima-quic takes a full
/// `Arc<ServerConfig>`, so the same resolver drives the UDP path), and it
/// generalizes the per-host [`build_tls_acceptor`] to a single any-host config.
pub struct ForgingResolver {
    ca: Arc<CaKeyPair>,
    provider: Arc<rustls::crypto::CryptoProvider>,
    cache: std::sync::Mutex<std::collections::HashMap<String, Arc<rustls::sign::CertifiedKey>>>,
}

impl std::fmt::Debug for ForgingResolver {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ForgingResolver")
            .finish_non_exhaustive()
    }
}

impl ForgingResolver {
    #[must_use]
    pub fn new(ca: Arc<CaKeyPair>) -> Self {
        // the process default if installed, else aws-lc-rs (what rcgen + the
        // workspace already pull) — get_default() can be None even when
        // ServerConfig::builder() works off the compiled feature-default.
        let provider = rustls::crypto::CryptoProvider::get_default()
            .cloned()
            .unwrap_or_else(|| Arc::new(rustls::crypto::aws_lc_rs::default_provider()));
        Self {
            ca,
            provider,
            cache: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Mint-or-cache a leaf for `sni`. Separated from the trait method so it is
    /// unit-testable without constructing a rustls `ClientHello`.
    fn forge(&self, sni: &str) -> Option<Arc<rustls::sign::CertifiedKey>> {
        if let Ok(cache) = self.cache.lock()
            && let Some(certified) = cache.get(sni)
        {
            return Some(Arc::clone(certified));
        }
        let (chain, key_der) = generate_domain_cert(&self.ca, sni).ok()?;
        let signing_key = self.provider.key_provider.load_private_key(key_der).ok()?;
        let certified = Arc::new(rustls::sign::CertifiedKey::new(chain, signing_key));
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(sni.to_string(), Arc::clone(&certified));
        }
        Some(certified)
    }
}

impl rustls::server::ResolvesServerCert for ForgingResolver {
    fn resolve(
        &self,
        client_hello: rustls::server::ClientHello<'_>,
    ) -> Option<Arc<rustls::sign::CertifiedKey>> {
        self.forge(client_hello.server_name()?)
    }
}

/// A single `ServerConfig` that forges a cert for ANY SNI via [`ForgingResolver`],
/// with the given ALPN. The TCP path can use this instead of one acceptor per
/// host; the QUIC/h3 server requires exactly this shape (it takes an
/// `Arc<ServerConfig>` and resolves cert by SNI through rustls).
pub fn forging_server_config(ca: Arc<CaKeyPair>, alpn: Vec<Vec<u8>>) -> Arc<rustls::ServerConfig> {
    let mut config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_cert_resolver(Arc::new(ForgingResolver::new(ca)));
    config.alpn_protocols = alpn;
    Arc::new(config)
}

pub fn build_tls_acceptor(
    ca: &CaKeyPair,
    domain: &str,
    offer_h2: bool,
) -> Result<tokio_rustls::TlsAcceptor, ProximaError> {
    let (certs, key) = generate_domain_cert(ca, domain)?;

    let mut config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|err| ProximaError::Config(format!("tls acceptor: {err}")))?;

    // h2 is advertised ONLY for hosts we intend to terminate as h2 (e.g. cursor);
    // an empty ALPN keeps the proven h1 integrations on http/1.1, so enabling this
    // for one host cannot regress another by accidentally negotiating h2.
    if offer_h2 {
        config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    }

    Ok(tokio_rustls::TlsAcceptor::from(Arc::new(config)))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn generate_ca_succeeds() {
        let _ca = generate_ca().expect("generate ca");
    }

    #[test]
    fn generate_domain_cert_produces_chain() {
        let ca = generate_ca().expect("generate ca");
        let (chain, _key) = generate_domain_cert(&ca, "example.com").expect("domain cert");
        assert_eq!(chain.len(), 2, "chain should have domain cert + ca cert");
    }

    #[test]
    fn build_tls_acceptor_succeeds() {
        let ca = generate_ca().expect("generate ca");
        let _acceptor = build_tls_acceptor(&ca, "api.github.com", false).expect("build acceptor");
    }

    #[test]
    fn forging_resolver_mints_and_caches_per_sni() {
        let ca = Arc::new(generate_ca().expect("generate ca"));
        let resolver = ForgingResolver::new(ca);
        let first = resolver
            .forge("api2.example.com")
            .expect("forge a leaf for the sni");
        assert!(!first.cert.is_empty(), "minted leaf carries a cert chain");
        let again = resolver.forge("api2.example.com").expect("second forge");
        assert!(
            Arc::ptr_eq(&first, &again),
            "same sni returns the cached CertifiedKey"
        );
        let other = resolver
            .forge("api3.example.com")
            .expect("forge a different sni");
        assert!(
            !Arc::ptr_eq(&first, &other),
            "a distinct sni mints a distinct cert"
        );
    }
}
