use std::path::Path;
use std::sync::Arc;

use proxima_core::ProximaError;
use rcgen::{
    BasicConstraints, CertificateParams, CertifiedIssuer, DistinguishedName, DnType,
    ExtendedKeyUsagePurpose, GeneralSubtree, IsCa, Issuer, KeyPair, KeyUsagePurpose,
    NameConstraints, PKCS_ECDSA_P256_SHA256,
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
    build_ca(None)
}

/// Same as [`generate_ca`], but the minted CA carries an RFC 5280 4.2.1.10
/// NameConstraints extension restricting which DNS names it may sign for.
/// rcgen always writes this extension critical, so a verifier that doesn't
/// understand it must reject the whole chain rather than silently ignore it.
pub fn generate_constrained_ca(permitted_dns_names: &[&str]) -> Result<CaKeyPair, ProximaError> {
    let permitted_subtrees = permitted_dns_names
        .iter()
        .map(|name| GeneralSubtree::DnsName((*name).to_string()))
        .collect();
    build_ca(Some(NameConstraints {
        permitted_subtrees,
        excluded_subtrees: Vec::new(),
    }))
}

fn build_ca(name_constraints: Option<NameConstraints>) -> Result<CaKeyPair, ProximaError> {
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
    params.name_constraints = name_constraints;
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
    use rustls::RootCertStore;
    use rustls::client::WebPkiServerVerifier;
    use rustls::client::danger::ServerCertVerifier;
    use rustls::pki_types::{ServerName, UnixTime};
    use x509_parser::extensions::GeneralName;

    use super::*;

    /// A `WebPkiServerVerifier` trusting only `ca_der`, built with an explicit
    /// crypto provider (never `install_default`, which is process-global and
    /// would race across parallel tests).
    fn webpki_verifier_for(ca_der: &CertificateDer<'static>) -> Arc<WebPkiServerVerifier> {
        let mut roots = RootCertStore::empty();
        roots
            .add(ca_der.clone())
            .expect("ca der is a valid trust anchor");
        WebPkiServerVerifier::builder_with_provider(
            Arc::new(roots),
            Arc::new(rustls::crypto::aws_lc_rs::default_provider()),
        )
        .build()
        .expect("build webpki verifier")
    }

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

    #[test]
    fn unconstrained_ca_has_no_name_constraints_extension() {
        let ca = generate_ca().expect("generate ca");
        let pem = ca_cert_pem(&ca).expect("ca pem");
        let der = pem::parse(&pem).expect("parse ca pem");
        let (_remainder, x509) =
            x509_parser::parse_x509_certificate(der.contents()).expect("parse ca der");

        assert!(
            x509
                .name_constraints()
                .expect("name constraints extension parses")
                .is_none(),
            "unconstrained ca must not carry a name constraints extension"
        );
    }

    #[test]
    fn generate_constrained_ca_carries_critical_permitted_subtree() {
        let ca =
            generate_constrained_ca(&["api.anthropic.com"]).expect("generate constrained ca");
        let pem = ca_cert_pem(&ca).expect("ca pem");
        let der = pem::parse(&pem).expect("parse ca pem");
        let (_remainder, x509) =
            x509_parser::parse_x509_certificate(der.contents()).expect("parse ca der");

        let constraints = x509
            .name_constraints()
            .expect("name constraints extension parses")
            .expect("name constraints extension present");
        assert!(
            constraints.critical,
            "name constraints extension must be critical"
        );

        let permitted = constraints
            .value
            .permitted_subtrees
            .as_ref()
            .expect("permitted subtrees present");
        assert_eq!(permitted.len(), 1, "exactly one permitted subtree");
        match permitted[0].base {
            GeneralName::DNSName(name) => assert_eq!(name, "api.anthropic.com"),
            ref other => panic!("expected a dns name subtree, got {other:?}"),
        }
        assert!(
            constraints.value.excluded_subtrees.is_none(),
            "no excluded subtrees were requested"
        );
    }

    #[test]
    fn constrained_ca_verifies_conforming_leaf_and_rejects_violating_leaf() {
        let ca =
            generate_constrained_ca(&["api.anthropic.com"]).expect("generate constrained ca");

        let (conforming_chain, _key) =
            generate_domain_cert(&ca, "api.anthropic.com").expect("conforming domain cert");
        let (violating_chain, _key) =
            generate_domain_cert(&ca, "api.openai.com").expect("violating domain cert");

        let verifier = webpki_verifier_for(&conforming_chain[1]);

        let conforming_name = ServerName::try_from("api.anthropic.com").expect("server name");
        verifier
            .verify_server_cert(
                &conforming_chain[0],
                &[],
                &conforming_name,
                &[],
                UnixTime::now(),
            )
            .expect("conforming leaf verifies under the permitted subtree");

        let violating_name = ServerName::try_from("api.openai.com").expect("server name");
        let error = verifier
            .verify_server_cert(
                &violating_chain[0],
                &[],
                &violating_name,
                &[],
                UnixTime::now(),
            )
            .expect_err("leaf outside the permitted subtree must fail closed");

        let rustls::Error::InvalidCertificate(rustls::CertificateError::Other(
            rustls::OtherError(inner),
        )) = error
        else {
            panic!("expected an InvalidCertificate(Other) error, got: {error:?}");
        };
        assert_eq!(
            inner.downcast_ref::<webpki::Error>(),
            Some(&webpki::Error::NameConstraintViolation),
            "must fail specifically due to the name-constraint violation"
        );
    }

    #[test]
    fn round_trip_constrained_ca_through_disk_still_signs_and_verifies() {
        let ca =
            generate_constrained_ca(&["api.anthropic.com"]).expect("generate constrained ca");
        let dir = tempfile::tempdir().expect("tempdir");
        let cert_path = dir.path().join("ca.pem");
        let key_path = dir.path().join("ca-key.pem");
        std::fs::write(&cert_path, ca_cert_pem(&ca).expect("ca pem")).expect("write ca cert");
        std::fs::write(&key_path, ca_key_pem(&ca)).expect("write ca key");

        let loaded = load_ca(&cert_path, &key_path).expect("load ca");
        let (chain, _key) =
            generate_domain_cert(&loaded, "api.anthropic.com").expect("sign conforming leaf");

        let verifier = webpki_verifier_for(&chain[1]);
        let server_name = ServerName::try_from("api.anthropic.com").expect("server name");
        verifier
            .verify_server_cert(&chain[0], &[], &server_name, &[], UnixTime::now())
            .expect("leaf signed by the loaded ca chains and verifies");
    }
}
