//! Transparent TLS capture — intercept a DIRECT (non-CONNECT) TLS connection that
//! bypasses an HTTP proxy. macOS pf redirects the target host's :443 here (after
//! /etc/hosts poisons it to 127.0.0.1); we forge a per-SNI cert (ForgingResolver),
//! terminate the client TLS, open our own TLS to the REAL upstream IP, and relay the
//! decrypted bytes both ways, teeing them to dumps. This is what catches a client
//! that ignores HTTPS_PROXY and connects straight to api2.example.com.
//!
//! Upstream resolution: because /etc/hosts is poisoned, the listener cannot use the
//! system resolver to find the real upstream (it would resolve back to 127.0.0.1).
//! Supply the real IPs out-of-band:
//!   PROXIMA_UPSTREAM_MAP="api2.example.com=54.87.52.199,api3.example.com=104.18.18.125"
//! and/or a single fallback for any unmapped SNI:
//!   PROXIMA_UPSTREAM_ADDR=54.87.52.199:443
//! Resolve real IPs with a query that bypasses /etc/hosts: `dig +short @1.1.1.1 <host>`.
//!
//! Run:  PROXIMA_TRANSPARENT_BIND=127.0.0.1:443 \
//!         PROXIMA_UPSTREAM_MAP="api2.example.com=<ip>,api3.example.com=<ip>" \
//!         cargo run -p proxima-intercept --example transparent-capture \
//!         --features intercept-capture

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use proxima_intercept::ca::{ca_cert_pem, ca_key_pem, generate_ca, load_ca};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const BIND_DEFAULT: &str = "127.0.0.1:8443";
static SEQ: AtomicU64 = AtomicU64::new(0);

#[proxima::main(runtime = "tokio", flavor = "multi_thread")]
async fn main() -> Result<(), proxima_core::ProximaError> {
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    let bind =
        std::env::var("PROXIMA_TRANSPARENT_BIND").unwrap_or_else(|_| BIND_DEFAULT.to_string());
    let dump_dir =
        std::env::var("PROXIMA_INTERCEPT_H2_DUMP").unwrap_or_else(|_| "/tmp".to_string());
    let upstreams = Arc::new(parse_upstreams()?);

    let ca = Arc::new(load_or_make_ca()?);
    let server_config = proxima_intercept::ca::forging_server_config(
        ca,
        vec![b"h2".to_vec(), b"http/1.1".to_vec()],
    );
    let acceptor = tokio_rustls::TlsAcceptor::from(server_config);

    let listener = TcpListener::bind(&bind)
        .await
        .map_err(|err| proxima_core::ProximaError::Upstream(format!("bind {bind}: {err}")))?;
    eprintln!(
        "transparent capture on {bind}; upstreams {:?}; dumps in {dump_dir}",
        upstreams.map
    );

    loop {
        let (client, _peer) = match listener.accept().await {
            Ok(pair) => pair,
            Err(err) => {
                eprintln!("[accept-err] {err}");
                continue;
            }
        };
        let acceptor = acceptor.clone();
        let dump_dir = dump_dir.clone();
        let upstreams = upstreams.clone();
        tokio::spawn(async move {
            if let Err(err) = handle(client, acceptor, &upstreams, &dump_dir).await {
                eprintln!("[conn-err] {err}");
            }
        });
    }
}

/// Per-SNI real-upstream addresses, with an optional catch-all for unmapped names.
struct Upstreams {
    map: HashMap<String, SocketAddr>,
    fallback: Option<SocketAddr>,
}

impl Upstreams {
    fn resolve(&self, sni: &str) -> Option<SocketAddr> {
        self.map.get(sni).copied().or(self.fallback)
    }
}

fn parse_upstreams() -> Result<Upstreams, proxima_core::ProximaError> {
    let mut map = HashMap::new();
    if let Ok(raw) = std::env::var("PROXIMA_UPSTREAM_MAP") {
        for entry in raw.split(',').filter(|pair| !pair.trim().is_empty()) {
            let (host, ip) = entry.split_once('=').ok_or_else(|| {
                proxima_core::ProximaError::Config(format!(
                    "upstream map entry {entry:?} not host=ip"
                ))
            })?;
            let addr: SocketAddr = format!("{}:443", ip.trim()).parse().map_err(|err| {
                proxima_core::ProximaError::Config(format!("upstream map ip {ip:?}: {err}"))
            })?;
            map.insert(host.trim().to_owned(), addr);
        }
    }
    let fallback =
        match std::env::var("PROXIMA_UPSTREAM_ADDR") {
            Ok(raw) => Some(raw.parse().map_err(|err| {
                proxima_core::ProximaError::Config(format!("upstream addr: {err}"))
            })?),
            Err(_) => None,
        };
    if map.is_empty() && fallback.is_none() {
        return Err(proxima_core::ProximaError::Config(
            "set PROXIMA_UPSTREAM_MAP=host=ip,... and/or PROXIMA_UPSTREAM_ADDR=ip:443".into(),
        ));
    }
    Ok(Upstreams { map, fallback })
}

async fn handle(
    client: TcpStream,
    acceptor: tokio_rustls::TlsAcceptor,
    upstreams: &Upstreams,
    dump_dir: &str,
) -> Result<(), proxima_core::ProximaError> {
    let client_tls = acceptor
        .accept(client)
        .await
        .map_err(|err| proxima_core::ProximaError::Upstream(format!("client tls accept: {err}")))?;
    let sni = client_tls
        .get_ref()
        .1
        .server_name()
        .unwrap_or("unknown")
        .to_owned();
    let upstream_addr = upstreams.resolve(&sni).ok_or_else(|| {
        proxima_core::ProximaError::Config(format!(
            "no upstream for SNI {sni} (add to PROXIMA_UPSTREAM_MAP)"
        ))
    })?;
    // match upstream protocol to what the client negotiated, or the raw byte-relay
    // would splice e.g. an http/1.1 client onto an h2 server.
    let alpn = client_tls.get_ref().1.alpn_protocol().map(<[u8]>::to_vec);
    eprintln!(
        "[transparent] SNI={sni} alpn={:?} -> upstream {upstream_addr}",
        alpn.as_deref()
    );

    let upstream = connect_upstream(&sni, upstream_addr, alpn).await?;

    let (cr, cw) = tokio::io::split(client_tls);
    let (ur, uw) = tokio::io::split(upstream);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let safe: String = sni
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '.' {
                character
            } else {
                '_'
            }
        })
        .collect();
    let c2s = PathBuf::from(dump_dir).join(format!("proxima-h2-{safe}-{seq:03}-c2s.bin"));
    let s2c = PathBuf::from(dump_dir).join(format!("proxima-h2-{safe}-{seq:03}-s2c.bin"));
    eprintln!("[transparent-dump] {} / {}", c2s.display(), s2c.display());

    let (a, b) = tokio::join!(copy_tee(cr, uw, c2s), copy_tee(ur, cw, s2c));
    a?;
    b?;
    Ok(())
}

async fn connect_upstream(
    sni: &str,
    addr: SocketAddr,
    alpn: Option<Vec<u8>>,
) -> Result<tokio_rustls::client::TlsStream<TcpStream>, proxima_core::ProximaError> {
    let tcp = TcpStream::connect(addr).await.map_err(|err| {
        proxima_core::ProximaError::Upstream(format!("upstream connect {addr}: {err}"))
    })?;
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let mut config = rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    // mirror the client EXACTLY: if it offered no ALPN (http/1.1, e.g. a client
    // with http2 disabled), offer none upstream too so both default to http/1.1 — else the
    // raw byte-relay would splice http/1.1 onto a server that picked h2.
    config.alpn_protocols = match alpn {
        Some(proto) => vec![proto],
        None => vec![],
    };
    let connector = tokio_rustls::TlsConnector::from(Arc::new(config));
    let server_name = rustls::pki_types::ServerName::try_from(sni.to_owned())
        .map_err(|err| proxima_core::ProximaError::Config(format!("sni {sni}: {err}")))?;
    connector
        .connect(server_name, tcp)
        .await
        .map_err(|err| proxima_core::ProximaError::Upstream(format!("upstream tls {sni}: {err}")))
}

async fn copy_tee<Reader, Writer>(
    mut from: Reader,
    mut to: Writer,
    dump_path: PathBuf,
) -> Result<(), proxima_core::ProximaError>
where
    Reader: tokio::io::AsyncRead + Unpin,
    Writer: tokio::io::AsyncWrite + Unpin,
{
    let mut dump = tokio::fs::File::create(&dump_path)
        .await
        .map_err(|err| proxima_core::ProximaError::Upstream(format!("dump create: {err}")))?;
    let mut buf = vec![0u8; 32 * 1024];
    loop {
        let n = from
            .read(&mut buf)
            .await
            .map_err(|err| proxima_core::ProximaError::Upstream(format!("relay read: {err}")))?;
        if n == 0 {
            break;
        }
        to.write_all(&buf[..n])
            .await
            .map_err(|err| proxima_core::ProximaError::Upstream(format!("relay write: {err}")))?;
        to.flush().await.ok();
        dump.write_all(&buf[..n])
            .await
            .map_err(|err| proxima_core::ProximaError::Upstream(format!("dump write: {err}")))?;
    }
    dump.flush().await.ok();
    Ok(())
}

fn load_or_make_ca() -> Result<proxima_intercept::ca::CaKeyPair, proxima_core::ProximaError> {
    let dir =
        PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".into())).join(".proxima");
    let cert = dir.join("ca.pem");
    let key = dir.join("ca-key.pem");
    if cert.exists() && key.exists() {
        load_ca(&cert, &key)
    } else {
        std::fs::create_dir_all(&dir).ok();
        let ca = generate_ca()?;
        std::fs::write(&cert, ca_cert_pem(&ca)?).map_err(proxima_core::ProximaError::Io)?;
        std::fs::write(&key, ca_key_pem(&ca)).map_err(proxima_core::ProximaError::Io)?;
        Ok(ca)
    }
}
