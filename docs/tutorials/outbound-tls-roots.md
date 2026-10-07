# tls roots for clients and listeners

This guide is for a Rust beginner who knows structs, methods, and `Result`, but has not used Proxima. It follows one setting from a universal client to the code that opens a connection, then contrasts that with the different trust decision made by a listener using mutual TLS (mTLS). It ends with the Hyper ALPN boundary.

## Learning sequence

1. A TLS client config is data, while a destination URL supplies the server name.
2. The universal client carries that data to the selected HTTP or gRPC factory, where the root policy becomes a verifier.
3. A listener's mTLS roots answer the inverse question: which client certificates may connect?
4. Hyper owns ALPN negotiation for its HTTPS connector, so its rustls config must not arrive with a caller-selected ALPN list.
5. On Windows, the Hyper alias is registered only when its feature gates are enabled; this helps distinguish factory availability from TLS root policy.

## 1. Client TLS settings travel with the client spec

Certificate verification needs two inputs: a trust policy and the name of the peer being checked. `TlsClientConfig` holds the policy: `root_source`, PEM `ca_bundle_paths`, and `alpn_protocols`. Its `server_name` is the SNI name and the name checked against the server certificate. The universal client can leave that field empty because its HTTP or gRPC factory derives the name from the request URL. The fields are defined in [`proxima-tls/src/connector.rs:270-292`](../../proxima-tls/src/connector.rs#L270).

`RootSource` has three choices: `Native` uses operating-system trust anchors, `Mozilla` uses the bundled WebPKI set, and `CustomOnly` trusts only configured PEM paths ([`proxima-tls/src/imp.rs:32-39`](../../proxima-tls/src/imp.rs#L32)). On Windows, the build-time default is `Native`; other target operating systems currently default to `Mozilla` ([`proxima-tls/src/build_defaults.rs:1-3`](../../proxima-tls/src/build_defaults.rs#L1), [`proxima-tls/build.rs:8-20`](../../proxima-tls/build.rs#L8)). Custom PEM bundles are additive for `Native` and `Mozilla`, and exclusive for `CustomOnly` ([`proxima-tls/src/connector.rs:56-79`](../../proxima-tls/src/connector.rs#L56)).

The layered builder lets application configuration override target defaults. It can read a TOML, JSON, or YAML file, then environment variables, and then apply fluent overrides ([`proxima-tls/src/connector.rs:401-459`](../../proxima-tls/src/connector.rs#L401)). For example, this is the actual builder flow exercised by the Client test:

```rust
use proxima::{Client, ClientProtocolExt};

let config = proxima_tls::TlsClientConfig::layered()
    .with_root_source(proxima_tls::RootSource::Native)
    .with_ca_bundle_paths(vec!["corp.pem".into()])
    .build_for_client()
    .expect("TLS client config");

let client = Client::builder()
    .https("https://api.example.com")
    .tls_client(config)
    .expect("attach TLS client config")
    .build()
    .expect("client build");
```

`build_for_client` validates root and ALPN settings without requiring an SNI name yet. `ClientBuilder::tls_client` serializes this config under the client's `tls_client` spec key ([`src/client/handle.rs:550-564`](../../src/client/handle.rs#L550)); that lets the URL-aware factory fill in the destination name later. The same settings can be provided in a value-based client configuration under `tls_client` ([`src/client/handle.rs:1135-1166`](../../src/client/handle.rs#L1135)).

## 2. The HTTP and gRPC factories turn policy into verification

The `http` shorthand is canonicalized into a factory spec, and `tls_client` is one of the fields forwarded along with the URL ([`src/load.rs:618-638`](../../src/load.rs#L618)). The Prime HTTP factory parses the URL, decides whether the connection is HTTPS, and wraps the direct or proxy-tunneled stream in TLS. Before doing that, it copies the configured settings and fills an omitted `server_name` from the URL host ([`proxima-http/src/http1/prime_upstream.rs:135-163`](../../proxima-http/src/http1/prime_upstream.rs#L135), [`proxima-http/src/http1/prime_upstream.rs:169-202`](../../proxima-http/src/http1/prime_upstream.rs#L169)).

`TlsStreamUpstream::from_config` validates the config and creates the rustls `ClientConfig` before a connection opens ([`proxima-tls/src/connector.rs:233-251`](../../proxima-tls/src/connector.rs#L233)). The custom-root loader reads each PEM bundle and returns a configuration error for unreadable, invalid, empty, or unusable certificate files ([`proxima-tls/src/connector.rs:106-138`](../../proxima-tls/src/connector.rs#L106)). This is why a misspelled custom-root path fails while Proxima is constructing the upstream instead of silently falling back to some other root set.

The gRPC-over-TLS factory follows the same root policy but owns a different protocol requirement: it derives the peer name from the URL and fixes ALPN to `h2` ([`src/upstreams/grpc_h2.rs:127-152`](../../src/upstreams/grpc_h2.rs#L127), [`src/upstreams/grpc_h2.rs:164-179`](../../src/upstreams/grpc_h2.rs#L164)). The setting can choose which certificates are trusted; it cannot make gRPC negotiate HTTP/1.1.

## 3. A listener trusts client certificates for a different reason

An outbound client asks, “Which server certificates do I trust?” A listener with mTLS asks, “Which client certificates may connect to me?” The second decision is not made from the machine's web-server roots. Proxima's listener TLS config has a separate `client_auth` policy ([`proxima-tls/src/imp.rs:57-72`](../../proxima-tls/src/imp.rs#L57)).

For listeners, `ClientAuth::optional_files` requests a client certificate but permits a handshake without one; `ClientAuth::required_files` rejects a handshake without a certificate. Both take PEM CA bundle paths ([`proxima-tls/src/imp.rs:159-182`](../../proxima-tls/src/imp.rs#L159)). When Proxima builds the server verifier, it reads those paths and applies the optional or required behavior ([`proxima-tls/src/imp.rs:251-270`](../../proxima-tls/src/imp.rs#L251)). These are explicit client-issuer roots, not the `RootSource::Native` choice used to verify remote servers.

The universal Listener attaches its server-side `TlsConfig` with `.tls(config)`. This is a listener-specific inherent method because a listener also needs its server certificate and key ([`src/listener/handle.rs:397-417`](../../src/listener/handle.rs#L397)). The Listener composes TLS around the selected listening protocol before serving ([`src/listener/handle.rs:511-517`](../../src/listener/handle.rs#L511)).

## 4. Hyper owns ALPN for its HTTPS connector

ALPN is the TLS handshake extension that lets peers agree which application protocol will use the connection, such as HTTP/1.1 or HTTP/2. The general `TlsClientConfig` contains an ALPN list because the Prime and direct TLS paths can pass that list to rustls ([`proxima-tls/src/connector.rs:287-292`](../../proxima-tls/src/connector.rs#L287), [`proxima-tls/src/connector.rs:313-325`](../../proxima-tls/src/connector.rs#L313)).

The Hyper connector has a different owner for this choice. `SharedHttpClient::with_client_tls_config` builds the rustls config, clears its ALPN list, then asks `hyper-rustls` to enable HTTP/1 and HTTP/2. The connector selects ALPN from those enabled protocols and rejects a pre-populated list, because the two lists could disagree ([`proxima-http/src/http1/shared_http.rs:94-110`](../../proxima-http/src/http1/shared_http.rs#L94)). Root trust still comes from the `TlsClientConfig`; only the protocol negotiation list is handed to the component that constructs the HTTP connector.

## 5. Windows factory availability is a separate feature gate

When both the Hyper HTTP backend and the Prime runtime are enabled, Proxima registers the Hyper factory under the alias `http-tokio`. Both the `AliasFactory` type and its registration are gated for `unix` and `windows`, as well as the corresponding Cargo features ([`src/load.rs:71-91`](../../src/load.rs#L71), [`src/load.rs:294-305`](../../src/load.rs#L294)). The `windows` cfg makes that alias available to a Windows build; it does not choose roots. Root selection remains the `RootSource` policy described above.
