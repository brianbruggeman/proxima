# proxima-intercept

Generic MITM intercept pipeline: a TLS-terminating CONNECT proxy that forwards,
captures, and replays traffic for any host, with no vendor knowledge baked in.
Heavy TLS/cert-gen deps live here only — kept out of `proxima-recording` so a
plain capture/replay consumer doesn't pull in `rcgen`/`rustls`.

## CA generation (`ca.rs`)

[`generate_ca`] mints an unconstrained root: it can sign a forged leaf for
any host. [`generate_constrained_ca`] mints the same root plus a critical
RFC 5280 §4.2.1.10 `NameConstraints` extension restricting which DNS names
it may sign for — rcgen always writes this extension critical, so a
verifier that doesn't understand it rejects the whole chain rather than
silently ignoring it. Use the constrained form whenever a consumer knows the
fixed set of hosts it will ever MITM (one CA per vendor, say); use the
unconstrained form for a general-purpose proxy that forges leaves for
arbitrary SNI.

[`ForgingResolver`] wraps either kind of CA behind rustls'
`ResolvesServerCert`: it mints (and caches) a per-SNI leaf on first sight,
so one `ServerConfig` can terminate TLS for any host without a
per-host acceptor. [`forging_server_config`] builds that config directly;
[`build_tls_acceptor`] is the equivalent for a single, pre-known host.

`load_ca`/`ca_cert_pem`/`ca_key_pem` round-trip a `CaKeyPair` through PEM on
disk — mint once, persist, reload on the next run rather than minting a new
root (and invalidating every client's trust store) on every restart.

See `examples/transparent-capture.rs` and `examples/quic-capture.rs` for a
CA wired into a live TCP/QUIC listener.

## Feature gates

- `intercept-capture` — record traffic to disk (`capture.rs`, `blake3`-addressed)
- `intercept-replay` — replay captured traffic (implies `intercept-capture`)
- `intercept-config` — `conflaguration`-driven runtime config
- `quic-intercept` — QUIC/h3 termination + decode, kept out of the default TCP build
- `delta-tee` — source-level cfg for the delta-tee capture path

Part of the [proxima](..) workspace.
