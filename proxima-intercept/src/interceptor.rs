//! The domain policy a caller injects into the generic MITM. [`InterceptPipe`]
//! owns the TLS/forward/capture mechanics; an `Interceptor` owns what is
//! domain-specific — rewrite the outbound request, observe the completed
//! exchange, route elsewhere, or gate the host. This is the seam that lets a
//! memory proxy (inject recalled context), a record/replay harness, or a policy
//! gateway reuse one MITM instead of hand-rolling its own.
//!
//! [`InterceptPipe`]: crate::InterceptPipe

use std::future::Future;
use std::pin::Pin;

/// What to do with an intercepted http/1.1 request after the interceptor sees it.
pub enum Interception {
    /// Forward this (possibly rewritten) full request wire to the upstream. Pass
    /// the original bytes through unchanged to forward verbatim.
    Forward(Vec<u8>),
    /// Short-circuit: write this raw response to the client and never touch the
    /// upstream — the key-free swap / replay path.
    Respond(Vec<u8>),
}

/// Whether the proxy should terminate + intercept a host at all.
pub enum HostPolicy {
    /// Terminate TLS and run the interceptor (the default).
    Intercept,
    /// Tunnel raw bytes with no MITM (e.g. a telemetry host, or a connection the
    /// operator wants observed-but-not-rewritten).
    Passthrough,
    /// Refuse the connection — drop the tunnel (an operator-disabled connection).
    Refuse,
}

/// Injected per-request policy + transforms for [`InterceptPipe`]. All methods
/// have inert defaults, so an interceptor implements only the hooks it needs.
pub trait Interceptor: Send + Sync + 'static {
    /// Inspect the full outbound request to `host` and decide its fate: forward
    /// it (optionally rewritten), or short-circuit with a direct response.
    /// Default: forward the request unchanged.
    fn intercept<'a>(
        &'a self,
        host: &'a str,
        request: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = Interception> + Send + 'a>> {
        let _ = host;
        let forwarded = request.to_vec();
        Box::pin(async move { Interception::Forward(forwarded) })
    }

    /// Observe the completed exchange (the ORIGINAL client request + the upstream
    /// response) — for persisting, enriching, accounting. Default: no-op.
    fn observe<'a>(
        &'a self,
        host: &'a str,
        request: &'a [u8],
        response: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        let _ = (host, request, response);
        Box::pin(async {})
    }

    /// Override the upstream address (`"host:port"`) the proxy forwards `host` to,
    /// instead of the resolved CONNECT target. Default: no override.
    fn upstream_override(&self, host: &str) -> Option<String> {
        let _ = host;
        None
    }

    /// Whether to intercept `host`. Default: [`HostPolicy::Intercept`].
    fn host_policy(&self, host: &str) -> HostPolicy {
        let _ = host;
        HostPolicy::Intercept
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    /// a bare interceptor that overrides nothing — exercises every default.
    struct Bare;
    impl Interceptor for Bare {}

    /// a host gate that refuses one host and rewrites the request body.
    struct Gate;
    impl Interceptor for Gate {
        fn host_policy(&self, host: &str) -> HostPolicy {
            if host == "blocked.example" {
                HostPolicy::Refuse
            } else {
                HostPolicy::Intercept
            }
        }

        fn intercept<'a>(
            &'a self,
            _host: &'a str,
            request: &'a [u8],
        ) -> Pin<Box<dyn Future<Output = Interception> + Send + 'a>> {
            let mut rewritten = request.to_vec();
            rewritten.extend_from_slice(b"-injected");
            Box::pin(async move { Interception::Forward(rewritten) })
        }
    }

    fn block_on<F: Future>(future: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(future)
    }

    #[test]
    fn defaults_intercept_every_host_and_forward_verbatim() {
        let bare = Bare;
        assert!(matches!(
            bare.host_policy("anything"),
            HostPolicy::Intercept
        ));
        assert!(bare.upstream_override("anything").is_none());
        block_on(bare.observe("h", b"req", b"resp"));
        match block_on(bare.intercept("h", b"hello")) {
            Interception::Forward(bytes) => {
                assert_eq!(bytes, b"hello", "default forwards the request unchanged")
            }
            Interception::Respond(_) => panic!("default must forward, not respond"),
        }
    }

    #[test]
    fn a_gate_refuses_and_rewrites() {
        let gate = Gate;
        assert!(
            matches!(gate.host_policy("blocked.example"), HostPolicy::Refuse),
            "the gate refuses its blocked host"
        );
        assert!(
            matches!(gate.host_policy("ok.example"), HostPolicy::Intercept),
            "other hosts are intercepted"
        );
        match block_on(gate.intercept("ok.example", b"req")) {
            Interception::Forward(bytes) => assert_eq!(
                bytes, b"req-injected",
                "the gate rewrites the outbound request"
            ),
            Interception::Respond(_) => panic!("this gate forwards"),
        }
    }
}
