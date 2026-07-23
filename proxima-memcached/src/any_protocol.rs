//! `MemcachedAnyProtocol` — memcached (text protocol) as an [`AnyProtocol`]
//! candidate for the open universal listener
//! (`Listener::builder().accept("memcached")` / `AnyListenProtocol`).
//! Authored directly against `AnyProtocol`, mirroring
//! `proxima_redis::any_protocol::RedisAnyProtocol` — there is no standalone
//! `MemcachedListenProtocol` bind+accept loop preceding this one.
//!
//! Positive-match probe: unlike RESP (every command is sigil-prefixed with
//! `*`), memcached's text protocol has no framing sigil at all — a command
//! is just its lowercase verb token. [`probe`] matches the accumulated
//! prefix against the fixed, closed set of known verbs
//! ([`KNOWN_VERBS`]), requiring the verb be immediately followed by a
//! space (has-argument commands) or `\r` (the zero-argument commands:
//! `version`, `quit`, bare `stats`) — this rules out `getx` false-matching
//! the `get` verb. Real memcached clients only ever send lowercase verbs,
//! so there's no collision with h1 (uppercase HTTP methods) or RESP
//! (`*`-prefixed).
//!
//! `drive` carries its own engine (`handler`, `config`) as a struct
//! field — the same `AnyHandler`-unused asymmetry
//! [`crate::pipe::MemcachedConnectionPipe`] docs. Each accepted connection
//! builds a FRESH [`MemcachedConnectionPipe`] carrying THIS connection's
//! [`ConnAdmission`] clone, erases it, and hands it to
//! [`proxima_listen::serve_pipe::handle_connection`] — the ONE
//! CONNECT-request/upgrade-handler driver pgwire, redis, and memcached now
//! share.

use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

use proxima_core::ProximaError;
use proxima_listen::admission::ConnAdmission;
use proxima_listen::any::{AnyHandler, AnyProtocol, ProbeVerdict};
use proxima_primitives::pipe::handler::into_handle;
use proxima_primitives::stream::{PeerInfo, StreamConnection};

use crate::config::MemcachedServerConfig;
use crate::pipe::MemcachedConnectionPipe;
use crate::pipes::MemcachedPipeHandle;

/// Every verb the sans-IO codec's `parse_command` recognizes. Ordered
/// longest-shares-a-prefix-first is not required — [`probe`] checks every
/// candidate on each call.
const KNOWN_VERBS: &[&[u8]] = &[
    b"get", b"gets", b"set", b"add", b"replace", b"append", b"prepend", b"cas", b"delete",
    b"incr", b"decr", b"touch", b"flush_all", b"stats", b"version", b"quit",
];

/// `len("flush_all")` (the longest known verb) plus one delimiter byte —
/// the most bytes [`probe`] ever needs before it can decide match-or-not
/// for every verb in [`KNOWN_VERBS`].
const MAX_PREFIX_BYTES: usize = 10;

/// memcached (text protocol) wire candidate for the open universal
/// listener.
pub struct MemcachedAnyProtocol {
    label: String,
    handler: MemcachedPipeHandle,
    config: MemcachedServerConfig,
}

impl MemcachedAnyProtocol {
    #[must_use]
    pub fn new(label: impl Into<String>, handler: MemcachedPipeHandle) -> Self {
        Self {
            label: label.into(),
            handler,
            config: MemcachedServerConfig::default(),
        }
    }

    /// Replaces the default [`MemcachedServerConfig`]; a `memcached` object
    /// in the listener spec still wins at drive time.
    #[must_use]
    pub fn with_config(mut self, config: MemcachedServerConfig) -> Self {
        self.config = config;
        self
    }
}

fn resolve_config(
    base: &MemcachedServerConfig,
    spec: &Value,
) -> Result<MemcachedServerConfig, ProximaError> {
    match spec.get("memcached") {
        None => Ok(base.clone()),
        Some(overrides) => serde_json::from_value(overrides.clone())
            .map_err(|error| ProximaError::Config(format!("memcached spec: {error}"))),
    }
}

/// Whether `prefix` matches `verb` followed by a space (has-argument
/// commands) or `\r` (the zero-argument commands).
fn matches_verb(prefix: &[u8], verb: &[u8]) -> bool {
    prefix.len() > verb.len()
        && &prefix[..verb.len()] == verb
        && matches!(prefix[verb.len()], b' ' | b'\r')
}

/// Whether `prefix` could still resolve to `verb`: `prefix` is no longer
/// than `verb` and matches it byte-for-byte so far. A `prefix` already
/// longer than `verb` without having matched a delimiter (checked by
/// [`matches_verb`] before this is ever consulted) has diverged — not
/// plausible.
fn is_plausible_prefix_of(prefix: &[u8], verb: &[u8]) -> bool {
    prefix.len() <= verb.len() && prefix == &verb[..prefix.len()]
}

impl AnyProtocol for MemcachedAnyProtocol {
    fn name(&self) -> &str {
        &self.label
    }

    fn max_prefix_bytes(&self) -> usize {
        MAX_PREFIX_BYTES
    }

    fn probe(&self, prefix: &[u8]) -> ProbeVerdict {
        if KNOWN_VERBS.iter().any(|verb| matches_verb(prefix, verb)) {
            return ProbeVerdict::Match { consumed: 0 };
        }
        let still_plausible = KNOWN_VERBS
            .iter()
            .any(|verb| is_plausible_prefix_of(prefix, verb));
        if still_plausible && prefix.len() < MAX_PREFIX_BYTES {
            return ProbeVerdict::NeedMore {
                at_least: MAX_PREFIX_BYTES,
            };
        }
        ProbeVerdict::No
    }

    fn drive<'a>(
        &'a self,
        stream: Box<dyn StreamConnection>,
        _handler: AnyHandler,
        spec: &'a Value,
        _peer: Option<PeerInfo>,
        admission: &'a ConnAdmission,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProximaError>> + Send + 'a>> {
        Box::pin(async move {
            let config = resolve_config(&self.config, spec)?;
            let connection_pipe = MemcachedConnectionPipe::new(
                self.label.clone(),
                self.handler.clone(),
                std::sync::Arc::new(config),
            )
            .with_admission(admission.clone());
            let pipe = into_handle(connection_pipe);
            proxima_listen::serve_pipe::handle_connection(stream, pipe).await
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_primitives::pipe::request::Response;

    struct EchoPipe;

    impl proxima_primitives::pipe::SendPipe for EchoPipe {
        type In = crate::pipes::MemcachedPipeRequest;
        type Out = crate::pipes::MemcachedPipeReply;
        type Err = ProximaError;

        async fn call(&self, _request: Self::In) -> Result<Self::Out, ProximaError> {
            Ok(Response::typed(200, proxima_protocols::memcached::Reply::Ok))
        }
    }

    fn handler() -> MemcachedPipeHandle {
        crate::pipes::into_memcached_handle(EchoPipe)
    }

    #[test]
    fn probe_matches_a_known_verb_followed_by_a_space() {
        let protocol = MemcachedAnyProtocol::new("memcached", handler());
        assert_eq!(
            protocol.probe(b"get mykey\r\n"),
            ProbeVerdict::Match { consumed: 0 }
        );
        assert_eq!(
            protocol.probe(b"set k 0 0 5\r\n"),
            ProbeVerdict::Match { consumed: 0 }
        );
    }

    #[test]
    fn probe_matches_a_zero_argument_verb_followed_by_cr() {
        let protocol = MemcachedAnyProtocol::new("memcached", handler());
        assert_eq!(
            protocol.probe(b"quit\r\n"),
            ProbeVerdict::Match { consumed: 0 }
        );
        assert_eq!(
            protocol.probe(b"version\r\n"),
            ProbeVerdict::Match { consumed: 0 }
        );
    }

    #[test]
    fn probe_rejects_a_verb_that_only_shares_a_prefix() {
        let protocol = MemcachedAnyProtocol::new("memcached", handler());
        // "getx" is not "get " or "gets" — a real memcached client never
        // sends this, so it must not false-positive-match `get`.
        assert_eq!(protocol.probe(b"getx foo\r\n"), ProbeVerdict::No);
    }

    #[test]
    fn probe_needs_more_bytes_for_a_still_plausible_short_prefix() {
        let protocol = MemcachedAnyProtocol::new("memcached", handler());
        assert_eq!(
            protocol.probe(b"ge"),
            ProbeVerdict::NeedMore {
                at_least: MAX_PREFIX_BYTES
            }
        );
    }

    #[test]
    fn probe_rejects_bytes_that_are_no_known_verb_prefix_at_all() {
        let protocol = MemcachedAnyProtocol::new("memcached", handler());
        assert_eq!(protocol.probe(b"*1\r\n"), ProbeVerdict::No);
        assert_eq!(protocol.probe(b"GET /\r\n"), ProbeVerdict::No);
    }
}
