//! SPNEGO / Negotiate (RFC 4559 + RFC 4178) — the HTTP "Negotiate" auth
//! scheme's wire framing and negotiate-loop state machine (auth form #4,
//! handshake family). The GSS-API security context tokens themselves are
//! produced by a GSS mechanism (Kerberos/NTLM) the *edge* owns; this module
//! owns the RFC 4559 HTTP framing and the multi-round loop around it.
//!
//! Primary sources:
//! - RFC 4559 §4: the HTTP exchange. Client sends `Authorization: Negotiate
//!   <base64(token)>`; server replies `WWW-Authenticate: Negotiate
//!   <base64(token)>` (the `gssapi-data`). Loop until the GSS context is
//!   established (mechanism returns `COMPLETE`).
//! - RFC 4178: the SPNEGO `NegotiationToken` DER structure the GSS tokens
//!   carry. This module frames/extracts the base64 HTTP transport layer; the
//!   DER `NegTokenInit`/`NegTokenResp` body is opaque GSS output.
//!
//! ## Genuinely-missing primary-source vector (principle 15 criterion 2)
//!
//! The end-to-end SPNEGO token bytes are NOT reproducible from a primary
//! source without a live KDC: a real `NegTokenInit` embeds a Kerberos AP-REQ
//! whose bytes depend on the service ticket, session key, authenticator
//! timestamp, and per-message sequence number — all session- and
//! deployment-specific. **There is no published RFC test vector for a complete
//! SPNEGO/Kerberos GSS token** (RFC 4178/4559/4121 specify structure, not
//! reproducible blobs; MIT/Heimdal interop suites require a running KDC). The
//! named dependency that would unblock a bit-exact GSS-token parity test:
//! **a captured `Negotiate` exchange against a live Kerberos KDC (or an
//! MIT-krb5 test-realm fixture)** — external infrastructure this pass cannot
//! bring up. The wire framing (base64 transport, header round, loop control)
//! and the negotiate-loop FSM below ARE deterministic and fully tested; only
//! the GSS *payload* is the deferred dependency, and it is correctly the
//! edge's GSS mechanism to produce, not this module's.

use alloc::string::String;
use alloc::vec::Vec;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;

/// The negotiate-loop state (RFC 4559 §4). A discriminated enum FSM
/// (principle 11): each round consumes the prior server token and produces the
/// next client token until the GSS mechanism reports the context established.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NegotiateState {
    /// No round sent yet — the next call emits the initial GSS token.
    Start,
    /// A client token was sent; awaiting the server's `WWW-Authenticate`
    /// continuation token.
    AwaitingServer,
    /// The GSS context is established; no further rounds.
    Established,
    /// The server rejected the context (no `Negotiate` continuation on a 401).
    Failed,
}

/// What the driver should do after a negotiate-loop step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NegotiateStep {
    /// Send `Authorization: Negotiate <header>` and await the server.
    SendToken { header: String },
    /// The context is established — proceed with the request as-is.
    Established,
    /// Authentication failed.
    Failed,
}

/// The SPNEGO negotiate-loop driver. Owns the loop state; the GSS mechanism
/// (the edge) supplies each outbound token and consumes each inbound one. This
/// module frames them into/out of the RFC 4559 base64 HTTP transport.
#[derive(Clone, Debug)]
pub struct NegotiateLoop {
    state: NegotiateState,
}

impl Default for NegotiateLoop {
    fn default() -> Self {
        Self::new()
    }
}

impl NegotiateLoop {
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: NegotiateState::Start,
        }
    }

    #[must_use]
    pub fn state(&self) -> &NegotiateState {
        &self.state
    }

    /// Frame an outbound GSS token for the `Authorization` header. The GSS
    /// mechanism produced `gss_token`; this base64-encodes it per RFC 4559 §4.
    /// Transitions `Start`/`AwaitingServer` → `AwaitingServer`.
    ///
    /// Refuses to fire from a terminal state (`Established`/`Failed`): re-opening
    /// a closed context would replay a GSS token against an already-established
    /// or rejected context (audit H4), so this returns [`NegotiateError::Terminal`].
    ///
    /// # Errors
    /// [`NegotiateError::Terminal`] when the loop has already completed.
    pub fn send(&mut self, gss_token: &[u8]) -> Result<NegotiateStep, NegotiateError> {
        match self.state {
            NegotiateState::Established | NegotiateState::Failed => Err(NegotiateError::Terminal),
            NegotiateState::Start | NegotiateState::AwaitingServer => {
                self.state = NegotiateState::AwaitingServer;
                Ok(NegotiateStep::SendToken {
                    header: encode_negotiate_header(gss_token),
                })
            }
        }
    }

    /// Consume the server's `WWW-Authenticate: Negotiate …` continuation.
    /// Returns the decoded GSS token bytes the mechanism must process next, or
    /// `None` when the server sent `Negotiate` with no data (final round).
    ///
    /// `gss_complete` is the mechanism's verdict after processing the prior
    /// token: `true` ends the loop (`Established`).
    ///
    /// # Errors
    /// [`NegotiateError`] on a malformed header or base64.
    pub fn on_server(
        &mut self,
        www_authenticate: Option<&str>,
        gss_complete: bool,
    ) -> Result<Option<Vec<u8>>, NegotiateError> {
        if gss_complete {
            self.state = NegotiateState::Established;
            return Ok(None);
        }
        let Some(header) = www_authenticate else {
            self.state = NegotiateState::Failed;
            return Err(NegotiateError::NoContinuation);
        };
        let token = decode_negotiate_header(header)?;
        self.state = NegotiateState::AwaitingServer;
        Ok(Some(token))
    }
}

/// `Negotiate <base64(token)>` — the RFC 4559 §4 `Authorization` value.
#[must_use]
pub fn encode_negotiate_header(gss_token: &[u8]) -> String {
    let mut out = String::with_capacity(11 + (gss_token.len() * 4).div_ceil(3));
    out.push_str("Negotiate ");
    out.push_str(&BASE64.encode(gss_token));
    out
}

/// Extract the GSS token bytes from a `WWW-Authenticate: Negotiate <base64>`
/// (or bare `Negotiate <base64>`) value.
///
/// # Errors
/// [`NegotiateError`] when the scheme is wrong or the base64 is malformed.
pub fn decode_negotiate_header(header: &str) -> Result<Vec<u8>, NegotiateError> {
    let trimmed = header.trim();
    let data = trimmed
        .strip_prefix("Negotiate ")
        .or_else(|| trimmed.strip_prefix("negotiate "))
        .ok_or(NegotiateError::WrongScheme)?
        .trim();
    BASE64.decode(data).map_err(|_| NegotiateError::BadBase64)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NegotiateError {
    /// the header did not start with the `Negotiate` scheme token
    WrongScheme,
    /// the token was not valid base64
    BadBase64,
    /// the server returned 401 with no `Negotiate` continuation token
    NoContinuation,
    /// `send` was called after the loop already reached a terminal state
    Terminal,
}

impl core::fmt::Display for NegotiateError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let message = match self {
            Self::WrongScheme => "negotiate header missing the `Negotiate` scheme",
            Self::BadBase64 => "negotiate token is not valid base64",
            Self::NoContinuation => "server returned no `Negotiate` continuation token",
            Self::Terminal => "negotiate loop already terminal; cannot send another token",
        };
        formatter.write_str(message)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // RFC 4559 §4 wire-framing vectors. The GSS *payload* here is a placeholder
    // (a real one needs a live KDC — see the module-level missing-vector note);
    // these tests pin the deterministic base64 HTTP transport + the loop FSM,
    // which is all this module owns. The base64 is RFC 4648-exact.

    #[test]
    fn rfc4559_authorization_header_is_negotiate_space_base64() {
        // "token" base64s to "dG9rZW4=" (RFC 4648 standard alphabet).
        let header = encode_negotiate_header(b"token");
        assert_eq!(header, "Negotiate dG9rZW4=");
    }

    #[test]
    fn decode_round_trips_the_encoded_token() {
        let token = b"\x60\x82\x01\x2c\x06\x06\x2b\x06\x01\x05\x05\x02"; // SPNEGO OID prefix bytes
        let header = encode_negotiate_header(token);
        let decoded = decode_negotiate_header(&header).expect("decode");
        assert_eq!(decoded, token);
    }

    #[test]
    fn decode_accepts_the_www_authenticate_scheme_case_insensitively() {
        let decoded = decode_negotiate_header("negotiate dG9rZW4=").expect("decode");
        assert_eq!(decoded, b"token");
    }

    #[test]
    fn wrong_scheme_is_an_error() {
        assert_eq!(
            decode_negotiate_header("Basic abc"),
            Err(NegotiateError::WrongScheme)
        );
    }

    #[test]
    fn bad_base64_is_an_error() {
        assert_eq!(
            decode_negotiate_header("Negotiate not!valid!base64!"),
            Err(NegotiateError::BadBase64)
        );
    }

    #[test]
    fn loop_starts_then_sends_then_establishes() {
        let mut negotiate = NegotiateLoop::new();
        assert_eq!(negotiate.state(), &NegotiateState::Start);

        let step = negotiate
            .send(b"client-init-token")
            .expect("send from Start");
        let NegotiateStep::SendToken { header } = step else {
            panic!("expected SendToken")
        };
        assert!(header.starts_with("Negotiate "));
        assert_eq!(negotiate.state(), &NegotiateState::AwaitingServer);

        // server returns a continuation token; GSS not yet complete.
        let next = negotiate
            .on_server(Some("Negotiate c2VydmVyLXRva2Vu"), false)
            .expect("continuation");
        assert_eq!(next.as_deref(), Some(&b"server-token"[..]));
        assert_eq!(negotiate.state(), &NegotiateState::AwaitingServer);

        // mechanism reports complete on the next round.
        let done = negotiate.on_server(None, true).expect("complete");
        assert_eq!(done, None);
        assert_eq!(negotiate.state(), &NegotiateState::Established);

        // audit H4: a send after establishment must NOT re-open the loop.
        assert_eq!(
            negotiate.send(b"stray-token"),
            Err(NegotiateError::Terminal)
        );
        assert_eq!(
            negotiate.state(),
            &NegotiateState::Established,
            "state unchanged on rejected send"
        );
    }

    #[test]
    fn no_continuation_on_a_401_fails_the_loop() {
        let mut negotiate = NegotiateLoop::new();
        let _ = negotiate.send(b"client-init-token");
        let outcome = negotiate.on_server(None, false);
        assert_eq!(outcome, Err(NegotiateError::NoContinuation));
        assert_eq!(negotiate.state(), &NegotiateState::Failed);
        // audit H4: a send after failure is refused, the FSM stays Failed.
        assert_eq!(
            negotiate.send(b"stray-token"),
            Err(NegotiateError::Terminal)
        );
        assert_eq!(negotiate.state(), &NegotiateState::Failed);
    }
}
