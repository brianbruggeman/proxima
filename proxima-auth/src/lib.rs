//! Sans-IO authentication FSM cores for the proxima auth axis.
//!
//! The auth axis is the third leg of `protocol × transport × auth`. Every form
//! reduces to a state machine in the middle with `Pipe` at the edges (the
//! per-request *attach* edge and, for fetched credentials, the *exchange*
//! edge). This crate owns the FSMs; the consumers (an HTTP wrapper pipe, the
//! pgwire SASL driver, …) own the edges.
//!
//! Forms, by FSM shape:
//! - **static** (token / API key) — degenerate FSM, always `Use`; just the
//!   attach edge.
//! - **lifecycle** ([`token::TokenLifecycle`]) — exchange creds for a
//!   short-lived token, refresh before expiry, single-flight. The novel core.
//! - **handshake** ([`Handshake`]) — multi-round challenge/response (SCRAM,
//!   Digest, Kerberos); `proxima_pgwire::ScramClient` is the reference instance.
//! - **signing** (`sigv4::SigV4Signer`, behind the `signing` feature) —
//!   degenerate FSM, but the attach edge *computes* from the request + key
//!   (AWS `SigV4`, HMAC, RFC 9421).
//!
//! Each form past the first two is a default-off feature (`signing`, `digest`,
//! `negotiate`), so a plain `-p proxima-auth` command compiles one module of
//! four. `scripts/proxima-auth-gate.sh` is what covers the rest.
//!
//! Sans-IO discipline (principle 11): no clock reads — the edge stamps
//! [`token::AuthTime`] (mirrors `proxima_protocols::quic::Instant`); no sockets.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod token;

#[cfg(feature = "signing")]
pub mod sigv4;

#[cfg(feature = "digest")]
pub mod digest;

#[cfg(feature = "negotiate")]
pub mod spnego;

pub use token::{AuthTime, Credential, TokenLifecycle, TokenStep};

#[cfg(feature = "signing")]
pub use sigv4::{SecretKey, SigV4Signer, SignedHeader};

#[cfg(feature = "digest")]
pub use digest::{DigestAlgorithm, DigestChallenge, DigestClient, DigestError};

#[cfg(feature = "negotiate")]
pub use spnego::{NegotiateError, NegotiateLoop, NegotiateState, NegotiateStep};

use alloc::vec::Vec;
use core::fmt::Debug;

/// A multi-round challenge/response handshake (auth form #4): SCRAM, HTTP
/// Digest, Kerberos/GSSAPI, NTLM. Bytes in (server challenge), bytes out
/// (client message) — the edge carries them over the wire (pgwire SASL
/// messages, an HTTP `WWW-Authenticate`/`Authorization` round, …).
/// `proxima_pgwire::ScramClient` (`client_first` / `client_final` /
/// `verify_server_final`) is the reference shape this generalizes.
pub trait Handshake {
    /// `Debug` so a driver generic over `H: Handshake` can log the failure it
    /// gets back; without the bound the error is opaque at exactly the edge
    /// that has to report it.
    type Error: Debug;

    /// The first client message to send (no server input yet).
    fn first(&mut self) -> Vec<u8>;

    /// Consume a server challenge, produce the next client message, or `None`
    /// when the handshake is complete and the last server message only needed
    /// verification.
    ///
    /// # Errors
    /// Implementation-defined on a malformed or rejected challenge.
    fn step(&mut self, server: &[u8]) -> Result<Option<Vec<u8>>, Self::Error>;
}
