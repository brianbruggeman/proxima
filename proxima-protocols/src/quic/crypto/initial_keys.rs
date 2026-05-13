//! QUIC v1 initial-secret derivation per [RFC 9001 §5.2].
//!
//! Initial-packet protection uses keys derived deterministically from
//! the client's Destination Connection ID via HKDF-SHA256. No RNG is
//! involved at this stage; the secret material is recoverable by any
//! middlebox that sees the Initial packet, which is why Initial packets
//! also carry the Retry Integrity Tag and CRYPTO frames for the TLS
//! handshake.
//!
//! # Algorithm
//!
//! ```text
//!   initial_salt = 0x38762cf7f55934b34d179ae6a4c80cadccbb7f0a  (RFC 9001 §5.2)
//!   initial_secret = HKDF-Extract(initial_salt, client_dcid)
//!
//!   client_initial_secret = HKDF-Expand-Label(initial_secret, "client in", "", 32)
//!   server_initial_secret = HKDF-Expand-Label(initial_secret, "server in", "", 32)
//!
//!   For each side:
//!     key = HKDF-Expand-Label(secret, "quic key", "", 16)   // AEAD packet protection
//!     iv  = HKDF-Expand-Label(secret, "quic iv",  "", 12)   // AEAD nonce base
//!     hp  = HKDF-Expand-Label(secret, "quic hp",  "", 16)   // header protection
//! ```
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`). All buffers are stack-allocated;
//! the [`InitialKeys`] output struct holds three fixed-size arrays.
//!
//! [RFC 9001 §5.2]: https://www.rfc-editor.org/rfc/rfc9001#section-5.2

use hkdf::Hkdf;
use sha2::Sha256;

use super::expand_label::{ExpandError, SHA256_OUTPUT_LEN, expand_label_from_prk};

/// Initial-salt for QUIC v1 per RFC 9001 §5.2.
pub const INITIAL_SALT_V1: [u8; 20] = [
    0x38, 0x76, 0x2c, 0xf7, 0xf5, 0x59, 0x34, 0xb3, 0x4d, 0x17, 0x9a, 0xe6, 0xa4, 0xc8, 0x0c, 0xad,
    0xcc, 0xbb, 0x7f, 0x0a,
];

/// AEAD key length for AES-128-GCM and ChaCha20-Poly1305 (the two MUST-implement AEADs).
pub const QUIC_KEY_LEN: usize = 16;

/// AEAD nonce base length for both AES-128-GCM and ChaCha20-Poly1305.
pub const QUIC_IV_LEN: usize = 12;

/// Header-protection key length.
pub const QUIC_HP_LEN: usize = 16;

/// Initial-secret (and any expanded-label secret) length when using SHA-256.
pub const QUIC_INITIAL_SECRET_LEN: usize = SHA256_OUTPUT_LEN;

/// Per-side initial keys (AEAD key, AEAD nonce base, header-protection key).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InitialKeys {
    pub key: [u8; QUIC_KEY_LEN],
    pub iv: [u8; QUIC_IV_LEN],
    pub hp: [u8; QUIC_HP_LEN],
}

/// Both directions of initial-packet protection — derived together from
/// a single Destination Connection ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InitialKeyPair {
    pub client: InitialKeys,
    pub server: InitialKeys,
    /// The 32-byte secret material from which the client/server keys were
    /// expanded. Exposed because key updates / retry-integrity verification
    /// re-derive against it.
    pub client_initial_secret: [u8; QUIC_INITIAL_SECRET_LEN],
    pub server_initial_secret: [u8; QUIC_INITIAL_SECRET_LEN],
}

/// Derive both client + server initial keys from a Destination Connection ID
/// per RFC 9001 §5.2.
///
/// # Errors
///
/// Returns [`ExpandError`] from the underlying HKDF-Expand-Label calls;
/// in practice, all six expansions use fixed short labels + fixed short
/// output sizes well within the limits, so this can only fail if the
/// caller passes a malformed `client_dcid` somehow (e.g. exceeding HKDF
/// info-blob limits).
pub fn derive(client_dcid: &[u8]) -> Result<InitialKeyPair, ExpandError> {
    let prk = Hkdf::<Sha256>::new(Some(&INITIAL_SALT_V1), client_dcid);

    let mut client_initial_secret = [0u8; QUIC_INITIAL_SECRET_LEN];
    expand_label_from_prk(&prk, b"client in", b"", &mut client_initial_secret)?;
    let mut server_initial_secret = [0u8; QUIC_INITIAL_SECRET_LEN];
    expand_label_from_prk(&prk, b"server in", b"", &mut server_initial_secret)?;

    let client = derive_side_keys(&client_initial_secret)?;
    let server = derive_side_keys(&server_initial_secret)?;

    Ok(InitialKeyPair {
        client,
        server,
        client_initial_secret,
        server_initial_secret,
    })
}

/// Given an initial secret (client or server), expand to the per-side
/// key/iv/hp triple. Public so callers can derive just one side, e.g.
/// the server's incoming-packet keys without computing its outbound keys.
///
/// # Errors
///
/// See [`ExpandError`].
pub fn derive_side_keys(
    secret: &[u8; QUIC_INITIAL_SECRET_LEN],
) -> Result<InitialKeys, ExpandError> {
    let prk = Hkdf::<Sha256>::from_prk(secret).map_err(|_| ExpandError::OutputTooLong)?;

    let mut key = [0u8; QUIC_KEY_LEN];
    expand_label_from_prk(&prk, b"quic key", b"", &mut key)?;
    let mut iv = [0u8; QUIC_IV_LEN];
    expand_label_from_prk(&prk, b"quic iv", b"", &mut iv)?;
    let mut hp = [0u8; QUIC_HP_LEN];
    expand_label_from_prk(&prk, b"quic hp", b"", &mut hp)?;

    Ok(InitialKeys { key, iv, hp })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // RFC 9001 Appendix A.1 — canonical test vectors with
    // client_dcid = 0x8394c8f03e515708.
    const RFC_DCID: [u8; 8] = [0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08];

    const EXPECTED_CLIENT_INITIAL_SECRET: [u8; 32] = [
        0xc0, 0x0c, 0xf1, 0x51, 0xca, 0x5b, 0xe0, 0x75, 0xed, 0x0e, 0xbf, 0xb5, 0xc8, 0x03, 0x23,
        0xc4, 0x2d, 0x6b, 0x7d, 0xb6, 0x78, 0x81, 0x28, 0x9a, 0xf4, 0x00, 0x8f, 0x1f, 0x6c, 0x35,
        0x7a, 0xea,
    ];
    const EXPECTED_CLIENT_KEY: [u8; 16] = [
        0x1f, 0x36, 0x96, 0x13, 0xdd, 0x76, 0xd5, 0x46, 0x77, 0x30, 0xef, 0xcb, 0xe3, 0xb1, 0xa2,
        0x2d,
    ];
    const EXPECTED_CLIENT_IV: [u8; 12] = [
        0xfa, 0x04, 0x4b, 0x2f, 0x42, 0xa3, 0xfd, 0x3b, 0x46, 0xfb, 0x25, 0x5c,
    ];
    const EXPECTED_CLIENT_HP: [u8; 16] = [
        0x9f, 0x50, 0x44, 0x9e, 0x04, 0xa0, 0xe8, 0x10, 0x28, 0x3a, 0x1e, 0x99, 0x33, 0xad, 0xed,
        0xd2,
    ];

    const EXPECTED_SERVER_INITIAL_SECRET: [u8; 32] = [
        0x3c, 0x19, 0x98, 0x28, 0xfd, 0x13, 0x9e, 0xfd, 0x21, 0x6c, 0x15, 0x5a, 0xd8, 0x44, 0xcc,
        0x81, 0xfb, 0x82, 0xfa, 0x8d, 0x74, 0x46, 0xfa, 0x7d, 0x78, 0xbe, 0x80, 0x3a, 0xcd, 0xda,
        0x95, 0x1b,
    ];
    const EXPECTED_SERVER_KEY: [u8; 16] = [
        0xcf, 0x3a, 0x53, 0x31, 0x65, 0x3c, 0x36, 0x4c, 0x88, 0xf0, 0xf3, 0x79, 0xb6, 0x06, 0x7e,
        0x37,
    ];
    const EXPECTED_SERVER_IV: [u8; 12] = [
        0x0a, 0xc1, 0x49, 0x3c, 0xa1, 0x90, 0x58, 0x53, 0xb0, 0xbb, 0xa0, 0x3e,
    ];
    const EXPECTED_SERVER_HP: [u8; 16] = [
        0xc2, 0x06, 0xb8, 0xd9, 0xb9, 0xf0, 0xf3, 0x76, 0x44, 0x43, 0x0b, 0x49, 0x0e, 0xea, 0xa3,
        0x14,
    ];

    #[test]
    fn rfc_9001_appendix_a1_client_keys_match() {
        let pair = derive(&RFC_DCID).expect("derive");
        assert_eq!(pair.client_initial_secret, EXPECTED_CLIENT_INITIAL_SECRET);
        assert_eq!(pair.client.key, EXPECTED_CLIENT_KEY);
        assert_eq!(pair.client.iv, EXPECTED_CLIENT_IV);
        assert_eq!(pair.client.hp, EXPECTED_CLIENT_HP);
    }

    #[test]
    fn rfc_9001_appendix_a1_server_keys_match() {
        let pair = derive(&RFC_DCID).expect("derive");
        assert_eq!(pair.server_initial_secret, EXPECTED_SERVER_INITIAL_SECRET);
        assert_eq!(pair.server.key, EXPECTED_SERVER_KEY);
        assert_eq!(pair.server.iv, EXPECTED_SERVER_IV);
        assert_eq!(pair.server.hp, EXPECTED_SERVER_HP);
    }

    #[test]
    fn derive_side_keys_matches_derive() {
        let pair = derive(&RFC_DCID).expect("derive");
        let client_keys = derive_side_keys(&pair.client_initial_secret).expect("derive_side_keys");
        assert_eq!(client_keys, pair.client);
    }

    #[test]
    fn deterministic_across_calls() {
        let a = derive(&RFC_DCID).expect("derive a");
        let b = derive(&RFC_DCID).expect("derive b");
        assert_eq!(a, b);
    }

    #[test]
    fn different_dcid_produces_different_keys() {
        let a = derive(&[0u8; 8]).expect("derive a");
        let b = derive(&[1u8; 8]).expect("derive b");
        assert_ne!(a.client.key, b.client.key);
        assert_ne!(a.server.key, b.server.key);
    }

    #[cfg(feature = "quic-alloc")]
    #[test]
    fn variable_length_dcid_supported() {
        // QUIC v1 allows DCID lengths from 0 to 20 bytes; verify the full sweep
        // compiles and produces deterministic output (no panic).
        for len in 0..=20 {
            let dcid = alloc::vec![0xab; len];
            let _ = derive(&dcid).expect("derive");
        }
    }
}
