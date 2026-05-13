//! Per-epoch packet/header key material yielded by a [`TlsProvider`].
//!
//! TLS 1.3 + QUIC uses a fixed three-cipher-suite set per RFC 8446 §B.4
//! and RFC 9001 §5.4: AES-128-GCM, AES-256-GCM, and ChaCha20-Poly1305.
//! Header protection uses AES (for AES-suite packet AEADs) or ChaCha20
//! (for the ChaCha20-Poly1305 suite). The enums below are CLOSED — the
//! RFC enumerates the complete set — so a `match` over them is total
//! at compile time.
//!
//! Each [`EpochSecrets`] owns key material in BOTH directions —
//! "local" (used to protect outbound) and "remote" (used to unprotect
//! inbound). Per the TlsProvider resolution this naming is provider-
//! perspective and is independent of `Side::Client` / `Side::Server`.
//!
//! [`TlsProvider`]: super::TlsProvider

use crate::quic::crypto::initial_keys::{QUIC_HP_LEN, QUIC_IV_LEN, QUIC_KEY_LEN};

/// QUIC packet protection epoch per RFC 9001 §5.
///
/// Each epoch has its own AEAD + header-protection keys per RFC 9001
/// §5.2 + §4.6. Packet-number spaces are a separate axis:
///
/// - [`Self::Initial`], [`Self::Handshake`], [`Self::Application`]
///   each have their OWN PN space (RFC 9000 §12.3).
/// - [`Self::ZeroRtt`] has its own AEAD keys (derived from the
///   TLS 1.3 early-data secret) but **shares the Application PN space**
///   per RFC 9001 §2.1 — "0-RTT and 1-RTT data exist in the same
///   packet number space to make loss recovery algorithms easier."
///
/// [`Self::pn_space_index`] gives the PN-space axis (3 distinct
/// values); [`Self::index`] gives the key-epoch axis (4 distinct
/// values). Call sites for PN-space-keyed arrays use the former;
/// AEAD / header-protection key lookups use the latter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum Epoch {
    /// Initial-packet epoch (RFC 9001 §5.2, derived from `INITIAL_SALT_V1`).
    Initial,
    /// 0-RTT-packet epoch (TLS 1.3 client_early_traffic_secret).
    /// Shares the Application PN space per RFC 9001 §2.1.
    ZeroRtt,
    /// Handshake-packet epoch (TLS 1.3 handshake-traffic secrets).
    Handshake,
    /// 1-RTT-packet epoch (TLS 1.3 application-traffic secrets).
    Application,
}

impl Epoch {
    /// All key epochs in protocol-deepening order (4 distinct values).
    ///
    /// Note: this is the KEY axis. For PN-space iteration use
    /// [`Self::pn_spaces`] (3 distinct values).
    #[must_use]
    pub const fn all() -> [Self; 4] {
        [
            Self::Initial,
            Self::ZeroRtt,
            Self::Handshake,
            Self::Application,
        ]
    }

    /// All distinct packet-number spaces (3 distinct values per
    /// RFC 9000 §12.3). 0-RTT and 1-RTT share the Application PN
    /// space per RFC 9001 §2.1, so ZeroRtt is NOT in this list.
    #[must_use]
    pub const fn pn_spaces() -> [Self; 3] {
        [Self::Initial, Self::Handshake, Self::Application]
    }

    /// PN-space index in `0..3`. 0-RTT and Application share index 2
    /// per RFC 9001 §2.1.
    ///
    /// Use this for PN-space-keyed arrays (loss detector, ACK
    /// scheduler, packet-number space tracker).
    #[must_use]
    pub const fn pn_space_index(self) -> usize {
        match self {
            Self::Initial => 0,
            Self::Handshake => 1,
            Self::ZeroRtt | Self::Application => 2,
        }
    }

    /// Backwards-compatible alias for [`Self::pn_space_index`] — every
    /// existing `Epoch::index()` call site is PN-space-keyed; this
    /// preserves source compatibility while making the semantic
    /// explicit on inspection.
    #[must_use]
    pub const fn index(self) -> usize {
        self.pn_space_index()
    }

    /// True iff this epoch shares the Application PN space (i.e.
    /// 0-RTT or 1-RTT).
    #[must_use]
    pub const fn shares_application_pn_space(self) -> bool {
        matches!(self, Self::ZeroRtt | Self::Application)
    }
}

/// Direction a packet flows relative to the calling provider.
///
/// `Local` = outbound (we protect with this key);
/// `Remote` = inbound (we unprotect with this key).
/// Provider-perspective; the `Side::Client` / `Side::Server` role does
/// not enter into the encoding. Server's RX key == client's TX key, and
/// vice versa — this naming makes the symmetry explicit at every call
/// site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum Direction {
    /// Local-to-peer (we protect).
    Local,
    /// Peer-to-local (we unprotect).
    Remote,
}

impl Direction {
    /// The opposite direction.
    #[must_use]
    pub const fn flip(self) -> Self {
        match self {
            Self::Local => Self::Remote,
            Self::Remote => Self::Local,
        }
    }
}

/// The QUIC packet-protection AEAD plus its IV (RFC 9001 §5.1).
///
/// One variant per RFC 8446 §B.4 cipher suite. The packet-number
/// length-bytes-and-bits ranges are encoded in [`build_nonce`] for
/// all three suites; only the underlying AEAD differs.
///
/// [`build_nonce`]: crate::quic::crypto::aead::build_nonce
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum PacketKeyMaterial {
    /// AES-128-GCM packet AEAD (TLS_AES_128_GCM_SHA256).
    Aes128Gcm {
        key: [u8; QUIC_KEY_LEN],
        iv: [u8; QUIC_IV_LEN],
    },
    /// AES-256-GCM packet AEAD (TLS_AES_256_GCM_SHA384). 32-byte key.
    Aes256Gcm {
        key: [u8; 32],
        iv: [u8; QUIC_IV_LEN],
    },
    /// ChaCha20-Poly1305 packet AEAD (TLS_CHACHA20_POLY1305_SHA256).
    ChaCha20Poly1305 {
        key: [u8; 32],
        iv: [u8; QUIC_IV_LEN],
    },
    /// External AEAD impl supplied by a TLS provider (e.g. rustls's
    /// `Box<dyn PacketKey>`). The provider's `seal_in_place` /
    /// `open_in_place` are invoked instead of dispatching on a raw
    /// key triple. Only present when the `tls-rustls` feature is on.
    ///
    /// Per principle 11 / axiom D the proto crate avoids `Box<dyn>`
    /// throughout, but real-world crypto providers (rustls, ring,
    /// aws-lc-rs) all expose AEAD state via trait objects — see
    /// `docs/proxima-quic/edges.md` "RustlsProvider design impedance".
    /// The `External` variant is the principled escape hatch: opt-in,
    /// feature-gated, and only the std-tier dispatch helpers ever
    /// observe it.
    #[cfg(feature = "quic-tls-rustls")]
    External {
        aead: alloc::sync::Arc<dyn ExternalPacketKey + Send + Sync>,
    },
}

/// External packet-AEAD trait — provider-supplied seal/open in place.
/// Mirrors rustls's `quic::PacketKey` shape.
#[cfg(feature = "quic-tls-rustls")]
pub trait ExternalPacketKey: core::fmt::Debug {
    /// Encrypt `payload` (plaintext on entry, ciphertext + 16-byte tag
    /// on exit) in place, with `aad` over the unprotected header.
    ///
    /// # Errors
    ///
    /// Returns [`super::TlsError::ProviderInternal`] when the provider
    /// rejects the input.
    fn seal_in_place(
        &self,
        packet_number: u64,
        aad: &[u8],
        payload: &mut [u8],
    ) -> Result<(), super::TlsError>;

    /// Decrypt + authenticate `payload` (ciphertext+tag on entry,
    /// plaintext on success) in place. Returns the plaintext length
    /// (== payload.len() - tag_len). On auth failure, contents are
    /// unspecified and caller MUST discard the packet.
    ///
    /// # Errors
    ///
    /// Returns [`super::TlsError::DecryptError`] on auth-tag mismatch.
    fn open_in_place(
        &self,
        packet_number: u64,
        aad: &[u8],
        payload: &mut [u8],
    ) -> Result<usize, super::TlsError>;
}

/// QUIC header-protection key per RFC 9001 §5.4.
///
/// AES suites use AES-ECB on the sample; ChaCha20-Poly1305 uses a
/// ChaCha20 keystream on the sample. One variant per AEAD suite.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum HeaderKeyMaterial {
    /// AES-128 header protection (16-byte key).
    Aes128 { hp: [u8; QUIC_HP_LEN] },
    /// AES-256 header protection (32-byte key).
    Aes256 { hp: [u8; 32] },
    /// ChaCha20 header protection (32-byte key).
    ChaCha20 { hp: [u8; 32] },
    /// External HP impl supplied by a TLS provider. See
    /// [`PacketKeyMaterial::External`] for the principle 11 rationale.
    #[cfg(feature = "quic-tls-rustls")]
    External {
        hp: alloc::sync::Arc<dyn ExternalHeaderKey + Send + Sync>,
    },
}

/// External header-protection trait — apply/remove HP in place
/// matching rustls's API shape (mask computation + bit-truncation
/// happen inside the provider).
#[cfg(feature = "quic-tls-rustls")]
pub trait ExternalHeaderKey: core::fmt::Debug {
    /// Apply header protection in place: XOR mask into first byte's
    /// low 4 (long form) / 5 (short form) bits + into the PN bytes.
    /// `sample` is the 16-byte payload sample at offset
    /// `pn_offset + 4` per RFC 9001 §5.4.2.
    fn encrypt_in_place(
        &self,
        sample: &[u8; 16],
        first: &mut u8,
        packet_number: &mut [u8],
    ) -> Result<(), super::TlsError>;

    /// Remove header protection in place. Same bit-twiddle as
    /// `encrypt_in_place` (XOR is its own inverse) but the function
    /// is named separately to match rustls's shape.
    fn decrypt_in_place(
        &self,
        sample: &[u8; 16],
        first: &mut u8,
        packet_number: &mut [u8],
    ) -> Result<(), super::TlsError>;
}

/// Per-direction packet + header keys.
#[derive(Debug, Clone)]
pub struct DirectionalKeys {
    pub packet: PacketKeyMaterial,
    pub header: HeaderKeyMaterial,
}

/// Per-epoch secrets a [`TlsProvider`] hands out via the event sink.
///
/// `generation` is the key-update sequence number (RFC 9001 §6.1) —
/// always 0 for `Initial` and `Handshake`; increments by 1 each time
/// the local or remote initiates a `KEY_UPDATE` on the `Application`
/// epoch. Wrap-around at u8 is well after a connection has long since
/// closed; even continuous key updates every 1 ms take ~4 minutes to
/// wrap, and RFC 9001 §6.1 mandates a 3× max_ack_delay floor.
///
/// [`TlsProvider`]: super::TlsProvider
#[derive(Debug, Clone)]
pub struct EpochSecrets {
    pub epoch: Epoch,
    pub generation: u8,
    pub local: DirectionalKeys,
    pub remote: DirectionalKeys,
}

impl EpochSecrets {
    /// Return the directional keys for the requested direction.
    #[must_use]
    pub const fn for_direction(&self, direction: Direction) -> &DirectionalKeys {
        match direction {
            Direction::Local => &self.local,
            Direction::Remote => &self.remote,
        }
    }
}

impl DirectionalKeys {
    /// Decompose into the AES-128-GCM triple `(key, iv, hp)` if both
    /// the packet AEAD and the header-protection scheme are AES-128
    /// suited. Returns `None` for AES-256 / ChaCha20 variants —
    /// callers handle those via the other branches of
    /// [`PacketKeyMaterial`] / [`HeaderKeyMaterial`].
    #[must_use]
    pub const fn aes128_triple(
        &self,
    ) -> Option<(&[u8; QUIC_KEY_LEN], &[u8; QUIC_IV_LEN], &[u8; QUIC_HP_LEN])> {
        match (&self.packet, &self.header) {
            (PacketKeyMaterial::Aes128Gcm { key, iv }, HeaderKeyMaterial::Aes128 { hp }) => {
                Some((key, iv, hp))
            }
            _ => None,
        }
    }

    /// Decompose into the AES-256-GCM triple `(key, iv, hp)` if both
    /// the packet AEAD and the header-protection scheme are AES-256
    /// suited.
    #[must_use]
    pub const fn aes256_triple(&self) -> Option<(&[u8; 32], &[u8; QUIC_IV_LEN], &[u8; 32])> {
        match (&self.packet, &self.header) {
            (PacketKeyMaterial::Aes256Gcm { key, iv }, HeaderKeyMaterial::Aes256 { hp }) => {
                Some((key, iv, hp))
            }
            _ => None,
        }
    }

    /// Decompose into the ChaCha20-Poly1305 triple `(key, iv, hp)`.
    #[must_use]
    pub const fn chacha20_triple(&self) -> Option<(&[u8; 32], &[u8; QUIC_IV_LEN], &[u8; 32])> {
        match (&self.packet, &self.header) {
            (
                PacketKeyMaterial::ChaCha20Poly1305 { key, iv },
                HeaderKeyMaterial::ChaCha20 { hp },
            ) => Some((key, iv, hp)),
            _ => None,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn sample_keys() -> DirectionalKeys {
        DirectionalKeys {
            packet: PacketKeyMaterial::Aes128Gcm {
                key: [0xAA; QUIC_KEY_LEN],
                iv: [0xBB; QUIC_IV_LEN],
            },
            header: HeaderKeyMaterial::Aes128 {
                hp: [0xCC; QUIC_HP_LEN],
            },
        }
    }

    #[test]
    fn epoch_index_is_stable() {
        // PN-space-keyed index. ZeroRtt shares Application's slot
        // per RFC 9001 §2.1.
        assert_eq!(Epoch::Initial.index(), 0);
        assert_eq!(Epoch::Handshake.index(), 1);
        assert_eq!(Epoch::Application.index(), 2);
        assert_eq!(Epoch::ZeroRtt.index(), 2);
        assert_eq!(
            Epoch::ZeroRtt.pn_space_index(),
            Epoch::Application.pn_space_index()
        );
    }

    #[test]
    fn epoch_all_is_deepening_order() {
        // 4 key epochs: Initial → ZeroRtt → Handshake → Application.
        let order = Epoch::all();
        assert_eq!(order[0], Epoch::Initial);
        assert_eq!(order[1], Epoch::ZeroRtt);
        assert_eq!(order[2], Epoch::Handshake);
        assert_eq!(order[3], Epoch::Application);
    }

    #[test]
    fn epoch_pn_spaces_returns_three_distinct_spaces() {
        let spaces = Epoch::pn_spaces();
        assert_eq!(spaces[0], Epoch::Initial);
        assert_eq!(spaces[1], Epoch::Handshake);
        assert_eq!(spaces[2], Epoch::Application);
        // ZeroRtt MUST NOT be in this list — it shares Application's PN space.
        assert!(!spaces.contains(&Epoch::ZeroRtt));
    }

    #[test]
    fn epoch_shares_application_pn_space_predicate() {
        assert!(Epoch::ZeroRtt.shares_application_pn_space());
        assert!(Epoch::Application.shares_application_pn_space());
        assert!(!Epoch::Initial.shares_application_pn_space());
        assert!(!Epoch::Handshake.shares_application_pn_space());
    }

    #[test]
    fn direction_flip_inverts() {
        assert_eq!(Direction::Local.flip(), Direction::Remote);
        assert_eq!(Direction::Remote.flip(), Direction::Local);
    }

    #[test]
    fn for_direction_selects_correct_keys() {
        let secrets = EpochSecrets {
            epoch: Epoch::Initial,
            generation: 0,
            local: sample_keys(),
            remote: DirectionalKeys {
                packet: PacketKeyMaterial::Aes128Gcm {
                    key: [0x11; QUIC_KEY_LEN],
                    iv: [0x22; QUIC_IV_LEN],
                },
                header: HeaderKeyMaterial::Aes128 {
                    hp: [0x33; QUIC_HP_LEN],
                },
            },
        };
        // PartialEq dropped (Arc<dyn> in External variant); pointer-
        // identity check is sufficient for the accessor contract.
        let local_ptr = secrets.for_direction(Direction::Local) as *const _;
        let remote_ptr = secrets.for_direction(Direction::Remote) as *const _;
        assert_eq!(local_ptr, &secrets.local as *const _);
        assert_eq!(remote_ptr, &secrets.remote as *const _);
    }
}
