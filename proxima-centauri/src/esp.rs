//! The child SA: per-packet AEAD with replay protection.
//!
//! This is the **hot path** — one seal or open per packet, versus one handshake
//! per connection. Everything here is in-place over a caller buffer, so a
//! packet is encrypted where it already sits.
//!
//! # Deliberate divergences from `csr-security::ChildSa`
//!
//! The handshake in [`crate::handshake`] is wire-compatible with the oracle
//! because the relay runs it. `ChildSa` is different: it is **not** wire
//! deployed — the relay bypasses it and drives a raw cipher with
//! counter-based nonces, precisely because of the first defect below. With no
//! deployed peer to stay compatible with, the format is fixed rather than
//! reproduced.
//!
//! 1. **The oracle's `nonce_base` is random per instance.** Two peers each
//!    generate their own, so neither can decrypt the other. Here both nonce
//!    bases are *derived* from the session keys, so the two ends agree by
//!    construction.
//! 2. **The oracle writes a 4-byte sequence but derives the nonce from the
//!    full `u64`.** Past 2^32 packets the wire value wraps while the nonce
//!    does not, and the SA silently stops decrypting — reachable in about an
//!    hour at a million packets per second. The sequence is 8 bytes here, so
//!    the wire value and the nonce cannot diverge.
//! 3. **The oracle holds its replay window behind a `std::sync::RwLock`**, in
//!    the per-packet path. A sans-IO type takes `&mut self` instead: the
//!    driver owns exclusivity, there is no lock to acquire, and it compiles
//!    without `std`.
//!
//! Also added: the header is bound as AEAD associated data, so tampering with
//! the SPI or sequence number fails authentication instead of being ignored.

use aead::{AeadInPlace, KeyInit};

use crate::aead::{Nonce, Tag};

use crate::error::CentauriError;
use crate::handshake::{Role, SessionKeys};
use crate::hash::derive_key;
use crate::sized::{ESP_MAX_PAYLOAD_BYTES, REPLAY_WINDOW_WORDS};

/// Poly1305 authentication tag.
pub const TAG_LEN: usize = 16;
/// SPI and sequence number, ahead of the ciphertext.
pub const HEADER_LEN: usize = 12;
/// Bytes a packet costs beyond its payload.
pub const OVERHEAD: usize = HEADER_LEN + TAG_LEN;

/// Packets tracked by the replay window. Baked from
/// `proxima-centauri.toml`'s `[replay].window_packets`.
pub use crate::sized::REPLAY_WINDOW_PACKETS as REPLAY_WINDOW;

const NONCE_LEN: usize = 12;
const NONCE_BASE_LEN: usize = 4;

const SEAL_NONCE_CONTEXT_INITIATOR: &str = "proxima-centauri-esp-nonce-i-v1";
const SEAL_NONCE_CONTEXT_RESPONDER: &str = "proxima-centauri-esp-nonce-r-v1";

/// An ESP security parameter index, identifying a child SA on the wire.
///
/// Distinct from [`crate::handshake::IkeSpi`] and a different width, so the
/// two cannot be confused at a call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EspSpi(u32);

impl EspSpi {
    /// Wrap a raw SPI.
    #[must_use]
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    /// The raw value, for writing to the wire.
    #[must_use]
    pub const fn as_raw(self) -> u32 {
        self.0
    }
}

/// A sliding replay window over the last [`REPLAY_WINDOW`] sequence numbers.
///
/// Bit 0 of `bitmap[0]` is `highest`; bit *n* is `highest - n`. Strict O(1):
/// a fixed bitmap of [`REPLAY_WINDOW_WORDS`] words, shifted by whole words
/// plus a remainder, never resized and never allocated. The window size is
/// build-time configurable, so this adapts rather than assuming 256.
#[derive(Debug, Clone, Copy)]
struct ReplayWindow {
    highest: u64,
    bitmap: [u64; REPLAY_WINDOW_WORDS],
}

impl ReplayWindow {
    const fn new() -> Self {
        Self {
            highest: 0,
            bitmap: [0; REPLAY_WINDOW_WORDS],
        }
    }

    /// Accept `seq` and record it, or reject it as replayed or too old.
    fn admit(&mut self, seq: u64) -> bool {
        if seq == 0 {
            return false;
        }

        if seq > self.highest {
            self.shift(seq - self.highest);
            self.highest = seq;
            self.set(0);
            return true;
        }

        let offset = self.highest - seq;
        if offset >= REPLAY_WINDOW || self.is_set(offset) {
            return false;
        }

        self.set(offset);
        true
    }

    fn shift(&mut self, distance: u64) {
        if distance >= REPLAY_WINDOW {
            self.bitmap = [0; REPLAY_WINDOW_WORDS];
            return;
        }

        let words = (distance / 64) as usize;
        let bits = (distance % 64) as u32;

        if words > 0 {
            for index in (0..REPLAY_WINDOW_WORDS).rev() {
                self.bitmap[index] = if index >= words {
                    self.bitmap[index - words]
                } else {
                    0
                };
            }
        }

        if bits > 0 {
            let mut carry = 0u64;
            for word in &mut self.bitmap {
                let shifted = (*word << bits) | carry;
                carry = *word >> (64 - bits);
                *word = shifted;
            }
        }
    }

    fn set(&mut self, offset: u64) {
        let word = (offset / 64) as usize;
        let bit = offset % 64;
        if let Some(slot) = self.bitmap.get_mut(word) {
            *slot |= 1u64 << bit;
        }
    }

    fn is_set(&self, offset: u64) -> bool {
        let word = (offset / 64) as usize;
        let bit = offset % 64;
        self.bitmap
            .get(word)
            .is_some_and(|slot| slot & (1u64 << bit) != 0)
    }
}

/// The AEAD suite this SA speaks.
///
/// A discriminated enum, not a `Box<dyn Aead>`: the set of suites is closed and
/// known at compile time, so `match` does the dispatch with no indirection and
/// no allocator. Each variant exists only when its feature is on, so a binary
/// that compiles one suite carries exactly one.
///
/// Additive rather than exclusive. Which suites a binary *can* speak is a build
/// decision; which one a session *does* speak is a deployment one, and
/// conflating them is what forces a fork per deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AeadSuite {
    /// ChaCha20-Poly1305. Constant-time in software on every target, which is
    /// why it is the default — a target without AES instructions running
    /// AES-GCM in software is both slower and harder to keep constant-time.
    #[cfg(feature = "aead-chacha20poly1305")]
    ChaCha20Poly1305,
    /// AES-256-GCM. Wins where AES-NI or ARM crypto extensions exist.
    #[cfg(feature = "aead-aes-gcm")]
    Aes256Gcm,
}

impl AeadSuite {
    /// The suite chosen when a caller does not say. ChaCha20-Poly1305 where it
    /// is compiled, since it is the safe answer on any target; AES only when
    /// it is the sole suite built.
    #[cfg(feature = "aead-chacha20poly1305")]
    pub const DEFAULT: Self = Self::ChaCha20Poly1305;

    /// See the other `DEFAULT`.
    #[cfg(all(not(feature = "aead-chacha20poly1305"), feature = "aead-aes-gcm"))]
    pub const DEFAULT: Self = Self::Aes256Gcm;
}

/// The keyed cipher behind a direction.
enum DirectionCipher {
    #[cfg(feature = "aead-chacha20poly1305")]
    ChaCha(chacha20poly1305::ChaCha20Poly1305),
    #[cfg(feature = "aead-aes-gcm")]
    Aes(aes_gcm::Aes256Gcm),
}

impl DirectionCipher {
    fn new(suite: &AeadSuite, key: &[u8; 32]) -> Self {
        match suite {
            #[cfg(feature = "aead-chacha20poly1305")]
            AeadSuite::ChaCha20Poly1305 => {
                Self::ChaCha(chacha20poly1305::ChaCha20Poly1305::new(key.into()))
            }
            #[cfg(feature = "aead-aes-gcm")]
            AeadSuite::Aes256Gcm => Self::Aes(aes_gcm::Aes256Gcm::new(key.into())),
        }
    }

    fn seal(&self, nonce: &Nonce, aad: &[u8], body: &mut [u8]) -> Result<Tag, CentauriError> {
        match self {
            #[cfg(feature = "aead-chacha20poly1305")]
            Self::ChaCha(cipher) => cipher
                .encrypt_in_place_detached(nonce, aad, body)
                .map_err(|_| CentauriError::EncryptionFailed),
            #[cfg(feature = "aead-aes-gcm")]
            Self::Aes(cipher) => cipher
                .encrypt_in_place_detached(nonce, aad, body)
                .map_err(|_| CentauriError::EncryptionFailed),
        }
    }

    fn open(
        &self,
        nonce: &Nonce,
        aad: &[u8],
        body: &mut [u8],
        tag: &Tag,
    ) -> Result<(), CentauriError> {
        match self {
            #[cfg(feature = "aead-chacha20poly1305")]
            Self::ChaCha(cipher) => cipher
                .decrypt_in_place_detached(nonce, aad, body, tag)
                .map_err(|_| CentauriError::AuthenticationFailed),
            #[cfg(feature = "aead-aes-gcm")]
            Self::Aes(cipher) => cipher
                .decrypt_in_place_detached(nonce, aad, body, tag)
                .map_err(|_| CentauriError::AuthenticationFailed),
        }
    }
}

/// A child security association: seal outbound packets, open inbound ones.
///
/// Derived from [`SessionKeys`], so the two ends agree on keys and nonce bases
/// without exchanging anything further.
pub struct ChildSa {
    spi: EspSpi,
    seal_cipher: DirectionCipher,
    open_cipher: DirectionCipher,
    seal_nonce_base: [u8; NONCE_BASE_LEN],
    open_nonce_base: [u8; NONCE_BASE_LEN],
    send_seq: u64,
    replay: ReplayWindow,
}

impl core::fmt::Debug for ChildSa {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ChildSa")
            .field("spi", &self.spi)
            .field("send_seq", &self.send_seq)
            .finish_non_exhaustive()
    }
}

impl ChildSa {
    /// Derive a child SA from an established handshake's keys.
    ///
    /// The nonce bases come from the direction keys, so an initiator and a
    /// responder that agreed on [`SessionKeys`] also agree here — the defect
    /// that makes the oracle's `ChildSa` unusable between two peers.
    #[must_use]
    pub fn from_session(keys: &SessionKeys, role: Role, spi: EspSpi) -> Self {
        Self::from_session_with(keys, role, spi, &AeadSuite::DEFAULT)
    }

    /// As [`ChildSa::from_session`], choosing the AEAD suite explicitly.
    #[must_use]
    pub fn from_session_with(
        keys: &SessionKeys,
        role: Role,
        spi: EspSpi,
        suite: &AeadSuite,
    ) -> Self {
        let initiator_base = derive_key(SEAL_NONCE_CONTEXT_INITIATOR, keys.seed());
        let responder_base = derive_key(SEAL_NONCE_CONTEXT_RESPONDER, keys.seed());

        let (seal_base, open_base) = match role {
            Role::Initiator => (initiator_base, responder_base),
            Role::Responder => (responder_base, initiator_base),
        };

        let mut seal_nonce_base = [0u8; NONCE_BASE_LEN];
        let mut open_nonce_base = [0u8; NONCE_BASE_LEN];
        seal_nonce_base.copy_from_slice(&seal_base[..NONCE_BASE_LEN]);
        open_nonce_base.copy_from_slice(&open_base[..NONCE_BASE_LEN]);

        Self {
            spi,
            seal_cipher: DirectionCipher::new(suite, keys.encrypt_key()),
            open_cipher: DirectionCipher::new(suite, keys.decrypt_key()),
            seal_nonce_base,
            open_nonce_base,
            send_seq: 0,
            replay: ReplayWindow::new(),
        }
    }

    /// This SA's security parameter index.
    #[must_use]
    pub const fn spi(&self) -> EspSpi {
        self.spi
    }

    /// Sequence number of the last sealed packet.
    #[must_use]
    pub const fn send_seq(&self) -> u64 {
        self.send_seq
    }

    /// Seal a packet in place.
    ///
    /// On entry `buffer[HEADER_LEN..HEADER_LEN + payload_len]` holds the
    /// plaintext. On return the packet occupies `buffer[..returned_len]`. The
    /// buffer must therefore be at least `payload_len + OVERHEAD` long.
    ///
    /// # Errors
    ///
    /// - [`CentauriError::PayloadTooLarge`] when the payload exceeds the
    ///   build-time `[esp].max_payload_bytes` cap.
    /// - [`CentauriError::BufferTooSmall`] when the buffer cannot hold the
    ///   header, payload, and tag.
    pub fn seal(&mut self, buffer: &mut [u8], payload_len: usize) -> Result<usize, CentauriError> {
        if payload_len > ESP_MAX_PAYLOAD_BYTES {
            return Err(CentauriError::PayloadTooLarge {
                len: payload_len,
                max: ESP_MAX_PAYLOAD_BYTES,
            });
        }

        let needed = payload_len + OVERHEAD;
        if buffer.len() < needed {
            return Err(CentauriError::BufferTooSmall {
                needed,
                available: buffer.len(),
            });
        }

        self.send_seq = self.send_seq.wrapping_add(1);
        let seq = self.send_seq;

        // split rather than stage: the header is written where it will be sent
        // and then read back as associated data in place, so nothing on the
        // inner loop is copied except the tag the AEAD returns by value.
        let (header, rest) = buffer[..needed].split_at_mut(HEADER_LEN);
        header[0..4].copy_from_slice(&self.spi.as_raw().to_be_bytes());
        header[4..12].copy_from_slice(&seq.to_be_bytes());
        let (body, tag_slot) = rest.split_at_mut(payload_len);

        let nonce = Self::nonce(&self.seal_nonce_base, seq);
        let tag = self.seal_cipher.seal(&nonce, header, body)?;

        tag_slot.copy_from_slice(&tag);

        Ok(needed)
    }

    /// Open a packet in place.
    ///
    /// Returns the payload length; the plaintext sits at
    /// `packet[HEADER_LEN..HEADER_LEN + returned_len]`.
    ///
    /// # Errors
    ///
    /// - [`CentauriError::InvalidMessage`] if the packet is too short to hold a
    ///   header and a tag.
    /// - [`CentauriError::ReplayDetected`] if the sequence number is outside the
    ///   window or already seen. Checked *before* decryption, so a flood of
    ///   replays costs a bitmap lookup rather than an AEAD pass.
    /// - [`CentauriError::AuthenticationFailed`] if the tag does not verify —
    ///   which also covers a tampered SPI or sequence number, since the header
    ///   is bound as associated data.
    pub fn open(&mut self, packet: &mut [u8]) -> Result<usize, CentauriError> {
        if packet.len() < OVERHEAD {
            return Err(CentauriError::InvalidMessage(
                "packet shorter than overhead",
            ));
        }

        let seq = u64::from_be_bytes(
            packet[4..12]
                .try_into()
                .map_err(|_| CentauriError::InvalidMessage("sequence"))?,
        );

        if !self.replay.admit(seq) {
            return Err(CentauriError::ReplayDetected(seq));
        }

        let payload_len = packet.len() - OVERHEAD;

        // header stays where it is and is borrowed as associated data; the tag
        // is verified from the packet rather than copied out of it.
        let (header, rest) = packet.split_at_mut(HEADER_LEN);
        let (body, tag) = rest.split_at_mut(payload_len);

        // `tag` is exactly TAG_LEN by construction — payload_len is
        // packet.len() - OVERHEAD, so splitting it off leaves precisely the
        // tag — and this borrows those 16 bytes rather than copying them.
        let tag_ref: &Tag = (&*tag).into();

        let nonce = Self::nonce(&self.open_nonce_base, seq);
        self.open_cipher.open(&nonce, header, body, tag_ref)?;

        Ok(payload_len)
    }

    /// `base || sequence`, so a nonce can never repeat within an SA and the
    /// wire value and the nonce cannot disagree.
    fn nonce(base: &[u8; NONCE_BASE_LEN], seq: u64) -> Nonce {
        let mut bytes = [0u8; NONCE_LEN];
        bytes[..NONCE_BASE_LEN].copy_from_slice(base);
        bytes[NONCE_BASE_LEN..].copy_from_slice(&seq.to_be_bytes());
        Nonce::from(bytes)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use proxima_clock::ticks::Ticks;

    use super::{
        AeadSuite, ChildSa, ESP_MAX_PAYLOAD_BYTES, EspSpi, HEADER_LEN, OVERHEAD, REPLAY_WINDOW,
        ReplayWindow,
    };
    use crate::entropy::Entropy32;
    use crate::error::CentauriError;
    use crate::handshake::{Handshake, IkeSpi, Role};

    const PSK: [u8; 32] = [0xAB; 32];

    /// Two SAs that agreed via a real handshake — the only honest way to build
    /// a pair, since agreement is the property under test.
    fn agreed_pair() -> (ChildSa, ChildSa) {
        let mut initiator = Handshake::initiator(PSK, IkeSpi::new(1));
        let mut responder = Handshake::responder(PSK, IkeSpi::new(2));
        let now = Ticks::from_raw(1);

        let _ = initiator
            .step(&[], Some(Entropy32::new([0x11; 32])), now)
            .unwrap();
        let mut init = [0u8; 92];
        init.copy_from_slice(initiator.outbound());

        let _ = responder
            .step(&init, Some(Entropy32::new([0x22; 32])), now)
            .unwrap();
        let mut reply = [0u8; 92];
        reply.copy_from_slice(responder.outbound());

        let _ = initiator.step(&reply, None, now).unwrap();

        (
            ChildSa::from_session(
                initiator.keys().unwrap(),
                Role::Initiator,
                EspSpi::new(0xAAAA),
            ),
            ChildSa::from_session(
                responder.keys().unwrap(),
                Role::Responder,
                EspSpi::new(0xBBBB),
            ),
        )
    }

    #[test]
    fn sealed_packet_opens_on_the_peer() {
        let (mut sender, mut receiver) = agreed_pair();
        let payload = b"centauri esp payload";

        let mut buffer = [0u8; 128];
        buffer[HEADER_LEN..HEADER_LEN + payload.len()].copy_from_slice(payload);
        let packet_len = sender.seal(&mut buffer, payload.len()).unwrap();

        assert_eq!(packet_len, payload.len() + OVERHEAD);

        let opened = receiver.open(&mut buffer[..packet_len]).unwrap();

        assert_eq!(opened, payload.len());
        assert_eq!(&buffer[HEADER_LEN..HEADER_LEN + opened], payload);
    }

    #[test]
    fn both_directions_work_independently() {
        let (mut initiator, mut responder) = agreed_pair();

        let mut outbound = [0u8; 64];
        outbound[HEADER_LEN..HEADER_LEN + 4].copy_from_slice(b"i->r");
        let len = initiator.seal(&mut outbound, 4).unwrap();
        assert_eq!(responder.open(&mut outbound[..len]).unwrap(), 4);
        assert_eq!(&outbound[HEADER_LEN..HEADER_LEN + 4], b"i->r");

        let mut inbound = [0u8; 64];
        inbound[HEADER_LEN..HEADER_LEN + 4].copy_from_slice(b"r->i");
        let len = responder.seal(&mut inbound, 4).unwrap();
        assert_eq!(initiator.open(&mut inbound[..len]).unwrap(), 4);
        assert_eq!(&inbound[HEADER_LEN..HEADER_LEN + 4], b"r->i");
    }

    #[test]
    fn a_replayed_packet_is_rejected() {
        let (mut sender, mut receiver) = agreed_pair();
        let mut buffer = [0u8; 64];
        buffer[HEADER_LEN..HEADER_LEN + 4].copy_from_slice(b"once");
        let len = sender.seal(&mut buffer, 4).unwrap();

        let replay = buffer;
        assert_eq!(receiver.open(&mut buffer[..len]).unwrap(), 4);

        let mut again = replay;
        assert_eq!(
            receiver.open(&mut again[..len]).err(),
            Some(CentauriError::ReplayDetected(1)),
            "the same sequence number must not be accepted twice"
        );
    }

    #[test]
    fn a_tampered_header_fails_authentication() {
        let (mut sender, mut receiver) = agreed_pair();
        let mut buffer = [0u8; 64];
        buffer[HEADER_LEN..HEADER_LEN + 4].copy_from_slice(b"bind");
        let len = sender.seal(&mut buffer, 4).unwrap();

        buffer[0] ^= 0xFF;

        assert_eq!(
            receiver.open(&mut buffer[..len]).err(),
            Some(CentauriError::AuthenticationFailed),
            "the header is associated data, so editing the spi must not pass"
        );
    }

    #[test]
    fn a_tampered_ciphertext_fails_authentication() {
        let (mut sender, mut receiver) = agreed_pair();
        let mut buffer = [0u8; 64];
        buffer[HEADER_LEN..HEADER_LEN + 4].copy_from_slice(b"peek");
        let len = sender.seal(&mut buffer, 4).unwrap();

        buffer[HEADER_LEN] ^= 0x01;

        assert_eq!(
            receiver.open(&mut buffer[..len]).err(),
            Some(CentauriError::AuthenticationFailed)
        );
    }

    #[test]
    fn a_short_buffer_is_reported_with_the_shortfall() {
        let (mut sender, _) = agreed_pair();
        let mut buffer = [0u8; 20];

        assert_eq!(
            sender.seal(&mut buffer, 16).err(),
            Some(CentauriError::BufferTooSmall {
                needed: 16 + OVERHEAD,
                available: 20,
            })
        );
    }

    #[test]
    fn a_payload_past_the_configured_cap_is_refused() {
        let (mut sender, _) = agreed_pair();
        let oversize = ESP_MAX_PAYLOAD_BYTES + 1;
        let mut buffer = [0u8; 64];

        assert_eq!(
            sender.seal(&mut buffer, oversize).err(),
            Some(CentauriError::PayloadTooLarge {
                len: oversize,
                max: ESP_MAX_PAYLOAD_BYTES,
            }),
            "the cap is checked before the buffer, so the caller learns the real reason"
        );
    }

    #[test]
    fn a_runt_packet_is_rejected_before_decryption() {
        let (_, mut receiver) = agreed_pair();
        let mut runt = [0u8; OVERHEAD - 1];

        assert_eq!(
            receiver.open(&mut runt).err(),
            Some(CentauriError::InvalidMessage(
                "packet shorter than overhead"
            ))
        );
    }

    /// A pair speaking a named suite, so a test can pin the choice rather
    /// than inherit the default.
    fn agreed_pair_with(suite: &AeadSuite) -> (ChildSa, ChildSa) {
        let mut initiator = Handshake::initiator(PSK, IkeSpi::new(1));
        let mut responder = Handshake::responder(PSK, IkeSpi::new(2));
        let now = Ticks::from_raw(1);

        let _ = initiator
            .step(&[], Some(Entropy32::new([0x11; 32])), now)
            .unwrap();
        let mut init = [0u8; 92];
        init.copy_from_slice(initiator.outbound());
        let _ = responder
            .step(&init, Some(Entropy32::new([0x22; 32])), now)
            .unwrap();
        let mut reply = [0u8; 92];
        reply.copy_from_slice(responder.outbound());
        let _ = initiator.step(&reply, None, now).unwrap();

        (
            ChildSa::from_session_with(
                initiator.keys().unwrap(),
                Role::Initiator,
                EspSpi::new(0xAAAA),
                suite,
            ),
            ChildSa::from_session_with(
                responder.keys().unwrap(),
                Role::Responder,
                EspSpi::new(0xBBBB),
                suite,
            ),
        )
    }

    fn round_trips(suite: &AeadSuite) {
        let (mut sender, mut receiver) = agreed_pair_with(suite);
        let payload = b"suite round trip";
        let mut buffer = [0u8; 128];
        buffer[HEADER_LEN..HEADER_LEN + payload.len()].copy_from_slice(payload);

        let len = sender.seal(&mut buffer, payload.len()).unwrap();
        let opened = receiver.open(&mut buffer[..len]).unwrap();

        assert_eq!(opened, payload.len(), "{suite:?}");
        assert_eq!(
            &buffer[HEADER_LEN..HEADER_LEN + opened],
            payload,
            "{suite:?}"
        );
    }

    #[cfg(feature = "aead-chacha20poly1305")]
    #[test]
    fn chacha20poly1305_round_trips() {
        round_trips(&AeadSuite::ChaCha20Poly1305);
    }

    #[cfg(feature = "aead-aes-gcm")]
    #[test]
    fn aes256gcm_round_trips() {
        round_trips(&AeadSuite::Aes256Gcm);
    }

    #[cfg(all(feature = "aead-chacha20poly1305", feature = "aead-aes-gcm"))]
    #[test]
    fn a_packet_sealed_with_one_suite_does_not_open_with_the_other() {
        // the suite is a wire agreement: peers that disagree must fail closed,
        // not silently produce garbage
        let (mut sender, _) = agreed_pair_with(&AeadSuite::ChaCha20Poly1305);
        let (_, mut receiver) = agreed_pair_with(&AeadSuite::Aes256Gcm);

        let mut buffer = [0u8; 128];
        buffer[HEADER_LEN..HEADER_LEN + 8].copy_from_slice(b"mismatch");
        let len = sender.seal(&mut buffer, 8).unwrap();

        assert_eq!(
            receiver.open(&mut buffer[..len]).err(),
            Some(CentauriError::AuthenticationFailed),
            "a suite mismatch must fail closed"
        );
    }

    #[cfg(feature = "aead-chacha20poly1305")]
    #[test]
    fn chacha_is_the_default_where_it_is_compiled() {
        // the safe answer on any target is the one a caller gets for free
        assert_eq!(AeadSuite::DEFAULT, AeadSuite::ChaCha20Poly1305);
    }

    #[test]
    fn every_single_bit_flip_in_a_packet_is_rejected() {
        // Exhaustive, and the property is total: unlike the handshake — which
        // validates only two header bytes — an AEAD packet has every byte
        // under the tag, so there is no position where a flip may pass. A
        // fresh receiver per flip, because opening advances the replay window.
        let payload = [0x5Au8; 16];
        let (mut sender, _) = agreed_pair();
        let mut sealed = [0u8; 16 + OVERHEAD];
        sealed[HEADER_LEN..HEADER_LEN + payload.len()].copy_from_slice(&payload);
        let packet_len = sender.seal(&mut sealed, payload.len()).unwrap();

        let mut authentication_failures = 0usize;
        let mut replay_rejections = 0usize;

        for byte_index in 0..packet_len {
            for bit in 0..8u32 {
                let (_, mut receiver) = agreed_pair();
                let mut corrupted = sealed;
                corrupted[byte_index] ^= 1u8 << bit;

                match receiver.open(&mut corrupted[..packet_len]) {
                    Err(CentauriError::AuthenticationFailed) => authentication_failures += 1,
                    // flipping a sequence bit can move the packet outside the
                    // replay window, which is refused before the AEAD runs —
                    // still a rejection, just an earlier one
                    Err(CentauriError::ReplayDetected(_)) => replay_rejections += 1,
                    other => panic!("byte {byte_index} bit {bit} was NOT rejected: {other:?}"),
                }
            }
        }

        assert_eq!(
            authentication_failures + replay_rejections,
            packet_len * 8,
            "every bit of an AEAD packet must be under the tag"
        );
        assert!(authentication_failures > 0 && replay_rejections > 0);
    }

    #[test]
    fn every_truncation_of_a_packet_is_rejected() {
        let payload = [0x5Au8; 16];
        let (mut sender, _) = agreed_pair();
        let mut sealed = [0u8; 16 + OVERHEAD];
        sealed[HEADER_LEN..HEADER_LEN + payload.len()].copy_from_slice(&payload);
        let packet_len = sender.seal(&mut sealed, payload.len()).unwrap();

        for length in 0..packet_len {
            let (_, mut receiver) = agreed_pair();
            let mut truncated = sealed;

            let outcome = receiver.open(&mut truncated[..length]);

            assert!(
                outcome.is_err(),
                "a {length}-byte truncation of a {packet_len}-byte packet was accepted"
            );
        }
    }

    #[test]
    fn arbitrary_bytes_never_panic_the_opener() {
        for filler in [0x00u8, 0xFF, 0x5A] {
            for length in [0, 1, OVERHEAD - 1, OVERHEAD, OVERHEAD + 1, 64] {
                let (_, mut receiver) = agreed_pair();
                let mut buffer = [0u8; 64];
                let bounded = length.min(buffer.len());
                buffer[..bounded].fill(filler);

                // contract is "never panics"; a garbage packet must not be
                // opened, but any Err is acceptable
                let outcome = receiver.open(&mut buffer[..bounded]);
                assert!(outcome.is_err(), "garbage of {bounded} bytes was opened");
            }
        }
    }

    #[test]
    fn out_of_order_within_the_window_is_accepted() {
        let mut window = ReplayWindow::new();

        assert!(window.admit(10));
        assert!(window.admit(8), "older but inside the window");
        assert!(window.admit(9));
        assert!(!window.admit(9), "and not twice");
    }

    #[test]
    fn packets_older_than_the_window_are_rejected() {
        let mut window = ReplayWindow::new();

        assert!(window.admit(REPLAY_WINDOW + 10));
        assert!(!window.admit(1), "far outside the window");
        assert!(window.admit(REPLAY_WINDOW + 9), "just inside it");
    }

    #[test]
    fn a_large_jump_clears_the_window() {
        let mut window = ReplayWindow::new();

        assert!(window.admit(5));
        assert!(window.admit(5 + REPLAY_WINDOW * 2));
        assert!(
            !window.admit(5),
            "the old entry is gone, so it reads as too old"
        );
    }

    #[test]
    fn sequence_zero_is_never_admitted() {
        let mut window = ReplayWindow::new();

        assert!(!window.admit(0), "sealing starts at one");
    }

    #[test]
    fn a_long_run_of_sequential_packets_all_open() {
        let (mut sender, mut receiver) = agreed_pair();

        for expected in 1..=300u64 {
            let mut buffer = [0u8; 64];
            buffer[HEADER_LEN..HEADER_LEN + 8].copy_from_slice(&expected.to_be_bytes());
            let len = sender.seal(&mut buffer, 8).unwrap();

            assert_eq!(sender.send_seq(), expected);
            assert_eq!(receiver.open(&mut buffer[..len]).unwrap(), 8);
            assert_eq!(&buffer[HEADER_LEN..HEADER_LEN + 8], &expected.to_be_bytes());
        }
    }

    #[test]
    fn peers_derive_matching_nonce_bases() {
        let (initiator, responder) = agreed_pair();

        assert_eq!(
            initiator.seal_nonce_base, responder.open_nonce_base,
            "what one seals with, the other must open with"
        );
        assert_eq!(initiator.open_nonce_base, responder.seal_nonce_base);
        assert_ne!(
            initiator.seal_nonce_base, initiator.open_nonce_base,
            "the two directions must not share a nonce base"
        );
    }
}
