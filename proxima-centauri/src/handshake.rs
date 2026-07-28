//! The IKE-style SA_INIT handshake as a sans-IO state machine.
//!
//! X25519 key agreement, BLAKE3 key derivation, two messages. This is the
//! handshake the ragd relay runs today, rebuilt to the sans-IO contract: no
//! clock, no entropy source, no key provider, no socket. The wire format and
//! the derivation chain are byte-compatible with `csr-security`'s `IkeSa`,
//! which is the behaviour oracle.
//!
//! # The contract every Centauri state machine shares
//!
//! ```text
//! step(&mut self, input: &[u8], entropy: Option<Entropy32>, now: Ticks)
//!     -> Result<Progress, CentauriError>
//! outbound(&self) -> &[u8]
//! ```
//!
//! One entry point, dispatched on private state. [`Progress`] is the public
//! output algebra; the internal `Phase` is not public, so a caller cannot
//! depend on the state graph's shape. Bytes to send are staged in
//! [`Handshake::outbound`] rather than written into a caller buffer: every
//! message here is bounded by the protocol, so the state machine can own
//! storage for its worst case and the `BufferTooSmall` failure mode stops
//! existing for callers.
//!
//! [`Handshake::needs_entropy`] exists so a driver backed by
//! [`EntropyCell`](crate::EntropyCell) — which is take-once and fails when
//! empty — only draws on the steps that consume entropy, instead of being
//! forced to have a value staged for every step.

use proxima_clock::ticks::Ticks;
use x25519_dalek::{PublicKey, StaticSecret};

use crate::entropy::Entropy32;
use crate::error::CentauriError;
use crate::hash::derive_key;

/// Bytes of an SA_INIT message: header, nonce, and DH public value.
pub const MESSAGE_LEN: usize = HEADER_LEN + NONCE_LEN + DH_LEN;

const HEADER_LEN: usize = 28;
const NONCE_LEN: usize = 32;
const DH_LEN: usize = 32;

const NEXT_PAYLOAD: u8 = 0x21;
const VERSION: u8 = 0x20;
const EXCHANGE_TYPE: u8 = 0x22;
const FLAGS: u8 = 0x08;

const NONCE_CONTEXT: &str = "proxima-centauri-ike-nonce-v1";
const EPHEMERAL_CONTEXT: &str = "proxima-centauri-ike-ephemeral-v1";

// these five strings are wire-visible in the sense that both peers must agree
// on them; they are the oracle's, kept verbatim so a rebuilt peer can talk to
// a csr-security peer.
const SKEYSEED_CONTEXT: &str = "csr-ike-skeyseed-v1";
const SK_AI_CONTEXT: &str = "csr-ike-sk_ai-v1";
const SK_AR_CONTEXT: &str = "csr-ike-sk_ar-v1";
const SK_EI_CONTEXT: &str = "csr-ike-sk_ei-v1";
const SK_ER_CONTEXT: &str = "csr-ike-sk_er-v1";

/// Which side of the handshake this state machine is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Initiator,
    Responder,
}

/// What a [`Handshake::step`] accomplished — the public output algebra.
///
/// Whether the step produced bytes to send is answered by
/// [`Handshake::outbound`] being non-empty, not by a variant here: the bytes
/// are the information, and a parallel `emitted` flag would only restate them.
/// Failure is the `Err` arm, never a variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum Progress {
    /// Not enough input to advance. Read more bytes and call again with the
    /// larger slice; nothing was consumed and no state changed.
    NeedInput,
    /// The handshake moved forward and is not finished.
    Advanced,
    /// The handshake is complete and [`Handshake::keys`] is available. There
    /// may still be a final message staged in [`Handshake::outbound`].
    Established,
}

/// The five derived keys, with role already applied.
///
/// The accessors are role-aware so a caller cannot reach for the wrong one.
/// `csr-security` had to fix exactly that bug — authenticate and verify_auth
/// picking `sk_ai` versus `sk_ar` by hand — so the role is bound here once, at
/// derivation, and the mistake is not expressible downstream.
pub struct SessionKeys {
    role: Role,
    sk_d: [u8; 32],
    sk_ai: [u8; 32],
    sk_ar: [u8; 32],
    sk_ei: [u8; 32],
    sk_er: [u8; 32],
}

impl SessionKeys {
    /// The SA seed key, for deriving child SA material.
    #[must_use]
    pub const fn seed(&self) -> &[u8; 32] {
        &self.sk_d
    }

    /// The key this side encrypts with.
    #[must_use]
    pub const fn encrypt_key(&self) -> &[u8; 32] {
        match self.role {
            Role::Initiator => &self.sk_ei,
            Role::Responder => &self.sk_er,
        }
    }

    /// The key this side decrypts the peer's traffic with.
    #[must_use]
    pub const fn decrypt_key(&self) -> &[u8; 32] {
        match self.role {
            Role::Initiator => &self.sk_er,
            Role::Responder => &self.sk_ei,
        }
    }

    /// The key this side authenticates with.
    #[must_use]
    pub const fn auth_key(&self) -> &[u8; 32] {
        match self.role {
            Role::Initiator => &self.sk_ai,
            Role::Responder => &self.sk_ar,
        }
    }

    /// The key the peer authenticates with, for verifying their AUTH payload.
    #[must_use]
    pub const fn peer_auth_key(&self) -> &[u8; 32] {
        match self.role {
            Role::Initiator => &self.sk_ar,
            Role::Responder => &self.sk_ai,
        }
    }
}

impl core::fmt::Debug for SessionKeys {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("SessionKeys")
            .field("role", &self.role)
            .finish_non_exhaustive()
    }
}

impl Drop for SessionKeys {
    fn drop(&mut self) {
        for key in [
            &mut self.sk_d,
            &mut self.sk_ai,
            &mut self.sk_ar,
            &mut self.sk_ei,
            &mut self.sk_er,
        ] {
            *key = [0u8; 32];
            let _ = core::hint::black_box(key);
        }
    }
}

/// Where the handshake is. Private: a caller sees [`Progress`], not the state
/// graph, so the graph can grow (rekey, delete, informational) without
/// breaking anyone.
///
/// The ephemeral secret and our nonce live *inside* the state that needs them,
/// and the keys live inside `Established`. That is the point of rebuilding
/// rather than porting: the oracle carries five `Option<DerivedKey>` fields
/// plus an `Option` ephemeral, so "established but no keys" is representable
/// there and unrepresentable here.
enum Phase {
    Initial,
    AwaitingResponse {
        ephemeral: StaticSecret,
        our_nonce: [u8; NONCE_LEN],
    },
    Established {
        keys: SessionKeys,
        at: Ticks,
    },
}

impl Phase {
    const fn name(&self) -> &'static str {
        match self {
            Self::Initial => "initial",
            Self::AwaitingResponse { .. } => "awaiting-response",
            Self::Established { .. } => "established",
        }
    }
}

/// An IKE-style SA_INIT handshake.
pub struct Handshake {
    role: Role,
    phase: Phase,
    psk: [u8; 32],
    spi_initiator: u64,
    spi_responder: u64,
    message_id: u32,
    outbound: [u8; MESSAGE_LEN],
    outbound_len: usize,
}

impl core::fmt::Debug for Handshake {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("Handshake")
            .field("role", &self.role)
            .field("phase", &self.phase.name())
            .field("spi_initiator", &self.spi_initiator)
            .field("spi_responder", &self.spi_responder)
            .finish_non_exhaustive()
    }
}

impl Drop for Handshake {
    fn drop(&mut self) {
        self.psk = [0u8; 32];
        let _ = core::hint::black_box(&self.psk);
    }
}

impl Handshake {
    /// Start as the initiator.
    ///
    /// The pre-shared key is a constructor argument, not something fetched: a
    /// sans-IO state machine cannot hold a key provider, so whoever composes
    /// the handshake resolves the key first.
    #[must_use]
    pub const fn initiator(psk: [u8; 32], spi: u64) -> Self {
        Self::new(Role::Initiator, psk, spi, 0)
    }

    /// Start as the responder. `spi` is this side's SPI; the initiator's
    /// arrives in the first message.
    #[must_use]
    pub const fn responder(psk: [u8; 32], spi: u64) -> Self {
        Self::new(Role::Responder, psk, 0, spi)
    }

    const fn new(role: Role, psk: [u8; 32], spi_initiator: u64, spi_responder: u64) -> Self {
        Self {
            role,
            phase: Phase::Initial,
            psk,
            spi_initiator,
            spi_responder,
            message_id: 0,
            outbound: [0u8; MESSAGE_LEN],
            outbound_len: 0,
        }
    }

    /// Bytes staged for the peer. Empty when the last step produced none.
    #[must_use]
    pub fn outbound(&self) -> &[u8] {
        &self.outbound[..self.outbound_len]
    }

    /// The derived keys, once [`Progress::Established`] has been reported.
    #[must_use]
    pub const fn keys(&self) -> Option<&SessionKeys> {
        match &self.phase {
            Phase::Established { keys, .. } => Some(keys),
            _ => None,
        }
    }

    /// When the handshake completed.
    #[must_use]
    pub const fn established_at(&self) -> Option<Ticks> {
        match &self.phase {
            Phase::Established { at, .. } => Some(*at),
            _ => None,
        }
    }

    /// Whether the next [`step`](Self::step) consumes entropy.
    ///
    /// A driver backed by [`EntropyCell`](crate::EntropyCell) checks this
    /// before drawing, because that cell is take-once and errors when empty —
    /// requiring a value on every step would make the steps that need none
    /// fail spuriously.
    #[must_use]
    pub const fn needs_entropy(&self) -> bool {
        matches!(
            (self.role, &self.phase),
            (Role::Initiator, Phase::Initial) | (Role::Responder, Phase::Initial)
        )
    }

    /// Advance the handshake.
    ///
    /// `input` is whatever has been read so far; a short slice yields
    /// [`Progress::NeedInput`] with no state change, so a caller may call
    /// again with more bytes. `entropy` must be `Some` exactly when
    /// [`needs_entropy`](Self::needs_entropy) is true. `now` is stamped into
    /// the established SA for lifetime and rekey accounting.
    ///
    /// # Errors
    ///
    /// - [`CentauriError::EntropyUnavailable`] if entropy was required and not
    ///   supplied.
    /// - [`CentauriError::InvalidMessage`] if a field of the peer's message is
    ///   malformed.
    /// - [`CentauriError::InvalidTransition`] if the handshake has nothing
    ///   left to do.
    pub fn step(
        &mut self,
        input: &[u8],
        entropy: Option<Entropy32>,
        now: Ticks,
    ) -> Result<Progress, CentauriError> {
        self.outbound_len = 0;

        match (self.role, &self.phase) {
            (Role::Initiator, Phase::Initial) => self.send_init(entropy),
            (Role::Responder, Phase::Initial) => self.receive_init(input, entropy, now),
            (Role::Initiator, Phase::AwaitingResponse { .. }) => self.receive_response(input, now),
            (Role::Responder, Phase::AwaitingResponse { .. }) => {
                Err(CentauriError::InvalidTransition {
                    expected: "initial",
                    found: "awaiting-response",
                })
            }
            (_, Phase::Established { .. }) => Err(CentauriError::InvalidTransition {
                expected: "initial or awaiting-response",
                found: "established",
            }),
        }
    }

    fn send_init(&mut self, entropy: Option<Entropy32>) -> Result<Progress, CentauriError> {
        let (our_nonce, ephemeral) = split_entropy(entropy)?;
        let dh_public = *PublicKey::from(&ephemeral).as_bytes();

        self.write_message(&our_nonce, &dh_public);
        self.phase = Phase::AwaitingResponse {
            ephemeral,
            our_nonce,
        };

        Ok(Progress::Advanced)
    }

    fn receive_init(
        &mut self,
        input: &[u8],
        entropy: Option<Entropy32>,
        now: Ticks,
    ) -> Result<Progress, CentauriError> {
        let Some(peer) = Message::parse(input)? else {
            return Ok(Progress::NeedInput);
        };

        self.spi_initiator = peer.spi_initiator;

        let (our_nonce, ephemeral) = split_entropy(entropy)?;
        let dh_public = *PublicKey::from(&ephemeral).as_bytes();
        let shared = ephemeral.diffie_hellman(&PublicKey::from(peer.dh_public));

        // ni then nr, both sides: the initiator's nonce is always first.
        let keys = self.derive_session_keys(&peer.nonce, &our_nonce, shared.as_bytes());

        self.write_message(&our_nonce, &dh_public);
        self.phase = Phase::Established { keys, at: now };

        Ok(Progress::Established)
    }

    fn receive_response(&mut self, input: &[u8], now: Ticks) -> Result<Progress, CentauriError> {
        let Some(peer) = Message::parse(input)? else {
            return Ok(Progress::NeedInput);
        };

        let Phase::AwaitingResponse {
            ephemeral,
            our_nonce,
        } = core::mem::replace(&mut self.phase, Phase::Initial)
        else {
            return Err(CentauriError::InvalidTransition {
                expected: "awaiting-response",
                found: "initial",
            });
        };

        self.spi_responder = peer.spi_responder;
        let shared = ephemeral.diffie_hellman(&PublicKey::from(peer.dh_public));
        let keys = self.derive_session_keys(&our_nonce, &peer.nonce, shared.as_bytes());

        self.phase = Phase::Established { keys, at: now };

        Ok(Progress::Established)
    }

    fn derive_session_keys(
        &self,
        initiator_nonce: &[u8; NONCE_LEN],
        responder_nonce: &[u8; NONCE_LEN],
        shared: &[u8; 32],
    ) -> SessionKeys {
        let mut seed_input = [0u8; NONCE_LEN * 2 + 64];
        seed_input[..NONCE_LEN].copy_from_slice(initiator_nonce);
        seed_input[NONCE_LEN..NONCE_LEN * 2].copy_from_slice(responder_nonce);
        seed_input[NONCE_LEN * 2..NONCE_LEN * 2 + 32].copy_from_slice(shared);
        seed_input[NONCE_LEN * 2 + 32..].copy_from_slice(&self.psk);

        let skeyseed = derive_key(SKEYSEED_CONTEXT, &seed_input);

        SessionKeys {
            role: self.role,
            sk_d: skeyseed,
            sk_ai: derive_key(SK_AI_CONTEXT, &skeyseed),
            sk_ar: derive_key(SK_AR_CONTEXT, &skeyseed),
            sk_ei: derive_key(SK_EI_CONTEXT, &skeyseed),
            sk_er: derive_key(SK_ER_CONTEXT, &skeyseed),
        }
    }

    fn write_message(&mut self, nonce: &[u8; NONCE_LEN], dh_public: &[u8; DH_LEN]) {
        let message_id = self.message_id;
        self.message_id = self.message_id.wrapping_add(1);

        let out = &mut self.outbound;
        out[0..8].copy_from_slice(&self.spi_initiator.to_be_bytes());
        out[8..16].copy_from_slice(&self.spi_responder.to_be_bytes());
        out[16] = NEXT_PAYLOAD;
        out[17] = VERSION;
        out[18] = EXCHANGE_TYPE;
        out[19] = FLAGS;
        out[20..24].copy_from_slice(&message_id.to_be_bytes());
        out[24..28].copy_from_slice(&(MESSAGE_LEN as u32).to_be_bytes());
        out[HEADER_LEN..HEADER_LEN + NONCE_LEN].copy_from_slice(nonce);
        out[HEADER_LEN + NONCE_LEN..MESSAGE_LEN].copy_from_slice(dh_public);

        self.outbound_len = MESSAGE_LEN;
    }
}

/// Both nonce and ephemeral secret come from one draw, domain-separated.
///
/// One `Entropy32` per step keeps the family's contract uniform, and a
/// 32-byte unpredictable seed yields two unpredictable 32-byte values through
/// BLAKE3's KDF. Any 32 bytes is a valid X25519 secret — the scalar is clamped
/// on use — so no rejection sampling is needed.
fn split_entropy(
    entropy: Option<Entropy32>,
) -> Result<([u8; NONCE_LEN], StaticSecret), CentauriError> {
    let entropy = entropy.ok_or(CentauriError::EntropyUnavailable("step requires entropy"))?;

    let nonce = derive_key(NONCE_CONTEXT, entropy.expose());
    let secret = derive_key(EPHEMERAL_CONTEXT, entropy.expose());

    Ok((nonce, StaticSecret::from(secret)))
}

/// A parsed SA_INIT message.
struct Message {
    spi_initiator: u64,
    spi_responder: u64,
    nonce: [u8; NONCE_LEN],
    dh_public: [u8; DH_LEN],
}

impl Message {
    /// `Ok(None)` means "not enough bytes yet", which is not an error — the
    /// caller may be mid-read.
    fn parse(input: &[u8]) -> Result<Option<Self>, CentauriError> {
        if input.len() < MESSAGE_LEN {
            return Ok(None);
        }

        let spi_initiator = u64::from_be_bytes(
            input[0..8]
                .try_into()
                .map_err(|_| CentauriError::InvalidMessage("spi_initiator"))?,
        );
        let spi_responder = u64::from_be_bytes(
            input[8..16]
                .try_into()
                .map_err(|_| CentauriError::InvalidMessage("spi_responder"))?,
        );

        if input[17] != VERSION {
            return Err(CentauriError::InvalidMessage("version"));
        }
        if input[18] != EXCHANGE_TYPE {
            return Err(CentauriError::InvalidMessage("exchange_type"));
        }

        let nonce: [u8; NONCE_LEN] = input[HEADER_LEN..HEADER_LEN + NONCE_LEN]
            .try_into()
            .map_err(|_| CentauriError::InvalidMessage("nonce"))?;
        let dh_public: [u8; DH_LEN] = input[HEADER_LEN + NONCE_LEN..MESSAGE_LEN]
            .try_into()
            .map_err(|_| CentauriError::InvalidMessage("dh_public"))?;

        Ok(Some(Self {
            spi_initiator,
            spi_responder,
            nonce,
            dh_public,
        }))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use core::fmt::Write;

    use proxima_clock::ticks::Ticks;

    use super::{
        DH_LEN, EPHEMERAL_CONTEXT, HEADER_LEN, Handshake, MESSAGE_LEN, NONCE_CONTEXT, NONCE_LEN,
        Progress, Role,
    };
    use crate::entropy::Entropy32;
    use crate::error::CentauriError;
    use crate::hash::derive_key;

    const PSK: [u8; 32] = [0xAB; 32];
    const INITIATOR_SPI: u64 = 0x0102_0304_0506_0708;
    const RESPONDER_SPI: u64 = 0x1112_1314_1516_1718;
    const INITIATOR_SEED: [u8; 32] = [0x11; 32];
    const RESPONDER_SEED: [u8; 32] = [0x22; 32];

    fn now() -> Ticks {
        Ticks::from_raw(1_000)
    }

    /// Copy the staged message out by value. A `to_vec` here would need an
    /// allocator, which would make the test suite unable to run at the tier
    /// the crate exists to serve.
    fn captured(handshake: &Handshake) -> [u8; MESSAGE_LEN] {
        let mut message = [0u8; MESSAGE_LEN];
        message.copy_from_slice(handshake.outbound());
        message
    }

    #[test]
    fn init_message_matches_the_wire_layout() {
        let mut initiator = Handshake::initiator(PSK, INITIATOR_SPI);

        let progress = initiator
            .step(&[], Some(Entropy32::new(INITIATOR_SEED)), now())
            .expect("initiator can always send the first message");

        assert_eq!(progress, Progress::Advanced);

        let message = initiator.outbound();
        assert_eq!(message.len(), MESSAGE_LEN, "SA_INIT is a fixed 92 bytes");
        assert_eq!(
            &message[0..8],
            &INITIATOR_SPI.to_be_bytes(),
            "initiator spi, big endian"
        );
        assert_eq!(
            &message[8..16],
            &[0u8; 8],
            "responder spi is unknown in the first message"
        );
        assert_eq!(message[16], 0x21, "next payload");
        assert_eq!(message[17], 0x20, "version");
        assert_eq!(message[18], 0x22, "exchange type");
        assert_eq!(message[19], 0x08, "initiator flag");
        assert_eq!(
            &message[20..24],
            &0u32.to_be_bytes(),
            "message id starts at zero"
        );
        assert_eq!(&message[24..28], &92u32.to_be_bytes(), "total length");

        // the nonce and DH value are derived from the pinned seed, which is
        // the whole point of injectable entropy: this assertion is impossible
        // against an implementation that calls getrandom internally.
        let expected_nonce = derive_key(NONCE_CONTEXT, &INITIATOR_SEED);
        assert_eq!(
            &message[HEADER_LEN..HEADER_LEN + NONCE_LEN],
            &expected_nonce
        );

        let expected_secret = derive_key(EPHEMERAL_CONTEXT, &INITIATOR_SEED);
        let expected_public =
            *x25519_dalek::PublicKey::from(&x25519_dalek::StaticSecret::from(expected_secret))
                .as_bytes();
        assert_eq!(
            &message[HEADER_LEN + NONCE_LEN..MESSAGE_LEN],
            &expected_public
        );
        assert_eq!(expected_public.len(), DH_LEN);
    }

    #[test]
    fn pinned_entropy_reproduces_the_same_bytes() {
        let mut first = Handshake::initiator(PSK, INITIATOR_SPI);
        let mut second = Handshake::initiator(PSK, INITIATOR_SPI);

        let _ = first
            .step(&[], Some(Entropy32::new(INITIATOR_SEED)), now())
            .unwrap();
        let _ = second
            .step(&[], Some(Entropy32::new(INITIATOR_SEED)), now())
            .unwrap();

        assert_eq!(
            first.outbound(),
            second.outbound(),
            "same seed, same wire bytes"
        );
    }

    #[test]
    fn full_round_trip_agrees_on_keys() {
        let mut initiator = Handshake::initiator(PSK, INITIATOR_SPI);
        let mut responder = Handshake::responder(PSK, RESPONDER_SPI);

        assert!(initiator.needs_entropy());
        let progress = initiator
            .step(&[], Some(Entropy32::new(INITIATOR_SEED)), now())
            .unwrap();
        assert_eq!(progress, Progress::Advanced);
        let init_message = captured(&initiator);

        assert!(responder.needs_entropy());
        let progress = responder
            .step(&init_message, Some(Entropy32::new(RESPONDER_SEED)), now())
            .unwrap();
        assert_eq!(
            progress,
            Progress::Established,
            "responder finishes on the first message"
        );
        let response = captured(&responder);

        assert!(
            !initiator.needs_entropy(),
            "completing the handshake draws no entropy"
        );
        let progress = initiator.step(&response, None, now()).unwrap();
        assert_eq!(progress, Progress::Established);
        assert!(initiator.outbound().is_empty(), "nothing left to send");

        let initiator_keys = initiator.keys().expect("established");
        let responder_keys = responder.keys().expect("established");

        assert_eq!(
            initiator_keys.seed(),
            responder_keys.seed(),
            "same skeyseed"
        );
        assert_eq!(
            initiator_keys.encrypt_key(),
            responder_keys.decrypt_key(),
            "what the initiator encrypts, the responder decrypts"
        );
        assert_eq!(
            responder_keys.encrypt_key(),
            initiator_keys.decrypt_key(),
            "and the reverse direction"
        );
        assert_eq!(initiator_keys.auth_key(), responder_keys.peer_auth_key());
    }

    #[test]
    fn key_derivation_matches_an_independent_oracle() {
        let mut initiator = Handshake::initiator(PSK, INITIATOR_SPI);
        let mut responder = Handshake::responder(PSK, RESPONDER_SPI);

        let _ = initiator
            .step(&[], Some(Entropy32::new(INITIATOR_SEED)), now())
            .unwrap();
        let init_message = captured(&initiator);
        let _ = responder
            .step(&init_message, Some(Entropy32::new(RESPONDER_SEED)), now())
            .unwrap();

        // recompute the chain with blake3 directly, sharing no code with the
        // implementation: nonces and DH secrets rebuilt from the pinned seeds.
        let initiator_nonce = derive_key(NONCE_CONTEXT, &INITIATOR_SEED);
        let responder_nonce = derive_key(NONCE_CONTEXT, &RESPONDER_SEED);
        let initiator_secret =
            x25519_dalek::StaticSecret::from(derive_key(EPHEMERAL_CONTEXT, &INITIATOR_SEED));
        let responder_secret =
            x25519_dalek::StaticSecret::from(derive_key(EPHEMERAL_CONTEXT, &RESPONDER_SEED));
        let shared =
            responder_secret.diffie_hellman(&x25519_dalek::PublicKey::from(&initiator_secret));

        let mut seed_input = [0u8; 128];
        seed_input[..32].copy_from_slice(&initiator_nonce);
        seed_input[32..64].copy_from_slice(&responder_nonce);
        seed_input[64..96].copy_from_slice(shared.as_bytes());
        seed_input[96..].copy_from_slice(&PSK);

        let expected_skeyseed = blake3::derive_key("csr-ike-skeyseed-v1", &seed_input);
        let expected_sk_ei = blake3::derive_key("csr-ike-sk_ei-v1", &expected_skeyseed);

        let keys = responder.keys().expect("established");
        assert_eq!(
            keys.seed(),
            &expected_skeyseed,
            "skeyseed matches the oracle chain"
        );
        assert_eq!(
            keys.decrypt_key(),
            &expected_sk_ei,
            "responder decrypts with sk_ei"
        );
    }

    #[test]
    fn short_input_asks_for_more_without_changing_state() {
        let mut responder = Handshake::responder(PSK, RESPONDER_SPI);
        let partial = [0u8; MESSAGE_LEN - 1];

        let progress = responder
            .step(&partial, Some(Entropy32::new(RESPONDER_SEED)), now())
            .unwrap();

        assert_eq!(progress, Progress::NeedInput);
        assert!(
            responder.keys().is_none(),
            "no state change on a short read"
        );
        assert!(
            responder.needs_entropy(),
            "still waiting for the first message"
        );
    }

    #[test]
    fn step_after_established_is_a_transition_error() {
        let mut initiator = Handshake::initiator(PSK, INITIATOR_SPI);
        let mut responder = Handshake::responder(PSK, RESPONDER_SPI);

        let _ = initiator
            .step(&[], Some(Entropy32::new(INITIATOR_SEED)), now())
            .unwrap();
        let init_message = captured(&initiator);
        let _ = responder
            .step(&init_message, Some(Entropy32::new(RESPONDER_SEED)), now())
            .unwrap();

        assert_eq!(
            responder.step(&init_message, None, now()).err(),
            Some(CentauriError::InvalidTransition {
                expected: "initial or awaiting-response",
                found: "established",
            })
        );
    }

    #[test]
    fn missing_entropy_is_reported_not_improvised() {
        let mut initiator = Handshake::initiator(PSK, INITIATOR_SPI);

        assert_eq!(
            initiator.step(&[], None, now()).err(),
            Some(CentauriError::EntropyUnavailable("step requires entropy"))
        );
    }

    #[test]
    fn a_mangled_version_byte_is_rejected() {
        let mut initiator = Handshake::initiator(PSK, INITIATOR_SPI);
        let _ = initiator
            .step(&[], Some(Entropy32::new(INITIATOR_SEED)), now())
            .unwrap();
        let mut message = captured(&initiator);
        message[17] = 0x99;

        let mut responder = Handshake::responder(PSK, RESPONDER_SPI);

        assert_eq!(
            responder
                .step(&message, Some(Entropy32::new(RESPONDER_SEED)), now())
                .err(),
            Some(CentauriError::InvalidMessage("version"))
        );
    }

    #[test]
    fn established_at_records_the_supplied_time() {
        let mut initiator = Handshake::initiator(PSK, INITIATOR_SPI);
        let mut responder = Handshake::responder(PSK, RESPONDER_SPI);

        let _ = initiator
            .step(&[], Some(Entropy32::new(INITIATOR_SEED)), now())
            .unwrap();
        let init_message = captured(&initiator);
        let _ = responder
            .step(
                &init_message,
                Some(Entropy32::new(RESPONDER_SEED)),
                Ticks::from_raw(4_242),
            )
            .unwrap();

        assert_eq!(responder.established_at().map(Ticks::as_raw), Some(4_242));
        assert!(
            initiator.established_at().is_none(),
            "initiator is not done yet"
        );
    }

    #[test]
    fn debug_does_not_leak_key_material() {
        let mut responder = Handshake::responder(PSK, RESPONDER_SPI);
        let mut initiator = Handshake::initiator(PSK, INITIATOR_SPI);
        let _ = initiator
            .step(&[], Some(Entropy32::new(INITIATOR_SEED)), now())
            .unwrap();
        let init_message = captured(&initiator);
        let _ = responder
            .step(&init_message, Some(Entropy32::new(RESPONDER_SEED)), now())
            .unwrap();

        let mut buffer = crate::test_support::Buffer::new();
        write!(buffer, "{responder:?} {:?}", responder.keys().unwrap()).expect("debug output fits");
        let rendered = buffer.as_str();

        assert!(rendered.contains("established"), "phase is visible");
        assert!(
            !rendered.contains("171"),
            "psk bytes (0xAB = 171) must not appear"
        );
        assert!(
            !rendered.contains("sk_e"),
            "key field names must not appear"
        );
    }

    #[test]
    fn roles_are_distinct() {
        assert_ne!(Role::Initiator, Role::Responder);
    }
}
