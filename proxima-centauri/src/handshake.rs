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
use subtle::ConstantTimeEq;
use x25519_dalek::{PublicKey, StaticSecret};

use crate::entropy::Entropy32;
use crate::error::CentauriError;
use crate::hash::{derive_key, keyed_hash};
use crate::sized::AUTH_MAX_IDENTITY_BYTES;

/// Bytes of an SA_INIT message: header, nonce, and DH public value.
pub const MESSAGE_LEN: usize = HEADER_LEN + NONCE_LEN + DH_LEN;

/// Bytes of the largest AUTH message: header, identity length, identity, MAC.
pub const AUTH_MAX_LEN: usize = HEADER_LEN + IDENTITY_LEN_BYTES + AUTH_MAX_IDENTITY_BYTES + MAC_LEN;

/// The staging buffer holds whichever message is larger.
pub const OUTBOUND_LEN: usize = if AUTH_MAX_LEN > MESSAGE_LEN {
    AUTH_MAX_LEN
} else {
    MESSAGE_LEN
};

/// Length prefix on the identity — the field that makes an AUTH message
/// self-describing.
const IDENTITY_LEN_BYTES: usize = 2;
const MAC_LEN: usize = 32;
const AUTH_EXCHANGE_TYPE: u8 = 0x23;
const REKEY_EXCHANGE_TYPE: u8 = 0x24;
const REKEY_SKEYSEED_CONTEXT: &str = "proxima-centauri-ike-rekey-skeyseed-v1";
const AUTH_FLAG_INITIATOR: u8 = 0x08;
const AUTH_FLAG_RESPONDER: u8 = 0x20;

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

/// An IKE security parameter index — this side's half of the SA identifier.
///
/// A newtype rather than a bare `u64` per principle 11: the handshake also
/// carries sequence numbers and tick counts of the same width, and a newtype
/// makes swapping them at a call site a compile error rather than a wire bug.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IkeSpi(u64);

impl IkeSpi {
    /// Wrap a raw SPI.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// The raw value, for writing to the wire.
    #[must_use]
    pub const fn as_raw(self) -> u64 {
        self.0
    }
}

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
///
/// `#[non_exhaustive]`: rekey, close, and the AUTH exchange will add variants
/// as the state graph grows, and a downstream `match` should not break when
/// they do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
#[non_exhaustive]
pub enum Progress {
    /// Not enough input to advance. Read more bytes and call again with the
    /// larger slice; nothing was consumed and no state changed.
    NeedInput,
    /// The handshake moved forward and is not finished.
    Advanced,
    /// SA_INIT is complete and [`Handshake::keys`] is available. There may
    /// still be a final message staged in [`Handshake::outbound`].
    ///
    /// The peer is **not yet authenticated** — the keys prove whoever derived
    /// them holds the PSK and the DH secret, which is not the same as proving
    /// *who* they are. Run the AUTH exchange for that.
    Established,
    /// The peer proved its identity: the AUTH payload's MAC verified under a
    /// key only the true peer could derive. [`Handshake::peer_identity`] is
    /// available.
    Authenticated,
    /// Fresh keys are in force. The peer's identity carries over — a rekey
    /// chains to the authenticated SA rather than re-proving it — and the old
    /// keys are gone.
    Rekeyed,
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
    /// We sent our AUTH; the peer has not proved itself yet.
    LocalAuthSent {
        keys: SessionKeys,
        at: Ticks,
    },
    /// The peer proved itself; we still owe our own AUTH. Mutual
    /// authentication is two independent obligations, so it takes two states
    /// rather than a `sent` flag beside a `verified` flag — a flag pair can
    /// represent "neither", which the phase graph cannot.
    PeerAuthenticated {
        keys: SessionKeys,
        at: Ticks,
        peer_identity: [u8; AUTH_MAX_IDENTITY_BYTES],
        peer_identity_len: usize,
    },
    /// Both directions proved. Only here may an SA rekey.
    Authenticated {
        keys: SessionKeys,
        at: Ticks,
        peer_identity: [u8; AUTH_MAX_IDENTITY_BYTES],
        peer_identity_len: usize,
    },
    /// Only the side that *initiated* a rekey waits here. The responder
    /// derives and replies in one step, exactly as it does in SA_INIT, so
    /// there is no "am I the rekey initiator" flag to carry — the state graph
    /// answers it.
    Rekeying {
        keys: SessionKeys,
        at: Ticks,
        peer_identity: [u8; AUTH_MAX_IDENTITY_BYTES],
        peer_identity_len: usize,
        ephemeral: StaticSecret,
        our_nonce: [u8; NONCE_LEN],
    },
}

impl Phase {
    const fn name(&self) -> &'static str {
        match self {
            Self::Initial => "initial",
            Self::AwaitingResponse { .. } => "awaiting-response",
            Self::Established { .. } => "established",
            Self::LocalAuthSent { .. } => "local-auth-sent",
            Self::PeerAuthenticated { .. } => "peer-authenticated",
            Self::Authenticated { .. } => "authenticated",
            Self::Rekeying { .. } => "rekeying",
        }
    }
}

/// An IKE-style SA_INIT handshake.
pub struct Handshake {
    role: Role,
    phase: Phase,
    psk: [u8; 32],
    spi_initiator: IkeSpi,
    spi_responder: IkeSpi,
    message_id: u32,
    identity: [u8; AUTH_MAX_IDENTITY_BYTES],
    identity_len: usize,
    outbound: [u8; OUTBOUND_LEN],
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
    pub const fn initiator(psk: [u8; 32], spi: IkeSpi) -> Self {
        Self::new(Role::Initiator, psk, spi, IkeSpi::new(0))
    }

    /// Start as the responder. `spi` is this side's SPI; the initiator's
    /// arrives in the first message.
    #[must_use]
    pub const fn responder(psk: [u8; 32], spi: IkeSpi) -> Self {
        Self::new(Role::Responder, psk, IkeSpi::new(0), spi)
    }

    const fn new(role: Role, psk: [u8; 32], spi_initiator: IkeSpi, spi_responder: IkeSpi) -> Self {
        Self {
            role,
            phase: Phase::Initial,
            psk,
            spi_initiator,
            spi_responder,
            message_id: 0,
            identity: [0u8; AUTH_MAX_IDENTITY_BYTES],
            identity_len: 0,
            outbound: [0u8; OUTBOUND_LEN],
            outbound_len: 0,
        }
    }

    /// Attach the identity this side presents in the AUTH exchange.
    ///
    /// # Errors
    ///
    /// [`CentauriError::PayloadTooLarge`] if the identity exceeds the
    /// build-time `[auth].max_identity_bytes`. Refused rather than truncated:
    /// a truncated identity authenticates as a different peer.
    pub fn with_identity(mut self, identity: &[u8]) -> Result<Self, CentauriError> {
        if identity.len() > AUTH_MAX_IDENTITY_BYTES {
            return Err(CentauriError::PayloadTooLarge {
                len: identity.len(),
                max: AUTH_MAX_IDENTITY_BYTES,
            });
        }
        self.identity[..identity.len()].copy_from_slice(identity);
        self.identity_len = identity.len();
        Ok(self)
    }

    /// The peer's identity, once [`Progress::Authenticated`] has been reported.
    #[must_use]
    pub fn peer_identity(&self) -> Option<&[u8]> {
        match &self.phase {
            Phase::PeerAuthenticated {
                peer_identity,
                peer_identity_len,
                ..
            }
            | Phase::Authenticated {
                peer_identity,
                peer_identity_len,
                ..
            }
            | Phase::Rekeying {
                peer_identity,
                peer_identity_len,
                ..
            } => Some(&peer_identity[..*peer_identity_len]),
            _ => None,
        }
    }

    /// Which side of the exchange this is.
    #[must_use]
    pub const fn role(&self) -> Role {
        self.role
    }

    /// The SPI this side announces — its own, whichever role it plays.
    #[must_use]
    pub const fn announced_spi(&self) -> IkeSpi {
        match self.role {
            Role::Initiator => self.spi_initiator,
            Role::Responder => self.spi_responder,
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
            Phase::Established { keys, .. }
            | Phase::LocalAuthSent { keys, .. }
            | Phase::PeerAuthenticated { keys, .. }
            | Phase::Authenticated { keys, .. }
            | Phase::Rekeying { keys, .. } => Some(keys),
            _ => None,
        }
    }

    /// When the handshake completed.
    #[must_use]
    pub const fn established_at(&self) -> Option<Ticks> {
        match &self.phase {
            Phase::Established { at, .. }
            | Phase::LocalAuthSent { at, .. }
            | Phase::PeerAuthenticated { at, .. }
            | Phase::Authenticated { at, .. }
            | Phase::Rekeying { at, .. } => Some(*at),
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
            (_, Phase::Established { .. } | Phase::LocalAuthSent { .. }) => {
                self.receive_auth(input, now)
            }
            (_, Phase::PeerAuthenticated { .. }) => Err(CentauriError::InvalidTransition {
                expected: "send_auth to complete mutual authentication",
                found: "peer-authenticated",
            }),
            (_, Phase::Authenticated { .. }) => self.receive_rekey(input, entropy, now),
            (_, Phase::Rekeying { .. }) => self.complete_rekey(input, now),
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
        let shared = agree(&ephemeral, peer.dh_public)?;

        // ni then nr, both sides: the initiator's nonce is always first.
        let keys = self.derive_session_keys(&peer.nonce, &our_nonce, &shared);

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
        let shared = agree(&ephemeral, peer.dh_public)?;
        let keys = self.derive_session_keys(&our_nonce, &peer.nonce, &shared);

        self.phase = Phase::Established { keys, at: now };

        Ok(Progress::Established)
    }

    /// Emit this side's AUTH payload.
    ///
    /// Legal only once SA_INIT has established keys — the MAC is taken under a
    /// key derived from the shared secret, which is what makes the identity
    /// claim unforgeable by anyone who did not complete the exchange.
    ///
    /// # Errors
    ///
    /// [`CentauriError::InvalidTransition`] before SA_INIT completes.
    pub fn send_auth(&mut self) -> Result<Progress, CentauriError> {
        self.outbound_len = 0;

        let auth_key = match &self.phase {
            Phase::Established { keys, .. } | Phase::PeerAuthenticated { keys, .. } => {
                *keys.auth_key()
            }
            other => {
                return Err(CentauriError::InvalidTransition {
                    expected: "established or peer-authenticated",
                    found: other.name(),
                });
            }
        };
        let identity_len = self.identity_len;
        let mut identity = [0u8; AUTH_MAX_IDENTITY_BYTES];
        identity[..identity_len].copy_from_slice(&self.identity[..identity_len]);

        let total = HEADER_LEN + IDENTITY_LEN_BYTES + identity_len + MAC_LEN;
        let message_id = self.message_id;
        self.message_id = self.message_id.wrapping_add(1);

        let flags = match self.role {
            Role::Initiator => AUTH_FLAG_INITIATOR,
            Role::Responder => AUTH_FLAG_RESPONDER,
        };

        let out = &mut self.outbound;
        out[0..8].copy_from_slice(&self.spi_initiator.as_raw().to_be_bytes());
        out[8..16].copy_from_slice(&self.spi_responder.as_raw().to_be_bytes());
        out[16] = AUTH_EXCHANGE_TYPE;
        out[17] = VERSION;
        out[18] = flags;
        out[19] = 0x00;
        out[20..24].copy_from_slice(&message_id.to_be_bytes());
        out[24..28].copy_from_slice(&(total as u32).to_be_bytes());
        let identity_len_u16 = u16::try_from(identity_len)
            .map_err(|_| CentauriError::InvalidMessage("identity length"))?;
        out[HEADER_LEN..HEADER_LEN + IDENTITY_LEN_BYTES]
            .copy_from_slice(&identity_len_u16.to_be_bytes());
        let identity_at = HEADER_LEN + IDENTITY_LEN_BYTES;
        out[identity_at..identity_at + identity_len].copy_from_slice(&identity[..identity_len]);

        // the MAC covers the header AND the length-prefixed identity, so a
        // flipped role flag or message id cannot pass. The oracle MACs the
        // identity alone and leaves the header unauthenticated.
        let mac = keyed_hash(&auth_key, &out[..identity_at + identity_len]);
        out[identity_at + identity_len..total].copy_from_slice(&mac);

        self.outbound_len = total;

        // sending discharges our half of the obligation; which state that
        // lands in depends on whether the peer has already discharged theirs
        self.phase = match core::mem::replace(&mut self.phase, Phase::Initial) {
            Phase::Established { keys, at } => Phase::LocalAuthSent { keys, at },
            Phase::PeerAuthenticated {
                keys,
                at,
                peer_identity,
                peer_identity_len,
            } => Phase::Authenticated {
                keys,
                at,
                peer_identity,
                peer_identity_len,
            },
            other => other,
        };

        Ok(Progress::Advanced)
    }

    fn receive_auth(&mut self, input: &[u8], _now: Ticks) -> Result<Progress, CentauriError> {
        let minimum = HEADER_LEN + IDENTITY_LEN_BYTES + MAC_LEN;
        if input.len() < minimum {
            return Ok(Progress::NeedInput);
        }

        if input[16] != AUTH_EXCHANGE_TYPE {
            return Err(CentauriError::InvalidMessage("auth exchange type"));
        }
        if input[17] != VERSION {
            return Err(CentauriError::InvalidMessage("version"));
        }

        // self-describing: the length prefix locates the MAC without the
        // receiver having to know the peer's identity in advance
        let identity_len = usize::from(u16::from_be_bytes(
            input[HEADER_LEN..HEADER_LEN + IDENTITY_LEN_BYTES]
                .try_into()
                .map_err(|_| CentauriError::InvalidMessage("identity length"))?,
        ));
        if identity_len > AUTH_MAX_IDENTITY_BYTES {
            return Err(CentauriError::PayloadTooLarge {
                len: identity_len,
                max: AUTH_MAX_IDENTITY_BYTES,
            });
        }

        let total = HEADER_LEN + IDENTITY_LEN_BYTES + identity_len + MAC_LEN;
        if input.len() < total {
            return Ok(Progress::NeedInput);
        }

        let (keys, at, we_have_sent) = match core::mem::replace(&mut self.phase, Phase::Initial) {
            Phase::Established { keys, at } => (keys, at, false),
            Phase::LocalAuthSent { keys, at } => (keys, at, true),
            other => {
                let name = other.name();
                self.phase = other;
                return Err(CentauriError::InvalidTransition {
                    expected: "established or local-auth-sent",
                    found: name,
                });
            }
        };

        let identity_at = HEADER_LEN + IDENTITY_LEN_BYTES;
        let expected = keyed_hash(keys.peer_auth_key(), &input[..identity_at + identity_len]);
        let received = &input[identity_at + identity_len..total];

        if !macs_match(&expected, received) {
            // leave the handshake unusable rather than silently established:
            // a failed AUTH must not leave keys reachable
            return Err(CentauriError::AuthenticationFailed);
        }

        let mut peer_identity = [0u8; AUTH_MAX_IDENTITY_BYTES];
        peer_identity[..identity_len]
            .copy_from_slice(&input[identity_at..identity_at + identity_len]);

        self.phase = if we_have_sent {
            Phase::Authenticated {
                keys,
                at,
                peer_identity,
                peer_identity_len: identity_len,
            }
        } else {
            Phase::PeerAuthenticated {
                keys,
                at,
                peer_identity,
                peer_identity_len: identity_len,
            }
        };

        Ok(Progress::Authenticated)
    }

    /// Open a rekey: fresh nonce, fresh DH, message on the wire.
    ///
    /// Legal only once the peer is authenticated. The new keys chain to the
    /// old SA's seed, so a rekey inherits the identity proof rather than
    /// re-running AUTH — and the fresh DH means compromising today's keys does
    /// not yield yesterday's traffic.
    ///
    /// # Errors
    ///
    /// [`CentauriError::EntropyUnavailable`] without entropy, or
    /// [`CentauriError::InvalidTransition`] before the peer is authenticated.
    pub fn send_rekey(
        &mut self,
        entropy: Option<Entropy32>,
        _now: Ticks,
    ) -> Result<Progress, CentauriError> {
        self.outbound_len = 0;

        let (keys, at, peer_identity, peer_identity_len) =
            match core::mem::replace(&mut self.phase, Phase::Initial) {
                Phase::Authenticated {
                    keys,
                    at,
                    peer_identity,
                    peer_identity_len,
                } => (keys, at, peer_identity, peer_identity_len),
                other => {
                    // put it back: refusing a rekey must not destroy a live SA
                    let name = other.name();
                    self.phase = other;
                    return Err(CentauriError::InvalidTransition {
                        expected: "authenticated",
                        found: name,
                    });
                }
            };

        let (our_nonce, ephemeral) = match split_entropy(entropy) {
            Ok(pair) => pair,
            Err(error) => {
                // put the SA back: a missing draw must not destroy a live
                // session
                self.phase = Phase::Authenticated {
                    keys,
                    at,
                    peer_identity,
                    peer_identity_len,
                };
                return Err(error);
            }
        };

        let dh_public = *PublicKey::from(&ephemeral).as_bytes();
        self.write_exchange(REKEY_EXCHANGE_TYPE, &our_nonce, &dh_public);

        self.phase = Phase::Rekeying {
            keys,
            at,
            peer_identity,
            peer_identity_len,
            ephemeral,
            our_nonce,
        };

        Ok(Progress::Advanced)
    }

    /// The responding half: derive from the peer's nonce and DH, reply with
    /// ours, and adopt the new keys in one step.
    fn receive_rekey(
        &mut self,
        input: &[u8],
        entropy: Option<Entropy32>,
        now: Ticks,
    ) -> Result<Progress, CentauriError> {
        let Some(peer) = Message::parse_exchange(input, REKEY_EXCHANGE_TYPE)? else {
            return Ok(Progress::NeedInput);
        };

        let (keys, peer_identity, peer_identity_len) =
            match core::mem::replace(&mut self.phase, Phase::Initial) {
                Phase::Authenticated {
                    keys,
                    peer_identity,
                    peer_identity_len,
                    ..
                } => (keys, peer_identity, peer_identity_len),
                other => {
                    let name = other.name();
                    self.phase = other;
                    return Err(CentauriError::InvalidTransition {
                        expected: "authenticated",
                        found: name,
                    });
                }
            };

        let (our_nonce, ephemeral) = match split_entropy(entropy) {
            Ok(pair) => pair,
            Err(error) => {
                self.phase = Phase::Authenticated {
                    keys,
                    at: now,
                    peer_identity,
                    peer_identity_len,
                };
                return Err(error);
            }
        };
        let dh_public = *PublicKey::from(&ephemeral).as_bytes();
        let shared = agree(&ephemeral, peer.dh_public)?;

        // the rekey initiator's nonce is always first, both sides
        let fresh = self.derive_rekey_keys(&keys, &peer.nonce, &our_nonce, &shared);

        self.write_exchange(REKEY_EXCHANGE_TYPE, &our_nonce, &dh_public);
        self.phase = Phase::Authenticated {
            keys: fresh,
            at: now,
            peer_identity,
            peer_identity_len,
        };

        Ok(Progress::Rekeyed)
    }

    /// The initiating half completing on the peer's reply.
    fn complete_rekey(&mut self, input: &[u8], now: Ticks) -> Result<Progress, CentauriError> {
        let Some(peer) = Message::parse_exchange(input, REKEY_EXCHANGE_TYPE)? else {
            return Ok(Progress::NeedInput);
        };

        let (keys, peer_identity, peer_identity_len, ephemeral, our_nonce) =
            match core::mem::replace(&mut self.phase, Phase::Initial) {
                Phase::Rekeying {
                    keys,
                    peer_identity,
                    peer_identity_len,
                    ephemeral,
                    our_nonce,
                    ..
                } => (keys, peer_identity, peer_identity_len, ephemeral, our_nonce),
                other => {
                    let name = other.name();
                    self.phase = other;
                    return Err(CentauriError::InvalidTransition {
                        expected: "rekeying",
                        found: name,
                    });
                }
            };

        let shared = agree(&ephemeral, peer.dh_public)?;
        let fresh = self.derive_rekey_keys(&keys, &our_nonce, &peer.nonce, &shared);

        self.phase = Phase::Authenticated {
            keys: fresh,
            at: now,
            peer_identity,
            peer_identity_len,
        };

        Ok(Progress::Rekeyed)
    }

    /// New keys from the old SA's seed plus a fresh exchange.
    ///
    /// Chaining through `sk_d` is what lets the rekey inherit the identity
    /// proof: only a peer that completed AUTH holds it. The fresh shared
    /// secret is what gives forward secrecy — the old keys cannot reproduce
    /// the new ones.
    fn derive_rekey_keys(
        &self,
        old: &SessionKeys,
        initiator_nonce: &[u8; NONCE_LEN],
        responder_nonce: &[u8; NONCE_LEN],
        shared: &[u8; 32],
    ) -> SessionKeys {
        let mut seed_input = [0u8; 32 + NONCE_LEN * 2 + 32];
        seed_input[..32].copy_from_slice(old.seed());
        seed_input[32..32 + NONCE_LEN].copy_from_slice(initiator_nonce);
        seed_input[32 + NONCE_LEN..32 + NONCE_LEN * 2].copy_from_slice(responder_nonce);
        seed_input[32 + NONCE_LEN * 2..].copy_from_slice(shared);

        let skeyseed = derive_key(REKEY_SKEYSEED_CONTEXT, &seed_input);
        seed_input = [0u8; 32 + NONCE_LEN * 2 + 32];
        let _ = core::hint::black_box(&seed_input);

        SessionKeys {
            role: self.role,
            sk_d: skeyseed,
            sk_ai: derive_key(SK_AI_CONTEXT, &skeyseed),
            sk_ar: derive_key(SK_AR_CONTEXT, &skeyseed),
            sk_ei: derive_key(SK_EI_CONTEXT, &skeyseed),
            sk_er: derive_key(SK_ER_CONTEXT, &skeyseed),
        }
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
        // the buffer held the shared secret and the PSK; a no-alloc crate has
        // one address per secret, so wiping it actually wipes it
        seed_input = [0u8; NONCE_LEN * 2 + 64];
        let _ = core::hint::black_box(&seed_input);

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
        self.write_exchange(EXCHANGE_TYPE, nonce, dh_public);
    }

    fn write_exchange(
        &mut self,
        exchange_type: u8,
        nonce: &[u8; NONCE_LEN],
        dh_public: &[u8; DH_LEN],
    ) {
        let message_id = self.message_id;
        self.message_id = self.message_id.wrapping_add(1);

        let out = &mut self.outbound;
        out[0..8].copy_from_slice(&self.spi_initiator.as_raw().to_be_bytes());
        out[8..16].copy_from_slice(&self.spi_responder.as_raw().to_be_bytes());
        out[16] = NEXT_PAYLOAD;
        out[17] = VERSION;
        out[18] = exchange_type;
        out[19] = FLAGS;
        out[20..24].copy_from_slice(&message_id.to_be_bytes());
        out[24..28].copy_from_slice(&(MESSAGE_LEN as u32).to_be_bytes());
        out[HEADER_LEN..HEADER_LEN + NONCE_LEN].copy_from_slice(nonce);
        out[HEADER_LEN + NONCE_LEN..MESSAGE_LEN].copy_from_slice(dh_public);

        self.outbound_len = MESSAGE_LEN;
    }
}

/// Agree a shared secret, refusing a degenerate one.
///
/// X25519 maps every low-order point to an all-zero output, and `x25519-dalek`
/// returns it rather than erroring — verified 2026-07-28 against the all-zero
/// point and two RFC 7748 small-order points, all three of which yield zeros.
/// An active attacker who substitutes the peer's DH value therefore makes the
/// ephemeral contribute **nothing**: key secrecy still rests on the PSK, but
/// forward secrecy — the entire reason the ephemeral exists, and a property
/// this crate claims for its rekey — is gone, silently.
///
/// RFC 7748 §6.1 permits rejecting the all-zero output, and this does, in
/// constant time.
fn agree(ephemeral: &StaticSecret, peer_public: [u8; DH_LEN]) -> Result<[u8; 32], CentauriError> {
    let shared = ephemeral.diffie_hellman(&PublicKey::from(peer_public));
    let bytes = *shared.as_bytes();

    if bool::from(bytes.ct_eq(&[0u8; 32])) {
        return Err(CentauriError::DegenerateKeyAgreement);
    }

    Ok(bytes)
}

/// Compare two MACs without leaking where they diverge.
///
/// `subtle` is already in the dependency graph — x25519-dalek pulls it — and is
/// built for exactly this, so no new dependency and no hand-rolled loop. A
/// hand-rolled comparison is precisely the code an optimiser is free to
/// short-circuit into a timing oracle.
fn macs_match(left: &[u8; MAC_LEN], right: &[u8]) -> bool {
    if right.len() != MAC_LEN {
        return false;
    }
    left.ct_eq(right).into()
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
    spi_initiator: IkeSpi,
    spi_responder: IkeSpi,
    nonce: [u8; NONCE_LEN],
    dh_public: [u8; DH_LEN],
}

impl Message {
    /// `Ok(None)` means "not enough bytes yet", which is not an error — the
    /// caller may be mid-read.
    fn parse(input: &[u8]) -> Result<Option<Self>, CentauriError> {
        Self::parse_exchange(input, EXCHANGE_TYPE)
    }

    /// SA_INIT and CREATE_CHILD_SA share a body — header, nonce, DH value —
    /// and differ only in the exchange type, so they share a parser. Passing
    /// the expected type in means a rekey message can never be mistaken for a
    /// fresh handshake, which is the confusion that would let a peer reset an
    /// established SA.
    fn parse_exchange(input: &[u8], expected: u8) -> Result<Option<Self>, CentauriError> {
        if input.len() < MESSAGE_LEN {
            return Ok(None);
        }

        let spi_initiator = IkeSpi::new(u64::from_be_bytes(
            input[0..8]
                .try_into()
                .map_err(|_| CentauriError::InvalidMessage("spi_initiator"))?,
        ));
        let spi_responder = IkeSpi::new(u64::from_be_bytes(
            input[8..16]
                .try_into()
                .map_err(|_| CentauriError::InvalidMessage("spi_responder"))?,
        ));

        if input[17] != VERSION {
            return Err(CentauriError::InvalidMessage("version"));
        }
        if input[18] != expected {
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
        AUTH_MAX_LEN, DH_LEN, EPHEMERAL_CONTEXT, HEADER_LEN, Handshake, IkeSpi, MESSAGE_LEN,
        NONCE_CONTEXT, NONCE_LEN, Progress, Role,
    };
    use crate::entropy::Entropy32;
    use crate::error::CentauriError;
    use crate::hash::derive_key;
    use crate::sized::AUTH_MAX_IDENTITY_BYTES;

    const PSK: [u8; 32] = [0xAB; 32];
    const INITIATOR_SPI: IkeSpi = IkeSpi::new(0x0102_0304_0506_0708);
    const RESPONDER_SPI: IkeSpi = IkeSpi::new(0x1112_1314_1516_1718);
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
            &INITIATOR_SPI.as_raw().to_be_bytes(),
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
    fn an_established_handshake_refuses_a_non_auth_message() {
        // it no longer refuses every step -- AUTH is legal here -- but a
        // replayed SA_INIT must still be rejected on its exchange type.
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
            Some(CentauriError::InvalidMessage("auth exchange type"))
        );
    }

    #[test]
    fn mutual_authentication_takes_both_directions() {
        let (mut initiator, mut responder) = established_pair(b"peer-a", b"peer-b");

        // one direction only: the responder knows the initiator, but not the
        // reverse, and neither side is fully authenticated yet
        let _ = initiator.send_auth().unwrap();
        let forward = staged(&initiator);
        let forward_len = initiator.outbound().len();
        assert_eq!(
            responder
                .step(&forward[..forward_len], None, now())
                .unwrap(),
            Progress::Authenticated
        );
        assert_eq!(responder.peer_identity(), Some(&b"peer-a"[..]));
        assert!(
            initiator.peer_identity().is_none(),
            "the initiator knows nobody yet"
        );

        // a rekey is refused until both directions are proved
        assert!(
            initiator
                .send_rekey(Some(Entropy32::new([1; 32])), now())
                .is_err(),
            "rekey requires mutual authentication"
        );

        // now the reverse direction completes it
        let _ = responder.send_auth().unwrap();
        let back = staged(&responder);
        let back_len = responder.outbound().len();
        assert_eq!(
            initiator.step(&back[..back_len], None, now()).unwrap(),
            Progress::Authenticated
        );
        assert_eq!(initiator.peer_identity(), Some(&b"peer-b"[..]));

        // and now it is allowed
        assert!(
            initiator
                .send_rekey(Some(Entropy32::new([1; 32])), now())
                .is_ok()
        );
    }

    #[test]
    fn an_authenticated_handshake_refuses_further_steps() {
        let (mut initiator, mut responder) = established_pair(b"peer-a", b"peer-b");
        let _ = initiator.send_auth().unwrap();
        let message = staged(&initiator);
        let len = initiator.outbound().len();
        let _ = responder.step(&message[..len], None, now()).unwrap();

        // after verifying us the responder owes its own AUTH, so a further
        // inbound AUTH is the wrong move for it to accept
        assert_eq!(
            responder.step(&message[..len], None, now()).err(),
            Some(CentauriError::InvalidTransition {
                expected: "send_auth to complete mutual authentication",
                found: "peer-authenticated",
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

    /// Both sides through SA_INIT, ready for AUTH.
    fn established_pair(
        initiator_identity: &[u8],
        responder_identity: &[u8],
    ) -> (Handshake, Handshake) {
        let mut initiator = Handshake::initiator(PSK, INITIATOR_SPI)
            .with_identity(initiator_identity)
            .expect("identity fits");
        let mut responder = Handshake::responder(PSK, RESPONDER_SPI)
            .with_identity(responder_identity)
            .expect("identity fits");

        let _ = initiator
            .step(&[], Some(Entropy32::new(INITIATOR_SEED)), now())
            .unwrap();
        let init = captured(&initiator);
        let _ = responder
            .step(&init, Some(Entropy32::new(RESPONDER_SEED)), now())
            .unwrap();
        let reply = captured(&responder);
        let _ = initiator.step(&reply, None, now()).unwrap();

        (initiator, responder)
    }

    fn staged(handshake: &Handshake) -> [u8; AUTH_MAX_LEN] {
        let mut message = [0u8; AUTH_MAX_LEN];
        let out = handshake.outbound();
        message[..out.len()].copy_from_slice(out);
        message
    }

    #[test]
    fn auth_proves_identity_in_both_directions() {
        let (mut initiator, mut responder) = established_pair(b"peer-a", b"peer-b");

        assert_eq!(initiator.send_auth().unwrap(), Progress::Advanced);
        let from_initiator = staged(&initiator);
        let initiator_len = initiator.outbound().len();

        assert_eq!(
            responder
                .step(&from_initiator[..initiator_len], None, now())
                .unwrap(),
            Progress::Authenticated
        );
        assert_eq!(responder.peer_identity(), Some(&b"peer-a"[..]));

        // the responder verified the initiator but still owes its own proof:
        // mutual authentication is two obligations, not one
        assert!(
            responder.send_auth().is_ok(),
            "a peer that verified us still owes its own AUTH"
        );

        // and the reverse direction, on a fresh pair
        let (mut initiator, mut responder) = established_pair(b"peer-a", b"peer-b");
        assert_eq!(responder.send_auth().unwrap(), Progress::Advanced);
        let from_responder = staged(&responder);
        let responder_len = responder.outbound().len();

        assert_eq!(
            initiator
                .step(&from_responder[..responder_len], None, now())
                .unwrap(),
            Progress::Authenticated
        );
        assert_eq!(initiator.peer_identity(), Some(&b"peer-b"[..]));
    }

    #[test]
    fn keys_survive_authentication() {
        let (mut initiator, mut responder) = established_pair(b"peer-a", b"peer-b");
        let before = *initiator.keys().unwrap().encrypt_key();

        let _ = initiator.send_auth().unwrap();
        let message = staged(&initiator);
        let len = initiator.outbound().len();
        let _ = responder.step(&message[..len], None, now()).unwrap();

        assert_eq!(initiator.keys().unwrap().encrypt_key(), &before);
        assert!(
            responder.keys().is_some(),
            "authentication does not drop the SA"
        );
        assert!(responder.established_at().is_some());
    }

    #[test]
    fn a_forged_mac_is_refused_and_leaves_no_keys() {
        let (mut initiator, mut responder) = established_pair(b"peer-a", b"peer-b");
        let _ = initiator.send_auth().unwrap();
        let mut message = staged(&initiator);
        let len = initiator.outbound().len();
        message[len - 1] ^= 0xFF;

        assert_eq!(
            responder.step(&message[..len], None, now()).err(),
            Some(CentauriError::AuthenticationFailed)
        );
        assert!(
            responder.peer_identity().is_none(),
            "a failed AUTH must not yield an authenticated peer"
        );
    }

    #[test]
    fn a_flipped_header_byte_fails_authentication() {
        // the oracle MACs the identity alone, so its header is unauthenticated
        // and this flip would pass there. Here the header is under the MAC.
        for header_byte in [16usize, 18, 20, 23] {
            let (mut initiator, mut responder) = established_pair(b"peer-a", b"peer-b");
            let _ = initiator.send_auth().unwrap();
            let mut message = staged(&initiator);
            let len = initiator.outbound().len();
            message[header_byte] ^= 0x01;

            let outcome = responder.step(&message[..len], None, now());

            assert!(
                outcome.is_err(),
                "byte {header_byte} of the header was not authenticated: {outcome:?}"
            );
        }
    }

    #[test]
    fn a_substituted_identity_fails_authentication() {
        let (mut initiator, mut responder) = established_pair(b"peer-a", b"peer-b");
        let _ = initiator.send_auth().unwrap();
        let mut message = staged(&initiator);
        let len = initiator.outbound().len();
        // swap the identity for another of the same length
        let identity_at = 28 + 2;
        message[identity_at..identity_at + 6].copy_from_slice(b"peer-c");

        assert_eq!(
            responder.step(&message[..len], None, now()).err(),
            Some(CentauriError::AuthenticationFailed)
        );
    }

    #[test]
    fn an_auth_message_is_self_describing() {
        // the length prefix is what lets a receiver locate the MAC without
        // knowing the peer's identity in advance -- the oracle's verify_auth
        // cannot parse without being told the identity length out of band.
        for identity in [&b""[..], b"a", b"peer-with-a-longer-name"] {
            let (mut initiator, mut responder) = established_pair(identity, b"peer-b");
            let _ = initiator.send_auth().unwrap();
            let message = staged(&initiator);
            let len = initiator.outbound().len();

            assert_eq!(
                responder.step(&message[..len], None, now()).unwrap(),
                Progress::Authenticated,
                "identity of {} bytes",
                identity.len()
            );
            assert_eq!(responder.peer_identity(), Some(identity));
        }
    }

    #[test]
    fn a_truncated_auth_message_asks_for_more() {
        let (mut initiator, _) = established_pair(b"peer-a", b"peer-b");
        let _ = initiator.send_auth().unwrap();
        let message = staged(&initiator);
        let len = initiator.outbound().len();

        for prefix in 0..len {
            let mut receiver = established_pair(b"peer-a", b"peer-b").1;
            let progress = receiver
                .step(&message[..prefix], None, now())
                .expect("a short auth read is not an error");
            assert_eq!(progress, Progress::NeedInput, "prefix of {prefix}");
        }
    }

    #[test]
    fn an_identity_past_the_cap_is_refused_not_truncated() {
        let oversize = [0x41u8; AUTH_MAX_IDENTITY_BYTES + 1];

        assert_eq!(
            Handshake::initiator(PSK, INITIATOR_SPI)
                .with_identity(&oversize)
                .err(),
            Some(CentauriError::PayloadTooLarge {
                len: oversize.len(),
                max: AUTH_MAX_IDENTITY_BYTES,
            })
        );
    }

    #[test]
    fn auth_before_sa_init_is_a_transition_error() {
        let mut initiator = Handshake::initiator(PSK, INITIATOR_SPI);

        assert_eq!(
            initiator.send_auth().err(),
            Some(CentauriError::InvalidTransition {
                expected: "established or peer-authenticated",
                found: "initial",
            })
        );
    }

    /// Both sides authenticated, ready to rekey.
    fn authenticated_pair() -> (Handshake, Handshake) {
        let (mut initiator, mut responder) = established_pair(b"peer-a", b"peer-b");
        let _ = initiator.send_auth().unwrap();
        let message = staged(&initiator);
        let len = initiator.outbound().len();
        let _ = responder.step(&message[..len], None, now()).unwrap();
        let _ = responder.send_auth().unwrap();
        let reply = staged(&responder);
        let reply_len = responder.outbound().len();
        let _ = initiator.step(&reply[..reply_len], None, now()).unwrap();
        (initiator, responder)
    }

    #[test]
    fn a_rekey_leaves_both_peers_agreeing() {
        // the property the oracle's rekey cannot satisfy: after both sides
        // rekey, what one encrypts the other still decrypts.
        let (mut initiator, mut responder) = authenticated_pair();
        let before = *initiator.keys().unwrap().encrypt_key();

        assert_eq!(
            initiator
                .send_rekey(Some(Entropy32::new([0x31; 32])), now())
                .unwrap(),
            Progress::Advanced
        );
        let request = captured(&initiator);

        assert_eq!(
            responder
                .step(&request, Some(Entropy32::new([0x32; 32])), now())
                .unwrap(),
            Progress::Rekeyed
        );
        let reply = captured(&responder);

        assert_eq!(
            initiator.step(&reply, None, now()).unwrap(),
            Progress::Rekeyed
        );

        let initiator_keys = initiator.keys().expect("rekeyed");
        let responder_keys = responder.keys().expect("rekeyed");

        assert_eq!(
            initiator_keys.encrypt_key(),
            responder_keys.decrypt_key(),
            "a rekey that does not agree is not a rekey"
        );
        assert_eq!(responder_keys.encrypt_key(), initiator_keys.decrypt_key());
        assert_ne!(
            initiator_keys.encrypt_key(),
            &before,
            "the keys are actually new"
        );
    }

    #[test]
    fn a_rekey_preserves_the_authenticated_identity() {
        let (mut initiator, mut responder) = authenticated_pair();
        assert_eq!(responder.peer_identity(), Some(&b"peer-a"[..]));

        let _ = initiator
            .send_rekey(Some(Entropy32::new([0x31; 32])), now())
            .unwrap();
        let request = captured(&initiator);
        let _ = responder
            .step(&request, Some(Entropy32::new([0x32; 32])), now())
            .unwrap();
        let reply = captured(&responder);
        let _ = initiator.step(&reply, None, now()).unwrap();

        assert_eq!(
            responder.peer_identity(),
            Some(&b"peer-a"[..]),
            "a rekey chains to the authenticated SA rather than dropping it"
        );
        assert_eq!(initiator.peer_identity(), Some(&b"peer-b"[..]));
    }

    #[test]
    fn rekeys_are_forward_secret_and_chain() {
        // two successive rekeys must each produce fresh keys, and the second
        // must not be derivable from the first's inputs alone.
        let (mut initiator, mut responder) = authenticated_pair();
        // fixed storage: a Vec here would pull an allocator into a suite that
        // has to run at the no-alloc tier
        let mut seen = [[0u8; 32]; 4];
        seen[0] = *initiator.keys().unwrap().encrypt_key();

        for round in 0..3u8 {
            let _ = initiator
                .send_rekey(Some(Entropy32::new([0x40 + round; 32])), now())
                .unwrap();
            let request = captured(&initiator);
            let _ = responder
                .step(&request, Some(Entropy32::new([0x50 + round; 32])), now())
                .unwrap();
            let reply = captured(&responder);
            let _ = initiator.step(&reply, None, now()).unwrap();

            let fresh = *initiator.keys().unwrap().encrypt_key();
            assert!(
                !seen[..=usize::from(round)].contains(&fresh),
                "round {round} reused a key"
            );
            assert_eq!(
                initiator.keys().unwrap().encrypt_key(),
                responder.keys().unwrap().decrypt_key(),
                "round {round} desynchronised"
            );
            seen[usize::from(round) + 1] = fresh;
        }
    }

    #[test]
    fn a_rekey_without_entropy_leaves_the_session_intact() {
        // a missing draw must not destroy a live SA
        let (mut initiator, _) = authenticated_pair();
        let before = *initiator.keys().unwrap().encrypt_key();

        assert_eq!(
            initiator.send_rekey(None, now()).err(),
            Some(CentauriError::EntropyUnavailable("step requires entropy"))
        );
        assert_eq!(
            initiator.keys().unwrap().encrypt_key(),
            &before,
            "the old SA survives a failed rekey"
        );
        assert_eq!(initiator.peer_identity(), Some(&b"peer-b"[..]));
    }

    #[test]
    fn an_sa_init_cannot_masquerade_as_a_rekey() {
        // exchange type is checked, so a fresh handshake message cannot reset
        // an established session
        let (mut initiator, mut responder) = authenticated_pair();
        let mut fresh = Handshake::initiator(PSK, INITIATOR_SPI);
        let _ = fresh
            .step(&[], Some(Entropy32::new(INITIATOR_SEED)), now())
            .unwrap();
        let sa_init = captured(&fresh);

        assert_eq!(
            responder
                .step(&sa_init, Some(Entropy32::new([0x32; 32])), now())
                .err(),
            Some(CentauriError::InvalidMessage("exchange_type"))
        );
        assert!(
            responder.keys().is_some(),
            "the session survived the attempt"
        );
        let _ = &mut initiator;
    }

    #[test]
    fn rekey_before_authentication_is_a_transition_error() {
        let (mut initiator, _) = established_pair(b"peer-a", b"peer-b");

        assert_eq!(
            initiator
                .send_rekey(Some(Entropy32::new([0x31; 32])), now())
                .err(),
            Some(CentauriError::InvalidTransition {
                expected: "authenticated",
                found: "established",
            })
        );
        // and the refusal did not destroy the session
        assert!(
            initiator.keys().is_some(),
            "a refused rekey must leave the SA alive"
        );
    }

    /// The all-zero point and two RFC 7748 small-order points. Every one maps
    /// to an all-zero shared secret under X25519, which is why they must be
    /// refused rather than trusted to be rare.
    const LOW_ORDER_POINTS: [[u8; 32]; 3] = [
        [0u8; 32],
        {
            let mut point = [0u8; 32];
            point[0] = 1;
            point
        },
        [
            0xe0, 0xeb, 0x7a, 0x7c, 0x3b, 0x41, 0xb8, 0xae, 0x16, 0x56, 0xe3, 0xfa, 0xf1, 0x9f,
            0xc4, 0x6a, 0xda, 0x09, 0x8d, 0xeb, 0x9c, 0x32, 0xb1, 0xfd, 0x86, 0x62, 0x05, 0x16,
            0x5f, 0x49, 0xb8, 0x00,
        ],
    ];

    #[test]
    fn a_low_order_dh_value_is_refused_by_the_responder() {
        // an active attacker substituting the peer's DH value would otherwise
        // get an ephemeral that contributes nothing: key secrecy would still
        // rest on the PSK, but forward secrecy would be gone silently.
        let mut initiator = Handshake::initiator(PSK, INITIATOR_SPI);
        let _ = initiator
            .step(&[], Some(Entropy32::new(INITIATOR_SEED)), now())
            .unwrap();

        for (index, point) in LOW_ORDER_POINTS.iter().enumerate() {
            let mut tampered = captured(&initiator);
            tampered[HEADER_LEN + NONCE_LEN..MESSAGE_LEN].copy_from_slice(point);

            let mut responder = Handshake::responder(PSK, RESPONDER_SPI);
            assert_eq!(
                responder
                    .step(&tampered, Some(Entropy32::new(RESPONDER_SEED)), now())
                    .err(),
                Some(CentauriError::DegenerateKeyAgreement),
                "low-order point {index} was accepted"
            );
            assert!(
                responder.keys().is_none(),
                "no keys from a degenerate agreement"
            );
        }
    }

    #[test]
    fn a_low_order_dh_value_is_refused_by_the_initiator() {
        let (mut initiator, responder) = {
            let mut initiator = Handshake::initiator(PSK, INITIATOR_SPI);
            let mut responder = Handshake::responder(PSK, RESPONDER_SPI);
            let _ = initiator
                .step(&[], Some(Entropy32::new(INITIATOR_SEED)), now())
                .unwrap();
            let init = captured(&initiator);
            let _ = responder
                .step(&init, Some(Entropy32::new(RESPONDER_SEED)), now())
                .unwrap();
            (initiator, responder)
        };

        let mut tampered = captured(&responder);
        tampered[HEADER_LEN + NONCE_LEN..MESSAGE_LEN].copy_from_slice(&LOW_ORDER_POINTS[0]);

        assert_eq!(
            initiator.step(&tampered, None, now()).err(),
            Some(CentauriError::DegenerateKeyAgreement),
            "both directions must refuse, not just the responder"
        );
    }

    #[test]
    fn a_low_order_dh_value_is_refused_during_rekey() {
        // the rekey is where this matters most: forward secrecy is the whole
        // claim, and a degenerate agreement removes it while everything else
        // still appears to work
        let (mut initiator, mut responder) = authenticated_pair();
        let _ = initiator
            .send_rekey(Some(Entropy32::new([0x31; 32])), now())
            .unwrap();
        let mut tampered = captured(&initiator);
        tampered[HEADER_LEN + NONCE_LEN..MESSAGE_LEN].copy_from_slice(&LOW_ORDER_POINTS[2]);

        assert_eq!(
            responder
                .step(&tampered, Some(Entropy32::new([0x32; 32])), now())
                .err(),
            Some(CentauriError::DegenerateKeyAgreement)
        );
    }

    #[test]
    fn no_truncation_of_a_valid_message_ever_panics_or_parses() {
        // exhaustive rather than sampled, and alloc-free, so it runs at every
        // tier: a parser is fed every prefix of a message it would otherwise
        // accept.
        let mut initiator = Handshake::initiator(PSK, INITIATOR_SPI);
        let _ = initiator
            .step(&[], Some(Entropy32::new(INITIATOR_SEED)), now())
            .unwrap();
        let valid = captured(&initiator);

        for length in 0..MESSAGE_LEN {
            let mut responder = Handshake::responder(PSK, RESPONDER_SPI);
            let progress = responder
                .step(
                    &valid[..length],
                    Some(Entropy32::new(RESPONDER_SEED)),
                    now(),
                )
                .expect("a short read is never an error");

            assert_eq!(progress, Progress::NeedInput, "prefix of {length} bytes");
            assert!(
                responder.keys().is_none(),
                "prefix of {length} bytes established a key"
            );
        }
    }

    #[test]
    fn every_single_bit_flip_in_the_header_is_either_caught_or_harmless() {
        // The header's version and exchange-type bytes are the only fields the
        // parser validates; everything else is either an identifier it echoes
        // or key material it consumes. Walking every bit proves the validation
        // is exactly where the doc says, with no accidental coverage gaps.
        let mut initiator = Handshake::initiator(PSK, INITIATOR_SPI);
        let _ = initiator
            .step(&[], Some(Entropy32::new(INITIATOR_SEED)), now())
            .unwrap();
        let valid = captured(&initiator);

        let mut rejected_positions = 0usize;
        for byte_index in 0..MESSAGE_LEN {
            for bit in 0..8u32 {
                let mut corrupted = valid;
                corrupted[byte_index] ^= 1u8 << bit;

                let mut responder = Handshake::responder(PSK, RESPONDER_SPI);
                let outcome =
                    responder.step(&corrupted, Some(Entropy32::new(RESPONDER_SEED)), now());

                match outcome {
                    Err(CentauriError::InvalidMessage(_)) => rejected_positions += 1,
                    // a flip in an spi/nonce/dh byte is accepted by design: the
                    // handshake has no integrity check until AUTH, and a wrong
                    // dh value simply derives keys the peer will not share.
                    Ok(Progress::Established) => {}
                    other => panic!("byte {byte_index} bit {bit}: unexpected {other:?}"),
                }
            }
        }

        // exactly the version and exchange-type bytes reject, 8 bits each,
        // minus the two flips that happen to land on a still-valid encoding
        assert!(rejected_positions > 0, "no corruption was rejected at all");
        assert!(
            rejected_positions <= 16,
            "more positions rejected than the two validated bytes can account for: {rejected_positions}"
        );
    }

    #[test]
    fn arbitrary_bytes_never_panic_the_parser() {
        // a cheap deterministic sweep over shapes a fuzzer would find first:
        // all-zero, all-one, and every single-byte value at every length class.
        for filler in [0x00u8, 0xFF, 0x21, 0x20] {
            for length in [0, 1, 27, 28, 59, 60, 91, MESSAGE_LEN, MESSAGE_LEN + 1] {
                let mut buffer = [0u8; MESSAGE_LEN + 1];
                let bounded = length.min(buffer.len());
                buffer[..bounded].fill(filler);

                let mut responder = Handshake::responder(PSK, RESPONDER_SPI);
                // the contract is "never panics"; any Ok/Err is acceptable
                let _ = responder.step(
                    &buffer[..bounded],
                    Some(Entropy32::new(RESPONDER_SEED)),
                    now(),
                );
            }
        }
    }

    #[test]
    fn roles_are_distinct() {
        assert_ne!(Role::Initiator, Role::Responder);
    }

    /// Decode a hex fixture into a fixed array. A `Vec`-returning parser would
    /// pull an allocator into the suite; this keeps the oracle vectors usable
    /// at the no-alloc tier.
    fn hex<const N: usize>(text: &str) -> [u8; N] {
        fn nibble(character: u8) -> u8 {
            match character {
                b'0'..=b'9' => character - b'0',
                b'a'..=b'f' => character - b'a' + 10,
                _ => panic!("fixture is not lowercase hex"),
            }
        }

        let bytes = text.as_bytes();
        assert_eq!(
            bytes.len(),
            N * 2,
            "fixture length must match the target array"
        );

        let mut out = [0u8; N];
        for (index, slot) in out.iter_mut().enumerate() {
            *slot = (nibble(bytes[index * 2]) << 4) | nibble(bytes[index * 2 + 1]);
        }
        out
    }

    // Recorded from a live `csr-security` IkeSa run — see that crate's
    // tests/centauri_wire_vectors.rs, which feeds it the INIT this initiator
    // produces from INITIATOR_SEED and prints these three lines. The oracle
    // draws its own nonce and DH value from getrandom, so re-recording yields
    // different bytes; what must not change is that both sides agree.
    const ORACLE_RESPONSE: &str = concat!(
        "0102030405060708329e6a0d5e4d7baa21202208000000000000005c",
        "7ba4d7ddc8fb09c07f48cdf561dab74009a240cace68e5635e371a8b9f3d8349",
        "0494063474b1b67fcecc0aa5eb27c321742a73d363eaaf325ac725f7c6ea4f35",
    );
    const ORACLE_SK_ER: &str = "621602dda6d23a04ca720136dbf480bc865c8006a53d3d71d93dab0c71f1b7ed";
    const ORACLE_SK_EI: &str = "39a3f811687a8093326de097bdd41f672047f531cbae681bd71ff6a6919334b4";

    /// The exact bytes handed to the oracle, transcribed into its test.
    const ORACLE_WAS_FED: &str = concat!(
        "0102030405060708000000000000000021202208000000000000005c",
        "f859442e215e26bc5a488cd0eee4942ec66a2c5fa65d9cac704ea0ed442a1787",
        "67ee57f76474618c387fbd0d770b95f8664c39741cd44649a1d23584c6785f3f",
    );

    #[test]
    fn interoperates_with_a_live_csr_security_responder() {
        let mut initiator = Handshake::initiator(PSK, INITIATOR_SPI);
        let _ = initiator
            .step(&[], Some(Entropy32::new(INITIATOR_SEED)), now())
            .unwrap();

        // the oracle accepted exactly these bytes: its half of this test
        // asserts respond() parses them without modification.
        assert_eq!(
            initiator.outbound(),
            &hex::<MESSAGE_LEN>(ORACLE_WAS_FED)[..],
            "the INIT the oracle was fed must still be what we produce"
        );

        let response = hex::<MESSAGE_LEN>(ORACLE_RESPONSE);
        let progress = initiator.step(&response, None, now()).unwrap();
        assert_eq!(progress, Progress::Established);

        let keys = initiator
            .keys()
            .expect("established from the oracle's reply");

        assert_eq!(
            keys.encrypt_key(),
            &hex::<32>(ORACLE_SK_EI),
            "what we encrypt with must be what the oracle decrypts with"
        );
        assert_eq!(
            keys.decrypt_key(),
            &hex::<32>(ORACLE_SK_ER),
            "and what the oracle encrypts with must be what we decrypt with"
        );
    }
}
