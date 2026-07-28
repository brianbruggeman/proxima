//! Stateless round-trip proof, so an unauthenticated packet cannot buy
//! expensive work.
//!
//! # The exposure this closes
//!
//! A responder completing SA_INIT performs two X25519 scalar multiplications —
//! measured at 48 µs — for a 92-byte packet from anyone at all. That is ~20k
//! packets per second to saturate a core, about 15 Mbit/s of attack traffic,
//! from a source that never has to receive a reply. Spoofed, it costs the
//! attacker nothing.
//!
//! RFC 7296 §2.6 answers this with a cookie, and the shape matters more than
//! the construction: **the responder must not allocate state or spend
//! asymmetric crypto before the peer proves it can receive.** So this module
//! is free functions over raw bytes, not a method on [`Handshake`] — by the
//! time you hold a `Handshake` you have already spent the state the cookie
//! exists to protect.
//!
//! ```text
//! bytes ── examine ──> Challenge  ── send 60 bytes, allocate nothing
//!               └────> Proceed    ── now build a Handshake
//! ```
//!
//! # What it does and does not buy
//!
//! It defeats **blind** flooding: an attacker who cannot receive at the
//! address it claims can never echo a valid cookie. It does **not** stop an
//! attacker who can complete a round trip — nothing stateless can — and it is
//! not a rate limiter. It converts an amplification into a symmetric cost.
//!
//! The cookie binds a caller-supplied `peer_token` (the driver's view of who
//! sent the packet: an IP, a connection id, whatever the transport knows),
//! which is why this is sans-IO — the state machine never learns an address,
//! the driver passes one in.

use subtle::ConstantTimeEq;

use crate::error::CentauriError;
use crate::handshake::{IkeSpi, MESSAGE_LEN};
use crate::hash::keyed_hash;

/// Bytes of a cookie.
pub const COOKIE_LEN: usize = 32;

/// A cookied SA_INIT: the ordinary message with a cookie appended.
///
/// Appending rather than inserting keeps an uncookied 92-byte SA_INIT parsing
/// exactly as before, so a peer that never sees a challenge is unaffected and
/// the wire compatibility C1 proved against `csr-security` survives.
pub const COOKIED_MESSAGE_LEN: usize = MESSAGE_LEN + COOKIE_LEN;

/// Bytes of a challenge message.
pub const CHALLENGE_LEN: usize = 28 + COOKIE_LEN;

/// No amplification: the reply must never exceed what provoked it, or this
/// becomes the very thing it defends against. A const assertion rather than a
/// test, because it is a property of the constants and should fail the build
/// rather than a test run.
const _: () = assert!(
    CHALLENGE_LEN < MESSAGE_LEN,
    "a challenge larger than the SA_INIT it refuses would make this an amplifier"
);

const CHALLENGE_EXCHANGE_TYPE: u8 = 0x25;
const VERSION: u8 = 0x20;

/// The responder's rotating cookie key.
///
/// Rotating it invalidates outstanding cookies, which is the intended way to
/// shed a flood that has learned one. Holding two — current and previous —
/// and accepting either lets a rotation happen without breaking initiators
/// mid-exchange; this type is one key, and that policy belongs to the driver.
#[derive(Clone)]
pub struct CookieSecret {
    key: [u8; 32],
}

impl core::fmt::Debug for CookieSecret {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("CookieSecret")
            .finish_non_exhaustive()
    }
}

impl Drop for CookieSecret {
    fn drop(&mut self) {
        self.key = [0u8; 32];
        let _ = core::hint::black_box(&self.key);
    }
}

impl CookieSecret {
    /// Wrap a secret drawn from a real entropy source.
    #[must_use]
    pub const fn new(key: [u8; 32]) -> Self {
        Self { key }
    }
}

/// What to do with a packet, decided before any state exists for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
#[non_exhaustive]
pub enum Verdict {
    /// The peer proved a round trip. Build a handshake and feed it the
    /// message, which starts at offset zero and runs [`MESSAGE_LEN`] bytes —
    /// the cookie is not part of what the state machine parses.
    Proceed,
    /// No valid cookie. Send the challenge and discard the packet; no state
    /// was created and no asymmetric crypto was performed.
    Challenge([u8; CHALLENGE_LEN]),
}

/// Derive the cookie for a peer and a message.
///
/// Binds the peer token, the initiator's SPI and its nonce, so a cookie is
/// useless to a different source or a different exchange.
fn derive(secret: &CookieSecret, peer_token: &[u8], spi: IkeSpi, nonce: &[u8]) -> [u8; COOKIE_LEN] {
    // one keyed hash, ~80 ns, against the 48 µs it guards
    let mut transcript = [0u8; 8 + 32];
    transcript[..8].copy_from_slice(&spi.as_raw().to_be_bytes());
    let bounded = nonce.len().min(32);
    transcript[8..8 + bounded].copy_from_slice(&nonce[..bounded]);

    let bound = keyed_hash(&secret.key, &transcript);
    keyed_hash(&bound, peer_token)
}

/// Decide whether a raw inbound SA_INIT has earned a handshake.
///
/// # Errors
///
/// [`CentauriError::InvalidMessage`] if the bytes are too short to be an
/// SA_INIT at all — rejected before anything else, since a runt cannot even be
/// challenged coherently.
pub fn examine(
    secret: &CookieSecret,
    peer_token: &[u8],
    message: &[u8],
) -> Result<Verdict, CentauriError> {
    if message.len() < MESSAGE_LEN {
        return Err(CentauriError::InvalidMessage(
            "sa_init too short to examine",
        ));
    }

    let spi = IkeSpi::new(u64::from_be_bytes(
        message[0..8]
            .try_into()
            .map_err(|_| CentauriError::InvalidMessage("spi_initiator"))?,
    ));
    let nonce = &message[28..60];
    let expected = derive(secret, peer_token, spi, nonce);

    if message.len() >= COOKIED_MESSAGE_LEN {
        let offered = &message[MESSAGE_LEN..COOKIED_MESSAGE_LEN];
        if bool::from(expected.ct_eq(offered)) {
            return Ok(Verdict::Proceed);
        }
    }

    Ok(Verdict::Challenge(challenge(spi, &expected)))
}

/// Build the challenge a responder sends back.
fn challenge(spi: IkeSpi, cookie: &[u8; COOKIE_LEN]) -> [u8; CHALLENGE_LEN] {
    let mut message = [0u8; CHALLENGE_LEN];
    message[0..8].copy_from_slice(&spi.as_raw().to_be_bytes());
    // responder SPI stays zero: none has been chosen, because no state exists
    message[16] = 0x29;
    message[17] = VERSION;
    message[18] = CHALLENGE_EXCHANGE_TYPE;
    message[19] = 0x20;
    message[24..28].copy_from_slice(&(CHALLENGE_LEN as u32).to_be_bytes());
    message[28..CHALLENGE_LEN].copy_from_slice(cookie);
    message
}

/// Read the cookie out of a challenge, so an initiator can retry.
///
/// # Errors
///
/// [`CentauriError::InvalidMessage`] if the message is not a challenge.
pub fn cookie_from_challenge(message: &[u8]) -> Result<[u8; COOKIE_LEN], CentauriError> {
    if message.len() < CHALLENGE_LEN {
        return Err(CentauriError::InvalidMessage("challenge too short"));
    }
    if message[18] != CHALLENGE_EXCHANGE_TYPE {
        return Err(CentauriError::InvalidMessage("not a cookie challenge"));
    }

    let mut cookie = [0u8; COOKIE_LEN];
    cookie.copy_from_slice(&message[28..CHALLENGE_LEN]);
    Ok(cookie)
}

/// Append a cookie to an SA_INIT for the retry.
///
/// # Errors
///
/// [`CentauriError::BufferTooSmall`] if the destination cannot hold the
/// cookied message.
pub fn attach_cookie(
    sa_init: &[u8],
    cookie: &[u8; COOKIE_LEN],
    out: &mut [u8],
) -> Result<usize, CentauriError> {
    let needed = sa_init.len() + COOKIE_LEN;
    if out.len() < needed {
        return Err(CentauriError::BufferTooSmall {
            needed,
            available: out.len(),
        });
    }

    out[..sa_init.len()].copy_from_slice(sa_init);
    out[sa_init.len()..needed].copy_from_slice(cookie);

    Ok(needed)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use proxima_clock::ticks::Ticks;

    use super::{
        COOKIED_MESSAGE_LEN, CookieSecret, Verdict, attach_cookie, cookie_from_challenge, examine,
    };
    use crate::entropy::Entropy32;
    use crate::error::CentauriError;
    use crate::handshake::{Handshake, IkeSpi, MESSAGE_LEN, Progress};

    const PSK: [u8; 32] = [0xAB; 32];
    const SECRET: [u8; 32] = [0x77; 32];
    const PEER: &[u8] = b"198.51.100.7:4500";

    fn an_sa_init() -> [u8; MESSAGE_LEN] {
        let mut initiator = Handshake::initiator(PSK, IkeSpi::new(0x0102_0304));
        let _ = initiator
            .step(&[], Some(Entropy32::new([0x11; 32])), Ticks::from_raw(1))
            .unwrap();
        let mut message = [0u8; MESSAGE_LEN];
        message.copy_from_slice(initiator.outbound());
        message
    }

    #[test]
    fn a_first_contact_is_challenged_not_served() {
        let secret = CookieSecret::new(SECRET);

        let verdict = examine(&secret, PEER, &an_sa_init()).unwrap();

        assert!(
            matches!(verdict, Verdict::Challenge(_)),
            "an unproven peer must not buy a scalar multiplication"
        );
    }

    #[test]
    fn a_returned_cookie_is_accepted() {
        let secret = CookieSecret::new(SECRET);
        let sa_init = an_sa_init();

        let Verdict::Challenge(challenge) = examine(&secret, PEER, &sa_init).unwrap() else {
            panic!("expected a challenge");
        };

        let cookie = cookie_from_challenge(&challenge).unwrap();
        let mut retry = [0u8; COOKIED_MESSAGE_LEN];
        let len = attach_cookie(&sa_init, &cookie, &mut retry).unwrap();

        assert_eq!(len, COOKIED_MESSAGE_LEN);
        assert_eq!(
            examine(&secret, PEER, &retry[..len]).unwrap(),
            Verdict::Proceed
        );
    }

    #[test]
    fn a_cookie_does_not_travel_to_another_peer() {
        // the whole point: a cookie earned from one source is useless from
        // another, so an attacker cannot harvest one and spray it
        let secret = CookieSecret::new(SECRET);
        let sa_init = an_sa_init();

        let Verdict::Challenge(challenge) = examine(&secret, PEER, &sa_init).unwrap() else {
            panic!("expected a challenge");
        };
        let cookie = cookie_from_challenge(&challenge).unwrap();
        let mut retry = [0u8; COOKIED_MESSAGE_LEN];
        let len = attach_cookie(&sa_init, &cookie, &mut retry).unwrap();

        let verdict = examine(&secret, b"203.0.113.9:4500", &retry[..len]).unwrap();

        assert!(
            matches!(verdict, Verdict::Challenge(_)),
            "a cookie bound to one peer token must not serve another"
        );
    }

    #[test]
    fn a_cookie_does_not_travel_to_another_exchange() {
        let secret = CookieSecret::new(SECRET);
        let first = an_sa_init();

        let Verdict::Challenge(challenge) = examine(&secret, PEER, &first).unwrap() else {
            panic!("expected a challenge");
        };
        let cookie = cookie_from_challenge(&challenge).unwrap();

        // a different exchange: different nonce, so the cookie must not carry
        let mut other = Handshake::initiator(PSK, IkeSpi::new(0x0102_0304));
        let _ = other
            .step(&[], Some(Entropy32::new([0x99; 32])), Ticks::from_raw(1))
            .unwrap();
        let mut second = [0u8; MESSAGE_LEN];
        second.copy_from_slice(other.outbound());

        let mut retry = [0u8; COOKIED_MESSAGE_LEN];
        let len = attach_cookie(&second, &cookie, &mut retry).unwrap();

        assert!(matches!(
            examine(&secret, PEER, &retry[..len]).unwrap(),
            Verdict::Challenge(_)
        ));
    }

    #[test]
    fn rotating_the_secret_invalidates_outstanding_cookies() {
        let secret = CookieSecret::new(SECRET);
        let sa_init = an_sa_init();
        let Verdict::Challenge(challenge) = examine(&secret, PEER, &sa_init).unwrap() else {
            panic!("expected a challenge");
        };
        let cookie = cookie_from_challenge(&challenge).unwrap();
        let mut retry = [0u8; COOKIED_MESSAGE_LEN];
        let len = attach_cookie(&sa_init, &cookie, &mut retry).unwrap();

        let rotated = CookieSecret::new([0x88; 32]);

        assert!(
            matches!(
                examine(&rotated, PEER, &retry[..len]).unwrap(),
                Verdict::Challenge(_)
            ),
            "rotation is how a responder sheds a flood that learned a cookie"
        );
    }

    #[test]
    fn a_forged_cookie_is_refused() {
        let secret = CookieSecret::new(SECRET);
        let sa_init = an_sa_init();
        let mut retry = [0u8; COOKIED_MESSAGE_LEN];
        let len = attach_cookie(&sa_init, &[0xFF; 32], &mut retry).unwrap();

        assert!(matches!(
            examine(&secret, PEER, &retry[..len]).unwrap(),
            Verdict::Challenge(_)
        ));
    }

    #[test]
    fn an_accepted_message_still_parses_as_an_ordinary_sa_init() {
        // the cookie is appended, so the state machine sees the message it
        // always saw — which is what keeps C1's proven wire compatibility
        let secret = CookieSecret::new(SECRET);
        let sa_init = an_sa_init();
        let Verdict::Challenge(challenge) = examine(&secret, PEER, &sa_init).unwrap() else {
            panic!("expected a challenge");
        };
        let cookie = cookie_from_challenge(&challenge).unwrap();
        let mut retry = [0u8; COOKIED_MESSAGE_LEN];
        let _ = attach_cookie(&sa_init, &cookie, &mut retry).unwrap();

        let mut responder = Handshake::responder(PSK, IkeSpi::new(9));
        let progress = responder
            .step(
                &retry[..MESSAGE_LEN],
                Some(Entropy32::new([0x22; 32])),
                Ticks::from_raw(1),
            )
            .unwrap();

        assert_eq!(progress, Progress::Established);
    }

    #[test]
    fn a_runt_is_rejected_before_anything_else() {
        let secret = CookieSecret::new(SECRET);

        assert_eq!(
            examine(&secret, PEER, &[0u8; 40]).err(),
            Some(CentauriError::InvalidMessage(
                "sa_init too short to examine"
            ))
        );
    }

    #[test]
    fn the_secret_does_not_print() {
        let mut buffer = crate::test_support::Buffer::new();
        let secret = CookieSecret::new(SECRET);
        core::fmt::write(&mut buffer, format_args!("{secret:?}")).unwrap();

        assert!(!buffer.as_str().contains("119"), "0x77 must not appear");
    }
}
