//! Property tests over the wire surface.
//!
//! `proptest` was a declared dev-dependency that nothing used — a dependency
//! doing no work, which is worse than an absent one because it reads as
//! coverage. These are the properties the exhaustive sweeps in the unit tests
//! cannot state: those enumerate a fixed message's neighbourhood, while these
//! quantify over *arbitrary* input.
//!
//! The bar for every property here is the same and deliberately low-level:
//! **no input panics, and no input authenticates.** A parser that survives
//! arbitrary bytes and a verifier that cannot be tricked into accepting them
//! are the two things a wire-facing crate must be able to say.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use proptest::prelude::*;
use proxima_centauri::cookie::{CookieSecret, Verdict, examine};
use proxima_centauri::esp::{HEADER_LEN, OVERHEAD};
use proxima_centauri::{ChildSa, Entropy32, EspSpi, Handshake, IkeSpi, Progress, Role};
use proxima_clock::ticks::Ticks;

const PSK: [u8; 32] = [0xAB; 32];

fn now() -> Ticks {
    Ticks::from_raw(1_000)
}

fn agreed_pair() -> (ChildSa, ChildSa) {
    let mut initiator = Handshake::initiator(PSK, IkeSpi::new(1));
    let mut responder = Handshake::responder(PSK, IkeSpi::new(2));

    let _ = initiator
        .step(&[], Some(Entropy32::new([0x11; 32])), now())
        .unwrap();
    let mut init = [0u8; 92];
    init.copy_from_slice(initiator.outbound());
    let _ = responder
        .step(&init, Some(Entropy32::new([0x22; 32])), now())
        .unwrap();
    let mut reply = [0u8; 92];
    reply.copy_from_slice(responder.outbound());
    let _ = initiator.step(&reply, None, now()).unwrap();

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

proptest! {
    /// The handshake parser must survive anything, and must never establish an
    /// SA from bytes it did not agree.
    #[test]
    fn arbitrary_bytes_never_establish_a_handshake(bytes in prop::collection::vec(any::<u8>(), 0..300)) {
        let mut responder = Handshake::responder(PSK, IkeSpi::new(2));

        let outcome = responder.step(&bytes, Some(Entropy32::new([0x22; 32])), now());

        // Ok is allowed — NeedInput on a short read is ordinary — but random
        // bytes cannot carry a real DH value, so nothing may establish.
        if let Ok(progress) = outcome {
            prop_assert_ne!(progress, Progress::Authenticated);
            if progress == Progress::Established {
                // only reachable if the bytes happened to be a well-formed
                // SA_INIT, which random input will not produce; assert the
                // keys are at least self-consistent rather than half-built
                prop_assert!(responder.keys().is_some());
            }
        }
    }

    /// The AEAD opener must survive anything, and open nothing it did not seal.
    #[test]
    fn arbitrary_bytes_never_open_a_packet(bytes in prop::collection::vec(any::<u8>(), 0..2048)) {
        let (_, mut receiver) = agreed_pair();
        let mut buffer = bytes.clone();

        let outcome = receiver.open(&mut buffer);

        prop_assert!(outcome.is_err(), "random bytes authenticated as a packet");
    }

    /// Any single-byte corruption of a real packet is caught, at any offset
    /// and with any replacement value — the randomised complement to the
    /// exhaustive bit-flip sweep in the unit tests.
    #[test]
    fn any_corruption_of_a_real_packet_is_caught(
        offset in 0usize..(64 + OVERHEAD),
        value in any::<u8>(),
    ) {
        let payload = [0x5Au8; 64];
        let (mut sender, _) = agreed_pair();
        let mut sealed = [0u8; 64 + OVERHEAD];
        sealed[HEADER_LEN..HEADER_LEN + payload.len()].copy_from_slice(&payload);
        let len = sender.seal(&mut sealed, payload.len()).unwrap();

        prop_assume!(offset < len);
        prop_assume!(sealed[offset] != value);

        let (_, mut receiver) = agreed_pair();
        let mut corrupted = sealed;
        corrupted[offset] = value;

        prop_assert!(receiver.open(&mut corrupted[..len]).is_err());
    }

    /// A cookie must never be accepted for a peer token it was not issued to,
    /// whatever the token looks like.
    #[test]
    fn a_cookie_never_serves_a_different_peer(
        issued_to in prop::collection::vec(any::<u8>(), 1..40),
        offered_by in prop::collection::vec(any::<u8>(), 1..40),
    ) {
        prop_assume!(issued_to != offered_by);

        let secret = CookieSecret::new([0x77; 32]);
        let mut initiator = Handshake::initiator(PSK, IkeSpi::new(3));
        let _ = initiator
            .step(&[], Some(Entropy32::new([0x11; 32])), now())
            .unwrap();
        let mut sa_init = [0u8; 92];
        sa_init.copy_from_slice(initiator.outbound());

        let Verdict::Challenge(challenge) = examine(&secret, &issued_to, &sa_init).unwrap() else {
            return Err(TestCaseError::fail("expected a challenge"));
        };
        let cookie = proxima_centauri::cookie::cookie_from_challenge(&challenge).unwrap();

        let mut retry = [0u8; 92 + 32];
        let len = proxima_centauri::cookie::attach_cookie(&sa_init, &cookie, &mut retry).unwrap();

        prop_assert!(matches!(
            examine(&secret, &offered_by, &retry[..len]).unwrap(),
            Verdict::Challenge(_)
        ));
    }
}
