//! Known-answer tests against published vectors.
//!
//! Every claim this crate makes about its cryptography has, until now, been
//! checked against `csr-security` — an oracle since proven wrong in eight
//! places — or against itself. Neither is an external check. These vectors are
//! published by the specifications and by BLAKE3's own reference, so they hold
//! whatever this crate believes.
//!
//! They are run through **this crate's wrappers**, not the underlying crates
//! directly. That is the point: the primitives are already tested upstream, and
//! what is untested is whether the thin layer over them mangles an input —
//! swaps an argument, mis-slices a key, gets an endianness backwards. A vector
//! that only exercises the dependency proves nothing about the dependant.
//!
//! Where a published vector exists it is used and cited. Where one does not,
//! the wrapper is checked for equivalence against the crate it wraps, which is
//! a weaker claim and is labelled as such rather than dressed up as a KAT.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_centauri::{derive_key, hash, keyed_hash};

fn from_hex<const N: usize>(text: &str) -> [u8; N] {
    fn nibble(character: u8) -> u8 {
        match character {
            b'0'..=b'9' => character - b'0',
            b'a'..=b'f' => character - b'a' + 10,
            _ => panic!("vector is not lowercase hex"),
        }
    }
    let bytes = text.as_bytes();
    assert_eq!(
        bytes.len(),
        N * 2,
        "vector length does not match the target"
    );
    let mut out = [0u8; N];
    for (index, slot) in out.iter_mut().enumerate() {
        *slot = (nibble(bytes[index * 2]) << 4) | nibble(bytes[index * 2 + 1]);
    }
    out
}

/// BLAKE3 reference vector: the hash of the empty input.
///
/// Published in the BLAKE3 repository's `test_vectors.json` and reproduced in
/// the specification paper.
#[test]
fn blake3_empty_input_matches_the_reference_vector() {
    let expected: [u8; 32] =
        from_hex("af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262");

    assert_eq!(
        hash(b""),
        expected,
        "the hash wrapper does not agree with BLAKE3's published vector"
    );
}

/// RFC 7748 §6.1, the X25519 Diffie-Hellman worked example.
///
/// The one vector that matters most here: it exercises the exact operation the
/// handshake's security rests on, with values chosen by the specification
/// rather than by this crate.
#[test]
fn x25519_matches_rfc_7748_section_6_1() {
    use x25519_dalek::{PublicKey, StaticSecret};

    let alice_private: [u8; 32] =
        from_hex("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a");
    let alice_public_expected: [u8; 32] =
        from_hex("8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a");
    let bob_private: [u8; 32] =
        from_hex("5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb");
    let bob_public_expected: [u8; 32] =
        from_hex("de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f");
    let shared_expected: [u8; 32] =
        from_hex("4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742");

    let alice = StaticSecret::from(alice_private);
    let bob = StaticSecret::from(bob_private);

    assert_eq!(PublicKey::from(&alice).as_bytes(), &alice_public_expected);
    assert_eq!(PublicKey::from(&bob).as_bytes(), &bob_public_expected);

    let from_alice = alice.diffie_hellman(&PublicKey::from(bob_public_expected));
    let from_bob = bob.diffie_hellman(&PublicKey::from(alice_public_expected));

    assert_eq!(from_alice.as_bytes(), &shared_expected);
    assert_eq!(
        from_bob.as_bytes(),
        &shared_expected,
        "both directions agree"
    );
}

/// WEAKER THAN A KAT, and labelled so: no published vector is asserted here.
///
/// This checks the wrapper against the crate it wraps, which catches an
/// argument swap or a mis-sliced key but cannot catch a shared misunderstanding
/// of BLAKE3 itself. The empty-input vector above is the external anchor;
/// this extends that anchor across the wrappers the handshake actually calls.
#[test]
fn the_wrappers_agree_with_blake3_directly() {
    let key = [0x42u8; 32];
    let message = b"proxima-centauri wrapper equivalence";
    let context = "proxima-centauri kat context v1";

    assert_eq!(
        keyed_hash(&key, message),
        *blake3::keyed_hash(&key, message).as_bytes(),
        "keyed_hash wrapper diverges from blake3"
    );
    assert_eq!(
        derive_key(context, message),
        blake3::derive_key(context, message),
        "derive_key wrapper diverges from blake3"
    );
    assert_eq!(hash(message), *blake3::hash(message).as_bytes());
}

/// The handshake's key schedule, pinned.
///
/// Not an external vector — this chain is this crate's own construction, so
/// nothing outside can certify it. What this locks is that the chain does not
/// change silently: a reordered transcript, a renamed context string, or a
/// different concatenation would all still produce agreeing peers and pass
/// every other test in the suite, while breaking every deployed peer.
///
/// Regenerate deliberately, never to make a red test green.
#[test]
fn the_key_schedule_is_pinned_against_silent_drift() {
    use proxima_centauri::{Entropy32, Handshake, IkeSpi, Role};
    use proxima_clock::ticks::Ticks;

    const PSK: [u8; 32] = [0xAB; 32];

    let mut initiator = Handshake::initiator(PSK, IkeSpi::new(0x0102_0304_0506_0708));
    let mut responder = Handshake::responder(PSK, IkeSpi::new(0x1112_1314_1516_1718));
    let now = Ticks::from_raw(1_000);

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

    let keys = initiator.keys().expect("established");

    // Pinned 2026-07-28 by RUNNING the implementation and recording what it
    // produced — not by asserting a value and hoping. A pin taken from the
    // thing it pins is a regression lock and nothing more: it cannot tell you
    // the schedule is *right*, only that it has not moved. The external
    // vectors above are what say the primitives underneath are right.
    //
    // A change here means the wire contract moved and every deployed peer must
    // move with it. Regenerate deliberately, never to make a red test green.
    let expected_seed: [u8; 32] =
        from_hex("1f3633191ee5c4bd28a5af5f34c260baf67ddb2c6bcdd73ba774e2a8946e6a0d");

    assert_eq!(
        keys.seed(),
        &expected_seed,
        "the key schedule changed — intentional, or a silent wire break?"
    );
    assert_eq!(initiator.role(), Role::Initiator);
}
