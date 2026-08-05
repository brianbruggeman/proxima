//! Every legal transition of the Centauri handshake and child SA, driven end
//! to end with commentary — the runnable teaching surface principle 11
//! requires for a state machine.
//!
//! ```text
//! cargo run -p proxima-centauri --example handshake_walkthrough
//! ```
//!
//! Nothing here is a mock. The two peers are the real state machines, fed by
//! the real entropy sources; the only thing an integration would add is a
//! socket between them. That is the whole point of sans-IO: the transport is
//! the caller's business, and the protocol runs identically without one.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]

use proxima_centauri::esp::{HEADER_LEN, OVERHEAD};
use proxima_centauri::{
    CentauriError, ChildSa, CounterDrbg, Entropy32, EntropyCell, EspSpi, FixedSequence, Handshake,
    IkeSpi, Progress, Role,
};
use proxima_clock::ticks::Ticks;

/// Both peers must already share this; a sans-IO state machine cannot fetch a
/// key, so it arrives as a constructor argument.
const PSK: [u8; 32] = [0xAB; 32];

const INITIATOR_SPI: IkeSpi = IkeSpi::new(0x0102_0304_0506_0708);
const RESPONDER_SPI: IkeSpi = IkeSpi::new(0x1112_1314_1516_1718);

fn step_banner(title: &str) {
    println!(
        "\n── {title} {}",
        "─".repeat(60usize.saturating_sub(title.len()))
    );
}

fn main() {
    println!("proxima-centauri — handshake walkthrough");
    println!("every legal transition, both roles, then the data path\n");

    // ── the three entropy sources, all the same pipe shape ────────────────
    step_banner("entropy sources are interchangeable");

    let scripted = [[0x11u8; 32], [0x22u8; 32]];
    let scripted_source = FixedSequence::new(&scripted);
    println!(
        "FixedSequence  first draw = {:02x?}…  (a test says what the next draw is)",
        &scripted_source.draw().unwrap().expose()[..4]
    );

    let drbg = CounterDrbg::new([0x33; 32]);
    let first = drbg.draw().unwrap();
    let second = drbg.draw().unwrap();
    println!(
        "CounterDrbg    two draws differ: {:02x?}… vs {:02x?}…  (a repeat would be nonce reuse)",
        &first.expose()[..4],
        &second.expose()[..4]
    );

    let cell = EntropyCell::new();
    cell.set([0x44; 32]).unwrap();
    let claimed = cell.draw().unwrap();
    println!(
        "EntropyCell    claimed {:02x?}…, and a second draw now fails: {}",
        &claimed.expose()[..4],
        cell.draw().is_err()
    );
    println!("               take-once — the cell is emptied by the draw, not re-read");

    // ── initiator: Initial → AwaitingResponse ─────────────────────────────
    step_banner("transition 1/4 — initiator sends SA_INIT");

    // the identity is a per-session constant like the PSK, so it is attached
    // at construction rather than mid-flow
    let mut initiator = Handshake::initiator(PSK, INITIATOR_SPI)
        .with_identity(b"peer-a")
        .expect("identity fits");
    let mut responder = Handshake::responder(PSK, RESPONDER_SPI)
        .with_identity(b"peer-b")
        .expect("identity fits");
    let now = Ticks::from_raw(1_000);

    println!(
        "initiator needs entropy for this step: {}",
        initiator.needs_entropy()
    );
    let progress = initiator
        .step(&[], Some(Entropy32::new([0x11; 32])), now)
        .expect("the initiator can always open");
    println!("  step -> {progress:?}");
    assert_eq!(progress, Progress::Advanced);

    let mut init_message = [0u8; 92];
    init_message.copy_from_slice(initiator.outbound());
    println!("  staged {} bytes to send", init_message.len());
    println!("  spi     {:02x?}", &init_message[0..8]);
    println!("  nonce   {:02x?}…", &init_message[28..32]);
    println!("  dh pub  {:02x?}…", &init_message[60..64]);

    // ── responder: NeedInput on a partial read ────────────────────────────
    step_banner("transition 2/4 — a short read yields NeedInput, no state change");

    let partial = &init_message[..40];
    let progress = responder
        .step(partial, Some(Entropy32::new([0x22; 32])), now)
        .expect("a partial message is not an error");
    println!("  fed {} of 92 bytes -> {progress:?}", partial.len());
    assert_eq!(progress, Progress::NeedInput);
    println!(
        "  still needs entropy (nothing was consumed): {}",
        responder.needs_entropy()
    );

    // ── responder: Initial → Established ──────────────────────────────────
    step_banner("transition 3/4 — responder completes on the full message");

    let progress = responder
        .step(&init_message, Some(Entropy32::new([0x22; 32])), now)
        .expect("the responder accepts a well-formed SA_INIT");
    println!("  step -> {progress:?}   (the responder finishes in one step)");
    assert_eq!(progress, Progress::Established);

    let mut response = [0u8; 92];
    response.copy_from_slice(responder.outbound());
    println!("  replied with {} bytes, and derived keys", response.len());

    // ── initiator: AwaitingResponse → Established ─────────────────────────
    step_banner("transition 4/4 — initiator completes; no entropy needed");

    println!(
        "initiator needs entropy for this step: {}",
        initiator.needs_entropy()
    );
    let progress = initiator
        .step(&response, None, now)
        .expect("the responder's reply completes the handshake");
    println!("  step -> {progress:?}");
    assert_eq!(progress, Progress::Established);
    println!(
        "  nothing left to send: outbound is {} bytes",
        initiator.outbound().len()
    );

    // ── illegal transition ────────────────────────────────────────────────
    step_banner("a replayed SA_INIT is refused, not ignored");

    // an established handshake is not inert -- AUTH is legal here -- so the
    // refusal comes from the exchange type rather than from the phase.
    match initiator.step(&response, None, now) {
        Err(CentauriError::InvalidMessage(field)) => {
            println!("  replaying SA_INIT into an established handshake -> invalid {field}");
            println!("  the phase accepts AUTH now, so the message itself is what is rejected");
        }
        other => panic!("expected InvalidMessage, got {other:?}"),
    }

    // ── AUTH: identity, on top of key agreement ───────────────────────────
    step_banner("transition 5/6 — initiator proves its identity");

    println!("  SA_INIT proved whoever derived these keys holds the PSK and the");
    println!("  DH secret. It did NOT prove who they are — that is what AUTH is for.");

    let progress = initiator
        .send_auth()
        .expect("established, so AUTH is legal");
    println!(
        "  send_auth -> {progress:?}, {} bytes staged",
        initiator.outbound().len()
    );

    let mut auth_message = [0u8; 128];
    let auth_len = initiator.outbound().len();
    auth_message[..auth_len].copy_from_slice(initiator.outbound());
    println!(
        "  identity length prefix {:02x?} makes the message self-describing",
        &auth_message[28..30]
    );

    step_banner("transition 6/6 — responder verifies it");

    let progress = responder
        .step(&auth_message[..auth_len], None, now)
        .expect("a well-formed AUTH verifies");
    println!("  step -> {progress:?}");
    assert_eq!(progress, Progress::Authenticated);
    println!(
        "  peer identity: {:?}",
        core::str::from_utf8(responder.peer_identity().expect("authenticated")).unwrap()
    );

    step_banner("a forged AUTH is refused");

    let (mut fresh_initiator, mut fresh_responder) = {
        let mut i = Handshake::initiator(PSK, INITIATOR_SPI)
            .with_identity(b"peer-a")
            .unwrap();
        let mut r = Handshake::responder(PSK, RESPONDER_SPI)
            .with_identity(b"peer-b")
            .unwrap();
        let _ = i.step(&[], Some(Entropy32::new([0x11; 32])), now).unwrap();
        let mut m = [0u8; 92];
        m.copy_from_slice(i.outbound());
        let _ = r.step(&m, Some(Entropy32::new([0x22; 32])), now).unwrap();
        let mut reply = [0u8; 92];
        reply.copy_from_slice(r.outbound());
        let _ = i.step(&reply, None, now).unwrap();
        (i, r)
    };
    let _ = fresh_initiator.send_auth().unwrap();
    let mut forged = [0u8; 128];
    let forged_len = fresh_initiator.outbound().len();
    forged[..forged_len].copy_from_slice(fresh_initiator.outbound());
    forged[18] ^= 0x01; // the role flag, which the oracle leaves unauthenticated

    match fresh_responder.step(&forged[..forged_len], None, now) {
        Err(CentauriError::AuthenticationFailed) => {
            println!("  flipped the role flag in the header -> AuthenticationFailed");
            println!("  the header is under the MAC, so it cannot be edited unnoticed");
        }
        other => panic!("expected AuthenticationFailed, got {other:?}"),
    }
    assert!(
        fresh_responder.peer_identity().is_none(),
        "no identity from a failed AUTH"
    );

    // ── mutual auth, then rekey ───────────────────────────────────────────
    step_banner("transition 7/8 — the responder proves itself in return");

    println!("  authentication is two obligations, not one: the responder has");
    println!("  verified us, and still owes its own proof before either side");
    println!("  may rekey.");

    let progress = responder.send_auth().expect("still owes its AUTH");
    println!("  responder send_auth -> {progress:?}");
    let mut back = [0u8; 128];
    let back_len = responder.outbound().len();
    back[..back_len].copy_from_slice(responder.outbound());

    let progress = initiator
        .step(&back[..back_len], None, now)
        .expect("verifies");
    println!("  initiator step -> {progress:?}");
    println!(
        "  peer identity: {:?}",
        core::str::from_utf8(initiator.peer_identity().expect("authenticated")).unwrap()
    );

    step_banner("transition 8/8 — rekey with forward secrecy");

    let before = *initiator.keys().expect("authenticated").encrypt_key();

    let progress = initiator
        .send_rekey(Some(Entropy32::new([0x31; 32])))
        .expect("mutually authenticated, so rekey is legal");
    println!(
        "  send_rekey -> {progress:?}, {} bytes staged",
        initiator.outbound().len()
    );
    let mut request = [0u8; 92];
    request.copy_from_slice(initiator.outbound());

    let progress = responder
        .step(&request, Some(Entropy32::new([0x32; 32])), now)
        .expect("responder rekeys and replies in one step");
    println!("  responder step -> {progress:?}");
    let mut reply = [0u8; 92];
    reply.copy_from_slice(responder.outbound());

    let progress = initiator
        .step(&reply, None, now)
        .expect("completes the rekey");
    println!("  initiator step -> {progress:?}");

    let after = *initiator.keys().expect("rekeyed").encrypt_key();
    println!("  keys changed: {}", before != after);
    println!(
        "  peers still agree: {}",
        initiator.keys().unwrap().encrypt_key() == responder.keys().unwrap().decrypt_key()
    );
    println!(
        "  identity survived the rekey: {:?}",
        core::str::from_utf8(responder.peer_identity().unwrap()).unwrap()
    );
    println!("  fresh DH each time, so today's keys cannot recover yesterday's traffic");
    assert_ne!(before, after);
    assert_eq!(
        initiator.keys().unwrap().encrypt_key(),
        responder.keys().unwrap().decrypt_key()
    );

    // ── the keys agree ────────────────────────────────────────────────────
    step_banner("both peers derived the same keys");

    let initiator_keys = initiator.keys().expect("established");
    let responder_keys = responder.keys().expect("established");

    println!(
        "  initiator encrypt == responder decrypt: {}",
        initiator_keys.encrypt_key() == responder_keys.decrypt_key()
    );
    println!(
        "  responder encrypt == initiator decrypt: {}",
        responder_keys.encrypt_key() == initiator_keys.decrypt_key()
    );
    println!("  keys are direction-bound, so a caller cannot reach for the wrong one");
    assert_eq!(initiator_keys.encrypt_key(), responder_keys.decrypt_key());

    // ── the data path ─────────────────────────────────────────────────────
    step_banner("data path — per-packet AEAD, in place");

    let mut sender = ChildSa::from_session(initiator_keys, Role::Initiator, EspSpi::new(0xAAAA));
    let mut receiver = ChildSa::from_session(responder_keys, Role::Responder, EspSpi::new(0xBBBB));

    let payload = b"the quick brown fox";
    let mut packet = [0u8; 128];
    packet[HEADER_LEN..HEADER_LEN + payload.len()].copy_from_slice(payload);
    println!("  plaintext sits at offset {HEADER_LEN}, and is encrypted where it lies");

    let packet_len = sender.seal(&mut packet, payload.len()).unwrap();
    println!(
        "  sealed {} bytes ({} payload + {OVERHEAD} overhead)",
        packet_len,
        payload.len()
    );
    println!("  ciphertext {:02x?}…", &packet[HEADER_LEN..HEADER_LEN + 8]);

    let keep = packet;
    let opened = receiver.open(&mut packet[..packet_len]).unwrap();
    println!(
        "  opened {opened} bytes -> {:?}",
        core::str::from_utf8(&packet[HEADER_LEN..HEADER_LEN + opened]).unwrap()
    );

    // ── replay ────────────────────────────────────────────────────────────
    step_banner("the same packet a second time is refused");

    let mut replayed = keep;
    match receiver.open(&mut replayed[..packet_len]) {
        Err(CentauriError::ReplayDetected(seq)) => {
            println!("  sequence {seq} was already seen -> ReplayDetected");
            println!("  refused on a bitmap probe, before any AEAD work");
        }
        other => panic!("expected ReplayDetected, got {other:?}"),
    }

    // ── tamper ────────────────────────────────────────────────────────────
    step_banner("editing the header fails authentication");

    let mut tampered = [0u8; 128];
    tampered[HEADER_LEN..HEADER_LEN + payload.len()].copy_from_slice(payload);
    let len = sender.seal(&mut tampered, payload.len()).unwrap();
    tampered[0] ^= 0xFF;

    match receiver.open(&mut tampered[..len]) {
        Err(CentauriError::AuthenticationFailed) => {
            println!("  flipped a byte of the SPI -> AuthenticationFailed");
            println!("  the header is associated data, so it cannot be edited unnoticed");
        }
        other => panic!("expected AuthenticationFailed, got {other:?}"),
    }

    step_banner("done");
    println!("every legal transition exercised: SA_INIT, mutual AUTH, rekey,");
    println!("the data path, and every refusal along the way.");
    println!("no sockets, no clock, no RNG reached for — all of it passed in.");
}
