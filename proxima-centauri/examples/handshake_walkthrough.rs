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

    let mut initiator = Handshake::initiator(PSK, INITIATOR_SPI);
    let mut responder = Handshake::responder(PSK, RESPONDER_SPI);
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
    step_banner("an illegal transition is refused, not ignored");

    match initiator.step(&response, None, now) {
        Err(CentauriError::InvalidTransition { expected, found }) => {
            println!("  stepping an established handshake -> expected {expected}, found {found}");
        }
        other => panic!("expected an InvalidTransition, got {other:?}"),
    }

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
    println!("every legal transition exercised, both roles, plus the refusals.");
    println!("no sockets, no clock, no RNG reached for — all of it passed in.");
}
