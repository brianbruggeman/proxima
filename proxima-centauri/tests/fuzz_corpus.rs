//! A deterministic fuzz sweep over both parsers.
//!
//! `cargo-fuzz` needs nightly, and a target that only compiles on a toolchain
//! CI does not run is a target that never runs. This is the same discipline in
//! a form that executes on every commit: a fixed-seed PRNG drives a large
//! corpus through the wire surface, so a crash is reproducible from the seed
//! alone rather than from a saved artifact.
//!
//! It trades coverage-guidance for reproducibility and reach — a real fuzzer
//! explores far better, and this one runs. Both are worth having; only one of
//! them is here today, and the trade is stated rather than glossed.
//!
//! The property under test is the one a wire-facing parser must have:
//! **no input causes a panic.** Rust's slicing and integer arithmetic turn a
//! bounds mistake into an abort rather than a memory-safety hole, so on this
//! surface a panic IS the vulnerability class — a remote denial of service.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use proxima_centauri::cookie::{CookieSecret, examine};
use proxima_centauri::esp::OVERHEAD;
use proxima_centauri::{ChildSa, Entropy32, EspSpi, Handshake, IkeSpi, Role};
use proxima_clock::ticks::Ticks;

const PSK: [u8; 32] = [0xAB; 32];
const ITERATIONS: usize = 20_000;
const MAX_LEN: usize = 300;

/// xorshift64*, so the corpus is identical on every host and every run: a
/// failure is reproducible from the seed, with no artifact to lose.
struct Corpus {
    state: u64,
}

impl Corpus {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn fill(&mut self, buffer: &mut [u8]) -> usize {
        let len = (self.next() as usize) % (MAX_LEN + 1);
        for slot in buffer[..len].iter_mut() {
            *slot = (self.next() & 0xFF) as u8;
        }
        len
    }

    /// Bias a slice toward structurally-plausible messages: pure noise almost
    /// never reaches past the first length check, so half the corpus starts
    /// from a real message and mutates it.
    fn mutate(&mut self, template: &[u8], buffer: &mut [u8]) -> usize {
        let len = template.len().min(buffer.len());
        buffer[..len].copy_from_slice(&template[..len]);
        let edits = (self.next() as usize) % 8;
        for _ in 0..edits {
            let at = (self.next() as usize) % len;
            buffer[at] ^= (self.next() & 0xFF) as u8;
        }
        // sometimes truncate, since a short read is its own class
        if self.next().is_multiple_of(4) {
            (self.next() as usize) % (len + 1)
        } else {
            len
        }
    }
}

fn an_sa_init() -> [u8; 92] {
    let mut initiator = Handshake::initiator(PSK, IkeSpi::new(7));
    let _ = initiator
        .step(&[], Some(Entropy32::new([0x11; 32])), Ticks::from_raw(1))
        .unwrap();
    let mut message = [0u8; 92];
    message.copy_from_slice(initiator.outbound());
    message
}

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
fn the_handshake_parser_survives_the_corpus() {
    let mut corpus = Corpus::new(0x1234_5678_9abc_def0);
    let template = an_sa_init();
    let mut buffer = [0u8; MAX_LEN];

    for iteration in 0..ITERATIONS {
        let len = if iteration % 2 == 0 {
            corpus.fill(&mut buffer)
        } else {
            corpus.mutate(&template, &mut buffer)
        };

        let mut responder = Handshake::responder(PSK, IkeSpi::new(2));
        // any Ok or Err is acceptable; a panic is not
        let _ = responder.step(
            &buffer[..len],
            Some(Entropy32::new([0x22; 32])),
            Ticks::from_raw(1),
        );
    }
}

#[test]
fn the_aead_opener_survives_the_corpus() {
    let mut corpus = Corpus::new(0x0fed_cba9_8765_4321);
    let (mut sender, _) = agreed_pair();
    let mut sealed = [0u8; 64 + OVERHEAD];
    let template_len = sender.seal(&mut sealed, 64).unwrap();
    let template = sealed;
    let mut buffer = [0u8; MAX_LEN];

    for iteration in 0..ITERATIONS {
        let len = if iteration % 2 == 0 {
            corpus.fill(&mut buffer)
        } else {
            corpus.mutate(&template[..template_len], &mut buffer)
        };

        let (_, mut receiver) = agreed_pair();
        let _ = receiver.open(&mut buffer[..len]);
    }
}

#[test]
fn the_cookie_examiner_survives_the_corpus() {
    let mut corpus = Corpus::new(0x5555_aaaa_3333_cccc);
    let secret = CookieSecret::new([0x77; 32]);
    let template = an_sa_init();
    let mut buffer = [0u8; MAX_LEN];

    for iteration in 0..ITERATIONS {
        let len = if iteration % 2 == 0 {
            corpus.fill(&mut buffer)
        } else {
            corpus.mutate(&template, &mut buffer)
        };

        let _ = examine(&secret, b"198.51.100.7", &buffer[..len]);
    }
}
