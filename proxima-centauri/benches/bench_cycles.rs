//! Instruction counts for the hot paths, via callgrind.
//!
//! ```text
//! cargo install iai-callgrind-runner --version 0.16.1
//! cargo bench -p proxima-centauri --bench bench_cycles
//! ```
//!
//! Principle 11 asks for cycle counts on ultra-hot paths "because throughput is
//! noisy and instruction count isn't". The criterion arms in the sibling bench
//! files carry a 1.1–1.5% run-to-run spread on this host, which is wider than
//! several of the deltas recorded in the discipline log — those deltas are real
//! only because they were larger than the noise, not because criterion resolved
//! them. Callgrind counts instructions deterministically, so a 0.5% change is
//! visible and reproducible.
//!
//! Requires a working callgrind. valgrind installs on macOS/aarch64 but
//! callgrind dies with SIGSEGV there (verified 2026-07-28), so this bench
//! compiles on that host and runs on the Linux leg of CI. The gate script
//! SKIPS it when callgrind is unusable rather than reporting a missing tool as
//! a passing cell — a skipped cell is announced, never silently counted.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;

use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use proxima_centauri::esp::{HEADER_LEN, OVERHEAD};
use proxima_centauri::{ChildSa, CounterDrbg, Entropy32, EspSpi, Handshake, IkeSpi, Role};
use proxima_clock::ticks::Ticks;

const PSK: [u8; 32] = [0xAB; 32];
const INITIATOR_SPI: IkeSpi = IkeSpi::new(0x0102_0304_0506_0708);
const RESPONDER_SPI: IkeSpi = IkeSpi::new(0x1112_1314_1516_1718);

fn now() -> Ticks {
    Ticks::from_raw(1_000)
}

/// Two child SAs that agreed through a real handshake.
fn agreed_pair() -> (ChildSa, ChildSa) {
    let mut initiator = Handshake::initiator(PSK, INITIATOR_SPI);
    let mut responder = Handshake::responder(PSK, RESPONDER_SPI);

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

// The 100% path: one outbound datagram.
#[library_benchmark]
fn seal_datagram() -> usize {
    let (mut sender, _) = agreed_pair();
    let mut buffer = [0u8; 1200 + OVERHEAD];
    buffer[HEADER_LEN..HEADER_LEN + 1200].fill(0x5A);

    black_box(sender.seal(black_box(&mut buffer), 1200).unwrap())
}

// The 100% path, inbound — including the replay-window probe.
#[library_benchmark]
fn open_datagram() -> usize {
    let (mut sender, mut receiver) = agreed_pair();
    let mut buffer = [0u8; 1200 + OVERHEAD];
    buffer[HEADER_LEN..HEADER_LEN + 1200].fill(0x5A);
    let len = sender.seal(&mut buffer, 1200).unwrap();

    black_box(receiver.open(black_box(&mut buffer[..len])).unwrap())
}

// Rejecting a replay must stay far cheaper than opening one, or a flood is
// an amplification vector. Instruction count is the honest way to state that
// ratio; wall-clock at this scale is mostly noise.
#[library_benchmark]
fn reject_replay() -> bool {
    let (mut sender, mut receiver) = agreed_pair();
    let mut buffer = [0u8; 1200 + OVERHEAD];
    let len = sender.seal(&mut buffer, 1200).unwrap();
    let replayed = buffer;
    let _ = receiver.open(&mut buffer[..len]).unwrap();

    let mut again = replayed;
    black_box(receiver.open(black_box(&mut again[..len])).is_err())
}

// One draw from the DRBG — called once per handshake step.
#[library_benchmark]
fn entropy_draw() -> [u8; 32] {
    let source = CounterDrbg::new([0x33; 32]);
    *black_box(source.draw().unwrap()).expose()
}

// The setup path, for the amortisation story the e2e bench tells.
#[library_benchmark]
fn full_handshake() -> bool {
    let mut initiator = Handshake::initiator(PSK, INITIATOR_SPI);
    let mut responder = Handshake::responder(PSK, RESPONDER_SPI);

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

    black_box(initiator.keys().is_some())
}

library_benchmark_group!(
    name = hot_paths;
    benchmarks = seal_datagram, open_datagram, reject_replay, entropy_draw, full_handshake
);

main!(library_benchmark_groups = hot_paths);
