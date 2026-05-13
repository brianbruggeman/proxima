//! Fuzz — the "prove it holds" rung: throw structured-random and
//! hand-picked adversarial bytes at a codec's parser and assert two
//! invariants hold for every input — it never panics, and every valid
//! frame it builds decodes back to itself.
//!
//! Builds on: `proxima-codec`'s `LengthDelimitedCodec` — the `[u32 BE
//! len][payload]` framer defined by the same crate's `FrameCodec` trait,
//! the shape an H1/H2/H3 listener hands raw, untrusted socket bytes to.
//!
//! Run:
//!     cargo run --example fuzz

use proxima_codec::{FrameCodec, FrameLimits, LengthDelimitedCodec};

const SEED: u64 = 0x5EED_C0DE_1234_5678;
const NO_PANIC_RANDOM_COUNT: usize = 4096;
const NO_PANIC_MAX_INPUT_LEN: usize = 256;
const ROUND_TRIP_RANDOM_COUNT: usize = 2048;
const ROUND_TRIP_MAX_PAYLOAD_LEN: usize = 512;

fn main() {
    let no_panic_total = no_panic_sweep();
    let round_trip_total = round_trip_sweep();

    println!(
        "fuzz: codec under test = proxima-codec::LengthDelimitedCodec (parse_frame / encode_frame)"
    );
    println!("fuzz: seed = {SEED:#018x} (deterministic, no real randomness)");
    println!(
        "fuzz: no-panic   — {no_panic_total} garbage/edge-case inputs fed to parse_frame, 0 panics"
    );
    println!(
        "fuzz: round-trip — {round_trip_total} valid frames through encode_frame -> parse_frame, 0 mismatches"
    );
}

// a real panic here would abort the process before this function returns —
// the sweep finishing, with every draw accounted for, is itself the proof.
fn no_panic_sweep() -> usize {
    let codec = LengthDelimitedCodec::new(FrameLimits::new(64, true));
    let edge_cases = fixed_edge_cases();
    let mut processed = 0usize;

    for input in &edge_cases {
        feed_and_prove_no_panic(&codec, input);
        processed += 1;
    }

    let mut generator = Xorshift64::new(SEED);
    for _ in 0..NO_PANIC_RANDOM_COUNT {
        let length = generator.next_below(NO_PANIC_MAX_INPUT_LEN + 1);
        let input = generator.next_bytes(length);
        feed_and_prove_no_panic(&codec, &input);
        processed += 1;
    }

    assert_eq!(
        processed,
        edge_cases.len() + NO_PANIC_RANDOM_COUNT,
        "every planned draw must have run — a panic partway through would have aborted first"
    );
    processed
}

fn feed_and_prove_no_panic(codec: &LengthDelimitedCodec, input: &[u8]) {
    match codec.parse_frame(input) {
        Ok((_frame, consumed)) => assert!(
            consumed <= input.len(),
            "parse must never claim to consume more bytes than it was given"
        ),
        Err(_rejected) => {}
    }
}

// hand-picked bytes chosen to land on each named branch: too short for a
// header, a complete header with nothing behind it, a declared length that
// overflows the codec's cap (two ways: header alone, header at u32::MAX), a
// zero-length frame under a reject-zero policy, a declared length exactly
// at the cap boundary, and a payload that is not valid utf-8.
fn fixed_edge_cases() -> Vec<Vec<u8>> {
    let mut boundary_at_cap = vec![0x00, 0x00, 0x00, 0x40];
    boundary_at_cap.extend(std::iter::repeat_n(0xab, 64));

    vec![
        vec![],
        vec![0x00],
        vec![0x00, 0x00],
        vec![0x00, 0x00, 0x00],
        vec![0x00, 0x00, 0x00, 0x00],
        vec![0x00, 0x00, 0x00, 0x05, 0x01, 0x02, 0x03],
        vec![0xff, 0xff, 0xff, 0xff],
        vec![0x00, 0x00, 0x00, 0xc8],
        vec![0x00, 0x00, 0x00, 0x03, 0xff, 0xfe, 0xfd],
        boundary_at_cap,
    ]
}

// every generated payload is valid input to encode_frame; parse_frame on
// the codec's own encoded output must yield back the exact same payload.
fn round_trip_sweep() -> usize {
    let codec = LengthDelimitedCodec::default();
    let mut generator = Xorshift64::new(SEED.wrapping_add(1));
    let mut verified = 0usize;

    for length in [0usize, 1, 4, 4096] {
        let payload = generator.next_bytes(length);
        verify_round_trip(&codec, &payload);
        verified += 1;
    }

    for _ in 0..ROUND_TRIP_RANDOM_COUNT {
        let length = generator.next_below(ROUND_TRIP_MAX_PAYLOAD_LEN + 1);
        let payload = generator.next_bytes(length);
        verify_round_trip(&codec, &payload);
        verified += 1;
    }

    verified
}

fn verify_round_trip(codec: &LengthDelimitedCodec, payload: &[u8]) {
    let mut encoded = Vec::new();
    let encode_outcome = codec.encode_frame(&payload, &mut encoded);
    assert!(
        encode_outcome.is_ok(),
        "encode must not fail for a payload within limits"
    );

    let parse_outcome = codec.parse_frame(&encoded);
    assert!(
        parse_outcome.is_ok(),
        "parse_frame must succeed on the codec's own encoded output"
    );
    if let Ok((frame, consumed)) = parse_outcome {
        assert_eq!(frame, payload, "decode(encode(payload)) must equal payload");
        assert_eq!(
            consumed,
            encoded.len(),
            "parse must consume exactly what encode wrote"
        );
    }
}

/// Fixed-seed xorshift64-star generator. Deterministic across runs — never
/// real randomness — so every fuzz run reproduces byte-for-byte.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        value.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        (self.next_u64() as usize) % bound
    }

    fn next_bytes(&mut self, length: usize) -> Vec<u8> {
        (0..length)
            .map(|_| (self.next_u64() & 0xff) as u8)
            .collect()
    }
}
