use alloc::sync::Arc;
use alloc::vec::Vec;
use portable_atomic::{AtomicU64, Ordering};

use bytes::Bytes;
pub use crate::pipe::capabilities::BytePayload;
use crate::pipe::when::When;
use serde::{Deserialize, Serialize};

use crate::pipe::request::{Request, Response};

/// The config-expressible seeded mutation kind set. Each kind is a pure
/// function of `(seed, call_index, input bytes)` — same inputs, same output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Mutation {
    /// Flip `bits` randomly-chosen bit positions in the payload. A zero-length
    /// payload or `bits == 0` is returned unchanged.
    BitFlip {
        /// How many bit positions to flip per application.
        bits: u32,
    },
    /// Truncate the payload to a seeded length in `[0, len)`. Already-empty
    /// payloads are returned unchanged.
    Truncate,
    /// Duplicate a seeded contiguous slice of the payload, inserting the copy
    /// immediately after it (grows the payload). Empty payloads are unchanged.
    Duplicate,
}

impl Mutation {
    /// Deterministically mutate `input` for call `call_index`. Pure in
    /// `(seed, call_index, input)`: the rng is seeded per call so the result
    /// never depends on prior calls, ordering, or any global state.
    #[must_use]
    pub fn mutate(&self, seed: u64, call_index: u64, input: &[u8]) -> Bytes {
        if input.is_empty() {
            return Bytes::copy_from_slice(input);
        }
        let mut rng = fastrand::Rng::with_seed(seed.wrapping_add(call_index));
        match self {
            Mutation::BitFlip { bits } => bit_flip(&mut rng, input, *bits),
            Mutation::Truncate => truncate(&mut rng, input),
            Mutation::Duplicate => duplicate(&mut rng, input),
        }
    }
}

fn bit_flip(rng: &mut fastrand::Rng, input: &[u8], bits: u32) -> Bytes {
    let mut buffer = input.to_vec();
    let total_bits = (buffer.len() as u64).saturating_mul(8);
    for _ in 0..bits {
        let position = rng.u64(0..total_bits);
        let byte_index = (position / 8) as usize;
        let bit = (position % 8) as u8;
        buffer[byte_index] ^= 1 << bit;
    }
    Bytes::from(buffer)
}

fn truncate(rng: &mut fastrand::Rng, input: &[u8]) -> Bytes {
    let new_len = rng.usize(0..input.len());
    Bytes::copy_from_slice(&input[..new_len])
}

fn duplicate(rng: &mut fastrand::Rng, input: &[u8]) -> Bytes {
    let start = rng.usize(0..input.len());
    let end = rng.usize(start + 1..=input.len());
    let mut buffer = Vec::with_capacity(input.len() + (end - start));
    buffer.extend_from_slice(&input[..end]);
    buffer.extend_from_slice(&input[start..end]);
    buffer.extend_from_slice(&input[end..]);
    Bytes::from(buffer)
}

/// A seeded mutation as a `Transform` op. Carries a per-op call counter so
/// successive applications advance the deterministic mutation sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutateOp {
    #[serde(flatten)]
    mutation: Mutation,
    seed: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    when: Option<When>,
    #[serde(skip)]
    calls: Arc<AtomicU64>,
}

impl PartialEq for MutateOp {
    fn eq(&self, other: &Self) -> bool {
        self.mutation == other.mutation && self.seed == other.seed && self.when == other.when
    }
}

impl MutateOp {
    /// Build a mutation op from a kind and a seed, starting its call counter at zero.
    #[must_use]
    pub fn new(mutation: Mutation, seed: u64) -> Self {
        Self {
            mutation,
            seed,
            when: None,
            calls: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Gate the mutation behind a seeded stochastic [`When`].
    #[must_use]
    pub fn with_when(mut self, when: When) -> Self {
        self.when = Some(when);
        self
    }

    #[must_use]
    pub fn mutation(&self) -> &Mutation {
        &self.mutation
    }

    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Apply this op to a byte payload, advancing the call counter.
    pub fn apply_to<Payload: BytePayload>(&self, mut payload: Payload) -> Payload {
        let index = self.calls.fetch_add(1, Ordering::Relaxed);
        if let Some(gate) = self.when
            && !gate.fires(index)
        {
            return payload;
        }
        let mutated = self.mutation.mutate(self.seed, index, payload.bytes());
        payload.set_bytes(mutated);
        payload
    }
}

// ── HTTP instantiation — Request/Response present their body bytes ───────────

impl BytePayload for Request<Bytes> {
    fn set_bytes(&mut self, bytes: Bytes) {
        self.payload = bytes;
    }

    fn bytes(&self) -> &[u8] {
        &self.payload
    }
}

impl BytePayload for Response<Bytes> {
    fn set_bytes(&mut self, bytes: Bytes) {
        self.payload = bytes;
    }

    fn bytes(&self) -> &[u8] {
        &self.payload
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const SAMPLE: &[u8] = b"the quick brown fox jumps over the lazy dog";

    #[test]
    fn same_seed_and_index_yields_identical_bytes() {
        let mutation = Mutation::BitFlip { bits: 3 };
        let first = mutation.mutate(0xABCD, 7, SAMPLE);
        let second = mutation.mutate(0xABCD, 7, SAMPLE);
        assert_eq!(
            first, second,
            "same (seed, index) reproduces the same mutation"
        );
    }

    #[test]
    fn bit_flip_changes_only_a_bounded_number_of_bits() {
        let mutated = Mutation::BitFlip { bits: 2 }.mutate(1, 0, SAMPLE);
        assert_eq!(mutated.len(), SAMPLE.len(), "bit flip preserves length");
        let differing_bits: u32 = SAMPLE
            .iter()
            .zip(mutated.iter())
            .map(|(original, changed)| (original ^ changed).count_ones())
            .sum();
        assert!(differing_bits >= 1, "at least one bit flips");
        assert!(differing_bits <= 2, "no more than `bits` flips");
    }

    #[test]
    fn truncate_produces_a_strictly_shorter_payload() {
        let mutated = Mutation::Truncate.mutate(5, 0, SAMPLE);
        assert!(
            mutated.len() < SAMPLE.len(),
            "truncate shortens the payload"
        );
        assert_eq!(
            &mutated[..],
            &SAMPLE[..mutated.len()],
            "truncate keeps a prefix"
        );
    }

    #[test]
    fn duplicate_grows_the_payload() {
        let mutated = Mutation::Duplicate.mutate(9, 0, SAMPLE);
        assert!(mutated.len() > SAMPLE.len(), "duplicate grows the payload");
    }

    #[test]
    fn empty_payload_is_returned_unchanged_for_every_kind() {
        for mutation in [
            Mutation::BitFlip { bits: 4 },
            Mutation::Truncate,
            Mutation::Duplicate,
        ] {
            let mutated = mutation.mutate(3, 0, b"");
            assert!(
                mutated.is_empty(),
                "empty input stays empty under {mutation:?}"
            );
        }
    }

    #[test]
    fn distinct_indices_diverge() {
        let mutation = Mutation::BitFlip { bits: 4 };
        let early = mutation.mutate(42, 0, SAMPLE);
        let late = mutation.mutate(42, 1, SAMPLE);
        assert_ne!(
            early, late,
            "different call indices walk different mutations"
        );
    }

    #[test]
    fn config_builder_round_trip_parity() {
        let op = MutateOp::new(Mutation::BitFlip { bits: 5 }, 0xC0FFEE);
        let json = serde_json::to_value(&op).expect("serialize");
        let parsed: MutateOp = serde_json::from_value(json.clone()).expect("deserialize");
        assert_eq!(
            parsed, op,
            "serde round-trip is lossless (ignoring live counter)"
        );
        assert_eq!(
            json,
            serde_json::json!({"kind": "bit_flip", "bits": 5, "seed": 0xC0FFEE_u64}),
            "mutation flattens into a single tagged op object"
        );
    }

    #[test]
    fn truncate_and_duplicate_round_trip_through_config() {
        for op in [
            MutateOp::new(Mutation::Truncate, 11),
            MutateOp::new(Mutation::Duplicate, 12),
        ] {
            let json = serde_json::to_value(&op).expect("serialize");
            let parsed: MutateOp = serde_json::from_value(json).expect("deserialize");
            assert_eq!(parsed, op, "every mutation kind round-trips through config");
        }
    }

    #[test]
    fn http_request_body_is_mutated_in_place() {
        let request = Request::builder()
            .method("POST")
            .path("/")
            .body(Bytes::from_static(SAMPLE))
            .build()
            .expect("builder");
        let mutated = MutateOp::new(Mutation::Truncate, 7).apply_to(request);
        assert!(
            mutated.payload.len() < SAMPLE.len(),
            "request body was truncated"
        );
    }

    #[test]
    fn http_response_body_is_mutated_in_place() {
        let response = Response::ok(Bytes::from_static(SAMPLE));
        let mutated = MutateOp::new(Mutation::BitFlip { bits: 3 }, 7).apply_to(response);
        assert_eq!(
            mutated.payload.len(),
            SAMPLE.len(),
            "bit flip preserves response length"
        );
        assert_ne!(&mutated.payload[..], SAMPLE, "response body was corrupted");
    }

    #[test]
    fn applying_an_op_twice_advances_the_sequence() {
        let op = MutateOp::new(Mutation::BitFlip { bits: 2 }, 99);
        let once = op.apply_to(Response::ok(Bytes::from_static(SAMPLE)));
        let twice = op.apply_to(Response::ok(once.payload.clone()));

        let direct_first = Mutation::BitFlip { bits: 2 }.mutate(99, 0, SAMPLE);
        let direct_second = Mutation::BitFlip { bits: 2 }.mutate(99, 1, &direct_first);
        assert_eq!(
            &once.payload[..],
            &direct_first[..],
            "first apply uses index 0"
        );
        assert_eq!(
            &twice.payload[..],
            &direct_second[..],
            "second apply uses index 1"
        );
    }

    #[test]
    fn gated_op_mutates_only_on_a_fire() {
        let always =
            MutateOp::new(Mutation::BitFlip { bits: 3 }, 7).with_when(When::prob(1.0).seed(1));
        let never =
            MutateOp::new(Mutation::BitFlip { bits: 3 }, 7).with_when(When::prob(0.0).seed(1));

        let corrupted = always.apply_to(Response::ok(Bytes::from_static(SAMPLE)));
        assert_ne!(
            &corrupted.payload[..],
            SAMPLE,
            "a fired gate corrupts the body"
        );

        let untouched = never.apply_to(Response::ok(Bytes::from_static(SAMPLE)));
        assert_eq!(
            &untouched.payload[..],
            SAMPLE,
            "a never-firing gate leaves the body untouched"
        );
    }

    #[test]
    fn gated_op_advances_the_counter_even_on_a_miss() {
        let op = MutateOp::new(Mutation::BitFlip { bits: 3 }, 7).with_when(When::prob(1.0).seed(1));
        let first = op.apply_to(Response::ok(Bytes::from_static(SAMPLE)));
        let second = op.apply_to(Response::ok(Bytes::from_static(SAMPLE)));
        let at_zero = Mutation::BitFlip { bits: 3 }.mutate(7, 0, SAMPLE);
        let at_one = Mutation::BitFlip { bits: 3 }.mutate(7, 1, SAMPLE);
        assert_eq!(
            &first.payload[..],
            &at_zero[..],
            "first apply walks index 0"
        );
        assert_eq!(
            &second.payload[..],
            &at_one[..],
            "second apply walks index 1"
        );
    }

    #[derive(Clone, PartialEq, Debug)]
    struct Blob(Vec<u8>);

    impl BytePayload for Blob {
        fn set_bytes(&mut self, bytes: Bytes) {
            self.0 = bytes.to_vec();
        }

        fn bytes(&self) -> &[u8] {
            &self.0
        }
    }

    #[test]
    fn mutation_is_generic_over_a_non_http_byte_payload() {
        let blob = MutateOp::new(Mutation::Duplicate, 0x5EED).apply_to(Blob(SAMPLE.to_vec()));
        assert!(
            blob.0.len() > SAMPLE.len(),
            "non-http blob grows under Duplicate"
        );

        let again = MutateOp::new(Mutation::Duplicate, 0x5EED).apply_to(Blob(SAMPLE.to_vec()));
        assert_eq!(blob, again, "same seed reproduces the same mutated blob");
    }
}
