//! Replay match-key computation: canonicalizes an HTTP request (method +
//! path + sorted query, optionally a body content digest) into the string
//! key `ReplayUpstream` indexes recordings by.
//!
//! This is the sans-IO core of `replay` (folded from the former
//! `proxima-replay` crate): pure functions over bytes and strings, no I/O,
//! no async, no `Runtime`. The live-request half (`match_key_from_request`,
//! `match_key_from_request_with`) composes [`proxima_primitives::pipe::request::Request`],
//! the no_std + alloc tier request type, so it compiles at the `alloc` tier.
//! The recorded-request half (`match_key_from_recording`) takes a
//! [`crate::event::RequestHeader`] and stays behind `feature = "std"` —
//! preserving the tier boundary the former standalone `proxima-replay` crate
//! had (its `proxima-recording-core` dependency was std-only there); it takes
//! no new std-only crate dependency now that both live in this crate. The
//! std cassette-loading adapter in `mod.rs` (file I/O, `Runtime`, the
//! `Pipe`/`PipeFactory` surface) is the thin consumer of this module, not a
//! dependency of it.

use alloc::string::String;

use bytes::Bytes;
use smallvec::SmallVec;

#[cfg(feature = "std")]
use crate::event::RequestHeader;
use proxima_primitives::pipe::request::Request;

/// What participates in the replay match key. Method + path + sorted query
/// always do; `include_body` adds a digest of the request payload so two
/// same-path requests with different bodies (the LLM POST case) resolve to
/// distinct recordings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MatchSpec {
    pub include_body: bool,
}

pub(crate) const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a fold step, exposed crate-internally so the std cassette-loading
/// adapter (`lib.rs::index_recording`) can fold a request body streamed in
/// as multiple `RequestChunk` events into the same running digest that
/// `content_digest` computes for an already-assembled request.
pub(crate) fn fnv1a64_fold(mut state: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        state ^= u64::from(byte);
        state = state.wrapping_mul(FNV_PRIME);
    }
    state
}

/// Stable 64-bit content digest (FNV-1a). Used for the body component of
/// match keys and for record-time divergence checks; NOT collision-resistant
/// against adversaries — cassette inputs are trusted test data.
#[must_use]
pub fn content_digest(chunks: &[&[u8]]) -> u64 {
    chunks
        .iter()
        .fold(FNV_OFFSET_BASIS, |state, chunk| fnv1a64_fold(state, chunk))
}

pub(crate) fn finish_match_key(
    base_key: String,
    body_digest: u64,
    match_spec: MatchSpec,
) -> String {
    if match_spec.include_body {
        alloc::format!("{base_key}#b={body_digest:016x}")
    } else {
        base_key
    }
}

pub(crate) fn match_key_from_request(request: &Request<Bytes>) -> String {
    let pairs: SmallVec<[(&[u8], &[u8]); 16]> = request
        .query
        .iter()
        .map(|(name, value)| (name.as_ref(), value.as_ref()))
        .collect();
    write_match_key(request.method.as_bytes(), request.path.as_ref(), pairs)
}

/// The match key a live request resolves to under `match_spec`. Public so
/// recorders (the cassette tee) can guard against key collisions with the
/// exact keying replay will use.
#[must_use]
pub fn match_key_from_request_with(request: &Request<Bytes>, match_spec: MatchSpec) -> String {
    let base = match_key_from_request(request);
    finish_match_key(base, content_digest(&[&request.payload]), match_spec)
}

#[cfg(feature = "std")]
pub(crate) fn match_key_from_recording(header: &RequestHeader) -> String {
    let pairs: SmallVec<[(&[u8], &[u8]); 16]> = header
        .query
        .iter()
        .map(|(name, value)| (name.as_bytes(), value.as_bytes()))
        .collect();
    write_match_key(header.method.as_bytes(), header.path.as_bytes(), pairs)
}

fn write_match_key(
    method: &[u8],
    path: &[u8],
    mut pairs: SmallVec<[(&[u8], &[u8]); 16]>,
) -> String {
    // sort once; typical http requests have < 16 query params so the
    // smallvec stays inline.
    pairs.sort_by(|left, right| left.0.cmp(right.0));
    let mut estimated = method.len() + 1 + path.len() + 1;
    for (name, value) in &pairs {
        estimated += name.len() + 1 + value.len() + 1;
    }
    let mut output = String::with_capacity(estimated);
    for &byte in method {
        output.push(byte.to_ascii_uppercase() as char);
    }
    output.push(' ');
    output.push_str(&String::from_utf8_lossy(path));
    output.push('?');
    let mut first = true;
    for (name, value) in pairs {
        if !first {
            output.push('&');
        }
        first = false;
        output.push_str(&String::from_utf8_lossy(name));
        output.push('=');
        output.push_str(&String::from_utf8_lossy(value));
    }
    output
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn content_digest_is_order_sensitive_within_a_chunk_but_stable_across_calls() {
        let first = content_digest(&[b"hello", b"world"]);
        let second = content_digest(&[b"hello", b"world"]);
        let different = content_digest(&[b"world", b"hello"]);
        assert_eq!(first, second, "same input, same digest");
        assert_ne!(first, different, "chunk order changes the digest");
    }

    #[test]
    fn finish_match_key_appends_body_digest_only_when_configured() {
        let with_body = finish_match_key(
            "GET /x?".to_string(),
            0xdead_beef,
            MatchSpec { include_body: true },
        );
        let without_body = finish_match_key(
            "GET /x?".to_string(),
            0xdead_beef,
            MatchSpec {
                include_body: false,
            },
        );
        assert_eq!(with_body, "GET /x?#b=00000000deadbeef");
        assert_eq!(without_body, "GET /x?");
    }

    #[test]
    fn write_match_key_uppercases_method_and_sorts_query_params() {
        let pairs: SmallVec<[(&[u8], &[u8]); 16]> = SmallVec::from_vec(vec![
            (b"z".as_slice(), b"1".as_slice()),
            (b"a".as_slice(), b"2".as_slice()),
        ]);
        let key = write_match_key(b"post", b"/v1/chat", pairs);
        assert_eq!(key, "POST /v1/chat?a=2&z=1");
    }
}
