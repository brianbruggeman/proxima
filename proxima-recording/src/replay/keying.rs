//! Replay match-key computation: canonicalizes an HTTP request (method +
//! path + sorted query, optionally a body content digest) into the string
//! key `ReplayUpstream` indexes recordings by.
//!
//! This is the sans-IO core of `replay` (folded from the former
//! `proxima-replay` crate): pure functions over bytes and strings, no I/O,
//! no async, no `Runtime`. The live-request half (`match_key_from_request`,
//! `match_key_from_request_with`) composes [`proxima_primitives::pipe::request::Request`],
//! the `no_std` + alloc tier request type, so it compiles at the `alloc` tier.
//! The recorded-request half (`match_key_from_recording`) takes a
//! [`crate::event::RequestHeader`] and stays behind `feature = "std"` —
//! preserving the tier boundary the former standalone `proxima-replay` crate
//! had (its `proxima-recording-core` dependency was std-only there); it takes
//! no new std-only crate dependency now that both live in this crate. The
//! std cassette-loading adapter in `mod.rs` (file I/O, `Runtime`, the
//! `Pipe`/`PipeFactory` surface) is the thin consumer of this module, not a
//! dependency of it.

use alloc::string::String;
use alloc::vec::Vec;

use bytes::Bytes;
use proxima_core::ProximaError;
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

/// Version of the request-key byte layout. Stamped into a cassette header so
/// a layout change refuses (or rebuilds) old cassettes instead of guessing.
pub const KEY_SCHEMA_VERSION: KeySchemaVersion = KeySchemaVersion(1);

const KEY_DOMAIN: &[u8] = b"pxk1";
const REQUEST_DOMAIN: &[u8] = b"pxr1";
const SPEC_DOMAIN: &[u8] = b"pxs1";
const SEP_FIELD: u8 = 0x00;
const SEP_NAME: u8 = 0x01;
const SEP_VALUE: u8 = 0x02;
const INLINE_QUERY_PAIRS: usize = 16;

/// Digest identifying one request under a [`HeaderAllowlist`]: the identity a
/// cassette entry is stored and looked up by. Fixed-width and `Copy`, so a
/// per-request lookup folds bytes and allocates nothing (guiding-principles
/// P11). Built by [`request_key`], or restored from a stamped cassette with
/// [`KeyDigest::from_raw`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KeyDigest(u64);

impl KeyDigest {
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Collision witness stored beside a [`KeyDigest`]; see [`request_digest`].
/// A distinct type from `KeyDigest` so the two cannot be swapped at the
/// comparison that decides whether a hit may be served.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestDigest(u64);

impl RequestDigest {
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Digest of a recorded or live response (status + body chunks); see
/// [`response_digest`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResponseDigest(u64);

impl ResponseDigest {
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Which byte layout [`request_key`] folded. See [`KEY_SCHEMA_VERSION`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeySchemaVersion(u16);

impl KeySchemaVersion {
    #[must_use]
    pub const fn from_raw(raw: u16) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn get(self) -> u16 {
        self.0
    }
}

/// Digest of the canonical header allowlist a key was computed under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeySpecDigest(u64);

impl KeySpecDigest {
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// The request headers that participate in [`request_key`], held lowercase,
/// sorted and deduplicated so `["Accept", "accept"]` and `["accept"]` are the
/// same spec with the same [`KeySpecDigest`]. A bare `&[String]` would make
/// the key depend on the caller's order and case; this type cannot.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HeaderAllowlist(Vec<String>);

impl HeaderAllowlist {
    /// Canonicalizes `names`.
    ///
    /// # Errors
    /// `ProximaError::Record` when a name is empty or holds a byte that is not
    /// a visible ASCII header-name character (whitespace, control bytes, `:`).
    pub fn new<'name>(names: impl IntoIterator<Item = &'name str>) -> Result<Self, ProximaError> {
        let mut canonical = Vec::new();
        for name in names {
            if !is_header_name(name) {
                return Err(ProximaError::Record(alloc::format!(
                    "invalid header allowlist name {name:?}: expected visible ASCII without ':'"
                )));
            }
            canonical.push(name.to_ascii_lowercase());
        }
        canonical.sort_unstable();
        canonical.dedup();
        Ok(Self(canonical))
    }

    #[must_use]
    pub fn names(&self) -> &[String] {
        &self.0
    }

    #[must_use]
    pub fn digest(&self) -> KeySpecDigest {
        let seed = fnv1a64_fold(FNV_OFFSET_BASIS, SPEC_DOMAIN);
        let folded = self.0.iter().fold(seed, |state, name| {
            fnv1a64_fold(fnv1a64_fold(state, name.as_bytes()), &[SEP_FIELD])
        });
        KeySpecDigest(folded)
    }
}

fn is_header_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte > 0x20 && byte < 0x7f && byte != b':')
}

fn fold_pair(state: u64, name: &[u8], value: &[u8]) -> u64 {
    let state = fnv1a64_fold(fnv1a64_fold(state, name), &[SEP_NAME]);
    fnv1a64_fold(fnv1a64_fold(state, value), &[SEP_VALUE])
}

fn fold_method(state: u64, method: &[u8]) -> u64 {
    method.iter().fold(state, |running, byte| {
        fnv1a64_fold(running, &[byte.to_ascii_uppercase()])
    })
}

fn fold_headers(state: u64, request: &Request<Bytes>, allowlist: &HeaderAllowlist) -> u64 {
    allowlist.0.iter().fold(state, |running, wanted| {
        request
            .metadata
            .iter()
            .filter(|(name, _)| name.as_ref().eq_ignore_ascii_case(wanted.as_bytes()))
            .fold(running, |inner, (_, value)| {
                fold_pair(inner, wanted.as_bytes(), value.as_ref())
            })
    })
}

fn fold_request(domain: &[u8], request: &Request<Bytes>, allowlist: &HeaderAllowlist) -> u64 {
    let mut pairs: SmallVec<[(&[u8], &[u8]); INLINE_QUERY_PAIRS]> = request
        .query
        .iter()
        .map(|(name, value)| (name.as_ref(), value.as_ref()))
        .collect();
    pairs.sort_unstable();
    let state = fnv1a64_fold(FNV_OFFSET_BASIS, domain);
    let state = fnv1a64_fold(fold_method(state, request.method.as_bytes()), &[SEP_FIELD]);
    let state = fnv1a64_fold(fnv1a64_fold(state, request.path.as_ref()), &[SEP_FIELD]);
    let state = pairs.iter().fold(state, |running, (name, value)| {
        fold_pair(running, name, value)
    });
    let state = fold_headers(fnv1a64_fold(state, &[SEP_FIELD]), request, allowlist);
    let body = content_digest(&[request.payload.as_ref()]);
    fnv1a64_fold(fnv1a64_fold(state, &[SEP_FIELD]), &body.to_le_bytes())
}

/// The identity `request` is recorded and replayed under.
///
/// Folds FNV-1a-64 (`fnv1a64_fold`) over `"pxk1"`, `METHOD 0x00`, `path 0x00`,
/// each query pair sorted by `(name, value)` as `name 0x01 value 0x02`, `0x00`,
/// each allowlisted header in allowlist order (values in request order) as
/// `lowercase-name 0x01 value 0x02`, `0x00`, then the body digest as 8 little
/// endian bytes. The separator bytes keep `a=b&c` and `a=b%26c` apart, which
/// the old text key could not.
///
/// Allocation: none for up to 16 query pairs (an inline `SmallVec` scratch).
#[must_use]
pub fn request_key(request: &Request<Bytes>, allowlist: &HeaderAllowlist) -> KeyDigest {
    KeyDigest(fold_request(KEY_DOMAIN, request, allowlist))
}

/// Second, independent fold of the same normalized request as [`request_key`]
/// (seed `"pxr1"`). A cassette entry stores both; a lookup hit whose request
/// digest differs from the live one is a key collision, never a served hit.
#[must_use]
pub fn request_digest(request: &Request<Bytes>, allowlist: &HeaderAllowlist) -> RequestDigest {
    RequestDigest(fold_request(REQUEST_DOMAIN, request, allowlist))
}

/// Digest of a response: `status.to_be_bytes()` then every body chunk, the
/// fold the cassette recorder uses to tell identical re-recordings from
/// divergent ones (`content_digest(&[&status.to_be_bytes(), &body])`).
#[must_use]
pub fn response_digest(status: u16, body: &[&[u8]]) -> ResponseDigest {
    let seed = fnv1a64_fold(FNV_OFFSET_BASIS, &status.to_be_bytes());
    ResponseDigest(
        body.iter()
            .fold(seed, |state, chunk| fnv1a64_fold(state, chunk)),
    )
}

/// Readable `METHOD /path?a=1&b=2` form of `request` for miss text and labels.
/// Cold path: allocates, and is not injective (the digest is the identity).
#[must_use]
pub fn render_request_line(request: &Request<Bytes>) -> String {
    match_key_from_request(request)
}

#[cfg(test)]
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

    const PINNED_K1: u64 = 4_370_001_267_500_638_195;
    const PINNED_K2: u64 = 2_793_989_062_046_074_694;

    use proxima_primitives::pipe::header_list::HeaderList;
    use proxima_primitives::pipe::method::Method;

    fn forecast_request(query: &[(&str, &str)], headers: &[(&str, &str)]) -> Request<Bytes> {
        let mut request = Request::builder()
            .method(Method::Get)
            .path("/v1/forecast")
            .build()
            .expect("a method and path are enough to build a request");
        request.query = query
            .iter()
            .map(|(name, value)| {
                (
                    Bytes::copy_from_slice(name.as_bytes()),
                    Bytes::copy_from_slice(value.as_bytes()),
                )
            })
            .collect::<Vec<_>>()
            .into();
        request.metadata = HeaderList::from_pairs(headers.iter().copied());
        request
    }

    fn accept_allowlist() -> HeaderAllowlist {
        HeaderAllowlist::new(["accept"]).expect("accept is a valid header name")
    }

    #[test]
    fn request_key_ignores_query_pair_order() {
        let forward = forecast_request(&[("city", "oslo"), ("units", "metric")], &[]);
        let reversed = forecast_request(&[("units", "metric"), ("city", "oslo")], &[]);
        let allowlist = accept_allowlist();
        assert_eq!(
            request_key(&forward, &allowlist),
            request_key(&reversed, &allowlist)
        );
    }

    #[test]
    fn request_key_keeps_every_value_of_a_multi_valued_query_name() {
        let allowlist = accept_allowlist();
        let both = forecast_request(&[("tag", "a"), ("tag", "b")], &[]);
        let swapped = forecast_request(&[("tag", "b"), ("tag", "a")], &[]);
        let only_first = forecast_request(&[("tag", "a")], &[]);
        assert_eq!(
            request_key(&both, &allowlist),
            request_key(&swapped, &allowlist)
        );
        assert_ne!(
            request_key(&both, &allowlist),
            request_key(&only_first, &allowlist)
        );
    }

    #[test]
    fn request_key_matches_allowlisted_header_names_case_insensitively() {
        let allowlist = accept_allowlist();
        let lower = forecast_request(&[], &[("accept", "text/csv")]);
        let upper = forecast_request(&[], &[("Accept", "text/csv")]);
        let other_value = forecast_request(&[], &[("Accept", "application/json")]);
        assert_eq!(
            request_key(&lower, &allowlist),
            request_key(&upper, &allowlist)
        );
        assert_ne!(
            request_key(&lower, &allowlist),
            request_key(&other_value, &allowlist)
        );
    }

    #[test]
    fn request_key_ignores_headers_outside_the_allowlist() {
        let allowlist = accept_allowlist();
        let bare = forecast_request(&[("city", "oslo")], &[]);
        let noisy = forecast_request(
            &[("city", "oslo")],
            &[
                ("user-agent", "curl/8.7.1"),
                ("x-request-id", "01KT4M602X24W4F7P4AZ5TRYEY"),
            ],
        );
        assert_eq!(
            request_key(&bare, &allowlist),
            request_key(&noisy, &allowlist)
        );
    }

    #[test]
    fn allowlist_digest_is_canonical_across_case_order_and_duplicates() {
        let canonical = HeaderAllowlist::new(["accept", "x-api-version"]).expect("valid names");
        let shuffled =
            HeaderAllowlist::new(["X-API-Version", "Accept", "accept"]).expect("valid names");
        let narrower = HeaderAllowlist::new(["accept"]).expect("valid names");
        assert_eq!(canonical.digest(), shuffled.digest());
        assert_eq!(canonical.names(), shuffled.names());
        assert_ne!(canonical.digest(), narrower.digest());
        assert!(
            HeaderAllowlist::new(["bad name"]).is_err(),
            "whitespace is not a header-name byte"
        );
        assert!(
            HeaderAllowlist::new([""]).is_err(),
            "an empty name can never match a header"
        );
    }

    #[test]
    fn request_key_separates_values_containing_query_delimiters() {
        let allowlist = accept_allowlist();
        let ampersand = forecast_request(&[("a", "b&c=d")], &[]);
        let split = forecast_request(&[("a", "b"), ("c", "d")], &[]);
        let equals = forecast_request(&[("a", "b=c")], &[]);
        let named = forecast_request(&[("a=b", "c")], &[]);
        assert_eq!(
            render_request_line(&ampersand),
            render_request_line(&split),
            "the text form is the ambiguous one"
        );
        assert_ne!(
            request_key(&ampersand, &allowlist),
            request_key(&split, &allowlist)
        );
        assert_ne!(
            request_key(&equals, &allowlist),
            request_key(&named, &allowlist)
        );
    }

    #[test]
    fn request_key_pins_the_forecast_keys_and_separates_header_variants() {
        let allowlist = accept_allowlist();
        let plain = forecast_request(&[("city", "oslo")], &[]);
        let csv = forecast_request(&[("city", "oslo")], &[("accept", "text/csv")]);
        let body_digest = content_digest(&[b""]).to_le_bytes();
        let mut spec = Vec::new();
        spec.extend_from_slice(b"pxk1GET\0/v1/forecast\0city\x01oslo\x02\0\0");
        spec.extend_from_slice(&body_digest);
        assert_eq!(
            request_key(&plain, &allowlist).get(),
            fnv1a64_fold(FNV_OFFSET_BASIS, &spec)
        );
        let mut csv_spec = Vec::new();
        csv_spec.extend_from_slice(
            b"pxk1GET\0/v1/forecast\0city\x01oslo\x02\0accept\x01text/csv\x02\0",
        );
        csv_spec.extend_from_slice(&body_digest);
        assert_eq!(
            request_key(&csv, &allowlist).get(),
            fnv1a64_fold(FNV_OFFSET_BASIS, &csv_spec)
        );
        assert_ne!(
            request_key(&plain, &allowlist),
            request_key(&csv, &allowlist)
        );
        assert_eq!(
            request_key(&plain, &allowlist).get(),
            PINNED_K1,
            "K1 changed: bump KEY_SCHEMA_VERSION"
        );
        assert_eq!(
            request_key(&csv, &allowlist).get(),
            PINNED_K2,
            "K2 changed: bump KEY_SCHEMA_VERSION"
        );
        assert_eq!(KEY_SCHEMA_VERSION.get(), 1);
    }

    #[test]
    fn request_key_and_response_digest_follow_the_body_bytes() {
        let allowlist = accept_allowlist();
        let mut chat = forecast_request(&[], &[]);
        chat.method = Method::Post;
        chat.payload = Bytes::from_static(b"{\"model\":\"gpt\",\"input\":\"hi\"}");
        let mut other = chat.clone();
        other.payload = Bytes::from_static(b"{\"model\":\"gpt\",\"input\":\"yo\"}");
        assert_ne!(
            request_key(&chat, &allowlist),
            request_key(&other, &allowlist)
        );

        let body: &[&[u8]] = &[b"{\"temp\":", b"12}"];
        let digest = response_digest(200, body);
        assert_eq!(
            digest.get(),
            content_digest(&[&200u16.to_be_bytes(), b"{\"temp\":", b"12}"])
        );
        assert_ne!(digest, response_digest(404, body));
    }

    #[test]
    fn render_request_line_is_the_readable_sorted_form() {
        let request = forecast_request(&[("units", "metric"), ("city", "oslo")], &[]);
        assert_eq!(
            render_request_line(&request),
            "GET /v1/forecast?city=oslo&units=metric"
        );
    }
}
