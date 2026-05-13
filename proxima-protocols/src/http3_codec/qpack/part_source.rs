//! Adapter proving `qpack::decoder`'s existing borrowing engine
//! (`FieldSink` + `decode_into`, RFC 9204 §4.5) IS a
//! `proxima_primitives::pipe::part::PartSource` — no new parsing, no codec rewrite. See
//! `docs/proxima-pipe/part-source-sink-design.md` step 1.
//!
//! [`decode_into`] calls `sink.field(name, value)` once per decoded field,
//! with `name`/`value` borrowed from `input` (raw literal), the RFC 9204
//! static table (`'static`), or the caller's `scratch` buffer (Huffman
//! output — REUSED, overwritten, per field). That last case means a
//! `PartSource` wanting to yield fields one at a time, AFTER the decode
//! call that produced them, cannot simply hold onto `decode_into`'s
//! per-call borrows — the Huffman ones would dangle by the next field.
//! [`HeaderBlockPartSource`] therefore copies every field's name+value
//! bytes ONCE, at construction, into its own fixed inline arena (0 heap
//! allocations — the arena is a `[u8; N]` on the struct, sized by
//! `proxima-h3-proto.toml`'s `qpack.part_source_arena_len`), and `next()`
//! steps through that arena.
//!
//! This is 0-ALLOC, not 0-COPY: a genuinely zero-copy incremental
//! `PartSource` needs `decode_into` itself to become per-field-resumable
//! (a real codec change), which is explicitly out of scope for step 1 —
//! see the design doc's migration step 3. The claim this module proves is
//! narrower and still load-bearing: consuming a decoded header block via
//! `PartSource::next()` performs 0 heap allocations, where draining the
//! same source into a [`proxima_primitives::pipe::request::Request`] performs the
//! allocations `Request` has always paid (method/path copy, `HeaderList`
//! growth, payload `Vec`).
//!
//! `:method` and `:path` pseudo-headers are routed to their own
//! [`Part::Method`] / [`Part::Path`] slots; every other field (including
//! other pseudo-headers such as `:status`) becomes a [`Part::Header`].
//! `next()` yields, in order: `Method` (if present), `Path` (if present),
//! each `Header` in decode order, then exactly one [`Part::End`].

//! # Two sources, one engine
//!
//! [`HeaderBlockPartSource`] (above) decodes EAGERLY at construction and
//! copies every field into its own inline arena — fully owned, queueable
//! by value, tier-3. That ownership costs a fixed-size struct move per
//! queue hop (measured as the C2 throughput regression — see
//! `docs/proxima-pipe/discipline.md` C3). [`FieldSectionSource`] is the
//! lazy sibling: it BORROWS the encoded block + a caller scratch and
//! decodes one field per [`PartSource::next`] call via
//! [`FieldSectionCursor`] — nothing is copied for static-table or raw
//! fields (they borrow the table / the block), Huffman fields decode
//! into the caller's scratch (one copy, inherent — Huffman output must
//! materialize somewhere). Use `HeaderBlockPartSource` when the source
//! must own its storage (embedded, no block to borrow); use
//! `FieldSectionSource` when the block outlives the stepping — the h3
//! client connection queues raw blocks and steps them at poll time.

#[cfg(feature = "http3_codec-alloc")]
use alloc::collections::VecDeque;
#[cfg(feature = "http3_codec-alloc")]
use alloc::vec::Vec;

use proxima_primitives::pipe::part::{Part, PartSource};

use super::decoder::{DecodeError, FieldSectionCursor, FieldSink, decode_into};
use crate::sized::{
    PROXIMA_PROTOCOLS_HTTP3_CODEC_QPACK_PART_SOURCE_ARENA_LEN as ARENA_LEN,
    PROXIMA_PROTOCOLS_HTTP3_CODEC_QPACK_PART_SOURCE_MAX_HEADERS as MAX_HEADERS,
};

#[derive(Debug, Clone, Copy)]
struct Span {
    start: usize,
    end: usize,
}

#[derive(Debug, Clone, Copy)]
struct HeaderSpan {
    name: Span,
    value: Span,
}

/// Which kind of `Part` `next()` yields next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    Method,
    Path,
    Header(usize),
    End,
    Done,
}

/// A [`PartSource`] over one decoded QPACK field section (RFC 9204 §4.5).
/// See the module docs for the 0-alloc-vs-0-copy distinction and the
/// `:method` / `:path` routing rule.
pub struct HeaderBlockPartSource {
    arena: [u8; ARENA_LEN],
    method: Option<Span>,
    path: Option<Span>,
    headers: heapless::Vec<HeaderSpan, MAX_HEADERS>,
    stage: Stage,
}

// Manual impl: the raw `arena` bytes aren't useful in a debug dump (and may
// carry sensitive header values) — report shape/state instead.
impl core::fmt::Debug for HeaderBlockPartSource {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("HeaderBlockPartSource")
            .field("has_method", &self.method.is_some())
            .field("has_path", &self.path.is_some())
            .field("header_count", &self.headers.len())
            .field("stage", &self.stage)
            .finish_non_exhaustive()
    }
}

impl HeaderBlockPartSource {
    /// Decode `wire` (one QPACK-encoded field section) and build a source
    /// over it. `cap` is `decode_into`'s
    /// `SETTINGS_MAX_FIELD_SECTION_SIZE` check; `huffman_scratch` backs
    /// Huffman-literal output during the decode call only — this source
    /// no longer needs it once `new` returns.
    ///
    /// # Errors
    ///
    /// Any [`DecodeError`] `decode_into` returns, plus
    /// [`DecodeError::SinkCapacityExceeded`] when a field doesn't fit this
    /// source's inline arena or header-slot count.
    pub fn new(wire: &[u8], cap: u64, huffman_scratch: &mut [u8]) -> Result<Self, DecodeError> {
        let mut arena = [0u8; ARENA_LEN];
        let mut used = 0usize;
        let mut method = None;
        let mut path = None;
        let mut headers: heapless::Vec<HeaderSpan, MAX_HEADERS> = heapless::Vec::new();

        {
            let mut sink = ArenaFieldSink {
                arena: &mut arena,
                used: &mut used,
                method: &mut method,
                path: &mut path,
                headers: &mut headers,
            };
            decode_into(wire, cap, huffman_scratch, &mut sink)?;
        }

        Ok(Self {
            arena,
            method,
            path,
            headers,
            stage: Stage::Method,
        })
    }

    fn slice(&self, span: Span) -> &[u8] {
        &self.arena[span.start..span.end]
    }
}

impl PartSource for HeaderBlockPartSource {
    fn next(&mut self) -> Option<Part<'_>> {
        loop {
            match self.stage {
                Stage::Method => {
                    self.stage = Stage::Path;
                    if let Some(span) = self.method {
                        return Some(Part::Method(self.slice(span)));
                    }
                }
                Stage::Path => {
                    self.stage = Stage::Header(0);
                    if let Some(span) = self.path {
                        return Some(Part::Path(self.slice(span)));
                    }
                }
                Stage::Header(index) => {
                    if index >= self.headers.len() {
                        self.stage = Stage::End;
                        continue;
                    }
                    self.stage = Stage::Header(index + 1);
                    let field = self.headers[index];
                    return Some(Part::Header(
                        self.slice(field.name),
                        self.slice(field.value),
                    ));
                }
                Stage::End => {
                    self.stage = Stage::Done;
                    return Some(Part::End);
                }
                Stage::Done => return None,
            }
        }
    }
}

/// The [`FieldSink`] [`HeaderBlockPartSource::new`] drives `decode_into`
/// with — copies every field's bytes into the caller's arena and routes
/// `:method` / `:path` to their own slots.
struct ArenaFieldSink<'source> {
    arena: &'source mut [u8; ARENA_LEN],
    used: &'source mut usize,
    method: &'source mut Option<Span>,
    path: &'source mut Option<Span>,
    headers: &'source mut heapless::Vec<HeaderSpan, MAX_HEADERS>,
}

impl ArenaFieldSink<'_> {
    fn copy_in(&mut self, bytes: &[u8]) -> Result<Span, DecodeError> {
        let start = *self.used;
        let end = start + bytes.len();
        if end > self.arena.len() {
            return Err(DecodeError::SinkCapacityExceeded {
                needed: end,
                available: self.arena.len(),
            });
        }
        self.arena[start..end].copy_from_slice(bytes);
        *self.used = end;
        Ok(Span { start, end })
    }
}

impl FieldSink for ArenaFieldSink<'_> {
    fn field(&mut self, name: &[u8], value: &[u8]) -> Result<(), DecodeError> {
        let name_span = self.copy_in(name)?;
        let value_span = self.copy_in(value)?;
        if name == b":method" {
            *self.method = Some(value_span);
        } else if name == b":path" {
            *self.path = Some(value_span);
        } else {
            self.headers
                .push(HeaderSpan {
                    name: name_span,
                    value: value_span,
                })
                .map_err(|_dropped| DecodeError::SinkCapacityExceeded {
                    needed: self.headers.len() + 1,
                    available: MAX_HEADERS,
                })?;
        }
        Ok(())
    }
}

/// A lazy, borrowing [`PartSource`] over one QPACK-encoded field section
/// — decodes one field per [`PartSource::next`] call via
/// [`FieldSectionCursor`] (the same engine `decode_into` drives; see the
/// module docs for when to prefer this over [`HeaderBlockPartSource`]).
/// `:method` / `:path` route to [`Part::Method`] / [`Part::Path`]; every
/// other field (including `:status`) yields [`Part::Header`]; exactly one
/// [`Part::End`] follows a cleanly-exhausted section.
///
/// # Errors
///
/// `next()` has no error channel ([`PartSource`] is infallible by
/// shape), so a decode failure ends the stream EARLY: `next()` returns
/// `None` **without** a preceding [`Part::End`], and [`Self::error`]
/// reports the [`DecodeError`]. Callers MUST check `error()` after
/// exhaustion and treat `Some(_)` as a QPACK decompression failure
/// (connection error per RFC 9204 §2.2.3) — the wire position of the
/// failed field is undefined, the source is not resumable.
#[derive(Debug)]
pub struct FieldSectionSource<'a> {
    cursor: Option<FieldSectionCursor<'a>>,
    scratch: &'a mut [u8],
    done: bool,
    error: Option<DecodeError>,
}

impl<'a> FieldSectionSource<'a> {
    /// Build a source over `wire` (one QPACK-encoded field section).
    /// `cap` is the `SETTINGS_MAX_FIELD_SECTION_SIZE` bound enforced
    /// cumulatively while stepping; `scratch` backs Huffman-literal
    /// output and is overwritten per field (the borrow returned by
    /// `next()` is valid until the following `next()` call, per the
    /// lending [`PartSource`] contract). Construction is infallible — a
    /// malformed section prefix surfaces through [`Self::error`] on the
    /// first `next()` call, the same channel as every later decode
    /// failure.
    #[must_use]
    pub fn new(wire: &'a [u8], cap: u64, scratch: &'a mut [u8]) -> Self {
        match FieldSectionCursor::new(wire, cap) {
            Ok(cursor) => Self {
                cursor: Some(cursor),
                scratch,
                done: false,
                error: None,
            },
            Err(err) => Self {
                cursor: None,
                scratch,
                done: true,
                error: Some(err),
            },
        }
    }

    /// The decode failure that ended this source early, if any. `None`
    /// after a clean [`Part::End`]; `Some(_)` means the section was
    /// malformed or exceeded a bound — see the type docs for the
    /// caller's obligation.
    #[must_use]
    pub fn error(&self) -> Option<DecodeError> {
        self.error
    }
}

impl PartSource for FieldSectionSource<'_> {
    fn next(&mut self) -> Option<Part<'_>> {
        if self.done {
            return None;
        }
        let cursor = self.cursor.as_mut()?;
        match cursor.next_field(&mut *self.scratch) {
            Ok(Some((name, value))) => Some(match name {
                b":method" => Part::Method(value),
                b":path" => Part::Path(value),
                _ => Part::Header(name, value),
            }),
            Ok(None) => {
                self.done = true;
                Some(Part::End)
            }
            Err(err) => {
                self.done = true;
                self.error = Some(err);
                None
            }
        }
    }
}

/// Ceiling on recycled block buffers — same free-list discipline as the
/// connection FSMs' outbound pools.
#[cfg(feature = "http3_codec-alloc")]
const BLOCK_POOL_CAP: usize = 64;

/// Pool-recycled queue of RAW (still-encoded) field sections plus the
/// current-block and Huffman-scratch plumbing a connection FSM needs to
/// hand out borrowed [`FieldSectionSource`]s one at a time. This is the
/// shared substrate of every `Source`-mode header path (h3 client, h3
/// server, and the h1/h2 dispatch migrations to come): `push` copies one
/// encoded section into a recycled buffer at feed time (steady-state 0
/// allocations); `poll` recycles the previously handed-out block, pops
/// the next, and lends a [`FieldSectionSource`] over it. Keyed by `K`
/// (a `Copy` stream identifier) so each protocol brings its own ID type.
#[cfg(feature = "http3_codec-alloc")]
#[derive(Debug)]
pub struct HeaderBlockQueue<K> {
    blocks: VecDeque<(K, Vec<u8>)>,
    pool: Vec<Vec<u8>>,
    current: Option<(K, Vec<u8>)>,
    scratch: Vec<u8>,
}

#[cfg(feature = "http3_codec-alloc")]
impl<K: Copy> HeaderBlockQueue<K> {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            blocks: VecDeque::new(),
            pool: Vec::new(),
            current: None,
            scratch: Vec::new(),
        }
    }

    /// Size the Huffman scratch once — the queue's only setup
    /// allocation. Call from the connection's `Source`-mode opt-in;
    /// never shrinks.
    pub fn size_scratch(&mut self, len: usize) {
        if self.scratch.len() < len {
            self.scratch.resize(len, 0);
        }
    }

    /// Copy one still-encoded field section into a pool-recycled buffer
    /// and queue it. Steady-state 0 allocations; the copy is of a frame
    /// the caller has already buffered (no amplification — cap
    /// enforcement runs per-field while STEPPING, before any
    /// Huffman/expansion work).
    pub fn push(&mut self, key: K, block: &[u8]) {
        let mut buf = self.pool.pop().unwrap_or_default();
        buf.clear();
        buf.extend_from_slice(block);
        self.blocks.push_back((key, buf));
    }

    /// Recycle the previously handed-out block, pop the next queued one,
    /// and lend a [`FieldSectionSource`] over it (`cap` =
    /// `SETTINGS_MAX_FIELD_SECTION_SIZE`). The source borrows this
    /// queue; drop it before the next `poll`. Callers MUST check
    /// [`FieldSectionSource::error`] after stepping — see the source's
    /// deferred-validation contract.
    #[must_use]
    pub fn poll(&mut self, cap: u64) -> Option<(K, FieldSectionSource<'_>)> {
        if let Some((_, block)) = self.current.take()
            && self.pool.len() < BLOCK_POOL_CAP
        {
            self.pool.push(block);
        }
        let next = self.blocks.pop_front()?;
        let (key, block) = self.current.insert(next);
        Some((*key, FieldSectionSource::new(block, cap, &mut self.scratch)))
    }
}

#[cfg(feature = "http3_codec-alloc")]
impl<K: Copy> Default for HeaderBlockQueue<K> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec::Vec;

    #[cfg(feature = "std")]
    use proxima_primitives::pipe::request::Request;

    use super::{FieldSectionSource, HeaderBlockPartSource, Part, PartSource};
    use crate::http3_codec::qpack::decoder::DecodeError;
    use crate::http3_codec::qpack::{encoder, integer, static_table};

    /// Synthesizes a browser/curl-shaped `GET` request header set via this
    /// crate's own canonical encoder (RFC 9204 §4.5 conformant emit) — same
    /// real-data exception already established by
    /// `decoder::tests::nginx_like_response_wire`: standing up a live
    /// client/server capture is external infra this pass doesn't have. The
    /// pseudo-headers (`:method`, `:path`) plus two ordinary headers give
    /// the adapter something to route to every `Part` kind it handles.
    fn request_like_wire() -> Vec<u8> {
        let mut out = Vec::new();
        encoder::encode_refs(
            [
                (b":method".as_slice(), b"GET".as_slice()),
                (b":path".as_slice(), b"/v1/items".as_slice()),
                (b"user-agent".as_slice(), b"curl/8.7.1".as_slice()),
                (b"accept".as_slice(), b"application/json".as_slice()),
            ],
            &mut out,
        )
        .expect("encode request-shaped header set");
        out
    }

    #[test]
    fn yields_method_path_headers_end_in_order() {
        let wire = request_like_wire();
        let mut scratch = [0u8; 256];
        let mut source =
            HeaderBlockPartSource::new(&wire, u64::MAX, &mut scratch).expect("decode wire");

        assert_eq!(source.next(), Some(Part::Method(b"GET")));
        assert_eq!(source.next(), Some(Part::Path(b"/v1/items")));
        assert_eq!(
            source.next(),
            Some(Part::Header(b"user-agent", b"curl/8.7.1"))
        );
        assert_eq!(
            source.next(),
            Some(Part::Header(b"accept", b"application/json"))
        );
        assert_eq!(source.next(), Some(Part::End));
        assert_eq!(source.next(), None);
    }

    #[test]
    fn missing_pseudo_headers_are_skipped_not_stubbed() {
        let mut out = Vec::new();
        encoder::encode_refs([(b"content-length".as_slice(), b"0".as_slice())], &mut out)
            .expect("encode a single ordinary header, no pseudo-headers");
        let mut scratch = [0u8; 64];
        let mut source =
            HeaderBlockPartSource::new(&out, u64::MAX, &mut scratch).expect("decode wire");

        assert_eq!(source.next(), Some(Part::Header(b"content-length", b"0")));
        assert_eq!(source.next(), Some(Part::End));
        assert_eq!(source.next(), None);
    }

    #[test]
    fn sink_capacity_exceeded_when_header_count_exceeds_max_headers() {
        // `qpack.part_source_max_headers` = 64 (proxima-h3-proto.toml); one
        // past that must be rejected, not silently dropped or truncated.
        let names: Vec<Vec<u8>> = (0..65)
            .map(|index| alloc::format!("x-generated-header-{index}").into_bytes())
            .collect();
        let values: Vec<Vec<u8>> = (0..65)
            .map(|index| alloc::format!("v{index}").into_bytes())
            .collect();

        let mut out = Vec::new();
        encoder::encode_refs(
            names
                .iter()
                .zip(values.iter())
                .map(|(name, value)| (name.as_slice(), value.as_slice())),
            &mut out,
        )
        .expect("encode 65 distinct headers");

        let mut scratch = [0u8; 512];
        let err = HeaderBlockPartSource::new(&out, u64::MAX, &mut scratch)
            .expect_err("65th non-pseudo field must exceed MAX_HEADERS capacity");
        assert!(matches!(
            err,
            crate::http3_codec::qpack::decoder::DecodeError::SinkCapacityExceeded { .. }
        ));
    }

    #[test]
    fn field_section_source_yields_wire_order_with_pseudo_header_routing() {
        let wire = request_like_wire();
        let mut scratch = [0u8; 256];
        let mut source = FieldSectionSource::new(&wire, u64::MAX, &mut scratch);

        assert_eq!(source.next(), Some(Part::Method(b"GET")));
        assert_eq!(source.next(), Some(Part::Path(b"/v1/items")));
        assert_eq!(
            source.next(),
            Some(Part::Header(b"user-agent", b"curl/8.7.1"))
        );
        assert_eq!(
            source.next(),
            Some(Part::Header(b"accept", b"application/json"))
        );
        assert_eq!(source.next(), Some(Part::End));
        assert_eq!(source.next(), None);
        assert_eq!(source.error(), None);
    }

    /// Same RFC 7541 §C.4.2 "no-cache" Huffman vector the decoder's own
    /// tests use — a lazy source must resolve Huffman literals through
    /// the caller's scratch exactly as the eager surfaces do.
    #[test]
    fn field_section_source_decodes_huffman_value_through_scratch() {
        let name_index =
            static_table::find_name(b"cache-control").expect("cache-control in static table");
        let mut wire = alloc::vec![0x00u8, 0x00];
        let mut name_ref = [0u8; 4];
        let name_written = integer::encode(name_index as u64, 4, 0b0101_0000, &mut name_ref)
            .expect("encode 4-bit name index");
        wire.extend_from_slice(&name_ref[..name_written]);
        let huffman_no_cache = [0xa8u8, 0xeb, 0x10, 0x64, 0x9c, 0xbf];
        let mut value_len = [0u8; 2];
        let value_len_written = integer::encode(
            huffman_no_cache.len() as u64,
            7,
            0b1000_0000,
            &mut value_len,
        )
        .expect("encode 7-bit H=1 value length");
        wire.extend_from_slice(&value_len[..value_len_written]);
        wire.extend_from_slice(&huffman_no_cache);

        let mut scratch = [0u8; 32];
        let mut source = FieldSectionSource::new(&wire, u64::MAX, &mut scratch);
        assert_eq!(
            source.next(),
            Some(Part::Header(b"cache-control", b"no-cache"))
        );
        assert_eq!(source.next(), Some(Part::End));
        assert_eq!(source.next(), None);
        assert_eq!(source.error(), None);
    }

    #[test]
    fn field_section_source_reports_malformed_prefix_via_error_not_panic() {
        let mut scratch = [0u8; 16];
        let mut source = FieldSectionSource::new(&[0x00], u64::MAX, &mut scratch);

        assert_eq!(source.next(), None, "no Part::End on a failed section");
        assert_eq!(source.error(), Some(DecodeError::Truncated));
    }

    #[test]
    fn field_section_source_cap_failure_ends_early_without_end_part() {
        let wire = request_like_wire();
        let mut scratch = [0u8; 256];
        // cap 50 admits the first small field (~35-45 decoded bytes) and
        // must reject a later one mid-section.
        let mut source = FieldSectionSource::new(&wire, 50, &mut scratch);

        let mut yielded = 0usize;
        let mut saw_end = false;
        while let Some(part) = source.next() {
            yielded += 1;
            saw_end = matches!(part, Part::End);
        }
        assert!(yielded >= 1, "first field fits under the cap");
        assert!(!saw_end, "a capped section must not yield Part::End");
        assert!(matches!(
            source.error(),
            Some(DecodeError::ExceedsMaxFieldSectionSize { cap: 50, .. })
        ));
    }

    /// The C3 load-bearing claim (`docs/proxima-pipe/discipline.md`):
    /// constructing + stepping a `FieldSectionSource` to exhaustion is 0
    /// heap allocations — no arena, no owned copy, the block and scratch
    /// are the caller's.
    #[cfg(feature = "std")]
    #[test]
    fn field_section_source_is_zero_alloc_stepping_to_exhaustion() {
        let wire = request_like_wire();
        let region = crate::alloc_test::exclusive_region();

        let before = region.change();
        let mut scratch = [0u8; 256];
        let mut source = FieldSectionSource::new(&wire, u64::MAX, &mut scratch);
        let mut parts_seen = 0usize;
        while source.next().is_some() {
            parts_seen += 1;
        }
        let after = region.change();

        assert_eq!(parts_seen, 5, "method + path + 2 headers + end");
        assert_eq!(source.error(), None);
        assert_eq!(
            after.allocations - before.allocations,
            0,
            "stepping a FieldSectionSource to exhaustion must perform 0 heap allocations"
        );
    }

    /// DC-H3-PART-SOURCE-ALLOC — the load-bearing claim of
    /// `docs/proxima-pipe/part-source-sink-design.md` step 1: stepping a
    /// `PartSource` (decode + `next()` to exhaustion) performs 0 heap
    /// allocations; draining the SAME decoded block into an owned
    /// `Request` via `Request::from_source` performs > 0 (the
    /// materialization cost a zero-copy dispatch path opts out of).
    #[cfg(feature = "std")]
    #[test]
    fn header_block_part_source_is_zero_alloc_request_drain_is_not() {
        let wire = request_like_wire();
        let region = crate::alloc_test::exclusive_region();

        let before_source = region.change();
        let mut scratch = [0u8; 256];
        let mut source =
            HeaderBlockPartSource::new(&wire, u64::MAX, &mut scratch).expect("decode wire");
        let mut parts_seen = 0usize;
        while source.next().is_some() {
            parts_seen += 1;
        }
        let after_source = region.change();
        assert_eq!(parts_seen, 5, "method + path + 2 headers + end");
        assert_eq!(
            after_source.allocations - before_source.allocations,
            0,
            "decoding + stepping a PartSource to exhaustion must perform 0 heap allocations"
        );

        let before_drain = region.change();
        let mut drain_scratch = [0u8; 256];
        let mut source_for_drain =
            HeaderBlockPartSource::new(&wire, u64::MAX, &mut drain_scratch).expect("decode wire");
        let request = Request::from_source(&mut source_for_drain);
        let after_drain = region.change();
        assert_eq!(request.method, "GET");
        assert_eq!(&request.path[..], b"/v1/items");
        assert_eq!(request.metadata.get_str("user-agent"), Some("curl/8.7.1"));
        assert!(
            after_drain.allocations > before_drain.allocations,
            "Request::from_source materializes an owned aggregate — it allocates, on purpose \
             (this is the cost stepping the source directly opts out of)"
        );
    }
}
