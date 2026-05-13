use core::fmt;

// opt-sweep note (id.rs): first tweak target is AVX2 nibble-pack on x86_64 and NEON on aarch64.
// both paths replace decode_hex_16/decode_hex_8 below. gate behind workspace `simd-hex` feature.
// scalar branchless table stays as the non-simd fallback.

const HEX_DECODE: [u8; 256] = {
    let mut table = [0xffu8; 256];
    let mut i = 0u8;
    loop {
        table[i as usize] = match i {
            b'0'..=b'9' => i - b'0',
            b'a'..=b'f' => i - b'a' + 10,
            _ => 0xff,
        };
        if i == 255 {
            break;
        }
        i += 1;
    }
    table
};

const HEX_ENCODE: &[u8; 16] = b"0123456789abcdef";

// traceparent: 00-{32}-{16}-{2}, total 55 bytes, version byte must be 0x30 0x30 ("00")
const TRACEPARENT_LEN: usize = 55;
const TRACE_HEX_START: usize = 3;
const TRACE_HEX_LEN: usize = 32;
const SPAN_HEX_START: usize = 36;
const SPAN_HEX_LEN: usize = 16;
const FLAGS_HEX_START: usize = 53;
const FLAGS_HEX_LEN: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraceId([u8; 16]);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpanId([u8; 8]);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraceFlags(pub u8);

impl TraceFlags {
    pub const SAMPLED: Self = Self(0x01);
    pub const NOT_SAMPLED: Self = Self(0x00);
    /// proxima-local marker (bit `0x02`): this record belongs to a
    /// verbose-buffered (error-elevation) trace, so `ElevationSink` retains it
    /// for a possible replay. NOT a W3C-standard flag — it is stamped on the
    /// `LogRecord` only, never on the span context that serializes to the
    /// outbound `traceparent`, so it stays in-process.
    pub const VERBOSE_BUFFERED: Self = Self(0x02);

    /// Set the verbose-buffered marker, preserving the sampled bit.
    #[must_use]
    pub const fn with_verbose_buffered(self) -> Self {
        Self(self.0 | Self::VERBOSE_BUFFERED.0)
    }

    /// Whether the verbose-buffered marker is set.
    #[must_use]
    pub const fn is_verbose_buffered(self) -> bool {
        self.0 & Self::VERBOSE_BUFFERED.0 != 0
    }
}

impl TraceId {
    pub const INVALID: Self = Self([0u8; 16]);

    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    pub const fn to_bytes(self) -> [u8; 16] {
        self.0
    }

    /// Mint a fresh random (non-zero) trace id. Non-crypto RNG is the W3C
    /// norm for ids; collision resistance, not unpredictability, is what
    /// matters. The low bit is forced set so the result is always valid.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn generate() -> Self {
        let high = fastrand::u64(..).to_be_bytes();
        let low = fastrand::u64(..).to_be_bytes();
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&high);
        bytes[8..].copy_from_slice(&low);
        bytes[15] |= 1;
        Self(bytes)
    }

    fn is_valid(self) -> bool {
        self.0 != [0u8; 16]
    }
}

impl SpanId {
    pub const INVALID: Self = Self([0u8; 8]);

    pub const fn from_bytes(bytes: [u8; 8]) -> Self {
        Self(bytes)
    }

    pub const fn to_bytes(self) -> [u8; 8] {
        self.0
    }

    /// Mint a fresh random (non-zero) span id. See [`TraceId::generate`].
    #[cfg(feature = "std")]
    #[must_use]
    pub fn generate() -> Self {
        let mut bytes = fastrand::u64(..).to_be_bytes();
        bytes[7] |= 1;
        Self(bytes)
    }

    fn is_valid(self) -> bool {
        self.0 != [0u8; 8]
    }
}

impl fmt::Display for TraceId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = [0u8; 32];
        for (index, byte) in self.0.iter().enumerate() {
            buf[index * 2] = HEX_ENCODE[(byte >> 4) as usize];
            buf[index * 2 + 1] = HEX_ENCODE[(byte & 0x0f) as usize];
        }
        // safety: buf contains only ascii hex chars from HEX_ENCODE
        let text = unsafe { core::str::from_utf8_unchecked(&buf) };
        formatter.write_str(text)
    }
}

impl fmt::Display for SpanId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = [0u8; 16];
        for (index, byte) in self.0.iter().enumerate() {
            buf[index * 2] = HEX_ENCODE[(byte >> 4) as usize];
            buf[index * 2 + 1] = HEX_ENCODE[(byte & 0x0f) as usize];
        }
        // safety: buf contains only ascii hex chars from HEX_ENCODE
        let text = unsafe { core::str::from_utf8_unchecked(&buf) };
        formatter.write_str(text)
    }
}

#[inline]
fn decode_hex_16(src: &[u8]) -> Option<[u8; 16]> {
    let mut out = [0u8; 16];
    let mut index = 0usize;
    while index < 16 {
        let hi = HEX_DECODE[src[index * 2] as usize];
        let lo = HEX_DECODE[src[index * 2 + 1] as usize];
        if hi | lo == 0xff {
            return None;
        }
        out[index] = (hi << 4) | lo;
        index += 1;
    }
    Some(out)
}

#[inline]
fn decode_hex_8(src: &[u8]) -> Option<[u8; 8]> {
    let mut out = [0u8; 8];
    let mut index = 0usize;
    while index < 8 {
        let hi = HEX_DECODE[src[index * 2] as usize];
        let lo = HEX_DECODE[src[index * 2 + 1] as usize];
        if hi | lo == 0xff {
            return None;
        }
        out[index] = (hi << 4) | lo;
        index += 1;
    }
    Some(out)
}

#[inline]
fn decode_hex_1(src: &[u8]) -> Option<u8> {
    let hi = HEX_DECODE[src[0] as usize];
    let lo = HEX_DECODE[src[1] as usize];
    if hi | lo == 0xff {
        return None;
    }
    Some((hi << 4) | lo)
}

/// Parse a W3C traceparent header value from raw bytes.
///
/// Returns `None` for any malformed input: wrong length, wrong version,
/// uppercase hex, invalid hex chars, all-zero trace_id, or all-zero span_id.
/// All-zero flags are valid per spec.
#[must_use]
pub fn parse_traceparent(input: &[u8]) -> Option<(TraceId, SpanId, TraceFlags)> {
    if input.len() != TRACEPARENT_LEN {
        return None;
    }

    // version must be literal "00"
    if input[0] != b'0' || input[1] != b'0' {
        return None;
    }

    if input[2] != b'-' || input[35] != b'-' || input[52] != b'-' {
        return None;
    }

    let trace_bytes = decode_hex_16(&input[TRACE_HEX_START..TRACE_HEX_START + TRACE_HEX_LEN])?;
    let span_bytes = decode_hex_8(&input[SPAN_HEX_START..SPAN_HEX_START + SPAN_HEX_LEN])?;
    let flags_byte = decode_hex_1(&input[FLAGS_HEX_START..FLAGS_HEX_START + FLAGS_HEX_LEN])?;

    let trace_id = TraceId::from_bytes(trace_bytes);
    let span_id = SpanId::from_bytes(span_bytes);

    if !trace_id.is_valid() || !span_id.is_valid() {
        return None;
    }

    Some((trace_id, span_id, TraceFlags(flags_byte)))
}

/// Render a W3C `traceparent` header value as the 55 ASCII bytes
/// `00-{trace:32}-{span:16}-{flags:2}`. Inverse of [`parse_traceparent`];
/// the output round-trips back through it. Allocation-free.
#[must_use]
pub fn format_traceparent(
    trace_id: &TraceId,
    span_id: &SpanId,
    flags: TraceFlags,
) -> [u8; TRACEPARENT_LEN] {
    let mut out = [b'-'; TRACEPARENT_LEN];
    out[0] = b'0';
    out[1] = b'0';
    for (index, byte) in trace_id.0.iter().enumerate() {
        out[TRACE_HEX_START + index * 2] = HEX_ENCODE[(byte >> 4) as usize];
        out[TRACE_HEX_START + index * 2 + 1] = HEX_ENCODE[(byte & 0x0f) as usize];
    }
    for (index, byte) in span_id.0.iter().enumerate() {
        out[SPAN_HEX_START + index * 2] = HEX_ENCODE[(byte >> 4) as usize];
        out[SPAN_HEX_START + index * 2 + 1] = HEX_ENCODE[(byte & 0x0f) as usize];
    }
    out[FLAGS_HEX_START] = HEX_ENCODE[(flags.0 >> 4) as usize];
    out[FLAGS_HEX_START + 1] = HEX_ENCODE[(flags.0 & 0x0f) as usize];
    out
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default,
        clippy::type_complexity,
        clippy::useless_vec,
        clippy::needless_range_loop,
        clippy::default_constructed_unit_structs
    )]

    use super::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use rstest::rstest;

    const REF_INPUT: &[u8] = b"00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";

    #[rstest]
    #[case::w3c_reference(
        b"00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        [0x0a, 0xf7, 0x65, 0x19, 0x16, 0xcd, 0x43, 0xdd, 0x84, 0x48, 0xeb, 0x21, 0x1c, 0x80, 0x31, 0x9c],
        [0xb7, 0xad, 0x6b, 0x71, 0x69, 0x20, 0x33, 0x31],
        0x01,
    )]
    #[case::flags_zero(
        b"00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-00",
        [0x0a, 0xf7, 0x65, 0x19, 0x16, 0xcd, 0x43, 0xdd, 0x84, 0x48, 0xeb, 0x21, 0x1c, 0x80, 0x31, 0x9c],
        [0xb7, 0xad, 0x6b, 0x71, 0x69, 0x20, 0x33, 0x31],
        0x00,
    )]
    fn parse_happy(
        #[case] input: &[u8],
        #[case] expected_trace: [u8; 16],
        #[case] expected_span: [u8; 8],
        #[case] expected_flags: u8,
    ) {
        let result = parse_traceparent(input).unwrap();
        assert_eq!(result.0.to_bytes(), expected_trace);
        assert_eq!(result.1.to_bytes(), expected_span);
        assert_eq!(result.2.0, expected_flags);
    }

    #[rstest]
    #[case::too_short(b"00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-0")]
    #[case::too_long(b"00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01x")]
    #[case::wrong_version(b"01-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01")]
    #[case::uppercase_hex(b"00-0AF7651916CD43DD8448EB211C80319C-b7ad6b7169203331-01")]
    #[case::non_hex_trace(b"00-zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz-b7ad6b7169203331-01")]
    #[case::missing_dash(b"00x0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01")]
    fn parse_sad(#[case] input: &[u8]) {
        assert!(parse_traceparent(input).is_none());
    }

    #[rstest]
    #[case::zero_trace_id(b"00-00000000000000000000000000000000-b7ad6b7169203331-01")]
    #[case::zero_span_id(b"00-0af7651916cd43dd8448eb211c80319c-0000000000000000-01")]
    fn parse_zero_ids_rejected(#[case] input: &[u8]) {
        assert!(parse_traceparent(input).is_none());
    }

    #[test]
    fn zero_flags_accepted() {
        let result = parse_traceparent(b"00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-00");
        assert!(result.is_some());
        assert_eq!(result.unwrap().2.0, 0x00);
    }

    #[test]
    fn minimum_valid_round_trip() {
        let input = b"00-00000000000000000000000000000001-0000000000000001-00";
        let (trace_id, span_id, flags) = parse_traceparent(input).unwrap();
        let formatted_trace = format!("{trace_id}");
        let formatted_span = format!("{span_id}");
        assert_eq!(formatted_trace, "00000000000000000000000000000001");
        assert_eq!(formatted_span, "0000000000000001");
        assert_eq!(flags.0, 0x00);
    }

    #[test]
    fn display_trace_id_lowercase_32_chars() {
        let (trace_id, span_id, _) = parse_traceparent(REF_INPUT).unwrap();
        let trace_str = format!("{trace_id}");
        let span_str = format!("{span_id}");
        assert_eq!(trace_str.len(), 32);
        assert_eq!(span_str.len(), 16);
        assert!(
            trace_str
                .chars()
                .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit())
        );
        assert!(
            span_str
                .chars()
                .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit())
        );
        assert_eq!(trace_str, "0af7651916cd43dd8448eb211c80319c");
        assert_eq!(span_str, "b7ad6b7169203331");
    }

    #[test]
    fn format_matches_canonical_reference() {
        let (trace_id, span_id, flags) = parse_traceparent(REF_INPUT).unwrap();
        let rendered = format_traceparent(&trace_id, &span_id, flags);
        assert_eq!(&rendered, REF_INPUT);
    }

    #[test]
    fn format_then_parse_round_trips() {
        let mut rng = SmallRng::seed_from_u64(0x0123_4567_89ab_cdef);
        for _ in 0..256 {
            let mut trace_bytes = [0u8; 16];
            let mut span_bytes = [0u8; 8];
            rng.fill_bytes(&mut trace_bytes);
            rng.fill_bytes(&mut span_bytes);
            trace_bytes[0] |= 1;
            span_bytes[0] |= 1;
            let mut flags_byte = [0u8; 1];
            rng.fill_bytes(&mut flags_byte);
            let trace_id = TraceId::from_bytes(trace_bytes);
            let span_id = SpanId::from_bytes(span_bytes);
            let flags = TraceFlags(flags_byte[0]);
            let rendered = format_traceparent(&trace_id, &span_id, flags);
            let (parsed_trace, parsed_span, parsed_flags) =
                parse_traceparent(&rendered).expect("formatted output must parse");
            assert_eq!(parsed_trace.to_bytes(), trace_bytes);
            assert_eq!(parsed_span.to_bytes(), span_bytes);
            assert_eq!(parsed_flags.0, flags.0);
        }
    }

    #[test]
    fn generate_produces_valid_distinct_ids() {
        let trace_a = TraceId::generate();
        let trace_b = TraceId::generate();
        assert!(trace_a.is_valid());
        assert!(trace_b.is_valid());
        assert_ne!(trace_a, trace_b, "trace ids must not collide trivially");
        let span_a = SpanId::generate();
        assert!(span_a.is_valid());
        // a generated trace + span must format into a parseable traceparent
        let rendered = format_traceparent(&trace_a, &span_a, TraceFlags::SAMPLED);
        assert!(parse_traceparent(&rendered).is_some());
    }

    #[test]
    fn property_random_valid_round_trip() {
        let mut rng = SmallRng::seed_from_u64(0xdeadbeef_cafef00d);
        for _ in 0..256 {
            let mut trace_bytes = [0u8; 16];
            let mut span_bytes = [0u8; 8];
            rng.fill_bytes(&mut trace_bytes);
            rng.fill_bytes(&mut span_bytes);
            // ensure non-zero
            trace_bytes[15] |= 1;
            span_bytes[7] |= 1;

            let trace_id = TraceId::from_bytes(trace_bytes);
            let span_id = SpanId::from_bytes(span_bytes);
            let flags = 0x01u8;

            let rendered = format!("00-{trace_id}-{span_id}-{flags:02x}");
            let parsed = parse_traceparent(rendered.as_bytes()).unwrap();
            assert_eq!(parsed.0.to_bytes(), trace_bytes);
            assert_eq!(parsed.1.to_bytes(), span_bytes);
            assert_eq!(parsed.2.0, flags);
        }
    }
}
