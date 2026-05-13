//! Teach proxima a new wire protocol: implement `proxima_codec::FrameCodec`
//! (`parse_frame` / `encode_frame`) for a toy protocol defined from scratch,
//! and it composes into the pipe algebra like any other codec — a decoded
//! frame's payload is just another `Pipe` input.
//!
//! The protocol here is `kv-tlv`: a one-byte record kind, a big-endian
//! `u16` value length, then the value bytes — the same shape as HTTP/2
//! frames or protobuf fields, small enough to read in one sitting.
//!
//! Run: `cargo run --example codec`
//!
//! Builds on: transform

use core::convert::Infallible;
use core::future::Future;

use proxima_codec::FrameCodec;
use proxima_primitives::pipe::Pipe;

const HEADER_BYTES: usize = 3;
const MAX_VALUE_BYTES: usize = u16::MAX as usize;

/// Record kind — the wire byte at offset 0. Only three kinds exist on
/// purpose: enough to show a multi-branch decode without padding the
/// example with protocol design that isn't the point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecordKind {
    Set,
    Ping,
    Ack,
}

impl RecordKind {
    fn to_wire(self) -> u8 {
        match self {
            Self::Set => 0x01,
            Self::Ping => 0x02,
            Self::Ack => 0x03,
        }
    }

    fn from_wire(byte: u8) -> Result<Self, FrameError> {
        match byte {
            0x01 => Ok(Self::Set),
            0x02 => Ok(Self::Ping),
            0x03 => Ok(Self::Ack),
            other => Err(FrameError::UnknownKind(other)),
        }
    }
}

/// One decoded `kv-tlv` record: a kind plus a borrowed view of its value.
/// `value` points into the caller's buffer — decoding never copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct KvFrame<'a> {
    kind: RecordKind,
    value: &'a [u8],
}

/// Errors and control signals from [`KvCodec`]. `Incomplete` is the normal
/// partial-read signal, not a failure — a read loop sees it and asks the
/// transport for more bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameError {
    Incomplete,
    UnknownKind(u8),
    ValueTooLarge { len: usize },
}

impl core::fmt::Display for FrameError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Incomplete => write!(formatter, "incomplete kv-tlv frame"),
            Self::UnknownKind(byte) => write!(formatter, "unknown record kind {byte:#04x}"),
            Self::ValueTooLarge { len } => {
                write!(formatter, "value length {len} exceeds {MAX_VALUE_BYTES}")
            }
        }
    }
}

impl core::error::Error for FrameError {}

/// `[u8 kind][u16 BE value_len][value]` frame codec for the `kv-tlv`
/// protocol. Stateless and borrow-only, like `LengthDelimitedCodec` — the
/// same `FrameCodec` shape an H1/H2/H3 listener already hands raw socket
/// bytes to.
struct KvCodec;

impl FrameCodec for KvCodec {
    type Frame<'a> = KvFrame<'a>;
    type Error = FrameError;

    fn parse_frame<'a>(&self, buf: &'a [u8]) -> Result<(KvFrame<'a>, usize), FrameError> {
        if buf.len() < HEADER_BYTES {
            return Err(FrameError::Incomplete);
        }
        let kind = RecordKind::from_wire(buf[0])?;
        let value_len = u16::from_be_bytes([buf[1], buf[2]]) as usize;
        let total = HEADER_BYTES + value_len;
        if buf.len() < total {
            return Err(FrameError::Incomplete);
        }
        let value = &buf[HEADER_BYTES..total];
        Ok((KvFrame { kind, value }, total))
    }

    fn encode_frame(&self, frame: &KvFrame<'_>, dest: &mut Vec<u8>) -> Result<(), FrameError> {
        if frame.value.len() > MAX_VALUE_BYTES {
            return Err(FrameError::ValueTooLarge {
                len: frame.value.len(),
            });
        }
        let value_len = frame.value.len() as u16;
        dest.push(frame.kind.to_wire());
        dest.extend_from_slice(&value_len.to_be_bytes());
        dest.extend_from_slice(frame.value);
        Ok(())
    }
}

/// transform: `Vec<u8> -> Vec<u8>`, ASCII-uppercase. Proves a codec's
/// decoded frame value is not special — it flows into an ordinary `Pipe`
/// like any other `In`.
struct Uppercase;

impl Pipe for Uppercase {
    type In = Vec<u8>;
    type Out = Vec<u8>;
    type Err = Infallible;

    fn call(&self, input: Vec<u8>) -> impl Future<Output = Result<Vec<u8>, Infallible>> {
        let output = input.iter().map(u8::to_ascii_uppercase).collect();
        async move { Ok(output) }
    }
}

async fn call_pipe<PipeImpl: Pipe<Err = Infallible>>(
    pipe: &PipeImpl,
    input: PipeImpl::In,
) -> PipeImpl::Out {
    match pipe.call(input).await {
        Ok(output) => output,
        Err(never) => match never {},
    }
}

#[proxima::main(cores = 1)]
async fn main() {
    let round_trip_count = round_trip_demo();
    let partial_signaled = partial_frame_demo();
    let malformed_rejected = malformed_frame_demo();
    let composed_value = compose_with_transform_demo().await;

    println!("codec: protocol = kv-tlv ([u8 kind][u16 BE value_len][value]), codec = KvCodec");
    println!("codec: round-trip — {round_trip_count} frames encoded then decoded, all equal");
    println!("codec: partial    — truncated buffer signaled {partial_signaled:?}, not a panic");
    println!("codec: malformed  — unknown kind byte rejected as {malformed_rejected:?}");
    let composed_display = String::from_utf8_lossy(&composed_value);
    println!(
        "codec: compose    — decoded Set value fed through Uppercase pipe -> {composed_display}"
    );
}

// encode-then-assert-ok, returning the filled buffer — the shared
// non-panicking way to get from a `KvFrame` to its wire bytes below.
fn must_encode(codec: &KvCodec, frame: &KvFrame<'_>) -> Vec<u8> {
    let mut encoded = Vec::new();
    let outcome = codec.encode_frame(frame, &mut encoded);
    assert!(
        outcome.is_ok(),
        "encode must not fail for a value within limits"
    );
    encoded
}

// encode three records back to back into one buffer, then decode them off
// the front one at a time — proof that parse_frame's `consumed` count lets
// a caller walk a multi-frame buffer without knowing frame boundaries up
// front.
fn round_trip_demo() -> usize {
    let codec = KvCodec;
    let originals = [
        KvFrame {
            kind: RecordKind::Set,
            value: b"answer=42",
        },
        KvFrame {
            kind: RecordKind::Ping,
            value: b"",
        },
        KvFrame {
            kind: RecordKind::Ack,
            value: b"ok",
        },
    ];

    let mut encoded = Vec::new();
    for frame in &originals {
        let outcome = codec.encode_frame(frame, &mut encoded);
        assert!(
            outcome.is_ok(),
            "encode must not fail for a value within limits"
        );
    }

    let mut cursor = encoded.as_slice();
    let mut decoded_count = 0;
    for expected in &originals {
        let parsed = codec.parse_frame(cursor);
        assert!(
            parsed.is_ok(),
            "parse_frame must succeed on the codec's own encoded output"
        );
        if let Ok((frame, consumed)) = parsed {
            assert_eq!(frame, *expected, "decode(encode(frame)) must equal frame");
            cursor = &cursor[consumed..];
            decoded_count += 1;
        }
    }
    assert!(cursor.is_empty(), "every encoded byte must be consumed");
    decoded_count
}

// a header declaring a value longer than what's actually in the buffer
// must signal Incomplete, never panic or silently truncate — the normal
// "read more and retry" path for a stream still filling up.
fn partial_frame_demo() -> FrameError {
    let codec = KvCodec;
    let frame = KvFrame {
        kind: RecordKind::Set,
        value: b"answer=42",
    };
    let encoded = must_encode(&codec, &frame);

    let truncated = &encoded[..encoded.len() - 2];
    match codec.parse_frame(truncated) {
        Err(error) => {
            assert_eq!(error, FrameError::Incomplete);
            error
        }
        Ok(_) => unreachable!("truncated frame must not parse"),
    }
}

// a kind byte outside {0x01, 0x02, 0x03} is malformed input, not a bug in
// the codec — parse_frame must return Err, never panic, on data an
// attacker or a corrupted stream could hand it.
fn malformed_frame_demo() -> FrameError {
    let codec = KvCodec;
    let garbage = [0xff_u8, 0x00, 0x00];
    match codec.parse_frame(&garbage) {
        Err(error) => {
            assert_eq!(error, FrameError::UnknownKind(0xff));
            error
        }
        Ok(_) => unreachable!("unknown kind must not parse"),
    }
}

// the payoff: a decoded frame's value is a plain byte slice, so it feeds
// directly into an ordinary Pipe — the codec's job ends at "here is a
// frame", composition into the rest of the algebra needs nothing extra.
async fn compose_with_transform_demo() -> Vec<u8> {
    let codec = KvCodec;
    let frame = KvFrame {
        kind: RecordKind::Set,
        value: b"answer=42",
    };
    let encoded = must_encode(&codec, &frame);

    match codec.parse_frame(&encoded) {
        Ok((decoded, _consumed)) => {
            let uppercase = Uppercase;
            call_pipe(&uppercase, decoded.value.to_vec()).await
        }
        Err(_) => unreachable!("parse_frame must succeed on the codec's own encoded output"),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_decodes_every_encoded_frame() {
        assert_eq!(round_trip_demo(), 3);
    }

    #[test]
    fn partial_buffer_signals_incomplete_not_panic() {
        assert_eq!(partial_frame_demo(), FrameError::Incomplete);
    }

    #[test]
    fn malformed_kind_signals_error_not_panic() {
        assert_eq!(malformed_frame_demo(), FrameError::UnknownKind(0xff));
    }

    #[proxima::test]
    async fn composed_value_is_uppercased_through_the_pipe() {
        assert_eq!(compose_with_transform_demo().await, b"ANSWER=42".to_vec());
    }
}
