//! Proves `FrameCodecPipe<DelimiterCodec>` actually COLLECTS bytes across
//! misaligned transport reads and EMITS whole NDJSON objects, one per
//! `\n` — the point of giving `DelimiterCodec` an [`OwnFrame`]/[`Incomplete`]
//! pair at all (without them a caller hand-rolls the exact accumulation
//! loop this pipe exists to replace).
//!
//! The corpus is real captured NDJSON
//! (`tests/fixtures/ndjson/duplex.ndjson`, ten lines, ranging 275-1434
//! bytes each), vendored in-repo (principle 16). Every case here re-drives
//! that SAME corpus through a different transport-shaped chunk split and
//! asserts the emitted frame sequence, plus the summed `consumed` count,
//! matches a single whole-buffer feed exactly — chunking is a transport
//! detail the pipe must be invisible to.
//!
//! Chunk sizes are transport-shaped (4096/8192/65536 — real
//! pipe/socket read sizes) plus a couple of small-but-plausible ones (3, 7)
//! to catch an off-by-one at a delimiter boundary. A byte-at-a-time driver
//! was deliberately dropped: no real reader delivers 1 byte per call, and
//! it is the SLOWEST possible driver of `DelimiterCodec`'s O(n^2) rescan
//! cost (see the crate's own notes on `FrameCodec::parse_frame` being
//! stateless), which would make a correctness-only test suite look green
//! while hiding that cost entirely.

#![allow(clippy::expect_used)]

use std::path::PathBuf;

use bytes::Bytes;
use proxima_codec::DelimiterCodec;
use proxima_primitives::pipe::Pipe;
use proxima_protocols::codec_pipe::FrameCodecPipe;
use rstest::rstest;

fn fixture() -> Vec<u8> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ndjson/duplex.ndjson");
    std::fs::read(&path).unwrap_or_else(|err| panic!("read fixture {}: {err}", path.display()))
}

/// Dependency-free executor for the always-ready probe futures — mirrors
/// `codec_pipe`'s own test helper.
fn block_on<Fut: std::future::Future>(future: Fut) -> Fut::Output {
    let mut pinned = std::pin::pin!(future);
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    loop {
        if let std::task::Poll::Ready(output) = pinned.as_mut().poll(&mut context) {
            return output;
        }
    }
}

/// Drive `chunks` through `FrameCodecPipe<DelimiterCodec>` as if each
/// element arrived from a separate transport read: append it to the
/// accumulator, then drain every complete frame currently buffered
/// (advancing by `consumed` and calling again — the documented driver
/// contract) before appending the next chunk. Returns the emitted frame
/// sequence plus the summed `consumed` bytes.
fn drive(chunks: &[&[u8]]) -> (Vec<Bytes>, usize) {
    let pipe = FrameCodecPipe::new(DelimiterCodec::unbounded(b"\n"));
    let mut accumulator: Vec<u8> = Vec::new();
    let mut frames = Vec::new();
    let mut total_consumed = 0usize;

    for chunk in chunks {
        accumulator.extend_from_slice(chunk);
        loop {
            let window = Bytes::copy_from_slice(&accumulator);
            let outcome =
                block_on(Pipe::call(&pipe, window)).expect("real NDJSON never hard-fails");
            let Some((frame, consumed)) = outcome else {
                break;
            };
            frames.push(frame);
            total_consumed += consumed;
            accumulator.drain(..consumed);
        }
    }
    (frames, total_consumed)
}

fn feed_uniform(bytes: &[u8], chunk_len: usize) -> (Vec<Bytes>, usize) {
    let chunks: Vec<&[u8]> = bytes.chunks(chunk_len.max(1)).collect();
    drive(&chunks)
}

#[rstest]
#[case::pathological_small_chunk_3(3)]
#[case::pathological_small_chunk_7(7)]
#[case::transport_read_4096(4096)]
#[case::transport_read_8192(8192)]
#[case::transport_read_65536(65536)]
fn misaligned_chunk_sizes_match_a_single_whole_buffer_feed(#[case] chunk_len: usize) {
    let bytes = fixture();
    let (expected_frames, expected_consumed) = feed_uniform(&bytes, bytes.len());
    let (frames, consumed) = feed_uniform(&bytes, chunk_len);

    assert_eq!(frames, expected_frames);
    assert_eq!(consumed, expected_consumed);
    assert_eq!(consumed, bytes.len());
    assert_eq!(
        frames.len(),
        10,
        "the fixture holds exactly ten NDJSON lines"
    );
}

// known, precomputed byte offsets of the first '\n' in the real fixture
// (line 1 is 1434 content bytes) — used to place a chunk boundary exactly
// on / one-before / one-after the delimiter itself.
const FIRST_DELIMITER_OFFSET: usize = 1434;

#[rstest]
#[case::split_immediately_before_the_delimiter(FIRST_DELIMITER_OFFSET)]
#[case::split_exactly_on_the_delimiter(FIRST_DELIMITER_OFFSET + 1)]
#[case::split_immediately_after_the_delimiter(FIRST_DELIMITER_OFFSET + 2)]
fn split_at_a_delimiter_boundary_matches_a_single_whole_buffer_feed(
    #[case] first_chunk_len: usize,
) {
    let bytes = fixture();
    assert_eq!(
        bytes[FIRST_DELIMITER_OFFSET], b'\n',
        "precomputed offset must actually be the first newline"
    );

    let (expected_frames, expected_consumed) = feed_uniform(&bytes, bytes.len());
    let (frames, consumed) = drive(&[&bytes[..first_chunk_len], &bytes[first_chunk_len..]]);

    assert_eq!(frames, expected_frames);
    assert_eq!(consumed, expected_consumed);
}

/// Real captured NDJSON has no multi-byte UTF-8 in it (checked directly on
/// the fixture bytes), so this case is a small, deterministic, hand-built
/// buffer instead — a real 4-byte UTF-8 codepoint (an emoji), split so the
/// chunk boundary lands INSIDE its encoding. `DelimiterCodec` scans bytes,
/// not codepoints, so this must behave identically to any other split;
/// proven here rather than assumed.
#[test]
fn split_mid_multi_byte_utf8_codepoint_still_yields_the_correct_frame() {
    let emoji = '\u{1F600}';
    let mut emoji_bytes = [0_u8; 4];
    let emoji_str = emoji.encode_utf8(&mut emoji_bytes);
    assert_eq!(
        emoji_str.len(),
        4,
        "grinning-face emoji is 4 bytes in UTF-8"
    );

    let line_one = format!("{{\"msg\":\"hi {emoji}\"}}");
    let mut buf = Vec::new();
    buf.extend_from_slice(line_one.as_bytes());
    buf.push(b'\n');
    buf.extend_from_slice(b"{\"msg\":\"next\"}");
    buf.push(b'\n');

    let emoji_offset = line_one.find(emoji).expect("emoji present in line one");
    // two bytes into the emoji's four-byte encoding: a genuine mid-codepoint cut.
    let split_at = emoji_offset + 2;

    let (expected_frames, expected_consumed) = drive(&[&buf]);
    let (frames, consumed) = drive(&[&buf[..split_at], &buf[split_at..]]);

    assert_eq!(frames, expected_frames);
    assert_eq!(consumed, expected_consumed);
    assert_eq!(frames.len(), 2);
    assert_eq!(&frames[0][..], line_one.as_bytes());
    assert_eq!(&frames[1][..], b"{\"msg\":\"next\"}".as_slice());
}
