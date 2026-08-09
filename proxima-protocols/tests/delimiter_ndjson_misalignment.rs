#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Misaligned-chunk correctness for `FrameCodecPipe<DelimiterCodec>` driven
//! over a REAL NDJSON duplex capture (`fixtures/ndjson/duplex.ndjson`) — one
//! turn's worth of `stream-json` traffic, not a synthetic fixture. Proves
//! the pipe adapter's per-call `parse_frame` re-scan yields the IDENTICAL
//! frame sequence and total consumed byte count no matter where a
//! transport hands bytes across in — including chunk boundaries that land
//! exactly on, one byte before, and one byte after a `\n` delimiter, where
//! a naive scan-from-the-wrong-offset bug would most likely surface.

use core::future::Future;

use bytes::{Bytes, BytesMut};
use proxima_codec::{DelimiterCodec, FrameLimits};
use proxima_primitives::pipe::Pipe;
use proxima_protocols::codec_pipe::FrameCodecPipe;

const FIXTURE: &[u8] = include_bytes!("fixtures/ndjson/duplex.ndjson");

/// Dependency-free executor for the always-ready probe futures — mirrors
/// `proxima_protocols::codec_pipe`'s own `block_on` test helper.
fn block_on<Fut: Future>(future: Fut) -> Fut::Output {
    let mut pinned = core::pin::pin!(future);
    let mut context = core::task::Context::from_waker(core::task::Waker::noop());
    loop {
        if let core::task::Poll::Ready(output) = pinned.as_mut().poll(&mut context) {
            return output;
        }
    }
}

/// Feeds `chunks` through `pipe` one at a time, appending to a growing
/// buffer and draining every complete frame after each chunk — a real
/// transport read loop, not a whole-buffer shortcut.
fn drive(pipe: &FrameCodecPipe<DelimiterCodec>, chunks: &[&[u8]]) -> (Vec<Bytes>, usize) {
    let mut buf = BytesMut::new();
    let mut frames = Vec::new();
    let mut total_consumed = 0_usize;

    for chunk in chunks {
        buf.extend_from_slice(chunk);
        loop {
            let window = Bytes::copy_from_slice(&buf);
            let outcome = block_on(Pipe::call(pipe, window)).expect("real ndjson never errors");
            match outcome {
                Some((frame, consumed)) => {
                    frames.push(frame);
                    total_consumed += consumed;
                    let _ = buf.split_to(consumed);
                }
                None => break,
            }
        }
    }
    (frames, total_consumed)
}

fn fixed_size_chunks(bytes: &[u8], size: usize) -> Vec<&[u8]> {
    bytes.chunks(size).collect()
}

fn first_delimiter_index(bytes: &[u8], delimiter: u8) -> usize {
    bytes
        .iter()
        .position(|&byte| byte == delimiter)
        .expect("fixture has at least one ndjson line")
}

#[test]
fn misaligned_chunks_reproduce_the_whole_buffer_frame_sequence() {
    let codec = DelimiterCodec::new(b"\n", FrameLimits::default());
    let pipe = FrameCodecPipe::new(codec);

    let (reference_frames, reference_consumed) = drive(&pipe, &[FIXTURE]);
    assert_eq!(reference_consumed, FIXTURE.len());
    assert!(
        !reference_frames.is_empty(),
        "fixture must yield at least one frame"
    );

    for chunk_size in [3_usize, 7, 4096, 8192, 65536] {
        let chunks = fixed_size_chunks(FIXTURE, chunk_size);
        let (frames, consumed) = drive(&pipe, &chunks);
        assert_eq!(
            frames, reference_frames,
            "chunk_size {chunk_size} produced a different frame sequence"
        );
        assert_eq!(
            consumed, reference_consumed,
            "chunk_size {chunk_size} consumed a different byte count"
        );
    }

    let delimiter_index = first_delimiter_index(FIXTURE, b'\n');
    let boundary_splits = [
        delimiter_index,
        delimiter_index.saturating_sub(1),
        delimiter_index + 1,
    ];
    for split_at in boundary_splits {
        let (head, tail) = FIXTURE.split_at(split_at);
        let (frames, consumed) = drive(&pipe, &[head, tail]);
        assert_eq!(
            frames, reference_frames,
            "split at byte {split_at} (first delimiter at {delimiter_index}) \
             produced a different frame sequence"
        );
        assert_eq!(
            consumed, reference_consumed,
            "split at byte {split_at} consumed a different byte count"
        );
    }
}
