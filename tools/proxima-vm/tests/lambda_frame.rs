//! The elf guest hand-encodes postcard because it has no allocator.
//! Nothing but this test stops its encoder drifting from the real wire
//! format, so the guest's own `frame.rs` is compiled in here (`#[path]`, the
//! same bytes the guest links) and every frame it produces or consumes is
//! compared against `postcard` over the real `ChildRequest` /
//! `ChildResponse` types.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use proptest::prelude::*;
use proxima_protocols::process::{ChildRequest, ChildResponse, ReadResponse, WriteResponse};

#[path = "../guests/lambda/src/frame.rs"]
mod frame;

use frame::{FRAME_CAPACITY, FrameError, Response};

fn guest_read(handle: i32, max_bytes: u32, offset: u64) -> Vec<u8> {
    let mut buffer = [0_u8; FRAME_CAPACITY];
    let length = frame::encode_read(&mut buffer, handle, max_bytes, offset).expect("encode read");
    buffer[..length].to_vec()
}

fn guest_write(handle: i32, bytes: &[u8]) -> Vec<u8> {
    let mut buffer = vec![0_u8; bytes.len() + 16];
    let length = frame::encode_write(&mut buffer, handle, bytes).expect("encode write");
    buffer[..length].to_vec()
}

fn wire(value: &impl serde::Serialize) -> Vec<u8> {
    postcard::to_allocvec(value).expect("postcard encode")
}

#[test]
fn read_request_matches_postcard_for_the_chunked_read_loop_values() {
    let cases = [
        (0, 256, 0),
        (0, 256, 127),
        (0, 256, 128),
        (0, 256, 300),
        (0, 256, 16_384),
        (0, 256, 262_144),
        (0, u32::MAX, u64::MAX),
        (-1, 0, 1),
        (63, 1, 1),
        (64, 128, 1 << 35),
        (i32::MIN, 16_384, 0),
        (i32::MAX, 2_097_152, 1 << 62),
    ];

    for (handle, max_bytes, offset) in cases {
        let expected = wire(&ChildRequest::Read {
            handle,
            max_bytes,
            offset,
        });
        assert_eq!(
            guest_read(handle, max_bytes, offset),
            expected,
            "Read {{ handle: {handle}, max_bytes: {max_bytes}, offset: {offset} }}"
        );
    }
}

#[test]
fn write_request_matches_postcard_across_the_length_varint_boundaries() {
    let text = include_bytes!("../../../LICENSE-APACHE");

    for length in [0, 1, 126, 127, 128, 129, 255, 256, 300, 5_000] {
        let bytes = &text[..length.min(text.len())];
        let expected = wire(&ChildRequest::Write {
            handle: 1,
            bytes: bytes.to_vec(),
        });
        assert_eq!(guest_write(1, bytes), expected, "Write of {} bytes", bytes.len());
    }
}

#[test]
fn write_request_with_a_multi_byte_handle_matches_postcard() {
    for handle in [0, 1, -1, 64, -65, 1_000_000, i32::MIN, i32::MAX] {
        let expected = wire(&ChildRequest::Write {
            handle,
            bytes: b"HELLO, LAMBDA".to_vec(),
        });
        assert_eq!(guest_write(handle, b"HELLO, LAMBDA"), expected, "handle {handle}");
    }
}

#[test]
fn read_response_decodes_postcard_for_a_chunk_and_for_eof() {
    let chunk = ChildResponse::Read(ReadResponse {
        bytes: b"hello, lambda".to_vec(),
        eof: false,
    });
    let last = ChildResponse::Read(ReadResponse {
        bytes: vec![0xff; 300],
        eof: true,
    });

    let chunk_wire = wire(&chunk);
    let last_wire = wire(&last);

    assert_eq!(
        frame::decode_response(&chunk_wire),
        Ok(Response::Read {
            bytes: b"hello, lambda",
            eof: false,
        })
    );
    assert_eq!(
        frame::decode_response(&last_wire),
        Ok(Response::Read {
            bytes: &[0xff; 300],
            eof: true,
        })
    );
}

#[test]
fn every_other_response_variant_decodes_postcard() {
    let write = wire(&ChildResponse::Write(WriteResponse { bytes_written: 300 }));
    let open = wire(&ChildResponse::Open { handle: -5 });
    let close = wire(&ChildResponse::Close);
    let stat = wire(&ChildResponse::Stat {
        size: 1 << 40,
        mode: 0o444,
        is_directory: false,
    });
    let error = wire(&ChildResponse::Error { errno: 28 });
    let negative_error = wire(&ChildResponse::Error { errno: -9 });

    assert_eq!(frame::decode_response(&write), Ok(Response::Write { bytes_written: 300 }));
    assert_eq!(frame::decode_response(&open), Ok(Response::Open { handle: -5 }));
    assert_eq!(frame::decode_response(&close), Ok(Response::Close));
    assert_eq!(
        frame::decode_response(&stat),
        Ok(Response::Stat {
            size: 1 << 40,
            mode: 0o444,
            is_directory: false,
        })
    );
    assert_eq!(frame::decode_response(&error), Ok(Response::Error { errno: 28 }));
    assert_eq!(frame::decode_response(&negative_error), Ok(Response::Error { errno: -9 }));
}

#[test]
fn trailing_bytes_after_a_reply_are_ignored_like_postcard_from_bytes() {
    let mut reply = wire(&ChildResponse::Close);
    reply.extend_from_slice(&[0xaa; 16]);

    assert_eq!(frame::decode_response(&reply), Ok(Response::Close));
}

#[test]
fn a_truncated_reply_is_an_error_never_a_short_read() {
    let full = wire(&ChildResponse::Read(ReadResponse {
        bytes: b"hello, lambda".to_vec(),
        eof: true,
    }));

    for cut in 0..full.len() {
        assert_eq!(
            frame::decode_response(&full[..cut]),
            Err(FrameError::Truncated),
            "cut at {cut} of {}",
            full.len()
        );
    }
}

#[test]
fn malformed_replies_name_their_defect() {
    assert_eq!(frame::decode_response(&[9]), Err(FrameError::UnknownVariant(9)));
    assert_eq!(
        frame::decode_response(&[0, 0, 2]),
        Err(FrameError::InvalidBool(2))
    );
    assert_eq!(
        frame::decode_response(&[1, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80]),
        Err(FrameError::Truncated)
    );
    assert_eq!(
        frame::decode_response(&[1, 0xff, 0xff, 0xff, 0xff, 0x7f]),
        Err(FrameError::VarintOverflow),
        "a u32 field holding more than 32 bits"
    );
    assert_eq!(
        frame::decode_response(&[0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 1]),
        Err(FrameError::VarintOverflow),
        "an 11-byte varint"
    );
}

#[test]
fn encoding_into_a_buffer_that_is_too_small_fails_instead_of_truncating() {
    let mut tiny = [0_u8; 4];

    assert_eq!(
        frame::encode_read(&mut tiny, 0, 256, 300),
        Err(FrameError::BufferTooSmall)
    );
    assert_eq!(
        frame::encode_write(&mut tiny, 1, b"HELLO"),
        Err(FrameError::BufferTooSmall)
    );
}

proptest! {
    #[test]
    fn any_read_request_matches_postcard(handle: i32, max_bytes: u32, offset: u64) {
        prop_assert_eq!(
            guest_read(handle, max_bytes, offset),
            wire(&ChildRequest::Read { handle, max_bytes, offset })
        );
    }

    #[test]
    fn any_write_request_matches_postcard(handle: i32, bytes in prop::collection::vec(any::<u8>(), 0..400)) {
        prop_assert_eq!(
            guest_write(handle, &bytes),
            wire(&ChildRequest::Write { handle, bytes: bytes.clone() })
        );
    }

    #[test]
    fn any_read_response_decodes_postcard(bytes in prop::collection::vec(any::<u8>(), 0..400), eof: bool) {
        let encoded = wire(&ChildResponse::Read(ReadResponse { bytes: bytes.clone(), eof }));
        prop_assert_eq!(
            frame::decode_response(&encoded),
            Ok(Response::Read { bytes: &bytes, eof })
        );
    }

    #[test]
    fn any_error_write_open_response_decodes_postcard(number: i32, count: u32) {
        let error = wire(&ChildResponse::Error { errno: number });
        let open = wire(&ChildResponse::Open { handle: number });
        let write = wire(&ChildResponse::Write(WriteResponse { bytes_written: count }));

        prop_assert_eq!(frame::decode_response(&error), Ok(Response::Error { errno: number }));
        prop_assert_eq!(frame::decode_response(&open), Ok(Response::Open { handle: number }));
        prop_assert_eq!(
            frame::decode_response(&write),
            Ok(Response::Write { bytes_written: count })
        );
    }

    #[test]
    fn any_stat_response_decodes_postcard(size: u64, mode: u32, is_directory: bool) {
        let stat = wire(&ChildResponse::Stat { size, mode, is_directory });

        prop_assert_eq!(
            frame::decode_response(&stat),
            Ok(Response::Stat { size, mode, is_directory })
        );
    }
}
