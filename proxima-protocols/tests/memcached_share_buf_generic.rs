#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Acceptance test for component C4 — "send my own struct through": a
//! caller-supplied buffer type ([`ArcSlice`], built with NO `bytes`
//! dependency of its own) drives `MemcachedCodec::<ArcSlice>` end to end,
//! and the owned frame it produces shares the SAME backing allocation the
//! wire bytes arrived in — zero-copy, not parameterized-on-paper.

use std::sync::Arc;

use proxima_codec::{FrameCodec, ShareBuf};
use proxima_protocols::codec_pipe::OwnFrame;
use proxima_protocols::memcached::frame_codec::{MemcachedCodec, MemcachedOwnedFrame};
use proxima_protocols::memcached::pipe_contract::{MemcachedRequest, iter_keys};

const DEFAULT_MAX_MESSAGE_BYTES: usize = 128 * 1024;

/// A non-`bytes` buffer: an `Arc`-backed window with its own `start`/`end`
/// span — `Deref<Target = [u8]>` + `Clone` + `ShareBuf`, no `bytes::Bytes`
/// anywhere in its construction.
#[derive(Debug, Clone)]
struct ArcSlice {
    backing: Arc<[u8]>,
    start: usize,
    end: usize,
}

impl ArcSlice {
    fn from_wire(wire: Vec<u8>) -> Self {
        let end = wire.len();
        Self {
            backing: Arc::from(wire),
            start: 0,
            end,
        }
    }
}

impl core::ops::Deref for ArcSlice {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.backing[self.start..self.end]
    }
}

impl ShareBuf for ArcSlice {
    fn share(&self, subset: &[u8]) -> Self {
        let backing_start = self.backing.as_ptr() as usize;
        let backing_end = backing_start + self.backing.len();
        let sub_start = subset.as_ptr() as usize;
        let sub_end = sub_start + subset.len();

        assert!(
            sub_start >= backing_start && sub_end <= backing_end,
            "subset must be a pointer-derived sub-slice of this ArcSlice's own backing"
        );

        Self {
            backing: Arc::clone(&self.backing),
            start: sub_start - backing_start,
            end: sub_end - backing_start,
        }
    }
}

fn codec() -> MemcachedCodec<ArcSlice> {
    MemcachedCodec::new(DEFAULT_MAX_MESSAGE_BYTES)
}

#[test]
fn set_command_drives_zero_copy_through_a_caller_supplied_buffer() {
    let wire = ArcSlice::from_wire(b"set mykey 5 60 5\r\nhello\r\n".to_vec());
    let wire_ptr = wire.backing.as_ptr();

    let (frame, consumed) = codec().parse_frame(&wire).expect("parses");
    assert_eq!(consumed, wire.len());

    let owned = MemcachedCodec::<ArcSlice>::own_frame(&wire, &frame);
    match owned {
        MemcachedOwnedFrame::Request(MemcachedRequest::Store {
            key, flags, exptime, value, noreply, ..
        }) => {
            assert_eq!(&*key, b"mykey");
            assert_eq!(flags, 5);
            assert_eq!(exptime, 60);
            assert_eq!(&*value, b"hello");
            assert!(!noreply);
            assert_eq!(
                key.backing.as_ptr(),
                wire_ptr,
                "owned Store::key must share the SAME allocation as the wire, not copy it"
            );
            assert_eq!(
                value.backing.as_ptr(),
                wire_ptr,
                "owned Store::value must share the SAME allocation as the wire, not copy it"
            );
        }
        other => panic!("unexpected owned frame: {other:?}"),
    }
}

#[test]
fn multi_get_keys_walk_zero_copy_through_a_caller_supplied_buffer() {
    let wire = ArcSlice::from_wire(b"get a b c\r\n".to_vec());
    let wire_ptr = wire.backing.as_ptr();

    let (frame, consumed) = codec().parse_frame(&wire).expect("parses");
    assert_eq!(consumed, wire.len());

    let owned = MemcachedCodec::<ArcSlice>::own_frame(&wire, &frame);
    match owned {
        MemcachedOwnedFrame::Request(MemcachedRequest::Get { keys, gets }) => {
            assert!(!gets);
            assert_eq!(&*keys, b"a b c");
            assert_eq!(keys.backing.as_ptr(), wire_ptr);

            let collected: Vec<Vec<u8>> = iter_keys(&keys).map(|key| key.to_vec()).collect();
            assert_eq!(collected, vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]);
        }
        other => panic!("unexpected owned frame: {other:?}"),
    }
}
