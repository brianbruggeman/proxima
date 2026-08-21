//! Signed-subprocess probe for `dispatch::run_hypercall_guest`: boots the
//! synthesized one-hypercall guest against a real hypervisor, dispatches
//! the pinned `ChildRequest::Read` payload (the same bytes
//! `dispatch.rs`'s `wire_format_round_trips_for_parity` pins) through a
//! `RecordingDispatcher`, and writes the postcard-encoded response to
//! stdout — `tests/dispatch_hypercall.rs` decodes it and asserts against
//! the configured response.

use std::error::Error;
use std::io::{self, Write};

use proxima_protocols::process::{ChildResponse, ReadResponse};
use proxima_vm::dispatch::{self, RecordingDispatcher};

/// Postcard variant discriminant for `ChildRequest::Read`, reused as the
/// hypercall verb — matches `dispatch.rs`'s test constant of the same name.
const CHILD_REQUEST_READ_VERB: u16 = 0x00;

/// `ChildRequest::Read { path: "/etc/hostname", max_bytes: 256, offset: 0 }`,
/// postcard-encoded, byte-for-byte the buffer `dispatch.rs`'s
/// `wire_format_round_trips_for_parity` pins as `expected`.
const CHILD_REQUEST_READ_WIRE_BYTES: [u8; 18] = [
    0x00, 13, b'/', b'e', b't', b'c', b'/', b'h', b'o', b's', b't', b'n', b'a', b'm', b'e', 0x80,
    0x02, 0x00,
];

const RESPONSE_CAPACITY: usize = 256;

fn main() -> Result<(), Box<dyn Error>> {
    let configured = ChildResponse::Read(ReadResponse {
        bytes: b"vm-side-canned".to_vec(),
        eof: true,
    });
    let dispatcher = RecordingDispatcher::new(configured);

    let response = dispatch::run_hypercall_guest(
        &dispatcher,
        CHILD_REQUEST_READ_VERB,
        &CHILD_REQUEST_READ_WIRE_BYTES,
        RESPONSE_CAPACITY,
    )?;

    io::stdout().write_all(&response)?;
    Ok(())
}
