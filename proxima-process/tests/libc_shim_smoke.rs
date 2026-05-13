#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
//! End-to-end smoke tests for the libc-interpose shim.
//!
//! Three regimes, each pinned by a test:
//!
//! 1. **shim off** — no `libc_shim` opt-in, no DYLD env. shim_probe
//!    runs vanilla `gethostname`. Baseline.
//! 2. **shim on + Empty chain** — shim loads via DYLD; dispatch fd
//!    is set; Empty chain returns empty `ReadResponse`. shim_probe
//!    should print an empty string (the dispatch RT works; the
//!    chain just has nothing to say).
//! 3. **shim on + Canned chain** — shim loads; dispatch fd is set;
//!    Canned chain returns the configured bytes. shim_probe prints
//!    those bytes verbatim. This is the load-bearing proof that the
//!    postcard wire format + framing + dispatch RT all wire end to
//!    end.
//!
//! Each regime exercises a different code path in
//! `src/interpose.c::shim_gethostname`.

use std::ffi::CString;

use bytes::Bytes;
use futures::executor::block_on;
use futures::stream::StreamExt;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::Request;
use proxima_primitives::pipe::handler::Handler;

use proxima_process::command_pipe::CommandPipe;
use proxima_process::descriptor::CommandDescriptor as Command;
use proxima_process::grounds::{Canned, Empty};

fn probe_command() -> Command {
    let path = env!("CARGO_BIN_EXE_shim_probe");
    let mut command = Command::new(CString::new(path).expect("probe path has no NUL"));
    command.inherit_current_env();
    command
}

fn run_and_capture<C>(pipe: C) -> String
where
    C: Handler,
{
    let request = Request::builder()
        .method("POST")
        .path("/")
        .body(Bytes::new())
        .build()
        .expect("request builds");
    let response = block_on(SendPipe::call(&pipe, request)).expect("call");
    let mut stream = response.into_chunk_stream();
    let mut buffer = Vec::new();
    block_on(async {
        while let Some(chunk) = stream.next().await {
            buffer.extend_from_slice(&chunk.expect("body chunk"));
        }
    });
    String::from_utf8(buffer).expect("utf-8 output")
}

#[test]
fn shim_off_returns_real_hostname() {
    let pipe = CommandPipe::builder()
        .command(probe_command())
        .dispatch(Empty)
        .build();
    let output = run_and_capture(pipe);
    assert!(
        !output.trim().is_empty(),
        "baseline gethostname returned empty: {output:?}"
    );
    assert!(
        !output.contains("proxima-shimmed"),
        "baseline must not contain interposed marker: {output:?}"
    );
}

#[test]
fn shim_on_with_empty_chain_returns_empty_via_dispatch() {
    // PROXIMA_DISPATCH_FD is set by spawn_and_dispatch; the shim
    // takes the dispatch path; Empty chain answers `Read` with
    // `ReadResponse { bytes: Vec::new(), eof: true }`. The shim
    // copies zero bytes into the buffer and returns success;
    // shim_probe reads the empty CStr and prints an empty line.
    let pipe = CommandPipe::builder()
        .command(probe_command())
        .dispatch(Empty)
        .libc_shim()
        .build();
    let output = run_and_capture(pipe);
    assert_eq!(
        output.trim(),
        "",
        "Empty dispatch chain must produce empty hostname, got: {output:?}"
    );
}

#[test]
fn shim_on_uname_returns_canned_bytes_for_all_fields() {
    // uname(2) becomes 5 ChildRequest::Read calls (one per
    // utsname field path). Canned chain answers every Read with
    // the same bytes — so all 5 fields receive the canned value.
    // Proves: the uname intercept wires correctly to dispatch RT
    // and the protocol round-trip is path-agnostic (no new
    // ChildRequest variant needed for uname per parity invariant).
    let probe_path = env!("CARGO_BIN_EXE_uname_probe");
    let mut command = Command::new(CString::new(probe_path).expect("path no NUL"));
    command.inherit_current_env();
    let canned_value = "shim-uname-canned";
    let pipe = CommandPipe::builder()
        .command(command)
        .dispatch(Canned::new(Bytes::from_static(canned_value.as_bytes())))
        .libc_shim()
        .build();
    let output = run_and_capture(pipe);
    // Probe prints: sysname|nodename|release|version|machine
    let fields: Vec<&str> = output.trim().split('|').collect();
    assert_eq!(
        fields.len(),
        5,
        "expected 5 utsname fields, got: {output:?}"
    );
    for (i, f) in fields.iter().enumerate() {
        assert_eq!(
            *f, canned_value,
            "field {i} should be {canned_value:?}, got {f:?} (full output: {output:?})"
        );
    }
}

#[test]
fn shim_on_with_canned_chain_returns_chain_bytes() {
    // Canned chain serves the canned bytes as a `Read` response.
    // The shim's postcard decoder unpacks the bytes from the
    // ChildResponse::Read and writes them into the caller's
    // buffer. End-to-end proof that the C-side encoder + parent
    // postcard decoder + parent postcard encoder + C-side decoder
    // all round-trip cleanly through PROXIMA_DISPATCH_FD.
    let canned_value = "honeypot.proxima.local";
    let pipe = CommandPipe::builder()
        .command(probe_command())
        .dispatch(Canned::new(Bytes::from_static(canned_value.as_bytes())))
        .libc_shim()
        .build();
    let output = run_and_capture(pipe);
    assert!(
        output.contains(canned_value),
        "expected canned value {canned_value:?} in output, got: {output:?}"
    );
}
