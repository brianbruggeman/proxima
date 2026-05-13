#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
//! Compare-bench for the libc-interpose shim (gate point 6).
//!
//! # Scope
//!
//! Per the project-memory invariant
//! `proxima.failure.hardened_dyld_interpose` (in
//! `proxima/ai_docs/invariants.jsonl`), the libc-interpose shim
//! works for **owned children only**. A signed macOS binary with
//! hardened runtime does NOT load the dylib. The home-turf
//! comparison here is therefore against the vanilla syscall on
//! an owned binary — NOT a claim about third-party containment (that
//! lives behind `proxima-vm`).
//!
//! # Incumbent design point
//!
//! Vanilla `libc::gethostname()` is the incumbent. It does a
//! single `uname(2)` syscall and copies the result into the
//! caller's buffer. The shim's intercepted variant currently
//! copies a static string (`"proxima-shimmed"`) with no
//! syscall + no protocol round-trip — so this bench measures
//! pure interposition overhead at the dyld level.
//!
//! Once C8c lands (protocol round-trip via PROXIMA_DISPATCH_FD),
//! the intercepted variant will add socket I/O cost; re-bench
//! at that point and record both arms.
//!
//! # Arms
//!
//! - `gethostname/vanilla`         — `design-favors: incumbent`. Real syscall.
//! - `gethostname/shim_stub`       — `design-favors: proxima`. Static string return; lower bound on interposition cost.
//! - `gethostname/shim_dispatch`   — `design-favors: incumbent`. PENDING C8c — adds socket round-trip; this is where the honest perf claim lives.
//!
//! # Running
//!
//! Without the shim loaded (vanilla path):
//! ```sh
//! cargo bench --bench bench_libc_shim -- gethostname/vanilla
//! ```
//!
//! With the shim loaded (intercepted path) — set DYLD/LD env first:
//! ```sh
//! # macOS
//! DYLD_INSERT_LIBRARIES=$(cargo metadata --format-version 1 \
//!     | jq -r '.target_directory')/debug/build/proxima-process-*/out/libproxima_process_shim.dylib \
//!     cargo bench --bench bench_libc_shim -- gethostname/shim_stub
//! # Linux
//! LD_PRELOAD=... cargo bench --bench bench_libc_shim -- gethostname/shim_stub
//! ```
//!
//! Cargo benches themselves don't auto-load the shim — the
//! interpose env var only affects spawned children, but the
//! bench process IS the child of cargo. Loading at bench time
//! is brittle (PATH-search before DYLD_*, hardened runtime
//! considerations); the practical recipe is the `cargo bench`
//! wrapped in a shell `env` invocation as above.
//!
//! # CoV discipline (per disciplined-component skill)
//!
//! libc syscall benchmarks have high noise floor on shared
//! hosts. Re-run 3-5 times and record the RANGE, not a single
//! number. Use `--save-baseline` to pin a known-quiet result
//! and compare against it for tweaks.

use criterion::{Criterion, criterion_group, criterion_main};
use std::ffi::CStr;
use std::hint::black_box;

fn bench_gethostname_vanilla(c: &mut Criterion) {
    let mut group = c.benchmark_group("gethostname");
    group.bench_function("vanilla", |b| {
        let mut buffer = [0u8; 256];
        b.iter(|| {
            // SAFETY: fixed-size buffer, libc writes ≤ len bytes,
            // NUL-terminates on success.
            let result = unsafe {
                libc::gethostname(buffer.as_mut_ptr().cast::<libc::c_char>(), buffer.len())
            };
            black_box(result);
            // Read back to force the OS write to be observable.
            let name = CStr::from_bytes_until_nul(&buffer)
                .map(CStr::to_bytes)
                .unwrap_or(b"");
            black_box(name.len());
        });
    });
    group.finish();
}

// `shim_stub` arm is the SAME code as `vanilla` — the bench
// harness can't tell whether libc::gethostname is the real one
// or the interposed one. The shim is selected by the loader at
// process start via DYLD_INSERT_LIBRARIES / LD_PRELOAD. Run this
// arm with the env var set; without, it produces the vanilla
// number (use the explicit `vanilla` arm for that case).
fn bench_gethostname_shim_stub(c: &mut Criterion) {
    let mut group = c.benchmark_group("gethostname");
    group.bench_function("shim_stub", |b| {
        let mut buffer = [0u8; 256];
        b.iter(|| {
            let result = unsafe {
                libc::gethostname(buffer.as_mut_ptr().cast::<libc::c_char>(), buffer.len())
            };
            black_box(result);
            let name = CStr::from_bytes_until_nul(&buffer)
                .map(CStr::to_bytes)
                .unwrap_or(b"");
            black_box(name.len());
        });
    });
    group.finish();
}

// `shim_dispatch` arm — exercises the FULL stage-2 dispatch RT
// (PROXIMA_DISPATCH_FD → postcard ChildRequest → handler thread →
// postcard ChildResponse → return).
//
// Setup wires a same-process socketpair, dup2's the child end
// onto DISPATCH_FD_TARGET=7, sets PROXIMA_DISPATCH_FD=7, and
// spawns a handler thread that decodes ChildRequest frames and
// emits canned ChildResponse::Read frames. The bench itself
// loops `libc::gethostname`; with DYLD_INSERT_LIBRARIES set, the
// shim's `dispatch_read_hostname` does the RT through fd 7.
//
// Measures: full intercepted-call cost including the round-trip
// syscall pair (write + read on the same-process socket) + the
// shim's encode/decode + the handler's decode/encode. This is
// the load-bearing perf number for the libc-shim component.
//
// Run with: `DYLD_INSERT_LIBRARIES=<.../libproxima_process_shim.dylib>
//            cargo bench --bench bench_libc_shim --
//            'gethostname/shim_dispatch'`
fn bench_gethostname_shim_dispatch(c: &mut Criterion) {
    use std::io::{Read, Write};
    use std::os::fd::IntoRawFd;
    use std::os::unix::net::UnixStream;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

    const DISPATCH_FD_TARGET: libc::c_int = 7;
    const CANNED_HOSTNAME: &[u8] = b"bench-dispatch-host";

    // socketpair: (parent_end, child_end). Handler thread reads
    // requests from parent_end and writes responses back; the
    // shim (running in this bench process) talks via child_end
    // dup2'd onto fd 7.
    let (parent_end, child_end) = UnixStream::pair().expect("socketpair");

    // dup2 child_end onto DISPATCH_FD_TARGET so the shim finds
    // it at the canonical fd number. Keep child_end alive so its
    // Drop doesn't close the underlying socket while we still
    // need it referenced from fd 7.
    let child_raw = child_end.into_raw_fd();
    if child_raw != DISPATCH_FD_TARGET {
        // SAFETY: dup2 on owned fds; closes whatever was at
        // DISPATCH_FD_TARGET before (typically nothing in a fresh
        // bench process), and leaves child_raw open under both
        // numbers. We close child_raw after.
        let result = unsafe { libc::dup2(child_raw, DISPATCH_FD_TARGET) };
        assert!(result >= 0, "dup2 onto DISPATCH_FD_TARGET failed");
        unsafe { libc::close(child_raw) };
    }

    // Tell the shim where to find the dispatch fd.
    // SAFETY: setenv is thread-safe enough for our bench setup
    // (before any iter spawns threads beyond the handler).
    unsafe { std::env::set_var("PROXIMA_DISPATCH_FD", "7") };

    // Spawn handler thread. Reads framed ChildRequest::Read,
    // emits framed ChildResponse::Read with CANNED_HOSTNAME.
    let stop = Arc::new(AtomicBool::new(false));
    let handler_stop = Arc::clone(&stop);
    let mut handler_stream = parent_end;
    let handler = thread::spawn(move || {
        let mut prefix = [0u8; 4];
        let mut payload = vec![0u8; 1024];
        while !handler_stop.load(Ordering::Acquire) {
            // Read frame prefix
            if handler_stream.read_exact(&mut prefix).is_err() {
                break;
            }
            let len = u32::from_be_bytes(prefix) as usize;
            if len > payload.len() {
                payload.resize(len, 0);
            }
            if handler_stream.read_exact(&mut payload[..len]).is_err() {
                break;
            }
            // We don't decode the request — we know it's
            // ChildRequest::Read for the hostname path. Emit
            // the same canned ChildResponse::Read every time.
            // Encode: [discriminant=0][bytes_len varint][bytes][eof=1]
            let mut resp = Vec::with_capacity(CANNED_HOSTNAME.len() + 8);
            resp.push(0u8); // ChildResponse::Read discriminant
            // bytes_len varint (single-byte for small)
            let bytes_len = CANNED_HOSTNAME.len() as u64;
            if bytes_len < 0x80 {
                resp.push(bytes_len as u8);
            } else {
                // (shouldn't happen for our short canned string)
                let mut v = bytes_len;
                while v >= 0x80 {
                    resp.push(((v & 0x7f) | 0x80) as u8);
                    v >>= 7;
                }
                resp.push(v as u8);
            }
            resp.extend_from_slice(CANNED_HOSTNAME);
            resp.push(1u8); // eof=true
            let resp_prefix = (resp.len() as u32).to_be_bytes();
            if handler_stream.write_all(&resp_prefix).is_err() {
                break;
            }
            if handler_stream.write_all(&resp).is_err() {
                break;
            }
        }
    });

    let mut group = c.benchmark_group("gethostname");
    group.bench_function("shim_dispatch", |b| {
        let mut buffer = [0u8; 256];
        b.iter(|| {
            let result = unsafe {
                libc::gethostname(buffer.as_mut_ptr().cast::<libc::c_char>(), buffer.len())
            };
            black_box(result);
            black_box(buffer[0]);
        });
    });
    group.finish();

    // Tear down: stop handler, close the dispatch fd to unblock
    // its read.
    stop.store(true, Ordering::Release);
    unsafe { libc::close(DISPATCH_FD_TARGET) };
    let _ = handler.join();
}

criterion_group!(
    benches,
    bench_gethostname_vanilla,
    bench_gethostname_shim_stub,
    bench_gethostname_shim_dispatch,
);
criterion_main!(benches);
