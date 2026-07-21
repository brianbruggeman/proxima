#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
//! Tests for the dispatch layer.
//!
//! Each test exercises the layer at a specific level:
//! - protocol round-trip
//! - marker trait impls (compile-time assertions)
//! - capability tokens (compile-time + construction)
//! - path validation (compile-time + runtime)
//! - taint conversions
//! - ground dispatch correctness
//! - operator composition with marker propagation
//! - hand-written Match routing

use super::capabilities::{CapFilesystem, CapNetwork, CapSpawn};
use super::framing::{FrameDecoder, FrameEncoder, decode_frame, encode_frame};
use super::grounds::{Canned, Deny, Empty, canned, deny_writes, empty};
use super::host_grounds::{FixedClock, HostRead, HostWrite, OsEntropy, SeededEntropy};
use super::markers::{
    AllocFree, Commutative, Deterministic, IdempotentSideEffectFree, IsPure, NoStd, Reproducible,
    WithoutFilesystem, WithoutNetwork, WithoutRandom, WithoutSpawn, WithoutTime,
};
use super::operators::{AndThen, dispatch_match};
use super::path::{AbsolutePath, AbsolutePathError};
use super::protocol::{ChildRequest, ChildResponse, ReadResponse, WriteResponse};
use super::taint::{Tainted, sanitize_absolute_path, trust_const_path};
use futures::executor::block_on;
use proxima_primitives::pipe::Pipe;
use proxima_primitives::pipe::PipeExt;
use proxima_primitives::pipe::ProximaError;
use proxima_primitives::pipe::SendPipe;

/// Test-only helper: synchronous wrapper for `SendPipe::call`.
/// Pipe calls return futures; tests want a sync method-call ergonomic.
trait BlockOnPipe: SendPipe {
    fn call_sync(&self, input: Self::In) -> Self::Out
    where
        Self: Sync,
    {
        block_on(SendPipe::call(self, input)).expect("pipe call failed in test")
    }
}
impl<T: SendPipe> BlockOnPipe for T {}

// ---- Compile-time marker trait assertions ----

#[allow(dead_code)]
fn assert_no_std<T: NoStd>() {}
#[allow(dead_code)]
fn assert_alloc_free<T: AllocFree>() {}
#[allow(dead_code)]
fn assert_is_pure<T: IsPure>() {}
#[allow(dead_code)]
fn assert_deterministic<T: Deterministic>() {}
#[allow(dead_code)]
fn assert_reproducible<T: Reproducible>() {}
#[allow(dead_code)]
fn assert_idempotent<T: IdempotentSideEffectFree>() {}
#[allow(dead_code)]
fn assert_commutative<T: Commutative>() {}
#[allow(dead_code)]
fn assert_without_filesystem<T: WithoutFilesystem>() {}
#[allow(dead_code)]
fn assert_without_network<T: WithoutNetwork>() {}
#[allow(dead_code)]
fn assert_without_spawn<T: WithoutSpawn>() {}
#[allow(dead_code)]
fn assert_without_time<T: WithoutTime>() {}
#[allow(dead_code)]
fn assert_without_random<T: WithoutRandom>() {}

#[test]
fn empty_qualifies_for_all_purity_markers() {
    assert_no_std::<Empty>();
    assert_alloc_free::<Empty>();
    assert_is_pure::<Empty>();
    assert_deterministic::<Empty>();
    assert_reproducible::<Empty>();
    assert_idempotent::<Empty>();
    assert_commutative::<Empty>();
}

#[test]
fn canned_qualifies_for_purity_but_not_alloc_free() {
    assert_no_std::<Canned>();
    assert_is_pure::<Canned>();
    assert_deterministic::<Canned>();
    assert_reproducible::<Canned>();
    assert_idempotent::<Canned>();
    // Canned does NOT impl AllocFree — Read responses allocate a Vec
    // for the returned bytes. Documented in grounds.rs.
}

#[test]
fn deny_qualifies_for_all_purity_markers() {
    assert_no_std::<Deny>();
    assert_alloc_free::<Deny>();
    assert_is_pure::<Deny>();
    assert_deterministic::<Deny>();
    assert_reproducible::<Deny>();
    assert_idempotent::<Deny>();
    assert_commutative::<Deny>();
}

#[test]
fn series_of_two_no_std_grounds_is_no_std() {
    assert_no_std::<AndThen<Empty, Deny>>();
}

#[test]
fn series_of_two_alloc_free_grounds_is_alloc_free() {
    assert_alloc_free::<AndThen<Empty, Deny>>();
}

#[test]
fn series_of_two_pure_grounds_is_pure() {
    assert_is_pure::<AndThen<Empty, Deny>>();
}

#[test]
fn series_propagates_deterministic() {
    assert_deterministic::<AndThen<Canned, Empty>>();
    assert_deterministic::<AndThen<Empty, Deny>>();
}

// ---- Without* effect-absence markers ----

#[test]
fn pure_grounds_carry_all_without_markers() {
    assert_without_filesystem::<Empty>();
    assert_without_network::<Empty>();
    assert_without_spawn::<Empty>();
    assert_without_time::<Empty>();
    assert_without_random::<Empty>();

    assert_without_filesystem::<Canned>();
    assert_without_network::<Canned>();
    assert_without_spawn::<Canned>();
    assert_without_time::<Canned>();
    assert_without_random::<Canned>();

    assert_without_filesystem::<Deny>();
    assert_without_network::<Deny>();
    assert_without_spawn::<Deny>();
    assert_without_time::<Deny>();
    assert_without_random::<Deny>();
}

#[test]
fn series_propagates_without_filesystem_via_and() {
    assert_without_filesystem::<AndThen<Empty, Deny>>();
    assert_without_filesystem::<AndThen<Canned, Empty>>();
    assert_without_filesystem::<AndThen<AndThen<Canned, Empty>, Deny>>();
}

#[test]
fn series_propagates_all_without_markers() {
    type C = AndThen<Canned, Empty>;
    assert_without_filesystem::<C>();
    assert_without_network::<C>();
    assert_without_spawn::<C>();
    assert_without_time::<C>();
    assert_without_random::<C>();
}

// ---- Wide grounds: HostRead / HostWrite ----

#[test]
fn host_read_requires_capability_at_construction() {
    let cap = CapFilesystem::grant();
    let _read = HostRead::new("/etc/passwd", &cap);
}

#[test]
fn host_read_carries_non_filesystem_without_markers() {
    assert_without_network::<HostRead>();
    assert_without_spawn::<HostRead>();
    assert_without_time::<HostRead>();
    assert_without_random::<HostRead>();
    // HostRead does NOT impl WithoutFilesystem — structurally enforced.
}

#[test]
fn host_write_carries_non_filesystem_without_markers() {
    assert_without_network::<HostWrite>();
    assert_without_spawn::<HostWrite>();
    assert_without_time::<HostWrite>();
    assert_without_random::<HostWrite>();
}

// ---- Wide grounds: Entropy split ----

#[test]
fn seeded_entropy_is_deterministic_and_reproducible() {
    assert_deterministic::<SeededEntropy>();
    assert_reproducible::<SeededEntropy>();
    assert_idempotent::<SeededEntropy>();
}

#[test]
fn seeded_entropy_carries_non_random_without_markers() {
    assert_without_filesystem::<SeededEntropy>();
    assert_without_network::<SeededEntropy>();
    assert_without_spawn::<SeededEntropy>();
    assert_without_time::<SeededEntropy>();
    // NOT WithoutRandom — IS random-shaped, just deterministic.
}

#[test]
fn seeded_entropy_same_seed_produces_same_bytes() {
    let entropy = SeededEntropy::new(42);
    let request = entropy.call_sync(ChildRequest::Read {
        path: "/dev/urandom".to_string(),
        max_bytes: 16,
        offset: 0,
    });
    let bytes_first = match request {
        ChildResponse::Read(ReadResponse { bytes, .. }) => bytes,
        other => panic!("expected Read response, got {other:?}"),
    };

    let again = entropy.call_sync(ChildRequest::Read {
        path: "/dev/urandom".to_string(),
        max_bytes: 16,
        offset: 0,
    });
    let bytes_second = match again {
        ChildResponse::Read(ReadResponse { bytes, .. }) => bytes,
        other => panic!("expected Read response, got {other:?}"),
    };

    assert_eq!(
        bytes_first, bytes_second,
        "seeded entropy must be reproducible"
    );
}

#[test]
fn os_entropy_carries_non_random_without_markers() {
    assert_without_filesystem::<OsEntropy>();
    assert_without_network::<OsEntropy>();
    assert_without_spawn::<OsEntropy>();
    assert_without_time::<OsEntropy>();
    // NOT WithoutRandom, NOT Deterministic — structurally enforced.
}

// ---- Wide grounds: Clock split ----

#[test]
fn fixed_clock_is_deterministic_and_reproducible() {
    assert_deterministic::<FixedClock>();
    assert_reproducible::<FixedClock>();
    assert_idempotent::<FixedClock>();
}

#[test]
fn fixed_clock_carries_non_time_without_markers() {
    assert_without_filesystem::<FixedClock>();
    assert_without_network::<FixedClock>();
    assert_without_spawn::<FixedClock>();
    assert_without_random::<FixedClock>();
}

#[test]
fn fixed_clock_emits_configured_epoch_decimal() {
    let clock = FixedClock::new(1_700_000_000);
    let response = clock.call_sync(ChildRequest::Read {
        path: "/proc/uptime".to_string(),
        max_bytes: 32,
        offset: 0,
    });
    match response {
        ChildResponse::Read(ReadResponse { bytes, .. }) => {
            assert_eq!(bytes, b"1700000000");
        }
        other => panic!("expected Read response, got {other:?}"),
    }
}

// ---- AndThen with wide grounds: marker propagation ----

#[test]
fn series_propagates_deterministic_through_wide_grounds() {
    assert_deterministic::<AndThen<SeededEntropy, FixedClock>>();
    assert_reproducible::<AndThen<SeededEntropy, FixedClock>>();
}

// ---- spawn_and_dispatch: wired CommandDescriptor + dispatch chain ----

#[test]
fn spawn_and_dispatch_runs_true_to_completion() {
    use super::CommandDescriptor;
    use super::dispatched::spawn_and_dispatch;
    use std::ffi::CString;

    let mut command = CommandDescriptor::new(CString::new("/usr/bin/true").unwrap());
    command.inherit_current_env();

    // Any pipe; /bin/true won't touch the dispatch fd.
    let pipe = deny_writes();

    let dispatched = spawn_and_dispatch(&command, pipe).expect("spawn_and_dispatch");
    let status = dispatched.wait().expect("wait");

    assert!(
        libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0,
        "expected /bin/true to exit cleanly, got status={status}"
    );
}

#[test]
fn spawn_and_dispatch_sets_dispatch_fd_env_var_for_child() {
    use super::CommandDescriptor;
    use super::dispatched::{DISPATCH_FD_ENV, spawn_and_dispatch};
    use std::ffi::CString;
    use std::io::Read;
    use std::os::fd::{FromRawFd, IntoRawFd};

    // Spawn /bin/sh that prints the env var value to a piped output.
    // We can't read stdin from the parent here without async wiring,
    // so use /bin/sh with output piped via the existing Stdio::Piped.
    let mut command = CommandDescriptor::new(CString::new("/bin/sh").unwrap());
    command
        .inherit_current_env()
        .arg(CString::new("-c").unwrap())
        .arg(CString::new(format!("printf '%s' \"${DISPATCH_FD_ENV}\"")).unwrap())
        .stdout(super::Stdio::Piped);

    let pipe = deny_writes();
    let mut dispatched = spawn_and_dispatch(&command, pipe).expect("spawn_and_dispatch");

    // Read the child's stdout pipe.
    let output_fd = dispatched
        .child
        .stdout
        .take()
        .expect("stdout should be piped");
    let mut file = unsafe { std::fs::File::from_raw_fd(output_fd.into_raw_fd()) };
    let mut captured = String::new();
    file.read_to_string(&mut captured)
        .expect("read child output");

    let status = dispatched.wait().expect("wait");
    assert!(libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0);
    assert_eq!(
        captured.trim(),
        "7",
        "PROXIMA_DISPATCH_FD should be 7 in the child env, got {captured:?}"
    );
}

// ---- IPC layer: socketpair-driven dispatch loop ----

#[test]
fn dispatch_loop_handles_a_single_round_trip_over_socketpair() {
    use super::ipc::{read_frame, run_dispatch_loop, write_frame};
    use std::os::unix::net::UnixStream;
    use std::thread;

    // Parent end + shim end of a socketpair.
    let (parent_end, shim_end) = UnixStream::pair().expect("socketpair");

    // Shim simulator: write one framed request, read one framed
    // response, drop the connection to signal EOF.
    let shim_handle = thread::spawn(move || {
        let mut shim_end = shim_end;
        let request = ChildRequest::Read {
            path: "/proc/sys/kernel/hostname".to_string(),
            max_bytes: 256,
            offset: 0,
        };
        write_frame(&mut shim_end, &request).expect("shim write");
        let response: ChildResponse = read_frame(&mut shim_end)
            .expect("shim read")
            .expect("response present");
        // Drop shim_end here → parent's dispatch loop sees EOF.
        drop(shim_end);
        response
    });

    // Parent dispatch loop with a single canned hostname.
    let mut parent_read = parent_end.try_clone().expect("dup parent end for read");
    let mut parent_write = parent_end;
    let hostname = canned(&b"honeypot.proxima.local"[..]);
    let fallback = deny_writes();
    let routes: &[(
        &str,
        &dyn proxima_primitives::pipe::alloc_tier::SendDynPipe<ChildRequest, ChildResponse>,
    )] = &[("/proc/sys/kernel/hostname", &hostname)];
    run_dispatch_loop(&mut parent_read, &mut parent_write, |request| {
        block_on(dispatch_match(request, routes, &fallback))
    })
    .expect("dispatch loop should terminate cleanly on EOF");

    let response = shim_handle.join().expect("shim thread joins");
    match &response {
        ChildResponse::Read(ReadResponse { bytes, .. }) => {
            assert_eq!(
                bytes.as_slice(),
                b"honeypot.proxima.local",
                "shim received honeypot hostname"
            );
        }
        other => panic!("expected Read response, got {other:?}"),
    }
}

#[test]
fn dispatch_loop_handles_multiple_requests_in_one_session() {
    use super::ipc::{read_frame, run_dispatch_loop, write_frame};
    use std::os::unix::net::UnixStream;
    use std::thread;

    let (parent_end, shim_end) = UnixStream::pair().expect("socketpair");

    let shim_handle = thread::spawn(move || -> Vec<ChildResponse> {
        let mut shim_end = shim_end;
        let mut responses = Vec::new();
        for path in [
            "/proc/sys/kernel/hostname",
            "/proc/sys/kernel/osrelease",
            "/proc/sys/kernel/ostype",
        ] {
            let request = ChildRequest::Read {
                path: path.to_string(),
                max_bytes: 256,
                offset: 0,
            };
            write_frame(&mut shim_end, &request).expect("shim write");
            let response: ChildResponse = read_frame(&mut shim_end)
                .expect("shim read")
                .expect("response present");
            responses.push(response);
        }
        drop(shim_end);
        responses
    });

    let mut parent_read = parent_end.try_clone().expect("dup parent end");
    let mut parent_write = parent_end;
    let hostname = canned(&b"honeypot.proxima.local"[..]);
    let osrelease = canned(&b"24.6.0"[..]);
    let ostype = canned(&b"ProximaOS"[..]);
    let fallback = deny_writes();
    let routes: &[(
        &str,
        &dyn proxima_primitives::pipe::alloc_tier::SendDynPipe<ChildRequest, ChildResponse>,
    )] = &[
        ("/proc/sys/kernel/hostname", &hostname),
        ("/proc/sys/kernel/osrelease", &osrelease),
        ("/proc/sys/kernel/ostype", &ostype),
    ];
    run_dispatch_loop(&mut parent_read, &mut parent_write, |request| {
        block_on(dispatch_match(request, routes, &fallback))
    })
    .expect("dispatch loop");

    let responses = shim_handle.join().expect("shim joins");
    assert_eq!(responses.len(), 3);
    let expected_bodies = [
        b"honeypot.proxima.local".to_vec(),
        b"24.6.0".to_vec(),
        b"ProximaOS".to_vec(),
    ];
    for (response, expected) in responses.iter().zip(expected_bodies.iter()) {
        match response {
            ChildResponse::Read(ReadResponse { bytes, .. }) => {
                assert_eq!(bytes, expected);
            }
            other => panic!("expected Read response, got {other:?}"),
        }
    }
}

#[test]
fn dispatch_loop_returns_cleanly_on_immediate_eof() {
    use super::ipc::run_dispatch_loop;
    use std::os::unix::net::UnixStream;

    let (parent_end, shim_end) = UnixStream::pair().expect("socketpair");
    drop(shim_end); // child closes immediately

    let mut parent_read = parent_end.try_clone().expect("dup");
    let mut parent_write = parent_end;
    let fallback = deny_writes();
    let routes: &[(
        &str,
        &dyn proxima_primitives::pipe::alloc_tier::SendDynPipe<ChildRequest, ChildResponse>,
    )] = &[];
    let result = run_dispatch_loop(&mut parent_read, &mut parent_write, |request| {
        block_on(dispatch_match(request, routes, &fallback))
    });
    assert!(
        result.is_ok(),
        "dispatch loop should return Ok on clean EOF before any frame, got {result:?}"
    );
}

// ---- C8e-lite: end-to-end architectural proof at the chain level ----
//
// Without the cdylib shim (C8c), we can still exercise the typed
// dispatch design by manually framing requests and feeding them
// through the chain. This proves the architecture composes
// correctly: Bytes → FrameDecoder → match → FrameEncoder → Bytes.
// Adding the cdylib makes the chain reachable by a real child
// process; the dispatch design itself is already complete.

#[test]
fn end_to_end_chain_synthesizes_honeypot_hostname() {
    // Honeypot dispatch chain — what the parent dispatcher does
    // when the shim sends a frame asking for the kernel hostname.
    let hostname_source = canned(&b"honeypot.proxima.local"[..]);
    let osrelease_source = canned(&b"24.6.0"[..]);
    let ostype_source = canned(&b"ProximaOS"[..]);
    let fallback = deny_writes();

    // Simulate the shim's outbound frame: the child wants to read
    // /proc/sys/kernel/hostname (translated by the shim from
    // gethostname(2) or uname(2)).
    let request = ChildRequest::Read {
        path: "/proc/sys/kernel/hostname".to_string(),
        max_bytes: 256,
        offset: 0,
    };
    let wire_bytes = encode_frame(&request);

    // Strip the 4-byte length prefix to get the postcard payload
    // (the inner FrameDecoder operates on the bare payload, not
    // the framed bytes — outer length-prefix handling lives at
    // the IPC reader).
    let payload_len =
        u32::from_be_bytes([wire_bytes[0], wire_bytes[1], wire_bytes[2], wire_bytes[3]]) as usize;
    let payload = bytes::Bytes::copy_from_slice(&wire_bytes[4..4 + payload_len]);

    // Stage 1: FrameDecoder turns bytes into a typed ChildRequest.
    let decoded_request = FrameDecoder.call_sync(payload);
    assert_eq!(
        decoded_request, request,
        "frame decode preserved the request"
    );

    // Stage 2: dispatch_match routes by path to the right ground.
    let routes: &[(
        &str,
        &dyn proxima_primitives::pipe::alloc_tier::SendDynPipe<ChildRequest, ChildResponse>,
    )] = &[
        ("/proc/sys/kernel/hostname", &hostname_source),
        ("/proc/sys/kernel/osrelease", &osrelease_source),
        ("/proc/sys/kernel/ostype", &ostype_source),
    ];
    let response = block_on(dispatch_match(decoded_request, routes, &fallback))
        .expect("dispatch_match should succeed");

    // Stage 3: FrameEncoder turns the typed ChildResponse into
    // wire bytes the shim can decode.
    let response_bytes = FrameEncoder.call_sync(response.clone());

    // Stage 4: simulate the shim decoding the response back.
    let recovered_response: ChildResponse =
        postcard::from_bytes(&response_bytes).expect("decode response");

    // Assert: the architecture round-trips the honeypot value.
    match &recovered_response {
        ChildResponse::Read(ReadResponse { bytes, .. }) => {
            assert_eq!(
                bytes.as_slice(),
                b"honeypot.proxima.local",
                "child should see the honeypot hostname after a full chain dispatch"
            );
        }
        other => panic!("expected Read response with honeypot value, got {other:?}"),
    }

    // Sanity: the response also serializes back identically.
    assert_eq!(
        recovered_response, response,
        "response round-trip through encoder/decoder is identity"
    );
}

#[test]
fn end_to_end_chain_falls_through_to_deny_on_unrouted_path() {
    let fallback = deny_writes();
    let routes: &[(
        &str,
        &dyn proxima_primitives::pipe::alloc_tier::SendDynPipe<ChildRequest, ChildResponse>,
    )] = &[];

    let request = ChildRequest::Read {
        path: "/unknown/path".to_string(),
        max_bytes: 256,
        offset: 0,
    };
    let response = block_on(dispatch_match(request, routes, &fallback))
        .expect("dispatch_match should succeed");
    match response {
        ChildResponse::Error { errno } => assert_eq!(errno, 30, "EROFS from deny"),
        other => panic!("expected Error response, got {other:?}"),
    }
}

// ---- Framing: encode/decode round trips ----

#[test]
fn frame_round_trip_for_each_child_request_variant() {
    let cases = vec![
        ChildRequest::Read {
            path: "/etc/passwd".to_string(),
            max_bytes: 4096,
            offset: 0,
        },
        ChildRequest::Write {
            path: "/dev/null".to_string(),
            bytes: vec![1, 2, 3, 4, 5],
        },
        ChildRequest::Open {
            path: "/proc/self/status".to_string(),
            flags: 0o644,
        },
        ChildRequest::Close {
            path: "/proc/self/status".to_string(),
        },
        ChildRequest::Stat {
            path: "/proc/uptime".to_string(),
        },
    ];

    for original in cases {
        let bytes = encode_frame(&original);
        let decoded: Option<ChildRequest> = decode_frame(&bytes);
        assert_eq!(
            decoded.as_ref(),
            Some(&original),
            "round-trip failed for {original:?}"
        );
    }
}

#[test]
fn frame_round_trip_for_each_child_response_variant() {
    let cases = vec![
        ChildResponse::Read(ReadResponse {
            bytes: b"hello".to_vec(),
            eof: false,
        }),
        ChildResponse::Write(WriteResponse { bytes_written: 42 }),
        ChildResponse::Open { handle: 7 },
        ChildResponse::Close,
        ChildResponse::Stat {
            size: 1024,
            mode: 0o755,
            is_directory: true,
        },
        ChildResponse::Error { errno: 38 },
    ];

    for original in cases {
        let bytes = encode_frame(&original);
        let decoded: Option<ChildResponse> = decode_frame(&bytes);
        assert_eq!(
            decoded.as_ref(),
            Some(&original),
            "round-trip failed for {original:?}"
        );
    }
}

#[test]
fn decode_frame_rejects_truncated_header() {
    let truncated = [0u8, 1, 2]; // < 4 bytes
    let result: Option<ChildRequest> = decode_frame(&truncated);
    assert!(result.is_none());
}

#[test]
fn decode_frame_rejects_truncated_payload() {
    // Header says 100 bytes but we only provide 4 header + 5 payload.
    let mut frame = vec![0u8, 0, 0, 100];
    frame.extend_from_slice(&[1, 2, 3, 4, 5]);
    let result: Option<ChildRequest> = decode_frame(&frame);
    assert!(result.is_none());
}

#[test]
fn decode_frame_rejects_invalid_postcard() {
    // Header says 5 bytes, payload is 5 bytes of garbage.
    let frame = vec![0u8, 0, 0, 5, 0xff, 0xff, 0xff, 0xff, 0xff];
    let result: Option<ChildRequest> = decode_frame(&frame);
    assert!(result.is_none());
}

// ---- FrameDecoder / FrameEncoder Pipe-shape ----

#[test]
fn frame_decoder_dispatches_postcard_bytes_to_typed_request() {
    let original = ChildRequest::Read {
        path: "/proc/sys/kernel/hostname".to_string(),
        max_bytes: 256,
        offset: 0,
    };
    let payload = postcard::to_allocvec(&original).expect("postcard encode");
    let decoded = FrameDecoder.call_sync(bytes::Bytes::from(payload));
    assert_eq!(decoded, original);
}

#[test]
fn frame_encoder_dispatches_typed_response_to_postcard_bytes() {
    let response = ChildResponse::Read(ReadResponse {
        bytes: b"honeypot.proxima.local".to_vec(),
        eof: true,
    });
    let bytes = FrameEncoder.call_sync(response.clone());
    let decoded: ChildResponse = postcard::from_bytes(&bytes).expect("postcard decode");
    assert_eq!(decoded, response);
}

#[test]
fn frame_decoder_and_encoder_carry_all_purity_markers() {
    assert_no_std::<FrameDecoder>();
    assert_is_pure::<FrameDecoder>();
    assert_deterministic::<FrameDecoder>();
    assert_reproducible::<FrameDecoder>();
    assert_idempotent::<FrameDecoder>();
    assert_without_filesystem::<FrameDecoder>();
    assert_without_network::<FrameDecoder>();
    assert_without_spawn::<FrameDecoder>();
    assert_without_time::<FrameDecoder>();
    assert_without_random::<FrameDecoder>();

    assert_no_std::<FrameEncoder>();
    assert_is_pure::<FrameEncoder>();
    assert_deterministic::<FrameEncoder>();
    assert_reproducible::<FrameEncoder>();
    assert_idempotent::<FrameEncoder>();
    assert_without_filesystem::<FrameEncoder>();
    assert_without_network::<FrameEncoder>();
    assert_without_spawn::<FrameEncoder>();
    assert_without_time::<FrameEncoder>();
    assert_without_random::<FrameEncoder>();
}

#[test]
fn series_does_not_carry_random_marker_when_either_side_violates() {
    // AndThen<SeededEntropy, FixedClock>: SeededEntropy is NOT WithoutRandom
    // → series is NOT WithoutRandom. We can only POSITIVELY assert markers
    // that hold; the absence is structurally enforced. We assert the
    // markers that DO hold:
    assert_without_filesystem::<AndThen<SeededEntropy, FixedClock>>();
    assert_without_network::<AndThen<SeededEntropy, FixedClock>>();
    assert_without_spawn::<AndThen<SeededEntropy, FixedClock>>();
    // Cannot assert WithoutRandom or WithoutTime — would compile-fail.
}

// ---- Capability token construction ----

#[test]
fn capabilities_are_zero_sized() {
    assert_eq!(core::mem::size_of::<CapFilesystem>(), 0);
    assert_eq!(core::mem::size_of::<CapNetwork>(), 0);
    assert_eq!(core::mem::size_of::<CapSpawn>(), 0);
}

#[test]
fn capabilities_can_be_constructed_at_trust_boundary() {
    let _fs = CapFilesystem::grant();
    let _net = CapNetwork::grant();
    let _spawn = CapSpawn::grant();
}

// ---- AbsolutePath validation ----

#[test]
fn const_absolute_path_accepts_valid() {
    let path = AbsolutePath::new_const("/etc/passwd");
    assert_eq!(path.as_str(), "/etc/passwd");
}

#[test]
fn const_absolute_path_accepts_dotted_filenames() {
    // `.bashrc` is fine — only `..` as a SEGMENT is forbidden.
    let path = AbsolutePath::new_const("/home/user/.bashrc");
    assert_eq!(path.as_str(), "/home/user/.bashrc");
}

#[test]
fn try_from_str_rejects_relative() {
    assert_eq!(
        AbsolutePath::try_from_str("etc/passwd"),
        Err(AbsolutePathError::NotAbsolute)
    );
}

#[test]
fn try_from_str_rejects_empty() {
    assert_eq!(
        AbsolutePath::try_from_str(""),
        Err(AbsolutePathError::Empty)
    );
}

#[test]
fn try_from_str_rejects_traversal_segment() {
    assert_eq!(
        AbsolutePath::try_from_str("/etc/../etc/passwd"),
        Err(AbsolutePathError::ContainsTraversal)
    );
}

#[test]
fn try_from_str_rejects_trailing_traversal_segment() {
    assert_eq!(
        AbsolutePath::try_from_str("/etc/.."),
        Err(AbsolutePathError::ContainsTraversal)
    );
}

#[test]
fn try_from_str_accepts_valid_path() {
    let path = AbsolutePath::try_from_str("/etc/passwd").expect("valid path");
    assert_eq!(path.as_str(), "/etc/passwd");
}

// ---- Taint tracking ----

#[test]
fn tainted_string_must_be_sanitized_to_become_trusted() {
    let tainted = Tainted::from_untrusted("/etc/passwd".to_string());
    let trusted = sanitize_absolute_path(tainted).expect("valid path");
    assert_eq!(trusted.inner().as_str(), "/etc/passwd");
}

#[test]
fn sanitize_rejects_malicious_tainted_path() {
    let tainted = Tainted::from_untrusted("../../../etc/shadow".to_string());
    let result = sanitize_absolute_path(tainted);
    assert!(matches!(result, Err(AbsolutePathError::NotAbsolute)));
}

#[test]
fn const_path_promotes_to_trusted_without_runtime_check() {
    let const_path = AbsolutePath::new_const("/etc/passwd");
    let trusted = trust_const_path(const_path);
    assert_eq!(trusted.inner().as_str(), "/etc/passwd");
}

// ---- Ground dispatch correctness ----

fn read_request(path: &str, max_bytes: u32, offset: u64) -> ChildRequest {
    ChildRequest::Read {
        path: path.to_string(),
        max_bytes,
        offset,
    }
}

fn write_request(path: &str, bytes: &[u8]) -> ChildRequest {
    ChildRequest::Write {
        path: path.to_string(),
        bytes: bytes.to_vec(),
    }
}

#[test]
fn canned_read_returns_payload_bytes() {
    let source = canned(&b"honeypot.proxima.local"[..]);
    let response = source.call_sync(read_request("/proc/sys/kernel/hostname", 256, 0));
    match response {
        ChildResponse::Read(ReadResponse { bytes, eof }) => {
            assert_eq!(bytes, b"honeypot.proxima.local");
            assert!(eof, "single-chunk read should set eof");
        }
        other => panic!("expected Read response, got {other:?}"),
    }
}

#[test]
fn canned_read_respects_offset_and_max_bytes() {
    let source = canned(&b"honeypot.proxima.local"[..]);
    let response = source.call_sync(read_request("/proc/sys/kernel/hostname", 4, 0));
    match response {
        ChildResponse::Read(ReadResponse { bytes, eof }) => {
            assert_eq!(bytes, b"hone");
            assert!(!eof, "partial read should not set eof");
        }
        other => panic!("expected Read response, got {other:?}"),
    }

    let response = source.call_sync(read_request("/proc/sys/kernel/hostname", 256, 4));
    match response {
        ChildResponse::Read(ReadResponse { bytes, eof }) => {
            assert_eq!(bytes, b"ypot.proxima.local");
            assert!(eof, "remainder-read should set eof");
        }
        other => panic!("expected Read response, got {other:?}"),
    }
}

#[test]
fn canned_read_at_eof_returns_empty_with_eof_set() {
    let source = canned(&b"abc"[..]);
    let response = source.call_sync(read_request("/some/path", 256, 100));
    match response {
        ChildResponse::Read(ReadResponse { bytes, eof }) => {
            assert!(bytes.is_empty());
            assert!(eof);
        }
        other => panic!("expected Read response, got {other:?}"),
    }
}

#[test]
fn canned_write_acknowledges_byte_count() {
    let source = canned(&b"original"[..]);
    let response = source.call_sync(write_request("/some/path", b"new data"));
    match response {
        ChildResponse::Write(WriteResponse { bytes_written }) => {
            assert_eq!(bytes_written, 8);
        }
        other => panic!("expected Write response, got {other:?}"),
    }
}

#[test]
fn empty_read_returns_eof_immediately() {
    let source = empty();
    let response = source.call_sync(read_request("/dev/null", 256, 0));
    match response {
        ChildResponse::Read(ReadResponse { bytes, eof }) => {
            assert!(bytes.is_empty());
            assert!(eof);
        }
        other => panic!("expected Read response, got {other:?}"),
    }
}

#[test]
fn empty_write_acknowledges_but_discards() {
    let source = empty();
    let response = source.call_sync(write_request("/dev/null", b"discarded"));
    match response {
        ChildResponse::Write(WriteResponse { bytes_written }) => {
            assert_eq!(bytes_written, 9);
        }
        other => panic!("expected Write response, got {other:?}"),
    }
}

#[test]
fn deny_returns_configured_errno_for_any_request() {
    let source = deny_writes();
    let response = source.call_sync(read_request("/etc/passwd", 256, 0));
    match response {
        ChildResponse::Error { errno } => assert_eq!(errno, 30),
        other => panic!("expected Error response, got {other:?}"),
    }

    let response = source.call_sync(write_request("/etc/passwd", b"unauthorized"));
    match response {
        ChildResponse::Error { errno } => assert_eq!(errno, 30),
        other => panic!("expected Error response, got {other:?}"),
    }
}

// ---- AndThen operator dispatch ----

/// A trivial transform pipe for testing AndThen type-chaining:
/// extracts the path string from a ChildRequest.
struct ExtractPath;

impl SendPipe for ExtractPath {
    type In = ChildRequest;
    type Out = String;
    type Err = ProximaError;
    fn call(
        &self,
        request: Self::In,
    ) -> impl core::future::Future<Output = Result<Self::Out, ProximaError>> + Send {
        let path = request.path().to_string();
        async move { Ok(path) }
    }
}

// base-tier mirror, delegating straight through — every pipe implements the
// root `Pipe` too, which is what lets `PipeExt::and_then` reach it.
impl Pipe for ExtractPath {
    type In = ChildRequest;
    type Out = String;
    type Err = ProximaError;
    fn call(
        &self,
        request: Self::In,
    ) -> impl core::future::Future<Output = Result<Self::Out, ProximaError>> {
        SendPipe::call(self, request)
    }
}

/// Pairs with ExtractPath to test AndThen with intermediate type.
struct LengthOnly;

impl SendPipe for LengthOnly {
    type In = String;
    type Out = usize;
    type Err = ProximaError;
    fn call(
        &self,
        input: Self::In,
    ) -> impl core::future::Future<Output = Result<Self::Out, ProximaError>> + Send {
        let len = input.len();
        async move { Ok(len) }
    }
}

impl Pipe for LengthOnly {
    type In = String;
    type Out = usize;
    type Err = ProximaError;
    fn call(
        &self,
        input: Self::In,
    ) -> impl core::future::Future<Output = Result<Self::Out, ProximaError>> {
        SendPipe::call(self, input)
    }
}

#[test]
fn series_chains_intermediate_types() {
    let pipeline = ExtractPath.and_then(LengthOnly);
    let length = pipeline.call_sync(read_request("/etc/passwd", 256, 0));
    assert_eq!(length, "/etc/passwd".len());
}

#[test]
fn nested_series_chains_three_stages() {
    let pipeline = ExtractPath.and_then(LengthOnly).and_then(IntoBytes);
    let bytes: Vec<u8> = pipeline.call_sync(read_request("/etc/passwd", 256, 0));
    assert_eq!(bytes, (11usize).to_le_bytes().to_vec());
}

struct IntoBytes;

impl SendPipe for IntoBytes {
    type In = usize;
    type Out = Vec<u8>;
    type Err = ProximaError;
    fn call(
        &self,
        input: Self::In,
    ) -> impl core::future::Future<Output = Result<Self::Out, ProximaError>> + Send {
        let bytes = input.to_le_bytes().to_vec();
        async move { Ok(bytes) }
    }
}

impl Pipe for IntoBytes {
    type In = usize;
    type Out = Vec<u8>;
    type Err = ProximaError;
    fn call(
        &self,
        input: Self::In,
    ) -> impl core::future::Future<Output = Result<Self::Out, ProximaError>> {
        SendPipe::call(self, input)
    }
}

// ---- dispatch_match routing ----

#[test]
fn dispatch_match_routes_by_path_prefix() {
    let hostname_source = canned(&b"honeypot.proxima.local"[..]);
    let osrelease_source = canned(&b"24.6.0"[..]);
    let fallback = deny_writes();

    let routes: &[(
        &str,
        &dyn proxima_primitives::pipe::alloc_tier::SendDynPipe<ChildRequest, ChildResponse>,
    )] = &[
        ("/proc/sys/kernel/hostname", &hostname_source),
        ("/proc/sys/kernel/osrelease", &osrelease_source),
    ];

    let response = block_on(dispatch_match(
        read_request("/proc/sys/kernel/hostname", 256, 0),
        routes,
        &fallback,
    ))
    .expect("dispatch_match should succeed");
    match response {
        ChildResponse::Read(ReadResponse { bytes, .. }) => {
            assert_eq!(bytes, b"honeypot.proxima.local");
        }
        other => panic!("expected Read response, got {other:?}"),
    }
}

#[test]
fn dispatch_match_falls_through_to_fallback() {
    let routes: &[(
        &str,
        &dyn proxima_primitives::pipe::alloc_tier::SendDynPipe<ChildRequest, ChildResponse>,
    )] = &[];
    let fallback = deny_writes();

    let response = block_on(dispatch_match(
        read_request("/unmatched/path", 256, 0),
        routes,
        &fallback,
    ))
    .expect("dispatch_match should succeed");
    match response {
        ChildResponse::Error { errno } => assert_eq!(errno, 30),
        other => panic!("expected Error response, got {other:?}"),
    }
}

#[test]
fn dispatch_match_first_matching_route_wins() {
    let first = canned(&b"first"[..]);
    let second = canned(&b"second"[..]);
    let fallback = deny_writes();

    let routes: &[(
        &str,
        &dyn proxima_primitives::pipe::alloc_tier::SendDynPipe<ChildRequest, ChildResponse>,
    )] = &[
        ("/proc", &first),
        ("/proc/sys/kernel", &second), // would also match but comes after
    ];

    let response = block_on(dispatch_match(
        read_request("/proc/sys/kernel/hostname", 256, 0),
        routes,
        &fallback,
    ))
    .expect("dispatch_match should succeed");
    match response {
        ChildResponse::Read(ReadResponse { bytes, .. }) => {
            assert_eq!(bytes, b"first", "first matching route should win");
        }
        other => panic!("expected Read response, got {other:?}"),
    }
}

// ---- ChildRequest::path() accessor ----

#[test]
fn child_request_path_accessor_covers_all_variants() {
    let read = read_request("/proc/read", 0, 0);
    assert_eq!(read.path(), "/proc/read");

    let write = write_request("/proc/write", b"");
    assert_eq!(write.path(), "/proc/write");

    let open = ChildRequest::Open {
        path: "/proc/open".to_string(),
        flags: 0,
    };
    assert_eq!(open.path(), "/proc/open");

    let close = ChildRequest::Close {
        path: "/proc/close".to_string(),
    };
    assert_eq!(close.path(), "/proc/close");

    let stat = ChildRequest::Stat {
        path: "/proc/stat".to_string(),
    };
    assert_eq!(stat.path(), "/proc/stat");
}
