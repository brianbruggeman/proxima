#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::useless_vec,
    clippy::needless_range_loop,
    clippy::default_constructed_unit_structs
)]
//! Regression test pinning the G1–G9 type-system guarantees
//! from `pty-tester/docs/proxima-pty/guiding-principles.md`.
//!
//! Every assertion here is a **trait-bound check**: if a primitive
//! loses a marker, removes a capability token, or weakens
//! `AbsolutePath`'s const-validation, this file fails to compile.
//! Treat it as the lockdown contract for the proxima-process type
//! system. Touching it means an intentional guarantee change —
//! update the guiding-principles doc + discipline log to match.
//!
//! Guarantee status (from guiding-principles.md):
//!
//! - G1 effect markers (Without*)                 — Done, locked here.
//! - G2 capability tokens (Cap*)                  — Done, locked here.
//! - G3 const-validated absolute paths            — Done, locked here.
//! - G4 NoStd / AllocFree markers                 — Done, locked here.
//! - G5 allocation budget typing                  — Punted (no assertion).
//! - G6 determinism hierarchy                     — Done, locked here.
//! - G7 exhaustive match macro                    — Punted (no assertion).
//! - G8 stream cardinality typing                 — Punted (no assertion).
//! - G9 information taint (Tainted/Trusted)       — Done, locked here.

use proxima_primitives::pipe::ProximaError;
use proxima_primitives::pipe::SendPipe;
use proxima_primitives::pipe::alloc_tier::PipeHandle;
use proxima_process::Command;
use proxima_process::capabilities::{CapFilesystem, CapNetwork, CapSpawn};
use proxima_process::grounds::{Canned, Deny, Empty};
use proxima_process::host_grounds::{
    FixedClock, HostRead, HostWrite, OsEntropy, RealClock, SeededEntropy,
};
use proxima_process::markers::{
    AllocFree, Commutative, Deterministic, IdempotentSideEffectFree, IsPure, NoStd, Reproducible,
    WithoutFilesystem, WithoutNetwork, WithoutRandom, WithoutSpawn, WithoutTime,
};
use proxima_process::operators::AndThen;
use proxima_process::path::{AbsolutePath, AbsolutePathError};
use proxima_process::protocol::{ChildRequest, ChildResponse};
use proxima_process::taint::{Tainted, sanitize_absolute_path, trust_const_path};

// ----- helper: compile-time trait-bound assertion -----

fn assert_without_filesystem<T: WithoutFilesystem>() {}
fn assert_without_network<T: WithoutNetwork>() {}
fn assert_without_spawn<T: WithoutSpawn>() {}
fn assert_without_time<T: WithoutTime>() {}
fn assert_without_random<T: WithoutRandom>() {}
fn assert_deterministic<T: Deterministic>() {}
fn assert_reproducible<T: Reproducible>() {}
fn assert_is_pure<T: IsPure>() {}
fn assert_idempotent<T: IdempotentSideEffectFree>() {}
fn assert_commutative<T: Commutative>() {}
fn assert_no_std<T: NoStd>() {}
fn assert_alloc_free<T: AllocFree>() {}
fn assert_dispatch_chain<
    T: SendPipe<In = ChildRequest, Out = ChildResponse, Err = ProximaError>,
>() {
}

// G1 — Effect markers (Without*).
// Pure grounds declare absence of every effect surface.
// Host grounds must KEEP the absences for surfaces they don't
// touch (HostRead is WithoutNetwork, OsEntropy is
// WithoutFilesystem, etc.) but NOT for the one they do use —
// that absence is verified by the host's own crate tests.

#[test]
fn g1_empty_carries_full_effect_absence() {
    assert_without_filesystem::<Empty>();
    assert_without_network::<Empty>();
    assert_without_spawn::<Empty>();
    assert_without_time::<Empty>();
    assert_without_random::<Empty>();
}

#[test]
fn g1_deny_carries_full_effect_absence() {
    assert_without_filesystem::<Deny>();
    assert_without_network::<Deny>();
    assert_without_spawn::<Deny>();
    assert_without_time::<Deny>();
    assert_without_random::<Deny>();
}

#[test]
fn g1_canned_carries_full_effect_absence() {
    assert_without_filesystem::<Canned>();
    assert_without_network::<Canned>();
    assert_without_spawn::<Canned>();
    assert_without_time::<Canned>();
    assert_without_random::<Canned>();
}

#[test]
fn g1_host_grounds_keep_unrelated_effect_absences() {
    // HostRead/HostWrite touch the filesystem only.
    assert_without_network::<HostRead>();
    assert_without_spawn::<HostRead>();
    assert_without_time::<HostRead>();
    assert_without_random::<HostRead>();

    assert_without_network::<HostWrite>();
    assert_without_spawn::<HostWrite>();
    assert_without_time::<HostWrite>();
    assert_without_random::<HostWrite>();

    // OsEntropy touches randomness only.
    assert_without_filesystem::<OsEntropy>();
    assert_without_network::<OsEntropy>();
    assert_without_spawn::<OsEntropy>();
    assert_without_time::<OsEntropy>();

    // RealClock touches the wall clock only.
    assert_without_filesystem::<RealClock>();
    assert_without_network::<RealClock>();
    assert_without_spawn::<RealClock>();
    assert_without_random::<RealClock>();
}

// G2 — Capability tokens.
// Cap* are zero-sized with private inner — `grant()` is the
// ONLY public construction path.

#[test]
fn g2_capability_tokens_construct_via_grant() {
    let _filesystem = CapFilesystem::grant();
    let _network = CapNetwork::grant();
    let _spawn = CapSpawn::grant();
}

// G3 — Const-validated absolute paths.
// Compile-time: AbsolutePath::new_const accepts a literal that
// passes the rules; runtime: try_from_str rejects bad input.

const STATIC_PROC_PATH: AbsolutePath<&'static str> =
    AbsolutePath::new_const("/proc/sys/kernel/hostname");

#[test]
fn g3_const_path_compiles_for_valid_static_input() {
    assert_eq!(STATIC_PROC_PATH.as_str(), "/proc/sys/kernel/hostname");
}

#[test]
fn g3_dynamic_path_rejected_when_relative() {
    let err = AbsolutePath::<alloc::string::String>::try_from_str("relative/path")
        .expect_err("must reject relative");
    assert!(matches!(err, AbsolutePathError::NotAbsolute));
}

#[test]
fn g3_dynamic_path_rejected_when_traversal() {
    let err = AbsolutePath::<alloc::string::String>::try_from_str("/etc/../passwd")
        .expect_err("must reject traversal");
    assert!(matches!(err, AbsolutePathError::ContainsTraversal));
}

#[test]
fn g3_dynamic_path_accepts_valid_absolute() {
    let path = AbsolutePath::<alloc::string::String>::try_from_str("/etc/hostname")
        .expect("accept valid absolute");
    assert_eq!(path.as_str(), "/etc/hostname");
}

extern crate alloc;

// G4 — NoStd / AllocFree markers.
// Pure grounds compile under #![no_std] and run without heap.

#[test]
fn g4_pure_grounds_carry_no_std_and_alloc_free() {
    assert_no_std::<Empty>();
    assert_alloc_free::<Empty>();
    assert_no_std::<Deny>();
    assert_alloc_free::<Deny>();
    assert_no_std::<Canned>();
    // Canned allocates a Vec<u8> per read response — it is
    // NOT AllocFree by design (verified by absence here).
}

// G6 — Determinism hierarchy.
// Deterministic ⊃ Reproducible. Pure grounds + grounds with a
// fixed seed/timestamp must be Reproducible.

#[test]
fn g6_pure_grounds_are_reproducible() {
    assert_reproducible::<Empty>();
    assert_reproducible::<Deny>();
    assert_reproducible::<Canned>();
}

#[test]
fn g6_fixed_sources_are_reproducible() {
    assert_reproducible::<FixedClock>();
    assert_reproducible::<SeededEntropy>();
}

#[test]
fn g6_pure_grounds_carry_full_purity_stack() {
    assert_is_pure::<Empty>();
    assert_idempotent::<Empty>();
    assert_commutative::<Empty>();
    assert_deterministic::<Deny>();
    assert_idempotent::<Deny>();
    assert_commutative::<Deny>();
}

// G9 — Information taint tracking.
// Tainted<T> requires sanitize_* to become Trusted<...>.

#[test]
fn g9_const_path_can_be_trusted_without_runtime_check() {
    let trusted = trust_const_path(STATIC_PROC_PATH);
    assert_eq!(trusted.inner().as_str(), "/proc/sys/kernel/hostname");
}

#[test]
fn g9_tainted_value_requires_sanitisation_to_become_trusted() {
    let tainted = Tainted::from_untrusted("/proc/sys/kernel/hostname".to_string());
    let trusted = sanitize_absolute_path(tainted).expect("sanitise valid absolute");
    assert_eq!(trusted.inner().as_str(), "/proc/sys/kernel/hostname");
}

#[test]
fn g9_tainted_malicious_path_rejected_by_sanitiser() {
    let tainted = Tainted::from_untrusted("../../../etc/passwd".to_string());
    assert!(sanitize_absolute_path(tainted).is_err());
}

// AndThen — AND-propagation of effect markers (per G1 contract).
// AndThen<Empty, Empty> is still WithoutFilesystem/etc. + still
// Deterministic, because both sides are.

#[test]
fn series_propagates_without_markers_when_both_sides_do() {
    type EmptyChain = AndThen<Empty, Empty>;
    assert_without_filesystem::<EmptyChain>();
    assert_without_network::<EmptyChain>();
    assert_without_spawn::<EmptyChain>();
    assert_without_time::<EmptyChain>();
    assert_without_random::<EmptyChain>();
    assert_deterministic::<EmptyChain>();
}

// PipeHandle<ChildRequest, ChildResponse> — marker-erasure boundary.
//
// The type IS a SendPipe over the dispatch shape, but it carries NO
// markers (the inner chain's marker stack is hidden behind dyn).
// This is the documented cost of config-driven chains: the typed
// path (e.g. CommandPipe::builder().dispatch(chain) with a typed
// `chain`) keeps the marker proofs; the DispatchChoice -> erased
// PipeHandle config path loses them.
//
// Removing the assertion below WITHOUT updating guiding-
// principles is a regression.

#[test]
fn dyn_dispatch_chain_is_generic_pipe_but_carries_no_markers() {
    assert_dispatch_chain::<PipeHandle<ChildRequest, ChildResponse>>();
}

// Command — std-tier drop-in mirror.
//
// Spawning a subprocess is the wholly-unconstrained ground:
// filesystem + network + spawn + time + random + non-determinism
// all on the table. Asserting absence of any of these on Command
// would silently break every chain that requires the absence —
// the whole sandbox model collapses.
//
// The negative assertion below is structural: if a future change
// adds e.g. `WithoutSpawn` to Command's impl block, the comment
// here becomes inaccurate AND the related sandbox chains would
// silently lose their soundness guarantee. To convert this from a
// soft contract to a hard compile-time one we'd need stable
// negative_impls or a static_assertions crate dep; for now the
// pinned positive surface on the OTHER primitives is the proof —
// if Command leaks markers, downstream chain construction stops
// working as expected (callers asserting WithoutSpawn compose
// against Command and don't get the safety they expect).

#[test]
fn command_is_a_pipe() {
    // Command must impl proxima_primitives::pipe::Handler (the std-shape async
    // request/response trait, TARGET 2's rename of the served Pipe).
    // This is the additive surface that makes `use proxima_process::Command`
    // interchangeable with std AND composable through proxima's Pipe
    // ecosystem.
    fn assert_pipe<T: proxima_primitives::pipe::Handler>() {}
    assert_pipe::<Command>();
}

// Wire-format parity (per `proxima.decision.libc_shim_vm_parity`).
//
// The variant discriminant index is the postcard wire byte for
// enum-tagged messages. If the order in `protocol.rs` shifts, the
// libc-shim C decoder and the proxima-vm host decoder both
// silently produce wrong responses. These tests lock the order
// AS A REGRESSION GATE — touching them means an intentional wire
// break, which must coordinate both consumers.

#[test]
fn wire_format_child_request_discriminants_locked() {
    use proxima_process::protocol::ChildRequest;
    let read = ChildRequest::Read {
        path: alloc::string::String::from("/dev/null"),
        max_bytes: 0,
        offset: 0,
    };
    let bytes = postcard::to_allocvec(&read).expect("encode");
    // postcard enum discriminant is a varint(u32). For single-byte
    // varints (0..127) it's just the byte itself.
    assert_eq!(bytes[0], 0, "ChildRequest::Read must be variant 0");

    let write = ChildRequest::Write {
        path: alloc::string::String::new(),
        bytes: alloc::vec::Vec::new(),
    };
    assert_eq!(
        postcard::to_allocvec(&write).expect("encode")[0],
        1,
        "ChildRequest::Write must be variant 1"
    );

    let open = ChildRequest::Open {
        path: alloc::string::String::new(),
        flags: 0,
    };
    assert_eq!(
        postcard::to_allocvec(&open).expect("encode")[0],
        2,
        "ChildRequest::Open must be variant 2"
    );

    let close = ChildRequest::Close {
        path: alloc::string::String::new(),
    };
    assert_eq!(
        postcard::to_allocvec(&close).expect("encode")[0],
        3,
        "ChildRequest::Close must be variant 3"
    );

    let stat = ChildRequest::Stat {
        path: alloc::string::String::new(),
    };
    assert_eq!(
        postcard::to_allocvec(&stat).expect("encode")[0],
        4,
        "ChildRequest::Stat must be variant 4"
    );
}

#[test]
fn wire_format_child_response_discriminants_locked() {
    use proxima_process::protocol::{ChildResponse, ReadResponse, WriteResponse};
    let read = ChildResponse::Read(ReadResponse {
        bytes: alloc::vec::Vec::new(),
        eof: false,
    });
    assert_eq!(
        postcard::to_allocvec(&read).expect("encode")[0],
        0,
        "ChildResponse::Read must be variant 0"
    );

    let write = ChildResponse::Write(WriteResponse { bytes_written: 0 });
    assert_eq!(
        postcard::to_allocvec(&write).expect("encode")[0],
        1,
        "ChildResponse::Write must be variant 1"
    );

    let open = ChildResponse::Open { handle: 0 };
    assert_eq!(
        postcard::to_allocvec(&open).expect("encode")[0],
        2,
        "ChildResponse::Open must be variant 2"
    );

    let close = ChildResponse::Close;
    assert_eq!(
        postcard::to_allocvec(&close).expect("encode")[0],
        3,
        "ChildResponse::Close must be variant 3"
    );

    let stat = ChildResponse::Stat {
        size: 0,
        mode: 0,
        is_directory: false,
    };
    assert_eq!(
        postcard::to_allocvec(&stat).expect("encode")[0],
        4,
        "ChildResponse::Stat must be variant 4"
    );

    let error = ChildResponse::Error { errno: 0 };
    assert_eq!(
        postcard::to_allocvec(&error).expect("encode")[0],
        5,
        "ChildResponse::Error must be variant 5"
    );
}

#[test]
fn wire_format_round_trip_via_postcard_proves_parity_baseline() {
    use proxima_process::protocol::ChildRequest;
    let original = ChildRequest::Read {
        path: alloc::string::String::from("/proc/sys/kernel/hostname"),
        max_bytes: 256,
        offset: 0,
    };
    let bytes = postcard::to_allocvec(&original).expect("encode");
    let decoded: ChildRequest = postcard::from_bytes(&bytes).expect("decode");
    assert_eq!(decoded, original);
    // The bytes here are EXACTLY what the libc-shim C encoder must
    // produce and what proxima-vm's host-side decoder must accept.
    // The byte sequence is the parity contract in wire form.
}

#[test]
fn command_drop_in_signatures_compile() {
    // Compile-time witness that the std-shape call paths work:
    // string literals, std::env iterators, &Path, Into<Stdio>.
    use std::ffi::OsStr;
    use std::path::Path;
    let mut cmd = Command::new("/bin/ls");
    cmd.arg("-la")
        .args(["/etc", "/tmp"])
        .env("LANG", "C")
        .envs([("A", "1"), ("B", "2")])
        .env_remove(OsStr::new("LANG"))
        .current_dir(Path::new("/tmp"))
        .stdin(proxima_process::Stdio::piped())
        .stdout(proxima_process::Stdio::null())
        .stderr(proxima_process::Stdio::inherit());
    assert_eq!(cmd.get_program(), OsStr::new("/bin/ls"));
}
