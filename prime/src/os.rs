//! std + libc layer. OS-specific implementations (reactor, threads, I/O).

pub mod sizing;

#[cfg(feature = "runtime-prime-reactor")]
pub mod reactor;

#[cfg(feature = "runtime-prime-bgpool")]
pub mod background;

#[cfg(feature = "runtime-prime-bgpool-par")]
pub mod par;

#[cfg(feature = "runtime-prime-cohort")]
pub mod cohort;

#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-inbox-alloc",
))]
pub mod core_shard;

// fd-generic wake-driver for callers holding a raw fd outside the reactor's
// own socket types (AF_XDP, signalfd, etc.) — needs the same worker plumbing
// as core_shard since it parks on `with_current_reactor`.
#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-inbox-alloc",
))]
pub mod readiness;

// process-wide SIGINT/SIGTERM wait over the self-pipe trick, parked on
// `readiness` above — unix-only because the signal numbers/`sigaction`
// this builds on are POSIX, not because the reactor itself is (the
// reactor's own epoll/kqueue backends are already unix-only).
#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-inbox-alloc",
    unix,
))]
pub mod signal;

#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
pub mod primitives;
#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-inbox-alloc",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-bgpool"
))]
pub mod runtime;

#[cfg(feature = "prime-tokio-compat")]
pub mod tokio_compat;

// matches net.rs's own #![cfg(...)] exactly — it imports core_shard::CURRENT_REACTOR,
// which needs executor + reactor + inbox-alloc, not reactor alone.
#[cfg(all(
    feature = "runtime-prime-executor",
    feature = "runtime-prime-reactor",
    feature = "runtime-prime-inbox-alloc",
    any(target_os = "macos", target_os = "linux"),
))]
pub mod net;

#[cfg(all(
    target_os = "linux",
    feature = "io-uring",
    feature = "runtime-prime-reactor"
))]
pub mod io_uring;
