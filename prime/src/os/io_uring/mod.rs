//! prime-native io_uring TCP backend (Linux only).
//!
//! exposes `TcpListener` and `TcpStream` implementing `futures::io::AsyncRead
//! + AsyncWrite` via the io_uring completion model rather than epoll readiness.
//!
//! this module is cfg-absent on non-Linux platforms and when the `io-uring`
//! or `runtime-prime-reactor` features are disabled — callers fall back to
//! the epoll-based `os::net` module transparently.

#![cfg(all(
    target_os = "linux",
    feature = "io-uring",
    feature = "runtime-prime-reactor"
))]

pub mod reactor;
pub mod tcp_listener;
pub mod tcp_stream;

pub use tcp_listener::TcpListener;
pub use tcp_stream::TcpStream;
