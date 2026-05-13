//! QUIC crypto leaves — HKDF-Expand-Label, AEAD packet protection,
//! header protection. All modules are tier-3 capable (`no_std + no_alloc`)
//! and composed from RustCrypto primitives (`sha2`, `hmac`, `hkdf` for
//! key derivation; AEAD backend lands in C6).
//!
//! See `docs/proxima-quic/edges.md` for the design call: tier-3 uses
//! RustCrypto because `aws-lc-rs` requires `std` (libc-backed C
//! bindings). The std-tier facade `proxima-quic` is free to wire in
//! `aws-lc-rs` as the perf backend for bulk AEAD (C6); the tier-3
//! reach stays on RustCrypto.

pub mod aead;
pub mod expand_label;
pub mod header_protection;
pub mod initial_keys;
pub mod packet_protection;
pub mod retry_integrity;
