//! QUIC packet codec — invariant header parse + encode.
//!
//! See [`header`] for the RFC 9000 §17 layout and the [`header::Header`]
//! enum. The payload region (packet number + AEAD-protected bytes) is
//! surfaced as a borrowed slice; downstream layers (C6 AEAD packet
//! protection, C7 header protection, C9 packet number space) interpret it.

pub mod header;
