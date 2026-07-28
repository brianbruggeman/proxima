//! Shared AEAD types, so the suite enum in [`crate::esp`] can name nonces and
//! tags without either cipher crate being the one that defines them.
//!
//! Both `chacha20poly1305` and `aes-gcm` build on the same `aead` traits and
//! agree on these shapes — a 12-byte nonce and a 16-byte tag — which is why a
//! single `ChildSa` can dispatch across them with no wrapper type.

/// 96-bit AEAD nonce, shared by both suites.
pub type Nonce = aead::Nonce<aes_gcm_shape::ShapeMarker>;

/// 128-bit authentication tag, shared by both suites.
pub type Tag = aead::Tag<aes_gcm_shape::ShapeMarker>;

/// Both suites are `AeadCore` with `NonceSize = U12` and `TagSize = U16`; this
/// marker names those sizes once rather than picking a winner.
mod aes_gcm_shape {
    use aead::consts::{U12, U16};
    use aead::{AeadCore, KeySizeUser};

    pub struct ShapeMarker;

    impl AeadCore for ShapeMarker {
        type NonceSize = U12;
        type TagSize = U16;
        type CiphertextOverhead = aead::consts::U0;
    }

    impl KeySizeUser for ShapeMarker {
        type KeySize = aead::consts::U32;
    }
}
