//! AEAD packet protection per [RFC 9001 §5.1] + nonce construction per
//! [RFC 9001 §5.3].
//!
//! Three AEAD algorithms are MUST-implement for QUIC v1 (RFC 9001 §5):
//!
//! - **AES-128-GCM** — `key_len = 16`, `nonce_len = 12`, `tag_len = 16`.
//! - **AES-256-GCM** — `key_len = 32`, `nonce_len = 12`, `tag_len = 16`.
//! - **ChaCha20-Poly1305** — `key_len = 32`, `nonce_len = 12`, `tag_len = 16`.
//!
//! # Nonce construction (RFC 9001 §5.3)
//!
//! For each protected packet:
//!
//! 1. Left-pad the 64-bit packet number with zeros to 96 bits (12 bytes).
//! 2. XOR the padded packet number with the AEAD nonce-base `iv` from C5.
//!
//! ```text
//!   packet_number_padded:  [0, 0, 0, 0, pn7, pn6, pn5, pn4, pn3, pn2, pn1, pn0]
//!   nonce = packet_number_padded ⊕ iv
//! ```
//!
//! **Nonce reuse breaks AEAD security.** The caller MUST ensure each
//! `packet_number` is used at most once per (key, direction) pair. The
//! packet-number-space machinery in C9 enforces this.
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`). The AEAD state types from
//! `aes-gcm` and `chacha20poly1305` are stack-only; all operations are
//! in-place over caller-supplied buffers. No `Vec`, no heap.
//!
//! [RFC 9001 §5.1]: https://www.rfc-editor.org/rfc/rfc9001#section-5.1
//! [RFC 9001 §5.3]: https://www.rfc-editor.org/rfc/rfc9001#section-5.3

use aes_gcm::aead::{AeadInPlace, KeyInit};

/// AEAD key length for AES-128-GCM.
pub const AES_128_GCM_KEY_LEN: usize = 16;
/// AEAD key length for AES-256-GCM.
pub const AES_256_GCM_KEY_LEN: usize = 32;
/// AEAD key length for ChaCha20-Poly1305.
pub const CHACHA20_POLY1305_KEY_LEN: usize = 32;
/// AEAD nonce length shared by both AEADs and used to size [`build_nonce`].
pub const NONCE_LEN: usize = 12;
/// AEAD tag length shared by both AEADs.
pub const TAG_LEN: usize = 16;

/// AEAD operation failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AeadError {
    /// Decryption failed — authentication tag did not verify. Per
    /// RFC 9001 §5.1 the packet MUST be discarded silently; surface
    /// to the caller only for telemetry / shutdown decisions.
    DecryptFailed,
    /// Buffer length exceeded AEAD limit (2^36 bytes for AES-GCM; far
    /// beyond any QUIC packet).
    PayloadTooLong,
}

/// Construct the per-packet AEAD nonce per RFC 9001 §5.3.
///
/// **Security**: the caller must ensure `packet_number` is unique per
/// `(iv, direction)`. Reusing a packet number with the same iv breaks
/// the AEAD's confidentiality + integrity guarantees.
#[inline]
#[must_use]
pub fn build_nonce(iv: &[u8; NONCE_LEN], packet_number: u64) -> [u8; NONCE_LEN] {
    // pn occupies the LOW 8 bytes of the 12-byte nonce (big-endian).
    // bytes 0..4 stay zero before XOR with iv.
    let pn_bytes = packet_number.to_be_bytes();
    let mut nonce = [0u8; NONCE_LEN];
    nonce[4..12].copy_from_slice(&pn_bytes);
    for index in 0..NONCE_LEN {
        nonce[index] ^= iv[index];
    }
    nonce
}

/// AES-128-GCM in-place encryption.
///
/// `buffer` holds plaintext on input; ciphertext on output (same length).
/// `tag` receives the 16-byte authentication tag.
///
/// # Errors
///
/// Returns [`AeadError::PayloadTooLong`] if `buffer.len() > 2^36 - 32`
/// (the AES-GCM limit).
pub fn aes_128_gcm_encrypt(
    key: &[u8; AES_128_GCM_KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    aad: &[u8],
    buffer: &mut [u8],
    tag: &mut [u8; TAG_LEN],
) -> Result<(), AeadError> {
    let cipher = aes_gcm::Aes128Gcm::new(key.into());
    let detached_tag = cipher
        .encrypt_in_place_detached(nonce.into(), aad, buffer)
        .map_err(|_| AeadError::PayloadTooLong)?;
    tag.copy_from_slice(detached_tag.as_slice());
    Ok(())
}

/// AES-128-GCM in-place decryption + authentication.
///
/// `buffer` holds ciphertext on input; plaintext on output if the tag
/// verifies. If verification fails, `buffer` contents are
/// **unspecified** — caller MUST discard the packet per RFC 9001 §5.1.
///
/// # Errors
///
/// Returns [`AeadError::DecryptFailed`] on auth-tag mismatch.
pub fn aes_128_gcm_decrypt(
    key: &[u8; AES_128_GCM_KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    aad: &[u8],
    buffer: &mut [u8],
    tag: &[u8; TAG_LEN],
) -> Result<(), AeadError> {
    let cipher = aes_gcm::Aes128Gcm::new(key.into());
    cipher
        .decrypt_in_place_detached(nonce.into(), aad, buffer, tag.into())
        .map_err(|_| AeadError::DecryptFailed)
}

/// AES-256-GCM in-place encryption. See [`aes_128_gcm_encrypt`].
///
/// # Errors
///
/// Returns [`AeadError::PayloadTooLong`] if `buffer.len() > 2^36 - 32`.
pub fn aes_256_gcm_encrypt(
    key: &[u8; AES_256_GCM_KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    aad: &[u8],
    buffer: &mut [u8],
    tag: &mut [u8; TAG_LEN],
) -> Result<(), AeadError> {
    let cipher = aes_gcm::Aes256Gcm::new(key.into());
    let detached_tag = cipher
        .encrypt_in_place_detached(nonce.into(), aad, buffer)
        .map_err(|_| AeadError::PayloadTooLong)?;
    tag.copy_from_slice(detached_tag.as_slice());
    Ok(())
}

/// AES-256-GCM in-place decryption + authentication.
///
/// # Errors
///
/// Returns [`AeadError::DecryptFailed`] on auth-tag mismatch.
pub fn aes_256_gcm_decrypt(
    key: &[u8; AES_256_GCM_KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    aad: &[u8],
    buffer: &mut [u8],
    tag: &[u8; TAG_LEN],
) -> Result<(), AeadError> {
    let cipher = aes_gcm::Aes256Gcm::new(key.into());
    cipher
        .decrypt_in_place_detached(nonce.into(), aad, buffer, tag.into())
        .map_err(|_| AeadError::DecryptFailed)
}

/// ChaCha20-Poly1305 in-place encryption.
///
/// See [`aes_128_gcm_encrypt`] for argument semantics.
///
/// # Errors
///
/// See [`AeadError`].
pub fn chacha20_poly1305_encrypt(
    key: &[u8; CHACHA20_POLY1305_KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    aad: &[u8],
    buffer: &mut [u8],
    tag: &mut [u8; TAG_LEN],
) -> Result<(), AeadError> {
    let cipher = chacha20poly1305::ChaCha20Poly1305::new(key.into());
    let detached_tag = cipher
        .encrypt_in_place_detached(nonce.into(), aad, buffer)
        .map_err(|_| AeadError::PayloadTooLong)?;
    tag.copy_from_slice(detached_tag.as_slice());
    Ok(())
}

/// ChaCha20-Poly1305 in-place decryption + authentication.
///
/// See [`aes_128_gcm_decrypt`] for argument semantics.
///
/// # Errors
///
/// See [`AeadError`].
pub fn chacha20_poly1305_decrypt(
    key: &[u8; CHACHA20_POLY1305_KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    aad: &[u8],
    buffer: &mut [u8],
    tag: &[u8; TAG_LEN],
) -> Result<(), AeadError> {
    let cipher = chacha20poly1305::ChaCha20Poly1305::new(key.into());
    cipher
        .decrypt_in_place_detached(nonce.into(), aad, buffer, tag.into())
        .map_err(|_| AeadError::DecryptFailed)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::quic::crypto::initial_keys;

    // RFC 9001 §A.2 — client Initial keys derived from
    // DCID = 0x8394c8f03e515708 (also covered by C5 tests).
    const RFC_DCID: [u8; 8] = [0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08];

    // RFC 9001 §A.5 — ChaCha20-Poly1305 1-RTT short-header test vector.
    // The packet protection key, iv, and a short test message are given;
    // we use the ChaCha20-Poly1305 packet protection direct values.
    //
    // From RFC 9001 §A.5:
    //   secret = 9ac312a7f877468ebe69422748ad00a1
    //            5443f18203a07d6060f688f30f21632b
    //   key    = c6d98ff3441c3fe1b2182094f69caa2e
    //            d4b716b65488960a7a984979fb23e1c8
    //   iv     = e0459b3474bdd0e44a41c144
    //   hp     = 25a282b9e82f06f21f488917a4fc8f1b
    //   pn     = 654360c8 (decoded 654360564, packet number 654360564)
    //   unprotected payload = 01

    #[test]
    fn build_nonce_xors_pn_with_iv_high_bytes_zero() {
        let iv = [0xff; NONCE_LEN];
        let pn = 0u64;
        // PN = 0 → padded = [0; 12]; nonce = iv XOR 0 = iv
        assert_eq!(build_nonce(&iv, pn), iv);
    }

    #[test]
    fn build_nonce_pn_changes_low_bytes() {
        let iv = [0; NONCE_LEN];
        let pn = 1u64;
        let mut expected = [0u8; NONCE_LEN];
        // pn = 1 lives in the low byte after big-endian padding:
        // [0,0,0,0, 0,0,0,0, 0,0,0,1] xor [0;12] = [0,0,0,0, 0,0,0,0, 0,0,0,1]
        expected[11] = 1;
        assert_eq!(build_nonce(&iv, pn), expected);
    }

    #[test]
    fn build_nonce_rfc_9001_a2_pn_2() {
        // RFC 9001 §A.2 uses packet number 2 for the sample client Initial.
        let pair = initial_keys::derive(&RFC_DCID).unwrap();
        let nonce = build_nonce(&pair.client.iv, 2);
        // expected: client.iv XOR [0,0,0,0, 0,0,0,0, 0,0,0,2]
        let mut expected = pair.client.iv;
        expected[11] ^= 2;
        assert_eq!(nonce, expected);
    }

    #[test]
    fn aes_128_gcm_round_trip() {
        let pair = initial_keys::derive(&RFC_DCID).unwrap();
        let key = pair.client.key;
        let iv = pair.client.iv;
        let nonce = build_nonce(&iv, 0);
        let aad = b"unprotected header bytes";
        let plaintext = b"this is a test payload for AES-128-GCM";

        let mut buffer = *plaintext;
        let buffer_mut: &mut [u8] = &mut buffer;
        let mut tag = [0u8; TAG_LEN];
        aes_128_gcm_encrypt(&key, &nonce, aad, buffer_mut, &mut tag).unwrap();
        assert_ne!(
            &buffer[..],
            &plaintext[..],
            "ciphertext must differ from plaintext"
        );

        aes_128_gcm_decrypt(&key, &nonce, aad, &mut buffer, &tag).unwrap();
        assert_eq!(&buffer[..], &plaintext[..], "decrypt round-trips");
    }

    #[test]
    fn aes_128_gcm_tampered_tag_rejected() {
        let key = [0xab; AES_128_GCM_KEY_LEN];
        let nonce = [0xcd; NONCE_LEN];
        let aad = b"aad";
        let plaintext = b"protected payload";
        let mut buffer = *plaintext;
        let mut tag = [0u8; TAG_LEN];
        aes_128_gcm_encrypt(&key, &nonce, aad, &mut buffer, &mut tag).unwrap();
        tag[0] ^= 0x01;
        assert_eq!(
            aes_128_gcm_decrypt(&key, &nonce, aad, &mut buffer, &tag),
            Err(AeadError::DecryptFailed)
        );
    }

    #[test]
    fn aes_128_gcm_tampered_aad_rejected() {
        let key = [0xab; AES_128_GCM_KEY_LEN];
        let nonce = [0xcd; NONCE_LEN];
        let aad = b"correct aad";
        let plaintext = b"payload";
        let mut buffer = *plaintext;
        let mut tag = [0u8; TAG_LEN];
        aes_128_gcm_encrypt(&key, &nonce, aad, &mut buffer, &mut tag).unwrap();
        assert_eq!(
            aes_128_gcm_decrypt(&key, &nonce, b"tampered aad", &mut buffer, &tag),
            Err(AeadError::DecryptFailed)
        );
    }

    #[test]
    fn aes_128_gcm_tampered_ciphertext_rejected() {
        let key = [0xab; AES_128_GCM_KEY_LEN];
        let nonce = [0xcd; NONCE_LEN];
        let aad = b"aad";
        let plaintext = b"payload bytes";
        let mut buffer = *plaintext;
        let mut tag = [0u8; TAG_LEN];
        aes_128_gcm_encrypt(&key, &nonce, aad, &mut buffer, &mut tag).unwrap();
        buffer[0] ^= 0xff;
        assert_eq!(
            aes_128_gcm_decrypt(&key, &nonce, aad, &mut buffer, &tag),
            Err(AeadError::DecryptFailed)
        );
    }

    #[test]
    fn chacha20_poly1305_round_trip() {
        let key = [0x12; CHACHA20_POLY1305_KEY_LEN];
        let nonce = [0x34; NONCE_LEN];
        let aad = b"chacha aad";
        let plaintext = b"chacha20-poly1305 payload";
        let mut buffer = *plaintext;
        let mut tag = [0u8; TAG_LEN];
        chacha20_poly1305_encrypt(&key, &nonce, aad, &mut buffer, &mut tag).unwrap();
        assert_ne!(&buffer[..], &plaintext[..]);
        chacha20_poly1305_decrypt(&key, &nonce, aad, &mut buffer, &tag).unwrap();
        assert_eq!(&buffer[..], &plaintext[..]);
    }

    #[test]
    fn chacha20_poly1305_tampered_tag_rejected() {
        let key = [0x12; CHACHA20_POLY1305_KEY_LEN];
        let nonce = [0x34; NONCE_LEN];
        let aad = b"aad";
        let plaintext = b"payload";
        let mut buffer = *plaintext;
        let mut tag = [0u8; TAG_LEN];
        chacha20_poly1305_encrypt(&key, &nonce, aad, &mut buffer, &mut tag).unwrap();
        tag[15] ^= 0x80;
        assert_eq!(
            chacha20_poly1305_decrypt(&key, &nonce, aad, &mut buffer, &tag),
            Err(AeadError::DecryptFailed)
        );
    }

    #[test]
    fn aes_256_gcm_round_trip() {
        let key = [0xab; AES_256_GCM_KEY_LEN];
        let nonce = [0xcd; NONCE_LEN];
        let aad = b"aad-256";
        let plaintext = b"payload bytes for aes-256-gcm";
        let mut buffer = *plaintext;
        let mut tag = [0u8; TAG_LEN];
        aes_256_gcm_encrypt(&key, &nonce, aad, &mut buffer, &mut tag).unwrap();
        assert_ne!(&buffer[..], &plaintext[..]);
        aes_256_gcm_decrypt(&key, &nonce, aad, &mut buffer, &tag).unwrap();
        assert_eq!(&buffer[..], &plaintext[..]);
    }

    #[test]
    fn aes_256_gcm_tampered_tag_rejected() {
        let key = [0xab; AES_256_GCM_KEY_LEN];
        let nonce = [0xcd; NONCE_LEN];
        let aad = b"aad";
        let mut buffer = *b"payload";
        let mut tag = [0u8; TAG_LEN];
        aes_256_gcm_encrypt(&key, &nonce, aad, &mut buffer, &mut tag).unwrap();
        tag[0] ^= 0x01;
        assert_eq!(
            aes_256_gcm_decrypt(&key, &nonce, aad, &mut buffer, &tag),
            Err(AeadError::DecryptFailed)
        );
    }

    #[test]
    fn aes_128_gcm_nist_test_vector() {
        // NIST AES-128-GCM test vector (Test Case 4 from gcm-spec).
        // key   = feffe9928665731c6d6a8f9467308308
        // iv    = cafebabefacedbaddecaf888
        // aad   = feedfacedeadbeeffeedfacedeadbeefabaddad2
        // pt    = d9313225f88406e5a55909c5aff5269a86a7a9531534f7da
        //         2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525
        //         b16aedf5aa0de657ba637b39
        // ct    = 42831ec2217774244b7221b784d0d49ce3aa212f2c02a4e0
        //         35c17e2329aca12e21d514b25466931c7d8f6a5aac84aa05
        //         1ba30b396a0aac973d58e091
        // tag   = 5bc94fbc3221a5db94fae95ae7121a47
        let key: [u8; 16] = [
            0xfe, 0xff, 0xe9, 0x92, 0x86, 0x65, 0x73, 0x1c, 0x6d, 0x6a, 0x8f, 0x94, 0x67, 0x30,
            0x83, 0x08,
        ];
        let nonce: [u8; 12] = [
            0xca, 0xfe, 0xba, 0xbe, 0xfa, 0xce, 0xdb, 0xad, 0xde, 0xca, 0xf8, 0x88,
        ];
        let aad: [u8; 20] = [
            0xfe, 0xed, 0xfa, 0xce, 0xde, 0xad, 0xbe, 0xef, 0xfe, 0xed, 0xfa, 0xce, 0xde, 0xad,
            0xbe, 0xef, 0xab, 0xad, 0xda, 0xd2,
        ];
        let plaintext: [u8; 60] = [
            0xd9, 0x31, 0x32, 0x25, 0xf8, 0x84, 0x06, 0xe5, 0xa5, 0x59, 0x09, 0xc5, 0xaf, 0xf5,
            0x26, 0x9a, 0x86, 0xa7, 0xa9, 0x53, 0x15, 0x34, 0xf7, 0xda, 0x2e, 0x4c, 0x30, 0x3d,
            0x8a, 0x31, 0x8a, 0x72, 0x1c, 0x3c, 0x0c, 0x95, 0x95, 0x68, 0x09, 0x53, 0x2f, 0xcf,
            0x0e, 0x24, 0x49, 0xa6, 0xb5, 0x25, 0xb1, 0x6a, 0xed, 0xf5, 0xaa, 0x0d, 0xe6, 0x57,
            0xba, 0x63, 0x7b, 0x39,
        ];
        let expected_ct: [u8; 60] = [
            0x42, 0x83, 0x1e, 0xc2, 0x21, 0x77, 0x74, 0x24, 0x4b, 0x72, 0x21, 0xb7, 0x84, 0xd0,
            0xd4, 0x9c, 0xe3, 0xaa, 0x21, 0x2f, 0x2c, 0x02, 0xa4, 0xe0, 0x35, 0xc1, 0x7e, 0x23,
            0x29, 0xac, 0xa1, 0x2e, 0x21, 0xd5, 0x14, 0xb2, 0x54, 0x66, 0x93, 0x1c, 0x7d, 0x8f,
            0x6a, 0x5a, 0xac, 0x84, 0xaa, 0x05, 0x1b, 0xa3, 0x0b, 0x39, 0x6a, 0x0a, 0xac, 0x97,
            0x3d, 0x58, 0xe0, 0x91,
        ];
        let expected_tag: [u8; 16] = [
            0x5b, 0xc9, 0x4f, 0xbc, 0x32, 0x21, 0xa5, 0xdb, 0x94, 0xfa, 0xe9, 0x5a, 0xe7, 0x12,
            0x1a, 0x47,
        ];

        let mut buffer = plaintext;
        let mut tag = [0u8; TAG_LEN];
        aes_128_gcm_encrypt(&key, &nonce, &aad, &mut buffer, &mut tag).unwrap();
        assert_eq!(buffer, expected_ct, "NIST ciphertext mismatch");
        assert_eq!(tag, expected_tag, "NIST tag mismatch");

        aes_128_gcm_decrypt(&key, &nonce, &aad, &mut buffer, &tag).unwrap();
        assert_eq!(buffer, plaintext, "NIST round-trip");
    }

    #[test]
    fn chacha20_poly1305_rfc_7539_test_vector() {
        // RFC 7539 §2.8.2 — canonical ChaCha20-Poly1305 test vector.
        let key: [u8; 32] = [
            0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d,
            0x8e, 0x8f, 0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0x9b,
            0x9c, 0x9d, 0x9e, 0x9f,
        ];
        let nonce: [u8; 12] = [
            0x07, 0x00, 0x00, 0x00, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
        ];
        let aad: [u8; 12] = [
            0x50, 0x51, 0x52, 0x53, 0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
        ];
        let plaintext: &[u8] = b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.";
        let expected_tag: [u8; 16] = [
            0x1a, 0xe1, 0x0b, 0x59, 0x4f, 0x09, 0xe2, 0x6a, 0x7e, 0x90, 0x2e, 0xcb, 0xd0, 0x60,
            0x06, 0x91,
        ];

        let mut buffer = plaintext.to_vec();
        let mut tag = [0u8; TAG_LEN];
        chacha20_poly1305_encrypt(&key, &nonce, &aad, &mut buffer, &mut tag).unwrap();
        assert_eq!(tag, expected_tag, "RFC 7539 tag mismatch");
        chacha20_poly1305_decrypt(&key, &nonce, &aad, &mut buffer, &tag).unwrap();
        assert_eq!(&buffer[..], plaintext, "RFC 7539 round-trip");
    }

    #[test]
    fn nonce_reuse_with_same_key_produces_same_ciphertext_keystream() {
        // demonstrates the danger of nonce reuse — same (key, nonce) gives
        // same keystream, so plaintext XOR can be recovered. This is NOT
        // an API guarantee; it's a security warning for the caller.
        let key = [0xab; AES_128_GCM_KEY_LEN];
        let nonce = [0xcd; NONCE_LEN];
        let mut a = *b"AAAAAAAAAAAAAAAA";
        let mut b = *b"BBBBBBBBBBBBBBBB";
        let mut tag_a = [0u8; TAG_LEN];
        let mut tag_b = [0u8; TAG_LEN];
        aes_128_gcm_encrypt(&key, &nonce, b"", &mut a, &mut tag_a).unwrap();
        aes_128_gcm_encrypt(&key, &nonce, b"", &mut b, &mut tag_b).unwrap();
        // a XOR b == plaintext-a XOR plaintext-b — the catastrophic leak.
        // This test exists to make the security property visible to future
        // readers and to ensure the AEAD APIs do not silently prevent
        // nonce reuse (it's the caller's responsibility — C9 packet number
        // space enforces).
        let mut leaked = [0u8; 16];
        for index in 0..16 {
            leaked[index] = a[index] ^ b[index];
        }
        assert_eq!(
            leaked,
            *b"\x03\x03\x03\x03\x03\x03\x03\x03\x03\x03\x03\x03\x03\x03\x03\x03"
        );
    }
}
