//! Initial-packet protect/unprotect compose layer per [RFC 9001 §5].
//!
//! Composes C5 (initial keys), C6 (AEAD packet protection), and C7
//! (header protection) into the full RFC 9001 §5 packet-protection
//! sequence:
//!
//! ```text
//!   protect:
//!     1. AEAD-encrypt(key, build_nonce(iv, pn), aad = unprotected_header,
//!                     buffer = plaintext) → ciphertext + tag
//!     2. sample  = ciphertext[pn_offset + 4 - pn_offset .. + 16]
//!                (always 16 bytes starting at pn_offset + 4 in the packet)
//!     3. mask    = aes_128_mask(hp_key, sample)  (or chacha20_mask)
//!     4. packet[0] ^= mask[0] & 0x0f  (long header)
//!     5. packet[pn_offset .. pn_offset + pn_byte_len] ^=
//!        mask[1 .. 1 + pn_byte_len]
//!
//!   unprotect: reverse — header protection first, then AEAD.
//! ```
//!
//! The scope for now is **Initial-packet** protection only (uses the
//! AES-128-GCM keys derived by C5 [`initial_keys::derive`]). The same
//! compose pattern extends trivially to Handshake / 1-RTT packets once
//! the TLS handshake (C11) starts producing those keys.
//!
//! # Tier
//!
//! Tier-3 (bare `no_std + no_alloc`). Operates in-place on the
//! caller's packet buffer. No `Vec`, no `Bytes`.
//!
//! # Security
//!
//! The `/security-audit` equivalent for this module is the
//! RFC 9001 §A.2 sample client Initial round-trip: protect →
//! unprotect → byte-exact match with the original plaintext.
//! Inserting any single bit mutation in the protected packet
//! triggers [`PacketProtectionError::DecryptFailed`] on unprotect.
//!
//! [RFC 9001 §5]: https://www.rfc-editor.org/rfc/rfc9001#section-5

use super::aead::{self, AES_128_GCM_KEY_LEN, AeadError, NONCE_LEN, TAG_LEN};
use super::header_protection::{self, HeaderProtectionError, SAMPLE_LEN};
use super::initial_keys::{InitialKeys, QUIC_HP_LEN, QUIC_IV_LEN, QUIC_KEY_LEN};
use crate::quic::packet_number;

const _: () = {
    assert!(QUIC_KEY_LEN == AES_128_GCM_KEY_LEN);
    assert!(QUIC_IV_LEN == NONCE_LEN);
    assert!(QUIC_HP_LEN == AES_128_GCM_KEY_LEN);
};

/// Failures from the compose layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PacketProtectionError {
    /// Packet buffer too small for the requested operation.
    BufferTooSmall,
    /// Packet-number length out of range (must be 1..=4).
    InvalidPacketNumberLen,
    /// AEAD authentication failed — packet MUST be discarded silently
    /// per RFC 9001 §5.1.
    DecryptFailed,
    /// Header-protection step failed (malformed packet shape).
    HeaderProtection,
    /// Packet-number space encode/decode failure.
    PacketNumber,
}

impl From<AeadError> for PacketProtectionError {
    fn from(err: AeadError) -> Self {
        match err {
            AeadError::DecryptFailed => Self::DecryptFailed,
            AeadError::PayloadTooLong => Self::BufferTooSmall,
        }
    }
}

impl From<HeaderProtectionError> for PacketProtectionError {
    fn from(_: HeaderProtectionError) -> Self {
        Self::HeaderProtection
    }
}

impl From<packet_number::PacketNumberError> for PacketProtectionError {
    fn from(_: packet_number::PacketNumberError) -> Self {
        Self::PacketNumber
    }
}

/// Apply RFC 9001 §5 full packet protection to an Initial packet using
/// AES-128-GCM + AES-128 header protection.
///
/// On entry, `packet` holds the unprotected header followed by the
/// plaintext payload, with at least `TAG_LEN` (16) free bytes after
/// the plaintext for the AEAD tag. On exit, the buffer holds the fully
/// protected packet bytes.
///
/// - `packet[..pn_offset]` is the header bytes preceding the packet number.
/// - `packet[pn_offset .. pn_offset + pn_byte_len]` is the packet number bytes
///   (clear text on entry; XOR'd by the header-protection mask on exit).
/// - `packet[pn_offset + pn_byte_len .. pn_offset + pn_byte_len + plaintext_len]`
///   is the plaintext payload on entry; ciphertext on exit.
/// - `packet[pn_offset + pn_byte_len + plaintext_len .. + TAG_LEN]`
///   receives the AEAD tag.
///
/// # Errors
///
/// See [`PacketProtectionError`].
pub fn protect_initial(
    keys: &InitialKeys,
    full_packet_number: u64,
    pn_byte_len: usize,
    packet: &mut [u8],
    pn_offset: usize,
    plaintext_len: usize,
) -> Result<(), PacketProtectionError> {
    protect_aes128gcm(
        &keys.key,
        &keys.iv,
        &keys.hp,
        full_packet_number,
        pn_byte_len,
        packet,
        pn_offset,
        plaintext_len,
        true,
    )
}

/// Generic AES-128-GCM packet protect — same compose layer as
/// [`protect_initial`] but with the AES key + IV + header-protection
/// key supplied separately, decoupled from the [`InitialKeys`] type.
///
/// Used by Handshake-epoch and 1-RTT-epoch protection where keys
/// come from the TLS key schedule (RFC 9001 §5.2) rather than the
/// Initial-secret derive (RFC 9001 §5.1).
///
/// `is_long_header` selects the [RFC 9001 §5.4.1] masking bit-pattern:
/// `true` masks the low 4 bits of the first byte (long header form);
/// `false` masks the low 5 bits (short header form, including the
/// key-phase bit).
///
/// [RFC 9001 §5.4.1]: https://www.rfc-editor.org/rfc/rfc9001#section-5.4.1
///
/// # Errors
///
/// See [`PacketProtectionError`].
#[allow(clippy::too_many_arguments)]
pub fn protect_aes128gcm(
    aead_key: &[u8; QUIC_KEY_LEN],
    aead_iv: &[u8; QUIC_IV_LEN],
    hp_key: &[u8; QUIC_HP_LEN],
    full_packet_number: u64,
    pn_byte_len: usize,
    packet: &mut [u8],
    pn_offset: usize,
    plaintext_len: usize,
    is_long_header: bool,
) -> Result<(), PacketProtectionError> {
    if !(1..=4).contains(&pn_byte_len) {
        return Err(PacketProtectionError::InvalidPacketNumberLen);
    }
    let plaintext_offset = pn_offset + pn_byte_len;
    let ciphertext_end = plaintext_offset + plaintext_len;
    let total_len = ciphertext_end + TAG_LEN;
    if packet.len() < total_len {
        return Err(PacketProtectionError::BufferTooSmall);
    }
    let sample_offset = pn_offset + 4;
    if packet.len() < sample_offset + SAMPLE_LEN {
        return Err(PacketProtectionError::BufferTooSmall);
    }
    let nonce = aead::build_nonce(aead_iv, full_packet_number);
    let (aad_region, payload_region) = packet.split_at_mut(plaintext_offset);
    let aad = &aad_region[..plaintext_offset];
    let buffer = &mut payload_region[..plaintext_len];
    let mut tag = [0u8; TAG_LEN];
    aead::aes_128_gcm_encrypt(aead_key, &nonce, aad, buffer, &mut tag)?;
    packet[ciphertext_end..ciphertext_end + TAG_LEN].copy_from_slice(&tag);
    let mut sample = [0u8; SAMPLE_LEN];
    sample.copy_from_slice(&packet[sample_offset..sample_offset + SAMPLE_LEN]);
    let mask = header_protection::aes_128_mask(hp_key, &sample);
    let (header_first, rest) = packet
        .split_first_mut()
        .ok_or(PacketProtectionError::BufferTooSmall)?;
    let pn_bytes_start = pn_offset - 1;
    let pn_bytes = &mut rest[pn_bytes_start..pn_bytes_start + pn_byte_len];
    header_protection::apply_mask(header_first, pn_bytes, &mask, is_long_header)?;
    Ok(())
}

/// Generic AES-256-GCM packet protect — same shape as
/// [`protect_aes128gcm`] but with the 32-byte AEAD key + 32-byte
/// header-protection key. Used by 1-RTT epoch when the TLS handshake
/// negotiates `TLS_AES_256_GCM_SHA384`.
///
/// # Errors
///
/// See [`PacketProtectionError`].
#[allow(clippy::too_many_arguments)]
pub fn protect_aes256gcm(
    aead_key: &[u8; aead::AES_256_GCM_KEY_LEN],
    aead_iv: &[u8; QUIC_IV_LEN],
    hp_key: &[u8; aead::AES_256_GCM_KEY_LEN],
    full_packet_number: u64,
    pn_byte_len: usize,
    packet: &mut [u8],
    pn_offset: usize,
    plaintext_len: usize,
    is_long_header: bool,
) -> Result<(), PacketProtectionError> {
    if !(1..=4).contains(&pn_byte_len) {
        return Err(PacketProtectionError::InvalidPacketNumberLen);
    }
    let plaintext_offset = pn_offset + pn_byte_len;
    let ciphertext_end = plaintext_offset + plaintext_len;
    let total_len = ciphertext_end + TAG_LEN;
    if packet.len() < total_len {
        return Err(PacketProtectionError::BufferTooSmall);
    }
    let sample_offset = pn_offset + 4;
    if packet.len() < sample_offset + SAMPLE_LEN {
        return Err(PacketProtectionError::BufferTooSmall);
    }
    let nonce = aead::build_nonce(aead_iv, full_packet_number);
    let (aad_region, payload_region) = packet.split_at_mut(plaintext_offset);
    let aad = &aad_region[..plaintext_offset];
    let buffer = &mut payload_region[..plaintext_len];
    let mut tag = [0u8; TAG_LEN];
    aead::aes_256_gcm_encrypt(aead_key, &nonce, aad, buffer, &mut tag)?;
    packet[ciphertext_end..ciphertext_end + TAG_LEN].copy_from_slice(&tag);
    let mut sample = [0u8; SAMPLE_LEN];
    sample.copy_from_slice(&packet[sample_offset..sample_offset + SAMPLE_LEN]);
    let mask = header_protection::aes_256_mask(hp_key, &sample);
    let (header_first, rest) = packet
        .split_first_mut()
        .ok_or(PacketProtectionError::BufferTooSmall)?;
    let pn_bytes_start = pn_offset - 1;
    let pn_bytes = &mut rest[pn_bytes_start..pn_bytes_start + pn_byte_len];
    header_protection::apply_mask(header_first, pn_bytes, &mask, is_long_header)?;
    Ok(())
}

/// Generic AES-256-GCM packet unprotect — inverse of
/// [`protect_aes256gcm`].
///
/// # Errors
///
/// See [`PacketProtectionError`].
pub fn unprotect_aes256gcm(
    aead_key: &[u8; aead::AES_256_GCM_KEY_LEN],
    aead_iv: &[u8; QUIC_IV_LEN],
    hp_key: &[u8; aead::AES_256_GCM_KEY_LEN],
    largest_received_pn: u64,
    packet: &mut [u8],
    pn_offset: usize,
    is_long_header: bool,
) -> Result<(u64, usize), PacketProtectionError> {
    let sample_offset = pn_offset + 4;
    if packet.len() < sample_offset + SAMPLE_LEN {
        return Err(PacketProtectionError::BufferTooSmall);
    }
    let mut sample = [0u8; SAMPLE_LEN];
    sample.copy_from_slice(&packet[sample_offset..sample_offset + SAMPLE_LEN]);
    let mask = header_protection::aes_256_mask(hp_key, &sample);
    let (header_first, rest) = packet
        .split_first_mut()
        .ok_or(PacketProtectionError::BufferTooSmall)?;
    let first_byte_mask = if is_long_header { 0x0f } else { 0x1f };
    *header_first ^= mask[0] & first_byte_mask;
    let pn_byte_len = usize::from(*header_first & 0x03) + 1;
    if !(1..=4).contains(&pn_byte_len) {
        return Err(PacketProtectionError::InvalidPacketNumberLen);
    }
    let pn_bytes_start = pn_offset - 1;
    let pn_bytes = &mut rest[pn_bytes_start..pn_bytes_start + pn_byte_len];
    for (index, byte) in pn_bytes.iter_mut().enumerate() {
        *byte ^= mask[1 + index];
    }
    let mut truncated = 0u64;
    for &byte in pn_bytes.iter() {
        truncated = (truncated << 8) | u64::from(byte);
    }
    let pn_nbits = (pn_byte_len * 8) as u32;
    let full_pn = packet_number::decode_packet_number(largest_received_pn, truncated, pn_nbits)?;
    let plaintext_offset = pn_offset + pn_byte_len;
    let nonce = aead::build_nonce(aead_iv, full_pn);
    let total_len = packet.len();
    if total_len < plaintext_offset + TAG_LEN {
        return Err(PacketProtectionError::BufferTooSmall);
    }
    let ciphertext_end = total_len - TAG_LEN;
    let plaintext_len = ciphertext_end - plaintext_offset;
    let mut tag = [0u8; TAG_LEN];
    tag.copy_from_slice(&packet[ciphertext_end..ciphertext_end + TAG_LEN]);
    let (aad_region, payload_region) = packet.split_at_mut(plaintext_offset);
    let buffer = &mut payload_region[..plaintext_len];
    aead::aes_256_gcm_decrypt(aead_key, &nonce, aad_region, buffer, &tag)?;
    Ok((full_pn, plaintext_len))
}

/// Generic ChaCha20-Poly1305 packet protect — same shape as
/// [`protect_aes128gcm`] but with the ChaCha20-Poly1305 AEAD + ChaCha20
/// header protection. RFC 9001 §5.4.4 — sample is split into a 4-byte
/// little-endian block counter + 12-byte nonce for HP mask gen.
///
/// # Errors
///
/// See [`PacketProtectionError`].
#[allow(clippy::too_many_arguments)]
pub fn protect_chacha20poly1305(
    aead_key: &[u8; aead::CHACHA20_POLY1305_KEY_LEN],
    aead_iv: &[u8; QUIC_IV_LEN],
    hp_key: &[u8; aead::CHACHA20_POLY1305_KEY_LEN],
    full_packet_number: u64,
    pn_byte_len: usize,
    packet: &mut [u8],
    pn_offset: usize,
    plaintext_len: usize,
    is_long_header: bool,
) -> Result<(), PacketProtectionError> {
    if !(1..=4).contains(&pn_byte_len) {
        return Err(PacketProtectionError::InvalidPacketNumberLen);
    }
    let plaintext_offset = pn_offset + pn_byte_len;
    let ciphertext_end = plaintext_offset + plaintext_len;
    let total_len = ciphertext_end + TAG_LEN;
    if packet.len() < total_len {
        return Err(PacketProtectionError::BufferTooSmall);
    }
    let sample_offset = pn_offset + 4;
    if packet.len() < sample_offset + SAMPLE_LEN {
        return Err(PacketProtectionError::BufferTooSmall);
    }
    let nonce = aead::build_nonce(aead_iv, full_packet_number);
    let (aad_region, payload_region) = packet.split_at_mut(plaintext_offset);
    let aad = &aad_region[..plaintext_offset];
    let buffer = &mut payload_region[..plaintext_len];
    let mut tag = [0u8; TAG_LEN];
    aead::chacha20_poly1305_encrypt(aead_key, &nonce, aad, buffer, &mut tag)?;
    packet[ciphertext_end..ciphertext_end + TAG_LEN].copy_from_slice(&tag);
    let mut sample = [0u8; SAMPLE_LEN];
    sample.copy_from_slice(&packet[sample_offset..sample_offset + SAMPLE_LEN]);
    let mask = header_protection::chacha20_mask(hp_key, &sample);
    let (header_first, rest) = packet
        .split_first_mut()
        .ok_or(PacketProtectionError::BufferTooSmall)?;
    let pn_bytes_start = pn_offset - 1;
    let pn_bytes = &mut rest[pn_bytes_start..pn_bytes_start + pn_byte_len];
    header_protection::apply_mask(header_first, pn_bytes, &mask, is_long_header)?;
    Ok(())
}

/// Generic ChaCha20-Poly1305 packet unprotect — inverse of
/// [`protect_chacha20poly1305`].
///
/// # Errors
///
/// See [`PacketProtectionError`].
pub fn unprotect_chacha20poly1305(
    aead_key: &[u8; aead::CHACHA20_POLY1305_KEY_LEN],
    aead_iv: &[u8; QUIC_IV_LEN],
    hp_key: &[u8; aead::CHACHA20_POLY1305_KEY_LEN],
    largest_received_pn: u64,
    packet: &mut [u8],
    pn_offset: usize,
    is_long_header: bool,
) -> Result<(u64, usize), PacketProtectionError> {
    let HeaderProtectionResult {
        full_pn,
        plaintext_offset,
    } = remove_header_protection_chacha20(
        hp_key,
        largest_received_pn,
        packet,
        pn_offset,
        is_long_header,
    )?;
    let plaintext_len =
        decrypt_chacha20poly1305_in_place(aead_key, aead_iv, full_pn, packet, plaintext_offset)?;
    Ok((full_pn, plaintext_len))
}

/// Remove RFC 9001 §5.4 header protection from `packet` in place
/// using the ChaCha20 HP key.
///
/// # Errors
///
/// See [`PacketProtectionError`].
pub fn remove_header_protection_chacha20(
    hp_key: &[u8; aead::CHACHA20_POLY1305_KEY_LEN],
    largest_received_pn: u64,
    packet: &mut [u8],
    pn_offset: usize,
    is_long_header: bool,
) -> Result<HeaderProtectionResult, PacketProtectionError> {
    let sample_offset = pn_offset + 4;
    if packet.len() < sample_offset + SAMPLE_LEN {
        return Err(PacketProtectionError::BufferTooSmall);
    }
    if packet.len() < TAG_LEN {
        return Err(PacketProtectionError::BufferTooSmall);
    }
    let mut sample = [0u8; SAMPLE_LEN];
    sample.copy_from_slice(&packet[sample_offset..sample_offset + SAMPLE_LEN]);
    let mask = header_protection::chacha20_mask(hp_key, &sample);
    let (header_first, rest) = packet
        .split_first_mut()
        .ok_or(PacketProtectionError::BufferTooSmall)?;
    let first_byte_mask = if is_long_header { 0x0f } else { 0x1f };
    *header_first ^= mask[0] & first_byte_mask;
    let pn_byte_len = usize::from(*header_first & 0x03) + 1;
    if !(1..=4).contains(&pn_byte_len) {
        return Err(PacketProtectionError::InvalidPacketNumberLen);
    }
    let pn_bytes_start = pn_offset - 1;
    let pn_bytes = &mut rest[pn_bytes_start..pn_bytes_start + pn_byte_len];
    for (index, byte) in pn_bytes.iter_mut().enumerate() {
        *byte ^= mask[1 + index];
    }
    let mut truncated = 0u64;
    for &byte in pn_bytes.iter() {
        truncated = (truncated << 8) | u64::from(byte);
    }
    let pn_nbits = (pn_byte_len * 8) as u32;
    let full_pn = packet_number::decode_packet_number(largest_received_pn, truncated, pn_nbits)?;
    let plaintext_offset = pn_offset + pn_byte_len;
    Ok(HeaderProtectionResult {
        full_pn,
        plaintext_offset,
    })
}

/// AEAD-decrypt + verify the tag in place using ChaCha20-Poly1305.
///
/// # Errors
///
/// See [`PacketProtectionError`].
pub fn decrypt_chacha20poly1305_in_place(
    aead_key: &[u8; aead::CHACHA20_POLY1305_KEY_LEN],
    aead_iv: &[u8; QUIC_IV_LEN],
    full_pn: u64,
    packet: &mut [u8],
    plaintext_offset: usize,
) -> Result<usize, PacketProtectionError> {
    let nonce = aead::build_nonce(aead_iv, full_pn);
    let total_len = packet.len();
    if total_len < plaintext_offset + TAG_LEN {
        return Err(PacketProtectionError::BufferTooSmall);
    }
    let ciphertext_end = total_len - TAG_LEN;
    let plaintext_len = ciphertext_end - plaintext_offset;
    let mut tag = [0u8; TAG_LEN];
    tag.copy_from_slice(&packet[ciphertext_end..ciphertext_end + TAG_LEN]);
    let (aad_region, payload_region) = packet.split_at_mut(plaintext_offset);
    let buffer = &mut payload_region[..plaintext_len];
    aead::chacha20_poly1305_decrypt(aead_key, &nonce, aad_region, buffer, &tag)?;
    Ok(plaintext_len)
}

/// Inverse of [`protect_initial`]. Given a protected Initial packet,
/// remove header protection, AEAD-decrypt the payload, and verify the tag.
///
/// `largest_received_pn` is the largest packet number observed in the
/// Initial space so far (from [`crate::quic::packet_number::RecvSpace`]). It
/// is used to expand the truncated wire packet number to the full PN.
///
/// On success returns `(full_packet_number, plaintext_len)`. The packet
/// buffer holds the unprotected header + decrypted plaintext at offsets
/// `[..plaintext_offset]` and `[plaintext_offset..plaintext_offset + plaintext_len]`.
/// The AEAD tag at the tail is left as-is (caller can ignore).
///
/// # Errors
///
/// See [`PacketProtectionError`].
pub fn unprotect_initial(
    keys: &InitialKeys,
    largest_received_pn: u64,
    packet: &mut [u8],
    pn_offset: usize,
) -> Result<(u64, usize), PacketProtectionError> {
    unprotect_aes128gcm(
        &keys.key,
        &keys.iv,
        &keys.hp,
        largest_received_pn,
        packet,
        pn_offset,
        true,
    )
}

/// Generic AES-128-GCM packet unprotect — inverse of
/// [`protect_aes128gcm`]. Used by Handshake-epoch and 1-RTT-epoch
/// unprotect paths.
///
/// `is_long_header` selects the [RFC 9001 §5.4.1] unmasking
/// bit-pattern; for short-header (1-RTT) the caller MUST pass
/// `false` so the key-phase bit + reserved bits are unmasked
/// correctly.
///
/// [RFC 9001 §5.4.1]: https://www.rfc-editor.org/rfc/rfc9001#section-5.4.1
///
/// # Errors
///
/// See [`PacketProtectionError`].
#[allow(clippy::too_many_arguments)]
pub fn unprotect_aes128gcm(
    aead_key: &[u8; QUIC_KEY_LEN],
    aead_iv: &[u8; QUIC_IV_LEN],
    hp_key: &[u8; QUIC_HP_LEN],
    largest_received_pn: u64,
    packet: &mut [u8],
    pn_offset: usize,
    is_long_header: bool,
) -> Result<(u64, usize), PacketProtectionError> {
    let HeaderProtectionResult {
        full_pn,
        plaintext_offset,
    } = remove_header_protection_aes128(
        hp_key,
        largest_received_pn,
        packet,
        pn_offset,
        is_long_header,
    )?;
    let plaintext_len =
        decrypt_aes128gcm_in_place(aead_key, aead_iv, full_pn, packet, plaintext_offset)?;
    Ok((full_pn, plaintext_len))
}

/// Output of [`remove_header_protection_aes128`].
///
/// `full_pn` is the reconstructed full packet number per RFC 9001
/// §5.4.2 / RFC 9000 §17.1. `plaintext_offset` is the byte index in
/// `packet` where the ciphertext (and eventual plaintext) starts —
/// it equals `pn_offset + pn_byte_len`, computed from the now-
/// unprotected first byte.
///
/// After this call, the packet's first byte and PN bytes are in
/// their unprotected form — callers can peek the RFC 9001 §5.4.1
/// key-phase bit (`packet[0] & 0x04`, short-header only) BEFORE
/// committing to an AEAD key choice. This is exactly the C23.3
/// peer-key-update detection path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeaderProtectionResult {
    pub full_pn: u64,
    pub plaintext_offset: usize,
}

/// Remove RFC 9001 §5.4 header protection from `packet` in place
/// using the AES-128 HP key. Decodes the truncated packet number
/// in the header and expands it to a full PN against
/// `largest_received_pn` per RFC 9000 §17.1.
///
/// On success, `packet`'s first byte + PN bytes are in their
/// unprotected form. Callers wanting `unprotect_aes128gcm` semantics
/// chain this with [`decrypt_aes128gcm_in_place`].
///
/// # Errors
///
/// See [`PacketProtectionError`].
pub fn remove_header_protection_aes128(
    hp_key: &[u8; QUIC_HP_LEN],
    largest_received_pn: u64,
    packet: &mut [u8],
    pn_offset: usize,
    is_long_header: bool,
) -> Result<HeaderProtectionResult, PacketProtectionError> {
    let sample_offset = pn_offset + 4;
    if packet.len() < sample_offset + SAMPLE_LEN {
        return Err(PacketProtectionError::BufferTooSmall);
    }
    if packet.len() < TAG_LEN {
        return Err(PacketProtectionError::BufferTooSmall);
    }

    let mut sample = [0u8; SAMPLE_LEN];
    sample.copy_from_slice(&packet[sample_offset..sample_offset + SAMPLE_LEN]);
    let mask = header_protection::aes_128_mask(hp_key, &sample);

    let (header_first, rest) = packet
        .split_first_mut()
        .ok_or(PacketProtectionError::BufferTooSmall)?;
    // RFC 9001 §5.4.1 first-byte mask: low 4 bits (long header) vs low 5 bits (short header).
    let first_byte_mask = if is_long_header { 0x0f } else { 0x1f };
    *header_first ^= mask[0] & first_byte_mask;
    let pn_byte_len = usize::from(*header_first & 0x03) + 1;
    if !(1..=4).contains(&pn_byte_len) {
        return Err(PacketProtectionError::InvalidPacketNumberLen);
    }
    let pn_bytes_start = pn_offset - 1;
    let pn_bytes = &mut rest[pn_bytes_start..pn_bytes_start + pn_byte_len];
    for (index, byte) in pn_bytes.iter_mut().enumerate() {
        *byte ^= mask[1 + index];
    }

    let mut truncated = 0u64;
    for &byte in pn_bytes.iter() {
        truncated = (truncated << 8) | u64::from(byte);
    }
    let pn_nbits = (pn_byte_len * 8) as u32;
    let full_pn = packet_number::decode_packet_number(largest_received_pn, truncated, pn_nbits)?;

    let plaintext_offset = pn_offset + pn_byte_len;
    Ok(HeaderProtectionResult {
        full_pn,
        plaintext_offset,
    })
}

/// Remove RFC 9001 §5.4 header protection from `packet` in place
/// using the AES-256 HP key. Same shape as
/// [`remove_header_protection_aes128`].
///
/// # Errors
///
/// See [`PacketProtectionError`].
pub fn remove_header_protection_aes256(
    hp_key: &[u8; aead::AES_256_GCM_KEY_LEN],
    largest_received_pn: u64,
    packet: &mut [u8],
    pn_offset: usize,
    is_long_header: bool,
) -> Result<HeaderProtectionResult, PacketProtectionError> {
    let sample_offset = pn_offset + 4;
    if packet.len() < sample_offset + SAMPLE_LEN {
        return Err(PacketProtectionError::BufferTooSmall);
    }
    if packet.len() < TAG_LEN {
        return Err(PacketProtectionError::BufferTooSmall);
    }
    let mut sample = [0u8; SAMPLE_LEN];
    sample.copy_from_slice(&packet[sample_offset..sample_offset + SAMPLE_LEN]);
    let mask = header_protection::aes_256_mask(hp_key, &sample);
    let (header_first, rest) = packet
        .split_first_mut()
        .ok_or(PacketProtectionError::BufferTooSmall)?;
    let first_byte_mask = if is_long_header { 0x0f } else { 0x1f };
    *header_first ^= mask[0] & first_byte_mask;
    let pn_byte_len = usize::from(*header_first & 0x03) + 1;
    if !(1..=4).contains(&pn_byte_len) {
        return Err(PacketProtectionError::InvalidPacketNumberLen);
    }
    let pn_bytes_start = pn_offset - 1;
    let pn_bytes = &mut rest[pn_bytes_start..pn_bytes_start + pn_byte_len];
    for (index, byte) in pn_bytes.iter_mut().enumerate() {
        *byte ^= mask[1 + index];
    }
    let mut truncated = 0u64;
    for &byte in pn_bytes.iter() {
        truncated = (truncated << 8) | u64::from(byte);
    }
    let pn_nbits = (pn_byte_len * 8) as u32;
    let full_pn = packet_number::decode_packet_number(largest_received_pn, truncated, pn_nbits)?;
    let plaintext_offset = pn_offset + pn_byte_len;
    Ok(HeaderProtectionResult {
        full_pn,
        plaintext_offset,
    })
}

/// AEAD-decrypt + verify the tag in place using AES-256-GCM.
///
/// # Errors
///
/// See [`PacketProtectionError`].
pub fn decrypt_aes256gcm_in_place(
    aead_key: &[u8; aead::AES_256_GCM_KEY_LEN],
    aead_iv: &[u8; QUIC_IV_LEN],
    full_pn: u64,
    packet: &mut [u8],
    plaintext_offset: usize,
) -> Result<usize, PacketProtectionError> {
    let nonce = aead::build_nonce(aead_iv, full_pn);
    let total_len = packet.len();
    if total_len < plaintext_offset + TAG_LEN {
        return Err(PacketProtectionError::BufferTooSmall);
    }
    let ciphertext_end = total_len - TAG_LEN;
    let plaintext_len = ciphertext_end - plaintext_offset;
    let mut tag = [0u8; TAG_LEN];
    tag.copy_from_slice(&packet[ciphertext_end..ciphertext_end + TAG_LEN]);
    let (aad_region, payload_region) = packet.split_at_mut(plaintext_offset);
    let buffer = &mut payload_region[..plaintext_len];
    aead::aes_256_gcm_decrypt(aead_key, &nonce, aad_region, buffer, &tag)?;
    Ok(plaintext_len)
}

/// AEAD-decrypt + verify the tag in place per RFC 9001 §5.1 using
/// AES-128-GCM. Assumes header protection has already been removed
/// (the first byte + PN bytes are in cleartext form so the AEAD AAD
/// region is correct).
///
/// On success returns the plaintext length (always
/// `packet.len() - plaintext_offset - TAG_LEN`).
///
/// # Errors
///
/// See [`PacketProtectionError`].
pub fn decrypt_aes128gcm_in_place(
    aead_key: &[u8; QUIC_KEY_LEN],
    aead_iv: &[u8; QUIC_IV_LEN],
    full_pn: u64,
    packet: &mut [u8],
    plaintext_offset: usize,
) -> Result<usize, PacketProtectionError> {
    let nonce = aead::build_nonce(aead_iv, full_pn);
    let total_len = packet.len();
    if total_len < plaintext_offset + TAG_LEN {
        return Err(PacketProtectionError::BufferTooSmall);
    }
    let ciphertext_end = total_len - TAG_LEN;
    let plaintext_len = ciphertext_end - plaintext_offset;

    let mut tag = [0u8; TAG_LEN];
    tag.copy_from_slice(&packet[ciphertext_end..ciphertext_end + TAG_LEN]);
    let (aad_region, payload_region) = packet.split_at_mut(plaintext_offset);
    let buffer = &mut payload_region[..plaintext_len];
    aead::aes_128_gcm_decrypt(aead_key, &nonce, aad_region, buffer, &tag)?;
    Ok(plaintext_len)
}

#[cfg(all(test, feature = "quic-alloc"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::quic::crypto::initial_keys;
    use alloc::vec;

    const RFC_DCID: [u8; 8] = [0x83, 0x94, 0xc8, 0xf0, 0x3e, 0x51, 0x57, 0x08];

    /// Build a minimal unprotected Initial packet header + plaintext.
    /// Layout (matches RFC 9000 §17.2.2):
    ///   c3 | 00000001 | 08 | dcid (8) | 00 | 00 (token_len) | LENGTH (varint)
    ///      | PN (4 bytes) | plaintext (N bytes) | tag space (16 bytes)
    fn build_unprotected_initial(
        plaintext: &[u8],
        packet_number: u64,
    ) -> (alloc::vec::Vec<u8>, usize) {
        let pn_byte_len = 4usize;
        let header_pre_length = 1 + 4 + 1 + 8 + 1 + 1; // type | version | dcid_len | dcid | scid_len | token_len
        let payload_for_length = pn_byte_len + plaintext.len() + TAG_LEN;
        // length varint: payload_for_length must fit in <= 2 bytes for typical packets;
        // use a 2-byte form for any value <= 16383.
        let length_byte_len = if payload_for_length < 64 {
            1
        } else if payload_for_length < 16_384 {
            2
        } else {
            4
        };
        let header_len = header_pre_length + length_byte_len + pn_byte_len;
        let total_len = header_len + plaintext.len() + TAG_LEN;
        let mut packet = alloc::vec![0u8; total_len];
        let mut cursor = 0;
        // first byte: long(1) | fixed(1) | Initial(00) | reserved(00) | pn_len=11 (4 bytes)
        packet[cursor] = 0b1100_0011;
        cursor += 1;
        packet[cursor..cursor + 4].copy_from_slice(&1u32.to_be_bytes()); // version 1
        cursor += 4;
        packet[cursor] = 8; // dcid_len
        cursor += 1;
        packet[cursor..cursor + 8].copy_from_slice(&RFC_DCID);
        cursor += 8;
        packet[cursor] = 0; // scid_len
        cursor += 1;
        packet[cursor] = 0; // token_len (varint, 1-byte form for value 0)
        cursor += 1;
        // length varint
        match length_byte_len {
            1 => {
                packet[cursor] = payload_for_length as u8;
                cursor += 1;
            }
            2 => {
                let value = payload_for_length as u16 | 0x4000;
                packet[cursor..cursor + 2].copy_from_slice(&value.to_be_bytes());
                cursor += 2;
            }
            _ => {
                let value = payload_for_length as u32 | 0x8000_0000;
                packet[cursor..cursor + 4].copy_from_slice(&value.to_be_bytes());
                cursor += 4;
            }
        }
        // PN bytes
        let pn_offset = cursor;
        let pn_bytes = (packet_number as u32).to_be_bytes();
        packet[cursor..cursor + 4].copy_from_slice(&pn_bytes);
        cursor += 4;
        // plaintext
        packet[cursor..cursor + plaintext.len()].copy_from_slice(plaintext);
        (packet, pn_offset)
    }

    #[test]
    fn round_trip_minimal_payload() {
        let pair = initial_keys::derive(&RFC_DCID).unwrap();
        let plaintext = b"hello world, this is a test plaintext payload for QUIC";
        let pn = 7u64;
        let (mut packet, pn_offset) = build_unprotected_initial(plaintext, pn);
        let original = packet.clone();
        let pn_byte_len = 4;

        protect_initial(
            &pair.client,
            pn,
            pn_byte_len,
            &mut packet,
            pn_offset,
            plaintext.len(),
        )
        .unwrap();
        // sanity: protected packet must differ from original
        assert_ne!(packet, original, "protect must modify bytes");

        let (decoded_pn, decoded_len) =
            unprotect_initial(&pair.client, pn - 1, &mut packet, pn_offset).unwrap();
        assert_eq!(decoded_pn, pn);
        assert_eq!(decoded_len, plaintext.len());
        // verify plaintext is back at the right offset
        let plaintext_offset = pn_offset + pn_byte_len;
        assert_eq!(
            &packet[plaintext_offset..plaintext_offset + decoded_len],
            plaintext,
            "plaintext must round-trip"
        );
    }

    #[test]
    fn round_trip_pn_zero() {
        let pair = initial_keys::derive(&RFC_DCID).unwrap();
        let plaintext = vec![0xabu8; 200];
        let (mut packet, pn_offset) = build_unprotected_initial(&plaintext, 0);
        protect_initial(&pair.client, 0, 4, &mut packet, pn_offset, plaintext.len()).unwrap();
        let (pn, len) = unprotect_initial(&pair.client, 0, &mut packet, pn_offset).unwrap();
        assert_eq!(pn, 0);
        assert_eq!(len, plaintext.len());
    }

    #[test]
    fn tampered_ciphertext_rejected() {
        let pair = initial_keys::derive(&RFC_DCID).unwrap();
        let plaintext = b"test payload bytes";
        let (mut packet, pn_offset) = build_unprotected_initial(plaintext, 1);
        protect_initial(&pair.client, 1, 4, &mut packet, pn_offset, plaintext.len()).unwrap();
        // flip a byte in the encrypted payload
        let tamper_offset = pn_offset + 4 + 5; // mid-ciphertext
        packet[tamper_offset] ^= 0x80;
        assert_eq!(
            unprotect_initial(&pair.client, 0, &mut packet, pn_offset),
            Err(PacketProtectionError::DecryptFailed)
        );
    }

    #[test]
    fn tampered_tag_rejected() {
        let pair = initial_keys::derive(&RFC_DCID).unwrap();
        let plaintext = b"test payload";
        let (mut packet, pn_offset) = build_unprotected_initial(plaintext, 1);
        protect_initial(&pair.client, 1, 4, &mut packet, pn_offset, plaintext.len()).unwrap();
        let tag_offset = packet.len() - 1;
        packet[tag_offset] ^= 0x01;
        assert_eq!(
            unprotect_initial(&pair.client, 0, &mut packet, pn_offset),
            Err(PacketProtectionError::DecryptFailed)
        );
    }

    #[test]
    fn tampered_header_rejected() {
        let pair = initial_keys::derive(&RFC_DCID).unwrap();
        let plaintext = b"test payload";
        let (mut packet, pn_offset) = build_unprotected_initial(plaintext, 1);
        protect_initial(&pair.client, 1, 4, &mut packet, pn_offset, plaintext.len()).unwrap();
        // flip a byte in the version field
        packet[1] ^= 0x80;
        assert_eq!(
            unprotect_initial(&pair.client, 0, &mut packet, pn_offset),
            Err(PacketProtectionError::DecryptFailed)
        );
    }

    #[test]
    fn server_keys_unprotect_client_protected_fails() {
        // a client-protected packet cannot be decrypted with the server's
        // initial keys — the per-direction key derivation in C5 produces
        // distinct AEAD keys for client and server.
        let pair = initial_keys::derive(&RFC_DCID).unwrap();
        let plaintext = b"client to server";
        let (mut packet, pn_offset) = build_unprotected_initial(plaintext, 3);
        protect_initial(&pair.client, 3, 4, &mut packet, pn_offset, plaintext.len()).unwrap();
        assert_eq!(
            unprotect_initial(&pair.server, 0, &mut packet, pn_offset),
            Err(PacketProtectionError::DecryptFailed)
        );
    }

    #[test]
    fn protect_buffer_too_small_rejected() {
        let pair = initial_keys::derive(&RFC_DCID).unwrap();
        let mut packet = alloc::vec![0u8; 4]; // way too small
        assert_eq!(
            protect_initial(&pair.client, 0, 4, &mut packet, 0, 100),
            Err(PacketProtectionError::BufferTooSmall)
        );
    }

    #[test]
    fn invalid_pn_byte_len_rejected() {
        let pair = initial_keys::derive(&RFC_DCID).unwrap();
        let mut packet = alloc::vec![0u8; 100];
        assert_eq!(
            protect_initial(&pair.client, 0, 0, &mut packet, 10, 50),
            Err(PacketProtectionError::InvalidPacketNumberLen)
        );
        assert_eq!(
            protect_initial(&pair.client, 0, 5, &mut packet, 10, 50),
            Err(PacketProtectionError::InvalidPacketNumberLen)
        );
    }

    /// RFC 9001 §5 — AES-256-GCM is a MUST-implement cipher suite.
    /// Round-trip protect/unprotect over a 1-RTT short-header packet.
    #[test]
    fn aes_256_gcm_short_header_round_trip() {
        let key = [0xab; aead::AES_256_GCM_KEY_LEN];
        let iv = [0xcd; QUIC_IV_LEN];
        let hp = [0xef; aead::AES_256_GCM_KEY_LEN];
        let plaintext = b"hello AES-256 short-header";
        let pn = 42u64;
        let pn_byte_len = 4usize;
        let dcid = [0u8; 8];
        let header_len = 1 + dcid.len() + pn_byte_len;
        let total_len = header_len + plaintext.len() + TAG_LEN;
        let mut packet = alloc::vec![0u8; total_len];
        packet[0] = 0x40 | u8::try_from(pn_byte_len - 1).unwrap();
        packet[1..1 + dcid.len()].copy_from_slice(&dcid);
        let pn_offset = 1 + dcid.len();
        packet[pn_offset..pn_offset + pn_byte_len].copy_from_slice(&(pn as u32).to_be_bytes());
        packet[pn_offset + pn_byte_len..pn_offset + pn_byte_len + plaintext.len()]
            .copy_from_slice(plaintext);

        protect_aes256gcm(
            &key,
            &iv,
            &hp,
            pn,
            pn_byte_len,
            &mut packet,
            pn_offset,
            plaintext.len(),
            false,
        )
        .expect("protect");
        let (recovered_pn, plaintext_len) = unprotect_aes256gcm(
            &key,
            &iv,
            &hp,
            pn.saturating_sub(1),
            &mut packet,
            pn_offset,
            false,
        )
        .expect("unprotect");
        assert_eq!(recovered_pn, pn);
        assert_eq!(plaintext_len, plaintext.len());
        assert_eq!(
            &packet[pn_offset + pn_byte_len..pn_offset + pn_byte_len + plaintext_len],
            plaintext
        );
    }

    /// RFC 9001 §5 — ChaCha20-Poly1305 is a MUST-implement cipher suite.
    /// Round-trip protect/unprotect over a 1-RTT short-header packet.
    #[test]
    fn chacha20_poly1305_short_header_round_trip() {
        let key = [0x12; aead::CHACHA20_POLY1305_KEY_LEN];
        let iv = [0x34; QUIC_IV_LEN];
        let hp = [0x56; aead::CHACHA20_POLY1305_KEY_LEN];
        let plaintext = b"hello ChaCha20-Poly1305 short-header";
        let pn = 7u64;
        let pn_byte_len = 4usize;
        let dcid = [0u8; 8];
        let header_len = 1 + dcid.len() + pn_byte_len;
        let total_len = header_len + plaintext.len() + TAG_LEN;
        let mut packet = alloc::vec![0u8; total_len];
        packet[0] = 0x40 | u8::try_from(pn_byte_len - 1).unwrap();
        packet[1..1 + dcid.len()].copy_from_slice(&dcid);
        let pn_offset = 1 + dcid.len();
        packet[pn_offset..pn_offset + pn_byte_len].copy_from_slice(&(pn as u32).to_be_bytes());
        packet[pn_offset + pn_byte_len..pn_offset + pn_byte_len + plaintext.len()]
            .copy_from_slice(plaintext);

        protect_chacha20poly1305(
            &key,
            &iv,
            &hp,
            pn,
            pn_byte_len,
            &mut packet,
            pn_offset,
            plaintext.len(),
            false,
        )
        .expect("protect");
        let (recovered_pn, plaintext_len) = unprotect_chacha20poly1305(
            &key,
            &iv,
            &hp,
            pn.saturating_sub(1),
            &mut packet,
            pn_offset,
            false,
        )
        .expect("unprotect");
        assert_eq!(recovered_pn, pn);
        assert_eq!(plaintext_len, plaintext.len());
        assert_eq!(
            &packet[pn_offset + pn_byte_len..pn_offset + pn_byte_len + plaintext_len],
            plaintext
        );
    }

    #[test]
    fn round_trip_packet_number_recovery() {
        // Send several packets, recover the full PN on the receive side.
        let pair = initial_keys::derive(&RFC_DCID).unwrap();
        let plaintext = b"hello";
        for full_pn in [0u64, 1, 100, 1023, 16_383, 65_535, 16_777_215] {
            let (mut packet, pn_offset) = build_unprotected_initial(plaintext, full_pn);
            protect_initial(
                &pair.client,
                full_pn,
                4,
                &mut packet,
                pn_offset,
                plaintext.len(),
            )
            .unwrap();
            let largest_received = full_pn.saturating_sub(1);
            let (decoded_pn, _len) =
                unprotect_initial(&pair.client, largest_received, &mut packet, pn_offset).unwrap();
            assert_eq!(decoded_pn, full_pn, "failed at PN={full_pn}");
        }
    }
}
