//! BLAKE3 hashing, keyed hashing, and key derivation, as fixed-size
//! functions.
//!
//! Every signature here takes slices and returns arrays. Nothing allocates,
//! nothing borrows for longer than the call, and nothing needs a context
//! object to hold: these are the leaf primitives the handshake state
//! machines are built from, so they are free functions rather than methods
//! on a `Kdf` type. A caller that wants a domain-separated sub-key calls
//! [`derive_key`]; there is no `Blake3Kdf` to construct first.

/// Unkeyed hash. Use [`keyed_hash`] for a MAC and [`derive_key`] for
/// deriving key material — an unkeyed hash is neither.
#[must_use]
pub fn hash(data: &[u8]) -> [u8; 32] {
    *blake3::hash(data).as_bytes()
}

/// Keyed hash — a PRF, and the MAC the handshake transcripts are
/// authenticated with.
#[must_use]
pub fn keyed_hash(key: &[u8; 32], data: &[u8]) -> [u8; 32] {
    *blake3::keyed_hash(key, data).as_bytes()
}

/// Derive a 32-byte sub-key from key material, domain-separated by
/// `context`.
///
/// `context` must be a hardcoded, globally unique string — never
/// attacker-influenced and never reused across two derivations that must
/// yield different keys. That is BLAKE3's own KDF contract; getting it wrong
/// silently collides sub-keys that the protocol assumes are independent.
#[must_use]
pub fn derive_key(context: &str, key_material: &[u8]) -> [u8; 32] {
    blake3::derive_key(context, key_material)
}

/// Derive arbitrary-length key material into a caller-provided buffer.
///
/// The buffer-shaped counterpart to [`derive_key`], for the cases that need
/// more or less than 32 bytes (an AEAD key plus its nonce prefix in one
/// draw). The caller owns the storage, so this stays no-alloc at any output
/// length.
pub fn derive_key_into(context: &str, key_material: &[u8], output: &mut [u8]) {
    let mut hasher = blake3::Hasher::new_derive_key(context);
    hasher.update(key_material);
    hasher.finalize_xof().fill(output);
}

#[cfg(test)]
mod tests {
    use super::{derive_key, derive_key_into, hash, keyed_hash};

    #[test]
    fn hash_is_deterministic_and_distinguishes_input() {
        assert_eq!(hash(b"centauri"), hash(b"centauri"));
        assert_ne!(hash(b"centauri"), hash(b"centaurj"));
    }

    #[test]
    fn keyed_hash_distinguishes_keys_and_messages() {
        let first_key = [1u8; 32];
        let second_key = [2u8; 32];

        assert_eq!(
            keyed_hash(&first_key, b"msg"),
            keyed_hash(&first_key, b"msg")
        );
        assert_ne!(
            keyed_hash(&first_key, b"msg"),
            keyed_hash(&second_key, b"msg")
        );
        assert_ne!(
            keyed_hash(&first_key, b"msg"),
            keyed_hash(&first_key, b"other")
        );
    }

    #[test]
    fn derive_key_is_domain_separated() {
        let material = [7u8; 32];

        assert_ne!(
            derive_key("proxima-centauri test a", &material),
            derive_key("proxima-centauri test b", &material)
        );
    }

    #[test]
    fn derive_key_into_matches_derive_key_at_32_bytes() {
        let material = [9u8; 32];
        let mut wide = [0u8; 32];

        derive_key_into("proxima-centauri parity", &material, &mut wide);

        assert_eq!(wide, derive_key("proxima-centauri parity", &material));
    }

    #[test]
    fn derive_key_into_fills_beyond_32_bytes() {
        let material = [3u8; 32];
        let mut wide = [0u8; 96];

        derive_key_into("proxima-centauri xof", &material, &mut wide);

        assert_ne!(wide[..32], wide[32..64]);
        assert_ne!(wide[32..64], wide[64..]);
    }
}
