//! PostgreSQL MD5 password authentication (deprecated upstream, removed
//! in PostgreSQL 18; kept for legacy clients).
//!
//! The wire value a client sends in response to AuthenticationMD5Password
//! is `"md5" + hex(md5(hex(md5(password ++ user)) ++ salt))`. The server
//! recomputes the same value from the stored plaintext and the salt it
//! issued, then compares constant-time.

use md5::{Digest, Md5};

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(char::from_digit(u32::from(byte >> 4), 16).unwrap_or('0'));
        out.push(char::from_digit(u32::from(byte & 0x0f), 16).unwrap_or('0'));
    }
    out
}

fn md5_hex(parts: &[&[u8]]) -> String {
    let mut hasher = Md5::new();
    for part in parts {
        hasher.update(part);
    }
    hex_lower(&hasher.finalize())
}

/// Computes the `md5...` wire digest a client presents for the given salt.
#[must_use]
pub fn md5_password(user: &str, password: &str, salt: [u8; 4]) -> String {
    let inner = md5_hex(&[password.as_bytes(), user.as_bytes()]);
    let outer = md5_hex(&[inner.as_bytes(), &salt]);
    format!("md5{outer}")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;

    #[test]
    fn known_vector_matches_postgres() {
        // locked vector: user=postgres, password=secret, salt=01020304
        let digest = md5_password("postgres", "secret", [0x01, 0x02, 0x03, 0x04]);

        assert_eq!(digest, "md5bb41a296aab6baccb36ff243a562abff");
    }

    #[test]
    fn wrong_password_yields_different_digest() {
        let salt = [0x01, 0x02, 0x03, 0x04];
        let right = md5_password("postgres", "secret", salt);
        let wrong = md5_password("postgres", "guess", salt);

        assert_ne!(right, wrong, "different passwords must not collide");
    }

    #[test]
    fn salt_changes_the_digest() {
        let first = md5_password("postgres", "secret", [0x01, 0x02, 0x03, 0x04]);
        let second = md5_password("postgres", "secret", [0x05, 0x06, 0x07, 0x08]);

        assert_ne!(
            first, second,
            "different salts must produce different digests"
        );
    }
}
