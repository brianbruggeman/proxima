//! Authentication policies for the pgwire facade.
//!
//! The codec models the full authentication message matrix (cleartext,
//! MD5, SASL/SCRAM, GSS); the facade currently drives Trust and
//! Cleartext. MD5 (deprecated upstream, removed in PostgreSQL 18) and
//! SCRAM (a crypto component with its own review gate) are sequenced in
//! the pgwire discipline log as gates G1/G2.

use std::sync::Arc;

use zeroize::Zeroizing;

/// Verifies a cleartext password for a startup identity. Cleartext on a
/// plaintext socket is only sane on loopback; pair with TLS otherwise.
pub trait PasswordVerifier: Send + Sync + 'static {
    fn verify(&self, user: &str, database: Option<&str>, password: &str) -> bool;
}

/// Single static identity, compared in constant time. The password is
/// held in [`Zeroizing`] so it is wiped from memory on drop.
#[derive(Debug, Clone)]
pub struct StaticCredentials {
    pub username: String,
    pub password: Zeroizing<String>,
}

impl StaticCredentials {
    /// Builds a credential pair, wrapping the password for zeroize-on-drop —
    /// so a caller never has to reach for `zeroize` to construct one.
    #[must_use]
    pub fn new(username: impl Into<String>, password: impl Into<String>) -> Self {
        Self {
            username: username.into(),
            password: Zeroizing::new(password.into()),
        }
    }
}

pub(crate) fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut diff = u8::from(left.len() != right.len());
    for index in 0..left.len().min(right.len()) {
        diff |= left[index] ^ right[index];
    }
    diff == 0
}

impl PasswordVerifier for StaticCredentials {
    fn verify(&self, user: &str, _database: Option<&str>, password: &str) -> bool {
        let user_ok = constant_time_eq(user.as_bytes(), self.username.as_bytes());
        let password_ok = constant_time_eq(password.as_bytes(), self.password.as_bytes());
        user_ok && password_ok
    }
}

/// The authentication exchange a listener requires.
#[derive(Clone)]
pub enum PgAuth {
    /// accept every startup (loopback / dev / mTLS-fronted deployments)
    Trust,
    /// AuthenticationCleartextPassword, verified by the given policy
    Cleartext(Arc<dyn PasswordVerifier>),
    /// AuthenticationMD5Password (deprecated; removed in PostgreSQL 18)
    Md5(Arc<dyn PasswordSource>),
    /// AuthenticationSASL offering SCRAM-SHA-256 (RFC 7677)
    Scram(Arc<dyn PasswordSource>),
}

impl std::fmt::Debug for PgAuth {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Trust => formatter.write_str("PgAuth::Trust"),
            Self::Cleartext(_) => formatter.write_str("PgAuth::Cleartext"),
            Self::Md5(_) => formatter.write_str("PgAuth::Md5"),
            Self::Scram(_) => formatter.write_str("PgAuth::Scram"),
        }
    }
}

/// Yields the plaintext password an MD5/SCRAM verifier needs. Both methods
/// recompute a digest from the plaintext (MD5 over the salt, SCRAM over
/// PBKDF2), so a hash-only policy cannot drive them.
pub trait PasswordSource: Send + Sync + 'static {
    fn password_for(&self, user: &str) -> Option<&str>;
}

impl PasswordSource for StaticCredentials {
    fn password_for(&self, user: &str) -> Option<&str> {
        if constant_time_eq(user.as_bytes(), self.username.as_bytes()) {
            Some(self.password.as_str())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::sync::Arc;

    use super::*;

    fn creds(username: &str, password: &str) -> StaticCredentials {
        StaticCredentials {
            username: username.into(),
            password: Zeroizing::new(password.into()),
        }
    }

    #[test]
    fn correct_user_and_password_returns_true() {
        let credentials = creds("alice", "s3cr3t!");

        assert!(credentials.verify("alice", Some("appdb"), "s3cr3t!"));
    }

    #[test]
    fn wrong_password_returns_false() {
        let credentials = creds("alice", "s3cr3t!");

        assert!(!credentials.verify("alice", Some("appdb"), "wrong"));
    }

    #[test]
    fn wrong_user_returns_false() {
        let credentials = creds("alice", "s3cr3t!");

        assert!(!credentials.verify("bob", Some("appdb"), "s3cr3t!"));
    }

    #[test]
    fn empty_password_against_non_empty_returns_false() {
        let credentials = creds("alice", "s3cr3t!");

        assert!(!credentials.verify("alice", None, ""));
    }

    #[test]
    fn equal_length_wrong_password_returns_false() {
        let credentials = creds("alice", "aaaaaa");

        assert!(!credentials.verify("alice", None, "bbbbbb"));
    }

    #[test]
    fn database_parameter_is_ignored_in_verification() {
        let credentials = creds("alice", "pass");

        assert!(credentials.verify("alice", None, "pass"));
        assert!(credentials.verify("alice", Some("db1"), "pass"));
        assert!(credentials.verify("alice", Some("db2"), "pass"));
    }

    #[test]
    fn pg_auth_trust_debug_does_not_contain_password() {
        let auth = PgAuth::Trust;
        let debug_output = format!("{auth:?}");

        assert_eq!(debug_output, "PgAuth::Trust");
    }

    #[test]
    fn pg_auth_cleartext_debug_does_not_leak_password() {
        let auth = PgAuth::Cleartext(Arc::new(creds("alice", "top_secret_password")));
        let debug_output = format!("{auth:?}");

        assert_eq!(debug_output, "PgAuth::Cleartext");
        assert!(
            !debug_output.contains("top_secret_password"),
            "password must not appear in debug"
        );
        assert!(
            !debug_output.contains("alice"),
            "username must not appear in debug"
        );
    }

    #[test]
    fn empty_credentials_verify_empty_user_and_empty_password() {
        let credentials = creds("", "");

        assert!(credentials.verify("", None, ""));
    }

    #[test]
    fn non_empty_credentials_reject_empty_user() {
        let credentials = creds("alice", "pass");

        assert!(!credentials.verify("", None, "pass"));
    }

    #[test]
    fn password_source_returns_password_for_matching_user_only() {
        let credentials = creds("alice", "s3cr3t");

        assert_eq!(credentials.password_for("alice"), Some("s3cr3t"));
        assert_eq!(credentials.password_for("bob"), None);
    }

    #[test]
    fn pg_auth_md5_and_scram_debug_do_not_leak() {
        let md5 = PgAuth::Md5(Arc::new(creds("alice", "top_secret_password")));
        let scram = PgAuth::Scram(Arc::new(creds("alice", "top_secret_password")));

        assert_eq!(format!("{md5:?}"), "PgAuth::Md5");
        assert_eq!(format!("{scram:?}"), "PgAuth::Scram");
        assert!(!format!("{md5:?}").contains("top_secret_password"));
        assert!(!format!("{scram:?}").contains("top_secret_password"));
    }
}
