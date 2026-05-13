//! Const-validated absolute paths.
//!
//! [`AbsolutePath`] enforces absolute-path / no-`..`-traversal /
//! no-NUL invariants. For `&'static str` paths the validation runs
//! at compile time via [`AbsolutePath::new_const`]; for dynamic paths
//! the fallible [`AbsolutePath::try_from_str`] runs the same checks at
//! runtime and surfaces failures as typed errors.
//!
//! The type carries the proof: once an `AbsolutePath` is constructed,
//! every subsequent use is guaranteed-validated. Sanitize once, use
//! anywhere.

extern crate alloc;

use alloc::string::String;
use core::fmt;

/// Validated absolute path. The type parameter `P` carries the
/// backing storage — `&'static str` for compile-time constants,
/// `String` for sanitized dynamic input.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AbsolutePath<P>(P);

impl<P> AbsolutePath<P> {
    /// Borrow the inner storage as `&str` for read access.
    #[must_use]
    pub fn as_str(&self) -> &str
    where
        P: AsRef<str>,
    {
        self.0.as_ref()
    }
}

impl AbsolutePath<&'static str> {
    /// Construct from a static string. Validates at compile time:
    /// must start with `/`, must not contain `..` segments, must
    /// not contain NUL bytes.
    ///
    /// # Panics
    ///
    /// Const-panics at compile time if `path` does not satisfy the
    /// invariants. The panic message is the violation reason.
    #[must_use]
    pub const fn new_const(path: &'static str) -> Self {
        let bytes = path.as_bytes();
        assert!(!bytes.is_empty(), "AbsolutePath: empty path");
        assert!(bytes[0] == b'/', "AbsolutePath: must start with '/'");

        let mut idx = 0;
        let len = bytes.len();
        while idx < len {
            assert!(bytes[idx] != 0, "AbsolutePath: contains NUL byte");
            // Check for ".." as a path segment.
            if idx + 1 < len
                && bytes[idx] == b'.'
                && bytes[idx + 1] == b'.'
                && (idx == 0 || bytes[idx - 1] == b'/')
                && (idx + 2 == len || bytes[idx + 2] == b'/')
            {
                panic!("AbsolutePath: contains '..' traversal");
            }
            idx += 1;
        }

        Self(path)
    }
}

impl AbsolutePath<String> {
    /// Construct from a runtime string. Validates at construction;
    /// the returned [`Result`] carries the failure reason on rejection.
    ///
    /// # Errors
    ///
    /// Returns [`AbsolutePathError`] if `path` fails the validity
    /// checks (empty / non-absolute / NUL byte / `..` segment).
    pub fn try_from_str(path: &str) -> Result<Self, AbsolutePathError> {
        let bytes = path.as_bytes();
        if bytes.is_empty() {
            return Err(AbsolutePathError::Empty);
        }
        if bytes[0] != b'/' {
            return Err(AbsolutePathError::NotAbsolute);
        }

        for (idx, byte) in bytes.iter().enumerate() {
            if *byte == 0 {
                return Err(AbsolutePathError::ContainsNul);
            }
            if *byte == b'.'
                && idx + 1 < bytes.len()
                && bytes[idx + 1] == b'.'
                && (idx == 0 || bytes[idx - 1] == b'/')
                && (idx + 2 == bytes.len() || bytes[idx + 2] == b'/')
            {
                return Err(AbsolutePathError::ContainsTraversal);
            }
        }

        Ok(Self(String::from(path)))
    }
}

/// Reasons a path failed [`AbsolutePath`] validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbsolutePathError {
    /// The path string was empty.
    Empty,
    /// The path did not start with `/`.
    NotAbsolute,
    /// The path contained a NUL byte.
    ContainsNul,
    /// The path contained a `..` segment (traversal).
    ContainsTraversal,
}

impl fmt::Display for AbsolutePathError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("path is empty"),
            Self::NotAbsolute => formatter.write_str("path is not absolute (must start with '/')"),
            Self::ContainsNul => formatter.write_str("path contains a NUL byte"),
            Self::ContainsTraversal => formatter.write_str("path contains a '..' segment"),
        }
    }
}
