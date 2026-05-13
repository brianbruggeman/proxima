//! Information taint tracking — `Tainted<T>` vs `Trusted<T>`.
//!
//! Data that came from the child process is [`Tainted`]. Data
//! synthesized by the dispatcher (from config, capabilities, or
//! const sources) is [`Trusted`]. Privileged grounds (real
//! filesystem, real network) accept ONLY `Trusted` arguments —
//! converting `Tainted` to `Trusted` requires explicit sanitization
//! through one of the `sanitize_*` functions in this module.
//!
//! Eliminates the class of "child-supplied path passed directly to
//! a real syscall" bugs at the type level.

extern crate alloc;

use alloc::string::String;

use super::path::{AbsolutePath, AbsolutePathError};

/// Data that arrived from the child process or from any untrusted
/// source. Cannot be passed to privileged APIs without explicit
/// sanitization.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Tainted<T>(pub(crate) T);

impl<T> Tainted<T> {
    /// Wrap data as tainted. This is the entry point for any data
    /// arriving from the child or other untrusted sources.
    #[must_use]
    pub const fn from_untrusted(value: T) -> Self {
        Self(value)
    }

    /// Borrow the inner data for inspection. Inspection is safe —
    /// only privileged USE is restricted.
    #[must_use]
    pub fn inspect(&self) -> &T {
        &self.0
    }
}

/// Data that has passed validation and is safe to use in privileged
/// contexts. Construction requires either a `const` source or a
/// successful sanitization via one of the `sanitize_*` functions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Trusted<T>(pub(crate) T);

impl<T> Trusted<T> {
    /// Wrap data as trusted. The caller asserts the data came from a
    /// trusted source (config, capability-holder, const).
    ///
    /// Mark `pub(crate)` access through wrappers — direct construction
    /// outside the crate must go through a `sanitize_*` function.
    #[must_use]
    pub(crate) const fn assert_trusted(value: T) -> Self {
        Self(value)
    }

    /// Borrow the inner trusted data.
    #[must_use]
    pub fn inner(&self) -> &T {
        &self.0
    }

    /// Unwrap into the inner value.
    #[must_use]
    pub fn into_inner(self) -> T {
        self.0
    }
}

/// Sanitize a tainted path string into a trusted absolute path.
/// Runs [`AbsolutePath::try_from_str`] validation and lifts a
/// successful result into `Trusted`.
///
/// # Errors
///
/// Returns [`AbsolutePathError`] if the path fails validation.
pub fn sanitize_absolute_path(
    tainted: Tainted<String>,
) -> Result<Trusted<AbsolutePath<String>>, AbsolutePathError> {
    let path = AbsolutePath::try_from_str(&tainted.0)?;
    Ok(Trusted::assert_trusted(path))
}

/// Promote a const-validated absolute path to trusted without any
/// runtime check. Safe because the const-validated path has already
/// passed all checks at compile time.
#[must_use]
pub const fn trust_const_path(
    path: AbsolutePath<&'static str>,
) -> Trusted<AbsolutePath<&'static str>> {
    Trusted::assert_trusted(path)
}
