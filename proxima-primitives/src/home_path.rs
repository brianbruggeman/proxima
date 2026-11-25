//! Home-relative path configuration values, shared by every crate that reads a cache directory from config.
//!
//! `std` only: it builds an [`std::path::PathBuf`]. The caller passes `$HOME` in, so a test never mutates the
//! process environment, and the caller owns the warning when the home is missing because only it knows which
//! setting the user should set instead.

use std::ffi::OsString;
use std::path::PathBuf;

/// `value` with a leading `~/` replaced by `home`; a value without that prefix is returned as it is and never
/// needs a home. `None` when the prefix is present and `home` is not.
#[must_use]
pub fn expand_home(value: &str, home: Option<OsString>) -> Option<PathBuf> {
    match value.strip_prefix("~/") {
        None => Some(PathBuf::from(value)),
        Some(relative) => home.map(|home| PathBuf::from(home).join(relative)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn home() -> Option<OsString> {
        Some(OsString::from("/home/proxima"))
    }

    #[test]
    fn a_leading_tilde_slash_is_replaced_by_the_home() {
        assert_eq!(
            expand_home("~/.cache/proxima/weights", home()),
            Some(PathBuf::from("/home/proxima/.cache/proxima/weights"))
        );
    }

    #[test]
    fn a_leading_tilde_slash_without_a_home_is_none() {
        assert_eq!(expand_home("~/.cache/proxima/weights", None), None);
    }

    #[test]
    fn a_path_without_the_prefix_is_returned_whether_or_not_a_home_exists() {
        assert_eq!(
            expand_home("/var/cache/proxima/weights", None),
            Some(PathBuf::from("/var/cache/proxima/weights"))
        );
        assert_eq!(
            expand_home("relative/dir", home()),
            Some(PathBuf::from("relative/dir"))
        );
    }
}
