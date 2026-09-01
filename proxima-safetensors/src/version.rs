//! Format-version stamp carried in the safetensors `__metadata__` map --
//! the format's own free-form string-to-string header
//! ([`crate::parser::Manifest::metadata`]) is the existing metadata
//! mechanism this rides on; no new wire structure.
//! [`crate::writer::write_complete`] always stamps
//! [`crate::sized::FORMAT_VERSION_KEY`]; [`crate::parser::Manifest::format_version`]
//! reads it back. Modeled on `csr_db::grouped_store`'s
//! `format_version`/`stamp_format_version` pair and `csr_zkp::stark::wire`'s
//! unknown-version rejection, adapted to a metadata-map stamp instead of a
//! fixed wire byte.
//!
//! Absent stamp means the file predates this scheme -- every safetensors
//! file this workspace wrote before this constant existed -- and is
//! accepted as v1. Backward compatibility is mandatory, not a migration
//! path: nothing already on disk is invalidated by landing this.

use alloc::collections::BTreeMap;
use alloc::string::String;

use crate::error::SafetensorsError;
use crate::sized::{FORMAT_VERSION_KEY, FORMAT_VERSION_MAJOR, FORMAT_VERSION_MINOR};

/// The `major.minor` string [`crate::writer::write_complete`] stamps into
/// every file it writes.
pub(crate) fn current_version_string() -> String {
    alloc::format!("{FORMAT_VERSION_MAJOR}.{FORMAT_VERSION_MINOR}")
}

/// Reads and validates the format-version stamp out of a parsed
/// `__metadata__` map. See the module doc for the accept/reject table;
/// concretely:
///
/// - No [`FORMAT_VERSION_KEY`] entry at all: `Ok((1, 0))`.
/// - A value that doesn't parse as `major.minor`:
///   [`SafetensorsError::InvalidFormatVersion`].
/// - A major greater than [`FORMAT_VERSION_MAJOR`]:
///   [`SafetensorsError::UnsupportedFormatVersion`].
/// - Any major at or below [`FORMAT_VERSION_MAJOR`] (any minor, including
///   one newer than [`FORMAT_VERSION_MINOR`]): accepted, since a minor
///   bump is additive by definition.
pub(crate) fn parse(metadata: &BTreeMap<String, String>) -> Result<(u16, u16), SafetensorsError> {
    let Some(raw) = metadata.get(FORMAT_VERSION_KEY) else {
        return Ok((1, 0));
    };

    let (major_text, minor_text) = raw
        .split_once('.')
        .ok_or_else(|| SafetensorsError::InvalidFormatVersion { found: raw.clone() })?;
    let major: u16 = major_text
        .parse()
        .map_err(|_| SafetensorsError::InvalidFormatVersion { found: raw.clone() })?;
    let minor: u16 = minor_text
        .parse()
        .map_err(|_| SafetensorsError::InvalidFormatVersion { found: raw.clone() })?;

    if major > FORMAT_VERSION_MAJOR {
        return Err(SafetensorsError::UnsupportedFormatVersion {
            found: raw.clone(),
            supported_major: FORMAT_VERSION_MAJOR,
        });
    }

    Ok((major, minor))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn absent_stamp_is_accepted_as_v1() {
        let metadata = BTreeMap::new();
        assert_eq!(parse(&metadata).expect("absent stamp is v1"), (1, 0));
    }

    #[test]
    fn current_stamp_round_trips() {
        let mut metadata = BTreeMap::new();
        metadata.insert(FORMAT_VERSION_KEY.to_string(), current_version_string());
        assert_eq!(
            parse(&metadata).expect("current stamp parses"),
            (FORMAT_VERSION_MAJOR, FORMAT_VERSION_MINOR)
        );
    }

    #[test]
    fn a_newer_minor_at_the_same_major_is_accepted() {
        let mut metadata = BTreeMap::new();
        metadata.insert(FORMAT_VERSION_KEY.to_string(), alloc::format!("{FORMAT_VERSION_MAJOR}.9999"));
        assert_eq!(parse(&metadata).expect("newer minor is additive"), (FORMAT_VERSION_MAJOR, 9999));
    }

    #[test]
    fn a_newer_major_is_a_typed_error_naming_the_found_and_supported_versions() {
        let mut metadata = BTreeMap::new();
        let newer_major = FORMAT_VERSION_MAJOR + 1;
        metadata.insert(FORMAT_VERSION_KEY.to_string(), alloc::format!("{newer_major}.0"));
        let outcome = parse(&metadata);
        assert!(matches!(
            outcome,
            Err(SafetensorsError::UnsupportedFormatVersion { supported_major, .. })
                if supported_major == FORMAT_VERSION_MAJOR
        ));
        let message = outcome.unwrap_err().to_string();
        assert!(message.contains(&alloc::format!("{newer_major}.0")), "error must name the file's version: {message}");
        assert!(message.contains(&FORMAT_VERSION_MAJOR.to_string()), "error must name the supported range: {message}");
    }

    #[test]
    fn a_malformed_stamp_is_a_typed_error_not_a_panic() {
        let mut metadata = BTreeMap::new();
        metadata.insert(FORMAT_VERSION_KEY.to_string(), "not-a-version".to_string());
        let outcome = parse(&metadata);
        assert!(matches!(outcome, Err(SafetensorsError::InvalidFormatVersion { .. })));
    }
}
