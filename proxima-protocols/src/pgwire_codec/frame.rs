//! Tagged-frame splitting shared by the frontend and backend decoders.
//!
//! Post-startup PostgreSQL messages share one frame shape: a one-byte
//! message tag, an Int32 length that counts itself plus the body (but
//! not the tag), then the body. Startup-phase messages are untagged and
//! handled by [`super::frontend::parse_initial`].

use super::error::ParseError;

/// Minimum tagged frame: tag byte + length field.
pub(crate) const TAGGED_HEADER: usize = 5;

/// A complete frame split off the input: tag, body, total consumed.
pub(crate) type TaggedFrame<'a> = (u8, &'a [u8], usize);

/// Splits one complete tagged frame off the front of `input`.
///
/// Returns `Ok(None)` when more bytes are needed, otherwise the tag,
/// the body slice, and the total bytes consumed from `input`.
pub(crate) fn split_tagged(input: &[u8]) -> Result<Option<TaggedFrame<'_>>, ParseError> {
    if input.len() < TAGGED_HEADER {
        return Ok(None);
    }
    let tag = input[0];
    let length = i32::from_be_bytes([input[1], input[2], input[3], input[4]]);
    let Ok(length) = usize::try_from(length) else {
        return Err(ParseError::BadLength { tag, length });
    };
    if length < 4 {
        let reported = i32::try_from(length).unwrap_or(i32::MAX);
        return Err(ParseError::BadLength {
            tag,
            length: reported,
        });
    }
    let total = 1 + length;
    if input.len() < total {
        return Ok(None);
    }
    Ok(Some((tag, &input[TAGGED_HEADER..total], total)))
}
