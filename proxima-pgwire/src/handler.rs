//! Internal wire-encoding helpers shared by the connection driver.
//!
//! The public query surface is no longer a bespoke trait — a SQL engine
//! is a `proxima_primitives::pipe::Pipe` (see [`crate::pipe_contract`]). What remains
//! here is the driver's own machinery: [`ErrorInfo`] (the in-flight
//! ErrorResponse shape, distinct from the contract's owned
//! [`crate::pipe_contract::ErrorReply`]) and the exact-size
//! reserve/commit/encode primitives every driver helper writes through.

use std::io;

use proxima_protocols::pgwire_codec::backend::ErrorResponseWriter;
use proxima_protocols::pgwire_codec::types::error_field;

/// SQL-level error the driver emits as an ErrorResponse. NUL bytes are
/// stripped at construction so encoding cannot fail on them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorInfo {
    pub severity: String,
    pub sqlstate: String,
    pub message: String,
    pub detail: Option<String>,
    pub hint: Option<String>,
}

fn strip_nul(text: impl Into<String>) -> String {
    let text = text.into();
    if text.as_bytes().contains(&0) {
        text.replace('\0', " ")
    } else {
        text
    }
}

impl ErrorInfo {
    #[must_use]
    pub fn new(sqlstate: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: "ERROR".into(),
            sqlstate: strip_nul(sqlstate),
            message: strip_nul(message),
            detail: None,
            hint: None,
        }
    }

    /// `0A000 feature_not_supported` — the typed answer for protocol
    /// surfaces the driver does not implement (COPY, fast-path calls).
    #[must_use]
    pub fn feature_not_supported(message: impl Into<String>) -> Self {
        Self::new("0A000", message)
    }

    /// `42601 syntax_error`.
    #[must_use]
    pub fn syntax(message: impl Into<String>) -> Self {
        Self::new("42601", message)
    }

    /// `28P01 invalid_password`.
    #[must_use]
    pub fn invalid_password(user: &str) -> Self {
        Self::new(
            "28P01",
            format!("password authentication failed for user \"{user}\""),
        )
    }

    #[must_use]
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(strip_nul(detail));
        self
    }

    #[must_use]
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(strip_nul(hint));
        self
    }

    #[must_use]
    pub fn fatal(mut self) -> Self {
        self.severity = "FATAL".into();
        self
    }

    /// Lowers a contract [`crate::pipe_contract::ErrorReply`] (an engine's
    /// reported SQL error) into the in-flight wire shape.
    #[must_use]
    pub fn from_reply(reply: &crate::pipe_contract::ErrorReply) -> Self {
        Self {
            severity: strip_nul(reply.severity.clone()),
            sqlstate: strip_nul(reply.sqlstate.clone()),
            message: strip_nul(reply.message.clone()),
            detail: reply.detail.as_ref().map(strip_nul),
            hint: reply.hint.as_ref().map(strip_nul),
        }
    }

    /// Lowers a contract [`crate::pipe_contract::NoticeReply`] into the
    /// wire shape so the driver can emit it as a NoticeResponse.
    #[must_use]
    pub fn from_notice(notice: &crate::pipe_contract::NoticeReply) -> Self {
        Self::new(notice.sqlstate.clone(), notice.message.clone())
    }
}

impl std::fmt::Display for ErrorInfo {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{}: {} ({})",
            self.severity, self.message, self.sqlstate
        )
    }
}

impl std::error::Error for ErrorInfo {}

pub(crate) fn error_response_size(info: &ErrorInfo) -> usize {
    let optional = info.detail.as_ref().map_or(0, |detail| detail.len() + 2)
        + info.hint.as_ref().map_or(0, |hint| hint.len() + 2);
    6 + 2 * (info.severity.len() + 2) + info.sqlstate.len() + 2 + info.message.len() + 2 + optional
}

pub(crate) fn write_error_fields(
    writer: &mut ErrorResponseWriter<'_>,
    info: &ErrorInfo,
) -> Result<(), proxima_protocols::pgwire_codec::EncodeError> {
    writer.field(error_field::SEVERITY, info.severity.as_bytes())?;
    writer.field(
        error_field::SEVERITY_NON_LOCALIZED,
        info.severity.as_bytes(),
    )?;
    writer.field(error_field::CODE, info.sqlstate.as_bytes())?;
    writer.field(error_field::MESSAGE, info.message.as_bytes())?;
    if let Some(detail) = &info.detail {
        writer.field(error_field::DETAIL, detail.as_bytes())?;
    }
    if let Some(hint) = &info.hint {
        writer.field(error_field::HINT, hint.as_bytes())?;
    }
    Ok(())
}

/// Grows `out` by exactly `size` zeroed bytes, returning the write
/// offset.
pub(crate) fn reserve(out: &mut Vec<u8>, size: usize) -> usize {
    let start = out.len();
    out.resize(start + size, 0);
    start
}

/// Truncates to the encoded size on success, rolls back on failure.
/// Encode errors at this layer mean a driver sizing bug, surfaced as
/// `InvalidInput` instead of corrupting the wire.
pub(crate) fn commit(
    out: &mut Vec<u8>,
    start: usize,
    outcome: Result<usize, proxima_protocols::pgwire_codec::EncodeError>,
) -> io::Result<()> {
    match outcome {
        Ok(written) => {
            out.truncate(start + written);
            Ok(())
        }
        Err(error) => {
            out.truncate(start);
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("encode: {error}"),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn error_info_strip_nul_in_sqlstate_and_message() {
        let info = ErrorInfo::new("42P0\x001", "bad\x00query\x00here");

        assert!(
            !info.sqlstate.as_bytes().contains(&0),
            "sqlstate must not contain NUL"
        );
        assert!(
            !info.message.as_bytes().contains(&0),
            "message must not contain NUL"
        );
        assert_eq!(info.sqlstate, "42P0 1");
        assert_eq!(info.message, "bad query here");
    }

    #[test]
    fn error_info_strip_nul_in_detail_and_hint() {
        let info = ErrorInfo::new("42601", "syntax error")
            .with_detail("near\x00position 5")
            .with_hint("try\x00this instead");

        assert_eq!(info.detail.as_deref(), Some("near position 5"));
        assert_eq!(info.hint.as_deref(), Some("try this instead"));
    }

    #[test]
    fn error_info_feature_not_supported_produces_0a000() {
        let info = ErrorInfo::feature_not_supported("COPY is not implemented");

        assert_eq!(info.sqlstate, "0A000");
        assert_eq!(info.severity, "ERROR");
        assert!(info.message.contains("COPY"));
    }

    #[test]
    fn error_info_invalid_password_produces_28p01_with_user_in_message() {
        let info = ErrorInfo::invalid_password("carol");

        assert_eq!(info.sqlstate, "28P01");
        assert!(
            info.message.contains("carol"),
            "user must appear in message"
        );
    }

    #[test]
    fn error_info_fatal_flips_severity() {
        let info = ErrorInfo::new("08P01", "protocol violation").fatal();

        assert_eq!(info.severity, "FATAL");
    }

    #[test]
    fn error_info_from_reply_carries_all_fields() {
        let reply = crate::pipe_contract::ErrorReply {
            severity: "ERROR".into(),
            sqlstate: "42601".into(),
            message: "syntax error".into(),
            detail: Some("near \"FORM\"".into()),
            hint: Some("did you mean FROM?".into()),
        };
        let info = ErrorInfo::from_reply(&reply);

        assert_eq!(info.sqlstate, "42601");
        assert_eq!(info.message, "syntax error");
        assert_eq!(info.detail.as_deref(), Some("near \"FORM\""));
        assert_eq!(info.hint.as_deref(), Some("did you mean FROM?"));
    }

    #[test]
    fn error_info_display_includes_severity_message_and_sqlstate() {
        let info = ErrorInfo::new("42601", "syntax error near \"FORM\"");
        let display = format!("{info}");

        assert!(display.contains("ERROR"));
        assert!(display.contains("42601"));
        assert!(display.contains("syntax error near"));
    }
}
