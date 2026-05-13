//! Small POD value types shared by the frontend and backend codecs.
//!
//! Everything here is `Copy` and borrowed — text stays bytes until the
//! semantic edge (per the workspace bytes-at-the-boundary rule). UTF-8
//! validation is the caller's choice via [`PgStr::to_str`].

use core::fmt;

/// Borrowed view of a wire string field (the bytes between the field
/// start and its NUL terminator, exclusive).
///
/// PostgreSQL strings are NUL-terminated byte sequences with no declared
/// encoding at the protocol layer; encoding is negotiated via the
/// `client_encoding` startup parameter. The codec therefore exposes raw
/// bytes and defers UTF-8 validation to the SQL / semantic edge.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PgStr<'a> {
    bytes: &'a [u8],
}

impl<'a> PgStr<'a> {
    /// Wraps raw bytes as a wire string view. The bytes must not contain
    /// a NUL; encode validates this, decode guarantees it.
    #[must_use]
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }

    #[must_use]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.bytes.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// UTF-8 view; fails on non-UTF-8 client encodings.
    ///
    /// # Errors
    /// Returns the underlying [`core::str::Utf8Error`] when the bytes are
    /// not valid UTF-8.
    pub const fn to_str(&self) -> Result<&'a str, core::str::Utf8Error> {
        core::str::from_utf8(self.bytes)
    }
}

impl fmt::Debug for PgStr<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "PgStr({})", self.bytes.escape_ascii())
    }
}

impl fmt::Display for PgStr<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.bytes.escape_ascii())
    }
}

impl PartialEq<[u8]> for PgStr<'_> {
    fn eq(&self, other: &[u8]) -> bool {
        self.bytes == other
    }
}

impl PartialEq<&[u8]> for PgStr<'_> {
    fn eq(&self, other: &&[u8]) -> bool {
        self.bytes == *other
    }
}

impl PartialEq<str> for PgStr<'_> {
    fn eq(&self, other: &str) -> bool {
        self.bytes == other.as_bytes()
    }
}

impl PartialEq<&str> for PgStr<'_> {
    fn eq(&self, other: &&str) -> bool {
        self.bytes == other.as_bytes()
    }
}

/// PostgreSQL object identifier (type OIDs in Parse / RowDescription /
/// ParameterDescription).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Oid(pub u32);

impl fmt::Display for Oid {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// Wire format code for a parameter or result column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i16)]
pub enum FormatCode {
    #[default]
    Text = 0,
    Binary = 1,
}

impl FormatCode {
    #[must_use]
    pub const fn from_i16(raw: i16) -> Option<Self> {
        match raw {
            0 => Some(Self::Text),
            1 => Some(Self::Binary),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_i16(self) -> i16 {
        self as i16
    }
}

/// Backend transaction status byte carried by ReadyForQuery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum TransactionStatus {
    /// not in a transaction block (`I`)
    #[default]
    Idle = b'I',
    /// in a transaction block (`T`)
    InTransaction = b'T',
    /// in a failed transaction block; queries rejected until end (`E`)
    Failed = b'E',
}

impl TransactionStatus {
    #[must_use]
    pub const fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            b'I' => Some(Self::Idle),
            b'T' => Some(Self::InTransaction),
            b'E' => Some(Self::Failed),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_byte(self) -> u8 {
        self as u8
    }
}

/// Target of a frontend Describe or Close message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum StatementTarget {
    /// a prepared statement (`S`)
    Statement = b'S',
    /// a portal (`P`)
    Portal = b'P',
}

impl StatementTarget {
    #[must_use]
    pub const fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            b'S' => Some(Self::Statement),
            b'P' => Some(Self::Portal),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_byte(self) -> u8 {
        self as u8
    }
}

/// Overall COPY format declared by CopyInResponse / CopyOutResponse /
/// CopyBothResponse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i8)]
pub enum CopyFormat {
    #[default]
    Text = 0,
    Binary = 1,
}

impl CopyFormat {
    #[must_use]
    pub const fn from_i8(raw: i8) -> Option<Self> {
        match raw {
            0 => Some(Self::Text),
            1 => Some(Self::Binary),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_i8(self) -> i8 {
        self as i8
    }
}

/// Negotiated protocol version from the startup packet.
///
/// The wire carries it as one Int32: `major << 16 | minor`. Version 3.0
/// is the baseline; 3.2 (PostgreSQL 17+) extends cancel-key lengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ProtocolVersion {
    pub major: u16,
    pub minor: u16,
}

impl ProtocolVersion {
    pub const V3_0: Self = Self { major: 3, minor: 0 };

    #[must_use]
    pub const fn from_code(code: i32) -> Self {
        let unsigned = code as u32;
        Self {
            major: (unsigned >> 16) as u16,
            minor: (unsigned & 0xffff) as u16,
        }
    }

    #[must_use]
    pub const fn as_code(self) -> i32 {
        (((self.major as u32) << 16) | self.minor as u32) as i32
    }
}

impl fmt::Display for ProtocolVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}.{}", self.major, self.minor)
    }
}

/// Field-type bytes used in ErrorResponse / NoticeResponse field pairs,
/// per the PostgreSQL "error and notice message fields" appendix.
pub mod error_field {
    /// localized severity (always present)
    pub const SEVERITY: u8 = b'S';
    /// non-localized severity (9.6+; always present)
    pub const SEVERITY_NON_LOCALIZED: u8 = b'V';
    /// SQLSTATE code (always present)
    pub const CODE: u8 = b'C';
    /// primary human-readable message (always present)
    pub const MESSAGE: u8 = b'M';
    pub const DETAIL: u8 = b'D';
    pub const HINT: u8 = b'H';
    pub const POSITION: u8 = b'P';
    pub const INTERNAL_POSITION: u8 = b'p';
    pub const INTERNAL_QUERY: u8 = b'q';
    pub const WHERE: u8 = b'W';
    pub const SCHEMA: u8 = b's';
    pub const TABLE: u8 = b't';
    pub const COLUMN: u8 = b'c';
    pub const DATA_TYPE: u8 = b'd';
    pub const CONSTRAINT: u8 = b'n';
    pub const FILE: u8 = b'F';
    pub const LINE: u8 = b'L';
    pub const ROUTINE: u8 = b'R';
}

// one test formats an Oid via alloc::format!; this crate carries no alloc
// dependency for its no_std tier, so the suite needs std, not just test
#[cfg(all(test, feature = "std"))]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::text(0i16, Some(FormatCode::Text))]
    #[case::binary(1i16, Some(FormatCode::Binary))]
    #[case::invalid_2(2i16, None)]
    #[case::invalid_7(7i16, None)]
    #[case::negative(-1i16, None)]
    fn format_code_from_i16(#[case] input: i16, #[case] expected: Option<FormatCode>) {
        assert_eq!(FormatCode::from_i16(input), expected);
    }

    #[rstest]
    #[case::text(FormatCode::Text, 0i16)]
    #[case::binary(FormatCode::Binary, 1i16)]
    fn format_code_as_i16(#[case] code: FormatCode, #[case] expected: i16) {
        assert_eq!(code.as_i16(), expected);
    }

    #[rstest]
    #[case::idle(b'I', Some(TransactionStatus::Idle))]
    #[case::in_transaction(b'T', Some(TransactionStatus::InTransaction))]
    #[case::failed(b'E', Some(TransactionStatus::Failed))]
    #[case::invalid_x(b'X', None)]
    #[case::invalid_zero(0u8, None)]
    fn transaction_status_from_byte(
        #[case] input: u8,
        #[case] expected: Option<TransactionStatus>,
    ) {
        assert_eq!(TransactionStatus::from_byte(input), expected);
    }

    #[rstest]
    #[case::idle(TransactionStatus::Idle, b'I')]
    #[case::in_transaction(TransactionStatus::InTransaction, b'T')]
    #[case::failed(TransactionStatus::Failed, b'E')]
    fn transaction_status_as_byte(#[case] status: TransactionStatus, #[case] expected: u8) {
        assert_eq!(status.as_byte(), expected);
    }

    #[rstest]
    #[case::statement(b'S', Some(StatementTarget::Statement))]
    #[case::portal(b'P', Some(StatementTarget::Portal))]
    #[case::invalid_x(b'X', None)]
    fn statement_target_from_byte(#[case] input: u8, #[case] expected: Option<StatementTarget>) {
        assert_eq!(StatementTarget::from_byte(input), expected);
    }

    #[rstest]
    #[case::text(0i8, Some(CopyFormat::Text))]
    #[case::binary(1i8, Some(CopyFormat::Binary))]
    #[case::invalid_2(2i8, None)]
    #[case::negative(-1i8, None)]
    fn copy_format_from_i8(#[case] input: i8, #[case] expected: Option<CopyFormat>) {
        assert_eq!(CopyFormat::from_i8(input), expected);
    }

    #[test]
    fn protocol_version_v3_0_round_trip() {
        let version = ProtocolVersion::V3_0;
        let code = version.as_code();
        let decoded = ProtocolVersion::from_code(code);
        assert_eq!(decoded, version);
        assert_eq!(version.major, 3);
        assert_eq!(version.minor, 0);
    }

    #[test]
    fn protocol_version_code_encodes_major_minor() {
        let version = ProtocolVersion { major: 3, minor: 2 };
        let code = version.as_code();
        let decoded = ProtocolVersion::from_code(code);
        assert_eq!(decoded.major, 3);
        assert_eq!(decoded.minor, 2);
    }

    #[test]
    fn pgstr_equality_with_bytes_and_str() {
        let pgstr = PgStr::new(b"hello");
        assert_eq!(pgstr, b"hello".as_slice());
        assert_eq!(pgstr, "hello");
        assert!(!pgstr.is_empty());
        assert_eq!(pgstr.len(), 5);
    }

    #[test]
    fn pgstr_empty_is_empty() {
        let pgstr = PgStr::new(b"");
        assert!(pgstr.is_empty());
        assert_eq!(pgstr.len(), 0);
    }

    #[test]
    fn pgstr_to_str_valid_utf8() {
        let pgstr = PgStr::new(b"select 1");
        assert_eq!(pgstr.to_str().expect("valid utf8"), "select 1");
    }

    #[test]
    fn oid_display_and_ordering() {
        let oid_a = Oid(23);
        let oid_b = Oid(25);
        assert!(oid_a < oid_b);
        assert_eq!(format!("{oid_a}"), "23");
    }
}
