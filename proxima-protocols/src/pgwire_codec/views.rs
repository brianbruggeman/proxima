//! Lazy borrowed views over variable-length message sections.
//!
//! Each view is validated structurally in a single pass at message parse
//! time, then iterated infallibly — the iterators re-walk the already
//! validated bytes and never allocate. This is what keeps the decode
//! path zero-allocation regardless of parameter / column / field counts:
//! there is no `Vec` to fill, only offsets into the caller's buffer.

use core::iter::FusedIterator;

use memchr::memchr;

use super::cursor::Reader;
use super::error::ParseError;
use super::types::{FormatCode, Oid, PgStr};

/// Startup parameter list: `key \0 value \0 ... \0` pairs terminated by
/// one empty key.
#[derive(Debug, Clone, Copy)]
pub struct StartupParams<'a> {
    pairs: &'a [u8],
}

impl<'a> StartupParams<'a> {
    pub(crate) fn validate(reader: &mut Reader<'a>) -> Result<Self, ParseError> {
        let mark = reader.mark();
        let mut end = mark;
        loop {
            let key = reader.take_cstr()?;
            if key.is_empty() {
                break;
            }
            reader.take_cstr()?;
            end = reader.mark();
        }
        Ok(Self {
            pairs: reader.slice(mark, end),
        })
    }

    /// Linear lookup of one parameter by key (`user`, `database`, ...).
    #[must_use]
    pub fn get(&self, key: &[u8]) -> Option<PgStr<'a>> {
        self.iter()
            .find(|(name, _)| *name == *key)
            .map(|(_, value)| value)
    }

    #[must_use]
    pub fn iter(&self) -> StartupParamsIter<'a> {
        StartupParamsIter { rest: self.pairs }
    }
}

impl<'a> IntoIterator for StartupParams<'a> {
    type Item = (PgStr<'a>, PgStr<'a>);
    type IntoIter = StartupParamsIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Debug, Clone)]
pub struct StartupParamsIter<'a> {
    rest: &'a [u8],
}

impl<'a> Iterator for StartupParamsIter<'a> {
    type Item = (PgStr<'a>, PgStr<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        let key_end = memchr(0, self.rest)?;
        let value_start = key_end + 1;
        let value_len = memchr(0, self.rest.get(value_start..)?)?;
        let key = PgStr::new(&self.rest[..key_end]);
        let value = PgStr::new(&self.rest[value_start..value_start + value_len]);
        self.rest = &self.rest[value_start + value_len + 1..];
        Some((key, value))
    }
}

impl FusedIterator for StartupParamsIter<'_> {}

/// Fixed-width Int32 OID list prefixed by an Int16 count (Parse,
/// ParameterDescription).
#[derive(Debug, Clone, Copy)]
pub struct Oids<'a> {
    raw: &'a [u8],
}

impl<'a> Oids<'a> {
    pub(crate) fn validate(
        reader: &mut Reader<'a>,
        field: &'static str,
    ) -> Result<Self, ParseError> {
        let count = reader.take_count16(field)?;
        Ok(Self {
            raw: reader.take_bytes(count * 4)?,
        })
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.raw.len() / 4
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

    #[must_use]
    pub fn iter(&self) -> OidsIter<'a> {
        OidsIter { rest: self.raw }
    }
}

impl<'a> IntoIterator for Oids<'a> {
    type Item = Oid;
    type IntoIter = OidsIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Debug, Clone)]
pub struct OidsIter<'a> {
    rest: &'a [u8],
}

impl Iterator for OidsIter<'_> {
    type Item = Oid;

    fn next(&mut self) -> Option<Self::Item> {
        let (chunk, rest) = self.rest.split_first_chunk::<4>()?;
        self.rest = rest;
        Some(Oid(u32::from_be_bytes(*chunk)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = self.rest.len() / 4;
        (exact, Some(exact))
    }
}

impl ExactSizeIterator for OidsIter<'_> {}
impl FusedIterator for OidsIter<'_> {}

/// Int16 format-code list prefixed by an Int16 count (Bind, FunctionCall,
/// Copy responses). Every entry is validated to be 0 or 1 at parse time.
#[derive(Debug, Clone, Copy)]
pub struct FormatCodes<'a> {
    raw: &'a [u8],
}

impl<'a> FormatCodes<'a> {
    pub(crate) fn validate(
        reader: &mut Reader<'a>,
        field: &'static str,
    ) -> Result<Self, ParseError> {
        let count = reader.take_count16(field)?;
        let tag = reader.tag();
        let raw = reader.take_bytes(count * 2)?;
        for chunk in raw.chunks_exact(2) {
            let code = i16::from_be_bytes([chunk[0], chunk[1]]);
            if FormatCode::from_i16(code).is_none() {
                return Err(ParseError::InvalidValue { tag, field });
            }
        }
        Ok(Self { raw })
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.raw.len() / 2
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }

    /// Resolves the effective format for entry `index` under the Bind
    /// rules: zero codes means all-text, one code applies to every
    /// entry, otherwise one code per entry.
    #[must_use]
    pub fn resolve(&self, index: usize) -> FormatCode {
        match self.len() {
            0 => FormatCode::Text,
            1 => self.iter().next().unwrap_or_default(),
            _ => self.iter().nth(index).unwrap_or_default(),
        }
    }

    #[must_use]
    pub fn iter(&self) -> FormatCodesIter<'a> {
        FormatCodesIter { rest: self.raw }
    }
}

impl<'a> IntoIterator for FormatCodes<'a> {
    type Item = FormatCode;
    type IntoIter = FormatCodesIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Debug, Clone)]
pub struct FormatCodesIter<'a> {
    rest: &'a [u8],
}

impl Iterator for FormatCodesIter<'_> {
    type Item = FormatCode;

    fn next(&mut self) -> Option<Self::Item> {
        let (chunk, rest) = self.rest.split_first_chunk::<2>()?;
        self.rest = rest;
        FormatCode::from_i16(i16::from_be_bytes(*chunk))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = self.rest.len() / 2;
        (exact, Some(exact))
    }
}

impl ExactSizeIterator for FormatCodesIter<'_> {}
impl FusedIterator for FormatCodesIter<'_> {}

/// Length-prefixed value list (Bind parameters, DataRow columns,
/// FunctionCall arguments): Int16 count, then per entry an Int32 byte
/// length (`-1` = NULL) followed by that many bytes.
#[derive(Debug, Clone, Copy)]
pub struct Values<'a> {
    count: usize,
    raw: &'a [u8],
}

impl<'a> Values<'a> {
    pub(crate) fn validate(
        reader: &mut Reader<'a>,
        field: &'static str,
    ) -> Result<Self, ParseError> {
        let count = reader.take_count16(field)?;
        let tag = reader.tag();
        let mark = reader.mark();
        for _ in 0..count {
            let length = reader.take_i32()?;
            match usize::try_from(length) {
                Ok(byte_count) => {
                    reader.take_bytes(byte_count)?;
                }
                Err(_) if length == -1 => {}
                Err(_) => return Err(ParseError::InvalidValue { tag, field }),
            }
        }
        Ok(Self {
            count,
            raw: reader.since(mark),
        })
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[must_use]
    pub fn iter(&self) -> ValuesIter<'a> {
        ValuesIter { rest: self.raw }
    }
}

impl<'a> IntoIterator for Values<'a> {
    type Item = Option<&'a [u8]>;
    type IntoIter = ValuesIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Debug, Clone)]
pub struct ValuesIter<'a> {
    rest: &'a [u8],
}

impl<'a> Iterator for ValuesIter<'a> {
    type Item = Option<&'a [u8]>;

    fn next(&mut self) -> Option<Self::Item> {
        let (chunk, rest) = self.rest.split_first_chunk::<4>()?;
        let length = i32::from_be_bytes(*chunk);
        let Ok(byte_count) = usize::try_from(length) else {
            self.rest = rest;
            return Some(None);
        };
        let value = rest.get(..byte_count)?;
        self.rest = &rest[byte_count..];
        Some(Some(value))
    }
}

impl FusedIterator for ValuesIter<'_> {}

/// One column description within a RowDescription message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldDescription<'a> {
    pub name: PgStr<'a>,
    /// originating table OID, or 0 when not a simple table column
    pub table_oid: u32,
    /// originating column attribute number, or 0
    pub column_attr: i16,
    pub type_oid: Oid,
    /// negative values mean variable-width
    pub type_size: i16,
    pub type_modifier: i32,
    pub format: FormatCode,
}

/// RowDescription field list: Int16 count + per-field descriptor.
#[derive(Debug, Clone, Copy)]
pub struct Fields<'a> {
    count: usize,
    raw: &'a [u8],
}

impl<'a> Fields<'a> {
    pub(crate) fn validate(reader: &mut Reader<'a>) -> Result<Self, ParseError> {
        let count = reader.take_count16("field count")?;
        let tag = reader.tag();
        let mark = reader.mark();
        for _ in 0..count {
            reader.take_cstr()?;
            reader.take_u32()?;
            reader.take_i16()?;
            reader.take_u32()?;
            reader.take_i16()?;
            reader.take_i32()?;
            let format = reader.take_i16()?;
            if FormatCode::from_i16(format).is_none() {
                return Err(ParseError::InvalidValue {
                    tag,
                    field: "field format code",
                });
            }
        }
        Ok(Self {
            count,
            raw: reader.since(mark),
        })
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[must_use]
    pub fn iter(&self) -> FieldsIter<'a> {
        FieldsIter { rest: self.raw }
    }
}

impl<'a> IntoIterator for Fields<'a> {
    type Item = FieldDescription<'a>;
    type IntoIter = FieldsIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Debug, Clone)]
pub struct FieldsIter<'a> {
    rest: &'a [u8],
}

impl<'a> Iterator for FieldsIter<'a> {
    type Item = FieldDescription<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let nul = memchr(0, self.rest)?;
        let name = PgStr::new(&self.rest[..nul]);
        let fixed = self.rest.get(nul + 1..nul + 1 + 18)?;
        let field = FieldDescription {
            name,
            table_oid: u32::from_be_bytes([fixed[0], fixed[1], fixed[2], fixed[3]]),
            column_attr: i16::from_be_bytes([fixed[4], fixed[5]]),
            type_oid: Oid(u32::from_be_bytes([fixed[6], fixed[7], fixed[8], fixed[9]])),
            type_size: i16::from_be_bytes([fixed[10], fixed[11]]),
            type_modifier: i32::from_be_bytes([fixed[12], fixed[13], fixed[14], fixed[15]]),
            format: FormatCode::from_i16(i16::from_be_bytes([fixed[16], fixed[17]]))?,
        };
        self.rest = &self.rest[nul + 1 + 18..];
        Some(field)
    }
}

impl FusedIterator for FieldsIter<'_> {}

/// ErrorResponse / NoticeResponse field pairs: one field-type byte plus
/// a string, repeated, terminated by a zero byte.
#[derive(Debug, Clone, Copy)]
pub struct ErrorFields<'a> {
    raw: &'a [u8],
}

impl<'a> ErrorFields<'a> {
    pub(crate) fn validate(reader: &mut Reader<'a>) -> Result<Self, ParseError> {
        let mark = reader.mark();
        let mut end = mark;
        loop {
            let field_type = reader.take_u8()?;
            if field_type == 0 {
                break;
            }
            reader.take_cstr()?;
            end = reader.mark();
        }
        Ok(Self {
            raw: reader.slice(mark, end),
        })
    }

    /// Linear lookup by field-type byte (see [`super::types::error_field`]).
    #[must_use]
    pub fn get(&self, field_type: u8) -> Option<PgStr<'a>> {
        self.iter()
            .find(|(code, _)| *code == field_type)
            .map(|(_, value)| value)
    }

    #[must_use]
    pub fn iter(&self) -> ErrorFieldsIter<'a> {
        ErrorFieldsIter { rest: self.raw }
    }
}

impl<'a> IntoIterator for ErrorFields<'a> {
    type Item = (u8, PgStr<'a>);
    type IntoIter = ErrorFieldsIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Debug, Clone)]
pub struct ErrorFieldsIter<'a> {
    rest: &'a [u8],
}

impl<'a> Iterator for ErrorFieldsIter<'a> {
    type Item = (u8, PgStr<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        let (&field_type, rest) = self.rest.split_first()?;
        let nul = memchr(0, rest)?;
        self.rest = &rest[nul + 1..];
        Some((field_type, PgStr::new(&rest[..nul])))
    }
}

impl FusedIterator for ErrorFieldsIter<'_> {}

/// NUL-terminated string list ended by one empty string (SASL mechanism
/// names in AuthenticationSASL).
#[derive(Debug, Clone, Copy)]
pub struct CStrList<'a> {
    raw: &'a [u8],
}

impl<'a> CStrList<'a> {
    pub(crate) fn validate(reader: &mut Reader<'a>) -> Result<Self, ParseError> {
        let mark = reader.mark();
        let mut end = mark;
        loop {
            let item = reader.take_cstr()?;
            if item.is_empty() {
                break;
            }
            end = reader.mark();
        }
        Ok(Self {
            raw: reader.slice(mark, end),
        })
    }

    #[must_use]
    pub fn iter(&self) -> CStrListIter<'a> {
        CStrListIter { rest: self.raw }
    }
}

impl<'a> IntoIterator for CStrList<'a> {
    type Item = PgStr<'a>;
    type IntoIter = CStrListIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Debug, Clone)]
pub struct CStrListIter<'a> {
    rest: &'a [u8],
}

impl<'a> Iterator for CStrListIter<'a> {
    type Item = PgStr<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let nul = memchr(0, self.rest)?;
        let item = PgStr::new(&self.rest[..nul]);
        self.rest = &self.rest[nul + 1..];
        Some(item)
    }
}

impl FusedIterator for CStrListIter<'_> {}

/// Int32-counted string list (NegotiateProtocolVersion unrecognized
/// option names).
#[derive(Debug, Clone, Copy)]
pub struct CountedCStrs<'a> {
    count: usize,
    raw: &'a [u8],
}

impl<'a> CountedCStrs<'a> {
    pub(crate) fn validate(
        reader: &mut Reader<'a>,
        field: &'static str,
    ) -> Result<Self, ParseError> {
        let raw_count = reader.take_i32()?;
        let tag = reader.tag();
        let count =
            usize::try_from(raw_count).map_err(|_| ParseError::InvalidValue { tag, field })?;
        let mark = reader.mark();
        for _ in 0..count {
            reader.take_cstr()?;
        }
        Ok(Self {
            count,
            raw: reader.since(mark),
        })
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[must_use]
    pub fn iter(&self) -> CStrListIter<'a> {
        CStrListIter { rest: self.raw }
    }
}

impl<'a> IntoIterator for CountedCStrs<'a> {
    type Item = PgStr<'a>;
    type IntoIter = CStrListIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// the test helpers build Vec<u8> frames; this crate carries no alloc
// dependency for its no_std tier, so the suite needs std, not just test
#[cfg(all(test, feature = "std"))]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use rstest::rstest;

    use super::*;
    use super::super::cursor::Reader;

    fn reader_from(bytes: &[u8]) -> Reader<'_> {
        Reader::new(bytes, 0)
    }

    #[test]
    fn startup_params_round_trip_four_params() {
        let raw = b"user\0alice\0database\0appdb\0application_name\0psql\0\0";
        let mut reader = reader_from(raw);
        let params = StartupParams::validate(&mut reader).expect("validate must succeed");

        assert_eq!(params.get(b"user"), Some(PgStr::new(b"alice")));
        assert_eq!(params.get(b"database"), Some(PgStr::new(b"appdb")));
        assert_eq!(params.get(b"application_name"), Some(PgStr::new(b"psql")));
        assert_eq!(params.get(b"missing"), None);
        assert_eq!(params.iter().count(), 3);
    }

    #[test]
    fn startup_params_empty_param_list() {
        let raw = b"\0";
        let mut reader = reader_from(raw);
        let params = StartupParams::validate(&mut reader).expect("validate must succeed");
        assert_eq!(params.iter().count(), 0);
    }

    #[test]
    fn startup_params_missing_nul_returns_error() {
        let raw = b"user\0alice";
        let mut reader = reader_from(raw);
        let result = StartupParams::validate(&mut reader);
        assert!(result.is_err(), "unterminated value must fail");
    }

    #[test]
    fn oids_empty_list() {
        let raw = &[0u8, 0][..];
        let mut reader = reader_from(raw);
        let oids = Oids::validate(&mut reader, "test").expect("validate must succeed");
        assert!(oids.is_empty());
        assert_eq!(oids.len(), 0);
        assert_eq!(oids.iter().count(), 0);
    }

    #[test]
    fn oids_three_entries_iterable() {
        let mut raw = vec![];
        raw.extend_from_slice(&3i16.to_be_bytes());
        raw.extend_from_slice(&23u32.to_be_bytes());
        raw.extend_from_slice(&25u32.to_be_bytes());
        raw.extend_from_slice(&16u32.to_be_bytes());
        let mut reader = reader_from(&raw);
        let oids = Oids::validate(&mut reader, "test").expect("validate must succeed");
        let collected: Vec<Oid> = oids.iter().collect();
        assert_eq!(collected, vec![Oid(23), Oid(25), Oid(16)]);
        assert_eq!(oids.len(), 3);
    }

    #[test]
    fn oids_truncated_returns_error() {
        let mut raw = vec![];
        raw.extend_from_slice(&2i16.to_be_bytes());
        raw.extend_from_slice(&23u32.to_be_bytes());
        let mut reader = reader_from(&raw);
        let result = Oids::validate(&mut reader, "test");
        assert!(result.is_err(), "truncated oids must fail");
    }

    #[rstest]
    #[case::zero_codes_resolves_text(0usize, FormatCode::Text)]
    #[case::one_code_applies_to_all(0usize, FormatCode::Text)]
    fn format_codes_resolve_zero_codes_is_text(#[case] index: usize, #[case] expected: FormatCode) {
        let raw = &[0u8, 0][..];
        let mut reader = reader_from(raw);
        let codes = FormatCodes::validate(&mut reader, "test").expect("validate must succeed");
        assert_eq!(codes.resolve(index), expected);
        assert_eq!(codes.resolve(99), FormatCode::Text);
    }

    #[test]
    fn format_codes_single_binary_applies_to_all() {
        let mut raw = vec![];
        raw.extend_from_slice(&1i16.to_be_bytes());
        raw.extend_from_slice(&1i16.to_be_bytes());
        let mut reader = reader_from(&raw);
        let codes = FormatCodes::validate(&mut reader, "test").expect("validate must succeed");
        assert_eq!(codes.resolve(0), FormatCode::Binary);
        assert_eq!(codes.resolve(5), FormatCode::Binary);
    }

    #[test]
    fn format_codes_positional_with_two_codes() {
        let mut raw = vec![];
        raw.extend_from_slice(&2i16.to_be_bytes());
        raw.extend_from_slice(&0i16.to_be_bytes());
        raw.extend_from_slice(&1i16.to_be_bytes());
        let mut reader = reader_from(&raw);
        let codes = FormatCodes::validate(&mut reader, "test").expect("validate must succeed");
        assert_eq!(codes.resolve(0), FormatCode::Text);
        assert_eq!(codes.resolve(1), FormatCode::Binary);
    }

    #[test]
    fn format_codes_invalid_code_7_returns_error() {
        let mut raw = vec![];
        raw.extend_from_slice(&1i16.to_be_bytes());
        raw.extend_from_slice(&7i16.to_be_bytes());
        let mut reader = reader_from(&raw);
        let result = FormatCodes::validate(&mut reader, "test");
        assert!(result.is_err(), "format code 7 must fail");
    }

    #[test]
    fn values_with_null_and_data() {
        let mut raw = vec![];
        raw.extend_from_slice(&3i16.to_be_bytes());
        raw.extend_from_slice(&2i32.to_be_bytes());
        raw.extend_from_slice(b"42");
        raw.extend_from_slice(&(-1i32).to_be_bytes());
        raw.extend_from_slice(&5i32.to_be_bytes());
        raw.extend_from_slice(b"hello");
        let mut reader = reader_from(&raw);
        let values = Values::validate(&mut reader, "test").expect("validate must succeed");
        assert_eq!(values.len(), 3);
        let collected: Vec<Option<&[u8]>> = values.iter().collect();
        assert_eq!(collected[0], Some(b"42".as_slice()));
        assert_eq!(collected[1], None);
        assert_eq!(collected[2], Some(b"hello".as_slice()));
    }

    #[test]
    fn values_empty_list() {
        let raw = &[0u8, 0][..];
        let mut reader = reader_from(raw);
        let values = Values::validate(&mut reader, "test").expect("validate must succeed");
        assert!(values.is_empty());
        assert_eq!(values.iter().count(), 0);
    }

    #[test]
    fn values_truncated_body_returns_error() {
        let mut raw = vec![];
        raw.extend_from_slice(&1i16.to_be_bytes());
        raw.extend_from_slice(&10i32.to_be_bytes());
        raw.extend_from_slice(b"short");
        let mut reader = reader_from(&raw);
        let result = Values::validate(&mut reader, "test");
        assert!(result.is_err(), "truncated value must fail");
    }

    #[test]
    fn error_fields_get_and_iter() {
        let raw = b"Slocalized_severity\0MERROR_MESSAGE\0\0";
        let mut reader = reader_from(raw);
        let fields = ErrorFields::validate(&mut reader).expect("validate must succeed");
        let severity = fields.get(b'S');
        assert_eq!(severity, Some(PgStr::new(b"localized_severity")));
        assert_eq!(fields.iter().count(), 2);
    }

    #[test]
    fn error_fields_missing_terminator_returns_error() {
        let raw = b"Sno_nul_at_end";
        let mut reader = reader_from(raw);
        let result = ErrorFields::validate(&mut reader);
        assert!(result.is_err(), "missing nul must fail");
    }

    #[test]
    fn cstr_list_two_mechanisms() {
        let raw = b"SCRAM-SHA-256\0SCRAM-SHA-256-PLUS\0\0";
        let mut reader = reader_from(raw);
        let list = CStrList::validate(&mut reader).expect("validate must succeed");
        let items: Vec<PgStr<'_>> = list.iter().collect();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], "SCRAM-SHA-256");
        assert_eq!(items[1], "SCRAM-SHA-256-PLUS");
    }

    #[test]
    fn cstr_list_empty_immediately_terminated() {
        let raw = b"\0";
        let mut reader = reader_from(raw);
        let list = CStrList::validate(&mut reader).expect("validate must succeed");
        assert_eq!(list.iter().count(), 0);
    }

    #[test]
    fn counted_cstrs_two_options() {
        let mut raw = vec![];
        raw.extend_from_slice(&2i32.to_be_bytes());
        raw.extend_from_slice(b"_pq_.option_a\0");
        raw.extend_from_slice(b"_pq_.option_b\0");
        let mut reader = reader_from(&raw);
        let cstrs = CountedCStrs::validate(&mut reader, "test").expect("validate must succeed");
        assert_eq!(cstrs.len(), 2);
        let items: Vec<PgStr<'_>> = cstrs.iter().collect();
        assert_eq!(items[0], "_pq_.option_a");
        assert_eq!(items[1], "_pq_.option_b");
    }

    #[test]
    fn counted_cstrs_empty_list() {
        let raw = &0i32.to_be_bytes();
        let mut reader = reader_from(raw);
        let cstrs = CountedCStrs::validate(&mut reader, "test").expect("validate must succeed");
        assert!(cstrs.is_empty());
    }

    #[test]
    fn fields_valid_two_column_row_description() {
        let mut raw = vec![];
        raw.extend_from_slice(&2i16.to_be_bytes());
        for name in [b"id".as_slice(), b"email".as_slice()] {
            raw.extend_from_slice(name);
            raw.push(0);
            raw.extend_from_slice(&0u32.to_be_bytes());
            raw.extend_from_slice(&1i16.to_be_bytes());
            raw.extend_from_slice(&23u32.to_be_bytes());
            raw.extend_from_slice(&4i16.to_be_bytes());
            raw.extend_from_slice(&(-1i32).to_be_bytes());
            raw.extend_from_slice(&0i16.to_be_bytes());
        }
        let mut reader = reader_from(&raw);
        let fields = Fields::validate(&mut reader).expect("validate must succeed");
        assert_eq!(fields.len(), 2);
        let descs: Vec<FieldDescription<'_>> = fields.iter().collect();
        assert_eq!(descs[0].name, "id");
        assert_eq!(descs[1].name, "email");
    }

    #[test]
    fn fields_invalid_format_code_returns_error() {
        let mut raw = vec![];
        raw.extend_from_slice(&1i16.to_be_bytes());
        raw.extend_from_slice(b"col\0");
        raw.extend_from_slice(&0u32.to_be_bytes());
        raw.extend_from_slice(&1i16.to_be_bytes());
        raw.extend_from_slice(&23u32.to_be_bytes());
        raw.extend_from_slice(&4i16.to_be_bytes());
        raw.extend_from_slice(&(-1i32).to_be_bytes());
        raw.extend_from_slice(&7i16.to_be_bytes());
        let mut reader = reader_from(&raw);
        let result = Fields::validate(&mut reader);
        assert!(result.is_err(), "format code 7 in field must fail");
    }
}
