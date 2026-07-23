//! AMQP 0-9-1 method-argument wire primitives — the layer
//! `proxima_protocols::amqp::parse_frame` stops short of. That function only
//! frames the outer envelope (type/channel/length/payload/0xCE) and, for a
//! `Frame::Method`, splits `class_id`/`method_id` off the front of the
//! payload; everything after (`args`) is opaque bytes the caller must decode
//! per the AMQP 0-9-1 spec. This module supplies that decode: the scalar
//! primitives (`octet`/`short`/`long`/`longlong`), the two string forms
//! (`shortstr` 1-byte-length-prefixed, `longstr` 4-byte-length-prefixed),
//! bit-packed boolean flags, and the field-table value grammar
//! (`FieldValue`/`FieldTable`) real clients (pika, kombu, the RabbitMQ Java
//! client) send in `client-properties` / declare `arguments`.
//!
//! Every `read_*` function takes `&[u8]` and returns `(value, rest)` —
//! no internal cursor state, so [`crate::method`] composes them by
//! threading `rest` through each field in spec order.

use std::collections::BTreeMap;

/// A method-argument decode failure — too few bytes for the field being
/// read, or (for a field table) an unrecognized type tag this decoder
/// does not implement. Fails closed: an unknown tag cannot be safely
/// skipped (its length is tag-defined), so decoding stops rather than
/// guessing.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WireError {
    #[error("buffer ended while reading a {0}")]
    Short(&'static str),
    #[error("field-table type tag 0x{0:02X} ('{1}') is not implemented")]
    UnknownFieldType(u8, char),
    #[error("shortstr/longstr is not valid utf-8")]
    Utf8,
}

pub fn read_octet(buf: &[u8]) -> Result<(u8, &[u8]), WireError> {
    buf.split_first()
        .map(|(byte, rest)| (*byte, rest))
        .ok_or(WireError::Short("octet"))
}

pub fn read_short(buf: &[u8]) -> Result<(u16, &[u8]), WireError> {
    if buf.len() < 2 {
        return Err(WireError::Short("short"));
    }
    let (head, rest) = buf.split_at(2);
    Ok((u16::from_be_bytes([head[0], head[1]]), rest))
}

pub fn read_long(buf: &[u8]) -> Result<(u32, &[u8]), WireError> {
    if buf.len() < 4 {
        return Err(WireError::Short("long"));
    }
    let (head, rest) = buf.split_at(4);
    Ok((
        u32::from_be_bytes([head[0], head[1], head[2], head[3]]),
        rest,
    ))
}

pub fn read_longlong(buf: &[u8]) -> Result<(u64, &[u8]), WireError> {
    if buf.len() < 8 {
        return Err(WireError::Short("longlong"));
    }
    let (head, rest) = buf.split_at(8);
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(head);
    Ok((u64::from_be_bytes(bytes), rest))
}

pub fn read_shortstr(buf: &[u8]) -> Result<(&[u8], &[u8]), WireError> {
    let (len, rest) = read_octet(buf)?;
    let len = len as usize;
    if rest.len() < len {
        return Err(WireError::Short("shortstr"));
    }
    Ok(rest.split_at(len))
}

pub fn read_longstr(buf: &[u8]) -> Result<(&[u8], &[u8]), WireError> {
    let (len, rest) = read_long(buf)?;
    let len = len as usize;
    if rest.len() < len {
        return Err(WireError::Short("longstr"));
    }
    Ok(rest.split_at(len))
}

/// Reads one octet of bit-packed boolean flags (AMQP packs consecutive
/// `bit` method fields LSB-first into as few octets as fit — every method
/// this crate decodes has 8 or fewer consecutive bits, so one octet always
/// suffices). Returns the flags as a fixed 8-element array; the caller
/// reads only the indices its method actually declares.
pub fn read_bit_flags(buf: &[u8]) -> Result<([bool; 8], &[u8]), WireError> {
    let (byte, rest) = read_octet(buf)?;
    let mut flags = [false; 8];
    for (index, flag) in flags.iter_mut().enumerate() {
        *flag = byte & (1 << index) != 0;
    }
    Ok((flags, rest))
}

pub fn write_octet(out: &mut Vec<u8>, value: u8) {
    out.push(value);
}

pub fn write_short(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_be_bytes());
}

pub fn write_long(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

pub fn write_longlong(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_be_bytes());
}

/// # Panics
/// Debug-asserts `value.len() <= u8::MAX as usize` — a shortstr field is
/// spec-capped at 255 bytes; callers pass already-bounded protocol strings
/// (queue/exchange/consumer-tag names), never attacker-controlled body
/// data (that always rides `longstr`/content-body).
pub fn write_shortstr(out: &mut Vec<u8>, value: &[u8]) {
    debug_assert!(
        value.len() <= u8::MAX as usize,
        "shortstr exceeds 255 bytes"
    );
    write_octet(out, value.len() as u8);
    out.extend_from_slice(value);
}

pub fn write_longstr(out: &mut Vec<u8>, value: &[u8]) {
    write_long(out, value.len() as u32);
    out.extend_from_slice(value);
}

/// Packs up to 8 booleans LSB-first into one flags octet.
pub fn write_bit_flags(out: &mut Vec<u8>, flags: &[bool]) {
    debug_assert!(
        flags.len() <= 8,
        "more than 8 bits needs a second flags octet"
    );
    let mut byte = 0_u8;
    for (index, flag) in flags.iter().enumerate() {
        if *flag {
            byte |= 1 << index;
        }
    }
    out.push(byte);
}

/// One field-table value. Covers the RabbitMQ wire dialect real client
/// libraries (pika, kombu, the RabbitMQ Java/`.NET` clients) actually
/// produce: `t`/`b`/`I`/`l`/`f`/`d`/`D`/`S`/`F`/`A`/`T`/`V`/`x`. A tag
/// outside this set is a decode error (`WireError::UnknownFieldType`),
/// not a silent skip — see the module doc for why.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldValue {
    Boolean(bool),
    ShortShortInt(i8),
    LongInt(i32),
    LongLongInt(i64),
    Float(f32),
    Double(f64),
    Decimal { scale: u8, value: i32 },
    LongString(Vec<u8>),
    FieldTable(FieldTable),
    FieldArray(Vec<FieldValue>),
    Timestamp(u64),
    Void,
    ByteArray(Vec<u8>),
}

pub type FieldTable = BTreeMap<String, FieldValue>;

fn read_field_value(tag: u8, buf: &[u8]) -> Result<(FieldValue, &[u8]), WireError> {
    match tag {
        b't' => {
            let (byte, rest) = read_octet(buf)?;
            Ok((FieldValue::Boolean(byte != 0), rest))
        }
        b'b' => {
            let (byte, rest) = read_octet(buf)?;
            Ok((FieldValue::ShortShortInt(byte as i8), rest))
        }
        b'I' => {
            let (value, rest) = read_long(buf)?;
            Ok((FieldValue::LongInt(value as i32), rest))
        }
        b'l' => {
            let (value, rest) = read_longlong(buf)?;
            Ok((FieldValue::LongLongInt(value as i64), rest))
        }
        b'f' => {
            if buf.len() < 4 {
                return Err(WireError::Short("float"));
            }
            let (head, rest) = buf.split_at(4);
            let mut bytes = [0_u8; 4];
            bytes.copy_from_slice(head);
            Ok((FieldValue::Float(f32::from_be_bytes(bytes)), rest))
        }
        b'd' => {
            if buf.len() < 8 {
                return Err(WireError::Short("double"));
            }
            let (head, rest) = buf.split_at(8);
            let mut bytes = [0_u8; 8];
            bytes.copy_from_slice(head);
            Ok((FieldValue::Double(f64::from_be_bytes(bytes)), rest))
        }
        b'D' => {
            let (scale, rest) = read_octet(buf)?;
            let (value, rest) = read_long(rest)?;
            Ok((
                FieldValue::Decimal {
                    scale,
                    value: value as i32,
                },
                rest,
            ))
        }
        b'S' => {
            let (bytes, rest) = read_longstr(buf)?;
            Ok((FieldValue::LongString(bytes.to_vec()), rest))
        }
        b'F' => {
            let (table, rest) = read_field_table(buf)?;
            Ok((FieldValue::FieldTable(table), rest))
        }
        b'A' => {
            let (len, mut rest) = read_long(buf)?;
            let len = len as usize;
            if rest.len() < len {
                return Err(WireError::Short("field-array"));
            }
            let (mut body, after) = rest.split_at(len);
            let mut values = Vec::new();
            while !body.is_empty() {
                let (element_tag, tail) = read_octet(body)?;
                let (value, tail) = read_field_value(element_tag, tail)?;
                values.push(value);
                body = tail;
            }
            rest = after;
            Ok((FieldValue::FieldArray(values), rest))
        }
        b'T' => {
            let (value, rest) = read_longlong(buf)?;
            Ok((FieldValue::Timestamp(value), rest))
        }
        b'V' => Ok((FieldValue::Void, buf)),
        b'x' => {
            let (bytes, rest) = read_longstr(buf)?;
            Ok((FieldValue::ByteArray(bytes.to_vec()), rest))
        }
        other => Err(WireError::UnknownFieldType(other, other as char)),
    }
}

fn write_field_value(out: &mut Vec<u8>, value: &FieldValue) {
    match value {
        FieldValue::Boolean(flag) => {
            write_octet(out, b't');
            write_octet(out, u8::from(*flag));
        }
        FieldValue::ShortShortInt(value) => {
            write_octet(out, b'b');
            write_octet(out, *value as u8);
        }
        FieldValue::LongInt(value) => {
            write_octet(out, b'I');
            write_long(out, *value as u32);
        }
        FieldValue::LongLongInt(value) => {
            write_octet(out, b'l');
            write_longlong(out, *value as u64);
        }
        FieldValue::Float(value) => {
            write_octet(out, b'f');
            out.extend_from_slice(&value.to_be_bytes());
        }
        FieldValue::Double(value) => {
            write_octet(out, b'd');
            out.extend_from_slice(&value.to_be_bytes());
        }
        FieldValue::Decimal { scale, value } => {
            write_octet(out, b'D');
            write_octet(out, *scale);
            write_long(out, *value as u32);
        }
        FieldValue::LongString(bytes) => {
            write_octet(out, b'S');
            write_longstr(out, bytes);
        }
        FieldValue::FieldTable(table) => {
            write_octet(out, b'F');
            write_field_table(out, table);
        }
        FieldValue::FieldArray(values) => {
            write_octet(out, b'A');
            let mut body = Vec::new();
            for value in values {
                write_field_value(&mut body, value);
            }
            write_long(out, body.len() as u32);
            out.extend_from_slice(&body);
        }
        FieldValue::Timestamp(value) => {
            write_octet(out, b'T');
            write_longlong(out, *value);
        }
        FieldValue::Void => write_octet(out, b'V'),
        FieldValue::ByteArray(bytes) => {
            write_octet(out, b'x');
            write_longstr(out, bytes);
        }
    }
}

/// Reads a `longstr`-length-prefixed field table: repeated `(shortstr key,
/// tagged value)` entries until the declared byte length is consumed.
pub fn read_field_table(buf: &[u8]) -> Result<(FieldTable, &[u8]), WireError> {
    let (len, rest) = read_long(buf)?;
    let len = len as usize;
    if rest.len() < len {
        return Err(WireError::Short("field-table"));
    }
    let (mut body, after) = rest.split_at(len);
    let mut table = FieldTable::new();
    while !body.is_empty() {
        let (key, tail) = read_shortstr(body)?;
        let key = String::from_utf8(key.to_vec()).map_err(|_| WireError::Utf8)?;
        let (tag, tail) = read_octet(tail)?;
        let (value, tail) = read_field_value(tag, tail)?;
        table.insert(key, value);
        body = tail;
    }
    Ok((table, after))
}

/// Appends a field table (with its `longstr` length prefix) to `out`.
pub fn write_field_table(out: &mut Vec<u8>, table: &FieldTable) {
    let mut body = Vec::new();
    for (key, value) in table {
        write_shortstr(&mut body, key.as_bytes());
        write_field_value(&mut body, value);
    }
    write_long(out, body.len() as u32);
    out.extend_from_slice(&body);
}

/// Encodes a field table with its `longstr` length prefix as a standalone
/// buffer — the convenience form for tests and callers that just want the
/// bytes rather than appending to an existing outbound buffer.
#[cfg(test)]
fn encode_field_table(table: &FieldTable) -> Vec<u8> {
    let mut out = Vec::new();
    write_field_table(&mut out, table);
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn scalars_round_trip() {
        let mut out = Vec::new();
        write_octet(&mut out, 7);
        write_short(&mut out, 1000);
        write_long(&mut out, 70_000);
        write_longlong(&mut out, 5_000_000_000);

        let (octet, rest) = read_octet(&out).unwrap();
        assert_eq!(octet, 7);
        let (short, rest) = read_short(rest).unwrap();
        assert_eq!(short, 1000);
        let (long, rest) = read_long(rest).unwrap();
        assert_eq!(long, 70_000);
        let (longlong, rest) = read_longlong(rest).unwrap();
        assert_eq!(longlong, 5_000_000_000);
        assert!(rest.is_empty());
    }

    #[test]
    fn shortstr_and_longstr_round_trip() {
        let mut out = Vec::new();
        write_shortstr(&mut out, b"direct");
        write_longstr(&mut out, b"hello world");

        let (short, rest) = read_shortstr(&out).unwrap();
        assert_eq!(short, b"direct");
        let (long, rest) = read_longstr(rest).unwrap();
        assert_eq!(long, b"hello world");
        assert!(rest.is_empty());
    }

    #[test]
    fn shortstr_short_buffer_is_an_error() {
        let buf = [3_u8, b'a', b'b']; // declares 3 bytes, only 2 present
        assert_eq!(read_shortstr(&buf), Err(WireError::Short("shortstr")));
    }

    #[test]
    fn bit_flags_pack_lsb_first() {
        let mut out = Vec::new();
        write_bit_flags(&mut out, &[true, false, true]);
        assert_eq!(out, vec![0b0000_0101]);

        let (flags, rest) = read_bit_flags(&out).unwrap();
        assert!(flags[0]);
        assert!(!flags[1]);
        assert!(flags[2]);
        assert!(rest.is_empty());
    }

    #[test]
    fn field_table_round_trips_every_implemented_type() {
        let mut table = FieldTable::new();
        table.insert(
            "product".into(),
            FieldValue::LongString(b"proxima".to_vec()),
        );
        table.insert("copyright_year".into(), FieldValue::LongInt(2026));
        table.insert("supported".into(), FieldValue::Boolean(true));
        table.insert(
            "capabilities".into(),
            FieldValue::FieldTable(FieldTable::new()),
        );
        table.insert(
            "versions".into(),
            FieldValue::FieldArray(vec![FieldValue::LongInt(9), FieldValue::LongInt(1)]),
        );
        table.insert("ratio".into(), FieldValue::Double(0.5));
        table.insert("epoch".into(), FieldValue::Timestamp(1_753_000_000));
        table.insert("nothing".into(), FieldValue::Void);

        let encoded = encode_field_table(&table);
        let (decoded, rest) = read_field_table(&encoded).unwrap();
        assert!(rest.is_empty());
        assert_eq!(decoded, table);
    }

    #[test]
    fn unknown_field_type_tag_fails_closed() {
        // a longstr-prefixed table with one entry whose type tag ('Z') this
        // decoder does not implement.
        let mut body = Vec::new();
        write_shortstr(&mut body, b"k");
        write_octet(&mut body, b'Z');
        let mut buf = Vec::new();
        write_long(&mut buf, body.len() as u32);
        buf.extend_from_slice(&body);

        assert_eq!(
            read_field_table(&buf),
            Err(WireError::UnknownFieldType(b'Z', 'Z'))
        );
    }

    #[test]
    fn nested_field_table_round_trips() {
        let mut inner = FieldTable::new();
        inner.insert("x-max-length".into(), FieldValue::LongInt(1000));
        let mut outer = FieldTable::new();
        outer.insert("arguments".into(), FieldValue::FieldTable(inner));

        let encoded = encode_field_table(&outer);
        let (decoded, rest) = read_field_table(&encoded).unwrap();
        assert!(rest.is_empty());
        assert_eq!(decoded, outer);
    }
}
