//! `proxima_codec::WireCodec` impl for the protobuf wire format.
//!
//! protobuf is the canonical example of a wire-level field iterator:
//! a message is a sequence of (tag, value) pairs, parsed one at a
//! time, with the caller deciding when to stop. the existing
//! [`crate::Fields`] iterator already yields `Result<Field<'a>,
//! ParseError>` — `WireCodec::iter_fields` matches that shape
//! exactly.

use proxima_codec::WireCodec;

use super::{Field, Fields, ParseError, parse_field};

/// protobuf wire-format [`WireCodec`]. Zero-sized; clone freely.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProtobufWireCodec;

impl WireCodec for ProtobufWireCodec {
    type Field<'a> = Field<'a>;
    type Error = ParseError;

    fn parse_field<'a>(&self, buf: &'a [u8]) -> Result<(Self::Field<'a>, usize), Self::Error> {
        parse_field(buf)
    }

    fn iter_fields<'a>(
        &self,
        buf: &'a [u8],
    ) -> impl Iterator<Item = Result<Self::Field<'a>, Self::Error>> {
        Fields::new(buf)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use super::super::encode_varint;
    use proxima_codec::WireCodec;

    fn encode_tag(field: u32, wire: u8) -> Vec<u8> {
        let mut out = Vec::new();
        encode_varint(u64::from((field << 3) | u32::from(wire)), &mut out);
        out
    }

    fn build_message_with_two_varints() -> Vec<u8> {
        // field 1 (varint, wire 0): value 42
        // field 2 (varint, wire 0): value 7
        let mut buf = encode_tag(1, 0);
        encode_varint(42, &mut buf);
        buf.extend(encode_tag(2, 0));
        encode_varint(7, &mut buf);
        buf
    }

    #[test]
    fn parse_field_returns_first_field_and_consumed() {
        let codec = ProtobufWireCodec;
        let buf = build_message_with_two_varints();
        let (field, consumed) = codec.parse_field(&buf).expect("first field");
        assert!(matches!(
            field,
            Field::Varint {
                field: 1,
                value: 42
            }
        ));
        assert!(consumed < buf.len(), "must leave bytes for second field");
    }

    #[test]
    fn iter_fields_yields_each_field_in_order() {
        let codec = ProtobufWireCodec;
        let buf = build_message_with_two_varints();
        let yielded: Vec<Field<'_>> = codec
            .iter_fields(&buf)
            .map(|result| result.expect("ok"))
            .collect();
        assert_eq!(yielded.len(), 2);
        assert!(matches!(
            yielded[0],
            Field::Varint {
                field: 1,
                value: 42
            }
        ));
        assert!(matches!(yielded[1], Field::Varint { field: 2, value: 7 }));
    }

    #[test]
    fn iter_fields_stops_at_end_of_buffer() {
        let codec = ProtobufWireCodec;
        let count = codec.iter_fields(b"").count();
        assert_eq!(count, 0);
    }
}
