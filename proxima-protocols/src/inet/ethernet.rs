use super::error::DecodeError;

/// EtherType of the payload that follows the 14-byte Ethernet II header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EtherType {
    Ipv4,
    Ipv6,
    Arp,
    Other(u16),
}

impl EtherType {
    #[must_use]
    pub const fn from_u16(value: u16) -> Self {
        match value {
            0x0800 => Self::Ipv4,
            0x86dd => Self::Ipv6,
            0x0806 => Self::Arp,
            other => Self::Other(other),
        }
    }

    #[must_use]
    pub const fn as_u16(self) -> u16 {
        match self {
            Self::Ipv4 => 0x0800,
            Self::Ipv6 => 0x86dd,
            Self::Arp => 0x0806,
            Self::Other(value) => value,
        }
    }
}

/// The 14-byte fixed header: 6-byte destination MAC, 6-byte source MAC,
/// 2-byte EtherType. 802.1Q VLAN tags are not parsed here.
pub const HEADER_LEN: usize = 14;

/// Borrowed view over an Ethernet II frame in a caller buffer.
#[derive(Debug, Clone, Copy)]
pub struct EthernetFrame<'frame> {
    bytes: &'frame [u8],
}

impl<'frame> EthernetFrame<'frame> {
    /// Borrow a frame view. Errors if the buffer cannot hold the fixed header.
    pub fn parse(bytes: &'frame [u8]) -> Result<Self, DecodeError> {
        if bytes.len() < HEADER_LEN {
            return Err(DecodeError::Truncated {
                need: HEADER_LEN,
                got: bytes.len(),
            });
        }
        Ok(Self { bytes })
    }

    #[must_use]
    pub fn destination(&self) -> [u8; 6] {
        let mut mac = [0u8; 6];
        mac.copy_from_slice(&self.bytes[0..6]);
        mac
    }

    #[must_use]
    pub fn source(&self) -> [u8; 6] {
        let mut mac = [0u8; 6];
        mac.copy_from_slice(&self.bytes[6..12]);
        mac
    }

    #[must_use]
    pub fn ether_type(&self) -> EtherType {
        EtherType::from_u16(u16::from_be_bytes([self.bytes[12], self.bytes[13]]))
    }

    /// Everything past the fixed header.
    #[must_use]
    pub fn payload(&self) -> &'frame [u8] {
        &self.bytes[HEADER_LEN..]
    }
}

/// Write a 14-byte Ethernet II header into the front of `out`. Returns the
/// payload start offset. Errors if `out` is too short.
pub fn write_header(
    out: &mut [u8],
    destination: [u8; 6],
    source: [u8; 6],
    ether_type: EtherType,
) -> Result<usize, DecodeError> {
    if out.len() < HEADER_LEN {
        return Err(DecodeError::Truncated {
            need: HEADER_LEN,
            got: out.len(),
        });
    }
    out[0..6].copy_from_slice(&destination);
    out[6..12].copy_from_slice(&source);
    out[12..14].copy_from_slice(&ether_type.as_u16().to_be_bytes());
    Ok(HEADER_LEN)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use proptest::prelude::*;

    const DST: [u8; 6] = [0x00, 0x11, 0x22, 0x33, 0x44, 0x55];
    const SRC: [u8; 6] = [0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb];

    #[test]
    fn parse_extracts_fields_and_payload() {
        let frame = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0x08, 0x00,
            0xde, 0xad,
        ];
        let view = EthernetFrame::parse(&frame).expect("fixed header fits");
        assert_eq!(view.destination(), DST);
        assert_eq!(view.source(), SRC);
        assert_eq!(view.ether_type(), EtherType::Ipv4);
        assert_eq!(view.payload(), &[0xde, 0xad]);
    }

    #[test]
    fn truncated_buffer_is_rejected() {
        assert_eq!(
            EthernetFrame::parse(&[0u8; 13]).unwrap_err(),
            DecodeError::Truncated { need: 14, got: 13 }
        );
    }

    #[test]
    fn write_then_parse_round_trips() {
        let mut out = [0u8; HEADER_LEN];
        let offset = write_header(&mut out, DST, SRC, EtherType::Arp).expect("buffer fits");
        assert_eq!(offset, HEADER_LEN);
        let view = EthernetFrame::parse(&out).expect("written header parses");
        assert_eq!(view.destination(), DST);
        assert_eq!(view.source(), SRC);
        assert_eq!(view.ether_type(), EtherType::Arp);
    }

    proptest! {
        // Parser must never panic regardless of what bytes arrive off the wire.
        #[test]
        fn parse_never_panics_on_arbitrary_bytes(
            data in prop::collection::vec(any::<u8>(), 0..128),
        ) {
            let _ = EthernetFrame::parse(&data);
        }

        // EtherType round-trips: any u16 encodes and decodes to itself.
        #[test]
        fn ether_type_u16_round_trips(value in any::<u16>()) {
            prop_assert_eq!(EtherType::from_u16(value).as_u16(), value);
        }

        // Write followed by parse must recover the exact dst MAC, src MAC, and
        // EtherType that were passed to write_header.
        #[test]
        fn write_then_parse_recovers_arbitrary_fields(
            dst in prop::array::uniform6(any::<u8>()),
            src in prop::array::uniform6(any::<u8>()),
            ether_type_raw in any::<u16>(),
        ) {
            let ether_type = EtherType::from_u16(ether_type_raw);
            let mut buf = [0u8; HEADER_LEN + 4];

            let offset = write_header(&mut buf, dst, src, ether_type)
                .expect("HEADER_LEN + 4 >= HEADER_LEN");

            prop_assert_eq!(offset, HEADER_LEN);
            let view = EthernetFrame::parse(&buf).expect("written header must parse");
            prop_assert_eq!(view.destination(), dst);
            prop_assert_eq!(view.source(), src);
            prop_assert_eq!(view.ether_type(), ether_type);
        }
    }
}
