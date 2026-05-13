use super::checksum::Checksum;
use super::error::DecodeError;
use super::ipv4::{self, Ipv4Protocol};

/// TCP control bits (byte 13 of the header).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TcpFlags {
    pub fin: bool,
    pub syn: bool,
    pub rst: bool,
    pub psh: bool,
    pub ack: bool,
    pub urg: bool,
}

impl TcpFlags {
    #[must_use]
    pub const fn from_u8(bits: u8) -> Self {
        Self {
            fin: bits & 0x01 != 0,
            syn: bits & 0x02 != 0,
            rst: bits & 0x04 != 0,
            psh: bits & 0x08 != 0,
            ack: bits & 0x10 != 0,
            urg: bits & 0x20 != 0,
        }
    }

    #[must_use]
    pub const fn as_u8(self) -> u8 {
        (self.fin as u8)
            | (self.syn as u8) << 1
            | (self.rst as u8) << 2
            | (self.psh as u8) << 3
            | (self.ack as u8) << 4
            | (self.urg as u8) << 5
    }
}

/// Minimum TCP header with no options (data offset = 5 words).
pub const MIN_HEADER_LEN: usize = 20;

/// Borrowed view over a TCP header in a caller buffer.
#[derive(Debug, Clone, Copy)]
pub struct TcpHeader<'segment> {
    bytes: &'segment [u8],
}

impl<'segment> TcpHeader<'segment> {
    /// Borrow a header view. Validates the data-offset field and that the
    /// declared header length fits the buffer.
    pub fn parse(bytes: &'segment [u8]) -> Result<Self, DecodeError> {
        if bytes.len() < MIN_HEADER_LEN {
            return Err(DecodeError::Truncated {
                need: MIN_HEADER_LEN,
                got: bytes.len(),
            });
        }
        let data_offset = bytes[12] >> 4;
        if data_offset < 5 {
            return Err(DecodeError::BadHeaderLen { field: data_offset });
        }
        let header_len = usize::from(data_offset) * 4;
        if bytes.len() < header_len {
            return Err(DecodeError::Truncated {
                need: header_len,
                got: bytes.len(),
            });
        }
        Ok(Self { bytes })
    }

    #[must_use]
    pub fn source_port(&self) -> u16 {
        u16::from_be_bytes([self.bytes[0], self.bytes[1]])
    }

    #[must_use]
    pub fn destination_port(&self) -> u16 {
        u16::from_be_bytes([self.bytes[2], self.bytes[3]])
    }

    #[must_use]
    pub fn sequence(&self) -> u32 {
        u32::from_be_bytes([self.bytes[4], self.bytes[5], self.bytes[6], self.bytes[7]])
    }

    #[must_use]
    pub fn acknowledgement(&self) -> u32 {
        u32::from_be_bytes([self.bytes[8], self.bytes[9], self.bytes[10], self.bytes[11]])
    }

    #[must_use]
    pub fn header_len(&self) -> usize {
        usize::from(self.bytes[12] >> 4) * 4
    }

    #[must_use]
    pub fn flags(&self) -> TcpFlags {
        TcpFlags::from_u8(self.bytes[13])
    }

    #[must_use]
    pub fn window(&self) -> u16 {
        u16::from_be_bytes([self.bytes[14], self.bytes[15]])
    }

    #[must_use]
    pub fn checksum(&self) -> u16 {
        u16::from_be_bytes([self.bytes[16], self.bytes[17]])
    }

    #[must_use]
    pub fn payload(&self) -> &'segment [u8] {
        &self.bytes[self.header_len()..]
    }

    /// Recompute the checksum over the IPv4 pseudo-header + segment and
    /// compare to the stored field (a valid segment folds to zero).
    #[must_use]
    pub fn checksum_valid(&self, source: [u8; 4], destination: [u8; 4]) -> bool {
        let mut accumulator = Checksum::new();
        let l4_len = self.bytes.len() as u16;
        ipv4::pseudo_header_sum(
            &mut accumulator,
            source,
            destination,
            Ipv4Protocol::Tcp,
            l4_len,
        );
        accumulator.add_bytes(self.bytes);
        accumulator.finish() == 0
    }
}

/// Write a 20-byte TCP header (no options) into the front of `out`, then
/// compute the checksum over the pseudo-header + header + `payload`. The
/// payload must already be placed at `out[20..20 + payload.len()]` by the
/// caller; it is passed here only to fold into the checksum.
#[allow(clippy::too_many_arguments)]
pub fn write_header(
    out: &mut [u8],
    source_ip: [u8; 4],
    destination_ip: [u8; 4],
    source_port: u16,
    destination_port: u16,
    sequence: u32,
    acknowledgement: u32,
    flags: TcpFlags,
    window: u16,
    payload: &[u8],
) -> Result<usize, DecodeError> {
    if out.len() < MIN_HEADER_LEN {
        return Err(DecodeError::Truncated {
            need: MIN_HEADER_LEN,
            got: out.len(),
        });
    }
    let header = &mut out[..MIN_HEADER_LEN];
    header.fill(0);
    header[0..2].copy_from_slice(&source_port.to_be_bytes());
    header[2..4].copy_from_slice(&destination_port.to_be_bytes());
    header[4..8].copy_from_slice(&sequence.to_be_bytes());
    header[8..12].copy_from_slice(&acknowledgement.to_be_bytes());
    header[12] = 5 << 4;
    header[13] = flags.as_u8();
    header[14..16].copy_from_slice(&window.to_be_bytes());
    let l4_len = MIN_HEADER_LEN as u16 + payload.len() as u16;
    let sum = {
        let mut accumulator = Checksum::new();
        ipv4::pseudo_header_sum(
            &mut accumulator,
            source_ip,
            destination_ip,
            Ipv4Protocol::Tcp,
            l4_len,
        );
        accumulator.add_bytes(header);
        accumulator.add_bytes(payload);
        accumulator.finish()
    };
    header[16..18].copy_from_slice(&sum.to_be_bytes());
    Ok(MIN_HEADER_LEN)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use proptest::prelude::*;

    const SRC_IP: [u8; 4] = [10, 0, 0, 1];
    const DST_IP: [u8; 4] = [10, 0, 0, 2];

    #[test]
    fn flags_round_trip_through_byte() {
        let flags = TcpFlags {
            syn: true,
            ack: true,
            ..Default::default()
        };
        assert_eq!(flags.as_u8(), 0x12);
        assert_eq!(TcpFlags::from_u8(0x12), flags);
    }

    #[test]
    fn write_then_parse_with_valid_checksum() {
        let payload = b"hello tcp";
        let mut out = [0u8; MIN_HEADER_LEN + 9];
        out[MIN_HEADER_LEN..].copy_from_slice(payload);
        let flags = TcpFlags {
            syn: true,
            ..Default::default()
        };
        write_header(
            &mut out,
            SRC_IP,
            DST_IP,
            0xabcd,
            0x0050,
            0x1000_0000,
            0,
            flags,
            0xffff,
            payload,
        )
        .expect("buffer fits");
        let header = TcpHeader::parse(&out).expect("written header parses");
        assert_eq!(header.source_port(), 0xabcd);
        assert_eq!(header.destination_port(), 0x0050);
        assert_eq!(header.sequence(), 0x1000_0000);
        assert_eq!(header.flags(), flags);
        assert_eq!(header.window(), 0xffff);
        assert_eq!(header.payload(), payload);
        assert!(header.checksum_valid(SRC_IP, DST_IP));
    }

    #[test]
    fn rejects_short_data_offset() {
        let mut bytes = [0u8; MIN_HEADER_LEN];
        bytes[12] = 4 << 4;
        assert_eq!(
            TcpHeader::parse(&bytes).unwrap_err(),
            DecodeError::BadHeaderLen { field: 4 }
        );
    }

    proptest! {
        // Parser must never panic regardless of what bytes arrive off the wire.
        #[test]
        fn parse_never_panics_on_arbitrary_bytes(
            data in prop::collection::vec(any::<u8>(), 0..128),
        ) {
            let _ = TcpHeader::parse(&data);
        }

        // TcpFlags uses only the low 6 bits; from_u8(b).as_u8() must equal b
        // with the upper 2 bits masked off.
        #[test]
        fn flags_any_byte_round_trips_low_six_bits(flags_byte in any::<u8>()) {
            let flags = TcpFlags::from_u8(flags_byte);
            prop_assert_eq!(flags.as_u8(), flags_byte & 0x3f,
                "TcpFlags should only encode the low 6 control bits");
        }

        // write_header followed by parse must recover all supplied fields and
        // produce a checksum that validates against the same IP addresses.
        #[test]
        fn write_then_parse_round_trips_arbitrary_fields(
            src_ip in prop::array::uniform4(any::<u8>()),
            dst_ip in prop::array::uniform4(any::<u8>()),
            src_port in any::<u16>(),
            dst_port in any::<u16>(),
            sequence in any::<u32>(),
            acknowledgement in any::<u32>(),
            flags_byte in any::<u8>(),
            window in any::<u16>(),
            payload in prop::collection::vec(any::<u8>(), 0..64),
        ) {
            let flags = TcpFlags::from_u8(flags_byte);
            let total = MIN_HEADER_LEN + payload.len();
            let mut buf = vec![0u8; total];
            buf[MIN_HEADER_LEN..].copy_from_slice(&payload);

            write_header(
                &mut buf, src_ip, dst_ip, src_port, dst_port,
                sequence, acknowledgement, flags, window, &payload,
            ).expect("buf is MIN_HEADER_LEN + payload bytes");

            let header = TcpHeader::parse(&buf).expect("write_header output must parse");

            prop_assert_eq!(header.source_port(), src_port);
            prop_assert_eq!(header.destination_port(), dst_port);
            prop_assert_eq!(header.sequence(), sequence);
            prop_assert_eq!(header.acknowledgement(), acknowledgement);
            prop_assert_eq!(header.flags(), flags);
            prop_assert_eq!(header.window(), window);
            prop_assert_eq!(header.payload(), payload.as_slice());
            prop_assert!(header.checksum_valid(src_ip, dst_ip),
                "checksum must be valid after write_header");
        }
    }
}
