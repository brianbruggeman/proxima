use super::checksum::Checksum;
use super::error::DecodeError;
use super::ipv4::{self, Ipv4Protocol};

/// The fixed UDP header: source port, destination port, length, checksum.
pub const HEADER_LEN: usize = 8;

/// Borrowed view over a UDP datagram in a caller buffer. The payload is what
/// a higher protocol — QUIC, in our path — parses next.
#[derive(Debug, Clone, Copy)]
pub struct UdpHeader<'datagram> {
    bytes: &'datagram [u8],
}

impl<'datagram> UdpHeader<'datagram> {
    pub fn parse(bytes: &'datagram [u8]) -> Result<Self, DecodeError> {
        if bytes.len() < HEADER_LEN {
            return Err(DecodeError::Truncated {
                need: HEADER_LEN,
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

    /// Length field: header plus payload, in bytes.
    #[must_use]
    pub fn length(&self) -> u16 {
        u16::from_be_bytes([self.bytes[4], self.bytes[5]])
    }

    #[must_use]
    pub fn checksum(&self) -> u16 {
        u16::from_be_bytes([self.bytes[6], self.bytes[7]])
    }

    /// Payload following the 8-byte header, capped by the length field.
    #[must_use]
    pub fn payload(&self) -> &'datagram [u8] {
        let end = usize::from(self.length())
            .min(self.bytes.len())
            .max(HEADER_LEN);
        &self.bytes[HEADER_LEN..end]
    }

    /// Recompute the checksum over the IPv4 pseudo-header + datagram. A zero
    /// stored checksum means "not computed" (legal over IPv4) and is treated
    /// as valid.
    #[must_use]
    pub fn checksum_valid(&self, source: [u8; 4], destination: [u8; 4]) -> bool {
        if self.checksum() == 0 {
            return true;
        }
        let mut accumulator = Checksum::new();
        let l4_len = self.bytes.len() as u16;
        ipv4::pseudo_header_sum(
            &mut accumulator,
            source,
            destination,
            Ipv4Protocol::Udp,
            l4_len,
        );
        accumulator.add_bytes(self.bytes);
        accumulator.finish() == 0
    }
}

/// Write an 8-byte UDP header into the front of `out`, computing the checksum
/// over the pseudo-header + header + `payload`. The payload must already be at
/// `out[8..8 + payload.len()]`; it is passed only to fold into the checksum.
///
/// A computed checksum of zero is transmitted as `0xffff` — on the wire a
/// literal zero means "no checksum", so the all-ones representation is used
/// instead (RFC 768).
pub fn write_header(
    out: &mut [u8],
    source_ip: [u8; 4],
    destination_ip: [u8; 4],
    source_port: u16,
    destination_port: u16,
    payload: &[u8],
) -> Result<usize, DecodeError> {
    if out.len() < HEADER_LEN {
        return Err(DecodeError::Truncated {
            need: HEADER_LEN,
            got: out.len(),
        });
    }
    let l4_len = HEADER_LEN as u16 + payload.len() as u16;
    let header = &mut out[..HEADER_LEN];
    header[0..2].copy_from_slice(&source_port.to_be_bytes());
    header[2..4].copy_from_slice(&destination_port.to_be_bytes());
    header[4..6].copy_from_slice(&l4_len.to_be_bytes());
    header[6..8].copy_from_slice(&[0, 0]);
    let sum = {
        let mut accumulator = Checksum::new();
        ipv4::pseudo_header_sum(
            &mut accumulator,
            source_ip,
            destination_ip,
            Ipv4Protocol::Udp,
            l4_len,
        );
        accumulator.add_bytes(header);
        accumulator.add_bytes(payload);
        accumulator.finish()
    };
    let on_wire = if sum == 0 { 0xffff } else { sum };
    header[6..8].copy_from_slice(&on_wire.to_be_bytes());
    Ok(HEADER_LEN)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use proptest::prelude::*;

    const SRC_IP: [u8; 4] = [10, 0, 0, 1];
    const DST_IP: [u8; 4] = [10, 0, 0, 2];

    #[test]
    fn write_then_parse_with_valid_checksum() {
        let payload = b"quic-ish payload";
        let mut out = [0u8; HEADER_LEN + 16];
        out[HEADER_LEN..].copy_from_slice(payload);
        write_header(&mut out, SRC_IP, DST_IP, 0xc000, 443, payload).expect("buffer fits");
        let header = UdpHeader::parse(&out).expect("written header parses");
        assert_eq!(header.source_port(), 0xc000);
        assert_eq!(header.destination_port(), 443);
        assert_eq!(header.length(), HEADER_LEN as u16 + 16);
        assert_eq!(header.payload(), payload);
        assert!(header.checksum_valid(SRC_IP, DST_IP));
    }

    #[test]
    fn zero_checksum_is_accepted() {
        let mut bytes = [0u8; HEADER_LEN + 4];
        bytes[4..6].copy_from_slice(&((HEADER_LEN + 4) as u16).to_be_bytes());
        let header = UdpHeader::parse(&bytes).expect("parses");
        assert_eq!(header.checksum(), 0);
        assert!(header.checksum_valid(SRC_IP, DST_IP));
    }

    #[test]
    fn truncated_is_rejected() {
        assert_eq!(
            UdpHeader::parse(&[0u8; 7]).unwrap_err(),
            DecodeError::Truncated { need: 8, got: 7 }
        );
    }

    proptest! {
        // Parser must never panic on any byte sequence regardless of length.
        #[test]
        fn parse_never_panics_on_arbitrary_bytes(
            data in prop::collection::vec(any::<u8>(), 0..128),
        ) {
            let _ = UdpHeader::parse(&data);
        }

        // write_header followed by parse must recover the supplied field values
        // and the computed checksum must validate against the same IP addresses.
        #[test]
        fn write_then_parse_round_trips_arbitrary_fields(
            src_ip in prop::array::uniform4(any::<u8>()),
            dst_ip in prop::array::uniform4(any::<u8>()),
            src_port in any::<u16>(),
            dst_port in any::<u16>(),
            payload in prop::collection::vec(any::<u8>(), 0..64),
        ) {
            let total = HEADER_LEN + payload.len();
            let mut buf = vec![0u8; total];
            buf[HEADER_LEN..].copy_from_slice(&payload);

            write_header(&mut buf, src_ip, dst_ip, src_port, dst_port, &payload)
                .expect("buf is HEADER_LEN + payload bytes");

            let header = UdpHeader::parse(&buf).expect("write_header output must parse");

            prop_assert_eq!(header.source_port(), src_port);
            prop_assert_eq!(header.destination_port(), dst_port);
            prop_assert_eq!(header.length(), HEADER_LEN as u16 + payload.len() as u16);
            prop_assert_eq!(header.payload(), payload.as_slice());
            prop_assert!(header.checksum_valid(src_ip, dst_ip),
                "checksum must be valid after write_header");
        }
    }
}
