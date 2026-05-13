use super::checksum::Checksum;
use super::error::DecodeError;

/// L4 protocol carried by an IPv4 packet (the `protocol` byte).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ipv4Protocol {
    Tcp,
    Udp,
    Icmp,
    Other(u8),
}

impl Ipv4Protocol {
    #[must_use]
    pub const fn from_u8(value: u8) -> Self {
        match value {
            6 => Self::Tcp,
            17 => Self::Udp,
            1 => Self::Icmp,
            other => Self::Other(other),
        }
    }

    #[must_use]
    pub const fn as_u8(self) -> u8 {
        match self {
            Self::Tcp => 6,
            Self::Udp => 17,
            Self::Icmp => 1,
            Self::Other(value) => value,
        }
    }
}

/// Minimum IPv4 header with no options (IHL = 5 words).
pub const MIN_HEADER_LEN: usize = 20;

/// Borrowed view over an IPv4 header in a caller buffer. Options, if present,
/// are exposed as a byte slice rather than parsed.
#[derive(Debug, Clone, Copy)]
pub struct Ipv4Header<'packet> {
    bytes: &'packet [u8],
}

impl<'packet> Ipv4Header<'packet> {
    /// Borrow a header view. Validates version, IHL, and that the declared
    /// header length fits the buffer.
    pub fn parse(bytes: &'packet [u8]) -> Result<Self, DecodeError> {
        if bytes.len() < MIN_HEADER_LEN {
            return Err(DecodeError::Truncated {
                need: MIN_HEADER_LEN,
                got: bytes.len(),
            });
        }
        let version = bytes[0] >> 4;
        if version != 4 {
            return Err(DecodeError::BadVersion { found: version });
        }
        let ihl = bytes[0] & 0x0f;
        if ihl < 5 {
            return Err(DecodeError::BadHeaderLen { field: ihl });
        }
        let header_len = usize::from(ihl) * 4;
        if bytes.len() < header_len {
            return Err(DecodeError::Truncated {
                need: header_len,
                got: bytes.len(),
            });
        }
        Ok(Self { bytes })
    }

    #[must_use]
    pub fn header_len(&self) -> usize {
        usize::from(self.bytes[0] & 0x0f) * 4
    }

    /// Total length field: header plus payload, in bytes.
    #[must_use]
    pub fn total_len(&self) -> u16 {
        u16::from_be_bytes([self.bytes[2], self.bytes[3]])
    }

    #[must_use]
    pub fn protocol(&self) -> Ipv4Protocol {
        Ipv4Protocol::from_u8(self.bytes[9])
    }

    #[must_use]
    pub fn ttl(&self) -> u8 {
        self.bytes[8]
    }

    #[must_use]
    pub fn source(&self) -> [u8; 4] {
        [
            self.bytes[12],
            self.bytes[13],
            self.bytes[14],
            self.bytes[15],
        ]
    }

    #[must_use]
    pub fn destination(&self) -> [u8; 4] {
        [
            self.bytes[16],
            self.bytes[17],
            self.bytes[18],
            self.bytes[19],
        ]
    }

    #[must_use]
    pub fn header_checksum(&self) -> u16 {
        u16::from_be_bytes([self.bytes[10], self.bytes[11]])
    }

    /// Recompute the header checksum and compare to the stored field.
    #[must_use]
    pub fn checksum_valid(&self) -> bool {
        let mut accumulator = Checksum::new();
        accumulator.add_bytes(&self.bytes[..self.header_len()]);
        accumulator.finish() == 0
    }

    /// Payload following the header (length capped by `total_len`).
    #[must_use]
    pub fn payload(&self) -> &'packet [u8] {
        let start = self.header_len();
        let end = usize::from(self.total_len())
            .min(self.bytes.len())
            .max(start);
        &self.bytes[start..end]
    }
}

/// Write a 20-byte IPv4 header (no options) into the front of `out`,
/// computing the header checksum. Returns the payload start offset.
#[allow(clippy::too_many_arguments)]
pub fn write_header(
    out: &mut [u8],
    source: [u8; 4],
    destination: [u8; 4],
    protocol: Ipv4Protocol,
    ttl: u8,
    payload_len: u16,
    identification: u16,
) -> Result<usize, DecodeError> {
    if out.len() < MIN_HEADER_LEN {
        return Err(DecodeError::Truncated {
            need: MIN_HEADER_LEN,
            got: out.len(),
        });
    }
    let header = &mut out[..MIN_HEADER_LEN];
    header.fill(0);
    header[0] = (4 << 4) | 5;
    let total = MIN_HEADER_LEN as u16 + payload_len;
    header[2..4].copy_from_slice(&total.to_be_bytes());
    header[4..6].copy_from_slice(&identification.to_be_bytes());
    header[6] = 0x40; // don't fragment
    header[8] = ttl;
    header[9] = protocol.as_u8();
    header[12..16].copy_from_slice(&source);
    header[16..20].copy_from_slice(&destination);
    let sum = {
        let mut accumulator = Checksum::new();
        accumulator.add_bytes(header);
        accumulator.finish()
    };
    header[10..12].copy_from_slice(&sum.to_be_bytes());
    Ok(MIN_HEADER_LEN)
}

/// Fold the IPv4 pseudo-header (source, destination, zero, protocol, L4
/// length) into `accumulator` ahead of the L4 header+payload. TCP and UDP
/// share this exact prefix for their checksums.
pub fn pseudo_header_sum(
    accumulator: &mut Checksum,
    source: [u8; 4],
    destination: [u8; 4],
    protocol: Ipv4Protocol,
    l4_len: u16,
) {
    accumulator.add_bytes(&source);
    accumulator.add_bytes(&destination);
    accumulator.add_bytes(&[0, protocol.as_u8()]);
    accumulator.add_bytes(&l4_len.to_be_bytes());
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use proptest::prelude::*;

    // Canonical IPv4 header from the discipline log; stored checksum 0xb861.
    const CANONICAL: [u8; 20] = [
        0x45, 0x00, 0x00, 0x73, 0x00, 0x00, 0x40, 0x00, 0x40, 0x11, 0xb8, 0x61, 0xc0, 0xa8, 0x00,
        0x01, 0xc0, 0xa8, 0x00, 0xc7,
    ];

    #[test]
    fn parse_canonical_header() {
        let header = Ipv4Header::parse(&CANONICAL).expect("valid header");
        assert_eq!(header.header_len(), 20);
        assert_eq!(header.total_len(), 0x73);
        assert_eq!(header.protocol(), Ipv4Protocol::Udp);
        assert_eq!(header.source(), [192, 168, 0, 1]);
        assert_eq!(header.destination(), [192, 168, 0, 199]);
        assert_eq!(header.header_checksum(), 0xb861);
        assert!(header.checksum_valid());
    }

    #[test]
    fn rejects_non_v4_and_short_ihl() {
        let mut bad_version = CANONICAL;
        bad_version[0] = 0x65;
        assert_eq!(
            Ipv4Header::parse(&bad_version).unwrap_err(),
            DecodeError::BadVersion { found: 6 }
        );
        let mut bad_ihl = CANONICAL;
        bad_ihl[0] = 0x44;
        assert_eq!(
            Ipv4Header::parse(&bad_ihl).unwrap_err(),
            DecodeError::BadHeaderLen { field: 4 }
        );
    }

    #[test]
    fn write_header_computes_valid_checksum() {
        let mut out = [0u8; MIN_HEADER_LEN];
        write_header(
            &mut out,
            [192, 168, 0, 1],
            [192, 168, 0, 199],
            Ipv4Protocol::Tcp,
            64,
            100,
            1,
        )
        .expect("buffer fits");
        let header = Ipv4Header::parse(&out).expect("written header parses");
        assert!(header.checksum_valid());
        assert_eq!(header.protocol(), Ipv4Protocol::Tcp);
        assert_eq!(header.total_len(), MIN_HEADER_LEN as u16 + 100);
    }

    proptest! {
        // Parser must never panic on any byte sequence regardless of length.
        #[test]
        fn parse_never_panics_on_arbitrary_bytes(
            data in prop::collection::vec(any::<u8>(), 0..128),
        ) {
            let _ = Ipv4Header::parse(&data);
        }

        // Ipv4Protocol round-trips: any u8 protocol byte encodes and decodes.
        #[test]
        fn protocol_u8_round_trips(value in any::<u8>()) {
            prop_assert_eq!(Ipv4Protocol::from_u8(value).as_u8(), value);
        }

        // write_header always produces a header whose checksum validates, and
        // the field values survive the encode→parse round-trip exactly.
        #[test]
        fn write_then_parse_round_trips_arbitrary_fields(
            src in prop::array::uniform4(any::<u8>()),
            dst in prop::array::uniform4(any::<u8>()),
            protocol_raw in any::<u8>(),
            ttl in any::<u8>(),
            // cap payload_len so total_len (20 + payload_len) does not overflow u16
            payload_len in 0u16..=60000u16,
            identification in any::<u16>(),
        ) {
            let protocol = Ipv4Protocol::from_u8(protocol_raw);
            let mut buf = [0u8; MIN_HEADER_LEN];

            write_header(&mut buf, src, dst, protocol, ttl, payload_len, identification)
                .expect("MIN_HEADER_LEN buffer always fits");

            let header = Ipv4Header::parse(&buf).expect("write_header output must parse");

            prop_assert!(header.checksum_valid(), "checksum must be valid after write_header");
            prop_assert_eq!(header.source(), src);
            prop_assert_eq!(header.destination(), dst);
            prop_assert_eq!(header.protocol(), protocol);
            prop_assert_eq!(header.ttl(), ttl);
            prop_assert_eq!(header.total_len(), MIN_HEADER_LEN as u16 + payload_len);
        }
    }
}
