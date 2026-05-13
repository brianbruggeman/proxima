//! `proxima_codec::Datagram` impl for memcached's UDP mode — the text
//! protocol's own [`Command`]/[`parse_command`] stay untouched; this
//! module adds only what UDP mode needs on top: the 8-byte per-datagram
//! header (request id / sequence number / total datagrams / reserved,
//! [protocol spec][1]) and an ASCII encoder for the reverse direction.
//!
//! Multi-datagram reassembly (a command whose ASCII body spans more
//! than one UDP packet, tracked by `(peer, request_id)`) is a stateful
//! concern that lives ABOVE `decode` — out of scope here, matching
//! [`proxima_codec::Datagram::decode`]'s own contract that each call
//! sees exactly one already-delivered packet.
//!
//! [1]: https://github.com/memcached/memcached/blob/master/doc/protocol.txt

use alloc::vec::Vec;
use core::net::SocketAddr;

use proxima_codec::{Addressed, Datagram};

use super::{Command, ParseError, StoreMode, parse_command};

/// UDP-mode 8-byte datagram header prefixing every memcached request
/// and reply datagram. All four fields are big-endian `u16`. `reserved`
/// must be zero on the wire per spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatagramHeader {
    pub request_id: u16,
    pub sequence_number: u16,
    pub total_datagrams: u16,
    pub reserved: u16,
}

/// On-wire byte width of [`DatagramHeader`].
pub const HEADER_BYTES: usize = 8;

fn parse_datagram_header(buf: &[u8]) -> Result<(DatagramHeader, &[u8]), ParseError> {
    if buf.len() < HEADER_BYTES {
        return Err(ParseError::DatagramHeaderShort);
    }
    let request_id = u16::from_be_bytes([buf[0], buf[1]]);
    let sequence_number = u16::from_be_bytes([buf[2], buf[3]]);
    let total_datagrams = u16::from_be_bytes([buf[4], buf[5]]);
    let reserved = u16::from_be_bytes([buf[6], buf[7]]);
    if reserved != 0 {
        return Err(ParseError::Malformed(
            "udp header reserved field must be zero",
        ));
    }
    Ok((
        DatagramHeader {
            request_id,
            sequence_number,
            total_datagrams,
            reserved,
        },
        &buf[HEADER_BYTES..],
    ))
}

fn write_decimal_u32(dest: &mut Vec<u8>, mut value: u32) {
    let mut digits = [0_u8; 10];
    let mut count = 0;
    loop {
        digits[count] = b'0' + (value % 10) as u8;
        value /= 10;
        count += 1;
        if value == 0 {
            break;
        }
    }
    dest.extend(digits[..count].iter().rev());
}

fn write_decimal_u64(dest: &mut Vec<u8>, mut value: u64) {
    let mut digits = [0_u8; 20];
    let mut count = 0;
    loop {
        digits[count] = b'0' + (value % 10) as u8;
        value /= 10;
        count += 1;
        if value == 0 {
            break;
        }
    }
    dest.extend(digits[..count].iter().rev());
}

fn write_noreply(dest: &mut Vec<u8>, noreply: bool) {
    if noreply {
        dest.extend_from_slice(b" noreply");
    }
}

fn store_verb(mode: StoreMode) -> &'static [u8] {
    match mode {
        StoreMode::Set => b"set",
        StoreMode::Add => b"add",
        StoreMode::Replace => b"replace",
        StoreMode::Append => b"append",
        StoreMode::Prepend => b"prepend",
    }
}

/// Render one [`Command`] as the ASCII line(s) [`parse_command`]
/// accepts — the encode-side mirror kept in this module because the
/// text protocol has no existing encoder; UDP mode is the first
/// caller that needs to turn a `Command` back into wire bytes rather
/// than only ever parsing them.
fn encode_command(command: &Command<'_>, dest: &mut Vec<u8>) {
    match *command {
        Command::Get { keys, gets } => {
            dest.extend_from_slice(if gets { b"gets " } else { b"get " });
            dest.extend_from_slice(keys);
            dest.extend_from_slice(b"\r\n");
        }
        Command::Store {
            mode,
            key,
            flags,
            exptime,
            value,
            noreply,
        } => {
            dest.extend_from_slice(store_verb(mode));
            dest.push(b' ');
            dest.extend_from_slice(key);
            dest.push(b' ');
            write_decimal_u32(dest, flags);
            dest.push(b' ');
            write_decimal_u32(dest, exptime);
            dest.push(b' ');
            write_decimal_u32(dest, value.len() as u32);
            write_noreply(dest, noreply);
            dest.extend_from_slice(b"\r\n");
            dest.extend_from_slice(value);
            dest.extend_from_slice(b"\r\n");
        }
        Command::Cas {
            key,
            flags,
            exptime,
            cas_unique,
            value,
            noreply,
        } => {
            dest.extend_from_slice(b"cas ");
            dest.extend_from_slice(key);
            dest.push(b' ');
            write_decimal_u32(dest, flags);
            dest.push(b' ');
            write_decimal_u32(dest, exptime);
            dest.push(b' ');
            write_decimal_u32(dest, value.len() as u32);
            dest.push(b' ');
            write_decimal_u64(dest, cas_unique);
            write_noreply(dest, noreply);
            dest.extend_from_slice(b"\r\n");
            dest.extend_from_slice(value);
            dest.extend_from_slice(b"\r\n");
        }
        Command::Delete { key, noreply } => {
            dest.extend_from_slice(b"delete ");
            dest.extend_from_slice(key);
            write_noreply(dest, noreply);
            dest.extend_from_slice(b"\r\n");
        }
        Command::Counter {
            increment,
            key,
            delta,
            noreply,
        } => {
            dest.extend_from_slice(if increment { b"incr " } else { b"decr " });
            dest.extend_from_slice(key);
            dest.push(b' ');
            write_decimal_u64(dest, delta);
            write_noreply(dest, noreply);
            dest.extend_from_slice(b"\r\n");
        }
        Command::Touch {
            key,
            exptime,
            noreply,
        } => {
            dest.extend_from_slice(b"touch ");
            dest.extend_from_slice(key);
            dest.push(b' ');
            write_decimal_u32(dest, exptime);
            write_noreply(dest, noreply);
            dest.extend_from_slice(b"\r\n");
        }
        Command::FlushAll { delay, noreply } => {
            dest.extend_from_slice(b"flush_all");
            if let Some(delay) = delay {
                dest.push(b' ');
                write_decimal_u32(dest, delay);
            }
            write_noreply(dest, noreply);
            dest.extend_from_slice(b"\r\n");
        }
        Command::Stats { args } => {
            dest.extend_from_slice(b"stats");
            if !args.is_empty() {
                dest.push(b' ');
                dest.extend_from_slice(args);
            }
            dest.extend_from_slice(b"\r\n");
        }
        Command::Version => dest.extend_from_slice(b"version\r\n"),
        Command::Quit => dest.extend_from_slice(b"quit\r\n"),
    }
}

/// [`Datagram`] impl for memcached's UDP mode. Zero-sized; clone
/// freely. `decode` strips and validates the 8-byte [`DatagramHeader`]
/// then hands the remainder to the unchanged text-protocol
/// [`parse_command`] — the header's fields are validated but not
/// retained on [`Datagram::Message`], matching the module's own
/// out-of-scope note on multi-datagram reassembly. `encode` always
/// emits a single-datagram header (`sequence_number = 0`,
/// `total_datagrams = 1`, `reserved = 0`); assigning `request_id` for
/// an outbound request is a session-level concern the caller layers
/// on top (there is no existing UDP-mode encoder in this crate whose
/// header-numbering behavior needs preserving).
#[derive(Debug, Clone, Copy, Default)]
pub struct MemcachedDatagramCodec;

impl Datagram for MemcachedDatagramCodec {
    type Message<'a> = Command<'a>;
    type Error = ParseError;

    fn decode<'a>(
        &self,
        peer: SocketAddr,
        bytes: &'a [u8],
    ) -> Result<Addressed<Command<'a>>, ParseError> {
        let (_header, payload) = parse_datagram_header(bytes)?;
        let (command, _consumed) = parse_command(payload)?;
        Ok(Addressed {
            peer,
            message: command,
        })
    }

    fn encode(
        &self,
        addressed: &Addressed<Command<'_>>,
        dest: &mut Vec<u8>,
    ) -> Result<(), ParseError> {
        dest.extend_from_slice(&0_u16.to_be_bytes()); // request_id
        dest.extend_from_slice(&0_u16.to_be_bytes()); // sequence_number
        dest.extend_from_slice(&1_u16.to_be_bytes()); // total_datagrams
        dest.extend_from_slice(&0_u16.to_be_bytes()); // reserved
        encode_command(&addressed.message, dest);
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn loopback_peer() -> SocketAddr {
        SocketAddr::from((core::net::Ipv4Addr::LOCALHOST, 11211))
    }

    fn udp_packet(request_id: u16, ascii: &[u8]) -> Vec<u8> {
        let mut packet = Vec::new();
        packet.extend_from_slice(&request_id.to_be_bytes());
        packet.extend_from_slice(&0_u16.to_be_bytes());
        packet.extend_from_slice(&1_u16.to_be_bytes());
        packet.extend_from_slice(&0_u16.to_be_bytes());
        packet.extend_from_slice(ascii);
        packet
    }

    #[test]
    fn decodes_get_datagram_after_header() {
        let codec = MemcachedDatagramCodec;
        let peer = loopback_peer();
        let packet = udp_packet(7, b"get mykey\r\n");

        let addressed = codec.decode(peer, &packet).expect("decode should succeed");
        assert_eq!(addressed.peer, peer);
        match addressed.message {
            Command::Get { keys, gets } => {
                assert_eq!(keys, b"mykey");
                assert!(!gets);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn short_datagram_before_header_is_hard_error() {
        // a UDP recvfrom() delivers the whole packet or nothing — 3
        // bytes is the entire datagram, never a partial-header signal.
        let codec = MemcachedDatagramCodec;
        let outcome = codec.decode(loopback_peer(), &[0, 0, 0]);
        assert!(matches!(outcome, Err(ParseError::DatagramHeaderShort)));
    }

    #[test]
    fn nonzero_reserved_field_is_hard_error() {
        let codec = MemcachedDatagramCodec;
        let mut packet = udp_packet(1, b"version\r\n");
        packet[7] = 1; // reserved low byte
        let outcome = codec.decode(loopback_peer(), &packet);
        assert!(matches!(outcome, Err(ParseError::Malformed(_))));
    }

    #[test]
    fn store_command_round_trips_through_encode_decode() {
        let codec = MemcachedDatagramCodec;
        let peer = loopback_peer();
        let packet = udp_packet(1, b"set foo 5 60 5\r\nhello\r\n");

        let decoded = codec.decode(peer, &packet).expect("decode should succeed");
        let mut encoded = Vec::new();
        codec
            .encode(&decoded, &mut encoded)
            .expect("encode should succeed");

        let re_decoded = codec
            .decode(peer, &encoded)
            .expect("re-decode of encoded bytes should succeed");
        match (decoded.message, re_decoded.message) {
            (
                Command::Store {
                    mode: mode_a,
                    key: key_a,
                    flags: flags_a,
                    exptime: exptime_a,
                    value: value_a,
                    noreply: noreply_a,
                },
                Command::Store {
                    mode: mode_b,
                    key: key_b,
                    flags: flags_b,
                    exptime: exptime_b,
                    value: value_b,
                    noreply: noreply_b,
                },
            ) => {
                assert_eq!(mode_a, mode_b);
                assert_eq!(key_a, key_b);
                assert_eq!(flags_a, flags_b);
                assert_eq!(exptime_a, exptime_b);
                assert_eq!(value_a, value_b);
                assert_eq!(noreply_a, noreply_b);
            }
            other => panic!("unexpected shape: {other:?}"),
        }
    }

    #[test]
    fn quit_command_round_trips() {
        let codec = MemcachedDatagramCodec;
        let peer = loopback_peer();
        let packet = udp_packet(1, b"quit\r\n");

        let decoded = codec.decode(peer, &packet).expect("decode should succeed");
        let mut encoded = Vec::new();
        codec
            .encode(&decoded, &mut encoded)
            .expect("encode should succeed");
        let re_decoded = codec
            .decode(peer, &encoded)
            .expect("re-decode should succeed");
        assert!(matches!(re_decoded.message, Command::Quit));
    }
}
