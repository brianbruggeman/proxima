//! Minimal sans-IO L2/L3 responder: enough of ARP and ICMP to answer a ping
//! over the dpdk packet path. Pure byte transforms over a caller-owned frame —
//! no I/O, no allocation — so they unit-test without dpdk. The dpdk binary just
//! feeds RX frames in and transmits the ones marked [`Action::Transmit`].

const ETHERTYPE_OFFSET: usize = 12;
const ETH_HEADER_LEN: usize = 14;
const ETHERTYPE_ARP: u16 = 0x0806;
const ETHERTYPE_IPV4: u16 = 0x0800;

const ARP_LEN: usize = 28;
const ARP_REQUEST: u16 = 1;
const ARP_REPLY: u16 = 2;

const IPV4_MIN_LEN: usize = 20;
const IP_PROTO_ICMP: u8 = 1;
const ICMP_ECHO_REQUEST: u8 = 8;
const ICMP_ECHO_REPLY: u8 = 0;

/// What the caller should do with the frame after [`handle_frame`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Frame was rewritten in place into a reply; transmit it back out.
    Transmit,
    /// Nothing to answer; free the buffer.
    Drop,
}

/// Rewrite `frame` in place into a reply if it is an ARP request or ICMP echo
/// addressed to us; otherwise leave it and return [`Action::Drop`].
pub fn handle_frame(frame: &mut [u8], our_mac: [u8; 6], our_ip: [u8; 4]) -> Action {
    if frame.len() < ETH_HEADER_LEN {
        return Action::Drop;
    }
    let ethertype = u16::from_be_bytes([frame[ETHERTYPE_OFFSET], frame[ETHERTYPE_OFFSET + 1]]);
    match ethertype {
        ETHERTYPE_ARP => handle_arp(frame, our_mac, our_ip),
        ETHERTYPE_IPV4 => handle_ipv4(frame, our_mac, our_ip),
        _ => Action::Drop,
    }
}

fn handle_arp(frame: &mut [u8], our_mac: [u8; 6], our_ip: [u8; 4]) -> Action {
    let arp = &frame[ETH_HEADER_LEN..];
    if arp.len() < ARP_LEN || u16::from_be_bytes([arp[6], arp[7]]) != ARP_REQUEST {
        return Action::Drop;
    }
    if arp[24..28] != our_ip {
        return Action::Drop;
    }

    let sender_mac = mac_at(&arp[8..14]);
    let sender_ip = ipv4_at(&arp[14..18]);

    let arp = &mut frame[ETH_HEADER_LEN..];
    arp[6..8].copy_from_slice(&ARP_REPLY.to_be_bytes());
    arp[8..14].copy_from_slice(&our_mac);
    arp[14..18].copy_from_slice(&our_ip);
    arp[18..24].copy_from_slice(&sender_mac);
    arp[24..28].copy_from_slice(&sender_ip);

    set_eth_endpoints(frame, sender_mac, our_mac);
    Action::Transmit
}

fn handle_ipv4(frame: &mut [u8], our_mac: [u8; 6], our_ip: [u8; 4]) -> Action {
    let ip = &frame[ETH_HEADER_LEN..];
    if ip.len() < IPV4_MIN_LEN {
        return Action::Drop;
    }
    let header_len = ((ip[0] & 0x0f) as usize) * 4;
    if header_len < IPV4_MIN_LEN || ip.len() < header_len + 8 || ip[9] != IP_PROTO_ICMP {
        return Action::Drop;
    }
    if ip[16..20] != our_ip {
        return Action::Drop;
    }
    if ip[header_len] != ICMP_ECHO_REQUEST {
        return Action::Drop;
    }

    let eth_src = mac_at(&frame[6..12]);

    // swap ip src/dst — the ipv4 header checksum is unchanged because one's-
    // complement addition is commutative over the two equal-width address fields.
    let ip = &mut frame[ETH_HEADER_LEN..];
    let source = ipv4_at(&ip[12..16]);
    let destination = ipv4_at(&ip[16..20]);
    ip[12..16].copy_from_slice(&destination);
    ip[16..20].copy_from_slice(&source);

    let icmp = &mut ip[header_len..];
    icmp[0] = ICMP_ECHO_REPLY;
    icmp[2] = 0;
    icmp[3] = 0;
    let sum = rfc1071(icmp);
    icmp[2..4].copy_from_slice(&sum.to_be_bytes());

    set_eth_endpoints(frame, eth_src, our_mac);
    Action::Transmit
}

fn set_eth_endpoints(frame: &mut [u8], destination: [u8; 6], source: [u8; 6]) {
    frame[0..6].copy_from_slice(&destination);
    frame[6..12].copy_from_slice(&source);
}

/// Build a broadcast ARP request ("who-has `target_ip`") into `out`. Returns the
/// frame length (42), or 0 if `out` is too short. Used by active-open clients to
/// resolve a peer MAC before sending the SYN.
#[must_use]
pub fn build_arp_request(
    out: &mut [u8],
    our_mac: [u8; 6],
    our_ip: [u8; 4],
    target_ip: [u8; 4],
) -> usize {
    let total = ETH_HEADER_LEN + ARP_LEN;
    if out.len() < total {
        return 0;
    }
    out[0..6].copy_from_slice(&[0xff; 6]);
    out[6..12].copy_from_slice(&our_mac);
    out[12..14].copy_from_slice(&ETHERTYPE_ARP.to_be_bytes());
    let arp = &mut out[ETH_HEADER_LEN..total];
    arp[0..2].copy_from_slice(&1u16.to_be_bytes()); // htype: ethernet
    arp[2..4].copy_from_slice(&ETHERTYPE_IPV4.to_be_bytes()); // ptype: ipv4
    arp[4] = 6;
    arp[5] = 4;
    arp[6..8].copy_from_slice(&ARP_REQUEST.to_be_bytes());
    arp[8..14].copy_from_slice(&our_mac);
    arp[14..18].copy_from_slice(&our_ip);
    arp[18..24].copy_from_slice(&[0u8; 6]);
    arp[24..28].copy_from_slice(&target_ip);
    total
}

/// Parse an ARP reply, returning `(sender_ip, sender_mac)` to learn into a cache.
#[must_use]
pub fn parse_arp_reply(frame: &[u8]) -> Option<([u8; 4], [u8; 6])> {
    if frame.len() < ETH_HEADER_LEN + ARP_LEN {
        return None;
    }
    if u16::from_be_bytes([frame[ETHERTYPE_OFFSET], frame[ETHERTYPE_OFFSET + 1]]) != ETHERTYPE_ARP {
        return None;
    }
    let arp = &frame[ETH_HEADER_LEN..];
    if u16::from_be_bytes([arp[6], arp[7]]) != ARP_REPLY {
        return None;
    }
    Some((ipv4_at(&arp[14..18]), mac_at(&arp[8..14])))
}

fn mac_at(bytes: &[u8]) -> [u8; 6] {
    let mut out = [0u8; 6];
    out.copy_from_slice(&bytes[..6]);
    out
}

fn ipv4_at(bytes: &[u8]) -> [u8; 4] {
    let mut out = [0u8; 4];
    out.copy_from_slice(&bytes[..4]);
    out
}

// RFC 1071 internet checksum over one contiguous buffer (mirrors
// proxima-inet-codec::checksum; inlined to keep this crate dpdk-only-standalone).
fn rfc1071(data: &[u8]) -> u16 {
    let mut sum: u32 = 0;
    let mut words = data.chunks_exact(2);
    for word in &mut words {
        sum += u32::from(u16::from_be_bytes([word[0], word[1]]));
    }
    if let [last] = words.remainder() {
        sum += u32::from(u16::from_be_bytes([*last, 0]));
    }
    while (sum >> 16) != 0 {
        sum = (sum & 0xffff) + (sum >> 16);
    }
    // folded above, so the low 16 bits are the whole value; try_from never fails.
    !u16::try_from(sum & 0xffff).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]
    use super::*;

    const OUR_MAC: [u8; 6] = [0x76, 0xde, 0x0f, 0xd1, 0xb9, 0xbf];
    const OUR_IP: [u8; 4] = [10, 0, 0, 2];
    const PEER_MAC: [u8; 6] = [0x02, 0x11, 0x22, 0x33, 0x44, 0x55];
    const PEER_IP: [u8; 4] = [10, 0, 0, 1];

    fn arp_request() -> Vec<u8> {
        let mut frame = vec![0xff; 6]; // broadcast dst
        frame.extend_from_slice(&PEER_MAC); // src
        frame.extend_from_slice(&ETHERTYPE_ARP.to_be_bytes());
        frame.extend_from_slice(&[0x00, 0x01, 0x08, 0x00, 0x06, 0x04]); // htype/ptype/hlen/plen
        frame.extend_from_slice(&ARP_REQUEST.to_be_bytes());
        frame.extend_from_slice(&PEER_MAC); // sha
        frame.extend_from_slice(&PEER_IP); // spa
        frame.extend_from_slice(&[0; 6]); // tha unknown
        frame.extend_from_slice(&OUR_IP); // tpa = who-has us
        frame
    }

    #[test]
    fn arp_request_for_us_becomes_a_reply() {
        let mut frame = arp_request();
        let action = handle_frame(&mut frame, OUR_MAC, OUR_IP);
        assert_eq!(action, Action::Transmit);
        let arp = &frame[ETH_HEADER_LEN..];
        assert_eq!(u16::from_be_bytes([arp[6], arp[7]]), ARP_REPLY);
        assert_eq!(&arp[8..14], &OUR_MAC, "reply sender mac is ours");
        assert_eq!(&arp[14..18], &OUR_IP, "reply sender ip is ours");
        assert_eq!(&arp[18..24], &PEER_MAC, "target mac is the requester");
        assert_eq!(&frame[0..6], &PEER_MAC, "eth dst is the requester");
        assert_eq!(&frame[6..12], &OUR_MAC, "eth src is ours");
    }

    #[test]
    fn arp_request_for_someone_else_is_dropped() {
        let mut frame = arp_request();
        let other_ip = [10, 0, 0, 9];
        assert_eq!(handle_frame(&mut frame, OUR_MAC, other_ip), Action::Drop);
    }

    fn icmp_echo_request() -> Vec<u8> {
        let mut frame = Vec::new();
        frame.extend_from_slice(&OUR_MAC); // dst = us
        frame.extend_from_slice(&PEER_MAC); // src
        frame.extend_from_slice(&ETHERTYPE_IPV4.to_be_bytes());
        // ipv4: ihl=5, total len 28, ttl 64, proto icmp, zero checksum, peer->us
        let mut ip = vec![
            0x45, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x40, 0x00, 0x40, 0x01, 0x00, 0x00,
        ];
        ip.extend_from_slice(&PEER_IP);
        ip.extend_from_slice(&OUR_IP);
        let ip_checksum = rfc1071(&ip);
        ip[10..12].copy_from_slice(&ip_checksum.to_be_bytes());
        frame.extend_from_slice(&ip);
        // icmp echo request: type 8, code 0, checksum, id 1, seq 1
        let mut icmp = vec![0x08, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01];
        let icmp_checksum = rfc1071(&icmp);
        icmp[2..4].copy_from_slice(&icmp_checksum.to_be_bytes());
        frame.extend_from_slice(&icmp);
        frame
    }

    #[test]
    fn icmp_echo_request_to_us_becomes_a_valid_reply() {
        let mut frame = icmp_echo_request();
        let action = handle_frame(&mut frame, OUR_MAC, OUR_IP);
        assert_eq!(action, Action::Transmit);

        let ip = &frame[ETH_HEADER_LEN..];
        assert_eq!(&ip[12..16], &OUR_IP, "reply source ip is ours");
        assert_eq!(&ip[16..20], &PEER_IP, "reply dest ip is the pinger");
        assert_eq!(
            rfc1071(&ip[..20]),
            0,
            "ipv4 header checksum stays valid after swap"
        );

        let icmp = &ip[20..];
        assert_eq!(icmp[0], ICMP_ECHO_REPLY);
        assert_eq!(rfc1071(icmp), 0, "icmp checksum verifies to zero");

        assert_eq!(&frame[0..6], &PEER_MAC, "eth dst is the pinger");
        assert_eq!(&frame[6..12], &OUR_MAC, "eth src is ours");
    }

    #[test]
    fn non_ip_non_arp_is_dropped() {
        let mut frame = vec![0u8; 14];
        frame[12] = 0x86;
        frame[13] = 0xdd; // ipv6
        assert_eq!(handle_frame(&mut frame, OUR_MAC, OUR_IP), Action::Drop);
    }

    #[test]
    fn runt_frame_is_dropped() {
        let mut frame = vec![0u8; 8];
        assert_eq!(handle_frame(&mut frame, OUR_MAC, OUR_IP), Action::Drop);
    }
}
