//! Host-side virtio-net transport: owns one
//! [`proxima_protocols::virtio::NetConfigSpace`] (the sans-IO device codec)
//! plus the TX queue's ring cursors, and reads real guest memory to walk a
//! descriptor chain once a `QueueNotify` effect names the transmit queue —
//! the same "one layer up" split `virtio_console::ConsoleTransport` uses
//! (MMIO register decode plus guest-physical-to-host-pointer translation
//! live here, never inside the sans-IO codec itself). Mirror, not a
//! redesign, of `super::virtio_console`'s shape.
//!
//! Per VIRTIO 1.2 spec §5.1.2, a non-multiqueue net device exposes exactly
//! two queues: receiveq1 = 0, transmitq1 = 1. This slice drains the
//! transmit queue only — receiveq1 delivery (host-to-guest frames) is out
//! of scope, matching M6's exit criterion (proving a frame crosses the ring
//! host-ward through a real chain walk, not a spec-complete NIC).
//!
//! Each drained chain is `virtio_net_hdr` (12 bytes, `proxima_protocols::
//! virtio::net::NET_HDR_LEN`) followed by one Ethernet frame (spec
//! §5.1.6.1). This transport strips the header and hands the frame to a
//! caller-supplied [`FrameSink`] — the same shape `proxima-net`'s own
//! backends (`proxima_net::xdp::stream_listener`,
//! `proxima_net::dpdk::stream_listener`) use to route a raw frame into
//! `proxima_net::stack::handle_frame` (ARP/ICMP) or a `TcpStack` (TCP): each
//! of those backends owns its own frame-classification glue over the same
//! shared sans-IO primitives rather than proxima-net hosting one, so this
//! transport doing the same for the virtio-net backend is the established
//! pattern, not a new one (reuse-first, principle 1). Wiring straight to
//! `proxima_net::stack::handle_frame` is possible because that function's
//! own signature already takes a whole Ethernet frame — the exact shape
//! this transport produces after stripping the header, so there is no layer
//! mismatch to route around (`proxima-net/src/stack.rs:31`).

use proxima_protocols::virtio::{
    AvailRing, DecodeError, DescriptorChain, MmioAccess, MmioEffect, MmioError, NetConfigSpace,
    NetHdr, RingCursor, UsedElem, write_used_elem,
};

/// receiveq1 (spec §5.1.2) — host-to-guest delivery, not yet implemented by
/// this slice.
pub const RX_QUEUE: u16 = 0;
/// transmitq1 (spec §5.1.2) — the only queue this transport drains.
pub const TX_QUEUE: u16 = 1;
const QUEUE_SIZE: u16 = 4;

const _: () = assert!(QUEUE_SIZE != 0 && QUEUE_SIZE.is_power_of_two());

/// Why [`NetTransport::drain_tx`] could not complete — mirrors
/// `virtio_console::DrainError` exactly, plus [`Self::HeaderTooShort`] for
/// the one failure mode unique to a header-carrying device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainError {
    /// The queue named by a `QueueNotify` is not the transmit queue.
    NotTransmitQueue { queue: u16 },
    /// The queue's descriptor/driver/device addresses are not all
    /// programmed yet (a notify arrived before queue setup finished).
    QueueNotConfigured,
    /// A ring or descriptor-chain byte range fell outside `guest_memory`.
    OutOfBounds { need: usize, have: usize },
    /// The ring or chain bytes themselves failed to decode.
    Decode(DecodeError),
    /// A published chain carried fewer than `NET_HDR_LEN` bytes total, so no
    /// `virtio_net_hdr` could be recovered from it at all.
    HeaderTooShort { got: usize },
}

impl core::fmt::Display for DrainError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotTransmitQueue { queue } => {
                write!(formatter, "queue {queue} is not the transmit queue")
            }
            Self::QueueNotConfigured => write!(formatter, "queue addresses not yet programmed"),
            Self::OutOfBounds { need, have } => write!(
                formatter,
                "guest memory access needs {need} bytes, only {have} available"
            ),
            Self::Decode(inner) => write!(formatter, "{inner}"),
            Self::HeaderTooShort { got } => {
                write!(
                    formatter,
                    "chain carried {got} bytes, shorter than one virtio_net_hdr"
                )
            }
        }
    }
}

impl std::error::Error for DrainError {}

/// One TX chain's payload once its `virtio_net_hdr` has been stripped —
/// the header this slice reads but does not yet act on (no checksum/GSO
/// offload is negotiable, per `NetConfigSpace`'s narrow feature offer) plus
/// the raw Ethernet frame bytes.
pub struct DrainedFrame {
    pub num_buffers: u16,
    pub frame: Vec<u8>,
}

/// Receives one drained Ethernet frame per call — the seam a caller wires
/// to `proxima_net::stack::handle_frame` for ARP/ICMP and to a `TcpStack`
/// for TCP, or to a test's own inspection buffer. Kept as a plain closure
/// bound rather than a trait object: the caller set is one host loop plus
/// tests, never an open/unbounded set, so a `Box<dyn Trait>` (or a new
/// named trait to host one method) would be a relocation, not a
/// capability — the pipe/blanket-impl question this crate's own AGENTS.md
/// binds (guiding-principles principle 1's "what can a caller do that they
/// could not before" test).
pub trait FrameSink {
    fn deliver(&mut self, frame: DrainedFrame);
}

impl<F: FnMut(DrainedFrame)> FrameSink for F {
    fn deliver(&mut self, frame: DrainedFrame) {
        self(frame);
    }
}

/// Owns the net device's mmio register-block FSM (via `NetConfigSpace`) and
/// the TX queue's avail/used cursors.
///
/// `Clone` for the same M7 reason `ConsoleTransport` derives it: every field
/// is plain, `Clone`-able data.
#[derive(Debug, Clone)]
pub struct NetTransport {
    device: NetConfigSpace,
    avail_cursor: RingCursor,
    used_cursor: RingCursor,
}

impl NetTransport {
    #[must_use]
    #[allow(
        clippy::expect_used,
        reason = "QUEUE_SIZE's power-of-two invariant is checked at compile time above; RingCursor::new cannot fail for this literal"
    )]
    pub fn new(mac: [u8; 6]) -> Self {
        Self {
            device: NetConfigSpace::new(mac, QUEUE_SIZE),
            avail_cursor: RingCursor::new(QUEUE_SIZE)
                .expect("QUEUE_SIZE is a fixed power of two, proven above"),
            used_cursor: RingCursor::new(QUEUE_SIZE)
                .expect("QUEUE_SIZE is a fixed power of two, proven above"),
        }
    }

    /// Apply one raw register access recovered from a trapped guest
    /// load/store — delegates straight to [`NetConfigSpace::apply`].
    pub fn apply(&mut self, access: MmioAccess) -> Result<MmioEffect, MmioError> {
        self.device.apply(access)
    }

    /// Walks every avail-ring entry published on the transmit queue since
    /// this transport last drained it: each chain is one `virtio_net_hdr`
    /// followed by one Ethernet frame (concatenated across however many
    /// descriptors the driver split it into, same shape
    /// `virtio_console::ConsoleTransport::drain_tx` walks), delivered to
    /// `sink` with the header stripped off. Publishes one used-ring
    /// completion per chain (`len = 0`: transmit-only, spec §2.7.8).
    pub fn drain_tx(
        &mut self,
        queue: u16,
        guest_memory: &mut [u8],
        sink: &mut dyn FrameSink,
    ) -> Result<usize, DrainError> {
        if queue != TX_QUEUE {
            return Err(DrainError::NotTransmitQueue { queue });
        }
        let descriptor_address = self
            .device
            .transport()
            .queue_descriptor_address(queue)
            .filter(|address| *address != 0)
            .ok_or(DrainError::QueueNotConfigured)?;
        let driver_address = self
            .device
            .transport()
            .queue_driver_address(queue)
            .filter(|address| *address != 0)
            .ok_or(DrainError::QueueNotConfigured)?;
        let device_address = self
            .device
            .transport()
            .queue_device_address(queue)
            .filter(|address| *address != 0)
            .ok_or(DrainError::QueueNotConfigured)?;

        let mut delivered = 0usize;
        loop {
            let avail_bytes = slice_at(
                guest_memory,
                driver_address,
                4 + usize::from(QUEUE_SIZE) * 2,
            )?;
            let avail = AvailRing::parse(avail_bytes, QUEUE_SIZE).map_err(DrainError::Decode)?;
            let pending = self.avail_cursor.pending(avail.idx());
            if pending == 0 {
                break;
            }
            let head = avail.ring_entry(self.avail_cursor.position());
            self.avail_cursor.advance();

            let table_bytes = slice_at(
                guest_memory,
                descriptor_address,
                usize::from(QUEUE_SIZE) * proxima_protocols::virtio::DESC_LEN,
            )?;
            let mut addresses = Vec::new();
            for descriptor in DescriptorChain::new(table_bytes, QUEUE_SIZE, head) {
                let descriptor = descriptor.map_err(DrainError::Decode)?;
                addresses.push((descriptor.addr(), descriptor.buffer_len()));
            }
            let mut chain_bytes = Vec::new();
            for (addr, len) in addresses {
                let bytes = slice_at(guest_memory, addr, len as usize)?;
                chain_bytes.extend_from_slice(bytes);
            }

            let hdr_bytes = chain_bytes
                .get(..proxima_protocols::virtio::NET_HDR_LEN)
                .ok_or(DrainError::HeaderTooShort {
                    got: chain_bytes.len(),
                })?;
            let hdr = NetHdr::parse(hdr_bytes).map_err(DrainError::Decode)?;
            let frame = chain_bytes[proxima_protocols::virtio::NET_HDR_LEN..].to_vec();
            sink.deliver(DrainedFrame {
                num_buffers: hdr.num_buffers(),
                frame,
            });
            delivered += 1;

            let used_offset = 4 + usize::from(self.used_cursor.position() % QUEUE_SIZE)
                * proxima_protocols::virtio::USED_ELEM_LEN;
            let used_slot = mut_slice_at(
                guest_memory,
                device_address + used_offset as u64,
                proxima_protocols::virtio::USED_ELEM_LEN,
            )?;
            write_used_elem(
                used_slot,
                UsedElem {
                    id: u32::from(head),
                    len: 0,
                },
            )
            .map_err(DrainError::Decode)?;
            let new_idx = self.used_cursor.advance();
            let idx_slot = mut_slice_at(guest_memory, device_address + 2, 2)?;
            idx_slot.copy_from_slice(&new_idx.to_le_bytes());
        }
        Ok(delivered)
    }
}

fn slice_at(guest_memory: &[u8], address: u64, len: usize) -> Result<&[u8], DrainError> {
    let start = usize::try_from(address).map_err(|_| DrainError::OutOfBounds {
        need: len,
        have: guest_memory.len(),
    })?;
    let end = start.checked_add(len).ok_or(DrainError::OutOfBounds {
        need: len,
        have: guest_memory.len(),
    })?;
    guest_memory.get(start..end).ok_or(DrainError::OutOfBounds {
        need: end,
        have: guest_memory.len(),
    })
}

fn mut_slice_at(
    guest_memory: &mut [u8],
    address: u64,
    len: usize,
) -> Result<&mut [u8], DrainError> {
    let total = guest_memory.len();
    let start = usize::try_from(address).map_err(|_| DrainError::OutOfBounds {
        need: len,
        have: total,
    })?;
    let end = start.checked_add(len).ok_or(DrainError::OutOfBounds {
        need: len,
        have: total,
    })?;
    guest_memory
        .get_mut(start..end)
        .ok_or(DrainError::OutOfBounds {
            need: end,
            have: total,
        })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use proxima_net::stack;
    use proxima_net::tcp_stack::TcpStack;
    use proxima_protocols::tcp::time::Instant as TcpInstant;
    use proxima_protocols::virtio::{DESC_LEN, USED_ELEM_LEN};

    const OUR_MAC: [u8; 6] = [0x02, 0x00, 0x00, 0x00, 0x00, 0x01];
    const PEER_MAC: [u8; 6] = [0x02, 0x11, 0x22, 0x33, 0x44, 0x55];
    const OUR_IP: [u8; 4] = [10, 0, 0, 2];

    // Build a 4 KiB guest_memory image with descriptor table at 0x1000,
    // avail ring at 0x2000, used ring at 0x3000, and one published TX chain
    // (head 0) whose two descriptors carry the 12-byte virtio_net_hdr and
    // then `frame` — the same three-region layout the console transport's
    // own test doubles use, extended with a real virtio_net_hdr.
    fn guest_memory_with_one_tx_chain(frame: &[u8]) -> Vec<u8> {
        let mut memory = vec![0u8; 0x4000];

        // descriptor 0: hdr (device-readable, NEXT -> 2).
        let mut desc0 = [0u8; DESC_LEN];
        desc0[0..8].copy_from_slice(&0x1100u64.to_le_bytes());
        desc0[8..12].copy_from_slice(&12u32.to_le_bytes());
        desc0[12..14].copy_from_slice(&1u16.to_le_bytes()); // flags = NEXT
        desc0[14..16].copy_from_slice(&2u16.to_le_bytes()); // next = 2
        memory[0x1000..0x1000 + DESC_LEN].copy_from_slice(&desc0);

        // descriptor 2: frame (device-readable, chain terminator).
        let mut desc2 = [0u8; DESC_LEN];
        desc2[0..8].copy_from_slice(&0x1200u64.to_le_bytes());
        desc2[8..12].copy_from_slice(&(frame.len() as u32).to_le_bytes());
        memory[0x1000 + 2 * DESC_LEN..0x1000 + 2 * DESC_LEN + DESC_LEN].copy_from_slice(&desc2);

        // hdr payload at 0x1100: flags=0 gso_type=NONE hdr_len=0 gso_size=0
        // csum_start=0 csum_offset=0 num_buffers=1.
        memory[0x1100 + 10..0x1100 + 12].copy_from_slice(&1u16.to_le_bytes());

        // frame payload at 0x1200.
        memory[0x1200..0x1200 + frame.len()].copy_from_slice(frame);

        // avail ring at 0x2000: idx=1, ring[0]=head 0.
        memory[0x2000 + 2..0x2000 + 4].copy_from_slice(&1u16.to_le_bytes());

        memory
    }

    fn bring_up(
        transport: &mut NetTransport,
        descriptor_addr: u64,
        driver_addr: u64,
        device_addr: u64,
    ) {
        use proxima_protocols::virtio::mmio::{
            REG_DRIVER_FEATURES, REG_DRIVER_FEATURES_SEL, REG_QUEUE_DESC_HIGH, REG_QUEUE_DESC_LOW,
            REG_QUEUE_DEVICE_HIGH, REG_QUEUE_DEVICE_LOW, REG_QUEUE_DRIVER_HIGH,
            REG_QUEUE_DRIVER_LOW, REG_QUEUE_NUM, REG_QUEUE_READY, REG_QUEUE_SEL, REG_STATUS,
        };
        use proxima_protocols::virtio::status::{
            STATUS_ACKNOWLEDGE, STATUS_DRIVER, STATUS_DRIVER_OK, STATUS_FEATURES_OK,
        };

        let write = |offset: u64, value: u32| MmioAccess {
            offset,
            is_write: true,
            value,
        };
        transport
            .apply(write(REG_STATUS, u32::from(STATUS_ACKNOWLEDGE)))
            .unwrap();
        transport
            .apply(write(
                REG_STATUS,
                u32::from(STATUS_ACKNOWLEDGE | STATUS_DRIVER),
            ))
            .unwrap();
        transport.apply(write(REG_DRIVER_FEATURES_SEL, 0)).unwrap();
        transport.apply(write(REG_DRIVER_FEATURES, 0)).unwrap();
        transport.apply(write(REG_DRIVER_FEATURES_SEL, 1)).unwrap();
        transport.apply(write(REG_DRIVER_FEATURES, 1)).unwrap();
        transport
            .apply(write(
                REG_STATUS,
                u32::from(STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK),
            ))
            .unwrap();

        transport
            .apply(write(REG_QUEUE_SEL, TX_QUEUE.into()))
            .unwrap();
        transport
            .apply(write(REG_QUEUE_NUM, QUEUE_SIZE.into()))
            .unwrap();
        transport
            .apply(write(REG_QUEUE_DESC_LOW, descriptor_addr as u32))
            .unwrap();
        transport.apply(write(REG_QUEUE_DESC_HIGH, 0)).unwrap();
        transport
            .apply(write(REG_QUEUE_DRIVER_LOW, driver_addr as u32))
            .unwrap();
        transport.apply(write(REG_QUEUE_DRIVER_HIGH, 0)).unwrap();
        transport
            .apply(write(REG_QUEUE_DEVICE_LOW, device_addr as u32))
            .unwrap();
        transport.apply(write(REG_QUEUE_DEVICE_HIGH, 0)).unwrap();
        transport.apply(write(REG_QUEUE_READY, 1)).unwrap();

        transport
            .apply(write(
                REG_STATUS,
                u32::from(
                    STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK | STATUS_DRIVER_OK,
                ),
            ))
            .unwrap();
    }

    fn arp_request_frame() -> Vec<u8> {
        let mut frame = vec![0xff; 6]; // eth dst = broadcast
        frame.extend_from_slice(&PEER_MAC); // eth src
        frame.extend_from_slice(&0x0806u16.to_be_bytes()); // ethertype ARP
        frame.extend_from_slice(&[0x00, 0x01, 0x08, 0x00, 0x06, 0x04]); // htype/ptype/hlen/plen
        frame.extend_from_slice(&1u16.to_be_bytes()); // ARP_REQUEST
        frame.extend_from_slice(&PEER_MAC); // sha
        frame.extend_from_slice(&[10, 0, 0, 1]); // spa
        frame.extend_from_slice(&[0; 6]); // tha unknown
        frame.extend_from_slice(&OUR_IP); // tpa = who-has us
        frame
    }

    /// The worked-example TX proof: a hand-built guest_memory image
    /// carrying one published chain (hdr + ARP-request frame) drains
    /// through `NetTransport::drain_tx` byte-exact, and the stripped frame
    /// — handed to the real `proxima_net::stack::handle_frame` seam, the
    /// same function `proxima_net`'s xdp/dpdk backends call from their own
    /// pump loops — is answered with a reply, proving the bridge from this
    /// slice's virtio ring codec into proxima-net's sans-IO stack is a real
    /// composition, not a synthesized shortcut.
    #[test]
    fn drained_arp_frame_is_answered_by_the_real_proxima_net_stack() {
        let frame = arp_request_frame();
        let mut guest_memory = guest_memory_with_one_tx_chain(&frame);

        let mut transport = NetTransport::new(OUR_MAC);
        bring_up(&mut transport, 0x1000, 0x2000, 0x3000);

        let mut received: Vec<DrainedFrame> = Vec::new();
        let delivered = transport
            .drain_tx(TX_QUEUE, &mut guest_memory, &mut |drained: DrainedFrame| {
                received.push(drained);
            })
            .expect("chain decodes and drains");
        assert_eq!(delivered, 1);
        assert_eq!(received.len(), 1);
        assert_eq!(received[0].num_buffers, 1);
        assert_eq!(
            received[0].frame, frame,
            "hdr stripped, frame bytes untouched"
        );

        let mut answered = received.pop().expect("one frame").frame;
        let action = stack::handle_frame(&mut answered, OUR_MAC, OUR_IP);
        assert_eq!(
            action,
            stack::Action::Transmit,
            "our stack answers the ARP request"
        );
        assert_eq!(&answered[0..6], &PEER_MAC, "reply eth dst is the requester");
        assert_eq!(&answered[6..12], &OUR_MAC, "reply eth src is ours");

        // used-ring completion published: idx advanced to 1, slot 0 = {id:0, len:0}.
        let used_idx = u16::from_le_bytes([guest_memory[0x3002], guest_memory[0x3003]]);
        assert_eq!(used_idx, 1);
        let used_id = u32::from_le_bytes(guest_memory[0x3004..0x3008].try_into().unwrap());
        let used_len = u32::from_le_bytes(
            guest_memory[0x3004 + 4..0x3004 + USED_ELEM_LEN]
                .try_into()
                .unwrap(),
        );
        assert_eq!(used_id, 0);
        assert_eq!(used_len, 0, "transmit-only queue: device wrote 0 bytes");
    }

    /// A TCP SYN drained off the ring reaches a real `TcpStack::on_inbound`
    /// call, proving the bridge covers the connection-oriented path too,
    /// not only the ARP/ICMP responder — mirrors
    /// `proxima_net::xdp::stream_listener`'s own `classify_tcp` +
    /// `TcpStack::on_inbound` composition (`proxima-net/src/xdp/
    /// stream_listener.rs:201-206`), reimplemented here at the scale this
    /// slice needs rather than importing that module's private helper.
    #[test]
    fn drained_tcp_syn_reaches_a_real_tcp_stack_and_gets_a_synack() {
        use proxima_protocols::inet::ethernet::{self, EtherType};
        use proxima_protocols::inet::ipv4::{self, Ipv4Header, Ipv4Protocol};
        use proxima_protocols::inet::tcp::{self, TcpFlags, TcpHeader};

        const ETH: usize = 14;
        const IP: usize = 20;
        const TCP: usize = 20;
        let mut frame = vec![0u8; ETH + IP + TCP];
        ethernet::write_header(&mut frame[..ETH], OUR_MAC, PEER_MAC, EtherType::Ipv4).unwrap();
        ipv4::write_header(
            &mut frame[ETH..ETH + IP],
            [10, 0, 0, 1],
            OUR_IP,
            Ipv4Protocol::Tcp,
            64,
            TCP as u16,
            0,
        )
        .unwrap();
        tcp::write_header(
            &mut frame[ETH + IP..],
            [10, 0, 0, 1],
            OUR_IP,
            51000,
            80,
            0x1000,
            0,
            TcpFlags {
                syn: true,
                ..TcpFlags::default()
            },
            65535,
            &[],
        )
        .unwrap();

        let mut guest_memory = guest_memory_with_one_tx_chain(&frame);
        let mut transport = NetTransport::new(OUR_MAC);
        bring_up(&mut transport, 0x1000, 0x2000, 0x3000);

        let mut received = Vec::new();
        transport
            .drain_tx(TX_QUEUE, &mut guest_memory, &mut |drained: DrainedFrame| {
                received.push(drained.frame);
            })
            .expect("chain decodes and drains");
        let drained_frame = received.pop().expect("one frame");
        assert_eq!(drained_frame, frame);

        let action = stack::handle_frame(&mut drained_frame.clone(), OUR_MAC, OUR_IP);
        assert_eq!(
            action,
            stack::Action::Drop,
            "the L2/L3 responder never answers TCP"
        );

        let eth = ethernet::EthernetFrame::parse(&drained_frame).expect("valid eth header");
        let ip = Ipv4Header::parse(eth.payload()).expect("valid ipv4 header");
        let tcp_header = TcpHeader::parse(ip.payload()).expect("valid tcp header");

        let mut stack_under_test = TcpStack::new(OUR_IP, 80, 0x9000);
        let inbound = proxima_net::tcp_listener::Inbound {
            source_mac: PEER_MAC,
            source_ip: [10, 0, 0, 1],
            source_port: 51000,
            flags: tcp_header.flags(),
            seq: tcp_header.sequence(),
            ack: tcp_header.acknowledgement(),
            window: tcp_header.window(),
            payload: tcp_header.payload(),
        };
        let outbound = stack_under_test.on_inbound(&inbound, TcpInstant::from_micros(0));
        assert_eq!(outbound.len(), 1, "one SYN-ACK reply");
        let (peer, segment) = &outbound[0];
        assert_eq!(peer.ip, [10, 0, 0, 1]);
        assert_eq!(peer.port, 51000);
        assert!(
            segment.flags.syn && segment.flags.ack,
            "handshake reply is SYN+ACK"
        );
    }
}
