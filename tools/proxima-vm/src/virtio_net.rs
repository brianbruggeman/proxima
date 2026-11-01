//! Host-side virtio-net device model: owns one
//! [`proxima_protocols::virtio::NetConfigSpace`] (the sans-IO device codec)
//! plus the ring cursors of both queues, and reads and writes real guest
//! memory once a `QueueNotify` effect or a pump names a queue — the same "one
//! layer up" split `virtio_console::ConsoleTransport` uses (MMIO register
//! decode plus guest-physical-to-host-pointer translation live here, never
//! inside the sans-IO codec itself).
//!
//! Per VIRTIO 1.2 spec §5.1.2, a non-multiqueue net device exposes exactly
//! two queues: receiveq1 = 0, transmitq1 = 1. Both directions are steady
//! state:
//!
//! - transmitq1 ([`NetTransport::drain_tx`]): each chain is a
//!   `virtio_net_hdr` followed by one Ethernet frame; the header is stripped
//!   and the frame is handed to the caller's sink and, when the transport is
//!   wired, pushed to its egress queue (the sink-pipe side of the device).
//! - receiveq1 ([`NetTransport::fill_rx`]): frames popped from the ingress
//!   queue (the source-pipe side) are written as `virtio_net_hdr` + frame into
//!   the next buffer the driver posted, and completed on the used ring.
//!
//! The device reaches its peer only through those two pipes
//! (`crate::frame_pipe`), never through a buffer or callback of its own.

use futures::FutureExt;
use proxima_primitives::pipe::Pipe;
use proxima_protocols::virtio::{
    AvailRing, DESC_LEN, DecodeError, DescriptorChain, MmioAccess, MmioEffect,
    MmioError, NET_HDR_LEN, NetConfigSpace, NetHdr, NetHdrFields, RingCursor, USED_ELEM_LEN,
    RECEIVEQ1, TRANSMITQ1, UsedElem, write_net_hdr, write_used_elem,
};

use crate::frame_pipe::{FrameError, FrameSink, FrameSource, frame_channel};

/// receiveq1 — host-to-guest delivery.
pub const RX_QUEUE: u16 = RECEIVEQ1;
/// transmitq1 — guest-to-host delivery.
pub const TX_QUEUE: u16 = TRANSMITQ1;
/// `QueueNumMax` this device advertises for both queues. The Linux driver stops a
/// transmit queue when fewer than 2 + MAX_SKB_FRAGS (19) descriptors are free, so a
/// ring smaller than that never wakes again after its first frame.
pub const QUEUE_NUM_MAX: u16 = 256;

/// Why a queue walk could not complete.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainError {
    /// The queue named by a `QueueNotify` is not the transmit queue.
    NotTransmitQueue { queue: u16 },
    /// The queue's size or descriptor/driver/device addresses are not all
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
            Self::QueueNotConfigured => write!(formatter, "queue not yet programmed"),
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

/// One TX chain's payload once its `virtio_net_hdr` has been stripped.
pub struct DrainedFrame {
    pub num_buffers: u16,
    pub frame: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
struct Rings {
    descriptor: u64,
    driver: u64,
    device: u64,
    size: u16,
}

/// Owns the net device's mmio register-block FSM (via `NetConfigSpace`) and
/// both queues' avail/used positions. The device's whole world is two pipes:
/// `ingress` is a source (`() -> frame`, host-to-guest) and `egress` a sink
/// (`frame -> ()`, guest-to-host). It knows nothing about what is behind
/// them — an in-process test peer, a recorder, or a proxy hop. `Clone` for the
/// M7 snapshot reason `ConsoleTransport` derives it.
#[derive(Debug, Clone)]
pub struct NetTransport<Ingress, Egress> {
    device: NetConfigSpace,
    tx_avail: u16,
    tx_used: u16,
    rx_avail: u16,
    rx_used: u16,
    ingress: Ingress,
    egress: Egress,
    rx_delivered: u64,
    tx_forwarded: u64,
    frames_dropped: u64,
}

/// The instantiation the C exit loop's `void *` casts back to
/// (`extern "C"` trampolines cannot be generic). Its pipes are the ends of
/// [`frame_channel`]s; whatever plays the peer holds the other ends.
pub type HostNetTransport = NetTransport<FrameSource, FrameSink>;

impl NetTransport<FrameSource, FrameSink> {
    /// A transport with no peer: the ingress source never has a frame and the
    /// egress sink's far end is dropped.
    #[must_use]
    pub fn new(mac: [u8; 6]) -> Self {
        let (_unused_sink, ingress) = frame_channel(1);
        let (egress, _unused_source) = frame_channel(1);
        Self::with_pipes(mac, ingress, egress)
    }
}

impl<Ingress, Egress> NetTransport<Ingress, Egress>
where
    Ingress: Pipe<In = (), Out = Vec<u8>, Err = FrameError>,
    Egress: Pipe<In = Vec<u8>, Out = (), Err = FrameError>,
{
    #[must_use]
    pub fn with_pipes(mac: [u8; 6], ingress: Ingress, egress: Egress) -> Self {
        Self {
            device: NetConfigSpace::new(mac, QUEUE_NUM_MAX),
            tx_avail: 0,
            tx_used: 0,
            rx_avail: 0,
            rx_used: 0,
            ingress,
            egress,
            rx_delivered: 0,
            tx_forwarded: 0,
            frames_dropped: 0,
        }
    }

    #[must_use]
    pub fn rx_delivered(&self) -> u64 {
        self.rx_delivered
    }

    #[must_use]
    pub fn tx_forwarded(&self) -> u64 {
        self.tx_forwarded
    }

    #[must_use]
    pub fn frames_dropped(&self) -> u64 {
        self.frames_dropped
    }

    /// Apply one raw register access recovered from a trapped guest
    /// load/store — delegates to [`NetConfigSpace::apply`].
    pub fn apply(&mut self, access: MmioAccess) -> Result<MmioEffect, MmioError> {
        self.device.apply(access)
    }

    #[must_use]
    pub fn interrupt_pending(&self) -> bool {
        self.device.interrupt_pending()
    }

    pub fn config_read(&self, offset: u64, length: usize) -> Result<u64, MmioError> {
        self.device.read_config(offset, length)
    }

    /// Walks every avail-ring entry published on the transmit queue since the
    /// last drain: each chain is one `virtio_net_hdr` followed by one
    /// Ethernet frame (concatenated across however many descriptors the
    /// driver split it into), sent out the egress sink pipe with the header
    /// stripped and also returned, for callers that want the bytes
    /// themselves. Publishes one used-ring completion per chain (`len = 0`:
    /// nothing written to a read-only chain, spec §2.7.8).
    pub fn drain_tx(
        &mut self,
        queue: u16,
        guest_memory: &mut [u8],
    ) -> Result<Vec<DrainedFrame>, DrainError> {
        if queue != TX_QUEUE {
            return Err(DrainError::NotTransmitQueue { queue });
        }
        let rings = self.rings(TX_QUEUE)?;
        let mut drained = Vec::new();
        while let Some(head) = pending_head(guest_memory, rings, self.tx_avail)? {
            self.tx_avail = self.tx_avail.wrapping_add(1);
            let frame = read_tx_chain(guest_memory, rings, head)?;
            self.forward(&frame.frame);
            drained.push(frame);
            complete(guest_memory, rings, &mut self.tx_used, head, 0)?;
            self.device.raise_used_buffer();
        }
        Ok(drained)
    }

    /// Fills posted receive buffers from the ingress source until either runs
    /// out. A frame too large for its buffer is dropped (and counted) after
    /// the buffer is completed with `len = 0`. When the source reports
    /// [`FrameError::Closed`] the link goes down, which the driver reads from
    /// config space. Returns how many frames were delivered; a receive queue
    /// the driver has not set up yet delivers none.
    pub fn fill_rx(&mut self, guest_memory: &mut [u8]) -> Result<usize, DrainError> {
        let rings = match self.rings(RX_QUEUE) {
            Ok(rings) => rings,
            Err(DrainError::QueueNotConfigured) => return Ok(0),
            Err(error) => return Err(error),
        };
        let mut delivered = 0usize;
        while let Some(head) = pending_head(guest_memory, rings, self.rx_avail)? {
            let frame = match self.ingress.call(()).now_or_never() {
                Some(Ok(frame)) => frame,
                Some(Err(FrameError::Closed)) => {
                    self.device.set_link_up(false);
                    break;
                }
                _ => break,
            };
            self.rx_avail = self.rx_avail.wrapping_add(1);
            let written = write_rx_frame(guest_memory, rings, head, &frame)?;
            complete(guest_memory, rings, &mut self.rx_used, head, written)?;
            self.device.raise_used_buffer();
            match written {
                0 => self.frames_dropped += 1,
                _ => {
                    self.rx_delivered += 1;
                    delivered += 1;
                }
            }
        }
        Ok(delivered)
    }

    fn forward(&mut self, frame: &[u8]) {
        match self.egress.call(frame.to_vec()).now_or_never() {
            Some(Ok(())) => self.tx_forwarded += 1,
            _ => self.frames_dropped += 1,
        }
    }

    fn rings(&self, queue: u16) -> Result<Rings, DrainError> {
        let transport = self.device.transport();
        let address = |lookup: Option<u64>| {
            lookup
                .filter(|address| *address != 0)
                .ok_or(DrainError::QueueNotConfigured)
        };
        let size = transport
            .queue_size(queue)
            .filter(|size| *size != 0)
            .ok_or(DrainError::QueueNotConfigured)?;
        Ok(Rings {
            descriptor: address(transport.queue_descriptor_address(queue))?,
            driver: address(transport.queue_driver_address(queue))?,
            device: address(transport.queue_device_address(queue))?,
            size,
        })
    }
}

fn pending_head(
    guest_memory: &[u8],
    rings: Rings,
    avail_position: u16,
) -> Result<Option<u16>, DrainError> {
    let avail_bytes = slice_at(guest_memory, rings.driver, 4 + usize::from(rings.size) * 2)?;
    let avail = AvailRing::parse(avail_bytes, rings.size).map_err(DrainError::Decode)?;
    let cursor = RingCursor::resume(rings.size, avail_position).map_err(DrainError::Decode)?;
    if cursor.pending(avail.idx()) == 0 {
        return Ok(None);
    }
    Ok(Some(avail.ring_entry(avail_position)))
}

fn chain_links(
    guest_memory: &[u8],
    rings: Rings,
    head: u16,
) -> Result<Vec<(u64, u32, bool)>, DrainError> {
    let table = slice_at(
        guest_memory,
        rings.descriptor,
        usize::from(rings.size) * DESC_LEN,
    )?;
    DescriptorChain::new(table, rings.size, head)
        .map(|link| {
            let link = link.map_err(DrainError::Decode)?;
            Ok((link.addr(), link.buffer_len(), link.flags().device_writable()))
        })
        .collect()
}

fn read_tx_chain(
    guest_memory: &[u8],
    rings: Rings,
    head: u16,
) -> Result<DrainedFrame, DrainError> {
    let mut chain_bytes = Vec::new();
    for (addr, len, _) in chain_links(guest_memory, rings, head)? {
        chain_bytes.extend_from_slice(slice_at(guest_memory, addr, len as usize)?);
    }
    let hdr_bytes = chain_bytes
        .get(..NET_HDR_LEN)
        .ok_or(DrainError::HeaderTooShort {
            got: chain_bytes.len(),
        })?;
    let hdr = NetHdr::parse(hdr_bytes).map_err(DrainError::Decode)?;
    Ok(DrainedFrame {
        num_buffers: hdr.num_buffers(),
        frame: chain_bytes[NET_HDR_LEN..].to_vec(),
    })
}

fn write_rx_frame(
    guest_memory: &mut [u8],
    rings: Rings,
    head: u16,
    frame: &[u8],
) -> Result<u32, DrainError> {
    let writable: Vec<(u64, u32)> = chain_links(guest_memory, rings, head)?
        .into_iter()
        .filter(|(_, _, device_writable)| *device_writable)
        .map(|(addr, len, _)| (addr, len))
        .collect();
    let capacity: usize = writable.iter().map(|(_, len, ..)| *len as usize).sum();
    let mut payload = vec![0u8; NET_HDR_LEN];
    write_net_hdr(
        &mut payload,
        NetHdrFields {
            num_buffers: 1,
            ..NetHdrFields::default()
        },
    )
    .map_err(DrainError::Decode)?;
    payload.extend_from_slice(frame);
    if payload.len() > capacity {
        return Ok(0);
    }
    let mut remaining = payload.as_slice();
    for (addr, len) in writable {
        let (chunk, rest) = remaining.split_at(remaining.len().min(len as usize));
        mut_slice_at(guest_memory, addr, chunk.len())?.copy_from_slice(chunk);
        remaining = rest;
    }
    Ok(payload.len() as u32)
}

fn complete(
    guest_memory: &mut [u8],
    rings: Rings,
    used_position: &mut u16,
    head: u16,
    len: u32,
) -> Result<(), DrainError> {
    let used_offset = 4 + usize::from(*used_position % rings.size) * USED_ELEM_LEN;
    let used_slot = mut_slice_at(
        guest_memory,
        rings.device + used_offset as u64,
        USED_ELEM_LEN,
    )?;
    write_used_elem(
        used_slot,
        UsedElem {
            id: u32::from(head),
            len,
        },
    )
    .map_err(DrainError::Decode)?;
    *used_position = used_position.wrapping_add(1);
    mut_slice_at(guest_memory, rings.device + 2, 2)?.copy_from_slice(&used_position.to_le_bytes());
    Ok(())
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
    use proxima_protocols::virtio::mmio::{
        REG_DRIVER_FEATURES, REG_DRIVER_FEATURES_SEL, REG_QUEUE_DESC_HIGH, REG_QUEUE_DESC_LOW,
        REG_QUEUE_DEVICE_HIGH, REG_QUEUE_DEVICE_LOW, REG_QUEUE_DRIVER_HIGH, REG_QUEUE_DRIVER_LOW,
        REG_QUEUE_NUM, REG_QUEUE_READY, REG_QUEUE_SEL, REG_STATUS,
    };
    use proxima_protocols::virtio::status::{
        STATUS_ACKNOWLEDGE, STATUS_DRIVER, STATUS_DRIVER_OK, STATUS_FEATURES_OK,
    };
    use proxima_protocols::virtio::{DriverQueue, UsedRing};

    use crate::frame_pipe::{FrameSink, FrameSource};

    const OUR_MAC: [u8; 6] = [0x02, 0x00, 0x00, 0x00, 0x00, 0x01];
    const PEER_MAC: [u8; 6] = [0x02, 0x11, 0x22, 0x33, 0x44, 0x55];
    const OUR_IP: [u8; 4] = [10, 0, 0, 2];
    const TEST_QUEUE_SIZE: u16 = 4;

    const TX_TABLE: u64 = 0x1000;
    const TX_AVAIL: u64 = 0x2000;
    const TX_USED: u64 = 0x3000;
    const RX_TABLE: u64 = 0x4000;
    const RX_AVAIL: u64 = 0x5000;
    const RX_USED: u64 = 0x6000;
    const BUFFERS: u64 = 0x8000;
    const MEMORY_SIZE: usize = 0x10000;

    fn write(offset: u64, value: u32) -> MmioAccess {
        MmioAccess {
            offset,
            is_write: true,
            value,
        }
    }

    fn negotiate(transport: &mut HostNetTransport) {
        for status in [
            STATUS_ACKNOWLEDGE,
            STATUS_ACKNOWLEDGE | STATUS_DRIVER,
        ] {
            transport
                .apply(write(REG_STATUS, u32::from(status)))
                .unwrap();
        }
        for (select, word) in [(0, 0), (1, 1)] {
            transport.apply(write(REG_DRIVER_FEATURES_SEL, select)).unwrap();
            transport.apply(write(REG_DRIVER_FEATURES, word)).unwrap();
        }
        let features_ok = STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK;
        transport
            .apply(write(REG_STATUS, u32::from(features_ok)))
            .unwrap();
    }

    fn set_up_queue(
        transport: &mut HostNetTransport,
        queue: u16,
        table: u64,
        avail: u64,
        used: u64,
    ) {
        let registers = [
            (REG_QUEUE_SEL, u32::from(queue)),
            (REG_QUEUE_NUM, u32::from(TEST_QUEUE_SIZE)),
            (REG_QUEUE_DESC_LOW, table as u32),
            (REG_QUEUE_DESC_HIGH, 0),
            (REG_QUEUE_DRIVER_LOW, avail as u32),
            (REG_QUEUE_DRIVER_HIGH, 0),
            (REG_QUEUE_DEVICE_LOW, used as u32),
            (REG_QUEUE_DEVICE_HIGH, 0),
            (REG_QUEUE_READY, 1),
        ];
        for (offset, value) in registers {
            transport.apply(write(offset, value)).unwrap();
        }
    }

    fn driver_ok(transport: &mut HostNetTransport) {
        let all = STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK | STATUS_DRIVER_OK;
        transport.apply(write(REG_STATUS, u32::from(all))).unwrap();
    }

    fn net_hdr_bytes() -> [u8; NET_HDR_LEN] {
        let mut hdr = [0u8; NET_HDR_LEN];
        write_net_hdr(
            &mut hdr,
            NetHdrFields {
                num_buffers: 1,
                ..NetHdrFields::default()
            },
        )
        .unwrap();
        hdr
    }

    fn arp_request_frame() -> Vec<u8> {
        let mut frame = vec![0xff; 6];
        frame.extend_from_slice(&PEER_MAC);
        frame.extend_from_slice(&0x0806u16.to_be_bytes());
        frame.extend_from_slice(&[0x00, 0x01, 0x08, 0x00, 0x06, 0x04]);
        frame.extend_from_slice(&1u16.to_be_bytes());
        frame.extend_from_slice(&PEER_MAC);
        frame.extend_from_slice(&[10, 0, 0, 1]);
        frame.extend_from_slice(&[0; 6]);
        frame.extend_from_slice(&OUR_IP);
        frame
    }

    struct Guest {
        memory: Vec<u8>,
        tx: DriverQueue<4>,
        rx: DriverQueue<4>,
    }

    impl Guest {
        fn new() -> Self {
            let mut memory = vec![0u8; MEMORY_SIZE];
            let tx = DriverQueue::<4>::new(&mut memory[TX_TABLE as usize..]).unwrap();
            let rx = DriverQueue::<4>::new(&mut memory[RX_TABLE as usize..]).unwrap();
            Self { memory, tx, rx }
        }

        fn transmit(&mut self, frame: &[u8]) -> u16 {
            let address = BUFFERS;
            let mut payload = net_hdr_bytes().to_vec();
            payload.extend_from_slice(frame);
            self.memory[address as usize..address as usize + payload.len()]
                .copy_from_slice(&payload);
            let (table, avail) = (TX_TABLE as usize, TX_AVAIL as usize);
            let (before, after) = self.memory.split_at_mut(avail);
            self.tx
                .post(
                    &mut before[table..],
                    after,
                    &[(address, payload.len() as u32)],
                    &[],
                )
                .unwrap()
        }

        fn post_rx_buffer(&mut self, address: u64, len: u32) -> u16 {
            let (table, avail) = (RX_TABLE as usize, RX_AVAIL as usize);
            let (before, after) = self.memory.split_at_mut(avail);
            self.rx
                .post(&mut before[table..], after, &[], &[(address, len)])
                .unwrap()
        }

        fn reap_rx(&mut self) -> Option<(UsedElem, u64)> {
            let used = self.memory[RX_USED as usize..RX_USED as usize + 4 + 4 * USED_ELEM_LEN]
                .to_vec();
            self.rx
                .reap(&mut self.memory[RX_TABLE as usize..], &used)
                .unwrap()
        }
    }

    fn wired_transport() -> (HostNetTransport, FrameSource, FrameSink) {
        let (to_guest, ingress) = frame_channel(8);
        let (egress, from_guest) = frame_channel(8);
        let mut transport = NetTransport::with_pipes(OUR_MAC, ingress, egress);
        negotiate(&mut transport);
        set_up_queue(&mut transport, RX_QUEUE, RX_TABLE, RX_AVAIL, RX_USED);
        set_up_queue(&mut transport, TX_QUEUE, TX_TABLE, TX_AVAIL, TX_USED);
        driver_ok(&mut transport);
        (transport, from_guest, to_guest)
    }

    fn send(sink: &FrameSink, frame: Vec<u8>) {
        sink.call(frame).now_or_never().unwrap().unwrap();
    }

    fn receive(source: &FrameSource) -> Option<Vec<u8>> {
        source.call(()).now_or_never().unwrap().ok()
    }

    /// The worked-example TX proof: a driver-codec-published chain (hdr +
    /// ARP-request frame) drains through `NetTransport::drain_tx` byte-exact,
    /// and the stripped frame — handed to the real
    /// `proxima_net::stack::handle_frame` — is answered with a reply.
    #[test]
    fn drained_arp_frame_is_answered_by_the_real_proxima_net_stack() {
        let frame = arp_request_frame();
        let mut guest = Guest::new();
        let mut transport = NetTransport::new(OUR_MAC);
        negotiate(&mut transport);
        set_up_queue(&mut transport, TX_QUEUE, TX_TABLE, TX_AVAIL, TX_USED);
        driver_ok(&mut transport);
        guest.transmit(&frame);

        let mut received = transport
            .drain_tx(TX_QUEUE, &mut guest.memory)
            .expect("chain decodes and drains");

        assert_eq!(received.len(), 1);
        assert_eq!(received[0].num_buffers, 1);
        assert_eq!(received[0].frame, frame, "hdr stripped, frame untouched");
        let mut answered = received.pop().expect("one frame").frame;
        let action = stack::handle_frame(&mut answered, OUR_MAC, OUR_IP);
        assert_eq!(action, stack::Action::Transmit);
        assert_eq!(&answered[0..6], &PEER_MAC);
        assert_eq!(&answered[6..12], &OUR_MAC);

        let used = UsedRing::parse(&guest.memory[TX_USED as usize..], TEST_QUEUE_SIZE).unwrap();
        assert_eq!(used.idx(), 1);
        assert_eq!(used.ring_entry(0), UsedElem { id: 0, len: 0 });
    }

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
        let mut guest = Guest::new();
        let mut transport = NetTransport::new(OUR_MAC);
        negotiate(&mut transport);
        set_up_queue(&mut transport, TX_QUEUE, TX_TABLE, TX_AVAIL, TX_USED);
        driver_ok(&mut transport);
        guest.transmit(&frame);

        let mut received = transport
            .drain_tx(TX_QUEUE, &mut guest.memory)
            .expect("chain decodes and drains");
        let drained_frame = received.pop().expect("one frame").frame;
        assert_eq!(drained_frame, frame);

        let action = stack::handle_frame(&mut drained_frame.clone(), OUR_MAC, OUR_IP);
        assert_eq!(action, stack::Action::Drop);
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
        assert!(outbound[0].1.flags.syn && outbound[0].1.flags.ack);
    }

    #[test]
    fn tx_frames_reach_the_egress_sink_pipe() {
        let (mut transport, from_guest, _to_guest) = wired_transport();
        let mut guest = Guest::new();
        guest.transmit(&arp_request_frame());

        transport.drain_tx(TX_QUEUE, &mut guest.memory).expect("drains");

        assert_eq!(
            receive(&from_guest).expect("one forwarded frame"),
            arp_request_frame()
        );
        assert_eq!(transport.tx_forwarded(), 1);
    }

    /// Host-to-guest: an ingress frame lands as `virtio_net_hdr` + frame in
    /// the buffer the driver codec posted, and the driver codec reaps the
    /// completion with `len = 12 + frame`.
    #[test]
    fn ingress_frame_fills_the_next_posted_receive_buffer() {
        let (mut transport, _from_guest, to_guest) = wired_transport();
        let mut guest = Guest::new();
        let buffer = BUFFERS + 0x1000;
        guest.post_rx_buffer(buffer, 2048);
        let frame = arp_request_frame();
        send(&to_guest, frame.clone());

        let delivered = transport.fill_rx(&mut guest.memory).expect("fills");

        assert_eq!(delivered, 1);
        let (elem, first_addr) = guest.reap_rx().expect("completion published");
        assert_eq!(first_addr, buffer);
        assert_eq!(elem.len as usize, NET_HDR_LEN + frame.len());
        let start = buffer as usize;
        assert_eq!(&guest.memory[start..start + NET_HDR_LEN], &net_hdr_bytes());
        assert_eq!(
            &guest.memory[start + NET_HDR_LEN..start + NET_HDR_LEN + frame.len()],
            &frame[..]
        );
        assert_eq!(transport.rx_delivered(), 1);
    }

    #[test]
    fn ingress_frame_waits_when_no_receive_buffer_is_posted() {
        let (mut transport, _from_guest, to_guest) = wired_transport();
        let mut guest = Guest::new();
        send(&to_guest, arp_request_frame());

        let before = transport.fill_rx(&mut guest.memory).expect("no-op");
        guest.post_rx_buffer(BUFFERS + 0x1000, 2048);
        let after = transport.fill_rx(&mut guest.memory).expect("fills");

        assert_eq!((before, after), (0, 1), "frame waits in the source pipe");
    }

    #[test]
    fn ingress_frame_larger_than_its_buffer_is_dropped_and_counted() {
        let (mut transport, _from_guest, to_guest) = wired_transport();
        let mut guest = Guest::new();
        guest.post_rx_buffer(BUFFERS + 0x1000, 32);
        send(&to_guest, arp_request_frame());

        let delivered = transport.fill_rx(&mut guest.memory).expect("drops");

        assert_eq!(delivered, 0);
        assert_eq!(transport.frames_dropped(), 1);
        let (elem, _) = guest.reap_rx().expect("buffer is returned");
        assert_eq!(elem.len, 0);
    }

    #[test]
    fn closing_the_ingress_pipe_takes_the_link_down_in_config_space() {
        let (mut transport, _from_guest, to_guest) = wired_transport();
        let mut guest = Guest::new();
        guest.post_rx_buffer(BUFFERS + 0x1000, 2048);
        let status_register = MmioAccess {
            offset: proxima_protocols::virtio::CONFIG_SPACE_BASE + 4,
            is_write: false,
            value: 0,
        };
        let up = transport.apply(status_register).unwrap();
        to_guest.close();
        transport.fill_rx(&mut guest.memory).expect("observes the close");
        let down = transport.apply(status_register).unwrap();

        assert_eq!(up, MmioEffect::ReadValue(0x0001_0100));
        assert_eq!(down, MmioEffect::ReadValue(0x0000_0100));
        assert!(transport.interrupt_pending(), "link change raises config-change");
    }

    #[test]
    fn fill_rx_raises_used_buffer_only_when_it_completed_an_entry() {
        let (mut transport, _from_guest, to_guest) = wired_transport();
        let mut guest = Guest::new();
        guest.post_rx_buffer(BUFFERS + 0x1000, 2048);

        transport.fill_rx(&mut guest.memory).expect("nothing to deliver");
        assert!(!transport.interrupt_pending());

        send(&to_guest, arp_request_frame());
        transport.fill_rx(&mut guest.memory).expect("fills");
        assert!(transport.interrupt_pending());
    }

    #[test]
    fn drain_tx_raises_used_buffer_after_completing_a_chain() {
        let (mut transport, _from_guest, _to_guest) = wired_transport();
        let mut guest = Guest::new();

        transport.drain_tx(TX_QUEUE, &mut guest.memory).expect("empty ring");
        assert!(!transport.interrupt_pending());

        guest.transmit(&arp_request_frame());
        transport.drain_tx(TX_QUEUE, &mut guest.memory).expect("drains");
        assert!(transport.interrupt_pending());
    }

    #[test]
    fn config_read_serves_the_mac_bytewise_through_the_transport() {
        let (transport, _from_guest, _to_guest) = wired_transport();
        let base = proxima_protocols::virtio::CONFIG_SPACE_BASE;

        let bytes: Vec<u8> = (0..6)
            .map(|index| transport.config_read(base + index, 1).unwrap() as u8)
            .collect();

        assert_eq!(bytes, OUR_MAC);
        assert_eq!(transport.config_read(base + 6, 2).unwrap(), 1);
    }
}
