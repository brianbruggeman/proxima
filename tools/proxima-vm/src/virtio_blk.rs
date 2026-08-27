//! Host-side virtio-blk transport: owns one
//! [`proxima_protocols::virtio::BlkConfigSpace`] (the sans-IO device codec)
//! plus `requestq`'s ring cursors, and reads/writes real guest memory to
//! service a descriptor chain once a `QueueNotify` effect names the queue —
//! the same "one layer up" split `virtio_console::ConsoleTransport` and
//! `virtio_net::NetTransport` use (MMIO register decode plus guest-
//! physical-to-host-pointer translation live here, never inside the sans-IO
//! codec itself). Mirror, not a redesign, of those two transports' shape.
//!
//! Storage seam: `proxima_storage::nvme::backend::QueueBackend`
//! (`proxima-storage/src/nvme/backend.rs:12`) is doorbell/register-shaped —
//! `write_submission`/`ring_submit_doorbell`/`read_completion` — not a
//! sector-addressed byte range, so it does not fit a `virtio_blk_req`'s
//! `(type, sector, buffer)` shape without inventing a translation layer this
//! slice has no evidence anything needs.
//! `proxima_storage::dax::MappedRegion` (`proxima-storage/src/dax/
//! region.rs:18`, the byte-addressable seam that WOULD fit) is `#[cfg(
//! target_os = "linux")]`-gated (`proxima-storage/src/dax/mod.rs:26-29`) and
//! this host is macOS, so it cannot even compile here. Per this slice's own
//! instructions, the floor is a plain in-memory block store local to the
//! transport — the same locality `virtio_net::NetTransport`'s own module
//! doc names for `FrameSink` (one host loop plus tests, never an open set).
//!
//! Per VIRTIO 1.2 spec §5.2.2, this device exposes exactly one queue,
//! `requestq` (index 0) — no multiqueue, matching M6's exit criterion of
//! proving one IN and one OUT request cross the ring, not a spec-complete
//! block device.

use proxima_protocols::virtio::blk::{
    BLK_REQ_HEADER_LEN, BlkConfigSpace, BlkReqHeader, RequestType, STATUS_IOERR, STATUS_OK,
    STATUS_UNSUPP,
};
use proxima_protocols::virtio::{
    AvailRing, DecodeError, DescriptorChain, MmioAccess, MmioEffect, MmioError, RingCursor,
    UsedElem, write_used_elem,
};

/// `requestq` (spec §5.2.2) — the only queue this device exposes.
pub const REQUEST_QUEUE: u16 = 0;
const QUEUE_SIZE: u16 = 4;
/// Sector size this codec models (spec §5.2.4: `capacity` is always counted
/// in 512-byte units regardless of the backing store's real block size).
pub const SECTOR_LEN: usize = 512;

const _: () = assert!(QUEUE_SIZE != 0 && QUEUE_SIZE.is_power_of_two());

/// Why [`BlkTransport::service_queue`] could not complete — mirrors
/// `virtio_net::DrainError` exactly; this device never surfaces a request
/// failure as a `DrainError` (an out-of-range sector or unsupported type
/// is a valid *virtio* outcome, answered with a status byte, not a host
/// panic or a dropped chain — spec §5.2.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainError {
    /// The queue named by a `QueueNotify` is not `requestq`.
    NotRequestQueue { queue: u16 },
    /// The queue's descriptor/driver/device addresses are not all
    /// programmed yet (a notify arrived before queue setup finished).
    QueueNotConfigured,
    /// A ring or descriptor-chain byte range fell outside `guest_memory`.
    OutOfBounds { need: usize, have: usize },
    /// The ring or chain bytes themselves failed to decode.
    Decode(DecodeError),
    /// A published chain carried fewer descriptors than the fixed
    /// header+status (and, for IN/OUT, +data) shape this transport expects.
    MalformedChain,
}

impl core::fmt::Display for DrainError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotRequestQueue { queue } => {
                write!(formatter, "queue {queue} is not the request queue")
            }
            Self::QueueNotConfigured => write!(formatter, "queue addresses not yet programmed"),
            Self::OutOfBounds { need, have } => write!(
                formatter,
                "guest memory access needs {need} bytes, only {have} available"
            ),
            Self::Decode(inner) => write!(formatter, "{inner}"),
            Self::MalformedChain => write!(
                formatter,
                "chain did not carry a header and a trailing 1-byte status descriptor"
            ),
        }
    }
}

impl std::error::Error for DrainError {}

/// One serviced request, reported back to the caller for tests/telemetry —
/// the status byte the device wrote, which sector it acted on, and (for
/// `IN`/`OUT`) the data bytes actually read from or written to the local
/// store. `data` lets a caller one layer up (the FFI trampoline driving a
/// real guest) prove the bytes that crossed the ring match what the guest
/// expected, without needing the transport itself to outlive the VM exit
/// loop that owns it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServicedRequest {
    pub sector: u64,
    pub status: u8,
    pub data: Vec<u8>,
}

/// Owns the block device's mmio register-block FSM (via `BlkConfigSpace`),
/// `requestq`'s avail/used cursors, and the local block store this M6 slice
/// uses in place of a real backing file (see module doc for the seam
/// search). `store` is sized `capacity_sectors * SECTOR_LEN` bytes.
///
/// `Clone` for the same M7 reason `ConsoleTransport` derives it: every field
/// is plain, `Clone`-able data, `store` included (a byte-for-byte device
/// snapshot the same way the guest-memory snapshot below is one).
#[derive(Debug, Clone)]
pub struct BlkTransport {
    device: BlkConfigSpace,
    avail_cursor: RingCursor,
    used_cursor: RingCursor,
    store: Vec<u8>,
}

impl BlkTransport {
    /// A freshly reset device backed by `capacity_sectors` sectors of
    /// zeroed storage.
    #[must_use]
    #[allow(
        clippy::expect_used,
        reason = "QUEUE_SIZE's power-of-two invariant is checked at compile time above; RingCursor::new cannot fail for this literal"
    )]
    pub fn new(capacity_sectors: u64) -> Self {
        Self {
            device: BlkConfigSpace::new(capacity_sectors, QUEUE_SIZE),
            avail_cursor: RingCursor::new(QUEUE_SIZE)
                .expect("QUEUE_SIZE is a fixed power of two, proven above"),
            used_cursor: RingCursor::new(QUEUE_SIZE)
                .expect("QUEUE_SIZE is a fixed power of two, proven above"),
            store: vec![0u8; capacity_sectors as usize * SECTOR_LEN],
        }
    }

    /// Seeds sector `sector` with `bytes` (test/setup helper — the "host-
    /// seeded pattern" a guest IN request reads back).
    pub fn seed_sector(&mut self, sector: u64, bytes: &[u8]) {
        let start = sector as usize * SECTOR_LEN;
        let end = start + bytes.len();
        self.store[start..end].copy_from_slice(bytes);
    }

    /// Reads sector `sector` back out of the local store (test helper — the
    /// OUT-request mirror of [`Self::seed_sector`]).
    #[must_use]
    pub fn read_sector(&self, sector: u64) -> &[u8] {
        let start = sector as usize * SECTOR_LEN;
        &self.store[start..start + SECTOR_LEN]
    }

    /// Apply one raw register access recovered from a trapped guest
    /// load/store — delegates straight to [`BlkConfigSpace::apply`].
    pub fn apply(&mut self, access: MmioAccess) -> Result<MmioEffect, MmioError> {
        self.device.apply(access)
    }

    /// Walks every avail-ring entry published on `queue` since this
    /// transport last drained it, services each request against the local
    /// block store, writes the data (IN) and status byte back into guest
    /// memory, and publishes one used-ring completion per chain. Returns the
    /// serviced requests, in publish order.
    pub fn service_queue(
        &mut self,
        queue: u16,
        guest_memory: &mut [u8],
    ) -> Result<Vec<ServicedRequest>, DrainError> {
        if queue != REQUEST_QUEUE {
            return Err(DrainError::NotRequestQueue { queue });
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

        let mut serviced = Vec::new();
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
            let mut links = Vec::new();
            for descriptor in DescriptorChain::new(table_bytes, QUEUE_SIZE, head) {
                let descriptor = descriptor.map_err(DrainError::Decode)?;
                links.push((
                    descriptor.addr(),
                    descriptor.buffer_len(),
                    descriptor.flags().device_writable(),
                ));
            }
            let (header_link, remaining) = links.split_first().ok_or(DrainError::MalformedChain)?;
            let (status_link, data_links) =
                remaining.split_last().ok_or(DrainError::MalformedChain)?;
            if status_link.1 as usize != 1 || !status_link.2 {
                return Err(DrainError::MalformedChain);
            }

            let header_bytes = slice_at(guest_memory, header_link.0, header_link.1 as usize)?;
            if header_bytes.len() < BLK_REQ_HEADER_LEN {
                return Err(DrainError::MalformedChain);
            }
            let sector = BlkReqHeader::parse(header_bytes)
                .map_err(DrainError::Decode)?
                .sector();
            let request_type = BlkReqHeader::parse(header_bytes)
                .map_err(DrainError::Decode)?
                .request_type();

            let (status, data) =
                self.service_one(request_type, sector, data_links, guest_memory)?;

            let status_slot = mut_slice_at(guest_memory, status_link.0, 1)?;
            status_slot[0] = status;

            serviced.push(ServicedRequest {
                sector,
                status,
                data,
            });

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
                    len: 1,
                },
            )
            .map_err(DrainError::Decode)?;
            let new_idx = self.used_cursor.advance();
            let idx_slot = mut_slice_at(guest_memory, device_address + 2, 2)?;
            idx_slot.copy_from_slice(&new_idx.to_le_bytes());
        }
        Ok(serviced)
    }

    /// Services one request against the local store: `IN` copies a sector
    /// out into the data descriptor(s), `OUT` copies the data descriptor(s)
    /// in, `FLUSH` is a no-op (the store has no write buffering to flush).
    /// Returns the status byte to publish plus the data bytes actually moved
    /// (empty for `FLUSH`/`Unsupported`) — never an `Err` for a request
    /// outcome: an out-of-range sector or unsupported type is a legal virtio
    /// outcome ([`STATUS_IOERR`] / [`STATUS_UNSUPP`]), not a host-side
    /// failure.
    fn service_one(
        &mut self,
        request_type: RequestType,
        sector: u64,
        data_links: &[(u64, u32, bool)],
        guest_memory: &mut [u8],
    ) -> Result<(u8, Vec<u8>), DrainError> {
        match request_type {
            RequestType::In => {
                let Some(&(addr, len, writable)) = data_links.first() else {
                    return Ok((STATUS_IOERR, Vec::new()));
                };
                if !writable || !self.sector_in_range(sector, len as usize) {
                    return Ok((STATUS_IOERR, Vec::new()));
                }
                let source = self.read_sector_range(sector, len as usize).to_vec();
                let destination = mut_slice_at(guest_memory, addr, len as usize)?;
                destination.copy_from_slice(&source);
                Ok((STATUS_OK, source))
            }
            RequestType::Out => {
                let Some(&(addr, len, _)) = data_links.first() else {
                    return Ok((STATUS_IOERR, Vec::new()));
                };
                if !self.sector_in_range(sector, len as usize) {
                    return Ok((STATUS_IOERR, Vec::new()));
                }
                let source = slice_at(guest_memory, addr, len as usize)?.to_vec();
                self.write_sector_range(sector, &source);
                Ok((STATUS_OK, source))
            }
            RequestType::Flush => Ok((STATUS_OK, Vec::new())),
            RequestType::Unsupported(_) => Ok((STATUS_UNSUPP, Vec::new())),
        }
    }

    fn sector_in_range(&self, sector: u64, len: usize) -> bool {
        let start = sector as usize * SECTOR_LEN;
        start
            .checked_add(len)
            .is_some_and(|end| end <= self.store.len())
    }

    fn read_sector_range(&self, sector: u64, len: usize) -> &[u8] {
        let start = sector as usize * SECTOR_LEN;
        &self.store[start..start + len]
    }

    fn write_sector_range(&mut self, sector: u64, bytes: &[u8]) {
        let start = sector as usize * SECTOR_LEN;
        self.store[start..start + bytes.len()].copy_from_slice(bytes);
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
    use proxima_protocols::virtio::mmio::{
        REG_DRIVER_FEATURES, REG_DRIVER_FEATURES_SEL, REG_QUEUE_DESC_HIGH, REG_QUEUE_DESC_LOW,
        REG_QUEUE_DEVICE_HIGH, REG_QUEUE_DEVICE_LOW, REG_QUEUE_DRIVER_HIGH, REG_QUEUE_DRIVER_LOW,
        REG_QUEUE_NUM, REG_QUEUE_READY, REG_QUEUE_SEL, REG_STATUS,
    };
    use proxima_protocols::virtio::status::{
        STATUS_ACKNOWLEDGE, STATUS_DRIVER, STATUS_DRIVER_OK, STATUS_FEATURES_OK,
    };
    use proxima_protocols::virtio::{DESC_LEN, USED_ELEM_LEN};

    fn bring_up(
        transport: &mut BlkTransport,
        descriptor_addr: u64,
        driver_addr: u64,
        device_addr: u64,
    ) {
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
            .apply(write(REG_QUEUE_SEL, REQUEST_QUEUE.into()))
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

    // Descriptor table at 0x1000, avail ring at 0x2000, used ring at 0x3000,
    // header at 0x1100, data buffer at 0x1200, status byte at 0x1400 — the
    // same three-region-plus-payload layout `virtio_net`'s own tests use,
    // extended with the third (status) descriptor this device needs.
    fn guest_memory_with_one_request(request_type: u32, sector: u64) -> Vec<u8> {
        let mut memory = vec![0u8; 0x4000];

        let mut desc0 = [0u8; DESC_LEN];
        desc0[0..8].copy_from_slice(&0x1100u64.to_le_bytes());
        desc0[8..12].copy_from_slice(&(BLK_REQ_HEADER_LEN as u32).to_le_bytes());
        desc0[12..14].copy_from_slice(&1u16.to_le_bytes()); // flags = NEXT
        desc0[14..16].copy_from_slice(&1u16.to_le_bytes()); // next = 1
        memory[0x1000..0x1000 + DESC_LEN].copy_from_slice(&desc0);

        let data_writable = request_type == 0; // IN
        let mut desc1 = [0u8; DESC_LEN];
        desc1[0..8].copy_from_slice(&0x1200u64.to_le_bytes());
        desc1[8..12].copy_from_slice(&(SECTOR_LEN as u32).to_le_bytes());
        let flags1: u16 = 1 | if data_writable { 2 } else { 0 }; // NEXT | (WRITE?)
        desc1[12..14].copy_from_slice(&flags1.to_le_bytes());
        desc1[14..16].copy_from_slice(&2u16.to_le_bytes()); // next = 2
        memory[0x1000 + DESC_LEN..0x1000 + 2 * DESC_LEN].copy_from_slice(&desc1);

        let mut desc2 = [0u8; DESC_LEN];
        desc2[0..8].copy_from_slice(&0x1400u64.to_le_bytes());
        desc2[8..12].copy_from_slice(&1u32.to_le_bytes());
        desc2[12..14].copy_from_slice(&2u16.to_le_bytes()); // flags = WRITE
        memory[0x1000 + 2 * DESC_LEN..0x1000 + 3 * DESC_LEN].copy_from_slice(&desc2);

        let mut header = [0u8; BLK_REQ_HEADER_LEN];
        header[0..4].copy_from_slice(&request_type.to_le_bytes());
        header[8..16].copy_from_slice(&sector.to_le_bytes());
        memory[0x1100..0x1100 + BLK_REQ_HEADER_LEN].copy_from_slice(&header);

        // avail ring: idx = 1, ring[0] = head 0.
        memory[0x2000 + 2..0x2000 + 4].copy_from_slice(&1u16.to_le_bytes());

        memory
    }

    /// Worked end-to-end proof: a guest publishes an IN request for sector
    /// 0; the transport copies the host-seeded pattern into the data
    /// descriptor and OK into the status descriptor, and the used ring
    /// records the completion, byte-exact.
    #[test]
    fn in_request_reads_the_host_seeded_sector_into_the_data_descriptor() {
        let mut guest_memory = guest_memory_with_one_request(0, 0);
        let mut transport = BlkTransport::new(4);
        let pattern: Vec<u8> = (0..SECTOR_LEN).map(|index| (index % 251) as u8).collect();
        transport.seed_sector(0, &pattern);
        bring_up(&mut transport, 0x1000, 0x2000, 0x3000);

        let serviced = transport
            .service_queue(REQUEST_QUEUE, &mut guest_memory)
            .expect("chain decodes and services");
        assert_eq!(
            serviced,
            vec![ServicedRequest {
                sector: 0,
                status: STATUS_OK,
                data: pattern.clone()
            }]
        );
        assert_eq!(
            &guest_memory[0x1200..0x1200 + SECTOR_LEN],
            pattern.as_slice()
        );
        assert_eq!(guest_memory[0x1400], STATUS_OK);

        let used_idx = u16::from_le_bytes([guest_memory[0x3002], guest_memory[0x3003]]);
        assert_eq!(used_idx, 1);
        let used_id = u32::from_le_bytes(guest_memory[0x3004..0x3008].try_into().unwrap());
        let used_len = u32::from_le_bytes(
            guest_memory[0x3008..0x3004 + USED_ELEM_LEN]
                .try_into()
                .unwrap(),
        );
        assert_eq!(used_id, 0);
        assert_eq!(used_len, 1);
    }

    /// The OUT mirror: a guest publishes a write; the transport copies the
    /// data descriptor's bytes into the local store, verified by reading it
    /// back through [`BlkTransport::read_sector`].
    #[test]
    fn out_request_writes_the_data_descriptor_into_the_local_store() {
        let mut guest_memory = guest_memory_with_one_request(1, 2);
        let pattern: Vec<u8> = (0..SECTOR_LEN)
            .map(|index| ((index * 3) % 253) as u8)
            .collect();
        guest_memory[0x1200..0x1200 + SECTOR_LEN].copy_from_slice(&pattern);

        let mut transport = BlkTransport::new(4);
        bring_up(&mut transport, 0x1000, 0x2000, 0x3000);

        let serviced = transport
            .service_queue(REQUEST_QUEUE, &mut guest_memory)
            .expect("chain decodes and services");
        assert_eq!(
            serviced,
            vec![ServicedRequest {
                sector: 2,
                status: STATUS_OK,
                data: pattern.clone()
            }]
        );
        assert_eq!(guest_memory[0x1400], STATUS_OK);
        assert_eq!(transport.read_sector(2), pattern.as_slice());
    }

    /// A sector past `capacity_sectors` is answered with `STATUS_IOERR`, not
    /// a panic.
    #[test]
    fn out_of_range_sector_is_answered_with_ioerr_not_a_panic() {
        let mut guest_memory = guest_memory_with_one_request(0, 999);
        let mut transport = BlkTransport::new(4); // only sectors 0..4 exist
        bring_up(&mut transport, 0x1000, 0x2000, 0x3000);

        let serviced = transport
            .service_queue(REQUEST_QUEUE, &mut guest_memory)
            .expect("chain decodes and services");
        assert_eq!(
            serviced,
            vec![ServicedRequest {
                sector: 999,
                status: STATUS_IOERR,
                data: Vec::new()
            }]
        );
        assert_eq!(guest_memory[0x1400], STATUS_IOERR);
    }

    /// An unknown `type` value is answered with `STATUS_UNSUPP`, not acted
    /// on.
    #[test]
    fn unknown_request_type_is_answered_with_unsupp() {
        let mut guest_memory = guest_memory_with_one_request(7, 0);
        let mut transport = BlkTransport::new(4);
        bring_up(&mut transport, 0x1000, 0x2000, 0x3000);

        let serviced = transport
            .service_queue(REQUEST_QUEUE, &mut guest_memory)
            .expect("chain decodes and services");
        assert_eq!(
            serviced,
            vec![ServicedRequest {
                sector: 0,
                status: STATUS_UNSUPP,
                data: Vec::new()
            }]
        );
        assert_eq!(guest_memory[0x1400], STATUS_UNSUPP);
    }

    #[test]
    fn servicing_the_wrong_queue_is_rejected() {
        let mut guest_memory = guest_memory_with_one_request(0, 0);
        let mut transport = BlkTransport::new(4);
        bring_up(&mut transport, 0x1000, 0x2000, 0x3000);
        assert_eq!(
            transport.service_queue(1, &mut guest_memory).unwrap_err(),
            DrainError::NotRequestQueue { queue: 1 }
        );
    }
}
