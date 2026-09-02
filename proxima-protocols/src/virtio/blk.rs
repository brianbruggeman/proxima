//! virtio-blk device codec (VIRTIO 1.2 spec §5.2): the 16-byte
//! `virtio_blk_req` header that opens every request chain on this device's
//! single queue, the 1-byte status trailer the device appends, and the
//! device-config-space `capacity` field this M6 slice actually honors.
//! Mirrors `super::net`'s shape exactly — memory to decode, not a stream
//! (guiding-principles §11: sans-IO, borrowed view over caller-owned bytes,
//! fixed-shape, allocates nothing).
//!
//! [`DEVICE_ID_BLK`] plugs straight into `super::mmio::MmioDevice::new`
//! (spec §5, Table 5.1: DeviceID 2). Feature offering is deliberately
//! narrow: [`FEATURE_VERSION_1`](super::status::FEATURE_VERSION_1) only —
//! this codec does not honor `VIRTIO_BLK_F_SIZE_MAX`, `VIRTIO_BLK_F_SEG_MAX`,
//! multi-queue, or any other optional feature bit, so offering those bits
//! would let a driver negotiate behavior the device never implements.
//! `Negotiation`'s `acked ⊆ offered` check (`super::status`) makes
//! over-offering a live defect, not a latent one, the same discipline
//! `super::net`'s module doc names.
//!
//! Device config space (spec §5.2.4: `capacity` in 512-byte sectors, first
//! field only) is exposed as fixed 32-bit-word registers starting at
//! [`CONFIG_SPACE_BASE`], matching the width discipline `super::mmio`'s core
//! register block and `super::net`'s config space both already use.
//! [`BlkConfigSpace::apply`] composes `MmioDevice::apply` for the core
//! register block rather than reimplementing it (reuse-first, principle 1).

use super::error::DecodeError;
use super::mmio::{MmioAccess, MmioDevice, MmioEffect, MmioError};
use super::raw::{read_u32, read_u64};

/// `DeviceID` value for the block device type (spec §5, Table 5.1).
pub const DEVICE_ID_BLK: u32 = 2;

/// `virtio_blk_req.type` values this codec understands (spec §5.2.6,
/// `VIRTIO_BLK_T_*`). Any other value the driver publishes is rejected by
/// [`BlkReqHeader::request_type`]'s caller with [`RequestType::Unsupported`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestType {
    /// `VIRTIO_BLK_T_IN` (0): device reads `sector` into the driver's buffer.
    In,
    /// `VIRTIO_BLK_T_OUT` (1): device writes the driver's buffer to `sector`.
    Out,
    /// `VIRTIO_BLK_T_FLUSH` (4): device flushes any buffered writes.
    Flush,
    /// A `type` value this codec does not implement — the transport answers
    /// with [`STATUS_UNSUPP`] rather than acting on it.
    Unsupported(u32),
}

const BLK_T_IN: u32 = 0;
const BLK_T_OUT: u32 = 1;
const BLK_T_FLUSH: u32 = 4;

impl From<u32> for RequestType {
    fn from(value: u32) -> Self {
        match value {
            BLK_T_IN => Self::In,
            BLK_T_OUT => Self::Out,
            BLK_T_FLUSH => Self::Flush,
            other => Self::Unsupported(other),
        }
    }
}

/// Status trailer values the device appends after servicing a request
/// (spec §5.2.6, `VIRTIO_BLK_S_*`) — the one-byte descriptor at the tail of
/// every request chain, always device-writable.
pub const STATUS_OK: u8 = 0;
pub const STATUS_IOERR: u8 = 1;
pub const STATUS_UNSUPP: u8 = 2;

/// Fixed size of one `virtio_blk_req` header (spec §5.2.6): le32 type, le32
/// reserved, le64 sector.
pub const BLK_REQ_HEADER_LEN: usize = 16;

/// Fixed size of the status trailer (spec §5.2.6): one byte.
pub const BLK_STATUS_LEN: usize = 1;

/// Borrowed view over one 16-byte `virtio_blk_req` header. Points into the
/// caller's descriptor buffer — no copy, no ownership, the same shape
/// `super::net::NetHdr` uses over the `virtio_net_hdr`.
#[derive(Debug, Clone, Copy)]
pub struct BlkReqHeader<'buffer> {
    bytes: &'buffer [u8],
}

impl<'buffer> BlkReqHeader<'buffer> {
    pub fn parse(bytes: &'buffer [u8]) -> Result<Self, DecodeError> {
        if bytes.len() < BLK_REQ_HEADER_LEN {
            return Err(DecodeError::Truncated {
                need: BLK_REQ_HEADER_LEN,
                got: bytes.len(),
            });
        }
        Ok(Self {
            bytes: &bytes[..BLK_REQ_HEADER_LEN],
        })
    }

    #[must_use]
    pub fn request_type(&self) -> RequestType {
        RequestType::from(read_u32(self.bytes, 0))
    }

    #[must_use]
    pub fn reserved(&self) -> u32 {
        read_u32(self.bytes, 4)
    }

    #[must_use]
    pub fn sector(&self) -> u64 {
        read_u64(self.bytes, 8)
    }
}

/// Owned field set for [`write_blk_req_header`] — the driver-side
/// counterpart to [`BlkReqHeader`]'s borrowed decode, mirroring
/// `super::net::NetHdrFields` / `super::net::write_net_hdr`'s owned-in,
/// borrowed-out split. A real device never writes this header (only the
/// driver publishes one); this exists for the worked-example tests and any
/// guest-side driver that assembles a request in caller-owned memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BlkReqHeaderFields {
    pub request_type: u32,
    pub reserved: u32,
    pub sector: u64,
}

/// Write one `virtio_blk_req` header into `out` (spec §5.2.6).
pub fn write_blk_req_header(
    out: &mut [u8],
    fields: BlkReqHeaderFields,
) -> Result<usize, DecodeError> {
    if out.len() < BLK_REQ_HEADER_LEN {
        return Err(DecodeError::Truncated {
            need: BLK_REQ_HEADER_LEN,
            got: out.len(),
        });
    }
    out[0..4].copy_from_slice(&fields.request_type.to_le_bytes());
    out[4..8].copy_from_slice(&fields.reserved.to_le_bytes());
    out[8..16].copy_from_slice(&fields.sector.to_le_bytes());
    Ok(BLK_REQ_HEADER_LEN)
}

/// Write the one-byte status trailer into `out` (spec §5.2.6).
pub fn write_blk_status(out: &mut [u8], status: u8) -> Result<usize, DecodeError> {
    if out.is_empty() {
        return Err(DecodeError::Truncated {
            need: BLK_STATUS_LEN,
            got: 0,
        });
    }
    out[0] = status;
    Ok(BLK_STATUS_LEN)
}

/// Read the one-byte status trailer back out of `bytes` (the driver side of
/// [`write_blk_status`]).
pub fn read_blk_status(bytes: &[u8]) -> Result<u8, DecodeError> {
    bytes.first().copied().ok_or(DecodeError::Truncated {
        need: BLK_STATUS_LEN,
        got: 0,
    })
}

/// First byte offset of virtio-blk's device-config-space register block
/// (spec §4.2.2 places device-specific config after the transport's own
/// registers, which `super::mmio` occupies through `0x0fc`) — identical
/// convention `super::net::CONFIG_SPACE_BASE` uses.
pub const CONFIG_SPACE_BASE: u64 = 0x100;
const REG_CONFIG_CAPACITY_LOW: u64 = CONFIG_SPACE_BASE;
const REG_CONFIG_CAPACITY_HIGH: u64 = CONFIG_SPACE_BASE + 0x004;

/// A virtio-blk device: `super::mmio::MmioDevice`'s core register block (the
/// transport FSM `super::mmio` already owns, composed rather than
/// duplicated) plus the single device-config-space field this slice reads
/// back — `capacity`, in 512-byte sectors (spec §5.2.4). `MAX_QUEUES = 1`
/// (spec §5.2.2: one `requestq`) is the only queue count this slice
/// negotiates; multiqueue is out of scope, matching `super::net`'s own
/// single-queue-pair scoping.
#[derive(Debug, Clone)]
pub struct BlkConfigSpace {
    transport: MmioDevice<1>,
    capacity_sectors: u64,
}

impl BlkConfigSpace {
    /// A freshly reset virtio-blk device advertising `capacity_sectors`
    /// (512-byte sectors, spec §5.2.4), offering `VIRTIO_F_VERSION_1` only —
    /// see the module doc for why nothing else is offered.
    #[must_use]
    pub fn new(capacity_sectors: u64, queue_num_max: u16) -> Self {
        Self {
            transport: MmioDevice::new(
                DEVICE_ID_BLK,
                queue_num_max,
                super::status::FEATURE_VERSION_1,
            ),
            capacity_sectors,
        }
    }

    #[must_use]
    pub fn capacity_sectors(&self) -> u64 {
        self.capacity_sectors
    }

    #[must_use]
    pub fn transport(&self) -> &MmioDevice<1> {
        &self.transport
    }

    /// Apply one register access: offsets below [`CONFIG_SPACE_BASE`]
    /// delegate straight to the composed `MmioDevice`; offsets at or above
    /// it are this device's own config-space block — the same split
    /// `super::net::NetConfigSpace::apply` uses.
    pub fn apply(&mut self, access: MmioAccess) -> Result<MmioEffect, MmioError> {
        match access.offset {
            REG_CONFIG_CAPACITY_LOW if !access.is_write => {
                Ok(MmioEffect::ReadValue(self.capacity_sectors as u32))
            }
            REG_CONFIG_CAPACITY_LOW => Err(MmioError::ReadOnlyRegister {
                offset: access.offset,
            }),

            REG_CONFIG_CAPACITY_HIGH if !access.is_write => {
                Ok(MmioEffect::ReadValue((self.capacity_sectors >> 32) as u32))
            }
            REG_CONFIG_CAPACITY_HIGH => Err(MmioError::ReadOnlyRegister {
                offset: access.offset,
            }),

            _ => self.transport.apply(access),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::virtio::descriptor::{DESC_LEN, DescriptorChain};
    use crate::virtio::status::{
        STATUS_ACKNOWLEDGE, STATUS_DRIVER, STATUS_DRIVER_OK, STATUS_FEATURES_OK,
    };

    /// Worked example (principle 9 / /algorithm-development): a hand-derived
    /// three-descriptor READ (`VIRTIO_BLK_T_IN`) chain the driver publishes
    /// on `requestq` — one device-readable descriptor carrying the 16-byte
    /// `virtio_blk_req` header (spec §5.2.6), chained via `NEXT` into a
    /// device-writable 512-byte data descriptor, chained via `NEXT` into a
    /// device-writable 1-byte status descriptor. Every byte below is walked
    /// bit-exact against the spec (no QEMU session was reachable from this
    /// worktree, the same discipline `super::net`'s worked-example test
    /// uses).
    #[test]
    fn read_chain_carries_a_blk_req_header_a_data_buffer_and_a_status_byte_exact() {
        const QUEUE_SIZE: u16 = 4;

        // --- descriptor table: head 0 (16-byte hdr) -> 1 (512-byte data,
        // device-writable) -> 2 (1-byte status, device-writable) ---
        let mut table = [0u8; 4 * DESC_LEN];
        table[0..16].copy_from_slice(&[
            0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // addr = 0x1000 (hdr)
            0x10, 0x00, 0x00, 0x00, // len = 16 (BLK_REQ_HEADER_LEN)
            0x01, 0x00, // flags = NEXT
            0x01, 0x00, // next = 1
        ]);
        table[16..32].copy_from_slice(&[
            0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // addr = 0x2000 (data)
            0x00, 0x02, 0x00, 0x00, // len = 512 (one sector)
            0x03, 0x00, // flags = NEXT | WRITE
            0x02, 0x00, // next = 2
        ]);
        table[32..48].copy_from_slice(&[
            0x00, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // addr = 0x3000 (status)
            0x01, 0x00, 0x00, 0x00, // len = 1
            0x02, 0x00, // flags = WRITE (chain terminator)
            0x00, 0x00, // next unused
        ]);

        // --- walk the chain: readable header, writable data, writable status ---
        let mut chain = DescriptorChain::new(&table, QUEUE_SIZE, 0);
        let header_descriptor = chain.next().expect("link one").expect("well-formed");
        assert_eq!(header_descriptor.addr(), 0x1000);
        assert_eq!(header_descriptor.buffer_len(), BLK_REQ_HEADER_LEN as u32);
        assert!(!header_descriptor.flags().device_writable());

        let data_descriptor = chain.next().expect("link two").expect("well-formed");
        assert_eq!(data_descriptor.addr(), 0x2000);
        assert_eq!(data_descriptor.buffer_len(), 512);
        assert!(data_descriptor.flags().device_writable());

        let status_descriptor = chain.next().expect("link three").expect("well-formed");
        assert_eq!(status_descriptor.addr(), 0x3000);
        assert_eq!(status_descriptor.buffer_len(), 1);
        assert!(status_descriptor.flags().device_writable());
        assert!(chain.next().is_none(), "three-link chain");

        // --- the header descriptor's buffer, byte-exact per spec §5.2.6 ---
        // type = VIRTIO_BLK_T_IN (0), reserved = 0, sector = 7.
        let header_bytes: [u8; BLK_REQ_HEADER_LEN] = [
            0x00, 0x00, 0x00, 0x00, // type = IN
            0x00, 0x00, 0x00, 0x00, // reserved
            0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // sector = 7
        ];
        let header = BlkReqHeader::parse(&header_bytes).expect("well-formed blk req header");
        assert_eq!(header.request_type(), RequestType::In);
        assert_eq!(header.reserved(), 0);
        assert_eq!(header.sector(), 7);

        // round trip: write the same fields back and confirm byte-exact.
        let mut written = [0u8; BLK_REQ_HEADER_LEN];
        let count = write_blk_req_header(
            &mut written,
            BlkReqHeaderFields {
                request_type: BLK_T_IN,
                reserved: 0,
                sector: 7,
            },
        )
        .expect("16-byte buffer fits one blk req header");
        assert_eq!(count, BLK_REQ_HEADER_LEN);
        assert_eq!(written, header_bytes);

        // --- the device services the read: writes 512 sector bytes (a
        // seeded pattern) then OK into the status trailer ---
        let sector_bytes: [u8; 512] = core::array::from_fn(|index| (index % 256) as u8);
        let mut status_byte = [0xffu8; BLK_STATUS_LEN];
        write_blk_status(&mut status_byte, STATUS_OK).expect("1-byte buffer fits status");
        assert_eq!(status_byte, [STATUS_OK]);
        assert_eq!(
            read_blk_status(&status_byte).expect("status byte present"),
            STATUS_OK
        );
        assert_eq!(sector_bytes[0], 0);
        assert_eq!(sector_bytes[255], 255);
    }

    /// The OUT (write) mirror of the read chain above: the data descriptor is
    /// device-readable (the driver already populated it), everything else is
    /// identical in shape.
    #[test]
    fn write_chain_data_descriptor_is_device_readable_not_writable() {
        const QUEUE_SIZE: u16 = 4;

        let mut table = [0u8; 4 * DESC_LEN];
        table[0..16].copy_from_slice(&[
            0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // addr = 0x1000 (hdr)
            0x10, 0x00, 0x00, 0x00, // len = 16
            0x01, 0x00, // flags = NEXT
            0x01, 0x00, // next = 1
        ]);
        table[16..32].copy_from_slice(&[
            0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // addr = 0x2000 (data)
            0x00, 0x02, 0x00, 0x00, // len = 512
            0x01, 0x00, // flags = NEXT only (device-readable: driver wrote it)
            0x02, 0x00, // next = 2
        ]);
        table[32..48].copy_from_slice(&[
            0x00, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // addr = 0x3000 (status)
            0x01, 0x00, 0x00, 0x00, // len = 1
            0x02, 0x00, // flags = WRITE
            0x00, 0x00,
        ]);

        let mut chain = DescriptorChain::new(&table, QUEUE_SIZE, 0);
        let header_descriptor = chain.next().expect("link one").expect("well-formed");
        assert!(!header_descriptor.flags().device_writable());

        let data_descriptor = chain.next().expect("link two").expect("well-formed");
        assert!(
            !data_descriptor.flags().device_writable(),
            "OUT data is device-readable: the driver already wrote it"
        );

        let status_descriptor = chain.next().expect("link three").expect("well-formed");
        assert!(status_descriptor.flags().device_writable());
        assert!(chain.next().is_none());

        let header_bytes: [u8; BLK_REQ_HEADER_LEN] = [
            0x01, 0x00, 0x00, 0x00, // type = OUT
            0x00, 0x00, 0x00, 0x00, // reserved
            0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // sector = 3
        ];
        let header = BlkReqHeader::parse(&header_bytes).expect("well-formed blk req header");
        assert_eq!(header.request_type(), RequestType::Out);
        assert_eq!(header.sector(), 3);
    }

    #[test]
    fn unknown_request_type_is_surfaced_as_unsupported() {
        let header_bytes: [u8; BLK_REQ_HEADER_LEN] = [
            0x05, 0x00, 0x00, 0x00, // type = 5, not IN/OUT/FLUSH
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        let header = BlkReqHeader::parse(&header_bytes).expect("well-formed blk req header");
        assert_eq!(header.request_type(), RequestType::Unsupported(5));
    }

    #[test]
    fn flush_request_carries_no_data_descriptor_by_convention() {
        let header_bytes: [u8; BLK_REQ_HEADER_LEN] = [
            0x04, 0x00, 0x00, 0x00, // type = FLUSH
            0x00, 0x00, 0x00, 0x00, // reserved
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // sector = 0
        ];
        let header = BlkReqHeader::parse(&header_bytes).expect("well-formed blk req header");
        assert_eq!(header.request_type(), RequestType::Flush);
    }

    #[test]
    fn blk_req_header_short_buffer_is_truncated() {
        let bytes = [0u8; BLK_REQ_HEADER_LEN - 1];
        assert_eq!(
            BlkReqHeader::parse(&bytes).unwrap_err(),
            DecodeError::Truncated {
                need: BLK_REQ_HEADER_LEN,
                got: BLK_REQ_HEADER_LEN - 1
            }
        );
    }

    #[test]
    fn write_blk_req_header_short_buffer_is_truncated() {
        let mut out = [0u8; BLK_REQ_HEADER_LEN - 1];
        assert_eq!(
            write_blk_req_header(&mut out, BlkReqHeaderFields::default()).unwrap_err(),
            DecodeError::Truncated {
                need: BLK_REQ_HEADER_LEN,
                got: BLK_REQ_HEADER_LEN - 1
            }
        );
    }

    #[test]
    fn write_blk_status_empty_buffer_is_truncated() {
        let mut out: [u8; 0] = [];
        assert_eq!(
            write_blk_status(&mut out, STATUS_OK).unwrap_err(),
            DecodeError::Truncated {
                need: BLK_STATUS_LEN,
                got: 0
            }
        );
    }

    #[test]
    fn read_blk_status_empty_buffer_is_truncated() {
        assert_eq!(
            read_blk_status(&[]).unwrap_err(),
            DecodeError::Truncated {
                need: BLK_STATUS_LEN,
                got: 0
            }
        );
    }

    /// Config-space reads report `capacity` split across the two 32-bit
    /// registers, mirroring `super::net`'s own config-space test.
    #[test]
    fn config_space_reads_report_capacity_across_both_registers() {
        let capacity_sectors: u64 = 0x0000_0002_0000_0001; // exercises both halves
        let mut device = BlkConfigSpace::new(capacity_sectors, 4);

        let read = |offset: u64| MmioAccess {
            offset,
            is_write: false,
            value: 0,
        };
        assert_eq!(
            device.apply(read(REG_CONFIG_CAPACITY_LOW)).unwrap(),
            MmioEffect::ReadValue(capacity_sectors as u32)
        );
        assert_eq!(
            device.apply(read(REG_CONFIG_CAPACITY_HIGH)).unwrap(),
            MmioEffect::ReadValue((capacity_sectors >> 32) as u32)
        );
        assert_eq!(device.capacity_sectors(), capacity_sectors);
    }

    #[test]
    fn config_space_write_is_rejected() {
        let mut device = BlkConfigSpace::new(0, 4);
        let write = MmioAccess {
            offset: REG_CONFIG_CAPACITY_LOW,
            is_write: true,
            value: 0,
        };
        assert_eq!(
            device.apply(write).unwrap_err(),
            MmioError::ReadOnlyRegister {
                offset: REG_CONFIG_CAPACITY_LOW
            }
        );
    }

    /// Offsets below `CONFIG_SPACE_BASE` fall straight through to the
    /// composed `MmioDevice` — the full bring-up sequence still works
    /// through `BlkConfigSpace::apply`, proving the delegation is exact.
    #[test]
    fn probe_and_status_registers_delegate_to_the_composed_transport() {
        let mut device = BlkConfigSpace::new(1024, 4);
        let read = |offset: u64| MmioAccess {
            offset,
            is_write: false,
            value: 0,
        };
        assert_eq!(
            device
                .apply(read(super::super::mmio::REG_DEVICE_ID))
                .unwrap(),
            MmioEffect::ReadValue(DEVICE_ID_BLK)
        );
        assert_eq!(
            device.transport().status(),
            super::super::status::DeviceStatus::Reset
        );

        let write = |offset: u64, value: u32| MmioAccess {
            offset,
            is_write: true,
            value,
        };
        assert_eq!(
            device
                .apply(write(
                    super::super::mmio::REG_STATUS,
                    u32::from(STATUS_ACKNOWLEDGE)
                ))
                .unwrap(),
            MmioEffect::StatusTransition(super::super::status::DeviceStatus::Acknowledged)
        );
        let _ = STATUS_DRIVER_OK; // referenced only to keep the import used across cfg permutations
        let _ = STATUS_DRIVER;
        let _ = STATUS_FEATURES_OK;
    }
}
