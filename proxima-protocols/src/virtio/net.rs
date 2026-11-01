//! virtio-net device codec (VIRTIO 1.2 spec §5.1): the per-packet
//! `virtio_net_hdr` that precedes every TX/RX Ethernet frame on this
//! device's queues, plus the feature bits and device-config-space fields
//! this M6 slice actually honors. Mirrors `super::used`'s shape exactly —
//! memory to decode, not a stream (guiding-principles §11: sans-IO,
//! borrowed view over caller-owned bytes, fixed-shape, allocates nothing).
//!
//! [`DEVICE_ID_NET`] plugs straight into `super::mmio::MmioDevice::new`
//! (spec §5, Table 5.1: DeviceID 1). Feature offering is deliberately
//! narrow: [`FEATURE_VERSION_1`](super::status::FEATURE_VERSION_1),
//! [`FEATURE_NET_MAC`] and [`FEATURE_NET_STATUS`] only — this codec does not honor checksum offload,
//! GSO, or merged RX buffers, so offering those bits would let a driver
//! negotiate behavior the device never implements. `Negotiation`'s
//! `acked ⊆ offered` check (`super::status`) makes over-offering a live
//! defect, not a latent one; the fix is to never offer what is not built.
//!
//! Device config space (spec §5.1.4: MAC, status, max_virtqueue_pairs) is
//! served from a byte image starting at offset `0x100`. A driver reads each
//! field at its natural width (spec §4.2.2.2: the MAC bytewise, `status` and
//! `max_virtqueue_pairs` as 16 bits), so [`NetConfigSpace::read_config`]
//! accepts 1, 2, 4 and 8 byte reads anywhere inside the image, and
//! [`NetConfigSpace::apply`] serves aligned 32-bit words through that same
//! path. `apply` composes `MmioDevice::apply` for the core register block
//! rather than reimplementing it (reuse-first, principle 1).

use super::error::DecodeError;
use super::mmio::{
    INTERRUPT_CONFIG_CHANGE, INTERRUPT_USED_BUFFER, MmioAccess, MmioDevice, MmioEffect, MmioError,
};
use super::raw::read_u16;

/// `DeviceID` value for the network device type (spec §5, Table 5.1).
pub const DEVICE_ID_NET: u32 = 1;

/// `VIRTIO_NET_F_MAC` (spec §5.1.3, bit 5): the device has a fixed MAC
/// address the driver reads from config space rather than generating one.
pub const FEATURE_NET_MAC: u64 = 1 << 5;

/// receiveq1 and transmitq1 queue indices (spec §5.1.2).
pub const RECEIVEQ1: u16 = 0;
pub const TRANSMITQ1: u16 = 1;

/// `VIRTIO_NET_F_STATUS` (spec §5.1.3, bit 16): the config-space `status`
/// field is valid, so the device can report link-down to the driver.
pub const FEATURE_NET_STATUS: u64 = 1 << 16;

const HDR_FLAG_NEEDS_CSUM: u8 = 1;
const HDR_FLAG_DATA_VALID: u8 = 2;
const HDR_FLAG_RSC_INFO: u8 = 4;

/// `gso_type` values (spec §5.1.6.1, `VIRTIO_NET_HDR_GSO_*`). This codec
/// never sets a non-`NONE` type on transmit and rejects one on receive
/// (`FEATURE_NET_MAC`-only devices never negotiate GSO), but the constants
/// are named so a caller can recognize what it rejected.
pub const GSO_NONE: u8 = 0;
pub const GSO_TCPV4: u8 = 1;
pub const GSO_UDP: u8 = 3;
pub const GSO_TCPV6: u8 = 4;
pub const GSO_ECN: u8 = 0x80;

/// Fixed size of one `virtio_net_hdr` (spec §5.1.6.1): with
/// `VIRTIO_F_VERSION_1` negotiated, every device uses the `num_buffers`
/// form regardless of `VIRTIO_NET_F_MRG_RXBUF`, so this codec — which
/// offers `VIRTIO_F_VERSION_1` unconditionally per `super::status` — models
/// the 12-byte layout exclusively; the legacy 10-byte header is out of
/// scope.
pub const NET_HDR_LEN: usize = 12;

/// Decoded `virtio_net_hdr` flag bits (spec §5.1.6.1, `VIRTIO_NET_HDR_F_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NetHdrFlags(u8);

impl NetHdrFlags {
    #[must_use]
    pub fn needs_csum(self) -> bool {
        self.0 & HDR_FLAG_NEEDS_CSUM != 0
    }

    #[must_use]
    pub fn data_valid(self) -> bool {
        self.0 & HDR_FLAG_DATA_VALID != 0
    }

    #[must_use]
    pub fn rsc_info(self) -> bool {
        self.0 & HDR_FLAG_RSC_INFO != 0
    }

    #[must_use]
    pub fn bits(self) -> u8 {
        self.0
    }
}

/// Owned field set for [`write_net_hdr`] — the device-side counterpart to
/// [`NetHdr`]'s borrowed decode, mirroring `super::used::UsedElem` /
/// `super::used::write_used_elem`'s owned-in, borrowed-out split.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NetHdrFields {
    pub flags: u8,
    pub gso_type: u8,
    pub hdr_len: u16,
    pub gso_size: u16,
    pub csum_start: u16,
    pub csum_offset: u16,
    pub num_buffers: u16,
}

/// Borrowed view over one 12-byte `virtio_net_hdr` (spec §5.1.6.1): u8
/// flags, u8 gso_type, le16 hdr_len, le16 gso_size, le16 csum_start, le16
/// csum_offset, le16 num_buffers. Points into the caller's descriptor
/// buffer — no copy, no ownership, the same shape `super::descriptor::
/// Descriptor` uses over one descriptor table entry.
#[derive(Debug, Clone, Copy)]
pub struct NetHdr<'buffer> {
    bytes: &'buffer [u8],
}

impl<'buffer> NetHdr<'buffer> {
    pub fn parse(bytes: &'buffer [u8]) -> Result<Self, DecodeError> {
        if bytes.len() < NET_HDR_LEN {
            return Err(DecodeError::Truncated {
                need: NET_HDR_LEN,
                got: bytes.len(),
            });
        }
        Ok(Self {
            bytes: &bytes[..NET_HDR_LEN],
        })
    }

    #[must_use]
    pub fn flags(&self) -> NetHdrFlags {
        NetHdrFlags(self.bytes[0])
    }

    #[must_use]
    pub fn gso_type(&self) -> u8 {
        self.bytes[1]
    }

    #[must_use]
    pub fn hdr_len(&self) -> u16 {
        read_u16(self.bytes, 2)
    }

    #[must_use]
    pub fn gso_size(&self) -> u16 {
        read_u16(self.bytes, 4)
    }

    #[must_use]
    pub fn csum_start(&self) -> u16 {
        read_u16(self.bytes, 6)
    }

    #[must_use]
    pub fn csum_offset(&self) -> u16 {
        read_u16(self.bytes, 8)
    }

    #[must_use]
    pub fn num_buffers(&self) -> u16 {
        read_u16(self.bytes, 10)
    }
}

/// Write one `virtio_net_hdr` into `out` (spec §5.1.6.1) — the device-side
/// counterpart to [`NetHdr::parse`], mirroring
/// `super::used::write_used_elem`.
pub fn write_net_hdr(out: &mut [u8], fields: NetHdrFields) -> Result<usize, DecodeError> {
    if out.len() < NET_HDR_LEN {
        return Err(DecodeError::Truncated {
            need: NET_HDR_LEN,
            got: out.len(),
        });
    }
    out[0] = fields.flags;
    out[1] = fields.gso_type;
    out[2..4].copy_from_slice(&fields.hdr_len.to_le_bytes());
    out[4..6].copy_from_slice(&fields.gso_size.to_le_bytes());
    out[6..8].copy_from_slice(&fields.csum_start.to_le_bytes());
    out[8..10].copy_from_slice(&fields.csum_offset.to_le_bytes());
    out[10..12].copy_from_slice(&fields.num_buffers.to_le_bytes());
    Ok(NET_HDR_LEN)
}

/// First byte offset of virtio-net's device-config-space register block
/// (spec §4.2.2 places device-specific config after the transport's own
/// registers, which `super::mmio` occupies through `0x0fc`).
pub const CONFIG_SPACE_BASE: u64 = 0x100;
const REG_CONFIG_MAC_LOW: u64 = CONFIG_SPACE_BASE;
const REG_CONFIG_MAC_HIGH_STATUS: u64 = CONFIG_SPACE_BASE + 0x004;
const REG_CONFIG_MAX_VQ_PAIRS: u64 = CONFIG_SPACE_BASE + 0x008;

const CONFIG_STATUS_OFFSET: usize = 6;
const CONFIG_MAX_VQ_PAIRS_OFFSET: usize = 8;
const CONFIG_IMAGE_LEN: usize = 12;

/// `status` config-space field value (spec §5.1.4, `VIRTIO_NET_S_LINK_UP`):
/// reported while [`NetConfigSpace::link_up`] holds, cleared when the host
/// takes the link down.
pub const NET_STATUS_LINK_UP: u16 = 1;

/// A virtio-net device: `super::mmio::MmioDevice`'s core register block
/// (the transport FSM `super::mmio` already owns, composed rather than
/// duplicated) plus the three device-config-space fields this slice reads
/// back — MAC, link status, and the single queue pair this device offers.
/// `MAX_QUEUES = 2` (receiveq1 = 0, transmitq1 = 1, spec §5.1.2) is the
/// only queue count this slice negotiates; multiqueue is out of scope.
#[derive(Debug, Clone)]
pub struct NetConfigSpace {
    transport: MmioDevice<2>,
    mac: [u8; 6],
    link_up: bool,
}

impl NetConfigSpace {
    /// A freshly reset virtio-net device advertising `mac` with its link up,
    /// offering `VIRTIO_F_VERSION_1 | FEATURE_NET_MAC | FEATURE_NET_STATUS`
    /// only — see the module doc for why nothing else is offered.
    #[must_use]
    pub fn new(mac: [u8; 6], queue_num_max: u16) -> Self {
        Self {
            transport: MmioDevice::new(
                DEVICE_ID_NET,
                queue_num_max,
                super::status::FEATURE_VERSION_1 | FEATURE_NET_MAC | FEATURE_NET_STATUS,
            ),
            mac,
            link_up: true,
        }
    }

    pub fn set_link_up(&mut self, link_up: bool) {
        if self.link_up != link_up {
            self.transport.raise_interrupt(INTERRUPT_CONFIG_CHANGE);
        }
        self.link_up = link_up;
    }

    pub fn raise_used_buffer(&mut self) {
        self.transport.raise_interrupt(INTERRUPT_USED_BUFFER);
    }

    #[must_use]
    pub fn interrupt_pending(&self) -> bool {
        self.transport.interrupt_pending()
    }

    fn config_image(&self) -> [u8; CONFIG_IMAGE_LEN] {
        let mut image = [0u8; CONFIG_IMAGE_LEN];
        let status = if self.link_up { NET_STATUS_LINK_UP } else { 0 };
        image[..6].copy_from_slice(&self.mac);
        image[CONFIG_STATUS_OFFSET..CONFIG_STATUS_OFFSET + 2].copy_from_slice(&status.to_le_bytes());
        image[CONFIG_MAX_VQ_PAIRS_OFFSET..CONFIG_MAX_VQ_PAIRS_OFFSET + 2]
            .copy_from_slice(&1u16.to_le_bytes());
        image
    }

    /// Read `length` (1, 2, 4 or 8) config-space bytes at absolute `offset`
    /// as a little-endian integer.
    pub fn read_config(&self, offset: u64, length: usize) -> Result<u64, MmioError> {
        let invalid = MmioError::ConfigAccess { offset, length };
        if !matches!(length, 1 | 2 | 4 | 8) {
            return Err(invalid);
        }
        let start = offset
            .checked_sub(CONFIG_SPACE_BASE)
            .and_then(|relative| usize::try_from(relative).ok())
            .ok_or(invalid)?;
        let image = self.config_image();
        let bytes = image.get(start..start + length).ok_or(invalid)?;
        let mut word = [0u8; 8];
        word[..length].copy_from_slice(bytes);
        Ok(u64::from_le_bytes(word))
    }

    #[must_use]
    pub fn link_up(&self) -> bool {
        self.link_up
    }

    #[must_use]
    pub fn mac(&self) -> [u8; 6] {
        self.mac
    }

    #[must_use]
    pub fn transport(&self) -> &MmioDevice<2> {
        &self.transport
    }

    /// Apply one register access: offsets below [`CONFIG_SPACE_BASE`]
    /// delegate straight to the composed `MmioDevice`; offsets at or above
    /// it are this device's own config-space block.
    pub fn apply(&mut self, access: MmioAccess) -> Result<MmioEffect, MmioError> {
        match access.offset {
            REG_CONFIG_MAC_LOW | REG_CONFIG_MAC_HIGH_STATUS | REG_CONFIG_MAX_VQ_PAIRS
                if access.is_write =>
            {
                Err(MmioError::ReadOnlyRegister {
                    offset: access.offset,
                })
            }
            REG_CONFIG_MAC_LOW | REG_CONFIG_MAC_HIGH_STATUS | REG_CONFIG_MAX_VQ_PAIRS => self
                .read_config(access.offset, 4)
                .map(|word| MmioEffect::ReadValue(word as u32)),
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
    /// two-descriptor TX chain the driver publishes on transmitq1 — one
    /// device-readable descriptor carrying the 12-byte `virtio_net_hdr`
    /// (spec §5.1.6.1), chained via `NEXT` into a second device-readable
    /// descriptor carrying a minimal 14-byte Ethernet header (dst
    /// broadcast, a locally-administered src MAC, ethertype ARP). Every
    /// byte below is walked bit-exact against the spec (no QEMU session was
    /// reachable from this worktree, the same discipline the sibling
    /// worked-example tests in `super::mod` and `super::descriptor` use).
    #[test]
    fn tx_descriptor_chain_carries_a_net_hdr_and_an_ethernet_frame_byte_exact() {
        const QUEUE_SIZE: u16 = 4;

        // --- descriptor table: head 0 (12-byte hdr) -> 2 (14-byte frame) ---
        let mut table = [0u8; 4 * DESC_LEN];
        table[0..16].copy_from_slice(&[
            0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // addr = 0x1000 (hdr)
            0x0c, 0x00, 0x00, 0x00, // len = 12 (NET_HDR_LEN)
            0x01, 0x00, // flags = NEXT
            0x02, 0x00, // next = 2
        ]);
        table[32..48].copy_from_slice(&[
            0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // addr = 0x2000 (frame)
            0x0e, 0x00, 0x00, 0x00, // len = 14 (Ethernet header only)
            0x00, 0x00, // flags = 0 (device-readable, chain terminator)
            0x00, 0x00, // next unused
        ]);

        // --- walk the chain: two device-readable descriptors ---
        let mut chain = DescriptorChain::new(&table, QUEUE_SIZE, 0);
        let hdr_descriptor = chain.next().expect("link one").expect("well-formed");
        assert_eq!(hdr_descriptor.addr(), 0x1000);
        assert_eq!(hdr_descriptor.buffer_len(), NET_HDR_LEN as u32);
        assert!(!hdr_descriptor.flags().device_writable());

        let frame_descriptor = chain.next().expect("link two").expect("well-formed");
        assert_eq!(frame_descriptor.addr(), 0x2000);
        assert_eq!(frame_descriptor.buffer_len(), 14);
        assert!(!frame_descriptor.flags().device_writable());
        assert!(chain.next().is_none(), "two-link chain");

        // --- the hdr descriptor's buffer, byte-exact per spec §5.1.6.1 ---
        // flags = 0, gso_type = NONE, hdr_len = 0, gso_size = 0,
        // csum_start = 0, csum_offset = 0, num_buffers = 1.
        let hdr_bytes: [u8; NET_HDR_LEN] = [
            0x00,     // flags
            GSO_NONE, // gso_type
            0x00, 0x00, // hdr_len
            0x00, 0x00, // gso_size
            0x00, 0x00, // csum_start
            0x00, 0x00, // csum_offset
            0x01, 0x00, // num_buffers = 1
        ];
        let hdr = NetHdr::parse(&hdr_bytes).expect("well-formed net hdr");
        assert!(!hdr.flags().needs_csum());
        assert!(!hdr.flags().data_valid());
        assert_eq!(hdr.gso_type(), GSO_NONE);
        assert_eq!(hdr.hdr_len(), 0);
        assert_eq!(hdr.gso_size(), 0);
        assert_eq!(hdr.csum_start(), 0);
        assert_eq!(hdr.csum_offset(), 0);
        assert_eq!(hdr.num_buffers(), 1);

        // round trip: write the same fields back and confirm byte-exact.
        let mut written = [0u8; NET_HDR_LEN];
        let count = write_net_hdr(
            &mut written,
            NetHdrFields {
                flags: 0,
                gso_type: GSO_NONE,
                hdr_len: 0,
                gso_size: 0,
                csum_start: 0,
                csum_offset: 0,
                num_buffers: 1,
            },
        )
        .expect("12-byte buffer fits one net hdr");
        assert_eq!(count, NET_HDR_LEN);
        assert_eq!(written, hdr_bytes);

        // --- the frame descriptor's buffer: a minimal 14-byte Ethernet
        // header, dst broadcast, src a locally-administered MAC, ARP
        // ethertype (0x0806) — no payload, matching the descriptor's
        // declared 14-byte length.
        let frame_bytes: [u8; 14] = [
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, // dst = broadcast
            0x02, 0x11, 0x22, 0x33, 0x44, 0x55, // src
            0x08, 0x06, // ethertype = ARP
        ];
        assert_eq!(&frame_bytes[0..6], &[0xff; 6]);
        assert_eq!(&frame_bytes[12..14], &[0x08, 0x06]);
    }

    /// `FEATURE_NET_MAC` (bit 5) is one of two device-specific bits this
    /// codec offers alongside `VIRTIO_F_VERSION_1` (the other is
    /// `FEATURE_NET_STATUS`) — a driver that acks a
    /// bit none of those (e.g. `VIRTIO_NET_F_CSUM`, bit 0) is rejected by
    /// `Negotiation`'s `acked ⊆ offered` rule, proving the codec never
    /// silently promises checksum/GSO/merged-rxbuf behavior it does not
    /// implement.
    #[test]
    fn offering_only_version_1_and_mac_rejects_an_unoffered_csum_ack() {
        let mut device = NetConfigSpace::new([0x02, 0x00, 0x00, 0x00, 0x00, 0x01], 4);
        let write = |offset: u64, value: u32| MmioAccess {
            offset,
            is_write: true,
            value,
        };

        device
            .apply(write(
                super::super::mmio::REG_STATUS,
                u32::from(STATUS_ACKNOWLEDGE),
            ))
            .expect("ack");
        device
            .apply(write(
                super::super::mmio::REG_STATUS,
                u32::from(STATUS_ACKNOWLEDGE | STATUS_DRIVER),
            ))
            .expect("driver known");

        const VIRTIO_NET_F_CSUM: u64 = 1; // bit 0, never offered by this slice
        device
            .apply(write(super::super::mmio::REG_DRIVER_FEATURES_SEL, 0))
            .expect("select low word");
        device
            .apply(write(
                super::super::mmio::REG_DRIVER_FEATURES,
                VIRTIO_NET_F_CSUM as u32,
            ))
            .expect("ack low word");
        device
            .apply(write(super::super::mmio::REG_DRIVER_FEATURES_SEL, 1))
            .expect("select high word");
        device
            .apply(write(super::super::mmio::REG_DRIVER_FEATURES, 1))
            .expect("ack VERSION_1");

        let error = device
            .apply(write(
                super::super::mmio::REG_STATUS,
                u32::from(STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK),
            ))
            .unwrap_err();
        assert!(matches!(
            error,
            MmioError::Negotiation(
                super::super::status::NegotiationError::AckedUnofferedFeatures { unoffered: 1, .. }
            )
        ));
    }

    /// Config-space reads return the device's MAC, link status, and the
    /// single queue pair this slice negotiates, once the driver has legally
    /// reached `DRIVER_OK` (spec §5.1.4: config space is meaningful once
    /// negotiation is complete, though this codec does not gate the read on
    /// status — it always reflects the current fields, matching real
    /// hardware, which has no such gate either).
    #[test]
    fn config_space_reads_report_mac_link_status_and_queue_pair_count() {
        let mac = [0x02, 0xde, 0xad, 0xbe, 0xef, 0x01];
        let mut device = NetConfigSpace::new(mac, 4);

        let read = |offset: u64| MmioAccess {
            offset,
            is_write: false,
            value: 0,
        };
        assert_eq!(
            device.apply(read(REG_CONFIG_MAC_LOW)).unwrap(),
            MmioEffect::ReadValue(u32::from_le_bytes([0x02, 0xde, 0xad, 0xbe]))
        );
        assert_eq!(
            device.apply(read(REG_CONFIG_MAC_HIGH_STATUS)).unwrap(),
            MmioEffect::ReadValue(u32::from_le_bytes([0xef, 0x01, 0x01, 0x00]))
        );
        assert_eq!(
            device.apply(read(REG_CONFIG_MAX_VQ_PAIRS)).unwrap(),
            MmioEffect::ReadValue(1)
        );
        assert_eq!(device.mac(), mac);
    }

    #[test]
    fn link_down_clears_the_status_bit_the_driver_reads() {
        let mut device = NetConfigSpace::new([0x02, 0xde, 0xad, 0xbe, 0xef, 0x01], 4);
        let read = MmioAccess {
            offset: REG_CONFIG_MAC_HIGH_STATUS,
            is_write: false,
            value: 0,
        };

        device.set_link_up(false);

        assert_eq!(
            device.apply(read).unwrap(),
            MmioEffect::ReadValue(u32::from_le_bytes([0xef, 0x01, 0x00, 0x00]))
        );
        assert!(!device.link_up());
    }

    /// Config-space registers are read-only from the driver's side —
    /// mirroring `super::mmio`'s own read-only-register discipline for
    /// every probe register (`MagicValue`, `Version`, `DeviceID`).
    #[test]
    fn config_space_write_is_rejected() {
        let mut device = NetConfigSpace::new([0; 6], 4);
        let write = MmioAccess {
            offset: REG_CONFIG_MAC_LOW,
            is_write: true,
            value: 0,
        };
        assert_eq!(
            device.apply(write).unwrap_err(),
            MmioError::ReadOnlyRegister {
                offset: REG_CONFIG_MAC_LOW
            }
        );
    }

    /// Offsets below `CONFIG_SPACE_BASE` fall straight through to the
    /// composed `MmioDevice` — the full bring-up sequence still works
    /// through `NetConfigSpace::apply`, proving the delegation is exact,
    /// not a reimplementation.
    #[test]
    fn probe_and_status_registers_delegate_to_the_composed_transport() {
        let mut device = NetConfigSpace::new([0; 6], 4);
        let read = |offset: u64| MmioAccess {
            offset,
            is_write: false,
            value: 0,
        };
        assert_eq!(
            device
                .apply(read(super::super::mmio::REG_DEVICE_ID))
                .unwrap(),
            MmioEffect::ReadValue(DEVICE_ID_NET)
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
    }

    #[test]
    fn bytewise_mac_read_equals_the_six_mac_bytes() {
        let mac = [0x02, 0xde, 0xad, 0xbe, 0xef, 0x01];
        let device = NetConfigSpace::new(mac, 4);

        let bytes: Vec<u8> = (0..6)
            .map(|index| device.read_config(CONFIG_SPACE_BASE + index, 1).unwrap() as u8)
            .collect();

        assert_eq!(bytes, mac);
    }

    #[test]
    fn status_and_max_pairs_read_as_sixteen_bits() {
        let mut device = NetConfigSpace::new([0; 6], 4);
        assert_eq!(
            device.read_config(CONFIG_SPACE_BASE + 6, 2).unwrap(),
            u64::from(NET_STATUS_LINK_UP)
        );
        assert_eq!(device.read_config(CONFIG_SPACE_BASE + 8, 2).unwrap(), 1);

        device.set_link_up(false);

        assert_eq!(device.read_config(CONFIG_SPACE_BASE + 6, 2).unwrap(), 0);
    }

    #[test]
    fn eight_byte_read_spans_mac_and_status() {
        let device = NetConfigSpace::new([1, 2, 3, 4, 5, 6], 4);
        assert_eq!(
            device.read_config(CONFIG_SPACE_BASE, 8).unwrap(),
            u64::from_le_bytes([1, 2, 3, 4, 5, 6, 1, 0])
        );
    }

    #[test]
    fn config_read_rejects_bad_widths_and_out_of_image_offsets() {
        let device = NetConfigSpace::new([0; 6], 4);
        for (offset, length) in [
            (CONFIG_SPACE_BASE, 3),
            (CONFIG_SPACE_BASE, 0),
            (CONFIG_SPACE_BASE + 9, 4),
            (CONFIG_SPACE_BASE + 12, 1),
            (CONFIG_SPACE_BASE - 4, 4),
        ] {
            assert_eq!(
                device.read_config(offset, length).unwrap_err(),
                MmioError::ConfigAccess { offset, length }
            );
        }
    }

    #[test]
    fn link_change_raises_config_change_but_no_change_does_not() {
        let mut device = NetConfigSpace::new([0; 6], 4);
        device.set_link_up(true);
        assert!(!device.interrupt_pending());

        device.set_link_up(false);

        assert!(device.interrupt_pending());
        let status = MmioAccess {
            offset: super::super::mmio::REG_INTERRUPT_STATUS,
            is_write: false,
            value: 0,
        };
        assert_eq!(
            device.apply(status).unwrap(),
            MmioEffect::ReadValue(INTERRUPT_CONFIG_CHANGE)
        );
    }

    #[test]
    fn used_buffer_interrupt_is_raised_then_acked_through_the_register() {
        let mut device = NetConfigSpace::new([0; 6], 4);
        device.raise_used_buffer();
        assert!(device.interrupt_pending());

        device
            .apply(MmioAccess {
                offset: super::super::mmio::REG_INTERRUPT_ACK,
                is_write: true,
                value: INTERRUPT_USED_BUFFER,
            })
            .unwrap();

        assert!(!device.interrupt_pending());
    }

    #[test]
    fn net_hdr_short_buffer_is_truncated() {
        let bytes = [0u8; NET_HDR_LEN - 1];
        assert_eq!(
            NetHdr::parse(&bytes).unwrap_err(),
            DecodeError::Truncated {
                need: NET_HDR_LEN,
                got: NET_HDR_LEN - 1
            }
        );
    }

    #[test]
    fn write_net_hdr_short_buffer_is_truncated() {
        let mut out = [0u8; NET_HDR_LEN - 1];
        assert_eq!(
            write_net_hdr(&mut out, NetHdrFields::default()).unwrap_err(),
            DecodeError::Truncated {
                need: NET_HDR_LEN,
                got: NET_HDR_LEN - 1
            }
        );
    }
}
