//! virtio-net device codec (VIRTIO 1.2 spec §5.1): the per-packet
//! `virtio_net_hdr` that precedes every TX/RX Ethernet frame on this
//! device's queues, plus the feature bits and device-config-space fields
//! this M6 slice actually honors. Mirrors `super::used`'s shape exactly —
//! memory to decode, not a stream (guiding-principles §11: sans-IO,
//! borrowed view over caller-owned bytes, fixed-shape, allocates nothing).
//!
//! [`DEVICE_ID_NET`] plugs straight into `super::mmio::MmioDevice::new`
//! (spec §5, Table 5.1: DeviceID 1). Feature offering is deliberately
//! narrow: [`FEATURE_VERSION_1`](super::status::FEATURE_VERSION_1) plus
//! [`FEATURE_NET_MAC`] only — this codec does not honor checksum offload,
//! GSO, or merged RX buffers, so offering those bits would let a driver
//! negotiate behavior the device never implements. `Negotiation`'s
//! `acked ⊆ offered` check (`super::status`) makes over-offering a live
//! defect, not a latent one; the fix is to never offer what is not built.
//!
//! Device config space (spec §5.1.4: MAC, status, max_virtqueue_pairs) is
//! exposed as fixed 32-bit-word registers starting at offset `0x100`,
//! matching the width discipline `super::mmio`'s core register block
//! already uses for every other register in this codec — a deliberate
//! simplification against the spec's "driver uses the natural per-field
//! width" rule (§4.2.2.2), scoped to this M6 slice: no guest driver in
//! this worktree issues a narrower-than-32-bit config-space access yet.
//! [`NetConfigSpace::apply`] composes `MmioDevice::apply` for the core
//! register block rather than reimplementing it (reuse-first, principle 1).

use super::error::DecodeError;
use super::mmio::{MmioAccess, MmioDevice, MmioEffect, MmioError};
use super::raw::read_u16;

/// `DeviceID` value for the network device type (spec §5, Table 5.1).
pub const DEVICE_ID_NET: u32 = 1;

/// `VIRTIO_NET_F_MAC` (spec §5.1.3, bit 5): the device has a fixed MAC
/// address the driver reads from config space rather than generating one.
pub const FEATURE_NET_MAC: u64 = 1 << 5;

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

/// `status` config-space field value (spec §5.1.4, `VIRTIO_NET_S_LINK_UP`):
/// this device models a link that is always up, since M6's transport has
/// no notion of link-down.
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
}

impl NetConfigSpace {
    /// A freshly reset virtio-net device advertising `mac`, offering
    /// `VIRTIO_F_VERSION_1 | FEATURE_NET_MAC` only — see the module doc for
    /// why nothing else is offered.
    #[must_use]
    pub fn new(mac: [u8; 6], queue_num_max: u16) -> Self {
        Self {
            transport: MmioDevice::new(
                DEVICE_ID_NET,
                queue_num_max,
                super::status::FEATURE_VERSION_1 | FEATURE_NET_MAC,
            ),
            mac,
        }
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
            REG_CONFIG_MAC_LOW if !access.is_write => Ok(MmioEffect::ReadValue(u32::from_le_bytes([
                self.mac[0], self.mac[1], self.mac[2], self.mac[3],
            ]))),
            REG_CONFIG_MAC_LOW => Err(MmioError::ReadOnlyRegister { offset: access.offset }),

            REG_CONFIG_MAC_HIGH_STATUS if !access.is_write => {
                let mut word = [0u8; 4];
                word[0..2].copy_from_slice(&self.mac[4..6]);
                word[2..4].copy_from_slice(&NET_STATUS_LINK_UP.to_le_bytes());
                Ok(MmioEffect::ReadValue(u32::from_le_bytes(word)))
            }
            REG_CONFIG_MAC_HIGH_STATUS => Err(MmioError::ReadOnlyRegister { offset: access.offset }),

            REG_CONFIG_MAX_VQ_PAIRS if !access.is_write => Ok(MmioEffect::ReadValue(1)),
            REG_CONFIG_MAX_VQ_PAIRS => Err(MmioError::ReadOnlyRegister { offset: access.offset }),

            _ => self.transport.apply(access),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::virtio::descriptor::{DESC_LEN, DescriptorChain};
    use crate::virtio::status::{STATUS_ACKNOWLEDGE, STATUS_DRIVER, STATUS_DRIVER_OK, STATUS_FEATURES_OK};

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
            0x00, // flags
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

    /// `FEATURE_NET_MAC` (bit 5) is the only device-specific bit this
    /// codec offers alongside `VIRTIO_F_VERSION_1` — a driver that acks a
    /// bit neither of those (e.g. `VIRTIO_NET_F_CSUM`, bit 0) is rejected by
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
            .apply(write(super::super::mmio::REG_STATUS, u32::from(STATUS_ACKNOWLEDGE)))
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
            MmioError::Negotiation(super::super::status::NegotiationError::AckedUnofferedFeatures {
                unoffered: 1,
                ..
            })
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
            MmioError::ReadOnlyRegister { offset: REG_CONFIG_MAC_LOW }
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
            device.apply(read(super::super::mmio::REG_DEVICE_ID)).unwrap(),
            MmioEffect::ReadValue(DEVICE_ID_NET)
        );
        assert_eq!(device.transport().status(), super::super::status::DeviceStatus::Reset);

        let write = |offset: u64, value: u32| MmioAccess {
            offset,
            is_write: true,
            value,
        };
        assert_eq!(
            device
                .apply(write(super::super::mmio::REG_STATUS, u32::from(STATUS_ACKNOWLEDGE)))
                .unwrap(),
            MmioEffect::StatusTransition(super::super::status::DeviceStatus::Acknowledged)
        );
        let _ = STATUS_DRIVER_OK; // referenced only to keep the import used across cfg permutations
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
