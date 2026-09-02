//! Sans-IO split-virtqueue wire codec (device side) for a userspace VMM.
//!
//! Tier-3: compiles under `#![no_std]` with no allocator. Decode borrows
//! views over caller-owned queue memory (descriptor table, avail ring, used
//! ring); the ring cursors are pure index arithmetic. Mirrors the shape of
//! the NVMe codec next door (`super::nvme`): avail is the driver's producer
//! ring and the device only reads it (like NVMe's submission ring); used is
//! the device's producer ring and the driver only reads it (like NVMe's
//! completion ring). The one shape NVMe does not have: a descriptor chain is
//! an indirection layer between a ring entry and the actual data buffer, so
//! [`descriptor::DescriptorChain`] composes with the ring cursors rather than
//! replacing them. Guest-physical-to-host-pointer translation and the
//! per-VM-exit register decode itself (`(offset, is_write, value)` recovered
//! from a trapped guest load/store) live one layer up in the VMM, mirroring
//! how `nvme`'s `QueueBackend` seam sits in `proxima-storage`.
//!
//! [`status::Negotiation`] is the device-status-register FSM (VIRTIO 1.2
//! spec §2.1): the enum-shaped handshake a driver drives before any ring
//! traffic is legal at all, so it composes with the ring codecs above rather
//! than sitting inside them. [`mmio::MmioDevice`] is the mmio transport's own
//! register-block FSM (VIRTIO 1.2 spec §4.2.2) — it composes
//! [`status::Negotiation`] for the `Status` register rather than
//! reimplementing the handshake, and turns every other register access into
//! a typed [`mmio::MmioEffect`] the VMM applies against real guest memory
//! (queue setup, `QueueNotify` → walk this module's own ring codecs).
//!
//! Virtqueue memory is guest DRAM, so every multi-byte field is
//! little-endian (VIRTIO 1.2 spec, split virtqueue layout, §2.7).

pub mod avail;
pub mod blk;
pub mod descriptor;
pub mod error;
pub mod mmio;
pub mod net;
pub mod raw;
pub mod status;
pub mod used;

pub use avail::{AvailRing, RingCursor};
pub use blk::{
    BLK_REQ_HEADER_LEN, BLK_STATUS_LEN, BlkConfigSpace, BlkReqHeader, BlkReqHeaderFields,
    DEVICE_ID_BLK, RequestType, STATUS_IOERR, STATUS_OK, STATUS_UNSUPP, read_blk_status,
    write_blk_req_header, write_blk_status,
};
pub use descriptor::{DESC_LEN, Descriptor, DescriptorChain, DescriptorFlags};
pub use error::DecodeError;
pub use mmio::{DEVICE_ID_CONSOLE, MAGIC_VALUE, MmioAccess, MmioDevice, MmioEffect, MmioError};
pub use net::{
    CONFIG_SPACE_BASE, DEVICE_ID_NET, FEATURE_NET_MAC, NET_HDR_LEN, NET_STATUS_LINK_UP,
    NetConfigSpace, NetHdr, NetHdrFields, NetHdrFlags, write_net_hdr,
};
pub use status::{DeviceStatus, FEATURE_VERSION_1, Negotiation, NegotiationError};
pub use used::{USED_ELEM_LEN, UsedElem, UsedRing, write_used_elem};

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    /// End-to-end worked example (principle 9 / /algorithm-development):
    /// descriptor table + avail ring + used ring round trip for one
    /// virtio-console write, hand-derived per VIRTIO 1.2 spec §2.7 (no QEMU
    /// session was reachable from this worktree; every byte offset below is
    /// walked bit-exact against the spec, the same discipline the nvme codec
    /// uses for its own worked examples).
    ///
    /// Scenario: queue_size = 4. The driver publishes a 2-descriptor chain
    /// rooted at head index 0 — descriptor 0 is a 5-byte device-readable
    /// buffer ("hello"), descriptor 2 is an 8-byte device-writable reply
    /// buffer. The device walks the chain, and (in place of a real MMIO
    /// buffer copy, which lives in the transport layer above this codec)
    /// reports 8 bytes written by publishing a used-ring element.
    #[test]
    fn descriptor_table_and_avail_used_rings_round_trip_one_chain() {
        const QUEUE_SIZE: u16 = 4;

        // --- descriptor table: 4 * 16-byte entries, head chain at 0 -> 2 ---
        let mut table = [0u8; 4 * DESC_LEN];
        table[0..16].copy_from_slice(&[
            0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // addr = 0x1000
            0x05, 0x00, 0x00, 0x00, // len = 5 ("hello")
            0x01, 0x00, // flags = NEXT
            0x02, 0x00, // next = 2
        ]);
        table[32..48].copy_from_slice(&[
            0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // addr = 0x2000
            0x08, 0x00, 0x00, 0x00, // len = 8 (reply buffer capacity)
            0x02, 0x00, // flags = WRITE
            0x00, 0x00, // next = 0 (ignored, NEXT unset)
        ]);

        // --- avail ring: driver publishes one entry, head = 0 ---
        let avail_bytes: [u8; 12] = [
            0x00, 0x00, // flags
            0x01, 0x00, // idx = 1 (one entry published)
            0x00, 0x00, // ring[0] = head 0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // ring[1..4] unpublished
        ];
        let avail = AvailRing::parse(&avail_bytes, QUEUE_SIZE).expect("well-formed avail ring");

        // device-side cursor starts at 0; avail.idx() == 1 means one entry
        // is pending, at free-running position 0.
        let mut avail_cursor = RingCursor::new(QUEUE_SIZE).expect("power of two");
        assert_eq!(avail_cursor.pending(avail.idx()), 1);
        let head = avail.ring_entry(avail_cursor.position());
        assert_eq!(head, 0, "the one published chain is rooted at descriptor 0");
        avail_cursor.advance();
        assert_eq!(avail_cursor.pending(avail.idx()), 0, "fully drained");

        // --- walk the chain the avail entry named ---
        let mut chain = DescriptorChain::new(&table, QUEUE_SIZE, head);
        let readable = chain.next().expect("link one").expect("well-formed");
        assert_eq!(readable.addr(), 0x1000);
        assert_eq!(readable.buffer_len(), 5);
        assert!(!readable.flags().device_writable());

        let writable = chain.next().expect("link two").expect("well-formed");
        assert_eq!(writable.addr(), 0x2000);
        assert_eq!(writable.buffer_len(), 8);
        assert!(writable.flags().device_writable());
        assert!(chain.next().is_none(), "two-link chain");

        // --- device reports completion: 8 bytes written, chain head 0 ---
        let mut used_bytes = [0u8; 4 + USED_ELEM_LEN * (QUEUE_SIZE as usize)];
        let mut used_cursor = RingCursor::new(QUEUE_SIZE).expect("power of two");
        let slot_offset = 4 + usize::from(used_cursor.position() % QUEUE_SIZE) * USED_ELEM_LEN;
        write_used_elem(
            &mut used_bytes[slot_offset..],
            UsedElem {
                id: u32::from(head),
                len: writable.buffer_len(),
            },
        )
        .expect("used-ring slot fits one element");
        let new_idx = used_cursor.advance();
        used_bytes[2..4].copy_from_slice(&new_idx.to_le_bytes());

        // --- driver side reads the used ring back ---
        let used = UsedRing::parse(&used_bytes, QUEUE_SIZE).expect("well-formed used ring");
        assert_eq!(used.idx(), 1);
        assert_eq!(used.ring_entry(0), UsedElem { id: 0, len: 8 });
    }
}
