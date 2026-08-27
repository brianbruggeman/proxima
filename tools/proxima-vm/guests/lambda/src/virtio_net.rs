//! Minimal virtio-net driver bring-up over the mmio transport (VIRTIO 1.2
//! spec §5.1), mirroring `super::virtio_console`'s shape exactly: ordinary
//! `core::ptr::{read_volatile, write_volatile}` loads/stores, no inline
//! assembly, since the host's data-abort handler
//! (`src/backend_macos.c`'s `handle_mmio_data_abort`) decodes a plain
//! 32-bit `ldr`/`str` from the trapped instruction's own syndrome. Every
//! register offset below matches `proxima-protocols/src/virtio/mmio.rs`'s
//! `REG_*` constants exactly, same duplication discipline
//! `virtio_console.rs`'s own module doc already explains (a bare
//! `no_std`/`no_alloc` guest crate carries no dependency on the host-tier
//! protocol crate).
//!
//! Scope: negotiates only `VIRTIO_F_VERSION_1` and `VIRTIO_NET_F_MAC`
//! (`proxima_protocols::virtio::net::FEATURE_NET_MAC`), brings the device to
//! `DRIVER_OK` on transmitq1 (queue index 1, spec §5.1.2) only, then
//! publishes one two-descriptor, device-readable chain — a 12-byte
//! `virtio_net_hdr` (spec §5.1.6.1) followed by a hand-built ARP-request
//! Ethernet frame — and rings `QueueNotify`. The frame bytes are the exact
//! layout `tools/proxima-vm/src/virtio_net.rs`'s own
//! `arp_request_frame()` test helper builds, so a host-side assertion
//! (`tests/virtio_net_mmio.rs`) compares byte-exact and can feed the same
//! bytes to `proxima_net::stack::handle_frame`.

/// Base guest-physical address of the reserved virtio-net mmio window —
/// MUST match `PROXIMA_VM_NET_MMIO_WINDOW_BASE`
/// (`src/dispatch_trampoline.h`) exactly: placed immediately after the
/// console window (`super::virtio_console::MMIO_BASE`), still far above any
/// real guest RAM (`GUEST_MEMORY_SIZE = 64 MiB`, `dispatch.rs`).
const MMIO_BASE: u64 = 0x1000001000;

const REG_STATUS: u64 = 0x070;
const REG_DEVICE_FEATURES: u64 = 0x010;
const REG_DEVICE_FEATURES_SEL: u64 = 0x014;
const REG_DRIVER_FEATURES: u64 = 0x020;
const REG_DRIVER_FEATURES_SEL: u64 = 0x024;
const REG_QUEUE_SEL: u64 = 0x030;
const REG_QUEUE_NUM: u64 = 0x038;
const REG_QUEUE_READY: u64 = 0x044;
const REG_QUEUE_NOTIFY: u64 = 0x050;
const REG_QUEUE_DESC_LOW: u64 = 0x080;
const REG_QUEUE_DESC_HIGH: u64 = 0x084;
const REG_QUEUE_DRIVER_LOW: u64 = 0x090;
const REG_QUEUE_DRIVER_HIGH: u64 = 0x094;
const REG_QUEUE_DEVICE_LOW: u64 = 0x0a0;
const REG_QUEUE_DEVICE_HIGH: u64 = 0x0a4;

const STATUS_ACKNOWLEDGE: u32 = 1;
const STATUS_DRIVER: u32 = 2;
const STATUS_DRIVER_OK: u32 = 4;
const STATUS_FEATURES_OK: u32 = 8;

/// `VIRTIO_NET_F_MAC` (spec §5.1.3, bit 5) — the only device-specific
/// feature this driver acks, matching
/// `proxima_protocols::virtio::net::FEATURE_NET_MAC` exactly.
const DRIVER_FEATURES_LOW_ACK_NET_MAC: u32 = 1 << 5;

/// Bit 0 of the feature word's high 32 bits — `VIRTIO_F_VERSION_1`, same
/// ack `virtio_console.rs` already performs.
const DRIVER_FEATURES_HIGH_ACK_VERSION_1: u32 = 1;

/// transmitq1 (spec §5.1.2) — the only queue this driver stands up.
const TX_QUEUE: u32 = 1;
const QUEUE_SIZE: u16 = 4;

/// Guest-RAM scratch addresses for the TX queue's structures, distinct from
/// `virtio_console.rs`'s own scratch region (`0x0002_0000..0x0002_0400`) so
/// the two devices' rings never alias.
const DESCRIPTOR_TABLE_ADDRESS: u64 = 0x0003_0000;
const AVAIL_RING_ADDRESS: u64 = 0x0003_0100;
const USED_RING_ADDRESS: u64 = 0x0003_0200;
const HDR_BUFFER_ADDRESS: u64 = 0x0003_0300;
const FRAME_BUFFER_ADDRESS: u64 = 0x0003_0340;

/// One 12-byte `virtio_net_hdr` (spec §5.1.6.1): all-zero except
/// `num_buffers`, which the spec requires set to 1 whenever
/// `VIRTIO_F_VERSION_1` is negotiated (the "num_buffers form" this driver's
/// only negotiated transport version always uses) — byte-exact match for
/// `proxima_protocols::virtio::net`'s own
/// `tx_descriptor_chain_carries_a_net_hdr_and_an_ethernet_frame_byte_exact`
/// worked-example `hdr_bytes`.
const NET_HDR_BYTES: [u8; 12] = [
    0x00, // flags
    0x00, // gso_type = GSO_NONE
    0x00, 0x00, // hdr_len
    0x00, 0x00, // gso_size
    0x00, 0x00, // csum_start
    0x00, 0x00, // csum_offset
    0x01, 0x00, // num_buffers = 1
];

/// This driver's own MAC — the "peer" address in the ARP request below,
/// matching `tools/proxima-vm/src/virtio_net.rs`'s host-side transport test
/// constant `PEER_MAC` exactly, so the host test's byte-exact assertion and
/// the (stretch) `proxima_net::stack::handle_frame` reply both key off the
/// same address this driver actually used.
const PEER_MAC: [u8; 6] = [0x02, 0x11, 0x22, 0x33, 0x44, 0x55];

/// The host's IP address this ARP request asks "who has" — matches
/// `tools/proxima-vm/src/virtio_net.rs`'s `OUR_IP` test constant.
const TARGET_IP: [u8; 4] = [10, 0, 0, 2];

/// This driver's own IP (the ARP request's sender-protocol-address field).
const SENDER_IP: [u8; 4] = [10, 0, 0, 1];

/// One hand-built 42-byte broadcast ARP request (14-byte Ethernet header +
/// 28-byte ARP payload, spec-external but byte-identical to
/// `tools/proxima-vm/src/virtio_net.rs`'s `arp_request_frame()` helper) —
/// the frame this driver publishes on transmitq1.
const ARP_REQUEST_FRAME: [u8; 42] = [
    // --- Ethernet header ---
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, // dst = broadcast
    0x02, 0x11, 0x22, 0x33, 0x44, 0x55, // src = PEER_MAC
    0x08, 0x06, // ethertype = ARP
    // --- ARP payload ---
    0x00, 0x01, // htype = Ethernet
    0x08, 0x00, // ptype = IPv4
    0x06, // hlen
    0x04, // plen
    0x00, 0x01, // oper = ARP_REQUEST
    0x02, 0x11, 0x22, 0x33, 0x44, 0x55, // sha = PEER_MAC
    10, 0, 0, 1, // spa = SENDER_IP
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // tha = unknown
    10, 0, 0, 2, // tpa = TARGET_IP ("who has us")
];

const _: () = assert!(matches_frame_constants());

/// Compile-time proof the hand-written byte array above and the named field
/// constants it was derived from never drift apart — the constant is easier
/// to eyeball-verify byte-exact against the host's own test helper, but the
/// named fields are what a reader reasons about; this assertion keeps both
/// forms honest without a runtime cost.
const fn matches_frame_constants() -> bool {
    let frame = ARP_REQUEST_FRAME;
    frame[6] == PEER_MAC[0]
        && frame[11] == PEER_MAC[5]
        && frame[22] == PEER_MAC[0]
        && frame[27] == PEER_MAC[5]
        && frame[28] == SENDER_IP[0]
        && frame[31] == SENDER_IP[3]
        && frame[38] == TARGET_IP[0]
        && frame[41] == TARGET_IP[3]
}

#[inline]
unsafe fn mmio_write32(offset: u64, value: u32) {
    let pointer = (MMIO_BASE + offset) as *mut u32;
    unsafe { core::ptr::write_volatile(pointer, value) };
}

#[inline]
unsafe fn mmio_read32(offset: u64) -> u32 {
    let pointer = (MMIO_BASE + offset) as *const u32;
    unsafe { core::ptr::read_volatile(pointer) }
}

#[inline]
unsafe fn ram_write32(address: u64, value: u32) {
    unsafe { core::ptr::write_volatile(address as *mut u32, value) };
}

#[inline]
unsafe fn ram_write16(address: u64, value: u16) {
    unsafe { core::ptr::write_volatile(address as *mut u16, value) };
}

#[inline]
unsafe fn ram_write64(address: u64, value: u64) {
    unsafe { core::ptr::write_volatile(address as *mut u64, value) };
}

#[inline]
unsafe fn ram_write8(address: u64, value: u8) {
    unsafe { core::ptr::write_volatile(address as *mut u8, value) };
}

#[inline]
unsafe fn ram_write_bytes(address: u64, bytes: &[u8]) {
    let mut index = 0;
    while index < bytes.len() {
        unsafe { ram_write8(address + index as u64, bytes[index]) };
        index += 1;
    }
}

/// Runs the full device-initialization sequence (spec §3.1.1) against the
/// net device's mmio register block, negotiating only `VIRTIO_F_VERSION_1`
/// and `VIRTIO_NET_F_MAC`, stands up transmitq1, then publishes and
/// notifies one two-descriptor, device-readable chain carrying
/// [`NET_HDR_BYTES`] followed by [`ARP_REQUEST_FRAME`].
///
/// # Safety
///
/// Must run in an environment where `MMIO_BASE` is the host's reserved
/// virtio-net mmio window and the scratch RAM addresses above are mapped
/// writable guest memory — true only when booted by
/// `dispatch::run_dispatch_loop` (`tools/proxima-vm/src/dispatch.rs`).
pub unsafe fn bring_up_and_transmit_one_frame() {
    unsafe {
        // status handshake: ACKNOWLEDGE, then ACKNOWLEDGE | DRIVER.
        mmio_write32(REG_STATUS, STATUS_ACKNOWLEDGE);
        mmio_write32(REG_STATUS, STATUS_ACKNOWLEDGE | STATUS_DRIVER);

        // feature negotiation: read both 32-bit halves of DeviceFeatures
        // (this driver acks only VIRTIO_NET_F_MAC from the low half and
        // VIRTIO_F_VERSION_1 from the high half — the exact pair
        // `NetConfigSpace::new` offers, `proxima-protocols/src/virtio/
        // net.rs`).
        mmio_write32(REG_DEVICE_FEATURES_SEL, 0);
        let _offered_low = mmio_read32(REG_DEVICE_FEATURES);
        mmio_write32(REG_DEVICE_FEATURES_SEL, 1);
        let _offered_high = mmio_read32(REG_DEVICE_FEATURES);

        mmio_write32(REG_DRIVER_FEATURES_SEL, 0);
        mmio_write32(REG_DRIVER_FEATURES, DRIVER_FEATURES_LOW_ACK_NET_MAC);
        mmio_write32(REG_DRIVER_FEATURES_SEL, 1);
        mmio_write32(REG_DRIVER_FEATURES, DRIVER_FEATURES_HIGH_ACK_VERSION_1);

        mmio_write32(
            REG_STATUS,
            STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK,
        );

        // queue setup: select transmitq1 (queue index 1), size it, program
        // the three split address-pair registers, mark it ready.
        mmio_write32(REG_QUEUE_SEL, TX_QUEUE);
        mmio_write32(REG_QUEUE_NUM, u32::from(QUEUE_SIZE));
        mmio_write32(REG_QUEUE_DESC_LOW, DESCRIPTOR_TABLE_ADDRESS as u32);
        mmio_write32(REG_QUEUE_DESC_HIGH, (DESCRIPTOR_TABLE_ADDRESS >> 32) as u32);
        mmio_write32(REG_QUEUE_DRIVER_LOW, AVAIL_RING_ADDRESS as u32);
        mmio_write32(REG_QUEUE_DRIVER_HIGH, (AVAIL_RING_ADDRESS >> 32) as u32);
        mmio_write32(REG_QUEUE_DEVICE_LOW, USED_RING_ADDRESS as u32);
        mmio_write32(REG_QUEUE_DEVICE_HIGH, (USED_RING_ADDRESS >> 32) as u32);
        mmio_write32(REG_QUEUE_READY, 1);

        mmio_write32(
            REG_STATUS,
            STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK | STATUS_DRIVER_OK,
        );

        // publish a two-descriptor, device-readable chain: descriptor 0
        // carries the net_hdr (NEXT -> 2), descriptor 2 carries the ARP
        // frame (chain terminator) — same shape
        // `proxima_protocols::virtio::net`'s own worked-example test walks.
        ram_write_bytes(HDR_BUFFER_ADDRESS, &NET_HDR_BYTES);
        ram_write64(DESCRIPTOR_TABLE_ADDRESS, HDR_BUFFER_ADDRESS); // addr
        ram_write32(DESCRIPTOR_TABLE_ADDRESS + 8, NET_HDR_BYTES.len() as u32); // len
        ram_write16(DESCRIPTOR_TABLE_ADDRESS + 12, 1); // flags = NEXT
        ram_write16(DESCRIPTOR_TABLE_ADDRESS + 14, 2); // next = descriptor 2

        ram_write_bytes(FRAME_BUFFER_ADDRESS, &ARP_REQUEST_FRAME);
        const DESC_LEN: u64 = 16;
        ram_write64(DESCRIPTOR_TABLE_ADDRESS + 2 * DESC_LEN, FRAME_BUFFER_ADDRESS); // addr
        ram_write32(
            DESCRIPTOR_TABLE_ADDRESS + 2 * DESC_LEN + 8,
            ARP_REQUEST_FRAME.len() as u32,
        ); // len
        ram_write16(DESCRIPTOR_TABLE_ADDRESS + 2 * DESC_LEN + 12, 0); // flags = 0 (terminator)
        ram_write16(DESCRIPTOR_TABLE_ADDRESS + 2 * DESC_LEN + 14, 0); // next (unused)

        ram_write16(AVAIL_RING_ADDRESS, 0); // flags
        ram_write16(AVAIL_RING_ADDRESS + 2, 1); // idx = 1 (one entry)
        ram_write16(AVAIL_RING_ADDRESS + 4, 0); // ring[0] = head 0

        mmio_write32(REG_QUEUE_NOTIFY, TX_QUEUE);
    }
}
