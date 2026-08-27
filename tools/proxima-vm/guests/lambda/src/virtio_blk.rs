//! Minimal virtio-blk driver bring-up over the mmio transport (VIRTIO 1.2
//! spec §5.2), mirroring `super::virtio_net`'s shape exactly: ordinary
//! `core::ptr::{read_volatile, write_volatile}` loads/stores, no inline
//! assembly — the host's data-abort handler (`src/backend_macos.c`'s
//! `handle_mmio_data_abort`) services a `QueueNotify` register write
//! SYNCHRONOUSLY, inside the same trap that resumes the guest, so by the
//! time `mmio_write32(REG_QUEUE_NOTIFY, ...)` returns here the host has
//! already walked the descriptor chain and written the data/status bytes
//! back into this guest's own RAM — no polling loop needed, the same
//! implicit synchrony `virtio_net.rs`'s own module doc relies on for its
//! used-ring completion.
//!
//! Scope: negotiates only `VIRTIO_F_VERSION_1`, brings the device to
//! `DRIVER_OK` on `requestq` (queue index 0, spec §5.2.2) only, then
//! publishes one three-descriptor `IN` chain for sector 0 (header
//! device-readable, data + status device-writable), reads the data buffer
//! back and compares it byte-for-byte against [`EXPECTED_SECTOR_PATTERN`]
//! (the same pattern `dispatch.rs`'s `run_dispatch_loop` seeds into sector 0
//! before boot) — and only on a match publishes a second, `OUT` chain that
//! writes [`EXPECTED_SECTOR_PATTERN`] back out to sector 1. A host-side test
//! observes both outcomes through the blk-emitted channel
//! (`crate::mmio_trampoline::proxima_vm_mmio_service_blk`'s own encoding),
//! which carries the actual data bytes the device moved for each request —
//! so a guest that failed its own comparison and skipped the OUT request is
//! visible to the host as one serviced request instead of two, not a silent
//! partial success.

/// Base guest-physical address of the reserved virtio-blk mmio window —
/// MUST match `PROXIMA_VM_BLK_MMIO_WINDOW_BASE`
/// (`src/dispatch_trampoline.h`) exactly: the third window, placed
/// immediately after the net window (`super::virtio_net::MMIO_BASE`).
const MMIO_BASE: u64 = 0x1000002000;

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

/// Bit 0 of the feature word's high 32 bits — `VIRTIO_F_VERSION_1`, the only
/// bit `BlkConfigSpace::new` offers (`proxima-protocols/src/virtio/blk.rs`).
const DRIVER_FEATURES_HIGH_ACK_VERSION_1: u32 = 1;

/// `requestq` (spec §5.2.2) — the only queue this driver stands up.
const REQUEST_QUEUE: u32 = 0;
const QUEUE_SIZE: u16 = 4;
const DESC_LEN: u64 = 16;

/// `VIRTIO_BLK_T_IN` / `VIRTIO_BLK_T_OUT` (spec §5.2.6), matching
/// `proxima_protocols::virtio::blk::RequestType` exactly.
const BLK_T_IN: u32 = 0;
const BLK_T_OUT: u32 = 1;

/// Guest-RAM scratch addresses for `requestq`'s structures, distinct from
/// `virtio_console.rs`'s (`0x0002_0000`) and `virtio_net.rs`'s
/// (`0x0003_0000`) own scratch regions so no two devices' rings ever alias.
const DESCRIPTOR_TABLE_ADDRESS: u64 = 0x0004_0000;
const AVAIL_RING_ADDRESS: u64 = 0x0004_0100;
const USED_RING_ADDRESS: u64 = 0x0004_0200;
const HEADER_BUFFER_ADDRESS: u64 = 0x0004_0300;
const DATA_BUFFER_ADDRESS: u64 = 0x0004_0400;
const STATUS_BYTE_ADDRESS: u64 = 0x0004_0600;

const SECTOR_LEN: usize = 512;
const BLK_REQ_HEADER_LEN: usize = 16;

/// The sector-0 pattern `dispatch.rs`'s `run_dispatch_loop` seeds into the
/// host's local block store before boot (`(index % 256) as u8`, hand-built
/// here byte-by-byte since a bare `no_std`/`no_alloc` guest has no
/// `core::iter` collect-into-array convenience worth pulling in for one
/// buffer) — this driver's own expected value for the `IN` request's data,
/// and the value it writes back out via the `OUT` request.
const fn expected_sector_pattern() -> [u8; SECTOR_LEN] {
    let mut pattern = [0u8; SECTOR_LEN];
    let mut index = 0;
    while index < SECTOR_LEN {
        pattern[index] = (index % 256) as u8;
        index += 1;
    }
    pattern
}
const EXPECTED_SECTOR_PATTERN: [u8; SECTOR_LEN] = expected_sector_pattern();

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
unsafe fn ram_read8(address: u64) -> u8 {
    unsafe { core::ptr::read_volatile(address as *const u8) }
}

#[inline]
unsafe fn ram_write_bytes(address: u64, bytes: &[u8]) {
    let mut index = 0;
    while index < bytes.len() {
        unsafe { ram_write8(address + index as u64, bytes[index]) };
        index += 1;
    }
}

/// Writes one `virtio_blk_req` header (spec §5.2.6) at [`HEADER_BUFFER_ADDRESS`].
#[inline]
unsafe fn write_request_header(request_type: u32, sector: u64) {
    unsafe {
        ram_write32(HEADER_BUFFER_ADDRESS, request_type);
        ram_write32(HEADER_BUFFER_ADDRESS + 4, 0); // reserved
        ram_write64(HEADER_BUFFER_ADDRESS + 8, sector);
    }
}

/// Publishes the fixed three-descriptor chain (header, data, status) at
/// head index 0, `data_writable` selecting `IN` (device-writable data) vs
/// `OUT` (device-readable data) shape, and notifies `requestq` — the
/// notify's own trap synchronously services the request before this
/// function returns (see module doc). `avail_idx` is the free-running
/// avail-ring index this call publishes (spec §2.7.6: strictly increasing,
/// never reset per-chain), so the caller passes 1 for the first request and
/// 2 for the second.
#[inline]
unsafe fn publish_and_notify(data_writable: bool, avail_idx: u16) {
    unsafe {
        ram_write64(DESCRIPTOR_TABLE_ADDRESS, HEADER_BUFFER_ADDRESS);
        ram_write32(DESCRIPTOR_TABLE_ADDRESS + 8, BLK_REQ_HEADER_LEN as u32);
        ram_write16(DESCRIPTOR_TABLE_ADDRESS + 12, 1); // flags = NEXT
        ram_write16(DESCRIPTOR_TABLE_ADDRESS + 14, 1); // next = descriptor 1

        let data_flags: u16 = 1 | if data_writable { 2 } else { 0 }; // NEXT | (WRITE?)
        ram_write64(DESCRIPTOR_TABLE_ADDRESS + DESC_LEN, DATA_BUFFER_ADDRESS);
        ram_write32(DESCRIPTOR_TABLE_ADDRESS + DESC_LEN + 8, SECTOR_LEN as u32);
        ram_write16(DESCRIPTOR_TABLE_ADDRESS + DESC_LEN + 12, data_flags);
        ram_write16(DESCRIPTOR_TABLE_ADDRESS + DESC_LEN + 14, 2); // next = descriptor 2

        ram_write64(DESCRIPTOR_TABLE_ADDRESS + 2 * DESC_LEN, STATUS_BYTE_ADDRESS);
        ram_write32(DESCRIPTOR_TABLE_ADDRESS + 2 * DESC_LEN + 8, 1);
        ram_write16(DESCRIPTOR_TABLE_ADDRESS + 2 * DESC_LEN + 12, 2); // flags = WRITE (terminator)
        ram_write16(DESCRIPTOR_TABLE_ADDRESS + 2 * DESC_LEN + 14, 0);

        ram_write16(AVAIL_RING_ADDRESS, 0); // flags
        ram_write16(AVAIL_RING_ADDRESS + 2, avail_idx);
        let ring_slot = u64::from((avail_idx - 1) % QUEUE_SIZE);
        ram_write16(AVAIL_RING_ADDRESS + 4 + ring_slot * 2, 0); // ring[slot] = head 0

        mmio_write32(REG_QUEUE_NOTIFY, REQUEST_QUEUE);
    }
}

/// Runs the full device-initialization sequence (spec §3.1.1) against the
/// blk device's mmio register block, negotiating only `VIRTIO_F_VERSION_1`,
/// stands up `requestq`, submits an `IN` request for sector 0, verifies the
/// data descriptor byte-for-byte against [`EXPECTED_SECTOR_PATTERN`], and —
/// only on a match — submits an `OUT` request writing the same pattern to
/// sector 1.
///
/// # Safety
///
/// Must run in an environment where `MMIO_BASE` is the host's reserved
/// virtio-blk mmio window and the scratch RAM addresses above are mapped
/// writable guest memory — true only when booted by
/// `dispatch::run_dispatch_loop` (`tools/proxima-vm/src/dispatch.rs`).
pub unsafe fn bring_up_and_exercise_one_sector() {
    unsafe {
        mmio_write32(REG_STATUS, STATUS_ACKNOWLEDGE);
        mmio_write32(REG_STATUS, STATUS_ACKNOWLEDGE | STATUS_DRIVER);

        mmio_write32(REG_DEVICE_FEATURES_SEL, 0);
        let _offered_low = mmio_read32(REG_DEVICE_FEATURES);
        mmio_write32(REG_DEVICE_FEATURES_SEL, 1);
        let _offered_high = mmio_read32(REG_DEVICE_FEATURES);

        mmio_write32(REG_DRIVER_FEATURES_SEL, 0);
        mmio_write32(REG_DRIVER_FEATURES, 0);
        mmio_write32(REG_DRIVER_FEATURES_SEL, 1);
        mmio_write32(REG_DRIVER_FEATURES, DRIVER_FEATURES_HIGH_ACK_VERSION_1);

        mmio_write32(
            REG_STATUS,
            STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK,
        );

        mmio_write32(REG_QUEUE_SEL, REQUEST_QUEUE);
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

        // --- IN: read sector 0, verify against the host-seeded pattern ---
        write_request_header(BLK_T_IN, 0);
        publish_and_notify(true, 1);

        let mut matches = true;
        let mut index = 0;
        while index < SECTOR_LEN {
            if ram_read8(DATA_BUFFER_ADDRESS + index as u64) != EXPECTED_SECTOR_PATTERN[index] {
                matches = false;
                break;
            }
            index += 1;
        }

        if matches {
            // --- OUT: write the verified pattern to sector 1 ---
            ram_write_bytes(DATA_BUFFER_ADDRESS, &EXPECTED_SECTOR_PATTERN);
            write_request_header(BLK_T_OUT, 1);
            publish_and_notify(false, 2);
        }
    }
}
