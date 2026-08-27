//! Minimal virtio-console driver bring-up over the mmio transport
//! (VIRTIO 1.2 spec §3.1.1, register layout §4.2.2), driven entirely by
//! ordinary `core::ptr::{read_volatile, write_volatile}` loads/stores — no
//! inline assembly needed, since the host's data-abort handler
//! (`src/backend_macos.c`'s `handle_mmio_data_abort`) decodes a plain
//! 32-bit `ldr`/`str` from the trapped instruction's own syndrome. Every
//! register offset below matches `proxima-protocols/src/virtio/mmio.rs`'s
//! `REG_*` constants exactly (mirrors this crate's `CHILD_REQUEST_*_VERB`
//! constants, which already duplicate `proxima-protocols/src/process`'s
//! wire discriminants for the same reason: a bare `no_std`/`no_alloc` guest
//! crate carries no dependency on the host-tier protocol crate).
//!
//! Scope: brings the device to `DRIVER_OK` on queue 0 only (this crate's
//! `ConsoleTransport` host counterpart drives exactly one queue too), then
//! publishes one single-descriptor, device-readable chain and rings
//! `QueueNotify` — proving one byte crosses the ring through a real VM
//! exit, not a spec-complete console device.

/// Base guest-physical address of the reserved virtio-mmio window —
/// MUST match `PROXIMA_VM_MMIO_WINDOW_BASE`
/// (`src/dispatch_trampoline.h`) exactly; deliberately far above any real
/// guest RAM (`GUEST_MEMORY_SIZE = 64 MiB`, `dispatch.rs`) so it can never
/// collide with an ELF segment or the stack.
const MMIO_BASE: u64 = 0x1000000000;

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

/// Bit 0 of the feature word's high 32 bits — `VIRTIO_F_VERSION_1`
/// (`proxima-protocols/src/virtio/status.rs`'s `FEATURE_VERSION_1`, bit 32
/// of the full 64-bit space).
const DRIVER_FEATURES_HIGH_ACK_VERSION_1: u32 = 1;

const QUEUE_SIZE: u16 = 4;

/// Guest-RAM scratch addresses for this queue's structures — real writable
/// memory (the stack reservation `dispatch.rs::run_dispatch_loop` maps),
/// never the mmio window above. Sized with headroom against
/// `QUEUE_SIZE`'s fixed layout (16-byte descriptors, 4 + 2*n avail bytes).
const DESCRIPTOR_TABLE_ADDRESS: u64 = 0x0002_0000;
const AVAIL_RING_ADDRESS: u64 = 0x0002_0100;
const USED_RING_ADDRESS: u64 = 0x0002_0200;
const TX_BUFFER_ADDRESS: u64 = 0x0002_0300;

/// The one byte this proof transmits — chosen distinct from every other
/// byte this guest already emits (`0x00`/`0x03`, the hypercall response
/// discriminants above) so a host-side assertion can never mistake it for
/// hypercall-channel output.
const TX_BYTE: u8 = 0xab;

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

/// Runs the full device-initialization sequence (spec §3.1.1) against the
/// mmio register block, then publishes and notifies one single-descriptor,
/// device-readable chain carrying [`TX_BYTE`].
///
/// # Safety
///
/// Must run in an environment where `MMIO_BASE` is the host's reserved
/// virtio-mmio window and the scratch RAM addresses above are mapped
/// writable guest memory — true only when booted by
/// `dispatch::run_dispatch_loop` (`tools/proxima-vm/src/dispatch.rs`).
pub unsafe fn bring_up_and_transmit_one_byte() {
    unsafe {
        // status handshake: ACKNOWLEDGE, then ACKNOWLEDGE | DRIVER.
        mmio_write32(REG_STATUS, STATUS_ACKNOWLEDGE);
        mmio_write32(REG_STATUS, STATUS_ACKNOWLEDGE | STATUS_DRIVER);

        // feature negotiation: read both 32-bit halves of DeviceFeatures
        // (the low half is unused by this proof; VIRTIO_F_VERSION_1 lives
        // in the high half), then ack only VIRTIO_F_VERSION_1.
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

        // queue setup: select queue 0, size it, program the three split
        // address-pair registers, mark it ready.
        mmio_write32(REG_QUEUE_SEL, 0);
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

        // publish one device-readable, single-descriptor chain carrying
        // TX_BYTE (descriptor 0, no NEXT/WRITE flags), then the avail-ring
        // entry naming it (idx 1, ring[0] = head 0).
        ram_write8(TX_BUFFER_ADDRESS, TX_BYTE);
        ram_write64(DESCRIPTOR_TABLE_ADDRESS, TX_BUFFER_ADDRESS); // addr
        ram_write32(DESCRIPTOR_TABLE_ADDRESS + 8, 1); // len = 1 byte
        ram_write16(DESCRIPTOR_TABLE_ADDRESS + 12, 0); // flags = 0
        ram_write16(DESCRIPTOR_TABLE_ADDRESS + 14, 0); // next (unused)

        ram_write16(AVAIL_RING_ADDRESS, 0); // flags
        ram_write16(AVAIL_RING_ADDRESS + 2, 1); // idx = 1 (one entry)
        ram_write16(AVAIL_RING_ADDRESS + 4, 0); // ring[0] = head 0

        mmio_write32(REG_QUEUE_NOTIFY, 0);
    }
}
