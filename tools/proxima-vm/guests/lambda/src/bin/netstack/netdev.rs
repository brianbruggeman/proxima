//! Steady-state virtio-net driver for the netstack guest, built on the
//! sans-IO driver codec (`proxima_protocols::virtio::DriverQueue`). Receive is
//! a source pipe ([`RxSource`]: `() -> Option<(addr, len)>`) and transmit a
//! sink pipe ([`TxSink`]: `(addr, len) -> ()`); both are the only code here
//! that touches queue memory, and register offsets and status bits are the
//! shared protocol crate's own constants.

use core::cell::Cell;
use core::ptr::{read_volatile, slice_from_raw_parts, write_volatile};
use core::sync::atomic::{Ordering, fence};

use proxima_primitives::pipe::Pipe;
use proxima_protocols::virtio::mmio::{
    REG_DEVICE_FEATURES_SEL, REG_DRIVER_FEATURES, REG_DRIVER_FEATURES_SEL, REG_QUEUE_DESC_HIGH,
    REG_QUEUE_DESC_LOW, REG_QUEUE_DEVICE_HIGH, REG_QUEUE_DEVICE_LOW, REG_QUEUE_DRIVER_HIGH,
    REG_QUEUE_DRIVER_LOW, REG_QUEUE_NOTIFY, REG_QUEUE_NUM, REG_QUEUE_READY, REG_QUEUE_SEL,
    REG_STATUS,
};
use proxima_protocols::virtio::status::{
    STATUS_ACKNOWLEDGE, STATUS_DRIVER, STATUS_DRIVER_OK, STATUS_FEATURES_OK,
};
use proxima_protocols::virtio::{
    CONFIG_SPACE_BASE, DecodeError, DriverQueue, FEATURE_NET_MAC, FEATURE_NET_STATUS,
    FEATURE_VERSION_1, NET_HDR_LEN, NET_STATUS_LINK_UP, NetHdrFields, RECEIVEQ1, TRANSMITQ1,
    USED_ELEM_LEN, write_net_hdr,
};

pub const QUEUE_SIZE: usize = 16;
const BUFFER_LEN: usize = 2048;
const USED_SNAPSHOT_LEN: usize = 4 + QUEUE_SIZE * USED_ELEM_LEN;

/// Guest-physical base of the virtio-net mmio window — must match
/// `PROXIMA_VM_NET_MMIO_WINDOW_BASE` in `src/dispatch_trampoline.h`.
const MMIO_BASE: u64 = 0x1000001000;

#[repr(C, align(16))]
struct Aligned<const BYTES: usize>([u8; BYTES]);

#[repr(C)]
struct Region {
    rx_table: Aligned<256>,
    rx_avail: Aligned<40>,
    rx_used: Aligned<136>,
    tx_table: Aligned<256>,
    tx_avail: Aligned<40>,
    tx_used: Aligned<136>,
    rx_buffers: [[u8; BUFFER_LEN]; QUEUE_SIZE],
    tx_buffers: [[u8; BUFFER_LEN]; QUEUE_SIZE],
}

static mut REGION: Region = Region {
    rx_table: Aligned([0; 256]),
    rx_avail: Aligned([0; 40]),
    rx_used: Aligned([0; 136]),
    tx_table: Aligned([0; 256]),
    tx_avail: Aligned([0; 40]),
    tx_used: Aligned([0; 136]),
    rx_buffers: [[0; BUFFER_LEN]; QUEUE_SIZE],
    tx_buffers: [[0; BUFFER_LEN]; QUEUE_SIZE],
};

/// # Safety
///
/// The guest is single-threaded and each pipe touches disjoint fields of the
/// region, so the aliasing `&mut` views never overlap in use.
unsafe fn region() -> &'static mut Region {
    let pointer: *mut Region = &raw mut REGION;
    unsafe { &mut *pointer }
}

fn mmio_write(offset: u64, value: u32) {
    unsafe { write_volatile((MMIO_BASE + offset) as *mut u32, value) };
}

fn mmio_read(offset: u64) -> u32 {
    unsafe { read_volatile((MMIO_BASE + offset) as *const u32) }
}

/// The device's MAC from config space (spec §5.1.4), which the host fixes.
pub fn read_mac() -> [u8; 6] {
    let low = mmio_read(CONFIG_SPACE_BASE).to_le_bytes();
    let high = mmio_read(CONFIG_SPACE_BASE + 4).to_le_bytes();
    [low[0], low[1], low[2], low[3], high[0], high[1]]
}

/// Whether the host still reports link-up; the host takes it down to end the
/// run.
pub fn link_up() -> bool {
    let status = (mmio_read(CONFIG_SPACE_BASE + 4) >> 16) as u16;
    status & NET_STATUS_LINK_UP != 0
}

fn volatile_copy(source: *const u8, destination: &mut [u8]) {
    for (index, byte) in destination.iter_mut().enumerate() {
        *byte = unsafe { read_volatile(source.add(index)) };
    }
}

fn snapshot_used(used: &Aligned<136>) -> [u8; USED_SNAPSHOT_LEN] {
    let source = used.0.as_ptr();
    let mut copy = [0u8; USED_SNAPSHOT_LEN];
    let (header, entries) = copy.split_at_mut(4);
    volatile_copy(source, header);
    fence(Ordering::Acquire);
    volatile_copy(unsafe { source.add(4) }, entries);
    copy
}

fn queue_address(bytes: &[u8]) -> u64 {
    bytes.as_ptr() as u64
}

fn program_queue(queue: u16, table: u64, avail: u64, used: u64) {
    let registers = [
        (REG_QUEUE_SEL, u32::from(queue)),
        (REG_QUEUE_NUM, QUEUE_SIZE as u32),
        (REG_QUEUE_DESC_LOW, table as u32),
        (REG_QUEUE_DESC_HIGH, (table >> 32) as u32),
        (REG_QUEUE_DRIVER_LOW, avail as u32),
        (REG_QUEUE_DRIVER_HIGH, (avail >> 32) as u32),
        (REG_QUEUE_DEVICE_LOW, used as u32),
        (REG_QUEUE_DEVICE_HIGH, (used >> 32) as u32),
        (REG_QUEUE_READY, 1),
    ];
    for (offset, value) in registers {
        mmio_write(offset, value);
    }
}

fn negotiate() {
    let driver = STATUS_ACKNOWLEDGE | STATUS_DRIVER;
    mmio_write(REG_STATUS, u32::from(STATUS_ACKNOWLEDGE));
    mmio_write(REG_STATUS, u32::from(driver));
    mmio_write(REG_DEVICE_FEATURES_SEL, 0);
    let wanted = FEATURE_NET_MAC | FEATURE_NET_STATUS | FEATURE_VERSION_1;
    mmio_write(REG_DRIVER_FEATURES_SEL, 0);
    mmio_write(REG_DRIVER_FEATURES, wanted as u32);
    mmio_write(REG_DRIVER_FEATURES_SEL, 1);
    mmio_write(REG_DRIVER_FEATURES, (wanted >> 32) as u32);
    mmio_write(REG_STATUS, u32::from(driver | STATUS_FEATURES_OK));
}

/// Runs the device-initialization sequence (spec §3.1.1) and returns the
/// receive source and transmit sink pipes, with every receive buffer posted.
///
/// # Safety
///
/// Must run once, under `net_run::run_net_guest`, which maps the net mmio
/// window and the guest RAM `REGION` lives in.
pub unsafe fn bring_up() -> Result<(RxSource, TxSink), DecodeError> {
    let region = unsafe { region() };
    negotiate();
    let mut rx = DriverQueue::<QUEUE_SIZE>::new(&mut region.rx_table.0)?;
    let tx = DriverQueue::<QUEUE_SIZE>::new(&mut region.tx_table.0)?;
    program_queue(
        RECEIVEQ1,
        queue_address(&region.rx_table.0),
        queue_address(&region.rx_avail.0),
        queue_address(&region.rx_used.0),
    );
    program_queue(
        TRANSMITQ1,
        queue_address(&region.tx_table.0),
        queue_address(&region.tx_avail.0),
        queue_address(&region.tx_used.0),
    );
    let ready = STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK | STATUS_DRIVER_OK;
    mmio_write(REG_STATUS, u32::from(ready));
    for buffer in &region.rx_buffers {
        let writable = [(queue_address(buffer), BUFFER_LEN as u32)];
        rx.post(&mut region.rx_table.0, &mut region.rx_avail.0, &[], &writable)?;
    }
    Ok((
        RxSource {
            queue: Cell::new(rx),
            leased: Cell::new(None),
        },
        TxSink {
            queue: Cell::new(tx),
            posted: Cell::new(0),
        },
    ))
}

/// Source pipe over receiveq1. Each call kicks the device so the host can
/// deliver pending frames, then yields the next received frame as the
/// guest-physical `(address, length)` of its Ethernet bytes. The frame stays
/// valid until the next call, which reposts its buffer.
pub struct RxSource {
    queue: Cell<DriverQueue<QUEUE_SIZE>>,
    leased: Cell<Option<u64>>,
}

impl RxSource {
    fn poll_frame(&self) -> Result<Option<(u64, u32)>, DecodeError> {
        let region = unsafe { region() };
        let mut queue = self.queue.get();
        if let Some(address) = self.leased.take() {
            let writable = [(address, BUFFER_LEN as u32)];
            queue.post(&mut region.rx_table.0, &mut region.rx_avail.0, &[], &writable)?;
        }
        mmio_write(REG_QUEUE_NOTIFY, u32::from(RECEIVEQ1));
        let used = snapshot_used(&region.rx_used);
        let reaped = queue.reap(&mut region.rx_table.0, &used)?;
        self.queue.set(queue);
        let Some((elem, address)) = reaped else {
            return Ok(None);
        };
        self.leased.set(Some(address));
        let header = NET_HDR_LEN as u32;
        Ok((elem.len > header).then(|| (address + u64::from(header), elem.len - header)))
    }
}

impl Pipe for RxSource {
    type In = ();
    type Out = Option<(u64, u32)>;
    type Err = DecodeError;

    async fn call(&self, _input: ()) -> Result<Self::Out, DecodeError> {
        self.poll_frame()
    }
}

/// Sink pipe over transmitq1: copies the frame at guest-physical
/// `(address, length)` behind a `virtio_net_hdr` into the next transmit
/// buffer, posts it, and kicks the device.
pub struct TxSink {
    queue: Cell<DriverQueue<QUEUE_SIZE>>,
    posted: Cell<usize>,
}

impl TxSink {
    fn transmit(&self, address: u64, length: u32) -> Result<(), DecodeError> {
        let region = unsafe { region() };
        let mut queue = self.queue.get();
        let used = snapshot_used(&region.tx_used);
        while queue.reap(&mut region.tx_table.0, &used)?.is_some() {}
        let frame_len = length as usize;
        let buffer = &mut region.tx_buffers[self.posted.get() % QUEUE_SIZE];
        let total = NET_HDR_LEN + frame_len;
        if total > BUFFER_LEN {
            return Err(DecodeError::Truncated {
                need: total,
                got: BUFFER_LEN,
            });
        }
        let fields = NetHdrFields {
            num_buffers: 1,
            ..NetHdrFields::default()
        };
        write_net_hdr(&mut buffer[..NET_HDR_LEN], fields)?;
        let frame = unsafe { &*slice_from_raw_parts(address as *const u8, frame_len) };
        buffer[NET_HDR_LEN..total].copy_from_slice(frame);
        let readable = [(queue_address(buffer), total as u32)];
        queue.post(&mut region.tx_table.0, &mut region.tx_avail.0, &readable, &[])?;
        self.queue.set(queue);
        self.posted.set(self.posted.get() + 1);
        mmio_write(REG_QUEUE_NOTIFY, u32::from(TRANSMITQ1));
        Ok(())
    }
}

impl Pipe for TxSink {
    type In = (u64, u32);
    type Out = ();
    type Err = DecodeError;

    async fn call(&self, (address, length): (u64, u32)) -> Result<(), DecodeError> {
        self.transmit(address, length)
    }
}
