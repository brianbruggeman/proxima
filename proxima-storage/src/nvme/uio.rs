//! UIO + physical-addressing NVMe device backend (std, Linux).
//!
//! The no-IOMMU path a userspace NVMe driver takes when there is no usable
//! IOMMU (or, like a QEMU emulated controller, the vIOMMU won't translate for
//! the emulated device): bind the controller to `uio_pci_generic`, mmap BAR0
//! from sysfs `resource0`, and use physical DMA addresses read from
//! `/proc/self/pagemap` over `mlock`-pinned memory — exactly the model DPDK/SPDK
//! use with `uio`. The controller init + admin queue here drive the SAME
//! [`proxima_protocols::nvme`] (`CommandBuilder`, `CompletionEntry`, ring FSM) the
//! I/O path does, so the codec is exercised on admin commands too. Proven
//! against a QEMU emulated NVMe 1.4 controller.
//!
//! This is the std I/O facade; the engine + codec it feeds stay no_std.

use core::cell::Cell;
use core::ffi::c_void;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::ptr::{self, read_volatile as rd, write_volatile as wr};

use proxima_protocols::nvme::{
    CommandBuilder, CompletionEntry, CompletionRing, DecodeError, StatusField, SubmissionRing,
    command, completion,
};

use crate::nvme::backend::QueueBackend;

const BAR_SIZE: usize = 0x4000;
const DMA_SIZE: usize = 2 * 1024 * 1024;
const ADMIN_DEPTH: u32 = 32;
const CC_ENABLE: u32 = 0x0046_0001; // IOCQES=4, IOSQES=6, EN=1
const SQE_LEN: usize = command::ENTRY_LEN;
const CQE_LEN: usize = completion::ENTRY_LEN;

// region layout — each structure is page-aligned and <=4KB, so a single PRP
// entry addresses it and `pagemap` gives one physical frame per structure.
const ASQ_O: usize = 0x0000;
const ACQ_O: usize = 0x1000;
const ISQ_O: usize = 0x3000;
const ICQ_O: usize = 0x4000;
const POOL_O: usize = 0x5000;

unsafe extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64)
    -> *mut c_void;
    fn mlock(addr: *const c_void, len: usize) -> i32;
}

fn phys(pagemap: &File, vaddr: usize) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    pagemap.read_exact_at(&mut buf, (vaddr / 4096 * 8) as u64)?;
    let entry = u64::from_le_bytes(buf);
    if entry & (1 << 63) == 0 {
        return Err(io::Error::other("page not present in pagemap"));
    }
    Ok(((entry & ((1u64 << 55) - 1)) << 12) | (vaddr as u64 & 0xfff))
}

fn codec_err(error: DecodeError) -> io::Error {
    io::Error::other(format!("nvme codec: {error}"))
}

unsafe fn w32(base: usize, off: usize, value: u32) {
    unsafe { wr((base + off) as *mut u32, value) }
}
unsafe fn r32(base: usize, off: usize) -> u32 {
    unsafe { rd((base + off) as *const u32) }
}
unsafe fn w64(base: usize, off: usize, value: u64) {
    unsafe { wr((base + off) as *mut u64, value) }
}

/// A live NVMe controller driven over UIO with physical addressing. Holds the
/// mmap'd BAR, the pinned DMA region, and the I/O queue pair created at init;
/// implements [`QueueBackend`] over that I/O queue so a [`crate::nvme::QueuePair`]
/// engine drives it.
pub struct UioNvme {
    _bar_file: File,
    pagemap: File,
    bar: usize,
    dma: usize,
    io_sq: usize,
    io_cq: usize,
    io_sq_db: usize,
    io_cq_db: usize,
    pool_next: Cell<usize>,
    depth: u32,
}

impl UioNvme {
    /// Bring up the controller at PCI `bdf` (e.g. `"0000:00:02.0"`, already
    /// bound to `uio_pci_generic` with bus-mastering enabled) and create one I/O
    /// queue pair of `io_depth` entries.
    ///
    /// # Errors
    /// Fails if sysfs/pagemap access, mmap/mlock, or the controller handshake
    /// fails, or an admin command returns a non-success status.
    pub fn open(bdf: &str, io_depth: u32) -> io::Result<Self> {
        let bar_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(format!("/sys/bus/pci/devices/{bdf}/resource0"))?;
        let pagemap = File::open("/proc/self/pagemap")?;

        // SAFETY: the mmap'd BAR is the controller's register window; the DMA
        // region is freshly mapped, zeroed and pinned. All accesses below stay
        // within these regions, and the controller is single-owner (we just took
        // it from the kernel driver).
        unsafe {
            let bar = mmap(ptr::null_mut(), BAR_SIZE, 3, 1, bar_file.as_raw_fd(), 0) as usize;
            if bar as isize == -1 {
                return Err(io::Error::last_os_error());
            }
            let dma = mmap(ptr::null_mut(), DMA_SIZE, 3, 0x22, -1, 0) as usize;
            if dma as isize == -1 {
                return Err(io::Error::last_os_error());
            }
            ptr::write_bytes(dma as *mut u8, 0, DMA_SIZE);
            if mlock(dma as *const c_void, DMA_SIZE) != 0 {
                return Err(io::Error::last_os_error());
            }

            let cap = rd(bar as *const u64);
            let stride = 4usize << ((cap >> 32) & 0xf); // doorbell stride from CAP.DSTRD

            // reset, program the admin queue, enable, wait ready.
            w32(bar, 0x14, 0);
            while r32(bar, 0x1c) & 1 != 0 {}
            w32(bar, 0x24, ((ADMIN_DEPTH - 1) << 16) | (ADMIN_DEPTH - 1));
            w64(bar, 0x28, phys(&pagemap, dma + ASQ_O)?);
            w64(bar, 0x30, phys(&pagemap, dma + ACQ_O)?);
            w32(bar, 0x14, CC_ENABLE);
            while r32(bar, 0x1c) & 1 == 0 {}

            let admin_sq_db = 0x1000;
            let admin_cq_db = 0x1000 + stride;
            let mut sub = SubmissionRing::new(ADMIN_DEPTH).map_err(codec_err)?;
            let mut com = CompletionRing::new(ADMIN_DEPTH).map_err(codec_err)?;

            // admin submit-and-poll, built on the codec exactly like the I/O path
            let mut admin = |command: &CommandBuilder| -> io::Result<StatusField> {
                let mut sqe = [0u8; SQE_LEN];
                command.write(&mut sqe).map_err(codec_err)?;
                let slot = sub.slot() as usize;
                ptr::copy_nonoverlapping(
                    sqe.as_ptr(),
                    (dma + ASQ_O + slot * SQE_LEN) as *mut u8,
                    SQE_LEN,
                );
                let tail = sub.advance();
                w32(bar, admin_sq_db, u32::from(tail));
                for _ in 0..200_000_000u64 {
                    let mut cqe = [0u8; CQE_LEN];
                    ptr::copy_nonoverlapping(
                        (dma + ACQ_O + com.slot() as usize * CQE_LEN) as *const u8,
                        cqe.as_mut_ptr(),
                        CQE_LEN,
                    );
                    let entry = CompletionEntry::parse(&cqe).map_err(codec_err)?;
                    if com.is_ready(entry.phase()) {
                        let (_, status) = entry.command_id_and_status();
                        let head = com.advance();
                        w32(bar, admin_cq_db, u32::from(head));
                        return Ok(status);
                    }
                    core::hint::spin_loop();
                }
                Err(io::Error::other("admin completion timeout"))
            };

            // Create I/O CQ (qid 1, PC=1), then I/O SQ (qid 1, cqid 1, PC=1).
            let qsize = (io_depth - 1) << 16 | 1;
            let cq = CommandBuilder::new(0x05, 1)
                .data_ptrs(phys(&pagemap, dma + ICQ_O)?, 0)
                .command_dword(0, qsize)
                .command_dword(1, 1);
            let status = admin(&cq)?;
            if !status.is_success() {
                return Err(io::Error::other(format!(
                    "create io cq sc={:#x}",
                    status.bits()
                )));
            }
            let sq = CommandBuilder::new(0x01, 2)
                .data_ptrs(phys(&pagemap, dma + ISQ_O)?, 0)
                .command_dword(0, qsize)
                .command_dword(1, (1 << 16) | 1);
            let status = admin(&sq)?;
            if !status.is_success() {
                return Err(io::Error::other(format!(
                    "create io sq sc={:#x}",
                    status.bits()
                )));
            }

            Ok(Self {
                _bar_file: bar_file,
                pagemap,
                bar,
                dma,
                io_sq: dma + ISQ_O,
                io_cq: dma + ICQ_O,
                io_sq_db: 0x1000 + 2 * stride,
                io_cq_db: 0x1000 + 3 * stride,
                pool_next: Cell::new(POOL_O),
                depth: io_depth,
            })
        }
    }

    /// Bump-allocate a page-aligned DMA buffer from the pinned region and return
    /// its `(virtual, physical)` addresses — the physical goes in a command's
    /// data pointer, the virtual is how the caller fills/reads it.
    ///
    /// # Errors
    /// Fails if the pinned region is exhausted or pagemap lookup fails.
    pub fn alloc_dma(&self, size: usize) -> io::Result<(usize, u64)> {
        let off = self.pool_next.get();
        let rounded = (size + 0xfff) & !0xfff;
        if off + rounded > DMA_SIZE {
            return Err(io::Error::other("dma pool exhausted"));
        }
        self.pool_next.set(off + rounded);
        let vaddr = self.dma + off;
        Ok((vaddr, phys(&self.pagemap, vaddr)?))
    }

    /// The I/O queue depth this backend was created with.
    #[must_use]
    pub fn depth(&self) -> u32 {
        self.depth
    }
}

impl QueueBackend for UioNvme {
    fn write_submission(&self, slot: u16, entry: &[u8; SQE_LEN]) {
        // SAFETY: slot < depth (the engine never exceeds the ring), so the write
        // stays inside the I/O SQ region.
        unsafe {
            ptr::copy_nonoverlapping(
                entry.as_ptr(),
                (self.io_sq + slot as usize * SQE_LEN) as *mut u8,
                SQE_LEN,
            );
        }
    }

    fn ring_submit_doorbell(&self, tail: u16) {
        unsafe { w32(self.bar, self.io_sq_db, u32::from(tail)) }
    }

    fn read_completion(&self, slot: u16) -> [u8; CQE_LEN] {
        let mut cqe = [0u8; CQE_LEN];
        // SAFETY: slot < depth; the read stays inside the I/O CQ region.
        unsafe {
            ptr::copy_nonoverlapping(
                (self.io_cq + slot as usize * CQE_LEN) as *const u8,
                cqe.as_mut_ptr(),
                CQE_LEN,
            );
        }
        cqe
    }

    fn ring_complete_doorbell(&self, head: u16) {
        unsafe { w32(self.bar, self.io_cq_db, u32::from(head)) }
    }
}
