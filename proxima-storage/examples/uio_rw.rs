//! Drive a real (emulated) NVMe controller through the full proxima-storage nvme
//! stack: the UIO backend brings the controller up and creates an I/O queue
//! pair, then the `QueuePair` engine (a `Pipe`) submits an NVM Write and an NVM
//! Read — the C1 codec + C2 engine + C3 cooperative poll, end to end on
//! hardware.
//!
//! Run inside the guest as root, with the device bound to uio_pci_generic and
//! bus-mastering on:  `sudo ./uio_rw 0000:00:02.0`

#![allow(clippy::expect_used, clippy::unwrap_used)]
use std::future::Future;

use proxima_primitives::pipe::Pipe;
use proxima_protocols::nvme::CommandBuilder;
use proxima_storage::nvme::QueuePair;
use proxima_storage::nvme::uio::UioNvme;

const OPC_WRITE: u8 = 0x01;
const OPC_READ: u8 = 0x02;
const BLOCK: usize = 512;
const DEPTH: u32 = 32;

fn block_on<Fut: Future>(future: Fut) -> Fut::Output {
    let mut pinned = core::pin::pin!(future);
    let mut cx = core::task::Context::from_waker(core::task::Waker::noop());
    loop {
        if let core::task::Poll::Ready(out) = pinned.as_mut().poll(&mut cx) {
            return out;
        }
    }
}

// an NVM read/write to LBA 0, one block: NSID 1, data at `prp`, CDW10/11 = SLBA,
// CDW12 = NLB (0-based).
fn rw(opcode: u8, cid: u16, prp: u64) -> CommandBuilder {
    CommandBuilder::new(opcode, cid)
        .namespace_id(1)
        .data_ptrs(prp, 0)
        .command_dword(0, 0)
        .command_dword(1, 0)
        .command_dword(2, 0)
}

fn main() {
    let bdf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "0000:00:02.0".into());

    let backend = UioNvme::open(&bdf, DEPTH).expect("bring up the controller");
    let (write_va, write_pa) = backend.alloc_dma(BLOCK).expect("alloc write buffer");
    let (read_va, read_pa) = backend.alloc_dma(BLOCK).expect("alloc read buffer");

    // fill the write buffer with a recognisable pattern.
    for index in 0..BLOCK {
        unsafe { core::ptr::write_volatile((write_va + index) as *mut u8, (index as u8) ^ 0x5A) };
    }

    let queue = QueuePair::new(backend, DEPTH).expect("queue pair");

    let written = block_on(Pipe::call(&queue, rw(OPC_WRITE, 0x10, write_pa))).expect("write call");
    assert!(
        written.status.is_success(),
        "write status {:#x}",
        written.status.bits()
    );
    println!("WRITE ok via QueuePair::call (cid {})", written.command_id);

    let read = block_on(Pipe::call(&queue, rw(OPC_READ, 0x11, read_pa))).expect("read call");
    assert!(
        read.status.is_success(),
        "read status {:#x}",
        read.status.bits()
    );
    println!("READ  ok via QueuePair::call (cid {})", read.command_id);

    let ok = (0..BLOCK).all(|index| unsafe {
        core::ptr::read_volatile((write_va + index) as *const u8)
            == core::ptr::read_volatile((read_va + index) as *const u8)
    });
    println!(
        "{}",
        if ok {
            "ENGINE READ/WRITE ROUND-TRIP VERIFIED OK (512 bytes match)"
        } else {
            "VERIFY FAILED"
        }
    );
    assert!(ok, "data mismatch");
}
