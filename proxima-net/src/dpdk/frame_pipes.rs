//! The dpdk port as three frame pipes for [`crate::frame_stack::Driver`]: an rx
//! source, a tx sink and a mint over the mempool. The frame is an owned mbuf, so
//! the driver parses and rewrites it in place and the sink hands it straight to
//! `tx_burst` with no copy.

use super::port::{self, Port};
use super::{DpdkError, Eal, Mempool, RawMbuf};
use core::cell::Cell;
use core::mem::ManuallyDrop;
use proxima_primitives::pipe::Pipe;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DpdkFrameError {
    #[error("mempool exhausted allocating a frame of {length} bytes")]
    PoolExhausted { length: u16 },

    #[error("mbuf has no room for {length} bytes")]
    NoRoom { length: u16 },

    #[error("tx ring rejected the frame")]
    TxRejected,
}

pub struct Hardware {
    pool: Mempool,
    port: Port,
    _eal: Eal,
}

// SAFETY: the dpdk resources are reached only from the single prime core that
// owns the per-core rx/tx queues, under the driver's async lock.
unsafe impl Send for Hardware {}
unsafe impl Sync for Hardware {}

impl Hardware {
    pub(crate) fn bring_up(
        pool_name: &str,
        iface: &str,
        pmd_dir: &str,
    ) -> Result<(Arc<Self>, [u8; 6]), DpdkError> {
        let vdev = format!("--vdev=net_tap0,iface={iface}");
        let eal = Eal::init(&["proxima-net-dpdk", "-l", "0", "--no-pci", "-d", pmd_dir, &vdev])?;
        let pool = Mempool::create(pool_name, 8192, -1)?;
        let port = Port::init(0, &pool)?;
        let mac = port.mac()?;
        Ok((Arc::new(Self { pool, port, _eal: eal }), mac))
    }

    pub(crate) fn pipes(hardware: Arc<Self>) -> (DpdkRx, DpdkTx, DpdkMint) {
        (DpdkRx::new(hardware.clone()), DpdkTx(hardware.clone()), DpdkMint(hardware))
    }
}

/// An owned dpdk packet buffer; dropping it returns it to the pool.
pub struct DpdkFrame {
    raw: RawMbuf,
}

impl DpdkFrame {
    fn into_raw(self) -> RawMbuf {
        ManuallyDrop::new(self).raw
    }
}

impl AsRef<[u8]> for DpdkFrame {
    fn as_ref(&self) -> &[u8] {
        unsafe { port::frame_bytes(self.raw) }
    }
}

impl AsMut<[u8]> for DpdkFrame {
    fn as_mut(&mut self) -> &mut [u8] {
        unsafe { port::frame_bytes_mut(self.raw) }
    }
}

impl Drop for DpdkFrame {
    fn drop(&mut self) {
        unsafe { port::free(self.raw) };
    }
}

const BURST: usize = 32;

/// Pulls up to `BURST` mbufs per `rx_burst` and hands them out one frame per call.
pub struct DpdkRx {
    hardware: Arc<Hardware>,
    batch: [Cell<RawMbuf>; BURST],
    next: Cell<usize>,
    filled: Cell<usize>,
}

pub struct DpdkTx(Arc<Hardware>);
pub struct DpdkMint(Arc<Hardware>);

impl DpdkRx {
    fn new(hardware: Arc<Hardware>) -> Self {
        Self {
            hardware,
            batch: core::array::from_fn(|_| Cell::new(core::ptr::null_mut())),
            next: Cell::new(0),
            filled: Cell::new(0),
        }
    }

    fn refill(&self) {
        let mut burst: [RawMbuf; BURST] = [core::ptr::null_mut(); BURST];
        let received = usize::from(self.hardware.port.rx_burst(&mut burst));
        self.batch.iter().zip(burst).for_each(|(slot, mbuf)| slot.set(mbuf));
        self.next.set(0);
        self.filled.set(received);
    }

    fn take(&self) -> Option<DpdkFrame> {
        if self.next.get() == self.filled.get() {
            self.refill();
        }
        let index = self.next.get();
        (index < self.filled.get()).then(|| {
            self.next.set(index + 1);
            DpdkFrame { raw: self.batch[index].get() }
        })
    }
}

impl Drop for DpdkRx {
    fn drop(&mut self) {
        for slot in &self.batch[self.next.get()..self.filled.get()] {
            // SAFETY: slots in next..filled hold mbufs rx_burst handed to this
            // pipe that no DpdkFrame owns yet, so each is live and freed once.
            unsafe { port::free(slot.get()) };
        }
    }
}

// SAFETY: the batch cells are touched only from the single prime core that owns
// the rx queue, under the driver's async lock, so no two threads race on them.
unsafe impl Sync for DpdkRx {}

// SAFETY: the batched mbufs are owned by this pipe alone, so moving the pipe to
// another thread moves that ownership with it, exactly as `Hardware` does.
unsafe impl Send for DpdkRx {}

impl Pipe for DpdkRx {
    type In = ();
    type Out = Option<DpdkFrame>;
    type Err = DpdkFrameError;

    fn call(&self, _input: ()) -> impl Future<Output = Result<Option<DpdkFrame>, DpdkFrameError>> {
        async move { Ok(self.take()) }
    }
}

impl Pipe for DpdkTx {
    type In = DpdkFrame;
    type Out = ();
    type Err = DpdkFrameError;

    fn call(&self, frame: DpdkFrame) -> impl Future<Output = Result<(), DpdkFrameError>> {
        async move {
            let mut one = [frame.into_raw()];
            if self.0.port.tx_burst(&mut one) == 0 {
                unsafe { port::free(one[0]) };
                return Err(DpdkFrameError::TxRejected);
            }
            Ok(())
        }
    }
}

impl Pipe for DpdkMint {
    type In = u16;
    type Out = DpdkFrame;
    type Err = DpdkFrameError;

    fn call(&self, length: u16) -> impl Future<Output = Result<DpdkFrame, DpdkFrameError>> {
        async move {
            let raw = self.0.pool.alloc();
            if raw.is_null() {
                return Err(DpdkFrameError::PoolExhausted { length });
            }
            let frame = DpdkFrame { raw };
            match unsafe { port::frame_append(frame.raw, length) } {
                Some(_) => Ok(frame),
                None => Err(DpdkFrameError::NoRoom { length }),
            }
        }
    }
}
