//! Host-side virtio-console transport: owns one
//! [`proxima_protocols::virtio::MmioDevice`] (the sans-IO register-block
//! codec) plus the per-queue ring cursors, and reads real guest memory to
//! walk a descriptor chain once a `QueueNotify` effect names it — the "one
//! layer up" the `virtio` module's own doc names (`proxima-protocols/src/
//! virtio/mod.rs`): MMIO register decode plus guest-physical-to-host-
//! pointer translation live here, never inside the sans-IO codec itself.
//!
//! Scope for this slice: one queue (index 0), used as the transmit queue —
//! a real virtio-console negotiates two (receiveq/transmitq per spec
//! §5.3.2), but M6's exit criterion is proving one byte crosses the ring
//! through a real VM exit, not a spec-complete console device.

use proxima_protocols::virtio::{
    AvailRing, DecodeError, DescriptorChain, MmioAccess, MmioDevice, MmioEffect, MmioError,
    RingCursor, UsedElem, write_used_elem,
};

/// The one queue this transport drives; `MmioDevice` itself is generic over
/// queue count (`super::mmio`'s worked example uses 2, matching a spec-
/// faithful console), kept here too so a later slice can add the receive
/// queue without changing this constant's meaning.
const TX_QUEUE: u16 = 0;
const QUEUE_SIZE: u16 = 4;

/// Why [`ConsoleTransport::drain_tx`] could not complete.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainError {
    /// The queue named by a `QueueNotify` is not the transport's TX queue.
    NotTransmitQueue { queue: u16 },
    /// The queue's descriptor/driver/device addresses are not all
    /// programmed yet (a notify arrived before queue setup finished).
    QueueNotConfigured,
    /// A ring or descriptor-chain byte range fell outside `guest_memory`.
    OutOfBounds { need: usize, have: usize },
    /// The ring or chain bytes themselves failed to decode.
    Decode(DecodeError),
}

impl core::fmt::Display for DrainError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotTransmitQueue { queue } => {
                write!(formatter, "queue {queue} is not the transmit queue")
            }
            Self::QueueNotConfigured => write!(formatter, "queue addresses not yet programmed"),
            Self::OutOfBounds { need, have } => write!(
                formatter,
                "guest memory access needs {need} bytes, only {have} available"
            ),
            Self::Decode(inner) => write!(formatter, "{inner}"),
        }
    }
}

impl std::error::Error for DrainError {}

/// Owns the mmio register-block FSM and the TX queue's avail/used cursors.
/// Fixed-shape aside from the `Vec` a std-tier host process allocates for
/// the drained bytes — this crate's own tier (`tools/proxima-vm` is a
/// std-only host binary, never the tier-3 leaf `proxima-protocols/src/
/// virtio` itself is held to).
///
/// `Clone` is the M7 payoff M6's own module doc named: every field here is
/// plain data (`MmioDevice`/`RingCursor` are themselves `Clone`, no interior
/// mutability, no handle), so a device-state snapshot is `.clone()` and a
/// restore is substituting the clone for a freshly constructed device —
/// no bespoke (de)serialization path, because the device was already a
/// state machine (`tools/proxima-vm/ROADMAP.md`'s M7 section).
#[derive(Debug, Clone)]
pub struct ConsoleTransport {
    device: MmioDevice<2>,
    avail_cursor: RingCursor,
    used_cursor: RingCursor,
}

// `QUEUE_SIZE` is a fixed compile-time literal (4), proven a nonzero power
// of two right here rather than only by convention — `RingCursor::new`'s
// `Result` exists for a caller-supplied runtime queue size, which this
// module never has.
const _: () = assert!(QUEUE_SIZE != 0 && QUEUE_SIZE.is_power_of_two());

impl ConsoleTransport {
    #[must_use]
    #[allow(
        clippy::expect_used,
        reason = "QUEUE_SIZE's power-of-two invariant is checked at compile time above; RingCursor::new cannot fail for this literal"
    )]
    pub fn new(offered_features: u64) -> Self {
        Self {
            device: MmioDevice::new(
                proxima_protocols::virtio::DEVICE_ID_CONSOLE,
                QUEUE_SIZE,
                offered_features,
            ),
            avail_cursor: RingCursor::new(QUEUE_SIZE)
                .expect("QUEUE_SIZE is a fixed power of two, proven above"),
            used_cursor: RingCursor::new(QUEUE_SIZE)
                .expect("QUEUE_SIZE is a fixed power of two, proven above"),
        }
    }

    /// Apply one raw register access recovered from a trapped guest
    /// load/store — delegates straight to [`MmioDevice::apply`], the sans-IO
    /// codec this transport composes rather than replaces.
    pub fn apply(&mut self, access: MmioAccess) -> Result<MmioEffect, MmioError> {
        self.device.apply(access)
    }

    /// Walks every avail-ring entry published on `queue` since this
    /// transport last drained it, copying each chain's device-readable
    /// bytes out of `guest_memory` and publishing one used-ring completion
    /// per chain (`len = 0`: this queue is transmit-only, so the device
    /// never writes into a device-writable buffer — spec §2.7.8 defines
    /// `len` as bytes the device wrote, which is legitimately zero here).
    /// Returns the concatenation of every drained chain's bytes, in
    /// publish order.
    pub fn drain_tx(&mut self, queue: u16, guest_memory: &mut [u8]) -> Result<Vec<u8>, DrainError> {
        if queue != TX_QUEUE {
            return Err(DrainError::NotTransmitQueue { queue });
        }
        let descriptor_address = self
            .device
            .queue_descriptor_address(queue)
            .filter(|address| *address != 0)
            .ok_or(DrainError::QueueNotConfigured)?;
        let driver_address = self
            .device
            .queue_driver_address(queue)
            .filter(|address| *address != 0)
            .ok_or(DrainError::QueueNotConfigured)?;
        let device_address = self
            .device
            .queue_device_address(queue)
            .filter(|address| *address != 0)
            .ok_or(DrainError::QueueNotConfigured)?;

        let mut emitted = Vec::new();
        loop {
            let avail_bytes = slice_at(
                guest_memory,
                driver_address,
                4 + usize::from(QUEUE_SIZE) * 2,
            )?;
            let avail = AvailRing::parse(avail_bytes, QUEUE_SIZE).map_err(DrainError::Decode)?;
            let pending = self.avail_cursor.pending(avail.idx());
            if pending == 0 {
                break;
            }
            let head = avail.ring_entry(self.avail_cursor.position());
            self.avail_cursor.advance();

            let table_bytes = slice_at(
                guest_memory,
                descriptor_address,
                usize::from(QUEUE_SIZE) * proxima_protocols::virtio::DESC_LEN,
            )?;
            let mut chain_bytes = Vec::new();
            let mut addresses = Vec::new();
            for descriptor in DescriptorChain::new(table_bytes, QUEUE_SIZE, head) {
                let descriptor = descriptor.map_err(DrainError::Decode)?;
                addresses.push((descriptor.addr(), descriptor.buffer_len()));
            }
            for (addr, len) in addresses {
                let bytes = slice_at(guest_memory, addr, len as usize)?;
                chain_bytes.extend_from_slice(bytes);
            }
            emitted.extend_from_slice(&chain_bytes);

            let used_offset = 4 + usize::from(self.used_cursor.position() % QUEUE_SIZE)
                * proxima_protocols::virtio::USED_ELEM_LEN;
            let used_slot = mut_slice_at(
                guest_memory,
                device_address + used_offset as u64,
                proxima_protocols::virtio::USED_ELEM_LEN,
            )?;
            write_used_elem(
                used_slot,
                UsedElem {
                    id: u32::from(head),
                    len: 0,
                },
            )
            .map_err(DrainError::Decode)?;
            let new_idx = self.used_cursor.advance();
            let idx_slot = mut_slice_at(guest_memory, device_address + 2, 2)?;
            idx_slot.copy_from_slice(&new_idx.to_le_bytes());
        }
        Ok(emitted)
    }
}

fn slice_at(guest_memory: &[u8], address: u64, len: usize) -> Result<&[u8], DrainError> {
    let start = usize::try_from(address).map_err(|_| DrainError::OutOfBounds {
        need: len,
        have: guest_memory.len(),
    })?;
    let end = start.checked_add(len).ok_or(DrainError::OutOfBounds {
        need: len,
        have: guest_memory.len(),
    })?;
    guest_memory.get(start..end).ok_or(DrainError::OutOfBounds {
        need: end,
        have: guest_memory.len(),
    })
}

fn mut_slice_at(
    guest_memory: &mut [u8],
    address: u64,
    len: usize,
) -> Result<&mut [u8], DrainError> {
    let total = guest_memory.len();
    let start = usize::try_from(address).map_err(|_| DrainError::OutOfBounds {
        need: len,
        have: total,
    })?;
    let end = start.checked_add(len).ok_or(DrainError::OutOfBounds {
        need: len,
        have: total,
    })?;
    guest_memory
        .get_mut(start..end)
        .ok_or(DrainError::OutOfBounds {
            need: end,
            have: total,
        })
}
