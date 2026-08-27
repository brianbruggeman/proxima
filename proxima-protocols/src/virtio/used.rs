use super::error::DecodeError;
use super::raw::{read_u16, read_u32, write_u32};

/// Size of one used-ring element (VIRTIO 1.2 spec §2.7.8): le32 id, le32 len.
pub const USED_ELEM_LEN: usize = 8;

/// One used-ring element: the descriptor chain's head index (`id`) and the
/// total bytes the device wrote into that chain's device-writable buffers
/// (`len`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UsedElem {
    pub id: u32,
    pub len: u32,
}

/// Write a used-ring element into `out` — the device's completion report for
/// one descriptor chain. Mirrors `super::nvme::completion::write_completion`.
pub fn write_used_elem(out: &mut [u8], elem: UsedElem) -> Result<usize, DecodeError> {
    if out.len() < USED_ELEM_LEN {
        return Err(DecodeError::Truncated {
            need: USED_ELEM_LEN,
            got: out.len(),
        });
    }
    write_u32(out, 0, elem.id);
    write_u32(out, 4, elem.len);
    Ok(USED_ELEM_LEN)
}

/// Borrowed view over a split-virtqueue used ring (VIRTIO 1.2 spec §2.7.8):
/// le16 flags, le16 idx, then `queue_size` [`UsedElem`]s. This is the
/// device's producer ring — the driver (this codec's caller, on the guest
/// side) only reads it, mirroring how NVMe's host only reads the
/// controller's completion ring. `avail_event` (`VIRTIO_F_EVENT_IDX`) is out
/// of scope for M6's virtio-console slice and is not parsed.
#[derive(Debug, Clone, Copy)]
pub struct UsedRing<'ring> {
    bytes: &'ring [u8],
    queue_size: u16,
}

impl<'ring> UsedRing<'ring> {
    pub fn parse(bytes: &'ring [u8], queue_size: u16) -> Result<Self, DecodeError> {
        let need = 4 + usize::from(queue_size) * USED_ELEM_LEN;
        if bytes.len() < need {
            return Err(DecodeError::Truncated {
                need,
                got: bytes.len(),
            });
        }
        Ok(Self { bytes, queue_size })
    }

    #[must_use]
    pub fn flags(&self) -> u16 {
        read_u16(self.bytes, 0)
    }

    #[must_use]
    pub fn idx(&self) -> u16 {
        read_u16(self.bytes, 2)
    }

    #[must_use]
    pub fn ring_entry(&self, position: u16) -> UsedElem {
        let slot = usize::from(position % self.queue_size);
        let offset = 4 + slot * USED_ELEM_LEN;
        UsedElem {
            id: read_u32(self.bytes, offset),
            len: read_u32(self.bytes, offset + 4),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn write_then_parse_round_trips_id_and_len() {
        let mut slot = [0u8; USED_ELEM_LEN];
        let written = write_used_elem(
            &mut slot,
            UsedElem {
                id: 0x0000_0007,
                len: 42,
            },
        )
        .expect("8-byte slot fits one used element");
        assert_eq!(written, USED_ELEM_LEN);
        assert_eq!(&slot[0..4], &7u32.to_le_bytes());
        assert_eq!(&slot[4..8], &42u32.to_le_bytes());
    }

    #[test]
    fn write_into_short_buffer_is_truncated() {
        let mut tiny = [0u8; USED_ELEM_LEN - 1];
        assert_eq!(
            write_used_elem(&mut tiny, UsedElem { id: 0, len: 0 }).unwrap_err(),
            DecodeError::Truncated {
                need: USED_ELEM_LEN,
                got: USED_ELEM_LEN - 1
            }
        );
    }

    // Worked example: queue_size = 4, one entry published — head 0 completed
    // with 8 bytes written. Byte layout per VIRTIO 1.2 spec §2.7.8.
    //   flags = 0                 -> LE16: 00 00
    //   idx   = 1                  -> LE16: 01 00
    //   ring[0] = { id: 0, len: 8 } -> LE32 00 00 00 00, LE32 08 00 00 00
    //   ring[1..4] = zeroed (unpublished)
    fn worked_example_used() -> [u8; 36] {
        let mut bytes = [0u8; 36];
        bytes[0..2].copy_from_slice(&0u16.to_le_bytes());
        bytes[2..4].copy_from_slice(&1u16.to_le_bytes());
        bytes[4..8].copy_from_slice(&0u32.to_le_bytes());
        bytes[8..12].copy_from_slice(&8u32.to_le_bytes());
        bytes
    }

    #[test]
    fn parse_reads_the_published_completion() {
        let bytes = worked_example_used();
        let ring = UsedRing::parse(&bytes, 4).expect("well-formed used ring");
        assert_eq!(ring.flags(), 0);
        assert_eq!(ring.idx(), 1);
        assert_eq!(ring.ring_entry(0), UsedElem { id: 0, len: 8 });
    }

    #[test]
    fn short_buffer_is_truncated() {
        let bytes = [0u8; 35];
        assert_eq!(
            UsedRing::parse(&bytes, 4).unwrap_err(),
            DecodeError::Truncated { need: 36, got: 35 }
        );
    }
}
