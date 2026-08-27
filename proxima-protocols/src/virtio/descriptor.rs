use super::error::DecodeError;
use super::raw::{read_u16, read_u32, read_u64};

/// Fixed size of one split-virtqueue descriptor table entry (VIRTIO 1.2 spec
/// §2.7.5): le64 addr, le32 len, le16 flags, le16 next.
pub const DESC_LEN: usize = 16;

const FLAG_NEXT: u16 = 1;
const FLAG_WRITE: u16 = 2;
const FLAG_INDIRECT: u16 = 4;

/// Decoded descriptor flags (spec §2.7.5, `VIRTQ_DESC_F_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorFlags(u16);

impl DescriptorFlags {
    /// This descriptor continues into `next` — the chain does not end here.
    #[must_use]
    pub fn has_next(self) -> bool {
        self.0 & FLAG_NEXT != 0
    }

    /// Buffer is device-writable (the device writes into it); otherwise it is
    /// device-readable (the driver wrote it before publishing the chain).
    #[must_use]
    pub fn device_writable(self) -> bool {
        self.0 & FLAG_WRITE != 0
    }

    /// `addr`/`len` point at a table of further descriptors rather than a
    /// data buffer. M6's virtio-console slice never publishes an indirect
    /// table, so the chain walker below treats this as an opaque flag it
    /// surfaces but does not follow.
    #[must_use]
    pub fn is_indirect(self) -> bool {
        self.0 & FLAG_INDIRECT != 0
    }

    #[must_use]
    pub fn bits(self) -> u16 {
        self.0
    }
}

/// Borrowed view over one 16-byte descriptor table entry. Points into the
/// caller's queue memory — no copy, no ownership.
#[derive(Debug, Clone, Copy)]
pub struct Descriptor<'table> {
    bytes: &'table [u8],
}

impl<'table> Descriptor<'table> {
    fn parse_at(table: &'table [u8], index: u16) -> Result<Self, DecodeError> {
        let offset = usize::from(index) * DESC_LEN;
        let end = offset + DESC_LEN;
        if table.len() < end {
            return Err(DecodeError::Truncated {
                need: end,
                got: table.len(),
            });
        }
        Ok(Self {
            bytes: &table[offset..end],
        })
    }

    /// Guest-physical address of the buffer this descriptor names.
    #[must_use]
    pub fn addr(&self) -> u64 {
        read_u64(self.bytes, 0)
    }

    /// Length in bytes of the buffer this descriptor names.
    #[must_use]
    pub fn buffer_len(&self) -> u32 {
        read_u32(self.bytes, 8)
    }

    #[must_use]
    pub fn flags(&self) -> DescriptorFlags {
        DescriptorFlags(read_u16(self.bytes, 12))
    }

    /// Index of the next descriptor in the chain. Only meaningful when
    /// [`DescriptorFlags::has_next`] is set.
    #[must_use]
    pub fn next(&self) -> u16 {
        read_u16(self.bytes, 14)
    }
}

/// Chain-walk state, the house enum-FSM shape (guiding-principles §11):
/// `Walking` carries exactly the data legal mid-walk (the next index to
/// visit and how many steps have been taken); `Done` carries nothing. No
/// runtime "are we finished" boolean check — the match on `Iterator::next`
/// below does that work at compile time.
#[derive(Debug, Clone, Copy)]
enum ChainState {
    Walking { index: u16, steps_taken: u16 },
    Done,
}

/// Sans-IO walker over a descriptor chain rooted at an avail-ring head
/// index. Fixed-shape, allocates nothing: all state is two `u16`s. Bounded
/// by `queue_size` steps so a corrupt or cyclic `next` chain from an
/// adversarial guest cannot loop the host forever — VIRTIO's descriptor
/// table has no chain-length field, so the queue size is the only available
/// bound (spec §2.7.5).
#[derive(Debug, Clone, Copy)]
pub struct DescriptorChain<'table> {
    table: &'table [u8],
    queue_size: u16,
    state: ChainState,
}

impl<'table> DescriptorChain<'table> {
    #[must_use]
    pub fn new(table: &'table [u8], queue_size: u16, head: u16) -> Self {
        Self {
            table,
            queue_size,
            state: ChainState::Walking {
                index: head,
                steps_taken: 0,
            },
        }
    }
}

impl<'table> Iterator for DescriptorChain<'table> {
    type Item = Result<Descriptor<'table>, DecodeError>;

    fn next(&mut self) -> Option<Self::Item> {
        let ChainState::Walking { index, steps_taken } = self.state else {
            return None;
        };
        if steps_taken >= self.queue_size {
            self.state = ChainState::Done;
            return Some(Err(DecodeError::ChainTooLong {
                limit: self.queue_size,
            }));
        }
        let descriptor = match Descriptor::parse_at(self.table, index) {
            Ok(descriptor) => descriptor,
            Err(error) => {
                self.state = ChainState::Done;
                return Some(Err(error));
            }
        };
        self.state = if descriptor.flags().has_next() {
            ChainState::Walking {
                index: descriptor.next(),
                steps_taken: steps_taken + 1,
            }
        } else {
            ChainState::Done
        };
        Some(Ok(descriptor))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    // Worked example (principle 9 / /algorithm-development): a 4-entry
    // descriptor table for a virtio-console `write` — hand-derived byte
    // layout per VIRTIO 1.2 spec §2.7.5, not captured from a running QEMU
    // session (none was available in this worktree), walked bit-exact.
    //
    // Descriptor 0 (device-readable, "hello", head of the chain):
    //   addr = 0x0000_0000_0000_1000  -> LE64: 00 10 00 00 00 00 00 00
    //   len  = 5                      -> LE32: 05 00 00 00
    //   flags = NEXT (0x0001)         -> LE16: 01 00
    //   next  = 2                     -> LE16: 02 00
    //
    // Descriptor 2 (device-writable reply buffer, chain terminator):
    //   addr = 0x0000_0000_0000_2000  -> LE64: 00 20 00 00 00 00 00 00
    //   len  = 8                      -> LE32: 08 00 00 00
    //   flags = WRITE (0x0002)        -> LE16: 02 00
    //   next  = 0 (ignored, NEXT unset) -> LE16: 00 00
    //
    // Descriptors 1 and 3 are unused (zeroed) slots in the free list.
    fn worked_example_table() -> [u8; 4 * DESC_LEN] {
        let mut table = [0u8; 4 * DESC_LEN];
        table[0..16].copy_from_slice(&[
            0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // addr = 0x1000
            0x05, 0x00, 0x00, 0x00, // len = 5
            0x01, 0x00, // flags = NEXT
            0x02, 0x00, // next = 2
        ]);
        table[32..48].copy_from_slice(&[
            0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // addr = 0x2000
            0x08, 0x00, 0x00, 0x00, // len = 8
            0x02, 0x00, // flags = WRITE
            0x00, 0x00, // next = 0 (ignored)
        ]);
        table
    }

    #[test]
    fn walk_follows_next_and_stops_at_the_write_only_terminator() {
        let table = worked_example_table();
        let mut chain = DescriptorChain::new(&table, 4, 0);

        let first = chain.next().expect("link one").expect("well-formed");
        assert_eq!(first.addr(), 0x1000);
        assert_eq!(first.buffer_len(), 5);
        assert!(first.flags().has_next());
        assert!(!first.flags().device_writable());
        assert_eq!(first.next(), 2);

        let second = chain.next().expect("link two").expect("well-formed");
        assert_eq!(second.addr(), 0x2000);
        assert_eq!(second.buffer_len(), 8);
        assert!(!second.flags().has_next());
        assert!(second.flags().device_writable());

        assert!(chain.next().is_none(), "chain has exactly two links");
    }

    #[test]
    fn single_descriptor_chain_has_no_next_flag() {
        let mut table = [0u8; DESC_LEN];
        table.copy_from_slice(&[
            0xef, 0xbe, 0xad, 0xde, 0x00, 0x00, 0x00, 0x00, // addr
            0x0a, 0x00, 0x00, 0x00, // len = 10
            0x00, 0x00, // flags = 0 (device-readable, no next)
            0x00, 0x00, // next (unused)
        ]);
        let mut chain = DescriptorChain::new(&table, 1, 0);
        let only = chain.next().expect("one link").expect("well-formed");
        assert_eq!(only.addr(), 0xdead_beef);
        assert!(!only.flags().has_next());
        assert!(chain.next().is_none(), "chain terminates after one link");
    }

    #[test]
    fn indirect_flag_is_surfaced_not_followed() {
        let mut table = [0u8; DESC_LEN];
        table[12] = FLAG_INDIRECT as u8;
        let mut chain = DescriptorChain::new(&table, 1, 0);
        let descriptor = chain.next().expect("one link").expect("well-formed");
        assert!(descriptor.flags().is_indirect());
        assert!(chain.next().is_none());
    }

    #[test]
    fn chain_head_past_the_table_end_is_truncated() {
        let table = [0u8; DESC_LEN];
        let mut chain = DescriptorChain::new(&table, 1, 5);
        assert_eq!(
            chain.next().expect("one attempt").unwrap_err(),
            DecodeError::Truncated {
                need: 6 * DESC_LEN,
                got: DESC_LEN
            }
        );
    }

    #[test]
    fn cyclic_next_is_bounded_by_queue_size_not_looped_forever() {
        // depth 2: descriptor 0 -> 1 -> 0 -> 1 ... an adversarial cycle.
        let mut table = [0u8; 2 * DESC_LEN];
        table[12] = FLAG_NEXT as u8; // desc 0: NEXT
        table[14] = 1; // desc 0 next = 1 (LE16 low byte)
        table[12 + DESC_LEN] = FLAG_NEXT as u8; // desc 1: NEXT
        table[14 + DESC_LEN] = 0; // desc 1 next = 0 (LE16 low byte)

        let mut chain = DescriptorChain::new(&table, 2, 0);

        // two legal links, then the bound trips on the third attempt.
        assert!(chain.next().expect("link one").is_ok());
        assert!(chain.next().expect("link two").is_ok());
        assert_eq!(
            chain.next().expect("bound trips here").unwrap_err(),
            DecodeError::ChainTooLong { limit: 2 }
        );
        assert!(
            chain.next().is_none(),
            "walker stops emitting after the bound trips"
        );
    }
}
