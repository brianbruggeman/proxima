//! Driver-side split-virtqueue codec (VIRTIO 1.2 spec §2.7): the mirror of
//! the device side in `super::avail`, `super::descriptor` and `super::used`.
//! The driver owns the descriptor table and the avail ring (it writes them),
//! and only reads the used ring. Sans-IO and tier-3: all queue memory is
//! caller-owned byte slices, the queue size is a const generic, and the only
//! state is four integers. The layouts and flag constants are the device
//! side's own (`super::descriptor::DESC_F_*`, `super::used::USED_ELEM_LEN`,
//! [`super::avail::RingCursor`] for the used cursor) — nothing is restated.
//!
//! Unused descriptors form a free list threaded through each descriptor's
//! `next` field, so no side table is needed and a chain's links are simply
//! consecutive free-list entries.

use core::sync::atomic::{Ordering, fence};

use super::avail::RingCursor;
use super::descriptor::{DESC_F_NEXT, DESC_F_WRITE, DESC_LEN, DescriptorChain};
use super::error::DecodeError;
use super::raw::{read_u16, read_u64, write_u16, write_u32, write_u64};
use super::used::{UsedElem, UsedRing};

const AVAIL_HEADER_LEN: usize = 4;
const AVAIL_IDX_OFFSET: usize = 2;
const DESC_ADDR_OFFSET: usize = 0;
const DESC_LEN_OFFSET: usize = 8;
const DESC_FLAGS_OFFSET: usize = 12;
const DESC_NEXT_OFFSET: usize = 14;

/// One `(guest-physical address, length)` buffer handed to [`DriverQueue::post`].
pub type Buffer = (u64, u32);

#[derive(Debug, Clone, Copy)]
pub struct DriverQueue<const SIZE: usize> {
    free_head: u16,
    free_count: u16,
    avail_idx: u16,
    used_cursor: RingCursor,
}

impl<const SIZE: usize> DriverQueue<SIZE> {
    const QUEUE_SIZE: u16 = {
        assert!(SIZE.is_power_of_two() && SIZE <= 1 << 15);
        SIZE as u16
    };

    /// Threads every descriptor of `table` onto the free list. `table` must be
    /// the queue's descriptor table, `SIZE * 16` bytes.
    pub fn new(table: &mut [u8]) -> Result<Self, DecodeError> {
        let need = SIZE * DESC_LEN;
        if table.len() < need {
            return Err(DecodeError::Truncated {
                need,
                got: table.len(),
            });
        }
        for index in 0..Self::QUEUE_SIZE {
            write_u16(table, desc_offset(index) + DESC_NEXT_OFFSET, index + 1);
        }
        Ok(Self {
            free_head: 0,
            free_count: Self::QUEUE_SIZE,
            avail_idx: 0,
            used_cursor: RingCursor::new(Self::QUEUE_SIZE)?,
        })
    }

    #[must_use]
    pub fn free_descriptors(&self) -> u16 {
        self.free_count
    }

    #[must_use]
    pub fn avail_idx(&self) -> u16 {
        self.avail_idx
    }

    /// Publish one chain: every `readable` buffer, then every `writable`
    /// buffer (spec §2.7.4.2 requires device-readable before device-writable).
    /// Returns the chain's head descriptor index. The ring entry is written,
    /// then a release fence, then the avail `idx` — the order a device that
    /// reads `idx` first relies on.
    pub fn post(
        &mut self,
        table: &mut [u8],
        avail: &mut [u8],
        readable: &[Buffer],
        writable: &[Buffer],
    ) -> Result<u16, DecodeError> {        let needed = readable.len() + writable.len();
        self.check_postable(table, avail, needed)?;
        let head = self.free_head;
        let links = readable
            .iter()
            .map(|buffer| (buffer, 0))
            .chain(writable.iter().map(|buffer| (buffer, DESC_F_WRITE)));
        let mut index = head;
        for (position, ((addr, len), write_flag)) in links.enumerate() {
            let following = read_u16(table, desc_offset(index) + DESC_NEXT_OFFSET);
            let is_last = position + 1 == needed;
            let flags = write_flag | if is_last { 0 } else { DESC_F_NEXT };
            let next = if is_last { 0 } else { following };
            write_descriptor(table, index, *addr, *len, flags, next);
            index = following;
        }
        self.free_head = index;
        self.free_count -= needed as u16;
        self.publish(avail, head);
        Ok(head)
    }

    /// Take the next completion the device has published in `used`, if any,
    /// and return its descriptors to the free list. `used` must be a snapshot
    /// of the used ring the caller read with volatile loads — the device
    /// mutates that memory behind the compiler's back. Returns the completion
    /// and the address of the chain's first buffer, which the caller needs to
    /// recycle it.
    pub fn reap(
        &mut self,
        table: &mut [u8],
        used: &[u8],
    ) -> Result<Option<(UsedElem, u64)>, DecodeError> {
        let ring = UsedRing::parse(used, Self::QUEUE_SIZE)?;
        if self.used_cursor.pending(ring.idx()) == 0 {
            return Ok(None);
        }
        let elem = ring.ring_entry(self.used_cursor.position());
        let head = self.outstanding_head(table, elem.id)?;
        let first_addr = read_u64(table, desc_offset(head) + DESC_ADDR_OFFSET);
        self.free_chain(table, head);
        self.used_cursor.advance();
        Ok(Some((elem, first_addr)))
    }

    fn check_postable(
        &self,
        table: &[u8],
        avail: &[u8],
        needed: usize,
    ) -> Result<(), DecodeError> {
        let avail_need = AVAIL_HEADER_LEN + SIZE * 2;
        if avail.len() < avail_need {
            return Err(DecodeError::Truncated {
                need: avail_need,
                got: avail.len(),
            });
        }
        if table.len() < SIZE * DESC_LEN {
            return Err(DecodeError::Truncated {
                need: SIZE * DESC_LEN,
                got: table.len(),
            });
        }
        if needed == 0 {
            return Err(DecodeError::EmptyChain);
        }
        if needed > usize::from(self.free_count) {
            return Err(DecodeError::QueueFull {
                needed,
                free: self.free_count,
            });
        }
        Ok(())
    }

    fn publish(&mut self, avail: &mut [u8], head: u16) {
        let slot = usize::from(self.avail_idx % Self::QUEUE_SIZE);
        write_u16(avail, AVAIL_HEADER_LEN + slot * 2, head);
        fence(Ordering::Release);
        self.avail_idx = self.avail_idx.wrapping_add(1);
        write_u16(avail, AVAIL_IDX_OFFSET, self.avail_idx);
    }

    fn outstanding_head(&self, table: &[u8], id: u32) -> Result<u16, DecodeError> {
        let bad = DecodeError::BadUsedId {
            id,
            queue_size: Self::QUEUE_SIZE,
        };
        let head = u16::try_from(id)
            .ok()
            .filter(|head| *head < Self::QUEUE_SIZE)
            .ok_or(bad)?;
        let mut chain_len = 0usize;
        for link in DescriptorChain::new(table, Self::QUEUE_SIZE, head) {
            link?;
            chain_len += 1;
        }
        if chain_len + usize::from(self.free_count) > SIZE {
            return Err(bad);
        }
        Ok(head)
    }

    fn free_chain(&mut self, table: &mut [u8], head: u16) {
        let mut index = head;
        loop {
            let flags = read_u16(table, desc_offset(index) + DESC_FLAGS_OFFSET);
            let next = read_u16(table, desc_offset(index) + DESC_NEXT_OFFSET);
            write_u16(table, desc_offset(index) + DESC_NEXT_OFFSET, self.free_head);
            self.free_head = index;
            self.free_count += 1;
            if flags & DESC_F_NEXT == 0 {
                return;
            }
            index = next;
        }
    }
}

fn desc_offset(index: u16) -> usize {
    usize::from(index) * DESC_LEN
}

fn write_descriptor(table: &mut [u8], index: u16, addr: u64, len: u32, flags: u16, next: u16) {
    let base = desc_offset(index);
    write_u64(table, base + DESC_ADDR_OFFSET, addr);
    write_u32(table, base + DESC_LEN_OFFSET, len);
    write_u16(table, base + DESC_FLAGS_OFFSET, flags);
    write_u16(table, base + DESC_NEXT_OFFSET, next);
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::virtio::avail::AvailRing;
    use crate::virtio::used::{USED_ELEM_LEN, write_used_elem};

    const USED_LEN: usize = 4 + 4 * USED_ELEM_LEN;

    type Consumed = (u16, Vec<(u64, u32, bool)>);

    struct DeviceModel {
        avail_cursor: RingCursor,
        used_cursor: RingCursor,
    }

    impl DeviceModel {
        fn new() -> Self {
            Self {
                avail_cursor: RingCursor::new(4).expect("power of two"),
                used_cursor: RingCursor::new(4).expect("power of two"),
            }
        }

        fn consume(
            &mut self,
            table: &[u8],
            avail: &[u8],
            used: &mut [u8],
        ) -> Option<Consumed> {
            let ring = AvailRing::parse(avail, 4).expect("avail ring parses");
            if self.avail_cursor.pending(ring.idx()) == 0 {
                return None;
            }
            let head = ring.ring_entry(self.avail_cursor.position());
            self.avail_cursor.advance();
            let links = DescriptorChain::new(table, 4, head)
                .map(|link| {
                    let link = link.expect("well-formed chain");
                    (link.addr(), link.buffer_len(), link.flags().device_writable())
                })
                .collect();
            let written = 0;
            self.complete(used, head, written);
            Some((head, links))
        }

        fn complete(&mut self, used: &mut [u8], head: u16, len: u32) {
            let slot = 4 + usize::from(self.used_cursor.position() % 4) * USED_ELEM_LEN;
            write_used_elem(
                &mut used[slot..],
                UsedElem {
                    id: u32::from(head),
                    len,
                },
            )
            .expect("slot fits");
            let idx = self.used_cursor.advance();
            used[2..4].copy_from_slice(&idx.to_le_bytes());
        }
    }

    /// Hand-derived from spec §2.7.5/§2.7.6, queue size 4. `new` threads the
    /// free list 0->1->2->3, so a 2-link chain (5 readable bytes at 0x1000, an
    /// 8-byte writable reply at 0x2000) takes descriptors 0 and 1: descriptor
    /// 0 is `addr=0x1000 len=5 flags=NEXT next=1`, descriptor 1 is
    /// `addr=0x2000 len=8 flags=WRITE next=0`; avail ring[0]=0, idx=1.
    #[test]
    fn two_link_chain_encodes_the_spec_example_bytes() {
        let mut table = [0u8; 4 * DESC_LEN];
        let mut avail = [0u8; 12];
        let mut queue = DriverQueue::<4>::new(&mut table).expect("table fits");

        let head = queue
            .post(&mut table, &mut avail, &[(0x1000, 5)], &[(0x2000, 8)])
            .expect("two descriptors are free");

        assert_eq!(head, 0);
        assert_eq!(
            &table[0..16],
            &[
                0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x01,
                0x00, 0x01, 0x00,
            ]
        );
        assert_eq!(
            &table[16..32],
            &[
                0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x02,
                0x00, 0x00, 0x00,
            ]
        );
        assert_eq!(
            avail,
            [0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(queue.free_descriptors(), 2);
    }

    #[test]
    fn two_readable_then_one_writable_sets_next_on_all_but_the_last() {
        let mut table = [0u8; 4 * DESC_LEN];
        let mut avail = [0u8; 12];
        let mut queue = DriverQueue::<4>::new(&mut table).expect("table fits");

        queue
            .post(
                &mut table,
                &mut avail,
                &[(0x1000, 12), (0x1100, 60)],
                &[(0x2000, 4)],
            )
            .expect("three descriptors are free");

        let flags: Vec<u16> = (0..3)
            .map(|index| read_u16(&table, index * DESC_LEN + DESC_FLAGS_OFFSET))
            .collect();
        assert_eq!(flags, [DESC_F_NEXT, DESC_F_NEXT, DESC_F_WRITE]);
    }

    /// Driver meets the existing device half: the device walks exactly the
    /// chain the driver published, completes it, and the driver gets its
    /// completion plus descriptors back — the next post reuses descriptor 1
    /// (the free list is last-freed-first: 1->0->2->3).
    #[test]
    fn existing_device_side_consumes_a_driver_chain_and_completion_frees_it() {
        let mut table = [0u8; 4 * DESC_LEN];
        let mut avail = [0u8; 12];
        let mut used = [0u8; USED_LEN];
        let mut queue = DriverQueue::<4>::new(&mut table).expect("table fits");
        let mut device = DeviceModel::new();
        queue
            .post(&mut table, &mut avail, &[(0x1000, 5)], &[(0x2000, 8)])
            .expect("post");

        let (head, links) = device
            .consume(&table, &avail, &mut used)
            .expect("one chain is pending");
        assert_eq!(head, 0);
        assert_eq!(links, [(0x1000, 5, false), (0x2000, 8, true)]);

        let (elem, first_addr) = queue
            .reap(&mut table, &used)
            .expect("well-formed used ring")
            .expect("one completion published");
        assert_eq!(elem, UsedElem { id: 0, len: 0 });
        assert_eq!(first_addr, 0x1000);
        assert_eq!(queue.free_descriptors(), 4);
        let next_head = queue
            .post(&mut table, &mut avail, &[(0x3000, 1)], &[])
            .expect("post");
        assert_eq!(next_head, 1);
        assert!(queue.reap(&mut table, &used).unwrap().is_none());
    }

    /// 70000 posts through a 4-entry ring cross both the ring-array wrap
    /// (every 4) and the 16-bit counter wrap (at 65536): the device sees each
    /// buffer address in order, and the final published idx is 70000 mod 65536.
    #[test]
    fn avail_and_used_indices_wrap_at_65536_without_losing_a_buffer() {
        let mut table = [0u8; 4 * DESC_LEN];
        let mut avail = [0u8; 12];
        let mut used = [0u8; USED_LEN];
        let mut queue = DriverQueue::<4>::new(&mut table).expect("table fits");
        let mut device = DeviceModel::new();

        for sequence in 0u64..70_000 {
            queue
                .post(&mut table, &mut avail, &[(sequence, 1)], &[])
                .expect("a descriptor is free every iteration");
            let (_, links) = device
                .consume(&table, &avail, &mut used)
                .expect("the chain just posted");
            assert_eq!(links, [(sequence, 1, false)]);
            let (_, first_addr) = queue
                .reap(&mut table, &used)
                .expect("well-formed used ring")
                .expect("completion published");
            assert_eq!(first_addr, sequence);
        }

        assert_eq!(queue.avail_idx(), (70_000u32 % 65_536) as u16);
        assert_eq!(read_u16(&avail, AVAIL_IDX_OFFSET), 4_464);
        assert_eq!(read_u16(&used, 2), 4_464);
    }

    #[test]
    fn posting_more_descriptors_than_are_free_is_queue_full() {
        let mut table = [0u8; 2 * DESC_LEN];
        let mut avail = [0u8; 8];
        let mut queue = DriverQueue::<2>::new(&mut table).expect("table fits");

        let three = queue.post(&mut table, &mut avail, &[(1, 1), (2, 1), (3, 1)], &[]);
        assert_eq!(three, Err(DecodeError::QueueFull { needed: 3, free: 2 }));
        queue
            .post(&mut table, &mut avail, &[(1, 1), (2, 1)], &[])
            .expect("exactly the free count fits");
        let one_more = queue.post(&mut table, &mut avail, &[(3, 1)], &[]);
        assert_eq!(one_more, Err(DecodeError::QueueFull { needed: 1, free: 0 }));
        assert_eq!(
            queue.post(&mut table, &mut avail, &[], &[]),
            Err(DecodeError::EmptyChain)
        );
    }

    #[test]
    fn device_completing_an_unknown_or_already_freed_head_is_rejected() {
        let mut table = [0u8; 4 * DESC_LEN];
        let mut avail = [0u8; 12];
        let mut used = [0u8; USED_LEN];
        let mut queue = DriverQueue::<4>::new(&mut table).expect("table fits");
        let mut device = DeviceModel::new();
        queue
            .post(&mut table, &mut avail, &[(0x1000, 5)], &[])
            .expect("post");

        device.complete(&mut used, 9, 0);
        assert_eq!(
            queue.reap(&mut table, &used),
            Err(DecodeError::BadUsedId {
                id: 9,
                queue_size: 4
            })
        );

        let mut used = [0u8; USED_LEN];
        let mut device = DeviceModel::new();
        device.complete(&mut used, 0, 0);
        device.complete(&mut used, 0, 0);
        assert!(queue.reap(&mut table, &used).unwrap().is_some());
        assert_eq!(
            queue.reap(&mut table, &used),
            Err(DecodeError::BadUsedId {
                id: 0,
                queue_size: 4
            }),
            "the second completion of head 0 would free a chain the driver no longer owns"
        );
    }

    #[test]
    fn short_table_is_truncated() {
        let mut table = [0u8; 4 * DESC_LEN - 1];
        assert_eq!(
            DriverQueue::<4>::new(&mut table).unwrap_err(),
            DecodeError::Truncated { need: 64, got: 63 }
        );
    }
}
