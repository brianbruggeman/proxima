//! Offline placement of a plan's output buffers.
//!
//! [`lay_out_whole_slots`] gives every output its own device buffer and hands a
//! retired buffer to a later output of the same byte length. [`lay_out_packed`]
//! gives every output that is released during the plan a byte range inside one
//! shared buffer, so a retired range serves a later output of any size that fits
//! the gap, and the shared buffer is as large as the plan's live peak plus
//! alignment, not as large as the sum of each size class's own peak.
//!
//! Both are pure functions of the allocation list, so the [`BufferArena`] build
//! that feeds them and the tests that check them need no device.

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

/// Metal requires a buffer binding offset to be a multiple of this on every
/// Apple GPU family the driver targets (`setBuffer:offset:` for device
/// address-space arguments needs 16-byte vector alignment; 256 keeps a range
/// on its own cache lines as well).
const RANGE_ALIGNMENT: usize = 256;

/// Bytes left after every packed range. A whole-slot buffer is rounded up to a
/// page by the allocator, which silently absorbed a kernel's store past its
/// last element; a range inside a shared buffer has a live neighbour there.
const RANGE_GUARD: usize = 256;

/// One output the plan writes: `bytes` long, first written at program position
/// `first`. A `resident` output is written once and read by every later call,
/// so it never shares storage with an output rewritten each call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Allocation {
    pub(super) bytes: usize,
    pub(super) first: usize,
    pub(super) resident: bool,
}

/// Where each allocation lives: `places[index]` is `(slot, byte offset)`, and
/// `slot_bytes[slot]` is how long that slot's device buffer must be.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Layout {
    pub(super) places: Vec<(usize, usize)>,
    pub(super) slot_bytes: Vec<usize>,
}

/// One buffer per output, a retired buffer reused by a later output of the
/// same length. `releases[position]` lists the allocations whose last reader is
/// `position`: they become reusable for allocations first written at
/// `position + 1` or later. `allocations` must be in program order.
pub(super) fn lay_out_whole_slots(allocations: &[Allocation], releases: &[Vec<usize>]) -> Layout {
    let mut places: Vec<(usize, usize)> = Vec::with_capacity(allocations.len());
    let mut slot_bytes: Vec<usize> = Vec::new();
    let mut free_by_size: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (position, released) in releases.iter().enumerate() {
        while let Some(allocation) = allocations
            .get(places.len())
            .filter(|each| each.first == position)
        {
            let reused = (!allocation.resident)
                .then(|| free_by_size.get_mut(&allocation.bytes).and_then(Vec::pop))
                .flatten();
            let slot = reused.unwrap_or_else(|| {
                slot_bytes.push(allocation.bytes);
                slot_bytes.len() - 1
            });
            places.push((slot, 0));
        }
        for index in released {
            let slot = places[*index].0;
            free_by_size.entry(slot_bytes[slot]).or_default().push(slot);
        }
    }
    Layout { places, slot_bytes }
}

/// One buffer per output that is never released, plus one shared buffer holding
/// every released output at a byte offset. Offsets are chosen largest first,
/// each at the lowest aligned offset that no output live at the same time
/// occupies. `group_of[position]` is the position's barrier group (non-decreasing;
/// a serial plan passes each position its own group, see [`serial_groups`]).
/// Ops in one group may run at once, so an output is live from the first
/// position of the group that first writes it to the last position of the
/// group that last reads it: two outputs overlap in bytes only when one's group
/// ends before the other's group begins, which a barrier separates.
pub(super) fn lay_out_packed(
    allocations: &[Allocation],
    releases: &[Vec<usize>],
    group_of: &[usize],
) -> Layout {
    let released_at = released_positions(allocations.len(), releases);
    let spans = lifted_spans(allocations, &released_at, group_of);
    let mut places: Vec<(usize, usize)> = vec![(0, 0); allocations.len()];
    let mut slot_bytes: Vec<usize> = vec![0];
    let mut packable: Vec<usize> = Vec::new();
    for (index, allocation) in allocations.iter().enumerate() {
        if allocation.resident || released_at[index].is_none() {
            slot_bytes.push(allocation.bytes);
            places[index] = (slot_bytes.len() - 1, 0);
        } else {
            packable.push(index);
        }
    }
    packable.sort_by_key(|index| core::cmp::Reverse(allocations[*index].bytes));
    let mut placed: Vec<(usize, usize, usize)> = Vec::with_capacity(packable.len());
    for index in packable {
        let extent = footprint(allocations[index].bytes);
        let offset = first_free_offset(&placed, &spans, spans[index], extent);
        places[index] = (0, offset);
        slot_bytes[0] = slot_bytes[0].max(offset + extent);
        placed.push((index, offset, extent));
    }
    Layout { places, slot_bytes }
}

/// Peak of the summed `bytes` of allocations live at once, counting an
/// allocation from the position that first writes it through the position that
/// last reads it.
pub(super) fn peak_live_bytes(allocations: &[Allocation], releases: &[Vec<usize>]) -> usize {
    let mut live = 0usize;
    let mut peak = 0usize;
    let mut next = 0usize;
    for (position, released) in releases.iter().enumerate() {
        while let Some(allocation) = allocations.get(next).filter(|each| each.first == position) {
            live += allocation.bytes;
            peak = peak.max(live);
            next += 1;
        }
        live -= released
            .iter()
            .map(|index| allocations[*index].bytes)
            .sum::<usize>();
    }
    peak
}

/// Every position its own barrier group: the serial dispatch order, where a
/// range released at `p` is reusable by a write at `p + 1`.
pub(super) fn serial_groups(positions: usize) -> Vec<usize> {
    (0..positions).collect()
}

/// Each allocation's lifetime lifted to whole barrier groups: from the first
/// position of the group holding its first write through the last position of
/// the group holding its last read (`usize::MAX` when never released).
fn lifted_spans(
    allocations: &[Allocation],
    released_at: &[Option<usize>],
    group_of: &[usize],
) -> Vec<(usize, usize)> {
    let group_start = |position: usize| group_of.partition_point(|group| *group < group_of[position]);
    let group_end = |position: usize| group_of.partition_point(|group| *group <= group_of[position]) - 1;
    allocations
        .iter()
        .zip(released_at)
        .map(|(each, last)| (group_start(each.first), last.map_or(usize::MAX, group_end)))
        .collect()
}

fn footprint(bytes: usize) -> usize {
    (bytes + RANGE_GUARD).next_multiple_of(RANGE_ALIGNMENT)
}

fn released_positions(count: usize, releases: &[Vec<usize>]) -> Vec<Option<usize>> {
    let mut released_at: Vec<Option<usize>> = vec![None; count];
    for (position, released) in releases.iter().enumerate() {
        for index in released {
            released_at[*index] = Some(position);
        }
    }
    released_at
}

fn first_free_offset(
    placed: &[(usize, usize, usize)],
    spans: &[(usize, usize)],
    lifetime: (usize, usize),
    extent: usize,
) -> usize {
    let mut occupied: Vec<(usize, usize)> = placed
        .iter()
        .filter(|(index, _, _)| {
            let other = spans[*index];
            other.0 <= lifetime.1 && lifetime.0 <= other.1
        })
        .map(|(_, offset, other_extent)| (*offset, *other_extent))
        .collect();
    occupied.sort_unstable();
    let mut offset = 0usize;
    for (other_offset, other_extent) in occupied {
        if offset + extent <= other_offset {
            break;
        }
        offset = offset.max(other_offset + other_extent);
    }
    offset
}

#[cfg(test)]
mod tests {
    use super::*;

    const GATE_BYTES: usize = 16_367_616;
    const DOWN_BYTES: usize = 32_735_232;
    const RESIDUAL_BYTES: usize = 4_091_904;
    const KEY_ROWS_BYTES: usize = 1_022_976;

    fn allocation(bytes: usize, first: usize) -> Allocation {
        Allocation {
            bytes,
            first,
            resident: false,
        }
    }

    fn lifetimes(allocations: &[Allocation], releases: &[Vec<usize>]) -> Vec<(usize, usize)> {
        let released_at = released_positions(allocations.len(), releases);
        allocations
            .iter()
            .zip(released_at)
            .map(|(each, last)| (each.first, last.unwrap_or(usize::MAX)))
            .collect()
    }

    fn assert_live_ranges_are_disjoint(
        allocations: &[Allocation],
        releases: &[Vec<usize>],
        layout: &Layout,
    ) {
        let spans = lifetimes(allocations, releases);
        for first in 0..allocations.len() {
            for second in first + 1..allocations.len() {
                let simultaneous =
                    spans[first].0 <= spans[second].1 && spans[second].0 <= spans[first].1;
                let (slot_a, offset_a) = layout.places[first];
                let (slot_b, offset_b) = layout.places[second];
                let overlap = slot_a == slot_b
                    && offset_a < offset_b + allocations[second].bytes
                    && offset_b < offset_a + allocations[first].bytes;
                assert!(
                    !(simultaneous && overlap),
                    "allocations {first} and {second} are live together yet share bytes"
                );
            }
        }
    }

    /// Three layers of a stacked routed feed-forward block at 999 tokens
    /// (`gate` and `hidden` are `[tokens, 8, 512]`, `down` is `[tokens, 8, 1024]`,
    /// sizes read from the granite prefill arena census), each layer also
    /// leaving one pinned key-rows output.
    fn stacked_block_layers() -> (Vec<Allocation>, Vec<Vec<usize>>) {
        let mut allocations = Vec::new();
        let mut releases: Vec<Vec<usize>> = Vec::new();
        for layer in 0..3 {
            let base = layer * 5;
            allocations.push(allocation(KEY_ROWS_BYTES, base));
            allocations.push(allocation(GATE_BYTES, base + 1));
            allocations.push(allocation(GATE_BYTES, base + 2));
            allocations.push(allocation(DOWN_BYTES, base + 3));
            allocations.push(allocation(RESIDUAL_BYTES, base + 4));
            let first = allocations.len() - 5;
            releases.extend([
                vec![],
                vec![],
                vec![first + 1],
                vec![first + 2],
                vec![first + 3, first + 4],
            ]);
        }
        (allocations, releases)
    }

    #[test]
    fn whole_slots_hand_a_released_buffer_to_a_later_output_of_the_same_length() {
        let allocations = vec![
            allocation(4096, 0),
            allocation(4096, 1),
            allocation(4096, 2),
        ];
        let releases = vec![vec![], vec![0], vec![]];

        let layout = lay_out_whole_slots(&allocations, &releases);

        assert_eq!(
            layout.slot_bytes.len(),
            2,
            "the third output reuses the first output's buffer"
        );
        assert_eq!(layout.places[2], layout.places[0]);
    }

    #[test]
    fn whole_slots_keep_a_resident_output_out_of_the_free_list() {
        let mut allocations = vec![allocation(4096, 0), allocation(4096, 1)];
        allocations[1].resident = true;
        let releases = vec![vec![0], vec![]];

        let layout = lay_out_whole_slots(&allocations, &releases);

        assert_ne!(layout.places[1].0, layout.places[0].0);
    }

    #[test]
    fn packed_range_serves_a_later_output_of_a_different_length() {
        let allocations = vec![
            allocation(8192, 0),
            allocation(1024, 1),
            allocation(4096, 2),
        ];
        let releases = vec![vec![0], vec![1], vec![2]];

        let layout = lay_out_packed(&allocations, &releases, &serial_groups(releases.len()));

        assert_eq!(
            layout.slot_bytes.len(),
            1,
            "every output is released, so one shared buffer holds all"
        );
        assert_eq!(
            layout.slot_bytes[0],
            footprint(8192),
            "released outputs overlap in time with nothing, so the buffer is the largest range"
        );
    }

    #[test]
    fn packed_outputs_live_together_never_share_bytes() {
        let (allocations, releases) = stacked_block_layers();
        let packed = lay_out_packed(&allocations, &releases, &serial_groups(releases.len()));
        let whole = lay_out_whole_slots(&allocations, &releases);

        assert_live_ranges_are_disjoint(&allocations, &releases, &packed);
        assert_live_ranges_are_disjoint(&allocations, &releases, &whole);
    }

    #[test]
    fn packed_stacked_block_needs_the_hidden_and_down_buffers_not_the_sum_of_size_classes() {
        let (allocations, releases) = stacked_block_layers();

        let packed = lay_out_packed(&allocations, &releases, &serial_groups(releases.len()));
        let whole = lay_out_whole_slots(&allocations, &releases);

        let packed_total: usize = packed.slot_bytes.iter().sum();
        let whole_total: usize = whole.slot_bytes.iter().sum();
        assert_eq!(
            packed.slot_bytes[0],
            footprint(DOWN_BYTES) + footprint(GATE_BYTES),
            "gate and the residual reuse the down range; hidden sits beside it"
        );
        assert!(
            whole_total - packed_total >= GATE_BYTES,
            "packing saves the second gate-sized buffer: whole {whole_total}, packed {packed_total}"
        );
    }

    #[test]
    fn packed_buffer_is_never_smaller_than_the_live_peak() {
        let (allocations, releases) = stacked_block_layers();

        let layout = lay_out_packed(&allocations, &releases, &serial_groups(releases.len()));

        let pinned: usize = allocations
            .iter()
            .enumerate()
            .filter(|(index, _)| layout.places[*index].0 != 0)
            .map(|(_, each)| each.bytes)
            .sum();
        let peak = peak_live_bytes(&allocations, &releases);
        assert!(layout.slot_bytes[0] + pinned >= peak);
    }

    #[test]
    fn packed_placement_is_disjoint_for_pseudo_random_programs() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move |bound: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % bound as u64) as usize
        };
        for _ in 0..200 {
            let positions = 8 + next(40);
            let count = 4 + next(60);
            let mut allocations: Vec<Allocation> = (0..count)
                .map(|_| Allocation {
                    bytes: 1 + next(1 << 16),
                    first: next(positions),
                    resident: next(9) == 0,
                })
                .collect();
            allocations.sort_by_key(|each| each.first);
            let mut releases: Vec<Vec<usize>> = vec![Vec::new(); positions];
            for (index, each) in allocations.iter().enumerate() {
                if !each.resident && next(4) != 0 {
                    releases[each.first + next(positions - each.first)].push(index);
                }
            }

            let packed = lay_out_packed(&allocations, &releases, &serial_groups(releases.len()));
            let whole = lay_out_whole_slots(&allocations, &releases);

            assert_live_ranges_are_disjoint(&allocations, &releases, &packed);
            assert_live_ranges_are_disjoint(&allocations, &releases, &whole);
            for (index, each) in allocations.iter().enumerate() {
                let (slot, offset) = packed.places[index];
                assert!(offset + each.bytes <= packed.slot_bytes[slot]);
                assert_eq!(offset % RANGE_ALIGNMENT, 0);
            }
        }
    }

    /// Worked example, derived by hand before the code (`FP(n) = (n + 256)
    /// rounded up to 256`): six ops in two barrier groups, `group_of =
    /// [0, 0, 0, 1, 1, 1]`, so ops 0..=2 may run at once and ops 3..=5 may
    /// run at once, with a barrier between the groups. Op 5 is output-placed
    /// and owns no allocation; ops 0..=4 write five outputs of distinct size:
    ///
    /// | out | bytes | FP   | first | last read | released at |
    /// |-----|-------|------|-------|-----------|-------------|
    /// | A   | 1792  | 2048 | 0     | 1         | 1           |
    /// | B   | 1280  | 1536 | 1     | 2         | 2           |
    /// | C   |  768  | 1024 | 2     | 3         | 3           |
    /// | D   |  512  |  768 | 3     | 5         | 5           |
    /// | E   |  256  |  512 | 4     | 5         | 5           |
    ///
    /// Lifted to groups (first write -> start of its group, last read -> end
    /// of its group; group 0 is positions 0..=2, group 1 is 3..=5):
    /// A [0,2], B [0,2], C [0,5] (written in group 0, read in group 1),
    /// D [3,5], E [3,5].
    ///
    /// Placement is largest first, each at the lowest offset clear of every
    /// output whose lifted span meets its own:
    /// - A: nothing placed yet, offset 0, ends at 2048.
    /// - B: meets A, so offset 2048, ends at 3584.
    /// - C: meets A and B, so offset 3584, ends at 4608.
    /// - D: meets only C (A and B end at 2 < 3), so the gap at 0 holds its 768
    ///   bytes before C at 3584: offset 0.
    /// - E: meets C and D, D occupies [0, 768), so offset 768, and 768 + 512
    ///   = 1280 <= 3584 fits before C.
    ///
    /// Slot 0 is `max(end) = 4608` bytes; no output is pinned, so there is no
    /// other slot.
    #[test]
    fn concurrent_two_group_plan_packs_to_the_hand_derived_offsets() {
        let allocations = vec![
            allocation(1792, 0),
            allocation(1280, 1),
            allocation(768, 2),
            allocation(512, 3),
            allocation(256, 4),
        ];
        let releases = vec![vec![], vec![0], vec![1], vec![2], vec![], vec![3, 4]];
        let group_of = vec![0, 0, 0, 1, 1, 1];

        let layout = lay_out_packed(&allocations, &releases, &group_of);

        assert_eq!(
            layout.places,
            vec![(0, 0), (0, 2048), (0, 3584), (0, 0), (0, 768)]
        );
        assert_eq!(layout.slot_bytes, vec![4608]);
    }

    #[test]
    fn concurrent_outputs_sharing_a_barrier_group_never_share_bytes_even_when_positions_are_disjoint() {
        let allocations = vec![allocation(1024, 0), allocation(1024, 2)];
        let releases = vec![vec![], vec![0], vec![], vec![1]];
        let one_group = vec![0, 0, 0, 0];

        let concurrent = lay_out_packed(&allocations, &releases, &one_group);
        let serial = lay_out_packed(&allocations, &releases, &serial_groups(4));

        assert_eq!(serial.places, vec![(0, 0), (0, 0)], "serial reuses the range");
        assert_eq!(
            concurrent.places,
            vec![(0, 0), (0, footprint(1024))],
            "ops 0 and 2 may run at once, so the second range sits beside the first"
        );
        assert_eq!(concurrent.slot_bytes, vec![2 * footprint(1024)]);
    }

    #[test]
    fn concurrent_packed_placement_is_disjoint_for_every_pair_that_meets_in_a_group() {
        let mut state = 0xd1b5_4a32_d192_ed03_u64;
        let mut next = move |bound: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % bound as u64) as usize
        };
        for _ in 0..200 {
            let positions = 8 + next(40);
            let mut group = 0;
            let group_of: Vec<usize> = (0..positions)
                .map(|_| {
                    group += usize::from(next(3) == 0);
                    group
                })
                .collect();
            let count = 4 + next(60);
            let mut allocations: Vec<Allocation> = (0..count)
                .map(|_| Allocation {
                    bytes: 1 + next(1 << 16),
                    first: next(positions),
                    resident: next(9) == 0,
                })
                .collect();
            allocations.sort_by_key(|each| each.first);
            let mut releases: Vec<Vec<usize>> = vec![Vec::new(); positions];
            for (index, each) in allocations.iter().enumerate() {
                if !each.resident && next(4) != 0 {
                    releases[each.first + next(positions - each.first)].push(index);
                }
            }

            let layout = lay_out_packed(&allocations, &releases, &group_of);

            let released_at = released_positions(allocations.len(), &releases);
            let spans = lifted_spans(&allocations, &released_at, &group_of);
            for first in 0..allocations.len() {
                for second in first + 1..allocations.len() {
                    let meet = spans[first].0 <= spans[second].1 && spans[second].0 <= spans[first].1;
                    let (slot_a, offset_a) = layout.places[first];
                    let (slot_b, offset_b) = layout.places[second];
                    let overlap = slot_a == slot_b
                        && offset_a < offset_b + allocations[second].bytes
                        && offset_b < offset_a + allocations[first].bytes;
                    assert!(!(meet && overlap), "{first} and {second} meet in a group yet share bytes");
                }
            }
        }
    }

    /// The decode plan's attention scratch: one `532_480`-byte range per
    /// attention op (the compiled split ceiling), written by the op's partial
    /// dispatch and read by its merge dispatch, so released at its own position.
    /// 49 ops, each in the barrier group opened by the previous op's output
    /// being read, never share a group, so the 49 ranges reuse one.
    #[test]
    fn sequential_attention_scratch_in_different_groups_shares_one_range() {
        const SCRATCH_BYTES: usize = 532_480;
        const ATTENTION_OPS: usize = 49;
        let allocations: Vec<Allocation> = (0..ATTENTION_OPS)
            .map(|position| allocation(SCRATCH_BYTES, position))
            .collect();
        let releases: Vec<Vec<usize>> = (0..ATTENTION_OPS).map(|index| vec![index]).collect();
        let group_of: Vec<usize> = (0..ATTENTION_OPS).collect();

        let layout = lay_out_packed(&allocations, &releases, &group_of);

        assert!(layout.places.iter().all(|place| *place == (0, 0)));
        assert_eq!(layout.slot_bytes, vec![footprint(SCRATCH_BYTES)]);
        assert_eq!(footprint(SCRATCH_BYTES), 532_736);
        assert_eq!(ATTENTION_OPS * SCRATCH_BYTES, 26_091_520);
    }

    #[test]
    fn packing_an_empty_plan_allocates_only_the_empty_shared_buffer() {
        let layout = lay_out_packed(&[], &[], &[]);

        assert_eq!(layout.slot_bytes, vec![0]);
        assert!(layout.places.is_empty());
    }
}
