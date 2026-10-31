use arrayvec::ArrayVec;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotRegion {
    pub slot: u32,
    pub flags: u32,
    pub guest_address: u64,
    pub size: u64,
    pub host: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotOp {
    Submit(SlotRegion),
    Retire { slot: u32, guest_address: u64, host: u64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("page range [{first_page}, {end_page}) does not lie inside one mapped run")]
pub struct RetargetOutsideRun {
    pub first_page: u64,
    pub end_page: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Run {
    first_page: u64,
    end_page: u64,
    host: u64,
    flags: u32,
    slot: u32,
}

#[derive(Clone, Debug)]
pub struct SlotTable {
    guest_base: u64,
    granule: u64,
    runs: Vec<Run>,
    free_slots: Vec<u32>,
    next_slot: u32,
}

impl SlotTable {
    #[must_use]
    pub fn new(guest_base: u64, granule: u64) -> Self {
        Self { guest_base, granule, runs: Vec::new(), free_slots: Vec::new(), next_slot: 0 }
    }

    #[must_use]
    pub fn granule(&self) -> u64 {
        self.granule
    }

    #[must_use]
    pub fn run_count(&self) -> usize {
        self.runs.len()
    }

    fn take_slot(&mut self) -> u32 {
        self.free_slots.pop().unwrap_or_else(|| {
            let slot = self.next_slot;
            self.next_slot += 1;
            slot
        })
    }

    fn submit(&self, run: &Run) -> SlotOp {
        SlotOp::Submit(SlotRegion {
            slot: run.slot,
            flags: run.flags,
            guest_address: self.guest_base + run.first_page * self.granule,
            size: (run.end_page - run.first_page) * self.granule,
            host: run.host,
        })
    }

    fn withdraw(&self, run: &Run) -> SlotOp {
        SlotOp::Retire {
            slot: run.slot,
            guest_address: self.guest_base + run.first_page * self.granule,
            host: run.host,
        }
    }

    pub fn install(&mut self, first_page: u64, end_page: u64, host: u64, flags: u32) -> SlotOp {
        let run = Run { first_page, end_page, host, flags, slot: self.take_slot() };
        self.runs.push(run);
        self.submit(&run)
    }

    pub fn clear(&mut self) -> Vec<SlotOp> {
        let retired = self.runs.iter().map(|run| self.withdraw(run)).collect();
        self.runs.clear();
        self.free_slots.clear();
        self.next_slot = 0;
        retired
    }

    #[must_use]
    pub fn find(&self, page: u64) -> Option<usize> {
        let index = self.runs.partition_point(|run| run.end_page <= page);
        self.runs.get(index).filter(|run| run.first_page <= page).map(|_| index)
    }

    #[must_use]
    pub fn host_of(&self, page: u64) -> Option<u64> {
        let run = self.runs[self.find(page)?];
        Some(run.host + (page - run.first_page) * self.granule)
    }

    pub fn retarget(
        &mut self,
        first_page: u64,
        end_page: u64,
        host: u64,
        flags: u32,
    ) -> Result<ArrayVec<SlotOp, 4>, RetargetOutsideRun> {
        let outside = RetargetOutsideRun { first_page, end_page };
        let found = self.find(first_page).ok_or(outside)?;
        let original = self.runs[found];
        if end_page > original.end_page {
            return Err(outside);
        }
        let mut pieces: ArrayVec<Run, 3> = ArrayVec::new();
        if original.first_page < first_page {
            pieces.push(Run { end_page: first_page, ..original });
        }
        pieces.push(Run { first_page, end_page, host, flags, slot: 0 });
        if end_page < original.end_page {
            let skipped = (end_page - original.first_page) * self.granule;
            pieces.push(Run { first_page: end_page, host: original.host + skipped, ..original });
        }
        let mut ops: ArrayVec<SlotOp, 4> = ArrayVec::new();
        ops.push(self.withdraw(&original));
        self.free_slots.push(original.slot);
        for piece in &mut pieces {
            piece.slot = self.take_slot();
            ops.push(self.submit(piece));
        }
        self.runs.splice(found..=found, pieces);
        Ok(ops)
    }

    pub fn set_all_flags(&mut self, flags: u32) -> Vec<SlotOp> {
        for run in &mut self.runs {
            run.flags = flags;
        }
        self.runs.iter().map(|run| self.submit(run)).collect()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    const GRANULE: u64 = 4096;
    const READ_ONLY: u32 = 1 << 3;

    fn region(op: SlotOp) -> SlotRegion {
        match op {
            SlotOp::Submit(region) => region,
            SlotOp::Retire { .. } => panic!("expected a submit"),
        }
    }

    #[test]
    fn retargeting_the_middle_page_of_a_run_splits_it_into_three_and_reuses_the_retired_slot() {
        let mut table = SlotTable::new(0x1000_0000, GRANULE);
        table.install(0, 8, 0xa000_0000, READ_ONLY);

        let ops = table.retarget(3, 4, 0xb000_0000, 0).expect("page 3 lies in run 0");

        assert_eq!(ops.len(), 4);
        assert_eq!(ops[0], SlotOp::Retire { slot: 0, guest_address: 0x1000_0000, host: 0xa000_0000 });
        let left = region(ops[1]);
        let target = region(ops[2]);
        let right = region(ops[3]);
        assert_eq!((left.slot, left.size, left.host, left.flags), (0, 3 * GRANULE, 0xa000_0000, READ_ONLY));
        assert_eq!((target.slot, target.guest_address, target.host, target.flags), (1, 0x1000_3000, 0xb000_0000, 0));
        assert_eq!((right.slot, right.size, right.host, right.flags), (2, 4 * GRANULE, 0xa000_4000, READ_ONLY));
        assert_eq!(table.run_count(), 3);
        assert_eq!(table.host_of(5), Some(0xa000_5000));
        assert_eq!(table.host_of(3), Some(0xb000_0000));
    }

    #[test]
    fn retargeting_a_whole_run_yields_one_retire_and_one_submit() {
        let mut table = SlotTable::new(0, GRANULE);
        table.install(0, 2, 0xa000, 0);

        let ops = table.retarget(0, 2, 0xc000, READ_ONLY).expect("whole run");

        assert_eq!(ops.len(), 2);
        assert_eq!(table.run_count(), 1);
    }

    #[test]
    fn retargeting_across_a_run_boundary_is_rejected() {
        let mut table = SlotTable::new(0, GRANULE);
        table.install(0, 2, 0xa000, 0);

        let error = table.retarget(1, 3, 0xc000, 0).expect_err("page 2 is unmapped");

        assert_eq!(error, RetargetOutsideRun { first_page: 1, end_page: 3 });
        assert_eq!(table.find(2), None);
    }

    #[test]
    fn clear_retires_every_run_and_resets_slot_numbering() {
        let mut table = SlotTable::new(0, GRANULE);
        table.install(0, 1, 0xa000, 0);
        table.install(4, 6, 0xb000, 0);

        let retired = table.clear();

        assert_eq!(retired.len(), 2);
        assert_eq!(table.run_count(), 0);
        assert_eq!(region(table.install(0, 1, 0xa000, 0)).slot, 0);
    }

    #[test]
    fn find_locates_the_run_holding_a_page_by_binary_search() {
        let mut table = SlotTable::new(0, GRANULE);
        for index in 0..16 {
            table.install(index * 4, index * 4 + 2, 0xa000 + index * 0x10000, 0);
        }

        assert_eq!(table.find(10), None);
        assert_eq!(table.find(8), Some(2));
        assert_eq!(table.find(13), Some(3));
    }

    #[test]
    fn set_all_flags_resubmits_every_run_with_the_new_flags() {
        let mut table = SlotTable::new(0, GRANULE);
        table.install(0, 1, 0xa000, 0);
        table.install(2, 3, 0xb000, 0);

        let ops = table.set_all_flags(READ_ONLY);

        assert!(ops.iter().all(|op| region(*op).flags == READ_ONLY));
    }
}
