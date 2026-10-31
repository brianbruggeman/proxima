#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Marked {
    Newly,
    Already,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LedgerError {
    #[error("layered run page index exceeds bitmap capacity")]
    BitmapFull,
    #[error("layered run dirty page index list exceeded capacity")]
    IndexListFull,
}

#[derive(Clone, Debug)]
pub struct DirtyLedger {
    bitmap: Vec<u8>,
    indices: Vec<u32>,
    capacity: usize,
}

impl DirtyLedger {
    #[must_use]
    pub fn new(page_count: usize) -> Self {
        let pages = page_count.max(1);
        Self { bitmap: vec![0; pages.div_ceil(8)], indices: Vec::with_capacity(pages), capacity: pages }
    }

    pub fn clear(&mut self) {
        self.bitmap.fill(0);
        self.indices.clear();
    }

    #[must_use]
    pub fn dirty_count(&self) -> usize {
        self.indices.len()
    }

    #[must_use]
    pub fn is_dirty(&self, page: u64) -> bool {
        let (byte, mask) = Self::locate(page);
        self.bitmap.get(byte).is_some_and(|bits| bits & mask != 0)
    }

    fn locate(page: u64) -> (usize, u8) {
        let byte = usize::try_from(page / 8).unwrap_or(usize::MAX);
        (byte, 1 << (page % 8))
    }

    pub fn mark(&mut self, page: u64) -> Result<Marked, LedgerError> {
        let (byte, mask) = Self::locate(page);
        let bits = self.bitmap.get_mut(byte).ok_or(LedgerError::BitmapFull)?;
        if *bits & mask != 0 {
            return Ok(Marked::Already);
        }
        if self.indices.len() >= self.capacity {
            return Err(LedgerError::IndexListFull);
        }
        *bits |= mask;
        self.indices.push(u32::try_from(page).map_err(|_| LedgerError::BitmapFull)?);
        Ok(Marked::Newly)
    }

    pub fn drain_sorted(&mut self) -> Vec<u32> {
        let mut drained = std::mem::take(&mut self.indices);
        drained.sort_unstable();
        for page in &drained {
            let (byte, mask) = Self::locate(u64::from(*page));
            if let Some(bits) = self.bitmap.get_mut(byte) {
                *bits &= !mask;
            }
        }
        self.indices.reserve(self.capacity);
        drained
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn the_first_mark_of_a_page_is_new_and_the_second_is_already() {
        let mut ledger = DirtyLedger::new(64);

        assert_eq!(ledger.mark(9), Ok(Marked::Newly));
        assert_eq!(ledger.mark(9), Ok(Marked::Already));
        assert_eq!(ledger.dirty_count(), 1);
        assert!(ledger.is_dirty(9));
        assert!(!ledger.is_dirty(10));
    }

    #[test]
    fn draining_returns_pages_in_ascending_order_and_clears_their_bits() {
        let mut ledger = DirtyLedger::new(64);
        for page in [40, 3, 17, 3, 8] {
            ledger.mark(page).expect("within capacity");
        }

        let drained = ledger.drain_sorted();

        assert_eq!(drained, vec![3, 8, 17, 40]);
        assert_eq!(ledger.dirty_count(), 0);
        assert!((0..64).all(|page| !ledger.is_dirty(page)));
        assert_eq!(ledger.mark(3), Ok(Marked::Newly));
    }

    #[test]
    fn a_page_past_the_bitmap_is_rejected_with_the_named_error() {
        let mut ledger = DirtyLedger::new(8);

        assert_eq!(ledger.mark(8 * 8), Err(LedgerError::BitmapFull));
        assert_eq!(
            LedgerError::BitmapFull.to_string(),
            "layered run page index exceeds bitmap capacity"
        );
    }

    #[test]
    fn more_distinct_pages_than_the_index_list_holds_is_rejected() {
        let mut ledger = DirtyLedger::new(4);
        for page in 0..4 {
            ledger.mark(page).expect("within capacity");
        }

        assert_eq!(ledger.mark(4), Err(LedgerError::IndexListFull));
    }

    #[test]
    fn clear_forgets_every_dirty_page() {
        let mut ledger = DirtyLedger::new(32);
        ledger.mark(5).expect("within capacity");

        ledger.clear();

        assert_eq!(ledger.dirty_count(), 0);
        assert!(!ledger.is_dirty(5));
    }
}
