use super::error::DecodeError;

const MIN_DEPTH: u32 = 2;
const MAX_DEPTH: u32 = 65536;

#[inline]
fn validate_depth(depth: u32) -> Result<u32, DecodeError> {
    if (MIN_DEPTH..=MAX_DEPTH).contains(&depth) {
        Ok(depth)
    } else {
        Err(DecodeError::BadQueueDepth { depth })
    }
}

#[inline]
fn validate_cursor(cursor: u16, depth: u32) -> Result<u16, DecodeError> {
    if u32::from(cursor) < depth {
        Ok(cursor)
    } else {
        Err(DecodeError::BadCursor { cursor, depth })
    }
}

/// Host-side producer cursor over a submission-queue ring. Pure index
/// arithmetic — it owns no memory and does no I/O. The caller writes a
/// [`super::command::SubmissionEntry`] at [`SubmissionRing::slot`], then calls
/// [`SubmissionRing::advance`] and writes the returned value to the SQ tail
/// doorbell register.
#[derive(Debug, Clone, Copy)]
pub struct SubmissionRing {
    depth: u32,
    tail: u16,
}

impl SubmissionRing {
    pub fn new(depth: u32) -> Result<Self, DecodeError> {
        Ok(Self {
            depth: validate_depth(depth)?,
            tail: 0,
        })
    }

    /// Rebuild a ring from a persisted/atomic cursor — the inverse of reading
    /// [`SubmissionRing::slot`] out. Lets a stateless engine keep the tail in an
    /// atomic and reconstitute the tested FSM per operation.
    pub fn resume(depth: u32, tail: u16) -> Result<Self, DecodeError> {
        let depth = validate_depth(depth)?;
        Ok(Self {
            depth,
            tail: validate_cursor(tail, depth)?,
        })
    }

    /// Slot index where the next SQE is written.
    #[must_use]
    pub fn slot(&self) -> u16 {
        self.tail
    }

    /// Would submitting one more command collide with the controller's head?
    /// The ring keeps one slot empty to distinguish full from empty, so full is
    /// when the next tail would equal `controller_head`.
    #[must_use]
    pub fn is_full(&self, controller_head: u16) -> bool {
        self.next(self.tail) == controller_head
    }

    /// Advance the tail past the slot just written and return the new tail —
    /// the value to publish to the SQ tail doorbell.
    pub fn advance(&mut self) -> u16 {
        self.tail = self.next(self.tail);
        self.tail
    }

    #[inline]
    fn next(&self, index: u16) -> u16 {
        ((u32::from(index) + 1) % self.depth) as u16
    }
}

/// Host-side consumer cursor over a completion-queue ring. Tracks the head and
/// the expected phase bit; flips the phase every wrap. Owns no memory, does no
/// I/O — the caller polls the slot's phase tag and feeds it to
/// [`CompletionRing::is_ready`].
#[derive(Debug, Clone, Copy)]
pub struct CompletionRing {
    depth: u32,
    head: u16,
    expected_phase: bool,
}

impl CompletionRing {
    pub fn new(depth: u32) -> Result<Self, DecodeError> {
        // First lap the controller writes phase tag 1 into zero-initialised
        // memory, so the host's first expectation is 1 (true).
        Ok(Self {
            depth: validate_depth(depth)?,
            head: 0,
            expected_phase: true,
        })
    }

    /// Rebuild a ring from a persisted head + expected-phase pair. Lets a
    /// stateless engine keep the cursor in atomics and reconstitute the FSM.
    pub fn resume(depth: u32, head: u16, expected_phase: bool) -> Result<Self, DecodeError> {
        let depth = validate_depth(depth)?;
        Ok(Self {
            depth,
            head: validate_cursor(head, depth)?,
            expected_phase,
        })
    }

    /// Slot index of the next completion to inspect.
    #[must_use]
    pub fn slot(&self) -> u16 {
        self.head
    }

    #[must_use]
    pub fn expected_phase(&self) -> bool {
        self.expected_phase
    }

    /// A slot whose phase tag matches the expected phase is a fresh completion;
    /// otherwise it is stale memory from the previous lap (or never written).
    #[must_use]
    pub fn is_ready(&self, slot_phase: bool) -> bool {
        slot_phase == self.expected_phase
    }

    /// Consume the current completion: advance the head, flip the expected phase
    /// on wrap, and return the new head — the value to publish to the CQ head
    /// doorbell.
    pub fn advance(&mut self) -> u16 {
        let next = ((u32::from(self.head) + 1) % self.depth) as u16;
        if next == 0 {
            self.expected_phase = !self.expected_phase;
        }
        self.head = next;
        self.head
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn depth_outside_legal_range_is_rejected() {
        for bad in [0u32, 1, 65537, 100_000] {
            assert_eq!(
                SubmissionRing::new(bad).unwrap_err(),
                DecodeError::BadQueueDepth { depth: bad }
            );
            assert_eq!(
                CompletionRing::new(bad).unwrap_err(),
                DecodeError::BadQueueDepth { depth: bad }
            );
        }
        assert!(SubmissionRing::new(2).is_ok());
        assert!(SubmissionRing::new(65536).is_ok());
    }

    #[test]
    fn resume_round_trips_cursors_and_rejects_out_of_range() {
        let mut ring = SubmissionRing::new(8).unwrap();
        ring.advance();
        ring.advance();
        let resumed = SubmissionRing::resume(8, ring.slot()).unwrap();
        assert_eq!(resumed.slot(), 2);

        let mut com = CompletionRing::new(4).unwrap();
        for _ in 0..4 {
            com.advance();
        }
        let resumed_com = CompletionRing::resume(4, com.slot(), com.expected_phase()).unwrap();
        assert_eq!(resumed_com.slot(), 0);
        assert!(!resumed_com.expected_phase());

        assert_eq!(
            SubmissionRing::resume(8, 8).unwrap_err(),
            DecodeError::BadCursor {
                cursor: 8,
                depth: 8
            }
        );
    }

    #[test]
    fn submission_tail_wraps_and_keeps_one_slot_empty() {
        // depth 4 holds at most 3 outstanding commands so full is
        // distinguishable from empty.
        let mut ring = SubmissionRing::new(4).unwrap();
        assert_eq!(ring.slot(), 0);
        assert!(!ring.is_full(0));

        assert_eq!(ring.advance(), 1);
        assert_eq!(ring.advance(), 2);
        assert_eq!(ring.advance(), 3);
        // tail==3, controller head==0: next tail would be 0 == head -> full.
        assert!(ring.is_full(0));
        // wrap back to 0 on the next advance.
        assert_eq!(ring.advance(), 0);
    }

    #[test]
    fn completion_phase_flips_every_wrap() {
        let mut ring = CompletionRing::new(4).unwrap();
        // first lap: controller writes phase tag 1, host expects 1.
        assert!(ring.expected_phase());
        for slot in 0..4 {
            assert_eq!(ring.slot(), slot);
            assert!(ring.is_ready(true), "phase-1 slot is fresh on lap one");
            assert!(!ring.is_ready(false), "phase-0 slot is stale on lap one");
            ring.advance();
        }
        // wrapped once: now expect phase 0 for the second lap.
        assert_eq!(ring.slot(), 0);
        assert!(!ring.expected_phase());
        for _ in 0..4 {
            assert!(ring.is_ready(false), "phase-0 slot is fresh on lap two");
            assert!(!ring.is_ready(true), "phase-1 slot is stale on lap two");
            ring.advance();
        }
        // wrapped twice: phase is back to 1.
        assert!(ring.expected_phase());
    }

    #[test]
    fn completion_head_doorbell_is_the_post_advance_index() {
        let mut ring = CompletionRing::new(8).unwrap();
        assert_eq!(ring.advance(), 1);
        assert_eq!(ring.advance(), 2);
        assert_eq!(ring.slot(), 2);
    }

    #[test]
    fn minimum_depth_two_alternates_phase_each_step() {
        let mut ring = CompletionRing::new(2).unwrap();
        assert!(ring.expected_phase());
        ring.advance();
        assert!(ring.expected_phase(), "no wrap yet at head 1");
        ring.advance();
        assert!(!ring.expected_phase(), "wrapped at head 0, phase flipped");
    }

    #[test]
    fn max_depth_indices_never_overflow_u16() {
        // depth 65536 puts the last slot at index 65535; advancing from there
        // must wrap to 0 without overflowing the u16 cursor.
        let mut ring = SubmissionRing::new(65536).unwrap();
        for _ in 0..65535 {
            ring.advance();
        }
        assert_eq!(ring.slot(), 65535);
        assert_eq!(ring.advance(), 0);
    }
}
