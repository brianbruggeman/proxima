//! Device-resident attention KV bookkeeping as a sans-IO state machine: events in, the next state and the device commands to run out. A backend executes the commands with its own primitives (allocate, copy, read back); the decisions (when a cache grows, where new rows land, how many rows survive a rejected draft) live here once, for every backend.
//!
//! `valid_rows` is the only authority on how much of a carried cache is real. A rejected speculative row is dropped by moving that count back ([`KvCommand::Truncate`]); no bytes move.

use alloc::vec::Vec;
use alloc::collections::{BTreeMap, BTreeSet};
use core::fmt::Debug;

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum KvCarryError<Key: Debug> {
    #[error("kv carry: {attempted} is not legal in the {phase} phase")]
    IllegalEvent {
        attempted: &'static str,
        phase: &'static str,
    },
    #[error("kv carry: the carried pair set cannot change while {live} inputs are live")]
    PairsChangedWhileLive { live: usize },
    #[error("kv carry: input {input:?} is not a selected pair")]
    UnknownInput { input: Key },
    #[error("kv carry: input {input:?} holds {valid_elements} valid elements in {capacity_elements} of capacity")]
    Overflow {
        input: Key,
        valid_elements: u64,
        capacity_elements: u64,
    },
    #[error("kv carry: input {input:?} kept {kept} rows of the {appended} it appended")]
    KeptBeyondAppended {
        input: Key,
        kept: u64,
        appended: u64,
    },
    #[error("kv carry: step began at {cached_rows} cached rows but input {input:?} holds {valid_rows}")]
    RowsDesynchronized {
        input: Key,
        cached_rows: u64,
        valid_rows: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowPair<Key> {
    pub input: Key,
    pub output: Key,
    pub row_width: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent<Key> {
    pub input: Key,
    pub cached_rows: Option<u64>,
    pub extent_elements: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Appended<Key> {
    pub input: Key,
    pub new_rows: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvEvent<Key> {
    Select { pairs: Vec<RowPair<Key>> },
    Resident { input: Key, valid_rows: u64, capacity_elements: u64 },
    StepBegan { extents: Vec<Extent<Key>> },
    RowsAppended { appended: Vec<Appended<Key>> },
    Accepted { kept: u64 },
    Parked,
    Ended,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCommand<Key> {
    Adopt { input: Key },
    Park { input: Key, valid_rows: u64 },
    Grow { input: Key, to_elements: u64, keep_elements: u64 },
    AppendCopy { input: Key, output: Key, offset_elements: u64, count_elements: u64 },
    Truncate { input: Key, to_rows: u64 },
    Drain { input: Key, valid_rows: u64, row_width: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Idle,
    Began,
    Appended,
}

impl Phase {
    const fn name(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Began => "began",
            Self::Appended => "appended",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Lane {
    valid_rows: u64,
    capacity_elements: u64,
    last_appended: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCarry<Key: Ord + Copy + Debug> {
    pairs: BTreeMap<Key, RowPair<Key>>,
    lanes: BTreeMap<Key, Lane>,
    phase: Phase,
}

impl<Key: Ord + Copy + Debug> Default for KvCarry<Key> {
    fn default() -> Self {
        Self {
            pairs: BTreeMap::new(),
            lanes: BTreeMap::new(),
            phase: Phase::Idle,
        }
    }
}

impl<Key: Ord + Copy + Debug> KvCarry<Key> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn is_live(&self, input: &Key) -> bool {
        self.lanes.contains_key(input)
    }

    #[must_use]
    pub fn is_pair(&self, input: &Key) -> bool {
        self.pairs.contains_key(input)
    }

    #[must_use]
    pub fn output_of(&self, input: &Key) -> Option<Key> {
        self.pairs.get(input).map(|pair| pair.output)
    }

    #[must_use]
    pub fn valid_rows(&self, input: &Key) -> Option<u64> {
        self.lanes.get(input).map(|lane| lane.valid_rows)
    }

    #[must_use]
    pub fn has_live(&self) -> bool {
        !self.lanes.is_empty()
    }

    #[must_use]
    pub fn selects(&self, pairs: &[RowPair<Key>]) -> bool {
        pairs.len() == self.pairs.len()
            && pairs.iter().all(|pair| self.pairs.get(&pair.input) == Some(pair))
    }

    pub fn step(&mut self, event: KvEvent<Key>) -> Result<Vec<KvCommand<Key>>, KvCarryError<Key>> {
        match event {
            KvEvent::Select { pairs } => self.select(pairs),
            KvEvent::Resident { input, valid_rows, capacity_elements } => {
                self.make_live(input, valid_rows, capacity_elements)
            }
            KvEvent::StepBegan { extents } => self.begin(&extents),
            KvEvent::RowsAppended { appended } => self.append(&appended),
            KvEvent::Accepted { kept } => self.accept(kept),
            KvEvent::Parked => Ok(self.park()),
            KvEvent::Ended => Ok(self.end()),
        }
    }

    pub fn park(&mut self) -> Vec<KvCommand<Key>> {
        self.release(|input, lane, _| KvCommand::Park {
            input,
            valid_rows: lane.valid_rows,
        })
    }

    pub fn end(&mut self) -> Vec<KvCommand<Key>> {
        self.release(|input, lane, pair| KvCommand::Drain {
            input,
            valid_rows: lane.valid_rows,
            row_width: pair.row_width,
        })
    }

    pub fn pairs(&self) -> impl Iterator<Item = &RowPair<Key>> {
        self.pairs.values()
    }

    fn select(
        &mut self,
        pairs: Vec<RowPair<Key>>,
    ) -> Result<Vec<KvCommand<Key>>, KvCarryError<Key>> {
        if self.selects(&pairs) {
            return Ok(self.adoptions());
        }
        if !self.lanes.is_empty() {
            return Err(KvCarryError::PairsChangedWhileLive { live: self.lanes.len() });
        }
        self.pairs = pairs.into_iter().map(|pair| (pair.input, pair)).collect();
        self.phase = Phase::Idle;
        Ok(self.adoptions())
    }

    fn adoptions(&self) -> Vec<KvCommand<Key>> {
        self.pairs
            .keys()
            .filter(|input| !self.lanes.contains_key(*input))
            .map(|input| KvCommand::Adopt { input: *input })
            .collect()
    }

    fn make_live(
        &mut self,
        input: Key,
        valid_rows: u64,
        capacity_elements: u64,
    ) -> Result<Vec<KvCommand<Key>>, KvCarryError<Key>> {
        if !self.pairs.contains_key(&input) {
            return Err(KvCarryError::UnknownInput { input });
        }
        self.lanes.insert(
            input,
            Lane {
                valid_rows,
                capacity_elements,
                last_appended: 0,
            },
        );
        Ok(Vec::new())
    }

    fn begin(&mut self, extents: &[Extent<Key>]) -> Result<Vec<KvCommand<Key>>, KvCarryError<Key>> {
        let mut commands = Vec::new();
        for extent in extents {
            let (Some(lane), Some(pair)) =
                (self.lanes.get_mut(&extent.input), self.pairs.get(&extent.input))
            else {
                continue;
            };
            if let Some(cached_rows) = extent.cached_rows.filter(|rows| *rows != lane.valid_rows) {
                return Err(KvCarryError::RowsDesynchronized {
                    input: extent.input,
                    cached_rows,
                    valid_rows: lane.valid_rows,
                });
            }
            lane.last_appended = 0;
            if lane.capacity_elements >= extent.extent_elements {
                continue;
            }
            let valid_elements = lane.valid_rows * pair.row_width;
            if valid_elements > lane.capacity_elements {
                return Err(KvCarryError::Overflow {
                    input: extent.input,
                    valid_elements,
                    capacity_elements: lane.capacity_elements,
                });
            }
            lane.capacity_elements = extent.extent_elements;
            commands.push(KvCommand::Grow {
                input: extent.input,
                to_elements: extent.extent_elements,
                keep_elements: valid_elements,
            });
        }
        self.phase = Phase::Began;
        Ok(commands)
    }

    fn append(
        &mut self,
        appended: &[Appended<Key>],
    ) -> Result<Vec<KvCommand<Key>>, KvCarryError<Key>> {
        if self.phase != Phase::Began {
            return Err(self.illegal("rows_appended"));
        }
        let mut commands = Vec::new();
        for rows in appended {
            let (Some(lane), Some(pair)) =
                (self.lanes.get_mut(&rows.input), self.pairs.get(&rows.input))
            else {
                continue;
            };
            let offset = lane.valid_rows * pair.row_width;
            let count = rows.new_rows * pair.row_width;
            if offset + count > lane.capacity_elements {
                let to_elements = (offset + count).next_power_of_two();
                commands.push(KvCommand::Grow {
                    input: rows.input,
                    to_elements,
                    keep_elements: offset.min(lane.capacity_elements),
                });
                lane.capacity_elements = to_elements;
            }
            if count > 0 {
                commands.push(KvCommand::AppendCopy {
                    input: rows.input,
                    output: pair.output,
                    offset_elements: offset,
                    count_elements: count,
                });
            }
            lane.valid_rows += rows.new_rows;
            lane.last_appended = rows.new_rows;
        }
        self.phase = Phase::Appended;
        Ok(commands)
    }

    fn accept(&mut self, kept: u64) -> Result<Vec<KvCommand<Key>>, KvCarryError<Key>> {
        if self.phase != Phase::Appended {
            return Err(self.illegal("accepted"));
        }
        if let Some((input, lane)) = self.lanes.iter().find(|(_, lane)| kept > lane.last_appended) {
            return Err(KvCarryError::KeptBeyondAppended {
                input: *input,
                kept,
                appended: lane.last_appended,
            });
        }
        let mut commands = Vec::new();
        for (input, lane) in &mut self.lanes {
            let rejected = lane.last_appended - kept;
            lane.last_appended = kept;
            if rejected > 0 {
                lane.valid_rows -= rejected;
                commands.push(KvCommand::Truncate {
                    input: *input,
                    to_rows: lane.valid_rows,
                });
            }
        }
        self.phase = Phase::Idle;
        Ok(commands)
    }

    fn release(
        &mut self,
        command: impl Fn(Key, &Lane, &RowPair<Key>) -> KvCommand<Key>,
    ) -> Vec<KvCommand<Key>> {
        self.phase = Phase::Idle;
        let lanes = core::mem::take(&mut self.lanes);
        lanes
            .iter()
            .filter_map(|(input, lane)| {
                self.pairs.get(input).map(|pair| command(*input, lane, pair))
            })
            .collect()
    }

    fn illegal(&self, attempted: &'static str) -> KvCarryError<Key> {
        KvCarryError::IllegalEvent {
            attempted,
            phase: self.phase.name(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateCommand<Key> {
    Adopt { input: Key },
    Swap { input: Key, output: Key },
    Park { input: Key, output: Key },
    Drain { input: Key, output: Key },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateEvent<Key> {
    Select { pairs: Vec<(Key, Key)> },
    Resident { input: Key },
    StepFinished { ready: Vec<Key> },
    Parked,
    Ended,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum StateCarryError<Key: Debug> {
    #[error("state carry: the carried pair set cannot change while {live} inputs are live")]
    PairsChangedWhileLive { live: usize },
    #[error("state carry: input {input:?} is not a selected pair")]
    UnknownInput { input: Key },
}

/// Recurrent state whose output buffer becomes the next call's input buffer: no rows, no growth, only which inputs are live and which swaps a finished step owes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateSwap<Key: Ord + Copy + Debug> {
    pairs: BTreeMap<Key, Key>,
    live: BTreeSet<Key>,
}

impl<Key: Ord + Copy + Debug> Default for StateSwap<Key> {
    fn default() -> Self {
        Self {
            pairs: BTreeMap::new(),
            live: BTreeSet::new(),
        }
    }
}

impl<Key: Ord + Copy + Debug> StateSwap<Key> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn selects(&self, pairs: &[(Key, Key)]) -> bool {
        pairs.len() == self.pairs.len()
            && pairs.iter().all(|(input, output)| self.pairs.get(input) == Some(output))
    }

    #[must_use]
    pub fn has_live(&self) -> bool {
        !self.live.is_empty()
    }

    #[must_use]
    pub fn is_live(&self, input: &Key) -> bool {
        self.live.contains(input)
    }

    #[must_use]
    pub fn is_pair(&self, input: &Key) -> bool {
        self.pairs.contains_key(input)
    }

    pub fn pairs(&self) -> impl Iterator<Item = (&Key, &Key)> {
        self.pairs.iter()
    }

    pub fn park(&mut self) -> Vec<StateCommand<Key>> {
        self.release(|input, output| StateCommand::Park { input, output })
    }

    pub fn step(&mut self, event: StateEvent<Key>) -> Result<Vec<StateCommand<Key>>, StateCarryError<Key>> {
        match event {
            StateEvent::Select { pairs } => self.select(pairs),
            StateEvent::Resident { input } => self.make_live(input),
            StateEvent::StepFinished { ready } => Ok(self.finish(&ready)),
            StateEvent::Parked => Ok(self.park()),
            StateEvent::Ended => Ok(self.release(|input, output| StateCommand::Drain { input, output })),
        }
    }

    fn select(&mut self, pairs: Vec<(Key, Key)>) -> Result<Vec<StateCommand<Key>>, StateCarryError<Key>> {
        if !self.selects(&pairs) {
            if !self.live.is_empty() {
                return Err(StateCarryError::PairsChangedWhileLive { live: self.live.len() });
            }
            self.pairs = pairs.into_iter().collect();
        }
        Ok(self
            .pairs
            .keys()
            .filter(|input| !self.live.contains(*input))
            .map(|input| StateCommand::Adopt { input: *input })
            .collect())
    }

    fn make_live(&mut self, input: Key) -> Result<Vec<StateCommand<Key>>, StateCarryError<Key>> {
        if !self.pairs.contains_key(&input) {
            return Err(StateCarryError::UnknownInput { input });
        }
        self.live.insert(input);
        Ok(Vec::new())
    }

    fn finish(&mut self, ready: &[Key]) -> Vec<StateCommand<Key>> {
        let mut commands = Vec::new();
        for input in ready {
            if let Some(output) = self.pairs.get(input) {
                self.live.insert(*input);
                commands.push(StateCommand::Swap { input: *input, output: *output });
            }
        }
        commands
    }

    fn release(&mut self, command: impl Fn(Key, Key) -> StateCommand<Key>) -> Vec<StateCommand<Key>> {
        let live = core::mem::take(&mut self.live);
        live.iter()
            .filter_map(|input| self.pairs.get(input).map(|output| command(*input, *output)))
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    extern crate std;

    use std::vec;
    use std::vec::Vec;

    use proptest::prelude::*;

    use super::{
        Appended, Extent, KvCarry, KvCarryError, KvCommand, KvEvent, RowPair,
    };

    const INPUT: u32 = 1;
    const OUTPUT: u32 = 101;
    const WIDTH: u64 = 4;

    fn pair() -> RowPair<u32> {
        RowPair { input: INPUT, output: OUTPUT, row_width: WIDTH }
    }

    fn selected() -> KvCarry<u32> {
        let mut carry = KvCarry::new();
        let adopt = carry.step(KvEvent::Select { pairs: vec![pair()] }).expect("select");
        assert_eq!(adopt, vec![KvCommand::Adopt { input: INPUT }]);
        carry
    }

    fn resident(valid_rows: u64, capacity_elements: u64) -> KvCarry<u32> {
        let mut carry = selected();
        carry
            .step(KvEvent::Resident { input: INPUT, valid_rows, capacity_elements })
            .expect("resident");
        carry
    }

    fn begin(carry: &mut KvCarry<u32>, cached_rows: u64, extent_elements: u64) -> Vec<KvCommand<u32>> {
        let extents = vec![Extent { input: INPUT, cached_rows: Some(cached_rows), extent_elements }];
        carry.step(KvEvent::StepBegan { extents }).expect("begin")
    }

    fn append(carry: &mut KvCarry<u32>, new_rows: u64) -> Vec<KvCommand<u32>> {
        let appended = vec![Appended { input: INPUT, new_rows }];
        carry.step(KvEvent::RowsAppended { appended }).expect("append")
    }

    #[proxima::test]
    #[case::append_one_accept_all(10, 64, 1, 1, None, None)]
    #[case::draft_four_accept_one(10, 64, 4, 1, None, Some(11))]
    #[case::draft_four_accept_none(10, 64, 4, 0, None, Some(10))]
    #[case::draft_four_accept_all(10, 64, 4, 4, None, None)]
    #[case::overflow_grows_to_power_of_two(10, 48, 4, 4, Some(64), None)]
    #[case::overflow_then_partial_accept(14, 56, 4, 2, Some(128), Some(16))]
    async fn decision_table(
        #[case] cached_rows: u64,
        #[case] capacity: u64,
        #[case] new_rows: u64,
        #[case] kept: u64,
        #[case] grow_to: Option<u64>,
        #[case] truncate_to: Option<u64>,
    ) {
        let mut carry = resident(cached_rows, capacity);
        assert!(begin(&mut carry, cached_rows, capacity).is_empty());
        let offset = cached_rows * WIDTH;
        let mut expected = Vec::new();
        if let Some(to_elements) = grow_to {
            expected.push(KvCommand::Grow { input: INPUT, to_elements, keep_elements: offset });
        }
        expected.push(KvCommand::AppendCopy {
            input: INPUT,
            output: OUTPUT,
            offset_elements: offset,
            count_elements: new_rows * WIDTH,
        });

        let appended = append(&mut carry, new_rows);
        let settled = carry.step(KvEvent::Accepted { kept }).expect("accept");

        assert_eq!(appended, expected);
        let expected_settled: Vec<_> = truncate_to
            .map(|to_rows| KvCommand::Truncate { input: INPUT, to_rows })
            .into_iter()
            .collect();
        assert_eq!(settled, expected_settled);
        assert_eq!(carry.valid_rows(&INPUT), Some(cached_rows + kept));
    }

    #[proxima::test]
    #[case::end_drains_valid_rows(true, KvCommand::Drain { input: INPUT, valid_rows: 7, row_width: WIDTH })]
    #[case::park_hands_over_valid_rows(false, KvCommand::Park { input: INPUT, valid_rows: 7 })]
    async fn release_empties_the_live_set(#[case] ended: bool, #[case] expected: KvCommand<u32>) {
        let mut carry = resident(7, 64);

        let event = if ended { KvEvent::Ended } else { KvEvent::Parked };
        let commands = carry.step(event).expect("release");

        assert_eq!(commands, vec![expected]);
        assert!(!carry.has_live());
        assert!(carry.is_pair(&INPUT));
    }

    #[proxima::test]
    async fn selecting_the_same_pairs_again_is_silent_while_live() {
        let mut carry = resident(3, 64);

        let again = carry.step(KvEvent::Select { pairs: vec![pair()] }).expect("same pairs");

        assert!(again.is_empty());
        assert!(carry.is_live(&INPUT));
    }

    #[proxima::test]
    async fn selecting_the_same_pairs_after_a_park_adopts_again() {
        let mut carry = resident(3, 64);
        carry.step(KvEvent::Parked).expect("park");

        let again = carry.step(KvEvent::Select { pairs: vec![pair()] }).expect("same pairs");

        assert_eq!(again, vec![KvCommand::Adopt { input: INPUT }]);
    }

    #[proxima::test]
    async fn changing_pairs_while_live_is_rejected() {
        let mut carry = resident(3, 64);
        let other = RowPair { input: 2, output: 102, row_width: WIDTH };

        let error = carry.step(KvEvent::Select { pairs: vec![other] }).expect_err("live set blocks");

        assert_eq!(error, KvCarryError::PairsChangedWhileLive { live: 1 });
    }

    #[proxima::test]
    async fn a_step_that_disagrees_with_valid_rows_is_rejected() {
        let mut carry = resident(10, 64);
        let extents = vec![Extent { input: INPUT, cached_rows: Some(9), extent_elements: 64 }];

        let error = carry.step(KvEvent::StepBegan { extents }).expect_err("desync");

        assert_eq!(
            error,
            KvCarryError::RowsDesynchronized { input: INPUT, cached_rows: 9, valid_rows: 10 }
        );
    }

    #[proxima::test]
    async fn valid_rows_beyond_capacity_overflow() {
        let mut carry = resident(20, 64);
        let extents = vec![Extent { input: INPUT, cached_rows: Some(20), extent_elements: 128 }];

        let error = carry.step(KvEvent::StepBegan { extents }).expect_err("overflow");

        assert_eq!(
            error,
            KvCarryError::Overflow { input: INPUT, valid_elements: 80, capacity_elements: 64 }
        );
    }

    #[proxima::test]
    async fn the_plan_extent_growing_past_capacity_grows_and_keeps_valid_rows() {
        let mut carry = resident(10, 64);

        let commands = begin(&mut carry, 10, 256);

        assert_eq!(
            commands,
            vec![KvCommand::Grow { input: INPUT, to_elements: 256, keep_elements: 40 }]
        );
    }

    #[proxima::test]
    async fn rows_appended_before_a_step_began_are_illegal() {
        let mut carry = resident(10, 64);

        let error = carry
            .step(KvEvent::RowsAppended { appended: vec![Appended { input: INPUT, new_rows: 1 }] })
            .expect_err("append needs a begun step");

        assert_eq!(error, KvCarryError::IllegalEvent { attempted: "rows_appended", phase: "idle" });
    }

    #[proxima::test]
    async fn accepting_twice_is_illegal() {
        let mut carry = resident(10, 64);
        begin(&mut carry, 10, 64);
        append(&mut carry, 2);
        carry.step(KvEvent::Accepted { kept: 2 }).expect("first accept");

        let error = carry.step(KvEvent::Accepted { kept: 2 }).expect_err("second accept");

        assert_eq!(error, KvCarryError::IllegalEvent { attempted: "accepted", phase: "idle" });
    }

    #[proxima::test]
    async fn keeping_more_rows_than_were_appended_is_rejected() {
        let mut carry = resident(10, 64);
        begin(&mut carry, 10, 64);
        append(&mut carry, 2);

        let error = carry.step(KvEvent::Accepted { kept: 3 }).expect_err("kept > appended");

        assert_eq!(error, KvCarryError::KeptBeyondAppended { input: INPUT, kept: 3, appended: 2 });
    }

    #[proxima::test]
    async fn an_unselected_input_cannot_become_resident() {
        let mut carry = selected();

        let error = carry
            .step(KvEvent::Resident { input: 9, valid_rows: 1, capacity_elements: 8 })
            .expect_err("unknown input");

        assert_eq!(error, KvCarryError::UnknownInput { input: 9 });
    }

    #[derive(Debug, Clone)]
    struct Step {
        new_rows: u64,
        kept: Option<u64>,
    }

    fn step_strategy() -> impl Strategy<Value = Step> {
        (1u64..9, proptest::option::of(0u64..9))
            .prop_map(|(new_rows, kept)| Step { new_rows, kept: kept.map(|rows| rows.min(new_rows)) })
    }

    fn execute(device: &mut Vec<u64>, commands: &[KvCommand<u32>], produced: &[u64]) {
        for command in commands {
            match *command {
                KvCommand::Grow { to_elements, keep_elements, .. } => {
                    let mut grown = vec![0; to_elements as usize];
                    grown[..keep_elements as usize].copy_from_slice(&device[..keep_elements as usize]);
                    *device = grown;
                }
                KvCommand::AppendCopy { offset_elements, count_elements, .. } => {
                    let range = offset_elements as usize..(offset_elements + count_elements) as usize;
                    device[range].copy_from_slice(&produced[..count_elements as usize]);
                }
                _ => {}
            }
        }
    }

    proptest! {
        #[test]
        fn valid_rows_always_equal_committed_tokens_and_device_bytes(
            steps in proptest::collection::vec(step_strategy(), 1..40)
        ) {
            let mut committed: Vec<u64> = (0..5).collect();
            let mut device = vec![0u64; 32];
            for (row, token) in committed.iter().enumerate() {
                device[row * WIDTH as usize..(row + 1) * WIDTH as usize].fill(*token);
            }
            let mut carry = resident(committed.len() as u64, 32);
            let mut next_token = 100u64;
            for step in steps {
                let cached = committed.len() as u64;
                let drafted: Vec<u64> = (next_token..next_token + step.new_rows).collect();
                next_token += step.new_rows;
                let produced: Vec<u64> = drafted.iter().flat_map(|token| [*token; WIDTH as usize]).collect();
                let grow = begin(&mut carry, cached, 32);
                execute(&mut device, &grow, &produced);
                let placed = append(&mut carry, step.new_rows);
                execute(&mut device, &placed, &produced);
                let kept = step.kept.unwrap_or(step.new_rows);
                if step.kept.is_some() {
                    carry.step(KvEvent::Accepted { kept }).expect("accept");
                }
                committed.extend(&drafted[..kept as usize]);
                prop_assert_eq!(carry.valid_rows(&INPUT), Some(committed.len() as u64));
                for (row, token) in committed.iter().enumerate() {
                    let cells = &device[row * WIDTH as usize..(row + 1) * WIDTH as usize];
                    prop_assert!(cells.iter().all(|cell| cell == token));
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod state_swap_tests {
    extern crate std;

    use std::vec;

    use super::{StateCarryError, StateCommand, StateEvent, StateSwap};

    const STATE: u32 = 3;
    const NEXT_STATE: u32 = 203;

    fn swapping() -> StateSwap<u32> {
        let mut carry = StateSwap::new();
        let adopt = carry
            .step(StateEvent::Select { pairs: vec![(STATE, NEXT_STATE)] })
            .expect("select");
        assert_eq!(adopt, vec![StateCommand::Adopt { input: STATE }]);
        carry
    }

    #[proxima::test]
    async fn a_finished_step_swaps_the_ready_pair_and_makes_it_live() {
        let mut carry = swapping();

        let commands = carry.step(StateEvent::StepFinished { ready: vec![STATE] }).expect("finish");

        assert_eq!(commands, vec![StateCommand::Swap { input: STATE, output: NEXT_STATE }]);
        assert!(carry.is_live(&STATE));
    }

    #[proxima::test]
    async fn a_finished_step_without_the_pair_ready_swaps_nothing() {
        let mut carry = swapping();

        let commands = carry.step(StateEvent::StepFinished { ready: vec![] }).expect("finish");

        assert!(commands.is_empty());
        assert!(!carry.is_live(&STATE));
    }

    #[proxima::test]
    #[case::park_hands_the_output_name_over(false, StateCommand::Park { input: STATE, output: NEXT_STATE })]
    #[case::end_drains_by_output(true, StateCommand::Drain { input: STATE, output: NEXT_STATE })]
    async fn release_empties_the_live_set(#[case] ended: bool, #[case] expected: StateCommand<u32>) {
        let mut carry = swapping();
        carry.step(StateEvent::StepFinished { ready: vec![STATE] }).expect("finish");

        let event = if ended { StateEvent::Ended } else { StateEvent::Parked };
        let commands = carry.step(event).expect("release");

        assert_eq!(commands, vec![expected]);
        assert!(!carry.has_live());
    }

    #[proxima::test]
    async fn selecting_the_same_pairs_after_a_park_adopts_again() {
        let mut carry = swapping();
        carry.step(StateEvent::StepFinished { ready: vec![STATE] }).expect("finish");
        carry.step(StateEvent::Parked).expect("park");

        let again = carry
            .step(StateEvent::Select { pairs: vec![(STATE, NEXT_STATE)] })
            .expect("same pairs");

        assert_eq!(again, vec![StateCommand::Adopt { input: STATE }]);
    }

    #[proxima::test]
    async fn changing_pairs_while_live_is_rejected() {
        let mut carry = swapping();
        carry.step(StateEvent::StepFinished { ready: vec![STATE] }).expect("finish");

        let error = carry
            .step(StateEvent::Select { pairs: vec![(4, 204)] })
            .expect_err("live state blocks");

        assert_eq!(error, StateCarryError::PairsChangedWhileLive { live: 1 });
    }

    #[proxima::test]
    async fn an_unselected_input_cannot_become_resident() {
        let mut carry = swapping();

        let error = carry.step(StateEvent::Resident { input: 9 }).expect_err("unknown input");

        assert_eq!(error, StateCarryError::UnknownInput { input: 9 });
    }
}
