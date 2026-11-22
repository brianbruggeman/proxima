//! Assign aarch64 registers to nova values by liveness.
//!
//! [`allocate`] numbers every block entry, instruction and terminator in layout order, takes each
//! value's live range from its definition to its last use, stretches ranges over loops, then
//! scans the ranges in start order handing out x0..x15. x16 stays free as the move scratch.
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::ops::Range;
use thiserror::Error;

use super::Reg;
use crate::ir::{Function, Inst, Terminator, Value};

const POOL: u8 = 16;
const MAX_ENTRY_PARAMS: usize = 8;
/// first callee-saved register the allocator hands out (AAPCS64 x19..x28)
pub const CALLEE_SAVED_FIRST: u8 = 19;
const CALLEE_SAVED_COUNT: usize = 10;

/// Why registers were not assigned.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RegAllocError {
    #[error("{live} values are live at once, only {POOL} registers (x0..x15) are assigned")]
    TooManyLive { live: usize },
    #[error(
        "{live} values are live across a call, only {CALLEE_SAVED_COUNT} callee-saved registers (x19..x28) are assigned"
    )]
    TooManyLiveAcrossCall { live: usize },
    #[error("entry block has {params} params, only {MAX_ENTRY_PARAMS} arrive in x0..x7")]
    TooManyParams { params: usize },
    #[error("value {value:?} has no register")]
    Unassigned { value: Value },
}

/// Live ranges (`start..end` in layout positions, end exclusive) and the register per value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Allocation {
    pub ranges: BTreeMap<Value, Range<u32>>,
    pub regs: BTreeMap<Value, Reg>,
}

impl Allocation {
    #[must_use = "the register or the refusal must be handled"]
    pub fn reg(&self, value: Value) -> Result<Reg, RegAllocError> {
        self.regs
            .get(&value)
            .copied()
            .ok_or(RegAllocError::Unassigned { value })
    }

    /// The callee-saved registers (x19..x28) this assignment uses, ascending.
    #[must_use]
    pub fn callee_saved(&self) -> Vec<Reg> {
        let mut used: Vec<Reg> = self
            .regs
            .values()
            .copied()
            .filter(|reg| is_callee_saved(*reg))
            .collect();
        used.sort_by_key(|reg| reg.0);
        used.dedup();
        used
    }
}

fn is_callee_saved(reg: Reg) -> bool {
    (CALLEE_SAVED_FIRST..CALLEE_SAVED_FIRST + CALLEE_SAVED_COUNT as u8).contains(&reg.0)
}

// layout positions of the `Call` instructions, numbered as `ranges_of` numbers them
fn call_positions(function: &Function) -> Vec<u32> {
    let mut position = 0u32;
    let mut calls = Vec::new();
    for block in &function.blocks {
        position += 1;
        for (_, inst) in &block.insts {
            if matches!(inst, Inst::Call { .. }) {
                calls.push(position);
            }
            position += 1;
        }
        position += 1;
    }
    calls
}

fn inst_uses(inst: &Inst) -> Vec<Value> {
    match inst {
        Inst::Iconst { .. } | Inst::SlotAddr(_) | Inst::DataAddr { .. } => Vec::new(),
        Inst::Iadd(lhs, rhs)
        | Inst::Isub(lhs, rhs)
        | Inst::Imul(lhs, rhs)
        | Inst::Sdiv(lhs, rhs)
        | Inst::Udiv(lhs, rhs) => {
            alloc::vec![*lhs, *rhs]
        }
        Inst::Icmp { a, b, .. } => alloc::vec![*a, *b],
        Inst::Sextend { value, .. } | Inst::Uextend { value, .. } | Inst::Ireduce { value, .. } => {
            alloc::vec![*value]
        }
        Inst::Load { addr, .. } => alloc::vec![*addr],
        Inst::Store { value, addr, .. } => alloc::vec![*value, *addr],
        Inst::Call { args, .. } => args.clone(),
    }
}

fn terminator_uses(terminator: &Terminator) -> Vec<Value> {
    match terminator {
        Terminator::Return(value) => value.iter().copied().collect(),
        Terminator::Unreachable => Vec::new(),
        Terminator::Jump { args, .. } => args.clone(),
        Terminator::Branch {
            cond,
            then_args,
            else_args,
            ..
        } => core::iter::once(*cond)
            .chain(then_args.iter().copied())
            .chain(else_args.iter().copied())
            .collect(),
    }
}

fn terminator_targets(terminator: &Terminator) -> Vec<usize> {
    match terminator {
        Terminator::Return(_) | Terminator::Unreachable => Vec::new(),
        Terminator::Jump { target, .. } => alloc::vec![target.0 as usize],
        Terminator::Branch {
            then_block,
            else_block,
            ..
        } => alloc::vec![then_block.0 as usize, else_block.0 as usize],
    }
}

fn define(ranges: &mut BTreeMap<Value, Range<u32>>, value: Value, position: u32) {
    ranges.insert(value, position..position + 1);
}

fn touch(ranges: &mut BTreeMap<Value, Range<u32>>, value: Value, position: u32) {
    if let Some(range) = ranges.get_mut(&value) {
        range.end = range.end.max(position);
    }
}

// a range is read as [start, end): a value read at position p ends at p, so a result defined at
// p may reuse the register of an operand that dies at p. a value never read lives one position.
// a back edge from the terminator at `term` to the header entered at `header` keeps every value
// that was defined before the header and is still read inside the loop alive through `term`.
fn ranges_of(function: &Function, loop_carried: bool) -> BTreeMap<Value, Range<u32>> {
    let mut ranges = BTreeMap::new();
    let mut entries = Vec::with_capacity(function.blocks.len());
    let mut back_edges = Vec::new();
    let mut position = 0u32;
    for (index, block) in function.blocks.iter().enumerate() {
        entries.push(position);
        for (value, _) in &block.params {
            define(&mut ranges, *value, position);
        }
        position += 1;
        for (result, inst) in &block.insts {
            for used in inst_uses(inst) {
                touch(&mut ranges, used, position);
            }
            if let Some(value) = result {
                define(&mut ranges, *value, position);
            }
            position += 1;
        }
        if let Some(terminator) = &block.terminator {
            for used in terminator_uses(terminator) {
                touch(&mut ranges, used, position);
            }
            back_edges.extend(
                terminator_targets(terminator)
                    .into_iter()
                    .filter(|target| *target <= index)
                    .map(|target| (target, position)),
            );
        }
        position += 1;
    }
    if loop_carried {
        extend_over_loops(&mut ranges, &entries, &back_edges);
    }
    ranges
}

fn extend_over_loops(
    ranges: &mut BTreeMap<Value, Range<u32>>,
    entries: &[u32],
    back_edges: &[(usize, u32)],
) {
    let mut changed = true;
    while changed {
        changed = false;
        for (target, term) in back_edges {
            let header = entries[*target];
            for range in ranges.values_mut() {
                if range.start < header && range.end > header && range.end <= *term {
                    range.end = *term + 1;
                    changed = true;
                }
            }
        }
    }
}

/// Live ranges of every value, loop-carried values stretched over their loop.
#[must_use]
pub fn live_ranges(function: &Function) -> BTreeMap<Value, Range<u32>> {
    ranges_of(function, true)
}

// linear scan in start order; entry params are precolored to x0..x7 and all start at position 0,
// so they are placed before anything else
fn assign(
    function: &Function,
    ranges: BTreeMap<Value, Range<u32>>,
) -> Result<Allocation, RegAllocError> {
    let params = function
        .blocks
        .first()
        .map_or(&[][..], |block| &block.params);
    if params.len() > MAX_ENTRY_PARAMS {
        return Err(RegAllocError::TooManyParams {
            params: params.len(),
        });
    }
    let calls = call_positions(function);
    // a value read after a call that began before it must survive the call
    let across = |range: &Range<u32>| calls.iter().any(|at| range.start < *at && range.end > *at);
    let precolored: BTreeMap<Value, Reg> = params
        .iter()
        .zip(0u8..)
        .filter(|((value, _), _)| ranges.get(value).is_none_or(|range| !across(range)))
        .map(|((value, _), number)| (*value, Reg(number)))
        .collect();
    let mut order: Vec<(&Value, &Range<u32>)> = ranges.iter().collect();
    order.sort_by_key(|(value, range)| (range.start, **value));
    let mut active: Vec<(u32, Reg)> = Vec::new();
    let mut regs = BTreeMap::new();
    for (value, range) in order {
        active.retain(|(end, _)| *end > range.start);
        let free = |candidate: &Reg| active.iter().all(|(_, used)| used != candidate);
        let reg = match precolored.get(value) {
            Some(reg) => *reg,
            None if across(range) => (CALLEE_SAVED_FIRST
                ..CALLEE_SAVED_FIRST + CALLEE_SAVED_COUNT as u8)
                .map(Reg)
                .find(free)
                .ok_or_else(|| RegAllocError::TooManyLiveAcrossCall {
                    live: active
                        .iter()
                        .filter(|(_, used)| is_callee_saved(*used))
                        .count()
                        + 1,
                })?,
            None => (0..POOL)
                .map(Reg)
                .find(free)
                .ok_or(RegAllocError::TooManyLive {
                    live: active.len() + 1,
                })?,
        };
        active.push((range.end, reg));
        regs.insert(*value, reg);
    }
    Ok(Allocation { ranges, regs })
}

/// Assign a register to every value of `function`. Lowering consumes the result.
#[must_use = "the assignment or the refusal must be handled"]
pub fn allocate(function: &Function) -> Result<Allocation, RegAllocError> {
    assign(function, live_ranges(function))
}

#[cfg(test)]
mod tests {
    extern crate std;

    use alloc::string::ToString;
    use alloc::vec;

    use super::*;
    use crate::ir::{Block, BlockId, Slot, SlotId, Type, verify};

    const CHAIN_LINKS: u32 = 39;
    const CHAIN_REGISTERS: usize = 3;
    const X9: Reg = Reg(9);

    fn function_of(blocks: Vec<Block>) -> Function {
        Function {
            name: "f".to_string(),
            params: vec![],
            returns: Some(Type::I32),
            slots: vec![],
            blocks,
        }
    }

    fn iconst(result: u32, imm: i64) -> (Option<Value>, Inst) {
        (Some(Value(result)), Inst::Iconst { ty: Type::I32, imm })
    }

    fn straight(insts: Vec<(Option<Value>, Inst)>, result: u32) -> Function {
        function_of(vec![Block {
            params: vec![],
            insts,
            terminator: Some(Terminator::Return(Some(Value(result)))),
        }])
    }

    fn chain() -> Function {
        let mut insts = vec![iconst(0, 1)];
        let mut next = 1;
        for _ in 0..CHAIN_LINKS {
            insts.push(iconst(next, 3));
            insts.push((
                Some(Value(next + 1)),
                Inst::Iadd(Value(next - 1), Value(next)),
            ));
            next += 2;
        }
        straight(insts, next - 1)
    }

    fn conflicts(ranges: &BTreeMap<Value, Range<u32>>, regs: &BTreeMap<Value, Reg>) -> usize {
        let entries: Vec<_> = ranges.iter().collect();
        entries
            .iter()
            .enumerate()
            .flat_map(|(index, left)| entries[index + 1..].iter().map(move |right| (left, right)))
            .filter(|((left, a), (right, b))| {
                regs[*left] == regs[*right] && a.start < b.end && b.start < a.end
            })
            .count()
    }

    #[test]
    fn shares_registers_across_dead_ranges() {
        let function = chain();
        let allocation = allocate(&function).expect("a 79-value chain needs few registers");
        let distinct: alloc::collections::BTreeSet<u8> =
            allocation.regs.values().map(|reg| reg.0).collect();
        let found = conflicts(&allocation.ranges, &allocation.regs);
        std::println!(
            "values={} registers<={CHAIN_REGISTERS} conflicts={found}",
            allocation.regs.len()
        );
        assert_eq!(
            (allocation.regs.len(), found),
            (79, 0),
            "79 values, no overlap"
        );
        assert!(distinct.len() <= CHAIN_REGISTERS, "registers: {distinct:?}");
        let all_x9: BTreeMap<Value, Reg> =
            allocation.regs.keys().map(|value| (*value, X9)).collect();
        let control = conflicts(&allocation.ranges, &all_x9);
        std::println!("control conflicts>=1 counted={control}");
        assert!(control >= 1, "mapping everything to x9 must conflict");
    }

    const MANY_SLOTS: u32 = 30;
    const MANY_SLOTS_REGISTERS: usize = 4;

    // store k into slot k, then load every slot into a running sum; `hoist` defines every
    // address once at entry instead of taking it again at each store and load
    fn many_slots(hoist: bool) -> Function {
        let mut next = 0;
        let mut fresh = || {
            next += 1;
            Value(next - 1)
        };
        let mut insts = Vec::new();
        let hoisted: Vec<Value> = (0..MANY_SLOTS)
            .filter(|_| hoist)
            .map(|slot| {
                let addr = fresh();
                insts.push((Some(addr), Inst::SlotAddr(SlotId(slot))));
                addr
            })
            .collect();
        let mut address = |slot: u32, insts: &mut Vec<(Option<Value>, Inst)>| {
            hoisted.get(slot as usize).copied().unwrap_or_else(|| {
                let addr = fresh();
                insts.push((Some(addr), Inst::SlotAddr(SlotId(slot))));
                addr
            })
        };
        for slot in 0..MANY_SLOTS {
            let value = Value(1000 + slot);
            insts.push((
                Some(value),
                Inst::Iconst {
                    ty: Type::I32,
                    imm: i64::from(slot + 1),
                },
            ));
            let addr = address(slot, &mut insts);
            insts.push((
                None,
                Inst::Store {
                    value,
                    addr,
                    offset: 0,
                },
            ));
        }
        let mut sum: Option<Value> = None;
        for slot in 0..MANY_SLOTS {
            let addr = address(slot, &mut insts);
            let loaded = Value(2000 + slot);
            insts.push((
                Some(loaded),
                Inst::Load {
                    ty: Type::I32,
                    addr,
                    offset: 0,
                },
            ));
            sum = Some(match sum {
                None => loaded,
                Some(previous) => {
                    let total = Value(3000 + slot);
                    insts.push((Some(total), Inst::Iadd(previous, loaded)));
                    total
                }
            });
        }
        let mut function = straight(insts, sum.map_or(0, |value| value.0));
        function.slots = vec![Slot { size: 4, align: 4 }; MANY_SLOTS as usize];
        function
    }

    #[test]
    fn slot_addresses_stay_short_lived() {
        let function = many_slots(false);
        verify(&function).expect("the slot function is valid ir");
        let allocation = allocate(&function).expect("30 short-lived addresses need few registers");
        let distinct: alloc::collections::BTreeSet<u8> =
            allocation.regs.values().map(|reg| reg.0).collect();
        let found = conflicts(&allocation.ranges, &allocation.regs);
        std::println!("slots={MANY_SLOTS} registers<={MANY_SLOTS_REGISTERS} conflicts={found}");
        assert_eq!(found, 0, "no overlapping ranges share a register");
        assert!(
            distinct.len() <= MANY_SLOTS_REGISTERS,
            "registers: {distinct:?}"
        );
        let hoisted = allocate(&many_slots(true)).expect_err("30 hoisted addresses exceed x0..x15");
        std::println!("control refusal: {hoisted}");
        assert!(
            matches!(hoisted, RegAllocError::TooManyLive { live } if live >= 17),
            "{hoisted:?}"
        );
    }

    #[test]
    fn refuses_seventeen_live_values() {
        let mut insts: Vec<_> = (0..17).map(|id| iconst(id, i64::from(id))).collect();
        insts.push((Some(Value(17)), Inst::Iadd(Value(0), Value(1))));
        for id in 2..17 {
            insts.push((Some(Value(16 + id)), Inst::Iadd(Value(15 + id), Value(id))));
        }
        let error = allocate(&straight(insts, 32)).expect_err("17 live values exceed x0..x15");
        std::println!("refusal: {error}");
        assert_eq!(error, RegAllocError::TooManyLive { live: 17 });
        assert!(error.to_string().contains("17"), "message: {error}");
    }

    // `count` constants defined before a call to `smash` and all summed after it
    fn live_across_call(count: u32) -> Function {
        let mut insts: Vec<_> = (0..count).map(|id| iconst(id, i64::from(id))).collect();
        insts.push((
            None,
            Inst::Call {
                callee: "smash".to_string(),
                args: vec![],
                ret: None,
            },
        ));
        insts.push((Some(Value(count)), Inst::Iadd(Value(0), Value(1))));
        for id in 2..count {
            insts.push((
                Some(Value(count + id - 1)),
                Inst::Iadd(Value(count + id - 2), Value(id)),
            ));
        }
        straight(insts, 2 * count - 2)
    }

    #[test]
    fn refuses_eleven_values_live_across_call() {
        let error = allocate(&live_across_call(11)).expect_err("11 exceed x19..x28");
        std::println!("refusal: {error}");
        assert_eq!(error, RegAllocError::TooManyLiveAcrossCall { live: 11 });
    }

    #[test]
    fn accepts_ten_values_live_across_call() {
        let allocation = allocate(&live_across_call(10)).expect("10 fit x19..x28");
        let held: Vec<u8> = (0..10).map(|id| allocation.regs[&Value(id)].0).collect();
        let distinct: alloc::collections::BTreeSet<u8> = held.iter().copied().collect();
        std::println!("held={held:?}");
        assert!(held.iter().all(|reg| (19..=28).contains(reg)), "{held:?}");
        assert_eq!(distinct.len(), 10, "pairwise distinct: {held:?}");
    }

    fn loop_function() -> Function {
        let entry = Block {
            params: vec![],
            insts: vec![iconst(0, 1), iconst(1, 5)],
            terminator: Some(Terminator::Jump {
                target: BlockId(1),
                args: vec![Value(1)],
            }),
        };
        let header = Block {
            params: vec![(Value(2), Type::I32)],
            insts: vec![],
            terminator: Some(Terminator::Branch {
                cond: Value(2),
                then_block: BlockId(2),
                then_args: vec![],
                else_block: BlockId(3),
                else_args: vec![],
            }),
        };
        let body = Block {
            params: vec![],
            insts: vec![
                (Some(Value(3)), Inst::Isub(Value(2), Value(0))),
                iconst(4, 2),
            ],
            terminator: Some(Terminator::Jump {
                target: BlockId(1),
                args: vec![Value(3)],
            }),
        };
        let exit = Block {
            params: vec![],
            insts: vec![],
            terminator: Some(Terminator::Return(Some(Value(2)))),
        };
        function_of(vec![entry, header, body, exit])
    }

    #[test]
    fn keeps_loop_carried_values_live() {
        let function = loop_function();
        verify(&function).expect("the loop is valid ir");
        let (counter, scratch) = (Value(0), Value(4));
        let allocation = allocate(&function).expect("loop allocates");
        let range = &allocation.ranges[&counter];
        // header is entered at position 4, the body terminator sits at position 9
        assert!(range.start < 4 && range.end > 9, "k range: {range:?}");
        assert_ne!(allocation.regs[&counter], allocation.regs[&scratch]);
        let layout_only = assign(&function, ranges_of(&function, false)).expect("layout-only");
        let control = conflicts(&allocation.ranges, &layout_only.regs);
        std::println!("loop k range={range:?} control conflicts>=1 counted={control}");
        assert!(control >= 1, "layout-only liveness must conflict");
    }
}
