use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use thiserror::Error;

use super::dominators::Dominators;
use super::{Block, BlockId, Function, Inst, SlotId, Terminator, Value};

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VerifyError {
    #[error("block {block:?} has no terminator")]
    MissingTerminator { block: BlockId },
    #[error("value {value:?} used in block {block:?} is never defined")]
    UndefinedValue { value: Value, block: BlockId },
    #[error("value {value:?} used in block {block:?} is not dominated by its definition")]
    NotDominated { value: Value, block: BlockId },
    #[error("value {value:?} is defined more than once")]
    Redefined { value: Value },
    #[error("jump to unknown block {block:?}")]
    UnknownBlock { block: BlockId },
    #[error("function has {count} blocks, more than a block id can index")]
    TooManyBlocks { count: usize },
    #[error("slot address of unknown slot {slot:?}")]
    UnknownSlot { slot: SlotId },
}

/// where a value is defined: its block index and a position in that block. Block params sit at
/// position 0 and the instruction at index `i` sits at `i + 1`, so a use at position `p` sees a
/// definition at `d` in the same block exactly when `d < p`. A terminator sits after every
/// instruction. Values are SSA: a second definition of one id is refused as `Redefined`.
type Definitions = BTreeMap<Value, (usize, usize)>;

/// Refuses a use whose definition does not dominate it (dominators over the CFG from block 0).
/// A block unreachable from block 0 never runs, so dominance holds vacuously for its uses; they
/// are still refused when the value is defined nowhere in the function.
#[must_use = "a verify result must be checked"]
pub fn verify(function: &Function) -> Result<(), VerifyError> {
    check_structure(function)?;
    let definitions = definitions(function)?;
    let dominators = Dominators::new(function);
    function
        .blocks
        .iter()
        .enumerate()
        .try_for_each(|(index, block)| {
            check_block_uses(
                &definitions,
                &dominators,
                block,
                index,
                block_id(function, index)?,
            )
        })
}

fn block_id(function: &Function, index: usize) -> Result<BlockId, VerifyError> {
    u32::try_from(index)
        .map(BlockId)
        .map_err(|_| VerifyError::TooManyBlocks {
            count: function.blocks.len(),
        })
}

fn check_structure(function: &Function) -> Result<(), VerifyError> {
    for (index, block) in function.blocks.iter().enumerate() {
        let id = block_id(function, index)?;
        for (_, inst) in &block.insts {
            if let Inst::SlotAddr(slot) = inst {
                check_slot(function, *slot)?;
            }
        }
        let terminator = block
            .terminator
            .as_ref()
            .ok_or(VerifyError::MissingTerminator { block: id })?;
        check_targets(function, terminator)?;
    }
    Ok(())
}

fn definitions(function: &Function) -> Result<Definitions, VerifyError> {
    let mut defined = Definitions::new();
    for (index, block) in function.blocks.iter().enumerate() {
        let params = block.params.iter().map(|(value, _)| (*value, 0));
        let results = block
            .insts
            .iter()
            .enumerate()
            .filter_map(|(position, (result, _))| result.map(|value| (value, position + 1)));
        for (value, position) in params.chain(results) {
            if defined.insert(value, (index, position)).is_some() {
                return Err(VerifyError::Redefined { value });
            }
        }
    }
    Ok(defined)
}

fn check_block_uses(
    definitions: &Definitions,
    dominators: &Dominators,
    block: &Block,
    index: usize,
    id: BlockId,
) -> Result<(), VerifyError> {
    let check = |value: Value, position: usize| {
        check_use(definitions, dominators, value, index, position, id)
    };
    for (position, (_, inst)) in block.insts.iter().enumerate() {
        inst_uses(inst)
            .into_iter()
            .try_for_each(|used| check(used, position + 1))?;
    }
    let after = block.insts.len() + 1;
    block
        .terminator
        .as_ref()
        .map(terminator_uses)
        .unwrap_or_default()
        .into_iter()
        .try_for_each(|used| check(used, after))
}

fn check_use(
    definitions: &Definitions,
    dominators: &Dominators,
    value: Value,
    index: usize,
    position: usize,
    id: BlockId,
) -> Result<(), VerifyError> {
    let Some(&(def_block, def_position)) = definitions.get(&value) else {
        return Err(VerifyError::UndefinedValue { value, block: id });
    };
    let dominated = !dominators.reachable(index)
        || if def_block == index {
            def_position < position
        } else {
            dominators.dominates(def_block, index)
        };
    if dominated {
        Ok(())
    } else {
        Err(VerifyError::NotDominated { value, block: id })
    }
}

fn inst_uses(inst: &Inst) -> Vec<Value> {
    match inst {
        Inst::Iconst { .. } | Inst::SlotAddr(_) | Inst::DataAddr { .. } => Vec::new(),
        Inst::Iadd(lhs, rhs)
        | Inst::Isub(lhs, rhs)
        | Inst::Imul(lhs, rhs)
        | Inst::Sdiv(lhs, rhs)
        | Inst::Udiv(lhs, rhs) => vec![*lhs, *rhs],
        Inst::Icmp { a, b, .. } => vec![*a, *b],
        Inst::Sextend { value, .. } | Inst::Uextend { value, .. } | Inst::Ireduce { value, .. } => {
            vec![*value]
        }
        Inst::Load { addr, .. } => vec![*addr],
        Inst::Store { value, addr, .. } => vec![*value, *addr],
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

fn check_slot(function: &Function, slot: SlotId) -> Result<(), VerifyError> {
    if (slot.0 as usize) < function.slots.len() {
        Ok(())
    } else {
        Err(VerifyError::UnknownSlot { slot })
    }
}

fn check_target(function: &Function, target: BlockId) -> Result<(), VerifyError> {
    if (target.0 as usize) < function.blocks.len() {
        Ok(())
    } else {
        Err(VerifyError::UnknownBlock { block: target })
    }
}

fn check_targets(function: &Function, terminator: &Terminator) -> Result<(), VerifyError> {
    match terminator {
        Terminator::Return(_) | Terminator::Unreachable => Ok(()),
        Terminator::Jump { target, .. } => check_target(function, *target),
        Terminator::Branch {
            then_block,
            else_block,
            ..
        } => {
            check_target(function, *then_block)?;
            check_target(function, *else_block)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    use crate::ir::{Block, Type};

    fn ret_block(value: Value) -> Block {
        Block {
            params: vec![],
            insts: vec![],
            terminator: Some(Terminator::Return(Some(value))),
        }
    }

    fn function(blocks: Vec<Block>) -> Function {
        Function {
            name: "f".to_string(),
            params: vec![],
            returns: Some(Type::I32),
            slots: vec![],
            blocks,
        }
    }

    #[test]
    fn builds_and_verifies_ret_42() {
        let block = Block {
            params: vec![],
            insts: vec![(
                Some(Value(0)),
                Inst::Iconst {
                    ty: Type::I32,
                    imm: 42,
                },
            )],
            terminator: Some(Terminator::Return(Some(Value(0)))),
        };
        let result = verify(&function(vec![block]));
        assert_eq!(result, Ok(()));
    }

    #[test]
    fn jump_carries_block_param() {
        let entry = Block {
            params: vec![],
            insts: vec![(
                Some(Value(0)),
                Inst::Iconst {
                    ty: Type::I32,
                    imm: 42,
                },
            )],
            terminator: Some(Terminator::Jump {
                target: BlockId(1),
                args: vec![Value(0)],
            }),
        };
        let exit = Block {
            params: vec![(Value(1), Type::I32)],
            ..ret_block(Value(1))
        };
        let result = verify(&function(vec![entry, exit]));
        assert_eq!(result, Ok(()));
    }

    #[test]
    fn refuses_block_without_terminator() {
        let entry = Block {
            params: vec![],
            insts: vec![(
                Some(Value(0)),
                Inst::Iconst {
                    ty: Type::I32,
                    imm: 42,
                },
            )],
            terminator: Some(Terminator::Jump {
                target: BlockId(1),
                args: vec![],
            }),
        };
        let open = Block {
            params: vec![],
            insts: vec![],
            terminator: None,
        };
        let result = verify(&function(vec![entry, open]));
        assert_eq!(
            result,
            Err(VerifyError::MissingTerminator { block: BlockId(1) })
        );
    }

    #[test]
    fn refuses_jump_to_unknown_block() {
        let entry = Block {
            params: vec![],
            insts: vec![],
            terminator: Some(Terminator::Jump {
                target: BlockId(9),
                args: vec![],
            }),
        };
        let result = verify(&function(vec![entry]));
        assert_eq!(result, Err(VerifyError::UnknownBlock { block: BlockId(9) }));
    }

    #[test]
    fn refuses_use_of_undefined_value() {
        let result = verify(&function(vec![ret_block(Value(7))]));
        assert_eq!(
            result,
            Err(VerifyError::UndefinedValue {
                value: Value(7),
                block: BlockId(0)
            })
        );
    }

    fn flag(result: u32) -> (Option<Value>, Inst) {
        (
            Some(Value(result)),
            Inst::Iconst {
                ty: Type::I8,
                imm: 1,
            },
        )
    }

    fn jump_to_join(params: Vec<(Value, Type)>) -> Block {
        Block {
            params,
            insts: vec![],
            terminator: Some(Terminator::Jump {
                target: BlockId(3),
                args: vec![],
            }),
        }
    }

    // entry, then arm, else arm, join: the then arm does not dominate the join
    fn diamond(
        entry_insts: Vec<(Option<Value>, Inst)>,
        then_arm: Block,
        joined: Value,
    ) -> Function {
        let entry = Block {
            params: vec![],
            insts: entry_insts,
            terminator: Some(Terminator::Branch {
                cond: Value(0),
                then_block: BlockId(1),
                then_args: vec![Value(0)]
                    .into_iter()
                    .take(then_arm.params.len())
                    .collect(),
                else_block: BlockId(2),
                else_args: vec![],
            }),
        };
        function(vec![
            entry,
            then_arm,
            jump_to_join(vec![]),
            ret_block(joined),
        ])
    }

    #[test]
    fn refuses_param_used_outside_its_block() {
        let param = Value(1);
        let then_arm = jump_to_join(vec![(param, Type::I8)]);
        let refused = verify(&diamond(vec![flag(0)], then_arm.clone(), param));
        let control = verify(&diamond(vec![flag(0)], then_arm, Value(0)));
        assert_eq!(
            (refused, control),
            (
                Err(VerifyError::NotDominated {
                    value: param,
                    block: BlockId(3)
                }),
                Ok(())
            )
        );
    }

    #[test]
    fn refuses_use_not_dominated() {
        let five = (
            Some(Value(1)),
            Inst::Iconst {
                ty: Type::I32,
                imm: 5,
            },
        );
        let then_arm = Block {
            insts: vec![five.clone()],
            ..jump_to_join(vec![])
        };
        let refused = verify(&diamond(vec![flag(0)], then_arm.clone(), Value(1)));
        let in_entry = Block {
            insts: vec![],
            ..then_arm
        };
        let control = verify(&diamond(vec![flag(0), five], in_entry, Value(1)));
        assert_eq!(
            (refused, control),
            (
                Err(VerifyError::NotDominated {
                    value: Value(1),
                    block: BlockId(3)
                }),
                Ok(())
            )
        );
    }

    #[test]
    fn refuses_redefined_value() {
        let five = (
            Some(Value(1)),
            Inst::Iconst {
                ty: Type::I32,
                imm: 5,
            },
        );
        let sibling = || Block {
            insts: vec![five.clone()],
            ..jump_to_join(vec![])
        };
        let mut redefining = diamond(vec![flag(0)], sibling(), Value(0));
        redefining.blocks[2] = sibling();
        let result = verify(&redefining);
        assert_eq!(result, Err(VerifyError::Redefined { value: Value(1) }));
    }

    #[test]
    fn accepts_loop_carried_use() {
        let entry = Block {
            params: vec![],
            insts: vec![(
                Some(Value(0)),
                Inst::Iconst {
                    ty: Type::I32,
                    imm: 0,
                },
            )],
            terminator: Some(Terminator::Jump {
                target: BlockId(1),
                args: vec![Value(0)],
            }),
        };
        let header = Block {
            params: vec![(Value(1), Type::I32)],
            insts: vec![flag(2)],
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
            insts: vec![(Some(Value(3)), Inst::Iadd(Value(1), Value(2)))],
            terminator: Some(Terminator::Jump {
                target: BlockId(1),
                args: vec![Value(3)],
            }),
        };
        let result = verify(&function(vec![entry, header, body, ret_block(Value(1))]));
        assert_eq!(result, Ok(()));
    }
}
