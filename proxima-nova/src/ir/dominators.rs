use alloc::vec;
use alloc::vec::Vec;

use super::{Function, Terminator};

/// the blocks a terminator can transfer control to; `Return` and `Unreachable` have none
pub(super) fn successors(terminator: &Terminator) -> Vec<usize> {
    match terminator {
        Terminator::Return(_) | Terminator::Unreachable => Vec::new(),
        Terminator::Jump { target, .. } => vec![target.0 as usize],
        Terminator::Branch {
            then_block,
            else_block,
            ..
        } => vec![then_block.0 as usize, else_block.0 as usize],
    }
}

/// Immediate dominators of the blocks reachable from block 0, by the iterative algorithm of
/// Cooper, Harvey and Kennedy, "A Simple, Fast Dominance Algorithm" (2001), over a reverse
/// postorder. A block not reachable from block 0 has no immediate dominator.
pub(super) struct Dominators {
    idom: Vec<Option<usize>>,
}

impl Dominators {
    /// every branch target in `function` must be a valid block index (the verifier checks first)
    pub(super) fn new(function: &Function) -> Self {
        let count = function.blocks.len();
        let succs: Vec<Vec<usize>> = function
            .blocks
            .iter()
            .map(|block| {
                block
                    .terminator
                    .as_ref()
                    .map(successors)
                    .unwrap_or_default()
            })
            .collect();
        let order = reverse_postorder(&succs);
        let mut number = vec![usize::MAX; count];
        order
            .iter()
            .enumerate()
            .for_each(|(rank, block)| number[*block] = rank);
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); count];
        order.iter().for_each(|block| {
            succs[*block]
                .iter()
                .for_each(|next| preds[*next].push(*block))
        });
        let mut idom: Vec<Option<usize>> = vec![None; count];
        if count > 0 {
            idom[0] = Some(0);
        }
        let mut changed = true;
        while changed {
            changed = false;
            for block in order.iter().skip(1) {
                let candidate = preds[*block]
                    .iter()
                    .filter(|pred| idom[**pred].is_some())
                    .copied()
                    .try_fold(None, |acc: Option<usize>, pred| match acc {
                        None => Some(Some(pred)),
                        Some(current) => intersect(&idom, &number, current, pred).map(Some),
                    })
                    .flatten();
                if candidate != idom[*block] {
                    idom[*block] = candidate;
                    changed = true;
                }
            }
        }
        Self { idom }
    }

    pub(super) fn reachable(&self, block: usize) -> bool {
        self.idom.get(block).copied().flatten().is_some()
    }

    /// true when every path from block 0 to `block` passes through `dominator`
    pub(super) fn dominates(&self, dominator: usize, block: usize) -> bool {
        let mut current = block;
        loop {
            if current == dominator {
                return true;
            }
            match self.idom.get(current).copied().flatten() {
                Some(next) if next != current => current = next,
                _ => return false,
            }
        }
    }
}

fn intersect(idom: &[Option<usize>], number: &[usize], left: usize, right: usize) -> Option<usize> {
    let (mut a_finger, mut b_finger) = (left, right);
    while a_finger != b_finger {
        while number[a_finger] > number[b_finger] {
            a_finger = idom[a_finger]?;
        }
        while number[b_finger] > number[a_finger] {
            b_finger = idom[b_finger]?;
        }
    }
    Some(a_finger)
}

fn reverse_postorder(succs: &[Vec<usize>]) -> Vec<usize> {
    if succs.is_empty() {
        return Vec::new();
    }
    let mut seen = vec![false; succs.len()];
    let mut postorder = Vec::with_capacity(succs.len());
    let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
    seen[0] = true;
    while let Some((block, cursor)) = stack.pop() {
        match succs[block].get(cursor) {
            Some(next) => {
                stack.push((block, cursor + 1));
                if !seen[*next] {
                    seen[*next] = true;
                    stack.push((*next, 0));
                }
            }
            None => postorder.push(block),
        }
    }
    postorder.reverse();
    postorder
}
