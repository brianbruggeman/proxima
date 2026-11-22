//! Parallel-move resolution for block-arg transfers.
//!
//! A plain function over `(dest, src)` register pairs, no trait: [`resolve`] returns the same pairs
//! reordered so that running them one after another equals assigning them all at once. [`lower`]
//! turns each pair into `add rd, rn, xzr`.
//!
//! [`lower`]: super::lower
use alloc::vec::Vec;

use super::inst::Reg;

/// Register that breaks cycles. The allocator keeps it out of the pool x0..x15.
pub const SCRATCH: Reg = Reg(16);

/// Sequentialise `moves` (`(dest, src)`, dests distinct, `SCRATCH` absent) so the result, run in
/// order, realises the parallel assignment. Self-moves are dropped. A cycle is broken by parking one
/// member's old value in [`SCRATCH`] and redirecting the readers of it there.
#[must_use = "the ordered moves are the only output"]
pub fn resolve(moves: &[(Reg, Reg)]) -> Vec<(Reg, Reg)> {
    let mut pending: Vec<(Reg, Reg)> = moves
        .iter()
        .copied()
        .filter(|(dest, source)| dest != source)
        .collect();
    let mut ordered = Vec::with_capacity(pending.len() + 1);
    while let Some(&(blocked, _)) = pending.first() {
        let free = pending
            .iter()
            .position(|(dest, _)| !pending.iter().any(|(_, source)| source == dest));
        match free {
            Some(index) => ordered.push(pending.swap_remove(index)),
            None => {
                ordered.push((SCRATCH, blocked));
                pending
                    .iter_mut()
                    .filter(|(_, source)| *source == blocked)
                    .for_each(|entry| entry.1 = SCRATCH);
            }
        }
    }
    ordered
}

#[cfg(test)]
mod tests {
    extern crate std;

    use alloc::vec;

    use super::*;

    const FILE_SIZE: usize = 32;

    type Registers = [u64; FILE_SIZE];

    fn initial() -> Registers {
        core::array::from_fn(|index| 1000 + index as u64)
    }

    fn run_in_order(moves: &[(Reg, Reg)]) -> Registers {
        let mut file = initial();
        for (dest, source) in moves {
            file[usize::from(dest.0)] = file[usize::from(source.0)];
        }
        file
    }

    fn assigned_at_once(moves: &[(Reg, Reg)]) -> Registers {
        let before = initial();
        let mut file = before;
        for (dest, source) in moves {
            file[usize::from(dest.0)] = before[usize::from(source.0)];
        }
        file
    }

    fn differs(left: &Registers, right: &Registers) -> bool {
        (0..FILE_SIZE)
            .filter(|index| *index != usize::from(SCRATCH.0))
            .any(|index| left[index] != right[index])
    }

    fn cases() -> vec::Vec<(&'static str, vec::Vec<(Reg, Reg)>)> {
        let reg = Reg;
        vec![
            ("two_cycle_swap", vec![(reg(9), reg(10)), (reg(10), reg(9))]),
            (
                "three_cycle_rotation",
                vec![(reg(9), reg(10)), (reg(10), reg(11)), (reg(11), reg(9))],
            ),
            (
                "chain_a_from_b_b_from_c",
                vec![(reg(9), reg(10)), (reg(10), reg(11))],
            ),
            ("self_move", vec![(reg(9), reg(9))]),
        ]
    }

    #[test]
    fn resolves_parallel_assignments() {
        let cases = cases();
        let resolver = cases
            .iter()
            .filter(|(_, moves)| differs(&run_in_order(&resolve(moves)), &assigned_at_once(moves)))
            .count();
        let naive = cases
            .iter()
            .filter(|(_, moves)| differs(&run_in_order(moves), &assigned_at_once(moves)))
            .count();
        std::println!("resolver mismatched={resolver} of={}", cases.len());
        std::println!("naive mismatched={naive} of={}", cases.len());
        assert_eq!((resolver, cases.len()), (0, 4));
        assert_eq!((naive, cases.len()), (2, 4));
        assert!(resolve(&[(Reg(9), Reg(9))]).is_empty());
    }
}
