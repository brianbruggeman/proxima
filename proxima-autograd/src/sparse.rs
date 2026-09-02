//! Applying a [`crate::adjoint::GatheredContribution`] back onto its full
//! operand — the step [`crate::adjoint::differentiate`] deliberately stops
//! short of.
//!
//! At embedding scale (vocab 128k, 4k touched rows, dim 2048) scattering
//! through the `Op` algebra would materialize an `O(vocab x touched)` mask
//! (`proxima-tensor/src/cpu.rs:16062`'s own doc: 524M elements to place 4k
//! rows). [`dedupe_and_sum_rows`] does the same scatter-*add* — colliding
//! indices sum, they never overwrite — over already-evaluated host buffers
//! instead, which costs `O(touched x row_len)`: no vocab-sized buffer is
//! ever materialized. A caller then runs the existing
//! [`crate::optimizer::adam_step`] at a rank sized to the *unique* touched
//! rows this returns, not the full vocab — the same function, a smaller
//! instantiation, never a second optimizer.
//!
//! This is plain data reduction over caller-owned buffers, not a graph
//! transform, so it takes and returns `&[f32]`/`Vec<f32>` rather than
//! [`proxima_tensor::op::NodeId`]s — the caller has already run
//! [`proxima_tensor::cpu::evaluate_named`] on the `values`/`indices` nodes a
//! [`crate::adjoint::GatheredContribution`] names by the time this runs.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::error::AutogradError;

/// Sums every source row into its destination row, keyed by `indices`.
///
/// `indices` carries one whole-number row id per source row, `row_len`
/// wide, packed into `values` — the same convention
/// [`proxima_tensor::map::IndexMap::Computed`] uses for its own `indices`
/// operand: every buffer, this one included, is f32-backed, so a caller
/// reads `indices[n] as u32` to recover the row id. Returns the unique
/// destination ids in first-seen order and one summed row per id:
/// collisions (the same id appearing more than once in `indices`)
/// accumulate instead of overwriting, which is the entire difference
/// between a scatter-add and a scatter-write — see this crate's own report
/// for the case where the same token appears twice in one batch.
pub fn dedupe_and_sum_rows(
    indices: &[f32],
    values: &[f32],
    row_len: usize,
) -> Result<(Vec<u32>, Vec<f32>), AutogradError> {
    if row_len == 0 || values.len() != indices.len() * row_len {
        return Err(AutogradError::SparseRowLengthMismatch {
            row_len,
            found: values.len(),
        });
    }

    let mut slot_of: BTreeMap<u32, usize> = BTreeMap::new();
    let mut unique_ids: Vec<u32> = Vec::new();
    let mut summed: Vec<f32> = Vec::new();

    for (position, &raw_index) in indices.iter().enumerate() {
        let id = raw_index as u32;
        let row = &values[position * row_len..(position + 1) * row_len];
        match slot_of.get(&id) {
            Some(&slot) => {
                let destination = &mut summed[slot * row_len..(slot + 1) * row_len];
                for (accumulator, source) in destination.iter_mut().zip(row) {
                    *accumulator += *source;
                }
            }
            None => {
                slot_of.insert(id, unique_ids.len());
                unique_ids.push(id);
                summed.extend_from_slice(row);
            }
        }
    }

    Ok((unique_ids, summed))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec;

    use super::*;

    /// Same index/source pattern as
    /// `proxima-tensor/src/cpu.rs:16067`'s own
    /// `scatter_add_into_a_known_destination_via_mask_composition` test
    /// (`ids = [0, 2, 0, 1]`, `src = [10, 20, 30, 40]`), so this result is
    /// cross-checked against that dense-mask oracle: `dest[0] = 10+30 = 40`
    /// (the collision), `dest[1] = 40`, `dest[2] = 20` -- this function
    /// returns those same three sums, just keyed by first-seen id order
    /// (`[0, 2, 1]`) instead of a materialized `[dest0, dest1, dest2]` array.
    #[test]
    fn colliding_indices_sum_instead_of_overwriting() {
        let indices = [0.0f32, 2.0, 0.0, 1.0];
        let values = vec![10.0f32, 20.0, 30.0, 40.0];

        let (unique_ids, summed) =
            dedupe_and_sum_rows(&indices, &values, 1).expect("row_len divides evenly");

        assert_eq!(
            unique_ids,
            vec![0, 2, 1],
            "first-seen order of distinct ids"
        );
        assert_eq!(
            summed,
            vec![40.0, 20.0, 40.0],
            "id 0 collides (src[0] + src[2] = 40); id 2 and id 1 are single writes"
        );
    }

    #[test]
    fn multi_element_rows_sum_elementwise() {
        let indices = [5.0f32, 5.0, 1.0];
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 9.0, 9.0];

        let (unique_ids, summed) =
            dedupe_and_sum_rows(&indices, &values, 2).expect("row_len divides evenly");

        assert_eq!(unique_ids, vec![5, 1]);
        assert_eq!(summed, vec![4.0, 6.0, 9.0, 9.0], "row 5 = [1,2] + [3,4]");
    }

    #[test]
    fn mismatched_row_length_is_a_typed_error_not_a_panic() {
        let indices = [0.0f32, 1.0];
        let values = vec![1.0f32, 2.0, 3.0];

        let outcome = dedupe_and_sum_rows(&indices, &values, 2);

        assert_eq!(
            outcome,
            Err(AutogradError::SparseRowLengthMismatch {
                row_len: 2,
                found: 3
            }),
            "3 values do not divide evenly into 2 rows of length 2"
        );
    }
}
