//! The greedy token pick as the last ops of the decode program.
//!
//! A greedy step used to read the whole `[1, vocab]` logits row back to the
//! host (262,144 floats, 1 MiB) and scan it with `proxima_tokenizer::
//! greedy_pick` -- about 0.45 ms on the critical path between one token's
//! last GPU dispatch and the next token's first. [`with_greedy_argmax`]
//! appends `proxima_tensor::spec::greedy_argmax` to a copy of the program, so
//! the device finishes the pick and the host reads two scalars
//! ([`device_token`]). The logits row itself is written to a caller-owned
//! [`PlacedBuffer`] (`omega::execute_plan_named_with_placements`) rather than
//! read back, which is what keeps a row that [`device_token`] refuses --
//! NaN or an infinity, which have no defined argmax -- recoverable by the host
//! path with exactly the values the device saw.
//!
//! Only plain greedy qualifies: `temperature <= 0` with neutral penalties,
//! because anything else (repetition/frequency/presence penalties,
//! temperature sampling, a forced `token_override`) needs the row's values,
//! not only its peak, and keeps the host path through
//! `select_decoded_token`. Speculative verify batches also keep it: they read
//! `new_count` rows and stop at the first mismatch, so the host already pays
//! only for the rows it uses.

use super::*;

/// `program` plus the greedy-pick ops over `logits_root`, with the `(token,
/// finite)` nodes to request, or `None` when the program does not accept
/// them -- a logits root that is not `[rows, vocab]`, say. The original ops
/// keep their node ids, so every layer root and cache leaf resolved against
/// `program` stays valid against the copy.
pub(super) fn with_greedy_argmax(
    program: &[Op],
    logits_root: NodeId,
    vocab: u32,
    symbols: &[u64],
) -> Option<(Vec<Op>, NodeId, NodeId)> {
    let mut augmented = program.to_vec();
    let (token, finite) =
        proxima_tensor::spec::greedy_argmax(&mut augmented, logits_root, vocab).ok()?;
    proxima_tensor::shape::infer(&augmented, symbols).ok()?;
    Some((augmented, token, finite))
}

/// The device's pick, only when the row held `vocab` clean entries and the
/// index is one the vocabulary has. `None` sends the caller to the host path
/// over the placed logits.
pub(super) fn device_token(
    evaluated: &Evaluated,
    token: NodeId,
    finite: NodeId,
    vocab: usize,
) -> Option<u32> {
    let (picked, _) = evaluated.get(token)?;
    let (clean, _) = evaluated.get(finite)?;
    let index = *picked.first()?;
    let clean_entries = *clean.first()?;
    let in_vocabulary = index >= 0.0 && index < vocab as f32;
    (clean_entries == vocab as f32 && in_vocabulary).then_some(index as u32)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proxima_tensor::{DType, NumericPolicy, cpu::QuantizedBlock, spec::input_leaf};
    use proxima_tokenizer::{SamplingConfig, sample_next_token};

    const VOCAB: usize = 262_144;

    fn greedy_config() -> SamplingConfig {
        SamplingConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }

    /// A softcapped row shaped like the sliding-pattern family's real logits: every entry inside
    /// `(-30, 30)`, spread unevenly, none repeating the peak by accident.
    fn softcapped_row() -> Vec<f32> {
        (0..VOCAB)
            .map(|index| {
                let scrambled = (index as u64).wrapping_mul(2_654_435_761) % 1_000_003;
                30.0 * ((scrambled as f32 / 1_000_003.0 - 0.5) * 6.0).tanh()
            })
            .collect()
    }

    fn pick_on_device(row: &[f32]) -> Option<u32> {
        let mut program = Vec::new();
        let logits = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                proxima_tensor::Extent::Symbolic(0),
                proxima_tensor::Extent::Static(VOCAB as u32)
            ],
            "logits",
        );
        let (augmented, token, finite) =
            with_greedy_argmax(&program, logits, VOCAB as u32, &[1]).expect("argmax fits [1, vocab]");
        let named = [("logits", QuantizedBlock::Float32(row))];
        let plan = omega::plan_named(&augmented, &[1], &named, &[token, finite], NumericPolicy::default())
            .expect("plans on the real Metal device");
        let evaluated = omega::execute_plan_named(&plan, &named).expect("argmax runs on the device");
        device_token(&evaluated, token, finite, VOCAB)
    }

    fn pick_on_host(row: &[f32]) -> Option<u32> {
        sample_next_token(row, &[], greedy_config(), &mut fastrand::Rng::with_seed(1))
    }

    #[test]
    fn the_device_pick_is_the_host_pick_on_a_softcapped_row() {
        let row = softcapped_row();

        assert_eq!(pick_on_device(&row), pick_on_host(&row));
    }

    #[test]
    fn a_tie_across_distant_blocks_resolves_to_the_lowest_index_as_on_the_host() {
        let mut row = softcapped_row();
        for index in [262_143, 131_071, 5, 200_000] {
            row[index] = 31.5;
        }

        assert_eq!(pick_on_device(&row), Some(5));
        assert_eq!(pick_on_device(&row), pick_on_host(&row));
    }

    #[test]
    fn a_row_with_nan_or_an_infinity_is_handed_back_to_the_host() {
        let mut with_nan = softcapped_row();
        with_nan[77] = f32::NAN;
        let mut with_infinity = softcapped_row();
        with_infinity[9] = f32::INFINITY;

        assert_eq!(pick_on_device(&with_nan), None);
        assert_eq!(pick_on_device(&with_infinity), None);
    }
}
