//! Greedy decoding: turns a model's raw logits into the token id the
//! model is most confident in. Pairs with [`crate::decode`] to complete
//! the logits -> id -> text path -- a free function over `&[f32]`, no new
//! type, because a sampler is a pure reduction over a slice and the pipe
//! algebra already expresses that as a function, not a form. Lives here
//! rather than a dedicated inference crate because it is the other half
//! of the same boundary [`crate::decode`] already owns: [`crate::decode`]
//! turns ids into text, [`greedy_pick`] turns logits into an id.

/// Returns the index of the largest value in `logits`, or `None` if
/// `logits` is empty. Ties resolve to the lowest index (the first
/// occurrence of the maximum) -- deterministic, and matching the
/// convention most reference decoders use for greedy argmax.
#[must_use]
pub fn greedy_pick(logits: &[f32]) -> Option<u32> {
    logits
        .iter()
        .enumerate()
        .fold(None, |best, (index, &value)| match best {
            Some((_, best_value)) if value <= best_value => best,
            _ => Some((index, value)),
        })
        .map(|(index, _)| index as u32)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec;

    use super::greedy_pick;

    #[test]
    fn empty_logits_pick_nothing() {
        assert_eq!(greedy_pick(&[]), None);
    }

    #[test]
    fn picks_the_single_peak() {
        let logits = vec![0.1, 0.9, -3.0, 0.2];
        assert_eq!(greedy_pick(&logits), Some(1));
    }

    #[test]
    fn ties_resolve_to_the_lowest_index() {
        let logits = vec![0.5, 0.9, 0.9, 0.1];
        assert_eq!(greedy_pick(&logits), Some(1));
    }

    #[test]
    fn constant_logits_do_not_leak_a_wrong_peak() {
        // degenerate control: a broken argmax that always returns a
        // fixed non-zero index would pass a naive "returns something"
        // assertion. asserting the exact deterministic tie-break (lowest
        // index) catches that class of bug.
        let logits = vec![3.0; 16];
        assert_eq!(greedy_pick(&logits), Some(0));
    }
}
