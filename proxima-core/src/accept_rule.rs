//! The accept decision for speculative rows as loadable data: [`AcceptRule::accepted_rows`] is the pure decision, and its count is the `accepted` argument of [`crate::serving_state::ServingState::accept_rows`].
//!
//! A similarity floor of 1.0 under exact equality, minimum run 0 and no cap reproduce [`crate::serving_state::ServingState::accept`].

/// Three scalars that decide how many leading drafted rows are accepted; copyable so it can ride in a config.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcceptRule {
    /// A drafted row counts as accepted when its similarity to the verifier's row is at least this; 1.0 admits only rows the caller's similarity scores 1.0.
    pub similarity_floor: f32,
    /// A leading accepted run shorter than this is discarded, so the count is 0; 3 for a rule that only commits runs of three.
    pub min_run: u16,
    /// The leading run is cut at this many rows; 8.
    pub max_rows: u16,
}

impl Default for AcceptRule {
    fn default() -> Self {
        Self {
            similarity_floor: 1.0,
            min_run: 0,
            max_rows: u16::MAX,
        }
    }
}

impl AcceptRule {
    /// Count of leading rows accepted; the caller is [`crate::serving_state::ServingState::accept_rows`]. The similarity is an argument because the entry type is generic, so only the caller can score two entries.
    #[must_use]
    pub fn accepted_rows<Entry>(
        &self,
        draft: &[Entry],
        choices: &[Entry],
        similarity: impl Fn(&Entry, &Entry) -> f32,
    ) -> usize {
        let run = draft
            .iter()
            .zip(choices)
            .take(usize::from(self.max_rows))
            .take_while(|(drafted, chosen)| similarity(drafted, chosen) >= self.similarity_floor)
            .count();
        if run < usize::from(self.min_run) { 0 } else { run }
    }
}

#[cfg(test)]
// a failed transition in a test is a broken test; expect names it
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::serving_state::ServingState;

    fn exact(drafted: &u32, chosen: &u32) -> f32 {
        if drafted == chosen { 1.0 } else { 0.0 }
    }

    fn rule(similarity_floor: f32, min_run: u16, max_rows: u16) -> AcceptRule {
        AcceptRule { similarity_floor, min_run, max_rows }
    }

    #[test]
    fn accept_rule_default_matches_leading_equal_accept() {
        let cases = [
            ([3_u32, 4, 5], [3_u32, 4, 5]),
            ([3, 4, 5], [3, 9, 5]),
            ([3, 4, 5], [8, 4, 5]),
        ];
        let results: alloc::vec::Vec<_> = cases
            .iter()
            .map(|(draft, choices)| {
                let state = ServingState::start(alloc::vec![1_u32], 0_usize)
                    .advance_prefill(2, 1)
                    .expect("prefill advances")
                    .enter_verify(draft.to_vec())
                    .expect("verify enters");
                let caches = alloc::vec![10_usize, 11, 12];
                let direct = state.clone().accept(choices, caches.clone());
                let ruled = state.accept_rows(
                    AcceptRule::default().accepted_rows(draft, choices, exact),
                    choices,
                    caches,
                );
                assert_eq!(direct, ruled);
                ruled
            })
            .collect();

        assert_eq!(
            results,
            alloc::vec![
                Ok(ServingState::Accept { n: 3, next: 5, cache: 12 }),
                Ok(ServingState::Rollback { snapshot: 11, to: 9 }),
                Ok(ServingState::Rollback { snapshot: 10, to: 8 }),
            ]
        );
    }

    #[test]
    fn accept_rule_similarity_floor_admits_near_matches() {
        let draft = [10_u8, 20, 30];
        let choices = [10_u8, 21, 35];
        let similarity = |a: &u8, b: &u8| 1.0 - f32::from(a.abs_diff(*b)) / 10.0;

        assert_eq!(rule(1.0, 0, u16::MAX).accepted_rows(&draft, &choices, similarity), 1);
        assert_eq!(rule(0.85, 0, u16::MAX).accepted_rows(&draft, &choices, similarity), 2);
        assert_eq!(rule(0.4, 0, u16::MAX).accepted_rows(&draft, &choices, similarity), 3);
    }

    #[test]
    fn accept_rule_row_cap_limits_the_accepted_run() {
        let rows = [1_u32, 2, 3, 4, 5];

        assert_eq!(rule(1.0, 0, 3).accepted_rows(&rows, &rows, exact), 3);
        assert_eq!(rule(1.0, 0, 9).accepted_rows(&rows, &rows, exact), 5);
        assert_eq!(rule(1.0, 0, 0).accepted_rows(&rows, &rows, exact), 0);
    }

    #[test]
    fn accept_rule_minimum_run_refuses_short_runs() {
        let draft = [1_u32, 2, 3, 9];
        let choices = [1_u32, 2, 3, 4];

        assert_eq!(rule(1.0, 0, u16::MAX).accepted_rows(&draft, &choices, exact), 3);
        assert_eq!(rule(1.0, 3, u16::MAX).accepted_rows(&draft, &choices, exact), 3);
        assert_eq!(rule(1.0, 4, u16::MAX).accepted_rows(&draft, &choices, exact), 0);
        assert_eq!(rule(1.0, 3, 2).accepted_rows(&draft, &choices, exact), 0);
    }
}
