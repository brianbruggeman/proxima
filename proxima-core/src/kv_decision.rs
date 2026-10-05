/// Rows of a layer that hold `cached_rows` rows and are sealed: only full blocks whose every row is at least `horizon_rows` old.
#[must_use]
pub const fn seal_target(cached_rows: usize, block_tokens: usize, horizon_rows: usize) -> usize {
    if block_tokens == 0 {
        return 0;
    }
    cached_rows.saturating_sub(horizon_rows) / block_tokens * block_tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_decision_seal_target_worked() {
        let targets: [usize; 10] = core::array::from_fn(|rows| seal_target(rows, 4, 1));

        assert_eq!(targets, [0, 0, 0, 0, 0, 4, 4, 4, 4, 8]);
    }

    #[test]
    fn kv_decision_seal_target_edges() {
        assert_eq!(seal_target(3, 4, 0), 0);
        assert_eq!(seal_target(4, 4, 0), 4);
        assert_eq!(seal_target(8, 4, 8), 0);
        assert_eq!(seal_target(100, 0, 1), 0);
        assert_eq!(seal_target(usize::MAX, 4, 0), usize::MAX - (usize::MAX % 4));
    }
}
