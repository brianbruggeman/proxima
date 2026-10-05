use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadSpec {
    /// Every cached row is read, today's behaviour.
    #[default]
    Dense,
    /// Each attention layer declares a per-row skip operand that is filled
    /// every decode step from the read rule the loaded model carries, so the
    /// rule decides which cached rows a step reads; until the decode step
    /// binds that rule, a request that sets it reads every row.
    Operand,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn serving_grammar_read_spec_round_trips_json() {
        let dense: ReadSpec = serde_json::from_str("\"dense\"").expect("dense parses");
        let operand: ReadSpec = serde_json::from_str("\"operand\"").expect("operand parses");

        assert_eq!(dense, ReadSpec::Dense);
        assert_eq!(operand, ReadSpec::Operand);
        assert_eq!(serde_json::to_string(&ReadSpec::Dense).expect("dense writes"), "\"dense\"");
        assert_eq!(serde_json::to_string(&ReadSpec::Operand).expect("operand writes"), "\"operand\"");
        assert_eq!(ReadSpec::default(), ReadSpec::Dense);
        assert!(serde_json::from_str::<ReadSpec>("\"block\"").is_err());
        assert!(serde_json::from_str::<ReadSpec>("\"Dense\"").is_err());
    }
}
