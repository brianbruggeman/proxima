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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AssembleStep {
    /// Start from the stored entry that shares the longest prefix with the prompt.
    Prefix,
    /// Also reuse stored chunks that moved position; needs the prompt cache's
    /// `cache_reuse_min` and `ring_rewind_slack` above 0.
    Shift,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

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

    #[test]
    fn serving_grammar_assemble_steps_round_trip_json() {
        let prefix: AssembleStep = serde_json::from_str(r#"{"kind":"prefix"}"#).expect("prefix parses");
        let shift: AssembleStep = serde_json::from_str(r#"{"kind":"shift"}"#).expect("shift parses");
        let ordered: Vec<AssembleStep> =
            serde_json::from_str(r#"[{"kind":"shift"},{"kind":"prefix"}]"#).expect("list parses");

        assert_eq!(prefix, AssembleStep::Prefix);
        assert_eq!(shift, AssembleStep::Shift);
        assert_eq!(ordered, vec![AssembleStep::Shift, AssembleStep::Prefix]);
        for step in [AssembleStep::Prefix, AssembleStep::Shift] {
            let written = serde_json::to_string(&step).expect("step writes");
            assert_eq!(serde_json::from_str::<AssembleStep>(&written).expect("step reparses"), step);
        }
        let unrun = r#"{"kind":"load","path":"/models/cartridges/legal-v3.cart"}"#;
        assert!(serde_json::from_str::<AssembleStep>(unrun).is_err());
        assert!(serde_json::from_str::<AssembleStep>(r#"{"kind":"blend"}"#).is_err());
        assert!(serde_json::from_str::<AssembleStep>(r#"{"kind":"Prefix"}"#).is_err());
    }
}
