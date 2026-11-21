use alloc::vec;
use alloc::vec::Vec;

use serde_json::json;

use super::*;

fn template(name: &str, extra_claim: &str) -> FamilyTemplate {
    serde_json::from_value(json!({
        "name": name,
        "claims": [extra_claim],
        "required_stages": ["only_stage"],
        "stacks": [{"role": "block", "pattern": "stack.{L}."}],
    }))
    .expect("template literal is well formed")
}

fn tensor(name: &str) -> TensorShape {
    TensorShape {
        name: name.into(),
        shape: vec![4, 4],
    }
}

fn sha() -> CheckpointSha {
    CheckpointSha::parse(&"ab".repeat(32)).expect("64 hex digits")
}

#[test]
fn ambiguous_family_names_every_template_that_matches() {
    let tensors = vec![tensor("stack.0.w"), tensor("stack.1.w")];
    let templates = vec![template("first", "x"), template("second", "y")];

    let error = infer(&tensors, &json!({}), &templates, sha()).expect_err("two match");

    let expected = InferError::AmbiguousFamily {
        families: vec!["first".into(), "second".into()],
    };
    assert_eq!(error, expected);
}

#[test]
fn unclaimed_tensors_lists_what_no_slot_claims() {
    let tensors = vec![tensor("stack.0.w"), tensor("stray.bias")];
    let templates = vec![template("only", "x")];

    let error = infer(&tensors, &json!({}), &templates, sha()).expect_err("stray tensor");

    let names: Vec<_> = vec!["stray.bias".into()];
    let expected = InferError::UnclaimedTensors {
        family: "only".into(),
        names,
    };
    assert_eq!(error, expected);
}

#[test]
fn pattern_with_two_slots_is_refused_at_parse() {
    let error = Pattern::parse("a.{L}.b.{L}").expect_err("two slots");

    assert_eq!(
        error,
        PatternError::SlotCount {
            pattern: "a.{L}.b.{L}".into(),
            found: 2
        }
    );
}

#[test]
fn stack_gap_names_the_missing_layer() {
    let tensors = vec![tensor("stack.0.w"), tensor("stack.2.w")];
    let templates = vec![template("only", "x")];

    let error = infer(&tensors, &json!({}), &templates, sha()).expect_err("layer 1 missing");

    let expected = InferError::StackGap {
        role: "block".into(),
        missing: 1,
    };
    assert_eq!(error, expected);
}

#[test]
fn template_without_required_stages_is_refused_at_validate() {
    let mut bare = template("bare", "x");
    bare.required_stages.clear();

    let error = bare.validate().expect_err("no stages declared");

    assert_eq!(
        error,
        TemplateError::NoRequiredStages {
            family: "bare".into()
        }
    );
}

#[test]
fn required_stages_are_carried_on_inferred_and_change_the_digest() {
    let tensors = vec![tensor("stack.0.w")];
    let first = infer(&tensors, &json!({}), &[template("only", "x")], sha()).expect("infers");
    let mut more = template("only", "x");
    more.required_stages.push("second".into());

    let second = infer(&tensors, &json!({}), &[more], sha()).expect("infers");

    assert_eq!(first.required_stages, ["only_stage"]);
    assert_ne!(first.digest(), second.digest());
}
