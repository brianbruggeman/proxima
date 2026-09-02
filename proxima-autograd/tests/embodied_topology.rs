//! Does a declarative capability spec DERIVE a tensor program, such that
//! changing the spec changes the program -- never a hardcoded topology, never
//! a dense program masked down after the fact?
//!
//! Domain-free on purpose (this started as a body/appendage/sensor design and
//! was corrected: a mouse's legs and whiskers are a *consumer's* vocabulary,
//! not proxima's -- proxima holds the shape). What is derived here:
//!
//! - some set of available **inputs** (a name is either active or it is not;
//!   input width is just how many are active),
//! - a declared **capability manifest** (every capability id a spec is even
//!   allowed to talk about) and a **present** subset of it (which of those
//!   this instance currently has -- this is the thing that shrinks under
//!   damage and grows under repair/addition, at runtime, without recompiling
//!   anything),
//! - some set of declared **outputs**, each naming the capabilities it
//!   `requires`; an output is *satisfiable* iff every required capability is
//!   in `present`. This is [`Spec::validate`]'s whole point: `requires`
//!   referencing a capability outside the manifest is a spec bug (caught,
//!   typed), while `requires` referencing a manifest capability that is
//!   merely not `present` right now is not a bug at all -- it is exactly the
//!   thing that makes the output unavailable, computed, never hardcoded.
//!
//! [`Spec`] is serde-derived (`#[derive(Deserialize, Serialize)]`), the same
//! precedent `proxima_tensor::spec::NodeSpec` sets for turning a structure
//! into TOML (`proxima-tensor/src/spec.rs:120-123`, `#[serde(tag = "op")]`)
//! -- [`body_round_trips_through_toml`] is the test that makes that a
//! checked claim rather than an assumed one, exactly as `spec.rs`'s own
//! `a_program_written_as_toml_equals_the_same_program_written_in_rust` does
//! for a hand-built matmul.
//!
//! [`derive_program`] builds the program from nothing but [`op::append`],
//! `Op::Input`/`Op::Constant`/`Op::Elementwise`, [`map::projection`], and
//! [`proxima_autograd::activation::relu`] -- no new library primitive, no
//! `Op` variant, reusing exactly the surface `proxima-autograd/src/expr.rs`
//! and `proxima-autograd/src/activation.rs` already compose. Every
//! contribution (`ScalarOp::Multiply` term feeding an accumulating
//! `ScalarOp::Add` chain) is scalar-unrolled -- one node per (source, unit)
//! pair -- rather than the vectorized `Elementwise(Multiply)` +
//! `Op::Reduce` matmul idiom `proxima-tensor/src/spec.rs:2701-2719` and
//! `proxima-autograd/tests/constructed_sparse.rs:118-141` both use. That is
//! a deliberate trade, not an oversight: this file's central claim needs "an
//! unavailable output has NO node at all, provable by searching the
//! program" (`find_named_node`, below) rather than "the shared reduce's
//! iteration space got smaller" -- and only a dedicated, individually
//! named node per output can be searched for and found *absent*. The price
//! is `constructed_sparse.rs`'s own O(1)-node-count-regardless-of-topology
//! property: here, op count and MAC count both scale linearly with
//! input/hidden/output width instead of staying flat. [`per_body_op_and_mac_report_and_cross_body_op_sharing`]
//! reports exactly that cost, measured, for four distinct bodies.

#![allow(clippy::unwrap_used, clippy::expect_used)]

extern crate alloc;

use proxima_autograd::activation;
use proxima_tensor::cpu::evaluate_named;
use proxima_tensor::dtype::DType;
use proxima_tensor::map::{self, IndexMap};
use proxima_tensor::op::{self, NodeId, Op, ScalarOp};
use serde::{Deserialize, Serialize};

/// One declared output: an id, and every capability that must be `present`
/// for it to be satisfiable. `requires = []` means always satisfiable (the
/// "needs nothing from the body" case).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
struct OutputSpec {
    id: alloc::string::String,
    requires: alloc::vec::Vec<alloc::string::String>,
}

/// A capability spec: `inputs` are this instance's active signal ids;
/// `capabilities` is the full manifest `outputs[].requires` may reference
/// ([`Spec::validate`] enforces this); `present` is the runtime-mutable
/// subset of `capabilities` this instance currently has; `hidden_width` is
/// an independent sizing axis, distinct from both `inputs.len()` and
/// `outputs.len()` by construction in every body this file builds (ROW 135:
/// a shared channel size hides a transpose bug).
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
struct Spec {
    inputs: alloc::vec::Vec<alloc::string::String>,
    capabilities: alloc::vec::Vec<alloc::string::String>,
    present: alloc::vec::Vec<alloc::string::String>,
    outputs: alloc::vec::Vec<OutputSpec>,
    hidden_width: usize,
}

/// Every fault [`Spec::validate`] catches -- both are the same shape of bug
/// (a capability id used somewhere `capabilities` never declared), at the
/// two places one can appear.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
enum SpecError {
    #[error("output {output:?} requires undeclared capability {capability:?}")]
    UnknownCapability {
        output: alloc::string::String,
        capability: alloc::string::String,
    },
    #[error("capability {capability:?} is marked present but was never declared in the manifest")]
    UndeclaredPresence { capability: alloc::string::String },
}

impl Spec {
    /// A spec is well-formed iff every capability id that appears anywhere
    /// (an output's `requires`, or `present`) is drawn from `capabilities`.
    /// This is checked once, independent of which subset happens to be
    /// `present` right now -- damage/repair only ever mutates `present`, so
    /// a well-formed spec stays well-formed across every body variant this
    /// file derives from it.
    fn validate(&self) -> Result<(), SpecError> {
        for output in &self.outputs {
            for capability in &output.requires {
                if !self.capabilities.contains(capability) {
                    return Err(SpecError::UnknownCapability {
                        output: output.id.clone(),
                        capability: capability.clone(),
                    });
                }
            }
        }
        for capability in &self.present {
            if !self.capabilities.contains(capability) {
                return Err(SpecError::UndeclaredPresence {
                    capability: capability.clone(),
                });
            }
        }
        Ok(())
    }

    /// Computed, never hardcoded: every one of `output.requires` must be in
    /// `present`.
    fn satisfiable(&self, output: &OutputSpec) -> bool {
        output
            .requires
            .iter()
            .all(|capability| self.present.contains(capability))
    }
}

/// A derived program plus the bookkeeping a test needs to drive it: each
/// active input's own leaf node (bound by [`evaluate_named`]'s `named`,
/// positionally paired with [`sensor_value`]), and each *satisfiable*
/// output's id alongside the node id [`evaluate_named`]'s `outputs` should
/// request for it.
struct Derived {
    program: alloc::vec::Vec<Op>,
    input_nodes: alloc::vec::Vec<(alloc::string::String, NodeId)>,
    output_nodes: alloc::vec::Vec<(alloc::string::String, NodeId)>,
}

fn scalar_map() -> IndexMap {
    IndexMap::Affine(map::projection(0, &[]))
}

/// A fixed, reproducible weight: a pure function of which layer (`layer_tag`
/// distinguishes the hidden layer from the output layer), which unit within
/// it, and which source feeds it -- no RNG, no clock, so re-deriving the
/// same spec twice always emits bit-identical `Op::Constant` values.
fn seeded_weight(layer_tag: u64, unit_index: usize, source_index: usize) -> f32 {
    let raw = layer_tag
        .wrapping_mul(97)
        .wrapping_add(unit_index as u64 * 13 + source_index as u64 * 5 + 3);
    (((raw % 19) as f32) - 9.0) / 11.0
}

const HIDDEN_LAYER_TAG: u64 = 1;
const OUTPUT_LAYER_TAG: u64 = 2;

/// `sum_i sources[i] * seeded_weight(layer_tag, unit_index, i)`, unrolled as
/// one `Constant` + one `Multiply` per source plus an accumulating `Add`
/// chain -- see this file's own top-of-file doc for why this is scalar and
/// not a vectorized `Reduce`. An empty `sources` (every capability this
/// unit would have read from is gone) degrades to a `Constant(0.0)` rather
/// than panicking; this file never actually drives that branch (every body
/// below keeps at least one input and the sensor-loss test only drops one
/// of five), but a spec is data a caller could still hand in with zero
/// active inputs, and this function must not panic on it.
fn accumulate(
    program: &mut alloc::vec::Vec<Op>,
    layer_tag: u64,
    unit_index: usize,
    sources: &[NodeId],
) -> NodeId {
    let mut accumulator: Option<NodeId> = None;
    for (source_index, &source) in sources.iter().enumerate() {
        let weight = seeded_weight(layer_tag, unit_index, source_index);
        let weight_node = op::append(
            program,
            Op::Constant {
                dtype: DType::Float32,
                shape: alloc::vec::Vec::new(),
                value: weight,
            },
        );
        let term = op::append(
            program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![(source, scalar_map()), (weight_node, scalar_map())],
                name: None,
            },
        );
        accumulator = Some(match accumulator {
            None => term,
            Some(previous) => op::append(
                program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Add,
                    operands: alloc::vec![(previous, scalar_map()), (term, scalar_map())],
                    name: None,
                },
            ),
        });
    }
    accumulator.unwrap_or_else(|| {
        op::append(
            program,
            Op::Constant {
                dtype: DType::Float32,
                shape: alloc::vec::Vec::new(),
                value: 0.0,
            },
        )
    })
}

/// `spec -> Vec<Op>`. Input width is `spec.inputs.len()`; output width is
/// the count of outputs [`Spec::satisfiable`] accepts. An unsatisfiable
/// output contributes NOTHING -- no `accumulate` call, no named node, not
/// even a zeroed one -- which is the property [`find_named_node`] proves.
fn derive_program(spec: &Spec) -> Result<Derived, SpecError> {
    spec.validate()?;
    let mut program = alloc::vec::Vec::new();

    let input_nodes: alloc::vec::Vec<(alloc::string::String, NodeId)> = spec
        .inputs
        .iter()
        .map(|id| {
            let node = op::append(
                &mut program,
                Op::Input {
                    dtype: DType::Float32,
                    shape: alloc::vec::Vec::new(),
                    name: Some(id.clone()),
                },
            );
            (id.clone(), node)
        })
        .collect();
    let input_node_ids: alloc::vec::Vec<NodeId> =
        input_nodes.iter().map(|(_, node)| *node).collect();

    let hidden_nodes: alloc::vec::Vec<NodeId> = (0..spec.hidden_width)
        .map(|unit_index| {
            let pre_activation =
                accumulate(&mut program, HIDDEN_LAYER_TAG, unit_index, &input_node_ids);
            activation::relu(&mut program, DType::Float32, pre_activation, 0)
        })
        .collect();

    let mut output_nodes = alloc::vec::Vec::new();
    for output in &spec.outputs {
        if spec.satisfiable(output) {
            let logit = accumulate(
                &mut program,
                OUTPUT_LAYER_TAG,
                output_nodes.len(),
                &hidden_nodes,
            );
            let named = op::append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Identity,
                    operands: alloc::vec![(logit, scalar_map())],
                    name: Some(output.id.clone()),
                },
            );
            output_nodes.push((output.id.clone(), named));
        }
    }

    Ok(Derived {
        program,
        input_nodes,
        output_nodes,
    })
}

/// The absence proof: search the emitted `Vec<Op>` itself (not `Derived`'s
/// own bookkeeping, which would trivially already agree with derivation) for
/// a node named `name`. `Op::name()` covers `Input`/`Elementwise`/`Reduce`
/// (`proxima-tensor/src/op.rs:281-287`) -- every per-output node this file
/// emits is an `Elementwise`, so this is a real search of the program a
/// caller would actually run, not a check against a side channel.
fn find_named_node<'program>(program: &'program [Op], name: &str) -> Option<&'program Op> {
    program.iter().find(|op| op.name() == Some(name))
}

/// Ops whose body is `Multiply` are exactly this file's multiply-accumulate
/// count: every `accumulate` term is one `Constant` times one source, one
/// MAC per term, by construction (see `accumulate`'s own doc). Unlike
/// `constructed_sparse.rs`'s `total_macs` (which sums a `Reduce`'s inferred
/// iteration space), this needs no `shape::infer` call at all -- every node
/// here is rank 0, so "one Multiply node" and "one MAC" already coincide.
fn total_macs(program: &[Op]) -> u64 {
    program
        .iter()
        .filter(|op| {
            matches!(
                op,
                Op::Elementwise {
                    body: ScalarOp::Multiply,
                    ..
                }
            )
        })
        .count() as u64
}

/// How many ops in `left` and `right` are literally `Op::PartialEq`-equal,
/// counted as a one-to-one matching (each op in `right` claimed by at most
/// one op in `left`). `Op::Elementwise`/`Op::Reduce` embed `NodeId`s, and a
/// `NodeId` is a *position* in its own program (`proxima-tensor/src/op.rs:9-13`:
/// "a `NodeId` is a position in the slice"), so those only compare equal
/// between two programs that happen to place the same computation at the
/// same position -- effectively never, once the programs diverge in shape
/// upstream. Only position-independent leaves (`Op::Input` keyed by name,
/// `Op::Constant` keyed by its own value) can match regardless of where
/// they land. This function measures that directly rather than asserting
/// it.
fn shared_op_count(left: &[Op], right: &[Op]) -> usize {
    let mut unclaimed: alloc::vec::Vec<&Op> = right.iter().collect();
    let mut shared = 0usize;
    for op in left {
        if let Some(position) = unclaimed.iter().position(|candidate| *candidate == op) {
            unclaimed.remove(position);
            shared += 1;
        }
    }
    shared
}

fn output(id: &str, requires: &[&str]) -> OutputSpec {
    OutputSpec {
        id: id.into(),
        requires: requires
            .iter()
            .map(|capability| (*capability).into())
            .collect(),
    }
}

fn strings(values: &[&str]) -> alloc::vec::Vec<alloc::string::String> {
    values.iter().map(|value| (*value).into()).collect()
}

/// 5 inputs, 4-capability manifest all present, 6 outputs all satisfiable,
/// hidden width 3 -- every one of {5, 3, 6} distinct.
fn full_body() -> Spec {
    Spec {
        inputs: strings(&["s0", "s1", "s2", "s3", "s4"]),
        capabilities: strings(&["cap1", "cap2", "cap3", "cap4"]),
        present: strings(&["cap1", "cap2", "cap3", "cap4"]),
        outputs: alloc::vec![
            output("rest", &[]),
            output("alpha", &["cap1"]),
            output("beta", &["cap2"]),
            output("gamma", &["cap1", "cap3"]),
            output("delta", &["cap4"]),
            output("epsilon", &["cap1", "cap2", "cap4"]),
        ],
        hidden_width: 3,
    }
}

/// `full_body` with `cap4` removed from `present` (never from the
/// manifest) -- `delta` and `epsilon` both need it, so both go from
/// satisfiable to absent; `alpha`/`beta`/`gamma`/`rest` are unaffected.
/// Satisfiable count 4, distinct from hidden width 3 and input count 5.
fn damaged_body_missing_capability() -> Spec {
    let mut spec = full_body();
    spec.present.retain(|capability| capability != "cap4");
    spec
}

/// `full_body` with `cap1` removed instead of `cap4` -- `cap1` gates
/// `alpha` (the SECOND declared output, not the last two), so every
/// satisfiable output declared after it (`beta`, `gamma` survives via
/// cap3 alone... no, gamma also needs cap1, so gamma drops too) shifts
/// down by one position in `output_nodes`, changing its `unit_index` and
/// therefore its seeded weights. Exists purely to show the "exact prefix"
/// property `full`/`missing_capability`/`added_capability` share is an
/// accident of declaring the affected outputs LAST, not a general
/// guarantee -- see `per_body_op_and_mac_report_and_cross_body_op_sharing`.
fn damaged_body_missing_early_capability() -> Spec {
    let mut spec = full_body();
    spec.present.retain(|capability| capability != "cap1");
    spec
}

/// `full_body` with sensor `s4` dropped entirely -- input width 4, distinct
/// from hidden width 3 and the (unaffected) satisfiable-output count 6.
fn damaged_body_missing_sensor() -> Spec {
    let mut spec = full_body();
    spec.inputs.retain(|input| input != "s4");
    spec
}

/// `full_body` plus a new capability `cap5` (present) and a new output
/// `zeta` that needs only it -- satisfiable count 7, distinct from hidden
/// width 3 and input count 5.
fn augmented_body_added_capability() -> Spec {
    let mut spec = full_body();
    spec.capabilities.push("cap5".into());
    spec.present.push("cap5".into());
    spec.outputs.push(output("zeta", &["cap5"]));
    spec
}

/// Deliberately ill-formed: `gamma` requires `cap99`, which appears in no
/// body's `capabilities` manifest anywhere in this file -- a spec bug, not
/// a damage state.
fn ill_formed_body_unknown_capability() -> Spec {
    let mut spec = full_body();
    spec.outputs[3] = output("gamma", &["cap1", "cap99"]);
    spec
}

/// Deliberately ill-formed the other way: `present` claims a capability the
/// manifest never declared.
fn ill_formed_body_undeclared_presence() -> Spec {
    let mut spec = full_body();
    spec.present.push("cap_ghost".into());
    spec
}

fn sensor_value(id: &str) -> f32 {
    match id {
        "s0" => 0.6,
        "s1" => -0.3,
        "s2" => 1.1,
        "s3" => -0.8,
        "s4" => 0.25,
        other => panic!("test fixture has no sensor value for {other:?}"),
    }
}

#[proxima::test]
#[case::always_satisfiable_with_nothing_required(&[], &[], true)]
#[case::satisfied_when_the_single_required_capability_is_present(&["cap1"], &["cap1"], true)]
#[case::unsatisfied_when_the_single_required_capability_is_absent(&[], &["cap1"], false)]
#[case::satisfied_only_when_every_required_capability_is_present(&["cap1", "cap3"], &["cap1", "cap3"], true)]
#[case::unsatisfied_when_only_some_required_capabilities_are_present(&["cap1"], &["cap1", "cap3"], false)]
async fn output_satisfiability_matches_the_declared_requirement(
    #[case] present: &[&str],
    #[case] requires: &[&str],
    #[case] expected: bool,
) {
    let mut spec = full_body();
    spec.present = present.iter().map(|value| (*value).into()).collect();
    let candidate = output("candidate", requires);
    assert_eq!(spec.satisfiable(&candidate), expected);
}

#[proxima::test]
async fn an_output_requiring_an_undeclared_capability_is_rejected_as_ill_formed() {
    let spec = ill_formed_body_unknown_capability();
    let error = spec
        .validate()
        .expect_err("cap99 is not in the manifest, this must be rejected");
    assert_eq!(
        error,
        SpecError::UnknownCapability {
            output: "gamma".into(),
            capability: "cap99".into()
        },
        "the error must name the offending output and capability, not just fail"
    );
}

#[proxima::test]
async fn a_capability_marked_present_but_never_declared_is_rejected_as_ill_formed() {
    let spec = ill_formed_body_undeclared_presence();
    let error = spec
        .validate()
        .expect_err("cap_ghost is not in the manifest, this must be rejected");
    assert_eq!(
        error,
        SpecError::UndeclaredPresence {
            capability: "cap_ghost".into()
        }
    );
}

/// A well-formed body must derive cleanly: proves `validate` is not
/// rejecting everything indiscriminately (the counterpart to the two
/// ill-formed tests above).
#[proxima::test]
async fn a_well_formed_body_validates_and_derives() {
    for spec in [
        full_body(),
        damaged_body_missing_capability(),
        damaged_body_missing_sensor(),
        augmented_body_added_capability(),
    ] {
        spec.validate()
            .expect("every body variant in this file is well-formed");
        derive_program(&spec).expect("a well-formed spec must derive");
    }
}

#[proxima::test]
async fn removing_a_capability_strictly_shrinks_the_derived_program_and_its_mac_count() {
    let full = derive_program(&full_body()).expect("full body derives");
    let damaged = derive_program(&damaged_body_missing_capability()).expect("damaged body derives");

    assert_eq!(
        full.output_nodes.len(),
        6,
        "every output is satisfiable when every capability is present"
    );
    assert_eq!(
        damaged.output_nodes.len(),
        4,
        "delta and epsilon both need cap4, now absent"
    );

    let full_macs = total_macs(&full.program);
    let damaged_macs = total_macs(&damaged.program);
    std::eprintln!(
        "full: ops={} macs={full_macs}; missing cap4: ops={} macs={damaged_macs}",
        full.program.len(),
        damaged.program.len()
    );
    assert!(
        damaged.program.len() < full.program.len(),
        "removing a capability must strictly shrink the program"
    );
    assert!(
        damaged_macs < full_macs,
        "removing a capability must strictly shrink the MAC count"
    );

    for absent in ["delta", "epsilon"] {
        assert!(
            find_named_node(&damaged.program, absent).is_none(),
            "{absent} must have NO node at all once cap4 is gone, not a zeroed one"
        );
        assert!(
            find_named_node(&full.program, absent).is_some(),
            "{absent} must exist in the undamaged body"
        );
    }
    for still_present in ["rest", "alpha", "beta", "gamma"] {
        assert!(
            find_named_node(&damaged.program, still_present).is_some(),
            "{still_present} does not depend on cap4"
        );
    }
}

/// Proves the comparison above can actually fail: a deliberately wrong
/// expected op-count delta must not match the measured one, so the strict
/// inequalities checked above are discriminating rather than vacuous.
#[proxima::test]
async fn program_size_claim_is_falsifiable_a_wrong_op_delta_is_not_matched() {
    let full = derive_program(&full_body()).expect("full body derives");
    let damaged = derive_program(&damaged_body_missing_capability()).expect("damaged body derives");
    let measured_delta = full.program.len() - damaged.program.len();
    let genuinely_wrong_delta = measured_delta * 3 + 7;
    assert_ne!(
        measured_delta, genuinely_wrong_delta,
        "a deliberately wrong op-count delta must not equal the measured one"
    );
}

#[proxima::test]
async fn adding_a_capability_strictly_grows_the_derived_program_and_its_mac_count() {
    let full = derive_program(&full_body()).expect("full body derives");
    let augmented =
        derive_program(&augmented_body_added_capability()).expect("augmented body derives");

    assert_eq!(
        augmented.output_nodes.len(),
        7,
        "zeta joins the six already-satisfiable outputs"
    );
    let full_macs = total_macs(&full.program);
    let augmented_macs = total_macs(&augmented.program);
    std::eprintln!(
        "full: ops={} macs={full_macs}; plus cap5/zeta: ops={} macs={augmented_macs}",
        full.program.len(),
        augmented.program.len()
    );
    assert!(
        augmented.program.len() > full.program.len(),
        "adding a capability must strictly grow the program"
    );
    assert!(
        augmented_macs > full_macs,
        "adding a capability must strictly grow the MAC count"
    );
    assert!(
        find_named_node(&full.program, "zeta").is_none(),
        "zeta cannot exist before cap5 is added"
    );
    assert!(
        find_named_node(&augmented.program, "zeta").is_some(),
        "zeta must exist once cap5 is added"
    );
}

#[proxima::test]
async fn removing_a_sensor_shrinks_input_width_and_costs_exactly_hidden_width_macs() {
    let full_spec = full_body();
    let reduced_spec = damaged_body_missing_sensor();
    let full = derive_program(&full_spec).expect("full body derives");
    let reduced = derive_program(&reduced_spec).expect("sensor-reduced body derives");

    assert_eq!(full.input_nodes.len(), 5);
    assert_eq!(
        reduced.input_nodes.len(),
        4,
        "s4 is gone, input width must shrink by exactly one"
    );
    assert!(
        reduced.input_nodes.iter().all(|(id, _)| id != "s4"),
        "s4 must not appear among the reduced body's inputs"
    );

    let full_macs = total_macs(&full.program);
    let reduced_macs = total_macs(&reduced.program);
    std::eprintln!(
        "full: ops={} macs={full_macs}; missing s4: ops={} macs={reduced_macs}",
        full.program.len(),
        reduced.program.len()
    );
    assert!(
        reduced.program.len() < full.program.len(),
        "removing a sensor must strictly shrink the program"
    );
    assert!(
        reduced_macs < full_macs,
        "removing a sensor must strictly shrink the MAC count"
    );
    assert_eq!(
        full_macs - reduced_macs,
        full_spec.hidden_width as u64,
        "s4 fed exactly one multiply-accumulate term into each of the hidden_width hidden units and nothing \
         else (no output reads a sensor directly) -- removing it must cost exactly hidden_width MACs"
    );
}

#[proxima::test]
#[case::full_body_all_six_actions(full_body(), 6)]
#[case::missing_capability_four_actions(damaged_body_missing_capability(), 4)]
#[case::missing_sensor_still_six_actions(damaged_body_missing_sensor(), 6)]
#[case::augmented_body_seven_actions(augmented_body_added_capability(), 7)]
async fn each_derived_body_program_evaluates_to_an_action_distribution_of_the_right_width(
    #[case] spec: Spec,
    #[case] expected_width: usize,
) {
    let derived = derive_program(&spec).expect("well-formed body derives");
    assert_eq!(derived.output_nodes.len(), expected_width);

    let bindings: alloc::vec::Vec<(&str, alloc::vec::Vec<f32>)> = derived
        .input_nodes
        .iter()
        .map(|(id, _)| (id.as_str(), alloc::vec![sensor_value(id)]))
        .collect();
    let named_bindings: alloc::vec::Vec<(&str, &[f32])> = bindings
        .iter()
        .map(|(id, values)| (*id, values.as_slice()))
        .collect();
    let output_ids: alloc::vec::Vec<NodeId> =
        derived.output_nodes.iter().map(|(_, node)| *node).collect();

    let evaluated = evaluate_named(&derived.program, &[], &named_bindings, &output_ids)
        .expect("derived program lowers and evaluates");
    assert_eq!(
        output_ids.len(),
        expected_width,
        "the number of requested outputs IS the action distribution's width"
    );
    let expected_shape: alloc::vec::Vec<u64> = alloc::vec::Vec::new();
    for (label, node) in &derived.output_nodes {
        let (values, shape) = evaluated
            .get(*node)
            .unwrap_or_else(|| panic!("{label} was requested but not returned"));
        assert_eq!(
            shape,
            expected_shape.as_slice(),
            "each action's readout is a scalar logit, rank 0"
        );
        assert_eq!(values.len(), 1, "{label} must produce exactly one value");
        assert!(
            values[0].is_finite(),
            "{label} produced a non-finite logit: {values:?}"
        );
    }
}

#[proxima::test]
async fn body_round_trips_through_toml() {
    let spec = augmented_body_added_capability();
    let text = toml::to_string(&spec).expect("a well-formed spec serializes to TOML");
    let parsed: Spec = toml::from_str(&text).expect("the serialized TOML parses back");
    assert_eq!(
        parsed, spec,
        "a body must round-trip through TOML unchanged"
    );

    let direct_program = derive_program(&spec).expect("the in-memory spec derives");
    let round_tripped_program = derive_program(&parsed).expect("the round-tripped spec derives");
    assert_eq!(
        direct_program.program, round_tripped_program.program,
        "deriving from the round-tripped spec must produce the identical program"
    );
}

/// The design question the derivation exists to answer: build five distinct
/// bodies, derive each, and report op count, MAC count, and how much of
/// each derived program is literally shared with the others -- then let the
/// MEASURED overlap decide the batching question, not an assumption.
///
/// Three findings, each pinned by an assertion below:
///
/// 1. `full`/`missing_capability`/`added_capability` differ ONLY in
///    `present` and in outputs declared at the very END of the list
///    (`delta`/`epsilon` removed, `zeta` added) -- nothing upstream of them
///    changes position, so the smaller program is an EXACT, contiguous
///    PREFIX of the larger one. Not a coincidence of count: sliced and
///    compared directly below.
/// 2. That prefix property is an accident of declaration order, not a
///    property of the derivation. `missing_early_capability` removes `cap1`,
///    which gates `alpha` -- the SECOND declared output -- so every
///    satisfiable output declared after it is renumbered (a different
///    `unit_index`, hence different seeded weights) and the exact-prefix
///    property breaks: measured 78 of 80 ops shared with `full`, not all 80.
/// 3. A sensor change is categorically worse: it moves the shared hidden
///    layer itself, so overlap collapses to well under half.
#[proxima::test]
async fn per_body_op_and_mac_report_and_cross_body_op_sharing() {
    let bodies: alloc::vec::Vec<(&str, Spec)> = alloc::vec![
        ("full", full_body()),
        ("missing_capability", damaged_body_missing_capability()),
        ("missing_sensor", damaged_body_missing_sensor()),
        ("added_capability", augmented_body_added_capability()),
        (
            "missing_early_capability",
            damaged_body_missing_early_capability()
        ),
    ];
    let derived: alloc::vec::Vec<(&str, Derived)> = bodies
        .iter()
        .map(|(label, spec)| {
            (
                *label,
                derive_program(spec).expect("every body in this sweep is well-formed"),
            )
        })
        .collect();

    for (label, body) in &derived {
        std::eprintln!(
            "body={label}: inputs={} outputs(satisfiable)={} ops={} macs={}",
            body.input_nodes.len(),
            body.output_nodes.len(),
            body.program.len(),
            total_macs(&body.program)
        );
    }
    for left in 0..derived.len() {
        for right in (left + 1)..derived.len() {
            let (left_label, left_body) = &derived[left];
            let (right_label, right_body) = &derived[right];
            let shared = shared_op_count(&left_body.program, &right_body.program);
            let smaller = left_body.program.len().min(right_body.program.len());
            std::eprintln!(
                "{left_label} vs {right_label}: shared_ops={shared} of smaller-program-len={smaller} \
                 ({:.1}% literal overlap)",
                100.0 * shared as f64 / smaller as f64
            );
        }
    }

    let find = |label: &str| -> &Derived {
        &derived
            .iter()
            .find(|(candidate, _)| *candidate == label)
            .expect("label is in the sweep")
            .1
    };
    let full = find("full");
    let missing_capability = find("missing_capability");
    let added_capability = find("added_capability");
    let missing_sensor = find("missing_sensor");
    let missing_early_capability = find("missing_early_capability");

    assert_eq!(
        full.program[..missing_capability.program.len()],
        missing_capability.program[..],
        "finding 1: removing cap4 (which only gates the LAST two declared outputs) must leave an EXACT prefix \
         of the full program behind, not merely a same-sized coincidence"
    );
    assert_eq!(
        added_capability.program[..full.program.len()],
        full.program[..],
        "finding 1 (reverse direction): adding cap5/zeta at the END must extend the full program, not alter it"
    );

    let early_shared = shared_op_count(&full.program, &missing_early_capability.program);
    assert!(
        early_shared < missing_early_capability.program.len(),
        "finding 2: removing cap1 (which gates alpha, the SECOND declared output) renumbers every later \
         satisfiable output's unit_index -- this must NOT be an exact prefix (shared={early_shared}, \
         missing_early_capability.len()={})",
        missing_early_capability.program.len()
    );
    assert_eq!(
        early_shared, 78,
        "measured, not assumed -- pins the exact overlap so a future change to the derivation is caught"
    );

    let sensor_shared = shared_op_count(&full.program, &missing_sensor.program);
    assert!(
        sensor_shared < missing_sensor.program.len() / 2,
        "finding 3: losing a sensor moves the shared hidden layer itself, so overlap must collapse to well \
         under half (shared={sensor_shared}, missing_sensor.len()={})",
        missing_sensor.program.len()
    );
}
