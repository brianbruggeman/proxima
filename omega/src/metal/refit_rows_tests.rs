//! A plan refitted by rows must be what a fresh plan at the new row count is: the same plan-owned
//! uniform bytes, the same dispatch grids and the same executed output, bit for bit. The program is
//! a prefill-shaped projection chain (`rows x 64` activations through a `64 x 32` weight, then a
//! per-row scale), bound at 512 rows like a warmed prefill chunk plan.

use proxima_numeric::{DType, NumericPolicy};
use proxima_tensor::{Extent, IndexMap, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp, append, projection};

use super::refit_rows::refit_rows;
use super::*;

const WIDTH: usize = 64;
const OUTPUT: usize = 32;
const BUILD_ROWS: u64 = 512;

fn projection_program() -> (Vec<Op>, NodeId) {
    let mut program = Vec::new();
    let named = |dtype, shape, name: &str| Op::Input {
        dtype,
        shape,
        name: Some(name.into()),
    };
    let activations = append(
        &mut program,
        named(DType::FLOAT32, vec![Extent::Symbolic(0), Extent::Static(WIDTH as u32)], "x"),
    );
    let weight = append(
        &mut program,
        named(DType::FLOAT32, vec![Extent::Static(WIDTH as u32), Extent::Static(OUTPUT as u32)], "w"),
    );
    let product = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![
                (activations, IndexMap::Affine(projection(3, &[0, 2]))),
                (weight, IndexMap::Affine(projection(3, &[2, 1]))),
            ],
            name: None,
        },
    );
    let sum = append(
        &mut program,
        Op::Reduce(Reduce {
            dtype: DType::FLOAT32,
            body: ScalarOp::Add,
            init: ReduceInit::Zero,
            operand: product,
            in_map: IndexMap::Affine(projection(3, &[0, 1, 2])),
            out_map: IndexMap::Affine(projection(3, &[0, 1])),
            keep: Keep::Reduce,
            name: None,
        }),
    );
    let scaled = append(
        &mut program,
        Op::Elementwise {
            dtype: DType::FLOAT32,
            body: ScalarOp::Multiply,
            operands: vec![
                (sum, IndexMap::Affine(projection(2, &[0, 1]))),
                (sum, IndexMap::Affine(projection(2, &[0, 1]))),
            ],
            name: None,
        },
    );
    (program, scaled)
}

fn inputs(rows: u64) -> Vec<(String, Vec<f32>)> {
    let wave = |index: usize, salt: usize| ((index * 31 + salt * 17) % 211) as f32 / 211.0 - 0.5;
    let activations = (0..rows as usize * WIDTH).map(|index| wave(index, 3)).collect();
    let weight = (0..WIDTH * OUTPUT).map(|index| wave(index, 7)).collect();
    vec![("x".to_string(), activations), ("w".to_string(), weight)]
}

fn named(owned: &[(String, Vec<f32>)]) -> Vec<(&str, QuantizedBlock<'_>)> {
    owned
        .iter()
        .map(|(name, data)| (name.as_str(), QuantizedBlock::Float32(data.as_slice())))
        .collect()
}

fn plan_at(program: &[Op], output: NodeId, rows: u64) -> Plan {
    let owned = inputs(rows);
    plan_named(program, &[rows], &named(&owned), &[output], NumericPolicy::default())
        .expect("the projection chain plans")
}

fn run_bits(plan: &Plan, output: NodeId, rows: u64) -> Vec<u32> {
    let owned = inputs(rows);
    let evaluated = execute_plan_named_with_placements(plan, &named(&owned), &[], &[])
        .expect("the plan executes through the placement executor");
    let (data, _) = evaluated.get(output).expect("the output is returned");
    data.iter().map(|value| value.to_bits()).collect()
}

fn uniform_bytes(plan: &Plan) -> Vec<Vec<u8>> {
    let uniforms = plan.uniforms.get().expect("an executed plan owns its uniforms");
    uniforms
        .buffers
        .iter()
        .map(|buffer| read_back_uniform_bytes(buffer, buffer.length()))
        .collect()
}

fn grids(plan: &Plan) -> Vec<crate::GridSpec> {
    let steps = plan.resolved_steps.borrow();
    let resolved = steps.as_ref().expect("an executed plan has resolved steps");
    resolved.steps.iter().map(|step| step.grid).collect()
}

#[test]
fn a_512_row_plan_refits_to_each_row_count_as_a_fresh_plan_at_that_count() {
    let (program, output) = projection_program();
    for rows in [16_u64, 256, 487, 511] {
        let mut refit = plan_at(&program, output, BUILD_ROWS);
        run_bits(&refit, output, BUILD_ROWS);
        let refitted = refit_rows(&mut refit, &[rows]).expect("the refit resolves");
        assert!(refitted, "{rows} rows are in the wide class and fit the 512-row arena");

        let fresh = plan_at(&program, output, rows);
        let refit_bits = run_bits(&refit, output, rows);
        let fresh_bits = run_bits(&fresh, output, rows);

        assert_eq!(refit.prepared.resolved, fresh.prepared.resolved, "{rows} rows: bound ops");
        assert_eq!(uniform_bytes(&refit), uniform_bytes(&fresh), "{rows} rows: uniform bytes");
        assert_eq!(grids(&refit), grids(&fresh), "{rows} rows: dispatch grids");
        assert_eq!(refit_bits.len(), rows as usize * OUTPUT, "{rows} rows: output size");
        assert_eq!(refit_bits, fresh_bits, "{rows} rows: executed output");
    }
}

#[test]
fn a_cold_512_row_plan_refits_to_487_rows_as_a_fresh_plan_does() {
    let (program, output) = projection_program();
    let mut refit = plan_at(&program, output, BUILD_ROWS);

    assert!(refit_rows(&mut refit, &[487]).expect("the refit resolves"));

    assert_eq!(run_bits(&refit, output, 487), run_bits(&plan_at(&program, output, 487), output, 487));
}

#[test]
fn a_refit_plan_moves_back_up_to_the_width_its_arena_was_built_for() {
    let (program, output) = projection_program();
    let mut plan = plan_at(&program, output, BUILD_ROWS);
    run_bits(&plan, output, BUILD_ROWS);

    assert!(refit_rows(&mut plan, &[256]).expect("down"));
    assert!(refit_rows(&mut plan, &[BUILD_ROWS]).expect("back up"));

    assert_eq!(
        run_bits(&plan, output, BUILD_ROWS),
        run_bits(&plan_at(&program, output, BUILD_ROWS), output, BUILD_ROWS)
    );
}

#[test]
fn rows_above_the_build_width_or_below_the_wide_class_are_not_refit() {
    let (program, output) = projection_program();
    for rows in [513_u64, 1024, 15, 1] {
        let mut plan = plan_at(&program, output, BUILD_ROWS);
        run_bits(&plan, output, BUILD_ROWS);
        let before = plan.prepared.resolved.clone();

        let refitted = plan.refit_symbols(&[rows]).expect("a declined refit is not an error");

        assert!(!refitted, "{rows} rows are outside 16..=512");
        assert_eq!(plan.prepared.resolved, before, "{rows} rows: the plan is untouched");
    }
}

#[test]
fn priming_the_row_template_leaves_the_plan_as_built_and_caches_what_a_refit_learns() {
    let (program, output) = projection_program();
    let mut primed = plan_at(&program, output, BUILD_ROWS);
    run_bits(&primed, output, BUILD_ROWS);
    let (ops, uniforms, step_grids) = (primed.prepared.resolved.clone(), uniform_bytes(&primed), grids(&primed));
    assert_eq!(primed.row_template_verdict(), None, "a built plan has not learned a template");

    assert!(primed.prime_row_template(&[BUILD_ROWS]).expect("priming binds"));

    assert_eq!(primed.row_template_verdict(), Some(true), "the template is cached before any refit");
    assert_eq!(primed.bound_symbols, vec![BUILD_ROWS], "priming does not move the plan");
    assert_eq!(primed.prepared.resolved, ops, "priming leaves the bound ops");
    assert_eq!(uniform_bytes(&primed), uniforms, "priming leaves the uniforms");
    assert_eq!(grids(&primed), step_grids, "priming leaves the grids");

    let mut learned_by_refit = plan_at(&program, output, BUILD_ROWS);
    run_bits(&learned_by_refit, output, BUILD_ROWS);
    assert!(refit_rows(&mut learned_by_refit, &[487]).expect("the refit resolves"));
    assert_eq!(
        primed.rows_template.borrow().as_ref().expect("primed").as_ref().expect("affine"),
        learned_by_refit.rows_template.borrow().as_ref().expect("learned").as_ref().expect("affine"),
        "priming fits the template the first refit would"
    );
}

#[test]
fn a_primed_512_row_plan_refits_to_487_rows_as_a_fresh_plan_does_without_learning() {
    let (program, output) = projection_program();
    let mut plan = plan_at(&program, output, BUILD_ROWS);
    run_bits(&plan, output, BUILD_ROWS);
    assert!(plan.prime_row_template(&[BUILD_ROWS]).expect("priming binds"));
    let cached = plan.row_template_verdict();

    assert!(refit_rows(&mut plan, &[487]).expect("the refit resolves"));

    assert_eq!(cached, Some(true), "the refit started from a cached template");
    let fresh = plan_at(&program, output, 487);
    let refit_bits = run_bits(&plan, output, 487);
    let fresh_bits = run_bits(&fresh, output, 487);
    assert_eq!(plan.prepared.resolved, fresh.prepared.resolved, "bound ops");
    assert_eq!(uniform_bytes(&plan), uniform_bytes(&fresh), "uniform bytes");
    assert_eq!(refit_bits, fresh_bits, "executed output");
}

#[test]
fn a_plan_outside_the_wide_class_primes_nothing() {
    let (program, output) = projection_program();
    let mut narrow = plan_at(&program, output, 8);

    assert!(!narrow.prime_row_template(&[8]).expect("a declined prime is not an error"));
    assert_eq!(narrow.row_template_verdict(), None);
}
