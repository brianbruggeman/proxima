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
