//! Moving a retained [`BoundOp`] list to a new row count without binding again.
//!
//! Within one kernel class, binding the same program at two row counts gives the same ops
//! (same kinds, fusion bodies and operand nodes) whose integers (extents, strides, bases,
//! lookup extents) are affine in the row count. [`row_template`] proves that for the
//! reference binds it is handed and records each integer as `base + slope * (rows - base_rows)`;
//! [`apply_row_template`] evaluates it. A cached-attention op is also handed back: its counts
//! follow its operands shapes ([`refresh_cached_attention`]).

use alloc::vec::Vec;
use core::cell::{Cell, RefCell};

use super::refit_cached_attention::refresh_attention;
use super::types_layout_boundop::{BoundOp, BoundOpKind, BoundOperands, Layout, Lookup};
use crate::shape::Shapes;

type Fields = Vec<(i128, i128)>;

/// `(rows of the base bind, one entry per op)`.
pub type RowTemplate = (u64, Vec<Fields>);

type Unsigned<'visit> = &'visit mut dyn FnMut(&mut u64);
type Signed<'visit> = &'visit mut dyn FnMut(&mut i64);

/// [`refresh_attention`] for a caller outside this crate: the op moved to `next`'s shapes.
#[must_use]
pub fn refresh_cached_attention(bound: &BoundOp, next: &Shapes) -> Option<BoundOp> {
    refresh_attention(bound, next)
}

fn visit_layout(layout: &mut Layout, signed: Signed<'_>) {
    signed(&mut layout.base);
    layout.strides.iter_mut().for_each(signed);
}

fn visit_lookup(lookup: &mut Lookup, unsigned: Unsigned<'_>, signed: Signed<'_>) {
    visit_layout(&mut lookup.index_layout, signed);
    signed(&mut lookup.element_stride);
    unsigned(&mut lookup.extent);
}

fn visit_operands(operands: &mut BoundOperands, unsigned: Unsigned<'_>, signed: Signed<'_>) {
    for (_, layout, lookup) in operands {
        visit_layout(layout, signed);
        if let Some(lookup) = lookup {
            visit_lookup(lookup, unsigned, signed);
        }
    }
}

/// Every integer `bound` carries, in a fixed order; `false` for a kind this does not enumerate.
/// `counts` adds a cached-attention op's row and key counts, which follow its operands' shapes
/// rather than the row count alone, so only the structural comparison visits them.
fn visit_integers(bound: &mut BoundOp, counts: bool, unsigned: Unsigned<'_>, signed: Signed<'_>) -> bool {
    bound.extents.iter_mut().for_each(&mut *unsigned);
    match &mut bound.kind {
        BoundOpKind::Iota | BoundOpKind::Constant { .. } => true,
        BoundOpKind::Elementwise { operands, .. } | BoundOpKind::ElementwiseTwin { operands, .. } => {
            visit_operands(operands, unsigned, signed);
            true
        }
        BoundOpKind::Reduce {
            operands,
            out_layout,
            out_scatter,
            epilogue_operands,
            ..
        } => {
            visit_operands(operands, unsigned, signed);
            visit_layout(out_layout, signed);
            if let Some(lookup) = out_scatter {
                visit_lookup(lookup, unsigned, signed);
            }
            visit_operands(epilogue_operands, unsigned, signed);
            true
        }
        BoundOpKind::CachedAttention {
            operands,
            query_rows,
            cached_key_rows,
            new_key_rows,
            cached_lower_inclusive,
            new_upper_inclusive,
            ..
        } => {
            visit_operands(operands, unsigned, signed);
            if counts {
                [query_rows, cached_key_rows, new_key_rows]
                    .into_iter()
                    .for_each(unsigned);
            }
            signed(cached_lower_inclusive);
            signed(new_upper_inclusive);
            true
        }
        _ => false,
    }
}

/// Whether every op in `resolved` is a kind whose integers a [`RowTemplate`] can move.
#[must_use]
pub fn rows_enumerable(resolved: &[BoundOp]) -> bool {
    resolved.iter().all(|bound| {
        let mut copy = bound.clone();
        visit_integers(&mut copy, true, &mut |_| {}, &mut |_| {})
    })
}

fn integers(bound: &BoundOp) -> Option<Vec<i128>> {
    let values = RefCell::new(Vec::new());
    let mut copy = bound.clone();
    let enumerated = visit_integers(
        &mut copy,
        false,
        &mut |value| values.borrow_mut().push(i128::from(*value)),
        &mut |value| values.borrow_mut().push(i128::from(*value)),
    );
    enumerated.then(|| values.into_inner())
}

fn skeleton(bound: &BoundOp) -> Option<BoundOp> {
    let mut copy = bound.clone();
    visit_integers(&mut copy, true, &mut |value| *value = 0, &mut |value| *value = 0).then_some(copy)
}

fn is_attention(bound: &BoundOp) -> bool {
    matches!(bound.kind, BoundOpKind::CachedAttention { .. })
}

fn affine_columns(rows: &[u64], columns: &[Vec<i128>]) -> Option<Fields> {
    let base_rows = i128::from(rows[0]);
    let run = i128::from(rows[1]) - base_rows;
    (0..columns[0].len())
        .map(|field| {
            let start = columns[0][field];
            let delta = columns[1][field] - start;
            let slope = (delta % run == 0).then_some(delta / run)?;
            let holds = rows
                .iter()
                .zip(columns)
                .all(|(at, column)| column[field] == start + slope * (i128::from(*at) - base_rows));
            holds.then_some((start, slope))
        })
        .collect()
}

fn three_distinct_rows(rows: &[u64]) -> bool {
    let mut distinct = rows.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    rows.len() >= 3 && distinct.len() == rows.len()
}

fn node_columns(shapes: &Shapes) -> Vec<Vec<i128>> {
    shapes
        .extents()
        .iter()
        .map(|extents| extents.iter().map(|extent| i128::from(*extent)).collect())
        .collect()
}

/// The template that reproduces every [`Shapes`] in `binds` (each `(rows, shapes)`, the first being
/// the base), or `None` when fewer than three distinct row counts are given, the shape tables differ
/// in node count or rank, or an extent is not affine in the row count.
#[must_use]
pub fn shapes_template(binds: &[(u64, &Shapes)]) -> Option<RowTemplate> {
    let (base_rows, base) = binds.first()?;
    let rows: Vec<u64> = binds.iter().map(|(at, _)| *at).collect();
    let shaped = three_distinct_rows(&rows) && binds.iter().all(|(_, shapes)| shapes.extents().len() == base.extents().len());
    shaped.then_some(())?;
    let tables: Vec<Vec<Vec<i128>>> = binds.iter().map(|(_, shapes)| node_columns(shapes)).collect();
    let positions = (0..base.extents().len())
        .map(|node| {
            let columns: Vec<Vec<i128>> = tables.iter().map(|table| table[node].clone()).collect();
            let uniform = columns.iter().all(|column| column.len() == columns[0].len());
            uniform.then(|| affine_columns(&rows, &columns)).flatten()
        })
        .collect::<Option<Vec<_>>>()?;
    Some((*base_rows, positions))
}

/// The shapes a [`shapes_template`] gives at `rows`, or `None` when an extent leaves `u64`.
#[must_use]
pub fn apply_shapes_template(template: &RowTemplate, rows: u64) -> Option<Shapes> {
    let (base_rows, positions) = template;
    let delta = i128::from(rows) - i128::from(*base_rows);
    positions
        .iter()
        .map(|fields| {
            fields
                .iter()
                .map(|(start, slope)| u64::try_from(start + slope * delta).ok())
                .collect::<Option<Vec<u64>>>()
        })
        .collect::<Option<Vec<_>>>()
        .map(Shapes::from_extents)
}

fn fit(binds: &[(u64, &[BoundOp])], position: usize) -> Option<Fields> {
    let rows: Vec<u64> = binds.iter().map(|(at, _)| *at).collect();
    let reference = skeleton(&binds[0].1[position])?;
    let columns: Vec<Vec<i128>> = binds
        .iter()
        .map(|(_, ops)| {
            let same = skeleton(&ops[position])? == reference;
            same.then(|| integers(&ops[position])).flatten()
        })
        .collect::<Option<_>>()?;
    let uniform = columns.iter().all(|column| column.len() == columns[0].len());
    uniform.then(|| affine_columns(&rows, &columns)).flatten()
}

/// The template that reproduces every bind in `binds` (each `(rows, ops)`, the first being the
/// base), or `None` when fewer than three binds are given (two points cannot show affinity),
/// the lists differ in length, two binds share a row count, any op differs in more than its
/// integers, an integer is not affine in the row count, or a kind is not enumerated.
#[must_use]
pub fn row_template(binds: &[(u64, &[BoundOp])]) -> Option<RowTemplate> {
    let (base_rows, base) = binds.first()?;
    let rows: Vec<u64> = binds.iter().map(|(at, _)| *at).collect();
    let shaped = three_distinct_rows(&rows) && binds.iter().all(|(_, ops)| ops.len() == base.len());
    shaped.then_some(())?;
    let positions = (0..base.len())
        .map(|position| fit(binds, position))
        .collect::<Option<Vec<_>>>()?;
    Some((*base_rows, positions))
}

fn set_fields(bound: &mut BoundOp, fields: &Fields, delta: i128) -> bool {
    let index = Cell::new(0_usize);
    let valid = Cell::new(true);
    let next = || {
        let current = index.get();
        index.set(current + 1);
        fields.get(current).map(|(start, slope)| start + slope * delta)
    };
    let enumerated = visit_integers(
        bound,
        false,
        &mut |value| match next().map(u64::try_from) {
            Some(Ok(set)) => *value = set,
            _ => valid.set(false),
        },
        &mut |value| match next().map(i64::try_from) {
            Some(Ok(set)) => *value = set,
            _ => valid.set(false),
        },
    );
    enumerated && valid.get() && index.get() == fields.len()
}

/// Sets every integer of every enumerated op in `resolved` to its value at `rows`; returns the
/// cached-attention positions the caller refreshes against the new shapes, or `None` when `resolved` is not the
/// list the template was fitted to.
#[must_use]
pub fn apply_row_template(template: &RowTemplate, rows: u64, resolved: &mut [BoundOp]) -> Option<Vec<usize>> {
    let (base_rows, positions) = template;
    (resolved.len() == positions.len()).then_some(())?;
    let delta = i128::from(rows) - i128::from(*base_rows);
    let mut left = Vec::new();
    for (position, (bound, fields)) in resolved.iter_mut().zip(positions).enumerate() {
        set_fields(bound, fields, delta).then_some(())?;
        if is_attention(bound) {
            left.push(position);
        }
    }
    Some(left)
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::dtype::DType;
    use crate::map::{self, IndexMap};
    use crate::op::{Extent, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp, append};
    use crate::{NumericPolicy, bind_with_fusion, shape};

    type Check = Result<(), &'static str>;

    fn matmul_program() -> (Vec<Op>, NodeId) {
        let mut program = Vec::new();
        let input = |shape: Vec<Extent>| Op::Input {
            dtype: DType::FLOAT32,
            shape,
            name: None,
        };
        let lhs = append(&mut program, input(vec![Extent::Symbolic(0), Extent::Static(8)]));
        let rhs = append(&mut program, input(vec![Extent::Static(8), Extent::Static(4)]));
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                    (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
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
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        (program, sum)
    }

    fn shapes_at(program: &[Op], rows: u64) -> Result<Shapes, &'static str> {
        shape::infer(program, &[rows]).map_err(|_| "the program infers")
    }

    fn bind_at(program: &[Op], output: NodeId, rows: u64) -> Result<Vec<BoundOp>, &'static str> {
        let shapes = shapes_at(program, rows)?;
        bind_with_fusion(program, &shapes, &[output], true, NumericPolicy::bit_exact())
            .map_err(|_| "the program binds")
    }

    #[test]
    fn a_template_fitted_at_three_row_counts_reproduces_a_fourth_bind() -> Check {
        let (program, output) = matmul_program();
        let (base, next, between) = (
            bind_at(&program, output, 16)?,
            bind_at(&program, output, 40)?,
            bind_at(&program, output, 28)?,
        );
        let template = row_template(&[(16, &base), (40, &next), (28, &between)]).ok_or("every integer is affine in rows")?;

        let mut moved = base.clone();
        let attention = apply_row_template(&template, 123, &mut moved).ok_or("the template fits these ops")?;

        assert!(attention.is_empty());
        assert_eq!(moved, bind_at(&program, output, 123)?);
        Ok(())
    }

    #[test]
    fn two_binds_cannot_show_affinity() -> Check {
        let (program, output) = matmul_program();
        let (base, next) = (bind_at(&program, output, 16)?, bind_at(&program, output, 40)?);

        assert_eq!(row_template(&[(16, &base), (40, &next)]), None);
        Ok(())
    }

    #[test]
    fn a_bind_that_is_not_on_the_line_refuses_the_template() -> Check {
        let (program, output) = matmul_program();
        let (base, next) = (bind_at(&program, output, 16)?, bind_at(&program, output, 40)?);
        let mut off_line = bind_at(&program, output, 28)?;
        off_line[0].extents[0] += 1;

        assert_eq!(row_template(&[(16, &base), (40, &next), (28, &off_line)]), None);
        Ok(())
    }

    #[test]
    fn binds_of_different_structure_refuse_the_template() -> Check {
        let (program, output) = matmul_program();
        let (base, next, between) = (
            bind_at(&program, output, 16)?,
            bind_at(&program, output, 40)?,
            bind_at(&program, output, 28)?,
        );

        assert_eq!(row_template(&[(16, &base), (40, &next[..next.len() - 1]), (28, &between)]), None);
        Ok(())
    }

    #[test]
    fn a_shapes_template_reproduces_the_inferred_shapes_at_a_fourth_row_count() -> Check {
        let (program, _) = matmul_program();
        let (base, next, between) = (shapes_at(&program, 16)?, shapes_at(&program, 40)?, shapes_at(&program, 28)?);
        let template = shapes_template(&[(16, &base), (40, &next), (28, &between)]).ok_or("every extent is affine in rows")?;

        let moved = apply_shapes_template(&template, 123).ok_or("the template evaluates")?;

        assert_eq!(moved, shapes_at(&program, 123)?);
        Ok(())
    }

    #[test]
    fn a_shapes_template_needs_three_distinct_row_counts() -> Check {
        let (program, _) = matmul_program();
        let (base, next) = (shapes_at(&program, 16)?, shapes_at(&program, 40)?);

        assert_eq!(shapes_template(&[(16, &base), (40, &next)]), None);
        assert_eq!(shapes_template(&[(16, &base), (40, &next), (16, &base)]), None);
        Ok(())
    }
}
