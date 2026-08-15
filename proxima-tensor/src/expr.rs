//! A tensor program: a flat, self-contained sequence of expressions.
//!
//! There is no arena, no builder, no interning — a program is a plain
//! `Vec<Expr>`, and every `Expr` owns its own data. That is deliberate: a
//! partition pass that ships part of a program across a wire is then just a
//! sub-slice plus renumbering, not a walk of a separate side-table. Two rules
//! make a slice safe to consume without a validation pass of its own:
//!
//! - **References point backwards only.** A [`NodeId`] is a position in the
//!   slice; an `Expr` may only name positions built before it. That makes
//!   acyclicity an O(1) comparison per reference instead of a traversal.
//! - **The last element is the root**, by the same rule — nothing later can
//!   exist to consume it.
//!
//! Together these mean a linear scan of the slice is always a valid
//! topological order, which is what [`shape::infer`](crate::shape::infer) and
//! [`nest::lower`](crate::nest::lower) both rely on.

use alloc::string::String;
use alloc::vec::Vec;

use crate::dtype::DType;
use crate::map::IndexMap;

/// Index of an [`Expr`] in the program slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
pub struct NodeId(pub u32);

impl core::fmt::Display for NodeId {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "%{}", self.0)
    }
}

/// One dimension of a leaf's shape.
///
/// Symbolic extents are not an edge case: sequence length is unknown until a
/// request arrives, so a program that cannot express one cannot describe a
/// model that serves traffic. This is why rank and shape are values here and
/// not type parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "config", serde(rename_all = "snake_case"))]
pub enum Extent {
    Static(u32),
    Symbolic(u16),
}

/// Scalar body of a [`Expr::Zip`] or a [`Fold`].
///
/// Closed on purpose, and it is the one closed set in this crate that stays
/// closed: these are scalar machine primitives, not an extension point.
/// Composite activations desugar into several expressions — `gelu` is a
/// handful of zips, not a variant here — which costs expressions and buys a
/// vocabulary that never grows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "config", serde(rename_all = "snake_case"))]
pub enum ScalarOp {
    Identity,
    Add,
    Subtract,
    Multiply,
    Divide,
    Maximum,
    Minimum,
    Negate,
    Reciprocal,
    Exponential,
    Logarithm,
    SquareRoot,
    Tanh,
    Greater,
    Equal,
    Select,
}

impl ScalarOp {
    /// Operand count the body consumes. A zip whose operand count disagrees
    /// with this is malformed, and [`shape::infer`](crate::shape::infer) says
    /// so.
    #[must_use]
    pub const fn arity(self) -> usize {
        match self {
            Self::Identity
            | Self::Negate
            | Self::Reciprocal
            | Self::Exponential
            | Self::Logarithm
            | Self::SquareRoot
            | Self::Tanh => 1,
            Self::Add
            | Self::Subtract
            | Self::Multiply
            | Self::Divide
            | Self::Maximum
            | Self::Minimum
            | Self::Greater
            | Self::Equal => 2,
            Self::Select => 3,
        }
    }

    /// Whether folding with this body is order-independent. Only associative
    /// bodies may be reassociated into a tree or a parallel scan, so a
    /// scheduler reads this before choosing a reduction strategy.
    #[must_use]
    pub const fn is_associative(self) -> bool {
        matches!(
            self,
            Self::Add | Self::Multiply | Self::Maximum | Self::Minimum
        )
    }
}

/// Seed value for a fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "config", serde(rename_all = "snake_case"))]
pub enum FoldInit {
    Zero,
    One,
    NegativeInfinity,
    PositiveInfinity,
    /// Seed from the first element — the form `argmax` needs, since no
    /// synthetic identity exists for a `(value, index)` accumulator.
    FirstElement,
}

/// Which prefixes of a fold survive.
///
/// The only thing separating a reduction from a scan. Log-depth prefix sum is
/// a *scheduling* decision about a `Keep::All` fold, not a different
/// operation, which is why there is no separate `Scan` expression form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "config", serde(rename_all = "snake_case"))]
pub enum Keep {
    /// Reduce: only the final accumulator.
    Last,
    /// Scan: every prefix, preserving the folded extent.
    All,
}

/// A fold's parameters, named because eight of them travel together through
/// every pass and every backend.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
pub struct Fold {
    pub dtype: DType,
    pub body: ScalarOp,
    pub init: FoldInit,
    pub operand: NodeId,
    /// Addresses the operand from the iteration space.
    pub in_map: IndexMap,
    /// Addresses the result. Data-dependent here is what makes a scatter.
    pub out_map: IndexMap,
    pub keep: Keep,
    pub name: Option<String>,
}

/// The three generators.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
pub enum Expr {
    /// A leaf. Where data enters.
    ///
    /// `name` is identity, not decoration: positional binding does not
    /// survive a partition pass that renumbers the slice, and a cut edge in a
    /// distributed program delivers a tensor to a `Block` over a wire keyed
    /// by name, not by index. It is also how weights load by name. Local,
    /// single-partition evaluation still binds `blocks` positionally in
    /// [`cpu::evaluate`](crate::cpu::evaluate) — the name is what survives a
    /// partition that positional binding does not.
    Block {
        dtype: DType,
        shape: Vec<Extent>,
        name: Option<String>,
    },

    /// N-ary elementwise. Each operand carries its own index map, so arity 1
    /// with a permuting map is a transpose and an operand with a computed map
    /// is a gather.
    Zip {
        dtype: DType,
        body: ScalarOp,
        operands: Vec<(NodeId, IndexMap)>,
        name: Option<String>,
    },

    /// Reduce, scan, scatter, contraction and argmax, distinguished by
    /// [`Fold::keep`] and by whether [`Fold::out_map`] is data-dependent.
    Fold(Fold),
}

impl Expr {
    #[must_use]
    pub const fn dtype(&self) -> DType {
        match self {
            Self::Block { dtype, .. } | Self::Zip { dtype, .. } => *dtype,
            Self::Fold(fold) => fold.dtype,
        }
    }

    #[must_use]
    pub fn name(&self) -> Option<&str> {
        match self {
            Self::Block { name, .. } | Self::Zip { name, .. } => name.as_deref(),
            Self::Fold(fold) => fold.name.as_deref(),
        }
    }
}

/// Append an expression, returning the [`NodeId`] it can be referenced by.
/// The id-is-index idiom in one place: every push returns an id greater than
/// any the program already contains, which is what keeps references
/// backwards-only by construction rather than by a check.
pub fn append(program: &mut Vec<Expr>, expr: Expr) -> NodeId {
    let id = NodeId(program.len() as u32);
    program.push(expr);
    id
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::unary(ScalarOp::Exponential, 1)]
    #[case::negate(ScalarOp::Negate, 1)]
    #[case::binary(ScalarOp::Add, 2)]
    #[case::compare(ScalarOp::Greater, 2)]
    #[case::ternary(ScalarOp::Select, 3)]
    fn arity_matches_the_body(#[case] body: ScalarOp, #[case] expected: usize) {
        assert_eq!(body.arity(), expected);
    }

    #[rstest]
    #[case::add(ScalarOp::Add, true)]
    #[case::multiply(ScalarOp::Multiply, true)]
    #[case::maximum(ScalarOp::Maximum, true)]
    #[case::subtract(ScalarOp::Subtract, false)]
    #[case::divide(ScalarOp::Divide, false)]
    fn only_associative_bodies_may_be_reassociated(
        #[case] body: ScalarOp,
        #[case] associative: bool,
    ) {
        assert_eq!(body.is_associative(), associative);
    }

    #[test]
    fn node_id_displays_in_ssa_form() {
        extern crate alloc;
        use alloc::string::ToString;
        assert_eq!(NodeId(12).to_string(), "%12");
    }

    #[test]
    fn append_returns_increasing_ids() {
        let mut program = Vec::new();
        let first = append(
            &mut program,
            Expr::Block {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let second = append(
            &mut program,
            Expr::Block {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        assert_eq!(first.0, 0);
        assert_eq!(second.0, 1);
        assert!(
            second.0 > first.0,
            "ids increase, so references point backwards"
        );
    }

    #[test]
    fn dtype_and_name_read_through_every_variant() {
        let block = Expr::Block {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(4)],
            name: Some("x".into()),
        };
        assert_eq!(block.dtype(), DType::Float32);
        assert_eq!(block.name(), Some("x"));

        let fold = Expr::Fold(Fold {
            dtype: DType::Int32,
            body: ScalarOp::Add,
            init: FoldInit::Zero,
            operand: NodeId(0),
            in_map: IndexMap::Affine(crate::map::projection(1, &[0])),
            out_map: IndexMap::Affine(crate::map::projection(1, &[])),
            keep: Keep::Last,
            name: None,
        });
        assert_eq!(fold.dtype(), DType::Int32);
        assert_eq!(fold.name(), None);
    }
}
