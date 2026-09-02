//! The index-pattern grammar — where this crate's expressiveness actually
//! lives.
//!
//! Three expression forms carry the algebra, but they only work because an
//! operand can be *related* to the iteration space by something richer than a
//! permutation. Every shape operation is an index pattern, not a variant:
//!
//! | operation | pattern |
//! |---|---|
//! | transpose | permute the projected iteration axes |
//! | broadcast | project fewer axes than the iteration space has |
//! | slice | a non-zero `offset` |
//! | stride / dilation | a `coeff` other than 1 |
//! | convolution | two terms in one axis: `h*stride + r*dilation` |
//! | gather (read-side) | [`IndexMap::Computed`] — one axis's index comes from a node |
//!
//! Convolution is why an axis is a *sum* of terms rather than a single
//! projection. Without that, windowed access needs its own expression form,
//! and the three generators become four, then a dozen.
//!
//! A pattern owns its axes and terms directly — there is no interned,
//! span-based arena here, because [`Op`](crate::op::Op) itself no
//! longer lives in one: a tensor program is a plain `Vec<Op>`, and each
//! `Op` is self-contained.

use alloc::vec::Vec;

use smallvec::SmallVec;

use crate::op::NodeId;

/// Inline capacity for one axis's term list. Every axis this crate builds
/// today has 1 term (a plain projection) or 2 (convolution's `stride +
/// dilation`, the doc table above) — no construction site anywhere in the
/// crate ever exceeds 2. `SmallVec` spills past this on a wider pattern
/// instead of truncating it, so a caller that legitimately needs more still
/// gets a correct (just heap-backed) result.
pub const MAX_INLINE_TERMS: usize = 2;

/// One `coeff * iter[axis]` contribution to an operand index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
pub struct AxisTerm {
    pub axis: u16,
    pub coeff: i32,
}

impl AxisTerm {
    #[must_use]
    pub const fn projection(axis: u16) -> Self {
        Self { axis, coeff: 1 }
    }

    #[must_use]
    pub const fn scaled(axis: u16, coeff: i32) -> Self {
        Self { axis, coeff }
    }
}

/// One operand axis, as `sum(terms) + offset` over the iteration space.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
pub struct AxisIndex {
    pub terms: SmallVec<[AxisTerm; MAX_INLINE_TERMS]>,
    pub offset: i32,
}

/// Relates an iteration space of rank `iter_rank` to an operand's index space.
///
/// `axes` holds one [`AxisIndex`] per operand axis, so the operand's rank is
/// `axes.len()` — which may be lower than `iter_rank` (a broadcast) or
/// reorder it (a transpose).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
pub struct IndexPattern {
    pub iter_rank: u16,
    pub axes: Vec<AxisIndex>,
}

/// How an operand is addressed. `Affine` is statically analysable and is what
/// shape inference and op building reason about; `Computed` is a
/// data-dependent index and is the reason gather needs no expression form of
/// its own.
///
/// A gather touches exactly one operand axis — the one the fetched index
/// selects (`gathered_dim`) — while the rest stay affine. `index_map` and
/// `base` are both drawn from the *same* iteration space: `index_map` says
/// how to read the `indices` tensor from it, `base` says how the operand's
/// other axes read from it. `base`'s entry at `gathered_dim` is never
/// consulted (its `terms` are empty by construction) because that axis's
/// address comes from the fetched index at evaluation time, not from
/// iteration.
///
/// `index_map` is always a plain [`IndexPattern`], never another `IndexMap`
/// — the indices tensor itself must be affinely addressed, so a gather
/// cannot nest inside another gather. This is enforced by the type, not a
/// runtime check.
///
/// `indices` is an integer [`crate::dtype::DType`] logically, but every
/// backend this crate ships (`cpu.rs`'s interpreter, `omega`'s Metal driver)
/// carries every buffer as f32 — including `indices` — rather than plumbing
/// a second integer-buffer kind through the stack for this one case. f32's
/// 24-bit mantissa represents every integer up to `2^24` (16,777,216)
/// exactly, so [`crate::shape::infer`] rejects any gathered axis wider than
/// that (`TensorError::GatherExtentExceedsExactFloat`) rather than let a
/// fetched index silently lose precision and select the wrong row. Lifting
/// the ceiling means adding real integer buffers, not raising a constant.
///
/// Scatter — a data-dependent *output* map on a [`crate::op::Reduce`] — is
/// the write-side twin of gather, using this exact same variant: `Reduce`'s
/// own `body`/`init` are the fold a colliding write reduces with (this crate
/// runs the CPU interpreter's reduce loop strictly sequentially, so a
/// scatter never needs atomics — see `cpu.rs`'s `run_reduce_scatter`), and
/// `indices`/`index_map` name where each iteration step's destination
/// address comes from, exactly as they do for a read.
///
/// One convention is specific to the write direction: `base`'s entry at
/// `gathered_dim` is unaddressable either way (its `terms` stay empty by
/// construction, same as a gather), but a gather has nothing else to say
/// about that axis while a scatter must state its output extent somewhere —
/// the destination's *shape* is static even though its *addressing* is not,
/// and nothing else in this program can supply that extent (it is not any
/// existing node's shape; see `shape.rs`'s `infer_reduce` doc for why a
/// `Reduce`-wide field was rejected on blast-radius grounds). So for a
/// scatter specifically, that otherwise-always-`0` `offset` carries the
/// destination axis's static extent instead — the one place in this type an
/// unused field's bit pattern is deliberately repurposed by context. See
/// [`IndexMap::scatter_extent`]/[`IndexMap::scatter`] for the one pair of
/// accessors that own this convention, so nothing outside `map.rs` reads or
/// writes `base`'s `gathered_dim` offset directly.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "config", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "config", serde(rename_all = "snake_case"))]
pub enum IndexMap {
    Affine(IndexPattern),
    Computed {
        /// The node supplying fetched index values; must be a backwards
        /// reference with an integer [`crate::dtype::DType`].
        indices: NodeId,
        /// How the iteration space addresses the `indices` tensor.
        index_map: IndexPattern,
        /// How the iteration space addresses the operand's non-gathered
        /// axes; the entry at `gathered_dim` is unused for a *read*
        /// (`terms` empty, `offset` `0`). For a `Reduce`'s `out_map` (a
        /// scatter), that same entry's `offset` instead carries the
        /// destination axis's static extent — see
        /// [`IndexMap::scatter_extent`].
        base: IndexPattern,
        /// Which operand axis the fetched index selects.
        gathered_dim: u16,
    },
}

impl IndexMap {
    #[must_use]
    pub const fn affine(&self) -> &IndexPattern {
        match self {
            Self::Affine(pattern) | Self::Computed { base: pattern, .. } => pattern,
        }
    }

    #[must_use]
    pub const fn is_data_dependent(&self) -> bool {
        matches!(self, Self::Computed { .. })
    }

    /// Builds a scatter `out_map`: a [`Self::Computed`] whose `base` carries
    /// `destination_extent` at `gathered_dim` via the convention this type's
    /// own doc names, and empty terms/offset `0` everywhere the caller's
    /// `non_scattered` axes don't otherwise fill in. `non_scattered` is one
    /// [`AxisIndex`] per output axis other than `gathered_dim`, in output
    /// axis order with `gathered_dim`'s own slot omitted — mirroring how
    /// `crate::bind`'s `pure_projection_axes` (private) reads the result back out.
    #[must_use]
    pub fn scatter(
        indices: NodeId,
        index_map: IndexPattern,
        iter_rank: u16,
        non_scattered: &[(u16, AxisIndex)],
        gathered_dim: u16,
        destination_extent: u32,
    ) -> Self {
        let rank = non_scattered.len() + 1;
        let mut axes = alloc::vec![AxisIndex::default(); rank];
        for (axis_index, axis) in non_scattered {
            axes[*axis_index as usize] = axis.clone();
        }
        axes[gathered_dim as usize] = AxisIndex {
            terms: SmallVec::new(),
            offset: destination_extent as i32,
        };
        Self::Computed {
            indices,
            index_map,
            base: IndexPattern { iter_rank, axes },
            gathered_dim,
        }
    }

    /// The static destination extent a scatter `out_map` carries at
    /// `gathered_dim` — `None` for an `Affine` map or a `Computed` map used
    /// as a *read* (a gather), where that slot is always `0` and means
    /// nothing. Negative is malformed (a caller error, not a runtime one);
    /// callers needing a validated `u64` go through
    /// [`crate::shape::ShapeTable`] instead, which is where that check
    /// actually lives (see `infer_reduce`'s own doc for why: the ceiling is
    /// the same `2^24` exact-float bound a gather's extent already needs).
    #[must_use]
    pub fn scatter_extent(&self) -> Option<i32> {
        match self {
            Self::Affine(_) => None,
            Self::Computed {
                base, gathered_dim, ..
            } => {
                if (*gathered_dim as usize) < base.axes.len() {
                    Some(base.axes[*gathered_dim as usize].offset)
                } else {
                    None
                }
            }
        }
    }

    /// The read-side counterpart of a scatter `out_map`: same `indices`,
    /// `index_map`, `gathered_dim`, and every non-scattered `base` axis, but
    /// `gathered_dim`'s own entry reset to the ordinary gather convention
    /// (`terms` empty, `offset` `0`) instead of carrying the destination
    /// extent — a scatter's `base` cannot be read with directly, since
    /// `layout_of` would fold that extent in as a real address contribution
    /// (`bind::build_scatter_out_layout`'s own doc has the full reasoning).
    ///
    /// This is exactly what a scatter's adjoint needs: `grad_out` gathered
    /// at the same destination each forward source position wrote to (a
    /// `Reduce(Add)` scatter's own gradient rule — see
    /// `proxima_autograd::adjoint`'s `differentiate_reduce`). `None` for an
    /// `Affine` map, where there is no `base`/`gathered_dim` to reuse.
    #[must_use]
    pub fn as_gather_from_output(&self) -> Option<Self> {
        match self {
            Self::Affine(_) => None,
            Self::Computed {
                indices,
                index_map,
                base,
                gathered_dim,
            } => {
                if *gathered_dim as usize >= base.axes.len() {
                    return None;
                }
                let mut base = base.clone();
                base.axes[*gathered_dim as usize] = AxisIndex::default();
                Some(Self::Computed {
                    indices: *indices,
                    index_map: index_map.clone(),
                    base,
                    gathered_dim: *gathered_dim,
                })
            }
        }
    }
}

/// A plain projection: operand axis `n` reads iteration axis `projected[n]`.
/// Covers transpose, broadcast, and identity — the overwhelming majority.
#[must_use]
pub fn projection(iter_rank: u16, projected: &[u16]) -> IndexPattern {
    let axes = projected
        .iter()
        .map(|axis| AxisIndex {
            terms: core::iter::once(AxisTerm::projection(*axis)).collect(),
            offset: 0,
        })
        .collect();
    IndexPattern { iter_rank, axes }
}

/// A general affine index pattern: one `(terms, offset)` pair per operand
/// axis. What a projection cannot express — convolution windows, dilation,
/// slices with a stride — goes through this constructor instead.
#[must_use]
pub fn affine(iter_rank: u16, axes: &[(&[AxisTerm], i32)]) -> IndexPattern {
    let axes = axes
        .iter()
        .map(|(terms, offset)| AxisIndex {
            terms: terms.iter().copied().collect(),
            offset: *offset,
        })
        .collect();
    IndexPattern { iter_rank, axes }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn projection_has_unit_coefficient_and_scaled_does_not() {
        assert_eq!(AxisTerm::projection(2).coeff, 1);
        assert_eq!(AxisTerm::scaled(2, 3).coeff, 3);
        assert_eq!(AxisTerm::scaled(2, 3).axis, 2);
    }

    #[test]
    fn computed_and_affine_both_expose_a_base_pattern() {
        let base = IndexPattern {
            iter_rank: 2,
            axes: alloc::vec![AxisIndex {
                terms: core::iter::once(AxisTerm::projection(0)).collect(),
                offset: 0,
            }],
        };
        let index_map = projection(2, &[1]);
        let direct = IndexMap::Affine(base.clone());
        let gathered = IndexMap::Computed {
            indices: NodeId(7),
            index_map,
            base: base.clone(),
            gathered_dim: 0,
        };
        assert_eq!(*direct.affine(), base);
        assert_eq!(*gathered.affine(), base);
        assert!(!direct.is_data_dependent());
        assert!(gathered.is_data_dependent());
    }

    #[test]
    fn a_convolution_axis_is_two_terms() {
        // h*stride + r*dilation is the whole reason AxisIndex sums terms.
        let window = affine(
            2,
            &[(&[AxisTerm::scaled(0, 2), AxisTerm::scaled(1, 1)], -1)],
        );
        assert_eq!(window.axes[0].terms.len(), 2);
        assert_eq!(window.axes[0].terms[0].coeff, 2, "stride");
        assert_eq!(window.axes[0].terms[1].coeff, 1, "dilation");
        assert_eq!(window.axes[0].offset, -1, "padding is the offset");
    }

    #[test]
    fn projection_reads_transpose_broadcast_and_identity() {
        let identity = projection(2, &[0, 1]);
        let transpose = projection(2, &[1, 0]);
        let broadcast = projection(2, &[1]);
        assert_eq!(identity.axes.len(), 2);
        assert_eq!(transpose.axes[0].terms[0].axis, 1);
        assert_eq!(
            broadcast.axes.len(),
            1,
            "broadcast projects fewer axes than iter_rank"
        );
    }
}
