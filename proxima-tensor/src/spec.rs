//! The configuration face: a tensor program as TOML, and the conversion into
//! a `Vec<Op>`.
//!
//! This module exists to *force* a property rather than to claim one. If a
//! program can be written as data, adding an operation to a model is a
//! config edit; if it cannot, the claim that this algebra is describable was
//! never tested. The round-trip test at the bottom is that test — the same
//! matmul built in Rust and parsed from TOML must produce an equal `Vec<Op>`.
//!
//! Index patterns are written in `operand->iteration` notation, which reads
//! like einsum: `ik->ijk` says the operand has axes `i,k` drawn from an
//! iteration space of `i,j,k`. That covers projection, transpose, and
//! broadcast — the overwhelming majority.
//!
//! An operand axis may also be a comma-separated *expression*, one term per
//! axis: `"s,2*i->si"` says axis 0 is plain `s` and axis 1 is `2*i` — the
//! `AxisTerm { axis, coeff }` sum [`map::affine`] already
//! builds in Rust, spelled as data. A term is `[coeff*]letter` (letters stay
//! single ASCII characters, the same alphabet the bare-letter grammar uses),
//! several terms may be summed with `+`/`-`, and a bare integer term
//! contributes to the offset instead of a coefficient: `"2*h+r-1"` is a
//! stride-2, dilation-1 convolution window with padding folded into the
//! offset, `"2*i+1"` is RoPE's odd half of a pair. The comma is the trigger —
//! without one, the operand is still the old bare letter run (`ik->ijk`), so
//! no existing spelling changes meaning. This is parsing only: [`AxisIndex`]
//! and [`AxisTerm`] already expressed every one of these patterns before
//! this module could spell them.
//!
//! A [`NodeSpec::Reduce`]'s `in_map` reads through this same richer grammar
//! (`parse_operand_pattern`) — the asymmetry where only `Elementwise`
//! operands could spell a multi-term axis was an oversight, not a design
//! decision, since a `Reduce`'s operand is windowed exactly the same way a
//! convolution's `Elementwise(Multiply)` operand is (see
//! `specs/conv2d.toml`). `out_map` stays on the older, bare-letter-only
//! `parse_projection` deliberately: `shape::project_output_shape` already
//! rejects any `out_map` axis that is not a pure single-term `coeff == 1`
//! projection (`NotLowerable`, "reduce output maps must be pure projections
//! in v1"), so parsing a richer `out_map` would only ever be thrown away at
//! bind time — `parse_projection`'s narrower grammar gives the same
//! rejection at parse time instead, before a spec that could never lower
//! reaches shape inference at all.
//!
//! A [`NodeSpec::Elementwise`] operand map may instead be a
//! [`MapSpec::Gather`] table: `{ gather = "ids", index_map = "s->sd", map =
//! "d->sd", dim = 0 }`. `index_map` addresses the `gather` node the same
//! einsum way; `map` addresses the operand's *non-gathered* axes only, in
//! operand-axis order, skipping the position `dim` names —
//! `build_base_pattern` splices an empty (gathered) entry back in at that
//! position.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};
use serde::{Deserialize, Serialize};

use crate::dtype::DType;
use crate::error::TensorError;
#[cfg(feature = "instrument")]
use crate::instrument;
use crate::map::{self, AxisIndex, AxisTerm, IndexMap, IndexPattern};
use crate::op::{self, Extent, Keep, NodeId, Op, Reduce, ReduceInit, ScalarOp};

/// A declarative tensor program. Nodes are order-dependent: a node may only
/// reference ids defined above it, which mirrors the program's
/// backwards-reference rule so the two representations cannot disagree.
#[derive(Debug, Clone, Default, PartialEq, Builder, Deserialize, Serialize, Settings)]
#[settings(prefix = "TENSOR")]
#[builder(derive(Clone, Debug))]
pub struct ProgramSpec {
    #[setting(skip)]
    #[serde(default)]
    #[builder(default)]
    pub node: Vec<NodeSpec>,
}

/// One dimension of a leaf. A bare integer is static; `"?0"` is the zeroth
/// symbolic extent, which is how sequence length is written.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ExtentSpec {
    Static(u32),
    Symbolic(String),
}

impl ExtentSpec {
    fn resolve(&self) -> Result<Extent, TensorError> {
        match self {
            Self::Static(size) => Ok(Extent::Static(*size)),
            Self::Symbolic(text) => text
                .strip_prefix('?')
                .and_then(|rest| rest.parse::<u16>().ok())
                .map(Extent::Symbolic)
                .ok_or_else(|| TensorError::MalformedExtent(text.clone())),
        }
    }
}

/// One [`NodeSpec::Elementwise`] operand map: the existing bare
/// `operand->iteration` string, or a table describing a gather.
/// `#[serde(untagged)]` picks the variant from shape alone — a string is
/// `Projection`, a table is `Gather`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum MapSpec {
    Projection(String),
    Gather {
        /// The id of the node supplying fetched index values.
        gather: String,
        /// How the iteration space addresses the `gather` node.
        index_map: String,
        /// How the iteration space addresses the operand's non-gathered
        /// axes, in operand-axis order, skipping `dim`'s position.
        map: String,
        /// Which operand axis the fetched index selects.
        dim: u16,
    },
}

/// One node, discriminated by `op` so TOML reads as `op = "elementwise"`.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum NodeSpec {
    Input {
        id: String,
        dtype: DType,
        shape: Vec<ExtentSpec>,
        #[serde(default)]
        name: Option<String>,
    },
    Elementwise {
        id: String,
        dtype: DType,
        body: ScalarOp,
        inputs: Vec<String>,
        maps: Vec<MapSpec>,
        #[serde(default)]
        name: Option<String>,
    },
    Reduce {
        id: String,
        dtype: DType,
        body: ScalarOp,
        init: ReduceInit,
        input: String,
        in_map: String,
        out_map: String,
        keep: Keep,
        #[serde(default)]
        name: Option<String>,
    },
    /// `Op::Iota`'s config face: a leaf that produces `0, 1, 2, ...` up to
    /// `extent`, spelled with the same [`ExtentSpec`] grammar `Input.shape`
    /// entries use. No `name` field — [`Op::Iota`] carries none (see that
    /// variant's own doc for why).
    Iota {
        id: String,
        dtype: DType,
        extent: ExtentSpec,
    },
    /// [`Op::Constant`]'s config face: a leaf whose every element is
    /// `value`, spelled with the same [`ExtentSpec`] grammar `Input.shape`
    /// entries use. `shape = []` is the rank-0 form that broadcasts into any
    /// consumer. No `name` field — [`Op::Constant`] carries none.
    Constant {
        id: String,
        dtype: DType,
        shape: Vec<ExtentSpec>,
        value: f32,
    },
}

impl NodeSpec {
    #[must_use]
    pub fn id(&self) -> &str {
        match self {
            Self::Input { id, .. }
            | Self::Elementwise { id, .. }
            | Self::Reduce { id, .. }
            | Self::Iota { id, .. }
            | Self::Constant { id, .. } => id,
        }
    }
}

/// Parse `operand->iteration` into an iteration rank and one projected
/// iteration axis per operand axis.
fn parse_projection(notation: &str) -> Result<(u16, Vec<u16>), TensorError> {
    let (operand, iteration) = notation
        .split_once("->")
        .ok_or_else(|| TensorError::MalformedMap(notation.to_string()))?;
    let space: Vec<char> = iteration.chars().collect();
    let projected = operand
        .chars()
        .map(|letter| find_axis(&space, letter, notation))
        .collect::<Result<Vec<u16>, TensorError>>()?;
    Ok((space.len() as u16, projected))
}

/// Position of `letter` in the iteration space, or the same
/// [`TensorError::UnknownIndexLetter`] every notation parser raises for it.
fn find_axis(space: &[char], letter: char, notation: &str) -> Result<u16, TensorError> {
    space
        .iter()
        .position(|candidate| *candidate == letter)
        .map(|found| found as u16)
        .ok_or_else(|| TensorError::UnknownIndexLetter {
            notation: notation.to_string(),
            letter,
        })
}

/// Parse `operand->iteration` into a full [`IndexPattern`], where an operand
/// axis is either the legacy bare letter (`ik->ijk`, one axis per character)
/// or, once the operand side contains a comma, one axis-expression per
/// comma-separated term (`"s,2*i->si"`). The comma is what selects the
/// richer grammar, so a legacy notation with no comma parses identically to
/// [`parse_projection`] and every existing spelling is unaffected.
fn parse_operand_pattern(notation: &str) -> Result<IndexPattern, TensorError> {
    let (operand, iteration) = notation
        .split_once("->")
        .ok_or_else(|| TensorError::MalformedMap(notation.to_string()))?;
    let space: Vec<char> = iteration.chars().collect();
    let axes = if operand.contains(',') {
        operand
            .split(',')
            .map(|token| parse_axis_expr(token, &space, notation))
            .collect::<Result<Vec<AxisIndex>, TensorError>>()?
    } else {
        operand
            .chars()
            .map(|letter| {
                let axis = find_axis(&space, letter, notation)?;
                Ok(AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(axis)).collect(),
                    offset: 0,
                    len: None,
                })
            })
            .collect::<Result<Vec<AxisIndex>, TensorError>>()?
    };
    Ok(IndexPattern {
        iter_rank: space.len() as u16,
        axes,
    })
}

/// One comma-separated axis expression: a sum of `[coeff*]letter` terms and
/// bare-integer constants, e.g. `2*i+1`. Constants fold into `offset` rather
/// than becoming a term, since [`AxisTerm`] only carries a coefficient over
/// an iteration axis.
///
/// A trailing `@length` (`"i@4"`) states [`AxisIndex::len`] directly: the
/// axis's true iteration extent, when it is narrower than the operand's own
/// width at this position. This is the address fact (`i`, everything before
/// `@`) and the extent fact (`4`) spelled as two distinct pieces of syntax
/// instead of one doing double duty — see [`AxisIndex`]'s own doc for why
/// that distinction is the fix, not the arithmetic.
fn parse_axis_expr(token: &str, space: &[char], notation: &str) -> Result<AxisIndex, TensorError> {
    let (address, len) = match token.split_once('@') {
        Some((address, length)) => {
            let length: u32 = length
                .parse()
                .map_err(|_| TensorError::MalformedMap(notation.to_string()))?;
            (address, Some(Extent::Static(length)))
        }
        None => (token, None),
    };

    let mut terms: Vec<AxisTerm> = Vec::new();
    let mut offset: i32 = 0;
    for (sign, part) in split_signed_terms(address) {
        if let Some((coeff_text, letter_text)) = part.split_once('*') {
            let coeff: i32 = coeff_text
                .parse()
                .map_err(|_| TensorError::MalformedMap(notation.to_string()))?;
            let axis = find_axis(space, single_letter(letter_text, notation)?, notation)?;
            terms.push(AxisTerm::scaled(axis, sign * coeff));
        } else if let Ok(constant) = part.parse::<i32>() {
            offset += sign * constant;
        } else {
            let axis = find_axis(space, single_letter(part, notation)?, notation)?;
            terms.push(AxisTerm::scaled(axis, sign));
        }
    }
    if terms.is_empty() {
        return Err(TensorError::MalformedMap(notation.to_string()));
    }
    let axis = AxisIndex {
        terms: terms.into_iter().collect(),
        offset,
        len,
    };
    // A `len` this crate cannot honor is a malformed map, not an accepted-
    // and-silently-ignored one: `len_target_axis` names the same
    // single-plain-term rule `shape::unify_iteration_space` resolves `len`
    // against, so a notation like `i+j@2` (two `coeff == 1` terms) rejects
    // here instead of shipping a program shape inference would either
    // silently mis-resolve or reject far from this call site.
    if axis.len.is_some() && axis.len_target_axis().is_none() {
        return Err(TensorError::MalformedMap(notation.to_string()));
    }
    Ok(axis)
}

/// A term's letter, rejecting anything but exactly one ASCII lowercase
/// character — the same alphabet the legacy bare-letter grammar uses.
fn single_letter(text: &str, notation: &str) -> Result<char, TensorError> {
    let mut chars = text.chars();
    match (chars.next(), chars.next()) {
        (Some(letter), None) if letter.is_ascii_lowercase() => Ok(letter),
        _ => Err(TensorError::MalformedMap(notation.to_string())),
    }
}

/// Splits `2*i+1` into `[(1, "2*i"), (1, "1")]` and `d-1` into
/// `[(1, "d"), (-1, "1")]` — a `+`/`-` not at position 0 starts a new signed
/// part. There is no unary-minus support (no term ever starts with `-`)
/// because nothing built by this module needs a negative coefficient.
fn split_signed_terms(token: &str) -> Vec<(i32, &str)> {
    let mut parts = Vec::new();
    let mut sign = 1;
    let mut start = 0;
    for (index, character) in token.char_indices() {
        if index != 0 && (character == '+' || character == '-') {
            parts.push((sign, &token[start..index]));
            sign = if character == '+' { 1 } else { -1 };
            start = index + character.len_utf8();
        }
    }
    parts.push((sign, &token[start..]));
    parts
}

impl Validate for ProgramSpec {
    fn validate(&self) -> conflaguration::Result<()> {
        let mut errors = Vec::new();
        let mut defined: BTreeMap<&str, ()> = BTreeMap::new();

        for entry in &self.node {
            if defined.insert(entry.id(), ()).is_some() {
                errors.push(ValidationMessage::new(entry.id(), "defined twice"));
            }
            match entry {
                NodeSpec::Input { .. } | NodeSpec::Iota { .. } | NodeSpec::Constant { .. } => {}
                NodeSpec::Elementwise {
                    id, inputs, maps, ..
                } => {
                    if inputs.len() != maps.len() {
                        errors.push(ValidationMessage::new(
                            id,
                            "inputs and maps differ in count",
                        ));
                    }
                    for reference in inputs {
                        if !defined.contains_key(reference.as_str()) {
                            errors
                                .push(ValidationMessage::new(id, "input is not defined above it"));
                        }
                    }
                    for map in maps {
                        if let MapSpec::Gather { gather, .. } = map
                            && !defined.contains_key(gather.as_str())
                        {
                            errors.push(ValidationMessage::new(
                                id,
                                "gather references a node not defined above it",
                            ));
                        }
                    }
                }
                NodeSpec::Reduce { id, input, .. } => {
                    if !defined.contains_key(input.as_str()) {
                        errors.push(ValidationMessage::new(id, "input is not defined above it"));
                    }
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(conflaguration::Error::Validation { errors })
        }
    }
}

impl TryFrom<&ProgramSpec> for Vec<Op> {
    type Error = TensorError;

    fn try_from(spec: &ProgramSpec) -> Result<Self, Self::Error> {
        let mut program = Vec::new();
        let mut resolved: BTreeMap<String, NodeId> = BTreeMap::new();

        for entry in &spec.node {
            let built = match entry {
                NodeSpec::Input {
                    dtype, shape, name, ..
                } => {
                    let extents = shape
                        .iter()
                        .map(ExtentSpec::resolve)
                        .collect::<Result<Vec<Extent>, TensorError>>()?;
                    op::append(
                        &mut program,
                        Op::Input {
                            dtype: *dtype,
                            shape: extents,
                            name: name.clone(),
                        },
                    )
                }
                NodeSpec::Elementwise {
                    id,
                    dtype,
                    body,
                    inputs,
                    maps,
                    name,
                } => {
                    if inputs.len() != maps.len() {
                        return Err(TensorError::SpecArityMismatch {
                            node: id.clone(),
                            inputs: inputs.len(),
                            maps: maps.len(),
                        });
                    }
                    let operands = inputs
                        .iter()
                        .zip(maps)
                        .map(|(reference, map_spec)| {
                            let node = lookup(&resolved, reference)?;
                            let index_map = resolve_map_spec(&resolved, map_spec)?;
                            Ok((node, index_map))
                        })
                        .collect::<Result<Vec<(NodeId, IndexMap)>, TensorError>>()?;
                    op::append(
                        &mut program,
                        Op::Elementwise {
                            dtype: *dtype,
                            body: *body,
                            operands,
                            name: name.clone(),
                        },
                    )
                }
                NodeSpec::Reduce {
                    dtype,
                    body,
                    init,
                    input,
                    in_map,
                    out_map,
                    keep,
                    name,
                    ..
                } => {
                    let operand = lookup(&resolved, input)?;
                    let in_pattern = parse_operand_pattern(in_map)?;
                    let (out_rank, out_projected) = parse_projection(out_map)?;
                    op::append(
                        &mut program,
                        Op::Reduce(Reduce {
                            dtype: *dtype,
                            body: *body,
                            init: *init,
                            operand,
                            in_map: IndexMap::Affine(in_pattern),
                            out_map: IndexMap::Affine(map::projection(out_rank, &out_projected)),
                            keep: *keep,
                            name: name.clone(),
                        }),
                    )
                }
                NodeSpec::Iota { dtype, extent, .. } => op::append(
                    &mut program,
                    Op::Iota {
                        dtype: *dtype,
                        extent: extent.resolve()?,
                    },
                ),
                NodeSpec::Constant {
                    dtype,
                    shape,
                    value,
                    ..
                } => op::append(
                    &mut program,
                    Op::Constant {
                        dtype: *dtype,
                        shape: shape
                            .iter()
                            .map(ExtentSpec::resolve)
                            .collect::<Result<Vec<Extent>, TensorError>>()?,
                        value: *value,
                    },
                ),
            };
            resolved.insert(entry.id().to_string(), built);
        }

        Ok(program)
    }
}

/// Builds an [`IndexMap`] from one [`MapSpec`] entry, resolving a `Gather`'s
/// `gather` node id the same way an `inputs` entry resolves.
fn resolve_map_spec(
    resolved: &BTreeMap<String, NodeId>,
    map_spec: &MapSpec,
) -> Result<IndexMap, TensorError> {
    match map_spec {
        MapSpec::Projection(notation) => Ok(IndexMap::Affine(parse_operand_pattern(notation)?)),
        MapSpec::Gather {
            gather,
            index_map,
            map: base_notation,
            dim,
        } => {
            let indices = lookup(resolved, gather)?;
            let (index_rank, index_projected) = parse_projection(index_map)?;
            let (base_rank, base_projected) = parse_projection(base_notation)?;
            Ok(IndexMap::Computed {
                indices,
                index_map: map::projection(index_rank, &index_projected),
                base: build_base_pattern(base_rank, &base_projected, *dim),
                gathered_dim: *dim,
            })
        }
    }
}

/// Builds a gather's `base` index pattern from its non-gathered projected
/// axes (in operand-axis order) plus the gathered axis's position: an empty
/// [`AxisIndex`] is spliced in at `gathered_dim`, since that axis's address
/// comes from the fetch, not from `axes`' own terms. `gathered_dim` past the
/// operand's rank is clamped rather than panicking — an out-of-range value
/// is a well-formed but invalid `IndexPattern` that
/// [`shape::infer`](crate::shape::infer) rejects downstream with
/// [`TensorError::GatheredDimOutOfRange`], the same as it would for one
/// built directly in Rust.
fn build_base_pattern(rank: u16, projected: &[u16], gathered_dim: u16) -> IndexPattern {
    let mut axes: Vec<AxisIndex> = projected
        .iter()
        .map(|axis| AxisIndex {
            terms: core::iter::once(AxisTerm::projection(*axis)).collect(),
            offset: 0,
            len: None,
        })
        .collect();
    let insert_at = (gathered_dim as usize).min(axes.len());
    axes.insert(insert_at, AxisIndex::default());
    IndexPattern {
        iter_rank: rank,
        axes,
    }
}

fn lookup(resolved: &BTreeMap<String, NodeId>, reference: &str) -> Result<NodeId, TensorError> {
    resolved
        .get(reference)
        .copied()
        .ok_or_else(|| TensorError::UnknownNode(reference.to_string()))
}

/// Appends one [`Op::Elementwise`], parsing each operand's `operand->
/// iteration` notation through the same `parse_operand_pattern` the TOML
/// lowering above uses. This is the whole reason a hand-built full-model
/// program stays honest to the TOML one node kind spells: both paths run the
/// identical grammar, so a generated layer cannot silently drift from
/// `specs/mistral_layer.toml`'s own addressing.
pub fn elementwise(
    program: &mut Vec<Op>,
    dtype: DType,
    body: ScalarOp,
    inputs: &[(NodeId, &str)],
) -> Result<NodeId, TensorError> {
    let operands = inputs
        .iter()
        .map(|(node, notation)| Ok((*node, IndexMap::Affine(parse_operand_pattern(notation)?))))
        .collect::<Result<Vec<(NodeId, IndexMap)>, TensorError>>()?;
    Ok(op::append(
        program,
        Op::Elementwise {
            dtype,
            body,
            operands,
            name: None,
        },
    ))
}

/// Appends one [`Op::Reduce`], same notation-parsing rationale as
/// [`elementwise`].
pub fn reduce(
    program: &mut Vec<Op>,
    dtype: DType,
    body: ScalarOp,
    init: ReduceInit,
    operand: NodeId,
    in_map: &str,
    out_map: &str,
) -> Result<NodeId, TensorError> {
    let in_pattern = parse_operand_pattern(in_map)?;
    let (out_rank, out_projected) = parse_projection(out_map)?;
    Ok(op::append(
        program,
        Op::Reduce(Reduce {
            dtype,
            body,
            init,
            operand,
            in_map: IndexMap::Affine(in_pattern),
            out_map: IndexMap::Affine(map::projection(out_rank, &out_projected)),
            keep: Keep::Reduce,
            name: None,
        }),
    ))
}

/// Which pairing a checkpoint's RoPE uses to split one head's channel axis
/// into rotation pairs -- interleaved (`(2*i, 2*i+1)`, llama's
/// `kernel_rope_norm`) for a checkpoint with no QK-norm, split-half
/// (`(i, i+pairs)`, llama's `kernel_rope_neox`,
/// `ggml-metal.metal:2795-2845`) for one with it -- mirroring the
/// `qk_norm.is_some()` match a few call sites below this one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopePairing {
    Interleaved,
    SplitHalf { pairs: u32 },
}

impl RopePairing {
    /// `(same, partner)` axis-expression suffixes reading a pair's two
    /// source elements off ONE `head_dim`-wide axis, given a pair index `i`
    /// (extent `pairs`) that is genuinely NARROWER than `source`'s own
    /// channel axis under [`Self::SplitHalf`] (`source` is `attn_head_dim`
    /// wide; `i` only ranges over `pairs = rotary_dim/2`). A bare `"i"`
    /// address is genuinely correct here -- `source` really is read
    /// starting at the origin with a plain identity map -- but `i`'s true
    /// extent is `pairs`, not `source`'s own on-disk width, whenever
    /// `pairs < attn_head_dim` (partial rotary). `"i@{pairs}"` states that
    /// directly via [`crate::map::AxisIndex::len`] (`parse_axis_expr`'s own
    /// doc has the grammar): the address is `i`, plain; the extent fact is
    /// `pairs`, separate. `cos_new`/`sin_new`'s own plain `"i"` (no `@`,
    /// their own width already IS `pairs`) still resolves the same value, so
    /// both agree without either lying about its arithmetic.
    /// [`Self::Interleaved`]'s `"2*i"` needs no such annotation: its
    /// coefficient-2 term already never qualifies as a size-definer (only a
    /// `coeff == 1` single term does), so it was never in tension with
    /// `cos_new`/`sin_new` the way a bare `"i"` was.
    fn offsets(self) -> (alloc::string::String, alloc::string::String) {
        match self {
            Self::Interleaved => (
                alloc::string::String::from("2*i"),
                alloc::string::String::from("2*i+1"),
            ),
            Self::SplitHalf { pairs } => (alloc::format!("i@{pairs}"), alloc::format!("i+{pairs}")),
        }
    }
}

/// Builds RoPE for one tensor (`source`, e.g. `q`/`k`/`k_new`, addressed
/// whole -- never a pre-sliced half) as two [`Op::Elementwise`] chains
/// reading `source` directly through `RopePairing::offsets` instead of
/// through a separately-materialized `per_head_channel_range` slice.
/// Returns `(first, second)`: `(rotated_even, rotated_odd)` under
/// [`RopePairing::Interleaved`], `(rotated_first, rotated_second)` under
/// [`RopePairing::SplitHalf`] -- the same two [`NodeId`]s every existing
/// downstream call site already threads separately into its own group
/// broadcast / cache dot product, unchanged.
///
/// A genuinely single-dispatch form (both halves from one
/// [`Op::Elementwise`], mirroring llama's `kernel_rope_neox`/
/// `kernel_rope_norm` writing both destinations from one pair read) was
/// tried and reverted: packing a parity axis into the source's own
/// `IndexMap` leaves that axis with no operand anywhere in the node that
/// gives `shape::infer`'s `unify_iteration_space`
/// (`shape.rs:208-256`) a pure single-coefficient-term projection to size
/// it from, since the source's own term is fused with `i` and every
/// available trig/sign operand is either broadcast-away from it or
/// entangled the same way -- `UnconstrainedDim` at construction, not a
/// runtime bug. This form still removes the two-op-per-half
/// `per_head_channel_range` mask-and-reduce this crate used to build
/// `q_first`/`q_second`/`k_first`/`k_second` before rotating them.
pub fn fused_rope_pair(
    program: &mut Vec<Op>,
    source: NodeId,
    head_letter: char,
    cos_new: NodeId,
    sin_new: NodeId,
    pairing: RopePairing,
) -> Result<(NodeId, NodeId), TensorError> {
    let out_axes = alloc::format!("s{head_letter}i");
    let out_identity = alloc::format!("{out_axes}->{out_axes}");
    let (same_offset, partner_offset) = pairing.offsets();

    let same_pattern = alloc::format!("s,{head_letter},{same_offset}->{out_axes}");
    let partner_pattern = alloc::format!("s,{head_letter},{partner_offset}->{out_axes}");
    let trig_pattern = alloc::format!("s,i->{out_axes}");

    let same_cos = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(source, same_pattern.as_str()), (cos_new, trig_pattern.as_str())],
    )?;
    let partner_sin = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(source, partner_pattern.as_str()), (sin_new, trig_pattern.as_str())],
    )?;
    let rotated_same = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(same_cos, out_identity.as_str()), (partner_sin, out_identity.as_str())],
    )?;

    let partner_cos = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(source, partner_pattern.as_str()), (cos_new, trig_pattern.as_str())],
    )?;
    let same_sin = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(source, same_pattern.as_str()), (sin_new, trig_pattern.as_str())],
    )?;
    let rotated_partner = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(partner_cos, out_identity.as_str()), (same_sin, out_identity.as_str())],
    )?;

    Ok((rotated_same, rotated_partner))
}

/// `[?0]`-shaped bound leaf. `eps` is the only caller left: RMSNorm's
/// epsilon is model metadata (`attention.layer_norm_rms_epsilon` in a GGUF
/// checkpoint), not a value this function's `u32` parameters determine, so
/// it is the one constant here that cannot become an [`Op::Constant`]
/// without `mistral_forward_program` taking it as a parameter.
/// `inv_dim`/`ones`/`group_ones` all could, and did — see [`scalar_constant`].
#[must_use]
pub fn symbolic_leaf(program: &mut Vec<Op>, dtype: DType, name: &str) -> NodeId {
    input_leaf(program, dtype, alloc::vec![Extent::Symbolic(0)], name)
}

/// Appends a bound [`Op::Input`] leaf, the primitive every forward-program
/// builder threads weights and activations through — see
/// [`qwen35_forward_program`] for the worked example of composing leaves
/// like this one into a full program.
#[must_use]
pub fn input_leaf(program: &mut Vec<Op>, dtype: DType, shape: Vec<Extent>, name: &str) -> NodeId {
    op::append(
        program,
        Op::Input {
            dtype,
            shape,
            name: Some(name.into()),
        },
    )
}

/// A rank-0 [`Op::Constant`]: one literal that broadcasts into any consumer
/// through an empty operand side (`"->sd"`, `"->stug"`). This is how every
/// scalar this module needs is spelled — `inv_dim`, `ones`,
/// `inv_sqrt_head_dim` and `neg_infinity` were a bound `Input` or a
/// multi-node `Iota` derivation before the variant existed, and
/// [`Op::Constant`]'s own doc records what each cost.
#[must_use]
pub fn scalar_constant(program: &mut Vec<Op>, value: f32) -> NodeId {
    op::append(
        program,
        Op::Constant {
            dtype: DType::Float32,
            shape: Vec::new(),
            value,
        },
    )
}

/// `table[ids[s], d]`, the exact pattern `shape.rs`'s
/// `embedding_lookup_program` unit test documents: `ids` selects `table`'s
/// vocab axis, `d` passes through as a plain projection. Every forward
/// program opens with this gather -- see [`qwen35_forward_program`] for the
/// worked example.
#[must_use]
pub fn embedding_lookup(program: &mut Vec<Op>, table: NodeId, ids: NodeId) -> NodeId {
    let gathered_map = IndexMap::Computed {
        indices: ids,
        index_map: map::projection(2, &[0]),
        base: IndexPattern {
            iter_rank: 2,
            axes: alloc::vec![
                AxisIndex::default(),
                AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(1)).collect(),
                    offset: 0,
                    len: None,
                },
            ],
        },
        gathered_dim: 0,
    };
    op::append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: alloc::vec![(table, gathered_map)],
            name: None,
        },
    )
}

/// `specs/mistral_layer.toml`'s `attn_norm`/`ffn_norm` node run, node for
/// node: `x` normalized by its own root-mean-square, then scaled by `gamma`
/// (`[embedding]`, broadcast `d->sd`), the checkpoint's learned
/// `*_norm.weight` — RMSNorm without it is a different, un-trained function.
pub fn rmsnorm(
    program: &mut Vec<Op>,
    x: NodeId,
    gamma: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
) -> Result<NodeId, TensorError> {
    let squared = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(x, "sd->sd"), (x, "sd->sd")],
    )?;
    let sum_squares = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        squared,
        "sd->sd",
        "s->sd",
    )?;
    let mean_square = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(sum_squares, "s->s"), (inv_dim, "->s")],
    )?;
    let mean_square_eps = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(mean_square, "s->s"), (eps, "s->s")],
    )?;
    let rms = elementwise(
        program,
        DType::Float32,
        ScalarOp::SquareRoot,
        &[(mean_square_eps, "s->s")],
    )?;
    let inv_rms = elementwise(
        program,
        DType::Float32,
        ScalarOp::Reciprocal,
        &[(rms, "s->s")],
    )?;
    let normed = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(x, "sd->sd"), (inv_rms, "s->sd")],
    )?;
    elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "sd->sd"), (gamma, "d->sd")],
    )
}

/// [`rmsnorm`]'s per-head counterpart -- `Lfm2MoeAttention.q_layernorm`/
/// `.k_layernorm`'s own shape (`transformers/models/lfm2_moe/modeling_lfm2_moe.py:317-318,331-332`):
/// normalizes over the head-dim axis only, broadcasting per token AND per
/// head, applied to Q/K right after the head reshape and BEFORE RoPE
/// (`apply_rotary_pos_emb` runs on the ALREADY-normalized `query_states`/
/// `key_states`, `modeling_lfm2_moe.py:331-336`) -- normalizing after RoPE
/// would rotate un-normalized vectors, silently degrading with position the
/// same way the SmolLM2 RoPE-ordering bug did. `head` distinguishes Q's
/// query-head axis (`h`, [`append_attention_mixer`]'s own letter) from K's
/// kv-head axis (`u`) so this one function serves both call sites without
/// two copies of the same six ops. `head` is a format-interpolated string
/// rather than a single letter so the same six ops also serve a head space
/// that is genuinely two axes -- [`append_qwen35_ssm_mixer`]'s own `u,g`
/// (kv-head, group) split, which [`repeat_kv_heads`]'s own doc proves this
/// algebra cannot merge into one physical axis -- since every interpolation
/// site here (`s{head}d`, `s{head}`) treats `head` as an opaque run of
/// letters, not a single character.
pub fn rmsnorm_per_head(
    program: &mut Vec<Op>,
    x: NodeId,
    gamma: NodeId,
    inv_head_dim: NodeId,
    eps: NodeId,
    head: &str,
) -> Result<NodeId, TensorError> {
    let full = alloc::format!("s{head}d->s{head}d");
    let identity = alloc::format!("s{head}->s{head}");
    let broadcast_over_d = alloc::format!("s{head}->s{head}d");
    let inv_head_dim_map = alloc::format!("->s{head}");
    let eps_map = alloc::format!("s->s{head}");
    let gamma_map = alloc::format!("d->s{head}d");

    let squared = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(x, full.as_str()), (x, full.as_str())],
    )?;
    let sum_squares = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        squared,
        full.as_str(),
        broadcast_over_d.as_str(),
    )?;
    let mean_square = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (sum_squares, identity.as_str()),
            (inv_head_dim, inv_head_dim_map.as_str()),
        ],
    )?;
    let mean_square_eps = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(mean_square, identity.as_str()), (eps, eps_map.as_str())],
    )?;
    let rms = elementwise(
        program,
        DType::Float32,
        ScalarOp::SquareRoot,
        &[(mean_square_eps, identity.as_str())],
    )?;
    let inv_rms = elementwise(
        program,
        DType::Float32,
        ScalarOp::Reciprocal,
        &[(rms, identity.as_str())],
    )?;
    let normed = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(x, full.as_str()), (inv_rms, broadcast_over_d.as_str())],
    )?;
    elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, full.as_str()), (gamma, gamma_map.as_str())],
    )
}

/// The causal mask's data-independent half: `is_future` (a `(query, key)`
/// comparison between two [`Op::Iota`]s) and `neg_infinity`, built once and
/// shared by every layer — position-only, no learned state, exactly like
/// `cos`/`sin`.
///
/// `is_future` is what `Iota` is *for*: its value at `(s, t)` genuinely
/// depends on position, so nothing but an index tensor can produce it.
/// `neg_infinity` is the opposite and used to be spelled the same way —
/// `Subtract(iota, iota)` for `0.0`, `Negate` for `-0.0`, `Reciprocal` for
/// `-inf`, three nodes and a materialized `[?0]` tensor per call to say a
/// number that never varies. It is now one rank-0 [`Op::Constant`], which
/// is also why it broadcasts as `"->stug"` rather than `"s->stug"`.
pub fn causal_mask(program: &mut Vec<Op>) -> Result<(NodeId, NodeId), TensorError> {
    let query_index = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Symbolic(0),
        },
    );
    let key_index = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Symbolic(0),
        },
    );
    let is_future = elementwise(
        program,
        DType::Float32,
        ScalarOp::Greater,
        &[(key_index, "t->st"), (query_index, "s->st")],
    )?;
    let neg_infinity = scalar_constant(program, f32::NEG_INFINITY);
    Ok((is_future, neg_infinity))
}

/// [`causal_mask`]'s single-range-attention counterpart:
/// [`append_mistral_single_range_cached_layer`]'s key axis (`t`) is no
/// longer this call's own new positions (symbol 0) but the *whole* merged
/// context (symbol 1) -- everything a growing KV cache holds once this
/// call's own freshly rotated keys are folded into it. A query at local
/// position `s` sits at absolute position `cached_len + s`, so causality is
/// `key_index > cached_len + s` rather than the block-local `key_index >
/// query_index` [`causal_mask`] builds. `cached_len` is a rank-0
/// [`Op::Input`], the same precedent `eps`/`rope_cos`/`rope_sin` set: a
/// value the host supplies per call, not a build-time constant, because it
/// grows every decode step without the graph being rebuilt.
pub fn causal_mask_merged(program: &mut Vec<Op>, cached_len: NodeId) -> Result<NodeId, TensorError> {
    let query_index = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Symbolic(0),
        },
    );
    let key_index = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Symbolic(1),
        },
    );
    let query_absolute = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(query_index, "s->s"), (cached_len, "->s")],
    )?;
    elementwise(
        program,
        DType::Float32,
        ScalarOp::Greater,
        &[(key_index, "t->st"), (query_absolute, "s->st")],
    )
}

/// One transformer layer, node-for-node the same graph
/// `specs/mistral_layer.toml` spells — attention (RoPE + GQA + causal mask)
/// then the SwiGLU feed-forward, each wrapped in its own residual. `x` in,
/// the layer's own residual-summed output out; every other argument is
/// either a per-layer weight (`wq`/`wk`/`wv`/`wo`/`w_gate`/`w_up`/`w_down`)
/// or one of the position-only constants [`causal_mask`]/`cos`/`sin` share
/// across every layer.
#[allow(clippy::too_many_arguments)]
pub fn append_mistral_layer(
    program: &mut Vec<Op>,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    ones: NodeId,
    inv_sqrt_head_dim: NodeId,
    cos: NodeId,
    sin: NodeId,
    group_ones: NodeId,
    is_future: NodeId,
    neg_infinity: NodeId,
    group: u32,
    attn_norm_weight: NodeId,
    ffn_norm_weight: NodeId,
    wq: NodeId,
    wk: NodeId,
    wv: NodeId,
    wo: NodeId,
    w_gate: NodeId,
    w_up: NodeId,
    w_down: NodeId,
) -> Result<NodeId, TensorError> {
    let normed = rmsnorm(program, x, attn_norm_weight, inv_dim, eps)?;

    let q_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->shdi"), (wq, "ihd->shdi")],
    )?;
    let q = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        q_product,
        "shdi->shdi",
        "shd->shdi",
    )?;

    let k_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->sudi"), (wk, "iud->sudi")],
    )?;
    let k = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        k_product,
        "sudi->sudi",
        "sud->sudi",
    )?;

    let v_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->sudi"), (wv, "iud->sudi")],
    )?;
    let v = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        v_product,
        "sudi->sudi",
        "sud->sudi",
    )?;

    let q_even_cos = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(q, "s,h,2*i->shi"), (cos, "si->shi")],
    )?;
    let q_odd_sin = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(q, "s,h,2*i+1->shi"), (sin, "si->shi")],
    )?;
    let rotated_q_even = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(q_even_cos, "shi->shi"), (q_odd_sin, "shi->shi")],
    )?;
    let q_even_sin = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(q, "s,h,2*i->shi"), (sin, "si->shi")],
    )?;
    let q_odd_cos = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(q, "s,h,2*i+1->shi"), (cos, "si->shi")],
    )?;
    let rotated_q_odd = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(q_even_sin, "shi->shi"), (q_odd_cos, "shi->shi")],
    )?;

    let k_even_cos = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(k, "s,u,2*i->sui"), (cos, "si->sui")],
    )?;
    let k_odd_sin = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(k, "s,u,2*i+1->sui"), (sin, "si->sui")],
    )?;
    let rotated_k_even = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(k_even_cos, "sui->sui"), (k_odd_sin, "sui->sui")],
    )?;
    let k_even_sin = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(k, "s,u,2*i->sui"), (sin, "si->sui")],
    )?;
    let k_odd_cos = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(k, "s,u,2*i+1->sui"), (cos, "si->sui")],
    )?;
    let rotated_k_odd = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(k_even_sin, "sui->sui"), (k_odd_cos, "sui->sui")],
    )?;

    let group_map = alloc::format!("s,{group}*u+g,i->sugi");
    let q_even_grouped = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (rotated_q_even, group_map.as_str()),
            (group_ones, "ug->sugi"),
        ],
    )?;
    let q_odd_grouped = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (rotated_q_odd, group_map.as_str()),
            (group_ones, "ug->sugi"),
        ],
    )?;

    let score_even_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (q_even_grouped, "sugi->stugi"),
            (rotated_k_even, "tui->stugi"),
        ],
    )?;
    let score_even = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_even_product,
        "stugi->stugi",
        "stug->stugi",
    )?;
    let score_odd_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (q_odd_grouped, "sugi->stugi"),
            (rotated_k_odd, "tui->stugi"),
        ],
    )?;
    let score_odd = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_odd_product,
        "stugi->stugi",
        "stug->stugi",
    )?;
    let scores = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(score_even, "stug->stug"), (score_odd, "stug->stug")],
    )?;

    // attention's usual `1/sqrt(d_k)`: without it QK^T over a real head_dim
    // (128) saturates softmax toward one-hot instead of blending.
    // `inv_sqrt_head_dim` is a rank-0 `Op::Constant`, so it broadcasts via
    // an empty operand side, the same way `neg_infinity` does.
    let scores_scaled = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(scores, "stug->stug"), (inv_sqrt_head_dim, "->stug")],
    )?;

    let scores_masked = elementwise(
        program,
        DType::Float32,
        ScalarOp::Select,
        &[
            (is_future, "st->stug"),
            (neg_infinity, "->stug"),
            (scores_scaled, "stug->stug"),
        ],
    )?;

    let score_max = reduce(
        program,
        DType::Float32,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        scores_masked,
        "stug->stug",
        "sug->stug",
    )?;
    let shifted = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(scores_masked, "stug->stug"), (score_max, "sug->stug")],
    )?;
    let weights = elementwise(
        program,
        DType::Float32,
        ScalarOp::Exponential,
        &[(shifted, "stug->stug")],
    )?;
    let weight_sum = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        weights,
        "stug->stug",
        "sug->stug",
    )?;
    let inv_weight_sum = elementwise(
        program,
        DType::Float32,
        ScalarOp::Reciprocal,
        &[(weight_sum, "sug->sug")],
    )?;
    let probabilities = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(weights, "stug->stug"), (inv_weight_sum, "sug->stug")],
    )?;

    let attended_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(probabilities, "stug->stugd"), (v, "tud->stugd")],
    )?;
    let attended = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        attended_product,
        "stugd->stugd",
        "sugd->stugd",
    )?;

    let wo_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(attended, "sugd->sugdo"), (wo, "ugdo->sugdo")],
    )?;
    let attn_out = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        wo_product,
        "sugdo->sugdo",
        "so->sugdo",
    )?;

    let residual1 = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(attn_out, "sd->sd"), (x, "sd->sd")],
    )?;

    let normed2 = rmsnorm(program, residual1, ffn_norm_weight, inv_dim, eps)?;

    let gate_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed2, "sd->sdg"), (w_gate, "dg->sdg")],
    )?;
    let gate = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        gate_product,
        "sdg->sdg",
        "sg->sdg",
    )?;
    let up_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed2, "sd->sdg"), (w_up, "dg->sdg")],
    )?;
    let up = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        up_product,
        "sdg->sdg",
        "sg->sdg",
    )?;

    let neg_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Negate,
        &[(gate, "sg->sg")],
    )?;
    let exp_neg_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Exponential,
        &[(neg_gate, "sg->sg")],
    )?;
    let one_plus_exp = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(exp_neg_gate, "sg->sg"), (ones, "->sg")],
    )?;
    let sigmoid_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Reciprocal,
        &[(one_plus_exp, "sg->sg")],
    )?;
    let silu_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(gate, "sg->sg"), (sigmoid_gate, "sg->sg")],
    )?;
    let ffn_hidden = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(silu_gate, "sg->sg"), (up, "sg->sg")],
    )?;

    let down_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(ffn_hidden, "sg->sgd"), (w_down, "gd->sgd")],
    )?;
    let ffn_out = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        down_product,
        "sgd->sgd",
        "sd->sgd",
    )?;

    elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(ffn_out, "sd->sd"), (residual1, "sd->sd")],
    )
}

/// Rust-code counterpart of `specs/moe_block.toml`'s `ffn_product` node:
/// gathers `stack[route[s], :, :]` (`stack` is a `[expert_count, d_in,
/// d_out]` weight slab) and multiplies it elementwise against `x`'s `[s,
/// d_in]`, broadcast over the `d_out` axis, ready for a later [`reduce`]
/// over `d_in` to finish the matmul. The same [`IndexMap::Computed`] gather
/// [`embedding_lookup`] uses, with one extra non-gathered axis (`d_out`)
/// spliced in after the gathered one instead of none.
#[must_use]
pub fn gathered_expert_product(
    program: &mut Vec<Op>,
    stack: NodeId,
    route: NodeId,
    x: NodeId,
) -> NodeId {
    let gathered_map = IndexMap::Computed {
        indices: route,
        index_map: map::projection(3, &[0]),
        base: IndexPattern {
            iter_rank: 3,
            axes: alloc::vec![
                AxisIndex::default(),
                AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(1)).collect(),
                    offset: 0,
                    len: None,
                },
                AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(2)).collect(),
                    offset: 0,
                    len: None,
                },
            ],
        },
        gathered_dim: 0,
    };
    let x_map = IndexMap::Affine(map::projection(3, &[0, 1]));
    op::append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Multiply,
            operands: alloc::vec![(stack, gathered_map), (x, x_map)],
            name: None,
        },
    )
}

/// Which function turns a MoE gate's raw logits into per-expert routing
/// scores -- llama.cpp's own `llama_expert_gating_func_type`
/// (`llama-hparams.h:11-14`), read from a checkpoint's own
/// `{architecture}.expert_gating_func` metadata key when present.
/// `Softmax` is llama.cpp's own fallback when that key is absent
/// (`llama-model.cpp:1237-1240`, "existing models that have no
/// `expert_gating_func` model parameter set") -- Mixtral carries no such
/// key, so `append_mistral_moe_layer`/`append_mistral_cached_moe_layer`
/// always pass `Softmax` unconditionally rather than reading a key that
/// does not exist on that checkpoint. `Sigmoid` is `_TYPE_SIGMOID` (`2`),
/// LFM2's own value (`transformers/models/lfm2_moe/modeling_lfm2_moe.py:209`'s
/// `router_logits.sigmoid()`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertGatingFunc {
    Softmax,
    Sigmoid,
}

/// One [`append_moe_ffn`] call's routing decision, returned alongside its
/// output node so the decode loop -- not this kernel-building function --
/// decides whether to evaluate and observe it. `selected` holds one
/// [`NodeId`] per `expert_used_count` round (each round's own `route`
/// reduce, `Int32`, one value per token position); `weights` holds each
/// round's own `weight` node in the same order, followed by the final
/// `weight_total` node used to renormalize them -- so `weights.len() ==
/// selected.len() + 1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeSite {
    pub layer: u32,
    pub selected: Vec<NodeId>,
    pub weights: Vec<NodeId>,
}

/// Every [`MoeSite`] a forward-program builder's [`append_moe_ffn`] calls
/// produced, in layer order -- empty for a dense (non-MoE) program. The
/// decode loop (`crate::instrument::ExpertObserver`'s consumer, gated behind
/// this crate's `instrument` feature) reads this to know which extra nodes
/// to request as evaluation outputs, rather than the kernel emitting a
/// routing event per gathered position the way
/// `crate::instrument::notify_expert_routed` used to.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MoeSites(pub Vec<MoeSite>);

/// The routed feed-forward `specs/moe_block.toml`/`specs/moe_topk2_probe.toml`
/// describe: a gate projects `x` to one logit per expert, `expert_used_count`
/// rounds of top-1 argmax-with-exclusion each route one token to one more
/// expert (`moe_topk2_probe.toml`'s own header proves a *fixed* k stays
/// inside the affine-only + `Iota`/`Computed`-gather algebra, no new
/// `Op`/`ScalarOp` variant, unrolled at spec-build time the same way this
/// whole function is), and each round's gathered expert runs the same
/// SwiGLU [`append_mistral_layer`]'s dense path uses, weighted by its own
/// share among only the selected experts.
///
/// `gating` picks how raw `logits` become the per-expert `scores` used both
/// to select AND (absent a bias) to weight experts:
/// [`ExpertGatingFunc::Softmax`] leaves `scores` aliased to `logits` --
/// `weight_r = exp(max_logit_r - max_logit_0)` then a final
/// divide-by-`weight_total` is *exactly* softmax restricted to the selected
/// top-k and renormalized (the standard Mixtral combination formula), so
/// this never diverges by so much as one node from the code this function
/// has always built. [`ExpertGatingFunc::Sigmoid`] materializes
/// `sigmoid(logits)` up front via the same `Negate`+`Exponential`+`Add(1)`+
/// `Reciprocal` construction the dense SwiGLU path already builds
/// (`spec.rs`'s own `silu_gate` node a few lines below this one) --
/// `ScalarOp` gained no `Sigmoid` variant for this, since the four ops
/// already existed for a different consumer.
///
/// `expert_bias` (`blk.{layer}.exp_probs_b.bias` on a real LFM2 checkpoint,
/// `[expert_count]`) is llama.cpp's own `ffn_exp_probs_b` /
/// `route_tokens_to_experts`'s own `self.expert_bias`
/// (`modeling_lfm2_moe.py:210-213`, `llama-graph.cpp`'s own
/// `build_moe_ffn`'s `selection_probs = ggml_add(probs, exp_probs_b)`
/// comment: "leave probs unbiased as it's later used to get expert
/// weights"): added to `scores` ONLY for the argmax that picks
/// `expert_used_count` experts, never for the weight a selected expert's
/// output is scaled by -- getting that backwards would still select
/// *plausible* experts (the bias is small relative to genuine routing
/// signal) while silently reweighting every token's combination, exactly
/// the "plausible output, wrong routing" failure mode metadata-absent
/// checkpoints (Mixtral, `expert_bias: None`) cannot exhibit since they
/// never reach this branch.
///
/// The routed feed-forward block [`lfm2_forward_program_with_experts`]
/// and [`mistral_cached_forward_program_with_experts`] both call per
/// layer; see [`qwen35_forward_program`] for this crate's own worked
/// example of a full per-layer builder chain (a dense, non-MoE FFN there).
#[allow(clippy::too_many_arguments)]
pub fn append_moe_ffn(
    program: &mut Vec<Op>,
    layer: u32,
    x: NodeId,
    gate_inp: NodeId,
    expert_w_gate: NodeId,
    expert_w_up: NodeId,
    expert_w_down: NodeId,
    expert_count: u32,
    expert_used_count: u32,
    ones: NodeId,
    gating: ExpertGatingFunc,
    expert_bias: Option<NodeId>,
) -> Result<(NodeId, MoeSite), TensorError> {
    if expert_used_count == 0 || expert_used_count > expert_count {
        return Err(TensorError::InvalidExpertConfig {
            expert_count,
            expert_used_count,
        });
    }

    let gate_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(x, "sd->sde"), (gate_inp, "de->sde")],
    )?;
    let logits = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        gate_product,
        "sde->sde",
        "se->sde",
    )?;

    let scores = match gating {
        ExpertGatingFunc::Softmax => logits,
        ExpertGatingFunc::Sigmoid => {
            let neg_logits = elementwise(
                program,
                DType::Float32,
                ScalarOp::Negate,
                &[(logits, "se->se")],
            )?;
            let exp_neg_logits = elementwise(
                program,
                DType::Float32,
                ScalarOp::Exponential,
                &[(neg_logits, "se->se")],
            )?;
            let one_plus_exp = elementwise(
                program,
                DType::Float32,
                ScalarOp::Add,
                &[(exp_neg_logits, "se->se"), (ones, "->se")],
            )?;
            elementwise(
                program,
                DType::Float32,
                ScalarOp::Reciprocal,
                &[(one_plus_exp, "se->se")],
            )?
        }
    };
    let mut selection_scores = match expert_bias {
        Some(bias) => elementwise(
            program,
            DType::Float32,
            ScalarOp::Add,
            &[(scores, "se->se"), (bias, "e->se")],
        )?,
        None => scores,
    };

    let expert_index = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(expert_count),
        },
    );
    let neg_infinity = scalar_constant(program, f32::NEG_INFINITY);

    let mut weighted_sum: Option<NodeId> = None;
    let mut weight_total: Option<NodeId> = None;
    let mut max_selection_0: Option<NodeId> = None;
    let mut selected_routes: Vec<NodeId> = Vec::with_capacity(expert_used_count as usize);
    let mut round_weights: Vec<NodeId> = Vec::with_capacity(expert_used_count as usize);

    for round in 0..expert_used_count {
        let max_selection = reduce(
            program,
            DType::Float32,
            ScalarOp::Maximum,
            ReduceInit::NegativeInfinity,
            selection_scores,
            "se->se",
            "s->se",
        )?;
        let mask = elementwise(
            program,
            DType::Float32,
            ScalarOp::Equal,
            &[(selection_scores, "se->se"), (max_selection, "s->se")],
        )?;
        let candidate = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(mask, "se->se"), (expert_index, "e->se")],
        )?;
        let route = reduce(
            program,
            DType::Int32,
            ScalarOp::Maximum,
            ReduceInit::Zero,
            candidate,
            "se->se",
            "s->se",
        )?;

        let weight = match gating {
            ExpertGatingFunc::Softmax => {
                let first_max = *max_selection_0.get_or_insert(max_selection);
                let shifted = elementwise(
                    program,
                    DType::Float32,
                    ScalarOp::Subtract,
                    &[(max_selection, "s->s"), (first_max, "s->s")],
                )?;
                elementwise(
                    program,
                    DType::Float32,
                    ScalarOp::Exponential,
                    &[(shifted, "s->s")],
                )?
            }
            ExpertGatingFunc::Sigmoid => {
                // unbiased `scores` at the masked (selected) position, never
                // `max_selection` itself -- that would be the biased score.
                let masked_scores = elementwise(
                    program,
                    DType::Float32,
                    ScalarOp::Multiply,
                    &[(mask, "se->se"), (scores, "se->se")],
                )?;
                reduce(
                    program,
                    DType::Float32,
                    ScalarOp::Add,
                    ReduceInit::Zero,
                    masked_scores,
                    "se->se",
                    "s->se",
                )?
            }
        };
        selected_routes.push(route);
        round_weights.push(weight);

        let gate_expert_product = gathered_expert_product(program, expert_w_gate, route, x);
        let gate_expert = reduce(
            program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            gate_expert_product,
            "sio->sio",
            "so->sio",
        )?;
        let up_expert_product = gathered_expert_product(program, expert_w_up, route, x);
        let up_expert = reduce(
            program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            up_expert_product,
            "sio->sio",
            "so->sio",
        )?;

        let neg_gate = elementwise(
            program,
            DType::Float32,
            ScalarOp::Negate,
            &[(gate_expert, "sg->sg")],
        )?;
        let exp_neg_gate = elementwise(
            program,
            DType::Float32,
            ScalarOp::Exponential,
            &[(neg_gate, "sg->sg")],
        )?;
        let one_plus_exp = elementwise(
            program,
            DType::Float32,
            ScalarOp::Add,
            &[(exp_neg_gate, "sg->sg"), (ones, "->sg")],
        )?;
        let sigmoid_gate = elementwise(
            program,
            DType::Float32,
            ScalarOp::Reciprocal,
            &[(one_plus_exp, "sg->sg")],
        )?;
        let silu_gate = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(gate_expert, "sg->sg"), (sigmoid_gate, "sg->sg")],
        )?;
        let ffn_hidden = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(silu_gate, "sg->sg"), (up_expert, "sg->sg")],
        )?;

        let down_expert_product =
            gathered_expert_product(program, expert_w_down, route, ffn_hidden);
        let round_ffn = reduce(
            program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            down_expert_product,
            "sio->sio",
            "so->sio",
        )?;

        let weighted_round = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(round_ffn, "sd->sd"), (weight, "s->sd")],
        )?;
        weighted_sum = Some(match weighted_sum {
            Some(accum) => elementwise(
                program,
                DType::Float32,
                ScalarOp::Add,
                &[(accum, "sd->sd"), (weighted_round, "sd->sd")],
            )?,
            None => weighted_round,
        });
        weight_total = Some(match weight_total {
            Some(accum) => elementwise(
                program,
                DType::Float32,
                ScalarOp::Add,
                &[(accum, "s->s"), (weight, "s->s")],
            )?,
            None => weight,
        });

        if round + 1 < expert_used_count {
            selection_scores = elementwise(
                program,
                DType::Float32,
                ScalarOp::Select,
                &[
                    (mask, "se->se"),
                    (neg_infinity, "->se"),
                    (selection_scores, "se->se"),
                ],
            )?;
        }
    }

    // `expert_used_count > 0` was checked above, so exactly that many rounds
    // ran and both accumulators are `Some`.
    let weight_total = weight_total.ok_or(TensorError::InvalidExpertConfig {
        expert_count,
        expert_used_count,
    })?;
    let weighted_sum = weighted_sum.ok_or(TensorError::InvalidExpertConfig {
        expert_count,
        expert_used_count,
    })?;
    let inv_weight_total = elementwise(
        program,
        DType::Float32,
        ScalarOp::Reciprocal,
        &[(weight_total, "s->s")],
    )?;
    let output = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(weighted_sum, "sd->sd"), (inv_weight_total, "s->sd")],
    )?;
    round_weights.push(weight_total);
    let site = MoeSite {
        layer,
        selected: selected_routes,
        weights: round_weights,
    };
    Ok((output, site))
}

/// [`append_mistral_layer`]'s mixture-of-experts counterpart: identical
/// attention block (RoPE + GQA + causal mask, node-for-node the same code),
/// [`append_moe_ffn`] in place of the dense SwiGLU triple. Kept as a
/// separate function rather than a branch inside [`append_mistral_layer`]
/// so the dense path's own node sequence — and therefore its generated
/// program bytes — never changes shape by so much as one node merely
/// because this function exists next to it.
#[allow(clippy::too_many_arguments)]
pub fn append_mistral_moe_layer(
    program: &mut Vec<Op>,
    layer: u32,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    ones: NodeId,
    inv_sqrt_head_dim: NodeId,
    cos: NodeId,
    sin: NodeId,
    group_ones: NodeId,
    is_future: NodeId,
    neg_infinity: NodeId,
    group: u32,
    attn_norm_weight: NodeId,
    ffn_norm_weight: NodeId,
    wq: NodeId,
    wk: NodeId,
    wv: NodeId,
    wo: NodeId,
    gate_inp: NodeId,
    expert_w_gate: NodeId,
    expert_w_up: NodeId,
    expert_w_down: NodeId,
    expert_count: u32,
    expert_used_count: u32,
) -> Result<(NodeId, MoeSite), TensorError> {
    let normed = rmsnorm(program, x, attn_norm_weight, inv_dim, eps)?;

    let q_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->shdi"), (wq, "ihd->shdi")],
    )?;
    let q = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        q_product,
        "shdi->shdi",
        "shd->shdi",
    )?;

    let k_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->sudi"), (wk, "iud->sudi")],
    )?;
    let k = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        k_product,
        "sudi->sudi",
        "sud->sudi",
    )?;

    let v_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->sudi"), (wv, "iud->sudi")],
    )?;
    let v = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        v_product,
        "sudi->sudi",
        "sud->sudi",
    )?;

    let q_even_cos = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(q, "s,h,2*i->shi"), (cos, "si->shi")],
    )?;
    let q_odd_sin = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(q, "s,h,2*i+1->shi"), (sin, "si->shi")],
    )?;
    let rotated_q_even = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(q_even_cos, "shi->shi"), (q_odd_sin, "shi->shi")],
    )?;
    let q_even_sin = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(q, "s,h,2*i->shi"), (sin, "si->shi")],
    )?;
    let q_odd_cos = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(q, "s,h,2*i+1->shi"), (cos, "si->shi")],
    )?;
    let rotated_q_odd = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(q_even_sin, "shi->shi"), (q_odd_cos, "shi->shi")],
    )?;

    let k_even_cos = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(k, "s,u,2*i->sui"), (cos, "si->sui")],
    )?;
    let k_odd_sin = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(k, "s,u,2*i+1->sui"), (sin, "si->sui")],
    )?;
    let rotated_k_even = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(k_even_cos, "sui->sui"), (k_odd_sin, "sui->sui")],
    )?;
    let k_even_sin = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(k, "s,u,2*i->sui"), (sin, "si->sui")],
    )?;
    let k_odd_cos = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(k, "s,u,2*i+1->sui"), (cos, "si->sui")],
    )?;
    let rotated_k_odd = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(k_even_sin, "sui->sui"), (k_odd_cos, "sui->sui")],
    )?;

    let group_map = alloc::format!("s,{group}*u+g,i->sugi");
    let q_even_grouped = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (rotated_q_even, group_map.as_str()),
            (group_ones, "ug->sugi"),
        ],
    )?;
    let q_odd_grouped = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (rotated_q_odd, group_map.as_str()),
            (group_ones, "ug->sugi"),
        ],
    )?;

    let score_even_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (q_even_grouped, "sugi->stugi"),
            (rotated_k_even, "tui->stugi"),
        ],
    )?;
    let score_even = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_even_product,
        "stugi->stugi",
        "stug->stugi",
    )?;
    let score_odd_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (q_odd_grouped, "sugi->stugi"),
            (rotated_k_odd, "tui->stugi"),
        ],
    )?;
    let score_odd = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_odd_product,
        "stugi->stugi",
        "stug->stugi",
    )?;
    let scores = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(score_even, "stug->stug"), (score_odd, "stug->stug")],
    )?;

    let scores_scaled = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(scores, "stug->stug"), (inv_sqrt_head_dim, "->stug")],
    )?;

    let scores_masked = elementwise(
        program,
        DType::Float32,
        ScalarOp::Select,
        &[
            (is_future, "st->stug"),
            (neg_infinity, "->stug"),
            (scores_scaled, "stug->stug"),
        ],
    )?;

    let score_max = reduce(
        program,
        DType::Float32,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        scores_masked,
        "stug->stug",
        "sug->stug",
    )?;
    let shifted = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(scores_masked, "stug->stug"), (score_max, "sug->stug")],
    )?;
    let weights = elementwise(
        program,
        DType::Float32,
        ScalarOp::Exponential,
        &[(shifted, "stug->stug")],
    )?;
    let weight_sum = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        weights,
        "stug->stug",
        "sug->stug",
    )?;
    let inv_weight_sum = elementwise(
        program,
        DType::Float32,
        ScalarOp::Reciprocal,
        &[(weight_sum, "sug->sug")],
    )?;
    let probabilities = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(weights, "stug->stug"), (inv_weight_sum, "sug->stug")],
    )?;

    let attended_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(probabilities, "stug->stugd"), (v, "tud->stugd")],
    )?;
    let attended = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        attended_product,
        "stugd->stugd",
        "sugd->stugd",
    )?;

    let wo_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(attended, "sugd->sugdo"), (wo, "ugdo->sugdo")],
    )?;
    let attn_out = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        wo_product,
        "sugdo->sugdo",
        "so->sugdo",
    )?;

    let residual1 = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(attn_out, "sd->sd"), (x, "sd->sd")],
    )?;

    let normed2 = rmsnorm(program, residual1, ffn_norm_weight, inv_dim, eps)?;

    let (ffn_out, site) = append_moe_ffn(
        program,
        layer,
        normed2,
        gate_inp,
        expert_w_gate,
        expert_w_up,
        expert_w_down,
        expert_count,
        expert_used_count,
        ones,
        ExpertGatingFunc::Softmax,
        None,
    )?;

    let output = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(ffn_out, "sd->sd"), (residual1, "sd->sd")],
    )?;
    Ok((output, site))
}

/// The whole model as one program: token embedding lookup, `block_count`
/// copies of `specs/mistral_layer.toml`'s layer (each with its own weights,
/// same node shape, generated rather than hand-authored — this is the whole
/// reason this function exists, since a 32-layer TOML would repeat one graph
/// 32 times with nothing but the weight names differing), a final RMSNorm,
/// and the LM head projection down to `[seq, vocab]` logits.
///
/// Config is plain `u32` parameters, not a struct — nothing here needs a
/// caller to hold them together as one type, and this crate deleted
/// `TensorExecutionConfig` for being unread rather than reintroduce that
/// shape. Composes this module's own `elementwise`/`reduce` (the exact
/// notation grammar `Vec<Op>::try_from(&ProgramSpec)` above already parses),
/// `embedding_lookup` (`shape.rs`'s `embedding_lookup_program` unit test is
/// the addressing reference), and `append_mistral_layer` (mirrors
/// `specs/mistral_layer.toml` node for node).
///
/// `expert_count == 0` means dense: every layer binds
/// `append_mistral_layer`'s plain `ffn_{gate,up,down}.weight` triple,
/// node-for-node the same program this function has always built, so a
/// dense checkpoint's generated program (and therefore its output) is
/// unaffected by this parameter's existence. `expert_count > 0` routes each
/// layer through `append_mistral_moe_layer` instead, gathering one of
/// `expert_count` experts' weight slabs per token per
/// `append_moe_ffn`'s doc.
#[allow(clippy::too_many_arguments)]
pub fn mistral_forward_program(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    block_count: u32,
    expert_count: u32,
    expert_used_count: u32,
) -> Result<Vec<Op>, TensorError> {
    let group = query_heads / kv_heads;
    let pairs = head_dim / 2;

    let mut program = Vec::new();

    let ids = input_leaf(
        &mut program,
        DType::Int32,
        alloc::vec![Extent::Symbolic(0)],
        "ids",
    );
    let table = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(vocab), Extent::Static(embedding)],
        "token_embd.weight",
    );
    let mut x = embedding_lookup(&mut program, table, ids);

    let inv_dim = scalar_constant(&mut program, 1.0 / embedding as f32);
    let eps = symbolic_leaf(&mut program, DType::Float32, "eps");
    let ones = scalar_constant(&mut program, 1.0);
    // attention's usual `1/sqrt(d_k)`, the same two IEEE ops the deleted
    // five-node `Iota` derivation performed, at build time instead of once
    // per forward pass.
    let inv_sqrt_head_dim = scalar_constant(&mut program, 1.0 / (head_dim as f32).sqrt());
    let cos = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(pairs)],
        "rope_cos",
    );
    let sin = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(pairs)],
        "rope_sin",
    );
    // the one constant here that is not rank-0: `q_*_grouped`'s `u` and `g`
    // iteration extents have no other operand to come from, so this leaf
    // carries them. Its values are all `1.0` either way.
    let group_ones = op::append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
            value: 1.0,
        },
    );
    let (is_future, neg_infinity) = causal_mask(&mut program)?;
    let mut moe_sites: Vec<MoeSite> = Vec::new();

    for layer in 0..block_count {
        let attn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.attn_norm.weight"),
        );
        let ffn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.ffn_norm.weight"),
        );
        let wq = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(embedding),
                Extent::Static(query_heads),
                Extent::Static(head_dim)
            ],
            &alloc::format!("blk.{layer}.attn_q.weight"),
        );
        let wk = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(embedding),
                Extent::Static(kv_heads),
                Extent::Static(head_dim)
            ],
            &alloc::format!("blk.{layer}.attn_k.weight"),
        );
        let wv = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(embedding),
                Extent::Static(kv_heads),
                Extent::Static(head_dim)
            ],
            &alloc::format!("blk.{layer}.attn_v.weight"),
        );
        let wo = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(kv_heads),
                Extent::Static(group),
                Extent::Static(head_dim),
                Extent::Static(embedding),
            ],
            &alloc::format!("blk.{layer}.attn_output.weight"),
        );
        x = if expert_count == 0 {
            let w_gate = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(feed_forward)],
                &alloc::format!("blk.{layer}.ffn_gate.weight"),
            );
            let w_up = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(feed_forward)],
                &alloc::format!("blk.{layer}.ffn_up.weight"),
            );
            let w_down = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(feed_forward), Extent::Static(embedding)],
                &alloc::format!("blk.{layer}.ffn_down.weight"),
            );

            append_mistral_layer(
                &mut program,
                x,
                inv_dim,
                eps,
                ones,
                inv_sqrt_head_dim,
                cos,
                sin,
                group_ones,
                is_future,
                neg_infinity,
                group,
                attn_norm_weight,
                ffn_norm_weight,
                wq,
                wk,
                wv,
                wo,
                w_gate,
                w_up,
                w_down,
            )?
        } else {
            let gate_inp = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(expert_count)],
                &alloc::format!("blk.{layer}.ffn_gate_inp.weight"),
            );
            let expert_w_gate = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(expert_count),
                    Extent::Static(embedding),
                    Extent::Static(feed_forward)
                ],
                &alloc::format!("blk.{layer}.ffn_gate_exps.weight"),
            );
            let expert_w_up = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(expert_count),
                    Extent::Static(embedding),
                    Extent::Static(feed_forward)
                ],
                &alloc::format!("blk.{layer}.ffn_up_exps.weight"),
            );
            let expert_w_down = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(expert_count),
                    Extent::Static(feed_forward),
                    Extent::Static(embedding)
                ],
                &alloc::format!("blk.{layer}.ffn_down_exps.weight"),
            );

            let (next_x, site) = append_mistral_moe_layer(
                &mut program,
                layer,
                x,
                inv_dim,
                eps,
                ones,
                inv_sqrt_head_dim,
                cos,
                sin,
                group_ones,
                is_future,
                neg_infinity,
                group,
                attn_norm_weight,
                ffn_norm_weight,
                wq,
                wk,
                wv,
                wo,
                gate_inp,
                expert_w_gate,
                expert_w_up,
                expert_w_down,
                expert_count,
                expert_used_count,
            )?;
            moe_sites.push(site);
            next_x
        };
    }

    let output_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding)],
        "output_norm.weight",
    );
    let normed_final = rmsnorm(&mut program, x, output_norm_weight, inv_dim, eps)?;

    let lm_head = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding), Extent::Static(vocab)],
        "output.weight",
    );
    let logits_product = elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed_final, "sd->sdv"), (lm_head, "dv->sdv")],
    )?;
    reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        logits_product,
        "sdv->sdv",
        "sv->sdv",
    )?;

    Ok(program)
}

/// One transformer layer's per-position outputs the caller appends into its
/// own key/value cache for the next call: `k_even`/`k_odd` are RoPE-rotated
/// already (the same halves this module's per-layer attention consumes
/// directly, so a later call never re-derives RoPE for a position it has
/// already seen), `v` is the un-rotated projected value. Not a library type
/// outside this module — three [`NodeId`]s a caller collects once per layer,
/// nothing more.
pub type CachedLayerRoots = (NodeId, NodeId, NodeId);

/// [`mistral_cached_forward_program_with_experts`]'s two named roots:
/// `logits` (the vocab-projection reduce, this program's terminal node) and
/// `hidden` (`normed_final` -- the LAST-norm activation `logits` is
/// projected FROM, one layer earlier in the graph). Before this type
/// existed, a caller that needed `hidden` (an embedding pooling the final
/// hidden state rather than decoding a token) had no way to reach it except
/// `NodeId(logits.0 - 3)` -- arithmetic over [`op::append`]'s id-is-index
/// invariant that silently breaks the moment a refactor inserts or removes
/// one node between `hidden` and `logits`. A plain two-field struct instead
/// of widening this builder's return arity again: every existing caller
/// that only wants `logits` destructures `ForwardRoots { logits, .. }` (one
/// pattern, no behavior change); `proxima-model-interop`'s
/// `LoadedModel::embed` is the one caller that reads `hidden` too.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForwardRoots {
    pub logits: NodeId,
    pub hidden: NodeId,
}

/// [`mistral_single_range_cached_forward_program`]'s own return shape:
/// the lowered program, its `logits` root, one [`CachedLayerRoots`] per
/// layer, and ROW 326/328's [`DuplicateHeadPosition`] scratch output
/// (`Some` only when that position is not [`DuplicateHeadPosition::None`]
/// -- see the function's own doc).
type SingleRangeForwardProgram = (Vec<Op>, NodeId, Vec<CachedLayerRoots>, Option<NodeId>);

/// [`mistral_cached_forward_program_with_experts_and_layer_taps`]'s own
/// return shape: the lowered program, its [`ForwardRoots`], one
/// [`CachedLayerRoots`] per layer, one residual [`NodeId`] per layer
/// (that function's own doc on what the fourth element is for), and one
/// [`MoeSite`] per MoE layer (empty on a dense checkpoint).
type MistralMoeForwardProgramWithLayerTaps =
    (Vec<Op>, ForwardRoots, Vec<CachedLayerRoots>, Vec<NodeId>, MoeSites);

/// Where, if anywhere, the ROW 326/328 diagnostic duplicate `output.weight`
/// reduce is emitted relative to the real head -- ROW 328 turns ROW 326's
/// original bool into this 3-way position to test whether the ~1.8ms head
/// cost follows a fixed slot in program order or the first GPU touch of the
/// `output.weight` range after 4 GB of other layer traffic has streamed
/// through the same no-copy mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DuplicateHeadPosition {
    /// no scratch reduce -- every production checkpoint.
    #[default]
    None,
    /// scratch reduce reads `x`, the raw embedding lookup output, before
    /// layer 0 runs -- first dispatch of the token, `output.weight` is
    /// touched before any other per-layer weight this token.
    Before,
    /// scratch reduce reads `normed_final` immediately after the real
    /// head -- last dispatch of the token, ROW 326's original behavior.
    After,
}

/// `append_qwen35_dense_attention_layer`'s own per-position cache roots --
/// [`CachedLayerRoots`]'s 4-wide counterpart, one extra [`NodeId`] for the
/// partial-rotary remainder [`CachedLayerRoots`] has no room for: `k_first`/
/// `k_second` are this checkpoint's split-half (NEOX/IMROPE-style) RoPE
/// halves of the rotated prefix (`k[..., :rotary_dim]`,
/// `modeling_qwen3_next.py:205-210`), `k_pass` is the untouched remainder
/// (`k[..., rotary_dim:]`, `modeling_qwen3_next.py:206`, concatenated back
/// in the oracle, never dropped), `v` the un-rotated projected value.
pub type Qwen35DenseAttentionRoots = (NodeId, NodeId, NodeId, NodeId);

/// [`append_mistral_layer`]'s key/value-cached counterpart: `x` carries only
/// the `new` positions this call introduces (`s`, sized by symbol 0), and
/// attention blends two disjoint key/value sources instead of one —
/// `k_even_cache`/`k_odd_cache`/`v_cache` (already-rotated positions from
/// every earlier call, bound [`Op::Input`] sized by symbol 1) and this
/// call's own freshly projected/rotated `k_new`/`v_new` (`w`, same size as
/// `s`, computed in-graph). Two [`Op::Reduce`] blocks — one per source —
/// combine through online-softmax arithmetic (`Maximum` for the shared max,
/// `Add` for the shared normalizer) rather than a literal concatenation:
/// [`Reduce::out_map`] must stay a pure projection
/// (`shape::project_output_shape`'s own doc), so nothing upstream of a
/// reduce can splice two tensors into one axis. The masking-only-within-`s,w`
/// asymmetry is what makes this correct without a `cached_len` scalar: a
/// cached key is definitionally in the past of every new query, and
/// `is_future` (built once by [`causal_mask`], sized `[s,w]` since `w` and
/// `s` share symbol 0's extent) already forbids a new query attending a
/// later new key, so the cached block never needs masking at all.
///
/// Returns `(x_next, k_new_even, k_new_odd, v_new)` — `x_next` feeds the next
/// layer (or the final RMSNorm/LM head after the last one), and the other
/// three are this layer's [`CachedLayerRoots`] for the caller to append.
///
/// `qk_norm`, when `Some((q_norm_weight, k_norm_weight, inv_head_dim))`, runs
/// [`rmsnorm_per_head`] on `q`/`k_new` right after their projection and
/// strictly BEFORE RoPE -- Qwen3's own per-head QK-norm
/// (`Qwen3Attention.q_norm`/`.k_norm`, `modeling_qwen3.py`, applied to
/// `query_states`/`key_states` before `apply_rotary_pos_emb`). `None` skips
/// both calls entirely, leaving `q`/`k_new` exactly as
/// [`mistral_cached_forward_program`]'s own dense checkpoints have always
/// computed them -- this one flag is what lets a single layer builder serve
/// both architectures rather than forking a parallel copy for the two extra
/// ops Qwen3 needs.
///
/// `paired_gate_up_reduce`, when `true`, requires the caller to have passed
/// the SAME `NodeId` for both `w_gate` and `w_up` -- a single `[2,
/// feed_forward, embedding]` leaf (gate rows then up rows, one dispatch
/// binds it) rather than two separate `[embedding, feed_forward]` leaves.
/// `gate`/`up` are then read back out of ONE `Op::Reduce`'s `[s, feed_forward]`
/// output via the parity axis fixed at `0`/`1` (the constant-offset
/// axis-expression grammar `spec.rs`'s own module doc already carries,
/// `"s,0*s+K,g->sg"` -- a zero-coefficient term on an unrelated iteration
/// letter selects a compile-time-fixed operand axis without adding an
/// `IndexMap` variant). `false` reproduces today's two independent matvecs
/// byte-for-byte -- every op below this branch is unaffected by which side
/// ran. Default `false` at every production call site; flipping it removes
/// one `Op::Reduce` (and its own kernel dispatch) per layer at load-time
/// cost only when the checkpoint's `ffn_gate`/`ffn_up` tensors are not
/// byte-adjacent (`proxima-model-interop::bind::bind_matmul_weight_paired`).
///
/// `fused_qkv_reduce`, when `true`, requires `query_heads` (this is the ONE
/// extra scalar this branch needs that `paired_gate_up_reduce` did not: q's
/// row count differs from k's/v's under GQA, so the flat row axis cannot
/// be recovered from `group`/`head_dim` alone), `head_shape_ones`/
/// `kv_head_shape_ones` (`[query_heads, head_dim]`/`[kv_heads, head_dim]`
/// constants of `1.0`, this function's own doc on those two parameters
/// below), and the SAME `NodeId` passed for `wq`, `wk`, `wv` -- a single
/// `[query_heads + 2 * kv_heads, head_dim, embedding]`-flattened-to-`[rows,
/// embedding]` leaf (q rows, then k rows, then v rows) rather than three
/// separate `[embedding, heads, head_dim]` leaves.
///
/// The IR CANNOT split the fused reduce's one real `[s, rows]` axis back
/// into two independent virtual sub-axes (`h`/`d` for q, `u`/`d` for k/v)
/// from a single operand alone -- confirmed empirically
/// (`shape::infer`'s own `UnconstrainedDim`, not inferred): a compound
/// axis term like `"{head_dim}*h+d"` needs BOTH `h`'s and `d`'s extents
/// pinned by SOME operand's own real, uncompounded axis, and an
/// `Op::Elementwise`'s operand count is fixed to its `ScalarOp`'s arity
/// (`op::ScalarOp::arity`), so there is no room to add a pure
/// shape-providing operand to an already-binary op (`Multiply(gate,
/// weight)`) the way `paired_gate_up_reduce`'s zero-coefficient parity
/// trick could ride a term that was already a bare constant. This is why
/// `paired_gate_up_reduce` (identical row counts either side of its split)
/// could read `gate`/`up` back with ZERO extra dispatch, and this flag
/// (three DIFFERENT row counts under GQA, so no single shared axis exists
/// to split on) cannot: `q_raw`/`k_new_raw`/`v_new` each need their own
/// `ScalarOp::Multiply`-against-a-ones-shaped-constant extract (arity 2,
/// satisfying shape inference; the ones constant contributes only shape,
/// value `1.0`, so the extracted values are bit-identical to a direct
/// read) -- three small dispatches, not one, added back. Net per layer: 3
/// reduces removed, 1 fused reduce added, 3 small extracts added -- ONE
/// MORE dispatch, not fewer. This corrects this feature's own premise (a
/// bandwidth-only reading of ROW 336 predicted `-2`/layer); the measured
/// win, if any, is per-dispatch bandwidth on the one big reduce, not
/// dispatch count. Requires `qk_norm` be `None` -- QK-norm's
/// `rmsnorm_per_head` call needs `q_raw`/`k_new_raw` as their own
/// full-shape node regardless, which this branch already provides via the
/// same extract, but no call site in this crate combines the two flags
/// today and the combination is untested.
/// Where `append_mistral_cached_layer` reads `q`/`k_new`/`v_new` from --
/// [`Self::Split`] is today's three independent reduces; [`Self::Fused`]
/// carries the one shared flat-row reduce plus the byte offset `v_new`'s
/// rows start at within it (`fused_qkv_reduce`'s own doc on that function).
enum QkvSource {
    Split,
    Fused {
        node: NodeId,
        v_offset: u32,
        head_dim: u32,
    },
}

#[allow(clippy::too_many_arguments)]
pub fn append_mistral_cached_layer(
    program: &mut Vec<Op>,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    ones: NodeId,
    inv_sqrt_head_dim: NodeId,
    cos_new: NodeId,
    sin_new: NodeId,
    group_ones: NodeId,
    head_shape_ones: NodeId,
    kv_head_shape_ones: NodeId,
    is_future: NodeId,
    group: u32,
    head_dim: u32,
    query_heads: u32,
    attn_norm_weight: NodeId,
    ffn_norm_weight: NodeId,
    wq: NodeId,
    wk: NodeId,
    wv: NodeId,
    wo: NodeId,
    w_gate: NodeId,
    w_up: NodeId,
    w_down: NodeId,
    k_even_cache: NodeId,
    k_odd_cache: NodeId,
    v_cache: NodeId,
    qk_norm: Option<(NodeId, NodeId, NodeId)>,
    paired_gate_up_reduce: bool,
    fused_qkv_reduce: bool,
) -> Result<(NodeId, CachedLayerRoots), TensorError> {
    let normed = rmsnorm(program, x, attn_norm_weight, inv_dim, eps)?;
    let kv_heads = query_heads / group;

    let (q_raw, k_new_raw, v_new_source): (NodeId, NodeId, QkvSource) = if fused_qkv_reduce {
        let qkv_product = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(normed, "si->spi"), (wq, "pi->spi")],
        )?;
        let qkv_reduced = reduce(
            program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            qkv_product,
            "spi->spi",
            "sp->spi",
        )?;
        let q_raw = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[
                (qkv_reduced, alloc::format!("s,{head_dim}*h+d->shd").as_str()),
                (head_shape_ones, "hd->shd"),
            ],
        )?;
        let k_offset = query_heads * head_dim;
        let k_new_raw = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[
                (
                    qkv_reduced,
                    alloc::format!("s,{head_dim}*u+d+{k_offset}->sud").as_str(),
                ),
                (kv_head_shape_ones, "ud->sud"),
            ],
        )?;
        (
            q_raw,
            k_new_raw,
            QkvSource::Fused {
                node: qkv_reduced,
                v_offset: (query_heads + kv_heads) * head_dim,
                head_dim,
            },
        )
    } else {
        let q_product = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(normed, "si->shdi"), (wq, "ihd->shdi")],
        )?;
        let q_raw = reduce(
            program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            q_product,
            "shdi->shdi",
            "shd->shdi",
        )?;

        let k_new_product = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(normed, "si->sudi"), (wk, "iud->sudi")],
        )?;
        let k_new_raw = reduce(
            program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            k_new_product,
            "sudi->sudi",
            "sud->sudi",
        )?;
        (q_raw, k_new_raw, QkvSource::Split)
    };

    let (q, k_new) = match qk_norm {
        Some((q_norm_weight, k_norm_weight, inv_head_dim)) => {
            let q = rmsnorm_per_head(program, q_raw, q_norm_weight, inv_head_dim, eps, "h")?;
            let k_new = rmsnorm_per_head(program, k_new_raw, k_norm_weight, inv_head_dim, eps, "u")?;
            (q, k_new)
        }
        None => (q_raw, k_new_raw),
    };

    let v_new = match v_new_source {
        QkvSource::Fused { node, v_offset, head_dim } => elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[
                (
                    node,
                    alloc::format!("s,{head_dim}*u+d+{v_offset}->sud").as_str(),
                ),
                (kv_head_shape_ones, "ud->sud"),
            ],
        )?,
        QkvSource::Split => {
            let v_product = elementwise(
                program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(normed, "si->sudi"), (wv, "iud->sudi")],
            )?;
            reduce(
                program,
                DType::Float32,
                ScalarOp::Add,
                ReduceInit::Zero,
                v_product,
                "sudi->sudi",
                "sud->sudi",
            )?
        }
    };

    // Two incompatible RoPE pairings live behind `qk_norm.is_some()`, not a
    // separate flag: llama.cpp's own GGUF converter permutes a "normal"
    // (interleaved, `(2*i, 2*i+1)`) architecture's on-disk Q/K rows into
    // that pairing at conversion time (Mistral/LLaMA), but never touches a
    // NEOX-style architecture's rows (Qwen -- `LLM_ARCH_QWEN3`'s own
    // `rope_type = LLAMA_ROPE_TYPE_NEOX`), which stay in HF's native
    // split-half layout (`x[..half]`/`x[half..]`) on disk. Every checkpoint
    // this crate has bound with `attn_q_norm.weight` present is exactly the
    // NEOX family, so the same presence check that gates QK-norm also
    // selects the matching RoPE pairing -- see
    // [`append_qwen35_dense_attention_layer`]'s own split-half section,
    // which this mirrors at `pass_dim = 0` (Qwen3's rotary width equals its
    // full head width, so there is no untouched remainder).
    // `q`/`k_new` are real, fully materialized `[s,h,d]`/`[s,u,d]` nodes
    // under BOTH `QkvSource` variants (the `Multiply`-by-shape-constant
    // extract above already re-materializes them under `Fused`), so every
    // op below reads them exactly as the split path always has -- zero
    // further changes needed downstream of this point.
    let (rotated_q_even, rotated_q_odd, rotated_k_new_even, rotated_k_new_odd) = match qk_norm {
        Some(_) => {
            let pairs = head_dim / 2;
            let (rotated_q_first, rotated_q_second) =
                fused_rope_pair(program, q, 'h', cos_new, sin_new, RopePairing::SplitHalf { pairs })?;
            let (rotated_k_first, rotated_k_second) =
                fused_rope_pair(program, k_new, 'u', cos_new, sin_new, RopePairing::SplitHalf { pairs })?;

            (rotated_q_first, rotated_q_second, rotated_k_first, rotated_k_second)
        }
        None => {
            let (rotated_q_even, rotated_q_odd) =
                fused_rope_pair(program, q, 'h', cos_new, sin_new, RopePairing::Interleaved)?;
            let (rotated_k_new_even, rotated_k_new_odd) =
                fused_rope_pair(program, k_new, 'u', cos_new, sin_new, RopePairing::Interleaved)?;

            (rotated_q_even, rotated_q_odd, rotated_k_new_even, rotated_k_new_odd)
        }
    };

    let group_map = alloc::format!("s,{group}*u+g,i->sugi");
    let q_even_grouped = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (rotated_q_even, group_map.as_str()),
            (group_ones, "ug->sugi"),
        ],
    )?;
    let q_odd_grouped = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (rotated_q_odd, group_map.as_str()),
            (group_ones, "ug->sugi"),
        ],
    )?;

    // cached block: query `s` against every already-rotated cached key `t`
    // (symbol 1's extent, zero on the very first call) -- never masked, a
    // cached position is always in the past of a new query.
    let score_cached_even_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (q_even_grouped, "sugi->stugi"),
            (k_even_cache, "tui->stugi"),
        ],
    )?;
    let score_cached_even = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_cached_even_product,
        "stugi->stugi",
        "stug->stugi",
    )?;
    let score_cached_odd_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(q_odd_grouped, "sugi->stugi"), (k_odd_cache, "tui->stugi")],
    )?;
    let score_cached_odd = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_cached_odd_product,
        "stugi->stugi",
        "stug->stugi",
    )?;
    let score_cached = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[
            (score_cached_even, "stug->stug"),
            (score_cached_odd, "stug->stug"),
        ],
    )?;
    let score_cached_scaled = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(score_cached, "stug->stug"), (inv_sqrt_head_dim, "->stug")],
    )?;

    // new block: query `s` against this call's own freshly rotated key `w`
    // (symbol 0's extent, same range as `s`) -- causal within the block,
    // reusing `is_future` unchanged since it is already `[s, w]`-shaped.
    let score_new_even_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (q_even_grouped, "sugi->swugi"),
            (rotated_k_new_even, "wui->swugi"),
        ],
    )?;
    let score_new_even = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_new_even_product,
        "swugi->swugi",
        "swug->swugi",
    )?;
    let score_new_odd_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (q_odd_grouped, "sugi->swugi"),
            (rotated_k_new_odd, "wui->swugi"),
        ],
    )?;
    let score_new_odd = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_new_odd_product,
        "swugi->swugi",
        "swug->swugi",
    )?;
    let score_new = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[
            (score_new_even, "swug->swug"),
            (score_new_odd, "swug->swug"),
        ],
    )?;
    let score_new_scaled = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(score_new, "swug->swug"), (inv_sqrt_head_dim, "->swug")],
    )?;
    let neg_infinity = scalar_constant(program, f32::NEG_INFINITY);
    let score_new_masked = elementwise(
        program,
        DType::Float32,
        ScalarOp::Select,
        &[
            (is_future, "sw->swug"),
            (neg_infinity, "->swug"),
            (score_new_scaled, "swug->swug"),
        ],
    )?;

    // online-softmax combine: two disjoint key ranges, one shared max and
    // one shared normalizer, no literal concatenation anywhere.
    let score_max_cached = reduce(
        program,
        DType::Float32,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        score_cached_scaled,
        "stug->stug",
        "sug->stug",
    )?;
    let score_max_new = reduce(
        program,
        DType::Float32,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        score_new_masked,
        "swug->swug",
        "sug->swug",
    )?;
    let global_max = elementwise(
        program,
        DType::Float32,
        ScalarOp::Maximum,
        &[(score_max_cached, "sug->sug"), (score_max_new, "sug->sug")],
    )?;

    let shifted_cached = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[
            (score_cached_scaled, "stug->stug"),
            (global_max, "sug->stug"),
        ],
    )?;
    let weights_cached = elementwise(
        program,
        DType::Float32,
        ScalarOp::Exponential,
        &[(shifted_cached, "stug->stug")],
    )?;
    let shifted_new = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(score_new_masked, "swug->swug"), (global_max, "sug->swug")],
    )?;
    let weights_new = elementwise(
        program,
        DType::Float32,
        ScalarOp::Exponential,
        &[(shifted_new, "swug->swug")],
    )?;

    let sum_cached = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        weights_cached,
        "stug->stug",
        "sug->stug",
    )?;
    let sum_new = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        weights_new,
        "swug->swug",
        "sug->swug",
    )?;
    let weight_sum = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(sum_cached, "sug->sug"), (sum_new, "sug->sug")],
    )?;
    let inv_weight_sum = elementwise(
        program,
        DType::Float32,
        ScalarOp::Reciprocal,
        &[(weight_sum, "sug->sug")],
    )?;

    let attended_cached_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(weights_cached, "stug->stugd"), (v_cache, "tud->stugd")],
    )?;
    let attended_cached = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        attended_cached_product,
        "stugd->stugd",
        "sugd->stugd",
    )?;
    let attended_new_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(weights_new, "swug->swugd"), (v_new, "wud->swugd")],
    )?;
    let attended_new = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        attended_new_product,
        "swugd->swugd",
        "sugd->swugd",
    )?;
    let attended_sum = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[
            (attended_cached, "sugd->sugd"),
            (attended_new, "sugd->sugd"),
        ],
    )?;
    let attended = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(attended_sum, "sugd->sugd"), (inv_weight_sum, "sug->sugd")],
    )?;

    #[cfg(feature = "instrument")]
    instrument::record_online_softmax_block_range(score_max_cached, attended);

    let wo_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(attended, "sugd->sugdo"), (wo, "ugdo->sugdo")],
    )?;
    let attn_out = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        wo_product,
        "sugdo->sugdo",
        "so->sugdo",
    )?;

    let residual1 = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(attn_out, "sd->sd"), (x, "sd->sd")],
    )?;

    let normed2 = rmsnorm(program, residual1, ffn_norm_weight, inv_dim, eps)?;

    // `paired_gate_up_reduce`: one `Op::Reduce` over `w_gate` (== `w_up`,
    // caller's contract, see this function's own doc) read as `[2,
    // feed_forward, embedding]` replaces the two independent matvecs below.
    // `out_map`'s letter order ("spg", not "sgp") is load-bearing, not
    // stylistic: `proxima_tensor::bind::correct_packed_matmul_layouts`
    // derives a packed weight's native stride per output axis from
    // `output_axes`' LISTED order (last-listed axis lands innermost, closest
    // to the reduced axis) -- "spg" places the broadcast `s` axis outermost
    // (its stride is discarded either way) and `g` innermost-of-features so
    // its native stride comes out `embedding` (one row), leaving `p`'s
    // native stride `feed_forward * embedding` (one whole gate/up half) --
    // exactly the real concatenated checkpoint's byte layout. `"sgp"` derives
    // the opposite (interleaved) stride pair and silently mis-reads the
    // buffer.
    let (gate, up, gate_map, up_map): (NodeId, NodeId, &str, &str) = if paired_gate_up_reduce {
        let paired_product = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(normed2, "sd->sdgp"), (w_gate, "pgd->sdgp")],
        )?;
        let paired_result = reduce(
            program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            paired_product,
            "sdgp->sdgp",
            "spg->sdgp",
        )?;
        (
            paired_result,
            paired_result,
            "s,0*s+0,g->sg",
            "s,0*s+1,g->sg",
        )
    } else {
        let gate_product = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(normed2, "sd->sdg"), (w_gate, "dg->sdg")],
        )?;
        let gate = reduce(
            program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            gate_product,
            "sdg->sdg",
            "sg->sdg",
        )?;
        let up_product = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(normed2, "sd->sdg"), (w_up, "dg->sdg")],
        )?;
        let up = reduce(
            program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            up_product,
            "sdg->sdg",
            "sg->sdg",
        )?;
        (gate, up, "sg->sg", "sg->sg")
    };

    let neg_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Negate,
        &[(gate, gate_map)],
    )?;
    let exp_neg_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Exponential,
        &[(neg_gate, "sg->sg")],
    )?;
    let one_plus_exp = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(exp_neg_gate, "sg->sg"), (ones, "->sg")],
    )?;
    let sigmoid_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Reciprocal,
        &[(one_plus_exp, "sg->sg")],
    )?;
    let silu_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(gate, gate_map), (sigmoid_gate, "sg->sg")],
    )?;
    let ffn_hidden = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(silu_gate, "sg->sg"), (up, up_map)],
    )?;

    let down_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(ffn_hidden, "sg->sgd"), (w_down, "gd->sgd")],
    )?;
    let ffn_out = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        down_product,
        "sgd->sgd",
        "sd->sgd",
    )?;

    let x_next = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(ffn_out, "sd->sd"), (residual1, "sd->sd")],
    )?;

    Ok((x_next, (rotated_k_new_even, rotated_k_new_odd, v_new)))
}

/// Hyper-connections replace a single residual stream with `hc` parallel
/// copies (`x`, shape `[tokens, hc, embedding]`, letters `s,h,i`) and mix
/// them down to one stream a token mixer or FFN can consume -- reference:
/// PR 27742 line 2705-2751, `build_hc_mix`. Built entirely from
/// [`elementwise`]/[`reduce`]/[`silu`]/[`sigmoid`], the same primitives
/// every other builder in this module composes -- no new [`Op`] variant
/// (`flash-next-plan.md` §3's own "hyper-connections" row: "fully
/// expressible IN the existing `Op` vocabulary").
///
/// Steps, each named after its `build_hc_mix` counterpart:
/// 1. Grouped RMSNorm (reference line 2717-2722): `x` normalized over the
///    embedding axis `i` *per stream* `h`, then scaled by `w_norm`
///    (`[hc, embedding]`, one gamma per `(stream, channel)` pair -- the
///    checkpoint's own flat `[hc_dim]` gamma reshaped, never a single
///    shared-across-streams gamma the way [`rmsnorm_per_head`]'s `gamma`
///    is shared across heads).
/// 2. Low-rank gate (reference line 2724-2729): `xn` down-projected
///    (`w_down`, `[hc, embedding, low_rank]`) to `[tokens, low_rank]`,
///    scaled by `1/hc`, `silu`'d, up-projected (`w_up`,
///    `[low_rank, hc, embedding]`) back to `[tokens, hc, embedding]`, then
///    `sigmoid`'d into a gate multiplied against `xn`.
/// 3. Mean-collapse (reference line 2732-2743): the gated `[tokens, hc,
///    embedding]` stream summed over `h` and scaled by `1/hc`.
/// 4. Optional inject (reference line 2745-2748): `w_inject`
///    (`[hc, embedding, hc]`) projects `xn` to a `[tokens, hc]` scatter
///    weight [`append_hyper_connection_combine`] consumes -- `None` for the
///    final output mixer (reference line 2860-2862: "there is no
///    output_norm: the final hyper-connection mixer carries it"), `Some`
///    for every per-layer attn/ffn hyper-connection module (reference line
///    2816-2821, 2843-2848).
///
/// `inv_dim` is `1/embedding` (the RMSNorm mean, [`rmsnorm`]'s own
/// parameter); `inv_hc` is `1/hc`, reused for the low-rank gate's scale,
/// the mean-collapse scale, and (when `w_inject` is `Some`) the inject
/// scale the caller's own [`append_hyper_connection_combine`] finishes.
///
/// Returns `(mixed, inject)`, `mixed` shaped `[tokens, embedding]`,
/// `inject` shaped `[tokens, hc]` (`Some` iff `w_inject` was `Some`).
///
/// No `qwen4exp_forward_program` call site lands in this crate (that
/// assembly is model-specific and lives in its own consuming crate); this
/// builder is public so that crate can compose one. See
/// [`qwen35_forward_program`] for this crate's own worked example of
/// wiring per-layer builders like this one into a full program.
#[allow(clippy::too_many_arguments)]
pub fn append_hyper_connection_mix(
    program: &mut Vec<Op>,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    inv_hc: NodeId,
    one: NodeId,
    w_norm: NodeId,
    w_down: NodeId,
    w_up: NodeId,
    w_inject: Option<NodeId>,
) -> Result<(NodeId, Option<NodeId>), TensorError> {
    let squared = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(x, "shi->shi"), (x, "shi->shi")])?;
    let sum_squares = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, squared, "shi->shi", "sh->shi")?;
    let mean_square = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(sum_squares, "sh->sh"), (inv_dim, "->sh")])?;
    let mean_square_eps = elementwise(program, DType::Float32, ScalarOp::Add, &[(mean_square, "sh->sh"), (eps, "s->sh")])?;
    let rms = elementwise(program, DType::Float32, ScalarOp::SquareRoot, &[(mean_square_eps, "sh->sh")])?;
    let inv_rms = elementwise(program, DType::Float32, ScalarOp::Reciprocal, &[(rms, "sh->sh")])?;
    let normed = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(x, "shi->shi"), (inv_rms, "sh->shi")])?;
    let xn = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(normed, "shi->shi"), (w_norm, "hi->shi")])?;

    let down_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(xn, "shi->shir"), (w_down, "hir->shir")])?;
    let down_sum_i = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, down_product, "shir->shir", "shr->shir")?;
    let lo = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, down_sum_i, "shr->shr", "sr->shr")?;
    let lo_scaled = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(lo, "sr->sr"), (inv_hc, "->sr")])?;
    let lo_silu = silu(program, lo_scaled, one, "sr->sr")?;

    let up_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(lo_silu, "sr->srhi"), (w_up, "rhi->srhi")])?;
    let up_sum_r = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, up_product, "srhi->srhi", "shi->srhi")?;
    let gate = sigmoid(program, up_sum_r, one, "shi->shi")?;

    let gated = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(xn, "shi->shi"), (gate, "shi->shi")])?;
    let mixed_sum = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, gated, "shi->shi", "si->shi")?;
    let mixed = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(mixed_sum, "si->si"), (inv_hc, "->si")])?;

    let inject = match w_inject {
        Some(w_inject) => {
            let inject_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(xn, "shi->shio"), (w_inject, "hio->shio")])?;
            let inject_sum_i = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, inject_product, "shio->shio", "sho->shio")?;
            let inject_flat = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, inject_sum_i, "sho->sho", "so->sho")?;
            Some(inject_flat)
        }
        None => None,
    };

    Ok((mixed, inject))
}

/// The residual side of a hyper-connection module -- reference: PR 27742
/// line 2753-2773, `build_hc_combine`: `2*sigmoid(inject/hc)` centres the
/// per-stream scatter weight on `1`, so a zero injection degenerates to a
/// plain residual add, then `block_out` (`[tokens, embedding]`) broadcasts
/// across every stream, scaled by that weight, and adds into `residual`
/// (`[tokens, hc, embedding]`). Pairs with
/// [`append_hyper_connection_mix`]'s `Some(w_inject)` arm; the final output
/// mixer has no combine call (reference line 2860-2869: the mixed stream
/// feeds `output` directly).
///
/// Returns the updated `[tokens, hc, embedding]` residual.
///
/// Same rationale as [`append_hyper_connection_mix`]'s own doc: no
/// production call site in this crate, public so a foreign architecture
/// crate can compose one. See [`qwen35_forward_program`] for this crate's
/// own worked example of a full per-layer builder chain.
#[allow(clippy::too_many_arguments)]
pub fn append_hyper_connection_combine(
    program: &mut Vec<Op>,
    residual: NodeId,
    block_out: NodeId,
    inject: NodeId,
    inv_hc: NodeId,
    one: NodeId,
    two: NodeId,
) -> Result<NodeId, TensorError> {
    let inject_scaled = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(inject, "sh->sh"), (inv_hc, "->sh")])?;
    let inject_sigmoid = sigmoid(program, inject_scaled, one, "sh->sh")?;
    let weight = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(inject_sigmoid, "sh->sh"), (two, "->sh")])?;

    let broadcast_out = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(block_out, "si->shi"), (weight, "sh->shi")])?;
    elementwise(program, DType::Float32, ScalarOp::Add, &[(residual, "shi->shi"), (broadcast_out, "shi->shi")])
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
mod hyper_connection_tests {
    use super::*;

    /// Deterministic xorshift64, not a real RNG -- reproducible "random"
    /// f32 inputs without a `rand` dependency in this crate's test code.
    fn next_f32(state: &mut u64) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (((*state >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5) * 2.0
    }

    fn filled(state: &mut u64, len: usize) -> Vec<f32> {
        (0..len).map(|_| next_f32(state)).collect()
    }

    fn sigmoid_f64(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn silu_f64(x: f64) -> f64 {
        x * sigmoid_f64(x)
    }

    /// f64 loop from PR 27742 line 2705-2751's own equations, independent
    /// of every builder above -- the oracle
    /// [`append_hyper_connection_mix`]'s test compares against.
    #[allow(clippy::too_many_arguments)]
    fn hc_mix_f64_reference(
        tokens: usize,
        hc: usize,
        embedding: usize,
        low_rank: usize,
        x: &[f32],
        w_norm: &[f32],
        w_down: &[f32],
        w_up: &[f32],
        w_inject: Option<&[f32]>,
        eps: f64,
    ) -> (Vec<f64>, Option<Vec<f64>>) {
        let at_x = |s: usize, h: usize, i: usize| f64::from(x[(s * hc + h) * embedding + i]);
        let at_w_norm = |h: usize, i: usize| f64::from(w_norm[h * embedding + i]);
        let at_w_down = |h: usize, i: usize, r: usize| f64::from(w_down[(h * embedding + i) * low_rank + r]);
        let at_w_up = |r: usize, h: usize, i: usize| f64::from(w_up[(r * hc + h) * embedding + i]);

        let mut xn = alloc::vec![0.0f64; tokens * hc * embedding];
        for s in 0..tokens {
            for h in 0..hc {
                let sum_sq: f64 = (0..embedding).map(|i| at_x(s, h, i).powi(2)).sum();
                let inv_rms = 1.0 / ((sum_sq / embedding as f64) + eps).sqrt();
                for i in 0..embedding {
                    xn[(s * hc + h) * embedding + i] = at_x(s, h, i) * inv_rms * at_w_norm(h, i);
                }
            }
        }
        let at_xn = |s: usize, h: usize, i: usize| xn[(s * hc + h) * embedding + i];

        let mut mixed = alloc::vec![0.0f64; tokens * embedding];
        for s in 0..tokens {
            let lo: Vec<f64> = (0..low_rank)
                .map(|r| {
                    let raw: f64 = (0..hc)
                        .flat_map(|h| (0..embedding).map(move |i| (h, i)))
                        .map(|(h, i)| at_xn(s, h, i) * at_w_down(h, i, r))
                        .sum();
                    silu_f64(raw / hc as f64)
                })
                .collect();
            for h in 0..hc {
                for i in 0..embedding {
                    let up: f64 = (0..low_rank).map(|r| lo[r] * at_w_up(r, h, i)).sum();
                    mixed[s * embedding + i] += at_xn(s, h, i) * sigmoid_f64(up);
                }
            }
            for i in 0..embedding {
                mixed[s * embedding + i] /= hc as f64;
            }
        }

        let inject = w_inject.map(|w_inject| {
            let at_w_inject = |h: usize, i: usize, o: usize| f64::from(w_inject[(h * embedding + i) * hc + o]);
            let mut inject = alloc::vec![0.0f64; tokens * hc];
            for s in 0..tokens {
                for o in 0..hc {
                    inject[s * hc + o] = (0..hc)
                        .flat_map(|h| (0..embedding).map(move |i| (h, i)))
                        .map(|(h, i)| at_xn(s, h, i) * at_w_inject(h, i, o))
                        .sum();
                }
            }
            inject
        });

        (mixed, inject)
    }

    fn hc_combine_f64_reference(
        tokens: usize,
        hc: usize,
        embedding: usize,
        residual: &[f32],
        block_out: &[f32],
        inject: &[f64],
    ) -> Vec<f64> {
        let mut result = alloc::vec![0.0f64; tokens * hc * embedding];
        for s in 0..tokens {
            for h in 0..hc {
                let weight = 2.0 * sigmoid_f64(inject[s * hc + h] / hc as f64);
                for i in 0..embedding {
                    let residual_value = f64::from(residual[(s * hc + h) * embedding + i]);
                    let block_out_value = f64::from(block_out[s * embedding + i]);
                    result[(s * hc + h) * embedding + i] = residual_value + block_out_value * weight;
                }
            }
        }
        result
    }

    /// Builds a program exercising just [`append_hyper_connection_mix`] at
    /// `(tokens, hc, embedding, low_rank)`, evaluates it on random f32
    /// inputs, and asserts every output element matches
    /// [`hc_mix_f64_reference`]'s independent f64 loop within `1e-5`.
    fn assert_mix_matches_reference(tokens: usize, hc: usize, embedding: usize, low_rank: usize, with_inject: bool, seed: u64) {
        let mut state = seed;
        let x_data = filled(&mut state, tokens * hc * embedding);
        let w_norm_data = filled(&mut state, hc * embedding);
        let w_down_data = filled(&mut state, hc * embedding * low_rank);
        let w_up_data = filled(&mut state, low_rank * hc * embedding);
        let w_inject_data = with_inject.then(|| filled(&mut state, hc * embedding * hc));
        let eps_data = alloc::vec![1e-6f32; tokens];

        let mut program = Vec::new();
        let x = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Symbolic(0), Extent::Static(hc as u32), Extent::Static(embedding as u32)], "x");
        let w_norm = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(hc as u32), Extent::Static(embedding as u32)], "w_norm");
        let w_down = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(hc as u32), Extent::Static(embedding as u32), Extent::Static(low_rank as u32)],
            "w_down",
        );
        let w_up = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(low_rank as u32), Extent::Static(hc as u32), Extent::Static(embedding as u32)],
            "w_up",
        );
        let w_inject = with_inject.then(|| {
            input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(hc as u32), Extent::Static(embedding as u32), Extent::Static(hc as u32)],
                "w_inject",
            )
        });
        let inv_dim = scalar_constant(&mut program, 1.0 / embedding as f32);
        let eps = symbolic_leaf(&mut program, DType::Float32, "eps");
        let inv_hc = scalar_constant(&mut program, 1.0 / hc as f32);
        let one = scalar_constant(&mut program, 1.0);

        let (mixed, inject) =
            append_hyper_connection_mix(&mut program, x, inv_dim, eps, inv_hc, one, w_norm, w_down, w_up, w_inject).expect("hyper-connection mix lowers");

        let mut named: Vec<(&str, &[f32])> = alloc::vec![
            ("x", x_data.as_slice()),
            ("w_norm", w_norm_data.as_slice()),
            ("w_down", w_down_data.as_slice()),
            ("w_up", w_up_data.as_slice()),
            ("eps", eps_data.as_slice()),
        ];
        if let Some(w_inject_data) = w_inject_data.as_deref() {
            named.push(("w_inject", w_inject_data));
        }
        let mut outputs = alloc::vec![mixed];
        if let Some(inject) = inject {
            outputs.push(inject);
        }

        let evaluated =
            crate::cpu::evaluate_named(&program, &[tokens as u64], &named, &outputs).expect("hyper-connection mix evaluates");

        let (mixed_values, _) = evaluated.get(mixed).expect("mixed output present");
        let (expected_mixed, expected_inject) = hc_mix_f64_reference(
            tokens,
            hc,
            embedding,
            low_rank,
            &x_data,
            &w_norm_data,
            &w_down_data,
            &w_up_data,
            w_inject_data.as_deref(),
            1e-6,
        );
        let max_abs_diff_mixed = mixed_values
            .iter()
            .zip(expected_mixed.iter())
            .map(|(actual, expected)| (f64::from(*actual) - expected).abs())
            .fold(0.0f64, f64::max);
        assert!(max_abs_diff_mixed <= 1e-5, "mixed max-abs diff {max_abs_diff_mixed} exceeds 1e-5");

        if let Some(inject_node) = inject {
            let (inject_values, _) = evaluated.get(inject_node).expect("inject output present");
            let expected_inject = expected_inject.expect("reference computed inject when w_inject was Some");
            let max_abs_diff_inject = inject_values
                .iter()
                .zip(expected_inject.iter())
                .map(|(actual, expected)| (f64::from(*actual) - expected).abs())
                .fold(0.0f64, f64::max);
            assert!(max_abs_diff_inject <= 1e-5, "inject max-abs diff {max_abs_diff_inject} exceeds 1e-5");
        }
    }

    #[test]
    fn mix_matches_f64_reference_at_hc_2() {
        assert_mix_matches_reference(3, 2, 4, 3, true, 0x517c_c1b7_2722_0a95);
    }

    #[test]
    fn mix_matches_f64_reference_at_hc_4() {
        assert_mix_matches_reference(2, 4, 3, 2, true, 0x9e37_79b9_7f4a_7c15);
    }

    /// The final output mixer's own shape (reference: PR 27742 line
    /// 2860-2862): `w_inject = None`, no scatter weight computed.
    #[test]
    fn mix_matches_f64_reference_for_the_final_mixer_form() {
        assert_mix_matches_reference(2, 2, 3, 2, false, 0xd1b5_4a32_d192_ed03);
    }

    #[test]
    fn combine_matches_f64_reference() {
        let tokens = 3usize;
        let hc = 2usize;
        let embedding = 4usize;
        let mut state = 0xbf58_476d_1ce4_e5b9u64;
        let residual_data = filled(&mut state, tokens * hc * embedding);
        let block_out_data = filled(&mut state, tokens * embedding);
        let inject_data = filled(&mut state, tokens * hc);

        let mut program = Vec::new();
        let residual = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(0), Extent::Static(hc as u32), Extent::Static(embedding as u32)],
            "residual",
        );
        let block_out = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Symbolic(0), Extent::Static(embedding as u32)], "block_out");
        let inject = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Symbolic(0), Extent::Static(hc as u32)], "inject");
        let inv_hc = scalar_constant(&mut program, 1.0 / hc as f32);
        let one = scalar_constant(&mut program, 1.0);
        let two = scalar_constant(&mut program, 2.0);

        let combined = append_hyper_connection_combine(&mut program, residual, block_out, inject, inv_hc, one, two)
            .expect("hyper-connection combine lowers");

        let named: Vec<(&str, &[f32])> = alloc::vec![
            ("residual", residual_data.as_slice()),
            ("block_out", block_out_data.as_slice()),
            ("inject", inject_data.as_slice()),
        ];
        let evaluated = crate::cpu::evaluate_named(&program, &[tokens as u64], &named, &[combined]).expect("combine evaluates");
        let (combined_values, _) = evaluated.get(combined).expect("combined output present");

        let inject_f64: Vec<f64> = inject_data.iter().map(|value| f64::from(*value)).collect();
        let expected = hc_combine_f64_reference(tokens, hc, embedding, &residual_data, &block_out_data, &inject_f64);
        let max_abs_diff = combined_values
            .iter()
            .zip(expected.iter())
            .map(|(actual, expected)| (f64::from(*actual) - expected).abs())
            .fold(0.0f64, f64::max);
        assert!(max_abs_diff <= 1e-5, "combine max-abs diff {max_abs_diff} exceeds 1e-5");
    }
}

/// [`append_mistral_cached_layer`]'s Qwen3.5 dense-attention counterpart --
/// same cached-attention/online-softmax shape, three real differences from
/// the oracle (`modeling_qwen3_next.py`'s `Qwen3NextAttention.forward`,
/// `apply_rotary_pos_emb`; cross-checked against `qwen35.cpp`'s own
/// `build_layer_attn`) `append_mistral_cached_layer` has no room for:
///
/// 1. Q/K carry a real per-head width (`attn_head_dim`, this checkpoint's
///    own `attention.key_length`) wider than the rotary width (`rotary_dim`,
///    `rope.dimension_count`) -- RoPE only touches the first `rotary_dim`
///    columns, the remaining `attn_head_dim - rotary_dim` ("pass") columns
///    are concatenated back untouched (`modeling_qwen3_next.py:204-214`,
///    `q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]` ... `q_embed
///    = torch.cat([q_embed, q_pass], dim=-1)`). This module has no
///    concatenation primitive, so the "pass" half is never physically
///    rejoined to the "rot" half -- instead every dot product that would
///    read the concatenated vector (the attention score) is split into a
///    rot-range term plus a pass-range term and summed, which is
///    mathematically identical (`(a‖b)·(c‖d) = a·c + b·d` for disjoint
///    ranges) and is exactly the same disjoint-sum trick this function's
///    own `score_cached_even + score_cached_odd` already uses one level
///    down, one level up.
/// 2. RoPE itself is split-half (NEOX/IMROPE style, `x_rot -> (x[..d/2],
///    x[d/2..])`, GGML_ROPE_TYPE_IMROPE's own `rotate_pairs(n_dims,
///    n_dims/2, ...)`, `ggml/src/ggml-cpu/ops.cpp:6210-6211`), not
///    [`append_mistral_cached_layer`]'s interleaved `(2*i, 2*i+1)` pairing.
///    The checkpoint's declared 3-section MRoPE (`rope.dimension_sections`)
///    collapses to this same plain single-section schedule for text-only
///    input: `ggml_mrope_cache_init`'s own `theta_t`/`theta_h`/`theta_w`
///    tracks are initialized from the SAME position (`llama-graph.cpp`'s
///    `llm_graph_input_pos::set_input`, "the 3 first dims are the same" for
///    a text ubatch) and advance by the identical `theta_scale` every pair
///    index, so `theta_h`/`theta_w` are byte-identical to `theta_t` at
///    every pair regardless of which section claims that pair
///    (`ggml_mrope_cache_init`, `ops.cpp:6027-6037`) -- the declared
///    `[11, 11, 10, 0]` split is real machinery for image/video position
///    streams this checkpoint's text-only forward program never feeds.
/// 3. Q's own projection is `q_proj` fused with a same-width sigmoid gate
///    (`attn_q.weight`'s on-disk width is `2 * query_heads * attn_head_dim`,
///    `modeling_qwen3_next.py:267-268`, `torch.chunk(..., 2, dim=-1)` on the
///    LAST axis of each head's own block, `:295-298`), applied to the
///    attention output right before `o_proj`
///    (`attn_output = attn_output * torch.sigmoid(gate)`,
///    `:325-328`; `qwen35.cpp:322-328` runs the identical
///    `ggml_mul(cur, ggml_sigmoid(gate))` before `wo`).
///
/// The full-attention layer [`qwen35_forward_program`] calls once per
/// `full_attention_interval`'th layer -- see it there for the worked
/// example of wiring this builder's cache inputs and outputs.
///
/// Attention block only -- everything up to and including the residual add
/// after `o_proj`, no FFN. [`append_qwen35_dense_attention_layer`] is a thin
/// wrapper adding the dense-FFN tail on top of this; a caller whose FFN is
/// NOT dense (a routed-MoE checkpoint such as `qwen35moe`, which carries no
/// `blk.N.ffn_{gate,up,down}.weight` on its attention layers at all) calls
/// this directly and appends its own FFN + residual against the returned
/// node, the same "per-layer builders are pub so a foreign crate can
/// compose them" contract this module's own
/// `public_builders_compose_a_one_layer_forward_program` test proves for
/// [`append_qwen35_ssm_mixer`].
#[allow(clippy::too_many_arguments)]
pub fn append_qwen35_dense_attention_only(
    program: &mut Vec<Op>,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    ones: NodeId,
    inv_sqrt_attn_head_dim: NodeId,
    inv_attn_head_dim: NodeId,
    cos_new: NodeId,
    sin_new: NodeId,
    group_ones: NodeId,
    is_future: NodeId,
    cached_len: NodeId,
    group: u32,
    rotary_dim: u32,
    attn_head_dim: u32,
    attn_norm_weight: NodeId,
    q_norm_weight: NodeId,
    k_norm_weight: NodeId,
    wq: NodeId,
    w_gate_q: NodeId,
    wk: NodeId,
    wv: NodeId,
    wo: NodeId,
    k_first_cache: NodeId,
    k_second_cache: NodeId,
    k_pass_cache: NodeId,
    v_cache: NodeId,
) -> Result<(NodeId, Qwen35DenseAttentionRoots), TensorError> {
    let pairs = rotary_dim / 2;
    let pass_dim = attn_head_dim - rotary_dim;

    let normed = rmsnorm(program, x, attn_norm_weight, inv_dim, eps)?;

    let q_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->shdi"), (wq, "ihd->shdi")],
    )?;
    let q_raw = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        q_product,
        "shdi->shdi",
        "shd->shdi",
    )?;

    let gate_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->shdi"), (w_gate_q, "ihd->shdi")],
    )?;
    let gate_raw = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        gate_product,
        "shdi->shdi",
        "shd->shdi",
    )?;

    let k_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->sudi"), (wk, "iud->sudi")],
    )?;
    let k_raw = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        k_product,
        "sudi->sudi",
        "sud->sudi",
    )?;

    let v_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->sudi"), (wv, "iud->sudi")],
    )?;
    let v_new = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        v_product,
        "sudi->sudi",
        "sud->sudi",
    )?;

    // `q_norm`/`k_norm` run on the FULL `attn_head_dim` width, before RoPE
    // ever splits it (`modeling_qwen3_next.py:300-301`,
    // `self.q_norm(query_states.view(hidden_shape))` where `hidden_shape`'s
    // last dim is `self.head_dim` = `attn_head_dim`; `qwen35.cpp:308-317`
    // normalizes `Qcur`/`Kcur` before `ggml_rope_multi` runs).
    let q = rmsnorm_per_head(program, q_raw, q_norm_weight, inv_attn_head_dim, eps, "h")?;
    let k = rmsnorm_per_head(program, k_raw, k_norm_weight, inv_attn_head_dim, eps, "u")?;

    let q_pass = per_head_channel_range(program, q, "h", attn_head_dim, rotary_dim, pass_dim)?;
    let k_pass = per_head_channel_range(program, k, "u", attn_head_dim, rotary_dim, pass_dim)?;

    // split-half RoPE (`ggml_compute_forward_rope_flt`'s
    // `GGML_ROPE_TYPE_IMROPE` arm, `rotate_pairs(n_dims, n_dims/2, ...)`):
    // `out[i] = x[i]*cos[i] - x[i+pairs]*sin[i]`,
    // `out[i+pairs] = x[i+pairs]*cos[i] + x[i]*sin[i]`. Read directly off
    // `q`/`k`'s own `attn_head_dim`-wide axis (not a pre-sliced
    // `q_first`/`q_second`) -- see [`fused_rope_pair`].
    let (rotated_q_first, rotated_q_second) =
        fused_rope_pair(program, q, 'h', cos_new, sin_new, RopePairing::SplitHalf { pairs })?;
    let (rotated_k_new_first, rotated_k_new_second) =
        fused_rope_pair(program, k, 'u', cos_new, sin_new, RopePairing::SplitHalf { pairs })?;

    let group_map_i = alloc::format!("s,{group}*u+g,i->sugi");
    let q_first_grouped = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(rotated_q_first, group_map_i.as_str()), (group_ones, "ug->sugi")],
    )?;
    let q_second_grouped = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(rotated_q_second, group_map_i.as_str()), (group_ones, "ug->sugi")],
    )?;
    let group_map_p = alloc::format!("s,{group}*u+g,p->sugp");
    let q_pass_grouped = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(q_pass, group_map_p.as_str()), (group_ones, "ug->sugp")],
    )?;

    let score_cached_first_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q_first_grouped, "sugi->stugi"), (k_first_cache, "tui->stugi")])?;
    let score_cached_first = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_cached_first_product,
        "stugi->stugi",
        "stug->stugi",
    )?;
    let score_cached_second_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q_second_grouped, "sugi->stugi"), (k_second_cache, "tui->stugi")])?;
    let score_cached_second = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_cached_second_product,
        "stugi->stugi",
        "stug->stugi",
    )?;
    let score_cached_pass_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q_pass_grouped, "sugp->stugp"), (k_pass_cache, "tup->stugp")])?;
    let score_cached_pass = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_cached_pass_product,
        "stugp->stugp",
        "stug->stugp",
    )?;
    let score_cached_rotated = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(score_cached_first, "stug->stug"), (score_cached_second, "stug->stug")],
    )?;
    let score_cached = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[
            (score_cached_rotated, "stug->stug"),
            (score_cached_pass, "stug->stug"),
        ],
    )?;
    let score_cached_scaled = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(score_cached, "stug->stug"), (inv_sqrt_attn_head_dim, "->stug")],
    )?;
    // `k_first_cache`/`k_second_cache`/`k_pass_cache`/`v_cache` are bound to
    // the CALLER's own bucketed `kv_extent`, not the real `cached_len`
    // (`Qwen35DenseAttentionPadScratch::fill`'s own doc) -- rows
    // `[cached_len, bound_extent)` are zero-padding, not history. Unlike the
    // `Attention` arm, which excludes that padding via the fused
    // `BoundOpKind::CachedAttention` op's own `cached_key_rows` runtime
    // bound, this graph has no such fusion, so the padding is masked here
    // exactly the way [`causal_mask_merged`] masks its own merged range:
    // `key_index >= cached_len` is invalid, scored `-inf` before either
    // softmax pass sees it.
    let neg_infinity = scalar_constant(program, f32::NEG_INFINITY);
    let cached_key_index = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Symbolic(1),
        },
    );
    let cached_len_exclusive_bound = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(cached_len, "->"), (ones, "->")],
    )?;
    let is_cached_padding = elementwise(
        program,
        DType::Float32,
        ScalarOp::Greater,
        &[(cached_key_index, "t->t"), (cached_len_exclusive_bound, "->t")],
    )?;
    let score_cached_scaled = elementwise(
        program,
        DType::Float32,
        ScalarOp::Select,
        &[
            (is_cached_padding, "t->stug"),
            (neg_infinity, "->stug"),
            (score_cached_scaled, "stug->stug"),
        ],
    )?;

    let score_new_first_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q_first_grouped, "sugi->swugi"), (rotated_k_new_first, "wui->swugi")])?;
    let score_new_first = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_new_first_product,
        "swugi->swugi",
        "swug->swugi",
    )?;
    let score_new_second_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q_second_grouped, "sugi->swugi"), (rotated_k_new_second, "wui->swugi")])?;
    let score_new_second = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_new_second_product,
        "swugi->swugi",
        "swug->swugi",
    )?;
    let score_new_pass_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q_pass_grouped, "sugp->swugp"), (k_pass, "wup->swugp")])?;
    let score_new_pass = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_new_pass_product,
        "swugp->swugp",
        "swug->swugp",
    )?;
    let score_new_rotated = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(score_new_first, "swug->swug"), (score_new_second, "swug->swug")],
    )?;
    let score_new = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[
            (score_new_rotated, "swug->swug"),
            (score_new_pass, "swug->swug"),
        ],
    )?;
    let score_new_scaled = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(score_new, "swug->swug"), (inv_sqrt_attn_head_dim, "->swug")],
    )?;
    let neg_infinity = scalar_constant(program, f32::NEG_INFINITY);
    let score_new_masked = elementwise(
        program,
        DType::Float32,
        ScalarOp::Select,
        &[
            (is_future, "sw->swug"),
            (neg_infinity, "->swug"),
            (score_new_scaled, "swug->swug"),
        ],
    )?;

    let score_max_cached = reduce(
        program,
        DType::Float32,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        score_cached_scaled,
        "stug->stug",
        "sug->stug",
    )?;
    let score_max_new = reduce(
        program,
        DType::Float32,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        score_new_masked,
        "swug->swug",
        "sug->swug",
    )?;
    let global_max = elementwise(
        program,
        DType::Float32,
        ScalarOp::Maximum,
        &[(score_max_cached, "sug->sug"), (score_max_new, "sug->sug")],
    )?;

    let shifted_cached = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(score_cached_scaled, "stug->stug"), (global_max, "sug->stug")],
    )?;
    let weights_cached = elementwise(
        program,
        DType::Float32,
        ScalarOp::Exponential,
        &[(shifted_cached, "stug->stug")],
    )?;
    let shifted_new = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(score_new_masked, "swug->swug"), (global_max, "sug->swug")],
    )?;
    let weights_new = elementwise(
        program,
        DType::Float32,
        ScalarOp::Exponential,
        &[(shifted_new, "swug->swug")],
    )?;

    let sum_cached = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        weights_cached,
        "stug->stug",
        "sug->stug",
    )?;
    let sum_new = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        weights_new,
        "swug->swug",
        "sug->swug",
    )?;
    let weight_sum = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(sum_cached, "sug->sug"), (sum_new, "sug->sug")],
    )?;
    let inv_weight_sum = elementwise(
        program,
        DType::Float32,
        ScalarOp::Reciprocal,
        &[(weight_sum, "sug->sug")],
    )?;

    let attended_cached_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(weights_cached, "stug->stugd"), (v_cache, "tud->stugd")])?;
    let attended_cached = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        attended_cached_product,
        "stugd->stugd",
        "sugd->stugd",
    )?;
    let attended_new_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(weights_new, "swug->swugd"), (v_new, "wud->swugd")])?;
    let attended_new = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        attended_new_product,
        "swugd->swugd",
        "sugd->swugd",
    )?;
    let attended_sum = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(attended_cached, "sugd->sugd"), (attended_new, "sugd->sugd")],
    )?;
    let attended = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(attended_sum, "sugd->sugd"), (inv_weight_sum, "sug->sugd")],
    )?;

    // per-head sigmoid gate, applied to the attention output before `o_proj`
    // (`modeling_qwen3_next.py:325-328`, `qwen35.cpp:322-328`).
    let group_map_d = alloc::format!("s,{group}*u+g,d->sugd");
    let gate_grouped = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(gate_raw, group_map_d.as_str()), (group_ones, "ug->sugd")],
    )?;
    let neg_attn_gate = elementwise(program, DType::Float32, ScalarOp::Negate, &[(gate_grouped, "sugd->sugd")])?;
    let exp_neg_attn_gate = elementwise(program, DType::Float32, ScalarOp::Exponential, &[(neg_attn_gate, "sugd->sugd")])?;
    let one_plus_exp_attn_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(exp_neg_attn_gate, "sugd->sugd"), (ones, "->sugd")],
    )?;
    let sigmoid_attn_gate = elementwise(program, DType::Float32, ScalarOp::Reciprocal, &[(one_plus_exp_attn_gate, "sugd->sugd")])?;
    let gated_attended = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(attended, "sugd->sugd"), (sigmoid_attn_gate, "sugd->sugd")],
    )?;

    let wo_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(gated_attended, "sugd->sugdo"), (wo, "ugdo->sugdo")],
    )?;
    let attn_out = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        wo_product,
        "sugdo->sugdo",
        "so->sugdo",
    )?;

    let residual1 = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(attn_out, "sd->sd"), (x, "sd->sd")],
    )?;

    Ok((residual1, (rotated_k_new_first, rotated_k_new_second, k_pass, v_new)))
}

/// [`append_qwen35_dense_attention_only`] plus the dense (non-MoE) SwiGLU
/// FFN tail `qwen35_forward_program`'s own non-routed checkpoints carry on
/// every layer -- see that function for the worked example of wiring this
/// builder's cache inputs and outputs. A caller whose FFN is routed
/// (`qwen35moe`-shaped) calls [`append_qwen35_dense_attention_only`]
/// directly instead of this wrapper.
#[allow(clippy::too_many_arguments)]
pub fn append_qwen35_dense_attention_layer(
    program: &mut Vec<Op>,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    ones: NodeId,
    inv_sqrt_attn_head_dim: NodeId,
    inv_attn_head_dim: NodeId,
    cos_new: NodeId,
    sin_new: NodeId,
    group_ones: NodeId,
    is_future: NodeId,
    cached_len: NodeId,
    group: u32,
    rotary_dim: u32,
    attn_head_dim: u32,
    attn_norm_weight: NodeId,
    ffn_norm_weight: NodeId,
    q_norm_weight: NodeId,
    k_norm_weight: NodeId,
    wq: NodeId,
    w_gate_q: NodeId,
    wk: NodeId,
    wv: NodeId,
    wo: NodeId,
    w_gate: NodeId,
    w_up: NodeId,
    w_down: NodeId,
    k_first_cache: NodeId,
    k_second_cache: NodeId,
    k_pass_cache: NodeId,
    v_cache: NodeId,
) -> Result<(NodeId, Qwen35DenseAttentionRoots), TensorError> {
    let (residual1, roots) = append_qwen35_dense_attention_only(
        program,
        x,
        inv_dim,
        eps,
        ones,
        inv_sqrt_attn_head_dim,
        inv_attn_head_dim,
        cos_new,
        sin_new,
        group_ones,
        is_future,
        cached_len,
        group,
        rotary_dim,
        attn_head_dim,
        attn_norm_weight,
        q_norm_weight,
        k_norm_weight,
        wq,
        w_gate_q,
        wk,
        wv,
        wo,
        k_first_cache,
        k_second_cache,
        k_pass_cache,
        v_cache,
    )?;

    let normed2 = rmsnorm(program, residual1, ffn_norm_weight, inv_dim, eps)?;

    let gate_product2 = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed2, "sd->sdg"), (w_gate, "dg->sdg")],
    )?;
    let ffn_gate = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        gate_product2,
        "sdg->sdg",
        "sg->sdg",
    )?;
    let up_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed2, "sd->sdg"), (w_up, "dg->sdg")],
    )?;
    let up = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        up_product,
        "sdg->sdg",
        "sg->sdg",
    )?;

    let neg_ffn_gate = elementwise(program, DType::Float32, ScalarOp::Negate, &[(ffn_gate, "sg->sg")])?;
    let exp_neg_ffn_gate = elementwise(program, DType::Float32, ScalarOp::Exponential, &[(neg_ffn_gate, "sg->sg")])?;
    let one_plus_exp_ffn_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(exp_neg_ffn_gate, "sg->sg"), (ones, "->sg")],
    )?;
    let sigmoid_ffn_gate = elementwise(program, DType::Float32, ScalarOp::Reciprocal, &[(one_plus_exp_ffn_gate, "sg->sg")])?;
    let silu_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(ffn_gate, "sg->sg"), (sigmoid_ffn_gate, "sg->sg")],
    )?;
    let ffn_hidden = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(silu_gate, "sg->sg"), (up, "sg->sg")],
    )?;

    let down_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(ffn_hidden, "sg->sgd"), (w_down, "gd->sgd")],
    )?;
    let ffn_out = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        down_product,
        "sgd->sgd",
        "sd->sgd",
    )?;

    let x_next = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(ffn_out, "sd->sd"), (residual1, "sd->sd")],
    )?;

    Ok((x_next, roots))
}

/// [`append_mistral_cached_layer`]'s single-range counterpart: the SAME
/// function -- same RoPE, same GQA grouping, same `CachedLayerRoots`
/// return -- but scored through ONE softmax over ONE key axis instead of
/// two disjoint ranges combined by hand. `k_even_cache`/`k_odd_cache`/
/// `v_cache` are no longer "everything before this call"; they are the
/// WHOLE merged context this call attends to, prior positions AND this
/// call's own freshly rotated keys already folded in by the caller between
/// calls (`kv_cache.{layer}.*`'s own shape grows from `cached_len` to
/// `cached_len + new_count`, same [`Extent::Symbolic`] slot, no new op).
/// `rotated_k_new_even`/`rotated_k_new_odd`/`v_new` are STILL computed
/// in-graph from `x`, unchanged from [`append_mistral_cached_layer`] --
/// this call's own [`CachedLayerRoots`] the caller folds into next call's
/// merged cache -- they are simply no longer read for THIS call's own
/// score, since this call's own keys are not yet part of the merged range
/// it attends over (a query never attends a key that does not exist until
/// after it is computed).
///
/// Score/softmax/attended here are node-for-node
/// [`append_mistral_layer`]'s own single-range pattern (`score`,
/// `score_max`, `shifted`, `weights`, `weight_sum`, `inv_weight_sum`,
/// `probabilities`, `attended_product`, `attended`) rather than
/// [`append_mistral_cached_layer`]'s two-block online-softmax combine --
/// the entire point of this function existing next to that one.
/// `is_future` here must come from [`causal_mask_merged`], not
/// [`causal_mask`]: shape `[s, t]` with `t` sized by [`Extent::Symbolic`]
/// slot 1 (the merged range), not slot 0.
///
/// `gate_before_up` decides only which of the FFN's two independent matvecs
/// (`ffn_gate.weight`, `ffn_up.weight` -- both read `normed2`, neither reads
/// the other's output) is PUSHED into `program` first; `gate`/`up` are
/// returned bound the same way either way, so every downstream op
/// (`silu_gate`, `ffn_hidden`) is byte-identical regardless of this flag.
/// This exists to measure whether ROW 310/311's `ffn_gate`/`ffn_up`
/// bandwidth asymmetry is positional (first-vs-second in a barrier-free
/// sibling pair, see `omega/src/metal.rs`'s `HazardTracker`) rather than
/// per-kernel -- see this module's own
/// `swapping_gate_and_up_order_keeps_dataflow_identical` test.
///
/// `qk_norm` (ROW 373) is the SAME `Option<(NodeId, NodeId, NodeId)>` shape
/// as [`append_mistral_cached_layer`]'s own parameter of that name --
/// q-norm weight, k-norm weight, `inv_head_dim` -- applied through the same
/// [`rmsnorm_per_head`] calls before RoPE, and selects the same
/// interleaved-vs-split-half pairing that function's doc already derives
/// from `qk_norm.is_some()`. This builder no longer rejects a qk-norm
/// checkpoint; it now builds it, node-for-node the same attention block the
/// two-range sibling would.
#[allow(clippy::too_many_arguments)]
pub fn append_mistral_single_range_cached_layer(
    program: &mut Vec<Op>,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    ones: NodeId,
    inv_sqrt_head_dim: NodeId,
    cos_new: NodeId,
    sin_new: NodeId,
    group_ones: NodeId,
    is_future: NodeId,
    group: u32,
    head_dim: u32,
    attn_norm_weight: NodeId,
    ffn_norm_weight: NodeId,
    wq: NodeId,
    wk: NodeId,
    wv: NodeId,
    wo: NodeId,
    w_gate: NodeId,
    w_up: NodeId,
    w_down: NodeId,
    k_even_cache: NodeId,
    k_odd_cache: NodeId,
    v_cache: NodeId,
    qk_norm: Option<(NodeId, NodeId, NodeId)>,
    gate_before_up: bool,
) -> Result<(NodeId, CachedLayerRoots), TensorError> {
    // Same architecture inputs as `append_mistral_cached_layer`'s own
    // `qk_norm: Option<(NodeId, NodeId, NodeId)>` (q-norm weight, k-norm
    // weight, `inv_head_dim`) -- ROW 373's typed rejection here was a class
    // defect, not a correct omission: this builder's own attention block is
    // otherwise node-for-node the two-range sibling's, so it can carry the
    // same per-head RMSNorm and the same pairing selection
    // (`qk_norm.is_some()`) that sibling already uses, see that function's
    // own `qk_norm` doc.
    let normed = rmsnorm(program, x, attn_norm_weight, inv_dim, eps)?;

    let q_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->shdi"), (wq, "ihd->shdi")],
    )?;
    let q_raw = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        q_product,
        "shdi->shdi",
        "shd->shdi",
    )?;

    let k_new_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->sudi"), (wk, "iud->sudi")],
    )?;
    let k_new_raw = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        k_new_product,
        "sudi->sudi",
        "sud->sudi",
    )?;

    let v_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->sudi"), (wv, "iud->sudi")],
    )?;
    let v_new = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        v_product,
        "sudi->sudi",
        "sud->sudi",
    )?;

    let (q, k_new) = match qk_norm {
        Some((q_norm_weight, k_norm_weight, inv_head_dim)) => {
            let q = rmsnorm_per_head(program, q_raw, q_norm_weight, inv_head_dim, eps, "h")?;
            let k_new = rmsnorm_per_head(program, k_new_raw, k_norm_weight, inv_head_dim, eps, "u")?;
            (q, k_new)
        }
        None => (q_raw, k_new_raw),
    };

    // Pairing selection mirrors `append_mistral_cached_layer`'s own
    // `qk_norm.is_some()` rule (that function's doc walks the NEOX-vs-
    // interleaved reasoning): a checkpoint carrying `attn_q_norm.weight` is
    // the split-half family, everything else stays interleaved.
    let (rotated_q_even, rotated_q_odd, rotated_k_new_even, rotated_k_new_odd) = match qk_norm {
        Some(_) => {
            let pairs = head_dim / 2;
            let (rotated_q_first, rotated_q_second) =
                fused_rope_pair(program, q, 'h', cos_new, sin_new, RopePairing::SplitHalf { pairs })?;
            let (rotated_k_first, rotated_k_second) =
                fused_rope_pair(program, k_new, 'u', cos_new, sin_new, RopePairing::SplitHalf { pairs })?;
            (rotated_q_first, rotated_q_second, rotated_k_first, rotated_k_second)
        }
        None => {
            let (rotated_q_even, rotated_q_odd) =
                fused_rope_pair(program, q, 'h', cos_new, sin_new, RopePairing::Interleaved)?;
            let (rotated_k_new_even, rotated_k_new_odd) =
                fused_rope_pair(program, k_new, 'u', cos_new, sin_new, RopePairing::Interleaved)?;
            (rotated_q_even, rotated_q_odd, rotated_k_new_even, rotated_k_new_odd)
        }
    };

    let group_map = alloc::format!("s,{group}*u+g,i->sugi");
    let q_even_grouped = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (rotated_q_even, group_map.as_str()),
            (group_ones, "ug->sugi"),
        ],
    )?;
    let q_odd_grouped = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (rotated_q_odd, group_map.as_str()),
            (group_ones, "ug->sugi"),
        ],
    )?;

    // single range: query `s` against the WHOLE merged key range `t`
    // (symbol 1's extent), `k_even_cache`/`k_odd_cache` already carrying
    // every position this query may attend -- no second source, no combine.
    let score_even_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (q_even_grouped, "sugi->stugi"),
            (k_even_cache, "tui->stugi"),
        ],
    )?;
    let score_even = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_even_product,
        "stugi->stugi",
        "stug->stugi",
    )?;
    let score_odd_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(q_odd_grouped, "sugi->stugi"), (k_odd_cache, "tui->stugi")],
    )?;
    let score_odd = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_odd_product,
        "stugi->stugi",
        "stug->stugi",
    )?;
    let scores = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(score_even, "stug->stug"), (score_odd, "stug->stug")],
    )?;
    let scores_scaled = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(scores, "stug->stug"), (inv_sqrt_head_dim, "->stug")],
    )?;
    let neg_infinity = scalar_constant(program, f32::NEG_INFINITY);
    let scores_masked = elementwise(
        program,
        DType::Float32,
        ScalarOp::Select,
        &[
            (is_future, "st->stug"),
            (neg_infinity, "->stug"),
            (scores_scaled, "stug->stug"),
        ],
    )?;

    let score_max = reduce(
        program,
        DType::Float32,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        scores_masked,
        "stug->stug",
        "sug->stug",
    )?;
    let shifted = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(scores_masked, "stug->stug"), (score_max, "sug->stug")],
    )?;
    let weights = elementwise(
        program,
        DType::Float32,
        ScalarOp::Exponential,
        &[(shifted, "stug->stug")],
    )?;
    let weight_sum = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        weights,
        "stug->stug",
        "sug->stug",
    )?;
    let inv_weight_sum = elementwise(
        program,
        DType::Float32,
        ScalarOp::Reciprocal,
        &[(weight_sum, "sug->sug")],
    )?;
    let probabilities = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(weights, "stug->stug"), (inv_weight_sum, "sug->stug")],
    )?;

    let attended_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(probabilities, "stug->stugd"), (v_cache, "tud->stugd")],
    )?;
    let attended = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        attended_product,
        "stugd->stugd",
        "sugd->stugd",
    )?;

    let wo_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(attended, "sugd->sugdo"), (wo, "ugdo->sugdo")],
    )?;
    let attn_out = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        wo_product,
        "sugdo->sugdo",
        "so->sugdo",
    )?;

    let residual1 = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(attn_out, "sd->sd"), (x, "sd->sd")],
    )?;

    let normed2 = rmsnorm(program, residual1, ffn_norm_weight, inv_dim, eps)?;

    let append_gate = |program: &mut Vec<Op>| -> Result<NodeId, TensorError> {
        let gate_product = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(normed2, "sd->sdg"), (w_gate, "dg->sdg")],
        )?;
        reduce(
            program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            gate_product,
            "sdg->sdg",
            "sg->sdg",
        )
    };
    let append_up = |program: &mut Vec<Op>| -> Result<NodeId, TensorError> {
        let up_product = elementwise(
            program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(normed2, "sd->sdg"), (w_up, "dg->sdg")],
        )?;
        reduce(
            program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            up_product,
            "sdg->sdg",
            "sg->sdg",
        )
    };
    // `gate_before_up` only decides encode ORDER of these two independent
    // matvecs (both read `normed2`, neither reads the other's output) --
    // `gate`/`up` bind identically either way, so every op below is
    // unaffected by which branch ran.
    let (gate, up) = if gate_before_up {
        let gate = append_gate(program)?;
        let up = append_up(program)?;
        (gate, up)
    } else {
        let up = append_up(program)?;
        let gate = append_gate(program)?;
        (gate, up)
    };

    let neg_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Negate,
        &[(gate, "sg->sg")],
    )?;
    let exp_neg_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Exponential,
        &[(neg_gate, "sg->sg")],
    )?;
    let one_plus_exp = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(exp_neg_gate, "sg->sg"), (ones, "->sg")],
    )?;
    let sigmoid_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Reciprocal,
        &[(one_plus_exp, "sg->sg")],
    )?;
    let silu_gate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(gate, "sg->sg"), (sigmoid_gate, "sg->sg")],
    )?;
    let ffn_hidden = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(silu_gate, "sg->sg"), (up, "sg->sg")],
    )?;

    let down_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(ffn_hidden, "sg->sgd"), (w_down, "gd->sgd")],
    )?;
    let ffn_out = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        down_product,
        "sgd->sgd",
        "sd->sgd",
    )?;

    let x_next = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(ffn_out, "sd->sd"), (residual1, "sd->sd")],
    )?;

    Ok((x_next, (rotated_k_new_even, rotated_k_new_odd, v_new)))
}


/// [`mistral_single_range_cached_forward_program`] is
/// [`mistral_cached_forward_program`]'s single-range counterpart: same
/// per-layer weight inputs, same [`CachedLayerRoots`] contract, one
/// difference -- `causal_mask_merged` in place of `causal_mask`
/// (needs a `cached_len` scalar the plain cache mask does not), and
/// `append_mistral_single_range_cached_layer` in place of
/// `append_mistral_cached_layer` for every layer. Dense-only (no MoE
/// branch): the mixture-of-experts FFN this function's counterpart also
/// supports is orthogonal to the attention-merge this function exists to
/// prove, and duplicating that branch here would test nothing new.
///
/// `qk_norm` (ROW 373) selects the same per-head QK-norm + split-half RoPE
/// pairing [`qwen3_cached_forward_program`] carries on the two-range path --
/// `true` declares `blk.{layer}.attn_q_norm.weight`/`attn_k_norm.weight`
/// inputs per layer and threads them through
/// [`append_mistral_single_range_cached_layer`]'s own
/// `Option<(NodeId, NodeId, NodeId)>` parameter, mirroring
/// [`mistral_cached_forward_program_with_experts`]'s own `inv_head_dim`/
/// `qk_norm_weights` construction below. `false` reproduces today's
/// interleaved, no-norm program node-for-node.
// ROW 326/328 diagnostic: `duplicate_head` mirrors `gate_before_up`'s own
// mechanism (a plain, always-compiled parameter a caller sets, production
// call sites pass a fixed literal) rather than a `#[cfg(test)]` item,
// because `#[cfg(test)]` on a `proxima-tensor` item is invisible
// cross-crate to `omega`/`proxima-model-interop`, which is where the
// Metal measurement this flag exists for actually runs.
// [`DuplicateHeadPosition::Before`]/`After` each append a second,
// identical `output.weight` reduce reusing this call's own `lm_head`
// (against `x`, the raw embedding, for `Before`; against `normed_final`
// for `After`), returned as the 4th tuple element so a caller can add it
// to a `Plan`'s requested outputs -- otherwise graph pruning drops it as
// unreachable dead code, same as any other unread node.
// [`DuplicateHeadPosition::None`] (every production call site) is
// byte-identical to this function's behavior before the flag existed.
#[allow(
    clippy::too_many_arguments,
    reason = "one architecture hyperparameter per positional arg, matching every other \
              forward-program builder in this file (see the other `too_many_arguments` \
              call sites above); `duplicate_head` is the 8th and last"
)]
pub fn mistral_single_range_cached_forward_program(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    block_count: u32,
    qk_norm: bool,
    duplicate_head: DuplicateHeadPosition,
) -> Result<SingleRangeForwardProgram, TensorError> {
    let group = query_heads / kv_heads;
    let pairs = head_dim / 2;

    let mut program = Vec::new();

    let ids = input_leaf(
        &mut program,
        DType::Int32,
        alloc::vec![Extent::Symbolic(0)],
        "ids",
    );
    let table = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(vocab), Extent::Static(embedding)],
        "token_embd.weight",
    );
    let mut x = embedding_lookup(&mut program, table, ids);

    // ROW 328: a SEPARATE, early `output.weight` `Op::Input` -- not a
    // second reference to the one declared before the real head below.
    // `resolve_named_blocks` (`proxima-tensor/src/cpu.rs`) resolves every
    // `Op::Input` node by NAME independently, so two nodes sharing the name
    // `output.weight` both bind to the same weight bytes with no special
    // casing; declaring a second one here (instead of hoisting the single
    // existing declaration) keeps the `None`/`After` program's own node
    // sequence byte-for-byte identical to before this row -- hoisting the
    // one declaration shifted every later `NodeId` by one and changed
    // `cached_attention_single_range_candidates`' fused bound-op count for
    // EVERY position, not just `Before` (`619` -> `523`, a regression this
    // row's own gate caught before it landed).
    let duplicate_head_scratch_before = if duplicate_head == DuplicateHeadPosition::Before {
        let lm_head_before = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding), Extent::Static(vocab)],
            "output.weight",
        );
        Some(duplicate_head_reduce(&mut program, x, lm_head_before)?)
    } else {
        None
    };

    let inv_dim = scalar_constant(&mut program, 1.0 / embedding as f32);
    let eps = symbolic_leaf(&mut program, DType::Float32, "eps");
    let ones = scalar_constant(&mut program, 1.0);
    let inv_sqrt_head_dim = scalar_constant(&mut program, 1.0 / (head_dim as f32).sqrt());
    // only materialized when a layer actually consumes it (`qk_norm`), same
    // guard `mistral_cached_forward_program_with_experts` uses so a dense
    // checkpoint's own node count is unaffected by this feature existing.
    let inv_head_dim = qk_norm.then(|| scalar_constant(&mut program, 1.0 / head_dim as f32));
    let cos_new = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(pairs)],
        "rope_cos",
    );
    let sin_new = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(pairs)],
        "rope_sin",
    );
    let group_ones = op::append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
            value: 1.0,
        },
    );
    let cached_len = input_leaf(&mut program, DType::Float32, Vec::new(), "cached_len");
    let is_future = causal_mask_merged(&mut program, cached_len)?;

    let mut cache_roots: Vec<CachedLayerRoots> = Vec::with_capacity(block_count as usize);

    for layer in 0..block_count {
        let attn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.attn_norm.weight"),
        );
        let ffn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.ffn_norm.weight"),
        );
        let wq = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(embedding),
                Extent::Static(query_heads),
                Extent::Static(head_dim)
            ],
            &alloc::format!("blk.{layer}.attn_q.weight"),
        );
        let wk = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(embedding),
                Extent::Static(kv_heads),
                Extent::Static(head_dim)
            ],
            &alloc::format!("blk.{layer}.attn_k.weight"),
        );
        let wv = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(embedding),
                Extent::Static(kv_heads),
                Extent::Static(head_dim)
            ],
            &alloc::format!("blk.{layer}.attn_v.weight"),
        );
        let wo = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(kv_heads),
                Extent::Static(group),
                Extent::Static(head_dim),
                Extent::Static(embedding),
            ],
            &alloc::format!("blk.{layer}.attn_output.weight"),
        );
        let k_even_cache = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Symbolic(1),
                Extent::Static(kv_heads),
                Extent::Static(pairs)
            ],
            &alloc::format!("kv_cache.{layer}.k_even"),
        );
        let k_odd_cache = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Symbolic(1),
                Extent::Static(kv_heads),
                Extent::Static(pairs)
            ],
            &alloc::format!("kv_cache.{layer}.k_odd"),
        );
        let v_cache = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Symbolic(1),
                Extent::Static(kv_heads),
                Extent::Static(head_dim)
            ],
            &alloc::format!("kv_cache.{layer}.v"),
        );
        let w_gate = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding), Extent::Static(feed_forward)],
            &alloc::format!("blk.{layer}.ffn_gate.weight"),
        );
        let w_up = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding), Extent::Static(feed_forward)],
            &alloc::format!("blk.{layer}.ffn_up.weight"),
        );
        let w_down = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(feed_forward), Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.ffn_down.weight"),
        );
        let qk_norm_weights = inv_head_dim.map(|inv_head_dim| {
            let q_norm_weight = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(head_dim)],
                &alloc::format!("blk.{layer}.attn_q_norm.weight"),
            );
            let k_norm_weight = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(head_dim)],
                &alloc::format!("blk.{layer}.attn_k_norm.weight"),
            );
            (q_norm_weight, k_norm_weight, inv_head_dim)
        });

        let (x_next, layer_roots) = append_mistral_single_range_cached_layer(
            &mut program,
            x,
            inv_dim,
            eps,
            ones,
            inv_sqrt_head_dim,
            cos_new,
            sin_new,
            group_ones,
            is_future,
            group,
            head_dim,
            attn_norm_weight,
            ffn_norm_weight,
            wq,
            wk,
            wv,
            wo,
            w_gate,
            w_up,
            w_down,
            k_even_cache,
            k_odd_cache,
            v_cache,
            qk_norm_weights,
            true,
        )?;
        x = x_next;
        cache_roots.push(layer_roots);
    }

    let output_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding)],
        "output_norm.weight",
    );
    let normed_final = rmsnorm(&mut program, x, output_norm_weight, inv_dim, eps)?;

    let lm_head = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding), Extent::Static(vocab)],
        "output.weight",
    );
    let logits = duplicate_head_reduce(&mut program, normed_final, lm_head)?;

    let duplicate_head_scratch = match duplicate_head {
        DuplicateHeadPosition::None => None,
        DuplicateHeadPosition::Before => duplicate_head_scratch_before,
        DuplicateHeadPosition::After => {
            Some(duplicate_head_reduce(&mut program, normed_final, lm_head)?)
        }
    };

    Ok((program, logits, cache_roots, duplicate_head_scratch))
}

/// `sum_d(activation[s, d] * lm_head[d, v])` -- the vocab-projection
/// multiply-reduce pair both the real head and every
/// [`DuplicateHeadPosition`] scratch reduce share, factored out so ROW 328's
/// `Before`/`After` positions differ only in which activation node (`x`
/// pre-layer-0 vs `normed_final` post-layer-31) they read, never in the
/// reduce shape itself.
pub fn duplicate_head_reduce(
    program: &mut Vec<Op>,
    activation: NodeId,
    lm_head: NodeId,
) -> Result<NodeId, TensorError> {
    let product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(activation, "sd->sdv"), (lm_head, "dv->sdv")],
    )?;
    reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        product,
        "sdv->sdv",
        "sv->sdv",
    )
}

/// [`append_mistral_cached_layer`]'s mixture-of-experts counterpart, the
/// same relationship [`append_mistral_moe_layer`] bears to
/// [`append_mistral_layer`]: cached attention block (RoPE + GQA +
/// online-softmax combine over the cached/new key split, the same shape as
/// [`append_mistral_cached_layer`]'s own, including that function's
/// `qk_norm`-gated per-head Q/K norm and RoPE-pairing switch -- Qwen3-MoE's
/// own checkpoint carries `attn_q_norm.weight`/`attn_k_norm.weight` on every
/// layer, every one of them MoE, so this arm needs the identical switch or
/// every MoE layer silently skips QK-norm and rotates Q/K with the wrong
/// (interleaved, not NEOX split-half) pairing), [`append_moe_ffn`] in place
/// of the dense SwiGLU triple. Kept as a separate function for the same
/// reason [`append_mistral_moe_layer`] is: the dense cached path's own node
/// sequence never changes shape merely because this function exists next to
/// it.
#[allow(clippy::too_many_arguments)]
pub fn append_mistral_cached_moe_layer(
    program: &mut Vec<Op>,
    layer: u32,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    ones: NodeId,
    inv_sqrt_head_dim: NodeId,
    cos_new: NodeId,
    sin_new: NodeId,
    group_ones: NodeId,
    is_future: NodeId,
    group: u32,
    head_dim: u32,
    attn_norm_weight: NodeId,
    ffn_norm_weight: NodeId,
    wq: NodeId,
    wk: NodeId,
    wv: NodeId,
    wo: NodeId,
    gate_inp: NodeId,
    expert_w_gate: NodeId,
    expert_w_up: NodeId,
    expert_w_down: NodeId,
    expert_count: u32,
    expert_used_count: u32,
    k_even_cache: NodeId,
    k_odd_cache: NodeId,
    v_cache: NodeId,
    qk_norm: Option<(NodeId, NodeId, NodeId)>,
) -> Result<(NodeId, CachedLayerRoots, MoeSite), TensorError> {
    let normed = rmsnorm(program, x, attn_norm_weight, inv_dim, eps)?;

    let q_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->shdi"), (wq, "ihd->shdi")],
    )?;
    let q_raw = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        q_product,
        "shdi->shdi",
        "shd->shdi",
    )?;

    let k_new_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->sudi"), (wk, "iud->sudi")],
    )?;
    let k_new_raw = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        k_new_product,
        "sudi->sudi",
        "sud->sudi",
    )?;

    let (q, k_new) = match qk_norm {
        Some((q_norm_weight, k_norm_weight, inv_head_dim)) => {
            let q = rmsnorm_per_head(program, q_raw, q_norm_weight, inv_head_dim, eps, "h")?;
            let k_new = rmsnorm_per_head(program, k_new_raw, k_norm_weight, inv_head_dim, eps, "u")?;
            (q, k_new)
        }
        None => (q_raw, k_new_raw),
    };

    let v_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->sudi"), (wv, "iud->sudi")],
    )?;
    let v_new = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        v_product,
        "sudi->sudi",
        "sud->sudi",
    )?;

    // Same `qk_norm.is_some()` switch as [`append_mistral_cached_layer`]
    // (see that function's own doc): a checkpoint carrying `attn_q_norm.weight`
    // is NEOX-family (Qwen3), whose on-disk Q/K rows stay in HF's native
    // split-half layout, never llama.cpp's converter-permuted interleaved
    // pairing a no-qk_norm (Mistral/LLaMA) checkpoint uses.
    let (rotated_q_even, rotated_q_odd, rotated_k_new_even, rotated_k_new_odd) = match qk_norm {
        Some(_) => {
            let pairs = head_dim / 2;
            let (rotated_q_first, rotated_q_second) =
                fused_rope_pair(program, q, 'h', cos_new, sin_new, RopePairing::SplitHalf { pairs })?;
            let (rotated_k_first, rotated_k_second) =
                fused_rope_pair(program, k_new, 'u', cos_new, sin_new, RopePairing::SplitHalf { pairs })?;
            (rotated_q_first, rotated_q_second, rotated_k_first, rotated_k_second)
        }
        None => {
            let (rotated_q_even, rotated_q_odd) =
                fused_rope_pair(program, q, 'h', cos_new, sin_new, RopePairing::Interleaved)?;
            let (rotated_k_new_even, rotated_k_new_odd) =
                fused_rope_pair(program, k_new, 'u', cos_new, sin_new, RopePairing::Interleaved)?;
            (rotated_q_even, rotated_q_odd, rotated_k_new_even, rotated_k_new_odd)
        }
    };

    let group_map = alloc::format!("s,{group}*u+g,i->sugi");
    let q_even_grouped = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (rotated_q_even, group_map.as_str()),
            (group_ones, "ug->sugi"),
        ],
    )?;
    let q_odd_grouped = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (rotated_q_odd, group_map.as_str()),
            (group_ones, "ug->sugi"),
        ],
    )?;

    let score_cached_even_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (q_even_grouped, "sugi->stugi"),
            (k_even_cache, "tui->stugi"),
        ],
    )?;
    let score_cached_even = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_cached_even_product,
        "stugi->stugi",
        "stug->stugi",
    )?;
    let score_cached_odd_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(q_odd_grouped, "sugi->stugi"), (k_odd_cache, "tui->stugi")],
    )?;
    let score_cached_odd = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_cached_odd_product,
        "stugi->stugi",
        "stug->stugi",
    )?;
    let score_cached = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[
            (score_cached_even, "stug->stug"),
            (score_cached_odd, "stug->stug"),
        ],
    )?;
    let score_cached_scaled = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(score_cached, "stug->stug"), (inv_sqrt_head_dim, "->stug")],
    )?;

    let score_new_even_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (q_even_grouped, "sugi->swugi"),
            (rotated_k_new_even, "wui->swugi"),
        ],
    )?;
    let score_new_even = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_new_even_product,
        "swugi->swugi",
        "swug->swugi",
    )?;
    let score_new_odd_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (q_odd_grouped, "sugi->swugi"),
            (rotated_k_new_odd, "wui->swugi"),
        ],
    )?;
    let score_new_odd = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_new_odd_product,
        "swugi->swugi",
        "swug->swugi",
    )?;
    let score_new = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[
            (score_new_even, "swug->swug"),
            (score_new_odd, "swug->swug"),
        ],
    )?;
    let score_new_scaled = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(score_new, "swug->swug"), (inv_sqrt_head_dim, "->swug")],
    )?;
    let neg_infinity = scalar_constant(program, f32::NEG_INFINITY);
    let score_new_masked = elementwise(
        program,
        DType::Float32,
        ScalarOp::Select,
        &[
            (is_future, "sw->swug"),
            (neg_infinity, "->swug"),
            (score_new_scaled, "swug->swug"),
        ],
    )?;

    let score_max_cached = reduce(
        program,
        DType::Float32,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        score_cached_scaled,
        "stug->stug",
        "sug->stug",
    )?;
    let score_max_new = reduce(
        program,
        DType::Float32,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        score_new_masked,
        "swug->swug",
        "sug->swug",
    )?;
    let global_max = elementwise(
        program,
        DType::Float32,
        ScalarOp::Maximum,
        &[(score_max_cached, "sug->sug"), (score_max_new, "sug->sug")],
    )?;

    let shifted_cached = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[
            (score_cached_scaled, "stug->stug"),
            (global_max, "sug->stug"),
        ],
    )?;
    let weights_cached = elementwise(
        program,
        DType::Float32,
        ScalarOp::Exponential,
        &[(shifted_cached, "stug->stug")],
    )?;
    let shifted_new = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(score_new_masked, "swug->swug"), (global_max, "sug->swug")],
    )?;
    let weights_new = elementwise(
        program,
        DType::Float32,
        ScalarOp::Exponential,
        &[(shifted_new, "swug->swug")],
    )?;

    let sum_cached = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        weights_cached,
        "stug->stug",
        "sug->stug",
    )?;
    let sum_new = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        weights_new,
        "swug->swug",
        "sug->swug",
    )?;
    let weight_sum = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(sum_cached, "sug->sug"), (sum_new, "sug->sug")],
    )?;
    let inv_weight_sum = elementwise(
        program,
        DType::Float32,
        ScalarOp::Reciprocal,
        &[(weight_sum, "sug->sug")],
    )?;

    let attended_cached_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(weights_cached, "stug->stugd"), (v_cache, "tud->stugd")],
    )?;
    let attended_cached = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        attended_cached_product,
        "stugd->stugd",
        "sugd->stugd",
    )?;
    let attended_new_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(weights_new, "swug->swugd"), (v_new, "wud->swugd")],
    )?;
    let attended_new = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        attended_new_product,
        "swugd->swugd",
        "sugd->swugd",
    )?;
    let attended_sum = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[
            (attended_cached, "sugd->sugd"),
            (attended_new, "sugd->sugd"),
        ],
    )?;
    let attended = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(attended_sum, "sugd->sugd"), (inv_weight_sum, "sug->sugd")],
    )?;

    let wo_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(attended, "sugd->sugdo"), (wo, "ugdo->sugdo")],
    )?;
    let attn_out = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        wo_product,
        "sugdo->sugdo",
        "so->sugdo",
    )?;

    let residual1 = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(attn_out, "sd->sd"), (x, "sd->sd")],
    )?;

    let normed2 = rmsnorm(program, residual1, ffn_norm_weight, inv_dim, eps)?;

    let (ffn_out, site) = append_moe_ffn(
        program,
        layer,
        normed2,
        gate_inp,
        expert_w_gate,
        expert_w_up,
        expert_w_down,
        expert_count,
        expert_used_count,
        ones,
        ExpertGatingFunc::Softmax,
        None,
    )?;

    let x_next = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(ffn_out, "sd->sd"), (residual1, "sd->sd")],
    )?;

    Ok((x_next, (rotated_k_new_even, rotated_k_new_odd, v_new), site))
}

/// Which mixer one transformer block runs. LFM2.5-8B-A1B (`general.architecture
/// = "lfm2moe"`) hybridizes short-convolution and attention blocks in the same
/// 24-layer stack, and GGUF carries no `layer_types` metadata key for this
/// architecture (confirmed absent on the real checkpoint's own metadata dump)
/// -- the only ground truth is which tensors a block's own name prefix owns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Attention,
    ShortConv,
}

impl LayerKind {
    /// Derives one block's kind from its own tensor name set: LFM2.5-8B-A1B's
    /// real checkpoint shows every block owns exactly one of
    /// `blk.{layer}.attn_q.weight` or `blk.{layer}.shortconv.conv.weight`,
    /// never neither and never both, so this is a presence check, not a
    /// classifier -- the caller (whoever already has the checkpoint's tensor
    /// directory, e.g. `proxima-model-interop`) walks that directory once per
    /// block and hands this a `layer`-scoped name iterator; this function
    /// never reads a file itself, keeping `proxima-tensor` free of a GGUF
    /// dependency.
    pub fn from_tensor_names<'name>(
        names: impl IntoIterator<Item = &'name str>,
        layer: u32,
    ) -> Result<Self, TensorError> {
        let attention_marker = alloc::format!("blk.{layer}.attn_q.weight");
        let conv_marker = alloc::format!("blk.{layer}.shortconv.conv.weight");
        for name in names {
            if name == attention_marker {
                return Ok(Self::Attention);
            }
            if name == conv_marker {
                return Ok(Self::ShortConv);
            }
        }
        Err(TensorError::UndeterminedLayerKind { layer })
    }
}

/// A fixed-width causal depthwise convolution (`l_cache` taps, one weight per
/// channel per tap, no cross-channel mixing), built from the existing
/// `Input`/`Elementwise`/`Reduce`/`Iota`/`Constant` vocabulary with no new
/// `Op` -- the pipe question this crate's own rule forces before any new
/// type, answered by writing the expression below rather than by arguing for
/// one.
///
/// `specs/conv2d.toml`'s own doc already proved the naive route is closed:
/// windowing `x` directly with a negative-offset `Affine` map
/// (`s-(l_cache-1)+l`) fails `shape::bounds_check` globally, because an
/// iteration axis always starts at 0 and the check is over the *whole*
/// symbolic extent, not per element -- at `s=0, l=0` the window reaches
/// index `-(l_cache-1)`, unconditionally out of bounds regardless of how
/// large the buffer is. `conv2d.toml` closes that gap by pre-padding its
/// input's own data; this crate's op set has no concat/pad primitive to build
/// that padding for an internal (not caller-supplied) tensor, so this
/// function takes a different, still-existing-primitives route: it never
/// forms the negative index at all.
///
/// `raw_position = s + l - (l_cache - 1)` is computed as data (two `Iota`s
/// plus a `Constant` offset, exactly [`causal_mask`]'s own `is_future`
/// composition), `clamped_position = max(raw_position, 0)` (always inside
/// `[0, s_max]`, since `raw_position`'s own maximum, reached at
/// `l = l_cache - 1`, is exactly `s`), and `clamped_position` addresses `x`
/// through [`IndexMap::Computed`] -- the same gather
/// [`gathered_expert_product`] already uses to read a data-dependent row.
/// Taps whose *unclamped* position is negative (real left-padding) are zeroed
/// post-gather via `Select`, mirroring how [`causal_mask`] masks attention
/// scores rather than ever reading an invalid position.
pub fn causal_conv1d(
    program: &mut Vec<Op>,
    x: NodeId,
    weight: NodeId,
    l_cache: u32,
) -> Result<NodeId, TensorError> {
    if l_cache == 0 {
        return Err(TensorError::InvalidConvConfig { l_cache });
    }

    let sequence_index = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Symbolic(0),
        },
    );
    let tap_index = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(l_cache),
        },
    );
    let window_offset = scalar_constant(program, -((l_cache - 1) as f32));

    let sequence_plus_tap = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(sequence_index, "s->sl"), (tap_index, "l->sl")],
    )?;
    let raw_position = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(sequence_plus_tap, "sl->sl"), (window_offset, "->sl")],
    )?;

    // `clamped_position` must be an `Op::Reduce`, not a plain `Elementwise`,
    // even though the fold itself is trivial (`max` over a synthetic 2-wide
    // axis holding `[raw_position, 0]`): `bind::BoundOpBuilder::push`'s
    // `Op::Elementwise` arm only forces materialization for nodes it finds in
    // its own `operands` list, and a `Computed` gather's `indices` reference
    // lives on a *different* node's operand entry -- a lone `Elementwise`
    // referenced only that way can sit `held` (fusion-deferred) past the
    // point a later gather needs its buffer, surfacing as
    // `TensorError::NotLowerable`'s "operand buffer missing at evaluation
    // time" (confirmed empirically: a first version of this function used
    // exactly that shape and hit precisely this). `Op::Reduce`'s own arm
    // always `push_ready`s immediately (`bind.rs`'s `push`, the
    // `Op::Reduce(reduce)` match arm), which is why every existing gather
    // index in this crate (`route` in [`gathered_expert_product`]) is already
    // a `Reduce`, never a bare `Elementwise` -- this mirrors that, rather
    // than being a new exception.
    let candidate_axis = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(2),
        },
    );
    let zero = scalar_constant(program, 0.0);
    // `zero_wide`, not the rank-0 `zero` above, is `is_raw_slot`'s second
    // operand: a rank-0 operand contributes no extent to any axis, so
    // `candidate_axis` alone (which only addresses `c`) would leave `s` and
    // `l` unconstrained on this node and `shape::infer` rejects that
    // (`TensorError::UnconstrainedDim`) -- every other broadcast pair in this
    // crate (e.g. `neg_infinity`/`is_future` in [`causal_mask`]'s callers)
    // always has a same-call sibling operand of the full iteration rank for
    // exactly this reason; `zero_wide`'s declared `[Symbolic(0), l_cache]`
    // shape is that sibling here, still comparing against literal `0.0`.
    let zero_wide = op::append(
        program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Symbolic(0), Extent::Static(l_cache)],
            value: 0.0,
        },
    );
    let is_raw_slot = elementwise(
        program,
        DType::Float32,
        ScalarOp::Equal,
        &[(candidate_axis, "c->slc"), (zero_wide, "sl->slc")],
    )?;
    let candidate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Select,
        &[
            (is_raw_slot, "slc->slc"),
            (raw_position, "sl->slc"),
            (zero, "->slc"),
        ],
    )?;
    let clamped_position = reduce(
        program,
        DType::Int32,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        candidate,
        "slc->slc",
        "sl->slc",
    )?;

    let negative_one = scalar_constant(program, -1.0);
    let is_valid = elementwise(
        program,
        DType::Float32,
        ScalarOp::Greater,
        &[(raw_position, "sl->sl"), (negative_one, "->sl")],
    )?;

    let gathered_map = IndexMap::Computed {
        indices: clamped_position,
        index_map: map::projection(3, &[0, 1]),
        base: IndexPattern {
            iter_rank: 3,
            axes: alloc::vec![
                AxisIndex::default(),
                AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(2)).collect(),
                    offset: 0,
                    len: None,
                },
            ],
        },
        gathered_dim: 0,
    };
    let windowed = op::append(
        program,
        Op::Elementwise {
            dtype: DType::Float32,
            body: ScalarOp::Identity,
            operands: alloc::vec![(x, gathered_map)],
            name: None,
        },
    );

    // `weight`'s own declared shape is `[embedding, l_cache]` (`d` axis
    // first/outer, `l` axis last/fastest) -- `dl->sld`, not `ld->sld` --
    // matching `row_major_strides`'s (`bind.rs`) own last-axis-fastest
    // convention against the REAL on-disk tensor's physical layout: GGUF's
    // own `ne[0] = l_cache` is ggml's fastest axis (confirmed against
    // `llama.cpp`'s own `create_tensor(.., {n_shortconv_l_cache, n_embd},
    // ..)`), so `l_cache` -- not `embedding` -- is genuinely contiguous per
    // channel on disk. Every caller of this function must declare `weight`'s
    // `Op::Input` shape the same way.
    let tap_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(windowed, "sld->sld"), (weight, "dl->sld")],
    )?;
    let zero_tap = scalar_constant(program, 0.0);
    let masked_tap = elementwise(
        program,
        DType::Float32,
        ScalarOp::Select,
        &[
            (is_valid, "sl->sld"),
            (tap_product, "sld->sld"),
            (zero_tap, "->sld"),
        ],
    )?;

    reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        masked_tap,
        "sld->sld",
        "sd->sld",
    )
}

/// LFM2's gated short-convolution mixer, [`append_mistral_layer`]'s
/// attention-block counterpart for a `LayerKind::ShortConv` block: three
/// separate `embedding x embedding` projections (`b_proj`/`c_proj`/`x_proj`)
/// stand in for the real checkpoint's single fused `blk.N.shortconv.in_proj.weight`
/// (`[embedding, 3*embedding]`, one matmul producing three same-width
/// branches) -- **not** a shape this function chose for its own sake. A
/// single reduce over the fused weight followed by three static-offset
/// slices back out was the first version of this function, and it does not
/// type-check: `shape::unify_iteration_space` (`shape.rs:195-212`) resolves a
/// pure single-term axis's extent from the *sliced operand's own buffer
/// width* regardless of its offset (confirmed empirically --
/// `TensorError::ExtentMismatch` at the first later consumer that expects
/// `embedding`, not `3*embedding`), so an offset-only slice of a fused
/// `[s, 3*embedding]` tensor can never narrow to `[s, embedding]` inside this
/// algebra's current `Affine` grammar -- only a *strided* axis (coefficient
/// != 1, [`append_attention_mixer`]'s own `2*i` RoPE pattern) escapes that
/// branch, and a contiguous 2048-wide slice is not a stride. Splitting into
/// three independently-shaped `Input`s sidesteps the gap entirely, at the
/// cost of pushing the fused-to-three-tensor split to whichever binder loads
/// the real checkpoint (unimplemented this session, same as
/// [`append_mistral_layer`]'s own `wq`/`wk`/`wv` already being separate
/// `Input`s despite some checkpoints fusing QKV on disk).
///
/// `b_proj` gates the ungated `x_proj` branch, [`causal_conv1d`] convolves
/// the gated result causally over `l_cache` taps, `c_proj` gates the
/// convolved result, and `out_proj` projects back to `embedding` width --
/// LiquidAI's published LFM2 short-convolution block, `y = out_proj(C ⊙
/// conv(B ⊙ x))`, no activation function inside the block itself, unlike the
/// SwiGLU FFN every layer still runs after it. This branch assignment and
/// tap direction are read directly off HuggingFace's own reference
/// implementation (`transformers/models/lfm2_moe/modeling_lfm2_moe.py`,
/// `Lfm2MoeShortConv.slow_forward`, lines 434-465 of the checked-out
/// package): `BCx = in_proj(x).transpose(-1,-2)` then `B, C, x =
/// BCx.chunk(3, dim=-2)` -- `B` first, `C` second, ungated `x` third along
/// the packed axis, exactly `b_proj`/`c_proj`/`x_proj`'s declared order
/// below -- `Bx = B * x`, `conv_out = self.conv(Bx)` (an `nn.Conv1d` with
/// `padding = l_cache - 1`, left-only), `y = C * conv_out`,
/// `out_proj(y)`. [`causal_conv1d`]'s own tap convention (`l = l_cache - 1`
/// is the current position, `l = 0` the furthest lookback) matches
/// `nn.Conv1d`'s left-padded-causal convolution exactly: with `K - 1` zeros
/// prepended, `output[t] = sum_k weight[k] * padded_input[t + k]`, so
/// `weight[K-1]` always pairs with `input[t]` and `weight[0]` with
/// `input[t - (K-1)]`, the same pairing this function's own weight map
/// (`ld->sld`) uses.
#[allow(clippy::too_many_arguments)]
pub fn append_lfm2_conv_mixer(
    program: &mut Vec<Op>,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    norm_weight: NodeId,
    b_proj: NodeId,
    c_proj: NodeId,
    x_proj: NodeId,
    conv_weight: NodeId,
    out_proj: NodeId,
    l_cache: u32,
) -> Result<NodeId, TensorError> {
    let normed = rmsnorm(program, x, norm_weight, inv_dim, eps)?;

    let branch_b_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "sd->sdg"), (b_proj, "dg->sdg")],
    )?;
    let branch_b = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        branch_b_product,
        "sdg->sdg",
        "sg->sdg",
    )?;

    let branch_x_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "sd->sdg"), (x_proj, "dg->sdg")],
    )?;
    let branch_x = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        branch_x_product,
        "sdg->sdg",
        "sg->sdg",
    )?;

    let branch_c_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "sd->sdg"), (c_proj, "dg->sdg")],
    )?;
    let branch_c = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        branch_c_product,
        "sdg->sdg",
        "sg->sdg",
    )?;

    let gated_input = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(branch_b, "sg->sg"), (branch_x, "sg->sg")],
    )?;
    let convolved = causal_conv1d(program, gated_input, conv_weight, l_cache)?;
    let gated_output = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(convolved, "sg->sg"), (branch_c, "sg->sg")],
    )?;

    let out_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(gated_output, "sd->sdo"), (out_proj, "do->sdo")],
    )?;
    let mixer_out = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        out_product,
        "sdo->sdo",
        "so->sdo",
    )?;

    elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(mixer_out, "sd->sd"), (x, "sd->sd")],
    )
}

/// One token's worth of the gated-DeltaNet recurrence Qwen3.5's linear
/// attention (SSM) layers run -- llama.cpp's own reference,
/// `llm_build_delta_net_base::build_delta_net_autoregressive`
/// (`src/models/delta-net-base.cpp`, the `n_tokens == 1` path
/// `llm_build_delta_net_base::build_delta_net` dispatches to), transcribed
/// op-for-op onto this crate's existing `Elementwise`/`Reduce` vocabulary --
/// no new [`Op`] variant, per this crate's own reuse-first rule: a
/// state-carrying IIR recurrence over a caller-owned `[key_dim, value_dim,
/// head]` matrix is exactly what an `Input`/`Output` pair already expresses
/// for [`append_mistral_cached_layer`]'s own KV cache, so the persistent
/// state here is a caller-provided `state_in` node returned again as
/// `state_out`, not a new stateful primitive.
///
/// Per head `h`, key axis `i`, value axis `j` (`state[i,j,h]`, `q`/`k`
/// share `i`, `v` shares `j` with `state`'s second axis): `state = state *
/// exp(gate)` (`decay`), `v_pred[j] = sum_i state[i,j] * k[i]`
/// (`llama.cpp:305-306`, `sk = sum_rows(state * k)`), `delta[j] = beta *
/// (v[j] - v_pred[j])` (`:309-311`), `state[i,j] += k[i] * delta[j]`
/// (`:313-317`, the outer-product update), `out[j] = sum_i state[i,j] *
/// q_scaled[i]` (`:322-323`, read-out uses the UPDATED state) -- `q_scaled
/// = q / sqrt(key_dim)` is applied by the caller (`llama.cpp:295`,
/// `q = ggml_scale(ctx0, q, scale)`), matching every other pre-scaled `q`
/// this crate's own attention mixers already take.
///
/// `gate` and `beta` arrive already reduced to one scalar per head per
/// token (`llama.cpp`'s own `softplus(alpha + dt_bias) * ssm_a` and
/// `sigmoid(beta_proj)` respectively) -- this function only runs the
/// recurrence, never the projections that produce its inputs.
///
/// `head` is a format-interpolated run of letters, not a single character,
/// the same widening [`rmsnorm_per_head`] already makes: [`repeat_kv_heads`]'s
/// own doc proves this algebra cannot merge a `u,g` (kv-head, group) split
/// back into one physical head axis, so [`append_qwen35_ssm_mixer`] calls
/// this with `head = "ug"` and every map below (`i{head}`, `{head}`,
/// `ij{head}`) carries both letters through unchanged -- the recurrence
/// itself is per-head and never mixes heads, so nothing in its math depends
/// on the head space being one physical axis.
#[allow(clippy::too_many_arguments)]
pub fn append_qwen35_delta_net_step(
    program: &mut Vec<Op>,
    query: NodeId,
    key: NodeId,
    value: NodeId,
    gate: NodeId,
    beta: NodeId,
    state_in: NodeId,
    inv_sqrt_key_dim: NodeId,
    head: &str,
) -> Result<(NodeId, NodeId), TensorError> {
    let i_head = alloc::format!("i{head}->i{head}");
    let i_head_bcast = alloc::format!("->i{head}");
    let head_head = alloc::format!("{head}->{head}");
    let ij_head = alloc::format!("ij{head}->ij{head}");
    let head_to_ij_head = alloc::format!("{head}->ij{head}");
    let j_head = alloc::format!("j{head}->j{head}");
    let head_to_j_head = alloc::format!("{head}->j{head}");
    let i_head_to_ij_head = alloc::format!("i{head}->ij{head}");
    let j_head_to_ij_head = alloc::format!("j{head}->ij{head}");
    let ij_head_reduce_in = alloc::format!("ij{head}->ij{head}");
    let ij_head_reduce_out = alloc::format!("j{head}->ij{head}");

    let query_scaled = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(query, i_head.as_str()), (inv_sqrt_key_dim, i_head_bcast.as_str())],
    )?;
    let decay = elementwise(
        program,
        DType::Float32,
        ScalarOp::Exponential,
        &[(gate, head_head.as_str())],
    )?;
    let state_decayed = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(state_in, ij_head.as_str()), (decay, head_to_ij_head.as_str())],
    )?;

    let value_pred_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(state_decayed, ij_head.as_str()), (key, i_head_to_ij_head.as_str())],
    )?;
    let value_pred = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        value_pred_product,
        ij_head_reduce_in.as_str(),
        ij_head_reduce_out.as_str(),
    )?;

    let residual = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(value, j_head.as_str()), (value_pred, j_head.as_str())],
    )?;
    let delta = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(residual, j_head.as_str()), (beta, head_to_j_head.as_str())],
    )?;

    let update = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(key, i_head_to_ij_head.as_str()), (delta, j_head_to_ij_head.as_str())],
    )?;
    let state_out = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(state_decayed, ij_head.as_str()), (update, ij_head.as_str())],
    )?;

    let out_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(state_out, ij_head.as_str()), (query_scaled, i_head_to_ij_head.as_str())],
    )?;
    let out = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        out_product,
        ij_head_reduce_in.as_str(),
        ij_head_reduce_out.as_str(),
    )?;

    Ok((out, state_out))
}

/// `log(1 + exp(x))`, Qwen3.5's own `alpha_softplus` gate input
/// (`llama.cpp:370`, `ggml_softplus`) -- not a [`ScalarOp`] primitive, so
/// composed from the two that are: [`ScalarOp::Exponential`] then
/// [`ScalarOp::Add`] against a `one` constant then [`ScalarOp::Logarithm`],
/// the same compose-not-mint move [`ExpertGatingFunc::Sigmoid`]'s own
/// `neg -> exp -> +1 -> reciprocal` chain already makes for a activation this
/// crate has no dedicated variant for.
pub fn softplus(program: &mut Vec<Op>, x: NodeId, one: NodeId, map: &str) -> Result<NodeId, TensorError> {
    let target = map.rsplit("->").next().unwrap_or(map);
    let one_map = alloc::format!("->{target}");
    let exp_x = elementwise(program, DType::Float32, ScalarOp::Exponential, &[(x, map)])?;
    let one_plus_exp = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(exp_x, map), (one, one_map.as_str())],
    )?;
    elementwise(program, DType::Float32, ScalarOp::Logarithm, &[(one_plus_exp, map)])
}

/// [`rmsnorm`]'s L2-normalize variant: `x / sqrt(sum(x^2) + eps)`, no
/// mean-divide and no learnable `gamma` -- `ggml_l2_norm`
/// (`qwen35.cpp:428-429`, applied to `q_conv`/`k_conv` with no weight
/// tensor), unlike [`rmsnorm`]'s `mean_square = sum_squares / dim` and its
/// trailing `gamma` multiply. `map`/`sum_map` follow [`rmsnorm_per_head`]'s
/// own per-axis convention so the same function serves whichever axis (head
/// dim here, embedding elsewhere) is being normalized.
pub fn l2norm(
    program: &mut Vec<Op>,
    x: NodeId,
    eps: NodeId,
    map: &str,
    sum_map: &str,
) -> Result<NodeId, TensorError> {
    let reduced = sum_map.split("->").next().unwrap_or(sum_map);
    let reduced_map = alloc::format!("{reduced}->{reduced}");

    let squared = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(x, map), (x, map)])?;
    let sum_squares = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        squared,
        map,
        sum_map,
    )?;
    let sum_squares_eps = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(sum_squares, reduced_map.as_str()), (eps, reduced_map.as_str())],
    )?;
    let norm = elementwise(
        program,
        DType::Float32,
        ScalarOp::SquareRoot,
        &[(sum_squares_eps, reduced_map.as_str())],
    )?;
    let inv_norm = elementwise(
        program,
        DType::Float32,
        ScalarOp::Reciprocal,
        &[(norm, reduced_map.as_str())],
    )?;
    elementwise(program, DType::Float32, ScalarOp::Multiply, &[(x, map), (inv_norm, sum_map)])
}

/// `x * sigmoid(x)`, `ggml_silu`'s own contract (`qwen35.cpp:391-392`, run on
/// `conv_output_proper` before the q/k/v split) -- composed from
/// [`ScalarOp::Negate`]/[`ScalarOp::Exponential`]/[`ScalarOp::Add`]/[`ScalarOp::Reciprocal`], the same
/// `1/(1+e^-x)` chain [`ExpertGatingFunc::Sigmoid`] already builds, then one
/// more [`ScalarOp::Multiply`] against the un-gated input. No dedicated
/// `Sigmoid`/`Silu` [`ScalarOp`] exists, matching that chain's own precedent
/// for an activation this crate composes rather than mints.
pub fn silu(program: &mut Vec<Op>, x: NodeId, one: NodeId, map: &str) -> Result<NodeId, TensorError> {
    let target = map.rsplit("->").next().unwrap_or(map);
    let one_map = alloc::format!("->{target}");
    let neg_x = elementwise(program, DType::Float32, ScalarOp::Negate, &[(x, map)])?;
    let exp_neg_x = elementwise(program, DType::Float32, ScalarOp::Exponential, &[(neg_x, map)])?;
    let one_plus_exp = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(exp_neg_x, map), (one, one_map.as_str())],
    )?;
    let gate = elementwise(program, DType::Float32, ScalarOp::Reciprocal, &[(one_plus_exp, map)])?;
    elementwise(program, DType::Float32, ScalarOp::Multiply, &[(x, map), (gate, map)])
}

/// Reads `width` contiguous channels of `x` (`[s, total_channels]`) starting
/// at `offset`, as a fresh `[s, width]` node -- the piece
/// [`append_qwen35_conv_branch`] needs three times (`q`/`k`/`v` out of one
/// fused conv output) that a plain offset [`AxisIndex`] slice cannot give it:
/// [`append_lfm2_conv_mixer`]'s own doc already proves a *nonzero*-offset
/// slice of a wider operand needs a same-width "donor" operand to escape
/// `shape::unify_iteration_space`'s pure-projection extent rule, and
/// `shape.rs`'s own
/// `an_offset_zero_slice_narrower_than_its_operand_is_still_ambiguous` test
/// proves the donor trick still fails at *zero* offset (`q`'s own case here,
/// `qkv_dim`'s first channel) -- there is no bit in `AxisIndex` that
/// disambiguates "the whole axis" from "a same-origin narrower window".
///
/// This sidesteps offset addressing entirely: `channel_index` (an
/// [`Op::Iota`] over `total_channels`) and `target = within + offset`
/// (`within` a second `Iota` over `width`) feed [`ScalarOp::Equal`] to build
/// a one-hot mask, `x` is multiplied against it and reduced over the
/// channel axis -- the same select-then-reduce shape [`causal_conv1d`]'s own
/// `is_raw_slot`/`masked_tap` already use for a data-computed position,
/// applied here to a compile-time-constant one. Zero offset costs nothing
/// extra by this route, unlike the donor slice which cannot express it at
/// all.
pub fn channel_slice(
    program: &mut Vec<Op>,
    x: NodeId,
    total_channels: u32,
    offset: u32,
    width: u32,
) -> Result<NodeId, TensorError> {
    let channel_index = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(total_channels),
        },
    );
    let within_index = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(width),
        },
    );
    let offset_const = scalar_constant(program, offset as f32);
    let target = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(within_index, "w->w"), (offset_const, "->w")],
    )?;
    let mask = elementwise(
        program,
        DType::Float32,
        ScalarOp::Equal,
        &[(channel_index, "d->dw"), (target, "w->dw")],
    )?;
    let selected = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(x, "sd->sdw"), (mask, "dw->sdw")],
    )?;
    reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        selected,
        "sdw->sdw",
        "sw->sdw",
    )
}

/// [`channel_slice`]'s own technique, generalized over a leading per-head
/// axis: `x` is `[s, heads, total_channels]` (`heads` contiguous blocks of
/// `total_channels`, e.g. a fused dense-attention `q`/`gate` chunk pair
/// repeated per head, `qwen3_next`'s own `q_proj(x).view(..., heads,
/// 2 * head_dim)` before its `torch.chunk(2, dim=-1)`), and this reads
/// `width` channels at `offset` within every head's own block
/// independently -- `channel_slice`'s single `(s, d)` mask can only carve
/// one contiguous window out of ONE flat channel axis, which is wrong here
/// because each head's window sits at a different flat offset (`head *
/// total_channels + offset`); the extra `Iota` over `heads` folds that
/// per-head stride into the same `target` the mask compares against, one
/// mask covering every head's window in a single select-then-reduce pass.
pub fn per_head_channel_slice(
    program: &mut Vec<Op>,
    x: NodeId,
    heads: u32,
    total_channels: u32,
    offset: u32,
    width: u32,
) -> Result<NodeId, TensorError> {
    let channel_index = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(heads * total_channels),
        },
    );
    let within_index = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(width),
        },
    );
    let head_index = op::append(program, Op::Iota { dtype: DType::Float32, extent: Extent::Static(heads) });
    let period_const = scalar_constant(program, total_channels as f32);
    let offset_const = scalar_constant(program, offset as f32);
    let head_base = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(head_index, "h->h"), (period_const, "->h")],
    )?;
    let head_start = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(head_base, "h->h"), (offset_const, "->h")],
    )?;
    let target = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(head_start, "h->hw"), (within_index, "w->hw")],
    )?;
    let mask = elementwise(
        program,
        DType::Float32,
        ScalarOp::Equal,
        &[(channel_index, "d->dhw"), (target, "hw->dhw")],
    )?;
    let selected = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(x, "sd->sdhw"), (mask, "dhw->sdhw")],
    )?;
    reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        selected,
        "sdhw->sdhw",
        "shw->sdhw",
    )
}

/// [`channel_slice`]'s own select-then-reduce technique, generalized over an
/// ALREADY-split leading per-head axis: `x` is `[s, head, total_channels]`
/// (a head axis of its own, not [`per_head_channel_slice`]'s flat
/// `heads*total_channels` an activation like [`append_qwen35_dense_attention_layer`]'s
/// own `q`/`k` never has after `rmsnorm_per_head`), and this reads `width`
/// channels at the SAME `offset` uniformly across every head (unlike
/// [`per_head_channel_slice`]'s per-head-varying stride, there needed only
/// because the input axis was still flat). A plain affine projection
/// (`"s,h,i+64->shi"`) cannot express this narrowing on its own -- shape
/// inference unifies every operand touching iteration letter `i` to the
/// SAME extent, so a bare projection off `x`'s own `total_channels`-wide
/// axis pins `i` at `total_channels`, not `width`; only the mask-then-reduce
/// route can produce a genuinely narrower output.
pub fn per_head_channel_range(
    program: &mut Vec<Op>,
    x: NodeId,
    head: &str,
    total_channels: u32,
    offset: u32,
    width: u32,
) -> Result<NodeId, TensorError> {
    let channel_index = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(total_channels),
        },
    );
    let within_index = op::append(
        program,
        Op::Iota {
            dtype: DType::Float32,
            extent: Extent::Static(width),
        },
    );
    let offset_const = scalar_constant(program, offset as f32);
    let target = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(within_index, "w->w"), (offset_const, "->w")],
    )?;
    let mask = elementwise(
        program,
        DType::Float32,
        ScalarOp::Equal,
        &[(channel_index, "d->dw"), (target, "w->dw")],
    )?;
    let x_map = alloc::format!("s{head}d->s{head}dw");
    let mask_map = alloc::format!("dw->s{head}dw");
    let selected = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(x, x_map.as_str()), (mask, mask_map.as_str())],
    )?;
    let in_map = alloc::format!("s{head}dw->s{head}dw");
    let out_map = alloc::format!("s{head}w->s{head}dw");
    reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        selected,
        in_map.as_str(),
        out_map.as_str(),
    )
}

/// Qwen3.5's `q|k|v` conv branch -- llama.cpp's own
/// `build_layer_attn_linear` (`qwen35.cpp:385-431`): `causal_conv1d` over
/// the fused `qkv_mixed` (`conv_input`, `:385`), [`silu`] (`:391-392`), a
/// three-way [`channel_slice`] split at `qkv_dim = 2*key_dim + value_dim`
/// (`q` at offset `0`, `k` at `key_dim`, `v` at `2*key_dim` --
/// `q_conv`/`k_conv`/`v_conv`'s own `ggml_view_4d` offsets, `:399-419`),
/// then [`l2norm`] on `q`/`k` only (`:428-429`, `v` is never normalized).
/// The GQA head repeat (`:437-440`) is a separate, independently-testable
/// step -- see [`repeat_kv_heads`].
// unwired: this one specifically, not the whole mixer -- `causal_conv1d`
// windows a whole in-graph sequence with zero-boundary padding, which fits
// a prefill call but not a decode step against a persisted history cache,
// so `append_qwen35_ssm_mixer` reimplements this function's own
// silu/channel_slice/l2norm body against the additive cached-conv split its
// own doc describes, rather than calling this. A prefill-only qwen35
// program (mirroring `lfm2_forward_program_with_experts`'s own prefill-only
// scope) is this function's real caller, not built this session.
#[allow(dead_code, clippy::too_many_arguments)]
pub fn append_qwen35_conv_branch(
    program: &mut Vec<Op>,
    qkv_mixed: NodeId,
    conv_weight: NodeId,
    eps: NodeId,
    one: NodeId,
    key_dim: u32,
    value_dim: u32,
    l_cache: u32,
) -> Result<(NodeId, NodeId, NodeId), TensorError> {
    let qkv_dim = 2 * key_dim + value_dim;
    let convolved = causal_conv1d(program, qkv_mixed, conv_weight, l_cache)?;
    let activated = silu(program, convolved, one, "sd->sd")?;

    let q_raw = channel_slice(program, activated, qkv_dim, 0, key_dim)?;
    let k_raw = channel_slice(program, activated, qkv_dim, key_dim, key_dim)?;
    let v_conv = channel_slice(program, activated, qkv_dim, 2 * key_dim, value_dim)?;

    let q_conv = l2norm(program, q_raw, eps, "sw->sw", "s->sw")?;
    let k_conv = l2norm(program, k_raw, eps, "sw->sw", "s->sw")?;

    Ok((q_conv, k_conv, v_conv))
}

/// The GQA head repeat `q_conv`/`k_conv` need before
/// [`append_qwen35_delta_net_step`] (`qwen35.cpp:437-440`,
/// `ggml_repeat_4d(.., num_v_heads, ..)`): `num_k_heads` (16) real kv heads
/// broadcast to `num_v_heads` (48) query/value heads, 3-wide groups.
///
/// [`append_attention_mixer`]'s own `group_map`/`group_ones` pair already
/// proves the technique this reuses: reading an operand's real axis while a
/// *new* iteration letter is simply absent from that operand's own map
/// broadcasts across it for free (`rotated_k`'s `"tui->stugi"` there never
/// mentions `g`), and multiplying against an all-ones donor of the new
/// letters' shape (`group_ones`, `"ug->sugi"`) is what makes
/// `shape::unify_iteration_space` resolve `g`'s extent at all -- `x` alone
/// (real axis `u`, no `g` term) leaves `g` unconstrained.
///
/// This never merges `u`/`g` back into one physical `h = group*u+g` axis:
/// `shape::project_output_shape` rejects any `Reduce` `out_map` axis that
/// is not a pure single-term projection ("reduce output maps must be pure
/// projections in v1"), and a plain [`Op::Elementwise`]'s output shape *is*
/// its iteration space, so two loop letters cannot collapse into one output
/// letter in this algebra's current grammar -- the same reason
/// [`append_attention_mixer`] itself never merges them either, keeping every
/// downstream op split as `u,g` through to its own final output. A
/// genuinely single-axis repeat would need a scatter with a data-computed
/// `h = group*u+g` destination (the same [`IndexMap::Computed`] shape
/// [`causal_conv1d`]'s own `clamped_position` gather already uses for a
/// data-computed *source*); not built this session.
pub fn repeat_kv_heads(
    program: &mut Vec<Op>,
    x: NodeId,
    kv_heads: u32,
    group: u32,
) -> Result<NodeId, TensorError> {
    let group_ones = op::append(
        program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
            value: 1.0,
        },
    );
    elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(x, "sud->sugd"), (group_ones, "ug->sugd")],
    )
}

/// `1/(1+e^-x)` -- the exact `negate -> exp -> +1 -> reciprocal` chain
/// [`ExpertGatingFunc::Sigmoid`] already composes inline, factored out once
/// [`append_qwen35_ssm_mixer`] needs it twice (`beta`, the attention gate),
/// the same "worth naming at two callers" threshold [`silu`]/[`softplus`]
/// already crossed for their own chains.
pub fn sigmoid(program: &mut Vec<Op>, x: NodeId, one: NodeId, map: &str) -> Result<NodeId, TensorError> {
    let target = map.rsplit("->").next().unwrap_or(map);
    let one_map = alloc::format!("->{target}");
    let neg_x = elementwise(program, DType::Float32, ScalarOp::Negate, &[(x, map)])?;
    let exp_neg_x = elementwise(program, DType::Float32, ScalarOp::Exponential, &[(neg_x, map)])?;
    let one_plus_exp = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(exp_neg_x, map), (one, one_map.as_str())],
    )?;
    elementwise(program, DType::Float32, ScalarOp::Reciprocal, &[(one_plus_exp, map)])
}

/// Qwen3.5's gated-DeltaNet mixer, one decode step (`n_tokens == 1`, the same
/// scope [`append_qwen35_delta_net_step`]'s own doc already commits to) --
/// llama.cpp's own `build_layer_attn_linear` (`qwen35.cpp:335-466`) run
/// op-for-op: `build_qkvz` (`:353-356`, `qkv_mixed`/`z`), `beta`/`gate`
/// (`:358-376`, `sigmoid(ssm_beta @ x)` / `ssm_a * softplus(ssm_alpha @ x +
/// ssm_dt)`), the causal conv + [`silu`] + channel split + [`l2norm`]
/// ([`append_qwen35_conv_branch`]'s own body, `:391-429`, reproduced here
/// against a persisted history window instead of [`causal_conv1d`]'s own
/// zero-boundary window -- see the conv step below), the GQA repeat
/// ([`repeat_kv_heads`], `:437-440`), the recurrence itself
/// ([`append_qwen35_delta_net_step`], `build_delta_net_autoregressive`,
/// `delta-net-base.cpp:289-370`), gated RMSNorm (`build_norm_gated`,
/// `qwen35.cpp:243-250`: `rmsnorm(out) * silu(z)`), and the output
/// projection + residual (`:456-464`, folded into the block-level
/// `ggml_add(cur, inpSA)` at `:180`) -- the pre-mixer `rmsnorm` and the
/// post-mixer residual add both happen INSIDE this function, the same
/// choice [`append_lfm2_conv_mixer`] already makes for its own block.
///
/// The `u,g` seam: [`repeat_kv_heads`]'s own doc proves this algebra can
/// never merge a `u` (kv-head)/`g` (group) split back into one physical head
/// axis -- `shape::project_output_shape` rejects any `Reduce` `out_map`
/// axis that is not a pure single-term projection, and a plain
/// `Elementwise`'s output shape IS its iteration space, so two loop letters
/// cannot collapse into one output letter. [`append_qwen35_delta_net_step`]'s
/// own maps are all per-head (nothing in the recurrence mixes heads), so
/// widening its `head` parameter from one letter to `"ug"` costs nothing but
/// string interpolation -- verified by reading its maps before relying on
/// it, not assumed. `value`/`gate`/`beta` all decompose from their real flat
/// `num_v_heads` axis via the identical `group*u+g` computed read
/// [`append_attention_mixer`]'s own `group_map` already proves; `value`
/// folds the within-head axis `j` into the same expression
/// (`(group*head_v_dim)*u + head_v_dim*g + j`, still one `Affine` axis
/// expression -- `parse_axis_expr` sums an arbitrary run of `+`-joined
/// terms, not just two, confirmed by reading it before relying on it).
///
/// State threading mirrors [`append_mistral_cached_layer`]: both caches
/// (`state_in`/`state_out`, [`append_qwen35_delta_net_step`]'s own contract,
/// and `conv_history_in`, the `l_cache - 1` previous raw `qkv_mixed` rows)
/// are caller-persisted [`Op::Input`]s/return values, never concatenated
/// in-graph -- [`causal_conv1d`]'s own doc already establishes this op set
/// has no concat primitive. Instead of windowing (which would need a real
/// concat), the cached conv is a plain additive split: `conv_out = sum_w
/// weight[.., w] * history[w] + weight[.., l_cache - 1] * qkv_mixed_new`,
/// the same disjoint-source blend [`append_mistral_cached_layer`]'s own
/// `score_cached` + `score_new` split already uses for attention, minus the
/// online-softmax combine step (a linear conv sum splits for free; attention
/// only splits after `Maximum`/`Add` recombine it). The caller is
/// responsible for trimming/appending `qkv_mixed` (this function's second
/// return) into its own persisted history buffer, exactly as
/// [`append_mistral_cached_layer`]'s own [`CachedLayerRoots`] callers manage
/// their KV cache outside the graph -- shift-and-trim lives on the host, not
/// in the graph.
///
/// Which nonlinearity gates [`append_qwen35_ssm_mixer`]'s output norm --
/// `rmsnorm(delta_out) * activation(z)` (reference: PR 27742 line 2896-2899,
/// `build_norm_gated`, whose own comment names this "the one numerical
/// difference from Qwen3.5's GDN: sigmoid output gate, not silu"). Qwen3.5's
/// own checkpoint keeps [`GdnOutputGate::Silu`]; qwen4exp's GDN layers pass
/// [`GdnOutputGate::Sigmoid`] -- a layer-kind flag on the shared builder
/// rather than a duplicated function, since every other line of the mixer
/// (fused QKVZ, causal conv, delta-rule recurrence) is identical between the
/// two checkpoints.
/// [`append_qwen35_ssm_mixer`]'s output-gate selector -- see
/// [`qwen35_forward_program`] for the worked example passing
/// [`GdnOutputGate::Silu`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GdnOutputGate {
    /// `qwen35_forward_program`'s own GDN layers (`qwen35.cpp:243-250`).
    Silu,
    /// qwen4exp's GDN layers (reference: PR 27742 line 2895-2897) -- no
    /// production call site in this crate (that forward-program assembly is
    /// model-specific and lives in its own consuming crate); exercised
    /// today by `qwen35_ssm_mixer_sigmoid_gate_moves_the_output_away_from_silu`.
    Sigmoid,
}

/// [`append_qwen35_ssm_mixer_with_taps`]'s own return shape: every
/// intermediate a caller needs to bisect the mixer's tail against an
/// independent reference, in the order the builder computes them
/// (`spec.rs:6608-6928`). `qkv_mixed` is the fused `wqkv` projection
/// (Q/K/V still concatenated, pre-conv); `state_out` is the delta-net
/// recurrence's carried state; `delta_out` is the delta-net read-out
/// (`jug` layout, pre-norm); `z` is the raw output-gate projection
/// BEFORE [`GdnOutputGate`]'s silu/sigmoid split, so a caller can apply
/// either nonlinearity independently of which one this program's own
/// `output_gate` argument baked in; `gated_rmsnorm_out` is the per-head
/// RMSNorm output after its own `ssm_norm_weight` scale (before the `z`
/// gate multiplies in); `gated_value` is that result after the `z` gate
/// multiplies in (still per-head, pre-projection); `ssm_out_result` is
/// the `ssm_out` projection's reduce, before the residual add that
/// produces `mixer_out`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SsmMixerTaps {
    pub qkv_mixed: NodeId,
    pub state_out: NodeId,
    pub delta_out: NodeId,
    pub z: NodeId,
    pub gated_rmsnorm_out: NodeId,
    pub gated_value: NodeId,
    pub ssm_out_result: NodeId,
}

/// The GDN (gated delta-net) mixer [`qwen35_forward_program`] calls once
/// per non-attention layer -- see it there for the worked example of
/// wiring this builder's inputs. Returns `(x_next, qkv_mixed, state_out)`.
/// Thin wrapper over [`append_qwen35_ssm_mixer_with_taps`] for callers that
/// only need the three roots this signature already returned before taps
/// existed -- byte-identical program, since this only reshapes the return
/// value the shared builder already computed.
#[allow(clippy::too_many_arguments)]
pub fn append_qwen35_ssm_mixer(
    program: &mut Vec<Op>,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    head_eps: NodeId,
    one: NodeId,
    inv_sqrt_key_dim: NodeId,
    inv_head_v_dim: NodeId,
    attn_norm_weight: Option<NodeId>,
    wqkv: NodeId,
    wqkv_gate: NodeId,
    conv_weight: NodeId,
    conv_history_in: NodeId,
    ssm_beta: NodeId,
    ssm_alpha: NodeId,
    ssm_dt_bias: NodeId,
    ssm_a: NodeId,
    ssm_norm_weight: NodeId,
    ssm_out: NodeId,
    state_in: NodeId,
    key_dim: u32,
    value_dim: u32,
    kv_heads: u32,
    group: u32,
    l_cache: u32,
    output_gate: GdnOutputGate,
) -> Result<(NodeId, NodeId, NodeId), TensorError> {
    let (mixer_out, taps) = append_qwen35_ssm_mixer_with_taps(
        program,
        x,
        inv_dim,
        eps,
        head_eps,
        one,
        inv_sqrt_key_dim,
        inv_head_v_dim,
        attn_norm_weight,
        wqkv,
        wqkv_gate,
        conv_weight,
        conv_history_in,
        ssm_beta,
        ssm_alpha,
        ssm_dt_bias,
        ssm_a,
        ssm_norm_weight,
        ssm_out,
        state_in,
        key_dim,
        value_dim,
        kv_heads,
        group,
        l_cache,
        output_gate,
    )?;
    Ok((mixer_out, taps.qkv_mixed, taps.state_out))
}

/// [`append_qwen35_ssm_mixer`]'s full implementation, returning every
/// [`SsmMixerTaps`] intermediate alongside `mixer_out` for a caller that
/// needs to bisect the tail (per-head gated RMSNorm, output gate, `ssm_out`
/// projection) against an independent reference.
#[allow(clippy::too_many_arguments)]
pub fn append_qwen35_ssm_mixer_with_taps(
    program: &mut Vec<Op>,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    head_eps: NodeId,
    one: NodeId,
    inv_sqrt_key_dim: NodeId,
    inv_head_v_dim: NodeId,
    attn_norm_weight: Option<NodeId>,
    wqkv: NodeId,
    wqkv_gate: NodeId,
    conv_weight: NodeId,
    conv_history_in: NodeId,
    ssm_beta: NodeId,
    ssm_alpha: NodeId,
    ssm_dt_bias: NodeId,
    ssm_a: NodeId,
    ssm_norm_weight: NodeId,
    ssm_out: NodeId,
    state_in: NodeId,
    key_dim: u32,
    value_dim: u32,
    kv_heads: u32,
    group: u32,
    l_cache: u32,
    output_gate: GdnOutputGate,
) -> Result<(NodeId, SsmMixerTaps), TensorError> {
    let head_k_dim = key_dim / kv_heads;
    let num_v_heads = kv_heads * group;
    let head_v_dim = value_dim / num_v_heads;

    let normed = match attn_norm_weight {
        Some(weight) => rmsnorm(program, x, weight, inv_dim, eps)?,
        None => x,
    };

    let qkv_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->siq"), (wqkv, "iq->siq")],
    )?;
    let qkv_mixed = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        qkv_product,
        "siq->siq",
        "sq->siq",
    )?;

    let z_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->siz"), (wqkv_gate, "iz->siz")],
    )?;
    let z = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        z_product,
        "siz->siz",
        "sz->siz",
    )?;
    let z_gated = match output_gate {
        GdnOutputGate::Silu => silu(program, z, one, "sz->sz")?,
        GdnOutputGate::Sigmoid => sigmoid(program, z, one, "sz->sz")?,
    };

    let beta_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->sin"), (ssm_beta, "in->sin")],
    )?;
    let beta_flat = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        beta_product,
        "sin->sin",
        "sn->sin",
    )?;
    let beta_sigmoid = sigmoid(program, beta_flat, one, "sn->sn")?;

    let alpha_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->sin"), (ssm_alpha, "in->sin")],
    )?;
    let alpha_flat = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        alpha_product,
        "sin->sin",
        "sn->sin",
    )?;
    let alpha_biased = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(alpha_flat, "sn->sn"), (ssm_dt_bias, "n->sn")],
    )?;
    let alpha_softplus = softplus(program, alpha_biased, one, "sn->sn")?;
    let gate_flat = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(alpha_softplus, "sn->sn"), (ssm_a, "n->sn")],
    )?;

    // cached causal conv: `weight`'s declared `[q, l_cache]` layout matches
    // `causal_conv1d`'s own (`l_cache` fastest/contiguous per channel) --
    // history (real axis `w`, no `s`) and this call's own new token (real
    // axis `s`) blend additively, no concat.
    let weight_history = channel_slice(program, conv_weight, l_cache, 0, l_cache - 1)?;
    let weight_new_wide = channel_slice(program, conv_weight, l_cache, l_cache - 1, 1)?;
    let weight_new = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        weight_new_wide,
        "qw->qw",
        "q->qw",
    )?;

    let history_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(conv_history_in, "wq->wq"), (weight_history, "qw->wq")],
    )?;
    let history_term = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        history_product,
        "wq->wq",
        "q->wq",
    )?;

    let new_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(qkv_mixed, "sq->sq"), (weight_new, "q->sq")],
    )?;
    let conv_raw = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(new_product, "sq->sq"), (history_term, "q->sq")],
    )?;

    let activated = silu(program, conv_raw, one, "sq->sq")?;
    let qkv_dim = 2 * key_dim + value_dim;
    let q_raw = channel_slice(program, activated, qkv_dim, 0, key_dim)?;
    let k_raw = channel_slice(program, activated, qkv_dim, key_dim, key_dim)?;
    let v_conv = channel_slice(program, activated, qkv_dim, 2 * key_dim, value_dim)?;

    let q_conv = l2norm(program, q_raw, eps, "sw->sw", "s->sw")?;
    let k_conv = l2norm(program, k_raw, eps, "sw->sw", "s->sw")?;

    // A read-side multi-term decomposition (`{coeff}*u+i`) constrains the
    // COMBINED axis, never `u`/`i` individually -- `shape::infer` cannot
    // solve one affine equation for two unknown extents, exactly the reason
    // [`repeat_kv_heads`]'s own `group_ones`/`ug->sugd` donor exists. Each
    // decomposition below pairs with the identical all-ones-donor technique,
    // scoped to whichever letters that decomposition introduces.
    let key_head_ones = op::append(
        program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(head_k_dim)],
            value: 1.0,
        },
    );
    let q_split_map = alloc::format!("s,{head_k_dim}*u+i->sui");
    let q_split = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(q_conv, q_split_map.as_str()), (key_head_ones, "ui->sui")],
    )?;
    let k_split = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(k_conv, q_split_map.as_str()), (key_head_ones, "ui->sui")],
    )?;

    let q_repeated = repeat_kv_heads(program, q_split, kv_heads, group)?;
    let k_repeated = repeat_kv_heads(program, k_split, kv_heads, group)?;

    let value_head_ones = op::append(
        program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(group), Extent::Static(head_v_dim)],
            value: 1.0,
        },
    );
    let v_split_map = alloc::format!("s,{}*u+{head_v_dim}*g+j->sugj", group * head_v_dim);
    let v_split = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(v_conv, v_split_map.as_str()), (value_head_ones, "ugj->sugj")],
    )?;

    let group_ones = op::append(
        program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
            value: 1.0,
        },
    );
    let head_split_map = alloc::format!("s,{group}*u+g->sug");
    let beta_split = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(beta_sigmoid, head_split_map.as_str()), (group_ones, "ug->sug")],
    )?;
    let gate_split = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(gate_flat, head_split_map.as_str()), (group_ones, "ug->sug")],
    )?;
    let z_split = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(z_gated, v_split_map.as_str()), (value_head_ones, "ugj->sugj")],
    )?;

    // squeeze the size-1 decode-step `s` axis away -- `append_qwen35_delta_net_step`
    // has no `s` letter at all (a single already-selected token per its own
    // doc), and reordering the surviving letters here (`dug`, not `ugd`)
    // doubles as the transpose `append_qwen35_delta_net_step`'s own
    // `i{head}`/`j{head}` maps expect.
    let query = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, q_repeated, "sugd->sugd", "dug->sugd")?;
    let key = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, k_repeated, "sugd->sugd", "dug->sugd")?;
    let value = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, v_split, "sugj->sugj", "jug->sugj")?;
    let beta = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, beta_split, "sug->sug", "ug->sug")?;
    let gate = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, gate_split, "sug->sug", "ug->sug")?;
    let z_head = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, z_split, "sugj->sugj", "ugj->sugj")?;

    let (delta_out, state_out) =
        append_qwen35_delta_net_step(program, query, key, value, gate, beta, state_in, inv_sqrt_key_dim, "ug")?;

    // gated RMSNorm over the per-head value axis `j`, `head_eps`/`inv_head_v_dim`
    // matched to the surviving `u,g` head space -- `build_norm_gated`
    // (`qwen35.cpp:243-250`): `rmsnorm(out, weight) * output_gate(z)`,
    // `output_gate` per [`GdnOutputGate`] (silu for qwen35, sigmoid for
    // qwen4exp, reference: PR 27742 line 2896-2899).
    let squared = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(delta_out, "jug->jug"), (delta_out, "jug->jug")])?;
    let sum_squares = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, squared, "jug->jug", "ug->jug")?;
    let mean_square = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(sum_squares, "ug->ug"), (inv_head_v_dim, "->ug")])?;
    let mean_square_eps = elementwise(program, DType::Float32, ScalarOp::Add, &[(mean_square, "ug->ug"), (head_eps, "ug->ug")])?;
    let rms = elementwise(program, DType::Float32, ScalarOp::SquareRoot, &[(mean_square_eps, "ug->ug")])?;
    let inv_rms = elementwise(program, DType::Float32, ScalarOp::Reciprocal, &[(rms, "ug->ug")])?;
    let normed_out = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(delta_out, "jug->jug"), (inv_rms, "ug->jug")])?;
    let normed_out_gamma = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed_out, "jug->jug"), (ssm_norm_weight, "j->jug")],
    )?;
    let gated_out = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed_out_gamma, "jug->jug"), (z_head, "ugj->jug")],
    )?;

    // output projection: `ssm_out`'s declared `[value_dim, n_embd]` layout
    // decomposed the same read-side way `q_split_map`/`v_split_map` already
    // decompose a flat checkpoint axis -- never a write-side merge.
    let out_weight_split_map = alloc::format!("{}*u+{head_v_dim}*g+j,d->ugjd", group * head_v_dim);
    let ssm_out_split = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (ssm_out, out_weight_split_map.as_str()),
            (value_head_ones, "ugj->ugjd"),
        ],
    )?;
    let cur_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(gated_out, "jug->jugd"), (ssm_out_split, "ugjd->jugd")],
    )?;
    let cur = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        cur_product,
        "jugd->jugd",
        "d->jugd",
    )?;

    let mixer_out = elementwise(program, DType::Float32, ScalarOp::Add, &[(x, "sd->sd"), (cur, "d->sd")])?;

    let taps = SsmMixerTaps {
        qkv_mixed,
        state_out,
        delta_out,
        z,
        gated_rmsnorm_out: normed_out_gamma,
        gated_value: gated_out,
        ssm_out_result: cur,
    };
    Ok((mixer_out, taps))
}

/// [`append_mistral_layer`]'s attention sub-block in isolation (RoPE + GQA +
/// causal mask + residual, no FFN) -- the piece [`lfm2_forward_program_with_experts`]
/// needs on its own, since an attention block there sits beside
/// [`append_lfm2_conv_mixer`] rather than always beside the same FFN choice
/// [`append_mistral_layer`] bundles it with. Node-for-node the same attention
/// graph [`append_mistral_layer`] runs before its own FFN call, extracted
/// rather than shared by refactoring that function, so the dense Mistral/Llama
/// path's own generated program bytes never change shape because this
/// function exists next to it.
#[allow(clippy::too_many_arguments)]
pub fn append_attention_mixer(
    program: &mut Vec<Op>,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    inv_sqrt_head_dim: NodeId,
    inv_head_dim: NodeId,
    cos: NodeId,
    sin: NodeId,
    group_ones: NodeId,
    is_future: NodeId,
    neg_infinity: NodeId,
    group: u32,
    attn_norm_weight: NodeId,
    q_norm_weight: NodeId,
    k_norm_weight: NodeId,
    wq: NodeId,
    wk: NodeId,
    wv: NodeId,
    wo: NodeId,
) -> Result<NodeId, TensorError> {
    let normed = rmsnorm(program, x, attn_norm_weight, inv_dim, eps)?;

    let q_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->shdi"), (wq, "ihd->shdi")],
    )?;
    let q_raw = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        q_product,
        "shdi->shdi",
        "shd->shdi",
    )?;
    // [`Lfm2MoeAttention.q_layernorm`]'s own placement
    // (`modeling_lfm2_moe.py:331`): normalizes right after the head
    // reshape, BEFORE `apply_rotary_pos_emb` -- never after.
    let q = rmsnorm_per_head(program, q_raw, q_norm_weight, inv_head_dim, eps, "h")?;

    let k_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->sudi"), (wk, "iud->sudi")],
    )?;
    let k_raw = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        k_product,
        "sudi->sudi",
        "sud->sudi",
    )?;
    let k = rmsnorm_per_head(program, k_raw, k_norm_weight, inv_head_dim, eps, "u")?;

    let v_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed, "si->sudi"), (wv, "iud->sudi")],
    )?;
    let v = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        v_product,
        "sudi->sudi",
        "sud->sudi",
    )?;

    let q_even_cos = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(q, "s,h,2*i->shi"), (cos, "si->shi")],
    )?;
    let q_odd_sin = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(q, "s,h,2*i+1->shi"), (sin, "si->shi")],
    )?;
    let rotated_q_even = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(q_even_cos, "shi->shi"), (q_odd_sin, "shi->shi")],
    )?;
    let q_even_sin = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(q, "s,h,2*i->shi"), (sin, "si->shi")],
    )?;
    let q_odd_cos = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(q, "s,h,2*i+1->shi"), (cos, "si->shi")],
    )?;
    let rotated_q_odd = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(q_even_sin, "shi->shi"), (q_odd_cos, "shi->shi")],
    )?;

    let k_even_cos = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(k, "s,u,2*i->sui"), (cos, "si->sui")],
    )?;
    let k_odd_sin = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(k, "s,u,2*i+1->sui"), (sin, "si->sui")],
    )?;
    let rotated_k_even = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(k_even_cos, "sui->sui"), (k_odd_sin, "sui->sui")],
    )?;
    let k_even_sin = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(k, "s,u,2*i->sui"), (sin, "si->sui")],
    )?;
    let k_odd_cos = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(k, "s,u,2*i+1->sui"), (cos, "si->sui")],
    )?;
    let rotated_k_odd = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(k_even_sin, "sui->sui"), (k_odd_cos, "sui->sui")],
    )?;

    let group_map = alloc::format!("s,{group}*u+g,i->sugi");
    let q_even_grouped = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (rotated_q_even, group_map.as_str()),
            (group_ones, "ug->sugi"),
        ],
    )?;
    let q_odd_grouped = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (rotated_q_odd, group_map.as_str()),
            (group_ones, "ug->sugi"),
        ],
    )?;

    let score_even_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (q_even_grouped, "sugi->stugi"),
            (rotated_k_even, "tui->stugi"),
        ],
    )?;
    let score_even = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_even_product,
        "stugi->stugi",
        "stug->stugi",
    )?;
    let score_odd_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (q_odd_grouped, "sugi->stugi"),
            (rotated_k_odd, "tui->stugi"),
        ],
    )?;
    let score_odd = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        score_odd_product,
        "stugi->stugi",
        "stug->stugi",
    )?;
    let scores = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(score_even, "stug->stug"), (score_odd, "stug->stug")],
    )?;

    let scores_scaled = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(scores, "stug->stug"), (inv_sqrt_head_dim, "->stug")],
    )?;

    let scores_masked = elementwise(
        program,
        DType::Float32,
        ScalarOp::Select,
        &[
            (is_future, "st->stug"),
            (neg_infinity, "->stug"),
            (scores_scaled, "stug->stug"),
        ],
    )?;

    let score_max = reduce(
        program,
        DType::Float32,
        ScalarOp::Maximum,
        ReduceInit::NegativeInfinity,
        scores_masked,
        "stug->stug",
        "sug->stug",
    )?;
    let shifted = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(scores_masked, "stug->stug"), (score_max, "sug->stug")],
    )?;
    let weights = elementwise(
        program,
        DType::Float32,
        ScalarOp::Exponential,
        &[(shifted, "stug->stug")],
    )?;
    let weight_sum = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        weights,
        "stug->stug",
        "sug->stug",
    )?;
    let inv_weight_sum = elementwise(
        program,
        DType::Float32,
        ScalarOp::Reciprocal,
        &[(weight_sum, "sug->sug")],
    )?;
    let probabilities = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(weights, "stug->stug"), (inv_weight_sum, "sug->stug")],
    )?;

    let attended_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(probabilities, "stug->stugd"), (v, "tud->stugd")],
    )?;
    let attended = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        attended_product,
        "stugd->stugd",
        "sugd->stugd",
    )?;

    let wo_product = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(attended, "sugd->sugdo"), (wo, "ugdo->sugdo")],
    )?;
    let attn_out = reduce(
        program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        wo_product,
        "sugdo->sugdo",
        "so->sugdo",
    )?;

    elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[(attn_out, "sd->sd"), (x, "sd->sd")],
    )
}

/// LFM2.5-8B-A1B's hybrid forward pass: `block_count` blocks, each either
/// `append_attention_mixer` or `append_lfm2_conv_mixer` per its own
/// `layer_kinds[layer]` (derived by [`LayerKind::from_tensor_names`] from the
/// real checkpoint's tensor directory, since `layer_types` is not a metadata
/// key this architecture writes), then a shared RMSNorm and
/// `append_moe_ffn`/dense-triple FFN exactly like
/// [`mistral_forward_program`]'s own MoE branch --
/// `leading_dense_block_count` (LFM2.5-8B-A1B: `2`) is threaded per layer
/// rather than a single crate-wide dense/MoE switch, since this checkpoint's
/// first two blocks are dense and the rest are routed.
///
/// Prefill-only: takes the whole prompt as one `[seq, embedding]` pass, the
/// same scope [`mistral_forward_program`] has. A KV-cached and
/// conv-state-cached incremental counterpart (mirroring
/// [`mistral_cached_forward_program_with_experts`]) is a further step this
/// function's own doc does not claim -- `causal_conv1d`'s masked-gather
/// composition only needs the whole sequence to be present at once, which a
/// one-token-at-a-time decode call does not have.
#[allow(clippy::too_many_arguments)]
pub fn lfm2_forward_program_with_experts(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    expert_feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    block_count: u32,
    expert_count: u32,
    expert_used_count: u32,
    leading_dense_block_count: u32,
    l_cache: u32,
    layer_kinds: &[LayerKind],
) -> Result<(Vec<Op>, NodeId, MoeSites), TensorError> {
    if layer_kinds.len() != block_count as usize {
        return Err(TensorError::LayerKindCountMismatch {
            expected: block_count,
            found: layer_kinds.len(),
        });
    }

    let group = query_heads / kv_heads;
    let pairs = head_dim / 2;

    let mut program = Vec::new();

    let ids = input_leaf(
        &mut program,
        DType::Int32,
        alloc::vec![Extent::Symbolic(0)],
        "ids",
    );
    let table = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(vocab), Extent::Static(embedding)],
        "token_embd.weight",
    );
    let mut x = embedding_lookup(&mut program, table, ids);

    let inv_dim = scalar_constant(&mut program, 1.0 / embedding as f32);
    let eps = symbolic_leaf(&mut program, DType::Float32, "eps");
    let ones = scalar_constant(&mut program, 1.0);
    let inv_sqrt_head_dim = scalar_constant(&mut program, 1.0 / (head_dim as f32).sqrt());
    let inv_head_dim = scalar_constant(&mut program, 1.0 / head_dim as f32);
    let cos = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(pairs)],
        "rope_cos",
    );
    let sin = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(pairs)],
        "rope_sin",
    );
    let group_ones = op::append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
            value: 1.0,
        },
    );
    let (is_future, neg_infinity) = causal_mask(&mut program)?;
    let mut moe_sites: Vec<MoeSite> = Vec::new();

    for (layer, kind) in layer_kinds.iter().enumerate() {
        let layer = layer as u32;
        let attn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.attn_norm.weight"),
        );
        let ffn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.ffn_norm.weight"),
        );

        let post_mixer = match kind {
            LayerKind::Attention => {
                let wq = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Static(embedding),
                        Extent::Static(query_heads),
                        Extent::Static(head_dim)
                    ],
                    &alloc::format!("blk.{layer}.attn_q.weight"),
                );
                let wk = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Static(embedding),
                        Extent::Static(kv_heads),
                        Extent::Static(head_dim)
                    ],
                    &alloc::format!("blk.{layer}.attn_k.weight"),
                );
                let wv = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Static(embedding),
                        Extent::Static(kv_heads),
                        Extent::Static(head_dim)
                    ],
                    &alloc::format!("blk.{layer}.attn_v.weight"),
                );
                let wo = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Static(kv_heads),
                        Extent::Static(group),
                        Extent::Static(head_dim),
                        Extent::Static(embedding),
                    ],
                    &alloc::format!("blk.{layer}.attn_output.weight"),
                );
                let q_norm_weight = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(head_dim)],
                    &alloc::format!("blk.{layer}.attn_q_norm.weight"),
                );
                let k_norm_weight = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(head_dim)],
                    &alloc::format!("blk.{layer}.attn_k_norm.weight"),
                );
                append_attention_mixer(
                    &mut program,
                    x,
                    inv_dim,
                    eps,
                    inv_sqrt_head_dim,
                    inv_head_dim,
                    cos,
                    sin,
                    group_ones,
                    is_future,
                    neg_infinity,
                    group,
                    attn_norm_weight,
                    q_norm_weight,
                    k_norm_weight,
                    wq,
                    wk,
                    wv,
                    wo,
                )?
            }
            LayerKind::ShortConv => {
                // `b_proj`/`c_proj`/`x_proj` are the real checkpoint's single
                // fused `blk.{layer}.shortconv.in_proj.weight`
                // (`[embedding, 3*embedding]`) split three ways -- see
                // `append_lfm2_conv_mixer`'s own doc for why this graph
                // cannot instead slice one fused `Input` by offset. Binding
                // these three names from that one on-disk tensor is a
                // binder-side split this session does not implement; the
                // names here are this program's contract for whoever does.
                let b_proj = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(embedding), Extent::Static(embedding)],
                    &alloc::format!("blk.{layer}.shortconv.in_proj.weight.b"),
                );
                let c_proj = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(embedding), Extent::Static(embedding)],
                    &alloc::format!("blk.{layer}.shortconv.in_proj.weight.c"),
                );
                let x_proj = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(embedding), Extent::Static(embedding)],
                    &alloc::format!("blk.{layer}.shortconv.in_proj.weight.x"),
                );
                // `[embedding, l_cache]`, NOT `[l_cache, embedding]` --
                // `causal_conv1d`'s own doc on its `dl->sld` map explains why:
                // the real on-disk tensor has `l_cache` as its fastest axis,
                // and `row_major_strides` (`bind.rs`) makes the LAST declared
                // shape axis the fastest one.
                let conv_weight = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(embedding), Extent::Static(l_cache)],
                    &alloc::format!("blk.{layer}.shortconv.conv.weight"),
                );
                let out_proj = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(embedding), Extent::Static(embedding)],
                    &alloc::format!("blk.{layer}.shortconv.out_proj.weight"),
                );
                append_lfm2_conv_mixer(
                    &mut program,
                    x,
                    inv_dim,
                    eps,
                    attn_norm_weight,
                    b_proj,
                    c_proj,
                    x_proj,
                    conv_weight,
                    out_proj,
                    l_cache,
                )?
            }
        };

        let normed2 = rmsnorm(&mut program, post_mixer, ffn_norm_weight, inv_dim, eps)?;

        let ffn_out = if layer < leading_dense_block_count {
            let w_gate = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(feed_forward)],
                &alloc::format!("blk.{layer}.ffn_gate.weight"),
            );
            let w_up = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(feed_forward)],
                &alloc::format!("blk.{layer}.ffn_up.weight"),
            );
            let w_down = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(feed_forward), Extent::Static(embedding)],
                &alloc::format!("blk.{layer}.ffn_down.weight"),
            );
            let gate_product = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(normed2, "sd->sdg"), (w_gate, "dg->sdg")],
            )?;
            let gate = reduce(
                &mut program,
                DType::Float32,
                ScalarOp::Add,
                ReduceInit::Zero,
                gate_product,
                "sdg->sdg",
                "sg->sdg",
            )?;
            let up_product = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(normed2, "sd->sdg"), (w_up, "dg->sdg")],
            )?;
            let up = reduce(
                &mut program,
                DType::Float32,
                ScalarOp::Add,
                ReduceInit::Zero,
                up_product,
                "sdg->sdg",
                "sg->sdg",
            )?;

            let neg_gate = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Negate,
                &[(gate, "sg->sg")],
            )?;
            let exp_neg_gate = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Exponential,
                &[(neg_gate, "sg->sg")],
            )?;
            let one_plus_exp = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Add,
                &[(exp_neg_gate, "sg->sg"), (ones, "->sg")],
            )?;
            let sigmoid_gate = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Reciprocal,
                &[(one_plus_exp, "sg->sg")],
            )?;
            let silu_gate = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(gate, "sg->sg"), (sigmoid_gate, "sg->sg")],
            )?;
            let ffn_hidden = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(silu_gate, "sg->sg"), (up, "sg->sg")],
            )?;

            let down_product = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(ffn_hidden, "sg->sgd"), (w_down, "gd->sgd")],
            )?;
            reduce(
                &mut program,
                DType::Float32,
                ScalarOp::Add,
                ReduceInit::Zero,
                down_product,
                "sgd->sgd",
                "sd->sgd",
            )?
        } else {
            let gate_inp = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(expert_count)],
                &alloc::format!("blk.{layer}.ffn_gate_inp.weight"),
            );
            let expert_w_gate = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(expert_count),
                    Extent::Static(embedding),
                    Extent::Static(expert_feed_forward)
                ],
                &alloc::format!("blk.{layer}.ffn_gate_exps.weight"),
            );
            let expert_w_up = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(expert_count),
                    Extent::Static(embedding),
                    Extent::Static(expert_feed_forward)
                ],
                &alloc::format!("blk.{layer}.ffn_up_exps.weight"),
            );
            let expert_w_down = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(expert_count),
                    Extent::Static(expert_feed_forward),
                    Extent::Static(embedding)
                ],
                &alloc::format!("blk.{layer}.ffn_down_exps.weight"),
            );
            let expert_bias = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(expert_count)],
                &alloc::format!("blk.{layer}.exp_probs_b.bias"),
            );
            let (ffn_out, site) = append_moe_ffn(
                &mut program,
                layer,
                normed2,
                gate_inp,
                expert_w_gate,
                expert_w_up,
                expert_w_down,
                expert_count,
                expert_used_count,
                ones,
                ExpertGatingFunc::Sigmoid,
                Some(expert_bias),
            )?;
            moe_sites.push(site);
            ffn_out
        };

        x = elementwise(
            &mut program,
            DType::Float32,
            ScalarOp::Add,
            &[(ffn_out, "sd->sd"), (post_mixer, "sd->sd")],
        )?;
    }

    let output_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding)],
        "output_norm.weight",
    );
    let normed_final = rmsnorm(&mut program, x, output_norm_weight, inv_dim, eps)?;

    let lm_head = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding), Extent::Static(vocab)],
        "output.weight",
    );
    let logits_product = elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed_final, "sd->sdv"), (lm_head, "dv->sdv")],
    )?;
    let logits = reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        logits_product,
        "sdv->sdv",
        "sv->sdv",
    )?;

    Ok((program, logits, MoeSites(moe_sites)))
}

/// [`mistral_forward_program`]'s key/value-cached counterpart: the same
/// architecture, but `ids`/`rope_cos`/`rope_sin` carry only the `new`
/// positions this call introduces (symbol 0), attention also draws on a
/// per-layer already-rotated key/value cache sized by symbol 1
/// (`kv_cache.{layer}.k_even`/`k_odd`/`v`, bound [`Op::Input`]s each layer's
/// own online-softmax attention combines with its freshly computed
/// key/value), and the returned roots are `(logits,
/// per_layer_cache_roots)` instead of one implicit last-node root, since a
/// caller now needs the per-layer [`CachedLayerRoots`] to grow its cache for
/// the next call. A caller passes `symbols = [new_positions, cached_len]`
/// to [`crate::shape::infer`]/[`crate::cpu::evaluate_quantized_named`], and
/// on the very first call binds every `kv_cache.*` name to a zero-length
/// buffer (`cached_len == 0`) -- the cached-block reduces both fold over an
/// empty range, which [`ReduceInit::Zero`]/[`ReduceInit::NegativeInfinity`]
/// already define as identity/`-inf`, so the first call degenerates to
/// plain causal self-attention over the whole prompt with no special case.
///
/// Dense-only: always binds `append_mistral_cached_layer`'s plain
/// `ffn_{gate,up,down}.weight` triple. Delegates to
/// [`mistral_cached_forward_program_with_experts`] with `expert_count = 0`,
/// `expert_used_count = 0` -- that function's own doc explains why those two
/// values select the identical dense program this function has always
/// built. Kept as its own entry point (rather than folding the two extra
/// parameters in here) because this signature already has real callers
/// outside this crate that a dense-only checkpoint never needs to pass an
/// expert config to.
#[allow(clippy::too_many_arguments)]
pub fn mistral_cached_forward_program(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    block_count: u32,
) -> Result<(Vec<Op>, NodeId, Vec<CachedLayerRoots>), TensorError> {
    mistral_cached_forward_program_with_experts(
        vocab,
        embedding,
        feed_forward,
        query_heads,
        kv_heads,
        head_dim,
        block_count,
        0,
        0,
        false,
        false,
        false,
    )
    .map(|(program, roots, cache_roots, _moe_sites)| (program, roots.logits, cache_roots))
}

/// [`mistral_cached_forward_program`]'s Qwen3 dense-attention counterpart:
/// the identical interleaved-RoPE cached layer, plus per-head QK-norm
/// (Qwen3's own `q_norm`/`k_norm`, `modeling_qwen3.py`'s `Qwen3Attention`)
/// applied to `q`/`k_new` before RoPE -- see
/// `append_mistral_cached_layer`'s `qk_norm` parameter doc for the exact
/// two ops this adds over the plain Mistral layer. Qwen3 has no
/// mixture-of-experts variant this crate has bound yet, so this takes no
/// `expert_count`/`expert_used_count`, the same dense-only shape
/// [`mistral_cached_forward_program`] itself uses.
#[allow(clippy::too_many_arguments)]
pub fn qwen3_cached_forward_program(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    block_count: u32,
) -> Result<(Vec<Op>, NodeId, Vec<CachedLayerRoots>), TensorError> {
    mistral_cached_forward_program_with_experts(
        vocab,
        embedding,
        feed_forward,
        query_heads,
        kv_heads,
        head_dim,
        block_count,
        0,
        0,
        true,
        false,
        false,
    )
    .map(|(program, roots, cache_roots, _moe_sites)| (program, roots.logits, cache_roots))
}

/// [`mistral_cached_forward_program`]'s mixture-of-experts-capable
/// counterpart, carrying the same `expert_count`/`expert_used_count`
/// parameters [`mistral_forward_program`] already takes. `expert_count == 0`
/// binds every layer through `append_mistral_cached_layer`'s plain
/// `ffn_{gate,up,down}.weight` triple, node-for-node the same program
/// [`mistral_cached_forward_program`] has always built, so a dense
/// checkpoint's generated program is unaffected by this function's
/// existence. `expert_count > 0` routes each layer through
/// `append_mistral_cached_moe_layer` instead, gathering one of
/// `expert_count` experts' weight slabs per token per `append_moe_ffn`'s
/// doc -- the same routed FFN [`mistral_forward_program`]'s own MoE branch
/// already runs, reused rather than reconstructed.
///
/// `paired_gate_up_reduce` is passed straight through to every dense layer's
/// `append_mistral_cached_layer` call (see that parameter's own doc) --
/// `false` at every call site in this crate today; a caller opts in only
/// once its loader has bound `blk.{layer}.ffn_gate_up.weight`
/// (`proxima-model-interop::bind::bind_matmul_weight_paired`). No effect on
/// the `expert_count > 0` branch (MoE's own gate/up weights are a separate
/// per-expert stack this flag does not touch).
///
/// `fused_qkv_reduce` is passed straight through to every dense layer's
/// `append_mistral_cached_layer` call (see that parameter's own doc) --
/// `false` at every call site in this crate today; a caller opts in only
/// once its loader has bound `blk.{layer}.attn_qkv.weight`
/// (`proxima-model-interop::bind::bind_matmul_weight_triple`). Requires
/// `qk_norm == false` (`append_mistral_cached_layer`'s own doc); no effect
/// on the `expert_count > 0` branch (attention projections are untouched by
/// which FFN branch runs).
#[allow(clippy::too_many_arguments)]
pub fn mistral_cached_forward_program_with_experts(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    block_count: u32,
    expert_count: u32,
    expert_used_count: u32,
    qk_norm: bool,
    paired_gate_up_reduce: bool,
    fused_qkv_reduce: bool,
) -> Result<(Vec<Op>, ForwardRoots, Vec<CachedLayerRoots>, MoeSites), TensorError> {
    let (program, roots, cache_roots, _layer_residuals, moe_sites) =
        mistral_cached_forward_program_with_experts_and_layer_taps(
            vocab,
            embedding,
            feed_forward,
            query_heads,
            kv_heads,
            head_dim,
            block_count,
            expert_count,
            expert_used_count,
            qk_norm,
            paired_gate_up_reduce,
            fused_qkv_reduce,
            false,
        )?;
    Ok((program, roots, cache_roots, moe_sites))
}

/// [`mistral_cached_forward_program_with_experts`]'s full implementation,
/// additionally returning one [`NodeId`] per layer -- the residual
/// (`x_next`, the post-MoE-add activation) each block hands the next layer,
/// in layer order, `block_count` entries. A caller bisecting a CPU-vs-Metal
/// divergence requests these as extra program outputs (48 x [seq, embedding]
/// floats for a 48-layer checkpoint, trivially small) to find the first
/// layer whose output disagrees, without materializing every intermediate
/// node in the graph as an output (the CPU evaluator keeps every requested
/// output's full lifetime alive, so requesting ALL nodes is the >130 GB
/// failure mode this narrower request set avoids).
///
/// `last_row_only` gates the vocab-projection matmul's own row count:
/// `true` slices the final-norm activation to its last row before
/// `output.weight` ever multiplies it (a host-supplied `lm_head_row`
/// `Op::Input`, gathered through the same [`IndexMap::Computed`] shape
/// [`embedding_lookup`] already proves correct -- see that leaf's own doc
/// at the call site below for why it is host-supplied rather than
/// in-graph-derived), so the matmul computes one row instead of
/// `new_count`. `false` (every existing caller today) reproduces the prior
/// per-row-logits program unchanged -- a caller genuinely needing every
/// new row's own logits (multi-token verification, prefill scoring,
/// logprobs) opts into that by passing `false`, not by this crate guessing
/// which one a caller wants.
#[allow(clippy::too_many_arguments)]
pub fn mistral_cached_forward_program_with_experts_and_layer_taps(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    block_count: u32,
    expert_count: u32,
    expert_used_count: u32,
    qk_norm: bool,
    paired_gate_up_reduce: bool,
    fused_qkv_reduce: bool,
    last_row_only: bool,
) -> Result<MistralMoeForwardProgramWithLayerTaps, TensorError> {
    let group = query_heads / kv_heads;
    let pairs = head_dim / 2;

    let mut program = Vec::new();

    let ids = input_leaf(
        &mut program,
        DType::Int32,
        alloc::vec![Extent::Symbolic(0)],
        "ids",
    );
    let table = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(vocab), Extent::Static(embedding)],
        "token_embd.weight",
    );
    let mut x = embedding_lookup(&mut program, table, ids);

    let inv_dim = scalar_constant(&mut program, 1.0 / embedding as f32);
    let eps = symbolic_leaf(&mut program, DType::Float32, "eps");
    let ones = scalar_constant(&mut program, 1.0);
    let inv_sqrt_head_dim = scalar_constant(&mut program, 1.0 / (head_dim as f32).sqrt());
    // only materialized when a layer actually consumes it (`qk_norm`), so a
    // dense checkpoint with no QK-norm keeps the identical node count this
    // function has always emitted.
    let inv_head_dim = qk_norm.then(|| scalar_constant(&mut program, 1.0 / head_dim as f32));
    let cos_new = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(pairs)],
        "rope_cos",
    );
    let sin_new = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(pairs)],
        "rope_sin",
    );
    let group_ones = op::append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
            value: 1.0,
        },
    );
    // Only ever consulted by `append_mistral_cached_layer`'s
    // `fused_qkv_reduce` branch (that parameter's own doc) -- built ONLY
    // when the flag is set, so `false` reproduces today's program
    // node-for-node (`cached_attention_rewrite_replaces_the_bound_attention_subgraph`'s
    // own literal bound-op-count fixture is the guard: it caught the
    // unconditional-`Op::Constant` version of this as a real +2 node
    // regression before this comment existed).
    let (head_shape_ones, kv_head_shape_ones) = if fused_qkv_reduce {
        let head_shape_ones = op::append(
            &mut program,
            Op::Constant {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(query_heads), Extent::Static(head_dim)],
                value: 1.0,
            },
        );
        let kv_head_shape_ones = op::append(
            &mut program,
            Op::Constant {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(head_dim)],
                value: 1.0,
            },
        );
        (head_shape_ones, kv_head_shape_ones)
    } else {
        // Never read (`append_mistral_cached_layer`'s `fused_qkv_reduce`
        // branch is the only reader, and it never runs when the flag is
        // `false`) -- `ones` (already built above) is reused as the
        // placeholder rather than adding an `Option` the callee would need
        // to `expect()` out of (this crate's own no-`expect`-in-production
        // rule), or building a real constant no `false` caller ever needs.
        (ones, ones)
    };
    let (is_future, _neg_infinity) = causal_mask(&mut program)?;
    // Rank-0 `Op::Input`, same precedent `eps`/`rope_cos`/`rope_sin` set
    // (`causal_mask_merged`'s own doc), named "cached_len" so
    // `proxima_tensor::bind::cached_attention_candidates` can find it by
    // name -- this crate's own precedent for what a name is for
    // (`Op::Input`'s own doc: "identity, not decoration"). It feeds no
    // arithmetic in this program: the host supplies the REAL `cached_len`
    // every call, independent of `kv_cache.{layer}.*`'s own
    // `Extent::Symbolic(1)` extent (which a caller may round up to a
    // bucket boundary, `ServingConfig::kv_bucket_tokens`, without
    // rebuilding this program), and the fused `BoundOpKind::CachedAttention`
    // reads it as a NINTH, runtime operand at execution time instead --
    // the bucket's own zero-padding is excluded by that bound, never by a
    // mask node in this graph (`BoundOpKind::CachedAttention`'s own doc on
    // the `cached_key_rows != 0` discriminator).
    let _cached_len = input_leaf(&mut program, DType::Float32, Vec::new(), "cached_len");

    let mut cache_roots: Vec<CachedLayerRoots> = Vec::with_capacity(block_count as usize);
    let mut layer_residuals: Vec<NodeId> = Vec::with_capacity(block_count as usize);
    let mut moe_sites: Vec<MoeSite> = Vec::new();

    for layer in 0..block_count {
        let attn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.attn_norm.weight"),
        );
        let ffn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.ffn_norm.weight"),
        );
        let (wq, wk, wv) = if fused_qkv_reduce {
            let rows = (query_heads + 2 * kv_heads) * head_dim;
            let w_qkv = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(rows), Extent::Static(embedding)],
                &alloc::format!("blk.{layer}.attn_qkv.weight"),
            );
            (w_qkv, w_qkv, w_qkv)
        } else {
            let wq = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(embedding),
                    Extent::Static(query_heads),
                    Extent::Static(head_dim)
                ],
                &alloc::format!("blk.{layer}.attn_q.weight"),
            );
            let wk = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(embedding),
                    Extent::Static(kv_heads),
                    Extent::Static(head_dim)
                ],
                &alloc::format!("blk.{layer}.attn_k.weight"),
            );
            let wv = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(embedding),
                    Extent::Static(kv_heads),
                    Extent::Static(head_dim)
                ],
                &alloc::format!("blk.{layer}.attn_v.weight"),
            );
            (wq, wk, wv)
        };
        let wo = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(kv_heads),
                Extent::Static(group),
                Extent::Static(head_dim),
                Extent::Static(embedding),
            ],
            &alloc::format!("blk.{layer}.attn_output.weight"),
        );
        let k_even_cache = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Symbolic(1),
                Extent::Static(kv_heads),
                Extent::Static(pairs)
            ],
            &alloc::format!("kv_cache.{layer}.k_even"),
        );
        let k_odd_cache = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Symbolic(1),
                Extent::Static(kv_heads),
                Extent::Static(pairs)
            ],
            &alloc::format!("kv_cache.{layer}.k_odd"),
        );
        let v_cache = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Symbolic(1),
                Extent::Static(kv_heads),
                Extent::Static(head_dim)
            ],
            &alloc::format!("kv_cache.{layer}.v"),
        );

        let (x_next, layer_roots) = if expert_count == 0 {
            let (w_gate, w_up) = if paired_gate_up_reduce {
                let w_gate_up = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![
                        Extent::Static(2),
                        Extent::Static(feed_forward),
                        Extent::Static(embedding)
                    ],
                    &alloc::format!("blk.{layer}.ffn_gate_up.weight"),
                );
                (w_gate_up, w_gate_up)
            } else {
                let w_gate = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(embedding), Extent::Static(feed_forward)],
                    &alloc::format!("blk.{layer}.ffn_gate.weight"),
                );
                let w_up = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(embedding), Extent::Static(feed_forward)],
                    &alloc::format!("blk.{layer}.ffn_up.weight"),
                );
                (w_gate, w_up)
            };
            let w_down = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(feed_forward), Extent::Static(embedding)],
                &alloc::format!("blk.{layer}.ffn_down.weight"),
            );
            let qk_norm_weights = inv_head_dim.map(|inv_head_dim| {
                let q_norm_weight = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(head_dim)],
                    &alloc::format!("blk.{layer}.attn_q_norm.weight"),
                );
                let k_norm_weight = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(head_dim)],
                    &alloc::format!("blk.{layer}.attn_k_norm.weight"),
                );
                (q_norm_weight, k_norm_weight, inv_head_dim)
            });

            append_mistral_cached_layer(
                &mut program,
                x,
                inv_dim,
                eps,
                ones,
                inv_sqrt_head_dim,
                cos_new,
                sin_new,
                group_ones,
                head_shape_ones,
                kv_head_shape_ones,
                is_future,
                group,
                head_dim,
                query_heads,
                attn_norm_weight,
                ffn_norm_weight,
                wq,
                wk,
                wv,
                wo,
                w_gate,
                w_up,
                w_down,
                k_even_cache,
                k_odd_cache,
                v_cache,
                qk_norm_weights,
                paired_gate_up_reduce,
                fused_qkv_reduce,
            )?
        } else {
            let gate_inp = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(expert_count)],
                &alloc::format!("blk.{layer}.ffn_gate_inp.weight"),
            );
            let expert_w_gate = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(expert_count),
                    Extent::Static(embedding),
                    Extent::Static(feed_forward)
                ],
                &alloc::format!("blk.{layer}.ffn_gate_exps.weight"),
            );
            let expert_w_up = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(expert_count),
                    Extent::Static(embedding),
                    Extent::Static(feed_forward)
                ],
                &alloc::format!("blk.{layer}.ffn_up_exps.weight"),
            );
            let expert_w_down = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(expert_count),
                    Extent::Static(feed_forward),
                    Extent::Static(embedding)
                ],
                &alloc::format!("blk.{layer}.ffn_down_exps.weight"),
            );
            let qk_norm_weights = inv_head_dim.map(|inv_head_dim| {
                let q_norm_weight = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(head_dim)],
                    &alloc::format!("blk.{layer}.attn_q_norm.weight"),
                );
                let k_norm_weight = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(head_dim)],
                    &alloc::format!("blk.{layer}.attn_k_norm.weight"),
                );
                (q_norm_weight, k_norm_weight, inv_head_dim)
            });

            let (next_x, next_roots, site) = append_mistral_cached_moe_layer(
                &mut program,
                layer,
                x,
                inv_dim,
                eps,
                ones,
                inv_sqrt_head_dim,
                cos_new,
                sin_new,
                group_ones,
                is_future,
                group,
                head_dim,
                attn_norm_weight,
                ffn_norm_weight,
                wq,
                wk,
                wv,
                wo,
                gate_inp,
                expert_w_gate,
                expert_w_up,
                expert_w_down,
                expert_count,
                expert_used_count,
                k_even_cache,
                k_odd_cache,
                v_cache,
                qk_norm_weights,
            )?;
            moe_sites.push(site);
            (next_x, next_roots)
        };
        x = x_next;
        cache_roots.push(layer_roots);
        layer_residuals.push(x_next);
    }

    let output_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding)],
        "output_norm.weight",
    );
    let normed_final = rmsnorm(&mut program, x, output_norm_weight, inv_dim, eps)?;

    // The decode loop only ever samples the LAST row's logits, prefill or
    // not (`proxima-model-interop::generate`'s own `logits[(new_count - 1)
    // * vocab_size..]` slice, every call site) -- greedy sampling needs one
    // row, never the whole prefill. Slicing here, before the vocab-sized
    // `output.weight` matmul, is what turns a 915-row Q6K reduce into a
    // 1-row one on an 850+-token prefill (`docs/discipline.md` ROW 418's
    // own `output.weight` measurement: 14.7s of 43.8s GPU time, 33% of
    // total, on the FULL 915-row projection). `embedding_lookup` is reused
    // verbatim, not a new primitive: it is already exactly `table[ids[s],
    // d]`, the same [`IndexMap::Computed`] gather this needs, just with a
    // 1-entry `lm_head_row` index instead of a `new_count`-entry `ids`.
    // `lm_head_row` is host-supplied (`new_count - 1`, same convention as
    // `cached_len`/`ids` above) rather than derived in-graph from
    // `Extent::Symbolic(0)`: `cpu.rs`'s own
    // `evaluate_typed_names_a_computed_gather_index_node_as_not_yet_supported`
    // test is this crate's own proof that an in-program-computed gather
    // index (an `Op::Iota`/`Op::Reduce` chain, not a caller-supplied
    // `Op::Input` block) is a named `NotLowerable` gap on the typed
    // evaluator, not a silently-guessed execution path -- a host-supplied
    // leaf is the one gather-index shape this crate's gather machinery
    // already proves correct end to end (`embedding_lookup`'s own `ids`).
    // `last_row_only: false` skips this leaf entirely (not merely bypasses
    // it) so the program a `false` caller gets is byte-for-byte the one
    // this function has always built -- no new node, no new required
    // binding, every existing per-position-logits caller unaffected.
    let normed_last = if last_row_only {
        let lm_head_row = input_leaf(
            &mut program,
            DType::Int32,
            alloc::vec![Extent::Static(1)],
            "lm_head_row",
        );
        embedding_lookup(&mut program, normed_final, lm_head_row)
    } else {
        normed_final
    };

    let lm_head = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding), Extent::Static(vocab)],
        "output.weight",
    );
    let logits_product = elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed_last, "sd->sdv"), (lm_head, "dv->sdv")],
    )?;
    let logits = reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        logits_product,
        "sdv->sdv",
        "sv->sdv",
    )?;

    Ok((
        program,
        ForwardRoots {
            logits,
            hidden: normed_last,
        },
        cache_roots,
        layer_residuals,
        MoeSites(moe_sites),
    ))
}

/// Per-layer roots [`qwen35_forward_program`]'s own caller threads back in
/// as next-call cache [`Op::Input`]s -- [`Qwen35DenseAttentionRoots`]'s own
/// 4-wide KV-cache shape for a dense-attention layer
/// (`append_qwen35_dense_attention_layer`'s own doc walks through why it
/// is 4-wide, not [`CachedLayerRoots`]'s 3), or `append_qwen35_ssm_mixer`'s
/// own `(qkv_mixed, state_out)` return for an SSM layer. A discriminated
/// enum, not a bool flag riding alongside a fixed-shape tuple: the layer
/// kinds carry genuinely different cache shapes, the same reason
/// [`LayerKind`] exists as its own type rather than a boolean.
///
/// `Attention(CachedLayerRoots)` is [`mistral_cached_forward_program_with_experts`]'s
/// own 3-wide shape, still constructed by that program's caller
/// (`crate::generate::LoadedModel::load`) for every non-qwen35 checkpoint --
/// kept as its own variant rather than folded into `DenseAttention` so that
/// caller's cache-threading loop, and its `LayerCache`, are unaffected by
/// this checkpoint's own partial-rotary gap.
#[derive(Debug, Clone, Copy)]
pub enum Qwen35LayerRoots {
    Attention(CachedLayerRoots),
    DenseAttention(Qwen35DenseAttentionRoots),
    Ssm { qkv_mixed: NodeId, state_out: NodeId },
}

/// Qwen3.5's whole-model incremental forward program: `full_attention_interval`
/// dense-attention layers (`append_mistral_cached_layer`, the same KV-cache
/// pattern [`mistral_cached_forward_program_with_experts`] already runs)
/// interleaved with gated-DeltaNet layers (`append_qwen35_ssm_mixer`),
/// following llama.cpp's own `hparams.is_recr_impl[i] = (i < n_layer) &&
/// ((i + 1) % full_attention_interval != 0)` (`qwen35.cpp:19-20`) -- layer
/// `full_attention_interval - 1`, `2 * full_attention_interval - 1`, ... are
/// dense attention, every other layer is SSM. Qwen3.5 never routes FFN
/// through experts (`qwen35.cpp:471`, `GGML_ASSERT(model.layers[il].ffn_gate_inp
/// == nullptr)`), so every layer's FFN is the plain dense triple
/// [`mistral_cached_forward_program_with_experts`]'s own `expert_count == 0`
/// branch already builds -- reused here rather than reconstructed.
///
/// `ssm_d_state`/`ssm_dt_rank`/`ssm_n_group`/`ssm_d_inner`/`ssm_d_conv` name
/// the same five hyperparameters `qwen35.cpp:335-343`'s own
/// `build_layer_attn_linear` reads off `hparams`, unpacked into
/// `append_qwen35_ssm_mixer`'s own `key_dim = ssm_d_state * ssm_n_group`,
/// `value_dim = ssm_d_inner`, `kv_heads = ssm_n_group`, `group = ssm_dt_rank
/// / ssm_n_group`, `l_cache = ssm_d_conv` (`head_v_dim = ssm_d_inner /
/// ssm_dt_rank` falls out inside the mixer itself, matching the oracle's own
/// `head_v_dim = d_inner / num_v_heads`). `rms_eps` is
/// `hparams.f_norm_rms_eps` baked as a graph-build-time constant, the same
/// choice this module already makes for `inv_dim`/`inv_sqrt_head_dim`
/// (Rust-side config values, not runtime-bound `Input`s) rather than a fresh
/// runtime-bound tensor shaped to `append_qwen35_ssm_mixer`'s own
/// `head_eps` (`[kv_heads, group]`) -- there is exactly one epsilon value
/// per checkpoint, known at program-build time.
///
/// Dense attention's own layers (`append_qwen35_dense_attention_layer`,
/// not `append_mistral_cached_layer`) run split-half RoPE over the
/// checkpoint's PARTIAL rotary width plus a concatenated-by-sum pass-through
/// remainder, and a per-head sigmoid gate on the attention output --
/// `append_qwen35_dense_attention_layer`'s own doc walks through why the
/// declared 3-section MRoPE (`rope.dimension_sections`) collapses to plain
/// single-section RoPE for this checkpoint's text-only forward program.
#[allow(clippy::too_many_arguments)]
pub fn qwen35_forward_program(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    attn_head_dim: u32,
    block_count: u32,
    full_attention_interval: u32,
    ssm_d_state: u32,
    ssm_dt_rank: u32,
    ssm_n_group: u32,
    ssm_d_inner: u32,
    ssm_d_conv: u32,
    rms_eps: f32,
) -> Result<(Vec<Op>, NodeId, Vec<Qwen35LayerRoots>), TensorError> {
    if full_attention_interval == 0 {
        return Err(TensorError::InvalidFullAttentionInterval { full_attention_interval });
    }

    let group = query_heads / kv_heads;
    let pairs = head_dim / 2;
    let ssm_group = ssm_dt_rank / ssm_n_group;
    let ssm_key_dim = ssm_d_state * ssm_n_group;

    let mut program = Vec::new();

    let ids = input_leaf(&mut program, DType::Int32, alloc::vec![Extent::Symbolic(0)], "ids");
    let table = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(vocab), Extent::Static(embedding)],
        "token_embd.weight",
    );
    let mut x = embedding_lookup(&mut program, table, ids);

    let inv_dim = scalar_constant(&mut program, 1.0 / embedding as f32);
    let eps = symbolic_leaf(&mut program, DType::Float32, "eps");
    let ones = scalar_constant(&mut program, 1.0);
    let one = ones;
    // `head_dim` here is `rope.dimension_count` -- this checkpoint's
    // PARTIAL-rotary width (`rotary_dim`), never the real per-head width.
    // Dense attention's own score scale is `attn_head_dim`-based
    // (`self.scaling = self.head_dim**-0.5` where `self.head_dim` is the
    // real width, `modeling_qwen3_next.py:262,264`), not
    // `rotary_dim`-based.
    let inv_sqrt_attn_head_dim = scalar_constant(&mut program, 1.0 / (attn_head_dim as f32).sqrt());
    let inv_attn_head_dim = scalar_constant(&mut program, 1.0 / attn_head_dim as f32);
    let inv_sqrt_key_dim = scalar_constant(&mut program, 1.0 / (ssm_d_state as f32).sqrt());
    let head_v_dim = ssm_d_inner / ssm_dt_rank;
    let inv_head_v_dim = scalar_constant(&mut program, 1.0 / head_v_dim as f32);
    let cos_new = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(pairs)],
        "rope_cos",
    );
    let sin_new = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Symbolic(0), Extent::Static(pairs)],
        "rope_sin",
    );
    let group_ones = op::append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
            value: 1.0,
        },
    );
    let head_eps = op::append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(ssm_n_group), Extent::Static(ssm_group)],
            value: rms_eps,
        },
    );
    let (is_future, _neg_infinity) = causal_mask(&mut program)?;
    // Same rank-0 leaf [`mistral_cached_forward_program_with_experts`] adds
    // right after its own `causal_mask` call, and for the same reason: named
    // "cached_len" so `bind::cached_attention_candidates`'s `find_named_input`
    // picks it up by NAME on the `Attention` arm's fused `CachedAttention`
    // op. The `DenseAttention` arm has no equivalent fusion, so this same
    // node is ALSO threaded directly into every
    // [`append_qwen35_dense_attention_layer`] call below to mask its own
    // padded cached range (that function's own doc).
    let cached_len = input_leaf(&mut program, DType::Float32, Vec::new(), "cached_len");

    let mut layer_roots: Vec<Qwen35LayerRoots> = Vec::with_capacity(block_count as usize);

    for layer in 0..block_count {
        let attn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.attn_norm.weight"),
        );
        // Named `post_attention_norm.weight` on disk, not `ffn_norm.weight`
        // -- this checkpoint's own GGUF writer names this tensor
        // differently from every other architecture this crate binds
        // (`proxima_model_interop::qwen35`'s own module doc, confirmed via
        // `strings` on the real file: no `blk.N.ffn_norm.weight` key
        // exists anywhere), on both layer kinds.
        let ffn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            &alloc::format!("blk.{layer}.post_attention_norm.weight"),
        );

        // `hparams.is_recr_impl[i] = (i + 1) % full_attention_interval != 0`
        // (`qwen35.cpp:19-20`) is TRUE for SSM layers -- dense attention is
        // its negation, `(i + 1) % full_attention_interval == 0`.
        let is_dense_attention = (layer + 1) % full_attention_interval == 0;

        let (x_next, roots) = if is_dense_attention {
            // real per-head width read off metadata (`attention.key_length`,
            // `attn_head_dim` param) rather than `embedding / query_heads`
            // -- the latter is not even an integer on the 27B checkpoint
            // (`5120 / 24 = 213.33`), confirmed wrong against the real file
            // by [`crate::qwen35::qwen35_architecture_from_metadata`]'s own
            // caller-side doc.
            let wq_flat = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(embedding),
                    Extent::Static(query_heads * attn_head_dim * 2)
                ],
                &alloc::format!("blk.{layer}.attn_q.weight"),
            );
            let wq = per_head_channel_slice(
                &mut program,
                wq_flat,
                query_heads,
                attn_head_dim * 2,
                0,
                attn_head_dim,
            )?;
            let w_gate_q = per_head_channel_slice(
                &mut program,
                wq_flat,
                query_heads,
                attn_head_dim * 2,
                attn_head_dim,
                attn_head_dim,
            )?;
            let wk_flat = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(kv_heads * attn_head_dim)],
                &alloc::format!("blk.{layer}.attn_k.weight"),
            );
            // `k` carries no gate and no partial-rotary truncation at the
            // weight level (the split into rotated/pass halves happens on
            // the ACTIVATION inside [`append_qwen35_dense_attention_layer`]
            // now that `q_norm`/`k_norm` need the full width first) -- the
            // same lossless-reshape donor trick `v`/`o` already use below.
            let k_head_ones = op::append(
                &mut program,
                Op::Constant {
                    dtype: DType::Float32,
                    shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(attn_head_dim)],
                    value: 1.0,
                },
            );
            let wk = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[
                    (wk_flat, alloc::format!("i,{attn_head_dim}*u+d->iud").as_str()),
                    (k_head_ones, "ud->iud"),
                ],
            )?;
            let v_head_ones = op::append(
                &mut program,
                Op::Constant {
                    dtype: DType::Float32,
                    shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(attn_head_dim)],
                    value: 1.0,
                },
            );
            let o_head_ones = op::append(
                &mut program,
                Op::Constant {
                    dtype: DType::Float32,
                    shape: alloc::vec![
                        Extent::Static(kv_heads),
                        Extent::Static(group),
                        Extent::Static(attn_head_dim)
                    ],
                    value: 1.0,
                },
            );
            let wv_flat = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(kv_heads * attn_head_dim)],
                &alloc::format!("blk.{layer}.attn_v.weight"),
            );
            let wv = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[
                    (wv_flat, alloc::format!("i,{attn_head_dim}*u+d->iud").as_str()),
                    (v_head_ones, "ud->iud"),
                ],
            )?;
            let wo_flat = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(query_heads * attn_head_dim),
                    Extent::Static(embedding)
                ],
                &alloc::format!("blk.{layer}.attn_output.weight"),
            );
            let wo = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[
                    (
                        wo_flat,
                        alloc::format!(
                            "{}*u+{attn_head_dim}*g+d,e->ugde",
                            attn_head_dim * group
                        )
                        .as_str(),
                    ),
                    (o_head_ones, "ugd->ugde"),
                ],
            )?;
            let w_gate = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(feed_forward)],
                &alloc::format!("blk.{layer}.ffn_gate.weight"),
            );
            let w_up = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(feed_forward)],
                &alloc::format!("blk.{layer}.ffn_up.weight"),
            );
            let w_down = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(feed_forward), Extent::Static(embedding)],
                &alloc::format!("blk.{layer}.ffn_down.weight"),
            );
            let q_norm_weight = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(attn_head_dim)],
                &alloc::format!("blk.{layer}.attn_q_norm.weight"),
            );
            let k_norm_weight = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(attn_head_dim)],
                &alloc::format!("blk.{layer}.attn_k_norm.weight"),
            );
            let pass_dim = attn_head_dim - head_dim;
            let k_first_cache = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Symbolic(1),
                    Extent::Static(kv_heads),
                    Extent::Static(pairs)
                ],
                &alloc::format!("kv_cache.{layer}.k_first"),
            );
            let k_second_cache = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Symbolic(1),
                    Extent::Static(kv_heads),
                    Extent::Static(pairs)
                ],
                &alloc::format!("kv_cache.{layer}.k_second"),
            );
            let k_pass_cache = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Symbolic(1),
                    Extent::Static(kv_heads),
                    Extent::Static(pass_dim)
                ],
                &alloc::format!("kv_cache.{layer}.k_pass"),
            );
            let v_cache = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Symbolic(1),
                    Extent::Static(kv_heads),
                    Extent::Static(attn_head_dim)
                ],
                &alloc::format!("kv_cache.{layer}.v"),
            );

            let (x_next, dense_attention_roots) = append_qwen35_dense_attention_layer(
                &mut program,
                x,
                inv_dim,
                eps,
                ones,
                inv_sqrt_attn_head_dim,
                inv_attn_head_dim,
                cos_new,
                sin_new,
                group_ones,
                is_future,
                cached_len,
                group,
                head_dim,
                attn_head_dim,
                attn_norm_weight,
                ffn_norm_weight,
                q_norm_weight,
                k_norm_weight,
                wq,
                w_gate_q,
                wk,
                wv,
                wo,
                w_gate,
                w_up,
                w_down,
                k_first_cache,
                k_second_cache,
                k_pass_cache,
                v_cache,
            )?;
            (x_next, Qwen35LayerRoots::DenseAttention(dense_attention_roots))
        } else {
            let qkv_dim = 2 * ssm_key_dim + ssm_d_inner;
            let wqkv = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(qkv_dim)],
                &alloc::format!("blk.{layer}.ssm_in.weight"),
            );
            let wqkv_gate = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(ssm_d_inner)],
                &alloc::format!("blk.{layer}.ssm_gate.weight"),
            );
            let conv_weight = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(qkv_dim), Extent::Static(ssm_d_conv)],
                &alloc::format!("blk.{layer}.ssm_conv1d.weight"),
            );
            let conv_history_in = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(ssm_d_conv - 1), Extent::Static(qkv_dim)],
                &alloc::format!("ssm_cache.{layer}.conv_history"),
            );
            let ssm_beta = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(ssm_dt_rank)],
                &alloc::format!("blk.{layer}.ssm_beta.weight"),
            );
            let ssm_alpha = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(ssm_dt_rank)],
                &alloc::format!("blk.{layer}.ssm_alpha.weight"),
            );
            let ssm_dt_bias = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(ssm_dt_rank)],
                &alloc::format!("blk.{layer}.ssm_dt.bias"),
            );
            let ssm_a = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(ssm_dt_rank)],
                &alloc::format!("blk.{layer}.ssm_a"),
            );
            let ssm_norm_weight = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(head_v_dim)],
                &alloc::format!("blk.{layer}.ssm_norm.weight"),
            );
            let ssm_out = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(ssm_d_inner), Extent::Static(embedding)],
                &alloc::format!("blk.{layer}.ssm_out.weight"),
            );
            let state_in = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![
                    Extent::Static(ssm_d_state),
                    Extent::Static(head_v_dim),
                    Extent::Static(ssm_n_group),
                    Extent::Static(ssm_group)
                ],
                &alloc::format!("ssm_cache.{layer}.state"),
            );

            let (mixer_out, qkv_mixed, state_out) = append_qwen35_ssm_mixer(
                &mut program,
                x,
                inv_dim,
                eps,
                head_eps,
                one,
                inv_sqrt_key_dim,
                inv_head_v_dim,
                Some(attn_norm_weight),
                wqkv,
                wqkv_gate,
                conv_weight,
                conv_history_in,
                ssm_beta,
                ssm_alpha,
                ssm_dt_bias,
                ssm_a,
                ssm_norm_weight,
                ssm_out,
                state_in,
                ssm_key_dim,
                ssm_d_inner,
                ssm_n_group,
                ssm_group,
                ssm_d_conv,
                GdnOutputGate::Silu,
            )?;

            // Unlike `append_mistral_cached_layer` (bundles FFN internally),
            // `append_qwen35_ssm_mixer` is mixer-plus-residual only -- the
            // same scope `append_lfm2_conv_mixer` has -- so the SSM branch
            // runs its own dense FFN pass here, matching
            // `mistral_cached_forward_program_with_experts`'s own
            // `expert_count == 0` FFN math exactly (Qwen3.5 never routes FFN
            // through experts, `qwen35.cpp:471`).
            let normed2 = rmsnorm(&mut program, mixer_out, ffn_norm_weight, inv_dim, eps)?;
            let w_gate = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(feed_forward)],
                &alloc::format!("blk.{layer}.ffn_gate.weight"),
            );
            let w_up = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(embedding), Extent::Static(feed_forward)],
                &alloc::format!("blk.{layer}.ffn_up.weight"),
            );
            let w_down = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(feed_forward), Extent::Static(embedding)],
                &alloc::format!("blk.{layer}.ffn_down.weight"),
            );
            let gate_product = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(normed2, "sd->sdg"), (w_gate, "dg->sdg")],
            )?;
            let gate = reduce(
                &mut program,
                DType::Float32,
                ScalarOp::Add,
                ReduceInit::Zero,
                gate_product,
                "sdg->sdg",
                "sg->sdg",
            )?;
            let up_product = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(normed2, "sd->sdg"), (w_up, "dg->sdg")],
            )?;
            let up = reduce(
                &mut program,
                DType::Float32,
                ScalarOp::Add,
                ReduceInit::Zero,
                up_product,
                "sdg->sdg",
                "sg->sdg",
            )?;
            let silu_gate = silu(&mut program, gate, one, "sg->sg")?;
            let ffn_hidden = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(silu_gate, "sg->sg"), (up, "sg->sg")],
            )?;
            let down_product = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(ffn_hidden, "sg->sgd"), (w_down, "gd->sgd")],
            )?;
            let ffn_out = reduce(
                &mut program,
                DType::Float32,
                ScalarOp::Add,
                ReduceInit::Zero,
                down_product,
                "sgd->sgd",
                "sd->sgd",
            )?;
            let x_after_ffn = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Add,
                &[(ffn_out, "sd->sd"), (mixer_out, "sd->sd")],
            )?;

            (x_after_ffn, Qwen35LayerRoots::Ssm { qkv_mixed, state_out })
        };

        x = x_next;
        layer_roots.push(roots);
    }

    let output_norm_weight = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding)],
        "output_norm.weight",
    );
    let normed_final = rmsnorm(&mut program, x, output_norm_weight, inv_dim, eps)?;

    let lm_head = input_leaf(
        &mut program,
        DType::Float32,
        alloc::vec![Extent::Static(embedding), Extent::Static(vocab)],
        "output.weight",
    );
    let logits_product = elementwise(
        &mut program,
        DType::Float32,
        ScalarOp::Multiply,
        &[(normed_final, "sd->sdv"), (lm_head, "dv->sdv")],
    )?;
    let logits = reduce(
        &mut program,
        DType::Float32,
        ScalarOp::Add,
        ReduceInit::Zero,
        logits_product,
        "sdv->sdv",
        "sv->sdv",
    )?;

    Ok((program, logits, layer_roots))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Proves the per-layer builders this module exports as `pub` are
    /// actually SUFFICIENT to build a forward program from outside this
    /// crate -- composes `embedding_lookup` -> [`append_hyper_connection_mix`]
    /// -> [`append_qwen35_ssm_mixer`] (`GdnOutputGate::Sigmoid`, the
    /// qwen4exp gate) -> [`append_hyper_connection_combine`] -> a final
    /// output mixer (`w_inject: None`, mirroring the doc's own
    /// "the final hyper-connection mixer carries [output_norm]") -> the
    /// same `rmsnorm` + multiply + reduce lm-head chain
    /// [`qwen35_forward_program`] ends every program with. Every dimension
    /// is the smallest non-degenerate size that keeps every builder's own
    /// einsum maps distinct (`embedding = 1` matches
    /// `build_ssm_mixer_test_program`'s own convention below); the
    /// assertion is finiteness and shape, not a numeric reference, since
    /// this test's job is proving the public surface COMPOSES, not
    /// re-proving any one builder's own arithmetic (each builder already
    /// has its own f64-reference test for that).
    #[test]
    fn public_builders_compose_a_one_layer_forward_program() {
        let tokens = 2usize;
        let vocab = 3u32;
        let embedding = 1u32;
        let hc = 2u32;
        let low_rank = 1u32;

        let mut program = Vec::new();

        let ids = input_leaf(&mut program, DType::Int32, alloc::vec![Extent::Symbolic(0)], "ids");
        let table = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(vocab), Extent::Static(embedding)],
            "token_embd.weight",
        );
        let embedded = embedding_lookup(&mut program, table, ids);

        let inv_dim = scalar_constant(&mut program, 1.0 / embedding as f32);
        // `eps` carries its own `s` (token) axis throughout this module's
        // rmsnorm-family ops (`rmsnorm`'s own `(eps, "s->s")`,
        // `append_hyper_connection_mix`'s `(eps, "s->sh")`) -- a per-token
        // leaf, never a rank-0 constant, matching every other builder's own
        // test fixture (`assert_mix_matches_reference`'s own `eps_data`).
        let eps = symbolic_leaf(&mut program, DType::Float32, "eps");
        let inv_hc = scalar_constant(&mut program, 1.0 / hc as f32);
        let one = scalar_constant(&mut program, 1.0);
        let two = scalar_constant(&mut program, 2.0);

        // Broadcast the `[tokens, embedding]` embedding lookup into the
        // `[tokens, hc, embedding]` hyper-connection residual every stream
        // starts identical at layer 0. Neither `embedded` (`si`, no `h`)
        // nor a rank-0 scalar owns the `h` axis, so shape inference cannot
        // size it from either alone -- `hc_ones`, a real `[hc]`-shaped
        // donor, is the same all-ones-donor idiom `group_ones`/`key_head_ones`
        // already use to constrain a read-side axis no other operand names.
        let hc_ones = op::append(
            &mut program,
            Op::Constant {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(hc)],
                value: 1.0,
            },
        );
        let residual = elementwise(
            &mut program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(embedded, "si->shi"), (hc_ones, "h->shi")],
        )
        .expect("broadcast into hc streams lowers");

        let w_norm = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(hc), Extent::Static(embedding)], "w_norm");
        let w_down = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(hc), Extent::Static(embedding), Extent::Static(low_rank)],
            "w_down",
        );
        let w_up = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(low_rank), Extent::Static(hc), Extent::Static(embedding)],
            "w_up",
        );
        let w_inject = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(hc), Extent::Static(embedding), Extent::Static(hc)],
            "w_inject",
        );

        let (mixed, inject) = append_hyper_connection_mix(&mut program, residual, inv_dim, eps, inv_hc, one, w_norm, w_down, w_up, Some(w_inject))
            .expect("hyper-connection mix lowers");
        let inject = inject.expect("w_inject was Some, so inject must be Some");

        let key_dim = 1u32;
        let value_dim = 2u32;
        let kv_heads = 1u32;
        let group = 2u32;
        let l_cache = 2u32;
        let qkv_dim = 2 * key_dim + value_dim;

        let head_eps = op::append(
            &mut program,
            Op::Constant {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
                value: 1e-6,
            },
        );
        let inv_sqrt_key_dim = scalar_constant(&mut program, 1.0);
        let inv_head_v_dim = scalar_constant(&mut program, 1.0);
        let attn_norm_weight = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(embedding)], "attn_norm_weight");
        let wqkv = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(embedding), Extent::Static(qkv_dim)], "wqkv");
        let wqkv_gate = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(embedding), Extent::Static(value_dim)], "wqkv_gate");
        let conv_weight = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(qkv_dim), Extent::Static(l_cache)], "conv_weight");
        let conv_history_in = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(l_cache - 1), Extent::Static(qkv_dim)],
            "conv_history_in",
        );
        let ssm_beta = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(embedding), Extent::Static(kv_heads * group)], "ssm_beta");
        let ssm_alpha = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(embedding), Extent::Static(kv_heads * group)], "ssm_alpha");
        let ssm_dt_bias = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(kv_heads * group)], "ssm_dt_bias");
        let ssm_a = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(kv_heads * group)], "ssm_a");
        let ssm_norm_weight = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(embedding)], "ssm_norm_weight");
        let ssm_out = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(value_dim), Extent::Static(embedding)], "ssm_out");
        let state_in = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(1),
                Extent::Static(1),
                Extent::Static(kv_heads),
                Extent::Static(group)
            ],
            "state_in",
        );

        let (block_out, _qkv_mixed, _state_out) = append_qwen35_ssm_mixer(
            &mut program,
            mixed,
            inv_dim,
            eps,
            head_eps,
            one,
            inv_sqrt_key_dim,
            inv_head_v_dim,
            Some(attn_norm_weight),
            wqkv,
            wqkv_gate,
            conv_weight,
            conv_history_in,
            ssm_beta,
            ssm_alpha,
            ssm_dt_bias,
            ssm_a,
            ssm_norm_weight,
            ssm_out,
            state_in,
            key_dim,
            value_dim,
            kv_heads,
            group,
            l_cache,
            GdnOutputGate::Sigmoid,
        )
        .expect("qwen4exp's own output gate lowers through the shared ssm mixer builder");

        let residual = append_hyper_connection_combine(&mut program, residual, block_out, inject, inv_hc, one, two)
            .expect("hyper-connection combine lowers");

        let final_w_norm = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(hc), Extent::Static(embedding)], "final_w_norm");
        let final_w_down = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(hc), Extent::Static(embedding), Extent::Static(low_rank)],
            "final_w_down",
        );
        let final_w_up = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(low_rank), Extent::Static(hc), Extent::Static(embedding)],
            "final_w_up",
        );
        let (final_mixed, no_inject) = append_hyper_connection_mix(&mut program, residual, inv_dim, eps, inv_hc, one, final_w_norm, final_w_down, final_w_up, None)
            .expect("final output mixer lowers");
        assert!(no_inject.is_none(), "the final output mixer must pass w_inject: None");

        // The same rmsnorm + multiply + reduce chain `qwen35_forward_program`
        // ends every program with.
        let output_norm_weight = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(embedding)], "output_norm.weight");
        let normed_final = rmsnorm(&mut program, final_mixed, output_norm_weight, inv_dim, eps).expect("final rmsnorm lowers");
        let lm_head_weight = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(embedding), Extent::Static(vocab)], "output.weight");
        let logits_product = elementwise(
            &mut program,
            DType::Float32,
            ScalarOp::Multiply,
            &[(normed_final, "sd->sdv"), (lm_head_weight, "dv->sdv")],
        )
        .expect("lm_head product lowers");
        let logits = reduce(
            &mut program,
            DType::Float32,
            ScalarOp::Add,
            ReduceInit::Zero,
            logits_product,
            "sdv->sdv",
            "sv->sdv",
        )
        .expect("lm_head reduce lowers");

        let mut state = 0x1234_5678_9abc_def0u64;
        let mut filled_input = |len: usize| -> Vec<f32> {
            (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
                })
                .collect()
        };

        // `evaluate_named` binds every input as an `&[f32]` regardless of
        // the `Op::Input::dtype` it was declared with -- `ids`'s own
        // `DType::Int32` only documents intent, since the gather this
        // module's `embedding_lookup` builds reads its index operand
        // through the same f32 buffer as every other input (the real
        // forward-program tests above bind their own `ids` this same way,
        // as `ids_f32`).
        let ids_data: Vec<f32> = (0..tokens as i32).map(|token| (token % vocab as i32) as f32).collect();
        let eps_data = alloc::vec![1e-6f32; tokens];
        let table_data = filled_input(vocab as usize * embedding as usize);
        let w_norm_data = filled_input(hc as usize * embedding as usize);
        let w_down_data = filled_input(hc as usize * embedding as usize * low_rank as usize);
        let w_up_data = filled_input(low_rank as usize * hc as usize * embedding as usize);
        let w_inject_data = filled_input(hc as usize * embedding as usize * hc as usize);
        let attn_norm_data = filled_input(embedding as usize);
        let wqkv_data = filled_input(embedding as usize * qkv_dim as usize);
        let wqkv_gate_data = filled_input(embedding as usize * value_dim as usize);
        let conv_weight_data = filled_input(qkv_dim as usize * l_cache as usize);
        let conv_history_data = filled_input((l_cache as usize - 1) * qkv_dim as usize);
        let ssm_beta_data = filled_input(embedding as usize * (kv_heads * group) as usize);
        let ssm_alpha_data = filled_input(embedding as usize * (kv_heads * group) as usize);
        let ssm_dt_bias_data = filled_input((kv_heads * group) as usize);
        let ssm_a_data = filled_input((kv_heads * group) as usize);
        let ssm_norm_data = filled_input(embedding as usize);
        let ssm_out_data = filled_input(value_dim as usize * embedding as usize);
        let state_in_data = filled_input(kv_heads as usize * group as usize);
        let final_w_norm_data = filled_input(hc as usize * embedding as usize);
        let final_w_down_data = filled_input(hc as usize * embedding as usize * low_rank as usize);
        let final_w_up_data = filled_input(low_rank as usize * hc as usize * embedding as usize);
        let output_norm_data = filled_input(embedding as usize);
        let lm_head_data = filled_input(embedding as usize * vocab as usize);

        let named: Vec<(&str, &[f32])> = alloc::vec![
            ("ids", ids_data.as_slice()),
            ("eps", eps_data.as_slice()),
            ("token_embd.weight", table_data.as_slice()),
            ("w_norm", w_norm_data.as_slice()),
            ("w_down", w_down_data.as_slice()),
            ("w_up", w_up_data.as_slice()),
            ("w_inject", w_inject_data.as_slice()),
            ("attn_norm_weight", attn_norm_data.as_slice()),
            ("wqkv", wqkv_data.as_slice()),
            ("wqkv_gate", wqkv_gate_data.as_slice()),
            ("conv_weight", conv_weight_data.as_slice()),
            ("conv_history_in", conv_history_data.as_slice()),
            ("ssm_beta", ssm_beta_data.as_slice()),
            ("ssm_alpha", ssm_alpha_data.as_slice()),
            ("ssm_dt_bias", ssm_dt_bias_data.as_slice()),
            ("ssm_a", ssm_a_data.as_slice()),
            ("ssm_norm_weight", ssm_norm_data.as_slice()),
            ("ssm_out", ssm_out_data.as_slice()),
            ("state_in", state_in_data.as_slice()),
            ("final_w_norm", final_w_norm_data.as_slice()),
            ("final_w_down", final_w_down_data.as_slice()),
            ("final_w_up", final_w_up_data.as_slice()),
            ("output_norm.weight", output_norm_data.as_slice()),
            ("output.weight", lm_head_data.as_slice()),
        ];

        let evaluated = crate::cpu::evaluate_named(&program, &[tokens as u64], &named, &[logits]).expect("the composed program evaluates on cpu");
        let (logits_values, logits_shape) = evaluated.get(logits).expect("logits output present");

        assert_eq!(
            logits_shape,
            [tokens as u64, vocab as u64],
            "logits must be [tokens, vocab] -- the same shape qwen35_forward_program's own lm_head produces"
        );
        assert!(
            logits_values.iter().all(|value| value.is_finite()),
            "every logit must be finite: {logits_values:?}"
        );
    }

    /// Regression proof for the qk-norm-dropped-on-MoE-layers bug fixed
    /// alongside this test: before the fix, `append_mistral_cached_moe_layer`
    /// took no `qk_norm` parameter at all, so `mistral_cached_forward_program_with_experts`
    /// silently discarded the `qk_norm` argument for every routed (MoE)
    /// layer -- flipping it produced the byte-identical program. A
    /// Qwen3-MoE-shaped checkpoint (`expert_count > 0`) carries
    /// `attn_q_norm.weight`/`attn_k_norm.weight` on every layer, so
    /// `qk_norm=true` must now append the extra per-head rmsnorm reduces
    /// (and swap RoPE pairing) the dense `qk_norm` path already gets --
    /// this asserts the MoE program's own length actually changes with the
    /// flag, the exact invariant the bug violated.
    #[test]
    fn mistral_cached_forward_program_with_experts_qk_norm_changes_the_moe_program() {
        let (qk_norm_off, _, _, _) = mistral_cached_forward_program_with_experts(
            32_000, 256, 128, 4, 2, 64, 1, 4, 1, false, false, false,
        )
        .expect("moe program without qk_norm lowers");
        let (qk_norm_on, _, _, _) = mistral_cached_forward_program_with_experts(
            32_000, 256, 128, 4, 2, 64, 1, 4, 1, true, false, false,
        )
        .expect("moe program with qk_norm lowers");

        assert_ne!(
            qk_norm_off.len(),
            qk_norm_on.len(),
            "a Qwen3-MoE-shaped forward program (expert_count > 0) must grow when qk_norm \
             flips on -- an unchanged length means the MoE layer builder is still dropping \
             attn_q_norm/attn_k_norm on the floor"
        );
    }

    /// [`mistral_cached_forward_program_with_experts_and_layer_taps`] must
    /// build the byte-identical program to its thin-wrapper sibling
    /// (`mistral_cached_forward_program_with_experts`'s own doc on that
    /// relationship) and return exactly one residual tap per layer, in
    /// layer order -- the invariant a caller bisecting CPU-vs-Metal
    /// divergence across a 48-layer checkpoint depends on to index
    /// `layer_residuals[layer]` directly.
    #[test]
    fn layer_taps_variant_matches_the_plain_program_and_returns_one_tap_per_layer() {
        let (plain_program, plain_roots, plain_cache_roots, _plain_moe_sites) =
            mistral_cached_forward_program_with_experts(
                32_000, 256, 128, 4, 2, 64, 3, 4, 1, true, false, false,
            )
            .expect("plain moe program lowers");
        let (taps_program, taps_roots, taps_cache_roots, layer_residuals, _taps_moe_sites) =
            mistral_cached_forward_program_with_experts_and_layer_taps(
                32_000, 256, 128, 4, 2, 64, 3, 4, 1, true, false, false, false,
            )
            .expect("taps moe program lowers");

        assert_eq!(
            plain_program, taps_program,
            "the taps variant must build the identical graph -- it only returns extra \
             NodeIds into the same program, never a structurally different one"
        );
        assert_eq!(plain_roots, taps_roots);
        assert_eq!(plain_cache_roots, taps_cache_roots);
        assert_eq!(
            layer_residuals.len(),
            3,
            "one residual NodeId per layer (block_count=3)"
        );
        assert!(
            layer_residuals.windows(2).all(|pair| pair[0].0 < pair[1].0),
            "residual taps must appear in strictly increasing program order across layers: \
             {layer_residuals:?}"
        );
    }

    /// [`ForwardRoots::hidden`] must be the pre-`lm_head` activation the
    /// vocab-projection multiply actually reads -- proved by walking the
    /// graph FROM `logits` backward (`logits`'s own `Reduce::operand` is
    /// the `logits_product` elementwise; that node's own operand set must
    /// contain `hidden`) rather than assuming any fixed distance between
    /// the two `NodeId`s. This is the structural guarantee
    /// `LoadedModel::embed` (`proxima-model-interop`) depends on: if a
    /// future refactor of this builder ever produced a `hidden` that is
    /// NOT actually upstream of `lm_head`'s multiply, this test fails
    /// before any real-checkpoint embedding test would even hint at it.
    #[test]
    fn forward_roots_hidden_is_an_operand_of_the_lm_head_product() {
        let (program, roots, _cache_roots, _moe_sites) = mistral_cached_forward_program_with_experts(
            32_002, 4096, 14336, 32, 8, 128, 2, 0, 0, false, false, false,
        )
        .expect("the dense cached forward pass lowers to a program");

        let Op::Reduce(logits_reduce) = &program[roots.logits.0 as usize] else {
            panic!("ForwardRoots::logits must name an Op::Reduce (the vocab-projection sum)");
        };
        let logits_product = logits_reduce.operand;

        let Op::Elementwise {
            operands: product_operands,
            ..
        } = &program[logits_product.0 as usize]
        else {
            panic!("logits's own Reduce::operand must name an Op::Elementwise (the multiply)");
        };

        assert!(
            product_operands
                .iter()
                .any(|(operand, _map)| *operand == roots.hidden),
            "ForwardRoots::hidden ({:?}) must be one of the lm_head product's own operands \
             ({:?}), or LoadedModel::embed would pool the wrong tensor",
            roots.hidden,
            product_operands
                .iter()
                .map(|(operand, _)| *operand)
                .collect::<alloc::vec::Vec<_>>(),
        );
    }

    const MATMUL_TOML: &str = r#"
[[node]]
op = "input"
id = "lhs"
dtype = "float32"
shape = ["?0", 768]

[[node]]
op = "input"
id = "rhs"
dtype = "float32"
shape = [768, 3072]

[[node]]
op = "elementwise"
id = "product"
dtype = "float32"
body = "multiply"
inputs = ["lhs", "rhs"]
maps = ["ik->ijk", "kj->ijk"]

[[node]]
op = "reduce"
id = "sum"
dtype = "float32"
body = "add"
init = "zero"
input = "product"
in_map = "ijk->ijk"
out_map = "ij->ijk"
keep = "reduce"
name = "matmul"
"#;

    fn matmul_in_rust() -> Vec<Op> {
        let mut program = Vec::new();
        let lhs = op::append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Symbolic(0), Extent::Static(768)],
                name: None,
            },
        );
        let rhs = op::append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(768), Extent::Static(3072)],
                name: None,
            },
        );
        let product = op::append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: alloc::vec![
                    (lhs, IndexMap::Affine(map::projection(3, &[0, 2]))),
                    (rhs, IndexMap::Affine(map::projection(3, &[2, 1]))),
                ],
                name: None,
            },
        );
        op::append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: Some("matmul".into()),
            }),
        );
        program
    }

    /// The whole reason this module exists: if these two disagree, the claim
    /// that the algebra is describable as data is false.
    #[test]
    fn a_program_written_as_toml_equals_the_same_program_written_in_rust() {
        let spec: ProgramSpec = toml::from_str(MATMUL_TOML).expect("spec parses");
        spec.validate().expect("spec is structurally sound");
        let from_config = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");
        assert_eq!(
            from_config,
            matmul_in_rust(),
            "config and code must produce the same program"
        );
        crate::shape::infer(&from_config, &[512]).expect("the parsed program also infers");
    }

    const EMBEDDING_TOML: &str = r#"
[[node]]
op = "input"
id = "table"
dtype = "float32"
shape = [50000, 8]

[[node]]
op = "input"
id = "ids"
dtype = "int32"
shape = [4]

[[node]]
op = "elementwise"
id = "gathered"
dtype = "float32"
body = "identity"
inputs = ["table"]
maps = [{ gather = "ids", index_map = "s->sd", map = "d->sd", dim = 0 }]
"#;

    fn embedding_lookup_in_rust() -> Vec<Op> {
        let mut program = Vec::new();
        let table = op::append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(50_000), Extent::Static(8)],
                name: None,
            },
        );
        let ids = op::append(
            &mut program,
            Op::Input {
                dtype: DType::Int32,
                shape: alloc::vec![Extent::Static(4)],
                name: None,
            },
        );
        let gathered_map = IndexMap::Computed {
            indices: ids,
            index_map: map::projection(2, &[0]),
            base: IndexPattern {
                iter_rank: 2,
                axes: alloc::vec![
                    AxisIndex::default(),
                    AxisIndex {
                        terms: core::iter::once(AxisTerm::projection(1)).collect(),
                        offset: 0,
                        len: None,
                    },
                ],
            },
            gathered_dim: 0,
        };
        op::append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Identity,
                operands: alloc::vec![(table, gathered_map)],
                name: None,
            },
        );
        program
    }

    /// The gather analogue of
    /// [`a_program_written_as_toml_equals_the_same_program_written_in_rust`]:
    /// an embedding lookup written as TOML must equal the same program built
    /// directly, and the parsed program must still pass shape inference.
    #[test]
    fn an_embedding_lookup_written_as_toml_equals_the_same_program_written_in_rust() {
        let spec: ProgramSpec = toml::from_str(EMBEDDING_TOML).expect("spec parses");
        spec.validate().expect("spec is structurally sound");
        let from_config = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");
        assert_eq!(
            from_config,
            embedding_lookup_in_rust(),
            "config and code must produce the same gather program"
        );
        crate::shape::infer(&from_config, &[]).expect("the parsed gather program also infers");
    }

    #[test]
    fn the_name_survives_the_config_round_trip() {
        let spec: ProgramSpec = toml::from_str(MATMUL_TOML).expect("spec parses");
        let program = Vec::<Op>::try_from(&spec).expect("lowers");
        let root = program.last().expect("root");
        assert_eq!(root.name(), Some("matmul"));
    }

    #[test]
    fn a_symbolic_extent_survives_as_a_symbol() {
        let spec: ProgramSpec = toml::from_str(MATMUL_TOML).expect("spec parses");
        let program = Vec::<Op>::try_from(&spec).expect("lowers");
        let Op::Input { shape, .. } = &program[0] else {
            panic!("first node is a leaf");
        };
        assert_eq!(
            shape[0],
            Extent::Symbolic(0),
            "sequence length stays unresolved"
        );
    }

    #[test]
    fn an_input_name_survives_the_config_round_trip() {
        let named = r#"
[[node]]
op = "input"
id = "x"
dtype = "float32"
shape = [4]
name = "weights.embedding"
"#;
        let spec: ProgramSpec = toml::from_str(named).expect("spec parses");
        let program = Vec::<Op>::try_from(&spec).expect("lowers");
        assert_eq!(program[0].name(), Some("weights.embedding"));
    }

    #[proxima::test]
    #[case::identity("ij->ij", 2, &[0, 1])]
    #[case::transpose("ji->ij", 2, &[1, 0])]
    #[case::broadcast("j->ij", 2, &[1])]
    #[case::contraction_lhs("ik->ijk", 3, &[0, 2])]
    #[case::full_reduction("->i", 1, &[])]
    async fn projection_notation_reads_like_einsum(
        #[case] notation: &str,
        #[case] rank: u16,
        #[case] projected: &[u16],
    ) {
        let (found_rank, found) = parse_projection(notation).expect("well-formed");
        assert_eq!(found_rank, rank);
        assert_eq!(found, projected);
    }

    #[test]
    fn a_map_without_an_arrow_is_rejected() {
        assert!(matches!(
            parse_projection("ijk").expect_err("no arrow"),
            TensorError::MalformedMap(_)
        ));
    }

    #[test]
    fn projecting_a_letter_the_iteration_space_lacks_is_rejected() {
        let error = parse_projection("iz->ijk").expect_err("z is not in ijk");
        assert!(
            matches!(error, TensorError::UnknownIndexLetter { letter: 'z', .. }),
            "{error}"
        );
    }

    /// A shifted, scaled, or multi-term address still honors a trailing
    /// `@length` -- the same declared-`len` fact `shape::unify_iteration_space`
    /// resolves regardless of how the address term is spelled
    /// (`AxisIndex::len_target_axis`'s own doc).
    #[proxima::test]
    #[case::shifted("s,i+1@2->si")]
    #[case::scaled("s,2*i@4->si")]
    #[case::multi_term_unit_coefficient("s,2*i+p@2->sip")]
    async fn a_len_declaration_parses_regardless_of_address_shape(#[case] notation: &str) {
        let pattern = parse_operand_pattern(notation).expect("a declared len parses");
        let len_bearing = pattern
            .axes
            .iter()
            .find(|axis| axis.len.is_some())
            .expect("one axis in the notation declares len");
        assert!(len_bearing.len_target_axis().is_some());
    }

    /// `i+j@2` has two equally-plain (`coeff == 1`) terms -- `len` cannot
    /// say which one it describes, so this is malformed at parse time
    /// rather than accepted and silently ignored (or rejected far later, at
    /// shape inference, over a program already built).
    #[test]
    fn a_len_on_two_unit_coefficient_terms_is_rejected_at_parse_time() {
        let error =
            parse_operand_pattern("s,i+j@2->sij").expect_err("len has no unambiguous target");
        assert!(matches!(error, TensorError::MalformedMap(_)), "{error}");
    }

    #[test]
    fn a_forward_reference_in_config_is_rejected() {
        let forward = r#"
[[node]]
op = "elementwise"
id = "early"
dtype = "float32"
body = "identity"
inputs = ["later"]
maps = ["i->i"]

[[node]]
op = "input"
id = "later"
dtype = "float32"
shape = [4]
"#;
        let spec: ProgramSpec = toml::from_str(forward).expect("parses");
        assert!(
            spec.validate().is_err(),
            "config order mirrors the program's backwards-reference rule"
        );
        assert!(matches!(
            Vec::<Op>::try_from(&spec).expect_err("cannot lower"),
            TensorError::UnknownNode(_)
        ));
    }

    #[test]
    fn a_duplicate_id_is_rejected() {
        let duplicate = r#"
[[node]]
op = "input"
id = "same"
dtype = "float32"
shape = [4]

[[node]]
op = "input"
id = "same"
dtype = "float32"
shape = [8]
"#;
        let spec: ProgramSpec = toml::from_str(duplicate).expect("parses");
        assert!(spec.validate().is_err(), "ids must be unique");
    }

    #[test]
    fn inputs_and_maps_must_agree_in_count() {
        let lopsided = r#"
[[node]]
op = "input"
id = "source"
dtype = "float32"
shape = [4]

[[node]]
op = "elementwise"
id = "bad"
dtype = "float32"
body = "add"
inputs = ["source", "source"]
maps = ["i->i"]
"#;
        let spec: ProgramSpec = toml::from_str(lopsided).expect("parses");
        assert!(spec.validate().is_err());
        assert!(matches!(
            Vec::<Op>::try_from(&spec).expect_err("cannot lower"),
            TensorError::SpecArityMismatch { .. }
        ));
    }

    #[test]
    fn a_malformed_extent_is_rejected() {
        let bad = r#"
[[node]]
op = "input"
id = "source"
dtype = "float32"
shape = ["seq"]
"#;
        let spec: ProgramSpec = toml::from_str(bad).expect("parses");
        assert!(matches!(
            Vec::<Op>::try_from(&spec).expect_err("`seq` is not `?n`"),
            TensorError::MalformedExtent(_)
        ));
    }

    use crate::test_support::Lcg;

    fn random_vec(seed: u64, count: usize) -> Vec<f32> {
        let mut lcg = Lcg(seed);
        (0..count).map(|_| lcg.next_unit()).collect()
    }

    /// The claim "a new architecture is a config file, not a PR" is only
    /// worth anything if a real architecture fits. A single-head attention
    /// block with RMSNorm and a full softmax does, and this checks it
    /// evaluates rather than merely parses — a spec that lowers and then
    /// produces garbage is still a PR waiting to happen.
    ///
    /// The softmax rows are the assertion that matters: finite output only
    /// proves the pipeline ran, whereas rows summing to one prove it computed
    /// attention. What this file still leaves as a plain input rather than
    /// deriving is a mask, which would need an index-derived tensor no `Op`
    /// produces — RoPE's multi-term affine no longer belongs on that list;
    /// see the pairwise-rotation test below.
    ///
    /// Inputs are LCG-derived rather than uniform constants: a uniform row
    /// makes every q/k/v row identical, so the scores collapse to a uniform
    /// distribution regardless of whether the index maps, reduction order,
    /// or broadcast are correct. Varied inputs make the softmax rows genuinely
    /// non-uniform, so a transposed axis or a wrong reduction actually shows
    /// up as a numeric difference instead of vanishing by symmetry.
    #[test]
    fn an_attention_block_written_as_toml_evaluates() {
        const SEQUENCE: usize = 4;
        const MODEL: usize = 8;

        let text = include_str!("../specs/attention_block.toml");
        let spec: ProgramSpec = toml::from_str(text).expect("spec parses");
        spec.validate().expect("spec is structurally sound");
        let program = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");

        let symbols = [SEQUENCE as u64];
        crate::shape::infer(&program, &symbols).expect("the block infers");

        let activations = random_vec(1, SEQUENCE * MODEL);
        let inverse_dim = alloc::vec![1.0 / MODEL as f32; SEQUENCE];
        let wq = random_vec(2, MODEL * MODEL);
        let wk = random_vec(3, MODEL * MODEL);
        let wv = random_vec(4, MODEL * MODEL);
        let blocks: [&[f32]; 5] = [&activations, &inverse_dim, &wq, &wk, &wv];

        let probabilities = spec
            .node
            .iter()
            .position(|node| node.id() == "probabilities")
            .expect("the spec defines a probabilities node");
        let probabilities = NodeId(probabilities as u32);
        let root = NodeId(program.len() as u32 - 1);

        let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");
        let evaluated = crate::cpu::evaluate_parallel(
            &program,
            &symbols,
            &blocks,
            &[root, probabilities],
            workers,
        )
        .expect("the block evaluates");

        let output = evaluated.root();
        assert_eq!(
            output.len(),
            SEQUENCE * MODEL,
            "a vacuous output proves nothing"
        );
        assert!(
            output.iter().all(|value| value.is_finite()),
            "output must be finite"
        );

        let (rows, _) = evaluated
            .get(probabilities)
            .expect("probabilities were requested");
        assert_eq!(rows.len(), SEQUENCE * SEQUENCE);
        for row in rows.as_chunks::<SEQUENCE>().0 {
            let total: f32 = row.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-5,
                "softmax row sums to {total}, not 1.0"
            );
            let max = row.iter().copied().fold(f32::MIN, f32::max);
            let min = row.iter().copied().fold(f32::MAX, f32::min);
            assert!(
                max - min > 1e-3,
                "softmax row {row:?} is uniform (max - min = {}); varied inputs should break score ties",
                max - min
            );
        }
    }

    /// RoPE's whole reason for existing in this module's doc: `2*i` and
    /// `2*i+1` are the multi-term affine that used to have no string
    /// spelling. This does not just check the spec parses — a parser that
    /// silently addressed the wrong elements would still parse — it checks
    /// the *evaluated* output obeys the one property that only holds if the
    /// pairwise addressing is right: a rotation preserves each pair's norm.
    /// `expected` is computed straight off the raw `x` buffer at the literal
    /// indices `2*i` / `2*i+1`, independently of anything the graph did, so
    /// an addressing bug (reading `i` instead of `2*i`, or the wrong operand
    /// axis order) would read a different pair and very likely a different
    /// norm — `x`'s eight values are pairwise distinct for exactly that
    /// reason.
    #[test]
    fn a_rope_pairwise_rotation_written_as_toml_preserves_pair_norm() {
        const SEQUENCE: usize = 2;
        const MODEL: usize = 4;
        const PAIRS: usize = MODEL / 2;

        let text = include_str!("../specs/rope.toml");
        let spec: ProgramSpec = toml::from_str(text).expect("spec parses");
        spec.validate().expect("spec is structurally sound");
        let program = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");

        let symbols: [u64; 0] = [];
        crate::shape::infer(&program, &symbols).expect("the rotation infers");

        let x: [f32; SEQUENCE * MODEL] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // two exact Pythagorean triples (3-4-5 and 7-24-25 scaled by 1/25),
        // so cos^2 + sin^2 = 1 exactly and any drift in the assertion below
        // is evaluation error, not a badly chosen rotation.
        let cos: [f32; SEQUENCE * PAIRS] = [0.6, 0.28, 0.6, 0.28];
        let sin: [f32; SEQUENCE * PAIRS] = [0.8, 0.96, 0.8, 0.96];
        let blocks: [&[f32]; 3] = [&x, &cos, &sin];

        let node_id = |id: &str| {
            let position = spec
                .node
                .iter()
                .position(|node| node.id() == id)
                .unwrap_or_else(|| panic!("the spec defines a {id} node"));
            NodeId(position as u32)
        };
        let rotated_even_id = node_id("rotated_even");
        let root = NodeId(program.len() as u32 - 1);
        assert_eq!(root, node_id("rotated_odd"), "rotated_odd is the last node");

        let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");
        let evaluated = crate::cpu::evaluate_parallel(
            &program,
            &symbols,
            &blocks,
            &[root, rotated_even_id],
            workers,
        )
        .expect("the rotation evaluates");

        let rotated_odd = evaluated.root();
        let (rotated_even, _) = evaluated
            .get(rotated_even_id)
            .expect("rotated_even was requested");
        assert_eq!(rotated_even.len(), SEQUENCE * PAIRS);
        assert_eq!(rotated_odd.len(), SEQUENCE * PAIRS);

        for sequence in 0..SEQUENCE {
            for pair in 0..PAIRS {
                let raw_even = x[sequence * MODEL + 2 * pair];
                let raw_odd = x[sequence * MODEL + 2 * pair + 1];
                let expected_norm = raw_even * raw_even + raw_odd * raw_odd;

                let rotated_index = sequence * PAIRS + pair;
                let found_even = rotated_even[rotated_index];
                let found_odd = rotated_odd[rotated_index];
                let found_norm = found_even * found_even + found_odd * found_odd;

                assert!(
                    (found_norm - expected_norm).abs() < 1e-3,
                    "pair ({raw_even}, {raw_odd}) has norm {expected_norm} but the rotated \
                     pair ({found_even}, {found_odd}) has norm {found_norm}"
                );
            }
        }
    }

    /// The test that makes `Op::Iota` worth having: `causal_attention.toml`
    /// is `attention_block.toml` plus a real causal mask built from two
    /// `Iota` leaves, and the property that makes a mask *causal* rather
    /// than decorative is checked directly on the evaluated softmax output —
    /// not just that the spec parses or that the output is finite.
    ///
    /// Two invariants, over every one of the `SEQUENCE * SEQUENCE` = 16
    /// probability cells (`checked` asserts that count, so a loop bug can't
    /// silently check zero of them):
    /// - every strictly-upper-triangular cell (`key > query`, a key position
    ///   later than its query) is *exactly* `0.0` — not merely small,
    ///   because `exp(-infinity)` is exact zero in IEEE-754 and a mask that
    ///   only suppresses without zeroing is not a causal mask;
    /// - every row still sums to `1.0`, the same softmax invariant
    ///   `an_attention_block_written_as_toml_evaluates` checks, proving the
    ///   mask did not just zero everything.
    ///
    /// Inputs are LCG-derived, not uniform, for the same reason
    /// `an_attention_block_written_as_toml_evaluates` gives: under uniform
    /// input every unmasked score in a row is identical, so a mask that
    /// masked the wrong cells (or none at all) could still coincidentally
    /// leave the *sum* at 1.0 — varied scores make a wrong mask show up as a
    /// nonzero cell instead of vanishing by symmetry.
    #[test]
    fn a_causal_attention_block_written_as_toml_masks_future_positions() {
        const SEQUENCE: usize = 4;
        const MODEL: usize = 8;

        let text = include_str!("../specs/causal_attention.toml");
        let spec: ProgramSpec = toml::from_str(text).expect("spec parses");
        spec.validate().expect("spec is structurally sound");
        let program = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");

        let symbols = [SEQUENCE as u64];
        crate::shape::infer(&program, &symbols).expect("the causal block infers");

        let activations = random_vec(11, SEQUENCE * MODEL);
        let inverse_dim = alloc::vec![1.0 / MODEL as f32; SEQUENCE];
        let wq = random_vec(12, MODEL * MODEL);
        let wk = random_vec(13, MODEL * MODEL);
        let wv = random_vec(14, MODEL * MODEL);
        let blocks: [&[f32]; 5] = [&activations, &inverse_dim, &wq, &wk, &wv];

        let probabilities = spec
            .node
            .iter()
            .position(|node| node.id() == "probabilities")
            .expect("the spec defines a probabilities node");
        let probabilities = NodeId(probabilities as u32);
        let root = NodeId(program.len() as u32 - 1);

        let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");
        let evaluated = crate::cpu::evaluate_parallel(
            &program,
            &symbols,
            &blocks,
            &[root, probabilities],
            workers,
        )
        .expect("the causal block evaluates");

        let output = evaluated.root();
        assert_eq!(
            output.len(),
            SEQUENCE * MODEL,
            "a vacuous output proves nothing"
        );
        assert!(
            output.iter().all(|value| value.is_finite()),
            "output must be finite"
        );

        let (rows, _) = evaluated
            .get(probabilities)
            .expect("probabilities were requested");
        assert_eq!(rows.len(), SEQUENCE * SEQUENCE);

        let mut checked = 0usize;
        for (query, row) in rows.as_chunks::<SEQUENCE>().0.iter().enumerate() {
            let total: f32 = row.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-5,
                "softmax row {query} sums to {total}, not 1.0"
            );
            for (key, &probability) in row.iter().enumerate() {
                if key > query {
                    assert_eq!(
                        probability, 0.0,
                        "row {query} col {key} is strictly upper-triangular (key {key} > \
                         query {query}) and must be masked to exactly 0.0, found {probability}"
                    );
                }
                checked += 1;
            }
        }
        assert_eq!(
            checked,
            SEQUENCE * SEQUENCE,
            "every probability cell must be checked, not a subset"
        );
    }

    /// A full llama-style block — attention plus its output projection and
    /// residual, a second RMSNorm, and a SwiGLU feed-forward with its own
    /// residual — built on top of the attention block above. Every addition
    /// lowers with the same node kinds and closed `ScalarOp` set the
    /// attention block already used; `transformer_block.toml`'s header
    /// records why nothing new was needed.
    ///
    /// Two invariants, not just finiteness:
    /// - the softmax rows inside it still sum to one, the same evidence
    ///   `an_attention_block_written_as_toml_evaluates` uses;
    /// - a degenerate control: zero every projection weight (Q/K/V, the
    ///   output projection, and all three FFN matrices) and the block must
    ///   return its own input unchanged, because both sub-blocks' nonlinear
    ///   interior gets multiplied away by a zeroed projection before either
    ///   residual add — only the residual path survives. If this assertion
    ///   fails, the residual wiring is broken and weakening it to an
    ///   approximate check would hide that.
    #[test]
    fn a_transformer_block_written_as_toml_evaluates() {
        const SEQUENCE: usize = 4;
        const MODEL: usize = 8;
        const FFN: usize = 16;

        let text = include_str!("../specs/transformer_block.toml");
        let spec: ProgramSpec = toml::from_str(text).expect("spec parses");
        spec.validate().expect("spec is structurally sound");
        let program = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");

        let symbols = [SEQUENCE as u64];
        crate::shape::infer(&program, &symbols).expect("the block infers");

        let probabilities = spec
            .node
            .iter()
            .position(|node| node.id() == "probabilities")
            .expect("the spec defines a probabilities node");
        let probabilities = NodeId(probabilities as u32);
        let root = NodeId(program.len() as u32 - 1);
        let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");

        // --- run 1: real weights, evaluates to something finite and the
        // softmax invariant still holds inside the larger block.
        let activations = alloc::vec![0.5f32; SEQUENCE * MODEL];
        let inverse_dim = alloc::vec![1.0 / MODEL as f32; SEQUENCE];
        let ones = alloc::vec![1.0f32; SEQUENCE];
        let square_weights = alloc::vec![0.125f32; MODEL * MODEL];
        let gate_up_weights = alloc::vec![0.0625f32; MODEL * FFN];
        let down_weights = alloc::vec![0.0625f32; FFN * MODEL];
        let real_blocks: [&[f32]; 10] = [
            &activations,
            &inverse_dim,
            &ones,
            &square_weights,
            &square_weights,
            &square_weights,
            &square_weights,
            &gate_up_weights,
            &gate_up_weights,
            &down_weights,
        ];

        let evaluated = crate::cpu::evaluate_parallel(
            &program,
            &symbols,
            &real_blocks,
            &[root, probabilities],
            workers,
        )
        .expect("the block evaluates");

        let output = evaluated.root();
        assert_eq!(
            output.len(),
            SEQUENCE * MODEL,
            "a vacuous output proves nothing"
        );
        assert!(
            output.iter().all(|value| value.is_finite()),
            "output must be finite"
        );

        let (rows, _) = evaluated
            .get(probabilities)
            .expect("probabilities were requested");
        for row in rows.as_chunks::<SEQUENCE>().0 {
            let total: f32 = row.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-5,
                "softmax row sums to {total}, not 1.0"
            );
        }

        // --- run 2: degenerate control. every projection weight is zero, so
        // attention's contribution and the feed-forward's contribution are
        // each multiplied to exactly zero before their residual add — the
        // block must hand its input straight through.
        let zero_square = alloc::vec![0.0f32; MODEL * MODEL];
        let zero_gate_up = alloc::vec![0.0f32; MODEL * FFN];
        let zero_down = alloc::vec![0.0f32; FFN * MODEL];
        let zeroed_blocks: [&[f32]; 10] = [
            &activations,
            &inverse_dim,
            &ones,
            &zero_square,
            &zero_square,
            &zero_square,
            &zero_square,
            &zero_gate_up,
            &zero_gate_up,
            &zero_down,
        ];

        let evaluated_zeroed =
            crate::cpu::evaluate_parallel(&program, &symbols, &zeroed_blocks, &[root], workers)
                .expect("the zeroed block evaluates");

        let residual_output = evaluated_zeroed.root();
        assert_eq!(residual_output.len(), activations.len());
        for (result, input) in residual_output.iter().zip(activations.iter()) {
            assert!(
                (result - input).abs() < 1e-5,
                "residual did not carry: got {result}, expected input {input}"
            );
        }
    }

    /// `specs/conv2d.toml`'s whole reason to exist: proves a [`NodeSpec::Reduce`]'s
    /// `in_map` can now spell the same multi-term windowing an `Elementwise`
    /// operand already could — `Reduce(Add)` over `Elementwise(Multiply)`,
    /// this file's own `matmul` shape, but with a two-term spatial axis
    /// (`h+y`, `w+x`) in place of a bare projection.
    ///
    /// Two invariants, not just finiteness:
    /// - output channel 0's kernel is all zero except a single 1 at the 3x3
    ///   window's centre, so every output pixel is exactly the padded
    ///   image's centre-tapped pixel — which, because the image was padded
    ///   by exactly the kernel's radius, is the *original* unpadded pixel at
    ///   the same coordinate. Reproducing 25 pixels exactly proves the
    ///   two-term axis addressed the right element at every position, not
    ///   merely that evaluation completed;
    /// - output channel 1's kernel is all zero, a degenerate control: every
    ///   one of its 25 pixels must be exactly zero, proving the reduction
    ///   actually depends on the kernel's weights rather than echoing its
    ///   windowed input regardless of them.
    #[test]
    fn a_conv2d_written_as_toml_reproduces_its_input_through_a_center_tap_kernel() {
        const IMAGE: usize = 5;
        const PADDED: usize = IMAGE + 2;
        const KERNEL: usize = 3;
        const CENTRE: usize = KERNEL / 2;

        let text = include_str!("../specs/conv2d.toml");
        let spec: ProgramSpec = toml::from_str(text).expect("spec parses");
        spec.validate().expect("spec is structurally sound");
        let program = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");

        let symbols: [u64; 0] = [];
        crate::shape::infer(&program, &symbols).expect("the convolution infers");

        // image: a zero border (the materialized padding) around a real,
        // non-constant 5x5 interior, so a transposed axis or a wrong offset
        // reads a different, numerically distinct pixel rather than
        // vanishing by symmetry.
        let interior = random_vec(11, IMAGE * IMAGE);
        let mut image = alloc::vec![0.0f32; PADDED * PADDED];
        for row in 0..IMAGE {
            for col in 0..IMAGE {
                image[(row + 1) * PADDED + (col + 1)] = interior[row * IMAGE + col];
            }
        }

        // kernel: [co, ho, wo, kh, kw] = [2, 5, 5, 3, 3]. Channel 0 is a
        // center-tap identity at every output position; channel 1 stays all
        // zero (the `vec!` default).
        let mut kernel = alloc::vec![0.0f32; 2 * IMAGE * IMAGE * KERNEL * KERNEL];
        for out_row in 0..IMAGE {
            for out_col in 0..IMAGE {
                let index = (((out_row * IMAGE + out_col) * KERNEL) + CENTRE) * KERNEL + CENTRE;
                kernel[index] = 1.0;
            }
        }

        let root = NodeId(program.len() as u32 - 1);
        let blocks: [&[f32]; 2] = [&image, &kernel];
        let evaluated = crate::cpu::evaluate(&program, &symbols, &blocks, &[root])
            .expect("the convolution evaluates");

        let output = evaluated.root();
        assert_eq!(
            output.len(),
            2 * IMAGE * IMAGE,
            "a vacuous output proves nothing"
        );

        let channel_0 = &output[..IMAGE * IMAGE];
        let channel_1 = &output[IMAGE * IMAGE..];

        assert_eq!(
            channel_0,
            interior.as_slice(),
            "channel 0's center-tap kernel must reproduce all {} interior pixels exactly",
            IMAGE * IMAGE
        );
        for (index, value) in channel_1.iter().enumerate() {
            assert_eq!(
                *value, 0.0,
                "channel 1's all-zero kernel must produce exactly zero at pixel {index}, got {value}"
            );
        }
    }

    /// `row @ matrix`, `row` length `d_in`, `matrix` row-major `[d_in,
    /// d_out]` — the reference computation `moe_block.toml`'s own test
    /// checks the graph against, independent of anything the graph did.
    fn matvec(row: &[f32], matrix: &[f32], d_in: usize, d_out: usize) -> alloc::vec::Vec<f32> {
        (0..d_out)
            .map(|out| {
                (0..d_in)
                    .map(|inp| row[inp] * matrix[inp * d_out + out])
                    .sum()
            })
            .collect()
    }

    /// The whole reason `Op::Iota` plus `IndexMap::Computed` together are
    /// worth having: a top-1 sparse mixture-of-experts feed-forward, built
    /// from gate -> argmax route -> gathered expert weights -> the expert's
    /// own linear layer, with zero new `Op`/`ScalarOp` variants over what
    /// `causal_attention.toml`'s mask and the embedding-lookup worked
    /// example already used. `moe_block.toml`'s own header spells out the
    /// argmax construction (`mask * iota`, no `Select`, no synthetic
    /// `-infinity`) and why the gather is the same mechanism as an
    /// embedding lookup with one more non-gathered axis.
    ///
    /// Two tokens, two experts, wired so token 0's gate logits favor expert
    /// 0 (3 vs 1) and token 1's favor expert 1 (4 vs 1). `expected_token0`/
    /// `expected_token1` are each computed directly from that token's own
    /// `x` row and its *routed* expert's weight matrix via [`matvec`],
    /// independently of the graph — if the gather read the wrong expert's
    /// slab, or the wrong token's `x` row, or `argmax` picked the wrong
    /// index, this is what would catch it, not a shape or finiteness check.
    ///
    /// The degenerate control reruns the identical graph with the gate
    /// weights swapped, which flips both tokens' routes (token 0 -> expert
    /// 1, token 1 -> expert 0 now — see the swapped-gate arithmetic in the
    /// comments below), but both experts' weight slabs set to
    /// `matrix_a`. If routing still leaked into the result, the output
    /// would differ from `x @ matrix_a` for one or both tokens; since the
    /// experts are equal, it must not.
    #[test]
    fn a_moe_block_written_as_toml_routes_each_token_to_its_own_experts_weights() {
        const SEQUENCE: usize = 2;
        const D_IN: usize = 3;
        const D_OUT: usize = 2;
        const N_EXPERTS: usize = 2;

        let text = include_str!("../specs/moe_block.toml");
        let spec: ProgramSpec = toml::from_str(text).expect("spec parses");
        spec.validate().expect("spec is structurally sound");
        let program = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");

        let symbols = [SEQUENCE as u64];
        crate::shape::infer(&program, &symbols).expect("the moe block infers");

        let root = NodeId(program.len() as u32 - 1);
        let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");

        // token 0 = [3, 2, 1]: logits = [x[0], x[2]] = [3, 1] -> expert 0.
        // token 1 = [1, 2, 4]: logits = [x[0], x[2]] = [1, 4] -> expert 1.
        let x: [f32; SEQUENCE * D_IN] = [3.0, 2.0, 1.0, 1.0, 2.0, 4.0];
        let gate_w: [f32; D_IN * N_EXPERTS] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let matrix_a: [f32; D_IN * D_OUT] = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let matrix_b: [f32; D_IN * D_OUT] = [2.0, 0.0, 0.0, 2.0, 1.0, -1.0];
        let expert_w: [f32; N_EXPERTS * D_IN * D_OUT] = [
            matrix_a[0],
            matrix_a[1],
            matrix_a[2],
            matrix_a[3],
            matrix_a[4],
            matrix_a[5],
            matrix_b[0],
            matrix_b[1],
            matrix_b[2],
            matrix_b[3],
            matrix_b[4],
            matrix_b[5],
        ];
        let blocks: [&[f32]; 3] = [&x, &gate_w, &expert_w];

        let evaluated =
            crate::cpu::evaluate_parallel(&program, &symbols, &blocks, &[root], workers)
                .expect("the moe block evaluates");
        let output = evaluated.root();
        assert_eq!(
            output.len(),
            SEQUENCE * D_OUT,
            "a vacuous output proves nothing"
        );

        let expected_token0 = matvec(&x[0..D_IN], &matrix_a, D_IN, D_OUT);
        let expected_token1 = matvec(&x[D_IN..2 * D_IN], &matrix_b, D_IN, D_OUT);
        for (found, expected) in output[0..D_OUT].iter().zip(&expected_token0) {
            assert!(
                (found - expected).abs() < 1e-5,
                "token 0 (routed to expert 0): got {found}, expected {expected}"
            );
        }
        for (found, expected) in output[D_OUT..2 * D_OUT].iter().zip(&expected_token1) {
            assert!(
                (found - expected).abs() < 1e-5,
                "token 1 (routed to expert 1): got {found}, expected {expected}"
            );
        }

        // --- degenerate control: swap the gate so routing flips.
        // token 0 = [3, 2, 1]: logits = [x[2], x[0]] = [1, 3] -> expert 1.
        // token 1 = [1, 2, 4]: logits = [x[2], x[0]] = [4, 1] -> expert 0.
        // Both experts' weights are `matrix_a`, so the flipped route must
        // not change the answer from `x @ matrix_a`.
        let swapped_gate_w: [f32; D_IN * N_EXPERTS] = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        let uniform_expert_w: [f32; N_EXPERTS * D_IN * D_OUT] = [
            matrix_a[0],
            matrix_a[1],
            matrix_a[2],
            matrix_a[3],
            matrix_a[4],
            matrix_a[5],
            matrix_a[0],
            matrix_a[1],
            matrix_a[2],
            matrix_a[3],
            matrix_a[4],
            matrix_a[5],
        ];
        let degenerate_blocks: [&[f32]; 3] = [&x, &swapped_gate_w, &uniform_expert_w];
        let evaluated_degenerate =
            crate::cpu::evaluate_parallel(&program, &symbols, &degenerate_blocks, &[root], workers)
                .expect("the degenerate moe block evaluates");
        let degenerate_output = evaluated_degenerate.root();

        let expected_uniform_token0 = matvec(&x[0..D_IN], &matrix_a, D_IN, D_OUT);
        let expected_uniform_token1 = matvec(&x[D_IN..2 * D_IN], &matrix_a, D_IN, D_OUT);
        for (found, expected) in degenerate_output[0..D_OUT]
            .iter()
            .zip(&expected_uniform_token0)
        {
            assert!(
                (found - expected).abs() < 1e-5,
                "degenerate control, token 0: got {found}, expected {expected} \
                 (routing flipped but experts are identical, so output must not move)"
            );
        }
        for (found, expected) in degenerate_output[D_OUT..2 * D_OUT]
            .iter()
            .zip(&expected_uniform_token1)
        {
            assert!(
                (found - expected).abs() < 1e-5,
                "degenerate control, token 1: got {found}, expected {expected} \
                 (routing flipped but experts are identical, so output must not move)"
            );
        }
    }

    /// Probe for the harder question `a_moe_block_written_as_toml_...` does
    /// not answer: does a *fixed* k > 1 stay expressible with zero new
    /// ops, or does top-k genuinely need something this crate lacks
    /// (`moe_topk2_probe.toml`'s own header names the boundary: a fixed,
    /// unrolled k is fine, a general variable-k `TopK` op is not)?
    ///
    /// Three experts, logits `[2, 5, 3]` by construction (see the spec's
    /// gate weights): top-2 must select expert 1 (5) then expert 2 (3) and
    /// exclude expert 0 (2). Expert 0's weight is `[100, 100]` — wildly
    /// different from experts 1 (`[1, 2]`) and 2 (`[3, 4]`) — so a wrong
    /// inclusion is not a rounding error, it is off by roughly 30-100x.
    /// `expected = x . expert1_weight + x . expert2_weight`, computed
    /// independently of the graph via [`matvec`].
    #[test]
    fn a_topk2_probe_unrolls_two_argmax_rounds_with_exclusion() {
        const D_IN: usize = 2;
        const D_OUT: usize = 1;
        const N_EXPERTS: usize = 3;

        let text = include_str!("../specs/moe_topk2_probe.toml");
        let spec: ProgramSpec = toml::from_str(text).expect("spec parses");
        spec.validate().expect("spec is structurally sound");
        let program = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");

        let symbols = [1u64];
        crate::shape::infer(&program, &symbols).expect("the top-2 probe infers");

        let root = NodeId(program.len() as u32 - 1);
        let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");

        // logits = x @ gate_w = [1*1+1*1, 1*2+1*3, 1*1+1*2] = [2, 5, 3].
        let x: [f32; D_IN] = [1.0, 1.0];
        let gate_w: [f32; D_IN * N_EXPERTS] = [1.0, 2.0, 1.0, 1.0, 3.0, 2.0];
        let expert0_weight: [f32; D_IN * D_OUT] = [100.0, 100.0];
        let expert1_weight: [f32; D_IN * D_OUT] = [1.0, 2.0];
        let expert2_weight: [f32; D_IN * D_OUT] = [3.0, 4.0];
        let expert_w: [f32; N_EXPERTS * D_IN * D_OUT] = [
            expert0_weight[0],
            expert0_weight[1],
            expert1_weight[0],
            expert1_weight[1],
            expert2_weight[0],
            expert2_weight[1],
        ];
        let blocks: [&[f32]; 3] = [&x, &gate_w, &expert_w];

        let evaluated =
            crate::cpu::evaluate_parallel(&program, &symbols, &blocks, &[root], workers)
                .expect("the top-2 probe evaluates");
        let output = evaluated.root();
        assert_eq!(output.len(), D_OUT, "a vacuous output proves nothing");

        let expected_expert1 = matvec(&x, &expert1_weight, D_IN, D_OUT);
        let expected_expert2 = matvec(&x, &expert2_weight, D_IN, D_OUT);
        let expected = expected_expert1[0] + expected_expert2[0];
        assert!(
            (output[0] - expected).abs() < 1e-5,
            "got {}, expected {expected} (expert 1's {expected_expert1:?} + expert 2's \
             {expected_expert2:?}); expert 0's [100, 100] weight must never contribute",
            output[0]
        );
    }

    /// SwiGLU over a raw `f32` slice, independent of the graph
    /// [`append_moe_ffn`] builds -- the same role [`matvec`] plays for the
    /// bare-linear probes above, just with the real per-layer nonlinearity
    /// [`append_mistral_layer`]'s dense FFN also runs.
    fn swiglu_ffn(
        x: &[f32],
        gate_w: &[f32],
        up_w: &[f32],
        down_w: &[f32],
        d_in: usize,
        hidden: usize,
    ) -> alloc::vec::Vec<f32> {
        let gate = matvec(x, gate_w, d_in, hidden);
        let up = matvec(x, up_w, d_in, hidden);
        let activated: alloc::vec::Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&gate_value, &up_value)| {
                let silu = gate_value / (1.0 + (-gate_value).exp());
                silu * up_value
            })
            .collect();
        matvec(&activated, down_w, hidden, d_in)
    }

    /// Independent top-`k` reference: which experts a token's `logits` route
    /// to (descending order, ties broken toward the lower index the same
    /// way [`append_moe_ffn`]'s `mask * iota` construction does) and their
    /// softmax shares among only that selected set --
    /// `weight_i = exp(logit_i - max) / sum_selected`, the same shift
    /// [`append_moe_ffn`]'s doc names.
    fn top_k_routes_and_weights(logits: &[f32], k: usize) -> alloc::vec::Vec<(usize, f32)> {
        let mut remaining: alloc::vec::Vec<usize> = (0..logits.len()).collect();
        let mut routes = alloc::vec::Vec::new();
        for _ in 0..k {
            let winner = *remaining
                .iter()
                .max_by(|&&left, &&right| {
                    logits[left]
                        .partial_cmp(&logits[right])
                        .expect("logits are finite")
                })
                .expect("k does not exceed the expert count");
            routes.push(winner);
            remaining.retain(|&candidate| candidate != winner);
        }
        let max_logit = routes
            .iter()
            .map(|&expert| logits[expert])
            .fold(f32::NEG_INFINITY, f32::max);
        let unnormalized: alloc::vec::Vec<f32> = routes
            .iter()
            .map(|&expert| (logits[expert] - max_logit).exp())
            .collect();
        let total: f32 = unnormalized.iter().sum();
        routes
            .into_iter()
            .zip(unnormalized)
            .map(|(expert, weight)| (expert, weight / total))
            .collect()
    }

    /// End-to-end proof for [`append_moe_ffn`]/[`append_mistral_moe_layer`]:
    /// two tokens, three experts, top-2 routing, real SwiGLU per expert
    /// (not the bare-linear stand-in the two probes above use) and a real
    /// softmax combination weight -- everything [`a_moe_block_written_as_toml_...`]
    /// and [`a_topk2_probe_...`] proved the algebra can express, now proven
    /// for the actual generated code this crate ships, not just the TOML
    /// worked examples.
    ///
    /// Router weights are chosen so token 0 (`[3, 2]`) routes to experts
    /// `2, 0` (logits `[3, 2, 4]`) and token 1 (`[1, 4]`) routes to experts
    /// `2, 1` (logits `[1, 4, 8]`) -- a different pair per token, so a
    /// cross-token routing bug (using token 0's route for token 1 or vice
    /// versa) is not masked by both tokens agreeing. `expected` is computed
    /// entirely independently: [`top_k_routes_and_weights`] picks the route
    /// and softmax shares from the same raw `logits` the graph computes
    /// on-the-fly, and [`swiglu_ffn`] runs each selected expert's own
    /// weights with no dependency on [`Op`]/[`IndexMap`]/[`append_moe_ffn`]
    /// itself.
    #[test]
    fn a_routed_ffn_built_by_append_moe_ffn_matches_an_independent_topk_swiglu_reference() {
        const SEQUENCE: usize = 2;
        const EMBEDDING: usize = 2;
        const FEED_FORWARD: usize = 2;
        const EXPERT_COUNT: u32 = 3;
        const EXPERT_USED_COUNT: u32 = 2;

        let x: [f32; SEQUENCE * EMBEDDING] = [3.0, 2.0, 1.0, 4.0];
        // gate_inp[d, e]: logits[s, e] = sum_d x[s, d] * gate_inp[d, e].
        let gate_inp: [f32; EMBEDDING * EXPERT_COUNT as usize] = [1.0, 0.0, 0.0, 0.0, 1.0, 2.0];

        let gate_weights: [[f32; EMBEDDING * FEED_FORWARD]; 3] = [
            [1.0, 0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0, 2.0],
            [1.0, 1.0, 1.0, 1.0],
        ];
        let up_weights: [[f32; EMBEDDING * FEED_FORWARD]; 3] = [
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0, 2.0],
        ];
        let down_weights: [[f32; FEED_FORWARD * EMBEDDING]; 3] = [
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
        ];

        let stack_experts =
            |weights: &[[f32; EMBEDDING * FEED_FORWARD]; 3]| -> alloc::vec::Vec<f32> {
                weights.iter().flatten().copied().collect()
            };
        let expert_w_gate = stack_experts(&gate_weights);
        let expert_w_up = stack_experts(&up_weights);
        let expert_w_down: alloc::vec::Vec<f32> = down_weights.iter().flatten().copied().collect();

        let mut program = Vec::new();
        let x_node = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(0), Extent::Static(EMBEDDING as u32)],
            "x",
        );
        let gate_inp_node = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(EMBEDDING as u32),
                Extent::Static(EXPERT_COUNT)
            ],
            "gate_inp",
        );
        let expert_w_gate_node = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(EXPERT_COUNT),
                Extent::Static(EMBEDDING as u32),
                Extent::Static(FEED_FORWARD as u32)
            ],
            "expert_w_gate",
        );
        let expert_w_up_node = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(EXPERT_COUNT),
                Extent::Static(EMBEDDING as u32),
                Extent::Static(FEED_FORWARD as u32)
            ],
            "expert_w_up",
        );
        let expert_w_down_node = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(EXPERT_COUNT),
                Extent::Static(FEED_FORWARD as u32),
                Extent::Static(EMBEDDING as u32)
            ],
            "expert_w_down",
        );
        let ones = scalar_constant(&mut program, 1.0);

        let (root, _site) = append_moe_ffn(
            &mut program,
            0,
            x_node,
            gate_inp_node,
            expert_w_gate_node,
            expert_w_up_node,
            expert_w_down_node,
            EXPERT_COUNT,
            EXPERT_USED_COUNT,
            ones,
            ExpertGatingFunc::Softmax,
            None,
        )
        .expect("the routed ffn lowers");

        let symbols = [SEQUENCE as u64];
        crate::shape::infer(&program, &symbols).expect("the routed ffn infers");

        let blocks: [&[f32]; 5] = [&x, &gate_inp, &expert_w_gate, &expert_w_up, &expert_w_down];
        let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");
        let evaluated =
            crate::cpu::evaluate_parallel(&program, &symbols, &blocks, &[root], workers)
                .expect("the routed ffn evaluates");
        let output = evaluated.root();
        assert_eq!(
            output.len(),
            SEQUENCE * EMBEDDING,
            "a vacuous output proves nothing"
        );

        for (token, x_row) in x.chunks(EMBEDDING).enumerate() {
            let logits: alloc::vec::Vec<f32> = (0..EXPERT_COUNT as usize)
                .map(|expert| {
                    (0..EMBEDDING)
                        .map(|dim| x_row[dim] * gate_inp[dim * EXPERT_COUNT as usize + expert])
                        .sum()
                })
                .collect();
            let routes = top_k_routes_and_weights(&logits, EXPERT_USED_COUNT as usize);
            let mut expected = alloc::vec![0.0f32; EMBEDDING];
            for (expert, weight) in routes {
                let expert_out = swiglu_ffn(
                    x_row,
                    &gate_weights[expert],
                    &up_weights[expert],
                    &down_weights[expert],
                    EMBEDDING,
                    FEED_FORWARD,
                );
                for (accum, value) in expected.iter_mut().zip(&expert_out) {
                    *accum += weight * value;
                }
            }
            let found = &output[token * EMBEDDING..(token + 1) * EMBEDDING];
            for (found_value, expected_value) in found.iter().zip(&expected) {
                assert!(
                    (found_value - expected_value).abs() < 1e-4,
                    "token {token}: got {found:?}, expected {expected:?} (independent top-{EXPERT_USED_COUNT} \
                     softmax-weighted swiglu reference)"
                );
            }
        }
    }

    /// [`ExpertGatingFunc::Sigmoid`]'s independent reference:
    /// `route_tokens_to_experts` (`modeling_lfm2_moe.py:208-220`) selects
    /// top-`k` by `sigmoid(logits) + bias`, then weights each selected
    /// expert by its OWN unbiased `sigmoid(logits)` value, normalized over
    /// only the selected set. Mirrors [`top_k_routes_and_weights`]'s own
    /// shape (max-by, retain, normalize) with the two extra steps LFM2's
    /// gating needs: a sigmoid instead of a softmax, and a bias that
    /// participates in the `max_by` but never in the returned weight.
    fn sigmoid_topk_routes_and_weights(
        logits: &[f32],
        bias: &[f32],
        k: usize,
    ) -> alloc::vec::Vec<(usize, f32)> {
        let scores: alloc::vec::Vec<f32> = logits
            .iter()
            .map(|&logit| 1.0 / (1.0 + (-logit).exp()))
            .collect();
        let selection: alloc::vec::Vec<f32> = scores
            .iter()
            .zip(bias)
            .map(|(&score, &b)| score + b)
            .collect();
        let mut remaining: alloc::vec::Vec<usize> = (0..logits.len()).collect();
        let mut routes = alloc::vec::Vec::new();
        for _ in 0..k {
            let winner = *remaining
                .iter()
                .max_by(|&&left, &&right| {
                    selection[left]
                        .partial_cmp(&selection[right])
                        .expect("selection scores are finite")
                })
                .expect("k does not exceed the expert count");
            routes.push(winner);
            remaining.retain(|&candidate| candidate != winner);
        }
        let total: f32 = routes.iter().map(|&expert| scores[expert]).sum();
        routes
            .into_iter()
            .map(|expert| (expert, scores[expert] / total))
            .collect()
    }

    /// [`sigmoid_topk_routes_and_weights`]'s own contract, proved directly
    /// against hand-computed `sigmoid` values before any graph is involved:
    /// a bias large enough to overturn one token's ranking changes which
    /// two experts are selected (proof the bias drives SELECTION), while
    /// the returned weight is always the UNBIASED score (proof the bias
    /// never reaches the weight) -- the two halves of `exp_probs_b`'s own
    /// contract this session closes.
    #[test]
    fn sigmoid_topk_reference_lets_bias_change_selection_but_never_the_weight() {
        let logits = [3.0f32, 2.0, 4.0];
        let sigmoid = |value: f32| 1.0 / (1.0 + (-value).exp());

        let unbiased = sigmoid_topk_routes_and_weights(&logits, &[0.0, 0.0, 0.0], 2);
        let mut unbiased_experts: alloc::vec::Vec<usize> =
            unbiased.iter().map(|(expert, _)| *expert).collect();
        unbiased_experts.sort_unstable();
        assert_eq!(
            unbiased_experts,
            alloc::vec![0, 2],
            "unbiased sigmoid ranking matches raw-logit ranking: e2 > e0 > e1"
        );

        // pushes e1 (raw sigmoid ~0.881) above e0 (raw sigmoid ~0.953) for
        // SELECTION only: 0.881 + 0.2 = 1.081 > 0.953, but e2 (~0.982) still
        // wins outright, so the selected PAIR changes from {e0, e2} to {e1, e2}.
        let biased = sigmoid_topk_routes_and_weights(&logits, &[0.0, 0.2, 0.0], 2);
        let mut biased_experts: alloc::vec::Vec<usize> =
            biased.iter().map(|(expert, _)| *expert).collect();
        biased_experts.sort_unstable();
        assert_eq!(
            biased_experts,
            alloc::vec![1, 2],
            "a large-enough bias on e1 must swap it in for e0"
        );

        let e1_weight = biased
            .iter()
            .find(|(expert, _)| *expert == 1)
            .map(|(_, weight)| *weight)
            .expect("e1 was selected");
        let e2_weight = biased
            .iter()
            .find(|(expert, _)| *expert == 2)
            .map(|(_, weight)| *weight)
            .expect("e2 was selected");
        let expected_e1 = sigmoid(2.0) / (sigmoid(2.0) + sigmoid(4.0));
        let expected_e2 = sigmoid(4.0) / (sigmoid(2.0) + sigmoid(4.0));
        assert!(
            (e1_weight - expected_e1).abs() < 1e-6,
            "e1's weight must be its UNBIASED sigmoid share ({expected_e1}), got {e1_weight} -- the bias must never reach the weight"
        );
        assert!(
            (e2_weight - expected_e2).abs() < 1e-6,
            "e2's weight must be its unbiased sigmoid share ({expected_e2}), got {e2_weight}"
        );
    }

    /// End-to-end proof for [`append_moe_ffn`]'s `Sigmoid` branch, same
    /// shape as [`a_routed_ffn_built_by_append_moe_ffn_matches_an_independent_topk_swiglu_reference`]
    /// (same `x`/`gate_inp`/expert weights, so the same logits `[3, 2, 4]`/
    /// `[1, 4, 8]` this time run through `sigmoid` + a per-expert bias
    /// instead of softmax): token 0's bias (`+0.2` on expert 1) flips its
    /// selected pair from `{0, 2}` (sigmoid-ranking-only) to `{1, 2}` --
    /// proof the graph's OWN `Op::Select` exclusion, not just the
    /// hand-rolled reference above, routes by the biased score. Token 1
    /// keeps the same pair its unbiased ranking already picked, proving the
    /// bias is a per-expert additive term the graph applies uniformly, not
    /// a per-token special case.
    #[test]
    fn a_routed_ffn_built_by_append_moe_ffn_with_sigmoid_gating_and_bias_matches_an_independent_reference()
     {
        const SEQUENCE: usize = 2;
        const EMBEDDING: usize = 2;
        const FEED_FORWARD: usize = 2;
        const EXPERT_COUNT: u32 = 3;
        const EXPERT_USED_COUNT: u32 = 2;

        let x: [f32; SEQUENCE * EMBEDDING] = [3.0, 2.0, 1.0, 4.0];
        let gate_inp: [f32; EMBEDDING * EXPERT_COUNT as usize] = [1.0, 0.0, 0.0, 0.0, 1.0, 2.0];
        let bias: [f32; EXPERT_COUNT as usize] = [0.0, 0.2, 0.0];

        let gate_weights: [[f32; EMBEDDING * FEED_FORWARD]; 3] = [
            [1.0, 0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0, 2.0],
            [1.0, 1.0, 1.0, 1.0],
        ];
        let up_weights: [[f32; EMBEDDING * FEED_FORWARD]; 3] = [
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0, 2.0],
        ];
        let down_weights: [[f32; FEED_FORWARD * EMBEDDING]; 3] = [
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
        ];

        let stack_experts =
            |weights: &[[f32; EMBEDDING * FEED_FORWARD]; 3]| -> alloc::vec::Vec<f32> {
                weights.iter().flatten().copied().collect()
            };
        let expert_w_gate = stack_experts(&gate_weights);
        let expert_w_up = stack_experts(&up_weights);
        let expert_w_down: alloc::vec::Vec<f32> = down_weights.iter().flatten().copied().collect();

        let mut program = Vec::new();
        let x_node = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(0), Extent::Static(EMBEDDING as u32)],
            "x",
        );
        let gate_inp_node = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(EMBEDDING as u32),
                Extent::Static(EXPERT_COUNT)
            ],
            "gate_inp",
        );
        let expert_w_gate_node = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(EXPERT_COUNT),
                Extent::Static(EMBEDDING as u32),
                Extent::Static(FEED_FORWARD as u32)
            ],
            "expert_w_gate",
        );
        let expert_w_up_node = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(EXPERT_COUNT),
                Extent::Static(EMBEDDING as u32),
                Extent::Static(FEED_FORWARD as u32)
            ],
            "expert_w_up",
        );
        let expert_w_down_node = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(EXPERT_COUNT),
                Extent::Static(FEED_FORWARD as u32),
                Extent::Static(EMBEDDING as u32)
            ],
            "expert_w_down",
        );
        let bias_node = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(EXPERT_COUNT)],
            "bias",
        );
        let ones = scalar_constant(&mut program, 1.0);

        let (root, _site) = append_moe_ffn(
            &mut program,
            0,
            x_node,
            gate_inp_node,
            expert_w_gate_node,
            expert_w_up_node,
            expert_w_down_node,
            EXPERT_COUNT,
            EXPERT_USED_COUNT,
            ones,
            ExpertGatingFunc::Sigmoid,
            Some(bias_node),
        )
        .expect("the sigmoid-gated routed ffn lowers");

        let symbols = [SEQUENCE as u64];
        crate::shape::infer(&program, &symbols).expect("the sigmoid-gated routed ffn infers");

        let blocks: [&[f32]; 6] = [
            &x,
            &gate_inp,
            &expert_w_gate,
            &expert_w_up,
            &expert_w_down,
            &bias,
        ];
        let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");
        let evaluated =
            crate::cpu::evaluate_parallel(&program, &symbols, &blocks, &[root], workers)
                .expect("the sigmoid-gated routed ffn evaluates");
        let output = evaluated.root();
        assert_eq!(
            output.len(),
            SEQUENCE * EMBEDDING,
            "a vacuous output proves nothing"
        );

        for (token, x_row) in x.chunks(EMBEDDING).enumerate() {
            let logits: alloc::vec::Vec<f32> = (0..EXPERT_COUNT as usize)
                .map(|expert| {
                    (0..EMBEDDING)
                        .map(|dim| x_row[dim] * gate_inp[dim * EXPERT_COUNT as usize + expert])
                        .sum()
                })
                .collect();
            let routes =
                sigmoid_topk_routes_and_weights(&logits, &bias, EXPERT_USED_COUNT as usize);
            let mut expected = alloc::vec![0.0f32; EMBEDDING];
            for (expert, weight) in &routes {
                let expert_out = swiglu_ffn(
                    x_row,
                    &gate_weights[*expert],
                    &up_weights[*expert],
                    &down_weights[*expert],
                    EMBEDDING,
                    FEED_FORWARD,
                );
                for (accum, value) in expected.iter_mut().zip(&expert_out) {
                    *accum += weight * value;
                }
            }
            let found = &output[token * EMBEDDING..(token + 1) * EMBEDDING];
            for (found_value, expected_value) in found.iter().zip(&expected) {
                assert!(
                    (found_value - expected_value).abs() < 1e-4,
                    "token {token}: got {found:?}, expected {expected:?} (independent sigmoid+bias top-{EXPERT_USED_COUNT} \
                     swiglu reference); routes={routes:?}"
                );
            }
        }
    }

    /// Deterministic, dependency-free pseudo-random `f32` row generator for
    /// the two packed-`Q4_K` tests below -- no RNG crate needed (principle 1:
    /// a `u64` multiply-and-shift is the whole requirement), just enough
    /// spread across `[-scale, scale]` that a wrong-expert read (a different
    /// `scale`, see [`quantized_moe_ffn_over_a_packed_q4k_expert_stack_matches_the_routed_experts_own_swiglu`]'s
    /// `1x`/`5x`/`20x` asymmetric per-expert scales) cannot land on the right
    /// answer by coincidence.
    fn synth_row(seed: u64, len: usize, scale: f32) -> alloc::vec::Vec<f32> {
        (0..len)
            .map(|index| {
                let mixed = seed
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add((index as u64).wrapping_mul(40_503));
                let unit = ((mixed >> 16) & 0xFFFF) as f32 / 65_535.0;
                (unit * 2.0 - 1.0) * scale
            })
            .collect()
    }

    /// Packs `rows` independent `[k]`-length `f32` rows (`k` must be
    /// [`proxima_gguf::quant::q4_k::QK_K`] exactly, one super-block per row)
    /// into one `Q4_K` byte buffer, row-major -- the same physical layout
    /// [`proxima_gguf::restack`] produces when it byte-concatenates a real
    /// GGUF checkpoint's per-expert tensors (see [`crate::cpu::run_reduce_quantized`]'s
    /// own doc on `per_expert_bytes`).
    fn quantize_rows(matrix: &[f32], rows: usize, k: usize) -> alloc::vec::Vec<u8> {
        use proxima_gguf::quant::q4_k::{BLOCK_BYTES, quantize};

        let mut packed = alloc::vec![0u8; rows * BLOCK_BYTES];
        for (row, out_block) in matrix
            .chunks_exact(k)
            .zip(packed.as_chunks_mut::<BLOCK_BYTES>().0)
        {
            quantize(row, out_block).expect("k is QK_K by construction");
        }
        packed
    }

    /// The exact inverse of [`quantize_rows`] -- the "equivalent dequantised
    /// f32 experts" the binder hand-off names: what a caller gets by
    /// dequantizing the packed bytes back to `f32` before handing them to
    /// [`crate::cpu::evaluate_parallel`], the non-quantized evaluator.
    fn dequantize_rows(packed: &[u8], rows: usize, k: usize) -> alloc::vec::Vec<f32> {
        use proxima_gguf::quant::q4_k::dequantize;

        let mut matrix = alloc::vec![0.0f32; rows * k];
        dequantize(packed, &mut matrix).expect("packed rows dequantize");
        matrix
    }

    /// Transposes a `[rows, cols]` row-major matrix into `[cols, rows]`.
    /// Needed because a packed `Q4_K` node's `rows`/`k` split
    /// ([`crate::cpu::run_reduce_quantized`]'s own derivation: `rows` is
    /// whichever axis the weight's `IndexMap` varies over among the
    /// reduce's OUTPUT axes, `k` is the reduced axis's own extent) and an
    /// `Op::Input`'s *declared* axis order (row-major, last axis fastest --
    /// [`matvec`]'s own `matrix[inp * d_out + out]` is the same convention)
    /// are two different, unrelated conventions for the SAME node: the
    /// packed bytes are whatever a real GGUF file's own native layout is
    /// (never read through the ordinary strided path at all), while a
    /// plain `f32` binding of the identical node IS read through it. The
    /// "equivalent dequantised f32 experts" this session's brief asks for
    /// therefore is not simply [`dequantize_rows`]'s own output -- it is
    /// that output transposed into the declared-shape convention.
    fn transpose_rows(matrix: &[f32], rows: usize, cols: usize) -> alloc::vec::Vec<f32> {
        let mut transposed = alloc::vec![0.0f32; rows * cols];
        for row in 0..rows {
            for col in 0..cols {
                transposed[col * rows + row] = matrix[row * cols + col];
            }
        }
        transposed
    }

    /// The crate's own packed-`Q4_K` matmul kernel, whichever one
    /// [`crate::cpu::run_reduce_quantized`]'s own `q4k-int8-dot`-gated arm
    /// would actually call for this build -- comparing the graph's gathered
    /// output against the OTHER kernel would fail on that kernel's own lossy
    /// `Q8_K` activation quantization, not on a wrong-expert read, exactly
    /// [`crate::cpu`]'s own `evaluate_quantized_gathered_moe_weight_matches_the_routed_experts_own_matmul`
    /// avoids.
    fn matmul_q4k_active(weights: &[u8], rows: usize, activation: &[f32]) -> alloc::vec::Vec<f32> {
        #[cfg(feature = "q4k-int8-dot")]
        {
            crate::cpu::matmul_q4k_q8k_f32(weights, rows, activation)
                .expect("packed q4k matmul evaluates")
        }
        #[cfg(not(feature = "q4k-int8-dot"))]
        {
            crate::cpu::matmul_q4k_f32(weights, rows, activation)
                .expect("packed q4k matmul evaluates")
        }
    }

    /// The decisive proof this session's hand-off exists for:
    /// [`gathered_expert_product`] -- unchanged, no widening, the exact
    /// construction [`append_moe_ffn`]'s callers (LFM2, Mistral) already
    /// build -- correctly gathers one expert's `[rows, k]` slab out of a
    /// stacked packed `Q4_K` buffer through the real evaluator
    /// (`evaluate_quantized` -> `run_reduce_quantized`'s gather-resolution
    /// branch), not a hand-rolled duplicate of the construction.
    ///
    /// Three tokens, three DISTINCT experts (`route = [2, 0, 1]`, chosen so
    /// no token's true route is its own position), asymmetric per-expert
    /// scales (`1x`/`5x`/`20x`) so a wrong-expert read is off by that same
    /// factor, not a rounding difference -- this is a mechanism check.
    ///
    /// The discriminator this session's brief calls for lives in the same
    /// function body: the identical program and packed bytes are evaluated a
    /// SECOND time with `route` forced to the constant `[0, 0, 0]` --
    /// proving the assertion below is capable of failing, not just capable
    /// of passing -- then the real route is restored and re-asserted, so the
    /// test ends green.
    #[test]
    fn gathered_expert_product_over_a_packed_q4k_stack_reads_the_routed_experts_own_bytes() {
        use proxima_gguf::quant::q4_k::QK_K;

        const EXPERT_COUNT: u32 = 3;
        const ROWS: usize = 2;
        const SEQUENCE: usize = 3;
        let k = QK_K;

        let expert_scales = [1.0f32, 5.0, 20.0];
        let expert_matrices: alloc::vec::Vec<alloc::vec::Vec<f32>> = expert_scales
            .iter()
            .enumerate()
            .map(|(expert, &scale)| synth_row(101 + expert as u64, ROWS * k, scale))
            .collect();
        let expert_blocks: alloc::vec::Vec<alloc::vec::Vec<u8>> = expert_matrices
            .iter()
            .map(|matrix| quantize_rows(matrix, ROWS, k))
            .collect();
        let stacked_weight: alloc::vec::Vec<u8> = expert_blocks.iter().flatten().copied().collect();

        let activation: alloc::vec::Vec<f32> = synth_row(211, SEQUENCE * k, 1.0);

        let mut program = Vec::new();
        // `[expert_count, k, rows]` -- NOT `[expert_count, rows, k]` --
        // matches [`gathered_expert_product`]'s own `x_map` (`Affine`
        // over iteration axes `0, 1`): the operand's declared axis 1 is
        // what iteration axis 1 (the CONTRACTED `i`/`k` axis `x` also maps
        // its own axis 1 onto) draws its extent from, and axis 2 is the
        // kept, non-reduced `o`/`rows` axis -- the same order
        // [`append_moe_ffn`]'s own `expert_w_gate`/`expert_w_up`
        // (`[expert_count, embedding, feed_forward]`, embedding is the
        // reduced axis) already declare.
        let stack_node = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(EXPERT_COUNT),
                Extent::Static(k as u32),
                Extent::Static(ROWS as u32)
            ],
            "expert_stack",
        );
        // `Int32`, not `Float32` -- the same declared dtype
        // `gathered_quantized_matmul_program` (`cpu.rs`'s own test) uses for
        // a gather-indices node: `reject_non_float32`'s `index_node_ids`
        // exemption keys off usage (any node referenced as a gather's
        // `indices`), not this declaration, but the gather-resolution code
        // itself (`run_reduce_quantized`'s `raw_index = *index_buffer.get(..)`)
        // always reads the bound buffer as `f32` regardless -- `Int32` here
        // is shape-inference metadata, not a storage format.
        let route_node = input_leaf(
            &mut program,
            DType::Int32,
            alloc::vec![Extent::Static(SEQUENCE as u32)],
            "route",
        );
        let x_node = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(SEQUENCE as u32), Extent::Static(k as u32)],
            "x",
        );

        let product = gathered_expert_product(&mut program, stack_node, route_node, x_node);
        let sum = op::append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                // keeps `s` (axis 0) and `rows`/`o` (axis 2); reduces `i`/`k`
                // (axis 1) -- the same "so->sio" shape [`append_moe_ffn`]'s
                // own gate/up reduces use.
                out_map: IndexMap::Affine(map::projection(3, &[0, 2])),
                keep: Keep::Reduce,
                name: Some("gathered_q4k_matmul".into()),
            }),
        );

        let true_route = [2.0f32, 0.0, 1.0];
        let expected: alloc::vec::Vec<f32> = true_route
            .iter()
            .enumerate()
            .flat_map(|(token, &route)| {
                let expert = route as usize;
                let activation_row = &activation[token * k..(token + 1) * k];
                matmul_q4k_active(&expert_blocks[expert], ROWS, activation_row)
            })
            .collect();

        let run = |route_data: &[f32]| -> alloc::vec::Vec<f32> {
            let quantized_blocks = [
                crate::cpu::QuantizedBlock::Q4K(&stacked_weight),
                crate::cpu::QuantizedBlock::Float32(route_data),
                crate::cpu::QuantizedBlock::Float32(&activation),
            ];
            crate::cpu::evaluate_quantized(&program, &[], &quantized_blocks, &[sum])
                .expect("the gathered packed matmul evaluates")
                .root()
                .to_vec()
        };

        let constant_route = [0.0f32, 0.0, 0.0];
        let broken = run(&constant_route);
        assert_ne!(
            broken, expected,
            "forcing every token's route to expert 0 must diverge from the true per-token routing \
             (tokens 0 and 2 route elsewhere) -- if this passed, the assertion below could not be trusted"
        );

        let fixed = run(&true_route);
        assert_eq!(
            fixed, expected,
            "gathered_expert_product over a packed Q4_K stack must read exactly the routed expert's own bytes"
        );
    }

    /// The headline proof: the full [`append_moe_ffn`] graph -- the same
    /// function LFM2's own `blk.{layer}.ffn_gate_exps.weight` call site
    /// (`spec.rs`'s own LFM2 builder) invokes, unmodified -- run over a
    /// stacked packed `Q4_K` expert block reproduces, per token, to within a
    /// single-ULP `f32` rounding bound (see the per-element assertion below
    /// for why this is not literal `assert_eq!`), what an independent
    /// SwiGLU built from the SAME packed bytes' own matmul kernel produces
    /// for that token's TRUE routed expert. Three
    /// tokens, three distinct experts (`x` is one-hot per token; `gate_inp`
    /// is built so token 0 routes to expert 2, token 1 to expert 0, token 2
    /// to expert 1 -- no token routes to its own position, so a routing bug
    /// that reused one token's route for another could not hide), top-1
    /// selection so the softmax combination weight is always exactly `1.0`
    /// and cannot mask a wrong-expert read behind a partial blend.
    ///
    /// This is [`gathered_expert_product_over_a_packed_q4k_stack_reads_the_routed_experts_own_bytes`]'s
    /// same discriminating construction (asymmetric `1x`/`5x`/`20x`
    /// per-expert scales), lifted to the whole FFN graph `append_moe_ffn`
    /// actually builds -- gate, up, SiLU, down, three packed operands, one
    /// per projection -- proving the widening this session's brief asked
    /// for needs no code change: `expert_w_gate`/`expert_w_up`/`expert_w_down`
    /// are declared exactly the way [`append_moe_ffn`]'s real callers
    /// already declare them (`DType::Float32`, `[expert_count, ..]`), and
    /// `evaluate_quantized` binds a `QuantizedBlock::Q4K` to that same node
    /// with no spec-side change at all.
    #[test]
    fn quantized_moe_ffn_over_a_packed_q4k_expert_stack_matches_the_routed_experts_own_swiglu() {
        use proxima_gguf::quant::q4_k::QK_K;

        const EMBEDDING: usize = QK_K;
        const FEED_FORWARD: usize = QK_K;
        const EXPERT_COUNT: u32 = 3;
        const EXPERT_USED_COUNT: u32 = 1;
        const SEQUENCE: usize = 3;

        let expert_scales = [1.0f32, 5.0, 20.0];
        let gate_matrices: alloc::vec::Vec<alloc::vec::Vec<f32>> = expert_scales
            .iter()
            .enumerate()
            .map(|(expert, &scale)| {
                synth_row(1_001 + expert as u64, FEED_FORWARD * EMBEDDING, scale)
            })
            .collect();
        let up_matrices: alloc::vec::Vec<alloc::vec::Vec<f32>> = expert_scales
            .iter()
            .enumerate()
            .map(|(expert, &scale)| {
                synth_row(2_002 + expert as u64, FEED_FORWARD * EMBEDDING, scale)
            })
            .collect();
        let down_matrices: alloc::vec::Vec<alloc::vec::Vec<f32>> = expert_scales
            .iter()
            .enumerate()
            .map(|(expert, &scale)| {
                synth_row(3_003 + expert as u64, EMBEDDING * FEED_FORWARD, scale)
            })
            .collect();

        let gate_blocks: alloc::vec::Vec<alloc::vec::Vec<u8>> = gate_matrices
            .iter()
            .map(|matrix| quantize_rows(matrix, FEED_FORWARD, EMBEDDING))
            .collect();
        let up_blocks: alloc::vec::Vec<alloc::vec::Vec<u8>> = up_matrices
            .iter()
            .map(|matrix| quantize_rows(matrix, FEED_FORWARD, EMBEDDING))
            .collect();
        let down_blocks: alloc::vec::Vec<alloc::vec::Vec<u8>> = down_matrices
            .iter()
            .map(|matrix| quantize_rows(matrix, EMBEDDING, FEED_FORWARD))
            .collect();

        let stacked_gate: alloc::vec::Vec<u8> = gate_blocks.iter().flatten().copied().collect();
        let stacked_up: alloc::vec::Vec<u8> = up_blocks.iter().flatten().copied().collect();
        let stacked_down: alloc::vec::Vec<u8> = down_blocks.iter().flatten().copied().collect();

        // token s is the one-hot vector at index s; gate_inp's rows 0..3
        // make index 0 favor expert 2, index 1 favor expert 0, index 2
        // favor expert 1 -- every other row is zero, so only the one-hot
        // position drives the route.
        let mut x = alloc::vec![0.0f32; SEQUENCE * EMBEDDING];
        for token in 0..SEQUENCE {
            x[token * EMBEDDING + token] = 1.0;
        }
        let true_route = [2usize, 0, 1];
        let mut gate_inp = alloc::vec![0.0f32; EMBEDDING * EXPERT_COUNT as usize];
        for (token, &expert) in true_route.iter().enumerate() {
            gate_inp[token * EXPERT_COUNT as usize + expert] = 5.0;
        }

        let mut program = Vec::new();
        let x_node = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(0), Extent::Static(EMBEDDING as u32)],
            "x",
        );
        let gate_inp_node = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(EMBEDDING as u32),
                Extent::Static(EXPERT_COUNT)
            ],
            "gate_inp",
        );
        let expert_w_gate_node = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(EXPERT_COUNT),
                Extent::Static(EMBEDDING as u32),
                Extent::Static(FEED_FORWARD as u32)
            ],
            "expert_w_gate",
        );
        let expert_w_up_node = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(EXPERT_COUNT),
                Extent::Static(EMBEDDING as u32),
                Extent::Static(FEED_FORWARD as u32)
            ],
            "expert_w_up",
        );
        let expert_w_down_node = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(EXPERT_COUNT),
                Extent::Static(FEED_FORWARD as u32),
                Extent::Static(EMBEDDING as u32)
            ],
            "expert_w_down",
        );
        let ones = scalar_constant(&mut program, 1.0);

        let (root, _site) = append_moe_ffn(
            &mut program,
            0,
            x_node,
            gate_inp_node,
            expert_w_gate_node,
            expert_w_up_node,
            expert_w_down_node,
            EXPERT_COUNT,
            EXPERT_USED_COUNT,
            ones,
            ExpertGatingFunc::Softmax,
            None,
        )
        .expect("the packed-stack routed ffn lowers");

        let symbols = [SEQUENCE as u64];
        crate::shape::infer(&program, &symbols).expect("the packed-stack routed ffn infers");

        let quantized_blocks = [
            crate::cpu::QuantizedBlock::Float32(&x),
            crate::cpu::QuantizedBlock::Float32(&gate_inp),
            crate::cpu::QuantizedBlock::Q4K(&stacked_gate),
            crate::cpu::QuantizedBlock::Q4K(&stacked_up),
            crate::cpu::QuantizedBlock::Q4K(&stacked_down),
        ];
        let evaluated =
            crate::cpu::evaluate_quantized(&program, &symbols, &quantized_blocks, &[root])
                .expect("the packed-stack routed ffn evaluates over Q4_K");
        let output = evaluated.root();
        assert_eq!(
            output.len(),
            SEQUENCE * EMBEDDING,
            "a vacuous output proves nothing"
        );

        for (token, &expert) in true_route.iter().enumerate() {
            let x_row = &x[token * EMBEDDING..(token + 1) * EMBEDDING];
            let gate = matmul_q4k_active(&gate_blocks[expert], FEED_FORWARD, x_row);
            let up = matmul_q4k_active(&up_blocks[expert], FEED_FORWARD, x_row);
            let hidden: alloc::vec::Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(&gate_value, &up_value)| {
                    let silu = gate_value / (1.0 + (-gate_value).exp());
                    silu * up_value
                })
                .collect();
            let expected = matmul_q4k_active(&down_blocks[expert], EMBEDDING, &hidden);

            let found = &output[token * EMBEDDING..(token + 1) * EMBEDDING];
            // A tight relative bound, not `assert_eq!`: at `rows=EMBEDDING=
            // FEED_FORWARD=256` (mandatory -- `Q4_K` needs a whole
            // `QK_K`-multiple contraction axis on BOTH projections), every
            // one of these matmuls clears `PARALLEL_THRESHOLD` (4096 macs;
            // `crate::sized::PARALLEL_THRESHOLD`), so `quantized_matmul_workers`
            // threads the row batch. This direct reference call's own
            // `session: None` and the graph's own internal session context
            // are not the same value, so they are not guaranteed to pick the
            // identical worker chunking -- rows are still computed
            // independently either way (no cross-row summation), so the
            // measured gap tops out at single-ULP `f32` rounding (~1e-7
            // relative, observed), five orders of magnitude below the
            // 1x/5x/20x scale gap a wrong-routed expert would produce. The
            // discriminator test above already proves THIS reads the right
            // expert; this bound proves it reads it correctly.
            for (found_value, expected_value) in found.iter().zip(expected.iter()) {
                let scale = found_value.abs().max(expected_value.abs()).max(1.0);
                assert!(
                    (found_value - expected_value).abs() / scale < 1e-4,
                    "token {token} routed to expert {expert}: append_moe_ffn over a packed Q4_K expert \
                     stack ({found_value}) diverges from that expert's own standalone Q4_K swiglu \
                     ({expected_value}) past single-ULP rounding"
                );
            }
        }

        // Secondary, honestly-labeled sanity check, not the primary
        // correctness gate: the SAME graph run over the DEQUANTIZED f32
        // experts (`evaluate_parallel`, no int8 activation path at all)
        // stays close to the packed-quantized run -- a LOOSE bound, since
        // `q4k-int8-dot`'s own `Q8_K` activation quantization is a real,
        // already-measured, expected source of numerical difference from a
        // naive f32 dot on dequantized weights (see `cpu.rs`'s own
        // `relative_max_diff < 0.01` sanity bound for the dense codec path),
        // not a routing defect this test exists to catch.
        let dequantized_gate: alloc::vec::Vec<f32> = gate_blocks
            .iter()
            .flat_map(|bytes| {
                transpose_rows(
                    &dequantize_rows(bytes, FEED_FORWARD, EMBEDDING),
                    FEED_FORWARD,
                    EMBEDDING,
                )
            })
            .collect();
        let dequantized_up: alloc::vec::Vec<f32> = up_blocks
            .iter()
            .flat_map(|bytes| {
                transpose_rows(
                    &dequantize_rows(bytes, FEED_FORWARD, EMBEDDING),
                    FEED_FORWARD,
                    EMBEDDING,
                )
            })
            .collect();
        let dequantized_down: alloc::vec::Vec<f32> = down_blocks
            .iter()
            .flat_map(|bytes| {
                transpose_rows(
                    &dequantize_rows(bytes, EMBEDDING, FEED_FORWARD),
                    EMBEDDING,
                    FEED_FORWARD,
                )
            })
            .collect();
        let f32_blocks: [&[f32]; 5] = [
            &x,
            &gate_inp,
            &dequantized_gate,
            &dequantized_up,
            &dequantized_down,
        ];
        let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");
        let dequantized_evaluated =
            crate::cpu::evaluate_parallel(&program, &symbols, &f32_blocks, &[root], workers)
                .expect("the dequantized-f32 routed ffn evaluates");
        let dequantized_output = dequantized_evaluated.root();
        for (found_value, dequantized_value) in output.iter().zip(dequantized_output.iter()) {
            let scale = found_value.abs().max(dequantized_value.abs()).max(1.0);
            assert!(
                (found_value - dequantized_value).abs() / scale < 0.05,
                "packed-Q4_K output {found_value} and dequantized-f32 output {dequantized_value} diverge \
                 past Q8_K activation-quantization's own expected error budget"
            );
        }
    }

    /// Independent reference for grouped-query attention: a plain
    /// `q @ k^T` -> causal softmax -> `@ v` over raw f32 slices, with no
    /// dependency on `Op`, `IndexMap`, or anything else the graph under test
    /// builds. `q`/`k`/`v` come from a linear projection (`project`) laid
    /// out the same row-major way `Input`'s declared `shape` implies
    /// (`[dim_in, heads, head_dim]`, slowest axis first) — the one place
    /// this function and the spec's `wq`/`wk`/`wv` shapes must agree, and
    /// the reason both are documented at the call site.
    fn project(
        x: &[f32],
        weight: &[f32],
        sequence: usize,
        dim_in: usize,
        heads: usize,
        head_dim: usize,
    ) -> alloc::vec::Vec<f32> {
        let mut projected = alloc::vec![0.0f32; sequence * heads * head_dim];
        for position in 0..sequence {
            for head in 0..heads {
                for dim in 0..head_dim {
                    let mut accumulator = 0.0f32;
                    for input_dim in 0..dim_in {
                        let activation = x[position * dim_in + input_dim];
                        let coefficient =
                            weight[input_dim * heads * head_dim + head * head_dim + dim];
                        accumulator += activation * coefficient;
                    }
                    projected[(position * heads + head) * head_dim + dim] = accumulator;
                }
            }
        }
        projected
    }

    /// The six sizes one grouped-query-attention case needs, gathered into
    /// one type so `expected_gqa_attended` and `run_gqa_case` each take a
    /// handful of arguments instead of one per size.
    #[derive(Debug, Clone, Copy)]
    struct GqaDims {
        sequence: usize,
        dim_in: usize,
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        group: usize,
    }

    /// `expected[((s*kv_heads+u)*group+g)*head_dim+d]` — the same `sugd`
    /// physical order `gqa_attention.toml`'s `attended` reduce declares in
    /// its `out_map`. `h = u*group + g` is the property under test, spelled
    /// here as plain arithmetic rather than an index map, so the two can
    /// disagree if the graph's addressing is wrong.
    fn expected_gqa_attended(
        x: &[f32],
        wq: &[f32],
        wk: &[f32],
        wv: &[f32],
        dims: GqaDims,
    ) -> alloc::vec::Vec<f32> {
        let GqaDims {
            sequence,
            dim_in,
            query_heads,
            kv_heads,
            head_dim,
            group,
        } = dims;
        let q = project(x, wq, sequence, dim_in, query_heads, head_dim);
        let k = project(x, wk, sequence, dim_in, kv_heads, head_dim);
        let v = project(x, wv, sequence, dim_in, kv_heads, head_dim);

        let mut output = alloc::vec![0.0f32; sequence * kv_heads * group * head_dim];
        for query_position in 0..sequence {
            for kv_head in 0..kv_heads {
                for offset in 0..group {
                    let query_head = kv_head * group + offset;
                    let mut scores = alloc::vec![f32::NEG_INFINITY; sequence];
                    for key_position in 0..=query_position {
                        let mut score = 0.0f32;
                        for dim in 0..head_dim {
                            let query_value =
                                q[(query_position * query_heads + query_head) * head_dim + dim];
                            let key_value = k[(key_position * kv_heads + kv_head) * head_dim + dim];
                            score += query_value * key_value;
                        }
                        scores[key_position] = score;
                    }
                    let max_score = scores.iter().copied().fold(f32::MIN, f32::max);
                    let exponentials: alloc::vec::Vec<f32> = scores
                        .iter()
                        .map(|&score| {
                            if score.is_finite() {
                                (score - max_score).exp()
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    let total: f32 = exponentials.iter().sum();
                    for dim in 0..head_dim {
                        let mut accumulator = 0.0f32;
                        for key_position in 0..sequence {
                            let probability = exponentials[key_position] / total;
                            let value_value =
                                v[(key_position * kv_heads + kv_head) * head_dim + dim];
                            accumulator += probability * value_value;
                        }
                        let index = ((query_position * kv_heads + kv_head) * group + offset)
                            * head_dim
                            + dim;
                        output[index] = accumulator;
                    }
                }
            }
        }
        output
    }

    /// The property that makes this GQA rather than plain multi-head
    /// attention: query heads sharing a kv head must attend against the
    /// *same* k/v head, and query heads in different groups must attend
    /// against *different* ones. `wk`/`wv` give kv head 0 and kv head 1 a
    /// +-10.0 offset on top of independent LCG noise, so a wrong kv-head
    /// selection (e.g. every group reading kv head 0) shows up as an
    /// order-of-magnitude disagreement, not a rounding error — the same
    /// sharpness `a_topk2_probe_...`'s 100-vs-1 weights use.
    ///
    /// `expected_gqa_attended` computes the same arithmetic independently
    /// of the graph, in `sugd` order, so it is compared element by element
    /// against `attended` (the spec's root) rather than read back from any
    /// intermediate the graph produced.
    fn run_gqa_case(text: &str, dims: GqaDims, seed: u64) {
        let GqaDims {
            sequence,
            dim_in,
            query_heads,
            kv_heads,
            head_dim,
            group,
        } = dims;

        let spec: ProgramSpec = toml::from_str(text).expect("spec parses");
        spec.validate().expect("spec is structurally sound");
        let program = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");

        let symbols = [sequence as u64];
        crate::shape::infer(&program, &symbols).expect("the gqa block infers");

        let x = random_vec(seed, sequence * dim_in);
        let wq = random_vec(seed + 1, dim_in * query_heads * head_dim);

        let wk_noise = random_vec(seed + 2, dim_in * kv_heads * head_dim);
        let wv_noise = random_vec(seed + 3, dim_in * kv_heads * head_dim);
        let mut wk = alloc::vec![0.0f32; dim_in * kv_heads * head_dim];
        let mut wv = alloc::vec![0.0f32; dim_in * kv_heads * head_dim];
        for input_dim in 0..dim_in {
            for kv_head in 0..kv_heads {
                let bias = if kv_head == 0 { 10.0 } else { -10.0 };
                for dim in 0..head_dim {
                    let index = input_dim * kv_heads * head_dim + kv_head * head_dim + dim;
                    wk[index] = wk_noise[index] + bias;
                    wv[index] = wv_noise[index] + bias;
                }
            }
        }

        // `group_ones` only pins `q_grouped`'s (kv-head, group) extents for
        // `shape::infer` (see `gqa_attention.toml`'s header) — it must stay
        // exactly 1.0 or it would silently rescale every query head's score.
        let group_ones = alloc::vec![1.0f32; kv_heads * group];

        let probabilities = spec
            .node
            .iter()
            .position(|node| node.id() == "probabilities")
            .expect("the spec defines a probabilities node");
        let probabilities = NodeId(probabilities as u32);
        let root = NodeId(program.len() as u32 - 1);

        let blocks: [&[f32]; 5] = [&x, &wq, &wk, &wv, &group_ones];
        let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");
        let evaluated = crate::cpu::evaluate_parallel(
            &program,
            &symbols,
            &blocks,
            &[root, probabilities],
            workers,
        )
        .expect("the gqa block evaluates");

        let output = evaluated.root();
        let expected_len = sequence * kv_heads * group * head_dim;
        assert_eq!(
            output.len(),
            expected_len,
            "a vacuous output proves nothing"
        );
        assert!(
            output.iter().all(|value| value.is_finite()),
            "output must be finite"
        );

        let expected = expected_gqa_attended(&x, &wq, &wk, &wv, dims);
        assert_eq!(expected.len(), expected_len);

        let mut compared = 0usize;
        for (index, (&found, &wanted)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (found - wanted).abs() < 1e-3,
                "element {index}: graph produced {found}, independent reference produced \
                 {wanted} — a query head is attending against the wrong kv head"
            );
            compared += 1;
        }
        assert_eq!(
            compared, expected_len,
            "every element must be checked, not a subset"
        );

        let (rows, _) = evaluated
            .get(probabilities)
            .expect("probabilities were requested");
        assert_eq!(rows.len(), sequence * sequence * kv_heads * group);

        let mut checked = 0usize;
        for query_position in 0..sequence {
            for kv_head in 0..kv_heads {
                for offset in 0..group {
                    let mut total = 0.0f32;
                    for key_position in 0..sequence {
                        let index = ((query_position * sequence + key_position) * kv_heads
                            + kv_head)
                            * group
                            + offset;
                        let probability = rows[index];
                        if key_position > query_position {
                            assert_eq!(
                                probability, 0.0,
                                "query {query_position} kv-head {kv_head} group-offset \
                                 {offset} key {key_position} is strictly upper-triangular \
                                 and must be masked to exactly 0.0, found {probability}"
                            );
                        }
                        total += probability;
                        checked += 1;
                    }
                    assert!(
                        (total - 1.0).abs() < 1e-5,
                        "query {query_position} kv-head {kv_head} group-offset {offset} \
                         softmax row sums to {total}, not 1.0"
                    );
                }
            }
        }
        assert_eq!(
            checked,
            sequence * sequence * kv_heads * group,
            "every probability cell must be checked, not a subset"
        );
    }

    #[test]
    fn a_gqa_attention_block_groups_query_heads_onto_shared_kv_heads() {
        let text = include_str!("../specs/gqa_attention.toml");
        let dims = GqaDims {
            sequence: 4,
            dim_in: 4,
            query_heads: 4,
            kv_heads: 2,
            head_dim: 4,
            group: 2,
        };
        run_gqa_case(text, dims, 31);
    }

    /// `deepseek-coder-33b` is `head_count=56`, `head_count_kv=8` — group 7,
    /// not a power of two. This is that shape at a hand-checkable size (6
    /// query heads, 2 kv heads, group 3): the only spec change from
    /// `gqa_attention.toml` is `wq`'s head extent and the affine
    /// coefficient (`2*u+g` -> `3*u+g`), so this test is the check that
    /// `coeff=3` behaves identically to `coeff=2`, not an assumption resting
    /// on the power-of-two case alone.
    #[test]
    fn a_gqa_attention_block_with_a_non_power_of_two_group_groups_query_heads_onto_shared_kv_heads()
    {
        let text = include_str!("../specs/gqa_attention_group3.toml");
        let dims = GqaDims {
            sequence: 4,
            dim_in: 4,
            query_heads: 6,
            kv_heads: 2,
            head_dim: 4,
            group: 3,
        };
        run_gqa_case(text, dims, 41);
    }

    /// The regression this fix exists for: `1/sqrt(head_dim)` missing from
    /// `scores` before the mask does not fail `sums to 1.0` — a saturated
    /// softmax is still a valid softmax — so that invariant alone cannot
    /// catch it. This builds the same score/scale/softmax composition
    /// `append_mistral_layer` now runs (`q . k`, multiply by
    /// [`scalar_constant`], then max-shift/exp/normalize, no mask
    /// — masking is `causal_attention.toml`'s own proven concern, not this
    /// one's), at the model's real `head_dim=128`, built TWICE on the same
    /// `q`/`k`: once with the scaling step omitted entirely (exactly the
    /// pre-fix graph — before this fix `scores` fed the mask directly, which
    /// is what `unscaled` below reproduces) and once with it present, using
    /// the actual production helper rather than a hand-rolled stand-in.
    ///
    /// `q` is the all-ones vector and key 0 is `0.15 * q` (dot product
    /// `0.15 * 128 = 19.2` exactly, no estimation); the other 15 keys are
    /// all-zero (dot product `0.0` exactly). Chosen, not randomly sampled,
    /// so the separation is provable arithmetic: unscaled, `exp(19.2)` so
    /// overwhelms `15 * exp(0)` that key 0 takes essentially the whole
    /// distribution; scaled by `1/sqrt(128)`, the same score drops to
    /// `1.697`, and `exp(1.697) = 5.46` split against `15 * exp(0) = 15`
    /// cannot exceed half the row.
    ///
    /// The assertion a plain "sums to 1.0" check would have missed: the
    /// unscaled row's largest weight must be near-one-hot (`> 0.9`) and the
    /// scaled row's must not (`< 0.5`) — both rows still sum to `1.0`, so
    /// only a degeneracy check, not a normalization check, tells them apart.
    #[test]
    fn scaling_attention_scores_by_inverse_sqrt_head_dim_prevents_softmax_saturation() {
        const HEAD_DIM: usize = 128;
        const KEYS: usize = 16;
        const KEY_ZERO_WEIGHT: f32 = 0.15;

        fn build(scaled: bool) -> (Vec<Op>, NodeId) {
            let mut program = Vec::new();
            let query = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Symbolic(0), Extent::Static(HEAD_DIM as u32)],
                "q",
            );
            let key = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(KEYS as u32), Extent::Static(HEAD_DIM as u32)],
                "k",
            );

            let score_product = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(query, "sh->sth"), (key, "th->sth")],
            )
            .expect("score product builds");
            let scores = reduce(
                &mut program,
                DType::Float32,
                ScalarOp::Add,
                ReduceInit::Zero,
                score_product,
                "sth->sth",
                "st->sth",
            )
            .expect("scores reduce builds");
            let scores = if scaled {
                let inv_sqrt_head_dim =
                    scalar_constant(&mut program, 1.0 / (HEAD_DIM as f32).sqrt());
                elementwise(
                    &mut program,
                    DType::Float32,
                    ScalarOp::Multiply,
                    &[(scores, "st->st"), (inv_sqrt_head_dim, "->st")],
                )
                .expect("scaling multiply builds")
            } else {
                scores
            };
            let score_max = reduce(
                &mut program,
                DType::Float32,
                ScalarOp::Maximum,
                ReduceInit::NegativeInfinity,
                scores,
                "st->st",
                "s->st",
            )
            .expect("max reduce builds");
            let shifted = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Subtract,
                &[(scores, "st->st"), (score_max, "s->st")],
            )
            .expect("shift builds");
            let weights = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Exponential,
                &[(shifted, "st->st")],
            )
            .expect("exponential builds");
            let weight_sum = reduce(
                &mut program,
                DType::Float32,
                ScalarOp::Add,
                ReduceInit::Zero,
                weights,
                "st->st",
                "s->st",
            )
            .expect("weight sum reduce builds");
            let inv_weight_sum = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Reciprocal,
                &[(weight_sum, "s->s")],
            )
            .expect("reciprocal builds");
            let probabilities = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Multiply,
                &[(weights, "st->st"), (inv_weight_sum, "s->st")],
            )
            .expect("probabilities multiply builds");
            (program, probabilities)
        }

        let query_vector = alloc::vec![1.0f32; HEAD_DIM];
        let mut key_vectors = alloc::vec![0.0f32; KEYS * HEAD_DIM];
        key_vectors[0..HEAD_DIM].fill(KEY_ZERO_WEIGHT);
        let symbols = [1u64];
        let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");

        let evaluate = |scaled: bool| -> Vec<f32> {
            let (program, probabilities) = build(scaled);
            crate::shape::infer(&program, &symbols)
                .expect("the isolated score/softmax slice infers");
            let root = NodeId(program.len() as u32 - 1);
            assert_eq!(
                root, probabilities,
                "probabilities is the program's own last node"
            );
            let blocks: [&[f32]; 2] = [&query_vector, &key_vectors];
            let evaluated =
                crate::cpu::evaluate_parallel(&program, &symbols, &blocks, &[root], workers)
                    .expect("the isolated score/softmax slice evaluates");
            evaluated.root().to_vec()
        };

        let unscaled = evaluate(false);
        let scaled = evaluate(true);

        for (label, row) in [
            ("unscaled (pre-fix)", &unscaled),
            ("scaled (post-fix)", &scaled),
        ] {
            let total: f32 = row.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-4,
                "{label} softmax row sums to {total}, not 1.0"
            );
        }

        let unscaled_max = unscaled[0];
        let scaled_max = scaled[0];
        assert!(
            unscaled_max > 0.9,
            "pre-fix (no scaling) softmax should saturate toward one-hot over head_dim={HEAD_DIM} \
             (key 0's raw score is {} against 15 keys at 0.0), but key 0's weight is only \
             {unscaled_max} — the test data no longer reproduces the bug this regression test \
             exists to catch",
            KEY_ZERO_WEIGHT * HEAD_DIM as f32
        );
        assert!(
            scaled_max < 0.5,
            "post-fix (scaled by 1/sqrt(head_dim)) softmax should blend across {KEYS} keys instead \
             of collapsing to one, but key 0's weight is {scaled_max}, no better than the unscaled \
             {unscaled_max} — 1/sqrt(head_dim) is not doing its job"
        );
    }

    /// The milestone this crate has been building toward: one real
    /// openchat-3.5-1210 / Mistral-7B transformer layer, RoPE + GQA +
    /// causal mask composed together, at the model's own dimensions
    /// (`embedding_length=4096`, `head_count=32`, `head_count_kv=8`,
    /// `head_dim=128`, `feed_forward_length=14336`) — not a toy shrink of
    /// them. `mistral_layer.toml`'s header records the one addressing
    /// decision the composition forced: the rotated dot product is
    /// recovered as even-pairs-plus-odd-pairs rather than re-interleaved,
    /// because interleaving needs a write-placement op this crate does not
    /// have. No new `Op` or `ScalarOp` was needed here; if one had been,
    /// this comment would say so instead.
    ///
    /// Shape inference is cheap — symbolic arithmetic over extents, not
    /// data — and runs here at the model's real context length (8192) and
    /// again at the small sequence length the evaluation test below uses,
    /// proving the same graph types at both.
    ///
    /// Evaluating this spec at its real embedding/feed-forward dimensions
    /// was tried and MEASURED, not assumed to be fine: random weight
    /// generation (~870MB across `wq`/`wk`/`wv`/`wo`/`w_gate`/`w_up`/
    /// `w_down`) took 2.77s, but `evaluate_parallel` itself did not finish
    /// inside a 90s budget even at `SEQUENCE=4` — the elementwise nodes
    /// feeding `gate_product`/`up_product`/`down_product` materialize a
    /// full `seq * embedding * feed_forward` product ahead of their reduce
    /// (`seq=4` gives `4 * 4096 * 14336` = 235M elements, ~940MB, per node,
    /// three of them), independent of how small `seq` is. So the evaluation
    /// test below runs `mistral_layer_small.toml` instead — node-for-node
    /// the same file with every non-sequence axis divided down while
    /// preserving the real ratios (GQA group stays 4, RoPE still rotates
    /// the full head_dim) — see that file's header for the exact numbers.
    #[test]
    fn a_mistral_layer_written_as_toml_infers_at_its_real_dimensions() {
        const REAL_CONTEXT: u64 = 8192;
        const SMALL_SEQUENCE: u64 = 4;

        let text = include_str!("../specs/mistral_layer.toml");
        let spec: ProgramSpec = toml::from_str(text).expect("spec parses");
        spec.validate().expect("spec is structurally sound");
        let program = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");

        crate::shape::infer(&program, &[REAL_CONTEXT])
            .expect("the layer infers at its real context length");
        crate::shape::infer(&program, &[SMALL_SEQUENCE])
            .expect("the layer infers at a small sequence length too");
    }

    /// Wall-clock probe for `bind.rs`'s reduce-fusion cost fix: runs
    /// `mistral_layer.toml` at the model's real dimensions
    /// (`embedding=4096`, `feed_forward=14336`) at `sequence=4`, the exact
    /// configuration the sibling milestone test above found too slow to run
    /// unfused (`ffn_out`'s reduce absorbing the whole SwiGLU activation
    /// chain recomputed it once per `embedding` element instead of once per
    /// its own `seq*feed_forward`). `#[ignore]`d — ~870MB of random weights
    /// plus a multi-second real run does not belong in the default
    /// `nextest` budget; run explicitly with `--ignored` when re-measuring.
    #[test]
    #[ignore = "measures the real-dimension mistral layer's wall clock; run explicitly"]
    fn a_mistral_layer_written_as_toml_evaluates_at_its_real_dimensions() {
        const SEQUENCE: usize = 4;
        const EMBEDDING: usize = 4096;
        const QUERY_HEADS: usize = 32;
        const KV_HEADS: usize = 8;
        const HEAD_DIM: usize = 128;
        const PAIRS: usize = HEAD_DIM / 2;
        const GROUP: usize = QUERY_HEADS / KV_HEADS;
        const FEED_FORWARD: usize = 14336;

        let text = include_str!("../specs/mistral_layer.toml");
        let spec: ProgramSpec = toml::from_str(text).expect("spec parses");
        spec.validate().expect("spec is structurally sound");
        let program = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");

        let symbols = [SEQUENCE as u64];
        let shapes = crate::shape::infer(&program, &symbols).expect("the real layer infers");

        let activations = random_vec(101, SEQUENCE * EMBEDDING);
        let epsilon = alloc::vec![1e-5f32; SEQUENCE];
        let wq = random_vec(102, EMBEDDING * QUERY_HEADS * HEAD_DIM);
        let wk = random_vec(103, EMBEDDING * KV_HEADS * HEAD_DIM);
        let wv = random_vec(104, EMBEDDING * KV_HEADS * HEAD_DIM);
        let wo = random_vec(105, KV_HEADS * GROUP * HEAD_DIM * EMBEDDING);
        let w_gate = random_vec(106, EMBEDDING * FEED_FORWARD);
        let w_up = random_vec(107, EMBEDDING * FEED_FORWARD);
        let w_down = random_vec(108, FEED_FORWARD * EMBEDDING);
        let cos = random_vec(109, SEQUENCE * PAIRS);
        let sin = random_vec(110, SEQUENCE * PAIRS);
        let attn_norm_weight = alloc::vec![1.0f32; EMBEDDING];
        let ffn_norm_weight = alloc::vec![1.0f32; EMBEDDING];

        let blocks: [&[f32]; 13] = [
            &activations,
            &epsilon,
            &wq,
            &wk,
            &wv,
            &wo,
            &w_gate,
            &w_up,
            &w_down,
            &cos,
            &sin,
            &attn_norm_weight,
            &ffn_norm_weight,
        ];

        let ffn_out = spec
            .node
            .iter()
            .position(|node| node.id() == "ffn_out")
            .expect("the spec defines an ffn_out node");
        let ffn_out = NodeId(ffn_out as u32);
        let root = NodeId(program.len() as u32 - 1);

        let bound =
            crate::bind::bind(&program, &shapes, &[root, ffn_out], crate::numeric::NumericPolicy::bit_exact()).expect("the real layer binds");
        let ffn_out_body_steps = bound
            .iter()
            .find(|op| op.node == ffn_out)
            .expect("ffn_out is a bound op")
            .element_body()
            .steps
            .len();
        std::println!("ffn_out body_steps={ffn_out_body_steps}");

        let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");
        let wall_start = std::time::Instant::now();
        let evaluated =
            crate::cpu::evaluate_parallel(&program, &symbols, &blocks, &[root], workers)
                .expect("the real mistral layer evaluates");
        let wall = wall_start.elapsed();
        std::println!("wall_clock={wall:?}");

        let output = evaluated.root();
        assert_eq!(
            output.len(),
            SEQUENCE * EMBEDDING,
            "a vacuous output proves nothing"
        );
        assert!(
            output.iter().all(|value| value.is_finite()),
            "output must be finite"
        );
    }

    /// The evaluation half of the milestone above: `mistral_layer_small.toml`
    /// is the same RoPE+GQA+causal-mask composition, small enough to
    /// actually run (see the sibling test's doc comment and that file's
    /// header for why). Two invariants, not just finiteness: the output is
    /// the right shape and every value is finite, and every softmax row —
    /// indexed explicitly because `probabilities`'s `(query, key, kv_head,
    /// group_offset)` layout makes a key-axis row a strided read, not a
    /// contiguous one, the same way `run_gqa_case` above handles it — sums
    /// to 1.0.
    #[test]
    fn a_mistral_layer_written_as_toml_evaluates() {
        const SEQUENCE: usize = 4;
        const EMBEDDING: usize = 16;
        const QUERY_HEADS: usize = 8;
        const KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 4;
        const PAIRS: usize = HEAD_DIM / 2;
        const GROUP: usize = QUERY_HEADS / KV_HEADS;
        const FEED_FORWARD: usize = 32;

        let text = include_str!("../specs/mistral_layer_small.toml");
        let spec: ProgramSpec = toml::from_str(text).expect("spec parses");
        spec.validate().expect("spec is structurally sound");
        let program = Vec::<Op>::try_from(&spec).expect("spec lowers to a program");

        let symbols = [SEQUENCE as u64];
        crate::shape::infer(&program, &symbols).expect("the small layer infers");

        let activations = random_vec(101, SEQUENCE * EMBEDDING);
        let epsilon = alloc::vec![1e-5f32; SEQUENCE];
        let wq = random_vec(102, EMBEDDING * QUERY_HEADS * HEAD_DIM);
        let wk = random_vec(103, EMBEDDING * KV_HEADS * HEAD_DIM);
        let wv = random_vec(104, EMBEDDING * KV_HEADS * HEAD_DIM);
        let wo = random_vec(105, KV_HEADS * GROUP * HEAD_DIM * EMBEDDING);
        let w_gate = random_vec(106, EMBEDDING * FEED_FORWARD);
        let w_up = random_vec(107, EMBEDDING * FEED_FORWARD);
        let w_down = random_vec(108, FEED_FORWARD * EMBEDDING);
        let cos = random_vec(109, SEQUENCE * PAIRS);
        let sin = random_vec(110, SEQUENCE * PAIRS);

        let blocks: [&[f32]; 11] = [
            &activations,
            &epsilon,
            &wq,
            &wk,
            &wv,
            &wo,
            &w_gate,
            &w_up,
            &w_down,
            &cos,
            &sin,
        ];

        let probabilities = spec
            .node
            .iter()
            .position(|node| node.id() == "probabilities")
            .expect("the spec defines a probabilities node");
        let probabilities = NodeId(probabilities as u32);
        let root = NodeId(program.len() as u32 - 1);

        let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");
        let evaluated = crate::cpu::evaluate_parallel(
            &program,
            &symbols,
            &blocks,
            &[root, probabilities],
            workers,
        )
        .expect("the small mistral layer evaluates");

        let output = evaluated.root();
        assert_eq!(
            output.len(),
            SEQUENCE * EMBEDDING,
            "a vacuous output proves nothing"
        );
        assert!(
            output.iter().all(|value| value.is_finite()),
            "output must be finite"
        );

        let (rows, _) = evaluated
            .get(probabilities)
            .expect("probabilities were requested");
        assert_eq!(rows.len(), SEQUENCE * SEQUENCE * KV_HEADS * GROUP);

        // probabilities is laid out `(query, key, kv_head, group_offset)`
        // row-major, so a softmax "row" over the key axis is not a
        // contiguous slice — index it explicitly, the same way
        // `run_gqa_case` above does for the same `stug` iteration order.
        let mut checked = 0usize;
        for query_position in 0..SEQUENCE {
            for kv_head in 0..KV_HEADS {
                for offset in 0..GROUP {
                    let mut total = 0.0f32;
                    for key_position in 0..SEQUENCE {
                        let index = ((query_position * SEQUENCE + key_position) * KV_HEADS
                            + kv_head)
                            * GROUP
                            + offset;
                        total += rows[index];
                        checked += 1;
                    }
                    assert!(
                        (total - 1.0).abs() < 1e-4,
                        "query {query_position} kv-head {kv_head} group-offset {offset} \
                         softmax row sums to {total}, not 1.0"
                    );
                }
            }
        }
        assert_eq!(
            checked,
            SEQUENCE * SEQUENCE * KV_HEADS * GROUP,
            "every probability cell must be checked, not a subset"
        );
    }

    /// The whole model, built as a program instead of authored as 32 copies
    /// of one TOML file: token embedding lookup, `block_count` layers (each
    /// [`append_mistral_layer`], mirroring `specs/mistral_layer.toml`), a
    /// final RMSNorm, and the LM head projection to `[seq, vocab]` logits.
    /// Shape inference is symbolic arithmetic over extents, not data — cheap
    /// enough to run unignored even at the model's real context length,
    /// matching `a_mistral_layer_written_as_toml_infers_at_its_real_dimensions`
    /// above for one layer.
    /// The contract the [`Op::Constant`] variant exists to hold: a literal
    /// is a node, so the only names crossing the binding surface are data
    /// (`ids`), model weights, position tables (`rope_cos`/`rope_sin`), and
    /// the one piece of model metadata this function's `u32` parameters do
    /// not carry (`eps`). `inv_dim`, `ones` and `group_ones` were bound
    /// `Input`s that `proxima-model-interop`'s `bind.rs` filled with a
    /// repeated scalar on every call; each was a name two files had to agree
    /// on forever, which is the drift class this asserts is gone.
    #[test]
    fn no_repeated_scalar_crosses_the_binding_surface() {
        let program = mistral_forward_program(128, 64, 172, 8, 4, 16, 2, 0, 0)
            .expect("the forward pass lowers to a program");

        let bound: Vec<&str> = program
            .iter()
            .filter_map(|expr| match expr {
                Op::Input { .. } => expr.name(),
                _ => None,
            })
            .collect();

        for collapsed in [
            "inv_dim",
            "ones",
            "group_ones",
            "inv_sqrt_head_dim",
            "neg_infinity",
        ] {
            assert!(
                !bound.contains(&collapsed),
                "{collapsed} is a literal and must be an Op::Constant, not a bound Input; \
                 bound names are {bound:?}"
            );
        }

        assert!(bound.contains(&"eps"), "eps is model metadata, still bound");
        assert!(bound.contains(&"ids"), "ids is per-call data, still bound");
        assert!(
            bound.contains(&"rope_cos"),
            "rope_cos varies with position, still bound"
        );
    }

    /// `op = "constant"` is the TOML face of [`Op::Constant`], and
    /// `shape = []` is the rank-0 spelling every scalar literal uses.
    #[test]
    fn a_constant_node_reads_from_toml_with_its_literal_and_shape() {
        const TOML: &str = r#"
[[node]]
op = "constant"
id = "eps"
dtype = "float32"
shape = []
value = 1e-5

[[node]]
op = "constant"
id = "group_ones"
dtype = "float32"
shape = [4, 2]
value = 1.0
"#;
        let spec: ProgramSpec = toml::from_str(TOML).expect("constant nodes parse");
        let program = Vec::<Op>::try_from(&spec).expect("constant nodes lower");

        assert_eq!(
            program[0],
            Op::Constant {
                dtype: DType::Float32,
                shape: Vec::new(),
                value: 1e-5,
            }
        );
        assert_eq!(
            program[1],
            Op::Constant {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(4), Extent::Static(2)],
                value: 1.0,
            }
        );
    }

    #[test]
    fn the_whole_mistral_forward_pass_infers_at_real_dimensions() {
        const REAL_CONTEXT: u64 = 8192;

        let build_start = std::time::Instant::now();
        let program = mistral_forward_program(32_002, 4096, 14336, 32, 8, 128, 32, 0, 0)
            .expect("the whole forward pass lowers to a program");
        let build_elapsed = build_start.elapsed();

        let infer_start = std::time::Instant::now();
        crate::shape::infer(&program, &[REAL_CONTEXT])
            .expect("the whole forward pass infers at its real context length");
        let infer_elapsed = infer_start.elapsed();

        std::println!(
            "mistral_forward_program: nodes={} build={build_elapsed:?} infer={infer_elapsed:?}",
            program.len()
        );
        assert!(
            program.len() > 2_000,
            "32 layers of dozens of nodes plus embedding/lm-head should be thousands of nodes, not {}",
            program.len()
        );
    }

    /// Node-count budget for the chunked key/value fold, established
    /// BEFORE that fold is built. [`mistral_cached_forward_program`] binds
    /// ONE cache buffer per layer sized by `Extent::Symbolic(1)`, so it is
    /// already the one-chunk case of an N-chunk fold. Splitting the cache
    /// into N fixed chunks replicates, per layer per chunk, the twelve
    /// cache-reading nodes in `append_mistral_cached_layer`
    /// (`score_cached_even_product`, `score_cached_even`,
    /// `score_cached_odd_product`, `score_cached_odd`, `score_cached`,
    /// `score_cached_scaled`, `score_max_cached`, `shifted_cached`,
    /// `weights_cached`, `sum_cached`, `attended_cached_product`,
    /// `attended_cached`), plus that chunk's own three `kv_cache.*`
    /// `Op::Input` leaves, plus three combine nodes -- one `Maximum` into
    /// `global_max`, one `Add` into `weight_sum`, one `Add` into
    /// `attended_sum`. Eighteen nodes per chunk per layer.
    ///
    /// The printed `per_chunk_per_model` figure is what a caller multiplies
    /// by its chunk count to decide whether an N-chunk fold can be flat
    /// graph nodes at all. It cannot, past a low N: the fold has to iterate
    /// chunks inside one reduce rather than have the program name each one.
    #[test]
    fn the_chunked_cache_fold_node_budget_is_measured_before_it_is_built() {
        let nodes_of = |block_count: u32| {
            mistral_cached_forward_program(32_002, 4096, 14336, 32, 8, 128, block_count)
                .expect("the cached forward pass lowers to a program")
                .0
                .len()
        };
        let per_layer = nodes_of(2) - nodes_of(1);
        let full = nodes_of(32);
        let uncached = mistral_forward_program(32_002, 4096, 14336, 32, 8, 128, 32, 0, 0)
            .expect("the whole forward pass lowers to a program")
            .len();
        // a built `Op` is not an executed op: `crate::bind` fuses
        // elementwise chains, so the graph the evaluator walks is smaller
        // than the program. Both counts are printed because the chunk
        // budget is built in program nodes and paid in bound ops.
        let (cached_program, cached_logits, cached_roots) =
            mistral_cached_forward_program(32_002, 4096, 14336, 32, 8, 128, 32)
                .expect("the cached forward pass lowers to a program");
        let mut cached_outputs = alloc::vec![cached_logits];
        for (even, odd, value) in &cached_roots {
            cached_outputs.extend_from_slice(&[*even, *odd, *value]);
        }
        let cached_shapes = crate::shape::infer(&cached_program, &[1, 71])
            .expect("one new position against a 71-position cache infers");
        let bound = crate::bind::bind(&cached_program, &cached_shapes, &cached_outputs, crate::numeric::NumericPolicy::bit_exact())
            .expect("the cached program binds")
            .len();

        const CACHE_READING_NODES: usize = 12;
        const CACHE_INPUT_LEAVES: usize = 3;
        const COMBINE_NODES: usize = 3;
        const PER_CHUNK_PER_LAYER: usize = CACHE_READING_NODES + CACHE_INPUT_LEAVES + COMBINE_NODES;
        const LAYERS: usize = 32;

        std::println!(
            "cached_fold_budget uncached_nodes={uncached} cached_nodes={full} cached_per_layer={per_layer} bound_ops_at_ctx71={bound} reduces_per_chunk_per_layer=5 per_chunk_per_layer={PER_CHUNK_PER_LAYER} per_chunk_per_model={}",
            PER_CHUNK_PER_LAYER * LAYERS
        );
        for chunks in [1_usize, 4, 16, 64, 256, 1024, 4096] {
            std::println!(
                "cached_fold_budget chunks={chunks} context_at_chunk_256={} added_nodes={} total_nodes={}",
                chunks * 256,
                (chunks - 1) * PER_CHUNK_PER_LAYER * LAYERS,
                full + (chunks - 1) * PER_CHUNK_PER_LAYER * LAYERS
            );
        }

        assert!(
            PER_CHUNK_PER_LAYER < per_layer,
            "a chunk replicates only the cache-reading part of a layer, never the whole {per_layer}-node layer"
        );
    }

    /// [`the_chunked_cache_fold_node_budget_is_measured_before_it_is_built`]'s
    /// single-range counterpart: [`append_mistral_single_range_cached_layer`]
    /// deletes the eighteen cache-reading nodes that test documents
    /// (`CACHE_READING_NODES=12` plus three combine nodes counted
    /// separately there) and replaces them with nothing -- there is no
    /// second block left to combine, so the eighteen-node-per-chunk cost
    /// this budget exists to warn about does not apply to the single-range
    /// path at all. Old->new, per layer: 83 raw `Op`s (baseline, matching
    /// [`append_mistral_cached_layer`]'s own doc) -> whatever `per_layer`
    /// prints below, deleting the 6-op cached score block
    /// (`score_cached_even_product`..`score_cached_scaled`) and the
    /// 17-op online-softmax combine (`score_max_cached`..`attended`),
    /// adding back an 8-op single-pass softmax
    /// (`score_max`,`shifted`,`weights`,`weight_sum`,`inv_weight_sum`,
    /// `probabilities`,`attended_product`,`attended`) node-for-node
    /// [`append_mistral_layer`]'s own pattern.
    #[test]
    fn the_single_range_cache_fold_node_budget_is_measured_against_the_two_range_baseline() {
        let two_range_nodes_of = |block_count: u32| {
            mistral_cached_forward_program(32_002, 4096, 14336, 32, 8, 128, block_count)
                .expect("the two-range cached forward pass lowers to a program")
                .0
                .len()
        };
        let single_range_nodes_of = |block_count: u32| {
            mistral_single_range_cached_forward_program(
                32_002,
                4096,
                14336,
                32,
                8,
                128,
                block_count,
                false,
                DuplicateHeadPosition::None,
            )
            .expect("the single-range cached forward pass lowers to a program")
            .0
            .len()
        };
        let two_range_per_layer = two_range_nodes_of(2) - two_range_nodes_of(1);
        let single_range_per_layer = single_range_nodes_of(2) - single_range_nodes_of(1);

        let (two_range_program, two_range_logits, two_range_roots) =
            mistral_cached_forward_program(32_002, 4096, 14336, 32, 8, 128, 32)
                .expect("the two-range cached forward pass lowers to a program");
        let mut two_range_outputs = alloc::vec![two_range_logits];
        for (even, odd, value) in &two_range_roots {
            two_range_outputs.extend_from_slice(&[*even, *odd, *value]);
        }
        let two_range_shapes = crate::shape::infer(&two_range_program, &[1, 71])
            .expect("one new position against a 71-position cache infers");
        // fusion held explicitly off: `bind`'s default `fuse_cached_attention
        // = true` (under `cached-attention-streaming`) collapses the
        // two-range baseline's online-softmax combine into `CachedAttention`
        // BoundOps but has no candidate to fuse on the single-range path
        // below, so an unpinned `bind` call here compares a fused count
        // against an unfused one instead of the structural raw-bind
        // difference this test names.
        let two_range_bound = crate::bind::bind_with_fusion(
            &two_range_program,
            &two_range_shapes,
            &two_range_outputs,
            false,
            crate::numeric::NumericPolicy::default(),
        )
        .expect("the two-range cached program binds")
        .len();

        let (single_range_program, single_range_logits, single_range_roots, _) =
            mistral_single_range_cached_forward_program(
                32_002,
                4096,
                14336,
                32,
                8,
                128,
                32,
                false,
                DuplicateHeadPosition::None,
            )
            .expect("the single-range cached forward pass lowers to a program");
        let mut single_range_outputs = alloc::vec![single_range_logits];
        for (even, odd, value) in &single_range_roots {
            single_range_outputs.extend_from_slice(&[*even, *odd, *value]);
        }
        // symbol 1 here is the MERGED length -- 71 total context, matching
        // the two-range baseline's 71-position existing cache plus its own
        // one new position folded in, so both measurements are read at the
        // same total context depth.
        let single_range_shapes = crate::shape::infer(&single_range_program, &[1, 71])
            .expect("one new position against a 71-position merged range infers");
        let single_range_bound = crate::bind::bind_with_fusion(
            &single_range_program,
            &single_range_shapes,
            &single_range_outputs,
            false,
            crate::numeric::NumericPolicy::default(),
        )
        .expect("the single-range cached program binds")
        .len();

        std::println!(
            "single_range_vs_two_range raw_ops_per_layer: before={two_range_per_layer} after={single_range_per_layer} bound_ops_at_ctx71: before={two_range_bound} after={single_range_bound}"
        );

        assert!(
            single_range_per_layer < two_range_per_layer,
            "single-range must emit fewer raw ops per layer than the two-range baseline: before={two_range_per_layer} after={single_range_per_layer}"
        );
        assert!(
            single_range_bound < two_range_bound,
            "single-range must bind fewer ops at ctx71 than the two-range baseline: before={two_range_bound} after={single_range_bound}"
        );
    }

    /// The falsifiable claim under test: for the SAME weights and the SAME
    /// `cached_len`, [`append_mistral_single_range_cached_layer`] must
    /// produce the SAME decode-step logits [`append_mistral_cached_layer`]'s
    /// own two-range online-softmax combine produces -- PROVIDED its cache
    /// input holds what write-placement (`proxima-wt-place`'s
    /// `execute_plan_with_placements`) actually hands it at runtime: the
    /// `cached_len` prior positions PLUS this call's OWN rotated K/V
    /// appended at the tail, sized `cached_len + new_count`. This is not a
    /// calling-convention change to the single-range graph -- it is the
    /// single-range graph's documented contract
    /// (`append_mistral_single_range_cached_layer`'s own doc: "the WHOLE
    /// merged context this call attends to ... already folded in by the
    /// caller between calls"). `cpu::evaluate` has no write-placement, so
    /// this test builds that merged cache by hand: run the two-range oracle
    /// first, read this call's own `CachedLayerRoots` back out of its
    /// `Evaluated`, and concatenate them onto the prior cache before
    /// evaluating the single-range arm -- exactly what write-placement
    /// would have left resident. Error is normalized against the two-range
    /// oracle's own BATCH PEAK magnitude (never per-row: a per-row relative
    /// error explodes at zero crossings and produced a false 872% "bug"
    /// report on this codebase before). `cached_len = 0` is covered first
    /// because it is the case most likely to be silently wrong -- with no
    /// prior cache the merged range is exactly this call's own new key(s),
    /// so a query must attend only itself.
    type PerLayerCacheColumns = (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>);

    #[test]
    fn a_single_range_decode_step_matches_the_two_range_decode_step() {
        const VOCAB: usize = 5;
        const EMBEDDING: usize = 4;
        const FEED_FORWARD: usize = 4;
        const QUERY_HEADS: usize = 2;
        const KV_HEADS: usize = 1;
        const HEAD_DIM: usize = 2;
        const PAIRS: usize = HEAD_DIM / 2;
        const GROUP: usize = QUERY_HEADS / KV_HEADS;
        const BLOCK_COUNT: u32 = 2;

        struct LayerWeights {
            attn_norm: Vec<f32>,
            ffn_norm: Vec<f32>,
            wq: Vec<f32>,
            wk: Vec<f32>,
            wv: Vec<f32>,
            wo: Vec<f32>,
            w_gate: Vec<f32>,
            w_up: Vec<f32>,
            w_down: Vec<f32>,
        }

        fn max_error_at(cached_len: usize, new_count: usize) -> (f32, f32) {
            let sequence = cached_len + new_count;
            let ids: Vec<u32> = (0..sequence as u32).map(|id| 1 + id % 3).collect();
            let ids_f32: Vec<f32> = ids.iter().map(|&id| id as f32).collect();

            let table = random_vec(10, VOCAB * EMBEDDING);
            let eps_cached = alloc::vec![1e-5f32; cached_len.max(1)];
            let eps_new = alloc::vec![1e-5f32; new_count];
            let (cos_cached, sin_cached) = rope_angles(0, cached_len.max(1), PAIRS, HEAD_DIM);
            let (cos_new, sin_new) = rope_angles(cached_len, new_count, PAIRS, HEAD_DIM);

            let mut layers = Vec::new();
            let mut seed = 200u64;
            for _ in 0..BLOCK_COUNT {
                layers.push(LayerWeights {
                    attn_norm: alloc::vec![1.0f32; EMBEDDING],
                    ffn_norm: alloc::vec![1.0f32; EMBEDDING],
                    wq: random_vec(seed, EMBEDDING * QUERY_HEADS * HEAD_DIM),
                    wk: random_vec(seed + 1, EMBEDDING * KV_HEADS * HEAD_DIM),
                    wv: random_vec(seed + 2, EMBEDDING * KV_HEADS * HEAD_DIM),
                    wo: random_vec(seed + 3, KV_HEADS * GROUP * HEAD_DIM * EMBEDDING),
                    w_gate: random_vec(seed + 4, EMBEDDING * FEED_FORWARD),
                    w_up: random_vec(seed + 5, EMBEDDING * FEED_FORWARD),
                    w_down: random_vec(seed + 6, FEED_FORWARD * EMBEDDING),
                });
                seed += 7;
            }
            let output_norm = alloc::vec![1.0f32; EMBEDDING];
            let lm_head = random_vec(seed, EMBEDDING * VOCAB);

            let layer_names: Vec<[alloc::string::String; 9]> = (0..BLOCK_COUNT as usize)
                .map(|layer| {
                    [
                        alloc::format!("blk.{layer}.attn_norm.weight"),
                        alloc::format!("blk.{layer}.ffn_norm.weight"),
                        alloc::format!("blk.{layer}.attn_q.weight"),
                        alloc::format!("blk.{layer}.attn_k.weight"),
                        alloc::format!("blk.{layer}.attn_v.weight"),
                        alloc::format!("blk.{layer}.attn_output.weight"),
                        alloc::format!("blk.{layer}.ffn_gate.weight"),
                        alloc::format!("blk.{layer}.ffn_up.weight"),
                        alloc::format!("blk.{layer}.ffn_down.weight"),
                    ]
                })
                .collect();
            let kv_cache_names: Vec<[alloc::string::String; 3]> = (0..BLOCK_COUNT as usize)
                .map(|layer| {
                    [
                        alloc::format!("kv_cache.{layer}.k_even"),
                        alloc::format!("kv_cache.{layer}.k_odd"),
                        alloc::format!("kv_cache.{layer}.v"),
                    ]
                })
                .collect();

            let mut common_named: Vec<(&str, &[f32])> =
                alloc::vec![("token_embd.weight", table.as_slice())];
            for (layer_index, weights) in layers.iter().enumerate() {
                let names = &layer_names[layer_index];
                common_named.push((names[0].as_str(), weights.attn_norm.as_slice()));
                common_named.push((names[1].as_str(), weights.ffn_norm.as_slice()));
                common_named.push((names[2].as_str(), weights.wq.as_slice()));
                common_named.push((names[3].as_str(), weights.wk.as_slice()));
                common_named.push((names[4].as_str(), weights.wv.as_slice()));
                common_named.push((names[5].as_str(), weights.wo.as_slice()));
                common_named.push((names[6].as_str(), weights.w_gate.as_slice()));
                common_named.push((names[7].as_str(), weights.w_up.as_slice()));
                common_named.push((names[8].as_str(), weights.w_down.as_slice()));
            }
            common_named.push(("output_norm.weight", output_norm.as_slice()));
            common_named.push(("output.weight", lm_head.as_slice()));

            // -- fold the cache up to `cached_len` via the two-range
            // program's own prefill path, the same mechanism
            // `a_cached_decode_step_matches_the_uncached_forward_pass_exactly`
            // already trusts.
            let (cached_program, _, cache_roots) = mistral_cached_forward_program(
                VOCAB as u32,
                EMBEDDING as u32,
                FEED_FORWARD as u32,
                QUERY_HEADS as u32,
                KV_HEADS as u32,
                HEAD_DIM as u32,
                BLOCK_COUNT,
            )
            .expect("cached forward pass lowers");

            let (k_even_cache, k_odd_cache, v_cache): PerLayerCacheColumns = if cached_len == 0 {
                    (
                        alloc::vec![Vec::new(); BLOCK_COUNT as usize],
                        alloc::vec![Vec::new(); BLOCK_COUNT as usize],
                        alloc::vec![Vec::new(); BLOCK_COUNT as usize],
                    )
                } else {
                    let prefill_cached_len_value = [0.0f32];
                    let mut prefill_named = common_named.clone();
                    prefill_named.push(("ids", &ids_f32[..cached_len]));
                    prefill_named.push(("eps", eps_cached.as_slice()));
                    prefill_named.push(("rope_cos", cos_cached.as_slice()));
                    prefill_named.push(("rope_sin", sin_cached.as_slice()));
                    prefill_named.push(("cached_len", prefill_cached_len_value.as_slice()));
                    let empty = Vec::<f32>::new();
                    for names in &kv_cache_names {
                        prefill_named.push((names[0].as_str(), empty.as_slice()));
                        prefill_named.push((names[1].as_str(), empty.as_slice()));
                        prefill_named.push((names[2].as_str(), empty.as_slice()));
                    }
                    let mut prefill_roots = Vec::new();
                    for (even, odd, value) in &cache_roots {
                        prefill_roots.push(*even);
                        prefill_roots.push(*odd);
                        prefill_roots.push(*value);
                    }
                    let prefill_symbols = [cached_len as u64, 0u64];
                    let prefill_evaluated = crate::cpu::evaluate_named(
                        &cached_program,
                        &prefill_symbols,
                        &prefill_named,
                        &prefill_roots,
                    )
                    .expect("prefill call evaluates");
                    let mut even_out = Vec::with_capacity(BLOCK_COUNT as usize);
                    let mut odd_out = Vec::with_capacity(BLOCK_COUNT as usize);
                    let mut value_out = Vec::with_capacity(BLOCK_COUNT as usize);
                    for (even, odd, value) in &cache_roots {
                        even_out.push(prefill_evaluated.get(*even).expect("k_even").0.to_vec());
                        odd_out.push(prefill_evaluated.get(*odd).expect("k_odd").0.to_vec());
                        value_out.push(prefill_evaluated.get(*value).expect("v").0.to_vec());
                    }
                    (even_out, odd_out, value_out)
                };

            // -- two-range decode step: the trusted incumbent.
            let two_range_cached_len_value = [cached_len as f32];
            let mut two_range_named = common_named.clone();
            two_range_named.push(("ids", &ids_f32[cached_len..]));
            two_range_named.push(("eps", eps_new.as_slice()));
            two_range_named.push(("rope_cos", cos_new.as_slice()));
            two_range_named.push(("rope_sin", sin_new.as_slice()));
            two_range_named.push(("cached_len", two_range_cached_len_value.as_slice()));
            for (layer_index, names) in kv_cache_names.iter().enumerate() {
                two_range_named.push((names[0].as_str(), k_even_cache[layer_index].as_slice()));
                two_range_named.push((names[1].as_str(), k_odd_cache[layer_index].as_slice()));
                two_range_named.push((names[2].as_str(), v_cache[layer_index].as_slice()));
            }
            let two_range_root = NodeId(cached_program.len() as u32 - 1);
            let two_range_symbols = [new_count as u64, cached_len as u64];
            let mut two_range_roots: Vec<NodeId> = Vec::with_capacity(cache_roots.len() * 3 + 1);
            for (even, odd, value) in &cache_roots {
                two_range_roots.push(*even);
                two_range_roots.push(*odd);
                two_range_roots.push(*value);
            }
            two_range_roots.push(two_range_root);
            let two_range_evaluated = crate::cpu::evaluate_named(
                &cached_program,
                &two_range_symbols,
                &two_range_named,
                &two_range_roots,
            )
            .expect("two-range decode call evaluates");
            let (two_range_logits, two_range_shape) = two_range_evaluated
                .get(two_range_root)
                .expect("two-range logits present");
            assert_eq!(two_range_shape, [new_count as u64, VOCAB as u64]);

            // -- this decode call's own rotated K/V, per layer: exactly
            // what write-placement would leave resident at the cache's
            // tail for the NEXT call. Concatenated onto the prior cache
            // below to build the single-range arm's merged input.
            let mut merged_k_even_cache = k_even_cache.clone();
            let mut merged_k_odd_cache = k_odd_cache.clone();
            let mut merged_v_cache = v_cache.clone();
            for (layer_index, (even, odd, value)) in cache_roots.iter().enumerate() {
                let new_even = two_range_evaluated.get(*even).expect("k_new_even").0;
                let new_odd = two_range_evaluated.get(*odd).expect("k_new_odd").0;
                let new_value = two_range_evaluated.get(*value).expect("v_new").0;
                merged_k_even_cache[layer_index].extend_from_slice(new_even);
                merged_k_odd_cache[layer_index].extend_from_slice(new_odd);
                merged_v_cache[layer_index].extend_from_slice(new_value);
            }

            // -- single-range decode step: the graph under test, fed the
            // MERGED cache (prior positions plus this call's own, folded
            // in by hand the way write-placement would fold them in at
            // runtime).
            let (single_range_program, single_range_root, _, _) =
                mistral_single_range_cached_forward_program(
                    VOCAB as u32,
                    EMBEDDING as u32,
                    FEED_FORWARD as u32,
                    QUERY_HEADS as u32,
                    KV_HEADS as u32,
                    HEAD_DIM as u32,
                    BLOCK_COUNT,
                    false,
                    DuplicateHeadPosition::None,
                )
                .expect("single-range cached forward pass lowers");
            let cached_len_scalar = alloc::vec![cached_len as f32];
            let mut single_range_named = common_named.clone();
            single_range_named.push(("ids", &ids_f32[cached_len..]));
            single_range_named.push(("eps", eps_new.as_slice()));
            single_range_named.push(("rope_cos", cos_new.as_slice()));
            single_range_named.push(("rope_sin", sin_new.as_slice()));
            single_range_named.push(("cached_len", cached_len_scalar.as_slice()));
            for (layer_index, names) in kv_cache_names.iter().enumerate() {
                single_range_named.push((
                    names[0].as_str(),
                    merged_k_even_cache[layer_index].as_slice(),
                ));
                single_range_named.push((
                    names[1].as_str(),
                    merged_k_odd_cache[layer_index].as_slice(),
                ));
                single_range_named.push((names[2].as_str(), merged_v_cache[layer_index].as_slice()));
            }
            let single_range_symbols = [new_count as u64, sequence as u64];
            let single_range_evaluated = crate::cpu::evaluate_named(
                &single_range_program,
                &single_range_symbols,
                &single_range_named,
                &[single_range_root],
            )
            .expect("single-range decode call evaluates");
            let (single_range_logits, single_range_shape) = single_range_evaluated
                .get(single_range_root)
                .expect("single-range logits present");
            assert_eq!(single_range_shape, [new_count as u64, VOCAB as u64]);

            let batch_peak = two_range_logits
                .iter()
                .fold(0.0f32, |peak, value| peak.max(value.abs()));
            let max_error = two_range_logits
                .iter()
                .zip(single_range_logits.iter())
                .map(|(oracle, candidate)| (oracle - candidate).abs())
                .fold(0.0f32, f32::max);
            let normalized_error = if batch_peak > 0.0 {
                max_error / batch_peak
            } else {
                max_error
            };
            std::println!(
                "single_range_vs_two_range_decode cached_len={cached_len} new_count={new_count} batch_peak={batch_peak} max_error={max_error} normalized_error={normalized_error} two_range={two_range_logits:?} single_range={single_range_logits:?}"
            );
            (max_error, normalized_error)
        }

        let cases = [(0usize, 1usize), (1usize, 1usize), (6usize, 2usize)];
        let results: Vec<((usize, usize), (f32, f32))> = cases
            .iter()
            .map(|&(cached_len, new_count)| ((cached_len, new_count), max_error_at(cached_len, new_count)))
            .collect();
        for (cached_len, new_count) in cases {
            let (max_error, normalized_error) = results
                .iter()
                .find(|(case, _)| *case == (cached_len, new_count))
                .expect("case present")
                .1;
            assert!(
                normalized_error < 1e-4,
                "single-range decode diverged from the two-range decode at cached_len={cached_len} new_count={new_count}: max_error={max_error} normalized_error={normalized_error}"
            );
        }
    }

    /// ROW 373's own parity check: [`a_single_range_decode_step_matches_the_two_range_decode_step`]'s
    /// exact harness, `qk_norm` flipped on for both arms
    /// ([`qwen3_cached_forward_program`] as the two-range oracle,
    /// [`mistral_single_range_cached_forward_program`]'s `qk_norm: true` as
    /// the candidate) and `attn_q_norm.weight`/`attn_k_norm.weight` (random,
    /// non-degenerate, so a wrong gamma or a missing normalization is
    /// visible) added per layer. Split-half RoPE is exercised by
    /// construction -- `qk_norm.is_some()` selects it in both builders, see
    /// `append_mistral_cached_layer`'s and
    /// `append_mistral_single_range_cached_layer`'s own doc on that rule.
    /// Same normalized-error tolerance as the plain arm: the online-softmax
    /// combine's own op ordering (two partial reduces, elementwise-summed)
    /// vs the single-range one-shot reduce is not required to be 0-ULP, only
    /// numerically equivalent -- `a_single_range_decode_step_matches_the_two_range_decode_step`'s
    /// own doc already established `< 1e-4` as this codebase's bar for that
    /// distinction.
    #[test]
    fn a_single_range_decode_step_with_qk_norm_matches_the_two_range_decode_step() {
        const VOCAB: usize = 5;
        const EMBEDDING: usize = 4;
        const FEED_FORWARD: usize = 4;
        const QUERY_HEADS: usize = 2;
        const KV_HEADS: usize = 1;
        const HEAD_DIM: usize = 2;
        const PAIRS: usize = HEAD_DIM / 2;
        const GROUP: usize = QUERY_HEADS / KV_HEADS;
        const BLOCK_COUNT: u32 = 2;

        struct LayerWeights {
            attn_norm: Vec<f32>,
            ffn_norm: Vec<f32>,
            wq: Vec<f32>,
            wk: Vec<f32>,
            wv: Vec<f32>,
            wo: Vec<f32>,
            w_gate: Vec<f32>,
            w_up: Vec<f32>,
            w_down: Vec<f32>,
            q_norm: Vec<f32>,
            k_norm: Vec<f32>,
        }

        fn max_error_at(cached_len: usize, new_count: usize) -> (f32, f32) {
            let sequence = cached_len + new_count;
            let ids: Vec<u32> = (0..sequence as u32).map(|id| 1 + id % 3).collect();
            let ids_f32: Vec<f32> = ids.iter().map(|&id| id as f32).collect();

            let table = random_vec(10, VOCAB * EMBEDDING);
            let eps_cached = alloc::vec![1e-5f32; cached_len.max(1)];
            let eps_new = alloc::vec![1e-5f32; new_count];
            let (cos_cached, sin_cached) = rope_angles(0, cached_len.max(1), PAIRS, HEAD_DIM);
            let (cos_new, sin_new) = rope_angles(cached_len, new_count, PAIRS, HEAD_DIM);

            let mut layers = Vec::new();
            let mut seed = 300u64;
            for _ in 0..BLOCK_COUNT {
                layers.push(LayerWeights {
                    attn_norm: alloc::vec![1.0f32; EMBEDDING],
                    ffn_norm: alloc::vec![1.0f32; EMBEDDING],
                    wq: random_vec(seed, EMBEDDING * QUERY_HEADS * HEAD_DIM),
                    wk: random_vec(seed + 1, EMBEDDING * KV_HEADS * HEAD_DIM),
                    wv: random_vec(seed + 2, EMBEDDING * KV_HEADS * HEAD_DIM),
                    wo: random_vec(seed + 3, KV_HEADS * GROUP * HEAD_DIM * EMBEDDING),
                    w_gate: random_vec(seed + 4, EMBEDDING * FEED_FORWARD),
                    w_up: random_vec(seed + 5, EMBEDDING * FEED_FORWARD),
                    w_down: random_vec(seed + 6, FEED_FORWARD * EMBEDDING),
                    q_norm: random_vec(seed + 7, HEAD_DIM),
                    k_norm: random_vec(seed + 8, HEAD_DIM),
                });
                seed += 9;
            }
            let output_norm = alloc::vec![1.0f32; EMBEDDING];
            let lm_head = random_vec(seed, EMBEDDING * VOCAB);

            let layer_names: Vec<[alloc::string::String; 11]> = (0..BLOCK_COUNT as usize)
                .map(|layer| {
                    [
                        alloc::format!("blk.{layer}.attn_norm.weight"),
                        alloc::format!("blk.{layer}.ffn_norm.weight"),
                        alloc::format!("blk.{layer}.attn_q.weight"),
                        alloc::format!("blk.{layer}.attn_k.weight"),
                        alloc::format!("blk.{layer}.attn_v.weight"),
                        alloc::format!("blk.{layer}.attn_output.weight"),
                        alloc::format!("blk.{layer}.ffn_gate.weight"),
                        alloc::format!("blk.{layer}.ffn_up.weight"),
                        alloc::format!("blk.{layer}.ffn_down.weight"),
                        alloc::format!("blk.{layer}.attn_q_norm.weight"),
                        alloc::format!("blk.{layer}.attn_k_norm.weight"),
                    ]
                })
                .collect();
            let kv_cache_names: Vec<[alloc::string::String; 3]> = (0..BLOCK_COUNT as usize)
                .map(|layer| {
                    [
                        alloc::format!("kv_cache.{layer}.k_even"),
                        alloc::format!("kv_cache.{layer}.k_odd"),
                        alloc::format!("kv_cache.{layer}.v"),
                    ]
                })
                .collect();

            let mut common_named: Vec<(&str, &[f32])> =
                alloc::vec![("token_embd.weight", table.as_slice())];
            for (layer_index, weights) in layers.iter().enumerate() {
                let names = &layer_names[layer_index];
                common_named.push((names[0].as_str(), weights.attn_norm.as_slice()));
                common_named.push((names[1].as_str(), weights.ffn_norm.as_slice()));
                common_named.push((names[2].as_str(), weights.wq.as_slice()));
                common_named.push((names[3].as_str(), weights.wk.as_slice()));
                common_named.push((names[4].as_str(), weights.wv.as_slice()));
                common_named.push((names[5].as_str(), weights.wo.as_slice()));
                common_named.push((names[6].as_str(), weights.w_gate.as_slice()));
                common_named.push((names[7].as_str(), weights.w_up.as_slice()));
                common_named.push((names[8].as_str(), weights.w_down.as_slice()));
                common_named.push((names[9].as_str(), weights.q_norm.as_slice()));
                common_named.push((names[10].as_str(), weights.k_norm.as_slice()));
            }
            common_named.push(("output_norm.weight", output_norm.as_slice()));
            common_named.push(("output.weight", lm_head.as_slice()));

            // -- fold the cache up to `cached_len` via the two-range qk-norm
            // program's own prefill path, same mechanism the plain-layer
            // parity test trusts.
            let (cached_program, _, cache_roots) = qwen3_cached_forward_program(
                VOCAB as u32,
                EMBEDDING as u32,
                FEED_FORWARD as u32,
                QUERY_HEADS as u32,
                KV_HEADS as u32,
                HEAD_DIM as u32,
                BLOCK_COUNT,
            )
            .expect("qk-norm cached forward pass lowers");

            let (k_even_cache, k_odd_cache, v_cache): PerLayerCacheColumns = if cached_len == 0 {
                (
                    alloc::vec![Vec::new(); BLOCK_COUNT as usize],
                    alloc::vec![Vec::new(); BLOCK_COUNT as usize],
                    alloc::vec![Vec::new(); BLOCK_COUNT as usize],
                )
            } else {
                let prefill_cached_len_value = [0.0f32];
                let mut prefill_named = common_named.clone();
                prefill_named.push(("ids", &ids_f32[..cached_len]));
                prefill_named.push(("eps", eps_cached.as_slice()));
                prefill_named.push(("rope_cos", cos_cached.as_slice()));
                prefill_named.push(("rope_sin", sin_cached.as_slice()));
                prefill_named.push(("cached_len", prefill_cached_len_value.as_slice()));
                let empty = Vec::<f32>::new();
                for names in &kv_cache_names {
                    prefill_named.push((names[0].as_str(), empty.as_slice()));
                    prefill_named.push((names[1].as_str(), empty.as_slice()));
                    prefill_named.push((names[2].as_str(), empty.as_slice()));
                }
                let mut prefill_roots = Vec::new();
                for (even, odd, value) in &cache_roots {
                    prefill_roots.push(*even);
                    prefill_roots.push(*odd);
                    prefill_roots.push(*value);
                }
                let prefill_symbols = [cached_len as u64, 0u64];
                let prefill_evaluated = crate::cpu::evaluate_named(
                    &cached_program,
                    &prefill_symbols,
                    &prefill_named,
                    &prefill_roots,
                )
                .expect("prefill call evaluates");
                let mut even_out = Vec::with_capacity(BLOCK_COUNT as usize);
                let mut odd_out = Vec::with_capacity(BLOCK_COUNT as usize);
                let mut value_out = Vec::with_capacity(BLOCK_COUNT as usize);
                for (even, odd, value) in &cache_roots {
                    even_out.push(prefill_evaluated.get(*even).expect("k_even").0.to_vec());
                    odd_out.push(prefill_evaluated.get(*odd).expect("k_odd").0.to_vec());
                    value_out.push(prefill_evaluated.get(*value).expect("v").0.to_vec());
                }
                (even_out, odd_out, value_out)
            };

            // -- two-range decode step: the trusted incumbent.
            let two_range_cached_len_value = [cached_len as f32];
            let mut two_range_named = common_named.clone();
            two_range_named.push(("ids", &ids_f32[cached_len..]));
            two_range_named.push(("eps", eps_new.as_slice()));
            two_range_named.push(("rope_cos", cos_new.as_slice()));
            two_range_named.push(("rope_sin", sin_new.as_slice()));
            two_range_named.push(("cached_len", two_range_cached_len_value.as_slice()));
            for (layer_index, names) in kv_cache_names.iter().enumerate() {
                two_range_named.push((names[0].as_str(), k_even_cache[layer_index].as_slice()));
                two_range_named.push((names[1].as_str(), k_odd_cache[layer_index].as_slice()));
                two_range_named.push((names[2].as_str(), v_cache[layer_index].as_slice()));
            }
            let two_range_root = NodeId(cached_program.len() as u32 - 1);
            let two_range_symbols = [new_count as u64, cached_len as u64];
            let mut two_range_roots: Vec<NodeId> = Vec::with_capacity(cache_roots.len() * 3 + 1);
            for (even, odd, value) in &cache_roots {
                two_range_roots.push(*even);
                two_range_roots.push(*odd);
                two_range_roots.push(*value);
            }
            two_range_roots.push(two_range_root);
            let two_range_evaluated = crate::cpu::evaluate_named(
                &cached_program,
                &two_range_symbols,
                &two_range_named,
                &two_range_roots,
            )
            .expect("two-range decode call evaluates");
            let (two_range_logits, two_range_shape) = two_range_evaluated
                .get(two_range_root)
                .expect("two-range logits present");
            assert_eq!(two_range_shape, [new_count as u64, VOCAB as u64]);

            // -- this decode call's own rotated K/V, per layer: exactly
            // what write-placement would leave resident at the cache's
            // tail for the NEXT call. Concatenated onto the prior cache
            // below to build the single-range arm's merged input.
            let mut merged_k_even_cache = k_even_cache.clone();
            let mut merged_k_odd_cache = k_odd_cache.clone();
            let mut merged_v_cache = v_cache.clone();
            for (layer_index, (even, odd, value)) in cache_roots.iter().enumerate() {
                let new_even = two_range_evaluated.get(*even).expect("k_new_even").0;
                let new_odd = two_range_evaluated.get(*odd).expect("k_new_odd").0;
                let new_value = two_range_evaluated.get(*value).expect("v_new").0;
                merged_k_even_cache[layer_index].extend_from_slice(new_even);
                merged_k_odd_cache[layer_index].extend_from_slice(new_odd);
                merged_v_cache[layer_index].extend_from_slice(new_value);
            }

            // -- single-range decode step: the graph under test, `qk_norm:
            // true`, fed the MERGED cache.
            let (single_range_program, single_range_root, _, _) =
                mistral_single_range_cached_forward_program(
                    VOCAB as u32,
                    EMBEDDING as u32,
                    FEED_FORWARD as u32,
                    QUERY_HEADS as u32,
                    KV_HEADS as u32,
                    HEAD_DIM as u32,
                    BLOCK_COUNT,
                    true,
                    DuplicateHeadPosition::None,
                )
                .expect("single-range qk-norm cached forward pass lowers");
            let cached_len_scalar = alloc::vec![cached_len as f32];
            let mut single_range_named = common_named.clone();
            single_range_named.push(("ids", &ids_f32[cached_len..]));
            single_range_named.push(("eps", eps_new.as_slice()));
            single_range_named.push(("rope_cos", cos_new.as_slice()));
            single_range_named.push(("rope_sin", sin_new.as_slice()));
            single_range_named.push(("cached_len", cached_len_scalar.as_slice()));
            for (layer_index, names) in kv_cache_names.iter().enumerate() {
                single_range_named.push((
                    names[0].as_str(),
                    merged_k_even_cache[layer_index].as_slice(),
                ));
                single_range_named.push((
                    names[1].as_str(),
                    merged_k_odd_cache[layer_index].as_slice(),
                ));
                single_range_named.push((names[2].as_str(), merged_v_cache[layer_index].as_slice()));
            }
            let single_range_symbols = [new_count as u64, sequence as u64];
            let single_range_evaluated = crate::cpu::evaluate_named(
                &single_range_program,
                &single_range_symbols,
                &single_range_named,
                &[single_range_root],
            )
            .expect("single-range decode call evaluates");
            let (single_range_logits, single_range_shape) = single_range_evaluated
                .get(single_range_root)
                .expect("single-range logits present");
            assert_eq!(single_range_shape, [new_count as u64, VOCAB as u64]);

            let batch_peak = two_range_logits
                .iter()
                .fold(0.0f32, |peak, value| peak.max(value.abs()));
            let max_error = two_range_logits
                .iter()
                .zip(single_range_logits.iter())
                .map(|(oracle, candidate)| (oracle - candidate).abs())
                .fold(0.0f32, f32::max);
            let normalized_error = if batch_peak > 0.0 {
                max_error / batch_peak
            } else {
                max_error
            };
            std::println!(
                "single_range_vs_two_range_decode_qk_norm cached_len={cached_len} new_count={new_count} batch_peak={batch_peak} max_error={max_error} normalized_error={normalized_error} two_range={two_range_logits:?} single_range={single_range_logits:?}"
            );
            (max_error, normalized_error)
        }

        let cases = [(0usize, 1usize), (1usize, 1usize), (6usize, 2usize)];
        let results: Vec<((usize, usize), (f32, f32))> = cases
            .iter()
            .map(|&(cached_len, new_count)| ((cached_len, new_count), max_error_at(cached_len, new_count)))
            .collect();
        for (cached_len, new_count) in cases {
            let (max_error, normalized_error) = results
                .iter()
                .find(|(case, _)| *case == (cached_len, new_count))
                .expect("case present")
                .1;
            assert!(
                normalized_error < 1e-4,
                "single-range qk-norm decode diverged from the two-range qk-norm decode at cached_len={cached_len} new_count={new_count}: max_error={max_error} normalized_error={normalized_error}"
            );
        }
    }

    /// Direct A/B on the SAME single-range program under the SAME data:
    /// `crate::bind::bind_with_fusion(.., true)` (fires
    /// [`cached_attention_single_range_candidates`], one
    /// `BoundOpKind::CachedAttention` per layer) against `bind_with_fusion(..,
    /// false)` (the literal `score_even`/`score_odd`/mask/softmax/`attended`
    /// chain [`crate::cpu::run_node_into`] would otherwise run node-for-node).
    /// [`a_single_range_decode_step_matches_the_two_range_decode_step`]
    /// already proves the fused kind agrees with the two-range oracle on
    /// real attention math; this test isolates the rewrite itself —
    /// same program, same weights, same cache, fused vs not — so a
    /// divergence here can only be the fusion transform, never a
    /// two-range-specific difference.
    #[test]
    #[cfg(feature = "cached-attention-streaming")]
    fn cached_attention_single_range_fused_matches_the_unfused_program() {
        use core::pin::pin;
        use core::task::{Context, Poll, Waker};

        use crate::bind::{
            BoundOp, BoundOpKind, READY_BATCH_CAPACITY, ReadyBatch, bind_with_fusion,
            block_node_ids,
        };
        use crate::cpu::Interpreter;
        use crate::numeric::NumericPolicy;
        use proxima_primitives::pipe::Pipe;

        const VOCAB: u32 = 5;
        const EMBEDDING: u32 = 4;
        const FEED_FORWARD: u32 = 4;
        const QUERY_HEADS: u32 = 2;
        const KV_HEADS: u32 = 1;
        const HEAD_DIM: u32 = 2;
        const BLOCK_COUNT: u32 = 2;
        const CACHED_LEN: usize = 3;
        const NEW_COUNT: usize = 2;
        const MERGED_LEN: usize = CACHED_LEN + NEW_COUNT;

        fn run_resolved(program_len: usize, resolved: &[BoundOp], inputs: Vec<(NodeId, Vec<f32>)>) -> Vec<Option<Vec<f32>>> {
            let mut buffers: Vec<Option<Vec<f32>>> = alloc::vec![None; program_len];
            for (node, data) in inputs {
                buffers[node.0 as usize] = Some(data);
            }
            let interpreter = Interpreter::new(&mut buffers);
            for chunk in resolved.chunks(READY_BATCH_CAPACITY) {
                let batch: ReadyBatch = chunk.iter().cloned().collect();
                let waker = Waker::noop();
                let mut context = Context::from_waker(waker);
                let mut future = pin!(interpreter.call(batch));
                match future.as_mut().poll(&mut context) {
                    Poll::Ready(result) => result.expect("resolved batch computes"),
                    Poll::Pending => unreachable!("cpu pipes never yield: no internal .await"),
                }
            }
            buffers
        }

        // `bucket_padding` reproduces `kv-capacity-bucket`'s KV extent
        // rounding directly on this fixture: the KV leaves grow by
        // `bucket_padding` zero-filled rows past `MERGED_LEN` (exactly what
        // `[merged_len, bucket)` looks like at runtime) while `cached_len`'s
        // own `Op::Input` value never moves, so any divergence this
        // introduces is the fused op reading the wrong band, never a
        // different cache content.
        fn max_error_for_padding(padding: usize, program: &[Op], root: NodeId) -> f32 {
            let pairs = (HEAD_DIM / 2) as usize;
            let group = (QUERY_HEADS / KV_HEADS) as usize;
            let sequence = MERGED_LEN + padding;

            let table = random_vec(1, VOCAB as usize * EMBEDDING as usize);
            let ids: Vec<f32> = (0..NEW_COUNT as u32).map(|id| 1.0 + (id % 3) as f32).collect();
            let eps = alloc::vec![1e-5f32; NEW_COUNT];
            let (cos_new, sin_new) = rope_angles(CACHED_LEN, NEW_COUNT, pairs, HEAD_DIM as usize);
            let cached_len_scalar = alloc::vec![CACHED_LEN as f32];

            let mut owned: Vec<(String, Vec<f32>)> = alloc::vec![
                (String::from("token_embd.weight"), table),
                (String::from("ids"), ids),
                (String::from("eps"), eps),
                (String::from("rope_cos"), cos_new),
                (String::from("rope_sin"), sin_new),
                (String::from("cached_len"), cached_len_scalar),
            ];
            let mut seed = 900u64;
            for layer in 0..BLOCK_COUNT as usize {
                owned.push((alloc::format!("blk.{layer}.attn_norm.weight"), alloc::vec![1.0f32; EMBEDDING as usize]));
                owned.push((alloc::format!("blk.{layer}.ffn_norm.weight"), alloc::vec![1.0f32; EMBEDDING as usize]));
                owned.push((
                    alloc::format!("blk.{layer}.attn_q.weight"),
                    random_vec(seed, EMBEDDING as usize * QUERY_HEADS as usize * HEAD_DIM as usize),
                ));
                owned.push((
                    alloc::format!("blk.{layer}.attn_k.weight"),
                    random_vec(seed + 1, EMBEDDING as usize * KV_HEADS as usize * HEAD_DIM as usize),
                ));
                owned.push((
                    alloc::format!("blk.{layer}.attn_v.weight"),
                    random_vec(seed + 2, EMBEDDING as usize * KV_HEADS as usize * HEAD_DIM as usize),
                ));
                owned.push((
                    alloc::format!("blk.{layer}.attn_output.weight"),
                    random_vec(seed + 3, KV_HEADS as usize * group * HEAD_DIM as usize * EMBEDDING as usize),
                ));
                owned.push((
                    alloc::format!("blk.{layer}.ffn_gate.weight"),
                    random_vec(seed + 4, EMBEDDING as usize * FEED_FORWARD as usize),
                ));
                owned.push((
                    alloc::format!("blk.{layer}.ffn_up.weight"),
                    random_vec(seed + 5, EMBEDDING as usize * FEED_FORWARD as usize),
                ));
                owned.push((
                    alloc::format!("blk.{layer}.ffn_down.weight"),
                    random_vec(seed + 6, FEED_FORWARD as usize * EMBEDDING as usize),
                ));
                let mut k_even = random_vec(seed + 7, MERGED_LEN * KV_HEADS as usize * pairs);
                k_even.resize(sequence * KV_HEADS as usize * pairs, 0.0);
                let mut k_odd = random_vec(seed + 8, MERGED_LEN * KV_HEADS as usize * pairs);
                k_odd.resize(sequence * KV_HEADS as usize * pairs, 0.0);
                let mut v = random_vec(seed + 9, MERGED_LEN * KV_HEADS as usize * HEAD_DIM as usize);
                v.resize(sequence * KV_HEADS as usize * HEAD_DIM as usize, 0.0);
                owned.push((alloc::format!("kv_cache.{layer}.k_even"), k_even));
                owned.push((alloc::format!("kv_cache.{layer}.k_odd"), k_odd));
                owned.push((alloc::format!("kv_cache.{layer}.v"), v));
                seed += 10;
            }
            owned.push((String::from("output_norm.weight"), alloc::vec![1.0f32; EMBEDDING as usize]));
            owned.push((String::from("output.weight"), random_vec(seed, EMBEDDING as usize * VOCAB as usize)));

            let shapes = crate::shape::infer(program, &[NEW_COUNT as u64, sequence as u64])
                .expect("single-range fused-vs-unfused fixture infers");

            let inputs_for = |resolved: &[BoundOp]| -> Vec<(NodeId, Vec<f32>)> {
                let _ = resolved;
                block_node_ids(program)
                    .into_iter()
                    .map(|node| {
                        let name = match &program[node.0 as usize] {
                            Op::Input { name: Some(name), .. } => name.clone(),
                            _ => unreachable!("block_node_ids only ever returns Op::Input nodes"),
                        };
                        let data = owned
                            .iter()
                            .find(|(candidate, _)| *candidate == name)
                            .unwrap_or_else(|| panic!("missing named input {name}"))
                            .1
                            .clone();
                        (node, data)
                    })
                    .collect()
            };

            let fused = bind_with_fusion(program, &shapes, &[root], true, NumericPolicy::default())
                .expect("fused single-range bind succeeds");
            let unfused = bind_with_fusion(program, &shapes, &[root], false, NumericPolicy::default())
                .expect("unfused single-range bind succeeds");

            assert!(
                fused
                    .iter()
                    .any(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. })),
                "fused bind must produce at least one cached-attention step"
            );
            assert!(
                !unfused
                    .iter()
                    .any(|bound| matches!(bound.kind, BoundOpKind::CachedAttention { .. })),
                "unfused bind must never produce a cached-attention step"
            );

            let fused_buffers = run_resolved(program.len(), &fused, inputs_for(&fused));
            let unfused_buffers = run_resolved(program.len(), &unfused, inputs_for(&unfused));

            let fused_logits = fused_buffers[root.0 as usize]
                .as_ref()
                .expect("fused logits present");
            let unfused_logits = unfused_buffers[root.0 as usize]
                .as_ref()
                .expect("unfused logits present");

            assert_eq!(fused_logits.len(), unfused_logits.len());
            let max_error = fused_logits
                .iter()
                .zip(unfused_logits.iter())
                .map(|(fused, unfused)| (fused - unfused).abs())
                .fold(0.0f32, f32::max);
            std::println!(
                "single_range_fused_vs_unfused bucket_padding={padding} max_error={max_error} fused={fused_logits:?} unfused={unfused_logits:?}"
            );
            max_error
        }

        let (program, root, _, _) = mistral_single_range_cached_forward_program(
            VOCAB,
            EMBEDDING,
            FEED_FORWARD,
            QUERY_HEADS,
            KV_HEADS,
            HEAD_DIM,
            BLOCK_COUNT,
            false,
            DuplicateHeadPosition::None,
        )
        .expect("single-range cached forward pass lowers");

        for bucket_padding in [0usize, 1, 5] {
            let max_error = max_error_for_padding(bucket_padding, &program, root);
            assert!(
                max_error <= 1e-5,
                "fused single-range cached attention diverged from the unfused program at bucket_padding={bucket_padding}: max_error={max_error}"
            );
        }
    }

    /// CARD 6.1's falsifiable claim: `proxima-model-interop`'s
    /// `kv-capacity-bucket` feature rounds the single-range KV extent
    /// (`Extent::Symbolic(1)`, bound via `symbols[1]`) up from the true,
    /// strictly-increasing `merged_len` to `bucket = ceil(merged_len /
    /// KV_BUCKET_TOKENS) * KV_BUCKET_TOKENS`, so the Metal plan-cache key
    /// (`(new_count, symbols[1])`) stays constant across a whole bucket of
    /// decode steps. This must not move a single bit of the decode step's
    /// logits, PROVIDED the padded tail `[merged_len, bucket)` reads as
    /// exactly zero: [`causal_mask_merged`]'s existing `key_index >
    /// query_absolute` comparison already masks every key index
    /// `>= merged_len` as "future" for every query in this call (no
    /// query's own absolute position ever reaches a padded key's index,
    /// since the last query sits at `merged_len - 1`), so
    /// `ScalarOp::Select` picks the constant `neg_infinity` for the whole
    /// padded tail without ever reading it -- zero new `Op`/`BoundOpKind`/
    /// `ScalarOp`/`IndexMap` variant, exactly `Cargo.toml`'s
    /// `kv-capacity-bucket` doc states. Covers the three bucket sizes CARD
    /// 6.1 names (8, 32, 256) with `cached_len` set to span each bucket's
    /// own boundary (one merged_len below it, exactly on it, one above
    /// it) against a one-new-token decode step -- 9 cases.
    #[cfg(feature = "kv-capacity-bucket")]
    #[test]
    fn cpu_mask_zero_ulp() {
        const VOCAB: usize = 5;
        const EMBEDDING: usize = 4;
        const FEED_FORWARD: usize = 4;
        const QUERY_HEADS: usize = 2;
        const KV_HEADS: usize = 1;
        const HEAD_DIM: usize = 2;
        const PAIRS: usize = HEAD_DIM / 2;
        const GROUP: usize = QUERY_HEADS / KV_HEADS;
        const BLOCK_COUNT: u32 = 2;
        const NEW_COUNT: usize = 1;

        fn bucket_of(merged_len: usize, bucket_tokens: usize) -> usize {
            merged_len.div_ceil(bucket_tokens) * bucket_tokens
        }

        // tight (unbucketed, `symbols[1] == merged_len`) vs padded
        // (`symbols[1] == bucket`) logits for the SAME weights, SAME
        // `cached_len`, SAME real KV content in `[0, merged_len)` --
        // built once and sliced/extended, never regenerated per arm, so
        // any divergence is the mask, not a different random draw.
        fn logits_at(cached_len: usize, bucket_tokens: usize) -> (Vec<f32>, Vec<f32>) {
            let merged_len = cached_len + NEW_COUNT;
            let bucket = bucket_of(merged_len, bucket_tokens);
            let ids_f32: Vec<f32> = (0..merged_len as u32).map(|id| (1 + id % 3) as f32).collect();

            let table = random_vec(10, VOCAB * EMBEDDING);
            let eps_new = alloc::vec![1e-5f32; NEW_COUNT];
            let (cos_new, sin_new) = rope_angles(cached_len, NEW_COUNT, PAIRS, HEAD_DIM);

            let layer_names: Vec<[alloc::string::String; 9]> = (0..BLOCK_COUNT as usize)
                .map(|layer| {
                    [
                        alloc::format!("blk.{layer}.attn_norm.weight"),
                        alloc::format!("blk.{layer}.ffn_norm.weight"),
                        alloc::format!("blk.{layer}.attn_q.weight"),
                        alloc::format!("blk.{layer}.attn_k.weight"),
                        alloc::format!("blk.{layer}.attn_v.weight"),
                        alloc::format!("blk.{layer}.attn_output.weight"),
                        alloc::format!("blk.{layer}.ffn_gate.weight"),
                        alloc::format!("blk.{layer}.ffn_up.weight"),
                        alloc::format!("blk.{layer}.ffn_down.weight"),
                    ]
                })
                .collect();
            let kv_cache_names: Vec<[alloc::string::String; 3]> = (0..BLOCK_COUNT as usize)
                .map(|layer| {
                    [
                        alloc::format!("kv_cache.{layer}.k_even"),
                        alloc::format!("kv_cache.{layer}.k_odd"),
                        alloc::format!("kv_cache.{layer}.v"),
                    ]
                })
                .collect();

            let mut common_named: Vec<(&str, &[f32])> =
                alloc::vec![("token_embd.weight", table.as_slice())];
            let mut layer_weights: Vec<[Vec<f32>; 9]> = Vec::with_capacity(BLOCK_COUNT as usize);
            let mut seed = 900u64;
            for _ in 0..BLOCK_COUNT {
                layer_weights.push([
                    alloc::vec![1.0f32; EMBEDDING],
                    alloc::vec![1.0f32; EMBEDDING],
                    random_vec(seed, EMBEDDING * QUERY_HEADS * HEAD_DIM),
                    random_vec(seed + 1, EMBEDDING * KV_HEADS * HEAD_DIM),
                    random_vec(seed + 2, EMBEDDING * KV_HEADS * HEAD_DIM),
                    random_vec(seed + 3, KV_HEADS * GROUP * HEAD_DIM * EMBEDDING),
                    random_vec(seed + 4, EMBEDDING * FEED_FORWARD),
                    random_vec(seed + 5, EMBEDDING * FEED_FORWARD),
                    random_vec(seed + 6, FEED_FORWARD * EMBEDDING),
                ]);
                seed += 7;
            }
            for (layer_index, weights) in layer_weights.iter().enumerate() {
                let names = &layer_names[layer_index];
                for (name, data) in names.iter().zip(weights.iter()) {
                    common_named.push((name.as_str(), data.as_slice()));
                }
            }
            let output_norm = alloc::vec![1.0f32; EMBEDDING];
            let lm_head = random_vec(seed, EMBEDDING * VOCAB);
            common_named.push(("output_norm.weight", output_norm.as_slice()));
            common_named.push(("output.weight", lm_head.as_slice()));

            // per-layer KV cache, `bucket`-long, real random content in
            // `[0, merged_len)`, EXACT zero in the padded tail
            // `[merged_len, bucket)` -- the tail-zero invariant
            // `run_decode_loop_placed_kv`'s own one-time buffer zero-fill
            // provides at runtime (`omega::metal::zero_placed_buffer`).
            let mut k_even_padded: Vec<Vec<f32>> = Vec::with_capacity(BLOCK_COUNT as usize);
            let mut k_odd_padded: Vec<Vec<f32>> = Vec::with_capacity(BLOCK_COUNT as usize);
            let mut v_padded: Vec<Vec<f32>> = Vec::with_capacity(BLOCK_COUNT as usize);
            let mut kv_seed = 5000u64;
            for _ in 0..BLOCK_COUNT {
                let mut k_even = random_vec(kv_seed, merged_len * KV_HEADS * PAIRS);
                k_even.resize(bucket * KV_HEADS * PAIRS, 0.0);
                let mut k_odd = random_vec(kv_seed + 1, merged_len * KV_HEADS * PAIRS);
                k_odd.resize(bucket * KV_HEADS * PAIRS, 0.0);
                let mut v = random_vec(kv_seed + 2, merged_len * KV_HEADS * HEAD_DIM);
                v.resize(bucket * KV_HEADS * HEAD_DIM, 0.0);
                k_even_padded.push(k_even);
                k_odd_padded.push(k_odd);
                v_padded.push(v);
                kv_seed += 3;
            }

            let (program, root, _, _) = mistral_single_range_cached_forward_program(
                VOCAB as u32,
                EMBEDDING as u32,
                FEED_FORWARD as u32,
                QUERY_HEADS as u32,
                KV_HEADS as u32,
                HEAD_DIM as u32,
                BLOCK_COUNT,
                false,
                DuplicateHeadPosition::None,
            )
            .expect("single-range cached forward pass lowers");

            let cached_len_scalar = alloc::vec![cached_len as f32];
            let run = |extent: usize, k_even: &[Vec<f32>], k_odd: &[Vec<f32>], v: &[Vec<f32>]| {
                let mut named = common_named.clone();
                named.push(("ids", ids_f32[cached_len..].as_ref()));
                named.push(("eps", eps_new.as_slice()));
                named.push(("rope_cos", cos_new.as_slice()));
                named.push(("rope_sin", sin_new.as_slice()));
                named.push(("cached_len", cached_len_scalar.as_slice()));
                for (layer_index, names) in kv_cache_names.iter().enumerate() {
                    named.push((names[0].as_str(), k_even[layer_index].as_slice()));
                    named.push((names[1].as_str(), k_odd[layer_index].as_slice()));
                    named.push((names[2].as_str(), v[layer_index].as_slice()));
                }
                let symbols = [NEW_COUNT as u64, extent as u64];
                let evaluated = crate::cpu::evaluate_named(&program, &symbols, &named, &[root])
                    .expect("single-range decode call evaluates");
                evaluated.get(root).expect("logits present").0.to_vec()
            };

            let tight_k_even: Vec<Vec<f32>> = k_even_padded
                .iter()
                .map(|column| column[..merged_len * KV_HEADS * PAIRS].to_vec())
                .collect();
            let tight_k_odd: Vec<Vec<f32>> = k_odd_padded
                .iter()
                .map(|column| column[..merged_len * KV_HEADS * PAIRS].to_vec())
                .collect();
            let tight_v: Vec<Vec<f32>> = v_padded
                .iter()
                .map(|column| column[..merged_len * KV_HEADS * HEAD_DIM].to_vec())
                .collect();

            let tight_logits = run(merged_len, &tight_k_even, &tight_k_odd, &tight_v);
            let padded_logits = run(bucket, &k_even_padded, &k_odd_padded, &v_padded);
            (tight_logits, padded_logits)
        }

        let cases: Vec<(usize, usize)> = [8usize, 32, 256]
            .into_iter()
            .flat_map(|bucket_tokens| {
                [
                    bucket_tokens.saturating_sub(2),
                    bucket_tokens.saturating_sub(1),
                    bucket_tokens,
                ]
                .into_iter()
                .map(move |cached_len| (bucket_tokens, cached_len))
            })
            .collect();
        assert_eq!(cases.len(), 9, "3 bucket sizes x 3 boundary-spanning cached_len values");

        for (bucket_tokens, cached_len) in cases {
            let (tight, padded) = logits_at(cached_len, bucket_tokens);
            std::println!(
                "cpu_mask_zero_ulp bucket_tokens={bucket_tokens} cached_len={cached_len} tight={tight:?} padded={padded:?}"
            );
            assert_eq!(
                tight, padded,
                "bucket_tokens={bucket_tokens} cached_len={cached_len}: bucketed KV extent diverged from the tight extent, 0-ULP required"
            );
        }
    }

    /// Classifies every [`crate::bind::BoundOp`] a real decode step binds as
    /// VARIANT (its resolved shape/layout/body changes when `cached_len`
    /// changes) or INVARIANT (it does not), by binding the SAME program
    /// twice against two different `cached_len` values and comparing the
    /// two `Vec<BoundOp>` positionally. `BoundOp: PartialEq`
    /// (`crate::bind::BoundOp`'s own derive) makes this an exact structural
    /// diff, not an inference about which ops "should" depend on the cache:
    /// any op whose extents, operand layouts, or fused body differ between
    /// the two binds is exactly the set a per-step re-resolve exists to
    /// recompute; anything unchanged was resolved for nothing.
    ///
    /// This is the count `docs/discipline.md`'s resolve-once row rests on --
    /// see that row for the paper estimate (~15%) this either confirms or
    /// refutes.
    #[test]
    fn bound_ops_are_classified_variant_or_invariant_in_cached_len() {
        const VOCAB: u32 = 32_002;
        const EMBEDDING: u32 = 4096;
        const FEED_FORWARD: u32 = 14336;
        const QUERY_HEADS: u32 = 32;
        const KV_HEADS: u32 = 8;
        const HEAD_DIM: u32 = 128;
        const BLOCK_COUNT: u32 = 2;
        const NEW_COUNT: u64 = 1;
        const CACHED_LEN_A: u64 = 50;
        const CACHED_LEN_B: u64 = 51;

        let header_nodes = mistral_cached_forward_program(
            VOCAB,
            EMBEDDING,
            FEED_FORWARD,
            QUERY_HEADS,
            KV_HEADS,
            HEAD_DIM,
            0,
        )
        .expect("a zero-layer program still lowers (embedding lookup plus final norm/lm-head)")
        .0
        .len();
        let one_layer_nodes = mistral_cached_forward_program(
            VOCAB,
            EMBEDDING,
            FEED_FORWARD,
            QUERY_HEADS,
            KV_HEADS,
            HEAD_DIM,
            1,
        )
        .expect("a one-layer program lowers")
        .0
        .len();
        let per_layer_program_nodes = one_layer_nodes - header_nodes;

        let (program, logits_root, cache_roots) = mistral_cached_forward_program(
            VOCAB,
            EMBEDDING,
            FEED_FORWARD,
            QUERY_HEADS,
            KV_HEADS,
            HEAD_DIM,
            BLOCK_COUNT,
        )
        .expect("the two-layer cached forward pass lowers to a program");
        let mut outputs = alloc::vec![logits_root];
        for (even, odd, value) in &cache_roots {
            outputs.extend_from_slice(&[*even, *odd, *value]);
        }

        let shapes_a = crate::shape::infer(&program, &[NEW_COUNT, CACHED_LEN_A])
            .expect("cached_len=50 infers");
        let resolved_a =
            crate::bind::bind(&program, &shapes_a, &outputs, crate::numeric::NumericPolicy::bit_exact()).expect("cached_len=50 binds");
        let shapes_b = crate::shape::infer(&program, &[NEW_COUNT, CACHED_LEN_B])
            .expect("cached_len=51 infers");
        let resolved_b =
            crate::bind::bind(&program, &shapes_b, &outputs, crate::numeric::NumericPolicy::bit_exact()).expect("cached_len=51 binds");

        assert_eq!(
            resolved_a.len(),
            resolved_b.len(),
            "the same program topology must bind to the same bound-op count regardless of cached_len"
        );

        let mut variant_total = 0usize;
        let mut invariant_total = 0usize;
        // layer index -> (variant, invariant); usize::MAX buckets header/lm-head nodes
        let mut per_layer: std::collections::BTreeMap<usize, (usize, usize)> =
            std::collections::BTreeMap::new();

        for (bound_a, bound_b) in resolved_a.iter().zip(resolved_b.iter()) {
            let variant = bound_a != bound_b;
            if variant {
                variant_total += 1;
            } else {
                invariant_total += 1;
            }
            let node_index = bound_a.node.0 as usize;
            let layer = if node_index >= header_nodes {
                let offset = node_index - header_nodes;
                let layer = offset / per_layer_program_nodes;
                if layer < BLOCK_COUNT as usize {
                    layer
                } else {
                    usize::MAX
                }
            } else {
                usize::MAX
            };
            let entry = per_layer.entry(layer).or_insert((0, 0));
            if variant {
                entry.0 += 1;
            } else {
                entry.1 += 1;
            }
        }

        std::println!(
            "bound_op_classification total_bound_ops={} variant={variant_total} invariant={invariant_total} variant_pct={:.1}",
            resolved_a.len(),
            100.0 * variant_total as f64 / resolved_a.len() as f64
        );
        for (layer, (variant, invariant)) in &per_layer {
            let label = if *layer == usize::MAX {
                "header_or_lm_head".to_string()
            } else {
                alloc::format!("layer_{layer}")
            };
            std::println!(
                "bound_op_classification bucket={label} variant={variant} invariant={invariant}"
            );
        }

        assert!(
            variant_total > 0,
            "the cache-reading ops must be classified variant, or this test cannot distinguish anything"
        );
        assert!(
            invariant_total > 0,
            "if every bound op is variant the resolve-once split has nothing to cache -- report this, do not build the split"
        );
    }

    /// The interpreter's per-node dispatch floor: how long a node costs
    /// when the node does essentially no arithmetic. This is the number the
    /// chunked-cache node budget above has to be multiplied by, because a
    /// chunk's own cache-reading nodes are tiny -- one 256-position slice
    /// of one head -- so what a chunk costs is dispatch, not math.
    ///
    /// Shaped as a balanced `Add` tree over `[1]`-shaped tensors, not a
    /// chain: `PROXIMA_CHAIN_DEPTH` below records that a linear chain
    /// overflows this evaluator's stack, and a balanced tree is what an
    /// N-way associative combine wants anyway.
    #[test]
    fn the_interpreter_per_node_dispatch_floor_is_measured() {
        const LEAVES: usize = 2_048;
        const REPEATS: usize = 20;

        let mut program = Vec::new();
        let seed = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(1)],
            "seed",
        );
        let mut level: Vec<NodeId> = (0..LEAVES)
            .map(|_| {
                elementwise(
                    &mut program,
                    DType::Float32,
                    ScalarOp::Add,
                    &[(seed, "a->a"), (seed, "a->a")],
                )
                .expect("a scalar add lowers")
            })
            .collect();
        while level.len() > 1 {
            level = level
                .chunks(2)
                .map(|pair| match pair {
                    [left, right] => elementwise(
                        &mut program,
                        DType::Float32,
                        ScalarOp::Add,
                        &[(*left, "a->a"), (*right, "a->a")],
                    )
                    .expect("a scalar add lowers"),
                    [only] => *only,
                    _ => unreachable!("chunks(2) yields one or two"),
                })
                .collect();
        }
        let root = level[0];
        let total = program.len();

        let seed_data = alloc::vec![1.0f32];
        let named: [(&str, &[f32]); 1] = [("seed", seed_data.as_slice())];

        let mut samples: Vec<f64> = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let started = std::time::Instant::now();
            crate::cpu::evaluate_named(&program, &[1], &named, &[root])
                .expect("the tree evaluates");
            samples.push(started.elapsed().as_secs_f64() * 1e9 / total as f64);
        }
        samples.sort_by(|left, right| left.partial_cmp(right).expect("no nan timings"));

        std::println!(
            "per_node_floor nodes={total} repeats={REPEATS} median_ns={:.1} min_ns={:.1} max_ns={:.1}",
            samples[REPEATS / 2],
            samples[0],
            samples[REPEATS - 1]
        );
        assert_eq!(samples.len(), REPEATS, "one timing per repeat");
    }

    /// How deep a dependency chain this evaluator survives. A flat N-chunk
    /// cache fold that combines chunks pairwise left-to-right builds a
    /// chain exactly N long, so this bounds that shape independently of the
    /// node-count budget. Depth comes from `PROXIMA_CHAIN_DEPTH` so a
    /// caller can walk it upward across separate processes -- a stack
    /// overflow aborts, it does not unwind, so one process cannot bisect it.
    #[test]
    #[ignore = "probes the evaluator's stack depth; aborts by design past the limit"]
    fn the_evaluator_survives_a_dependency_chain_of_a_given_depth() {
        let depth: usize = std::env::var("PROXIMA_CHAIN_DEPTH")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(256);

        let mut program = Vec::new();
        let seed = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(1)],
            "seed",
        );
        let mut tip = seed;
        for _ in 0..depth {
            tip = elementwise(
                &mut program,
                DType::Float32,
                ScalarOp::Add,
                &[(tip, "a->a"), (seed, "a->a")],
            )
            .expect("a scalar add chains");
        }

        let seed_data = alloc::vec![1.0f32];
        let named: [(&str, &[f32]); 1] = [("seed", seed_data.as_slice())];
        let evaluated = crate::cpu::evaluate_named(&program, &[1], &named, &[tip])
            .expect("the chain evaluates");
        let (data, _) = evaluated.get(tip).expect("chain tip present");

        std::println!(
            "chain_depth depth={depth} nodes={} value={}",
            program.len(),
            data[0]
        );
        assert_eq!(data[0], 1.0 + depth as f32, "each link adds one seed");
    }

    /// Wall-clock probe for the whole forward pass, `SEQUENCE=4`, RANDOM
    /// weights, at the model's real dimensions — the 32-layer analogue of
    /// `a_mistral_layer_written_as_toml_evaluates_at_its_real_dimensions`
    /// above, gated `#[ignore]` for the same reason and then some: ~32
    /// layers' worth of real-dimension weights is tens of GB and a
    /// multi-second-per-layer run, neither of which belongs in the default
    /// `nextest` budget. Run explicitly with `--ignored --release`.
    #[test]
    #[ignore = "measures the whole real-dimension mistral forward pass's wall clock; run explicitly"]
    fn the_whole_mistral_forward_pass_evaluates_at_real_dimensions() {
        const SEQUENCE: usize = 4;
        const VOCAB: usize = 32_002;
        const EMBEDDING: usize = 4096;
        const QUERY_HEADS: usize = 32;
        const KV_HEADS: usize = 8;
        const HEAD_DIM: usize = 128;
        const PAIRS: usize = HEAD_DIM / 2;
        const GROUP: usize = QUERY_HEADS / KV_HEADS;
        const FEED_FORWARD: usize = 14336;
        const BLOCK_COUNT: u32 = 32;

        let program = mistral_forward_program(
            VOCAB as u32,
            EMBEDDING as u32,
            FEED_FORWARD as u32,
            QUERY_HEADS as u32,
            KV_HEADS as u32,
            HEAD_DIM as u32,
            BLOCK_COUNT,
            0,
            0,
        )
        .expect("the whole forward pass lowers to a program");

        let symbols = [SEQUENCE as u64];
        crate::shape::infer(&program, &symbols).expect("the whole forward pass infers");

        // block order mirrors `mistral_forward_program`'s own `Input`
        // emission order exactly: ids, table, eps, cos/sin, then each
        // layer's attn_norm_weight/ffn_norm_weight/wq/wk/wv/wo/w_gate/
        // w_up/w_down, then the lm head. `inv_dim`, `ones`, `group_ones`,
        // `inv_sqrt_head_dim` and `neg_infinity` are `Op::Constant` now, so
        // none of them has a block here — that collapse is what
        // `no_repeated_scalar_crosses_the_binding_surface` asserts.
        // `block_node_ids` (cpu.rs) reads `Input`s positionally, which is
        // why this order is load-bearing, not cosmetic.
        let ids: Vec<f32> = (0..SEQUENCE)
            .map(|position| (position % VOCAB) as f32)
            .collect();
        let table = random_vec(200, VOCAB * EMBEDDING);
        let epsilon = alloc::vec![1e-5f32; SEQUENCE];
        let cos = random_vec(201, SEQUENCE * PAIRS);
        let sin = random_vec(202, SEQUENCE * PAIRS);

        let mut owned: Vec<Vec<f32>> = Vec::new();
        let mut seed = 300u64;
        for _layer in 0..BLOCK_COUNT {
            owned.push(alloc::vec![1.0f32; EMBEDDING]);
            owned.push(alloc::vec![1.0f32; EMBEDDING]);
            owned.push(random_vec(seed, EMBEDDING * QUERY_HEADS * HEAD_DIM));
            seed += 1;
            owned.push(random_vec(seed, EMBEDDING * KV_HEADS * HEAD_DIM));
            seed += 1;
            owned.push(random_vec(seed, EMBEDDING * KV_HEADS * HEAD_DIM));
            seed += 1;
            owned.push(random_vec(seed, KV_HEADS * GROUP * HEAD_DIM * EMBEDDING));
            seed += 1;
            owned.push(random_vec(seed, EMBEDDING * FEED_FORWARD));
            seed += 1;
            owned.push(random_vec(seed, EMBEDDING * FEED_FORWARD));
            seed += 1;
            owned.push(random_vec(seed, FEED_FORWARD * EMBEDDING));
            seed += 1;
        }
        let lm_head = random_vec(seed, EMBEDDING * VOCAB);

        let mut blocks: Vec<&[f32]> = alloc::vec![
            ids.as_slice(),
            table.as_slice(),
            epsilon.as_slice(),
            cos.as_slice(),
            sin.as_slice(),
        ];
        for layer_weights in &owned {
            blocks.push(layer_weights.as_slice());
        }
        blocks.push(lm_head.as_slice());

        let root = NodeId(program.len() as u32 - 1);
        let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");

        let wall_start = std::time::Instant::now();
        let evaluated =
            crate::cpu::evaluate_parallel(&program, &symbols, &blocks, &[root], workers)
                .expect("the whole real-dimension mistral forward pass evaluates");
        let wall = wall_start.elapsed();
        std::println!(
            "whole_forward_pass: wall_clock={wall:?} per_layer={:?}",
            wall / BLOCK_COUNT
        );

        let output = evaluated.root();
        assert_eq!(
            output.len(),
            SEQUENCE * VOCAB,
            "logits must be [seq, vocab]"
        );
        assert!(
            output.iter().all(|value| value.is_finite()),
            "logits must be finite"
        );

        let last_row = &output[(SEQUENCE - 1) * VOCAB..SEQUENCE * VOCAB];
        let argmax = last_row
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .expect("logits row is nonempty");
        assert!(
            argmax < VOCAB,
            "argmax {argmax} must address a real vocab entry, meaningless as it is with random weights"
        );
        std::println!("argmax(last position)={argmax}");
    }

    /// Absolute-position RoPE angles for `count` positions starting at
    /// `start` -- the same formula `bind.rs`'s `build_position_inputs`
    /// computes per call, generalized with a start offset so a decode
    /// step's lone new position gets its true absolute angle instead of
    /// position 0.
    fn rope_angles(
        start: usize,
        count: usize,
        pairs: usize,
        head_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut cos = alloc::vec![0.0f32; count * pairs];
        let mut sin = alloc::vec![0.0f32; count * pairs];
        for offset in 0..count {
            let position = (start + offset) as f32;
            for pair in 0..pairs {
                let theta = position
                    * crate::sized::ROPE_FREQ_BASE_DEFAULT
                        .powf(-((2 * pair) as f32) / (head_dim as f32));
                cos[offset * pairs + pair] = theta.cos();
                sin[offset * pairs + pair] = theta.sin();
            }
        }
        (cos, sin)
    }

    /// The falsifiable claim under test: a prefill call followed by a
    /// one-token decode call through [`mistral_cached_forward_program`]
    /// must produce the SAME last-position logits [`mistral_forward_program`]
    /// produces evaluating the whole sequence at once, with NO per-step
    /// growth in the amount of new work the decode call performs (it binds
    /// a fixed `N=1` symbol regardless of how long the cache has grown).
    /// This is the acceptance criterion from the task brief, proven here at
    /// tiny synthetic dimensions instead of the real 226-tensor checkpoint
    /// so a wrong index map fails in milliseconds, not after a 36-second
    /// real-model run.
    #[test]
    fn a_cached_decode_step_matches_the_uncached_forward_pass_exactly() {
        const VOCAB: usize = 5;
        const EMBEDDING: usize = 4;
        const FEED_FORWARD: usize = 4;
        const QUERY_HEADS: usize = 2;
        const KV_HEADS: usize = 1;
        const HEAD_DIM: usize = 2;
        const PAIRS: usize = HEAD_DIM / 2;
        const GROUP: usize = QUERY_HEADS / KV_HEADS;
        const BLOCK_COUNT: u32 = 2;
        const PROMPT_LEN: usize = 2;
        const SEQUENCE: usize = PROMPT_LEN + 1;

        let ids: Vec<u32> = alloc::vec![1, 3, 2];
        let ids_f32: Vec<f32> = ids.iter().map(|&id| id as f32).collect();

        let table = random_vec(10, VOCAB * EMBEDDING);
        let epsilon_full = alloc::vec![1e-5f32; SEQUENCE];
        let epsilon_one = alloc::vec![1e-5f32; 1];
        let epsilon_prompt = alloc::vec![1e-5f32; PROMPT_LEN];
        let (cos_full, sin_full) = rope_angles(0, SEQUENCE, PAIRS, HEAD_DIM);
        let (cos_prompt, sin_prompt) = rope_angles(0, PROMPT_LEN, PAIRS, HEAD_DIM);
        let (cos_decode, sin_decode) = rope_angles(PROMPT_LEN, 1, PAIRS, HEAD_DIM);

        struct LayerWeights {
            attn_norm: Vec<f32>,
            ffn_norm: Vec<f32>,
            wq: Vec<f32>,
            wk: Vec<f32>,
            wv: Vec<f32>,
            wo: Vec<f32>,
            w_gate: Vec<f32>,
            w_up: Vec<f32>,
            w_down: Vec<f32>,
        }

        let mut layers = Vec::new();
        let mut seed = 100u64;
        for _ in 0..BLOCK_COUNT {
            let weights = LayerWeights {
                attn_norm: alloc::vec![1.0f32; EMBEDDING],
                ffn_norm: alloc::vec![1.0f32; EMBEDDING],
                wq: random_vec(seed, EMBEDDING * QUERY_HEADS * HEAD_DIM),
                wk: random_vec(seed + 1, EMBEDDING * KV_HEADS * HEAD_DIM),
                wv: random_vec(seed + 2, EMBEDDING * KV_HEADS * HEAD_DIM),
                wo: random_vec(seed + 3, KV_HEADS * GROUP * HEAD_DIM * EMBEDDING),
                w_gate: random_vec(seed + 4, EMBEDDING * FEED_FORWARD),
                w_up: random_vec(seed + 5, EMBEDDING * FEED_FORWARD),
                w_down: random_vec(seed + 6, FEED_FORWARD * EMBEDDING),
            };
            seed += 7;
            layers.push(weights);
        }
        let output_norm = alloc::vec![1.0f32; EMBEDDING];
        let lm_head = random_vec(seed, EMBEDDING * VOCAB);

        // -- uncached oracle: the whole 3-token sequence in one shot.
        let uncached_program = mistral_forward_program(
            VOCAB as u32,
            EMBEDDING as u32,
            FEED_FORWARD as u32,
            QUERY_HEADS as u32,
            KV_HEADS as u32,
            HEAD_DIM as u32,
            BLOCK_COUNT,
            0,
            0,
        )
        .expect("uncached forward pass lowers");
        // real `blk.{layer}.*` names, built with `alloc::format!` so
        // ownership outlives the `&str` borrows below.
        let layer_names: Vec<[alloc::string::String; 9]> = layers
            .iter()
            .enumerate()
            .map(|(layer, _)| {
                [
                    alloc::format!("blk.{layer}.attn_norm.weight"),
                    alloc::format!("blk.{layer}.ffn_norm.weight"),
                    alloc::format!("blk.{layer}.attn_q.weight"),
                    alloc::format!("blk.{layer}.attn_k.weight"),
                    alloc::format!("blk.{layer}.attn_v.weight"),
                    alloc::format!("blk.{layer}.attn_output.weight"),
                    alloc::format!("blk.{layer}.ffn_gate.weight"),
                    alloc::format!("blk.{layer}.ffn_up.weight"),
                    alloc::format!("blk.{layer}.ffn_down.weight"),
                ]
            })
            .collect();
        let mut uncached_named: Vec<(&str, &[f32])> = alloc::vec![
            ("ids", ids_f32.as_slice()),
            ("token_embd.weight", table.as_slice()),
            ("eps", epsilon_full.as_slice()),
            ("rope_cos", cos_full.as_slice()),
            ("rope_sin", sin_full.as_slice())
        ];
        for (layer_index, weights) in layers.iter().enumerate() {
            let names = &layer_names[layer_index];
            uncached_named.push((names[0].as_str(), weights.attn_norm.as_slice()));
            uncached_named.push((names[1].as_str(), weights.ffn_norm.as_slice()));
            uncached_named.push((names[2].as_str(), weights.wq.as_slice()));
            uncached_named.push((names[3].as_str(), weights.wk.as_slice()));
            uncached_named.push((names[4].as_str(), weights.wv.as_slice()));
            uncached_named.push((names[5].as_str(), weights.wo.as_slice()));
            uncached_named.push((names[6].as_str(), weights.w_gate.as_slice()));
            uncached_named.push((names[7].as_str(), weights.w_up.as_slice()));
            uncached_named.push((names[8].as_str(), weights.w_down.as_slice()));
        }
        uncached_named.push(("output_norm.weight", output_norm.as_slice()));
        uncached_named.push(("output.weight", lm_head.as_slice()));

        let uncached_root = NodeId(uncached_program.len() as u32 - 1);
        let uncached_evaluated = crate::cpu::evaluate_named(
            &uncached_program,
            &[SEQUENCE as u64],
            &uncached_named,
            &[uncached_root],
        )
        .expect("uncached forward pass evaluates");
        let (uncached_logits, uncached_shape) = uncached_evaluated
            .get(uncached_root)
            .expect("uncached logits present");
        assert_eq!(uncached_shape, [SEQUENCE as u64, VOCAB as u64]);
        let uncached_last_position = &uncached_logits[(SEQUENCE - 1) * VOCAB..SEQUENCE * VOCAB];

        // -- cached path: prefill the first PROMPT_LEN positions, then one
        // decode step for the final position, growing the cache in between
        // exactly the way `bind.rs`'s decode loop would.
        let (cached_program, cached_logits_root, cache_roots) = mistral_cached_forward_program(
            VOCAB as u32,
            EMBEDDING as u32,
            FEED_FORWARD as u32,
            QUERY_HEADS as u32,
            KV_HEADS as u32,
            HEAD_DIM as u32,
            BLOCK_COUNT,
        )
        .expect("cached forward pass lowers");

        let empty_k_even = Vec::<f32>::new();
        let empty_k_odd = Vec::<f32>::new();
        let empty_v = Vec::<f32>::new();
        let prefill_cached_len = [0.0f32];
        let mut prefill_named: Vec<(&str, &[f32])> = alloc::vec![
            ("ids", &ids_f32[..PROMPT_LEN]),
            ("token_embd.weight", table.as_slice()),
            ("eps", epsilon_prompt.as_slice()),
            ("rope_cos", cos_prompt.as_slice()),
            ("rope_sin", sin_prompt.as_slice()),
            ("cached_len", prefill_cached_len.as_slice()),
        ];
        for (layer_index, weights) in layers.iter().enumerate() {
            let names = &layer_names[layer_index];
            prefill_named.push((names[0].as_str(), weights.attn_norm.as_slice()));
            prefill_named.push((names[1].as_str(), weights.ffn_norm.as_slice()));
            prefill_named.push((names[2].as_str(), weights.wq.as_slice()));
            prefill_named.push((names[3].as_str(), weights.wk.as_slice()));
            prefill_named.push((names[4].as_str(), weights.wv.as_slice()));
            prefill_named.push((names[5].as_str(), weights.wo.as_slice()));
            prefill_named.push((names[6].as_str(), weights.w_gate.as_slice()));
            prefill_named.push((names[7].as_str(), weights.w_up.as_slice()));
            prefill_named.push((names[8].as_str(), weights.w_down.as_slice()));
        }
        prefill_named.push(("output_norm.weight", output_norm.as_slice()));
        prefill_named.push(("output.weight", lm_head.as_slice()));
        let kv_cache_names: Vec<[alloc::string::String; 3]> = (0..BLOCK_COUNT as usize)
            .map(|layer| {
                [
                    alloc::format!("kv_cache.{layer}.k_even"),
                    alloc::format!("kv_cache.{layer}.k_odd"),
                    alloc::format!("kv_cache.{layer}.v"),
                ]
            })
            .collect();
        for names in &kv_cache_names {
            prefill_named.push((names[0].as_str(), empty_k_even.as_slice()));
            prefill_named.push((names[1].as_str(), empty_k_odd.as_slice()));
            prefill_named.push((names[2].as_str(), empty_v.as_slice()));
        }

        let mut prefill_roots: Vec<NodeId> = alloc::vec![cached_logits_root];
        for (even, odd, value) in &cache_roots {
            prefill_roots.push(*even);
            prefill_roots.push(*odd);
            prefill_roots.push(*value);
        }
        let prefill_symbols = [PROMPT_LEN as u64, 0u64];
        let prefill_evaluated = crate::cpu::evaluate_named(
            &cached_program,
            &prefill_symbols,
            &prefill_named,
            &prefill_roots,
        )
        .expect("prefill call evaluates");

        let mut k_even_cache: Vec<Vec<f32>> = Vec::with_capacity(BLOCK_COUNT as usize);
        let mut k_odd_cache: Vec<Vec<f32>> = Vec::with_capacity(BLOCK_COUNT as usize);
        let mut v_cache: Vec<Vec<f32>> = Vec::with_capacity(BLOCK_COUNT as usize);
        for (even, odd, value) in &cache_roots {
            let (even_data, _) = prefill_evaluated
                .get(*even)
                .expect("prefill k_even present");
            let (odd_data, _) = prefill_evaluated.get(*odd).expect("prefill k_odd present");
            let (value_data, _) = prefill_evaluated.get(*value).expect("prefill v present");
            k_even_cache.push(even_data.to_vec());
            k_odd_cache.push(odd_data.to_vec());
            v_cache.push(value_data.to_vec());
        }

        let decode_cached_len = [PROMPT_LEN as f32];
        let mut decode_named: Vec<(&str, &[f32])> = alloc::vec![
            ("ids", &ids_f32[PROMPT_LEN..]),
            ("token_embd.weight", table.as_slice()),
            ("eps", epsilon_one.as_slice()),
            ("rope_cos", cos_decode.as_slice()),
            ("rope_sin", sin_decode.as_slice()),
            ("cached_len", decode_cached_len.as_slice()),
        ];
        for (layer_index, weights) in layers.iter().enumerate() {
            let names = &layer_names[layer_index];
            decode_named.push((names[0].as_str(), weights.attn_norm.as_slice()));
            decode_named.push((names[1].as_str(), weights.ffn_norm.as_slice()));
            decode_named.push((names[2].as_str(), weights.wq.as_slice()));
            decode_named.push((names[3].as_str(), weights.wk.as_slice()));
            decode_named.push((names[4].as_str(), weights.wv.as_slice()));
            decode_named.push((names[5].as_str(), weights.wo.as_slice()));
            decode_named.push((names[6].as_str(), weights.w_gate.as_slice()));
            decode_named.push((names[7].as_str(), weights.w_up.as_slice()));
            decode_named.push((names[8].as_str(), weights.w_down.as_slice()));
        }
        decode_named.push(("output_norm.weight", output_norm.as_slice()));
        decode_named.push(("output.weight", lm_head.as_slice()));
        for (layer_index, names) in kv_cache_names.iter().enumerate() {
            decode_named.push((names[0].as_str(), k_even_cache[layer_index].as_slice()));
            decode_named.push((names[1].as_str(), k_odd_cache[layer_index].as_slice()));
            decode_named.push((names[2].as_str(), v_cache[layer_index].as_slice()));
        }

        let decode_symbols = [1u64, PROMPT_LEN as u64];
        let decode_evaluated = crate::cpu::evaluate_named(
            &cached_program,
            &decode_symbols,
            &decode_named,
            &[cached_logits_root],
        )
        .expect("decode call evaluates");
        let (decode_logits, decode_shape) = decode_evaluated
            .get(cached_logits_root)
            .expect("decode logits present");
        assert_eq!(decode_shape, [1u64, VOCAB as u64]);

        let max_diff = uncached_last_position
            .iter()
            .zip(decode_logits.iter())
            .map(|(oracle, cached)| (oracle - cached).abs())
            .fold(0.0f32, f32::max);
        std::println!(
            "cached_decode_vs_uncached: oracle={uncached_last_position:?} cached={decode_logits:?} max_diff={max_diff}"
        );
        assert!(
            uncached_last_position
                .iter()
                .any(|&value| value != uncached_last_position[0]),
            "degenerate control: oracle logits are all-equal, this run proves nothing"
        );
        assert!(
            max_diff < 1e-4,
            "cached decode step diverged from the uncached oracle: max_diff={max_diff}"
        );
    }

    /// [`causal_conv1d`]'s whole reason for existing, checked against
    /// arithmetic worked out by hand rather than trusted from the
    /// implementation: `l_cache=3`, one channel, `weight = [1, 10, 100]`
    /// (tap `l=2` is the current position, `l=0` the furthest lookback --
    /// [`append_lfm2_conv_mixer`]'s own convention), `x = [1, 2, 3, 4]`.
    /// `out[s] = sum_l valid(s,l) * weight[l] * x[s - 2 + l]`, zero where the
    /// window reaches before position 0:
    /// - `out[0] = 100*x[0]                               = 100`
    /// - `out[1] = 10*x[0]  + 100*x[1]                    = 210`
    /// - `out[2] = 1*x[0]   + 10*x[1]  + 100*x[2]         = 321`
    /// - `out[3] = 1*x[1]   + 10*x[2]  + 100*x[3]         = 432`
    #[proxima::test]
    async fn causal_conv1d_matches_a_hand_computed_causal_window() {
        let mut program = Vec::new();
        let x = op::append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Symbolic(0), Extent::Static(1)],
                name: Some("x".into()),
            },
        );
        let weight = op::append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                // `[embedding=1, l_cache=3]`, matching `causal_conv1d`'s own
                // `dl->sld` map -- `l_cache` last/fastest, the real
                // checkpoint's own on-disk axis order (`spec.rs`'s own doc on
                // this map explains why). One channel makes this
                // indistinguishable from the old `[3, 1]` shape byte-for-byte
                // (see `causal_conv1d_catches_a_transposed_multi_channel_weight`
                // below for the shape this single-channel test cannot catch).
                shape: alloc::vec![Extent::Static(1), Extent::Static(3)],
                name: Some("weight".into()),
            },
        );
        let output = causal_conv1d(&mut program, x, weight, 3).expect("causal conv lowers");

        let x_data = [1.0f32, 2.0, 3.0, 4.0];
        let weight_data = [1.0f32, 10.0, 100.0];
        let evaluated = crate::cpu::evaluate_named(
            &program,
            &[4],
            &[("x", &x_data), ("weight", &weight_data)],
            &[output],
        )
        .expect("causal conv evaluates");
        let (result, shape) = evaluated.get(output).expect("conv output present");

        std::println!("causal_conv1d result={result:?} shape={shape:?}");
        assert_eq!(shape, [4u64, 1u64]);
        assert_eq!(result, [100.0, 210.0, 321.0, 432.0]);
    }

    /// **The rule census.** For the real Mistral/OpenChat cached-forward
    /// program, names every rewrite [`crate::bind::bind`] actually applies
    /// and counts how many times each fired, reconciling against
    /// `docs/discipline.md`'s own measured split (ROW at line 4780): 1196
    /// `BoundOp`s, 225 `reduce_matmul_quantized`, 385 `reduce_f32_dense`,
    /// 547 `elementwise`, 37 `constant`, 2 `iota`.
    ///
    /// The 225/385 split within `docs/discipline.md`'s own figure is a
    /// RUNTIME classification (`cpu::is_quantized_matmul_operand`'s own
    /// discriminator: whether a bound weight buffer's byte length matches
    /// its declared `Float32` element count, or is smaller because the
    /// checkpoint actually stored it `Q4_K`/`Q5_K`/`Q6_K` packed) -- it is
    /// not recoverable from this symbolic program alone, which declares
    /// every weight `DType::Float32` (`mistral_cached_forward_program_with_
    /// experts`, `spec.rs:4561-4646` and onward: every `input_leaf` weight
    /// call passes `DType::Float32`, never a quantized tag). Reproducing it
    /// here would require binding the real `openchat-3.5-1210.Q4_K_S.gguf`
    /// checkpoint's own weight bytes, out of this round's scope. What IS a
    /// structural (graph-topology) property, checked below by an inline
    /// mirror of `cpu::reduce_is_gemm_shaped`'s own distinct-operand-count
    /// discriminator (a `#[cfg]`-gated private fn, not reachable across this
    /// module boundary without changing its visibility): whether a reduce
    /// reads one operand (LayerNorm mean/variance, softmax max/sum) or two
    /// (every matmul-shaped fold, weight-projection AND attention-score
    /// alike) -- a DIFFERENT, coarser partition than quantized/dense, since
    /// attention's own `Q@K^T`/`softmax@V`/cached-score folds are also
    /// two-operand but never weight-quantized. Measured 417 two-operand /
    /// 193 one-operand, summing to the same 610 total the 225+385 figure
    /// does -- the total reconciles exactly; the sub-split names a
    /// different, real distinction, not the same one.
    ///
    /// Also reads back the two elementwise-fusion decisions inline inside
    /// `BoundOpBuilder::push` (`bind.rs:641`, `bind.rs:697`) and the
    /// masked-window-reduce elimination (`bind.rs:659`) under the
    /// `instrument` feature -- fired/declined, never a bare bool, so a rule
    /// that fires zero times on this program is visibly distinct from a
    /// rule that does not exist.
    #[cfg(feature = "instrument")]
    #[test]
    fn the_rule_census_reconciles_against_the_measured_mistral_forward_split() {
        crate::instrument::reset_fuse_decline();
        crate::instrument::reset_window_reduce();
        crate::instrument::reset_online_softmax_block_range();

        let (program, logits, roots) =
            mistral_cached_forward_program(32_002, 4096, 14336, 32, 8, 128, 32)
                .expect("the cached forward pass lowers to a program");
        let mut outputs = alloc::vec![logits];
        for (even, odd, value) in &roots {
            outputs.extend_from_slice(&[*even, *odd, *value]);
        }
        let shapes = crate::shape::infer(&program, &[1, 71])
            .expect("one new position against a 71-position cache infers");
        // fusion held explicitly off: this census documents the UNFUSED
        // split, so it must bind that way under every feature combination,
        // including `cached-attention-streaming`, where `bind`'s own
        // default (`bind_with_fusion(.., true)`) would otherwise fuse 32
        // attention chains into `BoundOpKind::CachedAttention` and silently
        // invalidate every count below.
        let bound = crate::bind::bind_with_fusion(
            &program,
            &shapes,
            &outputs,
            false,
            crate::numeric::NumericPolicy::default(),
        )
        .expect("the program binds");
        // `reduce-epilogue-fusion` is a bind-time REWRITE gated only by this
        // crate feature (`bind::bind_with_fusion`'s own doc: it "runs
        // unconditionally after this ... gated only by the crate feature"),
        // not by the `fuse_cached_attention` bool this call passes `false`
        // for -- so `bound` above already carries the epilogue-fused shape
        // whenever this feature is compiled in, and every count below that
        // depends on it must be read per-feature, never pinned to one number.
        #[cfg(feature = "reduce-epilogue-fusion")]
        let epilogued_reduce_count = bound
            .iter()
            .filter(|op| {
                matches!(
                    &op.kind,
                    crate::bind::BoundOpKind::Reduce { epilogue_operands, .. }
                        if !epilogue_operands.is_empty()
                )
            })
            .count();
        // MEASURED (this test, `reduce-epilogue-fusion` on): 128 = 4 fusions
        // x 32 layers, one per `append_mistral_cached_layer` call
        // (`spec.rs:2378`). Each fusion is an `Op::Elementwise` whose SOLE
        // operand-of-interest is an `Op::Reduce` with no other consumer, read
        // at full identity -- exactly `bind::reduce_epilogue_candidates`'s
        // own three conditions -- so the reduce's own `BoundOp` disappears
        // and its producer becomes the consumer's epilogue instead. The four,
        // in per-layer source order:
        //   1. `global_max` (`spec.rs:2678`, `Elementwise::Maximum` over
        //      `score_max_cached`/`score_max_new`) absorbs `score_max_cached`
        //      (the first, hence first-matching, `Reduce` operand) as its
        //      epilogue -- the online-softmax running-max combine.
        //   2. `residual1` (`spec.rs:2809`, `Elementwise::Add` of `attn_out`
        //      and `x`) absorbs `attn_out`, the attention output-projection
        //      reduce (`spec.rs:2799`).
        //   3. `ffn_hidden` (`spec.rs:2879`, `Elementwise::Multiply` of
        //      `silu_gate` and `up`) absorbs `up`, the FFN up-projection
        //      reduce (`spec.rs:2839`).
        //   4. `x_next` (`spec.rs:2902`, `Elementwise::Add` of `ffn_out` and
        //      `residual1`) absorbs `ffn_out`, the FFN down-projection
        //      reduce (`spec.rs:2892`).
        // Every other `Reduce` in the layer (Q/K/V projections, the two
        // per-range attention-score reduces, the two per-range softmax-sum
        // reduces, the two per-range attended-value reduces) keeps a second
        // real consumer or a non-identity/broadcast one, so none of them
        // qualifies -- this is why the count is 4/layer, not higher.
        #[cfg(feature = "reduce-epilogue-fusion")]
        assert_eq!(
            epilogued_reduce_count,
            4 * 32,
            "reduce-epilogue-fusion must absorb exactly 4 reduces per layer on this \
             32-layer Mistral cached-forward program -- global_max, residual1's attn_out, \
             ffn_hidden's up-projection, and x_next's down-projection reduce"
        );

        let mut elementwise = 0_usize;
        let mut reduce_two_operand = 0_usize;
        let mut reduce_one_operand = 0_usize;
        let mut constant = 0_usize;
        let mut iota = 0_usize;
        // `BoundOpKind::CachedAttention` (landed after this census's own
        // baseline figures were measured) is a bind-time fusion held off
        // explicitly above via `bind_with_fusion(.., false)` -- this
        // program never reaches it here regardless of feature set, so it
        // is counted separately as a proof the unfused bind stayed
        // unfused, not merely a side effect of a feature flag being off.
        // The feature-gated block below re-binds WITH fusion on to assert
        // the fused count directly.
        let mut cached_attention = 0_usize;
        for op in &bound {
            match &op.kind {
                crate::bind::BoundOpKind::CachedAttention { .. } => cached_attention += 1,
                crate::bind::BoundOpKind::Elementwise { .. } => elementwise += 1,
                crate::bind::BoundOpKind::Reduce { .. } => {
                    let operands = op.operands();
                    let is_two_operand = operands
                        .first()
                        .is_some_and(|(first, _, _)| operands.iter().any(|(node, _, _)| node != first));
                    if is_two_operand {
                        reduce_two_operand += 1;
                    } else {
                        reduce_one_operand += 1;
                    }
                }
                crate::bind::BoundOpKind::Constant { .. } => constant += 1,
                crate::bind::BoundOpKind::Iota => iota += 1,
            }
        }
        assert_eq!(
            cached_attention, 0,
            "this program/feature-set was expected to never fuse a CachedAttention BoundOp -- \
             the census's four-bucket reconciliation below needs updating if that changed"
        );
        let total = bound.len();
        let reduce_total = reduce_two_operand + reduce_one_operand;

        let (fuse_elementwise_fired, fuse_reduce_fired, fuse_distinct_declines) =
            crate::instrument::fuse_totals();
        let (window_reduce_fired, window_reduce_declined) =
            crate::instrument::window_reduce_totals();

        std::println!(
            "rule_census total={total} reduce_total={reduce_total} reduce_two_operand={reduce_two_operand} reduce_one_operand={reduce_one_operand} elementwise={elementwise} constant={constant} iota={iota}"
        );
        std::println!(
            "rule_census fuse_elementwise_operand_fired={fuse_elementwise_fired} fuse_reduce_operand_fired={fuse_reduce_fired} fuse_distinct_declines={fuse_distinct_declines} window_reduce_fired={window_reduce_fired} window_reduce_declined={window_reduce_declined}"
        );

        // costing-vs-free split (2026-09-02 correction, verified against
        // source): `Op::Input` (`bind.rs:607`, `push`'s own match arm)
        // emits NOTHING -- no `BoundOp`, no dispatch, confirmed by this
        // module's own doc ("`Op::Input` never does -- it is where data
        // enters"). `Op::Constant`/`Op::Iota` DO have a real `push` arm
        // (`bind.rs:621`, and the `Iota` arm above it) and this census's
        // own total already proves both dispatch: `constant=37 iota=2` are
        // real `BoundOp`s. So FREE is `Op::Input` alone; COSTING is
        // `Elementwise`, `Reduce`, `Constant`, or `Iota` -- a decline whose
        // operand is any of those corresponds to a materialized, dispatched
        // buffer, exactly like an `Elementwise` decline does.
        let is_free_operand = |node: u32| matches!(program[node as usize], Op::Input { .. });

        let mut still_live_costing = 0_u64;
        let mut still_live_free = 0_u64;
        let mut non_identity_costing = 0_u64;
        let mut non_identity_free = 0_u64;
        let mut not_held_costing = 0_u64;
        let mut not_held_free = 0_u64;
        for (node, _site, reason, calls) in crate::instrument::fuse_decline_snapshot() {
            let free = is_free_operand(node);
            match (reason, free) {
                (crate::instrument::FuseDeclineReason::StillLive, true) => still_live_free += calls,
                (crate::instrument::FuseDeclineReason::StillLive, false) => {
                    still_live_costing += calls;
                }
                (crate::instrument::FuseDeclineReason::NonIdentityProjection, true) => {
                    non_identity_free += calls;
                }
                (crate::instrument::FuseDeclineReason::NonIdentityProjection, false) => {
                    non_identity_costing += calls;
                }
                (crate::instrument::FuseDeclineReason::NotHeld, true) => not_held_free += calls,
                (crate::instrument::FuseDeclineReason::NotHeld, false) => {
                    not_held_costing += calls;
                }
                // `quarantine_broadcast_operands` only ever walks children
                // still `held` (`bind.rs:895`'s own `contains_key` guard),
                // and a `held` node is by construction an `Op::Elementwise`
                // chain, never an `Op::Input` leaf -- so this reason is
                // always costing, counted separately below rather than
                // folded into the free/costing split above.
                (crate::instrument::FuseDeclineReason::BroadcastQuarantined, _) => {}
            }
        }
        let still_live_total = still_live_costing + still_live_free;
        let non_identity_total = non_identity_costing + non_identity_free;
        let not_held_total = not_held_costing + not_held_free;
        std::println!(
            "rule_census fuse_decline_still_live={still_live_total} costing={still_live_costing} free={still_live_free}"
        );
        std::println!(
            "rule_census fuse_decline_non_identity_projection={non_identity_total} costing={non_identity_costing} free={non_identity_free}"
        );
        std::println!(
            "rule_census fuse_decline_not_held={not_held_total} costing={not_held_costing} free={not_held_free}"
        );

        // quarantine-broadcast census (this round's task): `bind.rs:901`'s
        // `quarantine_broadcast_operands` is a genuine fuse/no-fuse decision
        // invisible to the counters above -- a decline here looks identical
        // to a rule that never ran without its own site. `N == 0` is a red
        // gate the same way `total > 0` below is.
        let mut quarantine_broadcast_declines = 0_u64;
        for (_node, site, reason, calls) in crate::instrument::fuse_decline_snapshot() {
            if site == crate::instrument::FuseSite::QuarantineBroadcast
                && reason == crate::instrument::FuseDeclineReason::BroadcastQuarantined
            {
                quarantine_broadcast_declines += calls;
            }
        }
        std::println!(
            "rule_census quarantine_broadcast_declines={quarantine_broadcast_declines}"
        );
        assert!(
            quarantine_broadcast_declines > 0,
            "rule census recorded zero quarantine-broadcast declines -- the site is wired to nothing"
        );
        assert_eq!(
            quarantine_broadcast_declines, 65,
            "quarantine-broadcast declines drifted off the measured Mistral cached-forward count"
        );

        // rope_cos/rope_sin named check: both are `Op::Input` (node 7, 8),
        // so under the corrected rule they must land 100% free, on every
        // reason, not just `StillLive`.
        for (node, label) in [(7_u32, "rope_cos"), (8, "rope_sin")] {
            let mut costing = 0_u64;
            let mut free = 0_u64;
            for (decline_node, _site, _reason, calls) in crate::instrument::fuse_decline_snapshot() {
                if decline_node != node {
                    continue;
                }
                if is_free_operand(decline_node) {
                    free += calls;
                } else {
                    costing += calls;
                }
            }
            std::println!(
                "rule_census named_node_check node={node} label={label} costing={costing} free={free}"
            );
            assert_eq!(costing, 0, "{label} (node {node}) is an Op::Input leaf -- every one of its declines must be free");
        }

        // deliverable #4: is `non_identity_projection=129` here the SAME
        // shape class as `width_tile_plan`'s `AxesShape=129` on node 90 in
        // the CPU train lane (`cpu.rs:9940`), or a bare numeric coincidence?
        // These are declines from THIS test's own bind-time fusion check
        // (`bind.rs:641`/`:697`), a different mechanism on a different
        // program (Mistral cached-forward here, BGE there) -- print which
        // node(s)/op(s) actually produce the 129 here so the comparison can
        // be made on evidence, not on the number alone.
        let mut non_identity_nodes: alloc::vec::Vec<(u32, u64)> = alloc::vec::Vec::new();
        for (node, _site, reason, calls) in crate::instrument::fuse_decline_snapshot() {
            if reason == crate::instrument::FuseDeclineReason::NonIdentityProjection {
                non_identity_nodes.push((node, calls));
            }
        }
        non_identity_nodes.sort_by_key(|entry| core::cmp::Reverse(entry.1));
        for (node, calls) in non_identity_nodes.iter().take(5) {
            let op = &program[*node as usize];
            std::println!(
                "rule_census non_identity_projection_node node={node} calls={calls} op={op:?}"
            );
        }
        std::println!(
            "rule_census non_identity_projection_distinct_nodes={}",
            non_identity_nodes.len()
        );

        // WHICH ops, not just how many (2026-09-02 follow-up): `fuse_decline_
        // snapshot` is now keyed by the STILL-LIVE OPERAND's own `NodeId`
        // (bind.rs's own fix -- it previously keyed by the consuming node,
        // which names WHERE a decline was checked, not WHAT had to
        // materialize). `online_softmax_block_ranges` brackets the combine
        // block (`spec.rs:2596-2726`) by construction -- `score_max_cached`
        // and `attended`, the block's own first/last emitted `NodeId`s,
        // recorded once per `append_mistral_cached_layer` call, never
        // assumed from reading the source alone.
        let block_ranges = crate::instrument::online_softmax_block_ranges();
        assert_eq!(
            block_ranges.len(),
            32,
            "one online-softmax block range per layer on a 32-layer forward"
        );
        let stride = block_ranges[1].0 - block_ranges[0].0;
        for window in block_ranges.windows(2) {
            assert_eq!(
                window[1].0 - window[0].0,
                stride,
                "every layer's block must start the same distance from the \
                 previous layer's -- confirms structural periodicity by \
                 measurement rather than assuming it from the source"
            );
            assert_eq!(
                window[0].1 - window[0].0,
                window[1].1 - window[1].0,
                "every layer's block must span the same number of nodes"
            );
        }
        let block_span = block_ranges[0].1 - block_ranges[0].0;
        std::println!(
            "rule_census online_softmax_block layers=32 stride={stride} block_span_nodes={} first_layer_range=[{},{}]",
            block_span + 1,
            block_ranges[0].0,
            block_ranges[0].1
        );

        let in_any_block = |node: u32| {
            block_ranges
                .iter()
                .any(|&(first, last)| node >= first && node <= last)
        };

        let mut still_live_costing_in_block = 0_u64;
        let mut still_live_costing_out_of_block = 0_u64;
        let mut still_live_free_in_block = 0_u64;
        let mut still_live_free_out_of_block = 0_u64;
        // phase = this operand's `NodeId` distance from ITS OWN layer's
        // block start, `rem_euclid(stride)` folding every layer onto one
        // canonical 0..stride ruler -- the "position within a layer" the
        // task asked for, measured against the real per-layer stride rather
        // than the raw-Op approximation. Split costing/free so a free
        // (`Op::Input`) repeat-offender like `rope_cos`/`rope_sin` cannot
        // hide inside the same ranking as a real materialize cost.
        let mut costing_phase_totals: alloc::collections::BTreeMap<u32, u64> =
            alloc::collections::BTreeMap::new();
        let mut costing_phase_example_node: alloc::collections::BTreeMap<u32, u32> =
            alloc::collections::BTreeMap::new();
        for (node, _site, reason, calls) in crate::instrument::fuse_decline_snapshot() {
            if reason != crate::instrument::FuseDeclineReason::StillLive {
                continue;
            }
            let free = is_free_operand(node);
            let inside = in_any_block(node);
            match (free, inside) {
                (true, true) => still_live_free_in_block += calls,
                (true, false) => still_live_free_out_of_block += calls,
                (false, true) => still_live_costing_in_block += calls,
                (false, false) => still_live_costing_out_of_block += calls,
            }
            if !free {
                let phase = (node.wrapping_sub(block_ranges[0].0)).rem_euclid(stride);
                *costing_phase_totals.entry(phase).or_insert(0) += calls;
                costing_phase_example_node.entry(phase).or_insert(node);
            }
        }
        let still_live_costing_total = still_live_costing_in_block + still_live_costing_out_of_block;
        std::println!(
            "rule_census still_live_costing_in_block={still_live_costing_in_block} still_live_costing_outside_block={still_live_costing_out_of_block} still_live_costing_total={still_live_costing_total}"
        );
        let still_live_costing_per_layer = still_live_costing_total as f64 / 32.0;
        std::println!(
            "rule_census still_live_costing_per_layer={still_live_costing_per_layer:.2} layers=32 program_wide={still_live_costing_total}"
        );
        std::println!(
            "rule_census still_live_free_in_block={still_live_free_in_block} still_live_free_outside_block={still_live_free_out_of_block}"
        );
        assert!(
            still_live_costing_total > 0,
            "rule census recorded zero costing still-live declines -- the split is wired to nothing"
        );
        // deliverable answer: what fraction of the COSTING total is the
        // online-softmax block's 96 (all three of global_max/weights_cached/
        // weights_new are Elementwise, so all 96 are costing by construction
        // -- confirmed below, not assumed).
        let block_fraction_permille =
            still_live_costing_in_block * 1000 / still_live_costing_total;
        std::println!(
            "rule_census online_softmax_block_share_of_costing calls={still_live_costing_in_block} of={still_live_costing_total} permille={block_fraction_permille}"
        );

        // named by construction order within the block (score_max_cached is
        // phase 0, the block's own first node): global_max is the 3rd node
        // emitted (phase 2), weights_cached the 5th (phase 4), weights_new
        // the 7th (phase 6) -- `spec.rs:2616,2632,2644`, read directly off
        // this test's own doc trace of the block, not guessed. All three
        // are `Op::Elementwise`, so they land in `costing_phase_totals`.
        for (phase, label) in [(2_u32, "global_max"), (4, "weights_cached"), (6, "weights_new")] {
            let calls = costing_phase_totals.get(&phase).copied().unwrap_or(0);
            std::println!("rule_census still_live_costing_phase={phase} label={label} calls={calls}");
        }

        let mut ranked_costing_phases: alloc::vec::Vec<(u32, u64)> =
            costing_phase_totals.into_iter().collect();
        ranked_costing_phases.sort_by_key(|entry| core::cmp::Reverse(entry.1));
        for (phase, calls) in ranked_costing_phases.iter().take(10) {
            std::println!("rule_census still_live_costing_top_phase phase={phase} calls={calls}");
        }
        for (phase, calls) in ranked_costing_phases.iter().take(6) {
            let node = costing_phase_example_node.get(phase).copied().unwrap_or(0);
            let op = &program[node as usize];
            std::println!(
                "rule_census still_live_costing_top_phase_op phase={phase} calls={calls} node={node} op={op:?}"
            );
        }

        // sanity check: 1196 = 610 reduce + 547 elementwise + 37 constant +
        // 2 iota. A costing decline whose operand is `Op::Elementwise`
        // corresponds to a node that either fuses away for free (never a
        // `BoundOp`) or is forced to materialize as one of the 547
        // elementwise `BoundOp`s -- the distinct count of Elementwise-kind
        // nodes that appear ANYWHERE in the decline snapshot (any of the
        // three reasons) is the direct witness for "forced to materialize
        // at least once", checked against 547 rather than assumed to agree
        // with it.
        let mut raw_elementwise_total = 0_u64;
        let mut raw_reduce_total = 0_u64;
        let mut raw_constant_total = 0_u64;
        let mut raw_iota_total = 0_u64;
        let mut raw_input_total = 0_u64;
        for op in &program {
            match op {
                Op::Elementwise { .. } => raw_elementwise_total += 1,
                Op::Reduce(_) => raw_reduce_total += 1,
                Op::Constant { .. } => raw_constant_total += 1,
                Op::Iota { .. } => raw_iota_total += 1,
                Op::Input { .. } => raw_input_total += 1,
            }
        }
        std::println!(
            "rule_census raw_op_totals elementwise={raw_elementwise_total} reduce={raw_reduce_total} constant={raw_constant_total} iota={raw_iota_total} input={raw_input_total} program_len={}",
            program.len()
        );
        assert_eq!(
            raw_reduce_total, 610,
            "reduces never fuse (build_reduce_op always yields exactly one BoundOp), \
             so the raw Op::Reduce count must equal the bound reduce_total exactly"
        );

        let mut elementwise_declined_nodes: alloc::collections::BTreeSet<u32> =
            alloc::collections::BTreeSet::new();
        for (node, _site, _reason, _calls) in crate::instrument::fuse_decline_snapshot() {
            if matches!(program[node as usize], Op::Elementwise { .. }) {
                elementwise_declined_nodes.insert(node);
            }
        }
        std::println!(
            "rule_census elementwise_declined_distinct_nodes={} elementwise_bound_ops=547 raw_elementwise_total={raw_elementwise_total}",
            elementwise_declined_nodes.len()
        );

        // REMATERIALIZATION sizing (2026-09-02, coordinator's rule -- NOT
        // implemented this round, only sized). Consumer count is computed
        // by a full scan of `program` (every place a NodeId is read as an
        // operand), independent of the decline snapshot -- the snapshot
        // only records DECLINE events, not total readers, so it cannot
        // answer "how many consumers" on its own.
        let mut consumer_count: alloc::collections::BTreeMap<u32, u64> =
            alloc::collections::BTreeMap::new();
        let count_map_indices = |map: &IndexMap, counts: &mut alloc::collections::BTreeMap<u32, u64>| {
            if let IndexMap::Computed { indices, .. } = map {
                *counts.entry(indices.0).or_insert(0) += 1;
            }
        };
        for op in &program {
            match op {
                Op::Elementwise { operands, .. } => {
                    for (operand_node, map) in operands {
                        *consumer_count.entry(operand_node.0).or_insert(0) += 1;
                        count_map_indices(map, &mut consumer_count);
                    }
                }
                Op::Reduce(reduce) => {
                    *consumer_count.entry(reduce.operand.0).or_insert(0) += 1;
                    count_map_indices(&reduce.in_map, &mut consumer_count);
                    count_map_indices(&reduce.out_map, &mut consumer_count);
                }
                Op::Input { .. } | Op::Constant { .. } | Op::Iota { .. } => {}
            }
        }

        // deliverable #1: of the 1086 costing still_live declines, how many
        // have an Op::Elementwise operand (the only rematerialization
        // candidate -- a Reduce always dispatches regardless of fusion, so
        // recomputing one buys nothing). A node that is ITSELF a named
        // graph output is excluded: `live::annotate` never retires an
        // output (it must persist to the end regardless of any single
        // consumer), so it declines StillLive even with as few as one
        // real consumer -- and rematerializing it into that consumer would
        // still leave the required output buffer unmaterialized, so it is
        // not a real candidate, not an undercounted one.
        let output_node_set: alloc::collections::BTreeSet<u32> =
            outputs.iter().map(|node| node.0).collect();
        let mut still_live_elementwise_candidates: alloc::collections::BTreeSet<u32> =
            alloc::collections::BTreeSet::new();
        let mut still_live_elementwise_calls = 0_u64;
        let mut still_live_non_elementwise_costing_calls = 0_u64;
        let mut still_live_output_pinned_calls = 0_u64;
        for (node, _site, reason, calls) in crate::instrument::fuse_decline_snapshot() {
            if reason != crate::instrument::FuseDeclineReason::StillLive || is_free_operand(node) {
                continue;
            }
            if !matches!(program[node as usize], Op::Elementwise { .. }) {
                still_live_non_elementwise_costing_calls += calls;
                continue;
            }
            if output_node_set.contains(&node) {
                still_live_output_pinned_calls += calls;
                continue;
            }
            still_live_elementwise_candidates.insert(node);
            still_live_elementwise_calls += calls;
        }
        std::println!(
            "rule_census rematerialize_candidates distinct_nodes={} decline_calls={still_live_elementwise_calls} non_elementwise_costing_calls={still_live_non_elementwise_costing_calls} output_pinned_calls={still_live_output_pinned_calls} of_costing_still_live={still_live_costing_total}",
            still_live_elementwise_candidates.len()
        );
        assert!(
            !still_live_elementwise_candidates.is_empty(),
            "rule census found zero rematerialization candidates -- either the rule genuinely \
             does not apply here or the measurement is wired to nothing"
        );

        // deliverable #2: consumer-count histogram over the candidates.
        let mut consumer_histogram: alloc::collections::BTreeMap<u64, u64> =
            alloc::collections::BTreeMap::new();
        for &node in &still_live_elementwise_candidates {
            let count = consumer_count.get(&node).copied().unwrap_or(0);
            *consumer_histogram.entry(count).or_insert(0) += 1;
        }
        for (consumers, nodes) in &consumer_histogram {
            std::println!("rule_census rematerialize_histogram consumers={consumers} nodes={nodes}");
        }
        assert!(
            consumer_histogram.keys().all(|&count| count >= 2),
            "a StillLive decline means another consumer exists later -- every candidate must \
             show at least 2 total consumers, or the consumer-count scan disagrees with the \
             decline mechanism itself"
        );

        // deliverable #3: projected dispatch saving at three thresholds.
        for threshold in [2_u64, 3, 4] {
            let saved = still_live_elementwise_candidates
                .iter()
                .filter(|&&node| consumer_count.get(&node).copied().unwrap_or(0) <= threshold)
                .count();
            let projected_elementwise = elementwise - saved;
            let projected_total = total - saved;
            std::println!(
                "rule_census rematerialize_projection threshold={threshold} saved_dispatches={saved} elementwise_547_to={projected_elementwise} total_1196_to={projected_total} fraction_of_1196={:.4}",
                saved as f64 / total as f64
            );
        }

        // deliverable #4: the ALU cost side, honest and unrounded. Element
        // counts come from `shapes` (the same `Shapes` table `bind::bind`
        // itself resolved against), never guessed from rank alone.
        let mut recompute_elements_total = 0_u128;
        let mut saved_dispatches_at_2 = 0_u64;
        for &node in &still_live_elementwise_candidates {
            let consumers = consumer_count.get(&node).copied().unwrap_or(0);
            if consumers > 2 {
                continue;
            }
            saved_dispatches_at_2 += 1;
            let extents = shapes.of(crate::op::NodeId(node));
            let element_count: u128 = extents.iter().map(|&extent| extent as u128).product();
            recompute_elements_total += element_count * u128::from(consumers - 1);
        }
        let dispatch_floor_ns = 4_000_u128; // coordinator's own cited ~4us floor
        let dispatch_saving_ns = u128::from(saved_dispatches_at_2) * dispatch_floor_ns;
        // MEASURED range from `docs/discipline.md`'s own instrument-counter
        // table (elementwise Generic fast=2.31 ns/element, slow=16.18
        // ns/element, a real decode step) -- reported as a range, not a
        // single assumed constant, because which arm a rematerialized body
        // would hit is not measured by this census.
        let alu_cost_fast_ns = recompute_elements_total * 231 / 100;
        let alu_cost_slow_ns = recompute_elements_total * 1618 / 100;
        std::println!(
            "rule_census rematerialize_alu_cost threshold=2 saved_dispatches={saved_dispatches_at_2} dispatch_saving_ns={dispatch_saving_ns} recompute_elements={recompute_elements_total} alu_cost_ns_fast_path={alu_cost_fast_ns} alu_cost_ns_slow_path={alu_cost_slow_ns}"
        );
        assert!(
            recompute_elements_total > 0,
            "rule census found rematerialization candidates but zero recompute-element cost -- \
             the shape lookup is wired to nothing"
        );

        // deliverable #5: verify the 547-482=65 gap is exactly the
        // elementwise-kind nodes in `effective_outputs` (never read as an
        // operand by anything else in `push`, so no decline event exists
        // for them, yet they still materialize as named outputs).
        let materialized_elementwise_nodes: alloc::collections::BTreeSet<u32> = bound
            .iter()
            .filter(|op| matches!(op.kind, crate::bind::BoundOpKind::Elementwise { .. }))
            .map(|op| op.node.0)
            .collect();
        // 419 = 547 - 128: the same 128 reduce-epilogue-fusion absorptions
        // asserted above remove one `BoundOpKind::Elementwise` per fusion --
        // the consumer that used to materialize on its own now IS the
        // epilogued `Reduce`, so it drops out of this `Elementwise`-kind
        // filter entirely.
        #[cfg(not(feature = "reduce-epilogue-fusion"))]
        assert_eq!(
            materialized_elementwise_nodes.len(),
            547,
            "the materialized-elementwise set must have exactly 547 members, matching the bound count"
        );
        #[cfg(feature = "reduce-epilogue-fusion")]
        assert_eq!(
            materialized_elementwise_nodes.len(),
            547 - 4 * 32,
            "419 = 547 unfused elementwise BoundOps minus the 128 reduce-epilogue-fusion \
             absorptions (4/layer x 32 layers) -- see epilogued_reduce_count's own doc above"
        );
        let unexplained_nodes: alloc::vec::Vec<u32> = materialized_elementwise_nodes
            .difference(&elementwise_declined_nodes)
            .copied()
            .collect();
        let output_node_set: alloc::collections::BTreeSet<u32> =
            outputs.iter().map(|node| node.0).collect();
        let unexplained_that_are_outputs = unexplained_nodes
            .iter()
            .filter(|node| output_node_set.contains(node))
            .count();
        let unexplained_that_are_not_outputs: alloc::vec::Vec<u32> = unexplained_nodes
            .iter()
            .filter(|node| !output_node_set.contains(node))
            .copied()
            .collect();
        std::println!(
            "rule_census unexplained_gap total={} outputs={unexplained_that_are_outputs} not_outputs={}",
            unexplained_nodes.len(),
            unexplained_that_are_not_outputs.len()
        );
        for node in unexplained_that_are_not_outputs.iter().take(5) {
            let op = &program[*node as usize];
            std::println!("rule_census unexplained_gap_node node={node} op={op:?}");
        }

        // N == 0 is a red gate, not a quiet pass: every rule this census
        // names either fired or declined at least once on a real 32-layer
        // forward, or the census measured nothing and must fail loudly.
        assert!(total > 0, "rule census processed zero bound ops");
        assert!(
            fuse_elementwise_fired + fuse_reduce_fired > 0,
            "rule census recorded zero fusion firings -- the counters are wired to nothing"
        );
        assert!(
            window_reduce_fired + window_reduce_declined > 0,
            "rule census recorded zero window-reduce attempts -- the counter is wired to nothing"
        );

        // `total` and `elementwise` both shift by exactly the 128
        // reduce-epilogue-fusion absorptions under that feature; every
        // fusion removes one whole `BoundOp` (the standalone `Reduce`
        // disappears, its consumer's `Elementwise` slot is repurposed as the
        // SAME `Reduce`'s epilogue rather than adding a new entry) --
        // `reduce_total`/`constant`/`iota` are untouched because the fused
        // reduce keeps its `BoundOpKind::Reduce` kind, just gains a
        // non-default `epilogue_body`.
        #[cfg(not(feature = "reduce-epilogue-fusion"))]
        assert_eq!(total, 1196, "total BoundOps must match the measured forward");
        #[cfg(feature = "reduce-epilogue-fusion")]
        assert_eq!(
            total,
            1196 - 4 * 32,
            "1068 = 1196 unfused total minus the 128 reduce-epilogue-fusion absorptions"
        );
        assert_eq!(
            reduce_total,
            225 + 385,
            "total reduces must match the measured reduce_matmul_quantized + \
             reduce_f32_dense population, even though this test's own two-operand \
             split is a different partition of that same 610 (see this test's own doc); \
             unaffected by reduce-epilogue-fusion -- an absorbed reduce keeps its \
             BoundOpKind::Reduce kind, it only gains a non-default epilogue"
        );
        #[cfg(not(feature = "reduce-epilogue-fusion"))]
        assert_eq!(elementwise, 547, "elementwise BoundOps must match the measured forward");
        #[cfg(feature = "reduce-epilogue-fusion")]
        assert_eq!(
            elementwise,
            547 - 4 * 32,
            "419 = 547 unfused elementwise BoundOps minus the 128 reduce-epilogue-fusion \
             absorptions (4/layer x 32 layers) -- see epilogued_reduce_count's own doc above"
        );
        assert_eq!(constant, 37, "constant BoundOps must match the measured forward");
        assert_eq!(iota, 2, "iota BoundOps must match the measured forward");
        assert_eq!(
            reduce_total + elementwise + constant + iota,
            total,
            "the four BoundOpKind buckets must exhaust the total with no remainder"
        );

        // fused count, measured directly rather than assumed: re-bind the
        // SAME program WITH `cached_attention_candidates` fusion turned on
        // (`bind_with_fusion(.., true)`, `bind.rs:2338`) so this census also
        // states what the fused split looks like under
        // `cached-attention-streaming`, instead of only proving the unfused
        // split held.
        #[cfg(feature = "cached-attention-streaming")]
        {
            let fused = crate::bind::bind_with_fusion(
                &program,
                &shapes,
                &outputs,
                true,
                crate::numeric::NumericPolicy::default(),
            )
            .expect("the program binds with fusion enabled");
            let fused_cached_attention = fused
                .iter()
                .filter(|op| matches!(op.kind, crate::bind::BoundOpKind::CachedAttention { .. }))
                .count();
            std::println!(
                "rule_census fused_total={} fused_cached_attention={fused_cached_attention}",
                fused.len()
            );
            assert_eq!(
                fused_cached_attention, 32,
                "one CachedAttention BoundOp fusion per layer on this 32-layer forward"
            );
            // MEASURED (`rule_census fused_total=620 fused_cached_attention=32`):
            // 1196 unfused - 620 fused = 576 BoundOps absorbed into the 32
            // CachedAttention fusions, 18 per fusion.
            assert_eq!(
                fused.len(),
                620,
                "fused total must be 620 on this program -- 1196 unfused minus 576 BoundOps \
                 absorbed across the 32 CachedAttention fusions (18 each); re-measure via the \
                 `rule_census fused_total=` println above if the fusion rewrite's own \
                 absorption count changes"
            );
        }
    }

    /// `paired_gate_up_reduce`'s own census, in the SAME relation form as
    /// [`the_rule_census_reconciles_against_the_measured_mistral_forward_split`]
    /// above -- deltas against the baseline program's own measured counts,
    /// never a new absolute literal. Building the `[2, feed_forward,
    /// embedding]`-leaf program and binding it exactly as the baseline is
    /// bound (`bind_with_fusion(.., false)`, fusion held off) isolates one
    /// thing: the paired reduce collapses `gate`'s and `up`'s two
    /// `Op::Reduce`s into one, so `reduce_total` must drop by exactly one
    /// per layer (32 layers, 32-layer program) relative to the baseline
    /// this same test computes fresh -- never re-typed from the other
    /// test's own docstring, which could drift.
    #[test]
    fn paired_gate_up_reduce_removes_one_reduce_per_layer_relative_to_the_baseline() {
        let (baseline_program, baseline_logits, baseline_roots) =
            mistral_cached_forward_program(32_002, 4096, 14336, 32, 8, 128, 32)
                .expect("the baseline cached forward pass lowers to a program");
        let mut baseline_outputs = alloc::vec![baseline_logits];
        for (even, odd, value) in &baseline_roots {
            baseline_outputs.extend_from_slice(&[*even, *odd, *value]);
        }
        let baseline_shapes = crate::shape::infer(&baseline_program, &[1, 71])
            .expect("baseline: one new position against a 71-position cache infers");
        let baseline_bound = crate::bind::bind_with_fusion(
            &baseline_program,
            &baseline_shapes,
            &baseline_outputs,
            false,
            crate::numeric::NumericPolicy::default(),
        )
        .expect("the baseline program binds");
        let baseline_reduce_total = baseline_bound
            .iter()
            .filter(|op| matches!(&op.kind, crate::bind::BoundOpKind::Reduce { .. }))
            .count();

        let (paired_program, paired_roots_bundle, paired_roots, _paired_moe_sites) =
            mistral_cached_forward_program_with_experts(
                32_002, 4096, 14336, 32, 8, 128, 32, 0, 0, false, true, false,
            )
            .expect("the paired cached forward pass lowers to a program");
        let paired_logits = paired_roots_bundle.logits;
        let mut paired_outputs = alloc::vec![paired_logits];
        for (even, odd, value) in &paired_roots {
            paired_outputs.extend_from_slice(&[*even, *odd, *value]);
        }
        let paired_shapes = crate::shape::infer(&paired_program, &[1, 71])
            .expect("paired: one new position against a 71-position cache infers");
        let paired_bound =
            crate::bind::bind_with_fusion(
                &paired_program,
                &paired_shapes,
                &paired_outputs,
                false,
                crate::numeric::NumericPolicy::default(),
            )
                .expect("the paired program binds");
        let paired_reduce_total = paired_bound
            .iter()
            .filter(|op| matches!(&op.kind, crate::bind::BoundOpKind::Reduce { .. }))
            .count();

        std::println!(
            "paired_gate_up_reduce_census baseline_reduce_total={baseline_reduce_total} paired_reduce_total={paired_reduce_total} baseline_bound_total={} paired_bound_total={}",
            baseline_bound.len(),
            paired_bound.len()
        );
        assert_eq!(
            baseline_reduce_total - paired_reduce_total,
            32,
            "paired_gate_up_reduce must remove exactly one Op::Reduce per layer (32 layers) \
             relative to the baseline program's own measured reduce total -- gate's and up's \
             two independent reduces collapsing into one paired reduce"
        );

        #[cfg(feature = "reduce-epilogue-fusion")]
        {
            let baseline_epilogued = baseline_bound
                .iter()
                .filter(|op| {
                    matches!(
                        &op.kind,
                        crate::bind::BoundOpKind::Reduce { epilogue_operands, .. }
                            if !epilogue_operands.is_empty()
                    )
                })
                .count();
            let paired_epilogued = paired_bound
                .iter()
                .filter(|op| {
                    matches!(
                        &op.kind,
                        crate::bind::BoundOpKind::Reduce { epilogue_operands, .. }
                            if !epilogue_operands.is_empty()
                    )
                })
                .count();
            std::println!(
                "paired_gate_up_reduce_census baseline_epilogued={baseline_epilogued} paired_epilogued={paired_epilogued}"
            );
            // `ffn_hidden` (`spec.rs`'s own `append_mistral_cached_layer`)
            // never absorbs the paired reduce's `up` slice: `up` is read
            // through a non-identity, base-shifted axis expression (the
            // parity-selecting `"s,0*s+1,g->sg"` map) from the SAME node the
            // `gate` chain also reads through a DIFFERENT map
            // (`"s,0*s+0,g->sg"`) — `find_epilogue_source`'s own decline for
            // two DIFFERENT projections of the same source. That site is
            // still MEASURED at zero fusion in EITHER program.
            //
            // The 32-op (one-per-layer) delta below is NEW, and it is
            // correct, not a leak: `find_epilogue_source`/
            // `resolved_reference_counts` used to also reject a consumer's
            // own REPEATED read of the SAME source through the SAME
            // projection (production SiLU reads `gate` once bare, once
            // inside `exp(-gate)`), so the `silu(gate) * up` tail never fused
            // onto `gate`'s own reduce in EITHER program. Fixing that bug
            // lets the baseline program's `gate`/`up` -- two INDEPENDENT
            // `Op::Reduce`s -- absorb that tail once per layer (`+32`
            // epilogued reduces). The paired program cannot gain the same
            // fusion: its `gate`/`up` share ONE `Op::Reduce`, read through
            // the two DIFFERENT parity-split projections above, so the SAME
            // consumer names that one reduce through two different
            // projections and `find_epilogue_source` correctly declines it,
            // exactly as the un-widened site always did. The 32-op delta is
            // therefore precisely paired_gate_up_reduce's own structural
            // cost: sharing one reduce forecloses an epilogue fusion the
            // unpaired baseline can still take.
            assert_eq!(
                baseline_epilogued - paired_epilogued,
                32,
                "paired_gate_up_reduce's shared reduce must foreclose exactly one SiLU-tail \
                 epilogue fusion per layer (32 layers) relative to the baseline's two \
                 independent reduces; a different delta means some OTHER site started \
                 admitting or rejecting asymmetrically between the two programs"
            );
        }
    }

    /// `fused_qkv_reduce`'s own census, same relation-form discipline as
    /// [`paired_gate_up_reduce_removes_one_reduce_per_layer_relative_to_the_baseline`]
    /// above -- deltas against a freshly computed baseline, never a
    /// re-typed literal. Q/K/V's three independent reduces collapse into
    /// one fused reduce (`-2` `Op::Reduce`/layer, `-64` total), but q's, k's,
    /// AND v's rows are three DIFFERENT sizes under GQA (unlike
    /// `paired_gate_up_reduce`'s identical-size gate/up), so none of the
    /// three can be read back out of the shared flat buffer at zero extra
    /// cost the way `paired_gate_up_reduce` reads its parity axis -- the IR
    /// cannot split one real axis into two unconstrained virtual sub-axes
    /// from a single operand (`append_mistral_cached_layer`'s
    /// `fused_qkv_reduce` doc traces the exact `shape::infer`
    /// `UnconstrainedDim` this hits and why `ScalarOp::arity` blocks the
    /// obvious fix of adding a shape-only operand to an existing binary
    /// op). All three of q/k/v need their own small
    /// `ScalarOp::Multiply`-by-shape-constant extract instead. Net per
    /// layer: 3 reduces removed, 1 fused reduce added, 3 extracts added --
    /// this test asserts the CORRECTED count, `+1` dispatch/layer (`+32`
    /// total), not the `-2`/layer a bandwidth-only reading of ROW 336 would
    /// predict.
    #[test]
    fn fused_qkv_reduce_adds_one_dispatch_per_layer_relative_to_the_baseline() {
        let (baseline_program, baseline_logits, baseline_roots) =
            mistral_cached_forward_program(32_002, 4096, 14336, 32, 8, 128, 32)
                .expect("the baseline cached forward pass lowers to a program");
        let mut baseline_outputs = alloc::vec![baseline_logits];
        for (even, odd, value) in &baseline_roots {
            baseline_outputs.extend_from_slice(&[*even, *odd, *value]);
        }
        let baseline_shapes = crate::shape::infer(&baseline_program, &[1, 71])
            .expect("baseline: one new position against a 71-position cache infers");
        let baseline_bound = crate::bind::bind_with_fusion(
            &baseline_program,
            &baseline_shapes,
            &baseline_outputs,
            false,
            crate::numeric::NumericPolicy::default(),
        )
        .expect("the baseline program binds");
        let baseline_total = baseline_bound.len();
        let baseline_reduce_total = baseline_bound
            .iter()
            .filter(|op| matches!(&op.kind, crate::bind::BoundOpKind::Reduce { .. }))
            .count();

        let (fused_program, fused_roots_bundle, fused_roots, _fused_moe_sites) =
            mistral_cached_forward_program_with_experts(
                32_002, 4096, 14336, 32, 8, 128, 32, 0, 0, false, false, true,
            )
            .expect("the fused-qkv cached forward pass lowers to a program");
        let fused_logits = fused_roots_bundle.logits;
        let mut fused_outputs = alloc::vec![fused_logits];
        for (even, odd, value) in &fused_roots {
            fused_outputs.extend_from_slice(&[*even, *odd, *value]);
        }
        let fused_shapes = crate::shape::infer(&fused_program, &[1, 71])
            .expect("fused: one new position against a 71-position cache infers");
        let fused_bound =
            crate::bind::bind_with_fusion(
                &fused_program,
                &fused_shapes,
                &fused_outputs,
                false,
                crate::numeric::NumericPolicy::default(),
            )
                .expect("the fused-qkv program binds");
        let fused_total = fused_bound.len();
        let fused_reduce_total = fused_bound
            .iter()
            .filter(|op| matches!(&op.kind, crate::bind::BoundOpKind::Reduce { .. }))
            .count();

        std::println!(
            "fused_qkv_reduce_census baseline_reduce_total={baseline_reduce_total} fused_reduce_total={fused_reduce_total} baseline_bound_total={baseline_total} fused_bound_total={fused_total}"
        );
        assert_eq!(
            baseline_reduce_total - fused_reduce_total,
            64,
            "fused_qkv_reduce must remove exactly two Op::Reduce per layer (32 layers) relative \
             to the baseline -- q's, k's, and v's three independent reduces collapsing into one \
             fused reduce"
        );
        assert_eq!(
            fused_total - baseline_total,
            34,
            "fused_qkv_reduce must ADD exactly one dispatch per layer (32 layers -> +32) plus the \
             two one-time Op::Constant shape hints built once for the whole program (+2), 34 \
             total -- 3 reduces removed, 1 fused reduce added, 3 shape-constant extracts added \
             (q/k/v each need their own, unlike paired_gate_up_reduce's zero-extra-cost parity \
             read) nets +1/layer, not the -2/layer a bandwidth-only reading of the reduce count \
             would predict"
        );
    }

    /// Proof the new test can fail: perturbing one tap weight must move the
    /// affected output positions away from the hand-computed reference.
    #[proxima::test]
    async fn causal_conv1d_hand_computed_check_actually_detects_a_wrong_weight() {
        let mut program = Vec::new();
        let x = op::append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Symbolic(0), Extent::Static(1)],
                name: Some("x".into()),
            },
        );
        let weight = op::append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                // see the previous test's own doc on why `[1, 3]`, not `[3, 1]`.
                shape: alloc::vec![Extent::Static(1), Extent::Static(3)],
                name: Some("weight".into()),
            },
        );
        let output = causal_conv1d(&mut program, x, weight, 3).expect("causal conv lowers");

        let x_data = [1.0f32, 2.0, 3.0, 4.0];
        // tap l=2 perturbed from 100 to 99: every output except out[0] still
        // matches (out[0]'s only real contribution is tap l=2, so this alone
        // would move it too -- included to show the test is not vacuous).
        let perturbed_weight_data = [1.0f32, 10.0, 99.0];
        let evaluated = crate::cpu::evaluate_named(
            &program,
            &[4],
            &[("x", &x_data), ("weight", &perturbed_weight_data)],
            &[output],
        )
        .expect("causal conv evaluates");
        let (result, _shape) = evaluated.get(output).expect("conv output present");

        std::println!("perturbed causal_conv1d result={result:?}");
        assert_ne!(
            result,
            [100.0, 210.0, 321.0, 432.0],
            "a perturbed tap weight must move the output away from the hand-computed reference \
             (if this assertion cannot fail, the test above proves nothing)"
        );
    }

    /// `causal_conv1d_matches_a_hand_computed_causal_window`'s own single
    /// channel (`embedding=1`) cannot distinguish `weight`'s `[l_cache,
    /// embedding]` axis order from `[embedding, l_cache]` -- with one
    /// channel, transposing does not move a single byte. This is exactly the
    /// gap that let the real checkpoint's own `blk.{layer}.shortconv.conv.weight`
    /// (GGUF on-disk `[l_cache=3, embedding=2048]`, `l_cache` the FASTEST
    /// axis) get bound with its axes swapped for months: two DIFFERENT
    /// per-channel weight patterns, so a transposed read produces
    /// hand-verifiably wrong numbers instead of silently-correct ones.
    /// Channel 0 reuses the single-channel test's own `weight = [1, 10,
    /// 100]`/`x = [1, 2, 3, 4]` (`out = [100, 210, 321, 432]`, worked out
    /// there); channel 1 uses `weight = [1000, 1, 1]`/`x = [2, 2, 2, 2]`:
    /// - `out[0] = 1*x[0]                       = 1*2                = 2`
    /// - `out[1] = 1*x[0]  + 1*x[1]             = 1*2 + 1*2          = 4`
    /// - `out[2] = 1000*x[0] + 1*x[1] + 1*x[2]  = 1000*2 + 2 + 2     = 2004`
    /// - `out[3] = 1000*x[1] + 1*x[2] + 1*x[3]  = 1000*2 + 2 + 2     = 2004`
    #[proxima::test]
    async fn causal_conv1d_keeps_channels_independent_and_catches_a_transposed_weight() {
        let mut program = Vec::new();
        let x = op::append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Symbolic(0), Extent::Static(2)],
                name: Some("x".into()),
            },
        );
        let weight = op::append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(2), Extent::Static(3)],
                name: Some("weight".into()),
            },
        );
        let output = causal_conv1d(&mut program, x, weight, 3).expect("causal conv lowers");

        // sequence-major, channel-fastest: `[s0_ch0, s0_ch1, s1_ch0, s1_ch1, ...]`.
        let x_data = [1.0f32, 2.0, 2.0, 2.0, 3.0, 2.0, 4.0, 2.0];
        // channel-major, tap-fastest: `[ch0_l0, ch0_l1, ch0_l2, ch1_l0, ch1_l1, ch1_l2]`
        // -- the real checkpoint's own on-disk axis order.
        let weight_data = [1.0f32, 10.0, 100.0, 1000.0, 1.0, 1.0];
        let evaluated = crate::cpu::evaluate_named(
            &program,
            &[4],
            &[("x", &x_data), ("weight", &weight_data)],
            &[output],
        )
        .expect("causal conv evaluates");
        let (result, shape) = evaluated.get(output).expect("conv output present");

        std::println!("multi-channel causal_conv1d result={result:?} shape={shape:?}");
        assert_eq!(shape, [4u64, 2u64]);
        assert_eq!(
            result,
            [100.0, 2.0, 210.0, 4.0, 321.0, 2004.0, 432.0, 2004.0]
        );
    }

    /// [`rmsnorm_per_head`] against a hand-computed RMS norm -- one token,
    /// two heads, `head_dim = 2`: head 0's raw values `[3, 4]` have
    /// `mean_square = (9+16)/2 = 12.5`, `rms = sqrt(12.5) ≈ 3.535534`,
    /// `inv_rms ≈ 0.282843`; scaled by `gamma = [2.0, 0.5]` that is
    /// `[3*0.282843*2, 4*0.282843*0.5] ≈ [1.697056, 0.565685]`. Head 1's
    /// `[1, 1]` have `mean_square = 1`, `rms = 1`, so `gamma` passes
    /// through unchanged: `[2.0, 0.5]`. Two heads with DIFFERENT norms in
    /// the same call proves the reduce is scoped per-head, not pooled
    /// across both (a pooled reduce would give both heads the same
    /// `inv_rms`, which is not `[1.697056, ...]` next to `[2.0, ...]`).
    #[proxima::test]
    async fn rmsnorm_per_head_matches_a_hand_computed_rms_norm() {
        let mut program = Vec::new();
        let x = op::append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Symbolic(0), Extent::Static(2), Extent::Static(2)],
                name: Some("x".into()),
            },
        );
        let gamma = op::append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(2)],
                name: Some("gamma".into()),
            },
        );
        let eps = op::append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Symbolic(0)],
                name: Some("eps".into()),
            },
        );
        let inv_head_dim = scalar_constant(&mut program, 0.5);
        let output = rmsnorm_per_head(&mut program, x, gamma, inv_head_dim, eps, "h")
            .expect("per-head rmsnorm lowers");

        let x_data = [3.0f32, 4.0, 1.0, 1.0];
        let gamma_data = [2.0f32, 0.5];
        let eps_data = [0.0f32];
        let evaluated = crate::cpu::evaluate_named(
            &program,
            &[1],
            &[("x", &x_data), ("gamma", &gamma_data), ("eps", &eps_data)],
            &[output],
        )
        .expect("per-head rmsnorm evaluates");
        let (result, shape) = evaluated.get(output).expect("rmsnorm output present");

        std::println!("rmsnorm_per_head result={result:?} shape={shape:?}");
        assert_eq!(shape, [1u64, 2u64, 2u64]);
        let expected = [1.6970563f32, 0.56568545, 2.0, 0.5];
        for (found, wanted) in result.iter().zip(&expected) {
            assert!(
                (found - wanted).abs() < 1e-5,
                "got {result:?}, expected {expected:?}"
            );
        }
    }

    /// Proof the new test can fail: perturbing `gamma` must move the
    /// output away from the hand-computed reference.
    #[proxima::test]
    async fn rmsnorm_per_head_hand_computed_check_actually_detects_a_wrong_gamma() {
        let mut program = Vec::new();
        let x = op::append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Symbolic(0), Extent::Static(2), Extent::Static(2)],
                name: Some("x".into()),
            },
        );
        let gamma = op::append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(2)],
                name: Some("gamma".into()),
            },
        );
        let eps = op::append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Symbolic(0)],
                name: Some("eps".into()),
            },
        );
        let inv_head_dim = scalar_constant(&mut program, 0.5);
        let output = rmsnorm_per_head(&mut program, x, gamma, inv_head_dim, eps, "h")
            .expect("per-head rmsnorm lowers");

        let x_data = [3.0f32, 4.0, 1.0, 1.0];
        // gamma[0] perturbed from 2.0 to 3.0.
        let perturbed_gamma_data = [3.0f32, 0.5];
        let eps_data = [0.0f32];
        let evaluated = crate::cpu::evaluate_named(
            &program,
            &[1],
            &[
                ("x", &x_data),
                ("gamma", &perturbed_gamma_data),
                ("eps", &eps_data),
            ],
            &[output],
        )
        .expect("per-head rmsnorm evaluates");
        let (result, _shape) = evaluated.get(output).expect("rmsnorm output present");

        std::println!("perturbed rmsnorm_per_head result={result:?}");
        let unperturbed = [1.6970563f32, 0.56568545, 2.0, 0.5];
        assert!(
            (result[0] - unperturbed[0]).abs() > 1e-3,
            "a perturbed gamma must move the output away from the hand-computed reference \
             (if this assertion cannot fail, the test above proves nothing)"
        );
    }

    #[proxima::test]
    #[case::attention_block(2, "blk.2.attn_q.weight", LayerKind::Attention)]
    #[case::conv_block(0, "blk.0.shortconv.conv.weight", LayerKind::ShortConv)]
    async fn layer_kind_derives_from_the_real_checkpoints_own_tensor_marker(
        #[case] layer: u32,
        #[case] marker: &str,
        #[case] expected: LayerKind,
    ) {
        let names = ["token_embd.weight", marker, "output_norm.weight"];
        let derived = LayerKind::from_tensor_names(names, layer)
            .expect("a real block names exactly one marker");
        assert_eq!(derived, expected);
    }

    #[proxima::test]
    async fn layer_kind_names_the_block_when_neither_marker_is_present() {
        let names = ["token_embd.weight", "output_norm.weight"];
        let error = LayerKind::from_tensor_names(names, 7)
            .expect_err("a block with no marker cannot derive a kind");
        assert!(
            matches!(error, TensorError::UndeterminedLayerKind { layer: 7 }),
            "got {error:?}"
        );
    }

    /// LFM2.5-8B-A1B's real dimensions (24 blocks: 2 leading dense, 22 MoE;
    /// 18 short-convolution layers at blocks `{0,1,3,4,5,7,8,9,11,12,13,15,
    /// 16,17,19,20,22,23}`, 6 attention layers at `{2,6,10,14,18,21}` -- the
    /// real checkpoint's own tensor directory, cross-checked against
    /// `lfm2moe.attention.head_count_kv`'s per-layer `[0,0,8,...]` sample in
    /// its metadata dump) -- proves the hybrid builder lowers and infers at
    /// this checkpoint's actual shapes without needing the 5 GB file itself,
    /// the same real-dimensions-without-real-weights convention
    /// `the_whole_mistral_forward_pass_infers_at_real_dimensions` already
    /// uses above.
    #[proxima::test]
    async fn the_whole_lfm2_forward_pass_infers_at_real_dimensions() {
        const REAL_CONTEXT: u64 = 8192;
        const ATTENTION_LAYERS: [u32; 6] = [2, 6, 10, 14, 18, 21];

        let layer_kinds: Vec<LayerKind> = (0..24)
            .map(|layer| {
                if ATTENTION_LAYERS.contains(&layer) {
                    LayerKind::Attention
                } else {
                    LayerKind::ShortConv
                }
            })
            .collect();

        let build_start = std::time::Instant::now();
        let (program, _logits, _moe_sites) = lfm2_forward_program_with_experts(
            128_000,
            2048,
            7168,
            1792,
            32,
            8,
            64,
            24,
            32,
            4,
            2,
            3,
            &layer_kinds,
        )
        .expect("the hybrid forward pass lowers to a program");
        let build_elapsed = build_start.elapsed();

        let infer_start = std::time::Instant::now();
        crate::shape::infer(&program, &[REAL_CONTEXT])
            .expect("the hybrid forward pass infers at its real context length");
        let infer_elapsed = infer_start.elapsed();

        std::println!(
            "lfm2_forward_program_with_experts: nodes={} build={build_elapsed:?} infer={infer_elapsed:?}",
            program.len()
        );
        assert!(
            program.len() > 1_000,
            "24 hybrid blocks plus embedding/lm-head should be well over a thousand nodes, not {}",
            program.len()
        );
    }

    #[proxima::test]
    async fn lfm2_forward_program_rejects_a_layer_kinds_length_mismatch() {
        let layer_kinds = [LayerKind::Attention, LayerKind::ShortConv];
        let error = lfm2_forward_program_with_experts(
            128_000,
            2048,
            7168,
            1792,
            32,
            8,
            64,
            24,
            32,
            4,
            2,
            3,
            &layer_kinds,
        )
        .expect_err("2 layer_kinds against block_count=24 must be rejected");
        assert!(
            matches!(
                error,
                TensorError::LayerKindCountMismatch {
                    expected: 24,
                    found: 2
                }
            ),
            "got {error:?}"
        );
    }

    #[proxima::test]
    async fn causal_conv1d_rejects_a_zero_width_window() {
        let mut program = Vec::new();
        let x = op::append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Symbolic(0), Extent::Static(1)],
                name: Some("x".into()),
            },
        );
        let weight = op::append(
            &mut program,
            Op::Input {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(0), Extent::Static(1)],
                name: Some("weight".into()),
            },
        );
        let error = causal_conv1d(&mut program, x, weight, 0)
            .expect_err("l_cache=0 has no window to convolve");
        assert!(
            matches!(error, TensorError::InvalidConvConfig { l_cache: 0 }),
            "got {error:?}"
        );
    }

    /// [`append_qwen35_delta_net_step`] against a hand-computed single-head,
    /// `key_dim = value_dim = 2` delta-rule step -- llama.cpp's own
    /// `build_delta_net_autoregressive` traced by hand: `decay = exp(0) =
    /// 1`, `state_decayed = state_in` (`[[1,2],[3,4]]`), `v_pred = [1,2]`
    /// (`state_decayed^T @ k` with `k = [1,0]`), `residual = v - v_pred =
    /// [4,4]` (`v = [5,6]`), `delta = residual * beta = [4,4]` (`beta = 1`),
    /// `state_out = state_decayed + k(outer)delta = [[5,6],[3,4]]`,
    /// `out = state_out^T @ q_scaled` with `q_scaled = q * 1 = [1,0]` gives
    /// `[5,6]`.
    #[proxima::test]
    async fn qwen35_delta_net_step_matches_a_hand_computed_recurrence() {
        let mut program = Vec::new();
        let shape_ih = alloc::vec![Extent::Static(2), Extent::Static(1)];
        let shape_jh = alloc::vec![Extent::Static(2), Extent::Static(1)];
        let shape_h = alloc::vec![Extent::Static(1)];
        let shape_ijh = alloc::vec![Extent::Static(2), Extent::Static(2), Extent::Static(1)];

        let query = input_leaf(&mut program, DType::Float32, shape_ih.clone(), "query");
        let key = input_leaf(&mut program, DType::Float32, shape_ih, "key");
        let value = input_leaf(&mut program, DType::Float32, shape_jh, "value");
        let gate = input_leaf(&mut program, DType::Float32, shape_h.clone(), "gate");
        let beta = input_leaf(&mut program, DType::Float32, shape_h, "beta");
        let state_in = input_leaf(&mut program, DType::Float32, shape_ijh, "state_in");
        let inv_sqrt_key_dim = scalar_constant(&mut program, 1.0);

        let (out, state_out) = append_qwen35_delta_net_step(
            &mut program,
            query,
            key,
            value,
            gate,
            beta,
            state_in,
            inv_sqrt_key_dim,
            "h",
        )
        .expect("delta net step lowers");

        let query_data = [1.0f32, 0.0];
        let key_data = [1.0f32, 0.0];
        let value_data = [5.0f32, 6.0];
        let gate_data = [0.0f32];
        let beta_data = [1.0f32];
        let state_in_data = [1.0f32, 2.0, 3.0, 4.0];

        let evaluated = crate::cpu::evaluate_named(
            &program,
            &[],
            &[
                ("query", &query_data),
                ("key", &key_data),
                ("value", &value_data),
                ("gate", &gate_data),
                ("beta", &beta_data),
                ("state_in", &state_in_data),
            ],
            &[out, state_out],
        )
        .expect("delta net step evaluates");

        let (out_values, _) = evaluated.get(out).expect("out present");
        let (state_out_values, _) = evaluated.get(state_out).expect("state_out present");

        assert_eq!(out_values, [5.0, 6.0], "read-out uses the UPDATED state");
        assert_eq!(
            state_out_values,
            [5.0, 6.0, 3.0, 4.0],
            "state_out = state_decayed + k(outer)delta"
        );
    }

    /// Proof the hand-computed test can fail: a nonzero `beta` on a
    /// perturbed run must move both `out` and `state_out` away from the
    /// `beta = 0` (no update at all) reference.
    #[proxima::test]
    async fn qwen35_delta_net_step_hand_computed_check_actually_detects_a_wrong_beta() {
        let mut program = Vec::new();
        let shape_ih = alloc::vec![Extent::Static(2), Extent::Static(1)];
        let shape_jh = alloc::vec![Extent::Static(2), Extent::Static(1)];
        let shape_h = alloc::vec![Extent::Static(1)];
        let shape_ijh = alloc::vec![Extent::Static(2), Extent::Static(2), Extent::Static(1)];

        let query = input_leaf(&mut program, DType::Float32, shape_ih.clone(), "query");
        let key = input_leaf(&mut program, DType::Float32, shape_ih, "key");
        let value = input_leaf(&mut program, DType::Float32, shape_jh, "value");
        let gate = input_leaf(&mut program, DType::Float32, shape_h.clone(), "gate");
        let beta = input_leaf(&mut program, DType::Float32, shape_h, "beta");
        let state_in = input_leaf(&mut program, DType::Float32, shape_ijh, "state_in");
        let inv_sqrt_key_dim = scalar_constant(&mut program, 1.0);

        let (out, state_out) = append_qwen35_delta_net_step(
            &mut program,
            query,
            key,
            value,
            gate,
            beta,
            state_in,
            inv_sqrt_key_dim,
            "h",
        )
        .expect("delta net step lowers");

        let query_data = [1.0f32, 0.0];
        let key_data = [1.0f32, 0.0];
        let value_data = [5.0f32, 6.0];
        let gate_data = [0.0f32];
        let beta_data = [0.0f32];
        let state_in_data = [1.0f32, 2.0, 3.0, 4.0];

        let evaluated = crate::cpu::evaluate_named(
            &program,
            &[],
            &[
                ("query", &query_data),
                ("key", &key_data),
                ("value", &value_data),
                ("gate", &gate_data),
                ("beta", &beta_data),
                ("state_in", &state_in_data),
            ],
            &[out, state_out],
        )
        .expect("delta net step evaluates");

        let (out_values, _) = evaluated.get(out).expect("out present");
        let (state_out_values, _) = evaluated.get(state_out).expect("state_out present");

        assert_ne!(
            out_values,
            [5.0, 6.0],
            "beta=0 must move out away from the beta=1 reference"
        );
        assert_ne!(
            state_out_values,
            [5.0, 6.0, 3.0, 4.0],
            "beta=0 must move state_out away from the beta=1 reference"
        );
    }

    /// [`softplus`] against `log(1 + exp(x))` hand-computed at `x = 0` and
    /// `x = 1`: `softplus(0) = ln(2) ≈ 0.6931`, `softplus(1) = ln(1 + e) ≈
    /// 1.3133` -- llama.cpp's own `ggml_softplus` input to Qwen3.5's
    /// `alpha_softplus` (`qwen35.cpp:370`).
    #[proxima::test]
    async fn softplus_matches_log_one_plus_exp() {
        let mut program = Vec::new();
        let shape_h = alloc::vec![Extent::Static(2)];
        let x = input_leaf(&mut program, DType::Float32, shape_h, "x");
        let one = scalar_constant(&mut program, 1.0);

        let out = softplus(&mut program, x, one, "h->h").expect("softplus lowers");

        let x_data = [0.0f32, 1.0];
        let evaluated = crate::cpu::evaluate_named(&program, &[], &[("x", &x_data)], &[out])
            .expect("softplus evaluates");
        let (out_values, _) = evaluated.get(out).expect("out present");

        assert!(
            (out_values[0] - core::f32::consts::LN_2).abs() < 1e-4,
            "softplus(0) = ln(2), got {}",
            out_values[0]
        );
        assert!(
            (out_values[1] - 1.313_262).abs() < 1e-4,
            "softplus(1) = ln(1+e), got {}",
            out_values[1]
        );
    }

    /// [`l2norm`] against a hand-computed `[3, 4]` vector: `norm = sqrt(9 +
    /// 16) = 5`, so the normalized output is `[0.6, 0.8]` -- `ggml_l2_norm`'s
    /// own contract (`qwen35.cpp:428-429`), no learnable weight and no
    /// mean-divide, unlike [`rmsnorm`].
    #[proxima::test]
    async fn l2norm_matches_a_hand_computed_unit_vector() {
        let mut program = Vec::new();
        let shape_d = alloc::vec![Extent::Static(1), Extent::Static(2)];
        let x = input_leaf(&mut program, DType::Float32, shape_d, "x");
        let eps = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(1)], "eps");

        let out = l2norm(&mut program, x, eps, "sd->sd", "s->sd").expect("l2norm lowers");

        let x_data = [3.0f32, 4.0];
        let eps_data = [0.0f32];
        let evaluated = crate::cpu::evaluate_named(
            &program,
            &[],
            &[("x", &x_data), ("eps", &eps_data)],
            &[out],
        )
        .expect("l2norm evaluates");
        let (out_values, _) = evaluated.get(out).expect("out present");

        assert!((out_values[0] - 0.6).abs() < 1e-5, "got {}", out_values[0]);
        assert!((out_values[1] - 0.8).abs() < 1e-5, "got {}", out_values[1]);
    }

    /// [`append_qwen35_conv_branch`] against a hand-computed depthwise causal
    /// conv (kernel 4, three channels `q|k|v` at `key_dim = value_dim = 1`,
    /// two steps): `causal_conv1d`'s own doc gives `out[s,d] = sum_l
    /// weight[d,l] * x[s+l-3, d]` (zero where the index is negative), so with
    /// only taps `l=2,3` ever landing in range for a 2-step sequence:
    /// `out[0,d] = weight[d,3]*x[0,d]`, `out[1,d] = weight[d,2]*x[0,d] +
    /// weight[d,3]*x[1,d]`. With `x[0] = [1,2,3]`, `x[1] = [4,5,6]` and
    /// `weight[.,2..4] = [[0.5,1.0], [1.0,-1.0], [2.0,0.5]]` (q,k,v):
    /// `out[0] = [1.0, -2.0, 1.5]`, `out[1] = [4.5, -3.0, 9.0]`. `v_conv`
    /// (never normalized) is checked against `silu` of those exactly;
    /// `q_conv`/`k_conv` are l2-normalized at width 1, which degenerates to
    /// `sign(x)` (`x / sqrt(x^2 + 0) = x / |x|`) -- `q`'s raw values are both
    /// positive, `k`'s both negative.
    #[proxima::test]
    async fn append_qwen35_conv_branch_matches_a_hand_computed_conv_silu_split_and_norm() {
        let key_dim = 1u32;
        let value_dim = 1u32;
        let l_cache = 4u32;
        let qkv_dim = 2 * key_dim + value_dim;

        let mut program = Vec::new();
        let qkv_mixed = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(0), Extent::Static(qkv_dim)],
            "qkv_mixed",
        );
        let conv_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(qkv_dim), Extent::Static(l_cache)],
            "conv_weight",
        );
        let eps = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Symbolic(0)], "eps");
        let one = scalar_constant(&mut program, 1.0);

        let (q_conv, k_conv, v_conv) = append_qwen35_conv_branch(
            &mut program,
            qkv_mixed,
            conv_weight,
            eps,
            one,
            key_dim,
            value_dim,
            l_cache,
        )
        .expect("conv branch lowers");

        // sequence-major, channel-fastest: `[s0_q, s0_k, s0_v, s1_q, s1_k, s1_v]`.
        let x_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        // channel-major, tap-fastest: taps 0,1 are unreachable at seq=2, so
        // only taps 2,3 (per channel) carry real weight.
        let weight_data = [
            0.0f32, 0.0, 0.5, 1.0, // q
            0.0, 0.0, 1.0, -1.0, // k
            0.0, 0.0, 2.0, 0.5, // v
        ];
        let eps_data = [0.0f32, 0.0];

        let evaluated = crate::cpu::evaluate_named(
            &program,
            &[2],
            &[
                ("qkv_mixed", &x_data),
                ("conv_weight", &weight_data),
                ("eps", &eps_data),
            ],
            &[q_conv, k_conv, v_conv],
        )
        .expect("conv branch evaluates");

        let (q_values, _) = evaluated.get(q_conv).expect("q_conv present");
        let (k_values, _) = evaluated.get(k_conv).expect("k_conv present");
        let (v_values, v_shape) = evaluated.get(v_conv).expect("v_conv present");

        assert_eq!(v_shape, [2u64, 1u64]);
        assert!((v_values[0] - 1.226_362).abs() < 1e-4, "got {}", v_values[0]);
        assert!((v_values[1] - 8.998_89).abs() < 1e-4, "got {}", v_values[1]);
        assert_eq!(
            q_values, [1.0, 1.0],
            "silu(q_raw) is positive at both steps, so l2norm at width 1 is +1"
        );
        assert_eq!(
            k_values, [-1.0, -1.0],
            "silu(k_raw) is negative at both steps, so l2norm at width 1 is -1"
        );
    }

    /// Proof the conv-branch reference above can fail: perturbing one `v`
    /// tap weight must move `v_conv` away from the hand-computed reference.
    #[proxima::test]
    async fn append_qwen35_conv_branch_hand_computed_check_actually_detects_a_wrong_weight() {
        let key_dim = 1u32;
        let value_dim = 1u32;
        let l_cache = 4u32;
        let qkv_dim = 2 * key_dim + value_dim;

        let mut program = Vec::new();
        let qkv_mixed = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(0), Extent::Static(qkv_dim)],
            "qkv_mixed",
        );
        let conv_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(qkv_dim), Extent::Static(l_cache)],
            "conv_weight",
        );
        let eps = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Symbolic(0)], "eps");
        let one = scalar_constant(&mut program, 1.0);

        let (_, _, v_conv) = append_qwen35_conv_branch(
            &mut program,
            qkv_mixed,
            conv_weight,
            eps,
            one,
            key_dim,
            value_dim,
            l_cache,
        )
        .expect("conv branch lowers");

        let x_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        // v's tap l=3 perturbed from 0.5 to 0.4 -- moves both v_conv positions,
        // since out[0] and out[1] both read tap l=3.
        let perturbed_weight_data = [
            0.0f32, 0.0, 0.5, 1.0, // q
            0.0, 0.0, 1.0, -1.0, // k
            0.0, 0.0, 2.0, 0.4, // v
        ];
        let eps_data = [0.0f32, 0.0];

        let evaluated = crate::cpu::evaluate_named(
            &program,
            &[2],
            &[
                ("qkv_mixed", &x_data),
                ("conv_weight", &perturbed_weight_data),
                ("eps", &eps_data),
            ],
            &[v_conv],
        )
        .expect("conv branch evaluates");
        let (v_values, _) = evaluated.get(v_conv).expect("v_conv present");

        assert!(
            (v_values[0] - 1.226_362).abs() > 1e-4 || (v_values[1] - 8.998_89).abs() > 1e-4,
            "a perturbed v tap weight must move v_conv away from the hand-computed reference \
             (if this assertion cannot fail, the test above proves nothing), got {v_values:?}"
        );
    }

    /// [`repeat_kv_heads`] against a hand-computed 2-kv-head, group-3 repeat
    /// (`num_v_heads = kv_heads * group = 6`, standing in for the real
    /// checkpoint's `16 * 3 = 48`): one token, `head_dim = 1`, kv head 0
    /// carries `10.0`, kv head 1 carries `20.0`. Every one of kv head 0's
    /// three query-head copies must read `10.0` and every one of kv head 1's
    /// three must read `20.0`, in `u`-major, `g`-minor order (`sugd`) --
    /// `[10,10,10,20,20,20]`, not interleaved (`[10,20,10,20,10,20]`, the
    /// shape a `u`/`g` axis swap would produce) and not collapsed onto one
    /// head (`[10,10,10,10,10,10]`, the shape a broadcast-only-`u`
    /// -- forgetting to size `g` from `group_ones` -- would produce).
    #[proxima::test]
    async fn repeat_kv_heads_maps_each_kv_head_to_its_own_three_query_heads() {
        let kv_heads = 2u32;
        let group = 3u32;

        let mut program = Vec::new();
        let x = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(0), Extent::Static(kv_heads), Extent::Static(1)],
            "x",
        );
        let repeated = repeat_kv_heads(&mut program, x, kv_heads, group).expect("repeat lowers");

        let x_data = [10.0f32, 20.0];
        let evaluated = crate::cpu::evaluate_named(&program, &[1], &[("x", &x_data)], &[repeated])
            .expect("repeat evaluates");
        let (values, shape) = evaluated.get(repeated).expect("repeated present");

        assert_eq!(shape, [1u64, 2u64, 3u64, 1u64]);
        assert_eq!(values, [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]);
    }

    /// Proof the repeat reference above can fail: swapping which kv head
    /// carries which value must move the repeated output away from the
    /// hand-computed reference.
    #[proxima::test]
    async fn repeat_kv_heads_hand_computed_check_actually_detects_a_swapped_head() {
        let kv_heads = 2u32;
        let group = 3u32;

        let mut program = Vec::new();
        let x = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(0), Extent::Static(kv_heads), Extent::Static(1)],
            "x",
        );
        let repeated = repeat_kv_heads(&mut program, x, kv_heads, group).expect("repeat lowers");

        // kv head 0 and kv head 1's values swapped relative to the reference.
        let swapped_x_data = [20.0f32, 10.0];
        let evaluated =
            crate::cpu::evaluate_named(&program, &[1], &[("x", &swapped_x_data)], &[repeated])
                .expect("repeat evaluates");
        let (values, _) = evaluated.get(repeated).expect("repeated present");

        assert_ne!(
            values,
            [10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
            "a swapped kv head must move the repeated output away from the hand-computed \
             reference (if this assertion cannot fail, the test above proves nothing)"
        );
    }

    /// Builds one [`append_qwen35_ssm_mixer`] decode step at `kv_heads = 1`,
    /// `group = 2` (the `u,g` seam, degenerate on `u` but real on `g`),
    /// `key_dim = value_dim_per_head = 1`, `l_cache = 2` -- every weight
    /// chosen to make the pipeline hand-traceable: `wqkv = 0` and the conv's
    /// new-token tap contributes nothing, so the whole conv branch reads
    /// straight off `conv_history_v0` (the one value this function
    /// parameterizes, for the mutation companion below); `attn_norm_weight
    /// = 1`, `x = 1`, `eps = 0` make `rmsnorm(x) = 1` exactly, so every
    /// downstream projection equals its own raw weight row; `ssm_beta =
    /// ssm_alpha = ssm_dt_bias = ssm_a = 0` collapse `beta` to `sigmoid(0) =
    /// 0.5` and `gate` (hence `decay`) to `0`/`1`, reusing
    /// [`append_qwen35_delta_net_step`]'s own already-proven `decay = 1`
    /// path; `head_v_dim = 1` degenerates the gated RMSNorm's own
    /// mean-square to `delta_out^2`, so `normed_out = sign(delta_out)`
    /// exactly, the same width-1-l2norm-is-sign identity
    /// [`append_qwen35_conv_branch`]'s own test already exploits.
    fn build_ssm_mixer_test_program(output_gate: GdnOutputGate) -> (Vec<Op>, NodeId, NodeId, NodeId) {
        let mut program = Vec::new();
        let key_dim = 1u32;
        let value_dim = 2u32;
        let kv_heads = 1u32;
        let group = 2u32;
        let l_cache = 2u32;
        let qkv_dim = 2 * key_dim + value_dim;

        let x = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Symbolic(0), Extent::Static(1)], "x");
        let inv_dim = scalar_constant(&mut program, 1.0);
        let eps = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Symbolic(0)], "eps");
        let head_eps = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
            "head_eps",
        );
        let one = scalar_constant(&mut program, 1.0);
        let inv_sqrt_key_dim = scalar_constant(&mut program, 1.0);
        let inv_head_v_dim = scalar_constant(&mut program, 1.0);
        let attn_norm_weight = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(1)], "attn_norm_weight");
        let wqkv = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(1), Extent::Static(qkv_dim)],
            "wqkv",
        );
        let wqkv_gate = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(1), Extent::Static(value_dim)],
            "wqkv_gate",
        );
        let conv_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(qkv_dim), Extent::Static(l_cache)],
            "conv_weight",
        );
        let conv_history_in = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(l_cache - 1), Extent::Static(qkv_dim)],
            "conv_history_in",
        );
        let ssm_beta = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(1), Extent::Static(kv_heads * group)],
            "ssm_beta",
        );
        let ssm_alpha = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(1), Extent::Static(kv_heads * group)],
            "ssm_alpha",
        );
        let ssm_dt_bias = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(kv_heads * group)],
            "ssm_dt_bias",
        );
        let ssm_a = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(kv_heads * group)],
            "ssm_a",
        );
        let ssm_norm_weight = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(1)], "ssm_norm_weight");
        let ssm_out = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(value_dim), Extent::Static(1)],
            "ssm_out",
        );
        let state_in = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(1),
                Extent::Static(1),
                Extent::Static(kv_heads),
                Extent::Static(group)
            ],
            "state_in",
        );

        let (mixer_out, qkv_mixed, state_out) = append_qwen35_ssm_mixer(
            &mut program,
            x,
            inv_dim,
            eps,
            head_eps,
            one,
            inv_sqrt_key_dim,
            inv_head_v_dim,
            Some(attn_norm_weight),
            wqkv,
            wqkv_gate,
            conv_weight,
            conv_history_in,
            ssm_beta,
            ssm_alpha,
            ssm_dt_bias,
            ssm_a,
            ssm_norm_weight,
            ssm_out,
            state_in,
            key_dim,
            value_dim,
            kv_heads,
            group,
            l_cache,
            output_gate,
        )
        .expect("ssm mixer lowers");

        (program, mixer_out, qkv_mixed, state_out)
    }

    /// Builds one call into [`append_qwen35_dense_attention_layer`] at the
    /// smallest non-degenerate dims that still separate all three fixed
    /// defects: `embedding = 1` (so every matmul is a scalar identity,
    /// `wq`/`wk`/`wv`/`w_gate_q`/`wo` ARE the per-dim activation), `rotary_dim
    /// = 2` (`pairs = 1`, exercises the split-half pairing) with `attn_head_dim
    /// = 4` (`pass_dim = 2`, exercises the concatenated-by-sum remainder),
    /// `kv_heads = query_heads = 1` (`group = 1`, no GQA broadcast to track
    /// by hand), `cached_len = 0` (only the "new" self-attention block).
    ///
    /// Hand computation: `x = [2.0]`, `attn_norm_weight = [1.0]`, `eps = 0`
    /// gives `normed = 2 / sqrt(4) = 1`, so every `w*` weight IS its own raw
    /// projection. `wq = wk = wv = [1,1,1,1]`, `q_norm_weight =
    /// k_norm_weight = [1,1,1,1]` -- `rmsnorm_per_head` of `[1,1,1,1]` is a
    /// no-op (`mean_square = 1`), keeping `q = k = [1,1,1,1]` exactly.
    /// `cos = [0]`, `sin = [1]` (`theta = pi/2` spelled as literals, no
    /// transcendental arithmetic needed): split-half RoPE gives
    /// `rotated_first = q[0]*0 - q[1]*1 = -1`, `rotated_second = q[1]*0 +
    /// q[0]*1 = 1` for both Q and K -- the ROTATED prefix is `[-1, 1]`, the
    /// PASS remainder stays `[1, 1]` (defect 2's own fix: dropped in the old
    /// code, present here as `score_..._pass`'s own nonzero contribution).
    /// `score = (-1)(-1) + (1)(1) + (1)(1) + (1)(1) = 4`, scaled by
    /// `inv_sqrt_attn_head_dim = 0.5` gives `2.0`; one key only (self), so
    /// softmax weight is `1.0` and `attended = v = [1,1,1,1]`.
    /// `w_gate_q = [0,0,0,0]` gives `sigmoid(gate) = 0.5` uniformly (defect
    /// 1's own fix: dropped in the old code, present here as the factor of
    /// `2` between `attended` and `attn_out` below): `gated_attended =
    /// [0.5,0.5,0.5,0.5]`. `wo = [1,1,1,1]` gives `attn_out = 0.5*4 = 2.0`,
    /// `residual1 = 2.0 + 2.0 = 4.0`. `normed2 = 4/|4| = 1` (`d=1` again),
    /// `feed_forward = 1`, `w_gate = w_up = w_down = [1]`: `silu(1) =
    /// 1 * sigmoid(1) ≈ 0.7310586`, `ffn_hidden = 0.7310586 * 1`,
    /// `ffn_out = 0.7310586`, `x_next = 0.7310586 + 4.0 ≈ 4.7310586`.
    #[test]
    fn append_qwen35_dense_attention_layer_matches_a_hand_computed_gate_and_partial_rotary_concat() {
        let (x_next, ..) =
            evaluate_dense_attention_test_program(0.0, [0.0f32, 0.0, 0.0, 0.0]);

        assert!(
            (x_next - 4.731_058_6).abs() < 1e-4,
            "x_next = ffn_out + residual1, hand-computed 4.7310586, got {x_next}"
        );
    }

    /// Proof the hand-computed test above can fail: a nonzero `w_gate_q`
    /// (the sigmoid gate this session's own defect 1 fix applies) MUST move
    /// `x_next` away from the `w_gate_q = [0,0,0,0]` reference above --
    /// `w_gate_q = [10,10,10,10]` pushes `sigmoid(gate)` from `0.5` toward
    /// `1.0`, doubling `attended`'s own contribution to `attn_out` before
    /// `wo`. A test that could not fail here would not have caught the old
    /// code's dropped gate either.
    #[test]
    fn append_qwen35_dense_attention_layer_hand_computed_check_actually_detects_a_dropped_gate() {
        let (x_next_no_gate, ..) =
            evaluate_dense_attention_test_program(0.0, [0.0f32, 0.0, 0.0, 0.0]);
        let (x_next_gated, ..) =
            evaluate_dense_attention_test_program(0.0, [10.0f32, 10.0, 10.0, 10.0]);

        assert!(
            (x_next_no_gate - x_next_gated).abs() > 0.1,
            "w_gate_q=[0,0,0,0] (sigmoid=0.5) vs [10,10,10,10] (sigmoid~1.0) must move x_next: \
             {x_next_no_gate} vs {x_next_gated}"
        );
    }

    /// One call into [`append_qwen35_dense_attention_layer`] at the fixed
    /// small dims the two tests above hand-compute against -- `gate_data`
    /// is the only knob a caller varies (`w_gate_q`'s own 4 values), so the
    /// mutation test above and the base test share every other weight byte
    /// for byte.
    fn evaluate_dense_attention_test_program(
        eps_value: f32,
        gate_data: [f32; 4],
    ) -> (f32, f32, f32, f32, f32) {
        let mut program = Vec::new();
        let scalar_shape = alloc::vec![Extent::Symbolic(0), Extent::Static(1)];
        let rotary_shape = alloc::vec![Extent::Symbolic(0), Extent::Static(1)];
        let head4_shape = alloc::vec![Extent::Static(1), Extent::Static(1), Extent::Static(4)];
        let cache4_shape = alloc::vec![Extent::Symbolic(1), Extent::Static(1), Extent::Static(1)];
        let cache_pass_shape = alloc::vec![Extent::Symbolic(1), Extent::Static(1), Extent::Static(2)];
        let cache_v_shape = alloc::vec![Extent::Symbolic(1), Extent::Static(1), Extent::Static(4)];
        let norm_shape = alloc::vec![Extent::Static(4)];
        let ffn_shape = alloc::vec![Extent::Static(1), Extent::Static(1)];

        let x = input_leaf(&mut program, DType::Float32, scalar_shape.clone(), "x");
        let inv_dim = scalar_constant(&mut program, 1.0);
        let eps = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Symbolic(0)], "eps");
        let ones = scalar_constant(&mut program, 1.0);
        let inv_sqrt_attn_head_dim = scalar_constant(&mut program, 0.5);
        let inv_attn_head_dim = scalar_constant(&mut program, 0.25);
        let cos_new = input_leaf(&mut program, DType::Float32, rotary_shape.clone(), "cos");
        let sin_new = input_leaf(&mut program, DType::Float32, rotary_shape, "sin");
        let group_ones = op::append(
            &mut program,
            Op::Constant {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(1), Extent::Static(1)],
                value: 1.0,
            },
        );
        let (is_future, _neg_infinity) = causal_mask(&mut program).expect("causal mask lowers");
        let cached_len = input_leaf(&mut program, DType::Float32, Vec::new(), "cached_len");

        let attn_norm_weight = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(1)], "attn_norm_weight");
        let ffn_norm_weight = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(1)], "ffn_norm_weight");
        let q_norm_weight = input_leaf(&mut program, DType::Float32, norm_shape.clone(), "q_norm_weight");
        let k_norm_weight = input_leaf(&mut program, DType::Float32, norm_shape, "k_norm_weight");
        let wq = input_leaf(&mut program, DType::Float32, head4_shape.clone(), "wq");
        let w_gate_q = input_leaf(&mut program, DType::Float32, head4_shape.clone(), "w_gate_q");
        let wk = input_leaf(&mut program, DType::Float32, head4_shape.clone(), "wk");
        let wv = input_leaf(&mut program, DType::Float32, head4_shape.clone(), "wv");
        let wo = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(1), Extent::Static(1), Extent::Static(4), Extent::Static(1)], "wo");
        let w_gate = input_leaf(&mut program, DType::Float32, ffn_shape.clone(), "w_gate");
        let w_up = input_leaf(&mut program, DType::Float32, ffn_shape.clone(), "w_up");
        let w_down = input_leaf(&mut program, DType::Float32, ffn_shape, "w_down");
        let k_first_cache = input_leaf(&mut program, DType::Float32, cache4_shape.clone(), "k_first_cache");
        let k_second_cache = input_leaf(&mut program, DType::Float32, cache4_shape, "k_second_cache");
        let k_pass_cache = input_leaf(&mut program, DType::Float32, cache_pass_shape, "k_pass_cache");
        let v_cache = input_leaf(&mut program, DType::Float32, cache_v_shape, "v_cache");

        let (x_next, (rotated_k_first, rotated_k_second, k_pass, v_new)) = append_qwen35_dense_attention_layer(
            &mut program,
            x,
            inv_dim,
            eps,
            ones,
            inv_sqrt_attn_head_dim,
            inv_attn_head_dim,
            cos_new,
            sin_new,
            group_ones,
            is_future,
            cached_len,
            1,
            2,
            4,
            attn_norm_weight,
            ffn_norm_weight,
            q_norm_weight,
            k_norm_weight,
            wq,
            w_gate_q,
            wk,
            wv,
            wo,
            w_gate,
            w_up,
            w_down,
            k_first_cache,
            k_second_cache,
            k_pass_cache,
            v_cache,
        )
        .expect("dense attention layer lowers");

        let x_data = [2.0f32];
        let eps_data = [eps_value];
        let cos_data = [0.0f32];
        let sin_data = [1.0f32];
        let attn_norm_data = [1.0f32];
        let ffn_norm_data = [1.0f32];
        let q_norm_data = [1.0f32, 1.0, 1.0, 1.0];
        let k_norm_data = [1.0f32, 1.0, 1.0, 1.0];
        let wq_data = [1.0f32, 1.0, 1.0, 1.0];
        let wk_data = [1.0f32, 1.0, 1.0, 1.0];
        let wv_data = [1.0f32, 1.0, 1.0, 1.0];
        let wo_data = [1.0f32, 1.0, 1.0, 1.0];
        let w_gate_data = [1.0f32];
        let w_up_data = [1.0f32];
        let w_down_data = [1.0f32];
        let empty: [f32; 0] = [];
        let cached_len_data = [0.0f32];

        let evaluated = crate::cpu::evaluate_named(
            &program,
            &[1, 0],
            &[
                ("x", &x_data),
                ("eps", &eps_data),
                ("cos", &cos_data),
                ("sin", &sin_data),
                ("cached_len", &cached_len_data),
                ("attn_norm_weight", &attn_norm_data),
                ("ffn_norm_weight", &ffn_norm_data),
                ("q_norm_weight", &q_norm_data),
                ("k_norm_weight", &k_norm_data),
                ("wq", &wq_data),
                ("w_gate_q", &gate_data),
                ("wk", &wk_data),
                ("wv", &wv_data),
                ("wo", &wo_data),
                ("w_gate", &w_gate_data),
                ("w_up", &w_up_data),
                ("w_down", &w_down_data),
                ("k_first_cache", &empty),
                ("k_second_cache", &empty),
                ("k_pass_cache", &empty),
                ("v_cache", &empty),
            ],
            &[x_next, rotated_k_first, rotated_k_second, k_pass, v_new],
        )
        .expect("dense attention layer evaluates");

        let (x_next_values, _) = evaluated.get(x_next).expect("x_next present");
        let (k_first_values, _) = evaluated.get(rotated_k_first).expect("k_first present");
        let (k_second_values, _) = evaluated.get(rotated_k_second).expect("k_second present");
        let (k_pass_values, _) = evaluated.get(k_pass).expect("k_pass present");
        let (v_new_values, _) = evaluated.get(v_new).expect("v_new present");

        (
            x_next_values[0],
            k_first_values[0],
            k_second_values[0],
            k_pass_values[0],
            v_new_values[0],
        )
    }

    /// [`qwen35_forward_program`]'s whole-program wiring, both layer kinds
    /// in one small stack (`block_count = 4`, `full_attention_interval =
    /// 2`, so layers 0,2 are SSM and layers 1,3 are dense attention, per
    /// its own `(layer + 1) % full_attention_interval == 0` doc) --
    /// mirroring [`the_whole_mistral_forward_pass_infers_at_real_dimensions`]'s
    /// own "lowers, then `shape::infer` succeeds" scope, not a numeric
    /// check (the mixer's own hand-computed test below already owns that).
    /// `symbols = [1, 0]`: one new decode-step token, an empty dense-attention
    /// KV cache -- [`append_mistral_cached_layer`]'s own doc already proves
    /// `cached_len == 0` degenerates to plain self-attention with no special
    /// case.
    #[test]
    fn the_whole_qwen35_forward_pass_infers_at_real_dimensions() {
        let (program, logits, roots) =
            qwen35_forward_program(100, 8, 16, 2, 1, 4, 8, 4, 2, 2, 2, 1, 4, 3, 1e-5)
                .expect("the whole qwen35 forward pass lowers to a program");

        assert_eq!(roots.len(), 4, "one root set per block");
        assert!(
            matches!(roots[0], Qwen35LayerRoots::Ssm { .. }),
            "layer 0 is SSM: (0 + 1) % 2 != 0"
        );
        assert!(
            matches!(roots[1], Qwen35LayerRoots::DenseAttention(_)),
            "layer 1 is dense attention: (1 + 1) % 2 == 0"
        );
        assert!(
            matches!(roots[2], Qwen35LayerRoots::Ssm { .. }),
            "layer 2 is SSM: (2 + 1) % 2 != 0"
        );
        assert!(
            matches!(roots[3], Qwen35LayerRoots::DenseAttention(_)),
            "layer 3 is dense attention: (3 + 1) % 2 == 0"
        );

        crate::shape::infer(&program, &[1, 0])
            .expect("the whole qwen35 forward pass infers at a real decode step");
        let _ = logits;
    }

    /// [`the_whole_qwen35_forward_pass_infers_at_real_dimensions`]'s own
    /// `(100, 8, 16, 2, 1, 4, 4, 2, 2, 2, 1, 4, 3, 1e-5)` never caught the
    /// `attn_q`/`attn_k`/`attn_v`/`attn_output` shape defect this test is
    /// named for -- ROOT CAUSE, proved by direct comparison against the
    /// real Qwen3.5-2B-Q4_K_M checkpoint's own on-disk tensor dims
    /// (`proxima_model_interop::qwen35`'s own
    /// `scratch_debug_real_attn_q_dims`-shaped probe, run against the real
    /// file): every dense-attention weight was declared using `head_dim`
    /// (`rope.dimension_count`, this checkpoint's PARTIAL-rotary width, `64`)
    /// as the per-head PROJECTION width too, but the real per-head
    /// projection width is `embedding / query_heads` (`256` on the 2B,
    /// matching `attn_q_norm.weight`'s/`attn_k_norm.weight`'s own on-disk
    /// width exactly, and `qwen3_next`'s own `Qwen3NextAttention.__init__`,
    /// `self.head_dim = hidden_size // num_attention_heads`) -- and
    /// `attn_q.weight`'s own on-disk width is DOUBLE that again
    /// (`query_heads * 256 * 2 = 4096`, not `query_heads * 64 = 512`): a
    /// same-width sigmoid gate fused per head
    /// (`modeling_qwen3_next.py:293-326`, `torch.chunk(2, dim=-1)` on each
    /// head's own `2 * head_dim`-wide block after `.view(..., heads, 2 *
    /// head_dim)`), which this program drops (never applies) rather than
    /// implements, an accepted extension of this program's existing
    /// single-section-RoPE gap.
    ///
    /// The toy test's own `query_heads = 2`, `embedding = 8` degenerate
    /// case makes `embedding / query_heads == 4 == head_dim` BY
    /// COINCIDENCE (the toy dims were never chosen to keep those two
    /// quantities apart), so the toy program's `attn_q`/`attn_k`/`attn_v`
    /// declared shapes matched what [`shape::infer`] expected regardless of
    /// which formula built them -- the defect is invisible at any dimension
    /// set where `embedding / query_heads == head_dim`, which is every toy
    /// dimension set this module's own tests use and no real checkpoint's
    /// own numbers. This test is the fix for THAT gap: real per-head
    /// dimensions, not toy ones that happen to collide.
    #[test]
    fn the_whole_qwen35_forward_pass_infers_at_the_2b_checkpoints_real_dimensions() {
        // Qwen3.5-2B-Q4_K_M's own metadata: vocab=151936, embedding=2048,
        // feed_forward=6144, query_heads=8, kv_heads=2, head_dim=64
        // (rope.dimension_count), block_count=24, full_attention_interval=4,
        // ssm_state_size=128, ssm_time_step_rank=16, ssm_group_count=16,
        // ssm_inner_size=2048, ssm_conv_kernel=4, rms_epsilon=1e-6.
        let (program, _logits, roots) = qwen35_forward_program(
            151936, 2048, 6144, 8, 2, 64, 256, 24, 4, 128, 16, 16, 2048, 4, 1e-6,
        )
        .expect("the real 2b's own dimensions lower to a program");

        assert_eq!(roots.len(), 24, "one root set per block");
        assert!(
            matches!(roots[3], Qwen35LayerRoots::DenseAttention(_)),
            "layer 3 is dense attention: (3 + 1) % 4 == 0"
        );
        assert!(
            matches!(roots[0], Qwen35LayerRoots::Ssm { .. }),
            "layer 0 is SSM: (0 + 1) % 4 != 0"
        );

        // prefill (6 new tokens, empty cache) and three decode steps against
        // a growing cache -- the shapes this program actually runs under
        // `proxima_model_interop::generate`, not just a single decode step.
        for (new_count, cached_len) in [(6u64, 0u64), (1, 6), (1, 7), (1, 8)] {
            crate::shape::infer(&program, &[new_count, cached_len]).unwrap_or_else(|err| {
                panic!(
                    "the real 2b's own dimensions infer at new_count={new_count} \
                     cached_len={cached_len}: {err:?}"
                )
            });
        }
    }

    /// [`append_qwen35_ssm_mixer`] against the hand-derivation in
    /// [`build_ssm_mixer_test_program`]'s own doc: `history = [q=3, k=-2,
    /// v0=1, v1=2]` conv-blends straight through (new-token tap weighted by
    /// a zero `qkv_mixed`), `silu` gives `q_raw = 2.85772238`, `k_raw =
    /// -0.23840584`, `v_conv = [0.73105858, 1.76159416]`; width-1 `l2norm`
    /// collapses `q_conv = 1`, `k_conv = -1`; the recurrence (`decay = 1`,
    /// `beta = 0.5`, zero `state_in`) gives `state_out = out = [-0.36552929,
    /// -0.88079708]` per group; the gated RMSNorm's width-1 mean-square
    /// gives `normed_out = sign(out) = -1` for both groups, so
    /// `normed_out_gamma = -2`, `gated_out = normed_out_gamma * silu(z)`
    /// with `z = [1, 2]` (same `silu` values as `v`) gives `[-1.46211716,
    /// -3.52318831]`, and the `[1, 1]` output weight sums those into `cur =
    /// -4.98530547`, `mixer_out = x + cur = -3.98530547`.
    #[proxima::test]
    async fn qwen35_ssm_mixer_matches_a_hand_computed_decode_step() {
        let (program, mixer_out, qkv_mixed, state_out) = build_ssm_mixer_test_program(GdnOutputGate::Silu);

        let x_data = [1.0f32];
        let eps_data = [0.0f32];
        let head_eps_data = [0.0f32, 0.0];
        let attn_norm_weight_data = [1.0f32];
        let wqkv_data = [0.0f32, 0.0, 0.0, 0.0];
        let wqkv_gate_data = [1.0f32, 2.0];
        let conv_weight_data = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let conv_history_in_data = [3.0f32, -2.0, 1.0, 2.0];
        let ssm_beta_data = [0.0f32, 0.0];
        let ssm_alpha_data = [0.0f32, 0.0];
        let ssm_dt_bias_data = [0.0f32, 0.0];
        let ssm_a_data = [0.0f32, 0.0];
        let ssm_norm_weight_data = [2.0f32];
        let ssm_out_data = [1.0f32, 1.0];
        let state_in_data = [0.0f32, 0.0];

        let evaluated = crate::cpu::evaluate_named(
            &program,
            &[1],
            &[
                ("x", &x_data),
                ("eps", &eps_data),
                ("head_eps", &head_eps_data),
                ("attn_norm_weight", &attn_norm_weight_data),
                ("wqkv", &wqkv_data),
                ("wqkv_gate", &wqkv_gate_data),
                ("conv_weight", &conv_weight_data),
                ("conv_history_in", &conv_history_in_data),
                ("ssm_beta", &ssm_beta_data),
                ("ssm_alpha", &ssm_alpha_data),
                ("ssm_dt_bias", &ssm_dt_bias_data),
                ("ssm_a", &ssm_a_data),
                ("ssm_norm_weight", &ssm_norm_weight_data),
                ("ssm_out", &ssm_out_data),
                ("state_in", &state_in_data),
            ],
            &[mixer_out, qkv_mixed, state_out],
        )
        .expect("ssm mixer evaluates");

        let (mixer_out_values, _) = evaluated.get(mixer_out).expect("mixer_out present");
        let (qkv_mixed_values, _) = evaluated.get(qkv_mixed).expect("qkv_mixed present");
        let (state_out_values, _) = evaluated.get(state_out).expect("state_out present");

        assert!(
            (mixer_out_values[0] - (-3.985_305_5)).abs() < 1e-4,
            "got {}",
            mixer_out_values[0]
        );
        assert_eq!(qkv_mixed_values, [0.0, 0.0, 0.0, 0.0], "wqkv is zero, so qkv_mixed is zero");
        assert!(
            (state_out_values[0] - (-0.365_529_3)).abs() < 1e-4,
            "got {}",
            state_out_values[0]
        );
        assert!(
            (state_out_values[1] - (-0.880_797_1)).abs() < 1e-4,
            "got {}",
            state_out_values[1]
        );
    }

    /// Same inputs as the hand-computed decode step above, but with
    /// [`GdnOutputGate::Sigmoid`] instead of [`GdnOutputGate::Silu`] --
    /// qwen4exp's own gate (reference: PR 27742 line 2895-2897). Proves the
    /// flag actually changes the lowered program's output (not silently
    /// ignored): `sigmoid(z) != silu(z)` for this test's own `z = [1, 2]`,
    /// so `mixer_out` must move away from the Silu path's own hand-computed
    /// `-3.985_305_5`.
    #[proxima::test]
    async fn qwen35_ssm_mixer_sigmoid_gate_moves_the_output_away_from_silu() {
        let (program, mixer_out, _, _) = build_ssm_mixer_test_program(GdnOutputGate::Sigmoid);

        let x_data = [1.0f32];
        let eps_data = [0.0f32];
        let head_eps_data = [0.0f32, 0.0];
        let attn_norm_weight_data = [1.0f32];
        let wqkv_data = [0.0f32, 0.0, 0.0, 0.0];
        let wqkv_gate_data = [1.0f32, 2.0];
        let conv_weight_data = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let conv_history_in_data = [3.0f32, -2.0, 1.0, 2.0];
        let ssm_beta_data = [0.0f32, 0.0];
        let ssm_alpha_data = [0.0f32, 0.0];
        let ssm_dt_bias_data = [0.0f32, 0.0];
        let ssm_a_data = [0.0f32, 0.0];
        let ssm_norm_weight_data = [2.0f32];
        let ssm_out_data = [1.0f32, 1.0];
        let state_in_data = [0.0f32, 0.0];

        let evaluated = crate::cpu::evaluate_named(
            &program,
            &[1],
            &[
                ("x", &x_data),
                ("eps", &eps_data),
                ("head_eps", &head_eps_data),
                ("attn_norm_weight", &attn_norm_weight_data),
                ("wqkv", &wqkv_data),
                ("wqkv_gate", &wqkv_gate_data),
                ("conv_weight", &conv_weight_data),
                ("conv_history_in", &conv_history_in_data),
                ("ssm_beta", &ssm_beta_data),
                ("ssm_alpha", &ssm_alpha_data),
                ("ssm_dt_bias", &ssm_dt_bias_data),
                ("ssm_a", &ssm_a_data),
                ("ssm_norm_weight", &ssm_norm_weight_data),
                ("ssm_out", &ssm_out_data),
                ("state_in", &state_in_data),
            ],
            &[mixer_out],
        )
        .expect("ssm mixer evaluates");

        let (mixer_out_values, _) = evaluated.get(mixer_out).expect("mixer_out present");
        assert!(
            (mixer_out_values[0] - (-3.985_305_5)).abs() > 1e-3,
            "sigmoid gate must move mixer_out away from the silu path's own -3.985305_5, got {}",
            mixer_out_values[0]
        );
    }

    /// Proof the mixer reference above can fail: perturbing the conv
    /// history's `v0` channel (`1.0 -> -3.0`, a sign flip -- see the data
    /// comment below for why a same-sign perturbation alone cannot move this
    /// particular degenerate configuration) must move `mixer_out` away from
    /// the hand-computed reference -- `v0` feeds `v_split`'s `g = 0` group
    /// straight into the recurrence's `value` operand and back out through
    /// the gated norm and output projection, so a wrong history value is
    /// never silently absorbed.
    #[proxima::test]
    async fn qwen35_ssm_mixer_hand_computed_check_actually_detects_a_wrong_history_value() {
        let (program, mixer_out, _, _) = build_ssm_mixer_test_program(GdnOutputGate::Silu);

        let x_data = [1.0f32];
        let eps_data = [0.0f32];
        let head_eps_data = [0.0f32, 0.0];
        let attn_norm_weight_data = [1.0f32];
        let wqkv_data = [0.0f32, 0.0, 0.0, 0.0];
        let wqkv_gate_data = [1.0f32, 2.0];
        let conv_weight_data = [1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        // v0 perturbed from the reference's 1.0 to -3.0 -- a SIGN flip, not
        // just a magnitude change: `head_v_dim = 1` degenerates the gated
        // RMSNorm's own normalize step to `sign(delta_out)` (the same
        // width-1-l2norm-is-sign identity `q_conv`/`k_conv` already exploit),
        // so a same-sign magnitude perturbation alone never moves
        // `mixer_out` at this degenerate width -- confirmed empirically (a
        // first version of this test perturbed v0 to 2.0 and the assertion
        // could not fail, exactly the failure mode this test's own doc
        // warns against).
        let conv_history_in_data = [3.0f32, -2.0, -3.0, 2.0];
        let ssm_beta_data = [0.0f32, 0.0];
        let ssm_alpha_data = [0.0f32, 0.0];
        let ssm_dt_bias_data = [0.0f32, 0.0];
        let ssm_a_data = [0.0f32, 0.0];
        let ssm_norm_weight_data = [2.0f32];
        let ssm_out_data = [1.0f32, 1.0];
        let state_in_data = [0.0f32, 0.0];

        let evaluated = crate::cpu::evaluate_named(
            &program,
            &[1],
            &[
                ("x", &x_data),
                ("eps", &eps_data),
                ("head_eps", &head_eps_data),
                ("attn_norm_weight", &attn_norm_weight_data),
                ("wqkv", &wqkv_data),
                ("wqkv_gate", &wqkv_gate_data),
                ("conv_weight", &conv_weight_data),
                ("conv_history_in", &conv_history_in_data),
                ("ssm_beta", &ssm_beta_data),
                ("ssm_alpha", &ssm_alpha_data),
                ("ssm_dt_bias", &ssm_dt_bias_data),
                ("ssm_a", &ssm_a_data),
                ("ssm_norm_weight", &ssm_norm_weight_data),
                ("ssm_out", &ssm_out_data),
                ("state_in", &state_in_data),
            ],
            &[mixer_out],
        )
        .expect("ssm mixer evaluates");

        let (mixer_out_values, _) = evaluated.get(mixer_out).expect("mixer_out present");

        assert!(
            (mixer_out_values[0] - (-3.985_305_5)).abs() > 1e-3,
            "a perturbed history v0 must move mixer_out away from the hand-computed reference \
             (if this assertion cannot fail, the test above proves nothing), got {}",
            mixer_out_values[0]
        );
    }

    /// One `append_mistral_single_range_cached_layer` invocation, minus the
    /// `gate_before_up` flag under test -- the minimal single-layer preamble
    /// [`mistral_single_range_cached_forward_program`]'s own loop body builds
    /// for `block_count = 1`, `query_heads = kv_heads = 1`, `head_dim = 2`,
    /// `embedding = feed_forward = 2` (small enough to read by eye, large
    /// enough that `w_gate`/`w_up`'s shapes are distinguishable from every
    /// other node's).
    fn single_range_layer_with_order(
        gate_before_up: bool,
        qk_norm: bool,
    ) -> Result<(Vec<Op>, NodeId, NodeId), TensorError> {
        let embedding = 2_u32;
        let feed_forward = 2_u32;
        let head_dim = 2_u32;
        let pairs = head_dim / 2;
        let group = 1_u32;

        let mut program = Vec::new();
        let x = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(0), Extent::Static(embedding)],
            "x",
        );
        let inv_dim = scalar_constant(&mut program, 1.0 / embedding as f32);
        let eps = symbolic_leaf(&mut program, DType::Float32, "eps");
        let ones = scalar_constant(&mut program, 1.0);
        let inv_sqrt_head_dim = scalar_constant(&mut program, 1.0 / (head_dim as f32).sqrt());
        let cos_new = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(0), Extent::Static(pairs)],
            "rope_cos",
        );
        let sin_new = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(0), Extent::Static(pairs)],
            "rope_sin",
        );
        let group_ones = op::append(
            &mut program,
            Op::Constant {
                dtype: DType::Float32,
                shape: alloc::vec![Extent::Static(1), Extent::Static(group)],
                value: 1.0,
            },
        );
        let cached_len = input_leaf(&mut program, DType::Float32, Vec::new(), "cached_len");
        let is_future = causal_mask_merged(&mut program, cached_len).expect("causal mask builds");
        let attn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            "attn_norm.weight",
        );
        let ffn_norm_weight = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding)],
            "ffn_norm.weight",
        );
        let wq = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(embedding),
                Extent::Static(1),
                Extent::Static(head_dim)
            ],
            "attn_q.weight",
        );
        let wk = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(embedding),
                Extent::Static(1),
                Extent::Static(head_dim)
            ],
            "attn_k.weight",
        );
        let wv = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(embedding),
                Extent::Static(1),
                Extent::Static(head_dim)
            ],
            "attn_v.weight",
        );
        let wo = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Static(1),
                Extent::Static(group),
                Extent::Static(head_dim),
                Extent::Static(embedding),
            ],
            "attn_output.weight",
        );
        let k_even_cache = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(1), Extent::Static(1), Extent::Static(pairs)],
            "kv_cache.k_even",
        );
        let k_odd_cache = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(1), Extent::Static(1), Extent::Static(pairs)],
            "kv_cache.k_odd",
        );
        let v_cache = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![
                Extent::Symbolic(1),
                Extent::Static(1),
                Extent::Static(head_dim)
            ],
            "kv_cache.v",
        );
        let w_gate = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding), Extent::Static(feed_forward)],
            "ffn_gate.weight",
        );
        let w_up = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding), Extent::Static(feed_forward)],
            "ffn_up.weight",
        );
        let w_down = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(feed_forward), Extent::Static(embedding)],
            "ffn_down.weight",
        );
        let qk_norm_weights = qk_norm.then(|| {
            let inv_head_dim = scalar_constant(&mut program, 1.0 / head_dim as f32);
            let q_norm_weight = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(head_dim)],
                "attn_q_norm.weight",
            );
            let k_norm_weight = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(head_dim)],
                "attn_k_norm.weight",
            );
            (q_norm_weight, k_norm_weight, inv_head_dim)
        });

        append_mistral_single_range_cached_layer(
            &mut program,
            x,
            inv_dim,
            eps,
            ones,
            inv_sqrt_head_dim,
            cos_new,
            sin_new,
            group_ones,
            is_future,
            group,
            head_dim,
            attn_norm_weight,
            ffn_norm_weight,
            wq,
            wk,
            wv,
            wo,
            w_gate,
            w_up,
            w_down,
            k_even_cache,
            k_odd_cache,
            v_cache,
            qk_norm_weights,
            gate_before_up,
        )?;

        Ok((program, w_gate, w_up))
    }

    /// ROW 373: the single-range builder no longer rejects a qk-norm
    /// checkpoint -- it binds the same shape [`append_mistral_cached_layer`]
    /// would. This is the acceptance replacement for ROW 372's rejection
    /// test: a qk-norm layer must produce the two extra `rmsnorm_per_head`
    /// reduces (one per q/k) that a plain interleaved layer does not, and
    /// nothing else about its op-kind census should move (same op COUNT per
    /// kind elsewhere -- the two-range sibling's own `qk_norm` doc already
    /// establishes those two reduces are the ENTIRE cost of this feature).
    #[test]
    fn single_range_layer_binds_qk_norm_with_two_extra_per_head_norm_reduces() {
        let (plain, _, _) = single_range_layer_with_order(true, false)
            .expect("a plain interleaved layer still builds");
        let (normed, _, _) =
            single_range_layer_with_order(true, true).expect("a qk-norm layer now builds");

        let reduce_count = |program: &[Op]| program.iter().filter(|op| matches!(op, Op::Reduce { .. })).count();
        let elementwise_count =
            |program: &[Op]| program.iter().filter(|op| matches!(op, Op::Elementwise { .. })).count();

        assert_eq!(
            reduce_count(&normed),
            reduce_count(&plain) + 2,
            "qk-norm adds exactly the two rmsnorm_per_head reduces (q, k) over the plain layer"
        );
        assert_eq!(
            elementwise_count(&normed),
            elementwise_count(&plain) + 14,
            "rmsnorm_per_head is 7 elementwise ops per call (squared, mean_square, \
             mean_square_eps, rms, inv_rms, normed, gamma-scale), twice (q and k) -- 14 more \
             elementwise ops, nothing else moves"
        );
    }

    /// The `PROXIMA_ENCODE_ORDER=gate_first|up_first` order-swap knob
    /// (`test_support::encode_order_from_env` in
    /// `proxima-model-interop`) is honored here at the program-builder
    /// level: `append_mistral_single_range_cached_layer`'s `gate_before_up`
    /// flag decides only which of the two independent FFN matvecs (both
    /// read `normed2`, neither reads the other) is PUSHED first — the node
    /// SET and every dependency is unchanged, only their relative order.
    #[test]
    fn swapping_gate_and_up_order_keeps_dataflow_identical() {
        let (gate_first, w_gate_a, w_up_a) = single_range_layer_with_order(true, false)
            .expect("single-range layer builds under either encode order");
        let (up_first, w_gate_b, w_up_b) = single_range_layer_with_order(false, false)
            .expect("single-range layer builds under either encode order");

        assert_eq!(
            gate_first.len(),
            up_first.len(),
            "swapping encode order must not add or drop a single node"
        );

        let reads_operand = |op: &Op, operand: NodeId| -> bool {
            matches!(op, Op::Elementwise { operands, .. }
                if operands.iter().any(|(id, _)| *id == operand))
        };
        let gate_position = |program: &[Op], w_gate: NodeId| -> usize {
            program
                .iter()
                .position(|op| reads_operand(op, w_gate))
                .expect("a ffn_gate.weight-reading elementwise node must exist")
        };
        let up_position = |program: &[Op], w_up: NodeId| -> usize {
            program
                .iter()
                .position(|op| reads_operand(op, w_up))
                .expect("a ffn_up.weight-reading elementwise node must exist")
        };

        let gate_before_gate_first = gate_position(&gate_first, w_gate_a);
        let up_before_gate_first = up_position(&gate_first, w_up_a);
        assert!(
            gate_before_gate_first < up_before_gate_first,
            "gate_before_up=true must encode ffn_gate ({gate_before_gate_first}) before \
             ffn_up ({up_before_gate_first})"
        );

        let gate_before_up_first = gate_position(&up_first, w_gate_b);
        let up_before_up_first = up_position(&up_first, w_up_b);
        assert!(
            up_before_up_first < gate_before_up_first,
            "gate_before_up=false must encode ffn_up ({up_before_up_first}) before \
             ffn_gate ({gate_before_up_first})"
        );

        // same node SET, different order: every op kind/shape present in one
        // program appears the same number of times in the other, just at a
        // different index -- a multiset comparison over each op's discriminant
        // plus its dtype (position-independent, unlike operand `NodeId`s,
        // which legitimately renumber when the gate/up pair swaps).
        let signature = |op: &Op| -> (core::mem::Discriminant<Op>, DType) {
            let dtype = match op {
                Op::Input { dtype, .. }
                | Op::Elementwise { dtype, .. }
                | Op::Constant { dtype, .. }
                | Op::Iota { dtype, .. } => *dtype,
                Op::Reduce(reduce) => reduce.dtype,
            };
            (core::mem::discriminant(op), dtype)
        };
        let mut gate_first_signatures: Vec<_> = gate_first.iter().map(signature).collect();
        let mut up_first_signatures: Vec<_> = up_first.iter().map(signature).collect();
        gate_first_signatures.sort_by_key(|(discriminant, dtype)| {
            (format!("{discriminant:?}"), format!("{dtype:?}"))
        });
        up_first_signatures.sort_by_key(|(discriminant, dtype)| {
            (format!("{discriminant:?}"), format!("{dtype:?}"))
        });
        assert_eq!(
            gate_first_signatures, up_first_signatures,
            "the two programs must carry the identical multiset of op kinds -- \
             the swap must reorder nodes, never add, drop, or retype one"
        );
    }
}
