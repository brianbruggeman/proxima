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
fn parse_axis_expr(token: &str, space: &[char], notation: &str) -> Result<AxisIndex, TensorError> {
    let mut terms: Vec<AxisTerm> = Vec::new();
    let mut offset: i32 = 0;
    for (sign, part) in split_signed_terms(token) {
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
    Ok(AxisIndex {
        terms: terms.into_iter().collect(),
        offset,
    })
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
/// iteration` notation through the same [`parse_operand_pattern`] the TOML
/// lowering above uses. This is the whole reason a hand-built full-model
/// program stays honest to the TOML one node kind spells: both paths run the
/// identical grammar, so a generated layer cannot silently drift from
/// `specs/mistral_layer.toml`'s own addressing.
fn elementwise(
    program: &mut Vec<Op>,
    dtype: DType,
    body: ScalarOp,
    inputs: &[(NodeId, &str)],
) -> Result<NodeId, TensorError> {
    let operands = inputs
        .iter()
        .map(|(node, notation)| {
            Ok((*node, IndexMap::Affine(parse_operand_pattern(notation)?)))
        })
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
fn reduce(
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

/// `[?0]`-shaped bound leaf. `eps` is the only caller left: RMSNorm's
/// epsilon is model metadata (`attention.layer_norm_rms_epsilon` in a GGUF
/// checkpoint), not a value this function's `u32` parameters determine, so
/// it is the one constant here that cannot become an [`Op::Constant`]
/// without `mistral_forward_program` taking it as a parameter.
/// `inv_dim`/`ones`/`group_ones` all could, and did — see [`scalar_constant`].
fn symbolic_leaf(program: &mut Vec<Op>, dtype: DType, name: &str) -> NodeId {
    input_leaf(program, dtype, alloc::vec![Extent::Symbolic(0)], name)
}

fn input_leaf(program: &mut Vec<Op>, dtype: DType, shape: Vec<Extent>, name: &str) -> NodeId {
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
fn scalar_constant(program: &mut Vec<Op>, value: f32) -> NodeId {
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
/// vocab axis, `d` passes through as a plain projection.
fn embedding_lookup(program: &mut Vec<Op>, table: NodeId, ids: NodeId) -> NodeId {
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
fn rmsnorm(program: &mut Vec<Op>, x: NodeId, gamma: NodeId, inv_dim: NodeId, eps: NodeId) -> Result<NodeId, TensorError> {
    let squared = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(x, "sd->sd"), (x, "sd->sd")])?;
    let sum_squares = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, squared, "sd->sd", "s->sd")?;
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
    let rms = elementwise(program, DType::Float32, ScalarOp::SquareRoot, &[(mean_square_eps, "s->s")])?;
    let inv_rms = elementwise(program, DType::Float32, ScalarOp::Reciprocal, &[(rms, "s->s")])?;
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
fn causal_mask(program: &mut Vec<Op>) -> Result<(NodeId, NodeId), TensorError> {
    let query_index = op::append(program, Op::Iota { dtype: DType::Float32, extent: Extent::Symbolic(0) });
    let key_index = op::append(program, Op::Iota { dtype: DType::Float32, extent: Extent::Symbolic(0) });
    let is_future = elementwise(
        program,
        DType::Float32,
        ScalarOp::Greater,
        &[(key_index, "t->st"), (query_index, "s->st")],
    )?;
    let neg_infinity = scalar_constant(program, f32::NEG_INFINITY);
    Ok((is_future, neg_infinity))
}

/// One transformer layer, node-for-node the same graph
/// `specs/mistral_layer.toml` spells — attention (RoPE + GQA + causal mask)
/// then the SwiGLU feed-forward, each wrapped in its own residual. `x` in,
/// the layer's own residual-summed output out; every other argument is
/// either a per-layer weight (`wq`/`wk`/`wv`/`wo`/`w_gate`/`w_up`/`w_down`)
/// or one of the position-only constants [`causal_mask`]/`cos`/`sin` share
/// across every layer.
#[allow(clippy::too_many_arguments)]
fn append_mistral_layer(
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

    let q_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(normed, "si->shdi"), (wq, "ihd->shdi")])?;
    let q = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, q_product, "shdi->shdi", "shd->shdi")?;

    let k_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(normed, "si->sudi"), (wk, "iud->sudi")])?;
    let k = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, k_product, "sudi->sudi", "sud->sudi")?;

    let v_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(normed, "si->sudi"), (wv, "iud->sudi")])?;
    let v = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, v_product, "sudi->sudi", "sud->sudi")?;

    let q_even_cos = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q, "s,h,2*i->shi"), (cos, "si->shi")])?;
    let q_odd_sin = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q, "s,h,2*i+1->shi"), (sin, "si->shi")])?;
    let rotated_q_even = elementwise(program, DType::Float32, ScalarOp::Subtract, &[(q_even_cos, "shi->shi"), (q_odd_sin, "shi->shi")])?;
    let q_even_sin = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q, "s,h,2*i->shi"), (sin, "si->shi")])?;
    let q_odd_cos = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q, "s,h,2*i+1->shi"), (cos, "si->shi")])?;
    let rotated_q_odd = elementwise(program, DType::Float32, ScalarOp::Add, &[(q_even_sin, "shi->shi"), (q_odd_cos, "shi->shi")])?;

    let k_even_cos = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(k, "s,u,2*i->sui"), (cos, "si->sui")])?;
    let k_odd_sin = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(k, "s,u,2*i+1->sui"), (sin, "si->sui")])?;
    let rotated_k_even = elementwise(program, DType::Float32, ScalarOp::Subtract, &[(k_even_cos, "sui->sui"), (k_odd_sin, "sui->sui")])?;
    let k_even_sin = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(k, "s,u,2*i->sui"), (sin, "si->sui")])?;
    let k_odd_cos = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(k, "s,u,2*i+1->sui"), (cos, "si->sui")])?;
    let rotated_k_odd = elementwise(program, DType::Float32, ScalarOp::Add, &[(k_even_sin, "sui->sui"), (k_odd_cos, "sui->sui")])?;

    let group_map = alloc::format!("s,{group}*u+g,i->sugi");
    let q_even_grouped = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(rotated_q_even, group_map.as_str()), (group_ones, "ug->sugi")])?;
    let q_odd_grouped = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(rotated_q_odd, group_map.as_str()), (group_ones, "ug->sugi")])?;

    let score_even_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q_even_grouped, "sugi->stugi"), (rotated_k_even, "tui->stugi")])?;
    let score_even = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, score_even_product, "stugi->stugi", "stug->stugi")?;
    let score_odd_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q_odd_grouped, "sugi->stugi"), (rotated_k_odd, "tui->stugi")])?;
    let score_odd = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, score_odd_product, "stugi->stugi", "stug->stugi")?;
    let scores = elementwise(program, DType::Float32, ScalarOp::Add, &[(score_even, "stug->stug"), (score_odd, "stug->stug")])?;

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
        &[(is_future, "st->stug"), (neg_infinity, "->stug"), (scores_scaled, "stug->stug")],
    )?;

    let score_max = reduce(program, DType::Float32, ScalarOp::Maximum, ReduceInit::NegativeInfinity, scores_masked, "stug->stug", "sug->stug")?;
    let shifted = elementwise(program, DType::Float32, ScalarOp::Subtract, &[(scores_masked, "stug->stug"), (score_max, "sug->stug")])?;
    let weights = elementwise(program, DType::Float32, ScalarOp::Exponential, &[(shifted, "stug->stug")])?;
    let weight_sum = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, weights, "stug->stug", "sug->stug")?;
    let inv_weight_sum = elementwise(program, DType::Float32, ScalarOp::Reciprocal, &[(weight_sum, "sug->sug")])?;
    let probabilities = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(weights, "stug->stug"), (inv_weight_sum, "sug->stug")])?;

    let attended_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(probabilities, "stug->stugd"), (v, "tud->stugd")])?;
    let attended = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, attended_product, "stugd->stugd", "sugd->stugd")?;

    let wo_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(attended, "sugd->sugdo"), (wo, "ugdo->sugdo")])?;
    let attn_out = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, wo_product, "sugdo->sugdo", "so->sugdo")?;

    let residual1 = elementwise(program, DType::Float32, ScalarOp::Add, &[(attn_out, "sd->sd"), (x, "sd->sd")])?;

    let normed2 = rmsnorm(program, residual1, ffn_norm_weight, inv_dim, eps)?;

    let gate_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(normed2, "sd->sdg"), (w_gate, "dg->sdg")])?;
    let gate = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, gate_product, "sdg->sdg", "sg->sdg")?;
    let up_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(normed2, "sd->sdg"), (w_up, "dg->sdg")])?;
    let up = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, up_product, "sdg->sdg", "sg->sdg")?;

    let neg_gate = elementwise(program, DType::Float32, ScalarOp::Negate, &[(gate, "sg->sg")])?;
    let exp_neg_gate = elementwise(program, DType::Float32, ScalarOp::Exponential, &[(neg_gate, "sg->sg")])?;
    let one_plus_exp = elementwise(program, DType::Float32, ScalarOp::Add, &[(exp_neg_gate, "sg->sg"), (ones, "->sg")])?;
    let sigmoid_gate = elementwise(program, DType::Float32, ScalarOp::Reciprocal, &[(one_plus_exp, "sg->sg")])?;
    let silu_gate = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(gate, "sg->sg"), (sigmoid_gate, "sg->sg")])?;
    let ffn_hidden = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(silu_gate, "sg->sg"), (up, "sg->sg")])?;

    let down_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(ffn_hidden, "sg->sgd"), (w_down, "gd->sgd")])?;
    let ffn_out = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, down_product, "sgd->sgd", "sd->sgd")?;

    elementwise(program, DType::Float32, ScalarOp::Add, &[(ffn_out, "sd->sd"), (residual1, "sd->sd")])
}

/// Rust-code counterpart of `specs/moe_block.toml`'s `ffn_product` node:
/// gathers `stack[route[s], :, :]` (`stack` is a `[expert_count, d_in,
/// d_out]` weight slab) and multiplies it elementwise against `x`'s `[s,
/// d_in]`, broadcast over the `d_out` axis, ready for a later [`reduce`]
/// over `d_in` to finish the matmul. The same [`IndexMap::Computed`] gather
/// [`embedding_lookup`] uses, with one extra non-gathered axis (`d_out`)
/// spliced in after the gathered one instead of none.
fn gathered_expert_product(program: &mut Vec<Op>, stack: NodeId, route: NodeId, x: NodeId) -> NodeId {
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
                },
                AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(2)).collect(),
                    offset: 0,
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
/// key, so [`append_mistral_moe_layer`]/[`append_mistral_cached_moe_layer`]
/// always pass `Softmax` unconditionally rather than reading a key that
/// does not exist on that checkpoint. `Sigmoid` is `_TYPE_SIGMOID` (`2`),
/// LFM2's own value (`transformers/models/lfm2_moe/modeling_lfm2_moe.py:209`'s
/// `router_logits.sigmoid()`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertGatingFunc {
    Softmax,
    Sigmoid,
}

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
#[allow(clippy::too_many_arguments)]
fn append_moe_ffn(
    program: &mut Vec<Op>,
    x: NodeId,
    gate_inp: NodeId,
    expert_w_gate: NodeId,
    expert_w_up: NodeId,
    expert_w_down: NodeId,
    expert_count: u32,
    expert_used_count: u32,
    ones: NodeId,
    gating: ExpertGatingFunc,
) -> Result<NodeId, TensorError> {
    if expert_used_count == 0 || expert_used_count > expert_count {
        return Err(TensorError::InvalidExpertConfig {
            expert_count,
            expert_used_count,
        });
    }

    let gate_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(x, "sd->sde"), (gate_inp, "de->sde")])?;
    let logits = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, gate_product, "sde->sde", "se->sde")?;

    let scores = match gating {
        ExpertGatingFunc::Softmax => logits,
        ExpertGatingFunc::Sigmoid => {
            let neg_logits = elementwise(program, DType::Float32, ScalarOp::Negate, &[(logits, "se->se")])?;
            let exp_neg_logits = elementwise(program, DType::Float32, ScalarOp::Exponential, &[(neg_logits, "se->se")])?;
            let one_plus_exp = elementwise(program, DType::Float32, ScalarOp::Add, &[(exp_neg_logits, "se->se"), (ones, "->se")])?;
            elementwise(program, DType::Float32, ScalarOp::Reciprocal, &[(one_plus_exp, "se->se")])?
        }
    };
    let mut selection_scores = scores;

    let expert_index = op::append(program, Op::Iota { dtype: DType::Float32, extent: Extent::Static(expert_count) });
    let neg_infinity = scalar_constant(program, f32::NEG_INFINITY);

    let mut weighted_sum: Option<NodeId> = None;
    let mut weight_total: Option<NodeId> = None;
    let mut max_selection_0: Option<NodeId> = None;

    for round in 0..expert_used_count {
        let max_selection = reduce(program, DType::Float32, ScalarOp::Maximum, ReduceInit::NegativeInfinity, selection_scores, "se->se", "s->se")?;
        let mask = elementwise(program, DType::Float32, ScalarOp::Equal, &[(selection_scores, "se->se"), (max_selection, "s->se")])?;
        let candidate = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(mask, "se->se"), (expert_index, "e->se")])?;
        let route = reduce(program, DType::Int32, ScalarOp::Maximum, ReduceInit::Zero, candidate, "se->se", "s->se")?;

        let weight = match gating {
            ExpertGatingFunc::Softmax => {
                let first_max = *max_selection_0.get_or_insert(max_selection);
                let shifted = elementwise(program, DType::Float32, ScalarOp::Subtract, &[(max_selection, "s->s"), (first_max, "s->s")])?;
                elementwise(program, DType::Float32, ScalarOp::Exponential, &[(shifted, "s->s")])?
            }
            ExpertGatingFunc::Sigmoid => {
                // unbiased `scores` at the masked (selected) position, never
                // `max_selection` itself -- that would be the biased score.
                let masked_scores = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(mask, "se->se"), (scores, "se->se")])?;
                reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, masked_scores, "se->se", "s->se")?
            }
        };

        let gate_expert_product = gathered_expert_product(program, expert_w_gate, route, x);
        let gate_expert = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, gate_expert_product, "sio->sio", "so->sio")?;
        let up_expert_product = gathered_expert_product(program, expert_w_up, route, x);
        let up_expert = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, up_expert_product, "sio->sio", "so->sio")?;

        let neg_gate = elementwise(program, DType::Float32, ScalarOp::Negate, &[(gate_expert, "sg->sg")])?;
        let exp_neg_gate = elementwise(program, DType::Float32, ScalarOp::Exponential, &[(neg_gate, "sg->sg")])?;
        let one_plus_exp = elementwise(program, DType::Float32, ScalarOp::Add, &[(exp_neg_gate, "sg->sg"), (ones, "->sg")])?;
        let sigmoid_gate = elementwise(program, DType::Float32, ScalarOp::Reciprocal, &[(one_plus_exp, "sg->sg")])?;
        let silu_gate = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(gate_expert, "sg->sg"), (sigmoid_gate, "sg->sg")])?;
        let ffn_hidden = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(silu_gate, "sg->sg"), (up_expert, "sg->sg")])?;

        let down_expert_product = gathered_expert_product(program, expert_w_down, route, ffn_hidden);
        let round_ffn = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, down_expert_product, "sio->sio", "so->sio")?;

        let weighted_round = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(round_ffn, "sd->sd"), (weight, "s->sd")])?;
        weighted_sum = Some(match weighted_sum {
            Some(accum) => elementwise(program, DType::Float32, ScalarOp::Add, &[(accum, "sd->sd"), (weighted_round, "sd->sd")])?,
            None => weighted_round,
        });
        weight_total = Some(match weight_total {
            Some(accum) => elementwise(program, DType::Float32, ScalarOp::Add, &[(accum, "s->s"), (weight, "s->s")])?,
            None => weight,
        });

        if round + 1 < expert_used_count {
            selection_scores = elementwise(program, DType::Float32, ScalarOp::Select, &[(mask, "se->se"), (neg_infinity, "->se"), (selection_scores, "se->se")])?;
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
    let inv_weight_total = elementwise(program, DType::Float32, ScalarOp::Reciprocal, &[(weight_total, "s->s")])?;
    elementwise(program, DType::Float32, ScalarOp::Multiply, &[(weighted_sum, "sd->sd"), (inv_weight_total, "s->sd")])
}

/// [`append_mistral_layer`]'s mixture-of-experts counterpart: identical
/// attention block (RoPE + GQA + causal mask, node-for-node the same code),
/// [`append_moe_ffn`] in place of the dense SwiGLU triple. Kept as a
/// separate function rather than a branch inside [`append_mistral_layer`]
/// so the dense path's own node sequence — and therefore its generated
/// program bytes — never changes shape by so much as one node merely
/// because this function exists next to it.
#[allow(clippy::too_many_arguments)]
fn append_mistral_moe_layer(
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
    gate_inp: NodeId,
    expert_w_gate: NodeId,
    expert_w_up: NodeId,
    expert_w_down: NodeId,
    expert_count: u32,
    expert_used_count: u32,
) -> Result<NodeId, TensorError> {
    let normed = rmsnorm(program, x, attn_norm_weight, inv_dim, eps)?;

    let q_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(normed, "si->shdi"), (wq, "ihd->shdi")])?;
    let q = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, q_product, "shdi->shdi", "shd->shdi")?;

    let k_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(normed, "si->sudi"), (wk, "iud->sudi")])?;
    let k = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, k_product, "sudi->sudi", "sud->sudi")?;

    let v_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(normed, "si->sudi"), (wv, "iud->sudi")])?;
    let v = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, v_product, "sudi->sudi", "sud->sudi")?;

    let q_even_cos = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q, "s,h,2*i->shi"), (cos, "si->shi")])?;
    let q_odd_sin = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q, "s,h,2*i+1->shi"), (sin, "si->shi")])?;
    let rotated_q_even = elementwise(program, DType::Float32, ScalarOp::Subtract, &[(q_even_cos, "shi->shi"), (q_odd_sin, "shi->shi")])?;
    let q_even_sin = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q, "s,h,2*i->shi"), (sin, "si->shi")])?;
    let q_odd_cos = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q, "s,h,2*i+1->shi"), (cos, "si->shi")])?;
    let rotated_q_odd = elementwise(program, DType::Float32, ScalarOp::Add, &[(q_even_sin, "shi->shi"), (q_odd_cos, "shi->shi")])?;

    let k_even_cos = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(k, "s,u,2*i->sui"), (cos, "si->sui")])?;
    let k_odd_sin = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(k, "s,u,2*i+1->sui"), (sin, "si->sui")])?;
    let rotated_k_even = elementwise(program, DType::Float32, ScalarOp::Subtract, &[(k_even_cos, "sui->sui"), (k_odd_sin, "sui->sui")])?;
    let k_even_sin = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(k, "s,u,2*i->sui"), (sin, "si->sui")])?;
    let k_odd_cos = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(k, "s,u,2*i+1->sui"), (cos, "si->sui")])?;
    let rotated_k_odd = elementwise(program, DType::Float32, ScalarOp::Add, &[(k_even_sin, "sui->sui"), (k_odd_cos, "sui->sui")])?;

    let group_map = alloc::format!("s,{group}*u+g,i->sugi");
    let q_even_grouped = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(rotated_q_even, group_map.as_str()), (group_ones, "ug->sugi")])?;
    let q_odd_grouped = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(rotated_q_odd, group_map.as_str()), (group_ones, "ug->sugi")])?;

    let score_even_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q_even_grouped, "sugi->stugi"), (rotated_k_even, "tui->stugi")])?;
    let score_even = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, score_even_product, "stugi->stugi", "stug->stugi")?;
    let score_odd_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q_odd_grouped, "sugi->stugi"), (rotated_k_odd, "tui->stugi")])?;
    let score_odd = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, score_odd_product, "stugi->stugi", "stug->stugi")?;
    let scores = elementwise(program, DType::Float32, ScalarOp::Add, &[(score_even, "stug->stug"), (score_odd, "stug->stug")])?;

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
        &[(is_future, "st->stug"), (neg_infinity, "->stug"), (scores_scaled, "stug->stug")],
    )?;

    let score_max = reduce(program, DType::Float32, ScalarOp::Maximum, ReduceInit::NegativeInfinity, scores_masked, "stug->stug", "sug->stug")?;
    let shifted = elementwise(program, DType::Float32, ScalarOp::Subtract, &[(scores_masked, "stug->stug"), (score_max, "sug->stug")])?;
    let weights = elementwise(program, DType::Float32, ScalarOp::Exponential, &[(shifted, "stug->stug")])?;
    let weight_sum = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, weights, "stug->stug", "sug->stug")?;
    let inv_weight_sum = elementwise(program, DType::Float32, ScalarOp::Reciprocal, &[(weight_sum, "sug->sug")])?;
    let probabilities = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(weights, "stug->stug"), (inv_weight_sum, "sug->stug")])?;

    let attended_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(probabilities, "stug->stugd"), (v, "tud->stugd")])?;
    let attended = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, attended_product, "stugd->stugd", "sugd->stugd")?;

    let wo_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(attended, "sugd->sugdo"), (wo, "ugdo->sugdo")])?;
    let attn_out = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, wo_product, "sugdo->sugdo", "so->sugdo")?;

    let residual1 = elementwise(program, DType::Float32, ScalarOp::Add, &[(attn_out, "sd->sd"), (x, "sd->sd")])?;

    let normed2 = rmsnorm(program, residual1, ffn_norm_weight, inv_dim, eps)?;

    let ffn_out = append_moe_ffn(
        program,
        normed2,
        gate_inp,
        expert_w_gate,
        expert_w_up,
        expert_w_down,
        expert_count,
        expert_used_count,
        ones,
        ExpertGatingFunc::Softmax,
    )?;

    elementwise(program, DType::Float32, ScalarOp::Add, &[(ffn_out, "sd->sd"), (residual1, "sd->sd")])
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
/// [`append_mistral_layer`]'s plain `ffn_{gate,up,down}.weight` triple,
/// node-for-node the same program this function has always built, so a
/// dense checkpoint's generated program (and therefore its output) is
/// unaffected by this parameter's existence. `expert_count > 0` routes each
/// layer through [`append_mistral_moe_layer`] instead, gathering one of
/// `expert_count` experts' weight slabs per token per
/// [`append_moe_ffn`]'s doc.
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
            alloc::vec![Extent::Static(embedding), Extent::Static(query_heads), Extent::Static(head_dim)],
            &alloc::format!("blk.{layer}.attn_q.weight"),
        );
        let wk = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding), Extent::Static(kv_heads), Extent::Static(head_dim)],
            &alloc::format!("blk.{layer}.attn_k.weight"),
        );
        let wv = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding), Extent::Static(kv_heads), Extent::Static(head_dim)],
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
                alloc::vec![Extent::Static(expert_count), Extent::Static(embedding), Extent::Static(feed_forward)],
                &alloc::format!("blk.{layer}.ffn_gate_exps.weight"),
            );
            let expert_w_up = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(expert_count), Extent::Static(embedding), Extent::Static(feed_forward)],
                &alloc::format!("blk.{layer}.ffn_up_exps.weight"),
            );
            let expert_w_down = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(expert_count), Extent::Static(feed_forward), Extent::Static(embedding)],
                &alloc::format!("blk.{layer}.ffn_down_exps.weight"),
            );

            append_mistral_moe_layer(
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
                gate_inp,
                expert_w_gate,
                expert_w_up,
                expert_w_down,
                expert_count,
                expert_used_count,
            )?
        };
    }

    let output_norm_weight = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(embedding)], "output_norm.weight");
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
    reduce(&mut program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, logits_product, "sdv->sdv", "sv->sdv")?;

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
#[allow(clippy::too_many_arguments)]
fn append_mistral_cached_layer(
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
) -> Result<(NodeId, CachedLayerRoots), TensorError> {
    let normed = rmsnorm(program, x, attn_norm_weight, inv_dim, eps)?;

    let q_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(normed, "si->shdi"), (wq, "ihd->shdi")])?;
    let q = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, q_product, "shdi->shdi", "shd->shdi")?;

    let k_new_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(normed, "si->sudi"), (wk, "iud->sudi")])?;
    let k_new = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, k_new_product, "sudi->sudi", "sud->sudi")?;

    let v_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(normed, "si->sudi"), (wv, "iud->sudi")])?;
    let v_new = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, v_product, "sudi->sudi", "sud->sudi")?;

    let q_even_cos = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q, "s,h,2*i->shi"), (cos_new, "si->shi")])?;
    let q_odd_sin = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q, "s,h,2*i+1->shi"), (sin_new, "si->shi")])?;
    let rotated_q_even = elementwise(program, DType::Float32, ScalarOp::Subtract, &[(q_even_cos, "shi->shi"), (q_odd_sin, "shi->shi")])?;
    let q_even_sin = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q, "s,h,2*i->shi"), (sin_new, "si->shi")])?;
    let q_odd_cos = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q, "s,h,2*i+1->shi"), (cos_new, "si->shi")])?;
    let rotated_q_odd = elementwise(program, DType::Float32, ScalarOp::Add, &[(q_even_sin, "shi->shi"), (q_odd_cos, "shi->shi")])?;

    let k_new_even_cos = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(k_new, "s,u,2*i->sui"), (cos_new, "si->sui")])?;
    let k_new_odd_sin = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(k_new, "s,u,2*i+1->sui"), (sin_new, "si->sui")])?;
    let rotated_k_new_even =
        elementwise(program, DType::Float32, ScalarOp::Subtract, &[(k_new_even_cos, "sui->sui"), (k_new_odd_sin, "sui->sui")])?;
    let k_new_even_sin = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(k_new, "s,u,2*i->sui"), (sin_new, "si->sui")])?;
    let k_new_odd_cos = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(k_new, "s,u,2*i+1->sui"), (cos_new, "si->sui")])?;
    let rotated_k_new_odd =
        elementwise(program, DType::Float32, ScalarOp::Add, &[(k_new_even_sin, "sui->sui"), (k_new_odd_cos, "sui->sui")])?;

    let group_map = alloc::format!("s,{group}*u+g,i->sugi");
    let q_even_grouped = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(rotated_q_even, group_map.as_str()), (group_ones, "ug->sugi")])?;
    let q_odd_grouped = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(rotated_q_odd, group_map.as_str()), (group_ones, "ug->sugi")])?;

    // cached block: query `s` against every already-rotated cached key `t`
    // (symbol 1's extent, zero on the very first call) -- never masked, a
    // cached position is always in the past of a new query.
    let score_cached_even_product =
        elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q_even_grouped, "sugi->stugi"), (k_even_cache, "tui->stugi")])?;
    let score_cached_even = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, score_cached_even_product, "stugi->stugi", "stug->stugi")?;
    let score_cached_odd_product =
        elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q_odd_grouped, "sugi->stugi"), (k_odd_cache, "tui->stugi")])?;
    let score_cached_odd = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, score_cached_odd_product, "stugi->stugi", "stug->stugi")?;
    let score_cached = elementwise(program, DType::Float32, ScalarOp::Add, &[(score_cached_even, "stug->stug"), (score_cached_odd, "stug->stug")])?;
    let score_cached_scaled = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(score_cached, "stug->stug"), (inv_sqrt_head_dim, "->stug")])?;

    // new block: query `s` against this call's own freshly rotated key `w`
    // (symbol 0's extent, same range as `s`) -- causal within the block,
    // reusing `is_future` unchanged since it is already `[s, w]`-shaped.
    let score_new_even_product =
        elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q_even_grouped, "sugi->swugi"), (rotated_k_new_even, "wui->swugi")])?;
    let score_new_even = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, score_new_even_product, "swugi->swugi", "swug->swugi")?;
    let score_new_odd_product =
        elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q_odd_grouped, "sugi->swugi"), (rotated_k_new_odd, "wui->swugi")])?;
    let score_new_odd = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, score_new_odd_product, "swugi->swugi", "swug->swugi")?;
    let score_new = elementwise(program, DType::Float32, ScalarOp::Add, &[(score_new_even, "swug->swug"), (score_new_odd, "swug->swug")])?;
    let score_new_scaled = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(score_new, "swug->swug"), (inv_sqrt_head_dim, "->swug")])?;
    let neg_infinity = scalar_constant(program, f32::NEG_INFINITY);
    let score_new_masked = elementwise(
        program,
        DType::Float32,
        ScalarOp::Select,
        &[(is_future, "sw->swug"), (neg_infinity, "->swug"), (score_new_scaled, "swug->swug")],
    )?;

    // online-softmax combine: two disjoint key ranges, one shared max and
    // one shared normalizer, no literal concatenation anywhere.
    let score_max_cached = reduce(program, DType::Float32, ScalarOp::Maximum, ReduceInit::NegativeInfinity, score_cached_scaled, "stug->stug", "sug->stug")?;
    let score_max_new = reduce(program, DType::Float32, ScalarOp::Maximum, ReduceInit::NegativeInfinity, score_new_masked, "swug->swug", "sug->swug")?;
    let global_max = elementwise(program, DType::Float32, ScalarOp::Maximum, &[(score_max_cached, "sug->sug"), (score_max_new, "sug->sug")])?;

    let shifted_cached = elementwise(program, DType::Float32, ScalarOp::Subtract, &[(score_cached_scaled, "stug->stug"), (global_max, "sug->stug")])?;
    let weights_cached = elementwise(program, DType::Float32, ScalarOp::Exponential, &[(shifted_cached, "stug->stug")])?;
    let shifted_new = elementwise(program, DType::Float32, ScalarOp::Subtract, &[(score_new_masked, "swug->swug"), (global_max, "sug->swug")])?;
    let weights_new = elementwise(program, DType::Float32, ScalarOp::Exponential, &[(shifted_new, "swug->swug")])?;

    let sum_cached = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, weights_cached, "stug->stug", "sug->stug")?;
    let sum_new = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, weights_new, "swug->swug", "sug->swug")?;
    let weight_sum = elementwise(program, DType::Float32, ScalarOp::Add, &[(sum_cached, "sug->sug"), (sum_new, "sug->sug")])?;
    let inv_weight_sum = elementwise(program, DType::Float32, ScalarOp::Reciprocal, &[(weight_sum, "sug->sug")])?;

    let attended_cached_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(weights_cached, "stug->stugd"), (v_cache, "tud->stugd")])?;
    let attended_cached = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, attended_cached_product, "stugd->stugd", "sugd->stugd")?;
    let attended_new_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(weights_new, "swug->swugd"), (v_new, "wud->swugd")])?;
    let attended_new = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, attended_new_product, "swugd->swugd", "sugd->swugd")?;
    let attended_sum = elementwise(program, DType::Float32, ScalarOp::Add, &[(attended_cached, "sugd->sugd"), (attended_new, "sugd->sugd")])?;
    let attended = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(attended_sum, "sugd->sugd"), (inv_weight_sum, "sug->sugd")])?;

    let wo_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(attended, "sugd->sugdo"), (wo, "ugdo->sugdo")])?;
    let attn_out = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, wo_product, "sugdo->sugdo", "so->sugdo")?;

    let residual1 = elementwise(program, DType::Float32, ScalarOp::Add, &[(attn_out, "sd->sd"), (x, "sd->sd")])?;

    let normed2 = rmsnorm(program, residual1, ffn_norm_weight, inv_dim, eps)?;

    let gate_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(normed2, "sd->sdg"), (w_gate, "dg->sdg")])?;
    let gate = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, gate_product, "sdg->sdg", "sg->sdg")?;
    let up_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(normed2, "sd->sdg"), (w_up, "dg->sdg")])?;
    let up = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, up_product, "sdg->sdg", "sg->sdg")?;

    let neg_gate = elementwise(program, DType::Float32, ScalarOp::Negate, &[(gate, "sg->sg")])?;
    let exp_neg_gate = elementwise(program, DType::Float32, ScalarOp::Exponential, &[(neg_gate, "sg->sg")])?;
    let one_plus_exp = elementwise(program, DType::Float32, ScalarOp::Add, &[(exp_neg_gate, "sg->sg"), (ones, "->sg")])?;
    let sigmoid_gate = elementwise(program, DType::Float32, ScalarOp::Reciprocal, &[(one_plus_exp, "sg->sg")])?;
    let silu_gate = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(gate, "sg->sg"), (sigmoid_gate, "sg->sg")])?;
    let ffn_hidden = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(silu_gate, "sg->sg"), (up, "sg->sg")])?;

    let down_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(ffn_hidden, "sg->sgd"), (w_down, "gd->sgd")])?;
    let ffn_out = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, down_product, "sgd->sgd", "sd->sgd")?;

    let x_next = elementwise(program, DType::Float32, ScalarOp::Add, &[(ffn_out, "sd->sd"), (residual1, "sd->sd")])?;

    Ok((x_next, (rotated_k_new_even, rotated_k_new_odd, v_new)))
}

/// [`append_mistral_cached_layer`]'s mixture-of-experts counterpart, the
/// same relationship [`append_mistral_moe_layer`] bears to
/// [`append_mistral_layer`]: identical cached attention block (RoPE + GQA +
/// online-softmax combine over the cached/new key split, node-for-node the
/// same code as [`append_mistral_cached_layer`]), [`append_moe_ffn`] in
/// place of the dense SwiGLU triple. Kept as a separate function for the
/// same reason [`append_mistral_moe_layer`] is: the dense cached path's own
/// node sequence never changes shape merely because this function exists
/// next to it.
#[allow(clippy::too_many_arguments)]
fn append_mistral_cached_moe_layer(
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
) -> Result<(NodeId, CachedLayerRoots), TensorError> {
    let normed = rmsnorm(program, x, attn_norm_weight, inv_dim, eps)?;

    let q_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(normed, "si->shdi"), (wq, "ihd->shdi")])?;
    let q = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, q_product, "shdi->shdi", "shd->shdi")?;

    let k_new_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(normed, "si->sudi"), (wk, "iud->sudi")])?;
    let k_new = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, k_new_product, "sudi->sudi", "sud->sudi")?;

    let v_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(normed, "si->sudi"), (wv, "iud->sudi")])?;
    let v_new = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, v_product, "sudi->sudi", "sud->sudi")?;

    let q_even_cos = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q, "s,h,2*i->shi"), (cos_new, "si->shi")])?;
    let q_odd_sin = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q, "s,h,2*i+1->shi"), (sin_new, "si->shi")])?;
    let rotated_q_even = elementwise(program, DType::Float32, ScalarOp::Subtract, &[(q_even_cos, "shi->shi"), (q_odd_sin, "shi->shi")])?;
    let q_even_sin = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q, "s,h,2*i->shi"), (sin_new, "si->shi")])?;
    let q_odd_cos = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q, "s,h,2*i+1->shi"), (cos_new, "si->shi")])?;
    let rotated_q_odd = elementwise(program, DType::Float32, ScalarOp::Add, &[(q_even_sin, "shi->shi"), (q_odd_cos, "shi->shi")])?;

    let k_new_even_cos = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(k_new, "s,u,2*i->sui"), (cos_new, "si->sui")])?;
    let k_new_odd_sin = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(k_new, "s,u,2*i+1->sui"), (sin_new, "si->sui")])?;
    let rotated_k_new_even =
        elementwise(program, DType::Float32, ScalarOp::Subtract, &[(k_new_even_cos, "sui->sui"), (k_new_odd_sin, "sui->sui")])?;
    let k_new_even_sin = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(k_new, "s,u,2*i->sui"), (sin_new, "si->sui")])?;
    let k_new_odd_cos = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(k_new, "s,u,2*i+1->sui"), (cos_new, "si->sui")])?;
    let rotated_k_new_odd =
        elementwise(program, DType::Float32, ScalarOp::Add, &[(k_new_even_sin, "sui->sui"), (k_new_odd_cos, "sui->sui")])?;

    let group_map = alloc::format!("s,{group}*u+g,i->sugi");
    let q_even_grouped = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(rotated_q_even, group_map.as_str()), (group_ones, "ug->sugi")])?;
    let q_odd_grouped = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(rotated_q_odd, group_map.as_str()), (group_ones, "ug->sugi")])?;

    let score_cached_even_product =
        elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q_even_grouped, "sugi->stugi"), (k_even_cache, "tui->stugi")])?;
    let score_cached_even = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, score_cached_even_product, "stugi->stugi", "stug->stugi")?;
    let score_cached_odd_product =
        elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q_odd_grouped, "sugi->stugi"), (k_odd_cache, "tui->stugi")])?;
    let score_cached_odd = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, score_cached_odd_product, "stugi->stugi", "stug->stugi")?;
    let score_cached = elementwise(program, DType::Float32, ScalarOp::Add, &[(score_cached_even, "stug->stug"), (score_cached_odd, "stug->stug")])?;
    let score_cached_scaled = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(score_cached, "stug->stug"), (inv_sqrt_head_dim, "->stug")])?;

    let score_new_even_product =
        elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q_even_grouped, "sugi->swugi"), (rotated_k_new_even, "wui->swugi")])?;
    let score_new_even = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, score_new_even_product, "swugi->swugi", "swug->swugi")?;
    let score_new_odd_product =
        elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q_odd_grouped, "sugi->swugi"), (rotated_k_new_odd, "wui->swugi")])?;
    let score_new_odd = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, score_new_odd_product, "swugi->swugi", "swug->swugi")?;
    let score_new = elementwise(program, DType::Float32, ScalarOp::Add, &[(score_new_even, "swug->swug"), (score_new_odd, "swug->swug")])?;
    let score_new_scaled = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(score_new, "swug->swug"), (inv_sqrt_head_dim, "->swug")])?;
    let neg_infinity = scalar_constant(program, f32::NEG_INFINITY);
    let score_new_masked = elementwise(
        program,
        DType::Float32,
        ScalarOp::Select,
        &[(is_future, "sw->swug"), (neg_infinity, "->swug"), (score_new_scaled, "swug->swug")],
    )?;

    let score_max_cached = reduce(program, DType::Float32, ScalarOp::Maximum, ReduceInit::NegativeInfinity, score_cached_scaled, "stug->stug", "sug->stug")?;
    let score_max_new = reduce(program, DType::Float32, ScalarOp::Maximum, ReduceInit::NegativeInfinity, score_new_masked, "swug->swug", "sug->swug")?;
    let global_max = elementwise(program, DType::Float32, ScalarOp::Maximum, &[(score_max_cached, "sug->sug"), (score_max_new, "sug->sug")])?;

    let shifted_cached = elementwise(program, DType::Float32, ScalarOp::Subtract, &[(score_cached_scaled, "stug->stug"), (global_max, "sug->stug")])?;
    let weights_cached = elementwise(program, DType::Float32, ScalarOp::Exponential, &[(shifted_cached, "stug->stug")])?;
    let shifted_new = elementwise(program, DType::Float32, ScalarOp::Subtract, &[(score_new_masked, "swug->swug"), (global_max, "sug->swug")])?;
    let weights_new = elementwise(program, DType::Float32, ScalarOp::Exponential, &[(shifted_new, "swug->swug")])?;

    let sum_cached = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, weights_cached, "stug->stug", "sug->stug")?;
    let sum_new = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, weights_new, "swug->swug", "sug->swug")?;
    let weight_sum = elementwise(program, DType::Float32, ScalarOp::Add, &[(sum_cached, "sug->sug"), (sum_new, "sug->sug")])?;
    let inv_weight_sum = elementwise(program, DType::Float32, ScalarOp::Reciprocal, &[(weight_sum, "sug->sug")])?;

    let attended_cached_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(weights_cached, "stug->stugd"), (v_cache, "tud->stugd")])?;
    let attended_cached = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, attended_cached_product, "stugd->stugd", "sugd->stugd")?;
    let attended_new_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(weights_new, "swug->swugd"), (v_new, "wud->swugd")])?;
    let attended_new = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, attended_new_product, "swugd->swugd", "sugd->swugd")?;
    let attended_sum = elementwise(program, DType::Float32, ScalarOp::Add, &[(attended_cached, "sugd->sugd"), (attended_new, "sugd->sugd")])?;
    let attended = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(attended_sum, "sugd->sugd"), (inv_weight_sum, "sug->sugd")])?;

    let wo_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(attended, "sugd->sugdo"), (wo, "ugdo->sugdo")])?;
    let attn_out = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, wo_product, "sugdo->sugdo", "so->sugdo")?;

    let residual1 = elementwise(program, DType::Float32, ScalarOp::Add, &[(attn_out, "sd->sd"), (x, "sd->sd")])?;

    let normed2 = rmsnorm(program, residual1, ffn_norm_weight, inv_dim, eps)?;

    let ffn_out = append_moe_ffn(
        program,
        normed2,
        gate_inp,
        expert_w_gate,
        expert_w_up,
        expert_w_down,
        expert_count,
        expert_used_count,
        ones,
        ExpertGatingFunc::Softmax,
    )?;

    let x_next = elementwise(program, DType::Float32, ScalarOp::Add, &[(ffn_out, "sd->sd"), (residual1, "sd->sd")])?;

    Ok((x_next, (rotated_k_new_even, rotated_k_new_odd, v_new)))
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
/// (`s-(l_cache-1)+l`) fails [`shape::bounds_check`] globally, because an
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
fn causal_conv1d(program: &mut Vec<Op>, x: NodeId, weight: NodeId, l_cache: u32) -> Result<NodeId, TensorError> {
    if l_cache == 0 {
        return Err(TensorError::InvalidConvConfig { l_cache });
    }

    let sequence_index = op::append(program, Op::Iota { dtype: DType::Float32, extent: Extent::Symbolic(0) });
    let tap_index = op::append(program, Op::Iota { dtype: DType::Float32, extent: Extent::Static(l_cache) });
    let window_offset = scalar_constant(program, -((l_cache - 1) as f32));

    let sequence_plus_tap = elementwise(program, DType::Float32, ScalarOp::Add, &[(sequence_index, "s->sl"), (tap_index, "l->sl")])?;
    let raw_position = elementwise(program, DType::Float32, ScalarOp::Add, &[(sequence_plus_tap, "sl->sl"), (window_offset, "->sl")])?;

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
    let candidate_axis = op::append(program, Op::Iota { dtype: DType::Float32, extent: Extent::Static(2) });
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
    let is_raw_slot = elementwise(program, DType::Float32, ScalarOp::Equal, &[(candidate_axis, "c->slc"), (zero_wide, "sl->slc")])?;
    let candidate = elementwise(
        program,
        DType::Float32,
        ScalarOp::Select,
        &[(is_raw_slot, "slc->slc"), (raw_position, "sl->slc"), (zero, "->slc")],
    )?;
    let clamped_position = reduce(program, DType::Int32, ScalarOp::Maximum, ReduceInit::NegativeInfinity, candidate, "slc->slc", "sl->slc")?;

    let negative_one = scalar_constant(program, -1.0);
    let is_valid = elementwise(program, DType::Float32, ScalarOp::Greater, &[(raw_position, "sl->sl"), (negative_one, "->sl")])?;

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

    let tap_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(windowed, "sld->sld"), (weight, "ld->sld")])?;
    let zero_tap = scalar_constant(program, 0.0);
    let masked_tap = elementwise(
        program,
        DType::Float32,
        ScalarOp::Select,
        &[(is_valid, "sl->sld"), (tap_product, "sld->sld"), (zero_tap, "->sld")],
    )?;

    reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, masked_tap, "sld->sld", "sd->sld")
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
fn append_lfm2_conv_mixer(
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

    let branch_b_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(normed, "sd->sdg"), (b_proj, "dg->sdg")])?;
    let branch_b = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, branch_b_product, "sdg->sdg", "sg->sdg")?;

    let branch_x_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(normed, "sd->sdg"), (x_proj, "dg->sdg")])?;
    let branch_x = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, branch_x_product, "sdg->sdg", "sg->sdg")?;

    let branch_c_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(normed, "sd->sdg"), (c_proj, "dg->sdg")])?;
    let branch_c = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, branch_c_product, "sdg->sdg", "sg->sdg")?;

    let gated_input = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(branch_b, "sg->sg"), (branch_x, "sg->sg")])?;
    let convolved = causal_conv1d(program, gated_input, conv_weight, l_cache)?;
    let gated_output = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(convolved, "sg->sg"), (branch_c, "sg->sg")])?;

    let out_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(gated_output, "sd->sdo"), (out_proj, "do->sdo")])?;
    let mixer_out = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, out_product, "sdo->sdo", "so->sdo")?;

    elementwise(program, DType::Float32, ScalarOp::Add, &[(mixer_out, "sd->sd"), (x, "sd->sd")])
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
fn append_attention_mixer(
    program: &mut Vec<Op>,
    x: NodeId,
    inv_dim: NodeId,
    eps: NodeId,
    inv_sqrt_head_dim: NodeId,
    cos: NodeId,
    sin: NodeId,
    group_ones: NodeId,
    is_future: NodeId,
    neg_infinity: NodeId,
    group: u32,
    attn_norm_weight: NodeId,
    wq: NodeId,
    wk: NodeId,
    wv: NodeId,
    wo: NodeId,
) -> Result<NodeId, TensorError> {
    let normed = rmsnorm(program, x, attn_norm_weight, inv_dim, eps)?;

    let q_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(normed, "si->shdi"), (wq, "ihd->shdi")])?;
    let q = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, q_product, "shdi->shdi", "shd->shdi")?;

    let k_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(normed, "si->sudi"), (wk, "iud->sudi")])?;
    let k = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, k_product, "sudi->sudi", "sud->sudi")?;

    let v_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(normed, "si->sudi"), (wv, "iud->sudi")])?;
    let v = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, v_product, "sudi->sudi", "sud->sudi")?;

    let q_even_cos = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q, "s,h,2*i->shi"), (cos, "si->shi")])?;
    let q_odd_sin = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q, "s,h,2*i+1->shi"), (sin, "si->shi")])?;
    let rotated_q_even = elementwise(program, DType::Float32, ScalarOp::Subtract, &[(q_even_cos, "shi->shi"), (q_odd_sin, "shi->shi")])?;
    let q_even_sin = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q, "s,h,2*i->shi"), (sin, "si->shi")])?;
    let q_odd_cos = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q, "s,h,2*i+1->shi"), (cos, "si->shi")])?;
    let rotated_q_odd = elementwise(program, DType::Float32, ScalarOp::Add, &[(q_even_sin, "shi->shi"), (q_odd_cos, "shi->shi")])?;

    let k_even_cos = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(k, "s,u,2*i->sui"), (cos, "si->sui")])?;
    let k_odd_sin = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(k, "s,u,2*i+1->sui"), (sin, "si->sui")])?;
    let rotated_k_even = elementwise(program, DType::Float32, ScalarOp::Subtract, &[(k_even_cos, "sui->sui"), (k_odd_sin, "sui->sui")])?;
    let k_even_sin = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(k, "s,u,2*i->sui"), (sin, "si->sui")])?;
    let k_odd_cos = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(k, "s,u,2*i+1->sui"), (cos, "si->sui")])?;
    let rotated_k_odd = elementwise(program, DType::Float32, ScalarOp::Add, &[(k_even_sin, "sui->sui"), (k_odd_cos, "sui->sui")])?;

    let group_map = alloc::format!("s,{group}*u+g,i->sugi");
    let q_even_grouped = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(rotated_q_even, group_map.as_str()), (group_ones, "ug->sugi")])?;
    let q_odd_grouped = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(rotated_q_odd, group_map.as_str()), (group_ones, "ug->sugi")])?;

    let score_even_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q_even_grouped, "sugi->stugi"), (rotated_k_even, "tui->stugi")])?;
    let score_even = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, score_even_product, "stugi->stugi", "stug->stugi")?;
    let score_odd_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(q_odd_grouped, "sugi->stugi"), (rotated_k_odd, "tui->stugi")])?;
    let score_odd = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, score_odd_product, "stugi->stugi", "stug->stugi")?;
    let scores = elementwise(program, DType::Float32, ScalarOp::Add, &[(score_even, "stug->stug"), (score_odd, "stug->stug")])?;

    let scores_scaled = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(scores, "stug->stug"), (inv_sqrt_head_dim, "->stug")])?;

    let scores_masked = elementwise(
        program,
        DType::Float32,
        ScalarOp::Select,
        &[(is_future, "st->stug"), (neg_infinity, "->stug"), (scores_scaled, "stug->stug")],
    )?;

    let score_max = reduce(program, DType::Float32, ScalarOp::Maximum, ReduceInit::NegativeInfinity, scores_masked, "stug->stug", "sug->stug")?;
    let shifted = elementwise(program, DType::Float32, ScalarOp::Subtract, &[(scores_masked, "stug->stug"), (score_max, "sug->stug")])?;
    let weights = elementwise(program, DType::Float32, ScalarOp::Exponential, &[(shifted, "stug->stug")])?;
    let weight_sum = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, weights, "stug->stug", "sug->stug")?;
    let inv_weight_sum = elementwise(program, DType::Float32, ScalarOp::Reciprocal, &[(weight_sum, "sug->sug")])?;
    let probabilities = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(weights, "stug->stug"), (inv_weight_sum, "sug->stug")])?;

    let attended_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(probabilities, "stug->stugd"), (v, "tud->stugd")])?;
    let attended = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, attended_product, "stugd->stugd", "sugd->stugd")?;

    let wo_product = elementwise(program, DType::Float32, ScalarOp::Multiply, &[(attended, "sugd->sugdo"), (wo, "ugdo->sugdo")])?;
    let attn_out = reduce(program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, wo_product, "sugdo->sugdo", "so->sugdo")?;

    elementwise(program, DType::Float32, ScalarOp::Add, &[(attn_out, "sd->sd"), (x, "sd->sd")])
}

/// LFM2.5-8B-A1B's hybrid forward pass: `block_count` blocks, each either
/// [`append_attention_mixer`] or [`append_lfm2_conv_mixer`] per its own
/// `layer_kinds[layer]` (derived by [`LayerKind::from_tensor_names`] from the
/// real checkpoint's tensor directory, since `layer_types` is not a metadata
/// key this architecture writes), then a shared RMSNorm and
/// [`append_moe_ffn`]/dense-triple FFN exactly like
/// [`mistral_forward_program_with_experts`]'s own MoE branch --
/// `leading_dense_block_count` (LFM2.5-8B-A1B: `2`) is threaded per layer
/// rather than a single crate-wide dense/MoE switch, since this checkpoint's
/// first two blocks are dense and the rest are routed.
///
/// Prefill-only: takes the whole prompt as one `[seq, embedding]` pass, the
/// same scope [`mistral_forward_program_with_experts`] has. A KV-cached and
/// conv-state-cached incremental counterpart (mirroring
/// [`mistral_cached_forward_program_with_experts`]) is a further step this
/// function's own doc does not claim -- [`causal_conv1d`]'s masked-gather
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
) -> Result<(Vec<Op>, NodeId), TensorError> {
    if layer_kinds.len() != block_count as usize {
        return Err(TensorError::LayerKindCountMismatch {
            expected: block_count,
            found: layer_kinds.len(),
        });
    }

    let group = query_heads / kv_heads;
    let pairs = head_dim / 2;

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
    let inv_sqrt_head_dim = scalar_constant(&mut program, 1.0 / (head_dim as f32).sqrt());
    let cos = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Symbolic(0), Extent::Static(pairs)], "rope_cos");
    let sin = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Symbolic(0), Extent::Static(pairs)], "rope_sin");
    let group_ones = op::append(
        &mut program,
        Op::Constant {
            dtype: DType::Float32,
            shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
            value: 1.0,
        },
    );
    let (is_future, neg_infinity) = causal_mask(&mut program)?;

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
                    alloc::vec![Extent::Static(embedding), Extent::Static(query_heads), Extent::Static(head_dim)],
                    &alloc::format!("blk.{layer}.attn_q.weight"),
                );
                let wk = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(embedding), Extent::Static(kv_heads), Extent::Static(head_dim)],
                    &alloc::format!("blk.{layer}.attn_k.weight"),
                );
                let wv = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(embedding), Extent::Static(kv_heads), Extent::Static(head_dim)],
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
                append_attention_mixer(
                    &mut program,
                    x,
                    inv_dim,
                    eps,
                    inv_sqrt_head_dim,
                    cos,
                    sin,
                    group_ones,
                    is_future,
                    neg_infinity,
                    group,
                    attn_norm_weight,
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
                let conv_weight = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(l_cache), Extent::Static(embedding)],
                    &alloc::format!("blk.{layer}.shortconv.conv.weight"),
                );
                let out_proj = input_leaf(
                    &mut program,
                    DType::Float32,
                    alloc::vec![Extent::Static(embedding), Extent::Static(embedding)],
                    &alloc::format!("blk.{layer}.shortconv.out_proj.weight"),
                );
                append_lfm2_conv_mixer(&mut program, x, inv_dim, eps, attn_norm_weight, b_proj, c_proj, x_proj, conv_weight, out_proj, l_cache)?
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
            let gate_product = elementwise(&mut program, DType::Float32, ScalarOp::Multiply, &[(normed2, "sd->sdg"), (w_gate, "dg->sdg")])?;
            let gate = reduce(&mut program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, gate_product, "sdg->sdg", "sg->sdg")?;
            let up_product = elementwise(&mut program, DType::Float32, ScalarOp::Multiply, &[(normed2, "sd->sdg"), (w_up, "dg->sdg")])?;
            let up = reduce(&mut program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, up_product, "sdg->sdg", "sg->sdg")?;

            let neg_gate = elementwise(&mut program, DType::Float32, ScalarOp::Negate, &[(gate, "sg->sg")])?;
            let exp_neg_gate = elementwise(&mut program, DType::Float32, ScalarOp::Exponential, &[(neg_gate, "sg->sg")])?;
            let one_plus_exp = elementwise(&mut program, DType::Float32, ScalarOp::Add, &[(exp_neg_gate, "sg->sg"), (ones, "->sg")])?;
            let sigmoid_gate = elementwise(&mut program, DType::Float32, ScalarOp::Reciprocal, &[(one_plus_exp, "sg->sg")])?;
            let silu_gate = elementwise(&mut program, DType::Float32, ScalarOp::Multiply, &[(gate, "sg->sg"), (sigmoid_gate, "sg->sg")])?;
            let ffn_hidden = elementwise(&mut program, DType::Float32, ScalarOp::Multiply, &[(silu_gate, "sg->sg"), (up, "sg->sg")])?;

            let down_product = elementwise(&mut program, DType::Float32, ScalarOp::Multiply, &[(ffn_hidden, "sg->sgd"), (w_down, "gd->sgd")])?;
            reduce(&mut program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, down_product, "sgd->sgd", "sd->sgd")?
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
                alloc::vec![Extent::Static(expert_count), Extent::Static(embedding), Extent::Static(expert_feed_forward)],
                &alloc::format!("blk.{layer}.ffn_gate_exps.weight"),
            );
            let expert_w_up = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(expert_count), Extent::Static(embedding), Extent::Static(expert_feed_forward)],
                &alloc::format!("blk.{layer}.ffn_up_exps.weight"),
            );
            let expert_w_down = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(expert_count), Extent::Static(expert_feed_forward), Extent::Static(embedding)],
                &alloc::format!("blk.{layer}.ffn_down_exps.weight"),
            );
            append_moe_ffn(
                &mut program,
                normed2,
                gate_inp,
                expert_w_gate,
                expert_w_up,
                expert_w_down,
                expert_count,
                expert_used_count,
                ones,
                ExpertGatingFunc::Sigmoid,
            )?
        };

        x = elementwise(&mut program, DType::Float32, ScalarOp::Add, &[(ffn_out, "sd->sd"), (post_mixer, "sd->sd")])?;
    }

    let output_norm_weight = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(embedding)], "output_norm.weight");
    let normed_final = rmsnorm(&mut program, x, output_norm_weight, inv_dim, eps)?;

    let lm_head = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(embedding), Extent::Static(vocab)], "output.weight");
    let logits_product = elementwise(&mut program, DType::Float32, ScalarOp::Multiply, &[(normed_final, "sd->sdv"), (lm_head, "dv->sdv")])?;
    let logits = reduce(&mut program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, logits_product, "sdv->sdv", "sv->sdv")?;

    Ok((program, logits))
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
/// Dense-only: always binds [`append_mistral_cached_layer`]'s plain
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
    mistral_cached_forward_program_with_experts(vocab, embedding, feed_forward, query_heads, kv_heads, head_dim, block_count, 0, 0)
}

/// [`mistral_cached_forward_program`]'s mixture-of-experts-capable
/// counterpart, carrying the same `expert_count`/`expert_used_count`
/// parameters [`mistral_forward_program`] already takes. `expert_count == 0`
/// binds every layer through [`append_mistral_cached_layer`]'s plain
/// `ffn_{gate,up,down}.weight` triple, node-for-node the same program
/// [`mistral_cached_forward_program`] has always built, so a dense
/// checkpoint's generated program is unaffected by this function's
/// existence. `expert_count > 0` routes each layer through
/// [`append_mistral_cached_moe_layer`] instead, gathering one of
/// `expert_count` experts' weight slabs per token per [`append_moe_ffn`]'s
/// doc -- the same routed FFN [`mistral_forward_program`]'s own MoE branch
/// already runs, reused rather than reconstructed.
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
) -> Result<(Vec<Op>, NodeId, Vec<CachedLayerRoots>), TensorError> {
    let group = query_heads / kv_heads;
    let pairs = head_dim / 2;

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
            shape: alloc::vec![Extent::Static(kv_heads), Extent::Static(group)],
            value: 1.0,
        },
    );
    let (is_future, _neg_infinity) = causal_mask(&mut program)?;

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
            alloc::vec![Extent::Static(embedding), Extent::Static(query_heads), Extent::Static(head_dim)],
            &alloc::format!("blk.{layer}.attn_q.weight"),
        );
        let wk = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding), Extent::Static(kv_heads), Extent::Static(head_dim)],
            &alloc::format!("blk.{layer}.attn_k.weight"),
        );
        let wv = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(embedding), Extent::Static(kv_heads), Extent::Static(head_dim)],
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
            alloc::vec![Extent::Symbolic(1), Extent::Static(kv_heads), Extent::Static(pairs)],
            &alloc::format!("kv_cache.{layer}.k_even"),
        );
        let k_odd_cache = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(1), Extent::Static(kv_heads), Extent::Static(pairs)],
            &alloc::format!("kv_cache.{layer}.k_odd"),
        );
        let v_cache = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Symbolic(1), Extent::Static(kv_heads), Extent::Static(head_dim)],
            &alloc::format!("kv_cache.{layer}.v"),
        );

        let (x_next, layer_roots) = if expert_count == 0 {
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
                is_future,
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
                k_even_cache,
                k_odd_cache,
                v_cache,
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
                alloc::vec![Extent::Static(expert_count), Extent::Static(embedding), Extent::Static(feed_forward)],
                &alloc::format!("blk.{layer}.ffn_gate_exps.weight"),
            );
            let expert_w_up = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(expert_count), Extent::Static(embedding), Extent::Static(feed_forward)],
                &alloc::format!("blk.{layer}.ffn_up_exps.weight"),
            );
            let expert_w_down = input_leaf(
                &mut program,
                DType::Float32,
                alloc::vec![Extent::Static(expert_count), Extent::Static(feed_forward), Extent::Static(embedding)],
                &alloc::format!("blk.{layer}.ffn_down_exps.weight"),
            );

            append_mistral_cached_moe_layer(
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
            )?
        };
        x = x_next;
        cache_roots.push(layer_roots);
    }

    let output_norm_weight = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(embedding)], "output_norm.weight");
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
    let logits = reduce(&mut program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, logits_product, "sdv->sdv", "sv->sdv")?;

    Ok((program, logits, cache_roots))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

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
        let evaluated =
            crate::cpu::evaluate_parallel(&program, &symbols, &blocks, &[root, probabilities], workers)
                .expect("the block evaluates");

        let output = evaluated.root();
        assert_eq!(output.len(), SEQUENCE * MODEL, "a vacuous output proves nothing");
        assert!(output.iter().all(|value| value.is_finite()), "output must be finite");

        let (rows, _) = evaluated.get(probabilities).expect("probabilities were requested");
        assert_eq!(rows.len(), SEQUENCE * SEQUENCE);
        for row in rows.as_chunks::<SEQUENCE>().0 {
            let total: f32 = row.iter().sum();
            assert!((total - 1.0).abs() < 1e-5, "softmax row sums to {total}, not 1.0");
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
        assert_eq!(output.len(), SEQUENCE * MODEL, "a vacuous output proves nothing");
        assert!(output.iter().all(|value| value.is_finite()), "output must be finite");

        let (rows, _) = evaluated.get(probabilities).expect("probabilities were requested");
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

        let evaluated =
            crate::cpu::evaluate_parallel(&program, &symbols, &real_blocks, &[root, probabilities], workers)
                .expect("the block evaluates");

        let output = evaluated.root();
        assert_eq!(output.len(), SEQUENCE * MODEL, "a vacuous output proves nothing");
        assert!(output.iter().all(|value| value.is_finite()), "output must be finite");

        let (rows, _) = evaluated.get(probabilities).expect("probabilities were requested");
        for row in rows.as_chunks::<SEQUENCE>().0 {
            let total: f32 = row.iter().sum();
            assert!((total - 1.0).abs() < 1e-5, "softmax row sums to {total}, not 1.0");
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
        let evaluated =
            crate::cpu::evaluate(&program, &symbols, &blocks, &[root]).expect("the convolution evaluates");

        let output = evaluated.root();
        assert_eq!(output.len(), 2 * IMAGE * IMAGE, "a vacuous output proves nothing");

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
            .map(|out| (0..d_in).map(|inp| row[inp] * matrix[inp * d_out + out]).sum())
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
            matrix_a[0], matrix_a[1], matrix_a[2], matrix_a[3], matrix_a[4], matrix_a[5],
            matrix_b[0], matrix_b[1], matrix_b[2], matrix_b[3], matrix_b[4], matrix_b[5],
        ];
        let blocks: [&[f32]; 3] = [&x, &gate_w, &expert_w];

        let evaluated =
            crate::cpu::evaluate_parallel(&program, &symbols, &blocks, &[root], workers)
                .expect("the moe block evaluates");
        let output = evaluated.root();
        assert_eq!(output.len(), SEQUENCE * D_OUT, "a vacuous output proves nothing");

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
            matrix_a[0], matrix_a[1], matrix_a[2], matrix_a[3], matrix_a[4], matrix_a[5],
            matrix_a[0], matrix_a[1], matrix_a[2], matrix_a[3], matrix_a[4], matrix_a[5],
        ];
        let degenerate_blocks: [&[f32]; 3] = [&x, &swapped_gate_w, &uniform_expert_w];
        let evaluated_degenerate = crate::cpu::evaluate_parallel(
            &program,
            &symbols,
            &degenerate_blocks,
            &[root],
            workers,
        )
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
    fn swiglu_ffn(x: &[f32], gate_w: &[f32], up_w: &[f32], down_w: &[f32], d_in: usize, hidden: usize) -> alloc::vec::Vec<f32> {
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
                .max_by(|&&left, &&right| logits[left].partial_cmp(&logits[right]).expect("logits are finite"))
                .expect("k does not exceed the expert count");
            routes.push(winner);
            remaining.retain(|&candidate| candidate != winner);
        }
        let max_logit = routes.iter().map(|&expert| logits[expert]).fold(f32::NEG_INFINITY, f32::max);
        let unnormalized: alloc::vec::Vec<f32> = routes.iter().map(|&expert| (logits[expert] - max_logit).exp()).collect();
        let total: f32 = unnormalized.iter().sum();
        routes.into_iter().zip(unnormalized).map(|(expert, weight)| (expert, weight / total)).collect()
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

        let gate_weights: [[f32; EMBEDDING * FEED_FORWARD]; 3] =
            [[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 2.0], [1.0, 1.0, 1.0, 1.0]];
        let up_weights: [[f32; EMBEDDING * FEED_FORWARD]; 3] =
            [[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0], [2.0, 0.0, 0.0, 2.0]];
        let down_weights: [[f32; FEED_FORWARD * EMBEDDING]; 3] =
            [[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]];

        let stack_experts = |weights: &[[f32; EMBEDDING * FEED_FORWARD]; 3]| -> alloc::vec::Vec<f32> {
            weights.iter().flatten().copied().collect()
        };
        let expert_w_gate = stack_experts(&gate_weights);
        let expert_w_up = stack_experts(&up_weights);
        let expert_w_down: alloc::vec::Vec<f32> = down_weights.iter().flatten().copied().collect();

        let mut program = Vec::new();
        let x_node = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Symbolic(0), Extent::Static(EMBEDDING as u32)], "x");
        let gate_inp_node = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(EMBEDDING as u32), Extent::Static(EXPERT_COUNT)],
            "gate_inp",
        );
        let expert_w_gate_node = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(EXPERT_COUNT), Extent::Static(EMBEDDING as u32), Extent::Static(FEED_FORWARD as u32)],
            "expert_w_gate",
        );
        let expert_w_up_node = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(EXPERT_COUNT), Extent::Static(EMBEDDING as u32), Extent::Static(FEED_FORWARD as u32)],
            "expert_w_up",
        );
        let expert_w_down_node = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(EXPERT_COUNT), Extent::Static(FEED_FORWARD as u32), Extent::Static(EMBEDDING as u32)],
            "expert_w_down",
        );
        let ones = scalar_constant(&mut program, 1.0);

        let root = append_moe_ffn(
            &mut program,
            x_node,
            gate_inp_node,
            expert_w_gate_node,
            expert_w_up_node,
            expert_w_down_node,
            EXPERT_COUNT,
            EXPERT_USED_COUNT,
            ones,
            ExpertGatingFunc::Softmax,
        )
        .expect("the routed ffn lowers");

        let symbols = [SEQUENCE as u64];
        crate::shape::infer(&program, &symbols).expect("the routed ffn infers");

        let blocks: [&[f32]; 5] = [&x, &gate_inp, &expert_w_gate, &expert_w_up, &expert_w_down];
        let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");
        let evaluated =
            crate::cpu::evaluate_parallel(&program, &symbols, &blocks, &[root], workers).expect("the routed ffn evaluates");
        let output = evaluated.root();
        assert_eq!(output.len(), SEQUENCE * EMBEDDING, "a vacuous output proves nothing");

        for (token, x_row) in x.chunks(EMBEDDING).enumerate() {
            let logits: alloc::vec::Vec<f32> = (0..EXPERT_COUNT as usize)
                .map(|expert| (0..EMBEDDING).map(|dim| x_row[dim] * gate_inp[dim * EXPERT_COUNT as usize + expert]).sum())
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
    /// top-`k` by `sigmoid(logits)`, then weights each selected expert by
    /// that same `sigmoid(logits)` value, normalized over only the
    /// selected set. Mirrors [`top_k_routes_and_weights`]'s own shape
    /// (max-by, retain, normalize) with a sigmoid instead of a softmax.
    fn sigmoid_topk_routes_and_weights(logits: &[f32], k: usize) -> alloc::vec::Vec<(usize, f32)> {
        let scores: alloc::vec::Vec<f32> = logits.iter().map(|&logit| 1.0 / (1.0 + (-logit).exp())).collect();
        let mut remaining: alloc::vec::Vec<usize> = (0..logits.len()).collect();
        let mut routes = alloc::vec::Vec::new();
        for _ in 0..k {
            let winner = *remaining
                .iter()
                .max_by(|&&left, &&right| scores[left].partial_cmp(&scores[right]).expect("scores are finite"))
                .expect("k does not exceed the expert count");
            routes.push(winner);
            remaining.retain(|&candidate| candidate != winner);
        }
        let total: f32 = routes.iter().map(|&expert| scores[expert]).sum();
        routes.into_iter().map(|expert| (expert, scores[expert] / total)).collect()
    }

    /// End-to-end proof for [`append_moe_ffn`]'s `Sigmoid` branch, same
    /// shape as [`a_routed_ffn_built_by_append_moe_ffn_matches_an_independent_topk_swiglu_reference`]
    /// (same `x`/`gate_inp`/expert weights, so the same logits `[3, 2, 4]`/
    /// `[1, 4, 8]` this time run through `sigmoid` instead of `softmax`):
    /// sigmoid preserves the raw-logit ranking, so both tokens route to the
    /// same PAIR of experts as the softmax test above -- the only thing
    /// that must differ is each selected expert's combination WEIGHT, which
    /// this asserts against an independently computed sigmoid share.
    #[test]
    fn a_routed_ffn_built_by_append_moe_ffn_with_sigmoid_gating_matches_an_independent_reference() {
        const SEQUENCE: usize = 2;
        const EMBEDDING: usize = 2;
        const FEED_FORWARD: usize = 2;
        const EXPERT_COUNT: u32 = 3;
        const EXPERT_USED_COUNT: u32 = 2;

        let x: [f32; SEQUENCE * EMBEDDING] = [3.0, 2.0, 1.0, 4.0];
        let gate_inp: [f32; EMBEDDING * EXPERT_COUNT as usize] = [1.0, 0.0, 0.0, 0.0, 1.0, 2.0];

        let gate_weights: [[f32; EMBEDDING * FEED_FORWARD]; 3] =
            [[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 2.0], [1.0, 1.0, 1.0, 1.0]];
        let up_weights: [[f32; EMBEDDING * FEED_FORWARD]; 3] =
            [[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0], [2.0, 0.0, 0.0, 2.0]];
        let down_weights: [[f32; FEED_FORWARD * EMBEDDING]; 3] =
            [[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]];

        let stack_experts = |weights: &[[f32; EMBEDDING * FEED_FORWARD]; 3]| -> alloc::vec::Vec<f32> {
            weights.iter().flatten().copied().collect()
        };
        let expert_w_gate = stack_experts(&gate_weights);
        let expert_w_up = stack_experts(&up_weights);
        let expert_w_down: alloc::vec::Vec<f32> = down_weights.iter().flatten().copied().collect();

        let mut program = Vec::new();
        let x_node = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Symbolic(0), Extent::Static(EMBEDDING as u32)], "x");
        let gate_inp_node = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(EMBEDDING as u32), Extent::Static(EXPERT_COUNT)],
            "gate_inp",
        );
        let expert_w_gate_node = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(EXPERT_COUNT), Extent::Static(EMBEDDING as u32), Extent::Static(FEED_FORWARD as u32)],
            "expert_w_gate",
        );
        let expert_w_up_node = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(EXPERT_COUNT), Extent::Static(EMBEDDING as u32), Extent::Static(FEED_FORWARD as u32)],
            "expert_w_up",
        );
        let expert_w_down_node = input_leaf(
            &mut program,
            DType::Float32,
            alloc::vec![Extent::Static(EXPERT_COUNT), Extent::Static(FEED_FORWARD as u32), Extent::Static(EMBEDDING as u32)],
            "expert_w_down",
        );
        let ones = scalar_constant(&mut program, 1.0);

        let root = append_moe_ffn(
            &mut program,
            x_node,
            gate_inp_node,
            expert_w_gate_node,
            expert_w_up_node,
            expert_w_down_node,
            EXPERT_COUNT,
            EXPERT_USED_COUNT,
            ones,
            ExpertGatingFunc::Sigmoid,
        )
        .expect("the sigmoid-gated routed ffn lowers");

        let symbols = [SEQUENCE as u64];
        crate::shape::infer(&program, &symbols).expect("the sigmoid-gated routed ffn infers");

        let blocks: [&[f32]; 5] = [&x, &gate_inp, &expert_w_gate, &expert_w_up, &expert_w_down];
        let workers = core::num::NonZeroUsize::new(1).expect("one worker is nonzero");
        let evaluated =
            crate::cpu::evaluate_parallel(&program, &symbols, &blocks, &[root], workers).expect("the sigmoid-gated routed ffn evaluates");
        let output = evaluated.root();
        assert_eq!(output.len(), SEQUENCE * EMBEDDING, "a vacuous output proves nothing");

        for (token, x_row) in x.chunks(EMBEDDING).enumerate() {
            let logits: alloc::vec::Vec<f32> = (0..EXPERT_COUNT as usize)
                .map(|expert| (0..EMBEDDING).map(|dim| x_row[dim] * gate_inp[dim * EXPERT_COUNT as usize + expert]).sum())
                .collect();
            let routes = sigmoid_topk_routes_and_weights(&logits, EXPERT_USED_COUNT as usize);
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
                    "token {token}: got {found:?}, expected {expected:?} (independent sigmoid top-{EXPERT_USED_COUNT} \
                     swiglu reference); routes={routes:?}"
                );
            }
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
                            let key_value =
                                k[(key_position * kv_heads + kv_head) * head_dim + dim];
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
                        let index =
                            ((query_position * kv_heads + kv_head) * group + offset) * head_dim
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
        assert_eq!(output.len(), expected_len, "a vacuous output proves nothing");
        assert!(output.iter().all(|value| value.is_finite()), "output must be finite");

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
        assert_eq!(compared, expected_len, "every element must be checked, not a subset");

        let (rows, _) = evaluated.get(probabilities).expect("probabilities were requested");
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
            let query = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Symbolic(0), Extent::Static(HEAD_DIM as u32)], "q");
            let key = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(KEYS as u32), Extent::Static(HEAD_DIM as u32)], "k");

            let score_product = elementwise(&mut program, DType::Float32, ScalarOp::Multiply, &[(query, "sh->sth"), (key, "th->sth")])
                .expect("score product builds");
            let scores = reduce(&mut program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, score_product, "sth->sth", "st->sth")
                .expect("scores reduce builds");
            let scores = if scaled {
                let inv_sqrt_head_dim = scalar_constant(&mut program, 1.0 / (HEAD_DIM as f32).sqrt());
                elementwise(&mut program, DType::Float32, ScalarOp::Multiply, &[(scores, "st->st"), (inv_sqrt_head_dim, "->st")])
                    .expect("scaling multiply builds")
            } else {
                scores
            };
            let score_max = reduce(&mut program, DType::Float32, ScalarOp::Maximum, ReduceInit::NegativeInfinity, scores, "st->st", "s->st")
                .expect("max reduce builds");
            let shifted = elementwise(&mut program, DType::Float32, ScalarOp::Subtract, &[(scores, "st->st"), (score_max, "s->st")])
                .expect("shift builds");
            let weights = elementwise(&mut program, DType::Float32, ScalarOp::Exponential, &[(shifted, "st->st")])
                .expect("exponential builds");
            let weight_sum = reduce(&mut program, DType::Float32, ScalarOp::Add, ReduceInit::Zero, weights, "st->st", "s->st")
                .expect("weight sum reduce builds");
            let inv_weight_sum = elementwise(&mut program, DType::Float32, ScalarOp::Reciprocal, &[(weight_sum, "s->s")])
                .expect("reciprocal builds");
            let probabilities = elementwise(&mut program, DType::Float32, ScalarOp::Multiply, &[(weights, "st->st"), (inv_weight_sum, "s->st")])
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
            crate::shape::infer(&program, &symbols).expect("the isolated score/softmax slice infers");
            let root = NodeId(program.len() as u32 - 1);
            assert_eq!(root, probabilities, "probabilities is the program's own last node");
            let blocks: [&[f32]; 2] = [&query_vector, &key_vectors];
            let evaluated = crate::cpu::evaluate_parallel(&program, &symbols, &blocks, &[root], workers)
                .expect("the isolated score/softmax slice evaluates");
            evaluated.root().to_vec()
        };

        let unscaled = evaluate(false);
        let scaled = evaluate(true);

        for (label, row) in [("unscaled (pre-fix)", &unscaled), ("scaled (post-fix)", &scaled)] {
            let total: f32 = row.iter().sum();
            assert!((total - 1.0).abs() < 1e-4, "{label} softmax row sums to {total}, not 1.0");
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

        let bound = crate::bind::bind(&program, &shapes, &[root, ffn_out])
            .expect("the real layer binds");
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
        assert_eq!(output.len(), SEQUENCE * EMBEDDING, "a vacuous output proves nothing");
        assert!(output.iter().all(|value| value.is_finite()), "output must be finite");
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

        let (rows, _) = evaluated.get(probabilities).expect("probabilities were requested");
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

        for collapsed in ["inv_dim", "ones", "group_ones", "inv_sqrt_head_dim", "neg_infinity"] {
            assert!(
                !bound.contains(&collapsed),
                "{collapsed} is a literal and must be an Op::Constant, not a bound Input; \
                 bound names are {bound:?}"
            );
        }

        assert!(bound.contains(&"eps"), "eps is model metadata, still bound");
        assert!(bound.contains(&"ids"), "ids is per-call data, still bound");
        assert!(bound.contains(&"rope_cos"), "rope_cos varies with position, still bound");
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
        let cached_shapes =
            crate::shape::infer(&cached_program, &[1, 71]).expect("one new position against a 71-position cache infers");
        let bound = crate::bind::bind(&cached_program, &cached_shapes, &cached_outputs)
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
        let seed = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(1)], "seed");
        let mut level: Vec<NodeId> = (0..LEAVES)
            .map(|_| {
                elementwise(&mut program, DType::Float32, ScalarOp::Add, &[(seed, "a->a"), (seed, "a->a")])
                    .expect("a scalar add lowers")
            })
            .collect();
        while level.len() > 1 {
            level = level
                .chunks(2)
                .map(|pair| match pair {
                    [left, right] => {
                        elementwise(&mut program, DType::Float32, ScalarOp::Add, &[(*left, "a->a"), (*right, "a->a")])
                            .expect("a scalar add lowers")
                    }
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
            crate::cpu::evaluate_named(&program, &[1], &named, &[root]).expect("the tree evaluates");
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
        let seed = input_leaf(&mut program, DType::Float32, alloc::vec![Extent::Static(1)], "seed");
        let mut tip = seed;
        for _ in 0..depth {
            tip = elementwise(&mut program, DType::Float32, ScalarOp::Add, &[(tip, "a->a"), (seed, "a->a")])
                .expect("a scalar add chains");
        }

        let seed_data = alloc::vec![1.0f32];
        let named: [(&str, &[f32]); 1] = [("seed", seed_data.as_slice())];
        let evaluated = crate::cpu::evaluate_named(&program, &[1], &named, &[tip]).expect("the chain evaluates");
        let (data, _) = evaluated.get(tip).expect("chain tip present");

        std::println!("chain_depth depth={depth} nodes={} value={}", program.len(), data[0]);
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
        let ids: Vec<f32> = (0..SEQUENCE).map(|position| (position % VOCAB) as f32).collect();
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
        let evaluated = crate::cpu::evaluate_parallel(&program, &symbols, &blocks, &[root], workers)
            .expect("the whole real-dimension mistral forward pass evaluates");
        let wall = wall_start.elapsed();
        std::println!(
            "whole_forward_pass: wall_clock={wall:?} per_layer={:?}",
            wall / BLOCK_COUNT
        );

        let output = evaluated.root();
        assert_eq!(output.len(), SEQUENCE * VOCAB, "logits must be [seq, vocab]");
        assert!(output.iter().all(|value| value.is_finite()), "logits must be finite");

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
    fn rope_angles(start: usize, count: usize, pairs: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
        let mut cos = alloc::vec![0.0f32; count * pairs];
        let mut sin = alloc::vec![0.0f32; count * pairs];
        for offset in 0..count {
            let position = (start + offset) as f32;
            for pair in 0..pairs {
                let theta =
                    position * crate::sized::ROPE_FREQ_BASE_DEFAULT.powf(-((2 * pair) as f32) / (head_dim as f32));
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
        let mut uncached_named: Vec<(&str, &[f32])> =
            alloc::vec![("ids", ids_f32.as_slice()), ("token_embd.weight", table.as_slice()), ("eps", epsilon_full.as_slice()), ("rope_cos", cos_full.as_slice()), ("rope_sin", sin_full.as_slice())];
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
        let uncached_evaluated = crate::cpu::evaluate_named(&uncached_program, &[SEQUENCE as u64], &uncached_named, &[uncached_root])
            .expect("uncached forward pass evaluates");
        let (uncached_logits, uncached_shape) = uncached_evaluated.get(uncached_root).expect("uncached logits present");
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
        let mut prefill_named: Vec<(&str, &[f32])> = alloc::vec![
            ("ids", &ids_f32[..PROMPT_LEN]),
            ("token_embd.weight", table.as_slice()),
            ("eps", epsilon_prompt.as_slice()),
            ("rope_cos", cos_prompt.as_slice()),
            ("rope_sin", sin_prompt.as_slice()),
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
        let prefill_evaluated = crate::cpu::evaluate_named(&cached_program, &prefill_symbols, &prefill_named, &prefill_roots)
            .expect("prefill call evaluates");

        let mut k_even_cache: Vec<Vec<f32>> = Vec::with_capacity(BLOCK_COUNT as usize);
        let mut k_odd_cache: Vec<Vec<f32>> = Vec::with_capacity(BLOCK_COUNT as usize);
        let mut v_cache: Vec<Vec<f32>> = Vec::with_capacity(BLOCK_COUNT as usize);
        for (even, odd, value) in &cache_roots {
            let (even_data, _) = prefill_evaluated.get(*even).expect("prefill k_even present");
            let (odd_data, _) = prefill_evaluated.get(*odd).expect("prefill k_odd present");
            let (value_data, _) = prefill_evaluated.get(*value).expect("prefill v present");
            k_even_cache.push(even_data.to_vec());
            k_odd_cache.push(odd_data.to_vec());
            v_cache.push(value_data.to_vec());
        }

        let mut decode_named: Vec<(&str, &[f32])> = alloc::vec![
            ("ids", &ids_f32[PROMPT_LEN..]),
            ("token_embd.weight", table.as_slice()),
            ("eps", epsilon_one.as_slice()),
            ("rope_cos", cos_decode.as_slice()),
            ("rope_sin", sin_decode.as_slice()),
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
        let decode_evaluated = crate::cpu::evaluate_named(&cached_program, &decode_symbols, &decode_named, &[cached_logits_root])
            .expect("decode call evaluates");
        let (decode_logits, decode_shape) = decode_evaluated.get(cached_logits_root).expect("decode logits present");
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
            uncached_last_position.iter().any(|&value| value != uncached_last_position[0]),
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
                shape: alloc::vec![Extent::Static(3), Extent::Static(1)],
                name: Some("weight".into()),
            },
        );
        let output = causal_conv1d(&mut program, x, weight, 3).expect("causal conv lowers");

        let x_data = [1.0f32, 2.0, 3.0, 4.0];
        let weight_data = [1.0f32, 10.0, 100.0];
        let evaluated = crate::cpu::evaluate_named(&program, &[4], &[("x", &x_data), ("weight", &weight_data)], &[output])
            .expect("causal conv evaluates");
        let (result, shape) = evaluated.get(output).expect("conv output present");

        std::println!("causal_conv1d result={result:?} shape={shape:?}");
        assert_eq!(shape, [4u64, 1u64]);
        assert_eq!(result, [100.0, 210.0, 321.0, 432.0]);
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
                shape: alloc::vec![Extent::Static(3), Extent::Static(1)],
                name: Some("weight".into()),
            },
        );
        let output = causal_conv1d(&mut program, x, weight, 3).expect("causal conv lowers");

        let x_data = [1.0f32, 2.0, 3.0, 4.0];
        // tap l=2 perturbed from 100 to 99: every output except out[0] still
        // matches (out[0]'s only real contribution is tap l=2, so this alone
        // would move it too -- included to show the test is not vacuous).
        let perturbed_weight_data = [1.0f32, 10.0, 99.0];
        let evaluated = crate::cpu::evaluate_named(&program, &[4], &[("x", &x_data), ("weight", &perturbed_weight_data)], &[output])
            .expect("causal conv evaluates");
        let (result, _shape) = evaluated.get(output).expect("conv output present");

        std::println!("perturbed causal_conv1d result={result:?}");
        assert_ne!(
            result, [100.0, 210.0, 321.0, 432.0],
            "a perturbed tap weight must move the output away from the hand-computed reference \
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
        let derived = LayerKind::from_tensor_names(names, layer).expect("a real block names exactly one marker");
        assert_eq!(derived, expected);
    }

    #[proxima::test]
    async fn layer_kind_names_the_block_when_neither_marker_is_present() {
        let names = ["token_embd.weight", "output_norm.weight"];
        let error = LayerKind::from_tensor_names(names, 7).expect_err("a block with no marker cannot derive a kind");
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
        let (program, _logits) =
            lfm2_forward_program_with_experts(128_000, 2048, 7168, 1792, 32, 8, 64, 24, 32, 4, 2, 3, &layer_kinds)
                .expect("the hybrid forward pass lowers to a program");
        let build_elapsed = build_start.elapsed();

        let infer_start = std::time::Instant::now();
        crate::shape::infer(&program, &[REAL_CONTEXT]).expect("the hybrid forward pass infers at its real context length");
        let infer_elapsed = infer_start.elapsed();

        std::println!("lfm2_forward_program_with_experts: nodes={} build={build_elapsed:?} infer={infer_elapsed:?}", program.len());
        assert!(
            program.len() > 1_000,
            "24 hybrid blocks plus embedding/lm-head should be well over a thousand nodes, not {}",
            program.len()
        );
    }

    #[proxima::test]
    async fn lfm2_forward_program_rejects_a_layer_kinds_length_mismatch() {
        let layer_kinds = [LayerKind::Attention, LayerKind::ShortConv];
        let error = lfm2_forward_program_with_experts(128_000, 2048, 7168, 1792, 32, 8, 64, 24, 32, 4, 2, 3, &layer_kinds)
            .expect_err("2 layer_kinds against block_count=24 must be rejected");
        assert!(
            matches!(error, TensorError::LayerKindCountMismatch { expected: 24, found: 2 }),
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
        let error = causal_conv1d(&mut program, x, weight, 0).expect_err("l_cache=0 has no window to convolve");
        assert!(matches!(error, TensorError::InvalidConvConfig { l_cache: 0 }), "got {error:?}");
    }
}
