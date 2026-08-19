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
#[allow(clippy::too_many_arguments)]
pub fn mistral_forward_program(
    vocab: u32,
    embedding: u32,
    feed_forward: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    block_count: u32,
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

        x = append_mistral_layer(
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
        )?;
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use rstest::rstest;

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

    #[rstest]
    #[case::identity("ij->ij", 2, &[0, 1])]
    #[case::transpose("ji->ij", 2, &[1, 0])]
    #[case::broadcast("j->ij", 2, &[1])]
    #[case::contraction_lhs("ik->ijk", 3, &[0, 2])]
    #[case::full_reduction("->i", 1, &[])]
    fn projection_notation_reads_like_einsum(
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
        for row in rows.chunks_exact(SEQUENCE) {
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
        for (query, row) in rows.chunks_exact(SEQUENCE).enumerate() {
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
        for row in rows.chunks_exact(SEQUENCE) {
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
        let program = mistral_forward_program(128, 64, 172, 8, 4, 16, 2)
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
        let program = mistral_forward_program(32_002, 4096, 14336, 32, 8, 128, 32)
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
}
