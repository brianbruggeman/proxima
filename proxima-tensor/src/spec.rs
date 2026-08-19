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
//! `AxisTerm { axis, coeff }` sum [`map::affine`](crate::map::affine) already
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
//! ([`parse_operand_pattern`]) — the asymmetry where only `Elementwise`
//! operands could spell a multi-term axis was an oversight, not a design
//! decision, since a `Reduce`'s operand is windowed exactly the same way a
//! convolution's `Elementwise(Multiply)` operand is (see
//! `specs/conv2d.toml`). `out_map` stays on the older, bare-letter-only
//! [`parse_projection`] deliberately: [`shape::project_output_shape`] already
//! rejects any `out_map` axis that is not a pure single-term `coeff == 1`
//! projection (`NotLowerable`, "reduce output maps must be pure projections
//! in v1"), so parsing a richer `out_map` would only ever be thrown away at
//! bind time — [`parse_projection`]'s narrower grammar gives the same
//! rejection at parse time instead, before a spec that could never lower
//! reaches shape inference at all.
//!
//! A [`NodeSpec::Elementwise`] operand map may instead be a
//! [`MapSpec::Gather`] table: `{ gather = "ids", index_map = "s->sd", map =
//! "d->sd", dim = 0 }`. `index_map` addresses the `gather` node the same
//! einsum way; `map` addresses the operand's *non-gathered* axes only, in
//! operand-axis order, skipping the position `dim` names —
//! [`build_base_pattern`] splices an empty (gathered) entry back in at that
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
#[derive(Debug, Clone, Default, PartialEq, Eq, Builder, Deserialize, Serialize, Settings)]
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
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
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
}

impl NodeSpec {
    #[must_use]
    pub fn id(&self) -> &str {
        match self {
            Self::Input { id, .. } | Self::Elementwise { id, .. } | Self::Reduce { id, .. }
            | Self::Iota { id, .. } => id,
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
                NodeSpec::Input { .. } | NodeSpec::Iota { .. } => {}
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
}
