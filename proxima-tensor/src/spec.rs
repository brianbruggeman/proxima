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
//! broadcast — the overwhelming majority — and deliberately does **not**
//! cover multi-term axes. A convolution's `h*stride + r*dilation` has no
//! einsum spelling, so it is built with [`map::affine`](crate::map::affine)
//! and stays out of the string grammar rather than growing it a syntax.
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
}

impl NodeSpec {
    #[must_use]
    pub fn id(&self) -> &str {
        match self {
            Self::Input { id, .. } | Self::Elementwise { id, .. } | Self::Reduce { id, .. } => id,
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
        .map(|letter| {
            space
                .iter()
                .position(|candidate| *candidate == letter)
                .map(|found| found as u16)
                .ok_or_else(|| TensorError::UnknownIndexLetter {
                    notation: notation.to_string(),
                    letter,
                })
        })
        .collect::<Result<Vec<u16>, TensorError>>()?;
    Ok((space.len() as u16, projected))
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
                NodeSpec::Input { .. } => {}
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
                    let (in_rank, in_projected) = parse_projection(in_map)?;
                    let (out_rank, out_projected) = parse_projection(out_map)?;
                    op::append(
                        &mut program,
                        Op::Reduce(Reduce {
                            dtype: *dtype,
                            body: *body,
                            init: *init,
                            operand,
                            in_map: IndexMap::Affine(map::projection(in_rank, &in_projected)),
                            out_map: IndexMap::Affine(map::projection(out_rank, &out_projected)),
                            keep: *keep,
                            name: name.clone(),
                        }),
                    )
                }
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
        MapSpec::Projection(notation) => {
            let (rank, projected) = parse_projection(notation)?;
            Ok(IndexMap::Affine(map::projection(rank, &projected)))
        }
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
}
