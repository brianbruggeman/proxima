use alloc::collections::BTreeMap;
use alloc::string::String;

use bon::Builder;
use conflaguration::{Settings, Validate, ValidationMessage};

use super::*;

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
pub(super) fn resolve_map_spec(
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
pub(super) fn build_base_pattern(rank: u16, projected: &[u16], gathered_dim: u16) -> IndexPattern {
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

pub(super) fn lookup(
    resolved: &BTreeMap<String, NodeId>,
    reference: &str,
) -> Result<NodeId, TensorError> {
    resolved
        .get(reference)
        .copied()
        .ok_or_else(|| TensorError::UnknownNode(reference.to_string()))
}

impl TryFrom<&[Op]> for ProgramSpec {
    type Error = TensorError;

    fn try_from(program: &[Op]) -> Result<Self, Self::Error> {
        let node = program
            .iter()
            .enumerate()
            .map(|(index, expr)| node_from_op(index, expr))
            .collect::<Result<Vec<NodeSpec>, TensorError>>()?;
        Ok(Self { node })
    }
}

fn node_id(index: usize) -> String {
    format!("n{index}")
}

fn reference(index: usize, node: NodeId) -> Result<String, TensorError> {
    let target = node.0 as usize;
    if target < index {
        Ok(node_id(target))
    } else {
        Err(TensorError::UnknownNode(node_id(target)))
    }
}

fn expressed(index: usize, notation: Option<String>) -> Result<String, TensorError> {
    notation.ok_or(TensorError::NoConfigFace {
        node: NodeId(index as u32),
        reason: "index map is not expressible in the spec notation",
    })
}

fn node_from_op(index: usize, expr: &Op) -> Result<NodeSpec, TensorError> {
    match expr {
        Op::Input { dtype, shape, name } => Ok(NodeSpec::Input {
            id: node_id(index),
            dtype: *dtype,
            shape: extents_to_spec(shape),
            name: name.clone(),
        }),
        Op::Elementwise {
            dtype,
            body,
            operands,
            name,
        } => elementwise_to_spec(index, *dtype, *body, operands, name.clone()),
        Op::Reduce(reduce) => reduce_to_spec(index, reduce),
        Op::Iota { dtype, extent } => Ok(NodeSpec::Iota {
            id: node_id(index),
            dtype: *dtype,
            extent: extent_to_spec(*extent),
        }),
        Op::Constant {
            dtype,
            shape,
            value,
        } => Ok(NodeSpec::Constant {
            id: node_id(index),
            dtype: *dtype,
            shape: extents_to_spec(shape),
            value: *value,
        }),
    }
}

fn extent_to_spec(extent: Extent) -> ExtentSpec {
    match extent {
        Extent::Static(size) => ExtentSpec::Static(size),
        Extent::Symbolic(symbol) => ExtentSpec::Symbolic(format!("?{symbol}")),
    }
}

fn extents_to_spec(extents: &[Extent]) -> Vec<ExtentSpec> {
    extents.iter().copied().map(extent_to_spec).collect()
}

fn elementwise_to_spec(
    index: usize,
    dtype: DType,
    body: ScalarOp,
    operands: &[(NodeId, IndexMap)],
    name: Option<String>,
) -> Result<NodeSpec, TensorError> {
    let inputs = operands
        .iter()
        .map(|(node, _)| reference(index, *node))
        .collect::<Result<Vec<String>, TensorError>>()?;
    let maps = operands
        .iter()
        .map(|(_, map)| map_to_spec(index, map))
        .collect::<Result<Vec<MapSpec>, TensorError>>()?;
    Ok(NodeSpec::Elementwise {
        id: node_id(index),
        dtype,
        body,
        inputs,
        maps,
        name,
    })
}

fn reduce_to_spec(index: usize, reduce: &Reduce) -> Result<NodeSpec, TensorError> {
    let (IndexMap::Affine(in_pattern), IndexMap::Affine(out_pattern)) =
        (&reduce.in_map, &reduce.out_map)
    else {
        return Err(TensorError::NoConfigFace {
            node: NodeId(index as u32),
            reason: "data-dependent reduce index map has no spec form",
        });
    };
    Ok(NodeSpec::Reduce {
        id: node_id(index),
        dtype: reduce.dtype,
        body: reduce.body,
        init: reduce.init,
        input: reference(index, reduce.operand)?,
        in_map: expressed(index, operand_notation(in_pattern))?,
        out_map: expressed(index, projection_notation(out_pattern))?,
        keep: reduce.keep,
        name: reduce.name.clone(),
    })
}

fn map_to_spec(index: usize, map: &IndexMap) -> Result<MapSpec, TensorError> {
    match map {
        IndexMap::Affine(pattern) => Ok(MapSpec::Projection(expressed(
            index,
            operand_notation(pattern),
        )?)),
        IndexMap::Computed {
            indices,
            index_map,
            base,
            gathered_dim,
        } => gather_to_spec(index, *indices, index_map, base, *gathered_dim),
    }
}

fn gather_to_spec(
    index: usize,
    indices: NodeId,
    index_map: &IndexPattern,
    base: &IndexPattern,
    dim: u16,
) -> Result<MapSpec, TensorError> {
    let base_rest = without_gathered_axis(base, dim).ok_or(TensorError::NoConfigFace {
        node: NodeId(index as u32),
        reason: "gather base has no gathered axis at its gathered position",
    })?;
    Ok(MapSpec::Gather {
        gather: reference(index, indices)?,
        index_map: expressed(index, projection_notation(index_map))?,
        map: expressed(index, projection_notation(&base_rest))?,
        dim,
    })
}

fn without_gathered_axis(base: &IndexPattern, dim: u16) -> Option<IndexPattern> {
    let position = dim as usize;
    if base.axes.get(position)? != &AxisIndex::default() {
        return None;
    }
    let mut rest = base.clone();
    rest.axes.remove(position);
    Some(rest)
}

fn letter_for(axis: u16) -> Option<char> {
    u8::try_from(axis)
        .ok()
        .filter(|position| *position < 26)
        .map(|position| char::from(b'a' + position))
}

fn iteration_notation(iter_rank: u16) -> Option<String> {
    (0..iter_rank).map(letter_for).collect()
}

fn term_notation(leading: bool, term: &AxisTerm) -> Option<String> {
    let letter = letter_for(term.axis)?;
    let sign = match (term.coeff < 0, leading) {
        (true, _) => "-",
        (false, true) => "",
        (false, false) => "+",
    };
    let magnitude = term.coeff.unsigned_abs();
    if magnitude == 1 && !(leading && term.coeff < 0) {
        Some(format!("{sign}{letter}"))
    } else {
        Some(format!("{sign}{magnitude}*{letter}"))
    }
}

fn axis_notation(axis: &AxisIndex) -> Option<String> {
    let terms = axis
        .terms
        .iter()
        .enumerate()
        .map(|(position, term)| term_notation(position == 0, term))
        .collect::<Option<String>>()?;
    let mut text = if terms.is_empty() {
        axis.offset.to_string()
    } else if axis.offset == 0 {
        terms
    } else {
        format!("{terms}{:+}", axis.offset)
    };
    match axis.len {
        None => {}
        Some(Extent::Static(size)) => text.push_str(&format!("@{size}")),
        Some(Extent::Symbolic(_)) => return None,
    }
    Some(text)
}

fn operand_notation(pattern: &IndexPattern) -> Option<String> {
    let operand = pattern
        .axes
        .iter()
        .map(axis_notation)
        .collect::<Option<Vec<String>>>()?
        .join(",");
    let notation = format!("{operand}->{}", iteration_notation(pattern.iter_rank)?);
    parse_operand_pattern(&notation)
        .ok()
        .filter(|parsed| parsed == pattern)
        .map(|_| notation)
}

fn plain_axis(axis: &AxisIndex) -> Option<u16> {
    match axis.terms.as_slice() {
        [term] if term.coeff == 1 && axis.offset == 0 && axis.len.is_none() => Some(term.axis),
        _ => None,
    }
}

fn projection_notation(pattern: &IndexPattern) -> Option<String> {
    let projected = pattern
        .axes
        .iter()
        .map(plain_axis)
        .collect::<Option<Vec<u16>>>()?;
    let operand = projected
        .iter()
        .copied()
        .map(letter_for)
        .collect::<Option<String>>()?;
    let notation = format!("{operand}->{}", iteration_notation(pattern.iter_rank)?);
    let (rank, parsed) = parse_projection(&notation).ok()?;
    (map::projection(rank, &parsed) == *pattern).then_some(notation)
}
