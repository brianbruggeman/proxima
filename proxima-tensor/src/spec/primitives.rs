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

/// Parse `operand->iteration` into an iteration rank and one projected
/// iteration axis per operand axis.
pub(super) fn parse_projection(notation: &str) -> Result<(u16, Vec<u16>), TensorError> {
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
pub(super) fn find_axis(space: &[char], letter: char, notation: &str) -> Result<u16, TensorError> {
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
pub(super) fn parse_operand_pattern(notation: &str) -> Result<IndexPattern, TensorError> {
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
pub(super) fn parse_axis_expr(token: &str, space: &[char], notation: &str) -> Result<AxisIndex, TensorError> {
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
pub(super) fn single_letter(text: &str, notation: &str) -> Result<char, TensorError> {
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
pub(super) fn split_signed_terms(token: &str) -> Vec<(i32, &str)> {
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

pub(super) fn lookup(resolved: &BTreeMap<String, NodeId>, reference: &str) -> Result<NodeId, TensorError> {
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
    pub(super) fn offsets(self) -> (alloc::string::String, alloc::string::String) {
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
        &[
            (source, same_pattern.as_str()),
            (cos_new, trig_pattern.as_str()),
        ],
    )?;
    let partner_sin = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (source, partner_pattern.as_str()),
            (sin_new, trig_pattern.as_str()),
        ],
    )?;
    let rotated_same = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[
            (same_cos, out_identity.as_str()),
            (partner_sin, out_identity.as_str()),
        ],
    )?;

    let partner_cos = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (source, partner_pattern.as_str()),
            (cos_new, trig_pattern.as_str()),
        ],
    )?;
    let same_sin = elementwise(
        program,
        DType::Float32,
        ScalarOp::Multiply,
        &[
            (source, same_pattern.as_str()),
            (sin_new, trig_pattern.as_str()),
        ],
    )?;
    let rotated_partner = elementwise(
        program,
        DType::Float32,
        ScalarOp::Add,
        &[
            (partner_cos, out_identity.as_str()),
            (same_sin, out_identity.as_str()),
        ],
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

/// Gathers a plan-fixed permutation along the middle axis of a rank-three
/// tensor. `indices` is a caller-owned index vector whose length is the
/// output extent; keeping it as an input allows converted checkpoint layouts
/// to express a permutation without inventing affine extent equations.
pub fn gather_computed(
    program: &mut Vec<Op>,
    source: NodeId,
    indices: NodeId,
    index_map: IndexPattern,
    base: IndexPattern,
    gathered_dim: u16,
    dtype: DType,
) -> NodeId {
    let gathered_map = IndexMap::Computed {
        indices,
        index_map,
        base,
        gathered_dim,
    };
    op::append(
        program,
        Op::Elementwise {
            dtype,
            body: ScalarOp::Identity,
            operands: alloc::vec![(source, gathered_map)],
            name: None,
        },
    )
}

pub fn gather_axis_permutation(
    program: &mut Vec<Op>,
    source: NodeId,
    indices: NodeId,
    dtype: DType,
    head_dim: u32,
) -> NodeId {
    gather_computed(
        program,
        source,
        indices,
        map::projection(3, &[1]),
        IndexPattern {
            iter_rank: 3,
            axes: alloc::vec![
                AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(0)).collect(),
                    offset: 0,
                    len: None,
                },
                AxisIndex::default(),
                AxisIndex {
                    terms: core::iter::once(AxisTerm::projection(2)).collect(),
                    offset: 0,
                    len: Some(Extent::Static(head_dim)),
                },
            ],
        },
        1,
        dtype,
    )
}

/// Gathers a fixed head permutation from `[s, kv_head, dim]` into
/// `[s, head, dim]`, retaining the Qwen convenience spelling while delegating
/// to the dtype-preserving permutation primitive.
pub fn gather_head_permutation(
    program: &mut Vec<Op>,
    source: NodeId,
    indices: NodeId,
    head_dim: u32,
) -> NodeId {
    gather_axis_permutation(program, source, indices, DType::Float32, head_dim)
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
    let (query_index, key_index) = causal_index_pair(program);
    let is_future = elementwise(
        program,
        DType::Float32,
        ScalarOp::Greater,
        &[(key_index, "t->st"), (query_index, "s->st")],
    )?;
    let neg_infinity = scalar_constant(program, f32::NEG_INFINITY);
    Ok((is_future, neg_infinity))
}

/// The two block-local position [`Op::Iota`]s [`causal_mask`] and
/// [`causal_mask_windowed`] both build: `query_index` and `key_index` over
/// the same axis (symbol 0), since a block's key range is exactly its own
/// query range. Factored out so the windowed variant doesn't re-litigate
/// `causal_mask`'s own iota shapes.
fn causal_index_pair(program: &mut Vec<Op>) -> (NodeId, NodeId) {
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
    (query_index, key_index)
}

/// [`causal_mask`]'s sliding-window counterpart — Gemma 3's local-attention
/// layers (25 of its 30) restrict each query to the most recent `window`
/// keys instead of the whole causal prefix. A cell is masked iff
/// `key_index > query_index` (exactly [`causal_mask`]'s own `is_future`) OR
/// `query_index - key_index >= window` (too far in the past). The two
/// `{0.0, 1.0}`-valued booleans are combined with `ScalarOp::Maximum`, which
/// is OR over that domain — no new `ScalarOp` variant needed, the same
/// closed set [`causal_mask`] already uses.
///
/// `window` of `None` or `Some(0)` delegates straight to [`causal_mask`]
/// rather than building an always-false `too_old` and OR-ing it in, so the
/// unwindowed path is byte-for-byte [`causal_mask`]'s own program, not an
/// algebraically-equivalent one.
pub fn causal_mask_windowed(
    program: &mut Vec<Op>,
    window: Option<u32>,
) -> Result<(NodeId, NodeId), TensorError> {
    let Some(window) = window.filter(|&window| window > 0) else {
        return causal_mask(program);
    };
    let (query_index, key_index) = causal_index_pair(program);
    let is_future = elementwise(
        program,
        DType::Float32,
        ScalarOp::Greater,
        &[(key_index, "t->st"), (query_index, "s->st")],
    )?;
    let distance = elementwise(
        program,
        DType::Float32,
        ScalarOp::Subtract,
        &[(query_index, "s->st"), (key_index, "t->st")],
    )?;
    let window_ceiling = scalar_constant(program, window as f32 - 1.0);
    let too_old = elementwise(
        program,
        DType::Float32,
        ScalarOp::Greater,
        &[(distance, "st->st"), (window_ceiling, "->st")],
    )?;
    let is_masked = elementwise(
        program,
        DType::Float32,
        ScalarOp::Maximum,
        &[(is_future, "st->st"), (too_old, "st->st")],
    )?;
    let neg_infinity = scalar_constant(program, f32::NEG_INFINITY);
    Ok((is_masked, neg_infinity))
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
pub fn causal_mask_merged(
    program: &mut Vec<Op>,
    cached_len: NodeId,
) -> Result<NodeId, TensorError> {
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
