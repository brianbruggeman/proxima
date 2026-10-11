//! Operand views: the `slice_by_index`, `transpose` and `expand_dims`
//! operations that bring an operand's tensor to the layout its `IndexPattern`
//! reads, emitted from the same classification the census tallies.

use proxima_tensor::{IndexMap, IndexPattern, NodeId};

use super::message::{
    Immediate, Kind, Name, begin_operation, input_immediate, input_name, operation_output,
};
use super::{Context, MilError};
use crate::wire::Encoder;

/// Marks the operand axis a gather consumes; no iteration axis has this number.
const GATHERED: u16 = u16::MAX;

/// Longest rank a view is planned for; ranks above it are refused, not truncated.
pub(super) const RANK_CAPACITY: usize = 8;

/// A stack-resident list no longer than [`RANK_CAPACITY`].
#[derive(Debug, Clone, Copy)]
pub(super) struct Stack<T: Copy + Default> {
    items: [T; RANK_CAPACITY],
    length: usize,
}

impl<T: Copy + Default> Stack<T> {
    pub(super) fn new() -> Self {
        Self {
            items: [T::default(); RANK_CAPACITY],
            length: 0,
        }
    }

    pub(super) fn from_slice(node: NodeId, source: &[T]) -> Result<Self, MilError> {
        let mut stack = Self::new();
        source.iter().try_for_each(|item| stack.push(node, *item))?;
        Ok(stack)
    }

    pub(super) fn push(&mut self, node: NodeId, item: T) -> Result<(), MilError> {
        let slot = self
            .items
            .get_mut(self.length)
            .ok_or(MilError::RankTooLarge {
                node,
                rank: self.length + 1,
            })?;
        *slot = item;
        self.length += 1;
        Ok(())
    }

    pub(super) fn as_slice(&self) -> &[T] {
        &self.items[..self.length]
    }
}

impl<T: Copy + Default + Ord> Stack<T> {
    pub(super) fn sort(&mut self) {
        self.items[..self.length].sort_unstable();
    }

    pub(super) fn sort_by_key<K: Ord>(&mut self, key: impl FnMut(&T) -> K) {
        self.items[..self.length].sort_unstable_by_key(key);
    }
}

/// An operand mid-alignment: the name of its latest tensor, its dims and the
/// iteration axis each of its axes reads.
#[derive(Debug, Clone, Copy)]
pub(super) struct Current {
    pub(super) name: Name,
    pub(super) dims: Stack<u64>,
    pub(super) axes: Stack<u16>,
    offsets: Stack<i64>,
}

impl Current {
    pub(super) fn from_parts(name: Name, dims: Stack<u64>, axes: Stack<u16>) -> Self {
        Self {
            name,
            dims,
            axes,
            offsets: Stack::new(),
        }
    }

    pub(super) fn has_offsets(&self) -> bool {
        self.offsets.as_slice().iter().any(|offset| *offset != 0)
    }
}

/// Where in a node's emission a view is created, so its name is unique.
#[derive(Debug, Clone, Copy)]
pub(super) struct Site {
    pub(super) host: NodeId,
    pub(super) position: usize,
    pub(super) kind: Kind,
}

impl Site {
    pub(super) fn name(&self, tag: &str, step: usize) -> Name {
        Name::step(self.host.0, tag, self.position * 8 + step)
    }
}

impl Current {
    /// The operand node read through `pattern`, before any view.
    pub(super) fn read(
        context: &Context<'_>,
        site: &Site,
        operand: NodeId,
        pattern: &IndexPattern,
    ) -> Result<Self, MilError> {
        Self::read_skipping(context, site, operand, pattern, None)
    }

    /// As [`Current::read`], with the axis at `skipped` left to a gather.
    fn read_skipping(
        context: &Context<'_>,
        site: &Site,
        operand: NodeId,
        pattern: &IndexPattern,
        skipped: Option<usize>,
    ) -> Result<Self, MilError> {
        let host = site.host;
        let dims = Stack::from_slice(host, context.shapes.of(operand))?;
        let mut axes = Stack::new();
        let mut offsets = Stack::new();
        for (position, index) in pattern.axes.iter().enumerate() {
            if skipped == Some(position) {
                axes.push(host, GATHERED)?;
                offsets.push(host, 0)?;
                continue;
            }
            let [term] = index.terms.as_slice() else {
                return Err(not_emitted(
                    host,
                    "operand axis that is not one iteration axis",
                ));
            };
            if term.coeff != 1 || index.len.is_some() || index.offset < 0 {
                return Err(not_emitted(
                    host,
                    "strided, bounded or negative operand window",
                ));
            }
            axes.push(host, term.axis)?;
            offsets.push(host, i64::from(index.offset))?;
        }
        if axes.as_slice().len() != dims.as_slice().len() {
            return Err(not_emitted(
                host,
                "operand pattern rank differs from its shape",
            ));
        }
        Ok(Self {
            name: Name::node(operand.0),
            dims,
            axes,
            offsets,
        })
    }

    /// Slices every axis that reads at an offset, to the iteration extent.
    pub(super) fn slice(
        mut self,
        encoder: &mut Encoder<'_>,
        site: &Site,
        extents: &[u64],
    ) -> Result<Self, MilError> {
        if self.offsets.as_slice().iter().all(|offset| *offset == 0) {
            return Ok(self);
        }
        let mut begin = Stack::<i64>::new();
        let mut end = Stack::<i64>::new();
        let mut dims = Stack::<u64>::new();
        for (position, axis) in self.axes.as_slice().iter().enumerate() {
            let extent = *extents.get(usize::from(*axis)).ok_or_else(|| {
                not_emitted(site.host, "operand axis outside the iteration space")
            })?;
            let start = self.offsets.as_slice()[position];
            begin.push(site.host, start)?;
            end.push(site.host, start + i64::try_from(extent).unwrap_or(i64::MAX))?;
            dims.push(site.host, extent)?;
        }
        let name = site.name("s", 0);
        let operation = begin_operation(encoder, "slice_by_index");
        input_name(encoder, "x", &self.name);
        input_immediate(encoder, "begin", Immediate::Ints(begin.as_slice()));
        input_immediate(encoder, "end", Immediate::Ints(end.as_slice()));
        operation_output(encoder, &name, site.kind, dims.as_slice());
        encoder.close(operation);
        self.name = name;
        self.dims = dims;
        self.offsets =
            Stack::from_slice(site.host, &[0; RANK_CAPACITY][..self.axes.as_slice().len()])?;
        Ok(self)
    }

    /// Transposes so the axes read in `target` order; `target` holds the same axes.
    pub(super) fn permute_to(
        mut self,
        encoder: &mut Encoder<'_>,
        site: &Site,
        target: &[u16],
        output: Name,
    ) -> Result<Self, MilError> {
        if self.axes.as_slice() == target {
            return Ok(self);
        }
        let mut perm = Stack::<i64>::new();
        let mut dims = Stack::<u64>::new();
        for wanted in target {
            let from = self
                .axes
                .as_slice()
                .iter()
                .position(|axis| axis == wanted)
                .ok_or_else(|| not_emitted(site.host, "operand does not read a required axis"))?;
            perm.push(site.host, i64::try_from(from).unwrap_or(0))?;
            dims.push(site.host, self.dims.as_slice()[from])?;
        }
        let name = output;
        let operation = begin_operation(encoder, "transpose");
        input_name(encoder, "x", &self.name);
        input_immediate(encoder, "perm", Immediate::Ints(perm.as_slice()));
        operation_output(encoder, &name, site.kind, dims.as_slice());
        encoder.close(operation);
        self.name = name;
        self.dims = dims;
        self.axes = Stack::from_slice(site.host, target)?;
        Ok(self)
    }

    /// Inserts size-1 axes where the iteration space has axes the operand does not read.
    ///
    /// With `full` false an operand that already reads the trailing axes is left
    /// alone, since MIL broadcasts leading size-1 axes by itself.
    pub(super) fn expand_to(
        mut self,
        encoder: &mut Encoder<'_>,
        site: &Site,
        rank: usize,
        full: bool,
    ) -> Result<Self, MilError> {
        let used = self.axes.as_slice().len();
        let trailing = self
            .axes
            .as_slice()
            .iter()
            .enumerate()
            .all(|(position, axis)| usize::from(*axis) == rank - used + position);
        if used == rank || (trailing && !full) {
            return Ok(self);
        }
        let mut inserted = Stack::<i64>::new();
        let mut dims = Stack::<u64>::new();
        let mut cursor = 0;
        for position in 0..rank {
            let reads = self
                .axes
                .as_slice()
                .get(cursor)
                .is_some_and(|axis| usize::from(*axis) == position);
            if reads {
                dims.push(site.host, self.dims.as_slice()[cursor])?;
                cursor += 1;
            } else {
                dims.push(site.host, 1)?;
                inserted.push(site.host, i64::try_from(position).unwrap_or(0))?;
            }
        }
        let name = site.name("e", 2);
        let operation = begin_operation(encoder, "expand_dims");
        input_name(encoder, "x", &self.name);
        input_immediate(encoder, "axes", Immediate::Ints(inserted.as_slice()));
        operation_output(encoder, &name, site.kind, dims.as_slice());
        encoder.close(operation);
        self.name = name;
        self.dims = dims;
        Ok(self)
    }

    /// The axes in ascending order, which [`Current::permute_to`] aligns to.
    pub(super) fn sorted_axes(&self, node: NodeId) -> Result<Stack<u16>, MilError> {
        let mut sorted = Stack::from_slice(node, self.axes.as_slice())?;
        sorted.sort();
        let repeats = sorted.as_slice().windows(2).any(|pair| pair[0] == pair[1]);
        if repeats {
            return Err(not_emitted(node, "operand reads one iteration axis twice"));
        }
        Ok(sorted)
    }
}

pub(super) fn not_emitted(node: NodeId, form: &'static str) -> MilError {
    MilError::NotEmitted { node, form }
}

/// Brings `operand` to the broadcast layout of an iteration space of `extents`.
///
/// # Errors
/// [`MilError::NotEmitted`] for a read no view expresses.
pub(super) fn broadcast_operand(
    context: &Context<'_>,
    encoder: &mut Encoder<'_>,
    site: &Site,
    operand: NodeId,
    map: &IndexMap,
    extents: &[u64],
) -> Result<Current, MilError> {
    let read = read_map(context, encoder, site, operand, map)?.slice(encoder, site, extents)?;
    let sorted = read.sorted_axes(site.host)?;
    read.permute_to(encoder, site, sorted.as_slice(), site.name("t", 1))?
        .expand_to(encoder, site, extents.len(), false)
}

/// The affine pattern of `map`.
///
/// # Errors
/// [`MilError::NotEmitted`] for a computed (gathered) map.
pub(super) fn affine(
    node: NodeId,
    map: &proxima_tensor::IndexMap,
) -> Result<&IndexPattern, MilError> {
    match map {
        proxima_tensor::IndexMap::Affine(pattern) => Ok(pattern),
        proxima_tensor::IndexMap::Computed { .. } => Err(not_emitted(node, "computed index map")),
    }
}

/// The iteration axis each pattern axis reads, when every one reads exactly one at unit stride.
///
/// # Errors
/// [`MilError::NotEmitted`] for a pattern that is not a projection.
pub(super) fn pattern_axes(node: NodeId, pattern: &IndexPattern) -> Result<Stack<u16>, MilError> {
    let mut axes = Stack::new();
    for index in &pattern.axes {
        match index.terms.as_slice() {
            [term] if term.coeff == 1 && index.offset == 0 && index.len.is_none() => {
                axes.push(node, term.axis)?;
            }
            _ => return Err(not_emitted(node, "pattern that is not a projection")),
        }
    }
    Ok(axes)
}

/// The operand node read through `map`: a plain read, or a gather for a computed map.
///
/// A computed map is `gather(x, indices, axis)`: the result keeps `x`'s axes
/// before `axis`, then the index tensor's axes, then `x`'s axes after it, which
/// is the order MIL's `gather` produces and so needs no copy decision here.
///
/// # Errors
/// [`MilError::NotEmitted`] for a read no view expresses.
pub(super) fn read_map(
    context: &Context<'_>,
    encoder: &mut Encoder<'_>,
    site: &Site,
    operand: NodeId,
    map: &IndexMap,
) -> Result<Current, MilError> {
    match map {
        IndexMap::Affine(pattern) => Current::read(context, site, operand, pattern),
        IndexMap::Computed {
            indices,
            index_map,
            base,
            gathered_dim,
        } => gather(
            context,
            encoder,
            site,
            [operand, *indices],
            [base, index_map],
            usize::from(*gathered_dim),
        ),
    }
}

fn gather(
    context: &Context<'_>,
    encoder: &mut Encoder<'_>,
    site: &Site,
    [operand, indices]: [NodeId; 2],
    [base, index_map]: [&IndexPattern; 2],
    axis: usize,
) -> Result<Current, MilError> {
    let host = site.host;
    let source = Current::read_skipping(context, site, operand, base, Some(axis))?;
    let positions = Current::read(context, site, indices, index_map)?;
    if source.has_offsets() || positions.has_offsets() {
        return Err(not_emitted(host, "gather over an offset window"));
    }
    let positions = positions.as_integer(encoder, context, site, indices)?;
    let (before, after) = (axis, source.axes.as_slice().len() - axis - 1);
    let axes = joined(
        host,
        &source.axes.as_slice()[..before],
        positions.axes.as_slice(),
        &source.axes.as_slice()[source.axes.as_slice().len() - after..],
    )?;
    let dims = joined(
        host,
        &source.dims.as_slice()[..before],
        positions.dims.as_slice(),
        &source.dims.as_slice()[source.dims.as_slice().len() - after..],
    )?;
    let name = site.name("g", 4);
    let operation = begin_operation(encoder, "gather");
    input_name(encoder, "x", &source.name);
    input_name(encoder, "indices", &positions.name);
    input_immediate(
        encoder,
        "axis",
        Immediate::Int(i64::try_from(axis).unwrap_or(0)),
    );
    operation_output(encoder, &name, site.kind, dims.as_slice());
    encoder.close(operation);
    Ok(Current::from_parts(name, dims, axes))
}

fn joined<T: Copy + Default>(
    node: NodeId,
    first: &[T],
    middle: &[T],
    last: &[T],
) -> Result<Stack<T>, MilError> {
    let mut joined = Stack::new();
    first
        .iter()
        .chain(middle)
        .chain(last)
        .try_for_each(|item| joined.push(node, *item))?;
    Ok(joined)
}

impl Current {
    /// The tensor as `int32`, casting when the index node is not already integer.
    fn as_integer(
        self,
        encoder: &mut Encoder<'_>,
        context: &Context<'_>,
        site: &Site,
        indices: NodeId,
    ) -> Result<Self, MilError> {
        let kind = super::node::kind_of(site.host, context.dtype_of(indices)?)?;
        if kind == Kind::Int32 {
            return Ok(self);
        }
        let name = site.name("i", 5);
        super::node::cast(
            encoder,
            &self.name,
            &name,
            Kind::Int32,
            self.dims.as_slice(),
        );
        Ok(Self { name, ..self })
    }
}

impl Current {
    /// Reshapes a run of adjacent axes equal to `run` into one axis named by `run[0]`.
    pub(super) fn merge_run(
        self,
        encoder: &mut Encoder<'_>,
        site: &Site,
        run: &[u16],
    ) -> Result<Self, MilError> {
        let width = run.len();
        let found = self
            .axes
            .as_slice()
            .windows(width)
            .position(|window| window == run);
        let Some(start) = found.filter(|_| width > 1) else {
            return Ok(self);
        };
        let mut dims = Stack::<u64>::new();
        let mut axes = Stack::<u16>::new();
        for (position, (extent, axis)) in self
            .dims
            .as_slice()
            .iter()
            .zip(self.axes.as_slice())
            .enumerate()
        {
            match position {
                at if at == start => {
                    let merged = self.dims.as_slice()[start..start + width].iter().product();
                    dims.push(site.host, merged)?;
                    axes.push(site.host, run[0])?;
                }
                at if at > start && at < start + width => {}
                _ => {
                    dims.push(site.host, *extent)?;
                    axes.push(site.host, *axis)?;
                }
            }
        }
        let name = site.name("r", 6);
        reshape(encoder, site, &self.name, &name, dims.as_slice())?;
        Ok(Self::from_parts(name, dims, axes))
    }
}

fn reshape(
    encoder: &mut Encoder<'_>,
    site: &Site,
    source: &Name,
    target: &Name,
    dims: &[u64],
) -> Result<(), MilError> {
    let mut shape = Stack::<i64>::new();
    dims.iter()
        .try_for_each(|extent| shape.push(site.host, i64::try_from(*extent).unwrap_or(0)))?;
    let operation = begin_operation(encoder, "reshape");
    input_name(encoder, "x", source);
    input_immediate(encoder, "shape", Immediate::Ints(shape.as_slice()));
    operation_output(encoder, target, site.kind, dims);
    encoder.close(operation);
    Ok(())
}
