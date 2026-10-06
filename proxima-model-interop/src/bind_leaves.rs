//! Weights bound by walking a lowered program's `Op::Input` leaves against a
//! GGUF tensor directory: lowering is a pure function of the config, binding
//! is a pure function of (program, directory), and no table of tensor names
//! exists anywhere in between.
//!
//! [`bind_program_leaves`] composes the per-tensor binders this crate already
//! ships ([`bind_dense_as`], [`bind_matmul_weight_as`],
//! [`bind_moe_expert_weights`], the paired and triple joins) and chooses among
//! them from the program, never from a name:
//!
//! - a leaf a `Multiply` feeds into an `Add` reduce is a matmul weight when its
//!   contracted axes lead its kept axes (the program declares `[in, out]`, the
//!   file stores `[out, in]`), and binds in file order otherwise;
//! - a leaf a gather indexes with a computed map and that is a rank-3 stack is
//!   an expert stack;
//! - every other leaf binds as stored.
//!
//! A leaf the directory does not name resolves through the family's
//! [`BindingProfile`] (a tied output, a fused tensor the program splits, a
//! tensor stored under another name). A leaf nothing resolves is a runtime
//! input (token ids, RoPE tables, cache rows) and is left for the step to
//! feed; such a leaf that was really a weight surfaces as an unbound input
//! naming it at the first step.
//!
//! Teaching pointer: to run a new model through this, lower its descriptor
//! with [`proxima_tensor::spec::build_forward`] and hand the program here. A
//! family whose files name tensors differently adds a file under
//! `profiles/binding/`, not a branch.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::tensor::TensorInfo;
use proxima_gguf::types::GgmlType;
use proxima_tensor::cpu::QuantizedBlock;
use proxima_tensor::op::{Extent, Op, Reduce, ScalarOp};
use proxima_tensor::IndexPattern;

use crate::bind::{
    BoundWeights, aligned_f32_view, as_block, bind_dense_as, bind_matmul_weight_as,
    bind_matmul_weight_paired, bind_matmul_weight_triple, bind_moe_expert_weights,
    bind_moe_stacked_experts, bind_native_f32, bind_matmul_weight_transposed_f32,
    codec_from_ggml_type, reinterpret_f32,
};
use crate::error::InteropError;
use crate::profiles::{BindingProfile, TensorAlias};
use crate::serving::WeightPrecisionRule;

/// How the program reads a leaf, decided from the ops that consume it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Role {
    /// The leaf is read in the order the file stores it.
    Native,
    /// The leaf is a matmul weight the program declares `[in, out]` over a file
    /// that stores `[out, in]`.
    InOut,
    /// The leaf is indexed by a computed map: a stack of per-expert matrices.
    Gathered,
}

/// What satisfies a leaf in the tensor directory.
enum Source<'directory> {
    Direct(&'directory TensorInfo),
    Part {
        tensor: &'directory TensorInfo,
        part: u32,
        of: u32,
    },
    Join(Vec<String>),
}

/// Binds every weight `program`'s `Op::Input` leaves name, from `parsed`'s
/// tensor directory, and returns them as one [`BoundWeights`].
///
/// `binding` supplies the names the directory uses that the program does not
/// ([`BindingProfile`]); `weight_precision` is
/// [`crate::serving::ServingConfig::weight_precision`]'s recode rule set,
/// consulted by the per-tensor binders exactly as before.
///
/// # Errors
///
/// [`InteropError::LeafShapeMismatch`] when a leaf and its tensor disagree on
/// element count; [`InteropError::LeafAxesInterleaved`] when the program
/// contracts and keeps a leaf's axes interleaved; [`InteropError::TensorPartInvalid`]
/// for a `part` alias that cannot cut its tensor into whole-block row slices;
/// whatever the per-tensor binders can fail with.
pub fn bind_program_leaves<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    program: &[Op],
    binding: &BindingProfile,
    weight_precision: &'file [WeightPrecisionRule<'file>],
) -> Result<BoundWeights<'file>, InteropError> {
    let mut state = BoundWeights::new(weight_precision);
    state.resident_bytes = file_bytes.len();
    bind_missing_leaves(parsed, file_bytes, program, binding, &mut state)?;
    Ok(state)
}

/// [`bind_program_leaves`] into weights that already hold some of them: a
/// leaf (or `extra` tensor) `state` already names is left as bound, so
/// re-lowering a checkpoint under a config binds only what the new program
/// adds, and a config that keeps the same leaves binds nothing.
///
/// # Errors
///
/// As [`bind_program_leaves`].
pub(crate) fn bind_missing_leaves<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    program: &[Op],
    binding: &BindingProfile,
    state: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    let directory: BTreeMap<&str, &TensorInfo> = parsed
        .tensors
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect();
    let consumers = consumers_of(program);
    let mut bound: BTreeSet<String> = state
        .owned()
        .iter()
        .map(|(name, _)| name.clone())
        .chain(state.packed().iter().map(|(name, _)| name.clone()))
        .chain(state.packed_owned().iter().map(|(name, _, _)| name.clone()))
        .collect();

    for (index, op) in program.iter().enumerate() {
        let Op::Input {
            shape,
            name: Some(leaf),
            ..
        } = op
        else {
            continue;
        };
        if !bound.insert(leaf.clone()) {
            continue;
        }
        let leaf = Leaf {
            name: leaf,
            shape,
            role: role_of(program, &consumers, index, leaf, shape)?,
        };
        match resolve(&directory, binding, leaf.name) {
            Some(source) => bind_leaf(parsed, file_bytes, binding, &leaf, &source, state)?,
            None => bind_per_expert_stack(parsed, file_bytes, &leaf, state)?,
        }
    }

    for name in &binding.extra {
        if bound.insert(name.clone()) {
            bind_native_f32(parsed, file_bytes, name, name.clone(), state)?;
        }
    }
    Ok(())
}

struct Leaf<'program> {
    name: &'program str,
    shape: &'program [Extent],
    role: Role,
}

impl Leaf<'_> {
    fn elements(&self) -> Option<u64> {
        self.shape
            .iter()
            .map(|extent| match extent {
                Extent::Static(width) => Some(u64::from(*width)),
                Extent::Symbolic(_) => None,
            })
            .product()
    }
}

fn resolve<'directory>(
    directory: &BTreeMap<&str, &'directory TensorInfo>,
    binding: &BindingProfile,
    leaf: &str,
) -> Option<Source<'directory>> {
    if let Some(tensor) = directory.get(leaf) {
        return Some(Source::Direct(tensor));
    }
    binding
        .aliases_for(leaf)
        .find_map(|(alias, prefix)| match alias {
            TensorAlias::Rename { from, .. } => directory
                .get(format!("{prefix}{from}").as_str())
                .map(|tensor| Source::Direct(tensor)),
            TensorAlias::Part { from, part, of, .. } => directory
                .get(format!("{prefix}{from}").as_str())
                .map(|tensor| Source::Part {
                    tensor,
                    part: *part,
                    of: *of,
                }),
            TensorAlias::Join { from, .. } => {
                let names: Vec<String> = from.iter().map(|part| format!("{prefix}{part}")).collect();
                names
                    .iter()
                    .all(|name| directory.contains_key(name.as_str()))
                    .then_some(Source::Join(names))
            }
        })
}

fn bind_leaf<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    binding: &BindingProfile,
    leaf: &Leaf<'_>,
    source: &Source<'_>,
    state: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    match source {
        Source::Direct(tensor) => {
            check_elements(leaf, tensor, tensor.element_count())?;
            bind_direct(parsed, file_bytes, binding, leaf, tensor, state)
        }
        Source::Part { tensor, part, of } => bind_part(parsed, file_bytes, leaf, tensor, *part, *of, state),
        Source::Join(names) => bind_join(parsed, file_bytes, leaf.name, names, state),
    }
}

fn check_elements(leaf: &Leaf<'_>, tensor: &TensorInfo, expected: u64) -> Result<(), InteropError> {
    match leaf.elements() {
        Some(declared) if declared != expected => Err(InteropError::LeafShapeMismatch {
            leaf: leaf.name.into(),
            leaf_elements: declared,
            tensor: tensor.name.clone(),
            tensor_elements: expected,
        }),
        _ => Ok(()),
    }
}

fn bind_direct<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    binding: &BindingProfile,
    leaf: &Leaf<'_>,
    tensor: &TensorInfo,
    state: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    let dims = tensor.dims.as_slice();
    let input_width = dims.first().copied().unwrap_or(1) as usize;
    let output_width = dims.iter().skip(1).product::<u64>() as usize;
    let decode_f32 = binding.decodes_to_f32(leaf.name);

    match (leaf.role, decode_f32) {
        (Role::Gathered, _) if dims.len() >= 3 => bind_moe_stacked_experts(
            parsed,
            file_bytes,
            tensor.name.clone(),
            leaf.name.into(),
            dims[2..].iter().product::<u64>() as usize,
            dims[1] as usize,
            input_width,
            state,
        ),
        (Role::InOut, true) => bind_matmul_weight_transposed_f32(
            parsed,
            file_bytes,
            &tensor.name,
            leaf.name.into(),
            output_width,
            input_width,
            state,
        ),
        (Role::InOut, false) => bind_matmul_weight_as(
            parsed,
            file_bytes,
            &tensor.name,
            leaf.name.into(),
            output_width,
            input_width,
            state,
        ),
        (_, true) => bind_native_f32(parsed, file_bytes, &tensor.name, leaf.name.into(), state),
        (_, false) => bind_dense_as(parsed, file_bytes, &tensor.name, leaf.name.into(), state),
    }
}

fn bind_join<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    leaf: &str,
    names: &[String],
    state: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    match names {
        [first, second] => {
            bind_matmul_weight_paired(parsed, file_bytes, first, second, leaf.into(), state)
        }
        [first, second, third] => bind_matmul_weight_triple(
            parsed,
            file_bytes,
            first,
            second,
            third,
            leaf.into(),
            state,
        ),
        _ => Err(InteropError::UnknownTensor { name: leaf.into() }),
    }
}

/// Binds part `part` of `of` equal row slices of `tensor`, within every outer
/// index past its row axis. One outer index borrows the slice out of the
/// mapping; several outer indices are not contiguous, so each part is
/// assembled into its own packed buffer by one memcpy per index.
fn bind_part<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    leaf: &Leaf<'_>,
    tensor: &TensorInfo,
    part: u32,
    of: u32,
    state: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    let dims = tensor.dims.as_slice();
    let (row_length, rows) = match dims {
        [row_length, rows, ..] => (*row_length, *rows),
        _ => (0, 0),
    };
    let layout = tensor.ggml_type.block_layout();
    let invalid = || InteropError::TensorPartInvalid {
        tensor: tensor.name.clone(),
        rows,
        of,
    };
    if dims.len() < 2
        || of == 0
        || part >= of
        || !rows.is_multiple_of(u64::from(of))
        || layout.block_elements == 0
        || !row_length.is_multiple_of(layout.block_elements)
    {
        return Err(invalid());
    }
    check_elements(leaf, tensor, tensor.element_count() / u64::from(of))?;
    let row_bytes = row_length / layout.block_elements * layout.block_bytes;
    let part_bytes = rows / u64::from(of) * row_bytes;
    let outer_bytes = rows * row_bytes;
    let outer: u64 = dims[2..].iter().product();

    let range = parsed.tensor_data_range(tensor, file_bytes.len() as u64)?;
    let source = &file_bytes[range.start as usize..range.end as usize];
    let slice_of = |outer_index: u64| {
        let start = (outer_index * outer_bytes + u64::from(part) * part_bytes) as usize;
        &source[start..start + part_bytes as usize]
    };

    if outer == 1 {
        return push_borrowed_part(leaf.name, tensor, slice_of(0), state);
    }
    let codec = codec_from_ggml_type(tensor.ggml_type).ok_or_else(|| {
        InteropError::UnrepresentableGgmlType {
            tensor: tensor.name.clone(),
            ggml_type: tensor.ggml_type,
        }
    })?;
    let mut assembled = Vec::with_capacity((part_bytes * outer) as usize);
    for outer_index in 0..outer {
        assembled.extend_from_slice(slice_of(outer_index));
    }
    state.resident_bytes += assembled.len();
    state.packed_owned.push((leaf.name.into(), assembled, codec));
    Ok(())
}

fn push_borrowed_part<'file>(
    leaf: &str,
    tensor: &TensorInfo,
    slice: &'file [u8],
    state: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    if tensor.ggml_type == GgmlType::F32 {
        match aligned_f32_view(slice) {
            Some(view) => state.packed.push((leaf.into(), QuantizedBlock::Float32(view))),
            None => {
                let owned = reinterpret_f32(slice);
                state.resident_bytes += owned.len() * core::mem::size_of::<f32>();
                state.owned.push((leaf.into(), owned));
            }
        }
        return Ok(());
    }
    let block = codec_from_ggml_type(tensor.ggml_type)
        .and_then(|codec| as_block(codec, slice))
        .ok_or_else(|| InteropError::UnrepresentableGgmlType {
            tensor: tensor.name.clone(),
            ggml_type: tensor.ggml_type,
        })?;
    state.packed.push((leaf.into(), block));
    Ok(())
}

/// A gathered rank-3 leaf the directory does not name whole: the per-expert
/// tensors (`blk.{layer}.{projection}.{expert}.weight`) restacked into one.
/// Any other unresolved leaf is a runtime input and binds nothing.
fn bind_per_expert_stack<'file>(
    parsed: &ParsedGguf,
    file_bytes: &'file [u8],
    leaf: &Leaf<'_>,
    state: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    let (Role::Gathered, [Extent::Static(experts), Extent::Static(input), Extent::Static(output)]) =
        (leaf.role, leaf.shape)
    else {
        return Ok(());
    };
    let Some((layer, projection)) = expert_stack_parts(leaf.name) else {
        return Ok(());
    };
    bind_moe_expert_weights(
        parsed,
        file_bytes,
        layer,
        projection,
        *experts,
        *output as usize,
        *input as usize,
        state,
    )
}

fn expert_stack_parts(leaf: &str) -> Option<(u32, &str)> {
    let (layer, tail) = leaf.strip_prefix("blk.")?.split_once('.')?;
    Some((layer.parse().ok()?, tail.strip_suffix("_exps.weight")?))
}

fn consumers_of(program: &[Op]) -> Vec<Vec<usize>> {
    let mut consumers = alloc::vec![Vec::new(); program.len()];
    for (index, op) in program.iter().enumerate() {
        for dependency in op.dependencies() {
            let slot = &mut consumers[dependency.0 as usize];
            if slot.last() != Some(&index) {
                slot.push(index);
            }
        }
    }
    consumers
}

fn role_of(
    program: &[Op],
    consumers: &[Vec<usize>],
    leaf_index: usize,
    leaf: &str,
    shape: &[Extent],
) -> Result<Role, InteropError> {
    let mut orders = BTreeSet::new();
    for &consumer in &consumers[leaf_index] {
        let Op::Elementwise {
            body: ScalarOp::Multiply,
            operands,
            ..
        } = &program[consumer]
        else {
            continue;
        };
        let Some((_, map)) = operands
            .iter()
            .find(|(node, _)| node.0 as usize == leaf_index)
        else {
            continue;
        };
        if map.is_data_dependent() {
            return Ok(Role::Gathered);
        }
        for &reducer in &consumers[consumer] {
            if let Op::Reduce(reduce) = &program[reducer]
                && reduce.body == ScalarOp::Add
                && reduce.operand.0 as usize == consumer
                && let Some(order) = contraction_order(leaf, map.affine(), reduce, shape)?
            {
                orders.insert(order);
            }
        }
    }
    let mut found = orders.into_iter();
    match (found.next(), found.next()) {
        (Some(_), Some(_)) => Err(InteropError::LeafAxesInterleaved { leaf: leaf.into() }),
        (Some(order), None) => Ok(order),
        (None, _) => Ok(Role::Native),
    }
}

/// How the reduce orders the leaf's contracted axes against its kept ones, or
/// `None` when the reduce contracts none of them.
fn contraction_order(
    leaf: &str,
    pattern: &IndexPattern,
    reduce: &Reduce,
    shape: &[Extent],
) -> Result<Option<Role>, InteropError> {
    let kept_iteration_axes: BTreeSet<u16> = reduce
        .out_map
        .affine()
        .axes
        .iter()
        .flat_map(|axis| axis.terms.iter().map(|term| term.axis))
        .collect();
    let contracted_operand_axes: BTreeSet<u16> = reduce
        .in_map
        .affine()
        .axes
        .iter()
        .enumerate()
        .filter(|(_, axis)| {
            !axis.terms.is_empty()
                && axis
                    .terms
                    .iter()
                    .all(|term| !kept_iteration_axes.contains(&term.axis))
        })
        .map(|(position, _)| position as u16)
        .collect();

    let mut contracted = Vec::new();
    let mut kept = Vec::new();
    for (position, axis) in pattern.axes.iter().enumerate() {
        let unit = matches!(shape.get(position), Some(Extent::Static(1)));
        if axis.terms.is_empty() || unit {
            continue;
        }
        if axis
            .terms
            .iter()
            .all(|term| contracted_operand_axes.contains(&term.axis))
        {
            contracted.push(position);
        } else {
            kept.push(position);
        }
    }

    let (Some(&first_contracted), Some(&last_contracted)) = (contracted.first(), contracted.last())
    else {
        return Ok(None);
    };
    let (Some(&first_kept), Some(&last_kept)) = (kept.first(), kept.last()) else {
        return Ok(Some(Role::Native));
    };
    if last_contracted < first_kept {
        Ok(Some(Role::InOut))
    } else if last_kept < first_contracted {
        Ok(Some(Role::Native))
    } else {
        Err(InteropError::LeafAxesInterleaved { leaf: leaf.into() })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use arrayvec::ArrayVec;
    use proxima_gguf::quant::q4_k;
    use proxima_gguf::{GgufModel, TensorPayload, parse_complete, write_complete};
    use proxima_tensor::{DType, IndexMap, Keep, NodeId, ReduceInit, append, map};

    use super::*;
    use crate::Codec;
    use crate::profiles::binding_profile;

    fn dims(values: &[u64]) -> ArrayVec<u64, { proxima_gguf::tensor::MAX_DIMS }> {
        values.iter().copied().collect()
    }

    fn gguf(tensors: &[(&str, &[u64], GgmlType, &[u8])]) -> Vec<u8> {
        let model = GgufModel {
            version: 3,
            metadata: Vec::new(),
            tensors: tensors
                .iter()
                .map(|(name, extents, ggml_type, data)| TensorPayload {
                    name: (*name).to_string(),
                    dims: dims(extents),
                    ggml_type: *ggml_type,
                    data,
                })
                .collect(),
        };
        write_complete(&model).expect("writes a real gguf file")
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_le_bytes()).collect()
    }

    fn input(program: &mut Vec<Op>, name: &str, extents: &[u32]) -> NodeId {
        append(
            program,
            Op::Input {
                dtype: DType::Float32,
                shape: extents.iter().map(|width| Extent::Static(*width)).collect(),
                name: Some(name.into()),
            },
        )
    }

    /// `activations[i, k] * weight[..] summed over k`, the contraction every
    /// matmul in a lowered program is: the iteration space is `(i, j, k)` and
    /// `weight_axes` says which of those the weight leaf's two axes read.
    fn contraction(weight: &str, weight_shape: &[u32], weight_axes: [u16; 2]) -> Vec<Op> {
        let mut program = Vec::new();
        let activations = input(&mut program, "activations", &[1, 3]);
        let leaf = input(&mut program, weight, weight_shape);
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (activations, IndexMap::Affine(map::projection(3, &[0, 2]))),
                    (leaf, IndexMap::Affine(map::projection(3, &weight_axes))),
                ],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        program
    }

    fn bind<'file>(
        bytes: &'file [u8],
        program: &[Op],
        family: &str,
    ) -> Result<BoundWeights<'file>, InteropError> {
        let parsed = parse_complete(bytes).expect("parses the gguf the encoder just wrote");
        bind_program_leaves(&parsed, bytes, program, &binding_profile(family)?, &[])
    }

    #[test]
    fn a_matmul_weight_the_program_declares_in_out_binds_transposed_f32() {
        let stored: Vec<f32> = (0..6).map(|value| value as f32).collect();
        let bytes = gguf(&[("blk.0.ffn_down.weight", &[3, 2], GgmlType::F32, &f32_bytes(&stored))]);
        let program = contraction("blk.0.ffn_down.weight", &[3, 2], [2, 1]);

        let weights = bind(&bytes, &program, "llama").expect("binds");

        let [(name, bound)] = weights.owned() else {
            panic!("one owned weight, got {:?}", weights.owned().len());
        };
        assert_eq!(name, "blk.0.ffn_down.weight");
        assert_eq!(bound, &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn a_leaf_the_program_declares_out_in_binds_as_stored() {
        let stored: Vec<f32> = (0..6).map(|value| value as f32).collect();
        let bytes = gguf(&[("blk.0.ffn_down.weight", &[3, 2], GgmlType::F32, &f32_bytes(&stored))]);
        let program = contraction("blk.0.ffn_down.weight", &[2, 3], [1, 2]);

        let weights = bind(&bytes, &program, "llama").expect("binds");

        let [(name, QuantizedBlock::Float32(bound))] = weights.packed() else {
            panic!("one borrowed f32 weight, got {:?}", weights.packed().len());
        };
        assert_eq!(name, "blk.0.ffn_down.weight");
        assert_eq!(*bound, stored.as_slice());
    }

    #[test]
    fn a_leaf_no_tensor_names_is_a_runtime_input_and_binds_nothing() {
        let bytes = gguf(&[("blk.0.ffn_down.weight", &[3, 2], GgmlType::F32, &f32_bytes(&[0.0; 6]))]);
        let program = contraction("blk.0.ffn_down.weight", &[3, 2], [2, 1]);

        let weights = bind(&bytes, &program, "llama").expect("binds");

        let bound: Vec<&str> = weights
            .owned()
            .iter()
            .map(|(name, _)| name.as_str())
            .chain(weights.packed().iter().map(|(name, _)| name.as_str()))
            .collect();
        assert_eq!(bound, ["blk.0.ffn_down.weight"], "`activations` is fed per step, never bound");
    }

    #[test]
    fn a_tied_output_reads_the_embedding_table_through_the_default_alias() {
        let stored: Vec<f32> = (0..6).map(|value| value as f32).collect();
        let bytes = gguf(&[("token_embd.weight", &[3, 2], GgmlType::F32, &f32_bytes(&stored))]);
        let program = contraction("output.weight", &[3, 2], [2, 1]);

        let weights = bind(&bytes, &program, "llama").expect("binds");

        let [(name, bound)] = weights.owned() else {
            panic!("one owned weight, got {:?}", weights.owned().len());
        };
        assert_eq!(name, "output.weight");
        assert_eq!(bound, &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn binding_into_existing_weights_adds_only_the_leaves_they_lack() {
        let stored = f32_bytes(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let bytes = gguf(&[
            ("blk.0.ffn_down.weight", &[3, 2], GgmlType::F32, &stored),
            ("blk.0.ffn_up.weight", &[3, 2], GgmlType::F32, &stored),
        ]);
        let parsed = parse_complete(&bytes).expect("parses");
        let profile = binding_profile("llama").expect("defaults parse");
        let down = contraction("blk.0.ffn_down.weight", &[3, 2], [2, 1]);
        let up = contraction("blk.0.ffn_up.weight", &[3, 2], [2, 1]);
        let mut weights = bind_program_leaves(&parsed, &bytes, &down, &profile, &[]).expect("binds");

        bind_missing_leaves(&parsed, &bytes, &up, &profile, &mut weights).expect("adds the new leaf");
        bind_missing_leaves(&parsed, &bytes, &down, &profile, &mut weights).expect("leaves the bound one");

        let names: Vec<&str> = weights.owned().iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, ["blk.0.ffn_down.weight", "blk.0.ffn_up.weight"]);
    }

    #[test]
    fn a_leaf_whose_element_count_disagrees_with_its_tensor_is_refused() {
        let bytes = gguf(&[("blk.0.ffn_down.weight", &[3, 2], GgmlType::F32, &f32_bytes(&[0.0; 6]))]);
        let program = contraction("blk.0.ffn_down.weight", &[3, 3], [2, 1]);

        let refused = bind(&bytes, &program, "llama");

        assert!(
            matches!(refused, Err(InteropError::LeafShapeMismatch { leaf_elements: 9, tensor_elements: 6, .. })),
            "got {:?}",
            refused.err()
        );
    }

    #[test]
    fn contracted_and_kept_axes_that_interleave_are_refused() {
        let mut program = Vec::new();
        let activations = input(&mut program, "activations", &[1, 3]);
        let leaf = input(&mut program, "blk.0.ffn_down.weight", &[2, 3, 2]);
        let product = append(
            &mut program,
            Op::Elementwise {
                dtype: DType::Float32,
                body: ScalarOp::Multiply,
                operands: vec![
                    (activations, IndexMap::Affine(map::projection(4, &[0, 2]))),
                    (leaf, IndexMap::Affine(map::projection(4, &[1, 2, 3]))),
                ],
                name: None,
            },
        );
        append(
            &mut program,
            Op::Reduce(Reduce {
                dtype: DType::Float32,
                body: ScalarOp::Add,
                init: ReduceInit::Zero,
                operand: product,
                in_map: IndexMap::Affine(map::projection(4, &[0, 1, 2, 3])),
                out_map: IndexMap::Affine(map::projection(4, &[0, 1, 3])),
                keep: Keep::Reduce,
                name: None,
            }),
        );
        let bytes = gguf(&[("blk.0.ffn_down.weight", &[3, 2, 2], GgmlType::F32, &f32_bytes(&[0.0; 12]))]);

        let refused = bind(&bytes, &program, "llama");

        assert!(matches!(refused, Err(InteropError::LeafAxesInterleaved { .. })), "got {:?}", refused.err());
    }

    fn quantized_rows(rows: usize, row_value: impl Fn(usize) -> f32) -> Vec<u8> {
        let mut flat = vec![0.0f32; rows * q4_k::QK_K];
        for row in 0..rows {
            flat[row * q4_k::QK_K..(row + 1) * q4_k::QK_K].fill(row_value(row));
        }
        let mut quantized = vec![0u8; rows * q4_k::BLOCK_BYTES];
        q4_k::quantize(&flat, &mut quantized).expect("quantize a real constant-per-row matrix");
        quantized
    }

    fn dequantized_row_values(bytes: &[u8], rows: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; rows * q4_k::QK_K];
        q4_k::dequantize(bytes, &mut output).expect("dequantize a bound slice");
        output.chunks(q4_k::QK_K).map(|row| row[0]).collect()
    }

    #[test]
    fn a_part_alias_borrows_each_third_of_a_fused_projection_in_row_order() {
        let fused = quantized_rows(768, |row| row as f32);
        let bytes = gguf(&[("blk.0.shortconv.in_proj.weight", &[256, 768], GgmlType::Q4_K, &fused)]);
        let mut program = Vec::new();
        for suffix in ["b", "c", "x"] {
            input(&mut program, &format!("blk.0.shortconv.in_proj.weight.{suffix}"), &[256, 256]);
        }

        let weights = bind(&bytes, &program, "lfm2").expect("binds");

        assert_eq!(weights.packed().len(), 3, "each third borrows from the mapping");
        for ((name, block), (suffix, first_row)) in weights
            .packed()
            .iter()
            .zip([("b", 0usize), ("c", 256), ("x", 512)])
        {
            assert_eq!(name, &format!("blk.0.shortconv.in_proj.weight.{suffix}"));
            let QuantizedBlock::Packed { codec: Codec::Q4K, bytes } = block else {
                panic!("expected a Q4K slice, got {block:?}");
            };
            let rows = dequantized_row_values(bytes, 256);
            for (offset, value) in rows.iter().enumerate() {
                let expected = (first_row + offset) as f32;
                assert!((value - expected).abs() < 0.5, "{suffix} row {offset}: {value} vs {expected}");
            }
        }
    }

    #[test]
    fn a_part_alias_over_a_stack_assembles_each_experts_half_into_its_own_buffer() {
        let fused = quantized_rows(8, |row| (row / 4 * 10 + row % 4) as f32);
        let bytes = gguf(&[("blk.0.ffn_gate_up_exps.weight", &[256, 4, 2], GgmlType::Q4_K, &fused)]);
        let mut program = Vec::new();
        input(&mut program, "blk.0.ffn_gate_exps.weight", &[2, 256, 2]);
        input(&mut program, "blk.0.ffn_up_exps.weight", &[2, 256, 2]);

        let weights = bind(&bytes, &program, "llama").expect("binds");

        assert!(weights.packed().is_empty(), "a stack's halves are not contiguous, so none borrow");
        let [(gate_name, gate, gate_codec), (up_name, up, _)] = weights.packed_owned() else {
            panic!("two assembled halves, got {}", weights.packed_owned().len());
        };
        assert_eq!((gate_name.as_str(), up_name.as_str()), ("blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight"));
        assert_eq!(*gate_codec, Codec::Q4K);
        let near = |actual: Vec<f32>, expected: [f32; 4]| {
            assert!(actual.iter().zip(expected).all(|(left, right)| (left - right).abs() < 0.5), "{actual:?} vs {expected:?}");
        };
        near(dequantized_row_values(gate, 4), [0.0, 1.0, 10.0, 11.0]);
        near(dequantized_row_values(up, 4), [2.0, 3.0, 12.0, 13.0]);
    }

    #[test]
    fn a_part_alias_that_cannot_cut_whole_rows_is_refused() {
        let fused = quantized_rows(3, |row| row as f32);
        let bytes = gguf(&[("blk.0.shortconv.in_proj.weight", &[256, 3], GgmlType::Q4_K, &fused)]);
        let mut program = Vec::new();
        input(&mut program, "blk.0.shortconv.in_proj.weight.b", &[384]);

        let parsed = parse_complete(&bytes).expect("parses");
        let profile = BindingProfile {
            aliases: vec![TensorAlias::Part {
                leaf: "shortconv.in_proj.weight.b".into(),
                from: "shortconv.in_proj.weight".into(),
                part: 0,
                of: 2,
            }],
            ..BindingProfile::default()
        };

        let refused = bind_program_leaves(&parsed, &bytes, &program, &profile, &[]);

        assert!(matches!(refused, Err(InteropError::TensorPartInvalid { rows: 3, of: 2, .. })));
    }

    #[test]
    fn a_family_extra_binds_a_tensor_no_leaf_reads() {
        let bytes = gguf(&[("rope_freqs.weight", &[4], GgmlType::F32, &f32_bytes(&[1.0, 1.0, 1.0e30, 1.0e30]))]);

        let weights = bind(&bytes, &[], "gemma4").expect("binds");

        let [(name, values)] = weights.owned() else {
            panic!("one owned table, got {}", weights.owned().len());
        };
        assert_eq!(name, "rope_freqs.weight");
        assert_eq!(values, &[1.0, 1.0, 1.0e30, 1.0e30]);
    }

    #[test]
    fn a_family_decode_rule_binds_a_quantized_matmul_weight_as_transposed_f32() {
        let rows = quantized_rows(2, |row| row as f32 + 1.0);
        let bytes = gguf(&[("blk.0.ssm_alpha.weight", &[256, 2], GgmlType::Q4_K, &rows)]);
        let program = {
            let mut program = Vec::new();
            let activations = input(&mut program, "activations", &[1, 256]);
            let leaf = input(&mut program, "blk.0.ssm_alpha.weight", &[256, 2]);
            let product = append(
                &mut program,
                Op::Elementwise {
                    dtype: DType::Float32,
                    body: ScalarOp::Multiply,
                    operands: vec![
                        (activations, IndexMap::Affine(map::projection(3, &[0, 2]))),
                        (leaf, IndexMap::Affine(map::projection(3, &[2, 1]))),
                    ],
                    name: None,
                },
            );
            append(
                &mut program,
                Op::Reduce(Reduce {
                    dtype: DType::Float32,
                    body: ScalarOp::Add,
                    init: ReduceInit::Zero,
                    operand: product,
                    in_map: IndexMap::Affine(map::projection(3, &[0, 1, 2])),
                    out_map: IndexMap::Affine(map::projection(3, &[0, 1])),
                    keep: Keep::Reduce,
                    name: None,
                }),
            );
            program
        };

        let packed = bind(&bytes, &program, "llama").expect("binds without a rule");
        let decoded = bind(&bytes, &program, "qwen35moe").expect("binds with the family rule");

        assert!(matches!(packed.packed(), [(_, QuantizedBlock::Packed { codec: Codec::Q4K, .. })]));
        let [(_, values)] = decoded.owned() else {
            panic!("one owned weight, got {}", decoded.owned().len());
        };
        assert_eq!(values.len(), 512);
        assert!((values[0] - 1.0).abs() < 0.5 && (values[1] - 2.0).abs() < 0.5, "input row 0, both outputs: {:?}", &values[..2]);
    }
}
