//! safetensors weights -> the same [`BoundWeights`] structures
//! [`crate::bind_leaves::bind_program_leaves`] produces from a GGUF checkpoint --
//! HF-directory counterpart to that module, over the tensor manifest
//! [`proxima_safetensors::Manifest`] hands back instead of GGUF's
//! [`ParsedGguf`] tensor directory.
//!
//! Reuses `bind.rs`'s own decode/transpose/state machinery rather than
//! duplicating it: [`crate::bind::reinterpret_f32`], [`crate::bind::dequantize`],
//! [`crate::bind::aligned_f32_view`], [`crate::bind::transpose_out_in_to_in_out`],
//! and [`BoundWeights`] itself are all format-agnostic once handed raw bytes
//! and a declared dtype -- the only genuinely new work here is (a) mapping
//! HF's per-tensor names (`model.layers.0.self_attn.q_proj.weight`) to the
//! same weight slots GGUF's `blk.0.attn_q.weight` fills, and (b) decoding
//! [`DType::Float16`]/[`DType::BFloat16`] (safetensors' own on-disk element
//! types for an unquantized bf16/fp16 checkpoint), which `bind.rs`'s GGUF
//! path has never needed since no GGUF checkpoint this crate has evaluated
//! stores `F16`/`Bf16` weights.
//!
//! `std`-gated, matching `bind.rs`'s own `bind_program_leaves`: both walk a
//! whole tensor directory and need [`proxima_gguf::restack`]-free but still
//! platform-shaped support (an owned `Vec<f32>` per non-packed weight, sized
//! by the checkpoint), and [`crate::generate::LoadedModel`] (the only
//! caller either bind function needs to serve) is itself `std`-only.
//!
//! Mixture-of-experts HF checkpoints bind only where a binding profile names
//! the stored tensors: `bind_safetensors_program_leaves` reads a routed
//! stack's source from the profile (`profiles/binding/granitemoe.toml`, names
//! taken from the vendored IBM header) and binds it as
//! [`crate::bind::bind_moe_stacked_experts`] binds a GGUF stack: a bf16/f16 stack is packed, a whole tensor borrowed from
//! the file and a fused part borrowed as one file range per expert, never `f32`;
//! only an `F32` checkpoint decodes to the program's `[experts, in, out]` leaf. A
//! profile that names none, and the dense
//! `bind_all_weights_from_safetensors`, refuse with
//! [`InteropError::HfMoeWeightsUnsupported`] rather than guessing at a
//! naming convention.

use alloc::collections::BTreeSet;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_safetensors::{Manifest, TensorEntry};
use proxima_tensor::DType;
use proxima_tensor::cpu::QuantizedBlock;
use proxima_tensor::op::{Extent, Op};
use proxima_tensor::spec::{LayerSchedule, ModelDescriptor, RopePairing};

use crate::Codec;
use crate::bind::{
    BoundWeights, ModelHparams, aligned_f32_view, dequantize, reinterpret_f32,
    transpose_expert_stack, transpose_out_in_to_in_out,
};
use crate::bind_leaves::{Role, consumers_of, role_of};
use crate::error::InteropError;
use crate::profiles::{BindingProfile, TensorAlias};

fn find_entry<'manifest>(
    manifest: &'manifest Manifest,
    name: &str,
) -> Result<&'manifest proxima_safetensors::TensorEntry, InteropError> {
    manifest
        .tensor(name)
        .ok_or_else(|| InteropError::UnknownTensor { name: name.into() })
}

/// `entry.data_offsets` are relative to the first byte AFTER the header
/// (`Manifest`'s own doc, `proxima-safetensors/src/parser.rs`), never to
/// the start of the whole file -- `data_start` (`8 + header_len`, exactly
/// what a caller who just parsed `file_bytes`'s own header already knows,
/// see [`bind_all_weights_from_safetensors`]'s doc) is added here so every
/// caller of this module hands in the WHOLE file buffer, matching
/// `bind.rs`'s GGUF convention (`parsed.tensor_data_range`) instead of a
/// caller having to pre-slice the header off itself.
fn tensor_bytes<'file>(
    file_bytes: &'file [u8],
    data_start: u64,
    entry: &proxima_safetensors::TensorEntry,
) -> Result<&'file [u8], InteropError> {
    let start = data_start
        .checked_add(entry.data_offsets.0)
        .and_then(|value| usize::try_from(value).ok());
    let end = data_start
        .checked_add(entry.data_offsets.1)
        .and_then(|value| usize::try_from(value).ok());
    match (start, end) {
        (Some(start), Some(end)) => {
            file_bytes
                .get(start..end)
                .ok_or_else(|| InteropError::UnknownTensor {
                    name: format!(
                        "{} (byte range {start}..{end} outside a {}-byte buffer)",
                        entry.name,
                        file_bytes.len()
                    ),
                })
        }
        _ => Err(InteropError::UnknownTensor {
            name: format!(
                "{} (data_start {data_start} + declared offsets overflow usize)",
                entry.name
            ),
        }),
    }
}

/// A named program leaf's validated source bytes before evaluator conversion.
#[derive(Debug)]
pub struct SafetensorsTensorView<'manifest, 'file> {
    /// Name used by the lowered program.
    pub leaf: String,
    /// Exact name in the checkpoint tensor directory.
    pub name: &'manifest str,
    /// Checkpoint scalar format.
    pub dtype: DType,
    /// Checkpoint row-major axes.
    pub shape: &'manifest [u64],
    /// Bytes borrowed from the caller's file buffer.
    pub bytes: &'file [u8],
    role: Role,
    /// Axes the lowered program declares for the leaf, e.g. `[1024, 16, 64]` for an
    /// attention query over a stored `[1024, 1024]` matrix.
    pub program_axes: Vec<u64>,
}

/// Resolve program leaves through `binding`, validate their source layout and
/// byte ranges, and return borrowed checkpoint views. Runtime inputs are skipped.
///
/// A routed expert stack cannot be a borrowed view (a fused gate and up
/// tensor is cut per expert), so it is refused here: with
/// [`InteropError::HfMoeWeightsUnsupported`] when `binding` names no source for
/// it, otherwise as an unsupported gathered role. [`bind_safetensors_program_leaves`]
/// binds the stacks.
pub fn bind_safetensors_program_views<'manifest, 'file>(
    manifest: &'manifest Manifest,
    file_bytes: &'file [u8],
    data_start: u64,
    program: &[Op],
    binding: &BindingProfile,
    runtime_inputs: &[String],
) -> Result<Vec<SafetensorsTensorView<'manifest, 'file>>, InteropError> {
    let (views, stacks) = collect_program_views(
        manifest,
        file_bytes,
        data_start,
        program,
        binding,
        runtime_inputs,
    )?;
    match stacks.first() {
        None => Ok(views),
        Some((leaf, shape)) => {
            moe_stack_source(manifest, binding, leaf, shape)?;
            Err(InteropError::SafetensorsBindingUnsupported {
                leaf: leaf.clone(),
                feature: "gathered role".into(),
            })
        }
    }
}

type ExpertStackLeaf = (String, Vec<Extent>);

fn collect_program_views<'manifest, 'file>(
    manifest: &'manifest Manifest,
    file_bytes: &'file [u8],
    data_start: u64,
    program: &[Op],
    binding: &BindingProfile,
    runtime_inputs: &[String],
) -> Result<
    (
        Vec<SafetensorsTensorView<'manifest, 'file>>,
        Vec<ExpertStackLeaf>,
    ),
    InteropError,
> {
    refuse_mlx_quantized_header(manifest)?;
    let mut stacks = Vec::new();
    let consumers = consumers_of(program);
    let runtime_inputs: BTreeSet<&str> = runtime_inputs.iter().map(String::as_str).collect();
    let mut bound = BTreeSet::new();
    let mut views = Vec::new();
    let mut ranges_validated = false;

    for (index, operation) in program.iter().enumerate() {
        let Op::Input {
            shape,
            name: Some(leaf),
            ..
        } = operation
        else {
            continue;
        };
        if runtime_inputs.contains(leaf.as_str()) || !bound.insert(leaf.as_str()) {
            continue;
        }
        let role = role_of(program, &consumers, index, leaf, shape)?;
        if role == Role::Gathered {
            if routed_expert_count(leaf, shape).is_none() {
                return Err(InteropError::SafetensorsBindingUnsupported {
                    leaf: leaf.clone(),
                    feature: "gathered role".into(),
                });
            }
            stacks.push((leaf.clone(), shape.clone()));
            continue;
        }
        if !ranges_validated {
            validate_safetensors_data_ranges(manifest, file_bytes, data_start)?;
            ranges_validated = true;
        }
        let entry = safetensors_source(manifest, binding, leaf, shape)?;
        views.push(validated_view(
            file_bytes, data_start, leaf, shape, role, entry,
        )?);
    }

    for name in &binding.extra {
        if bound.insert(name.as_str()) {
            if !ranges_validated {
                validate_safetensors_data_ranges(manifest, file_bytes, data_start)?;
                ranges_validated = true;
            }
            let entry = find_entry(manifest, name)?;
            let shape = entry
                .shape
                .iter()
                .map(|axis| {
                    u32::try_from(*axis).map(Extent::Static).map_err(|_| {
                        InteropError::SafetensorsBindingUnsupported {
                            leaf: name.clone(),
                            feature: "extent larger than u32".into(),
                        }
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            views.push(validated_view(
                file_bytes,
                data_start,
                name,
                &shape,
                Role::Native,
                entry,
            )?);
        }
    }
    Ok((views, stacks))
}

/// Binds safetensors by walking the same lowered program leaves as the GGUF
/// binder, while using the existing HF scalar decode and layout routines.
/// Family differences belong in `binding`; the program determines each
/// tensor's storage role.
/// `manifest`, `file_bytes`, and `data_start` must come from the same parsed
/// safetensors file so offsets cannot select unrelated bytes.
pub fn bind_safetensors_program_leaves<'file>(
    manifest: &Manifest,
    file_bytes: &'file [u8],
    data_start: u64,
    program: &[Op],
    binding: &BindingProfile,
    runtime_inputs: &[String],
) -> Result<BoundWeights<'file>, InteropError> {
    let mut weights = BoundWeights::new(&[]);
    weights.resident_bytes = file_bytes.len();
    let (views, stacks) = collect_program_views(
        manifest,
        file_bytes,
        data_start,
        program,
        binding,
        runtime_inputs,
    )?;
    for view in &views {
        bind_safetensors_leaf(
            manifest,
            file_bytes,
            data_start,
            binding,
            view,
            &mut weights,
        )?;
    }
    for (leaf, shape) in &stacks {
        bind_safetensors_expert_stack(
            manifest,
            file_bytes,
            data_start,
            binding,
            leaf,
            shape,
            &mut weights,
        )?;
    }
    Ok(weights)
}

/// The routed-MoE arm. A routed leaf is a stacked expert projection
/// (`blk.N.ffn_{gate,up,down}_exps.weight`) or the router
/// (`blk.N.ffn_gate_inp.weight`); the expert count is read off its program
/// shape (`[experts, in, out]` for a stack, `[in, experts]` for the router).
fn routed_expert_count(leaf: &str, shape: &[Extent]) -> Option<u32> {
    let static_count = |extent: Option<&Extent>| match extent {
        Some(Extent::Static(count)) => Some(*count),
        _ => None,
    };
    if leaf.ends_with("_exps.weight") {
        static_count(shape.first())
    } else if leaf.ends_with("ffn_gate_inp.weight") {
        static_count(shape.last())
    } else {
        None
    }
}

/// `(stored name, part, of)` of the first alias of `binding` whose source the
/// checkpoint's `manifest` stores, for a routed leaf: a rename is part 0 of 1,
/// a part alias cuts `part` of `of` row groups. A profile none of whose
/// sources the checkpoint stores (every dense profile) refuses with
/// [`InteropError::HfMoeWeightsUnsupported`] rather than guessing at a name.
fn moe_stack_source(
    manifest: &Manifest,
    binding: &BindingProfile,
    leaf: &str,
    shape: &[Extent],
) -> Result<(String, u32, u32), InteropError> {
    binding
        .aliases_for(leaf)
        .find_map(|(alias, prefix)| {
            let (from, part, of) = match alias {
                TensorAlias::Part { from, part, of, .. } => (from, *part, *of),
                TensorAlias::Rename { from, .. } => (from, 0, 1),
                TensorAlias::Join { .. } => return None,
            };
            let name = format!("{prefix}{from}");
            manifest.tensor(&name).map(|_| (name, part, of))
        })
        .ok_or_else(|| InteropError::HfMoeWeightsUnsupported {
            expert_count: routed_expert_count(leaf, shape).unwrap_or(0),
        })
}

/// Cuts `part` of `of` equal row groups out of every expert of a decoded
/// `[experts, rows, width]` stack, giving `[experts, rows / of, width]`.
/// `width` counts elements of `T` per row, so bytes cut the same way by
/// passing the row's byte width; each expert's run is one contiguous
/// `extend_from_slice`.
fn expert_row_part<T: Copy>(
    stack: &[T],
    experts: usize,
    rows: usize,
    width: usize,
    part: usize,
    of: usize,
) -> Vec<T> {
    let part_len = rows / of * width;
    let mut cut = Vec::with_capacity(experts * part_len);
    for expert in 0..experts {
        let start = expert * rows * width + part * part_len;
        cut.extend_from_slice(&stack[start..start + part_len]);
    }
    cut
}

/// The packed [`Codec`] a stored element type binds as, `None` for the one
/// dtype the gathered kernel rejects (`run_reduce_quantized` refuses a
/// `Float32` weight block), which keeps the dequantize-and-transpose path
/// [`crate::bind::bind_moe_stacked_experts`] keeps for the same reason.
#[cfg(not(feature = "float-layout-dtype"))]
fn expert_stack_codec(dtype: DType) -> Option<Codec> {
    match dtype {
        DType::BFloat16 => Some(Codec::BFloat16),
        DType::Float16 => Some(Codec::Float16),
        _ => None,
    }
}

/// Selects the codec whose layout fields equal the stored element type's, among the two the
/// gathered kernel serves; every other layout keeps the dequantize-and-transpose path.
#[cfg(feature = "float-layout-dtype")]
fn expert_stack_codec(dtype: DType) -> Option<Codec> {
    let DType::Float(layout) = dtype else {
        return None;
    };
    [Codec::BFloat16, Codec::Float16].into_iter().find(|codec| {
        codec
            .layout()
            .is_some_and(|candidate| candidate.same_format(&layout))
    })
}

/// Binds one routed `[experts, in, out]` leaf the way
/// [`crate::bind::bind_moe_stacked_experts`] binds the same leaf from GGUF.
///
/// HF stores a stack as `[experts, out, in]`, `in` contiguous -- the physical
/// layout the gather kernel slices `expert * per_expert_bytes` out of -- so a
/// whole tensor (`of == 1`, `output_linear`) is borrowed straight out of
/// `file_bytes` as [`QuantizedBlock::Packed`], no copy and no transpose. A
/// fused part (`input_linear` -> gate, up) interleaves the parts per expert,
/// but each expert's part is itself one contiguous run of the file, so it is
/// bound as one borrowed range per expert in [`BoundWeights::expert_ranges`]:
/// still the stored 2-byte element, never `f32`, never transposed, never
/// copied. A `Float32` checkpoint decodes and transposes as before.
fn bind_safetensors_expert_stack<'file>(
    manifest: &Manifest,
    file_bytes: &'file [u8],
    data_start: u64,
    binding: &BindingProfile,
    leaf: &str,
    shape: &[Extent],
    weights: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    let (name, part, of) = moe_stack_source(manifest, binding, leaf, shape)?;
    let [
        Extent::Static(experts),
        Extent::Static(input),
        Extent::Static(output),
    ] = shape
    else {
        return Err(InteropError::SafetensorsBindingUnsupported {
            leaf: leaf.into(),
            feature: "expert stack extents".into(),
        });
    };
    let entry = find_entry(manifest, &name)?;
    let (experts, input, output) = (*experts as usize, *input as usize, *output as usize);
    let rows = output * of as usize;
    let expected = [experts as u64, rows as u64, input as u64];
    if of == 0 || part >= of || entry.shape != expected {
        return Err(InteropError::SafetensorsAxesMismatch {
            leaf: leaf.into(),
            tensor: name,
            leaf_axes: expected.to_vec(),
            tensor_axes: entry.shape.clone(),
        });
    }
    match expert_stack_codec(entry.dtype) {
        Some(codec) => {
            let bytes = tensor_bytes(file_bytes, data_start, entry)?;
            bind_packed_expert_stack(
                bytes,
                leaf,
                codec,
                [experts, rows, input],
                (part, of),
                weights,
            )
        }
        None => {
            let decoded = safetensors_tensor_as_f32(manifest, file_bytes, data_start, &name)?;
            let cut = expert_row_part(&decoded, experts, rows, input, part as usize, of as usize);
            let stacked = transpose_expert_stack(&cut, leaf, experts, output, input)?;
            weights.resident_bytes += stacked.len() * core::mem::size_of::<f32>();
            weights.owned.push((leaf.into(), stacked));
            Ok(())
        }
    }
}

const PACKED_ELEMENT_BYTES: usize = 2;

fn bind_packed_expert_stack<'file>(
    bytes: &'file [u8],
    leaf: &str,
    codec: Codec,
    [experts, rows, input]: [usize; 3],
    (part, of): (u32, u32),
    weights: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    let expected = experts * rows * input * PACKED_ELEMENT_BYTES;
    if bytes.len() != expected {
        return Err(InteropError::SafetensorsByteLengthMismatch {
            tensor: leaf.into(),
            expected: expected as u64,
            actual: bytes.len() as u64,
        });
    }
    if of == 1 {
        weights
            .packed
            .push((leaf.into(), QuantizedBlock::Packed { codec, bytes }));
        return Ok(());
    }
    let expert_bytes = rows * input * PACKED_ELEMENT_BYTES;
    let part_bytes = expert_bytes / of as usize;
    let ranges = (0..experts)
        .map(|expert| {
            let start = expert * expert_bytes + part as usize * part_bytes;
            &bytes[start..start + part_bytes]
        })
        .collect();
    weights.expert_ranges.push((leaf.into(), ranges, codec));
    Ok(())
}

fn validate_safetensors_data_ranges(
    manifest: &Manifest,
    file_bytes: &[u8],
    data_start: u64,
) -> Result<(), InteropError> {
    let data_start =
        usize::try_from(data_start).map_err(|_| InteropError::SafetensorsDataRangeInvalid {
            tensor: "<data section>".into(),
            reason: "data section start exceeds address space",
        })?;
    let data_length = file_bytes.len().checked_sub(data_start).ok_or_else(|| {
        InteropError::SafetensorsDataRangeInvalid {
            tensor: "<data section>".into(),
            reason: "data section starts beyond file buffer",
        }
    })? as u64;
    let mut ranges = Vec::with_capacity(manifest.tensors.len());
    for entry in &manifest.tensors {
        let (start, end) = entry.data_offsets;
        if start > end || end > data_length || end > manifest.declared_data_len() {
            return Err(InteropError::SafetensorsDataRangeInvalid {
                tensor: entry.name.clone(),
                reason: "offsets exceed the supplied data section",
            });
        }
        ranges.push((start, end, entry.name.as_str()));
    }
    ranges.sort_unstable_by_key(|(start, _, _)| *start);
    for pair in ranges.windows(2) {
        let (_, first_end, first_name) = pair[0];
        let (second_start, _, second_name) = pair[1];
        if first_end > second_start {
            return Err(InteropError::SafetensorsDataRangesOverlap {
                first: first_name.into(),
                second: second_name.into(),
            });
        }
    }
    Ok(())
}

fn safetensors_source<'manifest>(
    manifest: &'manifest Manifest,
    binding: &BindingProfile,
    leaf: &str,
    shape: &[Extent],
) -> Result<&'manifest proxima_safetensors::TensorEntry, InteropError> {
    if let Some(entry) = manifest.tensor(leaf) {
        return Ok(entry);
    }
    for (alias, prefix) in binding.aliases_for(leaf) {
        match alias {
            TensorAlias::Rename { from, .. } => {
                let source = format!("{prefix}{from}");
                if let Some(entry) = manifest.tensor(&source) {
                    return Ok(entry);
                }
            }
            TensorAlias::Part { .. } => {
                return Err(InteropError::SafetensorsBindingUnsupported {
                    leaf: leaf.into(),
                    feature: "part alias".into(),
                });
            }
            TensorAlias::Join { .. } => {
                return Err(InteropError::SafetensorsBindingUnsupported {
                    leaf: leaf.into(),
                    feature: "join alias".into(),
                });
            }
        }
    }
    Err(match routed_expert_count(leaf, shape) {
        Some(expert_count) => InteropError::HfMoeWeightsUnsupported { expert_count },
        None => InteropError::UnknownTensor { name: leaf.into() },
    })
}

fn validated_view<'manifest, 'file>(
    file_bytes: &'file [u8],
    data_start: u64,
    leaf: &str,
    shape: &[Extent],
    role: Role,
    entry: &'manifest TensorEntry,
) -> Result<SafetensorsTensorView<'manifest, 'file>, InteropError> {
    let mut program_axes = Vec::with_capacity(shape.len());
    for extent in shape {
        let Extent::Static(axis) = extent else {
            return Err(InteropError::SafetensorsBindingUnsupported {
                leaf: leaf.into(),
                feature: "symbolic weight extent".into(),
            });
        };
        program_axes.push(u64::from(*axis));
    }
    let axes_match = match role {
        Role::Native => native_view_matches(&entry.shape, &program_axes),
        Role::InOut => inout_view_matches(&entry.shape, &program_axes),
        Role::Gathered => false,
    };
    if !axes_match {
        return Err(InteropError::SafetensorsAxesMismatch {
            leaf: leaf.into(),
            tensor: entry.name.clone(),
            leaf_axes: program_axes,
            tensor_axes: entry.shape.clone(),
        });
    }

    let elements = entry
        .shape
        .iter()
        .try_fold(1u64, |product, axis| product.checked_mul(*axis));
    let expected = elements
        .and_then(|count| count.checked_mul(entry.dtype.size_bytes() as u64))
        .ok_or_else(|| InteropError::SafetensorsBindingUnsupported {
            leaf: leaf.into(),
            feature: "overflowing shape or byte size".into(),
        })?;
    let actual = entry
        .data_offsets
        .1
        .checked_sub(entry.data_offsets.0)
        .ok_or_else(|| InteropError::SafetensorsByteLengthMismatch {
            tensor: entry.name.clone(),
            expected,
            actual: 0,
        })?;
    if actual != expected {
        return Err(InteropError::SafetensorsByteLengthMismatch {
            tensor: entry.name.clone(),
            expected,
            actual,
        });
    }
    let bytes = tensor_bytes(file_bytes, data_start, entry)?;
    Ok(SafetensorsTensorView {
        leaf: leaf.into(),
        name: entry.name.as_str(),
        dtype: entry.dtype,
        shape: entry.shape.as_slice(),
        bytes,
        role,
        program_axes,
    })
}

/// An `[out, in]` file tensor viewed as the program's `[in axes.., out axes..]`
/// leaf: some split of the leaf's axes has the trailing product equal to the
/// file's rows and the leading product equal to its columns. A rank-3 attention
/// weight `[1024, 16, 64]` over a stored `[1024, 1024]` splits after axis 0;
/// a rank-2 leaf splits after axis 0 too, which is the plain transpose check.
fn inout_view_matches(stored: &[u64], program_axes: &[u64]) -> bool {
    let [rows, columns] = stored else {
        return false;
    };
    (1..program_axes.len()).any(|split| {
        program_axes[split..].iter().product::<u64>() == *rows
            && program_axes[..split].iter().product::<u64>() == *columns
    })
}

/// A stored tensor viewed as a program leaf of the same axes, or a one-element
/// `[1]` tensor viewed as the program's rank-0 scalar (gemma4's
/// `layer_scalar` is `[1]` in the file, GGUF `layer_output_scale.weight` the same).
fn native_view_matches(stored: &[u64], program_axes: &[u64]) -> bool {
    stored == program_axes || (program_axes.is_empty() && stored == [1])
}

fn bind_safetensors_leaf<'file>(
    manifest: &Manifest,
    file_bytes: &'file [u8],
    data_start: u64,
    binding: &BindingProfile,
    view: &SafetensorsTensorView<'_, 'file>,
    weights: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    let leaf = view.leaf.as_str();
    let role = view.role;
    let bytes = view.bytes;
    let dtype = view.dtype;
    if binding.decodes_to_f32(leaf) {
        let decoded = safetensors_tensor_as_f32(manifest, file_bytes, data_start, view.name)?;
        let output = if role == Role::InOut {
            transpose_out_in_to_in_out(
                &decoded,
                leaf,
                view.shape[0] as usize,
                view.shape[1] as usize,
            )?
        } else {
            decoded
        };
        weights.resident_bytes += output.len() * core::mem::size_of::<f32>();
        weights.owned.push((leaf.into(), output));
    } else if role == Role::InOut && dtype == DType::Float32 {
        let decoded = reinterpret_f32(bytes);
        let output = transpose_out_in_to_in_out(
            &decoded,
            leaf,
            view.shape[0] as usize,
            view.shape[1] as usize,
        )?;
        weights.resident_bytes += output.len() * core::mem::size_of::<f32>();
        weights.owned.push((leaf.into(), output));
    } else if role == Role::InOut {
        let packed =
            safetensors_tensor_as_packed_block(manifest, file_bytes, data_start, view.name)?;
        if matches!(packed, QuantizedBlock::Float32(_)) {
            return Err(InteropError::SafetensorsBindingUnsupported {
                leaf: leaf.into(),
                feature: "unexpected f32 packed matmul".into(),
            });
        }
        weights.packed.push((leaf.into(), packed));
    } else if dtype == DType::Float32 {
        if let Some(view) = aligned_f32_view(bytes) {
            weights.resident_bytes += core::mem::size_of_val(view);
            weights
                .packed
                .push((leaf.into(), QuantizedBlock::Float32(view)));
        } else {
            let decoded = reinterpret_f32(bytes);
            weights.resident_bytes += decoded.len() * core::mem::size_of::<f32>();
            weights.owned.push((leaf.into(), decoded));
        }
    } else {
        let decoded = safetensors_tensor_as_f32(manifest, file_bytes, data_start, view.name)?;
        weights.resident_bytes += decoded.len() * core::mem::size_of::<f32>();
        weights.owned.push((leaf.into(), decoded));
    }
    Ok(())
}

/// Zero-copy counterpart to [`safetensors_tensor_as_f32`], mirroring
/// [`crate::bind::gguf_tensor_as_packed_block`]: borrows `name`'s raw bytes
/// straight out of `file_bytes` for [`DType::Float32`] (alignment-checked,
/// same reasoning as the GGUF path), [`DType::Float16`], or
/// [`DType::BFloat16`] -- safetensors never block-quantizes, so every
/// mapped dtype here is a flat per-element scalar array with no sub-block
/// scale to unpack, unlike GGUF's `Q4_K`/`Q5_K`/`Q6_K`.
///
/// # Errors
///
/// [`InteropError::UnknownTensor`] if `name` isn't in `manifest`'s tensor
/// directory, or its declared byte range doesn't fit `file_bytes`;
/// [`InteropError::UndecodableSafetensorsDType`] for any other `DType`
/// (an integer type, or a quantized layout's own packed payload type).
pub(crate) fn safetensors_tensor_as_packed_block<'file>(
    manifest: &Manifest,
    file_bytes: &'file [u8],
    data_start: u64,
    name: &str,
) -> Result<QuantizedBlock<'file>, InteropError> {
    let entry = find_entry(manifest, name)?;
    let bytes = tensor_bytes(file_bytes, data_start, entry)?;
    if entry.dtype == DType::Float32 {
        aligned_f32_view(bytes)
            .map(QuantizedBlock::Float32)
            .ok_or_else(|| InteropError::MisalignedFloat32Tensor {
                tensor: entry.name.clone(),
            })
    } else if entry.dtype == DType::Float16 {
        Ok(QuantizedBlock::Packed {
            codec: Codec::Float16,
            bytes,
        })
    } else if entry.dtype == DType::BFloat16 {
        Ok(QuantizedBlock::Packed {
            codec: Codec::BFloat16,
            bytes,
        })
    } else {
        Err(InteropError::UndecodableSafetensorsDType {
            tensor: entry.name.clone(),
            dtype: entry.dtype,
        })
    }
}

/// Owned-decode counterpart, mirroring [`crate::bind::gguf_tensor_as_f32`]:
/// reads `name`'s bytes and decodes to an owned `Vec<f32>` -- a straight
/// reinterpret for `Float32`, and [`proxima_gguf::quant::f16`]/
/// [`proxima_gguf::quant::bf16`]'s own `dequantize` (reused, not
/// reimplemented -- both already have exactly this crate's `fn(&[u8], &mut
/// [f32]) -> Result<(), QuantError>` shape) for `Float16`/`BFloat16`.
///
/// # Errors
///
/// Same as [`safetensors_tensor_as_packed_block`], plus
/// [`InteropError::Quant`] if a `Float16`/`BFloat16` tensor's byte length is
/// not a whole number of 2-byte elements.
pub(crate) fn safetensors_tensor_as_f32(
    manifest: &Manifest,
    file_bytes: &[u8],
    data_start: u64,
    name: &str,
) -> Result<Vec<f32>, InteropError> {
    let entry = find_entry(manifest, name)?;
    let bytes = tensor_bytes(file_bytes, data_start, entry)?;
    let element_count = entry.shape.iter().product::<u64>() as usize;
    if entry.dtype == DType::Float32 {
        Ok(reinterpret_f32(bytes))
    } else if entry.dtype == DType::Float16 {
        dequantize(bytes, element_count, proxima_gguf::quant::f16::dequantize)
    } else if entry.dtype == DType::BFloat16 {
        dequantize(bytes, element_count, proxima_gguf::quant::bf16::dequantize)
    } else {
        Err(InteropError::UndecodableSafetensorsDType {
            tensor: entry.name.clone(),
            dtype: entry.dtype,
        })
    }
}

/// A learned 1-D scale (RMSNorm weight) or the embedding table (indexed by
/// row via `embedding_lookup`, never projected) -- same role as
/// [`crate::bind::bind_dense`], reused under a new name only because its
/// GGUF counterpart is `pub(crate)` to `bind.rs`'s own module, not this one.
///
/// `lookup_name` is the name HF's own manifest carries on disk
/// (`model.layers.0.input_layernorm.weight`); `store_name` is the name
/// [`BoundWeights`] keys this tensor under -- the forward program's own
/// GGUF-convention node name (`blk.0.attn_norm.weight`), since
/// [`crate::generate::LoadedModel::call`] matches every bound weight against
/// the compiled program purely by that string (`generate.rs`'s
/// `named_blocks.push((name.as_str(), ..))` loop), never by position. The
/// two names differ for every weight this module binds; kept as two
/// parameters rather than reusing one, so a caller cannot silently store a
/// tensor under the very on-disk name the program will never ask for.
///
/// Only `Float32` binds packed/zero-copy here -- a packed `Float16`/
/// `BFloat16` block lands in the interpreter's `quantized_weights` map
/// (`proxima_tensor::cpu`'s own `block_nodes`/`blocks` split), read ONLY by
/// the quantized-matmul reduce kernel (`run_reduce_quantized`); a gather
/// (`embedding_lookup`) or a plain elementwise scale (RMSNorm) instead reads
/// the plain per-node `buffers` array via `buffer_of`, which never looks
/// there at all. Confirmed the hard way against the real, downloaded
/// `HuggingFaceTB/SmolLM2-135M-Instruct/model.safetensors` (BF16 weights
/// throughout): binding `token_embd.weight` packed here produced
/// `NotLowerable { reason: "operand buffer missing at evaluation time" }`
/// the first time this path ever ran against a real bf16 HF checkpoint --
/// this is that fix. [`hf_bind_matmul_weight`] is the one binder that feeds
/// a genuine 2-D matmul operand, where the packed `Float16`/`BFloat16` path
/// (`matmul_f16_f32`/`matmul_bf16_f32`) is correct and stays.
fn hf_bind_dense<'file>(
    manifest: &Manifest,
    file_bytes: &'file [u8],
    data_start: u64,
    lookup_name: String,
    store_name: String,
    state: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    match safetensors_tensor_as_packed_block(manifest, file_bytes, data_start, &lookup_name) {
        Ok(block @ QuantizedBlock::Float32(borrowed)) => {
            state.resident_bytes += core::mem::size_of_val(borrowed);
            state.packed.push((store_name, block));
        }
        Ok(_) | Err(_) => {
            let decoded =
                safetensors_tensor_as_f32(manifest, file_bytes, data_start, &lookup_name)?;
            state.resident_bytes += decoded.len() * core::mem::size_of::<f32>();
            state.owned.push((store_name, decoded));
        }
    }
    Ok(())
}

/// Reorders `flat`'s (`[out_dim, in_dim]` row-major, `out_dim ==
/// head_count * head_dim`) rows from HF's own RoPE layout into the
/// interleaved layout [`proxima_tensor::spec`]'s RoPE math (`spec.rs`'s
/// `q_even`/`q_odd`, indexed `"2*i"`/`"2*i+1"`) consumes -- the same
/// per-head row permutation `llama.cpp`'s own `convert_hf_to_gguf.py`
/// applies (its `permute()`) before writing `blk.N.attn_q.weight`/
/// `blk.N.attn_k.weight` into a GGUF file, which is why `crate::bind`'s
/// GGUF path has never needed this: a GGUF checkpoint's bytes already
/// carry it.
///
/// HF/PyTorch's own rotary embedding (`transformers`' `rotate_half`)
/// treats a head's `head_dim` channels as two contiguous halves, `x1 =
/// x[..half]`, `x2 = x[half..]`, and rotates each `(x1[i], x2[i])` pair.
/// `spec.rs`'s RoPE instead rotates adjacent-channel pairs `(x[2*i],
/// x[2*i+1])` -- mathematically the same rotation, over a differently
/// numbered set of pairs, so it needs `x1[i]`/`x2[i]` delivered at channels
/// `2*i`/`2*i+1` instead of at `i`/`i+head_dim/2`. Per head:
///
/// ```text
/// permuted[head*head_dim + 2*i]     = original[head*head_dim + i]
/// permuted[head*head_dim + 2*i + 1] = original[head*head_dim + head_dim/2 + i]
/// ```
///
/// Confirmed against the real, downloaded `HuggingFaceTB/SmolLM2-135M-Instruct/model.safetensors`:
/// without this permutation, this crate's own forward pass matches
/// `llama.cpp`'s bit-for-bit at RoPE position 0 (where every pair's angle
/// is zero, so no permutation convention can matter) and diverges starting
/// at position 1 already inside layer 0 -- exactly the position dependence
/// this permutation, and only this permutation, explains.
/// Reorders interleaved RoPE pairs `(x[i], x[head_dim/2 + i])` into
/// adjacent-channel pairs `(x[2*i], x[2*i+1])`, the convention this crate's
/// own RoPE rotation expects; see the module doc above for the derivation.
#[must_use]
pub fn permute_rope_rows(
    flat: &[f32],
    head_count: usize,
    head_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    let half = head_dim / 2;
    let mut permuted = alloc::vec![0.0f32; flat.len()];
    for head in 0..head_count {
        for i in 0..half {
            let source_even = (head * head_dim + i) * in_dim;
            let source_odd = (head * head_dim + half + i) * in_dim;
            let dest_even = (head * head_dim + 2 * i) * in_dim;
            let dest_odd = (head * head_dim + 2 * i + 1) * in_dim;
            permuted[dest_even..dest_even + in_dim]
                .copy_from_slice(&flat[source_even..source_even + in_dim]);
            permuted[dest_odd..dest_odd + in_dim]
                .copy_from_slice(&flat[source_odd..source_odd + in_dim]);
        }
    }
    permuted
}

/// Whether `manifest` carries per-head QK-norm weights
/// (`model.layers.0.self_attn.q_norm.weight`) -- the HF-naming counterpart
/// of [`crate::bind::checkpoint_has_qk_norm`]'s GGUF-side presence check,
/// same "read the file, don't assume the shape from a name" reasoning.
///
/// Doubles as this module's RoPE-convention discriminator: every checkpoint
/// this crate has evaluated that carries QK-norm (Qwen3's own `q_norm`/
/// `k_norm`, `modeling_qwen3.py`'s `Qwen3Attention`) is also the family
/// `llama.cpp`'s own GGUF converter leaves in HF's native split-half RoPE
/// layout on disk (`LLM_ARCH_QWEN3`'s `rope_type = LLAMA_ROPE_TYPE_NEOX`,
/// see `proxima_tensor::spec`'s own `qk_norm.is_some()` RoPE-pairing
/// branch) -- confirmed downstream by tensor-by-tensor comparison of a
/// Qwen3-1.7B safetensors `attn_q`/`attn_k` bind against a known-good GGUF:
/// permuting rows lands `rel_err ~= 1.15` (scrambled), leaving them in
/// place lands `rel_err ~= 0.043` (quant noise). A family with neither
/// QK-norm nor the split-half convention (the dense decoders) still needs
/// [`permute_rope_rows`] to convert HF's `rotate_half` layout into this
/// crate's interleaved RoPE pairing, which is why this same presence check
/// gates both.
fn safetensors_has_qk_norm(manifest: &Manifest) -> bool {
    manifest
        .tensor("model.layers.0.self_attn.q_norm.weight")
        .is_some()
}

/// [`hf_bind_matmul_weight`]'s counterpart for `q_proj`/`k_proj` alone:
/// always decodes to an owned `[out_dim, in_dim]` buffer (never takes the
/// packed zero-copy path -- [`permute_rope_rows`]'s reorder is a physical
/// byte move a borrowed packed block cannot express) and applies
/// [`permute_rope_rows`] before the same transpose [`hf_bind_matmul_weight`]
/// already does, but ONLY for a `rotate_half` family (`permute_rows`,
/// [`safetensors_has_qk_norm`]'s doc names which families set this) -- a
/// split-half/NEOX family (Qwen3) already carries its rows in this crate's
/// expected interleaved layout, and permuting them again would scramble
/// `attn_q`/`attn_k`.
// one weight's own real shape (lookup/store names, head geometry, in_dim)
// plus the file/state every binder in this module threads through -- the
// same shape `proxima_tensor::spec::append_gqa_cached_routed_layer`
// carries its own `#[allow(clippy::too_many_arguments)]` for.
#[allow(clippy::too_many_arguments)]
fn hf_bind_rope_weight<'file>(
    manifest: &Manifest,
    file_bytes: &'file [u8],
    data_start: u64,
    lookup_name: String,
    store_name: String,
    head_count: usize,
    head_dim: usize,
    in_dim: usize,
    permute_rows: bool,
    state: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    let decoded = safetensors_tensor_as_f32(manifest, file_bytes, data_start, &lookup_name)?;
    let rows = if permute_rows {
        permute_rope_rows(&decoded, head_count, head_dim, in_dim)
    } else {
        decoded
    };
    let out_dim = head_count * head_dim;
    state.resident_bytes += rows.len() * core::mem::size_of::<f32>();
    state.owned.push((
        store_name,
        transpose_out_in_to_in_out(&rows, &lookup_name, out_dim, in_dim)?,
    ));
    Ok(())
}

/// A 2-D projection weight bound as one matmul operand -- HF/PyTorch's
/// `nn.Linear` weight is `[out_features, in_features]`, row-major, the
/// identical on-disk convention GGUF's own `[out, in]` layout uses (see
/// [`crate::bind::transpose_out_in_to_in_out`]'s doc), so the same
/// dequantize-then-transpose fallback applies unchanged. Packed
/// `Float16`/`BFloat16` binds straight through zero-copy: [`QuantizedBlock`]'s
/// own matmul kernel (`proxima_tensor::cpu::matmul_f16_f32`/`matmul_bf16_f32`)
/// walks the packed bytes in their native `[out, in]` order directly, the
/// same way a packed `Q4_K` operand does, so no transpose is needed on that
/// path at all.
///
/// `Float32` is the one codec [`safetensors_tensor_as_packed_block`] can
/// decode but this function does NOT hand to the packed path, mirroring
/// [`crate::bind::bind_matmul_weight_as`]'s own fix: neither the CPU generic
/// evaluator (`proxima_tensor::cpu::evaluate_quantized_with_scratch`'s
/// `QuantizedBlock::Float32` arm binds as a plain buffer, never entering
/// `run_reduce_quantized`'s byte-native bypass) nor the GPU driver
/// (`omega::metal.rs`'s own `packed_operands_of` explicitly excludes
/// `QuantizedBlock::Float32` from `correct_packed_matmul_layouts`'s node set)
/// has any mechanism that rewrites a packed `F32` operand's layout from
/// safetensors' native `[out, in]` to the `[in, out]` every consuming
/// `IndexMap` declares. So `Float32` always takes the dequantize-then-transpose
/// path instead, the same as any other undecodable-packed dtype -- only the
/// OWNED f32 fallback needs the transpose, exactly mirroring
/// [`crate::bind::bind_matmul_weight`]/[`crate::bind::bind_matmul_weight_as`].
///
/// `lookup_name`/`store_name` split for the same reason [`hf_bind_dense`]'s
/// doc explains -- and this split is also what lets a tied-embedding
/// checkpoint bind its `output.weight` program node from
/// `model.embed_tokens.weight`'s on-disk bytes (`lookup_name`) while still
/// storing the result under `output.weight` (`store_name`): the identical
/// transpose this function already applies to any other `[out, in]`
/// projection weight, no new bind path.
// two names (lookup/store, see this doc) plus the tensor's own shape
// (out_dim/in_dim) plus the file/state every binder in this module threads
// through -- the same real parameter count `bind_moe_expert_weights`
// (`bind.rs`) carries for the identical reason.
#[allow(clippy::too_many_arguments)]
fn hf_bind_matmul_weight<'file>(
    manifest: &Manifest,
    file_bytes: &'file [u8],
    data_start: u64,
    lookup_name: String,
    store_name: String,
    out_dim: usize,
    in_dim: usize,
    state: &mut BoundWeights<'file>,
) -> Result<(), InteropError> {
    match safetensors_tensor_as_packed_block(manifest, file_bytes, data_start, &lookup_name) {
        Ok(QuantizedBlock::Float32(_)) | Err(_) => {
            let decoded =
                safetensors_tensor_as_f32(manifest, file_bytes, data_start, &lookup_name)?;
            state.resident_bytes += decoded.len() * core::mem::size_of::<f32>();
            state.owned.push((
                store_name,
                transpose_out_in_to_in_out(&decoded, &lookup_name, out_dim, in_dim)?,
            ));
        }
        Ok(block) => state.packed.push((store_name, block)),
    }
    Ok(())
}

/// Every dense-checkpoint weight name `crate::bind_leaves::bind_program_leaves`'s
/// GGUF loop binds, HF's own naming instead -- the standard dense-decoder
/// Qwen `transformers` layout (`model.layers.{layer}.*`), the convention
/// every dense checkpoint on HuggingFace this crate has been checked against
/// uses, as opposed to GGUF's `blk.{n}.*` convention (see
/// [`node_names`]).
pub mod names {
    use alloc::format;
    use alloc::string::String;

    #[must_use]
    pub fn embed_tokens() -> String {
        "model.embed_tokens.weight".into()
    }
    #[must_use]
    pub fn input_layernorm(layer: u32) -> String {
        format!("model.layers.{layer}.input_layernorm.weight")
    }
    #[must_use]
    pub fn post_attention_layernorm(layer: u32) -> String {
        format!("model.layers.{layer}.post_attention_layernorm.weight")
    }
    /// The second norm a sandwich-normed block applies to its attention output
    /// before the residual add (`input_layernorm_2`).
    #[must_use]
    pub fn input_layernorm_2(layer: u32) -> String {
        format!("model.layers.{layer}.input_layernorm_2.weight")
    }
    /// The second norm a sandwich-normed block applies to its FFN output
    /// before the residual add (`post_attention_layernorm_2`).
    #[must_use]
    pub fn post_attention_layernorm_2(layer: u32) -> String {
        format!("model.layers.{layer}.post_attention_layernorm_2.weight")
    }
    #[must_use]
    pub fn q_norm(layer: u32) -> String {
        format!("model.layers.{layer}.self_attn.q_norm.weight")
    }
    #[must_use]
    pub fn k_norm(layer: u32) -> String {
        format!("model.layers.{layer}.self_attn.k_norm.weight")
    }
    #[must_use]
    pub fn q_proj(layer: u32) -> String {
        format!("model.layers.{layer}.self_attn.q_proj.weight")
    }
    #[must_use]
    pub fn k_proj(layer: u32) -> String {
        format!("model.layers.{layer}.self_attn.k_proj.weight")
    }
    #[must_use]
    pub fn v_proj(layer: u32) -> String {
        format!("model.layers.{layer}.self_attn.v_proj.weight")
    }
    #[must_use]
    pub fn o_proj(layer: u32) -> String {
        format!("model.layers.{layer}.self_attn.o_proj.weight")
    }
    #[must_use]
    pub fn gate_proj(layer: u32) -> String {
        format!("model.layers.{layer}.mlp.gate_proj.weight")
    }
    #[must_use]
    pub fn up_proj(layer: u32) -> String {
        format!("model.layers.{layer}.mlp.up_proj.weight")
    }
    #[must_use]
    pub fn down_proj(layer: u32) -> String {
        format!("model.layers.{layer}.mlp.down_proj.weight")
    }
    #[must_use]
    pub fn final_norm() -> String {
        "model.norm.weight".into()
    }
    #[must_use]
    pub fn lm_head() -> String {
        "lm_head.weight".into()
    }
}

/// The forward program's own node names -- exactly the GGUF-convention
/// strings [`proxima_tensor::spec::gqa_cached_forward_program_with_experts`]
/// builds its `Op::Input` leaves with (`spec.rs`'s own `blk.{layer}.*`/
/// `token_embd.weight`/`output_norm.weight`/`output.weight` literals), never
/// [`names`]'s HF-convention strings. `LoadedModel`'s `Pipe::call` impl
/// matches every bound weight against the compiled program purely by name
/// (see `hf_bind_dense`'s doc), so every [`BoundWeights`] entry this module
/// produces MUST be stored under one of these, regardless of what the
/// on-disk tensor was called.
pub mod node_names {
    use alloc::format;
    use alloc::string::String;

    #[must_use]
    pub fn token_embd() -> String {
        "token_embd.weight".into()
    }
    #[must_use]
    pub fn attn_norm(layer: u32) -> String {
        format!("blk.{layer}.attn_norm.weight")
    }
    #[must_use]
    pub fn ffn_norm(layer: u32) -> String {
        format!("blk.{layer}.ffn_norm.weight")
    }
    #[must_use]
    pub fn post_attention_norm(layer: u32) -> String {
        format!("blk.{layer}.post_attention_norm.weight")
    }
    #[must_use]
    pub fn post_ffw_norm(layer: u32) -> String {
        format!("blk.{layer}.post_ffw_norm.weight")
    }
    #[must_use]
    pub fn attn_q_norm(layer: u32) -> String {
        format!("blk.{layer}.attn_q_norm.weight")
    }
    #[must_use]
    pub fn attn_k_norm(layer: u32) -> String {
        format!("blk.{layer}.attn_k_norm.weight")
    }
    #[must_use]
    pub fn attn_q(layer: u32) -> String {
        format!("blk.{layer}.attn_q.weight")
    }
    #[must_use]
    pub fn attn_k(layer: u32) -> String {
        format!("blk.{layer}.attn_k.weight")
    }
    #[must_use]
    pub fn attn_v(layer: u32) -> String {
        format!("blk.{layer}.attn_v.weight")
    }
    #[must_use]
    pub fn attn_output(layer: u32) -> String {
        format!("blk.{layer}.attn_output.weight")
    }
    #[must_use]
    pub fn ffn_gate(layer: u32) -> String {
        format!("blk.{layer}.ffn_gate.weight")
    }
    #[must_use]
    pub fn ffn_up(layer: u32) -> String {
        format!("blk.{layer}.ffn_up.weight")
    }
    #[must_use]
    pub fn ffn_down(layer: u32) -> String {
        format!("blk.{layer}.ffn_down.weight")
    }
    #[must_use]
    pub fn output_norm() -> String {
        "output_norm.weight".into()
    }
    #[must_use]
    pub fn output_weight() -> String {
        "output.weight".into()
    }
}

/// HF/safetensors counterpart to [`crate::bind_leaves::bind_program_leaves`]: binds
/// every weight [`proxima_tensor::spec::gqa_cached_forward_program`]
/// needs out of a single safetensors buffer's [`Manifest`], using HF's own
/// per-tensor names ([`names`]) in place of GGUF's `blk.{n}.*` convention.
///
/// `file_bytes` is the WHOLE safetensors buffer (matching `bind.rs`'s GGUF
/// convention), and `data_start` is the byte offset where tensor data
/// begins -- `8 + header_len`, exactly what a caller already computed while
/// parsing `file_bytes`'s own header into `manifest` (the same value
/// `bf16_real_checkpoint_parity.rs`'s own `real_manifest` helper derives by
/// hand for the identical reason: [`Manifest`]/[`proxima_safetensors::TensorEntry`]'s
/// own `data_offsets` are relative to the byte AFTER the header, never to
/// the start of the file, per that type's own doc).
///
/// Single-shard only: a real multi-file HF checkpoint (this crate's own
/// evidence, `~/.lmstudio/models/lmstudio-community/Qwen3-30B-A3B-MLX-4bit/`,
/// ships four `model-0000N-of-00004.safetensors` shards plus a
/// `model.safetensors.index.json` mapping each tensor name to its shard) is
/// NOT assembled here -- a caller with several shards' `(Manifest,
/// file_bytes, data_start)` triples calls this once per shard it owns and
/// merges the resulting [`BoundWeights`], or (unimplemented) this crate
/// grows an index-aware entry point that resolves a name to its shard
/// first. That index-following step is the concrete piece still missing
/// between this function and a full multi-shard HF directory load.
///
/// # Errors
///
/// [`InteropError::HfMoeWeightsUnsupported`] if `architecture.expert_count`
/// is nonzero; otherwise whatever [`hf_bind_dense`]/[`hf_bind_matmul_weight`]
/// can fail with (see [`safetensors_tensor_as_f32`]/
/// [`safetensors_tensor_as_packed_block`]).
pub(crate) fn bind_all_weights_from_safetensors<'file>(
    manifest: &Manifest,
    file_bytes: &'file [u8],
    data_start: u64,
    architecture: &ModelHparams,
) -> Result<BoundWeights<'file>, InteropError> {
    bind_dense_decoder_weights(manifest, file_bytes, data_start, architecture, None)
}

/// [`bind_all_weights_from_safetensors`] for a block schedule the caller's
/// `descriptor` states instead of the one a dense decoder implies: each layer's
/// RoPE pairing says whether its Q/K rows are reordered (adjacent pairing
/// reorders HF's `rotate_half` rows; split-half pairing keeps them as stored),
/// its `qk_norm` whether the per-head norm weights bind, and its FFN norm flags
/// whether the second branch norms of a sandwich-normed block bind. Nothing is
/// inferred from which tensors the file happens to hold.
///
/// # Errors
///
/// Whatever [`bind_all_weights_from_safetensors`] can fail with, plus
/// [`InteropError::UnknownTensor`] for a norm the schedule needs and the file
/// lacks.
pub(crate) fn bind_all_weights_from_safetensors_with_descriptor<'file>(
    manifest: &Manifest,
    file_bytes: &'file [u8],
    data_start: u64,
    architecture: &ModelHparams,
    descriptor: &ModelDescriptor,
) -> Result<BoundWeights<'file>, InteropError> {
    bind_dense_decoder_weights(
        manifest,
        file_bytes,
        data_start,
        architecture,
        Some(&descriptor.layers),
    )
}

fn bind_dense_decoder_weights<'file>(
    manifest: &Manifest,
    file_bytes: &'file [u8],
    data_start: u64,
    architecture: &ModelHparams,
    schedule: Option<&[LayerSchedule]>,
) -> Result<BoundWeights<'file>, InteropError> {
    refuse_mlx_quantized_header(manifest)?;
    if architecture.expert_count != 0 {
        return Err(InteropError::HfMoeWeightsUnsupported {
            expert_count: architecture.expert_count,
        });
    }

    let mut state = BoundWeights {
        resident_bytes: file_bytes.len(),
        owned: Vec::new(),
        packed: Vec::new(),
        packed_owned: Vec::new(),
        expert_ranges: Vec::new(),
        precision: &[],
    };

    let embedding = architecture.embedding as usize;
    let kv_dim = architecture.kv_heads as usize * architecture.head_dim as usize;
    let feed_forward = architecture.feed_forward as usize;
    let vocab = architecture.vocab as usize;
    let permute_rope_rows = !safetensors_has_qk_norm(manifest);

    hf_bind_dense(
        manifest,
        file_bytes,
        data_start,
        names::embed_tokens(),
        node_names::token_embd(),
        &mut state,
    )?;

    for layer in 0..architecture.block_count {
        let entry = schedule.and_then(|layers| layers.get(layer as usize));
        let permute_layer_rows = entry.map_or(permute_rope_rows, |entry| {
            entry.attention.rope_pairing == RopePairing::Interleaved
        });
        hf_bind_dense(
            manifest,
            file_bytes,
            data_start,
            names::input_layernorm(layer),
            node_names::attn_norm(layer),
            &mut state,
        )?;
        hf_bind_dense(
            manifest,
            file_bytes,
            data_start,
            names::post_attention_layernorm(layer),
            node_names::ffn_norm(layer),
            &mut state,
        )?;
        if entry.is_some_and(|entry| entry.attention.qk_norm) {
            hf_bind_dense(
                manifest,
                file_bytes,
                data_start,
                names::q_norm(layer),
                node_names::attn_q_norm(layer),
                &mut state,
            )?;
            hf_bind_dense(
                manifest,
                file_bytes,
                data_start,
                names::k_norm(layer),
                node_names::attn_k_norm(layer),
                &mut state,
            )?;
        }
        if entry.is_some_and(|entry| entry.ffn.post_attention_norm) {
            hf_bind_dense(
                manifest,
                file_bytes,
                data_start,
                names::input_layernorm_2(layer),
                node_names::post_attention_norm(layer),
                &mut state,
            )?;
        }
        if entry.is_some_and(|entry| entry.ffn.exclusive_dense_post_norm) {
            hf_bind_dense(
                manifest,
                file_bytes,
                data_start,
                names::post_attention_layernorm_2(layer),
                node_names::post_ffw_norm(layer),
                &mut state,
            )?;
        }
        hf_bind_rope_weight(
            manifest,
            file_bytes,
            data_start,
            names::q_proj(layer),
            node_names::attn_q(layer),
            architecture.query_heads as usize,
            architecture.head_dim as usize,
            embedding,
            permute_layer_rows,
            &mut state,
        )?;
        hf_bind_rope_weight(
            manifest,
            file_bytes,
            data_start,
            names::k_proj(layer),
            node_names::attn_k(layer),
            architecture.kv_heads as usize,
            architecture.head_dim as usize,
            embedding,
            permute_layer_rows,
            &mut state,
        )?;
        hf_bind_matmul_weight(
            manifest,
            file_bytes,
            data_start,
            names::v_proj(layer),
            node_names::attn_v(layer),
            kv_dim,
            embedding,
            &mut state,
        )?;
        hf_bind_matmul_weight(
            manifest,
            file_bytes,
            data_start,
            names::o_proj(layer),
            node_names::attn_output(layer),
            embedding,
            embedding,
            &mut state,
        )?;
        hf_bind_matmul_weight(
            manifest,
            file_bytes,
            data_start,
            names::gate_proj(layer),
            node_names::ffn_gate(layer),
            feed_forward,
            embedding,
            &mut state,
        )?;
        hf_bind_matmul_weight(
            manifest,
            file_bytes,
            data_start,
            names::up_proj(layer),
            node_names::ffn_up(layer),
            feed_forward,
            embedding,
            &mut state,
        )?;
        hf_bind_matmul_weight(
            manifest,
            file_bytes,
            data_start,
            names::down_proj(layer),
            node_names::ffn_down(layer),
            embedding,
            feed_forward,
            &mut state,
        )?;
    }

    hf_bind_dense(
        manifest,
        file_bytes,
        data_start,
        names::final_norm(),
        node_names::output_norm(),
        &mut state,
    )?;

    // A tied checkpoint (HF's `tie_word_embeddings: true`, e.g. real
    // SmolLM2-135M-Instruct) ships no `lm_head.weight` tensor at all -- its
    // token-embedding table doubles as the LM head. Binding the SAME
    // on-disk tensor at the `output.weight` program node needs no new bind
    // path: `hf_bind_matmul_weight` already dequantizes-then-transposes (or
    // binds packed) any `[out, in]`-shaped weight, and `embed_tokens`'s own
    // on-disk shape (`[vocab, embedding]`) is exactly that for
    // `out_dim = vocab, in_dim = embedding` -- the same shape `lm_head.weight`
    // would have carried on an untied checkpoint.
    let output_lookup_name = if architecture.tied_embeddings {
        names::embed_tokens()
    } else {
        names::lm_head()
    };
    hf_bind_matmul_weight(
        manifest,
        file_bytes,
        data_start,
        output_lookup_name,
        node_names::output_weight(),
        vocab,
        embedding,
        &mut state,
    )?;
    Ok(state)
}

mod mlx_refusal;

mod shards;

pub use shards::{HfShard, bind_safetensors_shards};

pub use mlx_refusal::{refuse_mlx_quantized, refuse_mlx_quantized_header};

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod matrix_lfm_bind;

#[cfg(test)]
mod granitemoe_binding;

#[cfg(test)]
mod expert_views;
#[cfg(test)]
mod inout_views;
#[cfg(test)]
mod rope_rows;
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::vec;

    #[cfg(feature = "float-layout-dtype")]
    use proxima_primitives::float_layout::{FloatLayout, SignEncoding, Specials};
    use proxima_safetensors::{SafetensorsModel, TensorPayload, write_complete};

    use super::*;

    #[cfg(feature = "float-layout-dtype")]
    #[test]
    fn float_layout_packed_hf_bind_maps_dtype_to_codec_by_fields() {
        let binary16_by_fields = FloatLayout {
            exponent_bits: 5,
            mantissa_bits: 10,
            specials: Specials::InfinityOnOverflow,
            bias_override: Some(15),
            sign: SignEncoding::Magnitude,
        };
        let brain16_by_fields = FloatLayout {
            exponent_bits: 8,
            mantissa_bits: 7,
            specials: Specials::InfinityOnOverflow,
            bias_override: None,
            sign: SignEncoding::Magnitude,
        };
        let unnamed = FloatLayout {
            exponent_bits: 4,
            mantissa_bits: 3,
            specials: Specials::None,
            bias_override: None,
            sign: SignEncoding::Magnitude,
        };

        assert_eq!(
            expert_stack_codec(DType::Float(binary16_by_fields)),
            Some(Codec::Float16)
        );
        assert_eq!(
            expert_stack_codec(DType::Float(brain16_by_fields)),
            Some(Codec::BFloat16)
        );
        assert_eq!(expert_stack_codec(DType::Float(FloatLayout::BF32)), None);
        assert_eq!(expert_stack_codec(DType::Float(unnamed)), None);
        assert_eq!(expert_stack_codec(DType::Int32), None);
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>()
    }

    /// `8 + header_len`, read off a real written safetensors buffer's own
    /// 8-byte little-endian length prefix -- the `data_start` every
    /// `bind_all_weights_from_safetensors` call in this module needs, since
    /// `Manifest`'s own `data_offsets` are relative to this point (see
    /// `bind_all_weights_from_safetensors`'s own doc), not to byte 0.
    fn header_data_start(file_bytes: &[u8]) -> u64 {
        let mut length_prefix = [0u8; 8];
        length_prefix.copy_from_slice(&file_bytes[..8]);
        8 + u64::from_le_bytes(length_prefix)
    }

    /// The smallest real dense architecture this crate's weight names can
    /// bind: 1 layer, embedding=4, feed_forward=8, 2 query heads, 1 kv head
    /// (GQA), head_dim=2, vocab=3 -- every dimension distinct so a
    /// transposed or mis-shaped bind would produce a length mismatch, not
    /// silently pass.
    fn tiny_dense_architecture() -> ModelHparams {
        ModelHparams {
            vocab: 3,
            embedding: 4,
            feed_forward: 8,
            query_heads: 2,
            kv_heads: 1,
            kv_heads_by_layer: vec![1],
            head_dim: 2,
            block_count: 1,
            expert_count: 0,
            expert_used_count: 0,
            rope_freq_base: proxima_tensor::sized::ROPE_FREQ_BASE_DEFAULT,
            rms_epsilon: 1e-5,
            tied_embeddings: false,
            family: String::new(),
            sliding_rope: None,
        }
    }

    /// Builds a real (in-memory-encoded, byte-for-byte real safetensors wire
    /// format via [`write_complete`]) checkpoint carrying every weight name
    /// [`bind_all_weights_from_safetensors`] looks up for
    /// [`tiny_dense_architecture`], each an `F32` tensor of the exact shape
    /// the architecture implies.
    fn tiny_dense_checkpoint() -> Vec<u8> {
        let architecture = tiny_dense_architecture();
        let embedding = architecture.embedding as usize;
        let feed_forward = architecture.feed_forward as usize;
        let kv_dim = architecture.kv_heads as usize * architecture.head_dim as usize;
        let vocab = architecture.vocab as usize;

        let mut owned_data: Vec<(String, Vec<u8>)> = Vec::new();
        let mut push = |name: String, elements: usize| {
            let values: Vec<f32> = (0..elements).map(|index| index as f32 * 0.5).collect();
            owned_data.push((name, f32_bytes(&values)));
        };

        push(names::embed_tokens(), vocab * embedding);
        push(names::input_layernorm(0), embedding);
        push(names::post_attention_layernorm(0), embedding);
        push(names::q_proj(0), embedding * embedding);
        push(names::k_proj(0), kv_dim * embedding);
        push(names::v_proj(0), kv_dim * embedding);
        push(names::o_proj(0), embedding * embedding);
        push(names::gate_proj(0), feed_forward * embedding);
        push(names::up_proj(0), feed_forward * embedding);
        push(names::down_proj(0), embedding * feed_forward);
        push(names::final_norm(), embedding);
        push(names::lm_head(), vocab * embedding);

        let tensors = owned_data
            .iter()
            .map(|(name, bytes)| {
                let elements = bytes.len() / 4;
                TensorPayload {
                    name: name.clone(),
                    dtype: DType::Float32,
                    shape: vec![elements as u64],
                    data: bytes.as_slice(),
                }
            })
            .collect();

        write_complete(&SafetensorsModel {
            tensors,
            metadata: alloc::collections::BTreeMap::new(),
        })
        .expect("writes a real safetensors buffer")
    }

    /// Every weight name [`bind_all_weights_from_safetensors`]'s loop looks
    /// up is present and binds without error, over a real (wire-format
    /// encoded) safetensors buffer -- proves the HF name mapping matches
    /// what a real dense checkpoint's own tensor directory would carry, not
    /// just that the function compiles.
    #[test]
    fn binds_every_dense_weight_name_from_a_real_safetensors_buffer() {
        let file_bytes = tiny_dense_checkpoint();
        let manifest = proxima_safetensors::parse_complete(&file_bytes)
            .expect("parses real safetensors buffer");
        let architecture = tiny_dense_architecture();
        let data_start = header_data_start(&file_bytes);

        let bound =
            bind_all_weights_from_safetensors(&manifest, &file_bytes, data_start, &architecture)
                .expect("binds every dense weight this architecture's forward program needs");

        // 12 tensors total: embed_tokens, 2 norms, 4 attention projections,
        // 3 ffn projections, final_norm, lm_head -- every one bound as an
        // owned f32 buffer here since none of this fixture's tensors are
        // 4-byte-misaligned within the written buffer.
        assert_eq!(
            bound.owned.len() + bound.packed.len(),
            12,
            "every named weight must bind, none silently skipped"
        );

        // Every stored key must be the forward program's own GGUF-convention
        // node name, NEVER the HF on-disk name this checkpoint's manifest
        // actually carries -- `LoadedModel::call` matches bound weights
        // against the compiled program purely by string (`generate.rs`'s
        // `named_blocks` loop), so a name mismatch here would bind silently
        // and then fail (or worse, mis-match) only once a real forward pass
        // runs.
        let stored_names: alloc::vec::Vec<&str> = bound
            .owned
            .iter()
            .map(|(name, _)| name.as_str())
            .chain(bound.packed.iter().map(|(name, _)| name.as_str()))
            .collect();
        for expected in [
            "token_embd.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
            "output_norm.weight",
            "output.weight",
        ] {
            assert!(
                stored_names.contains(&expected),
                "expected the GGUF-convention node name {expected:?} among stored keys {stored_names:?}"
            );
        }
        for hf_name in [
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.norm.weight",
            "lm_head.weight",
        ] {
            assert!(
                !stored_names.contains(&hf_name),
                "the HF on-disk name {hf_name:?} must never be the STORED key -- only the lookup key"
            );
        }
    }

    /// [`tiny_dense_architecture`]'s `head_dim = 2` gives [`permute_rope_rows`]
    /// exactly one pair per head (`half = 1`), where the interleaved
    /// destination indices `2*i`/`2*i+1` and the split-half source indices
    /// `i`/`half+i` both reduce to `0`/`1` -- the permutation is the
    /// identity either way, so no fixture built on that architecture can
    /// tell a correct interleave apart from a regression that dropped the
    /// permutation entirely. `head_dim = 4` (two pairs per head) breaks that
    /// coincidence: the interleaved layout `[x1[0], x2[0], x1[1], x2[1]]` is
    /// a real transposition of the split-half input `[x1[0], x1[1], x2[0],
    /// x2[1]]`, so a wrong-convention regression (skipping the permutation,
    /// or swapping which half each destination slot reads) changes the
    /// asserted output.
    #[test]
    fn permute_rope_rows_interleaves_split_half_pairs_when_two_pairs_exist() {
        const HEAD_COUNT: usize = 2;
        const HEAD_DIM: usize = 4;
        const IN_DIM: usize = 1;

        // per head, split-half layout: [x1[0], x1[1], x2[0], x2[1]].
        let original: Vec<f32> = (0..HEAD_COUNT * HEAD_DIM)
            .map(|index| index as f32)
            .collect();

        let permuted = permute_rope_rows(&original, HEAD_COUNT, HEAD_DIM, IN_DIM);

        // per head, interleaved layout: [x1[0], x2[0], x1[1], x2[1]].
        let expected: Vec<f32> = vec![0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0];
        assert_eq!(
            permuted, expected,
            "two pairs per head must interleave x1/x2, not pass the split-half layout through"
        );
        assert_ne!(
            permuted, original,
            "a wrong convention that no-ops the permutation must be distinguishable at two pairs per head"
        );
    }

    /// [`bind_all_weights_from_safetensors`]'s family gate, proven through
    /// the real bind pipeline over [`tiny_dense_architecture`]'s `head_dim =
    /// 4` (two pairs per head, so a wrong-convention regression is
    /// distinguishable -- see
    /// [`permute_rope_rows_interleaves_split_half_pairs_when_two_pairs_exist`]'s
    /// own doc for why `head_dim = 2` cannot tell): a checkpoint that also
    /// carries `model.layers.0.self_attn.q_norm.weight` (Qwen3's own
    /// split-half/NEOX family, per [`safetensors_has_qk_norm`]'s doc) must
    /// bind `attn_q`/`attn_k` with rows exactly as HF wrote them, while the
    /// identical checkpoint minus that tensor (the `rotate_half`
    /// family) must bind them permuted -- the defect this guards is
    /// [`hf_bind_rope_weight`] applying [`permute_rope_rows`]
    /// unconditionally, which scrambles Qwen3's `attn_q`/`attn_k`
    /// (downstream evidence: `rel_err ~= 1.15` permuted vs. `~= 0.043` left
    /// in place, against a known-good GGUF).
    #[test]
    fn qwen3_hf_bind_leaves_q_and_k_rows_in_place() {
        let mut architecture = tiny_dense_architecture();
        architecture.head_dim = 4;
        architecture.query_heads = 1;
        architecture.kv_heads = 1;
        architecture.embedding = 4;
        architecture.feed_forward = 8;

        let llama_family_bytes = dense_checkpoint_with_extra_tensors(&architecture, &[]);
        let qwen3_family_bytes = dense_checkpoint_with_extra_tensors(
            &architecture,
            &[("model.layers.0.self_attn.q_norm.weight".into(), 4)],
        );

        let llama_manifest = proxima_safetensors::parse_complete(&llama_family_bytes)
            .expect("parses real safetensors buffer");
        let qwen3_manifest = proxima_safetensors::parse_complete(&qwen3_family_bytes)
            .expect("parses real safetensors buffer");

        let llama_data_start = header_data_start(&llama_family_bytes);
        let qwen3_data_start = header_data_start(&qwen3_family_bytes);

        let llama_bound = bind_all_weights_from_safetensors(
            &llama_manifest,
            &llama_family_bytes,
            llama_data_start,
            &architecture,
        )
        .expect("binds a rotate_half-family checkpoint");
        let qwen3_bound = bind_all_weights_from_safetensors(
            &qwen3_manifest,
            &qwen3_family_bytes,
            qwen3_data_start,
            &architecture,
        )
        .expect("binds a split-half/NEOX-family checkpoint");

        let llama_attn_q = attn_q_rows(&llama_bound);
        let qwen3_attn_q = attn_q_rows(&qwen3_bound);

        // on-disk `q_proj.weight` is [out_dim=4, in_dim=4] row-major,
        // values `index * 0.5` for `index` in `0..16` (`dense_checkpoint_with_extra_tensors`'s
        // own fixture generator); `attn_q_rows` reads back
        // [`crate::bind::transpose_out_in_to_in_out`]'s `[in_dim, out_dim]`
        // storage, so both expectations below are that same matrix
        // transposed, by hand, after (Qwen3) or before (a rotate_half family) permuting
        // its rows.
        let qwen3_expected = vec![
            0.0, 2.0, 4.0, 6.0, 0.5, 2.5, 4.5, 6.5, 1.0, 3.0, 5.0, 7.0, 1.5, 3.5, 5.5, 7.5,
        ];
        let llama_expected = vec![
            0.0, 4.0, 2.0, 6.0, 0.5, 4.5, 2.5, 6.5, 1.0, 5.0, 3.0, 7.0, 1.5, 5.5, 3.5, 7.5,
        ];

        assert_eq!(
            qwen3_attn_q, qwen3_expected,
            "a split-half/NEOX family (Qwen3) must leave attn_q rows exactly as HF wrote them"
        );
        assert_eq!(
            llama_attn_q, llama_expected,
            "a rotate_half family must still permute attn_q rows before transposing"
        );
        assert_ne!(
            llama_attn_q, qwen3_attn_q,
            "the two families must bind attn_q into genuinely different row orders"
        );
    }

    /// [`hf_bind_matmul_weight`]'s bind stores `attn_q` transposed
    /// (`[in_dim, out_dim]`), so this reads it back row-major over
    /// `out_dim` to compare directly against [`permute_rope_rows`]'s own
    /// `[out_dim, in_dim]` convention -- `in_dim == embedding == 4` here,
    /// matching [`qwen3_hf_bind_leaves_q_and_k_rows_in_place`]'s fixture.
    fn attn_q_rows(bound: &BoundWeights<'_>) -> Vec<f32> {
        bound
            .owned
            .iter()
            .find(|(name, _)| name == &node_names::attn_q(0))
            .map(|(_, values)| values.clone())
            .expect("attn_q.weight bound as an owned buffer")
    }

    /// [`tiny_dense_checkpoint`] plus whatever `(name, element_count)` pairs
    /// `extra` names -- lets a fixture add `q_norm.weight`/`k_norm.weight`
    /// without duplicating every other tensor
    /// [`bind_all_weights_from_safetensors`] looks up.
    fn dense_checkpoint_with_extra_tensors(
        architecture: &ModelHparams,
        extra: &[(String, usize)],
    ) -> Vec<u8> {
        let embedding = architecture.embedding as usize;
        let feed_forward = architecture.feed_forward as usize;
        let kv_dim = architecture.kv_heads as usize * architecture.head_dim as usize;
        let vocab = architecture.vocab as usize;

        let mut owned_data: Vec<(String, Vec<u8>)> = Vec::new();
        let mut push = |name: String, elements: usize| {
            let values: Vec<f32> = (0..elements).map(|index| index as f32 * 0.5).collect();
            owned_data.push((name, f32_bytes(&values)));
        };

        push(names::embed_tokens(), vocab * embedding);
        push(names::input_layernorm(0), embedding);
        push(names::post_attention_layernorm(0), embedding);
        push(names::q_proj(0), embedding * embedding);
        push(names::k_proj(0), kv_dim * embedding);
        push(names::v_proj(0), kv_dim * embedding);
        push(names::o_proj(0), embedding * embedding);
        push(names::gate_proj(0), feed_forward * embedding);
        push(names::up_proj(0), feed_forward * embedding);
        push(names::down_proj(0), embedding * feed_forward);
        push(names::final_norm(), embedding);
        push(names::lm_head(), vocab * embedding);
        for (name, elements) in extra {
            push(name.clone(), *elements);
        }

        let tensors = owned_data
            .iter()
            .map(|(name, bytes)| {
                let elements = bytes.len() / 4;
                TensorPayload {
                    name: name.clone(),
                    dtype: DType::Float32,
                    shape: vec![elements as u64],
                    data: bytes.as_slice(),
                }
            })
            .collect();

        write_complete(&SafetensorsModel {
            tensors,
            metadata: alloc::collections::BTreeMap::new(),
        })
        .expect("writes a real safetensors buffer")
    }

    /// A real bf16 embedding table must bind as an OWNED f32 buffer, never
    /// packed -- the exact defect this row's own fix closes, reproduced
    /// directly rather than only through the full multi-tensor pipeline:
    /// binding `token_embd.weight` packed would land it in the
    /// interpreter's `quantized_weights` map, which `embedding_lookup`'s
    /// gather (`proxima_tensor::cpu::buffer_of`) never reads at all, so a
    /// packed bind here is a silent setup for `NotLowerable { reason:
    /// "operand buffer missing at evaluation time" }` at forward time
    /// rather than a bind-time failure -- this test catches it at bind time
    /// instead.
    #[test]
    fn bf16_embed_tokens_binds_as_owned_f32_not_packed() {
        let architecture = tiny_dense_architecture();
        let vocab = architecture.vocab as usize;
        let embedding = architecture.embedding as usize;
        let values: Vec<half::bf16> = (0..vocab * embedding)
            .map(|index| half::bf16::from_f32(index as f32 * 0.5))
            .collect();
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let tensors = vec![TensorPayload {
            name: names::embed_tokens(),
            dtype: DType::BFloat16,
            shape: vec![vocab as u64, embedding as u64],
            data: bytes.as_slice(),
        }];
        let file_bytes = write_complete(&SafetensorsModel {
            tensors,
            metadata: alloc::collections::BTreeMap::new(),
        })
        .expect("writes a real bf16 safetensors buffer");
        let manifest = proxima_safetensors::parse_complete(&file_bytes)
            .expect("parses real safetensors buffer");
        let data_start = header_data_start(&file_bytes);

        let mut state = BoundWeights {
            resident_bytes: 0,
            owned: Vec::new(),
            packed: Vec::new(),
            packed_owned: Vec::new(),
            expert_ranges: Vec::new(),
            precision: &[],
        };
        hf_bind_dense(
            &manifest,
            &file_bytes,
            data_start,
            names::embed_tokens(),
            node_names::token_embd(),
            &mut state,
        )
        .expect("binds the real bf16 embedding table");

        assert_eq!(
            state.owned.len(),
            1,
            "the bf16 embedding table must decode to an OWNED f32 buffer"
        );
        assert_eq!(
            state.packed.len(),
            0,
            "the bf16 embedding table must NEVER bind packed -- gather cannot read it there"
        );
        assert_eq!(state.owned[0].0, "token_embd.weight");
        assert_eq!(state.owned[0].1.len(), vocab * embedding);
        // real-value check, not just shape: the first decoded element must
        // match an independent bf16 decode of the same bytes.
        assert!((state.owned[0].1[0] - 0.0).abs() < 1e-6);
        assert!((state.owned[0].1[1] - 0.5).abs() < 1e-6);
    }

    /// A tied checkpoint (real SmolLM2-135M-Instruct's own shape: no
    /// `lm_head.weight` tensor at all) must still bind an `output.weight`
    /// program node, reusing `embed_tokens`'s own on-disk bytes -- proves
    /// [`bind_all_weights_from_safetensors`]'s tied-embedding branch, not
    /// just that the untied path works.
    #[test]
    fn tied_embeddings_bind_output_weight_from_embed_tokens_with_no_lm_head_tensor() {
        let mut architecture = tiny_dense_architecture();
        architecture.tied_embeddings = true;
        let embedding = architecture.embedding as usize;
        let feed_forward = architecture.feed_forward as usize;
        let kv_dim = architecture.kv_heads as usize * architecture.head_dim as usize;
        let vocab = architecture.vocab as usize;

        let mut owned_data: Vec<(String, Vec<u8>)> = Vec::new();
        let mut push = |name: String, elements: usize| {
            let values: Vec<f32> = (0..elements).map(|index| index as f32 * 0.5).collect();
            owned_data.push((name, f32_bytes(&values)));
        };
        push(names::embed_tokens(), vocab * embedding);
        push(names::input_layernorm(0), embedding);
        push(names::post_attention_layernorm(0), embedding);
        push(names::q_proj(0), embedding * embedding);
        push(names::k_proj(0), kv_dim * embedding);
        push(names::v_proj(0), kv_dim * embedding);
        push(names::o_proj(0), embedding * embedding);
        push(names::gate_proj(0), feed_forward * embedding);
        push(names::up_proj(0), feed_forward * embedding);
        push(names::down_proj(0), embedding * feed_forward);
        push(names::final_norm(), embedding);
        // deliberately NO names::lm_head() tensor -- this is the whole point

        let tensors = owned_data
            .iter()
            .map(|(name, bytes)| TensorPayload {
                name: name.clone(),
                dtype: DType::Float32,
                shape: vec![(bytes.len() / 4) as u64],
                data: bytes.as_slice(),
            })
            .collect();
        let file_bytes = write_complete(&SafetensorsModel {
            tensors,
            metadata: alloc::collections::BTreeMap::new(),
        })
        .expect("writes a real safetensors buffer with no lm_head.weight tensor");
        let manifest = proxima_safetensors::parse_complete(&file_bytes)
            .expect("parses real safetensors buffer");
        let data_start = header_data_start(&file_bytes);

        let bound = bind_all_weights_from_safetensors(&manifest, &file_bytes, data_start, &architecture)
            .expect("a tied checkpoint must bind output.weight from embed_tokens, not error looking for lm_head.weight");

        let stored_names: alloc::vec::Vec<&str> = bound
            .owned
            .iter()
            .map(|(name, _)| name.as_str())
            .chain(bound.packed.iter().map(|(name, _)| name.as_str()))
            .collect();
        assert!(
            stored_names.contains(&"output.weight"),
            "output.weight must still be bound: {stored_names:?}"
        );
        // 12 STORED entries, same as the untied fixture: 11 on-disk tensors,
        // but `embed_tokens` is read TWICE -- once for `token_embd.weight`,
        // once for `output.weight` -- so the program still gets 12 named
        // inputs even though the checkpoint carries one fewer on-disk tensor.
        assert_eq!(
            bound.owned.len() + bound.packed.len(),
            12,
            "embed_tokens binds twice: once as token_embd.weight, once as output.weight"
        );
        assert_eq!(
            manifest.tensors.len(),
            11,
            "the checkpoint fixture itself carries 11 on-disk tensors, one fewer than the untied fixture's 12, since lm_head.weight is absent"
        );
    }

    /// The one weight this test perturbs by transposing it wrong would prove
    /// [`transpose_out_in_to_in_out`] is actually being applied: corrupt
    /// `q_proj`'s declared shape so it disagrees with `out_dim*in_dim`, and
    /// the typed shape-mismatch error must fire, not a panic mid-transpose.
    /// Forces the OWNED decode-then-transpose fallback (not the zero-copy
    /// packed path, which never validates a shape against `out_dim`/`in_dim`
    /// at all -- see [`hf_bind_matmul_weight`]'s own doc) by prepending a
    /// dummy leading tensor whose byte length shifts `q_proj`'s absolute
    /// file offset off a 4-byte boundary: safetensors writes tensors
    /// back-to-back with zero padding between them (`writer.rs`'s own doc),
    /// so the padding length is fully under this test's control -- tried
    /// 0..4 until the real, measured offset actually lands on an unaligned
    /// byte, rather than assumed.
    #[test]
    fn shape_mismatched_weight_is_a_typed_error_not_a_panic() {
        let architecture = tiny_dense_architecture();
        let embedding = architecture.embedding as usize;
        let bad_q_proj = f32_bytes(&vec![0.0f32; embedding * embedding - 1]);

        for padding in 0..4u64 {
            let dummy_bytes = vec![0u8; padding as usize];
            let tensors = vec![
                TensorPayload {
                    name: "dummy_padding".into(),
                    dtype: DType::Int8,
                    shape: vec![padding],
                    data: dummy_bytes.as_slice(),
                },
                TensorPayload {
                    name: names::q_proj(0),
                    dtype: DType::Float32,
                    shape: vec![(embedding * embedding - 1) as u64],
                    data: bad_q_proj.as_slice(),
                },
            ];
            let file_bytes = write_complete(&SafetensorsModel {
                tensors,
                metadata: alloc::collections::BTreeMap::new(),
            })
            .expect("writes a real safetensors buffer");
            let manifest = proxima_safetensors::parse_complete(&file_bytes)
                .expect("parses real safetensors buffer");
            let entry = manifest
                .tensor(&names::q_proj(0))
                .expect("q_proj entry present");
            let data_start =
                8 + u64::from_le_bytes(file_bytes[..8].try_into().expect("8-byte length prefix"));
            let absolute_offset = data_start + entry.data_offsets.0;
            if absolute_offset.is_multiple_of(4) {
                continue; // still aligned with this padding length; try the next one
            }

            let outcome = hf_bind_matmul_weight(
                &manifest,
                &file_bytes,
                data_start,
                names::q_proj(0),
                node_names::attn_q(0),
                embedding,
                embedding,
                &mut BoundWeights {
                    resident_bytes: 0,
                    owned: Vec::new(),
                    packed: Vec::new(),
                    packed_owned: Vec::new(),
                    expert_ranges: Vec::new(),
                    precision: &[],
                },
            );

            assert!(
                matches!(outcome, Err(InteropError::DenseWeightShapeMismatch { .. })),
                "a short decoded buffer reached through the misaligned owned-decode fallback must \
                 surface as a typed error, got {outcome:?}"
            );
            return;
        }
        panic!(
            "could not construct a misaligned f32 tensor offset with padding in 0..4 -- test setup bug"
        );
    }

    /// The decisive proof for this file's own fix: a raw-packed `F32`
    /// matmul weight bound through [`hf_bind_matmul_weight`] must land in
    /// [`BoundWeights::owned`], transposed, and produce the exact same
    /// matmul result as an independent hand computation over the tensor's
    /// own on-disk bytes -- run through the real
    /// [`proxima_tensor::cpu::evaluate_quantized_named`] evaluation a forward
    /// program actually uses, not merely compared byte-for-byte against the
    /// dequantize-then-transpose path. Mirrors `bind.rs`'s own
    /// `raw_packed_f32_matmul_weight_matches_an_independent_hand_computed_matmul`
    /// against safetensors' wire format instead of GGUF's.
    ///
    /// `out_dim=4`/`in_dim=6` are deliberately asymmetric: a transpose of a
    /// square or single-row buffer can coincidentally read back correctly,
    /// or silently permute symmetric data, proving nothing. With `out_dim !=
    /// in_dim`, a buffer addressed through the wrong axis order reads
    /// flatly wrong values.
    ///
    /// A single-tensor safetensors buffer's data offset is NOT guaranteed
    /// 4-byte aligned (safetensors pads nothing between tensors, unlike
    /// GGUF), so [`safetensors_tensor_as_packed_block`]'s own alignment
    /// check on `Float32` can reject this tensor for reasons that have
    /// nothing to do with this fix, silently forcing the safe owned path
    /// either way and making the test vacuous. A dummy leading `Int8`
    /// tensor whose byte length shifts the target's absolute offset (same
    /// technique [`shape_mismatched_weight_is_a_typed_error_not_a_panic`]
    /// uses in reverse) is prepended, tried over `0..4` bytes of padding
    /// until the offset actually lands 4-byte ALIGNED, so this test
    /// genuinely exercises the packed-vs-owned dispatch rather than an
    /// accidental fallback.
    #[test]
    fn raw_packed_f32_matmul_weight_matches_an_independent_hand_computed_matmul() {
        let out_dim = 4usize;
        let in_dim = 6usize;
        // safetensors' own on-disk convention (identical to GGUF's): `out_dim`
        // rows, each a contiguous run of `in_dim` elements -- weight(out, in)
        // = out*10 + in, distinct per element so a scrambled read is
        // detectable.
        let mut on_disk = vec![0.0f32; out_dim * in_dim];
        for out_index in 0..out_dim {
            for in_index in 0..in_dim {
                on_disk[out_index * in_dim + in_index] = (out_index * 10 + in_index) as f32;
            }
        }
        let bytes = f32_bytes(&on_disk);

        let (file_bytes, manifest, data_start) = (0..4u64)
            .find_map(|padding| {
                let dummy_bytes = vec![0u8; padding as usize];
                let tensors = vec![
                    TensorPayload {
                        name: "dummy_padding".into(),
                        dtype: DType::Int8,
                        shape: vec![padding],
                        data: dummy_bytes.as_slice(),
                    },
                    TensorPayload {
                        name: "model.layers.0.mlp.gate_proj.weight".into(),
                        dtype: DType::Float32,
                        shape: vec![out_dim as u64, in_dim as u64],
                        data: bytes.as_slice(),
                    },
                ];
                let file_bytes = write_complete(&SafetensorsModel {
                    tensors,
                    metadata: alloc::collections::BTreeMap::new(),
                })
                .expect("writes a real safetensors buffer with an asymmetric f32 matmul weight");
                let manifest = proxima_safetensors::parse_complete(&file_bytes).expect("parses real safetensors buffer");
                let data_start = header_data_start(&file_bytes);
                let entry = manifest.tensor("model.layers.0.mlp.gate_proj.weight").expect("gate_proj entry present");
                let absolute_offset = data_start + entry.data_offsets.0;
                absolute_offset.is_multiple_of(4).then_some((file_bytes, manifest, data_start))
            })
            .expect("could not construct a 4-byte-aligned f32 tensor offset with padding in 0..4 -- test setup bug");

        let mut state = BoundWeights {
            resident_bytes: 0,
            owned: Vec::new(),
            packed: Vec::new(),
            packed_owned: Vec::new(),
            expert_ranges: Vec::new(),
            precision: &[],
        };
        hf_bind_matmul_weight(
            &manifest,
            &file_bytes,
            data_start,
            "model.layers.0.mlp.gate_proj.weight".into(),
            "gate".into(),
            out_dim,
            in_dim,
            &mut state,
        )
        .expect("binds the asymmetric f32 matmul weight");
        assert!(
            state.packed.is_empty(),
            "an F32 matmul weight must never take the raw-packed path -- nothing downstream corrects its layout"
        );
        assert_eq!(
            state.owned.len(),
            1,
            "the F32 matmul weight must land in the transposed owned path"
        );
        let bound_weight = &state.owned[0].1;

        // Deliberately NOT symmetric around zero (`index - 2.5` sums to
        // zero over `0..6`, which cancels every `out_dim`-dependent term and
        // makes all four outputs identical regardless of axis order -- a
        // vacuous test that would pass under a transpose too). `index + 1`
        // sums to a nonzero, out-axis-coupled value instead.
        let activation: Vec<f32> = (0..in_dim).map(|index| (index as f32) + 1.0).collect();

        let mut program: Vec<proxima_tensor::op::Op> = Vec::new();
        let activation_node = proxima_tensor::op::append(
            &mut program,
            proxima_tensor::op::Op::Input {
                dtype: proxima_tensor::dtype::DType::Float32,
                shape: alloc::vec![proxima_tensor::op::Extent::Static(in_dim as u32)],
                name: Some("activation".into()),
            },
        );
        let weight_node = proxima_tensor::op::append(
            &mut program,
            proxima_tensor::op::Op::Input {
                dtype: proxima_tensor::dtype::DType::Float32,
                shape: alloc::vec![
                    proxima_tensor::op::Extent::Static(in_dim as u32),
                    proxima_tensor::op::Extent::Static(out_dim as u32)
                ],
                name: Some("gate".into()),
            },
        );
        let product = proxima_tensor::op::append(
            &mut program,
            proxima_tensor::op::Op::Elementwise {
                dtype: proxima_tensor::dtype::DType::Float32,
                body: proxima_tensor::op::ScalarOp::Multiply,
                operands: alloc::vec![
                    (
                        activation_node,
                        proxima_tensor::map::IndexMap::Affine(proxima_tensor::map::projection(
                            2,
                            &[1]
                        ))
                    ),
                    (
                        weight_node,
                        proxima_tensor::map::IndexMap::Affine(proxima_tensor::map::projection(
                            2,
                            &[1, 0]
                        ))
                    ),
                ],
                name: None,
            },
        );
        let logits = proxima_tensor::op::append(
            &mut program,
            proxima_tensor::op::Op::Reduce(proxima_tensor::op::Reduce {
                dtype: proxima_tensor::dtype::DType::Float32,
                body: proxima_tensor::op::ScalarOp::Add,
                init: proxima_tensor::op::ReduceInit::Zero,
                operand: product,
                in_map: proxima_tensor::map::IndexMap::Affine(proxima_tensor::map::projection(
                    2,
                    &[0, 1],
                )),
                out_map: proxima_tensor::map::IndexMap::Affine(proxima_tensor::map::projection(
                    2,
                    &[0],
                )),
                keep: proxima_tensor::op::Keep::Reduce,
                name: Some("logits".into()),
            }),
        );

        let named = [
            ("activation", QuantizedBlock::Float32(activation.as_slice())),
            ("gate", QuantizedBlock::Float32(bound_weight.as_slice())),
        ];
        let evaluated =
            proxima_tensor::cpu::evaluate_quantized_named(&program, &[], &named, &[logits])
                .expect("evaluate the bound matmul weight through the real interpreter");
        let ours = evaluated.root();

        let mut oracle = vec![0.0f32; out_dim];
        for (out_index, logit) in oracle.iter_mut().enumerate() {
            let mut accumulator = 0.0f32;
            for in_index in 0..in_dim {
                accumulator += activation[in_index] * on_disk[out_index * in_dim + in_index];
            }
            *logit = accumulator;
        }

        std::println!("raw_packed_f32_matmul ours={ours:?} oracle={oracle:?}");
        for (out_index, (found, wanted)) in ours.iter().zip(&oracle).enumerate() {
            let diff = (found - wanted).abs();
            assert!(
                diff < 1e-4,
                "output {out_index}: found={found} wanted={wanted} diff={diff}"
            );
        }
    }

    #[test]
    fn moe_architecture_is_a_typed_error_not_an_attempted_wrong_bind() {
        let mut architecture = tiny_dense_architecture();
        architecture.expert_count = 8;
        architecture.expert_used_count = 2;
        let file_bytes: Vec<u8> = Vec::new();
        let manifest = Manifest::default();

        let outcome = bind_all_weights_from_safetensors(&manifest, &file_bytes, 0, &architecture);
        let Err(error) = outcome else {
            panic!("a moe architecture must surface as a typed error, not an attempted bind");
        };

        assert!(
            matches!(
                error,
                InteropError::HfMoeWeightsUnsupported { expert_count: 8 }
            ),
            "expected HfMoeWeightsUnsupported{{expert_count: 8}}, got {error:?}"
        );
    }
}

#[cfg(test)]
mod native_views {
    use super::native_view_matches;

    #[test]
    fn a_one_element_tensor_stands_in_for_a_rank_zero_scalar_leaf() {
        assert!(native_view_matches(&[1], &[]));
    }

    #[test]
    fn equal_axes_match_and_a_wider_tensor_does_not_stand_in_for_a_scalar() {
        assert!(native_view_matches(&[256, 1536], &[256, 1536]));
        assert!(!native_view_matches(&[2], &[]));
        assert!(!native_view_matches(&[1536], &[256, 1536]));
    }
}
