//! Binding a program out of a sharded Hugging Face directory: each shard is
//! bound by [`super::bind_safetensors_program_leaves`] for the leaves it
//! stores, and the per-shard [`BoundWeights`] are concatenated.
//!
//! A fused expert tensor that the single-file binder borrows as one range per
//! expert is copied here into one contiguous stack, because the Metal decode
//! path reads a named contiguous stack (the per-expert ranges substitute only
//! on the CPU path); that copy is the cost this loader pays that
//! [`super::bind_safetensors_program_leaves`] does not.
//!
//! Teaching pointer: a single-file checkpoint is the one-element case, so a
//! caller holds one code path for both layouts. An input the program declares
//! that no shard stores and whose name does not end in `.weight` is a runtime
//! input the decode loop supplies (`ids`, `rope_cos`, `kv_cache.*`); a
//! `.weight` leaf no shard stores is a missing tensor and is reported.

use alloc::collections::BTreeSet;
use alloc::string::String;
use alloc::vec::Vec;

use proxima_safetensors::Manifest;
use proxima_tensor::op::Op;

use super::bind_safetensors_program_leaves;
use crate::bind::BoundWeights;
use crate::error::InteropError;
use crate::profiles::{BindingProfile, TensorAlias};

const WEIGHT_SUFFIX: &str = ".weight";

/// One safetensors file of a directory: its parsed header, the whole file's
/// bytes and the offset where tensor data begins (`8 + header length`).
pub type HfShard<'file> = (Manifest, &'file [u8], u64);

fn stores(manifest: &Manifest, binding: &BindingProfile, name: &str) -> bool {
    manifest.tensor(name).is_some()
        || binding
            .aliases_for(name)
            .any(|(alias, prefix)| match alias {
                TensorAlias::Rename { from, .. } | TensorAlias::Part { from, .. } => {
                    manifest.tensor(&alloc::format!("{prefix}{from}")).is_some()
                }
                TensorAlias::Join { .. } => false,
            })
}

fn named_inputs(program: &[Op]) -> BTreeSet<&str> {
    program
        .iter()
        .filter_map(|operation| match operation {
            Op::Input {
                name: Some(name), ..
            } => Some(name.as_str()),
            _ => None,
        })
        .collect()
}

fn bound_names<'weights>(weights: &'weights BoundWeights<'_>) -> BTreeSet<&'weights str> {
    let owned = weights.owned.iter().map(|(name, _)| name.as_str());
    let packed = weights.packed.iter().map(|(name, _)| name.as_str());
    let packed_owned = weights
        .packed_owned
        .iter()
        .map(|(name, _, _)| name.as_str());
    let ranges = weights
        .expert_ranges
        .iter()
        .map(|(name, _, _)| name.as_str());
    owned
        .chain(packed)
        .chain(packed_owned)
        .chain(ranges)
        .collect()
}

fn absorb<'file>(into: &mut BoundWeights<'file>, from: BoundWeights<'file>) {
    into.resident_bytes += from.resident_bytes;
    into.owned.extend(from.owned);
    into.packed.extend(from.packed);
    into.packed_owned.extend(from.packed_owned);
    into.expert_ranges.extend(from.expert_ranges);
}

fn contiguous_expert_stacks<'file>(weights: &mut BoundWeights<'file>) {
    let ranges = core::mem::take(&mut weights.expert_ranges);
    for (name, parts, codec) in ranges {
        let bytes = parts.concat();
        weights.resident_bytes += bytes.len();
        weights.packed_owned.push((name, bytes, codec));
    }
}

fn bind_one_shard<'file>(
    shard: &HfShard<'file>,
    program: &[Op],
    binding: &BindingProfile,
) -> Result<BoundWeights<'file>, InteropError> {
    let (manifest, bytes, data_start) = shard;
    let skipped: Vec<String> = named_inputs(program)
        .into_iter()
        .filter(|name| !stores(manifest, binding, name))
        .map(String::from)
        .collect();
    bind_safetensors_program_leaves(manifest, bytes, *data_start, program, binding, &skipped)
}

/// Binds every weight leaf of `program` out of `shards`.
///
/// # Errors
///
/// [`InteropError::UnknownTensor`] naming a `.weight` leaf no shard stores;
/// whatever [`super::bind_safetensors_program_leaves`] can fail with.
pub fn bind_safetensors_shards<'file>(
    shards: &[HfShard<'file>],
    program: &[Op],
    binding: &BindingProfile,
) -> Result<BoundWeights<'file>, InteropError> {
    let own_binding = BindingProfile {
        extra: Vec::new(),
        ..binding.clone()
    };
    let mut merged = BoundWeights::new(&[]);
    for shard in shards {
        absorb(&mut merged, bind_one_shard(shard, program, &own_binding)?);
    }
    contiguous_expert_stacks(&mut merged);
    let bound = bound_names(&merged);
    match named_inputs(program)
        .into_iter()
        .find(|name| name.ends_with(WEIGHT_SUFFIX) && !bound.contains(name))
    {
        Some(missing) => Err(InteropError::UnknownTensor {
            name: String::from(missing),
        }),
        None => Ok(merged),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::Path;

    use proxima_primitives::pipe::sans_io::Outcome;
    use proxima_safetensors::{SafetensorsModel, SafetensorsParser, TensorPayload, write_complete};
    use proxima_tensor::DType;
    use proxima_tensor::op::Extent;

    use super::*;
    use crate::profiles::binding_profile;

    const REAL_BF16_TENSOR: &str = "tests/fixtures/mlx-reference/gemma4-e2b/plain/language_model.model.layers.0.self_attn.k_proj.weight.bf16.bin";
    const EMBEDDING_NAME: &str = "language_model.model.embed_tokens.weight";
    const NORM_NAME: &str = "language_model.model.norm.weight";
    const NORM_WIDTH: usize = 1536;

    fn real_bytes() -> Vec<u8> {
        std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join(REAL_BF16_TENSOR))
            .expect("vendored gemma4 k_proj bf16 tensor")
    }

    fn shard_file(name: &str, shape: &[u64], data: &[u8]) -> (Vec<u8>, Manifest, u64) {
        let file = write_complete(&SafetensorsModel {
            tensors: vec![TensorPayload {
                name: name.into(),
                dtype: DType::BFLOAT16,
                shape: shape.to_vec(),
                data,
            }],
            metadata: BTreeMap::new(),
        })
        .expect("writes a real safetensors buffer");
        let data_start = 8 + u64::from_le_bytes(file[..8].try_into().expect("8-byte prefix"));
        let mut parser = SafetensorsParser::new();
        parser.feed(&file[..data_start as usize]);
        let Outcome::Event(manifest) = parser.poll().expect("header parses") else {
            panic!("a complete header must produce a manifest");
        };
        let manifest = manifest.clone();
        (file, manifest, data_start)
    }

    fn leaf(program: &mut Vec<Op>, name: &str, shape: &[u32]) {
        program.push(Op::Input {
            dtype: DType::FLOAT32,
            shape: shape.iter().map(|axis| Extent::Static(*axis)).collect(),
            name: Some(name.into()),
        });
    }

    fn program() -> Vec<Op> {
        let mut program = Vec::new();
        leaf(&mut program, "ids", &[4]);
        leaf(&mut program, "token_embd.weight", &[256, 1536]);
        leaf(&mut program, "output_norm.weight", &[1536]);
        program
    }

    #[test]
    fn each_leaf_binds_from_the_shard_that_stores_it_and_runtime_inputs_need_no_shard() {
        let data = real_bytes();
        let (first_file, first_manifest, first_start) =
            shard_file(EMBEDDING_NAME, &[256, 1536], &data);
        let (second_file, second_manifest, second_start) =
            shard_file(NORM_NAME, &[1536], &data[..NORM_WIDTH * 2]);
        let shards: Vec<HfShard<'_>> = vec![
            (first_manifest, &first_file[..], first_start),
            (second_manifest, &second_file[..], second_start),
        ];

        let bound =
            bind_safetensors_shards(&shards, &program(), &binding_profile("gemma4").unwrap())
                .expect("both leaves are stored somewhere");

        let names: Vec<&str> = bound
            .owned()
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        assert_eq!(names, ["token_embd.weight", "output_norm.weight"]);
        assert_eq!(bound.owned()[0].1.len(), 256 * 1536);
        assert_eq!(bound.owned()[1].1.len(), NORM_WIDTH);
    }

    #[test]
    fn a_weight_leaf_no_shard_stores_is_reported_by_name() {
        let data = real_bytes();
        let (file, manifest, start) = shard_file(EMBEDDING_NAME, &[256, 1536], &data);
        let shards: Vec<HfShard<'_>> = vec![(manifest, &file[..], start)];

        let outcome =
            bind_safetensors_shards(&shards, &program(), &binding_profile("gemma4").unwrap());

        assert!(
            matches!(&outcome, Err(InteropError::UnknownTensor { name }) if name == "output_norm.weight"),
            "output_norm.weight is stored in no shard"
        );
    }

    #[test]
    fn per_expert_ranges_of_a_fused_tensor_become_one_contiguous_stack() {
        let data = real_bytes();
        let ranges: Vec<&[u8]> = vec![&data[..64], &data[128..192], &data[256..320]];
        let mut weights = BoundWeights::new(&[]);
        weights.expert_ranges.push((
            "blk.0.ffn_gate_exps.weight".into(),
            ranges,
            crate::Codec::BFloat16,
        ));

        contiguous_expert_stacks(&mut weights);

        let expected: Vec<u8> = [&data[..64], &data[128..192], &data[256..320]].concat();
        assert!(weights.expert_ranges.is_empty());
        assert_eq!(weights.packed_owned.len(), 1);
        assert_eq!(weights.packed_owned[0].1, expected);
        assert_eq!(weights.resident_bytes, 192);
    }
}
