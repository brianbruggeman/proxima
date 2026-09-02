//! Load-time restacking of a GGUF tensor directory's per-expert entries
//! into one contiguous `[n_experts, ...]` buffer.
//!
//! `proxima-tensor/specs/moe_block.toml` gathers a sparse MoE expert's
//! weight slab out of one stacked `[n_experts, d_in, d_out]` tensor via
//! `IndexMap::Computed`. A real GGUF file never stores that: llama.cpp's
//! MoE convention (verified against
//! `~/.lmstudio/models/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF/*.gguf`)
//! writes `n_experts` independent tensors per layer per projection —
//! `blk.{layer}.{projection}.{expert}.weight` for `expert` in
//! `0..n_experts`, each its own directory entry with its own byte range —
//! plus a separate `blk.{layer}.{projection}_inp.weight` router. This
//! module discovers that family from an already-parsed
//! [`crate::pipe::ParsedGguf::tensors`] slice, validates every member
//! shares one dtype and shape, and describes (then performs) the byte
//! concatenation that turns them into the stacked layout the spec assumes.
//!
//! Sans-IO like the rest of this crate: nothing here opens a file or reads
//! a byte range itself. The caller slices each expert's bytes out of its
//! own buffer (typically via
//! [`crate::pipe::ParsedGguf::tensor_data_range`]) and hands the slices to
//! [`restack_into`] along with a destination buffer it owns.
//!
//! # Why concatenation is valid for a quantized dtype
//!
//! Every ggml block-quantized type packs a fixed element count per block
//! ([`crate::types::GgmlType::block_layout`] — `256` elements / `144`
//! bytes for `Q4_K`, the type Mixtral's expert weights use). The GGUF
//! parser already rejects any tensor whose first dimension isn't a
//! multiple of its type's block size
//! ([`crate::error::GgufError::RowSizeNotBlockMultiple`]), so every
//! individual expert tensor's total element count is *itself* an exact
//! multiple of `block_elements` and therefore occupies a whole number of
//! blocks with no partial block at either end. Appending one expert's
//! bytes immediately after another's therefore never splits a block
//! across the seam — the stack is a pure byte concatenation, no
//! dequantization required. [`plan_stack`] re-derives and checks this
//! (`element_count % block_elements == 0`) independently rather than
//! trusting that invariant silently, and returns
//! [`RestackError::NotBlockMultiple`] if it ever doesn't hold (e.g. a
//! future ggml type with per-element rather than per-row block boundaries,
//! or a hand-built [`crate::tensor::TensorInfo`] that bypassed the
//! parser's own check).

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use thiserror::Error;

use crate::tensor::{MAX_DIMS, TensorInfo};
use crate::types::GgmlType;

type Dims = arrayvec::ArrayVec<u64, MAX_DIMS>;

/// Everything that can go wrong discovering or restacking an expert group.
/// Every failure mode is a typed variant — a mismatched or missing expert
/// never panics.
#[derive(Debug, Error, PartialEq, Eq, Clone)]
pub enum RestackError {
    #[error(
        "layer {layer} projection '{projection}' is missing expert {index} (expected tensor '{name}')"
    )]
    MissingExpert {
        layer: u64,
        projection: String,
        index: u64,
        name: String,
    },

    #[error(
        "layer {layer} projection '{projection}' expert {index} has ggml type {found:?}, expected {expected:?} (expert 0's type)"
    )]
    DtypeMismatch {
        layer: u64,
        projection: String,
        index: u64,
        expected: GgmlType,
        found: GgmlType,
    },

    #[error(
        "layer {layer} projection '{projection}' expert {index} has shape {found:?}, expected {expected:?} (expert 0's shape)"
    )]
    DimsMismatch {
        layer: u64,
        projection: String,
        index: u64,
        expected: Dims,
        found: Dims,
    },

    #[error("expert group is empty -- at least one expert is required to plan a stack")]
    EmptyExpertGroup,

    #[error(
        "expert tensor '{tensor}' has {elements} elements, not a multiple of its {ggml_type:?} block size {block_elements}"
    )]
    NotBlockMultiple {
        tensor: String,
        elements: u64,
        block_elements: u64,
        ggml_type: GgmlType,
    },

    #[error("expert {index} source bytes are {found} bytes, expected {expected} per expert")]
    SourceLengthMismatch {
        index: u64,
        expected: u64,
        found: usize,
    },

    #[error("{found} source slices were given, expected one per expert ({expected})")]
    SourceCountMismatch { expected: usize, found: usize },

    #[error(
        "destination buffer is {found} bytes, expected {expected} for {expert_count} stacked experts"
    )]
    DestinationLengthMismatch {
        expected: u64,
        found: usize,
        expert_count: usize,
    },

    #[error("arithmetic overflow computing {context}")]
    Overflow { context: &'static str },
}

/// The llama.cpp MoE per-expert tensor name: `blk.{layer}.{projection}.{expert}.weight`.
#[must_use]
pub fn expert_tensor_name(layer: u64, projection: &str, expert: u64) -> String {
    format!("blk.{layer}.{projection}.{expert}.weight")
}

/// Finds `blk.{layer}.{projection}.{0..expert_count}.weight` in `tensors`
/// (a linear scan, same cost model as
/// [`crate::pipe::ParsedGguf::metadata_value`] — a layer's expert count is
/// single digits to low tens, not worth an index) and validates every
/// member shares expert 0's dtype and shape. Returns references in expert
/// order.
///
/// # Errors
///
/// [`RestackError::MissingExpert`] if any index in `0..expert_count` has no
/// matching tensor name; [`RestackError::DtypeMismatch`] /
/// [`RestackError::DimsMismatch`] if a later expert's dtype or shape
/// disagrees with expert 0's.
pub fn discover_experts<'a>(
    tensors: &'a [TensorInfo],
    layer: u64,
    projection: &str,
    expert_count: u64,
) -> Result<Vec<&'a TensorInfo>, RestackError> {
    let mut experts: Vec<&'a TensorInfo> = Vec::with_capacity(expert_count as usize);
    for index in 0..expert_count {
        let name = expert_tensor_name(layer, projection, index);
        let tensor = tensors
            .iter()
            .find(|candidate| candidate.name == name)
            .ok_or_else(|| RestackError::MissingExpert {
                layer,
                projection: projection.into(),
                index,
                name: name.clone(),
            })?;

        if let Some(first) = experts.first() {
            if first.ggml_type != tensor.ggml_type {
                return Err(RestackError::DtypeMismatch {
                    layer,
                    projection: projection.into(),
                    index,
                    expected: first.ggml_type,
                    found: tensor.ggml_type,
                });
            }
            if first.dims != tensor.dims {
                return Err(RestackError::DimsMismatch {
                    layer,
                    projection: projection.into(),
                    index,
                    expected: first.dims.clone(),
                    found: tensor.dims.clone(),
                });
            }
        }
        experts.push(tensor);
    }
    Ok(experts)
}

/// The byte-level shape of a stacked expert group: how many experts, how
/// many bytes each contributes, and the total destination size —
/// everything [`restack_into`] and [`gather_expert`] need without
/// re-deriving it from the tensor directory on every call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackPlan {
    pub expert_count: usize,
    pub per_expert_bytes: u64,
    pub total_bytes: u64,
    pub ggml_type: GgmlType,
}

/// Derives a [`StackPlan`] from an already-[`discover_experts`]-validated
/// group. Independently re-checks the block-alignment arithmetic described
/// in this module's doc rather than assuming the parser's own
/// `RowSizeNotBlockMultiple` check already covers it.
///
/// # Errors
///
/// [`RestackError::EmptyExpertGroup`] for an empty slice;
/// [`RestackError::NotBlockMultiple`] if expert 0's element count isn't a
/// whole multiple of its ggml type's block size; [`RestackError::Overflow`]
/// if the per-expert or total byte size can't be computed.
pub fn plan_stack(experts: &[&TensorInfo]) -> Result<StackPlan, RestackError> {
    let first = experts
        .first()
        .copied()
        .ok_or(RestackError::EmptyExpertGroup)?;

    let layout = first.ggml_type.block_layout();
    let elements = first.element_count();
    if layout.block_elements == 0 || elements % layout.block_elements != 0 {
        return Err(RestackError::NotBlockMultiple {
            tensor: first.name.clone(),
            elements,
            block_elements: layout.block_elements,
            ggml_type: first.ggml_type,
        });
    }

    let per_expert_bytes = first.nbytes().ok_or(RestackError::Overflow {
        context: "expert tensor byte size",
    })?;
    let total_bytes =
        per_expert_bytes
            .checked_mul(experts.len() as u64)
            .ok_or(RestackError::Overflow {
                context: "stacked total byte size",
            })?;

    Ok(StackPlan {
        expert_count: experts.len(),
        per_expert_bytes,
        total_bytes,
        ggml_type: first.ggml_type,
    })
}

/// Writes `sources[i]` (expert `i`'s raw tensor bytes, sliced by the caller
/// out of wherever it holds the file — mmap, `std::fs::read`, a network
/// buffer) into `dest[i * per_expert_bytes .. (i + 1) * per_expert_bytes]`,
/// in expert order. Pure byte concatenation: for a block-quantized dtype
/// this never touches quantized values, only moves whole blocks (see the
/// module doc for why that's safe); this function has no dtype-specific
/// logic at all; it copies exactly the number of bytes `plan` says each
/// expert has.
///
/// # Errors
///
/// [`RestackError::SourceCountMismatch`] if `sources.len() != plan.expert_count`;
/// [`RestackError::SourceLengthMismatch`] if any source slice's length
/// disagrees with `plan.per_expert_bytes`; [`RestackError::DestinationLengthMismatch`]
/// if `dest.len() as u64 != plan.total_bytes`.
pub fn restack_into(
    dest: &mut [u8],
    plan: &StackPlan,
    sources: &[&[u8]],
) -> Result<(), RestackError> {
    if sources.len() != plan.expert_count {
        return Err(RestackError::SourceCountMismatch {
            expected: plan.expert_count,
            found: sources.len(),
        });
    }
    if dest.len() as u64 != plan.total_bytes {
        return Err(RestackError::DestinationLengthMismatch {
            expected: plan.total_bytes,
            found: dest.len(),
            expert_count: plan.expert_count,
        });
    }

    for (index, source) in sources.iter().enumerate() {
        if source.len() as u64 != plan.per_expert_bytes {
            return Err(RestackError::SourceLengthMismatch {
                index: index as u64,
                expected: plan.per_expert_bytes,
                found: source.len(),
            });
        }
        let start = (index as u64) * plan.per_expert_bytes;
        let end = start + plan.per_expert_bytes;
        dest[start as usize..end as usize].copy_from_slice(source);
    }
    Ok(())
}

/// Slices expert `expert`'s bytes back out of an already-stacked buffer —
/// the byte-level operation `moe_block.toml`'s `IndexMap::Computed` gather
/// performs at the tensor-algebra level. `None` if `expert >= plan.expert_count`
/// or `stacked` is shorter than the plan expects.
#[must_use]
pub fn gather_expert<'a>(stacked: &'a [u8], plan: &StackPlan, expert: u64) -> Option<&'a [u8]> {
    if expert >= plan.expert_count as u64 {
        return None;
    }
    let start = expert * plan.per_expert_bytes;
    let end = start + plan.per_expert_bytes;
    stacked.get(start as usize..end as usize)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use alloc::string::ToString;
    use alloc::vec;

    use super::*;
    use crate::writer::{GgufModel, TensorPayload, write_complete};

    fn dims(values: &[u64]) -> Dims {
        values.iter().copied().collect()
    }

    /// Deterministic, distinguishable-per-expert byte pattern: expert
    /// `expert`'s bytes are `expert * 251 + position (mod 251)`, so no two
    /// experts' payloads collide and a slicing bug shows up as a value
    /// mismatch rather than an accidental pass.
    fn expert_pattern(expert: u8, len: usize) -> Vec<u8> {
        (0..len)
            .map(|position| {
                expert
                    .wrapping_mul(251)
                    .wrapping_add((position % 251) as u8)
            })
            .collect()
    }

    /// Q4_0 (block_elements = 32, block_bytes = 18): dims `[32, 2]` gives
    /// 64 elements = 2 blocks = 36 bytes per expert, small enough for an
    /// in-memory synthetic fixture while still exercising real
    /// block-quantized arithmetic (not F32, which would trivially pass any
    /// block-alignment check with block_elements == 1).
    fn build_synthetic_experts(
        layer: u64,
        projection: &str,
        expert_count: u8,
    ) -> (Vec<u8>, Vec<Vec<u8>>) {
        let payloads: Vec<Vec<u8>> = (0..expert_count)
            .map(|expert| expert_pattern(expert, 36))
            .collect();
        let tensors: Vec<TensorPayload<'_>> = payloads
            .iter()
            .enumerate()
            .map(|(index, data)| TensorPayload {
                name: expert_tensor_name(layer, projection, index as u64),
                dims: dims(&[32, 2]),
                ggml_type: GgmlType::Q4_0,
                data: data.as_slice(),
            })
            .collect();
        let model = GgufModel {
            version: 3,
            metadata: vec![(
                "general.architecture".to_string(),
                crate::value::MetadataValue::String("mixtral".to_string()),
            )],
            tensors,
        };
        let bytes = write_complete(&model).expect("writes synthetic expert gguf");
        (bytes, payloads)
    }

    #[test]
    fn restacked_buffer_equals_expert_order_concatenation_and_gathers_round_trip() {
        let (gguf_bytes, payloads) = build_synthetic_experts(0, "ffn_gate", 4);
        let parsed = crate::parse_complete(&gguf_bytes).expect("parses synthetic gguf");

        let experts = discover_experts(&parsed.tensors, 0, "ffn_gate", 4)
            .expect("discovers all four experts");
        assert_eq!(experts.len(), 4);

        let plan = plan_stack(&experts).expect("plans stack");
        assert_eq!(plan.per_expert_bytes, 36);
        assert_eq!(plan.total_bytes, 144);
        assert_eq!(plan.ggml_type, GgmlType::Q4_0);

        let sources: Vec<&[u8]> = experts
            .iter()
            .map(|tensor| {
                let range = parsed
                    .tensor_data_range(tensor, gguf_bytes.len() as u64)
                    .expect("tensor range");
                &gguf_bytes[range.start as usize..range.end as usize]
            })
            .collect();

        let mut stacked = alloc::vec![0u8; plan.total_bytes as usize];
        restack_into(&mut stacked, &plan, &sources).expect("restacks into destination buffer");

        let expected_concat: Vec<u8> = payloads.iter().flatten().copied().collect();
        assert_eq!(
            stacked, expected_concat,
            "stacked buffer must equal expert-order concatenation"
        );

        for expert in 0..4u64 {
            let gathered = gather_expert(&stacked, &plan, expert).expect("gathers expert back out");
            assert_eq!(
                gathered,
                payloads[expert as usize].as_slice(),
                "gathering expert {expert} from the stack must yield exactly that expert's original bytes"
            );
        }
    }

    #[test]
    fn discover_experts_reports_missing_index_as_typed_error() {
        let (gguf_bytes, _payloads) = build_synthetic_experts(0, "ffn_gate", 4);
        let parsed = crate::parse_complete(&gguf_bytes).expect("parses synthetic gguf");

        // ask for 5 experts when only 4 exist.
        let outcome = discover_experts(&parsed.tensors, 0, "ffn_gate", 5);
        assert_eq!(
            outcome,
            Err(RestackError::MissingExpert {
                layer: 0,
                projection: "ffn_gate".to_string(),
                index: 4,
                name: "blk.0.ffn_gate.4.weight".to_string(),
            })
        );
    }

    #[test]
    fn discover_experts_reports_dtype_mismatch_as_typed_error() {
        let (gguf_bytes, payloads) = build_synthetic_experts(0, "ffn_gate", 2);
        let mut parsed = crate::parse_complete(&gguf_bytes).expect("parses synthetic gguf");
        // corrupt expert 1's recorded ggml type in the already-parsed
        // directory -- cheaper than hand-building a mixed-type gguf byte
        // stream, and exercises exactly the check under test.
        parsed.tensors[1].ggml_type = GgmlType::F32;

        let outcome = discover_experts(&parsed.tensors, 0, "ffn_gate", 2);
        assert_eq!(
            outcome,
            Err(RestackError::DtypeMismatch {
                layer: 0,
                projection: "ffn_gate".to_string(),
                index: 1,
                expected: GgmlType::Q4_0,
                found: GgmlType::F32,
            })
        );
        drop(payloads);
    }

    #[test]
    fn discover_experts_reports_dims_mismatch_as_typed_error() {
        let (gguf_bytes, _payloads) = build_synthetic_experts(0, "ffn_gate", 2);
        let mut parsed = crate::parse_complete(&gguf_bytes).expect("parses synthetic gguf");
        let original_dims = parsed.tensors[1].dims.clone();
        parsed.tensors[1].dims = dims(&[64, 1]);

        let outcome = discover_experts(&parsed.tensors, 0, "ffn_gate", 2);
        assert_eq!(
            outcome,
            Err(RestackError::DimsMismatch {
                layer: 0,
                projection: "ffn_gate".to_string(),
                index: 1,
                expected: original_dims,
                found: dims(&[64, 1]),
            })
        );
    }

    #[test]
    fn plan_stack_rejects_non_block_multiple_element_count() {
        // hand-built TensorInfo bypassing the parser's own row-length
        // check: 33 elements can never be an integer number of Q4_0's
        // 32-element blocks.
        let tensor = TensorInfo {
            name: "blk.0.ffn_gate.0.weight".to_string(),
            dims: dims(&[33]),
            ggml_type: GgmlType::Q4_0,
            offset: 0,
        };
        let experts = vec![&tensor];
        let outcome = plan_stack(&experts);
        assert_eq!(
            outcome,
            Err(RestackError::NotBlockMultiple {
                tensor: "blk.0.ffn_gate.0.weight".to_string(),
                elements: 33,
                block_elements: 32,
                ggml_type: GgmlType::Q4_0,
            })
        );
    }

    #[test]
    fn plan_stack_rejects_empty_expert_group() {
        let experts: Vec<&TensorInfo> = Vec::new();
        assert_eq!(plan_stack(&experts), Err(RestackError::EmptyExpertGroup));
    }

    #[test]
    fn restack_into_rejects_wrong_source_count() {
        let (gguf_bytes, _payloads) = build_synthetic_experts(0, "ffn_gate", 2);
        let parsed = crate::parse_complete(&gguf_bytes).expect("parses synthetic gguf");
        let experts =
            discover_experts(&parsed.tensors, 0, "ffn_gate", 2).expect("discovers experts");
        let plan = plan_stack(&experts).expect("plans stack");

        let mut dest = alloc::vec![0u8; plan.total_bytes as usize];
        let outcome = restack_into(&mut dest, &plan, &[&[0u8; 36]]);
        assert_eq!(
            outcome,
            Err(RestackError::SourceCountMismatch {
                expected: 2,
                found: 1
            })
        );
    }

    #[test]
    fn restack_into_rejects_wrong_source_length() {
        let (gguf_bytes, _payloads) = build_synthetic_experts(0, "ffn_gate", 2);
        let parsed = crate::parse_complete(&gguf_bytes).expect("parses synthetic gguf");
        let experts =
            discover_experts(&parsed.tensors, 0, "ffn_gate", 2).expect("discovers experts");
        let plan = plan_stack(&experts).expect("plans stack");

        let mut dest = alloc::vec![0u8; plan.total_bytes as usize];
        let short = [0u8; 10];
        let full = [0u8; 36];
        let outcome = restack_into(&mut dest, &plan, &[&full, &short]);
        assert_eq!(
            outcome,
            Err(RestackError::SourceLengthMismatch {
                index: 1,
                expected: 36,
                found: 10
            })
        );
    }

    #[test]
    fn restack_into_rejects_wrong_destination_length() {
        let (gguf_bytes, _payloads) = build_synthetic_experts(0, "ffn_gate", 2);
        let parsed = crate::parse_complete(&gguf_bytes).expect("parses synthetic gguf");
        let experts =
            discover_experts(&parsed.tensors, 0, "ffn_gate", 2).expect("discovers experts");
        let plan = plan_stack(&experts).expect("plans stack");

        let mut dest = alloc::vec![0u8; 10];
        let full = [0u8; 36];
        let outcome = restack_into(&mut dest, &plan, &[&full, &full]);
        assert_eq!(
            outcome,
            Err(RestackError::DestinationLengthMismatch {
                expected: 72,
                found: 10,
                expert_count: 2
            })
        );
    }

    #[test]
    fn gather_expert_returns_none_past_expert_count() {
        let (gguf_bytes, _payloads) = build_synthetic_experts(0, "ffn_gate", 2);
        let parsed = crate::parse_complete(&gguf_bytes).expect("parses synthetic gguf");
        let experts =
            discover_experts(&parsed.tensors, 0, "ffn_gate", 2).expect("discovers experts");
        let plan = plan_stack(&experts).expect("plans stack");
        let stacked = alloc::vec![0u8; plan.total_bytes as usize];

        assert_eq!(gather_expert(&stacked, &plan, 2), None);
    }

    // -- Real Mixtral-8x7B experts, not a synthetic fixture. Opportunistic
    // like `crate::tests::real_file`: this 25 GB model cache is specific to
    // this host, so `#[ignore]`d rather than made a hard failure elsewhere.
    // Only the metadata/tensor-directory prefix and each of the 8 experts'
    // own `Q4_K` byte range for one layer's `ffn_gate` projection are read
    // via direct `seek`+`read` -- never the whole file.
    #[cfg(feature = "std")]
    mod real_mixtral_file {
        use std::io::{Read, Seek, SeekFrom};

        use proxima_telemetry::debug;

        use super::*;
        use crate::pipe::parse_complete;

        const FIXTURE_PATH: &str = "/Users/brianbruggeman/.lmstudio/models/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF/Nous-Hermes-2-Mixtral-8x7B-DPO.Q4_K_S.gguf";

        #[test]
        #[ignore = "depends on a 25 GB host-local mixtral gguf checkout outside this repo"]
        fn restacks_one_real_layers_ffn_gate_experts() {
            let path = std::path::Path::new(FIXTURE_PATH);
            if !path.exists() {
                eprintln!("skipping: no host-local mixtral gguf fixture at {FIXTURE_PATH}");
                return;
            }

            let mut file = std::fs::File::open(path).expect("open host-local mixtral gguf fixture");
            let file_len = file.metadata().expect("stat gguf fixture").len();

            // Grow the metadata-region read until parse_complete stops
            // reporting truncation -- Mixtral's tensor directory (995
            // tensors) fits well under a few MiB, nowhere near the
            // multi-gigabyte payload.
            let mut header_buf = Vec::new();
            let parsed = 'grow: {
                for cap in [4usize << 20, 16 << 20, 64 << 20] {
                    header_buf.resize(cap, 0);
                    file.seek(SeekFrom::Start(0)).expect("seek to file start");
                    let read = file.read(&mut header_buf).expect("read gguf header region");
                    header_buf.truncate(read);
                    if let Ok(parsed) = parse_complete(&header_buf) {
                        break 'grow parsed;
                    }
                }
                panic!("gguf metadata region did not fit in 64 MiB");
            };

            let layer = 0u64;
            let projection = "ffn_gate";
            let expert_count = 8u64;

            let experts = discover_experts(&parsed.tensors, layer, projection, expert_count)
                .expect("discovers all eight real experts for layer 0's ffn_gate projection");
            for expert in &experts {
                debug!(
                    name = %expert.name,
                    dims = ?expert.dims,
                    ggml_type = ?expert.ggml_type,
                    "restack.mixtral discovered expert tensor"
                );
            }

            let plan = plan_stack(&experts).expect("plans stack for real experts");

            let mut sources_owned: Vec<Vec<u8>> = Vec::with_capacity(experts.len());
            for expert in &experts {
                let range = parsed
                    .tensor_data_range(expert, file_len)
                    .expect("expert tensor range within file");
                let mut bytes = alloc::vec![0u8; (range.end - range.start) as usize];
                file.seek(SeekFrom::Start(range.start))
                    .expect("seek to expert tensor data");
                file.read_exact(&mut bytes)
                    .expect("read expert tensor bytes");
                sources_owned.push(bytes);
            }
            let sources: Vec<&[u8]> = sources_owned.iter().map(Vec::as_slice).collect();

            let mut stacked = alloc::vec![0u8; plan.total_bytes as usize];
            restack_into(&mut stacked, &plan, &sources)
                .expect("restacks real experts into destination buffer");

            assert_eq!(
                stacked.len() as u64,
                8 * plan.per_expert_bytes,
                "stacked buffer must equal 8 * single_expert_bytes"
            );
            debug!(
                layer,
                projection,
                expert_count = experts.len() as u64,
                per_expert_bytes = plan.per_expert_bytes,
                total_stacked_bytes = stacked.len() as u64,
                "restack.mixtral restacked one layer's experts"
            );

            for expert in 0..8u64 {
                let gathered = gather_expert(&stacked, &plan, expert)
                    .expect("gathers expert back out of real stack");
                assert_eq!(
                    gathered, sources[expert as usize],
                    "expert {expert} round trip from the real file"
                );
            }
        }
    }
}
