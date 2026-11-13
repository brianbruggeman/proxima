//! PROXNUM-297 slice B: which weight tables stay on the host for a backend
//! with its own memory.
//!
//! Composes [`proxima_tensor::partition::row_gathers`] (the tables only a row
//! gather reads) and [`proxima_tensor::partition::hoist_row_gather`] (the
//! in-place rewrite), and reads the weights through
//! [`crate::bind::BoundWeights`], the same view
//! `LoadedModel::resident_names` walks. This module only decides; the decode
//! loop runs each [`Hoisted::producer`] on a CPU runtime per step.

use alloc::collections::BTreeMap;
use alloc::collections::btree_map::Entry;
use alloc::vec::Vec;

use proxima_tensor::cpu::QuantizedBlock;
use proxima_tensor::partition::{Hoisted, hoist_row_gather, row_gathers};
use proxima_tensor::{DeviceMemory, Executes, Extent, Op};

use crate::bind::BoundWeights;
use crate::error::InteropError;

/// Where a bound weight's bytes live: `(address, byte length)`. Two leaves
/// with one identity are one tensor under two names (a tied head).
type SourceIdentity = (usize, usize);

fn block_source(block: &QuantizedBlock<'_>) -> SourceIdentity {
    match block {
        QuantizedBlock::Float32(data) => (data.as_ptr() as usize, core::mem::size_of_val(*data)),
        QuantizedBlock::Int32(data) => (data.as_ptr() as usize, core::mem::size_of_val(*data)),
        QuantizedBlock::Packed { bytes, .. } | QuantizedBlock::PackedStrided { bytes, .. } => {
            (bytes.as_ptr() as usize, bytes.len())
        }
    }
}

fn weight_source(weights: &BoundWeights<'_>, name: &str) -> Option<SourceIdentity> {
    if let Some((_, data)) = weights.owned.iter().find(|(owned, _)| owned == name) {
        return Some((data.as_ptr() as usize, core::mem::size_of_val(data.as_slice())));
    }
    if let Some((_, block)) = weights.packed.iter().find(|(packed, _)| packed == name) {
        return Some(block_source(block));
    }
    weights
        .packed_owned
        .iter()
        .find(|(packed, _, _)| packed == name)
        .map(|(_, bytes, _)| (bytes.as_ptr() as usize, bytes.len()))
}

/// Bytes of one f32 row of a `[vocab, row...]` table; `None` when a row
/// extent is symbolic (the rule needs a static row to compare against).
fn row_f32_bytes(shape: &[Extent]) -> Option<usize> {
    shape.iter().skip(1).try_fold(core::mem::size_of::<f32>(), |bytes, extent| match extent {
        Extent::Static(count) => bytes.checked_mul(*count as usize),
        Extent::Symbolic(_) => None,
    })
}

/// A program rewritten by [`host_rows`]: the device program, its recomputed
/// `program_hash`, and the host producers of the hoisted rows.
pub(crate) type HostRewrite = (Vec<Op>, u64, Vec<Hoisted>);

/// The rewrite of the program whose content hash is `key`, computed by
/// `rewrite` on first sight. The key is a content hash, never an address: two
/// programs with equal content share one rewrite, and a reused address cannot
/// return another program's.
///
/// # Errors
/// Whatever `rewrite` returns.
pub(crate) fn rewrite_cached(
    cache: &mut BTreeMap<u64, Option<HostRewrite>>,
    key: u64,
    rewrite: impl FnOnce() -> Result<Option<HostRewrite>, InteropError>,
) -> Result<Option<&HostRewrite>, InteropError> {
    let entry = match cache.entry(key) {
        Entry::Occupied(occupied) => occupied.into_mut(),
        Entry::Vacant(vacant) => vacant.insert(rewrite()?),
    };
    Ok(entry.as_ref())
}

/// What [`host_rows`] decided: the program the device runs and the producers
/// of the rows it no longer holds.
pub(crate) struct HostRows {
    pub(crate) device_program: Vec<Op>,
    pub(crate) hoisted: Vec<Hoisted>,
}

/// The table name a [`Hoisted`] replaced (its `leaf` is `"{table}@rows"`).
pub(crate) fn table_name(hoisted: &Hoisted) -> &str {
    hoisted.leaf.strip_suffix("@rows").unwrap_or(&hoisted.leaf)
}

/// A bound weight's block, by name: what the host producer reads the table from.
pub(crate) fn weight_block<'weights>(
    weights: &'weights BoundWeights<'_>,
    name: &str,
) -> Result<Option<QuantizedBlock<'weights>>, InteropError> {
    if let Some((_, data)) = weights.owned.iter().find(|(owned, _)| owned == name) {
        return Ok(Some(QuantizedBlock::Float32(data.as_slice())));
    }
    if let Some((_, block)) = weights.packed.iter().find(|(packed, _)| packed == name) {
        return Ok(Some(*block));
    }
    let Some((_, bytes, kind)) = weights.packed_owned.iter().find(|(packed, _, _)| packed == name) else {
        return Ok(None);
    };
    crate::bind::as_block(*kind, bytes)
        .map(Some)
        .ok_or(InteropError::UnsupportedCodec { codec: crate::error::codec_name(*kind) })
}

/// The device program and the host producers for `program` on `executes`.
///
/// `None` when nothing moves: the backend shares host memory (so Metal and
/// CPU programs and plan keys stay byte-identical), or no table qualifies.
/// A table qualifies when only a row gather reads it
/// ([`row_gathers`]), no other leaf shares its bytes (a tied head keeps it
/// on the device), and a step reads fewer bytes than the table holds:
/// `max_new_count * row_f32_bytes < table_bytes`.
///
/// Pre-flight against the device limit (design step 4) is slice C.
///
/// # Errors
/// [`InteropError::Tensor`] when a gather the scan found cannot be rewritten.
#[must_use = "the rewritten program is the one the device must run"]
pub(crate) fn host_rows(
    program: &[Op],
    executes: &Executes,
    weights: &BoundWeights<'_>,
    max_new_count: u64,
) -> Result<Option<HostRows>, InteropError> {
    if executes.memory() != DeviceMemory::Separate {
        return Ok(None);
    }
    let gathers = row_gathers(program);
    let shared_sources: Vec<SourceIdentity> = program
        .iter()
        .enumerate()
        .filter(|(position, _)| gathers.iter().all(|gather| gather.table.0 as usize != *position))
        .filter_map(|(_, op)| match op {
            Op::Input { name: Some(name), .. } => weight_source(weights, name),
            _ => None,
        })
        .collect();
    let mut device_program = program.to_vec();
    let mut hoisted = Vec::new();
    for gather in &gathers {
        let Some(Op::Input { name: Some(name), shape, .. }) = program.get(gather.table.0 as usize) else {
            continue;
        };
        let (Some(source), Some(row_bytes)) = (weight_source(weights, name), row_f32_bytes(shape)) else {
            continue;
        };
        let step_bytes = max_new_count.saturating_mul(row_bytes as u64);
        if shared_sources.contains(&source) || step_bytes >= source.1 as u64 {
            continue;
        }
        hoisted.push(hoist_row_gather(&mut device_program, gather)?);
    }
    Ok((!hoisted.is_empty()).then_some(HostRows { device_program, hoisted }))
}

#[cfg(test)]
mod tests {
    use alloc::string::String;
    use alloc::vec;

    use proxima_gguf::quant::q6_k::{BLOCK_BYTES, QK_K, quantize};
    use proxima_numeric::DType;
    use proxima_numeric::codec::Codec;
    use proxima_tensor::cpu::evaluate_quantized_named;
    use proxima_tensor::map::{self, IndexMap};
    use proxima_tensor::spec::{embedding_lookup, input_leaf};
    use proxima_tensor::{NodeId, ScalarOp, append};

    use super::*;

    const VOCAB: u32 = 64;
    const DIM: u32 = 256;
    const SEQ: u32 = 3;
    const PER_LAYER: &str = "per_layer_token_embd.weight";
    const TIED: &str = "token_embd.weight";
    const HEAD: &str = "output.weight";

    fn xorshift_floats(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
            })
            .collect()
    }

    fn q6k_table(seed: u64) -> Vec<u8> {
        let floats = xorshift_floats(seed, (VOCAB * DIM) as usize);
        let mut bytes = vec![0u8; floats.len() / QK_K * BLOCK_BYTES];
        quantize(&floats, &mut bytes).expect("whole q6_k rows quantize");
        bytes
    }

    fn table_shape() -> Vec<Extent> {
        vec![Extent::Static(VOCAB), Extent::Static(DIM)]
    }

    fn negate(program: &mut Vec<Op>, operand: NodeId) -> NodeId {
        append(
            program,
            Op::Elementwise {
                dtype: DType::FLOAT32,
                body: ScalarOp::Negate,
                operands: vec![(operand, IndexMap::Affine(map::projection(2, &[0, 1])))],
                name: None,
            },
        )
    }

    /// ids -> gather(per-layer table) and ids -> gather(tied table), with the
    /// head leaf (the tied table's bytes) read whole: the gemma4 shape.
    fn gemma4_shaped_program() -> Vec<Op> {
        let mut program = Vec::new();
        let ids = input_leaf(&mut program, DType::INT32, vec![Extent::Static(SEQ)], "ids");
        let per_layer = input_leaf(&mut program, DType::UINT8, table_shape(), PER_LAYER);
        let tied = input_leaf(&mut program, DType::UINT8, table_shape(), TIED);
        let head = input_leaf(&mut program, DType::UINT8, table_shape(), HEAD);
        let per_layer_rows = embedding_lookup(&mut program, per_layer, ids);
        let tied_rows = embedding_lookup(&mut program, tied, ids);
        negate(&mut program, per_layer_rows);
        negate(&mut program, tied_rows);
        negate(&mut program, head);
        program
    }

    fn packed(bytes: &[u8]) -> QuantizedBlock<'_> {
        QuantizedBlock::Packed { codec: Codec::Q6K, bytes }
    }

    fn bound<'file>(
        per_layer: &'file [u8],
        tied: &'file [u8],
    ) -> BoundWeights<'file> {
        let mut weights = BoundWeights::new(&[]);
        weights.packed.push((String::from(PER_LAYER), packed(per_layer)));
        weights.packed.push((String::from(TIED), packed(tied)));
        weights.packed.push((String::from(HEAD), packed(tied)));
        weights
    }

    fn separate() -> Executes {
        Executes::BASE.with_memory(DeviceMemory::Separate)
    }

    #[test]
    fn unified_memory_returns_none_and_leaves_the_program_alone() {
        let (per_layer, tied) = (q6k_table(0x313_0001), q6k_table(0x313_0002));
        let weights = bound(&per_layer, &tied);

        let outcome = host_rows(&gemma4_shaped_program(), &Executes::BASE, &weights, 1)
            .expect("a unified backend needs no rewrite");

        assert!(outcome.is_none());
    }

    #[test]
    fn separate_memory_hoists_the_gather_only_table_and_keeps_the_tied_one() {
        let (per_layer, tied) = (q6k_table(0x313_0001), q6k_table(0x313_0002));
        let weights = bound(&per_layer, &tied);

        let HostRows { device_program, hoisted } =
            host_rows(&gemma4_shaped_program(), &separate(), &weights, u64::from(SEQ))
                .expect("the scan succeeds")
                .expect("one table qualifies");

        assert_eq!(hoisted.len(), 1);
        assert_eq!(hoisted[0].leaf, "per_layer_token_embd.weight@rows");
        let leaf_names: Vec<&str> = device_program.iter().filter_map(Op::name).collect();
        assert!(leaf_names.contains(&TIED), "the tied table stays a device leaf");
        assert!(!leaf_names.contains(&PER_LAYER), "the hoisted table is no longer a leaf");
    }

    #[test]
    fn a_step_that_reads_more_than_the_table_holds_stays_on_the_device() {
        let (per_layer, tied) = (q6k_table(0x313_0001), q6k_table(0x313_0002));
        let weights = bound(&per_layer, &tied);

        let outcome = host_rows(&gemma4_shaped_program(), &separate(), &weights, u64::from(VOCAB) * 8)
            .expect("the scan succeeds");

        assert!(outcome.is_none());
    }

    fn content_key(program: &[Op]) -> u64 {
        use core::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        format!("{program:?}").hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn two_distinct_programs_through_the_cache_each_get_their_own_rewrite() {
        let (per_layer, tied) = (q6k_table(0x313_0001), q6k_table(0x313_0002));
        let weights = bound(&per_layer, &tied);
        let executes = separate();
        let full = gemma4_shaped_program();
        let mut single = Vec::new();
        let ids = input_leaf(&mut single, DType::INT32, vec![Extent::Static(SEQ)], "ids");
        let table = input_leaf(&mut single, DType::UINT8, table_shape(), PER_LAYER);
        let rows = embedding_lookup(&mut single, table, ids);
        negate(&mut single, rows);
        let mut cache = BTreeMap::new();
        let mut computed = 0u32;
        let mut rewrite_of = |program: &[Op], cache: &mut BTreeMap<u64, Option<HostRewrite>>| {
            let rewrite = rewrite_cached(cache, content_key(program), || {
                computed += 1;
                Ok(host_rows(program, &executes, &weights, u64::from(SEQ))?
                    .map(|rows| (rows.device_program, 0, rows.hoisted)))
            })
            .expect("the rewrite succeeds")
            .expect("the program has a hoistable table");
            (rewrite.0.len(), rewrite.2.len())
        };

        let (full_len, full_hoisted) = rewrite_of(&full, &mut cache);
        let (single_len, single_hoisted) = rewrite_of(&single, &mut cache);
        let (again_len, _) = rewrite_of(&full, &mut cache);

        assert_eq!((full_len, full_hoisted), (full.len(), 1));
        assert_eq!((single_len, single_hoisted), (single.len(), 1));
        assert_ne!(full_len, single_len, "each program keeps its own device program");
        assert_eq!(again_len, full_len);
        assert_eq!(computed, 2, "an equal-content program is a cache hit, a new one is not");
    }

    #[test]
    fn hoisted_device_program_with_host_rows_equals_the_unhoisted_program() {
        let per_layer = q6k_table(0x313_0001);
        let mut program = Vec::new();
        let ids = input_leaf(&mut program, DType::INT32, vec![Extent::Static(SEQ)], "ids");
        let table = input_leaf(&mut program, DType::UINT8, table_shape(), PER_LAYER);
        let rows = embedding_lookup(&mut program, table, ids);
        let negated = negate(&mut program, rows);
        let mut weights = BoundWeights::new(&[]);
        weights.packed.push((String::from(PER_LAYER), packed(&per_layer)));
        let ids_data = [41.0f32, 0.0, 63.0];
        let ids_block = QuantizedBlock::Float32(ids_data.as_slice());

        let unhoisted = evaluate_quantized_named(
            &program,
            &[],
            &[("ids", ids_block), (PER_LAYER, packed(&per_layer))],
            &[negated],
        )
        .expect("the unhoisted program evaluates");
        let (expected, expected_shape) = unhoisted.get(negated).expect("negated output");
        let HostRows { device_program, hoisted } = host_rows(&program, &separate(), &weights, u64::from(SEQ))
            .expect("the scan succeeds")
            .expect("the table qualifies");
        let produced = evaluate_quantized_named(
            &hoisted[0].producer,
            &[],
            &[("ids", ids_block), (PER_LAYER, packed(&per_layer))],
            &[hoisted[0].output],
        )
        .expect("the producer evaluates on the host");
        let (rows_data, _) = produced.get(hoisted[0].output).expect("rows output");
        let rows_data = rows_data.to_vec();
        let hoisted_run = evaluate_quantized_named(
            &device_program,
            &[],
            &[("ids", ids_block), (hoisted[0].leaf.as_str(), QuantizedBlock::Float32(&rows_data))],
            &[negated],
        )
        .expect("the device program evaluates with the rows fed in");
        let (actual, actual_shape) = hoisted_run.get(negated).expect("negated output");

        assert_eq!(actual_shape, expected_shape);
        let actual_bits: Vec<u32> = actual.iter().map(|value| value.to_bits()).collect();
        let expected_bits: Vec<u32> = expected.iter().map(|value| value.to_bits()).collect();
        assert_eq!(actual_bits, expected_bits);
    }
}
