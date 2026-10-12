//! Interleaved A/B replay of kernel-body variants on captured gemma4-E2B
//! decode dispatches.
//!
//! The decode runs once; the dispatches of one step are captured live
//! (`omega::take_captured_dispatches`). For every `<sha16>.<tag>.metal` file in
//! `AB_VARIANT_DIR` whose `sha16` prefixes a captured kernel's `msl_sha256`,
//! the variant is compiled against that dispatch's own buffers and uniform
//! bytes (`CapturedDispatch::with_kernel_variant`) and timed against the
//! production kernel in alternating rounds inside this one process, so GPU
//! clock state and background load hit every arm alike. An optional
//! `<sha16>.<tag>.width` file holds the new threadgroup width; the thread
//! count scales with it. An optional `<sha16>.<tag>.scale` file holds `n/d`, a further
//! multiplier on the thread count, for a variant that changes how many rows one
//! simdgroup folds. An optional `<sha16>.<tag>.launch` file holds `<width x> <width y>`
//! and relaunches the variant through `dispatchThreadgroups` (ggml's launch API).
//!
//! A `<sha16>.<tag>.f16` file lists comma-separated binding indices whose buffers the variant reads as `half` (the replay narrows them from f32). `AB_GEMM_ONLY` times a compacted gemm without its route prepass (the compaction buffer is refilled by the prepass alone, which writes nothing else), `AB_PREPASS_ONLY` the prepass alone.
//! Each `ab group` line carries the median over `AB_ROUNDS` rounds beside the minimum, maximum and coefficient of variation of those rounds (`round_min_us`, `round_max_us`, `round_cov_pct`).
//! Each arm also prints an `ab agree` line (max-abs, cosine and relative L2 error against the base arm over every compared output). With `AB_ATTENTION_REFERENCE` set, a cached-attention group also prints an `ab ref` line per arm against an f64 CPU attention computed from the group's own bound buffers (query planes, in-graph keys and values; the cached range must be empty).
//! Every kernel arm also prints an `ab res` line with its resources (static threadgroup bytes, bound buffer bytes, CPU, RSS, footprint, Metal bytes, load; `AB_RESOURCE_ITERS` replays).
//!
//! Knobs: `AB_VARIANT_DIR`, `AB_STEP` (5), `AB_ROUNDS` (60), `AB_BATCH` (16), `AB_SKIP_GROUPS`
//! (`;`-separated prefixes of `<sha16>:<extents>`; a group whose omission stalls the GPU is named here and skipped),
//! `PROXIMA_PROMPT`.
//!
//! Packed-weight kernels (a `Q4_0` matvec) are skipped unless `AB_PACKED` is set.
//! `AB_ONLY_MEMBER=<k>` swaps only the group's k-th member in a step arm.
//! `AB_WINDOW=<n>` times only the `2n + 1` dispatches around the group's first member
//! (the variant in place of that member), to tell a kernel's own cost from its effect on
//! its neighbours.
//! The bit comparison covers `output_total` values; a fused norm writes its whole
//! iteration space, so `AB_SPAN_FULL` compares over every element of the extents.
//! With `AB_SEQUENCE` set, an arm is timed as one command buffer running every
//! captured member of the group in program order (each member reads its own weight
//! tensor, as the live step does), after `AB_FLUSH_MIB` (384) of CPU writes evict the
//! system cache; the figure is microseconds per dispatch. `AB_SHA` restricts the run to
//! groups whose kernel sha256 starts with the given prefix. With `AB_STEP_SEQUENCE` set
//! (implies `AB_SEQUENCE`), an arm is the whole captured step replayed in program order with
//! the group's members swapped for the variant, so a variant is judged by the step time it
//! moves with the real kernel interleaving and real weight streaming; set `AB_FLUSH_MIB=0`
//! there, the step streams more bytes than the system cache holds. With `AB_OMIT_ALL` set,
//! every kernel group is timed by omission instead: the step with the group's dispatches
//! removed against the whole step, one `ab omit` line per group (its in-situ cost on a
//! GPU kept busy, unlike an isolated single-dispatch command buffer).
//! With `AB_STEP_ALL=<set>;<set>` (a set is comma-separated variant tags) one step arm per set replays the whole captured step with every group that has a variant of one of the set's tags replaced at once, against the unmodified step (one `ab stepall` line per arm).
#![allow(clippy::expect_used, clippy::unwrap_used)]

#[cfg(feature = "metal-attn-variants")]
use std::collections::BTreeSet;

#[cfg(feature = "metal-attn-variants")]
use sha2::{Digest, Sha256};

#[cfg(feature = "metal-attn-variants")]
use omega::{
    AttentionDispatchManifest, AttentionKvReuse, AttentionKvStorage, AttentionMmaPrecision,
    AttentionPrefetch, AttentionQueryParallelism, AttentionSimdTopology, AttentionTileHeight,
    AttentionVariant, PackedOperands, inspect_attention_variant,
};

#[cfg(feature = "metal-attn-variants")]
use proxima_tensor::{BoundOp, BoundOpKind, DType, Layout, NodeId, NumericPolicy};

#[cfg(feature = "metal-attn-variants")]
fn parse_attention_variant(value: &str) -> Result<AttentionVariant, String> {
    let mut selected = AttentionVariant::default();
    let mut seen = BTreeSet::new();
    for field in value.split(',') {
        let (name, selected_value) = field
            .split_once('=')
            .ok_or_else(|| format!("invalid attention variant field `{field}`"))?;
        if !seen.insert(name) {
            return Err(format!("duplicate attention variant field `{name}`"));
        }
        match (name, selected_value) {
            ("kv_storage", "f32") => selected.kv_storage = AttentionKvStorage::F32,
            ("kv_storage", "bf16") => selected.kv_storage = AttentionKvStorage::Bf16,
            ("kv_storage", "bf8") => selected.kv_storage = AttentionKvStorage::Bf8,
            ("mma_precision", "legacy") => selected.mma_precision = AttentionMmaPrecision::Legacy,
            ("mma_precision", "f32") => selected.mma_precision = AttentionMmaPrecision::F32,
            ("mma_precision", "f16") => selected.mma_precision = AttentionMmaPrecision::F16,
            ("kv_reuse", "legacy") => selected.kv_reuse = AttentionKvReuse::Legacy,
            ("kv_reuse", "shared_k") => selected.kv_reuse = AttentionKvReuse::SharedK,
            ("kv_reuse", "shared_kv") => selected.kv_reuse = AttentionKvReuse::SharedKv,
            ("tile_height", "legacy") => selected.tile_height = AttentionTileHeight::Legacy,
            ("tile_height", "rows_2") => selected.tile_height = AttentionTileHeight::Rows2,
            ("tile_height", "rows_4") => selected.tile_height = AttentionTileHeight::Rows4,
            ("tile_height", "rows_8") => selected.tile_height = AttentionTileHeight::Rows8,
            ("tile_height", "rows_16") => selected.tile_height = AttentionTileHeight::Rows16,
            ("query_parallelism", "legacy") => {
                selected.query_parallelism = AttentionQueryParallelism::Legacy;
            }
            ("query_parallelism", "simdgroup_rows") => {
                selected.query_parallelism = AttentionQueryParallelism::SimdgroupRows;
            }
            ("simd_topology", "legacy") => selected.simd_topology = AttentionSimdTopology::Legacy,
            ("simd_topology", "per_head") => {
                selected.simd_topology = AttentionSimdTopology::PerHead
            }
            ("simd_topology", "grouped_queries") => {
                selected.simd_topology = AttentionSimdTopology::GroupedQueries;
            }
            ("prefetch", "off") => selected.prefetch = AttentionPrefetch::Off,
            ("prefetch", "next_block") => selected.prefetch = AttentionPrefetch::NextBlock,
            ("kv_storage", _) => return Err(format!("invalid kv_storage `{selected_value}`")),
            ("mma_precision", _) => {
                return Err(format!("invalid mma_precision `{selected_value}`"));
            }
            ("kv_reuse", _) => return Err(format!("invalid kv_reuse `{selected_value}`")),
            ("tile_height", _) => return Err(format!("invalid tile_height `{selected_value}`")),
            ("query_parallelism", _) => {
                return Err(format!("invalid query_parallelism `{selected_value}`"));
            }
            ("simd_topology", _) => {
                return Err(format!("invalid simd_topology `{selected_value}`"));
            }
            ("prefetch", _) => return Err(format!("invalid prefetch `{selected_value}`")),
            _ => return Err(format!("unknown attention variant field `{name}`")),
        }
    }
    let expected = [
        "kv_storage",
        "mma_precision",
        "kv_reuse",
        "tile_height",
        "query_parallelism",
        "simd_topology",
        "prefetch",
    ];
    if seen.len() != expected.len() || expected.iter().any(|name| !seen.contains(name)) {
        return Err("attention variant requires all seven named fields".to_string());
    }
    Ok(selected)
}

#[cfg(feature = "metal-attn-variants")]
fn granite_dispatch_manifest(
    variant: AttentionVariant,
) -> Result<AttentionDispatchManifest, String> {
    let operation = BoundOp {
        node: NodeId(9),
        dtype: DType::Float32,
        extents: vec![1000, 8, 2, 64],
        kind: BoundOpKind::CachedAttention {
            operands: (0..9)
                .map(|index| {
                    (
                        NodeId(index),
                        Layout {
                            base: 0,
                            strides: vec![1_i64].into(),
                        },
                        None,
                    )
                })
                .collect(),
            query_rows: 1000,
            cached_key_rows: 512,
            new_key_rows: 1000,
            kv_heads: 8,
            query_groups: 2,
            head_dim: 64,
            rotary_dim: 64,
            scale: 1.0,
            cached_lower_inclusive: -511,
            new_upper_inclusive: 0,
        },
    };
    let mut packed_operands = PackedOperands::new();
    let codec = match variant.kv_storage {
        AttentionKvStorage::F32 => None,
        AttentionKvStorage::Bf16 => Some(omega::Codec::BFloat16),
        AttentionKvStorage::Bf8 => Some(omega::Codec::BFloat8),
    };
    if let Some(codec) = codec {
        for operand_index in [2, 3, 6] {
            packed_operands.insert(NodeId(operand_index), codec);
        }
    }
    inspect_attention_variant(
        &operation,
        &packed_operands,
        NumericPolicy::llama_relaxed(),
        variant,
    )
    .map_err(|error| error.to_string())
}

#[cfg(feature = "metal-attn-variants")]
fn describe_attention_variant(variant: AttentionVariant) -> Result<(), String> {
    let manifest = granite_dispatch_manifest(variant)?;
    let digest = Sha256::digest(manifest.kernel.source.as_bytes());
    let source_sha = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let selected = manifest.variant;
    let labels = attention_variant_labels(selected);
    println!(
        "ab variant kv_storage={} mma_precision={} kv_reuse={} tile_height={} query_parallelism={} simd_topology={} prefetch={} form={:?} cache_codec={:?} accumulator={:?} entry={} msl_sha256={source_sha} grid_threads={} threadgroup_width={:?} grid_depth={} dispatch_identity={}",
        labels[0],
        labels[1],
        labels[2],
        labels[3],
        labels[4],
        labels[5],
        labels[6],
        manifest.form,
        manifest.cache_codec,
        manifest.accumulator,
        manifest.kernel.entry,
        manifest.kernel.grid.threads,
        manifest.kernel.grid.threadgroup_width,
        manifest.kernel.grid.depth,
        manifest.dispatch_identity,
    );
    Ok(())
}

#[cfg(feature = "metal-attn-variants")]
fn attention_variant_labels(variant: AttentionVariant) -> [&'static str; 7] {
    let kv_storage = match variant.kv_storage {
        AttentionKvStorage::F32 => "f32",
        AttentionKvStorage::Bf16 => "bf16",
        AttentionKvStorage::Bf8 => "bf8",
    };
    let mma_precision = match variant.mma_precision {
        AttentionMmaPrecision::Legacy => "legacy",
        AttentionMmaPrecision::F32 => "f32",
        AttentionMmaPrecision::F16 => "f16",
    };
    let kv_reuse = match variant.kv_reuse {
        AttentionKvReuse::Legacy => "legacy",
        AttentionKvReuse::SharedK => "shared_k",
        AttentionKvReuse::SharedKv => "shared_kv",
    };
    let tile_height = match variant.tile_height {
        AttentionTileHeight::Legacy => "legacy",
        AttentionTileHeight::Rows2 => "rows_2",
        AttentionTileHeight::Rows4 => "rows_4",
        AttentionTileHeight::Rows8 => "rows_8",
        AttentionTileHeight::Rows16 => "rows_16",
    };
    let query_parallelism = match variant.query_parallelism {
        AttentionQueryParallelism::Legacy => "legacy",
        AttentionQueryParallelism::SimdgroupRows => "simdgroup_rows",
    };
    let simd_topology = match variant.simd_topology {
        AttentionSimdTopology::Legacy => "legacy",
        AttentionSimdTopology::PerHead => "per_head",
        AttentionSimdTopology::GroupedQueries => "grouped_queries",
    };
    let prefetch = match variant.prefetch {
        AttentionPrefetch::Off => "off",
        AttentionPrefetch::NextBlock => "next_block",
    };
    [
        kv_storage,
        mma_precision,
        kv_reuse,
        tile_height,
        query_parallelism,
        simd_topology,
        prefetch,
    ]
}

#[cfg(all(test, feature = "metal-attn-variants"))]
mod attention_variant_tests {
    use super::{
        AttentionVariant, attention_variant_labels, granite_dispatch_manifest,
        parse_attention_variant,
    };

    #[test]
    fn card_22_bench_entry_inspects_legacy_and_granite_variants() {
        let legacy_selector = parse_attention_variant(
            "kv_storage=f32,mma_precision=legacy,kv_reuse=legacy,tile_height=legacy,query_parallelism=legacy,simd_topology=legacy,prefetch=off",
        )
        .expect("all seven legacy fields parse");
        assert_eq!(legacy_selector, AttentionVariant::default());
        let legacy_manifest = granite_dispatch_manifest(legacy_selector)
            .expect("all-legacy Granite dispatch inspects");
        assert_eq!(legacy_manifest.variant, legacy_selector);
        assert_eq!(legacy_manifest.cache_codec, None);
        assert_eq!(legacy_manifest.kernel.grid.threads, 64_000);
        assert_eq!(legacy_manifest.kernel.grid.threadgroup_width, Some(64));
        assert!(
            legacy_manifest
                .kernel
                .entry
                .starts_with("omega_cached_attention_")
        );

        let selected = parse_attention_variant(
            "kv_storage=bf16,mma_precision=f16,kv_reuse=shared_k,tile_height=rows_8,query_parallelism=simdgroup_rows,simd_topology=per_head,prefetch=off",
        )
        .expect("the seven-axis Granite selection parses");
        assert_eq!(
            attention_variant_labels(selected),
            [
                "bf16",
                "f16",
                "shared_k",
                "rows_8",
                "simdgroup_rows",
                "per_head",
                "off",
            ]
        );
        let selected_manifest =
            granite_dispatch_manifest(selected).expect("the multi-axis Granite dispatch inspects");
        assert_eq!(selected_manifest.variant, selected);
        assert_eq!(selected_manifest.cache_codec, Some(omega::Codec::BFloat16));
        assert_eq!(
            selected_manifest.accumulator,
            proxima_tensor::DType::Float32
        );
        assert_ne!(
            selected_manifest.dispatch_identity,
            legacy_manifest.dispatch_identity
        );
        assert!(selected_manifest.kernel.source.contains("shared_key_even"));
        assert!(
            selected_manifest
                .kernel
                .source
                .contains("simd_per_head = true")
        );
        assert!(
            selected_manifest
                .kernel
                .source
                .contains("query_parallel_rows = true")
        );
    }

    #[test]
    fn card_22_bench_entry_refuses_duplicate_and_missing_fields() {
        let duplicate = parse_attention_variant(
            "kv_storage=f32,kv_storage=bf16,mma_precision=legacy,kv_reuse=legacy,tile_height=legacy,query_parallelism=legacy,simd_topology=legacy,prefetch=off",
        )
        .expect_err("duplicate fields are not accepted");
        assert!(duplicate.contains("duplicate attention variant field"));
        let missing = parse_attention_variant(
            "kv_storage=f32,mma_precision=legacy,kv_reuse=legacy,tile_height=legacy,query_parallelism=legacy,simd_topology=legacy",
        )
        .expect_err("every one of the seven named fields is required");
        assert!(missing.contains("requires all seven named fields"));
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[path = "cell_resources/attention_reference.rs"]
mod attention_reference;
#[cfg(all(feature = "metal", target_os = "macos"))]
#[path = "cell_resources/cell.rs"]
mod cell;

#[cfg(all(feature = "metal", target_os = "macos"))]
mod harness {
    use core::ops::ControlFlow;
    use std::collections::BTreeMap;
    use std::fs::File;
    use std::path::PathBuf;

    use memmap2::{Mmap, MmapOptions};
    use omega::CapturedDispatch;
    use proxima_gguf::parse_complete;
    use proxima_gguf::types::GgmlType;
    use proxima_model_interop::{GPU_LAYERS_ALL, LoadedModel, ServingConfig, TokenEvent};
    use proxima_tensor::NumericPolicy;

    use super::attention_reference;
    use super::cell::Cell;
    use super::{describe_attention_variant, parse_attention_variant};

    const MODEL_ENV: &str = "PROXIMA_GEMMA4_E2B_GGUF";
    const SHA_PREFIX_CHARS: usize = 16;

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    }

    fn numeric_policy() -> NumericPolicy {
        match std::env::var("AB_NUMERIC").as_deref() {
            Ok("bit_exact") => NumericPolicy::bit_exact(),
            Ok(other) => panic!("AB_NUMERIC={other}: expected `bit_exact` or unset"),
            Err(_) => ServingConfig::default().numeric_policy,
        }
    }

    fn median(values: &mut [f64]) -> f64 {
        values.sort_by(|left, right| left.partial_cmp(right).expect("finite"));
        values[values.len() / 2]
    }

    fn decode(step: usize) {
        let path = std::env::var(MODEL_ENV).expect("model path env");
        let file = File::open(path).expect("open gemma4-E2B blob");
        // SAFETY: read-only mapping of a checkpoint no other process writes.
        let bytes: Mmap = unsafe { MmapOptions::new().map(&file) }.expect("map blob");
        let parsed = parse_complete(&bytes).expect("parse header");
        let model = LoadedModel::load(&parsed, &bytes).expect("bind gemma4-E2B");
        let serving_config = ServingConfig {
            gpu_layers: GPU_LAYERS_ALL,
            kv_cache_key_quant: GgmlType::F32,
            kv_cache_value_quant: GgmlType::F32,
            flash_attention: false,
            batch_size: 0,
            ubatch_size: 0,
            reasoning_budget: 0,
            dispatch_type: omega::DispatchType::Serial,
            ..ServingConfig::default()
        };
        let prompt = std::fs::read_to_string(
            std::env::var("PROXIMA_PROMPT_FILE").expect("PROXIMA_PROMPT_FILE"),
        )
        .expect("read prompt file");
        let mut on_token = |_event: TokenEvent<'_>| ControlFlow::Continue(());
        model
            .generate_streaming(&prompt, step + 1, serving_config, &mut on_token)
            .expect("greedy decode");
    }

    #[derive(PartialEq, Eq, PartialOrd, Ord)]
    struct GroupKey {
        sha: String,
        threads: u64,
        width: Option<u64>,
        extents: Vec<u64>,
    }

    struct Arm {
        label: String,
        dispatches: Vec<CapturedDispatch>,
    }

    fn load_variants(dir: &PathBuf, members: &[&CapturedDispatch], base_width: u64) -> Vec<Arm> {
        let base = members[0];
        let needle = format!("kernel void {}(", base.entry);
        let mut arms = Vec::new();
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .expect("read variant dir")
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .filter(|name| name.ends_with(".metal"))
            .filter(|name| name.starts_with(&base.msl_sha256[..SHA_PREFIX_CHARS]))
            .collect();
        names.sort();
        for name in names {
            let source = std::fs::read_to_string(dir.join(&name)).expect("read variant");
            if !source.contains(&needle) {
                continue;
            }
            let width_path = dir.join(name.replace(".metal", ".width"));
            let (threads, width) = match std::fs::read_to_string(&width_path) {
                Ok(text) => {
                    let width: u64 = text.trim().parse().expect("width integer");
                    (base.grid.threads / base_width * width, Some(width))
                }
                Err(_) => (base.grid.threads, base.grid.threadgroup_width),
            };
            let scale_path = dir.join(name.replace(".metal", ".scale"));
            let threads = match std::fs::read_to_string(&scale_path) {
                Ok(text) => {
                    let (numerator, denominator) = text.trim().split_once('/').expect("scale n/d");
                    let numerator: u64 = numerator.parse().expect("scale numerator integer");
                    let denominator: u64 = denominator.parse().expect("scale denominator integer");
                    threads * numerator / denominator
                }
                Err(_) => threads,
            };
            let compiled = base
                .with_kernel_variant(&source, &base.entry, threads, width, numeric_policy())
                .unwrap_or_else(|error| panic!("variant {name} does not compile: {error}"));
            let template = match std::fs::read_to_string(dir.join(name.replace(".metal", ".launch")))
            {
                Ok(text) => {
                    let fields: Vec<u64> = text
                        .split_whitespace()
                        .map(|field| field.parse().expect("launch integer"))
                        .collect();
                    let [width_x, width_y] = fields[..] else {
                        panic!("launch file holds `<width x> <width y>`, got {text:?}");
                    };
                    let threadgroups = compiled.grid.threads / (width_x * width_y);
                    compiled.with_threadgroup_launch(threadgroups, width_x, width_y)
                }
                Err(_) => compiled,
            };
            let narrowed: Vec<usize> =
                std::fs::read_to_string(dir.join(name.replace(".metal", ".f16")))
                    .map(|text| {
                        text.trim()
                            .split(',')
                            .map(|index| index.trim().parse().expect("f16 binding index integer"))
                            .collect()
                    })
                    .unwrap_or_default();
            let narrow_codec = std::fs::read_to_string(dir.join(name.replace(".metal", ".narrow")))
                .ok()
                .map(|text| parse_narrow_spec(text.trim()));
            let dispatches = members
                .iter()
                .map(|member| {
                    let replaced = member.with_pipeline_of(&template);
                    if let Some((indices, codec)) = &narrow_codec {
                        return replaced
                            .with_narrowed_buffers(indices, *codec)
                            .unwrap_or_else(|error| {
                                panic!("variant {name} narrowed buffers: {error}")
                            });
                    }
                    if narrowed.is_empty() {
                        replaced
                    } else {
                        replaced
                            .with_f16_buffers(&narrowed)
                            .unwrap_or_else(|error| panic!("variant {name} f16 buffers: {error}"))
                    }
                })
                .collect();
            arms.push(Arm {
                label: name.trim_end_matches(".metal").to_string(),
                dispatches,
            });
        }
        arms
    }

    fn replay_bytes(dispatch: &CapturedDispatch, elements: Option<u64>) -> Vec<u8> {
        if dispatch
            .bindings
            .iter()
            .any(|binding| matches!(binding, omega::Binding::Output(_)))
        {
            return dispatch
                .replay_output_elements(elements)
                .expect("output replay");
        }
        let scratch = dispatch
            .bindings
            .iter()
            .position(|binding| matches!(binding, omega::Binding::Scratch))
            .expect("a captured dispatch with neither output nor scratch binding");
        assert!(
            dispatch.poison_bound_buffer(scratch),
            "scratch buffer is bound and poisonable"
        );
        dispatch.time_gpu_ns(1).expect("scratch replay");
        let mut bytes = dispatch
            .bound_buffer_bytes_at(scratch)
            .expect("scratch bytes");
        if let Some(elements) = elements {
            bytes.truncate(
                usize::try_from(elements).expect("element count") * core::mem::size_of::<f32>(),
            );
        }
        bytes
    }

    fn parse_narrow_spec(text: &str) -> (Vec<usize>, omega::Codec) {
        let (indices, codec) = text
            .split_once(':')
            .expect("narrow spec `<i,j,k>:<f16|bf16|bf8>`");
        let codec = match codec {
            "f16" => omega::Codec::Float16,
            "bf16" => omega::Codec::BFloat16,
            "bf8" => omega::Codec::BFloat8,
            other => panic!("narrow codec `{other}`: expected f16, bf16 or bf8"),
        };
        let indices = indices
            .split(',')
            .map(|index| index.trim().parse().expect("narrow binding index integer"))
            .collect();
        (indices, codec)
    }

    fn ulp_distance(left: f32, right: f32) -> u32 {
        let ordered = |value: f32| {
            let bits = value.to_bits() as i32;
            if bits < 0 {
                i32::MIN.wrapping_sub(bits)
            } else {
                bits
            }
        };
        ordered(left).abs_diff(ordered(right))
    }

    fn compare_outputs(base: &[u8], other: &[u8]) -> String {
        let to_floats = |bytes: &[u8]| -> Vec<f32> {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect()
        };
        let (left, right) = (to_floats(base), to_floats(other));
        let pairs = || left.iter().zip(&right);
        let differing = pairs()
            .filter(|(base_value, other_value)| base_value.to_bits() != other_value.to_bits())
            .count();
        let max_ulp = pairs()
            .map(|(base_value, other_value)| ulp_distance(*base_value, *other_value))
            .max()
            .unwrap_or(0);
        let max_abs = pairs()
            .map(|(base_value, other_value)| (base_value - other_value).abs())
            .fold(0.0_f32, f32::max);
        let largest = left.iter().map(|value| value.abs()).fold(0.0_f32, f32::max);
        format!(
            "elements={} differing={differing} max_ulp={max_ulp} max_abs={max_abs:e} max_abs_over_largest={:e}",
            left.len(),
            max_abs / largest.max(f32::MIN_POSITIVE)
        )
    }

    fn evict_system_cache(scratch: &mut [u8], round: usize) {
        scratch.fill(round as u8);
        std::hint::black_box(&scratch);
    }

    fn measure_sequence(
        arms: &[(String, Vec<&CapturedDispatch>)],
        rounds: usize,
        scratch: &mut [u8],
        per_dispatch: bool,
    ) -> Vec<f64> {
        measure_sequence_spread(arms, rounds, scratch, per_dispatch)
            .into_iter()
            .map(|(median_ns, _)| median_ns)
            .collect()
    }

    fn measure_sequence_spread(
        arms: &[(String, Vec<&CapturedDispatch>)],
        rounds: usize,
        scratch: &mut [u8],
        per_dispatch: bool,
    ) -> Vec<(f64, RoundSpread)> {
        let mut spans: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        for round in 0..rounds {
            for offset in 0..arms.len() {
                let index = (round + offset) % arms.len();
                evict_system_cache(scratch, round + offset);
                let total = CapturedDispatch::time_gpu_sequence_ns(&arms[index].1)
                    .expect("sequence replay");
                let divisor = if per_dispatch {
                    arms[index].1.len() as f64
                } else {
                    1.0
                };
                spans[index].push(total / divisor);
            }
        }
        spans
            .iter_mut()
            .map(|samples| {
                let spread = round_spread(samples);
                (median(samples), spread)
            })
            .collect()
    }

    struct GroupVariants {
        members: Vec<usize>,
        arms: Vec<Arm>,
    }

    fn run_step_all(
        spec: &str,
        dispatches: &[CapturedDispatch],
        groups: &BTreeMap<GroupKey, Vec<usize>>,
        variant_dir: &PathBuf,
        rounds: usize,
        scratch: &mut [u8],
    ) {
        let loaded: Vec<GroupVariants> = groups
            .iter()
            .filter_map(|(key, members)| {
                let member_refs: Vec<&CapturedDispatch> =
                    members.iter().map(|index| &dispatches[*index]).collect();
                let base_width = key.width.unwrap_or(key.threads).max(1);
                let arms = load_variants(variant_dir, &member_refs, base_width);
                (!arms.is_empty()).then(|| GroupVariants {
                    members: members.clone(),
                    arms,
                })
            })
            .collect();
        assert!(!loaded.is_empty(), "N==0: no group carries a variant");
        let sets: Vec<Vec<&str>> = spec
            .split(';')
            .map(|set| set.split(',').collect())
            .collect();
        let mut step_arms: Vec<(String, Vec<&CapturedDispatch>)> =
            vec![("base".to_string(), dispatches.iter().collect())];
        for set in &sets {
            let mut step: Vec<&CapturedDispatch> = dispatches.iter().collect();
            let mut swapped = 0usize;
            for group in &loaded {
                let chosen = group
                    .arms
                    .iter()
                    .find(|arm| set.contains(&arm_tag(&arm.label).as_str()));
                if let Some(arm) = chosen {
                    for (member, replacement) in group.members.iter().zip(&arm.dispatches) {
                        step[*member] = replacement;
                    }
                    swapped += group.members.len();
                }
            }
            println!(
                "ab stepall set={} swapped_dispatches={swapped} of {}",
                set.join(","),
                dispatches.len()
            );
            step_arms.push((set.join("+"), step));
        }
        let results = measure_sequence_spread(&step_arms, rounds, scratch, false);
        for (arm, (median_ns, spread)) in step_arms.iter().zip(results) {
            println!(
                "ab stepall arm={} step_ms={:.4} round_min_ms={:.4} round_max_ms={:.4} round_cov_pct={:.3} rounds={rounds}",
                arm.0,
                median_ns / 1e6,
                spread.minimum / 1e6,
                spread.maximum / 1e6,
                spread.cov_percent
            );
        }
    }

    fn step_with_group_replaced<'a>(
        dispatches: &'a [CapturedDispatch],
        members: &[usize],
        replacements: &'a [CapturedDispatch],
    ) -> Vec<&'a CapturedDispatch> {
        let only_member = std::env::var("AB_ONLY_MEMBER")
            .ok()
            .and_then(|value| value.parse::<usize>().ok());
        let replaced: BTreeMap<usize, &CapturedDispatch> = members
            .iter()
            .copied()
            .zip(replacements.iter())
            .enumerate()
            .filter(|(ordinal, _)| only_member.is_none_or(|only| only == *ordinal))
            .map(|(_, pair)| pair)
            .collect();
        dispatches
            .iter()
            .enumerate()
            .map(|(index, dispatch)| replaced.get(&index).copied().unwrap_or(dispatch))
            .collect()
    }

    struct RoundSpread {
        minimum: f64,
        maximum: f64,
        cov_percent: f64,
    }

    fn round_spread(samples: &[f64]) -> RoundSpread {
        let count = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / count;
        let variance = samples
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (count - 1.0).max(1.0);
        RoundSpread {
            minimum: samples.iter().copied().fold(f64::INFINITY, f64::min),
            maximum: samples.iter().copied().fold(0.0, f64::max),
            cov_percent: 100.0 * variance.sqrt() / mean,
        }
    }

    fn measure(
        arms: &[(String, Vec<&CapturedDispatch>)],
        rounds: usize,
        batch: usize,
    ) -> Vec<(f64, f64, RoundSpread)> {
        let mut single: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        let mut batched: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
        for round in 0..rounds {
            for offset in 0..arms.len() {
                let index = (round + offset) % arms.len();
                single[index].push(arms[index].1[0].time_gpu_ns(1).expect("replay"));
                batched[index].push(arms[index].1[0].time_gpu_ns(batch).expect("replay"));
            }
        }
        (0..arms.len())
            .map(|index| {
                let spread = round_spread(&single[index]);
                let single_ns = median(&mut single[index]);
                let batched_ns = median(&mut batched[index]);
                let marginal = (batched_ns - single_ns) / (batch as f64 - 1.0);
                (marginal, single_ns, spread)
            })
            .collect()
    }

    fn print_arm_resources(sha: &str, arm: &(String, Vec<&CapturedDispatch>), iterations: usize) {
        let lead = arm.1[0];
        let (threadgroup_bytes, max_threads, execution_width) = lead.pipeline_resources();
        let cell = Cell::begin();
        for _ in 0..iterations {
            lead.time_gpu_ns(1).expect("resource replay");
        }
        let line = cell.end(&format!("{sha}:{}", arm.0));
        println!(
            "ab res sha={sha} arm={} tg_static_bytes={threadgroup_bytes} max_threads={max_threads} exec_width={execution_width} bound_buffer_bytes={} iters={iterations} {line}",
            arm.0,
            lead.bound_buffer_bytes()
        );
    }

    fn gemm_only(dispatches: Vec<CapturedDispatch>) -> Vec<CapturedDispatch> {
        dispatches
            .into_iter()
            .map(|dispatch| {
                if dispatch.route_prepass_dispatches == 0 {
                    return dispatch;
                }
                let prepass_alone = dispatch
                    .prepass_only()
                    .expect("a record with a route prepass");
                prepass_alone
                    .time_gpu_ns(1)
                    .expect("fill the compaction buffer");
                if std::env::var_os("AB_PREPASS_ONLY").is_some() {
                    return prepass_alone;
                }
                dispatch.without_prepass()
            })
            .collect()
    }

    fn step_with_pair_replaced<'a>(
        dispatches: &'a [CapturedDispatch],
        partial_members: &[usize],
        partial_replacements: &[&'a CapturedDispatch],
        merge_members: &[usize],
        merge_replacements: &[&'a CapturedDispatch],
    ) -> Vec<&'a CapturedDispatch> {
        let replaced: BTreeMap<usize, &CapturedDispatch> = partial_members
            .iter()
            .copied()
            .zip(partial_replacements.iter().copied())
            .chain(
                merge_members
                    .iter()
                    .copied()
                    .zip(merge_replacements.iter().copied()),
            )
            .collect();
        dispatches
            .iter()
            .enumerate()
            .map(|(index, dispatch)| replaced.get(&index).copied().unwrap_or(dispatch))
            .collect()
    }

    struct PairArm<'a> {
        tag: String,
        partial: Vec<&'a CapturedDispatch>,
        merge: Vec<&'a CapturedDispatch>,
    }

    fn arm_tag(label: &str) -> String {
        label
            .split_once('.')
            .map_or_else(|| label.to_string(), |(_, tag)| tag.to_string())
    }

    fn build_pair_arms<'a>(
        partial_arms: &'a [(String, Vec<&'a CapturedDispatch>)],
        merge_base: &[&'a CapturedDispatch],
        merge_variants: &'a [Arm],
    ) -> Vec<PairArm<'a>> {
        let mut tags: Vec<String> = Vec::new();
        for label in partial_arms
            .iter()
            .skip(1)
            .map(|arm| arm.0.as_str())
            .chain(merge_variants.iter().map(|arm| arm.label.as_str()))
        {
            let tag = arm_tag(label);
            if !tags.contains(&tag) {
                tags.push(tag);
            }
        }
        let mut arms = vec![PairArm {
            tag: "base".to_string(),
            partial: partial_arms[0].1.clone(),
            merge: merge_base.to_vec(),
        }];
        for tag in tags {
            let partial = partial_arms
                .iter()
                .skip(1)
                .find(|arm| arm_tag(&arm.0) == tag)
                .map_or_else(|| partial_arms[0].1.clone(), |arm| arm.1.clone());
            let merge = merge_variants
                .iter()
                .find(|arm| arm_tag(&arm.label) == tag)
                .map_or_else(
                    || merge_base.to_vec(),
                    |arm| arm.dispatches.iter().collect(),
                );
            arms.push(PairArm {
                tag,
                partial,
                merge,
            });
        }
        arms
    }

    fn print_pair_bits(
        sha: &str,
        extents: &[u64],
        pair_arms: &[PairArm<'_>],
        output_elements: u64,
        reference: Option<&Vec<f32>>,
    ) {
        let outputs: Vec<Vec<u8>> = pair_arms
            .iter()
            .map(|arm| {
                arm.partial[0].time_gpu_ns(1).expect("pair partial replay");
                replay_bytes(arm.merge[0], Some(output_elements))
            })
            .collect();
        let base_floats = attention_reference::floats_from_bytes(&outputs[0]);
        for (arm, output) in pair_arms.iter().zip(&outputs) {
            let floats = attention_reference::floats_from_bytes(output);
            println!(
                "ab pbits sha={sha} extents={extents:?} arm={} {}",
                arm.tag,
                compare_outputs(&outputs[0], output)
            );
            println!(
                "ab pagree sha={sha} extents={extents:?} arm={} against=base {}",
                arm.tag,
                attention_reference::agreement(&base_floats, &floats)
            );
            if let Some(reference) = reference {
                println!(
                    "ab pref sha={sha} extents={extents:?} arm={} against=cpu_f64 {}",
                    arm.tag,
                    attention_reference::agreement(reference, &floats)
                );
            }
        }
    }

    struct PairSetup<'a> {
        dispatches: &'a [CapturedDispatch],
        groups: &'a BTreeMap<GroupKey, Vec<usize>>,
        variant_dir: &'a PathBuf,
        pair_prefix: &'a str,
        rounds: usize,
        resource_iterations: usize,
    }

    fn run_pair(
        setup: &PairSetup<'_>,
        partial_members: &[usize],
        partial_arms: &[(String, Vec<&CapturedDispatch>)],
        (sha, extents): (&str, &[u64]),
        scratch: &mut [u8],
    ) {
        let PairSetup {
            dispatches,
            groups,
            variant_dir,
            pair_prefix,
            rounds,
            resource_iterations,
        } = *setup;
        let (merge_key, merge_members) = groups
            .iter()
            .find(|(key, _)| key.sha.starts_with(pair_prefix))
            .expect("N==0: AB_PAIR_SHA matches no captured group");
        assert_eq!(
            merge_members.len(),
            partial_members.len(),
            "pair groups differ in member count"
        );
        let merge_refs: Vec<&CapturedDispatch> = merge_members
            .iter()
            .map(|index| &dispatches[*index])
            .collect();
        let merge_width = merge_key.width.unwrap_or(merge_key.threads).max(1);
        let merge_variants = load_variants(variant_dir, &merge_refs, merge_width);
        let pair_arms = build_pair_arms(partial_arms, &merge_refs, &merge_variants);
        let sha = &sha[..SHA_PREFIX_CHARS];
        for arm in &pair_arms {
            print_arm_resources(
                sha,
                &(format!("partial:{}", arm.tag), arm.partial.clone()),
                resource_iterations,
            );
            print_arm_resources(
                sha,
                &(format!("merge:{}", arm.tag), arm.merge.clone()),
                resource_iterations,
            );
        }
        let reference = std::env::var_os("AB_ATTENTION_REFERENCE")
            .and_then(|_| {
                attention_reference::AttentionShape::from_entry(&partial_arms[0].1[0].entry)
            })
            .map(|shape| {
                attention_reference::reference_output_decode(partial_arms[0].1[0], &shape)
            });
        print_pair_bits(
            sha,
            extents,
            &pair_arms,
            extents.iter().product(),
            reference.as_ref(),
        );
        let layers = partial_members.len();
        let sequence_arms: Vec<(String, Vec<&CapturedDispatch>)> = pair_arms
            .iter()
            .map(|arm| {
                let ordered = arm
                    .partial
                    .iter()
                    .zip(&arm.merge)
                    .flat_map(|(partial, merge)| [*partial, *merge]);
                (arm.tag.clone(), ordered.collect())
            })
            .collect();
        let single_arms: Vec<(String, Vec<&CapturedDispatch>)> = pair_arms
            .iter()
            .map(|arm| (arm.tag.clone(), vec![arm.partial[0], arm.merge[0]]))
            .collect();
        let mut step_arms: Vec<(String, Vec<&CapturedDispatch>)> = pair_arms
            .iter()
            .map(|arm| {
                let replaced = step_with_pair_replaced(
                    dispatches,
                    partial_members,
                    &arm.partial,
                    merge_members,
                    &arm.merge,
                );
                (arm.tag.clone(), replaced)
            })
            .collect();
        let omitted: Vec<&CapturedDispatch> = dispatches
            .iter()
            .enumerate()
            .filter(|(index, _)| !partial_members.contains(index) && !merge_members.contains(index))
            .map(|(_, dispatch)| dispatch)
            .collect();
        step_arms.push(("omit_pair".to_string(), omitted));
        let sequence = measure_sequence_spread(&sequence_arms, rounds, scratch, false);
        let single = measure_sequence_spread(&single_arms, rounds, scratch, false);
        let step = measure_sequence_spread(&step_arms, rounds, scratch, false);
        for (index, arm) in pair_arms.iter().enumerate() {
            println!(
                "ab pair sha={sha} extents={extents:?} layers={layers} arm={} pair_seq_us={:.3} seq_cov_pct={:.3} pair_single_us={:.3} single_cov_pct={:.3} step_ms={:.4} step_cov_pct={:.3} rounds={rounds} p_threads={} p_tg={:?} m_threads={} m_tg={:?}",
                arm.tag,
                sequence[index].0 / layers as f64 / 1e3,
                sequence[index].1.cov_percent,
                single[index].0 / 1e3,
                single[index].1.cov_percent,
                step[index].0 / 1e6,
                step[index].1.cov_percent,
                arm.partial[0].grid.threads,
                arm.partial[0].grid.threadgroup_width,
                arm.merge[0].grid.threads,
                arm.merge[0].grid.threadgroup_width,
            );
        }
        let omit = &step[pair_arms.len()];
        println!(
            "ab pairomit sha={sha} extents={extents:?} layers={layers} arm=omit_pair step_ms={:.4} step_cov_pct={:.3} rounds={rounds}",
            omit.0 / 1e6,
            omit.1.cov_percent
        );
    }

    pub fn run() {
        let attention_variant = std::env::var("AB_ATTENTION_VARIANT").ok().map(|value| {
            parse_attention_variant(&value)
                .unwrap_or_else(|error| panic!("AB_ATTENTION_VARIANT: {error}"))
        });
        if std::env::var("AB_VARIANT_DESCRIBE_ONLY").as_deref() == Ok("1") {
            describe_attention_variant(attention_variant.unwrap_or_default())
                .unwrap_or_else(|error| panic!("describe attention variant: {error}"));
            return;
        }
        assert!(
            attention_variant.is_none(),
            "AB_ATTENTION_VARIANT is currently supported only with AB_VARIANT_DESCRIBE_ONLY=1"
        );
        let process_cell = Cell::begin();
        let resource_iterations = env_usize("AB_RESOURCE_ITERS", 50);
        let step = env_usize("AB_STEP", 5);
        let rounds = env_usize("AB_ROUNDS", 60);
        let batch = env_usize("AB_BATCH", 16);
        let passes = env_usize("AB_PASSES", 1);
        let allow_packed = std::env::var_os("AB_PACKED").is_some();
        let describe = std::env::var_os("AB_DESCRIBE").is_some();
        let span_full = std::env::var_os("AB_SPAN_FULL").is_some();
        let omit_all = std::env::var_os("AB_OMIT_ALL").is_some();
        let skip_groups: Vec<String> = std::env::var("AB_SKIP_GROUPS")
            .map(|list| list.split(';').map(str::to_string).collect())
            .unwrap_or_default();
        let window_radius = std::env::var("AB_WINDOW")
            .ok()
            .and_then(|value| value.parse::<usize>().ok());
        let step_sequence = std::env::var_os("AB_STEP_SEQUENCE").is_some();
        let pair_prefix = std::env::var("AB_PAIR_SHA").ok();
        let sequence = step_sequence
            || pair_prefix.is_some()
            || window_radius.is_some()
            || std::env::var_os("AB_SEQUENCE").is_some();
        let mut scratch = vec![0u8; env_usize("AB_FLUSH_MIB", 384) << 20];
        let variant_dir = PathBuf::from(std::env::var("AB_VARIANT_DIR").expect("AB_VARIANT_DIR"));
        // SAFETY: called from `main` before any thread is spawned.
        unsafe {
            std::env::set_var("PROXIMA_CAPTURE_NODES", "all");
            std::env::set_var("PROXIMA_CAPTURE_STEPS", step.to_string());
            std::env::set_var("PROXIMA_CAPTURE_LIVE", "1");
        }
        decode(step);
        let launched: Vec<CapturedDispatch> = omega::take_captured_dispatches()
            .into_iter()
            .filter(|dispatch| dispatch.grid.threads > 0)
            .collect();
        let launched_total = launched.len();
        let replayable: Vec<CapturedDispatch> = launched
            .into_iter()
            .filter(|dispatch| dispatch.unreplayable.is_none())
            .collect();
        let dispatches = if std::env::var_os("AB_GEMM_ONLY").is_some()
            || std::env::var_os("AB_PREPASS_ONLY").is_some()
        {
            gemm_only(replayable)
        } else {
            replayable
        };
        println!(
            "ab capture excludes {} unreplayable dispatches of {launched_total}",
            launched_total - dispatches.len()
        );
        assert!(!dispatches.is_empty(), "N==0: nothing captured");
        let mut groups: BTreeMap<GroupKey, Vec<usize>> = BTreeMap::new();
        for (index, dispatch) in dispatches.iter().enumerate() {
            let key = GroupKey {
                sha: dispatch.msl_sha256.clone(),
                threads: dispatch.grid.threads,
                width: dispatch.grid.threadgroup_width,
                extents: dispatch.extents.clone(),
            };
            groups.entry(key).or_default().push(index);
        }
        println!(
            "ab capture: step={step} dispatches={} groups={}",
            dispatches.len(),
            groups.len()
        );
        if let Ok(spec) = std::env::var("AB_STEP_ALL") {
            run_step_all(
                &spec,
                &dispatches,
                &groups,
                &variant_dir,
                rounds,
                &mut scratch,
            );
            println!("ab {}", process_cell.end("process"));
            return;
        }
        let mut timed_groups = 0usize;
        for (
            GroupKey {
                sha,
                threads,
                width,
                extents,
            },
            members,
        ) in &groups
        {
            let base = &dispatches[members[0]];
            if !allow_packed && base.operands.iter().any(|(_, codec)| codec != "unpacked") {
                continue;
            }
            let sha_filter = std::env::var("AB_SHA").ok();
            if sha_filter
                .as_ref()
                .is_some_and(|prefix| !sha.starts_with(prefix.as_str()))
            {
                continue;
            }
            let group_name = format!("{}:{extents:?}", &sha[..SHA_PREFIX_CHARS]);
            if skip_groups
                .iter()
                .any(|skipped| group_name.starts_with(skipped.as_str()))
            {
                println!("ab skip group={group_name} count={}", members.len());
                continue;
            }
            if describe {
                println!(
                    "ab describe sha={} extents={extents:?} count={} entry={}",
                    &sha[..SHA_PREFIX_CHARS],
                    members.len(),
                    base.entry
                );
                for line in base.describe_buffers() {
                    println!("ab describe   {line}");
                }
                let uniforms: Vec<i64> = base
                    .uniform_bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|chunk| i64::from_le_bytes(*chunk))
                    .collect();
                println!(
                    "ab describe   uniform_i64={uniforms:?} grid_threads={} threadgroup_width={:?}",
                    base.grid.threads, base.grid.threadgroup_width
                );
                timed_groups += 1;
                continue;
            }
            if omit_all {
                let kept: Vec<&CapturedDispatch> = dispatches
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| !members.contains(index))
                    .map(|(_, dispatch)| dispatch)
                    .collect();
                let arms = vec![
                    ("base".to_string(), dispatches.iter().collect::<Vec<_>>()),
                    ("omit".to_string(), kept),
                ];
                let totals = measure_sequence(&arms, rounds, &mut scratch, false);
                let cost_ms = (totals[0] - totals[1]) / 1e6;
                println!(
                    "ab omit entry={} sha={} threads={threads} tg={width:?} extents={extents:?} count={} base_ms={:.4} cost_ms={cost_ms:.4} per_dispatch_us={:.2}",
                    base.entry,
                    &sha[..SHA_PREFIX_CHARS],
                    members.len(),
                    totals[0] / 1e6,
                    cost_ms * 1e3 / members.len() as f64
                );
                timed_groups += 1;
                continue;
            }
            let base_width = width.unwrap_or(*threads).max(1);
            let member_refs: Vec<&CapturedDispatch> = if sequence {
                members.iter().map(|index| &dispatches[*index]).collect()
            } else {
                vec![base]
            };
            let variants = load_variants(&variant_dir, &member_refs, base_width);
            if variants.is_empty() {
                continue;
            }
            let mut arms: Vec<(String, Vec<&CapturedDispatch>)> =
                vec![("base".to_string(), member_refs.clone())];
            arms.extend(
                variants
                    .iter()
                    .map(|arm| (arm.label.clone(), arm.dispatches.iter().collect())),
            );
            timed_groups += 1;
            let reference_label =
                std::env::var("AB_REFERENCE").unwrap_or_else(|_| "base".to_string());
            let reference = arms
                .iter()
                .find(|arm| arm.0 == reference_label)
                .map_or(base, |arm| arm.1[0]);
            let span = std::env::var("AB_SPAN_ELEMENTS")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .or_else(|| span_full.then(|| extents.iter().product::<u64>()));
            let base_output = replay_bytes(reference, span);
            if let Some(prefix) = &pair_prefix {
                let setup = PairSetup {
                    dispatches: &dispatches,
                    groups: &groups,
                    variant_dir: &variant_dir,
                    pair_prefix: prefix,
                    rounds,
                    resource_iterations,
                };
                run_pair(
                    &setup,
                    members,
                    &arms,
                    (sha.as_str(), extents.as_slice()),
                    &mut scratch,
                );
                continue;
            }
            for arm in &arms {
                print_arm_resources(&sha[..SHA_PREFIX_CHARS], arm, resource_iterations);
            }
            let attention_reference = std::env::var_os("AB_ATTENTION_REFERENCE")
                .and_then(|_| attention_reference::AttentionShape::from_entry(&base.entry))
                .map(|shape| attention_reference::reference_output(base, &shape));
            for arm in &arms {
                let output = replay_bytes(arm.1[0], span);
                println!(
                    "ab bits sha={} threads={threads} extents={extents:?} arm={} {}",
                    &sha[..SHA_PREFIX_CHARS],
                    arm.0,
                    compare_outputs(&base_output, &output)
                );
                println!(
                    "ab agree sha={} extents={extents:?} arm={} against=base {}",
                    &sha[..SHA_PREFIX_CHARS],
                    arm.0,
                    attention_reference::agreement(
                        &attention_reference::floats_from_bytes(&base_output),
                        &attention_reference::floats_from_bytes(&output)
                    )
                );
                if let Some(reference) = &attention_reference {
                    println!(
                        "ab ref sha={} extents={extents:?} arm={} against=cpu_f64 {}",
                        &sha[..SHA_PREFIX_CHARS],
                        arm.0,
                        attention_reference::agreement(
                            reference,
                            &attention_reference::floats_from_bytes(&output)
                        )
                    );
                }
            }
            if std::env::var_os("AB_PREPASS_ONLY").is_some() {
                let slot = base.bindings.len();
                base.poison_bound_buffer(slot);
                base.time_gpu_ns(1).expect("prepass replay");
                let reference_bytes = base.bound_buffer_bytes_at(slot).expect("compaction buffer");
                for arm in &arms {
                    arm.1[0].poison_bound_buffer(slot);
                    arm.1[0].time_gpu_ns(1).expect("prepass arm replay");
                    let bytes = arm.1[0]
                        .bound_buffer_bytes_at(slot)
                        .expect("compaction buffer");
                    let differing = reference_bytes
                        .iter()
                        .zip(&bytes)
                        .filter(|(left, right)| left != right)
                        .count();
                    println!(
                        "ab compaction sha={} extents={extents:?} arm={} bytes={} differing={differing}",
                        &sha[..SHA_PREFIX_CHARS],
                        arm.0,
                        bytes.len()
                    );
                }
            }
            for pass in 0..passes {
                if let Some(radius) = window_radius {
                    let ordinals: Vec<usize> = match std::env::var("AB_MEMBER").as_deref() {
                        Ok("all") => (0..members.len()).collect(),
                        Ok(value) => {
                            vec![value.parse::<usize>().unwrap_or(0).min(members.len() - 1)]
                        }
                        Err(_) => vec![0],
                    };
                    for ordinal in ordinals {
                        let first = members[ordinal];
                        let low = first.saturating_sub(radius);
                        let high = (first + radius + 1).min(dispatches.len());
                        let window_arms: Vec<(String, Vec<&CapturedDispatch>)> = arms
                            .iter()
                            .map(|(label, group)| {
                                let window = (low..high)
                                    .map(|index| {
                                        if index == first {
                                            group[ordinal]
                                        } else {
                                            &dispatches[index]
                                        }
                                    })
                                    .collect();
                                (label.clone(), window)
                            })
                            .collect();
                        let results = measure_sequence(&window_arms, rounds, &mut scratch, false);
                        for (arm, total) in window_arms.iter().zip(results) {
                            println!(
                                "ab window sha={} extents={extents:?} member={ordinal} first={first} radius={radius} pass={pass} arm={} window_us={:.2}",
                                &sha[..SHA_PREFIX_CHARS],
                                arm.0,
                                total / 1e3
                            );
                        }
                    }
                    continue;
                }
                if step_sequence {
                    let omitted: Vec<&CapturedDispatch> = dispatches
                        .iter()
                        .enumerate()
                        .filter(|(index, _)| !members.contains(index))
                        .map(|(_, dispatch)| dispatch)
                        .collect();
                    let step_arms: Vec<(String, Vec<&CapturedDispatch>)> =
                        std::iter::once(("base".to_string(), dispatches.iter().collect()))
                            .chain(std::iter::once(("omit_group".to_string(), omitted)))
                            .chain(variants.iter().map(|arm| {
                                (
                                    arm.label.clone(),
                                    step_with_group_replaced(&dispatches, members, &arm.dispatches),
                                )
                            }))
                            .collect();
                    let results = measure_sequence(&step_arms, rounds, &mut scratch, false);
                    for (arm, total) in step_arms.iter().zip(results) {
                        println!(
                            "ab step sha={} threads={threads} extents={extents:?} count={} pass={pass} arm={} step_ms={:.4}",
                            &sha[..SHA_PREFIX_CHARS],
                            members.len(),
                            arm.0,
                            total / 1e6
                        );
                    }
                    continue;
                }
                if sequence {
                    let results = measure_sequence(&arms, rounds, &mut scratch, true);
                    for (arm, per_dispatch) in arms.iter().zip(results) {
                        println!(
                            "ab sequence sha={} threads={threads} extents={extents:?} tg={width:?} count={} pass={pass} arm={} per_dispatch_us={:.3}",
                            &sha[..SHA_PREFIX_CHARS],
                            members.len(),
                            arm.0,
                            per_dispatch / 1e3
                        );
                    }
                    continue;
                }
                let results = measure(&arms, rounds, batch);
                for (arm, (marginal, single, spread)) in arms.iter().zip(results) {
                    println!(
                        "ab group sha={} threads={threads} extents={extents:?} tg={width:?} count={} pass={pass} arm={} marginal_us={:.3} single_us={:.3} rounds={rounds} round_min_us={:.3} round_max_us={:.3} round_cov_pct={:.3}",
                        &sha[..SHA_PREFIX_CHARS],
                        members.len(),
                        arm.0,
                        marginal / 1e3,
                        single / 1e3,
                        spread.minimum / 1e3,
                        spread.maximum / 1e3,
                        spread.cov_percent
                    );
                }
            }
        }
        println!("ab {}", process_cell.end("process"));
        assert!(
            timed_groups > 0,
            "N==0: no captured kernel matched a variant file"
        );
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    harness::run();
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("unsupported target for this example");
    std::process::exit(1);
}
