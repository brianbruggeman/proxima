//! Header agreement of the plain and affine MLX exports with the vendored
//! mlx-community references under `tests/fixtures/mlx-reference`.
//!
//! Each model is exported through the real chain: `export_plain` on the GGUF,
//! `dense_names` for the HF names, `export_affine` for the quantized leaves,
//! `MlxExport::assemble` for the shards and `proxima_safetensors::parse_complete`
//! to read the produced headers back. The headers are what is compared, so a
//! tensor whose leading (row or expert) axis would need gigabytes to
//! materialise is exported on its first rows only; the reference's leading
//! axis is compared against the GGUF's full row count separately.

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs::File;
    use std::path::{Path, PathBuf};

    use proxima_gguf::parse_complete;
    use proxima_gguf::pipe::ParsedGguf;
    use proxima_safetensors::{TensorPayload, map_dtype, parse_complete as parse_safetensors};
    use proxima_tensor::DType;
    use serde_json::Value;

    use crate::bind::{ModelHparams, architecture_from_metadata};
    use crate::profiles::binding_profile;
    use crate::sliding_pattern::hparams::from_metadata as sliding_from_metadata;
    use crate::transform::mlx::{MLX_MAX_SHARD_BYTES, MlxExport};
    use crate::transform::mlx_config::GRANITE_QUANTIZED;
    use crate::transform::mlx_names::FusedName;
    use crate::transform::{
        AffineExport, AffineParams, DEFAULT_PLAIN_DTYPE, Layout, MlxNames, PlainTensor,
        QuantizedConventions, SANDWICH_NORM_LAYOUT, STANDARD_LAYOUT, dense_names, export_affine,
        export_plain, hf_config_bytes, sliding_pattern_hparams,
    };

    const E2B_GGUF: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";
    const GRANITE_GGUF: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80";
    const REDUCE_ABOVE_ELEMENTS: u64 = 1 << 20;
    const REDUCED_ROWS: u64 = 4;

    struct Case {
        model: &'static str,
        gguf: &'static str,
        family: &'static str,
        layout: Layout,
        prefix: &'static str,
        quantized: QuantizedConventions,
    }

    const GRANITE: Case = Case {
        model: "granite-3.1-1b-a400m-moe",
        gguf: GRANITE_GGUF,
        family: "granitemoe",
        layout: STANDARD_LAYOUT,
        prefix: "model.",
        quantized: GRANITE_QUANTIZED,
    };

    const E2B: Case = Case {
        model: "gemma4-e2b",
        gguf: E2B_GGUF,
        family: "gemma4",
        layout: SANDWICH_NORM_LAYOUT,
        prefix: "language_model.model.",
        quantized: QuantizedConventions::NONE,
    };

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Entry {
        dtype: DType,
        shape: Vec<u64>,
    }

    type Header = BTreeMap<String, Entry>;
    type Rows = BTreeMap<String, (u64, u64)>;

    #[derive(Debug, Default, PartialEq, Eq)]
    struct Disagreement {
        only_export: Vec<String>,
        only_reference: Vec<String>,
        shape: Vec<String>,
        dtype: Vec<String>,
    }

    fn fixture_dir(model: &str, flavor: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/mlx-reference")
            .join(model)
            .join(flavor)
    }

    fn header_entries(path: &Path) -> Header {
        let header: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        header
            .as_object()
            .unwrap()
            .iter()
            .filter(|(key, _)| *key != "__metadata__")
            .map(|(key, entry)| {
                let dtype = map_dtype(key, entry["dtype"].as_str().unwrap()).unwrap();
                let shape = entry["shape"].as_array().unwrap();
                let shape = shape.iter().map(|dim| dim.as_u64().unwrap()).collect();
                (key.clone(), Entry { dtype, shape })
            })
            .collect()
    }

    fn reference_header(model: &str, flavor: &str) -> Header {
        std::fs::read_dir(fixture_dir(model, flavor))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                let name = path.file_name().and_then(|name| name.to_str());
                name.is_some_and(|name| name.starts_with("header") && name.ends_with(".json"))
            })
            .flat_map(|path| header_entries(&path))
            .collect()
    }

    fn reference_metadata(model: &str, flavor: &str) -> BTreeMap<String, String> {
        let path = fixture_dir(model, flavor).join("header.json");
        let header: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let object = header["__metadata__"].as_object().unwrap();
        object
            .iter()
            .map(|(key, value)| (key.clone(), value.as_str().unwrap().to_string()))
            .collect()
    }

    fn reference_config(model: &str, flavor: &str) -> Value {
        let bytes = std::fs::read(fixture_dir(model, flavor).join("config.json")).unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    struct Source {
        mapping: memmap2::Mmap,
        reduced: ParsedGguf,
        rows: Rows,
    }

    fn open(path: &str) -> (memmap2::Mmap, ParsedGguf) {
        assert!(
            Path::new(path).exists(),
            "host-local gguf blob missing at {path}"
        );
        let mapping = unsafe { memmap2::Mmap::map(&File::open(path).unwrap()) }.unwrap();
        let parsed = parse_complete(&mapping).unwrap();
        (mapping, parsed)
    }

    fn reduce(parsed: &ParsedGguf) -> (ParsedGguf, Rows) {
        let mut reduced = parsed.clone();
        let rows = reduced
            .tensors
            .iter_mut()
            .filter(|tensor| tensor.element_count() > REDUCE_ABOVE_ELEMENTS)
            .filter_map(|tensor| {
                let last = tensor.dims.last_mut()?;
                let original = *last;
                *last = original.min(REDUCED_ROWS);
                Some((tensor.name.clone(), (original, *last)))
            })
            .collect();
        (reduced, rows)
    }

    fn source(case: &Case) -> Source {
        let (mapping, parsed) = open(case.gguf);
        let (reduced, rows) = reduce(&parsed);
        Source {
            mapping,
            reduced,
            rows,
        }
    }

    fn names_of(case: &Case, parsed: &ParsedGguf, conventions: QuantizedConventions) -> MlxNames {
        let binding = binding_profile(case.family).unwrap();
        let directory: Vec<&str> = parsed.tensors.iter().map(|t| t.name.as_str()).collect();
        let leaves: Vec<&str> = directory
            .iter()
            .copied()
            .filter(|name| !binding.extra.iter().any(|extra| extra == name))
            .collect();
        conventions.apply(dense_names(
            &leaves,
            &directory,
            &binding,
            case.layout,
            case.prefix,
        ))
    }

    fn interleave(name: &str, parts: &[PlainTensor]) -> PlainTensor {
        let experts = usize::try_from(parts[0].shape[0]).unwrap();
        let chunk = parts[0].bytes.len() / experts;
        let slice_of = |part: &PlainTensor, expert: usize| {
            part.bytes[expert * chunk..(expert + 1) * chunk].to_vec()
        };
        let bytes = (0..experts)
            .flat_map(|expert| parts.iter().flat_map(move |part| slice_of(part, expert)))
            .collect();
        let mut shape = parts[0].shape.clone();
        shape[1] *= parts.len() as u64;
        PlainTensor {
            name: name.to_string(),
            shape,
            dtype: parts[0].dtype,
            bytes,
        }
    }

    fn fuse_groups(
        fused: &[FusedName],
        by_leaf: &mut BTreeMap<String, PlainTensor>,
    ) -> Vec<PlainTensor> {
        let hf_names: BTreeSet<&str> = fused.iter().map(|item| item.hf.as_str()).collect();
        hf_names
            .into_iter()
            .map(|hf| {
                let mut group: Vec<&FusedName> =
                    fused.iter().filter(|item| item.hf == hf).collect();
                group.sort_by_key(|item| item.part);
                assert_eq!(group.len(), group[0].of as usize, "all parts of {hf}");
                let parts: Vec<PlainTensor> = group
                    .iter()
                    .map(|item| by_leaf.remove(&item.leaf).unwrap())
                    .collect();
                interleave(hf, &parts)
            })
            .collect()
    }

    fn hf_tensors(
        case: &Case,
        source: &Source,
        dtype: DType,
        conventions: QuantizedConventions,
    ) -> Vec<PlainTensor> {
        let names = names_of(case, &source.reduced, conventions);
        assert!(names.unmapped.is_empty(), "unmapped: {:?}", names.unmapped);
        let exported = export_plain(&source.reduced, &source.mapping, dtype).unwrap();
        let mut by_leaf: BTreeMap<String, PlainTensor> = exported
            .into_iter()
            .map(|tensor| (tensor.name.clone(), tensor))
            .collect();
        let mut named: Vec<PlainTensor> = names
            .mapped
            .iter()
            .filter_map(|(leaf, hf)| {
                by_leaf.remove(leaf).map(|mut tensor| {
                    tensor.name.clone_from(hf);
                    tensor
                })
            })
            .collect();
        named.extend(fuse_groups(&names.fused, &mut by_leaf));
        named
    }

    fn kept_rows(case: &Case, source: &Source, conventions: QuantizedConventions) -> Rows {
        let names = names_of(case, &source.reduced, conventions);
        let fused = names.fused.iter().map(|item| (&item.leaf, &item.hf));
        let mapped = names.mapped.iter().map(|(leaf, hf)| (leaf, hf));
        mapped
            .chain(fused)
            .filter_map(|(leaf, hf)| source.rows.get(leaf).map(|rows| (hf.clone(), *rows)))
            .collect()
    }

    fn quantizable(tensor: &PlainTensor) -> bool {
        tensor.name.ends_with(".weight") && tensor.shape.len() >= 2
    }

    fn f32_values(tensor: &PlainTensor) -> Vec<f32> {
        tensor
            .bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect()
    }

    fn pack_tensor(tensor: &PlainTensor, params: AffineParams) -> Option<Vec<PlainTensor>> {
        let columns = usize::try_from(*tensor.shape.last()?).unwrap();
        let values = f32_values(tensor);
        let rows = values.len() / columns;
        let exported = export_affine(&values, [rows, columns], params, DType::BFloat16).unwrap();
        let AffineExport::Packed(packed) = exported else {
            return None;
        };
        let base = tensor.name.strip_suffix(".weight")?;
        let lead = &tensor.shape[..tensor.shape.len() - 1];
        let shaped = |last: usize| [lead, &[last as u64]].concat();
        let words = packed.weight.iter().flat_map(|word| word.to_le_bytes());
        Some(vec![
            PlainTensor {
                name: format!("{base}.weight"),
                shape: shaped(packed.weight_shape[1]),
                dtype: DType::UInt32,
                bytes: words.collect(),
            },
            PlainTensor {
                name: format!("{base}.scales"),
                shape: shaped(packed.scales_shape[1]),
                dtype: packed.dtype,
                bytes: packed.scales,
            },
            PlainTensor {
                name: format!("{base}.biases"),
                shape: shaped(packed.scales_shape[1]),
                dtype: packed.dtype,
                bytes: packed.biases,
            },
        ])
    }

    fn affine_tensors(
        case: &Case,
        source: &Source,
        params_for: &dyn Fn(&str) -> AffineParams,
    ) -> Vec<PlainTensor> {
        let floats: BTreeMap<String, PlainTensor> =
            hf_tensors(case, source, DType::Float32, case.quantized)
                .into_iter()
                .map(|tensor| (tensor.name.clone(), tensor))
                .collect();
        hf_tensors(case, source, DEFAULT_PLAIN_DTYPE, case.quantized)
            .into_iter()
            .flat_map(|plain| {
                let packed = floats
                    .get(&plain.name)
                    .filter(|float| quantizable(float))
                    .and_then(|float| pack_tensor(float, params_for(&float.name)));
                packed.unwrap_or_else(|| vec![plain])
            })
            .collect()
    }

    fn assembled_header(tensors: &[PlainTensor]) -> (Header, BTreeMap<String, String>) {
        let payloads = tensors
            .iter()
            .map(|tensor| TensorPayload {
                name: tensor.name.clone(),
                dtype: tensor.dtype,
                shape: tensor.shape.clone(),
                data: &tensor.bytes,
            })
            .collect();
        let export = MlxExport::assemble(payloads, MLX_MAX_SHARD_BYTES).unwrap();
        let manifests: Vec<_> = export
            .shards
            .iter()
            .map(|shard| parse_safetensors(&shard.bytes).unwrap())
            .collect();
        let metadata = manifests[0].metadata.clone();
        let header = manifests
            .iter()
            .flat_map(|manifest| manifest.tensors.iter())
            .map(|entry| {
                let entry_shape = entry.shape.clone();
                let shape = Entry {
                    dtype: entry.dtype,
                    shape: entry_shape,
                };
                (entry.name.clone(), shape)
            })
            .collect();
        (header, metadata)
    }

    fn reference_base(name: &str) -> String {
        ["scales", "biases"]
            .iter()
            .find_map(|leaf| {
                name.strip_suffix(&format!(".{leaf}"))
                    .map(|base| format!("{base}.weight"))
            })
            .unwrap_or_else(|| name.to_string())
    }

    fn compare(exported: &Header, reference: &Header, kept: &Rows) -> Disagreement {
        let expected_shape = |name: &str, entry: &Entry| {
            let mut shape = entry.shape.clone();
            if let Some((_, rows)) = kept.get(&reference_base(name)) {
                shape[0] = *rows;
            }
            shape
        };
        let mut result = Disagreement::default();
        for (name, entry) in exported {
            let Some(wanted) = reference.get(name) else {
                result.only_export.push(name.clone());
                continue;
            };
            if entry.shape != expected_shape(name, wanted) {
                result.shape.push(format!(
                    "{name}: exported {:?} reference {:?}",
                    entry.shape, wanted.shape
                ));
            }
            if entry.dtype != wanted.dtype {
                result.dtype.push(format!(
                    "{name}: exported {:?} reference {:?}",
                    entry.dtype, wanted.dtype
                ));
            }
        }
        result.only_reference = reference
            .keys()
            .filter(|name| !exported.contains_key(*name))
            .cloned()
            .collect();
        result
    }

    fn leading_axis_disagreements(kept: &Rows, reference: &Header) -> Vec<String> {
        kept.iter()
            .filter_map(|(name, (original, _))| {
                let wanted = reference.get(name)?.shape[0];
                (wanted != *original)
                    .then(|| format!("{name}: gguf rows {original} reference rows {wanted}"))
            })
            .collect()
    }

    fn quantization_overrides(config: &Value) -> BTreeMap<String, (u8, usize)> {
        config["quantization"]
            .as_object()
            .unwrap()
            .iter()
            .filter(|(_, value)| value.is_object())
            .map(|(module, value)| {
                let bits = u8::try_from(value["bits"].as_u64().unwrap()).unwrap();
                let group = usize::try_from(value["group_size"].as_u64().unwrap()).unwrap();
                (module.clone(), (bits, group))
            })
            .collect()
    }

    fn params_from_overrides(
        overrides: &BTreeMap<String, (u8, usize)>,
        name: &str,
    ) -> AffineParams {
        let module = name.strip_suffix(".weight").unwrap_or(name);
        overrides
            .get(module)
            .map_or_else(AffineParams::default, |(bits, group)| {
                AffineParams::new(*bits, *group).unwrap()
            })
    }

    fn uniform(_: &str) -> AffineParams {
        AffineParams::default()
    }

    fn plain_export(case: &Case) -> (Header, Rows) {
        let source = source(case);
        let tensors = hf_tensors(
            case,
            &source,
            DEFAULT_PLAIN_DTYPE,
            QuantizedConventions::NONE,
        );
        (
            assembled_header(&tensors).0,
            kept_rows(case, &source, QuantizedConventions::NONE),
        )
    }

    fn affine_export(case: &Case, params_for: &dyn Fn(&str) -> AffineParams) -> (Header, Rows) {
        let source = source(case);
        let tensors = affine_tensors(case, &source, params_for);
        (
            assembled_header(&tensors).0,
            kept_rows(case, &source, case.quantized),
        )
    }

    fn prefix_counts(names: &[String]) -> BTreeMap<String, usize> {
        names.iter().fold(BTreeMap::new(), |mut counts, name| {
            let prefix = name.split('.').next().unwrap_or_default().to_string();
            *counts.entry(prefix).or_insert(0) += 1;
            counts
        })
    }

    fn counts_of(pairs: &[(&str, usize)]) -> BTreeMap<String, usize> {
        pairs
            .iter()
            .map(|(prefix, count)| ((*prefix).to_string(), *count))
            .collect()
    }

    fn key_paths(config: &Value) -> BTreeSet<String> {
        let text = config["text_config"].as_object().into_iter();
        let nested = text.flat_map(|text| text.keys().map(|key| format!("text_config.{key}")));
        let top = config.as_object().unwrap().keys().cloned();
        top.chain(nested).collect()
    }

    fn hparams_of(case: &Case, parsed: &ParsedGguf) -> ModelHparams {
        match case.family {
            "gemma4" => sliding_pattern_hparams(&sliding_from_metadata(parsed).unwrap()),
            _ => architecture_from_metadata(parsed).unwrap(),
        }
    }

    fn config_key_gap(
        case: &Case,
        flavor: &str,
        quantization: Option<(u8, u32)>,
    ) -> (Vec<String>, Vec<String>) {
        let (_, parsed) = open(case.gguf);
        let bytes = hf_config_bytes(&hparams_of(case, &parsed), &parsed, quantization).unwrap();
        let written = key_paths(&serde_json::from_slice(&bytes).unwrap());
        let vendored = key_paths(&reference_config(case.model, flavor));
        let missing = vendored.difference(&written).cloned().collect();
        let invented = written.difference(&vendored).cloned().collect();
        (missing, invented)
    }

    fn strings(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| (*item).to_string()).collect()
    }

    fn assert_plain_shapes(case: &Case) {
        let (exported, kept) = plain_export(case);
        let reference = reference_header(case.model, "plain");
        let found = compare(&exported, &reference, &kept);
        assert!(found.only_export.is_empty(), "{:?}", found.only_export);
        assert!(found.shape.is_empty(), "{:?}", found.shape);
        assert_eq!(
            leading_axis_disagreements(&kept, &reference),
            Vec::<String>::new()
        );
    }

    fn assert_plain_dtypes(case: &Case) {
        let (exported, kept) = plain_export(case);
        let found = compare(&exported, &reference_header(case.model, "plain"), &kept);
        assert!(found.dtype.is_empty(), "{:?}", found.dtype);
        assert!(
            exported
                .values()
                .all(|entry| entry.dtype == DEFAULT_PLAIN_DTYPE)
        );
    }

    fn assert_affine_names(case: &Case, reference_only: &BTreeMap<String, usize>) {
        let (exported, kept) = affine_export(case, &uniform);
        let found = compare(&exported, &reference_header(case.model, "affine4"), &kept);
        assert!(found.only_export.is_empty(), "{:?}", found.only_export);
        assert_eq!(&prefix_counts(&found.only_reference), reference_only);
    }

    fn assert_affine_shapes(case: &Case) {
        let reference = reference_header(case.model, "affine4");
        let overrides = quantization_overrides(&reference_config(case.model, "affine4"));
        let (uniform_header, kept) = affine_export(case, &uniform);
        let found = compare(&uniform_header, &reference, &kept);
        let explained: BTreeSet<String> = overrides
            .iter()
            .filter(|(_, (bits, _))| *bits != AffineParams::default().bits())
            .map(|(module, _)| format!("{module}.weight"))
            .collect();
        let differing: BTreeSet<String> = found
            .shape
            .iter()
            .filter_map(|line| line.split(':').next().map(str::to_string))
            .collect();
        assert_eq!(differing, explained, "uniform (4, 64) vs per-module bits");
        let by_override = |name: &str| params_from_overrides(&overrides, name);
        let (matched, kept) = affine_export(case, &by_override);
        assert_eq!(
            compare(&matched, &reference, &kept).shape,
            Vec::<String>::new()
        );
    }

    fn assert_affine_dtypes(case: &Case) {
        let source = source(case);
        let (exported, metadata) = assembled_header(&affine_tensors(case, &source, &uniform));
        let kept = kept_rows(case, &source, case.quantized);
        let found = compare(&exported, &reference_header(case.model, "affine4"), &kept);
        assert!(found.dtype.is_empty(), "{:?}", found.dtype);
        let reference = reference_metadata(case.model, "affine4");
        assert_eq!(metadata.get("format"), reference.get("format"));
        let extra: Vec<&String> = metadata
            .keys()
            .filter(|key| !reference.contains_key(*key))
            .collect();
        assert_eq!(extra, vec!["proxima_format_version"]);
    }

    const GRANITE_UNDERIVABLE: [&str; 8] = [
        "attention_dropout",
        "initializer_range",
        "output_router_logits",
        "rope_scaling",
        "router_aux_loss_coef",
        "torch_dtype",
        "transformers_version",
        "use_cache",
    ];

    const E2B_UNDERIVABLE: [&str; 25] = [
        "audio_config",
        "audio_token_id",
        "boa_token_id",
        "boi_token_id",
        "dtype",
        "eoa_token_id",
        "eoa_token_index",
        "eoi_token_id",
        "eos_token_id",
        "image_token_id",
        "initializer_range",
        "text_config.attention_dropout",
        "text_config.attention_k_eq_v",
        "text_config.dtype",
        "text_config.initializer_range",
        "text_config.moe_intermediate_size",
        "text_config.num_experts",
        "text_config.num_global_key_value_heads",
        "text_config.top_k_experts",
        "text_config.use_bidirectional_attention",
        "text_config.use_cache",
        "transformers_version",
        "video_token_id",
        "vision_config",
        "vision_soft_tokens_per_image",
    ];

    fn e2b_reference_only() -> BTreeMap<String, usize> {
        counts_of(&[
            ("audio_tower", 751),
            ("embed_audio", 3),
            ("embed_vision", 3),
            ("vision_tower", 658),
        ])
    }

    #[test]
    fn mlx_reference_header_granite_plain_shapes_equal_the_vendored_header() {
        assert_plain_shapes(&GRANITE);
    }

    #[test]
    fn mlx_reference_header_e2b_plain_shapes_equal_the_vendored_header_for_every_exported_name() {
        assert_plain_shapes(&E2B);
    }

    #[test]
    fn mlx_reference_header_granite_plain_dtypes_equal_the_vendored_header() {
        assert_plain_dtypes(&GRANITE);
    }

    #[test]
    fn mlx_reference_header_e2b_plain_dtypes_equal_the_vendored_header() {
        assert_plain_dtypes(&E2B);
    }

    #[test]
    fn mlx_reference_header_granite_affine_names_equal_the_synthetic_header() {
        assert_affine_names(&GRANITE, &BTreeMap::new());
    }

    #[test]
    fn mlx_reference_header_e2b_affine_names_lack_only_the_vision_and_audio_towers() {
        assert_affine_names(&E2B, &e2b_reference_only());
    }

    #[test]
    fn mlx_reference_header_granite_affine_shapes_equal_the_synthetic_header() {
        assert_affine_shapes(&GRANITE);
    }

    #[test]
    fn mlx_reference_header_e2b_affine_shapes_differ_only_where_the_reference_uses_8_bits() {
        assert_affine_shapes(&E2B);
    }

    #[test]
    fn mlx_reference_header_granite_affine_dtypes_equal_the_synthetic_header() {
        assert_affine_dtypes(&GRANITE);
    }

    #[test]
    fn mlx_reference_header_e2b_affine_dtypes_equal_the_vendored_header() {
        assert_affine_dtypes(&E2B);
    }

    #[test]
    fn mlx_reference_header_granite_plain_config_lacks_only_the_underivable_keys() {
        let (missing, invented) = config_key_gap(&GRANITE, "plain", None);
        assert_eq!(missing, strings(&GRANITE_UNDERIVABLE));
        assert_eq!(invented, Vec::<String>::new());
    }

    #[test]
    fn mlx_reference_header_granite_affine_config_lacks_only_the_underivable_keys() {
        let (missing, invented) = config_key_gap(&GRANITE, "affine4", Some((4, 64)));
        assert_eq!(missing, strings(&GRANITE_UNDERIVABLE));
        assert_eq!(invented, Vec::<String>::new());
    }

    #[test]
    fn mlx_reference_header_e2b_plain_config_nests_the_text_keys_and_lacks_only_the_underivable_keys()
     {
        let (missing, invented) = config_key_gap(&E2B, "plain", None);
        assert_eq!(missing, strings(&E2B_UNDERIVABLE));
        assert_eq!(invented, Vec::<String>::new());
    }

    #[test]
    fn mlx_reference_header_e2b_affine_config_nests_the_text_keys_and_lacks_only_the_underivable_keys()
     {
        let (missing, invented) = config_key_gap(&E2B, "affine4", Some((4, 64)));
        assert_eq!(missing, strings(&E2B_UNDERIVABLE));
        assert_eq!(invented, Vec::<String>::new());
    }

    #[test]
    fn mlx_reference_header_identity_control_reference_compared_to_itself_agrees() {
        let reference = reference_header(E2B.model, "affine4");

        let found = compare(&reference, &reference, &BTreeMap::new());

        assert!(reference.len() > 2000);
        assert_eq!(found, Disagreement::default());
    }

    #[test]
    fn mlx_reference_header_degenerate_control_one_renamed_tensor_is_located() {
        let reference = reference_header(GRANITE.model, "affine4");
        let mut altered = reference.clone();
        let entry = altered
            .remove("model.layers.3.self_attn.q_proj.scales")
            .unwrap();
        altered.insert("model.layers.3.self_attn.q_proj.scale".to_string(), entry);

        let found = compare(&altered, &reference, &BTreeMap::new());

        assert_eq!(
            found.only_export,
            strings(&["model.layers.3.self_attn.q_proj.scale"])
        );
        assert_eq!(
            found.only_reference,
            strings(&["model.layers.3.self_attn.q_proj.scales"])
        );
        assert!(found.shape.is_empty() && found.dtype.is_empty());
    }
}
