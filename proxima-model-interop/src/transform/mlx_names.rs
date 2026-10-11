//! Dense GGUF leaf to Hugging Face tensor name, for the MLX export.
//!
//! A leaf the tensor directory does not store (a tied output projection)
//! resolves through [`BindingProfile::aliases_for`] to the stored tensor it
//! reads, the loader's own rule run in reverse: the HF name of the leaf is
//! the HF name of the tensor that holds its bytes. A leaf nothing stores, or
//! that the layout table has no row for, is reported in
//! [`MlxNames::unmapped`] and no name is produced for it.
//!
//! The layout rows are the same names `hf_bind::names` reads
//! checkpoints with; `STANDARD_LAYOUT` is checked against that module in the
//! tests so the two cannot drift.

use alloc::collections::BTreeSet;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use crate::profiles::{BindingProfile, TensorAlias};

/// Rows of `(gguf name, hf name)`. A `blk.N.` leaf matches on the text after
/// `blk.N.` and maps under `layers.N.`; any other leaf matches whole.
pub type Layout = &'static [(&'static str, &'static str)];

/// Llama-style decoder: two norms per layer, `lm_head` tied or absent.
pub const STANDARD_LAYOUT: Layout = &[
    ("token_embd.weight", "embed_tokens.weight"),
    ("output_norm.weight", "norm.weight"),
    ("attn_norm.weight", "input_layernorm.weight"),
    ("ffn_norm.weight", "post_attention_layernorm.weight"),
    ("attn_q.weight", "self_attn.q_proj.weight"),
    ("attn_k.weight", "self_attn.k_proj.weight"),
    ("attn_v.weight", "self_attn.v_proj.weight"),
    ("attn_output.weight", "self_attn.o_proj.weight"),
    ("ffn_gate.weight", "mlp.gate_proj.weight"),
    ("ffn_up.weight", "mlp.up_proj.weight"),
    ("ffn_down.weight", "mlp.down_proj.weight"),
];

/// Four norms per layer, per-head q/k norms and per-layer input tables.
pub const SANDWICH_NORM_LAYOUT: Layout = &[
    ("token_embd.weight", "embed_tokens.weight"),
    ("output_norm.weight", "norm.weight"),
    (
        "per_layer_token_embd.weight",
        "embed_tokens_per_layer.weight",
    ),
    (
        "per_layer_model_proj.weight",
        "per_layer_model_projection.weight",
    ),
    (
        "per_layer_proj_norm.weight",
        "per_layer_projection_norm.weight",
    ),
    ("attn_norm.weight", "input_layernorm.weight"),
    (
        "post_attention_norm.weight",
        "post_attention_layernorm.weight",
    ),
    ("ffn_norm.weight", "pre_feedforward_layernorm.weight"),
    ("post_ffw_norm.weight", "post_feedforward_layernorm.weight"),
    ("attn_q.weight", "self_attn.q_proj.weight"),
    ("attn_k.weight", "self_attn.k_proj.weight"),
    ("attn_v.weight", "self_attn.v_proj.weight"),
    ("attn_output.weight", "self_attn.o_proj.weight"),
    ("attn_q_norm.weight", "self_attn.q_norm.weight"),
    ("attn_k_norm.weight", "self_attn.k_norm.weight"),
    ("ffn_gate.weight", "mlp.gate_proj.weight"),
    ("ffn_up.weight", "mlp.up_proj.weight"),
    ("ffn_down.weight", "mlp.down_proj.weight"),
    ("inp_gate.weight", "per_layer_input_gate.weight"),
    ("proj.weight", "per_layer_projection.weight"),
    ("post_norm.weight", "post_per_layer_input_norm.weight"),
    ("layer_output_scale.weight", "layer_scalar"),
];

/// The outcome of naming a set of leaves.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MlxNames {
    /// `(leaf, hf name)` in input order, e.g. `("blk.0.attn_q.weight",
    /// "language_model.model.layers.0.self_attn.q_proj.weight")`.
    pub mapped: Vec<(String, String)>,
    /// Leaves that are one slice of a fused HF tensor, in input order, e.g.
    /// `blk.0.ffn_gate_exps.weight` and `blk.0.ffn_up_exps.weight`, which are
    /// parts 0 and 1 of 2 of one `input_linear.weight`.
    pub fused: Vec<FusedName>,
    /// Leaves with no stored tensor or no layout row, e.g.
    /// `blk.3.mystery_gate.weight`; never given an invented name.
    pub unmapped: Vec<String>,
}

/// One leaf's slice of a fused HF tensor: part `part` of `of` equal slices of
/// the row axis, the same relation as `TensorAlias::Part`, which the loader
/// binds with. An exporter emits the fused tensor `hf` once, from all `of`
/// leaves that share it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusedName {
    /// The GGUF leaf, e.g. `blk.0.ffn_up_exps.weight`.
    pub leaf: String,
    /// The fused HF tensor, e.g. `model.layers.0.block_sparse_moe.input_linear.weight`.
    pub hf: String,
    /// Zero-based slice index, e.g. `1`.
    pub part: u32,
    /// Slice count of the fused tensor, e.g. `2`.
    pub of: u32,
}

enum Source {
    Stored(String),
    Literal {
        hf: String,
        slice: Option<(u32, u32)>,
    },
}

fn layer_split(leaf: &str) -> Option<(&str, &str)> {
    let (index, rest) = leaf.strip_prefix("blk.")?.split_once('.')?;
    index
        .bytes()
        .all(|byte| byte.is_ascii_digit())
        .then_some((index, rest))
}

fn has_layout_row(name: &str, layout: Layout) -> bool {
    let key = layer_split(name).map_or(name, |(_, rest)| rest);
    layout.iter().any(|(gguf, _)| *gguf == key)
}

fn alias_source(
    alias: &TensorAlias,
    prefix: &str,
    directory: &BTreeSet<&str>,
    layout: Layout,
) -> Option<Source> {
    match alias {
        TensorAlias::Rename { from, .. } => {
            let source = format!("{prefix}{from}");
            match has_layout_row(&source, layout) {
                true => directory
                    .contains(source.as_str())
                    .then_some(Source::Stored(source)),
                false => Some(Source::Literal {
                    hf: source,
                    slice: None,
                }),
            }
        }
        TensorAlias::Part { from, part, of, .. } => Some(Source::Literal {
            hf: format!("{prefix}{from}"),
            slice: Some((*part, *of)),
        }),
        TensorAlias::Join { .. } => None,
    }
}

fn resolve(
    leaf: &str,
    directory: &BTreeSet<&str>,
    binding: &BindingProfile,
    layout: Layout,
) -> Option<Source> {
    if directory.contains(leaf) && has_layout_row(leaf, layout) {
        return Some(Source::Stored(leaf.to_string()));
    }
    binding
        .aliases_for(leaf)
        .find_map(|(alias, prefix)| alias_source(alias, prefix, directory, layout))
}

fn hf_name(stored: &str, layout: Layout, hf_prefix: &str) -> Option<String> {
    let row_for = |key: &str| {
        layout
            .iter()
            .find(|(gguf, _)| *gguf == key)
            .map(|(_, hf)| *hf)
    };
    match layer_split(stored) {
        Some((index, rest)) => row_for(rest).map(|hf| format!("{hf_prefix}layers.{index}.{hf}")),
        None => row_for(stored).map(|hf| format!("{hf_prefix}{hf}")),
    }
}

fn place(
    names: &mut MlxNames,
    leaf: &str,
    source: Option<Source>,
    layout: Layout,
    hf_prefix: &str,
) {
    let leaf_name = leaf.to_string();
    match source {
        Some(Source::Stored(stored)) => match hf_name(&stored, layout, hf_prefix) {
            Some(hf) => names.mapped.push((leaf_name, hf)),
            None => names.unmapped.push(leaf_name),
        },
        Some(Source::Literal { hf, slice: None }) => names.mapped.push((leaf_name, hf)),
        Some(Source::Literal {
            hf,
            slice: Some((part, of)),
        }) => names.fused.push(FusedName {
            leaf: leaf_name,
            hf,
            part,
            of,
        }),
        None => names.unmapped.push(leaf_name),
    }
}

/// Names `leaves` for an HF checkpoint whose tensors sit under `hf_prefix`
/// (`"model."`, `"language_model.model."`). `directory` is the GGUF tensor
/// directory the leaves are bound from. Composes
/// [`BindingProfile::aliases_for`], the loader's alias lookup, in reverse; the
/// layout rows are the loader's own HF names (`hf_bind::names`).
///
/// An alias whose source has a layout row names a stored GGUF tensor (a tied
/// output). An alias whose source has none is already a literal HF name, as the
/// routed-expert aliases are (`hf_bind::moe_stack_source`): a rename maps the
/// leaf to it, a part alias lands in [`MlxNames::fused`].
#[must_use]
pub fn dense_names(
    leaves: &[&str],
    directory: &[&str],
    binding: &BindingProfile,
    layout: Layout,
    hf_prefix: &str,
) -> MlxNames {
    let stored_set: BTreeSet<&str> = directory.iter().copied().collect();
    leaves.iter().fold(MlxNames::default(), |mut names, leaf| {
        let source = resolve(leaf, &stored_set, binding, layout);
        place(&mut names, leaf, source, layout, hf_prefix);
        names
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs::File;
    use std::path::Path;

    use proxima_gguf::parse_complete;
    use serde_json::Value;

    use super::*;
    use crate::hf_bind::names;
    use crate::profiles::binding_profile;

    const E2B_GGUF: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-3646b4c147cd235a44d91df1546d3b7d8e29b547dbe4e1f80856419aa455e6fd";
    const GRANITE_GGUF: &str = "/Users/brianbruggeman/.ollama/models/blobs/sha256-cd60b3e8bb445d4c05e0b0b99b1bb41e8bb77211b161e783c71931168131df80";
    const E2B_PREFIX: &str = "language_model.model.";
    const GRANITE_PREFIX: &str = "model.";

    fn reference_dir(model: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/mlx-reference")
            .join(model)
            .join("plain")
    }

    fn header_shapes(model: &str) -> BTreeMap<String, Vec<u64>> {
        let mut found = BTreeMap::new();
        for entry in std::fs::read_dir(reference_dir(model)).expect("reference dir exists") {
            let path = entry.expect("dir entry").path();
            let file_name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_string();
            if file_name.starts_with("header") && file_name.ends_with(".json") {
                let header: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
                found.extend(
                    header
                        .as_object()
                        .unwrap()
                        .iter()
                        .filter(|(key, _)| *key != "__metadata__")
                        .map(|(key, entry)| {
                            let shape = entry["shape"].as_array().unwrap();
                            (
                                key.clone(),
                                shape.iter().map(|dim| dim.as_u64().unwrap()).collect(),
                            )
                        }),
                );
            }
        }
        found
    }

    fn header_names(model: &str) -> BTreeSet<String> {
        header_shapes(model).into_keys().collect()
    }

    fn gguf_infos(path: &str) -> Vec<proxima_gguf::TensorInfo> {
        assert!(
            Path::new(path).exists(),
            "host-local gguf blob missing at {path}"
        );
        let file = File::open(path).unwrap();
        let mapping = unsafe { memmap2::Mmap::map(&file) }.unwrap();
        parse_complete(&mapping).unwrap().tensors
    }

    fn gguf_directory(path: &str) -> Vec<String> {
        gguf_infos(path)
            .into_iter()
            .map(|tensor| tensor.name)
            .collect()
    }

    fn program_leaves<'a>(directory: &'a [String], binding: &BindingProfile) -> Vec<&'a str> {
        directory
            .iter()
            .map(String::as_str)
            .filter(|name| !binding.extra.iter().any(|extra| extra == name))
            .collect()
    }

    fn name_dense(
        path: &str,
        family: &str,
        layout: Layout,
        prefix: &str,
    ) -> (MlxNames, Vec<String>) {
        let directory = gguf_directory(path);
        let binding = binding_profile(family).unwrap();
        let mut leaves = program_leaves(&directory, &binding);
        leaves.push("output.weight");
        let stored: Vec<&str> = directory.iter().map(String::as_str).collect();
        (
            dense_names(&leaves, &stored, &binding, layout, prefix),
            directory,
        )
    }

    #[test]
    fn mlx_names_dense_e2b_leaves_match_plain_header_with_zero_unmapped() {
        let (named, directory) = name_dense(E2B_GGUF, "gemma4", SANDWICH_NORM_LAYOUT, E2B_PREFIX);
        let header = header_names("gemma4-e2b");

        let missing: Vec<&str> = named
            .mapped
            .iter()
            .map(|(_, hf)| hf.as_str())
            .filter(|hf| !header.contains(*hf))
            .collect();

        assert!(named.unmapped.is_empty(), "unmapped: {:?}", named.unmapped);
        assert!(
            missing.is_empty(),
            "names absent from the plain header: {missing:?}"
        );
        assert_eq!(
            named.mapped.len(),
            directory.len(),
            "every stored tensor but rope_freqs, plus the tied output leaf"
        );
    }

    #[test]
    fn mlx_names_dense_granite_leaves_match_plain_header_with_zero_unmapped() {
        let (named, _) = name_dense(GRANITE_GGUF, "granitemoe", STANDARD_LAYOUT, GRANITE_PREFIX);
        let header = header_names("granite-3.1-1b-a400m-moe");

        let missing: Vec<&str> = named
            .mapped
            .iter()
            .map(|(_, hf)| hf.as_str())
            .filter(|hf| !header.contains(*hf))
            .collect();

        assert!(
            missing.is_empty(),
            "names absent from the plain header: {missing:?}"
        );
        assert!(named.unmapped.is_empty(), "unmapped: {:?}", named.unmapped);
        assert!(
            named
                .mapped
                .iter()
                .any(|(leaf, hf)| leaf == "output.weight" && hf == "model.embed_tokens.weight")
        );
    }

    #[test]
    fn mlx_names_dense_identity_control_reference_names_compare_equal_to_themselves() {
        let (named, _) = name_dense(GRANITE_GGUF, "granitemoe", STANDARD_LAYOUT, GRANITE_PREFIX);
        let reference: Vec<String> = named.mapped.iter().map(|(_, hf)| hf.clone()).collect();

        let compared: Vec<(&String, &String)> = reference
            .iter()
            .zip(reference.iter())
            .filter(|(left, right)| left != right)
            .collect();

        assert!(!reference.is_empty() && compared.is_empty());
    }

    #[test]
    fn mlx_names_dense_degenerate_control_one_altered_name_is_detected() {
        let (named, _) = name_dense(GRANITE_GGUF, "granitemoe", STANDARD_LAYOUT, GRANITE_PREFIX);
        let header = header_names("granite-3.1-1b-a400m-moe");
        let mut altered: Vec<String> = named.mapped.iter().map(|(_, hf)| hf.clone()).collect();
        altered[3] = altered[3].replace("layers", "layer");

        let missing = altered
            .iter()
            .filter(|hf| !header.contains(hf.as_str()))
            .count();

        assert_eq!(
            missing, 1,
            "exactly the altered name must fail the header comparison"
        );
    }

    #[test]
    fn mlx_names_dense_leaf_without_a_profile_entry_is_unmapped_and_gets_no_name() {
        let binding = binding_profile("granitemoe").unwrap();
        let directory = [
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "blk.0.mystery_gate.weight",
        ];

        let named = dense_names(
            &directory,
            &directory,
            &binding,
            STANDARD_LAYOUT,
            GRANITE_PREFIX,
        );

        assert_eq!(
            named.unmapped,
            vec!["blk.0.mystery_gate.weight".to_string()]
        );
        assert_eq!(named.mapped.len(), 2);
        assert!(
            named
                .mapped
                .iter()
                .all(|(leaf, _)| leaf != "blk.0.mystery_gate.weight")
        );
    }

    fn routed(name: &str) -> bool {
        name.contains("_exps.") || name.contains("ffn_gate_inp.")
    }

    fn gguf_shape(infos: &[proxima_gguf::TensorInfo], leaf: &str) -> Vec<u64> {
        let info = infos.iter().find(|info| info.name == leaf).unwrap();
        info.dims.iter().rev().copied().collect()
    }

    fn moe_header_names(named: &MlxNames) -> Vec<String> {
        let routed_mapped = named.mapped.iter().filter(|(leaf, _)| routed(leaf));
        let fused = named.fused.iter().map(|fused| &fused.hf);
        routed_mapped
            .map(|(_, hf)| hf.clone())
            .chain(fused.cloned())
            .collect()
    }

    #[test]
    fn mlx_names_moe_leaves_match_plain_header_names_shapes_and_fused_parts() {
        let (named, _) = name_dense(GRANITE_GGUF, "granitemoe", STANDARD_LAYOUT, GRANITE_PREFIX);
        let infos = gguf_infos(GRANITE_GGUF);
        let header = header_shapes("granite-3.1-1b-a400m-moe");

        let whole_mismatch: Vec<&str> = named
            .mapped
            .iter()
            .filter(|(leaf, _)| routed(leaf))
            .filter(|(leaf, hf)| header.get(hf) != Some(&gguf_shape(&infos, leaf)))
            .map(|(leaf, _)| leaf.as_str())
            .collect();
        let fused_mismatch: Vec<&str> = named
            .fused
            .iter()
            .filter(|fused| {
                let mut shape = gguf_shape(&infos, &fused.leaf);
                shape[1] *= u64::from(fused.of);
                header.get(&fused.hf) != Some(&shape)
            })
            .map(|fused| fused.leaf.as_str())
            .collect();

        assert!(
            whole_mismatch.is_empty(),
            "shape or name differs: {whole_mismatch:?}"
        );
        assert!(
            fused_mismatch.is_empty(),
            "fused shape or name differs: {fused_mismatch:?}"
        );
        assert_eq!(named.fused.len(), 48, "gate and up of 24 layers");
        let gate = named
            .fused
            .iter()
            .find(|fused| fused.leaf == "blk.7.ffn_gate_exps.weight");
        let up = named
            .fused
            .iter()
            .find(|fused| fused.leaf == "blk.7.ffn_up_exps.weight");
        let relation = gate
            .zip(up)
            .map(|(gate, up)| (gate.hf == up.hf, (gate.part, gate.of), (up.part, up.of)));
        assert_eq!(relation, Some((true, (0, 2), (1, 2))));
    }

    #[test]
    fn mlx_names_moe_every_routed_leaf_is_named_and_unmapped_is_empty() {
        let (named, directory) =
            name_dense(GRANITE_GGUF, "granitemoe", STANDARD_LAYOUT, GRANITE_PREFIX);
        let routed_leaves = directory.iter().filter(|name| routed(name)).count();

        let routed_named =
            named.mapped.iter().filter(|(leaf, _)| routed(leaf)).count() + named.fused.len();

        assert!(named.unmapped.is_empty(), "unmapped: {:?}", named.unmapped);
        assert_eq!(routed_named, routed_leaves);
        assert_eq!(routed_leaves, 96);
    }

    #[test]
    fn mlx_names_moe_identity_control_reference_names_compare_equal_to_themselves() {
        let (named, _) = name_dense(GRANITE_GGUF, "granitemoe", STANDARD_LAYOUT, GRANITE_PREFIX);
        let reference = moe_header_names(&named);

        let differing = reference
            .iter()
            .zip(reference.iter())
            .filter(|(left, right)| left != right)
            .count();

        assert!(!reference.is_empty() && differing == 0);
    }

    #[test]
    fn mlx_names_moe_degenerate_control_one_altered_name_is_detected() {
        let (named, _) = name_dense(GRANITE_GGUF, "granitemoe", STANDARD_LAYOUT, GRANITE_PREFIX);
        let header = header_names("granite-3.1-1b-a400m-moe");
        let mut altered = moe_header_names(&named);
        altered[3] = altered[3].replace("block_sparse_moe", "sparse_moe");

        let missing = altered.iter().filter(|hf| !header.contains(*hf)).count();

        assert_eq!(
            missing, 1,
            "exactly the altered name must fail the header comparison"
        );
    }

    #[test]
    fn standard_layout_agrees_with_the_loader_names() {
        let layer = 3;
        let expected = [
            ("blk.3.attn_norm.weight", names::input_layernorm(layer)),
            (
                "blk.3.ffn_norm.weight",
                names::post_attention_layernorm(layer),
            ),
            ("blk.3.attn_q.weight", names::q_proj(layer)),
            ("blk.3.attn_k.weight", names::k_proj(layer)),
            ("blk.3.attn_v.weight", names::v_proj(layer)),
            ("blk.3.attn_output.weight", names::o_proj(layer)),
            ("blk.3.ffn_gate.weight", names::gate_proj(layer)),
            ("blk.3.ffn_up.weight", names::up_proj(layer)),
            ("blk.3.ffn_down.weight", names::down_proj(layer)),
            ("token_embd.weight", names::embed_tokens()),
            ("output_norm.weight", names::final_norm()),
        ];
        let leaves: Vec<&str> = expected.iter().map(|(leaf, _)| *leaf).collect();

        let named = dense_names(
            &leaves,
            &leaves,
            &BindingProfile::default(),
            STANDARD_LAYOUT,
            "model.",
        );

        let produced: Vec<(&str, &str)> = named
            .mapped
            .iter()
            .map(|(leaf, hf)| (leaf.as_str(), hf.as_str()))
            .collect();
        let wanted: Vec<(&str, &str)> = expected
            .iter()
            .map(|(leaf, hf)| (*leaf, hf.as_str()))
            .collect();
        assert_eq!(produced, wanted);
    }
}
