#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeSet;
use std::path::Path;

use proxima_primitives::pipe::sans_io::Outcome;
use proxima_safetensors::{Manifest, SafetensorsParser};
use serde_json::Value;

use super::*;
use crate::profiles::binding_profile;

const LAYER: u32 = 7;
const EXPERTS: u32 = 32;

fn header_names() -> BTreeSet<String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/mlx-reference/granite-3.1-1b-a400m-moe/plain/header.json");
    let header: Value =
        serde_json::from_slice(&std::fs::read(path).expect("vendored header exists"))
            .expect("vendored header is json");
    header
        .as_object()
        .expect("header is an object")
        .keys()
        .cloned()
        .collect()
}

fn reference_manifest() -> Manifest {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/mlx-reference/granite-3.1-1b-a400m-moe/plain/header.json");
    let json = std::fs::read(path).expect("vendored header exists");
    let mut framed = (json.len() as u64).to_le_bytes().to_vec();
    framed.extend_from_slice(&json);
    let mut parser = SafetensorsParser::new();
    parser.feed(&framed);
    match parser.poll().expect("vendored header parses") {
        Outcome::Event(manifest) => manifest.clone(),
        Outcome::NeedMore => panic!("the complete vendored header must produce a manifest"),
    }
}
fn first_alias(binding: &BindingProfile, leaf: &str) -> TensorAlias {
    binding
        .aliases_for(leaf)
        .next()
        .map(|(alias, _)| alias.clone())
        .unwrap_or_else(|| panic!("no alias for {leaf}"))
}

fn stack_shape(rows: u32) -> Vec<Extent> {
    vec![
        Extent::Static(EXPERTS),
        Extent::Static(1024),
        Extent::Static(rows),
    ]
}

#[test]
fn granitemoe_binding_profile_is_registered_and_is_not_the_default() {
    let granite = binding_profile("granitemoe").expect("granitemoe parses");
    let default = binding_profile("llama").expect("defaults parse");

    assert_eq!(granite.aliases.len(), default.aliases.len() + 24 * 4);
    assert_ne!(
        first_alias(&granite, "blk.0.ffn_gate_exps.weight"),
        first_alias(&default, "blk.0.ffn_gate_exps.weight")
    );
}

#[test]
fn granitemoe_binding_gate_and_up_are_parts_zero_and_one_of_input_linear() {
    let binding = binding_profile("granitemoe").expect("granitemoe parses");
    let stored = format!("model.layers.{LAYER}.block_sparse_moe.input_linear.weight");
    let part = |leaf: &str, index: u32| TensorAlias::Part {
        leaf: format!("blk.{LAYER}.{leaf}"),
        from: stored.clone(),
        part: index,
        of: 2,
    };

    let gate = first_alias(&binding, &format!("blk.{LAYER}.ffn_gate_exps.weight"));
    let up = first_alias(&binding, &format!("blk.{LAYER}.ffn_up_exps.weight"));
    let fused: Vec<f32> = (0..2 * 4 * 3).map(|value| value as f32).collect();
    let gate_rows = expert_row_part(&fused, 2, 4, 3, 0, 2);
    let up_rows = expert_row_part(&fused, 2, 4, 3, 1, 2);

    assert_eq!(
        (gate, up),
        (
            part("ffn_gate_exps.weight", 0),
            part("ffn_up_exps.weight", 1)
        )
    );
    assert!(
        header_names().contains(&stored),
        "{stored} absent from the vendored header"
    );
    assert_eq!(
        gate_rows,
        [
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0
        ]
    );
    assert_eq!(
        up_rows,
        [
            6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0
        ]
    );
}

#[test]
fn granitemoe_binding_down_is_output_linear() {
    let binding = binding_profile("granitemoe").expect("granitemoe parses");
    let stored = format!("model.layers.{LAYER}.block_sparse_moe.output_linear.weight");

    let down = first_alias(&binding, &format!("blk.{LAYER}.ffn_down_exps.weight"));

    assert_eq!(
        down,
        TensorAlias::Rename {
            leaf: format!("blk.{LAYER}.ffn_down_exps.weight"),
            from: stored.clone()
        }
    );
    assert!(
        header_names().contains(&stored),
        "{stored} absent from the vendored header"
    );
}

#[test]
fn granitemoe_binding_router_is_router_layer() {
    let binding = binding_profile("granitemoe").expect("granitemoe parses");
    let stored = format!("model.layers.{LAYER}.block_sparse_moe.router.layer.weight");

    let router = first_alias(&binding, &format!("blk.{LAYER}.ffn_gate_inp.weight"));

    assert_eq!(
        router,
        TensorAlias::Rename {
            leaf: format!("blk.{LAYER}.ffn_gate_inp.weight"),
            from: stored.clone()
        }
    );
    assert!(
        header_names().contains(&stored),
        "{stored} absent from the vendored header"
    );
}

#[test]
fn granitemoe_binding_dense_profile_refuses_moe_leaves() {
    let manifest = reference_manifest();
    let dense = binding_profile("llama").expect("defaults parse");
    let granite = binding_profile("granitemoe").expect("granitemoe parses");
    let router_shape = [Extent::Static(1024), Extent::Static(EXPERTS)];
    let gate = "blk.0.ffn_gate_exps.weight";

    let stack = moe_stack_source(&manifest, &dense, gate, &stack_shape(512));
    let router = safetensors_source(
        &manifest,
        &dense,
        "blk.0.ffn_gate_inp.weight",
        &router_shape,
    );
    let routed = moe_stack_source(&manifest, &granite, gate, &stack_shape(512));

    assert!(matches!(
        stack,
        Err(InteropError::HfMoeWeightsUnsupported {
            expert_count: EXPERTS
        })
    ));
    assert!(matches!(
        router,
        Err(InteropError::HfMoeWeightsUnsupported {
            expert_count: EXPERTS
        })
    ));
    assert_eq!(
        routed.expect("granitemoe names the stack"),
        (
            "model.layers.0.block_sparse_moe.input_linear.weight".to_string(),
            0,
            2
        )
    );
}
