#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;

use proxima_primitives::pipe::sans_io::Outcome;
use proxima_safetensors::{SafetensorsModel, SafetensorsParser, TensorPayload, write_complete};

use super::*;
use crate::profiles::binding_profile;

const HEADS: u32 = 2;
const HEAD_DIM: u32 = 2;
const HIDDEN: u32 = 3;
const Q_PROJ: &str = "model.layers.0.self_attn.q_proj.weight";

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn checkpoint(stored_shape: [u64; 2]) -> (Vec<u8>, Manifest, u64) {
    let rows = (stored_shape[0] * stored_shape[1]) as u32;
    let values: Vec<f32> = (1..=rows).map(|value| value as f32).collect();
    let data = f32_bytes(&values);
    let file = write_complete(&SafetensorsModel {
        tensors: vec![TensorPayload {
            name: Q_PROJ.into(),
            dtype: DType::FLOAT32,
            shape: stored_shape.to_vec(),
            data: &data,
        }],
        metadata: BTreeMap::new(),
    })
    .expect("writes a real safetensors buffer");
    let header_length = u64::from_le_bytes(file[..8].try_into().expect("8-byte prefix"));
    let data_start = 8 + header_length;
    let mut parser = SafetensorsParser::new();
    parser.feed(&file[..data_start as usize]);
    let Outcome::Event(manifest) = parser.poll().expect("header parses") else {
        panic!("a complete header must produce a manifest");
    };
    let manifest = manifest.clone();
    (file, manifest, data_start)
}

fn rank_three_leaf() -> [Extent; 3] {
    [
        Extent::Static(HIDDEN),
        Extent::Static(HEADS),
        Extent::Static(HEAD_DIM),
    ]
}

fn view_of<'file>(
    file: &'file [u8],
    manifest: &'file Manifest,
    data_start: u64,
    shape: &[Extent],
) -> Result<SafetensorsTensorView<'file, 'file>, InteropError> {
    let entry = manifest.tensor(Q_PROJ).expect("q_proj is stored");
    validated_view(
        file,
        data_start,
        "blk.0.attn_q.weight",
        shape,
        Role::InOut,
        entry,
    )
}

#[test]
fn hf_inout_view_rank_three_attention_leaf_matches_the_stored_out_in_matrix() {
    let (file, manifest, data_start) = checkpoint([4, 3]);

    let view = view_of(&file, &manifest, data_start, &rank_three_leaf());

    assert!(
        view.is_ok(),
        "[3, 2, 2] is a view of stored [4, 3]: {view:?}"
    );
}

#[test]
fn hf_inout_view_rank_three_output_leaf_splits_after_the_head_axes() {
    assert!(inout_view_matches(&[1024, 1024], &[16, 64, 1024]));
    assert!(inout_view_matches(&[1024, 1024], &[1024, 16, 64]));
    assert!(inout_view_matches(&[512, 1024], &[1024, 8, 64]));
}

#[test]
fn hf_inout_view_refuses_a_stored_matrix_whose_products_disagree() {
    let (file, manifest, data_start) = checkpoint([5, 3]);

    let refused = view_of(&file, &manifest, data_start, &rank_three_leaf());

    assert!(matches!(
        refused,
        Err(InteropError::SafetensorsAxesMismatch { .. })
    ));
}

#[test]
fn hf_inout_view_refuses_a_stored_tensor_that_is_not_a_matrix() {
    assert!(!inout_view_matches(&[4, 3, 1], &[3, 2, 2]));
    assert!(!inout_view_matches(&[12], &[3, 2, 2]));
}

#[test]
fn hf_inout_view_rank_three_f32_weight_transposes_with_the_stored_extents() {
    let (file, manifest, data_start) = checkpoint([4, 3]);
    let view = view_of(&file, &manifest, data_start, &rank_three_leaf()).expect("view validates");
    let mut weights = BoundWeights::new(&[]);
    let binding = binding_profile("granitemoe").expect("granitemoe parses");

    bind_safetensors_leaf(&manifest, &file, data_start, &binding, &view, &mut weights)
        .expect("the leaf binds");

    let (_, values) = weights
        .owned
        .first()
        .expect("an f32 in_out weight binds owned");
    let expected: Vec<f32> = (0..3)
        .flat_map(|column| (0..4).map(move |row| (row * 3 + column + 1) as f32))
        .collect();
    assert_eq!(values, &expected, "[hidden][heads * head_dim] row-major");
}
