// workspace denies expect_used; a failed fixture must abort the test with its message
#![allow(clippy::expect_used)]

//! Which Q/K row order reaches the graph. Hugging Face stores each head's rows
//! in `rotate_half` order, which is split-half RoPE pairing; an adjacent pairing
//! needs those rows reordered on load. The choice is the descriptor's RoPE
//! pairing, never whether the file happens to hold Q/K norm tensors, and a head
//! of width four is the smallest that tells the orders apart.

use proxima_safetensors::{SafetensorsModel, TensorPayload, parse_complete, write_complete};
use proxima_tensor::DType;

use super::*;

fn width_four_projection() -> (Vec<u8>, Manifest, u64) {
    let source: Vec<u8> = (1..=16)
        .flat_map(|value| (value as f32).to_le_bytes())
        .collect();
    let bytes = write_complete(&SafetensorsModel {
        tensors: vec![TensorPayload {
            name: "q".into(),
            dtype: DType::Float32,
            shape: vec![4, 4],
            data: &source,
        }],
        metadata: Default::default(),
    })
    .expect("write the width-four projection");
    let manifest = parse_complete(&bytes).expect("parse the projection");
    let start = u64::from_le_bytes(bytes[..8].try_into().expect("eight header bytes")) + 8;
    (bytes, manifest, start)
}

fn bound_rows(permute: bool) -> Vec<f32> {
    let (bytes, manifest, start) = width_four_projection();
    let mut weights = BoundWeights::new(&[]);
    hf_bind_rope_weight(
        &manifest,
        &bytes,
        start,
        "q".into(),
        "blk.0.attn_q.weight".into(),
        1,
        4,
        4,
        permute,
        &mut weights,
    )
    .expect("bind the projection");
    weights.owned[0].1.clone()
}

#[test]
fn a_split_half_pairing_keeps_the_rows_as_stored() {
    assert_eq!(
        bound_rows(false),
        [
            1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0, 16.0
        ]
    );
}

#[test]
fn an_adjacent_pairing_reorders_rotate_half_rows_into_adjacent_pairs() {
    let stored = bound_rows(false);
    let reordered = bound_rows(true);

    assert_ne!(stored, reordered, "the wrong pairing must change the rows");
    assert_eq!(
        reordered,
        [
            1.0, 9.0, 5.0, 13.0, 2.0, 10.0, 6.0, 14.0, 3.0, 11.0, 7.0, 15.0, 4.0, 12.0, 8.0, 16.0
        ],
        "output rows 0, 2, 1, 3: the pairs (0, 2) and (1, 3) become adjacent"
    );
}

#[test]
fn both_orders_rotate_a_vector_to_the_same_values() {
    let split_half = [1.0f32, 2.0, 3.0, 4.0];
    let adjacent = [1.0f32, 3.0, 2.0, 4.0];
    let cosines = [0.0f32, 1.0];
    let sines = [1.0f32, 0.0];

    let rotated_split = [
        split_half[0] * cosines[0] - split_half[2] * sines[0],
        split_half[1] * cosines[1] - split_half[3] * sines[1],
        split_half[2] * cosines[0] + split_half[0] * sines[0],
        split_half[3] * cosines[1] + split_half[1] * sines[1],
    ];
    let rotated_adjacent = [
        adjacent[0] * cosines[0] - adjacent[1] * sines[0],
        adjacent[0] * sines[0] + adjacent[1] * cosines[0],
        adjacent[2] * cosines[1] - adjacent[3] * sines[1],
        adjacent[2] * sines[1] + adjacent[3] * cosines[1],
    ];
    let split_pairing_on_adjacent = [
        adjacent[0] * cosines[0] - adjacent[2] * sines[0],
        adjacent[1] * cosines[1] - adjacent[3] * sines[1],
        adjacent[2] * cosines[0] + adjacent[0] * sines[0],
        adjacent[3] * cosines[1] + adjacent[1] * sines[1],
    ];

    assert_eq!(rotated_split, [-3.0, 2.0, 1.0, 4.0]);
    assert_eq!(rotated_adjacent, [-3.0, 1.0, 2.0, 4.0]);
    assert_eq!(
        rotated_split,
        [
            rotated_adjacent[0],
            rotated_adjacent[2],
            rotated_adjacent[1],
            rotated_adjacent[3]
        ]
    );
    assert_ne!(split_pairing_on_adjacent, rotated_adjacent);
}
