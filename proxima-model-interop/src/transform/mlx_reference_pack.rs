//! Packing agreement of `export_affine` at (4, 64) with the vendored affine4
//! reference tensors under `tests/fixtures/mlx-reference`.
//!
//! The input is the vendored plain bf16 `k_proj` weight read as exact f32; the
//! reference is the same module's `weight`, `scales` and `biases` bytes. The
//! gemma rows are external (mlx-community's 4-bit conversion); the granite
//! rows are the crate's own output (the fixtures README), so they check
//! determinism of the bytes, not agreement with mlx-lm.

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::path::Path;

    use half::bf16;
    use proxima_tensor::DType;

    use crate::transform::{AffineExport, AffinePacked, AffineParams, export_affine};

    struct Case {
        model: &'static str,
        module: &'static str,
        shape: [usize; 2],
    }

    const GRANITE: Case = Case {
        model: "granite-3.1-1b-a400m-moe",
        module: "model.layers.0.self_attn.k_proj",
        shape: [512, 1024],
    };

    const E2B: Case = Case {
        model: "gemma4-e2b",
        module: "language_model.model.layers.0.self_attn.k_proj",
        shape: [256, 1536],
    };

    fn fixture(model: &str, flavor: &str, file: &str) -> Vec<u8> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/mlx-reference")
            .join(model)
            .join(flavor)
            .join(file);
        std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
    }

    fn plain_values(case: &Case) -> Vec<f32> {
        let file = format!("{}.weight.bf16.bin", case.module);
        let bytes = fixture(case.model, "plain", &file);
        let (pairs, _) = bytes.as_chunks::<2>();
        pairs
            .iter()
            .map(|pair| bf16::from_bits(u16::from_le_bytes(*pair)).to_f32())
            .collect()
    }

    fn packed(case: &Case) -> AffinePacked {
        let values = plain_values(case);
        let export = export_affine(
            &values,
            case.shape,
            AffineParams::default(),
            DType::BFLOAT16,
        );
        match export {
            Ok(AffineExport::Packed(packed)) => packed,
            other => panic!("{} k_proj did not pack at (4, 64): {other:?}", case.model),
        }
    }

    fn weight_bytes(packed: &AffinePacked) -> Vec<u8> {
        packed
            .weight
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect()
    }

    fn reference(case: &Case, suffix: &str) -> Vec<u8> {
        fixture(
            case.model,
            "affine4",
            &format!("{}.{suffix}.bin", case.module),
        )
    }

    fn first_disagreement(exported: &[u8], reference: &[u8]) -> Option<String> {
        if exported.len() != reference.len() {
            return Some(format!("lengths {} vs {}", exported.len(), reference.len()));
        }
        exported
            .iter()
            .zip(reference)
            .position(|(left, right)| left != right)
            .map(|at| format!("byte {at}: {:#04x} vs {:#04x}", exported[at], reference[at]))
    }

    fn assert_agrees(exported: &[u8], reference: &[u8]) {
        let disagreement = first_disagreement(exported, reference);
        assert_eq!(disagreement, None, "packed bytes differ from the reference");
    }

    #[test]
    fn mlx_reference_pack_granite_weight_equals_the_self_referential_reference() {
        assert_agrees(
            &weight_bytes(&packed(&GRANITE)),
            &reference(&GRANITE, "weight.u32"),
        );
    }

    #[test]
    fn mlx_reference_pack_granite_scales_equal_the_self_referential_reference() {
        assert_agrees(
            &packed(&GRANITE).scales,
            &reference(&GRANITE, "scales.bf16"),
        );
    }

    #[test]
    fn mlx_reference_pack_granite_biases_equal_the_self_referential_reference() {
        assert_agrees(
            &packed(&GRANITE).biases,
            &reference(&GRANITE, "biases.bf16"),
        );
    }

    #[test]
    fn mlx_reference_pack_e2b_weight_equals_the_mlx_community_reference() {
        assert_agrees(&weight_bytes(&packed(&E2B)), &reference(&E2B, "weight.u32"));
    }

    #[test]
    fn mlx_reference_pack_e2b_scales_equal_the_mlx_community_reference() {
        assert_agrees(&packed(&E2B).scales, &reference(&E2B, "scales.bf16"));
    }

    #[test]
    fn mlx_reference_pack_e2b_biases_equal_the_mlx_community_reference() {
        assert_agrees(&packed(&E2B).biases, &reference(&E2B, "biases.bf16"));
    }

    #[test]
    fn mlx_reference_pack_identity_control_reference_compared_to_itself_agrees() {
        let weight = reference(&E2B, "weight.u32");

        assert_eq!(first_disagreement(&weight, &weight), None);
    }

    #[test]
    fn mlx_reference_pack_degenerate_control_one_flipped_bit_is_located() {
        let mut altered = reference(&E2B, "scales.bf16");
        altered[17] ^= 0x01;

        let located = first_disagreement(&packed(&E2B).scales, &altered);

        assert!(located.is_some_and(|text| text.starts_with("byte 17:")));
    }
}
