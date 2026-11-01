//! PrismML `prism.hadamard.*` metadata: the rotation and sign leaves a Hadamard-folded program
//! reads.
//!
//! Spec: github.com/PrismML-Eng/llama.cpp, branch `prism`, `src/llama-model.cpp`
//! (`load_hparams_hadamard`, `load_tensors_hadamard`: rotation entry `(-1)^popcount(row & col) /
//! sqrt(block)`, one sign vector per input width), `src/llama-graph.cpp` (`build_hadamard_input`
//! rotates `D * x` by `H`; `build_embd_rows` restores a latent embedding row `z` as `D * (H * z)`).
//! The graph side is `proxima_tensor::spec::HadamardFold`.

use proxima_gguf::pipe::ParsedGguf;
use proxima_gguf::value::{MetadataArray, MetadataValue};

use crate::error::InteropError;

const PREFIX: &str = "prism.hadamard";
const ROTATION_LEAF: &str = "prism.hadamard.rotation";
const SIGNS_LEAF_PREFIX: &str = "prism.hadamard.signs.";
const LATENT_EMBEDDING: &str = "token_embd.weight";

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PrismHadamard {
    block: usize,
    gdn_v_grouped: bool,
    signs: Vec<(usize, Vec<f32>)>,
    inverse_weights: Vec<String>,
}

fn malformed(message: &str) -> InteropError {
    InteropError::UnsupportedServingConfig(format!("{PREFIX}: {message}"))
}

fn i32_array(parsed: &ParsedGguf, key: &str) -> Result<Vec<i32>, InteropError> {
    match parsed.metadata_value(key) {
        Some(MetadataValue::Array(MetadataArray::I32(values))) => Ok(values.clone()),
        _ => Err(InteropError::MissingMetadataKey { key: key.into() }),
    }
}

fn block_size(parsed: &ParsedGguf) -> Result<usize, InteropError> {
    let key = format!("{PREFIX}.block_size");
    match parsed.metadata_value(&key) {
        Some(MetadataValue::U32(value)) if value.is_power_of_two() => Ok(*value as usize),
        Some(_) => Err(malformed("block_size is not a power of two")),
        None => Err(InteropError::MissingMetadataKey { key }),
    }
}

fn split_signs(
    widths: &[i32],
    values: &[i32],
    block: usize,
) -> Result<Vec<(usize, Vec<f32>)>, InteropError> {
    let mut start = 0;
    let mut signs = Vec::with_capacity(widths.len());
    for width in widths {
        let width = usize::try_from(*width).map_err(|_| malformed("negative sign width"))?;
        let slice = values
            .get(start..start + width)
            .ok_or_else(|| malformed("sign_values is shorter than sign_widths"))?;
        if width == 0 || !width.is_multiple_of(block) || slice.iter().any(|sign| sign.abs() != 1) {
            return Err(malformed(
                "sign vector is not +/-1 over a whole number of blocks",
            ));
        }
        signs.push((width, slice.iter().map(|sign| *sign as f32).collect()));
        start += width;
    }
    Ok(signs)
}

fn inverse_weights(parsed: &ParsedGguf) -> Vec<String> {
    match parsed.metadata_value(&format!("{PREFIX}.inverse_weight_names")) {
        Some(MetadataValue::Array(MetadataArray::String(names))) => names.clone(),
        _ => Vec::new(),
    }
}

/// `None` when the checkpoint carries no `prism.hadamard.version`.
pub(crate) fn prism_hadamard_from_metadata(
    parsed: &ParsedGguf,
) -> Result<Option<PrismHadamard>, InteropError> {
    if parsed
        .metadata_value(&format!("{PREFIX}.version"))
        .is_none()
    {
        return Ok(None);
    }
    let block = block_size(parsed)?;
    let widths = i32_array(parsed, &format!("{PREFIX}.sign_widths"))?;
    let values = i32_array(parsed, &format!("{PREFIX}.sign_values"))?;
    let gdn_v_grouped = matches!(
        parsed.metadata_value(&format!("{PREFIX}.gdn_v_grouped")),
        Some(MetadataValue::Bool(true))
    );
    Ok(Some(PrismHadamard {
        block,
        gdn_v_grouped,
        signs: split_signs(&widths, &values, block)?,
        inverse_weights: inverse_weights(parsed),
    }))
}

impl PrismHadamard {
    pub(crate) const fn block(&self) -> usize {
        self.block
    }

    pub(crate) const fn gdn_v_grouped(&self) -> bool {
        self.gdn_v_grouped
    }

    pub(crate) fn latent_embedding(&self) -> bool {
        self.inverse_weights
            .iter()
            .any(|name| name == LATENT_EMBEDDING)
    }

    /// The data of a `prism.hadamard.*` program leaf, `None` for any other leaf name.
    pub(crate) fn leaf(&self, name: &str) -> Option<Vec<f32>> {
        if name == ROTATION_LEAF {
            return Some(self.rotation());
        }
        let width: usize = name.strip_prefix(SIGNS_LEAF_PREFIX)?.parse().ok()?;
        self.signs
            .iter()
            .find(|(candidate, _)| *candidate == width)
            .map(|(_, signs)| signs.clone())
    }

    fn rotation(&self) -> Vec<f32> {
        let scale = 1.0 / (self.block as f32).sqrt();
        (0..self.block)
            .flat_map(|row| {
                (0..self.block).map(move |column| {
                    if (row & column).count_ones().is_multiple_of(2) {
                        scale
                    } else {
                        -scale
                    }
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(block: usize) -> PrismHadamard {
        PrismHadamard {
            block,
            gdn_v_grouped: true,
            signs: vec![(
                block,
                (0..block)
                    .map(|index| if index % 3 == 0 { -1.0 } else { 1.0 })
                    .collect(),
            )],
            inverse_weights: vec![LATENT_EMBEDDING.to_owned()],
        }
    }

    #[test]
    fn rotation_is_a_symmetric_orthonormal_sylvester_matrix() {
        let hadamard = fixture(8);
        let rotation = hadamard.rotation();

        for row in 0..8 {
            for column in 0..8 {
                assert_eq!(rotation[row * 8 + column], rotation[column * 8 + row]);
                let dot: f32 = (0..8)
                    .map(|inner| rotation[row * 8 + inner] * rotation[column * 8 + inner])
                    .sum();
                let expected = if row == column { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-6, "({row}, {column}): {dot}");
            }
        }
    }

    #[test]
    fn leaves_are_named_by_the_program_declaration() {
        let hadamard = fixture(8);

        assert_eq!(
            hadamard.leaf(ROTATION_LEAF).map(|data| data.len()),
            Some(64)
        );
        assert_eq!(
            hadamard
                .leaf("prism.hadamard.signs.8")
                .map(|data| data.len()),
            Some(8)
        );
        assert_eq!(hadamard.leaf("prism.hadamard.signs.16"), None);
        assert_eq!(hadamard.leaf("blk.0.attn_q.weight"), None);
    }

    #[test]
    fn the_token_embedding_is_latent_only_when_the_metadata_names_it() {
        let mut hadamard = fixture(8);
        assert!(hadamard.latent_embedding());

        hadamard.inverse_weights.clear();
        assert!(!hadamard.latent_embedding());
    }
}
